"""DSA sparse MLA, TileLang implementation.

Same math as solution/triton/kernel.py ``run_fused``. One program owns every
G-th key of one token and a running softmax over the whole 512-wide value.
A second kernel merges the G partials. Both launches are replayed from one
CUDA graph, so the host does not re-issue them on the timed calls.

config.toml still points at the CUDA submission. This file is a side-by-side
port. The gather is element-wise: a 16-byte load comes back as a vector this
frontend cannot index, which is the same constraint as the FP8 indexer port.
"""

import collections

import tilelang
import tilelang.language as TL
import torch

H = 16
D = 512
D_PE = 64
TOPK = 2048
G = 16
BLOCK = 128
MAX_GRAPHS = 32

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

_built: dict = {}
_scratch: dict = {}
_scale: dict = {}
_graphs: collections.OrderedDict = collections.OrderedDict()
_streams: dict = {}


def _build(g: int, block: int):
    """Compile the partial kernel and the merge kernel for one tile shape."""
    ntiles = TOPK // g // block

    @tilelang.jit(target="cuda", pass_configs=_PASS)
    def partial(q_nope, q_pe, ckv, kpe, indices, part_m, part_l, part_acc, scale):
        n_tokens = TL.dynamic("n_tokens")
        n_kv = TL.dynamic("n_kv")
        q_nope: TL.Tensor((n_tokens, H, D), TL.bfloat16)
        q_pe: TL.Tensor((n_tokens, H, D_PE), TL.bfloat16)
        ckv: TL.Tensor((n_kv, D), TL.bfloat16)
        kpe: TL.Tensor((n_kv, D_PE), TL.bfloat16)
        indices: TL.Tensor((n_tokens, TOPK), TL.int32)
        part_m: TL.Tensor((n_tokens, g, H), TL.float32)
        part_l: TL.Tensor((n_tokens, g, H), TL.float32)
        part_acc: TL.Tensor((n_tokens, g, H, D), TL.float32)
        scale: TL.Tensor((1,), TL.float32)

        with TL.Kernel(n_tokens, g, threads=128) as (token, slot):
            q_s = TL.alloc_shared((H, D), TL.bfloat16)
            qp_s = TL.alloc_shared((H, D_PE), TL.bfloat16)
            k_s = TL.alloc_shared((block, D), TL.bfloat16)
            kp_s = TL.alloc_shared((block, D_PE), TL.bfloat16)
            p_s = TL.alloc_shared((H, block), TL.bfloat16)
            acc_s = TL.alloc_shared((H, D), TL.float32)
            alpha_s = TL.alloc_shared((H,), TL.float32)
            valid_s = TL.alloc_shared((block,), TL.int32)
            idx_s = TL.alloc_shared((block,), TL.int32)
            cnt = TL.alloc_shared((1,), TL.int32)
            score = TL.alloc_fragment((H, block), TL.float32)
            pe = TL.alloc_fragment((H, block), TL.float32)
            mix = TL.alloc_fragment((H, D), TL.float32)
            m_i = TL.alloc_fragment((H,), TL.float32)
            l_i = TL.alloc_fragment((H,), TL.float32)
            m_row = TL.alloc_fragment((H,), TL.float32)
            row_sum = TL.alloc_fragment((H,), TL.float32)
            sm = scale[0]

            for h, d in TL.Parallel(H, D):
                q_s[h, d] = q_nope[token, h, d]
                acc_s[h, d] = 0
            for h, d in TL.Parallel(H, D_PE):
                qp_s[h, d] = q_pe[token, h, d]
            for h in TL.Parallel(H):
                m_i[h] = -1.0e30
                l_i[h] = 0

            for tile in TL.serial(ntiles):
                for i in TL.Parallel(block):
                    pos = slot + (tile * block + i) * g
                    got = indices[token, pos]
                    ok = (got >= 0) and (got < n_kv)
                    idx_s[i] = TL.if_then_else(ok, got, 0)
                    valid_s[i] = TL.if_then_else(ok, 1, 0)
                if TL.get_thread_binding() == 0:
                    cnt[0] = 0
                TL.sync_threads()
                for i in TL.Parallel(block):
                    if valid_s[i] == 1:
                        TL.atomic_add(cnt[0], 1)
                for i, d in TL.Parallel(block, D):
                    k_s[i, d] = TL.if_then_else(valid_s[i] == 1, ckv[idx_s[i], d], TL.cast(0, "bfloat16"))
                for i, d in TL.Parallel(block, D_PE):
                    kp_s[i, d] = TL.if_then_else(valid_s[i] == 1, kpe[idx_s[i], d], TL.cast(0, "bfloat16"))
                TL.sync_threads()

                TL.clear(score)
                TL.gemm(q_s, k_s, score, transpose_B=True, policy=TL.GemmWarpPolicy.FullRow)
                TL.clear(pe)
                TL.gemm(qp_s, kp_s, pe, transpose_B=True, policy=TL.GemmWarpPolicy.FullRow)
                for h, i in TL.Parallel(H, block):
                    total = (score[h, i] + pe[h, i]) * sm
                    score[h, i] = TL.if_then_else(valid_s[i] == 1, total, -1.0e30)

                # An empty tile must not move the running softmax. cnt is the
                # same value for every thread of the program.
                if cnt[0] > 0:
                    TL.reduce_max(score, m_row, dim=1)
                    for h in TL.Parallel(H):
                        m_row[h] = TL.max(m_row[h], m_i[h])
                        alpha_s[h] = TL.exp2(m_i[h] - m_row[h])
                        l_i[h] = l_i[h] * alpha_s[h]
                        m_i[h] = m_row[h]
                    for h, i in TL.Parallel(H, block):
                        score[h, i] = TL.exp2(score[h, i] - m_i[h])
                    TL.reduce_sum(score, row_sum, dim=1)
                    for h in TL.Parallel(H):
                        l_i[h] = l_i[h] + row_sum[h]
                    TL.sync_threads()
                    for h, i in TL.Parallel(H, block):
                        p_s[h, i] = TL.cast(score[h, i], "bfloat16")
                    # One gemm covers all 512 columns. k_s still holds the keys.
                    TL.clear(mix)
                    TL.gemm(p_s, k_s, mix, policy=TL.GemmWarpPolicy.FullRow)
                    for h, d in TL.Parallel(H, D):
                        acc_s[h, d] = acc_s[h, d] * alpha_s[h] + mix[h, d]

            for h in TL.Parallel(H):
                part_m[token, slot, h] = m_i[h]
                part_l[token, slot, h] = l_i[h]
            for h, d in TL.Parallel(H, D):
                part_acc[token, slot, h, d] = acc_s[h, d]

    @tilelang.jit(target="cuda", pass_configs=_PASS)
    def merge(part_m, part_l, part_acc, out, lse):
        n_tokens = TL.dynamic("n_tokens")
        part_m: TL.Tensor((n_tokens, g, H), TL.float32)
        part_l: TL.Tensor((n_tokens, g, H), TL.float32)
        part_acc: TL.Tensor((n_tokens, g, H, D), TL.float32)
        out: TL.Tensor((n_tokens, H, D), TL.bfloat16)
        lse: TL.Tensor((n_tokens, H), TL.float32)

        width = D // g
        with TL.Kernel(n_tokens, g, threads=128) as (token, slot):
            # A scalar assigned inside TL.serial is scoped to that loop.
            m_top = TL.alloc_shared((H,), TL.float32)
            l_sum = TL.alloc_shared((H,), TL.float32)
            merged = TL.alloc_shared((H, width), TL.float32)
            for h in TL.Parallel(H):
                m_top[h] = -1.0e30
                l_sum[h] = 0
            for h, c in TL.Parallel(H, width):
                merged[h, c] = 0
            for src in TL.serial(g):
                for h in TL.Parallel(H):
                    m_top[h] = TL.max(m_top[h], part_m[token, src, h])
            for h in TL.Parallel(H):
                # -inf minus -inf is NaN. An empty token has no finite partial.
                m_top[h] = TL.if_then_else(m_top[h] < -1.0e20, 0, m_top[h])
            for src in TL.serial(g):
                for h in TL.Parallel(H):
                    weight = TL.exp2(part_m[token, src, h] - m_top[h])
                    l_sum[h] = l_sum[h] + part_l[token, src, h] * weight
                for h, c in TL.Parallel(H, width):
                    col = slot * width + c
                    weight = TL.exp2(part_m[token, src, h] - m_top[h])
                    merged[h, c] = merged[h, c] + part_acc[token, src, h, col] * weight
            for h, c in TL.Parallel(H, width):
                col = slot * width + c
                out[token, h, col] = TL.cast(
                    TL.if_then_else(l_sum[h] > 0, merged[h, c] / l_sum[h], 0), "bfloat16"
                )
            if slot == 0:
                for h in TL.Parallel(H):
                    lse[token, h] = TL.if_then_else(
                        l_sum[h] > 0, m_top[h] + TL.log2(TL.max(l_sum[h], 1.0e-30)), float("-inf")
                    )

    return partial, merge


def _kernels(g: int, block: int):
    key = (g, block)
    pair = _built.get(key)
    if pair is None:
        pair = _build(g, block)
        _built[key] = pair
    return pair


def _workspace(n_tokens: int, g: int, device: torch.device):
    key = (n_tokens, g, device.type, device.index)
    buf = _scratch.get(key)
    if buf is None:
        buf = (
            torch.empty((n_tokens, g, H), dtype=torch.float32, device=device),
            torch.empty((n_tokens, g, H), dtype=torch.float32, device=device),
            torch.empty((n_tokens, g, H, D), dtype=torch.float32, device=device),
        )
        _scratch[key] = buf
    return buf


def _log2e_scale(sm_scale, device: torch.device):
    """sm_scale * log2(e), cached per device. The kernel multiplies logits by it."""
    value = float(sm_scale) * 1.4426950408889634
    buf = _scale.get(device.index)
    if buf is None or abs(float(buf[0]) - value) > 1e-12:
        buf = torch.tensor([value], dtype=torch.float32, device=device)
        _scale[device.index] = buf
    return buf


def _replay(key, device, launch):
    """Run ``launch`` through a cached CUDA graph. The first call stays eager."""
    if key not in _graphs:
        launch()
        try:
            stream = _streams.get(device.index)
            if stream is None:
                stream = _streams[device.index] = torch.cuda.Stream(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launch()
        except Exception as exc:
            print(f"tilelang graph capture failed: {type(exc).__name__}: {exc}", flush=True)
            graph = None
        _graphs[key] = graph
        if len(_graphs) > MAX_GRAPHS:
            _graphs.popitem(last=False)
        return
    graph = _graphs[key]
    _graphs.move_to_end(key)
    if graph is None:
        launch()
        return
    stream = _streams[device.index]
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    graph.replay()
    current.wait_stream(stream)


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse,
           g: int = G, block: int = BLOCK, graph: bool = True):
    """Destination-passing entry. ``g`` and ``block`` are the tuning knobs."""
    if TOPK % (g * block) != 0 or D % g != 0:
        raise ValueError("g * block must divide 2048 and g must divide 512")
    ckv = ckv_cache.reshape(-1, D)
    kpe = kpe_cache.reshape(-1, D_PE)
    if ckv.stride(1) != 1:
        ckv = ckv.contiguous()
    if kpe.stride(1) != 1:
        kpe = kpe.contiguous()
    indices = sparse_indices if sparse_indices.is_contiguous() else sparse_indices.contiguous()
    n_tokens = q_nope.shape[0]
    partial, merge = _kernels(g, block)
    part_m, part_l, part_acc = _workspace(n_tokens, g, q_nope.device)
    scale = _log2e_scale(sm_scale, q_nope.device)

    def launch():
        partial(q_nope, q_pe, ckv, kpe, indices, part_m, part_l, part_acc, scale)
        merge(part_m, part_l, part_acc, out, lse)

    if not graph:
        launch()
        return
    key = (
        q_nope.data_ptr(), q_pe.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        indices.data_ptr(), out.data_ptr(), lse.data_ptr(),
        n_tokens, ckv.shape[0], g, block, float(scale[0]),
    )
    _replay(key, q_nope.device, launch)
