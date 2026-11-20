"""DSA sparse MLA, translated from the CUDA submission.

Same math, same dispatcher, same two kernels. Logits are stored in base 2
(multiplied by ``sm_scale * log2(e)``) and the online softmax uses ``exp2``,
matching ``mla_legacy.cuh`` and ``mla_cta.cuh``.

Dispatch, matching ``kernel_fn``:

  * H == 16 and the average number of valid indices per token is >= 210:
    one program per token (``mla_sparse_attn_kernel_cta_per_t_mma_v``).
    A KV tile is gathered once and contracted with all 16 heads. ``tl.dot``
    stands in for the WMMA score and the WMMA output mix.
  * otherwise:
    one program per (token, head) (``mla_sparse_attn_kernel``).

Split-K uses the same G tables as ``pick_split_G`` and
``pick_split_G_cta_per_t``. Partials are fp32 and merged by the same
rescale as ``mla_split_reduce_kernel``.

``run`` is that port: tiles of 16 rows, no load pipelining, eager launches.
``launch`` exposes the knobs the port fixes (kernel, split factor, tile
size, pipeline depth, CUDA-graph replay), and ``run_tuned`` is the setting
that measured fastest on B200.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_H = 16
_D = 512
_D_PE = 64
_TOPK = 2048
_BLOCK_K = 16
_BLOCK_D = 128

# sm_scale * LOG2E puts the logits in the base the CUDA kernels use.
_LOG2E = 1.4426950408889634

# kernel.cu: fall back to the per-head kernel below this average.
_DENSITY_MIN_AVG = 210
# pick_split_G: fill 148 SMs * 3 CTAs, and skip split-K below 64 CTAs.
_SPLIT_TARGET = 444
_SPLIT_MIN = 64
# pick_split_G_cta_per_t: one CTA per token, target a bit above 148 SMs.
_CPT_TARGET = 222

_scratch: dict = {}
_density: dict = {}
_graphs: dict = {}
_graph_seen: set = set()


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def _rows(cache: torch.Tensor) -> torch.Tensor:
    rows = cache.reshape(-1, cache.shape[-1])
    return rows if rows.is_contiguous() else rows.contiguous()


def pick_split_g(base_ctas: int) -> int:
    """``pick_split_G`` in kernel.cu. ``base_ctas`` is T * H."""
    if base_ctas < _SPLIT_MIN or base_ctas >= _SPLIT_TARGET:
        return 1
    if base_ctas * 2 >= _SPLIT_TARGET:
        return 2
    if base_ctas * 4 >= _SPLIT_TARGET:
        return 4
    return 16


def pick_split_g_cta(n_tokens: int) -> int:
    """``pick_split_G_cta_per_t`` in kernel.cu. One CTA per token per slice."""
    if n_tokens >= _CPT_TARGET:
        return 1
    if n_tokens * 2 >= _CPT_TARGET:
        return 2
    if n_tokens * 4 >= _CPT_TARGET:
        return 4
    if n_tokens * 8 >= _CPT_TARGET:
        return 8
    return 16


def _density_avg(indices: torch.Tensor, n_kv: int) -> int:
    """Valid indices per token. Cached on the pointer, like ``DensityCache``."""
    n_tokens = indices.shape[0]
    key = (indices.data_ptr(), n_tokens, n_kv)
    cached = _density.get(key)
    if cached is not None:
        return cached
    valid = (indices >= 0) & (indices < n_kv)
    count = int(valid.sum().item())
    avg = count // max(n_tokens, 1)
    if len(_density) >= 32:
        _density.pop(next(iter(_density)))
    _density[key] = avg
    return avg


def _workspace(n_tokens: int, g: int, device: torch.device):
    """fp32 partials [T, H, G, D], m and l [T, H, G], contiguous for each shape."""
    key = (n_tokens, g, device.type, device.index)
    buf = _scratch.get(key)
    if buf is None:
        buf = (
            torch.empty((n_tokens, _H, g, _D), dtype=torch.float32, device=device),
            torch.empty((n_tokens, _H, g), dtype=torch.float32, device=device),
            torch.empty((n_tokens, _H, g), dtype=torch.float32, device=device),
        )
        _scratch[key] = buf
    return buf


@triton.jit
def _load_rows(base, rows, col0, stride_row, stride_col, valid, WIDTH: tl.constexpr):
    cols = col0 + tl.arange(0, WIDTH)
    ptr = base + rows[:, None] * stride_row + cols[None, :] * stride_col
    return tl.load(ptr, mask=valid[:, None], other=0)


@triton.jit
def _load_heads(base, token, heads, col0, stride_t, stride_h, stride_d, WIDTH: tl.constexpr):
    cols = col0 + tl.arange(0, WIDTH)
    ptr = base + token * stride_t + heads[:, None] * stride_h + cols[None, :] * stride_d
    return tl.load(ptr)


@triton.jit
def _per_head(
    q_nope, q_pe, ckv, kpe, indices,
    out, lse, partial_o, partial_m, partial_l,
    sm_log2e, n_kv,
    stride_qt, stride_qh, stride_qd,
    stride_pt, stride_ph, stride_pd,
    stride_ck, stride_cd, stride_kp, stride_kd,
    stride_ot, stride_oh, stride_od,
    stride_lt, stride_lh,
    H: tl.constexpr, D: tl.constexpr, D_PE: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    G: tl.constexpr, FINAL: tl.constexpr,
):
    """``mla_sparse_attn_kernel`` / ``mla_split_partial_kernel``.

    One program owns one head and one slice of the top-k axis. The dot is
    the same fp32 accumulation the warp reduction produces.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    group = tl.program_id(2)
    cols = tl.arange(0, BLOCK_D)

    q_base = q_nope + token * stride_qt + head * stride_qh
    q0 = tl.load(q_base + cols * stride_qd).to(tl.float32)
    q1 = tl.load(q_base + (BLOCK_D + cols) * stride_qd).to(tl.float32)
    q2 = tl.load(q_base + (2 * BLOCK_D + cols) * stride_qd).to(tl.float32)
    q3 = tl.load(q_base + (3 * BLOCK_D + cols) * stride_qd).to(tl.float32)
    pe = tl.arange(0, D_PE)
    qp = tl.load(q_pe + token * stride_pt + head * stride_ph + pe * stride_pd).to(tl.float32)

    m = tl.full((), float("-inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    a0 = tl.zeros((BLOCK_D,), tl.float32)
    a1 = tl.zeros((BLOCK_D,), tl.float32)
    a2 = tl.zeros((BLOCK_D,), tl.float32)
    a3 = tl.zeros((BLOCK_D,), tl.float32)

    k_per: tl.constexpr = TOPK // G
    k_lo = group * k_per
    # Prescan, same as the CUDA kernel: stop at the last valid index so
    # the padding tail is not gathered.
    pos = tl.arange(0, k_per)
    scanned = tl.load(indices + token * TOPK + k_lo + pos).to(tl.int32)
    scanned_ok = (scanned >= 0) & (scanned < n_kv)
    last = tl.max(tl.where(scanned_ok, pos, -1))
    n_tiles = (last + BLOCK_K) // BLOCK_K

    for tile in tl.range(0, n_tiles):
        offs = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        idx = tl.load(indices + token * TOPK + k_lo + offs).to(tl.int32)
        valid = (idx >= 0) & (idx < n_kv)
        rows = tl.where(valid, idx, 0).to(tl.int64)

        k0v = _load_rows(ckv, rows, 0, stride_ck, stride_cd, valid, BLOCK_D).to(tl.float32)
        k1v = _load_rows(ckv, rows, BLOCK_D, stride_ck, stride_cd, valid, BLOCK_D).to(tl.float32)
        k2v = _load_rows(ckv, rows, 2 * BLOCK_D, stride_ck, stride_cd, valid, BLOCK_D).to(tl.float32)
        k3v = _load_rows(ckv, rows, 3 * BLOCK_D, stride_ck, stride_cd, valid, BLOCK_D).to(tl.float32)
        kp = _load_rows(kpe, rows, 0, stride_kp, stride_kd, valid, D_PE).to(tl.float32)

        score = tl.sum(k0v * q0[None, :], axis=1)
        score += tl.sum(k1v * q1[None, :], axis=1)
        score += tl.sum(k2v * q2[None, :], axis=1)
        score += tl.sum(k3v * q3[None, :], axis=1)
        score += tl.sum(kp * qp[None, :], axis=1)
        score = score * sm_log2e

        # Online softmax, base 2. A fully masked tile is a no-op; the first
        # valid tile has alpha 0 and replaces the zero accumulator.
        masked = tl.where(valid, score, float("-inf"))
        tile_max = tl.max(masked, axis=0)
        empty = tile_max == float("-inf")
        m_new = tl.maximum(m, tile_max)
        m_safe = tl.where(m == float("-inf"), 0.0, m)
        m_new_safe = tl.where(empty, 0.0, m_new)
        alpha = tl.math.exp2(m_safe - m_new_safe)
        alpha = tl.where(empty, 1.0, tl.where(m == float("-inf"), 0.0, alpha))
        prob = tl.math.exp2(tl.where(valid, score, 0.0) - m_new_safe)
        prob = tl.where(valid & ~empty, prob, 0.0)
        l = tl.where(empty, l, l * alpha + tl.sum(prob, axis=0))
        m = tl.where(empty, m, m_new)
        a0 = a0 * alpha + tl.sum(prob[:, None] * k0v, axis=0)
        a1 = a1 * alpha + tl.sum(prob[:, None] * k1v, axis=0)
        a2 = a2 * alpha + tl.sum(prob[:, None] * k2v, axis=0)
        a3 = a3 * alpha + tl.sum(prob[:, None] * k3v, axis=0)

    if FINAL:
        inv = tl.where(l > 0, 1.0 / l, 0.0)
        base = out + token * stride_ot + head * stride_oh
        tl.store(base + cols * stride_od, (a0 * inv).to(tl.bfloat16))
        tl.store(base + (BLOCK_D + cols) * stride_od, (a1 * inv).to(tl.bfloat16))
        tl.store(base + (2 * BLOCK_D + cols) * stride_od, (a2 * inv).to(tl.bfloat16))
        tl.store(base + (3 * BLOCK_D + cols) * stride_od, (a3 * inv).to(tl.bfloat16))
        lse_val = tl.where(l > 0, m + tl.math.log2(tl.where(l > 0, l, 1.0)), float("-inf"))
        tl.store(lse + token * stride_lt + head * stride_lh, lse_val)
    else:
        slot = ((token * H + head) * G + group) * D
        tl.store(partial_o + slot + cols, a0)
        tl.store(partial_o + slot + BLOCK_D + cols, a1)
        tl.store(partial_o + slot + 2 * BLOCK_D + cols, a2)
        tl.store(partial_o + slot + 3 * BLOCK_D + cols, a3)
        ml = (token * H + head) * G + group
        tl.store(partial_m + ml, m)
        tl.store(partial_l + ml, l)


@triton.jit
def _per_token(
    q_nope, q_pe, ckv, kpe, indices,
    out, lse, partial_o, partial_m, partial_l,
    sm_log2e, n_kv,
    stride_qt, stride_qh, stride_qd,
    stride_pt, stride_ph, stride_pd,
    stride_ck, stride_cd, stride_kp, stride_kd,
    stride_ot, stride_oh, stride_od,
    stride_lt, stride_lh,
    H: tl.constexpr, D: tl.constexpr, D_PE: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    G: tl.constexpr, FINAL: tl.constexpr,
):
    """``mla_sparse_attn_kernel_cta_per_t_mma_v``.

    One program owns every head of one token and one top-k slice. The KV
    tile is loaded once. ``tl.dot`` is the score GEMM and the output mix;
    probabilities are bf16, the accumulator stays fp32.
    """
    token = tl.program_id(0)
    group = tl.program_id(1)
    heads = tl.arange(0, H)

    q0 = _load_heads(q_nope, token, heads, 0, stride_qt, stride_qh, stride_qd, BLOCK_D)
    q1 = _load_heads(q_nope, token, heads, BLOCK_D, stride_qt, stride_qh, stride_qd, BLOCK_D)
    q2 = _load_heads(q_nope, token, heads, 2 * BLOCK_D, stride_qt, stride_qh, stride_qd, BLOCK_D)
    q3 = _load_heads(q_nope, token, heads, 3 * BLOCK_D, stride_qt, stride_qh, stride_qd, BLOCK_D)
    qp = _load_heads(q_pe, token, heads, 0, stride_pt, stride_ph, stride_pd, D_PE)

    m = tl.full((H,), float("-inf"), tl.float32)
    l = tl.zeros((H,), tl.float32)
    a0 = tl.zeros((H, BLOCK_D), tl.float32)
    a1 = tl.zeros((H, BLOCK_D), tl.float32)
    a2 = tl.zeros((H, BLOCK_D), tl.float32)
    a3 = tl.zeros((H, BLOCK_D), tl.float32)

    k_per: tl.constexpr = TOPK // G
    k_lo = group * k_per
    pos = tl.arange(0, k_per)
    gathered = tl.load(indices + token * TOPK + k_lo + pos).to(tl.int32)
    gathered_valid = (gathered >= 0) & (gathered < n_kv)
    last = tl.max(tl.where(gathered_valid, pos, -1))
    n_tiles = (last + BLOCK_K) // BLOCK_K

    for tile in tl.range(0, n_tiles):
        offs = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        idx = tl.load(indices + token * TOPK + k_lo + offs).to(tl.int32)
        valid = (idx >= 0) & (idx < n_kv) & (offs < k_per)
        rows = tl.where(valid, idx, 0).to(tl.int64)

        k0v = _load_rows(ckv, rows, 0, stride_ck, stride_cd, valid, BLOCK_D)
        k1v = _load_rows(ckv, rows, BLOCK_D, stride_ck, stride_cd, valid, BLOCK_D)
        k2v = _load_rows(ckv, rows, 2 * BLOCK_D, stride_ck, stride_cd, valid, BLOCK_D)
        k3v = _load_rows(ckv, rows, 3 * BLOCK_D, stride_ck, stride_cd, valid, BLOCK_D)
        kp = _load_rows(kpe, rows, 0, stride_kp, stride_kd, valid, D_PE)

        score = tl.dot(q0, tl.trans(k0v), out_dtype=tl.float32)
        score += tl.dot(q1, tl.trans(k1v), out_dtype=tl.float32)
        score += tl.dot(q2, tl.trans(k2v), out_dtype=tl.float32)
        score += tl.dot(q3, tl.trans(k3v), out_dtype=tl.float32)
        score += tl.dot(qp, tl.trans(kp), out_dtype=tl.float32)
        score = score * sm_log2e

        masked = tl.where(valid[None, :], score, float("-inf"))
        tile_max = tl.max(masked, axis=1)
        empty = tile_max == float("-inf")
        m_new = tl.maximum(m, tile_max)
        m_safe = tl.where(m == float("-inf"), 0.0, m)
        m_new_safe = tl.where(empty, 0.0, m_new)
        alpha = tl.math.exp2(m_safe - m_new_safe)
        alpha = tl.where(empty, 1.0, tl.where(m == float("-inf"), 0.0, alpha))
        prob = tl.math.exp2(tl.where(valid[None, :], score, 0.0) - m_new_safe[:, None])
        prob = tl.where(valid[None, :] & (~empty)[:, None], prob, 0.0)
        l = tl.where(empty, l, l * alpha + tl.sum(prob, axis=1))
        m = tl.where(empty, m, m_new)
        prob_bf = prob.to(tl.bfloat16)
        a0 = a0 * alpha[:, None] + tl.dot(prob_bf, k0v, out_dtype=tl.float32)
        a1 = a1 * alpha[:, None] + tl.dot(prob_bf, k1v, out_dtype=tl.float32)
        a2 = a2 * alpha[:, None] + tl.dot(prob_bf, k2v, out_dtype=tl.float32)
        a3 = a3 * alpha[:, None] + tl.dot(prob_bf, k3v, out_dtype=tl.float32)

    d_ids = tl.arange(0, BLOCK_D)
    if FINAL:
        inv = tl.where(l > 0, 1.0 / l, 0.0)[:, None]
        base = out + token * stride_ot + heads[:, None] * stride_oh
        tl.store(base + d_ids[None, :] * stride_od, (a0 * inv).to(tl.bfloat16))
        tl.store(base + (BLOCK_D + d_ids)[None, :] * stride_od, (a1 * inv).to(tl.bfloat16))
        tl.store(base + (2 * BLOCK_D + d_ids)[None, :] * stride_od, (a2 * inv).to(tl.bfloat16))
        tl.store(base + (3 * BLOCK_D + d_ids)[None, :] * stride_od, (a3 * inv).to(tl.bfloat16))
        lse_val = tl.where(l > 0, m + tl.math.log2(tl.where(l > 0, l, 1.0)), float("-inf"))
        tl.store(lse + token * stride_lt + heads * stride_lh, lse_val)
    else:
        base = partial_o + ((token * H + heads) * G + group)[:, None] * D
        tl.store(base + d_ids[None, :], a0)
        tl.store(base + BLOCK_D + d_ids[None, :], a1)
        tl.store(base + 2 * BLOCK_D + d_ids[None, :], a2)
        tl.store(base + 3 * BLOCK_D + d_ids[None, :], a3)
        ml = (token * H + heads) * G + group
        tl.store(partial_m + ml, m)
        tl.store(partial_l + ml, l)


@triton.jit
def _reduce(
    partial_o, partial_m, partial_l, out, lse,
    stride_ot, stride_oh, stride_od,
    stride_lt, stride_lh,
    H: tl.constexpr, D: tl.constexpr, G: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """``mla_split_reduce_kernel``. Merge G partials with ``exp2`` rescale."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_D)

    groups = tl.arange(0, G)
    ml = (token * H + head) * G
    m_g = tl.load(partial_m + ml + groups)
    m_final = tl.max(m_g, axis=0)
    empty = m_final == float("-inf")
    m_final_safe = tl.where(empty, 0.0, m_final)

    a0 = tl.zeros((BLOCK_D,), tl.float32)
    a1 = tl.zeros((BLOCK_D,), tl.float32)
    a2 = tl.zeros((BLOCK_D,), tl.float32)
    a3 = tl.zeros((BLOCK_D,), tl.float32)
    l_final = tl.zeros((), tl.float32)
    for g in tl.static_range(G):
        mg = tl.load(partial_m + ml + g)
        lg = tl.load(partial_l + ml + g)
        base = partial_o + ((token * H + head) * G + g) * D
        safe_mg = tl.where(mg == float("-inf"), m_final_safe, mg)
        alpha = tl.where(mg == float("-inf"), 0.0, tl.math.exp2(safe_mg - m_final_safe))
        l_final += lg * alpha
        a0 += tl.load(base + cols) * alpha
        a1 += tl.load(base + BLOCK_D + cols) * alpha
        a2 += tl.load(base + 2 * BLOCK_D + cols) * alpha
        a3 += tl.load(base + 3 * BLOCK_D + cols) * alpha

    inv = tl.where(l_final > 0, 1.0 / l_final, 0.0)
    out_base = out + token * stride_ot + head * stride_oh
    tl.store(out_base + cols * stride_od, (a0 * inv).to(tl.bfloat16))
    tl.store(out_base + (BLOCK_D + cols) * stride_od, (a1 * inv).to(tl.bfloat16))
    tl.store(out_base + (2 * BLOCK_D + cols) * stride_od, (a2 * inv).to(tl.bfloat16))
    tl.store(out_base + (3 * BLOCK_D + cols) * stride_od, (a3 * inv).to(tl.bfloat16))
    lse_val = tl.where(
        l_final > 0,
        m_final + tl.math.log2(tl.where(l_final > 0, l_final, 1.0)),
        float("-inf"),
    )
    tl.store(lse + token * stride_lt + head * stride_lh, lse_val)


def _check(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, out, lse):
    if q_nope.shape[1] != _H or q_nope.shape[2] != _D:
        raise ValueError(f"q_nope must be [T, {_H}, {_D}], got {tuple(q_nope.shape)}")
    if q_pe.shape[2] != _D_PE or sparse_indices.shape[1] != _TOPK:
        raise ValueError("unexpected q_pe or sparse_indices shape")
    if out.shape != q_nope.shape or lse.shape[:2] != q_nope.shape[:2]:
        raise ValueError("out/lse shape does not match q_nope")
    if ckv_cache.shape[-1] != _D or kpe_cache.shape[-1] != _D_PE:
        raise ValueError("cache trailing dim mismatch")


def plan(n_tokens: int, density_avg: int):
    """The choice ``kernel_fn`` makes: which kernel, and the split-K factor."""
    if density_avg >= _DENSITY_MIN_AVG:
        return "token", pick_split_g_cta(n_tokens)
    return "head", pick_split_g(n_tokens * _H)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse):
    """Destination-passing entry. Same choice as ``kernel_fn``."""
    launch(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse)


def run_tuned(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse):
    """The configuration the B200 sweep found fastest (scripts/profile_triton.py).

    Always the per-token kernel, with 64-row tiles and a three-stage load
    pipeline. One or two tokens run as a single kernel. More tokens split
    the top-k axis 16 ways and replay both launches from a CUDA graph. The
    choice depends on the shape only, so nothing is cached about the data.
    """
    g = 1 if q_nope.shape[0] <= 2 else 16
    launch(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse,
        path="token", g=g, num_warps=8, block_k=64, num_stages=3, graph=g > 1,
    )


def _replay_graph(key, emit) -> bool:
    """Replay ``emit``'s kernels as one CUDA graph, like the CUDA submission.

    Split-K is two launches. From Python the second one reaches the GPU
    after the first has already finished, so the GPU idles in between; a
    graph submits both at once. A key is captured the second time it is
    seen, so one-off calls stay eager. Returns False when the caller still
    has to run ``emit`` itself.
    """
    graph = _graphs.get(key)
    if graph is not None:
        graph.replay()
        return True
    if key not in _graph_seen:
        if len(_graph_seen) >= 256:
            _graph_seen.clear()
        _graph_seen.add(key)
        return False
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        emit()
    if len(_graphs) >= 32:
        _graphs.pop(next(iter(_graphs)))
    _graphs[key] = graph
    graph.replay()
    return True


def launch(
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse,
    path=None, g=None, num_warps=8, block_k=_BLOCK_K, num_stages=1, graph=False,
):
    """Run one configuration.

    With ``path`` and ``g`` left as None this is the CUDA dispatcher. The
    profiling script passes them to time each kernel on every workload.
    ``graph`` replays the launches from a CUDA graph keyed on the tensors'
    addresses.
    """
    _check(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, out, lse)
    ckv = _rows(ckv_cache)
    kpe = _rows(kpe_cache)
    indices = sparse_indices if sparse_indices.is_contiguous() else sparse_indices.contiguous()
    n_tokens = q_nope.shape[0]
    n_kv = ckv.shape[0]
    sm_log2e = _as_float(sm_scale) * _LOG2E

    if path is None or g is None:
        auto_path, auto_g = plan(n_tokens, _density_avg(indices, n_kv))
        if path is None:
            path, g = auto_path, auto_g
        else:
            g = pick_split_g_cta(n_tokens) if path == "token" else pick_split_g(n_tokens * _H)
    cta = path == "token"
    final = g == 1

    partial_o = partial_m = partial_l = out
    if not final:
        partial_o, partial_m, partial_l = _workspace(n_tokens, g, q_nope.device)

    args = (
        q_nope, q_pe, ckv, kpe, indices,
        out, lse, partial_o, partial_m, partial_l,
        sm_log2e, n_kv,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv.stride(0), ckv.stride(1), kpe.stride(0), kpe.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
    )
    kernel = _per_token if cta else _per_head
    grid = (n_tokens, g) if cta else (n_tokens, _H, g)

    def emit():
        kernel[grid](
            *args,
            H=_H, D=_D, D_PE=_D_PE, TOPK=_TOPK,
            BLOCK_K=block_k, BLOCK_D=_BLOCK_D,
            G=g, FINAL=final,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if not final:
            _reduce[(n_tokens, _H)](
                partial_o, partial_m, partial_l, out, lse,
                out.stride(0), out.stride(1), out.stride(2),
                lse.stride(0), lse.stride(1),
                H=_H, D=_D, G=g, BLOCK_D=_BLOCK_D,
                num_warps=4,
                num_stages=1,
            )

    # A graph holds raw addresses, so it is only used when the kernels read
    # the caller's own storage (no temporary contiguous copy).
    borrowed = (
        ckv.data_ptr() == ckv_cache.data_ptr()
        and kpe.data_ptr() == kpe_cache.data_ptr()
        and indices is sparse_indices
    )
    if graph and borrowed:
        key = (
            q_nope.data_ptr(), q_pe.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
            indices.data_ptr(), out.data_ptr(), lse.data_ptr(),
            n_tokens, n_kv, sm_log2e, args[12:],
            path, g, num_warps, block_k, num_stages,
        )
        if _replay_graph(key, emit):
            return
    emit()
