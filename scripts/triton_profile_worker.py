"""Runs inside the Modal container (see profile_triton.py).

For every contest workload it records:

  * how many sparse indices are valid, and where the last one sits
  * latency of the CUDA submission and of every Triton configuration, timed
    the way flashinfer-bench times them (CUPTI, first kernel start to last
    kernel end)
  * which GPU kernels each call launched and how long each one ran
  * host time per call

A configuration is one of

  auto                                   kernel.run, the CUDA dispatcher
  path,G,num_warps,BLOCK_K,num_stages    kernel.launch, optionally ",graph"
  fused,name=value,...                   kernel.launch_fused keyword arguments
  fn=name                                kernel.<name>, an entry point
  extra                                  the --extra file's entry point

Results go to a JSON file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import traceback

import torch

DEFINITION = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
DEVICE = "cuda:0"
TOPK = 2048


def default_sweep() -> list:
    configs = ["auto"]
    # Per-token kernel: split factor, rows per tile, and software pipelining.
    for g in (1, 2, 4, 8, 16, 32, 64):
        for block_k in (16, 32, 64):
            if TOPK // g < block_k:
                continue
            for stages in (1, 2, 3):
                configs.append(f"token,{g},8,{block_k},{stages}")
    for g in (1, 16, 64):
        for block_k in (16, 32):
            configs.append(f"token,{g},4,{block_k},1")
    for g in (16, 64):
        for block_k in (16, 32):
            configs.append(f"token,{g},16,{block_k},1")
    # Per-head kernel.
    for g in (1, 4, 16):
        for block_k in (16, 32, 64):
            for warps in (4, 8):
                configs.append(f"head,{g},{warps},{block_k},1")
    configs.append("head,1,8,16,3")
    return configs


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_triton_fn(kernel, config: str, extra=None):
    if config == "auto":
        return kernel.run
    if config == "extra":
        return extra
    if config.startswith("fn="):
        return getattr(kernel, config[3:])
    if config.startswith("fused"):
        # "fused,g=8,block=64,thresh=64": keyword arguments of launch_fused.
        kwargs = {}
        for item in config.split(",")[1:]:
            name, _, value = item.partition("=")
            kwargs[name] = int(value)

        def fused(*args):
            kernel.launch_fused(*args, **kwargs)

        return fused
    path, g, warps, block_k, stages, *flags = config.split(",")

    def fn(*args):
        kernel.launch(
            *args,
            path=path, g=int(g), num_warps=int(warps),
            block_k=int(block_k), num_stages=int(stages),
            graph="graph" in flags,
        )

    return fn


def new_outputs(inputs):
    q_nope = inputs[0]
    out = torch.empty_like(q_nope)
    lse = torch.empty(q_nope.shape[:2], dtype=torch.float32, device=q_nope.device)
    return [out, lse]


def reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """fp32 PyTorch form of the definition's reference loop."""
    n_tokens, n_heads, dim = q_nope.shape
    ckv = ckv_cache.reshape(-1, dim)
    kpe = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    out = torch.zeros((n_tokens, n_heads, dim), dtype=torch.float32, device=q_nope.device)
    lse = torch.full((n_tokens, n_heads), float("-inf"), dtype=torch.float32, device=q_nope.device)
    for t in range(n_tokens):
        idx = sparse_indices[t]
        idx = idx[idx != -1].long()
        if idx.numel() == 0:
            continue
        kc = ckv[idx].float()
        kp = kpe[idx].float()
        score = (q_nope[t].float() @ kc.T + q_pe[t].float() @ kp.T) * float(sm_scale)
        lse[t] = torch.logsumexp(score, dim=-1) / math.log(2.0)
        out[t] = torch.softmax(score, dim=-1) @ kc
    return out, lse


def max_errors(outs, ref):
    out_err = (outs[0].float() - ref[0]).abs().max().item()
    finite = torch.isfinite(ref[1])
    lse_err = 0.0
    if finite.any():
        lse_err = (outs[1][finite] - ref[1][finite]).abs().max().item()
    if (outs[1][~finite] != ref[1][~finite]).any():
        lse_err = float("inf")
    return out_err, lse_err


_l2_flush = None


def l2_flush():
    """Evict the inputs from L2 before a timed call, as flashinfer-bench does."""
    global _l2_flush
    if _l2_flush is None:
        try:
            from flashinfer.utils import make_l2_flush

            size = torch.cuda.get_device_properties(DEVICE).L2_cache_size * 2
            _l2_flush = make_l2_flush((size // (1024 * 1024)) * 1024 * 1024, DEVICE)
        except Exception:
            buffer = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=DEVICE)
            _l2_flush = buffer.zero_
    _l2_flush()


def cupti_measure(fn, args, warmup: int, iters: int) -> dict:
    """flashinfer-bench's timer (flashinfer.testing.bench_gpu_time_with_cupti).

    One call's latency is the span from the first kernel's start to the last
    kernel's end. Host time before the first launch is not counted; host
    time between two launches is. Inputs are reused across calls. This copy
    also keeps each kernel's own run time.
    """
    import bisect
    from functools import partial

    from cupti import cupti

    kinds = (
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    )
    launches: list = []
    kernels: list = []

    def buffer_requested():
        return 8 * 1024 * 1024, 0

    def buffer_completed(launches, kernels, activities):
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                kernels.append((activity.name, activity.start, activity.end, activity.correlation_id))
            elif activity.kind == cupti.ActivityKind.MEMCPY:
                kernels.append(("MEMCPY", activity.start, activity.end, activity.correlation_id))
            elif activity.kind == cupti.ActivityKind.MEMSET:
                kernels.append(("MEMSET", activity.start, activity.end, activity.correlation_id))
            elif activity.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append((activity.start, activity.correlation_id))

    torch.cuda.synchronize()
    for _ in range(warmup):
        l2_flush()
        fn(*args)
    torch.cuda.synchronize()

    for kind in kinds:
        cupti.activity_enable(kind)
    cupti.activity_register_callbacks(buffer_requested, partial(buffer_completed, launches, kernels))
    stamps = []
    host = []
    try:
        for _ in range(iters):
            l2_flush()
            torch.cuda.synchronize()
            start = cupti.get_timestamp()
            fn(*args)
            end = cupti.get_timestamp()
            torch.cuda.synchronize()
            stamps.append((start, end))
            host.append((end - start) / 1e3)
        cupti.activity_flush_all(0)
    finally:
        for kind in kinds:
            cupti.activity_disable(kind)
        cupti.finalize()

    launches.sort()
    launch_starts = [launch[0] for launch in launches]
    by_correlation: dict = {}
    for kernel in kernels:
        by_correlation.setdefault(kernel[3], []).append(kernel)

    spans, busy = [], []
    per_kernel: dict = {}
    for start, end in stamps:
        lo = bisect.bisect_left(launch_starts, start)
        hi = bisect.bisect_right(launch_starts, end)
        ids = {launches[i][1] for i in range(lo, hi)}
        call = [k for cid in ids for k in by_correlation.get(cid, [])]
        if not call:
            raise ValueError("no kernel activity recorded for a timed call")
        spans.append((max(k[2] for k in call) - min(k[1] for k in call)) / 1e3)
        busy.append(sum(k[2] - k[1] for k in call) / 1e3)
        for name, k_start, k_end, _ in call:
            entry = per_kernel.setdefault(name[:110], [0, 0.0])
            entry[0] += 1
            entry[1] += (k_end - k_start) / 1e3
    return {
        "median_us": statistics.median(spans),
        "mean_us": statistics.mean(spans),
        "min_us": min(spans),
        # Sum of the kernels' own run times. span - busy is time the GPU sat
        # idle between two launches of the same call.
        "busy_us": statistics.median(busy),
        "host_us": statistics.median(host),
        "kernels": {
            name: {"calls": count / iters, "gpu_us": total / count}
            for name, (count, total) in per_kernel.items()
        },
    }


def index_stats(sparse_indices, n_kv: int) -> dict:
    valid = (sparse_indices >= 0) & (sparse_indices < n_kv)
    count = valid.sum(dim=1)
    pos = torch.arange(sparse_indices.shape[1], device=sparse_indices.device)
    last = torch.where(valid, pos[None, :], -1).max(dim=1).values + 1
    return {
        "valid_per_token": count.tolist(),
        "valid_avg": int(count.sum().item()) // max(sparse_indices.shape[0], 1),
        "last_valid_plus1": last.tolist(),
        "padding_is_tail": bool((count == last).all().item()),
    }


def stress(fn, calls: int, seed: int = 0, label: str = "") -> dict:
    """Check ``fn`` against the reference on synthetic calls.

    Every call has new contents, so a merge that read a partial left over
    from the call before would show up; the benchmark repeats one input and
    would hide it. Token counts go past the range of the workloads, and the
    padding sits either at the tail or anywhere in the row.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    pages, page_size, n_heads, dim, dim_pe, topk = 512, 64, 16, 512, 64, 2048
    n_kv = pages * page_size
    lengths = [0, 1, 2, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300, 511, 1024, 1500, 2047, 2048]
    worst = {"out": 0.0, "lse": 0.0}
    per_t: dict = {}
    for call in range(calls):
        n_tokens = (1, 2, 3, 6, 8, 9, 12, 18, 20)[call % 9]
        scatter = (call // 9) % 2 == 1
        ckv = (torch.randn((pages, page_size, dim), generator=gen, device=DEVICE) * 0.5).to(torch.bfloat16)
        kpe = (torch.randn((pages, page_size, dim_pe), generator=gen, device=DEVICE) * 0.5).to(torch.bfloat16)
        q_nope = torch.randn((n_tokens, n_heads, dim), generator=gen, device=DEVICE).to(torch.bfloat16)
        q_pe = torch.randn((n_tokens, n_heads, dim_pe), generator=gen, device=DEVICE).to(torch.bfloat16)
        indices = torch.full((n_tokens, topk), -1, dtype=torch.int32, device=DEVICE)
        for t in range(n_tokens):
            pick = int(torch.randint(len(lengths), (1,), generator=gen, device=DEVICE))
            count = lengths[pick]
            keys = torch.randint(n_kv, (count,), generator=gen, device=DEVICE, dtype=torch.int32)
            if scatter:
                where = torch.randperm(topk, generator=gen, device=DEVICE)[:count]
                indices[t, where] = keys
            else:
                indices[t, :count] = keys
        inputs = (q_nope, q_pe, ckv, kpe, indices, 0.1352337788608801)
        outs = new_outputs(inputs)
        outs[0].fill_(float("nan"))
        outs[1].fill_(float("nan"))
        # Printed before the call so a hang names the shape that caused it.
        print(f"stress {label} call {call} T={n_tokens} scatter={scatter}", flush=True)
        fn(*inputs, *outs)
        torch.cuda.synchronize()
        out_err, lse_err = max_errors(outs, reference(*inputs))
        if not math.isfinite(out_err):
            out_err = float("inf")
        worst["out"] = max(worst["out"], out_err)
        worst["lse"] = max(worst["lse"], lse_err)
        seen = per_t.setdefault(n_tokens, [0.0, 0.0])
        seen[0], seen[1] = max(seen[0], out_err), max(seen[1], lse_err)
    return {"calls": calls, "max_out_err": worst["out"], "max_lse_err": worst["lse"], "per_tokens": per_t}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--cuda-dir", default="")
    parser.add_argument("--trace-set", default="/data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--configs", default="")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--extra", default="", help="file.py::function, timed as config 'extra'")
    parser.add_argument("--stress", type=int, default=0, help="synthetic calls per configuration, checked only")
    args = parser.parse_args()

    import triton
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    report = {
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "workloads": {},
        "errors": {},
    }
    print(f"triton {triton.__version__}  torch {torch.__version__}  {report['gpu']}", flush=True)

    kernel = load_module(args.kernel, "triton_kernel")
    configs = [c for c in args.configs.split(";") if c] or default_sweep()
    extra = None
    if args.extra:
        extra_path, _, extra_name = args.extra.partition("::")
        extra = getattr(load_module(extra_path, "extra_kernel"), extra_name)
    triton_fns = {config: make_triton_fn(kernel, config, extra) for config in configs}

    cuda_fn = None
    if args.cuda_dir:
        sys.path.insert(0, args.cuda_dir)
        import solution as cuda_solution

        cuda_fn = cuda_solution.run

    trace_set = TraceSet.from_path(args.trace_set)
    definition = trace_set.definitions[DEFINITION]
    failed: set = set()
    # Both implementations cache the density decision on the index tensor's
    # address. Keeping every index tensor alive gives each workload its own
    # address, so a previous workload's decision is never reused here.
    keep_alive = []

    for trace in trace_set.workloads.get(DEFINITION, []):
        workload = trace.workload
        safe = load_safetensors(definition, workload, trace_set.root)
        inputs = gen_inputs(definition, workload, device=DEVICE, safe_tensors=safe)
        keep_alive.append(inputs[4])
        n_kv = inputs[2].shape[0] * inputs[2].shape[1]
        row = {"T": inputs[0].shape[0], **index_stats(inputs[4], n_kv), "latency": {}, "error": {}}
        row["plan"] = list(kernel.plan(row["T"], row["valid_avg"]))
        ref = reference(*inputs)

        def measure(name, fn):
            if name in failed:
                return
            try:
                outs = new_outputs(inputs)
                fn(*inputs, *outs)
                torch.cuda.synchronize()
                first = max_errors(outs, ref)
                row["latency"][name] = cupti_measure(fn, (*inputs, *outs), args.warmup, args.iters)
                # Checked again after timing: a graph replay writes the same
                # outputs through a different launch path.
                last = max_errors(outs, ref)
                row["error"][name] = (max(first[0], last[0]), max(first[1], last[1]))
            except Exception:
                failed.add(name)
                report["errors"][name] = traceback.format_exc()[-1200:]
                last = report["errors"][name].strip().splitlines()[-1]
                print(f"  {name} failed: {last[:200]}", flush=True)

        if cuda_fn is not None:
            measure("cuda", cuda_fn)
        for config, fn in triton_fns.items():
            measure(config, fn)

        report["workloads"][workload.uuid] = row
        lat = row["latency"]
        best = min(
            (c for c in lat if c not in ("cuda", "auto")),
            key=lambda c: lat[c]["median_us"],
            default=None,
        )
        summary = "  ".join(
            f"{name}={lat[name]['median_us']:.1f}us" for name in ("cuda", "auto") if name in lat
        )
        if best:
            summary += f"  best={best} {lat[best]['median_us']:.1f}us"
        print(
            f"{workload.uuid[:8]} T={row['T']} valid_avg={row['valid_avg']} "
            f"plan={row['plan']}  {summary}",
            flush=True,
        )
        with open(args.out, "w") as handle:
            json.dump(report, handle)
        del inputs, ref
        torch.cuda.empty_cache()

    with open(args.out, "w") as handle:
        json.dump(report, handle)

    # Last, because a kernel with a barrier can hang on a shape it was not
    # built for, and the timings above are already on disk by then.
    if args.stress:
        report["stress"] = {}
        for config, fn in triton_fns.items():
            try:
                report["stress"][config] = stress(fn, args.stress, label=config)
            except Exception:
                report["stress"][config] = {"failed": traceback.format_exc()[-1200:]}
            print(f"stress {config}: {report['stress'][config]}", flush=True)
            with open(args.out, "w") as handle:
                json.dump(report, handle)


if __name__ == "__main__":
    main()
