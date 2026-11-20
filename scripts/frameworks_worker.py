"""Time CUDA, fused Triton, TileLang, and ThunderKittens on the contest workloads.

Runs inside the Modal container (see profile_frameworks.py). The timer is the
same CUPTI span the Triton profile uses, so the numbers compare directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback

import torch


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--which", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tilelang", default="16x128")
    parser.add_argument("--trace-set", default="/data")
    args = parser.parse_args()

    tw = load("/root/triton_worker.py", "triton_worker")
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    wanted = set(args.which.split(",")) if args.which != "all" else {"cuda", "triton", "tilelang", "tk"}
    fns = {}
    if "cuda" in wanted:
        sys.path.insert(0, "/root/solution/python")
        import solution as cuda_solution

        fns["cuda"] = cuda_solution.run
    if "triton" in wanted:
        triton_kernel = load("/root/solution/triton/kernel.py", "triton_kernel")
        fns["triton-fused"] = triton_kernel.run_fused
    if "tilelang" in wanted:
        tilelang_kernel = load("/root/solution/tilelang/solution.py", "tilelang_kernel")
        for spec in args.tilelang.split(","):
            g, block = (int(x) for x in spec.lower().split("x"))
            name = f"tilelang-g{g}-b{block}"

            def make(g=g, block=block):
                def fn(*call):
                    tilelang_kernel.kernel(*call, g=g, block=block)

                return fn

            fns[name] = make()
    if "tk" in wanted:
        tk_kernel = load("/root/solution/tk/solution.py", "tk_kernel")
        fns["thunderkittens"] = tk_kernel.kernel

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "which": sorted(fns),
        "workloads": {},
        "errors": {},
    }
    try:
        import triton

        report["triton_version"] = triton.__version__
    except Exception:
        report["triton_version"] = ""
    try:
        import tilelang

        report["tilelang_version"] = tilelang.__version__
    except Exception:
        report["tilelang_version"] = ""

    trace_set = TraceSet.from_path(args.trace_set)
    definition = trace_set.definitions[tw.DEFINITION]
    failed: set = set()
    keep = []
    traces = list(trace_set.workloads.get(tw.DEFINITION, []))
    if args.limit:
        traces = traces[: args.limit]

    for trace in traces:
        workload = trace.workload
        safe = load_safetensors(definition, workload, trace_set.root)
        inputs = gen_inputs(definition, workload, device=tw.DEVICE, safe_tensors=safe)
        keep.append(inputs[4])
        n_kv = inputs[2].shape[0] * inputs[2].shape[1]
        row = {"T": inputs[0].shape[0], **tw.index_stats(inputs[4], n_kv), "latency": {}, "error": {}}
        ref = tw.reference(*inputs)

        def measure(name, fn):
            if name in failed:
                return
            try:
                outs = tw.new_outputs(inputs)
                fn(*inputs, *outs)
                torch.cuda.synchronize()
                first = tw.max_errors(outs, ref)
                row["latency"][name] = tw.cupti_measure(fn, (*inputs, *outs), args.warmup, args.iters)
                last = tw.max_errors(outs, ref)
                row["error"][name] = [max(first[0], last[0]), max(first[1], last[1])]
            except Exception:
                failed.add(name)
                report["errors"][name] = traceback.format_exc()[-2000:]
                print(f"  {name} failed: {report['errors'][name].strip().splitlines()[-1][:300]}", flush=True)

        for name, fn in fns.items():
            measure(name, fn)
        report["workloads"][workload.uuid] = row
        bits = "  ".join(
            f"{name}={row['latency'][name]['median_us']:.1f}us e{row['error'][name][0]:.1e}"
            for name in fns if name in row["latency"]
        )
        print(f"{workload.uuid[:8]} T={row['T']} {bits}", flush=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle)
        del inputs, ref
        torch.cuda.empty_cache()

    with open(args.out, "w") as handle:
        json.dump(report, handle)

    # A second call on the same storage, with the padding scattered, catches a
    # merge that reads a partial left behind by the call before it.
    for name in ("tilelang-g16-b128", "thunderkittens"):
        if name in fns and name not in failed:
            try:
                report.setdefault("stress", {})[name] = tw.stress(fns[name], 18, label=name)
            except Exception:
                report.setdefault("stress", {})[name] = traceback.format_exc()[-1500:]
                print(f"  stress {name} failed", flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle)


if __name__ == "__main__":
    main()
