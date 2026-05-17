"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Pin to the same image digest documented in EVALUATION.md (evaluator environment).
# CUDA 13.2 + nvcc; required to compile the TVM-FFI CUDA entry at runtime.
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("wheel", "setuptools")
    # cupti-python is in the evaluation environment (see EVALUATION.md).
    .pip_install("cupti-python")
    .run_commands(
        "git clone --recursive --depth 1 https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM",
        "pip install --no-build-isolation /tmp/DeepGEMM",
        "git clone --recursive --depth 1 https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer",
        "pip install --no-build-isolation /tmp/flashinfer",
        "git clone --depth 1 https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench",
        "pip install /tmp/flashinfer-bench",
    )
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    disable_density_cache: bool = False,
    debug_density_cache: bool = False,
    shape_dispatch: bool = False,
    shape_dispatch_t_min: int = 6,
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    # [DEBUG] Forward density-cache control flags into env vars before the
    # JIT extension is built/loaded. Local-only.
    import os
    if disable_density_cache:
        os.environ["MLA_DENSITY_CACHE_DISABLE"] = "1"
    if debug_density_cache:
        os.environ["MLA_DENSITY_CACHE_DEBUG"] = "1"
    if shape_dispatch:
        os.environ["MLA_SHAPE_DISPATCH"] = "1"
        os.environ["MLA_SHAPE_DISPATCH_T_MIN"] = str(int(shape_dispatch_t_min))
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            # [DEBUG] Forward the worker's captured stdio log back to the
            # local entrypoint so we can inspect kernel-side fprintf output
            # (e.g. [density_cache] hit/miss diagnostics). Local-only.
            entry["log"] = trace.evaluation.log or ""
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

    # [DEBUG] Surface kernel-side density-cache diagnostics from each
    # worker's captured stdio log. Local-only — remove with the matching
    # kernel.cu instrumentation before submitting.
    print("\n=== [density_cache] diagnostics (per workload) ===")
    for def_name, traces in results.items():
        for workload_uuid, result in traces.items():
            log = result.get("log", "") or ""
            dbg_lines = [ln for ln in log.splitlines() if "density_cache" in ln]
            if not dbg_lines:
                continue
            hits = sum(1 for ln in dbg_lines if "HIT " in ln or "HIT\t" in ln)
            miss = sum(1 for ln in dbg_lines if "MISS" in ln)
            print(f"\n[{workload_uuid[:8]}] hits={hits} misses={miss} "
                  f"(showing first 6 + last 4 of {len(dbg_lines)} lines):")
            head = dbg_lines[:6]
            tail = dbg_lines[-4:] if len(dbg_lines) > 10 else []
            for ln in head:
                print(f"    {ln}")
            if tail:
                print(f"    ... ({len(dbg_lines) - len(head) - len(tail)} lines elided) ...")
                for ln in tail:
                    print(f"    {ln}")


@app.local_entrypoint()
def main(
    disable_density_cache: bool = False,
    debug_density_cache: bool = False,
    shape_dispatch: bool = False,
    shape_dispatch_t_min: int = 6,
):
    """Pack solution and run benchmark on Modal.

    Flags (local-only, for cache-impact A/B):
        --disable-density-cache: force every call to recompute the density
            count (no host-side cache reuse). Use to measure perf without
            the cache.
        --debug-density-cache: emit per-call HIT/MISS lines to the worker
            log (default off — adds I/O noise to the timing path).
        --shape-dispatch: skip the density-count scan entirely; pick path
            from T alone (T >= shape_dispatch_t_min => CPT). Sync-free.
        --shape-dispatch-t-min N: threshold for shape dispatch (default 6).
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    if shape_dispatch:
        mode = f"SHAPE_DISPATCH(T_min={shape_dispatch_t_min})"
    elif disable_density_cache:
        mode = "CACHE_DISABLED (recompute every call)"
    else:
        mode = "CACHE_ENABLED"
    print(f"\nRunning benchmark on Modal B200... (mode={mode}, "
          f"debug={'ON' if debug_density_cache else 'OFF'})")
    results = run_benchmark.remote(
        solution,
        disable_density_cache=disable_density_cache,
        debug_density_cache=debug_density_cache,
        shape_dispatch=shape_dispatch,
        shape_dispatch_t_min=shape_dispatch_t_min,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)
