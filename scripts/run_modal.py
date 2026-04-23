"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Usage:
    # Full suite (contest-style), all workloads:
    modal run scripts/run_modal.py

    # Smoke (fast correctness check): 10 random workloads, warmup=1 / iters=1 / trials=1:
    modal run scripts/run_modal.py --smoke

    # Smoke with custom sample size:
    modal run scripts/run_modal.py --smoke --smoke-workloads 5
"""

import random
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

SMOKE_WORKLOAD_COUNT_DEFAULT = 10
SMOKE_RANDOM_SEED = 42

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Pin to the same image digest documented in EVALUATION.md (evaluator environment).
# CUDA 13.2 + nvcc; required to compile the TVM-FFI CUDA entry at runtime.
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .apt_install("git")
    # sm_100a enables Blackwell-only features (tcgen05, TMA tensor ops, ...).
    # Without the trailing "a", nvcc targets the portable sm_100 subset and
    # the tcgen05.mma / tcgen05.ld / tcgen05.alloc intrinsics fail to
    # assemble. The BuildSpec's extra_cuda_cflags are not honored by the
    # tvm_ffi.cpp.build() caller path, so we inject via TVM_FFI_CUDA_ARCH_LIST.
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TVM_FFI_CUDA_ARCH_LIST": "10.0a",
        }
    )
    .pip_install("wheel", "setuptools")
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
    smoke: bool = False,
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT_DEFAULT,
    smoke_seed: int = SMOKE_RANDOM_SEED,
    tma: bool = False,
) -> dict:
    """Run benchmark on Modal B200 and return results.

    If ``smoke`` is True, randomly sample ``smoke_workloads`` workloads (seeded)
    and use warmup=1 / iterations=1 / num_trials=1 for a fast correctness pass.

    If ``tma`` is True, sets MLA_SPARSE_ATTN_TMA=1 before the solution's
    kernel.cu is imported, routing the G==1 path through the TMA-gather4
    variant (Phase 5 step 2).
    """
    import os as _os
    if tma:
        _os.environ["MLA_SPARSE_ATTN_TMA"] = "1"
        print("  TMA enabled (MLA_SPARSE_ATTN_TMA=1)")

    if config is None:
        if smoke:
            config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)
        else:
            config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if smoke and workloads and len(workloads) > smoke_workloads:
        rng = random.Random(smoke_seed)
        workloads = rng.sample(workloads, smoke_workloads)
        print(
            f"  Smoke mode: {len(workloads)} workloads (seed={smoke_seed}), "
            f"warmup={config.warmup_runs}, iterations={config.iterations}, "
            f"num_trials={config.num_trials}"
        )

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
            # Capture any potentially useful debug fields
            for attr in ("error_message", "error", "message", "detail", "log", "stderr"):
                val = getattr(trace.evaluation, attr, None)
                if val:
                    entry["error_message"] = str(val)
                    break
            try:
                entry["evaluation_repr"] = repr(trace.evaluation)[:1500]
            except Exception:
                pass
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
            if result.get("error_message"):
                msg = result["error_message"]
                print("    error:", msg[:3000])
            elif result.get("status") not in (None, "PASSED", "PASS", "passed"):
                er = result.get("evaluation_repr")
                if er:
                    print("    eval:", er)


@app.local_entrypoint()
def main(
    smoke: bool = False,
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT_DEFAULT,
    smoke_seed: int = SMOKE_RANDOM_SEED,
    tma: bool = False,
):
    """Pack solution and run benchmark on Modal.

    Args:
        smoke: If True, run on a random subset with minimal timing (fast correctness check).
        smoke_workloads: How many random workloads to run in smoke mode (default 10).
        smoke_seed: Random seed for the smoke workload sample (default 42).
        tma: If True, sets MLA_SPARSE_ATTN_TMA=1 on the Modal worker before the
            solution's kernel.cu is imported (Phase 5 step 2 gather4 path).
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    if smoke:
        print(f"\nRunning SMOKE benchmark on Modal B200 "
              f"({smoke_workloads} random workloads, seed={smoke_seed})"
              + (" [TMA=1]" if tma else "") + "...")
    else:
        print("\nRunning FULL benchmark on Modal B200"
              + (" [TMA=1]" if tma else "") + "...")
    results = run_benchmark.remote(
        solution,
        smoke=smoke,
        smoke_workloads=smoke_workloads,
        smoke_seed=smoke_seed,
        tma=tma,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)
