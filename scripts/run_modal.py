"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

from __future__ import annotations

import random
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

# Match contest CI image; install FlashInfer / DeepGEMM from source (same as evaluation).
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
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

# Smoke: fewer timing iterations and a fixed random subset of workloads.
SMOKE_WORKLOAD_COUNT = 17
SMOKE_RANDOM_SEED = 42
SMOKE_BENCHMARK_CONFIG = BenchmarkConfig(
    warmup_runs=1,
    iterations=5,
    num_trials=2,
    use_isolated_runner=True,
    timeout_seconds=600,
    profile_baseline=True,
)


def _subsample_workloads(
    trace_set: TraceSet,
    definition_name: str,
    count: int,
    seed: int,
) -> TraceSet:
    workloads = trace_set.workloads.get(definition_name, [])
    if len(workloads) <= count:
        return trace_set
    rng = random.Random(seed)
    picked = rng.sample(workloads, count)
    new_workloads = dict(trace_set.workloads)
    new_workloads[definition_name] = picked
    return TraceSet(
        root=trace_set.root,
        definitions=trace_set.definitions,
        solutions=trace_set.solutions,
        workloads=new_workloads,
        traces=trace_set.traces,
    )


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig | None = None,
    smoke: bool = True,
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT,
    smoke_seed: int = SMOKE_RANDOM_SEED,
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = SMOKE_BENCHMARK_CONFIG if smoke else BenchmarkConfig(
            warmup_runs=3, iterations=100, num_trials=5, use_isolated_runner=True
        )

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    if smoke:
        trace_set = _subsample_workloads(
            trace_set, solution.definition, smoke_workloads, smoke_seed
        )

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


@app.local_entrypoint()
def main(smoke: bool = True):
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    if smoke:
        print(
            f"  Smoke mode: {SMOKE_WORKLOAD_COUNT} workloads (seed={SMOKE_RANDOM_SEED}), "
            f"warmup={SMOKE_BENCHMARK_CONFIG.warmup_runs}, "
            f"iterations={SMOKE_BENCHMARK_CONFIG.iterations}, "
            f"trials={SMOKE_BENCHMARK_CONFIG.num_trials}"
        )
    results = run_benchmark.remote(solution, smoke=smoke)

    if not results:
        print("No results returned!")
        return

    print_results(results)
