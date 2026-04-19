"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Smoke (default): 17 random workloads, light timing.
Full suite: ``modal run scripts/run_modal.py --no-smoke`` — all workloads, saves JSON under artifacts/.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

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

# Full dataset: closer to flashinfer-bench defaults / contest-style timing (see EVALUATION.md).
FULL_BENCHMARK_CONFIG = BenchmarkConfig(
    warmup_runs=3,
    iterations=50,
    num_trials=5,
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


@app.function(
    image=image,
    gpu="B200:1",
    # Full definition sweeps (e.g. 128 workloads × heavy timing) can take hours.
    timeout=43200,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_benchmark(
    solution: Solution,
    smoke: bool = True,
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT,
    smoke_seed: int = SMOKE_RANDOM_SEED,
) -> dict:
    """Run benchmark on Modal B200 and return results.

    BenchmarkConfig is constructed here so we do not pickle it from the client:
    the Modal image may use a different flashinfer-bench than your local venv.
    """
    config = SMOKE_BENCHMARK_CONFIG if smoke else FULL_BENCHMARK_CONFIG

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


def print_results(results: dict) -> None:
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


def save_results_json(
    results: dict,
    solution: Solution,
    smoke: bool,
    config: BenchmarkConfig,
    output_path: Path,
) -> None:
    """Write results + metadata for baseline comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    def_names = list(results.keys())
    if len(def_names) != 1:
        raise ValueError(f"expected one definition in results, got {def_names}")
    def_name = def_names[0]
    entries = results[def_name]
    passed = sum(1 for e in entries.values() if e.get("status") == "PASSED")

    latencies = [
        e["latency_ms"] for e in entries.values() if e.get("latency_ms") is not None
    ]
    speedups = [
        e["speedup_factor"] for e in entries.values() if e.get("speedup_factor") is not None
    ]

    payload = {
        "meta": {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner": "modal",
            "solution_name": solution.name,
            "solution_hash": solution.hash(),
            "definition": solution.definition,
            "smoke": smoke,
            "benchmark_config": asdict(config),
            "workload_count": len(entries),
            "passed_count": passed,
            "failed_count": len(entries) - passed,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "mean_speedup_factor": sum(speedups) / len(speedups) if speedups else None,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")


@app.local_entrypoint()
def main(
    smoke: bool = True,
    save: str | None = None,
):
    """Pack solution and run benchmark on Modal.

    Parameters
    ----------
    smoke
        If True (default), run 17 random workloads with light timing.
        Pass ``--no-smoke`` to run every workload in the trace set and save a baseline JSON.
    save
        Optional path for JSON output. If ``--no-smoke`` and omitted, writes
        ``artifacts/modal_baseline_full_<UTC time>.json``.
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    bench_config = SMOKE_BENCHMARK_CONFIG if smoke else FULL_BENCHMARK_CONFIG

    print("\nRunning benchmark on Modal B200...")
    if smoke:
        print(
            f"  Smoke mode: {SMOKE_WORKLOAD_COUNT} workloads (seed={SMOKE_RANDOM_SEED}), "
            f"warmup={SMOKE_BENCHMARK_CONFIG.warmup_runs}, "
            f"iterations={SMOKE_BENCHMARK_CONFIG.iterations}, "
            f"trials={SMOKE_BENCHMARK_CONFIG.num_trials}"
        )
    else:
        print(
            f"  Full suite: all workloads, "
            f"warmup={FULL_BENCHMARK_CONFIG.warmup_runs}, "
            f"iterations={FULL_BENCHMARK_CONFIG.iterations}, "
            f"trials={FULL_BENCHMARK_CONFIG.num_trials}"
        )

    results = run_benchmark.remote(solution, smoke=smoke)

    if not results:
        print("No results returned!")
        return

    print_results(results)

    def_name = next(iter(results))
    n = len(results[def_name])
    passed = sum(1 for e in results[def_name].values() if e.get("status") == "PASSED")
    print(f"\nSummary: {passed}/{n} workloads PASSED")

    if not smoke:
        out = Path(save) if save else ARTIFACTS_DIR / (
            f"modal_baseline_full_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        save_results_json(results, solution, smoke=False, config=bench_config, output_path=out)
