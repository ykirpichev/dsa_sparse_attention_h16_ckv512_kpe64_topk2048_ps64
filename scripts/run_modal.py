"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Smoke (default): **2** random workloads, minimal timing, **num_trials=1** (no repeated
correctness trials). Override count with ``--smoke-workloads`` (use 1–3 while debugging).

Full suite: ``modal run scripts/run_modal.py --no-smoke`` — all workloads, saves JSON under artifacts/.

Alternate config for packing (e.g. batched PyTorch baseline on the CUDA branch)::

    FIB_CONFIG_TOML=config.batched_pytorch.toml modal run scripts/run_modal.py::main --no-smoke

Torch-compile A/B (same container)::

    modal run scripts/run_modal.py::compare_torch_compile --smoke-workloads 5
    modal run scripts/run_modal.py::compare_torch_compile --no-smoke
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict
from typing import Literal
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.data import EvaluationStatus

# Cap embedded log size in JSON; full text still printed for first failure on stdout.
_MAX_LOG_JSON_CHARS = 50_000

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

# Smoke / debug: broad subsample by default so we catch correctness / perf surprises
# that wouldn't show up in 2 random workloads (see auto-campaign lesson in Stage 1).
# num_trials=1 so correctness is checked once per workload (no benchmark retries).
SMOKE_WORKLOAD_COUNT_DEFAULT = 10
SMOKE_RANDOM_SEED = 42
SMOKE_BENCHMARK_CONFIG = BenchmarkConfig(
    warmup_runs=1,
    iterations=5,
    num_trials=1,
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


def _apply_attention_tail_env(mode: Literal["inherit", "compile", "eager"]) -> None:
    """Control ``torch.compile`` for the PyTorch attention tail (see solution ``kernel.py``)."""
    if mode == "inherit":
        return
    if mode == "compile":
        os.environ.pop("DSA_DISABLE_TORCH_COMPILE", None)
    else:
        os.environ["DSA_DISABLE_TORCH_COMPILE"] = "1"


def _run_benchmark_core(
    solution: Solution,
    smoke: bool,
    smoke_workloads: int,
    smoke_seed: int,
) -> dict:
    """Run flashinfer-bench (caller sets ``DSA_DISABLE_TORCH_COMPILE`` for A/B)."""
    config = SMOKE_BENCHMARK_CONFIG if smoke else FULL_BENCHMARK_CONFIG

    if smoke:
        smoke_workloads = max(1, int(smoke_workloads))

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
            ev = trace.evaluation
            entry = {
                "status": ev.status.value,
                "solution": trace.solution,
            }
            if ev.performance:
                entry["latency_ms"] = ev.performance.latency_ms
                entry["reference_latency_ms"] = ev.performance.reference_latency_ms
                entry["speedup_factor"] = ev.performance.speedup_factor
            if ev.correctness:
                entry["max_abs_error"] = ev.correctness.max_absolute_error
                entry["max_rel_error"] = ev.correctness.max_relative_error
            # Isolated runner embeds worker stderr + traceback here (see flashinfer_bench make_eval).
            if ev.status != EvaluationStatus.PASSED and ev.log:
                entry["log"] = ev.log
            results[definition.name][trace.workload.uuid] = entry

    return results


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
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT_DEFAULT,
    smoke_seed: int = SMOKE_RANDOM_SEED,
    attention_tail: Literal["inherit", "compile", "eager"] = "inherit",
) -> dict:
    """Run benchmark on Modal B200 and return results.

    BenchmarkConfig is constructed here so we do not pickle it from the client:
    the Modal image may use a different flashinfer-bench than your local venv.

    attention_tail
        ``compile`` / ``eager`` sets ``DSA_DISABLE_TORCH_COMPILE`` for isolated workers
        (A/B timing of ``torch.compile`` vs eager PyTorch tail). ``inherit`` leaves env unchanged.
    """
    _apply_attention_tail_env(attention_tail)
    return _run_benchmark_core(solution, smoke, smoke_workloads, smoke_seed)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=43200,
    volumes={TRACE_SET_PATH: trace_volume},
)
def compare_torch_compile_benchmark(
    solution: Solution,
    smoke: bool = True,
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT_DEFAULT,
    smoke_seed: int = SMOKE_RANDOM_SEED,
) -> dict:
    """Run the same smoke/full suite twice: ``torch.compile`` tail vs eager PyTorch tail (same container)."""
    _apply_attention_tail_env("compile")
    r_compile = _run_benchmark_core(solution, smoke, smoke_workloads, smoke_seed)
    _apply_attention_tail_env("eager")
    r_eager = _run_benchmark_core(solution, smoke, smoke_workloads, smoke_seed)
    return {"compile": r_compile, "eager": r_eager}


def _mean_latency_ms(results: dict) -> float | None:
    if not results:
        return None
    def_name = next(iter(results))
    lat = [e["latency_ms"] for e in results[def_name].values() if e.get("latency_ms") is not None]
    return sum(lat) / len(lat) if lat else None


def _mean_speedup(results: dict) -> float | None:
    if not results:
        return None
    def_name = next(iter(results))
    sp = [e["speedup_factor"] for e in results[def_name].values() if e.get("speedup_factor") is not None]
    return sum(sp) / len(sp) if sp else None


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


def print_first_failure_log(results: dict, max_chars: int = 24_000) -> None:
    """Print embedded evaluation log for the first non-PASSED workload (stderr + traceback)."""
    for def_name, traces in results.items():
        for workload_uuid, result in traces.items():
            if result.get("status") == "PASSED":
                continue
            log = result.get("log")
            if not log:
                print(
                    f"\n--- No embedded log for {workload_uuid[:8]}... "
                    f"(status={result.get('status')}) ---\n"
                )
                return
            print(f"\n{'=' * 72}")
            print(
                f"First failure log (workload {workload_uuid}, status={result.get('status')})"
            )
            print(f"{'=' * 72}\n")
            text = log if len(log) <= max_chars else log[:max_chars] + "\n\n[log truncated for stdout]\n"
            print(text)
            return
    print("\n(No failed workloads with logs to show.)")


def _truncate_logs_for_json(results: dict) -> dict:
    """Avoid multi‑MB JSON if every workload embeds the same traceback."""
    out = {}
    for def_name, entries in results.items():
        out[def_name] = {}
        for uid, entry in entries.items():
            e = dict(entry)
            if "log" in e and len(e["log"]) > _MAX_LOG_JSON_CHARS:
                e["log"] = e["log"][:_MAX_LOG_JSON_CHARS] + "\n\n[log truncated for JSON]\n"
            out[def_name][uid] = e
    return out


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
        "results": _truncate_logs_for_json(results),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")


def save_compare_torch_compile_json(
    merged: dict,
    solution: Solution,
    smoke: bool,
    output_path: Path,
) -> None:
    """Write compile vs eager benchmark results (``compile`` / ``eager`` definition blocks)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = SMOKE_BENCHMARK_CONFIG if smoke else FULL_BENCHMARK_CONFIG
    rc = merged["compile"]
    re = merged["eager"]
    def_name = next(iter(rc))
    n = len(rc[def_name])
    nc = sum(1 for e in rc[def_name].values() if e.get("status") == "PASSED")
    ne = sum(1 for e in re[def_name].values() if e.get("status") == "PASSED")

    payload = {
        "meta": {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner": "modal",
            "kind": "compare_torch_compile",
            "solution_name": solution.name,
            "solution_hash": solution.hash(),
            "definition": solution.definition,
            "smoke": smoke,
            "benchmark_config": asdict(cfg),
            "workload_count": n,
            "passed_compile": nc,
            "passed_eager": ne,
            "mean_latency_ms_compile": _mean_latency_ms(rc),
            "mean_latency_ms_eager": _mean_latency_ms(re),
            "mean_speedup_compile": _mean_speedup(rc),
            "mean_speedup_eager": _mean_speedup(re),
        },
        "compile": _truncate_logs_for_json(rc),
        "eager": _truncate_logs_for_json(re),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved compare results to {output_path}")


@app.local_entrypoint()
def main(
    smoke: bool = True,
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT_DEFAULT,
    save: str | None = None,
):
    """Pack solution and run benchmark on Modal.

    Parameters
    ----------
    smoke
        If True (default), subsample workloads (see ``smoke_workloads``) with light timing
        and ``num_trials=1``. Pass ``--no-smoke`` to run every workload and optionally save JSON.
    smoke_workloads
        When ``smoke`` is True, number of random workloads to run (default **2**). Use **1–3**
        for fast debugging; larger values are allowed for broader smoke coverage.
    save
        Optional path for JSON output. If ``--no-smoke`` and omitted, writes
        ``artifacts/modal_baseline_full_<UTC time>.json``.
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    cfg = os.environ.get("FIB_CONFIG_TOML")
    solution_path = pack_solution(config_path=Path(cfg) if cfg else None)

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    bench_config = SMOKE_BENCHMARK_CONFIG if smoke else FULL_BENCHMARK_CONFIG

    print("\nRunning benchmark on Modal B200...")
    if smoke:
        print(
            f"  Smoke mode: {smoke_workloads} workloads (seed={SMOKE_RANDOM_SEED}), "
            f"warmup={SMOKE_BENCHMARK_CONFIG.warmup_runs}, "
            f"iterations={SMOKE_BENCHMARK_CONFIG.iterations}, "
            f"num_trials={SMOKE_BENCHMARK_CONFIG.num_trials} (single correctness pass per workload)"
        )
    else:
        print(
            f"  Full suite: all workloads, "
            f"warmup={FULL_BENCHMARK_CONFIG.warmup_runs}, "
            f"iterations={FULL_BENCHMARK_CONFIG.iterations}, "
            f"trials={FULL_BENCHMARK_CONFIG.num_trials}"
        )

    results = run_benchmark.remote(
        solution,
        smoke=smoke,
        smoke_workloads=smoke_workloads,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)

    def_name = next(iter(results))
    n = len(results[def_name])
    passed = sum(1 for e in results[def_name].values() if e.get("status") == "PASSED")
    print(f"\nSummary: {passed}/{n} workloads PASSED")

    if passed < n:
        print_first_failure_log(results)

    if not smoke:
        out = Path(save) if save else ARTIFACTS_DIR / (
            f"modal_baseline_full_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        save_results_json(results, solution, smoke=False, config=bench_config, output_path=out)


@app.local_entrypoint()
def compare_torch_compile(
    smoke: bool = True,
    smoke_workloads: int = 5,
    save: str | None = None,
):
    """A/B: ``torch.compile`` attention tail vs eager PyTorch (same B200 container, same workloads).

    Usage::

        modal run scripts/run_modal.py::compare_torch_compile --smoke-workloads 5
        modal run scripts/run_modal.py::compare_torch_compile --no-smoke
        modal run scripts/run_modal.py::compare_torch_compile --no-smoke --save artifacts/compare_full.json
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    cfg = os.environ.get("FIB_CONFIG_TOML")
    solution_path = pack_solution(config_path=Path(cfg) if cfg else None)
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    sw = max(1, int(smoke_workloads))
    if smoke:
        print(
            f"\nComparing attention tail: torch.compile vs eager "
            f"(smoke, {sw} workloads, seed={SMOKE_RANDOM_SEED})"
        )
    else:
        print(
            "\nComparing attention tail: torch.compile vs eager "
            f"(full suite: all workloads, warmup={FULL_BENCHMARK_CONFIG.warmup_runs}, "
            f"iterations={FULL_BENCHMARK_CONFIG.iterations}, trials={FULL_BENCHMARK_CONFIG.num_trials})"
        )

    merged = compare_torch_compile_benchmark.remote(solution, smoke, sw, SMOKE_RANDOM_SEED)

    rc = merged["compile"]
    re = merged["eager"]

    print("\n========== torch.compile tail ==========")
    print_results(rc)
    print("\n========== eager PyTorch tail (DSA_DISABLE_TORCH_COMPILE) ==========")
    print_results(re)

    def_name = next(iter(rc))
    nc = sum(1 for e in rc[def_name].values() if e.get("status") == "PASSED")
    ne = sum(1 for e in re[def_name].values() if e.get("status") == "PASSED")
    n = len(rc[def_name])

    mc = _mean_latency_ms(rc)
    me = _mean_latency_ms(re)
    sc = _mean_speedup(rc)
    se = _mean_speedup(re)

    print("\n=== Mean over workloads ===")
    if mc is not None:
        print(f"  torch.compile:  mean latency {mc:.4f} ms   mean speedup vs ref {sc:.3f}x   PASSED {nc}/{n}")
    if me is not None:
        print(f"  eager:          mean latency {me:.4f} ms   mean speedup vs ref {se:.3f}x   PASSED {ne}/{n}")
    if mc is not None and me is not None and me > 0:
        print(f"  latency ratio (compile / eager): {mc / me:.4f}  (<1 means compile faster on average)")

    if nc < n:
        print("\n(compile path) First failure:")
        print_first_failure_log(rc)
    if ne < n:
        print("\n(eager path) First failure:")
        print_first_failure_log(re)

    if save:
        out = Path(save)
    elif not smoke:
        out = ARTIFACTS_DIR / (
            f"modal_compare_torch_compile_full_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
    else:
        out = None

    if out is not None:
        save_compare_torch_compile_json(merged, solution, smoke, out)
