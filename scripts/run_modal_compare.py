"""Compare multiple solutions side-by-side on Modal B200.

Benchmarks up to N pre-packed Solution JSON files against the same workloads
in a single Modal session, and saves a JSON artifact with per-workload
latencies (our_ms, ref_pytorch_ms, speedup, status) for each solution.

This is what lets us build a paired "PyTorch ref / FlashInfer baseline /
v4 Stage 3c / v5_fused" comparison table: all four latencies are measured
on the same GPU, same image, same BenchmarkConfig, same 23 workloads.

Usage (local):
    # Pack solutions first (any combination; paths resolved locally):
    python scripts/run_modal_compare.py \
        --solution /home/.../flashinfer_wrapper_5af199.json \
        --solution ./artifacts/v4_stage3c.solution.json \
        --solution ./artifacts/v5_fused.solution.json \
        --output ./artifacts/compare_four.json
"""

import argparse
import datetime as _dt
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench-compare")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .apt_install("git")
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


@app.function(
    image=image, gpu="B200:1", timeout=7200, volumes={TRACE_SET_PATH: trace_volume}
)
def run_compare(
    solutions_json: list,
    smoke: bool = False,
    smoke_workloads: int = 10,
    smoke_seed: int = 42,
    repeats: int = 1,
) -> dict:
    """Benchmark multiple solutions against the same PyTorch reference.

    Parameters
    ----------
    solutions_json : list of str
        JSON-encoded Solution strings (model_dump_json output).
    repeats : int
        Number of independent full-benchmark invocations within this Modal
        session. Each produces its own per-workload latency measurements so
        we can quantify cross-run noise. Builds are cached across repeats.
    """
    if smoke:
        cfg = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)
    else:
        cfg = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    solutions = [Solution.model_validate_json(j) for j in solutions_json]
    if not solutions:
        raise ValueError("No solutions provided")
    defname = solutions[0].definition
    for s in solutions:
        if s.definition != defname:
            raise ValueError(
                f"All solutions must target the same definition; got "
                f"'{defname}' and '{s.definition}'"
            )

    if defname not in trace_set.definitions:
        raise ValueError(f"Definition '{defname}' not found in trace set")

    definition = trace_set.definitions[defname]
    workloads = trace_set.workloads.get(defname, [])
    if smoke and workloads and len(workloads) > smoke_workloads:
        rng = random.Random(smoke_seed)
        workloads = rng.sample(workloads, smoke_workloads)
        print(
            f"Smoke mode: {len(workloads)} workloads (seed={smoke_seed}), "
            f"warmup={cfg.warmup_runs}, iters={cfg.iterations}, "
            f"trials={cfg.num_trials}"
        )

    if not workloads:
        raise ValueError(f"No workloads for definition '{defname}'")

    print(
        f"Comparing {len(solutions)} solutions on {len(workloads)} workloads, "
        f"repeats={repeats}:"
    )
    for s in solutions:
        print(f"  - {s.name}  ({s.spec.language})")

    def _collect_one_run() -> dict:
        bench_trace_set = TraceSet(
            root=trace_set.root,
            definitions={defname: definition},
            solutions={defname: solutions},
            workloads={defname: workloads},
            traces={defname: []},
        )
        benchmark = Benchmark(bench_trace_set, cfg)
        result_trace_set = benchmark.run_all(dump_traces=True)
        traces = result_trace_set.traces.get(defname, [])
        one = {}
        for trace in traces:
            sol_name = trace.solution
            one.setdefault(sol_name, {})
            wl_uuid = trace.workload.uuid if trace.workload else "unknown"
            entry = {"workload": wl_uuid}
            if trace.evaluation:
                entry["status"] = trace.evaluation.status.value
                if trace.evaluation.performance:
                    entry["latency_ms"] = trace.evaluation.performance.latency_ms
                    entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                    entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
                if trace.evaluation.correctness:
                    entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                    entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            one[sol_name][wl_uuid] = entry
        return one

    runs = []
    for r in range(repeats):
        print(f"\n--- repeat {r + 1}/{repeats} ---")
        t0 = __import__("time").time()
        runs.append(_collect_one_run())
        print(f"--- repeat {r + 1} done in {__import__('time').time() - t0:.1f}s ---")

    return {"definition": defname, "repeats": repeats, "runs": runs}


def _summary_table(results: dict) -> None:
    """Print an aggregate per-solution summary across all repeats.

    For each solution we report mean latency (average across repeats of each
    workload's mean latency), geomean speedup, and the standard deviation of
    per-workload latency across repeats (noise level).
    """
    import math

    runs = results.get("runs", [])
    if not runs:
        print("(no repeats recorded)")
        return
    n_repeats = len(runs)

    sol_names = set()
    for run in runs:
        sol_names.update(run.keys())

    print(f"\n=== Aggregate across {n_repeats} repeats ===")
    print(f"{'solution':40s}  {'mean(us)':>10s}  {'stddev_run(%)':>14s}  "
          f"{'gmean(sp)':>10s}  {'amean(sp)':>10s}  {'min(sp)':>8s}  "
          f"{'max(sp)':>8s}  {'passed':>8s}")

    for sol in sorted(sol_names):
        # Collect per-workload latency across repeats
        per_wl_lats = {}
        per_wl_sps = {}
        passed = 0
        total = 0
        for run in runs:
            sol_traces = run.get(sol, {})
            for wl, e in sol_traces.items():
                total += 1
                if e.get("status") in ("PASSED", "PASS"):
                    passed += 1
                if e.get("latency_ms") is not None:
                    per_wl_lats.setdefault(wl, []).append(float(e["latency_ms"]))
                    per_wl_sps.setdefault(wl, []).append(float(e["speedup_factor"]))

        if not per_wl_lats:
            print(f"{sol:40s}  (no measurements)")
            continue

        # mean latency of each workload, then arithmetic mean across workloads
        wl_mean_lats = [sum(v) / len(v) for v in per_wl_lats.values()]
        mean_us = 1000.0 * sum(wl_mean_lats) / len(wl_mean_lats)

        # cross-repeat noise: mean(coeff-of-variation) per workload (%)
        def _stddev(xs):
            m = sum(xs) / len(xs)
            if len(xs) < 2:
                return 0.0
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
        cvs = []
        for wl, lats in per_wl_lats.items():
            if len(lats) < 2:
                continue
            m = sum(lats) / len(lats)
            if m > 0:
                cvs.append(_stddev(lats) / m * 100.0)
        cv_mean = sum(cvs) / len(cvs) if cvs else 0.0

        # per-workload geomean speedup (averaged across repeats)
        wl_mean_sps = [sum(v) / len(v) for v in per_wl_sps.values()]
        gmean = math.exp(sum(math.log(s) for s in wl_mean_sps) / len(wl_mean_sps))
        amean = sum(wl_mean_sps) / len(wl_mean_sps)

        print(f"{sol:40s}  {mean_us:>10.2f}  {cv_mean:>13.2f}%  "
              f"{gmean:>10.2f}  {amean:>10.2f}  {min(wl_mean_sps):>8.2f}  "
              f"{max(wl_mean_sps):>8.2f}  {passed:>3d}/{total:<3d}")


@app.local_entrypoint()
def main(
    solutions: str = "",
    output: str = None,
    smoke: bool = False,
    smoke_workloads: int = 10,
    smoke_seed: int = 42,
    repeats: int = 1,
):
    """Pack N solutions and run a paired benchmark on Modal.

    Parameters
    ----------
    solutions : str
        Comma-separated list of Solution JSON file paths.
    """
    paths = [p.strip() for p in solutions.split(",") if p.strip()]
    if not paths:
        raise SystemExit(
            "Must pass --solutions 'path1.json,path2.json,path3.json' "
            "(comma-separated)"
        )

    print(f"Loading {len(paths)} solutions:")
    sols_json = []
    for p in paths:
        txt = Path(p).read_text()
        s = Solution.model_validate_json(txt)
        print(f"  - {s.name}  <-  {p}")
        sols_json.append(txt)

    print(f"\nRunning paired benchmark on Modal B200 (repeats={repeats})...")
    results = run_compare.remote(
        sols_json,
        smoke=smoke,
        smoke_workloads=smoke_workloads,
        smoke_seed=smoke_seed,
        repeats=repeats,
    )

    if not results:
        print("No results returned!")
        return

    if output is None:
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
        output = f"artifacts/compare_{ts}.json"

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nFull per-workload results written to {out_path}")

    _summary_table(results)
