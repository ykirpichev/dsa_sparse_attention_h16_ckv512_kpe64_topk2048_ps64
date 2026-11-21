"""A/B two kernel.cu variants on Modal B200 against the official traces.

Used by the Research-agent kernel_experiment executor. Prints a JSON block
that the executor parses:

    ===AB_JSON===
    {...}
    ===END_AB_JSON===

Usage:
    modal run scripts/run_modal_ab.py \\
        --baseline-cu solution/python/kernel.cu \\
        --candidate-cu /tmp/candidate/kernel.cu \\
        --solution-py solution/python/solution.py \\
        --max-workloads 8
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

DEFINITION = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
TRACE_SET_PATH = "/data"

app = modal.App("wombat-dsa-ab")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
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


def _pack(name: str, kernel_cu: str, solution_py: str):
    from flashinfer_bench import BuildSpec, Solution
    from flashinfer_bench.data.solution import SourceFile

    spec = BuildSpec(
        language="python",
        target_hardware=["cuda"],
        entry_point="solution.py::run",
        destination_passing_style=True,
    )
    return Solution(
        name=name,
        definition=DEFINITION,
        author="Wombat-agent",
        description=name,
        spec=spec,
        sources=[
            SourceFile(path="kernel.cu", content=kernel_cu),
            SourceFile(path="solution.py", content=solution_py),
        ],
    )


def _pack_triton(name: str, py_file: str, source: str, entry: str):
    """External Triton solution (e.g. a competition winner) as the candidate."""
    from flashinfer_bench import BuildSpec, Solution
    from flashinfer_bench.data.solution import SourceFile

    spec = BuildSpec(
        language="triton",
        target_hardware=["cuda"],
        entry_point=f"{py_file}::{entry}",
        destination_passing_style=True,
    )
    return Solution(
        name=name,
        definition=DEFINITION,
        author="external",
        description=name,
        spec=spec,
        sources=[SourceFile(path=py_file, content=source)],
    )


def _error_excerpt(log: str, limit: int = 3000) -> str:
    """Compiler/runtime error lines (with a little context) from a failed evaluation."""
    if not log:
        return ""
    lines = log.splitlines()
    keep: list[str] = []
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in ("error", "exception", "traceback", "illegal", "assert")):
            keep.extend(lines[max(0, i - 1): i + 3])
    text = "\n".join(dict.fromkeys(keep)) or log[-limit:]
    return text[:limit]


def _geomean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


@app.function(
    image=image,
    gpu="B200:1",
    timeout=2400,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_ab(
    baseline_cu: str,
    candidate_cu: str,
    solution_py: str,
    max_workloads: int = 8,
    warmup: int = 1,
    iterations: int = 20,
    num_trials: int = 2,
    triton_file: str = "",
    triton_entry: str = "kernel",
    candidate_solution_py: str = "",
) -> dict:
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    baseline = _pack("wombat_baseline", baseline_cu, solution_py)
    if triton_file:
        candidate = _pack_triton("wombat_candidate", triton_file, candidate_cu, triton_entry)
    else:
        candidate = _pack("wombat_candidate", candidate_cu, candidate_solution_py or solution_py)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if DEFINITION not in trace_set.definitions:
        return {"ok": False, "error": f"definition {DEFINITION} not in trace set"}

    definition = trace_set.definitions[DEFINITION]
    workloads = list(trace_set.workloads.get(DEFINITION, []))
    workloads.sort(key=lambda w: getattr(getattr(w, "workload", w), "uuid", str(w)))
    if max_workloads > 0:
        workloads = workloads[:max_workloads]
    if not workloads:
        return {"ok": False, "error": "no workloads"}

    config = BenchmarkConfig(
        warmup_runs=warmup,
        iterations=iterations,
        num_trials=num_trials,
    )
    # An A/A run (same kernel twice) showed a ~12% position bias on one workload,
    # so benchmark both solution orders and average each latency across them.
    samples: dict[tuple[str, str], list[dict]] = {}
    axes_by_wl: dict[str, dict] = {}
    for order in ([baseline, candidate], [candidate, baseline]):
        bench_trace_set = TraceSet(
            root=trace_set.root,
            definitions={definition.name: definition},
            solutions={definition.name: order},
            workloads={definition.name: workloads},
            traces={definition.name: []},
        )
        result_trace_set = Benchmark(bench_trace_set, config).run_all(dump_traces=True)
        for trace in result_trace_set.traces.get(definition.name, []):
            ev = trace.evaluation
            if ev is None:
                continue
            wl = getattr(trace.workload, "uuid", str(trace.workload))
            axes_by_wl[wl] = dict(getattr(trace.workload, "axes", {}) or {})
            entry = {"status": ev.status.value}
            if ev.status.value != "PASSED" and getattr(ev, "log", ""):
                entry["log"] = ev.log
            if ev.performance:
                entry["latency_ms"] = ev.performance.latency_ms
            if ev.correctness:
                entry["max_abs_error"] = ev.correctness.max_absolute_error
            samples.setdefault((wl, trace.solution), []).append(entry)

    candidate_log = next(
        (e["log"] for (_, sol), entries in samples.items() if sol == "wombat_candidate"
         for e in entries if e.get("log")),
        "",
    )
    per: dict[str, dict] = {}
    for (wl, solution), entries in samples.items():
        lats = [e["latency_ms"] for e in entries if e.get("latency_ms")]
        errs = [e["max_abs_error"] for e in entries if e.get("max_abs_error") is not None]
        statuses = {e["status"] for e in entries}
        per.setdefault(wl, {})[solution] = {
            "status": "PASSED" if statuses == {"PASSED"} else sorted(statuses - {"PASSED"})[0],
            "latency_ms": sum(lats) / len(lats) if lats else None,
            "max_abs_error": max(errs) if errs else None,
        }

    rows = []
    base_lats, cand_lats, ratios = [], [], []
    base_pass = cand_pass = 0
    for uuid, pair in sorted(per.items()):
        b = pair.get("wombat_baseline", {})
        c = pair.get("wombat_candidate", {})
        if b.get("status") == "PASSED":
            base_pass += 1
        if c.get("status") == "PASSED":
            cand_pass += 1
        bl, cl = b.get("latency_ms"), c.get("latency_ms")
        if bl:
            base_lats.append(bl)
        if cl:
            cand_lats.append(cl)
        ratio = (bl / cl) if bl and cl else None
        if ratio:
            ratios.append(ratio)
        rows.append(
            {
                "workload": uuid[:8],
                "axes": axes_by_wl.get(uuid, {}),
                "baseline_status": b.get("status"),
                "candidate_status": c.get("status"),
                "baseline_ms": bl,
                "candidate_ms": cl,
                "speedup_vs_baseline": ratio,
                "candidate_abs_err": c.get("max_abs_error"),
            }
        )

    gm = _geomean(ratios)
    base_mean = sum(base_lats) / len(base_lats) if base_lats else None
    cand_mean = sum(cand_lats) / len(cand_lats) if cand_lats else None
    improved = bool(
        gm
        and gm > 1.01
        and cand_pass == len(rows)
        and base_pass == len(rows)
        and rows
    )
    return {
        "ok": True,
        "n_workloads": len(rows),
        "baseline_pass": base_pass,
        "candidate_pass": cand_pass,
        "baseline_mean_ms": base_mean,
        "candidate_mean_ms": cand_mean,
        "geomean_speedup_vs_baseline": gm,
        "mean_delta_ms": (base_mean - cand_mean) if base_mean and cand_mean else None,
        "improved": improved,
        "rows": rows,
        "candidate_log": _error_excerpt(candidate_log),
        "config": {
            "max_workloads": max_workloads,
            "warmup": warmup,
            "iterations": iterations,
            "num_trials": num_trials,
        },
    }


@app.local_entrypoint()
def main(
    baseline_cu: str,
    candidate_cu: str,
    solution_py: str = "solution/python/solution.py",
    max_workloads: int = 8,
    warmup: int = 1,
    iterations: int = 20,
    num_trials: int = 2,
    candidate_triton_entry: str = "",
    candidate_solution_py: str = "",
):
    """--candidate-triton-entry kernel treats --candidate-cu as a Triton .py solution."""
    root = PROJECT_ROOT
    b_path = Path(baseline_cu)
    c_path = Path(candidate_cu)
    p_path = Path(solution_py)
    if not b_path.is_absolute():
        b_path = root / b_path
    if not c_path.is_absolute():
        c_path = root / c_path
    if not p_path.is_absolute():
        p_path = root / p_path

    payload = run_ab.remote(
        b_path.read_text(),
        c_path.read_text(),
        p_path.read_text(),
        max_workloads,
        warmup,
        iterations,
        num_trials,
        c_path.name if candidate_triton_entry else "",
        candidate_triton_entry or "kernel",
        Path(candidate_solution_py).read_text() if candidate_solution_py else "",
    )
    print("===AB_JSON===")
    print(json.dumps(payload, indent=2))
    print("===END_AB_JSON===")
    if not payload.get("ok"):
        raise SystemExit(payload.get("error") or "A/B failed")
    gm = payload.get("geomean_speedup_vs_baseline")
    print(
        f"summary: n={payload.get('n_workloads')} "
        f"pass={payload.get('candidate_pass')}/{payload.get('n_workloads')} "
        f"geomean={gm:.3f}x vs baseline "
        f"improved={payload.get('improved')}"
        if gm
        else f"summary: {payload}"
    )
