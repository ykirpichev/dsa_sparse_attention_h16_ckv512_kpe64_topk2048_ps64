"""
Four-way Modal B200 comparison:

    1. Python reference (definition's built-in reference impl)
    2. FlashInfer baseline (flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla)
    3. Our current solution at HEAD (CUDA-graph cache always on)
    4. Previous version (passed via --previous-log path on the local entrypoint)
       — only shown in the final table if the log file exists.

Both (2) and (3) are benched in a single Modal invocation so their numbers
are comparable (same container, same B200 instance, same reference timing).

Usage:
    modal run scripts/run_modal_compare.py \
        [--previous-log artifacts/graph_cache/bench_graph_off_20260424T042117Z.log]
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

FLASHINFER_BASELINE_JSON = Path(
    "/home/ykirpichev/sources/mlsys26-contest/solutions/baseline/dsa/"
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/"
    "flashinfer_wrapper_5af199.json"
)

app = modal.App("flashinfer-bench-compare")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

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


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solutions: list, label: str) -> dict:
    """Run a list of solutions against the same reference on Modal B200.

    Returns a nested dict: {solution_name: {workload_uuid: {...}}}.
    reference_latency_ms is attached per-workload (identical across solutions
    for a given workload, since the reference is deterministic input-wise).
    """
    sol_objs = [Solution.model_validate(s) for s in solutions]

    definition_name = sol_objs[0].definition
    for s in sol_objs[1:]:
        assert s.definition == definition_name, \
            f"Solution '{s.name}' targets a different definition"

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[definition_name]
    workloads = trace_set.workloads.get(definition_name, [])
    assert workloads, f"No workloads for definition '{definition_name}'"

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: sol_objs},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    out: dict = {}
    for trace in traces:
        ev = trace.evaluation
        if ev is None:
            continue
        per_sol = out.setdefault(trace.solution, {})
        entry = {"status": ev.status.value}
        if ev.performance:
            entry["latency_ms"] = ev.performance.latency_ms
            entry["reference_latency_ms"] = ev.performance.reference_latency_ms
            entry["speedup_factor"] = ev.performance.speedup_factor
        if ev.correctness:
            entry["max_abs_error"] = ev.correctness.max_absolute_error
            entry["max_rel_error"] = ev.correctness.max_relative_error
        per_sol[trace.workload.uuid] = entry

    return {"label": label, "results": out}


# -----------------------------------------------------------------------------
# Local-side helpers: parse logs and render a comparison table.
# -----------------------------------------------------------------------------

_LOG_LINE_RE = re.compile(
    r"Workload\s+(?P<uuid>[0-9a-f]+)\.{3}:\s+PASSED\s+\|\s+"
    r"(?P<lat>[0-9.]+)\s+ms\s+\|\s+"
    r"(?P<spd>[0-9.]+)x speedup"
)


def parse_log(path: Path) -> dict:
    """Parse a printed benchmark log into {uuid: {latency_ms, speedup}}."""
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = _LOG_LINE_RE.search(line)
        if m:
            uuid = m.group("uuid")
            out[uuid] = {
                "latency_ms": float(m.group("lat")),
                "speedup_factor": float(m.group("spd")),
            }
    return out


def print_comparison_table(
    current: dict, flashinfer: dict, previous: dict,
):
    """Render: workload | python-ref | flashinfer | previous | current | speedup-vs-flashinfer.

    `previous` is keyed by 8-char UUID prefix (from parsed log output);
    `current` and `flashinfer` are keyed by full 32-char UUIDs from Modal.
    We normalize by 8-char prefix for the lookup.
    """
    import math

    prev_by_prefix = {k[:8]: v for k, v in previous.items()}

    def fmt(x):
        return f"{x:.3f} ms" if x is not None else "       -"

    def ratio(a, b):
        if a and b:
            return f"{a / b:5.2f}x"
        return "    -"

    header = (
        f"{'workload':>10}  "
        f"{'python ref':>12}  "
        f"{'flashinfer':>12}  "
        f"{'prev (off)':>12}  "
        f"{'current':>9}  "
        f"{'vs FI':>7}  "
        f"{'vs prev':>7}"
    )
    print()
    print(header)
    print("-" * len(header))

    all_uuids = sorted(set(current) | set(flashinfer))
    row_records = []  # for aggregation

    for uuid in all_uuids:
        cur = current.get(uuid, {})
        fi_r = flashinfer.get(uuid, {})
        prev = prev_by_prefix.get(uuid[:8], {})

        cur_lat = cur.get("latency_ms")
        fi_lat = fi_r.get("latency_ms")
        prev_lat = prev.get("latency_ms")

        ref_lat = None
        for r in (cur, fi_r):
            if r.get("reference_latency_ms") is not None:
                ref_lat = r["reference_latency_ms"]
                break
            if (r.get("latency_ms") is not None
                    and r.get("speedup_factor") is not None):
                ref_lat = r["latency_ms"] * r["speedup_factor"]
                break

        print(
            f"{uuid[:8]:>10}  "
            f"{fmt(ref_lat):>12}  "
            f"{fmt(fi_lat):>12}  "
            f"{fmt(prev_lat):>12}  "
            f"{fmt(cur_lat):>9}  "
            f"{ratio(fi_lat, cur_lat):>7}  "
            f"{ratio(prev_lat, cur_lat):>7}"
        )
        row_records.append({
            "uuid": uuid,
            "ref": ref_lat,
            "fi": fi_lat,
            "prev": prev_lat,
            "cur": cur_lat,
        })

    def geomean(vals):
        vals = [v for v in vals if v and v > 0]
        if not vals:
            return None
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    vs_fi = [r["fi"] / r["cur"] for r in row_records if r["fi"] and r["cur"]]
    vs_prev = [r["prev"] / r["cur"] for r in row_records if r["prev"] and r["cur"]]
    vs_py = [r["ref"] / r["cur"] for r in row_records if r["ref"] and r["cur"]]

    gm_fi = geomean(vs_fi)
    gm_prev = geomean(vs_prev)
    gm_py = geomean(vs_py)

    print("-" * len(header))
    print(
        f"{'geomean':>10}  "
        f"{'':>12}  "
        f"{'':>12}  "
        f"{'':>12}  "
        f"{'':>9}  "
        f"{(f'{gm_fi:5.2f}x' if gm_fi else '-'):>7}  "
        f"{(f'{gm_prev:5.2f}x' if gm_prev else '-'):>7}"
    )
    if gm_py is not None:
        print(f"\n  current vs python reference (geomean speedup): {gm_py:.1f}x")
    if gm_fi is not None:
        print(f"  current vs flashinfer baseline (geomean speedup): {gm_fi:.1f}x")
    if gm_prev is not None:
        print(f"  current vs previous version   (geomean speedup): {gm_prev:.2f}x")


@app.local_entrypoint()
def main(previous_log: str = "artifacts/graph_cache/bench_graph_off_20260424T042117Z.log"):
    """Run our solution and the FlashInfer baseline on Modal B200, then
    print a consolidated comparison including the previous-version log."""
    from scripts.pack_solution import pack_solution

    print("Packing our solution from source...")
    solution_path = pack_solution()
    our_solution = json.loads(solution_path.read_text())

    print(f"Loading FlashInfer baseline from {FLASHINFER_BASELINE_JSON}...")
    fi_solution = json.loads(FLASHINFER_BASELINE_JSON.read_text())

    print("\nLaunching combined benchmark on Modal B200 "
          "(our solution + FlashInfer baseline, same reference, same B200)...\n")
    payload = run_benchmark.remote(
        [our_solution, fi_solution], label="compare",
    )

    results = payload["results"]

    our_name = our_solution["name"]
    fi_name = fi_solution["name"]

    our_results = results.get(our_name, {})
    fi_results = results.get(fi_name, {})

    print(f"\n=== Raw per-workload numbers ({our_name}) ===")
    for uuid, r in sorted(our_results.items()):
        lat = r.get("latency_ms")
        spd = r.get("speedup_factor")
        ref = r.get("reference_latency_ms")
        print(f"  {uuid[:8]}  lat={lat:.4f} ms  ref={ref:.4f} ms  speedup={spd:.2f}x")

    print(f"\n=== Raw per-workload numbers ({fi_name}) ===")
    for uuid, r in sorted(fi_results.items()):
        lat = r.get("latency_ms")
        spd = r.get("speedup_factor")
        ref = r.get("reference_latency_ms")
        if lat is None:
            print(f"  {uuid[:8]}  status={r.get('status')}")
        else:
            print(f"  {uuid[:8]}  lat={lat:.4f} ms  ref={ref:.4f} ms  speedup={spd:.2f}x")

    previous = parse_log(PROJECT_ROOT / previous_log)
    if not previous:
        print(f"\n[note] previous-version log not found or empty: {previous_log}")

    print_comparison_table(
        current=our_results, flashinfer=fi_results, previous=previous,
    )
