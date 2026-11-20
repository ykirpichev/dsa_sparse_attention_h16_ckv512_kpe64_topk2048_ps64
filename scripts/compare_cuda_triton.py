"""Compare the submitted CUDA kernel with its Triton translation on one B200.

Packs three solutions and runs the contest workloads:

  * wombat CUDA (solution/python, the submission kernel)
  * Triton (solution/triton, kernel.py::run), the same dispatcher and kernels
  * Triton tuned (kernel.py::run_tuned), the fastest setting from the sweep

Short timing: 5 warmup iterations, 50 measured, 1 trial. Correctness uses
atol 0.05 / rtol 1e-2, which is tighter than the contest CLI default and
matches the error bar the submission reported. The PyTorch reference is
checked once and is not included in the timing loop.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, Solution, TraceSet
from flashinfer_bench.agents import pack_solution_from_files

app = modal.App("dsa-cuda-vs-triton")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
DEFINITION = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("wheel", "setuptools")
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


def _pack(directory: Path, language: str, entry: str, name: str) -> Solution:
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry,
        destination_passing_style=True,
    )
    return pack_solution_from_files(
        path=str(directory),
        spec=spec,
        name=name,
        definition=DEFINITION,
        author="Wombat",
    )


def _bench_one(solution: Solution, config: BenchmarkConfig) -> dict:
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[DEFINITION]
    workloads = trace_set.workloads.get(DEFINITION, [])
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )
    result = Benchmark(bench_trace_set, config).run_all(dump_traces=True)
    out = {}
    for trace in result.traces.get(DEFINITION, []):
        if not trace.evaluation:
            continue
        entry = {"status": trace.evaluation.status.value}
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
        if trace.evaluation.correctness:
            entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        out[trace.workload.uuid] = entry
    return out


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_all(solutions: list) -> dict:
    config = BenchmarkConfig(
        warmup_runs=5,
        iterations=50,
        num_trials=1,
        atol=0.05,
        rtol=1e-2,
        profile_baseline=False,
        use_isolated_runner=False,
        timeout_seconds=600,
    )
    report = {}
    for solution in solutions:
        name = solution.name
        print(f"\n===== {name} =====", flush=True)
        try:
            report[name] = _bench_one(solution, config)
        except Exception:
            report[name] = {"error": traceback.format_exc()}
            print(report[name]["error"], flush=True)
    return report


def _load_tokens() -> dict:
    path = Path(
        "/Users/ykirpichev/sources/mlsys26-contest/workloads/dsa_paged/"
        "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    )
    tokens = {}
    if not path.exists():
        return tokens
    import json
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        workload = json.loads(line)["workload"]
        tokens[workload["uuid"]] = workload["axes"]["num_tokens"]
    return tokens


def _print_report(report: dict, tokens: dict) -> None:
    names = list(report)
    rows = []
    uuids = []
    for name in names:
        payload = report[name]
        if isinstance(payload, dict) and "error" in payload and len(payload) == 1:
            print(f"\n{name} FAILED\n{payload['error']}")
            continue
        for uuid in payload:
            if uuid not in uuids:
                uuids.append(uuid)

    print("\nuuid      T   " + "".join(f"{name:>22}" for name in names))
    cuda_name = next((n for n in names if "cuda" in n), None)
    for uuid in sorted(uuids, key=lambda u: (tokens.get(u, 0), u)):
        cells = []
        lat = {}
        for name in names:
            payload = report[name]
            if not isinstance(payload, dict) or uuid not in payload:
                cells.append(f"{'—':>22}")
                continue
            row = payload[uuid]
            status = row.get("status", "?")
            ms = row.get("latency_ms")
            err = row.get("max_abs_error")
            if ms is None:
                cells.append(f"{status:>22}")
            else:
                lat[name] = ms
                err_s = f" e{err:.1e}" if err is not None else ""
                cells.append(f"{ms * 1e3:7.1f}us {status[:4]}{err_s:>8}")
        t = tokens.get(uuid, "?")
        print(f"{uuid[:8]}  {t!s:>2} " + "".join(f"{c:>22}" for c in cells))
        rows.append(lat)

    if cuda_name:
        for name in names:
            if name == cuda_name:
                continue
            ratios = []
            for lat in rows:
                if cuda_name in lat and name in lat and lat[cuda_name] > 0:
                    ratios.append(lat[name] / lat[cuda_name])
            if not ratios:
                continue
            ratios.sort()
            geo = 1.0
            for r in ratios:
                geo *= r
            geo = geo ** (1.0 / len(ratios))
            print(
                f"\n{name} / {cuda_name}: geomean {geo:.2f}x  "
                f"(median {ratios[len(ratios)//2]:.2f}x, "
                f"best {ratios[0]:.2f}x, worst {ratios[-1]:.2f}x)  "
                f"over {len(ratios)} workloads"
            )


@app.local_entrypoint()
def main(extra: str = "", out: str = "triton_compare.json"):
    """``--extra "name=/dir/of/triton/sources:file.py::entry"`` adds an outside
    Triton solution (destination-passing) to the same run; separate several
    with ``;``."""
    solutions = [
        _pack(PROJECT_ROOT / "solution" / "python", "python", "solution.py::run", "wombat-cuda"),
        _pack(PROJECT_ROOT / "solution" / "triton", "triton", "kernel.py::run", "triton"),
        _pack(PROJECT_ROOT / "solution" / "triton", "triton", "kernel.py::run_tuned", "triton-tuned"),
    ]
    for item in filter(None, (part.strip() for part in extra.split(";"))):
        name, _, rest = item.partition("=")
        directory, _, entry = rest.partition(":")
        solutions.append(_pack(Path(directory), "triton", entry, name))
    for solution in solutions:
        print(f"packed {solution.name}: {[s.path for s in solution.sources]}")
    report = run_all.remote(solutions)
    out_path = PROJECT_ROOT / "artifacts" / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    _print_report(report, _load_tokens())
