"""
NCU (Nsight Compute) profiling of mla_sparse_attn_kernel on Modal B200.

Runs the full benchmark under `ncu` with a kernel-name filter and reports
section-level metrics (speed-of-light, memory, warp stall reasons, scheduler,
SMEM).  Writes raw NCU report + a parsed summary to ./artifacts/.

Usage:
    modal run scripts/profile_ncu.py                 # default long workload
    modal run scripts/profile_ncu.py --workload 0c23b10c
    modal run scripts/profile_ncu.py --sections full # all NCU sections (slow)
"""

import json
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench-ncu")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Same image as run_modal.py (evaluator environment).
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
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def profile_remote(
    solution: Solution,
    workload_prefix: str,
    sections: str = "default",
) -> dict:
    """Run `ncu` on a single workload's kernel launch and return the report text.

    Writes a Python harness to disk that invokes the benchmark's kernel on the
    chosen workload, then wraps the whole process in `ncu --set ... --target-
    processes all`. Captures the full stdout/stderr (NCU prints tables there).
    """
    import os
    import subprocess
    import tempfile

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    def _uuid_of(w):
        # TraceSet.workloads stores Trace objects whose `.workload.uuid` is
        # the workload identifier printed by the benchmark.
        return getattr(getattr(w, "workload", w), "uuid", None)

    picked = [w for w in workloads if (_uuid_of(w) or "").startswith(workload_prefix)]
    if not picked:
        available = [(_uuid_of(w) or "")[:8] for w in workloads[:10]]
        raise ValueError(
            f"workload prefix '{workload_prefix}' did not match; first 10: {available}"
        )
    workload = picked[0]
    wl_uuid = _uuid_of(workload)
    print(f"Profiling workload {wl_uuid}")

    # Write the sub-process harness. It does the full compile + run of a single
    # workload. NCU attaches to this whole python invocation and filters on
    # kernel name with `--kernel-name regex:.*mla_sparse_attn_kernel.*`.
    harness = f"""
import json
import sys
sys.path.insert(0, {repr(str(PROJECT_ROOT))})

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

ts = TraceSet.from_path({repr(TRACE_SET_PATH)})
defn = ts.definitions[{repr(solution.definition)}]
_want = {repr(wl_uuid)}
def _uuid_of(w):
    return getattr(getattr(w, "workload", w), "uuid", None)
workloads = [w for w in ts.workloads[defn.name] if _uuid_of(w) == _want]
sol = Solution.model_validate_json({repr(solution.model_dump_json())})
# Warmup only 1, iterations 3, 1 trial: ncu slows everything 50-1000x, we only
# need enough launches for profiling. NCU `--launch-skip 2 --launch-count 1`
# filters to one clean profiled launch.
config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)
tset = TraceSet(
    root=ts.root,
    definitions={{defn.name: defn}},
    solutions={{defn.name: [sol]}},
    workloads={{defn.name: workloads}},
    traces={{defn.name: []}},
)
Benchmark(tset, config).run_all(dump_traces=False)
print("HARNESS_DONE")
"""

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(harness)
        harness_path = f.name

    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.0a")

    # Run the harness bare first so we can see any compile errors clearly.
    print("=== bare harness run ===")
    bare = subprocess.run(
        ["python", harness_path], capture_output=True, text=True, env=env, timeout=2400
    )
    print("bare rc:", bare.returncode)
    print("bare stdout tail:\n", bare.stdout[-2000:])
    print("bare stderr tail:\n", bare.stderr[-2000:])
    if bare.returncode != 0:
        return {
            "workload_uuid": wl_uuid,
            "returncode": bare.returncode,
            "csv_stdout": bare.stdout,
            "csv_stderr": bare.stderr,
            "text_report": "harness failed before ncu could attach",
        }

    # Section set: `default` is ~15s/launch, `full` is minutes.
    if sections == "full":
        section_args = ["--set", "full"]
    else:
        # Minimum viable set for perf analysis: SoL + memory + warp state +
        # scheduler + SMEM.
        section_args = [
            "--section", "SpeedOfLight",
            "--section", "MemoryWorkloadAnalysis",
            "--section", "WarpStateStats",
            "--section", "SchedulerStats",
            "--section", "ComputeWorkloadAnalysis",
            "--section", "Occupancy",
            "--section", "LaunchStats",
        ]

    # Launch-skip high enough to get past warmup compiles, launch-count 1 so
    # the report is fast / focused.
    ncu_cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name", "regex:.*mla_sparse_attn_kernel.*",
        "--launch-skip", "1",
        "--launch-count", "1",
        *section_args,
        "--csv",
        "python", harness_path,
    ]
    print("Running:", " ".join(shlex.quote(x) for x in ncu_cmd))

    proc = subprocess.run(
        ncu_cmd, capture_output=True, text=True, env=env, timeout=2400
    )

    # Also do a non-CSV run (human-readable) for a second pass, since CSV
    # omits some text. Only if the first call succeeded, to save time.
    text_report = ""
    if proc.returncode == 0 and sections != "full":
        ncu_cmd_txt = [
            "ncu",
            "--target-processes", "all",
            "--kernel-name", "regex:.*mla_sparse_attn_kernel.*",
            "--launch-skip", "1",
            "--launch-count", "1",
            *section_args,
            "python", harness_path,
        ]
        proc_txt = subprocess.run(
            ncu_cmd_txt, capture_output=True, text=True, env=env, timeout=2400
        )
        text_report = proc_txt.stdout + "\n--STDERR--\n" + proc_txt.stderr

    return {
        "workload_uuid": wl_uuid,
        "returncode": proc.returncode,
        "csv_stdout": proc.stdout,
        "csv_stderr": proc.stderr,
        "text_report": text_report,
    }


@app.local_entrypoint()
def main(
    workload: str = "5096e459",  # one of the long-trace workloads
    sections: str = "default",
    out_prefix: str = "ncu_exp06",
):
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")
    print(f"Profiling workload prefix '{workload}' with sections='{sections}'")

    result = profile_remote.remote(solution, workload, sections)
    print(f"ncu returncode: {result['returncode']}")

    art_dir = PROJECT_ROOT / "artifacts"
    art_dir.mkdir(exist_ok=True)
    csv_path = art_dir / f"{out_prefix}_{workload}_{sections}.csv"
    txt_path = art_dir / f"{out_prefix}_{workload}_{sections}.txt"
    err_path = art_dir / f"{out_prefix}_{workload}_{sections}.stderr.log"

    csv_path.write_text(result["csv_stdout"])
    txt_path.write_text(result["text_report"])
    err_path.write_text(result["csv_stderr"])
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {err_path}")
