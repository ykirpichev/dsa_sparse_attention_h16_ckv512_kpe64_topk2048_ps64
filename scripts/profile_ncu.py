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

    # Skip the bare pre-flight run: it timed out opaquely on 2026-04-22 (the
    # subprocess pipe-buffered a large compile log and Modal killed the task
    # at its 2400 s timeout). Go straight to NCU; any compile / harness error
    # will surface in NCU's captured output just as well.

    # Diagnostics: print driver / NCU / CUDA versions so we can debug any
    # "LibraryNotLoaded" class of NCU errors.
    print("=== environment diagnostics ===", flush=True)
    for cmd in [
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
        ["ncu", "--version"],
        ["nvcc", "--version"],
        ["bash", "-lc", "find / -name 'libcupti*.so*' 2>/dev/null | head -20"],
        ["bash", "-lc", "ldconfig -p | grep -i cupti | head -20"],
        ["bash", "-lc", "ls -la /usr/local/cuda-*/extras/CUPTI/lib64 2>/dev/null | head -20"],
        ["bash", "-lc", "ls /usr/local/cuda-*/version.txt /usr/local/cuda-*/version.json 2>/dev/null | head -5"],
        ["bash", "-lc", "cat /proc/driver/nvidia/params 2>/dev/null | grep -E '(NVreg_RestrictProfil|EnableResiz|Rm)' | head -5"],
    ]:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            print(f"$ {' '.join(cmd)}", flush=True)
            print(out.stdout, flush=True)
            if out.stderr.strip():
                print("stderr:", out.stderr, flush=True)
        except Exception as e:
            print(f"  ({cmd[0]} failed: {e})", flush=True)
    print("=== end diagnostics ===", flush=True)

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
    print("Running:", " ".join(shlex.quote(x) for x in ncu_cmd), flush=True)

    # Stream child stdout/stderr line-by-line so Modal shows progress as NCU
    # compiles + warms up + captures metrics. Without this, subprocess can
    # block for 10+ min behind full pipe buffers with zero visible output.
    import threading
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    proc_h = subprocess.Popen(
        ncu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )

    def _pump(stream, sink, tag):
        for line in iter(stream.readline, ""):
            sink.append(line)
            print(f"[{tag}] {line.rstrip()}", flush=True)
        stream.close()

    t_out = threading.Thread(target=_pump, args=(proc_h.stdout, stdout_lines, "ncu-out"))
    t_err = threading.Thread(target=_pump, args=(proc_h.stderr, stderr_lines, "ncu-err"))
    t_out.start(); t_err.start()
    try:
        proc_h.wait(timeout=2400)
    except subprocess.TimeoutExpired:
        proc_h.kill()
        proc_h.wait()
    t_out.join(); t_err.join()

    class _P:
        pass
    proc = _P()
    proc.returncode = proc_h.returncode
    proc.stdout = "".join(stdout_lines)
    proc.stderr = "".join(stderr_lines)

    # If NCU failed, dump its latest crash log so we can see the real cause.
    if proc.returncode != 0:
        try:
            import glob
            logs = sorted(glob.glob("/tmp/nsight-compute-*.log"))
            if logs:
                print(f"=== {logs[-1]} ===", flush=True)
                with open(logs[-1]) as lf:
                    print(lf.read(), flush=True)
                print("=== end ncu log ===", flush=True)
        except Exception as e:
            print(f"(failed to dump ncu log: {e})", flush=True)

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
        print("Running (text):", " ".join(shlex.quote(x) for x in ncu_cmd_txt), flush=True)
        txt_out: list[str] = []
        txt_err: list[str] = []
        pt = subprocess.Popen(
            ncu_cmd_txt, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, bufsize=1,
        )
        t_o = threading.Thread(target=_pump, args=(pt.stdout, txt_out, "ncu-txt-out"))
        t_e = threading.Thread(target=_pump, args=(pt.stderr, txt_err, "ncu-txt-err"))
        t_o.start(); t_e.start()
        try:
            pt.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            pt.kill(); pt.wait()
        t_o.join(); t_e.join()
        text_report = "".join(txt_out) + "\n--STDERR--\n" + "".join(txt_err)

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
