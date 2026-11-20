"""Time CUDA, fused Triton, TileLang, and ThunderKittens on one B200.

  modal run scripts/profile_frameworks.py --which tilelang --limit 2
  modal run scripts/profile_frameworks.py --tilelang "16x128,16x64,8x128"

``--which`` is a comma list of cuda, triton, tilelang, tk, or "all".
``--tilelang`` is g x block, comma separated. The timer is the CUPTI span
used for the Triton profile, so the latencies compare directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("dsa-frameworks")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

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
    .run_commands(
        "pip install 'tilelang==0.1.15'",
        "git clone --depth 1 https://github.com/HazyResearch/ThunderKittens.git /opt/ThunderKittens",
    )
    .add_local_dir(str(PROJECT_ROOT / "solution"), "/root/solution")
    .add_local_file(str(PROJECT_ROOT / "scripts" / "triton_profile_worker.py"), "/root/triton_worker.py")
    .add_local_file(str(PROJECT_ROOT / "scripts" / "frameworks_worker.py"), "/root/frameworks_worker.py")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_worker(worker_args: list, timeout_s: int = 2400) -> dict:
    import json
    import os
    import subprocess
    import sys

    out_path = "/tmp/frameworks.json"
    command = [sys.executable, "/root/frameworks_worker.py", "--out", out_path, *worker_args]
    try:
        proc = subprocess.run(command, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
        code, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as expired:
        code, stderr = -1, f"worker timed out after {timeout_s} s\n" + str(expired.stderr or "")[-2000:]
    if os.path.exists(out_path):
        with open(out_path) as handle:
            report = json.load(handle)
    else:
        report = {}
    if code != 0:
        report["fatal"] = stderr[-4000:]
        print(report["fatal"], flush=True)
    return report


def _print(report: dict) -> None:
    rows = report.get("workloads", {})
    names = report.get("which", [])
    print(f"\n{report.get('gpu')}  tilelang {report.get('tilelang_version')}  triton {report.get('triton_version')}")
    if not rows:
        print(report.get("fatal", "no workloads"))
        return
    order = sorted(rows, key=lambda u: (rows[u]["T"], rows[u].get("valid_avg", 0), u))
    print(f"{'uuid':8} {'T':>2} " + " ".join(f"{n[:14]:>14}" for n in names))
    for uuid in order:
        row = rows[uuid]
        cells = []
        for name in names:
            us = (row["latency"].get(name) or {}).get("median_us")
            err = (row["error"].get(name) or [None])[0]
            cells.append(f"{us:6.1f}" if us else "  fail")
            if err is not None and err > 0.05:
                cells[-1] += "!"
        print(f"{uuid[:8]} {row['T']:>2} " + " ".join(f"{c:>14}" for c in cells))
    if report.get("errors"):
        print("\nFailures:")
        for name, text in report["errors"].items():
            print(f"  {name}: {text.strip().splitlines()[-1][:240]}")
    if report.get("fatal"):
        print("\n" + report["fatal"][-1500:])


@app.local_entrypoint()
def main(which: str = "all", tilelang: str = "16x128", limit: int = 0, iters: int = 10,
         out: str = "frameworks.json", timeout: int = 2400):
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    report = run_worker.remote(
        ["--which", which, "--tilelang", tilelang, "--limit", str(limit), "--iters", str(iters)],
        timeout,
    )
    path = artifacts / out
    path.write_text(json.dumps(report, indent=1))
    print(f"\nWrote {path}")
    _print(report)
