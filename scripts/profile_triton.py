"""Find out where the Triton port loses time against the CUDA submission.

Two modes, both on one B200 per run:

  modal run scripts/profile_triton.py
      Profile on the image's Triton: per-workload index density, the kernels
      each implementation launched, host time, and a sweep over Triton
      configurations (kernel, split-K factor, warps, tile size).
      Writes artifacts/triton_profile.json.

  modal run scripts/profile_triton.py --versions 3.5.1,3.6.0 --configs "auto;token,16,8,16,1"
      Run the listed configurations under each Triton release, one container
      per release. Writes artifacts/triton_versions.json.

``--stress N`` also checks every configuration on N synthetic calls (new
contents each call, 1 to 20 tokens). ``--timeout`` caps the run in seconds,
since a kernel that waits on other programs can hang.

The measurement code is scripts/triton_profile_worker.py.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("dsa-triton-profile")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Same layers as scripts/run_modal.py, so the built image is reused.
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
    .add_local_dir(str(PROJECT_ROOT / "solution"), "/root/solution")
    .add_local_file(str(PROJECT_ROOT / "scripts" / "triton_profile_worker.py"), "/root/worker.py")
)

# DSA_EXTRA_KERNEL=/path/file.py::function times an outside Triton entry
# point in the same run, as the configuration named "extra".
EXTRA_KERNEL = os.environ.get("DSA_EXTRA_KERNEL", "")
if EXTRA_KERNEL and os.path.exists(EXTRA_KERNEL.partition("::")[0]):
    image = image.add_local_file(EXTRA_KERNEL.partition("::")[0], "/root/extra_kernel.py")


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_worker(worker_args: list, triton_version: str = "", timeout_s: int = 1800) -> dict:
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    # "image" means the Triton that ships in the container, with no install.
    if triton_version and triton_version != "image":
        target = f"/tmp/triton-{triton_version}"
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--target", target,
             f"triton=={triton_version}"],
            capture_output=True, text=True,
        )
        if install.returncode != 0:
            return {"triton_version": triton_version, "fatal": install.stderr[-1500:]}
        env["PYTHONPATH"] = target + os.pathsep + env.get("PYTHONPATH", "")
        env["TRITON_CACHE_DIR"] = f"/tmp/triton-cache-{triton_version}"

    out_path = "/tmp/profile.json"
    command = [
        sys.executable, "/root/worker.py",
        "--kernel", "/root/solution/triton/kernel.py",
        "--out", out_path, *worker_args,
    ]
    # A kernel that waits on other programs can hang; do not wait an hour.
    try:
        proc = subprocess.run(command, env=env, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
        code, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as expired:
        code, stderr = -1, f"worker timed out after {timeout_s} s\n" + str(expired.stderr or "")[-2000:]
    if os.path.exists(out_path):
        with open(out_path) as handle:
            report = json.load(handle)
    else:
        report = {"triton_version": triton_version}
    if code != 0:
        report["fatal"] = stderr[-3000:]
        print(report["fatal"], flush=True)
    return report


def _geomean(values: list) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _mean(row: dict, name: str):
    """Median latency in microseconds, the number flashinfer-bench reports."""
    entry = row["latency"].get(name)
    return entry["median_us"] if entry else None


def _short(name: str) -> str:
    """Readable name for a mangled CUDA kernel symbol."""
    import re

    ints = "/".join(re.findall(r"Li(\d+)E", name)[:3])
    for needle, label in (
        ("cta_per_t", "per-token"),
        ("split_partial", "split-partial"),
        ("split_reduce", "reduce"),
        ("count_valid", "density-count"),
        ("mla_sparse_attn_kernel", "per-head"),
    ):
        if needle in name:
            return f"{label}<{ints}>" if ints else label
    return name[:36]


def _print_profile(report: dict) -> None:
    print(f"\nTriton {report.get('triton_version')}  torch {report.get('torch_version')}  {report.get('gpu')}")
    rows = report.get("workloads", {})
    order = sorted(rows, key=lambda u: (rows[u]["T"], rows[u]["valid_avg"], u))

    print("\nPer workload (us, median over timed calls):")
    print(f"{'uuid':8} {'T':>2} {'valid':>5} {'plan':>9} {'cuda':>7} {'auto':>7} {'x':>5}  best config")
    for uuid in order:
        row = rows[uuid]
        cuda, auto = _mean(row, "cuda"), _mean(row, "auto")
        variants = {c: v["median_us"] for c, v in row["latency"].items() if c not in ("cuda", "auto")}
        best = min(variants, key=variants.get) if variants else None
        plan = f"{row['plan'][0]}/{row['plan'][1]}"
        ratio = f"{auto / cuda:5.2f}" if cuda and auto else "    -"
        best_s = f"{best} {variants[best]:.1f}" if best else ""
        if best and cuda:
            best_s += f" ({variants[best] / cuda:.2f}x)"
        print(
            f"{uuid[:8]} {row['T']:>2} {row['valid_avg']:>5} {plan:>9} "
            f"{cuda or 0:7.1f} {auto or 0:7.1f} {ratio}  {best_s}"
        )

    print("\nKernels per call: span, summed kernel time, host time, then each kernel (us):")
    for uuid in order:
        row = rows[uuid]
        for name in ("cuda", "auto"):
            entry = row["latency"].get(name)
            if not entry:
                continue
            parts = ", ".join(
                f"{_short(k)} x{v['calls']:g} {v['gpu_us']:.1f}" for k, v in entry["kernels"].items()
            )
            print(
                f"{uuid[:8]} {name:5} span {entry['median_us']:6.1f} busy {entry['busy_us']:6.1f} "
                f"host {entry['host_us']:6.1f} | {parts}"
            )

    configs = sorted({c for row in rows.values() for c in row["latency"] if c != "cuda"})
    if len(configs) <= 14:
        print("\nEvery configuration on every workload (us):")
        for i, config in enumerate(configs):
            print(f"  [{i}] {config}")
        print(f"{'uuid':8} {'T':>2} {'max':>5} {'cuda':>6} " + " ".join(f"[{i}]".rjust(6) for i in range(len(configs))))
        for uuid in order:
            row = rows[uuid]
            cells = " ".join(f"{_mean(row, c) or 0:6.1f}" for c in configs)
            print(f"{uuid[:8]} {row['T']:>2} {max(row['valid_per_token']):>5} {_mean(row, 'cuda') or 0:6.1f} {cells}")
    scored = []
    for config in configs:
        ratios = [
            _mean(row, config) / _mean(row, "cuda")
            for row in rows.values()
            if _mean(row, config) and _mean(row, "cuda")
        ]
        if len(ratios) == len(rows):
            scored.append((_geomean(ratios), max(ratios), config))
    if scored:
        print("\nOne configuration for every workload, geomean latency vs CUDA (best 12):")
        for geo, worst, config in sorted(scored)[:12]:
            print(f"  {config:22} {geo:5.2f}x  (worst {worst:.2f}x)")
        best_each = [
            min(v["median_us"] for c, v in row["latency"].items() if c not in ("cuda", "auto"))
            / _mean(row, "cuda")
            for row in rows.values()
            if _mean(row, "cuda") and len(row["latency"]) > 2
        ]
        if best_each:
            print(f"  best config per workload   {_geomean(best_each):5.2f}x")

    bad = {
        (uuid[:8], c): e
        for uuid, row in rows.items()
        for c, e in row["error"].items()
        if e[0] > 0.05 or e[1] > 0.05
    }
    if bad:
        print("\nConfigurations over 0.05 max abs error:", bad)
    for config, result in report.get("stress", {}).items():
        print(f"\nSynthetic calls, {config}: {result}")
    if report.get("errors"):
        print("\nFailures:")
        for name, text in report["errors"].items():
            print(f"  {name}: {text.strip().splitlines()[-1][:200]}")


def _print_versions(reports: list) -> None:
    configs = sorted({c for r in reports for row in r.get("workloads", {}).values() for c in row["latency"]})
    print("\nLatency over the workloads (us): mean / geomean")
    print(f"{'triton':>8} " + " ".join(f"{c:>22}" for c in configs))
    for report in reports:
        version = report.get("triton_version", "?")
        if "workloads" not in report or not report["workloads"]:
            print(f"{version:>8} failed: {str(report.get('fatal', ''))[-300:]}")
            continue
        cells = []
        for config in configs:
            values = [_mean(row, config) for row in report["workloads"].values()]
            if all(values):
                cells.append(f"{sum(values) / len(values):9.1f} / {_geomean(values):9.1f}")
            else:
                cells.append(f"{'failed':>21}")
        print(f"{version:>8} " + " ".join(f"{c:>22}" for c in cells))
        for name, text in report.get("errors", {}).items():
            print(f"         {name}: {text.strip().splitlines()[-1][:160]}")


@app.local_entrypoint()
def main(versions: str = "", configs: str = "", iters: int = 20, out: str = "", stress: int = 0, timeout: int = 1800):
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    base_args = ["--iters", str(iters), "--stress", str(stress)]
    if configs:
        base_args += ["--configs", configs]
    if EXTRA_KERNEL:
        base_args += ["--extra", "/root/extra_kernel.py::" + (EXTRA_KERNEL.partition("::")[2] or "kernel")]

    if not versions:
        report = run_worker.remote(
            base_args + ["--cuda-dir", "/root/solution/python"], "", timeout
        )
        path = artifacts / (out or "triton_profile.json")
        path.write_text(json.dumps(report, indent=1))
        print(f"\nWrote {path}")
        _print_profile(report)
        return

    wanted = [v.strip() for v in versions.split(",") if v.strip()]
    reports = list(run_worker.starmap([(base_args, version) for version in wanted]))
    for version, report in zip(wanted, reports):
        report.setdefault("triton_version", version)
        report["requested_version"] = version
    path = artifacts / (out or "triton_versions.json")
    path.write_text(json.dumps(reports, indent=1))
    print(f"\nWrote {path}")
    _print_versions(reports)
