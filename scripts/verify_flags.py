"""One-off Modal helper: compile kernel.cu with -Xptxas=-v for each of a
few variant flag sets and print ptxas output + SASS sha256 so we can confirm
the flags actually reach nvcc and change codegen.
"""

import hashlib
import subprocess
from pathlib import Path

import modal

app = modal.App("nvcc-verify-flags")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)


@app.function(image=image, gpu="B200:1", timeout=1200)
def verify(kernel_cu: str) -> dict:
    workdir = Path("/tmp/verify_flags")
    workdir.mkdir(exist_ok=True)
    cu_path = workdir / "kernel.cu"
    cu_path.write_text(kernel_cu)

    # Need TVM-FFI headers for the kernel to compile.
    try:
        import tvm_ffi
        from pathlib import Path as _P
        tvm_ffi_inc = _P(tvm_ffi.__file__).parent / "include"
        dlpack_inc = _P(tvm_ffi.__file__).parent / "3rdparty" / "dlpack" / "include"
    except ImportError:
        subprocess.run(
            "pip install --no-deps git+https://github.com/flashinfer-ai/flashinfer-bench.git",
            shell=True,
            check=True,
        )
        import tvm_ffi
        tvm_ffi_inc = Path(tvm_ffi.__file__).parent / "include"
        dlpack_inc = Path(tvm_ffi.__file__).parent / "3rdparty" / "dlpack" / "include"

    base = [
        "-gencode=arch=compute_100a,code=sm_100a",
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
    ]

    variants = {
        "A_baseline": [],
        "B_expensive": ["-Xptxas=-allow-expensive-optimizations=true", "-DNDEBUG"],
        "C_regs72": ["-maxrregcount=72"],
        "D_regs64": ["-maxrregcount=64"],
        "G_regs96": ["-maxrregcount=96"],
    }

    results = {}
    for name, extra in variants.items():
        out_o = workdir / f"{name}.o"
        cmd = [
            "/usr/local/cuda/bin/nvcc",
            "-c",
            str(cu_path),
            "-o",
            str(out_o),
            f"-I{tvm_ffi_inc}",
            f"-I{dlpack_inc}",
            "-Xptxas=-v",
            "-Xptxas=-warn-spills",
        ] + base + extra

        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(workdir)
        )
        if proc.returncode != 0:
            results[name] = {"error": proc.stderr[-2000:]}
            continue

        ptxas_out = proc.stderr
        if out_o.exists():
            h = hashlib.sha256(out_o.read_bytes()).hexdigest()[:16]
            size = out_o.stat().st_size
        else:
            h, size = None, None

        # Extract the "ptxas info" lines
        ptxas_lines = [
            ln for ln in ptxas_out.splitlines()
            if "ptxas info" in ln or "Used" in ln or "spill" in ln.lower()
        ]
        results[name] = {
            "sha256": h,
            "size_bytes": size,
            "ptxas_summary": ptxas_lines,
            "ptxas_raw_tail": ptxas_out[-1500:],
        }
    return results


@app.local_entrypoint()
def main():
    kernel_cu = Path(
        "solution/python/kernel.cu"
    ).read_text()
    res = verify.remote(kernel_cu)

    import json
    print(json.dumps(res, indent=2, default=str))

    print("\n=== Summary ===")
    print(f"{'variant':15s}  {'sha256(.o)':18s}  {'size':>10s}")
    for name, d in res.items():
        if "error" in d:
            print(f"{name:15s}  ERROR: {d['error'][:80]}")
        else:
            print(f"{name:15s}  {d['sha256']:18s}  {d['size_bytes']:>10}")

    print("\n=== ptxas info (registers / spills / occupancy) ===")
    for name, d in res.items():
        if "ptxas_summary" in d:
            print(f"\n-- {name} --")
            for ln in d["ptxas_summary"]:
                print(f"  {ln}")
