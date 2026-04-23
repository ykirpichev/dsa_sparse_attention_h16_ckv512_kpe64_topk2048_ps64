"""Python wrapper that self-compiles kernel.cu with explicit ``-arch=sm_100a``.

Per ``FAQ.md``:

  "The evaluation machine supports sm_100a. If your CUDA kernel needs
   sm_100a-specific features, specify the architecture explicitly in your
   build flags (e.g., -arch=sm_100a). The build system does not
   automatically set SM targets."

  "The builders currently do not support custom compile flags. As a
   workaround, submit a Python solution and compile the CUDA kernel yourself
   within the code (using torch.utils.cpp_extension.load() or
   tvm_ffi.cpp.load())."

So we submit as ``language = "python"`` and JIT-compile ``kernel.cu`` at
import time with ``tvm_ffi.cpp.build`` + ``-gencode arch=compute_100a,
code=sm_100a``. The kernel.cu source itself is byte-identical to what we
would submit as a ``language = "cuda"`` solution; only the build path is
different. This guarantees sm_100a SASS regardless of what the evaluator's
default SM-detection picks.
"""

from __future__ import annotations

import os
from pathlib import Path

import tvm_ffi

_MOD_DIR = Path(__file__).resolve().parent
_KERNEL_CU = str(_MOD_DIR / "kernel.cu")

_MODULE_NAME = "wombat_dsa_sparse_mla_v5_fused_sm100a"

_EXTRA_CUDA_CFLAGS = [
    "-gencode=arch=compute_100a,code=sm_100a",
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-Xcompiler",
    "-fPIC",
]

_EXTRA_CFLAGS = [
    "-O3",
    "-std=c++17",
    "-fPIC",
]


def _build_and_load():
    """Compile kernel.cu with sm_100a SASS and load the resulting .so.

    Results are cached by tvm_ffi under ``$TVM_FFI_CACHE_DIR`` (defaults to
    ``~/.cache/tvm-ffi``). Repeated imports just re-load the cached .so.
    """
    lib_path = tvm_ffi.cpp.build(
        name=_MODULE_NAME,
        cuda_files=[_KERNEL_CU],
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
        extra_cflags=_EXTRA_CFLAGS,
        extra_include_paths=[str(_MOD_DIR)],
    )
    return tvm_ffi.load_module(lib_path)


_mod = _build_and_load()
_kernel_fn = getattr(_mod, "kernel")


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse):
    """Destination-passing-style entry point used by flashinfer-bench.

    Matches the definition's input order followed by the output order:
      inputs  : (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
      outputs : (out, lse)

    The underlying CUDA kernel writes into ``out`` / ``lse`` in place; this
    function returns ``None`` (DPS contract).
    """
    _kernel_fn(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        float(sm_scale) if not isinstance(sm_scale, float) else sm_scale,
        out,
        lse,
    )
