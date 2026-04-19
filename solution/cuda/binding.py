"""
TVM FFI bindings for the CUDA MLA kernel (starter-kit pattern).

Loads ``mla_cuda.cu`` via ``torch.utils.cpp_extension.load`` (PyTorch CUDA extension with pybind11 ``run``).
Registers the entry point as ``flashinfer.kernel`` for TVM FFI tooling.

See: https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/solution/cuda/binding.py
"""

from __future__ import annotations

import ctypes
import os

import torch
from torch.utils.cpp_extension import load

try:
    from tvm.ffi import register_func
except ImportError:  # pragma: no cover — evaluation / Modal image provides tvm.ffi
    try:
        from tvm_ffi import register_func  # type: ignore[no-redef,import-not-found]
    except ImportError:

        def register_func(name):  # type: ignore[misc]
            def _decorator(fn):
                return fn

            return _decorator

_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="mla_cuda_bench",
            sources=[os.path.join(here, "mla_cuda.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


@register_func("flashinfer.kernel")
def kernel(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale,
):
    """Sparse MLA forward; delegates to compiled ``mla_cuda.cu``."""
    return _load_ext().run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
    )


# FlashInfer-Bench entry symbol (config ``entry_point = "binding.py::run"``)
run = kernel
