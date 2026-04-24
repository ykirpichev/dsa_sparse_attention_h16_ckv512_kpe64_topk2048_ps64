"""Python wrapper that JIT-compiles kernel.cu as a PyTorch extension.

Per ``FAQ.md``:

  "The evaluation machine supports sm_100a. If your CUDA kernel needs
   sm_100a-specific features, specify the architecture explicitly in your
   build flags (e.g., -arch=sm_100a). The build system does not
   automatically set SM targets."

  "The builders currently do not support custom compile flags. As a
   workaround, submit a Python solution and compile the CUDA kernel yourself
   within the code (using torch.utils.cpp_extension.load() or
   tvm_ffi.cpp.load())."

We submit as ``language = "python"`` and JIT-compile ``kernel.cu`` on first
call with ``torch.utils.cpp_extension.load(..., extra_cuda_cflags=["-gencode",
"arch=compute_100a,code=sm_100a", ...])``. This guarantees sm_100a SASS
regardless of what the evaluator's default SM-detection picks, while
matching the Torch-extension path used by the starter-kit ``TorchBuilder``.

The ``kernel.cu`` source itself is identical across TVM-FFI / Torch builds;
only the host-side tensor glue and the module-export macro differ. We ship
the Torch-extension variant (PYBIND11_MODULE + torch::Tensor).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.utils.cpp_extension

_THIS_DIR = Path(__file__).resolve().parent
_KERNEL_CU = str(_THIS_DIR / "kernel.cu")
_MODULE_NAME = "wombat_dsa_sparse_mla_sm100a_final"

# NOTE: `-gencode=arch=compute_100a,code=sm_100a` is REQUIRED here; without
# it nvcc's default SM target on the evaluator host is sm_90 (Hopper), and
# the kernel's sm_100a-only features (TMA gather, fp32 split-K partials,
# cp.async.bulk.tensor.gather4, etc.) fail to assemble.
_EXTRA_CUDA_CFLAGS = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-std=c++17",
    "-gencode", "arch=compute_100a,code=sm_100a",
]

# Link against libcuda.so (driver API) for cuTensorMapEncodeTiled — libcudart
# does NOT expose these symbols.
_EXTRA_LDFLAGS = ["-lcuda"]

_ext = None


def _load_ext():
    """JIT-compile kernel.cu once and cache the loaded extension."""
    global _ext
    if _ext is None:
        _ext = torch.utils.cpp_extension.load(
            name=_MODULE_NAME,
            sources=[_KERNEL_CU],
            extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
            extra_ldflags=_EXTRA_LDFLAGS,
            extra_include_paths=[str(_THIS_DIR)],
            verbose=False,
        )
    return _ext


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse):
    """Destination-passing-style entry point used by flashinfer-bench.

    Matches the definition's input order followed by the output order:
      inputs  : (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
      outputs : (out, lse)

    The underlying CUDA kernel writes into ``out`` / ``lse`` in place; this
    function returns ``None`` (DPS contract).
    """
    _load_ext().run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        float(sm_scale),
        out,
        lse,
    )
