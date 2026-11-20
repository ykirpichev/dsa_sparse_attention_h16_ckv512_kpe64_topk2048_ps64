"""DSA sparse MLA, ThunderKittens implementation.

The score is kittens::mm_ABt, the Blackwell tcgen05 MMA: a 128-key tile by a
64-wide chunk of the 512-d key, into FP32 tensor memory. The KV rows are
gathered into the swizzled shared tile that descriptor reads. The softmax,
the output mix, and the merge of partials are ordinary CUDA. ThunderKittens
has no sparse-gather or online-softmax primitive; the indexer port makes the
same split.

ThunderKittens is not part of the contest image. The extension looks for
headers in /opt/ThunderKittens and compiles for sm_100a.
"""

import os
from pathlib import Path

import torch

_EXT = None


def _load():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    _EXT = load(
        name="dsa_tk_sparse_mla",
        sources=[str(here / "attention.cu")],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "-DKITTENS_SM100",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-I/opt/ThunderKittens/include",
            "-I/opt/ThunderKittens/prototype",
        ],
        verbose=False,
    )
    return _EXT


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, out, lse):
    _load().run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, float(sm_scale), out, lse)
