"""
TEMPORARY FlashInfer wrapper for DSA sparse MLA using
``flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla``.

Not a submission candidate — used only to measure how fast the FlashInfer TRT-LLM-gen
MLA path runs on the contest traces so we can compare against our custom CUDA kernel
in ``solution/cuda/mla_cuda.cu``.

Pack / run (on Modal B200) via the alternate TOML::

    FIB_CONFIG_TOML=config.flashinfer_trtllm.toml modal run scripts/run_modal.py::main --no-smoke
"""

from __future__ import annotations

import torch
import flashinfer.decode

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache: dict[str, torch.Tensor] = {}

QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TOPK = 2048


def _get_workspace(device: torch.device) -> torch.Tensor:
    key = str(device)
    buf = _workspace_cache.get(key)
    if buf is None:
        buf = torch.zeros(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buf
    return buf


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    device = q_nope.device

    if isinstance(sm_scale, torch.Tensor):
        bmm1_scale = float(sm_scale.item())
    else:
        bmm1_scale = float(sm_scale)

    query = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)
    kv_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)
    block_tables = sparse_indices.unsqueeze(1)

    seq_lens = (sparse_indices != -1).sum(dim=1).to(torch.int32)
    max_seq_len = int(seq_lens.max().item())
    workspace = _get_workspace(device)

    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=TOPK,
        bmm1_scale=bmm1_scale,
    )
    output = output.squeeze(1)

    return (output,)
