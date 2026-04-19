"""
DSA sparse MLA: batched PyTorch GEMMs (reference-style), no Triton / FlashInfer runtime.

``Q @ K^T`` over the sparse-gathered K blocks, then softmax / LSE / value projection in fp32.
"""

from __future__ import annotations

import math
from typing import Union

import torch


@torch.no_grad()
def run(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: Union[torch.Tensor, float],
):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    if isinstance(sm_scale, torch.Tensor):
        sm = float(sm_scale.item())
    else:
        sm = float(sm_scale)

    device = q_nope.device

    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    invalid_mask = sparse_indices == -1
    safe_indices = sparse_indices.clone()
    safe_indices[invalid_mask] = 0

    Kc = Kc_all[safe_indices.long()]
    Kp = Kp_all[safe_indices.long()]

    qn = q_nope.to(torch.float32)
    qp = q_pe.to(torch.float32)

    # [T, H, D] @ [T, D, K] -> [T, H, K]
    logits = qn @ Kc.transpose(-1, -2) + qp @ Kp.transpose(-1, -2)
    logits_scaled = logits * sm
    logits_scaled = logits_scaled.masked_fill(invalid_mask.unsqueeze(1), float("-inf"))

    lse = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
    attn = torch.softmax(logits_scaled, dim=-1)
    attn = torch.nan_to_num(attn, 0.0)
    output = (attn @ Kc).to(torch.bfloat16)

    valid_2d = sparse_indices >= 0
    empty = ~valid_2d.any(dim=-1)
    output = torch.where(empty[:, None, None], torch.zeros_like(output), output)
    lse = torch.where(empty[:, None], torch.full_like(lse, float("-inf")), lse)

    return (output, lse)
