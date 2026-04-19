"""
DSA sparse MLA: Triton logits (one program per top-k slot), PyTorch softmax / LSE / value.

Semantics: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 (FlashInfer trace definition).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# Triton 3.x: use `tl.constexpr(...)` assignment; annotated globals are not visible in @jit.
D_CKV = tl.constexpr(512)
D_KPE = tl.constexpr(64)
_NEG_INF = tl.constexpr(-1e30)


@triton.jit
def _logits_kernel(
    qn,
    qp,
    kc_flat,
    kp_flat,
    sparse_idx,
    logits_out,
    s_qnt,
    s_qnh,
    s_qnd,
    s_qpt,
    s_qph,
    s_qpd,
    s_kc0,
    s_kp0,
    s_it,
    s_ik,
    s_lot,
    s_loh,
    s_lok,
):
    """One program per (token, head, k): single scalar logit."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    kk = tl.program_id(2)

    qn_row = qn + t * s_qnt + h * s_qnh
    qp_row = qp + t * s_qpt + h * s_qph

    idx = tl.load(sparse_idx + t * s_it + kk * s_ik)
    valid = idx >= 0
    si = tl.where(valid, idx, 0)
    kc_ptr = kc_flat + si * s_kc0
    kp_ptr = kp_flat + si * s_kp0

    acc = 0.0
    for ii in tl.static_range(0, D_CKV):
        qv = tl.load(qn_row + ii * s_qnd).to(tl.float32)
        kv = tl.load(kc_ptr + ii, mask=valid, other=0.0).to(tl.float32)
        acc = acc + qv * kv

    acc2 = 0.0
    for ii in tl.static_range(0, D_KPE):
        qv = tl.load(qp_row + ii * s_qpd).to(tl.float32)
        kv = tl.load(kp_ptr + ii, mask=valid, other=0.0).to(tl.float32)
        acc2 = acc2 + qv * kv

    logit = tl.where(valid, acc + acc2, _NEG_INF)
    tl.store(logits_out + t * s_lot + h * s_loh + kk * s_lok, logit)


def run(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor | float,
):
    device = q_nope.device
    num_tokens, num_heads, d_ckv = q_nope.shape
    topk = sparse_indices.shape[-1]

    if isinstance(sm_scale, torch.Tensor):
        sm = float(sm_scale.item())
    else:
        sm = float(sm_scale)

    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    if sparse_indices.dtype != torch.int32:
        sparse_indices = sparse_indices.to(torch.int32)
    sparse_indices = sparse_indices.contiguous()

    kc_flat = ckv_cache.view(-1, d_ckv)
    kp_flat = kpe_cache.view(-1, q_pe.shape[-1])

    logits = torch.empty((num_tokens, num_heads, topk), dtype=torch.float32, device=device)

    grid = (num_tokens, num_heads, topk)
    _logits_kernel[grid](
        q_nope,
        q_pe,
        kc_flat,
        kp_flat,
        sparse_indices,
        logits,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        kc_flat.stride(0),
        kp_flat.stride(0),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
    )

    logits_scaled = logits * sm
    # [T, K] mask must broadcast to [T, H, K] (head count is workload-dependent, e.g. 8 or 16).
    valid_2d = sparse_indices >= 0
    valid_h = valid_2d.unsqueeze(1)
    logits_masked = logits_scaled.masked_fill(~valid_h, float("-inf"))
    lse = torch.logsumexp(logits_masked, dim=-1) / math.log(2)
    attn = torch.softmax(logits_masked, dim=-1)
    attn = torch.nan_to_num(attn, 0.0)

    idx_safe = sparse_indices.clamp(min=0)
    kc_g = kc_flat[idx_safe]
    output = torch.einsum("thk,tkd->thd", attn, kc_g.float())
    output = output.to(torch.bfloat16)

    empty = ~valid_2d.any(dim=-1)
    output[empty] = 0
    lse[empty] = float("-inf")

    return (output, lse)
