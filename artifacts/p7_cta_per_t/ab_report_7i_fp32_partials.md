# Phase 7i — fp32 split-K partials (numerical fix)

## What prompted this work

The Phase 7h final 4-way comparison run produced two *intermittent*
`INCORRECT_NUMERICAL` failures on workloads `7a389715dc…` and
`78b2e11c30…`. Standalone v5_7h runs (5 repeats, 115/115) were clean,
but when `v5_7h` ran after FlashInfer / v4 / v5_fused in the same
session, the abs_err on these two specific workloads would sometimes
reach 0.031 instead of the usual 0.016 — just far enough to trip the
workload tolerance check (`abs > 0.01 AND rel > 0.01` per element).

## Root cause (confirmed)

The tolerance is a **per-element** AND check. Legacy (`v5_fused`) holds
abs_err at **0.0039-0.0078 µ** on these workloads because it runs
without split-K (G=1) and quantizes the output exactly once to bf16.

CPT (`v5_7h`) always runs with split-K (G=16 at T=8). The partial state
`O_partial[T,H,G,D_CK]` was stored as **bf16 in HBM**, and each of the
G partials contributed its own bf16-rounding error. The reducer then
summed G weighted partials, and the worst-case element could land at
0.015-0.031 — occasionally catching an element whose rel_err was also
> 0.01.

```
Legacy    : [fp32 accum] → bf16 out      (1 bf16 quantization)
7h (CPT)  : [fp32 accum] → bf16 partial
            bf16 partial × G=16 → fp32   (G bf16 quantizations,
              → bf16 out                   noise stacks)
```

## Fix

Store the split-K partial `O_partial` in **fp32** instead of bf16. The
fp32 partial path exists solely in HBM; only the final output tensor
is quantized to bf16. Memory cost is trivial: at peak (T=8, G=16)
`T·H·G·D_CK·4B = 4 MiB`, reused across launches. Per-launch HBM traffic
on partials doubles (bf16→fp32), but each partial is written once and
read once by the (same-CTA or neighbour-CTA) reducer — in practice it
all stays resident in L2.

### Code changes (`solution/python/kernel.cu`)

- `struct SplitKScratch::O`  — `void* (bf16)` → `float* (fp32)`.
- `O_bytes` sizing in all 5 launch paths — `sizeof(__nv_bfloat16)` → `sizeof(float)`.
- `fused_reduce_tail<G_>` — parameter type + `__bfloat1622float2(op2[i])` → direct `float2` load.
- `mla_split_reduce_kernel<G_>` — same change (it's the standalone reducer the CPT paths use).
- `mla_split_partial_kernel<G_>` — O_partial parameter + write sites (zero-init + final).
- 4 CPT kernel variants (`scalar`, `th512`, `mma`, `mma_v`) — same
  (parameter + zero-init + final write).

No SMEM layout is affected — partial storage is HBM-only.

## Results

### 1. Standalone accuracy (10 repeats × 23 workloads = 230 samples)

Phase 7h (before fix) — the two flaky workloads:

```
78b2e11c  abs_err samples (5 repeats):  0.0156 ×4  0.0313 ×1
7a389715  abs_err samples (5 repeats):  0.0156 ×4  0.0313 ×1
```

Phase 7i (after fix) — 10 repeats × the two flaky workloads:

```
78b2e11c  abs_err samples (10 repeats):  0.015625 ×10   (stdev 0)
7a389715  abs_err samples (10 repeats):  0.015625 ×10   (stdev 0)
```

**230/230 passed in standalone, with zero abs_err variance.**

### 2. 5-way compare, 5 repeats = 575 samples

```
solution                                    mean(us)   max(us)   gmean(sp)   passed
flashinfer_wrapper_5af199                    260.51    (N/A)       5.88      115/115
wombat_dsa_sparse_mla_cuda_v4                 82.94    136.87     28.60      115/115
wombat_dsa_sparse_mla_cuda_v5_fused           69.77    135.56     33.28      115/115
wombat_dsa_sparse_mla_v5_fused_sm100a         66.76    137.70     34.33      115/115
wombat_dsa_sparse_mla_v5_7i                   33.67     58.40     58.40      115/115
```

All 575 tests passed. **No `INCORRECT_NUMERICAL` anywhere.**

### 3. Perf impact of the fix

Head-to-head (median over 5 repeats), `7h → 7i`:

| metric                 | 7h       | 7i       | Δ      |
| ---------------------- | -------- | -------- | ------ |
| Geomean speedup vs leg | 1.754×   | 1.757×   | +0.2%  |
| Mean wall-time         | 33.57 µs | 33.64 µs | +0.2%  |
| Max wall-time          | 57.85 µs | 59.52 µs | +1.7 µs|

One workload (`564007ac`) picks up ~3 µs. Every other delta is within
±1 µs run-to-run noise. The fix is effectively free.

## Invariants preserved

- Sub-60 µs goal: max wall-time 59.52 µs ≤ 60 µs ✓
- Regression goal: all `385742b2` / `4c46a94b` routed to legacy by the
  density pre-scan, so they match the legacy numbers (≈40 / 42 µs) ✓
- Correctness: 575/575 passing in the 5-way compare context ✓

## Artifacts

- `artifacts/staging/v5_fused_7i.solution.json` — packed solution
  (CPT + density pre-scan + fp32 partials).
- `artifacts/p7_cta_per_t/phase7i/v5_7i_only_10rep_*.json` — 10-repeat
  standalone stability test (230/230 pass, abs_err stdev = 0).
- `artifacts/p7_cta_per_t/phase7i/compare_final_*.json` — 5-way
  compare, 5 repeats, 575/575 pass.
- `artifacts/p7_cta_per_t/phase7i/smoke_*.log` — initial 5-workload
  smoke after the change.
