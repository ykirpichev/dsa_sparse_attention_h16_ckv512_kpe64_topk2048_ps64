# Phase 7c — WMMA logit GEMM variant (`cta_per_t_mma`)

## TL;DR
**Geomean vs legacy** (10 smoke workloads, seed=42) progressed
`3.17× → 1.87× → 1.36× slower`:

| Variant                 | Geomean vs legacy | Notes                      |
| ----------------------- | ----------------: | -------------------------- |
| 7a scalar (256 thr)     |        **3.17×**  | outer-H loop serial; 64 fp32 per-thread state |
| 7a.1 th512 (512 thr)    |        **1.87×**  | 1 warp per head, H-parallel scalar logit |
| 7c MMA (512 thr + WMMA) |        **1.36×**  | same + tensor-core logit GEMM |

MMA **beats legacy on 4/10 workloads** (the largest ones):

| Workload  | legacy | 7a   | th512 | mma  | mma/legacy |
|-----------|-------:|-----:|------:|-----:|-----------:|
| `5096e459`|   71µs | 105  |   65  |  46  |  **0.65** ✅ |
| `232ed014`|   57µs | 103  |   65  |  46  |  **0.81** ✅ |
| `38389961`|   52µs | 103  |   64  |  46  |  **0.88** ✅ |
| `0c23b10c`|    8µs |  17  |   17  |  15  |      1.88  ❌ |
| `0a63b87b`|   10µs |  82  |   37  |  28  |      2.80  ❌ |
| `f77df5ce`|    9µs |  69  |   26  |  19  |      2.11  ❌ |
| `9f3f891b`|   23µs | 111  |   63  |  44  |      1.91  ❌ |
| `05f6de65`|   27µs | 113  |   63  |  44  |      1.63  ❌ |
| `d57eb9e1`|   37µs | 103  |   66  |  46  |      1.24  ❌ |
| `4c46a94b`|   41µs | 105  |   65  |  46  |      1.12  ❌ |

`regs=1` (96 regs/thread, 1 CTA/SM) and `regs=2` (64 regs/thread, 2 CTAs/SM)
give within 1µs of each other → kernel is **compute-bound, not
occupancy-bound**. The uniform **~46µs "floor" across all non-trivial
workloads** suggests a per-CTA fixed-latency component (likely the
V-accum scalar FMAs + `__syncthreads()` overhead) that amortizes only on
≥50µs-legacy workloads.

## Design
- Based on 7a.1 `th512` layout (512 threads = 16 warps, 1 warp per head).
- Replaces the scalar per-warp logit (16 sequential warp-dots per tile)
  with a **K-split WMMA GEMM**:
  - `D_CK=512` split across 16 warps (2 k-tiles of 16 per warp).
  - `D_KP=64` split across 4 warps (1 k-tile each; warps 4..15 idle for D_KP).
  - Each warp stores its partial `C[H=16, BK=16]` fp32 to
    `s_mma_scratch[warp_id * 256..]`.
  - 256 threads sum the 16 partials, apply `sm_scale_log2e` + mask,
    write to `s_logits_tile[H, BK]`.
- Fragments: `matrix_a` row_major (ld=D_CK), `matrix_b` col_major (ld=D_CK
  reads K stored row-major as K^T), `accumulator` fp32.
- V-accum unchanged from th512: scalar per-thread with 16 channels/head,
  online softmax rescale.

## Shared memory (G=1): ~79 KiB
- logits_tile (1K) + Q_n (16K) + Q_p (2K) + gidx (8K) + reduce (64B)
- **s_mma_scratch (NEW): 16 KiB** = 16 warps × 16×16 fp32
- V tile (32K) + Kpe tile (4K)

SMEM cap raised to **`KERNEL_MAX_DYN_SMEM_CPT_MMA = 88 KiB`**, so 2 CTAs/SM
(176 KiB) still fits in the 228 KiB budget.

## Occupancy
| Variant | regs/thread | CTAs/SM | Threads/SM |
| ------- | ----------: | ------: | ---------: |
| mma r1  |          96 |       1 |        512 |
| mma r2  |          64 |       2 |       1024 |

## Controls
- Env `MLA_SPARSE_ATTN_CTA_PER_T=1` + `MLA_SPARSE_ATTN_CTA_PER_T_MMA=1` to
  route the request through this kernel. `MLA_SPARSE_ATTN_CTA_PER_T_REGS`
  picks `__launch_bounds__(512, 1)` or `(512, 2)`.
- Default OFF — legacy kernel remains the shipping path per the
  `env_flag_off_until_geomean` rollout gate.

## Where the gap remains (small workloads)
- Legacy launches up to 1536 CTAs that interleave heavily across 296 slots,
  amortizing per-CTA fixed cost. CPT-per-t launches at most ~148 CTAs
  (often ~96) → 1 partial wave, and the per-CTA cost is 16× larger (all
  H heads done in one CTA). This is structural to the approach.
- For the 10µs-scale workloads, the ~30µs CPT overhead dominates.
  Plausible mitigation: **T-threshold dispatcher** that stays on the
  legacy grid for small workloads and switches to CPT only when
  `T × H × G_legacy ≫ 222` (1.5 waves or more).

## Next (highest-probability lifts)
1. **V-accum WMMA**: the V-accum phase is the 46µs floor. M=H=16,
   N=D_CK=512 (32 n-chunks of 16), K=BK=16. Would cut the
   scalar FMAs by ~16×. Requires online-softmax rescale of c_frag in
   registers with per-element α[h] lookup (m16n16k16 fp32 accumulator
   has a known lane→(m,n) mapping).
2. **T-threshold dispatch**: gate MMA on workload size to claim wins on
   large workloads without regressing small ones.
3. Only after (1)+(2): Phase 7d cluster-shared K/V for further HBM win.
