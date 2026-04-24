# Phase 7c-next — V-accum WMMA (`cta_per_t_mma_v`)

## TL;DR
**No perf change vs `mma`**; correctness passes. The ~46 µs CPT floor is
NOT dominated by the scalar V-accum as hypothesized.

| Variant                 | Geomean vs legacy | Smoke 46 µs-floor |
| ----------------------- | ----------------: | ----------------: |
| 7c MMA (WMMA logit)     |             1.36× |            46 µs  |
| 7c-next MMA-V (WMMA V)  |          **1.38×**|         **46 µs** |

Smoke (10 workloads, seed=42, 1/1/1):

| Workload  | legacy | mma  | mma_v | mma_v / legacy |
|-----------|-------:|-----:|------:|---------------:|
| `5096e459`|   71µs |  46  |  48   |     0.68 ✅   |
| `232ed014`|   57µs |  46  |  48   |     0.84 ✅   |
| `38389961`|   52µs |  46  |  46   |     0.88 ✅   |
| `0c23b10c`|    8µs |  15  |  15   |     1.88 ❌   |
| `0a63b87b`|   10µs |  28  |  30   |     3.00 ❌   |
| `f77df5ce`|    9µs |  19  |  21   |     2.33 ❌   |
| `9f3f891b`|   23µs |  44  |  46   |     2.00 ❌   |
| `05f6de65`|   27µs |  44  |  46   |     1.70 ❌   |
| `d57eb9e1`|   37µs |  46  |  46   |     1.24 ❌   |
| `4c46a94b`|   41µs |  46  |  46   |     1.12 ❌   |

## Design

Building on 7c `mma`:

- **Logit phase:** unchanged (K-split WMMA).
- **Softmax:** lanes 0..15 of each warp compute `p = exp2(logit - m_new)`
  in parallel for the warp's head, writing bf16 to
  `s_p_tile[H=16, BK=16]`. This eliminates the 128 redundant `exp2f`
  calls per thread per CTA (→ 16× SFU reduction).
- **Rescale:** per-head `α[h]` rescale of the fp32 accumulator fragment
  uses a store→per-element-mul→load path through `s_O_scratch`. Each
  warp owns 2 `m16n16k16` fragments (total 32 output channels of
  D_CK=512, across all H=16 heads per warp).
- **V-accum:** 2 `mma.m16n16k16 bf16→fp32` per warp (`P @ V_chunk`).

**Persistent accumulator:** `c_o[2]` fragments held in registers across
tiles; store/load only for rescale.

### SMEM overlay
`s_mma_scratch` (16 KiB, live during logit phase) and `s_O_scratch`
(32 KiB, live during rescale / final-write) alias the same backing
region. Safe because:
- At tile i start, O lives in registers (c_frag). s_O_scratch is dead.
- Logit MMA of tile i writes s_mma_scratch (phase 2). s_O_scratch aliases
  clobbered; OK — not used until phase 5.
- Rescale writes s_O_scratch, aliasing s_mma_scratch. OK — phase 2 done.
- V-accum reads c_frag only; next iter's logit MMA writes the alias
  after a `__syncthreads()`.

Net SMEM increase vs `mma`: +16 KiB (max(mma_scratch, O_scratch) − mma_scratch)
+ 0.5 KiB s_p_tile + 0.06 KiB s_alpha ≈ **+16.5 KiB** → total ~96 KiB.
Cap `KERNEL_MAX_DYN_SMEM_CPT_MMA_V = 100 KiB`; 2 CTAs/SM still fits
(200 KiB < 228 KiB budget).

## Occupancy
| Variant | regs/thread | SMEM/CTA | CTAs/SM |
|---------|------------:|---------:|--------:|
| mma r1  |          96 |    79 KB |       1 |
| mma_v r1|          92 |    96 KB |       1 |

## Why no speedup?

The hypothesis was that ~46 µs is V-accum-bound (scalar FMAs + redundant
`exp2f`). Empirically **replacing both with MMA had zero effect**. The
floor persists to 46 µs.

Back-of-envelope accounting per CTA (8 tiles, G=16, T∈{small}):
- Logit MMA (warp-parallel, 2-3 × 16-cycle mmas):  < 0.5 µs
- Logit reduce + mask:                              < 0.1 µs
- Softmax (per-warp, 16 exp2f on lanes 0..15):      < 0.1 µs
- Rescale (per-warp, 16 × FMUL):                    < 0.1 µs
- V-accum MMA (2 mmas/warp):                        < 0.5 µs
- HBM traffic (144 KB/CTA @ 54 GB/SM-eff):           ≈ 2.7 µs
- `__syncthreads()` × 3 × 8 tiles:                    ≈ 0.4 µs

Sum: **≈ 4 µs** active. Actual: **≈ 46 µs**. Discrepancy: **≈ 42 µs
unaccounted**.

The remaining time is almost certainly **instruction-issue starvation**:
at 1 CTA/SM with 16 warps/SM, each warp gets 1/4 of the 4 issue slots
per cycle (B200 has 4 partitions × 1 issue/cycle = 4 warp-issues/cycle,
but only active warps contend). For a warp with ~500 non-MMA
instructions per tile × 8 tiles = 4000 instructions, effective wall is
4000 × 4 / (2 GHz) ≈ 8 µs. Plus pipeline latencies and SMEM bank traffic
amplified by sparse memory patterns.

The only path to pushing the floor below ~20 µs per CTA requires **2+
CTAs/SM and/or wider warps**, i.e. a genuine occupancy-lift. That is
blocked by the 32 KiB `s_O_scratch` regardless of `regs=2` setting.

## What changes (code)

- `mla_sparse_attn_kernel_cta_per_t_mma_v` (new, ~200 LOC).
- `configure_cpt_mma_v_kernel_attrs_once`, `launch_cta_per_t_mma_v_mb`,
  `launch_cta_per_t_mma_v` (new launchers).
- `dispatch_cta_per_t` routes to `_mma_v` when
  `MLA_SPARSE_ATTN_CTA_PER_T_MMA_V=1`.
- `KERNEL_MAX_DYN_SMEM_CPT_MMA_V = 100 * 1024`.
- `cta_per_t_mma_v_enabled()` env gate.
- `scripts/run_modal.py` exposes `--cta-per-t-mma-v`.

## Conclusion

The V-accum MMA reduces the theoretical compute cost of the softmax +
V-accum phase by ~10×, but the **observed wall time does not change**.
This rules out V-accum / `exp2f` as the dominant cost of the 46 µs floor
and shifts the bottleneck hypothesis to **issue-slot starvation under
1 CTA/SM occupancy** — which is structural to CPT-per-t given the Q+V
SMEM footprint.

Next levers:
1. Reduce SMEM footprint enough to fit 2 CTAs/SM (hard: Q alone is 18 KB,
   V pipe is 32 KB).
2. Keep CPT off by default; use T-threshold dispatch to route only
   large-T workloads to CPT.
3. Full Phase 7d rearchitecture (CTA-per-t with cluster-shared K/V and
   H-split across cluster members) to finally drop SMEM per-CTA and
   hit 2-4 CTAs/SM.

Kept **`mma_v` as experimental** behind `MLA_SPARSE_ATTN_CTA_PER_T_MMA_V=1`
for reproducibility and future H-split-cluster experiments.
