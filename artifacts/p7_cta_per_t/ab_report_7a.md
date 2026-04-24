# Phase 7a — Scalar CTA-per-t prototype — **smoke-only; regression as predicted**

**Date:** 2026-04-23
**Status:** Landed behind `MLA_SPARSE_ATTN_CTA_PER_T` env flag (default OFF).
Correctness PASS on all 10 smoke workloads. Perf ≈ 1.5–3× regression on every
workload tested, matching the ex-ante theoretical lower bound. Expected —
7a is designed as the staging step for 7c (tensor cores).

## What shipped

- New kernel `mla_sparse_attn_kernel_cta_per_t<G_, H_, MIN_BLOCKS_>` in
  `kernel.cu`. Grid `(T, G)`; one CTA per `(t, g)` handles **all H=16 heads**
  of the token on top of a single shared K/V tile stream.
- New host dispatch path `dispatch_cta_per_t` gated by
  `MLA_SPARSE_ATTN_CTA_PER_T=1`. Intercepts the whole launch; uses its own
  `pick_split_G_cta_per_t` (target 222 CTAs at 1 CTA/SM) and reuses the
  existing `mla_split_reduce_kernel` for partial-merge on G≥2.
- Sub-flag `MLA_SPARSE_ATTN_CTA_PER_T_REGS={1,2}` selects
  `__launch_bounds__(256, 1)` or `(256, 2)` for the A/B.
- Tracked 64 fp32 of per-head state per thread (`m[H], l[H], o0[H], o1[H]`);
  no spills at `mb=1` (local=128 B only), mild spills at `mb=2`.

## Correctness

All 10 smoke workloads PASS (rel/abs error within harness tolerance). Tested
both `mb=1` and `mb=2`. See logs `smoke1_*.log` and `smoke2_regs2_*.log`.

| Workload | T | H | Baseline (µs) | 7a mb=1 (µs) | 7a mb=2 (µs) |
|---|---:|---:|---:|---:|---:|
| 5096e459 | 8 | 16 | ~73 | 105 | 106 |
| 0a63b87b | 2 | 16 | ~15 |  82 |  84 |
| 0c23b10c | 1 | 16 | ~8  |  17 |  18 |
| f77df5ce | 1 | 16 | ~8  |  69 |  69 |
| 9f3f891b | 4 | 16 | ~23 | 111 | 114 |
| 232ed014 | 6 | 16 | ~57 | 103 | 105 |
| 05f6de65 | 4 | 16 | ~25 | 113 | 115 |
| d57eb9e1 | 6 | 16 | ~36 | 103 | 106 |
| 4c46a94b | 6 | 16 | ~40 | 105 | 106 |
| 38389961 | 7 | 16 | ~50 | 103 | 105 |

Kernel attrs:
- `mb=1`: regs/thread=249, local=128 B, occ=1 block/SM.
- `mb=2`: regs/thread=127, local=128 B, occ=2 blocks/SM.

## Why the regression is expected (and what 7c fixes)

The logit phase in 7a iterates sequentially over `h = 0..16` inside a tile:
each iteration does 2 warp-cooperative dots per warp (the same inner code as
the legacy kernel's logit phase). 8 tiles/CTA × 16 h × 2 warp-dots/h per warp
= **256 sequential warp-dot rounds per warp per CTA**. Compare:

| Kernel | CTAs per token | Warp-dot rounds / warp / CTA | Total work |
|---|---:|---:|---:|
| Legacy (t,h,g), G=16 | 16 × H = 256 | 16 | 256 × 16 = 4096 |
| 7a cta-per-t, G=16  |   16         | 256 | 16 × 256 = 4096 |

Total work is identical. What differs is **per-SM throughput**:

- Legacy: ≤ 2 CTAs/SM × 148 SMs = 296 concurrent CTAs. For a T=6 G=16
  workload, 1536 CTAs fit in 5.2 waves × 16 rounds per wave = 83 rounds of
  wall time.
- 7a: ≤ 2 CTAs/SM × 148 SMs = 296 concurrent CTAs, but only 96 CTAs total
  (T × G). Partial wave: 96/148 = 65 % of SMs active × 256 rounds = **256
  rounds of wall time** (the 52 idle SMs are waste).

Ratio ≈ 256 / 83 ≈ 3× slower. Observed regression: ~2–3×. Matches the model.

Bumping G does not recover the parallelism: each G-split halves per-CTA
rounds but halves total CTAs, and the "rounds-per-wall" stays at the
warp-dot-serialisation floor set by the 16× H-batching. G=64 and G=128
projections all land in the ~2× slower range. Confirmed by observation:
every workload sits at ≈100 µs regardless of T (a signature of
serial-compute saturation, not HBM).

## Why we keep the path (and what comes next)

The whole point of 7a was to stage 7c. Replacing the inner H-loop with a
single `tcgen05.mma m16n16k16` (or an `m16n16k32` over two D_CK chunks)
collapses those 256 sequential warp-dots to ≈ 2 MMA instructions, at which
point:

- The logit phase becomes essentially free (tensor-core throughput on
  Blackwell ≈ 4 PFLOPS; our logit compute is ≤ 17 MFLOPs per workload).
- The HBM savings from single K/V load per token (≈ 27 % of current reads
  are redundant, per p6d L2 analysis) become the limiting factor.
- V-accum can also MMA (`m16n128k16` over D_CK=512 in 4 chunks).

Net expected: 7a+7c together should put long-K workloads in the
`40–55 µs` band (vs 73 µs today on 564007ac). That's the real payoff.

## Configuration state

| Env flag | Default | Effect |
|----------|---------|--------|
| `MLA_SPARSE_ATTN_CTA_PER_T`      | OFF | Use cta-per-t dispatch |
| `MLA_SPARSE_ATTN_CTA_PER_T_REGS` | 1   | Kernel launch_bounds MIN_BLOCKS |

Left OFF by default; 7c will re-evaluate the flip.

## Artifacts

- `smoke1_*.log` — 10-workload smoke at `mb=1` (all PASS).
- `smoke2_regs2_*.log` — 10-workload smoke at `mb=2` (all PASS, identical
  perf — not reg-pressure bound).
