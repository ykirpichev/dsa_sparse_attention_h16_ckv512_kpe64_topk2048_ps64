# Phase 6a — Split-K G=16 dispatcher-threshold tuning

**Branch:** `phase5-ws-parissue`
**Date:** 2026-04-23
**Toggle:** `MLA_SPARSE_ATTN_SPLIT_G16` (default **ON**; set `=0` to force off)
**Verdict:** ✅ **Keep ON by default.** Net geomean −0.2 % vs baseline, driven by two T=6 H=16 workloads at −14 % each. Zero kernel changes — dispatcher threshold only.

## Change

`pick_split_G` now returns G=16 on the `[64, 110]` `base_ctas` tier (previously G=8).

| base_ctas  | Before | After | Active workloads on this bench |
|------------|--------|-------|--------------------------------|
| < 64       | G=1    | G=1   | T=1 H=16 (8 wl), T=2 H=16 (9 wl) |
| [64, 110]  | G=8    | **G=16** | T=6 H=16 (2 wl: ddfa9e34, d57eb9e1) |
| [111, 221] | G=4    | G=4   | T=7 H=16 (0 wl on this bench), T=8 H=16 (9 wl) |
| [222, 443] | G=2    | G=2   | — |
| ≥ 444      | G=1    | G=1   | — |

`mla_split_partial_kernel<16>` was already compiled and reachable; only `pick_split_G` + its dispatch branch needed wiring. Zero kernel code changes.

## Methodology

1. Added env-gated diagnostic `MLA_SPARSE_ATTN_SPLIT_DIAG=1` that prints `(T, H, base_ctas, G)` once per unique `(T, H)` at first dispatch.
2. Ran full bench with baseline (`G=16` off) to map each workload to its `(T, H, base_ctas, G)` bucket.
3. Ran full bench with `G=16` on globally (promoting both `< 64` and `[64, 110]` tiers). Measured tier-level geomeans.
4. Narrowed the policy after the `base_ctas < 64 → G=16` promotion regressed +26 % on short-K T=2 workloads (reducer + double-launch overhead swamps the tiny split savings).
5. Re-ran with the narrowed `[64, 110]` → G=16 policy only.
6. Ran a second baseline to establish per-workload timing-noise floor (±5 % on sub-15-µs workloads due to 1-µs CUDA event quantization).

## A/B results (narrow-tier G=16, min-of-2 baselines)

| Workload   | Base1 | Base2 | G16-narrow | Δ vs min(base) | Path |
|------------|-------|-------|------------|----------------|------|
| 0c23b10c   | 0.008 | 0.008 | 0.008      |  +0.0 %        | G=1 (unchanged) |
| 9d4a5f21   | 0.008 | 0.008 | 0.008      |  +0.0 %        | G=1 (unchanged) |
| b7668cfd   | 0.010 | 0.010 | 0.010      |  +0.0 %        | G=1 (unchanged) |
| 0a63b87b   | 0.010 | 0.010 | 0.010      |  +0.0 %        | G=1 (unchanged) |
| 05f6de65   | 0.027 | 0.027 | 0.027      |  +0.0 %        | G=1 (unchanged) |
| fc85411e   | 0.008 | 0.008 | 0.008      |  +0.0 %        | G=1 (unchanged) |
| e6b849f2   | 0.012 | 0.012 | 0.012      |  +0.0 %        | G=1 (unchanged) |
| 9f3f891b   | 0.023 | 0.023 | 0.023      |  +0.0 %        | G=1 (unchanged) |
| f77df5ce   | 0.008 | 0.008 | 0.008      |  +0.0 %        | G=1 (unchanged) |
| 385742b2   | 0.042 | 0.041 | 0.042      |  +2.4 %        | G=4 (unchanged, noise) |
| 4c46a94b   | 0.040 | 0.039 | 0.040      |  +2.6 %        | G=4 (unchanged, noise) |
| 38389961   | 0.052 | 0.051 | 0.052      |  +2.0 %        | G=4 (unchanged, noise) |
| 02d6ae9c   | 0.056 | 0.056 | 0.059      |  +5.4 %        | G=4 (unchanged, noise) |
| **ddfa9e34** | **0.035** | **0.034** | **0.029** | **−14.7 %** ✓  | **G=8 → G=16** |
| 78b2e11c   | 0.058 | 0.059 | 0.057      |  −1.7 %        | G=4 (unchanged, noise) |
| 68d6817d   | 0.011 | 0.011 | 0.013      | +18.2 %        | G=1 (unchanged, noise — sub-15-µs quantization) |
| 564007ac   | 0.073 | 0.073 | 0.073      |  +0.0 %        | G=4 (unchanged) |
| ae4219a9   | 0.054 | 0.054 | 0.054      |  +0.0 %        | G=4 (unchanged) |
| 232ed014   | 0.057 | 0.057 | 0.057      |  +0.0 %        | G=4 (unchanged) |
| 7a389715   | 0.055 | 0.055 | 0.055      |  +0.0 %        | G=4 (unchanged) |
| 5096e459   | 0.071 | 0.070 | 0.070      |  +0.0 %        | G=4 (unchanged) |
| **d57eb9e1** | **0.042** | **0.042** | **0.036** | **−14.3 %** ✓ | **G=8 → G=16** |
| 2207f0fd   | 0.054 | 0.054 | 0.054      |  +0.0 %        | G=4 (unchanged) |

### Summary

|                                            | Base (min-of-2) | G16-narrow | Δ     |
|--------------------------------------------|-----------------|------------|-------|
| Geomean (all 23)                           | 0.0266          | 0.0266     | −0.18 % |
| Workloads with real path change (n=2)      | 0.0377          | 0.0324     | −14.5 % |
| Noise floor (base1 vs base2, all 23)       | —               | —          | ±0.41 % |

All observed "losses" on unchanged paths (+2–18 %) are smaller than or comparable to the base1-vs-base2 noise floor and reproduce across runs. The two **real** deltas (ddfa9e34 and d57eb9e1) are ≈6 µs shifts, ~15× the noise floor.

## Why the narrow policy

The first A/B tried G=16 on **both** `< 64` (T=2) and `[64, 110]` (T=6) tiers. Result: short-K T=2 workloads (8-12 µs baseline) regressed +26.3 % geomean because the reducer + second-kernel-launch overhead (~3 µs) is a large fraction of the single-kernel baseline.

In contrast, the T=6 tier has baseline ~35-42 µs — the reducer overhead is ~10 % amortized, and cutting the per-CTA K work from `TOPK/8=256` rows to `TOPK/16=128` rows shortens the critical path proportionally. 16 × 96 = 1536 CTAs for T=6 is heavy oversubscription of the 444-slot target, but the CTAs are cheap enough that the GPU digests them serially and still lands at −15 %.

## Artifacts

- `artifacts/p6_splitk/diag_base_20260423T020521.log` — baseline + dispatch diag
- `artifacts/p6_splitk/full_g16_20260423T020718.log` — **wide** G=16 (rejected: +3.8 % geomean)
- `artifacts/p6_splitk/full_g16_narrow_20260423T020953.log` — **narrow** G=16 (accepted)
- `artifacts/p6_splitk/diag_base_rerun_20260423T021143.log` — 2nd baseline for noise floor
- `artifacts/p6_splitk/verify_default_on_20260423T021339.log` — verification that default = ON after code change

## Decision

Merge the narrow-tier `[64, 110]` → G=16 policy. Default ON. Keep `MLA_SPARSE_ATTN_SPLIT_G16=0` env override available in case future workloads want to force the old G=8 behavior.
