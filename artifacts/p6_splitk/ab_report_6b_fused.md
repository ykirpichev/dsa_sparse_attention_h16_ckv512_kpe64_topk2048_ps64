# Phase 6b — Fused partial+reduce kernel (last-CTA pattern)

**Branch:** `phase5-ws-parissue`
**Date:** 2026-04-23
**Toggle:** `MLA_SPARSE_ATTN_FUSED` (default **OFF**; set `=1` to enable)
**Verdict:** ⚠️ **No measurable win on this bench.** Kept behind an env flag for future reuse. Not dispatched by default.

## Hypothesis

The split-K path currently launches two kernels per call: `mla_split_partial<G_>` followed by `mla_split_reduce<G_>`. On B200 each launch dispatch costs ~0.5–3 µs, so 14/23 workloads on this bench could in theory save ~0.5–3 µs each (1–5 %) if the second launch is eliminated.

## Design

Last-CTA reduce via atomic counter in HBM scratch:

1. Added `int32_t* done` counter to `SplitKScratch`, size `T*H`, zero-initialized on alloc/grow.
2. Every partial CTA executes the normal compute and writes its `(O_g, m_g, l_g)` to HBM. After the writes, it issues `__threadfence()` and then `atomicAdd(&done[t*H+h], 1)`. The CTA that observes `prev+1 == G_` is the "winner".
3. The winning CTA stays alive and performs the reduce inline, reusing the (now unused) `s_qn` SMEM region for `s_m[G_] / s_l[G_]` scratch. It writes directly to `out` / `lse_out`.
4. The winning CTA resets `done[t*H+h] = 0` for the next launch, avoiding a host-side memset. Counter stays at 0 steady-state.
5. Early-exit path (`n_valid_tiles == 0`) takes the same tail after writing its zero partial — so the per-(t,h) arrival count is always exactly G_.

Dispatcher drops the second kernel launch when `done_ptr != nullptr`. The legacy `mla_split_reduce_kernel` is retained as the `MLA_SPARSE_ATTN_FUSED=0` fallback.

Minimal compiler / resource impact:
- `mla_split_partial<4>` regs/thread: **68 → 72** (+4). Still 3 CTAs/SM (69–72 × 256 × 3 = 52–55 K regs < 64 K).
- Static SMEM: 0 → 16 B (one `__shared__ int` for `s_is_last`, plus alignment).
- Dynamic SMEM unchanged: `s_qn` doubles as reduce scratch after compute.

## A/B results (min-of-2 on both sides)

Baseline: **G=16 narrow** (Phase 6a), fused OFF.
Comparison: same narrow-tier G=16, fused ON.

| Workload | G  | min-base | min-fused | Δ |
|----------|----|----------|-----------|---|
| 0c23b10c | 1  | 0.008    | 0.007     | −12.5 %   (G=1 unaffected — timer quantization) |
| 02d6ae9c | 4  | 0.056    | 0.055     |  −1.8 %   |
| 2207f0fd | 4  | 0.054    | 0.054     |  +0.0 %   |
| 232ed014 | 4  | 0.057    | 0.057     |  +0.0 %   |
| 38389961 | 4  | 0.052    | 0.051     |  −1.9 %   |
| 385742b2 | 4  | 0.041    | 0.041     |  +0.0 %   |
| 4c46a94b | 4  | 0.039    | 0.039     |  +0.0 %   |
| 5096e459 | 4  | 0.070    | 0.070     |  +0.0 %   |
| 564007ac | 4  | 0.073    | 0.073     |  +0.0 %   |
| 78b2e11c | 4  | 0.057    | 0.056     |  −1.8 %   |
| 7a389715 | 4  | 0.055    | 0.055     |  +0.0 %   |
| ae4219a9 | 4  | 0.054    | 0.054     |  +0.0 %   |
| **ddfa9e34** | 16 | 0.029 | 0.029 |  +0.0 %   |
| **d57eb9e1** | 16 | 0.035 | 0.037 |  +5.7 %  (noise — d57eb9e1 ranges 0.035–0.042 across runs) |
| 68d6817d | 1  | 0.013    | 0.015     | +15.4 %   (G=1 unaffected — timer quantization) |
| b7668cfd | 1  | 0.010    | 0.011     | +10.0 %   (G=1 unaffected — timer quantization) |
| … 7 other G=1 workloads … | 1 | | | ±0 % |

### Summary

|                                            | min-base | min-fused | Δ       |
|--------------------------------------------|----------|-----------|---------|
| Geomean ALL (n=23)                         | 0.0264   | 0.0265    | +0.46 % |
| Geomean SPLIT only (G=4 or 16, n=13)       | 0.0501   | 0.0501    |  0.00 % |

The per-run CV on unchanged G=1 workloads (those never going through split-K) is ±0–18 %, dominated by 1 µs timer quantization on sub-15-µs kernels. The split-tier deltas on the fused path fall well inside this noise band — **no detectable win or loss**.

## Why the launch-latency savings are below the noise floor

On recent CUDA + B200 drivers, kernel launches are amortized via command-list buffering. Typical GPU-side kernel-to-kernel gap is **<500 ns** on streams with no CPU-side serialization. The reduce kernel itself runs 1 CTA per (t,h), which finishes in ≈ 2 µs on ≥64 SMs simultaneously, dominated by its HBM reads of G partials. Any save we can make on the launch side is squeezed down to ≤500 ns, which is well under the 1-µs quantization of the CUDA event timer on this flashinfer-bench methodology.

## Correctness

All 23 full-bench workloads PASSED with fused ON (smoke + full bench, two independent runs). The last-CTA pattern is correct: `__threadfence()` before the atomic ensures global visibility of partial state, and `atomicAdd`'s acq_rel semantics ensure the winning CTA sees all G payloads.

## Decision

Flip default to **OFF**. Keep code behind `MLA_SPARSE_ATTN_FUSED` env flag for future workloads where:
- Splits are small enough that launch latency matters more (fewer CTAs per launch).
- Driver/runtime has higher per-launch overhead (older CUDA, Windows WDDM, etc.).
- Stream has CPU contention that makes launch pipelining matter.

## Artifacts

- `artifacts/p6_splitk/smoke_fused_20260423T022017.log` — correctness smoke (10 workloads)
- `artifacts/p6_splitk/full_fused_20260423T022121.log` — fused full-bench run 1
- `artifacts/p6_splitk/full_fused_2_20260423T022425.log` — fused full-bench run 2 (stability)
- `artifacts/p6_splitk/verify_fused_off_20260423T022619.log` — default-OFF verification (post-flip)
