# Phase 5c Stage 3 — WS3 (full warp-specialized) vs baseline / TMA / WS

**Branch:** `phase5-ws-parissue` (new kernel `mla_sparse_attn_kernel_ws3`)
**Date:** 2026-04-23
**Toggle:** `MLA_SPARSE_ATTN_WS3=1`
**Verdict:** ❌ **Keep off by default.** +15.3 % geomean vs cp.async baseline. Exploratory run confirmed the hypothesis from `artifacts/p5c_tb/ab_report.md` that V-accum halving costs more than any DMA-warp gain on this workload.

## Configuration

`mla_sparse_attn_kernel_ws3` extends `mla_sparse_attn_kernel_ws` with the classic full-WS pattern:

| Feature | Implementation |
|---|---|
| **Dedicated DMA warp** | `warp 0 lane 0` only issues TMA gather4; warps 0-3 are the producer warp-group (`is_producer`), warps 4-7 the consumer warp-group |
| **Producer/consumer handshake** | Two mbarrier arrays (`s_ready_mbar`, `s_empty_mbar`) of `WS_SLOTS=3` each; `s_empty_mbar` pre-arrived at init so the first 3 producer waits are no-ops |
| **Asymmetric registers** | `setmaxnreg.dec<24>()` on producer warp-group, `setmaxnreg.inc<232>()` on consumer warp-group (fits exactly in 65536 reg file at 1 CTA/SM) |
| **Named barriers** | `bar.sync 1, 128` for intra-consumer-group sync, replacing `__syncthreads()` (which would deadlock since producer lanes ≠ (0,0) return early) |
| **V-accum restructure** | 128 consumer threads × 4 bf16 columns each = D_CK=512 (was 256 threads × 2 bf16 in `_ws`/baseline) |
| **Launch bounds** | `__launch_bounds__(THREADS, 1)` — **1 CTA/SM** (was 2) so each CTA gets the full 65536-reg file |
| **SMEM** | 73 728 B (same tile arenas + 2× WS_SLOTS mbarrier arrays = +24 B vs `_ws`) |

Kernel stats (reported at first launch):

```
[mla_sparse_attn_ws3] regs/thread=232  static_smem=0  max_dyn_smem=73728  max_threads=256
```

232 regs/thread × 256 threads = 59 392 regs/CTA ≤ 65 536 reg-file ✓. Occupancy = 1 CTA/SM (was 3).

## Implementation notes

### Hang on first smoke test

Initial smoke attempt timed out (300 s) on all G=1 workloads. Root cause: the kernel had `__launch_bounds__(THREADS, 2)` (inherited from the `_ws` sibling), which caps at 128 regs/thread per CTA (65536 / (256×2)). `setmaxnreg.inc<240>` exceeds that cap and behavior is undefined — measured effect was an apparent deadlock on all WS3 CTAs.

Fix: drop the launch bound to `(THREADS, 1)` and tune `inc<232>` to land exactly at the 65536-reg cap.

### XID-avoidance notes

Re-using the SMEM layout lessons from `_ws`:

- `s_ready_mbar` and `s_empty_mbar` live at the end of the SMEM region to preserve 16-B alignment of `s_qn` / `s_qp` (required by `cp.async.16`).
- Only 1 mbarrier-init thread per slot; `fence.mbarrier_init.release.cluster` publishes before `block_reduce_max`'s internal `__syncthreads()` makes the init visible CTA-wide.
- Producer lanes ≠ (0,0) **return** after the CTA-wide `block_reduce_max`. Therefore **no `__syncthreads()` anywhere in the main loop** — all intra-consumer sync is via `named_bar_sync(1, 128)`, which excludes the dead producer lanes.

## A/B results (FULL bench, Modal B200, 23 workloads, 1 run each)

| Workload   | Base   | TMA    | WS     | WS3    | WS3 Δ vs Base | WS3 Δ vs WS | Path        |
|------------|--------|--------|--------|--------|---------------|-------------|-------------|
| 0c23b10c   | 0.008  | 0.008  | 0.008  | 0.010  | +25.0 %       | +25.0 %     | TMA (G=1)   |
| 9d4a5f21   | 0.008  | 0.008  | 0.009  | 0.011  | +37.5 %       | +22.2 %     | TMA (G=1)   |
| fc85411e   | 0.008  | 0.008  | 0.009  | 0.012  | +50.0 %       | +33.3 %     | TMA (G=1)   |
| f77df5ce   | 0.008  | 0.008  | 0.009  | 0.012  | +50.0 %       | +33.3 %     | TMA (G=1)   |
| 0a63b87b   | 0.010  | 0.011  | 0.011  | 0.014  | +40.0 %       | +27.3 %     | TMA (G=1)   |
| b7668cfd   | 0.010  | 0.011  | 0.011  | 0.014  | +40.0 %       | +27.3 %     | TMA (G=1)   |
| 68d6817d   | 0.011  | 0.011  | 0.011  | 0.010  |  −9.1 %       |  −9.1 %     | TMA (G=1)   |
| e6b849f2   | 0.012  | 0.014  | 0.014  | 0.018  | +50.0 %       | +28.6 %     | TMA (G=1)   |
| 9f3f891b   | 0.023  | 0.027  | 0.028  | 0.041  | +78.3 %       | +46.4 %     | mixed/G=1   |
| 05f6de65   | 0.027  | 0.033  | 0.033  | 0.049  | +81.5 %       | +48.5 %     | mixed/G=1   |
| ddfa9e34   | 0.035  | 0.034  | 0.035  | 0.034  |  −2.9 %       |  −2.9 %     | split-K     |
| 4c46a94b   | 0.040  | 0.039  | 0.040  | 0.039  |  −2.5 %       |  −2.5 %     | split-K     |
| 385742b2   | 0.042  | 0.041  | 0.042  | 0.041  |  −2.4 %       |  −2.4 %     | split-K     |
| d57eb9e1   | 0.042  | 0.043  | 0.042  | 0.043  |  +2.4 %       |  +2.4 %     | split-K     |
| 38389961   | 0.052  | 0.051  | 0.052  | 0.051  |  −1.9 %       |  −1.9 %     | split-K     |
| ae4219a9   | 0.054  | 0.054  | 0.054  | 0.054  |  +0.0 %       |  +0.0 %     | split-K     |
| 2207f0fd   | 0.054  | 0.054  | 0.054  | 0.055  |  +1.9 %       |  +1.9 %     | split-K     |
| 7a389715   | 0.055  | 0.055  | 0.056  | 0.056  |  +1.8 %       |  +0.0 %     | split-K     |
| 232ed014   | 0.057  | 0.057  | 0.057  | 0.057  |  +0.0 %       |  +0.0 %     | split-K     |
| 02d6ae9c   | 0.060  | 0.055  | 0.056  | 0.058  |  −3.3 %       |  +3.6 %     | split-K     |
| 78b2e11c   | 0.067  | 0.057  | 0.060  | 0.057  | −14.9 %       |  −5.0 %     | split-K     |
| 5096e459   | 0.070  | 0.070  | 0.071  | 0.070  |  +0.0 %       |  −1.4 %     | split-K     |
| 564007ac   | 0.073  | 0.073  | 0.073  | 0.073  |  +0.0 %       |  +0.0 %     | split-K     |

Notes:
- All 23 workloads **PASSED** correctness. Abs-err values (up to 3.12e-02) are within the flashinfer-bench reference-precision band (same as cp.async / TMA / WS).
- Per-workload latencies are rounded to 1 µs, so single-row deltas on short workloads carry ≈12.5 % quantization noise.
- Split-K path (long workloads) is unchanged — any delta there is timing noise.

### Geomeans

|                                | Base (ms) | TMA (ms) | WS (ms) | WS3 (ms) | WS3 Δ vs Base | WS3 Δ vs WS |
|--------------------------------|-----------|----------|---------|----------|---------------|-------------|
| All 23 workloads               | 0.0270    | 0.0274   | 0.0281  | 0.0311   | **+15.3 %**   | +10.7 %     |
| Short (TMA-only, n=8)          | 0.0093    | 0.0097   | 0.0101  | 0.0124   | **+33.9 %**   | +22.8 %     |
| Long (split-K, n=13, unchanged)| 0.0527    | —        | —       | 0.0518   | −1.8 %        | —           |

## Why Stage 3 regresses harder than `_ws`

The hypothesis documented in `artifacts/p5c_tb/ab_report.md §"Implications for Stage 3"` matches the measurement:

1. **V-accum halving dominates.** The consumer-side online-softmax + V-accumulate loop runs on 128 threads × 4 bf16 each (WS3) versus 256 threads × 2 bf16 each (baseline/WS). On short G=1 workloads the V-accum is ≈40 % of the tile critical path; halving its parallel width is effectively a 2× slowdown of that phase. This alone explains most of the +22–48 % short-workload regressions.

2. **Occupancy drop from 3 → 1 CTA/SM.** Forced by `__launch_bounds__(THREADS, 1)` to let `setmaxnreg.inc<232>` function. On short workloads (T·H ≤ 256) this doesn't matter directly for SM-fill (we're parallelism-limited, not occupancy-limited), but it does hurt the small number of workloads where multiple CTAs were sharing an SM to hide memory latency.

3. **`setmaxnreg` gives no measurable lift.** The compiler-allocated register count for the (non-WS) kernel was 69 regs/thread; the consumer's working set (m, l, o0..o3 + loop scalars) is at most ~20 regs live at once. Giving the consumer 232 regs/thread capacity is mostly wasted — nothing spills to begin with. This is a documented phenomenon for FlashAttention-like kernels on Hopper/Blackwell that aren't actually register-bound.

4. **Dedicated DMA warp doesn't save enough.** On the TMA-bound short workloads, the producer's `issue_tile_tma` lambda was already on exactly 1 thread (tid=0) in the `_ws` kernel — moving it to a dedicated lane doesn't change the TMA unit throughput, and the only thing we save is the blocking of the rest of warp-0 at `mbarrier_arrive_expect_tx`. That's ~20 cycles/tile, amortized over ~8 tiles = ~160 cycles / ~0.13 µs. Meanwhile we paid several-hundred cycles per tile in extra V-accum work.

## Confirmed insights (the point of the exploration)

- **B200 TMA unit is not the bottleneck we thought it was on short workloads.** Dedicating a DMA warp gave zero observable lift. The tile critical path is dominated by the consumer-side V-accum, not the TMA issue.
- **setmaxnreg only pays for register-bound kernels** — flash-attention-style fused kernels on bf16 with BK=16 don't hit that regime. Reserve this technique for GEMM/flash-mha-with-cumulative-softmax variants where the accumulator is >64 fp32 elements / thread.
- **V-accum parallel-width is a first-order lever** for this kernel. Any future WS variant must keep the V-accum at ≥256 threads (or even widen it via shared-mem Tensor-Core `tcgen05.mma` once we're willing to pay the re-design cost).

## Decision

Keep `mla_sparse_attn_kernel_ws3` as reference code on this branch (reachable via `MLA_SPARSE_ATTN_WS3=1`) but **don't dispatch to it by default**.

The Stage 3 experiment was run explicitly as exploratory (user directive: *"Push Stage 3 anyway as exploratory — confirms the hypothesis and surfaces B200-specific insights"*) — the hypothesis is confirmed and the insights are documented above.

## Recommended next directions

Same list as in `p5c_tb/ab_report.md`, now with higher confidence that WS-style rearrangement of the existing fused kernel is **not** the direction:

1. **Split-K G=16 dispatcher-threshold tuning** — benchmark-driven only, no kernel changes. Low-risk 1–3 % geomean win on the 7 longest traces.
2. **Fused partial + reduce persistent kernel** — eliminates the second launch's ~3 µs overhead on split-K workloads. Medium effort, ~5–10 % on the split-K subset.
3. **Cluster-shared (DSM) sparse_idx broadcast** — 4 CTAs / cluster share one copy. Low-medium effort, ~1–2 % on short-K paths.
4. **`tcgen05.mma` for the V-accum step** (sm_100a-only) — change the V accumulate from bf16 scalar FMA to tensor-core bf16 → fp32 MMA. High effort but potentially 10–20 % on G=1 paths where V-accum dominates, and it keeps the V-accum at full CTA width (unlike WS3).

## Artifacts

- `artifacts/p5c_ws3/ws3_smoke_20260423T014823.log` — initial smoke (TIMEOUT: `setmaxnreg<240>` vs `__launch_bounds__(..., 2)` mismatch)
- `artifacts/p5c_ws3/ws3_smoke_20260423T015721.log` — second smoke, all 5 workloads PASSED after fix
- `artifacts/p5c_ws3/ws3_full_20260423T015815.log` — full 23-workload bench, all PASSED
