# Phase 6d — tcgen05.mma for V-accum — **ANALYSIS-GATED SKIP**

**Date:** 2026-04-23
**Verdict:** V-accum is not compute-bound; tensor-core MMA saves ≤ 0.1 % of kernel time. Skipped.

## Background

The recommended next direction in `artifacts/p5c_ws3/ab_report.md` was to accelerate the V-accumulate step with `tcgen05.mma` (Blackwell's 5th-generation tensor-core instruction for sm_100a). The hypothesis: V-accum is the hot path (the WS3 experiment regressed −15 % when halving V-accum threads), and tensor cores have 10–20× the FMA throughput of scalar FP32.

## Compute intensity of V-accum (per T=8 workload)

Per-thread per-tile: `BK × 2 = 32 FMA` (two bf16 channels × 16 K rows).
Per-CTA (8 tiles): `32 × 8 × 256 threads = 65 536 FMA`.
Per T=8 H=16 workload: `65 536 × 128 CTAs = 8.4 × 10⁶ FMA ≈ 17 MFLOP per workload`.

On B200:
- Sustained FP32 scalar throughput (whole GPU): ~ 2 PFLOPS.
- V-accum FP32 time if compute-bound: **0.07 µs**.
- BF16 tensor-core (tcgen05 m64n128k16): ~ 4 PFLOPS sustained, but our GEMV has `N=1` mapped onto `N ≥ 128` — 1/128 effective utilization, or ~ 30 GFLOPS.
- WMMA (m16n8k16, older), `N=1` on `N=8` mapping: 1/8 utilization, ~ 0.27 µs projected.

**The kernel's observed time on 564007ac (T=8 longest) is 73 µs.** V-accum compute is therefore **≤ 0.1 %** of kernel time. It is clearly **not the bottleneck**.

## Reconciling with the WS3 regression data

WS3 (`artifacts/p5c_ws3/ab_report.md`) halved V-accum parallelism (256 → 128 threads), and measured ≈ 15 % regression. If the FMA rate were the bottleneck, halving would cost ~ 0.07 µs (0.1 %), not 15 %.

What actually costs time in V-accum:
- **SMEM reads**: each thread reads `ckv_buf[r × D_CK + i × 2]` — 2 bf16 per row × 16 rows × 8 tiles = 256 bf16 = 512 B per thread per workload. Parallel across 256 threads, SMEM BW is 128 KiB / step. Not a bottleneck (B200 SMEM ≈ 19 TB/s per SM).
- **Register pressure**: the unrolled inner loop keeps `o0, o1, m, l, p[0..BK-1], k[0..BK-1]` live. Halving threads doubles per-thread state, pushing past the 64 K register file boundary → occupancy drop from 3 CTAs/SM to 1.
- **ILP bubbles**: the FMAs have data dependencies via `s_logits_tile[r]`. Halving threads serializes more of this per-SM.

So the 15 % WS3 regression was **register-pressure- and latency-bound**, not FMA-throughput-bound. Tensor cores address the FMA-throughput axis — which is not the real constraint here.

## Why tcgen05 is a poor fit for this kernel's shape

- **Shape mismatch.** tcgen05.mma natural shapes are m64n128k16 (and bigger): it expects a GEMM with `N ≥ 128`. Our V-accum is a per-(t,h) GEMV with `N = 1` (one probability vector). Mapped onto `N = 128`, 127/128 of the tensor core's lanes are idle — effective throughput drops from 4 PFLOPS to ~30 GFLOPS, worse than the scalar baseline.
- **Layout overhead.** Loading into tensor memory (`tmem`) for tcgen05 requires a dedicated `tcgen05.cp.shared::cluster.async.tmem` step, which on a GEMV has throughput lower than the FMA it replaces.
- **Batch-H refactor required.** The only way to reclaim the N dim is to batch H = 16 heads per CTA (instead of one CTA per (t, h)). That is a full kernel rearchitecture: change grid from (T, H [, G]) to (T [, G]) with H as an inner loop; rework SMEM to hold all-H Q and O; rewire register accumulators to per-head. Multi-day effort with high risk of regression on the many H=1-like paths we currently hit (register pressure for 16× the per-CTA output is substantial: 16 × 256 × 8 B = 32 KiB of accumulator state).
- **No clear win even after refactor.** Even if H-batching + tcgen05 eliminated all V-accum compute, the theoretical cap is 0.07 µs on a 73-µs workload (0.1 %). The kernel is fundamentally **HBM- and latency-bound**, not compute-bound.

## Where the real bottleneck lives

Rough per-workload HBM footprint for T=8 H=16: K/V + KPE loads ≈ 290 MiB. At B200's ~ 3 TB/s effective HBM BW, the theoretical floor is ≈ **100 µs**. We hit **73 µs**, which implies ~ 27 % of reads are L2-cache-absorbed (overlapping sparse indices across CTAs). To push below 73 µs we'd need:

1. **Reduce HBM reads** — e.g., structured reuse of K/V across H-heads in the same token via cluster-shared tiles or a refactored CTA-per-t kernel (the same refactor that would enable tcgen05). Higher-upside but higher-risk path.
2. **Improve L2 hit-rate** — coordinate CTA scheduling so heads of the same token co-reside temporally. Possible via cluster launches.
3. **Reduce SMEM-to-register traffic** during V-accum — e.g., load K tile via `cp.async.bulk` into a register file subset (`tcgen05.cp` could help here even without MMA).

Of these, only **option 1 plus tcgen05** would both reduce HBM and exploit tensor cores. It's the right long-term target but not the right fit for this session's budget.

## Decision

Skip the tcgen05.mma V-accum experiment. Reasoning documented above.

The kernel has been optimized to ~27 % of the HBM-bandwidth floor. The next-bigger lift requires **CTA-per-t rearchitecture** to both batch H and enable cluster-shared K/V. That is a multi-session project and substantially larger scope than the "all_in_order" step-4 request.

## Artifacts

No code changes. Analysis only.
