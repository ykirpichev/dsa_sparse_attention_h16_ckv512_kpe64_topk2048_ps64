# Phase 5c — WS (triple-buffer TMA) variant vs cp.async baseline vs TMA

**Branch:** `phase5-ws-parissue` (new kernel `mla_sparse_attn_kernel_ws`)  
**Date:** 2026-04-23  
**Toggle:** `MLA_SPARSE_ATTN_WS=1`  
**Verdict:** ❌ **Keep off by default.** Same ~+4–9 % regression band as TMA; triple-buffering the `SYNC_C-TMA` barrier away did not recover the underlying TMA overhead.

## Configuration

`mla_sparse_attn_kernel_ws` is a sibling of `mla_sparse_attn_kernel_tma` with the following deltas:

- `V_PIPE = 2` → `WS_SLOTS = 3`, `WS_PREFETCH = 2`.
- Per-tile SMEM arenas (`s_v_lo`, `s_v_hi`, `s_kpe_tile`) and the mbarrier array expanded to `WS_SLOTS`.
- The post-V-accum `__syncthreads()` (`SYNC_C-TMA` in the TMA variant) was **removed**. With 3 slots and prefetch depth 2, the producer's target slot `(T+2) mod 3` is always disjoint from the consumer's current slot `T mod 3`, so gather4 writes via the async proxy cannot race with generic-proxy reads.
- SMEM layout reordered: `s_tile_mbar` placed **last** so the 24-byte u64 array doesn't break 16-B alignment of `s_qn` (required for `cp.async.16` / `uint4`). The first layout attempt crashed with XID-13 Misaligned Address; fixed.
- Max dynamic SMEM cap raised to `KERNEL_MAX_DYN_SMEM_WS = 72 KiB` (actual: 64 760 B).

Kernel stats (reported by `configure_ws_kernel_attrs`):

```
[mla_sparse_attn_ws] regs/thread=60  static_smem=0  max_dyn_smem=73728  max_threads=256
```

For comparison: cp.async baseline uses 69 regs/thread; the WS kernel drops to **60 regs/thread** — lower register pressure, matches the same `__launch_bounds__(THREADS, 2)`.

## A/B results (FULL bench, Modal B200, 23 workloads, 1 run each)

| Workload   | Base (ms) | TMA (ms) | WS (ms) | Δ WS vs Base | Path      |
|------------|-----------|----------|---------|--------------|-----------|
| 0c23b10c   | 0.008     | 0.008    | 0.008   |  +0.0 %      | TMA (G=1) |
| 9d4a5f21   | 0.008     | 0.008    | 0.009   | +12.5 %      | TMA (G=1) |
| fc85411e   | 0.008     | 0.008    | 0.009   | +12.5 %      | TMA (G=1) |
| f77df5ce   | 0.008     | 0.008    | 0.009   | +12.5 %      | TMA (G=1) |
| 0a63b87b   | 0.010     | 0.011    | 0.011   | +10.0 %      | TMA (G=1) |
| b7668cfd   | 0.010     | 0.011    | 0.011   | +10.0 %      | TMA (G=1) |
| 68d6817d   | 0.011     | 0.011    | 0.011   |  +0.0 %      | TMA (G=1) |
| e6b849f2   | 0.012     | 0.014    | 0.014   | +16.7 %      | TMA (G=1) |
| 9f3f891b   | 0.023     | 0.027    | 0.028   | +21.7 %      | mixed/G=1 |
| 05f6de65   | 0.027     | 0.033    | 0.033   | +22.2 %      | mixed/G=1 |
| ddfa9e34   | 0.035     | 0.034    | 0.035   |  +0.0 %      | split-K   |
| 4c46a94b   | 0.040     | 0.039    | 0.040   |  +0.0 %      | split-K   |
| 385742b2   | 0.042     | 0.041    | 0.042   |  +0.0 %      | split-K   |
| d57eb9e1   | 0.042     | 0.043    | 0.042   |  +0.0 %      | split-K   |
| 38389961   | 0.052     | 0.051    | 0.052   |  +0.0 %      | split-K   |
| ae4219a9   | 0.054     | 0.054    | 0.054   |  +0.0 %      | split-K   |
| 2207f0fd   | 0.054     | 0.054    | 0.054   |  +0.0 %      | split-K   |
| 7a389715   | 0.055     | 0.055    | 0.056   |  +1.8 %      | split-K   |
| 232ed014   | 0.057     | 0.057    | 0.057   |  +0.0 %      | split-K   |
| 02d6ae9c   | 0.060     | 0.055    | 0.056   |  -6.7 %      | split-K   |
| 78b2e11c   | 0.067     | 0.057    | 0.060   | -10.4 %      | split-K   |
| 5096e459   | 0.070     | 0.070    | 0.071   |  +1.4 %      | split-K   |
| 564007ac   | 0.073     | 0.073    | 0.073   |  +0.0 %      | split-K   |

Notes:
- All 23 correctness-passed. Abs_err values (1.5e-02 at most) are at/under the flashinfer-bench reference-precision band for this definition.
- The long (split-K) workloads all use the unchanged cp.async partial kernel — those rows are same-path reruns of the same code and any delta is timing noise.
- Quantization: latencies are reported rounded to 3 decimal digits (1 µs). For an 8 µs workload, that's ≈12.5 % per-workload rounding noise; individual rows at ±12 % should not be over-interpreted — the geomeans below are more informative.

### Geomeans

|                        | Base (ms) | TMA (ms)   | WS (ms)    | WS Δ vs Base | WS Δ vs TMA |
|------------------------|-----------|------------|------------|--------------|-------------|
| All 23 workloads       | 0.0270    | 0.0274     | 0.0279     | +3.3 %       | +1.8 %      |
| Short (TMA-only, n=8)  | 0.0093    | 0.0097     | 0.0101     | +9.0 %       | +4.2 %      |
| Long (split-K, n=13)   | 0.0493    | 0.0491     | 0.0493     |  0.0 %       | +0.4 %      |

The WS variant is slightly **worse** than the TMA variant on the short workloads where both are active (+4.2 % WS/TMA), and equal-to-noise on the long split-K workloads (as expected — that path is unchanged).

## Why the triple-buffer change didn't help

Predicted saving: SYNC_C-TMA barrier removal ≈ 30 cycles/tile × ~8 tiles (short) / ~128 tiles (long G=1) = ~0.25 µs per CTA on short workloads and ~4 µs on the longest. Measured saving: **unobservable**, swamped by the underlying TMA gather4 overhead that already regressed TMA vs cp.async by +4.4 %.

Hypotheses (not separately measured; would need ncu profiling):

1. **Occupancy-limited TMA dispatch**: only one gather4 can be in flight at a time on B200's single TMA unit per SM partition, and with 3 gather4 instructions per tile (lo/hi/kpe) × 4 row-groups = 12 serial issues per tile, the main-loop is TMA-dispatch-bound, not SMEM-fence-bound. Removing the SYNC_C-TMA barrier doesn't help if the critical path is already in the producer issue queue.
2. **Extra SMEM causes extra DRAM spill?** — unlikely, 63 KiB is well inside occupancy tier 2 (≤73 KiB for 2 CTAs/SM on B200); `configure_ws_kernel_attrs` didn't complain about occupancy reduction.
3. **Register drop from 69 → 60 doesn't matter** because the kernel is memory-bound anyway (K dot-products), not register-pressure bound. Freeing 9 regs/thread doesn't increase occupancy past the SMEM tier cap.

## Implications for Stage 3 (dedicated DMA warp + setmaxnreg + V-accum restructure)

The theoretical upside of Stage 3 over the current WS kernel is:

- **Dedicated DMA warp** that only issues TMA + services mbarrier arrives, freeing the other 7 warps from touching the producer critical path. Saves cycles equivalent to the `issue_tile_tma` lambda on tid=0 blocking the rest of the warp at `mbarrier_arrive_expect_tx`.
- **`setmaxnreg.inc` on consumers** / **`setmaxnreg.dec` on producer** to shift registers asymmetrically — producer doesn't need many registers, consumers can use more. But our consumer is already at 60 regs/thread, far below any occupancy constraint (SMEM dominates).
- **V-accum restructure to 128 threads × 4 bf16 each** to free half the CTA for the producer warp group. This means halving the effective compute throughput of the online-softmax V-accumulate loop. On the short G=1 workloads where WS is active, the V-accum step is a sizeable fraction (≈40 %) of the tile critical path. Halving its parallel width is likely a **net regression** for short workloads, regardless of any producer-side savings.

Given (a) TMA itself is already a +4.4 % regression that WS's structural improvements couldn't recover, and (b) the Stage 3 V-accum change risks adding another regression on the exact workloads that are latency-critical, I estimate **< 15 % probability** that Stage 3 recovers a geomean-positive result on these 23 workloads.

## Decision

Keep `mla_sparse_attn_kernel_ws` as reference code on this branch (reachable via `MLA_SPARSE_ATTN_WS=1`) but **don't dispatch to it by default**. The +3.3 % geomean regression is not worth carrying the extra complexity.

## Recommended next directions (non-WS)

1. **Split-K G=16 tuning** — dispatcher threshold lowering. Zero kernel work, just benchmark-driven threshold tuning. Expected win: 1–3 % on the 7 longest traces.
2. **Fused partial+reduce persistent kernel** — eliminate the second kernel launch's ~3 µs overhead on split-K workloads. Medium effort, ~5–10 % win on workloads currently taking the split-K path.
3. **Cluster-shared (DSM) sparse_idx broadcast** — load once per cluster (4 CTAs) and share via distributed shared memory. Low-to-medium effort, ~1–2 % win on short-K workloads.
4. **Compute-bound tuning of V-accum** — experiment with bf162 → bf16 tensor-core `wgmma` for the accumulate step. High effort, but potentially 10–20 % on the long G=1 workloads where V-accum dominates.
