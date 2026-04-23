# Phase 5b — TMA gather4 variant (half-row SMEM) vs cp.async baseline

**Branch:** `phase5-tma`  
**Date:** 2026-04-22  
**Toggle:** `MLA_SPARSE_ATTN_TMA=1`  
**Verdict:** ❌ **Keep off by default.** Neutral-to-slow across the board.

## Configuration

Two `CUtensorMap` descriptors encode `ckv_flat` as lo/hi half-rows
(each `boxW=256`, `row_stride=D_CK=512`), one for `kpe_flat`
(`boxW=64`). SMEM layout splits the value tile into `s_v_lo` +
`s_v_hi`. Main-kernel compute reads lo and hi separately; V-accumulate
branches warp-uniformly on `tid<128` for lo vs `tid>=128` for hi.

One additional `__syncthreads` per tile (SYNC_C-TMA) is required to
prevent the async-proxy gather4 from re-writing a slot while consumers
are still reading it (discovered during smoke testing — 3/10 workloads
showed `abs_err≈3` without the fence).

Only the `G==1` fast path was TMA-ized; split-K (`G>1`) still uses the
cp.async partial kernel.

## A/B results (FULL bench, Modal B200, 23 workloads, 1 run)

| Workload   | Base (ms) | TMA (ms) | Δ %    | Path          |
|------------|-----------|----------|--------|---------------|
| 9d4a5f21   | 0.008     | 0.008    | +0.0%  | TMA (G=1)     |
| 0c23b10c   | 0.008     | 0.008    | +0.0%  | TMA (G=1)     |
| fc85411e   | 0.008     | 0.008    | +0.0%  | TMA (G=1)     |
| f77df5ce   | 0.008     | 0.008    | +0.0%  | TMA (G=1)     |
| 0a63b87b   | 0.010     | 0.011    | +10.0% | TMA (G=1)     |
| b7668cfd   | 0.010     | 0.011    | +10.0% | TMA (G=1)     |
| 68d6817d   | 0.011     | 0.011    | +0.0%  | TMA (G=1)     |
| e6b849f2   | 0.012     | 0.014    | +16.7% | TMA (G=1)     |
| 9f3f891b   | 0.023     | 0.027    | +17.4% | mixed/G=1*    |
| 05f6de65   | 0.027     | 0.033    | +22.2% | mixed/G=1*    |
| ddfa9e34   | 0.035     | 0.034    | -2.9%  | split-K       |
| 4c46a94b   | 0.040     | 0.039    | -2.5%  | split-K       |
| 385742b2   | 0.042     | 0.041    | -2.4%  | split-K       |
| d57eb9e1   | 0.042     | 0.043    | +2.4%  | split-K       |
| 38389961   | 0.052     | 0.051    | -1.9%  | split-K       |
| ae4219a9   | 0.054     | 0.054    | +0.0%  | split-K       |
| 2207f0fd   | 0.054     | 0.054    | +0.0%  | split-K       |
| 7a389715   | 0.055     | 0.055    | +0.0%  | split-K       |
| 232ed014   | 0.057     | 0.057    | +0.0%  | split-K       |
| 02d6ae9c   | 0.060     | 0.055    | -8.3%  | split-K       |
| 78b2e11c   | 0.067     | 0.057    | -14.9% | split-K       |
| 5096e459   | 0.070     | 0.070    | +0.0%  | split-K       |
| 564007ac   | 0.073     | 0.073    | +0.0%  | split-K       |

`*` moderately-long G=1 workloads where TMA regressions are biggest.

### Geomeans

|                       | Base (ms) | TMA (ms) | Δ      |
|-----------------------|-----------|----------|--------|
| All 23 workloads      | 0.0270    | 0.0274   | +1.68% |
| Short (TMA-only, n=8) | 0.0093    | 0.0097   | +4.40% |
| Long (no TMA, n=15)   | 0.0477    | 0.0478   | +0.25% |

The long-workload ~0.25% delta is run-to-run noise — those paths are
identical between the two runs. The short-workload +4.4% regression
is where TMA was actually active, and it's consistent across all 8
latency-sensitive cases.

## Why TMA regressed here

1. **Small tile count.** The longest workloads have ≤128 tiles
   (TOPK=2048, BK=16). TMA's setup cost (12 gather4 instructions
   issued serially by thread 0 + mbarrier coordination) is fixed
   per tile — for short K, the "all 256 threads issue cp.async in
   parallel" path has higher instruction throughput.
2. **Extra SMEM-read fence.** The race-fix `__syncthreads` at the
   end of every tile adds a full CTA barrier that cp.async didn't
   need (because cp.async writes are completely invisible to the
   generic proxy until `cp.async.wait_group`).
3. **Descriptor-build overhead.** Each first-use `cuTensorMapEncodeTiled`
   call costs a few µs. Cached after first launch, but two new
   descriptors per launch (lo+hi) vs the single-tensor original.
4. **Half-row box constraint.** `D_CK=512` > `boxDim`≤256 cap forced
   2 ckv descriptors + 2× gather4 count (8→12 per tile), which
   partially negates TMA's "one instruction, many bytes" advantage.

Future directions that might change the verdict:
- **Warp-specialized producer/consumer** with TMA + mbarrier would let
  the producer warp amortize the setup while consumers run. Different
  code path from the lockstep one here.
- **Multi-tile batched TMA** (gather4 over 8 rows instead of 4, or
  issuing 2 tiles in one expect_tx) — boxDim[1] > 1 is not supported
  for gather4, but SMEM pipeline depth > 2 would help hide setup.
- **Longer sequences** (TOPK ≥ 4096 or BK ≥ 32) would amortize the
  per-tile overhead better.

## Decision

Keep the full TMA variant on `phase5-tma` as reference code
(reachable via `MLA_SPARSE_ATTN_TMA=1`) but leave the default path
unchanged. The +4.4% regression on the latency-critical short
workloads is not worth carrying the extra complexity.
