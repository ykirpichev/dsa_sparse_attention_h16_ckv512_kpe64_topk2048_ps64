# Phase 5c — Warp-Specialization Experiment (parallel gather4 issue)

Branch: `phase5-ws-parissue` off `phase5-tma`.

## Change

`mla_sparse_attn_kernel_tma::issue_tile_tma` was refactored from thread-0
serial issue (12 gather4s in sequence) to **warp-0 parallel issue** — lanes
0..11 each fire one gather4, lane 0 also does the combined `arrive.expect_tx`.
Warps 1..7 short-circuit out of the lambda. This is "WS-lite": a true
dedicated DMA warp isn't feasible here because V-accum fundamentally needs
all 256 threads (packed bf162 output, thread i owns pair (2i, 2i+1)).

- Per-tile issue cost: 3 warp cycles (3-way diverged if/else on descriptor
  kind) vs 12 serial issues from thread 0.
- Compiler register allocation dropped from **69 → 64 regs/thread** on the
  TMA kernel (occupancy unchanged at 2 blocks/SM).

## Measurement (Modal B200, full 23-workload suite, same sample-id seed)

```
Geomean WS-parissue / prior TMA (serial):   +3.87%  (REGRESSION)
Geomean WS-parissue / non-TMA baseline:     +5.62%  (REGRESSION)
```

Worst-case regressions hit the mid-range G=1 workloads that route through
the TMA path (5-14 µs band), e.g.:
- `9f3f891b`  0.027 → 0.029 ms  (+7.4%)
- `78b2e11c`  0.057 → 0.065 ms  (+14.0%)
- `9d4a5f21`  0.008 → 0.009 ms  (+12.5%)
- `fc85411e`  0.008 → 0.009 ms  (+12.5%)

Split-K workloads (T·H < 444) don't use the TMA kernel and are unchanged
(within measurement noise).

## Why it doesn't help

1. **TMA unit is the bottleneck, not instruction dispatch.** Saving ~9
   warp cycles of issue overhead is invisible against the 12 serial TMA
   transfers that the shared HW unit performs.
2. **Divergent lane issue adds latency** — the 3-way if/else on descriptor
   kind runs in ~3 warp cycles, each active on ~4 lanes, vs thread-0 issue
   which is a tight single-lane loop. Warp-scheduler fill matters more than
   throughput at this granularity.
3. **Register count drop is irrelevant** — previous kernel had 0 spills
   (localSz=0), so 69 vs 64 regs doesn't affect memory traffic. And
   occupancy didn't change.
4. **TMA's +4.4% baseline overhead** from Phase 5b (`SYNC_C-TMA` extra
   barrier + descriptor encoding) is unchanged — parallel issue doesn't
   attack that cost.

## Broader verdict on full "warp specialization" for this kernel

Before attempting a bigger WS rewrite (dedicated DMA warp + `setmaxnreg` +
named-barrier producer/consumer), consider the structural blockers on this
shape:

- **V-accum uses all 256 threads** (packed bf162 output). A dedicated
  1-warp DMA can't be "idle during compute" because it's still needed for
  V-accum. Either restructure V-accum to use ≤224 threads (algorithmic
  rewrite, maybe wider per-thread output = 4 bf16/thread), or accept that
  the DMA "warp" is only specialized during logit and rejoins for V-accum.
- **Persistent grid is counterproductive**: T·H ranges from 16 to 256;
  B200 has 148 SMs × 2 blocks/SM = 296 concurrent CTAs. Current dispatch
  already fills most SMs for every workload. Capping the grid at 148 to
  enable persistence would serialize 16 heads per CTA, losing ~9× parallelism
  on T=16 workloads.
- **Sparse_idx amortization across heads of a token requires cluster-shared
  SMEM (DSM)** — 16 CTAs per token forming a cluster. That's a separate
  much-larger refactor (a Phase-3 backlog item in the optimization log).
- **`SYNC_C-TMA` cannot be replaced by mbarrier producer/consumer cheaply**
  — it's guarding the point where ALL 256 compute threads have finished
  reading slot `buf` before the producer rewrites it. Any replacement still
  requires all-thread participation → equivalent to `__syncthreads` cost.

## Decision

**Revert this change**; keep it on the `phase5-ws-parissue` branch for
reference. `phase5-tma` remains the opt-in TMA variant (off by default due
to its own +4.4% regression documented in `artifacts/p5b/ab_report.md`).
`main` continues to use the cp.async baseline.

## Files

- `solution/python/kernel.cu` — `mla_sparse_attn_kernel_tma::issue_tile_tma`
  refactored for parallel lane issue; `warp_id`/`lane`/`i` declarations
  hoisted above the lambda.
- `artifacts/p5c/smoke_parissue_*.log` — 6-workload correctness smoke
  (PASSED, errors within spec).
- `artifacts/p5c/full_ws_parissue_*.log` — full 23-workload benchmark.
