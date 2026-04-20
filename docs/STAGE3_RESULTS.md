# Stage 3 — Head Tiling (NEGATIVE RESULT)

## TL;DR

Head tiling with `H_TILE=2` (two heads per CTA, sharing all K-side loads and
`sparse_idx`) gave **no speedup** over Stage 2 on our 23-workload suite:

| Metric                                  | Stage 2      | Stage 3      | Delta  |
| --------------------------------------- | -----------: | -----------: | -----: |
| Mean latency (ms)                       | **0.230**    | **0.229**    | -0.5%  |
| Mean speedup vs. definition reference   | **11.61×**   | **11.01×**   | -0.6×  |
| Workloads where Stage 3 beats Stage 2 (>2%) | —        | **0 / 23**   | —      |
| Workloads where Stage 3 regresses (>2%)     | —        | **9 / 23**   | —      |

Stage 2 remains the best kernel. Stage 3 was reverted on this branch; the
experiment is preserved in the git history for reference
(`artifacts/modal_stage3_full_20260420T071129Z.json` and this doc).

## What was tried

- Grid reshaped from `(T, H)` to `(T, H / H_TILE)` with `H_TILE = 2`.
- Each block now handles two consecutive heads of the same token, so every
  packed K load is reused for:
  - `H_TILE = 2` logit dot products in the QK phase (1 HBM read → 2 accumulators).
  - `H_TILE * 2 = 4` FMAs in the value projection (1 HBM read → 4 FMAs).
- `sparse_idx` is per-token in this workload, so the two heads touch the exact
  same K rows — no re-gather needed.
- SMEM per block roughly doubled for Q/logit storage (~28.6 KiB total); still
  well under the 100 KiB attribute cap.
- Register pressure grew by `H_TILE * 2 = 4` output accumulators plus two
  softmax states (`m_h`, `l_h`).

All 23 workloads passed correctness.

## Per-workload comparison (sorted by Stage 2 latency)

| workload   | S2 latency (µs) | S3 latency (µs) | delta   | S2 speedup | S3 speedup |
| ---------- | --------------: | --------------: | ------: | ---------: | ---------: |
| 0c23b10c   |            49.2 |            56.5 | +14.9%  |    19.19×  |    16.96×  |
| f77df5ce   |            51.2 |            53.3 |  +4.1%  |    21.31×  |    20.31×  |
| fc85411e   |            51.2 |            53.3 |  +4.1%  |    21.34×  |    20.39×  |
| 9d4a5f21   |            51.2 |            53.5 |  +4.6%  |    20.93×  |    20.28×  |
| 68d6817d   |            52.8 |            55.2 |  +4.5%  |    31.43×  |    28.78×  |
| b7668cfd   |            53.3 |            55.7 |  +4.5%  |    20.61×  |    19.43×  |
| 0a63b87b   |            55.3 |            57.3 |  +3.7%  |    19.79×  |    18.88×  |
| e6b849f2   |            59.4 |            61.5 |  +3.5%  |    18.48×  |    17.65×  |
| 9f3f891b   |           100.3 |           102.4 |  +2.1%  |    10.94×  |    10.78×  |
| 05f6de65   |           107.7 |           108.6 |  +0.9%  |    10.23×  |    10.04×  |
| 4c46a94b   |           233.5 |           231.4 |  -0.9%  |     8.39×  |     7.92×  |
| 385742b2   |           241.8 |           240.1 |  -0.7%  |     8.06×  |     7.71×  |
| 38389961   |           247.8 |           247.8 |   0.0%  |     7.40×  |     6.90×  |
| 232ed014   |           249.9 |           247.8 |  -0.8%  |     7.84×  |     7.64×  |
| ddfa9e34   |           352.3 |           348.1 |  -1.2%  |     4.71×  |     4.56×  |
| 7a389715   |           354.3 |           350.2 |  -1.1%  |     5.50×  |     5.27×  |
| 5096e459   |           419.5 |           413.7 |  -1.4%  |     4.67×  |     4.48×  |
| d57eb9e1   |           421.9 |           417.8 |  -1.0%  |     4.01×  |     3.86×  |
| 2207f0fd   |           422.9 |           417.8 |  -1.2%  |     4.31×  |     4.16×  |
| 02d6ae9c   |           430.1 |           423.9 |  -1.4%  |     4.58×  |     4.34×  |
| ae4219a9   |           430.1 |           424.9 |  -1.2%  |     4.18×  |     4.04×  |
| 78b2e11c   |           430.1 |           422.9 |  -1.7%  |     4.53×  |     4.40×  |
| 564007ac   |           430.2 |           426.0 |  -1.0%  |     4.59×  |     4.36×  |

## Why it didn't work

The working hypothesis was that the large traces (~420 µs at Stage 2) were
bottlenecked on K-side HBM bandwidth, and halving the bytes-per-head would
give up to ~2× headroom. The measurements contradict that:

1. **Large traces are compute/issue-bound, not K-bandwidth-bound.**
   At Stage 2, ~320 blocks × 2.13 MiB = 682 MiB of HBM traffic takes ~420 µs
   of wall time, giving ~1.6 TB/s effective — versus ~7.5 TB/s peak HBM3e on
   B200. We were nowhere near the wall. Doubling compute per block (H_TILE=2)
   kept per-block latency essentially unchanged (~420 µs still), so halving
   the number of blocks just halved GPU utilization for the same wall time.
   Large traces moved 1-2%, within noise.

2. **Small traces regressed because they were already occupancy-starved.**
   For T=1-2 tokens × H=16 heads = 16-32 blocks on 148 SMs × 2 blocks/SM =
   ~296 block slots. Stage 2 had poor SM utilization there already; Stage 3
   halves the block count (to 8-16), moving utilization from ~5-10% down to
   ~3-5% of available concurrency. With the GPU this underused, latency hiding
   collapses: even though each block now "should" finish just as fast, warp
   schedulers have fewer blocks to swap between during memory stalls. Net
   effect: +3-15% latency on the smallest workloads.

3. **SMEM pressure grew without a matching reduction in HBM pressure.**
   Stage 3 SMEM per block (~28.6 KiB) > Stage 2 (~18.5 KiB). Fewer blocks fit
   per SM by SMEM alone, re-narrowing an already-narrow scheduling window. We
   still hit the `launch_bounds(THREADS, 2)` cap either way, so this mostly
   hurt the L1/SMEM bank utilization, not raw occupancy.

The practical lesson: on this workload, **K-side traffic is not the dominant
cost** for any trace size. The large traces want more ILP / overlap inside a
single block; the small traces want *more* blocks, not fewer.

## What to try instead (Stage 3b candidates)

With head tiling ruled out, the remaining high-value options are:

1. **cp.async double-buffered tile prefetch for the value projection.**
   Retry the Stage-2-v1 shape, but make the K tile load asynchronous with
   `cp.async.bulk` / `cp.async.commit_group` and a per-tile mbarrier so the
   next tile's load overlaps with the current tile's 256-thread FMA loop.
   This directly attacks the "issue/stall bound per block" diagnosis: the FMA
   loop keeps issuing while the memory system prefetches the next tile.

2. **Split-K along the TOPK dimension** for large traces.
   Each block currently serializes `for k in 0..TOPK`. Splitting into, say,
   `NK=4` blocks per (t, h) running [0, 512), [512, 1024), … then merging
   their partial softmax state through a small second kernel would 4× the
   parallelism on the T=20 traces (320 → 1280 blocks), pushing kernel latency
   below one wave.

3. **wgmma `P @ V`** (harder, highest ceiling).
   Cooperate on a per-CTA `[M=1, K=2048, N=512]` matmul using tensor cores.
   Requires packing the softmax output into bf16 SMEM first and an async
   pipeline; likely needs H_TILE>1 to get meaningful M to feed wgmma.

Given that head tiling didn't move the needle and cp.async directly
addresses the observed bottleneck, cp.async + double buffering is the
recommended next experiment.

## Operational notes

- Full-suite run: `artifacts/modal_stage3_full_20260420T071129Z.json`
  (23/23 PASSED).
- Kernel version during the Stage 3 run is preserved in git on this branch's
  parent commit; HEAD on this branch has been reverted back to the Stage 2
  kernel so the "best" artifact on disk continues to match
  `artifacts/modal_stage2_full_20260420T065116Z.json`.
