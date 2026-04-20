# Stage 3b — cp.async Double-Buffered Value Projection

Team: Wombat (agent-assisted)
Hardware: NVIDIA B200 (compute capability 10.0) via Modal
Suite: 23/23 workloads, 50 iterations × 5 trials, flashinfer-bench defaults
Artifact: `artifacts/modal_stage3b_full_20260420T082159Z.json`

## Summary

| metric                      | Stage 2   | Stage 3b  | Δ        |
| --------------------------- | --------- | --------- | -------- |
| mean latency                | 230.2 µs  | 186.8 µs  | **-18.9%** |
| mean speedup vs reference   | 11.61×    | **25.46×** | +2.19×    |
| mean per-workload s2/s3b    | —         | **1.77×**  | geo-style |
| workloads PASSED            | 23/23     | **23/23** | —        |
| workloads improved >2%      | —         | **23/23** | —        |
| workloads regressed         | —         | **0/23**  | —        |

Every workload got faster. No regressions.

## Head-to-head (latencies in µs)

| uuid     | Stage 2 | Stage 3b | s2/s3b | s3b vs ref |
| -------- | ------: | -------: | -----: | ---------: |
| 0c23b10c |    49.2 |     16.1 |  3.06× |    58.33×  |
| fc85411e |    51.2 |     16.4 |  3.12× |    64.94×  |
| 9d4a5f21 |    51.2 |     16.4 |  3.12× |    65.05×  |
| f77df5ce |    51.2 |     16.4 |  3.12× |    64.86×  |
| 68d6817d |    52.8 |     18.4 |  2.86× |    **84.49×** |
| b7668cfd |    53.3 |     20.5 |  2.60× |    51.91×  |
| 0a63b87b |    55.3 |     22.5 |  2.45× |    47.15×  |
| e6b849f2 |    59.4 |     28.7 |  2.07× |    37.12×  |
| 9f3f891b |   100.3 |     63.5 |  1.58× |    16.81×  |
| 05f6de65 |   107.7 |     71.7 |  1.50× |    14.92×  |
| 4c46a94b |   233.5 |    189.7 |  1.23× |     9.59×  |
| 385742b2 |   241.8 |    198.7 |  1.22× |     9.17×  |
| 38389961 |   247.8 |    204.8 |  1.21× |     8.24×  |
| 232ed014 |   249.9 |    204.8 |  1.22× |     8.91×  |
| ddfa9e34 |   352.3 |    303.1 |  1.16× |     5.16×  |
| 7a389715 |   354.3 |    303.1 |  1.17× |     5.97×  |
| 5096e459 |   419.5 |    364.5 |  1.15× |     5.02×  |
| d57eb9e1 |   421.9 |    368.6 |  1.14× |     4.29×  |
| 2207f0fd |   422.9 |    368.7 |  1.15× |     4.61×  |
| 02d6ae9c |   430.1 |    374.8 |  1.15× |     4.84×  |
| 564007ac |   430.2 |    374.8 |  1.15× |     4.88×  |
| 78b2e11c |   430.1 |    374.8 |  1.15× |     4.85×  |
| ae4219a9 |   430.1 |    374.8 |  1.15× |     4.48×  |

Two distinct regimes:
* **Short-sequence (T≤2, many -1 entries in `sparse_idx`):** 2.5-3.1× speedup driven
  by the tile-loop *truncation* — we only iterate over tiles that actually contain
  valid rows.
* **Long-sequence (T≥10, dense `sparse_idx`):** 1.14-1.23× speedup driven by
  asynchronous K-row prefetch overlapping with the FMA pass.

## vs FlashInfer TRT-LLM MLA

The previous measurement in `docs/PERFORMANCE_VS_FLASHINFER.md` put FI TRT-LLM
at **mean 259 µs / 6.05× vs reference**. Stage 3b at **187 µs** beats it by
**1.38×** on the mean, while clearing every workload with 23/23 correctness.

## Why this works (what the previous attempts taught us)

* **Stage-3 head-tiling** proved the bottleneck is *not* K-side HBM bandwidth —
  halving K reads per head did nothing overall (see `docs/STAGE3_RESULTS.md`).
* **Branch-free + `#pragma unroll` in place** (first Stage-3b attempt) made
  small traces 10-16% slower, because the old `if (g < 0) continue;` was
  skipping ~95% of iterations on short sequences; removing it caused the
  kernel to issue 2048 fake loads per block regardless of seq_len.
* **cp.async without truncation** (second attempt) gave -11% on large traces
  but +17-28% on short ones, for the same "issues 128 tiles every time"
  reason.
* **cp.async + tile-loop truncation** (current Stage-3b) unifies both wins:
  we keep the async overlap for long sequences *and* cap tile count at
  `ceil((max_valid_k + 1)/BK)` so short sequences barely touch HBM.

## What changed in the kernel

File: `solution/cuda/mla_cuda.cu`

1. New constants — `BK=16`, `V_PIPE=2`, `NUM_V_TILES=TOPK/BK=128`,
   `CHUNKS_PER_THREAD=4` (tile is 16 rows × 512 bf16 = 16 KiB; 256 threads × 4 ×
   16 B = 16 KiB).
2. Added SMEM buffer `s_v_tile[V_PIPE][BK][D_CK]` (32 KiB), bringing total
   dynamic SMEM to ~50.6 KiB (well under the 100 KiB `cudaFuncAttribute`
   cap). s_reduce is 32 B so s_v_tile stays 16 B aligned.
3. While caching `sparse_idx` into `s_gidx`, track the largest valid `k`.
   After a `block_reduce_max` this gives `n_valid_tiles = ceil((max+1)/BK)`.
   That's the *only* extra reduction Stage 3b adds over Stage 2.
4. New inline-PTX helpers `cp_async_16`, `cp_async_commit_group`,
   `cp_async_wait_group<N>` (templated on the immediate waitdepth).
5. Value projection rewritten around a two-buffer async pipeline:
    * prime: issue tile 0 to buf 0, `commit_group`;
    * loop `t = 0 .. n_valid_tiles-1`:
      * if there's a next tile, issue it to the other buf and commit, then
        `wait_group<1>` (leave just the newest group in flight);
      * otherwise `wait_group<0>` (drain);
      * `__syncthreads` → run the FMA inner loop entirely out of SMEM
        (`tile_pairs[r * (D_CK/2) + tid]`) → `__syncthreads`.
    * `if (g < 0) continue;` is kept *inside* the BK loop: short-sequence
      traces still skip per-row compute; cp.async issues `ckv_flat[0]` as
      a safe fallback for those rows (L2-cached, negligible).

The rest of the kernel (Q load, logit pass, block softmax, output store) is
unchanged from Stage 2.

## SMEM / occupancy

| item      | bytes  |
| --------- | -----: |
| s_logits  |  8192  |
| s_qn      |  2048  |
| s_qp      |   256  |
| s_gidx    |  8192  |
| s_reduce  |    32  |
| s_v_tile  | 32768  |
| **total** | **51,488 B (~50.3 KiB)** |

Still fits 2 CTAs/SM (`__launch_bounds__(256, 2)`), so no loss of occupancy
vs Stage 2.

## Candidates for Stage 3c

* **Apply the same cp.async pipeline to the logit pass.** The logit phase
  reads `ckv_flat[g]` *and* `kpe_flat[g]` per k — a double-buffered
  prefetch there could reclaim another 20-30% on long sequences.
* **`cp.async.cg` with `.cache_global` and `ld.global.nc` hints** once we
  confirm L1 vs L2 behaviour on B200 with nsight-compute.
* **Three-way buffering (`V_PIPE=3`)** — worth trying only if we see
  wait-group stalls on the current configuration.

For now, Stage 3b is the final submission candidate on this branch; all 23
workloads pass with a 1.77× mean-ratio improvement over Stage 2 and a 1.38×
improvement over FlashInfer TRT-LLM.
