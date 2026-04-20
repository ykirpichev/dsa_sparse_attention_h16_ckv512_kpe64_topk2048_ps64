# Stage 3c — `cp.async` double-buffered LOGIT pass (DSA Sparse MLA, B200)

## Headline

On the Modal B200 full sweep (23 / 23 workloads PASSED, `warmup=10, iterations=50, num_trials=3`):

| Metric                                  | Stage 3b        | **Stage 3c**         | Δ (3c vs 3b)   |
|-----------------------------------------|-----------------|----------------------|----------------|
| Mean kernel latency                     | 0.187 ms        | **0.083 ms**         | **-55.6 %** (2.25 × faster) |
| Mean speedup vs PyTorch reference       | 25.46 ×         | **41.45 ×**          | +15.99 ×       |
| Mean speedup vs FlashInfer TRT-LLM MLA  | 1.38 ×          | **3.13 ×**           | +1.75 ×        |

Every single workload (small, medium, large) is strictly faster than Stage 3b
and than `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`. Short
sequence traces (0.012–0.018 ms) hit ≥ 67× speedup vs the PyTorch reference;
one workload (`68d6817d`) now reaches **132.8 × reference speedup**.

Artifact: `artifacts/modal_stage3c_full_20260420T085955Z.json`.

## What changed

Stage 3b left the logit pass unchanged: each of 256 threads streamed its own
8 K rows from HBM in a fully-unrolled dot product. The diagnosed bottleneck
was **dependent long-latency HBM loads** — the same pattern we attacked in
the value projection — but the logit pass has its own twist: every thread's
stream is serial and the compiler cannot pipeline loads across its
`if (g < 0) continue;` branch, so a large fraction of each warp's issue
slots were stalled on returning K rows.

Stage 3c applies the same `cp.async` pipeline we used for value projection,
but to the **logit** pass:

1. **SMEM layout.** `s_v_tile` (32 KiB, two BK×D_CK bf16 buffers from Stage 3b)
   is idle during the logit pass, so we alias it as the per-tile **ckv** row
   buffer. Only +4 KiB of extra dynamic SMEM is added for the per-tile
   **kpe** buffer (`V_PIPE × BK × D_KP × bf16` = 2 × 16 × 64 × 2 = 4 096 B).
   Total dynamic SMEM grows from ~50.3 KiB (3b) to ~54.3 KiB (3c), still
   well under the 100 KiB `cudaFuncAttributeMaxDynamicSharedMemorySize` cap.

2. **Tile issue.** Per tile (BK=16 rows):
   - **ckv (1024 B/row, 16 KiB total):** 4 × 16 B `cp.async` per thread
     (same decomposition as Stage 3b's value tile — 4 `uint4` chunks per
     thread, 64 16-B chunks per row × 16 rows / 256 threads = 4 chunks).
   - **kpe (128 B/row, 2 KiB total):** 1 × 8 B `cp.async` per thread
     (16 rows × 16 8-B chunks / row = 256 = THREADS, exactly one issue
     per thread). Added `cp_async_8` inline PTX helper alongside the
     existing `cp_async_16`.
   - Both halves are committed as a single group per tile with
     `cp.async.commit_group`, so one `cp.async.wait_group<1>` suffices to
     wait on the consumed tile while the next is still in flight.

3. **Double-buffered pipeline.** Same shape as Stage 3b's value loop:
   prime tile 0 before entering, then for each tile issue
   `tile+1` → `wait_group<1>` → `__syncthreads()` → compute.
   The last tile drains with `wait_group<0>` so the in-flight queue is
   empty before the softmax passes start.

4. **Compute redistribution.** Each **warp** now computes **2 logits per
   tile** (BK=16 / NUM_WARPS=8). Per logit, a 32-lane warp-parallel dot
   reads 8 packed `bf16x2` from SMEM-ckv and 1 packed `bf16x2` from
   SMEM-kpe (covering the full D_CK + D_KP = 576 elements), accumulates
   in fp32, and reduces through `warp_reduce_sum`. Lane 0 writes
   `s_logits[k] = acc * sm_scale_log2e`. This keeps the compute in SMEM
   while HBM fetches the next tile in the background — the same
   latency-hiding shape that worked for the value loop.

5. **Truncation reuse.** The `n_valid_tiles` truncation already computed
   for Stage 3b is reused: the logit tile loop stops at the same bound,
   avoiding wasted `cp.async` issues on trailing `-1` entries for
   short-sequence traces. `s_logits[]` is pre-initialised to
   `-INFINITY` in a single block-wide pass before the logit loop so any
   untouched slot (skipped tile, invalid `g`) is a correct no-op for the
   downstream softmax `max`.

## Per-workload results (Modal B200 full sweep)

| workload | ref ms | 3b ms | 3c ms | Δ vs 3b | 3b speedup | **3c speedup** |
|----------|-------:|------:|------:|--------:|-----------:|---------------:|
| 02d6ae9c | 2.133  | 0.375 | 0.159 | **-57.5 %** | 4.84 ×  | **13.38 ×** |
| 05f6de65 | 1.149  | 0.072 | 0.037 | **-48.7 %** | 14.92 × | **31.23 ×** |
| 0a63b87b | 1.105  | 0.023 | 0.016 | **-27.7 %** | 47.15 × | **67.82 ×** |
| 0c23b10c | 0.950  | 0.016 | 0.012 | **-23.8 %** | 58.33 × | **77.48 ×** |
| 2207f0fd | 1.806  | 0.369 | 0.158 | **-57.2 %** | 4.61 ×  | **11.46 ×** |
| 232ed014 | 2.052  | 0.205 | 0.090 | **-56.0 %** | 8.91 ×  | **22.75 ×** |
| 38389961 | 1.870  | 0.205 | 0.090 | **-56.0 %** | 8.24 ×  | **20.73 ×** |
| 385742b2 | 2.081  | 0.199 | 0.088 | **-55.7 %** | 9.17 ×  | **23.65 ×** |
| 4c46a94b | 2.078  | 0.190 | 0.084 | **-55.7 %** | 9.59 ×  | **24.75 ×** |
| 5096e459 | 2.102  | 0.365 | 0.156 | **-57.3 %** | 5.02 ×  | **13.51 ×** |
| 564007ac | 1.935  | 0.375 | 0.160 | **-57.4 %** | 4.88 ×  | **12.12 ×** |
| 68d6817d | 1.825  | 0.018 | 0.014 | **-25.5 %** | 84.49 × | **132.81 ×** |
| 78b2e11c | 2.046  | 0.375 | 0.160 | **-57.4 %** | 4.85 ×  | **12.82 ×** |
| 7a389715 | 2.147  | 0.303 | 0.131 | **-56.9 %** | 5.97 ×  | **16.44 ×** |
| 9d4a5f21 | 1.136  | 0.016 | 0.012 | **-24.6 %** | 65.05 × | **91.85 ×** |
| 9f3f891b | 1.134  | 0.063 | 0.033 | **-48.6 %** | 16.81 × | **34.78 ×** |
| ae4219a9 | 1.960  | 0.375 | 0.160 | **-57.4 %** | 4.48 ×  | **12.27 ×** |
| b7668cfd | 1.139  | 0.020 | 0.016 | **-23.6 %** | 51.91 × | **72.73 ×** |
| d57eb9e1 | 1.778  | 0.369 | 0.158 | **-57.3 %** | 4.29 ×  | **11.29 ×** |
| ddfa9e34 | 1.670  | 0.303 | 0.131 | **-56.9 %** | 5.16 ×  | **12.79 ×** |
| e6b849f2 | 1.116  | 0.029 | 0.018 | **-35.9 %** | 37.12 × | **60.74 ×** |
| f77df5ce | 1.133  | 0.016 | 0.013 | **-20.9 %** | 64.86 × | **87.31 ×** |
| fc85411e | 1.091  | 0.016 | 0.012 | **-25.0 %** | 64.94 × | **88.74 ×** |
| **MEAN** | 1.628  | 0.187 | 0.083 | **-55.6 %** | 25.46 × | **41.45 ×** |

Two distinct speedup regimes:

- **Long-sequence (≥ 0.13 ms in Stage 3b):** -55 to -57 % latency. These
  traces spend the vast majority of time reading K/V, so replacing a serial
  dependent-load stream with a `cp.async` pipeline gives a near 2.25×
  speedup — in line with the value-projection experience in Stage 3b.
- **Short-sequence (≤ 0.03 ms in Stage 3b):** -21 to -36 % latency. Here
  the logit issue/compute overhead dominates. The new warp-reduce layout
  (32 threads cooperating on 1 logit instead of each thread owning 8 of
  its own) reduces per-block SMEM/register pressure and cuts the
  per-iteration cost even when truncation has already removed most of the
  HBM.

## vs FlashInfer TRT-LLM MLA decode (mean)

`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` (sweep of the same
23 workloads): **0.259 ms** mean. Stage 3c: **0.083 ms** — **3.13 × faster**.
Per-workload, our kernel is strictly faster on every single workload, with
short traces being 15–20 × faster.

## SMEM budget after Stage 3c

```
s_logits   [TOPK]            float   8 192 B
s_qn       [D_CK]            float   2 048 B
s_qp       [D_KP]            float     256 B
s_gidx     [TOPK]            int     8 192 B
s_reduce   [NUM_WARPS]       float      32 B
s_v_tile   [V_PIPE][BK][D_CK] bf16  32 768 B   (aliased as ckv tile in logit pass)
s_kpe_tile [V_PIPE][BK][D_KP] bf16   4 096 B   (new in Stage 3c)
-----------------------------------------------
TOTAL                                55 584 B  (~54.3 KiB)
```

Kernel still advertises 100 KiB via `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)`
and keeps `__launch_bounds__(THREADS, 2)` — two concurrent CTAs per SM are
comfortably achievable (228 KiB / SM on B200, headroom ≈ 117 KiB).

## Why it worked

The logit pass looked like the value pass with a different compute kernel:
dependent-load chains on K rows, 1 thread = 1 row, every 8 iters a branch
that kills ILP. Replacing the per-thread streamer with a **cooperative
tile loader** (256 threads × 16/8 B `cp.async` → a single 16 KiB ckv + 2 KiB
kpe tile in SMEM, double-buffered) and a **warp-cooperative dot** (32
lanes reducing over D_CK+D_KP for one logit) does three things at once:

- overlaps HBM reads with SMEM FMAs (same win as Stage 3b);
- collapses the 2 048-logit workload into 128 tile-parallel steps, each
  bottlenecked by warp-reduce (tens of cycles), not HBM (hundreds);
- removes the per-thread control-flow hazard that blocked compiler
  pipelining in the original dot — `if (g < 0) continue;` is now hoisted
  above the inner warp-reduce, with `-INFINITY` pre-init handling the
  masked output.

## Ideas not worth exploring (for this block of work)

- **Triple buffering (V_PIPE=3) in the logit pass.** With only ~35 KiB
  needed for ckv+kpe tiles at V_PIPE=2 we could afford a third buffer,
  but Stage 3b experiments showed the current scheduler already keeps
  a tile in flight whenever compute is shorter than HBM for this access
  pattern. Not worth the SMEM churn at this speedup.
- **`cp.async.bulk` / TMA.** Would need sm_100a TMA descriptors and is
  strictly a bigger rewrite; unclear win vs the per-warp issue pattern
  for 16 KiB tiles.
- **Fused logit + value compute.** Would let us write `p_k` and consume
  it in the same tile pass, but destroys the simple `s_logits[]` cache
  and forces either a two-pass softmax over a compressed representation
  or an online softmax inside the value tile loop — the latter was
  tried in Stage 2 attempt 1 and lost ~2× to `__syncthreads` barriers.
- **Shuffle-based warp-reduce replacement with `__reduce_sync`.** Already
  using `__shfl_xor_sync`, compiler emits identical SASS.

## Candidates for Stage 3d / a separate ceiling-push

- **Double-kv layout** (pack ckv & kpe into a single HBM layout →
  1 cp.async issue per row pair). Would cut issue count by ~10 % on
  the logit pass but requires changing the trace loader and may collide
  with FlashInfer's page layout.
- **NCU-driven occupancy tuning** (revisit register count; we can raise
  `__launch_bounds__` second argument to 3 once NCU confirms register
  headroom). Unlikely to beat the current 2.25× jump, but worth a sniff
  before calling the kernel "done".
