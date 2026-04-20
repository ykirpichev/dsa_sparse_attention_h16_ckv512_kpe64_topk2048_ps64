# Stage 2 results: packed bf162 output + fused divide

Branch: `stage2/fused-online-softmax`
Artifact: [`artifacts/modal_stage2_full_20260420T065116Z.json`](../artifacts/modal_stage2_full_20260420T065116Z.json)

All 23 workloads PASSED.

## Headline

**We now beat FlashInfer TRT-LLM MLA on the dataset mean** (230 µs vs 259 µs, 1.13× faster), and on **14 of 23 workloads**. On the short traces the margin is 4–5×; on the 8 large ones FI is still 1.35–1.65× ahead (its compute-bound flat-~260 µs profile).

| Metric | Baseline | Stage 1 | **Stage 2** | FI TRT-LLM |
|--------|---------:|--------:|------------:|-----------:|
| Mean latency | 503 µs | 289 µs | **230 µs** | 259 µs |
| Mean speedup vs definition reference | 5.13× | 8.38× | **11.61×** | 6.05× |
| Mean ratio ours / FI | 1.92× | 1.10× | **0.88×** | 1.00× |
| Workloads faster than FI | 0 / 23 | 5 / 23 | **14 / 23** | 23 / 23 |

Stage-2 speedup over Stage 1: **1.33× mean, 1.22× min, 1.54× max** (small traces benefit the most — they were instruction-count bound).

## What changed

Stage 2 additive changes on top of Stage 1 (`solution/cuda/mla_cuda.cu`):

1. **Packed `__nv_bfloat162` loads in the value projection** (hot loop): each thread now owns a single packed output slot (`d = 2*tid`, `2*tid+1`) and the inner `k` loop issues **1 packed GMEM load + 2 FMAs** per k instead of 2 scalar loads + 2 FMAs. Halves the load instruction count; within a warp the accesses become one aligned 128-byte sector.
2. **Packed `__nv_bfloat162` stores of the final output** (including the all-masked row fast path).
3. **Fold the softmax normalization into the output write**: we no longer run the `s_logits[k] *= inv` pass over shared memory — we keep `s_logits` as unnormalized probabilities and multiply by `1/sum` once per output lane at write time. One fewer SMEM pass, one fewer `__syncthreads`.

## What did *not* work (useful prior)

First Stage-2 attempt was **tiled online softmax with `K_tile` in SMEM** (FlashAttention-style, BK=32). It was correct but **~2× slower** than Stage 1 across the board. Diagnosis: the per-tile `__syncthreads` barriers serialize the memory pipeline that Stage 1 had kept maximally parallel (~500k outstanding GMEM loads at once across 256 threads × 2048 k's). Coalescing-per-warp was improved in the tiled version, but the tail-latency of each K tile load at the barrier dominated. Detailed numbers and analysis kept in this doc so we don't retry the same tile-SMEM shape without fixing the pipelining first (cp.async double-buffering is the natural next move).

## Per-workload (latency, µs)

| Workload | Baseline | Stage 1 | **Stage 2** | FI | **S2/S1** | **S2/FI** |
|----------|---------:|--------:|------------:|---:|----------:|----------:|
| `0c23b10c` |  98.3 |  75.9 |  **49.2** | 255.6 | 1.54× | 0.19× |
| `9d4a5f21` | 108.8 |  76.1 |  **51.2** | 249.8 | 1.49× | 0.20× |
| `b7668cfd` | 116.8 |  78.7 |  **53.3** | 250.0 | 1.48× | 0.21× |
| `0a63b87b` | 118.1 |  80.5 |  **55.3** | 250.2 | 1.46× | 0.22× |
| `05f6de65` | 283.4 | 143.3 | **107.7** | 259.1 | 1.33× | 0.42× |
| `fc85411e` | 108.4 |  77.2 |  **51.2** | 252.0 | 1.51× | 0.20× |
| `e6b849f2` | 121.3 |  85.8 |  **59.4** | 250.0 | 1.45× | 0.24× |
| `9f3f891b` | 260.5 | 132.6 | **100.3** | 258.9 | 1.32× | 0.39× |
| `f77df5ce` | 108.6 |  75.8 |  **51.2** | 249.8 | 1.48× | 0.20× |
| `385742b2` | 555.7 | 301.0 | **241.8** | 283.3 | 1.25× | 0.85× |
| `4c46a94b` | 525.2 | 290.8 | **233.5** | 261.0 | 1.25× | 0.89× |
| `38389961` | 567.1 | 310.6 | **247.8** | 262.0 | 1.25× | 0.95× |
| `02d6ae9c` | 915.2 | 523.4 | **430.1** | 263.3 | 1.22× | 1.63× |
| `ddfa9e34` | 766.6 | 434.1 | **352.3** | 260.7 | 1.23× | 1.35× |
| `78b2e11c` | 913.3 | 523.8 | **430.1** | 262.8 | 1.22× | 1.64× |
| `68d6817d` | 116.7 |  77.3 |  **52.8** | 252.7 | 1.46× | 0.21× |
| `564007ac` | 923.0 | 528.6 | **430.2** | 262.9 | 1.23× | 1.64× |
| `ae4219a9` | 915.9 | 526.0 | **430.1** | 261.0 | 1.22× | 1.65× |
| `232ed014` | 568.9 | 311.3 | **249.9** | 262.0 | 1.25× | 0.95× |
| `7a389715` | 770.7 | 437.1 | **354.3** | 263.2 | 1.23× | 1.35× |
| `5096e459` | 898.9 | 516.0 | **419.5** | 263.5 | 1.23× | 1.59× |
| `d57eb9e1` | 903.6 | 516.1 | **421.9** | 261.1 | 1.22× | 1.62× |
| `2207f0fd` | 905.3 | 518.1 | **422.9** | 262.0 | 1.23× | 1.61× |

Correctness preserved: max abs error ≤ 0.0078 across all workloads.

## Remaining gap to FI (large traces)

On the 8 largest traces (`02d6ae9c`, `78b2e11c`, `564007ac`, `ae4219a9`, `5096e459`, `d57eb9e1`, `2207f0fd`, and two 0.35 ms traces) we're at ~420–430 µs while FI is flat at ~260 µs. At this size the block latency is bound by scalar FMA throughput of the value projection (512 d × 2048 k per block). Closing this ~1.6× gap needs a structural change: **tensor-core matmul** (wgmma / tcgen05.mma for the `P @ V` block) or **FlashAttention-style tiled fusion with cp.async double-buffering** (the shape Stage 2 v1 tried — the fusion was right, the pipelining wasn't).

## Next

Stage 3 candidates (ordered by likely payoff / difficulty):
1. **cp.async double-buffered tile load** for the value projection (retry Stage 2 v1 with pipelining so the tile barrier no longer stalls the memory system).
2. **wgmma / tcgen05.mma for `P @ V`** on the large traces: one `bf16 @ bf16 → fp32` matmul of shape `(D_CK=512) × (TOPK=2048) × (1 head)` per block.
3. **Head tiling** (2 heads per block sharing the K tile) so the fusion ratio improves to 2× per K read.
