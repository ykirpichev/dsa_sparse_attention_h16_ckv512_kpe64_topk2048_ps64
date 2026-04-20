# Stage 1 results: batched low-risk optimizations

Branch: `stage1/batched-optimizations`
Artifact: [`artifacts/modal_stage1_full_20260420T055813Z.json`](../artifacts/modal_stage1_full_20260420T055813Z.json)

All 23 workloads PASSED.

## Aggregate

| Metric | Baseline | **Stage 1** | FlashInfer TRT-LLM |
|--------|---------:|------------:|-------------------:|
| Mean solution latency | 503 µs | **289 µs** | 259 µs |
| Mean speedup vs definition reference | 5.13× | **8.38×** | 6.05× |
| Mean ratio ours / FI | 1.92× | **1.10×** | 1.00× |
| Max ratio ours / FI | 3.51× | **2.02×** | 1.00× |
| Min ratio ours / FI | 0.38× | **0.30×** | 1.00× |

Stage-1 speedup over our own baseline: **1.67× mean, 1.29× min, 1.98× max.**

## What changed in [`solution/cuda/mla_cuda.cu`](../solution/cuda/mla_cuda.cu)

1. **Packed BF16 loads** (`__nv_bfloat162`, `__bfloat1622float2`). Logit dot products now issue half as many memory instructions.
2. **Cached `sparse_idx` in shared memory** (`s_gidx[TOPK]`). Value-projection pass no longer re-reads `sparse_indices` for every `d`.
3. **Warp + block parallel softmax** (`__shfl_xor_sync` reductions for max and sum). No more `tid == 0` serial stall.
4. **`exp2f` with fused `sm_scale · log2(e)`**. Logits live in base-2 from the start; LSE is `mx + log2f(sum)` — removes the final `/log(2)`.
5. **`__launch_bounds__(256, 2)`** and `cudaFuncSetAttribute(... 100 KiB ...)` via `std::call_once`. Ready to opt into larger SMEM for Stage 2/3 tile sizes without touching launch code again.

Shared-memory footprint: ~18.3 KiB (well under the 100 KiB enabled attribute; plenty of room for future KV tile buffers).

## Per-workload (latency, µs)

| Workload | Baseline | **Stage 1** | FI | Stage 1 speedup vs baseline |
|----------|---------:|------------:|---:|----------------------------:|
| `0c23b10c` | 98.3 | 75.9 | 255.6 | 1.29× |
| `9d4a5f21` | 108.8 | 76.1 | 249.8 | 1.43× |
| `b7668cfd` | 116.8 | 78.7 | 250.0 | 1.48× |
| `0a63b87b` | 118.1 | 80.5 | 250.2 | 1.47× |
| `05f6de65` | 283.4 | 143.3 | 259.1 | 1.98× |
| `fc85411e` | 108.4 | 77.2 | 252.0 | 1.40× |
| `e6b849f2` | 121.3 | 85.8 | 250.0 | 1.41× |
| `9f3f891b` | 260.5 | 132.6 | 258.9 | 1.96× |
| `f77df5ce` | 108.6 | 75.8 | 249.8 | 1.43× |
| `385742b2` | 555.7 | 301.0 | 283.3 | 1.85× |
| `4c46a94b` | 525.2 | 290.8 | 261.0 | 1.81× |
| `38389961` | 567.1 | 310.6 | 262.0 | 1.83× |
| `02d6ae9c` | 915.2 | 523.4 | 263.3 | 1.75× |
| `ddfa9e34` | 766.6 | 434.1 | 260.7 | 1.77× |
| `78b2e11c` | 913.3 | 523.8 | 262.8 | 1.74× |
| `68d6817d` | 116.7 | 77.3 | 252.7 | 1.51× |
| `564007ac` | 923.0 | 528.6 | 262.9 | 1.75× |
| `ae4219a9` | 915.9 | 526.0 | 261.0 | 1.74× |
| `232ed014` | 568.9 | 311.3 | 262.0 | 1.83× |
| `7a389715` | 770.7 | 437.1 | 263.2 | 1.76× |
| `5096e459` | 898.9 | 516.0 | 263.5 | 1.74× |
| `d57eb9e1` | 903.6 | 516.1 | 261.1 | 1.75× |
| `2207f0fd` | 905.3 | 518.1 | 262.0 | 1.75× |

Numeric accuracy preserved: max abs error ≤ 0.0078 across all workloads (well inside bench tolerance 1e-2).

## Remaining gap to FlashInfer TRT-LLM

On the **8 large traces** (385742b2 … 2207f0fd), we still land at ~290–530 µs vs FI's ~260 µs flat. Expected: these are compute-bound on scalar FMAs, which is exactly what the Stage 2 / Stage 3 items (online softmax + fused projection, head tiling, tensor cores, TMA) are meant to address.

## Operational fixes applied after "stuck" incident

- `SMOKE_WORKLOAD_COUNT_DEFAULT = 10` (was 2) and smoke `iterations=5` (was 3) so future smoke tests catch the large-workload tail, not just the 2 random small ones.
- Future Modal full runs: the bench itself completes in ~3–4 min; the Modal CLI can then hang for several minutes waiting for the final log drain (known "Timed out waiting for final app logs" behavior). Poll `artifacts/modal_*_full.json` — once it exists, the run is done; the CLI can be killed safely.

## Next

Stage 2 (online softmax + fused value projection, head-tiling) is the big lever for large traces, where FI is still ~2× ahead. Stage 3 (wgmma / tmem / TMA) is where we can match FI's flat-latency profile.
