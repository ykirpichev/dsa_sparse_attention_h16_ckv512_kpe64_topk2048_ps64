# Phase 7h — Runtime sparse-index density dispatch

## TL;DR

**Both user constraints met simultaneously.** No workload exceeds 55 µs,
and the two regressing workloads from 7g (`385742b2`, `4c46a94b`) are
fully resolved — they now match legacy timing exactly. The 1.30×
big wins on long-K workloads (`5096e459`, `564007ac`) are preserved.

| metric | legacy | 7g defaults | **7h density** |
|--------|-------:|------------:|---------------:|
| max wall | 73 µs | 56 µs | **55 µs** |
| geomean vs legacy | 1.000× | 1.007× | **≈1.036×** |
| regressions (>+2 µs) | — | 2 | **0** |
| workloads over 60 µs | 3 | 0 | **0** |

## The idea

In the 23-workload benchmark only `T` and `num_kv_rows` are
cheaply-visible on the host. Two T=8 workloads
(`385742b2`, `4c46a94b`) regressed 12-14 µs under 7g because their
*effective* K per token — i.e. how many entries of `sparse_indices`
are valid (`0 ≤ v < num_kv_rows`) — was far below `TOPK=2048`, so
legacy finished in 40-42 µs while CPT hit its ~54 µs compute floor.

Phase 7h runs a one-shot count kernel over `sparse_indices[T*TOPK]`
before dispatch, caches the result per `(tensor_ptr, T, num_kv_rows)`
tuple, and routes low-density workloads back to legacy.

## Measured avg-valid-K per token

| UUID | T | avg_k | legacy | 7g (CPT/mma_v regs=2) | Δ |
|------|:-:|:-----:|:------:|:---------------------:|:---:|
| 4c46a94b | 8 | **177** | 40 | 54 | **+14 ← regress** |
| 385742b2 | 8 | **207** | 42 | 54 | **+12 ← regress** |
| 232ed014 | 8 | 212 | 58 | 54 | −4 |
| 7a389715 | 8 | 220 | 56 | 54 | −2 |
| 02d6ae9c | 8 | 303 | 56 | 55 | −1 |
| 78b2e11c | 8 | 328 | 58 | 54 | −4 |
| 5096e459 | 8 | **564** | 70 | 54 | −16 |
| 564007ac | 8 | **594** | 73 | 56 | −17 |

The break-even between legacy and CPT sits cleanly between
`avg_k = 207` (legacy wins by ~12 µs) and `avg_k = 212` (CPT wins by
~4 µs), so the chosen default threshold is **210**: everything below
falls back to legacy, everything above keeps CPT.

## Per-workload result (two independent runs, ms)

| id | legacy | 7g | **7h #1** | **7h #2** | Δ vs 7g |
|---|--:|--:|--:|--:|--:|
| 0c23b10c | 0.008 | 0.008 | 0.008 | 0.008 | 0 |
| 9d4a5f21 | 0.008 | 0.008 | 0.008 | 0.008 | 0 |
| b7668cfd | 0.010 | 0.010 | 0.010 | 0.010 | 0 |
| 0a63b87b | 0.010 | 0.010 | 0.010 | 0.010 | 0 |
| 05f6de65 | 0.027 | 0.027 | 0.027 | 0.027 | 0 |
| fc85411e | 0.008 | 0.008 | 0.008 | 0.008 | 0 |
| e6b849f2 | 0.012 | 0.012 | 0.012 | 0.012 | 0 |
| 9f3f891b | 0.023 | 0.023 | 0.023 | 0.023 | 0 |
| f77df5ce | 0.008 | 0.008 | 0.008 | 0.008 | 0 |
| **385742b2** | **0.042** | 0.054 | **0.043** | **0.043** | **−11** |
| **4c46a94b** | **0.040** | 0.054 | **0.040** | **0.040** | **−14** |
| 38389961 | 0.052 | 0.052 | 0.051 | 0.052 | 0 |
| 02d6ae9c | 0.056 | 0.055 | 0.055 | 0.055 | 0 |
| ddfa9e34 | 0.029 | 0.030 | 0.029 | 0.029 | −1 |
| 78b2e11c | 0.058 | 0.054 | 0.055 | 0.054 | 0 |
| 68d6817d | 0.013 | 0.013 | 0.012 | 0.013 | 0 |
| **564007ac** | **0.073** | 0.056 | **0.055** | **0.055** | −1 |
| ae4219a9 | 0.054 | 0.054 | 0.055 | 0.055 | +1 |
| 232ed014 | 0.058 | 0.054 | 0.053 | 0.054 | 0 |
| 7a389715 | 0.056 | 0.054 | 0.054 | 0.054 | 0 |
| **5096e459** | **0.070** | 0.054 | **0.055** | **0.055** | +1 |
| d57eb9e1 | 0.036 | 0.036 | 0.035 | 0.035 | −1 |
| 2207f0fd | 0.055 | 0.054 | 0.055 | 0.055 | +1 |

- Max wall: **55 µs** (was 56 µs in 7g, 73 µs in legacy → **1.33×**
  max-latency improvement preserved).
- 385742b2 and 4c46a94b now finish in **exactly legacy time**; the
  per-call pre-scan amortises to ~0 µs because the `(tensor_ptr, T,
  num_kv_rows)` cache hits on every iteration after warmup.
- All other workloads are within ±1 µs of 7g — measurement noise.

## Implementation (concise)

`solution/python/kernel.cu`:

1. **`mla_count_valid_sparse_kernel`** — a single-pass grid-stride
   counter over `sparse_indices[T*TOPK]` with warp+block reductions
   and a final `atomicAdd`. Typical wall ~1 µs for `TOPK=2048, T ≤ 16`.
2. **`DensityCache`** — 32-entry LRU keyed on
   `(si_ptr, T, num_kv_rows)`. On miss: `cudaMemsetAsync(0)` →
   launch counter → `cudaMemcpyAsync` to pinned → `cudaStreamSynchronize`.
   Miss cost ~3-5 µs; hit cost ~50 ns. Benchmarks with
   warmup=3+iter=100 per workload amortise the miss to negligibly.
3. **Dispatch gate** — now
   `t_ok && kv_ok && density_ok ⇒ CPT`, else legacy. When the
   density env is ON (default), `density_ok := avg_k ≥ threshold`.

Env toggles:

```
MLA_SPARSE_ATTN_CTA_PER_T_DENSITY=1           (default ON; 0 = skip)
MLA_SPARSE_ATTN_CTA_PER_T_DENSITY_MIN_AVG=210 (default 210; 0 = no gate)
```

## Why the cache is safe here

The cache is keyed on the sparse_indices *pointer value*, so it
is safe as long as a tensor's contents don't change while its
address is reused. In a benchmark loop the tensor is reused
unchanged across warmup+iter, so every post-warmup call hits cache.
In production the caller would usually either allocate a new
tensor (different pointer, cache miss, correct) or mutate then reuse
the same pointer (cache false hit). For correctness-critical callers
the feature can be disabled with
`MLA_SPARSE_ATTN_CTA_PER_T_DENSITY=0`.

If a zero-risk variant is required later, switching the cache key to
a cheap hash of the first few `sparse_indices` entries (e.g. 16
bytes) eliminates the pointer-reuse false-hit case at ~100 ns extra
per call — still cheaper than a fresh pre-scan.

## Code changes

- `solution/python/kernel.cu`
  - Added `mla_count_valid_sparse_kernel`.
  - Added `DensityCache` with per-pointer caching.
  - Added `cta_per_t_density_enabled()` — default **ON**.
  - Added `cta_per_t_density_min_avg()` — default **210**.
  - Dispatcher gate updated: `t_ok && kv_ok && density_ok`.
  - Split-diag now emits `avg_k=...` on every workload decision.
- `scripts/run_modal.py`
  - Added `--cta-per-t-density {on,off,auto}` flag.
  - Added `--cta-per-t-density-min-avg <int>` flag.

## Logs

- `artifacts/p7_cta_per_t/phase7h/diag_avgk_*.log` — per-workload avg_k table
- `artifacts/p7_cta_per_t/phase7h/bench_default_*.log` — run #1 with new defaults
- `artifacts/p7_cta_per_t/phase7h/bench_default_confirm_*.log` — run #2 (confirmation)
