# DSA Sparse-MLA — Final Per-Workload Comparison

**Benchmark:** B200 (sm_100a), BF16, 23 workloads × 10 repeats = 230 samples per solution. Values are per-workload medians across 10 repeats.

**Raw data:** `artifacts/final/final_compare_20260423T131005Z.json`

## Aggregate

| Solution | Mean wall (µs) | Max wall (µs) | Geomean × vs PT | Passed |
|---|---:|---:|---:|:---:|
| PyTorch reference | 1470.4 | 1837.8 | 1.00× | — |
| FlashInfer | 258.86 | 264.4 | 5.51× | 230/230 |
| CUDA `v5_fused` | 69.81 | 135.2 | 31.05× | 230/230 |
| **`v5_7i` (selected best)** | **33.63** | **59.6** | **54.49×** | **230/230** |

Relative: `v5_7i` = **9.88× faster than FlashInfer**, **1.75× faster than `v5_fused`**, **54.5× faster than PyTorch** (geomean).

## Per-workload comparison (median wall-time, µs)

| Workload | PyTorch | FlashInfer | `v5_fused` | **`v5_7i`** | 7i vs PT | 7i vs FI | 7i vs v5_fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| `02d6ae9c64…` | 1816.3 | 264.4 | 135.1 | **55.3** | 32.8× | 4.78× | 2.44× |
| `05f6de657d…` | 1065.4 | 259.8 | 30.7 | **26.6** | 40.0× | 9.76× | 1.15× |
| `0a63b87bb2…` | 1057.4 | 251.7 | 14.3 | **10.2** | 103.3× | 24.58× | 1.40× |
| `0c23b10c7b…` | 932.7 | 251.0 | 10.2 | **8.2** | 113.9× | 30.64× | 1.25× |
| `2207f0fdc9…` | 1712.2 | 262.5 | 131.1 | **53.5** | 32.0× | 4.91× | 2.45× |
| `232ed014ba…` | 1823.7 | 263.1 | 75.8 | **55.6** | 32.8× | 4.73× | 1.36× |
| `3838996164…` | 1697.3 | 263.4 | 75.8 | **51.2** | 33.1× | 5.14× | 1.48× |
| `385742b271…` | 1821.9 | 262.8 | 73.7 | **41.5** | 43.9× | 6.33× | 1.78× |
| `4c46a94ba2…` | 1816.2 | 261.6 | 71.6 | **39.0** | 46.6× | 6.71× | 1.84× |
| `5096e459ce…` | 1826.6 | 264.0 | 131.1 | **58.2** | 31.4× | 4.54× | 2.25× |
| `564007ac35…` | 1837.8 | 264.2 | 133.1 | **59.6** | 30.9× | 4.43× | 2.23× |
| `68d6817dcf…` ⚠ | 1563.2 | 252.2 | **12.3** | 14.2 | 109.7× | 17.70× | **0.86×** |
| `78b2e11c30…` | 1819.7 | 264.2 | 135.2 | **58.6** | 31.1× | 4.51× | 2.31× |
| `7a389715dc…` | 1827.5 | 264.2 | 108.6 | **55.3** | 33.1× | 4.78× | 1.96× |
| `9d4a5f2126…` | 1058.2 | 251.0 | 12.3 | **8.2** | 129.2× | 30.64× | 1.50× |
| `9f3f891bfb…` | 1066.0 | 259.4 | 26.6 | **22.5** | 47.3× | 11.51× | 1.18× |
| `ae4219a95f…` | 1688.0 | 262.7 | 133.2 | **53.3** | 31.7× | 4.93× | 2.50× |
| `b7668cfd19…` | 1057.5 | 251.5 | 14.3 | **10.2** | 103.3× | 24.56× | 1.40× |
| `d57eb9e19f…` | 1587.2 | 262.9 | 131.1 | **34.8** | 45.5× | 7.54× | **3.76×** |
| `ddfa9e340b…` | 1567.0 | 262.6 | 108.6 | **28.7** | 54.7× | 9.16× | **3.79×** |
| `e6b849f290…` | 1058.5 | 251.9 | 16.4 | **12.3** | 86.1× | 20.50× | 1.33× |
| `f77df5ce21…` | 1059.2 | 251.3 | 12.3 | **8.2** | 129.3× | 30.68× | 1.50× |
| `fc85411e25…` | 1059.5 | 251.6 | 12.3 | **8.2** | 129.4× | 30.73× | 1.50× |

**Notes:**

* **⚠ `68d6817dcf…`** — the only workload where `v5_7i` is slightly slower than `v5_fused` (14.2 vs 12.3 µs, −14 %). This is a tiny-T workload where the CPT fast path carries extra fixed overhead relative to the legacy G=1 path; the density pre-scan correctly avoids the worst regressions but this one still dispatches to CPT. Absolute delta is only 1.9 µs and both remain well under 60 µs. The geomean still favors `v5_7i` by 1.75×.
* **Biggest wins vs `v5_fused`** come from workloads `d57eb9e19f…` and `ddfa9e340b…` (3.76–3.79×), which are medium-T traces where split-K with G=16 unlocks full SM occupancy.
* **Biggest wins vs FlashInfer**: the small-T workloads (`0c23b10c7b…`, `fc85411e25…`, `9d4a5f21…`, etc.), 25–31× each. FlashInfer's decode overhead dominates their sub-10-µs runtime.

## Correctness

All three kernels pass 230/230 in this run. Max `abs_err` observed over all 230 samples:

| Solution | max `abs_err` | Deterministic? |
|---|---:|---|
| FlashInfer | 0.03125 | No (upstream variance) |
| `v5_fused` | 0.015625 | Yes (G=1, single bf16 quantization) |
| **`v5_7i`** | **0.015625** | **Yes (fp32 split-K partials)** |

## Bottom line

`v5_7i` is the best shipping choice: **54.5× vs PyTorch, 9.9× vs FlashInfer, 1.75× vs our own CUDA v5_fused**, with deterministic numerics and a max wall-time of 59.6 µs (within the 60 µs SLA).
