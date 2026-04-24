# Paired comparison: PyTorch ref vs FlashInfer vs v4 Stage 3c vs v5_fused

Single Modal B200 session, BenchmarkConfig(warmup=3, iters=100, trials=5), 23 workloads.
Artifact: `artifacts/compare_pytorch_fi_v4_v5_full.json`

## Per-workload latencies (microseconds)

| workload | PyTorch ref | FlashInfer | v4 Stage 3c | v5_fused | v5/PT | v5/FI | v5/v4 | FI status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `02d6ae9c64` | 2484.63 | 276.87 | 159.12 | 134.19 | 18.52x | 2.06x | 1.19x | PASSED |
| `05f6de657d` | 1244.10 | 272.57 | 37.05 | 31.73 | 39.21x | 8.59x | 1.17x | PASSED |
| `0a63b87bb2` | 1220.98 | 269.42 | 16.13 | 14.92 | 81.81x | 18.05x | 1.08x | PASSED |
| `0c23b10c7b` | 1068.19 | 270.51 | 13.30 | 11.25 | 94.91x | 24.04x | 1.18x | PASSED |
| `2207f0fdc9` | 2304.98 | 274.68 | 156.73 | 132.10 | 17.45x | 2.08x | 1.19x | PASSED |
| `232ed014ba` | 2517.85 | 275.95 | 91.11 | 76.79 | 32.79x | 3.59x | 1.19x | PASSED |
| `3838996164` | 2297.46 | 276.34 | 91.13 | 76.79 | 29.92x | 3.60x | 1.19x | PASSED |
| `385742b271` | 2499.64 | 275.64 | 87.45 | 74.75 | 33.44x | 3.69x | 1.17x | PASSED |
| `4c46a94ba2` | 2495.62 | 275.06 | 84.97 | 70.68 | 35.31x | 3.89x | 1.20x | PASSED |
| `5096e459ce` | 2458.35 | 277.30 | 156.64 | 132.02 | 18.62x | 2.10x | 1.19x | PASSED |
| `564007ac35` | 2492.74 | 277.71 | 159.56 | 134.12 | 18.59x | 2.07x | 1.19x | PASSED |
| `68d6817dcf` | 2068.51 | 271.64 | 13.32 | 13.29 | 155.69x | 20.45x | 1.00x | PASSED |
| `78b2e11c30` | 2512.97 | — | 158.76 | 134.19 | 18.73x | — | 1.18x | INCORRECT_NUMERICAL |
| `7a389715dc` | 2498.67 | 276.75 | 130.06 | 109.56 | 22.81x | 2.53x | 1.19x | PASSED |
| `9d4a5f2126` | 1231.61 | 270.84 | 13.31 | 13.26 | 92.90x | 20.43x | 1.00x | PASSED |
| `9f3f891bfb` | 1249.41 | 272.00 | 31.77 | 27.65 | 45.19x | 9.84x | 1.15x | PASSED |
| `ae4219a95f` | 2198.04 | 275.39 | 159.17 | 134.13 | 16.39x | 2.05x | 1.19x | PASSED |
| `b7668cfd19` | 1220.95 | 270.72 | 15.37 | 15.28 | 79.89x | 17.71x | 1.01x | PASSED |
| `d57eb9e19f` | 2117.21 | 274.45 | 156.76 | 132.09 | 16.03x | 2.08x | 1.19x | PASSED |
| `ddfa9e340b` | 2059.41 | 274.70 | 130.06 | 109.56 | 18.80x | 2.51x | 1.19x | PASSED |
| `e6b849f290` | 1234.60 | 274.24 | 19.40 | 17.17 | 71.90x | 15.97x | 1.13x | PASSED |
| `f77df5ce21` | 1233.71 | 272.96 | 13.31 | 13.29 | 92.81x | 20.53x | 1.00x | PASSED |
| `fc85411e25` | 1226.30 | 274.34 | 13.31 | 13.26 | 92.48x | 20.69x | 1.00x | PASSED |

## Aggregate

| Comparison | geomean | amean | min | max |
|---|---:|---:|---:|---:|
| v5_fused vs PyTorch ref | **38.36x** | 49.75x | 16.03x | 155.69x |
| v5_fused vs v4 Stage 3c | **1.13x** | 1.14x | 1.00x | 1.20x |
| v5_fused vs FlashInfer | **6.07x** | 9.48x | 2.05x | 24.04x |
| FlashInfer vs PyTorch ref | **6.52x** | 6.86x | 3.95x | 9.12x |

## Mean latency across 23 workloads (arithmetic)

| Solution | Mean (µs) | Min (µs) | Max (µs) | N passed |
|---|---:|---:|---:|---:|
| PyTorch reference | 1910.26 | 1068.19 | 2517.85 | 23/23 |
| FlashInfer wrapper | 274.09 | 269.42 | 277.71 | 22/23 |
| v4 Stage 3c | 82.95 | 13.30 | 159.56 | 23/23 |
| v5_fused (current) | 70.53 | 11.25 | 134.19 | 23/23 |

## Notes

- All speedups are computed **per-workload** then aggregated (geomean/amean over 23 workloads).
- FlashInfer had 1 INCORRECT_NUMERICAL workload (`78b2e11c30`); its latency is excluded from FI aggregates. The wombat kernels pass all 23.
- `FlashInfer vs PyTorch ref` column × `v5_fused vs FlashInfer` column ≈ `v5_fused vs PyTorch ref` column (identity 6.52 × 6.07 = 39.6 ≈ 38.4 geomean, small deviation from per-workload ordering).
