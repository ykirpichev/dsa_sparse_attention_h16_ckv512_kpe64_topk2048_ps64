# Paired comparison (noise-checked), 5 repeats

Single Modal B200 session, BenchmarkConfig(warmup=3, iters=100, trials=5),
23 workloads × 5 independent benchmark invocations inside the same session.

Artifacts:
- `artifacts/compare_pytorch_fi_v4_v5_5repeats.json` — raw (all 5 repeats)
- `artifacts/compare_pytorch_fi_v4_v5_5repeats.log` — full run log

## Per-workload latency (µs): mean ± stddev across 5 repeats (CV%)

| workload | PyTorch ref | FlashInfer | v4 Stage 3c | v5_fused |
|---|---:|---:|---:|---:|
| `02d6ae9c64` | 1938.86 ± 14.57 (0.8%) | 267.15 ± 0.82 (0.3%) | 159.40 ± 0.24 (0.1%) | 133.75 ± 0.24 (0.2%) |
| `05f6de657d` | 1097.49 ± 5.03 (0.5%) | 256.85 ± 0.62 (0.2%) | 36.83 ± 0.02 (0.1%) | 30.84 ± 0.25 (0.8%) |
| `0a63b87bb2` | 1092.11 ± 6.87 (0.6%) | 254.13 ± 0.84 (0.3%) | 16.34 ± 0.04 (0.2%) | 14.31 ± 0.03 (0.2%) |
| `0c23b10c7b` | 949.86 ± 2.05 (0.2%) | 254.35 ± 0.74 (0.3%) | 12.36 ± 0.22 (1.8%) | 10.88 ± 0.27 (2.5%) |
| `2207f0fdc9` | 1840.33 ± 26.61 (1.4%) | 263.98 ± 2.82 (1.1%) | 157.55 ± 0.04 (0.0%) | 131.22 ± 0.05 (0.0%) |
| `232ed014ba` | 1975.01 ± 28.66 (1.5%) | 264.86 ± 2.13 (0.8%) | 90.33 ± 0.25 (0.3%) | 76.01 ± 0.25 (0.3%) |
| `3838996164` | 1814.78 ± 35.53 (2.0%) | 264.85 ± 1.38 (0.5%) | 90.36 ± 0.28 (0.3%) | 76.13 ± 0.27 (0.4%) |
| `385742b271` | 1948.81 ± 22.05 (1.1%) | 265.00 ± 0.97 (0.4%) | 87.84 ± 0.16 (0.2%) | 73.91 ± 0.27 (0.4%) |
| `4c46a94ba2` | 1946.01 ± 25.30 (1.3%) | 262.82 ± 2.35 (0.9%) | 84.10 ± 0.22 (0.3%) | 71.43 ± 0.15 (0.2%) |
| `5096e459ce` | 1976.10 ± 27.84 (1.4%) | 267.70 ± 0.97 (0.4%) | 155.62 ± 0.01 (0.0%) | 131.02 ± 0.02 (0.0%) |
| `564007ac35` | 1965.23 ± 22.93 (1.2%) | 267.12 ± 1.11 (0.4%) | 159.66 ± 0.03 (0.0%) | 133.19 ± 0.22 (0.2%) |
| `68d6817dcf` | 1660.09 ± 16.52 (1.0%) | 254.31 ± 0.93 (0.4%) | 14.16 ± 0.12 (0.9%) | 12.36 ± 0.22 (1.8%) |
| `78b2e11c30` | 1950.85 ± 21.39 (1.1%) | 267.21 ± 1.01 (0.4%) | 159.52 ± 0.14 (0.1%) | 133.68 ± 0.27 (0.2%) |
| `7a389715dc` | 1954.75 ± 19.38 (1.0%) | 267.54 ± 0.90 (0.3%) | 130.57 ± 0.31 (0.2%) | 108.80 ± 0.22 (0.2%) |
| `9d4a5f2126` | 1094.32 ± 5.61 (0.5%) | 253.91 ± 0.83 (0.3%) | 13.60 ± 0.55 (4.0%) | 12.32 ± 0.15 (1.2%) |
| `9f3f891bfb` | 1101.29 ± 6.26 (0.6%) | 256.66 ± 0.60 (0.2%) | 32.69 ± 0.05 (0.2%) | 26.83 ± 0.21 (0.8%) |
| `ae4219a95f` | 1816.46 ± 22.66 (1.2%) | 263.38 ± 1.72 (0.7%) | 159.58 ± 0.12 (0.1%) | 133.24 ± 0.25 (0.2%) |
| `b7668cfd19` | 1091.99 ± 4.77 (0.4%) | 254.06 ± 0.81 (0.3%) | 16.29 ± 0.03 (0.2%) | 14.33 ± 0.07 (0.5%) |
| `d57eb9e19f` | 1693.73 ± 21.90 (1.3%) | 263.87 ± 1.50 (0.6%) | 157.57 ± 0.06 (0.0%) | 131.24 ± 0.22 (0.2%) |
| `ddfa9e340b` | 1664.24 ± 22.22 (1.3%) | 263.76 ± 1.74 (0.7%) | 129.99 ± 0.31 (0.2%) | 108.92 ± 0.29 (0.3%) |
| `e6b849f290` | 1093.23 ± 5.75 (0.5%) | 254.25 ± 0.73 (0.3%) | 18.44 ± 0.08 (0.4%) | 16.34 ± 0.03 (0.2%) |
| `f77df5ce21` | 1093.42 ± 6.18 (0.6%) | 253.83 ± 0.82 (0.3%) | 13.81 ± 0.49 (3.6%) | 12.37 ± 0.18 (1.5%) |
| `fc85411e25` | 1094.83 ± 7.03 (0.6%) | 254.32 ± 0.70 (0.3%) | 12.72 ± 0.25 (1.9%) | 12.31 ± 0.14 (1.1%) |

## Pairwise speedups (per-workload mean over 5 repeats, then aggregated over 23 workloads)

| comparison | geomean | amean | min | max | N workloads |
|---|---:|---:|---:|---:|---:|
| **v5_fused vs v4 Stage 3c** | **1.17×** | 1.17× | 1.03× | 1.22× | 23 |
| **v5_fused vs FlashInfer wrapper** | **5.66×** | 9.04× | 1.98× | 23.39× | 23 |
| **v5_fused vs PyTorch reference** | **32.65×** | 44.35× | 12.91× | 134.33× | 23 |
| FlashInfer vs PyTorch reference | 5.77× | 5.95× | 3.73× | 7.46× | 23 |

## Mean latency across 23 workloads (mean of per-workload mean across 5 repeats)

| solution | mean (µs) | min (µs) | max (µs) |
|---|---:|---:|---:|
| v5_fused (current) | 69.80 | 10.88 | 133.75 |
| v4 Stage 3c | 83.01 | 12.36 | 159.66 |
| FlashInfer wrapper | 260.69 | 253.83 | 267.70 |
| PyTorch reference | 1558.86 | 949.86 | 1976.10 |

## Noise level (coefficient of variation across 5 repeats, mean over 23 workloads)

| solution | mean CV% | median CV% | max CV% |
|---|---:|---:|---:|
| FlashInfer wrapper | 0.45% | 0.36% | 1.07% |
| v4 Stage 3c | 0.66% | 0.24% | 4.04% |
| v5_fused | 0.57% | 0.26% | 2.48% |
| PyTorch reference | 0.96% | 0.99% | 1.96% |

## Interpretation

- **Intra-session measurement noise is low (< 1 % CV for most cells)** — the 5 repeats agree tightly within one Modal session.
- **v5_fused beats v4 Stage 3c on every workload**, geomean **1.17×** (range 1.03× – 1.22×). Tiny-K workloads (~12 µs) are already launch-bound, so v5 matches v4 there (~1.0×); all the win comes from the compute-bound shapes.
- **v5_fused is ≈5.66× faster than the FlashInfer wrapper** (geomean).
- Speedup vs PyTorch ref shows larger session-to-session variance than vs FI/v4, because the PyTorch ref runs for ~1–2 ms (longer kernels are more sensitive to GPU state/L2 warm-up). The 32.65× geomean vs PyTorch is therefore the conservative number across today's two sessions; in the earlier single run it measured 38.36× under slightly different warm-up conditions. Both are consistent with the ~6× FlashInfer-to-us ratio, which is the most noise-resistant of the four.
