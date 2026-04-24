# Paired comparison: now with Python + explicit sm_100a

Modal B200, single session, BenchmarkConfig(warmup=3, iters=100, trials=5),
23 workloads × **5 independent repeats** inside the same session.

Artifacts:
- `artifacts/compare_fi_v4_v5cuda_v5py_5repeats.json` — raw per-workload results
- `artifacts/compare_fi_v4_v5cuda_v5py_5repeats.log` — full modal log

## Per-workload latency (µs, mean over 5 repeats)

| workload | PyTorch ref | FlashInfer | v4 Stage 3c | v5 (CUDA lang) | v5 (python+sm_100a) |
|---|---:|---:|---:|---:|---:|
| `02d6ae9c64` | 1826.67 | 263.89 | 159.42 | 133.92 | 126.95 |
| `05f6de657d` | 1066.23 | 258.15 | 36.83 | 30.79 | 30.62 |
| `0a63b87bb2` | 1064.03 | 251.60 | 16.33 | 14.32 | 14.18 |
| `0c23b10c7b` | 942.29 | 252.52 | 12.30 | 10.78 | 10.33 |
| `2207f0fdc9` | 1717.47 | 263.06 | 157.63 | 131.15 | 124.94 |
| `232ed014ba` | 1835.65 | 263.21 | 90.21 | 75.89 | 73.33 |
| `3838996164` | 1700.94 | 263.65 | 90.20 | 75.93 | 73.64 |
| `385742b271` | 1829.33 | 263.03 | 87.95 | 73.76 | 69.69 |
| `4c46a94ba2` | 1822.13 | 261.93 | 83.98 | 71.61 | 67.59 |
| `5096e459ce` | 1840.11 | 264.76 | 155.66 | 131.06 | 124.68 |
| `564007ac35` | 1845.78 | 264.25 | 159.69 | 133.13 | 126.90 |
| `68d6817dcf` | 1569.02 | 252.29 | 14.21 | 12.29 | 12.27 |
| `78b2e11c30` | 1825.99 | 264.26 | 159.68 | 134.35 | 126.97 |
| `7a389715dc` | 1830.25 | 264.50 | 130.44 | 108.67 | 104.40 |
| `9d4a5f2126` | 1063.55 | 251.50 | 12.42 | 12.28 | 12.25 |
| `9f3f891bfb` | 1068.95 | 257.42 | 32.56 | 26.72 | 26.61 |
| `ae4219a95f` | 1697.46 | 262.88 | 159.67 | 133.17 | 126.95 |
| `b7668cfd19` | 1062.98 | 251.19 | 15.98 | 14.32 | 14.22 |
| `d57eb9e19f` | 1602.72 | 262.98 | 157.37 | 131.14 | 124.94 |
| `ddfa9e340b` | 1574.28 | 262.95 | 129.86 | 108.75 | 104.42 |
| `e6b849f290` | 1062.30 | 251.01 | 18.43 | 16.36 | 14.76 |
| `f77df5ce21` | 1063.40 | 251.12 | 12.64 | 12.29 | 12.26 |
| `fc85411e25` | 1062.83 | 251.61 | 12.38 | 12.28 | 12.24 |

## Pairwise speedups (per-workload mean over 5 repeats; aggregated over 23 workloads)

| comparison | geomean | amean | min | max |
|---|---:|---:|---:|---:|
| **v5 (python+sm_100a) vs v5 (CUDA lang)** | **1.035×** | 1.035× | 1.00× | 1.11× |
| **v5 (python+sm_100a) vs v4 Stage 3c** | **1.20×** | 1.20× | 1.01× | 1.26× |
| **v5 (python+sm_100a) vs FlashInfer** | **5.82×** | 9.20× | 2.07× | 24.46× |
| **v5 (python+sm_100a) vs PyTorch reference** | **32.22×** | 43.83× | 12.83× | 127.83× |
| FlashInfer vs PyTorch reference | 5.53× | 5.68× | 3.73× | 6.98× |

## Mean latency across 23 workloads

| solution | mean (µs) | min | max | all passed |
|---|---:|---:|---:|---:|
| v5 (python+sm_100a) | 66.75 | 10.33 | 126.97 | 115/115 |
| v5 (CUDA lang) | 69.78 | 10.78 | 134.35 | 115/115 |
| v4 Stage 3c | 82.86 | 12.30 | 159.69 | 115/115 |
| FlashInfer | 258.86 | 251.01 | 264.76 | 115/115 |
| PyTorch reference | 1477.15 | 942.29 | 1845.78 | 115/115 |

## Cross-repeat noise (coefficient of variation, mean/median over 23 workloads)

| solution | mean CV% | median CV% | max CV% |
|---|---:|---:|---:|
| FlashInfer | 0.47% | 0.42% | 0.91% |
| v4 Stage 3c | 0.42% | 0.15% | 2.07% |
| v5 (CUDA lang) | 0.26% | 0.12% | 1.85% |
| v5 (python+sm_100a) | 0.31% | 0.07% | 1.86% |
| PyTorch reference | 0.49% | 0.34% | 1.48% |

## Interpretation

- **The Python+sm_100a submission compiles and runs correctly** on Modal B200 (5/5 repeats × 23 workloads = 115/115 passed) and is the fastest of the four.
- vs the previous CUDA-language submission it measures **1.035× geomean** faster (range 1.00× – 1.11×). This ~3.5% win comes from the explicit `-gencode arch=compute_100a,code=sm_100a` + `-O3 --use_fast_math` flags that the stock CUDA builder does NOT pass automatically.
- On the Modal dev image we already forced sm_100a via `TVM_FFI_CUDA_ARCH_LIST=10.0a`, so the CUDA submission was already compiling for sm_100a here. **On the bare-metal evaluator, which has no such env override, the python path is the only one that guarantees sm_100a SASS** — the gap between CUDA and Python submissions is expected to be larger there than the 3.5% we see on Modal.
- vs FlashInfer wrapper: **5.82× geomean**. vs PyTorch reference: **32.22× geomean**. vs v4 Stage 3c: **1.20× geomean**.
