# Submission Final v2 – Comparison Report

3-way A/B/C evaluation on Modal B200 for the DSA sparse attention track
(`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`), comparing the
contest's PyTorch reference, the FlashInfer baseline, and the submitted
kernel (`submission-final-v2`).

## Metadata

| Field | Value |
|---|---|
| Track / definition | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| Submitted solution | `wombat_dsa_sparse_mla_sm100a_final` |
| Git commit | `de475b9` on `feature/cuda-graph-cache` |
| Git tag | `submission-final-v2` |
| Harness | `scripts/run_modal_compare.py` (modal B200:1, 3600s timeout) |
| Container image | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` |
| Hardware | NVIDIA B200 (sm_100a) via Modal |
| Bench config | `warmup_runs=3, iterations=100, num_trials=5` |
| FlashInfer baseline | `flashinfer_wrapper_5af199.json` (from `mlsys26-contest/solutions/baseline/dsa/…`) |
| Python reference | Track definition's built-in PyTorch reference implementation |
| Date (UTC) | 2026-04-24 |

Both the submitted kernel and the FlashInfer baseline were benched inside
the **same Modal invocation, same B200 instance, same container**, so
their latencies are directly comparable. The PyTorch reference latency is
emitted by the benchmark harness itself (column `reference_latency_ms`)
and is identical for both solutions on a given workload.

The `prev (graph-off)` column comes from
`artifacts/graph_cache/bench_graph_off_20260424T042117Z.log`, a previous
single-kernel run done *without* the in-kernel CUDA-graph cache (now the
only in-tree kernel supports the graph cache unconditionally). It's
included purely to show the marginal contribution of the graph-cache
feature that shipped in this tag.

## Headline Numbers (geomean over 23 workloads)

| Baseline | Geomean speedup of `submission-final-v2` vs baseline |
|---|---|
| PyTorch reference | **52.33×** |
| FlashInfer baseline | **9.46×** |
| Previous version (graph-cache OFF, same CUDA core) | **1.031×** |

Aggregate latency (sum across all 23 workloads, single-pass):

| Implementation | Sum latency | Min | Max |
|---|---|---|---|
| PyTorch reference | 34.108 ms | - | - |
| FlashInfer baseline | 5.980 ms | - | - |
| **Submitted (submission-final-v2)** | **0.821 ms** | 0.008 ms | 0.056 ms |

## Per-Workload Results

All 23 workloads completed with finite latency measurements for both the
submitted kernel and the FlashInfer baseline (i.e. all correctness
checks passed; the separate `run_modal.py` run at the same commit
reported `23/23 PASSED` explicitly with `abs_err ≤ 1.56e-2` and the
evaluator's default tolerances).

| Workload | PyTorch ref | FlashInfer | Prev (graph-off) | Submitted | vs ref | vs FI | vs prev |
|---|---:|---:|---:|---:|---:|---:|---:|
| `02d6ae9c` |    1.834 ms |    0.267 ms |    0.056 ms | **   0.055 ms** | 33.22x | 4.83x | 1.01x |
| `05f6de65` |    1.072 ms |    0.261 ms |    0.028 ms | **   0.026 ms** | 40.45x | 9.85x | 1.06x |
| `0a63b87b` |    1.064 ms |    0.252 ms |    0.011 ms | **   0.010 ms** | 105.38x | 24.96x | 1.09x |
| `0c23b10c` |    0.954 ms |    0.255 ms |    0.009 ms | **   0.008 ms** | 117.74x | 31.44x | 1.11x |
| `2207f0fd` |    1.712 ms |    0.262 ms |    0.056 ms | **   0.055 ms** | 31.02x | 4.75x | 1.01x |
| `232ed014` |    1.905 ms |    0.265 ms |    0.055 ms | **   0.055 ms** | 34.56x | 4.80x | 1.00x |
| `38389961` |    1.724 ms |    0.264 ms |    0.055 ms | **   0.055 ms** | 31.52x | 4.82x | 1.01x |
| `385742b2` |    1.824 ms |    0.264 ms |    0.042 ms | **   0.043 ms** | 42.52x | 6.15x | 0.98x |
| `4c46a94b` |    1.881 ms |    0.261 ms |    0.040 ms | **   0.041 ms** | 46.11x | 6.39x | 0.98x |
| `5096e459` |    1.829 ms |    0.269 ms |    0.056 ms | **   0.055 ms** | 33.13x | 4.88x | 1.01x |
| `564007ac` |    1.826 ms |    0.265 ms |    0.057 ms | **   0.056 ms** | 32.38x | 4.70x | 1.01x |
| `68d6817d` |    1.559 ms |    0.255 ms |    0.013 ms | **   0.014 ms** | 109.82x | 17.94x | 0.92x |
| `78b2e11c` |    1.834 ms |    0.265 ms |    0.056 ms | **   0.055 ms** | 33.22x | 4.79x | 1.01x |
| `7a389715` |    1.850 ms |    0.266 ms |    0.056 ms | **   0.055 ms** | 33.57x | 4.83x | 1.02x |
| `9d4a5f21` |    1.067 ms |    0.251 ms |    0.009 ms | **   0.008 ms** | 131.67x | 31.04x | 1.11x |
| `9f3f891b` |    1.072 ms |    0.259 ms |    0.024 ms | **   0.024 ms** | 44.68x | 10.78x | 1.00x |
| `ae4219a9` |    1.687 ms |    0.265 ms |    0.056 ms | **   0.055 ms** | 30.57x | 4.79x | 1.01x |
| `b7668cfd` |    1.065 ms |    0.252 ms |    0.011 ms | **   0.010 ms** | 105.44x | 24.98x | 1.09x |
| `d57eb9e1` |    1.583 ms |    0.265 ms |    0.056 ms | **   0.055 ms** | 28.68x | 4.80x | 1.01x |
| `ddfa9e34` |    1.572 ms |    0.264 ms |    0.055 ms | **   0.055 ms** | 28.53x | 4.79x | 1.00x |
| `e6b849f2` |    1.067 ms |    0.252 ms |    0.013 ms | **   0.012 ms** | 87.46x | 20.62x | 1.07x |
| `f77df5ce` |    1.066 ms |    0.252 ms |    0.009 ms | **   0.008 ms** | 131.57x | 31.06x | 1.11x |
| `fc85411e` |    1.061 ms |    0.252 ms |    0.009 ms | **   0.008 ms** | 130.99x | 31.07x | 1.11x |
| **geomean** | | | | | **52.33×** | **9.46×** | **1.031×** |

## Observations

- **vs PyTorch reference:** the submitted kernel is **52×** faster geo-mean
  across all 23 workloads. Small/sparse shapes (`0c23b10c`, `9d4a5f21`,
  `f77df5ce`, `fc85411e`) clear ~130× individually, while the big dense
  shapes (~0.055 ms band) still sit at ~30–35×.
- **vs FlashInfer baseline:** **9.5×** geo-mean. The FlashInfer
  baseline sits at a near-constant ~0.25–0.27 ms floor across all shapes
  — consistent with its paged-MLA wrapper's per-call overhead dominating
  the small/low-density shapes. Our kernel tracks workload size much
  more closely (0.008 ms on the smallest, 0.056 ms on the largest),
  which is why the per-workload ratio ranges from 4.7× (big/dense) to
  31× (small/sparse).
- **vs previous (graph-cache OFF):** **1.031×** — essentially a tie
  (within the run-to-run noise). On shapes where the dispatch overhead
  is already amortised by the CUDA-graph replay path, the in-kernel
  graph cache adds little; on the smallest shapes (`9d4a5f21`,
  `0c23b10c`, `f77df5ce`, `fc85411e`) it still contributes ~9–12%. One
  workload (`68d6817d`) shows 0.91× — within the ±10–15% per-shape
  jitter we've consistently seen across Modal runs in this codebase.
- **Measurement caveat:** this Modal container did *not* have
  `cupti-python` installed (see the `UserWarning` in the raw log); the
  harness fell back to CUDA-event timing. Absolute latencies for our
  kernel at the ~8–10 µs end may be slightly noisier than they would be
  on the official evaluator (`EVALUATION.md` mandates cupti-python).
  The run-to-run jitter observed across the four `run_modal.py` runs
  earlier in the session (geo-mean ranging 87× → 125× vs reference for
  the same commit) is consistent with this.

## Raw Modal Log

Full captured log from the comparison run is reproduced verbatim below.
It includes Modal app setup, the two per-solution raw-number sections,
the comparison table, and the harness's geo-mean summary.

Log path: `artifacts/compare/compare_submission_final_v2_20260424T085244Z.log`

```

✓ Initialized. View run at 
https://modal.com/apps/ykirpichev/main/ap-X41bhGyZVeAxlVxGsp9cIV


✓ Created objects.
├── 🔨 Created mount 
│   /home/ykirpichev/sources/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
│   /scripts/run_modal_compare.py
└── 🔨 Created function run_benchmark.
Packing our solution from source...
Solution packed: 
/home/ykirpichev/sources/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/sol
ution.json
  Name: wombat_dsa_sparse_mla_sm100a_final
  Definition: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
  Author: Wombat
  Language: python
Loading FlashInfer baseline from 
/home/ykirpichev/sources/mlsys26-contest/solutions/baseline/dsa/dsa_sparse_atten
tion_h16_ckv512_kpe64_topk2048_ps64/flashinfer_wrapper_5af199.json...

Launching combined benchmark on Modal B200 (our solution + FlashInfer baseline, 
same reference, same B200)...


==========
== CUDA ==
==========

CUDA Version 13.2.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/opt/conda/envs/py312/lib/python3.12/site-packages/flashinfer_bench/bench/timing.py:74: UserWarning: CUPTI is not installed. Try 'pip install -U cupti-python'. Falling back to CUDA events for benchmarking.
  times = bench_gpu_time_with_cupti(

=== Raw per-workload numbers (wombat_dsa_sparse_mla_sm100a_final) ===
  02d6ae9c  lat=0.0552 ms  ref=1.8339 ms  speedup=33.24x
  05f6de65  lat=0.0265 ms  ref=1.0718 ms  speedup=40.43x
  0a63b87b  lat=0.0101 ms  ref=1.0643 ms  speedup=105.09x
  0c23b10c  lat=0.0081 ms  ref=0.9537 ms  speedup=118.27x
  2207f0fd  lat=0.0552 ms  ref=1.7122 ms  speedup=31.04x
  232ed014  lat=0.0551 ms  ref=1.9045 ms  speedup=34.55x
  38389961  lat=0.0547 ms  ref=1.7241 ms  speedup=31.54x
  385742b2  lat=0.0429 ms  ref=1.8243 ms  speedup=42.53x
  4c46a94b  lat=0.0408 ms  ref=1.8814 ms  speedup=46.11x
  5096e459  lat=0.0552 ms  ref=1.8289 ms  speedup=33.15x
  564007ac  lat=0.0564 ms  ref=1.8261 ms  speedup=32.40x
  68d6817d  lat=0.0142 ms  ref=1.5594 ms  speedup=109.75x
  78b2e11c  lat=0.0552 ms  ref=1.8340 ms  speedup=33.24x
  7a389715  lat=0.0551 ms  ref=1.8499 ms  speedup=33.54x
  9d4a5f21  lat=0.0081 ms  ref=1.0665 ms  speedup=131.99x
  9f3f891b  lat=0.0240 ms  ref=1.0722 ms  speedup=44.59x
  ae4219a9  lat=0.0552 ms  ref=1.6872 ms  speedup=30.59x
  b7668cfd  lat=0.0101 ms  ref=1.0649 ms  speedup=105.08x
  d57eb9e1  lat=0.0552 ms  ref=1.5831 ms  speedup=28.69x
  ddfa9e34  lat=0.0551 ms  ref=1.5720 ms  speedup=28.52x
  e6b849f2  lat=0.0122 ms  ref=1.0670 ms  speedup=87.79x
  f77df5ce  lat=0.0081 ms  ref=1.0657 ms  speedup=132.15x
  fc85411e  lat=0.0081 ms  ref=1.0610 ms  speedup=131.57x

=== Raw per-workload numbers (flashinfer_wrapper_5af199) ===
  02d6ae9c  lat=0.2666 ms  ref=1.8339 ms  speedup=6.88x
  05f6de65  lat=0.2611 ms  ref=1.0718 ms  speedup=4.10x
  0a63b87b  lat=0.2521 ms  ref=1.0643 ms  speedup=4.22x
  0c23b10c  lat=0.2547 ms  ref=0.9537 ms  speedup=3.74x
  2207f0fd  lat=0.2620 ms  ref=1.7122 ms  speedup=6.54x
  232ed014  lat=0.2647 ms  ref=1.9045 ms  speedup=7.20x
  38389961  lat=0.2639 ms  ref=1.7241 ms  speedup=6.53x
  385742b2  lat=0.2640 ms  ref=1.8243 ms  speedup=6.91x
  4c46a94b  lat=0.2608 ms  ref=1.8814 ms  speedup=7.22x
  5096e459  lat=0.2691 ms  ref=1.8289 ms  speedup=6.80x
  564007ac  lat=0.2649 ms  ref=1.8261 ms  speedup=6.89x
  68d6817d  lat=0.2548 ms  ref=1.5594 ms  speedup=6.12x
  78b2e11c  lat=0.2646 ms  ref=1.8340 ms  speedup=6.93x
  7a389715  lat=0.2659 ms  ref=1.8499 ms  speedup=6.96x
  9d4a5f21  lat=0.2514 ms  ref=1.0665 ms  speedup=4.24x
  9f3f891b  lat=0.2588 ms  ref=1.0722 ms  speedup=4.14x
  ae4219a9  lat=0.2645 ms  ref=1.6872 ms  speedup=6.38x
  b7668cfd  lat=0.2523 ms  ref=1.0649 ms  speedup=4.22x
  d57eb9e1  lat=0.2649 ms  ref=1.5831 ms  speedup=5.98x
  ddfa9e34  lat=0.2639 ms  ref=1.5720 ms  speedup=5.96x
  e6b849f2  lat=0.2516 ms  ref=1.0670 ms  speedup=4.24x
  f77df5ce  lat=0.2516 ms  ref=1.0657 ms  speedup=4.23x
  fc85411e  lat=0.2517 ms  ref=1.0610 ms  speedup=4.22x

  workload    python ref    flashinfer    prev (off)    current    vs FI  vs 
prev
--------------------------------------------------------------------------------
-
  02d6ae9c      1.834 ms      0.267 ms      0.056 ms   0.055 ms    4.83x    
1.02x
  05f6de65      1.072 ms      0.261 ms      0.028 ms   0.027 ms    9.85x    
1.06x
  0a63b87b      1.064 ms      0.252 ms      0.011 ms   0.010 ms   24.90x    
1.09x
  0c23b10c      0.954 ms      0.255 ms      0.009 ms   0.008 ms   31.58x    
1.12x
  2207f0fd      1.712 ms      0.262 ms      0.056 ms   0.055 ms    4.75x    
1.02x
  232ed014      1.904 ms      0.265 ms      0.055 ms   0.055 ms    4.80x    
1.00x
  38389961      1.724 ms      0.264 ms      0.055 ms   0.055 ms    4.83x    
1.01x
  385742b2      1.824 ms      0.264 ms      0.042 ms   0.043 ms    6.15x    
0.98x
  4c46a94b      1.881 ms      0.261 ms      0.040 ms   0.041 ms    6.39x    
0.98x
  5096e459      1.829 ms      0.269 ms      0.056 ms   0.055 ms    4.88x    
1.02x
  564007ac      1.826 ms      0.265 ms      0.057 ms   0.056 ms    4.70x    
1.01x
  68d6817d      1.559 ms      0.255 ms      0.013 ms   0.014 ms   17.93x    
0.91x
  78b2e11c      1.834 ms      0.265 ms      0.056 ms   0.055 ms    4.80x    
1.02x
  7a389715      1.850 ms      0.266 ms      0.056 ms   0.055 ms    4.82x    
1.02x
  9d4a5f21      1.066 ms      0.251 ms      0.009 ms   0.008 ms   31.11x    
1.11x
  9f3f891b      1.072 ms      0.259 ms      0.024 ms   0.024 ms   10.76x    
1.00x
  ae4219a9      1.687 ms      0.264 ms      0.056 ms   0.055 ms    4.79x    
1.02x
  b7668cfd      1.065 ms      0.252 ms      0.011 ms   0.010 ms   24.89x    
1.09x
  d57eb9e1      1.583 ms      0.265 ms      0.056 ms   0.055 ms    4.80x    
1.01x
  ddfa9e34      1.572 ms      0.264 ms      0.055 ms   0.055 ms    4.79x    
1.00x
  e6b849f2      1.067 ms      0.252 ms      0.013 ms   0.012 ms   20.70x    
1.07x
  f77df5ce      1.066 ms      0.252 ms      0.009 ms   0.008 ms   31.21x    
1.12x
  fc85411e      1.061 ms      0.252 ms      0.009 ms   0.008 ms   31.21x    
1.12x
--------------------------------------------------------------------------------
-
   geomean                                                         9.47x    
1.03x

  current vs python reference (geomean speedup): 52.4x
  current vs flashinfer baseline (geomean speedup): 9.5x
  current vs previous version   (geomean speedup): 1.03x

Stopping app - local entrypoint completed.
[modal-client] 2026-04-24T01:54:46-0700 Timed out waiting for final app logs.
✓ App completed. View run at 
https://modal.com/apps/ykirpichev/main/ap-X41bhGyZVeAxlVxGsp9cIV
```
