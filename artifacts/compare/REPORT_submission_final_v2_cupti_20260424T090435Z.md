# Submission Final v2 – Comparison Report (cupti-timed)

3-way A/B/C evaluation on Modal B200 for the DSA sparse attention track
(`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`), comparing the
contest's PyTorch reference, the FlashInfer baseline, and the submitted
kernel (`submission-final-v2`).

This is a re-run of the earlier comparison with `cupti-python` installed
in the Modal image, matching the pinned evaluator environment described
in `EVALUATION.md`. CUDA-event fallback timing has a fixed per-event
overhead that materially inflates reported latency for sub-10 µs
kernels; cupti measures actual GPU execution time, so the absolute
latencies (and therefore the speedup ratios) below are what the
official evaluator would see.

## Metadata

| Field | Value |
|---|---|
| Track / definition | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| Submitted solution | `wombat_dsa_sparse_mla_sm100a_final` |
| Git commit | `de475b9` on `feature/cuda-graph-cache` |
| Git tag | `submission-final-v2` |
| Harness | `scripts/run_modal_compare.py` (Modal B200:1, 3600s timeout) |
| Container image | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` |
| Hardware | NVIDIA B200 (sm_100a) via Modal |
| Bench config | `warmup_runs=3, iterations=100, num_trials=5` |
| Timing backend | **cupti-python** (matches evaluator; no CUDA-event fallback) |
| FlashInfer baseline | `flashinfer_wrapper_5af199.json` (`mlsys26-contest/solutions/baseline/dsa/…`) |
| Python reference | Track definition's built-in PyTorch reference implementation |
| Date (UTC) | 2026-04-24 |

Both the submitted kernel and the FlashInfer baseline were benched
inside the **same Modal invocation, same B200 instance, same
container**, so their latencies are directly comparable. The PyTorch
reference latency is emitted by the benchmark harness itself per
workload (column `reference_latency_ms`).

The `prev (graph-off, events)` column comes from
`artifacts/graph_cache/bench_graph_off_20260424T042117Z.log`, a previous
single-kernel run taken *before* cupti was added to the compare image
and with the in-kernel CUDA-graph cache OFF. Because that earlier log
was CUDA-event-timed while the current run is cupti-timed, the
`vs prev` column conflates the real graph-cache benefit with the
event-vs-cupti timing delta. **It is shown only as a rough qualitative
indicator and should not be read as a clean A/B for the graph cache
alone.** The clean apples-to-apples comparisons are `vs ref` and
`vs FI`, both measured in the same cupti-timed invocation.

## Headline Numbers (geomean over 23 workloads)

| Baseline | Geomean speedup of `submission-final-v2` |
|---|---:|
| PyTorch reference | **126.03×** |
| FlashInfer baseline | **14.18×** |
| Previous version (graph-cache OFF, events-timed; mixed-timing, see note) | 1.36× ⚠ |

Aggregate latency (sum across all 23 workloads, single-pass):

| Implementation | Sum latency | Min | Max |
|---|---:|---:|---:|
| PyTorch reference | 66.079 ms | - | - |
| FlashInfer baseline | 6.793 ms | - | - |
| **Submitted (submission-final-v2)** | **0.719 ms** | 0.003 ms | 0.052 ms |

## Per-Workload Results

All 23 workloads completed with finite cupti latency measurements for
both the submitted kernel and the FlashInfer baseline (i.e. all
correctness checks passed; the separate `run_modal.py` run at the same
commit reported `23/23 PASSED` explicitly with `abs_err ≤ 1.56e-2`).

| Workload | PyTorch ref | FlashInfer | Prev (graph-off, events) | Submitted | vs ref | vs FI | vs prev ⚠ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `02d6ae9c` |    3.893 ms |    0.302 ms |    0.056 ms | **   0.051 ms** | 75.73x | 5.88x | 1.09x |
| `05f6de65` |    1.536 ms |    0.290 ms |    0.028 ms | **   0.023 ms** | 67.98x | 12.84x | 1.24x |
| `0a63b87b` |    1.526 ms |    0.285 ms |    0.011 ms | **   0.006 ms** | 254.33x | 47.50x | 1.83x |
| `0c23b10c` |    1.226 ms |    0.286 ms |    0.009 ms | **   0.003 ms** | 360.56x | 84.24x | 2.65x |
| `2207f0fd` |    3.566 ms |    0.299 ms |    0.056 ms | **   0.051 ms** | 70.19x | 5.88x | 1.10x |
| `232ed014` |    4.055 ms |    0.302 ms |    0.055 ms | **   0.051 ms** | 80.30x | 5.99x | 1.09x |
| `38389961` |    3.511 ms |    0.298 ms |    0.055 ms | **   0.050 ms** | 70.22x | 5.97x | 1.10x |
| `385742b2` |    3.984 ms |    0.303 ms |    0.042 ms | **   0.038 ms** | 105.12x | 7.99x | 1.11x |
| `4c46a94b` |    4.161 ms |    0.301 ms |    0.040 ms | **   0.036 ms** | 116.89x | 8.45x | 1.12x |
| `5096e459` |    3.782 ms |    0.297 ms |    0.056 ms | **   0.051 ms** | 74.60x | 5.85x | 1.10x |
| `564007ac` |    4.285 ms |    0.301 ms |    0.057 ms | **   0.052 ms** | 82.88x | 5.82x | 1.10x |
| `68d6817d` |    3.157 ms |    0.286 ms |    0.013 ms | **   0.009 ms** | 335.83x | 30.47x | 1.38x |
| `78b2e11c` |    3.995 ms |    0.303 ms |    0.056 ms | **   0.051 ms** | 78.33x | 5.94x | 1.10x |
| `7a389715` |    3.856 ms |    0.300 ms |    0.056 ms | **   0.051 ms** | 76.21x | 5.92x | 1.11x |
| `9d4a5f21` |    1.576 ms |    0.286 ms |    0.009 ms | **   0.004 ms** | 384.44x | 69.68x | 2.20x |
| `9f3f891b` |    1.625 ms |    0.297 ms |    0.024 ms | **   0.019 ms** | 86.01x | 15.70x | 1.27x |
| `ae4219a9` |    3.711 ms |    0.307 ms |    0.056 ms | **   0.051 ms** | 73.20x | 6.06x | 1.10x |
| `b7668cfd` |    1.605 ms |    0.285 ms |    0.011 ms | **   0.006 ms** | 267.58x | 47.50x | 1.83x |
| `d57eb9e1` |    3.180 ms |    0.297 ms |    0.056 ms | **   0.051 ms** | 62.84x | 5.87x | 1.11x |
| `ddfa9e34` |    3.141 ms |    0.301 ms |    0.055 ms | **   0.050 ms** | 62.33x | 5.97x | 1.09x |
| `e6b849f2` |    1.520 ms |    0.289 ms |    0.013 ms | **   0.008 ms** | 192.41x | 36.56x | 1.65x |
| `f77df5ce` |    1.636 ms |    0.289 ms |    0.009 ms | **   0.004 ms** | 389.48x | 68.81x | 2.14x |
| `fc85411e` |    1.552 ms |    0.289 ms |    0.009 ms | **   0.004 ms** | 378.41x | 70.54x | 2.20x |
| **geomean** | | | | | **126.03×** | **14.18×** | 1.36× ⚠ |

## Observations

- **vs PyTorch reference:** **126×** geo-mean speedup across 23
  workloads. Per-workload ratios span 62× on the longest dense shapes
  (`d57eb9e1` / `ddfa9e34`; submitted ~0.050–0.051 ms vs PyTorch
  ~3.1–3.2 ms) up to **389×** on the smallest sparse shapes
  (`f77df5ce`: 0.004 ms vs 1.64 ms ref; `9d4a5f21`: 384×;
  `fc85411e`: 378×; `0c23b10c`: 361×). The largest absolute speedup
  multipliers cluster at shapes where PyTorch's per-element dispatch
  overhead dominates.

- **vs FlashInfer baseline:** **14.2×** geo-mean. The baseline sits at a
  near-constant ~0.28–0.31 ms floor across all shapes — its paged-MLA
  wrapper's per-call overhead dominates for the small/sparse shapes.
  Our kernel tracks workload size much more closely (0.003 ms on the
  smallest, 0.052 ms on the largest), giving per-workload ratios from
  5.8× (big/dense, e.g. `564007ac`) to **84×** (small/sparse,
  `0c23b10c`).

- **Aggregate wall time:** if you summed all 23 single-pass latencies,
  the submitted kernel finishes in **0.719 ms** vs
  6.793 ms for FlashInfer and 66.08 ms
  for the PyTorch reference — roughly **92×**
  and **9.5×** end-to-end respectively.

- **Delta from the earlier (non-cupti) compare run:**
    - vs PyTorch: 52× → **126×**
    - vs FlashInfer: 9.5× → **14.2×**
    - vs prev: 1.03× → 1.36× (mixed-timing; see caveat below)

  The jump is almost entirely a timing-methodology effect, not a code
  change: the earlier run fell back to CUDA events, whose ~few-µs
  per-event overhead dominated when our kernel was measured at ~8 µs.
  cupti reports the actual GPU runtime, which is much closer to the
  true hardware-level performance the evaluator will see.

- **Caveat on the `vs prev` column (⚠):** the previous-version latencies
  in that column are CUDA-event-timed (0.009–0.056 ms). Any ratio
  against our current (cupti-timed) latencies conflates the real
  contribution of the in-kernel CUDA-graph cache with the
  event-vs-cupti timing delta. The prev column is kept only to show
  that neither correctness nor the general shape of the speedup curve
  has regressed relative to the prior submission; it is **not** a clean
  A/B for the graph-cache feature itself. A cupti-timed
  `WOMBAT_CUDA_GRAPH=0` would be needed for that, but the in-tree
  kernel no longer has a disable knob — the graph cache is
  unconditional in this submission.

- **Per-shape jitter:** the earlier session showed ±10–15% per-shape
  run-to-run jitter across four sequential Modal runs (geo-mean vs ref
  spanning 87× → 125× on events timing). Expect a similar ±10% band in
  the cupti numbers above if this is re-run.

## Raw Modal Log

Full captured log from the cupti-timed comparison run is reproduced
verbatim below. It includes Modal app setup, the two per-solution
raw-number sections, the in-harness comparison table, and the
geometric-mean summary lines.

Log path: `artifacts/compare/compare_submission_final_v2_cupti_20260424T085836Z.log`

```

✓ Initialized. View run at 
https://modal.com/apps/ykirpichev/main/ap-4jnNbC8BNLWRAB1MNJ01IR


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


=== Raw per-workload numbers (wombat_dsa_sparse_mla_sm100a_final) ===
  02d6ae9c  lat=0.0514 ms  ref=3.8925 ms  speedup=75.79x
  05f6de65  lat=0.0226 ms  ref=1.5363 ms  speedup=68.01x
  0a63b87b  lat=0.0060 ms  ref=1.5260 ms  speedup=253.12x
  0c23b10c  lat=0.0034 ms  ref=1.2259 ms  speedup=363.82x
  2207f0fd  lat=0.0508 ms  ref=3.5655 ms  speedup=70.20x
  232ed014  lat=0.0505 ms  ref=4.0551 ms  speedup=80.25x
  38389961  lat=0.0500 ms  ref=3.5108 ms  speedup=70.21x
  385742b2  lat=0.0379 ms  ref=3.9840 ms  speedup=105.23x
  4c46a94b  lat=0.0356 ms  ref=4.1614 ms  speedup=116.96x
  5096e459  lat=0.0507 ms  ref=3.7821 ms  speedup=74.59x
  564007ac  lat=0.0517 ms  ref=4.2849 ms  speedup=82.87x
  68d6817d  lat=0.0094 ms  ref=3.1568 ms  speedup=335.01x
  78b2e11c  lat=0.0510 ms  ref=3.9948 ms  speedup=78.29x
  7a389715  lat=0.0506 ms  ref=3.8560 ms  speedup=76.27x
  9d4a5f21  lat=0.0041 ms  ref=1.5762 ms  speedup=382.94x
  9f3f891b  lat=0.0189 ms  ref=1.6255 ms  speedup=86.05x
  ae4219a9  lat=0.0507 ms  ref=3.7114 ms  speedup=73.24x
  b7668cfd  lat=0.0060 ms  ref=1.6055 ms  speedup=265.60x
  d57eb9e1  lat=0.0506 ms  ref=3.1797 ms  speedup=62.82x
  ddfa9e34  lat=0.0504 ms  ref=3.1414 ms  speedup=62.30x
  e6b849f2  lat=0.0079 ms  ref=1.5200 ms  speedup=193.00x
  f77df5ce  lat=0.0042 ms  ref=1.6358 ms  speedup=393.76x
  fc85411e  lat=0.0041 ms  ref=1.5515 ms  speedup=377.89x

=== Raw per-workload numbers (flashinfer_wrapper_5af199) ===
  02d6ae9c  lat=0.3020 ms  ref=3.8925 ms  speedup=12.89x
  05f6de65  lat=0.2901 ms  ref=1.5363 ms  speedup=5.30x
  0a63b87b  lat=0.2850 ms  ref=1.5260 ms  speedup=5.35x
  0c23b10c  lat=0.2864 ms  ref=1.2259 ms  speedup=4.28x
  2207f0fd  lat=0.2988 ms  ref=3.5655 ms  speedup=11.93x
  232ed014  lat=0.3024 ms  ref=4.0551 ms  speedup=13.41x
  38389961  lat=0.2984 ms  ref=3.5108 ms  speedup=11.77x
  385742b2  lat=0.3029 ms  ref=3.9840 ms  speedup=13.15x
  4c46a94b  lat=0.3007 ms  ref=4.1614 ms  speedup=13.84x
  5096e459  lat=0.2966 ms  ref=3.7821 ms  speedup=12.75x
  564007ac  lat=0.3010 ms  ref=4.2849 ms  speedup=14.24x
  68d6817d  lat=0.2864 ms  ref=3.1568 ms  speedup=11.02x
  78b2e11c  lat=0.3029 ms  ref=3.9948 ms  speedup=13.19x
  7a389715  lat=0.2998 ms  ref=3.8560 ms  speedup=12.86x
  9d4a5f21  lat=0.2857 ms  ref=1.5762 ms  speedup=5.52x
  9f3f891b  lat=0.2967 ms  ref=1.6255 ms  speedup=5.48x
  ae4219a9  lat=0.3071 ms  ref=3.7114 ms  speedup=12.08x
  b7668cfd  lat=0.2850 ms  ref=1.6055 ms  speedup=5.63x
  d57eb9e1  lat=0.2970 ms  ref=3.1797 ms  speedup=10.71x
  ddfa9e34  lat=0.3011 ms  ref=3.1414 ms  speedup=10.43x
  e6b849f2  lat=0.2888 ms  ref=1.5200 ms  speedup=5.26x
  f77df5ce  lat=0.2890 ms  ref=1.6358 ms  speedup=5.66x
  fc85411e  lat=0.2892 ms  ref=1.5515 ms  speedup=5.36x

  workload    python ref    flashinfer    prev (off)    current    vs FI  vs 
prev
--------------------------------------------------------------------------------
-
  02d6ae9c      3.893 ms      0.302 ms      0.056 ms   0.051 ms    5.88x    
1.09x
  05f6de65      1.536 ms      0.290 ms      0.028 ms   0.023 ms   12.84x    
1.24x
  0a63b87b      1.526 ms      0.285 ms      0.011 ms   0.006 ms   47.27x    
1.82x
  0c23b10c      1.226 ms      0.286 ms      0.009 ms   0.003 ms   85.00x    
2.67x
  2207f0fd      3.565 ms      0.299 ms      0.056 ms   0.051 ms    5.88x    
1.10x
  232ed014      4.055 ms      0.302 ms      0.055 ms   0.051 ms    5.98x    
1.09x
  38389961      3.511 ms      0.298 ms      0.055 ms   0.050 ms    5.97x    
1.10x
  385742b2      3.984 ms      0.303 ms      0.042 ms   0.038 ms    8.00x    
1.11x
  4c46a94b      4.161 ms      0.301 ms      0.040 ms   0.036 ms    8.45x    
1.12x
  5096e459      3.782 ms      0.297 ms      0.056 ms   0.051 ms    5.85x    
1.10x
  564007ac      4.285 ms      0.301 ms      0.057 ms   0.052 ms    5.82x    
1.10x
  68d6817d      3.157 ms      0.286 ms      0.013 ms   0.009 ms   30.40x    
1.38x
  78b2e11c      3.995 ms      0.303 ms      0.056 ms   0.051 ms    5.94x    
1.10x
  7a389715      3.856 ms      0.300 ms      0.056 ms   0.051 ms    5.93x    
1.11x
  9d4a5f21      1.576 ms      0.286 ms      0.009 ms   0.004 ms   69.41x    
2.19x
  9f3f891b      1.626 ms      0.297 ms      0.024 ms   0.019 ms   15.70x    
1.27x
  ae4219a9      3.711 ms      0.307 ms      0.056 ms   0.051 ms    6.06x    
1.11x
  b7668cfd      1.606 ms      0.285 ms      0.011 ms   0.006 ms   47.14x    
1.82x
  d57eb9e1      3.180 ms      0.297 ms      0.056 ms   0.051 ms    5.87x    
1.11x
  ddfa9e34      3.141 ms      0.301 ms      0.055 ms   0.050 ms    5.97x    
1.09x
  e6b849f2      1.520 ms      0.289 ms      0.013 ms   0.008 ms   36.67x    
1.65x
  f77df5ce      1.636 ms      0.289 ms      0.009 ms   0.004 ms   69.55x    
2.17x
  fc85411e      1.551 ms      0.289 ms      0.009 ms   0.004 ms   70.44x    
2.19x
--------------------------------------------------------------------------------
-
   geomean                                                        14.18x    
1.36x

  current vs python reference (geomean speedup): 126.1x
  current vs flashinfer baseline (geomean speedup): 14.2x
  current vs previous version   (geomean speedup): 1.36x

Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/ykirpichev/main/ap-4jnNbC8BNLWRAB1MNJ01IR
```
