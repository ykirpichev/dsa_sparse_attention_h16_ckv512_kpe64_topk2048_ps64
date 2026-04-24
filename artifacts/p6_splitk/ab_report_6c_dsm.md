# Phase 6c — Cluster-shared (DSM) sparse_idx broadcast — **SKIPPED**

**Date:** 2026-04-23
**Verdict:** Expected win < noise floor. Skipped.

## Hypothesis

On sm_100a, CTAs within a cluster can share SMEM via distributed shared memory (DSM). The hypothesis was that a cluster of 4 CTAs could read `sparse_idx[t, *]` **once** (into rank-0's SMEM) and let the other 3 CTAs access it via DSM, saving HBM bandwidth on the sparse-index prescan.

## Pre-implementation estimate

Per-CTA sparse_idx HBM read: `TOPK × 4 B = 2048 × 4 = 8 KiB`.

With a cluster of 4 and one designated reader, the other 3 CTAs skip their HBM read:
- Save per cluster: `3 × 8 KiB = 24 KiB`.
- Per-workload HBM baseline is dominated by K/V loads: `TOPK × (D_CK + D_KP) × H × T × 2 B` (e.g. ≈ 289 MiB for `T=8 H=16`).

| T | Total HBM (MiB) | sparse_idx (KiB) | DSM save / cluster=4 (KiB) | % of HBM saved |
|---|---|---|---|---|
| 1 | 36.1 | 128.0 | 96.0 | **0.26 %** |
| 2 | 72.2 | 256.0 | 192.0 | 0.26 % |
| 6 | 216.8 | 768.0 | 576.0 | 0.26 % |
| 8 | 289.0 | 1024.0 | 768.0 | 0.26 % |

The max theoretical bandwidth savings is **0.26 % of total HBM traffic**, constant across all `T` because the K/V read scales the same way.

Adding the prescan's per-thread scalar compute (bounds checks, `-1` sentinel replace, `local_max_valid_k_p1` update) — roughly 100–200 cycles per CTA — at best gives another ~1 % savings on that phase. Amortized across the full kernel (~35 000 cycles per tile × 8 tiles = 280 000 cycles), the wall-clock impact is **< 1.5 %** per workload.

## Why we skipped

The measured noise floor on this bench (per `artifacts/p6_splitk/ab_report.md` and `ab_report_6b_fused.md`) is:
- ±5 % on G=1 workloads (sub-15 µs, timer quantization dominates).
- ±2 % on long split-K workloads.

A theoretical <1.5 % win is **strictly below** our ability to measure it reliably on Modal B200 with the current CUDA-events benchmark methodology.

Implementation cost is non-trivial:
- Switch the kernel launch to `cudaLaunchKernelEx` with a `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` attribute.
- Non-portable cluster size opt-in for > 8 CTAs/cluster (on sm_100a).
- Rework prescan: rank-0 does the scan, others issue `cluster.sync` and `dsm_ptr_translate` to read remote SMEM. Adds ~50 lines and a new sync primitive.
- New env gate + separate kernel variant for A/B.

**Cost/benefit verdict: not worth it at this workload shape.**

## Conditions under which this would become worthwhile

Revisit this optimization if any of the following change:
- `TOPK` grows substantially (e.g., to 8192+): sparse_idx HBM share grows.
- K/V tiles are cached or compressed (reducing the denominator).
- Per-kernel wall-clock measurement resolution improves below 0.5 µs (CUPTI or direct cycle counters).
- A cluster-launch cleanup becomes available elsewhere in the kernel (e.g., TMA cluster-based tiles).

## Artifacts

No code changes. Analysis only.
