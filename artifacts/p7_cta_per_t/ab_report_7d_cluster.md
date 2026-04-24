# Phase 7d — Cluster-shared K/V (SKIPPED this session)

## TL;DR

**Not implemented this session.** Phase 7c-next's V-accum WMMA experiment
conclusively showed the 46 µs CPT-per-t floor is **issue-slot
starvation** under 1-CTA/SM occupancy, **not HBM bandwidth**. 7d's core
mechanism (halving K/V HBM traffic via DSM-shared tiles within a 2-CTA
cluster) therefore attacks the wrong bottleneck.

The meaningful rearchitecture — an **H-split 2-CTA cluster** that both
shares K/V *and* halves per-CTA SMEM → 2 CTAs/SM — is a multi-session
project that would need to rewrite `configure_cpt_mma*_kernel_attrs_once`,
the Q and V SMEM layout, `c_frag` channel ownership, and the output
write path. Left for a future phase.

## Rationale

### Observed
- `mma_v` vs `mma`: geomean identical (both 1.36–1.38× slower than
  legacy), same 46 µs floor per CTA.
- HBM budget per CTA: 144 KB / 8 tiles. At B200 HBM 8 TB/s (54 GB/s
  per-SM-share), total HBM time **≈ 2.7 µs** — not a 46 µs cost.
- Active compute (incl. MMA + reduce + softmax + rescale + V-MMA)
  accounts for **≈ 4 µs** per CTA.
- **The ~42 µs unaccounted gap is consistent with instruction-issue
  starvation at 16 warps/SM** (B200 has 4 issue slots/cycle × 4
  partitions = 16 issues/cycle, but only 16 warps/SM compete → each
  warp gets 1 issue per cycle effectively; latency-bound non-MMA paths
  don't get interleaved with another CTA's work).

### What 7d (classic DSM K/V share) would do
- Cluster size 2: CTA-A loads K/V into s_v_tile, CTA-B reads CTA-A's
  SMEM via DSM. Halves HBM traffic.
- Expected savings: **≤ 1.5 µs/CTA** on HBM-bound phase.
- Would leave the issue-starvation (≈ 42 µs) untouched → net effect
  indistinguishable from noise.

### What the real 7d (H-split cluster) would do
- Cluster of 2 CTAs, each handling H=8 heads of the same token t.
- Per-CTA SMEM drops: Q (16 → 8 KB), Q_pe (2 → 1 KB), logit scratch
  (16 → 8 KB). Combined with overlapping DSM K/V share, per-CTA peak
  drops from 96 → ~62 KB → **2 CTAs/SM possible** on B200.
- 2 CTAs/SM = 2× the warps available for issue → ~halves the
  issue-starvation wall.
- **Expected wall:** 46 → ~24 µs per-CTA. Dominant workloads (T large)
  gain ~1.9×; small workloads still go through legacy via the T-threshold
  dispatcher.
- **Effort:** Non-trivial — new kernel variant, new SMEM layout, new
  launch path with `cudaLaunchKernelEx` + `cudaLaunchAttributeCooperative`
  + `cudaLaunchAttributeClusterDimension`, DSM pointer mapping, cluster
  barriers. Estimated 200–400 LOC + debugging cycles.

## Decision

**Defer 7d**. Recommended replacement before the cluster rearchitecture:

1. Ship the current **T-threshold dispatch (`MLA_SPARSE_ATTN_CTA_PER_T_MIN_T`)**
   as the recommended CPT entry point. Claims the 0.65-0.88× wins on the
   largest workloads while dodging the 46 µs floor on small ones.
2. Before attempting 7d-proper, decode whether the 46 µs floor can be
   attacked by lower-occupancy-cost changes:
   - Cut SMEM (drop one of the V-pipes → 16 KB savings, at the cost of
     pipelining); would enable 2 CTAs/SM today.
   - Re-measure with Nsight Compute to confirm "warp stalls (no
     instruction)" is the top stall reason.

## Status

- Env flag allocation reserved: `MLA_SPARSE_ATTN_CTA_PER_T_CLUSTER=1`
  (not wired up).
- Stub kernel variant not added.
- Left TODO in `solution/python/kernel.cu` near `dispatch_cta_per_t` for
  future wiring.
