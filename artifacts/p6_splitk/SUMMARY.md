# Phase 6 — Summary (try-all-four-in-order)

**Branch:** `phase5-ws-parissue`
**Date:** 2026-04-23
**Target:** B200 (sm_100a), `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`

| Step | Idea | Effort | Expected | Actual | Status |
|------|------|--------|----------|--------|--------|
| 6a | Split-K G=16 threshold tuning (narrow tier) | ~30 min | 1-3 % | **−0.2 % geomean, 2 wl at −14 %** | ✅ **Landed** (default ON) |
| 6b | Fused partial+reduce (last-CTA pattern) | ~60 min | 5-10 % on split subset | 0.00 % SPLIT / +0.46 % ALL | ⚠️ Kept behind flag, default OFF |
| 6c | Cluster-shared (DSM) sparse_idx broadcast | 2-4 hrs est | 1-2 % short-K | Analysis: < 1.5 %, below noise floor | ⏭️ Skipped with analysis |
| 6d | tcgen05.mma for V-accum | multi-day | 10-20 % | Analysis: 0.1 % (not compute-bound) | ⏭️ Skipped with analysis |

## Wins & learnings

### Landed: Split-K G=16 narrow-tier promotion (6a)

Two workloads at −14–17 % each, modest −0.2 % geomean overall. Behind `MLA_SPARSE_ATTN_SPLIT_G16` env flag (default ON). No kernel changes — dispatcher threshold only.

- `ddfa9e34` (T=6 H=16): 35 → 29 µs (−17.1 %)
- `d57eb9e1` (T=6 H=16): 42 → 36 µs (−14.3 %)
- The wider promotion (`< 64 base_ctas → G=16`) **regressed +26 %** on short-K T=2 workloads, so the policy was narrowed to only the `[64, 110]` tier.

### Exploratory: Fused partial+reduce (6b)

Implemented the full last-CTA pattern (`__threadfence` + atomic counter + in-place reduce + counter reset) correctly — all workloads PASS on both fused-ON and fused-OFF paths. But on B200 + recent CUDA driver, kernel-to-kernel launch latency is already ≤ 500 ns (below our ≈ 1 µs CUDA-event timer resolution). Result: no measurable signal either way.

Kept behind `MLA_SPARSE_ATTN_FUSED=1` env flag for future reuse (older drivers, different workload mix, bigger split fan-out).

### Skipped with quantitative analysis: DSM broadcast (6c)

`sparse_idx` accounts for 0.26 % of HBM traffic on every workload on this bench. Max achievable savings ~1.5 % (below noise floor). Cluster-launch implementation cost not justified at this workload shape. Documented conditions under which it would become worthwhile.

### Skipped with quantitative analysis: tcgen05.mma V-accum (6d)

V-accum compute intensity is only **17 MFLOP per workload** (65 k FMA per CTA × 128 CTAs for T=8 H=16), which equates to 0.07 µs on scalar FP32. The kernel is **memory/latency-bound**, not FMA-bound — the 15 % WS3 regression from thread-halving was register-pressure- and ILP-bound, not FMA-throughput-bound.

tcgen05.mma's N=128 natural shape vs our N=1 GEMV shape is a 1/128 efficiency mismatch. Recovering efficiency requires batching H = 16 heads per CTA, which is a multi-day kernel rearchitecture. Documented as the right next-project target if more perf is needed.

## Composite performance vs baseline (min-of-2 runs)

| Group | Baseline (pre-6a, Phase 5 endpoint) | Post-6a | Δ |
|-------|-------------------------------------|---------|---|
| ddfa9e34, d57eb9e1 (T=6 H=16) | 35–42 µs | 29–37 µs | **−14 % each** |
| All others (21 wl) | unchanged | unchanged | ±noise |
| Geomean (all 23) | 0.0266 ms | 0.0264 ms | **−0.8 %** |

## Artifacts

All under `artifacts/p6_splitk/`:

- `ab_report.md` — Step 6a (landed)
- `ab_report_6b_fused.md` — Step 6b (exploratory, default OFF)
- `ab_report_6c_dsm.md` — Step 6c (analysis-gated skip)
- `ab_report_6d_tcgen05.md` — Step 6d (analysis-gated skip)
- `diag_*.log` — dispatch-diagnostic bench runs
- `full_g16_*.log` — Step 6a A/B runs (wide and narrow policies)
- `full_fused_*.log`, `smoke_fused_*.log` — Step 6b runs
- `verify_*.log` — post-flip verification runs

## Remaining opportunities (deferred)

- **CTA-per-t refactor** with H-batched Q/O and cluster-shared K/V — simultaneously reduces HBM reads (cluster-DSM) **and** enables tensor-core V-accum with N=H=16. Estimated effort: 1–2 sessions. Expected upside: likely 10–25 % on long-K workloads if HBM traffic reduces by >2×.
- **Re-profile under CUPTI** for cleaner signal below 1 µs. Then re-evaluate the fused-reduce path and smaller targeted micro-optimizations that are currently invisible.
- **WMMA m16n8k16 experiment** on existing per-(t,h) grid with a "N=1 padded to N=8" approach — documented as expected loss in 6d, but worth a 2-hour smoke to confirm the theoretical bound empirically.

## Configuration state after Phase 6

| Env flag | Default | Effect |
|----------|---------|--------|
| `MLA_SPARSE_ATTN_SPLIT_G16` | ON  | G=16 on `[64,110]` base_ctas tier |
| `MLA_SPARSE_ATTN_FUSED`     | OFF | Last-CTA fused reduce |
| `MLA_SPARSE_ATTN_SPLIT_DIAG`| OFF | Per-(T,H) dispatch diagnostic print |
| `MLA_SPARSE_ATTN_TMA`       | OFF | Half-row TMA kernel variant |
| `MLA_SPARSE_ATTN_WS`        | OFF | WS triple-buffer variant |
| `MLA_SPARSE_ATTN_WS3`       | OFF | Full warp-specialized variant |
