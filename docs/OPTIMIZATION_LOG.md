# DSA Sparse MLA — Optimization Log (branch `perf_ideas_v5`)

Tracks incremental optimizations on top of `v4_sm100a` (submission-v1) kernel.
All measurements on Modal B200 (1 GPU), 23 workloads, geomean speedup vs FlashInfer reference.

Baseline:
- Commit: `60192e4` (clean_baseline / submission-v1 restored)
- Kernel: `v4` — per-(T,H) CTA, cp.async double-buffered logit + value passes, scalar FMA.
- Result: **geomean 36.45×** (from `artifacts/full_bench_v4_restored_2026-04-22.log`).
- Per-workload speedup range: 14.6× … 163.7× (long sequences bottomed at ~15× over FI).

## Ranked backlog (from research: FA-4, FlashMLA, vLLM DSA PRs, PTX ISA)

| # | Idea | Impact | Effort | Risk |
| - | ---- | ------ | ------ | ---- |
| 4 | L2 `EVICT_FIRST` cache hints on K/V cp.async (`createpolicy` + `L2::cache_hint`) | Low | XS | Low |
| 14 | Store Q in SMEM as bf16 (saves 2.25 KiB, enables HFMA2 in logit dot) | Low-Med | XS | Low |
| 16 | Deeper pipeline `V_PIPE 2 → 4` | Low-Med | XS | Low (SMEM up to ~90 KiB) |
| 15 | Cut per-tile `__syncthreads` via named barriers (logit+value passes) | Med | S | Low |
| 17 | Keep live sparse indices in registers during value pass | Low | XS | Low |
| 18 | Vectorised / coalesced bf162 output writes (no-op today; audit) | Low | XS | Low |
| 7 | Conditional FA-style rescale (skip when new max ≤ current max) | Med | M | Med |
| 5 | Fused logit + weighted-V online softmax (single pass over K) | High | L | Med |
| 8 | Polynomial `exp2` emulated on FMA units (frees MUFU pipe) | Low-Med | M | Low |
| 13 | `HFMA2` bf16×2 logit dot (requires bf16 Q + careful error behaviour) | Med | M | Low |
| 9  | Split-K for short-T tail | Med | M | Low |
| 10 | Persistent CTA scheduler | Med | M | Low |
| 19 | Programmatic Dependent Launch between layers | Small (framework) | M | Med |
| 20 | Heuristic-switched kernel variants per T bucket | Med | M | Med |
| 1  | TMA gather (`cp.async.bulk.tensor.gather4`) replaces cp.async tile load | High | L | Med |
| 6  | Warp-specialised producer/consumer (async DMA → MMA) | High | L | Med |
| 11 | tcgen05 UMMA **value** pass on top of v4 (SWIZZLE_128B layout) | High | L | High |
| 12 | Per-(T,H) tiny UMMA value (keep grid parallelism) | Med-High | XL | High |
| 2  | FP8 KV cache | High | XL | High (schema) |
| 3  | CTA clusters with K/V multicast across H heads per token | High | XL | High |

## Run-log (measured)

| Rev | Idea(s) | Geomean | Δ vs v4 | Notes |
| --- | ------- | ------- | ------- | ----- |
| baseline (`60192e4`) | — | 36.45× | 0 | submission-v1 kernel restored. |
| exp01 (reverted) | 16: V_PIPE 2→4 | ~27× est. | **−25%** | Regressed uniformly (long seqs: 14.6→10.6×, 17.4→12.6×, …). Root cause: 90 KiB SMEM footprint dropped occupancy 2→1 CTA/SM. Log: `artifacts/full_bench_idea16_vpipe4_2026-04-22.log`. |

### Lessons / notes

- **V_PIPE=4 is occupancy-limited.** On B200 with 228 KiB SMEM/SM, 2 × 90 KiB = 180 KiB should nominally fit, but measured regression shows the scheduler drops to 1 CTA/SM (likely because the bumped attribute cap raises the SMEM carveout reservation). To try a deeper pipeline we'd need to **shrink the static SMEM** first (e.g. Idea 14: bf16 Q saves 2 KiB; merge s_logits/s_gidx regions; halve BK to 8 which would also halve the tile, freeing budget for V_PIPE=4 with no net SMEM growth).
- **Orthogonal path:** rather than deeper pipeline, focus on **reducing dominant work** (Ideas 5, 11, 13, 14) and **cross-head data reuse** (Ideas 3, 11) — long sequences (T≥256) are the geomean drag at 11-17×.

