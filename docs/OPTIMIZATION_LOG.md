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
| exp02 (reverted) | 5: Fused logit+V online softmax | ~27× est. | **−18%** (short), **−27%** (long) | Same pattern as exp01. Halved ckv HBM traffic (−47% nominal) but introduced m/ℓ scalars across tiles + per-tile rescale path; compiler likely bumped registers above the 128/thread threshold for 2 CTAs/SM. Logs: `artifacts/full_bench_idea5_fused_2026-04-22.log`, `artifacts/full_bench_idea5_fused_condrescale_2026-04-22.log` (conditional rescale helped smoke but didn't rescue the full bench). |
| exp03 (reverted) | 5 + 7 combo | ~28× est. | **−25%** | Adding conditional `if (tile_max > m)` rescale guard didn't recover occupancy. |
| exp04 (reverted) | 3 (lite): H-major grid | ~27× est. | **−28%** | Swapped grid(T,H)→grid(H,T). No codegen change. Likely L2 / HBM channel contention from 16 co-scheduled H-CTAs of the same token hitting scattered sparse indices. |
| diag01 | runtime occupancy probe added | 36.45× | 0 | **Diagnostic instrumentation** (persistent): `cudaFuncGetAttributes` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at kernel configure time. |
| exp05 (reverted) | H_PER_CTA=2 (batch 2 heads per CTA) | smoke-only | **−42% long**, **−25% short** | Batched 2 heads (same token share sparse_idx) per CTA hoping to halve K/V HBM. Diag: regs 65→77, **local spill 0→24 B**, occupancy stayed at 2 blocks/SM. Long traces REGRESSED despite halved HBM (0.156→0.221 ms on workload 5096e459). **Key implication: long traces are NOT HBM-bound; they are compute or SMEM-bw bound.** Short traces regressed too, dominated by doubled softmax cost + register spill. Log: `artifacts/smoke_h2_per_cta_2026-04-22.log`. |
| **exp06 (kept)** | bf16 Q in SMEM + SMEM cap 100K→55K to unlock 4 CTAs/SM | modest positive | **−2.7% kernel time** (consistent across workloads) | Two tightly-coupled tweaks: (a) store Q in SMEM as bf16 (not fp32): halves s_qn/s_qp bytes, halves SMEM reads per FMA pair in the logit hot loop, drops regs/thread 65→58, saves 1.125 KiB SMEM. (b) Lower the `cudaFuncAttributeMaxDynamicSharedMemorySize` cap from 100 KiB to 55 KiB (tight fit for the actual 53.2 KiB footprint). Together these unlock **4 CTAs/SM** (was 2), confirmed by the runtime probe: `occ@cap=4`. Runtime geomean ratio 0.973 vs v4. All 23 workloads pass, no regressions. Log: `artifacts/full_bench_exp06_bf16q_4cta_2026-04-22.log`. |
| exp07 (reverted) | bf16 logits in SMEM + fold 1/sum into post-softmax write | ~flat vs exp06 | — | Stored softmax probs as bf16 and paired 2 rows per prob-load in the value pass (halving s_logits SMEM to 4 KiB and halving prob-reads in the inner loop). All 23 workloads still PASS, but `abs_err` pinned at bf16-quantum 1.56e-02 for every workload (was 1.9e-06 … 7.8e-03 in exp06), and `rel_err` blew up to ~30 on several. Kernel runtime was basically a wash with exp06 (no further win since exp06 already unlocked 4 CTAs/SM). Precision regression isn't worth the neutral performance → reverted. Log: `artifacts/full_bench_exp07_bf16logits_2026-04-22.log`. |

### Critical diagnostic (diag01) — occupancy theory DEBUNKED

Runtime probe output on v4 baseline:

```
[mla_sparse_attn] regs/thread=65  localSz=0  static_smem=0  max_dyn_smem=102400  max_threads=256  occupancy=2 blocks/SM
```

**v4 uses only 65 regs/thread (well below the 128 threshold for 2 CTAs/SM), has zero local memory spill, and already runs at 2 blocks/SM.** The −25-28% regressions seen in exp01-04 are NOT from an occupancy drop — they must have a different root cause.

Revised hypotheses for each regression:

- **exp01 (V_PIPE 2→4):** Doubling outstanding `cp.async` requests likely flooded the L1→L2→HBM fabric. With only ~ 1 kB of tile-row working-set per warp, deeper pipelining didn't hide more latency and added cp.async issue-queue pressure (instruction mix).
- **exp02 (fused):** Not a register blowup. Likely bottleneck: scalar FMA throughput. The fused kernel has to run the weighted-V FMAs (`BK * D_CK / THREADS` = 32 fp32 FMAs / thread / tile) on EVERY tile, instead of amortized over softmax-weighted values. Same issues/tile but 2× the FMAs.
- **exp04 (H-major):** All 16 heads of a token launching together smash the same 2-4 MiB of scattered K/V rows. Under T-major, co-scheduled CTAs are distinct tokens → different cache footprints → better L2 hit rate.

### Lessons / notes

- **Do not use SMEM/reg budget as a proxy for expected speedup.** v4 has ample headroom.
- **Verification path:** the runtime probe is now live. All future experiments should log their `regs/thread` + `occupancy`.
- **Orthogonal paths** that don't touch the occupancy balance:
  - **Reduce THREADS**: 256→128 halves SMEM-for-state and doubles occupancy headroom, but the packed output path is hard-coded to `D_CK == 2*THREADS`. Would require regrouping the output write.
  - **Persistent-CTA** (idea 10): one CTA does many (T,H) — no per-CTA SMEM growth, better cache warmth.
  - **H-fusion** (idea 3 cluster multicast / idea 11 UMMA per (T,H)) — shares the ckv tile across all 16 heads of a token. Fundamentally reduces HBM by 16× on the K side.
  - **Indexer split** — currently the kernel computes the full Q·K dot; if we precompute per-token top-k dots in a warp-specialized producer (FlashMLA style), we could fan out across heads more efficiently.

