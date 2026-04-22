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

## Summary of v5 experiments (current state)

**Kept:**
- `diag01` – runtime `regs/thread` + occupancy probe, prints once per process.
- `exp06` – bf16 Q in SMEM + tight 55 KiB SMEM cap → 4 CTAs/SM (was 2). Kernel runtime geomean −2.7% vs v4 (all 23 workloads pass).

**Tried and reverted (all −18 … −42% regressions):**
- `exp01` – V_PIPE 2→4 (deeper cp.async pipeline)
- `exp02` – fused logit+V online softmax (single pass over K)
- `exp03` – exp02 + conditional softmax rescale
- `exp04` – H-major grid reorder (grid(H,T) vs grid(T,H))
- `exp05` – H_PER_CTA=2 (batch 2 heads per CTA): exposed that **long traces are NOT HBM-bound**; halving HBM made them slower, not faster
- `exp07` – bf16 logits in SMEM: ~flat perf, precision pinned to bf16 quantum (abs_err 1.56e-02 uniformly), not worth it

**Key characterisation of v4 bottleneck** (from failed experiments):
- v4 is *not* register or SMEM limited (only 65 regs/thread, 2 CTAs/SM was self-inflicted by the 100 KiB cap).
- v4 is *not* HBM-bound on long traces — halving K/V traffic via head batching *slowed* them down.
- Most likely bottleneck: scalar FMA throughput + SMEM read bandwidth + `__syncthreads` serialization in the cp.async pipeline. All three contribute; no single "silver bullet" remaining.

**Remaining high-impact ideas (all High effort, High risk):**
- Idea 1 – TMA gather (`cp.async.bulk.tensor.gather4`) replaces cp.async tile loads
- Idea 6 – warp-specialised producer/consumer pipeline (async DMA → MMA)
- Idea 11 – tcgen05 UMMA value pass (previous attempt failed on SWIZZLE/numerical issues)
- Idea 3 – CTA clusters with multicast across H heads per token (this is the only remaining idea that directly attacks the "16 heads × same sparse_idx" redundancy without the L2 contention observed in exp04)
- Idea 2 – FP8 KV cache (requires schema change)

---

## diag02 — in-kernel phase cycle profile (post-exp06 baseline)

Added `PROFILE_PHASES` instrumentation (off by default) that makes block (0,0)
print `clock64()` deltas for five phases of every kernel launch:

```
qload  = Q load + sparse_idx gather/mask + n_valid_tiles reduction + s_logits -INF init
logit  = cp.async double-buffered tile loop computing all 2048 logits
softmax= block_reduce_max + exp2 + block_reduce_sum + log2f(sum)
vproj  = cp.async double-buffered tile loop for the weighted-V FMAs
owrite = packed bf162 store to `out`
```

Raw data: `artifacts/smoke_diag02_phases.log` (smoke pass, 23 workloads × 9
launches each). Post-warmup median cycles per phase, grouped by
`n_valid_tiles` (`nvt` = ceil(actual sparse support / BK=16)):

| nvt | qload | logit | softmax | vproj | owrite | total |   log/tile | vp/tile |
|-----|-------|-------|---------|-------|--------|-------|------------|---------|
|   1 |  ~8.5 k |  ~2.0 k |  2135  |  ~750 |  135 |  ~13.5 k | 1878 | 744 |
|   2 |  ~8.3 k |  ~3.2 k |  2135  | ~1505 |  133 |  ~15.3 k | 1600 | 752 |
|   3 |  ~8.2 k |  ~4.6 k |  2134  | ~2345 |  133 |  ~17.4 k | 1528 | 781 |
|   4 |  ~8.8 k |  ~6.1 k |  2139  | ~3230 |  134 |  ~20.4 k | 1520 | 808 |
|   6 |  ~8.3 k |  ~9.1 k |  2137  | ~4900 |  134 |  ~24.6 k | 1513 | 816 |
|   9 |  ~8.4 k |  ~13.3 k |  2135  | ~7410 |  136 |  ~31.4 k | 1475 | 823 |
|  18 |  ~8.5 k | ~25.8 k |  2135  | ~14.9 k |  133 | ~51.4 k | 1434 | 830 |
|  26 |  ~8.8 k | ~36.2 k |  2139  | ~21.6 k |  134 | ~69.0 k | 1392 | 831 |

### Linear model (fit across all 207 post-warmup launches)

```
total_cycles  =  10,805  +  2,249 * n_valid_tiles
               ├──────────┤    ├──────────────┤
               fixed overhead  per-tile cost
```

Per-phase decomposition:

```
qload     = 8,125 +   24 * nvt   (nearly constant — one-shot HBM + 3 syncs)
logit     =   549 + 1,388 * nvt   (per-tile logit dot+reduce)
softmax   = 2,134 +    0 * nvt   (independent of nvt; 2x block_reduce)
vproj     =  -138 +   837 * nvt   (per-tile weighted-V MAC from SMEM)
owrite    =   135                (1 packed bf162 store per thread)
```

### Phase-share breakdown (cycles pooled across launches)

| Regime               | qload    | logit    | softmax | vproj   | owrite |
|----------------------|----------|----------|---------|---------|--------|
| **all launches**     | **37.1%** | 34.1%  |  9.6%   | 18.5%   |  0.6%  |
| **long** (nvt ≥ 8)   | 16.8%     | **49.8%** |  4.2%   | **29.0%** |  0.3%  |
| **short** (nvt < 4)  | **53.8%** | 21.3%  | 14.2%   |  9.8%   |  0.9%  |

### Headline findings

1.  **Short traces are qload-bound** (54 % of cycles in a ~13 k-cycle prologue).
    This was invisible before: every prior experiment focused on the logit /
    value passes. Cutting qload from 8 k→3 k cycles would save ≈ 5 k × 113 /
    1,699 k = **~ 33 % of short-trace runtime**.
2.  **Long traces are logit-bound** (50 %). Logit is **1.65 × more expensive
    per tile than vproj** (1388 vs 837 cycles) despite both phases doing
    roughly the same amount of FMA work per tile. The difference is in the
    warp-reduce chain (5 `__shfl_xor` cycles × 2 logits/warp) plus the
    double `__syncthreads` around each tile.
3.  **Per-tile throughput ≈ 2,249 cycles.** At 1.8 GHz × 4 CTAs/SM × 148 SMs
    that is ≈ **7.3 TB/s effective HBM throughput** per tile (18,432 B /
    tile). B200 peak is ~8 TB/s → we're within 10 % of peak on tile loads.
    This **re-opens** the HBM-bound hypothesis that exp05 appeared to
    disprove: halving K/V reads (exp05) regressed because the doubled
    register pressure + local spill cost more than the HBM save — not
    because HBM was already free.
4.  **Softmax is a flat 2,134 cycles regardless of nvt** — 14 % of short-
    trace time but < 5 % of long-trace. No further softmax tricks worth it.
5.  **Output write is trivially cheap** (135 cycles, 0.3-0.9 %). Idea 18 is
    dead.

### Revised action plan (priorities)

**Tier 1 – Tackle qload first** (dominates short traces, 53 % of cycles):

| # | Idea | Expected win | Effort | Risk |
|---|------|-------------|--------|------|
| A | Async `cp.async` load of `sparse_idx` into SMEM while Q is loaded + start issue_logit_tile(0) **before** the idx-cache sync; remove the redundant -INF pre-init by carrying the `g<0` mask into softmax | 3-5 k cycles off short traces (~25 % of short-trace total) | S | Low |
| B | Replace the separate `block_reduce_max(local_max_valid_k_p1)` reduction with an atomicMax to shared int (skips one block-wide sync) | ~500 cycles | XS | Low |
| C | Kick off `issue_v_tile(0, 0)` at the **end of the logit pass** (before softmax) so its cp.async latency hides behind the softmax reductions | ~400 cycles off every trace | S | Low |

**Tier 2 – Cut per-tile logit cost** (50 % of long-trace):

| # | Idea | Expected win | Effort | Risk |
|---|------|-------------|--------|------|
| D | Collapse the 2 `__syncthreads` per logit tile into 1 named barrier (wait-on-producer, not wait-on-all) | ~100 cycles/tile × 128 tiles = ~13 k | M | Med |
| E | Vectorise warp-reduce: each warp handles 2 logits by doing 1 D_CK+D_KP dot per tile, reduced via a single group-shuffle tree (currently 2 serial reductions) | ~200 cycles/tile | M | Low |
| F | Warp-specialised pipeline (Idea 6): 2 warps do pure `cp.async` issue+wait, 6 warps do FMA. Moves the 2 syncs out of the critical path | ~300 cycles/tile | L | High |

**Tier 3 – Cut per-tile vproj cost / HBM traffic** (29 % of long-trace):

| # | Idea | Expected win | Effort | Risk |
|---|------|-------------|--------|------|
| G | FP8 KV (Idea 2): halves HBM per tile, vproj ≈ 0.5 × long-trace cycles | up to 30 % long-trace, 15 % smoke geomean | XL | High (schema) |
| H | CTA clusters + TMA multicast (Idea 3): 4 heads share one ckv gather via cluster-SMEM multicast, 4 × HBM reduction on ckv | up to 25 % long-trace | XL | High |
| I | tcgen05 UMMA value pass (Idea 11, retry): trades the scalar FMA pipe for tensor core; needs SWIZZLE_128B + per-output accumulation | 15-25 % long-trace if it works | XL | Very high (prior attempt regressed) |

**Next concrete step:** implement **Tier 1 item A** on a new sub-branch
(`perf_ideas_v5_qload`). It's XS-Low-Risk, directly hits the biggest
underexplored bottleneck, and re-uses the diag02 instrumentation to verify.
If it lands +3 % geomean, proceed to item C (free latency hide across a
phase boundary). After that, revisit item D (named barriers) as Tier 2's
lowest-risk entry.

