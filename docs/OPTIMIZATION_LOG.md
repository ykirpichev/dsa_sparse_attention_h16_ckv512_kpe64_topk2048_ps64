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

---

## itemA — skip s_logits pre-init + tight softmax loop bound (branch `perf_ideas_v5_qload`)

### Change (kernel.cu)
1.  Remove the block-wide `for (k=tid; k<TOPK; k+=THREADS) s_logits[k] = -INFINITY;` pre-init and its trailing `__syncthreads()`.
2.  In the logit pass, replace the `if (g<0) continue;` early-out with an **unconditional dot + predicated write**: lane 0 stores either `acc * sm_scale_log2e` (for g ≥ 0) or `-INFINITY` (for g < 0). The dot runs on the cp.async fallback (`ckv_flat[0]`) for g < 0 rows. Keeping the hot path branch-free matters — the earlier `if (g<0) { lane-0 write -INF; continue; }` variant regressed long-trace logit by +9% because the compiler couldn't schedule the unroll past the warp-uniform branch.
3.  Defer the `s_gidx[k]` load that selects the write value until just before the lane-0 store, so the compiler can overlap it with the FMA chain instead of keeping an extra live register through the dot.
4.  Bound the softmax max / sum loops by `n_valid_rows_full = n_valid_tiles * BK` (was `TOPK = 2048`). For `nvt=1` this collapses 2048 iterations over 256 threads → as few as 16 iterations total.

### diag02 smoke comparison (23 workloads × 9 launches, post-warmup medians)

Phase cycles, summed across all (T,H,nvt) groupings:

| phase   |  exp06  | itemA  |   Δ     |   Δ%     |
|---------|---------|--------|---------|----------|
| qload   | 159,830 | 155,508 | −4,322 | **−2.7%** |
| logit   | 166,669 | 172,406 | +5,737 | **+3.4%** |
| softmax |  40,565 |  26,442 | −14,123 | **−34.8%** |
| vproj   |  91,186 |  88,094 | −3,092 | **−3.4%** |
| total   | 461,035 | 444,979 | −16,056 | **−3.5%** |

Regime split:

| regime                  | Δ total | softmax | qload | logit  | vproj  |
|-------------------------|--------:|--------:|------:|-------:|-------:|
| short-trace (nvt < 4)   | **−5.3%** | −34.5% | −1.6% | +2.1% | −4.1% |
| long-trace (nvt ≥ 8)    | **−1.3%** | −34.4% | −3.6% | +3.4% | −2.2% |

Raw logs: `artifacts/smoke_qload_itemA_v4_phases.log` (itemA final), `artifacts/smoke_diag02_phases.log` (exp06 baseline).

### Notes & remaining regressions

-   **Softmax win is huge (−34.8 %)** — dwarfs everything else and justifies the change even with the logit regression.
-   **Long-trace logit regression (+3.4 %)** persists even after the hot-path branch removal. Probable cause: the unconditional dot runs on every row, even g < 0 rows for short traces (though those workloads are ≪ long-trace weight); the ternary select at the write adds ~1 cycle per logit. Not worth fighting further — the net is still positive.
-   **vproj also improved (−3.4 %)** even though we didn't touch the value pass. Likely explanation: the trailing `__syncthreads()` at the end of the logit pass now lands with all warps arriving together (no warp exits early via `continue`), reducing the barrier-last-arrival tail.
-   Full-benchmark geomean numbers pending (Modal workspace billing cap hit; smoke cycle counts used as primary signal).

### Full-benchmark verdict (23 workloads)

| binary | geomean speedup vs FI | geomean runtime | vs exp06 |
|--------|----------------------:|----------------:|---------:|
| exp06 (4 CTA + bf16 Q) | 27.12× | 52.2 µs | — |
| itemA (this change)    | **28.33×** | **50.6 µs** | **+4.46 % geomean, −3.18 % runtime** |

All 23 workloads pass. Biggest wins (−7 % to −14 %) are on the short-trace workloads (`68d6817d`, `b7668cfd`, `0a63b87b`, `9d4a5f21`, `f77df5ce`, `e6b849f2`) — exactly what diag02 predicted since those had softmax eating 14 % of short-trace cycles.

Raw log: `artifacts/full_bench_itemA_qload_2026-04-22.log`.

### itemB-attempted — warp-reduced atomicMax for `local_max_valid_k_p1` (reverted)

**Change.** Replaced `block_reduce_max` (3 internal syncs) with a warp-local `__shfl_xor_sync` max tree followed by one `atomicMax` to a shared int slot per warp (8 atomics instead of 256 to sidestep SMEM-atomic contention). Net sync delta: −2 `__syncthreads()`.

**Data.**

-   Smoke (diag02 phase cycles, summed over 23 workloads): **−3.4 % qload, −3.0 % logit, −2.3 % total** — looked like a clean win.
-   Full bench (2 independent runs, kernel wall-clock): **+2.5 % runtime consistently**. Long-trace workloads went from 152-155 µs → 157-160 µs in every run. Geomean vs FI: 28.12× (worse than itemA's 28.33×).

Register count (57 regs/thread) and CTAs/SM (4) identical to itemA (61 regs). Best guess: the two removed `__syncthreads()` were serving as implicit pipeline barriers that let the HBM load/store scheduler drain; without them some downstream stall worsens. Reverted. Raw log: `artifacts/full_bench_itemB_warpatomic_2026-04-22.log`.

Both a naive (per-thread atomicMax, 256-way contention) and warp-reduced variant regressed identically, so contention on the SMEM atomic is not the primary cause.

### itemC-attempted — pre-issue first v-tile before softmax (reverted)

**Change.** Moved `issue_v_tile(0, 0); cp_async_commit_group();` to the end of the logit pass, so the cp.async HBM load would overlap with the softmax max/sum reductions.

**Data.** Smoke (itemA → itemC): total **+2.0 %**. `vproj` improved −14.5 % on short-trace (n_valid_tiles≤1), but softmax regressed +23 % everywhere because the ~100-cycle cp.async issue cost moved from the v-tile loop into the softmax-measurement window. Long traces regressed outright (+1.7 % total) because their tile-0 wait was already cheap. Reverted. Raw log: `artifacts/smoke_qload_itemC_phases.log`.

### Next on the qload branch

With itemA banked (+4.46 % geomean) and items B and C falsified by full-bench data, the remaining Tier 1 ideas are exhausted. The best options now are:

-   **Tier 2 item D** — collapse the 2 `__syncthreads()` per logit tile into 1 named barrier (wait-on-producer, not wait-on-all). This is the lowest-risk Tier 2 entry and hits the long-trace critical path directly (logit is still 50 % of long-trace cycles). Expected ~100 cycles/tile × 128 = ~13 k cycles on long traces, ~25 % of the remaining logit budget.
-   **Tier 2 item E** — vectorise warp-reduce so each warp handles 2 logits per tile with a single group-shuffle tree. Moderate effort, complementary to D.
-   **Meta-observation from B/C.** diag02 phase timings aren't a reliable predictor of full-bench wall-clock when the change shifts pipeline barriers around. Future ideas should be validated on the full bench before being accepted.

### itemD-attempted — reorder logit tile loop to drop SYNC_B (reverted, branch `perf_ideas_v5_syncs`)

**Change.** Restructured the logit tile loop from

```
issue(next); commit; wait<1>;   // (or wait<0> on last)
SYNC_A;
compute buf = tile & 1;
SYNC_B;
```

to

```
wait<0>;
SYNC_A;                           // acts as barrier-before-next-issue
issue(next); commit;              // writes to other buf (safe w.r.t. compute)
compute buf = tile & 1;
// SYNC_B removed
```

One `__syncthreads()` saved per tile. Correctness preserved (iter+1's SYNC_A is block-wide, so by the time iter+1 issues to tile&1, all threads finished iter's compute on tile&1).

**Data (23-workload full bench, itemA → itemD_logit):**

| regime                    | latency Δ | speedup Δ |
|---------------------------|----------:|----------:|
| short traces (0.013-0.017 ms, nvt=1-2) | ~0 % | ~0 % |
| long traces  (0.13-0.16 ms, nvt≈26) | **+2.3 to +2.6 %** | **−1.5 to −4.6 %** |
| geomean                   | **+1.23 %** | **−0.65 %** |

Smoke correctness identical to itemA (same abs/rel err, regs=60, occ=4). Raw log: `artifacts/full_bench_itemD_logit_2026-04-22.log`.

**Root cause.** In the original layout, `issue + commit + wait<1>` runs *before* SYNC_A, so the cp.async-issue latency (per-thread instruction stream of 4× `cp.async.ca.shared.global` + 1× `cp.async.ca.shared.global`) overlaps with the block-wide barrier. itemD moves the issue *after* SYNC_A, serialising the two. On long traces (26 tiles × ~40 cycles of lost overlap ≈ ~1 k cycles), this precisely matches the +2-3 % regression we see.

**Generalised lesson (confirming B/C).** Three attempts now (B, C, D) to reduce syncs or repack pipeline operations have all regressed, because each removed/moved sync was serving implicit pipeline work:

-   **itemB** — removed syncs were implicit HBM-scheduler barriers.
-   **itemC** — pre-issued cp.async pushed its instruction cost into the softmax measurement window.
-   **itemD** — the `issue`-before-`SYNC_A` ordering gave the scheduler free barrier/async overlap that's lost when we reorder.

The current kernel is at a local optimum for local reorders of its 2-buffer cp.async pipeline. Further wins require either (a) a fundamentally different pipeline primitive (TMA replaces cp.async at a much lower instruction issue count) or (b) a different algorithm (fused online softmax removes the separate logit / softmax / vproj phases entirely).

### Revised next step — TMA gather (Idea 1)

The 3 failed local reorders (B, C, D) confirm that the remaining wins live in **changing the instruction mix**, not in shuffling existing syncs. qload is still the #1 cycle consumer (53 % of short-trace cycles), and per-thread `cp.async.ca.shared.global` is an instruction-issue-bound workload — exactly what TMA was built to replace.

Plan:
1. Set up a `CUtensorMap` for `ckv_cache` and `kpe_cache` host-side, pass both as kernel args (re-uses the existing contiguous `[S, D]` layout).
2. Replace the `issue_logit_tile` / `issue_v_tile` helpers with a single-thread TMA `cp.async.bulk.tensor.gather4.shared::cluster.global` per tile:
    -   gather 4 contiguous rows per call → 4 TMA calls per BK=16 tile.
    -   cuts per-thread instruction issue count from CHUNKS_PER_THREAD × THREADS = 1024 to O(4) per tile.
3. Coordinate completion via `mbarrier` rather than `cp.async.commit_group` / `wait_group`.
4. Keep the 2-buffer ping-pong; only the *producer path* changes.

Expected win: 20-40 % on short-trace qload (entirely instruction-issue-bound) and 5-10 % on long-trace (where qload is a smaller slice but ckv/kpe gather is still per-thread).

This is the lowest-risk high-impact option because the rest of the kernel (compute, softmax, vproj-FMA) stays exactly as it is today.

---

## v5_fused — fused online-softmax K loop (Idea 5, branch `perf_ideas_v5_fused`)

### Change

Collapsed the separate `logit → softmax → vproj` three-phase pipeline into a
single K-tile loop with online (FlashAttention-style) softmax:

```
m = -INF; l = 0; O[2] = 0
for tile = 0..n_valid_tiles-1:
    cp.async-load ckv+kpe tile          # ONE HBM load per tile (was 2)
    compute 16 logits warp-cooperative → s_logits_tile[16]
    tile_max = max(s_logits_tile)
    m_new = max(m, tile_max)
    alpha = exp2(m - m_new)             # (guarded for -INF)
    l *= alpha; O *= alpha
    for r = 0..15:
        p = exp2(s_logits_tile[r] - m_new)  # 0 for g<0 rows
        l += p; O += p * V[r]
    m = m_new
out = O / l; lse = m + log2(l)
```

Structural consequences:
1. **HBM ckv load halved.** The old kernel read each ckv tile twice
   (logit pass + vproj pass). The fused loop reads it once.
2. **s_logits[TOPK] deleted.** 8 KiB SMEM saved — total per-CTA SMEM drops
   from ~54 KiB to ~46 KiB. Opens headroom for V_PIPE=3 / TMA mbarriers.
3. **2 syncs per tile** (SYNC_A after wait, SYNC_B after logit write)
   instead of **4** (logit pass had 2, vproj pass had 2).
4. **Softmax phase deleted entirely.** No more block-wide max / sum
   reductions or the `s_logits` exp2 rewrite pass.

### Correctness

Smoke (10 sampled workloads): all PASSED with `abs_err` / `rel_err` in the
same range as `itemA` (e.g. `1.95e-03 / 2.03e-02` on `232ed014`).

Full bench: **23 / 23 PASSED**, max `abs_err` up to `1.56e-02` (same
tolerance bucket as itemA), max `rel_err` up to `2.57` (same as itemA's
`5.48` on `d57eb9e1` — driven by tiny reference values).

### Full-bench performance (23 workloads, 2 independent runs)

Absolute kernel latency (stable across runs; FI-reference speedup numbers
drift significantly between Modal invocations and aren't directly
comparable):

| regime              | itemA (ms)  | fused (ms)  | Δ%        |
|---------------------|------------:|------------:|----------:|
| short trace (nvt=1-2)   | 0.013-0.017 | 0.011-0.014 | **-10 to -15 %** |
| medium trace (nvt≈11)  | 0.083-0.090 | 0.072-0.076 | **-13 to -15 %** |
| long trace  (nvt≈26)   | 0.153-0.159 | 0.131-0.135 | **-14 to -15 %** |
| **geomean**         |             |             | **-11.75 %** |

Consistent 12-15 % wall-clock improvement on every workload class. The
long-trace win is slightly larger than short-trace (as expected: long
traces benefit most from halving ckv HBM traffic).

Register / occupancy trade-off: regs/thread went from 60 → 69 and
occupancy dropped from 4 CTAs/SM to 3 CTAs/SM. Even so, the fused loop is
faster in wall-clock because the per-CTA compute is now dramatically
faster (fewer phases, one K pass instead of two, no s_logits traffic).

Raw logs: `artifacts/full_bench_fused_v1_2026-04-22.log`,
`artifacts/full_bench_fused_run2_2026-04-22.log`.

### Why this worked (when the 3 local reorders didn't)

itemB, itemC, and itemD each tried to save 1-2 syncs or reshuffle the
cp.async timing while keeping the same 3-phase split. Each regressed
because any given sync was doing invisible pipeline work.

The fused loop takes the opposite direction: it **eliminates the phase
split entirely**, so the 2 remaining syncs serve fundamentally different
purposes (publishing cp.async writes, publishing logits) and can't be
reshuffled away. It's a structural change, not a local reorder.

Stacking with later optimisations: the fused loop frees 8 KiB SMEM and
is **additive with TMA** (next step), **CTA clusters** (needs careful
multicast design), and **UMMA for the K-dot / V-acc** (risky but high
reward).

### Next step

Continue with **TMA gather** on the fused producer path. Expected to add
another 10-15 % on short traces (instruction-issue bottleneck) and
~5-8 % on long traces (TMA bulk transfer is more HBM-efficient than
256-thread cp.async).


---

## v8_umma_svp (FAILED — archived): fused-heads UMMA logit + scalar value pass

**Motivation.** User set a 2× target (SoTA). After exhausting micro-opts on
`v5_fused` (~33× geomean), the only remaining lever for the compute-bound
tiles was Tensor Cores (UMMA / `tcgen05.mma.kind::f16`). Goal: use UMMA for
the logit pass, keep scalar value pass first, measure, then try UMMA V pass.

**Design.** Copied the archived `v7_umma` infrastructure (`tcgen05_ptx.h`,
`umma_desc.h`) and built a *new* fused-heads kernel on top of the fused
online-softmax idea:

- Grid `(T,)` (fused heads). SMEM `SWIZZLE_128B` layout for Q / K.
- `MMA_M=64, MMA_N=BK=16`, 32+4 K-atoms per tile for Q_nope·K_ckv^T and
  Q_pe·K_kpe^T respectively.
- Fused online softmax: warp 0 lanes 0..15 own one head each, maintain
  running `(m, l)` in SMEM; softmax output `P` broadcast to the whole CTA
  via a new small `s_p_row [HEADS, BK]` fp32 buffer.
- Scalar value pass reading directly from the swizzled `s_ckv` buffer via
  `swz128b_chunk16b_offset()` (no V^T transpose, no separate value-pass
  SMEM).

**Correctness.** All 23 full-bench workloads PASSED.

**Performance.** Geomean dropped from **~33×** (`v5_fused`) to **12.11×**:
short traces stayed fast (38-58×), medium traces regressed ~40 %, long
traces collapsed to 4-8×.

**Root cause.** Occupancy collapse from SMEM pressure, not a logit-pass
speed problem. Per-CTA SMEM for the fused-heads layout:

| Buffer    | Size    | Notes                                       |
|-----------|---------|---------------------------------------------|
| `s_q_n`   | 64 KiB  | `MMA_M=64` forces 64 rows but only 16 real  |
| `s_q_p`   |  8 KiB  | Same `MMA_M=64` padding                     |
| `s_ckv`   | 32 KiB  | `V_PIPE=2 × BK=16 × D_CK=512`               |
| `s_kpe`   |  4 KiB  | `V_PIPE=2 × BK=16 × D_KP=64`                |
| `s_gidx`  |  8 KiB  | `TOPK=2048` int                             |
| Misc      | ~2 KiB  | `s_p_row`, `s_m/l/scale`, mbar, tmem ptr    |
| **Total** |**~118 KiB** |                                         |

The B200 dynamic SMEM cap is 228 KiB → **1 CTA per SM**. The `v5_fused`
per-(T,H) kernel is ~46 KiB per CTA → **3 CTAs per SM**. All UMMA kinds
(`f16`, `tf32`, `f8f6f4`, `i8`) require `M ≥ 64`, so the `s_q_n` padding
cost is unavoidable unless we change instruction kind (no smaller one
exists for f16-accumulating f16-input UMMA on SM100).

Raw totals on long traces (T≈4096):
- `v5_fused`: 65 536 CTAs × `30k cycles` / (148 SMs × 3 occ) ≈ 4.5 M
  cycles per SM.
- `v8_umma_svp`: 4 096 CTAs × `480k cycles` / (148 SMs × 1 occ) ≈
  13.3 M cycles per SM → matches the ~3× regression we observed.

Even fixing the broken V-pass UMMA (next on the Idea list) would not
recover: best case is ~2× per-CTA speedup from scalar-V → UMMA-V, but we
are paying a 3× occupancy penalty. Net would still regress vs `v5_fused`.

**Decision.** Abandoned. Reverted to `v5_fused`. Archived kernel in
`docs/archive/kernel_v8_umma_svp_attempt.cu` for future reference.

**Key lesson.** On this workload the compute-bound bottleneck is V
accumulate (≈60 % of per-tile cycles) and the kernel is already close
to L2-bandwidth-bound at `v5_fused` occupancy. The only way UMMA could
help is a per-(T, H) variant where the CTA structure doesn't force the
`MMA_M=64` padding. Such a variant needs 64 `(t, h)` pairs to share the
same K — which they fundamentally don't (each `t` has its own
`sparse_idx`). There is no architecturally clean UMMA path for this
workload at its current shape.

### What a realistic 2× would need

Achieving 2× vs `v5_fused` would require one of:
1. **Algorithmic change**: reduce total K/V HBM traffic (FP8 KV cache,
   cache-reuse across adjacent T via cluster multicast). FP8 KV halves
   HBM bytes → ~15-25 % long-trace win. Requires quantization in
   producer and dequantization in the kernel. Moderate risk.
2. **Persistent kernel / grid-level scheduling**: amortize kernel launch
   / prologue cost across many tokens. Helps short traces where we
   already do well; modest geomean win.
3. **Split-K for short traces**: extra parallelism for T < 148. Helps
   the very fast workloads get even faster. Low marginal geomean
   contribution because those workloads are already small absolute
   time.

None of these individually delivers 2×. The realistic ceiling from the
current architecture with `v5_fused` as the baseline is probably
~1.2-1.3× from stacking multiple small wins.

### Next step

Revert to `v5_fused` (`wombat_dsa_sparse_mla_cuda_v5_fused`). Pivot to
lower-risk optimisations that stack additively without touching the
core CTA / SMEM structure: HFMA2 for the logit dot product (Idea 13),
FP8 KV cache feasibility study (Idea 2), and per-T-bucket kernel
variants (Idea 20) driven by a runtime dispatch heuristic.

---

## v9_h4_scalar (FAILED — archived): per-(T, H/4) CTA, pure scalar

**Motivation.** After v8's UMMA failure, reproposed a per-(T, H/4) CTA that
packs 4 heads per block as a *stepping-stone* for a later UMMA-V swap
(v10). The hypothesis was that V-reuse across 4 heads in the inner r-loop
would give ~15-25 % on long traces even in the pure-scalar variant, and
set up the SMEM / register budget for UMMA V later.

**Design.**
- Grid `dim3(T, H/4)`, 256 threads. Per-CTA state: `m[4], l[4], o0[4],
  o1[4]` plus temporaries.
- Logit pass: each warp does 8 warp-reduced dots per tile (vs 2 in v5),
  covering the full `H_PER_CTA × BK = 64` logit grid.
- Softmax/V: classic online-softmax per head; inner r-loop reuses one
  `bf162` V load for all 4 heads (`p * v0, p * v1` for each h).
- SMEM `~49 KiB` per CTA (Q × 4 heads pushed this up from 46 KiB).
- `__launch_bounds__(256, 3)` forces 80 regs/thread, `localSz=32 B`,
  `occ=3 CTAs/SM` — identical to v5_fused.

**Correctness.** All 10 smoke workloads PASSED.

**Performance (smoke, abs latency ms vs `v5_fused`).**
| workload    | v5_fused | v9_h2 | v9_h4 |
|-------------|---------:|------:|------:|
| 5096e459 (long)  | 0.131 | 0.230 | 0.432 |
| d57eb9e1 (long)  | 0.131 | 0.231 | 0.430 |
| 232ed014         | 0.076 | 0.132 | 0.242 |
| 4c46a94b         | 0.070 | 0.121 | 0.223 |
| 9f3f891b         | 0.027 | 0.043 | 0.074 |
| 0a63b87b (short) | 0.015 | 0.018 | 0.027 |
| 0c23b10c (short) | 0.010 | 0.013 | 0.016 |

**Every workload regresses — strictly linearly in `H_PER_CTA`:**
- `h4` is **~3.3× slower** than `v5_fused` everywhere (long + short).
- `h2` is **~1.76× slower** than `v5_fused` everywhere.
- Doubling `H_PER_CTA` nearly doubles wall time.

**Root cause.** The hoped-for V-reuse amortisation simply is not there.
On `v5_fused`:
- 3 CTAs/SM × 148 SM × 16 heads/token → hardware grid scheduler co-
  schedules same-`t` CTAs within an L2 window. Every K/V row is served
  from L2 for the 15 subsequent head-CTAs that want it. Explicit
  in-register V reuse in `v9_h4` replaces 16-way L2 sharing with 4-way
  per-CTA-register sharing — no improvement, because the L2 path was
  already free.
- Per-CTA SMEM-load bandwidth is NOT the bottleneck for v5; SMEM reads
  overlap with FMAs and `exp2f` issues. Merging 4× the V FMAs and 4× the
  `exp2f`s into one CTA just piles 4× more work onto the same compute
  pipe, for no gain — and the grid is 4× smaller, so SM-time is
  unchanged best case, strictly worse in practice (register / scheduler
  overhead per extra head).

**The strict linear scaling is diagnostic.** Per-CTA time scales as
`H_PER_CTA × (v5 per-CTA cost) + overhead`. Dividing by `H_PER_CTA`
CTAs keeps total work unchanged → same wall time *iff* there's no
overhead. We observe **1.76× regression per doubling**, meaning there IS
a per-extra-head overhead of ~1.76× / 2 = ~0.88 on top of the multi-head
work. Likely a mix of (a) register-file pressure hurting FMA ILP, (b)
SMEM bank-conflict-less but broadcast-heavy reads of `s_logits_tile[h*BK+r]`,
(c) cross-head dep chains the compiler can't schedule through.

**Implication for the planned UMMA-V on top (v10).**
- The scalar baseline is already 1.76-3.3× slower than v5.
- UMMA V would need to reclaim that AND add 20 % to beat v5.
- `tcgen05.mma.kind::f16` MMA for `[heads=4..8, D_CK=512, BK=16]` is
  architecturally possible (via `b_major=MN` on the natural V layout),
  but the best-case theoretical saving is the V FMA cost — maybe 50 %
  of per-CTA time — which gives ~2× on that slice, i.e. new
  `v9_h4` per-CTA = ~1.8× → still ~45 % slower than v5_fused.
- Even under optimistic UMMA-V assumptions, v10 cannot beat v5_fused.
  The h4 restructure is the wrong vehicle for UMMA on this workload.

**Decision.** Abandon D. Revert to `v5_fused`. Archive the v9 kernel in
`docs/archive/kernel_v9_h4_scalar_attempt.cu`.

**Bottom line (2× target).** v5_fused is *at or very near* the ceiling
for this kernel shape on B200. Both UMMA directions (v8: fused-heads,
v9/v10: head-grouped) have now been eliminated by architecture
constraints:

- UMMA `M≥64` requirement × `H=16` makes fused-heads unaffordable on
  SMEM / occupancy.
- Head-grouping CTAs is strictly pessimal in scalar form; UMMA on top
  cannot recover 1.76-3.3× per-CTA overhead plus hit `target > v5`.

The 2× target appears unreachable from `v5_fused` without an
algorithmic change that is *outside the contest rules as understood*
(e.g. pre-quantised KV cache). Realistic remaining wins are
low-single-digit % from micro-opts (HFMA2, per-T-bucket dispatch,
persistent CTAs).

---

## Phase 1 — `cp.async` Q load + `int4` sparse_idx prescan + zero-init invalid rows (branch `phase1-qasync-idx4-zeros`, commit `98752bc`)

**Status.** KEPT. Full bench geomean latency **−16.4%** vs v5_fused, speedup geomean **+20%** (all 23 workloads PASS). Log: `artifacts/phase1_full_bench_20260422T224042.log`.

**Rationale.** The `v5_fused` prologue was a synchronous gauntlet: 3× `ldg.128`
per thread for Q_nope (256 thr × 8 B × 3 = 6 KiB), then a scalar per-thread
loop over 2048 `sparse_idx` entries (8 scalar loads/thread). All three ideas
are pure micro-opts, none touches the hot loop, and all three share one
theme: cut setup latency and overlap what's left.

**Changes (all in `solution/python/kernel.cu`, single-kernel path).**

1. **`cp.async` for Q_nope + Q_pe.** Replaced 12×128-bit synchronous `uint4`
   reads with non-blocking `cp.async.ca.shared.global` (group 0). Q now
   arrives in SMEM as a background copy while the sparse_idx prescan runs.

2. **`int4` vectorised sparse_idx prescan.** The max-valid-k scan was a
   scalar loop reading 2048 `int32` one at a time per thread (8 loads). With
   `TOPK = 2048` and `THREADS = 256`, each thread owns `2048/256 = 8`
   indices = 2 `int4` (128-bit) words. Rewrote the prescan as 2 `int4`
   loads/thread + 4 scalar extracts per word. Same work, 4× fewer
   transactions, and overlapped with the Q `cp.async`.

3. **Zero-init SMEM for invalid rows.** `g_idx < 0` tiles previously fell
   back to `ckv_flat[0]` (which was simply wasted HBM bw — the zero-guard
   later masked them out of the logit/value paths). Replaced with a local
   zeroed store into SMEM. Saves 1-8 HBM transactions per invalid row per
   tile across the K loop.

**Measured per-workload (23 workloads, absolute latency in ms).**
- Tiny workloads (8-10 µs baseline) drop 20-38% → 5-8 µs.
- Mid workloads (50-75 µs baseline) drop ~10-15%.
- Long workloads (123-125 µs baseline) drop ~6-8%.

Small-workload gains dominate because the prologue is a larger fraction of
wall time. On long traces the prologue is dwarfed by the K-loop so the
micro-opts see diminishing returns — which flagged the need for **Phase 3
(split-K)** to attack the K-loop directly.

**Safety.** Added `cp_async_wait_group<0>()` before the early-return branch
for `n_valid_tiles == 0`; otherwise Q cp.async groups could leak into the
next kernel slot. No other correctness changes.

| Rev | Idea(s) | Geomean latency Δ | Geomean speedup Δ | Notes |
| --- | ------- | ----------------- | ------------------- | ----- |
| **phase1 (kept)** | cp.async Q + int4 prescan + zero-init invalid | **−16.4%** | **+20.0%** | 23/23 PASS. Gains dominated by small workloads. Log: `artifacts/phase1_full_bench_20260422T224042.log`. |

---

## Phase 3 — Split-K (FlashDecoding-style) for small T×H grids (branch `phase3-splitk`)

**Status.** KEPT. Full bench geomean latency **−30.35% vs Phase 1**
(and **~−42% vs v5_fused**); geomean speedup **+43% vs Phase 1** (~+72%
vs v5_fused). 23/23 PASS. Log:
`artifacts/p3/phase3_full_bench_20260423T055844.log`.

**Rationale.** After Phase 1 the long (T × H small) workloads still sat at
~115 µs, dominated by the K-loop running on too few CTAs — e.g. T=4,
H=16 → 64 CTAs, vs ~444 available slots on B200 (148 SMs × 3 CTAs/SM at
our occupancy). The SM array was half-idle. Split-K is the textbook fix:
decompose the K dimension across `G` groups, each slice handles `TOPK/G`
sparse indices, multiplies the effective grid by `G` and then reduces
the G partial outputs to the final `O` + LSE.

**Kernels added (all in `solution/python/kernel.cu`).**

1. **`mla_split_partial_kernel<int G_>`** – Templated copy of the main
   kernel where each CTA handles only `TOPK_PG = TOPK/G_` of the sparse
   indices. Outputs **unnormalized** partials: `O_partial` (bf16 per-head
   context × D_CK), `m_partial` (fp32, per-head max logit), `l_partial`
   (fp32, per-head softmax denominator). SMEM is smaller because
   `s_gidx` is only `TOPK_PG`; everything else is the same as the main
   kernel (including `__launch_bounds__` and 3 CTAs/SM at the tight
   dyn-SMEM cap).

2. **`mla_split_reduce_kernel<int G_>`** – One block per (t, h) that
   log-sum-exp-merges the `G_` partials into the final `out` + `lse_out`.
   Uses the stable pattern:
   `m = max(m_g) ; p_g = exp(m_g - m) · l_g ; Σp_g = l ; o = Σ (o_g · exp(m_g - m)) / l ; lse = m + log(l)`.

3. **Host dispatcher (`pick_split_G`)** – Selects `G ∈ {1, 2, 4, 8}`
   based on `T*H`, filling CTA slots up to ~444 with a **no-split floor
   at 64 CTAs**. Below the floor the reducer + second-launch overhead
   (~3-5 µs) exceeds any K-loop gain, so tiny workloads stay on the
   original single-kernel path.

```c++
// SPLIT_TARGET_SLOTS = 444 (148 SMs × 3 CTAs/SM), SPLIT_MIN_CTAS = 64
if (base_ctas < SPLIT_MIN_CTAS)            return 1;   // too tiny to amortise
if (base_ctas >= SPLIT_TARGET_SLOTS)       return 1;   // fill already
if (base_ctas * 2  >= SPLIT_TARGET_SLOTS)  return 2;   // [222, 443]
if (base_ctas * 4  >= SPLIT_TARGET_SLOTS)  return 4;   // [111, 221]
return 8;                                              // [64, 110]
```

4. **Scratch allocator (`SplitKScratch`)** – Per-process singleton with
   mutex, lazily grows `cudaMalloc` buffers for `O_partial`,
   `m_partial`, `l_partial` to fit the worst-case
   `B × G_max × T × H × D_CK`. Reused across every split-K launch.

5. **Occupancy probe** – `configure_kernel_attrs_once()` now also logs
   `regs/thread`, `static_smem`, `max_dyn_smem`, and
   `occ(40160 B)` for each `mla_split_partial_kernel<G_>` instantiation.
   Measured: **68 regs/thread, 3 CTAs/SM**, identical profile to the
   main kernel.

**Measured Phase 3 vs Phase 1 (23 workloads).**

| bucket | Phase 1 µs | Phase 3 µs | Δ µs | Δ % |
| --- | ---: | ---: | ---: | ---: |
| 8 × tiny (T*H < 64) | 8-27 | 8-27 | ~0 | ~0% (by design, no-split path) |
| 3 × medium | 66-71 | 39-57 | −22 | −20 … −41% |
| 11 × long-T*H (64…256) | 102-125 | 34-73 | −48 | −43 … −67% |
| 1 × regression | 8 (68d6817d) | 11 | +3 | +37.5% |

**Top wins (Δµs):**
- `ddfa9e34`: 102 → 34 µs (**−66.7%**)
- `d57eb9e1`: 123 → 43 µs (**−65.0%**, T*H ≈ 64 → G=8 activated)
- `ae4219a9`: 125 → 53 µs (**−57.6%**)
- `2207f0fd`: 123 → 54 µs (**−56.1%**)
- `02d6ae9c`: 125 → 56 µs (**−55.2%**)
- `78b2e11c`: 125 → 62 µs (**−50.4%**)

**Regression:** `68d6817d` from 8 → 11 µs (+37.5%). Absolute 3 µs delta on
an already-150×-speedup workload: its `T*H` lands in the [64, 110]
G=8 bucket but its nvt is tiny (short-K sparse), so the extra reducer
launch (~3 µs) is the entire difference. This can't be resolved on the
host without runtime info on `nvt`. Leaving as-is; the 11 µs result is
still a 100×+ speedup vs reference.

**Tuning notes captured.**
- First attempt used `SPLIT_MIN_CTAS = 128`. This missed the T*H=64
  long-K workloads (e.g. `d57eb9e1`), so the floor was lowered to 64
  with a G=8 bucket added. That single change unlocked the largest
  wins in the full bench.
- Only one `G` variant is live per launch; the G=16 instantiation exists
  in source for future use (trace > 500 µs regime) but isn't reachable
  by the current dispatcher.

| Rev | Idea(s) | Geomean latency Δ (vs P1) | Geomean speedup Δ (vs P1) | Notes |
| --- | ------- | --- | --- | --- |
| **phase3 (kept)** | split-K, G ∈ {1,2,4,8}, dispatcher with floor=64 | **−30.35%** | **+43.3%** | 23/23 PASS. 11 workloads improved 43-67%. One +3 µs regression on a 150×-speedup tiny workload. Log: `artifacts/p3/phase3_full_bench_20260423T055844.log`. |

**Combined Phase 1 + Phase 3 vs v5_fused baseline (derived):**
- latency: `0.836 × 0.697 ≈ 0.582` → **−41.8%**
- speedup: `1.20 × 1.43 ≈ 1.72` → **+71.5%** geomean speedup over baseline.

**Future Phase 3 follow-ups (not implemented here).**
- **CTA-cluster sparse_idx** (Idea S6): within a cluster of G partial
  CTAs, all share the same full 2048 sparse_idx. Loading it once via
  multicast DSMEM would save (G−1) × 2048 × 4 B HBM per (t,h). Non-trivial
  (DSM across CTAs in a cluster), only worth the complexity if split-K
  becomes the dominant hot path and the HBM of sparse_idx shows up in a
  profile (it currently doesn't with the Phase 1 `int4` prescan).
- **Device-side split decision** (bail reducer early if all G partials
  had `nvt_slice==0` except one) would eliminate the `68d6817d`
  regression. Not free — requires a zero-copy flag. Defer.

