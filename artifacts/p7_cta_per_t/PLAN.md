# Phase 7 — CTA-per-t rearchitecture

**Goal:** Collapse the `(T, H)` grid to `(T)` so all `H = 16` heads of the same
token share one load of the sparse K/V working set. This attacks the *real*
bottleneck — HBM / latency, not FMA throughput — and also unlocks
tensor-core V-accum (`tcgen05.mma`) because we gain an `N = H = 16` (or more,
with cluster-sharing) GEMV-→-GEMM dimension.

Today: one CTA per `(t, h)`, 16 CTAs per token, each redundantly loading the
same `TOPK = 2048` K/V rows. Theoretical HBM floor for the longest workload
(`564007ac`, T=8 H=16) ≈ 100 µs; we hit 73 µs ⇒ ~27 % efficiency above the
floor. Removing the redundant 16× K/V reads is the single biggest lever left.

## Scope classification

| Item | Size | Risk |
|------|------|------|
| Core refactor (grid, SMEM, loops, Q/O batching) | ~1 day | Med |
| Correctness matching across all 23 workloads | ~½ day | Low-Med |
| Tensor-core V-accum via `tcgen05.mma` | ~1-2 days | Med-High |
| Cluster-shared K/V (cluster = 2-4 CTAs on a single t) | ~1 day | Med |
| Split-K (G ≥ 2) integration / dispatcher | ~½ day | Low |
| Benchmark sweep + A/B + write-up | ~½ day | Low |
| **Total** | **3-5 days** |

Staged so every phase is independently shippable behind an env flag.

## Phase 7a — Scalar CTA-per-t prototype (**first step**)

**Claim to validate:** sharing K/V across H=16 cuts K/V HBM reads by up to
16× for TOPK-dominated workloads, yielding large wins on long-K traces.

### Grid & threads

- Old: `grid = (T, H)`,  `block = 256`, one CTA per `(t, h)`
- New: `grid = (T,)`,     `block = 256`, one CTA per `t`

With `block = 256` we keep:
- 128 threads per half of the D=512 V-channels (two threads per output float)
- 16 threads per row of BK (so 16 H-head dots in parallel on a 16-row tile)
- Same L1 / L2 / register occupancy story as today

### SMEM layout (per CTA)

| Buffer | Today (per-head) | 7a (H-batched) | Δ |
|--------|-------------------|----------------|---|
| `s_qn` (Q nope)   | 1 KiB       | **16 KiB** (H × D_CK × 2 B)         | +15 KiB |
| `s_qp` (Q pe)     | 128 B       | **2 KiB**  (H × D_KP × 2 B)         | +1.9 KiB |
| `s_gidx`          | 8 KiB       | 8 KiB (unchanged — shared)          | 0 |
| `s_logits_tile`   | 64 B (BK)   | **1 KiB** (H × BK × 4 B)            | +1 KiB |
| `s_reduce`        | 32 B        | **512 B** (H × NUM_WARPS × 4 B)     | +0.5 KiB |
| `s_v_tile` (2×)   | 32 KiB      | 32 KiB (unchanged — shared)         | 0 |
| `s_kpe_tile` (2×) | 4 KiB       | 4 KiB (unchanged — shared)          | 0 |
| **Total**         | ~46 KiB     | **~64 KiB**                         | +18 KiB |

B200 opt-in SMEM cap per CTA = 227 KiB. Headroom is ample.

### Register state (per thread)

Today: `float m, l, o0, o1` (4 regs).
7a:   `float m[H], l[H], o0[H], o1[H]` → **64 regs** per thread of per-head
state. With 256 thr × 1 CTA/SM we have 256 regs/thread available — plenty.
But with the current 2-CTA-per-SM target we'd be tight at 128 regs. So:

- **Phase 7a decision:** drop to `__launch_bounds__(THREADS, 1)`. One CTA
  per SM. We go from 32 CTAs/GPU (148 SM × 2 today) → 148 CTAs/GPU, but
  each covers 16× the work. For long T we still fill.
- For short-T (T < 148) we fall back to split-K on the `T` axis (G ≥ 2) as
  today — grid becomes `(T, G)`.

### Kernel loop sketch

```
for tile in 0..n_valid_tiles:                            # shared across H
    cp.async K/V+KPE tile[next] to s_v_tile[next&1]      # ONE load

    ### logit phase ###
    # 256 thr × 16 rows × 16 heads = 65 536 dots
    # Assignment: 2 warps per head, 16 rows, per-warp row-chunked dot
    # OR a direct K^T·Q GEMV: N = H = 16 → a candidate for tcgen05.mma
    for h in 0..H:
        for r in 0..BK:
            logit = dot( Q_h · K_r · scale )
            s_logits_tile[h, r] = logit

    __syncthreads()

    ### softmax + V-accum phase (per-head online softmax) ###
    for h in 0..H:
        # each thread owns (2*tid, 2*tid+1) channels of head h's output
        tile_max = max(s_logits_tile[h, :])
        if tile_max > -INF:
            m_new = max(m[h], tile_max)
            alpha = exp2(m[h] - m_new)
            l[h] *= alpha;  o0[h] *= alpha;  o1[h] *= alpha
            for r in 0..BK:
                p = exp2(s_logits_tile[h, r] - m_new)
                # K[r, 2i..2i+1] read from s_v_tile[buf]
                l[h] += p
                o0[h] += p * k.x
                o1[h] += p * k.y
            m[h] = m_new
```

### Compute pattern rationale

- **Logit phase** is the natural tensor-core target later (H = 16 rows of Q,
  BK = 16 rows of K, D_CK = 512 reduction). 7a implements the scalar / warp-
  cooperative version first for correctness; 7c swaps this to `tcgen05.mma
  m16n16k16` over D_CK in chunks.
- **V-accum phase** unchanged in structure. Just outer-looped over H.
- **Memory traffic per token:** K/V = one `TOPK × (D_CK + D_KP)` sweep.
  Today: `H × TOPK × (D_CK + D_KP) × 2 B` per token (16× today). Worst-case
  HBM savings = 16× on the K/V part, modulated by the current L2 hit rate.

### Early-stopping & padding

- `n_valid_tiles` is the same (token-scope) quantity it is today. No
  per-head variation.
- Invalid rows (`g < 0`) still zero-filled via ST.shared as today.
- Q load: H × 576 bf16 = 18 KiB. 9 issue chunks per thread vs today's 0-1 —
  we add a second `cp.async` group.

### Output write

- Today: one `bf162` per thread per kernel.
- 7a: 16 `bf162` per thread per kernel (one per head). Write in a loop,
  per-head, to `out + (t*H + h) * D_CK + 2*tid`.
- `lse_out`: `thread 0 of warp h` writes lse[h] (one warp handles one head
  of book-keeping if we want to parallelize, otherwise serial loop at the
  end).

### Correctness strategy

- Keep the old kernel as the `MLA_SPARSE_ATTN_CTA_PER_T=0` path.
- Default initial rollout: env flag `MLA_SPARSE_ATTN_CTA_PER_T=1` (OFF by
  default until 7a passes all 23 workloads and has geomean at parity or
  better).
- First smoke: 5 workloads × `MLA_SPARSE_ATTN_CTA_PER_T=1` — numerics via
  same abs/rel-err thresholds the harness already runs.

### Expected perf (7a only, no tensor cores yet)

Back-of-envelope, HBM-limited portion only:

| Workload | Current (µs) | K/V-only HBM share | Projected 7a (µs) | Δ vs current |
|----------|-------------:|-------------------:|------------------:|-------------:|
| 564007ac T=8 H=16 longest | 73  | ~70 % | **~40-50**  | −30 to −45 % |
| d57eb9e1  T=6 H=16        | 36  | ~55 % | **~25-30**  | −15 to −30 % |
| ddfa9e34  T=6 H=16        | 29  | ~50 % | **~22-26**  | −10 to −25 % |
| short-K <15 µs workloads  | ≤15 | <20 % | ≈15          | ~0 (latency) |

These assume L2 hit rate stays similar (redundant H-reads already getting
some L2 absorption today). Real numbers could be lower if L2 pressure also
drops, or higher if the H-batched logit compute becomes the new bottleneck
(Phase 7b covers optimizing that).

## Phase 7b — SMEM / pipeline tuning on 7a

- Revisit V_PIPE depth (2 → 3 or 4 given the SMEM headroom on a single-CTA
  kernel).
- TMA descriptor reuse on the shared K/V tile (unified ckv descriptor; KPE
  unchanged).
- Head-parallel logit compute — each head's dot uses H warps of 2 each, or
  parallel H-fanned rows.

## Phase 7c — `tcgen05.mma` for logit + V-accum

The payoff phase.

With `N = H = 16` as the natural outer dim:
- **Logit GEMM:** A = Q [H=16, D=512] bf16, B = K^T [D=512, BK=16] bf16,
  C = S [H=16, BK=16] fp32. Matches `m16n16k16` or composed multi-tile.
- **V-accum GEMM:** A = softmax(S) [H=16, BK=16] bf16 or fp32, B = V [BK=16,
  D=512] bf16, C = O [H=16, D=512] fp32. Accumulates across K tiles.

On B200, `tcgen05.mma.async.m128n128k16` at 4 PFLOPS sustained. Our m=16 is
1/8 of native, so ~0.5 PFLOPS effective. Still ≈ 100× scalar. V-accum pops
from 0.07 µs to essentially free; logit compute gets ~100× compression.

**Caveat:** the redeeming value here is more about freeing *latency slots*
(moving logit compute off the critical path so K/V prefetch can continue)
than raw throughput — we're already below 1 µs in that phase.

## Phase 7d — cluster-shared K/V (optional, bigger hit)

Cluster size = 2 (or 4) CTAs on the same `t`. Split H-work: CTA 0 owns
heads 0..7, CTA 1 owns heads 8..15. K/V is loaded once by CTA 0 and the
second CTA reads it via distributed SMEM (`st.async.shared::cluster` or the
cluster TMA intrinsics).

- Halves per-CTA register pressure (H → 8).
- Halves per-CTA Q/O SMEM.
- K/V HBM reads go from 1× per token (7a) → 1× per (t, cluster) = identical
  to 7a — so the HBM win is already captured in 7a. Benefit here is
  *register pressure* and *occupancy*, not HBM.

Only pursue 7d if 7a register pressure is measurably limiting perf.

## Phase 7e — split-K integration

Re-home the existing `mla_split_partial_kernel` onto CTA-per-t. G stays
on the T axis but now each G-slice is a `(H, K_slice)` pair — code path
mostly reuses 7a with an extra partial-write at the end.

## Decisions to lock before coding

1. **Scope:** land 7a alone first (HBM win), or push to 7a+7c (add tensor
   cores) in the same session?
2. **Correctness baseline:** old kernel stays as the default (7a gated by
   env flag) or we cutover once 7a passes — when? After correctness, or
   after perf?
3. **Register budget:** `__launch_bounds__(THREADS, 1)` (1 CTA/SM, more
   regs) vs aggressive (256, 2) (higher occupancy, risk of spills).
   Recommend start at `(256, 1)`; promote later if it turns out H-state
   fits.

## Short-workload regression guard

Today's sub-15-µs workloads are latency-bound, not BW-bound. CTA-per-t
consolidates 16 CTAs into 1, which:
- Halves kernel launch overhead ~ invisibly (we're already below the
  measurable floor).
- Increases per-CTA work linearly — latency grows ~ H× in worst case if
  the consolidation can't be hidden behind the same cp.async pipe.
- Most of those workloads have T ≤ 3, so GPU fill becomes the issue:
  `grid = (T,)` on T=2 only dispatches 2 CTAs → **requires split-K** (G
  high) to spread. The existing split-K path already covers this.

So 7a must keep the split-K fallback for small T, dispatching G ≥ 2 by the
same `pick_split_G` logic. The dispatcher change is one-line:
`base_ctas = T  /* instead of T*H */` when CTA-per-t is on.

## First-step implementation plan for 7a

1. Add env flag `MLA_SPARSE_ATTN_CTA_PER_T` (default OFF).
2. New kernel `mla_sparse_attn_kernel_cta_per_t` (template on `H_` to let
   the compiler fully unroll). Grid = `(T,)`.
3. SMEM layout struct + sizing helper.
4. Adapt `mla_sparse_attn_host_launch` to pick grid + SMEM.
5. Adapt `pick_split_G` to use `base_ctas = T` when CTA-per-t path is
   active.
6. Smoke on 5 workloads (T range: 2, 4, 6, 8, 12). Verify all PASS.
7. Full bench (all 23 workloads) CTA-per-t OFF vs ON. Multiple runs to
   filter noise. Keep OFF by default if not at geomean parity.
8. Only flip default to ON if 7a alone wins (or keep OFF until 7c lands if
   mixed).

Time box: **1 day for code + smoke**, **½ day for A/B + write-up**. Stop
and re-plan at the end of day 1 if 7a smoke fails correctness or shows
unexpected perf cliff.
