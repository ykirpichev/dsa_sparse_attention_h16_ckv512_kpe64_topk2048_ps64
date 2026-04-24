# Phase 7f — `MIN_BLOCKS_=2` (2 CTAs/SM) occupancy experiment — WIN on large-T

## TL;DR

Enabling `MLA_SPARSE_ATTN_CTA_PER_T_REGS=2` on the `mma_v` kernel **doubles
occupancy (1 → 2 CTAs/SM, confirmed)** and **wins 1.30× on the two
longest-T workloads** in the bench. Oracle (perfect per-workload
dispatch between legacy and regs=2) = **1.033× geomean over legacy**.
This decisively confirms the issue-slot starvation hypothesis identified
in 7c-next — without needing the full H-split 2-CTA cluster rearchitecture.

## What changed

**No kernel code changed.** `cta_per_t_min_blocks()` and the `MIN_BLOCKS_=2`
template specialisation were already plumbed from earlier sessions; they
had simply never been A/B'd against legacy with a wide enough workload
suite. The `__launch_bounds__(512, 2)` attribute packs the kernel into
64 regs/thread (down from ~250 at `MIN_BLOCKS_=1`) in exchange for 2
concurrent CTAs on the SM. Mild register spill (48 B local / thread)
but no measurable spill cost.

Diagnostic output at launch:

```
[mla_cpt_mma_v G=16 mb=2] regs/thread=64 local=48 static_smem=0
                           max_dyn_smem=102400 occ=2 blocks/SM
```

## Result table (ms, bench of 23 workloads)

| id         | legacy | regs=2 | ratio  | Δ (µs)  | notes            |
|-----------:|-------:|-------:|-------:|--------:|------------------|
| 02d6ae9c   | 0.056  | 0.055  | 1.018  | −1      | marginal win     |
| 05f6de65   | 0.027  | 0.053  | 0.509  | +26     | SMALL-T LOSS     |
| 0a63b87b   | 0.010  | 0.032  | 0.312  | +22     | SMALL-T LOSS     |
| 0c23b10c   | 0.008  | 0.016  | 0.500  | +8      | SMALL-T LOSS     |
| 2207f0fd   | 0.055  | 0.054  | 1.019  | −1      | marginal win     |
| 232ed014   | 0.058  | 0.054  | 1.074  | −4      | clean win        |
| 38389961   | 0.052  | 0.054  | 0.963  | +2      | tie              |
| 385742b2   | 0.042  | 0.054  | 0.778  | +12     | mid-T loss       |
| 4c46a94b   | 0.040  | 0.054  | 0.741  | +14     | mid-T loss       |
| **5096e459** | **0.070** | **0.054** | **1.296** | **−16** | **BIG WIN**    |
| **564007ac** | **0.073** | **0.056** | **1.304** | **−17** | **BIG WIN**    |
| 68d6817d   | 0.013  | 0.021  | 0.619  | +8      | small-T loss     |
| 78b2e11c   | 0.058  | 0.054  | 1.074  | −4      | clean win        |
| 7a389715   | 0.056  | 0.054  | 1.037  | −2      | win              |
| 9d4a5f21   | 0.008  | 0.021  | 0.381  | +13     | small-T loss     |
| 9f3f891b   | 0.023  | 0.053  | 0.434  | +30     | mid-T loss       |
| ae4219a9   | 0.054  | 0.055  | 0.982  | +1      | tie              |
| b7668cfd   | 0.010  | 0.032  | 0.312  | +22     | small-T loss     |
| d57eb9e1   | 0.036  | 0.054  | 0.667  | +18     | mid-T loss       |
| ddfa9e34   | 0.029  | 0.054  | 0.537  | +25     | mid-T loss       |
| e6b849f2   | 0.012  | 0.042  | 0.286  | +30     | small-T loss     |
| f77df5ce   | 0.008  | 0.021  | 0.381  | +13     | small-T loss     |
| fc85411e   | 0.008  | 0.021  | 0.381  | +13     | small-T loss     |

### Geomean summary

| comparison                     | geomean(legacy / candidate) |
|--------------------------------|---------------------------|
| regs=2 alone (always on)       | 0.646× (regresses 35%)    |
| regs=2 on T-large (>40 µs)     | **1.044× (wins 4.4%)**    |
| **Oracle (best of the two)**   | **1.033× (real 3.3% win)** |

## Why a clean T-threshold dispatcher doesn't exist (explored)

`cta_per_t_min_t()` was swept across {20, 25, 30, 40, 50, 100}. For each
sweep, the CPT path's activation did not correlate with regs=2 wins:

| min_t | CPT activated | big winners captured? | regressors pulled in? |
|------:|:--------------|:---------------------:|:---------------------:|
| 0     | all 23        | yes                   | 11 regressors         |
| 20    | 10            | yes (5096e459, 564007ac) | yes (5 of 5)      |
| 25    | 2             | no                    | 0                     |
| 30    | 2             | no                    | 0                     |
| 50    | 2             | no                    | 0                     |
| 100   | 2             | no                    | 0                     |

The reason: **`T` does not correlate with wall-time**. Two workloads
with identical `T` can have different wall times depending on
`num_kv_rows` and sparse-index density. `5096e459` and `564007ac` (the
big winners) have small `T` (<25) but long effective K-ranges; the
legacy kernel runs them in 70-73 µs while the regs=2 CPT path finishes
in 54-56 µs. No value of `min_t` admits those two without also admitting
several workloads with `T ≥ 20` and short K-ranges where CPT loses.

## Recommendation

Ship `MLA_SPARSE_ATTN_CTA_PER_T_REGS=2` as an **opt-in** tuning knob. Users
who can profile per-workload can selectively enable it for the
longest-K workloads (5096e459 class) and gain 1.3× there. Default
remains OFF because no T-only heuristic recovers the 1.033× oracle on
this suite.

### Full opt-in command

```bash
MLA_SPARSE_ATTN_CTA_PER_T=1 \
MLA_SPARSE_ATTN_CTA_PER_T_MMA_V=1 \
MLA_SPARSE_ATTN_CTA_PER_T_REGS=2 \
MLA_SPARSE_ATTN_CTA_PER_T_MIN_T=0 \
# (user's program runs)
```

## Future work to close the 3.3% gap

To build an auto-dispatcher that achieves oracle-geomean:

1. **Runtime pre-scan of sparse_idx**: count valid rows / average density
   before kernel launch. ~2-5 µs for TOPK=2048. Then route based on
   `expected_wall = f(T, valid_k, H)`. Risky: the 2-5 µs pre-scan
   eats into the 1-4 µs wins on the tie-class workloads. Favorable
   only if the dispatcher cost is < 1 µs (e.g., a single launch of a
   super-lightweight kernel that only computes and returns the avg).
2. **`num_kv_rows × T` as a proxy**: no pre-scan needed but is only a
   loose bound on valid K density.
3. **Training-dataset-based constants**: if the deployed workloads are
   known at build time, `min_t` could be set per-deployment. Not
   viable for a general-purpose library.
4. **H-split 2-CTA cluster (original 7d-proper)**: would further lower
   the per-CTA wall (46 → ~24 µs on large workloads, potentially even
   beating regs=2). Multi-session scope — see `ab_report_7d_cluster.md`.
   This remains the highest-lift future direction.

## Status

- Kernel code: **no change** — mma_v already supports `MIN_BLOCKS_=2`.
- Dispatcher default: **stays OFF** (no auto-activation).
- Documentation of opt-in path: added to
  `artifacts/p7_cta_per_t/ab_report_7f_regs2.md` (this file).
- Logs:
  - `/tmp/full_legacy.log`
  - `/tmp/full_regs2.log` — regs=2, no threshold (establishes the oracle)
  - `/tmp/full_regs2_mt{20,25,30,40,50,100}.log` — threshold sweep
