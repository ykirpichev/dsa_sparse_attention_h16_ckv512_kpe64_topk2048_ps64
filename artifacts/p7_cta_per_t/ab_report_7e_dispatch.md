# Phase 7e — Dispatcher unification (GATE NOT MET — default stays OFF)

## TL;DR

Rollout gate (`env_flag_off_until_geomean`) requires **geomean(legacy/CPT)
≥ 1.0** for the dispatcher default to flip ON. Measured across 23
workloads with two sweep points:

| config                       | geomean(legacy/CPT) | CPT-activated | decision |
|------------------------------|---------------------|---------------|----------|
| CPT-mma, no threshold (≡T≥0) | (prior session) ≈0.72× | 23/23        | OFF       |
| CPT-mma, `min_t=20`          | **0.976×**          | 10/23         | OFF       |
| CPT-mma, `min_t=50`          | **0.998×**          | 3/23          | OFF       |

Gate not met. **Default `MLA_SPARSE_ATTN_CTA_PER_T` remains OFF.**

## Per-workload detail (ms)

| id         | legacy | tth20  | r20   | tth50  | r50   | CPT active |
|-----------:|-------:|-------:|------:|-------:|------:|-----------:|
| 02d6ae9c   | 0.056  | 0.058  | 0.966 | 0.056  | 1.000 | min_t=20 only |
| 05f6de65   | 0.027  | 0.027  | 1.000 | 0.027  | 1.000 | none       |
| 0a63b87b   | 0.010  | 0.011  | 0.909 | 0.010  | 1.000 | min_t=20 only |
| 0c23b10c   | 0.008  | 0.008  | 1.000 | 0.008  | 1.000 | none       |
| 2207f0fd   | 0.055  | 0.054  | **1.019** | 0.055 | 1.000 | min_t=20 only |
| 232ed014   | 0.058  | 0.058  | 1.000 | 0.058  | 1.000 | none       |
| 38389961   | 0.052  | 0.052  | 1.000 | 0.052  | 1.000 | none       |
| 385742b2   | 0.042  | 0.042  | 1.000 | 0.042  | 1.000 | none       |
| 4c46a94b   | 0.040  | 0.040  | 1.000 | 0.040  | 1.000 | none       |
| 5096e459   | 0.070  | 0.071  | 0.986 | 0.070  | 1.000 | min_t=20 only |
| 564007ac   | 0.073  | 0.074  | 0.986 | 0.074  | 0.986 | both       |
| 68d6817d   | 0.013  | 0.013  | 1.000 | 0.013  | 1.000 | none       |
| 78b2e11c   | 0.058  | 0.064  | 0.906 | 0.059  | 0.983 | both       |
| 7a389715   | 0.056  | 0.056  | 1.000 | 0.056  | 1.000 | none       |
| 9d4a5f21   | 0.008  | 0.008  | 1.000 | 0.008  | 1.000 | none       |
| 9f3f891b   | 0.023  | 0.023  | 1.000 | 0.023  | 1.000 | none       |
| ae4219a9   | 0.054  | 0.054  | 1.000 | 0.055  | 0.982 | none       |
| b7668cfd   | 0.010  | 0.011  | 0.909 | 0.010  | 1.000 | min_t=20 only |
| d57eb9e1   | 0.036  | 0.036  | 1.000 | 0.036  | 1.000 | none       |
| ddfa9e34   | 0.029  | 0.030  | 0.967 | 0.029  | 1.000 | min_t=20 only |
| e6b849f2   | 0.012  | 0.013  | 0.923 | 0.012  | 1.000 | min_t=20 only |
| f77df5ce   | 0.008  | 0.009  | 0.889 | 0.008  | 1.000 | min_t=20 only |
| fc85411e   | 0.008  | 0.008  | 1.000 | 0.008  | 1.000 | none       |

**Finding:** CPT-mma *never* beats legacy on any workload in this bench,
at any threshold. The single "win" (2207f0fd +1.9%) is within run-to-run
noise (see 5096e459 at −1.4% and 564007ac at −1.4% on the same tier).

## Why CPT loses even where it "should" win

The CPT-mma path's 46 µs per-CTA floor (measured in 7c-next) is
**higher than the legacy split-K partial+reduce wall-time** for every
workload in this suite:

* Legacy's split-K launches 2 CTAs/SM (headroom from smaller per-CTA
  SMEM) → hides its slightly higher compute under higher issue-slot
  occupancy.
* CPT-mma keeps 16 heads in a single 512-thread CTA → 1 CTA/SM →
  issue-slot starvation dominates the 46 µs floor (see `ab_report_7d_cluster.md`).

## What would unblock the dispatcher flip

Unblocking requires either of:

1. **Phase 7d-proper (H-split 2-CTA cluster)** — halves per-CTA SMEM
   (Q, logit scratch) while sharing K/V via DSM. Projected per-CTA
   wall: 46 → ~24 µs. Would flip geomean above 1.0× decisively. See
   `ab_report_7d_cluster.md` for the full plan.
2. **Bigger workloads (T ≫ 100)** — the 46 µs floor is amortized across
   more tokens in a persistent-CTA launch. Not present in this bench.

## Action taken

* `MLA_SPARSE_ATTN_CTA_PER_T` remains OFF by default in
  `solution/python/kernel.cu::cta_per_t_enabled()`.
* `MLA_SPARSE_ATTN_CTA_PER_T_MIN_T` and `MLA_SPARSE_ATTN_CTA_PER_T_MMA`
  plumbing ships as-is; `min_t=50` is the lowest-regression
  opt-in configuration (~0.2% geomean slowdown).
* Logs:
  - `/tmp/full_legacy.log` — legacy baseline (23 workloads)
  - `/tmp/full_tthresh20.log` — CPT-mma, min_t=20
  - `/tmp/full_tthresh50.log` — CPT-mma, min_t=50

## Recommendation for future sessions

1. Implement the H-split 2-CTA cluster from 7d-proper.
2. Re-run this exact A/B against legacy.
3. If geomean ≥ 1.0×, flip `cta_per_t_enabled()` default to true and
   set `cta_per_t_min_t()` default to the smallest T where CPT ≥ legacy.
