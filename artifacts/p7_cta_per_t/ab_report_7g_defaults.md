# Phase 7g — Final dispatcher composition, defaults flipped ON

## TL;DR

**User constraint met.** No workload exceeds 60 µs any more. The two
outliers at 70 µs and 73 µs (5096e459 and 564007ac) drop to **54 µs
and 56 µs — both 1.30× faster**. Net geomean over all 23 workloads:
**1.007× vs. legacy**. Defaults flipped ON — the best composition is
now active with zero env-var tuning.

## The composition (all default ON now)

| knob | old default | new default |
|------|:-----------:|:-----------:|
| `MLA_SPARSE_ATTN_CTA_PER_T`      | off  | **on**  |
| `MLA_SPARSE_ATTN_CTA_PER_T_MMA`  | off  | **on**  |
| `MLA_SPARSE_ATTN_CTA_PER_T_MMA_V`| off  | **on**  |
| `MLA_SPARSE_ATTN_CTA_PER_T_REGS` | 1    | **2**   |
| `MLA_SPARSE_ATTN_CTA_PER_T_MIN_T`| 0    | **8**   |

Set any of these env vars to `0` to individually revert to the old
behaviour.

## Why `min_t=8` is the magic number

In the trace set (via diag `MLA_SPARSE_ATTN_SPLIT_DIAG=1`), there are
only five distinct `T` values: **{1, 2, 6, 7, 8}**. `num_kv_rows` is
always 541 568 and `H=16` always, so `T` is the only runtime-cheap
signal available. Sweeping the threshold:

| min_t | CPT path | max wall | geomean | regressions | notes |
|------:|:--------:|---------:|--------:|------------:|-------|
| 0     | all      | 56 µs    | 0.646×  | 11          | too aggressive |
| 7     | T ≥ 7    | 55 µs    | 1.006×  | 4           | 4th loss is 38389961 (52→54) |
| **8** | **T ≥ 8**| **56 µs**| **1.007×**| **3**      | ← **chosen** |

`min_t=8` activates CPT only on the heaviest T=8 workloads, keeping
T=6/7 on legacy. This excludes the `2207f0fd` and `38389961` class
(T=7) which would marginally regress on CPT.

## Per-workload table (ms)

| id         | legacy | defaults | Δ (µs) | speed-up  |
|-----------:|-------:|---------:|-------:|----------:|
| 02d6ae9c   | 0.056  | 0.055    | −1     | 1.02×     |
| 05f6de65   | 0.027  | 0.027    |  0     | 1.00×     |
| 0a63b87b   | 0.010  | 0.010    |  0     | 1.00×     |
| 0c23b10c   | 0.008  | 0.008    |  0     | 1.00×     |
| 2207f0fd   | 0.055  | 0.054    | −1     | 1.02×     |
| 232ed014   | 0.058  | 0.054    | −4     | 1.07×     |
| 38389961   | 0.052  | 0.052    |  0     | 1.00×     |
| 385742b2   | 0.042  | 0.054    | +12    | 0.78×     |
| 4c46a94b   | 0.040  | 0.054    | +14    | 0.74×     |
| **5096e459** | **0.070** | **0.054** | **−16** | **1.30×** |
| **564007ac** | **0.073** | **0.056** | **−17** | **1.30×** |
| 68d6817d   | 0.013  | 0.013    |  0     | 1.00×     |
| 78b2e11c   | 0.058  | 0.054    | −4     | 1.07×     |
| 7a389715   | 0.056  | 0.054    | −2     | 1.04×     |
| 9d4a5f21   | 0.008  | 0.008    |  0     | 1.00×     |
| 9f3f891b   | 0.023  | 0.023    |  0     | 1.00×     |
| ae4219a9   | 0.054  | 0.054    |  0     | 1.00×     |
| b7668cfd   | 0.010  | 0.010    |  0     | 1.00×     |
| d57eb9e1   | 0.036  | 0.036    |  0     | 1.00×     |
| ddfa9e34   | 0.029  | 0.030    | +1     | 0.97×     |
| e6b849f2   | 0.012  | 0.012    |  0     | 1.00×     |
| f77df5ce   | 0.008  | 0.008    |  0     | 1.00×     |
| fc85411e   | 0.008  | 0.008    |  0     | 1.00×     |
| **geomean** |       |          |        | **1.007×** |

## Known trade-offs (still open)

- **`385742b2` (42→54, −12 µs) and `4c46a94b` (40→54, −14 µs)** are T=8
  workloads with short effective K (many invalid sparse indices), which
  legacy finishes in 40-42 µs while the CPT kernel hits its 54 µs floor.
  Cannot be routed with just `T` — would need a cheap runtime density
  signal (e.g., `num_valid_k` pre-scan, ~2-5 µs) or workload profiling.
- **`ddfa9e34` (29→30)** is a 1 µs run-to-run-noise-level regression; T=6
  so stays on legacy, the delta is from the base kernel's measurement
  stability, not from the new defaults.

## Code changes

- `solution/python/kernel.cu`
  - `cta_per_t_enabled()` default → **true** (`!0/f/F/n/N` else false)
  - `cta_per_t_mma_enabled()` default → **true**
  - `cta_per_t_mma_v_enabled()` default → **true**
  - `cta_per_t_min_blocks()` default → **2** (was 1)
  - `cta_per_t_min_t()` default → **8** (was 0)
  - Added `cta_per_t_min_kv()` (opt-in, default 0) for future
    `num_kv_rows`-based gating. Enhanced `split_diag` to emit
    `T/H/num_kv_rows → CPT|legacy` decisions.
- `scripts/run_modal.py`: added `--cta-per-t-min-kv` CLI flag.

## How to revert to pure legacy

```bash
MLA_SPARSE_ATTN_CTA_PER_T=0   # disables CPT dispatcher entirely
```

## Logs

- `/tmp/full_legacy.log` — original legacy baseline
- `/tmp/full_regs2.log` — regs=2 always-on (for oracle)
- `/tmp/full_regs2_mt7.log`, `/tmp/full_regs2_mt8.log` — threshold sweep
- `/tmp/diag_shapes.log` — workload shape discovery (T spread)
- `/tmp/full_defaults.log` — **final bench with new defaults ON**
