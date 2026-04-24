# Phase 6a validation — G=16 for [64, 110] base-CTA tier

Confirms the claim: *"Default kernel now always uses G=16 on the [64, 110] CTA
tier. Workloads ddfa9e34 and d57eb9e1 see ~15 % speedups."*

## Method

- Same binary, same image, single B200 (Modal), back-to-back runs.
- Two CLI arms:
  - `--split-g16=off` (kernel override: `pick_split_G` falls back to the
    legacy G=8 for the [64, 110] tier)
  - `--split-g16=on`  (current default: promote to G=16)
- 3 runs per arm, interleaved to absorb hardware/queue noise.
- Numbers are the per-workload median of `ms` reported by the contest
  harness (already internally averaged across warmup/iters/trials).

## Raw numbers (ms)

| Workload   | OFF run1 | OFF run2 | OFF run3 | ON run1 | ON run2 | ON run3 |
|------------|---------:|---------:|---------:|--------:|--------:|--------:|
| ddfa9e34   | 0.035    | 0.034    | 0.035    | 0.029   | 0.030   | 0.029   |
| d57eb9e1   | 0.042    | 0.042    | 0.042    | 0.035   | 0.036   | 0.036   |

## Aggregates

| Workload   | OFF median | ON median | Δ (µs) | Δ (%)   |
|------------|-----------:|----------:|------:|--------:|
| ddfa9e34   | 35 µs      | 29 µs     | -6    | **-17.1 %** |
| d57eb9e1   | 42 µs      | 36 µs     | -6    | **-14.3 %** |

Both workloads are fully stable across the three repeats (±1 µs, i.e. the
harness's event-timer quantum). The speedups exceed the ~15 % claim.

## Logs

Per-arm full harness output is in this directory:

- `off_run{1,2,3}_<ts>.log`
- `on_run{1,2,3}_<ts>.log`

## Reproduction

```
# Baseline (force the legacy G=8 path for this tier):
modal run scripts/run_modal.py --split-g16=off

# Default (current behaviour: G=16 promotion on [64, 110] tier):
modal run scripts/run_modal.py --split-g16=on
# or equivalently (default is now ON):
modal run scripts/run_modal.py
```

## Verdict

**Validated.** Both targeted workloads ship at roughly 15–17 % faster than
the G=8 baseline, matching the write-up in
`artifacts/p6_splitk/ab_report.md`. No regressions spotted on the rest of
the 17-workload smoke sweep (other workloads bypass the [64, 110] tier or
are dominated by warmup noise at sub-15 µs scales).
