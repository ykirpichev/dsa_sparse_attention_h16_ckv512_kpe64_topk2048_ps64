# Wombat — DSA Attention Submission

**Subject:** [MLSys26 Contest — DSA Attention] Wombat final submission, tag `submission-final`

Hi FlashInfer-Bench team,

This is the final submission from team **Wombat** for the **DSA Attention** track
(definition `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`) of the
MLSys 2026 FlashInfer AI Kernel Generation Contest.

## Submission details

| | |
|---|---|
| Team | Wombat |
| Track | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| Category | agent-assisted |
| Repository | `https://github.com/ykirpichev/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| Evaluation tag | `submission-final` |
| Solution name | `wombat_dsa_sparse_mla_sm100a_final` |
| Commit | Peel of `submission-final` — run `git rev-parse submission-final^{}` |
| Layout | `language = "python"`, `entry_point = "solution.py::run"`, `destination_passing_style = true` |
| JIT build | `torch.utils.cpp_extension.load(..., extra_cuda_cflags=["-gencode", "arch=compute_100a,code=sm_100a", "-O3", "--use_fast_math", "--expt-relaxed-constexpr", "-std=c++17"], extra_ldflags=["-lcuda"])` |
| Sources | `solution/python/solution.py` (torch JIT wrapper) + `solution/python/kernel.cu` (CUDA + PYBIND11_MODULE) |

`solution.py` compiles `kernel.cu` on first call via the Torch cpp-extension
path (same machinery the starter-kit `TorchBuilder` uses). `-gencode arch=compute_100a,code=sm_100a`
is passed explicitly so the sm_100a-only features (TMA gather, `cp.async.bulk.tensor.gather4`,
fp32 split-K partials) assemble regardless of the evaluator's default SM
detection. `-lcuda` links the driver API for `cuTensorMapEncodeTiled`.

`scripts/run_modal.py` pins the Modal build image to the same digest as the
official evaluator environment (`EVALUATION.md`):
`flashinfer/flashinfer-ci-cu132:20260401-2c675fb`.

## Kernel summary — "v5_7i" path

B200 / sm_100a-only. Default dispatch is **CTA-per-t with runtime
sparse-index density pre-scan** (Phase 7h) on top of **fp32 split-K partials**
(Phase 7i). Enabled paths:

- **CTA-per-t kernel** (`mla_sparse_attn_kernel_cta_per_t_mma_v`) — one CTA per
  token batches all H=16 heads, uses WMMA (`nvcuda::wmma` bf16×bf16→fp32) for
  both logit and V-accum GEMMs.
- **Runtime density pre-scan** gates CTA-per-t vs legacy grid based on average
  valid-K per token (threshold = 210). Routes T=8 low-density workloads
  (e.g. `385742b2`, `4c46a94b`) back to the legacy grid, which finishes in
  ~42 µs instead of the 54 µs CPT floor.
- **fp32 split-K partials** — `O_partial` in HBM is fp32 across all split-K
  reducer kernels, eliminating the `0.03` intermittent `max_abs_error` spikes
  we saw on two workloads under the previous bf16 partial storage.
- **Fused partial + reduce** (Phase 6f) — saves the second launch's ~3 µs
  dispatch overhead on G∈{2,4,8,16} split-K paths.
- **TMA gather4 descriptors** cached per (tensor-ptr, row-count) for the K/V
  sparse loads.
- Online softmax with `max/sum` rescale, per-head.

The legacy per-(T,H) grid kernel is preserved for the small-KV / low-density
fallback path.

## End-to-end verification (Modal B200, evaluator CI image digest)

Results for `wombat_dsa_sparse_mla_sm100a_final` (this submission) measured
via `modal run scripts/run_modal.py` on `B200:1`, full benchmark (warmup=3,
iterations=100, num_trials=5):

- **23 / 23 workloads PASSED**
- Wall-time range **8 – 56 µs** (mean ≈ **33 µs**)
- Speedups **34.2× – 136.0× vs PyTorch reference** (geomean ≈ **59×**, arith
  mean ≈ **69×**)
- `max_abs_error ≤ 0.0156` across all 23 workloads (well under the definition's
  `atol=0.05` tolerance)

Aggregate comparison against the strong baselines (median of 10 repeats, 230
samples per solution; see `artifacts/final/FINAL_REPORT.md` for the full table):

| Solution | Mean wall-time | Geomean speedup vs PT | Max speedup | Min speedup | PASS rate |
|---|---:|---:|---:|---:|---:|
| PyTorch reference | 1818 µs | 1.00× | — | — | 230/230 |
| FlashInfer | 338 µs | 5.51× | — | — | 230/230 |
| CUDA v4 (Stage 3c legacy) | 108 µs | 17.81× | 127.62× | 12.10× | 230/230 |
| CUDA v5_fused (best legacy) | 70 µs | 31.09× | 127.62× | 12.10× | 230/230 |
| **This submission (v5_7i)** | **34 µs** | **54.6×** | **148.16×** | **28.50×** | **230/230** |

Key wins over the legacy kernel (v5_fused):

- **1.76× geomean** improvement (54.6 / 31.09) on 230/230 samples.
- Worst-case latency **59.6 µs** (vs 127 µs for v5_fused on long-T workloads).
- Deterministic correctness: 0 numerical failures across all repeats.

Techniques intentionally NOT adopted (with rationale in `artifacts/`):

- **`tcgen05.mma` (Blackwell UMMA)** — investigated in Phase 6d
  (`artifacts/p6_splitk/ab_report_6d_tcgen05.md`). Shape mismatch: tcgen05's
  natural `m64n128k16` expects N ≥ 128; our V-accum is a GEMV with N=1 (1/128
  effective utilization). Kernel is HBM/issue-slot-bound, not compute-bound;
  V-accum compute is ≤0.1% of kernel time.
- **Full warp-specialisation (WS3)** — correct but regressed on benchmarks
  (issue-slot starvation in the consumer warps). Implemented and gated OFF
  behind `MLA_SPARSE_ATTN_WS3=1`.

## Submission integrity

- `config.toml`: `name = "wombat_dsa_sparse_mla_sm100a_final"`,
  `language = "python"`, `entry_point = "solution.py::run"`,
  `destination_passing_style = true`.
- `solution.json`: packed via `python scripts/pack_solution.py`; contains
  `kernel.cu` + `solution.py` as the only source files.
- Kernel source has an `#error` guard that refuses to compile for anything
  other than `__CUDA_ARCH__ == 1000` (sm_100a).
- `solution.py` is self-contained — only imports are `pathlib`, `torch`, and
  `torch.utils.cpp_extension`; no out-of-tree dependencies.

Please let me know if anything else is needed.

Thanks,
Yury (`ykirpichev@gmail.com`) — team Wombat
