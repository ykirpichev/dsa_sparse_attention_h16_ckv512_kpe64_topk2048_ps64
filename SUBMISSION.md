# Wombat — DSA Attention Submission

**Subject:** [MLSys26 Contest — DSA Attention] Wombat submission, tag `submission-v1`

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
| Evaluation tag | `submission-v1` |
| Commit | Peel of `submission-v1` — run `git rev-parse submission-v1^{}` |
| Layout | `language = "cuda"`, `entry_point = "kernel.cu::kernel"`, `destination_passing_style = true`, default `tvm-ffi` binding |
| Sources | `solution/cuda/kernel.cu` (TVM-FFI C++ entry, `TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel_fn)`) + `solution/cuda/binding.py` (stock placeholder) |

`scripts/pack_solution.py` is bit-identical to the upstream `flashinfer-bench-starter-kit`.
`scripts/run_modal.py` pins the Modal build image to the same digest as the official
evaluator environment (`EVALUATION.md`): `flashinfer/flashinfer-ci-cu132:20260401-2c675fb`.

## Kernel summary

Single-block-per-(token, head) fused CUDA kernel, B200 / sm_100a only.
Stage-3c optimization layer: `cp.async` double-buffered tile prefetch for both
the logit pass and the weighted cKV projection, packed bf16×bf16 dot products
with fp32 accumulation, warp-cooperative reductions, and SMEM aliasing of the
value-tile buffer as the logit-tile buffer to keep per-CTA SMEM inside the
100 KiB per-CTA budget.

## End-to-end verification (Modal B200, evaluator CI image digest)

Exact evaluation command reproduced (isolated runner, same flags as `EVALUATION.md`):

```
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300
```

Results for `wombat_dsa_sparse_mla_cuda_v4_sm100a` (Stage 1 kernel rebuilt
against `sm_100a`; per-(T,H) CTA structure preserved):

- **23 / 23 workloads PASSED**
- Latencies 0.012 – 0.160 ms
- Speedups 14.28× – 152.40× vs the reference (geomean ≈ 35.6×, median 30.1×)
- TVM-FFI compile + isolated-runner subprocess mode both succeed.
- CLI exit code 0.

Head-fusion experiment (`v5a_fused_sm100a`, one CTA per token covering
H=16 heads) was measured on the same 23 workloads and REGRESSED to
geomean 5.25×: on large-T workloads the reduction in CTA count (T vs
T×H) starves B200 SMs, and the HBM-reuse win doesn't outweigh it
because the per-head kernel was already L2-resident. The per-(T,H)
structure is kept as the production configuration.

The contest baseline (`flashinfer_wrapper_5af199`) also passes 23/23 in the same
container, confirming the environment is sane.

Please let me know if anything else is needed.

Thanks,
Yury (`ykirpichev@gmail.com`) — team Wombat
