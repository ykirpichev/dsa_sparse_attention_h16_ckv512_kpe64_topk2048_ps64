# Performance: custom CUDA vs definition reference vs FlashInfer baseline

This note is for **optimization planning**. It separates three different comparisons the contest and tooling use.

> Concrete kernel-optimization backlog (ranked ideas, ownership-ready): [`OPTIMIZATION_IDEAS.md`](OPTIMIZATION_IDEAS.md).

## Terminology (do not mix these up)

| Name | What it is | Used for |
|------|------------|----------|
| **Definition reference** | Simple Python reference shipped with the definition (similar spirit to batched PyTorch GEMMs: gather indices → `QK^T` + masked softmax + projection). | **Official speedup** in FlashInfer-Bench (`reference_latency_ms` vs your solution). See [FAQ.md](../FAQ.md) (“speedup … not the optimized FlashInfer baseline”). |
| **FlashInfer dataset baseline** | Packaged solution `flashinfer_wrapper_5af199` for `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`. | Strong **library** comparison on the same workloads; **not** the denominator for contest speedup. Listed in [EVALUATION.md](../EVALUATION.md#flashinfer-baselines). |
| **This repo’s CUDA kernel** | `solution/cuda/mla_cuda.cu` + `binding.py::run` (`baseline_dsa_sparse_mla_cuda_v1`). | Submitted track implementation. |

Local PyTorch-style code that mirrors the reference math lives in [`solution/python/kernel.py`](../solution/python/kernel.py) (optional baseline / sanity check).

---

## Measured run: CUDA vs definition reference (Modal B200)

**Artifact:** [`artifacts/modal_baseline_full_20260419T214908Z.json`](../artifacts/modal_baseline_full_20260419T214908Z.json)

| Field | Value |
|-------|--------|
| Saved (UTC) | 2026-04-19T21:49:08 |
| Runner | Modal (`scripts/run_modal.py`) |
| Image / stack | `flashinfer/flashinfer-ci-cu132`-based (see Modal app in repo) |
| Solution | `baseline_dsa_sparse_mla_cuda_v1` |
| Workloads | 23 / 23 passed |
| Benchmark config | warmup 3, iterations 50, trials 5, isolated runner, `profile_baseline: true` |

### Aggregate latency and speedup (vs definition reference)

| Metric | Value |
|--------|--------|
| Mean solution latency | **0.503 ms** |
| Mean definition reference latency | **1.479 ms** |
| **Mean speedup** (reference / solution) | **5.13×** |
| Median speedup | 3.48× |
| Min speedup | 1.74× |
| Max speedup | 13.36× |
| Population stdev (speedup) | ~3.61 |

Speedup varies a lot by trace shape (token/head counts and memory footprint); treat the mean as the headline number aligned with contest scoring (mean of per-workload speedups).

### Per-workload breakdown

Workload IDs are trace workload keys from the dataset.

| Workload ID | Solution ms | Reference ms | Speedup |
|-------------|-------------|--------------|---------|
| `0c23b10c7b7645719517828c12eaa1d2` | 0.098 | 0.944 | 9.60× |
| `9d4a5f21268e484ea05a2f2af91d9fa7` | 0.109 | 1.060 | 9.74× |
| `b7668cfd194c4b95ab600feb205ebac6` | 0.117 | 1.074 | 9.20× |
| `0a63b87bb2e54e9db1ca3c4c53a1d521` | 0.118 | 1.056 | 8.94× |
| `05f6de657db543ae9e4c46796522843a` | 0.283 | 1.062 | 3.75× |
| `fc85411e250c41879b6d6a1edb80f0a7` | 0.108 | 1.058 | 9.75× |
| `e6b849f2900446148c01d81152efae23` | 0.121 | 1.056 | 8.71× |
| `9f3f891bfbe24776adfcd5a579d093dd` | 0.261 | 1.066 | 4.09× |
| `f77df5ce21634f7ca0586af7e3c13434` | 0.109 | 1.054 | 9.71× |
| `385742b2717e4f02b918c7349dde23d8` | 0.556 | 1.932 | 3.48× |
| `4c46a94ba2364dc7ab476286dee8dce3` | 0.525 | 1.897 | 3.61× |
| `3838996164a94d728710f913477feba8` | 0.567 | 1.776 | 3.13× |
| `02d6ae9c64ab42ff93f05c23c53bcb7d` | 0.915 | 1.799 | 1.97× |
| `ddfa9e340b264f76abe7418692faa876` | 0.767 | 1.582 | 2.06× |
| `78b2e11c30004cceb84355722a7c6b0a` | 0.913 | 1.807 | 1.98× |
| `68d6817dcfd1433aa8d2ddeefd54b6ea` | 0.117 | 1.559 | **13.36×** |
| `564007ac354e4662a62cc4d6352dc494` | 0.923 | 1.824 | 1.98× |
| `ae4219a95f044f45bd10e17ab63c6e8f` | 0.916 | 1.683 | 1.84× |
| `232ed014bafc4835b9881bb308c659b0` | 0.569 | 1.813 | 3.19× |
| `7a389715dcc3479e9b0512309f9d1d56` | 0.771 | 1.810 | 2.35× |
| `5096e459ce3f4cdf82773ce1a0c73c8a` | 0.899 | 1.818 | 2.02× |
| `d57eb9e19f0642e8af9bd76ad0823303` | 0.904 | 1.574 | **1.74×** |
| `2207f0fdc96347c59d106e4976cfad57` | 0.905 | 1.706 | 1.88× |

*(Table values rounded for readability; full precision is in the JSON artifact.)*

---

## FlashInfer library baseline (`flashinfer_wrapper_5af199`) — not in the table above

The contest ships an **optimized FlashInfer-backed** solution for the same definition. **This repo’s Modal artifact does not include** that baseline’s latencies side-by-side.

To measure **your CUDA vs `flashinfer_wrapper_5af199`** on the same machine and dataset (local B200 or contest image):

```bash
flashinfer-bench run \
  --local /path/to/mlsys26-contest \
  --definitions dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
  --solutions baseline_dsa_sparse_mla_cuda_v1 flashinfer_wrapper_5af199 \
  --use-isolated-runner --save-results
```

(Use your packed solution name if it differs; baseline name is fixed in [EVALUATION.md](../EVALUATION.md).)

**Why run this:** Official scoring uses the **definition reference** denominator. The FlashInfer baseline answers “how far am I from a strong vendor-style implementation on the same traces?”

---

## `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` (TRT-LLM MLA API) — measured

Full-suite Modal B200 run of the wrapper in
[`solution/flashinfer_trtllm/kernel.py`](../solution/flashinfer_trtllm/kernel.py) (packed via
[`config.flashinfer_trtllm.toml`](../config.flashinfer_trtllm.toml)).

**Artifact:** [`artifacts/modal_flashinfer_trtllm_full.json`](../artifacts/modal_flashinfer_trtllm_full.json) — 23/23 PASSED (default bench atol/rtol = 0.01).

### Head-to-head: FlashInfer TRT-LLM MLA vs our CUDA

| Metric | FlashInfer TRT-LLM MLA | Our CUDA (`baseline_dsa_sparse_mla_cuda_v1`) |
|--------|------------------------|-----------------------------------------------|
| Mean solution latency | **0.259 ms** | 0.503 ms |
| Mean speedup vs definition reference | **6.05×** | 5.13× |
| Ours / FI latency — mean | — | **1.92×** slower |
| Ours / FI latency — median | — | **2.01×** slower |
| Ours / FI latency — min / max | — | 0.38× / 3.51× |

### Per-workload (solution latency in µs, speedup vs definition reference)

| Workload | FI µs | Ours µs | FI speedup | Ours speedup | Ours/FI |
|----------|------:|-------:|-----------:|------------:|-------:|
| `0c23b10c` | 255.6 | 98.3 | 3.73× | 9.60× | 0.38× |
| `9d4a5f21` | 249.8 | 108.8 | 4.50× | 9.74× | 0.44× |
| `b7668cfd` | 250.0 | 116.8 | 4.42× | 9.20× | 0.47× |
| `0a63b87b` | 250.2 | 118.1 | 4.39× | 8.94× | 0.47× |
| `05f6de65` | 259.1 | 283.4 | 4.29× | 3.75× | 1.09× |
| `fc85411e` | 252.0 | 108.4 | 4.38× | 9.75× | 0.43× |
| `e6b849f2` | 250.0 | 121.3 | 4.42× | 8.71× | 0.49× |
| `9f3f891b` | 258.9 | 260.5 | 4.34× | 4.09× | 1.01× |
| `f77df5ce` | 249.8 | 108.6 | 4.39× | 9.71× | 0.43× |
| `385742b2` | 283.3 | 555.7 | 7.11× | 3.48× | 1.96× |
| `4c46a94b` | 261.0 | 525.2 | 7.55× | 3.61× | 2.01× |
| `38389961` | 262.0 | 567.1 | 6.96× | 3.13× | 2.16× |
| `02d6ae9c` | 263.3 | 915.2 | 7.43× | 1.97× | **3.48×** |
| `ddfa9e34` | 260.7 | 766.6 | 6.43× | 2.06× | 2.94× |
| `78b2e11c` | 262.8 | 913.3 | 7.49× | 1.98× | **3.48×** |
| `68d6817d` | 252.7 | 116.7 | 6.61× | 13.36× | 0.46× |
| `564007ac` | 262.9 | 923.0 | 7.55× | 1.98× | **3.51×** |
| `ae4219a9` | 261.0 | 915.9 | 6.97× | 1.84× | **3.51×** |
| `232ed014` | 262.0 | 568.9 | 7.53× | 3.19× | 2.17× |
| `7a389715` | 263.2 | 770.7 | 7.46× | 2.35× | 2.93× |
| `5096e459` | 263.5 | 898.9 | 7.51× | 2.02× | 3.41× |
| `d57eb9e1` | 261.1 | 903.6 | 6.51× | 1.74× | 3.46× |
| `2207f0fd` | 262.0 | 905.3 | 7.11× | 1.88× | 3.46× |

### Key observations from the numbers

- **FlashInfer’s latency is nearly flat at ~260 µs** across all traces (min 249 µs, max 283 µs). That is the signature of a kernel that is **saturating tensor cores / memory bandwidth** and spending its time on roughly the **same `T·H·TOPK·D_total` per-token work**, not on scaling with batch.
- **Our CUDA latency scales visibly with workload size** (≈100 µs for small traces up to ≈920 µs for the largest). That is expected for a **one-block-per-`(T,H)`** kernel running **scalar bf16→fp32 dots** and **two passes** over cKV.
- **Small traces:** we’re actually **2–2.6× faster** than FI (launch + workspace overhead dominates their path).
- **Large traces (high `T·H`):** FI pulls ahead by **~3.5×**, which is the main gap to close.

### Caveats

- Correctness **only passes at the bench’s default `atol=rtol=0.01`**; some traces show very large **relative** error when absolute error is already small (printed as `rel_err=5e+02` for values near zero). This is why the TRT-LLM wrapper is kept as a **reference measurement**, not a submission.
- The wrapper returns **`(output,)` only** — no `lse`. Our solution returns `(output, lse)` as the definition expects; do not ship a submission that drops `lse` without confirming the harness accepts it.
- `seq_lens = (sparse_indices != -1).sum(dim=1)` assumes **valid indices are packed at the front** of each row on these traces.

### (Same API overview kept for context)

This is the **TensorRT-LLM–style batch MLA decode** entry point in FlashInfer (often chosen as `trtllm-gen` on Blackwell when `backend="auto"`). It is **related in name** to the contest’s “sparse MLA” trace, but it is **not** a drop-in substitute for the definition’s `run(q_nope, q_pe, …)` without addressing the points below.

### Fixed geometry vs contest tensors

FlashInfer’s own docstring for this function states (abridged):

- `qk_nope_head_dim` **must be 128**
- `kv_lora_rank` **must be 512**
- `qk_rope_head_dim` **must be 64**
- Query layout: concatenated `q_nope ‖ q_rope` with last dim `(qk_nope_head_dim + qk_rope_head_dim)` → **192** for the usual TRT-LLM MLA query head.

The contest definition `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` uses:

- `q_nope`: **\[T, H, 512]**
- `q_pe`: **\[T, H, 64]**

So the **query** head dimensionality is **576** (512 + 64), and the **512** matches the **cKV latent** width used in the dot product, not a **128**-dimensional TRT-LLM “nope” slot. That is a **different MLA wiring** than this API expects: you cannot simply `torch.cat` and pass `qk_nope_head_dim=128` while feeding **512**-wide `q_nope` slices—the constants **128 / 512 / 64** are **hard constraints** on the kernel, not labels you can set to match arbitrary shapes.

**Implication:** A fair **numerical** comparison to your CUDA kernel on **bench workloads** requires either:

- using tensors that **already** satisfy that API’s layout (not what the trace provides), or  
- inserting the **missing linear maps** (e.g. 512→128 on the query side) if your goal is to match TRT-LLM semantics, which is **not** what [`solution/python/kernel.py`](../solution/python/kernel.py) implements.

### Benchmark contract (`run` return value)

`trtllm_batch_decode_with_kv_cache_mla` returns a **single** `torch.Tensor` (decoded output). The FlashInfer-Bench definition for DSA sparse attention expects **`(output, lse)`** with **log-sum-exp in base 2** (see [`solution/python/kernel.py`](../solution/python/kernel.py) and [`solution/cuda/mla_cuda.cu`](../solution/cuda/mla_cuda.cu)). So even if shapes matched, you would still need to **compute or recover `lse`** in a way that matches the reference to pass correctness.

### Sparse indices / `seq_lens` assumption

A pattern like:

```python
seq_lens = (sparse_indices != -1).sum(dim=1).to(torch.int32)
```

only matches kernels that read **only the first `seq_lens[t]`** slots and assume **valid indices are packed at the front**. The reference implementation treats **`-1` anywhere** in the row as invalid (`masked_fill`). If a trace leaves **padding** in the middle or tail, **results differ**. Confirm against the actual trace layout before trusting timings.

### How this relates to `flashinfer_wrapper_5af199`

The dataset baseline **`flashinfer_wrapper_5af199`** is the **intended** apples-to-apples FlashInfer implementation for this definition (same I/O contract as the bench). It is the right thing to benchmark against **on the contest traces**. The TRT-LLM MLA Python API above is a **different product surface** (fixed 128/512/64 TRT-LLM geometry); use it for **performance ideas** and backend behavior (`trtllm-gen` vs `xqa`), not as a guaranteed stand-in for the trace math.

---

## Implementation comparison (techniques)

This is a **qualitative** map for optimization direction, not a claim about every internal detail of `flashinfer_wrapper_5af199`.

| Technique | Definition reference (typical) | This CUDA kernel (`mla_cuda.cu`) | FlashInfer-style baselines (typical goals) |
|-----------|-------------------------------|-----------------------------------|---------------------------------------------|
| Structure | Multiple PyTorch ops → many kernels / temp tensors | **Single fused kernel** per launch | Fused CUDA; minimal host round-trips |
| `QK^T` / projection | **cuBLAS** batched GEMM on gathered `K` | Per-`k` **scalar** dot in registers + shared `Q` | **Tensor Core** (WMMA / CUTLASS / library GEMM) where shapes allow |
| Gather | Index → big `K` tensor then matmul | **Random global gather** per `k` slot | Same gather pressure unless blocked / cached |
| Softmax | `torch.softmax` / `logsumexp` (parallel library kernel) | **Serial** max/sum on thread 0 | Warp/block reductions, often fused with logits |
| Precision | Often fp32 matmul after upcast | bf16 I/O, fp32 softmax accum | bf16/fp8 paths tuned for throughput |
| Duplicate memory traffic | `K` used in logits and in `attn @ K` | **Two passes** over `ckv` rows (logits + output) | Libraries often **fuse** to avoid re-reading `V`/cKV |

**Takeaway for next optimizations:** The largest gaps vs a mature FlashInfer path are likely **Tensor Core matmuls** (or batched GEMM) for the 512+64 dot products, **parallel softmax**, and **eliminating the second pass** over KV rows for cKV—while preserving correctness vs the definition.

---

## Concrete ideas to close the gap to TRT-LLM MLA (`~3.5×` on large traces)

Ranked by expected payoff based on the measured pattern (flat TRT-LLM vs linear scaling of our kernel with `T·H`):

1. **Tensor-Core MMA on the `(Q_nope · K_c) + (Q_pe · K_p)` dots.**
   The 512- and 64-wide inner products per `k` are exactly the shapes WGMMA / WMMA BF16 matmuls eat on Blackwell. Replace the scalar `for d` loops with `mma.sync` / `wgmma.mma_async` over KV tiles. This is the single biggest reason FlashInfer’s large-trace time stays flat.

2. **One-pass online softmax (FlashAttention-2 style), fusing logits + output projection.**
   Currently we write **all 2048 logits to shared memory, softmax them, then re-read cKV** for the value projection. Run a **running-max / running-sum** update over tiles of `k`, update the accumulator for `out = Σ softmax_k · cKV_k` on the fly, and **never revisit cKV**. Saves one global pass over `T·H·TOPK·512` bf16.

3. **Tile K (and Q) across heads, not across `(T,H)`-blocks.**
   Today we launch `grid = (T, H)` and each block re-reads the same `K` rows once per head. Tile so **one block serves all `H=16` heads of a token** (or a group): load a K-row tile **once**, compute logits for every head from it in registers / shared. Typical 4-8× saving on KV bandwidth on the large-`T·H` traces where we lose.

4. **Vectorized bf16 loads.**
   Cast `ckv_flat`/`kpe_flat` accesses to `__nv_bfloat162` / `float4` (128-bit) loads. 8 bf16 per 16-byte transaction is the right unit for B200 L2, and it matches the `D_CK=512` / `D_KP=64` divisibility.

5. **Warp- / block-parallel softmax with `exp2f`.**
   Replace the serial `tid==0` max/sum/norm with a **warp-wide fmax + warp-wide sum** (or 1-warp reduction then broadcast), and switch `expf(x-mx)` to `exp2f(x·log2e - mx·log2e)` so we avoid the extra `logf(2)` at the end (LSE is already base-2). Small absolute win but removes a serial stall in the shadow of compute.

6. **Compact valid indices per block.**
   Some traces leave `-1` padding. Do a quick `__ballot_sync` compaction of valid `gidx` into a block-local array, then all inner loops iterate **only** over valid count, not always 2048. Especially helps short `seq_lens[t]` rows (what `seq_lens = (sparse_indices != -1).sum(dim=1)` in the FI wrapper exploits).

7. **TMA / `cp.async.bulk` pipelined KV tile fetch (Blackwell).**
   Double-buffer KV tile loads so the **next tile is in transit while the current MMA runs**. This is what converts memory-bound gathers into compute-bound matmuls.

8. **Swap the hot path to `wgmma` FP8 for K (if numerics allow).**
   FI clearly has enough compute headroom on B200 that it runs in ~260 µs regardless of trace. An FP8 KV path (BF16 Q × FP8 K) with a scale would approximately double the achievable throughput of the dot-product phase; bench tolerance is `atol=rtol=1e-2`, but some FI runs already show `abs_err = 1.56e-2` — stay within bench tolerance after adding FP8.

9. **Cooperative groups across multiple tokens per block for small `T·H`.**
   Our small-trace wins (`~100 µs`) already beat TRT-LLM; extend that regime by **packing several tokens’ `(T,H)` work into one block** on large traces, so we don’t pay launch / shared-memory setup per `(T,H)` when `T·H` is large.

10. **Persistent-kernel / single-launch strategy.**
    Launch a **fixed number of SM-resident CTAs** that walk a work-queue of `(T,H)` tuples. Avoids per-`(T,H)` launch overhead and keeps warps busy during tail work — one of the reasons TRT-LLM stays flat even when `T·H` shrinks.

Secondary / cleanup wins (should be free):

- Drop the `if (gidx < 0)` branches from the **inner `d` loop** by pre-writing `-INF` to `s_logits[k]` (already done) **and** skipping the row entirely (done) — but also **skip** the row in the second (value) pass by consulting `s_logits[k] == 0` after normalization.
- Only compute `lse` when the bench actually consumes it; right now it is an extra fp32 write per `(T,H)`.
- Precompute `sm_scale * log2e` on the host side and fold it into BMM1 (TRT-LLM does this via `bmm1_scale`).

### Mapping to observed gap

The **constant ~260 µs** TRT-LLM curve implies its inner loop is **GEMM-bound on tensor cores** and the **`T·H`-independent** pieces (softmax, output projection) are **fused**. Items (1)–(3) attack that directly; items (4), (5), (7) reduce the constant factor you need under (1)–(3).

---

## Refreshing this document

After new Modal or local runs, copy the JSON path and replace the artifact reference and tables, or regenerate the summary with:

```bash
python3 -c "import json,statistics,pathlib; p=pathlib.Path('artifacts/modal_baseline_full_20260419T214908Z.json'); d=json.loads(p.read_text()); r=list(d['results'].values())[0]; rows=[(k,v['latency_ms'],v['reference_latency_ms'],v['speedup_factor']) for k,v in r.items()]; print('mean_speedup', statistics.mean(x[3] for x in rows))"
```

Point `p` at your newest `artifacts/modal_*.json`.
