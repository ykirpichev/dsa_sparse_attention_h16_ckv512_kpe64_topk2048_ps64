# Optimization ideas for `solution/cuda/mla_cuda.cu`

Dedicated backlog so work can be picked up later without re-deriving context. For the measured numbers backing these ideas see [`docs/PERFORMANCE_VS_FLASHINFER.md`](PERFORMANCE_VS_FLASHINFER.md).

## Current state (Modal B200, full suite)

| Solution | Mean latency | Mean speedup vs definition ref | Notes |
|----------|-------------:|-------------------------------:|-------|
| Our CUDA (`baseline_dsa_sparse_mla_cuda_v1`) | 0.503 ms | 5.13× | `solution/cuda/mla_cuda.cu` + `binding.py` (TVM FFI) |
| FlashInfer TRT-LLM MLA (reference) | 0.259 ms | 6.05× | `solution/flashinfer_trtllm/kernel.py`, 23/23 PASSED |

Observed pattern:

- **FlashInfer** ≈ **constant ~260 µs** for all 23 traces — tensor-core / memory saturated.
- **Ours** scales with `T·H`: **~100 µs → ~920 µs**.
- **Small traces:** we are **~2–2.6× faster** than FI.
- **Large traces:** FI is **~3.5× faster** than us — this is the gap to close.

## Kernel structure today (one CTA per `(token, head)`)

See [`solution/cuda/mla_cuda.cu`](../solution/cuda/mla_cuda.cu). Key phases inside the kernel:

1. Broadcast load of `q_nope[t,h,:]` and `q_pe[t,h,:]` into shared.
2. `for k in [0..TOPK)`: random gather of `ckv[gidx,:]` (512 bf16) + `kpe[gidx,:]` (64 bf16), scalar `bf16→fp32` dot, write `s_logits[k]`.
3. Serial (thread 0) max / exp / sum / normalize on 2048 logits, then base-2 LSE.
4. `for d in [0..512)`: re-gather `ckv[gidx,:]` for every `k` and accumulate the softmax-weighted value projection.

The two dominant problems are (a) **scalar dot products**, not tensor-core matmuls, for the 512+64-wide reductions, and (b) **two full passes** over `ckv` rows (logits pass + value projection pass).

---

## Top 10 ideas (ranked by expected payoff)

Each item includes the symptom it targets and a first-cut implementation note.

### 1. Tensor-Core MMA (WGMMA / WMMA BF16) for `Q·K^T` dots
- **Why:** The 512- and 64-wide inner products per `k` are the exact shapes B200 tensor cores consume. Scalar `acc += s_qn[d] * bf16_to_f(kc[d])` will never match an mma pipeline.
- **Rough plan:** Stage a tile of `B_K` rows of cKV/kPE into shared memory as a bf16 matrix, treat `Q_nope` / `Q_pe` as the `M=1` (or `M=H_per_block`) side, and issue `wgmma.mma_async` / `mma.sync.m16n8k16` (or `m64n8k16`) ops. Accumulate in fp32, then scale and write logits.
- **Correctness guardrails:** Keep final LSE base-2; only the inner reduction changes.
- **Expected impact:** Largest single lever. Targets the flat-latency regime FlashInfer lives in.

### 2. One-pass online softmax + fused value projection (FlashAttention-2 style)
- **Symptom:** We read every selected `ckv` row twice — once for logits, once for `attn @ cKV`.
- **Plan:** Walk `k` in tiles. Maintain per-thread `m_i` (running max), `l_i` (running sum), and a per-head `o` accumulator (512 fp32 lanes). For each tile: compute logits, rescale `o` and `l_i` by `exp(m_old - m_new)`, then add `softmax_tile · cKV_tile` to `o`. Never revisit cKV.
- **Expected impact:** Halves KV bandwidth on the critical path; prerequisite for item 1 to not be re-bottlenecked by reload.

### 3. Tile across heads in one block (share KV loads per token)
- **Symptom:** Today `grid = (T, H)` and **each head re-reads the same K rows**.
- **Plan:** Launch `grid = (T,)` (or `grid = (T, H / Hg)`), one CTA handles `Hg ≤ H` heads of the same token. Load K rows once, compute logits/output for every head from the same shared K tile. For `H = 16`, even `Hg = 4` saves ~4× on KV traffic.
- **Expected impact:** Big on large-`T·H` traces where we currently lose 3–3.5×.

### 4. Vectorized bf16 loads
- **Plan:** Access `ckv_flat` / `kpe_flat` via `__nv_bfloat162` (4 bytes → 2 bf16) or `float4` (16 bytes → 8 bf16). `D_CK=512` and `D_KP=64` are divisible, rows are contiguous, alignment is 16 B by default.
- **Expected impact:** Reduces memory-instruction issue count significantly — useful on top of items 1–3.

### 5. Warp-parallel softmax with `exp2f`
- **Symptom:** `if (tid == 0) { for k … }` is a serial stall that scales with TOPK.
- **Plan:** Warp reduction for `max`, a block-wide or warp-wide `+` for `sum`, broadcast `1/sum` through shared mem. Use `exp2f(x · log2e − m · log2e)` so no extra `logf(2)` at the end (`lse` is already base-2).
- **Expected impact:** Small absolute win, but removes a serial tail that shows up on small traces where everything else is already fast.

### 6. Compact valid `sparse_idx` per block
- **Symptom:** We always iterate `TOPK=2048`, even when many `gidx == -1`.
- **Plan:** Warp `__ballot_sync` + prefix-sum → pack valid `gidx` into a block-local `int` array of size `cnt ≤ TOPK`. Iterate only `cnt`. Matches what the FI wrapper does with `seq_lens = (sparse_indices != -1).sum(dim=1)`.
- **Expected impact:** Traces with padding get proportional wins.

### 7. TMA / `cp.async.bulk` pipelined KV tile fetch
- **Plan:** Double-buffer: while the compute pipe issues MMA on tile `t`, the TMA engine async-copies tile `t+1` into the other half of shared memory. Sync with `cp.async.mbarrier` or Hopper-style barriers.
- **Expected impact:** Hides gather latency behind compute — converts a memory-bound loop into a compute-bound one once items 1 / 4 land.

### 8. FP8 KV path (within bench tolerance)
- **Symptom:** FI seems to exploit FP8 throughput internally. Bench tolerance is `atol = rtol = 1e-2`.
- **Plan:** Convert cKV / kPE to FP8 (E4M3) ahead of the kernel, dequantize per-tile with a precomputed scale (or use `wgmma` FP8×BF16). Validate on all 23 traces; we already see `abs_err ≈ 1.56e-2` from the FI wrapper — stay strictly under the threshold after dequant.
- **Expected impact:** Up to ~2× on the dot-product phase if items 1–2 are in place.

### 9. Pack multiple tokens per CTA on large traces
- **Symptom:** We already win on small `T·H`; the tail is large-`T·H`.
- **Plan:** Let a single CTA handle a small batch of tokens (`Tg` tokens × `Hg` heads). Shared Q cache grows to `Tg·Hg·(512+64) · 2 B`, but KV loads stay shared. Choose `Tg·Hg` to fit in ~96 KiB shared memory on sm_100.
- **Expected impact:** Extends the small-trace efficiency into the large-trace regime.

### 10. Persistent kernel with work queue
- **Symptom:** Many small launches and tail imbalance on large `T·H`.
- **Plan:** Fixed number of resident CTAs (e.g. `num_sm * cta_per_sm`) loop on `atomicAdd(&work_counter, 1)` to pull the next `(T, H)` tile. Pair with items 3 / 9 so each work item does enough work to amortize the atomic.
- **Expected impact:** Removes per-`(T,H)` launch and setup overhead; keeps warps busy during ragged tails. This is a big part of why TRT-LLM stays flat across trace sizes.

---

## Secondary cleanups (cheap / should be free)

- Precompute `sm_scale * log2e` host-side; pass the fused constant. Matches TRT-LLM’s `bmm1_scale`.
- In the value-projection pass, skip rows whose normalized weight is effectively 0 (e.g. `< 2^-24`) instead of always iterating all `k`.
- Remove the per-`k` `gidx < 0 || gidx >= num_kv_rows` branch from the inner `d` loop by pre-filtering in item 6.
- Only materialize `lse` when downstream consumers need it; today we always write `lse_out`.

---

## Numerics / correctness guardrails

- Final `lse` **must** stay base-2 (`mx + logf(sum)) / logf(2.f)` today). If you switch softmax to `exp2`, the formula becomes `mx · log2e + log2f(sum)` — double-check.
- `__nv_bfloat16` accumulation must remain in fp32; do not fuse weighted value projection in bf16.
- Bench default tolerance is `atol = rtol = 1e-2` but per-workload evaluation has been observed tighter on our current kernel (`abs_err ≤ ~8e-3`). Keep a safety margin before turning on FP8.

---

## Measurement & validation workflow

For each change, run the same full suite and diff against the current artifact:

```bash
python3 scripts/pack_solution.py                        # or -c config.*.toml
modal run scripts/run_modal.py::main --no-smoke --save artifacts/modal_baseline_full_<tag>.json
python3 - <<'PY'
import json, statistics, pathlib
a = json.loads(pathlib.Path("artifacts/modal_baseline_full_<tag>.json").read_text())
b = json.loads(pathlib.Path("artifacts/modal_baseline_full_20260419T214908Z.json").read_text())
d = next(iter(a["results"]))
for w, ra in a["results"][d].items():
    rb = b["results"][d][w]
    print(f"{w[:8]}  new {ra['latency_ms']*1000:7.2f}us  old {rb['latency_ms']*1000:7.2f}us  ratio {rb['latency_ms']/ra['latency_ms']:.2f}x")
PY
```

If a change regresses any workload by more than a few % for no benefit elsewhere, revert or gate it per-trace size.

---

## B200-specific ideas (sm_100a / Blackwell only)

We already gate the kernel with `require_b200()` but don't touch any Blackwell hardware. These ideas require sm_100a PTX and **will not run on older GPUs**.

### Build / launch setup (prerequisites)

- Add explicit arch + diagnostics to [`solution/cuda/binding.py`](../solution/cuda/binding.py) build flags:

  ```python
  extra_cuda_cflags=[
      "-O3", "--use_fast_math",
      "-gencode=arch=compute_100a,code=sm_100a",
      "-Xptxas", "-v",            # register / SMEM report
  ]
  ```

- Opt into the large per-CTA shared memory budget (B200 supports ~228 KiB dyn SMEM per CTA — we currently use ~10 KiB):

  ```cpp
  cudaFuncSetAttribute(
      mla_sparse_attn_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      228 * 1024);
  ```

- Add `__launch_bounds__(256, 2)` (or equivalent) so the compiler picks a register budget matched to our target occupancy.

### Hardware features we should actually use

1. **5th-gen Tensor Cores with Tensor Memory (`tcgen05.mma`, `tcgen05.alloc`).**
   Blackwell adds a new Tensor Memory address space used by `tcgen05.mma` (BF16 × BF16 → FP32 and FP8 × FP8 variants, up to `m128n128k{16,32}` shapes). Put the 512-wide output accumulator and per-tile `Q·K^T` results in `tmem` instead of registers/SMEM. Pairs naturally with idea #1 in the top-10.

2. **2-CTA cluster MMA.**
   `tcgen05.mma.cta_group::2.kind::...` makes **two CTAs of a thread-block cluster cooperatively execute one MMA**, effectively doubling peak TC throughput per work item. Map one cluster to a pair of heads of the same token (we have `H=16` heads — 8 clusters per token).

3. **TMA bulk-tensor loads with cluster multicast.**
   `cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx.bulk_group.multicast::cluster` loads the same KV tile into **every CTA of the cluster** from a **single HBM read**. Perfect for "same token, many heads" — we're currently re-reading cKV rows per head.

4. **TMA descriptors for gathered cKV / kPE rows.**
   Describe cKV as a 2D TMA tensor map (`num_pages * page_size` rows × `D_CK` cols, 128-B swizzle). TMA handles address generation, global→shared copy, and bank-conflict-free layout for us. Replaces the manual `ckv_flat + gidx * D_CK` scalar gather.

5. **Distributed Shared Memory (DSMEM) for Q broadcast.**
   In a cluster, store each token's Q tiles in one CTA's SMEM and have sibling CTAs read via `ld.shared::cluster` (DSMEM). Eliminates duplicated Q loads across heads/CTAs.

6. **Async pipeline with `mbarrier::arrive.expect_tx` and phased barriers.**
   Triple-buffer KV tiles: while CTA is MMA'ing tile *t*, TMA is fetching tile *t+1*, and tile *t−1* is free. Use phased `mbarrier` waits (`cp.async.mbarrier.wait_parity`) instead of `__syncthreads`.

7. **FP8 MMA (`tcgen05.mma.kind::f8f6f4`).**
   BF16 Q × FP8 K is a supported Blackwell MMA shape. Store cKV / kPE in FP8 (E4M3) in HBM with a per-row or per-page scale, dequantize fused at MMA output. Bench tolerance is `atol = rtol = 1e-2`; FI's own wrapper already produces `abs_err ≈ 1.56e-2`, so there is headroom if we validate.

8. **Large L2 residency control (`cudaAccessPolicyWindow` / `cudaStreamSetAttribute`).**
   B200's L2 is ~128 MB — plenty of room for the hot subset of cKV pages. Mark the cKV cache window as `cudaAccessProperty::Persistent` so the hardware prefers keeping it resident across launches.

9. **Persistent kernel on ~148 SMs.**
   Launch `~296 CTAs` (2 per SM) once, consume `(T, H)`-tuples from an atomic work counter. Amortizes TMA-descriptor setup, `tcgen05.alloc`, and `cudaFuncSetAttribute` choices across all tokens. Complements "one block per (T,H)" in the top-10 list.

10. **Packed `__bfloat1622float2` / `__hfma2` for the fallback scalar path.**
    Until the MMA path lands, use the **packed** bf16 converter and half2 FMA so each 32-bit load already provides 2 ops — halves the scalar-loop latency we measured dominating large traces.

11. **`exp2f` with precomputed `sm_scale * log2e` constant.**
    Base-2 LSE output means we can skip the `logf(2)` divide. Pass `sm_scale * log2e` into the kernel; use `exp2f(acc - mx)` with `mx` already in base-2. Tiny kernel but hits the serial softmax.

12. **`__grid_constant__` kernel parameters.**
    Pass TMA tensor maps (`CUtensorMap`) and scalar constants as `__grid_constant__` — they land in constant memory, not SMEM, and all threads share one copy.

13. **`cluster.sync.aligned` + DSMEM reduction for softmax across CTAs.**
    If we split TOPK=2048 across CTAs of a 2- or 4-CTA cluster (1024 or 512 logits per CTA), do the FlashAttention-2 running-max/sum reduction across CTAs via DSMEM. Converts a serial tail-reduction into a parallel cluster reduction.

14. **`cp.async` (non-bulk) for small Q loads, bulk TMA for KV.**
    For the tiny `Q_nope[512] + Q_pe[64]` load, plain `cp.async.shared::cta.global` already beats explicit loads; reserve bulk TMA for the expensive KV tiles.

### Mapping to measured gap vs TRT-LLM

The `~3.5×` gap on large traces comes from:

| Gap component | Primary fix |
|---------------|-------------|
| Scalar dot vs tensor-core MMA | (1), (2), (7), (10) |
| Redundant HBM re-reads of same cKV rows | (3), (5), (8) |
| Blocking memory latency | (4), (6) |
| Per-`(T,H)` kernel launch overhead | (9) |
| Serial softmax / LSE | (11), (13) |

Items (1)–(4) + (6) together are the minimum set to explain TRT-LLM's flat-latency shape.

---

## Useful pointers

- Our kernel: [`solution/cuda/mla_cuda.cu`](../solution/cuda/mla_cuda.cu)
- Binding: [`solution/cuda/binding.py`](../solution/cuda/binding.py)
- Reference (math-ground-truth we must match): [`solution/python/kernel.py`](../solution/python/kernel.py)
- TRT-LLM MLA reference wrapper: [`solution/flashinfer_trtllm/kernel.py`](../solution/flashinfer_trtllm/kernel.py), config [`config.flashinfer_trtllm.toml`](../config.flashinfer_trtllm.toml)
- FlashInfer dataset baseline name for head-to-head: `flashinfer_wrapper_5af199` (see [`EVALUATION.md`](../EVALUATION.md#flashinfer-baselines))
