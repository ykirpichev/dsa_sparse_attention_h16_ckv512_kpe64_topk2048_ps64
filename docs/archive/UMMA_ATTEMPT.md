# UMMA / tcgen05 Kernel Attempt — Archived Notes

This directory archives the `v6_umma` / `v7_umma` Blackwell tcgen05 UMMA
experimental kernels. They are **not** part of the submitted solution — we
reverted to `v4_sm100a` (the submission-v1 kernel rebuilt with the sm_100a
target). The files here are kept so we can resume this line of work later
without having to reconstruct the infrastructure from scratch.

## Archived files

| File                          | What it is                                              |
|-------------------------------|---------------------------------------------------------|
| `kernel_v7_umma_attempt.cu`   | Fused-heads kernel: UMMA logit pass (working) + UMMA value pass (numerically broken). |
| `tcgen05_ptx.h`               | Inline-PTX wrappers for `tcgen05.mma / .ld / .alloc / .commit / .wait` + fences. |
| `umma_desc.h`                 | Vendored CUTLASS-style `SmemDescriptor` / `InstrDescriptor` bitfields + builders. |

## High-level design (what we tried)

**Grid:** one CTA per token (`dim3(T,)`), 8 warps × 32 lanes = 256 threads.
All 16 heads are processed inside a single CTA so `sparse_idx[T, TOPK]` and
the `ckv` / `kpe` rows are loaded only once per token instead of 16× (MLA
has no head dimension in K / V).

**Logit pass (working):** two `tcgen05.mma.cta_group::1.kind::f16` chains,
`Q_nope · K_ckv^T` (32 × K-atom=16 bf16 = K_total=512) then
`Q_pe · K_kpe^T` (4 × K-atom=16 bf16 = K_total=64, accum=1), `MMA_M=64`
padded (16 real heads + 48 zero rows), `MMA_N=BK=16`, output in TMEM.
Operands in SMEM use the canonical `SWIZZLE_128B` K-major layout from
CUTLASS `((8,m),(T,2k)):((8T,SBO),(1,T))`.

**Softmax + online accumulation:** warp-cooperative max / sum (`__shfl_xor`),
per-head running `(m, l)` in SMEM, per-thread output register
`O[HEADS=16][DIMS_PER_THREAD=2]`.

**Value pass (broken):** softmax probabilities written back to
`s_p_umma [MMA_M=64, BK=16]` bf16 in a `SWIZZLE_NONE` K-major
"interleave" layout, V transposed from `s_ckv [BK=16, D_CK=512]`
(`SWIZZLE_128B` K-major) into `s_v_t [D_CK=512, BK=16]` bf16
(`SWIZZLE_NONE` K-major). A single `tcgen05.mma` with `MMA_M=64`,
`MMA_N=VALUE_N_HALF=256`, K=BK=16 produces `P · V^T` in TMEM;
issued in two halves so total N=512=D_CK. 8 warps cooperatively read
TMEM back into an `s_pv[HEADS, D_CK]` fp32 staging buffer, then the
per-thread online-softmax update writes to `O[h][d]`.

## Status

### Pass: UMMA logit pass
`v6_umma` (UMMA logit pass + **scalar** value pass) passed all 23 workloads
on Modal B200 but was **~15× geomean** — a regression vs `v4_sm100a`
(~35.6× geomean). The regression is structural, not numerical:

1. **Lost grid parallelism.** Collapsing the grid from `(T, H=16) → (T,)` puts
   16× fewer CTAs on the SMs. For large-T workloads the per-(T,H) kernel was
   already saturating the B200's ~148 SMs.
2. **MMA_M padding waste.** `tcgen05.mma.kind::f16` smallest `MMA_M = 64`,
   but only 16 rows are real heads. 48/64 ≈ 75 % of MMA cycles are wasted.
3. **Scalar value pass** still dominates the loop. UMMA only accelerated the
   logit pass, which is a smaller fraction of the critical path.

### Fail: UMMA value pass
`v7_umma` (full UMMA) fails numerical correctness. Debugging flow
(summarised from the transcript):

* Isolated the broken stage by **replacing the MMA with a scalar `P · V`**
  using the exact same `s_p_umma` and `s_v_t` buffers. That **passed**
  correctness → the inputs in shared memory are correctly placed in the
  supposed `SWIZZLE_NONE` K-major layout, so the bug is in the UMMA's
  interpretation of the descriptor.
* Walked both the NVIDIA PTX ISA `Table 40` (shared-memory descriptor)
  and CUTLASS `include/cute/atom/mma_traits_sm100.hpp::make_umma_desc()`
  carefully. Derived the canonical layout

  ```text
  LayoutType::INTERLEAVE : ((8,n),2) : ((1,SBO),LBO)   // K-major, in uint128
  ```

  which for packed bf16 gives `SBO = 8 uint128 = 128 B`,
  `LBO = M_total uint128 = M_total·16 B`. CUTLASS specifically swaps
  which field carries LBO vs SBO for `SWIZZLE_NONE` (see
  `make_umma_desc` branches), but for the **K-major** path the assignment
  ends up `desc.stride_byte_offset_ = SBO` and
  `desc.leading_byte_offset_ = LBO`, which is what the archived
  `make_kmajor_desc_noswz` produces.
* Despite descriptor and layout matching all three references (PTX docs,
  CUTLASS code, CUTLASS comments), the tcgen05 MMA produces garbage
  (`abs_err ≈ 3–4`, `rel_err` up to `~47`). Not debugged further.
* Cross-checked against CUTLASS `make_umma_desc<K-major>()`:
  `desc.stride_byte_offset_ = stride_01 = SBO`,
  `desc.leading_byte_offset_ = stride_10 = LBO` (in uint128 units, so the
  encoded values shift out the low 4 bits). Our archived `make_kmajor_desc_noswz`
  uses `SBO=128 B → encoded 8` and `LBO=M_total·16 B → encoded M_total`,
  i.e. matches the CUTLASS reference, yet the MMA still fails numerically.
  The scalar control test in this configuration PASSES
  (`abs_err=1.56e-02`, no UMMA used), confirming data is correctly laid
  out in memory. So the remaining bug is in how the UMMA instruction
  decodes the descriptor (alignment? `base_offset_` encoding? some fence
  we are missing?). Did not find it in time.

Likely remaining issues:

* Base-offset / alignment requirement we are not honouring for
  `SWIZZLE_NONE` (PTX Table 41 gives the repeating-pattern alignment for
  the swizzled modes but not for no-swizzle).
* The CUTLASS canonical check expects `size<0>(u128_tensor) % 8 == 0` —
  satisfied for P (M=64) and V^T (M=512), so that's not it.
* Some CUTLASS examples hint that the `leading_byte_offset_mode` bit
  (bit 52 of the shared-memory descriptor) may need `1` on `sm_103a` for
  the "byte address absolute" LBO mode; we kept it at `0` (legacy
  relative). Worth trying `1` next time.

## Possible paths forward (when resuming)

### Option (b): Per-(T, H) CTA + tiny UMMA
Go back to `dim3(T, HEADS)` so the grid is saturated, then use a single
`tcgen05.mma` per CTA with `MMA_M=16` (one row per head handled by the
CTA → no padding waste) or the minimum MMA_M=64 with `BK = 64` (pack 4
BK-tiles into one MMA). Value pass can stay scalar initially.

### Option (d): UMMA value pass via `b_major = MN`
The current attempt uses `b_major = K` (V transposed into `s_v_t`). An
alternative avoids the transpose entirely: store V in the logit-pass
`s_ckv` buffer (we already do) and set `b_major = MN` for the value MMA.
This needs the `s_ckv` SMEM layout to be **MN-major canonical**, not the
K-major one used for the logit pass, so it requires repacking. The
memory saving is non-trivial (16 KiB of `s_v_t` dropped).

### Option (e): Switch V^T layout to `SWIZZLE_128B` with K padded to 64
`SWIZZLE_128B` is the layout we know works (logit pass passes
correctness). K needs to be a multiple of 64 bf16 (one full 128 B
K-block); pad V's K-dim from `BK=16` to `BK_pad=64` (zero the rest).
Cost: +48 KiB `s_v_t`, +6 KiB `s_p_umma`. We'd need to drop
`V_PIPE=2 → 1` (save 36 KiB) to fit under the 200 KiB per-CTA SMEM
ceiling we empirically found.

## Reproducing a compile of the archived kernel

```bash
cp docs/archive/kernel_v7_umma_attempt.cu solution/cuda/kernel.cu
cp docs/archive/tcgen05_ptx.h              solution/cuda/tcgen05_ptx.h
cp docs/archive/umma_desc.h                solution/cuda/umma_desc.h
# then edit config.toml: name = "wombat_dsa_sparse_mla_cuda_v6_umma"
modal run scripts/run_modal.py --smoke
```

## Reference material used

* NVIDIA CUDA PTX ISA §9.7.16 — Shared memory descriptor (Table 40),
  Instruction descriptor (Table 42), repeating-pattern alignment
  (Table 41), `tcgen05.mma` / `.ld` / `.alloc` / `.commit`.
* CUTLASS `include/cute/arch/mma_sm100_desc.hpp` — `SmemDescriptor`
  bitfield + `LayoutType` enum.
* CUTLASS `include/cute/atom/mma_traits_sm100.hpp::make_umma_desc` —
  reference construction of the descriptor from a CuTe tensor layout
  (this is the authoritative source for SBO / LBO placement per
  `LayoutType`).
* CUTLASS `include/cute/arch/mma_sm100.hpp` — PTX wrappers for
  `tcgen05.mma.cta_group::{1,2}.kind::{f16,tf32,f8f6f4,...}`.
* Blog: "tcgen05 for dummies" — worked examples of the canonical
  layouts with figures (useful for intuition, but the PTX docs + CUTLASS
  code are the source of truth).
