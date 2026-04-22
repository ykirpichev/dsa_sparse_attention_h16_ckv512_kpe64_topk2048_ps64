// SPDX-License-Identifier: BSD-3-Clause
//
// Inline-PTX wrappers for the subset of SM100 (Blackwell, sm_100a)
// instructions needed for the fused-heads DSA MLA attention kernel:
//
//   * Tensor Memory (TMEM) allocation (tcgen05.alloc / dealloc /
//                                      relinquish_alloc_permit)
//   * F16 UMMA with bf16 inputs        (tcgen05.mma.kind::f16, .ss variant)
//   * TMEM readback                    (tcgen05.ld.sync.aligned.32x32b.x{N}.b32)
//   * TMEM commit + wait               (tcgen05.commit, tcgen05.wait::ld)
//   * Fences                           (tcgen05.fence::{before,after}_thread_sync)
//   * cp.async                         (cp.async.ca / commit / wait)
//   * mbarrier                         (init / inval / arrive / try_wait_parity)
//
// Derived from the DSA TopK-indexer project's tcgen05_ptx.h
// (https://github.com/ykirpichev/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64),
// with additional F16-kind UMMA wrapper and additional tmem load widths.

#pragma once

#include <cstdint>

#include "umma_desc.h"

namespace dsa_ptx {

// ---------------------------------------------------------------------------
// SMEM generic-to-shared address conversion (bypasses aperture test).
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* smem_ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 u64_addr;\n"
                 "  cvta.to.shared.u64 u64_addr, %1;\n"
                 "  cvt.u32.u64 %0, u64_addr; }\n"
                 : "=r"(addr)
                 : "l"(smem_ptr));
    return addr;
}

// ---------------------------------------------------------------------------
// cp.async (SM_80+): 16 / 8 / 4 byte async copies from global to shared.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
cp_async_16B(uint32_t smem_addr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr));
#else
    (void)smem_addr; (void)gmem_ptr; __trap();
#endif
}

__device__ __forceinline__ void
cp_async_8B(uint32_t smem_addr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr));
#else
    (void)smem_addr; (void)gmem_ptr; __trap();
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// ---------------------------------------------------------------------------
// mbarrier (SM_80+)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mbarrier_init(uint32_t mbar_smem_addr, uint32_t count) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :
                 : "r"(mbar_smem_addr), "r"(count));
#else
    (void)mbar_smem_addr; (void)count;
#endif
}

// Required after `mbarrier.init` on SM90+ so other threads see the
// initialised barrier before any arrive / wait occurs.
__device__ __forceinline__ void fence_barrier_init() {
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n");
#elif __CUDA_ARCH__ >= 800
    __threadfence_block();
#endif
}

__device__ __forceinline__ void mbarrier_inval(uint32_t mbar_smem_addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :
                 : "r"(mbar_smem_addr));
#else
    (void)mbar_smem_addr;
#endif
}

__device__ __forceinline__ bool
mbarrier_try_wait_parity(uint32_t mbar_smem_addr, uint32_t phase) {
#if __CUDA_ARCH__ >= 800
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(done)
                 : "r"(mbar_smem_addr), "r"(phase));
    return done != 0u;
#else
    (void)mbar_smem_addr; (void)phase; return true;
#endif
}

__device__ __forceinline__ void
mbarrier_wait_parity(uint32_t mbar_smem_addr, uint32_t phase) {
    while (!mbarrier_try_wait_parity(mbar_smem_addr, phase)) { /* spin */ }
}

// ---------------------------------------------------------------------------
// TMEM allocation (SM_100a)
// ---------------------------------------------------------------------------
// `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], ncols;`
// Stores the 32-bit TMEM pointer into shared memory at `dst`.
// Must be issued by a single fully-active warp of the CTA.

__device__ __forceinline__ void
tcgen05_alloc_1sm(uint32_t dst_smem_addr, uint32_t num_columns) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(dst_smem_addr), "r"(num_columns));
#else
    (void)dst_smem_addr; (void)num_columns; __trap();
#endif
}

__device__ __forceinline__ void
tcgen05_dealloc_1sm(uint32_t tmem_addr, uint32_t num_columns) {
#if __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(tmem_addr), "r"(num_columns));
#else
    (void)tmem_addr; (void)num_columns; __trap();
#endif
}

__device__ __forceinline__ void
tcgen05_relinquish_alloc_permit_1sm() {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
#else
    __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05 fences (SM_100a)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
#if __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.fence::before_thread_sync;\n");
#endif
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
#if __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.fence::after_thread_sync;\n");
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.commit: signal an mbarrier when outstanding MMA / LD retires.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
tcgen05_commit_1sm(uint32_t mbar_smem_addr) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];\n"
        :
        : "r"(mbar_smem_addr));
#else
    (void)mbar_smem_addr; __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.mma.cta_group::1.kind::f16  (dense f16/bf16 UMMA, SS variant)
// ---------------------------------------------------------------------------
//
//   [tmem_c] = scale_c * desc_a * desc_b [+ [tmem_c]]
//
// desc_a / desc_b : 64-bit SmemDescriptor values
// instr_desc_hi   : the 32-bit InstrDescriptor passed as the 4th operand
// accumulate      : if non-zero, += into [tmem_c] ; if zero, overwrite.

__device__ __forceinline__ void
tcgen05_mma_f16_ss(uint32_t tmem_c,
                   uint64_t desc_a,
                   uint64_t desc_b,
                   uint32_t instr_desc_hi,
                   uint32_t accumulate) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 "
        "  [%0], %1, %2, %3, p;\n"
        "}\n"
        :
        : "r"(tmem_c),
          "l"(desc_a),
          "l"(desc_b),
          "r"(instr_desc_hi),
          "r"(accumulate));
#else
    (void)tmem_c; (void)desc_a; (void)desc_b;
    (void)instr_desc_hi; (void)accumulate; __trap();
#endif
}

__device__ __forceinline__ void
tcgen05_mma_f16_ss(uint32_t                   tmem_c,
                   dsa_umma::SmemDescriptor   desc_a,
                   dsa_umma::SmemDescriptor   desc_b,
                   dsa_umma::InstrDescriptor  instr_desc,
                   uint32_t                   accumulate) {
    tcgen05_mma_f16_ss(
        tmem_c,
        static_cast<uint64_t>(desc_a),
        static_cast<uint64_t>(desc_b),
        static_cast<uint32_t>(instr_desc.desc_),
        accumulate);
}

// ---------------------------------------------------------------------------
// tcgen05.mma.cta_group::1.kind::f8f6f4 (kept for indexer parity; unused in
// the DSA attention kernel but useful if this header is shared).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
tcgen05_mma_f8f6f4_ss(uint32_t tmem_c,
                      uint64_t desc_a,
                      uint64_t desc_b,
                      uint32_t instr_desc_hi,
                      uint32_t accumulate) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 "
        "  [%0], %1, %2, %3, p;\n"
        "}\n"
        :
        : "r"(tmem_c),
          "l"(desc_a),
          "l"(desc_b),
          "r"(instr_desc_hi),
          "r"(accumulate));
#else
    (void)tmem_c; (void)desc_a; (void)desc_b;
    (void)instr_desc_hi; (void)accumulate; __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.wait: block the issuing thread until prior tcgen05 loads retire.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void tcgen05_wait_ld() {
#if __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
#else
    __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.ld.sync.aligned.32x32b.x{N}.b32
// ---------------------------------------------------------------------------
//
// Reads N column-tiles of 32 DP rows × 32 bits (fp32) from TMEM back into
// 32-bit registers.  For M=64 UMMA, two warps (warp 0 and warp 1) share the
// M dimension: warp 0 reads rows 0..31, warp 1 reads rows 32..63.
//
// Each lane receives N 32-bit values; .xN is the number of contiguous columns
// to fetch from TMEM starting at `tmem_addr`.

__device__ __forceinline__ void
tcgen05_ld_32x32b_x16_b32(uint32_t tmem_addr, uint32_t (&out)[16]) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{ %0, %1, %2, %3, %4, %5, %6, %7, "
        "  %8, %9, %10, %11, %12, %13, %14, %15 }, "
        "[%16];\n"
        : "=r"(out[ 0]), "=r"(out[ 1]), "=r"(out[ 2]), "=r"(out[ 3]),
          "=r"(out[ 4]), "=r"(out[ 5]), "=r"(out[ 6]), "=r"(out[ 7]),
          "=r"(out[ 8]), "=r"(out[ 9]), "=r"(out[10]), "=r"(out[11]),
          "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15])
        : "r"(tmem_addr));
#else
    (void)tmem_addr; for (int i = 0; i < 16; ++i) out[i] = 0u; __trap();
#endif
}

__device__ __forceinline__ void
tcgen05_ld_32x32b_x32_b32(uint32_t tmem_addr, uint32_t (&out)[32]) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{ %0, %1, %2, %3, %4, %5, %6, %7, "
        "  %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31 }, "
        "[%32];\n"
        : "=r"(out[ 0]), "=r"(out[ 1]), "=r"(out[ 2]), "=r"(out[ 3]),
          "=r"(out[ 4]), "=r"(out[ 5]), "=r"(out[ 6]), "=r"(out[ 7]),
          "=r"(out[ 8]), "=r"(out[ 9]), "=r"(out[10]), "=r"(out[11]),
          "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15]),
          "=r"(out[16]), "=r"(out[17]), "=r"(out[18]), "=r"(out[19]),
          "=r"(out[20]), "=r"(out[21]), "=r"(out[22]), "=r"(out[23]),
          "=r"(out[24]), "=r"(out[25]), "=r"(out[26]), "=r"(out[27]),
          "=r"(out[28]), "=r"(out[29]), "=r"(out[30]), "=r"(out[31])
        : "r"(tmem_addr));
#else
    (void)tmem_addr; for (int i = 0; i < 32; ++i) out[i] = 0u; __trap();
#endif
}

__device__ __forceinline__ void
tcgen05_ld_32x32b_x64_b32(uint32_t tmem_addr, uint32_t (&out)[64]) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{ %0, %1, %2, %3, %4, %5, %6, %7, "
        "  %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63 }, "
        "[%64];\n"
        : "=r"(out[ 0]), "=r"(out[ 1]), "=r"(out[ 2]), "=r"(out[ 3]),
          "=r"(out[ 4]), "=r"(out[ 5]), "=r"(out[ 6]), "=r"(out[ 7]),
          "=r"(out[ 8]), "=r"(out[ 9]), "=r"(out[10]), "=r"(out[11]),
          "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15]),
          "=r"(out[16]), "=r"(out[17]), "=r"(out[18]), "=r"(out[19]),
          "=r"(out[20]), "=r"(out[21]), "=r"(out[22]), "=r"(out[23]),
          "=r"(out[24]), "=r"(out[25]), "=r"(out[26]), "=r"(out[27]),
          "=r"(out[28]), "=r"(out[29]), "=r"(out[30]), "=r"(out[31]),
          "=r"(out[32]), "=r"(out[33]), "=r"(out[34]), "=r"(out[35]),
          "=r"(out[36]), "=r"(out[37]), "=r"(out[38]), "=r"(out[39]),
          "=r"(out[40]), "=r"(out[41]), "=r"(out[42]), "=r"(out[43]),
          "=r"(out[44]), "=r"(out[45]), "=r"(out[46]), "=r"(out[47]),
          "=r"(out[48]), "=r"(out[49]), "=r"(out[50]), "=r"(out[51]),
          "=r"(out[52]), "=r"(out[53]), "=r"(out[54]), "=r"(out[55]),
          "=r"(out[56]), "=r"(out[57]), "=r"(out[58]), "=r"(out[59]),
          "=r"(out[60]), "=r"(out[61]), "=r"(out[62]), "=r"(out[63])
        : "r"(tmem_addr));
#else
    (void)tmem_addr; for (int i = 0; i < 64; ++i) out[i] = 0u; __trap();
#endif
}

} // namespace dsa_ptx
