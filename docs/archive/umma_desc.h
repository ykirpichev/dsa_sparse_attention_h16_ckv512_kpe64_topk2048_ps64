// SPDX-License-Identifier: BSD-3-Clause
//
// Minimal vendored subset of CUTLASS UMMA descriptor layouts required to
// drive SM100 (Blackwell) `tcgen05.mma` instructions via inline PTX.
//
// Source (verbatim bitfields, reformatted):
//   NVIDIA CUTLASS, `cute/arch/mma_sm100_desc.hpp`
//   <https://github.com/NVIDIA/cutlass> (BSD-3-Clause)
//
// We vendor the subset of descriptors + enums needed to build
//   * a shared-memory UMMA descriptor pair (SmemDescriptor)
//   * an instruction descriptor (InstrDescriptor) for
//     `tcgen05.mma.cta_group::1.kind::{f16,f8f6f4}`
//
// Derived from the DSA TopK-indexer project's umma_desc.h
// (https://github.com/ykirpichev/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64),
// with F16Format added for bf16/f16 inputs on kind::f16.

#pragma once

#include <cstdint>

namespace dsa_umma {

// ---------------------------------------------------------------------------
// Enums (verbatim from cute::UMMA)
// ---------------------------------------------------------------------------

enum class Major : uint8_t {
    K  = 0,
    MN = 1,
};

enum class ScaleIn : uint8_t {
    One = 0,
    Neg = 1,
};

enum class Saturate : uint8_t {
    False = 0,
    True  = 1,
};

// Swizzle encoding for SmemDescriptor.layout_type_.  Integer values follow
// the SM100 UMMA ISA encoding:
//   0 = no swizzle, 1 = 128B base 32B, 2 = 128B, 4 = 64B, 6 = 32B.
enum class LayoutType : uint8_t {
    SWIZZLE_NONE           = 0,
    SWIZZLE_128B_BASE32B   = 1,
    SWIZZLE_128B           = 2,
    SWIZZLE_64B            = 4,
    SWIZZLE_32B            = 6,
};

// Operand data type encoding for InstrDescriptor.a_format_ / b_format_.
// For `tcgen05.mma.kind::f8f6f4` (matches cute::UMMA::MXF8F6F4Format):
enum class F8F6F4Format : uint8_t {
    E4M3 = 0,
    E5M2 = 1,
    E2M3 = 3,
    E3M2 = 4,
    E2M1 = 5,
};

// For `tcgen05.mma.kind::f16` (matches cute::UMMA::MXF16Format):
//   FP16 = 0, BF16 = 1
// Both map into the same `a_format_` / `b_format_` bitfield of
// InstrDescriptor but the instruction kind encoded in the PTX mnemonic
// determines which enum applies.
enum class F16Format : uint8_t {
    FP16 = 0,
    BF16 = 1,
};

// Accumulator format encoding for InstrDescriptor.c_format_.
// Matches cute::UMMA::CFormat.
enum class CFormat : uint8_t {
    F16 = 0,
    F32 = 1,
    S32 = 2,
};

// ---------------------------------------------------------------------------
// SmemDescriptor (64-bit)
// ---------------------------------------------------------------------------
//
// Layout (ranges inclusive of `from`, exclusive of `to`; LSB == bit 0):
//   [ 0,14)  start_address       (14 bits, SMEM byte addr >> 4)
//   [14,16)  unused
//   [16,30)  leading_byte_offset (14 bits, byte stride >> 4)
//   [30,32)  unused
//   [32,46)  stride_byte_offset  (14 bits, byte stride >> 4)
//   [46,48)  version             (2 bits; SM100 UMMA uses value 1)
//   [48,49)  unused
//   [49,52)  base_offset         (3 bits)
//   [52,53)  lbo_mode            (1 bit; SM100 UMMA "legacy mode" uses 0)
//   [53,61)  unused
//   [61,64)  layout_type         (3 bits, LayoutType enum)

union SmemDescriptor {
    uint64_t desc_;

    struct {
        uint64_t start_address_       : 14;
        uint64_t /* pad */            : 2;
        uint64_t leading_byte_offset_ : 14;
        uint64_t /* pad */            : 2;
        uint64_t stride_byte_offset_  : 14;
        uint64_t version_             : 2;
        uint64_t /* pad */            : 1;
        uint64_t base_offset_         : 3;
        uint64_t lbo_mode_            : 1;
        uint64_t /* pad */            : 8;
        uint64_t layout_type_         : 3;
    };

    struct {
        uint32_t lo;
        uint32_t hi;
    };

    __host__ __device__ constexpr
    operator uint64_t() const noexcept { return desc_; }
};

static_assert(sizeof(SmemDescriptor) == 8, "SmemDescriptor must be 64-bit");

__device__ __forceinline__ SmemDescriptor
make_smem_desc(LayoutType layout,
               uint32_t   smem_uint_ptr,
               uint32_t   stride_byte_offset,
               uint32_t   leading_byte_offset) {
    SmemDescriptor d{};
    d.desc_                = 0;
    d.version_             = 1;
    d.lbo_mode_            = 0;
    d.base_offset_         = 0;
    d.layout_type_         = static_cast<uint8_t>(layout);
    d.start_address_       = static_cast<uint16_t>(smem_uint_ptr >> 4);
    d.stride_byte_offset_  = static_cast<uint16_t>(stride_byte_offset >> 4);
    d.leading_byte_offset_ = static_cast<uint16_t>(leading_byte_offset >> 4);
    return d;
}

__device__ __forceinline__ void
smem_desc_replace_addr(SmemDescriptor& d, uint32_t smem_uint_ptr) {
    d.start_address_ = static_cast<uint16_t>(smem_uint_ptr >> 4);
}

// ---------------------------------------------------------------------------
// InstrDescriptor (32-bit) for dense F8F6F4 / F16 UMMA
// ---------------------------------------------------------------------------
//
// Layout:
//   [ 0, 2) sparse_id2    (0 for dense)
//   [ 2, 3) sparse_flag   (0 = dense)
//   [ 3, 4) saturate
//   [ 4, 6) c_format      (CFormat enum)
//   [ 6, 7) unused
//   [ 7,10) a_format      (F8F6F4Format / F16Format enum)
//   [10,13) b_format      (F8F6F4Format / F16Format enum)
//   [13,14) a_negate
//   [14,15) b_negate
//   [15,16) a_major       (Major enum; 0 = K-major)
//   [16,17) b_major       (Major enum; 0 = K-major)
//   [17,23) n_dim         (N >> 3; range 1..32 -> N=8..256)
//   [23,24) unused
//   [24,29) m_dim         (M >> 4; 4 = M64, 8 = M128, 16 = M256)
//   [29,30) unused
//   [30,32) max_shift     (0 = no shift)

union InstrDescriptor {
    uint32_t desc_;

    struct {
        uint32_t sparse_id2_  : 2;
        uint32_t sparse_flag_ : 1;
        uint32_t saturate_    : 1;
        uint32_t c_format_    : 2;
        uint32_t /* pad */    : 1;
        uint32_t a_format_    : 3;
        uint32_t b_format_    : 3;
        uint32_t a_negate_    : 1;
        uint32_t b_negate_    : 1;
        uint32_t a_major_     : 1;
        uint32_t b_major_     : 1;
        uint32_t n_dim_       : 6;
        uint32_t /* pad */    : 1;
        uint32_t m_dim_       : 5;
        uint32_t /* pad */    : 1;
        uint32_t max_shift_   : 2;
    };

    __host__ __device__ constexpr explicit
    operator uint32_t() const noexcept { return desc_; }
};

static_assert(sizeof(InstrDescriptor) == 4, "InstrDescriptor must be 32-bit");

// Build an InstrDescriptor for `tcgen05.mma.cta_group::1.kind::f8f6f4`.
__host__ __device__ inline InstrDescriptor
make_instr_desc_f8f6f4(F8F6F4Format a_fmt,
                       F8F6F4Format b_fmt,
                       CFormat      c_fmt,
                       Major        a_major,
                       Major        b_major,
                       uint32_t     m,
                       uint32_t     n,
                       ScaleIn      a_neg = ScaleIn::One,
                       ScaleIn      b_neg = ScaleIn::One,
                       Saturate     sat   = Saturate::False) {
    InstrDescriptor d{};
    d.desc_         = 0;
    d.sparse_id2_   = 0;
    d.sparse_flag_  = 0;
    d.saturate_     = static_cast<uint8_t>(sat);
    d.c_format_     = static_cast<uint8_t>(c_fmt);
    d.a_format_     = static_cast<uint8_t>(a_fmt);
    d.b_format_     = static_cast<uint8_t>(b_fmt);
    d.a_negate_     = static_cast<uint8_t>(a_neg);
    d.b_negate_     = static_cast<uint8_t>(b_neg);
    d.a_major_      = static_cast<uint8_t>(a_major);
    d.b_major_      = static_cast<uint8_t>(b_major);
    d.n_dim_        = (n >> 3);
    d.m_dim_        = (m >> 4);
    d.max_shift_    = 0;
    return d;
}

// Build an InstrDescriptor for `tcgen05.mma.cta_group::1.kind::f16`.
// Same bitfield layout as f8f6f4, but `a_fmt` / `b_fmt` use F16Format
// (FP16 / BF16) instead of the F8F6F4 enum.
__host__ __device__ inline InstrDescriptor
make_instr_desc_f16(F16Format    a_fmt,
                    F16Format    b_fmt,
                    CFormat      c_fmt,
                    Major        a_major,
                    Major        b_major,
                    uint32_t     m,
                    uint32_t     n,
                    ScaleIn      a_neg = ScaleIn::One,
                    ScaleIn      b_neg = ScaleIn::One,
                    Saturate     sat   = Saturate::False) {
    InstrDescriptor d{};
    d.desc_         = 0;
    d.sparse_id2_   = 0;
    d.sparse_flag_  = 0;
    d.saturate_     = static_cast<uint8_t>(sat);
    d.c_format_     = static_cast<uint8_t>(c_fmt);
    d.a_format_     = static_cast<uint8_t>(a_fmt);
    d.b_format_     = static_cast<uint8_t>(b_fmt);
    d.a_negate_     = static_cast<uint8_t>(a_neg);
    d.b_negate_     = static_cast<uint8_t>(b_neg);
    d.a_major_      = static_cast<uint8_t>(a_major);
    d.b_major_      = static_cast<uint8_t>(b_major);
    d.n_dim_        = (n >> 3);
    d.m_dim_        = (m >> 4);
    d.max_shift_    = 0;
    return d;
}

} // namespace dsa_umma
