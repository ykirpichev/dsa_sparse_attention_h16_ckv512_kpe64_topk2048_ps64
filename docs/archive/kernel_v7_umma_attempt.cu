/*
 * DSA sparse MLA — B200 / sm_100a (Blackwell) kernel.
 *
 * Stage-7 (v7): fused-heads + Blackwell tensor-core logit AND value passes.
 *
 *   Grid  : (T,)       — one CTA per token, covering all H=16 heads.
 *   Block : 256 threads (8 warps).
 *
 * Logit pass: tcgen05.mma.kind::f16 UMMA (bf16 inputs, fp32 accum) with
 * MMA_M=64, N=BK=16. K-atom = 16 bf16 → Q_nope·K_ckv unrolls to 32 K-iters,
 * Q_pe·K_kpe unrolls to 4 K-iters (all accumulating into the same TMEM).
 *
 * Value pass: tcgen05.mma.kind::f16 UMMA with MMA_M=64, K=BK=16 (single K-atom),
 * N=D_CK=512 split into 2 halves (N=256 each). Inputs are staged into SMEM
 * with SWIZZLE_NONE K-major chunk-interleaved layout (see below):
 *   P    [M=64, K=16] bf16    — softmax output, padded heads are zero.
 *   V^T  [N=512, K=16] bf16   — V transposed per tile from the swizzled
 *                               K_ckv buffer.
 * The 2 N-half MMAs write into adjacent TMEM cols [0,256) and [256,512).
 * After commit/wait, 8 warps cooperatively read 64 cols each back into an
 * SMEM bridge `s_pv [16, D_CK] fp32`, and every thread then folds its
 * owned dims into its per-head register accumulator O[h][d] via the
 * online-softmax rescale+add update.
 *
 * SMEM layout (SWIZZLE_128B for logit operands, SWIZZLE_NONE for value):
 *   s_q_n   [MMA_M=64, D_CK=512] bf16 = 64 KiB  (rows 16..63 zero-padded)
 *   s_q_p   [MMA_M=64, D_KP=64 ] bf16 =  8 KiB
 *   s_ckv   [V_PIPE=2, BK=16, D_CK] bf16 = 32 KiB
 *   s_kpe   [V_PIPE=2, BK=16, D_KP] bf16 =  4 KiB
 *   s_p_umma [MMA_M=64, BK=16] bf16 (chunk-interleaved K-major) = 2 KiB
 *   s_v_t    [D_CK=512, BK=16] bf16 (chunk-interleaved K-major) = 16 KiB
 *   s_pv     [HEADS=16, D_CK=512] fp32                         = 32 KiB
 *   s_gidx  [TOPK=2048] int         =    8 KiB
 *   s_m/l[HEADS] fp32 + s_reduce[NUM_WARPS] fp32 + misc  ≲  1 KiB
 *   mbarriers + tmem ptr                              ≲    32 B
 * Total ≈ 168 KiB (within the 228 KiB sm_100 dynamic SMEM cap).
 *
 * SWIZZLE_128B physical layout for a [M, K_bf16] operand:
 *   - Divide into (M/8) × (K_bf16/64) swizzle-blocks of 1024 B each
 *     (= 8 rows × 128 B, one "swizzle-block-unit"; canonical for B128).
 *   - Storage order: [m_group][k_block][swizzled 8x8 bf162 block].
 *   - M-group stride (byte) = num_k_blocks_per_row * 1024.
 *   - In-block swizzle: within one 1024 B block, 64 chunks of 16 B are
 *     permuted by `phys_chunk = (r*8 + c) XOR r` for r=row_in_group,
 *     c=chunk_in_block. This matches the SM100a SWIZZLE_128B pattern
 *     used by cute::Swizzle<3,4,3>.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <mutex>

#include "tcgen05_ptx.h"
#include "umma_desc.h"

namespace {

// Definition constants (baked into the submission name).
constexpr int D_CK = 512;
constexpr int D_KP = 64;
constexpr int TOPK = 2048;
constexpr int HEADS = 16;

// Launch and tiling.
constexpr int THREADS = 256;
constexpr int WARP = 32;
constexpr int NUM_WARPS = THREADS / WARP;  // 8
constexpr int BK = 16;
constexpr int V_PIPE = 2;
static_assert(TOPK % BK == 0, "TOPK must be divisible by BK");
constexpr int NUM_TILES = TOPK / BK;       // 128

// UMMA logit shapes.  MMA_M=64 ∈ {64,128,256} (only 16 heads used, rows
// 16..63 are zero-padded in SMEM to make the UMMA legal).
constexpr int MMA_M = 64;
constexpr int MMA_N = BK;                  // 16, fits N ∈ 8..256 step 8
constexpr int K_ATOM_BF16 = 16;            // kind::f16 K-atom = 16 bf16
constexpr int K_ATOM_BYTES = K_ATOM_BF16 * sizeof(__nv_bfloat16);  // 32 B
constexpr int K_ITERS_NOPE = D_CK / K_ATOM_BF16;  // 32
constexpr int K_ITERS_PE   = D_KP / K_ATOM_BF16;  //  4

// SWIZZLE_128B block geometry.  One block = 8 rows × 128 B = 1024 B.
constexpr int SWZ_BLOCK_BYTES      = 1024;
constexpr int SWZ_BLOCK_CHUNKS_16B = SWZ_BLOCK_BYTES / 16;  // 64
constexpr int SWZ_ROWS_PER_BLOCK   = 8;
constexpr int BYTES_PER_SWZ_ROW    = 128;
// Number of K-blocks per row for Q/K_ckv (D_CK = 8 * 64 bf16 blocks/row).
constexpr int NOPE_K_BLOCKS = D_CK * 2 / BYTES_PER_SWZ_ROW;  // 8
constexpr int PE_K_BLOCKS   = D_KP * 2 / BYTES_PER_SWZ_ROW;  // 1
static_assert(NOPE_K_BLOCKS * BYTES_PER_SWZ_ROW == D_CK * 2, "D_CK row bytes must be multiple of 128B");
static_assert(PE_K_BLOCKS   * BYTES_PER_SWZ_ROW == D_KP * 2, "D_KP row bytes must be multiple of 128B");
// M-group strides (byte): #K-blocks-per-row * 1024 B.
constexpr uint32_t SBO_NOPE_BYTES = NOPE_K_BLOCKS * SWZ_BLOCK_BYTES;  // 8192
constexpr uint32_t SBO_PE_BYTES   = PE_K_BLOCKS   * SWZ_BLOCK_BYTES;  // 1024
constexpr uint32_t LBO_BYTES      = 16;  // stride between 2 K-chunks of a K-atom

// Value-pass UMMA partitioning.  P·V = O_tile, with K=BK=16 (1 K-atom),
// N=D_CK split into VALUE_N_HALF each = 256 (max for kind::f16).
constexpr int VALUE_N_HALF = 256;                         // max 256
constexpr int VALUE_N_HALVES = D_CK / VALUE_N_HALF;       // 2
static_assert(VALUE_N_HALVES * VALUE_N_HALF == D_CK, "D_CK must split into N halves");

// SWIZZLE_NONE K-major "slice-major" layout per NVIDIA PTX canonical
// (see tcgen05 canonical table `((8,m),(16B,2k)):((1x16B,SBO),(1,LBO))`).
//
//   offset_bf16(m, k, M) = (m/8)*64  + (k/8)*(M*8) + (m%8)*8 + (k%8)
//
// i.e.  slice `k/8` starts at offset `(k/8)*M*8` bf16 = `(k/8)*M*16` B.
// Within a slice, the 8x16B tiles are packed as m/8 → +128 B (= +64 bf16)
// each, so the full slice-major layout has SBO = 128 B constant and
// LBO = M_total*16 B (depends on the operand's real M dimension).
constexpr uint32_t SWZ0_SBO_BYTES       = 128;           // const for NONE
constexpr uint32_t SWZ0_LBO_P_BYTES     = MMA_M * 16;    // 1024 (P: M=64)
constexpr uint32_t SWZ0_LBO_V_BYTES     = D_CK  * 16;    // 8192 (V^T: M=512)
constexpr int VT_SIZE_BF16        = D_CK * BK;  // 8192
constexpr int P_UMMA_SIZE_BF16    = MMA_M * BK; // 1024

// Value-pass output partitioning in the per-thread O register accumulator.
static_assert(D_CK % THREADS == 0, "D_CK must divide THREADS");
constexpr int DIMS_PER_THREAD = D_CK / THREADS;  // 2
constexpr float LOG2E_F = 1.4426950408889634f;

// TMEM allocation: 512 cols (shared between logit and value outputs; value
// MMA writes accum=0 which overwrites the logit cols 0..15).
constexpr uint32_t TMEM_NUM_COLS = 512;

// ----------------------------------------------------------------------------
// Swizzled SMEM addressing helpers (SWIZZLE_128B, K-major, bf16).
//
// A canonical [M, K_bf16] operand lives in SMEM as (M/8) × (K_bf16/64)
// swizzle-blocks. Each block is stored in [m_group][k_block] row-major,
// so the offset of the first chunk of (m_group, k_block) is
//   m_group_offset_bytes = m_group * #K_blocks_per_row * 1024
//   k_block_offset_bytes = k_block * 1024
// Within a 1024 B block, the 64 chunks are permuted by XOR:
//   logical_chunk_in_block = row_in_group * 8 + chunk_in_block
//   physical_chunk_in_block = logical XOR row_in_group
// ----------------------------------------------------------------------------

// Compute the swizzled 16-byte chunk offset (from the operand's SMEM
// base) for element (m, k_chunk) where m ∈ [0, M), k_chunk ∈
// [0, K_bf16/8) i.e. 0..63 for D_CK.
__device__ __forceinline__ int
swz128b_chunk16b_offset(int m, int k_chunk, int k_blocks_per_row) {
  const int m_group = m >> 3;        // m / 8
  const int row_in_group = m & 7;    // m % 8
  const int k_block = k_chunk >> 3;  // chunk / 8
  const int chunk_in_block = k_chunk & 7;
  const int logical_in_block = row_in_group * 8 + chunk_in_block;
  const int phys_in_block = logical_in_block ^ row_in_group;
  return m_group * k_blocks_per_row * SWZ_BLOCK_CHUNKS_16B  // M-group offset
       + k_block * SWZ_BLOCK_CHUNKS_16B                     // K-block offset
       + phys_in_block;                                     // in-block swizzle
}

// 8-byte variant (for D_KP loads — kpe rows are 128 B = 16 × 8 B).
__device__ __forceinline__ int
swz128b_chunk8b_offset(int m, int k_chunk8, int k_blocks_per_row) {
  // Two 8B chunks form one 16B chunk, so convert via k_chunk = k_chunk8 / 2.
  const int k_chunk16 = k_chunk8 >> 1;
  const int in_16 = k_chunk8 & 1;  // 0 = low 8B, 1 = high 8B
  return 2 * swz128b_chunk16b_offset(m, k_chunk16, k_blocks_per_row) + in_16;
}

// Byte offset of the 16-bf16 K-atom `k` (k ∈ [0, K/16)) for row m=0:
// this is the `start_address` used for the per-K-iter MMA.
//   k_block = k / 4,  k_atom_in_block = k % 4
//   offset = k_block * 1024 + k_atom_in_block * 32
__device__ __forceinline__ int
swz128b_katom_base_bytes(int k, int k_blocks_per_row) {
  const int k_block = k >> 2;          // k / 4 (4 K-atoms per 128B K-block)
  const int k_atom_in_block = k & 3;   // k % 4
  (void)k_blocks_per_row;
  return k_block * SWZ_BLOCK_BYTES + k_atom_in_block * K_ATOM_BYTES;
}

// ----------------------------------------------------------------------------
// Warp-level reductions (same as v4).
// ----------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int off = WARP / 2; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int off = WARP / 2; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
  v = warp_reduce_max(v);
  const int lane = threadIdx.x & (WARP - 1);
  const int wid  = threadIdx.x / WARP;
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  float r = -INFINITY;
  if (wid == 0) {
    float x = (lane < NUM_WARPS) ? smem[lane] : -INFINITY;
    x = warp_reduce_max(x);
    if (lane == 0) smem[0] = x;
  }
  __syncthreads();
  r = smem[0];
  __syncthreads();
  return r;
}

// ----------------------------------------------------------------------------
// SmemDescriptor builder (SWIZZLE_128B, K-major).
// sbo_bytes = stride between M-groups of 8 rows.
// ----------------------------------------------------------------------------

__device__ __forceinline__ dsa_umma::SmemDescriptor
make_kmajor_desc_b128(const void* smem_base, uint32_t sbo_bytes) {
  dsa_umma::SmemDescriptor d{};
  d.desc_                = 0;
  d.version_             = 1;
  d.lbo_mode_            = 0;
  d.base_offset_         = 0;
  d.layout_type_         = static_cast<uint8_t>(dsa_umma::LayoutType::SWIZZLE_128B);
  d.leading_byte_offset_ = static_cast<uint16_t>(LBO_BYTES >> 4);      // = 1
  d.stride_byte_offset_  = static_cast<uint16_t>(sbo_bytes >> 4);
  d.start_address_       = static_cast<uint16_t>(
      dsa_ptx::smem_ptr_to_uint(smem_base) >> 4);
  return d;
}

// SWIZZLE_NONE K-major "slice-major" descriptor builder for value pass.
// `lbo_bytes` depends on the operand's real M dimension (see canonical
// layout in tcgen05 PTX docs).
__device__ __forceinline__ dsa_umma::SmemDescriptor
make_kmajor_desc_noswz(const void* smem_base, uint32_t lbo_bytes) {
  dsa_umma::SmemDescriptor d{};
  d.desc_                = 0;
  d.version_             = 1;
  d.lbo_mode_            = 0;
  d.base_offset_         = 0;
  d.layout_type_         = static_cast<uint8_t>(dsa_umma::LayoutType::SWIZZLE_NONE);
  d.leading_byte_offset_ = static_cast<uint16_t>(lbo_bytes >> 4);
  d.stride_byte_offset_  = static_cast<uint16_t>(SWZ0_SBO_BYTES >> 4);  // 128 B / 16 = 8
  d.start_address_       = static_cast<uint16_t>(
      dsa_ptx::smem_ptr_to_uint(smem_base) >> 4);
  return d;
}

// Slice-major bf16 offset for SWIZZLE_NONE K-major:
//   offset(m, k) = (m/8)*64 + (k/8)*(M*8) + (m%8)*8 + (k%8).
// `M_total` is the full M extent of the operand's SMEM buffer (64 for
// s_p_umma, 512 for s_v_t).
__device__ __forceinline__ int
swz0_offset_bf16(int m, int k, int M_total) {
  return (m >> 3) * 64 + (k >> 3) * (M_total * 8) + (m & 7) * 8 + (k & 7);
}

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------

__global__ __launch_bounds__(THREADS, 1) void mla_sparse_attn_umma_kernel(
    const __nv_bfloat16* __restrict__ q_nope,    // [T, H, D_CK]
    const __nv_bfloat16* __restrict__ q_pe,      // [T, H, D_KP]
    const __nv_bfloat16* __restrict__ ckv_flat,  // [N_rows, D_CK]
    const __nv_bfloat16* __restrict__ kpe_flat,  // [N_rows, D_KP]
    const int32_t* __restrict__ sparse_idx,      // [T, TOPK]
    float sm_scale_log2e,
    __nv_bfloat16* __restrict__ out,             // [T, H, D_CK]
    float* __restrict__ lse_out,                 // [T, H]
    int64_t num_kv_rows) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane    = tid & (WARP - 1);

  const int64_t q_row_base = static_cast<int64_t>(t) * HEADS;
  const int64_t si_off     = static_cast<int64_t>(t) * TOPK;

  // -------------------- SMEM ------------------------------------------------
  extern __shared__ __align__(16) unsigned char raw[];
  __nv_bfloat16* s_q_n    = reinterpret_cast<__nv_bfloat16*>(raw);
  __nv_bfloat16* s_q_p    = s_q_n + MMA_M * D_CK;
  __nv_bfloat16* s_ckv    = s_q_p + MMA_M * D_KP;
  __nv_bfloat16* s_kpe    = s_ckv + V_PIPE * BK * D_CK;
  __nv_bfloat16* s_p_umma = s_kpe + V_PIPE * BK * D_KP;
  __nv_bfloat16* s_v_t    = s_p_umma + P_UMMA_SIZE_BF16;
  // s_pv lives as fp32 after the bf16 buffers above. Align to 4B.
  uintptr_t s_pv_raw = reinterpret_cast<uintptr_t>(s_v_t + VT_SIZE_BF16);
  s_pv_raw = (s_pv_raw + 3u) & ~uintptr_t{3u};
  float* s_pv = reinterpret_cast<float*>(s_pv_raw);
  int*    s_gidx   = reinterpret_cast<int*>(s_pv + HEADS * D_CK);
  float*  s_m      = reinterpret_cast<float*>(s_gidx + TOPK);
  float*  s_l      = s_m + HEADS;
  float*  s_scale  = s_l + HEADS;
  float*  s_reduce = s_scale + HEADS;
  // 8B-align the mbarrier region.
  uintptr_t mbar_raw = reinterpret_cast<uintptr_t>(s_reduce + NUM_WARPS);
  mbar_raw = (mbar_raw + 7u) & ~uintptr_t{7u};
  uint64_t* s_mbar      = reinterpret_cast<uint64_t*>(mbar_raw);
  uint32_t* s_tmem_ptr  = reinterpret_cast<uint32_t*>(s_mbar + 1);

  const uint32_t s_q_n_addr = dsa_ptx::smem_ptr_to_uint(s_q_n);
  const uint32_t s_q_p_addr = dsa_ptx::smem_ptr_to_uint(s_q_p);

  // -------------------- Phase 0: zero-pad + init state ----------------------
  // Zero rows 16..63 of s_q_n / s_q_p (bf16) — they are padding for the
  // MMA_M=64 UMMA. We zero the whole bf16 buffer first; real rows are then
  // overwritten during Q load. Also zero the entire s_p_umma buffer so that
  // the padded rows 16..63 stay zero across tiles (real heads are rewritten
  // each tile by the softmax stage below).
  #pragma unroll 1
  for (int i = tid; i < MMA_M * D_CK; i += THREADS) {
    s_q_n[i] = __float2bfloat16(0.f);
  }
  #pragma unroll 1
  for (int i = tid; i < MMA_M * D_KP; i += THREADS) {
    s_q_p[i] = __float2bfloat16(0.f);
  }
  #pragma unroll 1
  for (int i = tid; i < P_UMMA_SIZE_BF16; i += THREADS) {
    s_p_umma[i] = __float2bfloat16(0.f);
  }
  if (tid < HEADS) {
    s_m[tid] = -INFINITY;
    s_l[tid] = 0.f;
  }
  __syncthreads();

  // -------------------- Phase 1: load Q / sparse_idx ------------------------
  // Store Q_nope[h, d] into swizzled s_q_n. Only real heads h ∈ [0, HEADS).
  // Each thread stores a few bf162 (= 4B) pairs. We do a scalar-ish store
  // for simplicity (swizzle math per element); this is a one-off cost.
  {
    constexpr int Q_NOPE_BF162 = HEADS * (D_CK / 2);  // 16 * 256 = 4096
    const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(
        q_nope + q_row_base * D_CK);
    __nv_bfloat162* q2_smem = reinterpret_cast<__nv_bfloat162*>(s_q_n);
    #pragma unroll 1
    for (int i = tid; i < Q_NOPE_BF162; i += THREADS) {
      const int h = i / (D_CK / 2);
      const int d2 = i - h * (D_CK / 2);        // bf162 index [0, D_CK/2)
      const int d = d2 * 2;                      // bf16 index
      const int k_chunk16 = d >> 3;              // d/8 → 16B chunk in row
      const int chunk_off = swz128b_chunk16b_offset(h, k_chunk16, NOPE_K_BLOCKS);
      const int bf162_off = chunk_off * 4 + (d & 7) / 2;  // 4 bf162 per 16B chunk
      q2_smem[bf162_off] = q2[h * (D_CK / 2) + d2];
    }
  }
  {
    constexpr int Q_PE_BF162 = HEADS * (D_KP / 2);    // 16 * 32 = 512
    const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(
        q_pe + q_row_base * D_KP);
    __nv_bfloat162* p2_smem = reinterpret_cast<__nv_bfloat162*>(s_q_p);
    #pragma unroll 1
    for (int i = tid; i < Q_PE_BF162; i += THREADS) {
      const int h = i / (D_KP / 2);
      const int d2 = i - h * (D_KP / 2);
      const int d = d2 * 2;
      const int k_chunk16 = d >> 3;
      const int chunk_off = swz128b_chunk16b_offset(h, k_chunk16, PE_K_BLOCKS);
      const int bf162_off = chunk_off * 4 + (d & 7) / 2;
      p2_smem[bf162_off] = p2[h * (D_KP / 2) + d2];
    }
  }

  // sparse_idx: validate & cache.  Also track max valid k+1 for tile truncation.
  int local_max_valid_k_p1 = 0;
  for (int k = tid; k < TOPK; k += THREADS) {
    const int32_t g = sparse_idx[si_off + k];
    const bool valid = (g >= 0 && static_cast<int64_t>(g) < num_kv_rows);
    s_gidx[k] = valid ? g : -1;
    if (valid && (k + 1) > local_max_valid_k_p1) local_max_valid_k_p1 = k + 1;
  }

  // -------------------- Phase 2: TMEM alloc + mbarrier ----------------------
  if (warp_id == 0) {
    dsa_ptx::tcgen05_alloc_1sm(dsa_ptx::smem_ptr_to_uint(s_tmem_ptr), TMEM_NUM_COLS);
    dsa_ptx::tcgen05_relinquish_alloc_permit_1sm();
    if (lane == 0) {
      dsa_ptx::mbarrier_init(dsa_ptx::smem_ptr_to_uint(s_mbar), 1);
    }
    dsa_ptx::fence_barrier_init();
  }
  __syncthreads();
  const uint32_t tmem_logit = *s_tmem_ptr;  // base of TMEM alloc, used for both
  const uint32_t tmem_pv    = tmem_logit;   // logit (cols 0..15) and value (cols 0..511)

  // Truncate tile loop at the last valid row.
  const float reduced_max = block_reduce_max(
      static_cast<float>(local_max_valid_k_p1), s_reduce);
  const int n_valid_rows = static_cast<int>(reduced_max);
  int n_valid_tiles = (n_valid_rows + BK - 1) / BK;
  if (n_valid_tiles > NUM_TILES) n_valid_tiles = NUM_TILES;

  // -------------------- Per-thread output registers ------------------------
  float O[HEADS][DIMS_PER_THREAD];
  #pragma unroll
  for (int h = 0; h < HEADS; ++h) {
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; ++d) O[h][d] = 0.f;
  }
  const int d0 = tid * DIMS_PER_THREAD;  // output dim ownership

  // -------------------- K/V tile cp.async issuer ---------------------------
  // Stores ckv[BK rows, D_CK bf16] and kpe[BK rows, D_KP bf16] into
  // SWIZZLE_128B-layout SMEM buffers `s_ckv[buf]` and `s_kpe[buf]`.
  //
  // One 16B chunk = 8 bf16 = 1/8 of a bf16 row (D_CK) or 1/4 of D_KP (since
  // D_KP=64 bf16 = 128 B = 8 × 16B).  Per tile:
  //   ckv: BK * D_CK / 8 = 16 * 64 = 1024 × 16B chunks → 4 / thread @ 256 T.
  //   kpe: BK * D_KP / 8 = 16 * 8  = 128  × 16B chunks → 1 every 2 threads.
  // For kpe we instead issue 8B copies (128 × 16B = 256 × 8B → 1/thread).
  auto issue_kv_tile = [&](int tile, int buf) {
    const int k0 = tile * BK;
    __nv_bfloat16* ckv_buf = s_ckv + buf * (BK * D_CK);
    __nv_bfloat16* kpe_buf = s_kpe + buf * (BK * D_KP);
    const uint32_t ckv_buf_addr = dsa_ptx::smem_ptr_to_uint(ckv_buf);
    const uint32_t kpe_buf_addr = dsa_ptx::smem_ptr_to_uint(kpe_buf);

    // ckv tile: BK*D_CK = 16*512 bf16 = 16 KiB = 1024 × 16B. 4 per thread @ 256T.
    constexpr int CKV_CHUNKS_16B = BK * D_CK / 8;
    constexpr int CKV_CHUNKS_PER_THREAD = CKV_CHUNKS_16B / THREADS;
    #pragma unroll
    for (int c = 0; c < CKV_CHUNKS_PER_THREAD; ++c) {
      const int chunk = tid + c * THREADS;
      const int r = chunk / (D_CK / 8);          // row in tile
      const int kc16 = chunk - r * (D_CK / 8);   // 16B chunk in row
      int g = s_gidx[k0 + r];
      if (g < 0) g = 0;
      const uint4* src = reinterpret_cast<const uint4*>(
          ckv_flat + static_cast<int64_t>(g) * D_CK) + kc16;
      const int dst_chunk = swz128b_chunk16b_offset(r, kc16, NOPE_K_BLOCKS);
      dsa_ptx::cp_async_16B(ckv_buf_addr + dst_chunk * 16u, src);
    }
    // kpe tile: BK*D_KP = 16*64 bf16 = 2 KiB = 256 × 8B. 1 per thread @ 256T.
    constexpr int KPE_CHUNKS_8B = BK * D_KP * 2 / 8;
    constexpr int KPE_CHUNKS_PER_THREAD = KPE_CHUNKS_8B / THREADS;
    #pragma unroll
    for (int c = 0; c < KPE_CHUNKS_PER_THREAD; ++c) {
      const int chunk = tid + c * THREADS;
      const int r = chunk / (D_KP / 4);          // row
      const int kc8 = chunk - r * (D_KP / 4);    //  8B chunk in row
      int g = s_gidx[k0 + r];
      if (g < 0) g = 0;
      const uint2* src = reinterpret_cast<const uint2*>(
          kpe_flat + static_cast<int64_t>(g) * D_KP) + kc8;
      const int dst_chunk8 = swz128b_chunk8b_offset(r, kc8, PE_K_BLOCKS);
      dsa_ptx::cp_async_8B(kpe_buf_addr + dst_chunk8 * 8u, src);
    }
  };

  // -------------------- Wait for Q loads to finish --------------------------
  // The Q stores above were regular shared stores, not cp.async, so
  // __syncthreads() suffices.
  __syncthreads();

  // Prime tile 0.
  if (n_valid_tiles > 0) {
    issue_kv_tile(0, 0);
    dsa_ptx::cp_async_commit_group();
  }

  uint32_t mbar_phase = 0;

  // -------------------- Main tile loop --------------------------------------
  const auto instr_logit =
      dsa_umma::make_instr_desc_f16(dsa_umma::F16Format::BF16,
                                    dsa_umma::F16Format::BF16,
                                    dsa_umma::CFormat::F32,
                                    dsa_umma::Major::K, dsa_umma::Major::K,
                                    MMA_M, MMA_N);
  const auto instr_value =
      dsa_umma::make_instr_desc_f16(dsa_umma::F16Format::BF16,
                                    dsa_umma::F16Format::BF16,
                                    dsa_umma::CFormat::F32,
                                    dsa_umma::Major::K, dsa_umma::Major::K,
                                    MMA_M, VALUE_N_HALF);

  for (int tile = 0; tile < n_valid_tiles; ++tile) {
    const int buf = tile & 1;
    const int next_tile = tile + 1;

    if (next_tile < n_valid_tiles) {
      issue_kv_tile(next_tile, next_tile & 1);
      dsa_ptx::cp_async_commit_group();
      dsa_ptx::cp_async_wait_group<1>();
    } else {
      dsa_ptx::cp_async_wait_group<0>();
    }
    __syncthreads();

    const __nv_bfloat16* ckv_buf = s_ckv + buf * (BK * D_CK);
    const __nv_bfloat16* kpe_buf = s_kpe + buf * (BK * D_KP);

    // --- Logit UMMA: Q_nope·K_ckv^T (32 K-iters) + Q_pe·K_kpe^T (4 K-iters, accum).
    if (warp_id == 0) {
      dsa_ptx::tcgen05_fence_after_thread_sync();
      if (lane == 0) {
        auto a_base_n = make_kmajor_desc_b128(s_q_n,    SBO_NOPE_BYTES);
        auto b_base_n = make_kmajor_desc_b128(ckv_buf,  SBO_NOPE_BYTES);
        auto a_base_p = make_kmajor_desc_b128(s_q_p,    SBO_PE_BYTES);
        auto b_base_p = make_kmajor_desc_b128(kpe_buf,  SBO_PE_BYTES);

        // Q_nope · K_ckv^T
        #pragma unroll
        for (int ki = 0; ki < K_ITERS_NOPE; ++ki) {
          const uint32_t katom_off =
              static_cast<uint32_t>(swz128b_katom_base_bytes(ki, NOPE_K_BLOCKS));
          dsa_umma::SmemDescriptor a = a_base_n;
          dsa_umma::SmemDescriptor b = b_base_n;
          a.start_address_ = static_cast<uint16_t>(a.start_address_ + (katom_off >> 4));
          b.start_address_ = static_cast<uint16_t>(b.start_address_ + (katom_off >> 4));
          dsa_ptx::tcgen05_mma_f16_ss(
              tmem_logit, a, b, instr_logit,
              /*accum=*/(ki == 0) ? 0u : 1u);
        }
        // Q_pe · K_kpe^T (accumulating into the same TMEM)
        #pragma unroll
        for (int ki = 0; ki < K_ITERS_PE; ++ki) {
          const uint32_t katom_off =
              static_cast<uint32_t>(swz128b_katom_base_bytes(ki, PE_K_BLOCKS));
          dsa_umma::SmemDescriptor a = a_base_p;
          dsa_umma::SmemDescriptor b = b_base_p;
          a.start_address_ = static_cast<uint16_t>(a.start_address_ + (katom_off >> 4));
          b.start_address_ = static_cast<uint16_t>(b.start_address_ + (katom_off >> 4));
          dsa_ptx::tcgen05_mma_f16_ss(
              tmem_logit, a, b, instr_logit, /*accum=*/1u);
        }
        dsa_ptx::tcgen05_commit_1sm(dsa_ptx::smem_ptr_to_uint(s_mbar));
      }
    }
    dsa_ptx::mbarrier_wait_parity(dsa_ptx::smem_ptr_to_uint(s_mbar), mbar_phase);
    mbar_phase ^= 1u;
    dsa_ptx::tcgen05_fence_after_thread_sync();

    // --- Load logit TMEM → regs (warp 0 only, lanes 0..15 each own one row/head).
    float logits[BK];
    if (warp_id == 0) {
      uint32_t regs[BK];
      dsa_ptx::tcgen05_ld_32x32b_x16_b32(tmem_logit, regs);
      dsa_ptx::tcgen05_wait_ld();
      #pragma unroll
      for (int r = 0; r < BK; ++r) logits[r] = __uint_as_float(regs[r]);
    }

    // --- Online softmax (one lane per real head, lane < HEADS = 16).
    // Writes P directly into s_p_umma in chunk-interleaved SWIZZLE_NONE
    // K-major layout so the value UMMA can consume it as A operand.
    if (warp_id == 0 && lane < HEADS) {
      const int h = lane;
      const int k0 = tile * BK;

      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        const int g = s_gidx[k0 + r];
        logits[r] = (g >= 0) ? (logits[r] * sm_scale_log2e) : -INFINITY;
      }

      const float m_old = s_m[h];
      float new_max = m_old;
      #pragma unroll
      for (int r = 0; r < BK; ++r) new_max = fmaxf(new_max, logits[r]);

      const float scale = (m_old == -INFINITY) ? 1.f : exp2f(m_old - new_max);
      float sum_p = 0.f;
      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        const float p = (new_max == -INFINITY) ? 0.f : exp2f(logits[r] - new_max);
        logits[r] = p;
        sum_p += p;
      }
      const float l_new = s_l[h] * scale + sum_p;
      s_scale[h] = scale;
      s_m[h]     = new_max;
      s_l[h]     = l_new;

      // Pack P[h, 0..15] into s_p_umma at slice-major bf16 offsets (M=MMA_M=64).
      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        s_p_umma[swz0_offset_bf16(h, r, MMA_M)] = __float2bfloat16_rn(logits[r]);
      }
    }

    // --- Transpose V from s_ckv (SWIZZLE_128B K-major) into s_v_t
    // (SWIZZLE_NONE K-major, slice-major, M_total=D_CK). All threads participate.
    // V[bk, d] = ckv_buf[swz128b(bk, d)], V^T[d, bk] = s_v_t[swz0(d, bk, D_CK)].
    // Per-thread: TOTAL/THREADS = 8192/256 = 32 bf16 stores/tile.
    {
      constexpr int TOTAL   = D_CK * BK;
      constexpr int PER_TH  = TOTAL / THREADS;
      #pragma unroll
      for (int i = 0; i < PER_TH; ++i) {
        const int idx = tid + i * THREADS;
        // (bk outer, d inner): idx = bk * D_CK + d.
        const int bk = idx >> 9;           // idx / 512
        const int d  = idx & (D_CK - 1);   // idx % 512
        const int src_chunk16 = swz128b_chunk16b_offset(bk, d >> 3, NOPE_K_BLOCKS);
        const __nv_bfloat16 v = ckv_buf[src_chunk16 * 8 + (d & 7)];
        s_v_t[swz0_offset_bf16(d, bk, D_CK)] = v;
      }
    }
    __syncthreads();

    // --- Value UMMA: tmem_pv = P · V^T, in 2 N=256 halves.
    if (warp_id == 0) {
      dsa_ptx::tcgen05_fence_after_thread_sync();
      if (lane == 0) {
        auto a_base = make_kmajor_desc_noswz(s_p_umma, SWZ0_LBO_P_BYTES);
        auto b_base = make_kmajor_desc_noswz(s_v_t,    SWZ0_LBO_V_BYTES);

        #pragma unroll
        for (int nh = 0; nh < VALUE_N_HALVES; ++nh) {
          dsa_umma::SmemDescriptor a = a_base;
          dsa_umma::SmemDescriptor b = b_base;
          // 2nd N-half: shift B to view m=256..511 of s_v_t. With SBO=128 B
          // constant, m_outer=32 within slice 0 is at 32*128 = 4096 B.
          if (nh == 1) {
            const uint32_t nh_off_bytes =
                static_cast<uint32_t>(VALUE_N_HALF) / 8u * SWZ0_SBO_BYTES;  // 32 * 128 = 4096
            b.start_address_ = static_cast<uint16_t>(
                b.start_address_ + (nh_off_bytes >> 4));
          }
          const uint32_t tmem_target = tmem_pv + static_cast<uint32_t>(nh * VALUE_N_HALF);
          dsa_ptx::tcgen05_mma_f16_ss(tmem_target, a, b, instr_value, /*accum=*/0u);
        }
        dsa_ptx::tcgen05_commit_1sm(dsa_ptx::smem_ptr_to_uint(s_mbar));
      }
    }
    dsa_ptx::mbarrier_wait_parity(dsa_ptx::smem_ptr_to_uint(s_mbar), mbar_phase);
    mbar_phase ^= 1u;
    dsa_ptx::tcgen05_fence_after_thread_sync();

    // --- Readback tmem_pv → s_pv [HEADS, D_CK] fp32 cooperatively.
    // 8 warps × (32x32b.x64) = 8 × 64 = 512 cols. Each warp handles 64 cols.
    // Lane l reads TMEM row l (= head l); only lanes 0..15 are real heads.
    {
      const int col_base = warp_id * 64;
      uint32_t pv_regs[64];
      dsa_ptx::tcgen05_ld_32x32b_x64_b32(tmem_pv + static_cast<uint32_t>(col_base),
                                         pv_regs);
      dsa_ptx::tcgen05_wait_ld();
      if (lane < HEADS) {
        float* dst = &s_pv[lane * D_CK + col_base];
        #pragma unroll
        for (int c = 0; c < 64; ++c) {
          dst[c] = __uint_as_float(pv_regs[c]);
        }
      }
    }
    __syncthreads();

    // --- Online-softmax update: O = scale * O_old + P·V for the owned dims.
    #pragma unroll
    for (int h = 0; h < HEADS; ++h) {
      const float s = s_scale[h];
      const float* pv_row = &s_pv[h * D_CK];
      #pragma unroll
      for (int d = 0; d < DIMS_PER_THREAD; ++d) {
        O[h][d] = s * O[h][d] + pv_row[d0 + d];
      }
    }
    __syncthreads();
  }

  // -------------------- Dealloc TMEM ----------------------------------------
  if (warp_id == 0) {
    dsa_ptx::tcgen05_fence_before_thread_sync();
    dsa_ptx::tcgen05_dealloc_1sm(tmem_logit, TMEM_NUM_COLS);
  }

  // -------------------- Epilogue --------------------------------------------
  if (tid < HEADS) {
    const int h = tid;
    const float m_h = s_m[h];
    const float l_h = s_l[h];
    lse_out[q_row_base + h] = (m_h == -INFINITY) ? -INFINITY : (m_h + log2f(l_h));
  }

  #pragma unroll
  for (int h = 0; h < HEADS; ++h) {
    const float l_h = s_l[h];
    const float inv = (l_h > 0.f) ? (1.f / l_h) : 0.f;
    const int64_t out_off = (q_row_base + h) * D_CK;
    const __nv_bfloat162 packed = __halves2bfloat162(
        __float2bfloat16_rn(O[h][0] * inv),
        __float2bfloat16_rn(O[h][1] * inv));
    reinterpret_cast<__nv_bfloat162*>(out + out_off)[tid] = packed;
  }
}

void require_b200() {
  int dev = -1;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  TVM_FFI_ICHECK_EQ(prop.major, 10) << "mla_cuda: requires B200 (cc 10.0), got "
                                    << prop.major << "." << prop.minor;
  TVM_FFI_ICHECK_EQ(prop.minor, 0) << "mla_cuda: requires B200 (cc 10.0), got "
                                   << prop.major << "." << prop.minor;
}

void configure_kernel_attrs_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    // B200 caps dynamic SMEM at 227 KiB (228 KiB is rejected). We need ~164 KiB.
    cudaFuncSetAttribute(mla_sparse_attn_umma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         200 * 1024);
  });
}

}  // namespace

void kernel_fn(
    tvm::ffi::TensorView q_nope,
    tvm::ffi::TensorView q_pe,
    tvm::ffi::TensorView ckv_cache,
    tvm::ffi::TensorView kpe_cache,
    tvm::ffi::TensorView sparse_indices,
    double sm_scale,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView lse) {
  require_b200();
  configure_kernel_attrs_once();

  TVM_FFI_ICHECK_EQ(q_nope.device().device_type, kDLCUDA) << "q_nope must be CUDA";
  TVM_FFI_ICHECK_EQ(q_pe.device().device_type, kDLCUDA) << "q_pe must be CUDA";
  TVM_FFI_ICHECK_EQ(ckv_cache.device().device_type, kDLCUDA) << "ckv_cache must be CUDA";
  TVM_FFI_ICHECK_EQ(kpe_cache.device().device_type, kDLCUDA) << "kpe_cache must be CUDA";
  TVM_FFI_ICHECK_EQ(sparse_indices.device().device_type, kDLCUDA) << "sparse_indices must be CUDA";
  TVM_FFI_ICHECK_EQ(out.device().device_type, kDLCUDA) << "out must be CUDA";
  TVM_FFI_ICHECK_EQ(lse.device().device_type, kDLCUDA) << "lse must be CUDA";

  TVM_FFI_ICHECK_EQ(q_nope.dtype().code, kDLBfloat); TVM_FFI_ICHECK_EQ(q_nope.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(q_pe.dtype().code,   kDLBfloat); TVM_FFI_ICHECK_EQ(q_pe.dtype().bits,   16);
  TVM_FFI_ICHECK_EQ(ckv_cache.dtype().code, kDLBfloat); TVM_FFI_ICHECK_EQ(ckv_cache.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(kpe_cache.dtype().code, kDLBfloat); TVM_FFI_ICHECK_EQ(kpe_cache.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(sparse_indices.dtype().code, kDLInt); TVM_FFI_ICHECK_EQ(sparse_indices.dtype().bits, 32);
  TVM_FFI_ICHECK_EQ(out.dtype().code, kDLBfloat); TVM_FFI_ICHECK_EQ(out.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(lse.dtype().code, kDLFloat);  TVM_FFI_ICHECK_EQ(lse.dtype().bits, 32);

  TVM_FFI_ICHECK_EQ(q_nope.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q_pe.ndim(),   3);
  TVM_FFI_ICHECK_EQ(sparse_indices.ndim(), 2);
  const int64_t T = q_nope.size(0);
  const int64_t H = q_nope.size(1);
  TVM_FFI_ICHECK_EQ(H, HEADS) << "head count must equal compile-time HEADS=" << HEADS;
  TVM_FFI_ICHECK_EQ(q_nope.size(2), D_CK);
  TVM_FFI_ICHECK_EQ(q_pe.size(0), T);  TVM_FFI_ICHECK_EQ(q_pe.size(1), H);
  TVM_FFI_ICHECK_EQ(q_pe.size(2), D_KP);
  TVM_FFI_ICHECK_EQ(sparse_indices.size(0), T);
  TVM_FFI_ICHECK_EQ(sparse_indices.size(1), TOPK);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), T); TVM_FFI_ICHECK_EQ(out.size(1), H); TVM_FFI_ICHECK_EQ(out.size(2), D_CK);
  TVM_FFI_ICHECK_EQ(lse.ndim(), 2);
  TVM_FFI_ICHECK_EQ(lse.size(0), T); TVM_FFI_ICHECK_EQ(lse.size(1), H);

  TVM_FFI_ICHECK_GE(ckv_cache.ndim(), 1);
  TVM_FFI_ICHECK_EQ(ckv_cache.size(ckv_cache.ndim() - 1), D_CK);
  TVM_FFI_ICHECK_GE(kpe_cache.ndim(), 1);
  TVM_FFI_ICHECK_EQ(kpe_cache.size(kpe_cache.ndim() - 1), D_KP);
  int64_t ckv_numel = 1;
  for (int i = 0; i < ckv_cache.ndim(); ++i) ckv_numel *= ckv_cache.size(i);
  int64_t kpe_numel = 1;
  for (int i = 0; i < kpe_cache.ndim(); ++i) kpe_numel *= kpe_cache.size(i);
  const int64_t num_kv_rows = ckv_numel / D_CK;
  TVM_FFI_ICHECK_EQ(kpe_numel / D_KP, num_kv_rows);

  const float sm = static_cast<float>(sm_scale);
  const float sm_log2e = sm * LOG2E_F;

  const dim3 grid(static_cast<unsigned>(T));
  const dim3 block(THREADS);

  size_t shmem = 0;
  shmem += static_cast<size_t>(MMA_M) * D_CK * sizeof(__nv_bfloat16);  // s_q_n
  shmem += static_cast<size_t>(MMA_M) * D_KP * sizeof(__nv_bfloat16);  // s_q_p
  shmem += static_cast<size_t>(V_PIPE) * BK * D_CK * sizeof(__nv_bfloat16);  // s_ckv
  shmem += static_cast<size_t>(V_PIPE) * BK * D_KP * sizeof(__nv_bfloat16);  // s_kpe
  shmem += static_cast<size_t>(MMA_M) * BK * sizeof(__nv_bfloat16);    // s_p_umma
  shmem += static_cast<size_t>(D_CK) * BK * sizeof(__nv_bfloat16);     // s_v_t
  shmem = (shmem + 3u) & ~size_t{3u};
  shmem += static_cast<size_t>(HEADS) * D_CK * sizeof(float);          // s_pv
  shmem += static_cast<size_t>(TOPK) * sizeof(int);                    // s_gidx
  shmem += 3 * static_cast<size_t>(HEADS) * sizeof(float);             // s_m/l/scale
  shmem += static_cast<size_t>(NUM_WARPS) * sizeof(float);             // s_reduce
  shmem = (shmem + 7u) & ~size_t{7u};
  shmem += sizeof(uint64_t);    // mbar
  shmem += sizeof(uint32_t);    // tmem ptr

  const DLDevice dev = q_nope.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  mla_sparse_attn_umma_kernel<<<grid, block, shmem, stream>>>(
      static_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      static_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      static_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
      static_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
      static_cast<const int32_t*>(sparse_indices.data_ptr()),
      sm_log2e,
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<float*>(lse.data_ptr()),
      num_kv_rows);

  const cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "mla_sparse_attn_umma_kernel: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel_fn);
