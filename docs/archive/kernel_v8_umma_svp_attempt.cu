/*
 * DSA sparse MLA — B200 / sm_100a (Blackwell) kernel.
 *
 * Stage-8 (v8): fused-heads + Blackwell tensor-core logit UMMA + scalar
 * value accumulate + fused online softmax.
 *
 *   Grid  : (T,)          — one CTA per token, covering all H=16 heads.
 *   Block : 256 threads (8 warps).
 *
 * Logit pass: tcgen05.mma.kind::f16 UMMA (bf16 inputs, fp32 accum) with
 * MMA_M=64 (16 real heads + 48 zero-padded rows), N=BK=16. K-atom = 16 bf16.
 *   Q_nope · K_ckv^T : 32 K-iters (D_CK=512 = 32×16 bf16)
 *   Q_pe   · K_kpe^T :  4 K-iters (D_KP= 64 =  4×16 bf16, accum=1)
 *
 * Online softmax (fused, per head): warp 0 lanes 0..15 own one head each;
 * each lane computes its row's new max, alpha rescale, exp2, sum, and
 * updates the running (m, l) state in SMEM. The P[h, r] values are written
 * to s_p_row[HEADS, BK] fp32 so every thread can read them for the scalar
 * V accumulate.
 *
 * Value pass (scalar):
 *   O[h, d] = alpha_h * O_prev[h, d] + sum_r P[h, r] * V[r, d]
 * executed by all 256 threads. Each thread owns DIMS_PER_THREAD = D_CK /
 * THREADS = 2 consecutive output dims. V is read from s_ckv (SWIZZLE_128B
 * K-major) via the same swz128b_chunk16b_offset() helper used for the
 * logit pass load.
 *
 * SMEM layout (rough):
 *   s_q_n    [MMA_M=64, D_CK=512] bf16   = 64 KiB (rows 16..63 zero-padded)
 *   s_q_p    [MMA_M=64, D_KP= 64] bf16   =  8 KiB
 *   s_ckv    [V_PIPE=2, BK=16, D_CK] bf16= 32 KiB  (SWIZZLE_128B)
 *   s_kpe    [V_PIPE=2, BK=16, D_KP] bf16=  4 KiB  (SWIZZLE_128B)
 *   s_p_row  [HEADS=16, BK=16] fp32      =  1 KiB  (row-major, readable by all)
 *   s_gidx   [TOPK=2048] int             =  8 KiB
 *   s_m/l/scale [HEADS]  fp32            =  192 B each
 *   s_reduce [NUM_WARPS=8] fp32          =   32 B
 *   mbarriers + tmem ptr                 ~   32 B
 * Total ≈ 118 KiB (dynamic SMEM cap on sm_100 is 227 KiB).
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

// ------------------------------- Constants ---------------------------------
constexpr int D_CK = 512;
constexpr int D_KP = 64;
constexpr int TOPK = 2048;
constexpr int HEADS = 16;

constexpr int THREADS = 256;
constexpr int WARP = 32;
constexpr int NUM_WARPS = THREADS / WARP;  // 8
constexpr int BK = 16;
constexpr int V_PIPE = 2;
static_assert(TOPK % BK == 0, "TOPK must be divisible by BK");
constexpr int NUM_TILES = TOPK / BK;       // 128

// UMMA logit shapes. Smallest MMA_M for kind::f16 is 64; we pad from 16 heads.
constexpr int MMA_M = 64;
constexpr int MMA_N = BK;                  // 16
constexpr int K_ATOM_BF16 = 16;
constexpr int K_ATOM_BYTES = K_ATOM_BF16 * sizeof(__nv_bfloat16);  // 32 B
constexpr int K_ITERS_NOPE = D_CK / K_ATOM_BF16;  // 32
constexpr int K_ITERS_PE   = D_KP / K_ATOM_BF16;  //  4

// SWIZZLE_128B: 8 rows × 128 B = 1024 B per "swizzle-block".
constexpr int SWZ_BLOCK_BYTES      = 1024;
constexpr int SWZ_BLOCK_CHUNKS_16B = SWZ_BLOCK_BYTES / 16;  // 64
constexpr int BYTES_PER_SWZ_ROW    = 128;
constexpr int NOPE_K_BLOCKS = D_CK * 2 / BYTES_PER_SWZ_ROW;  // 8
constexpr int PE_K_BLOCKS   = D_KP * 2 / BYTES_PER_SWZ_ROW;  // 1
static_assert(NOPE_K_BLOCKS * BYTES_PER_SWZ_ROW == D_CK * 2, "D_CK row bytes 128B multiple");
static_assert(PE_K_BLOCKS   * BYTES_PER_SWZ_ROW == D_KP * 2, "D_KP row bytes 128B multiple");
constexpr uint32_t SBO_NOPE_BYTES = NOPE_K_BLOCKS * SWZ_BLOCK_BYTES;  // 8192
constexpr uint32_t SBO_PE_BYTES   = PE_K_BLOCKS   * SWZ_BLOCK_BYTES;  // 1024
constexpr uint32_t LBO_BYTES      = 16;

// Per-thread output partitioning.
static_assert(D_CK % THREADS == 0, "D_CK must divide THREADS");
constexpr int DIMS_PER_THREAD = D_CK / THREADS;  // 2
constexpr float LOG2E_F = 1.4426950408889634f;

// TMEM: logit pass uses cols 0..15 (fp32).
constexpr uint32_t TMEM_NUM_COLS = 64;

// --------------------- Swizzled SMEM indexing helpers ----------------------
__device__ __forceinline__ int
swz128b_chunk16b_offset(int m, int k_chunk, int k_blocks_per_row) {
  const int m_group = m >> 3;
  const int row_in_group = m & 7;
  const int k_block = k_chunk >> 3;
  const int chunk_in_block = k_chunk & 7;
  const int logical_in_block = row_in_group * 8 + chunk_in_block;
  const int phys_in_block = logical_in_block ^ row_in_group;
  return m_group * k_blocks_per_row * SWZ_BLOCK_CHUNKS_16B
       + k_block * SWZ_BLOCK_CHUNKS_16B
       + phys_in_block;
}

__device__ __forceinline__ int
swz128b_chunk8b_offset(int m, int k_chunk8, int k_blocks_per_row) {
  const int k_chunk16 = k_chunk8 >> 1;
  const int in_16 = k_chunk8 & 1;
  return 2 * swz128b_chunk16b_offset(m, k_chunk16, k_blocks_per_row) + in_16;
}

__device__ __forceinline__ int
swz128b_katom_base_bytes(int k, int k_blocks_per_row) {
  const int k_block = k >> 2;
  const int k_atom_in_block = k & 3;
  (void)k_blocks_per_row;
  return k_block * SWZ_BLOCK_BYTES + k_atom_in_block * K_ATOM_BYTES;
}

// -------------------------- Warp / block reductions ------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int off = WARP / 2; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  }
  return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
  v = warp_reduce_max(v);
  const int lane = threadIdx.x & (WARP - 1);
  const int wid  = threadIdx.x / WARP;
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float x = (lane < NUM_WARPS) ? smem[lane] : -INFINITY;
    x = warp_reduce_max(x);
    if (lane == 0) smem[0] = x;
  }
  __syncthreads();
  float r = smem[0];
  __syncthreads();
  return r;
}

// -------------------------- SmemDescriptor builder -------------------------
__device__ __forceinline__ dsa_umma::SmemDescriptor
make_kmajor_desc_b128(const void* smem_base, uint32_t sbo_bytes) {
  dsa_umma::SmemDescriptor d{};
  d.desc_                = 0;
  d.version_             = 1;
  d.lbo_mode_            = 0;
  d.base_offset_         = 0;
  d.layout_type_         = static_cast<uint8_t>(dsa_umma::LayoutType::SWIZZLE_128B);
  d.leading_byte_offset_ = static_cast<uint16_t>(LBO_BYTES >> 4);
  d.stride_byte_offset_  = static_cast<uint16_t>(sbo_bytes >> 4);
  d.start_address_       = static_cast<uint16_t>(
      dsa_ptx::smem_ptr_to_uint(smem_base) >> 4);
  return d;
}

// ------------------------------ The kernel ---------------------------------
__global__ __launch_bounds__(THREADS, 1) void mla_sparse_attn_umma_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_flat,
    const __nv_bfloat16* __restrict__ kpe_flat,
    const int32_t* __restrict__ sparse_idx,
    float sm_scale_log2e,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse_out,
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
  float*  s_p_row  = reinterpret_cast<float*>(s_kpe + V_PIPE * BK * D_KP);
  int*    s_gidx   = reinterpret_cast<int*>(s_p_row + HEADS * BK);
  float*  s_m      = reinterpret_cast<float*>(s_gidx + TOPK);
  float*  s_l      = s_m + HEADS;
  float*  s_scale  = s_l + HEADS;
  float*  s_reduce = s_scale + HEADS;
  uintptr_t mbar_raw = reinterpret_cast<uintptr_t>(s_reduce + NUM_WARPS);
  mbar_raw = (mbar_raw + 7u) & ~uintptr_t{7u};
  uint64_t* s_mbar      = reinterpret_cast<uint64_t*>(mbar_raw);
  uint32_t* s_tmem_ptr  = reinterpret_cast<uint32_t*>(s_mbar + 1);

  // -------------------- Phase 0: init state & zero-pad Q --------------------
  // Zero full s_q_n / s_q_p (MMA_M=64, only rows 0..15 are real heads).
  #pragma unroll 1
  for (int i = tid; i < MMA_M * D_CK; i += THREADS) {
    s_q_n[i] = __float2bfloat16(0.f);
  }
  #pragma unroll 1
  for (int i = tid; i < MMA_M * D_KP; i += THREADS) {
    s_q_p[i] = __float2bfloat16(0.f);
  }
  if (tid < HEADS) {
    s_m[tid] = -INFINITY;
    s_l[tid] = 0.f;
  }
  __syncthreads();

  // -------------------- Phase 1: load Q / sparse_idx ------------------------
  // Q_nope → swizzled s_q_n (bf162 stores).
  {
    constexpr int Q_NOPE_BF162 = HEADS * (D_CK / 2);  // 16 * 256 = 4096
    const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(
        q_nope + q_row_base * D_CK);
    __nv_bfloat162* q2_smem = reinterpret_cast<__nv_bfloat162*>(s_q_n);
    #pragma unroll 1
    for (int i = tid; i < Q_NOPE_BF162; i += THREADS) {
      const int h = i / (D_CK / 2);
      const int d2 = i - h * (D_CK / 2);
      const int d = d2 * 2;
      const int k_chunk16 = d >> 3;
      const int chunk_off = swz128b_chunk16b_offset(h, k_chunk16, NOPE_K_BLOCKS);
      const int bf162_off = chunk_off * 4 + (d & 7) / 2;
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

  // sparse_idx: validate & cache. Track local max valid k+1 for tile truncation.
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
  const uint32_t tmem_logit = *s_tmem_ptr;

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
  const int d0 = tid * DIMS_PER_THREAD;

  // -------------------- K/V tile cp.async issuer ----------------------------
  auto issue_kv_tile = [&](int tile, int buf) {
    const int k0 = tile * BK;
    __nv_bfloat16* ckv_buf = s_ckv + buf * (BK * D_CK);
    __nv_bfloat16* kpe_buf = s_kpe + buf * (BK * D_KP);
    const uint32_t ckv_buf_addr = dsa_ptx::smem_ptr_to_uint(ckv_buf);
    const uint32_t kpe_buf_addr = dsa_ptx::smem_ptr_to_uint(kpe_buf);

    constexpr int CKV_CHUNKS_16B = BK * D_CK / 8;
    constexpr int CKV_CHUNKS_PER_THREAD = CKV_CHUNKS_16B / THREADS;
    #pragma unroll
    for (int c = 0; c < CKV_CHUNKS_PER_THREAD; ++c) {
      const int chunk = tid + c * THREADS;
      const int r = chunk / (D_CK / 8);
      const int kc16 = chunk - r * (D_CK / 8);
      int g = s_gidx[k0 + r];
      if (g < 0) g = 0;
      const uint4* src = reinterpret_cast<const uint4*>(
          ckv_flat + static_cast<int64_t>(g) * D_CK) + kc16;
      const int dst_chunk = swz128b_chunk16b_offset(r, kc16, NOPE_K_BLOCKS);
      dsa_ptx::cp_async_16B(ckv_buf_addr + dst_chunk * 16u, src);
    }
    constexpr int KPE_CHUNKS_8B = BK * D_KP * 2 / 8;
    constexpr int KPE_CHUNKS_PER_THREAD = KPE_CHUNKS_8B / THREADS;
    #pragma unroll
    for (int c = 0; c < KPE_CHUNKS_PER_THREAD; ++c) {
      const int chunk = tid + c * THREADS;
      const int r = chunk / (D_KP / 4);
      const int kc8 = chunk - r * (D_KP / 4);
      int g = s_gidx[k0 + r];
      if (g < 0) g = 0;
      const uint2* src = reinterpret_cast<const uint2*>(
          kpe_flat + static_cast<int64_t>(g) * D_KP) + kc8;
      const int dst_chunk8 = swz128b_chunk8b_offset(r, kc8, PE_K_BLOCKS);
      dsa_ptx::cp_async_8B(kpe_buf_addr + dst_chunk8 * 8u, src);
    }
  };

  // Wait for Q stores.
  __syncthreads();

  // Prime tile 0.
  if (n_valid_tiles > 0) {
    issue_kv_tile(0, 0);
    dsa_ptx::cp_async_commit_group();
  }

  uint32_t mbar_phase = 0;

  const auto instr_logit =
      dsa_umma::make_instr_desc_f16(dsa_umma::F16Format::BF16,
                                    dsa_umma::F16Format::BF16,
                                    dsa_umma::CFormat::F32,
                                    dsa_umma::Major::K, dsa_umma::Major::K,
                                    MMA_M, MMA_N);

  // ---- Precompute per-thread V-load swizzle offsets (one per r in BK) ----
  // Each thread owns 2 consecutive dims (d0, d0+1) which live in the same
  // 16B chunk (d0 is 2-aligned, the low 3 bits give position within chunk).
  // For each K-row r, the 16B chunk base offset is swz128b_chunk16b_offset(
  //   r, d0>>3, NOPE_K_BLOCKS). We cache the 16B chunk offsets for r=0..BK-1.
  int v_chunk_off[BK];
  const int my_kc16 = d0 >> 3;
  const int in_chunk = d0 & 7;      // 0..6 (even)
  #pragma unroll
  for (int r = 0; r < BK; ++r) {
    v_chunk_off[r] = swz128b_chunk16b_offset(r, my_kc16, NOPE_K_BLOCKS);
  }

  // -------------------- Main tile loop --------------------------------------
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

    // --- Logit UMMA: Q·K^T into TMEM [0..15] ---
    if (warp_id == 0) {
      dsa_ptx::tcgen05_fence_after_thread_sync();
      if (lane == 0) {
        auto a_base_n = make_kmajor_desc_b128(s_q_n,    SBO_NOPE_BYTES);
        auto b_base_n = make_kmajor_desc_b128(ckv_buf,  SBO_NOPE_BYTES);
        auto a_base_p = make_kmajor_desc_b128(s_q_p,    SBO_PE_BYTES);
        auto b_base_p = make_kmajor_desc_b128(kpe_buf,  SBO_PE_BYTES);

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

    // --- TMEM readback: warp 0 lanes 0..15 each own one head (row) ---
    // Per CUTLASS sm100 TMEM layout: MMA_M=64 maps rows 0..31 to warp 0,
    // rows 32..63 to warp 1. For logit output (MMA_N=16 cols in TMEM), we
    // use .x16 which returns 16 fp32 values per lane corresponding to that
    // lane's row (= head) across 16 cols.
    float logits[BK];
    if (warp_id == 0) {
      uint32_t regs[BK];
      dsa_ptx::tcgen05_ld_32x32b_x16_b32(tmem_logit, regs);
      dsa_ptx::tcgen05_wait_ld();
      #pragma unroll
      for (int r = 0; r < BK; ++r) logits[r] = __uint_as_float(regs[r]);
    }

    // --- Online softmax (warp 0 lane=h, writes P[h] → s_p_row[h,:] fp32) ---
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

      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        s_p_row[h * BK + r] = logits[r];
      }
    }
    __syncthreads();

    // --- Scalar value accumulate ---
    // Each thread owns 2 consecutive dims: d0, d0+1.
    // O[h][d_own] = scale[h] * O_prev + sum_{r} P[h,r] * V[r, d_own]
    // V[r, d] is at ckv_buf[swz128b_chunk16b_offset(r, d>>3)*8 + (d&7)].
    #pragma unroll
    for (int h = 0; h < HEADS; ++h) {
      const float sc = s_scale[h];
      float acc0 = sc * O[h][0];
      float acc1 = sc * O[h][1];
      const float* p_row = &s_p_row[h * BK];
      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        const int chunk_off = v_chunk_off[r];
        // Load the 2 bf16 values at (in_chunk, in_chunk+1) as one bf162.
        const __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(
            ckv_buf + chunk_off * 8 + in_chunk);
        const float2 vf = __bfloat1622float2(v2);
        const float p = p_row[r];
        acc0 = fmaf(p, vf.x, acc0);
        acc1 = fmaf(p, vf.y, acc1);
      }
      O[h][0] = acc0;
      O[h][1] = acc1;
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

// ---------------------- Host setup ----------------------------------------
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

constexpr int KERNEL_MAX_DYN_SMEM = 128 * 1024;

void configure_kernel_attrs_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    cudaFuncSetAttribute(mla_sparse_attn_umma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         KERNEL_MAX_DYN_SMEM);
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
  shmem += static_cast<size_t>(MMA_M) * D_CK * sizeof(__nv_bfloat16);     // s_q_n  64 KiB
  shmem += static_cast<size_t>(MMA_M) * D_KP * sizeof(__nv_bfloat16);     // s_q_p   8 KiB
  shmem += static_cast<size_t>(V_PIPE) * BK * D_CK * sizeof(__nv_bfloat16);// s_ckv 32 KiB (V_PIPE=2)
  shmem += static_cast<size_t>(V_PIPE) * BK * D_KP * sizeof(__nv_bfloat16);// s_kpe  4 KiB
  shmem += static_cast<size_t>(HEADS) * BK * sizeof(float);               // s_p_row 1 KiB
  shmem += static_cast<size_t>(TOPK) * sizeof(int);                       // s_gidx 8 KiB
  shmem += 3 * static_cast<size_t>(HEADS) * sizeof(float);                // s_m/l/scale 192 B
  shmem += static_cast<size_t>(NUM_WARPS) * sizeof(float);                // s_reduce 32 B
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
