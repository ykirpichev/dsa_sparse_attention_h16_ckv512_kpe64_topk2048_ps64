/*
 * DSA sparse MLA — custom CUDA (B200 / sm_100, capability 10.0).
 *
 * v9_h4_scalar: per-(T, H/H_PER_CTA) CTA restructure.
 *
 *   Grid  : (T, H / H_PER_CTA) with H_PER_CTA = 4 heads per CTA.
 *   Block : 256 threads (8 warps).
 *
 * Each CTA now processes H_PER_CTA=4 heads of a single token. This amortises
 * K/V HBM+L2 traffic 4×: both ckv and kpe rows for this tile are loaded once
 * and reused across all 4 heads' logit dots and V accumulates. V in
 * particular is reused 4× in the value-accumulate inner loop — each V bf162
 * pair loaded from SMEM is multiplied by 4 heads' softmax probabilities
 * instead of 1.
 *
 * Per-tile fused K loop (same high-level structure as v5_fused but with
 * H_PER_CTA parallel heads):
 *   SYNC_A
 *   Logit compute: warp-parallel, 4*BK=64 logits per tile (vs 16 in v5).
 *     8 warps, each warp does 8 warp-reduced dots (vs 2 in v5).
 *   SYNC_B
 *   Online softmax rescale for each head + V accumulate with V shared across
 *   heads in the inner r-loop.
 *
 * SMEM layout (~48 KiB per CTA):
 *   s_logits_tile [H_PER_CTA * BK]   fp32   = 256 B
 *   s_qn          [H_PER_CTA, D_CK]  bf16   = 4 KiB
 *   s_qp          [H_PER_CTA, D_KP]  bf16   = 512 B
 *   s_gidx        [TOPK]             int32  = 8 KiB
 *   s_reduce      [NUM_WARPS]        fp32   = 32 B
 *   s_v_tile      [V_PIPE, BK, D_CK] bf16   = 32 KiB  (V_PIPE=2)
 *   s_kpe_tile    [V_PIPE, BK, D_KP] bf16   = 4 KiB
 *   TOTAL ≈ 48.8 KiB → still 3+ CTAs/SM on B200's 228 KiB dynamic SMEM cap.
 *
 * Grid parallelism:
 *   T × (H/4) total CTAs. For T=4096, H=16: 16 384 CTAs (was 65 536).
 *   For T=32: 128 CTAs (was 512). Short-trace grid is still well above the
 *   148-SM saturation point for all practical workloads.
 *
 * History / motivation:
 *   v5_fused is at ~33× geomean and is close to the L2-bandwidth ceiling
 *   for its current CTA structure. The per-(T, H) layout loads every K/V
 *   row 16× (once per head), relying entirely on L2 to amortise the
 *   repeated HBM requests. The h4 restructure guarantees same-CTA sharing
 *   of K/V within the tile loop, which should:
 *     - cut effective HBM / L2 request rate ~4×
 *     - keep the same per-thread compute pattern (logit dot + V FMA)
 *     - run the same softmax algorithm (now per-head x4)
 *   Expected win: ~15-25 % on long traces where L2 reuse was imperfect.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace {

constexpr int D_CK = 512;
constexpr int D_KP = 64;
constexpr int TOPK = 2048;
constexpr int THREADS = 256;
constexpr int WARP = 32;
constexpr int NUM_WARPS = THREADS / WARP;  // 8
constexpr int H_PER_CTA = 4;   // heads per CTA; H must be a multiple of this
constexpr float LOG2E_F = 1.4426950408889634f;

// Value-projection tiling parameters (same as v5_fused).
constexpr int BK = 16;
constexpr int V_PIPE = 2;
static_assert(TOPK % BK == 0, "TOPK must be divisible by BK");
static_assert(BK * D_CK % (THREADS * 16 / 2) == 0,
              "BK tile size must be a multiple of THREADS * 8 bf16 elements");
constexpr int NUM_V_TILES = TOPK / BK;                                   // 128
constexpr int TILE_CHUNKS_16B = BK * D_CK * sizeof(__nv_bfloat16) / 16;  // 1024
constexpr int CHUNKS_PER_THREAD = TILE_CHUNKS_16B / THREADS;             // 4

// Logit-pass cooperation: each warp computes LOGITS_PER_WARP warp-reduced
// dots per tile. Total logits per tile = NUM_WARPS * LOGITS_PER_WARP
// = H_PER_CTA * BK.
constexpr int LOGITS_PER_WARP = (H_PER_CTA * BK + NUM_WARPS - 1) / NUM_WARPS;  // 8
static_assert(NUM_WARPS * LOGITS_PER_WARP == H_PER_CTA * BK,
              "logit partition must cover H_PER_CTA*BK exactly");

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
  const int wid = threadIdx.x / WARP;
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

__device__ __forceinline__ void cp_async_16(unsigned smem_int, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_8(unsigned smem_int, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
               :
               : "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ __nv_bfloat16 f_to_bf16(float x) {
  return __float2bfloat16_rn(x);
}

__global__ __launch_bounds__(THREADS, 3) void mla_sparse_attn_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_flat,
    const __nv_bfloat16* __restrict__ kpe_flat,
    const int32_t* __restrict__ sparse_idx,
    float sm_scale_log2e,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse_out,
    int64_t H,
    int64_t num_kv_rows) {
  const int t  = blockIdx.x;
  const int hg = blockIdx.y;              // head-group index [0, H/H_PER_CTA)
  const int h0 = hg * H_PER_CTA;          // first head index for this CTA
  const int tid = threadIdx.x;

  const int64_t th_base = t * H + h0;     // first (t, h) pair
  const int64_t si_off  = t * TOPK;

  // --------------------------- SMEM layout --------------------------------
  extern __shared__ unsigned char raw[];
  float*          s_logits_tile = reinterpret_cast<float*>(raw);        // [H_PER_CTA*BK]
  __nv_bfloat16*  s_qn          = reinterpret_cast<__nv_bfloat16*>(s_logits_tile + H_PER_CTA * BK);
  __nv_bfloat16*  s_qp          = s_qn + H_PER_CTA * D_CK;
  int*            s_gidx        = reinterpret_cast<int*>(s_qp + H_PER_CTA * D_KP);
  float*          s_reduce      = reinterpret_cast<float*>(s_gidx + TOPK);
  __nv_bfloat16*  s_v_tile      = reinterpret_cast<__nv_bfloat16*>(s_reduce + NUM_WARPS);
  __nv_bfloat16*  s_kpe_tile    = s_v_tile + V_PIPE * BK * D_CK;

  // --------------------------- Q load (4 heads) ---------------------------
  // s_qn [H_PER_CTA, D_CK] row-major; s_qp [H_PER_CTA, D_KP] row-major.
  {
    constexpr int Q_NOPE_BF16 = H_PER_CTA * D_CK;                       // 2048
    const __nv_bfloat16* q_src = q_nope + th_base * D_CK;
    for (int i = tid; i < Q_NOPE_BF16; i += THREADS) {
      s_qn[i] = q_src[i];
    }
  }
  {
    constexpr int Q_PE_BF16 = H_PER_CTA * D_KP;                         // 256
    const __nv_bfloat16* q_src = q_pe + th_base * D_KP;
    for (int i = tid; i < Q_PE_BF16; i += THREADS) {
      s_qp[i] = q_src[i];
    }
  }

  // Cache sparse_idx and track the last valid tile.
  int local_max_valid_k_p1 = 0;
  for (int k = tid; k < TOPK; k += THREADS) {
    int32_t g = sparse_idx[si_off + k];
    const bool valid = (g >= 0 && static_cast<int64_t>(g) < num_kv_rows);
    s_gidx[k] = valid ? g : -1;
    if (valid && (k + 1) > local_max_valid_k_p1) local_max_valid_k_p1 = k + 1;
  }
  __syncthreads();

  const float reduced_max = block_reduce_max(
      static_cast<float>(local_max_valid_k_p1), s_reduce);
  const int n_valid_rows = static_cast<int>(reduced_max);
  int n_valid_tiles = (n_valid_rows + BK - 1) / BK;
  if (n_valid_tiles > NUM_V_TILES) n_valid_tiles = NUM_V_TILES;

  // Empty-token fast path.
  if (n_valid_tiles == 0) {
    if (tid < H_PER_CTA) {
      lse_out[th_base + tid] = -INFINITY;
    }
    const __nv_bfloat162 zero =
        __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    #pragma unroll
    for (int h = 0; h < H_PER_CTA; ++h) {
      const int64_t o_off = (th_base + h) * D_CK;
      reinterpret_cast<__nv_bfloat162*>(out + o_off)[tid] = zero;
    }
    return;
  }

  // --------------------------- Tile issuer --------------------------------
  auto issue_tile = [&](int tile, int buf) {
    const int k0 = tile * BK;
    __nv_bfloat16* ckv_buf = s_v_tile + buf * (BK * D_CK);
    __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);

    #pragma unroll
    for (int c = 0; c < CHUNKS_PER_THREAD; ++c) {
      const int chunk = tid + c * THREADS;
      const int r = chunk / (D_CK / 8);
      const int ch16 = chunk % (D_CK / 8);
      const int k = k0 + r;
      int g = s_gidx[k];
      if (g < 0) g = 0;
      const uint4* src =
          reinterpret_cast<const uint4*>(ckv_flat + static_cast<int64_t>(g) * D_CK) + ch16;
      uint4* dst = reinterpret_cast<uint4*>(ckv_buf + r * D_CK) + ch16;
      cp_async_16(__cvta_generic_to_shared(dst), src);
    }
    {
      const int chunk = tid;
      const int r = chunk / (D_KP / 4);
      const int ch8 = chunk % (D_KP / 4);
      const int k = k0 + r;
      int g = s_gidx[k];
      if (g < 0) g = 0;
      const uint2* src =
          reinterpret_cast<const uint2*>(kpe_flat + static_cast<int64_t>(g) * D_KP) + ch8;
      uint2* dst = reinterpret_cast<uint2*>(kpe_buf + r * D_KP) + ch8;
      cp_async_8(__cvta_generic_to_shared(dst), src);
    }
  };

  static_assert(D_CK == 2 * THREADS, "packed output path requires D_CK == 2*THREADS");

  // --------------------------- Per-thread state ---------------------------
  // Per-head running (m, l, O) state. Each thread owns output dims
  // (2*tid, 2*tid+1) for each of H_PER_CTA heads.
  float m[H_PER_CTA];
  float l[H_PER_CTA];
  float o0[H_PER_CTA];
  float o1[H_PER_CTA];
  #pragma unroll
  for (int h = 0; h < H_PER_CTA; ++h) {
    m[h]  = -INFINITY;
    l[h]  = 0.f;
    o0[h] = 0.f;
    o1[h] = 0.f;
  }

  issue_tile(0, 0);
  cp_async_commit_group();

  const int warp_id = tid >> 5;
  const int lane    = tid & (WARP - 1);
  const int i       = tid;   // output-channel-pair owner: (2*tid, 2*tid+1)

  // --------------------------- Main tile loop -----------------------------
  for (int tile = 0; tile < n_valid_tiles; ++tile) {
    const int buf = tile & 1;
    const int next_tile = tile + 1;

    if (next_tile < n_valid_tiles) {
      issue_tile(next_tile, next_tile & 1);
      cp_async_commit_group();
      cp_async_wait_group<1>();
    } else {
      cp_async_wait_group<0>();
    }
    __syncthreads();  // SYNC_A: cp.async visible

    const __nv_bfloat16* ckv_buf = s_v_tile + buf * (BK * D_CK);
    const __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);

    // --- Logit compute (warp-parallel) ---
    // 8 warps × LOGITS_PER_WARP=8 logits each = 64 logits per tile.
    // Indexing: logit index L in [0, H_PER_CTA*BK). We pack as L = h*BK + r,
    // so warp_id stride is LOGITS_PER_WARP contiguous L's. This keeps each
    // warp's 8 logits within the same Q-row (+/-) so the warp loads one Q row
    // fewer times.
    #pragma unroll
    for (int w = 0; w < LOGITS_PER_WARP; ++w) {
      const int L = warp_id * LOGITS_PER_WARP + w;
      const int h = L / BK;         // head within CTA [0, H_PER_CTA)
      const int r = L - h * BK;     // k-row within tile [0, BK)
      const int k = tile * BK + r;

      const __nv_bfloat162* qn2 =
          reinterpret_cast<const __nv_bfloat162*>(s_qn + h * D_CK);
      const __nv_bfloat162* qp2 =
          reinterpret_cast<const __nv_bfloat162*>(s_qp + h * D_KP);
      const __nv_bfloat162* kc2 =
          reinterpret_cast<const __nv_bfloat162*>(ckv_buf + r * D_CK);
      const __nv_bfloat162* kp2 =
          reinterpret_cast<const __nv_bfloat162*>(kpe_buf + r * D_KP);

      float acc = 0.f;
      #pragma unroll
      for (int j = 0; j < D_CK / 2 / WARP; ++j) {
        const int i2 = j * WARP + lane;
        const float2 kf = __bfloat1622float2(kc2[i2]);
        const float2 qf = __bfloat1622float2(qn2[i2]);
        acc += qf.x * kf.x + qf.y * kf.y;
      }
      {
        const float2 kf = __bfloat1622float2(kp2[lane]);
        const float2 qf = __bfloat1622float2(qp2[lane]);
        acc += qf.x * kf.x + qf.y * kf.y;
      }
      acc = warp_reduce_sum(acc);
      if (lane == 0) {
        const int g = s_gidx[k];
        s_logits_tile[L] = (g >= 0) ? (acc * sm_scale_log2e) : -INFINITY;
      }
    }
    __syncthreads();  // SYNC_B: publish logits

    // --- Softmax state update per head (scalar temps, no arrays) ---
    // Compute per-head tile_max + rescale (l, o0, o1) in a tight loop; each
    // head's tile_max only lives in a single register while we process it.
    // Whole-tile mask check: read logit[0] scan separately since tile_max is
    // consumed immediately.
    float tile_scan0 = -INFINITY;  // head 0's tile_max — block-uniform mask
    #pragma unroll
    for (int h = 0; h < H_PER_CTA; ++h) {
      float tm = -INFINITY;
      #pragma unroll
      for (int r = 0; r < BK; ++r) tm = fmaxf(tm, s_logits_tile[h * BK + r]);
      if (h == 0) tile_scan0 = tm;
      if (tm > -INFINITY) {
        const float mn = fmaxf(m[h], tm);
        const float alpha =
            (m[h] == -INFINITY) ? 0.f : exp2f(m[h] - mn);
        l[h]  *= alpha;
        o0[h] *= alpha;
        o1[h] *= alpha;
        m[h] = mn;
      }
    }

    // Block-uniform skip when the whole tile is masked (all g<0).
    if (tile_scan0 > -INFINITY) {
      const __nv_bfloat162* tile_pairs =
          reinterpret_cast<const __nv_bfloat162*>(ckv_buf);

      // V-reuse inner loop: load one V pair per r, fan out across H_PER_CTA
      // heads. This is where the h4 restructure pays off: one SMEM load per
      // r is amortised across 4 FMA operations (one per head).
      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        const float2 vf =
            __bfloat1622float2(tile_pairs[r * (D_CK / 2) + i]);

        #pragma unroll
        for (int h = 0; h < H_PER_CTA; ++h) {
          const float logit = s_logits_tile[h * BK + r];
          const float p     = exp2f(logit - m[h]);
          l[h]  += p;
          o0[h] = fmaf(p, vf.x, o0[h]);
          o1[h] = fmaf(p, vf.y, o1[h]);
        }
      }
    }
  }

  // --------------------------- Epilogue -----------------------------------
  if (tid < H_PER_CTA) {
    const int h = tid;
    lse_out[th_base + h] = (m[h] == -INFINITY) ? -INFINITY : (m[h] + log2f(l[h]));
  }

  #pragma unroll
  for (int h = 0; h < H_PER_CTA; ++h) {
    const float inv_l = (l[h] > 0.f) ? (1.f / l[h]) : 0.f;
    const __nv_bfloat162 packed =
        __halves2bfloat162(f_to_bf16(o0[h] * inv_l),
                           f_to_bf16(o1[h] * inv_l));
    const int64_t o_off = (th_base + h) * D_CK;
    reinterpret_cast<__nv_bfloat162*>(out + o_off)[i] = packed;
  }
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void require_b200() {
  int dev = -1;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  TVM_FFI_ICHECK_EQ(prop.major, 10) << "mla_cuda: requires B200-class GPU (compute capability 10.0), got "
                                    << prop.major << "." << prop.minor;
  TVM_FFI_ICHECK_EQ(prop.minor, 0) << "mla_cuda: requires B200-class GPU (compute capability 10.0), got "
                                   << prop.major << "." << prop.minor;
}

constexpr int KERNEL_MAX_DYN_SMEM = 51 * 1024;

void configure_kernel_attrs_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    cudaFuncSetAttribute(mla_sparse_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         KERNEL_MAX_DYN_SMEM);

    cudaFuncAttributes attrs{};
    cudaFuncGetAttributes(&attrs, (const void*)mla_sparse_attn_kernel);
    int occ_full_cap = 0;
    int occ_actual = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_full_cap, mla_sparse_attn_kernel, THREADS, KERNEL_MAX_DYN_SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_actual, mla_sparse_attn_kernel, THREADS, 49 * 1024);
    fprintf(stderr,
            "[mla_sparse_attn v9_h4] regs/thread=%d  localSz=%d  "
            "static_smem=%zu  max_dyn_smem=%d  max_threads=%d  "
            "occ@cap=%d  occ@actual(49K)=%d blocks/SM\n",
            attrs.numRegs, (int)attrs.localSizeBytes,
            (size_t)attrs.sharedSizeBytes, attrs.maxDynamicSharedSizeBytes,
            attrs.maxThreadsPerBlock, occ_full_cap, occ_actual);
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
  TVM_FFI_ICHECK_EQ(q_nope.size(2), D_CK);
  TVM_FFI_ICHECK_EQ(q_pe.size(0), T);  TVM_FFI_ICHECK_EQ(q_pe.size(1), H);
  TVM_FFI_ICHECK_EQ(q_pe.size(2), D_KP);
  TVM_FFI_ICHECK_EQ(sparse_indices.size(0), T);
  TVM_FFI_ICHECK_EQ(sparse_indices.size(1), TOPK);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), T); TVM_FFI_ICHECK_EQ(out.size(1), H); TVM_FFI_ICHECK_EQ(out.size(2), D_CK);
  TVM_FFI_ICHECK_EQ(lse.ndim(), 2);
  TVM_FFI_ICHECK_EQ(lse.size(0), T); TVM_FFI_ICHECK_EQ(lse.size(1), H);

  TVM_FFI_ICHECK_EQ(H % H_PER_CTA, 0)
      << "H must be a multiple of H_PER_CTA=" << H_PER_CTA << ", got H=" << H;

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

  const dim3 grid(static_cast<unsigned>(T),
                  static_cast<unsigned>(H / H_PER_CTA));
  const dim3 block(THREADS);

  const size_t shmem =
      static_cast<size_t>(H_PER_CTA * BK) * sizeof(float)
      + static_cast<size_t>(H_PER_CTA) * (D_CK + D_KP) * sizeof(__nv_bfloat16)
      + static_cast<size_t>(TOPK) * sizeof(int)
      + static_cast<size_t>(NUM_WARPS) * sizeof(float)
      + static_cast<size_t>(V_PIPE) * BK * D_CK * sizeof(__nv_bfloat16)
      + static_cast<size_t>(V_PIPE) * BK * D_KP * sizeof(__nv_bfloat16);

  const DLDevice dev = q_nope.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  mla_sparse_attn_kernel<<<grid, block, shmem, stream>>>(
      static_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      static_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      static_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
      static_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
      static_cast<const int32_t*>(sparse_indices.data_ptr()),
      sm_log2e,
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<float*>(lse.data_ptr()),
      H,
      num_kv_rows);

  const cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "mla_sparse_attn_kernel: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel_fn);
