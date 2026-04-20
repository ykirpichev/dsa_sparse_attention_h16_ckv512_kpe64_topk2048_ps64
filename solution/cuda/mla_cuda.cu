/*
 * DSA sparse MLA — custom CUDA (B200 / sm_100, capability 10.0).
 * Stage-3 structure: one CUDA block per (token, head-tile) where a head-tile
 * groups H_TILE heads that share the K loads.  sparse_idx is per-token (not
 * per-head), so every head in the tile touches the exact same set of K rows.
 *
 * Stage-1 optimizations (batched):
 *   - Packed BF16 loads: __nv_bfloat162 (2 bf16 per 32-bit load) via __bfloat1622float2.
 *   - Cached sparse_idx per block in shared memory (avoid re-reading in value projection).
 *   - Warp + block parallel softmax (max / sum) instead of serial thread-0 loops.
 *   - exp2f with pre-fused sm_scale * log2e so LSE is base-2 without a final /log(2).
 *
 * Stage-2 (additive, keeps the high-MLP structure of Stage 1):
 *   - Packed bf162 loads AND stores in the value projection: each thread owns
 *     exactly one packed output slot (d = 2*tid, 2*tid+1).
 *   - Fold the softmax normalization (division by sum) into the final bf16 write.
 *   - Earlier tiled-SMEM fusion regressed because per-tile __syncthreads chopped
 *     up the memory pipeline; kept the sync-light single-pass design.
 *
 * Stage-3: head tiling (H_TILE heads per block share K loads)
 *   - Grid is now (T, H / H_TILE). Each block handles H_TILE consecutive heads
 *     of the same token, loading Q for all of them into SMEM.
 *   - In the logit pass, each packed K load (__nv_bfloat162) is reused for
 *     H_TILE dot products (one per head) instead of 1. The single HBM read
 *     amortizes across H_TILE logit accumulations.
 *   - In the value projection, each packed K load is reused for H_TILE * 2 FMAs
 *     (2 output lanes per thread × H_TILE heads).
 *   - K-side HBM traffic per head halves (H_TILE=2), so the compute-bound large
 *     traces where FI was still ~1.6× ahead become the ones with the biggest
 *     win here.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cmath>
#include <mutex>

namespace py = pybind11;

namespace {

constexpr int D_CK = 512;
constexpr int D_KP = 64;
constexpr int TOPK = 2048;
constexpr int THREADS = 256;
constexpr int WARP = 32;
constexpr int NUM_WARPS = THREADS / WARP;  // 8
constexpr int H_TILE = 2;                  // heads per block (Stage 3)
constexpr float LOG2E_F = 1.4426950408889634f;

static_assert(D_CK == 2 * THREADS, "packed output path requires D_CK == 2*THREADS");

float sm_scale_to_float(py::object o) {
  try {
    const torch::Tensor t = o.cast<torch::Tensor>();
    return static_cast<float>(t.item<double>());
  } catch (const py::cast_error&) {
    return static_cast<float>(py::cast<double>(o));
  }
}

__device__ __forceinline__ float bf16_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f_to_bf16(float x) { return __float2bfloat16_rn(x); }

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

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
  v = warp_reduce_sum(v);
  const int lane = threadIdx.x & (WARP - 1);
  const int wid = threadIdx.x / WARP;
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  float r = 0.f;
  if (wid == 0) {
    float x = (lane < NUM_WARPS) ? smem[lane] : 0.f;
    x = warp_reduce_sum(x);
    if (lane == 0) smem[0] = x;
  }
  __syncthreads();
  r = smem[0];
  __syncthreads();
  return r;
}

__global__ __launch_bounds__(THREADS, 2) void mla_sparse_attn_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_flat,
    const __nv_bfloat16* __restrict__ kpe_flat,
    const int32_t* __restrict__ sparse_idx,
    float sm_scale_log2e,  // = sm_scale * log2(e); logits live in base-2 from the start
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse_out,
    int64_t H,
    int64_t num_kv_rows) {
  const int t = blockIdx.x;
  const int h_tile_idx = blockIdx.y;
  const int h_base = h_tile_idx * H_TILE;
  const int tid = threadIdx.x;

  const int64_t si_off = t * TOPK;

  // SMEM layout (for H_TILE=2):
  //   s_logits[H_TILE][TOPK]  float    16 KiB
  //   s_qn    [H_TILE][D_CK]  float     4 KiB
  //   s_qp    [H_TILE][D_KP]  float   512 B
  //   s_gidx  [TOPK]          int       8 KiB (shared across heads in the tile)
  //   s_reduce[NUM_WARPS]     float    32 B
  // Total: ~28.6 KiB (well under the 100 KiB attribute cap).
  extern __shared__ unsigned char raw[];
  unsigned char* cur = raw;
  float* s_logits_base = reinterpret_cast<float*>(cur);
  cur += static_cast<size_t>(H_TILE) * TOPK * sizeof(float);
  float* s_qn_base = reinterpret_cast<float*>(cur);
  cur += static_cast<size_t>(H_TILE) * D_CK * sizeof(float);
  float* s_qp_base = reinterpret_cast<float*>(cur);
  cur += static_cast<size_t>(H_TILE) * D_KP * sizeof(float);
  int* s_gidx = reinterpret_cast<int*>(cur);
  cur += TOPK * sizeof(int);
  float* s_reduce = reinterpret_cast<float*>(cur);

  auto s_logits_row = [&](int h) { return s_logits_base + h * TOPK; };
  auto s_qn_row     = [&](int h) { return s_qn_base + h * D_CK; };
  auto s_qp_row     = [&](int h) { return s_qp_base + h * D_KP; };

  // Load Q for all heads in the tile as fp32 (packed bf162 -> 2 floats at a time).
  // Each thread covers one packed slot per head; one iter for q_nope (D_CK/2 == THREADS)
  // and a partial set of threads for q_pe.
  {
    const int i = tid;  // packed index in [0, D_CK/2)
    #pragma unroll
    for (int hi = 0; hi < H_TILE; ++hi) {
      const int64_t qh_off = (t * H + (h_base + hi)) * D_CK;
      const __nv_bfloat162* qn2 = reinterpret_cast<const __nv_bfloat162*>(q_nope + qh_off);
      const float2 qf = __bfloat1622float2(qn2[i]);
      float* dst = s_qn_row(hi);
      dst[2 * i]     = qf.x;
      dst[2 * i + 1] = qf.y;
    }
  }
  if (tid < D_KP / 2) {
    const int i = tid;
    #pragma unroll
    for (int hi = 0; hi < H_TILE; ++hi) {
      const int64_t qp_off = (t * H + (h_base + hi)) * D_KP;
      const __nv_bfloat162* qp2 = reinterpret_cast<const __nv_bfloat162*>(q_pe + qp_off);
      const float2 qf = __bfloat1622float2(qp2[i]);
      float* dst = s_qp_row(hi);
      dst[2 * i]     = qf.x;
      dst[2 * i + 1] = qf.y;
    }
  }

  // Cache sparse_idx (shared across heads in the tile).
  for (int k = tid; k < TOPK; k += THREADS) {
    int32_t g = sparse_idx[si_off + k];
    s_gidx[k] = (g >= 0 && static_cast<int64_t>(g) < num_kv_rows) ? g : -1;
  }
  __syncthreads();

  // Logits: each thread handles a subset of k. For each k, read the (ckv, kpe)
  // row ONCE and fold it into H_TILE separate logit accumulators.
  for (int k = tid; k < TOPK; k += THREADS) {
    const int g = s_gidx[k];
    if (g < 0) {
      #pragma unroll
      for (int hi = 0; hi < H_TILE; ++hi) s_logits_row(hi)[k] = -INFINITY;
      continue;
    }
    const __nv_bfloat162* kc2 =
        reinterpret_cast<const __nv_bfloat162*>(ckv_flat + static_cast<int64_t>(g) * D_CK);
    const __nv_bfloat162* kp2 =
        reinterpret_cast<const __nv_bfloat162*>(kpe_flat + static_cast<int64_t>(g) * D_KP);

    float acc[H_TILE];
    #pragma unroll
    for (int hi = 0; hi < H_TILE; ++hi) acc[hi] = 0.f;

    #pragma unroll
    for (int i = 0; i < D_CK / 2; ++i) {
      const float2 kf = __bfloat1622float2(kc2[i]);
      const int d = i * 2;
      #pragma unroll
      for (int hi = 0; hi < H_TILE; ++hi) {
        const float* qn = s_qn_row(hi);
        acc[hi] += qn[d] * kf.x + qn[d + 1] * kf.y;
      }
    }
    #pragma unroll
    for (int i = 0; i < D_KP / 2; ++i) {
      const float2 kf = __bfloat1622float2(kp2[i]);
      const int d = i * 2;
      #pragma unroll
      for (int hi = 0; hi < H_TILE; ++hi) {
        const float* qp = s_qp_row(hi);
        acc[hi] += qp[d] * kf.x + qp[d + 1] * kf.y;
      }
    }
    #pragma unroll
    for (int hi = 0; hi < H_TILE; ++hi) {
      s_logits_row(hi)[k] = acc[hi] * sm_scale_log2e;
    }
  }
  __syncthreads();

  // Per-head softmax state (unnormalized; divide at output write time).
  // Also track per-head (mx, sum) in a small SMEM scratch so we can write LSE
  // and skip empty rows without an extra cross-head reduction.
  float mx_h[H_TILE];
  float sum_h[H_TILE];

  #pragma unroll
  for (int hi = 0; hi < H_TILE; ++hi) {
    float local_max = -INFINITY;
    const float* sl = s_logits_row(hi);
    for (int k = tid; k < TOPK; k += THREADS) {
      local_max = fmaxf(local_max, sl[k]);
    }
    mx_h[hi] = block_reduce_max(local_max, s_reduce);
  }

  // All-masked short circuit: if both heads are -INF, write zeros + LSE=-INF and return.
  // (Both heads of the same token share sparse_idx, so if one is empty, both are.)
  bool any_valid = false;
  #pragma unroll
  for (int hi = 0; hi < H_TILE; ++hi) any_valid = any_valid || (mx_h[hi] != -INFINITY);
  if (!any_valid) {
    const __nv_bfloat162 zero = __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    #pragma unroll
    for (int hi = 0; hi < H_TILE; ++hi) {
      const int64_t out_off = (t * H + (h_base + hi)) * D_CK;
      reinterpret_cast<__nv_bfloat162*>(out + out_off)[tid] = zero;
      if (tid == 0) lse_out[t * H + (h_base + hi)] = -INFINITY;
    }
    return;
  }

  // Exp + sum, per head.
  #pragma unroll
  for (int hi = 0; hi < H_TILE; ++hi) {
    const float mx = mx_h[hi];
    float* sl = s_logits_row(hi);
    float local_sum = 0.f;
    if (mx == -INFINITY) {
      // This head is empty (shouldn't happen if any_valid and both heads share gidx,
      // but handle defensively): zero out logits so the output accumulation is 0.
      for (int k = tid; k < TOPK; k += THREADS) sl[k] = 0.f;
      sum_h[hi] = 0.f;
    } else {
      for (int k = tid; k < TOPK; k += THREADS) {
        const float e = exp2f(sl[k] - mx);
        sl[k] = e;
        local_sum += e;
      }
      sum_h[hi] = block_reduce_sum(local_sum, s_reduce);
      if (tid == 0) lse_out[t * H + (h_base + hi)] = mx + log2f(sum_h[hi]);
    }
  }

  // Value projection, packed. Each thread holds H_TILE * 2 accumulators.
  // Each packed K load is reused for H_TILE * 2 FMAs.
  const int i = tid;
  float o0[H_TILE], o1[H_TILE];
  #pragma unroll
  for (int hi = 0; hi < H_TILE; ++hi) { o0[hi] = 0.f; o1[hi] = 0.f; }

  for (int k = 0; k < TOPK; ++k) {
    const int g = s_gidx[k];
    if (g < 0) continue;
    const __nv_bfloat162* kc2 =
        reinterpret_cast<const __nv_bfloat162*>(ckv_flat + static_cast<int64_t>(g) * D_CK);
    const float2 kf = __bfloat1622float2(kc2[i]);
    #pragma unroll
    for (int hi = 0; hi < H_TILE; ++hi) {
      const float p = s_logits_row(hi)[k];
      o0[hi] += p * kf.x;
      o1[hi] += p * kf.y;
    }
  }

  #pragma unroll
  for (int hi = 0; hi < H_TILE; ++hi) {
    const float s = sum_h[hi];
    const float inv = (s == 0.f) ? 0.f : 1.f / s;
    const int64_t out_off = (t * H + (h_base + hi)) * D_CK;
    const __nv_bfloat162 packed =
        __halves2bfloat162(f_to_bf16(o0[hi] * inv), f_to_bf16(o1[hi] * inv));
    reinterpret_cast<__nv_bfloat162*>(out + out_off)[i] = packed;
  }
}

void require_b200() {
  int dev = -1;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  if (prop.major != 10 || prop.minor != 0) {
    TORCH_CHECK(false, "mla_cuda: requires B200-class GPU (compute capability 10.0), got ", prop.major, ".", prop.minor);
  }
}

// Opt into the large per-CTA dynamic SMEM budget on sm_100 (up to ~228 KiB).
// Stage-3 kernel uses ~28.6 KiB; the 100 KiB cap leaves headroom for further
// tiling (cp.async double buffering / larger H_TILE) without touching this again.
void configure_kernel_attrs_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    cudaFuncSetAttribute(mla_sparse_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         100 * 1024);
  });
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> run(
    const torch::Tensor& q_nope,
    const torch::Tensor& q_pe,
    const torch::Tensor& ckv_cache,
    const torch::Tensor& kpe_cache,
    const torch::Tensor& sparse_indices,
    const py::object& sm_scale_obj) {
  require_b200();
  configure_kernel_attrs_once();

  TORCH_CHECK(q_nope.is_cuda() && q_nope.is_contiguous(), "q_nope must be contiguous CUDA");
  TORCH_CHECK(q_pe.is_cuda() && q_pe.is_contiguous(), "q_pe must be contiguous CUDA");
  TORCH_CHECK(ckv_cache.is_cuda() && ckv_cache.is_contiguous(), "ckv_cache must be contiguous CUDA");
  TORCH_CHECK(kpe_cache.is_cuda() && kpe_cache.is_contiguous(), "kpe_cache must be contiguous CUDA");
  TORCH_CHECK(sparse_indices.is_cuda() && sparse_indices.is_contiguous(), "sparse_indices must be contiguous CUDA");

  TORCH_CHECK(q_nope.scalar_type() == at::ScalarType::BFloat16, "q_nope must be bf16");
  TORCH_CHECK(q_pe.scalar_type() == at::ScalarType::BFloat16, "q_pe must be bf16");
  TORCH_CHECK(ckv_cache.scalar_type() == at::ScalarType::BFloat16, "ckv_cache must be bf16");
  TORCH_CHECK(kpe_cache.scalar_type() == at::ScalarType::BFloat16, "kpe_cache must be bf16");

  const int64_t T = q_nope.size(0);
  const int64_t H = q_nope.size(1);
  TORCH_CHECK(q_pe.size(0) == T && q_pe.size(1) == H, "q_pe batch/heads must match q_nope");
  TORCH_CHECK(q_nope.size(2) == D_CK && q_pe.size(2) == D_KP, "head dims mismatch");
  TORCH_CHECK(sparse_indices.size(0) == T && sparse_indices.size(1) == TOPK, "sparse_indices shape");
  TORCH_CHECK(H % H_TILE == 0, "H must be divisible by H_TILE (", H_TILE, ")");

  auto idx = sparse_indices.scalar_type() == at::ScalarType::Int ? sparse_indices : sparse_indices.to(at::ScalarType::Int);
  TORCH_CHECK(idx.is_contiguous());

  const float sm = sm_scale_to_float(sm_scale_obj);
  const float sm_log2e = sm * LOG2E_F;

  auto ckv = ckv_cache.reshape({-1, D_CK}).contiguous();
  auto kpe = kpe_cache.reshape({-1, D_KP}).contiguous();
  const int64_t num_kv_rows = ckv.size(0);
  TORCH_CHECK(num_kv_rows == kpe.size(0), "ckv/kpe flat mismatch");

  auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q_nope.device());
  auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device());
  auto out = torch::empty({T, H, D_CK}, opts_bf16);
  auto lse = torch::empty({T, H}, opts_f32);

  const int64_t H_GROUPS = H / H_TILE;
  const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H_GROUPS));
  const dim3 block(THREADS);
  const size_t shmem =
      static_cast<size_t>(H_TILE) * (TOPK + D_CK + D_KP) * sizeof(float)  // s_logits + s_qn + s_qp
      + static_cast<size_t>(TOPK) * sizeof(int)                            // s_gidx
      + static_cast<size_t>(NUM_WARPS) * sizeof(float);                    // s_reduce

  const auto stream = at::cuda::getCurrentCUDAStream(q_nope.device().index());

  mla_sparse_attn_kernel<<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(ckv.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kpe.data_ptr()),
      idx.data_ptr<int32_t>(),
      sm_log2e,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      lse.data_ptr<float>(),
      H,
      num_kv_rows);

  {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mla_sparse_attn_kernel: ", cudaGetErrorString(err));
  }

  return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "DSA sparse MLA (CUDA)");
}
