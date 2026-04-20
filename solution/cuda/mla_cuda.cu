/*
 * DSA sparse MLA — custom CUDA (B200 / sm_100, capability 10.0).
 * One CUDA block per (token, head): gather K rows, logits, stable softmax, LSE base-2, weighted cKV.
 *
 * Stage-1 optimizations (batched):
 *   - Packed BF16 loads: __nv_bfloat162 (2 bf16 per 32-bit load) via __bfloat1622float2.
 *   - Cached sparse_idx per block in shared memory (avoid re-reading in value projection).
 *   - Warp + block parallel softmax (max / sum) instead of serial thread-0 loops.
 *   - exp2f with pre-fused sm_scale * log2e so LSE is base-2 without a final /log(2).
 *   - Minor cleanups: single return, FLT-style constants, __launch_bounds__ hint.
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
constexpr float LOG2E_F = 1.4426950408889634f;

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

// Packed dot(fp32 Q in smem, bf16 K in gmem) over D elements using nv_bfloat162.
template <int D>
__device__ __forceinline__ float packed_dot_bf16_f32(const float* __restrict__ q_f, const __nv_bfloat16* __restrict__ k_bf) {
  static_assert(D % 2 == 0, "D must be even for packed bf16 load");
  const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(k_bf);
  float acc = 0.f;
  #pragma unroll
  for (int i = 0; i < D / 2; ++i) {
    float2 kf = __bfloat1622float2(k2[i]);
    const int d = i * 2;
    acc += q_f[d] * kf.x + q_f[d + 1] * kf.y;
  }
  return acc;
}

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
  const int h = blockIdx.y;
  const int tid = threadIdx.x;

  const int64_t qh_off = (t * H + h) * D_CK;
  const int64_t qp_off = (t * H + h) * D_KP;
  const int64_t si_off = t * TOPK;
  const int64_t out_off = (t * H + h) * D_CK;
  const int64_t lse_i = t * H + h;

  extern __shared__ unsigned char raw[];
  float* s_logits = reinterpret_cast<float*>(raw);           // TOPK   floats
  float* s_qn = s_logits + TOPK;                              // D_CK   floats
  float* s_qp = s_qn + D_CK;                                  // D_KP   floats
  int* s_gidx = reinterpret_cast<int*>(s_qp + D_KP);          // TOPK   ints
  float* s_reduce = reinterpret_cast<float*>(s_gidx + TOPK);  // NUM_WARPS floats

  // Load Q into shared as fp32 (reused by every logit).
  for (int d = tid; d < D_CK; d += THREADS) {
    s_qn[d] = bf16_to_f(q_nope[qh_off + d]);
  }
  for (int d = tid; d < D_KP; d += THREADS) {
    s_qp[d] = bf16_to_f(q_pe[qp_off + d]);
  }
  // Cache sparse_idx (clamped to valid range) so the value-projection pass does not re-read global.
  for (int k = tid; k < TOPK; k += THREADS) {
    int32_t g = sparse_idx[si_off + k];
    s_gidx[k] = (g >= 0 && static_cast<int64_t>(g) < num_kv_rows) ? g : -1;
  }
  __syncthreads();

  // Logits: each thread handles a subset of k, full D_CK+D_KP dot per k.
  for (int k = tid; k < TOPK; k += THREADS) {
    const int g = s_gidx[k];
    if (g < 0) {
      s_logits[k] = -INFINITY;
      continue;
    }
    const __nv_bfloat16* kc = ckv_flat + static_cast<int64_t>(g) * D_CK;
    const __nv_bfloat16* kp = kpe_flat + static_cast<int64_t>(g) * D_KP;
    float acc = packed_dot_bf16_f32<D_CK>(s_qn, kc)
              + packed_dot_bf16_f32<D_KP>(s_qp, kp);
    s_logits[k] = acc * sm_scale_log2e;  // base-2 logits
  }
  __syncthreads();

  // Block-parallel max.
  float local_max = -INFINITY;
  for (int k = tid; k < TOPK; k += THREADS) {
    local_max = fmaxf(local_max, s_logits[k]);
  }
  const float mx = block_reduce_max(local_max, s_reduce);

  // All-masked row handling: write zeros and LSE=-inf, nothing else to compute.
  if (mx == -INFINITY) {
    if (tid == 0) lse_out[lse_i] = -INFINITY;
    for (int d = tid; d < D_CK; d += THREADS) {
      out[out_off + d] = f_to_bf16(0.f);
    }
    return;
  }

  // Block-parallel normalized softmax (base-2 exp).
  float local_sum = 0.f;
  for (int k = tid; k < TOPK; k += THREADS) {
    const float e = exp2f(s_logits[k] - mx);
    s_logits[k] = e;
    local_sum += e;
  }
  const float sum = block_reduce_sum(local_sum, s_reduce);
  const float inv = 1.f / sum;

  if (tid == 0) lse_out[lse_i] = mx + log2f(sum);

  for (int k = tid; k < TOPK; k += THREADS) {
    s_logits[k] *= inv;
  }
  __syncthreads();

  // Weighted cKV projection. Each thread accumulates a different d across all k.
  for (int d = tid; d < D_CK; d += THREADS) {
    float acc = 0.f;
    for (int k = 0; k < TOPK; ++k) {
      const int g = s_gidx[k];
      if (g < 0) continue;
      const __nv_bfloat16* kc = ckv_flat + static_cast<int64_t>(g) * D_CK;
      acc += s_logits[k] * bf16_to_f(kc[d]);
    }
    out[out_off + d] = f_to_bf16(acc);
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
// Stage-1 kernel only uses ~18.5 KiB but we set a 100 KiB cap now so Stage-2/3
// tile sizes don't require touching this again.
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

  const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
  const dim3 block(THREADS);
  // s_logits + s_qn + s_qp + s_gidx + s_reduce
  const size_t shmem =
      (static_cast<size_t>(TOPK) + D_CK + D_KP) * sizeof(float)
      + static_cast<size_t>(TOPK) * sizeof(int)
      + static_cast<size_t>(NUM_WARPS) * sizeof(float);

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
