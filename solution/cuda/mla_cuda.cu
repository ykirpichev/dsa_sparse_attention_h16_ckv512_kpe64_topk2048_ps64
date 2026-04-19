/*
 * DSA sparse MLA — custom CUDA (B200 / sm_100, capability 10.0).
 * One CUDA block per (token, head): gather K rows, logits, stable softmax, LSE base-2, weighted cKV.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cmath>

namespace py = pybind11;

namespace {

float sm_scale_to_float(py::object o) {
  try {
    const torch::Tensor t = o.cast<torch::Tensor>();
    return static_cast<float>(t.item<double>());
  } catch (const py::cast_error&) {
    return static_cast<float>(py::cast<double>(o));
  }
}

constexpr int D_CK = 512;
constexpr int D_KP = 64;
constexpr int TOPK = 2048;
constexpr int THREADS = 256;

__device__ __forceinline__ float bf16_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ __nv_bfloat16 f_to_bf16(float x) { return __float2bfloat16_rn(x); }

__global__ void mla_sparse_attn_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_flat,
    const __nv_bfloat16* __restrict__ kpe_flat,
    const int32_t* __restrict__ sparse_idx,
    float sm_scale,
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
  float* s_logits = reinterpret_cast<float*>(raw);
  float* s_qn = s_logits + TOPK;
  float* s_qp = s_qn + D_CK;
  __shared__ int s_empty;

  for (int d = tid; d < D_CK; d += THREADS) {
    s_qn[d] = bf16_to_f(q_nope[qh_off + d]);
  }
  for (int d = tid; d < D_KP; d += THREADS) {
    s_qp[d] = bf16_to_f(q_pe[qp_off + d]);
  }
  __syncthreads();

  if (tid == 0) {
    bool empty = true;
    for (int k = 0; k < TOPK; ++k) {
      if (sparse_idx[si_off + k] >= 0) {
        empty = false;
        break;
      }
    }
    s_empty = empty ? 1 : 0;
  }
  __syncthreads();

  if (s_empty) {
    if (tid == 0) {
      lse_out[lse_i] = -INFINITY;
    }
    for (int d = tid; d < D_CK; d += THREADS) {
      out[out_off + d] = f_to_bf16(0.f);
    }
  } else {
  for (int k = tid; k < TOPK; k += THREADS) {
    int32_t gidx = sparse_idx[si_off + k];
    if (gidx < 0) {
      s_logits[k] = -INFINITY;
      continue;
    }
    if (gidx >= num_kv_rows) {
      s_logits[k] = -INFINITY;
      continue;
    }
    const __nv_bfloat16* kc = ckv_flat + static_cast<int64_t>(gidx) * D_CK;
    const __nv_bfloat16* kp = kpe_flat + static_cast<int64_t>(gidx) * D_KP;
    float acc = 0.f;
#pragma unroll 4
    for (int d = 0; d < D_CK; ++d) {
      acc += s_qn[d] * bf16_to_f(kc[d]);
    }
#pragma unroll 4
    for (int d = 0; d < D_KP; ++d) {
      acc += s_qp[d] * bf16_to_f(kp[d]);
    }
    s_logits[k] = acc * sm_scale;
  }
  __syncthreads();

  if (tid == 0) {
    float mx = -INFINITY;
    for (int k = 0; k < TOPK; ++k) {
      mx = fmaxf(mx, s_logits[k]);
    }
    float sum = 0.f;
    for (int k = 0; k < TOPK; ++k) {
      float e = expf(s_logits[k] - mx);
      s_logits[k] = e;
      sum += e;
    }
    const float inv = 1.f / sum;
    for (int k = 0; k < TOPK; ++k) {
      s_logits[k] *= inv;
    }
    const float lse = (mx + logf(sum)) / logf(2.f);
    lse_out[lse_i] = lse;
  }
  __syncthreads();

  for (int d = tid; d < D_CK; d += THREADS) {
    float acc = 0.f;
    for (int k = 0; k < TOPK; ++k) {
      int32_t gidx = sparse_idx[si_off + k];
      if (gidx < 0) {
        continue;
      }
      const __nv_bfloat16* kc = ckv_flat + static_cast<int64_t>(gidx) * D_CK;
      acc += s_logits[k] * bf16_to_f(kc[d]);
    }
    out[out_off + d] = f_to_bf16(acc);
  }
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

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> run(
    const torch::Tensor& q_nope,
    const torch::Tensor& q_pe,
    const torch::Tensor& ckv_cache,
    const torch::Tensor& kpe_cache,
    const torch::Tensor& sparse_indices,
    const py::object& sm_scale_obj) {
  require_b200();

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
  const size_t shmem = (static_cast<size_t>(TOPK) + D_CK + D_KP) * sizeof(float);

  const auto stream = at::cuda::getCurrentCUDAStream(q_nope.device().index());

  mla_sparse_attn_kernel<<<grid, block, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(ckv.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kpe.data_ptr()),
      idx.data_ptr<int32_t>(),
      sm,
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
