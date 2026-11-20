// Shared constants and device helpers.
// Included from kernel.cu inside the anonymous namespace.

constexpr int D_CK = 512;
constexpr int D_KP = 64;
constexpr int TOPK = 2048;
constexpr int THREADS = 256;
constexpr int WARP = 32;
constexpr int NUM_WARPS = THREADS / WARP;  // 8
constexpr float LOG2E_F = 1.4426950408889634f;

// Value-projection tiling parameters (Stage 3b cp.async double buffering).
// BK = K rows per tile; V_PIPE = number of SMEM buffers (double-buffer => 2).
// Tile size (bf16) = BK * D_CK * 2 B = 16 * 1024 = 16 KiB; 2 buffers = 32 KiB.
// Per thread per tile load: 4 * 16 B = 64 B (256 threads * 64 B = 16 KiB).
constexpr int BK = 16;
constexpr int V_PIPE = 2;
static_assert(TOPK % BK == 0, "TOPK must be divisible by BK");
static_assert(BK * D_CK % (THREADS * 16 / 2) == 0,
              "BK tile size must be a multiple of THREADS * 8 bf16 elements");
constexpr int NUM_V_TILES = TOPK / BK;                               // 128
constexpr int TILE_CHUNKS_16B = BK * D_CK * sizeof(__nv_bfloat16) / 16;  // 1024
constexpr int CHUNKS_PER_THREAD = TILE_CHUNKS_16B / THREADS;         // 4

// Async Q load: Q_nope is D_CK=512 bf16 = 1024 B = 64 uint4; Q_pe is
// D_KP=64 bf16 = 128 B = 8 uint4. Offsets qh_off/qp_off are multiples of
// D_CK/D_KP bf16 (i.e. 1024 B / 128 B) so always 16 B aligned.
constexpr int Q_CK_CHUNKS = D_CK * sizeof(__nv_bfloat16) / 16;  // 64
constexpr int Q_KP_CHUNKS = D_KP * sizeof(__nv_bfloat16) / 16;  //  8
static_assert(D_CK * sizeof(__nv_bfloat16) % 16 == 0, "Q_nope row must be 16 B aligned");
static_assert(D_KP * sizeof(__nv_bfloat16) % 16 == 0, "Q_pe row must be 16 B aligned");

// int4 prescan of sparse_idx. TOPK int32s = 8 KB → TOPK/4 = 512 int4 chunks.
// Each thread handles TOPK/4/THREADS = 2 chunks.
constexpr int IDX_INT4_CHUNKS = TOPK / 4;            // 512
constexpr int IDX_INT4_PER_THREAD = IDX_INT4_CHUNKS / THREADS;  // 2
static_assert(TOPK % 4 == 0, "TOPK must be divisible by 4 for int4 prescan");
static_assert(IDX_INT4_CHUNKS % THREADS == 0,
              "IDX_INT4_CHUNKS must be a multiple of THREADS");

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

// cp.async.ca.shared.global wrappers (inline PTX).
// `smem_int` must come from __cvta_generic_to_shared on the SMEM destination.
// 16 B copies are the widest the instruction supports and match our float4 K layout.
__device__ __forceinline__ void cp_async_16(unsigned smem_int, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_int), "l"(gmem_src));
}

// 8-byte variant: used to stream D_KP=64 bf16 rows (128 B / row) in a single
// issue per thread across 256 threads (16 rows × 16 8-byte chunks = 256 per
// tile).
__device__ __forceinline__ void cp_async_8(unsigned smem_int, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
               :
               : "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// cp.async.wait_group needs an immediate argument; template over the depth.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
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

// Phase-timing instrumentation (diag02).
//   Define `PROFILE_PHASES` (e.g. via env / NVCC_PREPEND_FLAGS) to have block
//   (0,0) print phase cycle counts to stderr at the end of the kernel. Prints
//   once per launch (per workload). Zero overhead when disabled.
#if PROFILE_PHASES
#define PHASE_MARK(var)                                                   \
  do {                                                                    \
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {         \
      var = clock64();                                                    \
    }                                                                     \
  } while (0)
#define PHASE_PRINT(c0, c1, c2, c3, c4, c5, T_, H_, nvt)                  \
  do {                                                                    \
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {         \
      printf("[diag02] T=%lld H=%lld n_valid_tiles=%d  "                  \
             "qload=%lld  logit=%lld  softmax=%lld  vproj=%lld  "         \
             "owrite=%lld  total=%lld  (cycles)\n",                       \
             (long long)T_, (long long)H_, (int)nvt,                      \
             (long long)(c1 - c0), (long long)(c2 - c1),                  \
             (long long)(c3 - c2), (long long)(c4 - c3),                  \
             (long long)(c5 - c4), (long long)(c5 - c0));                 \
    }                                                                     \
  } while (0)
#else
#define PHASE_MARK(var) do { } while (0)
#define PHASE_PRINT(c0, c1, c2, c3, c4, c5, T_, H_, nvt) do { } while (0)
#endif
