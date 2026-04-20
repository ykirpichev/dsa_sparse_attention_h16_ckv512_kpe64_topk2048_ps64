/*
 * DSA sparse MLA — custom CUDA (B200 / sm_100, capability 10.0).
 * One CUDA block per (token, head): gather K rows, logits, stable softmax, LSE base-2, weighted cKV.
 *
 * Stage-1 optimizations (batched):
 *   - Packed BF16 loads: __nv_bfloat162 (2 bf16 per 32-bit load) via __bfloat1622float2.
 *   - Cached sparse_idx per block in shared memory (avoid re-reading in value projection).
 *   - Warp + block parallel softmax (max / sum) instead of serial thread-0 loops.
 *   - exp2f with pre-fused sm_scale * log2e so LSE is base-2 without a final /log(2).
 *
 * Stage-2 (additive, keeps the high-MLP structure of Stage 1):
 *   - Packed bf162 loads AND stores in the value projection: each thread owns
 *     exactly one packed output slot (d = 2*tid, 2*tid+1), so every k in the
 *     inner loop becomes 1 packed GMEM load + 2 FMAs instead of 2 scalar
 *     loads + 2 FMAs. Halves the number of memory instructions in the hot loop
 *     and eliminates the outer d-stride.
 *   - Fold the softmax normalization (division by sum) into the final bf16 write,
 *     so we skip the `s_logits[k] *= inv` pass entirely.
 *   - Earlier tiled-SMEM fusion (Stage-2 v1) regressed ~2x: the per-tile
 *     __syncthreads chopped up the memory pipeline that Stage 1 kept fully
 *     parallelized. Kept the sync-light single-pass design and optimized the
 *     instruction mix instead.
 *
 * Stage-3b: cp.async double-buffered value-projection tile prefetch.
 *   - Stage-3 (head tiling) was a measured negative result: our bottleneck is
 *     NOT K-side HBM bandwidth (effective ~1.6 TB/s vs 7.5 TB/s peak on B200)
 *     but per-block stall time on dependent K loads in the value projection —
 *     every inner iteration had a ~400-cycle serial dependent GMEM load that
 *     the compiler couldn't pipeline across the `if (g < 0) continue;` branch.
 *   - Simpler attempts (branch-free clamp; `#pragma unroll 4`) either wasted
 *     BW on fake loads for short-sequence traces or failed to break the
 *     serial-load dependence.
 *   - Final Stage-3b pipeline: load BK=16 K rows at a time into SMEM with
 *     `cp.async.ca.shared.global` 16-byte copies, double-buffered across two
 *     SMEM tile slots. While the value FMAs for tile `t` run from SMEM, the
 *     next tile's HBM reads (tile `t+1`) are already in flight. This turns
 *     the inner FMA loop into SMEM reads (~20 cycle) backed by an
 *     asynchronous memory pipeline, directly attacking the per-block stall-
 *     bound diagnosis.
 *   - Short-sequence traces: rows with `g < 0` are issued from `ckv_flat[0]`
 *     (safe fallback) and then masked at compute time by `continue`. The
 *     wasted HBM is cached in L2 and does not dominate.
 *
 * Stage-3c: cp.async double-buffered LOGIT tile prefetch.
 *   - The Stage-3b value projection is now ~186 us/trace mean. The remaining
 *     time is dominated by the logit pass (each thread streams its own 8 K
 *     rows from HBM in a serial unrolled dot). We apply the same tile-based
 *     async pipeline: load BK=16 rows of (ckv, kpe) into SMEM with cp.async,
 *     double-buffered, and have 8 warps each compute 2 logits per tile via
 *     warp-reduce over D_CK+D_KP = 576 packed elements from SMEM.
 *   - `s_v_tile` is idle during the logit phase, so we alias it as the
 *     per-tile ckv buffer (same 32 KiB, same layout). Only +4 KiB extra
 *     SMEM for a dedicated kpe tile (2 buffers × 16 rows × 64 bf16).
 *   - The same `n_valid_tiles` truncation as Stage 3b avoids doing cp.async
 *     for all-invalid tiles on short-sequence traces. s_logits is
 *     pre-initialised to -INFINITY so skipped tiles/rows remain correctly
 *     masked for the downstream softmax.
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

// cp.async.ca.shared.global wrappers (inline PTX).
// `smem_int` must come from __cvta_generic_to_shared on the SMEM destination.
// 16 B copies are the widest the instruction supports and match our float4 K layout.
__device__ __forceinline__ void cp_async_16(unsigned smem_int, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_int), "l"(gmem_src));
}

// 8-byte variant: used by the Stage-3c logit pass to stream D_KP=64 bf16 rows
// (128 B / row) in a single issue per thread across 256 threads (16 rows × 16
// 8-byte chunks = 256 chunks per tile).
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

  // SMEM layout:
  //   s_logits  [TOPK]              float   8 KiB
  //   s_qn      [D_CK]              float   2 KiB
  //   s_qp      [D_KP]              float 256 B
  //   s_gidx    [TOPK]              int     8 KiB
  //   s_reduce  [NUM_WARPS]         float  32 B
  //   s_v_tile  [V_PIPE][BK][D_CK]  bf16   32 KiB   (Stage 3b value tile)
  //   s_kpe_tile[V_PIPE][BK][D_KP]  bf16    4 KiB   (Stage 3c logit tile, kpe part)
  //
  // s_v_tile is aliased as the ckv-row buffer during the Stage-3c logit pass
  // (s_v_tile sits idle before the value projection). Total ~54.6 KiB, still
  // under the 100 KiB cudaFuncAttribute cap.
  extern __shared__ unsigned char raw[];
  float* s_logits = reinterpret_cast<float*>(raw);
  float* s_qn = s_logits + TOPK;
  float* s_qp = s_qn + D_CK;
  int* s_gidx = reinterpret_cast<int*>(s_qp + D_KP);
  float* s_reduce = reinterpret_cast<float*>(s_gidx + TOPK);
  // s_reduce is NUM_WARPS floats = 32 B, already 16 B aligned, so the bf16
  // tile region is 16 B aligned for cp.async.
  __nv_bfloat16* s_v_tile = reinterpret_cast<__nv_bfloat16*>(s_reduce + NUM_WARPS);
  __nv_bfloat16* s_kpe_tile = s_v_tile + V_PIPE * BK * D_CK;

  // Load Q into shared as fp32 (reused by every logit).
  for (int d = tid; d < D_CK; d += THREADS) {
    s_qn[d] = bf16_to_f(q_nope[qh_off + d]);
  }
  for (int d = tid; d < D_KP; d += THREADS) {
    s_qp[d] = bf16_to_f(q_pe[qp_off + d]);
  }
  // Cache sparse_idx (clamped to valid range) so the value-projection pass does not re-read global.
  // Also track the maximum k-position that has any valid gather index. For
  // short sequences (many -1 entries), the value-projection tile loop can
  // truncate at that position rather than always walking all NUM_V_TILES.
  int local_max_valid_k_p1 = 0;  // stored as k+1 so "none valid" reduces to 0
  for (int k = tid; k < TOPK; k += THREADS) {
    int32_t g = sparse_idx[si_off + k];
    const bool valid = (g >= 0 && static_cast<int64_t>(g) < num_kv_rows);
    s_gidx[k] = valid ? g : -1;
    if (valid && (k + 1) > local_max_valid_k_p1) local_max_valid_k_p1 = k + 1;
  }
  __syncthreads();
  const float reduced_max = block_reduce_max(static_cast<float>(local_max_valid_k_p1), s_reduce);
  const int n_valid_rows = static_cast<int>(reduced_max);
  int n_valid_tiles = (n_valid_rows + BK - 1) / BK;
  if (n_valid_tiles > NUM_V_TILES) n_valid_tiles = NUM_V_TILES;

  // Stage 3c: cp.async double-buffered logit pass.
  //
  //   * Pre-initialise s_logits[k] = -INFINITY so any k we don't compute
  //     (invalid g or tile beyond n_valid_tiles) is correctly masked for the
  //     downstream softmax max.
  //   * Tile layout: 16 K rows per tile. For each row we cp.async both the
  //     ckv (1024 B) and the kpe (128 B) halves into SMEM. s_v_tile is
  //     aliased as the per-tile ckv buffer (idle until the value projection).
  //     s_kpe_tile is a small (4 KiB) dedicated region for the kpe halves.
  //   * Compute: each warp owns 2 logits per tile (BK=16 / NUM_WARPS=8 = 2),
  //     computed via `warp_reduce_sum` over 32 lanes × (8 packed bf162 from
  //     ckv + 1 packed bf162 from kpe) = full D_CK+D_KP dot.
  //   * Every thread's 32-bit reads of s_qn / s_qp / the tile buffers land
  //     on distinct banks (stride-1 lane access to bf162 = 4 B / lane).

  // Pre-init logits to -INF so skipped tiles/rows are masked at softmax time.
  for (int k = tid; k < TOPK; k += THREADS) {
    s_logits[k] = -INFINITY;
  }
  __syncthreads();

  auto issue_logit_tile = [&](int tile, int buf) {
    const int k0 = tile * BK;
    __nv_bfloat16* ckv_buf = s_v_tile + buf * (BK * D_CK);
    __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);

    // ckv half: 4 × 16 B per thread (identical decomposition to Stage 3b).
    #pragma unroll
    for (int c = 0; c < CHUNKS_PER_THREAD; ++c) {
      const int chunk = tid + c * THREADS;           // 0..1023
      const int r = chunk / (D_CK / 8);              // chunk/64 -> row in tile
      const int ch16 = chunk % (D_CK / 8);           // chunk%64 -> 16 B col
      const int k = k0 + r;
      int g = s_gidx[k];
      if (g < 0) g = 0;  // safe fallback; compute-time `continue` masks it
      const uint4* src =
          reinterpret_cast<const uint4*>(ckv_flat + static_cast<int64_t>(g) * D_CK) + ch16;
      uint4* dst = reinterpret_cast<uint4*>(ckv_buf + r * D_CK) + ch16;
      cp_async_16(__cvta_generic_to_shared(dst), src);
    }

    // kpe half: 1 × 8 B per thread. Per tile: 16 rows × (64 bf16 / 4) = 256
    // 8-byte chunks = exactly THREADS chunks, so one cp.async per thread.
    {
      const int chunk = tid;                          // 0..255
      const int r = chunk / (D_KP / 4);               // chunk/16 -> row in tile
      const int ch8 = chunk % (D_KP / 4);             // chunk%16 -> 8 B col
      const int k = k0 + r;
      int g = s_gidx[k];
      if (g < 0) g = 0;
      const uint2* src =
          reinterpret_cast<const uint2*>(kpe_flat + static_cast<int64_t>(g) * D_KP) + ch8;
      uint2* dst = reinterpret_cast<uint2*>(kpe_buf + r * D_KP) + ch8;
      cp_async_8(__cvta_generic_to_shared(dst), src);
    }
  };

  if (n_valid_tiles > 0) {
    issue_logit_tile(0, 0);
    cp_async_commit_group();

    const int warp_id = tid >> 5;       // 0..NUM_WARPS-1 (8)
    const int lane = tid & (WARP - 1);  // 0..31

    for (int tile = 0; tile < n_valid_tiles; ++tile) {
      const int buf = tile & 1;
      const int next_tile = tile + 1;

      if (next_tile < n_valid_tiles) {
        issue_logit_tile(next_tile, next_tile & 1);
        cp_async_commit_group();
        cp_async_wait_group<1>();
      } else {
        cp_async_wait_group<0>();
      }
      __syncthreads();

      const __nv_bfloat16* ckv_buf = s_v_tile + buf * (BK * D_CK);
      const __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);

      #pragma unroll
      for (int wr = 0; wr < 2; ++wr) {               // 2 logits / warp
        const int r = warp_id * 2 + wr;              // 0..15
        const int k = tile * BK + r;
        const int g = s_gidx[k];
        if (g < 0) continue;  // s_logits[k] is already -INFINITY

        const __nv_bfloat162* kc2 =
            reinterpret_cast<const __nv_bfloat162*>(ckv_buf + r * D_CK);
        const __nv_bfloat162* kp2 =
            reinterpret_cast<const __nv_bfloat162*>(kpe_buf + r * D_KP);

        float acc = 0.f;
        // D_CK / 2 / WARP = 8 packed bf162 per lane across D_CK.
        #pragma unroll
        for (int j = 0; j < D_CK / 2 / WARP; ++j) {
          const int i2 = j * WARP + lane;
          const float2 kf = __bfloat1622float2(kc2[i2]);
          acc += s_qn[2 * i2] * kf.x + s_qn[2 * i2 + 1] * kf.y;
        }
        // D_KP / 2 = 32 packed bf162 = exactly one per lane.
        {
          const float2 kf = __bfloat1622float2(kp2[lane]);
          acc += s_qp[2 * lane] * kf.x + s_qp[2 * lane + 1] * kf.y;
        }

        acc = warp_reduce_sum(acc);
        if (lane == 0) s_logits[k] = acc * sm_scale_log2e;
      }
      __syncthreads();
    }
  }

  // Block-parallel max.
  float local_max = -INFINITY;
  for (int k = tid; k < TOPK; k += THREADS) {
    local_max = fmaxf(local_max, s_logits[k]);
  }
  const float mx = block_reduce_max(local_max, s_reduce);

  // All-masked row handling: write zeros and LSE=-inf, nothing else to compute.
  if (mx == -INFINITY) {
    if (tid == 0) lse_out[lse_i] = -INFINITY;
    const __nv_bfloat162 zero =
        __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    reinterpret_cast<__nv_bfloat162*>(out + out_off)[tid] = zero;
    return;
  }

  // Block-parallel unnormalized softmax (base-2 exp). We keep s_logits as
  // unnormalized probs and divide by sum at output write time.
  float local_sum = 0.f;
  for (int k = tid; k < TOPK; k += THREADS) {
    const float e = exp2f(s_logits[k] - mx);
    s_logits[k] = e;
    local_sum += e;
  }
  const float sum = block_reduce_sum(local_sum, s_reduce);

  if (tid == 0) lse_out[lse_i] = mx + log2f(sum);

  // Weighted cKV projection — cp.async double-buffered tile prefetch.
  //
  // Structure:
  //   * NUM_V_TILES = TOPK / BK (128) tiles of K rows, BK=16 rows each.
  //   * Two SMEM buffers of shape [BK, D_CK] bf16. At any moment buffer
  //     `tile & 1` holds the K rows the inner FMA loop is consuming, while
  //     buffer `(tile + 1) & 1` is being filled by in-flight cp.async loads.
  //   * `cp_async_commit_group` + `cp_async_wait_group<1>` keep exactly one
  //     tile-load in flight at a time: we always wait for the older of the
  //     two outstanding groups before compute.
  //   * For rows with `g < 0`, we issue the cp.async from `ckv_flat[0]` as a
  //     safe fallback (avoids illegal addresses) and skip compute via
  //     `if (g < 0) continue;` inside the BK loop. The fallback load is
  //     cheap (L2 hit) and happens only for short-sequence traces.
  //
  // Per-tile issue: each thread launches CHUNKS_PER_THREAD=4 16-byte
  // cp.async copies, covering the whole 16 KiB tile (1024 × 16 B chunks
  // across 256 threads, 4 per thread).
  static_assert(D_CK == 2 * THREADS, "packed output path requires D_CK == 2*THREADS");

  auto issue_v_tile = [&](int tile, int buf) {
    const int k0 = tile * BK;
    __nv_bfloat16* buf_base = s_v_tile + buf * (BK * D_CK);
    #pragma unroll
    for (int c = 0; c < CHUNKS_PER_THREAD; ++c) {
      const int chunk = tid + c * THREADS;          // 0..TILE_CHUNKS_16B-1
      const int r = chunk / (D_CK / 8);              // chunk/64 -> row in tile
      const int ch16 = chunk % (D_CK / 8);           // chunk%64 -> 16-byte col
      const int k = k0 + r;
      int g = s_gidx[k];
      if (g < 0) g = 0;  // safe fallback; masked at compute time
      const uint4* src =
          reinterpret_cast<const uint4*>(ckv_flat + static_cast<int64_t>(g) * D_CK) + ch16;
      uint4* dst = reinterpret_cast<uint4*>(buf_base + r * D_CK) + ch16;
      const unsigned smem_int = __cvta_generic_to_shared(dst);
      cp_async_16(smem_int, src);
    }
  };

  const int i = tid;
  const float inv_sum = 1.f / sum;
  float o0 = 0.f, o1 = 0.f;

  // For short-sequence traces we may have 0 valid rows (but the all-masked
  // case was already handled above by the `mx == -INFINITY` early return, so
  // n_valid_tiles >= 1 here). Still guard defensively.
  if (n_valid_tiles > 0) {
    // Prime: launch tile 0 into buffer 0.
    issue_v_tile(0, 0);
    cp_async_commit_group();

    for (int tile = 0; tile < n_valid_tiles; ++tile) {
      const int buf = tile & 1;
      const int next_tile = tile + 1;

      // Kick off the next tile before waiting for the current one so its HBM
      // reads overlap with our FMAs.
      if (next_tile < n_valid_tiles) {
        issue_v_tile(next_tile, next_tile & 1);
        cp_async_commit_group();
        cp_async_wait_group<1>();  // leave the just-issued group in flight
      } else {
        cp_async_wait_group<0>();  // drain: this is the last tile
      }
      __syncthreads();  // make this tile's cp.async writes visible to all threads

      const int k0 = tile * BK;
      const __nv_bfloat162* tile_pairs =
          reinterpret_cast<const __nv_bfloat162*>(s_v_tile + buf * (BK * D_CK));
      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        const int k = k0 + r;
        const int g = s_gidx[k];
        if (g < 0) continue;
        const float2 kf = __bfloat1622float2(tile_pairs[r * (D_CK / 2) + i]);
        const float p = s_logits[k];
        o0 += p * kf.x;
        o1 += p * kf.y;
      }
      __syncthreads();  // ensure all threads done with this buffer before next issue reuses it
    }
  }

  const __nv_bfloat162 packed =
      __halves2bfloat162(f_to_bf16(o0 * inv_sum), f_to_bf16(o1 * inv_sum));
  reinterpret_cast<__nv_bfloat162*>(out + out_off)[i] = packed;
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
  // s_logits + s_qn + s_qp + s_gidx + s_reduce + s_v_tile + s_kpe_tile
  //   s_v_tile  : V_PIPE * BK * D_CK bf16 (32 KiB, Stage 3b)
  //   s_kpe_tile: V_PIPE * BK * D_KP bf16 ( 4 KiB, Stage 3c)
  const size_t shmem =
      (static_cast<size_t>(TOPK) + D_CK + D_KP) * sizeof(float)
      + static_cast<size_t>(TOPK) * sizeof(int)
      + static_cast<size_t>(NUM_WARPS) * sizeof(float)
      + static_cast<size_t>(V_PIPE) * BK * D_CK * sizeof(__nv_bfloat16)
      + static_cast<size_t>(V_PIPE) * BK * D_KP * sizeof(__nv_bfloat16);

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
