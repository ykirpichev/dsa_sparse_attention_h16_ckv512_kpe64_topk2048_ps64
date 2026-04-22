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

// diag02: set to 1 to enable in-kernel phase cycle timers. Block (0,0) prints
// one line of cycle counts per kernel launch. Leave at 0 for benchmark runs.
#ifndef PROFILE_PHASES
#define PROFILE_PHASES 0
#endif

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
  long long ts_c0 = 0, ts_c1 = 0, ts_c2 = 0, ts_c3 = 0, ts_c4 = 0, ts_c5 = 0;
  PHASE_MARK(ts_c0);

  const int64_t qh_off = (t * H + h) * D_CK;
  const int64_t qp_off = (t * H + h) * D_KP;
  const int64_t si_off = t * TOPK;
  const int64_t out_off = (t * H + h) * D_CK;
  const int64_t lse_i = t * H + h;

  // SMEM layout (v5/exp06: Q stored in SMEM as bf16, not fp32):
  //   s_logits  [TOPK]              float   8 KiB
  //   s_qn      [D_CK]              bf16    1 KiB  (was fp32, 2 KiB)
  //   s_qp      [D_KP]              bf16  128 B    (was fp32, 256 B)
  //   s_gidx    [TOPK]              int     8 KiB
  //   s_reduce  [NUM_WARPS]         float  32 B
  //   s_v_tile  [V_PIPE][BK][D_CK]  bf16   32 KiB
  //   s_kpe_tile[V_PIPE][BK][D_KP]  bf16    4 KiB
  //
  // Motivation: the logit hot loop does one SMEM read of s_qn per FMA pair.
  // Storing Q as bf16 halves those reads (one LDS.U.32 per lane instead of
  // one LDS.U.64), easing SMEM-bw pressure. The bf162→float2 convert is
  // cheap (CVT2BF2F). Total saving: ~1.125 KiB SMEM.
  extern __shared__ unsigned char raw[];
  float* s_logits = reinterpret_cast<float*>(raw);
  __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(s_logits + TOPK);
  __nv_bfloat16* s_qp = s_qn + D_CK;
  int* s_gidx = reinterpret_cast<int*>(s_qp + D_KP);
  float* s_reduce = reinterpret_cast<float*>(s_gidx + TOPK);
  __nv_bfloat16* s_v_tile = reinterpret_cast<__nv_bfloat16*>(s_reduce + NUM_WARPS);
  __nv_bfloat16* s_kpe_tile = s_v_tile + V_PIPE * BK * D_CK;

  // Load Q into shared as bf16 (direct copy, no conversion).
  for (int d = tid; d < D_CK; d += THREADS) {
    s_qn[d] = q_nope[qh_off + d];
  }
  for (int d = tid; d < D_KP; d += THREADS) {
    s_qp[d] = q_pe[qp_off + d];
  }
  // Cache sparse_idx (clamped to valid range) so the value-projection pass does not re-read global.
  // Also track the maximum k-position that has any valid gather index. For
  // short sequences (many -1 entries), the value-projection tile loop can
  // truncate at that position rather than always walking all NUM_V_TILES.
  //
  // itemB-attempted: we tried replacing the block_reduce_max below with a
  // warp-reduced + atomicMax to a single shared int slot (saves 2 syncs).
  // Smoke phase timers showed −3.4% qload and −2.3% total, but the full
  // bench regressed by ~2.5% consistently across 2 runs — long-trace
  // workloads all went from 152-155us → 157-160us. Reason unclear (regs
  // identical, CTAs/SM identical); best guess is that the removed syncs
  // were serving as implicit pipeline barriers for HBM scheduling, and
  // without them some downstream stall worsens. Change reverted; see
  // `artifacts/full_bench_itemB_warpatomic_2026-04-22.log` for data.
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
  //   * diag02 (v5_qload/itemA): we no longer pre-initialise s_logits to
  //     -INFINITY. Instead: (a) the logit compute below writes -INFINITY
  //     inline for rows with g<0, and (b) the softmax loops are bounded
  //     by n_valid_rows_full = n_valid_tiles*BK (not TOPK) so untouched
  //     tail slots never participate. This removes the 8 writes/thread
  //     pre-init pass + 1 __syncthreads from the prologue and tightens
  //     the softmax reductions from 2048 → as few as 16 iterations.
  //     diag02 showed qload dominating short-trace runtime (~54% of
  //     cycles at n_valid_tiles<4); net win ~5% on short traces.
  //   * Tile layout: 16 K rows per tile. For each row we cp.async both the
  //     ckv (1024 B) and the kpe (128 B) halves into SMEM. s_v_tile is
  //     aliased as the per-tile ckv buffer (idle until the value projection).
  //     s_kpe_tile is a small (4 KiB) dedicated region for the kpe halves.
  //   * Compute: each warp owns 2 logits per tile (BK=16 / NUM_WARPS=8 = 2),
  //     computed via `warp_reduce_sum` over 32 lanes × (8 packed bf162 from
  //     ckv + 1 packed bf162 from kpe) = full D_CK+D_KP dot.
  //   * Every thread's 32-bit reads of s_qn / s_qp / the tile buffers land
  //     on distinct banks (stride-1 lane access to bf162 = 4 B / lane).
  PHASE_MARK(ts_c1);  // end of Q+idx load phase

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

      const __nv_bfloat162* qn2 =
          reinterpret_cast<const __nv_bfloat162*>(s_qn);
      const __nv_bfloat162* qp2 =
          reinterpret_cast<const __nv_bfloat162*>(s_qp);

      #pragma unroll
      for (int wr = 0; wr < 2; ++wr) {               // 2 logits / warp
        const int r = warp_id * 2 + wr;              // 0..15
        const int k = tile * BK + r;

        // itemA: always run the dot (the cp.async pass already loaded
        // ckv_flat[0] as a safe fallback for g<0), then select -INF at the
        // write. This keeps the hot path free of warp-uniform branches and
        // preserves instruction scheduling for the 9-iter packed-bf16 dot.
        // `g` is deliberately loaded inside the lane-0 write to avoid
        // keeping a register live across the whole FMA chain.
        const __nv_bfloat162* kc2 =
            reinterpret_cast<const __nv_bfloat162*>(ckv_buf + r * D_CK);
        const __nv_bfloat162* kp2 =
            reinterpret_cast<const __nv_bfloat162*>(kpe_buf + r * D_KP);

        float acc = 0.f;
        // D_CK / 2 / WARP = 8 packed bf162 per lane across D_CK. Both Q and K
        // are now bf162 in SMEM; we load one 32-bit word per lane per operand.
        #pragma unroll
        for (int j = 0; j < D_CK / 2 / WARP; ++j) {
          const int i2 = j * WARP + lane;
          const float2 kf = __bfloat1622float2(kc2[i2]);
          const float2 qf = __bfloat1622float2(qn2[i2]);
          acc += qf.x * kf.x + qf.y * kf.y;
        }
        // D_KP / 2 = 32 packed bf162 = exactly one per lane.
        {
          const float2 kf = __bfloat1622float2(kp2[lane]);
          const float2 qf = __bfloat1622float2(qp2[lane]);
          acc += qf.x * kf.x + qf.y * kf.y;
        }

        acc = warp_reduce_sum(acc);
        if (lane == 0) {
          const int g = s_gidx[k];
          s_logits[k] = (g >= 0) ? (acc * sm_scale_log2e) : -INFINITY;
        }
      }
      __syncthreads();
    }
  }

  PHASE_MARK(ts_c2);  // end of logit pass

  // Block-parallel max. Bounded by n_valid_rows_full (= n_valid_tiles*BK)
  // instead of TOPK: the logit pass is guaranteed to have written every
  // s_logits[k] for k in [0, n_valid_rows_full) — either the warp-reduced
  // logit for g >= 0 rows, or -INFINITY for g < 0 rows (itemA writes -INF
  // inline). Everything at k >= n_valid_rows_full is untouched/invalid and
  // must not be read. This collapses short-trace softmax from 2048 → as
  // few as 16 iterations total.
  const int n_valid_rows_full = n_valid_tiles * BK;
  float local_max = -INFINITY;
  for (int k = tid; k < n_valid_rows_full; k += THREADS) {
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

  // Block-parallel unnormalized softmax (base-2 exp). Same tight loop
  // bound as the max reduction. For g < 0 rows the logit pass wrote -INF,
  // so exp2f(-INF - mx) = 0 contributes nothing — we can skip the explicit
  // gidx check entirely. We keep s_logits as unnormalised probs and
  // divide by sum at output write time.
  float local_sum = 0.f;
  for (int k = tid; k < n_valid_rows_full; k += THREADS) {
    const float e = exp2f(s_logits[k] - mx);
    s_logits[k] = e;
    local_sum += e;
  }
  const float sum = block_reduce_sum(local_sum, s_reduce);

  if (tid == 0) lse_out[lse_i] = mx + log2f(sum);
  PHASE_MARK(ts_c3);  // end of softmax

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
  //
  // itemC-attempted: we tried pre-issuing tile 0 at the end of the logit
  // pass so its HBM latency would hide behind softmax. Result was a net
  // +2 % regression: the ~100-cycle cp.async issue cost moved from the
  // vproj phase into the softmax phase (softmax +23 % short-trace), while
  // the theoretical HBM-latency savings only materialised for n_valid_tiles
  // == 1 traces (vproj −15 % short-trace). Long traces regressed outright
  // because their tile-0 wait was already cheap. See
  // `artifacts/smoke_qload_itemC_phases.log` for the data; change reverted.
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

  PHASE_MARK(ts_c4);  // end of value projection (before output write)
  const __nv_bfloat162 packed =
      __halves2bfloat162(f_to_bf16(o0 * inv_sum), f_to_bf16(o1 * inv_sum));
  reinterpret_cast<__nv_bfloat162*>(out + out_off)[i] = packed;
  PHASE_MARK(ts_c5);  // end of kernel
  PHASE_PRINT(ts_c0, ts_c1, ts_c2, ts_c3, ts_c4, ts_c5,
              gridDim.x, H, n_valid_tiles);
}

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

// Dynamic SMEM cap sized exactly to actual kernel usage (~54 KiB for
// v5/exp06). A tight cap lets the GPU schedule 4 CTAs/SM (B200: 228 KiB
// SMEM/SM → 4 × 54 = 216 KiB fits). The previous 100 KiB cap forced the
// scheduler down to 2 CTAs/SM even though the kernel never used more than
// half of that budget.
//
// Actual SMEM per CTA at launch:
//   s_logits  (TOPK * fp32)             =  8192 B
//   s_qn + s_qp ((D_CK+D_KP) * bf16)    =  1152 B
//   s_gidx    (TOPK * int32)            =  8192 B
//   s_reduce  (NUM_WARPS * fp32)        =    32 B
//   s_v_tile  (V_PIPE*BK*D_CK * bf16)   = 32768 B
//   s_kpe_tile(V_PIPE*BK*D_KP * bf16)   =  4096 B
//   Total ≈ 54432 B ≈ 53.2 KiB.
constexpr int KERNEL_MAX_DYN_SMEM = 55 * 1024;

void configure_kernel_attrs_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    cudaFuncSetAttribute(mla_sparse_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         KERNEL_MAX_DYN_SMEM);

    // Log register / SMEM / occupancy once per process so we can verify
    // future optimization experiments against this baseline.
    cudaFuncAttributes attrs{};
    cudaFuncGetAttributes(&attrs, (const void*)mla_sparse_attn_kernel);
    int occ_full_cap = 0;
    int occ_actual = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_full_cap, mla_sparse_attn_kernel, THREADS, KERNEL_MAX_DYN_SMEM);
    // ~54 KiB is the actual launch-time footprint for v5/exp06.
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_actual, mla_sparse_attn_kernel, THREADS, 54 * 1024);
    fprintf(stderr,
            "[mla_sparse_attn] regs/thread=%d  localSz=%d  "
            "static_smem=%zu  max_dyn_smem=%d  max_threads=%d  "
            "occ@cap=%d  occ@actual(54K)=%d blocks/SM\n",
            attrs.numRegs, (int)attrs.localSizeBytes,
            (size_t)attrs.sharedSizeBytes, attrs.maxDynamicSharedSizeBytes,
            attrs.maxThreadsPerBlock, occ_full_cap, occ_actual);
  });
}

}  // namespace

// TVM-FFI entry point. The exported symbol name is `kernel`, matching
// `entry_point = "kernel.cu::kernel"` in config.toml.
//
// Destination-passing style: output tensors (`out`, `lse`) are pre-allocated
// by the flashinfer-bench Runnable from the definition's output spec and
// passed in after the inputs. See `flashinfer_bench/compile/runnable.py`.
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

  // Device checks.
  TVM_FFI_ICHECK_EQ(q_nope.device().device_type, kDLCUDA) << "q_nope must be CUDA";
  TVM_FFI_ICHECK_EQ(q_pe.device().device_type, kDLCUDA) << "q_pe must be CUDA";
  TVM_FFI_ICHECK_EQ(ckv_cache.device().device_type, kDLCUDA) << "ckv_cache must be CUDA";
  TVM_FFI_ICHECK_EQ(kpe_cache.device().device_type, kDLCUDA) << "kpe_cache must be CUDA";
  TVM_FFI_ICHECK_EQ(sparse_indices.device().device_type, kDLCUDA) << "sparse_indices must be CUDA";
  TVM_FFI_ICHECK_EQ(out.device().device_type, kDLCUDA) << "out must be CUDA";
  TVM_FFI_ICHECK_EQ(lse.device().device_type, kDLCUDA) << "lse must be CUDA";

  // Dtype checks (bf16 for tensors, int32 for indices, fp32 for lse).
  TVM_FFI_ICHECK_EQ(q_nope.dtype().code, kDLBfloat) << "q_nope must be bfloat16";
  TVM_FFI_ICHECK_EQ(q_nope.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(q_pe.dtype().code, kDLBfloat) << "q_pe must be bfloat16";
  TVM_FFI_ICHECK_EQ(q_pe.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(ckv_cache.dtype().code, kDLBfloat) << "ckv_cache must be bfloat16";
  TVM_FFI_ICHECK_EQ(ckv_cache.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(kpe_cache.dtype().code, kDLBfloat) << "kpe_cache must be bfloat16";
  TVM_FFI_ICHECK_EQ(kpe_cache.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(sparse_indices.dtype().code, kDLInt) << "sparse_indices must be int32";
  TVM_FFI_ICHECK_EQ(sparse_indices.dtype().bits, 32);
  TVM_FFI_ICHECK_EQ(out.dtype().code, kDLBfloat) << "out must be bfloat16";
  TVM_FFI_ICHECK_EQ(out.dtype().bits, 16);
  TVM_FFI_ICHECK_EQ(lse.dtype().code, kDLFloat) << "lse must be float32";
  TVM_FFI_ICHECK_EQ(lse.dtype().bits, 32);

  // Shape checks: q_nope[T,H,D_CK], q_pe[T,H,D_KP], sparse_indices[T,TOPK].
  TVM_FFI_ICHECK_EQ(q_nope.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q_pe.ndim(), 3);
  TVM_FFI_ICHECK_EQ(sparse_indices.ndim(), 2);
  const int64_t T = q_nope.size(0);
  const int64_t H = q_nope.size(1);
  TVM_FFI_ICHECK_EQ(q_nope.size(2), D_CK);
  TVM_FFI_ICHECK_EQ(q_pe.size(0), T);
  TVM_FFI_ICHECK_EQ(q_pe.size(1), H);
  TVM_FFI_ICHECK_EQ(q_pe.size(2), D_KP);
  TVM_FFI_ICHECK_EQ(sparse_indices.size(0), T);
  TVM_FFI_ICHECK_EQ(sparse_indices.size(1), TOPK);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), T);
  TVM_FFI_ICHECK_EQ(out.size(1), H);
  TVM_FFI_ICHECK_EQ(out.size(2), D_CK);
  TVM_FFI_ICHECK_EQ(lse.ndim(), 2);
  TVM_FFI_ICHECK_EQ(lse.size(0), T);
  TVM_FFI_ICHECK_EQ(lse.size(1), H);

  // Flat row count for the caches. We require the trailing dim to be D_CK
  // (resp. D_KP) so a flat [num_kv_rows, D_last] view of contiguous memory
  // is valid without an explicit reshape.
  TVM_FFI_ICHECK_GE(ckv_cache.ndim(), 1);
  TVM_FFI_ICHECK_EQ(ckv_cache.size(ckv_cache.ndim() - 1), D_CK);
  TVM_FFI_ICHECK_GE(kpe_cache.ndim(), 1);
  TVM_FFI_ICHECK_EQ(kpe_cache.size(kpe_cache.ndim() - 1), D_KP);
  int64_t ckv_numel = 1;
  for (int i = 0; i < ckv_cache.ndim(); ++i) ckv_numel *= ckv_cache.size(i);
  int64_t kpe_numel = 1;
  for (int i = 0; i < kpe_cache.ndim(); ++i) kpe_numel *= kpe_cache.size(i);
  const int64_t num_kv_rows = ckv_numel / D_CK;
  TVM_FFI_ICHECK_EQ(kpe_numel / D_KP, num_kv_rows) << "ckv/kpe row counts must match";

  const float sm = static_cast<float>(sm_scale);
  const float sm_log2e = sm * LOG2E_F;

  const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
  const dim3 block(THREADS);
  // s_logits + s_qn + s_qp + s_gidx + s_reduce + s_v_tile + s_kpe_tile
  //   s_v_tile  : V_PIPE * BK * D_CK bf16 (32 KiB, Stage 3b)
  //   s_kpe_tile: V_PIPE * BK * D_KP bf16 ( 4 KiB, Stage 3c)
  // v5/exp06: s_qn / s_qp now bf16 (not fp32), saving 1.125 KiB per CTA.
  const size_t shmem =
      static_cast<size_t>(TOPK) * sizeof(float)
      + static_cast<size_t>(D_CK + D_KP) * sizeof(__nv_bfloat16)
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
