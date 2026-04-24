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
  // Grid-order trade-off (ATTEMPTED, REVERTED):
  // Tried dim3(H, T) so all 16 heads of the same token are adjacent in
  // launch order — the theory being that same-token CTAs gather the exact
  // same ckv/kpe rows from HBM (sparse_indices is per-token, no head dim)
  // so co-locating them gives H× L2 reuse on ckv. Full bench showed
  // +1.70 % absolute latency geomean (consistent +1-2 % on all long
  // traces). Probable cause: forcing 3 CTAs/SM to all want the same
  // HBM addresses saturates the SM's MIO / L2 request path, and the
  // original dim3(T, H) order already gets good L2 reuse via the
  // hardware grid scheduler's tile-swizzle. The "+20 % speedup" seen in
  // the sp column was pure FI-reference timing drift between Modal
  // sessions (ref latency varied 15-30 % between runs).
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

  // SMEM layout (v5_fused: Idea 5 — fused single-pass K loop, no s_logits):
  //   s_logits_tile [BK]             float   64 B   (was s_logits[TOPK] = 8 KiB)
  //   s_qn          [D_CK]           bf16    1 KiB
  //   s_qp          [D_KP]           bf16    128 B
  //   s_gidx        [TOPK]           int     8 KiB
  //   s_reduce      [NUM_WARPS]      float   32 B
  //   s_v_tile      [V_PIPE][BK][D_CK]  bf16 32 KiB
  //   s_kpe_tile    [V_PIPE][BK][D_KP]  bf16  4 KiB
  //   TOTAL ≈ 46 KiB (was ~54 KiB) — opens 8 KiB of headroom for future
  //   V_PIPE=3 / deeper pipelines.
  //
  // Rationale: the previous kernel loaded each ckv tile TWICE from HBM
  // (once for the logit dot, once for the weighted-V accumulate) because
  // the 8 KiB s_logits buffer forced a separate logit → softmax → vproj
  // phase split. With online softmax we keep per-thread running state
  // (m, l, O) and update it as we see each tile's 16 logits, halving
  // ckv HBM traffic.
  extern __shared__ unsigned char raw[];
  float* s_logits_tile = reinterpret_cast<float*>(raw);           // [BK] fp32
  __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(s_logits_tile + BK);
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

  // Fused single-pass K loop (Idea 5, v5_fused).
  //
  // Each iteration loads ONE ckv+kpe tile (16 rows × D_CK+D_KP bf16), computes
  // the 16 logits via the same warp-cooperative dot as the logit pass, then
  // updates the per-thread running (m, l, O) state (online softmax) using the
  // same ckv data — so each HBM ckv byte is consumed exactly once per token.
  //
  // Per-tile SMEM primitives:
  //   * s_v_tile   — ping-pong buffer [V_PIPE=2][BK=16][D_CK=512] bf16.
  //   * s_kpe_tile — ping-pong buffer [V_PIPE=2][BK=16][D_KP=64]  bf16.
  //   * s_logits_tile — tiny [BK=16] fp32 scratch for cross-warp logit share.
  //
  // Per-tile sync topology (just 2 syncs instead of 4 in the previous split):
  //   wait<1> (or wait<0> on last)
  //   SYNC_A   — publish cp.async writes and implicit barrier-before-next-
  //              iter's s_logits_tile write (iter+1 won't overwrite
  //              s_logits_tile until after its own SYNC_A, by which point
  //              all threads in this iter have finished reading it).
  //   <logit compute, warp-parallel, writes s_logits_tile[0..15]>
  //   SYNC_B   — publish s_logits_tile writes
  //   <softmax-rescale + V-accumulate, all 256 threads, writes regs m/l/O>
  //   (no SYNC_C — see SYNC_A comment)
  //
  // Buffer-reuse dependence (identical to the old logit / vproj loops):
  //   Iter+1's cp.async writes to (tile+1)&1 ≠ tile&1, so no conflict with
  //   iter's compute that reads buf = tile&1. Iter+2's cp.async writes back
  //   to tile&1, which is safe because iter+1's SYNC_A (block-wide) is the
  //   barrier-before-write for that reuse.
  //
  // Early-exit: n_valid_tiles == 0 means the token has no valid gather
  // indices at all — write zeros / lse=-INF and return.
  PHASE_MARK(ts_c1);  // end of Q+idx load phase

  if (n_valid_tiles == 0) {
    if (tid == 0) lse_out[lse_i] = -INFINITY;
    const __nv_bfloat162 zero =
        __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    reinterpret_cast<__nv_bfloat162*>(out + out_off)[tid] = zero;
    return;
  }

  // Single ckv+kpe tile issuer (identical to the previous issue_logit_tile).
  auto issue_tile = [&](int tile, int buf) {
    const int k0 = tile * BK;
    __nv_bfloat16* ckv_buf = s_v_tile + buf * (BK * D_CK);
    __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);

    // ckv half: 4 × 16 B per thread = 1024 16 B chunks = full 16 KiB tile.
    #pragma unroll
    for (int c = 0; c < CHUNKS_PER_THREAD; ++c) {
      const int chunk = tid + c * THREADS;
      const int r = chunk / (D_CK / 8);
      const int ch16 = chunk % (D_CK / 8);
      const int k = k0 + r;
      int g = s_gidx[k];
      if (g < 0) g = 0;  // safe fallback; masked at compute time via -INF logit
      const uint4* src =
          reinterpret_cast<const uint4*>(ckv_flat + static_cast<int64_t>(g) * D_CK) + ch16;
      uint4* dst = reinterpret_cast<uint4*>(ckv_buf + r * D_CK) + ch16;
      cp_async_16(__cvta_generic_to_shared(dst), src);
    }

    // kpe half: 1 × 8 B per thread (exactly THREADS 8 B chunks per tile).
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

  // Per-thread running softmax+output state.
  float m = -INFINITY;  // running max over all logits seen so far
  float l = 0.f;        // running sum of exp2(logit - m)
  float o0 = 0.f, o1 = 0.f;  // running weighted V for the two output channels
                             // (2*tid, 2*tid+1) owned by this thread

  issue_tile(0, 0);
  cp_async_commit_group();

  const int warp_id = tid >> 5;       // 0..NUM_WARPS-1 (8)
  const int lane = tid & (WARP - 1);  // 0..31
  const int i = tid;                  // output channel pair owner

  const __nv_bfloat162* qn2 =
      reinterpret_cast<const __nv_bfloat162*>(s_qn);
  const __nv_bfloat162* qp2 =
      reinterpret_cast<const __nv_bfloat162*>(s_qp);

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
    __syncthreads();  // SYNC_A

    const __nv_bfloat16* ckv_buf = s_v_tile + buf * (BK * D_CK);
    const __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);

    // --- Phase 1: logit compute (warp-parallel, 2 dots per warp) ---
    //
    // Exactly the same compute as the old logit pass, writing to a 16-entry
    // scratch instead of s_logits[k] (no more indexing by global k).
    #pragma unroll
    for (int wr = 0; wr < 2; ++wr) {
      const int r = warp_id * 2 + wr;
      const int k = tile * BK + r;
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
        s_logits_tile[r] = (g >= 0) ? (acc * sm_scale_log2e) : -INFINITY;
      }
    }
    __syncthreads();  // SYNC_B

    // --- Phase 2: online-softmax rescale + V accumulate ---
    //
    // All 256 threads do the same scalar reduction over the 16 logits
    // (broadcast read from the same SMEM bank-slot gives no bank conflict —
    // it's a warp-wide broadcast). Each thread owns output channels
    // (2*tid, 2*tid+1) and walks through BK=16 V-rows in parallel.
    float tile_max = -INFINITY;
    #pragma unroll
    for (int r = 0; r < BK; ++r) {
      tile_max = fmaxf(tile_max, s_logits_tile[r]);
    }

    // Skip fully-masked tiles (all 16 rows had g<0) — otherwise exp2(-INF -
    // (-INF)) = NaN would poison the accumulators. This branch is block-
    // uniform (tile_max is identical across all threads) so zero divergence.
    //
    // Idea 7 ATTEMPTED (REVERTED): wrapping the rescale in
    //     `if (tile_max > m) { m_new = tile_max; ... }`
    // to skip the 3× fp32 mults when m doesn't grow. Full bench showed
    // +0.35 % geomean latency regression (see
    // artifacts/full_bench_fused_condresc_2026-04-22.log). Reason: the
    // extra block-uniform branch splits the CFG of the tight inner loop
    // and inhibits the compiler's instruction scheduling. Keeping the
    // unconditional rescale — 3 fp32 mults per thread are free next to
    // a 16-iter exp2f + FMA chain.
    if (tile_max > -INFINITY) {
      const float m_new = fmaxf(m, tile_max);
      // First-valid-tile case: m == -INF ⇒ alpha = 0 (O, l are already 0,
      // so multiplying by 0 is a no-op). Normal case: alpha = 2^(m - m_new).
      const float alpha = (m == -INFINITY) ? 0.f : exp2f(m - m_new);
      l  *= alpha;
      o0 *= alpha;
      o1 *= alpha;

      // V load is bf162-packed: one 32-bit LDS per thread per row gives the
      // two channels (2*i, 2*i+1). For rows with logit == -INFINITY (g < 0),
      // p = exp2(-INF - m_new) = 0 so the FMA contributes nothing.
      const __nv_bfloat162* tile_pairs =
          reinterpret_cast<const __nv_bfloat162*>(ckv_buf);
      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        const float logit = s_logits_tile[r];
        const float p = exp2f(logit - m_new);
        const float2 kf = __bfloat1622float2(tile_pairs[r * (D_CK / 2) + i]);
        l  += p;
        o0 += p * kf.x;
        o1 += p * kf.y;
      }
      m = m_new;
    }
    // No SYNC_C — iter+1's SYNC_A is the barrier-before-s_logits_tile-write.
  }

  PHASE_MARK(ts_c4);  // end of fused K loop (old logit+softmax+vproj)

  // Final normalize + output write. Because n_valid_tiles > 0 guarantees at
  // least one row had g >= 0 (n_valid_tiles comes from max_valid_k_p1 > 0),
  // we know m is finite and l > 0 here.
  const float inv_l = 1.f / l;
  if (tid == 0) lse_out[lse_i] = m + log2f(l);
  const __nv_bfloat162 packed =
      __halves2bfloat162(f_to_bf16(o0 * inv_l), f_to_bf16(o1 * inv_l));
  reinterpret_cast<__nv_bfloat162*>(out + out_off)[i] = packed;

  PHASE_MARK(ts_c5);  // end of kernel
  // ts_c2 / ts_c3 are reused: both point at the same spot (end of fused
  // loop) so the diag02 output stays compatible with the 6-phase format.
  ts_c2 = ts_c4;
  ts_c3 = ts_c4;
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

// Dynamic SMEM cap sized exactly to actual kernel usage (~46 KiB for
// v5_fused). A tight cap lets the GPU schedule 4 CTAs/SM (B200: 228 KiB
// SMEM/SM → 4 × 46 = 184 KiB fits comfortably).
//
// Actual SMEM per CTA at launch (v5_fused: Idea 5, fused K loop):
//   s_logits_tile (BK * fp32)            =    64 B   (was s_logits[TOPK]=8192)
//   s_qn + s_qp ((D_CK+D_KP) * bf16)     =  1152 B
//   s_gidx    (TOPK * int32)             =  8192 B
//   s_reduce  (NUM_WARPS * fp32)         =    32 B
//   s_v_tile  (V_PIPE*BK*D_CK * bf16)    = 32768 B
//   s_kpe_tile(V_PIPE*BK*D_KP * bf16)    =  4096 B
//   Total ≈ 46304 B ≈ 45.2 KiB.
constexpr int KERNEL_MAX_DYN_SMEM = 47 * 1024;

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
    // ~46 KiB is the actual launch-time footprint for v5_fused.
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_actual, mla_sparse_attn_kernel, THREADS, 46 * 1024);
    fprintf(stderr,
            "[mla_sparse_attn] regs/thread=%d  localSz=%d  "
            "static_smem=%zu  max_dyn_smem=%d  max_threads=%d  "
            "occ@cap=%d  occ@actual(46K)=%d blocks/SM\n",
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
  // v5_fused (Idea 5): s_logits[TOPK] replaced by s_logits_tile[BK] — saves
  //                    ~8 KiB per CTA. Total ≈ 46 KiB.
  //   s_logits_tile: BK * fp32 (64 B)
  //   s_qn + s_qp  : (D_CK + D_KP) * bf16 (1152 B)
  //   s_gidx       : TOPK * int32 (8 KiB)
  //   s_reduce     : NUM_WARPS * fp32 (32 B)
  //   s_v_tile     : V_PIPE * BK * D_CK * bf16 (32 KiB)
  //   s_kpe_tile   : V_PIPE * BK * D_KP * bf16 (4 KiB)
  const size_t shmem =
      static_cast<size_t>(BK) * sizeof(float)
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
