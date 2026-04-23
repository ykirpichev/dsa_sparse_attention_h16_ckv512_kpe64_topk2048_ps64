/*
 * DSA sparse MLA — custom CUDA kernel, **sm_100a / B200 only** (CC 10.0).
 *
 * One CTA per (token, head, [split-K group]). Each CTA:
 *   1. Loads Q (NoPE + PE) and prescans sparse_idx (int4-vectorised) in SMEM.
 *   2. Iterates BK=16 K-rows per tile with cp.async double-buffered ckv+kpe.
 *   3. Runs a fused single-pass online-softmax K loop (logit → rescale →
 *      V-accumulate) so each HBM byte is consumed exactly once per token.
 *   4. Writes BF16 weighted-cKV output and fp32 base-2 LSE.
 *
 * A FlashDecoding-style split-K variant (G ∈ {1,2,4,8}) is dispatched when
 * the base grid (T*H) would underfill the SM array; a cheap reducer kernel
 * combines per-G partials into the final output (identical online-softmax
 * math on the G axis).
 *
 * Target arch: sm_100a exclusively. Compile flags (see kernel.py):
 *   -gencode=arch=compute_100a,code=sm_100a -O3 --use_fast_math
 *
 * Key sm_100a invariants leaned on here (not portable to pre-sm_100a):
 *   - cp.async family (`cp.async.ca.shared.global`, `cp.async.commit_group`,
 *     `cp.async.wait_group`) is always available — no fallback loads.
 *   - 227 KiB opt-in dynamic SMEM per CTA (attribute raised at init).
 *   - 2-way grid-wide SM occupancy target of 444 CTAs (148 SMs × 3).
 *
 * History / ablations that informed the current design (kept for context,
 * deliberately terse — see docs/OPTIMIZATION_LOG.md for full data):
 *   - Head tiling (H_PER_CTA>1): regressed — long traces are FMA-bound,
 *     not HBM-bound, so consolidating heads hurt ILP.
 *   - Per-tile rescale with saturating `-87` clamp: regressed — the current
 *     logit scale never overflows exp2f in practice.
 *   - Stage-2 tiled-SMEM fusion with mid-tile __syncthreads: regressed ~2×
 *     because the syncs chopped up the memory pipeline.
 */

#include <cuda.h>           // CUtensorMap, cuTensorMapEncodeTiled (driver API; sm_100a TMA)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#error "mla_sparse_attn: sm_100a is required (build with -gencode=arch=compute_100a,code=sm_100a)."
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

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

// ---------------------------------------------------------------------------
// TMA (Tensor Memory Accelerator) primitives — sm_100a gather4 variant.
//
// Used by mla_sparse_attn_kernel_tma to load the per-tile ckv / kpe rows in
// bulk via `cp.async.bulk.tensor.2d.tile::gather4`, replacing the per-thread
// cp.async.ca.shared.global loops (~1280 per tile → ~8 TMA instructions).
// Completion is tracked via an mbarrier per V_PIPE slot.
//
// Layout invariants:
//   * The TMA descriptor must be built with box = (boxW, 1) where boxW is the
//     row element count (D_CK for ckv, D_KP for kpe). gather4 hardware only
//     supports box dim1 = 1 (one row per tile).
//   * Destination SMEM for each gather4 call must be 128-byte aligned.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* smem_generic) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(smem_generic));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
  uint32_t a = smem_addr(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(count));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* bar) {
  uint32_t a = smem_addr(bar);
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(a));
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
}

// Sets the expected tx-count on the mbarrier in bytes AND decrements the
// pending-arrival count by 1 (combined arrive + expect_tx). Call once per
// tile load from exactly one thread (the TMA issuer).
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
  uint32_t a = smem_addr(bar);
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
      :: "r"(a), "r"(tx_bytes) : "memory");
}

// Blocks until the mbarrier phase-parity bit flips from the caller's current
// `phase`. `phase` should be toggled (^= 1) after each successful wait.
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
  uint32_t a = smem_addr(bar);
  uint32_t done = 0;
  // mbarrier.try_wait.parity sometimes spuriously returns early; loop.
  do {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done) : "r"(a), "r"(phase));
  } while (!done);
}

// Issue a single gather4 TMA load: reads 4 rows (at row0..row3) of the 2D
// tensor described by `tmap` into `dst_smem_i` (128-byte aligned SMEM addr).
// Each row contributes `box_w` elements of the descriptor's dtype.
__device__ __forceinline__ void tma_gather4_2d(
    uint32_t dst_smem_i,
    const CUtensorMap* tmap,
    uint64_t mbar_gaddr,
    int32_t col,
    int32_t row0, int32_t row1, int32_t row2, int32_t row3) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
      :: "r"(dst_smem_i), "l"(tmap),
         "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3),
         "r"(static_cast<uint32_t>(mbar_gaddr))
      : "memory");
}

// Overload: mbarrier addressed via smem-local pointer (we cast inside).
__device__ __forceinline__ void tma_gather4_2d(
    uint32_t dst_smem_i,
    const CUtensorMap* tmap,
    uint64_t* mbar_smem,
    int32_t col,
    int32_t row0, int32_t row1, int32_t row2, int32_t row3) {
  uint32_t mb = smem_addr(mbar_smem);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
      :: "r"(dst_smem_i), "l"(tmap),
         "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3),
         "r"(mb)
      : "memory");
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
  // Grid order is dim3(T, H): the hardware grid scheduler's tile-swizzle
  // already gives good L2 reuse on ckv; reordering to dim3(H, T) regressed.
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

  // SMEM layout (fused single-pass K loop — no full-TOPK s_logits):
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

  // Async Q: issue Q_nope + Q_pe via cp.async so the HBM load overlaps with
  // the sparse_idx prescan below (which doesn't touch Q SMEM). Group 0 in the
  // cp.async pipeline; waited-for by the first SYNC_A inside the tile loop.
  if (tid < Q_CK_CHUNKS) {
    const uint4* src =
        reinterpret_cast<const uint4*>(q_nope + qh_off) + tid;
    uint4* dst = reinterpret_cast<uint4*>(s_qn) + tid;
    cp_async_16(__cvta_generic_to_shared(dst), src);
  }
  if (tid < Q_KP_CHUNKS) {
    const uint4* src =
        reinterpret_cast<const uint4*>(q_pe + qp_off) + tid;
    uint4* dst = reinterpret_cast<uint4*>(s_qp) + tid;
    cp_async_16(__cvta_generic_to_shared(dst), src);
  }
  cp_async_commit_group();  // group 0: Q

  // int4 prescan of sparse_idx (4× fewer loop iters than scalar).
  // Each thread loads 2 int4 chunks (8 indices), validates, and int4-stores
  // the canonicalised result (invalid → -1) so the tile issuer downstream
  // only has to check `< 0`. Also tracks per-thread max_valid_k+1.
  int local_max_valid_k_p1 = 0;  // stored as k+1 so "none valid" reduces to 0
  const int4* sp_base = reinterpret_cast<const int4*>(sparse_idx + si_off);
  int4* sg_base = reinterpret_cast<int4*>(s_gidx);
  #pragma unroll
  for (int c = 0; c < IDX_INT4_PER_THREAD; ++c) {
    const int kb = tid + c * THREADS;
    const int k0 = kb * 4;
    const int4 g4 = sp_base[kb];
    int4 s4;
    s4.x = (g4.x >= 0 && static_cast<int64_t>(g4.x) < num_kv_rows) ? g4.x : -1;
    s4.y = (g4.y >= 0 && static_cast<int64_t>(g4.y) < num_kv_rows) ? g4.y : -1;
    s4.z = (g4.z >= 0 && static_cast<int64_t>(g4.z) < num_kv_rows) ? g4.z : -1;
    s4.w = (g4.w >= 0 && static_cast<int64_t>(g4.w) < num_kv_rows) ? g4.w : -1;
    sg_base[kb] = s4;
    if (s4.x >= 0 && (k0 + 1) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 1;
    if (s4.y >= 0 && (k0 + 2) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 2;
    if (s4.z >= 0 && (k0 + 3) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 3;
    if (s4.w >= 0 && (k0 + 4) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 4;
  }
  // (P5/T1) No pre-reduce __syncthreads(): block_reduce_max starts with a
  // CTA-wide __syncthreads() (see block_reduce_max line 1, smem[wid]=v → sync),
  // which orders the preceding s_gidx stores. Issue-tile reads of s_gidx are
  // additionally ordered by the per-tile SYNC_A barrier.
  const float reduced_max = block_reduce_max(static_cast<float>(local_max_valid_k_p1), s_reduce);
  const int n_valid_rows = static_cast<int>(reduced_max);
  int n_valid_tiles = (n_valid_rows + BK - 1) / BK;
  if (n_valid_tiles > NUM_V_TILES) n_valid_tiles = NUM_V_TILES;

  // Fused single-pass K loop.
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
    // Drain the in-flight Q cp.async (group 0) before the CTA exits — the
    // SMEM destination would otherwise be torn down under the hood while
    // the DMA is still in flight.
    cp_async_wait_group<0>();
    if (tid == 0) lse_out[lse_i] = -INFINITY;
    const __nv_bfloat162 zero =
        __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    reinterpret_cast<__nv_bfloat162*>(out + out_off)[tid] = zero;
    return;
  }

  // Single ckv+kpe tile issuer. Invalid rows (g < 0) are
  // zero-initialised directly via ST.shared rather than cp.async'ing from
  // ckv_flat[0] / kpe_flat[0]. This skips ~1 KiB of wasted HBM per invalid
  // ckv row (and 128 B per kpe row). For partially-masked tiles on short-
  // sequence traces this saves real bandwidth; for fully-valid tiles the
  // branch is warp-uniform (all 64 threads that cover a row see the same
  // s_gidx[k]), so zero divergence penalty. The ST.shared writes are
  // ordered with the cp.async writes via the SYNC_A __syncthreads in the
  // consumer iter (same barrier that flushes cp.async for this buf).
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
      const int g = s_gidx[k];
      uint4* dst = reinterpret_cast<uint4*>(ckv_buf + r * D_CK) + ch16;
      if (g >= 0) {
        const uint4* src =
            reinterpret_cast<const uint4*>(ckv_flat + static_cast<int64_t>(g) * D_CK) + ch16;
        cp_async_16(__cvta_generic_to_shared(dst), src);
      } else {
        const uint4 zero = {0u, 0u, 0u, 0u};
        *dst = zero;
      }
    }

    // kpe half: 1 × 8 B per thread (exactly THREADS 8 B chunks per tile).
    {
      const int chunk = tid;
      const int r = chunk / (D_KP / 4);
      const int ch8 = chunk % (D_KP / 4);
      const int k = k0 + r;
      const int g = s_gidx[k];
      uint2* dst = reinterpret_cast<uint2*>(kpe_buf + r * D_KP) + ch8;
      if (g >= 0) {
        const uint2* src =
            reinterpret_cast<const uint2*>(kpe_flat + static_cast<int64_t>(g) * D_KP) + ch8;
        cp_async_8(__cvta_generic_to_shared(dst), src);
      } else {
        const uint2 zero = {0u, 0u};
        *dst = zero;
      }
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
    // Unconditional rescale is retained: a cond-rescale variant (`if (tile_max
    // > m)`) regressed +0.35 % geomean by splitting the CFG of the inner loop
    // and defeating compiler scheduling.
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

// ==========================================================================
// TMA (gather4) variant — Phase 5 / Step 2.
// ==========================================================================
//
// Same math as `mla_sparse_attn_kernel`, but the per-tile ckv + kpe loads go
// through `cp.async.bulk.tensor.2d.tile::gather4` (sm_100a) instead of
// per-thread `cp.async.ca.shared.global` copies. Key differences:
//
//   * ckv / kpe are addressed via pre-built CUtensorMap descriptors passed
//     by value as __grid_constant__ kernel args (the driver API encodes the
//     base pointer + row stride; dimension-0 is the row, dimension-1 is the
//     element index, box = (boxW, 1) as required for gather4).
//
//   * Completion is tracked by a per-V_PIPE-slot mbarrier. Thread 0 issues
//     one `arrive.expect_tx` for the combined ckv+kpe byte count, then the
//     stream of gather4 instructions increments the mbarrier's tx counter
//     via the `.mbarrier::complete_tx::bytes` completion mechanism. Every
//     thread in the CTA waits on the mbarrier via `try_wait.parity` and
//     tracks its own phase bit per slot.
//
//   * SMEM layout is rearranged so s_v_tile[0], s_v_tile[1], s_kpe_tile[0]
//     and s_kpe_tile[1] all land at 128-B-aligned offsets (the gather4 dst
//     alignment requirement). The rest of the layout (q, gidx, logits,
//     reduce) is shifted after the tile buffers and mbarriers.
//
//   * Invalid rows (sparse_idx == -1) are replaced with row 0 at issue
//     time (TMA requires in-bounds coordinates); the logit stage still
//     writes -INFINITY for those rows based on the original s_gidx value,
//     so the loaded junk row contributes zero after the softmax.
//
//   * The `__syncthreads` after the mbarrier wait still serves the dual
//     role the old SYNC_A had: (a) block-wide observation point after TMA,
//     (b) fence preventing iter N+1's s_logits_tile write from racing with
//     iter N's readers.
// D_CK=512 exceeds cuTensorMapEncodeTiled's boxDim ≤ 256-element cap, so
// we encode ckv as two half-row descriptors (lo: elements [0..256),
// hi: [256..512)) and lay ckv out in SMEM as two row-indexed arenas of
// D_HALF=256 bf16 each. The gather4 dst stride (= boxW*elem_size = 512 B)
// matches exactly one bf16 half-row, so 4-row-group gather4s land at
// offsets `4g*256, (4g+1)*256, (4g+2)*256, (4g+3)*256` bf16 in the lo/hi
// arenas with no interleaving. Logit + V-accumulate adapt to read lo and
// hi halves separately (warp-uniform branch on tid < 128 for V-accum).
constexpr int D_HALF = D_CK / 2;  // 256
constexpr uint32_t TMA_CKV_BOXW = D_HALF;  // 256 — within the 256 cap
constexpr uint32_t TMA_KPE_BOXW = D_KP;    //  64

__global__ __launch_bounds__(THREADS, 2) void mla_sparse_attn_kernel_tma(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __grid_constant__ CUtensorMap tmap_ckv_lo,
    const __grid_constant__ CUtensorMap tmap_ckv_hi,
    const __grid_constant__ CUtensorMap tmap_kpe,
    const int32_t* __restrict__ sparse_idx,
    float sm_scale_log2e,
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

  // SMEM layout (TMA / half-row variant, all tile bufs 128-B aligned).
  //   s_v_lo        [V_PIPE][BK][D_HALF]  bf16  16 KiB  (offset 0)
  //   s_v_hi        [V_PIPE][BK][D_HALF]  bf16  16 KiB  (offset 16 KiB)
  //   s_kpe_tile    [V_PIPE][BK][D_KP]    bf16   4 KiB  (offset 32 KiB)
  //   s_tile_mbar   [V_PIPE]              u64    16 B   (offset 36 KiB)
  //   s_logits_tile [BK]                  fp32   64 B
  //   s_qn          [D_CK]                bf16    1 KiB
  //   s_qp          [D_KP]                bf16  128 B
  //   s_gidx        [TOPK]                int32   8 KiB
  //   s_reduce      [NUM_WARPS]           fp32   32 B
  //   TOTAL ≈ 46 KiB — same as non-TMA variant within 16 B.
  extern __shared__ unsigned char raw[];
  __nv_bfloat16* s_v_lo      = reinterpret_cast<__nv_bfloat16*>(raw);
  __nv_bfloat16* s_v_hi      = s_v_lo + V_PIPE * BK * D_HALF;
  __nv_bfloat16* s_kpe_tile  = s_v_hi + V_PIPE * BK * D_HALF;
  uint64_t* s_tile_mbar      = reinterpret_cast<uint64_t*>(s_kpe_tile + V_PIPE * BK * D_KP);
  float* s_logits_tile       = reinterpret_cast<float*>(s_tile_mbar + V_PIPE);
  __nv_bfloat16* s_qn        = reinterpret_cast<__nv_bfloat16*>(s_logits_tile + BK);
  __nv_bfloat16* s_qp        = s_qn + D_CK;
  int* s_gidx                = reinterpret_cast<int*>(s_qp + D_KP);
  float* s_reduce            = reinterpret_cast<float*>(s_gidx + TOPK);

  // Init mbarriers: one arrival (thread 0) per phase; tx bytes accumulate
  // via the gather4 completion mechanism. A proxy fence is required between
  // the generic-proxy ST.shared of mbarrier.init and the async-proxy read
  // the TMA engine performs; the __syncthreads inside block_reduce_max
  // downstream provides the CTA-wide visibility, but we issue the fence
  // explicitly so the intent is clear and not dependent on reduction
  // internals.
  if (tid < V_PIPE) {
    mbarrier_init(&s_tile_mbar[tid], 1);
  }
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::);

  // Async Q issue (identical to the non-TMA variant — we keep Q on the old
  // cp.async path because it's 1152 B total, not worth a TMA descriptor).
  if (tid < Q_CK_CHUNKS) {
    const uint4* src = reinterpret_cast<const uint4*>(q_nope + qh_off) + tid;
    uint4* dst = reinterpret_cast<uint4*>(s_qn) + tid;
    cp_async_16(__cvta_generic_to_shared(dst), src);
  }
  if (tid < Q_KP_CHUNKS) {
    const uint4* src = reinterpret_cast<const uint4*>(q_pe + qp_off) + tid;
    uint4* dst = reinterpret_cast<uint4*>(s_qp) + tid;
    cp_async_16(__cvta_generic_to_shared(dst), src);
  }
  cp_async_commit_group();

  // int4 prescan of sparse_idx (same as non-TMA variant).
  int local_max_valid_k_p1 = 0;
  const int4* sp_base = reinterpret_cast<const int4*>(sparse_idx + si_off);
  int4* sg_base = reinterpret_cast<int4*>(s_gidx);
  #pragma unroll
  for (int c = 0; c < IDX_INT4_PER_THREAD; ++c) {
    const int kb = tid + c * THREADS;
    const int k0 = kb * 4;
    const int4 g4 = sp_base[kb];
    int4 s4;
    s4.x = (g4.x >= 0 && static_cast<int64_t>(g4.x) < num_kv_rows) ? g4.x : -1;
    s4.y = (g4.y >= 0 && static_cast<int64_t>(g4.y) < num_kv_rows) ? g4.y : -1;
    s4.z = (g4.z >= 0 && static_cast<int64_t>(g4.z) < num_kv_rows) ? g4.z : -1;
    s4.w = (g4.w >= 0 && static_cast<int64_t>(g4.w) < num_kv_rows) ? g4.w : -1;
    sg_base[kb] = s4;
    if (s4.x >= 0 && (k0 + 1) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 1;
    if (s4.y >= 0 && (k0 + 2) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 2;
    if (s4.z >= 0 && (k0 + 3) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 3;
    if (s4.w >= 0 && (k0 + 4) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 4;
  }
  const float reduced_max = block_reduce_max(static_cast<float>(local_max_valid_k_p1), s_reduce);
  const int n_valid_rows = static_cast<int>(reduced_max);
  int n_valid_tiles = (n_valid_rows + BK - 1) / BK;
  if (n_valid_tiles > NUM_V_TILES) n_valid_tiles = NUM_V_TILES;

  if (n_valid_tiles == 0) {
    cp_async_wait_group<0>();
    if (tid == 0) lse_out[lse_i] = -INFINITY;
    const __nv_bfloat162 zero = __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    reinterpret_cast<__nv_bfloat162*>(out + out_off)[tid] = zero;
    return;
  }

  // gather4 tile issuer — runs on thread 0 only. Reads 16 (int32) row
  // indices from s_gidx[tile*BK .. tile*BK+15], substitutes -1 with 0
  // (the logit stage will mask those rows via the original s_gidx check),
  // and fires 4 gather4s per descriptor (lo, hi, kpe) against the slot's
  // mbarrier. One combined `arrive.expect_tx` accounts for all 12 loads.
  //
  // Per-tile byte budget:
  //   ckv_lo:  BK rows × D_HALF bf16 = 16 * 256 * 2 =  8192 B
  //   ckv_hi:  same                                = 8192 B
  //   kpe:     BK rows × D_KP   bf16 = 16 *  64 * 2 = 2048 B
  //                                      total     = 18432 B
  constexpr uint32_t TILE_CKV_HALF_BYTES = BK * D_HALF * sizeof(__nv_bfloat16);  // 8192
  constexpr uint32_t TILE_KPE_BYTES      = BK * D_KP   * sizeof(__nv_bfloat16);  // 2048
  constexpr uint32_t TILE_TX_BYTES       = 2 * TILE_CKV_HALF_BYTES + TILE_KPE_BYTES;  // 18432
  auto issue_tile_tma = [&](int tile, int buf) {
    if (tid != 0) return;
    const int k0 = tile * BK;
    __nv_bfloat16* lo_buf  = s_v_lo     + buf * (BK * D_HALF);
    __nv_bfloat16* hi_buf  = s_v_hi     + buf * (BK * D_HALF);
    __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);
    uint64_t* mbar = &s_tile_mbar[buf];
    mbarrier_arrive_expect_tx(mbar, TILE_TX_BYTES);

    // Row indices in batches of 4, with -1 → 0 fixup (mask handled downstream
    // by the logit stage writing -INFINITY when s_gidx[k] < 0).
    #pragma unroll
    for (int gr = 0; gr < BK / 4; ++gr) {
      const int rbase = gr * 4;
      const int r0 = s_gidx[k0 + rbase + 0];
      const int r1 = s_gidx[k0 + rbase + 1];
      const int r2 = s_gidx[k0 + rbase + 2];
      const int r3 = s_gidx[k0 + rbase + 3];
      const int32_t sr0 = (r0 >= 0) ? r0 : 0;
      const int32_t sr1 = (r1 >= 0) ? r1 : 0;
      const int32_t sr2 = (r2 >= 0) ? r2 : 0;
      const int32_t sr3 = (r3 >= 0) ? r3 : 0;

      // ckv lo: 4 rows × 256 bf16 = 2 KiB per gather4. gather4 packs them
      // at 512 B stride; matching s_v_lo[r*256..r*256+256] row-indexed layout.
      {
        uint32_t dst = smem_addr(lo_buf + rbase * D_HALF);
        tma_gather4_2d(dst, &tmap_ckv_lo, mbar, /*col=*/0, sr0, sr1, sr2, sr3);
      }
      // ckv hi: same but into s_v_hi (descriptor's base is ckv_flat + D_HALF
      // bf16, so col=0 covers elements [D_HALF .. D_CK)).
      {
        uint32_t dst = smem_addr(hi_buf + rbase * D_HALF);
        tma_gather4_2d(dst, &tmap_ckv_hi, mbar, /*col=*/0, sr0, sr1, sr2, sr3);
      }
      // kpe: 4 rows × 64 bf16 = 512 B per gather4.
      {
        uint32_t dst = smem_addr(kpe_buf + rbase * D_KP);
        tma_gather4_2d(dst, &tmap_kpe, mbar, /*col=*/0, sr0, sr1, sr2, sr3);
      }
    }
  };

  static_assert(D_CK == 2 * THREADS, "packed output path requires D_CK == 2*THREADS");
  static_assert(D_HALF == D_CK / 2, "D_HALF derivation");
  static_assert(THREADS == D_HALF,
                "V-accum expects thread i ∈ [0, THREADS) to map to bf162 "
                "column i within lo (i<128) or hi (i>=128) half-row");

  float m = -INFINITY, l = 0.f, o0 = 0.f, o1 = 0.f;

  // Per-slot phase bit tracker (thread-local; all threads stay in lockstep).
  uint32_t tile_phase[V_PIPE];
  #pragma unroll
  for (int p = 0; p < V_PIPE; ++p) tile_phase[p] = 0;

  // Prefetch: issue up to V_PIPE tiles ahead.
  #pragma unroll
  for (int p = 0; p < V_PIPE; ++p) {
    if (p < n_valid_tiles) issue_tile_tma(p, p);
  }

  const int warp_id = tid >> 5;
  const int lane = tid & (WARP - 1);
  const int i = tid;

  const __nv_bfloat162* qn2 = reinterpret_cast<const __nv_bfloat162*>(s_qn);
  const __nv_bfloat162* qp2 = reinterpret_cast<const __nv_bfloat162*>(s_qp);

  // Drain the Q cp.async now that we've also issued the V_PIPE tile
  // prefetches — gives ~V_PIPE tiles worth of TMA in-flight overlap with
  // the Q load. The SYNC_A-TMA __syncthreads inside the loop publishes Q
  // to all threads block-wide before the first logit read.
  cp_async_wait_group<0>();

  for (int tile = 0; tile < n_valid_tiles; ++tile) {
    const int buf = tile & 1;

    // Wait for this tile's TMA load to complete.
    mbarrier_wait(&s_tile_mbar[buf], tile_phase[buf]);
    tile_phase[buf] ^= 1;
    __syncthreads();  // SYNC_A-TMA: cross-thread fence + ordering for
                      // iter+1's s_logits_tile reuse (no SYNC_C later).

    // Issue V_PIPE ahead (same slot we just consumed).
    const int pref_tile = tile + V_PIPE;
    if (pref_tile < n_valid_tiles) {
      issue_tile_tma(pref_tile, pref_tile & 1);
    }

    const __nv_bfloat16* lo_buf  = s_v_lo     + buf * (BK * D_HALF);
    const __nv_bfloat16* hi_buf  = s_v_hi     + buf * (BK * D_HALF);
    const __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);

    // --- Phase 1: logit compute (warp-parallel, 2 dots per warp) ---
    //
    // Under the half-row SMEM layout the K-side load becomes two 128-bf162
    // inner sweeps (lo and hi) instead of one 256-bf162 sweep. Q is still
    // row-contiguous in s_qn, so we split its index accordingly: lo uses
    // qn2[i2], hi uses qn2[D_HALF/2 + i2].
    #pragma unroll
    for (int wr = 0; wr < 2; ++wr) {
      const int r = warp_id * 2 + wr;
      const int k = tile * BK + r;
      const __nv_bfloat162* klo2 = reinterpret_cast<const __nv_bfloat162*>(lo_buf + r * D_HALF);
      const __nv_bfloat162* khi2 = reinterpret_cast<const __nv_bfloat162*>(hi_buf + r * D_HALF);
      const __nv_bfloat162* kp2  = reinterpret_cast<const __nv_bfloat162*>(kpe_buf + r * D_KP);

      float acc = 0.f;
      #pragma unroll
      for (int j = 0; j < D_HALF / 2 / WARP; ++j) {  // 4 per lane per half
        const int i2 = j * WARP + lane;  // i2 ∈ [0, 128)
        {
          const float2 kf = __bfloat1622float2(klo2[i2]);
          const float2 qf = __bfloat1622float2(qn2[i2]);
          acc += qf.x * kf.x + qf.y * kf.y;
        }
        {
          const float2 kf = __bfloat1622float2(khi2[i2]);
          const float2 qf = __bfloat1622float2(qn2[D_HALF / 2 + i2]);
          acc += qf.x * kf.x + qf.y * kf.y;
        }
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
    // Thread tid=i owns output pair (2i, 2i+1). With half-row SMEM, i < 128
    // picks from the lo arena (bf162 column i), otherwise from the hi arena
    // (bf162 column i-128). Branch is warp-uniform (warps 0..3 are all lo,
    // warps 4..7 are all hi), so zero divergence penalty.
    float tile_max = -INFINITY;
    #pragma unroll
    for (int r = 0; r < BK; ++r) tile_max = fmaxf(tile_max, s_logits_tile[r]);

    if (tile_max > -INFINITY) {
      const float m_new = fmaxf(m, tile_max);
      const float alpha = (m == -INFINITY) ? 0.f : exp2f(m - m_new);
      l  *= alpha;
      o0 *= alpha;
      o1 *= alpha;

      const bool in_lo = (i < D_HALF / 2);
      const __nv_bfloat16* v_base = in_lo ? lo_buf : hi_buf;
      const int col_bf162 = in_lo ? i : (i - D_HALF / 2);
      const __nv_bfloat162* tile_pairs = reinterpret_cast<const __nv_bfloat162*>(v_base);
      #pragma unroll
      for (int r = 0; r < BK; ++r) {
        const float logit = s_logits_tile[r];
        const float p = exp2f(logit - m_new);
        const float2 kf = __bfloat1622float2(tile_pairs[r * (D_HALF / 2) + col_bf162]);
        l  += p;
        o0 += p * kf.x;
        o1 += p * kf.y;
      }
      m = m_new;
    }
  }

  // Invalidate mbarriers before CTA exit (recommended by PTX spec to allow
  // the SMEM backing to be reused cleanly under launch_bounds).
  if (tid < V_PIPE) mbarrier_inval(&s_tile_mbar[tid]);

  const float inv_l = 1.f / l;
  if (tid == 0) lse_out[lse_i] = m + log2f(l);
  const __nv_bfloat162 packed =
      __halves2bfloat162(f_to_bf16(o0 * inv_l), f_to_bf16(o1 * inv_l));
  reinterpret_cast<__nv_bfloat162*>(out + out_off)[i] = packed;
}

// ==========================================================================
// Phase-3 Split-K.
// ==========================================================================
//
// Workload shape analysis (see OPTIMIZATION_LOG.md) showed the 7 largest
// traces (T≈16, H=16 ⇒ T*H≈256, long nvt=26) only fill 256/(148 SM ×
// 3 CTA/SM = 444) ≈ 58 % of the GPU's CTA slots.  `mla_sparse_attn_kernel`
// processes one (token, head) per CTA, so there is no way for a single
// (t,h) to scale up.  We add a FlashDecoding-style split over the K
// dimension: launch G extra "K-groups" per (t,h) such that
// T*H*G ≥ target_slots, each group covering TOPK/G sparse indices and
// emitting the un-normalised online-softmax state (O_g, m_g, l_g).
// A small reducer kernel then combines the G partials into the final
// out / lse tensor.
//
// Only workloads with T*H < 444 take this path; for larger launches the
// original kernel is launched directly (G=1) with zero overhead.
//
// Shared-memory footprint of the partial kernel is SMALLER than the main
// kernel because s_gidx shrinks to TOPK/G.  Occupancy stays at 3 CTAs/SM
// (register-bound — compiler allocated 69 regs/thread for both kernels).

template <int G_>
__global__ __launch_bounds__(THREADS, 2) void mla_split_partial_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_flat,
    const __nv_bfloat16* __restrict__ kpe_flat,
    const int32_t* __restrict__ sparse_idx,
    float sm_scale_log2e,
    __nv_bfloat16* __restrict__ O_partial,   // [T, H, G, D_CK]
    float* __restrict__ m_partial,           // [T, H, G]
    float* __restrict__ l_partial,           // [T, H, G]
    int64_t H,
    int64_t num_kv_rows) {
  constexpr int TOPK_PG = TOPK / G_;
  constexpr int NUM_V_TILES_PG = TOPK_PG / BK;
  constexpr int IDX_INT4_CHUNKS_PG = TOPK_PG / 4;
  static_assert(TOPK_PG >= BK, "TOPK_PG must be >= BK");
  static_assert(TOPK_PG % 4 == 0, "TOPK_PG must be divisible by 4 for int4 prescan");

  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int g = blockIdx.z;
  const int tid = threadIdx.x;

  const int64_t qh_off = (t * H + h) * D_CK;
  const int64_t qp_off = (t * H + h) * D_KP;
  const int64_t si_off = t * TOPK + g * TOPK_PG;
  const int64_t o_part_off = ((t * H + h) * G_ + g) * D_CK;
  const int64_t ml_part_i = (t * H + h) * G_ + g;

  extern __shared__ unsigned char raw[];
  float* s_logits_tile = reinterpret_cast<float*>(raw);
  __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(s_logits_tile + BK);
  __nv_bfloat16* s_qp = s_qn + D_CK;
  int* s_gidx = reinterpret_cast<int*>(s_qp + D_KP);
  float* s_reduce = reinterpret_cast<float*>(s_gidx + TOPK_PG);
  __nv_bfloat16* s_v_tile = reinterpret_cast<__nv_bfloat16*>(s_reduce + NUM_WARPS);
  __nv_bfloat16* s_kpe_tile = s_v_tile + V_PIPE * BK * D_CK;

  // Async Q load (group 0).
  if (tid < Q_CK_CHUNKS) {
    const uint4* src = reinterpret_cast<const uint4*>(q_nope + qh_off) + tid;
    uint4* dst = reinterpret_cast<uint4*>(s_qn) + tid;
    cp_async_16(__cvta_generic_to_shared(dst), src);
  }
  if (tid < Q_KP_CHUNKS) {
    const uint4* src = reinterpret_cast<const uint4*>(q_pe + qp_off) + tid;
    uint4* dst = reinterpret_cast<uint4*>(s_qp) + tid;
    cp_async_16(__cvta_generic_to_shared(dst), src);
  }
  cp_async_commit_group();

  // int4 prescan of the K-group slice.
  int local_max_valid_k_p1 = 0;
  const int4* sp_base = reinterpret_cast<const int4*>(sparse_idx + si_off);
  int4* sg_base = reinterpret_cast<int4*>(s_gidx);
  #pragma unroll
  for (int kb = tid; kb < IDX_INT4_CHUNKS_PG; kb += THREADS) {
    const int k0 = kb * 4;
    const int4 g4 = sp_base[kb];
    int4 s4;
    s4.x = (g4.x >= 0 && static_cast<int64_t>(g4.x) < num_kv_rows) ? g4.x : -1;
    s4.y = (g4.y >= 0 && static_cast<int64_t>(g4.y) < num_kv_rows) ? g4.y : -1;
    s4.z = (g4.z >= 0 && static_cast<int64_t>(g4.z) < num_kv_rows) ? g4.z : -1;
    s4.w = (g4.w >= 0 && static_cast<int64_t>(g4.w) < num_kv_rows) ? g4.w : -1;
    sg_base[kb] = s4;
    if (s4.x >= 0 && (k0 + 1) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 1;
    if (s4.y >= 0 && (k0 + 2) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 2;
    if (s4.z >= 0 && (k0 + 3) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 3;
    if (s4.w >= 0 && (k0 + 4) > local_max_valid_k_p1) local_max_valid_k_p1 = k0 + 4;
  }
  // (P5/T1) See rationale in mla_sparse_attn_kernel — no pre-reduce sync.
  const float reduced_max = block_reduce_max(static_cast<float>(local_max_valid_k_p1), s_reduce);
  const int n_valid_rows = static_cast<int>(reduced_max);
  int n_valid_tiles = (n_valid_rows + BK - 1) / BK;
  if (n_valid_tiles > NUM_V_TILES_PG) n_valid_tiles = NUM_V_TILES_PG;

  if (n_valid_tiles == 0) {
    cp_async_wait_group<0>();
    if (tid == 0) {
      m_partial[ml_part_i] = -INFINITY;
      l_partial[ml_part_i] = 0.f;
    }
    const __nv_bfloat162 zero =
        __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    reinterpret_cast<__nv_bfloat162*>(O_partial + o_part_off)[tid] = zero;
    return;
  }

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
      const int g_idx = s_gidx[k];
      uint4* dst = reinterpret_cast<uint4*>(ckv_buf + r * D_CK) + ch16;
      if (g_idx >= 0) {
        const uint4* src =
            reinterpret_cast<const uint4*>(ckv_flat + static_cast<int64_t>(g_idx) * D_CK) + ch16;
        cp_async_16(__cvta_generic_to_shared(dst), src);
      } else {
        const uint4 zero = {0u, 0u, 0u, 0u};
        *dst = zero;
      }
    }

    {
      const int chunk = tid;
      const int r = chunk / (D_KP / 4);
      const int ch8 = chunk % (D_KP / 4);
      const int k = k0 + r;
      const int g_idx = s_gidx[k];
      uint2* dst = reinterpret_cast<uint2*>(kpe_buf + r * D_KP) + ch8;
      if (g_idx >= 0) {
        const uint2* src =
            reinterpret_cast<const uint2*>(kpe_flat + static_cast<int64_t>(g_idx) * D_KP) + ch8;
        cp_async_8(__cvta_generic_to_shared(dst), src);
      } else {
        const uint2 zero = {0u, 0u};
        *dst = zero;
      }
    }
  };

  float m = -INFINITY, l = 0.f, o0 = 0.f, o1 = 0.f;
  issue_tile(0, 0);
  cp_async_commit_group();

  const int warp_id = tid >> 5;
  const int lane = tid & (WARP - 1);
  const int i = tid;
  const __nv_bfloat162* qn2 = reinterpret_cast<const __nv_bfloat162*>(s_qn);
  const __nv_bfloat162* qp2 = reinterpret_cast<const __nv_bfloat162*>(s_qp);

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
    __syncthreads();

    const __nv_bfloat16* ckv_buf = s_v_tile + buf * (BK * D_CK);
    const __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);

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
        const int g_idx = s_gidx[k];
        s_logits_tile[r] = (g_idx >= 0) ? (acc * sm_scale_log2e) : -INFINITY;
      }
    }
    __syncthreads();

    float tile_max = -INFINITY;
    #pragma unroll
    for (int r = 0; r < BK; ++r) tile_max = fmaxf(tile_max, s_logits_tile[r]);

    if (tile_max > -INFINITY) {
      const float m_new = fmaxf(m, tile_max);
      const float alpha = (m == -INFINITY) ? 0.f : exp2f(m - m_new);
      l  *= alpha;
      o0 *= alpha;
      o1 *= alpha;

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
  }

  // Write UN-NORMALISED partial state (no /l divide); the reducer will do it.
  if (tid == 0) {
    m_partial[ml_part_i] = m;
    l_partial[ml_part_i] = l;
  }
  const __nv_bfloat162 packed =
      __halves2bfloat162(f_to_bf16(o0), f_to_bf16(o1));
  reinterpret_cast<__nv_bfloat162*>(O_partial + o_part_off)[i] = packed;
}

// Reducer: combines G partial (m, l, O_unnorm) states into the final
// (out, lse).  Grid = (T, H), block = THREADS (one thread per 2-channel
// output slot).  G is a compile-time template param so the inner loops
// fully unroll and the compiler keeps s_m/s_l in registers if needed.
template <int G_>
__global__ void mla_split_reduce_kernel(
    const __nv_bfloat16* __restrict__ O_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse_out,
    int64_t H) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;

  const int64_t out_off = (t * H + h) * D_CK;
  const int64_t lse_i = t * H + h;
  const int64_t o_part_base = ((t * H + h) * G_) * D_CK;
  const int64_t ml_base = (t * H + h) * G_;

  __shared__ float s_m[G_];
  __shared__ float s_l[G_];
  if (tid < G_) {
    s_m[tid] = m_partial[ml_base + tid];
    s_l[tid] = l_partial[ml_base + tid];
  }
  __syncthreads();

  float m_final = -INFINITY;
  #pragma unroll
  for (int g = 0; g < G_; ++g) m_final = fmaxf(m_final, s_m[g]);

  if (m_final == -INFINITY) {
    const __nv_bfloat162 zero =
        __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    reinterpret_cast<__nv_bfloat162*>(out + out_off)[tid] = zero;
    if (tid == 0) lse_out[lse_i] = -INFINITY;
    return;
  }

  float l_final = 0.f;
  #pragma unroll
  for (int g = 0; g < G_; ++g) {
    l_final += s_l[g] * exp2f(s_m[g] - m_final);
  }
  const float inv_l = 1.f / l_final;

  float o0 = 0.f, o1 = 0.f;
  const int i = tid;
  #pragma unroll
  for (int g = 0; g < G_; ++g) {
    const float alpha = exp2f(s_m[g] - m_final);
    const __nv_bfloat162* op2 =
        reinterpret_cast<const __nv_bfloat162*>(O_partial + o_part_base + g * D_CK);
    const float2 of = __bfloat1622float2(op2[i]);
    o0 += of.x * alpha;
    o1 += of.y * alpha;
  }

  const __nv_bfloat162 packed =
      __halves2bfloat162(f_to_bf16(o0 * inv_l), f_to_bf16(o1 * inv_l));
  reinterpret_cast<__nv_bfloat162*>(out + out_off)[i] = packed;
  if (tid == 0) lse_out[lse_i] = m_final + log2f(l_final);
}

// Dynamic SMEM cap sized exactly to the fused-K-loop kernel (~45 KiB). A
// tight cap lets B200 schedule 4 CTAs/SM (228 KiB SMEM/SM → 4×46=184 KiB).
//
// Actual per-CTA breakdown (bytes):
//   s_logits_tile (BK * fp32)            =    64
//   s_qn + s_qp ((D_CK+D_KP) * bf16)     =  1152
//   s_gidx    (TOPK * int32)             =  8192
//   s_reduce  (NUM_WARPS * fp32)         =    32
//   s_v_tile  (V_PIPE*BK*D_CK * bf16)    = 32768
//   s_kpe_tile(V_PIPE*BK*D_KP * bf16)    =  4096
//   Total ≈ 46304 B ≈ 45.2 KiB.
constexpr int KERNEL_MAX_DYN_SMEM = 47 * 1024;

// Split-K partial-kernel SMEM footprint (bytes) for each compile-time G.
//   base = s_logits_tile(64) + (s_qn+s_qp)(1152) + s_reduce(32)
//        + s_v_tile(32768) + s_kpe_tile(4096) = 38112
//   + s_gidx (TOPK/G * int) = 8192 / G
template <int G_>
constexpr size_t split_partial_shmem() {
  return 38112ULL + (static_cast<size_t>(TOPK) * sizeof(int)) / G_;
}
constexpr size_t SPLIT_PARTIAL_MAX_DYN_SMEM = 47 * 1024;

// Target CTA occupancy used by the dispatcher to pick G. B200 has 148 SMs.
// The main kernel runs at 3 CTAs/SM (register-bound), so we aim to fill
// 148 * 3 = 444 CTA slots across the whole grid.  Any (T*H) smaller than
// this leaves part of the GPU idle and is a candidate for split-K.
constexpr int SPLIT_TARGET_SLOTS = 444;
// Floor on T*H below which split-K is NEVER profitable: the reducer +
// double-launch overhead (~3–5 µs on B200) exceeds any gain when the
// non-split kernel itself is already sub-20 µs.  Tiny workloads (T ≤ 2 with
// short-K sparse patterns) all fall here and stay on the single-kernel path.
// Empirically (smoke bench with floor=128): the 123 µs workload "d57eb9e1"
// was NOT split and stayed at 123 µs while the equivalent-latency "5096e459"
// (T*H ≥ 128) split to 67 µs — so the 123 µs tier has two sub-families at
// T*H around 64 vs 256 and the 64-family was being left out.  Lowering the
// floor to 64 with G=8 added brings them in.
constexpr int SPLIT_MIN_CTAS = 64;

static int pick_split_G(int base_ctas) {
  if (base_ctas < SPLIT_MIN_CTAS)                return 1;   // too tiny to amortise
  if (base_ctas >= SPLIT_TARGET_SLOTS)           return 1;   // fill already
  if (base_ctas * 2  >= SPLIT_TARGET_SLOTS)      return 2;   // [222, 443]
  if (base_ctas * 4  >= SPLIT_TARGET_SLOTS)      return 4;   // [111, 221]
  return 8;                                                 // [64, 110]
}

// Per-process scratch cache for split-K partials.  Resized on first use and
// whenever a larger workload comes in; never shrunk.  A single mutex keeps
// concurrent invocations from stomping on each other.  cudaFree on grow is
// synchronous but happens only O(log max_bytes) times across the run.
struct SplitKScratch {
  void* O = nullptr;   // bf16, max [T*H*G*D_CK]
  float* m = nullptr;  // fp32, max [T*H*G]
  float* l = nullptr;  // fp32, max [T*H*G]
  size_t O_bytes = 0;
  size_t ml_bytes = 0;
  std::mutex mu;

  void ensure(size_t new_O_bytes, size_t new_ml_bytes) {
    std::lock_guard<std::mutex> lk(mu);
    if (new_O_bytes > O_bytes) {
      if (O) cudaFree(O);
      cudaMalloc(&O, new_O_bytes);
      O_bytes = new_O_bytes;
    }
    if (new_ml_bytes > ml_bytes) {
      if (m) cudaFree(m);
      if (l) cudaFree(l);
      cudaMalloc(reinterpret_cast<void**>(&m), new_ml_bytes);
      cudaMalloc(reinterpret_cast<void**>(&l), new_ml_bytes);
      ml_bytes = new_ml_bytes;
    }
  }
};

static SplitKScratch g_split_scratch;

// ---------------------------------------------------------------------------
// TMA descriptor cache (host side).
//
// For each (base_ptr, num_kv_rows, boxW) triple the driver API's
// `cuTensorMapEncodeTiled` call populates a 128-byte CUtensorMap. Encoding
// costs a few microseconds per call — acceptable for long traces but a real
// tax on tiny workloads. We cache by (ptr, rows, boxW) so each benchmark
// only pays it once.
//
// Caveat: if a tensor pointer is freed and another tensor reallocates at
// the same address with a different shape, the cache would return a stale
// descriptor. We key on num_kv_rows and boxW to catch the shape-change case
// but NOT a same-shape reallocation. Acceptable risk for the benchmark
// scenario (tensors are allocated once and reused).
// ---------------------------------------------------------------------------
struct TmaDescKey {
  uintptr_t ptr;
  int64_t num_rows;
  int32_t boxW;
  int32_t row_stride_elems;
  bool operator==(const TmaDescKey& o) const noexcept {
    return ptr == o.ptr && num_rows == o.num_rows
        && boxW == o.boxW && row_stride_elems == o.row_stride_elems;
  }
};
struct TmaDescKeyHash {
  size_t operator()(const TmaDescKey& k) const noexcept {
    size_t h = std::hash<uintptr_t>()(k.ptr);
    h ^= std::hash<int64_t>()(k.num_rows)         + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int32_t>()(k.boxW)             + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int32_t>()(k.row_stride_elems) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

struct TmaDescCache {
  std::unordered_map<TmaDescKey, CUtensorMap, TmaDescKeyHash> map;
  std::mutex mu;

  // Encode a 2D tiled descriptor over a (num_rows × row_stride_elems) tensor,
  // gathering a (boxW, 1) box. Set `row_stride_elems` to the real HBM row
  // pitch (=D_CK for lo/hi sub-descriptors where boxW=D_HALF), or == boxW
  // when the descriptor covers a full contiguous row. `base_ptr` may point
  // into the interior of a tensor row (e.g. ckv_flat + D_HALF for the hi
  // half) — each half-descriptor therefore gets a distinct key.
  const CUtensorMap* get_or_make(
      const void* base_ptr,
      int64_t num_rows,
      int32_t boxW,
      int32_t row_stride_elems) {
    std::lock_guard<std::mutex> lk(mu);
    TmaDescKey key{reinterpret_cast<uintptr_t>(base_ptr), num_rows, boxW, row_stride_elems};
    auto it = map.find(key);
    if (it != map.end()) return &it->second;

    CUtensorMap tmap{};
    // 2D tensor size: inner=col extent (=boxW), outer=row count. The logical
    // tensor seen via this descriptor is (num_rows × boxW); the gather4 box
    // is (boxW, 1).
    uint64_t tensor_size[2] = {
        static_cast<uint64_t>(boxW),
        static_cast<uint64_t>(num_rows)
    };
    // Outer-dim stride in bytes. Must match real HBM row pitch (can be
    // larger than boxW for half-row descriptors that strad­dle a wider row).
    uint64_t tensor_stride[1] = {
        static_cast<uint64_t>(row_stride_elems) * sizeof(__nv_bfloat16)
    };
    uint32_t box_size[2] = { static_cast<uint32_t>(boxW), 1u };
    uint32_t elem_stride[2] = { 1u, 1u };
    CUresult res = cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<void*>(base_ptr),
        tensor_size, tensor_stride, box_size, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK_EQ(static_cast<int>(res), static_cast<int>(CUDA_SUCCESS))
        << "cuTensorMapEncodeTiled failed (res=" << static_cast<int>(res) << ")";
    auto ins = map.emplace(key, tmap);
    return &ins.first->second;
  }
};

static TmaDescCache g_tma_cache;

// Kernel-attrs init for the TMA variant. Called from configure_kernel_attrs_once.
static bool g_tma_attrs_ready = false;
static void configure_tma_kernel_attrs() {
  if (g_tma_attrs_ready) return;
  g_tma_attrs_ready = true;
  cudaFuncSetAttribute(mla_sparse_attn_kernel_tma,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       KERNEL_MAX_DYN_SMEM);
  cudaFuncAttributes attrs{};
  cudaFuncGetAttributes(&attrs, (const void*)mla_sparse_attn_kernel_tma);
  fprintf(stderr,
          "[mla_sparse_attn_tma] regs/thread=%d  static_smem=%zu  "
          "max_dyn_smem=%d  max_threads=%d\n",
          attrs.numRegs, (size_t)attrs.sharedSizeBytes,
          attrs.maxDynamicSharedSizeBytes, attrs.maxThreadsPerBlock);
}

// Env-gated opt-in for the TMA path (default off during bring-up).
static bool tma_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLA_SPARSE_ATTN_TMA");
    if (!env) return false;
    return env[0] == '1' || env[0] == 't' || env[0] == 'T'
        || env[0] == 'y' || env[0] == 'Y';
  }();
  return enabled;
}

void configure_kernel_attrs_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    cudaFuncSetAttribute(mla_sparse_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         KERNEL_MAX_DYN_SMEM);

    // Split-K: instantiate & configure attrs for every G we might dispatch.
    cudaFuncSetAttribute(mla_split_partial_kernel<2>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SPLIT_PARTIAL_MAX_DYN_SMEM);
    cudaFuncSetAttribute(mla_split_partial_kernel<4>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SPLIT_PARTIAL_MAX_DYN_SMEM);
    cudaFuncSetAttribute(mla_split_partial_kernel<8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SPLIT_PARTIAL_MAX_DYN_SMEM);
    cudaFuncSetAttribute(mla_split_partial_kernel<16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SPLIT_PARTIAL_MAX_DYN_SMEM);

    cudaFuncAttributes attrs{};
    cudaFuncGetAttributes(&attrs, (const void*)mla_sparse_attn_kernel);
    int occ_full_cap = 0;
    int occ_actual = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_full_cap, mla_sparse_attn_kernel, THREADS, KERNEL_MAX_DYN_SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_actual, mla_sparse_attn_kernel, THREADS, 46 * 1024);

    cudaFuncAttributes split4_attrs{};
    cudaFuncGetAttributes(&split4_attrs, (const void*)mla_split_partial_kernel<4>);
    int occ_split4 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ_split4, mla_split_partial_kernel<4>, THREADS,
        static_cast<int>(split_partial_shmem<4>()));

    fprintf(stderr,
            "[mla_sparse_attn] regs/thread=%d  localSz=%d  "
            "static_smem=%zu  max_dyn_smem=%d  max_threads=%d  "
            "occ@cap=%d  occ@actual(46K)=%d blocks/SM\n",
            attrs.numRegs, (int)attrs.localSizeBytes,
            (size_t)attrs.sharedSizeBytes, attrs.maxDynamicSharedSizeBytes,
            attrs.maxThreadsPerBlock, occ_full_cap, occ_actual);
    fprintf(stderr,
            "[mla_split_partial<4>] regs/thread=%d  static_smem=%zu  "
            "max_dyn_smem=%d  occ(%zu B)=%d blocks/SM\n",
            split4_attrs.numRegs,
            (size_t)split4_attrs.sharedSizeBytes,
            split4_attrs.maxDynamicSharedSizeBytes,
            split_partial_shmem<4>(), occ_split4);
  });
}

template <int G_>
static void launch_split_partial_and_reduce(
    int64_t T, int64_t H,
    const __nv_bfloat16* q_nope,
    const __nv_bfloat16* q_pe,
    const __nv_bfloat16* ckv_flat,
    const __nv_bfloat16* kpe_flat,
    const int32_t* sparse_idx,
    float sm_log2e,
    __nv_bfloat16* out,
    float* lse,
    int64_t num_kv_rows,
    cudaStream_t stream) {
  const size_t O_bytes =
      static_cast<size_t>(T) * H * G_ * D_CK * sizeof(__nv_bfloat16);
  const size_t ml_bytes =
      static_cast<size_t>(T) * H * G_ * sizeof(float);
  g_split_scratch.ensure(O_bytes, ml_bytes);

  const dim3 part_grid(static_cast<unsigned>(T),
                       static_cast<unsigned>(H),
                       static_cast<unsigned>(G_));
  const dim3 block(THREADS);
  const size_t part_shmem = split_partial_shmem<G_>();

  mla_split_partial_kernel<G_><<<part_grid, block, part_shmem, stream>>>(
      q_nope, q_pe, ckv_flat, kpe_flat, sparse_idx, sm_log2e,
      static_cast<__nv_bfloat16*>(g_split_scratch.O),
      g_split_scratch.m, g_split_scratch.l,
      H, num_kv_rows);

  const dim3 red_grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
  mla_split_reduce_kernel<G_><<<red_grid, block, 0, stream>>>(
      static_cast<__nv_bfloat16*>(g_split_scratch.O),
      g_split_scratch.m, g_split_scratch.l,
      out, lse, H);
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

  const dim3 block(THREADS);

  const DLDevice dev = q_nope.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  const int base_ctas = static_cast<int>(T * H);
  const int G = pick_split_G(base_ctas);

  const auto* q_nope_p = static_cast<const __nv_bfloat16*>(q_nope.data_ptr());
  const auto* q_pe_p   = static_cast<const __nv_bfloat16*>(q_pe.data_ptr());
  const auto* ckv_p    = static_cast<const __nv_bfloat16*>(ckv_cache.data_ptr());
  const auto* kpe_p    = static_cast<const __nv_bfloat16*>(kpe_cache.data_ptr());
  const auto* si_p     = static_cast<const int32_t*>(sparse_indices.data_ptr());
  auto* out_p          = static_cast<__nv_bfloat16*>(out.data_ptr());
  auto* lse_p          = static_cast<float*>(lse.data_ptr());

  if (G == 1) {
    // Default path: one CTA per (t, h).
    //   s_logits_tile: BK * fp32 (64 B)
    //   s_qn + s_qp  : (D_CK + D_KP) * bf16 (1152 B)
    //   s_gidx       : TOPK * int32 (8 KiB)
    //   s_reduce     : NUM_WARPS * fp32 (32 B)
    //   s_v_tile     : V_PIPE * BK * D_CK * bf16 (32 KiB)
    //   s_kpe_tile   : V_PIPE * BK * D_KP * bf16 (4 KiB)
    //   + TMA path only: 2 × u64 mbarriers (16 B, negligible).
    const size_t shmem_base =
        static_cast<size_t>(BK) * sizeof(float)
        + static_cast<size_t>(D_CK + D_KP) * sizeof(__nv_bfloat16)
        + static_cast<size_t>(TOPK) * sizeof(int)
        + static_cast<size_t>(NUM_WARPS) * sizeof(float)
        + static_cast<size_t>(V_PIPE) * BK * D_CK * sizeof(__nv_bfloat16)
        + static_cast<size_t>(V_PIPE) * BK * D_KP * sizeof(__nv_bfloat16);
    const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(H));
    if (tma_enabled()) {
      configure_tma_kernel_attrs();
      // Half-row split: D_CK=512 exceeds cuTensorMapEncodeTiled's 256-element
      // boxDim cap, so we encode two boxW=256 descriptors over the same
      // ckv_flat base. The `hi` descriptor starts D_HALF bf16 elements into
      // each row, so the TMA engine fetches the second half when col=0.
      const auto* ckv_bf16 = reinterpret_cast<const __nv_bfloat16*>(ckv_p);
      const CUtensorMap* tm_ckv_lo = g_tma_cache.get_or_make(
          ckv_bf16,          num_kv_rows, /*boxW=*/D_HALF, /*row_stride=*/D_CK);
      const CUtensorMap* tm_ckv_hi = g_tma_cache.get_or_make(
          ckv_bf16 + D_HALF, num_kv_rows, /*boxW=*/D_HALF, /*row_stride=*/D_CK);
      const CUtensorMap* tm_kpe    = g_tma_cache.get_or_make(
          kpe_p,             num_kv_rows, /*boxW=*/D_KP,   /*row_stride=*/D_KP);
      const size_t shmem_tma = shmem_base
          + static_cast<size_t>(V_PIPE) * sizeof(uint64_t);
      mla_sparse_attn_kernel_tma<<<grid, block, shmem_tma, stream>>>(
          q_nope_p, q_pe_p, *tm_ckv_lo, *tm_ckv_hi, *tm_kpe, si_p, sm_log2e,
          out_p, lse_p, H, num_kv_rows);
    } else {
      mla_sparse_attn_kernel<<<grid, block, shmem_base, stream>>>(
          q_nope_p, q_pe_p, ckv_p, kpe_p, si_p, sm_log2e,
          out_p, lse_p, H, num_kv_rows);
    }
  } else if (G == 2) {
    launch_split_partial_and_reduce<2>(
        T, H, q_nope_p, q_pe_p, ckv_p, kpe_p, si_p,
        sm_log2e, out_p, lse_p, num_kv_rows, stream);
  } else if (G == 4) {
    launch_split_partial_and_reduce<4>(
        T, H, q_nope_p, q_pe_p, ckv_p, kpe_p, si_p,
        sm_log2e, out_p, lse_p, num_kv_rows, stream);
  } else if (G == 8) {
    launch_split_partial_and_reduce<8>(
        T, H, q_nope_p, q_pe_p, ckv_p, kpe_p, si_p,
        sm_log2e, out_p, lse_p, num_kv_rows, stream);
  } else {
    // G == 16 (the cap).
    launch_split_partial_and_reduce<16>(
        T, H, q_nope_p, q_pe_p, ckv_p, kpe_p, si_p,
        sm_log2e, out_p, lse_p, num_kv_rows, stream);
  }

  const cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "mla_sparse_attn_kernel: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel_fn);
