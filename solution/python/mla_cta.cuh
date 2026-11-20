// Per-token kernel. Included from kernel.cu inside the anonymous namespace.
//
// One CTA owns every head of one token (16 warps, one head each), so a KV
// row loaded into shared memory is reused across all heads. Tensor-core
// WMMA does both the score and the output mix. Split-K cuts the top-k
// axis when a single CTA per token would leave the GPU idle.

// ==========================================================================
// CTA-per-t kernel variant "mma_v" (Phase 7c-next) — MMA for both logit
// and V-accumulation. Attacks the ~46µs CPT floor dominated by the scalar
// V-accum compute and redundant per-thread exp2f.
//
// Differences from `mma`:
//   * Softmax computes p = exp2(logit - m_new) ONCE per (head, r) (in lane
//     0..15 of the warp for each head) and writes to `s_p_tile[H, BK]` bf16.
//   * V-accumulation uses WMMA (m16n16k16 bf16→fp32). Each warp owns 2
//     n-chunks of 16 channels each (= 32 channels of D_CK=512 per warp;
//     16 warps × 32 = 512 = D_CK).
//   * Output accumulator `c_frag[2]` is persistent in registers across
//     tiles. Online-softmax rescale uses store→per-element α[h] mul→load
//     through `s_O_scratch` (overlaid with `s_mma_scratch`; both are
//     single-tile-live so the overlay is safe).
//
// WARPS, TH, K-split layout are identical to `mma`.
// ==========================================================================
template <int G_, int H_, int MIN_BLOCKS_>
__global__ __launch_bounds__(512, MIN_BLOCKS_)
void mla_sparse_attn_kernel_cta_per_t_mma_v(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_flat,
    const __nv_bfloat16* __restrict__ kpe_flat,
    const int32_t* __restrict__ sparse_idx,
    float sm_scale_log2e,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse_out,
    float* __restrict__ O_partial,  // Phase 7i: fp32 partials
    float* __restrict__ m_partial,
    float* __restrict__ l_partial,
    int64_t H,
    int64_t num_kv_rows) {
  using namespace nvcuda::wmma;
  constexpr int TH              = 512;
  constexpr int WARPS           = TH / WARP;           // 16
  constexpr int CH_PT           = D_CK / WARP;         // 16 — channels per thread (for final write)
  constexpr int TOPK_PG         = TOPK / G_;
  constexpr int NUM_V_TILES_PG  = TOPK_PG / BK;
  constexpr int IDX_INT4_CHUNKS_PG = TOPK_PG / 4;
  constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
  constexpr int CK_KTILES = D_CK / MMA_K;              // 32
  constexpr int KP_KTILES = D_KP / MMA_K;              // 4
  constexpr int CK_KT_PER_WARP = CK_KTILES / WARPS;    // 2
  constexpr int N_CHUNKS_PER_WARP = CK_KTILES / WARPS; // 2 (same split for V-accum N)
  constexpr int N_PER_WARP = N_CHUNKS_PER_WARP * MMA_N;// 32 channels per warp

  static_assert(H_ == 16,             "mma_v kernel requires H_=16");
  static_assert(WARPS == H_,          "one warp per head: WARPS must equal H_");
  static_assert(D_CK == WARP * CH_PT, "D_CK must equal WARP × CH_PT");
  static_assert((BK * D_CK * 2) % (TH * 16) == 0,
                "BK*D_CK bf16 bytes must be divisible by TH × 16");
  static_assert(TOPK_PG >= BK, "TOPK_PG must be >= BK");
  static_assert(TOPK_PG % 4 == 0, "TOPK_PG must be divisible by 4");
  static_assert((H_ * Q_CK_CHUNKS) % TH == 0,
                "Q_nope uint4 count must be a multiple of TH");
  static_assert(CK_KTILES == CK_KT_PER_WARP * WARPS, "CK split must be exact");
  static_assert(N_PER_WARP * WARPS == D_CK, "N split must cover D_CK exactly");
  static_assert(BK == MMA_N && H_ == MMA_M, "BK=MMA_N and H_=MMA_M required");

  const int t = blockIdx.x;
  const int g = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane    = tid & (WARP - 1);
  const int my_h    = warp_id;
  const int ch_base = lane * CH_PT;

  const int64_t si_off = t * TOPK + g * TOPK_PG;

  extern __shared__ unsigned char raw[];
  float* s_logits_tile = reinterpret_cast<float*>(raw);
  __nv_bfloat16* s_qn   = reinterpret_cast<__nv_bfloat16*>(s_logits_tile + H_ * BK);
  __nv_bfloat16* s_qp   = s_qn + H_ * D_CK;
  int* s_gidx           = reinterpret_cast<int*>(s_qp + H_ * D_KP);
  float* s_reduce       = reinterpret_cast<float*>(s_gidx + TOPK_PG);
  // OVERLAY: s_mma_scratch (16 KiB) and s_O_scratch (32 KiB) share the same
  // backing storage. Per-tile order of use guarantees safety:
  //   phase 2 (logit MMA)   writes s_mma_scratch  (16 KiB, heads 0..7 of s_O view)
  //   phase 3 (reduce)      reads  s_mma_scratch  → s_logits_tile
  //   phase 5 (rescale)     writes s_O_scratch    (32 KiB, c_frag store)  ← clobbers
  //   phase 6 (V-MMA)       reads  s_O_scratch    (load rescaled frag)
  //   phase 1 (next tile)   writes s_mma_scratch  again (OK, prev tile's frag in reg)
  float* s_scratch_pool = s_reduce + WARPS;           // MAX(mma,O) = 32 KiB
  constexpr int O_SCRATCH_FP32 = H_ * D_CK;           // 8192 fp32 = 32 KiB
  __nv_bfloat16* s_p_tile = reinterpret_cast<__nv_bfloat16*>(
      s_scratch_pool + O_SCRATCH_FP32);               // H*BK bf16 = 512 B
  float* s_alpha          = reinterpret_cast<float*>(s_p_tile + H_ * BK);  // H fp32
  __nv_bfloat16* s_v_tile   = reinterpret_cast<__nv_bfloat16*>(
      s_alpha + H_);
  __nv_bfloat16* s_kpe_tile = s_v_tile + V_PIPE * BK * D_CK;

  float* const s_mma_scratch = s_scratch_pool;        // aliases first 16 KiB
  float* const s_O_scratch   = s_scratch_pool;        // aliases full 32 KiB

  // ---- Async Q load (same as mma) --------------------------------------
  constexpr int QN_CHUNKS_TOTAL = H_ * Q_CK_CHUNKS;
  constexpr int QN_PER_THREAD   = QN_CHUNKS_TOTAL / TH;
  #pragma unroll
  for (int c = 0; c < QN_PER_THREAD; ++c) {
    const int chunk = tid + c * TH;
    const int h     = chunk / Q_CK_CHUNKS;
    const int i     = chunk % Q_CK_CHUNKS;
    const uint4* src = reinterpret_cast<const uint4*>(q_nope + (t * H + h) * D_CK) + i;
    uint4* dst = reinterpret_cast<uint4*>(s_qn + h * D_CK) + i;
    cp_async_16(__cvta_generic_to_shared(dst), src);
  }
  constexpr int QP_CHUNKS_TOTAL = H_ * (D_KP / 4);
  if (tid < QP_CHUNKS_TOTAL) {
    const int chunk = tid;
    const int h     = chunk / (D_KP / 4);
    const int i     = chunk % (D_KP / 4);
    const uint2* src = reinterpret_cast<const uint2*>(q_pe + (t * H + h) * D_KP) + i;
    uint2* dst = reinterpret_cast<uint2*>(s_qp + h * D_KP) + i;
    cp_async_8(__cvta_generic_to_shared(dst), src);
  }
  cp_async_commit_group();

  // ---- int4 prescan of sparse_idx (same as mma) -----------------------
  int local_max_valid_k_p1 = 0;
  const int4* sp_base = reinterpret_cast<const int4*>(sparse_idx + si_off);
  int4* sg_base = reinterpret_cast<int4*>(s_gidx);
  #pragma unroll
  for (int kb = tid; kb < IDX_INT4_CHUNKS_PG; kb += TH) {
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
  float max_v = static_cast<float>(local_max_valid_k_p1);
  max_v = warp_reduce_max(max_v);
  if (lane == 0) s_reduce[warp_id] = max_v;
  __syncthreads();
  if (warp_id == 0) {
    float v = (lane < WARPS) ? s_reduce[lane] : -INFINITY;
    v = warp_reduce_max(v);
    if (lane == 0) s_reduce[0] = v;
  }
  __syncthreads();
  const int n_valid_rows = static_cast<int>(s_reduce[0]);
  int n_valid_tiles = (n_valid_rows + BK - 1) / BK;
  if (n_valid_tiles > NUM_V_TILES_PG) n_valid_tiles = NUM_V_TILES_PG;

  // ---- Early exit ------------------------------------------------------
  if (n_valid_tiles == 0) {
    cp_async_wait_group<0>();
    const __nv_bfloat162 zero =
        __halves2bfloat162(f_to_bf16(0.f), f_to_bf16(0.f));
    if constexpr (G_ == 1) {
      __nv_bfloat162* out_row =
          reinterpret_cast<__nv_bfloat162*>(out + (t * H + my_h) * D_CK);
      #pragma unroll
      for (int c2 = 0; c2 < CH_PT / 2; ++c2) {
        out_row[(ch_base >> 1) + c2] = zero;
      }
      if (lane == 0) lse_out[t * H + my_h] = -INFINITY;
    } else {
      // Phase 7i: fp32 partials.
      float2* out_row = reinterpret_cast<float2*>(
          O_partial + ((t * H + my_h) * G_ + g) * D_CK);
      #pragma unroll
      for (int c2 = 0; c2 < CH_PT / 2; ++c2) {
        out_row[(ch_base >> 1) + c2] = make_float2(0.f, 0.f);
      }
      if (lane == 0) {
        const int64_t ml_i = (t * H + my_h) * G_ + g;
        m_partial[ml_i] = -INFINITY;
        l_partial[ml_i] = 0.f;
      }
    }
    return;
  }

  // ---- Tile issuer (same as mma) --------------------------------------
  constexpr int TILE_CHUNKS_PT = (BK * D_CK * 2 / 16) / TH;  // 2
  auto issue_tile = [&](int tile, int buf) {
    const int k0 = tile * BK;
    __nv_bfloat16* ckv_buf = s_v_tile + buf * (BK * D_CK);
    __nv_bfloat16* kpe_buf = s_kpe_tile + buf * (BK * D_KP);
    #pragma unroll
    for (int c = 0; c < TILE_CHUNKS_PT; ++c) {
      const int chunk = tid + c * TH;
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
    if (tid < BK * (D_KP / 4)) {
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

  // Per-warp (per-head) running softmax state. Only lane 0 is authoritative.
  float m = -INFINITY, l = 0.f;

  // Persistent V-accum registers (2 n-chunks per warp). Initialized to 0.
  fragment<accumulator, MMA_M, MMA_N, MMA_K, float> c_o[N_CHUNKS_PER_WARP];
  #pragma unroll
  for (int nc = 0; nc < N_CHUNKS_PER_WARP; ++nc) fill_fragment(c_o[nc], 0.f);

  issue_tile(0, 0);
  cp_async_commit_group();

  // ---- Fused online-softmax K loop -------------------------------------
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

    // ---- Logit phase: WMMA GEMM, K-split across 16 warps ----------------
    {
      fragment<matrix_a, MMA_M, MMA_N, MMA_K, __nv_bfloat16, row_major> a_frag;
      fragment<matrix_b, MMA_M, MMA_N, MMA_K, __nv_bfloat16, col_major> b_frag;
      fragment<accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;
      fill_fragment(c_frag, 0.f);

      #pragma unroll
      for (int ks = 0; ks < CK_KT_PER_WARP; ++ks) {
        const int k_off = (warp_id * CK_KT_PER_WARP + ks) * MMA_K;
        load_matrix_sync(a_frag, s_qn + k_off, D_CK);
        load_matrix_sync(b_frag, ckv_buf + k_off, D_CK);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      if (warp_id < KP_KTILES) {
        const int k_off = warp_id * MMA_K;
        load_matrix_sync(a_frag, s_qp + k_off, D_KP);
        load_matrix_sync(b_frag, kpe_buf + k_off, D_KP);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      store_matrix_sync(s_mma_scratch + warp_id * H_ * BK, c_frag, BK, mem_row_major);
    }
    __syncthreads();  // SYNC_A2

    // ---- Reduce partials + scale + mask → s_logits_tile -----------------
    if (tid < H_ * BK) {
      const int r = tid & (BK - 1);
      float sum = 0.f;
      #pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        sum += s_mma_scratch[w * (H_ * BK) + tid];
      }
      const int k = tile * BK + r;
      const int g_idx = s_gidx[k];
      s_logits_tile[tid] = (g_idx >= 0) ? (sum * sm_scale_log2e) : -INFINITY;
    }
    __syncthreads();  // SYNC_B

    // ---- Per-warp softmax (one head per warp) ---------------------------
    //
    // Each warp computes m_new for its head, produces s_alpha[warp_id] for
    // the rescale, and writes BK=16 bf16 p-values to s_p_tile[warp_id,:].
    // lanes 0..15 do the exp2f work; others idle.
    float tile_max = -INFINITY;
    #pragma unroll
    for (int r = 0; r < BK; ++r) {
      tile_max = fmaxf(tile_max, s_logits_tile[my_h * BK + r]);
    }
    float alpha = 1.f;
    float m_new = m;
    float lane_p = 0.f;
    if (tile_max > -INFINITY) {
      m_new = fmaxf(m, tile_max);
      alpha = (m == -INFINITY) ? 0.f : exp2f(m - m_new);
      if (lane < BK) {
        const float logit = s_logits_tile[my_h * BK + lane];
        const float p_lane = (logit > -INFINITY) ? exp2f(logit - m_new) : 0.f;
        s_p_tile[my_h * BK + lane] = f_to_bf16(p_lane);
        lane_p = p_lane;
      }
      // Lanes >= BK write nothing (s_p_tile has exactly H*BK slots).
    } else {
      // tile_max == -INFINITY: no new contributions; P bf16 is 0 and alpha is
      // the "no-op" value (1.f means no rescale of c_frag).
      if (lane < BK) s_p_tile[my_h * BK + lane] = f_to_bf16(0.f);
    }
    // α broadcast to all warps (for c_frag rescale).
    if (lane == 0) s_alpha[my_h] = alpha;
    // m must be kept identical on ALL lanes so that next tile's
    // exp2f(m - m_new) works for every lane. l is only needed on lane 0
    // (we broadcast it to the warp at epilogue via s_alpha[my_h]).
    const float tile_sum_p = warp_reduce_sum(lane_p);
    m = m_new;
    if (lane == 0) l = l * alpha + tile_sum_p;
    __syncthreads();  // SYNC_C — s_p_tile, s_alpha ready for all warps

    // ---- Rescale c_frag (store → per-element α[h] mul → load) -----------
    //
    // Per-head α must scale the (m=h, n=*) elements of c_frag. Since the
    // wmma accumulator layout distributes (m, n) across a lane's 8 elements
    // non-trivially, we use the safe path: store c_frag to s_O_scratch,
    // multiply each fp32 entry by α[h] (h known from position), load back.
    if (tile > 0) {
      // Only needed after the first tile (c_frag starts at 0 anyway).
      #pragma unroll
      for (int nc = 0; nc < N_CHUNKS_PER_WARP; ++nc) {
        const int n_col = warp_id * N_PER_WARP + nc * MMA_N;
        store_matrix_sync(s_O_scratch + n_col, c_o[nc], D_CK, mem_row_major);
      }
      __syncwarp();
      // Rescale: warp owns [H=16, 32] = 512 fp32 of s_O_scratch.
      // Lane owns 16 fp32 (one per h). Use strided read: lane l reads
      // s_O[h, n_base + l%32] for h in 0..15... wait lane<32 so:
      const int n_base = warp_id * N_PER_WARP;
      float local_alpha[H_];
      #pragma unroll
      for (int h = 0; h < H_; ++h) local_alpha[h] = s_alpha[h];
      // 32 lanes × 16 h = 512 entries exactly match warp's chunk.
      #pragma unroll
      for (int h = 0; h < H_; ++h) {
        const int idx = h * D_CK + n_base + lane;
        s_O_scratch[idx] *= local_alpha[h];
      }
      __syncwarp();
      #pragma unroll
      for (int nc = 0; nc < N_CHUNKS_PER_WARP; ++nc) {
        const int n_col = warp_id * N_PER_WARP + nc * MMA_N;
        load_matrix_sync(c_o[nc], s_O_scratch + n_col, D_CK, mem_row_major);
      }
    }
    __syncthreads();  // SYNC_D — V-MMA about to read s_v_tile + s_p_tile;
                      // also guarantees no warp still writing s_O_scratch.

    // ---- V-accum MMA ----------------------------------------------------
    //
    // P[H=16, BK=16] × V[BK=16, 16 per n-chunk] = O[H=16, 16].
    // Each warp does 2 such MMAs for its owned n-chunks.
    {
      fragment<matrix_a, MMA_M, MMA_N, MMA_K, __nv_bfloat16, row_major> a_p;
      fragment<matrix_b, MMA_M, MMA_N, MMA_K, __nv_bfloat16, row_major> b_v;
      load_matrix_sync(a_p, s_p_tile, BK);
      #pragma unroll
      for (int nc = 0; nc < N_CHUNKS_PER_WARP; ++nc) {
        const int n_col = warp_id * N_PER_WARP + nc * MMA_N;
        load_matrix_sync(b_v, ckv_buf + n_col, D_CK);
        mma_sync(c_o[nc], a_p, b_v, c_o[nc]);
      }
    }
  }

  // ---- Output write ---------------------------------------------------
  //
  // c_o[nc] holds O[H=16, 16] fp32 for n_col ∈ {warp*32, warp*32 + 16}.
  // We need to store normalized bf16 to out[t, H, D_CK] (row-major, with
  // H rows). Use store_matrix_sync to dump c_o to s_O_scratch (reused),
  // then each lane writes bf16 from s_O_scratch.
  __syncthreads();  // ensure s_v_tile no longer being read; s_O_scratch
                    // was last read by the V-MMA, free to reuse.
  #pragma unroll
  for (int nc = 0; nc < N_CHUNKS_PER_WARP; ++nc) {
    const int n_col = warp_id * N_PER_WARP + nc * MMA_N;
    store_matrix_sync(s_O_scratch + n_col, c_o[nc], D_CK, mem_row_major);
  }
  __syncthreads();

  // Broadcast the warp's l to all lanes via SMEM (reuse s_alpha slot).
  if (lane == 0) s_alpha[my_h] = l;
  __syncwarp();
  const float my_l = s_alpha[my_h];

  if constexpr (G_ == 1) {
    const float inv_l = 1.f / my_l;
    __nv_bfloat162* out_row =
        reinterpret_cast<__nv_bfloat162*>(out + (t * H + my_h) * D_CK);
    // Lane writes CH_PT/2 = 8 bf162 (16 bf16) starting at ch_base = lane*16.
    #pragma unroll
    for (int c2 = 0; c2 < CH_PT / 2; ++c2) {
      const int c0 = ch_base + 2 * c2;
      const float v0 = s_O_scratch[my_h * D_CK + c0    ] * inv_l;
      const float v1 = s_O_scratch[my_h * D_CK + c0 + 1] * inv_l;
      out_row[(ch_base >> 1) + c2] =
          __halves2bfloat162(f_to_bf16(v0), f_to_bf16(v1));
    }
    if (lane == 0) lse_out[t * H + my_h] = m + log2f(my_l);
  } else {
    // Phase 7i: fp32 partials.
    float2* out_row = reinterpret_cast<float2*>(
        O_partial + ((t * H + my_h) * G_ + g) * D_CK);
    #pragma unroll
    for (int c2 = 0; c2 < CH_PT / 2; ++c2) {
      const int c0 = ch_base + 2 * c2;
      const float v0 = s_O_scratch[my_h * D_CK + c0    ];
      const float v1 = s_O_scratch[my_h * D_CK + c0 + 1];
      out_row[(ch_base >> 1) + c2] = make_float2(v0, v1);
    }
    if (lane == 0) {
      const int64_t ml_i = (t * H + my_h) * G_ + g;
      m_partial[ml_i] = m;
      l_partial[ml_i] = my_l;
    }
  }
}
