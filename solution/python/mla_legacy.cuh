// Per-(token, head) kernel, plus the split-K partial and reduce kernels.
// Included from kernel.cu inside the anonymous namespace.
//
// One CTA owns one head. It gathers sparse KV rows in tiles of 16, computes
// the score in fp32, and updates a running online-softmax output so each
// compressed-KV row is read once.

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
    float* __restrict__ O_partial,           // Phase 7i: fp32 partials [T,H,G,D_CK]
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
    // Phase 7i: partials are fp32. One thread writes one float2 (2 floats).
    reinterpret_cast<float2*>(O_partial + o_part_off)[tid] =
        make_float2(0.f, 0.f);
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
  // Phase 7i: fp32 partials — one float2 per thread.
  reinterpret_cast<float2*>(O_partial + o_part_off)[i] = make_float2(o0, o1);
}

// Reducer: combines G partial (m, l, O_unnorm) states into the final
// (out, lse).  Grid = (T, H), block = THREADS (one thread per 2-channel
// output slot).  G is a compile-time template param so the inner loops
// fully unroll and the compiler keeps s_m/s_l in registers if needed.
template <int G_>
__global__ void mla_split_reduce_kernel(
    const float* __restrict__ O_partial,  // Phase 7i: fp32 partials (was bf16)
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
    // Phase 7i: partials are fp32 (was bf16). Each lane reads 2 fp32 values.
    const float2* op2 =
        reinterpret_cast<const float2*>(O_partial + o_part_base + g * D_CK);
    const float2 of = op2[i];
    o0 += of.x * alpha;
    o1 += of.y * alpha;
  }

  const __nv_bfloat162 packed =
      __halves2bfloat162(f_to_bf16(o0 * inv_l), f_to_bf16(o1 * inv_l));
  reinterpret_cast<__nv_bfloat162*>(out + out_off)[i] = packed;
  if (tid == 0) lse_out[lse_i] = m_final + log2f(l_final);
}


