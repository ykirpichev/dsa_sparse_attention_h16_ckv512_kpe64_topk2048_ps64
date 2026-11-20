// DSA sparse MLA on ThunderKittens.
//
// The score is mm_ABt / mma_ABt: Blackwell tcgen05, BF16 shared tiles in and
// FP32 tensor memory out. One tile is 128 gathered keys by a 64-wide chunk
// of the key. Eight chunks cover the 512-d latent and a ninth adds the 64-d
// positional key. Softmax, the output mix, and the merge of partials are
// ordinary CUDA; ThunderKittens has no sparse gather and no online softmax.
// The FP8 indexer port makes the same split.
//
// A token whose last valid index sits below 128 is scored by one CTA. A
// longer token is split across G CTAs that meet at a per-token counter. G is
// chosen so every CTA is resident, which is what makes that counter open.

#include "kittens.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

using namespace kittens;

namespace {

constexpr int kHeads = 16;
constexpr int kDim = 512;
constexpr int kDimPe = 64;
constexpr int kTopK = 2048;
constexpr int kTile = 128;
constexpr int kChunk = 64;
constexpr int kWide = 128;
constexpr int kThreads = 128;
constexpr int kScoreChunks = kDim / kWide;

__device__ __forceinline__ int warp_max(int v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = max(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ int warp_sum(int v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Largest valid position in one token's index row, or -1 when there is none.
__device__ int last_valid(const int* row, int n_kv) {
    int last = -1;
    for (int i = threadIdx.x; i < kTopK; i += kThreads) {
        const int id = row[i];
        if (id >= 0 && id < n_kv) last = max(last, i);
    }
    last = warp_max(last);
    __shared__ int per_warp[4];
    if ((threadIdx.x & 31) == 0) per_warp[threadIdx.x >> 5] = last;
    __syncthreads();
    if (threadIdx.x < 32) {
        const int v = (threadIdx.x < 4) ? per_warp[threadIdx.x] : -1;
        const int all = warp_max(v);
        if (threadIdx.x == 0) per_warp[0] = all;
    }
    __syncthreads();
    return per_warp[0];
}

// 128-byte swizzle, as a 16-byte vector index. A row of 64 bf16 is 128 bytes,
// and tma_swizzle_allocator aligns the tile so this matches st_bf::operator[].
__device__ __forceinline__ int swizzle_16b(int idx) {
    return idx ^ ((idx >> 3) & 7);
}

// One 16-byte vector per 8 bf16 columns. A tile wider than 64 columns is
// stored as contiguous 64-column subtiles, each with the 128-byte swizzle.
template <int Rows, int Cols>
__device__ void fill_tile(st_bf<Rows, Cols>& dst, const __nv_bfloat16* cache, int stride,
                          const int* idx, int col, int row_limit, bool gather) {
    static_assert(Cols % kChunk == 0, "column count has to be a whole subtile");
    uint4* out = reinterpret_cast<uint4*>(dst.data);
    constexpr int kSubs = Cols / kChunk;
    constexpr int kVecs = Rows * (kChunk / 8);
    for (int s = 0; s < kSubs; ++s) {
        for (int v = threadIdx.x; v < kVecs; v += kThreads) {
            const int row = v >> 3;
            const int vec = v & 7;
            uint4 bits = make_uint4(0, 0, 0, 0);
            if (row < row_limit) {
                const int src_col = col + s * kChunk + vec * 8;
                if (gather) {
                    const int id = idx[row];
                    if (id >= 0) {
                        const __nv_bfloat16* src =
                            cache + static_cast<long long>(id) * stride + src_col;
                        bits = *reinterpret_cast<const uint4*>(src);
                    }
                } else {
                    const __nv_bfloat16* src =
                        cache + static_cast<long long>(row) * stride + src_col;
                    bits = *reinterpret_cast<const uint4*>(src);
                }
            }
            out[s * kVecs + swizzle_16b(v)] = bits;
        }
    }
}

// All 128 softmax weights, padded to 128 rows. One store feeds every output MMA.
__device__ void fill_probs(st_bf<kTile, kWide>& dst, const float prob[kHeads][kTile]) {
    uint4* out = reinterpret_cast<uint4*>(dst.data);
    constexpr int kSubs = kWide / kChunk;
    constexpr int kVecs = kTile * (kChunk / 8);
    for (int s = 0; s < kSubs; ++s) {
        for (int v = threadIdx.x; v < kVecs; v += kThreads) {
            const int row = v >> 3;
            const int vec = v & 7;
            uint4 bits = make_uint4(0, 0, 0, 0);
            if (row < kHeads) {
                __nv_bfloat16 tmp[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    tmp[j] = __float2bfloat16(prob[row][s * kChunk + vec * 8 + j]);
                bits = *reinterpret_cast<const uint4*>(tmp);
            }
            out[s * kVecs + swizzle_16b(v)] = bits;
        }
    }
}

// Issue one MMA and wait for it. `phase` is the mbarrier parity; it flips
// after every wait. Every thread has to execute the wait. AB is the output
// mix (probs @ values); ABt is the score (keys @ queries^T).
template <bool AB, typename TA, typename TB>
__device__ void mma_chunk(tt<float, kTile, kChunk>& acc, TA& a, TB& b, semaphore& sem,
                          int& phase, bool clear) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if constexpr (AB) {
            if (clear) mm_AB(acc, a, b, sem);
            else mma_AB(acc, a, b, sem);
        } else {
            if (clear) mm_ABt(acc, a, b, sem);
            else mma_ABt(acc, a, b, sem);
        }
    }
    wait(sem, phase);
    phase ^= 1;
}

__device__ void dump_acc(tt<float, kTile, kChunk>& acc, st_fl<kTile, kChunk>& dst) {
    rt_fl<32, kChunk> regs;
    warpgroup::load_async(regs, acc);
    tensor_load_wait();
    warpgroup::sync(warpgroup::groupid() + 1);
    warpgroup::store(dst, regs);
    __syncthreads();
}

struct Running {
    float acc[kHeads][kDim];
    float prob[kHeads][kTile];
    float score[kTile][kHeads];
    float m[kHeads], l[kHeads], alpha[kHeads], m_tile[kHeads], l_tile[kHeads];
    int idx[kTile];
};

__device__ void attend(const __nv_bfloat16* q, int stride_q,
                       const __nv_bfloat16* qp, int stride_qp,
                       const __nv_bfloat16* ckv, const __nv_bfloat16* kpe,
                       const int* row, int n_kv, int last, float sm_log2e,
                       int start, int stride, int ntiles,
                       st_bf<kTile, kWide>& sk, st_bf<kChunk, kWide>& sq,
                       st_bf<kTile, kWide>& pk,
                       st_bf<kTile, kChunk>& k_st, st_bf<kChunk, kChunk>& q_st,
                       st_fl<kTile, kChunk>& c_smem, tt<float, kTile, kChunk>& acc_tm,
                       semaphore& sem, int& phase, Running& s) {
    const int tx = threadIdx.x;
    for (int e = tx; e < kHeads * kDim; e += kThreads) s.acc[e / kDim][e % kDim] = 0.f;
    if (tx < kHeads) {
        s.m[tx] = -INFINITY;
        s.l[tx] = 0.f;
    }
    __syncthreads();

    for (int tile = 0; tile < ntiles; ++tile) {
        const int origin = start + tile * kTile * stride;
        if (origin > last) break;

        for (int i = tx; i < kTile; i += kThreads) {
            const int pos = origin + i * stride;
            int id = -1;
            if (pos <= last && pos < kTopK) {
                id = row[pos];
                if (id < 0 || id >= n_kv) id = -1;
            }
            s.idx[i] = id;
        }
        __syncthreads();
        __shared__ int alive;
        if (tx == 0) alive = 0;
        __syncthreads();
        int local = 0;
        for (int i = tx; i < kTile; i += kThreads) local += s.idx[i] >= 0;
        local = warp_sum(local);
        if ((tx & 31) == 0) atomicAdd(&alive, local);
        __syncthreads();
        if (alive == 0) continue;

        for (int chunk = 0; chunk < kScoreChunks; ++chunk) {
            fill_tile(sk, ckv, kDim, s.idx, chunk * kWide, kTile, true);
            fill_tile(sq, q, stride_q, nullptr, chunk * kWide, kHeads, false);
            mma_chunk<false>(acc_tm, sk, sq, sem, phase, chunk == 0);
        }
        fill_tile(k_st, kpe, kDimPe, s.idx, 0, kTile, true);
        fill_tile(q_st, qp, stride_qp, nullptr, 0, kHeads, false);
        mma_chunk<false>(acc_tm, k_st, q_st, sem, phase, false);
        dump_acc(acc_tm, c_smem);

        for (int e = tx; e < kTile * kHeads; e += kThreads) {
            const int r = e / kHeads;
            const int h = e % kHeads;
            s.score[r][h] = (s.idx[r] >= 0) ? c_smem[{r, h}] * sm_log2e : -INFINITY;
        }
        __syncthreads();

        if (tx < kHeads) {
            float m_tile = -INFINITY;
            for (int i = 0; i < kTile; ++i) m_tile = fmaxf(m_tile, s.score[i][tx]);
            float l_tile = 0.f;
            for (int i = 0; i < kTile; ++i) {
                const float s_i = s.score[i][tx];
                const float p = (m_tile > -1.0e20f && s_i > -1.0e20f) ? exp2f(s_i - m_tile) : 0.f;
                s.prob[tx][i] = p;
                l_tile += p;
            }
            float alpha = 1.f;
            if (m_tile > -1.0e20f) {
                const float m_new = fmaxf(s.m[tx], m_tile);
                // Probs were computed against the tile max. The running total
                // lives at m_new, so both sides have to move there.
                alpha = (s.m[tx] > -1.0e20f) ? exp2f(s.m[tx] - m_new) : 0.f;
                const float scale_tile = exp2f(m_tile - m_new);
                for (int i = 0; i < kTile; ++i) s.prob[tx][i] *= scale_tile;
                s.l[tx] = s.l[tx] * alpha + l_tile * scale_tile;
                s.m[tx] = m_new;
            }
            s.alpha[tx] = alpha;
        }
        __syncthreads();

        for (int d = tx; d < kDim; d += kThreads) {
            for (int h = 0; h < kHeads; ++h) s.acc[h][d] *= s.alpha[h];
        }
        __syncthreads();

        // One MMA per 64 value columns. The weight tile already holds all 128
        // keys, so there is no K-split.
        fill_probs(pk, s.prob);
        for (int chunk = 0; chunk < kDim / kChunk; ++chunk) {
            const int col = chunk * kChunk;
            fill_tile(k_st, ckv, kDim, s.idx, col, kTile, true);
            mma_chunk<true>(acc_tm, pk, k_st, sem, phase, true);
            dump_acc(acc_tm, c_smem);
            for (int c = tx; c < kChunk; c += kThreads)
                for (int h = 0; h < kHeads; ++h)
                    s.acc[h][col + c] += c_smem[{h, c}];
            __syncthreads();
        }
    }
}

__device__ void publish(float* part_acc, float* part_m, float* part_l, int slot_base, Running& s) {
    const int tx = threadIdx.x;
    for (int d = tx; d < kDim; d += kThreads) {
        for (int h = 0; h < kHeads; ++h)
            part_acc[(slot_base + h) * kDim + d] = s.acc[h][d];
    }
    if (tx < kHeads) {
        part_m[slot_base + tx] = s.m[tx];
        part_l[slot_base + tx] = s.l[tx];
    }
}

__device__ void write_final(const Running& s, __nv_bfloat16* out, int stride_oh, float* lse) {
    const int tx = threadIdx.x;
    for (int d = tx; d < kDim; d += kThreads) {
        for (int h = 0; h < kHeads; ++h) {
            const float inv = s.l[h] > 0.f ? 1.f / s.l[h] : 0.f;
            out[h * stride_oh + d] = __float2bfloat16(s.acc[h][d] * inv);
        }
    }
    if (tx < kHeads)
        lse[tx] = s.l[tx] > 0.f ? s.m[tx] + log2f(s.l[tx]) : -INFINITY;
}

// G key-groups per token. G == 1 scores the whole row in one CTA.
template <int G>
__global__ void __launch_bounds__(kThreads)
sparse_mla(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
           const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
           const int* __restrict__ indices, __nv_bfloat16* __restrict__ out, float* __restrict__ lse,
           float* __restrict__ part_acc, float* __restrict__ part_m, float* __restrict__ part_l,
           int* __restrict__ counter, float sm_log2e, int n_kv,
           long long stride_qt, long long stride_qh, long long stride_pt, long long stride_ph,
           long long stride_ot, long long stride_oh) {
    constexpr int kTiles = kTopK / G / kTile;
    const int token = blockIdx.x;
    const int slot = blockIdx.y;
    const int tx = threadIdx.x;

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    auto& k_st = al.allocate<st_bf<kTile, kChunk>>();
    auto& q_st = al.allocate<st_bf<kChunk, kChunk>>();
    auto& c_smem = al.allocate<st_fl<kTile, kChunk>>();
    auto& sk = al.allocate<st_bf<kTile, kWide>>();
    auto& sq = al.allocate<st_bf<kChunk, kWide>>();
    auto& pk = al.allocate<st_bf<kTile, kWide>>();

    __shared__ semaphore sem;
    __shared__ alignas(1024) Running running;
    __shared__ uint32_t tmem_addr;
    __shared__ int target;

    if (tx == 0) init_semaphore(sem, 0, 1);
    tensor_allocator<1, 1, false> tm{};
    if (tx < 32) tm.provision(tmem_addr);
    __syncthreads();
    tm.set_addr(tmem_addr);
    tt<float, kTile, kChunk> acc_tm = tm.allocate<tt<float, kTile, kChunk>>(0);
    int phase = 0;

    const int* row = indices + static_cast<long long>(token) * kTopK;
    const int last = last_valid(row, n_kv);
    const __nv_bfloat16* q = q_nope + token * stride_qt;
    const __nv_bfloat16* qp = q_pe + token * stride_pt;
    __nv_bfloat16* out_row = out + token * stride_ot;
    float* lse_row = lse + static_cast<long long>(token) * kHeads;

    // Short tokens: one CTA, the first 128 index slots, no partials.
    if (G > 1 && last < kTile) {
        if (slot == 0 && last >= 0) {
            attend(q, stride_qh, qp, stride_ph, ckv, kpe, row, n_kv, last, sm_log2e,
                   0, 1, 1, sk, sq, pk, k_st, q_st, c_smem, acc_tm, sem, phase, running);
            write_final(running, out_row, stride_oh, lse_row);
        } else if (slot == 0) {
            for (int d = tx; d < kDim; d += kThreads)
                for (int h = 0; h < kHeads; ++h) out_row[h * stride_oh + d] = __float2bfloat16(0.f);
            if (tx < kHeads) lse_row[tx] = -INFINITY;
        }
        __syncthreads();
        if (tx < 32) tm.deprovision();
        return;
    }

    const int ntiles = (G == 1) ? kTiles : kTiles;
    attend(q, stride_qh, qp, stride_ph, ckv, kpe, row, n_kv, last, sm_log2e,
           (G == 1) ? 0 : slot, (G == 1) ? 1 : G, ntiles,
           sk, sq, pk, k_st, q_st, c_smem, acc_tm, sem, phase, running);

    if (G == 1) {
        write_final(running, out_row, stride_oh, lse_row);
        __syncthreads();
        if (tx < 32) tm.deprovision();
        return;
    }

    // Read the counter before this CTA's own increment. The next multiple of
    // G is the value it reaches once every key-group has published.
    if (tx == 0) target = (counter[token] & -G) + G;
    __syncthreads();
    const int slot_base = (token * G + slot) * kHeads;
    publish(part_acc, part_m, part_l, slot_base, running);
    __threadfence();
    __syncthreads();
    if (tx < 32) tm.deprovision();
    __syncthreads();
    if (tx == 0) {
        atomicAdd(counter + token, 1);
        int count;
        do {
            count = *static_cast<volatile int*>(counter + token);
        } while (count - target < 0);
    }
    __syncthreads();

    constexpr int kWidth = kDim / G;
    for (int c = tx; c < kWidth; c += kThreads) {
        const int col = slot * kWidth + c;
        float m_top[kHeads], l_sum[kHeads], mix[kHeads];
        for (int h = 0; h < kHeads; ++h) {
            m_top[h] = -INFINITY;
            l_sum[h] = 0.f;
            mix[h] = 0.f;
        }
        for (int g = 0; g < G; ++g) {
            const int base = (token * G + g) * kHeads;
            for (int h = 0; h < kHeads; ++h) m_top[h] = fmaxf(m_top[h], part_m[base + h]);
        }
        for (int g = 0; g < G; ++g) {
            const int base = (token * G + g) * kHeads;
            for (int h = 0; h < kHeads; ++h) {
                const float m_ref = m_top[h] > -1.0e20f ? m_top[h] : 0.f;
                const float weight = exp2f(part_m[base + h] - m_ref);
                l_sum[h] += part_l[base + h] * weight;
                mix[h] += part_acc[(base + h) * kDim + col] * weight;
            }
        }
        for (int h = 0; h < kHeads; ++h) {
            const float inv = l_sum[h] > 0.f ? 1.f / l_sum[h] : 0.f;
            out_row[h * stride_oh + col] = __float2bfloat16(mix[h] * inv);
        }
        if (c == 0 && slot == 0) {
            for (int h = 0; h < kHeads; ++h)
                lse_row[h] = l_sum[h] > 0.f ? m_top[h] + log2f(l_sum[h]) : -INFINITY;
        }
    }
}

int dynamic_smem() {
    // The swizzle allocator rounds each tile up, so the raw sizes are short.
    return static_cast<int>(sizeof(st_bf<kTile, kChunk>) + sizeof(st_bf<kChunk, kChunk>) +
                            sizeof(st_fl<kTile, kChunk>) + sizeof(st_bf<kTile, kWide>) * 2 +
                            sizeof(st_bf<kChunk, kWide>) + 4096);
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

template <int G>
void launch_g(const __nv_bfloat16* q, const __nv_bfloat16* qp, const __nv_bfloat16* ckv,
              const __nv_bfloat16* kpe, const int* indices, __nv_bfloat16* out, float* lse,
              float* part_acc, float* part_m, float* part_l, int* counter, float sm_log2e,
              int n_tokens, int n_kv, long long sqt, long long sqh, long long spt, long long sph,
              long long sot, long long soh, cudaStream_t stream) {
    auto* fn = sparse_mla<G>;
    const int smem = dynamic_smem();
    static int configured = -1;
    if (configured != smem) {
        check(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "smem");
        configured = smem;
    }
    fn<<<dim3(n_tokens, G), kThreads, smem, stream>>>(
        q, qp, ckv, kpe, indices, out, lse, part_acc, part_m, part_l, counter, sm_log2e, n_kv,
        sqt, sqh, spt, sph, sot, soh);
    check(cudaGetLastError(), "sparse_mla");
}

struct Buffers {
    torch::Tensor acc, m, l, counter;
    int tokens = -1;
    int g = -1;
};

Buffers& buffers_for(int n_tokens, int g, torch::Device device) {
    static Buffers buf;
    if (!buf.acc.defined() || buf.tokens != n_tokens || buf.g != g || buf.acc.device() != device) {
        buf.acc = torch::empty({n_tokens, g, kHeads, kDim}, torch::dtype(torch::kFloat32).device(device));
        buf.m = torch::empty({n_tokens, g, kHeads}, buf.acc.options());
        buf.l = torch::empty({n_tokens, g, kHeads}, buf.acc.options());
        buf.counter = torch::zeros({n_tokens}, torch::dtype(torch::kInt32).device(device));
        buf.tokens = n_tokens;
        buf.g = g;
    }
    return buf;
}

}  // namespace

void run(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv_cache, torch::Tensor kpe_cache,
         torch::Tensor sparse_indices, double sm_scale, torch::Tensor out, torch::Tensor lse) {
    const int n_tokens = q_nope.size(0);
    auto ckv = ckv_cache.reshape({-1, kDim});
    auto kpe = kpe_cache.reshape({-1, kDimPe});
    if (!ckv.is_contiguous()) ckv = ckv.contiguous();
    if (!kpe.is_contiguous()) kpe = kpe.contiguous();
    if (!sparse_indices.is_contiguous()) sparse_indices = sparse_indices.contiguous();

    int n_sm = 1;
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, q_nope.get_device());
    int g = 16;
    while (g > 1 && n_tokens * g > n_sm) g >>= 1;
    if (g < 8) g = 1;

    auto& buf = buffers_for(n_tokens, max(g, 1), q_nope.device());
    const float sm_log2e = static_cast<float>(sm_scale) * 1.4426950408889634f;
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* q = reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr());
    const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr());
    const auto* ck = reinterpret_cast<const __nv_bfloat16*>(ckv.data_ptr());
    const auto* kp = reinterpret_cast<const __nv_bfloat16*>(kpe.data_ptr());
    auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    auto go = [&](auto launcher) {
        launcher(q, qp, ck, kp, sparse_indices.data_ptr<int>(), o, lse.data_ptr<float>(),
                 buf.acc.data_ptr<float>(), buf.m.data_ptr<float>(), buf.l.data_ptr<float>(),
                 buf.counter.data_ptr<int>(), sm_log2e, n_tokens, static_cast<int>(ckv.size(0)),
                 q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1),
                 out.stride(0), out.stride(1), stream);
    };
    if (g == 16) go(launch_g<16>);
    else if (g == 8) go(launch_g<8>);
    else go(launch_g<1>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
}
