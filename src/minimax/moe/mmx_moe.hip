#include "minimax/engine/route_identity.hpp"
// minimax-m3 decode MoE and dense EXL3 linears (identity and launch structure: mmx_moe.hpp).
#include "minimax/moe/mmx_moe.hpp"

#include "minimax/moe/mmx_trellis_device.hpp"
#include "strix/minimax_msa.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifndef STRIX_MM_GU4_SCALAR
#define STRIX_MM_GU4_SCALAR 1
#endif

namespace strix::minimax::moe::opt {
namespace {

namespace d = dev;
namespace route = strix::minimax::route;

constexpr unsigned kOutBlocks = kHidden / 128;    // 48 down output blocks
constexpr unsigned kInterBlocks = kInter / 128;   // 12 gate/up output blocks
constexpr unsigned kRouterWaves = 8, kRouterCtas = kExperts / kRouterWaves, kRouterChunk = 1536;
constexpr unsigned kGateSplits = d::vector_splits(kHidden, kInter);  // 16
constexpr unsigned kDownSplits = d::vector_splits(kInter, kHidden);  // 4
static_assert(kGateSplits == 16 && kDownSplits == 4, "strix-trellis split policy for the MiniMax expert shapes");
constexpr int kDefaultDepth = 0;
constexpr int kDefaultVersion = 4;  // STRIX_MMX_MOE_V: 4 = gate_up4 + down4 (default, measurement: never slower); 1 = v1; 3 = v3
constexpr float kAlpha = 1.702F, kLimit = 7.0F, kRouteScale = 2.0F, kSumEps = 1e-20F, kNormEps = 1e-6F;

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }
constexpr std::size_t kOffScores = align256(sizeof(Dispatch));
constexpr std::size_t kOffChoice = kOffScores + align256(kMaxRows * kExperts * 4);
constexpr std::size_t kOffCounters = kOffChoice + align256(kMaxRows * kExperts * 4);
constexpr unsigned kGuCounters = (kMaxSlots + 1) * kInterBlocks;
constexpr unsigned kCounters = 1 + kOutBlocks + kGuCounters;
constexpr std::size_t kOffXt = kOffCounters + align256(kCounters * 4);
constexpr std::size_t kOffAt = kOffXt + align256(std::size_t{kMaxPairs} * 2 * kHidden * 2);
constexpr std::size_t kOffY = kOffAt + align256(std::size_t{kMaxPairs} * kInter * 2);
constexpr std::size_t kOffGu = kOffY + align256(std::size_t{kMaxPairs} * kHidden * 4);
constexpr std::size_t kOffPartGu = kOffGu + align256(std::size_t{kMaxPairs} * 2 * kInter * 4);
constexpr std::size_t kOffPartDn = kOffPartGu + align256(std::size_t{kMaxPairs} * 2 * 16 * kInter * 4);
constexpr std::size_t kOffCounters4 = kOffPartDn + align256(std::size_t{kMaxPairs} * 4 * kHidden * 4);
constexpr unsigned kCounters4 = (kMaxSlots + 1) * kInterBlocks + kOutBlocks;
constexpr std::size_t kOffPipe = kOffCounters4 + align256(std::size_t{kCounters4} * 4);
constexpr unsigned kPipeWords = 2 + (kMaxSlots + 1) * kInterBlocks;  // epoch, ticket, gate/up block ready epochs
constexpr std::size_t kMoeBytes = kOffPipe + align256(std::size_t{kPipeWords} * 4);

struct Args {
    unsigned rows;
    float* h;  // fused post-exchange when set
    const float* p0;
    const float* p1;
    const float* norm_w;
    const float* y;
    const float* router_w;
    const float* bias;
    const ExpertSlices* experts;
    ExpertSlices shared;
    float* out;
    std::uint32_t* ids;
    float* weights;
    unsigned char* ws;
    PublishTarget pub{};  // down4: publish `out` (STRIX_MM_XB_PUBLISH)
    const std::uint32_t* controlled_ids = nullptr;
    const float* controlled_weights = nullptr;
    std::uint32_t* single_reader_counts = nullptr;
    bool single_stream = false;
    bool interleave_gu = true;
    unsigned down_split = 0;  // STRIX_MMX_MOE_DOWN_SPLIT mode; route::moe_down_split decides per launch
    hipEvent_t down_done = nullptr;  // host enqueue boundary only; never read on device
    __device__ Dispatch* disp() const { return reinterpret_cast<Dispatch*>(ws); }
    __device__ float* scores() const { return reinterpret_cast<float*>(ws + kOffScores); }
    __device__ float* choice() const { return reinterpret_cast<float*>(ws + kOffChoice); }
    __device__ std::uint32_t* counters() const { return reinterpret_cast<std::uint32_t*>(ws + kOffCounters); }
    // xt: per (pair, matrix 0 gate / 1 up) the transformed 6144 inputs; at: per pair the transformed 1536 down
    // inputs; yslot: per pair the 6144 down outputs.
    __device__ _Float16* xt(unsigned pair, unsigned m) const {
        return reinterpret_cast<_Float16*>(ws + kOffXt) + (static_cast<std::size_t>(pair) * 2 + m) * kHidden;
    }
    __device__ _Float16* at(unsigned pair) const {
        return reinterpret_cast<_Float16*>(ws + kOffAt) + static_cast<std::size_t>(pair) * kInter;
    }
    __device__ float* yslot(unsigned pair) const {
        return reinterpret_cast<float*>(ws + kOffY) + static_cast<std::size_t>(pair) * kHidden;
    }
    __device__ float* gu(unsigned pair, unsigned m) const {  // v3: gate (m 0) / up (m 1) outputs after svh
        return reinterpret_cast<float*>(ws + kOffGu) + (static_cast<std::size_t>(pair) * 2 + m) * kInter;
    }
    __device__ std::uint32_t* gu_counter(unsigned slot, unsigned block) const {
        return counters() + 1 + kOutBlocks + slot * kInterBlocks + block;
    }
    // v4: split partials (raw per split, summed in ascending split order by the finishing CTA) and their counters.
    __device__ float* part_gu(unsigned pair, unsigned m, unsigned split) const {
        return reinterpret_cast<float*>(ws + kOffPartGu) + ((static_cast<std::size_t>(pair) * 2 + m) * 16 + split) * kInter;
    }
    __device__ float* part_dn(unsigned pair, unsigned split) const {
        return reinterpret_cast<float*>(ws + kOffPartDn) + (static_cast<std::size_t>(pair) * 4 + split) * kHidden;
    }
    __device__ std::uint32_t* gu4_counter(unsigned slot, unsigned block) const {
        return reinterpret_cast<std::uint32_t*>(ws + kOffCounters4) + slot * kInterBlocks + block;
    }
    __device__ std::uint32_t* dn4_counter(unsigned block) const {
        return reinterpret_cast<std::uint32_t*>(ws + kOffCounters4) + (kMaxSlots + 1) * kInterBlocks + block;
    }
    // routed pipeline (STRIX_MMX_MOE_PIPE): route epoch (bumped by every route_tail), CTA ticket, and the epoch at which
    // each (slot, 128-channel gate/up block) of the down inputs was finished
    __device__ std::uint32_t* pipe_epoch() const { return reinterpret_cast<std::uint32_t*>(ws + kOffPipe); }
    __device__ std::uint32_t* pipe_ticket() const { return pipe_epoch() + 1; }
    __device__ std::uint32_t* gu_ready(unsigned slot, unsigned block) const { return pipe_epoch() + 2 + slot * kInterBlocks + block; }
};

__device__ __forceinline__ float load_agent(const float* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// True in every thread of the CTA that arrives last at `counter` (expected arrivals); that CTA resets the counter.
__device__ __forceinline__ bool last_arrival(std::uint32_t* counter, unsigned expected) {
    __shared__ unsigned last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned old = atomicAdd(counter, 1U);
        last = old + 1U == expected ? 1U : 0U;
        if (last) *counter = 0;
    }
    __syncthreads();
    if (last) __threadfence();
    return last != 0;
}

// decode lean producer publish (PublishTarget, STRIX_MM_XB_PUBLISH): ALL threads of every CTA that FINISHES an output
// tile call it once after that tile's last store (the finisher is decided by a CTA-uniform last arrival); `blocks` =
// tiles of the message (48 for a 6144-output row). Each lane releases its own stores at
// agent scope (no cache write-back), the CTAs count on an agent-scope counter, and only the last CTA resets it,
// releases at system scope and stores the sequence into the host-visible flag. flag == nullptr: nothing.
__device__ __forceinline__ void lean_publish(const PublishTarget& p, unsigned blocks) {
    if (p.flag == nullptr) return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const unsigned old = __hip_atomic_fetch_add(p.counter, 1U, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (old + 1U == blocks) {
            __hip_atomic_store(p.counter, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");  // system scope: every CTA's message stores before the flag
            __hip_atomic_store(p.flag, p.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

// ---------------------------------------------------------------- router + top-4 + dispatch table
template <unsigned NT>
__device__ void route_tail(const Args& a) {
    // Top-4 and dispatch table, built in LDS by parallel threads (only stores go to global memory: a single thread
    // walking global/private arrays serialized ~100 us of dependent accesses).
    __shared__ float s_ch[kMaxRows][kExperts], s_sc[kMaxRows][kExperts], s_w[kMaxRows][kTopE];
    __shared__ std::uint32_t s_id[kMaxRows][kTopE], slot_of[kExperts], s_present[kExperts], s_expert[kMaxSlots],
        s_nrows[kMaxSlots], s_first[kMaxSlots + 1], s_count;
    const unsigned t = threadIdx.x, rows = a.rows;
    for (unsigned q = t; q < rows * kExperts; q += NT) {
        s_ch[q / kExperts][q % kExperts] = load_agent(a.choice() + q);
        s_sc[q / kExperts][q % kExperts] = load_agent(a.scores() + q);
    }
    __syncthreads();
    Dispatch* dp = a.disp();
    if (t < rows) {  // route_row / post_exchange_kernel selection: k-th best by (choice desc, id asc)
        std::uint32_t best[kTopE];
        for (unsigned k = 0; k < kTopE; ++k) {
            std::uint32_t pick = kExperts;
            for (std::uint32_t x = 0; x < kExperts; ++x) {
                bool taken = false;
                for (unsigned q = 0; q < k; ++q) taken |= best[q] == x;
                if (taken) continue;
                if (pick == kExperts || s_ch[t][x] > s_ch[t][pick]) pick = x;  // ascending scan: ties keep the smaller id
            }
            best[k] = pick;
        }
        float sum = 0.0F;
        for (unsigned k = 0; k < kTopE; ++k) sum = sum + s_sc[t][best[k]];
        for (unsigned k = 0; k < kTopE; ++k) {
            const float wk = (s_sc[t][best[k]] / (sum + kSumEps)) * kRouteScale;
            const unsigned id = a.controlled_ids != nullptr ? a.controlled_ids[t * kTopE + k] : best[k];
            const float weight = a.controlled_weights != nullptr ? a.controlled_weights[t * kTopE + k] : wk;
            s_id[t][k] = id;
            s_w[t][k] = weight;
            dp->ids[t][k] = id;
            dp->weights[t][k] = weight;
            if (a.ids != nullptr) a.ids[t * kTopE + k] = id;
            if (a.weights != nullptr) a.weights[t * kTopE + k] = weight;
        }
    }
    __syncthreads();
    if (t < kExperts) {
        std::uint32_t present = 0;
        for (unsigned q = 0; q < rows * kTopE; ++q) present |= s_id[q / kTopE][q % kTopE] == t ? 1U : 0U;
        s_present[t] = present;
    }
    __syncthreads();
    if (t == 0) {  // slots = distinct experts in ascending id
        unsigned count = 0;
        for (unsigned x = 0; x < kExperts; ++x) {
            slot_of[x] = count;
            if (s_present[x]) s_expert[count++] = x;
        }
        s_count = count;
    }
    __syncthreads();
    const unsigned count = s_count;
    if (t < count) {  // each slot's rows in ascending order
        const std::uint32_t x = s_expert[t];
        unsigned n = 0;
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned k = 0; k < kTopE; ++k)
                if (s_id[r][k] == x) dp->row[t][n++] = r;
        s_nrows[t] = n;
        dp->expert[t] = x;
        dp->nrows[t] = n;
    }
    __syncthreads();
    if (t == 0) {  // pairs numbered slot by slot, then the shared expert's rows
        unsigned pairs = 0;
        for (unsigned q = 0; q < count; ++q) {
            s_first[q] = pairs;
            dp->first[q] = pairs;
            pairs += s_nrows[q];
        }
        unsigned singles = 0, reuse = 0;
        for (unsigned q = 0; q < count; ++q) {
            if (s_nrows[q] == 1) dp->single_slot[singles++] = q;
            else dp->reuse_slot[reuse++] = q;
        }
        dp->single_count = singles;
        dp->reuse_count = reuse;
        s_first[kMaxSlots] = pairs;
        dp->first[kSharedSlot] = pairs;
        dp->nrows[kSharedSlot] = rows;
        dp->count = count;
        dp->rows = rows;
        const std::uint32_t epoch = *a.pipe_epoch() + 1U;
        if (epoch == 0)  // wrap: no stale ready word may carry the reused epoch 1
            for (unsigned q = 0; q < (kMaxSlots + 1) * kInterBlocks; ++q) a.gu_ready(0, 0)[q] = 0;
        *a.pipe_epoch() = epoch != 0 ? epoch : 1U;
    }
    __syncthreads();
    if (t < rows * kTopE) {  // per row: its pairs in ascending expert id
        const unsigned r = t / kTopE, k = t % kTopE;
        const std::uint32_t x = s_id[r][k];
        unsigned rank = 0;  // position of x among the row's (distinct) ids
        for (unsigned q = 0; q < kTopE; ++q) rank += s_id[r][q] < x ? 1U : 0U;
        unsigned local = 0;  // rows before r that selected x
        for (unsigned q = 0; q < r; ++q)
            for (unsigned kk = 0; kk < kTopE; ++kk) local += s_id[q][kk] == x ? 1U : 0U;
        dp->row_pair[r][rank] = s_first[slot_of[x]] + local;
        dp->row_weight[r][rank] = s_w[r][k];
        dp->row_expert[r][rank] = x;
    }
}

template <bool Fused>
__global__ __launch_bounds__(256) void router_kernel(Args a) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    // Fused post-exchange: every CTA recomputes x = h + (p0 + p1) and the norm scale of each row with the order of
    // post_exchange_kernel (thread t chains fmaf over j = t, t + 256, ...; tree p[t] += p[t + s]); only the last CTA
    // writes h and y (after every CTA has read the old h).
    __shared__ float red[256], rs_s[kMaxRows];
    if constexpr (Fused) {
        for (unsigned r = 0; r < a.rows; ++r) {
            const std::size_t o = static_cast<std::size_t>(r) * kHidden;
            float acc = 0.0F;
            for (unsigned j = threadIdx.x; j < kHidden; j += 256) {
                const float x = a.h[o + j] + (a.p0[o + j] + a.p1[o + j]);
                acc = __builtin_fmaf(x, x, acc);
            }
            red[threadIdx.x] = acc;
            __syncthreads();
            for (unsigned sstep = 128; sstep >= 1; sstep >>= 1U) {
                if (threadIdx.x < sstep) red[threadIdx.x] = red[threadIdx.x] + red[threadIdx.x + sstep];
                __syncthreads();
            }
            if (threadIdx.x == 0) rs_s[r] = 1.0F / __builtin_sqrtf(red[0] / static_cast<float>(kHidden) + kNormEps);
            __syncthreads();
        }
    }
    const unsigned e = blockIdx.x * kRouterWaves + wave;
    const float* w = a.router_w + static_cast<std::size_t>(e) * kHidden;
    float acc[kMaxRows];
#pragma unroll
    for (unsigned r = 0; r < kMaxRows; ++r) acc[r] = 0.0F;
    // The rows' inputs go through LDS in chunks (every wave of the CTA reads them; from global they were re-read per
    // wave); each lane's chain still runs over j = lane, lane + 32, ... in ascending order across the chunks.
    __shared__ float ys[kMaxRows][kRouterChunk];
    for (unsigned c0 = 0; c0 < kHidden; c0 += kRouterChunk) {
        __syncthreads();
        for (unsigned q = threadIdx.x; q < a.rows * kRouterChunk; q += 256) {
            const std::size_t gi = static_cast<std::size_t>(q / kRouterChunk) * kHidden + c0 + q % kRouterChunk;
            if constexpr (Fused) {
                const float x = a.h[gi] + (a.p0[gi] + a.p1[gi]);
                ys[q / kRouterChunk][q % kRouterChunk] = (x * rs_s[q / kRouterChunk]) * (1.0F + a.norm_w[c0 + q % kRouterChunk]);
            } else {
                ys[q / kRouterChunk][q % kRouterChunk] = a.y[gi];
            }
        }
        __syncthreads();
        const float* wc = w + c0;
#pragma unroll 16
        for (unsigned i = 0; i < kRouterChunk / 32; ++i) {
            const unsigned j = lane + 32U * i;
            const float wj = wc[j];
#pragma unroll
            for (unsigned r = 0; r < kMaxRows; ++r)
                if (r < a.rows) acc[r] = __builtin_fmaf(ys[r][j], wj, acc[r]);
        }
    }
#pragma unroll
    for (unsigned r = 0; r < kMaxRows; ++r) {
        if (r >= a.rows) break;
        float v = acc[r];
        for (unsigned dd = 16; dd >= 1; dd >>= 1U) v = v + __shfl_xor(v, static_cast<int>(dd), 32);
        if (lane == 0) {
            const float s = 1.0F / (1.0F + msa::exp_f(-v));
            a.scores()[r * kExperts + e] = s;
            a.choice()[r * kExperts + e] = s + a.bias[e];
        }
    }
    if (!last_arrival(a.counters(), kRouterCtas)) return;
    if constexpr (Fused) {  // every CTA has read the old h: write h and y
        for (unsigned q = threadIdx.x; q < a.rows * kHidden; q += 256) {
            const float x = a.h[q] + (a.p0[q] + a.p1[q]);
            a.h[q] = x;
            const_cast<float*>(a.y)[q] = (x * rs_s[q / kHidden]) * (1.0F + a.norm_w[q % kHidden]);
        }
    }
    route_tail<256>(a);
}

// Router v2 (STRIX_MMX_ROUTER=2, inputs y already in memory): one CTA per expert loads its 24 KB weight row into LDS
// with wide loads (all in flight at once), then wave r computes row r's 32 lane chains (j = lane, lane + 32, ...,
// ascending; the same order and bits as route_row) reading the weights from LDS. The last of the 128 CTAs builds
// the top-4 and the dispatch table.
__global__ __launch_bounds__(256) void router2_kernel(Args a) {
    __shared__ float wsm[kHidden];
    const unsigned e = blockIdx.x, wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    {
        const auto* src = reinterpret_cast<const float4*>(a.router_w + static_cast<std::size_t>(e) * kHidden);
        auto* dst = reinterpret_cast<float4*>(wsm);
        for (unsigned q = threadIdx.x; q < kHidden / 4; q += 256) dst[q] = src[q];
    }
    __syncthreads();
    if (wave < a.rows) {
        const float* yr = a.y + static_cast<std::size_t>(wave) * kHidden;
        float acc = 0.0F;
#pragma unroll 16
        for (unsigned i = 0; i < kHidden / 32; ++i) {
            const unsigned j = lane + 32U * i;
            acc = __builtin_fmaf(yr[j], wsm[j], acc);
        }
        for (unsigned dd = 16; dd >= 1; dd >>= 1U) acc = acc + __shfl_xor(acc, static_cast<int>(dd), 32);
        if (lane == 0) {
            const float sc = 1.0F / (1.0F + msa::exp_f(-acc));
            a.scores()[wave * kExperts + e] = sc;
            a.choice()[wave * kExperts + e] = sc + a.bias[e];
        }
    }
    if (!last_arrival(a.counters(), kExperts)) return;
    route_tail<256>(a);
}

__device__ __forceinline__ const ExpertSlices& slot_slices(const Args& a, const Dispatch* dp, unsigned slot) {
    return slot == kSharedSlot ? a.shared : a.experts[dp->expert[slot]];
}

// ---------------------------------------------------------------- input transforms of gate/up (routed + shared)
__global__ __launch_bounds__(256) void prepare_kernel(Args a) {
    const Dispatch* dp = a.disp();
    const unsigned slot = blockIdx.x, m = blockIdx.y;
    if (slot != kSharedSlot && slot >= dp->count) return;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const ExpertSlices& E = slot_slices(a, dp, slot);
    const std::uint16_t* suh = m == 0 ? E.gate.suh : E.up.suh;
    const unsigned nrows = dp->nrows[slot], first = dp->first[slot];
    constexpr unsigned blocks = kHidden / 128;
    for (unsigned job = wave; job < nrows * blocks; job += 8) {
        const unsigned lr = job / blocks, b = job % blocks;
        const unsigned row = slot == kSharedSlot ? lr : dp->row[slot][lr];
        d::transform_block(a.y + static_cast<std::size_t>(row) * kHidden, suh, a.xt(first + lr, m), b * 128U, lane);
    }
}

// ---------------------------------------------------------------- gate + up (+ SwiGLU + down input transform)
// Column body of one wave for R row slots (R = the slot's row bucket, never above the call's MaxR).
template <int Bits, int R, int Depth>
__device__ void column_core(const Exl3& w, unsigned output, unsigned kt, unsigned splits, unsigned col, const _Float16* x,
                            std::uint32_t xstride, unsigned nrows, unsigned lane, float (*fin)[128], unsigned cl) {
    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
    d::stream_column<Bits, R, Depth>(w.trellis, output, col, kt, splits, 0, splits, x, xstride, nrows, lane, acc);
    if ((lane & 3U) == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < nrows) {
                fin[r][cl * 16 + lane / 4] = acc[r][0];
                fin[r][cl * 16 + lane / 4 + 8] = acc[r][1];
            }
}
template <int Bits, int MaxR, int Depth>
__device__ void column_rows(const Exl3& w, unsigned output, unsigned kt, unsigned splits, unsigned col, const _Float16* x,
                            std::uint32_t xstride, unsigned nrows, unsigned lane, float (*fin)[128], unsigned cl) {
    if constexpr (MaxR == 1) {
        column_core<Bits, 1, Depth>(w, output, kt, splits, col, x, xstride, nrows, lane, fin, cl);
    } else {
        if (nrows == 1) column_core<Bits, 1, Depth>(w, output, kt, splits, col, x, xstride, nrows, lane, fin, cl);
        else if (nrows == 2 || MaxR == 2) column_core<Bits, 2, Depth>(w, output, kt, splits, col, x, xstride, nrows, lane, fin, cl);
        else if constexpr (MaxR >= 4) {
            if (nrows <= 4 || MaxR == 4) column_core<Bits, 4, Depth>(w, output, kt, splits, col, x, xstride, nrows, lane, fin, cl);
            else column_core<Bits, MaxR, Depth>(w, output, kt, splits, col, x, xstride, nrows, lane, fin, cl);
        }
    }
}
template <int MaxR, int Depth>
__device__ void column_bits(const Exl3& w, unsigned output, unsigned kt, unsigned splits, unsigned col, const _Float16* x,
                            std::uint32_t xstride, unsigned nrows, unsigned lane, float (*fin)[128], unsigned cl) {
    if (w.bits == 3) column_rows<3, MaxR, Depth>(w, output, kt, splits, col, x, xstride, nrows, lane, fin, cl);
    else column_rows<5, MaxR, Depth>(w, output, kt, splits, col, x, xstride, nrows, lane, fin, cl);
}

__device__ __forceinline__ float swiglu(float g, float u) {  // tests/minimax_gate_common.hpp swiglu_kernel
    const float gg = g < kLimit ? g : kLimit;
    const float uu = u < -kLimit ? -kLimit : (u > kLimit ? kLimit : u);
    return gg * (1.0F / (1.0F + expf(-kAlpha * gg))) * (uu + 1.0F);
}

template <int MaxR, int Depth>
__global__ __launch_bounds__(512) void gate_up_kernel(Args a) {
    __shared__ float fin[2][MaxR][128];
    const Dispatch* dp = a.disp();
    const unsigned block = blockIdx.x, slot = blockIdx.y;
    if (slot != kSharedSlot && slot >= dp->count) return;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned m = wave >> 3U, cl = wave & 7U;
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(block * 8U + cl)));
    const ExpertSlices& E = slot_slices(a, dp, slot);
    const Exl3& w = m == 0 ? E.gate : E.up;
    const unsigned nrows = dp->nrows[slot], first = dp->first[slot];
    column_bits<MaxR, Depth>(w, kInter, kHidden / 16, kGateSplits, col, a.xt(first, m), 2 * kHidden, nrows, lane, fin[m], cl);
    __syncthreads();
    for (unsigned job = wave; job < 2 * nrows; job += 16) {  // output transforms (sum of splits already in fin)
        const unsigned mm = job & 1U, r = job >> 1U;
        const std::uint16_t* svh = mm == 0 ? E.gate.svh : E.up.svh;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = fin[mm][r][lane + 32U * static_cast<unsigned>(j)];
        d::hadamard128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = lane + 32U * static_cast<unsigned>(j);
            fin[mm][r][i] = v[j] * d::half_bits(svh[block * 128 + i]);
        }
    }
    __syncthreads();
    for (unsigned r = wave; r < nrows; r += 16) {  // SwiGLU, then the down input transform of these 128 channels
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = lane + 32U * static_cast<unsigned>(j);
            const float act = swiglu(fin[0][r][i], fin[1][r][i]);
            const float scaled = d::materialize(act * d::half_bits(E.down.suh[block * 128 + i]));
            v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        }
        d::hadamard128(v, lane);
        _Float16* out = a.at(first + r);
#pragma unroll
        for (int j = 0; j < 4; ++j) out[block * 128 + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
    }
}

// ---------------------------------------------------------------- down + fixed-order combine
template <int MaxR, int Depth>
__global__ __launch_bounds__(256) void down_kernel(Args a) {
    __shared__ float fin[MaxR][128];
    const Dispatch* dp = a.disp();
    const unsigned block = blockIdx.x, slot = blockIdx.y;
    if (slot != kSharedSlot && slot >= dp->count) return;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(block * 8U + wave)));
    const Exl3& w = slot_slices(a, dp, slot).down;
    const unsigned nrows = dp->nrows[slot], first = dp->first[slot];
    column_bits<MaxR, Depth>(w, kHidden, kInter / 16, kDownSplits, col, a.at(first), kInter, nrows, lane, fin, wave);
    __syncthreads();
    for (unsigned r = wave; r < nrows; r += 8) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = fin[r][lane + 32U * static_cast<unsigned>(j)];
        d::hadamard128(v, lane);
        float* y = a.yslot(first + r);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(j);
            y[i] = v[j] * d::half_bits(w.svh[i]);
        }
    }
    if (!last_arrival(a.counters() + 1 + block, dp->count + 1)) return;
    // Combine (G4 order): acc = 0; acc = fmaf(w, y_e, acc) over the row's experts in ascending id; acc + shared.
    const unsigned rows = dp->rows;
    for (unsigned q = threadIdx.x; q < rows * 128; q += 256) {
        const unsigned r = q / 128, i = block * 128 + (q % 128);
        float acc = 0.0F;
        for (unsigned k = 0; k < kTopE; ++k) acc = __builtin_fmaf(dp->row_weight[r][k], load_agent(a.yslot(dp->row_pair[r][k]) + i), acc);
        acc = acc + load_agent(a.yslot(dp->first[kSharedSlot] + r) + i);
        a.out[static_cast<std::size_t>(r) * kHidden + i] = acc;
    }
}

// ================================================================ v3 experts (STRIX_MMX_MOE_V=3)
// gate_up3: one CTA per (128 outputs, slot, matrix), 8 waves = 8 tile columns. The CTA computes its matrix's input
// transform of the slot's rows straight into LDS (in chunks of kV3Chunk(MaxR) channels), streams the weights with a
// register ring and reads activations from LDS; output transform; the second CTA of the (slot, block) pair to finish
// applies SwiGLU and the down input transform. down3: the slot's down inputs staged in LDS, same ring.
template <int MaxR> constexpr unsigned kV3Chunk = MaxR <= 2 ? kHidden : 12288U / MaxR;  // 6144, 6144, 3072, 1536

template <int Bits, int R, int MaxR, int Depth>
__device__ void gate_up3_body(const Args& a, const Dispatch* dp, const Exl3& w, unsigned slot, unsigned col,
                              unsigned nrows, _Float16* xs, float (*fin)[128], unsigned wave, unsigned lane) {
    constexpr unsigned CH = kV3Chunk<MaxR>, chunks = kHidden / CH, cblocks = CH / 128;
    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
    for (unsigned c = 0; c < chunks; ++c) {
        __syncthreads();
        for (unsigned job = wave; job < nrows * cblocks; job += 8) {
            const unsigned lr = job / cblocks, b = job % cblocks;
            const unsigned row = slot == kSharedSlot ? lr : dp->row[slot][lr];
            d::transform_block_at(a.y + static_cast<std::size_t>(row) * kHidden, w.suh, xs + static_cast<std::size_t>(lr) * CH,
                                  c * CH + b * 128U, b * 128U, lane);
        }
        __syncthreads();
        d::stream_column_lds<Bits, R, Depth>(w.trellis, kInter, col, kHidden / 16, kGateSplits, c * kGateSplits / chunks,
                                             (c + 1) * kGateSplits / chunks, xs, CH, c * CH / 16, nrows, lane, acc);
    }
    if ((lane & 3U) == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < nrows) {
                fin[r][wave * 16 + lane / 4] = acc[r][0];
                fin[r][wave * 16 + lane / 4 + 8] = acc[r][1];
            }
}

template <int MaxR, int Depth>
__global__ __launch_bounds__(256) void gate_up3_kernel(Args a) {
    __shared__ _Float16 xs[MaxR * kV3Chunk<MaxR>];
    __shared__ float fin[MaxR][128];
    const Dispatch* dp = a.disp();
    const unsigned block = blockIdx.x, slot = blockIdx.y, m = blockIdx.z;
    if (slot != kSharedSlot && slot >= dp->count) return;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(block * 8U + wave)));
    const ExpertSlices& E = slot_slices(a, dp, slot);
    const Exl3& w = m == 0 ? E.gate : E.up;
    const unsigned nrows = dp->nrows[slot], first = dp->first[slot];
    auto body = [&](auto bits_tag) {
        constexpr int Bits = decltype(bits_tag)::value;
        if constexpr (MaxR == 1) {
            gate_up3_body<Bits, 1, MaxR, Depth>(a, dp, w, slot, col, nrows, xs, fin, wave, lane);
        } else {
            if (nrows == 1) gate_up3_body<Bits, 1, MaxR, Depth>(a, dp, w, slot, col, nrows, xs, fin, wave, lane);
            else if (nrows == 2 || MaxR == 2) gate_up3_body<Bits, 2, MaxR, Depth>(a, dp, w, slot, col, nrows, xs, fin, wave, lane);
            else if constexpr (MaxR >= 4) {
                if (nrows <= 4 || MaxR == 4) gate_up3_body<Bits, 4, MaxR, Depth>(a, dp, w, slot, col, nrows, xs, fin, wave, lane);
                else gate_up3_body<Bits, MaxR, MaxR, Depth>(a, dp, w, slot, col, nrows, xs, fin, wave, lane);
            }
        }
    };
    if (w.bits == 3) body(std::integral_constant<int, 3>{});
    else body(std::integral_constant<int, 5>{});
    __syncthreads();
    for (unsigned r = wave; r < nrows; r += 8) {  // output transform of this matrix (sum of splits already in fin)
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = fin[r][lane + 32U * static_cast<unsigned>(j)];
        d::hadamard128(v, lane);
        float* out = a.gu(first + r, m);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = lane + 32U * static_cast<unsigned>(j);
            const float o = v[j] * d::half_bits(w.svh[block * 128 + i]);
            fin[r][i] = o;
            out[block * 128 + i] = o;
        }
    }
    if (!last_arrival(a.gu_counter(slot, block), 2)) return;
    for (unsigned r = wave; r < nrows; r += 8) {  // SwiGLU, then the down input transform of these 128 channels
        const float* other = a.gu(first + r, 1U - m) + block * 128;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = lane + 32U * static_cast<unsigned>(j);
            const float mine = fin[r][i], theirs = load_agent(other + i);
            const float act = m == 0 ? swiglu(mine, theirs) : swiglu(theirs, mine);
            const float scaled = d::materialize(act * d::half_bits(E.down.suh[block * 128 + i]));
            v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        }
        d::hadamard128(v, lane);
        _Float16* out = a.at(first + r);
#pragma unroll
        for (int j = 0; j < 4; ++j) out[block * 128 + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
    }
}

template <int MaxR, int Depth>
__global__ __launch_bounds__(256) void down3_kernel(Args a) {
    __shared__ _Float16 xs[MaxR * kInter];
    __shared__ float fin[MaxR][128];
    const Dispatch* dp = a.disp();
    const unsigned block = blockIdx.x, slot = blockIdx.y;
    if (slot != kSharedSlot && slot >= dp->count) return;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(block * 8U + wave)));
    const Exl3& w = slot_slices(a, dp, slot).down;
    const unsigned nrows = dp->nrows[slot], first = dp->first[slot];
    {
        const auto* src = reinterpret_cast<const std::uint32_t*>(a.at(first));
        auto* dst = reinterpret_cast<std::uint32_t*>(xs);
        for (unsigned q = threadIdx.x; q < nrows * kInter / 2; q += 256) dst[q] = src[q];
    }
    __syncthreads();
    auto body = [&](auto bits_tag, auto r_tag) {
        constexpr int Bits = decltype(bits_tag)::value, R = decltype(r_tag)::value;
        float acc[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
        d::stream_column_lds<Bits, R, Depth>(w.trellis, kHidden, col, kInter / 16, kDownSplits, 0, kDownSplits, xs, kInter, 0,
                                             nrows, lane, acc);
        if ((lane & 3U) == 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (static_cast<unsigned>(r) < nrows) {
                    fin[r][wave * 16 + lane / 4] = acc[r][0];
                    fin[r][wave * 16 + lane / 4 + 8] = acc[r][1];
                }
    };
    auto rows = [&](auto bits_tag) {
        if constexpr (MaxR == 1) {
            body(bits_tag, std::integral_constant<int, 1>{});
        } else {
            if (nrows == 1) body(bits_tag, std::integral_constant<int, 1>{});
            else if (nrows == 2 || MaxR == 2) body(bits_tag, std::integral_constant<int, 2>{});
            else if constexpr (MaxR >= 4) {
                if (nrows <= 4 || MaxR == 4) body(bits_tag, std::integral_constant<int, 4>{});
                else body(bits_tag, std::integral_constant<int, MaxR>{});
            }
        }
    };
    if (w.bits == 3) rows(std::integral_constant<int, 3>{});
    else rows(std::integral_constant<int, 5>{});
    __syncthreads();
    for (unsigned r = wave; r < nrows; r += 8) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = fin[r][lane + 32U * static_cast<unsigned>(j)];
        d::hadamard128(v, lane);
        float* y = a.yslot(first + r);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(j);
            y[i] = v[j] * d::half_bits(w.svh[i]);
        }
    }
    if (!last_arrival(a.counters() + 1 + block, dp->count + 1)) return;
    const unsigned rows_n = dp->rows;
    for (unsigned q = threadIdx.x; q < rows_n * 128; q += 256) {
        const unsigned r = q / 128, i = block * 128 + (q % 128);
        float acc = 0.0F;
        for (unsigned k = 0; k < kTopE; ++k) acc = __builtin_fmaf(dp->row_weight[r][k], load_agent(a.yslot(dp->row_pair[r][k]) + i), acc);
        acc = acc + load_agent(a.yslot(dp->first[kSharedSlot] + r) + i);
        a.out[static_cast<std::size_t>(r) * kHidden + i] = acc;
    }
}

// ================================================================ v4 experts (STRIX_MMX_MOE_V=4)
// The library's GEMV grid shape (CTA = 4 adjacent tile columns x one split, so concurrent CTAs sweep the same k-rows),
// with: the split's input transform computed by the CTA into LDS (gate/up: 3 H128 blocks per row, no prepare launch;
// down: the 384 transformed channels copied), a register ring of weight words inside the split, raw split partials to
// the workspace, and the last CTA of each 128-output block finishing it (gate/up: H128, svh, SwiGLU, down transform;
// down: H128, svh of every slot, then the fixed-order combine). Grid y = slot index, active slots first
// (0..count-1 routed, count = shared), so inactive CTAs sit at the end of the grid and exit at once.
constexpr unsigned kSplitCh = 384;  // 24 k-tiles: gate/up (6144 / 16 splits) and down (1536 / 4 splits)

__device__ __forceinline__ bool map_slot(const Dispatch* dp, unsigned y, unsigned& slot) {
    if (y < dp->count) slot = y;
    else if (y == dp->count) slot = kSharedSlot;
    else return false;
    return true;
}

// CTA-uniform, device-only decision from this step's router. Keep the low-union reader and its launch count.
// Engine-resolved default on; raw-library unset off. Only Depth-4 narrow K3 launches enable this reader.
template <int Depth>
__device__ __forceinline__ bool single_stream_for(const Args& a, const Dispatch* dp, unsigned slot) {
    return Depth == 4 && a.single_stream && route::moe_single_slot(dp->rows, dp->count, dp->single_count,
                                                        slot == kSharedSlot, dp->nrows[slot]);
}

// K3 singleton stream: each wave coalesces a complete 96-byte tile with one dword/lane, then shuffles the two
// needed words to their decoder lanes. Four k-tiles in flight, one row's two fdot2 chains, no MaxR=8 arrays.
// K3's third word cannot affect any of its eight states (the same identity used by LdsRingU).
__device__ __forceinline__ void singleton3_core(const Exl3& w, unsigned output, unsigned col, unsigned split,
                                               const _Float16* xs, unsigned lane, float* part) {
    constexpr unsigned depth = 4, tiles = kSplitCh / 16;
    const d::LaneGeom<3> geo(lane);
    const auto* base = (d::GlobalU32*)(d::uniform(w.trellis)) +
                       static_cast<std::size_t>(split * tiles) * (output / 16U) * 24U + col * 24U;
    const std::size_t stride = static_cast<std::size_t>(output / 16U) * 24U;
    std::uint32_t ring[depth] = {};
    auto load = [&](unsigned k) { return lane < 24U ? base[static_cast<std::size_t>(k) * stride + lane] : 0U; };
#pragma unroll
    for (unsigned j = 0; j < depth; ++j) ring[j] = load(j);
    const auto* xw = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
    float sums[1][2] = {{0.0F, 0.0F}};
    for (unsigned kb = 0; kb < tiles; kb += depth) {
#pragma unroll
        for (unsigned j = 0; j < depth; ++j) {
            const unsigned k = kb + j;
            const std::uint32_t word = ring[j];
            d::Slot<3, 1> cur;
            cur.a = __shfl(word, static_cast<int>(geo.oa / 4U), 32);
            cur.b = __shfl(word, static_cast<int>(geo.ob / 4U), 32);
            cur.c = 0;
            if (k + depth < tiles) ring[j] = load(k + depth);
            cur.act[0][0] = __builtin_bit_cast(d::Half2, xw[k * 8U]);
            cur.act[0][1] = __builtin_bit_cast(d::Half2, xw[k * 8U + 4U]);
            d::compute_slot<3, 1>(cur, geo, 1, sums);
        }
    }
    const float a0 = 0.0F + d::reduce4(sums[0][0]);
    const float a1 = 0.0F + d::reduce4(sums[0][1]);
    if ((lane & 3U) == 0) {
        part[col * 16U + lane / 4U] = a0;
        part[col * 16U + lane / 4U + 8U] = a1;
    }
}

template <int Bits, int R, int Depth>
__device__ __forceinline__ void split4_core(const Exl3& w, unsigned output, unsigned kt, unsigned splits, unsigned col, unsigned split,
                            const _Float16* xs, unsigned nrows, unsigned lane, float* part0, std::size_t part_stride) {
    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
#if STRIX_MM_GU4_SCALAR  // gu4 addressing ON (6a8d4ab6, exact); build switch -DSTRIX_MINIMAX_GU4_SCALAR=OFF = the former reader
    d::stream_column_lds_u<Bits, R, Depth>(w.trellis, output, col, kt, splits, split, split + 1, xs, kSplitCh,
                                         split * kSplitCh / 16, nrows, lane, acc);
#else
    d::stream_column_lds<Bits, R, Depth>(w.trellis, output, col, kt, splits, split, split + 1, xs, kSplitCh,
                                         split * kSplitCh / 16, nrows, lane, acc);
#endif
    if ((lane & 3U) == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < nrows) {
                float* pr = part0 + static_cast<std::size_t>(r) * part_stride + col * 16 + lane / 4;
                pr[0] = acc[r][0];
                pr[8] = acc[r][1];
            }
}
template <int MaxR, int Depth, int DepthBig = Depth>
__device__ __forceinline__ void split4_rows(const Exl3& w, unsigned output, unsigned kt, unsigned splits, unsigned col, unsigned split,
                            const _Float16* xs, unsigned nrows, unsigned lane, float* part0, std::size_t part_stride) {
    auto run = [&](auto bits_tag) {
        constexpr int Bits = decltype(bits_tag)::value;
        if constexpr (MaxR == 1) {
            split4_core<Bits, 1, Depth>(w, output, kt, splits, col, split, xs, nrows, lane, part0, part_stride);
        } else {
            if (nrows == 1) split4_core<Bits, 1, Depth>(w, output, kt, splits, col, split, xs, nrows, lane, part0, part_stride);
            else if (nrows == 2 || MaxR == 2) split4_core<Bits, 2, Depth>(w, output, kt, splits, col, split, xs, nrows, lane, part0, part_stride);
            else if constexpr (MaxR >= 4) {
                if (nrows <= 4 || MaxR == 4) split4_core<Bits, 4, Depth>(w, output, kt, splits, col, split, xs, nrows, lane, part0, part_stride);
                else split4_core<Bits, MaxR, DepthBig>(w, output, kt, splits, col, split, xs, nrows, lane, part0, part_stride);
            }
        }
    };
    if (w.bits == 3) run(std::integral_constant<int, 3>{});
    else run(std::integral_constant<int, 5>{});
}


// Wide MoE core (STRIX_MMX_MOE_WIDE=1; the linear v10 scheme): CTA = 4 waves x 16 adjacent columns x one split; wave
// owns 4 columns, streams their contiguous k-row runs (4 x Bits x 32 bytes) with 16-byte loads through a register
// ring and a wave-private LDS row (rowbuf, 8 x Bits uint4 per wave; no barrier), reuses one activation read per k for
// its 4 columns. Rows: static bucket R (rows past nrows read finite or stale LDS inputs; their results are never
// stored). Same chains as split4_core (bitwise).
constexpr unsigned kWideDepth = 4;
using WideV4 = std::uint32_t __attribute__((ext_vector_type(4)));
template <int Bits, int R>
__device__ __forceinline__ void split16_core(const Exl3& w, unsigned output, unsigned kt, unsigned splits, unsigned col0,
                                             unsigned split, const _Float16* xs, unsigned nrows, unsigned wave, unsigned lane,
                                             WideV4* rowbuf_all, float* part0, std::size_t part_stride) {
    constexpr unsigned runU = 8U * Bits, UPL = (runU + 31U) / 32U;
    WideV4* rowbuf = rowbuf_all + wave * runU;
    const unsigned k0 = kt * split / splits, k1 = kt * (split + 1) / splits, cols = output / 16U;
    const auto* gw = reinterpret_cast<const WideV4*>(d::uniform(w.trellis));
    const std::size_t kstride = static_cast<std::size_t>(cols) * 2U * Bits, cbase = static_cast<std::size_t>(col0) * 2U * Bits;
    WideV4 ring[kWideDepth][UPL] = {};
    auto load = [&](unsigned k, auto slot) __attribute__((always_inline)) {
        constexpr unsigned q = decltype(slot)::value;
#pragma unroll
        for (unsigned u = 0; u < UPL; ++u) {
            const unsigned e = lane + 32U * u;
            if (e < runU) ring[q][u] = gw[static_cast<std::size_t>(k) * kstride + cbase + e];
        }
    };
    const d::LaneGeom<Bits> geo(lane);
    float sums[4][R][2];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < R; ++r) sums[c][r][0] = sums[c][r][1] = 0.0F;
    const auto* rl = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(rowbuf));
    const auto* xw = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
    auto step = [&](unsigned k, auto slot) __attribute__((always_inline)) {
        constexpr unsigned q = decltype(slot)::value;
#pragma unroll
        for (unsigned u = 0; u < UPL; ++u) {
            const unsigned e = lane + 32U * u;
            if (e < runU) rowbuf[e] = ring[q][u];
        }
        if (k + kWideDepth < k1) load(k + kWideDepth, slot);
        d::Half2 act[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const auto* pr = xw + (static_cast<unsigned>(r) * kSplitCh) / 2U + (k - k0) * 8U;
            act[r][0] = __builtin_bit_cast(d::Half2, pr[0]);
            act[r][1] = __builtin_bit_cast(d::Half2, pr[4]);
        }
#pragma unroll
        for (unsigned c = 0; c < 4; ++c) {
            d::Slot<Bits, R> cur;
            const unsigned tw = c * Bits * 8U;
            cur.a = rl[tw + geo.oa / 4U];
            cur.b = rl[tw + geo.ob / 4U];
            cur.c = rl[tw + geo.oc / 4U];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                cur.act[r][0] = act[r][0];
                cur.act[r][1] = act[r][1];
            }
            d::compute_slot<Bits, R>(cur, geo, static_cast<unsigned>(R), sums[c]);
        }
    };
    using S0 = std::integral_constant<unsigned, 0>;
    using S1 = std::integral_constant<unsigned, 1>;
    using S2 = std::integral_constant<unsigned, 2>;
    using S3 = std::integral_constant<unsigned, 3>;
    if (k0 + 0 < k1) load(k0 + 0, S0{});
    if (k0 + 1 < k1) load(k0 + 1, S1{});
    if (k0 + 2 < k1) load(k0 + 2, S2{});
    if (k0 + 3 < k1) load(k0 + 3, S3{});
    for (unsigned k = k0; k < k1; k += 4) {
        step(k, S0{});
        if (k + 1 < k1) step(k + 1, S1{});
        if (k + 2 < k1) step(k + 2, S2{});
        if (k + 3 < k1) step(k + 3, S3{});
    }
#pragma unroll
    for (unsigned c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v0 = 0.0F + d::reduce4(sums[c][r][0]), v1 = 0.0F + d::reduce4(sums[c][r][1]);
            if ((lane & 3U) == 0 && static_cast<unsigned>(r) < nrows) {
                float* pr = part0 + static_cast<std::size_t>(r) * part_stride + (col0 + c) * 16 + lane / 4;
                pr[0] = v0;
                pr[8] = v1;
            }
        }
}
template <int MaxR>
__device__ void split16_rows(const Exl3& w, unsigned output, unsigned kt, unsigned splits, unsigned col0, unsigned split,
                             const _Float16* xs, unsigned nrows, unsigned wave, unsigned lane, WideV4* rowbuf, float* part0,
                             std::size_t part_stride) {
    auto run = [&](auto bits_tag) {
        constexpr int Bits = decltype(bits_tag)::value;
        if (nrows == 1 || MaxR == 1) split16_core<Bits, 1>(w, output, kt, splits, col0, split, xs, nrows, wave, lane, rowbuf, part0, part_stride);
        else if (nrows == 2 || MaxR == 2) split16_core<Bits, 2>(w, output, kt, splits, col0, split, xs, nrows, wave, lane, rowbuf, part0, part_stride);
        else if constexpr (MaxR >= 4) {
            if (nrows <= 4 || MaxR == 4) split16_core<Bits, 4>(w, output, kt, splits, col0, split, xs, nrows, wave, lane, rowbuf, part0, part_stride);
            else split16_core<Bits, MaxR>(w, output, kt, splits, col0, split, xs, nrows, wave, lane, rowbuf, part0, part_stride);
        }
    };
    if (w.bits == 3) run(std::integral_constant<int, 3>{});
    else run(std::integral_constant<int, 5>{});
}

// Pipeline producer: after the finisher's at() stores of `block`, release them at agent scope and record the epoch.
__device__ __forceinline__ void publish_gu_ready(const Args& a, unsigned slot, unsigned block, std::uint32_t epoch) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_store(a.gu_ready(slot, block), epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// One gate/up CTA of slot `slot` (rows: the slot's call rows, or 0..nrows-1 for the shared expert).
// ready != 0: the pipelined launch; each finished block is published with that epoch.
// Wide: cg indexes 16-column groups (2 output blocks per CTA) and the wide core runs (rowbuf: 4 x 8 x Bits uint4).
template <int MaxR, int Depth, int W, int DepthBig, bool Wide = false, bool Single = false>
__device__ void gate_up4_body(const Args& a, const Dispatch* dp, const ExpertSlices& E, unsigned slot, unsigned nrows,
                              unsigned first, unsigned cg, unsigned split, unsigned m, _Float16* xs, WideV4* rowbuf = nullptr,
                              bool single = false, std::uint32_t ready = 0) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const Exl3& w = m == 0 ? E.gate : E.up;
    for (unsigned job = wave; job < nrows * 3; job += W) {  // input transform of the split's 3 H128 blocks per row
        const unsigned lr = job / 3, b = job % 3;
        const unsigned row = slot == kSharedSlot ? lr : dp->row[slot][lr];
        d::transform_block_at(a.y + static_cast<std::size_t>(row) * kHidden, w.suh, xs + lr * kSplitCh, split * kSplitCh + b * 128U,
                              b * 128U, lane);
    }
    __syncthreads();
    auto finish = [&](unsigned block) {
    for (unsigned r = wave; r < nrows; r += W) {
        float g[4], u[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) g[j] = u[j] = 0.0F;
        for (unsigned sp = 0; sp < kGateSplits; ++sp) {
            const float* pg = a.part_gu(first + r, 0, sp) + block * 128 + lane;
            const float* pu = a.part_gu(first + r, 1, sp) + block * 128 + lane;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                g[j] += load_agent(pg + 32 * j);
                u[j] += load_agent(pu + 32 * j);
            }
        }
        d::hadamard128(g, lane);
        d::hadamard128(u, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(j);
            const float act = swiglu(g[j] * d::half_bits(E.gate.svh[i]), u[j] * d::half_bits(E.up.svh[i]));
            const float scaled = d::materialize(act * d::half_bits(E.down.suh[i]));
            v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        }
        d::hadamard128(v, lane);
        _Float16* out = a.at(first + r);
#pragma unroll
        for (int j = 0; j < 4; ++j) out[block * 128 + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
    }
    };
    if constexpr (Wide) {
        const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * 16U + wave * 4U)));
        split16_rows<MaxR>(w, kInter, kHidden / 16, kGateSplits, col0, split, xs, nrows, wave, lane, rowbuf,
                           a.part_gu(first, m, split), std::size_t{2} * 16 * kInter);
#pragma unroll
        for (unsigned h = 0; h < 2; ++h)
            if (last_arrival(a.gu4_counter(slot, cg * 2U + h), 2 * kGateSplits)) {
                finish(cg * 2U + h);
                if (ready != 0) publish_gu_ready(a, slot, cg * 2U + h, ready);
            }
    } else {
        const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * static_cast<unsigned>(W) + wave)));
        if constexpr (Single) {
            if (single && w.bits == 3) {
                if (a.single_reader_counts != nullptr && threadIdx.x == 0) atomicAdd(a.single_reader_counts, 1U);
                singleton3_core(w, kInter, col, split, xs, lane, a.part_gu(first, m, split));
            }
            else split4_rows<MaxR, Depth, DepthBig>(w, kInter, kHidden / 16, kGateSplits, col, split, xs, nrows, lane,
                                                     a.part_gu(first, m, split), std::size_t{2} * 16 * kInter);
        } else split4_rows<MaxR, Depth, DepthBig>(w, kInter, kHidden / 16, kGateSplits, col, split, xs, nrows, lane,
                                                 a.part_gu(first, m, split), std::size_t{2} * 16 * kInter);
        const unsigned block = cg * W / 8;
        if (last_arrival(a.gu4_counter(slot, block), (8 / W) * 2 * kGateSplits)) {
            finish(block);
            if (ready != 0) publish_gu_ready(a, slot, block, ready);
        }
    }
}

template <int MaxR, int Depth, int W, int DepthBig, bool SkipShared, bool Wide = false, bool Single = false>
__device__ __forceinline__ void gate_up4_cta(const Args& a, unsigned bxr, unsigned by, unsigned bz, _Float16* xs, WideV4* rowbuf,
                                             std::uint32_t ready = 0) {
    const Dispatch* dp = a.disp();
    constexpr unsigned groups = Wide ? kInter / 256U : kInter / (16U * W);
    // Former grid separated a slot's gate/up CTAs by an entire z plane spanning all slots. Pair the two
    // matrices per tile to reduce partial/counter lifetime; CTA counts and split arithmetic stay unchanged.
    const unsigned bx = a.interleave_gu ? bxr / 2U : bxr;
    const unsigned m = a.interleave_gu ? bxr % 2U : bz;
    unsigned slot;
    if (!map_slot(dp, by, slot)) return;
    if (SkipShared && slot == kSharedSlot) return;  // done by the router + shared launch
    gate_up4_body<MaxR, Depth, W, DepthBig, Wide, Single>(a, dp, slot_slices(a, dp, slot), slot, dp->nrows[slot], dp->first[slot],
                                                       bx % groups, bx / groups, m, xs, rowbuf,
                                                       Single && single_stream_for<Depth>(a, dp, slot), ready);
}
template <int MaxR, int Depth, int W, int DepthBig, bool SkipShared, bool Wide = false, bool Single = false>
__global__ __launch_bounds__(32 * W) void gate_up4_kernel(Args a) {
    __shared__ _Float16 xs[MaxR * kSplitCh];
    __shared__ WideV4 rowbuf[Wide ? 4 * 8 * 5 : 1];
    gate_up4_cta<MaxR, Depth, W, DepthBig, SkipShared, Wide, Single>(a, blockIdx.x, blockIdx.y, blockIdx.z, xs, rowbuf);
}

// Router + shared expert in one launch (STRIX_MMX_MOE_OVERLAP=1): CTAs 0..127 are router CTAs (one expert each, its
// weights streamed through LDS in 2048-float chunks, rows across the 4 waves; the same chains as route_row), the
// other CTAs run the shared expert's gate/up (rows 0..N-1, pairs 4N.., known before routing) so its weight stream
// fills the memory system while the router waits on latency. The routed gate/up launch then skips the shared slot.
constexpr unsigned kRsChunk = 2048;
template <int MaxR, int Depth, int DepthBig, bool Wide = false>
__global__ __launch_bounds__(128) void router_shared_kernel(Args a) {
    __shared__ float wsm[kRsChunk];
    __shared__ _Float16 xs[MaxR * kSplitCh];
    __shared__ WideV4 rowbuf[Wide ? 4 * 8 * 5 : 1];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    if (blockIdx.x < kExperts) {
        const unsigned e = blockIdx.x;
        float acc[2] = {0.0F, 0.0F};  // rows wave and wave + 4
        for (unsigned c0 = 0; c0 < kHidden; c0 += kRsChunk) {
            __syncthreads();
            const auto* src = reinterpret_cast<const float4*>(a.router_w + static_cast<std::size_t>(e) * kHidden + c0);
            auto* dst = reinterpret_cast<float4*>(wsm);
            for (unsigned q = threadIdx.x; q < kRsChunk / 4; q += 128) dst[q] = src[q];
            __syncthreads();
#pragma unroll
            for (unsigned h = 0; h < 2; ++h) {
                const unsigned r = wave + 4U * h;
                if (r >= a.rows) break;
                const float* yr = a.y + static_cast<std::size_t>(r) * kHidden + c0;
                float v = acc[h];
#pragma unroll 16
                for (unsigned i = 0; i < kRsChunk / 32; ++i) {
                    const unsigned j = lane + 32U * i;
                    v = __builtin_fmaf(yr[j], wsm[j], v);
                }
                acc[h] = v;
            }
        }
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            const unsigned r = wave + 4U * h;
            if (r >= a.rows) break;
            float v = acc[h];
            for (unsigned dd = 16; dd >= 1; dd >>= 1U) v = v + __shfl_xor(v, static_cast<int>(dd), 32);
            if (lane == 0) {
                const float sc = 1.0F / (1.0F + msa::exp_f(-v));
                a.scores()[r * kExperts + e] = sc;
                a.choice()[r * kExperts + e] = sc + a.bias[e];
            }
        }
        if (!last_arrival(a.counters(), kExperts)) return;
        route_tail<128>(a);
        return;
    }
    constexpr unsigned groups = Wide ? kInter / 256U : kInter / 64U;  // CTA = 16 (wide) or 4 columns per split
    const unsigned local = blockIdx.x - kExperts, per_m = groups * kGateSplits;
    const unsigned m = local / per_m, bx = local % per_m;
    gate_up4_body<MaxR, Depth, 4, DepthBig, Wide>(a, nullptr, a.shared, kSharedSlot, a.rows, kTopE * a.rows, bx % groups,
                                                  bx / groups, m, xs, rowbuf);
}

// Shared by the last-arrival path and the stream-ordered finish launch: identical four-split sum,
// H128/scales, ascending-expert fmaf and shared add. Each wave owns complete rows of one output tile.
template <int W>
__device__ __forceinline__ void down4_combine(const Args& a, const Dispatch* dp, unsigned block) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    // Finish every slot's block, then the G4 combine per row: acc = 0; fmaf over its experts in ascending id; + shared.
    const unsigned rows = dp->rows;
    auto finish = [&](unsigned pair, const std::uint16_t* svh, float (&v)[4]) {
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = 0.0F;
        for (unsigned sp = 0; sp < kDownSplits; ++sp) {
            const float* pp = a.part_dn(pair, sp) + block * 128 + lane;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] += load_agent(pp + 32 * j);
        }
        d::hadamard128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * d::half_bits(svh[block * 128 + lane + 32U * static_cast<unsigned>(j)]);
    };
    for (unsigned r = wave; r < rows; r += W) {
        float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (unsigned k = 0; k < kTopE; ++k) {
            float v[4];
            finish(dp->row_pair[r][k], a.experts[dp->row_expert[r][k]].down.svh, v);
            const float wk = dp->row_weight[r][k];
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] = __builtin_fmaf(wk, v[j], acc[j]);
        }
        float v[4];
        finish(dp->first[kSharedSlot] + r, a.shared.down.svh, v);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a.out[static_cast<std::size_t>(r) * kHidden + block * 128 + lane + 32U * static_cast<unsigned>(j)] = acc[j] + v[j];
    }
}

__global__ __launch_bounds__(128) void down4_combine_kernel(Args a) {
    down4_combine<4>(a, a.disp(), blockIdx.x);
    lean_publish(a.pub, a.pub.tiles);  // exactly one publisher per completed 128-output tile
}

template <int MaxR, int Depth, int W, int DepthBig, bool Wide = false, bool Single = false, bool PartialOnly = false>
__device__ __forceinline__ void down4_cta(const Args& a, unsigned bx, unsigned by, _Float16* xs, WideV4* rowbuf,
                                          std::uint32_t ready = 0) {
    const Dispatch* dp = a.disp();
    unsigned slot;
    if (!map_slot(dp, by, slot)) return;
    constexpr unsigned groups = Wide ? kHidden / 256U : kHidden / (16U * W);
    const unsigned cg = bx % groups, split = bx / groups;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const Exl3& w = slot_slices(a, dp, slot).down;
    const unsigned nrows = dp->nrows[slot], first = dp->first[slot];
    if (ready != 0 && slot != kSharedSlot) {  // the split's 384 inputs = gate/up blocks 3 split .. 3 split + 2 of this slot
        if (threadIdx.x == 0)
            for (unsigned b = 0; b < kSplitCh / 128; ++b) {
                const std::uint32_t* f = a.gu_ready(slot, split * (kSplitCh / 128) + b);
                for (unsigned spin = 0; __hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != ready; ++spin) {
                    if (spin == (1U << 26)) __builtin_trap();  // a producer never finished: fault instead of hanging
                    __builtin_amdgcn_s_sleep(1);
                }
            }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    for (unsigned q = threadIdx.x; q < nrows * (kSplitCh / 2); q += 32U * W) {  // the split's transformed channels
        const unsigned lr = q / (kSplitCh / 2), c = q % (kSplitCh / 2);
        reinterpret_cast<std::uint32_t*>(xs)[lr * (kSplitCh / 2) + c] =
            reinterpret_cast<const std::uint32_t*>(a.at(first + lr) + split * kSplitCh)[c];
    }
    __syncthreads();
    if constexpr (Wide) {
        const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * 16U + wave * 4U)));
        split16_rows<MaxR>(w, kHidden, kInter / 16, kDownSplits, col0, split, xs, nrows, wave, lane, rowbuf,
                           a.part_dn(first, split), std::size_t{4} * kHidden);
        if constexpr (!PartialOnly) {
#pragma unroll
            for (unsigned h = 0; h < 2; ++h)
                if (last_arrival(a.dn4_counter(cg * 2U + h), (dp->count + 1) * kDownSplits)) {
                    down4_combine<W>(a, dp, cg * 2U + h);
                    lean_publish(a.pub, a.pub.tiles);
                }
        }
    } else {
        const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * static_cast<unsigned>(W) + wave)));
        if constexpr (Single) {
            if (single_stream_for<Depth>(a, dp, slot) && w.bits == 3) {
                if (a.single_reader_counts != nullptr && threadIdx.x == 0) atomicAdd(a.single_reader_counts + 1, 1U);
                singleton3_core(w, kHidden, col, split, xs, lane, a.part_dn(first, split));
            }
            else split4_rows<MaxR, Depth, DepthBig>(w, kHidden, kInter / 16, kDownSplits, col, split, xs, nrows, lane,
                                                     a.part_dn(first, split), std::size_t{4} * kHidden);
        } else split4_rows<MaxR, Depth, DepthBig>(w, kHidden, kInter / 16, kDownSplits, col, split, xs, nrows, lane,
                                                 a.part_dn(first, split), std::size_t{4} * kHidden);
        if constexpr (!PartialOnly) {
            const unsigned block = cg * W / 8;
            if (last_arrival(a.dn4_counter(block), (dp->count + 1) * (8 / W) * kDownSplits)) {
                down4_combine<W>(a, dp, block);
                lean_publish(a.pub, a.pub.tiles);
            }
        }
    }
}
template <int MaxR, int Depth, int W, int DepthBig, bool Wide = false, bool Single = false, bool PartialOnly = false>
__global__ __launch_bounds__(32 * W) void down4_kernel(Args a) {
    __shared__ _Float16 xs[MaxR * kSplitCh];
    __shared__ WideV4 rowbuf[Wide ? 4 * 8 * 5 : 1];
    down4_cta<MaxR, Depth, W, DepthBig, Wide, Single, PartialOnly>(a, blockIdx.x, blockIdx.y, xs, rowbuf);
}

// Routed gate/up and down in one launch (STRIX_MMX_MOE_PIPE, overlap form only; the shared expert's gate/up ran in the
// router + shared launch). CTAs take a ticket in arrival order, so every CTA waits only on lower tickets, which are
// already resident: no dependence on hardware dispatch order. Tickets: [shared down if SharedFirst][routed gate/up, the
// former x-fastest grid order][routed down (+ shared down)]. A down CTA polls its split's three gate/up blocks of its
// slot (epoch of this route) instead of a kernel boundary; partials, counters, finish and combine are unchanged.
template <int MaxR, int Depth, int W, int DepthBig, bool Wide, bool Single, bool SharedFirst>
__global__ __launch_bounds__(32 * W) void moe_pipe4_kernel(Args a, unsigned gu_x, unsigned gu_z, unsigned slots, unsigned dn_x) {
    __shared__ _Float16 xs[MaxR * kSplitCh];
    __shared__ WideV4 rowbuf[Wide ? 4 * 8 * 5 : 1];
    __shared__ unsigned s_ticket;
    if (threadIdx.x == 0) {
        const unsigned t = atomicAdd(a.pipe_ticket(), 1U);
        if (t + 1U == gridDim.x) __hip_atomic_store(a.pipe_ticket(), 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_ticket = t;
    }
    __syncthreads();
    const std::uint32_t epoch = *a.pipe_epoch();
    unsigned t = s_ticket;
    if constexpr (SharedFirst) {
        if (t < dn_x) {
            down4_cta<MaxR, Depth, W, DepthBig, Wide, Single>(a, t, a.disp()->count, xs, rowbuf, epoch);
            return;
        }
        t -= dn_x;
    }
    const unsigned gu_n = gu_x * slots * gu_z;
    if (t < gu_n) {
        gate_up4_cta<MaxR, Depth, W, DepthBig, true, Wide, Single>(a, t % gu_x, (t / gu_x) % slots, t / (gu_x * slots), xs, rowbuf,
                                                                  epoch);
        return;
    }
    t -= gu_n;
    if (SharedFirst && t / dn_x == a.disp()->count) return;  // shared down already ran first
    down4_cta<MaxR, Depth, W, DepthBig, Wide, Single>(a, t % dn_x, t / dn_x, xs, rowbuf, epoch);
}

Args make_args(const MoeView& v) {
    if (v.rows < 1 || v.rows > kMaxRows) throw std::invalid_argument("minimax moe: rows must be 1..8");
    if (v.y == nullptr || v.router_w == nullptr || v.bias == nullptr || v.experts == nullptr || v.out == nullptr ||
        v.ws.base == nullptr)
        throw std::invalid_argument("minimax moe: null pointer in view");
    const ExpertSlices& s = v.shared;
    if (s.gate.bits != s.up.bits || (s.gate.bits != 3 && s.gate.bits != 5) || (s.down.bits != 3 && s.down.bits != 5) ||
        s.gate.input != kHidden || s.gate.output != kInter || s.down.input != kInter || s.down.output != kHidden)
        throw std::invalid_argument("minimax moe: shared expert must be 6144 -> 1536 -> 6144 at K3 or K5");
    if (v.h != nullptr && (v.part0 == nullptr || v.part1 == nullptr || v.norm_w == nullptr))
        throw std::invalid_argument("minimax moe: fused post-exchange needs part0, part1 and norm_w");
    if ((v.controlled_ids == nullptr) != (v.controlled_weights == nullptr))
        throw std::invalid_argument("minimax moe: controlled ids and weights must be supplied together");
    Args a{v.rows, v.h, v.part0, v.part1, v.norm_w, v.y, v.router_w, v.bias, v.experts, v.shared, v.out, v.ids, v.weights,
           static_cast<unsigned char*>(v.ws.base)};
    a.pub = v.pub;
    a.controlled_ids = v.controlled_ids;
    a.controlled_weights = v.controlled_weights;
    a.single_reader_counts = v.single_reader_counts;
    const char* interleave = std::getenv("STRIX_MMX_MOE_GU_INTERLEAVE");
    if (interleave != nullptr && std::string(interleave) != "0" && std::string(interleave) != "1")
        throw std::invalid_argument("STRIX_MMX_MOE_GU_INTERLEAVE must be 0 or 1");
    a.interleave_gu = route::moe_interleave(a.rows, interleave != nullptr && std::string(interleave) == "1");  // default 1 via mm_defaults
    const char* down = std::getenv("STRIX_MMX_MOE_DOWN_SPLIT");
    if (down != nullptr && std::string(down) != "0" && std::string(down) != "1" && std::string(down) != "2")
        throw std::invalid_argument("STRIX_MMX_MOE_DOWN_SPLIT must be 0, 1 or 2");
    // default 1 via mm_defaults (measured regime only, see route::moe_down_split); read per call
    a.down_split = down == nullptr ? 0U : static_cast<unsigned>(down[0] - '0');
    a.pub.tiles = kHidden / 128U;  // down4 combine finishers: one per 128-output tile (both wide and narrow forms)
    return a;
}

void launched(const char* what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax moe ") + what + ": " + hipGetErrorString(e));
}

}  // namespace

std::size_t moe_workspace_bytes() { return kMoeBytes; }
std::size_t moe_pipe_offset() { return kOffPipe; }
const Dispatch* moe_dispatch(const MoeWorkspace& ws) { return static_cast<const Dispatch*>(ws.base); }

// STRIX_MMX_ROUTER: 1 = 16 CTAs x 8 experts (also the fused post-exchange form), 2 = one CTA per expert (inputs y).
int router_version() {
    static const int v = [] {
        const char* e = std::getenv("STRIX_MMX_ROUTER");
        const int x = e != nullptr ? std::atoi(e) : 2;  // measurement: v2 0.67 (N1) / 0.80 (N8) of route_row; v1 1.08 / 1.66
        if (x != 1 && x != 2) throw std::invalid_argument("STRIX_MMX_ROUTER must be 1 or 2");
        return x;
    }();
    return v;
}

void enqueue_router(const MoeView& v, hipStream_t stream) {
    const Args a = make_args(v);
    if (a.h != nullptr) hipLaunchKernelGGL(router_kernel<true>, dim3(kRouterCtas), dim3(256), 0, stream, a);
    else if (router_version() == 2) hipLaunchKernelGGL(router2_kernel, dim3(kExperts), dim3(256), 0, stream, a);
    else hipLaunchKernelGGL(router_kernel<false>, dim3(kRouterCtas), dim3(256), 0, stream, a);
    launched("router");
}

namespace {
// STRIX_MMX_MOE_DEPTH=D (0, 4, 8) applies D up to the row bucket STRIX_MMX_MOE_DEPTH_MAXR (default 2; the ring of
// activations spills above it) and 0 beyond.
int moe_depth(unsigned rows) {
    static const int depth = [] {
        const char* e = std::getenv("STRIX_MMX_MOE_DEPTH");
        const int v = e != nullptr ? std::atoi(e) : kDefaultDepth;
        if (v != 0 && v != 4 && v != 8) throw std::invalid_argument("STRIX_MMX_MOE_DEPTH must be 0, 4 or 8");
        return v;
    }();
    static const unsigned maxr = [] {
        const char* e = std::getenv("STRIX_MMX_MOE_DEPTH_MAXR");
        return e != nullptr ? static_cast<unsigned>(std::atoi(e)) : 2U;
    }();
    return rows <= maxr ? depth : 0;
}
template <int MaxR, int Depth> void launch_experts(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    hipLaunchKernelGGL((gate_up_kernel<MaxR, Depth>), dim3(kInterBlocks, kMaxSlots + 1), dim3(512), 0, stream, a);
    launched("gate_up");
    if (marks != nullptr) (void)hipEventRecord(marks[1], stream);
    hipLaunchKernelGGL((down_kernel<MaxR, Depth>), dim3(kOutBlocks, kMaxSlots + 1), dim3(256), 0, stream, a);
    launched("down");
    if (a.down_done != nullptr) (void)hipEventRecord(a.down_done, stream);
}
template <int MaxR, int Depth> void launch_experts3(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    hipLaunchKernelGGL((gate_up3_kernel<MaxR, Depth>), dim3(kInterBlocks, kMaxSlots + 1, 2), dim3(256), 0, stream, a);
    launched("gate_up3");
    if (marks != nullptr) (void)hipEventRecord(marks[1], stream);
    hipLaunchKernelGGL((down3_kernel<MaxR, Depth>), dim3(kOutBlocks, kMaxSlots + 1), dim3(256), 0, stream, a);
    launched("down3");
    if (a.down_done != nullptr) (void)hipEventRecord(a.down_done, stream);
}
template <int Depth> void launch_experts3_rows(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    if (a.rows == 1) launch_experts3<1, Depth>(a, stream, marks);
    else if (a.rows == 2) launch_experts3<2, Depth>(a, stream, marks);
    else if (a.rows <= 4) launch_experts3<4, Depth>(a, stream, marks);
    else launch_experts3<8, Depth>(a, stream, marks);
}
// STRIX_MMX_MOE_W: waves per v4 CTA, 4 (default: one 64-output column group per split) or 8 (a 128-output block).
int moe_waves() {
    static const int w = [] {
        const char* e = std::getenv("STRIX_MMX_MOE_W");
        const int x = e != nullptr ? std::atoi(e) : 4;
        if (x != 4 && x != 8) throw std::invalid_argument("STRIX_MMX_MOE_W must be 4 or 8");
        return x;
    }();
    return w;
}
// STRIX_MMX_MOE_PIPE (read per call; 0 / unset = off): in the decode overlap form, the routed gate/up and down run as one
// launch whose down CTAs poll per-block gate/up completion instead of the kernel boundary (1 = routed down after all
// gate/up tickets, 2 = the shared expert's down first, its inputs being ready from the router + shared launch). Same
// partials, counters, finish and combine: bitwise. Calls of 2-4 rows; takes precedence over the stream-ordered down split.
// Measurement (two-node decode, in-process A/B, logits bitwise): N2 58.9-59.3 -> 57.3-57.7 ms/step, N4 81.8-82.6 ->
// 80.6-80.8; N1 44.13-44.17 -> 44.24-44.27 ms/token (hence excluded).
unsigned moe_pipe() {
    const char* value = std::getenv("STRIX_MMX_MOE_PIPE");
    if (value == nullptr || *value == 0 || std::string(value) == "0") return 0;
    if (std::string(value) == "1") return 1;
    if (std::string(value) == "2") return 2;
    throw std::invalid_argument("STRIX_MMX_MOE_PIPE must be 0, 1 or 2");
}
// STRIX_MMX_MOE_WIDE: 1 = wide expert kernels at every N, 0 = 4-column CTAs at every N, unset (2) = the split by N:
// measurement (both forms bitwise; WIDE 0-1-0-1-0-1-0 processes, inline v4 vs wide, MoE layer us, process medians) wide is
// faster at N2-4 (0.943 / 0.911 / 0.907) and slower at N1 (1.021), N5 (1.040), N6 (1.019), N8 (1.050); N7 even (0.998).
// Confirmed in measurement (WIDE 0-2-0-2-0-2-0 processes, predeclared bracket rule, noise bound 0.020 from the same-kernel N):
// wide/4-column brackets N2 0.941-0.954, N3 0.909-0.926, N4 0.906-0.917.
int moe_wide() {
    static const int w = [] {
        const char* e = std::getenv("STRIX_MMX_MOE_WIDE");
        const int x = e != nullptr ? std::atoi(e) : 2;
        if (x < 0 || x > 2) throw std::invalid_argument("STRIX_MMX_MOE_WIDE must be 0, 1 or 2");
        return x;
    }();
    return w;
}
bool wide_for(unsigned rows) { return route::wide_for(rows, moe_wide()); }
// STRIX_MMX_MOE_TIGHT_GRID (default 1, 0 = off; read per call for in-process A/B): the v4 routed gate/up and down grids
// get count + 1 <= 4 N + 1 slot rows instead of kMaxSlots + 1. map_slot() makes every CTA with y > count return at
// once, so the rows dropped never did work; the kept CTAs have the same block indices, work and bits. At N = 1 this
// removes 28 of 33 rows (about 10,000 empty CTAs per launch, two launches per MoE layer).
unsigned moe_grid_slots(unsigned rows) {
    const char* value = std::getenv("STRIX_MMX_MOE_TIGHT_GRID");
    if (value != nullptr && std::string(value) == "0") return kMaxSlots + 1;
    if (value != nullptr && std::string(value) != "1") throw std::invalid_argument("STRIX_MMX_MOE_TIGHT_GRID must be 0 or 1");
    return std::min(kMaxSlots, kTopE * std::max(rows, 1U)) + 1;
}
// Default 1 via mm_defaults (STRIX_MM_BASELINE=1 or =0 switches it off). Read on every call for in-process A/B.
bool moe_single() {
    const char* value = std::getenv("STRIX_MMX_MOE_SINGLE");
    if (value == nullptr || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument("STRIX_MMX_MOE_SINGLE must be 0 or 1");
}
template <int MaxR, int Depth, int W, int DepthBig, bool SkipShared, bool Wide, bool Single = false>
void launch_experts4_grid(const Args& input, hipStream_t stream, hipEvent_t* marks) {
    static_assert(!Single || (MaxR == 8 && Depth == 4 && W == 4 && !Wide));
    Args a = input;
    a.single_stream = Single;
    constexpr unsigned gu_groups = Wide ? kInter / 256U : kInter / (16U * W);
    constexpr unsigned dn_groups = Wide ? kHidden / 256U : kHidden / (16U * W);
    const unsigned slots = moe_grid_slots(a.rows);
    // Calls of 2-4 rows only: merged, the 8-row bodies' live ranges take ~150 VGPRs (85 / 81 apart) and halve occupancy;
    // capped at 96 they spill (measured +15-19 % at N5-8 either way); at N1 two launches measured 0.1 ms/token faster.
    if constexpr (MaxR <= 4) {
        const unsigned pipe = SkipShared && a.rows >= 2 ? moe_pipe() : 0U;
        if (pipe != 0) {  // replaces the stream-ordered down split too (measured faster at N4)
            const unsigned gu_x = gu_groups * kGateSplits * (a.interleave_gu ? 2U : 1U), gu_z = a.interleave_gu ? 1U : 2U;
            const unsigned dn_x = dn_groups * kDownSplits, grid = gu_x * slots * gu_z + dn_x * slots;
            if (pipe == 2)
                hipLaunchKernelGGL((moe_pipe4_kernel<MaxR, Depth, W, DepthBig, Wide, Single, true>), dim3(grid), dim3(32 * W), 0, stream, a,
                                   gu_x, gu_z, slots, dn_x);
            else
                hipLaunchKernelGGL((moe_pipe4_kernel<MaxR, Depth, W, DepthBig, Wide, Single, false>), dim3(grid), dim3(32 * W), 0, stream,
                                   a, gu_x, gu_z, slots, dn_x);
            launched("routed gate/up + down pipeline");
            if (marks != nullptr) (void)hipEventRecord(marks[1], stream);  // parts: gate/up includes the down
            if (a.down_done != nullptr) (void)hipEventRecord(a.down_done, stream);
            return;
        }
    }
    const dim3 gu_grid(gu_groups * kGateSplits * (a.interleave_gu ? 2U : 1U), slots,
                       a.interleave_gu ? 1U : 2U);
    hipLaunchKernelGGL((gate_up4_kernel<MaxR, Depth, W, DepthBig, SkipShared, Wide, Single>),
                       gu_grid, dim3(32 * W), 0, stream, a);
    launched("gate_up4");
    if (marks != nullptr) (void)hipEventRecord(marks[1], stream);
    if (route::moe_down_split(a.down_split, a.rows, Wide, W, Depth)) {
        // Same producers and scratch addresses, without any U-wide arrival counters or finish epilogue.
        hipLaunchKernelGGL((down4_kernel<MaxR, Depth, W, DepthBig, Wide, Single, true>),
                           dim3(dn_groups * kDownSplits, slots), dim3(32 * W), 0, stream, a);
        launched("down4 partials");
        if (a.down_done != nullptr) (void)hipEventRecord(a.down_done, stream);
        // Stream ordering makes all partial stores visible; no spin, fences or producer counters needed.
        hipLaunchKernelGGL(down4_combine_kernel, dim3(kOutBlocks), dim3(128), 0, stream, a);
        launched("down4 combine");
    } else {
        hipLaunchKernelGGL((down4_kernel<MaxR, Depth, W, DepthBig, Wide, Single>),
                           dim3(dn_groups * kDownSplits, slots), dim3(32 * W), 0, stream, a);
        launched("down4");
        if (a.down_done != nullptr) (void)hipEventRecord(a.down_done, stream);
    }
}

template <int MaxR, int Depth, int W, int DepthBig, bool SkipShared = false>
void launch_experts4w(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    [[maybe_unused]] const bool single = moe_single();  // validate even when this shape uses the unchanged reader
    if constexpr (W == 4) {
        if (wide_for(a.rows)) {
            launch_experts4_grid<MaxR, Depth, W, DepthBig, SkipShared, true>(a, stream, marks);
            return;
        }
        if constexpr (MaxR == 8 && Depth == 4) {
            if (single && a.rows == 8) {
                launch_experts4_grid<MaxR, Depth, W, DepthBig, SkipShared, false, true>(a, stream, marks);
                return;
            }
        }
    }
    launch_experts4_grid<MaxR, Depth, W, DepthBig, SkipShared, false>(a, stream, marks);
}
// STRIX_MMX_MOE_DEPTH8R: ring depth of the 8-row bucket's bodies (the shared expert at N >= 5): 0 = the common depth,
// 2 = a shallower ring (lower register pressure for 8 rows; same bits).
int moe_depth8r() {
    static const int d = [] {
        const char* e = std::getenv("STRIX_MMX_MOE_DEPTH8R");
        const int x = e != nullptr ? std::atoi(e) : 0;
        if (x != 0 && x != 2) throw std::invalid_argument("STRIX_MMX_MOE_DEPTH8R must be 0 or 2");
        return x;
    }();
    return d;
}
int moe_overlap() {
    static const int o = [] {
        const char* e = std::getenv("STRIX_MMX_MOE_OVERLAP");
        return e != nullptr ? std::atoi(e) : 1;  // measurement: exact, B/A 0.48-0.56 vs 0.51-0.60 without, every N
    }();
    return o;
}
// Router + shared gate/up launch, then the routed gate/up (shared slot skipped) and down.
template <int MaxR, int Depth> void launch_overlap4(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    constexpr int DB = MaxR == 8 ? Depth : Depth;
    if (wide_for(a.rows))
        hipLaunchKernelGGL((router_shared_kernel<MaxR, Depth, DB, true>), dim3(kExperts + 2 * (kInter / 256) * kGateSplits), dim3(128),
                           0, stream, a);
    else
        hipLaunchKernelGGL((router_shared_kernel<MaxR, Depth, DB>), dim3(kExperts + 2 * (kInter / 64) * kGateSplits), dim3(128), 0,
                           stream, a);
    launched("router + shared");
    if (marks != nullptr) (void)hipEventRecord(marks[0], stream);
    launch_experts4w<MaxR, Depth, 4, DB, true>(a, stream, marks);
}
template <int MaxR, int Depth> void launch_experts4(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    if constexpr (MaxR == 8) {
        if (moe_depth8r() == 2) {
            if (moe_waves() == 8) launch_experts4w<MaxR, Depth, 8, 2, false>(a, stream, marks);
            else launch_experts4w<MaxR, Depth, 4, 2, false>(a, stream, marks);
            return;
        }
    }
    if (moe_waves() == 8) launch_experts4w<MaxR, Depth, 8, Depth, false>(a, stream, marks);
    else launch_experts4w<MaxR, Depth, 4, Depth, false>(a, stream, marks);
}
template <int Depth> void launch_experts4_rows(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    if (a.rows == 1) launch_experts4<1, Depth>(a, stream, marks);
    else if (a.rows == 2) launch_experts4<2, Depth>(a, stream, marks);
    else if (a.rows <= 4) launch_experts4<4, Depth>(a, stream, marks);
    else launch_experts4<8, Depth>(a, stream, marks);
}
int moe_version() {
    static const int v = [] {
        const char* e = std::getenv("STRIX_MMX_MOE_V");
        const int x = e != nullptr ? std::atoi(e) : kDefaultVersion;
        if (x != 1 && x != 3 && x != 4) throw std::invalid_argument("STRIX_MMX_MOE_V must be 1, 3 or 4");
        return x;
    }();
    return v;
}
int moe_depth3() {
    static const int depth = [] {
        const char* e = std::getenv("STRIX_MMX_MOE_DEPTH3");
        const int x = e != nullptr ? std::atoi(e) : 4;  // measurement: depth 4 fastest for MoE v4 at N = 1..8
        if (x != 4 && x != 8 && x != 12) throw std::invalid_argument("STRIX_MMX_MOE_DEPTH3 must be 4, 8 or 12");
        return x;
    }();
    return depth;
}
template <int Depth> void launch_experts_rows(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    if (a.rows == 1) launch_experts<1, Depth>(a, stream, marks);
    else if (a.rows == 2) launch_experts<2, Depth>(a, stream, marks);
    else if (a.rows <= 4) launch_experts<4, Depth>(a, stream, marks);
    else launch_experts<8, Depth>(a, stream, marks);
}
}  // namespace

namespace {
void check_single_resource(const void* kernel, const char* name, int depth, int big, bool skip) {
    hipFuncAttributes attrs{};
    const hipError_t status = hipFuncGetAttributes(&attrs, kernel);
    if (status != hipSuccess) throw std::runtime_error(std::string("singleton resource query: ") + hipGetErrorString(status));
    std::printf("singleton resource kernel=%s Depth=%d DepthBig=%d SkipShared=%d VGPR=%d scratch=%zu LDS=%zu\n",
                name, depth, big, skip ? 1 : 0, attrs.numRegs, attrs.localSizeBytes, attrs.sharedSizeBytes);
    if (attrs.localSizeBytes != 0) throw std::runtime_error("singleton kernel has scratch");
}
template <int Depth, int Big, bool Skip> void check_single_instances() {
    check_single_resource(reinterpret_cast<const void*>(gate_up4_kernel<8, Depth, 4, Big, Skip, false, true>),
                          "gate_up4", Depth, Big, Skip);
    if constexpr (!Skip)
        check_single_resource(reinterpret_cast<const void*>(down4_kernel<8, Depth, 4, Big, false, true>),
                              "down4", Depth, Big, false);
}
template <int Depth> void check_single_depth() {
    check_single_instances<Depth, Depth, true>();
    check_single_instances<Depth, Depth, false>();
    check_single_instances<Depth, 2, false>();
}
}  // namespace
void check_moe_single_resources() {
    check_single_depth<4>();
}

void enqueue_experts(const MoeView& v, hipStream_t stream, hipEvent_t* marks, hipEvent_t down_done) {
    Args a = make_args(v);
    a.down_done = down_done;
    if (moe_version() == 4) {
        if (marks != nullptr) (void)hipEventRecord(marks[0], stream);
        switch (moe_depth3()) {
            case 4: launch_experts4_rows<4>(a, stream, marks); break;
            case 12: launch_experts4_rows<12>(a, stream, marks); break;
            default: launch_experts4_rows<8>(a, stream, marks); break;
        }
        return;
    }
    if (moe_version() == 3) {
        if (marks != nullptr) (void)hipEventRecord(marks[0], stream);
        switch (moe_depth3()) {
            case 4: launch_experts3_rows<4>(a, stream, marks); break;
            case 12: launch_experts3_rows<12>(a, stream, marks); break;
            default: launch_experts3_rows<8>(a, stream, marks); break;
        }
        return;
    }
    hipLaunchKernelGGL(prepare_kernel, dim3(kMaxSlots + 1, 2), dim3(256), 0, stream, a);
    launched("prepare");
    if (marks != nullptr) (void)hipEventRecord(marks[0], stream);
    switch (moe_depth(a.rows)) {
        case 4: launch_experts_rows<4>(a, stream, marks); break;
        case 8: launch_experts_rows<8>(a, stream, marks); break;
        default: launch_experts_rows<0>(a, stream, marks); break;
    }
}

template <int Depth> void launch_overlap4_rows(const Args& a, hipStream_t stream, hipEvent_t* marks) {
    if (a.rows == 1) launch_overlap4<1, Depth>(a, stream, marks);
    else if (a.rows == 2) launch_overlap4<2, Depth>(a, stream, marks);
    else if (a.rows <= 4) launch_overlap4<4, Depth>(a, stream, marks);
    else launch_overlap4<8, Depth>(a, stream, marks);
}

void enqueue_moe_profiled(const MoeView& v, hipStream_t stream, hipEvent_t* marks, hipEvent_t down_done) {
    if (v.pub.flag != nullptr && !(moe_overlap() == 1 && v.h == nullptr && moe_version() == 4 && moe_waves() == 4))
        throw std::logic_error("minimax moe: the publish epilogue needs the decode overlap form (v4, 4 waves)");
    if (moe_overlap() == 1 && v.h == nullptr && moe_version() == 4 && moe_waves() == 4) {
        Args a = make_args(v);
        a.down_done = down_done;
        switch (moe_depth3()) {
            case 8: launch_overlap4_rows<8>(a, stream, marks); break;
            case 12: launch_overlap4_rows<12>(a, stream, marks); break;
            default: launch_overlap4_rows<4>(a, stream, marks); break;
        }
        return;
    }
    enqueue_router(v, stream);
    enqueue_experts(v, stream, marks, down_done);
}

void enqueue_moe(const MoeView& v, hipStream_t stream) { enqueue_moe_profiled(v, stream, nullptr); }

// ================================================================ router for many rows
namespace {
constexpr unsigned kRowsTile = 32;
__global__ __launch_bounds__(256) void router_rows_kernel(unsigned rows, const float* y, const float* router_w,
                                                          const float* bias, float* scratch) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned e = blockIdx.x * kRouterWaves + wave, r0 = blockIdx.y * kRowsTile;
    const unsigned n = rows - r0 < kRowsTile ? rows - r0 : kRowsTile;
    const float* w = router_w + static_cast<std::size_t>(e) * kHidden;
    const float* yr = y + static_cast<std::size_t>(r0) * kHidden;
    float acc[kRowsTile];
#pragma unroll
    for (unsigned r = 0; r < kRowsTile; ++r) acc[r] = 0.0F;
    for (unsigned j = lane; j < kHidden; j += 32) {
        const float wj = w[j];
#pragma unroll
        for (unsigned r = 0; r < kRowsTile; ++r)
            if (r < n) acc[r] = __builtin_fmaf(yr[static_cast<std::size_t>(r) * kHidden + j], wj, acc[r]);
    }
#pragma unroll
    for (unsigned r = 0; r < kRowsTile; ++r) {
        float v = acc[r];
        for (unsigned dd = 16; dd >= 1; dd >>= 1U) v = v + __shfl_xor(v, static_cast<int>(dd), 32);
        if (lane == 0 && r < n) {
            const float s = 1.0F / (1.0F + msa::exp_f(-v));
            float* out = scratch + (static_cast<std::size_t>(r0 + r) * kExperts + e) * 2;
            out[0] = s;
            out[1] = s + bias[e];
        }
    }
}
__global__ __launch_bounds__(64) void topk_rows_kernel(unsigned rows, const float* scratch, std::uint32_t* ids, float* weights) {
    const unsigned t = blockIdx.x * 64 + threadIdx.x;
    if (t >= rows) return;
    const float* sc = scratch + static_cast<std::size_t>(t) * kExperts * 2;
    std::uint32_t best[kTopE];
    for (unsigned k = 0; k < kTopE; ++k) {
        std::uint32_t pick = kExperts;
        for (std::uint32_t x = 0; x < kExperts; ++x) {
            bool taken = false;
            for (unsigned q = 0; q < k; ++q) taken |= best[q] == x;
            if (taken) continue;
            if (pick == kExperts || sc[2 * x + 1] > sc[2 * pick + 1]) pick = x;
        }
        best[k] = pick;
    }
    float sum = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k) sum = sum + sc[2 * best[k]];
    for (unsigned k = 0; k < kTopE; ++k) {
        ids[static_cast<std::size_t>(t) * kTopE + k] = best[k];
        weights[static_cast<std::size_t>(t) * kTopE + k] = (sc[2 * best[k]] / (sum + kSumEps)) * kRouteScale;
    }
}
}  // namespace

void enqueue_router_rows(unsigned rows, const float* y, const float* router_w, const float* bias, float* scratch,
                         std::uint32_t* ids, float* weights, hipStream_t stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(router_rows_kernel, dim3(kRouterCtas, (rows + kRowsTile - 1) / kRowsTile), dim3(256), 0, stream, rows,
                       y, router_w, bias, scratch);
    launched("router rows");
    hipLaunchKernelGGL(topk_rows_kernel, dim3((rows + 63) / 64), dim3(64), 0, stream, rows, scratch, ids, weights);
    launched("topk rows");
}

// ================================================================ dense linears
namespace {

struct LinJob {
    Exl3 w;
    const float* x;
    float* y;
    std::uint32_t x_stride, y_stride;
    std::uint32_t splits, cta_begin, block_begin;  // CTAs of this job in the core grid; counter base
    std::size_t xt_off, part_off;                  // halves / floats into the workspace
    std::uint32_t prep_begin;                      // v6: first (row, 128-block) transform job of this job
};
constexpr unsigned kLinCounters = 4096;
struct LinArgs {
    LinJob job[kMaxJobs];
    unsigned count, rows;
    unsigned xs_stride;   // v4: LDS row stride (halves) = the largest split of the call's jobs
    unsigned xs_stride5;  // v5: LDS row stride (halves) = the largest 128-aligned span covering one split
    unsigned prep_jobs;   // v6: rows x input / 128 summed over the jobs
    unsigned prep_ctas;   // v6: leading CTAs of the launch that run the input transforms
    std::uint64_t* trace; // profiling (set_linear_trace): per CTA 4 realtime stamps + HW_ID1, v4/v7 only
    unsigned char* ws;
    std::size_t counter_off;
    PublishTarget pub{};  // v4 core: publish the outputs (STRIX_MM_XB_PUBLISH)
};

__global__ __launch_bounds__(256) void linear_prepare_kernel(LinArgs a) {
    const LinJob& j = a.job[blockIdx.y];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned blocks = j.w.input / 128U;
    auto* xt = reinterpret_cast<_Float16*>(a.ws) + j.xt_off;
    for (unsigned q = blockIdx.x * 8U + wave; q < a.rows * blocks; q += gridDim.x * 8U) {
        const unsigned r = q / blocks, b = q % blocks;
        d::transform_block(j.x + static_cast<std::size_t>(r) * j.x_stride, j.w.suh,
                           xt + static_cast<std::size_t>(r) * j.w.input, b * 128U, lane);
    }
}

template <int Bits, int R, int Depth>
__device__ void linear_split(const LinJob& j, unsigned col, unsigned split, const _Float16* xt, unsigned rows,
                             unsigned crows, unsigned lane, float* part) {
    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
    d::stream_column<Bits, R, Depth>(j.w.trellis, j.w.output, col, j.w.input / 16, j.splits, split, split + 1, xt, j.w.input,
                              crows, lane, acc);
    if ((lane & 3U) == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < rows) {
                float* p = part + (static_cast<std::size_t>(split) * rows + static_cast<unsigned>(r)) * j.w.output + col * 16 + lane / 4;
                p[0] = acc[r][0];
                p[8] = acc[r][1];
            }
}


// One CTA = 4 tile columns (64 outputs) x one split of one job (the library's GEMV grid); the last of the 2 x splits
// CTAs of a 128-output block runs the output transform of that block for every row.
// Static: the core computes all R rows without row predicates (v1: only when the call has exactly R rows; v8: always,
// rows past the call's count read finite stale workspace inputs and their results are never stored: each row's chain
// is independent of the others, so the stored rows keep their bits).
template <int R, int Depth, bool Static>
__global__ __launch_bounds__(128) void linear_core_kernel(LinArgs a) {
    const unsigned rows = a.rows, crows = Static ? static_cast<unsigned>(R) : a.rows;
    unsigned ji = 0;
    while (ji + 1 < a.count && blockIdx.x >= a.job[ji + 1].cta_begin) ++ji;
    const LinJob& j = a.job[ji];
    const unsigned local = blockIdx.x - j.cta_begin, groups = j.w.output / 64U;
    const unsigned split = local / groups, group = local % groups;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(group * 4U + wave)));
    const auto* xt = reinterpret_cast<const _Float16*>(a.ws) + j.xt_off;
    float* part = reinterpret_cast<float*>(a.ws) + j.part_off;
    switch (j.w.bits) {
        case 3: linear_split<3, R, Depth>(j, col, split, xt, rows, crows, lane, part); break;
        case 4: linear_split<4, R, Depth>(j, col, split, xt, rows, crows, lane, part); break;
        case 5: linear_split<5, R, Depth>(j, col, split, xt, rows, crows, lane, part); break;
        case 6: linear_split<6, R, Depth>(j, col, split, xt, rows, crows, lane, part); break;
        default: linear_split<8, R, Depth>(j, col, split, xt, rows, crows, lane, part); break;
    }
    const unsigned block = group / 2U;
    auto* counters = reinterpret_cast<std::uint32_t*>(a.ws + a.counter_off);
    if (!last_arrival(counters + j.block_begin + block, 2U * j.splits)) return;
    for (unsigned r = wave; r < rows; r += 4) {  // output_transform: values = 0; += partial over splits ascending
        float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (unsigned s = 0; s < j.splits; ++s)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                v[q] += load_agent(part + (static_cast<std::size_t>(s) * rows + r) * j.w.output + block * 128 + lane + 32U * static_cast<unsigned>(q));
        d::hadamard128(v, lane);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(q);
            j.y[static_cast<std::size_t>(r) * j.y_stride + i] = v[q] * d::half_bits(j.w.svh[i]);
        }
    }
}

// ---------------------------------------------------------------- linear v3 (STRIX_MMX_LINEAR_V=3)
// One CTA = one 128-output block of one job, 8 waves = its 8 tile columns, every split in ascending order inside the
// wave (acc = ((0 + p_0) + p_1) + ...). The CTA writes the input transform of its rows into LDS in chunks that end on
// split boundaries (<= 12288 / MaxR halves per row), streams the weights with a register ring and finishes the block
// (H128, svh) in place: no partial buffers, no counters, one launch.
template <int MaxR> constexpr unsigned kLin3Budget = MaxR == 1 ? 8192U : 12288U / MaxR;

template <int Bits, int R, int MaxR, int Depth>
__device__ void linear3_body(const LinJob& j, unsigned col, unsigned rows, _Float16* xs, float (*fin)[128], unsigned wave,
                             unsigned lane) {
    const unsigned input = j.w.input, kt = input / 16, splits = j.splits;
    // Chunk = whole splits that fit the per-row LDS budget.
    const unsigned split_ch = input / splits;  // channels per split (all MiniMax shapes divide evenly)
    const unsigned per_chunk = std::max(1U, kLin3Budget<MaxR> / split_ch);
    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
    for (unsigned s0 = 0; s0 < splits; s0 += per_chunk) {
        const unsigned s1 = std::min(splits, s0 + per_chunk);
        const unsigned c0 = s0 * split_ch, ch = (s1 - s0) * split_ch, cblocks = (ch + 127) / 128;
        const unsigned b0 = c0 / 128;  // transform blocks are 128-aligned; a chunk may start inside one
        const unsigned lo = c0 % 128;
        __syncthreads();
        for (unsigned q = wave; q < rows * (cblocks + (lo != 0 ? 1U : 0U)); q += 8) {
            const unsigned nb = cblocks + (lo != 0 ? 1U : 0U);
            const unsigned r = q / nb, b = q % nb;
            float v[4];
            const float* x = j.x + static_cast<std::size_t>(r) * j.x_stride;
            const unsigned begin = (b0 + b) * 128;
            if (begin >= input) continue;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const unsigned c = begin + lane + 32U * static_cast<unsigned>(jj);
                const float scaled = d::materialize(x[c] * d::half_bits(j.w.suh[c]));
                v[jj] = static_cast<float>(static_cast<_Float16>(scaled));
            }
            d::hadamard128(v, lane);
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const unsigned c = begin + lane + 32U * static_cast<unsigned>(jj);
                if (c >= c0 && c < c0 + ch) xs[static_cast<std::size_t>(r) * kLin3Budget<MaxR> + (c - c0)] = static_cast<_Float16>(v[jj]);
            }
        }
        __syncthreads();
        d::stream_column_lds<Bits, R, Depth>(j.w.trellis, j.w.output, col, kt, splits, s0, s1, xs, kLin3Budget<MaxR>,
                                             c0 / 16, rows, lane, acc);
    }
    if ((lane & 3U) == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < rows) {
                fin[r][wave * 16 + lane / 4] = acc[r][0];
                fin[r][wave * 16 + lane / 4 + 8] = acc[r][1];
            }
}

template <int MaxR, int Depth>
__global__ __launch_bounds__(256) void linear3_kernel(LinArgs a) {
    __shared__ _Float16 xs[MaxR * kLin3Budget<MaxR>];
    __shared__ float fin[MaxR][128];
    unsigned ji = 0;
    while (ji + 1 < a.count && blockIdx.x >= a.job[ji + 1].block_begin) ++ji;
    const LinJob& j = a.job[ji];
    const unsigned block = blockIdx.x - j.block_begin;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(block * 8U + wave)));
    const unsigned rows = a.rows;
    switch (j.w.bits) {
        case 3: linear3_body<3, MaxR, MaxR, Depth>(j, col, rows, xs, fin, wave, lane); break;
        case 4: linear3_body<4, MaxR, MaxR, Depth>(j, col, rows, xs, fin, wave, lane); break;
        case 5: linear3_body<5, MaxR, MaxR, Depth>(j, col, rows, xs, fin, wave, lane); break;
        case 6: linear3_body<6, MaxR, MaxR, Depth>(j, col, rows, xs, fin, wave, lane); break;
        default: linear3_body<8, MaxR, MaxR, Depth>(j, col, rows, xs, fin, wave, lane); break;
    }
    __syncthreads();
    for (unsigned r = wave; r < rows; r += 8) {
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = fin[r][lane + 32U * static_cast<unsigned>(q)];
        d::hadamard128(v, lane);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(q);
            j.y[static_cast<std::size_t>(r) * j.y_stride + i] = v[q] * d::half_bits(j.w.svh[i]);
        }
    }
}

// Linear v4 (STRIX_MMX_LINEAR_V=4): the v1 grid with the split's prepared activations copied to LDS (one copy per CTA)
// and a ring of weight words inside the split (STRIX_MMX_MOE_DEPTH3, default 8).
// LDS per CTA = R x (largest split of the call) halves, sized at launch (q/k/v: 384, o: 1024, head/dense: 1536).
// Static: the call has exactly R rows, so the row predicates fold away (the library has the same specialization).
// Prime (v7, STRIX_MMX_LINEAR_V=7): the weight ring is primed before the split's inputs are copied to LDS, so the two
// memory latencies of a CTA's start overlap instead of adding up (same chains and bits).
template <bool Trace> __device__ __forceinline__ void lin_stamp(const LinArgs& a, unsigned slot) {
    if constexpr (!Trace) return;
    if (threadIdx.x == 0) {
        a.trace[static_cast<std::size_t>(blockIdx.x) * 5 + slot] = __builtin_readsteadycounter();
        if (slot == 0) a.trace[static_cast<std::size_t>(blockIdx.x) * 5 + 4] = static_cast<unsigned>(__builtin_amdgcn_s_getreg((31 << 11) | 23));
    }
}
// Trace: profiling instantiation (set_linear_trace); the normal one carries no stamp code.
template <int R, int Depth, bool Static, bool Prime = false, bool Trace = false>
__global__ __launch_bounds__(128) void linear_core4_kernel(LinArgs a) {
    extern __shared__ __attribute__((aligned(16))) unsigned char lin4_smem[];  // R x xs_stride halves (dynamic)
    auto* xs = reinterpret_cast<_Float16*>(lin4_smem);
    lin_stamp<Trace>(a, 0);
    unsigned ji = 0;
    while (ji + 1 < a.count && blockIdx.x >= a.job[ji + 1].cta_begin) ++ji;
    const LinJob& j = a.job[ji];
    const unsigned local = blockIdx.x - j.cta_begin, groups = j.w.output / 64U;
    const unsigned split = local / groups, group = local % groups;
    const unsigned rows = Static ? static_cast<unsigned>(R) : a.rows;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned kt = j.w.input / 16, k0 = kt * split / j.splits, k1 = kt * (split + 1) / j.splits;
    const unsigned ch = (k1 - k0) * 16;
    const auto* xt = reinterpret_cast<const _Float16*>(a.ws) + j.xt_off;
    auto copy_inputs = [&] {
        for (unsigned q = threadIdx.x; q < rows * (ch / 2); q += 128) {
            const unsigned r = q / (ch / 2), c = q % (ch / 2);
            reinterpret_cast<std::uint32_t*>(xs)[r * (a.xs_stride / 2) + c] =
                reinterpret_cast<const std::uint32_t*>(xt + static_cast<std::size_t>(r) * j.w.input + k0 * 16)[c];
        }
    };
    if constexpr (!Prime) {
        copy_inputs();
        __syncthreads();
        lin_stamp<Trace>(a, 1);
    }
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(group * 4U + wave)));
    float* part = reinterpret_cast<float*>(a.ws) + j.part_off;
    auto run = [&](auto bits_tag) {
        constexpr int Bits = decltype(bits_tag)::value;
        float acc[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
        if constexpr (Prime) {
            d::LdsRing<Bits, Depth> ring(j.w.trellis, j.w.output, col, kt, j.splits, split, split + 1, lane);
            copy_inputs();
            d::lds_barrier();
            lin_stamp<Trace>(a, 1);
            ring.template run<R>(xs, a.xs_stride, k0, rows, lane, acc);
        } else {
            d::stream_column_lds<Bits, R, Depth>(j.w.trellis, j.w.output, col, kt, j.splits, split, split + 1, xs, a.xs_stride,
                                                 k0, rows, lane, acc);
        }
        if ((lane & 3U) == 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (static_cast<unsigned>(r) < rows) {
                    float* pr = part + (static_cast<std::size_t>(split) * rows + static_cast<unsigned>(r)) * j.w.output + col * 16 + lane / 4;
                    pr[0] = acc[r][0];
                    pr[8] = acc[r][1];
                }
    };
    switch (j.w.bits) {
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
        default: run(std::integral_constant<int, 8>{}); break;
    }
    lin_stamp<Trace>(a, 2);
    const unsigned block = group / 2U;
    auto* counters = reinterpret_cast<std::uint32_t*>(a.ws + a.counter_off);
    if (!last_arrival(counters + j.block_begin + block, 2U * j.splits)) {
        lin_stamp<Trace>(a, 3);
        return;
    }
    lin_stamp<Trace>(a, 3);  // the finishing CTA's last stamp is its finish start; its end is the kernel's end
    for (unsigned r = wave; r < rows; r += 4) {
        float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (unsigned sp = 0; sp < j.splits; ++sp)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                v[q] += load_agent(part + (static_cast<std::size_t>(sp) * rows + r) * j.w.output + block * 128 + lane + 32U * static_cast<unsigned>(q));
        d::hadamard128(v, lane);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(q);
            j.y[static_cast<std::size_t>(r) * j.y_stride + i] = v[q] * d::half_bits(j.w.svh[i]);
        }
    }
    lean_publish(a.pub, a.pub.tiles);  // the finisher of each 128-output tile counts once (STRIX_MM_XB_PUBLISH)
}

// Output transform of one 128-output block by the last-arriving CTA (shared by v4/v5/v6; v4 has its own copy).
__device__ __forceinline__ void linear_finish(const LinJob& j, const float* part, unsigned block, unsigned rows, unsigned wave,
                                              unsigned lane) {
    for (unsigned r = wave; r < rows; r += 4) {  // output_transform: values = 0; += partial over splits ascending
        float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (unsigned sp = 0; sp < j.splits; ++sp)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                v[q] += load_agent(part + (static_cast<std::size_t>(sp) * rows + r) * j.w.output + block * 128 + lane + 32U * static_cast<unsigned>(q));
        d::hadamard128(v, lane);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(q);
            j.y[static_cast<std::size_t>(r) * j.y_stride + i] = v[q] * d::half_bits(j.w.svh[i]);
        }
    }
}
template <int R>
__device__ __forceinline__ void linear_store_partial(const LinJob& j, float* part, unsigned split, unsigned col, unsigned rows,
                                                     unsigned lane, const float (&acc)[R][2]) {
    if ((lane & 3U) == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < rows) {
                float* pr = part + (static_cast<std::size_t>(split) * rows + static_cast<unsigned>(r)) * j.w.output + col * 16 + lane / 4;
                pr[0] = acc[r][0];
                pr[8] = acc[r][1];
            }
}

// Linear v5 (STRIX_MMX_LINEAR_V=5): one launch, no prepare. Each CTA primes its weight ring, then computes the input
// transform of the 128-channel blocks that cover its split for every row straight into LDS (the MoE v4 scheme), then
// streams its split. Same chains as v4 (the transform of a block does not depend on who computes it).
template <int R, int Depth, bool Static>
__global__ __launch_bounds__(128) void linear_core5_kernel(LinArgs a) {
    extern __shared__ __attribute__((aligned(16))) unsigned char lin5_smem[];  // R x xs_stride5 halves (dynamic)
    auto* xs = reinterpret_cast<_Float16*>(lin5_smem);
    unsigned ji = 0;
    while (ji + 1 < a.count && blockIdx.x >= a.job[ji + 1].cta_begin) ++ji;
    const LinJob& j = a.job[ji];
    const unsigned local = blockIdx.x - j.cta_begin, groups = j.w.output / 64U;
    const unsigned split = local / groups, group = local % groups;
    const unsigned rows = Static ? static_cast<unsigned>(R) : a.rows;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned kt = j.w.input / 16, k0 = kt * split / j.splits, k1 = kt * (split + 1) / j.splits;
    const unsigned b0 = k0 / 8U, b1 = (k1 + 7U) / 8U, nb = b1 - b0;
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(group * 4U + wave)));
    float* part = reinterpret_cast<float*>(a.ws) + j.part_off;
    auto run = [&](auto bits_tag) {
        constexpr int Bits = decltype(bits_tag)::value;
        d::LdsRing<Bits, Depth> ring(j.w.trellis, j.w.output, col, kt, j.splits, split, split + 1, lane);
        for (unsigned q = wave; q < rows * nb; q += 4) {
            const unsigned r = q / nb, b = q % nb;
            d::transform_block_at(j.x + static_cast<std::size_t>(r) * j.x_stride, j.w.suh, xs + r * a.xs_stride5, (b0 + b) * 128U,
                                  b * 128U, lane);
        }
        d::lds_barrier();
        float acc[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
        ring.template run<R>(xs, a.xs_stride5, b0 * 8U, rows, lane, acc);
        linear_store_partial<R>(j, part, split, col, rows, lane, acc);
    };
    switch (j.w.bits) {
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
        default: run(std::integral_constant<int, 8>{}); break;
    }
    const unsigned block = group / 2U;
    auto* counters = reinterpret_cast<std::uint32_t*>(a.ws + a.counter_off);
    if (!last_arrival(counters + j.block_begin + block, 2U * j.splits)) return;
    linear_finish(j, part, block, rows, wave, lane);
}

// Linear v6 (STRIX_MMX_LINEAR_V=6): one launch. CTAs [0, prep_ctas) run the input transforms into the workspace (as the
// v4 prepare launch) and count themselves done; every core CTA primes its weight ring, waits for that count, copies its
// split's transformed inputs to LDS (agent-scope loads: the workspace was written by other CTAs of this launch) and
// streams. The dispatcher starts CTAs in index order, so the transform CTAs are resident before any core CTA waits.
// Counters kLinCounters - 2 (transforms done) and - 1 (core CTAs past the wait; the last one resets both).
__device__ __forceinline__ std::uint32_t load_agent_u32(const std::uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
template <int R, int Depth, bool Static>
__global__ __launch_bounds__(128) void linear_core6_kernel(LinArgs a) {
    extern __shared__ __attribute__((aligned(16))) unsigned char lin6_smem[];  // R x xs_stride halves (dynamic)
    auto* xs = reinterpret_cast<_Float16*>(lin6_smem);
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    auto* counters = reinterpret_cast<std::uint32_t*>(a.ws + a.counter_off);
    std::uint32_t* done = counters + (kLinCounters - 2);
    std::uint32_t* passed = counters + (kLinCounters - 1);
    const unsigned rows = Static ? static_cast<unsigned>(R) : a.rows;
    if (blockIdx.x < a.prep_ctas) {
        for (unsigned q = blockIdx.x * 4U + wave; q < a.prep_jobs; q += a.prep_ctas * 4U) {
            unsigned ji = 0;
            while (ji + 1 < a.count && q >= a.job[ji + 1].prep_begin) ++ji;
            const LinJob& j = a.job[ji];
            const unsigned blocks = j.w.input / 128U, t = q - j.prep_begin, r = t / blocks, b = t % blocks;
            auto* xt = reinterpret_cast<_Float16*>(a.ws) + j.xt_off;
            d::transform_block(j.x + static_cast<std::size_t>(r) * j.x_stride, j.w.suh, xt + static_cast<std::size_t>(r) * j.w.input,
                               b * 128U, lane);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(done, 1U);
        }
        return;
    }
    const unsigned cta = blockIdx.x - a.prep_ctas;
    unsigned ji = 0;
    while (ji + 1 < a.count && cta >= a.job[ji + 1].cta_begin) ++ji;
    const LinJob& j = a.job[ji];
    const unsigned local = cta - j.cta_begin, groups = j.w.output / 64U;
    const unsigned split = local / groups, group = local % groups;
    const unsigned kt = j.w.input / 16, k0 = kt * split / j.splits, k1 = kt * (split + 1) / j.splits;
    const unsigned ch = (k1 - k0) * 16;
    const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(group * 4U + wave)));
    float* part = reinterpret_cast<float*>(a.ws) + j.part_off;
    const unsigned ncore = gridDim.x - a.prep_ctas;
    auto run = [&](auto bits_tag) {
        constexpr int Bits = decltype(bits_tag)::value;
        d::LdsRing<Bits, Depth> ring(j.w.trellis, j.w.output, col, kt, j.splits, split, split + 1, lane);
        if (threadIdx.x == 0) {
            while (load_agent_u32(done) < a.prep_ctas) __builtin_amdgcn_s_sleep(2);
            __threadfence();
            if (atomicAdd(passed, 1U) + 1U == ncore) {
                *passed = 0;
                *done = 0;
            }
        }
        __syncthreads();
        const auto* xt = reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const _Float16*>(a.ws) + j.xt_off);
        for (unsigned q = threadIdx.x; q < rows * (ch / 2); q += 128) {
            const unsigned r = q / (ch / 2), c = q % (ch / 2);
            reinterpret_cast<std::uint32_t*>(xs)[r * (a.xs_stride / 2) + c] = load_agent_u32(xt + (static_cast<std::size_t>(r) * j.w.input + k0 * 16) / 2 + c);
        }
        d::lds_barrier();
        float acc[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
        ring.template run<R>(xs, a.xs_stride, k0, rows, lane, acc);
        linear_store_partial<R>(j, part, split, col, rows, lane, acc);
    };
    switch (j.w.bits) {
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
        default: run(std::integral_constant<int, 8>{}); break;
    }
    const unsigned block = group / 2U;
    if (!last_arrival(counters + j.block_begin + block, 2U * j.splits)) return;
    linear_finish(j, part, block, rows, wave, lane);
}

// Linear v9 (STRIX_MMX_LINEAR_V=9): the v4 grid (CTA = 4 adjacent tile columns x one split) with the weight stream
// read as whole k-row runs: the CTA's 4 tiles of one k are contiguous in the file layout (4 x Bits x 32 bytes), so all
// 128 threads load chunks of kLin9Rows k-rows with 16-byte loads (instead of three 4-byte loads per lane and tile; measurement:
// this request pattern alone lifts the stream from 172-205 to 228-233 GB/s), together with the chunk's activations,
// into a double-buffered LDS ring; each wave reads its tile's three words and the activation pairs from LDS. The
// decode, the per-column fdot2 chains, the reduction and the finish are those of v4 (bitwise). Rows are computed in
// the static bucket R (padded rows read finite stale inputs and are never stored, as v8).
constexpr unsigned kLin9Rows = 8, kLin9Ahead = 3;
template <int R> constexpr std::size_t lin9_lds_bytes(unsigned max_bits) {
    return 2 * (std::size_t{kLin9Rows} * 8 * max_bits * 16 + std::size_t{kLin9Rows} * R * 2 * 16);
}
// Mode (diagnostic, STRIX_MMX_V9_MODE; results NOT the linear's): 0 normal; 1 no compute (stream + LDS stage only);
// 2 no global weight loads (LDS stage of whatever the registers hold + compute).
template <int R, int Mode = 0>
__global__ __launch_bounds__(128) void linear_core9_kernel(LinArgs a) {
    extern __shared__ __attribute__((aligned(16))) unsigned char lin9_smem[];
    unsigned ji = 0;
    while (ji + 1 < a.count && blockIdx.x >= a.job[ji + 1].cta_begin) ++ji;
    const LinJob& j = a.job[ji];
    const unsigned cols = j.w.output / 16U, groups = j.w.output / 64U;
    const unsigned local = blockIdx.x - j.cta_begin, split = local / groups, group = local % groups;
    const unsigned rows = a.rows;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U, tid = threadIdx.x;
    const unsigned kt = j.w.input / 16, k0 = kt * split / j.splits, k1 = kt * (split + 1) / j.splits;
    const unsigned nc = (k1 - k0 + kLin9Rows - 1) / kLin9Rows;
    const auto* xt = reinterpret_cast<const _Float16*>(a.ws) + j.xt_off;
    float* part = reinterpret_cast<float*>(a.ws) + j.part_off;
    const unsigned col = group * 4U + wave;
    auto run = [&](auto bits_tag) __attribute__((always_inline)) {
        constexpr unsigned Bits = static_cast<unsigned>(decltype(bits_tag)::value);
        constexpr unsigned rowU = 8U * Bits, wU = kLin9Rows * rowU, WPT = (wU + 127U) / 128U, aU = kLin9Rows * R * 2U;
        using V4 = std::uint32_t __attribute__((ext_vector_type(4)));
        auto* wbuf = reinterpret_cast<V4*>(lin9_smem);  // [2][wU], then [2][aU]
        auto* abuf = wbuf + 2 * wU;
        const auto* gw = reinterpret_cast<const V4*>(d::uniform(j.w.trellis));
        const std::size_t gbase = static_cast<std::size_t>(group) * 4U * 2U * Bits;  // uint4 offset of the group in a k-row
        const std::size_t kstride = static_cast<std::size_t>(cols) * 2U * Bits;      // uint4 per k-row of the matrix
        // Register stage of kLin9Ahead chunks in flight (slot = chunk % kLin9Ahead); a store waits only for its own
        // chunk's loads (the compiler's vmcnt counts the later ones), so the ring never drains between chunks.
        V4 wr[kLin9Ahead][WPT] = {};
        V4 ar[kLin9Ahead];
        auto load = [&](unsigned c, auto slot) __attribute__((always_inline)) {
            constexpr unsigned q = decltype(slot)::value;
            const unsigned kb = k0 + c * kLin9Rows, rc = std::min(kLin9Rows, k1 - kb);
#pragma unroll
            for (unsigned i = 0; i < WPT; ++i) {
                const unsigned e = tid + 128U * i;
                if (Mode != 2 && e < rc * rowU) wr[q][i] = gw[(kb + e / rowU) * kstride + gbase + e % rowU];
            }
            if (tid < aU) {
                const unsigned r = tid / (kLin9Rows * 2U), kk = (tid / 2U) % kLin9Rows, half = tid % 2U;
                if (kk < rc)
                    ar[q] = *reinterpret_cast<const V4*>(xt + static_cast<std::size_t>(r) * j.w.input + (kb + kk) * 16U + half * 8U);
            }
        };
        auto store = [&](unsigned b, unsigned c, auto slot) __attribute__((always_inline)) {
            constexpr unsigned q = decltype(slot)::value;
            const unsigned kb = k0 + c * kLin9Rows, rc = std::min(kLin9Rows, k1 - kb);
#pragma unroll
            for (unsigned i = 0; i < WPT; ++i) {
                const unsigned e = tid + 128U * i;
                if (e < rc * rowU) wbuf[b * wU + e] = wr[q][i];
            }
            if (tid < aU) abuf[b * aU + tid] = ar[q];
        };
        const d::LaneGeom<static_cast<int>(Bits)> geo(lane);
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        using S0 = std::integral_constant<unsigned, 0>;
        using S1 = std::integral_constant<unsigned, 1>;
        using S2 = std::integral_constant<unsigned, 2>;
        static_assert(kLin9Ahead == 3);
        if (nc > 0) load(0, S0{});
        if (nc > 1) load(1, S1{});
        auto step = [&](unsigned c, auto slot, auto next) __attribute__((always_inline)) {  // slot = c % 3, next = (c + 2) % 3
            if (c + 2 < nc) load(c + 2, next);
            store(c & 1U, c, slot);  // all waves finished chunk c - 2 (same buffer) before the previous barrier
            d::lds_barrier();
            const unsigned b = c & 1U, rc = std::min(kLin9Rows, k1 - (k0 + c * kLin9Rows));
            const auto* wl = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(wbuf + b * wU));
            const auto* al = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(abuf + b * aU)) + (lane & 3U);
#pragma unroll
            for (unsigned kk = 0; kk < kLin9Rows; ++kk) {
                if (kk < rc) {
                    d::Slot<static_cast<int>(Bits), R> cur;
                    const unsigned tw = (kk * 4U + wave) * Bits * 8U;  // tile start in u32 words
                    cur.a = wl[tw + geo.oa / 4U];
                    cur.b = wl[tw + geo.ob / 4U];
                    cur.c = wl[tw + geo.oc / 4U];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const auto* pr = al + (static_cast<unsigned>(r) * kLin9Rows + kk) * 8U;
                        cur.act[r][0] = __builtin_bit_cast(d::Half2, pr[0]);
                        cur.act[r][1] = __builtin_bit_cast(d::Half2, pr[4]);
                    }
                    if constexpr (Mode == 1) {
                        sums[0][0] += __builtin_bit_cast(float, cur.a ^ cur.b ^ cur.c);  // keep the LDS reads
                    } else {
                        d::compute_slot<static_cast<int>(Bits), R>(cur, geo, static_cast<unsigned>(R), sums);
                    }
                }
            }
        };
        for (unsigned c = 0; c < nc; c += 3) {
            step(c, S0{}, S2{});
            if (c + 1 < nc) step(c + 1, S1{}, S0{});
            if (c + 2 < nc) step(c + 2, S2{}, S1{});
        }
        float acc[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r][0] = 0.0F + d::reduce4(sums[r][0]);
            acc[r][1] = 0.0F + d::reduce4(sums[r][1]);
        }
        linear_store_partial<R>(j, part, split, col, rows, lane, acc);
    };
    switch (j.w.bits) {
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
        default: run(std::integral_constant<int, 8>{}); break;
    }
    const unsigned block = group / 2U;
    auto* counters = reinterpret_cast<std::uint32_t*>(a.ws + a.counter_off);
    if (!last_arrival(counters + j.block_begin + block, 2U * j.splits)) return;
    linear_finish(j, part, block, rows, wave, lane);
}

// Linear v10 (STRIX_MMX_LINEAR_V=10): no CTA barriers in the weight stream. CTA = 4 waves x one split; wave w owns 4
// adjacent tile columns, whose tiles of one k are one contiguous run of 4 x Bits x 32 bytes in the file layout. The
// wave streams its runs with 16-byte loads through a register ring (kLin10Depth k-rows in flight), stages each run in
// a wave-private LDS row (LDS operations of one wave complete in order, so no barrier), reads each lane's three words
// per column from it, and reuses one read of the k-row's activation pairs for its 4 columns. The split's transformed
// inputs are copied to LDS once (as v4). Decode, per-column fdot2 chains, reduction and finish are those of v4
// (bitwise). Rows: static bucket R (padded rows never stored, as v8).
constexpr unsigned kLin10Depth = 4;
template <int R>
__global__ __launch_bounds__(128) void linear_core10_kernel(LinArgs a) {
    extern __shared__ __attribute__((aligned(16))) unsigned char lin10_smem[];  // [R][xs_stride] halves, then 4 rows
    unsigned ji = 0;
    while (ji + 1 < a.count && blockIdx.x >= a.job[ji + 1].cta_begin) ++ji;
    const LinJob& j = a.job[ji];
    const unsigned cols = j.w.output / 16U, groups = j.w.output / 256U;  // CTA = 16 columns
    const unsigned local = blockIdx.x - j.cta_begin, split = local / groups, group = local % groups;
    const unsigned rows = a.rows;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned kt = j.w.input / 16, k0 = kt * split / j.splits, k1 = kt * (split + 1) / j.splits;
    const unsigned ch = (k1 - k0) * 16;
    const auto* xt = reinterpret_cast<const _Float16*>(a.ws) + j.xt_off;
    auto* xs = reinterpret_cast<_Float16*>(lin10_smem);
    for (unsigned q = threadIdx.x; q < static_cast<unsigned>(R) * (ch / 2); q += 128) {
        const unsigned r = q / (ch / 2), c = q % (ch / 2);
        reinterpret_cast<std::uint32_t*>(xs)[r * (a.xs_stride / 2) + c] =
            reinterpret_cast<const std::uint32_t*>(xt + static_cast<std::size_t>(r) * j.w.input + k0 * 16)[c];
    }
    __syncthreads();
    const unsigned col0 = group * 16U + wave * 4U;  // the wave's first column
    float* part = reinterpret_cast<float*>(a.ws) + j.part_off;
    using V4 = std::uint32_t __attribute__((ext_vector_type(4)));
    auto run = [&](auto bits_tag) __attribute__((always_inline)) {
        constexpr unsigned Bits = static_cast<unsigned>(decltype(bits_tag)::value);
        constexpr unsigned runU = 8U * Bits;               // V4 units in the wave's 4-tile run
        constexpr unsigned UPL = (runU + 31U) / 32U;       // units per lane
        auto* rowbuf = reinterpret_cast<V4*>(lin10_smem + static_cast<std::size_t>(R) * a.xs_stride * 2) + wave * runU;
        const auto* gw = reinterpret_cast<const V4*>(d::uniform(j.w.trellis));
        const std::size_t kstride = static_cast<std::size_t>(cols) * 2U * Bits;  // V4 per k-row of the matrix
        const std::size_t cbase = static_cast<std::size_t>(col0) * 2U * Bits;
        V4 ring[kLin10Depth][UPL] = {};
        auto load = [&](unsigned k, auto slot) __attribute__((always_inline)) {
            constexpr unsigned q = decltype(slot)::value;
#pragma unroll
            for (unsigned u = 0; u < UPL; ++u) {
                const unsigned e = lane + 32U * u;
                if (e < runU) ring[q][u] = gw[static_cast<std::size_t>(k) * kstride + cbase + e];
            }
        };
        const d::LaneGeom<static_cast<int>(Bits)> geo(lane);
        float sums[4][R][2];
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) sums[c][r][0] = sums[c][r][1] = 0.0F;
        const auto* rl = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(rowbuf));
        const auto* xw = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
        auto step = [&](unsigned k, auto slot) __attribute__((always_inline)) {
            constexpr unsigned q = decltype(slot)::value;
#pragma unroll
            for (unsigned u = 0; u < UPL; ++u) {
                const unsigned e = lane + 32U * u;
                if (e < runU) rowbuf[e] = ring[q][u];
            }
            if (k + kLin10Depth < k1) load(k + kLin10Depth, slot);
            d::Half2 act[R][2];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const auto* pr = xw + (static_cast<unsigned>(r) * a.xs_stride) / 2U + (k - k0) * 8U;
                act[r][0] = __builtin_bit_cast(d::Half2, pr[0]);
                act[r][1] = __builtin_bit_cast(d::Half2, pr[4]);
            }
#pragma unroll
            for (unsigned c = 0; c < 4; ++c) {
                d::Slot<static_cast<int>(Bits), R> cur;
                const unsigned tw = c * Bits * 8U;
                cur.a = rl[tw + geo.oa / 4U];
                cur.b = rl[tw + geo.ob / 4U];
                cur.c = rl[tw + geo.oc / 4U];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    cur.act[r][0] = act[r][0];
                    cur.act[r][1] = act[r][1];
                }
                d::compute_slot<static_cast<int>(Bits), R>(cur, geo, static_cast<unsigned>(R), sums[c]);
            }
        };
        using S0 = std::integral_constant<unsigned, 0>;
        using S1 = std::integral_constant<unsigned, 1>;
        using S2 = std::integral_constant<unsigned, 2>;
        using S3 = std::integral_constant<unsigned, 3>;
        static_assert(kLin10Depth == 4);
        if (k0 + 0 < k1) load(k0 + 0, S0{});
        if (k0 + 1 < k1) load(k0 + 1, S1{});
        if (k0 + 2 < k1) load(k0 + 2, S2{});
        if (k0 + 3 < k1) load(k0 + 3, S3{});
        for (unsigned k = k0; k < k1; k += 4) {
            step(k, S0{});
            if (k + 1 < k1) step(k + 1, S1{});
            if (k + 2 < k1) step(k + 2, S2{});
            if (k + 3 < k1) step(k + 3, S3{});
        }
#pragma unroll
        for (unsigned c = 0; c < 4; ++c) {
            float acc[R][2];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r][0] = 0.0F + d::reduce4(sums[c][r][0]);
                acc[r][1] = 0.0F + d::reduce4(sums[c][r][1]);
            }
            linear_store_partial<R>(j, part, split, col0 + c, rows, lane, acc);
        }
    };
    switch (j.w.bits) {
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
        default: run(std::integral_constant<int, 8>{}); break;
    }
    // CTA = 2 output blocks of 128; each block's arrivals: one CTA per split.
    auto* counters = reinterpret_cast<std::uint32_t*>(a.ws + a.counter_off);
#pragma unroll
    for (unsigned h = 0; h < 2; ++h) {
        const unsigned block = group * 2U + h;
        if (last_arrival(counters + j.block_begin + block, j.splits)) linear_finish(j, part, block, rows, wave, lane);
    }
}

// Linear v11 (reached only from enqueue_linears_scheduled at 7 rows, the DSpark draft block; STRIX_MMX_LINEAR_V11=0
// disables): the v10 core with (a) an exact 7-row bucket (no padded eighth row), (b) W waves per CTA sharing ONE LDS copy
// of the split inputs (W = 8 doubles the resident waves per SIMD at the same LDS; v10 is W = 4), (c) a register ring of
// D k-rows, (d) for K3 the two-word reader (the third word of the lane window never reaches a K3 state: lane shifts
// 3/11/19/27, the cross-word states need the top 2/5 bits of `low`). Per-column chains, split boundaries, reductions and
// the finish (ascending splits, H128, svh) are those of v10 -> the same bits (measured: 56 forms x 5 layers x 2 calls ==
// enqueue_linears_scheduled). CTA = W waves x one split; wave w owns 4 adjacent tile columns;
// CTA = 4W columns = W/2 output blocks.
template <int Bits, int R, bool Two>
__device__ __forceinline__ void v11_compute(std::uint32_t a_, std::uint32_t b_, std::uint32_t c_, const d::LaneGeom<Bits>& g,
                                            const d::Half2 (&act)[R][2], float (&sums)[R][2]) {
    d::TileStates<Bits> t;
    t.high = __builtin_amdgcn_alignbit(a_, b_, g.align);
    if constexpr (Two) {
        t.low = b_ << (32U - g.align);
        t.tail = 0;
    } else {
        t.low = __builtin_amdgcn_alignbit(b_, c_, g.align);
        t.tail = c_ << (32U - g.align);
    }
    const d::Half2 w0 = d::mul1_pair(t.template state<0>(), t.template state<1>());
    const d::Half2 w1 = d::mul1_pair(t.template state<2>(), t.template state<3>());
    const d::Half2 w2 = d::mul1_pair(t.template state<4>(), t.template state<5>());
    const d::Half2 w3 = d::mul1_pair(t.template state<6>(), t.template state<7>());
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sums[r][0] = __builtin_amdgcn_fdot2(w0, act[r][0], sums[r][0], false);
        sums[r][0] = __builtin_amdgcn_fdot2(w1, act[r][1], sums[r][0], false);
        sums[r][1] = __builtin_amdgcn_fdot2(w2, act[r][0], sums[r][1], false);
        sums[r][1] = __builtin_amdgcn_fdot2(w3, act[r][1], sums[r][1], false);
    }
}
template <int Bits, int R, int D, int W>
__global__ __launch_bounds__(32 * W) void linear_core11_kernel(LinArgs a) {
    extern __shared__ __attribute__((aligned(16))) unsigned char lin11_smem[];  // [R][xs_stride] halves, then W rows
    unsigned ji = 0;
    while (ji + 1 < a.count && blockIdx.x >= a.job[ji + 1].cta_begin) ++ji;
    const LinJob& j = a.job[ji];
    constexpr unsigned kCols = 4U * W;
    const unsigned cols = j.w.output / 16U, groups = j.w.output / (16U * kCols);
    const unsigned local = blockIdx.x - j.cta_begin, split = local / groups, group = local % groups;
    const unsigned rows = a.rows;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned kt = j.w.input / 16, k0 = kt * split / j.splits, k1 = kt * (split + 1) / j.splits;
    const unsigned ch = (k1 - k0) * 16;
    const auto* xt = reinterpret_cast<const _Float16*>(a.ws) + j.xt_off;
    auto* xs = reinterpret_cast<_Float16*>(lin11_smem);
    for (unsigned q = threadIdx.x; q < static_cast<unsigned>(R) * (ch / 2); q += 32U * W) {
        const unsigned r = q / (ch / 2), c = q % (ch / 2);
        reinterpret_cast<std::uint32_t*>(xs)[r * (a.xs_stride / 2) + c] =
            reinterpret_cast<const std::uint32_t*>(xt + static_cast<std::size_t>(r) * j.w.input + k0 * 16)[c];
    }
    __syncthreads();
    const unsigned col0 = group * kCols + wave * 4U;
    float* part = reinterpret_cast<float*>(a.ws) + j.part_off;
    using V4 = std::uint32_t __attribute__((ext_vector_type(4)));
    constexpr unsigned runU = 8U * Bits, UPL = (runU + 31U) / 32U;
    constexpr bool Two = Bits == 3;
    auto* rowbuf = reinterpret_cast<V4*>(lin11_smem + static_cast<std::size_t>(R) * a.xs_stride * 2) + wave * runU;
    const auto* gw = reinterpret_cast<const V4*>(d::uniform(j.w.trellis));
    const std::size_t kstride = static_cast<std::size_t>(cols) * 2U * Bits;
    const std::size_t cbase = static_cast<std::size_t>(col0) * 2U * Bits;
    V4 ring[D][UPL] = {};
    const d::LaneGeom<Bits> geo(lane);
    float sums[4][R][2];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < R; ++r) sums[c][r][0] = sums[c][r][1] = 0.0F;
    const auto* rl = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(rowbuf));
    const auto* xw = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
#pragma unroll
    for (int q = 0; q < D; ++q)
        if (k0 + static_cast<unsigned>(q) < k1)
#pragma unroll
            for (unsigned u = 0; u < UPL; ++u) {
                const unsigned e = lane + 32U * u;
                if (e < runU) ring[q][u] = gw[static_cast<std::size_t>(k0 + static_cast<unsigned>(q)) * kstride + cbase + e];
            }
    for (unsigned kb = k0; kb < k1; kb += D) {
#pragma unroll
        for (int q = 0; q < D; ++q) {
            const unsigned k = kb + static_cast<unsigned>(q);
            if (k < k1) {
#pragma unroll
                for (unsigned u = 0; u < UPL; ++u) {
                    const unsigned e = lane + 32U * u;
                    if (e < runU) rowbuf[e] = ring[q][u];
                }
                if (k + D < k1)
#pragma unroll
                    for (unsigned u = 0; u < UPL; ++u) {
                        const unsigned e = lane + 32U * u;
                        if (e < runU) ring[q][u] = gw[static_cast<std::size_t>(k + D) * kstride + cbase + e];
                    }
                d::Half2 act[R][2];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const auto* pr = xw + (static_cast<unsigned>(r) * a.xs_stride) / 2U + (k - k0) * 8U;
                    act[r][0] = __builtin_bit_cast(d::Half2, pr[0]);
                    act[r][1] = __builtin_bit_cast(d::Half2, pr[4]);
                }
#pragma unroll
                for (unsigned c = 0; c < 4; ++c) {
                    const unsigned tw = c * Bits * 8U;
                    const std::uint32_t wa = rl[tw + geo.oa / 4U], wb = rl[tw + geo.ob / 4U];
                    const std::uint32_t wc = Two ? 0U : rl[tw + geo.oc / 4U];
                    v11_compute<Bits, R, Two>(wa, wb, wc, geo, act, sums[c]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned c = 0; c < 4; ++c) {
        float acc[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r][0] = 0.0F + d::reduce4(sums[c][r][0]);
            acc[r][1] = 0.0F + d::reduce4(sums[c][r][1]);
        }
        linear_store_partial<R>(j, part, split, col0 + c, rows, lane, acc);
    }
    auto* counters = reinterpret_cast<std::uint32_t*>(a.ws + a.counter_off);
#pragma unroll
    for (unsigned h = 0; h < W / 2; ++h) {
        const unsigned block = group * (W / 2) + h;
        if (last_arrival(counters + j.block_begin + block, j.splits)) {
            for (unsigned r = wave; r < rows; r += W) {  // = linear_finish with a W-wave row stride (same bits per row)
                float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                for (unsigned sp = 0; sp < j.splits; ++sp)
#pragma unroll
                    for (int qq = 0; qq < 4; ++qq)
                        v[qq] += load_agent(part + (static_cast<std::size_t>(sp) * rows + r) * j.w.output + block * 128 + lane + 32U * static_cast<unsigned>(qq));
                d::hadamard128(v, lane);
#pragma unroll
                for (int qq = 0; qq < 4; ++qq) {
                    const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(qq);
                    j.y[static_cast<std::size_t>(r) * j.y_stride + i] = v[qq] * d::half_bits(j.w.svh[i]);
                }
            }
        }
    }
}

struct LinPlan {
    LinArgs args{};
    unsigned ctas = 0, prep_ctas = 0;
    std::size_t bytes = 0;
};
LinPlan plan_linears(const LinearJob* jobs, unsigned count, unsigned rows) {
    if (count < 1 || count > kMaxJobs) throw std::invalid_argument("minimax linears: 1..4 jobs");
    if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("minimax linears: rows must be 1..8");
    LinPlan p;
    p.args.count = count;
    p.args.rows = rows;
    std::size_t xt = 0, part = 0;
    unsigned blocks = 0, max_in = 0;
    for (unsigned i = 0; i < count; ++i) {
        const Exl3& w = jobs[i].w;
        if (w.input % 128 || w.output % 128 || w.bits < 3 || w.bits > 8 || w.bits == 7)
            throw std::invalid_argument("minimax linears: input/output multiples of 128, bits 3, 4, 5, 6 or 8");
        LinJob& j = p.args.job[i];
        j.w = w;
        j.x = jobs[i].x;
        j.y = jobs[i].y;
        j.x_stride = jobs[i].x_stride;
        j.y_stride = jobs[i].y_stride;
        j.splits = std::min(d::vector_splits(w.input, w.output), w.input / 16);
        j.cta_begin = p.ctas;
        j.block_begin = blocks;
        j.xt_off = xt;
        j.part_off = part;
        p.ctas += (w.output / 64) * j.splits;
        blocks += w.output / 128;
        xt += (std::size_t{kMaxRows} * w.input + 127) & ~std::size_t{127};
        part += (std::size_t{j.splits} * kMaxRows * w.output + 63) & ~std::size_t{63};
        max_in = std::max(max_in, w.input);
        p.args.xs_stride = std::max(p.args.xs_stride, ((w.input / 16 + j.splits - 1) / j.splits) * 16U);
        const unsigned kt = w.input / 16;
        for (unsigned sp = 0; sp < j.splits; ++sp) {
            const unsigned k0 = kt * sp / j.splits, k1 = kt * (sp + 1) / j.splits;
            p.args.xs_stride5 = std::max(p.args.xs_stride5, ((k1 + 7) / 8 - k0 / 8) * 128U);
        }
        j.prep_begin = p.args.prep_jobs;
        p.args.prep_jobs += rows * (w.input / 128);
    }
    // Counters at a fixed offset (every call of a workspace finds them at 0 and leaves them at 0), then xt (halves),
    // then partials (floats).
    if (blocks > kLinCounters - 2) throw std::invalid_argument("minimax linears: too many output blocks");
    const std::size_t cnt_bytes = align256(std::size_t{kLinCounters} * 4);
    const std::size_t xt_bytes = align256(xt * 2);
    for (unsigned i = 0; i < count; ++i) {
        p.args.job[i].xt_off += cnt_bytes / 2;
        p.args.job[i].part_off += (cnt_bytes + xt_bytes) / 4;
    }
    p.args.counter_off = 0;
    p.bytes = cnt_bytes + xt_bytes + align256(part * 4);
    p.prep_ctas = std::max(1U, (rows * (max_in / 128) + 7) / 8);
    p.args.prep_ctas = std::min(64U, (p.args.prep_jobs + 3) / 4);
    return p;
}

}  // namespace

int linear_version() {
    static const int v = [] {
        const char* e = std::getenv("STRIX_MMX_LINEAR_V");
        const int x = e != nullptr ? std::atoi(e) : 0;  // 0 = the measured schedule (scheduled_linear_version)
        if (x != 0 && x != 1 && (x < 3 || x > 10)) throw std::invalid_argument("STRIX_MMX_LINEAR_V must be 0, 1 or 3..10");
        return x;
    }();
    return v;
}
template <int MaxR, int Depth> void launch_linear3_r(const LinPlan& p, unsigned blocks, hipStream_t stream) {
    hipLaunchKernelGGL((linear3_kernel<MaxR, Depth>), dim3(blocks), dim3(256), 0, stream, p.args);
    launched("linear3");
}
template <int Depth> void launch_linear3(const LinPlan& p, unsigned blocks, hipStream_t stream) {
    if (p.args.rows == 1) launch_linear3_r<1, Depth>(p, blocks, stream);
    else if (p.args.rows == 2) launch_linear3_r<2, Depth>(p, blocks, stream);
    else if (p.args.rows <= 4) launch_linear3_r<4, Depth>(p, blocks, stream);
    else launch_linear3_r<8, Depth>(p, blocks, stream);
}
template <int Depth, bool Prime = false, bool Trace = false> void launch_linear_core4(const LinPlan& p, hipStream_t stream) {
    if constexpr (!Trace) {
        if (p.args.trace != nullptr) {
            launch_linear_core4<Depth, Prime, true>(p, stream);
            return;
        }
    }
    const unsigned R = p.args.rows == 1 ? 1U : (p.args.rows == 2 ? 2U : (p.args.rows <= 4 ? 4U : 8U));
    const std::size_t lds = std::size_t{R} * p.args.xs_stride * 2;
    const bool exact = p.args.rows == R;
    if (R == 1) hipLaunchKernelGGL((linear_core4_kernel<1, Depth, true, Prime, Trace>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    else if (R == 2) hipLaunchKernelGGL((linear_core4_kernel<2, Depth, true, Prime, Trace>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    else if (R == 4 && exact) hipLaunchKernelGGL((linear_core4_kernel<4, Depth, true, Prime, Trace>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    else if (R == 4) hipLaunchKernelGGL((linear_core4_kernel<4, Depth, false, Prime, Trace>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    else if (exact) hipLaunchKernelGGL((linear_core4_kernel<8, Depth, true, Prime, Trace>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    else hipLaunchKernelGGL((linear_core4_kernel<8, Depth, false, Prime, Trace>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    launched("linear core4");
}
template <int R, int Depth, bool Static, int V> void launch_lin56_rs(const LinPlan& p, std::size_t lds, unsigned grid, hipStream_t stream) {
    if constexpr (V == 5) hipLaunchKernelGGL((linear_core5_kernel<R, Depth, Static>), dim3(grid), dim3(128), lds, stream, p.args);
    else hipLaunchKernelGGL((linear_core6_kernel<R, Depth, Static>), dim3(grid), dim3(128), lds, stream, p.args);
}
template <int Depth, int V> void launch_linear_core56(const LinPlan& p, hipStream_t stream) {
    const unsigned R = p.args.rows == 1 ? 1U : (p.args.rows == 2 ? 2U : (p.args.rows <= 4 ? 4U : 8U));
    const std::size_t lds = std::size_t{R} * (V == 5 ? p.args.xs_stride5 : p.args.xs_stride) * 2;
    const unsigned grid = p.ctas + (V == 6 ? p.args.prep_ctas : 0U);
    const bool exact = p.args.rows == R;
    if (R == 1) launch_lin56_rs<1, Depth, true, V>(p, lds, grid, stream);
    else if (R == 2) launch_lin56_rs<2, Depth, true, V>(p, lds, grid, stream);
    else if (R == 4 && exact) launch_lin56_rs<4, Depth, true, V>(p, lds, grid, stream);
    else if (R == 4) launch_lin56_rs<4, Depth, false, V>(p, lds, grid, stream);
    else if (exact) launch_lin56_rs<8, Depth, true, V>(p, lds, grid, stream);
    else launch_lin56_rs<8, Depth, false, V>(p, lds, grid, stream);
    launched(V == 5 ? "linear core5" : "linear core6");
}
int v9_mode() {
    static const int m = [] {
        const char* e = std::getenv("STRIX_MMX_V9_MODE");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    return m;
}
// v11 schedule (bracketed A-V-A x3 vs enqueue_linears_scheduled at 7 rows, K3 / K4, 5 layers; V/A medians):
//   q+k+v (3 jobs 6144 -> 8192 / 512 / 512)  K3 D6 W8 0.79   K4 D4 W8 0.78
//   o (8192 -> 6144)                          K3 D4 W8 0.80   K4 D4 W8 0.81
//   gate+up (2 jobs 6144 -> 12288)            K3 D6 W8 0.84   K4 D8 W4 0.93
//   down (12288 -> 6144): every v10/v11 form slower than v9 -> not taken.
// try_linear_v11 returns false when the call is not one of these (the caller keeps its schedule).
namespace {
bool linear_v11_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_LINEAR_V11");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}
template <int Bits, int D, int W> void launch_v11(const LinPlan& p, unsigned max_bits, hipStream_t stream) {
    LinPlan q = p;  // re-based grid: per job (output / (64 W)) x splits CTAs
    unsigned ctas = 0;
    for (unsigned i = 0; i < q.args.count; ++i) {
        q.args.job[i].cta_begin = ctas;
        ctas += (q.args.job[i].w.output / (64U * W)) * q.args.job[i].splits;
    }
    const std::size_t lds = std::size_t{7} * q.args.xs_stride * 2 + std::size_t{W} * 128 * max_bits;
    hipLaunchKernelGGL((linear_core11_kernel<Bits, 7, D, W>), dim3(ctas), dim3(32 * W), lds, stream, q.args);
    launched("linear core11");
}
bool try_linear_v11(const LinPlan& p, const LinearJob* jobs, unsigned count, unsigned rows, hipStream_t stream) {
    if (!linear_v11_enabled() || rows != 7) return false;
    const unsigned bits = jobs[0].w.bits;
    if (bits != 3 && bits != 4) return false;
    for (unsigned i = 0; i < count; ++i)
        if (jobs[i].w.bits != bits || jobs[i].w.output % 512 != 0) return false;
    const Exl3& w = jobs[0].w;
    const bool qkv = count == 3 && w.input == 6144 && w.output == 8192 && jobs[1].w.input == 6144 && jobs[1].w.output == 512 &&
                     jobs[2].w.input == 6144 && jobs[2].w.output == 512;
    const bool o = count == 1 && w.input == 8192 && w.output == 6144;
    const bool gu = count == 2 && w.input == 6144 && w.output == 12288 && jobs[1].w.input == 6144 && jobs[1].w.output == 12288;
    if (!qkv && !o && !gu) return false;
    if (bits == 3) {
        if (o) launch_v11<3, 4, 8>(p, 3, stream);
        else launch_v11<3, 6, 8>(p, 3, stream);
    } else {
        if (gu) launch_v11<4, 8, 4>(p, 4, stream);
        else launch_v11<4, 4, 8>(p, 4, stream);
    }
    return true;
}
}  // namespace

template <int R> void launch_linear_core10_r(const LinPlan& p, hipStream_t stream) {
    unsigned max_bits = 0, ctas = 0;
    for (unsigned i = 0; i < p.args.count; ++i) {
        const LinJob& j = p.args.job[i];
        max_bits = std::max(max_bits, j.w.bits);
        if (j.w.output % 256) throw std::invalid_argument("minimax linears v10: outputs must be multiples of 256");
    }
    // v10 grid: per job (output / 256) x splits CTAs; the plan's cta_begin is for the v4 grid, so a copy is re-based.
    LinPlan q = p;
    for (unsigned i = 0; i < q.args.count; ++i) {
        q.args.job[i].cta_begin = ctas;
        ctas += (q.args.job[i].w.output / 256) * q.args.job[i].splits;
    }
    const std::size_t lds = std::size_t{R} * q.args.xs_stride * 2 + 4 * std::size_t{128} * max_bits;
    hipLaunchKernelGGL((linear_core10_kernel<R>), dim3(ctas), dim3(128), lds, stream, q.args);
    launched("linear core10");
}
template <int R> void launch_linear_core9_r(const LinPlan& p, hipStream_t stream) {
    unsigned max_bits = 0;
    for (unsigned i = 0; i < p.args.count; ++i) max_bits = std::max(max_bits, p.args.job[i].w.bits);
    const std::size_t lds = lin9_lds_bytes<R>(max_bits);
    if (v9_mode() == 1) hipLaunchKernelGGL((linear_core9_kernel<R, 1>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    else if (v9_mode() == 2) hipLaunchKernelGGL((linear_core9_kernel<R, 2>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    else hipLaunchKernelGGL((linear_core9_kernel<R>), dim3(p.ctas), dim3(128), lds, stream, p.args);
    launched("linear core9");
}
// pad (v8): always the static R-row body (padded rows are computed but never stored).
template <int R, int Depth> void launch_linear_core_r(const LinPlan& p, hipStream_t stream, bool pad) {
    if (p.args.rows == R || pad) hipLaunchKernelGGL((linear_core_kernel<R, Depth, true>), dim3(p.ctas), dim3(128), 0, stream, p.args);
    else hipLaunchKernelGGL((linear_core_kernel<R, Depth, false>), dim3(p.ctas), dim3(128), 0, stream, p.args);
}
template <int Depth> void launch_linear_core(const LinPlan& p, hipStream_t stream, bool pad) {
    if (p.args.rows == 1) launch_linear_core_r<1, Depth>(p, stream, pad);
    else if (p.args.rows == 2) launch_linear_core_r<2, Depth>(p, stream, pad);
    else if (p.args.rows <= 4) launch_linear_core_r<4, Depth>(p, stream, pad);
    else launch_linear_core_r<8, Depth>(p, stream, pad);
}

// Default form per site and row count (STRIX_MMX_LINEAR_V unset): the fastest exact form measured (all forms bitwise,
// dense cells ABA x3 against strix_gemv in one process), v4 where nothing beat it. v8 runs at depth 4 for every row count.
//   q, q+k+v      v8 at every N (q 0.82-0.95, q+k+v 0.66-0.73 of strix_gemv)
//   k, v          v4 at N1-4, v8 at N5-8
//   o             v10 at N1, v9 at N2, v4 at N3-4, v8 at N5-8
//   dense 6144^2  v10 at N1-4 (0.69-0.72), v8 at N5-8
//   other shapes  v10 (expert / shared shapes as single calls: 0.58-0.68, measurement); outputs not a multiple of 256:
//                 v4 at N1 and N8, v8 at N2-7
//   head          v4 (the library's strix_gemv stays the head's path, see mmx_moe.hpp)
//   DSpark draft (measurement, K4, whole matrices; confirmed in measurement against v10, processes
//   S-10-S-10-S, predeclared bracket rule, noise bound 0.020; gate+up 0.79-0.91, down 0.83-0.90):
//     6144 -> 12288 (gate+up)  v8 at N1-4 (v8/v10 0.78 / 0.90 / 0.90 / 0.90), v10 at N5-8
//     12288 -> 6144 (down)     v10 at N1-2, v9 at N3-8 (v9/v10 0.87-0.90)
//     6144 -> 8192 (q+k+v) v8 as the other q+k+v groups; 8192 -> 6144 (o) v10 as the other shapes
int scheduled_linear_version(const LinearJob* jobs, unsigned count, unsigned rows) {
    return route::scheduled_linear_version(jobs[0].w.input, jobs[0].w.output, count, rows);
}

namespace {
std::uint64_t* g_lin_trace = nullptr;
}
void set_linear_trace(std::uint64_t* trace) { g_lin_trace = trace; }

std::size_t linear_workspace_bytes(const LinearJob* jobs, unsigned count) {
    return plan_linears(jobs, count, kMaxRows).bytes;
}

namespace {
void enqueue_linears_impl(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream,
                          hipEvent_t* marks, bool force_schedule, bool prepared = false, const PublishTarget* pub = nullptr,
                          bool draft_block = false, int force_version = 0) {
    LinPlan p = plan_linears(jobs, count, rows);
    p.args.ws = static_cast<unsigned char*>(workspace);
    p.args.trace = g_lin_trace;
    // force_version (enqueue_linears_version): that form exactly as STRIX_MMX_LINEAR_V=force_version selects it
    const bool sched = force_version == 0 && (force_schedule || linear_version() == 0);
    const int v = force_version != 0 ? force_version : (sched ? scheduled_linear_version(jobs, count, rows) : linear_version());
    if (pub != nullptr && pub->flag != nullptr) {
        if (v != 4) throw std::logic_error("minimax linears: the publish epilogue needs the v4 core (STRIX_MMX_LINEAR_V=4)");
        p.args.pub = *pub;
        p.args.pub.tiles = 0;
        for (unsigned i = 0; i < count; ++i) p.args.pub.tiles += jobs[i].w.output / 128U;
    }
    if (prepared && (v == 3 || v == 5 || v == 6))
        throw std::logic_error("minimax linears: inputs_prepared needs a two-launch form (STRIX_MMX_LINEAR_V 1, 4, 7..10)");
    if (v == 3) {
        unsigned blocks = 0;
        for (unsigned i = 0; i < count; ++i) blocks += jobs[i].w.output / 128;
        switch (moe_depth3()) {
            case 4: launch_linear3<4>(p, blocks, stream); break;
            case 12: launch_linear3<12>(p, blocks, stream); break;
            default: launch_linear3<8>(p, blocks, stream); break;
        }
        return;
    }
    if (v == 5 || v == 6) {
        const bool v5 = v == 5;
        switch (moe_depth3()) {
            case 8: v5 ? launch_linear_core56<8, 5>(p, stream) : launch_linear_core56<8, 6>(p, stream); break;
            case 12: v5 ? launch_linear_core56<12, 5>(p, stream) : launch_linear_core56<12, 6>(p, stream); break;
            default: v5 ? launch_linear_core56<4, 5>(p, stream) : launch_linear_core56<4, 6>(p, stream); break;
        }
        return;
    }
    if (!prepared) {
        hipLaunchKernelGGL(linear_prepare_kernel, dim3(p.prep_ctas, count), dim3(256), 0, stream, p.args);
        launched("linear prepare");
    }
    if (marks != nullptr) (void)hipEventRecord(marks[0], stream);
    if (force_schedule && draft_block && try_linear_v11(p, jobs, count, rows, stream)) return;  // v11: draft block, 7 rows
    if (v == 10) {
        if (rows == 1) launch_linear_core10_r<1>(p, stream);
        else if (rows == 2) launch_linear_core10_r<2>(p, stream);
        else if (rows <= 4) launch_linear_core10_r<4>(p, stream);
        else launch_linear_core10_r<8>(p, stream);
        return;
    }
    if (v == 9) {
        if (rows == 1) launch_linear_core9_r<1>(p, stream);
        else if (rows == 2) launch_linear_core9_r<2>(p, stream);
        else if (rows <= 4) launch_linear_core9_r<4>(p, stream);
        else launch_linear_core9_r<8>(p, stream);
        return;
    }
    if (v == 4 || v == 7) {
        const bool prime = v == 7;
        switch (moe_depth3()) {
            case 4: prime ? launch_linear_core4<4, true>(p, stream) : launch_linear_core4<4>(p, stream); break;
            case 12: prime ? launch_linear_core4<12, true>(p, stream) : launch_linear_core4<12>(p, stream); break;
            default: prime ? launch_linear_core4<8, true>(p, stream) : launch_linear_core4<8>(p, stream); break;
        }
        return;
    }
    if (v == 8 && sched) {
        launch_linear_core<4>(p, stream, true);  // the schedule's v8: depth 4, static padded rows
    } else {
        switch (moe_depth(rows)) {
            case 4: launch_linear_core<4>(p, stream, v == 8); break;
            case 8: launch_linear_core<8>(p, stream, v == 8); break;
            default: launch_linear_core<0>(p, stream, v == 8); break;
        }
    }
    launched("linear core");
}
}  // namespace

void enqueue_linears(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream, hipEvent_t* marks) {
    enqueue_linears_impl(jobs, count, rows, workspace, stream, marks, false);
}
void enqueue_linears_prepared(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream) {
    enqueue_linears_impl(jobs, count, rows, workspace, stream, nullptr, false, true);
}
void enqueue_linears_publish(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream,
                             const PublishTarget& pub, bool prepared) {
    enqueue_linears_impl(jobs, count, rows, workspace, stream, nullptr, false, prepared, &pub);
}
bool producer_publish_supported() {
    return linear_version() == 4 && moe_overlap() == 1 && moe_version() == 4 && moe_waves() == 4;
}
void linear_xt_offsets(const LinearJob* jobs, unsigned count, std::size_t* offs) {
    const LinPlan p = plan_linears(jobs, count, 1);  // the xt layout depends on the jobs' inputs only (kMaxRows rows each)
    for (unsigned i = 0; i < count; ++i) offs[i] = p.args.job[i].xt_off;
}
void enqueue_linears_scheduled(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream,
                               bool draft_block) {
    enqueue_linears_impl(jobs, count, rows, workspace, stream, nullptr, true, false, nullptr, draft_block);
}
void enqueue_linears_version(const LinearJob* jobs, unsigned count, unsigned rows, void* workspace, hipStream_t stream, int version) {
    if (version != 1 && (version < 3 || version > 10)) throw std::invalid_argument("enqueue_linears_version: 1 or 3..10");
    enqueue_linears_impl(jobs, count, rows, workspace, stream, nullptr, false, false, nullptr, false, version);
}

}  // namespace strix::minimax::moe::opt
