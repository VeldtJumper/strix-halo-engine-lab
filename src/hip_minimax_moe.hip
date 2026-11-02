// minimax-m3 fused post-exchange kernel (declared order: strix/minimax_moe.hpp).
#include "strix/hip_minimax_moe.hpp"
#include "minimax/moe/mmx_trellis_device.hpp"  // dev::transform_block (STRIX_MM_XB_PREP)

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace strix::minimax::moe::hip {
namespace {

constexpr unsigned kWaves = kNormThreads / 32;

template <unsigned Ilp>
__global__ __launch_bounds__(kNormThreads) void post_exchange_kernel(PostExchangeView v) {
    __shared__ __attribute__((aligned(16))) float hs[kHidden];  // 16-byte aligned: mode 2 stages float4 rows here
    __shared__ __attribute__((aligned(16))) float ys[kHidden];
    __shared__ float part[kNormThreads];
    __shared__ float sc[kExperts], ch[kExperts];
    const unsigned r = blockIdx.x, t = threadIdx.x, lane = t & 31U, wave = t >> 5U;
    constexpr unsigned kPer = kHidden / kNormThreads;
    static_assert(kHidden % kNormThreads == 0);
    [[maybe_unused]] float hv[Ilp == 2 ? kPer : 1], nw[Ilp == 2 ? kPer : 1];
    if constexpr (Ilp == 2) {  // mode 2: the residual and the norm weights do not depend on the exchange: load them
                               // before the receive poll (registers), so only the partials remain after it
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            hv[q] = v.h[static_cast<std::size_t>(r) * kHidden + t + q * kNormThreads];
            nw[q] = v.norm_w[t + q * kNormThreads];
        }
    }
    if (v.rx_flag != nullptr) {  // the exchange module's in-kernel wait (xfer::mmx_wait without the gate)
        // Every block (one per row, <= 8 in decode) polls the RX flag itself: no block depends on another block being
        // scheduled. A failed exchange releases the flag with ~0, so the wait always ends.
        if (t == 0)
            while (__hip_atomic_load(v.rx_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < v.rx_seq) __builtin_amdgcn_s_sleep(2);
        __syncthreads();
    }
    float* h = v.h + static_cast<std::size_t>(r) * kHidden;
    const float* p0 = v.part0 + static_cast<std::size_t>(r) * kHidden;
    const float* p1 = v.part1 + static_cast<std::size_t>(r) * kHidden;
    float acc = 0.0F;
    if constexpr (Ilp == 2) {  // STRIX_MM_XB_POSTILP mode 2: the partials (GPU-uncached exchange slots) staged into LDS
        // with coalesced 16-byte loads (4x fewer uncached requests), then the plain chain reads them from LDS: same
        // values, same per-thread j order, same expressions -> identical bits. ys is overwritten by y only after the
        // reduction barriers; each hs[j] is read and then written by the same thread.
        const auto* p0v = reinterpret_cast<const float4*>(p0);
        const auto* p1v = reinterpret_cast<const float4*>(p1);
        for (unsigned k = t; k < kHidden / 4U; k += kNormThreads) {
            const float4 x0 = p0v[k], x1 = p1v[k];
            reinterpret_cast<float4*>(hs)[k] = x0;
            reinterpret_cast<float4*>(ys)[k] = x1;
        }
        __syncthreads();
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            const unsigned j = t + q * kNormThreads;
            const float x = hv[q] + (hs[j] + ys[j]);
            h[j] = x;
            hs[j] = x;
            acc = __builtin_fmaf(x, x, acc);
        }
    } else if constexpr (Ilp == 1) {  // STRIX_MM_XB_POSTILP mode 1: every load issued before the first store (the partials live in GPU-uncached
                  // exchange slots; the store to h may alias them for the compiler, which serialises the plain loop into
                  // 24 round trips). Same per-thread chain (j ascending) and the same expressions: identical bits.
        float a[kPer], b[kPer], c[kPer];
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            const unsigned j = t + q * kNormThreads;
            a[q] = h[j];
            b[q] = p0[j];
            c[q] = p1[j];
        }
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            const unsigned j = t + q * kNormThreads;
            const float x = a[q] + (b[q] + c[q]);
            h[j] = x;
            hs[j] = x;
            acc = __builtin_fmaf(x, x, acc);
        }
    } else {
        for (unsigned j = t; j < kHidden; j += kNormThreads) {
            const float x = h[j] + (p0[j] + p1[j]);
            h[j] = x;
            hs[j] = x;
            acc = __builtin_fmaf(x, x, acc);
        }
    }
    part[t] = acc;
    __syncthreads();
    for (unsigned s = kNormThreads / 2; s >= 1; s >>= 1U) {
        if (t < s) part[t] = part[t] + part[t + s];
        __syncthreads();
    }
    const float rs = 1.0F / __builtin_sqrtf(part[0] / static_cast<float>(kHidden) + kNormEps);
    float* y = v.y + static_cast<std::size_t>(r) * kHidden;
    if constexpr (Ilp == 2) {  // the same expression with the preloaded weights (no load behind a y store)
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            const unsigned j = t + q * kNormThreads;
            const float o = (hs[j] * rs) * (1.0F + nw[q]);
            y[j] = o;
            ys[j] = o;
        }
    } else {
        for (unsigned j = t; j < kHidden; j += kNormThreads) {
            const float o = (hs[j] * rs) * (1.0F + v.norm_w[j]);
            y[j] = o;
            ys[j] = o;
        }
    }
    if (v.prep_jobs != 0) {  // the next linears' input transform (their prepare launch's function, same bits)
        __syncthreads();
        constexpr unsigned kBlocks = kHidden / 128U;
        for (unsigned q = wave; q < v.prep_jobs * kBlocks; q += kWaves) {
            const unsigned i = q / kBlocks, b = q % kBlocks;
            auto* xt = static_cast<_Float16*>(v.prep_xt[i]) + static_cast<std::size_t>(r) * kHidden;
            opt::dev::transform_block(ys, v.prep_suh[i], xt, b * 128U, lane);
        }
    }
    if (v.router_w == nullptr) return;
    __syncthreads();
    for (unsigned e = wave; e < kExperts; e += kWaves) {
        const float* w = v.router_w + static_cast<std::size_t>(e) * kHidden;
        float a = 0.0F;
        for (unsigned j = lane; j < kHidden; j += kRouterLanes) a = __builtin_fmaf(ys[j], w[j], a);
        for (unsigned d = 16; d >= 1; d >>= 1U) a = a + __shfl_xor(a, static_cast<int>(d), 32);
        if (lane == 0) {
            const float s = 1.0F / (1.0F + msa::exp_f(-a));
            sc[e] = s;
            ch[e] = s + v.bias[e];
        }
    }
    __syncthreads();
    if (t == 0) {
        std::uint32_t best[kTopE];
        for (unsigned k = 0; k < kTopE; ++k) {  // k-th best by (choice desc, id asc) among the not yet taken
            std::uint32_t pick = kExperts;
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                bool taken = false;
                for (unsigned q = 0; q < k; ++q) taken |= best[q] == e;
                if (taken) continue;
                if (pick == kExperts || ch[e] > ch[pick]) pick = e;  // ascending scan: equal choice keeps the smaller id
            }
            best[k] = pick;
        }
        float sum = 0.0F;
        for (unsigned k = 0; k < kTopE; ++k) sum = sum + sc[best[k]];
        for (unsigned k = 0; k < kTopE; ++k) {
            v.ids[r * kTopE + k] = best[k];
            v.weights[r * kTopE + k] = (sc[best[k]] / (sum + kSumEps)) * kRouteScale;
        }
    }
}

template <unsigned R>
__global__ __launch_bounds__(kNormThreads) void router_batch_kernel(
    std::uint32_t rows, const float* y, const float* router_w, const float* bias,
    std::uint32_t* ids, float* weights) {
    static_assert(R == 4 || R == 8);
    static_assert(kRouterLanes == 32);
    // The R rows' y are staged through LDS in chunks of kChunk inputs (ascending), each wave keeps the accumulators of
    // its kExperts / kWaves experts x R rows in registers across chunks: per (row, expert) lane L still runs
    // a = fmaf(y[j], w[e][j], a) over j = L, L + 32, ... ascending from 0.0F (kChunk % 32 == 0), so the same chain.
    constexpr unsigned kChunk = 512, kPerWave = kExperts / kWaves;
    static_assert(kHidden % kChunk == 0 && kChunk % kRouterLanes == 0 && kExperts % kWaves == 0);
    __shared__ float ys[R][kChunk];
    __shared__ float sc[R][kExperts], ch[R][kExperts];
    const unsigned t = threadIdx.x, lane = t & 31U, wave = t >> 5U;
    const unsigned first = blockIdx.x * R;
    const unsigned valid = rows - first < R ? rows - first : R;  // >= 1 by the grid
    float a[kPerWave][R];
#pragma unroll
    for (unsigned q = 0; q < kPerWave; ++q)
#pragma unroll
        for (unsigned row = 0; row < R; ++row) a[q][row] = 0.0F;
    for (unsigned c0 = 0; c0 < kHidden; c0 += kChunk) {
        __syncthreads();  // the previous chunk's readers are done
        for (unsigned i = t; i < R * kChunk; i += kNormThreads) {
            const unsigned row = i / kChunk, j = i % kChunk;
            ys[row][j] = row < valid ? y[static_cast<std::size_t>(first + row) * kHidden + c0 + j] : 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (unsigned q = 0; q < kPerWave; ++q) {
            const float* w = router_w + static_cast<std::size_t>(wave + q * kWaves) * kHidden + c0;
            for (unsigned j = lane; j < kChunk; j += kRouterLanes) {
                const float wj = w[j];
#pragma unroll
                for (unsigned row = 0; row < R; ++row) a[q][row] = __builtin_fmaf(ys[row][j], wj, a[q][row]);
            }
        }
    }
#pragma unroll
    for (unsigned q = 0; q < kPerWave; ++q) {
        const unsigned e = wave + q * kWaves;
#pragma unroll
        for (unsigned row = 0; row < R; ++row) {
            float x = a[q][row];
            for (unsigned d = 16; d >= 1; d >>= 1U) x = x + __shfl_xor(x, static_cast<int>(d), 32);
            if (lane == 0 && row < valid) {
                const float s = 1.0F / (1.0F + msa::exp_f(-x));
                sc[row][e] = s;
                ch[row][e] = s + bias[e];
            }
        }
    }
    __syncthreads();
    if (t < R && first + t < rows) {
        const unsigned row = t, r = first + row;
        std::uint32_t best[kTopE];
        for (unsigned k = 0; k < kTopE; ++k) {  // k-th best by (choice desc, id asc) among the not yet taken
            std::uint32_t pick = kExperts;
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                bool taken = false;
                for (unsigned q = 0; q < k; ++q) taken |= best[q] == e;
                if (taken) continue;
                if (pick == kExperts || ch[row][e] > ch[row][pick]) pick = e;  // ascending scan: equal choice keeps the smaller id
            }
            best[k] = pick;
        }
        float sum = 0.0F;
        for (unsigned k = 0; k < kTopE; ++k) sum = sum + sc[row][best[k]];
        for (unsigned k = 0; k < kTopE; ++k) {
            ids[r * kTopE + k] = best[k];
            weights[r * kTopE + k] = (sc[row][best[k]] / (sum + kSumEps)) * kRouteScale;
        }
    }
}

// Prefill router, tiled (enqueue_router_tiled): CTA = kRtRows rows x kRtExperts experts. Per (row, expert) lane L still
// runs a = fmaf(y[j], w[e][j], a) over j = L, L + 32, ... ascending from 0.0F (kRtChunk % 32 == 0), then the same xor
// butterfly, sigmoid and bias: the same bits. The j loop is outside the expert loop, so each staged y element is read
// from LDS once per row for the wave's kRtPerWave experts. blockIdx.x = expert group (fastest), so the groups of one row
// tile run together and share its y rows in cache. Scores go to scratch [2][kExperts][rows] (rows adjacent: coalesced
// selection reads); router_select_kernel runs the selection of router_batch_kernel unchanged.
constexpr unsigned kRtRows = 16, kRtExperts = 32, kRtChunk = 256, kRtPerWave = kRtExperts / kWaves;
static_assert(kRtExperts % kWaves == 0 && kExperts % kRtExperts == 0 && kHidden % kRtChunk == 0 && kRtChunk % kRouterLanes == 0);

__global__ __launch_bounds__(kNormThreads) void router_tiled_kernel(std::uint32_t rows, const float* y, const float* router_w,
                                                                     const float* bias, float* scores) {
    static_assert(kRouterLanes == 32);
    __shared__ float ys[kRtRows][kRtChunk];
    const unsigned t = threadIdx.x, lane = t & 31U, wave = t >> 5U;
    const unsigned e0 = blockIdx.x * kRtExperts, first = blockIdx.y * kRtRows;
    const unsigned valid = rows - first < kRtRows ? rows - first : kRtRows;
    float a[kRtPerWave][kRtRows];
#pragma unroll
    for (unsigned q = 0; q < kRtPerWave; ++q)
#pragma unroll
        for (unsigned row = 0; row < kRtRows; ++row) a[q][row] = 0.0F;
    const float* w[kRtPerWave];
#pragma unroll
    for (unsigned q = 0; q < kRtPerWave; ++q) w[q] = router_w + static_cast<std::size_t>(e0 + wave + q * kWaves) * kHidden;
    for (unsigned c0 = 0; c0 < kHidden; c0 += kRtChunk) {
        __syncthreads();
        for (unsigned i = t; i < kRtRows * kRtChunk; i += kNormThreads) {
            const unsigned row = i / kRtChunk, j = i % kRtChunk;
            ys[row][j] = row < valid ? y[static_cast<std::size_t>(first + row) * kHidden + c0 + j] : 0.0F;
        }
        __syncthreads();
        for (unsigned j = lane; j < kRtChunk; j += kRouterLanes) {
            float wj[kRtPerWave];
#pragma unroll
            for (unsigned q = 0; q < kRtPerWave; ++q) wj[q] = w[q][c0 + j];
#pragma unroll
            for (unsigned row = 0; row < kRtRows; ++row) {
                const float yv = ys[row][j];
#pragma unroll
                for (unsigned q = 0; q < kRtPerWave; ++q) a[q][row] = __builtin_fmaf(yv, wj[q], a[q][row]);
            }
        }
    }
#pragma unroll
    for (unsigned q = 0; q < kRtPerWave; ++q) {
        const unsigned e = e0 + wave + q * kWaves;
#pragma unroll
        for (unsigned row = 0; row < kRtRows; ++row) {
            float x = a[q][row];
            for (unsigned d = 16; d >= 1; d >>= 1U) x = x + __shfl_xor(x, static_cast<int>(d), 32);
            if (lane == 0 && row < valid) {
                const float sv = 1.0F / (1.0F + msa::exp_f(-x));
                scores[static_cast<std::size_t>(e) * rows + first + row] = sv;
                scores[(static_cast<std::size_t>(kExperts) + e) * rows + first + row] = sv + bias[e];
            }
        }
    }
}

// Strided views of one row's scores in the [2][kExperts][rows] scratch.
struct RowScores {
    const float* base;
    std::uint32_t stride;
    __device__ float operator[](std::uint32_t e) const { return base[static_cast<std::size_t>(e) * stride]; }
};

__global__ __launch_bounds__(256) void router_select_kernel(std::uint32_t rows, const float* scores, std::uint32_t* ids,
                                                            float* weights) {
    const unsigned r = blockIdx.x * 256U + threadIdx.x;
    if (r >= rows) return;
    const RowScores sc{scores + r, rows};
    const RowScores ch{scores + static_cast<std::size_t>(kExperts) * rows + r, rows};
    std::uint32_t best[kTopE];
    for (unsigned k = 0; k < kTopE; ++k) {  // k-th best by (choice desc, id asc) among the not yet taken
        std::uint32_t pick = kExperts;
        for (std::uint32_t e = 0; e < kExperts; ++e) {
            bool taken = false;
            for (unsigned q = 0; q < k; ++q) taken |= best[q] == e;
            if (taken) continue;
            if (pick == kExperts || ch[e] > ch[pick]) pick = e;  // ascending scan: equal choice keeps the smaller id
        }
        best[k] = pick;
    }
    float sum = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k) sum = sum + sc[best[k]];
    for (unsigned k = 0; k < kTopE; ++k) {
        ids[r * kTopE + k] = best[k];
        weights[r * kTopE + k] = (sc[best[k]] / (sum + kSumEps)) * kRouteScale;
    }
}

}  // namespace

void enqueue_post_exchange(const PostExchangeView& v, hipStream_t stream) {
    if (v.ilp) {  // the loads-first form needs h disjoint from both partials (it reads them all before writing h)
        const auto* hb = reinterpret_cast<const char*>(v.h);
        const auto* he = hb + std::size_t{v.rows} * kHidden * sizeof(float);
        for (const float* p : {v.part0, v.part1}) {
            const auto* pb = reinterpret_cast<const char*>(p);
            const auto* pe = pb + std::size_t{v.rows} * kHidden * sizeof(float);
            if (pb < he && hb < pe) throw std::invalid_argument("minimax post-exchange: ilp form with h overlapping a partial");
        }
    }
    if (v.ilp == 2 && ((reinterpret_cast<std::uintptr_t>(v.part0) | reinterpret_cast<std::uintptr_t>(v.part1)) & 15U) != 0)
        throw std::invalid_argument("minimax post-exchange: ilp mode 2 needs 16-byte aligned partials");
    if (v.ilp == 2) hipLaunchKernelGGL(post_exchange_kernel<2>, dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    else if (v.ilp == 1) hipLaunchKernelGGL(post_exchange_kernel<1>, dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    else hipLaunchKernelGGL(post_exchange_kernel<0>, dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax post-exchange: ") + hipGetErrorString(e));
}

void enqueue_router_batch(std::uint32_t rows, const float* y, const float* router_w, const float* bias,
                          std::uint32_t* ids, float* weights, unsigned batch_rows, hipStream_t stream) {
    if (batch_rows != 4 && batch_rows != 8)
        throw std::invalid_argument("minimax router batch: rows must be 4 or 8");
    if (rows == 0) return;
    const dim3 grid(rows / batch_rows + (rows % batch_rows != 0));
    if (batch_rows == 4)
        hipLaunchKernelGGL(router_batch_kernel<4>, grid, dim3(kNormThreads), 0, stream, rows, y, router_w, bias, ids, weights);
    else
        hipLaunchKernelGGL(router_batch_kernel<8>, grid, dim3(kNormThreads), 0, stream, rows, y, router_w, bias, ids, weights);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax router batch: ") + hipGetErrorString(e));
}

void enqueue_router_tiled(std::uint32_t rows, const float* y, const float* router_w, const float* bias,
                          std::uint32_t* ids, float* weights, float* scratch, hipStream_t stream) {
    if (rows == 0) return;
    if (scratch == nullptr) throw std::invalid_argument("minimax router tiled: scratch is required");
    const dim3 grid(kExperts / kRtExperts, rows / kRtRows + (rows % kRtRows != 0));
    hipLaunchKernelGGL(router_tiled_kernel, grid, dim3(kNormThreads), 0, stream, rows, y, router_w, bias, scratch);
    hipLaunchKernelGGL(router_select_kernel, dim3((rows + 255) / 256), dim3(256), 0, stream, rows, scratch, ids, weights);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax router tiled: ") + hipGetErrorString(e));
}

}  // namespace strix::minimax::moe::hip
