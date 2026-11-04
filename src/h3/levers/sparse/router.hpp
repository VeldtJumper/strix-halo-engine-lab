#pragma once
// Device router kernels of the sparse-attention lever (identity h3-attn-sparse-thr-v1): bit-for-bit mirrors of
// router_ref.hpp (same per-element operation order, contraction off, correctly rounded divide / sqrt). Layout and
// global indices as in router_ref.hpp. All launches on the caller's stream; no host synchronisation except the
// LUT size read in build_lut (it needs the total entry count to size the entry buffer).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/levers/sparse/router_ref.hpp"

namespace strix::h3::levers::sparse {

namespace dev {
// kc (bf16) or qbar (fp32) of tiles j0 .. j0 + gridDim.x - 1. One work-group per tile covering all 56 x 128 elements
// of a row: 224 threads x 4 chunks of 8 consecutive elements (16-byte loads, coalesced rows), 32 independent fp32
// chains per thread; each element is still ONE chain over the tile's rows in ascending order (== the host).
constexpr std::uint32_t kCenThreads = 224U, kCenChunks = 4U;  // 224 * 4 * 8 = 7168 = 56 * 128
template <bool kBf16Out>
__global__ __launch_bounds__(224) void centroid_kernel(const std::uint16_t* __restrict__ x, std::uint64_t ld, std::uint32_t S, std::uint32_t j0,
    void* __restrict__ out) {
#pragma clang fp contract(off)
    static_assert(kCenThreads * kCenChunks * 8U == kHeads * kDim);
    const std::uint32_t jl = blockIdx.x, j = j0 + jl, t = threadIdx.x;
    const std::uint32_t n = S - j * kTile < kTile ? S - j * kTile : kTile;
    float acc[kCenChunks][8] = {};
    const std::uint16_t* base = x + static_cast<std::uint64_t>(jl) * kTile * ld;
#pragma unroll 2
    for (std::uint32_t r = 0; r < n; ++r) {
        const std::uint16_t* xr = base + static_cast<std::uint64_t>(r) * ld;
        uint4 v[kCenChunks];
#pragma unroll
        for (std::uint32_t c = 0; c < kCenChunks; ++c) v[c] = *reinterpret_cast<const uint4*>(xr + 8U * (t + kCenThreads * c));
#pragma unroll
        for (std::uint32_t c = 0; c < kCenChunks; ++c) {
            const std::uint32_t w[4] = {v[c].x, v[c].y, v[c].z, v[c].w};
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i) {
                acc[c][2U * i] = acc[c][2U * i] + gemm::bf16_to_f32(static_cast<std::uint16_t>(w[i] & 0xffffU));
                acc[c][2U * i + 1U] = acc[c][2U * i + 1U] + gemm::bf16_to_f32(static_cast<std::uint16_t>(w[i] >> 16U));
            }
        }
    }
    const float fn = static_cast<float>(n);
#pragma unroll
    for (std::uint32_t c = 0; c < kCenChunks; ++c) {
        const std::uint64_t o = static_cast<std::uint64_t>(jl) * kHeads * kDim + 8U * (t + kCenThreads * c);
        if constexpr (kBf16Out) {
            std::uint32_t pk[4];
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i)
                pk[i] = gemm::f32_to_bf16(acc[c][2U * i] / fn) | (static_cast<std::uint32_t>(gemm::f32_to_bf16(acc[c][2U * i + 1U] / fn)) << 16U);
            *reinterpret_cast<uint4*>(static_cast<std::uint16_t*>(out) + o) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        } else {
            float m[8];
#pragma unroll
            for (std::uint32_t i = 0; i < 8U; ++i) m[i] = acc[c][i] / fn;
            *reinterpret_cast<float4*>(static_cast<float*>(out) + o) = make_float4(m[0], m[1], m[2], m[3]);
            *reinterpret_cast<float4*>(static_cast<float*>(out) + o + 4U) = make_float4(m[4], m[5], m[6], m[7]);
        }
    }
}
// S1, S2 of `tiles` bf16 centroids; one thread per (head, dim).
template <int = 0>
__global__ __launch_bounds__(128) void stats_partial_kernel(const std::uint16_t* __restrict__ kc, std::uint32_t tiles, float* __restrict__ s1,
    float* __restrict__ s2) {
#pragma clang fp contract(off)
    const std::uint32_t e = blockIdx.x * kDim + threadIdx.x;
    float a = 0.0F, b = 0.0F;
    for (std::uint32_t t = 0; t < tiles; ++t) {
        const float v = gemm::bf16_to_f32(kc[static_cast<std::uint64_t>(t) * kHeads * kDim + e]);
        const float sq = v * v;
        a = a + v;
        b = b + sq;
    }
    s1[e] = a;
    s2[e] = b;
}
// mean / var from the two partials in block order (p0 then p1; n1 may be 0 with null p1).
template <int = 0>
__global__ __launch_bounds__(128) void stats_finalize_kernel(const float* __restrict__ a1, const float* __restrict__ a2, const float* __restrict__ b1,
    const float* __restrict__ b2, std::uint32_t nt, float* __restrict__ mean, float* __restrict__ var) {
#pragma clang fp contract(off)
    const std::uint32_t e = blockIdx.x * kDim + threadIdx.x;
    float s1 = 0.0F, s2 = 0.0F;
    s1 = s1 + a1[e];
    s2 = s2 + a2[e];
    if (b1) {
        s1 = s1 + b1[e];
        s2 = s2 + b2[e];
    }
    const float fn = static_cast<float>(nt);
    const float mu = s1 / fn, e2 = s2 / fn, mu2 = mu * mu, v = e2 - mu2;
    mean[e] = mu;
    var[e] = v > 0.0F ? v : 0.0F;
}
// thr[i][h] for `blocks` query centroids; one thread per (block, head).
template <int = 0>
__global__ __launch_bounds__(64) void threshold_kernel(const float* __restrict__ qbar, std::uint32_t blocks, const float* __restrict__ mean,
    const float* __restrict__ var, float c, float c2, float tau, float* __restrict__ thr) {
#pragma clang fp contract(off)
    const std::uint32_t k = blockIdx.x * 64U + threadIdx.x;
    if (k >= blocks * kHeads) return;
    const std::uint32_t h = k % kHeads;
    const float* qv = qbar + static_cast<std::uint64_t>(k) * kDim;
    float dot = 0.0F, vs = 0.0F;
    for (std::uint32_t d = 0; d < kDim; ++d) {
        const float p = qv[d] * mean[h * kDim + d];
        dot = dot + p;
        const float q2 = qv[d] * qv[d];
        const float pv = q2 * var[h * kDim + d];
        vs = vs + pv;
    }
    const float mu = dot * c, v = vs * c2;
    const float sd = __builtin_sqrtf((v > 0.0F ? v : 0.0F) + 1e-6F);
    const float ts = tau * sd;
    thr[k] = mu + ts;
}
// LUT build pass 1 (one work-group = 8 waves per (query tile t, head)): selection over key tiles j0 .. j1-1 in chunks
// of 256; each wave stores its 32-key ballot as one mask word (word w of the list = keys j0 + 32 w ..), the list
// count and the telemetry come from popcounts. The selection arithmetic per (i, j) is the host one: dot over d
// ascending (product rounded, then added), times c, strictly greater than the threshold; band / sink / dense tile too.
template <int = 0>
__global__ __launch_bounds__(256) void select_kernel(std::uint32_t KT, std::uint32_t P, std::uint32_t band, std::uint32_t t0, const float* __restrict__ qbar,
    const float* __restrict__ thr, std::uint32_t j0, std::uint32_t j1, const std::uint16_t* __restrict__ kc, float c, std::uint32_t words,
    std::uint32_t* __restrict__ mask, std::uint32_t* __restrict__ counts, unsigned long long* __restrict__ tele) {
#pragma clang fp contract(off)
    __shared__ float qs[2][kDim];
    __shared__ float ts[2];
    __shared__ std::uint32_t cnt[3];
    const std::uint32_t t = t0 + blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, list = blockIdx.x * kHeads + h;
    const std::uint32_t i_base = 2U * t0;
    const bool dense_tile = kQTile * t < P;
    if (tid < 3U) cnt[tid] = 0U;
    for (std::uint32_t e = tid; e < 2U * kDim; e += 256U) {
        const std::uint32_t b = e / kDim, i = 2U * t + b;
        qs[b][e % kDim] = i < KT ? qbar[(static_cast<std::uint64_t>(i - i_base) * kHeads + h) * kDim + e % kDim] : 0.0F;
    }
    if (tid < 2U) ts[tid] = 2U * t + tid < KT ? thr[static_cast<std::uint64_t>(2U * t + tid - i_base) * kHeads + h] : 0.0F;
    __syncthreads();
    std::uint32_t my_sel = 0, my_a = 0, my_b = 0;
    for (std::uint32_t c0 = j0; c0 < j1; c0 += 256U) {
        const std::uint32_t j = c0 + tid;
        bool sel = false, ha = false, hb = false;
        if (j < j1) {
            if (dense_tile) {
                sel = true;
            } else {
                bool hit[2] = {false, false};
                const uint4* kv = reinterpret_cast<const uint4*>(kc + (static_cast<std::uint64_t>(j - j0) * kHeads + h) * kDim);
                for (std::uint32_t b = 0; b < 2U; ++b) {
                    const std::uint32_t i = 2U * t + b;
                    if (i >= KT) continue;
                    const std::uint32_t di = i > j ? i - j : j - i;
                    if (di <= band || kTile * j < P) {
                        hit[b] = true;
                        continue;
                    }
                    float dot = 0.0F;
                    for (std::uint32_t q = 0; q < kDim / 8U; ++q) {
                        const uint4 v = kv[q];
                        const std::uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                        for (std::uint32_t k = 0; k < 4U; ++k) {
                            const float p0 = qs[b][8U * q + 2U * k] * gemm::bf16_to_f32(static_cast<std::uint16_t>(w[k] & 0xffffU));
                            dot = dot + p0;
                            const float p1 = qs[b][8U * q + 2U * k + 1U] * gemm::bf16_to_f32(static_cast<std::uint16_t>(w[k] >> 16U));
                            dot = dot + p1;
                        }
                    }
                    hit[b] = dot * c > ts[b];
                }
                ha = hit[0];
                hb = hit[1];
                sel = ha || hb;
            }
        }
        const std::uint32_t ms = __builtin_amdgcn_ballot_w32(sel), ma = __builtin_amdgcn_ballot_w32(ha), mb = __builtin_amdgcn_ballot_w32(hb);
        const std::uint32_t word = (c0 - j0) / 32U + wave;
        if (lane == 0U && word < words) mask[static_cast<std::uint64_t>(list) * words + word] = ms;
        if (lane == 0U) {
            my_sel += static_cast<std::uint32_t>(__builtin_popcount(ms));
            my_a += static_cast<std::uint32_t>(__builtin_popcount(ma));
            my_b += static_cast<std::uint32_t>(__builtin_popcount(mb));
        }
    }
    if (lane == 0U) {
        atomicAdd(&cnt[0], my_sel);
        atomicAdd(&cnt[1], my_a);
        atomicAdd(&cnt[2], my_b);
    }
    __syncthreads();
    if (tid == 0U) {
        counts[list] = cnt[0];
        if (!dense_tile) {
            atomicAdd(&tele[0], static_cast<unsigned long long>(cnt[0]));
            atomicAdd(&tele[1], static_cast<unsigned long long>(cnt[1] + cnt[2]));
        } else {
            atomicAdd(&tele[2], 1ULL);
        }
    }
}
// LUT build pass 2: one thread per mask word; its position = list offset + popcounts of the earlier words of the list;
// set bits are written in ascending key order (the same entries as a serial compaction).
template <int = 0>
__global__ __launch_bounds__(256) void compact_kernel(std::uint32_t lists, std::uint32_t words, std::uint32_t j0, const std::uint32_t* __restrict__ mask,
    const std::uint32_t* __restrict__ offsets, std::int32_t* __restrict__ entries) {
    const std::uint64_t g = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
    if (g >= static_cast<std::uint64_t>(lists) * words) return;
    const std::uint32_t list = static_cast<std::uint32_t>(g / words), w = static_cast<std::uint32_t>(g % words);
    const std::uint32_t* m = mask + static_cast<std::uint64_t>(list) * words;
    std::uint32_t pos = offsets[list];
    for (std::uint32_t k = 0; k < w; ++k) pos += static_cast<std::uint32_t>(__builtin_popcount(m[k]));
    std::uint32_t bits = m[w];
    while (bits) {
        const std::uint32_t b = static_cast<std::uint32_t>(__builtin_ctz(bits));
        entries[pos++] = static_cast<std::int32_t>(j0 + 32U * w + b);
        bits &= bits - 1U;
    }
}

class Buf {
public:
    Buf() = default;
    explicit Buf(std::size_t bytes) {
        if (bytes && hipMalloc(&p_, bytes) != hipSuccess) {
            p_ = nullptr;
            static_cast<void>(hipGetLastError());
            throw std::runtime_error("h3 sparse: hipMalloc failed");
        }
    }
    Buf(Buf&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) {
            if (p_) static_cast<void>(hipFree(p_));
            p_ = o.p_;
            o.p_ = nullptr;
        }
        return *this;
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() {
        if (p_) static_cast<void>(hipFree(p_));
    }
    template <class T> T* as() const { return static_cast<T*>(p_); }

private:
    void* p_ = nullptr;
};
inline void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 sparse: ") + what + ": " + hipGetErrorString(e));
}
}  // namespace dev

// Launchers (pointers: x / q / k at the row of global index 64 * first tile).
inline void launch_key_centroids(const std::uint16_t* k, std::uint64_t ld, const Geometry& g, std::uint32_t j0, std::uint32_t j1, std::uint16_t* kc,
    hipStream_t s) {
    if (j0 >= j1 || j1 > g.KT) throw std::out_of_range("h3 sparse: key tile range");
    if (!k || !kc || ld % 8U || ld < kHeads * kDim || (reinterpret_cast<std::uintptr_t>(k) & 15U) || (reinterpret_cast<std::uintptr_t>(kc) & 15U))
        throw std::invalid_argument("h3 sparse: centroid layout (16-byte aligned, ld % 8 == 0)");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::centroid_kernel<true>), dim3(j1 - j0), dim3(dev::kCenThreads), 0, s, k, ld, g.S, j0, static_cast<void*>(kc));
}
inline void launch_query_centroids(const std::uint16_t* q, std::uint64_t ld, const Geometry& g, std::uint32_t i0, std::uint32_t i1, float* qbar,
    hipStream_t s) {
    if (i0 >= i1 || i1 > g.KT) throw std::out_of_range("h3 sparse: query block range");
    if (!q || !qbar || ld % 8U || ld < kHeads * kDim || (reinterpret_cast<std::uintptr_t>(q) & 15U) || (reinterpret_cast<std::uintptr_t>(qbar) & 15U))
        throw std::invalid_argument("h3 sparse: centroid layout (16-byte aligned, ld % 8 == 0)");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::centroid_kernel<false>), dim3(i1 - i0), dim3(dev::kCenThreads), 0, s, q, ld, g.S, i0, static_cast<void*>(qbar));
}
inline void launch_stats_partial(const std::uint16_t* kc, std::uint32_t tiles, float* s1, float* s2, hipStream_t s) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::stats_partial_kernel<0>), dim3(kHeads), dim3(kDim), 0, s, kc, tiles, s1, s2);
}
inline void launch_stats_finalize(const float* a1, const float* a2, const float* b1, const float* b2, std::uint32_t nt, float* mean, float* var,
    hipStream_t s) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::stats_finalize_kernel<0>), dim3(kHeads), dim3(kDim), 0, s, a1, a2, b1, b2, nt, mean, var);
}
inline void launch_thresholds(const float* qbar, std::uint32_t blocks, const float* mean, const float* var, float tau, float* thr, hipStream_t s) {
    const float c = score_scale(), c2 = c * c;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::threshold_kernel<0>), dim3((blocks * kHeads + 63U) / 64U), dim3(64), 0, s, qbar, blocks, mean, var, c, c2, tau, thr);
}

// Device LUT of query tiles t0 .. t1-1 against key tiles j0 .. j1-1 (qbar / thr of blocks 2 t0 .., kc of tiles j0 ..).
struct DeviceLut {
    std::uint32_t t0 = 0, t1 = 0, j0 = 0, j1 = 0, entries_n = 0;
    dev::Buf offsets, entries;  // uint32 [(t1 - t0) * 56 + 1], int32 [entries_n]
    unsigned long long union_sum = 0, parts_sum = 0, dense_tiles = 0;
};
inline DeviceLut build_lut(const Geometry& g, const Params& p, std::uint32_t t0, std::uint32_t t1, const float* qbar, const float* thr, std::uint32_t j0,
    std::uint32_t j1, const std::uint16_t* kc, hipStream_t s) {
    if (g.dense) throw std::logic_error("h3 sparse: route on a dense geometry");
    if (t0 >= t1 || t1 > g.QT || j0 >= j1 || j1 > g.KT) throw std::out_of_range("h3 sparse: route ranges");
    DeviceLut L;
    L.t0 = t0;
    L.t1 = t1;
    L.j0 = j0;
    L.j1 = j1;
    const std::uint32_t lists = (t1 - t0) * kHeads, words = (j1 - j0 + 31U) / 32U;
    if (reinterpret_cast<std::uintptr_t>(kc) & 15U) throw std::invalid_argument("h3 sparse: kc must be 16-byte aligned");
    dev::Buf counts(lists * 4U), tele(3U * 8U), mask(static_cast<std::size_t>(lists) * words * 4U);
    L.offsets = dev::Buf((lists + 1U) * 4U);
    dev::check(hipMemsetAsync(tele.as<void>(), 0, 24, s), "memset");
    dev::check(hipMemsetAsync(mask.as<void>(), 0, static_cast<std::size_t>(lists) * words * 4U, s), "memset");
    const float c = score_scale();
    static_cast<void>(hipGetLastError());
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::select_kernel<0>), dim3(t1 - t0, kHeads), dim3(256), 0, s, g.KT, g.P, p.band, t0, qbar, thr, j0, j1, kc, c,
        words, mask.as<std::uint32_t>(), counts.as<std::uint32_t>(), tele.as<unsigned long long>());
    dev::check(hipGetLastError(), "route select");
    std::vector<std::uint32_t> cnt(lists), off(lists + 1U, 0U);
    unsigned long long tv[3];
    dev::check(hipMemcpyAsync(cnt.data(), counts.as<void>(), lists * 4U, hipMemcpyDeviceToHost, s), "counts");
    dev::check(hipMemcpyAsync(tv, tele.as<void>(), 24, hipMemcpyDeviceToHost, s), "tele");
    dev::check(hipStreamSynchronize(s), "sync");
    for (std::uint32_t l = 0; l < lists; ++l) off[l + 1U] = off[l] + cnt[l];
    L.entries_n = off[lists];
    L.union_sum = tv[0];
    L.parts_sum = tv[1];
    L.dense_tiles = tv[2] / kHeads;
    dev::check(hipMemcpyAsync(L.offsets.as<void>(), off.data(), off.size() * 4U, hipMemcpyHostToDevice, s), "offsets");
    L.entries = dev::Buf(static_cast<std::size_t>(L.entries_n) * 4U);
    if (L.entries_n) {
        const std::uint64_t total = static_cast<std::uint64_t>(lists) * words;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::compact_kernel<0>), dim3(static_cast<std::uint32_t>((total + 255U) / 256U)), dim3(256), 0, s, lists, words, j0,
            mask.as<std::uint32_t>(), L.offsets.as<std::uint32_t>(), L.entries.as<std::int32_t>());
    }
    dev::check(hipGetLastError(), "route compact");
    dev::check(hipStreamSynchronize(s), "sync");  // the host offsets vector and the mask must outlive the async work
    return L;
}

}  // namespace strix::h3::levers::sparse
