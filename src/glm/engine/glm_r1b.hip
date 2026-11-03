// R1b generated identity (default; default prompts use R1) (approved arithmetic contract): device kernels of the blocked chains (glm_math.hpp
// r1b_dot). Scope and allowlist: glm_r1b.hpp. Bitwise equal to the host emulation by construction (same fmaf
// sequence per block, same tree, same R6 / collapse code as dev::hc_pre), gated by tests/hip_glm_r1b_tests.hip.
// Build flags as the Tier A kernels (-ffp-contract=off -fno-fast-math, denormals kept).
#include "glm/engine/glm_r1b.hpp"

#include "glm/common/glm_math.hpp"

#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::glm::r1b {
namespace {

std::atomic<bool> g_active{false};

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glm r1b ") + what + ": " + hipGetErrorString(e));
}

constexpr unsigned kMaxRows = 8;

// ---------------------------------------------------------------- K = 4096 F16 dense (router, KDA f_a / g_a / b)
// Launch 1: workgroup = (block j of 256 inputs, group of 64 outputs of one job); the job's x slice [rows][256] is staged in
// LDS; thread = one output: ascending fmaf chain from +0 over the block's 256 inputs per row (8 F16 per 16-byte load,
// order unchanged) -> partial [job][r][o][j]. Launch 2: thread = (job, r, o): tree over the 16 partials at offsets
// 8, 4, 2, 1 (glm_math r1b_dot). Up to 3 jobs per call (KDA f_a, g_a, b share one call).
constexpr unsigned kBlocks4k = 4096 / kR1bBlock;  // 16
constexpr unsigned kOutWG = 64;
constexpr unsigned kMaxJobs = 3;
static_assert((kBlocks4k & (kBlocks4k - 1)) == 0, "power-of-two tree");
struct DJobs {
    const float* x[kMaxJobs];
    const std::uint16_t* W[kMaxJobs];
    float* y[kMaxJobs];
    unsigned n[kMaxJobs], ldx[kMaxJobs], ldy[kMaxJobs];
    unsigned g0[kMaxJobs + 1];    // first output group of each job (prefix), g0[nj] = total groups
    unsigned off[kMaxJobs + 1];   // first (r, o) slot of each job in the partial / tree index space
    unsigned nj, rows;
};
__device__ float g_dpart[kMaxRows * 512 * kBlocks4k];  // [slot][16], slot = off[job] + r * n + o

__global__ __launch_bounds__(kOutWG) void dense_k4096_part_kernel(DJobs J) {
    __shared__ float xs[kMaxRows][kR1bBlock + 1];
    const unsigned j = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    unsigned q = 0;
    while (q + 1 < J.nj && g >= J.g0[q + 1]) ++q;
    const unsigned o = (g - J.g0[q]) * kOutWG + t, rows = J.rows;
    for (unsigned idx = t; idx < rows * kR1bBlock; idx += kOutWG) {
        const unsigned r = idx / kR1bBlock, i = idx % kR1bBlock;
        xs[r][i] = J.x[q][static_cast<std::size_t>(r) * J.ldx[q] + j * kR1bBlock + i];
    }
    __syncthreads();
    if (o >= J.n[q]) return;
    const std::uint16_t* w = J.W[q] + static_cast<std::size_t>(o) * 4096 + j * kR1bBlock;
    float acc[kMaxRows];
    for (unsigned r = 0; r < kMaxRows; ++r) acc[r] = 0.0F;
#pragma unroll 4
    for (unsigned c = 0; c < kR1bBlock; c += 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(w + c);
        const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(&v);
        for (unsigned k = 0; k < 8; ++k) {
            const float wf = f16_to_float(h[k]);
            for (unsigned r = 0; r < kMaxRows; ++r)
                if (r < rows) acc[r] = __builtin_fmaf(xs[r][c + k], wf, acc[r]);
        }
    }
    for (unsigned r = 0; r < rows; ++r) g_dpart[(static_cast<std::size_t>(J.off[q]) + r * J.n[q] + o) * kBlocks4k + j] = acc[r];
}
__global__ __launch_bounds__(256) void dense_k4096_tree_kernel(DJobs J) {
    const unsigned s = blockIdx.x * 256 + threadIdx.x;
    if (s >= J.off[J.nj]) return;
    unsigned q = 0;
    while (q + 1 < J.nj && s >= J.off[q + 1]) ++q;
    const unsigned rel = s - J.off[q], r = rel / J.n[q], o = rel % J.n[q];
    float p[kBlocks4k];
    for (unsigned i = 0; i < kBlocks4k; ++i) p[i] = g_dpart[static_cast<std::size_t>(s) * kBlocks4k + i];
    for (unsigned hh = kBlocks4k / 2; hh >= 1; hh /= 2)
        for (unsigned i = 0; i < hh; ++i) p[i] = p[i] + p[i + hh];
    J.y[q][static_cast<std::size_t>(r) * J.ldy[q] + o] = p[0];
}
// Decode-only variant: a width-16 subgroup owns one output's exact block tree.
// Inactive tail outputs still execute every shuffle; only valid outputs load weights/store.
template <bool Packed>
__global__ __launch_bounds__(128) void dense_k4096_v4_kernel(DJobs J) {
    __shared__ float xs[kBlocks4k][kR1bBlock + 1];
    const unsigned t = threadIdx.x, g = blockIdx.x, j = t % kBlocks4k;
    unsigned q = 0;
    while (q + 1 < J.nj && g >= J.g0[q + 1]) ++q;
    const unsigned o = (g - J.g0[q]) * 8 + t / kBlocks4k;
    for (unsigned idx = t; idx < 4096; idx += 128)
        xs[idx / kR1bBlock][idx % kR1bBlock] = J.x[q][idx];
    __syncthreads();
    float acc = 0.0F;
    if (o < J.n[q]) {
        const std::uint16_t* w = J.W[q] + static_cast<std::size_t>(o) * 4096 + j * kR1bBlock;
#pragma unroll 4
        for (unsigned c = 0; c < kR1bBlock; c += 8) {
            const auto* src = Packed ? J.W[q] + ((static_cast<std::size_t>(c / 8) * J.n[q] + o) * kBlocks4k + j) * 8 : w + c;
            const uint4 v = *reinterpret_cast<const uint4*>(src);
            const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(&v);
            for (unsigned k = 0; k < 8; ++k)
                acc = __builtin_fmaf(xs[j][c + k], f16_to_float(h[k]), acc);
        }
    }
    for (unsigned h = 8; h > 0; h >>= 1U) {
        const float peer = __shfl_down(acc, h, 16);
        if (j < h) acc = acc + peer;
    }
    if (j == 0 && o < J.n[q]) J.y[q][o] = acc;
}
// Packed multi-row variant: one lane owns ONE output's 256-input block,
// not eight outputs. At R=8 only eight independent accumulators are live.
// Read x through the vector cache: all R full rows in LDS would cost 128 KiB;
// per-c8 staging would instead require 32 workgroup barriers.
// Bitwise construction: for each (r,o,j), acc[r] starts at +0 and receives
// fmaf(x[r][j*256+c+k], W[o][j*256+c+k], acc[r]) in ascending c+k order.
// Packing only permutes weight bits; sharing each uint4 across rows introduces
// no arithmetic between rows. The width-16 8,4,2,1 tree is exactly v4/v5's
// (and v2/host's) fixed 16-block tree. Tail lanes execute every shuffle.
template <unsigned R>
__global__ __launch_bounds__(128) void dense_k4096_v6_kernel(DJobs J) {
    const unsigned t = threadIdx.x, g = blockIdx.x, j = t % kBlocks4k;
    unsigned q = 0;
    while (q + 1 < J.nj && g >= J.g0[q + 1]) ++q;
    const unsigned o = (g - J.g0[q]) * 8 + t / kBlocks4k;
    float acc[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) acc[r] = 0.0F;
    if (o < J.n[q]) {
        // Keep only one weight vector/chunk in flight at the source level.
#pragma unroll 1
        for (unsigned c = 0; c < kR1bBlock; c += 8) {
            const auto* src = J.W[q] + ((static_cast<std::size_t>(c / 8) * J.n[q] + o) * kBlocks4k + j) * 8;
            const uint4 v = *reinterpret_cast<const uint4*>(src);
            const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(&v);
#pragma unroll
            for (unsigned k = 0; k < 8; ++k) {
                const float wf = f16_to_float(h[k]);
#pragma unroll
                for (unsigned r = 0; r < R; ++r)
                    acc[r] = __builtin_fmaf(J.x[q][static_cast<std::size_t>(r) * J.ldx[q] + j * kR1bBlock + c + k],
                                             wf, acc[r]);
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        for (unsigned h = 8; h > 0; h >>= 1U) {
            const float peer = __shfl_down(acc[r], h, 16);
            if (j < h) acc[r] = acc[r] + peer;
        }
        if (j == 0 && o < J.n[q]) J.y[q][static_cast<std::size_t>(r) * J.ldy[q] + o] = acc[r];
    }
}
void dense_jobs(DJobs J, hipStream_t s, int variant) {
    if (variant < 0) variant = dense_variant();
    if (variant == 7) variant = J.rows == 1 ? 5 : 6;
    if (variant == 5 && J.rows != 1) throw std::invalid_argument("r1b dense v5: packed weights require rows == 1");
    const unsigned outputs_per_group = variant == 6 || ((variant == 4 || variant == 5) && J.rows == 1) ? 8U : kOutWG;
    if (J.rows == 0 || J.rows > kMaxRows) throw std::invalid_argument("r1b dense_f16_k4096: rows 1..8");
    J.g0[0] = 0;
    J.off[0] = 0;
    for (unsigned q = 0; q < J.nj; ++q) {
        if (J.n[q] == 0) throw std::invalid_argument("r1b dense_f16_k4096: n >= 1");
        J.g0[q + 1] = J.g0[q] + (J.n[q] + outputs_per_group - 1) / outputs_per_group;
        J.off[q + 1] = J.off[q] + J.rows * J.n[q];
    }
    if (J.off[J.nj] > kMaxRows * 512) throw std::invalid_argument("r1b dense_f16_k4096: more than 4096 row outputs per call");
    if (variant == 6) {
        switch (J.rows) {
            case 1: dense_k4096_v6_kernel<1><<<J.g0[J.nj], 128, 0, s>>>(J); break;
            case 2: dense_k4096_v6_kernel<2><<<J.g0[J.nj], 128, 0, s>>>(J); break;
            case 3: dense_k4096_v6_kernel<3><<<J.g0[J.nj], 128, 0, s>>>(J); break;
            case 4: dense_k4096_v6_kernel<4><<<J.g0[J.nj], 128, 0, s>>>(J); break;
            case 5: dense_k4096_v6_kernel<5><<<J.g0[J.nj], 128, 0, s>>>(J); break;
            case 6: dense_k4096_v6_kernel<6><<<J.g0[J.nj], 128, 0, s>>>(J); break;
            case 7: dense_k4096_v6_kernel<7><<<J.g0[J.nj], 128, 0, s>>>(J); break;
            case 8: dense_k4096_v6_kernel<8><<<J.g0[J.nj], 128, 0, s>>>(J); break;
        }
        ok(hipGetLastError(), "dense_k4096_v6");
        return;
    }
    if ((variant == 4 || variant == 5) && J.rows == 1) {
        if (variant == 5) dense_k4096_v4_kernel<true><<<J.g0[J.nj], 128, 0, s>>>(J);
        else dense_k4096_v4_kernel<false><<<J.g0[J.nj], 128, 0, s>>>(J);
        ok(hipGetLastError(), variant == 5 ? "dense_k4096_v5" : "dense_k4096_v4");
        return;
    }
    dense_k4096_part_kernel<<<dim3(kBlocks4k, J.g0[J.nj]), kOutWG, 0, s>>>(J);
    ok(hipGetLastError(), "dense_k4096_part");
    dense_k4096_tree_kernel<<<(J.off[J.nj] + 255) / 256, 256, 0, s>>>(J);
    ok(hipGetLastError(), "dense_k4096_tree");
}

// Lossless permutation: destination vectors [c8][output][block_j][8].
__global__ void pack_k4096_kernel(const std::uint16_t* W, unsigned n, std::uint16_t* packed) {
    const std::size_t v = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (v >= static_cast<std::size_t>(n) * 512) return;
    const unsigned j = static_cast<unsigned>(v % 16), o = static_cast<unsigned>((v / 16) % n),
                   c8 = static_cast<unsigned>(v / (static_cast<std::size_t>(16) * n));
    *reinterpret_cast<uint4*>(packed + v * 8) =
        *reinterpret_cast<const uint4*>(W + static_cast<std::size_t>(o) * 4096 + j * 256 + c8 * 8);
}

// ---------------------------------------------------------------- mHC hc_pre (24 x 16,384 F32 chains)
constexpr unsigned kN = kHc * kHidden;          // 16384
constexpr unsigned kHcBlocks = kN / kR1bBlock;  // 64
constexpr unsigned kPad = kR1bBlock + 1;        // LDS row stride (bank spread across chains / rows)
static_assert(kHcBlocks == kR1bMaxBlocks && (kHcBlocks & (kHcBlocks - 1)) == 0, "64-block tree");

__device__ float g_rr[kMaxRows];
__device__ float g_part[kMaxRows * kHcMix * kHcBlocks];

// R2 tree in LDS over L lanes (identical to glm_tier_a.hip lds_tree).
template <unsigned L> __device__ __forceinline__ float lds_tree(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// Launch 1: per row, the R2/R3 sum of squares (256 lane-strided fmaf partials + tree) -> rr = 1 / sqrt(ss / n + eps).
__global__ __launch_bounds__(256) void hc_rr_kernel(const float* X) {
    __shared__ float s[256];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * kN;
    float a = 0.0F;
    for (unsigned i = l; i < kN; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = lds_tree<256>(s, a);
    if (l == 0) g_rr[r] = inv_sqrt(ss / static_cast<float>(kN) + kEps);
}

// Launch 2: workgroup j = block j of 256 inputs. F = X * rr (one rounding, as dev::hc_pre), written out; thread
// (row r, chain o) runs the ascending fmaf chain of block j from +0.
constexpr unsigned kPartThreads = kHcMix * kMaxRows;  // 192
template <unsigned Rows>
__global__ __launch_bounds__(kPartThreads) void hc_part_kernel(const float* X, const float* fn, float* F, unsigned rows) {
    __shared__ float fs[kHcMix][kPad];
    __shared__ float xs[Rows][kPad];
    const unsigned j = blockIdx.x, t = threadIdx.x, i0 = j * kR1bBlock;
    for (unsigned idx = t; idx < kHcMix * kR1bBlock; idx += kPartThreads) {
        const unsigned o = idx / kR1bBlock, i = idx % kR1bBlock;
        fs[o][i] = fn[static_cast<std::size_t>(o) * kN + i0 + i];
    }
    for (unsigned idx = t; idx < rows * kR1bBlock; idx += kPartThreads) {
        const unsigned r = idx / kR1bBlock, i = idx % kR1bBlock;
        const std::size_t g = static_cast<std::size_t>(r) * kN + i0 + i;
        const float v = X[g] * g_rr[r];
        xs[r][i] = v;
        F[g] = v;
    }
    __syncthreads();
    const unsigned o = t % kHcMix, r = t / kHcMix;
    if (r >= rows) return;
    float acc = 0.0F;
    for (unsigned i = 0; i < kR1bBlock; ++i) acc = __builtin_fmaf(xs[r][i], fs[o][i], acc);
    g_part[(static_cast<std::size_t>(r) * kHcMix + o) * kHcBlocks + j] = acc;
}

// R6 (host hc_mix order; copied from dev::hc_pre): z[24] -> m[24] (pre, post, comb after sinkhorn), serial.
__device__ void r6_serial(const float* z, const float* base, const float* scale, float* m) {
    for (unsigned j = 0; j < kHc; ++j) m[j] = sigmoid(z[j] * scale[0] + base[j]) + kHcEps;
    for (unsigned j = 0; j < kHc; ++j) m[kHc + j] = 2.0F * sigmoid(z[kHc + j] * scale[1] + base[kHc + j]);
    float* c = m + 2 * kHc;
    for (unsigned k = 0; k < kHc * kHc; ++k) c[k] = z[2 * kHc + k] * scale[2] + base[2 * kHc + k];
    for (unsigned j = 0; j < kHc; ++j) {
        float mx = c[j * kHc];
        for (unsigned k = 1; k < kHc; ++k) mx = c[j * kHc + k] > mx ? c[j * kHc + k] : mx;
        float e[kHc];
        for (unsigned k = 0; k < kHc; ++k) e[k] = declared_exp(c[j * kHc + k] - mx);
        const float sum = ((e[0] + e[1]) + e[2]) + e[3];
        for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = e[k] / sum + kHcEps;
    }
    for (unsigned it = 0; it < kHcIters; ++it) {
        if (it > 0) {
            float sr[kHc];
            for (unsigned j = 0; j < kHc; ++j)
                sr[j] = (((c[j * kHc + 0] + c[j * kHc + 1]) + c[j * kHc + 2]) + c[j * kHc + 3]) + kHcEps;
            for (unsigned j = 0; j < kHc; ++j)
                for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sr[j];
        }
        float sc[kHc];
        for (unsigned k = 0; k < kHc; ++k)
            sc[k] = (((c[0 * kHc + k] + c[1 * kHc + k]) + c[2 * kHc + k]) + c[3 * kHc + k]) + kHcEps;
        for (unsigned j = 0; j < kHc; ++j)
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sc[k];
    }
}

// Copied unchanged from glm_post_fast.hip; glmxch_bench gates this R6 against dev::hc_pre.
// R6 on wave 0 from z[24] in LDS into m[24] (hc_pre_v2's form: same op sequence per value as Tier A's lane 0).
__device__ __forceinline__ void r6_wave(const float* z, float* m, const float* base, const float* scale) {
    const unsigned l = threadIdx.x;
    if (l >= 32) return;
    const unsigned t = l < 16 ? l : 0, j = t / kHc, k = t % kHc;
    if (l >= 16 && l < 20) m[l - 16] = sigmoid(z[l - 16] * scale[0] + base[l - 16]) + kHcEps;
    if (l >= 20 && l < 24) m[kHc + l - 20] = 2.0F * sigmoid(z[kHc + l - 20] * scale[1] + base[kHc + l - 20]);
    float c = z[2 * kHc + t] * scale[2] + base[2 * kHc + t];
    {
        float e[kHc], cr[kHc];
        #pragma unroll
        for (unsigned q = 0; q < kHc; ++q) cr[q] = __shfl(c, static_cast<int>(j * kHc + q));
        float mx = cr[0];
        for (unsigned q = 1; q < kHc; ++q) mx = cr[q] > mx ? cr[q] : mx;
        #pragma unroll
        for (unsigned q = 0; q < kHc; ++q) e[q] = declared_exp(cr[q] - mx);
        const float sum = ((e[0] + e[1]) + e[2]) + e[3];
        const float ek = k == 0 ? e[0] : k == 1 ? e[1] : k == 2 ? e[2] : e[3];  // constant indices only (no scratch)
        c = ek / sum + kHcEps;
    }
    for (unsigned it = 0; it < kHcIters; ++it) {
        if (it > 0) {
            const float sr = (((__shfl(c, static_cast<int>(j * kHc + 0)) + __shfl(c, static_cast<int>(j * kHc + 1))) +
                               __shfl(c, static_cast<int>(j * kHc + 2))) +
                              __shfl(c, static_cast<int>(j * kHc + 3))) +
                             kHcEps;
            c = c / sr;
        }
        const float sc = (((__shfl(c, static_cast<int>(0 * kHc + k)) + __shfl(c, static_cast<int>(1 * kHc + k))) +
                           __shfl(c, static_cast<int>(2 * kHc + k))) +
                          __shfl(c, static_cast<int>(3 * kHc + k))) +
                         kHcEps;
        c = c / sc;
    }
    if (l < 16) m[2 * kHc + l] = c;
}

// Launch 3: per row, the 64-partial tree (offsets 32 .. 1) per chain, then R6 + collapse exactly as dev::hc_pre.
// Coherent: the partials were stored by other workgroups of the same launch (hc_site_kernel), read at agent scope.
template <bool Distributed, bool ParallelR6 = false, bool Norm = false, bool Coherent = false>
__device__ __forceinline__ void hc_fin_row(unsigned r, const float* X, const float* base, const float* scale, float* mix, float* xc,
                                           const std::uint16_t* norm, float* xn) {
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    const unsigned l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * kN;
    if constexpr (Distributed) {
        // Lane a owns indices a + 8*k. These local stages are exactly the
        // serial tree's offsets 32, 16, 8, including the left operand.
        if (l < kHcMix * 8) {
            const unsigned chain = l / 8, lane = l % 8;
            const float* src = g_part + (static_cast<std::size_t>(r) * kHcMix + chain) * kHcBlocks;
            float p[8];
#pragma unroll
            for (unsigned k = 0; k < 8; ++k)
                p[k] = Coherent ? __hip_atomic_load(&src[lane + 8 * k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : src[lane + 8 * k];
#pragma unroll
            for (unsigned h = 4; h > 0; h >>= 1U) {
#pragma unroll
                for (unsigned k = 0; k < h; ++k) p[k] = p[k] + p[k + h];
            }
            float acc = p[0];
            for (unsigned h = 4; h > 0; h >>= 1U) {
                const float peer = __shfl_down(acc, h, 8);
                if (lane < h) acc = acc + peer;
            }
            if (lane == 0) z[chain] = acc;
        }
    } else if (l < kHcMix) {
        float p[kHcBlocks];
        const float* src = g_part + (static_cast<std::size_t>(r) * kHcMix + l) * kHcBlocks;
        for (unsigned j = 0; j < kHcBlocks; ++j) p[j] = src[j];
        for (unsigned hh = kHcBlocks / 2; hh >= 1; hh /= 2)
            for (unsigned i = 0; i < hh; ++i) p[i] = p[i] + p[i + hh];
        z[l] = p[0];
    }
    __syncthreads();
    if constexpr (ParallelR6) {
        r6_wave(z, m, base, scale);
    } else if (l == 0) {
        r6_serial(z, base, scale, m);
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(r) * kHcMix + k] = m[k];
    }
    __syncthreads();
    if constexpr (ParallelR6)
        if (l < kHcMix) mix[static_cast<std::size_t>(r) * kHcMix + l] = m[l];
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
    if constexpr (Norm) {
        // Materialize xc before reading it, exactly as the separate rmsnorm launch.
        __syncthreads();
        __shared__ float ns[256];
        float a = 0.0F;
        for (unsigned i = l; i < kHidden; i += 256) a = __builtin_fmaf(xr[i], xr[i], a);
        const float ss = lds_tree<256>(ns, a);
        const float rr = inv_sqrt(ss / static_cast<float>(kHidden) + kEps);
        float* yr = xn + static_cast<std::size_t>(r) * kHidden;
        for (unsigned i = l; i < kHidden; i += 256)
            yr[i] = norm ? (xr[i] * rr) * bf16_to_float(norm[i]) : xr[i] * rr;
    }
}
template <bool Distributed, bool ParallelR6 = false, bool Norm = false>
__global__ __launch_bounds__(256) void hc_fin_kernel(const float* X, const float* base, const float* scale, float* mix, float* xc,
                                                     const std::uint16_t* norm = nullptr, float* xn = nullptr) {
    hc_fin_row<Distributed, ParallelR6, Norm>(blockIdx.x, X, base, scale, mix, xc, norm, xn);
}

// One launch per decode mHC site (hc_pre_norm with STRIX_GLM_HC_SITE): launches 1 + 2 (+ 3 when Fin) of hc_pre_norm in
// one grid of kHcBlocks workgroups. Every workgroup recomputes rr of each row exactly as hc_rr_kernel (the same 256
// lane-strided fmaf chains and the same lds_tree pairs, one tree per row), then runs hc_part's block j with that rr; with
// Fin the last workgroup to arrive (device-scope counter, as variant 2) runs hc_fin<true, true, true> row after row. Fin
// is for one row: the last workgroup would run rows serially, so multi-row calls launch hc_fin<..., Norm> separately.
__device__ unsigned g_site_cnt;
template <unsigned Rows, bool Fin>
__global__ __launch_bounds__(256) void hc_site_kernel(const float* X, const float* fn, float* F, const float* base, const float* scale,
                                                      float* mix, float* xc, const std::uint16_t* norm, float* xn, unsigned rows) {
    __shared__ float fs[kHcMix][kPad];
    __shared__ float xs[Rows][kPad];
    __shared__ float ss[Rows][256];
    __shared__ float rrs[Rows];
    __shared__ unsigned last;
    const unsigned j = blockIdx.x, t = threadIdx.x, i0 = j * kR1bBlock;
    for (unsigned r = 0; r < rows; ++r) {
        // All 64 loads in flight first, then hc_rr's chain in its order (i = t, t + 256, ...).
        const float* Xr = X + static_cast<std::size_t>(r) * kN + t;
        float v[kN / 256];
#pragma unroll
        for (unsigned k = 0; k < kN / 256; ++k) v[k] = Xr[k * 256];
        float a = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < kN / 256; ++k) a = __builtin_fmaf(v[k], v[k], a);
        ss[r][t] = a;
    }
    for (unsigned idx = t; idx < kHcMix * kR1bBlock; idx += 256) {
        const unsigned o = idx / kR1bBlock, i = idx % kR1bBlock;
        fs[o][i] = fn[static_cast<std::size_t>(o) * kN + i0 + i];
    }
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        for (unsigned r = 0; r < rows; ++r)
            if (t < h) ss[r][t] = ss[r][t] + ss[r][t + h];
        __syncthreads();
    }
    if (t < rows) rrs[t] = inv_sqrt(ss[t][0] / static_cast<float>(kN) + kEps);
    __syncthreads();
    for (unsigned idx = t; idx < rows * kR1bBlock; idx += 256) {
        const unsigned r = idx / kR1bBlock, i = idx % kR1bBlock;
        const std::size_t g = static_cast<std::size_t>(r) * kN + i0 + i;
        const float v = X[g] * rrs[r];
        xs[r][i] = v;
        F[g] = v;
    }
    __syncthreads();
    const unsigned o = t % kHcMix, r = t / kHcMix;
    if (t < kPartThreads && r < rows) {
        float acc = 0.0F;
        for (unsigned i = 0; i < kR1bBlock; ++i) acc = __builtin_fmaf(xs[r][i], fs[o][i], acc);
        float* dst = &g_part[(static_cast<std::size_t>(r) * kHcMix + o) * kHcBlocks + j];
        if constexpr (Fin) __hip_atomic_store(dst, acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        else *dst = acc;
    }
    if constexpr (Fin) {
        __threadfence();
        __syncthreads();
        if (t == 0) last = atomicAdd(&g_site_cnt, 1U) == gridDim.x - 1 ? 1U : 0U;
        __syncthreads();
        if (last == 0) return;
        __threadfence();
        for (unsigned q = 0; q < rows; ++q) {
            __syncthreads();
            hc_fin_row<true, true, true, true>(q, X, base, scale, mix, xc, norm, xn);
        }
        if (t == 0) atomicExch(&g_site_cnt, 0U);
    }
}


// Variant 2 (STRIX_GLM_R1B_HC=2): launches 2 + 3 fused. Every workgroup stores its block partials; the last workgroup
// to arrive (device-scope counter) does the trees, R6 and the collapse for all rows. Same arithmetic as variant 1.
__device__ unsigned g_hc_cnt;
__global__ __launch_bounds__(kPartThreads) void hc_part_fin_kernel(const float* X, const float* fn, float* F, const float* base,
                                                                   const float* scale, float* mix, float* xc, unsigned rows) {
    __shared__ float fs[kHcMix][kPad];
    __shared__ float xs[kMaxRows][kPad];
    __shared__ float zs[kMaxRows][kHcMix];
    __shared__ float ms[kMaxRows][kHcMix];
    __shared__ unsigned last;
    const unsigned j = blockIdx.x, t = threadIdx.x, i0 = j * kR1bBlock;
    for (unsigned idx = t; idx < kHcMix * kR1bBlock; idx += kPartThreads) {
        const unsigned o = idx / kR1bBlock, i = idx % kR1bBlock;
        fs[o][i] = fn[static_cast<std::size_t>(o) * kN + i0 + i];
    }
    for (unsigned idx = t; idx < rows * kR1bBlock; idx += kPartThreads) {
        const unsigned r = idx / kR1bBlock, i = idx % kR1bBlock;
        const std::size_t g = static_cast<std::size_t>(r) * kN + i0 + i;
        const float v = X[g] * g_rr[r];
        xs[r][i] = v;
        F[g] = v;
    }
    __syncthreads();
    const unsigned o = t % kHcMix, r = t / kHcMix;
    if (r < rows) {
        float acc = 0.0F;
        for (unsigned i = 0; i < kR1bBlock; ++i) acc = __builtin_fmaf(xs[r][i], fs[o][i], acc);
        __hip_atomic_store(&g_part[(static_cast<std::size_t>(r) * kHcMix + o) * kHcBlocks + j], acc, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
    }
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(&g_hc_cnt, 1U) == gridDim.x - 1 ? 1U : 0U;
    __syncthreads();
    if (last == 0) return;
    __threadfence();
    if (r < rows) {
        float p[kHcBlocks];
        const std::size_t base_i = (static_cast<std::size_t>(r) * kHcMix + o) * kHcBlocks;
        for (unsigned b = 0; b < kHcBlocks; ++b) p[b] = __hip_atomic_load(&g_part[base_i + b], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        for (unsigned hh = kHcBlocks / 2; hh >= 1; hh /= 2)
            for (unsigned i = 0; i < hh; ++i) p[i] = p[i] + p[i + hh];
        zs[r][o] = p[0];
    }
    __syncthreads();
    if (t < rows) {
        r6_serial(zs[t], base, scale, ms[t]);
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(t) * kHcMix + k] = ms[t][k];
    }
    __syncthreads();
    for (unsigned idx = t; idx < rows * kHidden; idx += kPartThreads) {
        const unsigned rr = idx / kHidden, d = idx % kHidden;
        const float* Xr = X + static_cast<std::size_t>(rr) * kN;
        const float* m = ms[rr];
        xc[static_cast<std::size_t>(rr) * kHidden + d] =
            ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
    }
    if (t == 0) atomicExch(&g_hc_cnt, 0U);
}

}  // namespace

void set_active(bool on) { g_active.store(on); }
bool active() { return g_active.load(); }

int dense_variant() {
    static const int v = std::getenv("STRIX_GLM_R1B_DENSE") != nullptr ? std::atoi(std::getenv("STRIX_GLM_R1B_DENSE")) : 7;
    return v;
}

void pack_k4096(const std::uint16_t* W, unsigned n, std::uint16_t* packed, hipStream_t s) {
    if (n == 0) throw std::invalid_argument("r1b pack_k4096: n >= 1");
    pack_k4096_kernel<<<static_cast<unsigned>((static_cast<std::size_t>(n) * 512 + 255) / 256), 256, 0, s>>>(W, n, packed);
    ok(hipGetLastError(), "pack_k4096");
}

void dense_f16_k4096(const float* x, unsigned ldx, const std::uint16_t* W, unsigned n, float* y, unsigned ldy, unsigned rows,
                     hipStream_t s, int variant) {
    DJobs J{};
    J.x[0] = x;
    J.W[0] = W;
    J.y[0] = y;
    J.n[0] = n;
    J.ldx[0] = ldx;
    J.ldy[0] = ldy;
    J.nj = 1;
    J.rows = rows;
    dense_jobs(J, s, variant);
}

void dense_f16_k4096_x3(const float* x, unsigned ldx, const std::uint16_t* const W[3], const unsigned n[3], float* const y[3],
                        const unsigned ldy[3], unsigned rows, hipStream_t s, int variant) {
    DJobs J{};
    for (unsigned q = 0; q < 3; ++q) {
        J.x[q] = x;
        J.W[q] = W[q];
        J.y[q] = y[q];
        J.n[q] = n[q];
        J.ldx[q] = ldx;
        J.ldy[q] = ldy[q];
    }
    J.nj = 3;
    J.rows = rows;
    dense_jobs(J, s, variant);
}

void router_dense(const float* x, const std::uint16_t* router, float* logits, unsigned rows, hipStream_t s, int variant) {
    dense_f16_k4096(x, kHidden, router, kExperts, logits, kExperts, rows, s, variant);
}

void hc_pre(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
            unsigned rows, hipStream_t s, int variant) {
    if (rows == 0 || rows > kMaxRows) throw std::invalid_argument("r1b hc_pre: rows 1..8");
    hc_rr_kernel<<<rows, 256, 0, s>>>(X);
    ok(hipGetLastError(), "hc_rr");
    static const int env_variant = std::getenv("STRIX_GLM_R1B_HC") != nullptr ? std::atoi(std::getenv("STRIX_GLM_R1B_HC")) : 4;
    if (variant < 0) variant = env_variant;
    if (variant == 2) {
        hc_part_fin_kernel<<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, base, scale, mix, xc, rows);
        ok(hipGetLastError(), "hc_part_fin");
        return;
    }
    if ((variant == 3 || variant == 4) && rows == 1) {
        hc_part_kernel<1><<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, rows);
    } else {
        hc_part_kernel<kMaxRows><<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, rows);
    }
    ok(hipGetLastError(), "hc_part");
    if (variant == 4) {
        hc_fin_kernel<true, true><<<rows, 256, 0, s>>>(X, base, scale, mix, xc);
    } else if (variant == 3) {
        hc_fin_kernel<true><<<rows, 256, 0, s>>>(X, base, scale, mix, xc);
    } else {
        hc_fin_kernel<false><<<rows, 256, 0, s>>>(X, base, scale, mix, xc);
    }
    ok(hipGetLastError(), "hc_fin");
}

bool hc_norm_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRIX_GLM_HC_NORM");
        // The fused finish is hc_pre variant 4 (the default); any other STRIX_GLM_R1B_HC keeps its own unfused path.
        const char* hc = std::getenv("STRIX_GLM_R1B_HC");
        return v != nullptr && std::string(v) == "1" && (hc == nullptr || std::atoi(hc) == 4);
    }();
    return enabled;
}

int hc_site_mode() {
    // 1 (default): one launch per site at one row, three at 2..8 rows; 2: also two launches (rr + part, fin + norm) at
    // 2..8 rows (measured +0.7 % R8 ms/step: the per-workgroup rr recompute costs more than the saved launch); 0: three.
    static const int mode = [] {
        const char* v = std::getenv("STRIX_GLM_HC_SITE");
        return v != nullptr && *v != '\0' ? std::atoi(v) : 1;
    }();
    return mode;
}

void hc_pre_norm(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
                 const std::uint16_t* norm, float* xn, unsigned rows, hipStream_t s, int site) {
    if (rows == 0 || rows > kMaxRows) throw std::invalid_argument("r1b hc_pre_norm: rows 1..8");
    if (site < 0) site = hc_site_mode();
    if (site == 2 || (site == 1 && rows == 1)) {
        if (rows == 1) {
            hc_site_kernel<1, true><<<kHcBlocks, 256, 0, s>>>(X, fn, F, base, scale, mix, xc, norm, xn, rows);
            ok(hipGetLastError(), "hc_site");
            return;
        }
        hc_site_kernel<kMaxRows, false><<<kHcBlocks, 256, 0, s>>>(X, fn, F, base, scale, mix, xc, norm, xn, rows);
        ok(hipGetLastError(), "hc_site_part");
        hc_fin_kernel<true, true, true><<<rows, 256, 0, s>>>(X, base, scale, mix, xc, norm, xn);
        ok(hipGetLastError(), "hc_site_fin");
        return;
    }
    hc_rr_kernel<<<rows, 256, 0, s>>>(X);
    ok(hipGetLastError(), "hc_norm_rr");
    if (rows == 1) hc_part_kernel<1><<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, rows);
    else hc_part_kernel<kMaxRows><<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, rows);
    ok(hipGetLastError(), "hc_norm_part");
    hc_fin_kernel<true, true, true><<<rows, 256, 0, s>>>(X, base, scale, mix, xc, norm, xn);
    ok(hipGetLastError(), "hc_norm_fin");
}

}  // namespace strix::glm::r1b
