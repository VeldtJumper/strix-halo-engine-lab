// CUDA routed-MoE prefill of the Qwen A8 profile, bit-identical to the HIP iu8 WMMA path (see include/strix/cuda_qwen_moe.hpp).
//
// Exactness notes (HIP sources at engine commit bf60d9d2; ISA = the gfx1151 disassembly):
// - The routed-MoE HIP TUs are built with -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee
//   and #pragma clang fp contract(off) (hip_moe_q4k_a8_known_rows.hip:14, hip_moe_q4k_a8_probe.hip:12). ISA check of
//   project_prefill_wmma<{0,1},16> (hip_moe_q4k_a8_known_rows.s:193768 / :188834): its 10 v_fma_f32 + 6 v_fmac_f32 +
//   1 v_dual_fmac_f32 are exactly the expf sequence (1 fma + 1 fmac) and three correctly rounded divisions
//   (v_div_scale/v_rcp/3x v_fma/2x v_fmac/v_div_fmas/v_div_fixup): g/(1+e), x/127, hv/delta. The chain epilogue is v_mul/
//   v_sub/v_add only. down_prefill_wmma<{1,0},16> (:197936 / :200810) has no FMA at all; float(r) is v_add_f32 0xcb400000
//   (x + -1.5*2^23, the same value as the source subtraction). prefill_prologue<1> (hip_moe_q4k_a8_probe.s:7306): 6 v_fma +
//   4 v_fmac = two correctly rounded divisions, v_rndne for rintf, v_maxmin for the clamp.
// - The merge is inline asm v_fmac_f32 current, w[slot], dot[slot] in slot order from current = 0
//   (hip_moe_experts_launcher.s:3152 + 215..470): __fmaf_rn(w, dot, current).
// - The expf of the SwiGLU (hip_moe_q4k_a8_known_rows.s:193768+3836..3857) multiplies g by -log2e (0xbfb8aa3b), fma with
//   -log2e, fmac with -log2e_lo (0xb2a5705f), rndne, v_exp, ldexp, then the two range selects on g with the thresholds
//   negated: the same operations as amd_expf(-g) (negation is exact).
// - Integer dots (iu8 WMMA in HIP, mma.sync s8 here) are exact: |dot| <= 32*127*31 < 2^22, accumulated from the magic seed
//   0x4b400000 so that int_as_float(acc) - 1.5*2^23 is float(dot) exactly (group_float, known_rows.hip:730).
// - Gate/up per group: HIP i = float(sc)*float(r), u = d*i, j = float(mn)*s, v = dm*j (known_rows.hip:833-836). Here
//   u = (d*sc)*float(r) and v = (dm*mn)*s with d*sc and dm*mn precomputed: d is FP16 (11-bit significand) and sc, mn < 64,
//   so d*sc and dm*mn are exact in FP32, and sc*r (< 2^23), mn*s (< 2^18) are exact too: both sides are the exact product
//   d*sc*r rounded once (same for zero signs, infinities and NaN). Every other operation is HIP's, in HIP's order.
#include "strix/cuda_qwen_moe.hpp"
#include "strix/cuda_qwen_quant.cuh"

#include <cuda_fp16.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::cuda::qwen_moe {
namespace {

constexpr int kMagic = 0x4b400000;
constexpr float kMagicF = 12582912.0F;  // 0x1.8p23

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T> __device__ __forceinline__ T* at(Strided<T> v, std::uint32_t r) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(v.base) + static_cast<std::uint64_t>(r) * v.stride);
}
__device__ __forceinline__ bool finite(float x) { return (__float_as_uint(x) & 0x7f800000U) != 0x7f800000U; }
// NaN stored to memory as the gfx1151 default quiet NaN. Every NaN of this pipeline is generated inside it (inf - inf,
// 0 * inf: inputs and weights are checked finite), and the AMD default NaN is 0x7fc00000; its sign after NaN propagation
// through v_sub/neg modifiers is not modelled (open question), which only matters on the fail-closed path.
__device__ __forceinline__ float canon(float x) { return x != x ? __uint_as_float(0x7fc00000U) : x; }
__device__ __forceinline__ float half_bits(std::uint32_t b) { return __half2float(__ushort_as_half(static_cast<unsigned short>(b))); }

__device__ __forceinline__ void cp16(void* s, const void* g) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void cp8(void* s, const void* g) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// D = A(16x32 s8, row) * B(32x8 s8, col) + magic. Fragments (PTX ISA, m16n8k32 .s8): g = lane/4, q = lane%4;
// a0: (row g, k 4q..4q+3) a1: (g+8, 4q..) a2: (g, 16+4q..) a3: (g+8, 16+4q..); b0: (k 4q.., col g) b1: (k 16+4q.., col g);
// d0,d1: (row g, cols 2q, 2q+1) d2,d3: (row g+8, cols 2q, 2q+1).
__device__ __forceinline__ void mma_magic(const unsigned (&a)[4], unsigned b0, unsigned b1, int (&d)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(kMagic));
}
__device__ __forceinline__ float group_float(int bits) { return __fsub_rn(__int_as_float(bits), kMagicF); }
__device__ __forceinline__ unsigned lds32(const unsigned char* p) { return *reinterpret_cast<const unsigned*>(p); }

// Q4_K / Q5_K scale and min of group 0..7 from the 16-byte header (known_rows.hip:39-43, = ggml get_scale_min_k4).
__device__ __forceinline__ unsigned factor(uint4 header, unsigned group, bool minimum) {
    const unsigned shift = 8U * (group & 3U);
    const unsigned a = (header.y >> shift) & 255U, b = (header.z >> shift) & 255U, c = (header.w >> shift) & 255U;
    return group < 4U ? (minimum ? b : a) & 63U : minimum ? (c >> 4U) | ((b >> 6U) << 4U) : (c & 15U) | ((a >> 6U) << 4U);
}

// ---------------------------------------------------------------------------------------------------------------------
// prologue: prefill_prologue<true> (probe.hip:159-173) with quantize_row<true> (probe.hip:41-73). One CTA per row; the
// group -> warp mapping differs from HIP (1024 threads) but each group is computed by one warp exactly as in HIP and the
// flags are an OR, so every byte is the same.
__global__ __launch_bounds__(256) void prologue_kernel(Rows v, std::uint32_t clear_words) {
    const std::uint32_t row = blockIdx.x, t = threadIdx.x, lane = t & 31U, wave = t >> 5;
    __shared__ __align__(16) signed char local_codes[hidden];
    __shared__ float local_deltas[hidden / 32U];
    __shared__ int local_sums[hidden / 32U];
    __shared__ unsigned route_bits, bad;
    if (t == 0) { route_bits = 0U; bad = 0U; }
    float* clear = at(v.accumulation, row);
    for (std::uint32_t i = t; i < clear_words; i += blockDim.x) clear[i] = 0.0F;
    __syncthreads();
    if (t < top_k && (at(v.routes, row)[t] >= experts || !finite(at(v.route_weights, row)[t]))) atomicOr(&route_bits, invalid_route);
    __syncthreads();
    if (route_bits == 0U) {
        const std::uint16_t* input = at(v.input, row);
        for (std::uint32_t group = wave; group < hidden / 32U; group += blockDim.x / 32U) {
            const std::uint32_t i = group * 32U + lane;
            float x = __uint_as_float(static_cast<unsigned>(input[i]) << 16U);
            if (!finite(x)) { atomicOr(&bad, nonfinite_activation); x = 0.0F; }
            float a = fabsf(x);
            for (int s = 16; s != 0; s /= 2) a = fmaxf(a, __shfl_xor_sync(0xffffffffU, a, s));
            float delta = a == 0.0F ? 1.0F : __fdiv_rn(a, 127.0F);
            if (!(delta > 0.0F) || !finite(delta)) { atomicOr(&bad, invalid_delta); delta = 1.0F; }
            const float rounded = rintf(__fdiv_rn(x, delta));
            const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
            local_codes[i] = static_cast<signed char>(code);
            int sum = code;
            for (int s = 16; s != 0; s /= 2) sum += __shfl_xor_sync(0xffffffffU, sum, s);
            if (lane == 0) { local_deltas[group] = delta; local_sums[group] = sum; }
        }
    }
    __syncthreads();
    const unsigned failure = route_bits | bad;
    if (route_bits == 0U && bad == 0U) {
        std::int8_t* codes = at(v.codes, row);
        for (std::uint32_t i = t; i < hidden / 16U; i += blockDim.x)
            reinterpret_cast<int4*>(codes)[i] = reinterpret_cast<const int4*>(local_codes)[i];
        for (std::uint32_t i = t; i < hidden / 32U; i += blockDim.x) { at(v.deltas, row)[i] = local_deltas[i]; at(v.sums, row)[i] = local_sums[i]; }
    }
    if (t == 0) *at(v.status, row) = failure;
}

// ---------------------------------------------------------------------------------------------------------------------
// csr: build_prefill_csr_waves (known_rows.hip:460-532) with cap 16, transcribed (32 waves of 32 lanes; the AMD
// ballot/shfl peer loop becomes __match_any_sync, and a __syncwarp separates the position reads from the leader updates,
// which the AMD wave executes in lockstep). Also emits this port's tiles (runs of <= 64 assignments of one expert).
__global__ __launch_bounds__(1024) void csr_kernel(Rows v, std::uint32_t rows) {
    constexpr unsigned Waves = 32U, cap = hip_batch, tcap = tile_rows;
    const Csr& csr = v.csr;
    const unsigned t = threadIdx.x, lane = t & 31U, wave = t / 32U, count = rows * top_k, span = (count + Waves - 1U) / Waves;
    const unsigned first = wave * span, last = min(count, first + span);
    __shared__ unsigned position[Waves][256], n_of[512], scan[4][512], hist[16], failure, status, duplicates, maximum;
    if (t == 0) { failure = 0U; status = 0U; duplicates = 0U; maximum = 0U; }
    if (t < 16U) hist[t] = 0U;
    for (unsigned k = t; k < Waves * 256U; k += 1024U) position[k / 256U][k % 256U] = 0U;
    __syncthreads();
    const auto expert_at = [&](unsigned a) { return at(v.routes, a / top_k)[a % top_k]; };
    for (unsigned a = first + lane; a < last; a += 32U) {
        const unsigned e = expert_at(a);
        if (e >= experts || !finite(at(v.route_weights, a / top_k)[a % top_k])) atomicOr(&failure, invalid_route);
        else atomicAdd(&position[wave][e / 2U], 1U << (16U * (e & 1U)));
    }
    for (unsigned r = t; r < rows; r += 1024U) { const unsigned s = *at(v.status, r); if (s) atomicOr(&status, s); }
    __syncthreads();
    const auto fail_summary = [&] {
        if (t < 32U) csr.summary[t] = t == 0U ? failure | status : t == 3U ? count : t == 6U ? rows : t == 7U ? 1U : 0U;
        if (t == 0U) *csr.tile_count = 0U;
    };
    if (failure) { fail_summary(); return; }
    if (t < 256U) {
        unsigned lo = 0U, hi = 0U;
        for (unsigned w = 0; w < Waves; ++w) { const unsigned c = position[w][t]; position[w][t] = lo | hi << 16U; lo += c & 65535U; hi += c >> 16U; }
        if (lo > 65535U || hi > 65535U) atomicOr(&failure, invalid_route);
        n_of[2U * t] = lo; n_of[2U * t + 1U] = hi;
    }
    __syncthreads();
    if (failure) { fail_summary(); return; }
    const unsigned n = t < 512U ? n_of[t] : 0U, nb = (n + cap - 1U) / cap, nt = (n + tcap - 1U) / tcap;
    if (t < 512U) { scan[0][t] = n; scan[1][t] = n != 0U; scan[2][t] = nb; scan[3][t] = nt; }
    __syncthreads();
    for (unsigned d = 1U; d < 512U; d <<= 1U) {
        unsigned x[4] = {};
        if (t >= d && t < 512U) for (unsigned k = 0; k < 4U; ++k) x[k] = scan[k][t - d];
        __syncthreads();
        if (t >= d && t < 512U) for (unsigned k = 0; k < 4U; ++k) scan[k][t] += x[k];
        __syncthreads();
    }
    for (unsigned step = first; step < last; step += 32U) {
        const unsigned a = step + lane;
        const bool valid = a < last;
        const unsigned e = valid ? expert_at(a) : 0xffffffffU;
        const unsigned vm = __ballot_sync(0xffffffffU, valid);
        const unsigned peers = __match_any_sync(0xffffffffU, e) & vm;
        unsigned slot = 0U;
        if (valid) slot = scan[0][e] - n_of[e] + ((position[wave][e / 2U] >> (16U * (e & 1U))) & 65535U);
        __syncwarp();
        if (valid) {
            const unsigned below = peers & ((1U << lane) - 1U);
            const unsigned at_pos = slot + static_cast<unsigned>(__popc(below));
            csr.rows[at_pos] = a / top_k; csr.slots[at_pos] = a % top_k;
            if (below == 0U) atomicAdd(&position[wave][e / 2U], static_cast<unsigned>(__popc(peers)) << (16U * (e & 1U)));
        }
        __syncwarp();
    }
    __syncthreads();
    if (t < experts) {
        const unsigned off = scan[0][t] - n, touched = scan[1][t] - (n != 0U), batch = scan[2][t] - nb, tile = scan[3][t] - nt;
        csr.offsets[t] = off;
        if (t == experts - 1U) csr.offsets[experts] = off + n;
        if (n) {
            csr.touched[touched] = t; atomicMax(&maximum, n);
            for (unsigned b = 0; b < nb; ++b) {
                csr.batch_experts[batch + b] = t; csr.batch_begins[batch + b] = off + b * cap;
                atomicAdd(&hist[8U + min(8U, min(cap, n - b * cap)) - 1U], 1U);
            }
            for (unsigned b = 0; b < nt; ++b) { csr.tile_experts[tile + b] = t; csr.tile_begins[tile + b] = off + b * tcap; }
            unsigned unique = 0U, previous = 0xffffffffU;
            for (unsigned k = 0; k < n; ++k) { const unsigned r = csr.rows[off + k]; unique += r != previous; previous = r; }
            atomicAdd(&hist[min(8U, unique) - 1U], 1U); atomicAdd(&duplicates, n - unique);
        }
    }
    __syncthreads();
    if (t == 0U) { csr.batch_begins[scan[2][511]] = count; csr.tile_begins[scan[3][511]] = count; *csr.tile_count = scan[3][511]; }
    if (t < 32U)
        csr.summary[t] = t == 0U ? status : t == 1U ? scan[1][511] : t == 2U ? scan[2][511] : t == 3U ? count : t == 4U ? duplicates
            : t == 5U ? maximum : t == 6U ? rows : t == 7U ? 1U : t >= 8U && t < 24U ? hist[t - 8U] : 0U;
}

// ---------------------------------------------------------------------------------------------------------------------
// project: project_prefill_wmma<Q5K,16> (known_rows.hip:860-909) for a tile of <= 64 assignments of one expert and 64
// outputs of both gate and up. 16 warps: warp w covers rows 16*(w%4).. and outputs 16*(w/4).. of both matrices
// (2 n8 tiles each). Per (row, output) element and matrix, four chains acc[p] (p = pair of groups 2p, 2p+1 of a K block),
// K blocks ascending, per pair (gate_up_chains :831-840):
//   u0 = (d*sc0)*float(r0), v0 = (dm*mn0)*s0, t0 = u0 - v0; u1, v1, t1 likewise; c0 = e0*t0, c1 = e1*t1;
//   pair = c0 + c1; acc[p] = acc[p] + pair;      fold (Chains::fold :782-785): (acc0 + acc1) + (acc2 + acc3).
// Pipeline: cp.async of K block b+1 (codes, raw weight blocks, deltas/sums) while block b computes.
constexpr unsigned kProjThreads = 512U, kProjOuts = 64U, kAStride = 272U;
template <bool Q5K> struct ProjLayout {
    static constexpr unsigned WB = Q5K ? 176U : 144U, QS = Q5K ? 48U : 16U;
    static constexpr unsigned a_bytes = tile_rows * kAStride, w_bytes = 2U * kProjOuts * WB, es_bytes = tile_rows * 64U;
    static constexpr unsigned stage = a_bytes + w_bytes + es_bytes;
    static constexpr unsigned scales = 2U * stage;                               // D[2][8][64], DM[2][8][64], E[8][64], S[8][64]
    static constexpr unsigned rowinfo = scales + (2U * 2U * 8U * 64U + 2U * 8U * 64U) * 4U;
    static constexpr unsigned total = rowinfo + 2U * tile_rows * 4U;
};
static_assert(ProjLayout<true>::total <= 101376U, "project smem");

template <bool Q5K>
__global__ __launch_bounds__(kProjThreads, 1) void project_kernel(const std::uint8_t* gate, const std::uint8_t* up, Rows v,
                                                                   amdmath::Tables tables) {
    using L = ProjLayout<Q5K>;
    const Csr& csr = v.csr;
    if ((csr.summary[0] & ~projection_nonfinite) || blockIdx.x >= *csr.tile_count) return;
    const unsigned begin = csr.tile_begins[blockIdx.x], count = csr.tile_begins[blockIdx.x + 1U] - begin;
    const unsigned expert = csr.tile_experts[blockIdx.x], out0 = blockIdx.y * kProjOuts;
    extern __shared__ __align__(16) unsigned char smem[];
    float* D = reinterpret_cast<float*>(smem + L::scales);  // [m][j][o]
    float* DM = D + 2U * 8U * 64U;
    float* E = DM + 2U * 8U * 64U;                          // [j][row]
    float* S = E + 8U * 64U;
    unsigned* row_of = reinterpret_cast<unsigned*>(smem + L::rowinfo);
    unsigned* slot_of = row_of + tile_rows;
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < tile_rows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    const auto load = [&](unsigned b) {
        unsigned char* st = smem + (b & 1U) * L::stage;
        for (unsigned i = t; i < tile_rows * 16U; i += kProjThreads) {
            const unsigned r = i >> 4, c = i & 15U;
            cp16(st + r * kAStride + 16U * c, at(v.codes, row_of[r]) + 256U * b + 16U * c);
        }
        constexpr unsigned CH = L::WB / 16U;
        for (unsigned i = t; i < 2U * kProjOuts * CH; i += kProjThreads) {
            const unsigned m = i / (kProjOuts * CH), rem = i % (kProjOuts * CH), o = rem / CH, c = rem % CH;
            const std::uint8_t* w = (m ? up : gate) + ((static_cast<std::uint64_t>(expert) * ffn + out0 + o) * (hidden / 256U) + b) * L::WB;
            cp16(st + L::a_bytes + (m * kProjOuts + o) * L::WB + 16U * c, w + 16U * c);
        }
        for (unsigned i = t; i < tile_rows * 4U; i += kProjThreads) {
            const unsigned r = i >> 2, c = i & 3U;
            const void* src = c < 2U ? static_cast<const void*>(at(v.deltas, row_of[r]) + 8U * b + 4U * c)
                                     : static_cast<const void*>(at(v.sums, row_of[r]) + 8U * b + 4U * (c - 2U));
            cp16(st + L::a_bytes + L::w_bytes + r * 64U + 16U * c, src);
        }
        cp_commit();
    };
    const unsigned g = lane >> 2, q = lane & 3U, wm = warp & 3U, wn = warp >> 2;
    const bool active = 16U * wm < count;
    float acc[2][2][4][4];
#pragma unroll
    for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
        for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
            for (unsigned e = 0; e < 4U; ++e)
#pragma unroll
                for (unsigned p = 0; p < 4U; ++p) acc[m][n][e][p] = 0.0F;
    constexpr unsigned blocks = hidden / 256U;
    load(0U);
#pragma unroll 1
    for (unsigned b = 0; b < blocks; ++b) {
        cp_wait_all();
        __syncthreads();  // block b landed; every warp finished block b-1 (its stage and the scale arrays are free)
        if (b + 1U < blocks) load(b + 1U);
        const unsigned char* st = smem + (b & 1U) * L::stage;
        const unsigned char* A = st;
        const unsigned char* W = st + L::a_bytes;
        const unsigned char* ES = st + L::a_bytes + L::w_bytes;
        for (unsigned i = t; i < 2U * 8U * 64U; i += kProjThreads) {
            const unsigned m = i >> 9, j = (i >> 6) & 7U, o = i & 63U;
            const uint4 h = *reinterpret_cast<const uint4*>(W + (m * kProjOuts + o) * L::WB);
            D[i] = __fmul_rn(half_bits(h.x & 65535U), static_cast<float>(factor(h, j, false)));
            DM[i] = __fmul_rn(half_bits(h.x >> 16U), static_cast<float>(factor(h, j, true)));
        }
        for (unsigned i = t; i < 8U * 64U; i += kProjThreads) {
            const unsigned j = i >> 6, r = i & 63U;
            E[i] = reinterpret_cast<const float*>(ES + r * 64U)[j];
            S[i] = static_cast<float>(reinterpret_cast<const int*>(ES + r * 64U + 32U)[j]);
        }
        __syncthreads();
        if (!active) continue;
        const unsigned char* A0 = A + (16U * wm + g) * kAStride;
        const unsigned r0 = 16U * wm + g;
#pragma unroll
        for (unsigned p = 0; p < 4U; ++p) {
            unsigned a[2][4];
            float e[2][2], s[2][2];
#pragma unroll
            for (unsigned G = 0; G < 2U; ++G) {
                const unsigned k0 = 64U * p + 32U * G, j = 2U * p + G;
                a[G][0] = lds32(A0 + k0 + 4U * q); a[G][1] = lds32(A0 + 8U * kAStride + k0 + 4U * q);
                a[G][2] = lds32(A0 + k0 + 16U + 4U * q); a[G][3] = lds32(A0 + 8U * kAStride + k0 + 16U + 4U * q);
                e[G][0] = E[j * 64U + r0]; e[G][1] = E[j * 64U + r0 + 8U];
                s[G][0] = S[j * 64U + r0]; s[G][1] = S[j * 64U + r0 + 8U];
            }
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
                for (unsigned n = 0; n < 2U; ++n) {
                    const unsigned ob = 16U * wn + 8U * n;
                    const unsigned char* wq = W + (m * kProjOuts + ob + g) * L::WB;
                    const unsigned w0 = lds32(wq + L::QS + 32U * p + 4U * q), w1 = lds32(wq + L::QS + 32U * p + 16U + 4U * q);
                    constexpr unsigned mask = 0x0f0f0f0fU;
                    unsigned b00 = w0 & mask, b01 = w1 & mask, b10 = (w0 >> 4U) & mask, b11 = (w1 >> 4U) & mask;
                    if constexpr (Q5K) {
                        const unsigned h0 = lds32(wq + 16U + 4U * q), h1 = lds32(wq + 32U + 4U * q);
                        const unsigned g0 = 2U * p, g1 = g0 + 1U;
                        b00 |= ((h0 >> g0) & 0x01010101U) << 4U; b01 |= ((h1 >> g0) & 0x01010101U) << 4U;
                        b10 |= ((h0 >> g1) & 0x01010101U) << 4U; b11 |= ((h1 >> g1) & 0x01010101U) << 4U;
                    }
                    int c0[4], c1[4];
                    mma_magic(a[0], b00, b01, c0);
                    mma_magic(a[1], b10, b11, c1);
                    const unsigned oc = ob + 2U * q;
                    const float2 d0 = *reinterpret_cast<const float2*>(D + (m * 8U + 2U * p) * 64U + oc);
                    const float2 d1 = *reinterpret_cast<const float2*>(D + (m * 8U + 2U * p + 1U) * 64U + oc);
                    const float2 m0 = *reinterpret_cast<const float2*>(DM + (m * 8U + 2U * p) * 64U + oc);
                    const float2 m1 = *reinterpret_cast<const float2*>(DM + (m * 8U + 2U * p + 1U) * 64U + oc);
#pragma unroll
                    for (unsigned el = 0; el < 4U; ++el) {
                        const unsigned hr = el >> 1, col = el & 1U;
                        const float u0 = __fmul_rn(col ? d0.y : d0.x, group_float(c0[el]));
                        const float v0 = __fmul_rn(col ? m0.y : m0.x, s[0][hr]);
                        const float t0 = __fsub_rn(u0, v0);
                        const float u1 = __fmul_rn(col ? d1.y : d1.x, group_float(c1[el]));
                        const float v1 = __fmul_rn(col ? m1.y : m1.x, s[1][hr]);
                        const float t1 = __fsub_rn(u1, v1);
                        const float x0 = __fmul_rn(e[0][hr], t0), x1 = __fmul_rn(e[1][hr], t1);
                        const float pair = __fadd_rn(x0, x1);
                        acc[m][n][el][p] = __fadd_rn(acc[m][n][el][p], pair);
                    }
                }
        }
    }
    __syncthreads();  // stages are reused for the g/u tile
    float* Gt = reinterpret_cast<float*>(smem);   // [64 rows][64 outputs]
    float* Ut = Gt + tile_rows * kProjOuts;
    if (active) {
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
            for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
                for (unsigned el = 0; el < 4U; ++el) {
                    const float* c = acc[m][n][el];
                    const float s01 = __fadd_rn(c[0], c[1]), s23 = __fadd_rn(c[2], c[3]);
                    const unsigned r = 16U * wm + g + 8U * (el >> 1), o = 16U * wn + 8U * n + 2U * q + (el & 1U);
                    (m ? Ut : Gt)[r * kProjOuts + o] = __fadd_rn(s01, s23);
                }
    }
    __syncthreads();
    // Serial down_activation_quantize per (assignment, 32-output group), one warp per job (known_rows.hip:888-909).
    const bool write_intermediates = v.intermediates.base != nullptr;
    for (unsigned job = warp; job < count * 2U; job += kProjThreads / 32U) {
        const unsigned a = job >> 1, quad = job & 1U, r = row_of[a], slot = slot_of[a];
        const unsigned feature = out0 + 32U * quad + lane, at_slot = slot * 2U * ffn + feature;
        const float gv = Gt[a * kProjOuts + 32U * quad + lane], uv = Ut[a * kProjOuts + 32U * quad + lane];
        const float ex = amdmath::amd_expf(tables, -gv);
        const float hv = __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0F, ex)), uv);
        float* output = write_intermediates ? at(v.intermediates, r) : nullptr;
        if (output) output[at_slot + ffn] = canon(uv);
        const unsigned bad = __ballot_sync(0xffffffffU, !finite(gv) || !finite(uv) || !finite(hv));
        if (bad != 0U) {
            if (output) output[at_slot] = canon(gv);
            if (lane == 0) { atomicOr(at(v.status, r), projection_nonfinite); atomicOr(csr.summary, projection_nonfinite); }
            continue;
        }
        float x = fabsf(hv);
        for (int s = 16; s != 0; s /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffffU, x, s));
        const bool tiny = x > 0.0F && x < 0x1p-119F;
        const float delta = x == 0.0F || tiny ? 1.0F : __fdiv_rn(x, 127.0F);
        const float rounded = tiny ? 0.0F : rintf(__fdiv_rn(hv, delta));
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int s = 16; s != 0; s /= 2) sum += __shfl_xor_sync(0xffffffffU, sum, s);
        if (output) output[at_slot] = hv;
        at(v.down_codes, r)[slot * ffn + feature] = static_cast<std::int8_t>(code);
        if (lane == 0) { at(v.down_deltas, r)[slot * (ffn / 32U) + feature / 32U] = delta; at(v.down_sums, r)[slot * (ffn / 32U) + feature / 32U] = sum; }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// down: down_prefill_wmma<Q51,16> (known_rows.hip:1005-1032) for a tile of <= 64 assignments and 64 hidden outputs.
// 8 warps: warp w covers rows 16*(w%4).. and outputs 32*(w/4).. (4 n8 tiles). Per element four chains, group 4k+j into
// chain j, groups ascending (down_chains :966-987):
//   first = d*float(r); Q5_1: combined = first + m*sum (Q8_0: combined = first); contribution = delta*combined;
//   acc[j] = acc[j] + contribution;   fold (acc0 + acc1) + (acc2 + acc3).
// K = 640 in 5 chunks of 4 groups, cp.async double buffered.
constexpr unsigned kDownThreads = 256U, kDownOuts = 64U, kDAStride = 144U;
template <bool Q51> struct DownLayout {
    static constexpr unsigned WB = Q51 ? 24U : 34U, WS = Q51 ? 112U : 144U;  // raw bytes / smem stride per output
    static constexpr unsigned a_bytes = tile_rows * kDAStride, w_bytes = kDownOuts * WS, es_bytes = tile_rows * 32U;
    static constexpr unsigned stage = a_bytes + w_bytes + es_bytes;
    static constexpr unsigned scales = 2U * stage;   // D[4][64], M[4][64], E[4][64], S[4][64]
    static constexpr unsigned rowinfo = scales + 4U * 4U * 64U * 4U;
    static constexpr unsigned total = rowinfo + 2U * tile_rows * 4U;
};

template <bool Q51>
__global__ __launch_bounds__(kDownThreads) void down_kernel(const std::uint8_t* weights, Rows v) {
    using L = DownLayout<Q51>;
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned begin = csr.tile_begins[blockIdx.x], count = csr.tile_begins[blockIdx.x + 1U] - begin;
    const unsigned expert = csr.tile_experts[blockIdx.x], out0 = blockIdx.y * kDownOuts;
    extern __shared__ __align__(16) unsigned char smem[];
    float* Dv = reinterpret_cast<float*>(smem + L::scales);  // [j][o]
    float* Mv = Dv + 4U * 64U;
    float* E = Mv + 4U * 64U;                                // [j][row]
    float* S = E + 4U * 64U;
    unsigned* row_of = reinterpret_cast<unsigned*>(smem + L::rowinfo);
    unsigned* slot_of = row_of + tile_rows;
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < tile_rows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    constexpr unsigned groups = ffn / 32U;
    const auto load = [&](unsigned c) {
        unsigned char* st = smem + (c & 1U) * L::stage;
        for (unsigned i = t; i < tile_rows * 8U; i += kDownThreads) {
            const unsigned r = i >> 3, k = i & 7U;
            cp16(st + r * kDAStride + 16U * k, at(v.down_codes, row_of[r]) + slot_of[r] * ffn + 128U * c + 16U * k);
        }
        if constexpr (Q51) {
            for (unsigned i = t; i < kDownOuts * 6U; i += kDownThreads) {
                const unsigned o = i / 6U, k = i % 6U;
                const std::uint8_t* w = weights + ((static_cast<std::uint64_t>(expert) * hidden + out0 + o) * groups + 4U * c) * L::WB;
                cp16(st + L::a_bytes + o * L::WS + 16U * k, w + 16U * k);
            }
        } else {
            for (unsigned i = t; i < kDownOuts * 17U; i += kDownThreads) {
                const unsigned o = i / 17U, k = i % 17U;
                const std::uint8_t* w = weights + ((static_cast<std::uint64_t>(expert) * hidden + out0 + o) * groups + 4U * c) * L::WB;
                cp8(st + L::a_bytes + o * L::WS + 8U * k, w + 8U * k);
            }
        }
        for (unsigned i = t; i < tile_rows * 2U; i += kDownThreads) {
            const unsigned r = i >> 1, k = i & 1U;
            const void* src = k == 0U ? static_cast<const void*>(at(v.down_deltas, row_of[r]) + slot_of[r] * groups + 4U * c)
                                      : static_cast<const void*>(at(v.down_sums, row_of[r]) + slot_of[r] * groups + 4U * c);
            cp16(st + L::a_bytes + L::w_bytes + r * 32U + 16U * k, src);
        }
        cp_commit();
    };
    const unsigned g = lane >> 2, q = lane & 3U, wm = warp & 3U, wn = warp >> 2;
    const bool active = 16U * wm < count;
    float acc[4][4][4];
#pragma unroll
    for (unsigned n = 0; n < 4U; ++n)
#pragma unroll
        for (unsigned e = 0; e < 4U; ++e)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) acc[n][e][j] = 0.0F;
    constexpr unsigned chunks = groups / 4U;
    load(0U);
#pragma unroll 1
    for (unsigned c = 0; c < chunks; ++c) {
        cp_wait_all();
        __syncthreads();
        if (c + 1U < chunks) load(c + 1U);
        const unsigned char* st = smem + (c & 1U) * L::stage;
        const unsigned char* A = st;
        const unsigned char* W = st + L::a_bytes;
        const unsigned char* ES = st + L::a_bytes + L::w_bytes;
        for (unsigned i = t; i < 4U * 64U; i += kDownThreads) {
            const unsigned j = i >> 6, o = i & 63U;
            const unsigned char* blk = W + o * L::WS + j * L::WB;
            const unsigned short* h = reinterpret_cast<const unsigned short*>(blk);
            Dv[i] = half_bits(h[0]);
            Mv[i] = Q51 ? half_bits(h[1]) : 0.0F;
            E[i] = reinterpret_cast<const float*>(ES + o * 32U)[j];
            S[i] = static_cast<float>(reinterpret_cast<const int*>(ES + o * 32U + 16U)[j]);
        }
        __syncthreads();
        if (!active) continue;
        const unsigned r0 = 16U * wm + g;
        const unsigned char* A0 = A + r0 * kDAStride;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            unsigned a[4];
            a[0] = lds32(A0 + 32U * j + 4U * q); a[1] = lds32(A0 + 8U * kDAStride + 32U * j + 4U * q);
            a[2] = lds32(A0 + 32U * j + 16U + 4U * q); a[3] = lds32(A0 + 8U * kDAStride + 32U * j + 16U + 4U * q);
            const float e0 = E[j * 64U + r0], e1 = E[j * 64U + r0 + 8U];
            const float s0 = S[j * 64U + r0], s1 = S[j * 64U + r0 + 8U];
#pragma unroll
            for (unsigned n = 0; n < 4U; ++n) {
                const unsigned ob = 32U * wn + 8U * n;
                const unsigned char* blk = W + (ob + g) * L::WS + j * L::WB;
                unsigned b0, b1;
                if constexpr (Q51) {
                    const unsigned qh = lds32(blk + 4U), qs = lds32(blk + 8U + 4U * q);
                    const unsigned l = (qh >> (4U * q)) & 15U, u = (qh >> (16U + 4U * q)) & 15U;
                    b0 = (qs & 0x0f0f0f0fU) | ((l * 0x00204081U) & 0x01010101U) << 4U;
                    b1 = ((qs >> 4U) & 0x0f0f0f0fU) | ((u * 0x00204081U) & 0x01010101U) << 4U;
                } else {
                    // quants at blk + 2 + k: 4-byte aligned when 34*j + 2 is (j odd), else two words and a byte select
                    const unsigned base = j * L::WB + 2U;  // offset inside the output's smem row (row stride is a multiple of 4)
                    const unsigned char* row = W + (ob + g) * L::WS;
                    if (base & 3U) {
                        const unsigned al = base & ~3U;
                        const unsigned x0 = lds32(row + al + 4U * q), x1 = lds32(row + al + 4U * q + 4U);
                        const unsigned y0 = lds32(row + al + 16U + 4U * q), y1 = lds32(row + al + 16U + 4U * q + 4U);
                        b0 = __byte_perm(x0, x1, 0x5432U); b1 = __byte_perm(y0, y1, 0x5432U);
                    } else {
                        b0 = lds32(row + base + 4U * q); b1 = lds32(row + base + 16U + 4U * q);
                    }
                }
                int r[4];
                mma_magic(a, b0, b1, r);
                const unsigned oc = ob + 2U * q;
                const float2 dd = *reinterpret_cast<const float2*>(Dv + j * 64U + oc);
                const float2 mm = *reinterpret_cast<const float2*>(Mv + j * 64U + oc);
#pragma unroll
                for (unsigned el = 0; el < 4U; ++el) {
                    const unsigned hr = el >> 1, col = el & 1U;
                    const float first = __fmul_rn(col ? dd.y : dd.x, group_float(r[el]));
                    float combined = first;
                    if constexpr (Q51) combined = __fadd_rn(first, __fmul_rn(col ? mm.y : mm.x, hr ? s1 : s0));
                    const float contribution = __fmul_rn(hr ? e1 : e0, combined);
                    acc[n][el][j] = __fadd_rn(acc[n][el][j], contribution);
                }
            }
        }
    }
    if (!active) return;
#pragma unroll
    for (unsigned n = 0; n < 4U; ++n)
#pragma unroll
        for (unsigned hr = 0; hr < 2U; ++hr) {
            const unsigned r = 16U * wm + g + 8U * hr;
            if (r >= count) continue;
            float out[2];
#pragma unroll
            for (unsigned col = 0; col < 2U; ++col) {
                const float* c = acc[n][2U * hr + col];
                out[col] = __fadd_rn(__fadd_rn(c[0], c[1]), __fadd_rn(c[2], c[3]));
            }
            const unsigned o = out0 + 32U * wn + 8U * n + 2U * q;
            *reinterpret_cast<float2*>(at(v.rawdots, row_of[r]) + slot_of[r] * hidden + o) = make_float2(out[0], out[1]);
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode shapes (rows <= 8, so a tile holds <= 8 assignments of one expert). Same per-element chains as project_kernel /
// down_kernel; the exact integer group dots come from dp4a over lane groups (integer adds, order-free).
// STRIX_QRTX_DECODE_KERNELS=0 keeps the tiled kernels (reference seam).
bool decode_rows_enabled() {
    static const bool on = [] { const char* e = std::getenv("STRIX_QRTX_DECODE_KERNELS"); return !(e && e[0] == '0'); }();
    return on;
}
constexpr unsigned kRowsMax = 8U;

// gate/up: block = (tile, 32 outputs), warp w = output w, gate and up chains in the same loop (their loads overlap). Lane L: pair p = L / 8, group of the
// pair G = (L / 4) % 2 (group j = 2p + G), elements 8 (L % 4) .. + 7 of group j. Per K block b: u = D_j * float(dot),
// v = DM_j * S, t = u - v, x = E * t; pair = x(G=0) + x(G=1); acc[p] += pair (p = this lane's pair); then
// (acc0 + acc1) + (acc2 + acc3). Then the SwiGLU + down activation quantizer of project_kernel per assignment.
template <bool Q5K>
__global__ __launch_bounds__(1024) void project_rows_kernel(const std::uint8_t* gate, const std::uint8_t* up, Rows v,
                                                           amdmath::Tables tables) {
    using L = ProjLayout<Q5K>;
    const Csr& csr = v.csr;
    if ((csr.summary[0] & ~projection_nonfinite) || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], total = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    // a tile holds <= 8 assignments when every route of a row is distinct; any more run in further chunks of 8
    for (unsigned base = 0; base < total; base += kRowsMax) {
    const unsigned begin = tile_begin + base, count = min(total - base, kRowsMax);
    __syncthreads();  // the previous chunk finished with the shared buffers
    const unsigned expert = csr.tile_experts[blockIdx.x], out0 = blockIdx.y * 32U;
    __shared__ __align__(16) std::uint32_t A[kRowsMax][hidden / 4U];
    __shared__ float E[kRowsMax][hidden / 32U], S[kRowsMax][hidden / 32U];
    __shared__ float GU[2][kRowsMax][32];
    __shared__ unsigned row_of[kRowsMax], slot_of[kRowsMax];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < kRowsMax) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    for (unsigned i = t; i < count * (hidden / 4U); i += 1024U) {
        const unsigned a = i / (hidden / 4U), w = i % (hidden / 4U);
        A[a][w] = reinterpret_cast<const std::uint32_t*>(at(v.codes, row_of[a]))[w];
    }
    for (unsigned i = t; i < count * (hidden / 32U); i += 1024U) {
        const unsigned a = i / (hidden / 32U), g = i % (hidden / 32U);
        E[a][g] = at(v.deltas, row_of[a])[g];
        S[a][g] = static_cast<float>(at(v.sums, row_of[a])[g]);
    }
    __syncthreads();
    const unsigned p = lane >> 3, G = (lane >> 2) & 1U, sub = lane & 3U, j = 2U * p + G;
    constexpr unsigned blocks = hidden / 256U;
    const unsigned o = out0 + warp;
    const std::uint8_t* wrow[2] = {gate + (static_cast<std::uint64_t>(expert) * ffn + o) * blocks * L::WB,
                                   up + (static_cast<std::uint64_t>(expert) * ffn + o) * blocks * L::WB};
    float acc[2][kRowsMax];
#pragma unroll
    for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
        for (unsigned a = 0; a < kRowsMax; ++a) acc[m][a] = 0.0F;
#pragma unroll 2
    for (unsigned b = 0; b < blocks; ++b) {
        uint4 h[2];
        unsigned q0[2], q1[2], qh0[2] = {0U, 0U}, qh1[2] = {0U, 0U};
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) {
            const std::uint8_t* blk = wrow[m] + b * L::WB;
            h[m] = *reinterpret_cast<const uint4*>(blk);
            const unsigned* qs = reinterpret_cast<const unsigned*>(blk + L::QS + 32U * p + 8U * sub);
            q0[m] = qs[0]; q1[m] = qs[1];
            if constexpr (Q5K) {
                const unsigned* qh = reinterpret_cast<const unsigned*>(blk + 16U + 8U * sub);
                qh0[m] = qh[0]; qh1[m] = qh[1];
            }
        }
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) {
            const float D = __fmul_rn(half_bits(h[m].x & 65535U), static_cast<float>(factor(h[m], j, false)));
            const float DM = __fmul_rn(half_bits(h[m].x >> 16U), static_cast<float>(factor(h[m], j, true)));
            unsigned w0 = G ? (q0[m] >> 4U) & 0x0f0f0f0fU : q0[m] & 0x0f0f0f0fU;
            unsigned w1 = G ? (q1[m] >> 4U) & 0x0f0f0f0fU : q1[m] & 0x0f0f0f0fU;
            if constexpr (Q5K) {
                w0 |= ((qh0[m] >> j) & 0x01010101U) << 4U;
                w1 |= ((qh1[m] >> j) & 0x01010101U) << 4U;
            }
#pragma unroll
            for (unsigned a = 0; a < kRowsMax; ++a) {
                if (a >= count) break;
                const unsigned* x = &A[a][(256U * b + 32U * j + 8U * sub) / 4U];
                int dot = __dp4a(static_cast<int>(w0), static_cast<int>(x[0]), 0);
                dot = __dp4a(static_cast<int>(w1), static_cast<int>(x[1]), dot);
                dot += __shfl_xor_sync(0xffffffffU, dot, 1);
                dot += __shfl_xor_sync(0xffffffffU, dot, 2);
                const float u = __fmul_rn(D, static_cast<float>(dot));
                const float vv = __fmul_rn(DM, S[a][8U * b + j]);
                const float tt = __fsub_rn(u, vv);
                const float xx = __fmul_rn(E[a][8U * b + j], tt);
                const float other = __shfl_xor_sync(0xffffffffU, xx, 4);
                const float pair = G ? __fadd_rn(other, xx) : __fadd_rn(xx, other);
                acc[m][a] = __fadd_rn(acc[m][a], pair);
            }
        }
    }
#pragma unroll
    for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
        for (unsigned a = 0; a < kRowsMax; ++a) {
            if (a >= count) break;
            const float c0 = __shfl_sync(0xffffffffU, acc[m][a], 0), c1 = __shfl_sync(0xffffffffU, acc[m][a], 8);
            const float c2 = __shfl_sync(0xffffffffU, acc[m][a], 16), c3 = __shfl_sync(0xffffffffU, acc[m][a], 24);
            if (lane == 0U) GU[m][a][warp] = __fadd_rn(__fadd_rn(c0, c1), __fadd_rn(c2, c3));
        }
    __syncthreads();
    // down_activation_quantize per (assignment, 32-output group), as project_kernel's epilogue
    const bool write_intermediates = v.intermediates.base != nullptr;
    for (unsigned a = warp; a < count; a += 32U) {
        const unsigned r = row_of[a], slot = slot_of[a];
        const unsigned feature = out0 + lane, at_slot = slot * 2U * ffn + feature;
        const float gv = GU[0][a][lane], uv = GU[1][a][lane];
        const float ex = amdmath::amd_expf(tables, -gv);
        const float hv = __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0F, ex)), uv);
        float* output = write_intermediates ? at(v.intermediates, r) : nullptr;
        if (output) output[at_slot + ffn] = canon(uv);
        const unsigned bad = __ballot_sync(0xffffffffU, !finite(gv) || !finite(uv) || !finite(hv));
        if (bad != 0U) {
            if (output) output[at_slot] = canon(gv);
            if (lane == 0) { atomicOr(at(v.status, r), projection_nonfinite); atomicOr(csr.summary, projection_nonfinite); }
            continue;
        }
        float x = fabsf(hv);
        for (int sh = 16; sh != 0; sh /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffffU, x, sh));
        const bool tiny = x > 0.0F && x < 0x1p-119F;
        const float delta = x == 0.0F || tiny ? 1.0F : __fdiv_rn(x, 127.0F);
        const float rounded = tiny ? 0.0F : rintf(__fdiv_rn(hv, delta));
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int sh = 16; sh != 0; sh /= 2) sum += __shfl_xor_sync(0xffffffffU, sum, sh);
        if (output) output[at_slot] = hv;
        at(v.down_codes, r)[slot * ffn + feature] = static_cast<std::int8_t>(code);
        if (lane == 0) { at(v.down_deltas, r)[slot * (ffn / 32U) + feature / 32U] = delta; at(v.down_sums, r)[slot * (ffn / 32U) + feature / 32U] = sum; }
    }
    }
}

// down: block = (tile, 8 hidden outputs), warp = one output. Lane L: chain j = L / 8, elements 4 (L % 8) .. + 3 of
// group g = 4c + j, chunks c ascending: first = d * float(dot), Q5_1: combined = first + m * S, contribution = E *
// combined, acc_j += contribution; then (acc0 + acc1) + (acc2 + acc3) into rawdots.
template <bool Q51>
__global__ __launch_bounds__(256) void down_rows_kernel(const std::uint8_t* weights, Rows v) {
    using L = DownLayout<Q51>;
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], total = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    // a tile holds <= 8 assignments when every route of a row is distinct; any more run in further chunks of 8
    for (unsigned base = 0; base < total; base += kRowsMax) {
    const unsigned begin = tile_begin + base, count = min(total - base, kRowsMax);
    __syncthreads();  // the previous chunk finished with the shared buffers
    const unsigned expert = csr.tile_experts[blockIdx.x];
    constexpr unsigned groups = ffn / 32U;
    __shared__ __align__(16) std::uint32_t A[kRowsMax][ffn / 4U];
    __shared__ float E[kRowsMax][groups], S[kRowsMax][groups];
    __shared__ unsigned row_of[kRowsMax], slot_of[kRowsMax];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < kRowsMax) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    for (unsigned i = t; i < count * (ffn / 4U); i += 256U) {
        const unsigned a = i / (ffn / 4U), w = i % (ffn / 4U);
        A[a][w] = reinterpret_cast<const std::uint32_t*>(at(v.down_codes, row_of[a]) + slot_of[a] * ffn)[w];
    }
    for (unsigned i = t; i < count * groups; i += 256U) {
        const unsigned a = i / groups, g = i % groups;
        E[a][g] = at(v.down_deltas, row_of[a])[slot_of[a] * groups + g];
        S[a][g] = static_cast<float>(at(v.down_sums, row_of[a])[slot_of[a] * groups + g]);
    }
    __syncthreads();
    const unsigned o = blockIdx.y * 8U + warp, j = lane >> 3, sub = lane & 7U;
    const std::uint8_t* wrow = weights + (static_cast<std::uint64_t>(expert) * hidden + o) * groups * L::WB;
    float acc[kRowsMax];
#pragma unroll
    for (unsigned a = 0; a < kRowsMax; ++a) acc[a] = 0.0F;
#pragma unroll
    for (unsigned c = 0; c < groups / 4U; ++c) {
        const unsigned g = 4U * c + j;
        const std::uint8_t* blk = wrow + g * L::WB;
        const unsigned short* hw = reinterpret_cast<const unsigned short*>(blk);
        const float d = half_bits(hw[0]);
        unsigned word;
        float mm = 0.0F;
        if constexpr (Q51) {
            mm = half_bits(hw[1]);
            const unsigned qh = *reinterpret_cast<const unsigned*>(blk + 4U);
            const unsigned qs = *reinterpret_cast<const unsigned*>(blk + 8U + 4U * (sub & 3U));
            const unsigned hb = (qh >> (4U * sub)) & 15U;
            word = (sub < 4U ? qs & 0x0f0f0f0fU : (qs >> 4U) & 0x0f0f0f0fU) | ((hb * 0x00204081U) & 0x01010101U) << 4U;
        } else {
            word = static_cast<unsigned>(hw[1U + 2U * sub]) | (static_cast<unsigned>(hw[2U + 2U * sub]) << 16U);
        }
#pragma unroll
        for (unsigned a = 0; a < kRowsMax; ++a) {
            if (a >= count) break;
            int dot = __dp4a(static_cast<int>(word), static_cast<int>(A[a][8U * g + sub]), 0);
            dot += __shfl_xor_sync(0xffffffffU, dot, 1);
            dot += __shfl_xor_sync(0xffffffffU, dot, 2);
            dot += __shfl_xor_sync(0xffffffffU, dot, 4);
            const float first = __fmul_rn(d, static_cast<float>(dot));
            float combined = first;
            if constexpr (Q51) combined = __fadd_rn(first, __fmul_rn(mm, S[a][g]));
            acc[a] = __fadd_rn(acc[a], __fmul_rn(E[a][g], combined));
        }
    }
#pragma unroll
    for (unsigned a = 0; a < kRowsMax; ++a) {
        if (a >= count) break;
        const float c0 = __shfl_sync(0xffffffffU, acc[a], 0), c1 = __shfl_sync(0xffffffffU, acc[a], 8);
        const float c2 = __shfl_sync(0xffffffffU, acc[a], 16), c3 = __shfl_sync(0xffffffffU, acc[a], 24);
        if (lane == 0U) at(v.rawdots, row_of[a])[slot_of[a] * hidden + o] = __fadd_rn(__fadd_rn(c0, c1), __fadd_rn(c2, c3));
    }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Native expert types (the GSQ-RCO IQ file: gate/up IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS, down IQ4_NL / Q2_0; also any other
// cuda_qwen_quant.cuh type). FAST arithmetic, not a node profile: FP32 activations (BF16 input rows, FP32 h), weights
// decoded per 32-value group with the shared decoders (bitwise the host reference values), FP32 dots, SwiGLU with expf. A warp owns
// one output; each lane decodes its groups once per chunk of up to 8 assignments and applies them to every assignment.
constexpr unsigned kNativeChunk = 8U;
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (unsigned d = 16U; d; d >>= 1U) v += __shfl_xor_sync(0xffffffffU, v, d);
    return v;
}
// grid (tiles, ffn / 32), 1024 threads: warp w = output 32 * blockIdx.y + w, gate and up. Writes h = silu(g) * u into
// intermediates[row][slot * 2 * ffn + o].
__global__ __launch_bounds__(1024) void native_project_kernel(const std::uint8_t* gate, const std::uint8_t* up, std::uint32_t type,
                                                              Rows v) {
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], total = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[blockIdx.x];
    __shared__ std::uint16_t X[kNativeChunk][hidden];  // BF16 input rows
    __shared__ unsigned row_of[kNativeChunk], slot_of[kNativeChunk];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, o = blockIdx.y * 32U + warp;
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(hidden) / quant::geometry(type).block_elements * quant::geometry(type).block_bytes;
    const std::uint8_t* wg = gate + (static_cast<std::uint64_t>(expert) * ffn + o) * row_bytes;
    const std::uint8_t* wu = up + (static_cast<std::uint64_t>(expert) * ffn + o) * row_bytes;
    for (unsigned base = 0; base < total; base += kNativeChunk) {
        const unsigned count = min(total - base, kNativeChunk);
        __syncthreads();
        if (t < kNativeChunk) {
            const unsigned a = t < count ? t : 0U;
            row_of[t] = csr.rows[tile_begin + base + a]; slot_of[t] = csr.slots[tile_begin + base + a];
        }
        __syncthreads();
        for (unsigned i = t; i < count * hidden; i += 1024U) {
            const unsigned a = i / hidden, f = i % hidden;
            X[a][f] = at(v.input, row_of[a])[f];
        }
        __syncthreads();
        float g[kNativeChunk], u[kNativeChunk];
#pragma unroll
        for (unsigned a = 0; a < kNativeChunk; ++a) { g[a] = 0.0F; u[a] = 0.0F; }
        for (unsigned grp = lane; grp < hidden / 32U; grp += 32U) {
            float wv[32];
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m) {
                quant::decode_row_group(type, m ? wu : wg, grp, wv);
#pragma unroll
                for (unsigned a = 0; a < kNativeChunk; ++a) {
                    if (a >= count) break;
                    const std::uint16_t* x = &X[a][32U * grp];
                    float sacc = 0.0F;
#pragma unroll
                    for (unsigned j = 0; j < 32U; ++j) sacc = __fmaf_rn(wv[j], __uint_as_float(static_cast<unsigned>(x[j]) << 16U), sacc);
                    if (m) u[a] += sacc; else g[a] += sacc;
                }
            }
        }
#pragma unroll
        for (unsigned a = 0; a < kNativeChunk; ++a) {
            if (a >= count) break;
            const float gv = warp_sum(g[a]), uv = warp_sum(u[a]);
            if (lane == 0U) {
                const float h = gv / (1.0F + expf(-gv)) * uv;
                at(v.intermediates, row_of[a])[slot_of[a] * 2U * ffn + o] = h;
                if (!isfinite(h)) { atomicOr(at(v.status, row_of[a]), projection_nonfinite); atomicOr(csr.summary, projection_nonfinite); }
            }
        }
    }
}
// grid (tiles, hidden / 32), 1024 threads: warp w = hidden output 32 * blockIdx.y + w; lanes 0..19 own the 20 groups of
// 640. Writes rawdots[row][slot * hidden + o].
__global__ __launch_bounds__(1024) void native_down_kernel(const std::uint8_t* down, std::uint32_t type, Rows v) {
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], total = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[blockIdx.x];
    __shared__ float H[kNativeChunk][ffn];
    __shared__ unsigned row_of[kNativeChunk], slot_of[kNativeChunk];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, o = blockIdx.y * 32U + warp;
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(ffn) / quant::geometry(type).block_elements * quant::geometry(type).block_bytes;
    const std::uint8_t* w = down + (static_cast<std::uint64_t>(expert) * hidden + o) * row_bytes;
    for (unsigned base = 0; base < total; base += kNativeChunk) {
        const unsigned count = min(total - base, kNativeChunk);
        __syncthreads();
        if (t < kNativeChunk) {
            const unsigned a = t < count ? t : 0U;
            row_of[t] = csr.rows[tile_begin + base + a]; slot_of[t] = csr.slots[tile_begin + base + a];
        }
        __syncthreads();
        for (unsigned i = t; i < count * ffn; i += 1024U) {
            const unsigned a = i / ffn, f = i % ffn;
            H[a][f] = at(v.intermediates, row_of[a])[slot_of[a] * 2U * ffn + f];
        }
        __syncthreads();
        float acc[kNativeChunk];
#pragma unroll
        for (unsigned a = 0; a < kNativeChunk; ++a) acc[a] = 0.0F;
        if (lane < ffn / 32U) {
            float wv[32];
            quant::decode_row_group(type, w, lane, wv);
#pragma unroll
            for (unsigned a = 0; a < kNativeChunk; ++a) {
                if (a >= count) break;
                float sacc = 0.0F;
#pragma unroll
                for (unsigned j = 0; j < 32U; ++j) sacc = __fmaf_rn(wv[j], H[a][32U * lane + j], sacc);
                acc[a] = sacc;
            }
        }
#pragma unroll
        for (unsigned a = 0; a < kNativeChunk; ++a) {
            if (a >= count) break;
            const float d = warp_sum(acc[a]);
            if (lane == 0U) at(v.rawdots, row_of[a])[slot_of[a] * hidden + o] = d;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// merge / commit (hip_moe_experts_launcher.hip:2660-2708).
__global__ __launch_bounds__(256) void merge_kernel(Rows v) {
    const unsigned row = blockIdx.y;
    std::uint32_t* status = at(v.status, row);
    std::uint32_t* header = v.csr.summary;
    if (header[0] || *status) return;
    const unsigned out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= hidden) return;
    const std::uint32_t* routes = at(v.routes, row);
    for (unsigned slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) return;
    const float* rawdots = at(v.rawdots, row);
    const float* w = at(v.route_weights, row);
    float current = 0.0F;
#pragma unroll
    for (unsigned slot = 0; slot < top_k; ++slot) current = __fmaf_rn(w[slot], rawdots[slot * hidden + out], current);
    at(v.accumulation, row)[out] = canon(current);
    if ((__float_as_uint(current) & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); }
}

__global__ __launch_bounds__(256) void commit_kernel(Rows v, std::uint32_t rows, std::uint32_t* record) {
    std::uint32_t* header = v.csr.summary;
    const std::uint32_t failure = header[0];
    if (blockIdx.x == 0U && blockIdx.y == 0U) {
        __shared__ std::uint32_t first_bad;
        if (threadIdx.x == 0U) first_bad = rows;
        __syncthreads();
        for (std::uint32_t r = threadIdx.x; r < rows; r += blockDim.x)
            if (*at(v.status, r)) { atomicMin(&first_bad, r); break; }
        __syncthreads();
        if (threadIdx.x == 0U) {
            if (failure & 7U) { for (const unsigned word : {1U, 2U, 4U, 5U}) header[word] = 0U; for (unsigned word = 8U; word < 24U; ++word) header[word] = 0U; }
            if (first_bad < rows && record) {
                const std::uint32_t bits = *at(v.status, first_bad);
                if (bits != 0U && record[0] == 0U) { record[1] = bits; record[0] = 2U; }  // DeferredMoeDomain::q4k_a8
            }
        }
    }
    if (failure) return;
    const unsigned out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out < hidden) {
        const std::uint32_t bits = __float_as_uint(at(v.accumulation, blockIdx.y)[out]);
        at(v.output, blockIdx.y)[out] = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
    }
}

template <class K> void set_smem(K kernel, unsigned bytes) {
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)), "MoE smem attribute");
}

}  // namespace

CsrSizes csr_sizes(std::uint32_t rows) {
    const std::uint32_t a = rows * top_k;
    const auto bound = [&](std::uint32_t cap) { const std::uint32_t x = experts + (a + cap - 1U) / cap; return x < a ? x : a; };
    return {a, bound(hip_batch), bound(tile_rows)};
}

void enqueue_prologue(const Rows& v, std::uint32_t rows, std::uint32_t clear_words, cudaStream_t s) {
    prologue_kernel<<<rows, 256, 0, s>>>(v, clear_words);
    check(cudaGetLastError(), "MoE prologue");
}
void enqueue_csr(const Rows& v, std::uint32_t rows, cudaStream_t s) {
    if (rows * top_k > 65535U * 32U) throw std::invalid_argument("MoE CSR rows");
    csr_kernel<<<1, 1024, 0, s>>>(v, rows);
    check(cudaGetLastError(), "MoE CSR");
}
void enqueue_project(const Rows& v, std::uint32_t rows, const Weights& w, const amdmath::Tables& tables, cudaStream_t s) {
    if (rows <= kRowsMax && decode_rows_enabled()) {
        const dim3 grid(csr_sizes(rows).max_tiles, ffn / 32U);
        if (w.gate_type == type_q5_k) project_rows_kernel<true><<<grid, 1024, 0, s>>>(w.gate, w.up, v, tables);
        else if (w.gate_type == type_q4_k) project_rows_kernel<false><<<grid, 1024, 0, s>>>(w.gate, w.up, v, tables);
        else throw std::invalid_argument("MoE gate type");
        check(cudaGetLastError(), "MoE gate/up rows");
        return;
    }
    const dim3 grid(csr_sizes(rows).max_tiles, ffn / kProjOuts);
    if (w.gate_type == type_q5_k) {
        static bool once = (set_smem(project_kernel<true>, ProjLayout<true>::total), true); (void)once;
        project_kernel<true><<<grid, kProjThreads, ProjLayout<true>::total, s>>>(w.gate, w.up, v, tables);
    } else if (w.gate_type == type_q4_k) {
        static bool once = (set_smem(project_kernel<false>, ProjLayout<false>::total), true); (void)once;
        project_kernel<false><<<grid, kProjThreads, ProjLayout<false>::total, s>>>(w.gate, w.up, v, tables);
    } else throw std::invalid_argument("MoE gate type");
    check(cudaGetLastError(), "MoE gate/up");
}
void enqueue_down(const Rows& v, std::uint32_t rows, const Weights& w, cudaStream_t s) {
    if (rows <= kRowsMax && decode_rows_enabled()) {
        const dim3 grid(csr_sizes(rows).max_tiles, hidden / 8U);
        if (w.down_type == type_q5_1) down_rows_kernel<true><<<grid, 256, 0, s>>>(w.down, v);
        else if (w.down_type == type_q8_0) down_rows_kernel<false><<<grid, 256, 0, s>>>(w.down, v);
        else throw std::invalid_argument("MoE down type");
        check(cudaGetLastError(), "MoE down rows");
        return;
    }
    const dim3 grid(csr_sizes(rows).max_tiles, hidden / kDownOuts);
    if (w.down_type == type_q5_1) {
        static bool once = (set_smem(down_kernel<true>, DownLayout<true>::total), true); (void)once;
        down_kernel<true><<<grid, kDownThreads, DownLayout<true>::total, s>>>(w.down, v);
    } else if (w.down_type == type_q8_0) {
        static bool once = (set_smem(down_kernel<false>, DownLayout<false>::total), true); (void)once;
        down_kernel<false><<<grid, kDownThreads, DownLayout<false>::total, s>>>(w.down, v);
    } else throw std::invalid_argument("MoE down type");
    check(cudaGetLastError(), "MoE down");
}
void enqueue_merge(const Rows& v, std::uint32_t rows, cudaStream_t s) {
    merge_kernel<<<dim3((hidden + 255U) / 256U, rows), 256, 0, s>>>(v);
    check(cudaGetLastError(), "MoE merge");
}
void enqueue_commit(const Rows& v, std::uint32_t rows, std::uint32_t* record, cudaStream_t s) {
    commit_kernel<<<dim3((hidden + 255U) / 256U, rows), 256, 0, s>>>(v, rows, record);
    check(cudaGetLastError(), "MoE commit");
}
bool native_types(const Weights& w) {
    const bool a8_gate = w.gate_type == type_q4_k || w.gate_type == type_q5_k;
    const bool a8_down = w.down_type == type_q5_1 || w.down_type == type_q8_0;
    return !(a8_gate && a8_down);
}

void enqueue(const Rows& v, std::uint32_t rows, std::uint32_t clear_words, const Weights& w, const amdmath::Tables& tables,
             std::uint32_t* record, cudaStream_t s) {
    enqueue_prologue(v, rows, clear_words, s);
    enqueue_csr(v, rows, s);
    if (native_types(w)) {
        if (!v.intermediates.base) throw std::invalid_argument("MoE native expert types need Rows::intermediates");
        if (quant::geometry(w.gate_type).block_elements == 0 || quant::geometry(w.down_type).block_elements == 0)
            throw std::invalid_argument("MoE expert type without a decoder");
        const unsigned tiles = csr_sizes(rows).max_tiles;
        native_project_kernel<<<dim3(tiles, ffn / 32U), 1024, 0, s>>>(w.gate, w.up, w.gate_type, v);
        check(cudaGetLastError(), "MoE native gate/up");
        native_down_kernel<<<dim3(tiles, hidden / 32U), 1024, 0, s>>>(w.down, w.down_type, v);
        check(cudaGetLastError(), "MoE native down");
        enqueue_merge(v, rows, s);
        enqueue_commit(v, rows, record, s);
        return;
    }
    enqueue_project(v, rows, w, tables, s);
    enqueue_down(v, rows, w, s);
    enqueue_merge(v, rows, s);
    enqueue_commit(v, rows, record, s);
}

}  // namespace strix::cuda::qwen_moe
