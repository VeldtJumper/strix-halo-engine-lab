// CUDA port of the Qwen PLE step and MTP predictor cache rows (see include/strix/cuda_qwen_plemtp.hpp).
// Exactness: every HIP floating-point instruction is one explicit IEEE operation here (__fmaf_rn/__fmul_rn/__fadd_rn/
// __fdiv_rn/__fsqrt_rn) with the operand order of the gfx1151 ISA (disassembly of hip_ple_launcher);
// rsqrtf/expf are the gfx1151 emulations of qwen_amd_math.cuh.
#include "strix/cuda_qwen_plemtp.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::cuda::qwen_plemtp {
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float bf16f(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16U); }
__host__ __device__ __forceinline__ std::uint16_t f2bf(float v) {
#ifdef __CUDA_ARCH__
    const std::uint32_t bits = __float_as_uint(v);
#else
    std::uint32_t bits; std::memcpy(&bits, &v, 4);
#endif
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

// ---------------------------------------------------------------------------------------------------------------
// Host: hash contract and 3-gram hashing (literal port of src/ple_lookup.cpp:15-256)
constexpr std::uint64_t kGamma = 0x9E3779B97F4A7C15ULL, kM1 = 0xBF58476D1CE4E5B9ULL, kM2 = 0x94D049BB133111EBULL;
std::uint64_t splitmix64(std::uint64_t v) {
    v = v + kGamma;
    v = (v ^ (v >> 30U)) * kM1;
    v = (v ^ (v >> 27U)) * kM2;
    return v ^ (v >> 31U);
}
bool is_prime(std::uint64_t v) {
    if (v < 2) return false;
    if (v % 2ULL == 0ULL) return v == 2;
    for (std::uint64_t d = 3; d <= v / d; d += 2) if (v % d == 0ULL) return false;
    return true;
}
std::uint64_t next_prime(std::uint64_t v) { ++v; while (!is_prime(v)) ++v; return v; }

}  // namespace

PleHashContract ple_hash_contract() {
    constexpr std::uint64_t vocab = 248320, seed = 1234, base = 20000000;
    PleHashContract c;
    const auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    const auto half_bound = std::max<std::uint64_t>(1ULL, maximum / vocab / 2ULL);
    for (std::uint32_t i = 0; i < kNgram; ++i)
        c.multipliers[i] = 2ULL * (splitmix64(seed + kGamma * static_cast<std::uint64_t>(i + 1U)) % half_bound) + 1ULL;
    std::uint64_t prime = base - 1ULL, offset = 0;
    for (std::uint32_t h = 0; h < kPleHeads; ++h) {
        prime = next_prime(prime);
        c.head_vocab_sizes[h] = prime; c.head_offsets[h] = offset; offset += prime;
    }
    c.minimum_table_rows = offset;
    return c;
}

void ple_hash_rows(const PleHashContract& c, const std::uint32_t* tokens, std::uint32_t rows, std::uint64_t position0,
    std::uint32_t history[2], std::uint64_t* row_ids) {
    // hip_ple_launcher.hip:2493-2497: position 0 starts from {EOS, EOS}, otherwise the n-gram history word.
    std::uint32_t prev[2] = {history[0], history[1]};
    if (position0 == 0) prev[0] = prev[1] = kPleEos;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::uint32_t token = tokens[r];
        std::uint64_t ctx[kNgram];
        ctx[0] = token;
        bool cut = false;
        for (std::uint32_t s = 1; s < kNgram; ++s) {   // context[s] = previous[len - s]; EOS cuts it and every older slot
            const std::uint32_t t = prev[2U - s];
            cut = cut || t == kPleEos;
            ctx[s] = cut ? kPleEos : t;
        }
        for (std::uint32_t n = 2; n <= kNgram; ++n) {
            std::uint64_t mixed = ctx[0] * c.multipliers[0];
            for (std::uint32_t p = 1; p < n; ++p) mixed ^= ctx[p] * c.multipliers[p];
            for (std::uint32_t slot = 0; slot < 8U; ++slot) {
                const std::uint32_t head = (n - 2U) * 8U + slot;
                row_ids[static_cast<std::uint64_t>(r) * kPleHeads + head] = mixed % c.head_vocab_sizes[head] + c.head_offsets[head];
            }
        }
        prev[0] = prev[1]; prev[1] = token;
    }
    history[0] = prev[0]; history[1] = prev[1];
}

namespace {
constexpr std::int8_t kIq4Codebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__constant__ std::int8_t d_iq4_codebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// src/iq4_nl.cpp fp16_bits_to_float (exact widening)
float host_half(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U, e = (bits >> 10U) & 0x1fU, m = bits & 0x3ffU;
    std::uint32_t f;
    if (e == 0) {
        if (m == 0) f = sign;
        else { std::uint32_t n = m, s = 0; while ((n & 0x400U) == 0) { n <<= 1U; ++s; } n &= 0x3ffU; f = sign | ((127U - 15U - s + 1U) << 23U) | (n << 13U); }
    } else if (e == 0x1fU) f = sign | 0x7f800000U | (m << 13U) | (m ? 0x00400000U : 0U);
    else f = sign | ((e + 112U) << 23U) | (m << 13U);
    float out; std::memcpy(&out, &f, 4); return out;
}
}  // namespace

void ple_gather_iq4nl_host(const std::uint8_t* table, std::uint64_t table_rows, const std::uint64_t* row_ids,
    std::uint32_t rows, std::uint16_t* out, std::uint64_t out_stride, unsigned threads) {
    threads = std::max(1U, std::min(threads, 64U));
    std::vector<std::exception_ptr> errors(threads);
    const auto work = [&](unsigned w) {
        try {
            for (std::uint64_t r = w; r < rows; r += threads) {
                std::uint16_t decoded[kPleWidth];   // host-cached decode, then one copy (the engine does the same)
                for (std::uint32_t h = 0; h < kPleHeads; ++h) {
                    const std::uint64_t id = row_ids[r * kPleHeads + h];
                    if (id >= table_rows) throw std::out_of_range("PLE row id exceeds table row count");
                    const std::uint8_t* row = table + id * kIq4RowBytes;
                    for (std::uint32_t b = 0; b < kPleHeadDim / 32U; ++b) {
                        const std::uint8_t* blk = row + b * 18U;
                        const float d = host_half(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8U)));
                        if (!std::isfinite(d)) throw std::invalid_argument("IQ4_NL PLE scale must be finite");
                        for (std::uint32_t e = 0; e < 32U; ++e) {
                            const std::uint8_t p = blk[2U + e % 16U];
                            const std::uint8_t code = e < 16U ? (p & 0x0fU) : (p >> 4U);
                            const float v = d * static_cast<float>(kIq4Codebook[code]);   // exact (11 x 7 bits)
                            decoded[h * kPleHeadDim + b * 32U + e] = f2bf(v);
                        }
                    }
                }
                std::memcpy(out + r * out_stride, decoded, sizeof(decoded));
            }
        } catch (...) { errors[w] = std::current_exception(); }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& t : pool) t.join();
    for (auto& e : errors) if (e) std::rethrow_exception(e);
}

namespace {
// One thread per output element: row r, head h, element e (0..159).
__global__ void ple_gather_iq4nl_kernel(const std::uint8_t* __restrict__ table, std::uint64_t table_rows,
    const std::uint64_t* __restrict__ row_ids, std::uint32_t rows, std::uint16_t* __restrict__ out, std::uint64_t out_stride,
    std::uint32_t* status) {
    const std::uint64_t t = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::uint64_t>(rows) * kPleWidth) return;
    const std::uint64_t r = t / kPleWidth;
    const std::uint32_t c = static_cast<std::uint32_t>(t % kPleWidth), h = c / kPleHeadDim, e = c % kPleHeadDim;
    const std::uint64_t id = row_ids[r * kPleHeads + h];
    std::uint16_t value = 0;
    if (id >= table_rows) { if (status) atomicOr(status, 2U); }
    else {
        const std::uint8_t* blk = table + id * kIq4RowBytes + (e / 32U) * 18U;
        const std::uint32_t w = e % 32U;
        const auto scale = static_cast<unsigned short>(blk[0] | (blk[1] << 8U));
        const float d = __half2float(__ushort_as_half(scale));   // exact widening
        if ((scale & 0x7c00U) == 0x7c00U && status) atomicOr(status, 1U);
        const std::uint8_t p = blk[2U + w % 16U];
        const std::uint8_t code = w < 16U ? (p & 0x0fU) : (p >> 4U);
        value = f2bf(__fmul_rn(d, static_cast<float>(d_iq4_codebook[code])));
    }
    out[r * out_stride + c] = value;
}
}  // namespace

void launch_ple_gather_iq4nl(const std::uint8_t* table, std::uint64_t table_rows, const std::uint64_t* row_ids,
    std::uint32_t rows, std::uint16_t* out, std::uint64_t out_stride, std::uint32_t* status, cudaStream_t stream) {
    if (!rows) return;
    const std::uint64_t n = static_cast<std::uint64_t>(rows) * kPleWidth;
    ple_gather_iq4nl_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, stream>>>(table, table_rows, row_ids, rows, out,
        out_stride, status);
    check(cudaGetLastError(), "PLE gather launch");
}

// ---------------------------------------------------------------------------------------------------------------
// Q8_0 projection with the R1 DAG (hip_ple_launcher.hip:496-538; the lane kernel hip_q8_rows_gemv.hpp:919-1035 is
// the same DAG, its tree "lambda bits 7,6,5,4,3 then 2,1,0"):
//   leaf lambda = 8 L + l (L = lane 0..31, l = 0..7): acc = +0; for k ascending: acc = fma(x[lambda + 256 k], c, acc),
//   c = float(fp16 scale) * float(code) (exact); FMA from the ISA (lane .inc: v_dual_fmaak_f32 ..., 0x0 first step,
//   then v_dual_fmac_f32);
//   tree: shfl_down 16, 8, 4, 2, 1 (lane L += lane L + d), then p0+=p4, p1+=p5, p2+=p6, p3+=p7; p0+=p2, p1+=p3; p0+=p1.
// Every tree addition is lower + upper of the same (lane, leaf) pairs; float addition is commutative, so the
// reduce-scatter below (each exchange keeps half the pairs per lane) computes the same values.
// A warp computes 4 outputs x 4 rows (16 pairs x 8 leaves = 128 accumulators per lane). The reduce-scatter runs the
// lane levels 16, 8, 4, 2 on halving pair sets (each lane ends with the 8 leaves of one pair, partial over lanes
// L, L^1), then level 1 as an xor exchange of all 8 leaves (both lanes form lower + upper), then the leaf tree.
// Skipped optimizations: LDS-staged decoded coefficients shared by the block, cp.async weight prefetch, more rows
// per warp (the HIP lane kernel's 64 rows x 64 outputs per block), packed FP32x2 FMA on sm_100+.
namespace {
constexpr unsigned kProjOuts = 4U, kProjRows = 4U, kProjWarps = 8U;
__global__ __launch_bounds__(256) void ple_project_kernel(const std::uint16_t* __restrict__ input, std::uint32_t in_stride,
    std::uint32_t rows, const std::uint8_t* __restrict__ weights, std::uint32_t width, std::uint32_t outputs,
    float* __restrict__ out, std::uint32_t out_stride) {
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
    const std::uint32_t o0 = (blockIdx.x * kProjWarps + warp) * kProjOuts, r0 = blockIdx.y * kProjRows;
    if (o0 >= outputs) return;
    const std::uint32_t blocks = width / 32U;
    float acc[kProjOuts * kProjRows][8];
#pragma unroll
    for (unsigned p = 0; p < kProjOuts * kProjRows; ++p)
#pragma unroll
        for (unsigned l = 0; l < 8; ++l) acc[p][l] = 0.0F;
    const std::uint8_t* wrow[kProjOuts];
#pragma unroll
    for (unsigned o = 0; o < kProjOuts; ++o)
        wrow[o] = weights + static_cast<std::uint64_t>(min(o0 + o, outputs - 1U)) * blocks * 34U;
    for (std::uint32_t chunk = 0; chunk < width; chunk += 256U) {
        const std::uint32_t f = chunk + 8U * lane;
        float x[kProjRows][8];
#pragma unroll
        for (unsigned r = 0; r < kProjRows; ++r) {
            const std::uint32_t row = min(r0 + r, rows - 1U);
            const uint4 v = *reinterpret_cast<const uint4*>(input + static_cast<std::uint64_t>(row) * in_stride + f);
            const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) { x[r][2 * i] = __uint_as_float(w[i] << 16U); x[r][2 * i + 1] = __uint_as_float(w[i] & 0xffff0000U); }
        }
#pragma unroll
        for (unsigned o = 0; o < kProjOuts; ++o) {
            const std::uint8_t* blk = wrow[o] + (f / 32U) * 34U;
            const float d = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(blk)));
            const unsigned short* q = reinterpret_cast<const unsigned short*>(blk + 2U + f % 32U);
            float c[8];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned short pair = q[i];
                c[2 * i] = __fmul_rn(d, static_cast<float>(static_cast<std::int8_t>(pair & 0xffU)));
                c[2 * i + 1] = __fmul_rn(d, static_cast<float>(static_cast<std::int8_t>(pair >> 8U)));
            }
#pragma unroll
            for (unsigned r = 0; r < kProjRows; ++r)
#pragma unroll
                for (unsigned l = 0; l < 8; ++l) acc[o * kProjRows + r][l] = __fmaf_rn(x[r][l], c[l], acc[o * kProjRows + r][l]);
        }
    }
    // Reduce-scatter over lane bits 4..1 (distances 16, 8, 4, 2): the pair set halves each level.
#pragma unroll
    for (unsigned level = 0; level < 4; ++level) {
        const unsigned d = 16U >> level, n = (kProjOuts * kProjRows) >> level, half = n / 2U;
        const bool hi = (lane & d) != 0U;
#pragma unroll
        for (unsigned j = 0; j < half; ++j)
#pragma unroll
            for (unsigned l = 0; l < 8; ++l) {
                const float a = acc[j][l], b = acc[j + half][l];
                const float send = hi ? a : b, mine = hi ? b : a;
                const float recv = __shfl_xor_sync(0xffffffffU, send, d);
                acc[j][l] = __fadd_rn(mine, recv);
            }
    }
    // Pair index held by this lane: bit 3 from lane bit 4, ..., bit 0 from lane bit 1.
    const unsigned pair = ((lane >> 4U) & 1U) << 3U | ((lane >> 3U) & 1U) << 2U | ((lane >> 2U) & 1U) << 1U | ((lane >> 1U) & 1U);
    float p[8];
#pragma unroll
    for (unsigned l = 0; l < 8; ++l) p[l] = __fadd_rn(acc[0][l], __shfl_xor_sync(0xffffffffU, acc[0][l], 1U));
    p[0] = __fadd_rn(p[0], p[4]); p[1] = __fadd_rn(p[1], p[5]); p[2] = __fadd_rn(p[2], p[6]); p[3] = __fadd_rn(p[3], p[7]);
    p[0] = __fadd_rn(p[0], p[2]); p[1] = __fadd_rn(p[1], p[3]);
    p[0] = __fadd_rn(p[0], p[1]);
    if ((lane & 1U) == 0U) {
        const std::uint32_t o = o0 + pair / kProjRows, r = r0 + pair % kProjRows;
        if (o < outputs && r < rows) out[static_cast<std::uint64_t>(r) * out_stride + o] = p[0];
    }
}
}  // namespace

void launch_ple_project(const std::uint16_t* input, std::uint32_t in_stride, std::uint32_t rows, const void* weights,
    std::uint32_t width, std::uint32_t outputs, float* out, std::uint32_t out_stride, cudaStream_t stream) {
    if (!rows || !outputs) return;
    if (width % 256U || in_stride % 8U || reinterpret_cast<std::uintptr_t>(input) % 16U)
        throw std::invalid_argument("PLE projection needs 256-wide chunks and 16-byte aligned rows");
    const dim3 grid((outputs + kProjOuts * kProjWarps - 1U) / (kProjOuts * kProjWarps), (rows + kProjRows - 1U) / kProjRows);
    ple_project_kernel<<<grid, 256, 0, stream>>>(input, in_stride, rows, static_cast<const std::uint8_t*>(weights), width, outputs,
        out, out_stride);
    check(cudaGetLastError(), "PLE projection launch");
}

// ---------------------------------------------------------------------------------------------------------------
// Rows-form PLE tail (hip_ple_launcher.hip:882-1054). ISA: _ZN5strix12_GLOBAL__N_125ple_rows_gate_norm_kernel...
// (hip_ple_launcher.s:2168-3202): sums v_dual_fmac_f32 (k*k, q*q), LDS tree v_add_f32, IEEE v_div_* then v_add eps,
// rsqrtf = OCML small-input scaling + v_rsq_f32; normalized key/query = v_mul(v_mul(x, inv), w), dot v_fmac_f32;
// score = dot / sqrtf(2560) (correctly rounded v_sqrt expansion, :2696-2722), v_max(|score|, 1e-6), sqrt, sign
// select, expf(-sr) folded into s_mul(sr, -log2e) (:2775-2795, = amd_expf(-sr)), 1 + e, IEEE 1 / x;
// gated = v_mul (inline asm), square v_fmac_f32; key = v_mul(v_mul(gated, inv_rms), w).
namespace {
namespace am = strix::cuda::amdmath;
constexpr unsigned kTailThreads = 256U;

__device__ __forceinline__ float tree256(float* s, unsigned lane, float v) {
    s[lane] = v;
    __syncthreads();
    for (unsigned stride = kTailThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) s[lane] = __fadd_rn(s[lane], s[lane + stride]);
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

__global__ __launch_bounds__(256) void ple_rows_gate_norm_kernel(PleTailArgs a, am::Tables t) {
    const unsigned stream = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    float* key = a.keys + static_cast<std::uint64_t>(row) * a.key_stride;
    const float* value = a.values + static_cast<std::uint64_t>(row) * a.value_stride;
    const std::uint16_t* query = a.input + static_cast<std::uint64_t>(row) * a.input_stride;
    __shared__ float red[kTailThreads], red2[kTailThreads];
    __shared__ float inv_k, inv_q, gate_s, inv_rms;
    const std::uint32_t base = stream * kHidden;
    float ks = 0.0F, qs = 0.0F;
    for (std::uint32_t f = lane; f < kHidden; f += kTailThreads) {
        const float k = key[base + f], q = bf16f(query[base + f]);
        ks = __fmaf_rn(k, k, ks);
        qs = __fmaf_rn(q, q, qs);
    }
    red[lane] = ks; red2[lane] = qs;
    __syncthreads();
    for (unsigned stride = kTailThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) { red[lane] = __fadd_rn(red[lane], red[lane + stride]); red2[lane] = __fadd_rn(red2[lane], red2[lane + stride]); }
        __syncthreads();
    }
    const float hf = static_cast<float>(kHidden);
    if (lane == 0) {
        inv_k = am::amd_rsqrtf(t, __fadd_rn(__fdiv_rn(red[0], hf), a.epsilon));
        inv_q = am::amd_rsqrtf(t, __fadd_rn(__fdiv_rn(red2[0], hf), a.epsilon));
    }
    __syncthreads();
    float dot = 0.0F;
    for (std::uint32_t f = lane; f < kHidden; f += kTailThreads) {
        const std::uint32_t i = base + f;
        const float nk = __fmul_rn(__fmul_rn(key[i], inv_k), a.norm_key[i]);
        const float nq = __fmul_rn(__fmul_rn(bf16f(query[i]), inv_q), a.norm_query[i]);
        dot = __fmaf_rn(nk, nq, dot);
    }
    const float dsum = tree256(red, lane, dot);
    if (lane == 0) {
        const float score = __fdiv_rn(dsum, __fsqrt_rn(hf));
        const float mag = __fsqrt_rn(fmaxf(fabsf(score), 1e-6F));
        const float sr = score > 0.0F ? mag : score < 0.0F ? -mag : 0.0F;
        const float g = __fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(t, -sr)));
        gate_s = g;
        a.gates[static_cast<std::uint64_t>(row) * kHc + stream] = g;
    }
    __syncthreads();
    const float g = gate_s;
    float sq = 0.0F;
    for (std::uint32_t f = lane; f < kHidden; f += kTailThreads) {
        const float gv = __fmul_rn(value[f], g);
        sq = __fmaf_rn(gv, gv, sq);
    }
    const float ssum = tree256(red, lane, sq);
    if (lane == 0) inv_rms = am::amd_rsqrtf(t, __fadd_rn(__fdiv_rn(ssum, hf), a.epsilon));
    __syncthreads();
    for (std::uint32_t f = lane; f < kHidden; f += kTailThreads) {
        const std::uint32_t i = base + f;
        key[i] = __fmul_rn(__fmul_rn(__fmul_rn(value[f], g), inv_rms), a.norm_conv[i]);
    }
}

// ple_rows_conv_residual_kernel (ISA hip_ple_launcher.s:3203-3628): conv v_fmac_f32 taps ascending; gated v_mul;
// (hidden + gated) then + silu (v_add_f32 :3501, :3530); silu = conv / (1 + expf(-conv)) (expf folded as above).
__global__ __launch_bounds__(256) void ple_rows_conv_residual_kernel(PleTailArgs a, am::Tables t) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (channel >= kExpanded) return;
    const std::uint16_t* state = (a.conv_state_slot ? *a.conv_state_slot : a.conv_state) + static_cast<std::uint64_t>(channel) * kConvHistory;
    float conv = 0.0F;
#pragma unroll
    for (std::uint32_t tap = 0; tap < kConvKernel; ++tap) {
        const std::uint32_t lag = (kConvKernel - 1U - tap) * kNgram;
        float x;
        if (lag == 0) x = a.keys[static_cast<std::uint64_t>(row) * a.key_stride + channel];
        else if (row >= lag) x = bf16f(f2bf(a.keys[static_cast<std::uint64_t>(row - lag) * a.key_stride + channel]));
        else x = bf16f(state[kConvHistory + row - lag]);
        conv = __fmaf_rn(x, a.conv[static_cast<std::uint64_t>(channel) * kConvKernel + tap], conv);
    }
    const std::uint32_t stream = channel / kHidden, feature = channel % kHidden;
    const float gated = __fmul_rn(a.values[static_cast<std::uint64_t>(row) * a.value_stride + feature], a.gates[static_cast<std::uint64_t>(row) * kHc + stream]);
    const float silu = __fdiv_rn(conv, __fadd_rn(1.0F, am::amd_expf(t, -conv)));
    const float h = bf16f(a.input[static_cast<std::uint64_t>(row) * a.input_stride + channel]);
    a.output[static_cast<std::uint64_t>(row) * a.output_stride + channel] = f2bf(__fadd_rn(__fadd_rn(h, gated), silu));
}

__global__ __launch_bounds__(256) void ple_rows_conv_state_kernel(PleTailArgs a) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kExpanded) return;
    std::uint16_t* state = (a.conv_state_slot ? *a.conv_state_slot : a.conv_state) + static_cast<std::uint64_t>(channel) * kConvHistory;
    for (std::uint32_t i = 0; i < kConvHistory; ++i) {
        const std::uint32_t source = a.rows + i;   // ascending: reads only entries not yet rewritten
        state[i] = source < kConvHistory ? state[source]
            : f2bf(a.keys[static_cast<std::uint64_t>(source - kConvHistory) * a.key_stride + channel]);
    }
}
}  // namespace

// ple_rows_conv_state_kernel's rule for `keep` rows, reading the pre-window history from `before`.
__global__ __launch_bounds__(256) void ple_conv_rollback_kernel(const std::uint16_t* __restrict__ before, const float* __restrict__ keys,
    std::uint32_t key_stride, std::uint32_t keep, std::uint16_t* __restrict__ conv_state) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kExpanded) return;
    for (std::uint32_t i = 0; i < kConvHistory; ++i) {
        const std::uint32_t source = keep + i;
        conv_state[static_cast<std::uint64_t>(channel) * kConvHistory + i] = source < kConvHistory
            ? before[static_cast<std::uint64_t>(channel) * kConvHistory + source]
            : f2bf(keys[static_cast<std::uint64_t>(source - kConvHistory) * key_stride + channel]);
    }
}

void launch_ple_conv_rollback(const std::uint16_t* before, const float* keys, std::uint32_t key_stride, std::uint32_t keep,
    std::uint16_t* conv_state, cudaStream_t stream) {
    ple_conv_rollback_kernel<<<(kExpanded + 255U) / 256U, 256, 0, stream>>>(before, keys, key_stride, keep, conv_state);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("ple conv rollback launch: ") + cudaGetErrorString(e));
}

void launch_ple_tail(const PleTailArgs& args, const amdmath::Tables& tables, cudaStream_t stream) {
    if (!args.rows) return;
    ple_rows_gate_norm_kernel<<<dim3(kHc, args.rows), kTailThreads, 0, stream>>>(args, tables);
    check(cudaGetLastError(), "PLE gate/norm launch");
    const unsigned blocks = (kExpanded + kTailThreads - 1U) / kTailThreads;
    ple_rows_conv_residual_kernel<<<dim3(blocks, args.rows), kTailThreads, 0, stream>>>(args, tables);
    check(cudaGetLastError(), "PLE convolution launch");
    ple_rows_conv_state_kernel<<<blocks, kTailThreads, 0, stream>>>(args);
    check(cudaGetLastError(), "PLE convolution state launch");
}

// ---------------------------------------------------------------------------------------------------------------
// MTP predictor cache rows. ISA facts (gfx1151 disassembly):
//  - mtp_q8_to_q4 (hip_mtp_fast_draft.s:7): values v_fma_mix(scale, code, -0) (exact product); first strict |v| max;
//    d = v_mul(max, -0.125) (:282); 1/d IEEE; q = v_fmaak(inverse, v, 8.5) = fma (:310), v_min 15, v_cvt_u32 (truncate).
//  - mtp_input_norms<1024> (hip_mtp_input_fusion.s:1497): v_dual_fmac sum, LDS v_add tree, IEEE div, v_add eps,
//    correctly rounded sqrt expansion, IEEE 1 / x; v_mul(v, inv) then v_mul(., gain).
//  - mtp_input_projections_wave_rows<1,2> (hip_mtp_input_fusion.s:6981): coefficient v_fma_mix(scale, code, -0), chains
//    v_dual_fmac / v_fmac, lane butterfly adds.
//  - prepare_hyper_norm_kernel<bf16> (hip_hyper_mix_launcher.s:513481): v_dual_fmac sum, v_add tree, IEEE div, + eps,
//    rsqrtf (scaled v_rsq); v_mul(v, inv) then v_mul(., w).
//  - rows_splitk_hyper_projection_kernel<2, HyperRows, false> (hip_hyper_mix_launcher.s:513870): chains v_fma(x, c, 0)
//    then v_fmac; butterfly v_add; tile sum v_add(0, red0) then ascending; IEEE div by streams; sigmoid = OCML expf(-s)
//    (v_mul -log2e ...), 1 + e, IEEE 1 / x; low = v_mul(s, sigmoid).
//  - lane_chain_hyper_output_kernel<bf16, false> (hip_hyper_mix_launcher.s:490574): gate v_fmac(low, coef);
//    mixed v_fmac(normalized, sig); IEEE div by streams; BF16 RNE.
//  - linear_to_float_kernel<q8_0 | bf16> (hip_indexed_attention_launcher.s:1631857 / 1643281): v_dual_fmac chain,
//    v_add tree 128..1, round_bf16.
//  - norm_rope_kernel<f32> (hip_indexed_attention_launcher.s:13721): reduction v_mul(x, x), v_add tree, IEEE div, + eps,
//    rsqrtf; round_bf16(v_mul(v_mul(x, inv), w)); RoPE: v_mul(a, cos) then low half v_fma(-b, sin, .), high half
//    v_fma(b, sin, .) (:14350-14370); round_bf16.
namespace {
__device__ __forceinline__ std::uint16_t f2bf_quiet(float x) {   // hip_mtp_input_fusion.hip to_bf16 (quiet NaN upper)
    const unsigned bits = __float_as_uint(x);
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU)) return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float rnd_bf16(float v) { return bf16f(f2bf(v)); }
__device__ __forceinline__ float half_at(const std::uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | (p[1] << 8U))));
}

__global__ void mtp_q8_to_q4_kernel(const std::uint8_t* q8, std::uint64_t blocks, std::uint8_t* q4) {
    const std::uint64_t b = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const std::uint8_t* src = q8 + b * 34U;
    std::uint8_t* dst = q4 + b * 18U;
    const float scale = half_at(src);
    float values[32], mx = 0.0F, amax = 0.0F;
    for (unsigned i = 0; i < 32; ++i) {
        values[i] = __fmul_rn(scale, static_cast<float>(static_cast<std::int8_t>(src[2 + i])));
        if (fabsf(values[i]) > amax) { amax = fabsf(values[i]); mx = values[i]; }
    }
    const float d = __fmul_rn(mx, -0.125F);
    const float inverse = d != 0.0F ? __fdiv_rn(1.0F, d) : 0.0F;
    const unsigned short hd = __half_as_ushort(__float2half_rn(d));
    *reinterpret_cast<unsigned short*>(dst) = hd;   // 18-byte blocks keep 2-byte alignment
    for (unsigned j = 0; j < 16; ++j) {
        const unsigned lo = __float2uint_rz(fminf(15.0F, __fmaf_rn(inverse, values[j], 8.5F)));
        const unsigned hi = __float2uint_rz(fminf(15.0F, __fmaf_rn(inverse, values[j + 16], 8.5F)));
        dst[2 + j] = static_cast<std::uint8_t>(lo | (hi << 4U));
    }
}

// Workspace layout per call (bytes, 256-aligned sections).
struct MtpLayout { std::uint64_t embedding, normalized, fused, hc_normalized, low, hidden0, index_key, key, value, first_bad, total; };
MtpLayout mtp_layout(std::uint32_t rows) {
    MtpLayout l{};
    std::uint64_t at = 0;
    const auto take = [&](std::uint64_t bytes) { const auto o = at; at += (bytes + 255U) / 256U * 256U; return o; };
    const std::uint64_t r = rows;
    l.embedding = take(r * kHidden * 2U); l.normalized = take(r * (kHidden + kExpanded) * 4U); l.fused = take(r * kExpanded * 2U);
    l.hc_normalized = take(r * kExpanded * 4U); l.low = take(r * kMtpRank * 4U); l.hidden0 = take(r * kHidden * 2U);
    l.index_key = take(r * kMtpIndexDim * 4U); l.key = take(r * kMtpKvWidth * 4U); l.value = take(r * kMtpKvWidth * 4U);
    l.first_bad = take(4U); l.total = at;
    return l;
}

// 1. embedding_kernel<bf16, q8_0>: BF16 RNE of scale * code (exact).
__global__ void mtp_embedding_kernel(const std::uint8_t* table, std::uint32_t vocab, const std::uint32_t* tokens,
    std::uint16_t* out, std::uint32_t* status) {
    const std::uint32_t row = blockIdx.y, f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= kHidden) return;
    std::uint32_t token = tokens[row];
    if (token >= vocab) { if (status && f == 0) atomicOr(status, 1U); token = 0; }
    const std::uint8_t* blk = table + (static_cast<std::uint64_t>(token) * (kHidden / 32U) + f / 32U) * 34U;
    out[static_cast<std::uint64_t>(row) * kHidden + f] = f2bf(__fmul_rn(half_at(blk), static_cast<float>(static_cast<std::int8_t>(blk[2 + f % 32U]))));
}

// 2. mtp_input_norms<1024> (hip_mtp_input_fusion.hip:54-79): block 0 embedding (2560), block 1 wide hidden (10240).
__global__ __launch_bounds__(256) void mtp_input_norms_kernel(const std::uint16_t* embedding, const std::uint16_t* hidden,
    const float* enorm, const float* hnorm, float* normalized, MtpRounding rd) {
    __shared__ float sums[256];
    __shared__ float inv_s;
    const unsigned lane = threadIdx.x, row = blockIdx.y;
    const bool wide = blockIdx.x != 0U;
    const unsigned n = wide ? kExpanded : kHidden;
    const std::uint16_t* in = wide ? hidden + static_cast<std::uint64_t>(row) * kExpanded : embedding + static_cast<std::uint64_t>(row) * kHidden;
    const float* w = wide ? hnorm : enorm;
    float* out = normalized + static_cast<std::uint64_t>(row) * (kHidden + kExpanded) + (wide ? kHidden : 0U);
    float sum = 0.0F;
    for (unsigned i = lane; i < n; i += 256U) { const float v = bf16f(in[i]); sum = __fmaf_rn(v, v, sum); }
    sums[lane] = sum;
    __syncthreads();
    for (unsigned stride = 128U; stride; stride >>= 1U) { if (lane < stride) sums[lane] = __fadd_rn(sums[lane], sums[lane + stride]); __syncthreads(); }
    if (lane == 0) inv_s = __fdiv_rn(1.0F, __fsqrt_rn(__fadd_rn(__fdiv_rn(sums[0], static_cast<float>(n)), rd.epsilon)));
    __syncthreads();
    const float inverse = inv_s;
    for (unsigned i = lane; i < n; i += 256U) {
        const float gain = rd.gemma_offset ? __fadd_rn(1.0F, w[i]) : w[i];
        const float v = __fmul_rn(__fmul_rn(bf16f(in[i]), inverse), gain);
        out[i] = rd.normalized_bf16 ? bf16f(f2bf_quiet(v)) : v;
    }
}

// 3. mtp_input_projections_wave_rows<Q4 = true, R = 2> (hip_mtp_input_fusion.hip:166-212): output o, lane L chains over
// columns 8 L + 256 k + i (k, then i ascending), coefficient d * (nibble - 8) (exact); butterfly xor 16..1 (lane 0 value =
// the halving tree of the lanes); out[s][o] = bf16q(rnd(e) + rnd(s_k)). A warp computes one output for 4 rows.
constexpr unsigned kFuseRows = 4U;
__global__ __launch_bounds__(256) void mtp_fused_projection_kernel(const float* normalized, const std::uint8_t* eh_q4,
    std::uint32_t rows, std::uint16_t* fused, MtpRounding rd) {
    const unsigned lane = threadIdx.x & 31U, o = blockIdx.x * 8U + (threadIdx.x >> 5U), r0 = blockIdx.y * kFuseRows;
    if (o >= kHidden) return;
    const std::uint8_t* packed = eh_q4 + static_cast<std::uint64_t>(o) * kMtpEhRowQ4;
    float emb[kFuseRows], sums[kFuseRows][kHc];
#pragma unroll
    for (unsigned r = 0; r < kFuseRows; ++r) { emb[r] = 0.0F; for (unsigned s = 0; s < kHc; ++s) sums[r][s] = 0.0F; }
    for (unsigned base = 8U * lane; base < kHidden; base += 256U) {
        const std::uint8_t* e = packed + (base / 32U) * 18U;
        const std::uint8_t* wq = packed + ((kHidden + base) / 32U) * 18U;
        const float es = half_at(e), ws = half_at(wq);
        float ce[8], cw[8];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const unsigned j = base % 32U + i;
            ce[i] = __fmul_rn(es, static_cast<float>(static_cast<int>((e[2U + (j & 15U)] >> (j >= 16U ? 4U : 0U)) & 15U) - 8));
            cw[i] = __fmul_rn(ws, static_cast<float>(static_cast<int>((wq[2U + (j & 15U)] >> (j >= 16U ? 4U : 0U)) & 15U) - 8));
        }
#pragma unroll
        for (unsigned r = 0; r < kFuseRows; ++r) {
            const float* x = normalized + static_cast<std::uint64_t>(min(r0 + r, rows - 1U)) * (kHidden + kExpanded);
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                emb[r] = __fmaf_rn(ce[i], x[base + i], emb[r]);
#pragma unroll
                for (unsigned s = 0; s < kHc; ++s) sums[r][s] = __fmaf_rn(cw[i], x[kHidden + s * kHidden + base + i], sums[r][s]);
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < kFuseRows; ++r)
#pragma unroll
        for (unsigned d = 16U; d; d >>= 1U) {
            emb[r] = __fadd_rn(emb[r], __shfl_xor_sync(0xffffffffU, emb[r], d));
#pragma unroll
            for (unsigned s = 0; s < kHc; ++s) sums[r][s] = __fadd_rn(sums[r][s], __shfl_xor_sync(0xffffffffU, sums[r][s], d));
        }
    if (lane) return;
#pragma unroll
    for (unsigned r = 0; r < kFuseRows; ++r) {
        if (r0 + r >= rows) break;
        const float projected = rd.projection_bf16 ? bf16f(f2bf_quiet(emb[r])) : emb[r];
        for (unsigned s = 0; s < kHc; ++s) {
            const float sk = rd.projection_bf16 ? bf16f(f2bf_quiet(sums[r][s])) : sums[r][s];
            fused[static_cast<std::uint64_t>(r0 + r) * kExpanded + s * kHidden + o] = f2bf_quiet(__fadd_rn(projected, sk));
        }
    }
}

// 4. prepare_hyper_norm_kernel<bf16> (hip_hyper_mix_launcher.hip:662-688), grid (4 streams, rows).
__global__ __launch_bounds__(256) void mtp_hyper_norm_kernel(const std::uint16_t* fused, const float* norm, float eps,
    float* out, am::Tables t) {
    __shared__ float red[256];
    __shared__ float inv_s;
    const unsigned stream = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * kExpanded + stream * kHidden;
    float sum = 0.0F;
    for (unsigned f = lane; f < kHidden; f += 256U) { const float v = bf16f(fused[base + f]); sum = __fmaf_rn(v, v, sum); }
    red[lane] = sum;
    __syncthreads();
    for (unsigned stride = 128U; stride; stride >>= 1U) { if (lane < stride) red[lane] = __fadd_rn(red[lane], red[lane + stride]); __syncthreads(); }
    if (lane == 0) inv_s = am::amd_rsqrtf(t, __fadd_rn(__fdiv_rn(red[0], static_cast<float>(kHidden)), eps));
    __syncthreads();
    const float inv = inv_s;
    for (unsigned f = lane; f < kHidden; f += 256U)
        out[base + f] = __fmul_rn(__fmul_rn(bf16f(fused[base + f]), inv), norm[stream * kHidden + f]);
}

// 5. rows_splitk_hyper_projection_kernel<2> low-rank part (hip_hyper_mix_launcher.hip:2010-2031 with
// rows_splitk_hyper_lane_dots): output o, tile t = wave 0..15 (features 640 t ..), lane L chain over
// f = 640 t + 128 s + 4 L + j (s = 0..4, j = 0..3 ascending) from +0; butterfly xor 16..1; total = 0 + red[0] + .. + red[15];
// scaled = total / 4; low = scaled * (1 / (1 + expf(-scaled))). Block: 2 outputs x 4 rows.
constexpr unsigned kSplitOuts = 2U, kSplitRows = 4U;
__global__ __launch_bounds__(512) void mtp_hyper_low_kernel(const float* hc_normalized, const std::uint8_t* down,
    std::uint32_t rows, float* low, am::Tables tb) {
    __shared__ float red[kSplitRows][16][kSplitOuts];
    const unsigned lane = threadIdx.x & 31U, tile = threadIdx.x >> 5U;
    const unsigned o0 = blockIdx.x * kSplitOuts, r0 = blockIdx.y * kSplitRows;
    float coef[kSplitOuts][5][4];
#pragma unroll
    for (unsigned o = 0; o < kSplitOuts; ++o)
#pragma unroll
        for (unsigned s = 0; s < 5U; ++s) {
            const std::uint8_t* blk = down + static_cast<std::uint64_t>(o0 + o) * (kExpanded / 32U) * 34U +
                static_cast<std::uint64_t>(tile * 20U + 4U * s + lane / 8U) * 34U;
            const float scale = half_at(blk);
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j)
                coef[o][s][j] = __fmul_rn(scale, static_cast<float>(static_cast<std::int8_t>(blk[2U + 4U * (lane % 8U) + j])));
        }
#pragma unroll
    for (unsigned r = 0; r < kSplitRows; ++r) {
        const float* x = hc_normalized + static_cast<std::uint64_t>(min(r0 + r, rows - 1U)) * kExpanded + tile * 640U + lane * 4U;
        float xs[5][4];
#pragma unroll
        for (unsigned s = 0; s < 5U; ++s) { const float4 v = *reinterpret_cast<const float4*>(x + s * 128U); xs[s][0] = v.x; xs[s][1] = v.y; xs[s][2] = v.z; xs[s][3] = v.w; }
#pragma unroll
        for (unsigned o = 0; o < kSplitOuts; ++o) {
            float sum = 0.0F;
#pragma unroll
            for (unsigned s = 0; s < 5U; ++s)
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) sum = __fmaf_rn(xs[s][j], coef[o][s][j], sum);
#pragma unroll
            for (unsigned d = 16U; d; d >>= 1U) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffU, sum, d));
            if (lane == 0) red[r][tile][o] = sum;
        }
    }
    __syncthreads();
    if (threadIdx.x < kSplitRows * kSplitOuts) {
        const unsigned r = threadIdx.x / kSplitOuts, o = threadIdx.x % kSplitOuts;
        if (r0 + r < rows && o0 + o < kMtpRank) {
            float total = 0.0F;
            for (unsigned k = 0; k < 16U; ++k) total = __fadd_rn(total, red[r][k][o]);
            const float scaled = __fdiv_rn(total, static_cast<float>(kHc));
            const float sig = __fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(tb, -scaled)));
            low[static_cast<std::uint64_t>(r0 + r) * kMtpRank + o0 + o] = __fmul_rn(scaled, sig);
        }
    }
}

// 6. lane_chain_hyper_output_kernel<bf16> (hip_hyper_mix_launcher.hip:1630-1673): block (32 features, row), wave = stream.
__global__ __launch_bounds__(128) void mtp_hyper_output_kernel(const float* hc_normalized, const float* low,
    const std::uint8_t* up, std::uint16_t* hidden0, am::Tables tb) {
    __shared__ float sig[kHc * 32U];
    const unsigned lane = threadIdx.x & 31U, stream = threadIdx.x >> 5U, row = blockIdx.y;
    const unsigned feature = blockIdx.x * 32U + lane;
    const std::uint8_t* w = up + (static_cast<std::uint64_t>(stream) * kHidden + feature) * (kMtpRank / 32U) * 34U;
    const float* lr = low + static_cast<std::uint64_t>(row) * kMtpRank;
    float gate = 0.0F;
    for (unsigned b = 0; b < kMtpRank / 32U; ++b) {
        const std::uint8_t* blk = w + b * 34U;
        const float scale = half_at(blk);
        for (unsigned k = 0; k < 32U; ++k)
            gate = __fmaf_rn(lr[b * 32U + k], __fmul_rn(scale, static_cast<float>(static_cast<std::int8_t>(blk[2U + k]))), gate);
    }
    sig[stream * 32U + lane] = __fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(tb, -gate)));
    __syncthreads();
    if (stream == 0U) {
        float mixed = 0.0F;
        for (unsigned t = 0; t < kHc; ++t)
            mixed = __fmaf_rn(hc_normalized[static_cast<std::uint64_t>(row) * kExpanded + t * kHidden + feature], sig[t * 32U + lane], mixed);
        hidden0[static_cast<std::uint64_t>(row) * kHidden + feature] = f2bf(__fdiv_rn(mixed, static_cast<float>(kHc)));
    }
}

// 7. linear_to_float_kernel (hip_indexed_attention_launcher.hip:605-650) for the index key (BF16 weights, 128 outputs),
// K and V (Q8_0, 512 each): thread t chains features t + 256 k from +0, LDS tree 128..1, round_bf16. Block: one output,
// 4 rows. blockIdx.x: [0, 128) index key, [128, 640) K, [640, 1152) V.
constexpr unsigned kLinRows = 4U;
__global__ __launch_bounds__(256) void mtp_cache_projection_kernel(const std::uint16_t* hidden0, std::uint32_t rows,
    const std::uint16_t* index_k, const std::uint8_t* kq, const std::uint8_t* vq, float* index_out, float* k_out, float* v_out) {
    __shared__ float scratch[kLinRows][256];
    const unsigned lane = threadIdx.x, r0 = blockIdx.y * kLinRows;
    unsigned out = blockIdx.x;
    int kind = 0;
    if (out >= kMtpIndexDim) { out -= kMtpIndexDim; kind = 1; if (out >= kMtpKvWidth) { out -= kMtpKvWidth; kind = 2; } }
    const std::uint8_t* q8 = kind == 1 ? kq : vq;
    float sum[kLinRows];
#pragma unroll
    for (unsigned r = 0; r < kLinRows; ++r) sum[r] = 0.0F;
    for (unsigned f = lane; f < kHidden; f += 256U) {
        float w;
        if (kind == 0) w = bf16f(index_k[static_cast<std::uint64_t>(out) * kHidden + f]);
        else {
            const std::uint8_t* blk = q8 + (static_cast<std::uint64_t>(out) * (kHidden / 32U) + f / 32U) * 34U;
            w = __fmul_rn(half_at(blk), static_cast<float>(static_cast<std::int8_t>(blk[2U + f % 32U])));
        }
#pragma unroll
        for (unsigned r = 0; r < kLinRows; ++r)
            sum[r] = __fmaf_rn(bf16f(hidden0[static_cast<std::uint64_t>(min(r0 + r, rows - 1U)) * kHidden + f]), w, sum[r]);
    }
#pragma unroll
    for (unsigned r = 0; r < kLinRows; ++r) scratch[r][lane] = sum[r];
    __syncthreads();
    for (unsigned stride = 128U; stride; stride >>= 1U) {
        if (lane < stride)
#pragma unroll
            for (unsigned r = 0; r < kLinRows; ++r) scratch[r][lane] = __fadd_rn(scratch[r][lane], scratch[r][lane + stride]);
        __syncthreads();
    }
    if (lane < kLinRows && r0 + lane < rows) {
        const float v = rnd_bf16(scratch[lane][0]);
        const std::uint64_t row = r0 + lane;
        if (kind == 0) index_out[row * kMtpIndexDim + out] = v;
        else if (kind == 1) k_out[row * kMtpKvWidth + out] = v;
        else v_out[row * kMtpKvWidth + out] = v;
    }
}

// 8a. norm_rope_body (hip_indexed_attention_launcher.hip:880-994) on K in place, grid (2 heads, rows) x 256.
__global__ __launch_bounds__(256) void mtp_key_norm_rope_kernel(float* key, const float* k_norm, const float* cs,
    std::uint64_t rope_positions, std::uint64_t position0, float eps, std::uint32_t* status, am::Tables tb) {
    __shared__ float source[kMtpHeadDim], reduction[kMtpHeadDim];
    const unsigned head = blockIdx.x, row = blockIdx.y, f = threadIdx.x;
    const std::uint64_t position = position0 + row;
    float* values = key + static_cast<std::uint64_t>(row) * kMtpKvWidth + head * kMtpHeadDim;
    const float input = values[f];
    source[f] = input;
    reduction[f] = __fmul_rn(input, input);
    __syncthreads();
    for (unsigned stride = kMtpHeadDim / 2U; stride; stride >>= 1U) { if (f < stride) reduction[f] = __fadd_rn(reduction[f], reduction[f + stride]); __syncthreads(); }
    const float mean = __fadd_rn(__fdiv_rn(reduction[0], static_cast<float>(kMtpHeadDim)), eps);
    const float inv = am::amd_rsqrtf(tb, mean);
    __syncthreads();
    source[f] = rnd_bf16(__fmul_rn(__fmul_rn(input, inv), k_norm[f]));
    __syncthreads();
    float result = source[f];
    if (f < kMtpRotary) {
        const unsigned half = kMtpRotary / 2U, pair = f % half;
        const std::uint64_t p = position < rope_positions ? position : 0U;
        if (position >= rope_positions && status && f == 0) atomicOr(status, 2U);
        const float c = cs[(p * half + pair) * 2U], s = cs[(p * half + pair) * 2U + 1U];
        const float ac = __fmul_rn(source[f], c);
        result = f < half ? __fmaf_rn(-source[f + half], s, ac) : __fmaf_rn(source[f - half], s, ac);
    }
    values[f] = rnd_bf16(result);
}

// 8b. validate_mtp_cache_casts (finite and no BF16 overflow), then store_raw_index_key_kernel / store_kv_kernel. HIP runs
// rows in order and a failed row sets the status, so it and every later row store nothing: the first bad row is found
// with an integer atomicMin (order-free) and stores are gated on it.
__device__ __forceinline__ bool cache_bad(float v) { return !isfinite(v) || (f2bf(v) & 0x7f80U) == 0x7f80U; }
__global__ void mtp_cache_validate_kernel(const float* index_key, const float* key, const float* value,
    std::uint32_t* first_bad, std::uint32_t* status) {
    const std::uint32_t row = blockIdx.x;
    bool bad = false;
    for (unsigned i = threadIdx.x; i < kMtpKvWidth; i += blockDim.x) {
        if (i < kMtpIndexDim) bad = bad || cache_bad(index_key[static_cast<std::uint64_t>(row) * kMtpIndexDim + i]);
        bad = bad || cache_bad(key[static_cast<std::uint64_t>(row) * kMtpKvWidth + i]) || cache_bad(value[static_cast<std::uint64_t>(row) * kMtpKvWidth + i]);
    }
    if (bad) { atomicMin(first_bad, row); if (status) atomicOr(status, 8U); }
}
__global__ void mtp_cache_store_kernel(const float* index_key, const float* key, const float* value, std::uint64_t position0,
    const std::uint32_t* first_bad, std::uint16_t* kv, std::uint16_t* raw_index) {
    const std::uint32_t row = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= *first_bad || i >= kMtpKvWidth) return;
    const std::uint64_t p = position0 + row;
    kv[p * kMtpKvWidth * 2U + i] = f2bf(key[static_cast<std::uint64_t>(row) * kMtpKvWidth + i]);
    kv[p * kMtpKvWidth * 2U + kMtpKvWidth + i] = f2bf(value[static_cast<std::uint64_t>(row) * kMtpKvWidth + i]);
    if (i < kMtpIndexDim) raw_index[p * kMtpIndexDim + i] = f2bf(index_key[static_cast<std::uint64_t>(row) * kMtpIndexDim + i]);
}
}  // namespace

void launch_mtp_q8_to_q4(const void* q8, std::uint64_t blocks, void* q4, cudaStream_t stream) {
    if (!blocks) return;
    mtp_q8_to_q4_kernel<<<static_cast<unsigned>((blocks + 255U) / 256U), 256, 0, stream>>>(static_cast<const std::uint8_t*>(q8), blocks,
        static_cast<std::uint8_t*>(q4));
    check(cudaGetLastError(), "MTP Q8 -> Q4 launch");
}

std::uint64_t mtp_workspace_bytes(std::uint32_t rows) { return mtp_layout(rows).total; }

MtpWorkspaceViews mtp_workspace_views(void* workspace, std::uint32_t rows) {
    const auto l = mtp_layout(rows);
    auto* b = static_cast<unsigned char*>(workspace);
    return {reinterpret_cast<std::uint16_t*>(b + l.embedding), reinterpret_cast<float*>(b + l.normalized),
        reinterpret_cast<std::uint16_t*>(b + l.fused), reinterpret_cast<float*>(b + l.hc_normalized), reinterpret_cast<float*>(b + l.low),
        reinterpret_cast<std::uint16_t*>(b + l.hidden0), reinterpret_cast<float*>(b + l.index_key), reinterpret_cast<float*>(b + l.key),
        reinterpret_cast<float*>(b + l.value)};
}

void launch_mtp_cache_rows(const MtpWeights& w, const MtpRounding& rd, const std::uint16_t* hidden,
    const std::uint32_t* next_tokens, std::uint32_t rows, std::uint64_t position0, std::uint16_t* kv, std::uint16_t* raw_index,
    void* workspace, std::uint32_t* status, const amdmath::Tables& tables, cudaStream_t stream) {
    if (!rows) return;
    const auto l = mtp_layout(rows);
    const auto v = mtp_workspace_views(workspace, rows);
    auto* first_bad = reinterpret_cast<std::uint32_t*>(static_cast<unsigned char*>(workspace) + l.first_bad);
    check(cudaMemsetAsync(first_bad, 0xff, 4, stream), "MTP first-bad reset");
    mtp_embedding_kernel<<<dim3(kHidden / 256U, rows), 256, 0, stream>>>(static_cast<const std::uint8_t*>(w.token_embd), w.vocab,
        next_tokens, v.embedding, status);
    mtp_input_norms_kernel<<<dim3(2, rows), 256, 0, stream>>>(v.embedding, hidden, w.enorm, w.hnorm, v.normalized, rd);
    mtp_fused_projection_kernel<<<dim3(kHidden / 8U, (rows + kFuseRows - 1U) / kFuseRows), 256, 0, stream>>>(v.normalized,
        static_cast<const std::uint8_t*>(w.eh_q4), rows, v.fused, rd);
    mtp_hyper_norm_kernel<<<dim3(kHc, rows), 256, 0, stream>>>(v.fused, w.hc_norm, rd.epsilon, v.hc_normalized, tables);
    mtp_hyper_low_kernel<<<dim3(kMtpRank / kSplitOuts, (rows + kSplitRows - 1U) / kSplitRows), 512, 0, stream>>>(v.hc_normalized,
        static_cast<const std::uint8_t*>(w.hc_down), rows, v.low, tables);
    mtp_hyper_output_kernel<<<dim3(kHidden / 32U, rows), 128, 0, stream>>>(v.hc_normalized, v.low,
        static_cast<const std::uint8_t*>(w.hc_up), v.hidden0, tables);
    mtp_cache_projection_kernel<<<dim3(kMtpIndexDim + 2U * kMtpKvWidth, (rows + kLinRows - 1U) / kLinRows), 256, 0, stream>>>(
        v.hidden0, rows, w.index_k, static_cast<const std::uint8_t*>(w.k), static_cast<const std::uint8_t*>(w.v), v.index_key, v.key, v.value);
    mtp_key_norm_rope_kernel<<<dim3(kMtpKvHeads, rows), 256, 0, stream>>>(v.key, w.k_norm, w.rope_cs, w.rope_positions, position0,
        rd.epsilon, status, tables);
    mtp_cache_validate_kernel<<<rows, 256, 0, stream>>>(v.index_key, v.key, v.value, first_bad, status);
    mtp_cache_store_kernel<<<dim3(kMtpKvWidth / 256U, rows), 256, 0, stream>>>(v.index_key, v.key, v.value, position0, first_bad, kv, raw_index);
    check(cudaGetLastError(), "MTP cache rows launch");
}

}  // namespace strix::cuda::qwen_plemtp
