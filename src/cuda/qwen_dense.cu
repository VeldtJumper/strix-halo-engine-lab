// CUDA dense A8 projections and shared expert, bit-identical to the HIP dense-a8-v1 prefill. See
// include/strix/cuda_qwen_dense.hpp for the contract; HIP references are given per kernel.
//
// ISA facts relied on (gfx1151 disassembly of the HIP translation units, engine bf60d9d2):
//  quantizer  hip_gdn_launcher.s, _ZN5strix7q8_rows24dense_a8_quantize_kernelINS_12_GLOBAL__N_116GdnInputRowsSiteILj256EEEEEvT_PaPf
//             (:257360): v_cmp_class_f32 0x207 (non-finite), v_max3_f32 |x| from 0, DPP row_xmask:1/2 quad max/or,
//             tiny = v_cmp_lt 0 / v_cmp_gt 0x4000000 (2^-119) (:257498-257499), IEEE v_div_scale/fmas/fixup amax/127
//             (:257598-257613) and x/delta (:257543-257558), v_rndne_f32 (:257560), v_maxmin_f32 -127/127 (:257561),
//             v_cvt_i32_f32 (:257563). float_denorm_mode_32 3 (IEEE denormals) in every kernel descriptor.
//  projection hip_gdn_launcher.s, ..project_rows_dense_a8_wmma_lds2_kernelINS_..GdnInputRowsSiteILj256EEELj8ELb0E.. (:274436):
//             v_wmma_i32_16x16x16_iu8 from the 0x4B400000 seed (:274851-274858), v_add_f32 0xcb400000 (exact float(dot)),
//             v_mul_f32 dw*delta, v_fmac_f32 acc += p*dot (:274854-274892), v_cndmask 0x7fc00000 canonical NaN (:279359...).
//  shared down store hip_shared_expert_launcher.s (:114776-114786): asm v_mul_f32 gate, value; then BF16 (v_add3 0x7fff).
//  activation hip_shared_expert_launcher.s, _ZN5strix12_GLOBAL__N_136rows_shared_activation_scalar_kernelEmmmmPKvjj (:165-462):
//             expf(-g) inlined (v_mul -log2e (:202), v_fma, v_fmamk lo, v_rndne, v_sub, v_add, v_exp, v_ldexp (:214),
//             selects 0x42ce8ed0 / 0xc2b17218) == amd_expf(-g) by sign symmetry; v_add 1.0 (:220); IEEE div g/den
//             (:221-236); v_mul u*q (:238); scalar dot v_dual_fmac_f32 sum = w * x + sum over features tid, tid+256, ...
//             ascending (:281); LDS tree strides 128..1, v_add_f32 (:301-401); 1/(1 + expf(-s)) (:416-457, IEEE div).
#include "strix/cuda_qwen_dense.hpp"

#include <cuda_fp16.h>

#include <cstdlib>
#include <mutex>

namespace strix::cuda::qwen_dense {
namespace {

// STRIX_QRTX_DECODE_KERNELS=0: keep the tiled kernels at rows <= 8 too (reference seam for the decode-shaped ones).
bool decode_rows_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QRTX_DECODE_KERNELS"); return !(v && v[0] == '0'); }();
    return on;
}

constexpr int kBM = 128, kBN = 128, kKB = 4, kThreads = 256, kRowWords = 36;
constexpr int kStageA = kBM * kRowWords * 4, kStageB = kBN * kRowWords * 4, kStageD = kKB * kBM * 4;
constexpr int kStage = kStageA + kStageB + kStageD;
constexpr int kSmem = 2 * kStage;

__device__ __forceinline__ std::uint32_t bf16_bits(float v) {
    const std::uint32_t b = __float_as_uint(v);
    return (b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U;   // HIP float_to_bf16, no NaN special case
}
__device__ __forceinline__ float bf16_float(std::uint32_t h) { return __uint_as_float(h << 16U); }

// v_mul_f32 under IEEE mode: a NaN operand is returned quieted (src0 first) [assumption on RDNA3 NaN propagation];
// CUDA would return 0x7fffffff, whose BF16 rounding is 0x8000 (-0), so the NaN path is emulated explicitly.
__device__ __forceinline__ float amd_mul(float a, float b) {
    if (a != a) return __uint_as_float(__float_as_uint(a) | 0x00400000U);
    if (b != b) return __uint_as_float(__float_as_uint(b) | 0x00400000U);
    return __fmul_rn(a, b);
}

// ---------------------------------------------------------------------------------------------------------------------
// Quantizer: HIP dense_a8_quantize_kernel / dense_a8_quantize8 (src/hip_q8_rows_gemv.hpp :1107-1144). Four threads per
// 32-feature block, 8 features each; the quad max/or is order-free (fmaxf, |).
template <Input I>
__global__ __launch_bounds__(256) void quantize_kernel(std::uint32_t rows, std::uint32_t width, const void* input,
                                                       std::uint64_t stride, std::int8_t* codes, float* deltas,
                                                       std::uint32_t pitch) {
    const std::uint32_t groups = width / 8U;
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
    const bool valid = item < static_cast<std::uint64_t>(rows) * groups;
    const std::uint32_t r = valid ? static_cast<std::uint32_t>(item / groups) : 0U;
    const std::uint32_t g = valid ? static_cast<std::uint32_t>(item % groups) : 0U;
    float x[8];
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) {
        const std::uint64_t at = static_cast<std::uint64_t>(r) * stride + 8U * g + l;
        if (!valid) x[l] = 0.0F;
        else if constexpr (I == Input::bf16) x[l] = bf16_float(static_cast<const std::uint16_t*>(input)[at]);
        else x[l] = static_cast<const float*>(input)[at];
    }
    int bad = 0;
    float amax = 0.0F;
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) { bad |= !isfinite(x[l]); amax = fmaxf(amax, fabsf(x[l])); }
    bad |= __shfl_xor_sync(0xffffffffU, bad, 1); bad |= __shfl_xor_sync(0xffffffffU, bad, 2);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 1)); amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 2));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = bad ? __uint_as_float(kCanonicalNan) : (amax == 0.0F || tiny) ? 1.0F : __fdiv_rn(amax, 127.0F);
    unsigned word[2] = {0U, 0U};
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) {
        const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(x[l], delta));
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        word[l / 4U] |= (static_cast<unsigned>(code) & 0xffU) << (8U * (l % 4U));
    }
    if (!valid) return;
    *reinterpret_cast<uint2*>(codes + static_cast<std::uint64_t>(r) * width + 8U * g) = make_uint2(word[0], word[1]);
    if (!(g & 3U)) deltas[static_cast<std::uint64_t>(g / 4U) * pitch + r] = delta;
}

// ---------------------------------------------------------------------------------------------------------------------
// Projection. CTA = 128 rows x 128 outputs, 8 warps (2 x 4), warp = 64 rows x 32 outputs = 4 x 4 IMMA m16n8k32 tiles.
// K = 32 = one Q8_0 block per IMMA, so the IMMA result is exactly dot_b (seeded with 0x4B400000 like the HIP WMMA; then
// float(dot) = bits - 1.5*2^23, one exact subtraction). Every thread owns 64 (row, output) accumulators and walks the
// blocks in ascending order: the HIP chain acc = fma(dw * delta, float(dot), acc), unchanged by tiling.
// Pipeline: stages of 4 blocks (codes 128 x 128 B, raw Q8_0 bytes 128 x 136 B, deltas 4 x 128), cp.async double buffer.
struct KSeg { const unsigned char* w; void* out; std::uint64_t stride; std::uint32_t begin, end; };
struct KSite {
    KSeg seg[kMaxSegments];
    std::uint32_t nseg, width, outputs;
    const float* gate;
    std::uint64_t gate_stride;
};

__device__ __forceinline__ unsigned find_seg(const KSite& s, std::uint32_t n) {
    unsigned m = 0;
    while (m + 1U < s.nseg && n >= s.seg[m].end) ++m;
    return m;
}
__device__ __forceinline__ void cp_async4(void* smem, const void* g, bool ok) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(g), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* g, bool ok) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

__device__ __forceinline__ void imma(int (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
    constexpr int kMagic = 0x4B400000;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(kMagic));
}

template <Store S>
__device__ __forceinline__ void store_value(const KSite& site, std::uint32_t row, std::uint32_t n, float acc) {
    const float y = acc != acc ? __uint_as_float(kCanonicalNan) : acc;   // dense_a8_canonical (:1101)
    const KSeg& sg = site.seg[find_seg(site, n)];
    const std::uint64_t at = static_cast<std::uint64_t>(row) * sg.stride + (n - sg.begin);
    if constexpr (S == Store::f32) static_cast<float*>(sg.out)[at] = y;
    else if constexpr (S == Store::f32_round_bf16) static_cast<float*>(sg.out)[at] = bf16_float(bf16_bits(y));
    else if constexpr (S == Store::bf16) static_cast<std::uint16_t*>(sg.out)[at] = static_cast<std::uint16_t>(bf16_bits(y));
    else {
        const float gated = amd_mul(site.gate[static_cast<std::uint64_t>(row) * site.gate_stride], y);
        static_cast<std::uint16_t*>(sg.out)[at] = static_cast<std::uint16_t>(bf16_bits(gated));
    }
}

template <Store S>
__global__ __launch_bounds__(kThreads) void project_kernel(KSite site, std::uint32_t rows, const std::int8_t* codes,
                                                          const float* deltas, std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int wm = warp & 1, wn = warp >> 1;
    const std::uint32_t m0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * 34U;
    const std::uint32_t stages = (nblocks + kKB - 1U) / kKB;

    // Weight loader: thread -> output row ln of the tile, half of its words.
    const int ln = tid >> 1, half = tid & 1;
    const std::uint32_t gn = n0 + static_cast<std::uint32_t>(ln);
    const bool nvalid = gn < site.outputs;
    const unsigned char* wrow = site.seg[0].w;
    if (nvalid) {
        const KSeg& sg = site.seg[find_seg(site, gn)];
        wrow = sg.w + static_cast<std::uint64_t>(gn - sg.begin) * row_bytes;
    }
    const auto load_stage = [&](std::uint32_t s, int buf) {
        unsigned char* base = smem + buf * kStage;
        auto* A = reinterpret_cast<std::uint32_t*>(base);
        auto* B = reinterpret_cast<std::uint32_t*>(base + kStageA);
        auto* D = reinterpret_cast<float*>(base + kStageA + kStageB);
        const std::uint32_t kb0 = s * kKB, nb = min(static_cast<std::uint32_t>(kKB), nblocks - kb0);
        const std::uint32_t chunks = nb * 2U;
        for (std::uint32_t i = static_cast<std::uint32_t>(tid); i < kBM * chunks; i += kThreads) {
            const std::uint32_t r = i / chunks, c = i % chunks, gr = m0 + r;
            const bool ok = gr < rows;
            const std::int8_t* src = ok ? codes + static_cast<std::uint64_t>(gr) * width + kb0 * 32U + c * 16U : codes;
            cp_async16(A + r * kRowWords + c * 4U, src, ok);
        }
        const std::uint32_t words = 17U * nb / 2U, h0 = half ? (words + 1U) / 2U : 0U, h1 = half ? words : (words + 1U) / 2U;
        const auto* wsrc = reinterpret_cast<const std::uint32_t*>(wrow + static_cast<std::uint64_t>(kb0) * 34U);
        for (std::uint32_t w = h0; w < h1; ++w)
            cp_async4(B + ln * kRowWords + w, nvalid ? wsrc + w : reinterpret_cast<const std::uint32_t*>(site.seg[0].w), nvalid);
        for (std::uint32_t i = static_cast<std::uint32_t>(tid); i < nb * kBM; i += kThreads) {
            const std::uint32_t j = i / kBM, r = i % kBM, gr = m0 + r;
            const bool ok = gr < rows;
            cp_async4(D + j * kBM + r, ok ? deltas + static_cast<std::uint64_t>(kb0 + j) * pitch + gr : deltas, ok);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[a][b][e] = 0.0F;

    load_stage(0U, 0);
    cp_async_commit();
    for (std::uint32_t s = 0; s < stages; ++s) {
        if (s + 1U < stages) load_stage(s + 1U, static_cast<int>((s + 1U) & 1U));
        cp_async_commit();
        cp_async_wait1();
        __syncthreads();
        const unsigned char* base = smem + (s & 1U) * kStage;
        const auto* A = reinterpret_cast<const std::uint32_t*>(base);
        const auto* B = reinterpret_cast<const std::uint32_t*>(base + kStageA);
        const auto* D = reinterpret_cast<const float*>(base + kStageA + kStageB);
        const std::uint32_t nb = min(static_cast<std::uint32_t>(kKB), nblocks - s * kKB);
        for (std::uint32_t j = 0; j < nb; ++j) {   // blocks ascending: the HIP chain order
            std::uint32_t a[4][4];
            float dl[4][2];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int r = wm * 64 + mt * 16 + gid;
                a[mt][0] = A[r * kRowWords + j * 8U + tig];
                a[mt][1] = A[(r + 8) * kRowWords + j * 8U + tig];
                a[mt][2] = A[r * kRowWords + j * 8U + 4U + tig];
                a[mt][3] = A[(r + 8) * kRowWords + j * 8U + 4U + tig];
                dl[mt][0] = D[j * kBM + r];
                dl[mt][1] = D[j * kBM + r + 8];
            }
            // Block j of the stage starts at byte 34 j: its scale there, its codes at 34 j + 2 (even j: two bytes into a
            // word, realigned with a byte permute; odd j: word aligned).
            const bool even = (j & 1U) == 0U;
            const std::uint32_t qword = even ? 17U * j / 2U : (17U * j + 1U) / 2U;
            const std::uint32_t sword = even ? 17U * j / 2U : (17U * j - 1U) / 2U;
            std::uint32_t b[4][2];
            float dw[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const std::uint32_t* W = B + (wn * 32 + nt * 8 + gid) * kRowWords + qword + tig;
                if (even) { b[nt][0] = __byte_perm(W[0], W[1], 0x5432); b[nt][1] = __byte_perm(W[4], W[5], 0x5432); }
                else { b[nt][0] = W[0]; b[nt][1] = W[4]; }
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const std::uint32_t sw = B[(wn * 32 + nt * 8 + 2 * tig + c) * kRowWords + sword];
                    const unsigned short h = static_cast<unsigned short>(even ? (sw & 0xffffU) : (sw >> 16U));
                    dw[nt][c] = __half2float(__ushort_as_half(h));   // exact
                }
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int d[4];
                    imma(d, a[mt], b[nt]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float dot = __fsub_rn(__int_as_float(d[e]), 12582912.0F);   // exact float(dot_b)
                        const float p = __fmul_rn(dw[nt][e & 1], dl[mt][e >> 1]);         // v_mul_f32 dw * delta
                        acc[mt][nt][e] = __fmaf_rn(p, dot, acc[mt][nt][e]);               // v_fmac_f32
                    }
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const std::uint32_t row = m0 + static_cast<std::uint32_t>(wm * 64 + mt * 16 + gid + 8 * hh);
            if (row >= rows) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const std::uint32_t n = n0 + static_cast<std::uint32_t>(wn * 32 + nt * 8 + 2 * tig + c);
                    if (n < site.outputs) store_value<S>(site, row, n, acc[mt][nt][2 * hh + c]);
                }
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode projection (rows <= 8): the same per-(row, output) chain acc = fma(dw * delta, float(dot_b), acc) over blocks
// b ascending, laid out for few rows. A quad of lanes owns one output: lane q takes codes 8q..8q+7 of each block and
// holds partial integer dots for all RP rows (RP = rows padded to 1, 2, 4 or 8). The exact group dots are completed by
// integer adds across the quad (order-free): for RP <= 2 a full xor reduction (every lane then runs the chains, lane 0
// stores); for RP >= 4 a transpose-reduce (xor 2 then xor 1, each level keeping half of the rows), so lane q finishes
// rows (q >> 1) * RP/2 + (q & 1) * RP/4 .. + RP/4 and runs only their chains. Activations sit in shared memory.
constexpr unsigned kRowsThreads = 128, kRowsQuads = kRowsThreads / 4;
template <Store S, unsigned R, unsigned RP>
__global__ __launch_bounds__(kRowsThreads) void project_rows_kernel(KSite site, const std::int8_t* codes, const float* deltas,
                                                                   std::uint32_t pitch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t width = site.width, nblocks = width / 32U, row_bytes = nblocks * 34U;
    auto* A = reinterpret_cast<std::uint32_t*>(smem);                        // [RP][width / 4]
    auto* D = reinterpret_cast<float*>(smem + RP * width);                   // [RP][nblocks]
    for (std::uint32_t i = threadIdx.x; i < RP * width / 4U; i += kRowsThreads) {
        const std::uint32_t r = i / (width / 4U), w = i % (width / 4U);
        A[i] = r < R ? reinterpret_cast<const std::uint32_t*>(codes + static_cast<std::uint64_t>(r) * width)[w] : 0U;
    }
    for (std::uint32_t i = threadIdx.x; i < RP * nblocks; i += kRowsThreads)
        D[i] = i / nblocks < R ? deltas[static_cast<std::uint64_t>(i % nblocks) * pitch + i / nblocks] : 0.0F;
    __syncthreads();
    const unsigned q = threadIdx.x & 3U;
    const std::uint32_t n = blockIdx.x * kRowsQuads + threadIdx.x / 4U;
    const bool valid = n < site.outputs;
    const unsigned char* wrow = site.seg[0].w;
    if (valid) {
        const KSeg& sg = site.seg[find_seg(site, n)];
        wrow = sg.w + static_cast<std::uint64_t>(n - sg.begin) * row_bytes;
    }
    constexpr unsigned OWN = RP >= 4U ? RP / 4U : RP;                        // rows whose chains this lane runs
    const unsigned first = RP >= 4U ? (q >> 1U) * (RP / 2U) + (q & 1U) * (RP / 4U) : 0U;
    float acc[OWN];
#pragma unroll
    for (unsigned r = 0; r < OWN; ++r) acc[r] = 0.0F;
#pragma unroll 4
    for (std::uint32_t b = 0; b < nblocks; ++b) {
        const auto* h = reinterpret_cast<const unsigned short*>(wrow + 34U * b);   // 2-byte aligned
        const float dw = __half2float(__ushort_as_half(valid ? h[0] : static_cast<unsigned short>(0)));
        const unsigned short* c = h + 1U + 4U * q;
        const std::uint32_t w0 = valid ? static_cast<std::uint32_t>(c[0]) | (static_cast<std::uint32_t>(c[1]) << 16U) : 0U;
        const std::uint32_t w1 = valid ? static_cast<std::uint32_t>(c[2]) | (static_cast<std::uint32_t>(c[3]) << 16U) : 0U;
        int dot[RP];
#pragma unroll
        for (unsigned r = 0; r < RP; ++r) {
            const std::uint32_t* a = A + r * (width / 4U) + b * 8U + 2U * q;
            dot[r] = __dp4a(static_cast<int>(w1), static_cast<int>(a[1]), __dp4a(static_cast<int>(w0), static_cast<int>(a[0]), 0));
        }
        if constexpr (RP >= 4U) {
            const bool hi2 = (q & 2U) != 0U, hi1 = (q & 1U) != 0U;
#pragma unroll
            for (unsigned i = 0; i < RP / 2U; ++i) {
                const int keep = hi2 ? dot[i + RP / 2U] : dot[i], send = hi2 ? dot[i] : dot[i + RP / 2U];
                dot[i] = keep + __shfl_xor_sync(0xffffffffU, send, 2);
            }
#pragma unroll
            for (unsigned i = 0; i < RP / 4U; ++i) {
                const int keep = hi1 ? dot[i + RP / 4U] : dot[i], send = hi1 ? dot[i] : dot[i + RP / 4U];
                dot[i] = keep + __shfl_xor_sync(0xffffffffU, send, 1);
            }
        } else {
#pragma unroll
            for (unsigned r = 0; r < RP; ++r) {
                dot[r] += __shfl_xor_sync(0xffffffffU, dot[r], 1);
                dot[r] += __shfl_xor_sync(0xffffffffU, dot[r], 2);
            }
        }
#pragma unroll
        for (unsigned i = 0; i < OWN; ++i) {
            const float p = __fmul_rn(dw, D[(first + i) * nblocks + b]);    // v_mul_f32 dw * delta
            acc[i] = __fmaf_rn(p, static_cast<float>(dot[i]), acc[i]);      // v_fmac_f32 (float(dot) exact, |dot| < 2^24)
        }
    }
    if (!valid || (RP < 4U && q != 0U)) return;
#pragma unroll
    for (unsigned i = 0; i < OWN; ++i)
        if (first + i < R) store_value<S>(site, first + i, n, acc[i]);
}

template <Store S, unsigned R> cudaError_t launch_rows_r(cudaStream_t stream, const KSite& k, const std::int8_t* codes,
                                                        const float* deltas, std::uint32_t pitch) {
    constexpr unsigned RP = R <= 1U ? 1U : R <= 2U ? 2U : R <= 4U ? 4U : 8U;
    const std::uint32_t smem = RP * k.width + RP * (k.width / 32U) * 4U;
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] {
        attr = cudaFuncSetAttribute(project_rows_kernel<S, R, RP>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 6144 + 8 * 192 * 4);
    });
    if (attr != cudaSuccess) return attr;
    project_rows_kernel<S, R, RP><<<(k.outputs + kRowsQuads - 1U) / kRowsQuads, kRowsThreads, smem, stream>>>(k, codes, deltas, pitch);
    return cudaGetLastError();
}
template <Store S> cudaError_t launch_rows(cudaStream_t stream, const KSite& k, std::uint32_t rows, const std::int8_t* codes,
                                           const float* deltas, std::uint32_t pitch) {
    switch (rows) {
    case 1: return launch_rows_r<S, 1>(stream, k, codes, deltas, pitch);
    case 2: return launch_rows_r<S, 2>(stream, k, codes, deltas, pitch);
    case 3: return launch_rows_r<S, 3>(stream, k, codes, deltas, pitch);
    case 4: return launch_rows_r<S, 4>(stream, k, codes, deltas, pitch);
    case 5: return launch_rows_r<S, 5>(stream, k, codes, deltas, pitch);
    case 6: return launch_rows_r<S, 6>(stream, k, codes, deltas, pitch);
    case 7: return launch_rows_r<S, 7>(stream, k, codes, deltas, pitch);
    default: return launch_rows_r<S, 8>(stream, k, codes, deltas, pitch);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// rows_shared_activation_scalar_kernel (src/hip_shared_expert_launcher.hip:535-562), 256 threads, blockIdx.y = row.
__global__ __launch_bounds__(256) void shared_activation_kernel(float* workspace_base, std::uint64_t workspace_stride,
                                                                const std::uint16_t* input_base, std::uint64_t input_stride,
                                                                const float* scalar, std::uint32_t intermediate,
                                                                std::uint32_t hidden, amdmath::Tables tables) {
    __shared__ float scratch[256];
    float* workspace = workspace_base + blockIdx.y * workspace_stride;
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < intermediate) {
        const float gate = workspace[index];
        const float up = workspace[intermediate + index];
        const float den = __fadd_rn(1.0F, amdmath::amd_expf(tables, -gate));
        workspace[index] = __fmul_rn(__fdiv_rn(gate, den), up);
    }
    if (blockIdx.x != 0) return;
    const std::uint16_t* input = input_base + blockIdx.y * input_stride;
    float sum = 0.0F;
    for (std::uint32_t feature = threadIdx.x; feature < hidden; feature += 256U)
        sum = __fmaf_rn(scalar[feature], bf16_float(input[feature]), sum);   // v_dual_fmac_f32 (contracted in the ISA)
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = 128U; stride > 0U; stride >>= 1U) {   // HIP :555-558, v_add_f32 upper + lower
        if (threadIdx.x < stride) scratch[threadIdx.x] = __fadd_rn(scratch[threadIdx.x + stride], scratch[threadIdx.x]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        workspace[intermediate] = __fdiv_rn(1.0F, __fadd_rn(1.0F, amdmath::amd_expf(tables, -scratch[0])));
}

template <Store S> cudaError_t launch_project(cudaStream_t stream, const KSite& k, std::uint32_t rows, const std::int8_t* codes,
                                              const float* deltas, std::uint32_t pitch) {
    if (rows <= 8U && k.width <= 6144U && decode_rows_enabled()) return launch_rows<S>(stream, k, rows, codes, deltas, pitch);
    static std::once_flag once;
    static cudaError_t attr = cudaSuccess;
    std::call_once(once, [] { attr = cudaFuncSetAttribute(project_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem); });
    if (attr != cudaSuccess) return attr;
    const dim3 grid((k.outputs + kBN - 1U) / kBN, (rows + kBM - 1U) / kBM);
    project_kernel<S><<<grid, kThreads, kSmem, stream>>>(k, rows, codes, deltas, pitch);
    return cudaGetLastError();
}

}  // namespace

cudaError_t quantize(cudaStream_t stream, std::uint32_t rows, std::uint32_t width, Input type, const void* input,
                     std::uint64_t input_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch) {
    if (!rows) return cudaSuccess;
    if (width == 0U || width % 64U != 0U || pitch < rows || !input || !codes || !deltas) return cudaErrorInvalidValue;
    const std::uint64_t items = static_cast<std::uint64_t>(rows) * (width / 8U);
    const unsigned blocks = static_cast<unsigned>((items + 255U) / 256U);
    if (type == Input::bf16) quantize_kernel<Input::bf16><<<blocks, 256, 0, stream>>>(rows, width, input, input_stride, codes, deltas, pitch);
    else quantize_kernel<Input::f32><<<blocks, 256, 0, stream>>>(rows, width, input, input_stride, codes, deltas, pitch);
    return cudaGetLastError();
}

cudaError_t project(cudaStream_t stream, const Site& site, std::uint32_t rows, const std::int8_t* codes, const float* deltas,
                    std::uint32_t pitch) {
    if (!rows) return cudaSuccess;
    if (site.width == 0U || site.width % 64U != 0U || site.segments == 0U || site.segments > kMaxSegments || pitch < rows ||
        (reinterpret_cast<std::uintptr_t>(codes) & 15U) != 0U)
        return cudaErrorInvalidValue;
    if (site.store == Store::bf16_gated && !site.gate) return cudaErrorInvalidValue;
    KSite k{};
    std::uint32_t n = 0;
    for (std::uint32_t m = 0; m < site.segments; ++m) {
        const Segment& s = site.segment[m];
        if (!s.weight || !s.output || !s.outputs || (reinterpret_cast<std::uintptr_t>(s.weight) & 3U) != 0U) return cudaErrorInvalidValue;
        k.seg[m] = {static_cast<const unsigned char*>(s.weight), s.output, s.output_stride, n, n + s.outputs};
        n += s.outputs;
    }
    k.nseg = site.segments; k.width = site.width; k.outputs = n; k.gate = site.gate; k.gate_stride = site.gate_stride;
    switch (site.store) {
    case Store::f32: return launch_project<Store::f32>(stream, k, rows, codes, deltas, pitch);
    case Store::f32_round_bf16: return launch_project<Store::f32_round_bf16>(stream, k, rows, codes, deltas, pitch);
    case Store::bf16: return launch_project<Store::bf16>(stream, k, rows, codes, deltas, pitch);
    case Store::bf16_gated: return launch_project<Store::bf16_gated>(stream, k, rows, codes, deltas, pitch);
    }
    return cudaErrorInvalidValue;
}

cudaError_t quantize_project(cudaStream_t stream, const Site& site, std::uint32_t rows, Input type, const void* input,
                             std::uint64_t input_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch) {
    const cudaError_t e = quantize(stream, rows, site.width, type, input, input_stride, codes, deltas, pitch);
    return e != cudaSuccess ? e : project(stream, site, rows, codes, deltas, pitch);
}

cudaError_t shared_activation(cudaStream_t stream, std::uint32_t rows, float* workspace, std::uint64_t workspace_stride,
                              const std::uint16_t* input, std::uint64_t input_stride, const float* scalar,
                              std::uint32_t intermediate, std::uint32_t hidden, const amdmath::Tables& tables) {
    if (!rows) return cudaSuccess;
    if (!intermediate || workspace_stride < 2ULL * intermediate || !workspace || !input || !scalar) return cudaErrorInvalidValue;
    const dim3 grid((intermediate + 255U) / 256U, rows);
    shared_activation_kernel<<<grid, 256, 0, stream>>>(workspace, workspace_stride, input, input_stride, scalar, intermediate,
                                                       hidden, tables);
    return cudaGetLastError();
}

cudaError_t shared_expert(cudaStream_t stream, std::uint32_t rows, const SharedExpert& w, const std::uint16_t* input,
                          std::uint64_t input_stride, float* workspace, std::uint64_t workspace_stride, std::uint16_t* output,
                          std::uint64_t output_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch,
                          const amdmath::Tables& tables) {
    Site gate_up{};
    gate_up.store = Store::f32;
    gate_up.width = w.hidden;
    gate_up.segments = 2;
    gate_up.segment[0] = {w.gate_q8, w.intermediate, workspace, workspace_stride};
    gate_up.segment[1] = {w.up_q8, w.intermediate, workspace + w.intermediate, workspace_stride};
    cudaError_t e = quantize_project(stream, gate_up, rows, Input::bf16, input, input_stride, codes, deltas, pitch);
    if (e != cudaSuccess) return e;
    e = shared_activation(stream, rows, workspace, workspace_stride, input, input_stride, w.scalar, w.intermediate, w.hidden, tables);
    if (e != cudaSuccess) return e;
    Site down{};
    down.store = Store::bf16_gated;
    down.width = w.intermediate;
    down.segments = 1;
    down.segment[0] = {w.down_q8, w.hidden, output, output_stride};
    down.gate = workspace + w.intermediate;   // the scalar overwrote up[0] (HIP site.gate[r], :1874)
    down.gate_stride = workspace_stride;
    return quantize_project(stream, down, rows, Input::f32, workspace, workspace_stride, codes, deltas, pitch);
}

}  // namespace strix::cuda::qwen_dense
