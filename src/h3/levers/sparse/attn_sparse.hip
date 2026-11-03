// Block-sparse K1 variant (identity h3-attn-sparse-thr-v1): a copy of the K1 v1 default schedule (h3/attn b0a559848
// attn.hip flash_kernel<128, false, 0, 8, Early = true, Persistent = false, PP = false, Tune = 0>) whose key loop walks
// the block's LUT. Everything inside a step (QK WMMA, masking, online softmax, P packing, PV WMMA, barriers, the
// partial store) is the K1 code unchanged; only the step's key offset and the next-tile prefetch come from the LUT.
// See attn_sparse.hpp for the LUT contract.
#include "h3/levers/sparse/attn_sparse.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace strix::h3::levers::sparse {
namespace {

using attn::KeyBlock;
using attn::kMaxBlocksPerLaunch;
using attn::Partials;
using attn::Queries;

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned short bf16x16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));

constexpr unsigned kVP = 72;
constexpr float kC = static_cast<float>(1.4426950408889634073599 / 11.313708498984760390231);  // == K1 scale_log2<128>()
constexpr float kScoreBound = 32768.0F;
constexpr unsigned kLutStage = 1024U;  // LUT entries staged in LDS per work-group (4 KB)

__device__ __forceinline__ u32x4 g16(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p));
}
__device__ __forceinline__ void s16f(void* p, f4 v) {
    *reinterpret_cast<__attribute__((address_space(1))) f4*>(reinterpret_cast<std::uintptr_t>(p)) = v;
}
__device__ __forceinline__ f8 wmma(u32x8 a, u32x8 b, f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(bf16x16, a), __builtin_bit_cast(bf16x16, b), c);
}
__device__ __forceinline__ void lds_barrier() { __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ float other_half(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(0U, __builtin_bit_cast(unsigned, x), 0x76543210U, 0xfedcba98U, false, false));
}
__device__ __forceinline__ float exp2_hw(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
    const unsigned ua = __builtin_bit_cast(unsigned, a), ub = __builtin_bit_cast(unsigned, b);
    const unsigned ra = ua + 0x7fffU + ((ua >> 16) & 1U), rb = ub + 0x7fffU + ((ub >> 16) & 1U);
    return __builtin_amdgcn_perm(rb, ra, 0x07060302U);
}
__device__ __forceinline__ unsigned slot(unsigned r) { return (r & ~15U) + 2U * (r & 7U) + ((r >> 3U) & 1U); }

struct Args {
    Queries q;
    KeyBlock blocks[kMaxBlocksPerLaunch];
    BlockLut luts[kMaxBlocksPerLaunch];
    Partials p;
    unsigned slot0;
    unsigned* lut_errors;
};

template <unsigned kStage>
__global__ __launch_bounds__(256) void sparse_flash_kernel(Args a) {
    constexpr unsigned D = 128U, W = 8U, kThreads = 32U * W, kRowsWg = 16U * W;
    constexpr unsigned kKP = D + 8U, NC = D / 16U, NS = 8U * D / kThreads, KCH = D / 8U;
    constexpr unsigned kKPitch = kKP, kVPitch = kVP, KT = 64U * kKPitch;
    __shared__ __align__(16) unsigned short smem[KT + D * kVPitch];
    unsigned short* const ks = smem;
    unsigned short* const vs = smem + KT;
    const auto kaddr = [](unsigned R, unsigned chunk) { return R * kKPitch + 8U * chunk; };
    const auto vaddr = [](unsigned R, unsigned chunk) { return R * kVPitch + 8U * chunk; };
    const auto frag = [](const unsigned short* base, unsigned lo, unsigned hi) {
        const u32x4 x = *reinterpret_cast<const u32x4*>(base + lo), y = *reinterpret_cast<const u32x4*>(base + hi);
        return u32x8{x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
    };
    const unsigned tid = threadIdx.x, lane = tid & 31U, n = lane & 15U, hh = lane >> 4U;
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(tid >> 5U)));
    const unsigned head = blockIdx.y, z = blockIdx.z, tile = blockIdx.x;
    const std::uint16_t* kp = a.blocks[0].k;
    const std::uint16_t* vp = a.blocks[0].vt;
    std::uint64_t k_hs = a.blocks[0].k_hs, k_rs = a.blocks[0].k_rs, vt_hs = a.blocks[0].vt_hs, vt_ds = a.blocks[0].vt_ds;
    unsigned keys = a.blocks[0].keys;
    const std::int32_t* ent = a.luts[0].entries;
    const std::uint32_t* off = a.luts[0].offsets;
    unsigned tile0 = a.luts[0].tile0;
#pragma unroll
    for (unsigned b = 1; b < kMaxBlocksPerLaunch; ++b)
        if (z == b) {
            kp = a.blocks[b].k; vp = a.blocks[b].vt; k_hs = a.blocks[b].k_hs; k_rs = a.blocks[b].k_rs;
            vt_hs = a.blocks[b].vt_hs; vt_ds = a.blocks[b].vt_ds; keys = a.blocks[b].keys;
            ent = a.luts[b].entries; off = a.luts[b].offsets; tile0 = a.luts[b].tile0;
        }
    kp += head * k_hs;
    vp += head * vt_hs;
    const unsigned rows = a.q.rows, row = tile * kRowsWg + wave * 16U + n;
    const bool row_ok = row < rows;

    // LUT walk (uniform): next valid local tile index, or -1 at the end of the list. Invalid entries (out of the
    // block, not strictly ascending) are skipped and counted; nothing past offsets[list + 1] is read.
    const unsigned list = tile * a.q.heads + head;
    const unsigned e_begin = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(off[list])));
    const unsigned e_end = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(off[list + 1U])));
    unsigned e = e_begin;
    // The first kLutStage entries of the list are staged in LDS at work-group start, so a step never waits on a
    // global load to know its next tile (entries beyond the stage, lists of > 1024 tiles, are read from global).
    __shared__ int lut_s[kStage];
    const unsigned n_stage = e_end - e_begin < kStage ? e_end - e_begin : kStage;
    for (unsigned i = tid; i < n_stage; i += kThreads) lut_s[i] = ent[e_begin + i];
    __syncthreads();
    const int ntiles = static_cast<int>((keys + 63U) / 64U);
    int prev = -1;
    unsigned bad = 0;
    const auto next_tile = [&]() -> int {
        while (e < e_end) {
            const unsigned k = e - e_begin;
            const int j = __builtin_amdgcn_readfirstlane(k < n_stage ? lut_s[k] : ent[e]);
            ++e;
            const int lt = j - static_cast<int>(tile0);
            if (lt >= 0 && lt < ntiles && lt > prev) {
                prev = lt;
                return lt;
            }
            ++bad;
        }
        return -1;
    };

    u32x8 qf[NC];
    {
        const std::uint16_t* qr = a.q.q + head * a.q.q_hs + static_cast<std::uint64_t>(row_ok ? row : 0U) * a.q.q_rs;
#pragma unroll
        for (unsigned c = 0; c < NC; ++c) {
            const u32x4 lo = g16(qr + 16 * c), hi = g16(qr + 16 * c + 8);
            qf[c] = row_ok ? u32x8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w} : u32x8{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
    u32x4 pre[NS], vpre[NS];
    const auto load_k = [&](unsigned key0) {
#pragma unroll
        for (unsigned i = 0; i < NS; ++i) {
            const unsigned c = tid + kThreads * i, key = key0 + c / KCH, part = c % KCH;
            pre[i] = key < keys ? g16(kp + key * k_rs + 8U * part) : u32x4{0, 0, 0, 0};
        }
    };
    const auto store_k = [&]() {
#pragma unroll
        for (unsigned i = 0; i < NS; ++i) {
            const unsigned c = tid + kThreads * i;
            *reinterpret_cast<u32x4*>(ks + kaddr(slot(c / KCH), c % KCH)) = pre[i];
        }
    };
    const auto load_v = [&](unsigned key0) {
#pragma unroll
        for (unsigned i = 0; i < NS; ++i) {
            const unsigned c = tid + kThreads * i, d = c >> 3U, j = key0 + 8U * (c & 7U);
            u32x4 x{0, 0, 0, 0};
            if (j < keys) {
                x = g16(vp + d * vt_ds + j);
                if (j + 8U > keys) {
                    const unsigned cnt = keys - j;
#pragma unroll
                    for (unsigned w = 0; w < 4; ++w) {
                        const unsigned keep = 2U * w + 1U < cnt ? 0xffffffffU : 2U * w < cnt ? 0x0000ffffU : 0U;
                        x[w] &= keep;
                    }
                }
            }
            vpre[i] = x;
        }
    };
    const auto store_v = [&]() {
#pragma unroll
        for (unsigned i = 0; i < NS; ++i) {
            const unsigned c = tid + kThreads * i;
            *reinterpret_cast<u32x4*>(vs + vaddr(slot(c >> 3U), c & 7U)) = vpre[i];
        }
    };

    f8 o[NC];
#pragma unroll
    for (unsigned d = 0; d < NC; ++d) o[d] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    const float ninf = -__builtin_inff();
    float m = ninf, mc = ninf, lh = 0.0F;
    bool out_of_bound = false;
    const unsigned ka = n;
    int cur = next_tile();  // FIRST LUT entry
    if (cur >= 0) {
        load_k(64U * static_cast<unsigned>(cur));
        store_k();
        load_v(64U * static_cast<unsigned>(cur));
        store_v();
        lds_barrier();
    }
#pragma unroll 1
    while (cur >= 0) {
        const unsigned key0 = 64U * static_cast<unsigned>(cur);
        const int nxt = next_tile();  // NEXT LUT entry (prefetch target)
        const bool more = nxt >= 0;
        const unsigned key_next = more ? 64U * static_cast<unsigned>(nxt) : 0U;
        if (more) {
            load_k(key_next);
            load_v(key_next);
        }
        f8 s[4];
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) s[t] = f8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (unsigned c = 0; c < NC; ++c)
#pragma unroll
            for (unsigned t = 0; t < 4; ++t) s[t] = wmma(frag(ks, kaddr(16U * t + ka, 2U * c), kaddr(16U * t + ka, 2U * c + 1U)), qf[c], s[t]);
        if (key0 + 64U > keys) {
#pragma unroll
            for (unsigned t = 0; t < 4; ++t)
#pragma unroll
                for (unsigned q8 = 0; q8 < 8; ++q8) s[t][q8] = key0 + 16U * t + 8U * hh + q8 < keys ? s[t][q8] : ninf;
        }
        float alpha = 1.0F, sum = 0.0F;
        unsigned x[4][4];
        {
            float mx = s[0][0];
#pragma unroll
            for (unsigned t = 0; t < 4; ++t)
#pragma unroll
                for (unsigned q8 = 0; q8 < 8; ++q8) mx = __builtin_fmaxf(mx, s[t][q8]);
            mx = __builtin_fmaxf(mx, other_half(mx));
            out_of_bound |= !(__builtin_fabsf(mx * kC) <= kScoreBound);
            const float m_new = __builtin_fmaxf(m, mx), mc_new = m_new * kC;
            const bool none = mc_new == ninf;
            alpha = none ? 1.0F : exp2_hw(mc - mc_new);
            const float mc_sub = none ? 0.0F : mc_new;
#pragma unroll
            for (unsigned t = 0; t < 4; ++t) {
                float p[8];
#pragma unroll
                for (unsigned q8 = 0; q8 < 8; ++q8) {
                    p[q8] = exp2_hw(__builtin_fmaf(s[t][q8], kC, -mc_sub));
                    sum = sum + p[q8];
                }
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) x[t][j] = pack_bf16(p[2 * j], p[2 * j + 1]);
            }
            lh = lh * alpha;
            lh = lh + sum;
            m = m_new;
            mc = mc_new;
        }
        lds_barrier();  // A: every wave is done with K(cur)
        if (more) store_k();
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0F) != 0U) {
#pragma unroll
            for (unsigned d = 0; d < NC; ++d)
#pragma unroll
                for (unsigned q8 = 0; q8 < 8; ++q8) o[d][q8] = o[d][q8] * alpha;
        }
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) {
            u32x8 f;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                f[j] = static_cast<unsigned>(__builtin_amdgcn_ds_swizzle(static_cast<int>(x[t][j]), 0x000f));
                f[4 + j] = static_cast<unsigned>(__builtin_amdgcn_ds_swizzle(static_cast<int>(x[t][j]), 0x020f));
            }
#pragma unroll
            for (unsigned d = 0; d < NC; ++d) o[d] = wmma(frag(vs, vaddr(16U * d + ka, 2U * t), vaddr(16U * d + ka, 2U * t + 1U)), f, o[d]);
        }
        lds_barrier();  // B: every wave is done with V(cur); K(nxt) is visible
        if (more) store_v();
        cur = nxt;
    }
    if (bad && tid == 0U && a.lut_errors) atomicAdd(a.lut_errors, bad);
    const float l = lh + other_half(lh);
    if (a.p.bound_rows && row_ok && hh == 0U && out_of_bound) atomicAdd(a.p.bound_rows, 1U);
    if (row_ok) {
        const std::uint64_t pr = (static_cast<std::uint64_t>(a.slot0 + z) * a.p.heads + head) * a.p.rows + row;
        float* op = a.p.o + pr * D + 8U * hh;
#pragma unroll
        for (unsigned d = 0; d < NC; ++d) {
            s16f(op + 16U * d, f4{o[d][0], o[d][1], o[d][2], o[d][3]});
            s16f(op + 16U * d + 4U, f4{o[d][4], o[d][5], o[d][6], o[d][7]});
        }
        if (hh == 0U) {
            a.p.ml[2U * pr] = mc;
            a.p.ml[2U * pr + 1U] = l;
        }
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0U; }

}  // namespace

static void launch_sparse_partials_stage(const Queries& q, const KeyBlock* blocks, const BlockLut* luts, unsigned nblocks, const Partials& p, unsigned slot0,
                                  unsigned* lut_errors, void* stream, unsigned stage) {
    if (!nblocks || nblocks > kMaxBlocksPerLaunch || q.dim != 128U || !q.q || !q.rows || !q.heads || !aligned16(q.q) || q.q_hs % 8U || q.q_rs % 8U ||
        p.rows != q.rows || p.heads != q.heads || p.dim != q.dim || !p.o || !p.ml || !aligned16(p.o) || (reinterpret_cast<std::uintptr_t>(p.ml) & 7U))
        throw std::invalid_argument("h3 sparse attn: queries / partials");
    Args a{};
    a.q = q;
    for (unsigned b = 0; b < nblocks; ++b) {
        const KeyBlock& kb = blocks[b];
        if (kb.keys && (!kb.k || !kb.vt || !aligned16(kb.k) || !aligned16(kb.vt) || kb.k_hs % 8U || kb.k_rs % 8U || kb.vt_hs % 8U || kb.vt_ds % 8U ||
                        kb.vt_ds < (kb.keys + 7ULL) / 8ULL * 8ULL))
            throw std::invalid_argument("h3 sparse attn: key block layout");
        if (!luts[b].entries && !luts[b].offsets) throw std::invalid_argument("h3 sparse attn: missing LUT");
        if (!luts[b].offsets) throw std::invalid_argument("h3 sparse attn: missing LUT offsets");
        a.blocks[b] = kb;
        a.luts[b] = luts[b];
    }
    a.p = p;
    a.slot0 = slot0;
    a.lut_errors = lut_errors;
    const auto s = static_cast<hipStream_t>(stream);
    static_cast<void>(hipGetLastError());
    const dim3 grid((q.rows + 127U) / 128U, q.heads, nblocks);
    if (stage == 4U) hipLaunchKernelGGL(sparse_flash_kernel<4U>, grid, dim3(256), 0, s, a);
    else hipLaunchKernelGGL(sparse_flash_kernel<kLutStage>, grid, dim3(256), 0, s, a);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 sparse attn: launch: ") + hipGetErrorString(e));
}

void launch_sparse_partials(const Queries& q, const KeyBlock* blocks, const BlockLut* luts, unsigned nblocks, const Partials& p, unsigned slot0,
                            unsigned* lut_errors, void* stream) {
    launch_sparse_partials_stage(q, blocks, luts, nblocks, p, slot0, lut_errors, stream, kLutStage);
}
void launch_sparse_partials_small_stage_for_test(const Queries& q, const KeyBlock* blocks, const BlockLut* luts, unsigned nblocks, const Partials& p,
                                                 unsigned slot0, unsigned* lut_errors, void* stream) {
    launch_sparse_partials_stage(q, blocks, luts, nblocks, p, slot0, lut_errors, stream, 4U);
}

}  // namespace strix::h3::levers::sparse
