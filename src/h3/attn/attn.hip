// MiniMax H3 dense non-causal flash attention, head dim 128 (gfx1151, wave32, bf16 WMMA 16x16x16 -> fp32).
// Contract: attn_ref.hpp (arithmetic), attn.hpp (layouts). Design notes: one work-group = 256 (default, 16 waves) or 128
// (8 waves) query rows of one head, 16 rows per wave, 64-key steps; the grid runs row tiles fastest so the resident
// work-groups stream one head's K/V together (L2 serves the re-reads).
//
// Transposed products keep the softmax lane-local:
//   S^T = K Q^T  (A = K tile from LDS, B = Q^T in registers)   C lane l: query l % 16, keys 8 (l / 16) + e
//   O^T = V^T P^T (A = V^T tile from LDS, B = P^T in registers) C lane l: query l % 16, dims 8 (l / 16) + e
// The A operands read their 16 rows in the order 8 (m & 1) + (m >> 1), which makes each lane's 8 C values 8
// consecutive keys (dims) of one query. P^T (B: lane = query, 16 keys) is assembled from the packed bf16 P of the
// two half-waves with ds_swizzle broadcasts.
#include "h3/attn/attn.hpp"

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace strix::h3::attn {
namespace {

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned short bf16x16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));

constexpr unsigned kVP = 72;  // V^T LDS pitch in halves (64 keys + 8)
// fp32(log2(e) / sqrt(D)) of the head dims the kernel is built for (sqrt(128) = 11.3137..., sqrt(64) = 8).
template <unsigned D> constexpr float scale_log2() {
    static_assert(D == 128U || D == 64U);
    return static_cast<float>(D == 128U ? 1.4426950408889634073599 / 11.313708498984760390231 : 1.4426950408889634073599 / 8.0);
}
// Telemetry bound: every 64-key step's max score s satisfies |s c| <= 2^15 (log2 units) under the operand contract
// (|s c| <= ||q|| ||k|| c <= 2^14, see attn.hpp); then mc = round(m c) is within 2^-10 of m c, so the p of the
// running maximum is within 2^(+-2^-10) of 1 (keys of a later step below that maximum may all have p ~ 0, which is
// the exact behaviour). A (row, head, block) with any step beyond it (or NaN) is counted once.
constexpr float kScoreBound = 32768.0F;

__device__ __forceinline__ u32x4 g16(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p));
}
__device__ __forceinline__ void s16(void* p, u32x4 v) {
    *reinterpret_cast<__attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p)) = v;
}
__device__ __forceinline__ void s16f(void* p, f4 v) {
    *reinterpret_cast<__attribute__((address_space(1))) f4*>(reinterpret_cast<std::uintptr_t>(p)) = v;
}
__device__ __forceinline__ f8 wmma(u32x8 a, u32x8 b, f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(bf16x16, a), __builtin_bit_cast(bf16x16, b), c);
}
// LDS-only barrier (the LDS tiles are the only data shared through it).
__device__ __forceinline__ void lds_barrier() { __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ float other_half(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(0U, __builtin_bit_cast(unsigned, x), 0x76543210U, 0xfedcba98U, false, false));
}
__device__ __forceinline__ float exp2_hw(float x) { return __builtin_amdgcn_exp2f(x); }
// RNE to bf16, two values packed (lo, hi). p is finite or NaN (NaN stays NaN).
__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
    const unsigned ua = __builtin_bit_cast(unsigned, a), ub = __builtin_bit_cast(unsigned, b);
    const unsigned ra = ua + 0x7fffU + ((ua >> 16) & 1U), rb = ub + 0x7fffU + ((ub >> 16) & 1U);
    return __builtin_amdgcn_perm(rb, ra, 0x07060302U);
}
__device__ __forceinline__ unsigned short bf16_rne(float a) {
    const unsigned u = __builtin_bit_cast(unsigned, a);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<unsigned short>((u >> 16) | 0x40U);
    return static_cast<unsigned short>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}
// Merge weight of a slot: an empty partial (mc = -inf, l = 0, o = 0) weighs 0, also when every slot is empty.
__device__ __forceinline__ float slot_weight(float mc, float M) { return mc == -__builtin_inff() ? 0.0F : exp2_hw(mc - M); }
// out = O / L, and 0 when L == 0 (every slot empty).
__device__ __forceinline__ float normalise(float O, float L) { return L == 0.0F ? 0.0F : O / L; }
// The canonical finalisation of one merged value set (shared by the merge and the fused path so their bits agree).
__device__ __forceinline__ void store_out8(const Output& out, std::uint64_t index, const float* v) {
    unsigned w[4];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) w[i] = static_cast<unsigned>(bf16_rne(v[2 * i])) | (static_cast<unsigned>(bf16_rne(v[2 * i + 1])) << 16);
    s16(out.out + index, u32x4{w[0], w[1], w[2], w[3]});
    if (out.out_f32) {
        s16f(out.out_f32 + index, f4{v[0], v[1], v[2], v[3]});
        s16f(out.out_f32 + index + 4, f4{v[4], v[5], v[6], v[7]});
    }
}

struct Args {
    Queries q;
    KeyBlock blocks[kMaxBlocksPerLaunch];
    Partials p;
    unsigned slot0;
    Output out;
    unsigned nz;  // key blocks of this launch (persistent schedule)
};

// A operand row order: lane m of a 16-row fragment holds row 8 (m & 1) + (m >> 1). The tiles are stored in LDS with
// row r of each 16-row group at slot slot(r) = 2 (r & 7) + (r >> 3) (the inverse), so a fragment reads 16
// consecutive LDS rows (b128 reads of 8 consecutive rows start on distinct 4-bank groups for both pitches).
__device__ __forceinline__ unsigned slot(unsigned r) { return (r & ~15U) + 2U * (r & 7U) + ((r >> 3U) & 1U); }

// Ablate (bench only, wrong results): 1 no softmax VALU, 2 no K/V tile loads after the first, 4 no PV WMMA, 8 no QK WMMA.
// D: head dim (128 for the DiT, 64 for the VAE decoder), a multiple of 32.
// W: waves per work-group (8: 128 query rows, 16: 256 rows); a row's arithmetic does not depend on W.
// Early: issue both next-tile loads at the top of a step (one full step ahead) instead of K after QK and V after
// barrier A. Scheduling only: same values.
// Persistent: a grid of (resident work-groups) walks the (block, head, row tile) items in that order, item g + k G
// on work-group g, so the work-groups of a round start together and stream the same K/V tiles at about the same
// time (L2 reuse) instead of drifting apart head by head. Scheduling only: same values.
// PP (ping-pong): two K/V tile buffers in LDS, unpadded rows with 16-byte chunks XOR-swizzled by the row
// (K: chunk ^ (row % chunks), V^T: chunk ^ ((row / 2) % 8)), 64 KB per work-group; the next tile is stored into the
// other buffer during the current step and each step ends with ONE barrier. Same values in the same order.
// Tune (scheduling only, same values): 4 = swizzled unpadded K / V^T rows in a single buffer (layout probe);
// bit 0 builds P^T with v_permlanex16 + selects (VALU) instead of ds_swizzle
// (keeps the LGKM queue for the K / V^T fragment reads); bit 1 interleaves each WMMA with the next fragment's two
// ds_read_b128 (sched_group_barrier), so fragment reads run one WMMA ahead.
template <unsigned D, bool Fused, unsigned Ablate = 0, unsigned W = 8, bool Early = false, bool Persistent = false, bool PP = false,
          unsigned Tune = 0>
__global__ __launch_bounds__(32 * W) void flash_kernel(Args a) {
    static_assert(D % 32U == 0U && D <= 128U && (W == 8U || W == 16U));
    constexpr float kC = scale_log2<D>();
    constexpr unsigned kThreads = 32U * W, kRowsWg = 16U * W;
    // K LDS pitch, dim chunks (WMMA K16), staging chunks per thread, 16-byte chunks per key row
    constexpr unsigned kKP = D + 8U, NC = D / 16U, NS = 8U * D / kThreads, KCH = D / 8U;
    constexpr bool SW = PP || (Tune & 4U) != 0U;  // swizzled unpadded rows (ping-pong, or tune bit 2 with one buffer)
    constexpr unsigned kKPitch = SW ? D : kKP, kVPitch = SW ? 64U : kVP, KT = 64U * kKPitch, VT = D * kVPitch, NBUF = PP ? 2U : 1U;
    __shared__ __align__(16) unsigned short smem[NBUF * (KT + VT)];
    unsigned short* const ks = smem;            // K buffer b at ks + b KT
    unsigned short* const vs = smem + NBUF * KT;  // V^T buffer b at vs + b VT
    // LDS half index of 16-byte chunk `chunk` of LDS row R
    const auto kaddr = [](unsigned R, unsigned chunk) { return R * kKPitch + 8U * (SW ? (chunk ^ (R & (KCH - 1U))) : chunk); };
    const auto vaddr = [](unsigned R, unsigned chunk) { return R * kVPitch + 8U * (SW ? (chunk ^ ((R >> 1U) & 7U)) : chunk); };
    const auto frag = [](const unsigned short* base, unsigned lo, unsigned hi) {
        const u32x4 x = *reinterpret_cast<const u32x4*>(base + lo), y = *reinterpret_cast<const u32x4*>(base + hi);
        return u32x8{x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
    };
    const unsigned tid = threadIdx.x, lane = tid & 31U, n = lane & 15U, hh = lane >> 4U;
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(tid >> 5U)));
    const unsigned tiles = (a.q.rows + kRowsWg - 1U) / kRowsWg, per_z = tiles * a.q.heads, items = per_z * a.nz;
#pragma unroll 1
    for (unsigned item = Persistent ? blockIdx.x : 0U; item < (Persistent ? items : 1U); item += Persistent ? gridDim.x : 1U) {
    if (Persistent && item != blockIdx.x) lds_barrier();  // the previous item's LDS reads are done
    const unsigned head = Persistent ? item % per_z / tiles : blockIdx.y, z = Persistent ? item / per_z : blockIdx.z;
    const unsigned tile = Persistent ? item % tiles : blockIdx.x;
    // The block's descriptor as scalars (uniform selects; no private copy of the argument array).
    const std::uint16_t* kp = a.blocks[0].k;
    const std::uint16_t* vp = a.blocks[0].vt;
    std::uint64_t k_hs = a.blocks[0].k_hs, k_rs = a.blocks[0].k_rs, vt_hs = a.blocks[0].vt_hs, vt_ds = a.blocks[0].vt_ds;
    unsigned keys = a.blocks[0].keys;
#pragma unroll
    for (unsigned b = 1; b < kMaxBlocksPerLaunch; ++b)
        if (z == b) {
            kp = a.blocks[b].k; vp = a.blocks[b].vt; k_hs = a.blocks[b].k_hs; k_rs = a.blocks[b].k_rs;
            vt_hs = a.blocks[b].vt_hs; vt_ds = a.blocks[b].vt_ds; keys = a.blocks[b].keys;
        }
    kp += head * k_hs;
    vp += head * vt_hs;
    const unsigned rows = a.q.rows, row = tile * kRowsWg + wave * 16U + n;
    const bool row_ok = row < rows;

    // Q^T B fragments: lane = query n, 16 dims per fragment (both half-waves hold the same row).
    u32x8 qf[NC];
    {
        const std::uint16_t* qr = a.q.q + head * a.q.q_hs + static_cast<std::uint64_t>(row_ok ? row : 0U) * a.q.q_rs;
#pragma unroll
        for (unsigned c = 0; c < NC; ++c) {
            const u32x4 lo = g16(qr + 16 * c), hi = g16(qr + 16 * c + 8);
            qf[c] = row_ok ? u32x8{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w} : u32x8{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }
    // Staging: K tile = 64 keys x 16 chunks of 16 B; V^T tile = 128 dims x 8 chunks of 16 B; 4 chunks per thread each.
    u32x4 pre[NS], vpre[NS];
    const auto load_k = [&](unsigned key0) {
#pragma unroll
        for (unsigned i = 0; i < NS; ++i) {
            const unsigned c = tid + kThreads * i, key = key0 + c / KCH, part = c % KCH;
            pre[i] = key < keys ? g16(kp + key * k_rs + 8U * part) : u32x4{0, 0, 0, 0};
        }
    };
    const auto store_k = [&](unsigned buf) {
#pragma unroll
        for (unsigned i = 0; i < NS; ++i) {
            const unsigned c = tid + kThreads * i;
            *reinterpret_cast<u32x4*>(ks + buf * KT + kaddr(slot(c / KCH), c % KCH)) = pre[i];
        }
    };
    const auto load_v = [&](unsigned key0) {
#pragma unroll
        for (unsigned i = 0; i < NS; ++i) {
            const unsigned c = tid + kThreads * i, d = c >> 3U, j = key0 + 8U * (c & 7U);
            u32x4 x{0, 0, 0, 0};
            if (j < keys) {
                x = g16(vp + d * vt_ds + j);
                if (j + 8U > keys) {  // the block's last chunk: keys >= keys are zero
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
    const auto store_v = [&](unsigned buf) {
#pragma unroll
        for (unsigned i = 0; i < NS; ++i) {
            const unsigned c = tid + kThreads * i;
            *reinterpret_cast<u32x4*>(vs + buf * VT + vaddr(slot(c >> 3U), c & 7U)) = vpre[i];
        }
    };

    f8 o[NC];
#pragma unroll
    for (unsigned d = 0; d < NC; ++d) o[d] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    const float ninf = -__builtin_inff();
    float m = ninf, mc = ninf, lh = 0.0F;
    bool out_of_bound = false;  // latched: some step's max score beyond kScoreBound (telemetry only)
    const unsigned nsteps = (keys + 63U) / 64U;
    const unsigned ka = n;  // LDS row of this lane within a 16-row fragment (see slot())
    if (nsteps) {
        load_k(0);
        store_k(0);
        load_v(0);
        store_v(0);
        lds_barrier();
    }
#pragma unroll 1
    for (unsigned step = 0; step < nsteps; ++step) {
        const unsigned key0 = step * 64U;
        const bool more = (Ablate & 2U) == 0U && step + 1U < nsteps;
        const unsigned cur = PP ? (step & 1U) : 0U;
        if ((Early || PP) && more) {
            load_k(key0 + 64U);
            load_v(key0 + 64U);
        }
        // S^T: 4 key tiles, K16 dim chunks ascending per accumulator.
        f8 s[4];
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) s[t] = f8{0, 0, 0, 0, 0, 0, 0, 0};
        if constexpr ((Ablate & 8U) == 0U) {
#pragma unroll
            for (unsigned c = 0; c < NC; ++c)
#pragma unroll
                for (unsigned t = 0; t < 4; ++t)
                    s[t] = wmma(frag(ks + cur * KT, kaddr(16U * t + ka, 2U * c), kaddr(16U * t + ka, 2U * c + 1U)), qf[c], s[t]);
            if constexpr ((Tune & 2U) != 0U) {
                __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);  // the first two fragments' reads
#pragma unroll
                for (unsigned i = 0; i < 4U * NC; ++i) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // WMMA i
                    __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);  // reads of fragment i + 2
                }
            }
        } else {
#pragma unroll
            for (unsigned t = 0; t < 4; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) s[t][e] = __builtin_bit_cast(float, qf[t][e] & 0x3f00ffffU) + static_cast<float>(ks[ka + t]);
        }
        if (!Early && !PP && more) load_k(key0 + 64U);
        if (key0 + 64U > keys) {
#pragma unroll
            for (unsigned t = 0; t < 4; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) s[t][e] = key0 + 16U * t + 8U * hh + e < keys ? s[t][e] : ninf;
        }
        float alpha = 1.0F, sum = 0.0F;
        unsigned x[4][4];
        if constexpr ((Ablate & 1U) != 0U) {
#pragma unroll
            for (unsigned t = 0; t < 4; ++t)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) x[t][j] = __builtin_bit_cast(unsigned, s[t][2 * j]) ^ __builtin_bit_cast(unsigned, s[t][2 * j + 1]);
        } else {
        float mx = s[0][0];
#pragma unroll
        for (unsigned t = 0; t < 4; ++t)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) mx = __builtin_fmaxf(mx, s[t][e]);
        mx = __builtin_fmaxf(mx, other_half(mx));
        out_of_bound |= !(__builtin_fabsf(mx * kC) <= kScoreBound);
        const float m_new = __builtin_fmaxf(m, mx), mc_new = m_new * kC;
        const bool none = mc_new == ninf;  // no key seen yet: p = 0, alpha = 1
        alpha = none ? 1.0F : exp2_hw(mc - mc_new);
        const float mc_sub = none ? 0.0F : mc_new;
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) {
            float p[8];
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                p[e] = exp2_hw(__builtin_fmaf(s[t][e], kC, -mc_sub));
                sum = sum + p[e];
            }
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) x[t][j] = pack_bf16(p[2 * j], p[2 * j + 1]);
        }
        lh = lh * alpha;
        lh = lh + sum;
        m = m_new;
        mc = mc_new;
        }
        if constexpr (PP) {
            if (more) {  // the other buffer: its readers (step - 1) passed the previous barrier
                store_k(cur ^ 1U);
                store_v(cur ^ 1U);
            }
        } else {
            lds_barrier();  // A: every wave is done with K(step)
            if (more) {
                store_k(0);
                if (!Early) load_v(key0 + 64U);
            }
        }
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0F) != 0U) {
#pragma unroll
            for (unsigned d = 0; d < NC; ++d)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) o[d][e] = o[d][e] * alpha;
        }
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) {
            u32x8 f;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                if constexpr ((Tune & 1U) != 0U) {
                    // the other half-wave's packed pair: lanes < 16 hold keys 2j, 2j+1 (low half), lanes >= 16 keys 8+2j, 9+2j
                    const unsigned xp = __builtin_amdgcn_permlanex16(0U, x[t][j], 0x76543210U, 0xfedcba98U, false, false);
                    f[j] = hh ? xp : x[t][j];
                    f[4 + j] = hh ? x[t][j] : xp;
                } else {
                    f[j] = static_cast<unsigned>(__builtin_amdgcn_ds_swizzle(static_cast<int>(x[t][j]), 0x000f));      // keys 2j, 2j+1
                    f[4 + j] = static_cast<unsigned>(__builtin_amdgcn_ds_swizzle(static_cast<int>(x[t][j]), 0x020f));  // keys 8+2j, 9+2j
                }
            }
            if constexpr ((Ablate & 4U) == 0U) {
#pragma unroll
                for (unsigned d = 0; d < NC; ++d)
                    o[d] = wmma(frag(vs + cur * VT, vaddr(16U * d + ka, 2U * t), vaddr(16U * d + ka, 2U * t + 1U)), f, o[d]);
                if constexpr ((Tune & 2U) != 0U) {
                    __builtin_amdgcn_sched_group_barrier(0x100, 4, 1);
#pragma unroll
                    for (unsigned i = 0; i < NC; ++i) {
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
                        __builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
                    }
                }
            } else {
                o[t % NC][0] = o[t % NC][0] + __builtin_bit_cast(float, (f[0] ^ f[5]) & 0x3f00ffffU);
            }
        }
        lds_barrier();  // B: every wave is done with V(step); K(step + 1) is visible (PP: both tiles of step + 1)
        if (!PP && more) store_v(0);
    }
    const float l = lh + other_half(lh);
    {   // telemetry (not a reduction): rows whose scaled running max leaves the qualified bound, or is NaN
        unsigned* bound = Fused ? a.out.bound_rows : a.p.bound_rows;
        if (bound && row_ok && hh == 0U && out_of_bound) atomicAdd(bound, 1U);
    }
    if (row_ok) {
    if constexpr (Fused) {
        // == launch_merge with one slot: M = mc, e = exp2(mc - M), L = l e, O = o e, out = O / L.
        const float M = mc, eb = slot_weight(mc, M), L = l * eb;
        const std::uint64_t base = static_cast<std::uint64_t>(row) * a.out.out_rs + head * D + 8U * hh;
#pragma unroll
        for (unsigned d = 0; d < NC; ++d) {
            float v[8];
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const float O = o[d][e] * eb;
                v[e] = normalise(O, L);
            }
            store_out8(a.out, base + 16U * d, v);
        }
    } else {
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
    }  // row_ok
    }  // items
}

// One thread per (row, head, 8 dims); slots merged in index order.
__global__ __launch_bounds__(256) void merge_kernel(Partials p, unsigned nslots, Output out) {
    const std::uint64_t idx = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
    const unsigned groups = p.dim / 8U;
    const std::uint64_t total = static_cast<std::uint64_t>(p.rows) * p.heads * groups;
    if (idx >= total) return;
    const unsigned g = static_cast<unsigned>(idx % groups);
    const std::uint64_t rh = idx / groups;
    const unsigned head = static_cast<unsigned>(rh % p.heads), row = static_cast<unsigned>(rh / p.heads);
    const std::uint64_t slot_stride = static_cast<std::uint64_t>(p.heads) * p.rows, pr0 = static_cast<std::uint64_t>(head) * p.rows + row;
    float M = p.ml[2U * pr0];
    for (unsigned b = 1; b < nslots; ++b) M = __builtin_fmaxf(M, p.ml[2U * (pr0 + b * slot_stride)]);
    float O[8], L = 0.0F;
    for (unsigned b = 0; b < nslots; ++b) {
        const std::uint64_t pr = pr0 + b * slot_stride;
        const float eb = slot_weight(p.ml[2U * pr], M), lb = p.ml[2U * pr + 1U] * eb;
        const float* op = p.o + pr * p.dim + 8U * g;
        const f4 lo = *reinterpret_cast<const f4*>(op), hi = *reinterpret_cast<const f4*>(op + 4);
        const float ov[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        if (b == 0) {
            L = lb;
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) O[e] = ov[e] * eb;
        } else {
            L = L + lb;
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const float t = ov[e] * eb;
                O[e] = O[e] + t;
            }
        }
    }
    float v[8];
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) v[e] = normalise(O[e], L);
    store_out8(out, static_cast<std::uint64_t>(row) * out.out_rs + head * p.dim + 8U * g, v);
}

// 32 keys x 32 dims per block through LDS.
__global__ __launch_bounds__(256) void transpose_kernel(const std::uint16_t* v, std::uint64_t v_hs, std::uint64_t v_rs, unsigned keys,
                                                        std::uint16_t* vt, std::uint64_t vt_hs, std::uint64_t vt_ds) {
    __shared__ std::uint16_t tile[32][33];
    const unsigned tx = threadIdx.x & 31U, ty = threadIdx.x >> 5U;
    const unsigned j0 = blockIdx.x * 32U, d0 = blockIdx.y * 32U, head = blockIdx.z;
    for (unsigned i = ty; i < 32U; i += 8U) {
        const unsigned j = j0 + i;
        tile[i][tx] = j < keys ? v[head * v_hs + j * v_rs + d0 + tx] : static_cast<std::uint16_t>(0);
    }
    __syncthreads();
    for (unsigned i = ty; i < 32U; i += 8U) {
        const unsigned j = j0 + tx;
        if (j < keys) vt[head * vt_hs + (d0 + i) * vt_ds + j] = tile[tx][i];
    }
}

// Strict environment switches: unset -> default; otherwise the value must be exactly one of the documented decimal
// strings, else std::invalid_argument at first use (before anything is enqueued). No partial parsing or masking.
int env_choice(const char* name, int dflt, std::initializer_list<int> allowed) {
    const char* v = std::getenv(name);
    if (!v) return dflt;
    std::string list;
    for (const int a : allowed) {
        if (std::to_string(a) == v) return a;
        list += (list.empty() ? "" : ", ") + std::to_string(a);
    }
    throw std::invalid_argument(std::string("h3 attn: ") + name + "='" + v + "' is not one of {" + list + "}");
}
// STRIX_H3_ATTN_WAVES: 16 = 256-row query tiles (the default: +4 % median in a paired per-block bench at the 2-node
// per-block shape; 226 VGPRs -> 6 waves/SIMD is the register-derived ceiling, not a measured residency), 8 = 128-row
// tiles (the previous default). Same bits either way.
unsigned g_waves = 0;  // 0: from the environment
int g_early = -1;      // -1: from the environment (default on; STRIX_H3_ATTN_EARLY=0 restores the late schedule)
bool early_loads() {
    static const bool env = env_choice("STRIX_H3_ATTN_EARLY", 1, {0, 1}) == 1;
    return g_early < 0 ? env : g_early == 1;
}
unsigned waves_per_wg() {
    static const unsigned env = static_cast<unsigned>(env_choice("STRIX_H3_ATTN_WAVES", 16, {8, 16}));
    return g_waves ? g_waves : env;
}
int g_persist = -1;  // -1: from the environment (STRIX_H3_ATTN_PERSIST=1)
bool persistent() {
    static const bool env = env_choice("STRIX_H3_ATTN_PERSIST", 0, {0, 1}) == 1;
    return g_persist < 0 ? env : g_persist == 1;
}
template <class K> unsigned resident_groups(K kernel, unsigned threads) {
    int per = 0, dev = 0;
    hipDeviceProp_t prop;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, static_cast<int>(threads), 0) != hipSuccess || per < 1)
        return 0;
    return static_cast<unsigned>(per) * static_cast<unsigned>(prop.multiProcessorCount);
}
int g_pingpong = -1;  // -1: from the environment (STRIX_H3_ATTN_PINGPONG=1)
bool pingpong() {
    static const bool env = env_choice("STRIX_H3_ATTN_PINGPONG", 0, {0, 1}) == 1;
    return g_pingpong < 0 ? env : g_pingpong == 1;
}
template <unsigned D, bool Fused, bool Early, unsigned W, bool PP> void launch_flash_wp(const Args& a, unsigned nblocks, hipStream_t s) {
    const unsigned rows_wg = 16U * W, items = (a.q.rows + rows_wg - 1U) / rows_wg * a.q.heads * nblocks;
    if (persistent()) {
        static const unsigned groups = resident_groups(flash_kernel<D, Fused, 0, W, Early, true, PP>, 32U * W);
        if (groups) {
            hipLaunchKernelGGL((flash_kernel<D, Fused, 0, W, Early, true, PP>), dim3(groups < items ? groups : items), dim3(32U * W), 0, s, a);
            return;
        }
    }
    hipLaunchKernelGGL((flash_kernel<D, Fused, 0, W, Early, false, PP>), dim3((a.q.rows + rows_wg - 1U) / rows_wg, a.q.heads, nblocks), dim3(32U * W), 0, s, a);
}
int g_tune = -1;  // -1: from the environment (STRIX_H3_ATTN_TUNE = bit mask, default 0)
unsigned tune() {
    static const unsigned env = static_cast<unsigned>(env_choice("STRIX_H3_ATTN_TUNE", 0, {0, 1, 2, 3, 4}));
    return g_tune < 0 ? env : static_cast<unsigned>(g_tune);
}
template <unsigned D, bool Fused, bool Early, unsigned W> void launch_flash_ww(const Args& a, unsigned nblocks, hipStream_t s) {
    // tuned variants exist for the early-load grid single-buffer schedule at 8 or 16 waves
    if (Early && !pingpong() && !persistent() && tune() != 0U) {
        const dim3 grid((a.q.rows + 16U * W - 1U) / (16U * W), a.q.heads, nblocks), threads(32U * W);
        switch (tune()) {
            case 1: hipLaunchKernelGGL((flash_kernel<D, Fused, 0, W, true, false, false, 1>), grid, threads, 0, s, a); return;
            case 2: hipLaunchKernelGGL((flash_kernel<D, Fused, 0, W, true, false, false, 2>), grid, threads, 0, s, a); return;
            case 3: hipLaunchKernelGGL((flash_kernel<D, Fused, 0, W, true, false, false, 3>), grid, threads, 0, s, a); return;
            default: hipLaunchKernelGGL((flash_kernel<D, Fused, 0, W, true, false, false, 4>), grid, threads, 0, s, a); return;
        }
    }
    // ping-pong has its own load timing (the Early flag is irrelevant there): one instantiation per (W, persistent)
    if (pingpong()) launch_flash_wp<D, Fused, true, W, true>(a, nblocks, s);
    else launch_flash_wp<D, Fused, Early, W, false>(a, nblocks, s);
}
template <unsigned D, bool Fused, bool Early> void launch_flash_w(const Args& a, unsigned nblocks, hipStream_t s) {
    if (waves_per_wg() == 16U) launch_flash_ww<D, Fused, Early, 16>(a, nblocks, s);
    else launch_flash_ww<D, Fused, Early, 8>(a, nblocks, s);
}
template <unsigned D, bool Fused> void launch_flash(const Args& a, unsigned nblocks, hipStream_t s) {
    if (early_loads()) launch_flash_w<D, Fused, true>(a, nblocks, s);
    else launch_flash_w<D, Fused, false>(a, nblocks, s);
}

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 attn ") + what + ": " + hipGetErrorString(e));
}
bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0U; }
// Head dims qualified against a host reference: 128 only (the D64 kernels are built but rejected until a
// D-generic reference and tests exist).
bool qualified_dim(unsigned dim) { return dim == 128U; }
void validate(const Queries& q, const KeyBlock& b) {
    if (!qualified_dim(q.dim) || !q.q || !q.heads || !q.rows || !aligned16(q.q) || q.q_hs % 8U || q.q_rs % 8U) throw std::invalid_argument("h3 attn: queries");
    if (b.keys && (!b.k || !b.vt || !aligned16(b.k) || !aligned16(b.vt) || b.k_hs % 8U || b.k_rs % 8U || b.vt_hs % 8U || b.vt_ds % 8U ||
                   b.vt_ds < (b.keys + 7ULL) / 8ULL * 8ULL))
        throw std::invalid_argument("h3 attn: key block layout");
}

}  // namespace

void set_waves_per_wg(unsigned waves) { g_waves = waves == 16U || waves == 8U ? waves : 0U; }
unsigned waves_per_wg_resolved() { return waves_per_wg(); }
void set_early_loads(int early) { g_early = early < 0 ? -1 : early ? 1 : 0; }
void set_persistent(int on) { g_persist = on < 0 ? -1 : on ? 1 : 0; }
void set_pingpong(int on) { g_pingpong = on < 0 ? -1 : on ? 1 : 0; }
void set_tune(int mask) { g_tune = mask < 0 || mask > 4 ? -1 : mask; }

// Head split (scheduling only, same bits): a launch whose key block holds more than kHeadSplitKeys keys (> 12 MiB of
// K/V per head at D128) is issued as one launch per (block, head). With all heads in one grid, the resident
// work-groups straddle two heads at every head transition, and two heads' K/V no longer fit the 32 MB last-level
// cache (fused 37910^2 measured 19.5 TF in one launch vs 34.0 TF as 56 one-head launches).
constexpr unsigned kHeadSplitKeys = 24576;
static int g_head_split = -2;  // -2: environment (STRIX_H3_ATTN_HEAD_SPLIT=0/1; unset = automatic), -1 automatic, 0 off, 1 on
static bool head_split(const KeyBlock* blocks, unsigned nblocks) {
    static const int env = env_choice("STRIX_H3_ATTN_HEAD_SPLIT", -1, {0, 1});
    const int mode = g_head_split == -2 ? env : g_head_split;
    if (mode >= 0) return mode == 1;
    for (unsigned b = 0; b < nblocks; ++b)
        if (blocks[b].keys > kHeadSplitKeys) return true;
    return false;
}
void set_head_split(int mode) { g_head_split = mode < -1 ? -2 : mode > 1 ? 1 : mode; }

void launch_partials(const Queries& q, const KeyBlock* blocks, unsigned nblocks, const Partials& p, unsigned slot0, void* stream) {
    if (!nblocks || nblocks > kMaxBlocksPerLaunch || p.rows != q.rows || p.heads != q.heads || p.dim != q.dim || !p.o || !p.ml ||
        !aligned16(p.o) || (reinterpret_cast<std::uintptr_t>(p.ml) & 7U))
        throw std::invalid_argument("h3 attn: partials");
    for (unsigned b = 0; b < nblocks; ++b) validate(q, blocks[b]);
    const auto s = static_cast<hipStream_t>(stream);
    if (q.heads > 1U && head_split(blocks, nblocks)) {
        for (unsigned b = 0; b < nblocks; ++b)
            for (unsigned h = 0; h < q.heads; ++h) {
                Args a{};
                a.q = q;
                a.q.q += h * q.q_hs;
                a.q.heads = 1;
                a.blocks[0] = blocks[b];
                a.blocks[0].k += h * blocks[b].k_hs;
                a.blocks[0].vt += h * blocks[b].vt_hs;
                const std::size_t pr = (std::size_t(slot0 + b) * p.heads + h) * p.rows;
                a.p = Partials{p.o + pr * p.dim, p.ml + 2U * pr, p.rows, 1, p.dim, p.bound_rows};
                a.slot0 = 0;
                a.nz = 1;
                launch_flash<128, false>(a, 1, s);
            }
        check(hipGetLastError(), "partials launch (head split)");
        return;
    }
    Args a{};
    a.q = q;
    for (unsigned b = 0; b < nblocks; ++b) a.blocks[b] = blocks[b];
    a.p = p;
    a.slot0 = slot0;
    a.nz = nblocks;
    launch_flash<128, false>(a, nblocks, s);  // dim 128 only (validated); D64 not instantiated
    check(hipGetLastError(), "partials launch");
}

void launch_partials_ablate(const Queries& q, const KeyBlock* blocks, unsigned nblocks, const Partials& p, unsigned ablate, void* stream) {
    Args a{};
    a.q = q;
    for (unsigned b = 0; b < nblocks; ++b) { validate(q, blocks[b]); a.blocks[b] = blocks[b]; }
    a.p = p;
    a.nz = nblocks;
    const dim3 grid((q.rows + 127U) / 128U, q.heads, nblocks), threads(256);
    const auto s = static_cast<hipStream_t>(stream);
    switch (ablate) {
        case 1: hipLaunchKernelGGL((flash_kernel<128, false, 1>), grid, threads, 0, s, a); break;
        case 2: hipLaunchKernelGGL((flash_kernel<128, false, 2>), grid, threads, 0, s, a); break;
        case 3: hipLaunchKernelGGL((flash_kernel<128, false, 3>), grid, threads, 0, s, a); break;
        case 4: hipLaunchKernelGGL((flash_kernel<128, false, 4>), grid, threads, 0, s, a); break;
        case 7: hipLaunchKernelGGL((flash_kernel<128, false, 7>), grid, threads, 0, s, a); break;
        case 11: hipLaunchKernelGGL((flash_kernel<128, false, 11>), grid, threads, 0, s, a); break;
        default: throw std::invalid_argument("h3 attn: ablation");
    }
    check(hipGetLastError(), "ablation launch");
}

void launch_fused_single(const Queries& q, const KeyBlock& block, const Output& out, void* stream) {
    validate(q, block);
    if (!block.keys || !out.out || !aligned16(out.out) || out.out_rs % 8U || out.out_rs < std::uint64_t(q.heads) * q.dim ||
        (out.out_f32 && !aligned16(out.out_f32)))
        throw std::invalid_argument("h3 attn: fused output");
    const auto s = static_cast<hipStream_t>(stream);
    const bool split = q.heads > 1U && head_split(&block, 1);
    for (unsigned h = 0; h < (split ? q.heads : 1U); ++h) {
        Args a{};
        a.q = q;
        a.blocks[0] = block;
        a.out = out;
        if (split) {
            a.q.q += h * q.q_hs;
            a.q.heads = 1;
            a.blocks[0].k += h * block.k_hs;
            a.blocks[0].vt += h * block.vt_hs;
            a.out.out += h * q.dim;
            if (a.out.out_f32) a.out.out_f32 += h * q.dim;
        }
        a.nz = 1;
        launch_flash<128, true>(a, 1, s);  // dim 128 only (validated)
    }
    check(hipGetLastError(), "fused launch");
}

void launch_merge(const Partials& p, unsigned nslots, const Output& out, void* stream) {
    if (!qualified_dim(p.dim) || !nslots || !p.o || !p.ml || !aligned16(p.o) || (reinterpret_cast<std::uintptr_t>(p.ml) & 7U) || !out.out ||
        !aligned16(out.out) || out.out_rs % 8U || out.out_rs < std::uint64_t(p.heads) * p.dim || (out.out_f32 && !aligned16(out.out_f32)))
        throw std::invalid_argument("h3 attn: merge");
    const std::uint64_t total = static_cast<std::uint64_t>(p.rows) * p.heads * (p.dim / 8U);
    hipLaunchKernelGGL(merge_kernel, dim3(static_cast<unsigned>((total + 255U) / 256U)), dim3(256), 0, static_cast<hipStream_t>(stream), p, nslots, out);
    check(hipGetLastError(), "merge launch");
}

std::size_t attention_scratch_floats(std::uint32_t rows, std::uint32_t heads, std::uint32_t dim, std::uint32_t nblocks) {
    return std::size_t(nblocks) * heads * rows * (dim + 2U);
}

void attention(const AttentionCall& c, void* stream) {
    // Everything is validated before anything is enqueued.
    if (!qualified_dim(c.dim) || !c.heads || !c.rows || !c.cuts || !c.nblocks || c.nblocks > kMaxBlocksPerLaunch || c.cuts[0] != 0U ||
        c.cuts[c.nblocks] != c.keys)
        throw std::invalid_argument("h3 attn: call");
    const std::uint64_t width = std::uint64_t(c.heads) * c.dim, out_rs = c.out_rs ? c.out_rs : width;
    if (c.vt_pitch % 8U || c.vt_pitch < (std::uint64_t(c.keys) + 7U) / 8U * 8U)
        throw std::invalid_argument("h3 attn: vt_pitch must be a multiple of 8 and >= round8(keys)");
    if (!c.out || !aligned16(c.out) || out_rs % 8U || out_rs < width) throw std::invalid_argument("h3 attn: output (out_rs >= heads * dim)");
    if (!c.scratch || !aligned16(c.scratch)) throw std::invalid_argument("h3 attn: scratch (16-byte aligned)");
    const Queries q{c.q, c.dim, width, c.rows, c.heads, c.dim};
    KeyBlock blocks[kMaxBlocksPerLaunch];
    for (unsigned b = 0; b < c.nblocks; ++b) {
        const std::uint32_t j0 = c.cuts[b], j1 = c.cuts[b + 1];
        if (j1 < j0 || (b > 0 && j0 % 64U)) throw std::invalid_argument("h3 attn: cut table (cuts must be multiples of 64)");
        blocks[b] = KeyBlock{c.k + j0 * width, c.vt + j0, c.dim, width, std::uint64_t(c.dim) * c.vt_pitch, c.vt_pitch, j1 - j0};
    }
    float* o = c.scratch;
    float* ml = o + std::size_t(c.nblocks) * c.heads * c.rows * c.dim;
    const Partials p{o, ml, c.rows, c.heads, c.dim, c.bound_rows};
    for (unsigned b = 0; b < c.nblocks; ++b) validate(q, blocks[b]);
    launch_partials(q, blocks, c.nblocks, p, 0, stream);
    launch_merge(p, c.nblocks, Output{c.out, nullptr, out_rs}, stream);
}

void launch_transpose_v(const std::uint16_t* v, std::uint64_t v_hs, std::uint64_t v_rs, std::uint32_t keys, std::uint32_t heads,
                        std::uint16_t* vt, std::uint64_t vt_hs, std::uint64_t vt_ds, void* stream, std::uint32_t dim) {
    if (!keys) return;
    if (dim % 32U) throw std::invalid_argument("h3 attn: transpose dim");
    hipLaunchKernelGGL(transpose_kernel, dim3((keys + 31U) / 32U, dim / 32U, heads), dim3(256), 0, static_cast<hipStream_t>(stream),
                       v, v_hs, v_rs, keys, vt, vt_hs, vt_ds);
    check(hipGetLastError(), "transpose launch");
}

}  // namespace strix::h3::attn
