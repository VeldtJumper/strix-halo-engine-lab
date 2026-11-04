#pragma once
// Host split-KV attention over the block forward's layouts (Q, K token-major [rows][56][128], V^T [56][128][pitch]):
// one fp32 partial per (query row, head, key block) in the attention package's kernel order (attn_ref.hpp contract,
// the same operations as attn::partial_row_f32), and the canonical merge (blocks in index order) -> bf16.
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "h3/attn/attn_ref.hpp"
#include "h3/block/host_ref.hpp"
#include "h3/block/parallel.hpp"
#include "h3/dist/schedule.hpp"

namespace strix::h3::dist {

// One key block as stored by a rank: K rows token-major (row pitch kInner), V^T keys contiguous.
struct KeyView {
    const std::uint16_t* k = nullptr;   // k[j * kInner + h * 128 + d]
    const std::uint16_t* vt = nullptr;  // vt[(h * 128 + d) * vt_pitch + j]
    std::uint64_t vt_pitch = 0;
    std::uint32_t keys = 0;
};

// attn::partial_row_f32 for one (row, head), reading token-major Q/K and V^T.
inline attn::Partial host_partial(const std::uint16_t* q_head, const KeyView& kv, unsigned head) {
    using attn::bf16_to_float;
    using attn::bf16_from_float;
    constexpr unsigned kStep = attn::kKeyStep;
    const float inf = std::numeric_limits<float>::infinity();
    attn::Partial p;
    float m = -inf, mc = -inf, l_half[2] = {0.0F, 0.0F};
    float q[kHeadDim];
    for (unsigned d = 0; d < kHeadDim; ++d) q[d] = bf16_to_float(q_head[d]);
    const std::uint16_t* kh = kv.k + std::size_t{head} * kHeadDim;
    const std::uint16_t* vh = kv.vt + std::size_t{head} * kHeadDim * kv.vt_pitch;
    for (unsigned key0 = 0; key0 < kv.keys; key0 += kStep) {
        float s[kStep];
        float step_max = -inf;
        for (unsigned j = 0; j < kStep; ++j) {
            const unsigned key = key0 + j;
            if (key >= kv.keys) { s[j] = -inf; continue; }
            const std::uint16_t* kr = kh + std::size_t{key} * kInner;
            float acc = 0.0F;
            for (unsigned c = 0; c < kHeadDim; c += 16) {
                float part = 0.0F;
                for (unsigned d = c; d < c + 16; ++d) part = part + q[d] * bf16_to_float(kr[d]);
                acc = acc + part;
            }
            s[j] = acc;
            step_max = std::max(step_max, acc);
        }
        const float m_new = std::max(m, step_max), mc_new = m_new * attn::kScaleLog2;
        const bool none = mc_new == -inf;
        const float alpha = none ? 1.0F : std::exp2(mc - mc_new), mc_sub = none ? 0.0F : mc_new;
        float pf[kStep];
        std::uint16_t pb[kStep];
        for (unsigned j = 0; j < kStep; ++j) {
            pf[j] = std::exp2(std::fma(s[j], attn::kScaleLog2, -mc_sub));
            pb[j] = bf16_from_float(pf[j]);
        }
        for (unsigned h = 0; h < 2; ++h) {
            float sum = 0.0F;
            for (unsigned t = 0; t < kStep / 16; ++t)
                for (unsigned e = 0; e < 8; ++e) sum = sum + pf[16 * t + 8 * h + e];
            l_half[h] = l_half[h] * alpha + sum;
        }
        for (unsigned d = 0; d < kHeadDim; ++d) {
            const std::uint16_t* vr = vh + std::size_t{d} * kv.vt_pitch;
            float o = p.o[d] * alpha;
            for (unsigned t = 0; t < kStep / 16; ++t) {
                float part = 0.0F;
                for (unsigned j = 16 * t; j < 16 * t + 16; ++j) {
                    const unsigned key = key0 + j;
                    const float v = key < kv.keys ? bf16_to_float(vr[key]) : 0.0F;
                    part = part + bf16_to_float(pb[j]) * v;
                }
                o = o + part;
            }
            p.o[d] = o;
        }
        m = m_new;
        mc = mc_new;
    }
    p.mc = mc;
    p.l = l_half[0] + l_half[1];
    return p;
}

// Partials of n query rows (q [n][56][128]) against one key block -> out [n][56].
inline void host_partials(const std::uint16_t* q, std::uint32_t n, const KeyView& kv, attn::Partial* out) {
    block::parallel_for(std::uint64_t{n} * kHeads, [&](std::uint64_t i) {
        const std::uint32_t r = static_cast<std::uint32_t>(i / kHeads), h = static_cast<std::uint32_t>(i % kHeads);
        out[i] = host_partial(q + (std::size_t{r} * kHeads + h) * kHeadDim, kv, h);
    });
}

// Canonical merge of the per-block partials (parts[b] = [n][56], b = 0 .. nb-1 in index order) -> out [n][7168] bf16.
inline void host_merge(const attn::Partial* const* parts, unsigned nb, std::uint32_t n, std::uint16_t* out) {
    block::parallel_for(std::uint64_t{n} * kHeads, [&](std::uint64_t i) {
        attn::Partial p[4];
        for (unsigned b = 0; b < nb; ++b) p[b] = parts[b][i];
        float o[kHeadDim];
        attn::merge_f32(p, nb, o);
        const std::uint32_t r = static_cast<std::uint32_t>(i / kHeads), h = static_cast<std::uint32_t>(i % kHeads);
        for (unsigned d = 0; d < kHeadDim; ++d) out[std::size_t{r} * kInner + h * kHeadDim + d] = attn::bf16_from_float(o[d]);
    });
}

// The block forward's AttentionOp with the canonical two-block split (1-node reference of the schedule; the refiner
// and S < 128 use one block). Same arithmetic as the dist driver's partials + merge.
struct SplitKvHostAttention final : block::AttentionOp {
    void run(const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* vt, std::uint32_t S, std::uint32_t keys,
        std::uint16_t* out) override {
        const std::uint32_t cut = canonical_cut(keys);
        const std::uint32_t cuts[3] = {0U, cut ? cut : keys, keys};
        const unsigned nb = cut ? 2U : 1U;
        std::vector<std::vector<attn::Partial>> parts(nb, std::vector<attn::Partial>(std::size_t{S} * kHeads));
        const attn::Partial* ptr[2] = {};
        for (unsigned b = 0; b < nb; ++b) {
            const KeyView kv{k + std::size_t{cuts[b]} * kInner, vt + cuts[b], keys, cuts[b + 1] - cuts[b]};
            host_partials(q, S, kv, parts[b].data());
            ptr[b] = parts[b].data();
        }
        host_merge(ptr, nb, S, out);
    }
};

}  // namespace strix::h3::dist
