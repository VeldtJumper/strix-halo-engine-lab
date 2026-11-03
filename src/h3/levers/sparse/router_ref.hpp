#pragma once
// Block-sparse attention threshold router (identity h3-attn-sparse-thr-v1, FAST-class, OFF by default): host reference
// (no HIP) of the frozen v1 arithmetic spec (this header). Every fp32 operation is a separate IEEE
// rounding (built with -ffp-contract=off / MSVC /fp:strict), sums from +0 in the stated ascending order. The device
// router kernels must match this bit for bit (kc / qbar / stats / thresholds / scores / LUT).
//
// Layout: Q and K are the post-K4 bf16 values, token-major [rows][56][128] with row pitch `ld` elements (7168 for the
// packed QKV buffer slices). All tile / block indices are GLOBAL (key tile j = rows [64 j, 64 j + 64) of the packed
// sequence); a rank passes its local buffers plus the global index of their first row.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/gemm/ref.hpp"

namespace strix::h3::levers::sparse {

inline constexpr const char* kSparseIdentity = "h3-attn-sparse-thr-v1";
constexpr std::uint32_t kHeads = 56U, kDim = 128U, kTile = 64U, kQTile = 128U;

struct Params {
    float tau = 0.75F;
    std::uint32_t dense_calls = 3U;
    std::uint32_t dense_layers = 2U;
    std::uint32_t band = 1U;
    // Extra dense layers (bit l = DiT layer l runs dense), on top of layers < dense_layers. Quality knob: the per-call
    // study found the worst sparse error in layers 39-42 (per-layer quality study): dense_layer_mask = layer_mask("39-42").
    std::uint64_t dense_layer_mask = 0;
};
// "a-b,c,..." -> bit mask of DiT layers (each < 64); throws on a malformed list.
inline std::uint64_t layer_mask(const std::string& list) {
    std::uint64_t m = 0;
    std::size_t i = 0;
    while (i < list.size()) {
        std::size_t j = list.find(',', i);
        if (j == std::string::npos) j = list.size();
        const std::string item = list.substr(i, j - i);
        const std::size_t dash = item.find('-');
        std::size_t used = 0;
        const unsigned long a = std::stoul(item.substr(0, dash), &used);
        if (used != (dash == std::string::npos ? item.size() : dash)) throw std::invalid_argument("h3 sparse: bad layer list " + list);
        unsigned long b = a;
        if (dash != std::string::npos) {
            b = std::stoul(item.substr(dash + 1), &used);
            if (used != item.size() - dash - 1) throw std::invalid_argument("h3 sparse: bad layer list " + list);
        }
        if (a > b || b >= 64U) throw std::invalid_argument("h3 sparse: bad layer range in " + list);
        for (unsigned long l = a; l <= b; ++l) m |= 1ULL << l;
        i = j + 1;
    }
    return m;
}

// c = fp32(log2(e) / sqrt(128)), computed in double and rounded once.
inline float score_scale() { return static_cast<float>(1.4426950408889634 * 0.08838834764831845); }

struct Geometry {
    std::uint32_t S = 0, P = 0;    // packed rows, prefix rows (before the target video)
    std::uint32_t KT = 0, QT = 0;  // 64-key tiles, 128-row query tiles
    std::uint32_t cut = 0;         // 128 * floor(S / 256): key block 0 = tiles [0, cut / 64)
    bool dense = true;             // degenerate shape -> the call runs dense
};
inline Geometry geometry(std::uint32_t S, std::uint32_t P) {
    Geometry g;
    g.S = S;
    g.P = P;
    g.KT = (S + kTile - 1U) / kTile;
    g.QT = (S + kQTile - 1U) / kQTile;
    g.cut = kQTile * (S / (2U * kQTile));
    g.dense = S < 2U * kQTile || P >= S || g.cut == 0U;
    return g;
}
inline std::uint32_t tile_rows(const Geometry& g, std::uint32_t j) { return std::min(kTile, g.S - j * kTile); }
// Is the call / layer sparse at all?
inline bool sparse_call(const Params& p, const Geometry& g, std::uint32_t model_call, std::uint32_t layer) {
    return !g.dense && model_call >= p.dense_calls && layer >= p.dense_layers && !(layer < 64U && ((p.dense_layer_mask >> layer) & 1U));
}

// kc[j - j0][h][d] (bf16) for key tiles j0 .. j1-1; k points at the row of global index 64 * j0.
inline std::vector<std::uint16_t> key_centroids(const std::uint16_t* k, std::uint64_t ld, const Geometry& g, std::uint32_t j0, std::uint32_t j1) {
    if (j0 > j1 || j1 > g.KT) throw std::out_of_range("h3 sparse: key tile range");
    std::vector<std::uint16_t> kc(static_cast<std::size_t>(j1 - j0) * kHeads * kDim);
    for (std::uint32_t j = j0; j < j1; ++j) {
        const std::uint32_t n = tile_rows(g, j);
        for (std::uint32_t h = 0; h < kHeads; ++h)
            for (std::uint32_t d = 0; d < kDim; ++d) {
                float s = 0.0F;
                for (std::uint32_t r = 0; r < n; ++r) s = s + gemm::bf16_to_f32(k[(static_cast<std::uint64_t>(j - j0) * kTile + r) * ld + h * kDim + d]);
                kc[(static_cast<std::size_t>(j - j0) * kHeads + h) * kDim + d] = gemm::f32_to_bf16(s / static_cast<float>(n));
            }
    }
    return kc;
}
// qbar[i - i0][h][d] (fp32) for 64-row query blocks i0 .. i1-1; q points at the row of global index 64 * i0.
inline std::vector<float> query_centroids(const std::uint16_t* q, std::uint64_t ld, const Geometry& g, std::uint32_t i0, std::uint32_t i1) {
    if (i0 > i1 || i1 > g.KT) throw std::out_of_range("h3 sparse: query block range");
    std::vector<float> qb(static_cast<std::size_t>(i1 - i0) * kHeads * kDim);
    for (std::uint32_t i = i0; i < i1; ++i) {
        const std::uint32_t n = tile_rows(g, i);
        for (std::uint32_t h = 0; h < kHeads; ++h)
            for (std::uint32_t d = 0; d < kDim; ++d) {
                float s = 0.0F;
                for (std::uint32_t r = 0; r < n; ++r) s = s + gemm::bf16_to_f32(q[(static_cast<std::uint64_t>(i - i0) * kTile + r) * ld + h * kDim + d]);
                qb[(static_cast<std::size_t>(i - i0) * kHeads + h) * kDim + d] = s / static_cast<float>(n);
            }
    }
    return qb;
}

// Per key block: S1 = sum kc, S2 = sum kc * kc over its tiles in ascending order, n tiles.
struct StatsPartial {
    std::vector<float> s1, s2;  // [56][128]
    std::uint32_t n = 0;
};
inline StatsPartial stats_partial(const std::vector<std::uint16_t>& kc, std::uint32_t tiles) {
    if (kc.size() != static_cast<std::size_t>(tiles) * kHeads * kDim) throw std::invalid_argument("h3 sparse: kc size");
    StatsPartial p;
    p.s1.assign(static_cast<std::size_t>(kHeads) * kDim, 0.0F);
    p.s2.assign(p.s1.size(), 0.0F);
    p.n = tiles;
    for (std::uint32_t t = 0; t < tiles; ++t)
        for (std::size_t e = 0; e < p.s1.size(); ++e) {
            const float v = gemm::bf16_to_f32(kc[static_cast<std::size_t>(t) * p.s1.size() + e]);
            const float sq = v * v;
            p.s1[e] = p.s1[e] + v;
            p.s2[e] = p.s2[e] + sq;
        }
    return p;
}
struct Stats {
    std::vector<float> mean, var;  // [56][128]
};
// Partials in KEY-BLOCK order (block 0 then block 1), merged before normalising by the global tile count.
inline Stats stats_finalize(const std::vector<StatsPartial>& parts) {
    if (parts.empty()) throw std::invalid_argument("h3 sparse: no stats partials");
    const std::size_t m = static_cast<std::size_t>(kHeads) * kDim;
    std::vector<float> s1(m, 0.0F), s2(m, 0.0F);
    std::uint32_t nt = 0;
    for (const StatsPartial& p : parts) {
        if (p.s1.size() != m || p.s2.size() != m) throw std::invalid_argument("h3 sparse: stats partial size");
        for (std::size_t e = 0; e < m; ++e) {
            s1[e] = s1[e] + p.s1[e];
            s2[e] = s2[e] + p.s2[e];
        }
        nt += p.n;
    }
    if (!nt) throw std::invalid_argument("h3 sparse: no key tiles");
    Stats st;
    st.mean.resize(m);
    st.var.resize(m);
    const float fn = static_cast<float>(nt);
    for (std::size_t e = 0; e < m; ++e) {
        const float mu = s1[e] / fn;
        const float e2 = s2[e] / fn;
        const float mu2 = mu * mu;
        const float v = e2 - mu2;
        st.mean[e] = mu;
        st.var[e] = v > 0.0F ? v : 0.0F;
    }
    return st;
}

// thr[i - i0][h] for the given query centroids.
inline std::vector<float> thresholds(const std::vector<float>& qbar, std::uint32_t blocks, const Stats& st, float tau) {
    if (qbar.size() != static_cast<std::size_t>(blocks) * kHeads * kDim) throw std::invalid_argument("h3 sparse: qbar size");
    const float c = score_scale(), c2 = c * c;
    std::vector<float> thr(static_cast<std::size_t>(blocks) * kHeads);
    for (std::uint32_t i = 0; i < blocks; ++i)
        for (std::uint32_t h = 0; h < kHeads; ++h) {
            const float* qv = qbar.data() + (static_cast<std::size_t>(i) * kHeads + h) * kDim;
            const float* mv = st.mean.data() + static_cast<std::size_t>(h) * kDim;
            const float* vv = st.var.data() + static_cast<std::size_t>(h) * kDim;
            float dot = 0.0F, vs = 0.0F;
            for (std::uint32_t d = 0; d < kDim; ++d) {
                const float p = qv[d] * mv[d];
                dot = dot + p;
                const float q2 = qv[d] * qv[d];
                const float pv = q2 * vv[d];
                vs = vs + pv;
            }
            const float mu = dot * c;
            const float v = vs * c2;
            const float sd = std::sqrt((v > 0.0F ? v : 0.0F) + 1e-6F);
            const float ts = tau * sd;
            thr[static_cast<std::size_t>(i) * kHeads + h] = mu + ts;
        }
    return thr;
}
inline float block_score(const float* qv, const std::uint16_t* kv) {
    float dot = 0.0F;
    for (std::uint32_t d = 0; d < kDim; ++d) {
        const float p = qv[d] * gemm::bf16_to_f32(kv[d]);
        dot = dot + p;
    }
    return dot * score_scale();
}

// Route of the query tiles t0 .. t1-1 against the key tiles j0 .. j1-1 (one key block, or both):
// lists[(t - t0) * 56 + h] = ascending selected GLOBAL key tiles in [j0, j1).
// qbar / thr cover query blocks 2 t0 .. (blocks given), kc covers tiles j0 .. j1-1.
struct Lut {
    std::uint32_t t0 = 0, t1 = 0, j0 = 0, j1 = 0;
    std::vector<std::uint32_t> offsets;  // [(t1 - t0) * 56 + 1]
    std::vector<std::int32_t> entries;
    std::uint64_t union_sum = 0, parts_sum = 0;  // telemetry: sum |A u B|, sum |A| + |B| over non-dense (tile, head)
    std::uint32_t dense_tiles = 0;
};
inline Lut route(const Geometry& g, const Params& p, std::uint32_t t0, std::uint32_t t1, const std::vector<float>& qbar, const std::vector<float>& thr,
    std::uint32_t j0, std::uint32_t j1, const std::vector<std::uint16_t>& kc) {
    if (g.dense) throw std::logic_error("h3 sparse: route on a dense geometry");
    if (t0 > t1 || t1 > g.QT || j0 > j1 || j1 > g.KT) throw std::out_of_range("h3 sparse: route ranges");
    const std::uint32_t i0 = 2U * t0, i1 = std::min(2U * t1, g.KT);
    if (qbar.size() != static_cast<std::size_t>(i1 - i0) * kHeads * kDim || thr.size() != static_cast<std::size_t>(i1 - i0) * kHeads ||
        kc.size() != static_cast<std::size_t>(j1 - j0) * kHeads * kDim)
        throw std::invalid_argument("h3 sparse: route input sizes");
    Lut L;
    L.t0 = t0;
    L.t1 = t1;
    L.j0 = j0;
    L.j1 = j1;
    L.offsets.push_back(0U);
    for (std::uint32_t t = t0; t < t1; ++t) {
        const bool dense_tile = kQTile * t < g.P;
        L.dense_tiles += dense_tile ? 1U : 0U;
        for (std::uint32_t h = 0; h < kHeads; ++h) {
            std::uint32_t na = 0, nb = 0;
            for (std::uint32_t j = j0; j < j1; ++j) {
                bool sel = dense_tile;
                if (!sel) {
                    bool hit[2] = {false, false};
                    for (std::uint32_t b = 0; b < 2U; ++b) {
                        const std::uint32_t i = 2U * t + b;
                        if (i >= g.KT) continue;
                        const std::uint32_t di = i > j ? i - j : j - i;
                        const bool sink = kTile * j < g.P;
                        if (di <= p.band || sink) {
                            hit[b] = true;
                            continue;
                        }
                        const float* qv = qbar.data() + (static_cast<std::size_t>(i - i0) * kHeads + h) * kDim;
                        const std::uint16_t* kv = kc.data() + (static_cast<std::size_t>(j - j0) * kHeads + h) * kDim;
                        hit[b] = block_score(qv, kv) > thr[static_cast<std::size_t>(i - i0) * kHeads + h];
                    }
                    na += hit[0] ? 1U : 0U;
                    nb += hit[1] ? 1U : 0U;
                    sel = hit[0] || hit[1];
                }
                if (sel) L.entries.push_back(static_cast<std::int32_t>(j));
            }
            if (!dense_tile) {
                L.union_sum += L.entries.size() - L.offsets.back();
                L.parts_sum += na + nb;
            }
            L.offsets.push_back(static_cast<std::uint32_t>(L.entries.size()));
        }
    }
    return L;
}

// LUT contract check (sorted, unique, inside [j0, j1)); throws on violation.
inline void validate_lut(const Lut& L) {
    const std::size_t lists = static_cast<std::size_t>(L.t1 - L.t0) * kHeads;
    if (L.offsets.size() != lists + 1U || L.offsets.front() != 0U || L.offsets.back() != L.entries.size()) throw std::invalid_argument("h3 sparse: LUT offsets");
    for (std::size_t l = 0; l < lists; ++l) {
        if (L.offsets[l] > L.offsets[l + 1U]) throw std::invalid_argument("h3 sparse: LUT offsets decrease");
        for (std::uint32_t e = L.offsets[l]; e < L.offsets[l + 1U]; ++e) {
            const std::int32_t j = L.entries[e];
            if (j < static_cast<std::int32_t>(L.j0) || j >= static_cast<std::int32_t>(L.j1)) throw std::invalid_argument("h3 sparse: LUT entry out of range");
            if (e > L.offsets[l] && L.entries[e - 1U] >= j) throw std::invalid_argument("h3 sparse: LUT not strictly ascending");
        }
    }
}

}  // namespace strix::h3::levers::sparse
