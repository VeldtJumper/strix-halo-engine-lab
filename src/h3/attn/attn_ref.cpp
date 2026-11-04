#include "h3/attn/attn_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace strix::h3::attn {

std::uint16_t bf16_from_float(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((u >> 16) | 0x40U);  // quiet NaN
    u += 0x7fffU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

float bf16_to_float(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

namespace {
// splitmix64 + Box-Muller: deterministic on every host.
struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    double normal() {
        const double u = uniform(), w = uniform();
        return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * w);
    }
};
}  // namespace

HostQkv make_qkv(unsigned heads, unsigned q_rows, unsigned k_rows, std::uint64_t seed, float q_scale) {
    HostQkv x;
    x.heads = heads; x.q_rows = q_rows; x.k_rows = k_rows;
    Rng rng{seed};
    x.q.resize(std::size_t(heads) * q_rows * kHeadDim);
    x.k.resize(std::size_t(heads) * k_rows * kHeadDim);
    x.v.resize(std::size_t(heads) * k_rows * kHeadDim);
    for (auto& e : x.q) e = bf16_from_float(static_cast<float>(rng.normal() * q_scale));
    for (auto& e : x.k) e = bf16_from_float(static_cast<float>(rng.normal()));
    for (auto& e : x.v) e = bf16_from_float(static_cast<float>(rng.normal()));
    return x;
}

void reference_row_f64(const HostQkv& x, unsigned head, unsigned row, double* out) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    std::vector<double> s(x.k_rows);
    double q[kHeadDim];
    for (unsigned d = 0; d < kHeadDim; ++d) q[d] = bf16_to_float(x.qv(head, row, d));
    double m = -std::numeric_limits<double>::infinity();
    for (unsigned j = 0; j < x.k_rows; ++j) {
        double acc = 0.0;
        for (unsigned d = 0; d < kHeadDim; ++d) acc += q[d] * bf16_to_float(x.kv(head, j, d));
        s[j] = acc * scale;
        m = std::max(m, s[j]);
    }
    double l = 0.0;
    for (unsigned d = 0; d < kHeadDim; ++d) out[d] = 0.0;
    for (unsigned j = 0; j < x.k_rows; ++j) {
        const double p = std::exp(s[j] - m);
        l += p;
        for (unsigned d = 0; d < kHeadDim; ++d) out[d] += p * bf16_to_float(x.vv(head, j, d));
    }
    for (unsigned d = 0; d < kHeadDim; ++d) out[d] /= l;
}

Partial partial_row_f32(const HostQkv& x, unsigned head, unsigned row, unsigned k0, unsigned k1) {
    const float inf = std::numeric_limits<float>::infinity();
    Partial p;
    float m = -inf, mc = -inf, l_half[2] = {0.0F, 0.0F};
    float q[kHeadDim];
    for (unsigned d = 0; d < kHeadDim; ++d) q[d] = bf16_to_float(x.qv(head, row, d));
    for (unsigned key0 = k0; key0 < k1; key0 += kKeyStep) {
        float s[kKeyStep];
        float step_max = -inf;
        for (unsigned j = 0; j < kKeyStep; ++j) {
            const unsigned key = key0 + j;
            if (key >= k1) { s[j] = -inf; continue; }
            float acc = 0.0F;
            for (unsigned c = 0; c < kHeadDim; c += 16) {  // one K16 product per WMMA, chunks ascending
                float part = 0.0F;
                for (unsigned d = c; d < c + 16; ++d) part = part + q[d] * bf16_to_float(x.kv(head, key, d));
                acc = acc + part;
            }
            s[j] = acc;
            step_max = std::max(step_max, acc);
        }
        const float m_new = std::max(m, step_max), mc_new = m_new * kScaleLog2;
        const bool none = mc_new == -inf;
        const float alpha = none ? 1.0F : std::exp2(mc - mc_new), mc_sub = none ? 0.0F : mc_new;
        float pf[kKeyStep];
        std::uint16_t pb[kKeyStep];
        for (unsigned j = 0; j < kKeyStep; ++j) {
            pf[j] = std::exp2(std::fma(s[j], kScaleLog2, -mc_sub));
            pb[j] = bf16_from_float(pf[j]);
        }
        // The kernel's lanes: within each 16-key tile, keys 0..7 sum in the low half-wave, 8..15 in the high one;
        // each half sums its 32 values tile by tile, key by key.
        for (unsigned h = 0; h < 2; ++h) {
            float sum = 0.0F;
            for (unsigned t = 0; t < kKeyStep / 16; ++t)
                for (unsigned e = 0; e < 8; ++e) sum = sum + pf[16 * t + 8 * h + e];
            l_half[h] = l_half[h] * alpha + sum;
        }
        for (unsigned d = 0; d < kHeadDim; ++d) {
            float o = p.o[d] * alpha;
            for (unsigned t = 0; t < kKeyStep / 16; ++t) {
                float part = 0.0F;
                for (unsigned j = 16 * t; j < 16 * t + 16; ++j) {
                    const unsigned key = key0 + j;
                    const float v = key < k1 ? bf16_to_float(x.vv(head, key, d)) : 0.0F;
                    part = part + bf16_to_float(pb[j]) * v;
                }
                o = o + part;
            }
            p.o[d] = o;
        }
        m = m_new; mc = mc_new;
    }
    p.mc = mc;
    p.l = l_half[0] + l_half[1];
    return p;
}

void merge_f32(const Partial* parts, unsigned nb, float* out) {
    float M = parts[0].mc;
    for (unsigned b = 1; b < nb; ++b) M = std::max(M, parts[b].mc);
    float e[16] = {};
    if (nb > 16) nb = 16;
    for (unsigned b = 0; b < nb; ++b) e[b] = parts[b].mc == -std::numeric_limits<float>::infinity() ? 0.0F : std::exp2(parts[b].mc - M);
    float L = parts[0].l * e[0];
    for (unsigned b = 1; b < nb; ++b) L = L + parts[b].l * e[b];
    for (unsigned d = 0; d < kHeadDim; ++d) {
        float O = parts[0].o[d] * e[0];
        for (unsigned b = 1; b < nb; ++b) O = O + parts[b].o[d] * e[b];
        out[d] = L == 0.0F ? 0.0F : O / L;
    }
}

PartialF64 partial_row_f64(const HostQkv& x, unsigned head, unsigned row, unsigned k0, unsigned k1) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    PartialF64 p;
    p.m = -std::numeric_limits<double>::infinity();
    std::vector<double> s(k1 - k0);
    for (unsigned j = k0; j < k1; ++j) {
        double acc = 0.0;
        for (unsigned d = 0; d < kHeadDim; ++d)
            acc += static_cast<double>(bf16_to_float(x.qv(head, row, d))) * bf16_to_float(x.kv(head, j, d));
        s[j - k0] = acc * scale;
        p.m = std::max(p.m, s[j - k0]);
    }
    for (unsigned j = k0; j < k1; ++j) {
        const double w = std::exp(s[j - k0] - p.m);
        p.l += w;
        for (unsigned d = 0; d < kHeadDim; ++d) p.o[d] += w * bf16_to_float(x.vv(head, j, d));
    }
    return p;
}

void merge_f64(const PartialF64* parts, unsigned nb, double* out) {
    double M = parts[0].m;
    for (unsigned b = 1; b < nb; ++b) M = std::max(M, parts[b].m);
    double L = 0.0;
    for (unsigned b = 0; b < nb; ++b) L += parts[b].l * std::exp(parts[b].m - M);
    for (unsigned d = 0; d < kHeadDim; ++d) {
        double O = 0.0;
        for (unsigned b = 0; b < nb; ++b) O += parts[b].o[d] * std::exp(parts[b].m - M);
        out[d] = O / L;
    }
}

double row_error(const float* got, const double* ref) {
    double amax = 0.0, emax = 0.0;
    for (unsigned d = 0; d < kHeadDim; ++d) {
        amax = std::max(amax, std::fabs(ref[d]));
        const double e = std::fabs(static_cast<double>(got[d]) - ref[d]);
        emax = std::isnan(e) ? std::numeric_limits<double>::infinity() : std::max(emax, e);
    }
    return amax > 0.0 ? emax / amax : emax;
}

}  // namespace strix::h3::attn
