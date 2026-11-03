// CPU tests of the Turbo v4 host pieces (src/h3/levers/lora/turbo_host.hpp, bypass_ref.hpp): module names, nearest
// curve row (ties -> first), adaLN delta vs an fp64 evaluation, strength scaling, bypass host mirror vs fp64.
// usage: strix-h3-levers-turbo-host-tests
#include "h3/levers/lora/bypass_ref.hpp"
#include "h3/levers/lora/turbo_host.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace strix::h3::levers::lora;
namespace g = strix::h3::gemm;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Rng {
    std::uint64_t s;
    std::uint64_t next() { std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL; return z ^ (z >> 31U); }
    float uni() { return static_cast<float>(next() >> 40U) * 0x1p-24F * 2.0F - 1.0F; }
};
std::vector<std::uint16_t> randbf(std::size_t n, float scale, std::uint64_t seed) {
    Rng r{seed};
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = g::f32_to_bf16(r.uni() * scale);
    return v;
}

void test_names() {
    CHECK(lora_module_name(Site::kQkv, 0) == "blocks.0.attn.qkv_proj", "qkv name");
    CHECK(lora_module_name(Site::kFc1, 49) == "blocks.49.mlp.fc1", "fc1 name");
    CHECK(lora_module_name(Site::kRefFc2, 1) == "token_refiner.blocks.1.mlp.fc2", "refiner name");
    CHECK(adaln_module_name(3, false) == "blocks.3.adaln_proj.linear" && adaln_module_name(0, true) == "final_layer.adaln_proj.linear", "adaln names");
    std::printf("names checked\n");
}

void test_nearest() {
    constexpr std::uint32_t rows = 9U, dim = 5U;
    std::vector<float> table(rows * dim);
    for (std::uint32_t j = 0; j < rows; ++j)
        for (std::uint32_t k = 0; k < dim; ++k) table[j * dim + k] = static_cast<float>(j) * (1.0F + 0.25F * static_cast<float>(k));
    std::vector<float> q(dim);
    const auto lerp_row = [&](std::uint32_t i0, float f) {
        for (std::uint32_t k = 0; k < dim; ++k) q[k] = table[i0 * dim + k] + f * (table[(i0 + 1U) * dim + k] - table[i0 * dim + k]);
    };
    lerp_row(3, 0.25F);
    CHECK(adaln_nearest_row(q.data(), table.data(), rows, dim) == 3U, "f 0.25 -> i0");
    lerp_row(3, 0.75F);
    CHECK(adaln_nearest_row(q.data(), table.data(), rows, dim) == 4U, "f 0.75 -> i0 + 1");
    lerp_row(3, 0.5F);
    CHECK(adaln_nearest_row(q.data(), table.data(), rows, dim) == 3U, "tie -> first");
    lerp_row(7, 1.0F);
    CHECK(adaln_nearest_row(q.data(), table.data(), rows, dim) == 8U, "last row");
    std::printf("nearest row checked\n");
}

void test_delta() {
    constexpr std::uint32_t out = 1536U;
    const auto a = randbf(kAdalnRank * kTembDim, 0.05F, 1), b = randbf(static_cast<std::size_t>(out) * kAdalnRank, 0.05F, 2),
               st = randbf(kTembDim, 1.5F, 3);
    const auto bs = scale_bf16(b, 0.70F);
    std::size_t scale_bad = 0;
    for (std::size_t i = 0; i < b.size(); ++i) scale_bad += bs[i] != g::f32_to_bf16(g::bf16_to_f32(b[i]) * 0.70F);
    std::vector<float> d(out);
    adaln_delta(st.data(), a.data(), bs.data(), out, d.data());
    double worst = 0.0;
    for (std::uint32_t o = 0; o < out; ++o) {
        double ref = 0.0, mag = 0.0;
        for (std::uint32_t r = 0; r < kAdalnRank; ++r) {
            double u = 0.0, um = 0.0;
            for (std::uint32_t k = 0; k < kTembDim; ++k) {
                const double p = static_cast<double>(g::bf16_to_f32(a[r * kTembDim + k])) * g::bf16_to_f32(st[k]);
                u += p;
                um += std::fabs(p);
            }
            const double bb = g::bf16_to_f32(bs[static_cast<std::size_t>(o) * kAdalnRank + r]);
            ref += bb * u;
            mag += std::fabs(bb) * um;
        }
        worst = std::fmax(worst, std::fabs(d[o] - ref) / (mag * 0x1p-24 * (kTembDim + kAdalnRank)));
    }
    std::printf("adaLN delta: worst |fp32 chain - fp64| / (gamma_n * sum|terms|) = %.3f (must be <= 1), %zu scale mismatches\n", worst, scale_bad);
    CHECK(worst <= 1.0 && scale_bad == 0, "adaLN delta");
    // the real pruned-checkpoint shapes: curve table adaln_t_table [1025][8] (t_emb rows are lerps of its rows), E-grid
    // [1025][2688]; the search runs in the 8-wide basis, the injected row comes from the E-grid
    constexpr std::uint32_t tdim = 8U;
    std::vector<float> x(2U * out, 1.0F), te(2U * tdim), table(static_cast<std::size_t>(kEgridRows) * tdim);
    std::vector<std::uint16_t> egrid(static_cast<std::size_t>(kEgridRows) * kTembDim);
    for (std::uint32_t j = 0; j < kEgridRows; ++j) {
        for (std::uint32_t k = 0; k < tdim; ++k) table[static_cast<std::size_t>(j) * tdim + k] = static_cast<float>(k + 1U) * static_cast<float>(j) / 1024.0F;
        for (std::uint32_t k = 0; k < kTembDim; ++k) egrid[static_cast<std::size_t>(j) * kTembDim + k] = st[(k + j) % kTembDim];
    }
    for (std::uint32_t k = 0; k < tdim; ++k) {
        te[k] = table[100U * tdim + k];
        te[tdim + k] = 0.4F * table[700U * tdim + k] + 0.6F * table[701U * tdim + k];
    }
    const auto idx = adaln_inject(x.data(), 2U, out, te.data(), tdim, table.data(), kEgridRows, egrid.data(), kEgridRows, a.data(), bs.data());
    std::vector<float> d1(out);
    adaln_delta(egrid.data() + 701U * kTembDim, a.data(), bs.data(), out, d1.data());
    std::size_t add_bad = 0;
    for (std::uint32_t o = 0; o < out; ++o) add_bad += x[out + o] != 1.0F + d1[o];
    int rej = 0;
    try { adaln_inject(x.data(), 2U, out, te.data(), tdim, table.data(), 1024U, egrid.data(), kEgridRows, a.data(), bs.data()); } catch (const std::invalid_argument&) { ++rej; }
    try { adaln_inject(x.data(), 2U, out, te.data(), tdim, table.data(), kEgridRows, egrid.data(), 1024U, a.data(), bs.data()); } catch (const std::invalid_argument&) { ++rej; }
    try { adaln_inject(x.data(), 2U, out, te.data(), 0U, table.data(), kEgridRows, egrid.data(), kEgridRows, a.data(), bs.data()); } catch (const std::invalid_argument&) { ++rej; }
    std::printf("adaLN inject (table [1025][8], E-grid [1025][2688]): rows %u / %u (want 100 / 701), %zu add mismatches, %d of 3 bad shapes rejected\n",
        idx[0], idx[1], add_bad, rej);
    CHECK(idx[0] == 100U && idx[1] == 701U && add_bad == 0 && rej == 3, "adaLN inject");
}

void test_bypass_host() {
    constexpr std::uint32_t m = 3U, k = 512U, n = 128U;
    const auto x = randbf(m * k, 2.0F, 7), a = randbf(kRank * k, 0.03F, 8), b = randbf(n * kRank, 0.03F, 9), y0 = randbf(m * n, 1.0F, 10);
    std::vector<std::uint16_t> y = y0;
    bypass_host(x.data(), m, k, a.data(), b.data(), n, 0.70F, y.data());
    std::size_t far = 0;
    for (std::uint32_t r = 0; r < m; ++r)
        for (std::uint32_t o = 0; o < n; ++o) {
            double l = 0.0, lm = 0.0;
            for (std::uint32_t j = 0; j < kRank; ++j) {
                double t = 0.0;
                for (std::uint32_t i = 0; i < k; ++i) t += static_cast<double>(g::bf16_to_f32(x[r * k + i])) * g::bf16_to_f32(a[j * k + i]);
                l += t * g::bf16_to_f32(b[o * kRank + j]);
                lm += std::fabs(t * g::bf16_to_f32(b[o * kRank + j]));
            }
            const double ref = g::bf16_to_f32(y0[r * n + o]) + 0.7 * l, got = g::bf16_to_f32(y[r * n + o]);
            far += std::fabs(got - ref) > 0x1p-6 * (std::fabs(ref) + 0.7 * lm + 1e-3);
        }
    std::printf("bypass host vs fp64: %zu of %u outside 2^-6 relative (bf16 t / l roundings)\n", far, m * n);
    CHECK(far == 0, "bypass host");
}
}  // namespace

int main() {
    test_names();
    test_nearest();
    test_delta();
    test_bypass_host();
    std::printf("turbo host tests %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
