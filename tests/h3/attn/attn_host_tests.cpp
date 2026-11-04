// CPU tests of the H3 split-KV attention arithmetic (no GPU): the fp32 per-block partial in the kernel's order,
// the canonical merge, and the fp64 merge identity.
#include "h3/attn/attn_ref.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace strix::h3::attn;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
std::vector<Partial> split(const HostQkv& x, unsigned h, unsigned r, const std::vector<unsigned>& bounds) {
    std::vector<Partial> p;
    for (std::size_t b = 0; b + 1 < bounds.size(); ++b) p.push_back(partial_row_f32(x, h, r, bounds[b], bounds[b + 1]));
    return p;
}
}  // namespace

int main() {
    // bf16 rounding.
    check(bf16_from_float(1.0F) == 0x3f80U && bf16_from_float(-2.0F) == 0xc000U, "bf16 exact values");
    {
        std::uint32_t u = 0x3f808000U;  // halfway, even below -> stays
        float f; std::memcpy(&f, &u, 4);
        const bool a = bf16_from_float(f) == 0x3f80U;
        u = 0x3f818000U;  // halfway, odd below -> up
        std::memcpy(&f, &u, 4);
        check(a && bf16_from_float(f) == 0x3f82U, "bf16 ties to even");
    }

    // fp64 merge identity: merging exact block partials equals the single pass.
    {
        const auto x = make_qkv(2, 8, 1000, 11, 1.0F);
        double worst = 0.0;
        for (unsigned h = 0; h < 2; ++h)
            for (unsigned r = 0; r < 8; ++r) {
                double ref[kHeadDim], got[kHeadDim];
                reference_row_f64(x, h, r, ref);
                const PartialF64 p[3] = {partial_row_f64(x, h, r, 0, 320), partial_row_f64(x, h, r, 320, 704), partial_row_f64(x, h, r, 704, 1000)};
                merge_f64(p, 3, got);
                for (unsigned d = 0; d < kHeadDim; ++d) worst = std::fmax(worst, std::fabs(got[d] - ref[d]));
            }
        std::printf("  fp64 3-block merge vs single pass: max abs %.3e\n", worst);
        check(worst < 1e-12, "fp64 merge identity (3 blocks)");
    }

    // fp32 kernel-order partials + canonical merge vs fp64, 1/2/3 blocks, partial steps, two score spreads.
    for (const float qs : {1.0F, 4.0F}) {
        const unsigned keys = 1000;
        const auto x = make_qkv(3, 12, keys, 7 + static_cast<unsigned>(qs), qs);
        const std::vector<std::vector<unsigned>> cuts = {{0, keys}, {0, 512, keys}, {0, 320, 704, keys}, {0, 64, 128, keys}};
        double worst[4] = {}, vs_single[4] = {};
        for (unsigned h = 0; h < 3; ++h)
            for (unsigned r = 0; r < 12; ++r) {
                double ref[kHeadDim];
                reference_row_f64(x, h, r, ref);
                float single[kHeadDim];
                {
                    const auto p = split(x, h, r, cuts[0]);
                    merge_f32(p.data(), 1, single);
                }
                for (std::size_t c = 0; c < cuts.size(); ++c) {
                    const auto p = split(x, h, r, cuts[c]);
                    float out[kHeadDim];
                    merge_f32(p.data(), static_cast<unsigned>(p.size()), out);
                    worst[c] = std::fmax(worst[c], row_error(out, ref));
                    double ss[kHeadDim];
                    for (unsigned d = 0; d < kHeadDim; ++d) ss[d] = single[d];
                    vs_single[c] = std::fmax(vs_single[c], row_error(out, ss));
                }
            }
        for (std::size_t c = 0; c < cuts.size(); ++c) {
            char name[256];
            std::snprintf(name, sizeof name, "fp32 %zu-block merge vs fp64 (q scale %.0f): G1 err %.3e (<= 2^-8), vs 1-block fp32 %.3e",
                          cuts[c].size() - 1, static_cast<double>(qs), worst[c], vs_single[c]);
            check(worst[c] <= 1.0 / 256.0 && vs_single[c] <= 1e-2, name);
        }
    }

    // Canonical order: the merged bits do not depend on the order in which the blocks' partials were produced,
    // and one block through the merge equals the direct normalisation o / l.
    {
        const auto x = make_qkv(1, 4, 777, 5, 2.0F);
        bool same = true, one = true;
        for (unsigned r = 0; r < 4; ++r) {
            Partial fwd[2] = {partial_row_f32(x, 0, r, 0, 384), partial_row_f32(x, 0, r, 384, 777)};
            Partial late[2];
            late[1] = partial_row_f32(x, 0, r, 384, 777);  // the remote block first
            late[0] = partial_row_f32(x, 0, r, 0, 384);
            float a[kHeadDim], b[kHeadDim];
            merge_f32(fwd, 2, a);
            merge_f32(late, 2, b);
            same &= std::memcmp(a, b, sizeof a) == 0;
            const Partial whole = partial_row_f32(x, 0, r, 0, 777);
            float m1[kHeadDim];
            merge_f32(&whole, 1, m1);
            for (unsigned d = 0; d < kHeadDim; ++d) {
                const float direct = whole.o[d] / whole.l;
                one &= std::memcmp(&direct, &m1[d], 4) == 0;
            }
        }
        check(same, "canonical merge is independent of partial arrival order (bitwise)");
        check(one, "1-block merge == o / l (bitwise)");
    }

    // An empty block merges as nothing.
    {
        const auto x = make_qkv(1, 2, 300, 9, 1.0F);
        bool ok = true;
        for (unsigned r = 0; r < 2; ++r) {
            const Partial with_empty[3] = {partial_row_f32(x, 0, r, 0, 128), partial_row_f32(x, 0, r, 128, 128), partial_row_f32(x, 0, r, 128, 300)};
            const Partial plain[2] = {with_empty[0], with_empty[2]};
            float a[kHeadDim], b[kHeadDim];
            merge_f32(with_empty, 3, a);
            merge_f32(plain, 2, b);
            ok &= std::memcmp(a, b, sizeof a) == 0;
        }
        check(ok, "empty block is neutral in the merge (bitwise)");
    }

    {
        const auto x = make_qkv(1, 1, 64, 3, 1.0F);
        const Partial e = partial_row_f32(x, 0, 0, 10, 10);
        float out[kHeadDim];
        merge_f32(&e, 1, out);
        bool z = e.mc == -std::numeric_limits<float>::infinity() && e.l == 0.0F;
        for (float v : out) z &= v == 0.0F;
        check(z, "all-empty merge -> zeros");
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
