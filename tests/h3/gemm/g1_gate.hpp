#pragma once
// G1 gate helpers shared by the CPU and GPU tests of the H3 gemm package.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "h3/gemm/ref.hpp"

namespace strix::h3::gemm::g1 {

// G1 rotation gate. A value of the reference rotation is RN_bf16(exact + e) with e the error of its fp32 summation in an
// unspecified order: |e| <= gamma_255 * S / 16, S = sum_i |x_i| of the group (any order, u = 2^-24). Ours is RN_bf16(exact).
// So |ours - ref| <= ulp_bf16(max(|ours|, |ref|)) + gamma_255 * S / 16 for EVERY valid reference; the gate asserts that
// bound (worst ratio <= 1) and reports the max ulp distance and how many values are more than 1 ulp apart (cancellation:
// the summation error is relative to S, not to the small result, so "<= 1 ulp" alone does not hold for the reference).
inline int ordered(std::uint16_t b) { return (b & 0x8000U) ? -static_cast<int>(b & 0x7fffU) : static_cast<int>(b); }
struct RotGate { unsigned max_ulp = 0; std::size_t over_1ulp = 0; double worst = 0.0; bool ok = false; };
inline RotGate rotation_gate(const std::vector<std::uint16_t>& x, const std::vector<std::uint16_t>& ours, const std::vector<std::uint16_t>& ref) {
    RotGate g;
    if (x.size() != ours.size() || x.size() != ref.size() || x.empty() || x.size() % 256U) return g;
    const double gamma = 255.0 * 0x1p-24 / (1.0 - 255.0 * 0x1p-24);
    for (std::size_t base = 0; base < x.size(); base += 256U) {
        double sabs = 0.0;
        for (unsigned i = 0; i < 256U; ++i) sabs += std::fabs(static_cast<double>(bf16_to_f32(x[base + i])));
        for (unsigned j = 0; j < 256U; ++j) {
            const std::uint16_t a = ours[base + j], b = ref[base + j];
            const int d = ordered(a) - ordered(b);
            const unsigned du = static_cast<unsigned>(d < 0 ? -d : d);
            g.max_ulp = std::max(g.max_ulp, du);
            g.over_1ulp += du > 1U;
            if (a == b) continue;
            const double va = bf16_to_f32(a), vb = bf16_to_f32(b), big = std::max(std::fabs(va), std::fabs(vb));
            int e;
            static_cast<void>(std::frexp(big, &e));
            const double ulp = std::ldexp(1.0, std::max(e - 8, -133));
            g.worst = std::max(g.worst, std::fabs(va - vb) / (ulp + gamma * sabs / 16.0));
        }
    }
    g.ok = g.worst <= 1.0;
    return g;
}
inline bool flips_ok(std::size_t flips, std::size_t codes) { return flips != ~std::size_t{0} && flips * 250000U <= codes + 500000U; }  // <= 2 + 4e-6 * codes
// The code-flip budget (<= 2 + 4e-6 of the codes) is EMPIRICAL for the qualification corpus (python dumps + quantiser fixture set): rounding boundaries and
// row-scale changes can amplify a 1-ulp rotation difference, so it is not a universal or batch-invariant guarantee.

}  // namespace strix::h3::gemm::g1
