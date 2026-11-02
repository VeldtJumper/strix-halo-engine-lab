// Two-translation-unit link check of the public header (with link_b.hip): every definition in api.hpp / kernels.hpp /
// ref.hpp must be header-safe (inline, templates), so two HIP units that instantiate the launchers link. No GPU work.
#include "h3/gemm/api.hpp"

int h3_gemm_link_b();

int h3_gemm_link_a() {
    using namespace strix::h3::gemm;
    void (*q)(const std::uint16_t*, std::uint64_t, std::uint32_t, std::uint32_t, std::int8_t*, float*, unsigned*, hipStream_t, int) = &launch_quantize;
    void (*g)(const dev::GemmArgs&, hipStream_t) = &launch_gemm<128U, dev::Epi::plain>;
    void (*s)(const dev::GemmArgs&, hipStream_t) = &launch_gemm<256U, dev::Epi::swiglu>;
    auto l = &linear;
    auto f = &linear_fc1_swiglu;
    return (q && g && s && l && f && silu_table()[0] == 0U) ? 1 : 0;
}

int main() { return h3_gemm_link_a() + h3_gemm_link_b() == 2 ? 0 : 1; }
