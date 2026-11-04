// Second unit of the two-translation-unit link check (see link_a.hip).
#include "h3/gemm/api.hpp"

int h3_gemm_link_b() {
    using namespace strix::h3::gemm;
    void (*q)(const std::uint16_t*, std::uint64_t, std::uint32_t, std::uint32_t, std::int8_t*, float*, unsigned*, hipStream_t, int) = &launch_quantize;
    void (*g)(const dev::GemmArgs&, hipStream_t) = &launch_gemm<128U, dev::Epi::plain>;
    void (*s)(const dev::GemmArgs&, hipStream_t) = &launch_gemm<256U, dev::Epi::swiglu>;
    auto l = &linear;
    auto f = &linear_fc1_swiglu;
    return (q && g && s && l && f && silu_table().size() == 65536U) ? 1 : 0;
}
