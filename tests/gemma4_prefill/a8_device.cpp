#include "gemma4/prefill/a8_linear.hpp"
#include "gemma4/prefill/device.hpp"
#include "gemma4/kernels/a8_host.hpp"
#include "../gemma4_gpu/fixture.hpp"
#include <chrono>
#include <cstring>
#include <random>

using namespace strix::gemma4;
namespace pf = strix::gemma4::prefill;
namespace gpu = strix::gemma4::gpu;

namespace {
// The fixture's packs plus negative and subnormal FP16 scales on some blocks.
std::vector<std::uint8_t> weights_for(WeightFormat f, unsigned outputs, unsigned width, unsigned seed) {
    auto b = fixture::packed(f, outputs, width, seed);
    const unsigned size = f == WeightFormat::q8_0 ? 34U : (f == WeightFormat::q4_k ? 144U : 210U);
    const unsigned field = f == WeightFormat::q6_k ? 208U : 0U;
    std::mt19937 rng(seed * 3 + 1);
    for (std::size_t i = 0; i < b.size(); i += size) {
        const unsigned kind = rng() % 8;
        if (kind == 0) b[i + field + 1] |= 0x80U;
        if (kind == 1) fixture::put16(b.data() + i + field, std::uint16_t(1U + rng() % 1023U));
        if (kind == 2 && f == WeightFormat::q4_k) fixture::put16(b.data() + i + 2, std::uint16_t(0x8000U | (1U + rng() % 1023U)));
    }
    return b;
}
std::vector<float> inputs_for(unsigned rows, unsigned width, unsigned seed) {
    std::vector<float> x(std::size_t(rows) * width);
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float u = float(int(rng() % 2000001) - 1000000) / 1000000.0F;
        const unsigned kind = rng() % 16;
        x[i] = kind == 0 ? -0.0F : kind == 1 ? u * 1700.0F : kind == 2 ? u * 1e-5F : kind == 3 ? 1.0F + u * 1e-3F : u * 8.0F;
    }
    // One all-zero group and one underflowing group (delta == 0 -> delta 1, codes 0).
    if (x.size() >= 64) {
        std::fill(x.begin(), x.begin() + 32, 0.0F);
        std::fill(x.begin() + 32, x.begin() + 64, 1e-44F);
    }
    return x;
}
void a8_case(WeightFormat format, unsigned width, unsigned rows, unsigned outputs) {
    const auto packed = weights_for(format, outputs, width, width * 5 + outputs);
    const auto x = inputs_for(rows, width, width * 7 + rows);
    gpu::Buffer weights(packed.size()), input(x.size() * sizeof(float)), output(std::size_t(rows) * outputs * sizeof(float));
    gpu::Buffer scratch(pf::a8_scratch_bytes(rows, width)), failure(sizeof(unsigned));
    weights.upload(packed.data(), packed.size());
    input.upload(x.data(), input.size()); failure.zero();
    const LinearView w{weights.as<std::uint8_t>(), outputs, width, packed_row_bytes(format, width), format};
    pf::A8Prepared prepared(w);
    pf::a8_linear(prepared, input.as<float>(), rows, output.as<float>(), scratch.data(), failure.as<unsigned>());
    gpu::synchronize();
    unsigned flag = 0; failure.download(&flag, sizeof(flag)); CHECK(flag == 0);
    std::vector<float> actual(std::size_t(rows) * outputs); output.download(actual.data(), output.size());
    auto host = w; host.data = packed.data();
    std::vector<std::int8_t> codes; std::vector<a8::Meta> meta;
    a8::quantize_rows(x.data(), rows, width, codes, meta);
    std::vector<float> expected(actual.size());
    const auto groups = width / a8::group_width;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned o = 0; o < outputs; ++o)
            expected[std::size_t(r) * outputs + o] = a8::dot(host, o, codes.data() + std::size_t(r) * width, meta.data() + std::size_t(r) * groups);
    bytes(actual, expected);
}
void a8_tests() {
    struct Shape { unsigned width, rows, outputs; };
    for (const auto format : {WeightFormat::q4_k, WeightFormat::q6_k, WeightFormat::q8_0})
        for (const auto s : {Shape{256, 1, 19}, Shape{512, 7, 64}, Shape{768, 33, 70}, Shape{2560, 16, 256}, Shape{2560, 300, 96},
                             Shape{2048, 45, 130}, Shape{10240, 37, 40}, Shape{2560, pf::max_chunk_rows, 16}}) {
            for (unsigned form = 0; form < 10; ++form) { pf::set_a8_kernel(form); a8_case(format, s.width, s.rows, s.outputs); }
            pf::set_a8_kernel(5);
            std::printf("a8 %s width %u rows %u outputs %u bytes PASS (10 kernel forms)\n", format == WeightFormat::q4_k ? "q4_k" : format == WeightFormat::q6_k ? "q6_k" : "q8_0",
                        s.width, s.rows, s.outputs);
        }
    std::puts("prefill g4-a8-v1 linears == a8 host oracle PASS");
}
// Interleaved timing of the a8 linear and the FP16 WMMA (v1) linear on E4B projection shapes.
void bench(unsigned rows) {
    struct Shape { const char* name; unsigned width, outputs; };
    for (const auto s : {Shape{"q", 2560, 2048}, Shape{"kv", 2560, 512}, Shape{"o", 2048, 2560}, Shape{"gate/up", 2560, 10240},
                         Shape{"down", 10240, 2560}}) {
        const auto packed = fixture::packed(WeightFormat::q4_k, s.outputs, s.width, 7);
        const auto x = inputs_for(rows, s.width, 11);
        gpu::Buffer weights(packed.size()), input(x.size() * sizeof(float)), output(std::size_t(rows) * s.outputs * sizeof(float));
        gpu::Buffer scratch(std::max(pf::a8_scratch_bytes(rows, s.width), 2 * std::size_t(rows) * s.width * 2)), failure(sizeof(unsigned)),
            maximum(sizeof(unsigned));
        weights.upload(packed.data(), packed.size());
        input.upload(x.data(), input.size()); failure.zero(); maximum.zero();
        const LinearView w{weights.as<std::uint8_t>(), s.outputs, s.width, packed_row_bytes(WeightFormat::q4_k, s.width), WeightFormat::q4_k};
        pf::A8Prepared a8(w);
        pf::PreparedLinear fp16(w);
        constexpr unsigned forms[] = {2, 5, 6, 7, 8, 9};
        constexpr int arms = 7;
        double ms[arms] = {};
        constexpr int reps = 8;
        for (int rep = 0; rep < arms * reps + arms; ++rep) {
            const int arm = rep % arms;
            gpu::synchronize();
            const auto t0 = std::chrono::steady_clock::now();
            if (arm < arms - 1) { pf::set_a8_kernel(forms[arm]); pf::a8_linear(a8, input.as<float>(), rows, output.as<float>(), scratch.data(), failure.as<unsigned>()); }
            else pf::linear(w, input.as<float>(), rows, output.as<float>(), scratch.data(), failure.as<unsigned>(), maximum.as<unsigned>(), nullptr, true, &fp16);
            gpu::synchronize();
            if (rep >= arms) ms[arm] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        const double flop = 2.0 * rows * s.width * s.outputs;
        std::printf("bench rows %u %-8s TOPS: 32x64 %.1f | w64x64 %.1f  w32x128 %.1f  w64x128 %.1f  w64x64s2 %.1f  w128x64 %.1f | fp16-v1 %.1f TF (%.3f ms)\n",
                    rows, s.name, flop / (ms[0] / reps) / 1e9, flop / (ms[1] / reps) / 1e9, flop / (ms[2] / reps) / 1e9, flop / (ms[3] / reps) / 1e9,
                    flop / (ms[4] / reps) / 1e9, flop / (ms[5] / reps) / 1e9, flop / (ms[6] / reps) / 1e9, ms[6] / reps);
        pf::set_a8_kernel(5);
    }
}
} // namespace
int main(int argc, char** argv) try {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        for (const unsigned rows : {256U, pf::max_chunk_rows}) bench(rows);
        return 0;
    }
    a8_tests();
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
