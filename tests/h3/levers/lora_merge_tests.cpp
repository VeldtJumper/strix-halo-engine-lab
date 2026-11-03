// CPU check of the Turbo LoRA fc2 merge (identity h3-turbo-v4-bypass) against the merge tool's fixture
// (tools/h3/levers/lora/merge_fc2.py --fixture DIR): an independent re-derivation of the reference chain.
//   1. the torch CPU generator port (MT19937 init_genrand(seed), rand float32 = (u32 & 0xffffff) * 2^-24, serial order)
//      vs the first 4096 values torch produced for the block's seed: bitwise;
//   2. the merge in exact (fp64) arithmetic with the reference's fp32 rounding points (fp32 q*s, B@A, x0.7f, rotations
//      W @ H^T, absmax, scale = fp32(absmax) / 127, fp32 x / scale + rand, floor, clamp) vs the fixture's merged codes and
//      scales. The reference's two fp32 GEMMs (inverse / forward rotation, B @ A) have the library's summation order, so
//      this is a near-bitwise cross-check: it reports and bounds the code flips (values within ~1e-7 of a floor boundary).
// usage: strix-h3-levers-lora-tests <fixture dir>   (env STRIX_H3_TURBO_FX)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "h3/levers/lora/merge_ref.hpp"

using namespace strix::h3::levers::lora;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

template <class T> std::vector<T> load(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::vector<T> v(static_cast<std::size_t>(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}
std::uint64_t json_u64(const std::string& s, const char* key) {
    const auto p = s.find(std::string("\"") + key + "\"");
    if (p == std::string::npos) return 0;
    return std::strtoull(s.c_str() + s.find(':', p) + 1, nullptr, 10);
}
double json_f(const std::string& s, const char* key) {
    const auto p = s.find(std::string("\"") + key + "\"");
    if (p == std::string::npos) return 0;
    return std::strtod(s.c_str() + s.find(':', p) + 1, nullptr);
}

void test_block(const std::filesystem::path& d) {
    std::ifstream mf(d / "meta.json");
    const std::string meta((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    const auto seed = static_cast<std::uint32_t>(json_u64(meta, "seed"));
    const auto n = static_cast<std::uint32_t>(json_u64(meta, "n")), k = static_cast<std::uint32_t>(json_u64(meta, "k"));
    const auto rank = static_cast<std::uint32_t>(json_u64(meta, "rank"));
    const auto strength = static_cast<float>(json_f(meta, "strength"));
    // 1. generator
    const auto head = load<float>(d / "rand_head.bin");
    TorchCpuGenerator g(seed);
    std::size_t gbad = 0;
    for (float v : head) gbad += g.rand_f32() != v;
    CHECK(!head.empty() && gbad == 0, "%s: generator %zu of %zu differ", d.filename().string().c_str(), gbad, head.size());
    // 2. merge
    const auto q8 = load<std::int8_t>(d / "base_q.bin");
    const auto s8 = load<float>(d / "base_scale.bin");
    const auto a = load<std::uint16_t>(d / "lora_A.bin");
    const auto b = load<std::uint16_t>(d / "lora_B.bin");
    const auto ref_q = load<std::int8_t>(d / "merged_q.bin");
    const auto ref_s = load<float>(d / "merged_scale.bin");
    const bool sizes_ok = q8.size() == static_cast<std::size_t>(n) * k && a.size() == static_cast<std::size_t>(rank) * k &&
        b.size() == static_cast<std::size_t>(n) * rank && ref_q.size() == q8.size() && ref_s.size() == n;
    CHECK(sizes_ok, "%s: fixture sizes", d.filename().string().c_str());
    if (!sizes_ok) return;
    std::vector<std::int8_t> q(q8.size());
    std::vector<float> s(n);
    merge_fc2_reference(q8.data(), s8.data(), a.data(), b.data(), n, k, rank, strength, seed, q.data(), s.data());
    std::size_t flips = 0, big = 0, sbad = 0, rows_flip = 0;
    for (std::uint32_t r = 0; r < n; ++r) {
        sbad += std::memcmp(&s[r], &ref_s[r], 4) != 0;
        bool any = false;
        for (std::uint32_t c = 0; c < k; ++c) {
            const int dq = q[static_cast<std::size_t>(r) * k + c] - ref_q[static_cast<std::size_t>(r) * k + c];
            flips += dq != 0;
            big += dq > 1 || dq < -1;
            any |= dq != 0;
        }
        rows_flip += any;
    }
    std::printf("%s (n %u k %u rank %u strength %.2f seed %u): generator ok; codes %zu of %zu differ (%zu by > 1, %zu rows), scales %zu of %u differ\n",
        d.filename().string().c_str(), n, k, rank, static_cast<double>(strength), seed, flips, q.size(), big, rows_flip, sbad, n);
    // budget 5e-5 of the codes, none by more than 1: the reference's fp32 GEMMs (library order) vs exact sums move values
    // by ~1 fp32 ulp, which flips a floor only within that distance of an integer
    CHECK(big == 0 && flips * 20000U <= q.size(), "%s: merge cross-check outside 5e-5 code flips (%zu)", d.filename().string().c_str(), flips);
}
}  // namespace

int main(int argc, char** argv) {
    {   // generator self-check: torch.Generator().manual_seed(42); torch.rand(3) = 0.8822692632675171, 0.9150039553642273, 0.38286375999450684
        TorchCpuGenerator g(42);
        const float e[3] = {0.8822692632675171F, 0.9150039553642273F, 0.38286375999450684F};
        for (float v : e) { const float x = g.rand_f32(); CHECK(x == v, "generator seed 42: %.9g vs %.9g", static_cast<double>(x), static_cast<double>(v)); }
        CHECK(string_to_seed("diffusion_model.blocks.0.mlp.fc2.weight") == 3068176245U, "CRC32 key seed");
    }
    const char* env = std::getenv("STRIX_H3_TURBO_FX");
    const char* dir = argc > 1 ? argv[1] : env;
    if (dir && std::filesystem::is_directory(dir)) {
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (std::filesystem::exists(e.path() / "meta.json")) test_block(e.path());
    } else std::printf("fixture: skipped (pass the merge tool's --fixture dir or set STRIX_H3_TURBO_FX)\n");
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
