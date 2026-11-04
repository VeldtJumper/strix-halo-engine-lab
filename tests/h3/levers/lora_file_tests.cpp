// CPU tests of the LoRA descriptor loader (src/h3/levers/lora/lora_file.hpp): a synthetic safetensors file (prefix,
// BF16 + F32 tensors, alpha, .diff, metadata) -> parse, ranks, scales, RNE conversion, diff application; malformed
// files rejected; optionally the real files (--real <dir>): FastH3 dense 6-step, Turbo v1.2 4-step, Turbo v4.
// usage: strix-h3-levers-lora-file-tests [--real <models dir>]
#include "h3/levers/lora/lora_file.hpp"
#include "h3/levers/lora/turbo_host.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace strix::h3::levers::lora;
namespace g = strix::h3::gemm;

namespace {
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct T {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    std::vector<char> bytes;
};
T bf16t(const std::string& n, std::vector<std::uint64_t> sh, float base) {
    T t{n, "BF16", sh, {}};
    std::uint64_t c = 1;
    for (auto d : sh) c *= d;
    for (std::uint64_t i = 0; i < c; ++i) {
        const std::uint16_t v = g::f32_to_bf16(base + 0.001F * static_cast<float>(i % 97));
        t.bytes.push_back(static_cast<char>(v & 0xffU));
        t.bytes.push_back(static_cast<char>(v >> 8U));
    }
    return t;
}
T f32t(const std::string& n, std::vector<std::uint64_t> sh, float base) {
    T t{n, "F32", sh, {}};
    std::uint64_t c = 1;
    for (auto d : sh) c *= d;
    for (std::uint64_t i = 0; i < c; ++i) {
        const float v = base + 0.000123F * static_cast<float>(i % 131);
        char b[4];
        std::memcpy(b, &v, 4);
        t.bytes.insert(t.bytes.end(), b, b + 4);
    }
    return t;
}
void write_st(const std::string& path, const std::vector<T>& ts, const std::string& meta) {
    std::string h = "{\"__metadata__\":" + meta;
    std::uint64_t off = 0;
    for (const auto& t : ts) {
        h += ",\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (std::size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
        h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) + "]}";
        off += t.bytes.size();
    }
    h += "}";
    std::ofstream f(path, std::ios::binary);
    const std::uint64_t n = h.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (const auto& t : ts) f.write(t.bytes.data(), static_cast<std::streamsize>(t.bytes.size()));
}
template <class F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

void test_synthetic() {
    const auto dir = std::filesystem::temp_directory_path();
    const std::string p = (dir / "h3_lora_file_test.safetensors").string();
    const float alpha = 8.0F;
    T al{"diffusion_model.blocks.0.attn.qkv_proj.alpha", "F32", {}, {}};
    al.bytes.resize(4);
    std::memcpy(al.bytes.data(), &alpha, 4);
    const std::vector<T> ts = {bf16t("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", {32, 64}, 0.5F),
                               bf16t("diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight", {192, 32}, -0.25F), al,
                               f32t("diffusion_model.blocks.0.mlp.fc1.lora_A.weight", {64, 64}, 0.1F),
                               f32t("diffusion_model.blocks.0.mlp.fc1.lora_B.weight", {128, 64}, 0.2F),
                               f32t("diffusion_model.blocks.0.norm1.diff", {64}, 0.01F), f32t("diffusion_model.video_patch_proj.diff_b", {64}, -0.02F)};
    write_st(p, ts, "{\"note\":\"a \\\"quoted\\\" {brace} value\",\"x\":\"1\"}");
    LoraFile f(p);
    bool ok = f.prefix() == "diffusion_model." && f.modules().size() == 2U && f.has_branch("blocks.0.attn.qkv_proj") && f.has_branch("blocks.0.mlp.fc1") &&
              f.diff_names().size() == 2U && f.metadata().at("note") == "a \"quoted\" {brace} value";
    const LoraBranch q = f.branch("blocks.0.attn.qkv_proj", 0.5F);
    ok = ok && q.rank == 32U && q.k == 64U && q.n == 192U && q.alpha == 8.0F && q.scale == 0.5F * 8.0F / 32.0F && q.a.size() == 32U * 64U &&
         q.b.size() == 192U * 32U && q.a[5] == g::f32_to_bf16(0.5F + 0.001F * 5.0F);
    const LoraBranch m = f.branch("blocks.0.mlp.fc1", 0.7F);
    std::size_t conv_bad = 0;
    for (std::size_t i = 0; i < m.a.size(); ++i) conv_bad += m.a[i] != g::f32_to_bf16(0.1F + 0.000123F * static_cast<float>(i % 131));
    ok = ok && m.rank == 64U && m.alpha == 0.0F && m.scale == 0.7F && conv_bad == 0;
    const std::vector<float> d = f.read_f32("blocks.0.norm1.diff");
    std::vector<std::uint16_t> w(64, g::f32_to_bf16(1.0F));
    apply_diff_bf16(w.data(), w.size(), d, 0.5F);
    std::size_t diff_bad = 0;
    for (std::size_t i = 0; i < 64; ++i) diff_bad += w[i] != g::f32_to_bf16(1.0F + 0.5F * d[i]);
    std::printf("synthetic file: prefix / modules / metadata ok %d, qkv r%u scale %g, fc1 r%u scale %g, F32 -> bf16 mismatches %zu, diff apply mismatches %zu\n",
        ok ? 1 : 0, q.rank, static_cast<double>(q.scale), m.rank, static_cast<double>(m.scale), conv_bad, diff_bad);
    CHECK(ok && diff_bad == 0, "synthetic LoRA file");
    // malformed files
    int rej = 0;
    const std::string p2 = (dir / "h3_lora_file_bad.safetensors").string();
    write_st(p2, {bf16t("blocks.0.attn.qkv_proj.lora_A.weight", {32, 64}, 0.5F)}, "{}");  // missing lora_B
    rej += throws([&] { LoraFile x(p2); });
    write_st(p2, {bf16t("blocks.0.attn.qkv_proj.weird", {32}, 0.5F)}, "{}");  // unknown key
    rej += throws([&] { LoraFile x(p2); });
    write_st(p2, {bf16t("diffusion_model.a.lora_A.weight", {32, 64}, 0.5F), bf16t("a.lora_B.weight", {64, 32}, 0.5F)}, "{}");  // mixed prefixes
    rej += throws([&] { LoraFile x(p2); });
    rej += throws([&] { f.branch("blocks.1.attn.qkv_proj", 1.0F); });
    rej += throws([&] { lora_identity("nope"); });
    std::printf("malformed: %d of 5 rejected\n", rej);
    CHECK(rej == 5, "malformed rejected");
    std::filesystem::remove(p);
    std::filesystem::remove(p2);
}

void test_real(const std::string& dir) {
    struct Want { const char* id; std::uint32_t rank_qkv, rank_other; float scale; std::size_t diffs; };
    for (const Want w : {Want{"h3-lora-fasth3-6step-v1", 192, 64, 1.0F, 29}, Want{"h3-lora-turbo-v12-v1", 384, 128, 0.0625F, 0}}) {
        const LoraIdentity id = lora_identity(w.id);  // a copy (gcc -Wdangling-reference false positive on the static table)
        const std::string path = dir + "/" + id.file;
        if (!std::filesystem::exists(path)) {
            std::printf("real %s: %s missing\n", w.id, path.c_str());
            ++failures;
            continue;
        }
        LoraFile f(path);
        std::size_t sites = 0, bad = 0;
        for (std::uint32_t b = 0; b < 50; ++b)
            for (Site s : {Site::kQkv, Site::kOut, Site::kFc1, Site::kFc2}) {
                const std::string mname = lora_module_name(s, b);
                if (!f.has_branch(mname)) { ++bad; continue; }
                ++sites;
                const LoraTensorInfo& ia = f.info(mname + ".lora_A.weight");
                bad += ia.shape[0] != (s == Site::kQkv ? w.rank_qkv : w.rank_other);
            }
        for (std::uint32_t b = 0; b < 2; ++b)
            for (Site s : {Site::kRefQkv, Site::kRefOut, Site::kRefFc1, Site::kRefFc2}) sites += f.has_branch(lora_module_name(s, b)) ? 1U : 0U;
        const LoraBranch q = f.branch(lora_module_name(Site::kQkv, 17), id.strength);
        const LoraBranch o = f.branch(lora_module_name(Site::kFc2, 3), id.strength);
        std::printf("real %s: prefix '%s', %zu branch sites (want 208), rank / shape mismatches %zu, qkv r%u k%u n%u scale %g, fc2 r%u k%u n%u scale %g, "
                    "%zu diffs (want %zu)\n", w.id, f.prefix().c_str(), sites, bad, q.rank, q.k, q.n, static_cast<double>(q.scale), o.rank, o.k, o.n,
            static_cast<double>(o.scale), f.diff_names().size(), w.diffs);
        CHECK(sites == 208U && bad == 0 && q.scale == w.scale && o.scale == w.scale && q.k == 5376U && q.n == 21504U && o.k == 14336U && o.n == 5376U &&
                  f.diff_names().size() == w.diffs, "real %s", w.id);
        if (w.diffs) {
            std::size_t vals = 0;
            for (const auto& dn : f.diff_names()) vals += f.read_f32(dn).size();
            std::printf("   diffs: %zu values in %zu tensors (first %s)\n", vals, f.diff_names().size(), f.diff_names().front().c_str());
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    test_synthetic();
    if (argc > 2 && std::string(argv[1]) == "--real") test_real(argv[2]);
    else std::printf("real files: skipped (pass --real <models dir>)\n");
    std::printf("lora file tests %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
