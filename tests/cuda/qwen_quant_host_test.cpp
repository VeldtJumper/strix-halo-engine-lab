// Host check of include/strix/cuda_qwen_quant.cuh (the decoders the RTX kernels use) against the CPU oracle
// src/ggml_dequant.cpp (the GGUF reference decoders) on real rows of GGUF files: every (tensor kind, type) present,
// a few rows each (first, last and spread), every 32-value group of the row compared bit for bit.
//   qwen_quant_host_test <file.gguf>...       (exit 0 = all equal)
#include "strix/cuda_qwen_quant.cuh"
#include "strix/ggml_dequant.hpp"
#include "strix/gguf_reader.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace q = strix::cuda::quant;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: qwen_quant_host_test file.gguf...\n"); return 1; }
    std::uint64_t groups = 0, rows_checked = 0, bad = 0;
    std::map<std::string, std::uint64_t> per_type;
    for (int a = 1; a < argc; ++a) {
        const auto info = strix::gguf::read_file(argv[a]);
        std::FILE* f = std::fopen(argv[a], "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[a]); return 2; }
        std::map<std::string, int> seen;  // tensor kind + type -> tensors checked
        for (const auto& t : info.tensors) {
            const auto geo = q::geometry(t.ggml_type);
            if (geo.block_elements == 0 || t.dimensions.empty()) {
                std::printf("skip %s type %u (not a decoder type)\n", t.name.c_str(), t.ggml_type);
                continue;
            }
            const auto dot = t.name.find('.', t.name.find('.') + 1);
            const std::string kind = (t.name.rfind("blk.", 0) == 0 ? t.name.substr(dot + 1) : t.name) + "/" +
                                     std::string(strix::ggml_dequant::name(t.ggml_type));
            if (seen[kind]++ >= 2) continue;  // two tensors per kind and type
            const std::uint64_t width = t.dimensions[0];
            if (width % 32 != 0) continue;
            const std::uint64_t row_bytes = strix::ggml_dequant::row_bytes(t.ggml_type, width);
            const std::uint64_t rows = t.byte_size / row_bytes;
            std::vector<std::uint64_t> pick{0, rows / 3, rows / 2, rows - 1};
            std::vector<std::uint8_t> raw(row_bytes);
            std::vector<float> ref(width), got(32);
            for (const auto r : pick) {
                _fseeki64(f, static_cast<long long>(t.absolute_offset + r * row_bytes), SEEK_SET);
                if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) { std::fprintf(stderr, "short read %s\n", t.name.c_str()); return 2; }
                strix::ggml_dequant::dequantize_row(t.ggml_type, std::as_bytes(std::span(raw)), ref);
                ++rows_checked;
                for (std::uint32_t g = 0; g < width / 32; ++g) {
                    q::decode_row_group(t.ggml_type, raw.data(), g, got.data());
                    ++groups;
                    ++per_type[std::string(strix::ggml_dequant::name(t.ggml_type))];
                    if (std::memcmp(got.data(), ref.data() + 32ULL * g, 128) != 0) {
                        if (bad < 8) {
                            for (int j = 0; j < 32; ++j)
                                if (std::memcmp(&got[j], &ref[32ULL * g + j], 4) != 0) {
                                    std::printf("DIFF %s row %llu group %u elem %d: %a vs %a\n", t.name.c_str(),
                                                static_cast<unsigned long long>(r), g, j, got[j], ref[32ULL * g + j]);
                                    break;
                                }
                        }
                        ++bad;
                    }
                }
            }
        }
        std::fclose(f);
    }
    for (const auto& [name, n] : per_type) std::printf("  %-8s %llu groups\n", name.c_str(), static_cast<unsigned long long>(n));
    std::printf("%s: %llu rows, %llu groups, %llu differing\n", bad ? "QUANT HOST FAIL" : "QUANT HOST PASS",
                static_cast<unsigned long long>(rows_checked), static_cast<unsigned long long>(groups),
                static_cast<unsigned long long>(bad));
    return bad ? 3 : 0;
}
