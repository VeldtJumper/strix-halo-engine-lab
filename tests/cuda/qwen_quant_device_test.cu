// Device check of the shared decoders and the Q8_0 transcode: for sampled rows of every (tensor kind, type) of the given
// GGUF files, the GPU decode of every 32-value group must equal the host decode bit for bit (the host decode equals the
// host reference oracle: qwen_quant_host_test), and transcode::to_q8_0 must equal the same rule computed on the host.
//   qwen_quant_device_test <file.gguf>...       (GPU; exit 0 = all equal)
#include "strix/cuda_qwen_quant.cuh"
#include "strix/cuda_qwen_transcode.hpp"
#include "strix/ggml_dequant.hpp"
#include "strix/gguf_reader.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace q = strix::cuda::quant;

__global__ void decode_kernel(const std::uint8_t* row, std::uint32_t type, std::uint32_t groups, float* out) {
    const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < groups) q::decode_row_group(type, row, g, out + 32ULL * g);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: qwen_quant_device_test file.gguf...\n"); return 1; }
    std::uint64_t groups_checked = 0, bad = 0, q8_bad = 0;
    for (int a = 1; a < argc; ++a) {
        const auto info = strix::gguf::read_file(argv[a]);
        std::FILE* f = std::fopen(argv[a], "rb");
        std::map<std::string, int> seen;
        for (const auto& t : info.tensors) {
            const auto geo = q::geometry(t.ggml_type);
            if (geo.block_elements == 0 || t.dimensions.empty() || t.dimensions[0] % 32 != 0) continue;
            const std::string kind = std::to_string(t.ggml_type) + "/" + t.name.substr(t.name.rfind('.', t.name.size() - 8) + 1);
            if (seen[kind]++ >= 1) continue;
            const std::uint64_t width = t.dimensions[0];
            const std::uint64_t row_bytes = strix::ggml_dequant::row_bytes(t.ggml_type, width);
            const std::uint64_t rows = t.byte_size / row_bytes;
            const std::uint32_t groups = static_cast<std::uint32_t>(width / 32);
            std::vector<std::uint8_t> raw(row_bytes);
            std::vector<float> host(width), dev(width);
            std::uint8_t* d_raw = nullptr;
            float* d_out = nullptr;
            void* d_q8 = nullptr;
            ck(cudaMalloc(&d_raw, row_bytes), "malloc");
            ck(cudaMalloc(&d_out, width * 4), "malloc");
            ck(cudaMalloc(&d_q8, width / 32 * 34), "malloc");
            for (const auto r : {std::uint64_t{0}, rows / 2, rows - 1}) {
                _fseeki64(f, static_cast<long long>(t.absolute_offset + r * row_bytes), SEEK_SET);
                if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) return 2;
                for (std::uint32_t g = 0; g < groups; ++g) q::decode_row_group(t.ggml_type, raw.data(), g, host.data() + 32ULL * g);
                ck(cudaMemcpy(d_raw, raw.data(), row_bytes, cudaMemcpyHostToDevice), "h2d");
                decode_kernel<<<(groups + 127) / 128, 128>>>(d_raw, t.ggml_type, groups, d_out);
                ck(cudaMemcpy(dev.data(), d_out, width * 4, cudaMemcpyDeviceToHost), "d2h");
                for (std::uint32_t g = 0; g < groups; ++g, ++groups_checked)
                    if (std::memcmp(host.data() + 32ULL * g, dev.data() + 32ULL * g, 128) != 0) {
                        if (bad++ < 6) std::printf("DECODE DIFF %s row %llu group %u\n", t.name.c_str(), static_cast<unsigned long long>(r), g);
                    }
                // Q8_0 transcode of the row vs the same rule on the host
                strix::cuda::transcode::to_q8_0(d_raw, t.ggml_type, width, d_q8, nullptr);
                std::vector<std::uint8_t> got(width / 32 * 34);
                ck(cudaMemcpy(got.data(), d_q8, got.size(), cudaMemcpyDeviceToHost), "d2h q8");
                for (std::uint32_t g = 0; g < groups; ++g) {
                    const float* v = host.data() + 32ULL * g;
                    float amax = 0.0F;
                    for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(v[j]));
                    const float dd = amax / 127.0F, id = dd != 0.0F ? 1.0F / dd : 0.0F;
                    const unsigned short h = __half_as_ushort(__float2half_rn(dd));
                    std::uint8_t want[34];
                    want[0] = static_cast<std::uint8_t>(h & 255U); want[1] = static_cast<std::uint8_t>(h >> 8U);
                    for (int j = 0; j < 32; ++j) want[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::round(v[j] * id)));
                    if (std::memcmp(want, got.data() + 34ULL * g, 34) != 0 && q8_bad++ < 6)
                        std::printf("Q8 DIFF %s row %llu group %u\n", t.name.c_str(), static_cast<unsigned long long>(r), g);
                }
            }
            cudaFree(d_raw); cudaFree(d_out); cudaFree(d_q8);
        }
        std::fclose(f);
    }
    std::printf("%s: %llu groups decoded on the device, %llu differing; Q8_0 transcode %llu differing\n",
                bad || q8_bad ? "QUANT DEVICE FAIL" : "QUANT DEVICE PASS", static_cast<unsigned long long>(groups_checked),
                static_cast<unsigned long long>(bad), static_cast<unsigned long long>(q8_bad));
    return bad || q8_bad ? 3 : 0;
}
