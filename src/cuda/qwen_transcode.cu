// Load-time transcoding of GGUF tensors into the storage forms the RTX kernels take (see cuda_qwen_transcode.hpp).
#include "strix/cuda_qwen_transcode.hpp"
#include "strix/cuda_qwen_quant.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace strix::cuda::transcode {
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("transcode ") + what + ": " + cudaGetErrorString(e));
}

// quantize_row_q8_0_ref of ggml: d = amax / 127, id = d ? 1 / d : 0, d stored as FP16, q = roundf(x * id).
__global__ void to_q8_0_kernel(const std::uint8_t* src, std::uint32_t type, std::uint64_t groups, std::uint8_t* dst) {
    const std::uint64_t g = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    float v[32];
    const quant::Geometry geo = quant::geometry(type);
    const std::uint64_t per = geo.block_elements / 32U;
    quant::decode_group(type, src + (g / per) * geo.block_bytes, static_cast<unsigned>(g % per), v);
    float amax = 0.0F;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(v[j]));
    const float d = __fdiv_rn(amax, 127.0F);
    const float id = d != 0.0F ? __fdiv_rn(1.0F, d) : 0.0F;
    std::uint8_t* out = dst + g * 34U;
    const unsigned short h = __half_as_ushort(__float2half_rn(d));
    out[0] = static_cast<std::uint8_t>(h & 255U);
    out[1] = static_cast<std::uint8_t>(h >> 8U);
    for (int j = 0; j < 32; ++j) out[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(__fmul_rn(v[j], id))));
}

__global__ void to_f32_kernel(const std::uint8_t* src, std::uint32_t type, std::uint64_t groups, float* dst) {
    const std::uint64_t g = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const quant::Geometry geo = quant::geometry(type);
    const std::uint64_t per = geo.block_elements / 32U;
    quant::decode_group(type, src + (g / per) * geo.block_bytes, static_cast<unsigned>(g % per), dst + 32U * g);
}

}  // namespace

bool supported(std::uint32_t ggml_type) { return quant::geometry(ggml_type).block_elements != 0; }

std::uint64_t q8_0_bytes(std::uint64_t elements) { return elements / 32U * 34U; }

void to_q8_0(const void* src, std::uint32_t ggml_type, std::uint64_t elements, void* dst, cudaStream_t stream) {
    if (!supported(ggml_type) || elements % 32U) throw std::invalid_argument("transcode to Q8_0: unsupported type or size");
    const std::uint64_t groups = elements / 32U;
    to_q8_0_kernel<<<static_cast<unsigned>((groups + 255U) / 256U), 256, 0, stream>>>(static_cast<const std::uint8_t*>(src), ggml_type,
        groups, static_cast<std::uint8_t*>(dst));
    check(cudaGetLastError(), "q8_0 launch");
}

void to_f32(const void* src, std::uint32_t ggml_type, std::uint64_t elements, float* dst, cudaStream_t stream) {
    if (!supported(ggml_type) || elements % 32U) throw std::invalid_argument("transcode to F32: unsupported type or size");
    const std::uint64_t groups = elements / 32U;
    to_f32_kernel<<<static_cast<unsigned>((groups + 255U) / 256U), 256, 0, stream>>>(static_cast<const std::uint8_t*>(src), ggml_type,
        groups, dst);
    check(cudaGetLastError(), "f32 launch");
}

}  // namespace strix::cuda::transcode
