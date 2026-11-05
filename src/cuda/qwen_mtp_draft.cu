// RTX MTP draft kernels (see include/strix/cuda_qwen_mtp_draft.hpp). Free arithmetic: drafts never reach output.
#include "strix/cuda_qwen_mtp_draft.hpp"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace strix::cuda::qwen_mtp_draft {
namespace {

constexpr unsigned kBlock = 34, kFull = 0xffffffffU;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qwen_mtp_draft ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float bf16f(std::uint16_t v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }
__device__ __forceinline__ std::uint16_t f2bf(float f) {  // RNE
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((u >> 16) | 64U);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float scale_of(const unsigned char* block) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(block)));
}

__global__ void q8_to_f32_kernel(const unsigned char* q8, std::uint64_t elements, float* out) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const unsigned char* b = q8 + i / 32U * kBlock;
    out[i] = scale_of(b) * static_cast<float>(static_cast<signed char>(b[2U + i % 32U]));
}

__global__ void gather_kernel(const unsigned char* src, std::uint64_t row_bytes, const std::uint32_t* ids, unsigned char* dst) {
    const unsigned char* s = src + static_cast<std::uint64_t>(ids[blockIdx.x]) * row_bytes;
    unsigned char* d = dst + static_cast<std::uint64_t>(blockIdx.x) * row_bytes;
    for (std::uint64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) d[i] = s[i];
}

// warp dot of a Q8_0 row (n features) with FP32 x in shared memory
template <unsigned N>
__device__ __forceinline__ float dot_q8(const unsigned char* row, const float* x, unsigned lane) {
    float acc = 0.0F;
    for (unsigned b = lane; b < N / 32U; b += 32U) {
        const unsigned char* blk = row + b * kBlock;
        float s = 0.0F;
#pragma unroll
        for (unsigned j = 0; j < 32U; ++j) s += x[b * 32U + j] * static_cast<float>(static_cast<signed char>(blk[2U + j]));
        acc += scale_of(blk) * s;
    }
#pragma unroll
    for (unsigned d = 16U; d; d /= 2U) acc += __shfl_xor_sync(kFull, acc, d);
    return acc;
}

// grid (kFfn / 8, kTopK), 256 threads: warp per ffn row of one slot
__global__ __launch_bounds__(256) void gate_up_kernel(const std::uint16_t* hidden, const std::uint32_t* routes, const unsigned char* gate,
    const unsigned char* up, float* scratch) {
    __shared__ float x[kHidden];
    for (unsigned i = threadIdx.x; i < kHidden; i += blockDim.x) x[i] = bf16f(hidden[i]);
    __syncthreads();
    const unsigned slot = blockIdx.y, r = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const std::uint64_t expert_bytes = std::uint64_t{kFfn} * (kHidden / 32U) * kBlock, row_bytes = (kHidden / 32U) * kBlock;
    const std::uint64_t off = routes[slot] * expert_bytes + r * row_bytes;
    const float g = dot_q8<kHidden>(gate + off, x, lane), u = dot_q8<kHidden>(up + off, x, lane);
    if (lane == 0U) scratch[slot * kFfn + r] = g / (1.0F + __expf(-g)) * u;
}

// grid kHidden / 8, 256 threads: warp per hidden output over all slots
__global__ __launch_bounds__(256) void down_kernel(const std::uint32_t* routes, const float* weights, const unsigned char* down,
    const float* scratch, std::uint16_t* out) {
    __shared__ float h[kTopK * kFfn];
    for (unsigned i = threadIdx.x; i < kTopK * kFfn; i += blockDim.x) h[i] = scratch[i];
    __syncthreads();
    const unsigned o = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const std::uint64_t expert_bytes = std::uint64_t{kHidden} * (kFfn / 32U) * kBlock, row_bytes = (kFfn / 32U) * kBlock;
    float sum = 0.0F;
    for (unsigned s = 0; s < kTopK; ++s)
        sum += weights[s] * dot_q8<kFfn>(down + routes[s] * expert_bytes + o * row_bytes, h + s * kFfn, lane);
    if (lane == 0U) out[o] = f2bf(sum);
}

__global__ __launch_bounds__(1024) void select_kernel(const float* logits, std::uint32_t count, const std::uint32_t* ids,
    std::uint32_t* out, std::uint32_t* next_token) {
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    float best = __int_as_float(0xff800000);
    unsigned index = 0xffffffffU;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
        if (logits[i] > best) { best = logits[i]; index = i; }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = index;
    __syncthreads();
    for (unsigned d = 512U; d; d /= 2U) {
        if (threadIdx.x < d) {
            const float v = bv[threadIdx.x + d];
            const unsigned j = bi[threadIdx.x + d];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && j < bi[threadIdx.x])) { bv[threadIdx.x] = v; bi[threadIdx.x] = j; }
        }
        __syncthreads();
    }
    const float m = bv[0];
    const unsigned arg = bi[0];
    __syncthreads();
    float s = 0.0F;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) s += __expf(logits[i] - m);
    bv[threadIdx.x] = s;
    __syncthreads();
    for (unsigned d = 512U; d; d /= 2U) {
        if (threadIdx.x < d) bv[threadIdx.x] += bv[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const unsigned token = arg < count ? ids[arg] : 0U;
        out[0] = token;
        out[1] = __float_as_uint(arg < count ? 1.0F / bv[0] : 0.0F);
        if (next_token) *next_token = token;
    }
}

}  // namespace

void q8_to_f32(const void* q8, std::uint64_t elements, float* out, cudaStream_t stream) {
    q8_to_f32_kernel<<<static_cast<unsigned>((elements + 255U) / 256U), 256, 0, stream>>>(static_cast<const unsigned char*>(q8), elements, out);
    check(cudaGetLastError(), "q8_to_f32");
}
void gather_rows(const void* src, std::uint64_t row_bytes, const std::uint32_t* ids, std::uint32_t count, void* dst,
    cudaStream_t stream) {
    if (!count) return;
    gather_kernel<<<count, 256, 0, stream>>>(static_cast<const unsigned char*>(src), row_bytes, ids, static_cast<unsigned char*>(dst));
    check(cudaGetLastError(), "gather");
}
void routed_q8_one_row(const std::uint16_t* hidden, const std::uint32_t* routes, const float* route_weights, const void* gate,
    const void* up, const void* down, float* scratch, std::uint16_t* out, cudaStream_t stream) {
    gate_up_kernel<<<dim3(kFfn / 8U, kTopK), 256, 0, stream>>>(hidden, routes, static_cast<const unsigned char*>(gate),
        static_cast<const unsigned char*>(up), scratch);
    check(cudaGetLastError(), "gate/up");
    down_kernel<<<kHidden / 8U, 256, 0, stream>>>(routes, route_weights, static_cast<const unsigned char*>(down), scratch, out);
    check(cudaGetLastError(), "down");
}
void select(const float* logits, std::uint32_t count, const std::uint32_t* ids, std::uint32_t* out, std::uint32_t* next_token,
    cudaStream_t stream) {
    select_kernel<<<1, 1024, 0, stream>>>(logits, count, ids, out, next_token);
    check(cudaGetLastError(), "select");
}

}  // namespace strix::cuda::qwen_mtp_draft
