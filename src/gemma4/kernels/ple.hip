#include "internal.hpp"
#include "a8_device.hpp"
#include "chunk.hpp"
#include "weight_policy.hpp"
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::gpu {
namespace {
bool stage_norm(unsigned width) {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_STAGE");
        return !p || std::string_view(p) != "0";
    }();
    return enabled && width <= 4096U;
}
bool register_heads() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_HEAD_REGISTERS");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool prefetch_norm() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_PREFETCH");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool parallel_search() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_SEARCH");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool bounded_search() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_BOUNDS");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
// Default on: seeded exact rsqrt in the fast epilogue and attention kernels.
bool seeded_rsqrt() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_SEEDED_RSQRT");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool interval_search() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_INTERVALS");
        return p && std::string_view(p) == "1";
    }();
    return enabled;
}
__device__ void stage_norm_row(const float* input, float* staged, unsigned width) {
    float values[16];
    // Width is admitted at <=4096 with 256 threads. Issue the independent
    // global reads before their LDS stores instead of waiting once per value.
#pragma unroll
    for (unsigned j = 0; j < 16; ++j) {
        const unsigned i = threadIdx.x + 256U * j;
        if (i < width)
            values[j] = input[i];
    }
#pragma unroll
    for (unsigned j = 0; j < 16; ++j) {
        const unsigned i = threadIdx.x + 256U * j;
        if (i < width)
            staged[i] = values[j];
    }
}
__device__ float norm_factor(const float* x, unsigned width, bool prefetch, bool parallel = false, bool bounded = true,
                             bool intervals = false) {
    const float sum = square_sum(x, width, prefetch);
    if ((bit_cast<std::uint32_t>(sum) & 0x7f800000U) == 0x7f800000U)
        return bit_cast<float>(0x7fc00000U);
    const float u = sum / float(width) + norm_eps;
    return parallel ? wave_rsqrt(u, bounded, intervals) : pow32_rsqrt(u);
}
__global__ void norm_kernel(const float* x, LinearView w, float* out, unsigned width, bool prefetch, unsigned* failure) {
    const auto base = std::size_t(blockIdx.x) * width;
    const float r = norm_factor(x + base, width, prefetch);
    for (unsigned i = threadIdx.x; i < width; i += 32U) {
        const float value = w.data ? (x[base + i] * r) * coefficient(w, 0, i) : x[base + i] * r;
        out[base + i] = value;
        if (failure)
            finite_flag(value, failure);
    }
}
template <bool Parallel, bool Staged = false, bool Quant = false>
__global__ void norm_group_kernel(const float* x, LinearView w, float* out, unsigned width, bool prefetch,
                                 unsigned* failure, bool bounded, bool intervals, A8Rows quant = {}) {
    const auto base = std::size_t(blockIdx.x) * width;
    __shared__ float factor;
    extern __shared__ float staged[];
    const float* input = x + base;
    if constexpr (Staged) {
        // Cooperating waves load the row; the first wave still owns the exact sum.
        stage_norm_row(input, staged, width);
        __syncthreads();
        input = staged;
    }
    // Only the first wave reduces. Its lane visits and tree are unchanged.
    if (threadIdx.x < 32U) {
        const float sum = square_sum(input, width, prefetch);
        float r = 0.0F;
        if constexpr (Parallel)
            r = wave_rsqrt(sum / float(width) + norm_eps, bounded, intervals);
        if (threadIdx.x == 0)
            factor = (bit_cast<std::uint32_t>(sum) & 0x7f800000U) == 0x7f800000U
                         ? bit_cast<float>(0x7fc00000U)
                         : (Parallel ? r : pow32_rsqrt(sum / float(width) + norm_eps));
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = w.data ? (input[i] * factor) * coefficient(w, 0, i) : input[i] * factor;
        out[base + i] = value;
        if (failure)
            finite_flag(value, failure);
        // g4-a8-v1: width is a multiple of 256, so every wave holds whole groups in lane order.
        if constexpr (Quant)
            a8_quantize_lane(value, quant, blockIdx.x, i);
    }
}
// Staged epilogue with every independent operand loaded before the first
// reduction: the branch row, residual, norm weights and following-norm weights
// are requested together, so the chain pays one memory round trip instead of
// three. The first wave's reductions and every elementwise operation keep the
// staged kernel's order; thread t still owns i = t + 256*j.
__global__ void __launch_bounds__(256) norm_residual_early_kernel(const float* branch, LinearView w, float* residual,
                                                                  unsigned width, LinearView scalar, float* next,
                                                                  LinearView next_weight, unsigned* failure,
                                                                  bool prefetch, bool bounded, bool intervals) {
    constexpr unsigned J = 16;
    const auto base = std::size_t(blockIdx.x) * width;
    __shared__ float factor;
    extern __shared__ float staged[];
    float b[J], r[J], g[J], h[J];
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned i = threadIdx.x + 256U * j;
        if (i < width) {
            b[j] = branch[base + i];
            r[j] = residual[base + i];
            g[j] = coefficient(w, 0, i);
            if (next)
                h[j] = coefficient(next_weight, 0, i);
        }
    }
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned i = threadIdx.x + 256U * j;
        if (i < width)
            staged[i] = b[j];
    }
    __syncthreads();
    if (threadIdx.x < 32U) {
        const float f = epilogue_factor<true>(staged, width, prefetch, bounded, intervals);
        if (threadIdx.x == 0)
            factor = f;
    }
    __syncthreads();
    const float scale = scalar.data ? coefficient(scalar, 0, 0) : 0.0F;
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned i = threadIdx.x + 256U * j;
        if (i < width) {
            const float normalized = (b[j] * factor) * g[j];
            finite_flag(normalized, failure);
            const float a = r[j] + normalized;
            const float value = scalar.data ? a * scale : a;
            finite_flag(value, failure);
            residual[base + i] = value;
            r[j] = value;
        }
    }
    if (!next)
        return;
    // Every thread has read its staged branch values before the first barrier
    // above and the reduction has finished; reuse the staging for the residual.
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned i = threadIdx.x + 256U * j;
        if (i < width)
            staged[i] = r[j];
    }
    __syncthreads();
    if (threadIdx.x < 32U) {
        const float f = epilogue_factor<true>(staged, width, prefetch, bounded, intervals);
        if (threadIdx.x == 0)
            factor = f;
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const unsigned i = threadIdx.x + 256U * j;
        if (i < width) {
            const float value = (r[j] * factor) * h[j];
            next[base + i] = value;
            finite_flag(value, failure);
        }
    }
}
// Norm weight lookup specialized on the packed format. An aligned f32 row is
// read as one word; its bits equal coefficient()'s little-endian byte assembly.
enum class NormWeights : unsigned { f32, bf16, any };
template <NormWeights F> __device__ inline float norm_weight(LinearView w, unsigned i) {
    if constexpr (F == NormWeights::f32)
        return reinterpret_cast<const float*>(w.data)[i];
    else
        return bf16_value(reinterpret_cast<const std::uint16_t*>(w.data)[i]);
}
// The staged epilogue with every operand (branch, residual, both norm weights,
// layer scalar) requested before the first reduction, no per-element format
// dispatch and no bounds guards (width = 256 * W), so the chain pays one memory
// round trip instead of one per element stride. Reductions, rsqrt and every
// elementwise operation keep norm_residual_kernel<true, true>'s order; thread t
// owns i = t + 256*j.
template <NormWeights F, unsigned W, bool Scaled, bool Next>
__global__ void __launch_bounds__(256) norm_residual_fast_kernel(const float* __restrict__ branch, LinearView w,
                                                                 float* __restrict__ residual, LinearView scalar,
                                                                 float* __restrict__ next, LinearView next_weight,
                                                                 unsigned* failure, bool prefetch, bool bounded,
                                                                 bool intervals, bool seeded, A8Rows quant) {
    constexpr unsigned width = 256U * W;
    const auto base = std::size_t(blockIdx.x) * width;
    __shared__ float factor;
    __shared__ float row_values[width];
    float b[W], r[W], g[W], h[W];
#pragma unroll
    for (unsigned j = 0; j < W; ++j) {
        const unsigned i = threadIdx.x + 256U * j;
        b[j] = branch[base + i];
        r[j] = residual[base + i];
        g[j] = norm_weight<F>(w, i);
        if constexpr (Next)
            h[j] = norm_weight<F>(next_weight, i);
    }
    float scale = 0.0F;
    if constexpr (Scaled)
        scale = norm_weight<F>(scalar, 0);
#pragma unroll
    for (unsigned j = 0; j < W; ++j)
        row_values[threadIdx.x + 256U * j] = b[j];
    __syncthreads();
    if (threadIdx.x < 32U) {
        const float f = epilogue_factor<true>(row_values, width, prefetch, bounded, intervals, seeded);
        if (threadIdx.x == 0)
            factor = f;
    }
    __syncthreads();
    const float first = factor;
#pragma unroll
    for (unsigned j = 0; j < W; ++j) {
        const float normalized = (b[j] * first) * g[j];
        finite_flag(normalized, failure);
        const float a = r[j] + normalized;
        const float value = Scaled ? a * scale : a;
        finite_flag(value, failure);
        residual[base + threadIdx.x + 256U * j] = value;
        r[j] = value;
        // g4-a8-v1: every wave holds whole 32-value groups in lane order.
        if (!Next && quant.codes)
            a8_quantize_lane(value, quant, blockIdx.x, threadIdx.x + 256U * j);
    }
    if constexpr (Next) {
        // The row_values branch was last read by the first wave's reduction, which
        // completed before the barrier above; reuse the staging for the residual.
#pragma unroll
        for (unsigned j = 0; j < W; ++j)
            row_values[threadIdx.x + 256U * j] = r[j];
        __syncthreads();
        if (threadIdx.x < 32U) {
            const float f = epilogue_factor<true>(row_values, width, prefetch, bounded, intervals, seeded);
            if (threadIdx.x == 0)
                factor = f;
        }
        __syncthreads();
        const float second = factor;
#pragma unroll
        for (unsigned j = 0; j < W; ++j) {
            const float value = (r[j] * second) * h[j];
            next[base + threadIdx.x + 256U * j] = value;
            finite_flag(value, failure);
            if (quant.codes)
                a8_quantize_lane(value, quant, blockIdx.x, threadIdx.x + 256U * j);
        }
    }
}
// The FFN epilogue (post-FFN norm + residual add, no layer scalar, no following
// norm) folded into the PLE gate linear with its activation store. Every block
// recomputes the epilogue rows with norm_residual_fast_kernel's arithmetic
// (first-wave ordered reduction from the staged branch, same rsqrt, same
// elementwise order) into shared memory. The last block to take a ticket (all
// blocks have consumed the old residual by then) stores the new residual.
// Each gate output is then the chunk kernel's lane-chunk dot over those values
// (the same inputs, visits and order) followed by the fused activation store.
template <NormWeights F, unsigned W, unsigned Rows, WeightFormat Format, bool WideHeader, bool Streaming>
__global__ void __launch_bounds__(128) ffn_ple_gate_kernel(const float* __restrict__ branch, LinearView norm,
                                                           float* __restrict__ residual, LinearView gate,
                                                           const float* __restrict__ ple, unsigned product_stride,
                                                           float* __restrict__ out, unsigned* failure, unsigned* ticket,
                                                           bool prefetch, bool bounded, bool intervals,
                                                           bool seeded, bool hardware_half) {
    constexpr unsigned width = 256U * W, J = 2U * W;
    __shared__ float x[Rows][width];
    __shared__ float factor;
    const unsigned lane = threadIdx.x % 32U;
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        const auto base = std::size_t(r) * width;
        float b[J], h[J], g[J];
#pragma unroll
        for (unsigned j = 0; j < J; ++j) {
            const unsigned i = threadIdx.x + 128U * j;
            b[j] = branch[base + i];
            h[j] = residual[base + i];
            g[j] = norm_weight<F>(norm, i);
        }
#pragma unroll
        for (unsigned j = 0; j < J; ++j)
            x[r][threadIdx.x + 128U * j] = b[j];
        __syncthreads();
        if (threadIdx.x < 32U) {
            const float f = epilogue_factor<true>(x[r], width, prefetch, bounded, intervals, seeded);
            if (threadIdx.x == 0)
                factor = f;
        }
        __syncthreads();
        const float scale = factor;
#pragma unroll
        for (unsigned j = 0; j < J; ++j) {
            const unsigned i = threadIdx.x + 128U * j;
            const float normalized = (b[j] * scale) * g[j];
            finite_flag(normalized, failure);
            const float value = h[j] + normalized;
            finite_flag(value, failure);
            x[r][i] = value;
        }
        // The next row's reduction rewrites factor only after its staging barrier.
    }
    __shared__ bool last;
    __syncthreads();
    if (threadIdx.x == 0)
        last = atomicAdd(ticket, 1U) == gridDim.x - 1U;
    __syncthreads();
    if (last) {
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r)
#pragma unroll
            for (unsigned j = 0; j < J; ++j)
                residual[std::size_t(r) * width + threadIdx.x + 128U * j] = x[r][threadIdx.x + 128U * j];
        // Launches on the queue are ordered: the next one sees the reset ticket.
        if (threadIdx.x == 0)
            *ticket = 0U;
    }
    const unsigned o = blockIdx.x * 4U + threadIdx.x / 32U;
    if (o >= gate.outputs)
        return;
    const auto* wp = gate.data + std::size_t(o) * gate.row_bytes;
    float acc[Rows] = {};
    for (unsigned c = lane; c < width / 16U; c += 32U) {
        const auto start = c * 16U;
        const WeightChunk<Format, Streaming> a(wp, start, WideHeader, hardware_half);
#pragma unroll
        for (unsigned e = 0; e < 16; ++e) {
            const float v = a.at(e);
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r)
                acc[r] = acc[r] + v * x[r][start + e];
        }
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        float value = wave_sum(acc[r]);
        if (lane == 0) {
            finite_flag(value, failure);
            value = gelu_tanh(value) * ple[std::size_t(r) * product_stride + o];
            out[std::size_t(r) * gate.outputs + o] = value;
            finite_flag(value, failure);
        }
    }
}
template <bool Parallel, bool Staged = false, bool Quant = false>
__global__ void norm_residual_kernel(const float* branch, LinearView w, float* residual, unsigned width,
                                     LinearView scalar, float* next, LinearView next_weight, unsigned* failure,
                                     bool prefetch, bool bounded, bool intervals, A8Rows quant = {}) {
    const auto base = std::size_t(blockIdx.x) * width;
    __shared__ float factor;
    extern __shared__ float staged[];
    const float* input = branch + base;
    if constexpr (Staged) {
        stage_norm_row(input, staged, width);
        __syncthreads();
        input = staged;
    }
    if (threadIdx.x < 32U) {
        const float r = epilogue_factor<Parallel>(input, width, prefetch, bounded, intervals);
        if (threadIdx.x == 0)
            factor = r;
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        const float b = (input[i] * factor) * coefficient(w, 0, i);
        finite_flag(b, failure);
        const float a = residual[base + i] + b;
        const float value = scalar.data ? a * coefficient(scalar, 0, 0) : a;
        finite_flag(value, failure);
        residual[base + i] = value;
        if constexpr (Staged)
            staged[i] = value;
        if constexpr (Quant)
            if (!next)
                a8_quantize_lane(value, quant, blockIdx.x, i);
    }
    if (next) {
        __syncthreads();
        if (threadIdx.x < 32U) {
            const float r = epilogue_factor<Parallel>(Staged ? staged : residual + base, width, prefetch,
                                                       bounded, intervals);
            if (threadIdx.x == 0)
                factor = r;
        }
        __syncthreads();
        for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
            const float value = ((Staged ? staged[i] : residual[base + i]) * factor) * coefficient(next_weight, 0, i);
            next[base + i] = value;
            finite_flag(value, failure);
            if constexpr (Quant)
                a8_quantize_lane(value, quant, blockIdx.x, i);
        }
    }
}
__global__ void head_kernel(float* x, LinearView w, unsigned count, unsigned dim, const std::uint64_t* positions,
                            const float* table, bool global, bool prefetch, bool parallel, unsigned* failure,
                            bool bounded, bool intervals) {
    auto* v = x + std::size_t(blockIdx.x) * dim;
    const float r = norm_factor(v, dim, prefetch, parallel, bounded, intervals);
    for (unsigned i = threadIdx.x; i < dim; i += 32U)
        v[i] = w.data ? (v[i] * r) * coefficient(w, 0, i) : v[i] * r;
    __syncthreads();
    const auto pairs = global ? 256U : 128U;
    const auto base = std::size_t(positions[blockIdx.x / count]) * pairs * 2U;
    const auto active = min(dim / 2U, global ? 64U : 128U);
    for (unsigned i = threadIdx.x; i < active; i += 32U) {
        const float a = v[i], b = v[i + dim / 2U], cs = table[base + 2U * i], sn = table[base + 2U * i + 1];
        v[i] = a * cs - b * sn;
        v[i + dim / 2U] = b * cs + a * sn;
    }
    if (failure) {
        // Small fixture heads can assign the two coordinates to different lanes.
        __syncthreads();
        for (unsigned i = threadIdx.x; i < dim; i += 32U)
            finite_flag(v[i], failure);
    }
}
template <unsigned Dim>
__global__ void register_head_kernel(float* x, LinearView w, unsigned count, const std::uint64_t* positions,
                                     const float* table, bool parallel, unsigned* failure, bool bounded, bool intervals) {
    constexpr unsigned Values = Dim / 32U, Active = Dim == 512U ? 64U : 128U, Rotations = Active / 32U;
    const unsigned lane = threadIdx.x;
    auto* out = x + std::size_t(blockIdx.x) * Dim;
    const auto base = std::size_t(positions[blockIdx.x / count]) * Dim;
    float values[Values], weights[Values], cs[Rotations], sn[Rotations];
#pragma unroll
    for (unsigned j = 0; j < Values; ++j)
        values[j] = out[lane + 32U * j];
#pragma unroll
    for (unsigned j = 0; j < Values; ++j)
        if (w.data)
            weights[j] = coefficient(w, 0, lane + 32U * j);
#pragma unroll
    for (unsigned j = 0; j < Rotations; ++j) {
        cs[j] = table[base + 2U * (lane + 32U * j)];
        sn[j] = table[base + 2U * (lane + 32U * j) + 1U];
    }
    float sum = 0.0F;
#pragma unroll
    for (unsigned j = 0; j < Values; ++j)
        sum = sum + values[j] * values[j];
    sum = wave_sum(sum);
    const float u = sum / float(Dim) + norm_eps;
    const float r = (bit_cast<std::uint32_t>(sum) & 0x7f800000U) == 0x7f800000U
                        ? bit_cast<float>(0x7fc00000U)
                        : (parallel ? wave_rsqrt(u, bounded, intervals) : pow32_rsqrt(u));
#pragma unroll
    for (unsigned j = 0; j < Values; ++j)
        values[j] = w.data ? (values[j] * r) * weights[j] : values[j] * r;
    // Both coordinates of each declared pair belong to this lane. Retain every
    // multiply/add boundary while avoiding the intermediate global round trip.
#pragma unroll
    for (unsigned j = 0; j < Rotations; ++j) {
        const float a = values[j], b = values[j + Values / 2U];
        values[j] = a * cs[j] - b * sn[j];
        values[j + Values / 2U] = b * cs[j] + a * sn[j];
    }
#pragma unroll
    for (unsigned j = 0; j < Values; ++j) {
        out[lane + 32U * j] = values[j];
        if (failure)
            finite_flag(values[j], failure);
    }
}
__global__ void embedding_kernel(LinearView table, const unsigned* tokens, float* out, float scale) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (i < table.inputs)
        out[std::size_t(row) * table.inputs + i] = embedding_value(coefficient(table, tokens[row], i), scale);
}
__global__ void ple_kernel(float* p, LinearView table, LinearView w, const unsigned* tokens, unsigned layer_count,
                           unsigned width, bool prefetch, bool parallel, unsigned* failure, bool bounded, bool intervals) {
    const unsigned row = blockIdx.x / layer_count, layer = blockIdx.x % layer_count;
    float* x = p + std::size_t(blockIdx.x) * width;
    const float sum = square_sum<true>(x, width, prefetch);
    const float r = (bit_cast<std::uint32_t>(sum) & 0x7f800000U) == 0x7f800000U
                        ? bit_cast<float>(0x7fc00000U)
                        : (parallel ? wave_rsqrt(sum / float(width) + norm_eps, bounded, intervals)
                                    : pow32_rsqrt(sum / float(width) + norm_eps));
    for (unsigned i = threadIdx.x; i < width; i += 32U) {
        const float a = ((x[i] * model_ple_scale) * r) * coefficient(w, 0, i);
        const float t = embedding_value(coefficient(table, tokens[row], layer * width + i), ple_embedding_scale);
        x[i] = (a + t) * ple_combine_scale;
        if (failure)
            finite_flag(x[i], failure);
    }
}
__global__ void ple_gate_kernel(float* gate, const float* ple, unsigned layer_count, unsigned layer, unsigned width,
                               unsigned* failure) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (i < width) {
        const auto offset = std::size_t(row) * width + i;
        gate[offset] = gelu_tanh(gate[offset]) * ple[(std::size_t(row) * layer_count + layer) * width + i];
        if (failure)
            finite_flag(gate[offset], failure);
    }
}
__global__ void residual_kernel(float* x, const float* b, unsigned n, LinearView scalar) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = x[i] + b[i];
        x[i] = scalar.data ? a * coefficient(scalar, 0, 0) : a;
    }
}
__global__ void product_kernel(float* g, const float* u, unsigned n) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        g[i] = gelu_tanh(g[i]) * u[i];
}
__global__ void product_finite_kernel(float* g, const float* u, unsigned n, unsigned* failure) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float value = gelu_tanh(g[i]) * u[i];
        g[i] = value;
        finite_flag(value, failure);
    }
}
__global__ void cap_kernel(float* x, unsigned n) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = softcap(x[i]);
}
__global__ void finite_kernel(const float* x, std::size_t n, unsigned* failure) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        finite_flag(x[i], failure);
}
__global__ void candidate_kernel(const float* x, unsigned vocabulary, Candidate* candidates) {
    if (threadIdx.x != 0)
        return;
    const auto row = blockIdx.y, begin = blockIdx.x * 256U, end = min(vocabulary, begin + 256U);
    Candidate a{bit_cast<float>(0xff800000U), UINT32_MAX};
    for (auto i = begin; i < end; ++i) {
        const float v = x[std::size_t(row) * vocabulary + i];
        if (better(v, i, a.value, a.id))
            a = {v, i};
    }
    candidates[std::size_t(row) * gridDim.x + blockIdx.x] = a;
}
__global__ void merge_kernel(const Candidate* candidates, unsigned count, Candidate* result) {
    if (threadIdx.x != 0)
        return;
    Candidate a{bit_cast<float>(0xff800000U), UINT32_MAX};
    for (unsigned i = 0; i < count; ++i) {
        const auto b = candidates[std::size_t(blockIdx.x) * count + i];
        if (better(b.value, b.id, a.value, a.id))
            a = b;
    }
    result[blockIdx.x] = a;
}
// (value, id) under better() is a total order on non-NaN values with distinct
// ids, and NaN never wins, so any reduction tree selects the same candidate as
// the ordered scan.
__device__ void keep_better(float& value, unsigned& id, float other, unsigned other_id) {
    if (better(other, other_id, value, id)) {
        value = other;
        id = other_id;
    }
}
__device__ void block_select(float value, unsigned id, Candidate* out) {
    __shared__ float values[8];
    __shared__ unsigned ids[8];
    for (unsigned d = 16; d; d >>= 1U)
        keep_better(value, id, __shfl_down(value, d, 32), unsigned(__shfl_down(int(id), d, 32)));
    if (threadIdx.x % 32U == 0) {
        values[threadIdx.x / 32U] = value;
        ids[threadIdx.x / 32U] = id;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned w = 1; w < blockDim.x / 32U; ++w)
            keep_better(value, id, values[w], ids[w]);
        *out = {value, id};
    }
}
__global__ void block_candidate_kernel(const float* x, unsigned vocabulary, Candidate* candidates) {
    const auto row = blockIdx.y, begin = blockIdx.x * 2048U;
    float value = bit_cast<float>(0xff800000U);
    unsigned id = UINT32_MAX;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        const auto i = begin + threadIdx.x + 256U * j;
        if (i < vocabulary)
            keep_better(value, id, x[std::size_t(row) * vocabulary + i], i);
    }
    block_select(value, id, candidates + std::size_t(row) * gridDim.x + blockIdx.x);
}
__global__ void block_merge_kernel(const Candidate* candidates, unsigned count, Candidate* result) {
    float value = bit_cast<float>(0xff800000U);
    unsigned id = UINT32_MAX;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const auto c = candidates[std::size_t(blockIdx.x) * count + i];
        keep_better(value, id, c.value, c.id);
    }
    block_select(value, id, result + blockIdx.x);
}
bool fast_epilogue() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_FAST_EPILOGUE");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
NormWeights norm_weights(LinearView w) {
    if (w.format == WeightFormat::f32 && reinterpret_cast<std::uintptr_t>(w.data) % 4U == 0)
        return NormWeights::f32;
    if (w.format == WeightFormat::bf16 && reinterpret_cast<std::uintptr_t>(w.data) % 2U == 0)
        return NormWeights::bf16;
    return NormWeights::any;
}
bool overlaps(const float* a, const float* b, std::size_t count) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    const auto bytes = count * sizeof(float);
    return x < y + bytes && y < x + bytes;
}
template <NormWeights F, unsigned W>
void launch_fast(const float* branch, LinearView w, float* residual, unsigned rows, LinearView scalar, float* next,
                 LinearView next_weight, unsigned* failure, Queue q, A8Rows quant) {
#define G4_FAST_LAUNCH(S, N)                                                                                               \
    hipLaunchKernelGGL((norm_residual_fast_kernel<F, W, S, N>), dim3(rows), dim3(256), 0, static_cast<hipStream_t>(q),    \
                       branch, w, residual, scalar, next, next_weight, failure, prefetch_norm(), bounded_search(),          \
                       interval_search(), seeded_rsqrt(), quant)
    if (scalar.data && next)
        G4_FAST_LAUNCH(true, true);
    else if (scalar.data)
        G4_FAST_LAUNCH(true, false);
    else if (next)
        G4_FAST_LAUNCH(false, true);
    else
        G4_FAST_LAUNCH(false, false);
#undef G4_FAST_LAUNCH
}
template <NormWeights F>
bool launch_fast_width(const float* branch, LinearView w, float* residual, unsigned rows, unsigned width,
                       LinearView scalar, float* next, LinearView next_weight, unsigned* failure, Queue q,
                       A8Rows quant) {
    switch (width) {
    case 2048U:
        launch_fast<F, 8>(branch, w, residual, rows, scalar, next, next_weight, failure, q, quant);
        return true;
    case 2560U:
        launch_fast<F, 10>(branch, w, residual, rows, scalar, next, next_weight, failure, q, quant);
        return true;
    case 3072U:
        launch_fast<F, 12>(branch, w, residual, rows, scalar, next, next_weight, failure, q, quant);
        return true;
    case 4096U:
        launch_fast<F, 16>(branch, w, residual, rows, scalar, next, next_weight, failure, q, quant);
        return true;
    default:
        return false;
    }
}
template <NormWeights F, unsigned Rows, WeightFormat Format, bool Streaming>
void launch_ffn_ple_gate(const float* branch, LinearView norm, float* residual, LinearView gate, const float* ple,
                         unsigned product_stride, float* out, unsigned* failure, unsigned* ticket, bool hardware_half,
                         Queue q) {
    hipLaunchKernelGGL((ffn_ple_gate_kernel<F, 10, Rows, Format, true, Streaming>), dim3((gate.outputs + 3U) / 4U),
                       dim3(128), 0, static_cast<hipStream_t>(q), branch, norm, residual, gate, ple, product_stride,
                       out, failure, ticket, prefetch_norm(), bounded_search(), interval_search(), seeded_rsqrt(),
                       hardware_half);
}
template <NormWeights F, unsigned Rows>
bool launch_ffn_ple_gate_format(const float* branch, LinearView norm, float* residual, LinearView gate,
                                const float* ple, unsigned product_stride, float* out, unsigned* failure,
                                unsigned* ticket, bool hardware_half, Queue q) {
    switch (gate.format) {
    case WeightFormat::q4_k:
        launch_ffn_ple_gate<F, Rows, WeightFormat::q4_k, false>(branch, norm, residual, gate, ple, product_stride, out,
                                                                failure, ticket, hardware_half, q);
        return true;
    case WeightFormat::q6_k:
        launch_ffn_ple_gate<F, Rows, WeightFormat::q6_k, false>(branch, norm, residual, gate, ple, product_stride, out,
                                                                failure, ticket, hardware_half, q);
        return true;
    default:
        return false;
    }
}
// Selects the fast epilogue for a specialized width when the norm weights (and
// the layer scalar) share an f32/bf16 format and the activation rows do not
// overlap (the kernel reads them restrict); false enqueues nothing.
bool launch_fast_epilogue(const float* branch, LinearView w, float* residual, unsigned rows, unsigned width,
                          LinearView scalar, float* next, LinearView next_weight, unsigned* failure, Queue q,
                          A8Rows quant) {
    const auto count = std::size_t(rows) * width;
    const auto format = norm_weights(w);
    if (format == NormWeights::any || (next && norm_weights(next_weight) != format) ||
        (scalar.data && (scalar.inputs != 1U || scalar.outputs != 1U || norm_weights(scalar) != format)) ||
        overlaps(branch, residual, count) || (next && (overlaps(next, residual, count) || overlaps(next, branch, count))) ||
        (quant.codes && (quant.width != width || width % 256U != 0)))
        return false;
    if (format == NormWeights::f32)
        return launch_fast_width<NormWeights::f32>(branch, w, residual, rows, width, scalar, next, next_weight, failure, q,
                                                            quant);
    return launch_fast_width<NormWeights::bf16>(branch, w, residual, rows, width, scalar, next, next_weight, failure, q,
                                                            quant);
}
} // namespace
bool norm(const float* x, LinearView w, float* out, unsigned vectors, unsigned width, Queue q, unsigned* failure,
          A8Rows quant) {
    if (!vectors || !width || (w.data && (w.outputs != 1 || w.inputs != width)))
        throw std::runtime_error("device norm geometry");
    static const bool grouped = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_GROUP");
        return !p || std::string_view(p) != "0";
    }();
    static const bool parallel = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_SEARCH");
        return !p || std::string_view(p) != "0";
    }();
    // Quantized output only from the staged parallel schedule; otherwise the plain norm runs and the caller
    // quantizes separately.
    if (quant.codes && grouped && stage_norm(width) && parallel && width % 256U == 0 && quant.width == width &&
        vectors <= 8U) {
        hipLaunchKernelGGL((norm_group_kernel<true, true, true>), dim3(vectors), dim3(256), width * sizeof(float),
                           static_cast<hipStream_t>(q), x, w, out, width, prefetch_norm(), failure,
                           bounded_search(), interval_search(), quant);
        launch_checked();
        return true;
    }
    if (grouped && stage_norm(width) && parallel)
        hipLaunchKernelGGL((norm_group_kernel<true, true>), dim3(vectors), dim3(256), width * sizeof(float),
                           static_cast<hipStream_t>(q), x, w, out, width, prefetch_norm(), failure,
                           bounded_search(), interval_search());
    else if (grouped && stage_norm(width))
        hipLaunchKernelGGL((norm_group_kernel<false, true>), dim3(vectors), dim3(256), width * sizeof(float),
                           static_cast<hipStream_t>(q), x, w, out, width, prefetch_norm(), failure,
                           bounded_search(), interval_search());
    else if (grouped && parallel)
        hipLaunchKernelGGL((norm_group_kernel<true>), dim3(vectors), dim3(256), 0, static_cast<hipStream_t>(q), x, w,
                           out, width, prefetch_norm(), failure, bounded_search(), interval_search());
    else if (grouped)
        hipLaunchKernelGGL((norm_group_kernel<false>), dim3(vectors), dim3(256), 0, static_cast<hipStream_t>(q), x, w,
                           out, width, prefetch_norm(), failure, bounded_search(), interval_search());
    else
        hipLaunchKernelGGL(norm_kernel, dim3(vectors), dim3(32), 0, static_cast<hipStream_t>(q), x, w, out, width,
                           prefetch_norm(), failure);
    launch_checked();
    return false;
}
void head_norm_rope(float* x, LinearView w, unsigned rows, unsigned count, unsigned dim, const std::uint64_t* p,
                    const float* t, bool global, Queue q, unsigned* failure) {
    valid_rows(rows);
    if (!count || !dim || dim % 2 || dim > (global ? 512U : 256U) || (w.data && (w.inputs != dim || w.outputs != 1)))
        throw std::runtime_error("device head geometry");
    if (register_heads() && !global && dim == 256U)
        hipLaunchKernelGGL((register_head_kernel<256>), dim3(rows * count), dim3(32), 0, static_cast<hipStream_t>(q),
                           x, w, count, p, t, parallel_search(), failure, bounded_search(), interval_search());
    else if (register_heads() && global && dim == 512U)
        hipLaunchKernelGGL((register_head_kernel<512>), dim3(rows * count), dim3(32), 0, static_cast<hipStream_t>(q),
                           x, w, count, p, t, parallel_search(), failure, bounded_search(), interval_search());
    else
        hipLaunchKernelGGL(head_kernel, dim3(rows * count), dim3(32), 0, static_cast<hipStream_t>(q), x, w, count, dim, p,
                           t, global, prefetch_norm(), parallel_search(), failure, bounded_search(), interval_search());
    launch_checked();
}
bool norm_residual(const float* branch, LinearView w, float* residual, unsigned rows, unsigned width, LinearView scalar,
                   float* next, LinearView next_weight, unsigned* failure, Queue q, A8Rows quant) {
    valid_rows(rows);
    if (!width || !w.data || w.outputs != 1 || w.inputs != width || !failure ||
        (next && (!next_weight.data || next_weight.outputs != 1 || next_weight.inputs != width)))
        throw std::runtime_error("device norm residual geometry");
    static const bool parallel = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_SEARCH");
        return !p || std::string_view(p) != "0";
    }();
    static const bool early = [] {
        // Opt-in: measured neutral to slightly slower at C1.
        const auto* p = std::getenv("STRIX_G4_EPILOGUE_EARLY_LOADS");
        return p && std::string_view(p) == "1";
    }();
    if (!early && stage_norm(width) && parallel && fast_epilogue() &&
        launch_fast_epilogue(branch, w, residual, rows, width, scalar, next, next_weight, failure, q,
                             quant)) {
        launch_checked();
        return quant.codes != nullptr;
    }
    if (quant.codes && !early && stage_norm(width) && parallel && width % 256U == 0 && quant.width == width) {
        hipLaunchKernelGGL((norm_residual_kernel<true, true, true>), dim3(rows), dim3(256), width * sizeof(float),
                           static_cast<hipStream_t>(q), branch, w, residual, width, scalar, next, next_weight, failure,
                           prefetch_norm(), bounded_search(), interval_search(), quant);
        launch_checked();
        return true;
    }
    if (early && stage_norm(width) && parallel)
        hipLaunchKernelGGL(norm_residual_early_kernel, dim3(rows), dim3(256), width * sizeof(float),
                           static_cast<hipStream_t>(q), branch, w, residual, width, scalar, next, next_weight, failure,
                           prefetch_norm(), bounded_search(), interval_search());
    else if (stage_norm(width) && parallel)
        hipLaunchKernelGGL((norm_residual_kernel<true, true>), dim3(rows), dim3(256), width * sizeof(float),
                           static_cast<hipStream_t>(q), branch, w, residual, width, scalar, next, next_weight, failure,
                           prefetch_norm(), bounded_search(), interval_search());
    else if (stage_norm(width))
        hipLaunchKernelGGL((norm_residual_kernel<false, true>), dim3(rows), dim3(256), width * sizeof(float),
                           static_cast<hipStream_t>(q), branch, w, residual, width, scalar, next, next_weight, failure,
                           prefetch_norm(), bounded_search(), interval_search());
    else if (parallel)
        hipLaunchKernelGGL((norm_residual_kernel<true>), dim3(rows), dim3(256), 0, static_cast<hipStream_t>(q), branch,
                           w, residual, width, scalar, next, next_weight, failure, prefetch_norm(),
                           bounded_search(), interval_search());
    else
        hipLaunchKernelGGL((norm_residual_kernel<false>), dim3(rows), dim3(256), 0, static_cast<hipStream_t>(q), branch,
                           w, residual, width, scalar, next, next_weight, failure, prefetch_norm(),
                           bounded_search(), interval_search());
    launch_checked();
    return false;
}
bool norm_residual_rows(const float* branch, LinearView w, float* residual, unsigned rows, unsigned width, LinearView scalar,
                        float* next, LinearView next_weight, unsigned* failure, Queue q) {
    if (!rows || !width || !w.data || w.outputs != 1 || w.inputs != width || !failure ||
        (next && (!next_weight.data || next_weight.outputs != 1 || next_weight.inputs != width)))
        throw std::runtime_error("device norm residual rows geometry");
    // The schedule norm() selects for these widths: grouped, staged and parallel.
    static const bool grouped = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_GROUP");
        return !p || std::string_view(p) != "0";
    }();
    if (!grouped || !parallel_search() || !stage_norm(width))
        return false;
    hipLaunchKernelGGL((norm_residual_kernel<true, true>), dim3(rows), dim3(256), width * sizeof(float),
                       static_cast<hipStream_t>(q), branch, w, residual, width, scalar, next, next_weight, failure,
                       prefetch_norm(), bounded_search(), interval_search());
    launch_checked();
    return true;
}
void gelu_product_finite(float* g, const float* u, unsigned n, unsigned* failure, Queue q) {
    hipLaunchKernelGGL(product_finite_kernel, dim3((n + 255U) / 256U), dim3(256), 0, static_cast<hipStream_t>(q), g, u, n,
                       failure);
    launch_checked();
}
bool ffn_ple_gate(const float* branch, LinearView w, float* residual, LinearView gate, const float* ple,
                  unsigned product_stride, unsigned rows, unsigned width, float* out, unsigned* failure,
                  unsigned* ticket, Queue q) {
    valid_rows(rows);
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_FFN_PLE_GATE");
        const auto* fast = std::getenv("STRIX_G4_FAST_EPILOGUE");
        const auto* early = std::getenv("STRIX_G4_EPILOGUE_EARLY_LOADS");
        const auto* header = std::getenv("STRIX_G4_PACKED_HEADER");
        const auto* chunks = std::getenv("STRIX_G4_CHUNK_DECODE");
        const auto off = [](const char* v) { return v && std::string_view(v) == "0"; };
        return !off(p) && !off(fast) && !off(header) && !off(chunks) && !(early && std::string_view(early) == "1");
    }();
    static const bool hardware_half = [] {
        const auto* p = std::getenv("STRIX_G4_HARDWARE_HALF");
        return !p || std::string_view(p) != "0";
    }();
    static const bool parallel = [] {
        const auto* p = std::getenv("STRIX_G4_NORM_SEARCH");
        return !p || std::string_view(p) != "0";
    }();
    const auto format = norm_weights(w);
    if (!enabled || !parallel || !stage_norm(width) || rows > 2U || width != 2560U || !w.data || w.outputs != 1U ||
        w.inputs != width || format == NormWeights::any || !failure || !ticket || !gate.data || gate.inputs != width ||
        !gate.outputs || stream_weights(gate.format, rows) ||
        overlaps(branch, residual, std::size_t(rows) * width) ||
        overlaps(out, residual, std::size_t(rows) * width) || overlaps(out, branch, std::size_t(rows) * width))
        return false;
    bool launched = false;
    if (format == NormWeights::f32)
        launched = rows == 1U ? launch_ffn_ple_gate_format<NormWeights::f32, 1>(branch, w, residual, gate, ple,
                                                                               product_stride, out, failure, ticket,
                                                                               hardware_half, q)
                              : launch_ffn_ple_gate_format<NormWeights::f32, 2>(branch, w, residual, gate, ple,
                                                                               product_stride, out, failure, ticket,
                                                                               hardware_half, q);
    else
        launched = rows == 1U ? launch_ffn_ple_gate_format<NormWeights::bf16, 1>(branch, w, residual, gate, ple,
                                                                                product_stride, out, failure, ticket,
                                                                                hardware_half, q)
                              : launch_ffn_ple_gate_format<NormWeights::bf16, 2>(branch, w, residual, gate, ple,
                                                                                product_stride, out, failure, ticket,
                                                                                hardware_half, q);
    if (launched)
        launch_checked();
    return launched;
}
void embedding(LinearView t, const unsigned* tokens, unsigned rows, float* out, float scale, Queue q) {
    valid_rows(rows);
    hipLaunchKernelGGL(embedding_kernel, dim3((t.inputs + 255U) / 256U, rows), dim3(256), 0,
                       static_cast<hipStream_t>(q), t, tokens, out, scale);
    launch_checked();
}
void ple_combine(float* p, LinearView t, LinearView w, const unsigned* tokens, unsigned rows, unsigned layer_count,
                 unsigned width, Queue q, unsigned* failure) {
    valid_rows(rows);
    if (!layer_count || !width || t.inputs != layer_count * width || w.inputs != width)
        throw std::runtime_error("device PLE geometry");
    hipLaunchKernelGGL(ple_kernel, dim3(rows * layer_count), dim3(32), 0, static_cast<hipStream_t>(q), p, t, w, tokens,
                       layer_count, width, prefetch_norm(), parallel_search(), failure, bounded_search(), interval_search());
    launch_checked();
}
void ple_gate(float* g, const float* p, unsigned rows, unsigned layer_count, unsigned layer, unsigned width, Queue q,
              unsigned* failure) {
    valid_rows(rows);
    if (layer >= layer_count || !width)
        throw std::runtime_error("device PLE gate geometry");
    hipLaunchKernelGGL(ple_gate_kernel, dim3((width + 255U) / 256U, rows), dim3(256), 0, static_cast<hipStream_t>(q), g,
                       p, layer_count, layer, width, failure);
    launch_checked();
}
void residual(float* x, const float* b, unsigned n, LinearView s, Queue q) {
    hipLaunchKernelGGL(residual_kernel, dim3((n + 255U) / 256U), dim3(256), 0, static_cast<hipStream_t>(q), x, b, n, s);
    launch_checked();
}
void gelu_product(float* g, const float* u, unsigned n, Queue q) {
    hipLaunchKernelGGL(product_kernel, dim3((n + 255U) / 256U), dim3(256), 0, static_cast<hipStream_t>(q), g, u, n);
    launch_checked();
}
void cap_logits(float* x, unsigned n, Queue q) {
    hipLaunchKernelGGL(cap_kernel, dim3((n + 255U) / 256U), dim3(256), 0, static_cast<hipStream_t>(q), x, n);
    launch_checked();
}
void check_finite(const float* x, std::size_t n, unsigned* failure, Queue q) {
    if (n) {
        hipLaunchKernelGGL(finite_kernel, dim3(unsigned((n + 255) / 256)), dim3(256), 0, static_cast<hipStream_t>(q), x,
                           n, failure);
        launch_checked();
    }
}
void argmax(const float* x, unsigned rows, unsigned vocabulary, Candidate* scratch, Candidate* result, Queue q) {
    valid_rows(rows);
    if (!vocabulary)
        throw std::runtime_error("empty device vocabulary");
    static const bool parallel = [] {
        const auto* p = std::getenv("STRIX_G4_FAST_ARGMAX");
        return !p || std::string_view(p) != "0";
    }();
    if (parallel) {
        const auto groups = (vocabulary + 2047U) / 2048U;
        hipLaunchKernelGGL(block_candidate_kernel, dim3(groups, rows), dim3(256), 0, static_cast<hipStream_t>(q), x,
                           vocabulary, scratch);
        launch_checked();
        hipLaunchKernelGGL(block_merge_kernel, dim3(rows), dim3(256), 0, static_cast<hipStream_t>(q), scratch, groups,
                           result);
        launch_checked();
        return;
    }
    const auto blocks = (vocabulary + 255U) / 256U;
    hipLaunchKernelGGL(candidate_kernel, dim3(blocks, rows), dim3(32), 0, static_cast<hipStream_t>(q), x, vocabulary,
                       scratch);
    launch_checked();
    hipLaunchKernelGGL(merge_kernel, dim3(rows), dim3(32), 0, static_cast<hipStream_t>(q), scratch, blocks, result);
    launch_checked();
}
} // namespace strix::gemma4::gpu
