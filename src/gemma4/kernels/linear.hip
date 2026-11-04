#include "internal.hpp"
#include "gemma4/exl3/device.hpp"
#include "chunk.hpp"
#include "weight_policy.hpp"
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::gpu {
namespace {
__device__ void store_result(float* out, std::size_t index, float value, unsigned* failure) {
    out[index] = value;
    if (failure)
        finite_flag(value, failure);
}
__device__ void store_fused(float* out, unsigned row, unsigned o, unsigned outputs, float value,
                            const LinearEpilogue& ep, unsigned* failure) {
    if (ep.product) {
        if (failure)
            finite_flag(value, failure);
        value = gelu_tanh(value) * ep.product[std::size_t(row) * ep.product_stride + o];
    } else if (ep.cap) {
        if (failure)
            finite_flag(value, failure);
        value = softcap(value);
    }
    store_result(out, std::size_t(row) * outputs + o, value, failure);
}
__device__ inline unsigned select_matrix(const LinearEpilogue& ep, LinearView& w, float*& out) {
    unsigned block = blockIdx.x;
    if (ep.extra_out[0] && block >= ep.extra_block[0]) {
        const bool second = ep.extra_out[1] && block >= ep.extra_block[1];
        w = second ? ep.extra[1] : ep.extra[0];
        out = second ? ep.extra_out[1] : ep.extra_out[0];
        block -= second ? ep.extra_block[1] : ep.extra_block[0];
    }
    return block;
}
inline unsigned grid_blocks(LinearView w, const LinearEpilogue& ep, unsigned per_block = 4U) {
    if (!ep.extra_out[0])
        return (w.outputs + per_block - 1U) / per_block;
    // Every matrix of a shared grid uses whole blocks.
    unsigned blocks = (w.outputs + per_block - 1U) / per_block;
    for (unsigned t = 0; t < 2 && ep.extra_out[t]; ++t)
        blocks += (ep.extra[t].outputs + per_block - 1U) / per_block;
    return blocks;
}
// One wave/output, with one lane accumulator per row. Weight decode is outside the row loop.
template <unsigned Rows, bool Gate>
__global__ void linear_kernel(LinearView w, LinearView up, const float* x, float* out, unsigned* failure) {
    const unsigned o = blockIdx.x * 4U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (o >= w.outputs)
        return;
    float acc[Rows] = {}, uacc[Rows] = {};
    for (unsigned c = lane; c < (w.inputs + 15U) / 16U; c += 32U) {
        for (unsigned e = 0; e < 16; ++e) {
            const auto i = c * 16U + e;
            const float a = i < w.inputs ? coefficient(w, o, i) : 0.0F;
            float b = 0.0F;
            if constexpr (Gate)
                b = i < w.inputs ? coefficient(up, o, i) : 0.0F;
            for (unsigned r = 0; r < Rows; ++r) {
                // Padding is an explicit +0, as in the shared host recipe.
                acc[r] = acc[r] + (i < w.inputs ? a * x[std::size_t(r) * w.inputs + i] : 0.0F);
                if constexpr (Gate)
                    uacc[r] = uacc[r] + (i < w.inputs ? b * x[std::size_t(r) * w.inputs + i] : 0.0F);
            }
        }
    }
    for (unsigned r = 0; r < Rows; ++r) {
        const float a = wave_sum(acc[r]);
        float b = 0.0F;
        if constexpr (Gate)
            b = wave_sum(uacc[r]);
        if (lane == 0)
            store_result(out, std::size_t(r) * w.outputs + o, Gate ? gelu_tanh(a) * b : a, failure);
    }
}
template <unsigned Rows, bool Gate, WeightFormat Format, bool WideHeader, bool Streaming>
__global__ void chunk_linear_kernel(LinearView w, LinearView up, const float* x, float* out, unsigned* failure,
                                    LinearEpilogue ep) {
    const unsigned o = select_matrix(ep, w, out) * 4U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (o < w.outputs) {
    const auto* wp = w.data + std::size_t(o) * w.row_bytes;
    const auto* uptr = Gate ? up.data + std::size_t(o) * up.row_bytes : wp;
    float acc[Rows] = {}, uacc[Rows] = {};
    for (unsigned c = lane; c < (w.inputs + 15U) / 16U; c += 32U) {
        const auto start = c * 16U;
        if (start + 16U <= w.inputs) {
            const WeightChunk<Format, Streaming> a(wp, start, WideHeader, ep.hardware_half),
                b(uptr, start, WideHeader, ep.hardware_half);
#pragma unroll
            for (unsigned e = 0; e < 16; ++e) {
                const float v = a.at(e);
                float u = 0.0F;
                if constexpr (Gate)
                    u = b.at(e);
#pragma unroll
                for (unsigned r = 0; r < Rows; ++r) {
                    const float input = x[std::size_t(r) * w.inputs + start + e];
                    acc[r] = acc[r] + v * input;
                    if constexpr (Gate)
                        uacc[r] = uacc[r] + u * input;
                }
            }
        } else {
            // No vector load may read the padding beyond the last stored element.
            for (unsigned e = 0; e < 16; ++e) {
                const auto i = start + e;
                const float a = i < w.inputs ? coefficient(w, o, i) : 0.0F;
                const float b = Gate && i < w.inputs ? coefficient(up, o, i) : 0.0F;
                for (unsigned r = 0; r < Rows; ++r) {
                    acc[r] = acc[r] + (i < w.inputs ? a * x[std::size_t(r) * w.inputs + i] : 0.0F);
                    if constexpr (Gate)
                        uacc[r] = uacc[r] + (i < w.inputs ? b * x[std::size_t(r) * w.inputs + i] : 0.0F);
                }
            }
        }
    }
    for (unsigned r = 0; r < Rows; ++r) {
        const float a = wave_sum(acc[r]);
        float b = 0.0F;
        if constexpr (Gate)
            b = wave_sum(uacc[r]);
        if (lane == 0)
            store_fused(out, r, o, w.outputs, Gate ? gelu_tanh(a) * b : a, ep, failure);
    }
    }
}
// Identical visits and arithmetic; the next lane chunk's weights are loaded
// before the current chunk is accumulated, to overlap load latency with work.
template <unsigned Rows, bool Gate, WeightFormat Format, bool WideHeader, bool Streaming>
__global__ void __launch_bounds__(128) pipelined_linear_kernel(LinearView w, LinearView up, const float* x, float* out,
                                                               unsigned* failure,
                                        LinearEpilogue ep) {
    const unsigned o = select_matrix(ep, w, out) * 4U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (o < w.outputs) {
        const auto* wp = w.data + std::size_t(o) * w.row_bytes;
        const auto* uptr = Gate ? up.data + std::size_t(o) * up.row_bytes : wp;
        float acc[Rows] = {}, uacc[Rows] = {};
        const auto chunks = w.inputs / 16U;
        WeightChunk<Format, Streaming> a, b;
        if (lane < chunks) {
            a = WeightChunk<Format, Streaming>(wp, lane * 16U, WideHeader, ep.hardware_half);
            if constexpr (Gate)
                b = WeightChunk<Format, Streaming>(uptr, lane * 16U, WideHeader, ep.hardware_half);
        }
        for (unsigned c = lane; c < chunks; c += 32U) {
            WeightChunk<Format, Streaming> na, nb;
            if (c + 32U < chunks) {
                na = WeightChunk<Format, Streaming>(wp, (c + 32U) * 16U, WideHeader, ep.hardware_half);
                if constexpr (Gate)
                    nb = WeightChunk<Format, Streaming>(uptr, (c + 32U) * 16U, WideHeader, ep.hardware_half);
            }
            const auto start = c * 16U;
#pragma unroll
            for (unsigned e = 0; e < 16; ++e) {
                const float v = a.at(e);
                float u = 0.0F;
                if constexpr (Gate)
                    u = b.at(e);
#pragma unroll
                for (unsigned r = 0; r < Rows; ++r) {
                    const float input = x[std::size_t(r) * w.inputs + start + e];
                    acc[r] = acc[r] + v * input;
                    if constexpr (Gate)
                        uacc[r] = uacc[r] + u * input;
                }
            }
            a = na;
            if constexpr (Gate)
                b = nb;
        }
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const float sa = wave_sum(acc[r]);
            float sb = 0.0F;
            if constexpr (Gate)
                sb = wave_sum(uacc[r]);
            if (lane == 0)
                store_fused(out, r, o, w.outputs, Gate ? gelu_tanh(sa) * sb : sa, ep, failure);
        }
    }
}
// Two outputs per wave share every input load. Inputs are read four elements
// at a time; each (output,row,lane) accumulator still visits its chunks and
// their elements in ascending order from +0, exactly as the one-output kernel.
template <unsigned Rows, bool Gate, WeightFormat Format, bool WideHeader, bool Streaming, unsigned T = 2>
__global__ void __launch_bounds__(128) pair_linear_kernel(LinearView w, LinearView up, const float* x, float* out,
                                                          unsigned* failure,
                                   LinearEpilogue ep) {
    const unsigned first = (select_matrix(ep, w, out) * 4U + threadIdx.x / 32U) * T, lane = threadIdx.x % 32U;
    if (first < w.outputs) {
        const auto chunks = w.inputs / 16U;
        const std::uint8_t* wp[T];
        const std::uint8_t* uptr[T];
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
            // A tail wave reuses a valid row for its unused loads; it never stores it.
            const auto o = min(first + t, w.outputs - 1U);
            wp[t] = w.data + std::size_t(o) * w.row_bytes;
            uptr[t] = Gate ? up.data + std::size_t(o) * up.row_bytes : wp[t];
        }
        float acc[T][Rows] = {}, uacc[T][Rows] = {};
        for (unsigned c = lane; c < chunks; c += 32U) {
            const auto start = c * 16U;
            WeightChunk<Format, Streaming> a[T], b[T];
#pragma unroll
            for (unsigned t = 0; t < T; ++t) {
                a[t] = WeightChunk<Format, Streaming>(wp[t], start, WideHeader, ep.hardware_half);
                if constexpr (Gate)
                    b[t] = WeightChunk<Format, Streaming>(uptr[t], start, WideHeader, ep.hardware_half);
            }
#pragma unroll
            for (unsigned g = 0; g < 4; ++g) {
                float4 input[Rows];
#pragma unroll
                for (unsigned r = 0; r < Rows; ++r)
                    input[r] = *reinterpret_cast<const float4*>(x + std::size_t(r) * w.inputs + start + 4U * g);
#pragma unroll
                for (unsigned k = 0; k < 4; ++k) {
                    const unsigned e = 4U * g + k;
#pragma unroll
                    for (unsigned t = 0; t < T; ++t) {
                        const float v = a[t].at(e);
                        float u = 0.0F;
                        if constexpr (Gate)
                            u = b[t].at(e);
#pragma unroll
                        for (unsigned r = 0; r < Rows; ++r) {
                            const float in = k == 0   ? input[r].x
                                             : k == 1 ? input[r].y
                                             : k == 2 ? input[r].z
                                                      : input[r].w;
                            acc[t][r] = acc[t][r] + v * in;
                            if constexpr (Gate)
                                uacc[t][r] = uacc[t][r] + u * in;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                const float sa = wave_sum(acc[t][r]);
                float sb = 0.0F;
                if constexpr (Gate)
                    sb = wave_sum(uacc[t][r]);
                if (lane == 0 && first + t < w.outputs)
                    store_fused(out, r, first + t, w.outputs, Gate ? gelu_tanh(sa) * sb : sa, ep, failure);
            }
        }
    }
}
template <WeightFormat Format, bool WideHeader, bool Streaming>
__global__ void cached_input_kernel(LinearView w, const float* x, float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x % 32U, chunks = w.inputs / 16U;
    extern __shared__ float4 input[];
    for (unsigned c = threadIdx.x; c < chunks; c += 512U) {
        float values[16];
#pragma unroll
        for (unsigned e = 0; e < 16; ++e)
            values[e] = x[c * 16U + e];
#pragma unroll
        for (unsigned g = 0; g < 4; ++g)
            input[g * chunks + c] = make_float4(values[4U * g], values[4U * g + 1U],
                                                values[4U * g + 2U], values[4U * g + 3U]);
    }
    __syncthreads();
    // All waves take part in the copy, including unused output waves at the tail.
    const unsigned o = blockIdx.x * 16U + threadIdx.x / 32U;
    if (o >= w.outputs)
        return;
    const auto* wp = w.data + std::size_t(o) * w.row_bytes;
    float acc = 0.0F;
    for (unsigned c = lane; c < chunks; c += 32U) {
        const WeightChunk<Format, Streaming> a(wp, c * 16U, WideHeader);
        float values[16];
#pragma unroll
        for (unsigned g = 0; g < 4; ++g) {
            const auto v = input[g * chunks + c];
            values[4U * g] = v.x;
            values[4U * g + 1U] = v.y;
            values[4U * g + 2U] = v.z;
            values[4U * g + 3U] = v.w;
        }
#pragma unroll
        for (unsigned e = 0; e < 16; ++e)
            acc = acc + a.at(e) * values[e];
    }
    const float value = wave_sum(acc);
    if (lane == 0)
        store_result(out, o, value, failure);
}
// Independent outputs give the scheduler multiple load/accumulate chains. Every
// (output,row,lane) still visits exactly the same chunks and elements in order.
template <unsigned Rows, bool Gate, WeightFormat Format, bool WideHeader, bool Streaming>
__global__ void tiled_linear_kernel(LinearView w, LinearView up, const float* x, float* out, unsigned* failure) {
    constexpr unsigned Outputs = Rows == 1 && !Gate ? 4U : 2U;
    const unsigned first = (blockIdx.x * 4U + threadIdx.x / 32U) * Outputs, lane = threadIdx.x % 32U;
    if (first >= w.outputs)
        return;
    float acc[Outputs][Rows] = {}, uacc[Outputs][Rows] = {};
    for (unsigned c = lane; c < w.inputs / 16U; c += 32U) {
        const auto start = c * 16U;
        WeightChunk<Format, Streaming> a[Outputs], b[Outputs];
#pragma unroll
        for (unsigned t = 0; t < Outputs; ++t) {
            // A tail wave uses a valid row for its unused loads; it never publishes it.
            const auto o = min(first + t, w.outputs - 1U);
            a[t] = WeightChunk<Format, Streaming>(w.data + std::size_t(o) * w.row_bytes, start, WideHeader);
            if constexpr (Gate)
                b[t] = WeightChunk<Format, Streaming>(up.data + std::size_t(o) * up.row_bytes, start, WideHeader);
        }
#pragma unroll
        for (unsigned e = 0; e < 16; ++e) {
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                const float input = x[std::size_t(r) * w.inputs + start + e];
#pragma unroll
                for (unsigned t = 0; t < Outputs; ++t) {
                    acc[t][r] = acc[t][r] + a[t].at(e) * input;
                    if constexpr (Gate)
                        uacc[t][r] = uacc[t][r] + b[t].at(e) * input;
                }
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < Outputs; ++t) {
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const float a = wave_sum(acc[t][r]);
            float b = 0.0F;
            if constexpr (Gate)
                b = wave_sum(uacc[t][r]);
            if (lane == 0 && first + t < w.outputs)
                store_result(out, std::size_t(r) * w.outputs + first + t, Gate ? gelu_tanh(a) * b : a, failure);
        }
    }
}
template <unsigned Rows, bool Gate, WeightFormat Format, bool WideHeader, bool Streaming>
__global__ void prefetched_linear_kernel(LinearView w, LinearView up, const float* x, float* out, unsigned* failure,
                                         LinearEpilogue ep) {
    const unsigned o = select_matrix(ep, w, out) * 4U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (o < w.outputs) {
    const auto* wp = w.data + std::size_t(o) * w.row_bytes;
    const auto* uptr = Gate ? up.data + std::size_t(o) * up.row_bytes : wp;
    float acc[Rows] = {}, uacc[Rows] = {};
    const auto chunks = w.inputs / 16U;
    for (unsigned c = lane; c < chunks; c += 64U) {
        WeightChunk<Format, Streaming> a[2], b[2];
        // Preload successive lane chunks, retaining the original output grid.
#pragma unroll
        for (unsigned t = 0; t < 2; ++t)
            if (c + 32U * t < chunks) {
                a[t] = WeightChunk<Format, Streaming>(wp, (c + 32U * t) * 16U, WideHeader, ep.hardware_half);
                if constexpr (Gate)
                    b[t] = WeightChunk<Format, Streaming>(uptr, (c + 32U * t) * 16U, WideHeader, ep.hardware_half);
            }
#pragma unroll
        for (unsigned t = 0; t < 2; ++t)
            if (c + 32U * t < chunks) {
                const auto start = (c + 32U * t) * 16U;
#pragma unroll
                for (unsigned e = 0; e < 16; ++e) {
                    const float v = a[t].at(e);
                    float u = 0.0F;
                    if constexpr (Gate)
                        u = b[t].at(e);
#pragma unroll
                    for (unsigned r = 0; r < Rows; ++r) {
                        const float input = x[std::size_t(r) * w.inputs + start + e];
                        acc[r] = acc[r] + v * input;
                        if constexpr (Gate)
                            uacc[r] = uacc[r] + u * input;
                    }
                }
            }
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
        const float a = wave_sum(acc[r]);
        float b = 0.0F;
        if constexpr (Gate)
            b = wave_sum(uacc[r]);
        if (lane == 0)
            store_fused(out, r, o, w.outputs, Gate ? gelu_tanh(a) * b : a, ep, failure);
    }
    }
}
// Several outputs per block share one copy of the inputs: each step the block
// stages the next 32 lane chunks of every row in LDS (contiguous float4 loads,
// double buffered), and the T outputs of a wave share every LDS read. Every
// (output,row,lane) accumulator visits the same chunks and elements in the same
// order from +0 with the same values, exactly as chunk_linear_kernel.
template <unsigned Rows, bool Gate, WeightFormat Format, bool WideHeader, bool Streaming, unsigned Waves, unsigned T>
__global__ void __launch_bounds__(Waves * 32U) staged_linear_kernel(LinearView w, LinearView up, const float* x,
                                                                     float* out, unsigned* failure, LinearEpilogue ep) {
    static_assert(T == 1U || T == 2U);
    __shared__ float4 staged[2][Rows * 4U * 32U];
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const unsigned first = select_matrix(ep, w, out) * (Waves * T) + wave * T;
    const bool valid = first < w.outputs;
    const auto chunks = w.inputs / 16U, steps = (chunks + 31U) / 32U;
    const auto stage = [&](unsigned step, unsigned buffer) {
        for (unsigned i = threadIdx.x; i < Rows * 128U; i += Waves * 32U) {
            const unsigned r = i / 128U, group = i % 128U, c = step * 32U + group / 4U;
            if (c < chunks)
                staged[buffer][(r * 4U + group % 4U) * 32U + group / 4U] =
                    *reinterpret_cast<const float4*>(x + std::size_t(r) * w.inputs + std::size_t(c) * 16U +
                                                     4U * (group % 4U));
        }
    };
    const std::uint8_t* wp[T];
    const std::uint8_t* uptr[T];
    if (valid) {
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
            // Reuse a valid weight row for unused tail loads; never store it.
            const auto o = min(first + t, w.outputs - 1U);
            wp[t] = w.data + std::size_t(o) * w.row_bytes;
            uptr[t] = Gate ? up.data + std::size_t(o) * up.row_bytes : wp[t];
        }
    }
    float acc[T][Rows] = {}, uacc[T][Rows] = {};
    stage(0U, 0U);
    __syncthreads();
    for (unsigned step = 0; step < steps; ++step) {
        if (step + 1U < steps)
            stage(step + 1U, (step + 1U) & 1U);
        const unsigned c = step * 32U + lane;
        if (valid && c < chunks) {
            const auto start = c * 16U;
            WeightChunk<Format, Streaming> a[T], b[T];
#pragma unroll
            for (unsigned t = 0; t < T; ++t) {
                a[t] = WeightChunk<Format, Streaming>(wp[t], start, WideHeader, ep.hardware_half);
                if constexpr (Gate)
                    b[t] = WeightChunk<Format, Streaming>(uptr[t], start, WideHeader, ep.hardware_half);
            }
            const float4* in = staged[step & 1U];
#pragma unroll
            for (unsigned g = 0; g < 4; ++g) {
                float4 input[Rows];
#pragma unroll
                for (unsigned r = 0; r < Rows; ++r)
                    input[r] = in[(r * 4U + g) * 32U + lane];
#pragma unroll
                for (unsigned k = 0; k < 4; ++k) {
                    const unsigned e = 4U * g + k;
#pragma unroll
                    for (unsigned t = 0; t < T; ++t) {
                        const float v = a[t].at(e);
                        float u = 0.0F;
                        if constexpr (Gate)
                            u = b[t].at(e);
#pragma unroll
                        for (unsigned r = 0; r < Rows; ++r) {
                            const float value = k == 0   ? input[r].x
                                                : k == 1 ? input[r].y
                                                : k == 2 ? input[r].z
                                                         : input[r].w;
                            acc[t][r] = acc[t][r] + v * value;
                            if constexpr (Gate)
                                uacc[t][r] = uacc[t][r] + u * value;
                        }
                    }
                }
            }
        }
        if (step + 1U < steps)
            __syncthreads();
    }
    if (valid) {
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r) {
                const float sa = wave_sum(acc[t][r]);
                float sb = 0.0F;
                if constexpr (Gate)
                    sb = wave_sum(uacc[t][r]);
                if (lane == 0 && first + t < w.outputs)
                    store_fused(out, r, first + t, w.outputs, Gate ? gelu_tanh(sa) * sb : sa, ep, failure);
            }
        }
    }
}
template <bool Gate, WeightFormat Format, bool WideHeader, bool Streaming>
bool chunk_dispatch_impl(LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure,
                         const LinearEpilogue& ep) {
    const bool fused = ep.product || ep.cap || ep.extra_out[0];
    static const bool tiled = [] {
        const auto* p = std::getenv("STRIX_G4_OUTPUT_TILE");
        return p && std::string_view(p) == "1";
    }();
    if (tiled && rows <= 4U && w.inputs % 16U == 0) {
        if (fused)
            return false;
#define G4_TILE_LAUNCH(R)                                                                                                  \
    case R:                                                                                                                \
        hipLaunchKernelGGL((tiled_linear_kernel<R, Gate, Format, WideHeader, Streaming>),                                  \
                           dim3((w.outputs + (R == 1 && !Gate ? 15U : 7U)) / (R == 1 && !Gate ? 16U : 8U)),                \
                           dim3(128), 0,                                                                                   \
                           static_cast<hipStream_t>(q), w, up, x, out, failure);                                           \
        break
        switch (rows) {
            G4_TILE_LAUNCH(1);
            G4_TILE_LAUNCH(2);
            G4_TILE_LAUNCH(3);
            G4_TILE_LAUNCH(4);
        }
#undef G4_TILE_LAUNCH
        return true;
    }
    if constexpr (!Gate) {
        static const bool cached_input = [] {
            const auto* p = std::getenv("STRIX_G4_INPUT_CACHE");
            return p && std::string_view(p) == "1";
        }();
        if (cached_input && rows == 1U && w.inputs >= 256U && w.inputs <= 4096U &&
            w.inputs % 16U == 0 && w.outputs <= 4096U) {
            if (fused)
                return false;
            hipLaunchKernelGGL((cached_input_kernel<Format, WideHeader, Streaming>), dim3((w.outputs + 15U) / 16U),
                               dim3(512), w.inputs * sizeof(float), static_cast<hipStream_t>(q), w, x, out, failure);
            return true;
        }
    }
    static const bool prefetch = [] {
        const auto* p = std::getenv("STRIX_G4_CHUNK_PREFETCH");
        return p ? std::string_view(p) == "1" : Format == WeightFormat::bf16;
    }();
    if (prefetch && rows == 1U && w.inputs >= 1024U && w.inputs % 16U == 0) {
        hipLaunchKernelGGL((prefetched_linear_kernel<1, Gate, Format, WideHeader, Streaming>), dim3(grid_blocks(w, ep)), dim3(128), 0,
                           static_cast<hipStream_t>(q), w, up, x, out, failure, ep);
        return true;
    }
    // The staged loads read x as float4, so x itself must be 16-byte aligned.
    if (staged_input_enabled(rows) && w.inputs % 16U == 0 && reinterpret_cast<std::uintptr_t>(x) % 16U == 0) {
        constexpr unsigned waves = 8U, per_block = waves * 2U;
        auto pe = ep;
        if (pe.extra_out[0]) {
            // Block offsets of a shared grid in units of per_block outputs.
            pe.extra_block[0] = (w.outputs + per_block - 1U) / per_block;
            if (pe.extra_out[1])
                pe.extra_block[1] = pe.extra_block[0] + (pe.extra[0].outputs + per_block - 1U) / per_block;
        }
#define G4_STAGED_LAUNCH(R, T)                                                                                             \
    case R:                                                                                                                \
        hipLaunchKernelGGL((staged_linear_kernel<R, Gate, Format, WideHeader, Streaming, waves, T>),                       \
                           dim3(grid_blocks(w, pe, per_block)), dim3(waves * 32U), 0, static_cast<hipStream_t>(q), w,      \
                           up, x, out, failure, pe);                                                                       \
        break
        switch (rows) {
            G4_STAGED_LAUNCH(3, 2);
            G4_STAGED_LAUNCH(4, 2);
            G4_STAGED_LAUNCH(5, 2);
            G4_STAGED_LAUNCH(6, 2);
            G4_STAGED_LAUNCH(7, 2);
            G4_STAGED_LAUNCH(8, 2);
        }
#undef G4_STAGED_LAUNCH
        return true;
    }
    if (pair_tile_enabled(rows) && w.inputs % 16U == 0) {
        // Four outputs per wave for single-matrix passes with at least four rows
        // (STRIX_G4_QUAD_TILE=1, opt-in until measured), otherwise two.
        static const bool quad = [] {
            const auto* p = std::getenv("STRIX_G4_QUAD_TILE");
            return p && std::string_view(p) == "1";
        }();
        const unsigned per_block = !Gate && quad && rows >= 4U ? 16U : 8U;
        auto pe = ep;
        if (pe.extra_out[0]) {
            // Block offsets of a shared grid in units of per_block outputs.
            pe.extra_block[0] = (w.outputs + per_block - 1U) / per_block;
            if (pe.extra_out[1])
                pe.extra_block[1] = pe.extra_block[0] + (pe.extra[0].outputs + per_block - 1U) / per_block;
        }
        if constexpr (!Gate) {
            if (per_block == 16U) {
#define G4_QUAD_LAUNCH(R)                                                                                                  \
    case R:                                                                                                                \
        hipLaunchKernelGGL((pair_linear_kernel<R, Gate, Format, WideHeader, Streaming, 4>),                                \
                           dim3(grid_blocks(w, pe, 16U)), dim3(128), 0, static_cast<hipStream_t>(q), w, up, x, out,        \
                           failure, pe);                                                                                   \
        break
                switch (rows) {
                    G4_QUAD_LAUNCH(4);
                    G4_QUAD_LAUNCH(5);
                    G4_QUAD_LAUNCH(6);
                    G4_QUAD_LAUNCH(7);
                    G4_QUAD_LAUNCH(8);
                }
#undef G4_QUAD_LAUNCH
                return true;
            }
        }
#define G4_PAIR_LAUNCH(R)                                                                                                  \
    case R:                                                                                                                \
        hipLaunchKernelGGL((pair_linear_kernel<R, Gate, Format, WideHeader, Streaming>), dim3(grid_blocks(w, pe, 8U)),     \
                           dim3(128), 0, static_cast<hipStream_t>(q), w, up, x, out, failure, pe);                         \
        break
        switch (rows) {
            G4_PAIR_LAUNCH(1);
            G4_PAIR_LAUNCH(2);
            G4_PAIR_LAUNCH(3);
            G4_PAIR_LAUNCH(4);
            G4_PAIR_LAUNCH(5);
            G4_PAIR_LAUNCH(6);
            G4_PAIR_LAUNCH(7);
            G4_PAIR_LAUNCH(8);
        }
#undef G4_PAIR_LAUNCH
        return true;
    }
    if (row_prefetch_enabled(rows) && w.inputs % 16U == 0) {
#define G4_PIPE_LAUNCH(R)                                                                                                  \
    case R:                                                                                                                \
        hipLaunchKernelGGL((pipelined_linear_kernel<R, Gate, Format, WideHeader, Streaming>), dim3(grid_blocks(w, ep)),     \
                           dim3(128), 0, static_cast<hipStream_t>(q), w, up, x, out, failure, ep);                         \
        break
        switch (rows) {
            G4_PIPE_LAUNCH(1);
            G4_PIPE_LAUNCH(2);
            G4_PIPE_LAUNCH(3);
            G4_PIPE_LAUNCH(4);
            G4_PIPE_LAUNCH(5);
            G4_PIPE_LAUNCH(6);
            G4_PIPE_LAUNCH(7);
            G4_PIPE_LAUNCH(8);
        }
#undef G4_PIPE_LAUNCH
        return true;
    }
#define G4_CHUNK_LAUNCH(R)                                                                                                 \
    case R:                                                                                                                \
        hipLaunchKernelGGL((chunk_linear_kernel<R, Gate, Format, WideHeader, Streaming>), dim3(grid_blocks(w, ep)), dim3(128), 0,\
                           static_cast<hipStream_t>(q), w, up, x, out, failure, ep);                                       \
        break
    switch (rows) {
        G4_CHUNK_LAUNCH(1);
        G4_CHUNK_LAUNCH(2);
        G4_CHUNK_LAUNCH(3);
        G4_CHUNK_LAUNCH(4);
        G4_CHUNK_LAUNCH(5);
        G4_CHUNK_LAUNCH(6);
        G4_CHUNK_LAUNCH(7);
        G4_CHUNK_LAUNCH(8);
    }
#undef G4_CHUNK_LAUNCH
    return true;
}
template <bool Gate, WeightFormat Format, bool Streaming>
bool chunk_dispatch_policy(LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure,
                           const LinearEpilogue& ep) {
    if constexpr (Format == WeightFormat::q4_k || Format == WeightFormat::q6_k) {
        static const bool wide_header = [] {
            const auto* p = std::getenv("STRIX_G4_PACKED_HEADER");
            return !p || std::string_view(p) != "0";
        }();
        if (wide_header)
            return chunk_dispatch_impl<Gate, Format, true, Streaming>(w, up, x, rows, out, q, failure, ep);
    }
    return chunk_dispatch_impl<Gate, Format, false, Streaming>(w, up, x, rows, out, q, failure, ep);
}
template <bool Gate, WeightFormat Format>
bool chunk_dispatch(LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure,
                    const LinearEpilogue& ep) {
    if (stream_weights(Format, rows))
        return chunk_dispatch_policy<Gate, Format, true>(w, up, x, rows, out, q, failure, ep);
    return chunk_dispatch_policy<Gate, Format, false>(w, up, x, rows, out, q, failure, ep);
}
template <bool Gate>
bool dispatch(LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure,
              const LinearEpilogue& epilogue = {}) {
    static const bool hardware_half = [] {
        const auto* p = std::getenv("STRIX_G4_HARDWARE_HALF");
        return !p || std::string_view(p) != "0";
    }();
    auto ep = epilogue;
    ep.hardware_half = hardware_half;
    static const bool chunks = [] {
        const auto* p = std::getenv("STRIX_G4_CHUNK_DECODE");
        return !p || std::string_view(p) != "0";
    }();
    if (chunks && (!Gate || w.format == up.format)) {
        bool launched = false;
        switch (w.format) {
#define G4_FORMAT(F) case WeightFormat::F: launched = chunk_dispatch<Gate, WeightFormat::F>(w, up, x, rows, out, q, failure, ep); break
            G4_FORMAT(bf16);
            G4_FORMAT(f32);
            G4_FORMAT(q8_0);
            G4_FORMAT(q4_k);
            G4_FORMAT(q6_k);
#undef G4_FORMAT
        default: break; // EXL3 packs have their own linears
        }
        if (launched)
            launch_checked();
        return launched;
    }
    if (ep.product || ep.cap || ep.extra_out[0])
        return false;
#define G4_LAUNCH(R)                                                                                                       \
    case R:                                                                                                                \
        hipLaunchKernelGGL((linear_kernel<R, Gate>), dim3((w.outputs + 3U) / 4U), dim3(128), 0,                            \
                           static_cast<hipStream_t>(q), w, up, x, out, failure);                                           \
        break
    switch (rows) {
        G4_LAUNCH(1);
        G4_LAUNCH(2);
        G4_LAUNCH(3);
        G4_LAUNCH(4);
        G4_LAUNCH(5);
        G4_LAUNCH(6);
        G4_LAUNCH(7);
        G4_LAUNCH(8);
    }
#undef G4_LAUNCH
    launch_checked();
    return true;
}
} // namespace
bool store_fusion_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_STORE_FUSION");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
// Unset or 1: shared passes of three to eight rows stage their inputs once per
// block (staged_linear_kernel, 8 waves x 2 outputs; measured faster than the
// chunk and pair kernels there, slower at two rows); 0 restores the previous
// schedules (chunk kernel, pair kernel at eight rows).
bool staged_input_enabled(unsigned rows) {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_STAGED_INPUT");
        return !p || std::string_view(p) != "0";
    }();
    return enabled && rows >= 3U && rows <= 8U;
}
// Unset: shared passes of eight rows (measured faster at C8, slower at C4);
// 0 off; 1 every row count.
bool pair_tile_enabled(unsigned rows) {
    static const int policy = [] {
        const auto* p = std::getenv("STRIX_G4_PAIR_TILE");
        return p ? (std::string_view(p) == "1" ? 1 : 0) : -1;
    }();
    return policy < 0 ? rows >= 8U : policy == 1;
}
// Opt-in (measured slower than the plain chunk kernel at C1, C2 and C4).
bool row_prefetch_enabled(unsigned) {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_ROW_PREFETCH");
        return p && std::string_view(p) == "1";
    }();
    return enabled;
}
bool linear_fused(LinearView w, const float* x, unsigned rows, float* out, const LinearEpilogue& epilogue, Queue q,
                  unsigned* failure) {
    // EXL3 linears use their own fused transform/core kernels; callers fall back.
    if (w.format == WeightFormat::exl3_mul1)
        return false;
    valid_rows(rows);
    auto ep = epilogue;
    if (ep.extra_out[0]) {
        if (ep.product || ep.cap)
            return false;
        ep.extra_block[0] = (w.outputs + 3U) / 4U;
        ep.blocks = ep.extra_block[0];
        for (unsigned t = 0; t < 2; ++t) {
            if (!ep.extra_out[t])
                break;
            const auto v = ep.extra[t];
            if (!v.data || v.format != w.format || v.inputs != w.inputs || !v.outputs)
                return false;
            ep.extra_block[t] = ep.blocks;
            ep.blocks += (v.outputs + 3U) / 4U;
        }
    }
    if (!w.data || !w.inputs || !w.outputs)
        throw std::runtime_error("empty device linear");
    if (!failure)
        return false;
    return dispatch<false>(w, {}, x, rows, out, q, failure, ep);
}
void linear(LinearView w, const float* x, unsigned rows, float* out, bool shared, Queue q, unsigned* failure, void* format_scratch) {
    if (w.format == WeightFormat::exl3_mul1) {
        exl3::device_linear(w, x, rows, out, format_scratch, failure, shared, q);
        return;
    }
    valid_rows(rows);
    if (!w.data || !w.inputs || !w.outputs)
        throw std::runtime_error("empty device linear");
    if (shared)
        (void)dispatch<false>(w, {}, x, rows, out, q, failure);
    else
        for (unsigned r = 0; r < rows; ++r)
            (void)dispatch<false>(w, {}, x + std::size_t(r) * w.inputs, 1, out + std::size_t(r) * w.outputs, q, failure);
}
void gate_up(LinearView gate, LinearView up, const float* x, unsigned rows, float* out, bool shared, Queue q,
             unsigned* failure) {
    valid_rows(rows);
    if (!gate.data || !up.data || gate.outputs != up.outputs || gate.inputs != up.inputs)
        throw std::runtime_error("gate/up geometry");
    if (shared)
        (void)dispatch<true>(gate, up, x, rows, out, q, failure);
    else
        for (unsigned r = 0; r < rows; ++r)
            (void)dispatch<true>(gate, up, x + std::size_t(r) * gate.inputs, 1, out + std::size_t(r) * gate.outputs, q, failure);
}
} // namespace strix::gemma4::gpu
