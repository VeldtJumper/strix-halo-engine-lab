#include <hip/hip_runtime.h>
#include "fp16_mirror.hpp"
#include "kv8_device.hpp"
#include "gemma4/kernels/internal.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace strix::gemma4::gpu {
namespace {
// f16: one thread per 4 elements of [rows][width] of K and V: halves of committed global rows [from, from + rows).
__global__ void mirror_kernel(const float* k, const float* v, const std::uint64_t* tags, std::uint64_t from,
                              unsigned rows, unsigned width, _Float16* hk, _Float16* hv, unsigned* failure) {
    const auto index = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4U;
    if (index >= std::size_t(rows) * width)
        return;
    const auto p = from + index / width;
    if (index % width == 0 && tags[p] != p)
        atomicOr(failure, 2U);
    const auto at = std::size_t(p) * width + index % width;
    const float4 a = *reinterpret_cast<const float4*>(k + at), b = *reinterpret_cast<const float4*>(v + at);
    const float ka[4] = {a.x, a.y, a.z, a.w}, va[4] = {b.x, b.y, b.z, b.w};
    bool overflow = false;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const auto x = static_cast<_Float16>(ka[j]), y = static_cast<_Float16>(va[j]);
        overflow |= (__builtin_bit_cast(unsigned short, x) & 0x7c00U) == 0x7c00U ||
                    (__builtin_bit_cast(unsigned short, y) & 0x7c00U) == 0x7c00U;
        hk[at + j] = x;
        hv[at + j] = y;
    }
    if (overflow)
        atomicOr(failure, 4U);
}
// kv8: one thread per group of kv8_group elements of K or V (blockIdx.y: 0 = K, 1 = V).
__global__ void mirror_kv8_kernel(const float* k, const float* v, const std::uint64_t* tags, std::uint64_t from,
                                  unsigned rows, unsigned width, std::int8_t* ck, std::int8_t* cv,
                                  std::uint16_t* sk, std::uint16_t* sv, unsigned* failure) {
    const auto group_index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned groups = width / kv8_group;
    if (group_index >= std::size_t(rows) * groups)
        return;
    const bool value = blockIdx.y != 0;
    const auto p = from + group_index / groups;
    if (!value && group_index % groups == 0 && tags[p] != p)
        atomicOr(failure, 2U);
    const auto at = std::size_t(p) * width + (group_index % groups) * kv8_group;
    const float* src = (value ? v : k) + at;
    float x[kv8_group];
    float amax = 0.0F;
#pragma unroll
    for (unsigned j = 0; j < kv8_group; j += 4) {
        const float4 f = *reinterpret_cast<const float4*>(src + j);
        x[j] = f.x; x[j + 1] = f.y; x[j + 2] = f.z; x[j + 3] = f.w;
    }
#pragma unroll
    for (unsigned j = 0; j < kv8_group; ++j) {
        kv8_finite(x[j], failure);
        const float a = kv8_abs(x[j]);
        amax = a > amax ? a : amax;
    }
    const auto bits = kv8_scale_bits(amax, failure);
    const float d = kv8_scale_value(bits);
    std::int8_t* codes = (value ? cv : ck) + at;
#pragma unroll
    for (unsigned j = 0; j < kv8_group; ++j)
        codes[j] = std::int8_t(kv8_code(x[j], d));
    (value ? sv : sk)[std::size_t(p) * groups + group_index % groups] = bits;
}
} // namespace
void fp16_mirror_sync(DeviceStream::Impl& stream, MirrorFormat format, std::uint32_t* failure, Queue queue) {
    if (format == MirrorFormat::none)
        throw std::invalid_argument("mirror format");
    const auto upto = stream.frontier / fp16_mirror_step * fp16_mirror_step;
    if (!stream.fp16_mirror) {
        if (!upto)
            return;
        auto mirror = std::make_shared<Fp16Mirror>();
        mirror->format = format;
        mirror->layer.resize(stream.cache.size());
        for (std::size_t l = 0; l < stream.cache.size(); ++l) {
            const auto& c = stream.cache[l];
            if (!c.width || !c.global || c.width % kv8_group)
                continue;
            auto& m = mirror->layer[l];
            const auto elements = 2ULL * c.slots * c.width;
            if (format == MirrorFormat::f16)
                m.values.resize(elements * sizeof(_Float16));
            else {
                m.values.resize(elements);
                m.scales.resize(elements / kv8_group * sizeof(std::uint16_t));
            }
            m.width = c.width;
            m.slots = c.slots;
        }
        stream.fp16_mirror = std::move(mirror);
    }
    if (stream.fp16_mirror->format != format)
        throw std::logic_error("stream mirror already holds another format");
    for (std::size_t l = 0; l < stream.cache.size(); ++l) {
        auto& m = stream.fp16_mirror->layer[l];
        m.pending = m.mirrored;
        if (!m.width || m.mirrored >= upto)
            continue;
        const auto& c = stream.cache[l];
        const auto rows = unsigned(upto - m.mirrored);
        const auto plane = std::size_t(m.slots) * m.width;
        const auto stream_q = static_cast<hipStream_t>(queue);
        if (format == MirrorFormat::f16) {
            const auto blocks = (std::size_t(rows) * m.width / 4U + 255U) / 256U;
            auto* halves = m.values.as<_Float16>();
            hipLaunchKernelGGL(mirror_kernel, dim3(unsigned(blocks)), dim3(256), 0, stream_q, c.k.as<float>(),
                               c.v.as<float>(), c.tags.as<std::uint64_t>(), m.mirrored, rows, m.width, halves,
                               halves + plane, failure);
        } else {
            const auto blocks = (std::size_t(rows) * (m.width / kv8_group) + 255U) / 256U;
            auto* codes = m.values.as<std::int8_t>();
            auto* scales = m.scales.as<std::uint16_t>();
            hipLaunchKernelGGL(mirror_kv8_kernel, dim3(unsigned(blocks), 2), dim3(256), 0, stream_q, c.k.as<float>(),
                               c.v.as<float>(), c.tags.as<std::uint64_t>(), m.mirrored, rows, m.width, codes,
                               codes + plane, scales, scales + plane / kv8_group, failure);
        }
        launch_checked();
        m.pending = upto;
    }
}
void fp16_mirror_confirm(DeviceStream::Impl& stream) {
    if (stream.fp16_mirror)
        for (auto& m : stream.fp16_mirror->layer)
            m.mirrored = std::max(m.mirrored, m.pending);
}
Fp16MirrorRead fp16_mirror_read(const DeviceStream::Impl& stream, std::uint32_t producer) {
    if (!stream.fp16_mirror || producer >= stream.fp16_mirror->layer.size())
        return {};
    const auto& m = stream.fp16_mirror->layer[producer];
    if (!m.width)
        return {};
    const auto plane = std::size_t(m.slots) * m.width;
    Fp16MirrorRead out;
    out.mirrored = std::max(m.mirrored, m.pending);
    out.width = m.width;
    out.format = std::uint32_t(stream.fp16_mirror->format);
    if (stream.fp16_mirror->format == MirrorFormat::f16) {
        out.k = m.values.as<std::uint16_t>();
        out.v = m.values.as<std::uint16_t>() + plane;
    } else {
        out.k = m.values.as<std::int8_t>();
        out.v = m.values.as<std::int8_t>() + plane;
        out.k_scale = m.scales.as<std::uint16_t>();
        out.v_scale = m.scales.as<std::uint16_t>() + plane / kv8_group;
    }
    return out;
}
} // namespace strix::gemma4::gpu
