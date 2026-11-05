#include "device.hpp"
#include "a8_linear.hpp"
#include "gemma4/kernels/a8_ops.hpp"
#include "validation.hpp"
#include "gemma4/kernels/rows_linear.hpp"
#include "gemma4/kernels/capture_guard.hpp"
#include "gemma4/exl3/device.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <string_view>
#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cstring>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace strix::gemma4::prefill {
namespace {
// STRIX_G4_PREFILL_QUEUE_IO (default on, read per pass): every per-pass upload, clear and download of the
// many-row prefill runs on its own non-blocking queue (uploads staged in pinned memory), and scratch that grows
// with the context grows by doubling, so a pass issues no null-stream work, synchronous copy or frequent
// reallocation beside another thread's decode pass. Off: the previous synchronous buffer calls.
bool queue_io_enabled() {
    const auto* p = std::getenv("STRIX_G4_PREFILL_QUEUE_IO");
    return !p || std::string_view(p) != "0";
}
bool pack_reuse_enabled() {
    static const bool enabled = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_PACK_REUSE"); return !p || std::string_view(p) != "0"; }();
    return enabled;
}
bool qkv_share_enabled() {
    static const bool enabled = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_QKV_SHARE"); return !p || std::string_view(p) != "0"; }();
    return enabled;
}
using gpu::Buffer;
void checked(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
void launched() { checked(hipGetLastError()); }
// FUSE_EPILOGUE: the producer flags its own stored values (gpu::check_finite's
// rule); a null failure keeps the unfused kernel followed by check_finite.
__device__ inline void flag(float x, unsigned* failure) {
    if (failure && (bit_cast<std::uint32_t>(x) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
}
__global__ void gather(LinearView table, const unsigned* tokens, unsigned rows, float scale, float* out, unsigned* failure) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < std::size_t(rows) * table.inputs) {
        const float value = embedding_value(gpu::coefficient(table, tokens[i / table.inputs], unsigned(i % table.inputs)), scale);
        out[i] = value;
        flag(value, failure);
    }
}
__global__ void scale_ple(float* x, unsigned count) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) x[i] = x[i] * model_ple_scale;
}
__global__ void combine_ple(float* x, LinearView table, const unsigned* tokens, unsigned count, unsigned* failure) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float v = embedding_value(gpu::coefficient(table, tokens[i / table.inputs], unsigned(i % table.inputs)), ple_embedding_scale);
        const float value = (x[i] + v) * ple_combine_scale;
        x[i] = value;
        flag(value, failure);
    }
}
__global__ void rotate(float* x, const std::uint64_t* positions, const float* table, unsigned rows, unsigned heads_count,
                        unsigned dim, bool global, unsigned* failure) {
    const unsigned active = min(dim / 2U, global ? 64U : 128U);
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(rows) * heads_count * active) return;
    const auto rh = i / active, pair = i % active;
    const auto index = rh * dim + pair, t = std::size_t(positions[rh / heads_count]) * (global ? 512U : 256U) + pair * 2;
    const float a = x[index], b = x[index + dim / 2], cs = table[t], sn = table[t + 1];
    const float first = a * cs - b * sn, second = b * cs + a * sn;
    x[index] = first;
    x[index + dim / 2] = second;
    flag(first, failure); flag(second, failure);
}
__global__ void ple_product(float* g, const float* ple, unsigned count, unsigned width, unsigned layers_count, unsigned layer,
                            unsigned* failure) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = gelu_tanh(g[i]) * ple[(i / width * layers_count + layer) * width + i % width];
        g[i] = value;
        flag(value, failure);
    }
}
} // namespace
struct DevicePrefill::Impl {
    HostWeights w;
    std::vector<Buffer> storage;
    std::map<const std::uint8_t*, LinearView> uploaded;
    std::map<const std::uint8_t*, std::unique_ptr<PreparedLinear>> prepared;
    std::map<const std::uint8_t*, std::unique_ptr<A8Prepared>> a8_prepared;
    Buffer a8_scratch;
    Buffer sliding, global, residual, x, q, k, v, att, out, gate, up, ple, logits, taps;
    Buffer tokens, positions, reads, writes, scores, maxima, partials, failure, halves, maximum, attention_halves, global_halves, pair_halves, last_row;
    Buffer canonical_scores, canonical_denominators, canonical_packed, exl3_scratch;
    unsigned table_positions = 0, stride = 0;
    // QUEUE_IO: pinned staging for one pass's uploads (reused once the pass has synchronized).
    void* staging = nullptr;
    std::size_t staging_bytes = 0, staged = 0;
    bool queue_io = false;
    void put(Buffer& dst, const void* src, std::size_t bytes) {
        if (!queue_io) { dst.upload(src, bytes); return; }
        if (!bytes) return;
        if (bytes > dst.size() || staged + bytes > staging_bytes) throw std::logic_error("prefill staging bounds");
        auto* at = static_cast<std::uint8_t*>(staging) + staged;
        std::memcpy(at, src, bytes);
        staged += (bytes + 255U) & ~std::size_t(255U);
        checked(hipMemcpyAsync(dst.data(), at, bytes, hipMemcpyHostToDevice, queue));
    }
    void clear(Buffer& dst) {
        if (!queue_io) { dst.zero(); return; }
        if (dst.size()) checked(hipMemsetAsync(dst.data(), 0, dst.size(), queue));
    }
    void get(void* dst, const Buffer& src, std::size_t bytes, std::size_t offset = 0) {
        if (!queue_io) { src.download(dst, bytes, offset); return; }
        if (offset > src.size() || bytes > src.size() - offset) throw std::logic_error("prefill download bounds");
        if (!bytes) return;
        checked(hipMemcpyAsync(dst, src.as<std::uint8_t>() + offset, bytes, hipMemcpyDeviceToHost, queue));
        checked(hipStreamSynchronize(queue));
    }
    // Growth-only resize; with QUEUE_IO it at least doubles so reallocations stay rare.
    void grow(Buffer& buffer, std::size_t bytes) {
        if (bytes <= buffer.size()) return;
        buffer.resize(queue_io ? std::max(bytes, 2 * buffer.size()) : bytes);
    }
    std::vector<Range> ranges;
    // (slot, source): a pre-converted dot reports its producer's input maximum.
    std::vector<std::pair<unsigned, unsigned>> range_aliases;
    std::map<std::string, float> range_maxima;
    Stats stats;
    bool profile = false, fuse_ffn = false;
    hipEvent_t start = nullptr, stop = nullptr, converted = nullptr;
    hipStream_t queue = nullptr;
    // PINNED_TAPS: pinned host rows and the copy queue that fills them.
    hipStream_t copy_queue = nullptr;
    hipEvent_t tapped = nullptr, copied = nullptr;
    float* pinned_taps = nullptr;
    float* pinned_hidden = nullptr;
    std::size_t pinned_taps_floats = 0, pinned_hidden_floats = 0;
    // SIDE_STREAM: second queue with its own linear scratch; work moves there by
    // swapping queue and scratch, so every helper keeps its single-queue code.
    hipStream_t side_queue = nullptr;
    hipEvent_t forked = nullptr, joined = nullptr, ple_done = nullptr;
    Buffer side_halves, side_exl3, side_a8;
    // Every EXL3 scratch a chunk may need, sized before the side queue copies its size.
    void reserve_exl3(unsigned rows) {
        std::size_t need = 0;
        for (const auto& y : w.layer) {
            for (const auto& m : {y.q, y.k, y.v, y.o, y.gate, y.up, y.down, y.ple_gate, y.ple_projection})
                if (m.data && m.format == WeightFormat::exl3_mul1) need = std::max({need, exl3::device_scratch_bytes, exl3::rows_scratch_bytes(m, rows)});
            if (y.gate.format == WeightFormat::exl3_mul1 && y.up.format == WeightFormat::exl3_mul1 && y.down.format == WeightFormat::exl3_mul1)
                need = std::max(need, exl3::ffn_rows_scratch_bytes(y.gate, y.up, y.down, rows));
        }
        if (exl3_scratch.size() < need) exl3_scratch.resize(need);
    }
    void side_ready() {
        if (!side_queue) {
            checked(hipStreamCreateWithFlags(&side_queue, hipStreamNonBlocking));
            for (auto* e : {&forked, &joined, &ple_done}) checked(hipEventCreateWithFlags(e, hipEventDisableTiming));
        }
        if (side_halves.size() < halves.size()) side_halves.resize(halves.size());
        if (side_exl3.size() < exl3_scratch.size()) side_exl3.resize(exl3_scratch.size());
        if (side_a8.size() < a8_scratch.size()) side_a8.resize(a8_scratch.size());
    }
    // body runs on the side queue after the queue's last fork() mark; done marks its end.
    void fork() { checked(hipEventRecord(forked, queue)); }
    template <class F> void on_side(hipEvent_t done, F body) {
        checked(hipStreamWaitEvent(side_queue, forked, 0));
        const auto swap = [&] { std::swap(queue, side_queue); std::swap(halves, side_halves); std::swap(exl3_scratch, side_exl3); std::swap(a8_scratch, side_a8); };
        swap();
        try { body(); } catch (...) { swap(); throw; }
        swap();
        checked(hipEventRecord(done, side_queue));
    }
    bool split_panels = false;  // a hilo policy: conversion scratch holds high and low panels
    unsigned rows_capacity = 0; // rows every row-sized buffer holds (chunk_rows)
    void size_rows(unsigned rows) {
        const auto& c = w.config;
        const std::size_t hbytes = rows * std::size_t(c.hidden) * sizeof(float), abytes = rows * std::size_t(c.heads) * 512 * sizeof(float);
        for (auto* p : {&residual, &x, &out}) p->resize(hbytes);
        if (stats.linears == Linears::a8) a8_scratch.resize(a8_scratch_bytes(rows, 10752));
        for (auto* p : {&q, &att}) p->resize(abytes);
        for (auto* p : {&k, &v}) p->resize(rows * std::size_t(c.kv_heads) * 512 * sizeof(float));
        gate.resize(rows * std::size_t(std::max(c.inter, c.ple_dim)) * sizeof(float)); up.resize(gate.size());
        ple.resize(rows * std::size_t(c.layers) * c.ple_dim * sizeof(float));
        taps.resize(5 * hbytes);
        tokens.resize(rows * sizeof(unsigned)); positions.resize(rows * sizeof(std::uint64_t));
        if (reads.size() < rows * sizeof(gpu::KvRead)) reads.resize(rows * sizeof(gpu::KvRead));
        if (writes.size() < rows * sizeof(gpu::KvWrite)) writes.resize(rows * sizeof(gpu::KvWrite));
        // hilo linears keep high then low panels in the same scratch.
        const std::size_t panels = split_panels ? 2 : 1;
        halves.resize(panels * rows * std::size_t((std::max({c.hidden, c.inter, c.heads * 512, c.ple_dim}) + 15U) / 16U) * 16 * sizeof(std::uint16_t));
        if (fuse_ffn) pair_halves.resize(panels * rows * std::size_t((c.hidden + 15U) / 16U) * 16 * sizeof(std::uint16_t));
        rows_capacity = rows;
    }
    ~Impl() {
        if (side_queue) (void)hipStreamSynchronize(side_queue);
        for (auto* e : {forked, joined, ple_done}) if (e) (void)hipEventDestroy(e);
        if (side_queue) (void)hipStreamDestroy(side_queue);
        if (copy_queue) (void)hipStreamSynchronize(copy_queue);
        if (queue) (void)hipStreamSynchronize(queue);
        if (pinned_taps) (void)hipHostFree(pinned_taps);
        if (pinned_hidden) (void)hipHostFree(pinned_hidden);
        if (staging) (void)hipHostFree(staging);
        if (tapped) (void)hipEventDestroy(tapped);
        if (copied) (void)hipEventDestroy(copied);
        if (copy_queue) (void)hipStreamDestroy(copy_queue);
        if (queue) (void)hipStreamDestroy(queue);
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
        if (converted) (void)hipEventDestroy(converted);
    }
    template <class F> void timing(double& total, F body) {
        if (!profile) { body(); return; }
        checked(hipEventRecord(start, queue)); body(); checked(hipEventRecord(stop, queue));
        checked(hipEventSynchronize(stop));
        float ms = 0.0F; checked(hipEventElapsedTime(&ms, start, stop)); total += ms;
    }
    LinearView upload(LinearView view, Residency residency) {
        if (!view.data || residency == Residency::device) return view;
        if (const auto it = uploaded.find(view.data); it != uploaded.end()) {
            const auto& d = it->second;
            if (d.inputs != view.inputs || d.outputs != view.outputs || d.format != view.format || d.row_bytes != view.row_bytes)
                throw std::invalid_argument("ambiguous prefill tensor alias");
            return d;
        }
        validate_finite(view);
        storage.emplace_back(weight_storage_bytes(view));
        storage.back().upload(view.data, storage.back().size());
        const auto* original = view.data;
        view.data = storage.back().as<std::uint8_t>();
        uploaded.emplace(original, view);
        return view;
    }
    static float* host_rows(float*& rows, std::size_t& have, std::size_t floats) {
        if (have < floats) {
            if (rows) { checked(hipHostFree(rows)); rows = nullptr; have = 0; }
            void* p = nullptr;
            checked(hipHostMalloc(&p, floats * sizeof(float), hipHostMallocDefault));
            rows = static_cast<float*>(p); have = floats;
        }
        return rows;
    }
    void copy_ready() {
        if (!copy_queue) {
            checked(hipStreamCreateWithFlags(&copy_queue, hipStreamNonBlocking));
            checked(hipEventCreateWithFlags(&tapped, hipEventDisableTiming));
            checked(hipEventCreateWithFlags(&copied, hipEventDisableTiming));
        }
    }
    // Copy queue: taps rows [0, rows) into pinned memory once the queue has written them.
    void drain_taps(unsigned rows, unsigned width) {
        checked(hipEventRecord(tapped, queue)); checked(hipStreamWaitEvent(copy_queue, tapped, 0));
        checked(hipMemcpyAsync(pinned_taps, taps.data(), std::size_t(rows) * width * sizeof(float), hipMemcpyDeviceToHost, copy_queue));
        checked(hipEventRecord(copied, copy_queue));
    }
    void finite(const Buffer& b, std::size_t count) { gpu::check_finite(b.as<float>(), count, failure.as<unsigned>(), queue); }
    // Fused gate/up/product/convert: leaves the down input panels in halves
    // and the three range maxima at the indices separate dots would use.
    LinearPolicy policy;
    std::map<std::string, LinearPrecision> precisions;
    LinearPrecision precision_of(const std::string& op) {
        if (policy.empty()) return LinearPrecision::fp16;
        auto it = precisions.find(op);
        if (it == precisions.end()) it = precisions.emplace(op, policy.lookup(op)).first;
        return it->second;
    }
    bool fused(const std::string& name, LinearView wg, LinearView wu, LinearView down) {
        if (!fuse_ffn || stats.direct < 3 || down.format == WeightFormat::exl3_mul1) return false;
        // A corrected down still fuses: the pair kernel emits its low panel.
        const auto pg = precision_of(name + "gate");
        if (pg == LinearPrecision::exact || precision_of(name + "up") != pg ||
            precision_of(name + "down") == LinearPrecision::exact) return false;
        const auto g = prepared.find(wg.data), u = prepared.find(wu.data), d = prepared.find(down.data);
        return g != prepared.end() && u != prepared.end() && d != prepared.end() && gelu_pair_supported(*g->second, *u->second);
    }
    void gelu_pair(const std::string& name, LinearView wg, LinearView wu, unsigned rows) {
        const auto index = unsigned(ranges.size());
        ranges.push_back({name + "gate", 0.0F}); ranges.push_back({name + "up", 0.0F});
        timing(stats.linear_ms, [&] {
            prefill::gelu_pair(*prepared.at(wg.data), *prepared.at(wu.data), x.as<float>(), rows, pair_halves.data(), halves.data(),
                               failure.as<unsigned>(), maximum.as<unsigned>() + index, queue,
                               precision_of(name + "down") != LinearPrecision::fp16, precision_of(name + "gate") != LinearPrecision::fp16,
                               precision_of(name + "gate") == LinearPrecision::hilo_k128);
        });
    }
    // EXL3 FFN on prepared panels (g4-exl3-wmma-pf-v1): down's input panels in halves.
    bool exl3_pair(const std::string& name, LinearView wg, LinearView wu, LinearView wd, unsigned rows) {
        if (!fuse_ffn || wg.format != WeightFormat::exl3_mul1) return false;
        for (const char* op : {"gate", "up", "down"})
            if (precision_of(name + op) != LinearPrecision::fp16) return false;
        const auto g = prepared.find(wg.data), u = prepared.find(wu.data);
        if (g == prepared.end() || u == prepared.end() || !prepared.contains(wd.data) ||
            !exl3_gelu_pair_supported(*g->second, *u->second, wd)) return false;
        const auto index = unsigned(ranges.size());
        ranges.push_back({name + "gate", 0.0F}); ranges.push_back({name + "up", 0.0F});
        record(name + "gate_up", [&] {
            prefill::exl3_gelu_pair(*g->second, *u->second, wd, x.as<float>(), rows, pair_halves.data(), gate.as<float>(),
                                    up.as<float>(), halves.data(), failure.as<unsigned>(), maximum.as<unsigned>() + index, queue);
        }, 4.0 * rows * wg.inputs * wg.outputs, true);
        return true;
    }
    // g4-v1 rows linear, or the EXL3 pack's own identity: many-row core for
    // chunks, its eight-row decode linear otherwise (same bytes either way).
    // g4-a8-v1: up to eight rows take the decode target's a8 kernels, longer chunks the IU8 WMMA form
    // (same bytes; panels prepared on first use).
    // quantized: the queue's scratch already holds this input's codes (the previous a8 dot on this queue
    // read the same rows and width).
    void a8_dot(LinearView matrix, const float* input, unsigned rows, float* output, bool quantized = false) {
        if (rows <= 8) {
            const std::size_t need = std::size_t(rows) * (matrix.inputs + std::size_t(matrix.inputs / a8::group_width) * sizeof(a8::Meta));
            if (a8_scratch.size() < need) throw std::logic_error("g4-a8-v1 scratch not sized");
            const gpu::A8Rows q{a8_scratch.as<std::int8_t>(), reinterpret_cast<a8::Meta*>(a8_scratch.as<std::int8_t>() + std::size_t(rows) * matrix.inputs),
                                matrix.inputs};
            if (!quantized) gpu::a8_quantize(input, rows, matrix.inputs, q, queue);
            if (!gpu::a8_linear(matrix, q, rows, output, {}, queue, failure.as<unsigned>())) throw std::logic_error("g4-a8-v1 decode linear refused");
            return;
        }
        auto it = a8_prepared.find(matrix.data);
        if (it == a8_prepared.end()) it = prepare_a8(matrix);
        if (a8_scratch.size() < a8_scratch_bytes(rows, matrix.inputs)) throw std::logic_error("g4-a8-v1 scratch not sized");
        a8_linear(*it->second, input, rows, output, a8_scratch.data(), failure.as<unsigned>(), queue, !quantized);
    }
    // FFN down under g4-a8-v1: the GELU product and its quantization in one launch, then the a8 linear.
    void a8_down(const std::string& name, LinearView down, unsigned rows) {
        ranges.push_back({name, 0.0F});
        const auto before = stats.linear_ms;
        auto it = a8_prepared.find(down.data);
        if (it == a8_prepared.end()) it = prepare_a8(down);
        timing(stats.linear_ms, [&] {
            a8_product_linear(*it->second, gate.as<float>(), up.as<float>(), rows, out.as<float>(), a8_scratch.data(), failure.as<unsigned>(), queue);
        });
        if (profile) {
            auto& p = stats.kernels[name]; ++p.calls; p.ms += stats.linear_ms - before; p.matrix_ms += stats.linear_ms - before;
            p.flops += 2.0 * rows * down.inputs * down.outputs;
        }
    }
    // The second of two consecutive a8 dots over the same input reuses its codes.
    bool a8_same(LinearView first, LinearView second) const {
        return stats.linears == Linears::a8 && a8::quantized(first.format) && a8::quantized(second.format) && first.inputs == second.inputs;
    }
    std::map<const std::uint8_t*, std::unique_ptr<A8Prepared>>::iterator prepare_a8(LinearView matrix) {
        auto p = std::make_unique<A8Prepared>(matrix, queue);
        stats.prepared_bytes += p->panels().size() + p->scales().size();
        return a8_prepared.emplace(matrix.data, std::move(p)).first;
    }
    void exact_linear(LinearView matrix, const float* input, unsigned rows, float* output) {
        if (matrix.format != WeightFormat::exl3_mul1) {
            gpu::linear_rows(matrix, input, rows, output, queue, failure.as<unsigned>());
            return;
        }
        const auto bytes = std::max(exl3::device_scratch_bytes, exl3::rows_scratch_bytes(matrix, rows));
        if (exl3_scratch.size() < bytes) exl3_scratch.resize(bytes);
        if (exl3::device_linear_rows(matrix, input, rows, output, exl3_scratch.data(), exl3_scratch.size(), failure.as<unsigned>(), queue))
            return;
        for (unsigned r = 0; r < rows; r += 8)
            exl3::device_linear(matrix, input + std::size_t(r) * matrix.inputs, std::min(8U, rows - r),
                                output + std::size_t(r) * matrix.outputs, exl3_scratch.data(), failure.as<unsigned>(), true, queue);
    }
    // EXL3 many-row FFN (gate, up, fused middle stage, down) into out; false,
    // enqueueing nothing, when the pack, identity or switches select the separate linears.
    bool exl3_ffn(const std::string& name, LinearView wg, LinearView wu, LinearView wd, unsigned rows) {
        if (wg.format != WeightFormat::exl3_mul1 || wu.format != WeightFormat::exl3_mul1 || wd.format != WeightFormat::exl3_mul1)
            return false;
        const auto bytes = exl3::ffn_rows_scratch_bytes(wg, wu, wd, rows);
        if (!bytes) return false;
        if (exl3_scratch.size() < bytes) exl3_scratch.resize(bytes);
        bool done = false;
        record(name + "ffn", [&] {
            done = exl3::device_ffn_rows(wg, wu, wd, x.as<float>(), rows, gate.as<float>(), up.as<float>(), out.as<float>(),
                                         exl3_scratch.data(), exl3_scratch.size(), failure.as<unsigned>(), queue);
        }, 2.0 * rows * (2.0 * wg.inputs * wg.outputs + double(wd.inputs) * wd.outputs), true);
        if (done) { ranges.push_back({name + "gate", 0.0F}); ranges.push_back({name + "up", 0.0F}); ranges.push_back({name + "down", 0.0F}); }
        return done;
    }
    // Profiled stage outside dot(): its own kernel record (linear time when linear).
    template <class F> void record(const std::string& name, F body, double flops = 0.0, bool linear = false) {
        double elapsed = 0.0;
        timing(elapsed, body);
        (linear ? stats.linear_ms : stats.other_ms) += elapsed;
        if (profile) { auto& p = stats.kernels[name]; ++p.calls; p.ms += elapsed; p.matrix_ms += linear ? elapsed : 0.0; p.flops += flops; }
    }
    void dot(const std::string& name, LinearView matrix, const float* input, float* output, unsigned rows, bool preconverted = false) {
        const auto index = unsigned(ranges.size());
        ranges.push_back({name, 0.0F});
        const auto before = stats.linear_ms;
        if (stats.linears == Linears::a8 && a8::quantized(matrix.format)) {
            timing(stats.linear_ms, [&] { a8_dot(matrix, input, rows, output, preconverted); });
            if (profile) {
                auto& p = stats.kernels[name]; ++p.calls; p.ms += stats.linear_ms - before; p.matrix_ms += stats.linear_ms - before;
                p.flops += 2.0 * rows * matrix.inputs * matrix.outputs;
                p.bytes += double(rows) * (matrix.inputs + matrix.outputs) * sizeof(float) + double(matrix.inputs) * matrix.outputs;
            }
            return;
        }
        const auto precision = stats.linears != Linears::wmma ? LinearPrecision::exact : precision_of(name);
        if (precision == LinearPrecision::exact) {
            if (preconverted) throw std::logic_error("exact linear cannot read pre-converted panels");
            timing(stats.linear_ms, [&] { exact_linear(matrix, input, rows, output); });
            if (profile) {
                auto& p = stats.kernels[name]; ++p.calls; p.ms += stats.linear_ms - before; p.matrix_ms += stats.linear_ms - before;
                p.flops += 2.0 * rows * matrix.inputs * matrix.outputs;
                p.bytes += double(rows) * (matrix.inputs + matrix.outputs) * sizeof(float) + double(matrix.row_bytes) * matrix.outputs;
            }
            return;
        }
        timing(stats.linear_ms, [&] {
            const auto it = prepared.find(matrix.data);
            prefill::linear(matrix, input, rows, output, halves.data(), failure.as<unsigned>(), maximum.as<unsigned>() + index, queue,
                            true, (rows > 8 || stats.direct >= 3) && it != prepared.end() ? it->second.get() : nullptr, converted, preconverted,
                            precision);
        });
        if (profile) {
            auto& p = stats.kernels[name]; ++p.calls; p.ms += stats.linear_ms - before;
            float ms = 0.0F;
            checked(hipEventElapsedTime(&ms, start, converted)); p.conversion_ms += ms;
            checked(hipEventElapsedTime(&ms, converted, stop)); p.matrix_ms += ms;
            p.flops += 2.0 * rows * matrix.inputs * matrix.outputs;
            // Algorithmic minimum traffic, not a hardware bandwidth counter.
            p.bytes += double(rows) * (matrix.inputs + matrix.outputs) * sizeof(float) +
                       (stats.prepared && (rows > 8 || stats.direct >= 3) ? 2.0 * matrix.inputs * matrix.outputs : double(matrix.row_bytes) * matrix.outputs);
        }
    }
    // q, k and v read the same normalized input: k and v may reuse the panels
    // q's conversion left in halves (same FP16 operands, same bytes) when all
    // three run panel GEMMs of one precision on non-EXL3 packs.
    bool share_qkv_panels(const std::string& name, LinearView wq, LinearView wk, LinearView wv) {
        if (stats.linears != Linears::wmma || stats.direct < 3 || !qkv_share_enabled()) return false;
        const auto mode = precision_of(name + "q");
        if (mode == LinearPrecision::exact || precision_of(name + "k") != mode || precision_of(name + "v") != mode) return false;
        for (const auto& m : {wq, wk, wv})
            if (m.format == WeightFormat::exl3_mul1 || m.inputs != wq.inputs || !prepared.contains(m.data)) return false;
        return true;
    }
    void norm(const Buffer& input, LinearView weight, Buffer& output, unsigned vectors, unsigned width) {
        gpu::norm(input.as<float>(), weight, output.as<float>(), vectors, width, queue, failure.as<unsigned>());
    }
    // FUSE_EPILOGUE for this chunk: the flag of a fused producer, else null (separate check).
    bool fuse = false;
    [[nodiscard]] unsigned* fused_flag() const { return fuse ? failure.as<unsigned>() : nullptr; }
    void head(Buffer& b, LinearView norm, unsigned rows, unsigned count, unsigned dim, bool is_global) {
        // The norm flags every value; the fused rotation flags the pairs it rewrites.
        gpu::norm(b.as<float>(), norm, b.as<float>(), rows * count, dim, queue, failure.as<unsigned>());
        const auto total = rows * count * std::min(dim / 2, is_global ? 64U : 128U);
        hipLaunchKernelGGL(rotate, dim3((total + 255U) / 256U), dim3(256), 0, queue, b.as<float>(), positions.as<std::uint64_t>(),
                           is_global ? global.as<float>() : sliding.as<float>(), rows, count, dim, is_global, fused_flag());
        launched();
        if (!fuse) finite(b, std::size_t(rows) * count * dim);
    }
    // residual += norm(branch) [* scalar], finite-checked; with next, x = norm(residual, next)
    // in the same launch when fused. Returns whether x now holds that next norm.
    bool add(const Buffer& branch, LinearView norm, unsigned rows, LinearView scalar = {}, LinearView next = {}) {
        if (fuse && norm.data &&
            gpu::norm_residual_rows(branch.as<float>(), norm, residual.as<float>(), rows, w.config.hidden, scalar,
                                    next.data ? x.as<float>() : nullptr, next, failure.as<unsigned>(), queue))
            return next.data != nullptr;
        this->norm(branch, norm, x, rows, w.config.hidden);
        gpu::residual(residual.as<float>(), x.as<float>(), rows * w.config.hidden, scalar, queue);
        finite(residual, std::size_t(rows) * w.config.hidden);
        return false;
    }
};
DevicePrefill::DevicePrefill(HostWeights weights, RopeTables t, Residency residency, bool profile, AttentionMode attention,
                             Linears linears) : impl_(std::make_unique<Impl>()) {
    detail::validate_weights(weights);
    gpu::require_device();
    if (std::fegetround() != FE_TONEAREST) throw std::invalid_argument("prefill requires RNE");
    detail::validate_tables(t);
    auto& b = *impl_;
    b.profile = profile;
    b.stats.attention = attention;
    b.stats.attention_packed = attention == AttentionMode::wmma && attention_pack_enabled();
    b.stats.canonical_kernel = attention == AttentionMode::canonical && canonical_kernel_enabled();
    b.stats.linears = linears;
    b.stats.tiled = linears == Linears::wmma && tiled_enabled();
    b.stats.prepared = prepared_enabled() && b.stats.tiled;
    b.stats.direct = b.stats.prepared ? direct_mode() : 0;
    if (profile) { checked(hipEventCreate(&b.start)); checked(hipEventCreate(&b.stop)); checked(hipEventCreate(&b.converted)); }
    b.w = weights;
    b.w.embedding = b.upload(weights.embedding, residency); b.w.ple_table = b.upload(weights.ple_table, residency);
    b.w.head = b.upload(weights.head, residency);
    b.w.ple_model = b.upload(weights.ple_model, residency); b.w.ple_model_norm = b.upload(weights.ple_model_norm, residency);
    b.w.final_norm = b.upload(weights.final_norm, residency);
    for (auto& y : b.w.layer)
        for (auto* view : {&y.input_norm, &y.post_attention_norm, &y.pre_ffn_norm, &y.post_ffn_norm, &y.post_ple_norm,
                          &y.q, &y.k, &y.v, &y.o, &y.q_norm, &y.k_norm, &y.gate, &y.up, &y.down, &y.ple_gate, &y.ple_projection, &y.scalar})
            *view = b.upload(*view, residency);
    b.table_positions = t.positions;
    b.sliding.resize(t.sliding.size_bytes()); b.sliding.upload(t.sliding.data(), b.sliding.size());
    b.global.resize(t.global.size_bytes()); b.global.upload(t.global.data(), b.global.size());
    checked(hipStreamCreateWithFlags(&b.queue, hipStreamNonBlocking));
    if (b.stats.prepared) {
        const auto prepare = [&](LinearView matrix) {
            if (matrix.data && !b.prepared.contains(matrix.data)) {
                auto p = std::make_unique<PreparedLinear>(matrix, b.queue);
                b.stats.prepared_bytes += p->panels().size();
                b.prepared.emplace(matrix.data, std::move(p));
            }
        };
        prepare(b.w.head.data ? b.w.head : b.w.embedding); prepare(b.w.ple_model);
        for (const auto& y : b.w.layer)
            for (auto matrix : {y.q, y.k, y.v, y.o, y.gate, y.up, y.down, y.ple_gate, y.ple_projection}) prepare(matrix);
    }
    if (linears == Linears::a8)
        for (const auto& y : b.w.layer)
            for (auto matrix : {y.q, y.k, y.v, y.o, y.gate, y.up, y.down, y.ple_gate, y.ple_projection})
                if (a8_prefill_supported(matrix) && !b.a8_prepared.contains(matrix.data)) b.prepare_a8(matrix);
    const auto& c = b.w.config;
    const std::size_t hbytes = chunk_rows * std::size_t(c.hidden) * sizeof(float), abytes = chunk_rows * std::size_t(c.heads) * 512 * sizeof(float);
    for (auto* p : {&b.residual, &b.x, &b.out}) p->resize(hbytes);
    for (auto* p : {&b.q, &b.att}) p->resize(abytes);
    for (auto* p : {&b.k, &b.v}) p->resize(chunk_rows * std::size_t(c.kv_heads) * 512 * sizeof(float));
    b.gate.resize(chunk_rows * std::size_t(std::max(c.inter, c.ple_dim)) * sizeof(float)); b.up.resize(b.gate.size());
    b.ple.resize(chunk_rows * std::size_t(c.layers) * c.ple_dim * sizeof(float));
    b.logits.resize(std::size_t(c.vocab) * sizeof(float)); b.taps.resize(5 * hbytes);
    b.tokens.resize(chunk_rows * sizeof(unsigned)); b.positions.resize(chunk_rows * sizeof(std::uint64_t));
    // Every layer's descriptors of a full chunk: the per-pass upload never reallocates.
    b.reads.resize(std::size_t(c.layers) * chunk_rows * sizeof(gpu::KvRead));
    b.writes.resize(std::size_t(c.layers) * chunk_rows * sizeof(gpu::KvWrite));
    b.staging_bytes = b.tokens.size() + b.positions.size() + b.reads.size() + b.writes.size() + 4 * 256;
    checked(hipHostMalloc(&b.staging, b.staging_bytes, hipHostMallocDefault));
    b.failure.resize(sizeof(unsigned)); b.maximum.resize((2 + 11 * c.layers) * sizeof(unsigned));
    b.last_row.resize(std::size_t(c.hidden) * sizeof(float));
    b.fuse_ffn = linears == Linears::wmma && fused_ffn_enabled();
    b.size_rows(chunk_rows);
    // Engine-visible selector until the profile row lands: the LinearPolicy
    // spec of the WMMA linears (e.g. 'ple_model=fp16;.*=hilo').
    if (const auto* p = std::getenv("STRIX_G4_PREFILL_LINEAR_POLICY"); p && *p && linears == Linears::wmma)
        set_linear_policy(LinearPolicy::parse(p));
}
DevicePrefill::~DevicePrefill() = default;
const std::vector<Range>& DevicePrefill::ranges() const { return impl_->ranges; }
const Stats& DevicePrefill::stats() const { return impl_->stats; }
const HostWeights& DevicePrefill::borrow_device_weights() const { return impl_->w; }
void DevicePrefill::set_profile(bool enabled) {
    auto& b = *impl_;
    if (enabled && !b.start) { checked(hipEventCreate(&b.start)); checked(hipEventCreate(&b.stop)); checked(hipEventCreate(&b.converted)); }
    b.profile = enabled;
}
void DevicePrefill::use_canonical_kernel(bool enabled) {
    impl_->stats.canonical_kernel = impl_->stats.attention == AttentionMode::canonical && enabled;
}
ChunkResult DevicePrefill::forward(KvTxn& transaction, std::span<const unsigned> input, bool all_logits, bool logits, bool hidden,
                                   bool taps, bool views) {
    const PrefillSegment one{&transaction, input};
    return std::move(pass({&one, 1}, all_logits, logits, hidden, taps, views).front());
}
std::vector<ChunkResult> DevicePrefill::forward_batch(std::span<const PrefillSegment> segments, bool hidden, bool taps, bool views) {
    return pass(segments, false, false, hidden, taps, views);
}
std::vector<ChunkResult> DevicePrefill::pass(std::span<const PrefillSegment> segments, bool all_logits, bool logits, bool hidden,
                                             bool taps, bool views) {
    using Clock = std::chrono::steady_clock;
    const auto entered = Clock::now();
    auto launched_at = entered, synced_at = entered;
    auto& b = *impl_; const auto& w = b.w; const auto& c = w.config;
    // Segment s owns rows [first, first + rows) of the pass; its attention uses its own stride.
    struct Part { gpu::DeviceStream::Impl::Txn* tx; unsigned first, rows, stride; };
    std::vector<Part> parts;
    std::vector<unsigned> input;
    unsigned n = 0;
    if (segments.empty() || ((all_logits || logits) && segments.size() != 1)) throw std::invalid_argument("prefill pass segments");
    for (const auto& s : segments) {
        auto* tx = dynamic_cast<gpu::DeviceStream::Impl::Txn*>(s.transaction);
        const auto rows = unsigned(s.tokens.size());
        if (!tx || tx->published || tx->failed || tx->evaluated || tx->rows || tx->stream->poisoned || !gpu::same_config(tx->stream->config, c))
            throw std::invalid_argument("invalid prefill transaction");
        if (!rows || s.tokens.size() > chunk_rows || tx->start > tx->stream->admitted || s.tokens.size() > tx->stream->admitted - tx->start ||
            tx->start > b.table_positions || s.tokens.size() > b.table_positions - tx->start)
            throw std::invalid_argument("prefill row or position bounds");
        for (const auto t : s.tokens) if (t >= c.vocab) throw std::invalid_argument("prefill token out of range");
        for (const auto& p : parts) if (p.tx->stream == tx->stream) throw std::invalid_argument("prefill pass repeats a stream");
        parts.push_back({tx, n, rows, unsigned(((tx->start + rows + 63U) / 64U) * 64U)});
        input.insert(input.end(), s.tokens.begin(), s.tokens.end());
        n += rows;
    }
    if (n > chunk_rows) throw std::invalid_argument("prefill row or position bounds");
    // Announced before any device work: no graph capture starts on another thread until this pass returns.
    const gpu::PrefillInFlight in_flight;
    // KV_QUEUE: the commit publishes this pass's rows on the queue that wrote them.
    for (const auto& p : parts) if (p.tx->stream->writer_queue) p.tx->stream->queue = b.queue;
    ChunkResult result; result.rows = n; result.vocab = c.vocab; result.tap_width = 5 * c.hidden;
    // PINNED_TAPS: unconsumed taps are never copied; consumed taps and hidden
    // rows land in pinned memory through asynchronous copies. Off: as before.
    const bool pinned = views && lever_enabled(lever_pinned_taps), want_taps = taps || !pinned;
    unsigned last_tap = 0;
    for (const auto t : tap_layers) last_tap = std::max(last_tap, unsigned(t));
    bool taps_issued = false;
    // DEVICE_FINITE: one finite check over the returned taps rows, inside the
    // failure fence, replaces the consumer's host scan of the same values.
    const bool device_finite = lever_enabled(lever_device_finite);
    bool taps_checked = false;
    try {
        b.queue_io = queue_io_enabled(); b.staged = 0;
        for (const auto& p : parts) {
            auto* tx = p.tx;
            tx->rows = p.rows;
            for (unsigned l = 0; l < c.layers; ++l) {
                const auto bytes = std::size_t(p.rows) * tx->stream->cache[l].width * sizeof(float);
                if (bytes > tx->overlay[l].k.size()) { tx->overlay[l].k.resize(bytes); tx->overlay[l].v.resize(bytes); }
            }
        }
        b.ranges.clear(); b.range_aliases.clear(); b.clear(b.maximum); b.clear(b.failure);
        // Pinned copies only the rows this chunk writes; a model with fewer
        // layers than a tap index leaves that tap's columns at zero, as the
        // vector path's full zero does.
        bool every_tap = true;
        for (const auto t : tap_layers) every_tap = every_tap && t < c.layers;
        if (!pinned) b.clear(b.taps);
        else if (want_taps && !every_tap)
            checked(hipMemsetAsync(b.taps.data(), 0, std::size_t(n) * result.tap_width * sizeof(float), b.queue));
        if (pinned) {
            b.copy_ready();
            if (want_taps) Impl::host_rows(b.pinned_taps, b.pinned_taps_floats, std::size_t(b.rows_capacity) * result.tap_width);
            if (hidden) Impl::host_rows(b.pinned_hidden, b.pinned_hidden_floats, std::size_t(b.rows_capacity) * c.hidden);
        }
        b.put(b.tokens, input.data(), input.size() * sizeof(unsigned));
        std::vector<std::uint64_t> positions(n);
        for (const auto& p : parts) for (unsigned r = 0; r < p.rows; ++r) positions[p.first + r] = p.tx->start + r;
        b.put(b.positions, positions.data(), positions.size() * sizeof(std::uint64_t));
        // Buffer capacity covers the widest segment; each segment's kernels take its own stride.
        unsigned stride = 0;
        for (const auto& p : parts) stride = std::max(stride, p.stride);
        if (stride > b.stride) {
            // Capacity only: every kernel takes this pass's stride; QUEUE_IO doubles the capacity.
            const auto capacity = b.queue_io ? std::max(stride, 2 * b.stride) : stride, capacity_chunks = capacity / 64;
            b.stride = capacity;
            b.scores.resize(8ULL * c.heads * capacity * sizeof(float));
            b.maxima.resize(8ULL * c.heads * (capacity_chunks + 1ULL) * sizeof(float));
            b.partials.resize(8ULL * c.heads * capacity_chunks * 513 * sizeof(float));
            if (b.stats.attention_packed) {
                b.attention_halves.resize(attention_packed_bytes(capacity, c.kv_heads * 512));
                b.global_halves.resize(attention_packed_bytes(capacity, c.kv_heads * 512));
            }
        }
        if (b.stats.canonical_kernel) {
            // Global layers span every key from 0: the widest canonical score span.
            const auto bytes = std::size_t(n) * c.heads * stride * sizeof(float);
            b.grow(b.canonical_scores, bytes);
            if (b.canonical_denominators.size() < chunk_rows * std::size_t(c.heads) * sizeof(float))
                b.canonical_denominators.resize(chunk_rows * std::size_t(c.heads) * sizeof(float));
            const auto packed = canonical_packed_floats(chunk_rows, c.heads, 512, stride) * sizeof(float);
            if (canonical_pack_enabled()) b.grow(b.canonical_packed, packed);
        }
        b.fuse = lever_enabled(lever_fuse_epilogue);
        if (b.fuse) result.levers |= lever_fuse_epilogue;
        // ROWS_TILE_V2 engages where the g4-v1 many-row linears run.
        if (b.stats.linears == Linears::exact && gpu::rows_linear_enabled() && lever_enabled(lever_rows_tile_v2))
            result.levers |= lever_rows_tile_v2;
        if (b.stats.linears == Linears::exact && gpu::rows_linear_enabled() && lever_enabled(lever_decode24)) {
            result.levers |= lever_decode24;
            if (lever_enabled(lever_shared_q6)) result.levers |= lever_shared_q6;
        }
        hipLaunchKernelGGL(gather, dim3((n * c.hidden + 255U) / 256U), dim3(256), 0, b.queue, w.embedding,
                           b.tokens.as<unsigned>(), n, embedding_scale, b.residual.as<float>(), b.fused_flag());
        launched(); launched_at = Clock::now();
        if (!b.fuse) b.finite(b.residual, std::size_t(n) * c.hidden);
        // SIDE_STREAM: independent work of the same bytes on a second queue. The
        // chunk's PLE projection reads the embedded rows until layer 0 first
        // writes the residual; producer layers' K/V run beside q and its head.
        // The PLE projection queues behind layer 0's K/V, so attention does not
        // wait for it.
        const bool side = !b.profile && lever_enabled(lever_side_stream);
        if (side) { b.reserve_exl3(n); b.side_ready(); result.levers |= lever_side_stream; }
        const auto pn = n * c.layers * c.ple_dim;
        const auto ple_stage = [&] {
            b.dot("ple_model", w.ple_model, b.residual.as<float>(), b.ple.as<float>(), n);
            hipLaunchKernelGGL(scale_ple, dim3((pn + 255U) / 256U), dim3(256), 0, b.queue, b.ple.as<float>(), pn);
            launched();
            gpu::norm(b.ple.as<float>(), w.ple_model_norm, b.ple.as<float>(), n * c.layers, c.ple_dim, b.queue, b.failure.as<unsigned>());
            hipLaunchKernelGGL(combine_ple, dim3((pn + 255U) / 256U), dim3(256), 0, b.queue, b.ple.as<float>(), w.ple_table, b.tokens.as<unsigned>(), pn,
                               b.fused_flag());
            launched();
            if (!b.fuse) b.finite(b.ple, pn);
        };
        bool ple_pending = side, ple_queued = !side;
        if (!side) ple_stage();
        // Descriptors are immutable for the chunk: upload every layer's reads
        // and the producers' writes once instead of draining the queue before a
        // synchronous upload in each layer. The canonical kernel also reads the
        // host copy.
        std::vector<gpu::KvRead> all_reads(std::size_t(c.layers) * n);
        {
            auto& reads = all_reads;
            std::vector<gpu::KvWrite> writes(std::size_t(c.layers) * n);
            for (unsigned l = 0; l < c.layers; ++l)
                for (const auto& p : parts)
                    for (unsigned r = 0; r < p.rows; ++r) {
                        reads[std::size_t(l) * n + p.first + r] = p.tx->read(c.layer[l].producer, p.tx->start + r);
                        writes[std::size_t(l) * n + p.first + r] = {p.tx->overlay[l].k.as<float>(), p.tx->overlay[l].v.as<float>(), r};
                    }
            if (b.reads.size() < reads.size() * sizeof(gpu::KvRead)) b.reads.resize(reads.size() * sizeof(gpu::KvRead));
            if (b.writes.size() < writes.size() * sizeof(gpu::KvWrite)) b.writes.resize(writes.size() * sizeof(gpu::KvWrite));
            b.put(b.reads, reads.data(), reads.size() * sizeof(gpu::KvRead));
            b.put(b.writes, writes.data(), writes.size() * sizeof(gpu::KvWrite));
        }
        // Sliding and global layers pack into separate buffers, so a layer that
        // reads its producer's KV (layers >= 24) reuses that producer's pack.
        unsigned packed_producer[2] = {~0U, ~0U};
        bool x_ready = false; // FUSE_EPILOGUE: x already holds this layer's input norm
        for (unsigned l = 0; l < c.layers; ++l) {
            const auto& y = w.layer[l]; const auto& a = c.layer[l]; const auto d = a.head_dim;
            auto* layer_reads = b.reads.as<gpu::KvRead>() + std::size_t(l) * n;
            const bool global = a.attention == AttentionKind::global;
            const auto name = "layer." + std::to_string(l) + ".";
            if (!x_ready) b.norm(b.residual, y.input_norm, b.x, n, c.hidden);
            x_ready = false;
            // Without the side queue, k and v read q's converted panels.
            const bool share = !side && a.producer == l && b.share_qkv_panels(name, y.q, y.k, y.v);
            const auto q_slot = unsigned(b.ranges.size());
            const auto kv = [&] {
                if (share) { b.range_aliases.push_back({q_slot + 1, q_slot}); b.range_aliases.push_back({q_slot + 2, q_slot}); }
                b.dot(name + "k", y.k, b.x.as<float>(), b.k.as<float>(), n, share); b.dot(name + "v", y.v, b.x.as<float>(), b.v.as<float>(), n, share || b.a8_same(y.k, y.v));
                b.head(b.k, y.k_norm, n, c.kv_heads, d, global);
                gpu::norm(b.v.as<float>(), {}, b.v.as<float>(), n * c.kv_heads, d, b.queue, b.failure.as<unsigned>());
                gpu::store_overlay(b.k.as<float>(), b.v.as<float>(), b.writes.as<gpu::KvWrite>() + std::size_t(l) * n, n, c.kv_heads * d, b.queue);
            };
            // The side queue forks after the input norm; ranges keep the order q, k, v.
            if (side && a.producer == l) b.fork();
            b.dot(name + "q", y.q, b.x.as<float>(), b.q.as<float>(), n); b.head(b.q, y.q_norm, n, c.heads, d, global);
            if (a.producer == l) {
                if (side) {
                    b.on_side(b.joined, kv); checked(hipStreamWaitEvent(b.queue, b.joined, 0));
                    if (!ple_queued) { b.on_side(b.ple_done, ple_stage); ple_queued = true; }
                }
                else kv();
            }
            b.timing(b.stats.attention_ms, [&] {
            // One attention call per segment, with that segment's rows, KV reads and stride (as forward() alone).
            for (const auto& p : parts) {
            const auto rows = p.rows;
            auto* q = b.q.as<float>() + std::size_t(p.first) * c.heads * d;
            auto* att = b.att.as<float>() + std::size_t(p.first) * c.heads * d;
            const auto* reads = layer_reads + p.first;
            if (b.stats.attention == AttentionMode::wmma) {
                // Packs are reused across layers only within a one-segment pass: segments share the pack buffers.
                auto& owner = packed_producer[global ? 1 : 0];
                const bool pack = !b.stats.attention_packed || owner != a.producer || !pack_reuse_enabled() || parts.size() > 1;
                owner = a.producer;
                prefill::attention(q, reads, rows, c.heads, d, att, b.failure.as<unsigned>(), b.queue,
                                    b.stats.attention_packed ? (global ? b.global_halves : b.attention_halves).data() : nullptr,
                                    p.stride, global ? p.stride : std::min(p.stride, c.window + rows - 1), pack);
            } else if (!b.stats.canonical_kernel ||
                       !canonical_attention(q, std::span<const gpu::KvRead>(all_reads).subspan(std::size_t(l) * n + p.first, rows), c.heads, d, att, b.failure.as<unsigned>(), b.queue,
                                            {b.canonical_scores.as<float>(), b.canonical_scores.size() / sizeof(float),
                                             b.canonical_denominators.as<float>(), b.canonical_packed.as<float>(),
                                             b.canonical_packed.size() / sizeof(float)})) {
            for (unsigned r = 0; r < rows; r += 8) {
                const auto count = std::min(8U, rows - r);
                gpu::attention(q + std::size_t(r) * c.heads * d, reads + r, count, c.heads, d,
                               p.stride, b.scores.as<float>(), b.maxima.as<float>(), b.partials.as<float>(),
                               att + std::size_t(r) * c.heads * d, b.failure.as<unsigned>(), b.queue);
            }
            }
            }
            });
            b.dot(name + "o", y.o, b.att.as<float>(), b.out.as<float>(), n);
            if (!ple_queued) { b.fork(); b.on_side(b.ple_done, ple_stage); ple_queued = true; }
            if (ple_pending) { checked(hipStreamWaitEvent(b.queue, b.ple_done, 0)); ple_pending = false; }
            if (!b.add(b.out, y.post_attention_norm, n, {}, y.pre_ffn_norm)) b.norm(b.residual, y.pre_ffn_norm, b.x, n, c.hidden);
            if (b.stats.linears == Linears::exact && b.exl3_ffn(name, y.gate, y.up, y.down, n)) {
                // EXL3 many-row FFN, down included.
            } else if (b.stats.linears != Linears::wmma) {
                // The decode target's gate_up: gelu_tanh(gate) * up stored per element.
                bool down_done = false;
                if (y.gate.format == WeightFormat::exl3_mul1 || y.up.format == WeightFormat::exl3_mul1 || b.stats.linears == Linears::a8) {
                    // EXL3 packs: separate linears (each profiled), then the decode target's gelu product.
                    b.dot(name + "gate", y.gate, b.x.as<float>(), b.gate.as<float>(), n);
                    b.dot(name + "up", y.up, b.x.as<float>(), b.up.as<float>(), n, b.a8_same(y.gate, y.up));
                    if (b.stats.linears == Linears::a8 && n > 8 && a8_prefill_supported(y.down)) {
                        b.a8_down(name + "down", y.down, n);
                        down_done = true;
                    } else
                        b.record(name + "gelu_product", [&] { gpu::gelu_product(b.gate.as<float>(), b.up.as<float>(), n * c.inter, b.queue); });
                } else {
                    b.ranges.push_back({name + "gate", 0.0F}); b.ranges.push_back({name + "up", 0.0F});
                    b.record(name + "gate_up", [&] {
                        gpu::gate_up_rows(y.gate, y.up, b.x.as<float>(), n, b.gate.as<float>(), b.queue, b.failure.as<unsigned>(),
                                          gpu::rows_linear_enabled(), -1, b.up.as<float>(), -1, b.fuse);
                    }, 4.0 * n * y.gate.inputs * y.gate.outputs, true);
                }
                if (!down_done) b.dot(name + "down", y.down, b.gate.as<float>(), b.out.as<float>(), n);
            } else if (b.exl3_pair(name, y.gate, y.up, y.down, n)) {
                b.dot(name + "down", y.down, b.gate.as<float>(), b.out.as<float>(), n, true);
            } else if (b.fused(name, y.gate, y.up, y.down)) {
                b.gelu_pair(name, y.gate, y.up, n);
                b.dot(name + "down", y.down, b.gate.as<float>(), b.out.as<float>(), n, true);
            } else {
                b.dot(name + "gate", y.gate, b.x.as<float>(), b.gate.as<float>(), n); b.dot(name + "up", y.up, b.x.as<float>(), b.up.as<float>(), n);
                if (b.fuse) gpu::gelu_product_finite(b.gate.as<float>(), b.up.as<float>(), n * c.inter, b.failure.as<unsigned>(), b.queue);
                else { gpu::gelu_product(b.gate.as<float>(), b.up.as<float>(), n * c.inter, b.queue); b.finite(b.gate, std::size_t(n) * c.inter); }
                b.dot(name + "down", y.down, b.gate.as<float>(), b.out.as<float>(), n);
            }
            b.add(b.out, y.post_ffn_norm, n);
            b.dot(name + "ple_gate", y.ple_gate, b.residual.as<float>(), b.gate.as<float>(), n);
            hipLaunchKernelGGL(ple_product, dim3((n * c.ple_dim + 255U) / 256U), dim3(256), 0, b.queue, b.gate.as<float>(),
                               b.ple.as<float>(), n * c.ple_dim, c.ple_dim, c.layers, l, b.fused_flag());
            launched();
            if (!b.fuse) b.finite(b.gate, std::size_t(n) * c.ple_dim);
            b.dot(name + "ple_projection", y.ple_projection, b.gate.as<float>(), b.out.as<float>(), n); x_ready = b.add(b.out, y.post_ple_norm, n, y.scalar, l + 1 < c.layers ? w.layer[l + 1].input_norm : w.final_norm);
            if (want_taps) for (unsigned t = 0; t < 5; ++t) if (tap_layers[t] == l)
                checked(hipMemcpy2DAsync(b.taps.as<float>() + t * c.hidden, 5ULL * c.hidden * sizeof(float), b.residual.data(),
                                        std::size_t(c.hidden) * sizeof(float), std::size_t(c.hidden) * sizeof(float), n, hipMemcpyDeviceToDevice, b.queue));
            if (want_taps && device_finite && l == last_tap) { b.finite(b.taps, std::size_t(n) * result.tap_width); taps_checked = true; }
            // The copy queue drains the taps while the remaining layers run.
            if (pinned && want_taps && l == last_tap) { b.drain_taps(n, result.tap_width); taps_issued = true; }
        }
        if (want_taps && device_finite && !taps_checked) { b.finite(b.taps, std::size_t(n) * result.tap_width); taps_checked = true; }
        if (pinned && want_taps && !taps_issued) { b.drain_taps(n, result.tap_width); taps_issued = true; }
        if (!x_ready) b.norm(b.residual, w.final_norm, b.x, n, c.hidden);
        // Last-row-only head uses the same WMMA tile and K order. Qualification
        // can request every row without changing any already computed state.
        const unsigned head_rows = all_logits ? n : 1;
        const bool head = logits || all_logits;
        if (head) {
            if (b.queue_io) b.grow(b.logits, std::size_t(head_rows) * c.vocab * sizeof(float));
            else b.logits.resize(std::size_t(head_rows) * c.vocab * sizeof(float));
            b.dot("head", w.head.data ? w.head : w.embedding, b.x.as<float>() + (all_logits ? 0 : std::size_t(n - 1) * c.hidden), b.logits.as<float>(), head_rows);
            gpu::cap_logits(b.logits.as<float>(), head_rows * c.vocab, b.queue); b.finite(b.logits, std::size_t(head_rows) * c.vocab);
        } else {
            checked(hipMemcpyAsync(b.last_row.data(), b.x.as<float>() + std::size_t(n - 1) * c.hidden, std::size_t(c.hidden) * sizeof(float),
                                   hipMemcpyDeviceToDevice, b.queue));
        }
        if (pinned && hidden)
            checked(hipMemcpyAsync(b.pinned_hidden, b.x.data(), std::size_t(n) * c.hidden * sizeof(float), hipMemcpyDeviceToHost, b.queue));
        gpu::synchronize(b.queue);
        if (taps_issued) checked(hipEventSynchronize(b.copied));
        synced_at = Clock::now();
        unsigned failure = 0; b.get(&failure, b.failure, sizeof(failure));
        std::vector<unsigned> maxima(b.ranges.size()); b.get(maxima.data(), b.maximum, maxima.size() * sizeof(unsigned));
        for (const auto& [slot, source] : b.range_aliases) maxima[slot] = maxima[source];
        std::string overflow;
        for (std::size_t i = 0; i < maxima.size(); ++i) {
            const float observed = bit_cast<float>(maxima[i]);
            auto& high = b.range_maxima[b.ranges[i].tensor];
            high = !std::isfinite(observed) ? observed : std::max(high, observed);
            b.ranges[i].input_maximum = high;
            if (overflow.empty() && (!std::isfinite(observed) || observed >= 65520.0F))
                overflow = "; first overflowing input " + b.ranges[i].tensor + " abs_max=" + std::to_string(observed);
        }
        if (failure) throw std::runtime_error("prefill failed: nonfinite arithmetic, FP16 overflow or missing KV (mask " + std::to_string(failure) + ")" + overflow);
        if (head) { result.last_logits.resize(c.vocab); b.get(result.last_logits.data(), b.logits, c.vocab * sizeof(float), std::size_t(head_rows - 1) * c.vocab * sizeof(float)); }
        if (pinned) {
            if (want_taps) result.pinned_taps = {b.pinned_taps, std::size_t(n) * result.tap_width};
            result.levers |= lever_pinned_taps;
        } else {
            result.taps.resize(std::size_t(n) * result.tap_width); b.get(result.taps.data(), b.taps, result.taps.size() * sizeof(float));
        }
        if (taps_checked) result.levers |= lever_device_finite;
        if (all_logits) { result.all_logits.resize(std::size_t(n) * c.vocab); b.get(result.all_logits.data(), b.logits, result.all_logits.size() * sizeof(float)); }
        if (hidden && pinned) result.pinned_hidden = {b.pinned_hidden, std::size_t(n) * c.hidden};
        else if (hidden) { result.hidden.resize(std::size_t(n) * c.hidden); b.get(result.hidden.data(), b.x, result.hidden.size() * sizeof(float)); }
        for (const auto& p : parts) p.tx->evaluated = true;
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        result.host_ms = ms(launched_at - entered) + ms(Clock::now() - synced_at);
    } catch (...) {
        (void)hipStreamSynchronize(b.queue);
        if (b.copy_queue) (void)hipStreamSynchronize(b.copy_queue);
        if (b.side_queue) (void)hipStreamSynchronize(b.side_queue);
        for (const auto& p : parts) p.tx->failed = true;
        throw;
    }
    if (parts.size() == 1) return {std::move(result)};
    // Each segment gets its own rows of the pass (taps, hidden; views stay views). Host time is the pass's.
    std::vector<ChunkResult> out(parts.size());
    for (std::size_t s = 0; s < parts.size(); ++s) {
        const auto& p = parts[s];
        auto& r = out[s];
        r.rows = p.rows; r.vocab = result.vocab; r.tap_width = result.tap_width; r.levers = result.levers; r.host_ms = result.host_ms;
        const auto t0 = std::size_t(p.first) * result.tap_width, tn = std::size_t(p.rows) * result.tap_width;
        const auto h0 = std::size_t(p.first) * c.hidden, hn = std::size_t(p.rows) * c.hidden;
        if (!result.pinned_taps.empty()) r.pinned_taps = result.pinned_taps.subspan(t0, tn);
        if (!result.taps.empty()) r.taps.assign(result.taps.begin() + std::ptrdiff_t(t0), result.taps.begin() + std::ptrdiff_t(t0 + tn));
        if (!result.pinned_hidden.empty()) r.pinned_hidden = result.pinned_hidden.subspan(h0, hn);
        if (!result.hidden.empty()) r.hidden.assign(result.hidden.begin() + std::ptrdiff_t(h0), result.hidden.begin() + std::ptrdiff_t(h0 + hn));
    }
    return out;
}
namespace {
bool defer_head_enabled() {
    static const bool enabled = [] { const auto* p = std::getenv("STRIX_G4_PREFILL_DEFER_HEAD"); return !p || std::string_view(p) != "0"; }();
    return enabled;
}
} // namespace
LinearPolicy LinearPolicy::parse(const std::string& spec) {
    LinearPolicy out;
    std::stringstream in(spec); std::string item;
    while (std::getline(in, item, ';')) {
        if (item.empty()) continue;
        const auto eq = item.rfind('=');
        Rule rule{item.substr(0, eq), LinearPrecision::hilo_k128};
        if (eq != std::string::npos) {
            const auto mode = item.substr(eq + 1);
            if (mode == "fp16") rule.mode = LinearPrecision::fp16;
            else if (mode == "exact") rule.mode = LinearPrecision::exact;
            else if (mode == "hilo") rule.mode = LinearPrecision::hilo;
            else if (mode == "hilo_k128") rule.mode = LinearPrecision::hilo_k128;
            else throw std::invalid_argument("unknown linear precision: " + mode);
        }
        (void)std::regex(rule.pattern); // validate
        out.rules.push_back(rule);
    }
    return out;
}
LinearPrecision LinearPolicy::lookup(const std::string& op) const {
    for (const auto& r : rules) if (std::regex_match(op, std::regex(r.pattern))) return r.mode;
    return LinearPrecision::fp16;
}
void DevicePrefill::set_linear_policy(const LinearPolicy& policy) {
    auto& b = *impl_;
    bool split = false;
    for (const auto& r : policy.rules) split |= r.mode == LinearPrecision::hilo || r.mode == LinearPrecision::hilo_k128;
    if (split && !b.stats.prepared) throw std::invalid_argument("hilo linear precision needs prepared WMMA panels");
    // High and low activation panels share the conversion scratch.
    if (split != b.split_panels) { b.split_panels = split; b.size_rows(b.rows_capacity); }
    b.policy = policy; b.precisions.clear();
    std::string text;
    for (const auto& r : policy.rules) {
        static const char* names[] = {"fp16", "exact", "hilo", "hilo_k128"};
        text += (text.empty() ? "" : ";") + r.pattern + "=" + names[unsigned(r.mode)];
    }
    b.stats.linear_policy = text;
}
std::vector<float> DevicePrefill::last_logits() {
    auto& b = *impl_; const auto& w = b.w; const auto& c = w.config;
    // Same head instruction sequence on the saved final-norm row; its range is
    // recorded after the deferring chunk's ranges, whose maxima slots are free.
    const auto index = b.ranges.size();
    b.queue_io = queue_io_enabled();
    b.clear(b.failure);
    if (b.logits.size() < std::size_t(c.vocab) * sizeof(float)) b.logits.resize(std::size_t(c.vocab) * sizeof(float));
    b.dot("head", w.head.data ? w.head : w.embedding, b.last_row.as<float>(), b.logits.as<float>(), 1);
    gpu::cap_logits(b.logits.as<float>(), c.vocab, b.queue); b.finite(b.logits, c.vocab);
    gpu::synchronize(b.queue);
    unsigned failure = 0, maximum = 0;
    b.get(&failure, b.failure, sizeof(failure)); b.get(&maximum, b.maximum, sizeof(maximum), index * sizeof(unsigned));
    const float observed = bit_cast<float>(maximum);
    auto& high = b.range_maxima[b.ranges[index].tensor];
    high = !std::isfinite(observed) ? observed : std::max(high, observed);
    b.ranges[index].input_maximum = high;
    if (failure) throw std::runtime_error("prefill head failed: nonfinite arithmetic or FP16 overflow (mask " + std::to_string(failure) + ")");
    std::vector<float> out(c.vocab); b.get(out.data(), b.logits, out.size() * sizeof(float));
    return out;
}
Result DevicePrefill::run(gpu::DeviceStream& stream, std::span<const unsigned> tokens, DraftContext* context) {
    Prompt prompt(*this, stream, context); prompt.append(tokens); return prompt.finish();
}
Prompt::Prompt(DevicePrefill& prefill, gpu::DeviceStream& stream, DraftContext* context)
    : prefill_(prefill), stream_(stream), context_(context) { result_.frontier = stream.frontier(); }
void Prompt::flush(bool final) {
    if (pending_.empty()) return;
    if (stream_.frontier() != result_.frontier) throw std::logic_error("prefill stream changed while buffering");
    auto tx = stream_.begin();
    const auto start = tx->frontier();
    auto out = prefill_.forward(*tx, pending_, false, final || !defer_head_enabled());
    // All values have passed the failure fence before any live cache is touched.
    tx->commit(pending_.size());
    Result chunk; chunk.frontier = stream_.frontier(); chunk.tail_taps = std::move(out.taps);
    for (unsigned r = 0; r < pending_.size(); ++r) chunk.tail_positions.push_back(start + r);
    if (context_) append_tail(*context_, chunk, out.tap_width / 5);
    result_.last_logits = std::move(out.last_logits);
    result_.tail_taps.insert(result_.tail_taps.end(), chunk.tail_taps.begin(), chunk.tail_taps.end());
    result_.tail_positions.insert(result_.tail_positions.end(), chunk.tail_positions.begin(), chunk.tail_positions.end());
    const auto excess = result_.tail_positions.size() > 1024 ? result_.tail_positions.size() - 1024 : 0;
    result_.tail_positions.erase(result_.tail_positions.begin(), result_.tail_positions.begin() + std::ptrdiff_t(excess));
    result_.tail_taps.erase(result_.tail_taps.begin(), result_.tail_taps.begin() + std::ptrdiff_t(excess * out.tap_width));
    result_.frontier = stream_.frontier(); pending_.clear();
}
void Prompt::append(std::span<const unsigned> tokens) {
    if (finished_ || failed_) throw std::logic_error("prefill prompt is closed");
    try {
        if (stream_.frontier() != result_.frontier) throw std::logic_error("prefill stream changed while buffering");
        (void)plan(stream_.frontier(), pending_.size() + tokens.size());
        while (!tokens.empty()) {
            const auto capacity = chunk_rows - unsigned(stream_.frontier() % chunk_rows);
            const auto count = std::min<std::size_t>(tokens.size(), capacity - pending_.size());
            pending_.insert(pending_.end(), tokens.begin(), tokens.begin() + std::ptrdiff_t(count)); tokens = tokens.subspan(count);
            if (pending_.size() == capacity) flush(false);
        }
    } catch (...) { failed_ = true; throw; }
}
Result Prompt::finish() {
    if (finished_ || failed_) throw std::logic_error("prefill prompt is closed");
    try {
        if (stream_.frontier() != result_.frontier) throw std::logic_error("prefill stream changed while buffering");
        flush(true);
        // A prompt ending exactly on a chunk boundary deferred its head.
        if (result_.last_logits.empty() && !result_.tail_positions.empty()) result_.last_logits = prefill_.last_logits();
        finished_ = true; return std::move(result_);
    }
    catch (...) { failed_ = true; throw; }
}
} // namespace strix::gemma4::prefill
