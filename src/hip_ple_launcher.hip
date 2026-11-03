#include "strix/hip_launch_context.hpp"
#include "strix/hip_ggml_weight_formats.hpp"
#include "strix/known_prefill_limits.hpp"
#include "strix/hip_ple_launcher.hpp"

#include "strix/dispatch_resources.hpp"
#include "strix/prepared_tensor_views.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/ple_reclaim.hpp"
#include "strix/ple_staged_gather.hpp"
#include "strix/iq4_nl.hpp"
#include "strix/ple_workspace.hpp"
#include "strix/graph_ple.hpp"
#include "hip_q8_rows_gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include <string>
#include <string_view>
#include <utility>
#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace strix {
// STRIX_PLE_WINDOW_WARM (default on; =0, STRIX_QPF2K_DEFAULTS=0 or STRIX_BASELINE=1 turn it off): decode cache
// warming for the next verify window's PLE rows. The MTP draft loop reports
// each window row's input token as soon as the host knows it (the pending token when the draft chain starts, each
// draft after its step); the row's n-gram table rows are hashed from the last published history exactly as launch()
// hashes them and a worker loads their pages into the rolling-residency decode cache (Turn::warm). No bytes are
// handed to the window: its launch() runs unchanged and its leased gather finds the pages cached. A wrong guess
// (stale history, rejected or padded rows, a min-p stop) only leaves warm pages that the decode quota evicts.
namespace ple_window_warm {
namespace {
struct Warmer {
    std::mutex mutex;
    std::condition_variable wake;
    std::thread thread;
    bool stop = false;
    // Registered by the last verify launch() (source) and the last history publication (history).
    std::weak_ptr<ple_reclaim::ResidencyManager::Turn> turn;
    std::uintptr_t table = 0;
    std::uint64_t generation = 0;
    PleHashContract contract{};
    bool source = false, history_valid = false;
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1> history{};
    // The cycle being warmed: row ids of rows [0, rows), queued rows not yet loaded.
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1> previous{};
    std::array<std::uint64_t, 8U * PleHashContract::maximum_heads> ids{};
    std::uint32_t rows = 0, queued_from = 0, heads = 0;
    bool cycle = false;
    unsigned jobs = 0;   // queued or running warm calls
    std::chrono::steady_clock::time_point warmed_at;
    // STRIX_CYCLE_TIMING counters.
    unsigned long long cycles = 0, warmed_rows = 0, row_reads = 0, pages_read = 0, errors = 0, windows = 0,
        window_rows = 0, matched_rows = 0, pending_at_window = 0;
    double warm_ms = 0, ahead_ms = 0;
    unsigned long long ahead_n = 0;
    bool timing = std::getenv("STRIX_CYCLE_TIMING") != nullptr;
    ~Warmer() {
        {
            const std::lock_guard lock(mutex);
            stop = true;
        }
        wake.notify_all();
        if (thread.joinable()) thread.join();
        if (!timing || !cycles) return;
        std::fprintf(stderr, "window PLE warm: %llu cycles, rows warmed %llu, row reads %llu, pages read %llu, warm %.4f ms/cycle, "
            "errors %llu; %llu windows: rows %llu matched %llu, warm done %.4f ms before window start (%llu windows), "
            "warm still running at window start %llu\n",
            cycles, warmed_rows, row_reads, pages_read, warm_ms / static_cast<double>(cycles), errors, windows, window_rows,
            matched_rows, ahead_n ? ahead_ms / static_cast<double>(ahead_n) : 0.0, ahead_n, pending_at_window);
    }
    // Caller holds mutex: hashes the cycle's next row and queues it.
    void add(std::uint32_t token) {
        if (!cycle || rows >= 8U) return;
        const auto context = static_cast<std::size_t>(contract.ngram_size - 1U);
        const auto hash = hash_ple_token(contract, token, std::span<const std::uint32_t>(previous.data(), context));
        if (hash.row_count != heads) { cycle = false; return; }
        std::copy_n(hash.row_ids.begin(), heads, ids.begin() + static_cast<std::size_t>(rows) * heads);
        std::copy_n(hash.next_context.begin(), context, previous.begin());
        ++rows;
        if (!thread.joinable()) thread = std::thread([this] { run(); });
        wake.notify_one();
    }
    void run() noexcept {
        std::unique_lock lock(mutex);
        for (;;) {
            wake.wait(lock, [this] { return stop || (cycle && queued_from < rows); });
            if (stop) return;
            const std::vector<std::uint64_t> batch(ids.begin() + static_cast<std::size_t>(queued_from) * heads,
                ids.begin() + static_cast<std::size_t>(rows) * heads);
            const auto count = rows - queued_from;
            queued_from = rows;
            const auto owner = turn.lock();
            const auto address = table;
            const auto gen = generation;
            ++jobs;
            lock.unlock();
            std::uint64_t read = 0;
            bool failed = !owner;
            const auto t0 = std::chrono::steady_clock::now();
            if (owner) {
                try { read = owner->warm(address, gen, batch); } catch (...) { failed = true; }
            }
            const auto t1 = std::chrono::steady_clock::now();
            lock.lock();
            --jobs;
            warmed_at = t1;
            warmed_rows += count; row_reads += batch.size(); pages_read += read; errors += failed ? 1U : 0U;
            warm_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }
};
Warmer& warmer() {
    static Warmer instance;
    return instance;
}
}  // namespace

bool enabled() noexcept {
    static const bool on = [] {
        if (const char* v = std::getenv("STRIX_PLE_WINDOW_WARM")) return std::strcmp(v, "0") != 0;
        const char* defaults = std::getenv("STRIX_QPF2K_DEFAULTS");
        const char* baseline = std::getenv("STRIX_BASELINE");
        return !(defaults && defaults[0] == '0') && !(baseline && std::strcmp(baseline, "1") == 0);
    }();
    return on;
}
void begin(std::uint32_t token) noexcept {
    if (!enabled()) return;
    auto& w = warmer();
    const std::lock_guard lock(w.mutex);
    try {
        w.cycle = w.source && w.history_valid && !w.turn.expired();
        if (!w.cycle) return;
        w.previous = w.history; w.rows = 0; w.queued_from = 0; w.heads = w.contract.head_count;
        if (!w.heads || w.heads > PleHashContract::maximum_heads) { w.cycle = false; return; }
        ++w.cycles;
        w.add(token);
    } catch (...) { w.cycle = false; }
}
void draft(std::uint32_t step, std::uint32_t token) noexcept {
    if (!enabled()) return;
    auto& w = warmer();
    const std::lock_guard lock(w.mutex);
    try {
        if (w.cycle && step + 1U == w.rows) w.add(token);
    } catch (...) { w.cycle = false; }
}
namespace {
// launch(): a verify window starts; compares its row ids with the warmed ones and ends the cycle.
void window(std::span<const PleHashResult> hashes, std::chrono::steady_clock::time_point start) noexcept {
    if (!enabled()) return;
    auto& w = warmer();
    const std::lock_guard lock(w.mutex);
    if (!w.cycle) return;
    ++w.windows; w.window_rows += hashes.size();
    for (std::size_t r = 0; r < hashes.size() && r < w.rows; ++r)
        if (hashes[r].row_count == w.heads &&
            std::equal(hashes[r].row_ids.begin(), hashes[r].row_ids.begin() + w.heads, w.ids.begin() + r * w.heads)) ++w.matched_rows;
    if (w.jobs || w.queued_from < w.rows) ++w.pending_at_window;
    else if (w.queued_from) {
        w.ahead_ms += std::chrono::duration<double, std::milli>(start - w.warmed_at).count(); ++w.ahead_n;
    }
    w.cycle = false;
}
void register_source(const std::shared_ptr<ple_reclaim::ResidencyManager::Turn>& turn, const RuntimeTensorBinding& table,
    const PleHashContract& contract) noexcept {
    if (!enabled()) return;
    auto& w = warmer();
    const std::lock_guard lock(w.mutex);
    w.source = static_cast<bool>(turn);
    if (!turn) return;
    w.turn = turn; w.table = table.address; w.generation = table.allocation_generation; w.contract = contract;
}
void register_history(const PleHashResult& hash) noexcept {
    if (!enabled()) return;
    auto& w = warmer();
    const std::lock_guard lock(w.mutex);
    w.history_valid = hash.context_count && hash.context_count <= w.history.size();
    std::copy_n(hash.next_context.begin(), std::min<std::size_t>(hash.context_count, w.history.size()), w.history.begin());
}
}  // namespace
}  // namespace ple_window_warm

namespace {

constexpr std::uint32_t kThreads = 256;
constexpr std::uint64_t kLauncherTag =
    0x504c454c41554e32ULL;
constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

struct Q8Block {
    __half scale;
    std::int8_t values[32];
};

struct IQ4XSBlock {
    __half d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};

static_assert(sizeof(Q8Block) == 34);
static_assert(sizeof(IQ4XSBlock) == 136);

__device__ __constant__
std::int8_t iq4_codebook[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
       1,   13,  25,  38,
      53,   69,  89, 113,
};

enum class WeightKind {
    f32,
    f16,
    bf16,
    q8_0,
    iq4_xs,
#define STRIX_GGML_KIND_ENUM(kind, type) kind,
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_ENUM) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_ENUM) STRIX_GGML_KIND_ENUM(iq4_nl, 20)
#undef STRIX_GGML_KIND_ENUM
};

enum class SmallWeightKind {
    f32,
    f16,
    bf16,
};

struct TensorPair {
    const ManifestTensor* manifest = nullptr;
    const RuntimeTensorBinding* runtime = nullptr;
};

struct PleTensors {
    TensorPair table;
    TensorPair key;
    TensorPair value;
    TensorPair norm_key;
    TensorPair norm_query;
    TensorPair norm_conv;
    TensorPair conv;
};

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

void validate_config(
    const HipPleLauncherConfig& config) {
    if (const auto error =
            config.model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    if (config.activation !=
        HipActivationStorage::bf16) {
        throw std::invalid_argument(
            "HIP PLE v1 targets canonical BF16 activations only");
    }
    if (config.model.ple_head_count() != 16 ||
        config.model.ple_head_dim() != 160 ||
        config.model.ple_conv_kernel_size != 4 ||
        config.model.ngram_size != 3 ||
        config.model.hc_count != 4) {
        throw std::invalid_argument(
            "HIP PLE v1 requires pinned Qwen4Exp PLE geometry");
    }
}

WeightKind weight_kind(
    std::uint32_t type) {
    switch (type) {
    case 0: return WeightKind::f32;
    case 1: return WeightKind::f16;
    case 8: return WeightKind::q8_0;
    case 23: return WeightKind::iq4_xs;
#define STRIX_GGML_KIND_MAP(kind, type) case type: return WeightKind::kind;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_MAP) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_MAP) STRIX_GGML_KIND_MAP(iq4_nl, 20)
#undef STRIX_GGML_KIND_MAP
    case 30: return WeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP PLE projection supports F32, F16, BF16, Q8_0 and IQ4_XS");
    }
}

SmallWeightKind small_weight_kind(
    std::uint32_t type) {
    switch (type) {
    case 0: return SmallWeightKind::f32;
    case 1: return SmallWeightKind::f16;
    case 30: return SmallWeightKind::bf16;
    default:
        throw std::invalid_argument(
            "HIP PLE norm/conv supports F32, F16 and BF16");
    }
}

__device__ float bf16_to_float(
    std::uint16_t bits) {
    return __uint_as_float(
        static_cast<std::uint32_t>(
            bits) << 16U);
}

__device__ std::uint16_t float_to_bf16(
    float value) {
    const auto bits =
        __float_as_uint(value);
    const auto rounded =
        bits +
        0x7fffU +
        ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(
        rounded >> 16U);
}

__device__ std::uint8_t iq4_local_scale(
    const IQ4XSBlock& block,
    std::uint32_t subblock) {
    const auto packed =
        block.scales_l[
            subblock / 2U];
    const auto low =
        static_cast<std::uint8_t>(
            (packed >>
             (4U *
              (subblock % 2U))) &
            0x0fU);
    const auto high =
        static_cast<std::uint8_t>(
            (block.scales_h >>
             (2U * subblock)) &
            0x03U);
    return static_cast<std::uint8_t>(
        low |
        static_cast<std::uint8_t>(
            high << 4U));
}

template <WeightKind Kind>
__device__ float load_weight(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature);

template <>
__device__ float load_weight<WeightKind::f32>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return static_cast<const float*>(
        weights)[
        row * row_width +
        feature];
}

template <>
__device__ float load_weight<WeightKind::f16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return __half2float(
        static_cast<const __half*>(
            weights)[
            row * row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::bf16>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    return bf16_to_float(
        static_cast<
            const std::uint16_t*>(
                weights)[
            row * row_width +
            feature]);
}

template <>
__device__ float load_weight<WeightKind::q8_0>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const Q8Block*>(
            weights);
    const auto blocks_per_row =
        row_width / 32U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
            feature / 32U];
    return __half2float(
               block.scale) *
           static_cast<float>(
               block.values[
                   feature % 32U]);
}

template <>
__device__ float load_weight<WeightKind::iq4_xs>(
    const void* weights,
    std::uint32_t row_width,
    std::uint64_t row,
    std::uint32_t feature) {
    const auto* blocks =
        static_cast<const IQ4XSBlock*>(
            weights);
    const auto blocks_per_row =
        row_width / 256U;
    const auto& block =
        blocks[
            row *
                blocks_per_row +
            feature / 256U];

    const auto within =
        feature % 256U;
    const auto subblock =
        within / 32U;
    const auto suboffset =
        within % 32U;
    const auto packed =
        block.qs[
            subblock * 16U +
            suboffset % 16U];
    const auto code =
        suboffset < 16U
            ? static_cast<std::uint8_t>(
                  packed & 0x0fU)
            : static_cast<std::uint8_t>(
                  packed >> 4U);
    const int local =
        static_cast<int>(
            iq4_local_scale(
                block,
                subblock)) -
        32;

    return __half2float(block.d) *
           static_cast<float>(local) *
           static_cast<float>(
               iq4_codebook[code]);
}

__device__ float load_small(
    const void* values,
    SmallWeightKind kind,
    std::uint64_t index) {
    switch (kind) {
    case SmallWeightKind::f32:
        return static_cast<
            const float*>(values)[index];
    case SmallWeightKind::f16:
        return __half2float(
            static_cast<
                const __half*>(values)[index]);
    case SmallWeightKind::bf16:
        return bf16_to_float(
            static_cast<
                const std::uint16_t*>(
                    values)[index]);
    }
    return 0.0F;
}

#define STRIX_GGML_KIND_LOAD(kind, type)                                                          \
    template <>                                                                                   \
    __device__ float load_weight<WeightKind::kind>(                                               \
        const void* weights, std::uint32_t row_width, std::uint64_t row, std::uint32_t feature) { \
        return hip_ggml::element<type>(weights, row_width, row, feature);                        \
    }
STRIX_GGML_KINDS_K(STRIX_GGML_KIND_LOAD) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_LOAD) STRIX_GGML_KIND_LOAD(iq4_nl, 20)
#undef STRIX_GGML_KIND_LOAD

std::uint32_t ggml_type_of(WeightKind kind) {
    switch (kind) {
#define STRIX_GGML_KIND_TYPE(k, type) case WeightKind::k: return type;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_TYPE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_TYPE) STRIX_GGML_KIND_TYPE(iq4_nl, 20)
#undef STRIX_GGML_KIND_TYPE
    default: return 0;
    }
}

std::uint64_t required_weight_bytes(
    WeightKind kind,
    std::uint32_t input_width,
    std::uint64_t output_rows) {
    switch (kind) {
    case WeightKind::f32:
        return
            static_cast<std::uint64_t>(
                input_width) *
            output_rows * 4ULL;
    case WeightKind::f16:
    case WeightKind::bf16:
        return
            static_cast<std::uint64_t>(
                input_width) *
            output_rows * 2ULL;
    case WeightKind::q8_0:
        if (input_width % 32U != 0) {
            throw std::invalid_argument(
                "Q8_0 PLE projection width must divide 32");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 32U) *
            output_rows *
            sizeof(Q8Block);
    case WeightKind::iq4_xs:
        if (input_width % 256U != 0) {
            throw std::invalid_argument(
                "IQ4_XS PLE projection width must divide 256");
        }
        return
            static_cast<std::uint64_t>(
                input_width / 256U) *
            output_rows *
            sizeof(IQ4XSBlock);
    default: {
        const auto g = hip_ggml::geometry(ggml_type_of(kind));
        if (g.elements == 0 || input_width % g.elements != 0) {
            throw std::invalid_argument("weight width is not a multiple of its GGUF block");
        }
        return static_cast<std::uint64_t>(input_width / g.elements) * output_rows * g.bytes;
    }
    }
    throw std::logic_error(
        "unreachable PLE weight kind");
}

std::uint64_t small_weight_bytes(
    SmallWeightKind kind,
    std::uint64_t elements) {
    switch (kind) {
    case SmallWeightKind::f32:
        return elements * 4ULL;
    case SmallWeightKind::f16:
    case SmallWeightKind::bf16:
        return elements * 2ULL;
    }
    throw std::logic_error(
        "unreachable PLE small weight kind");
}

template <WeightKind Kind>
__global__ void project_bf16_to_fp32_kernel(
    const std::uint16_t* input,
    const void* weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    const auto out =
        static_cast<std::uint32_t>(
            blockIdx.x);
    if (out >= output_width) {
        return;
    }

    __shared__ float scratch[kThreads];
    float sum = 0.0F;
    for (std::uint32_t feature =
             static_cast<std::uint32_t>(
                 threadIdx.x);
         feature < input_width;
         feature += kThreads) {
        sum +=
            bf16_to_float(
                input[feature]) *
            load_weight<Kind>(
                weights,
                input_width,
                out,
                feature);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] +=
                scratch[
                    threadIdx.x +
                    stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out] =
            scratch[0];
    }
}

// Same 256-leaf DAG as project_bf16_to_fp32_kernel<q8_0> inside one wave per row:
// leaf t=8*lane+local accumulates features t, t+256, ... in order, then the
// LDS tree (strides 128..8 are lane shuffles 16..1, strides 4..1 are local).
__global__ __launch_bounds__(256) void project_wave8_q8_bf16_to_fp32_kernel(
    const std::uint16_t* input, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) + static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint32_t chunk = 0; chunk < input_width; chunk += kThreads) {
        const auto feature = chunk + 8U * lane;
        if (feature < input_width) {
            const auto& block = blocks[feature / 32U];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                partial[local] += bf16_to_float(input[feature + local]) *
                    (__half2float(block.scale) * static_cast<float>(block.values[feature % 32U + local]));
        }
    }
#pragma unroll
    for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned local = 0; local < 8; ++local) {
            const auto partner = __shfl_down(partial[local], distance, 32);
            if (lane < distance) partial[local] += partner;
        }
    }
    if (lane == 0) {
        partial[0] += partial[4]; partial[1] += partial[5];
        partial[2] += partial[6]; partial[3] += partial[7];
        partial[0] += partial[2]; partial[1] += partial[3];
        partial[0] += partial[1];
        output[row] = partial[0];
    }
}

// project_wave8_q8_bf16_to_fp32_kernel for Rows inputs at once: each Q8 block is decoded once and
// applied to every row with that row's own leaf accumulation and tree (bit-identical per row).
template <unsigned Rows>
__global__ __launch_bounds__(256) void project_wave8_q8_bf16_to_fp32_rows_kernel(
    const std::uint16_t* input, std::uint32_t input_stride, const void* weights,
    std::uint32_t input_width, std::uint32_t output_width, float* output, std::uint32_t output_stride) {
    const auto row = static_cast<std::uint32_t>(blockIdx.x) * 8U + static_cast<std::uint32_t>(threadIdx.x) / 32U;
    if (row >= output_width) return;
    const auto lane = static_cast<std::uint32_t>(threadIdx.x) % 32U;
    float partial[Rows][8]{};
    const auto* blocks = static_cast<const Q8Block*>(weights) + static_cast<std::uint64_t>(row) * (input_width / 32U);
    for (std::uint32_t chunk = 0; chunk < input_width; chunk += kThreads) {
        const auto feature = chunk + 8U * lane;
        if (feature < input_width) {
            const auto& block = blocks[feature / 32U];
            float weight[8];
#pragma unroll
            for (unsigned local = 0; local < 8; ++local)
                weight[local] = __half2float(block.scale) * static_cast<float>(block.values[feature % 32U + local]);
#pragma unroll
            for (unsigned r = 0; r < Rows; ++r)
#pragma unroll
                for (unsigned local = 0; local < 8; ++local)
                    partial[r][local] += bf16_to_float(input[r * input_stride + feature + local]) * weight[local];
        }
    }
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) {
#pragma unroll
        for (unsigned distance = 16; distance > 0; distance /= 2) {
#pragma unroll
            for (unsigned local = 0; local < 8; ++local) {
                const auto partner = __shfl_down(partial[r][local], distance, 32);
                if (lane < distance) partial[r][local] += partner;
            }
        }
        if (lane == 0) {
            auto* p = partial[r];
            p[0] += p[4]; p[1] += p[5];
            p[2] += p[6]; p[3] += p[7];
            p[0] += p[2]; p[1] += p[3];
            p[0] += p[1];
            output[static_cast<std::uint64_t>(r) * output_stride + row] = p[0];
        }
    }
}

__global__ void normalize_key_and_gate_kernel(
    float* key,
    const std::uint16_t* query,
    const void* key_norm,
    SmallWeightKind key_norm_kind,
    const void* query_norm,
    SmallWeightKind query_norm_kind,
    float* gates,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    float epsilon) {
    const auto stream =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (stream >= hc_count) {
        return;
    }

    __shared__ float key_reduce[kThreads];
    __shared__ float query_reduce[kThreads];
    __shared__ float inverse_key;
    __shared__ float inverse_query;

    float key_square = 0.0F;
    float query_square = 0.0F;
    const auto base =
        static_cast<std::uint64_t>(
            stream) *
        hidden_size;

    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += kThreads) {
        const float k =
            key[base + feature];
        const float q =
            bf16_to_float(
                query[base + feature]);
        key_square += k * k;
        query_square += q * q;
    }

    key_reduce[lane] =
        key_square;
    query_reduce[lane] =
        query_square;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            key_reduce[lane] +=
                key_reduce[
                    lane + stride];
            query_reduce[lane] +=
                query_reduce[
                    lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        inverse_key =
            rsqrtf(
                key_reduce[0] /
                    static_cast<float>(
                        hidden_size) +
                epsilon);
        inverse_query =
            rsqrtf(
                query_reduce[0] /
                    static_cast<float>(
                        hidden_size) +
                epsilon);
    }
    __syncthreads();

    float dot = 0.0F;
    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += kThreads) {
        const auto index =
            base + feature;
        const float normalized_key =
            key[index] *
            inverse_key *
            load_small(
                 key_norm,
                 key_norm_kind,
                 index);
        key[index] =
            normalized_key;

        const float normalized_query =
            bf16_to_float(
                query[index]) *
            inverse_query *
            load_small(
                 query_norm,
                 query_norm_kind,
                 index);
        dot +=
            normalized_key *
            normalized_query;
    }

    key_reduce[lane] = dot;
    __syncthreads();
    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            key_reduce[lane] +=
                key_reduce[
                    lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        const float score =
            key_reduce[0] /
            sqrtf(
                static_cast<float>(
                    hidden_size));
        const float magnitude =
            sqrtf(
                fmaxf(
                    fabsf(score),
                    1e-6F));
        const float signed_root =
            score > 0.0F
                ? magnitude
                : score < 0.0F
                    ? -magnitude
                    : 0.0F;
        gates[stream] =
            1.0F /
            (1.0F +
             expf(-signed_root));
    }
}

__global__ void gated_value_kernel(
    const float* value,
    const float* gates,
    float* gated,
    std::uint32_t hidden_size,
    std::uint32_t hc_count) {
    const auto index =
        static_cast<std::uint64_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    const auto total =
        static_cast<std::uint64_t>(
            hidden_size) *
        hc_count;
    if (index >= total) {
        return;
    }
    const auto stream =
        static_cast<std::uint32_t>(
            index /
            hidden_size);
    const auto feature =
        static_cast<std::uint32_t>(
            index %
            hidden_size);
    gated[index] =
        value[feature] *
        gates[stream];
}

__global__ void grouped_norm_kernel(
    const float* input,
    const void* norm_weight,
    SmallWeightKind norm_kind,
    float* output,
    std::uint32_t hidden_size,
    std::uint32_t hc_count,
    float epsilon) {
    const auto stream =
        static_cast<std::uint32_t>(
            blockIdx.x);
    const auto lane =
        static_cast<std::uint32_t>(
            threadIdx.x);
    if (stream >= hc_count) {
        return;
    }

    __shared__ float reduction[kThreads];
    __shared__ float inverse_rms;

    const auto base =
        static_cast<std::uint64_t>(
            stream) *
        hidden_size;
    float square = 0.0F;
    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += kThreads) {
        const float value =
            input[base + feature];
        square +=
            value * value;
    }
    reduction[lane] = square;
    __syncthreads();

    for (std::uint32_t stride =
             kThreads / 2U;
         stride > 0;
         stride >>= 1U) {
        if (lane < stride) {
            reduction[lane] +=
                reduction[
                    lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        inverse_rms =
            rsqrtf(
                reduction[0] /
                    static_cast<float>(
                        hidden_size) +
                epsilon);
    }
    __syncthreads();

    for (std::uint32_t feature = lane;
         feature < hidden_size;
         feature += kThreads) {
        const auto index =
            base + feature;
        output[index] =
            input[index] *
            inverse_rms *
            load_small(
                 norm_weight,
                 norm_kind,
                 index);
    }
}

__global__ void conv_residual_kernel(
    const std::uint16_t* hidden,
    const float* gated,
    const float* normalized,
    std::uint16_t* conv_state,
    const void* conv_weight,
    SmallWeightKind conv_kind,
    std::uint16_t* output,
    std::uint32_t channels,
    std::uint32_t history,
    std::uint32_t kernel_size,
    std::uint32_t dilation, std::uint32_t* graph_status = nullptr) {
    if (graph_status && atomicAdd(graph_status, 0U) != 0U) return;
    const auto channel =
        static_cast<std::uint32_t>(
            blockIdx.x) *
            blockDim.x +
        threadIdx.x;
    if (channel >= channels) {
        return;
    }

    auto* state =
        conv_state +
        static_cast<std::uint64_t>(
            channel) *
        history;

    float conv = 0.0F;
    for (std::uint32_t tap = 0;
         tap < kernel_size;
         ++tap) {
        const auto lag =
            (kernel_size - 1U - tap) *
            dilation;
        const float value =
            lag == 0
                ? normalized[channel]
                : bf16_to_float(
                      state[
                          history -
                          lag]);
        conv +=
            value *
            load_small(
                conv_weight,
                conv_kind,
                static_cast<std::uint64_t>(
                    channel) *
                    kernel_size +
                tap);
    }

    for (std::uint32_t index = 0;
         index + 1U < history;
         ++index) {
        state[index] =
            state[index + 1U];
    }
    state[history - 1U] =
        float_to_bf16(
            normalized[channel]);

    const float silu =
        conv /
        (1.0F +
         expf(-conv));
    output[channel] =
        float_to_bf16(
            bf16_to_float(
                hidden[channel]) +
            gated[channel] +
            silu);
}

// Rows form of the per-row PLE tail for the T rows of a prefill block (blockIdx.y is the row).
// Each row keeps the per-row kernels' arithmetic: the same per-lane strided sums, LDS trees,
// gate, gated product, norm and convolution tap order. The convolution state of every row is
// the bf16 of the previous rows' normalized values, so a row's taps read either the old state
// (lag beyond the block's start) or the bf16 of an earlier row's normalized value; the state
// after the block is written by a separate kernel once every row has read the old state.
__device__ __forceinline__ float ple_rows_exact_mul(float a, float b) {
    float result;  // a plain product, never contracted into the sum that consumes it
    asm("v_mul_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
// Gate (normalize_key_and_gate_kernel) and grouped norm of the gated value (gated_value_kernel,
// grouped_norm_kernel) of one row's stream; the normalized values overwrite the row's key, which
// the per-row path does not read after the gate.
__global__ void ple_rows_gate_norm_kernel(
    float* key, std::uint32_t key_stride,
    const float* value, std::uint32_t value_stride,
    const std::uint16_t* query, std::uint32_t query_stride,
    const void* key_norm, SmallWeightKind key_norm_kind,
    const void* query_norm, SmallWeightKind query_norm_kind,
    const void* conv_norm, SmallWeightKind conv_norm_kind,
    float* gates, std::uint32_t hidden_size, std::uint32_t hc_count, float epsilon) {
    const auto stream = static_cast<std::uint32_t>(blockIdx.x);
    const auto row = static_cast<std::uint32_t>(blockIdx.y);
    const auto lane = static_cast<std::uint32_t>(threadIdx.x);
    if (stream >= hc_count) return;
    key += static_cast<std::uint64_t>(row) * key_stride;
    value += static_cast<std::uint64_t>(row) * value_stride;
    query += static_cast<std::uint64_t>(row) * query_stride;

    __shared__ float key_reduce[kThreads];
    __shared__ float query_reduce[kThreads];
    __shared__ float inverse_key;
    __shared__ float inverse_query;
    __shared__ float gate;
    __shared__ float inverse_rms;

    float key_square = 0.0F;
    float query_square = 0.0F;
    const auto base = static_cast<std::uint64_t>(stream) * hidden_size;
    for (std::uint32_t feature = lane; feature < hidden_size; feature += kThreads) {
        const float k = key[base + feature];
        const float q = bf16_to_float(query[base + feature]);
        key_square += k * k;
        query_square += q * q;
    }
    key_reduce[lane] = key_square;
    query_reduce[lane] = query_square;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) {
            key_reduce[lane] += key_reduce[lane + stride];
            query_reduce[lane] += query_reduce[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        inverse_key = rsqrtf(key_reduce[0] / static_cast<float>(hidden_size) + epsilon);
        inverse_query = rsqrtf(query_reduce[0] / static_cast<float>(hidden_size) + epsilon);
    }
    __syncthreads();

    float dot = 0.0F;
    for (std::uint32_t feature = lane; feature < hidden_size; feature += kThreads) {
        const auto index = base + feature;
        const float normalized_key = key[index] * inverse_key * load_small(key_norm, key_norm_kind, index);
        const float normalized_query =
            bf16_to_float(query[index]) * inverse_query * load_small(query_norm, query_norm_kind, index);
        dot += normalized_key * normalized_query;
    }
    key_reduce[lane] = dot;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) key_reduce[lane] += key_reduce[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        const float score = key_reduce[0] / sqrtf(static_cast<float>(hidden_size));
        const float magnitude = sqrtf(fmaxf(fabsf(score), 1e-6F));
        const float signed_root = score > 0.0F ? magnitude : score < 0.0F ? -magnitude : 0.0F;
        gate = 1.0F / (1.0F + expf(-signed_root));
        gates[static_cast<std::uint64_t>(row) * hc_count + stream] = gate;
    }
    __syncthreads();

    const float g = gate;
    float square = 0.0F;
    for (std::uint32_t feature = lane; feature < hidden_size; feature += kThreads) {
        const float gated = ple_rows_exact_mul(value[feature], g);
        square += gated * gated;
    }
    key_reduce[lane] = square;
    __syncthreads();
    for (std::uint32_t stride = kThreads / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) key_reduce[lane] += key_reduce[lane + stride];
        __syncthreads();
    }
    if (lane == 0) inverse_rms = rsqrtf(key_reduce[0] / static_cast<float>(hidden_size) + epsilon);
    __syncthreads();
    for (std::uint32_t feature = lane; feature < hidden_size; feature += kThreads) {
        const auto index = base + feature;
        const float gated = ple_rows_exact_mul(value[feature], g);
        key[index] = gated * inverse_rms * load_small(conv_norm, conv_norm_kind, index);
    }
}

// conv_residual_kernel for every row of the block; the old state is only read.
__global__ void ple_rows_conv_residual_kernel(
    const std::uint16_t* hidden, std::uint32_t hidden_stride,
    const float* value, std::uint32_t value_stride,
    const float* gates,
    const float* normalized, std::uint32_t normalized_stride,
    const std::uint16_t* conv_state, const void* conv_weight, SmallWeightKind conv_kind,
    std::uint16_t* output, std::uint32_t output_stride,
    std::uint32_t hidden_size, std::uint32_t hc_count, std::uint32_t history, std::uint32_t kernel_size,
    std::uint32_t dilation, std::uint32_t* graph_status) {
    if (graph_status && atomicAdd(graph_status, 0U) != 0U) return;
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto row = static_cast<std::uint32_t>(blockIdx.y);
    if (channel >= hidden_size * hc_count) return;
    const auto* state = conv_state + static_cast<std::uint64_t>(channel) * history;

    float conv = 0.0F;
    for (std::uint32_t tap = 0; tap < kernel_size; ++tap) {
        const auto lag = (kernel_size - 1U - tap) * dilation;
        float value_at;
        if (lag == 0) value_at = normalized[static_cast<std::uint64_t>(row) * normalized_stride + channel];
        else if (row >= lag)
            value_at = bf16_to_float(float_to_bf16(
                normalized[static_cast<std::uint64_t>(row - lag) * normalized_stride + channel]));
        else value_at = bf16_to_float(state[history + row - lag]);
        conv += value_at *
            load_small(conv_weight, conv_kind, static_cast<std::uint64_t>(channel) * kernel_size + tap);
    }
    const auto stream = channel / hidden_size, feature = channel % hidden_size;
    const float gated = ple_rows_exact_mul(value[static_cast<std::uint64_t>(row) * value_stride + feature],
        gates[static_cast<std::uint64_t>(row) * hc_count + stream]);
    const float silu = conv / (1.0F + expf(-conv));
    output[static_cast<std::uint64_t>(row) * output_stride + channel] =
        float_to_bf16(bf16_to_float(hidden[static_cast<std::uint64_t>(row) * hidden_stride + channel]) + gated + silu);
}

// The convolution state after `rows` rows: the last `history` entries of old state ++ bf16(rows).
__global__ void ple_rows_conv_state_kernel(
    const float* normalized, std::uint32_t normalized_stride, std::uint16_t* conv_state,
    std::uint32_t channels, std::uint32_t history, std::uint32_t rows, std::uint32_t* graph_status) {
    if (graph_status && atomicAdd(graph_status, 0U) != 0U) return;
    const auto channel = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    auto* state = conv_state + static_cast<std::uint64_t>(channel) * history;
    for (std::uint32_t index = 0; index < history; ++index) {
        const auto source = rows + index;  // ascending: reads only entries not yet rewritten
        state[index] = source < history ? state[source]
            : float_to_bf16(normalized[static_cast<std::uint64_t>(source - history) * normalized_stride + channel]);
    }
}

// Arguments of the rows-form PLE tail (prefill rows and the test fixture).
struct PleTailRows {
    float* keys; std::uint32_t key_stride;
    const float* values; std::uint32_t value_stride;
    const std::uint16_t* input; std::uint32_t input_stride;
    std::uint16_t* output; std::uint32_t output_stride;
    std::uint16_t* conv_state;
    const void *norm_key, *norm_query, *norm_conv, *conv;
    SmallWeightKind norm_key_kind, norm_query_kind, norm_conv_kind, conv_kind;
    float* gates;
    std::uint32_t rows, hidden_size, hc_count, kernel_size, ngram_size;
    float epsilon;
};
void enqueue_ple_tail_rows(const PleTailRows& t, std::uint32_t* graph_status, hipStream_t stream) {
    const auto channels = t.hidden_size * t.hc_count;
    const auto history = (t.kernel_size - 1U) * t.ngram_size;
    hipLaunchKernelGGL(ple_rows_gate_norm_kernel, dim3(t.hc_count, t.rows), dim3(kThreads), 0, stream,
        t.keys, t.key_stride, t.values, t.value_stride, t.input, t.input_stride, t.norm_key, t.norm_key_kind,
        t.norm_query, t.norm_query_kind, t.norm_conv, t.norm_conv_kind, t.gates, t.hidden_size, t.hc_count, t.epsilon);
    check_hip(hipGetLastError(), "HIP PLE rows gate/norm launch");
    const auto blocks = (channels + kThreads - 1U) / kThreads;
    hipLaunchKernelGGL(ple_rows_conv_residual_kernel, dim3(blocks, t.rows), dim3(kThreads), 0, stream,
        t.input, t.input_stride, t.values, t.value_stride, t.gates, t.keys, t.key_stride, t.conv_state, t.conv,
        t.conv_kind, t.output, t.output_stride, t.hidden_size, t.hc_count, history, t.kernel_size, t.ngram_size,
        graph_status);
    check_hip(hipGetLastError(), "HIP PLE rows convolution launch");
    hipLaunchKernelGGL(ple_rows_conv_state_kernel, dim3(blocks), dim3(kThreads), 0, stream,
        t.keys, t.key_stride, t.conv_state, channels, history, t.rows, graph_status);
    check_hip(hipGetLastError(), "HIP PLE rows convolution state launch");
}

template <WeightKind Kind>
void launch_projection(
    const std::uint16_t* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            project_bf16_to_fp32_kernel<
                Kind>),
        dim3(output_width),
        dim3(kThreads),
        0, hip_execution_stream(),
        input,
        reinterpret_cast<
            const void*>(
                weights.address),
        input_width,
        output_width,
        output);
    check_hip(
        hipGetLastError(),
        "HIP PLE projection launch");
}

void launch_projection_by_kind(
    WeightKind kind,
    const std::uint16_t* input,
    const RuntimeTensorBinding& weights,
    std::uint32_t input_width,
    std::uint32_t output_width,
    float* output) {
    switch (kind) {
    case WeightKind::f32:
        launch_projection<
            WeightKind::f32>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
    case WeightKind::f16:
        launch_projection<
            WeightKind::f16>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
    case WeightKind::bf16:
        launch_projection<
            WeightKind::bf16>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
    case WeightKind::q8_0:
        if (input_width % 32U == 0U) {
            hipLaunchKernelGGL(project_wave8_q8_bf16_to_fp32_kernel, dim3((output_width + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), input, reinterpret_cast<const void*>(weights.address), input_width, output_width, output);
            check_hip(hipGetLastError(), "HIP PLE wave8 Q8 projection launch");
            return;
        }
        launch_projection<
            WeightKind::q8_0>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
    case WeightKind::iq4_xs:
        launch_projection<
            WeightKind::iq4_xs>(
                input,
                weights,
                input_width,
                output_width,
                output);
        return;
#define STRIX_GGML_KIND_CASE(KIND, type) \
    case WeightKind::KIND: \
        launch_projection< \
            WeightKind::KIND>( \
                input, \
                weights, \
                input_width, \
                output_width, \
                output); \
        return;
    STRIX_GGML_KINDS_K(STRIX_GGML_KIND_CASE) STRIX_GGML_KINDS_GSQ(STRIX_GGML_KIND_CASE) STRIX_GGML_KIND_CASE(iq4_nl, 20)
#undef STRIX_GGML_KIND_CASE
    }
}

bool contains(
    std::string_view text,
    std::string_view needle) {
    return text.find(needle) !=
        std::string_view::npos;
}

TensorPair tensor_pair(
    const ExecutionDispatchContext& context,
    std::uint32_t index) {
    if (context.resources == nullptr ||
        context.resources->
            model_manifest == nullptr) {
        throw std::invalid_argument(
            "HIP PLE requires model manifest");
    }
    if (index >=
        context.resources->
            model_manifest->
            tensors.size()) {
        throw std::out_of_range(
            "PLE tensor index exceeds manifest");
    }
    return TensorPair{
        &context.resources->
            model_manifest->
            tensors[index],
        &resolve_dispatch_tensor(
            context,
            index),
    };
}

PleTensors identify_tensors(
    const ExecutionDispatchContext& context,
    const HipPleLauncherConfig& config) {
    PleTensors result;

    const auto hc_hidden=config.model.hidden_size*config.model.hc_count;
    if(const auto* cached=context.prepared_tensor_views) {
        if(!context.resources) throw std::invalid_argument("prepared PLE roles require retained resources");
        cached->validate_stage_scope(context.resources->distributed_stage,context.step_id);
        cached->validate_source_handles(context.resources->model_manifest,context.resources->model_weights,context.resources->tensor_access);
        const auto& r=cached->ple_roles(context.execution_identity,context.step_id);
        result={tensor_pair(context,r.table),tensor_pair(context,r.key),tensor_pair(context,r.value),tensor_pair(context,r.norm_key),
            tensor_pair(context,r.norm_query),tensor_pair(context,r.norm_conv),tensor_pair(context,r.conv)};
    } else {
    const auto& table_access =
        resolve_dispatch_tensor_access(
            context, 0);
    if (table_access.tensor_indices.size() !=
        1U) {
        throw std::invalid_argument(
            "HIP PLE requires exactly one n-gram table");
    }
    result.table =
        tensor_pair(
            context,
            table_access.tensor_indices[0]);

    const auto& parameter_access =
        resolve_dispatch_tensor_access(
            context, 1);

    std::uint32_t key = 0;
    std::uint32_t value = 0;
    std::uint32_t norm_key = 0;
    std::uint32_t norm_query = 0;
    std::uint32_t norm_conv = 0;
    std::uint32_t conv = 0;

    for (const auto index :
         parameter_access.tensor_indices) {
        const auto pair =
            tensor_pair(
                context,
                index);
        const auto& tensor =
            *pair.manifest;
        if (tensor.role !=
            TensorRole::ple_parameter) {
            continue;
        }

        if (contains(
                tensor.name,
                "ple_norm_key")) {
            result.norm_key = pair;
            ++norm_key;
        } else if (contains(
                       tensor.name,
                       "ple_norm_query")) {
            result.norm_query = pair;
            ++norm_query;
        } else if (contains(
                       tensor.name,
                       "ple_norm_conv")) {
            result.norm_conv = pair;
            ++norm_conv;
        } else if (contains(
                       tensor.name,
                       "ple_conv1d")) {
            result.conv = pair;
            ++conv;
        } else if (contains(
                       tensor.name,
                       "ple_key")) {
            result.key = pair;
            ++key;
        } else if (contains(
                       tensor.name,
                       "ple_value")) {
            result.value = pair;
            ++value;
        }
    }

    if (key != 1 ||
        value != 1 ||
        norm_key != 1 ||
        norm_query != 1 ||
        norm_conv != 1 ||
        conv != 1) {
        throw std::invalid_argument(
            "PLE parameter tensor family is missing or ambiguous");
    }

    }

    if (result.table.manifest->role !=
            TensorRole::ngram_table ||
        result.table.manifest->
            dimensions.size() < 2 ||
        result.table.manifest->
            dimensions[0] !=
            config.model.ple_head_dim()) {
        throw std::invalid_argument(
            "PLE table geometry does not match contract");
    }
    if (result.key.manifest->
            dimensions.size() < 2 ||
        result.key.manifest->
            dimensions[0] !=
            config.model.ple_embed_dim ||
        result.key.manifest->
            dimensions[1] !=
            hc_hidden ||
        result.value.manifest->
            dimensions.size() < 2 ||
        result.value.manifest->
            dimensions[0] !=
            config.model.ple_embed_dim ||
        result.value.manifest->
            dimensions[1] !=
            config.model.hidden_size) {
        throw std::invalid_argument(
            "PLE projection geometry does not match contract");
    }

    return result;
}

void validate_projection(
    const TensorPair& pair,
    std::uint32_t input,
    std::uint64_t output,
    const char* name) {
    if (pair.runtime == nullptr ||
        !runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " is not GPU-visible");
    }
    const auto required =
        required_weight_bytes(
            weight_kind(
                pair.manifest->ggml_type),
            input,
            output);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " binding is undersized");
    }
}

void validate_small(
    const TensorPair& pair,
    std::uint64_t elements,
    const char* name) {
    if (pair.runtime == nullptr ||
        !runtime_tensor_gpu_visible(pair.runtime->storage) ||
        pair.runtime->address == 0) {
        throw std::invalid_argument(
            std::string(name) +
            " is not GPU-visible");
    }
    const auto required =
        small_weight_bytes(
            small_weight_kind(
                pair.manifest->ggml_type),
            elements);
    if (pair.runtime->bytes <
        required) {
        throw std::invalid_argument(
            std::string(name) +
            " binding is undersized");
    }
}

struct DevicePleHash {
    std::uint32_t ngram, heads_per_ngram, heads, width, eos;
    std::uint64_t multipliers[8], offsets[32], vocabularies[32];
};
__global__ void graph_ple_gather_kernel(DevicePleHash h, const std::uint16_t* table,
    const std::uint32_t* token, const std::uint64_t* position, const std::uint32_t* previous,
    std::uint16_t* gathered, std::uint32_t vocabulary, std::uint64_t context,
    std::uint32_t* status) {
    if (atomicAdd(status, 0U)) return;
    if (*token >= vocabulary || *position >= context) { atomicOr(status, 16U); return; }
    std::uint64_t values[8]; values[0] = *token;
    bool cut = false;
    for (std::uint32_t shift = 1; shift < h.ngram; ++shift) {
        const auto value = *position == 0 ? h.eos : previous[h.ngram - 1U - shift];
        if (value >= vocabulary) { atomicOr(status, 16U); return; }
        cut = cut || value == h.eos; values[shift] = cut ? h.eos : value;
    }
    const auto element = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element >= h.heads * h.width) return;
    const auto head = element / h.width;
    const auto ngram = head / h.heads_per_ngram + 2U;
    auto mixed = values[0] * h.multipliers[0];
    for (std::uint32_t shift = 1; shift < ngram; ++shift) mixed ^= values[shift] * h.multipliers[shift];
    const auto row = mixed % h.vocabularies[head] + h.offsets[head];
    gathered[element] = table[row * h.width + element % h.width];
}
__global__ void graph_ple_history_kernel(const std::uint32_t* token, const std::uint64_t* position,
    const std::uint32_t* previous, std::uint32_t* next, std::uint32_t count,
    std::uint32_t eos, std::uint32_t* status) {
    if (threadIdx.x || atomicAdd(status, 0U)) return;
    for (std::uint32_t i = 0; i + 1U < count; ++i) next[i] = *position == 0 ? eos : previous[i + 1U];
    next[count - 1U] = *token;
}

struct PreparedPleLaunch {
    const HipPleLauncherConfig* config;
    ExecutionBufferAddress input, output, temporary;
    PleWorkspaceLayout workspace_layout;
    std::uint64_t expanded;
    PleTensors tensors;
    SessionStateAddress ngram_state, conv_state;
    PleHashContract hash_contract;
};
PreparedPleLaunch prepare_ple_launch(const ExecutionDispatchContext& context, void* user_data) {
    if (context.op !=
        SerialOp::ple) {
        throw std::invalid_argument(
            "HIP PLE launcher attached to wrong operator");
    }
    if (context.reads.size() != 1 ||
        context.writes.size() != 1 ||
        context.temporaries.size() != 1) {
        throw std::invalid_argument(
            "HIP PLE expects one hyper read/write and one workspace");
    }
    if (context.logical_rows != 1 ||
        context.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP PLE v1 supports only C1 one-row execution");
    }
    if (user_data == nullptr ||
        context.resources == nullptr ||
        ((!current_hip_launch_context() || !current_hip_launch_context()->graph_capture) &&
        context.resources->ple_lookup == nullptr)) {
        throw std::invalid_argument(
            "HIP PLE requires config and lookup backend");
    }

    const auto& config =
        *static_cast<
            HipPleLauncherConfig*>(
                user_data);
    validate_config(config);

    const auto input =
        resolve_dispatch_read(
            context, 0);
    const auto output =
        resolve_dispatch_write(
            context, 0);
    const auto temporary =
        resolve_dispatch_temporary(
            context, 0);
    const auto workspace_layout =
        build_ple_workspace_layout(
            config.model);

    const auto expanded =
        config.model.hidden_size *
        config.model.hc_count;
    const auto expanded_bytes =
        expanded *
        config.model.bytes_bf16;

    if (input.bytes <
            expanded_bytes ||
        output.bytes <
            expanded_bytes ||
        temporary.bytes <
            workspace_layout.total_bytes) {
        throw std::invalid_argument(
            "HIP PLE buffers are undersized");
    }

    const auto tensors =
        identify_tensors(
            context,
            config);

    validate_projection(
        tensors.key,
        static_cast<std::uint32_t>(
            config.model.ple_embed_dim),
        expanded,
        "PLE key projection");
    validate_projection(
        tensors.value,
        static_cast<std::uint32_t>(
            config.model.ple_embed_dim),
        config.model.hidden_size,
        "PLE value projection");

    validate_small(
        tensors.norm_key,
        expanded,
        "PLE key norm");
    validate_small(
        tensors.norm_query,
        expanded,
        "PLE query norm");
    validate_small(
        tensors.norm_conv,
        expanded,
        "PLE conv norm");
    validate_small(
        tensors.conv,
        config.model.ple_conv_kernel_size *
            expanded,
        "PLE convolution");

    const auto ngram_state =
        resolve_dispatch_state(
            context,
            SessionStateComponent::ngram_history);
    const auto conv_state =
        resolve_dispatch_state(
            context,
            SessionStateComponent::ple_convolution);

    if (ngram_state.storage !=
            SessionStateStorage::host ||
        ngram_state.address == 0 ||
        ngram_state.bytes <
            (config.model.ngram_size - 1ULL) *
                sizeof(std::uint32_t) ||
        conv_state.storage !=
            SessionStateStorage::device ||
        conv_state.address == 0 ||
        conv_state.bytes <
            static_cast<std::uint64_t>(
                expanded) *
                (config.model.ple_conv_kernel_size - 1ULL) *
                config.model.ngram_size *
                config.model.bytes_bf16) {
        throw std::invalid_argument(
            "HIP PLE persistent state binding is invalid");
    }

    const auto hash_contract =
        build_ple_hash_contract(
            config.model);

    return {&config, input, output, temporary, workspace_layout, expanded, tensors,
        ngram_state, conv_state, hash_contract};
}
void preflight_ple(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_ple_launch(context, user_data);
    if (const auto* scope = current_hip_launch_context(); scope && scope->graph_capture)
        throw std::invalid_argument("known window PLE preflight is eager only");
    // P0 uses the existing synchronous lookup/upload boundary. Never start or
    // finish the single pending backend lookup during metadata admission.
    if (prepared.config->upload_pool || prepared.config->enable_owned_event_upload)
        throw std::invalid_argument("known window PLE requires synchronous upload without an event pool");
    if (!dynamic_cast<FileMappedPleLookupBackend*>(context.resources->ple_lookup))
        throw std::invalid_argument("known window PLE currently admits the synchronous file-mapped backend only");
    const auto& table = *prepared.tensors.table.manifest;
    const auto& binding = *prepared.tensors.table.runtime;
    if (table.dimensions.size() != 2 || !table.dimensions[0] ||
        table.dimensions[1] < prepared.hash_contract.minimum_table_rows ||
        (table.ggml_type != 30U && table.ggml_type != 20U) ||
        table.dimensions[0] > std::numeric_limits<std::uint64_t>::max() / 2U ||
        table.dimensions[0] * 2U != prepared.hash_contract.row_bytes ||
        binding.storage != RuntimeTensorStorage::file_mapped || !binding.address ||
        binding.bytes > std::numeric_limits<std::uintptr_t>::max() - binding.address)
        throw std::invalid_argument("known window PLE table metadata or mapping is incompatible");
    auto stored_row_bytes = prepared.hash_contract.row_bytes;
    if (table.ggml_type == 20U) {
        if (table.dimensions[0] % iq4_nl::block_elements != 0U)
            throw std::invalid_argument("known window PLE IQ4_NL rows require complete blocks");
        stored_row_bytes = table.dimensions[0] / iq4_nl::block_elements * iq4_nl::block_bytes;
    }
    if (!stored_row_bytes || table.dimensions[1] > std::numeric_limits<std::uint64_t>::max() / stored_row_bytes ||
        table.byte_size < table.dimensions[1] * stored_row_bytes || binding.bytes < table.byte_size)
        throw std::invalid_argument("known window PLE declared table rows exceed the retained mapping");
    if (context.resources->ple_hash_contract) {
        const auto& a = *context.resources->ple_hash_contract;
        const auto& b = prepared.hash_contract;
        if (a.ngram_size != b.ngram_size || a.heads_per_ngram != b.heads_per_ngram ||
            a.head_count != b.head_count || a.head_dim != b.head_dim || a.eos_token_id != b.eos_token_id ||
            a.row_bytes != b.row_bytes || a.multipliers != b.multipliers || a.head_offsets != b.head_offsets ||
            a.head_vocab_sizes != b.head_vocab_sizes || a.minimum_table_rows != b.minimum_table_rows)
            throw std::invalid_argument("known window PLE prefetch hash contract differs from the selected launcher");
    }
    const auto bytes = static_cast<std::uint64_t>(prepared.hash_contract.head_count) * prepared.hash_contract.row_bytes;
    if (bytes != prepared.workspace_layout.gathered_bytes || bytes > PleHashContract::maximum_heads * 512ULL)
        throw std::invalid_argument("PLE gathered upload does not match its exact workspace region");
}

// key/value: the row's key and value projections when a window computed them already for all its rows.
void enqueue_ple_device(const PreparedPleLaunch& prepared, std::uint16_t* gathered, float* key_rows = nullptr,
    float* value_rows = nullptr) {
    const auto& config = *prepared.config;
    const auto& input = prepared.input;
    const auto& output = prepared.output;
    const auto& temporary = prepared.temporary;
    const auto& workspace_layout = prepared.workspace_layout;
    const auto expanded = prepared.expanded;
    const auto& tensors = prepared.tensors;
    const auto& conv_state = prepared.conv_state;
    auto* workspace =
        reinterpret_cast<std::byte*>(
            temporary.address);
    auto* key =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.key_offset);
    auto* value =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.value_offset);
    auto* gated =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.gated_offset);
    auto* normalized =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.normalized_offset);
    auto* gates =
        reinterpret_cast<float*>(
            workspace +
            workspace_layout.gate_offset);

    if (key_rows) {
        key = key_rows; value = value_rows;
    } else {
    launch_projection_by_kind(
        weight_kind(
            tensors.key.manifest->
                ggml_type),
        gathered,
        *tensors.key.runtime,
        static_cast<std::uint32_t>(
            config.model.ple_embed_dim),
        static_cast<std::uint32_t>(
            expanded),
        key);

    launch_projection_by_kind(
        weight_kind(
            tensors.value.manifest->
                ggml_type),
        gathered,
        *tensors.value.runtime,
        static_cast<std::uint32_t>(
            config.model.ple_embed_dim),
        static_cast<std::uint32_t>(
            config.model.hidden_size),
        value);
    }

    hipLaunchKernelGGL(
        normalize_key_and_gate_kernel,
        dim3(
            static_cast<std::uint32_t>(
                config.model.hc_count)),
        dim3(kThreads),
        0, hip_execution_stream(),
        key,
        reinterpret_cast<
            const std::uint16_t*>(
                input.address),
        reinterpret_cast<
            const void*>(
                tensors.norm_key.
                    runtime->address),
        small_weight_kind(
            tensors.norm_key.
                manifest->ggml_type),
        reinterpret_cast<
            const void*>(
                tensors.norm_query.
                    runtime->address),
        small_weight_kind(
            tensors.norm_query.
                manifest->ggml_type),
        gates,
        static_cast<std::uint32_t>(
            config.model.hidden_size),
        static_cast<std::uint32_t>(
            config.model.hc_count),
        config.model.rms_norm_epsilon);
    check_hip(
        hipGetLastError(),
        "HIP PLE key/query gate launch");

    const auto value_blocks =
        (expanded +
         kThreads - 1ULL) /
        kThreads;
    hipLaunchKernelGGL(
        gated_value_kernel,
        dim3(
            static_cast<std::uint32_t>(
                value_blocks)),
        dim3(kThreads),
        0, hip_execution_stream(),
        value,
        gates,
        gated,
        static_cast<std::uint32_t>(
            config.model.hidden_size),
        static_cast<std::uint32_t>(
            config.model.hc_count));
    check_hip(
        hipGetLastError(),
        "HIP PLE gated value launch");

    hipLaunchKernelGGL(
        grouped_norm_kernel,
        dim3(
            static_cast<std::uint32_t>(
                config.model.hc_count)),
        dim3(kThreads),
        0, hip_execution_stream(),
        gated,
        reinterpret_cast<
            const void*>(
                tensors.norm_conv.
                    runtime->address),
        small_weight_kind(
            tensors.norm_conv.
                manifest->ggml_type),
        normalized,
        static_cast<std::uint32_t>(
            config.model.hidden_size),
        static_cast<std::uint32_t>(
            config.model.hc_count),
        config.model.rms_norm_epsilon);
    check_hip(
        hipGetLastError(),
        "HIP PLE conv norm launch");

    const auto history =
        static_cast<std::uint32_t>(
            (config.model.ple_conv_kernel_size -
             1ULL) *
            config.model.ngram_size);
    const auto conv_blocks =
        (expanded +
         kThreads - 1ULL) /
        kThreads;

    hipLaunchKernelGGL(
        conv_residual_kernel,
        dim3(
            static_cast<std::uint32_t>(
                conv_blocks)),
        dim3(kThreads),
        0, hip_execution_stream(),
        reinterpret_cast<
            const std::uint16_t*>(
                input.address),
        gated,
        normalized,
        reinterpret_cast<
            std::uint16_t*>(
                conv_state.address),
        reinterpret_cast<
            const void*>(
                tensors.conv.runtime->
                    address),
        small_weight_kind(
            tensors.conv.manifest->
                ggml_type),
        reinterpret_cast<
            std::uint16_t*>(
                output.address),
        static_cast<std::uint32_t>(
            expanded),
        history,
        static_cast<std::uint32_t>(
            config.model.ple_conv_kernel_size),
        static_cast<std::uint32_t>(
            config.model.ngram_size), hip_graph_status());
    check_hip(
        hipGetLastError(),
        "HIP PLE convolution launch");

}

void launch_ple(const ExecutionDispatchContext& context, void* user_data) {
    const auto prepared = prepare_ple_launch(context, user_data);
    const auto& config = *prepared.config;
    const auto& temporary = prepared.temporary;
    const auto& workspace_layout = prepared.workspace_layout;
    const auto& tensors = prepared.tensors;
    const auto& ngram_state = prepared.ngram_state;
    const auto& hash_contract = prepared.hash_contract;

    alignas(64)
    std::array<
        std::byte,
        PleHashContract::maximum_heads * 512>
        host_staging{};

    PleHashResult hash;
    const auto* graph = current_hip_launch_context();
    if (graph && graph->graph_capture) {
        if (!config.enable_resident_graph || config.upload_pool || config.enable_owned_event_upload ||
            tensors.table.manifest->ggml_type != 30)
            throw std::invalid_argument("captured PLE requires opt-in resident BF16 table without host upload");
        hash.row_count = hash_contract.head_count;
    } else {
    auto& backend =
        *context.resources->ple_lookup;

    if (context.resources->ple_prefetch != nullptr &&
        context.resources->ple_prefetch->active) {
        const auto prefetched_bytes =
            context.resources->ple_prefetch->
                lookup_bytes;
        if (prefetched_bytes >
            host_staging.size()) {
            throw std::overflow_error(
                "PLE prefetched lookup exceeds fixed host staging");
        }
        hash =
            finish_ple_lookup_prefetch(
                context.input_token_id,
                context.position,
                backend,
                *context.resources->ple_prefetch,
                std::span<std::byte>(
                    host_staging.data(),
                    static_cast<std::size_t>(
                        prefetched_bytes)));
    } else {
        std::array<
            std::uint32_t,
            PleHashContract::maximum_ngram - 1>
            previous{};
        const auto context_count =
            static_cast<std::size_t>(
                config.model.ngram_size - 1ULL);

        if (context.position == 0) {
            std::fill_n(
                previous.begin(),
                context_count,
                static_cast<std::uint32_t>(
                    config.model.ple_eos_token_id));
        } else {
            std::memcpy(
                previous.data(),
                reinterpret_cast<const void*>(
                    ngram_state.address),
                context_count *
                    sizeof(std::uint32_t));
        }

        hash =
            hash_ple_token(
                hash_contract,
                context.input_token_id,
                std::span<const std::uint32_t>(
                    previous.data(),
                    context_count));

        PleLookupRequest lookup;
        lookup.row_ids =
            std::span<const std::uint64_t>(
                hash.row_ids.data(),
                hash.row_count);
        lookup.row_bytes =
            hash_contract.row_bytes;
        lookup.table_manifest =
            tensors.table.manifest;
        lookup.table_runtime =
            tensors.table.runtime;

        const auto sync_lookup_bytes =
            static_cast<std::uint64_t>(
                hash.row_count) *
            hash_contract.row_bytes;
        if (sync_lookup_bytes >
            host_staging.size()) {
            throw std::overflow_error(
                "PLE lookup exceeds fixed host staging");
        }

        PleLookupHandle handle = 0;
        try {
            handle =
                backend.start(
                    lookup);
            backend.finish(
                handle,
                std::span<std::byte>(
                    host_staging.data(),
                    static_cast<std::size_t>(
                        sync_lookup_bytes)));
            handle = 0;
        } catch (...) {
            if (handle != 0) {
                backend.cancel(handle);
            }
            throw;
        }
    }

    }

    const auto lookup_bytes =
        static_cast<std::uint64_t>(
            hash.row_count) *
        hash_contract.row_bytes;

    auto* workspace =
        reinterpret_cast<std::byte*>(
            temporary.address);
    auto* gathered =
        reinterpret_cast<std::uint16_t*>(
            workspace +
            workspace_layout.gathered_offset);

    if (lookup_bytes != workspace_layout.gathered_bytes || lookup_bytes > host_staging.size()) {
        throw std::invalid_argument("PLE gathered upload does not match its exact workspace region");
    }
    if (graph && graph->graph_capture) {
        auto gathered_view = temporary;
        gathered_view.address = reinterpret_cast<std::uintptr_t>(gathered);
        gathered_view.bytes = workspace_layout.gathered_bytes;
        validate_graph_ple_bindings(hash_contract, graph->vocabulary_size, graph->context_tokens,
            *tensors.table.runtime, graph->arguments, graph->status, gathered_view,
            graph->ngram_input, graph->ngram_output);
        DevicePleHash h{};
        h.ngram = hash_contract.ngram_size; h.heads_per_ngram = hash_contract.heads_per_ngram;
        h.heads = hash_contract.head_count; h.width = hash_contract.head_dim; h.eos = hash_contract.eos_token_id;
        std::copy_n(hash_contract.multipliers.begin(), 8, h.multipliers);
        std::copy_n(hash_contract.head_offsets.begin(), 32, h.offsets);
        std::copy_n(hash_contract.head_vocab_sizes.begin(), 32, h.vocabularies);
        hipLaunchKernelGGL(graph_ple_gather_kernel, dim3((h.heads * h.width + 255U) / 256U), dim3(256),
            0, graph->stream, h, reinterpret_cast<const std::uint16_t*>(tensors.table.runtime->address),
            reinterpret_cast<const std::uint32_t*>(graph->arguments.memory.arena_base + graph->arguments.token_offset),
            hip_graph_position(), reinterpret_cast<const std::uint32_t*>(graph->ngram_input.arena_base),
            gathered, graph->vocabulary_size, graph->context_tokens, hip_graph_status());
        check_hip(hipGetLastError(), "resident PLE graph hash/gather");
    } else {
    // The synchronous host upload uses the backend/default copy stream. An
    // explicit nonblocking eager stream must first release reused arena views.
    // This also keeps stack staging lifetime independent of async copy errors.
    if (graph) check_hip(hipStreamSynchronize(graph->stream), "PLE eager upload arena fence");
    if (config.upload_pool != nullptr) {
        auto destination = temporary;
        destination.address = reinterpret_cast<std::uintptr_t>(gathered);
        destination.bytes = workspace_layout.gathered_bytes;
        const auto ticket = config.upload_pool->start(
            std::span<const std::byte>(host_staging.data(), static_cast<std::size_t>(lookup_bytes)),
            destination);
        // Preserve the eager synchronous boundary: dependent projections only
        // start after the copy's event proves the retained host input released.
        config.upload_pool->wait(ticket, destination);
    } else {
        check_hip(hipMemcpy(gathered, host_staging.data(), static_cast<std::size_t>(lookup_bytes),
                            hipMemcpyHostToDevice), "copy PLE gathered embedding");
    }
    }

    enqueue_ple_device(prepared, gathered);

    if (graph && graph->graph_capture) {
        hipLaunchKernelGGL(graph_ple_history_kernel, dim3(1), dim3(1), 0, graph->stream,
            reinterpret_cast<const std::uint32_t*>(graph->arguments.memory.arena_base + graph->arguments.token_offset),
            hip_graph_position(), reinterpret_cast<const std::uint32_t*>(graph->ngram_input.arena_base),
            reinterpret_cast<std::uint32_t*>(graph->ngram_output.arena_base), hash_contract.ngram_size - 1U,
            hash_contract.eos_token_id, hip_graph_status());
        check_hip(hipGetLastError(), "resident PLE graph next history");
        return; // Host history publishes only after whole-graph completion/status.
    }

    // PLE v1 deliberately synchronizes here: host n-gram history must not
    // advance until the device convolution/state update has succeeded.
    check_hip(
        hipStreamSynchronize(hip_execution_stream()),
        "HIP PLE synchronize");

    std::memcpy(
        reinterpret_cast<void*>(
            ngram_state.address),
        hash.next_context.data(),
        static_cast<std::size_t>(
            hash.context_count) *
            sizeof(std::uint32_t));
    ple_window_warm::register_history(hash);
}

std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipPleLauncherConfig& config) {
    return
        "hip.ple." +
        std::string(
            config.activation ==
                    HipActivationStorage::bf16
                ? "bf16"
                : "fp32") +
        ".h" +
        std::to_string(
            config.model.hidden_size) +
        ".hc" +
        std::to_string(
            config.model.hc_count) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (config.enable_resident_graph ? ".resident-graph" : "");
}

std::uint64_t implementation_identity(
    const HipPleLauncherConfig& config) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(hash, kLauncherTag);
    hash_u64(
        hash,
        config.model.hidden_size);
    hash_u64(
        hash,
        config.model.hc_count);
    hash_u64(
        hash,
        config.model.ple_embed_dim);
    hash_u64(
        hash,
        config.model.ple_conv_kernel_size);
    hash_u64(
        hash,
        config.model.ngram_size);
    hash_u64(
        hash,
        static_cast<std::uint64_t>(
            config.activation));
    hash_u64(hash, config.enable_resident_graph ? 1 : 0);
    if (config.upload_pool != nullptr || config.enable_owned_event_upload) {
        hash_u64(hash, 0x484f535455504c31ULL);
    }
    return hash;
}

}  // namespace
void testing::enqueue_ple_q8_projection(bool wave8,const std::uint16_t* input,const void* q8,std::uint32_t input_width,
    std::uint32_t output_width,float* output,void* stream) {
    const auto s=static_cast<hipStream_t>(stream);
    if(wave8) hipLaunchKernelGGL(project_wave8_q8_bf16_to_fp32_kernel,dim3((output_width+7U)/8U),dim3(256),0,s,input,q8,input_width,output_width,output);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_bf16_to_fp32_kernel<WeightKind::q8_0>),dim3(output_width),dim3(kThreads),0,s,
        input,q8,input_width,output_width,output);
    check_hip(hipGetLastError(),"PLE Q8 projection test launch");
}
void testing::enqueue_ple_tail(bool rows_form,float* keys,const float* values,const std::uint16_t* input,std::uint32_t input_stride,
    std::uint16_t* output,std::uint32_t output_stride,std::uint16_t* conv_state,const void* norm_key,const void* norm_query,
    const void* norm_conv,const void* conv,bool bf16_weights,std::uint32_t rows,std::uint32_t hidden_size,std::uint32_t hc_count,
    std::uint32_t kernel_size,std::uint32_t ngram_size,float epsilon,float* scratch,void* stream) {
    const auto s=static_cast<hipStream_t>(stream);
    const auto kind=bf16_weights?SmallWeightKind::bf16:SmallWeightKind::f32;
    const auto expanded=hidden_size*hc_count,history=(kernel_size-1U)*ngram_size,blocks=(expanded+kThreads-1U)/kThreads;
    if(rows_form){
        const PleTailRows t{keys,expanded,values,hidden_size,input,input_stride,output,output_stride,conv_state,norm_key,norm_query,
            norm_conv,conv,kind,kind,kind,kind,scratch,rows,hidden_size,hc_count,kernel_size,ngram_size,epsilon};
        enqueue_ple_tail_rows(t,nullptr,s);
        return;
    }
    // The launch sequence of enqueue_ple_device after its projections, one row at a time.
    float* gates=scratch;float* gated=scratch+hc_count;float* normalized=gated+expanded;
    for(std::uint32_t r=0;r<rows;++r){
        float* key=keys+static_cast<std::uint64_t>(r)*expanded;const float* value=values+static_cast<std::uint64_t>(r)*hidden_size;
        const auto* in=input+static_cast<std::uint64_t>(r)*input_stride;auto* out=output+static_cast<std::uint64_t>(r)*output_stride;
        hipLaunchKernelGGL(normalize_key_and_gate_kernel,dim3(hc_count),dim3(kThreads),0,s,key,in,norm_key,kind,norm_query,kind,gates,
            hidden_size,hc_count,epsilon);
        hipLaunchKernelGGL(gated_value_kernel,dim3(blocks),dim3(kThreads),0,s,value,gates,gated,hidden_size,hc_count);
        hipLaunchKernelGGL(grouped_norm_kernel,dim3(hc_count),dim3(kThreads),0,s,gated,norm_conv,kind,normalized,hidden_size,hc_count,epsilon);
        hipLaunchKernelGGL(conv_residual_kernel,dim3(blocks),dim3(kThreads),0,s,in,gated,normalized,conv_state,conv,kind,out,expanded,
            history,kernel_size,ngram_size,nullptr);
    }
    check_hip(hipGetLastError(),"PLE tail test launch");
}

struct HipPleLauncherBundle::OwnedUploads {
    AllocationGenerationSource generations;
    HipHostUploadBackend backend{1};
    HostUploadPool pool{backend, generations,
        HostUploadPolicy{1, HipPleLauncherConfig::owned_event_upload_maximum_bytes,
                        HipPleLauncherConfig::owned_event_upload_maximum_bytes, 4096}};
};

HipPleLauncherBundle::HipPleLauncherBundle(HipPleLauncherConfig config)
    : config_(std::move(config)) {
    if (config_.enable_owned_event_upload) {
        if (config_.upload_pool != nullptr) {
            throw std::invalid_argument("PLE must select either an owned or injected upload pool");
        }
        uploads_ = std::make_unique<OwnedUploads>();
        config_.upload_pool = &uploads_->pool;
    }
}

HipPleLauncherBundle::~HipPleLauncherBundle() = default;

void register_hip_ple_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipPleLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification) {
    if (const auto error =
            model.validate();
        !error.empty()) {
        throw std::invalid_argument(
            error);
    }
    validate_config(config);
    if (config.model.hidden_size !=
            model.hidden_size ||
        config.model.ple_embed_dim !=
            model.ple_embed_dim ||
        config.model.ngram_size !=
            model.ngram_size) {
        throw std::invalid_argument(
            "HIP PLE config does not match model contract");
    }
    if (request.logical_rows != 1 ||
        request.request_concurrency != 1) {
        throw std::invalid_argument(
            "HIP PLE v1 registers only C1 one-row variants");
    }
    if (qualification ==
            kernels::QualificationStatus::target_qualified ||
        qualification ==
            kernels::QualificationStatus::production ||
        qualification ==
            kernels::QualificationStatus::rejected) {
        throw std::invalid_argument(
            "unmeasured HIP PLE may only be experimental or oracle-matched");
    }

    kernels::KernelVariant variant;
    variant.id =
        kernel_id(
            request,
            config);
    variant.family =
        kernels::OperatorFamily::
            sparse_lookup;
    variant.phase =
        kernels::ExecutionPhase::decode;
    variant.shape =
        execution_shape_for(
            model,
            SerialOp::ple,
            request);
    variant.wave =
        kernels::WaveSize::wave32;
    variant.resources.workgroup_threads =
        kThreads;
    variant.resources.lds_bytes =
        2U *
        kThreads *
        sizeof(float);
    variant.resources.scratch_bytes =
        build_ple_workspace_layout(
            model).total_bytes;
    variant.numerics.input_precision =
        "bf16";
    variant.numerics.accumulator_precision =
        "fp32";
    variant.numerics.output_precision =
        "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        qualification;
    // Host hash/lookup and explicit state-commit synchronization make v1
    // unsuitable for graph capture.
    variant.graph_safe = config.enable_resident_graph;
    variant.input_layout =
        "hc-stream-major+file-backed-ngram";
    variant.output_layout =
        "hc-stream-major";
    variant.compiler_identity =
        "hip-gfx1151-ple-v2";
    variant.measured_seconds = 0.0;

    registry.add(
        std::move(variant));
}

ExecutionLauncher
make_hip_ple_launcher(
    std::string kernel_id_value,
    HipPleLauncherConfig* config) {
    if (kernel_id_value.empty() ||
        config == nullptr) {
        throw std::invalid_argument(
            "HIP PLE launcher requires id and config");
    }
    validate_config(*config);
    if (config->enable_owned_event_upload && config->upload_pool == nullptr) {
        throw std::invalid_argument("PLE owned event upload requires a launcher bundle");
    }
    return ExecutionLauncher{
        std::move(kernel_id_value),
        implementation_identity(
            *config),
        &launch_ple,
        config,
        &preflight_ple,
    };
}

void HipPleLauncherBundle::
register_for_compiled_execution(
    const CompiledSerialExecution& compiled,
    ExecutionLauncherRegistry& registry) {
    if (compiled.execution_identity == 0 ||
        compiled.execution.steps.size() !=
            compiled.bindings.bindings.size()) {
        throw std::invalid_argument(
            "HIP PLE bundle requires valid compiled execution");
    }

    for (std::size_t index = 0;
         index < compiled.execution.steps.size();
         ++index) {
        if (compiled.execution.steps[index].op !=
            SerialOp::ple) {
            continue;
        }
        const auto& binding =
            compiled.bindings.bindings[index];
        if (binding.host_or_runtime_step ||
            !binding.kernel_id.has_value() ||
            binding.kernel_id->empty()) {
            throw std::invalid_argument(
                "HIP PLE step has no kernel binding");
        }
        registry.add_kernel_launcher(
            make_hip_ple_launcher(
                *binding.kernel_id,
                &config_));
        return;
    }

    throw std::invalid_argument(
        "compiled execution has no PLE step");
}

#include "hip_ple_rows_site.inc"

// Gathers rows (heads row ids each) from the file-mapped table into staging + r * stride on up to 8 threads, each with
// its own stateless FileMappedPleLookupBackend: the same calls as the launch path's deferred gather.
static void gather_staged_rows(std::byte* staging, std::uint64_t stride, std::span<const std::uint32_t> row_ids, std::uint32_t heads,
    std::uint64_t row_bytes, TensorPair table, std::uint64_t* majflt,
    const ple_reclaim::Prefetcher::PromptSource* source = nullptr,
    const std::shared_ptr<ple_reclaim::ResidencyManager::Turn>& turn = {},
    std::size_t worker_limit = 0) {
    if (!heads || heads > PleHashContract::maximum_heads || row_ids.size() % heads || heads * row_bytes > stride)
        throw std::invalid_argument("PLE staged gather geometry is invalid");
    const auto rows = row_ids.size() / heads;
    const auto workers = worker_limit ? worker_limit
        : static_cast<std::size_t>(std::min(8U, std::max(1U, std::thread::hardware_concurrency())));
    std::vector<std::exception_ptr> errors(workers);
    std::vector<std::uint64_t> faults(workers, 0);
    const auto gather = [&](std::size_t worker) {
        try {
            FileMappedPleLookupBackend local;
            local.bind_residency(turn);
            if (source) local.bind_prompt_source(source->view, source->table, source->offset);
            std::array<std::uint64_t, PleHashContract::maximum_heads> ids{};
            for (std::size_t r = worker; r < rows; r += workers) {
                for (std::uint32_t h = 0; h < heads; ++h) ids[h] = row_ids[r * heads + h];
                PleLookupRequest lookup;
                lookup.row_ids = std::span<const std::uint64_t>(ids.data(), heads);
                lookup.row_bytes = row_bytes;
                lookup.table_manifest = table.manifest;
                lookup.table_runtime = table.runtime;
                const auto handle = local.start(lookup);
                try {
                    local.finish(handle, std::span<std::byte>(staging + r * stride, static_cast<std::size_t>(heads * row_bytes)));
                } catch (...) { local.cancel(handle); throw; }
            }
#if defined(__linux__)
            rusage usage{};
            if (getrusage(RUSAGE_THREAD, &usage) == 0) faults[worker] = static_cast<std::uint64_t>(usage.ru_majflt);
#endif
        } catch (...) { errors[worker] = std::current_exception(); }
    };
    std::vector<std::thread> threads;
    // Joins partial construction if thread creation throws (no std::jthread: node libstdc++ lacks its stop-token waits).
    struct JoinAll { std::vector<std::thread>& t; ~JoinAll() { for (auto& x : t) if (x.joinable()) x.join(); } } join_all{threads};
    threads.reserve(workers - 1U);
    for (std::size_t t = 1; t < workers; ++t) threads.emplace_back(gather, t);
    gather(0);
    for (auto& thread : threads) thread.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    if (majflt) for (const auto f : faults) *majflt += f;   // worker threads are fresh: their totals are this gather's
}

struct HipPleKnownWindow::Impl {
    HipPleLauncherConfig config;
    std::uint32_t max_rows = 0, rows = 0, slot = 0;
    bool prefill = false;
    std::uint64_t stride = 0, conv_bytes = 0;
    std::array<std::byte*, 2> staging{};
    std::array<hipEvent_t, 2> released{};
    std::array<bool, 2> recorded{};
    std::byte* snapshots = nullptr;
    float* projections = nullptr;   // every row's key, then every row's value
    q8_rows::LaneBuffer batch_rows; // barrier-ordered scratch owned by this window
    float* gates = nullptr;         // prefill: every row's stream gates (rows-form tail)
    std::uintptr_t conv_address = 0, ngram_address = 0;
    std::vector<PleHashResult> hashes;
    // STRIX_PLE_RECLAIMABLE c1 (src/ple_reclaim.cpp): a worker gathers the next prompt block's rows into the idle
    // staging slot while the GPU runs the current block; the next launch joins it and uses the slot only if every
    // row id it hashes equals the worker's (same rows, same gather, same staging bytes).
    struct StagePrefetch {
        std::thread thread;
        std::uint64_t first = 0, rows = 0, ns = 0, majflt = 0;
        std::uint32_t slot = 0, heads = 0;
        std::uint64_t row_bytes = 0, copy_ns = 0, turn_id = 0, epoch = 0;
        ple_reclaim::DirectActivity activity;
        const RuntimeTensorBinding* table_runtime = nullptr;   // gather source, compared at the hit
        std::vector<std::uint32_t> row_ids;
        std::exception_ptr error;
        bool active = false;
        bool serve = false;
        unsigned workers = 0;
        ple_detail::StageSource source;
    } stage;
    void join_stage() noexcept { if (stage.thread.joinable()) stage.thread.join(); }
    void start_rolling_stage(ple_reclaim::Prefetcher* reclaim, std::uint64_t next,
        std::uint64_t row_bytes, TensorPair table,
        std::shared_ptr<ple_reclaim::ResidencyManager::Turn> turn = {}) {
        if (turn && !turn->prompt()) return;
        std::vector<std::uint32_t> ids;
        // The next block's size is the plan's (T, or a STRIX_QPF2K_SCHED merged/balanced partial block): stage the
        // window's capacity (clipped to the prompt); the next launch uses its first rows.size() rows.
        if (const auto staged = reclaim->stage_rows(next, max_rows, ids)) {
            auto& st = stage;
            st.serve = false;
            st.first = next; st.rows = staged; st.slot = slot ^ 1U; st.heads = reclaim->heads();
            st.row_ids = std::move(ids); st.error = nullptr; st.ns = 0; st.majflt = 0; st.copy_ns = 0; st.activity = {}; st.active = true;
            st.row_bytes = row_bytes; st.table_runtime = table.runtime;
            st.turn_id = turn ? turn->id() : 0U; st.epoch = turn ? turn->epoch() : 0U;
            const bool tracked_stage = reclaim->rolling();
            if (turn) turn->stage_begin();
            try { if (tracked_stage) reclaim->stage_begin(); }
            catch (...) { if (turn) turn->stage_end(std::current_exception()); throw; }
            try { st.thread = std::thread([&st, destination = staging[st.slot], released = released[st.slot],
                                     wait = recorded[st.slot], stride = this->stride, row_bytes,
                                     table, reclaim, tracked_stage, turn] {
                const bool stats = reclaim->block_stats();
                const ple_reclaim::DirectReadScope read_scope(tracked_stage, stats ? &st.activity : nullptr);
                try {
                    const auto t0 = std::chrono::steady_clock::now();
                    // The slot was last read by the previous block's PLE launches.
                    if (wait) check_hip(hipEventSynchronize(released), "known-window PLE staged slot reuse");
                    {
                        const auto copy_range = [&](auto first, auto count) {
                            const auto offset = first - st.first;
                            const auto copy0 = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                            auto source = turn ? ple_reclaim::Prefetcher::PromptSource{
                                turn->prompt_view(first, count), table.runtime->address, table.manifest->absolute_offset}
                                : reclaim->prompt_source(first, count);
                            gather_staged_rows(destination + offset * stride, stride,
                                std::span<const std::uint32_t>(st.row_ids).subspan(offset * st.heads, count * st.heads),
                                st.heads, row_bytes, table, &st.majflt, &source, turn);
                            if (stats) st.copy_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - copy0).count());
                        };
                        if (turn) turn->gather_prompt(st.first, st.rows, copy_range, st.epoch);
                        else reclaim->gather_range(st.first, st.rows, copy_range);
                    }
                    st.ns = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
                } catch (...) { st.error = std::current_exception(); }
                if (tracked_stage) reclaim->stage_end(st.error);
                if (turn) turn->stage_end(st.error);
            }); } catch (...) {
                if (tracked_stage) reclaim->stage_end(std::current_exception());
                if (turn) turn->stage_end(std::current_exception());
                throw;
            }
        }
    }
    ~Impl() {
        join_stage();
        for (auto* p : staging) if (p) (void)hipHostFree(p);
        for (auto e : released) if (e) (void)hipEventDestroy(e);
        if (snapshots) (void)hipFree(snapshots);
        if (projections) (void)hipFree(projections);
        if (gates) (void)hipFree(gates);
        if (batch_rows.base) (void)hipFree(batch_rows.base);
    }
    // Key and value projections of all rows, each Q8 weight block read once (the per-row
    // wave8 arithmetic). False when a projection is not wave8 Q8: rows project one by one.
    // project_rows for kPleWideRows rows at once (prefill); false when the shapes or alignment do not fit.
    bool project_rows_wide(const PreparedPleLaunch& p, const std::uint16_t* gathered, std::uint32_t first) const {
        const auto width = static_cast<std::uint32_t>(config.model.ple_embed_dim);
        if (weight_kind(p.tensors.key.manifest->ggml_type) != WeightKind::q8_0 ||
            weight_kind(p.tensors.value.manifest->ggml_type) != WeightKind::q8_0 || width % 32U ||
            (reinterpret_cast<std::uintptr_t>(gathered) | stride) % 16U) return false;
        const auto expanded = static_cast<std::uint32_t>(p.expanded), hidden = static_cast<std::uint32_t>(config.model.hidden_size);
        const auto project = [&](const void* weights, std::uint32_t outputs, float* out) {
            PleRowsSite<kPleWideRows> site{};
            site.segments = q8_rows::segments({{weights, outputs}});
            site.width = width; site.outputs = outputs;
            site.in_stride = static_cast<std::uint32_t>(stride / sizeof(std::uint16_t)); site.out_stride = outputs;
            site.base = gathered; site.out = out;
            q8_rows::launch(site, false, hip_execution_stream());
        };
        project(reinterpret_cast<const void*>(p.tensors.key.runtime->address), expanded, projections + static_cast<std::uint64_t>(first) * expanded);
        project(reinterpret_cast<const void*>(p.tensors.value.runtime->address), hidden,
            projections + static_cast<std::uint64_t>(max_rows) * expanded + static_cast<std::uint64_t>(first) * hidden);
        check_hip(hipGetLastError(), "prefill PLE wide rows projection launch");
        return true;
    }
    bool project_rows_batch(const PreparedPleLaunch& p, const std::uint16_t* gathered, std::uint32_t groups) {
        const auto width = static_cast<std::uint32_t>(config.model.ple_embed_dim);
        if (!groups || weight_kind(p.tensors.key.manifest->ggml_type) != WeightKind::q8_0 ||
            weight_kind(p.tensors.value.manifest->ggml_type) != WeightKind::q8_0 ||
            (reinterpret_cast<std::uintptr_t>(gathered) | stride) % 16U || stride / 2U > UINT32_MAX) return false;
        const auto expanded = static_cast<std::uint32_t>(p.expanded), hidden = static_cast<std::uint32_t>(config.model.hidden_size);
        const auto site = [&](const void* weights, std::uint32_t outputs, float* out) {
            PleRowsSite<kPleWideRows> result{};
            result.segments = q8_rows::segments({{weights, outputs}});
            result.width = width; result.outputs = outputs;
            result.in_stride = static_cast<std::uint32_t>(stride / 2U); result.out_stride = outputs;
            result.base = gathered; result.out = out;
            return result;
        };
        const auto key = site(reinterpret_cast<const void*>(p.tensors.key.runtime->address), expanded, projections);
        const auto value = site(reinterpret_cast<const void*>(p.tensors.value.runtime->address), hidden,
            projections + static_cast<std::uint64_t>(max_rows) * expanded);
        if (!ple_batch_admits(key, groups) || !ple_batch_admits(value, groups)) return false;
        if (!ple_batch_reserve(batch_rows, static_cast<std::uint64_t>(groups) * kPleWideRows * width * sizeof(float))) return false;
        // Resolve both caches before submitting consumers.
        const auto kw = q8_rows::lane_weights(key, hip_execution_stream());
        const auto vw = q8_rows::lane_weights(value, hip_execution_stream());
        auto* packed = static_cast<float*>(batch_rows.base);
        ple_batch_launch(key, kw, groups, packed, hip_execution_stream());
        ple_batch_launch(value, vw, groups, packed, hip_execution_stream());
        check_hip(hipGetLastError(), "prefill PLE batched projection launch");
        return true;
    }
    bool project_rows(const PreparedPleLaunch& p, const std::uint16_t* gathered, std::uint32_t count, std::uint32_t first = 0U) const {
        const auto width = static_cast<std::uint32_t>(config.model.ple_embed_dim);
        if (weight_kind(p.tensors.key.manifest->ggml_type) != WeightKind::q8_0 ||
            weight_kind(p.tensors.value.manifest->ggml_type) != WeightKind::q8_0 || width % 32U) return false;
        const auto expanded = static_cast<std::uint32_t>(p.expanded), hidden = static_cast<std::uint32_t>(config.model.hidden_size);
        const auto in_stride = static_cast<std::uint32_t>(stride / sizeof(std::uint16_t));
        float* keys = projections + static_cast<std::uint64_t>(first) * expanded;
        float* values = projections + static_cast<std::uint64_t>(max_rows) * expanded + static_cast<std::uint64_t>(first) * hidden;
        const auto launch = [&](auto rows) {
            constexpr unsigned R = decltype(rows)::value;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_wave8_q8_bf16_to_fp32_rows_kernel<R>), dim3((expanded + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), gathered, in_stride, reinterpret_cast<const void*>(p.tensors.key.runtime->address), width,
                expanded, keys, expanded);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_wave8_q8_bf16_to_fp32_rows_kernel<R>), dim3((hidden + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), gathered, in_stride, reinterpret_cast<const void*>(p.tensors.value.runtime->address), width,
                hidden, values, hidden);
        };
        switch (count) {
        case 1: launch(std::integral_constant<unsigned, 1>{}); break;
        case 2: launch(std::integral_constant<unsigned, 2>{}); break;
        case 3: launch(std::integral_constant<unsigned, 3>{}); break;
        case 4: launch(std::integral_constant<unsigned, 4>{}); break;
        case 5: launch(std::integral_constant<unsigned, 5>{}); break;
        case 6: launch(std::integral_constant<unsigned, 6>{}); break;
        case 7: launch(std::integral_constant<unsigned, 7>{}); break;
        case 8: {
            // Eight register-resident rows run 3x slower than two four-row passes (same rows' arithmetic).
            launch(std::integral_constant<unsigned, 4>{});
            const auto* upper = gathered + 4U * in_stride;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_wave8_q8_bf16_to_fp32_rows_kernel<4>), dim3((expanded + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), upper, in_stride, reinterpret_cast<const void*>(p.tensors.key.runtime->address), width,
                expanded, keys + 4ULL * expanded, expanded);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_wave8_q8_bf16_to_fp32_rows_kernel<4>), dim3((hidden + 7U) / 8U), dim3(256), 0,
                hip_execution_stream(), upper, in_stride, reinterpret_cast<const void*>(p.tensors.value.runtime->address), width,
                hidden, values + 4ULL * hidden, hidden);
            break;
        }
        default: return false;
        }
        check_hip(hipGetLastError(), "known-window PLE rows projection launch");
        return true;
    }
    // The rows-form tail of all prefill rows after project_rows: false (rows run one by one) unless
    // every row's input and output sit at one uniform stride, rows do not overlap and the outputs
    // are either the inputs themselves or disjoint from all of them.
    bool tail_rows(const std::vector<PreparedPleLaunch>& p, PleTailRows& t) const {
        const auto count = static_cast<std::uint32_t>(p.size());
        const auto expanded = p[0].expanded, bytes = expanded * config.model.bytes_bf16;
        const auto in0 = p[0].input.address, out0 = p[0].output.address;
        const auto in_step = count > 1U && p[1].input.address > in0 ? p[1].input.address - in0 : bytes;
        const auto out_step = count > 1U && p[1].output.address > out0 ? p[1].output.address - out0 : bytes;
        if (!gates || (in0 | out0 | in_step | out_step) % 2U || in_step < bytes || out_step < bytes ||
            in_step / 2U > std::numeric_limits<std::uint32_t>::max() || out_step / 2U > std::numeric_limits<std::uint32_t>::max())
            return false;
        for (std::uint32_t r = 0; r < count; ++r)
            if (p[r].input.address != in0 + r * in_step || p[r].output.address != out0 + r * out_step) return false;
        const auto in_end = in0 + (count - 1U) * in_step + bytes, out_end = out0 + (count - 1U) * out_step + bytes;
        if (!(in0 == out0 && in_step == out_step) && out0 < in_end && in0 < out_end) return false;
        const auto& s = p[0].tensors;
        t.keys = projections; t.key_stride = static_cast<std::uint32_t>(expanded);
        t.values = projections + static_cast<std::uint64_t>(max_rows) * expanded;
        t.value_stride = static_cast<std::uint32_t>(config.model.hidden_size);
        t.input = reinterpret_cast<const std::uint16_t*>(in0); t.input_stride = static_cast<std::uint32_t>(in_step / 2U);
        t.output = reinterpret_cast<std::uint16_t*>(out0); t.output_stride = static_cast<std::uint32_t>(out_step / 2U);
        t.conv_state = reinterpret_cast<std::uint16_t*>(p[0].conv_state.address);
        t.norm_key = reinterpret_cast<const void*>(s.norm_key.runtime->address);
        t.norm_query = reinterpret_cast<const void*>(s.norm_query.runtime->address);
        t.norm_conv = reinterpret_cast<const void*>(s.norm_conv.runtime->address);
        t.conv = reinterpret_cast<const void*>(s.conv.runtime->address);
        t.norm_key_kind = small_weight_kind(s.norm_key.manifest->ggml_type);
        t.norm_query_kind = small_weight_kind(s.norm_query.manifest->ggml_type);
        t.norm_conv_kind = small_weight_kind(s.norm_conv.manifest->ggml_type);
        t.conv_kind = small_weight_kind(s.conv.manifest->ggml_type);
        t.gates = gates; t.rows = count;
        t.hidden_size = static_cast<std::uint32_t>(config.model.hidden_size);
        t.hc_count = static_cast<std::uint32_t>(config.model.hc_count);
        t.kernel_size = static_cast<std::uint32_t>(config.model.ple_conv_kernel_size);
        t.ngram_size = static_cast<std::uint32_t>(config.model.ngram_size);
        t.epsilon = config.model.rms_norm_epsilon;
        return true;
    }
    std::vector<PreparedPleLaunch> prepare(std::span<const ExecutionDispatchContext> spans) {
        if (spans.empty() || spans.size() > max_rows)
            throw std::invalid_argument("known-window PLE row count exceeds its staging");
        if (!spans[0].resources || !spans[0].resources->ple_lookup)
            throw std::invalid_argument("known-window PLE requires a host lookup backend");
        std::vector<PreparedPleLaunch> out;
        out.reserve(spans.size());
        // Window rows hold per-row copies of one resource set (only the scratch binding differs, and
        // each row resolves its own arena views below): the same sources resolve the same tensors.
        const auto same_sources = [](const ExecutionRuntimeResources* a, const ExecutionRuntimeResources* b) {
            return a == b || (a && b && a->model_manifest == b->model_manifest && a->model_weights == b->model_weights &&
                a->tensor_access == b->tensor_access && a->session_state_layout == b->session_state_layout &&
                a->session_state_memory == b->session_state_memory && a->ple_lookup == b->ple_lookup &&
                a->ple_hash_contract == b->ple_hash_contract && a->distributed_stage == b->distributed_stage);
        };
        for (std::size_t r = 0; r < spans.size(); ++r) {
            const bool shared = r > 0U && (prefill || (same_sources(spans[r].resources, spans[0].resources) &&
                spans[r].prepared_tensor_views == spans[0].prepared_tensor_views));
            if (shared) {
                // Rows of one step share config, tensors, layout, hash contract and session state with
                // row 0 (checked below); only the row's own arena views differ, resolved and size-checked here.
                const auto& row = spans[r];
                const auto& head = spans[0];
                if (row.op != head.op || row.step_id != head.step_id || row.layer != head.layer ||
                    row.kernel_id != head.kernel_id || row.execution_identity != head.execution_identity ||
                    !same_sources(row.resources, head.resources) || row.logical_rows != 1U || row.request_concurrency != 1U ||
                    row.reads.size() != 1U || row.writes.size() != 1U || row.temporaries.size() != 1U)
                    throw std::invalid_argument("prefill PLE rows differ from their first row");
                auto copy = out.front();
                copy.input = resolve_dispatch_read(row, 0);
                copy.output = resolve_dispatch_write(row, 0);
                copy.temporary = resolve_dispatch_temporary(row, 0);
                copy.ngram_state = resolve_dispatch_state(row, SessionStateComponent::ngram_history);
                copy.conv_state = resolve_dispatch_state(row, SessionStateComponent::ple_convolution);
                if (copy.input.bytes < out.front().input.bytes || copy.output.bytes < out.front().output.bytes ||
                    copy.temporary.bytes < out.front().temporary.bytes)
                    throw std::invalid_argument("prefill PLE row buffers are undersized");
                out.push_back(copy);
            } else out.push_back(prepare_ple_launch(spans[r], &config));
            const auto& a = out.front(); const auto& b = out.back();
            if (spans[r].position != spans[0].position + r || b.conv_state.address != a.conv_state.address ||
                b.conv_state.bytes != a.conv_state.bytes || b.ngram_state.address != a.ngram_state.address ||
                b.tensors.table.runtime != a.tensors.table.runtime ||
                spans[r].resources->ple_lookup != spans[0].resources->ple_lookup)
                throw std::invalid_argument("known-window PLE rows do not share one ordered session");
        }
        if (out[0].conv_state.bytes < conv_bytes)
            throw std::invalid_argument("known-window PLE convolution state is undersized");
        return out;
    }
};

HipPleKnownWindow::HipPleKnownWindow(const ModelContract& model, HipActivationStorage activation, std::uint32_t max_rows,
    bool prefill, const ExecutionRuntimeResources* prime_resources)
    : impl_(std::make_unique<Impl>()) {
    auto& w = *impl_;
    if (!max_rows || max_rows > (prefill ? known_prefill_max_rows : 8U))
        throw std::invalid_argument("known-window PLE supports one to eight rows (prefill: up to " +
            std::to_string(known_prefill_max_rows) + ")");
    w.prefill = prefill;
    w.config.model = model; w.config.activation = activation; validate_config(w.config);
    w.max_rows = max_rows;
    w.stride = PleHashContract::maximum_heads * 512ULL;
    w.conv_bytes = static_cast<std::uint64_t>(model.hidden_size * model.hc_count) *
        (model.ple_conv_kernel_size - 1ULL) * model.ngram_size * model.bytes_bf16;
    for (std::size_t i = 0; i < 2; ++i) {
        void* p = nullptr;
        check_hip(hipHostMalloc(&p, w.stride * max_rows, hipHostMallocMapped | hipHostMallocCoherent), "known-window PLE staging");
        w.staging[i] = static_cast<std::byte*>(p);
        check_hip(hipEventCreateWithFlags(&w.released[i], hipEventDisableTiming), "known-window PLE staging event");
    }
    void* snap = nullptr;
    check_hip(hipMalloc(&snap, w.conv_bytes * (prefill ? 1U : max_rows)), "known-window PLE convolution snapshots");
    w.snapshots = static_cast<std::byte*>(snap);
    void* projections = nullptr;
    check_hip(hipMalloc(&projections, sizeof(float) * max_rows * model.hidden_size * (model.hc_count + 1ULL)),
        "known-window PLE row projections");
    w.projections = static_cast<float*>(projections);
    if (prefill) {
        void* gates = nullptr;
        check_hip(hipMalloc(&gates, sizeof(float) * max_rows * model.hc_count), "known-window PLE row gates");
        w.gates = static_cast<float*>(gates);
    }
    w.hashes.resize(max_rows);
    // The primary prefill owner is constructed before tail owners. Claim one
    // setup task only; the prompt ids and capped direct lookahead already exist.
    auto* reclaim = ple_reclaim::active();
    if (prefill && prime_resources && prime_resources->model_manifest && prime_resources->model_weights &&
        prime_resources->ple_hash_contract && reclaim && reclaim->claim_stage_prime()) {
        const auto& tensors = prime_resources->model_manifest->tensors;
        for (std::size_t i = 0; i < tensors.size(); ++i) if (tensors[i].role == TensorRole::ngram_table) {
            const auto* runtime = find_runtime_tensor_binding(*prime_resources->model_weights, static_cast<std::uint32_t>(i));
            if (!runtime) throw std::invalid_argument("PLE setup prime lacks its actual table binding");
            w.start_rolling_stage(reclaim, 0, prime_resources->ple_hash_contract->row_bytes, {&tensors[i], runtime});
            break;
        }
    }
}
HipPleKnownWindow::~HipPleKnownWindow() = default;
std::uint32_t HipPleKnownWindow::last_rows() const noexcept { return impl_->rows; }

void HipPleKnownWindow::preflight(std::span<const ExecutionDispatchContext> rows) const {
    const auto prepared = impl_->prepare(rows);
    // First-block source reads have already been primed by Prefetcher setup.
    // Start its copy while the preflight and earlier GPU steps are still ahead.
    auto* reclaim = ple_reclaim::active();
    if (impl_->prefill && rows[0].position == 0 && reclaim && reclaim->rolling() && !impl_->stage.active) {
        const auto* backend = dynamic_cast<FileMappedPleLookupBackend*>(rows[0].resources->ple_lookup);
        if (backend && !(rows[0].resources->ple_prefetch && rows[0].resources->ple_prefetch->active)) {
            const auto turn = backend->residency();
            impl_->start_rolling_stage(reclaim, 0, prepared[0].hash_contract.row_bytes, prepared[0].tensors.table, turn);
        }
    }
}

void HipPleKnownWindow::stage_serve(std::span<const ExecutionDispatchContext> rows) {
    auto& w=*impl_;
    static const bool enabled=[] { const char* v=std::getenv("STRIX_PLE_STAGE_SERVE"); return !v || v[0]!='0'; }();
    if(!enabled || !w.prefill || rows.size()<32U || ple_reclaim::active() || w.stage.active) return;
    const auto workers=ple_detail::serve_stage_workers(std::thread::hardware_concurrency());
    if(!workers) return;
    const char* copy=std::getenv("STRIX_MODEL_COPY_ALL");
    if(copy && copy[0] && copy[0]!='0') return;
    auto* backend=dynamic_cast<FileMappedPleLookupBackend*>(rows[0].resources->ple_lookup);
    if(!backend || backend->residency() || (rows[0].resources->ple_prefetch && rows[0].resources->ple_prefetch->active)) return;
    const auto prepared=w.prepare(rows);
    const auto& p=prepared[0];
    if(p.tensors.table.manifest->residency!=ResidencyPolicy::file_backed_candidate ||
        p.tensors.table.runtime->storage!=RuntimeTensorStorage::file_mapped) return;
    w.join_stage();
    auto& st=w.stage;
    // Called before the predecessor launch: it flips to slot^1, and the
    // successor flips back to slot. Never fill the predecessor's inline slot.
    st.serve=true;st.active=true;st.error=nullptr;st.slot=w.slot;st.workers=workers;
    st.first=rows[0].position;st.rows=rows.size();st.row_bytes=p.hash_contract.row_bytes;
    st.table_runtime=p.tensors.table.runtime;st.turn_id=0;st.epoch=0;
    st.ns=0;st.majflt=0;st.copy_ns=0;st.activity={};
    st.source={st.first,st.rows,st.row_bytes,st.table_runtime->allocation_generation,
        st.table_runtime,p.tensors.table.manifest,st.table_runtime->address};
    try {
        const auto context_count=static_cast<std::size_t>(w.config.model.ngram_size-1ULL);
        std::array<std::uint32_t,PleHashContract::maximum_ngram-1> previous{};
        if(!st.first) std::fill_n(previous.begin(),context_count,static_cast<std::uint32_t>(w.config.model.ple_eos_token_id));
        else std::memcpy(previous.data(),reinterpret_cast<const void*>(p.ngram_state.address),context_count*sizeof(std::uint32_t));
        st.row_ids.clear();st.row_ids.reserve(rows.size()*PleHashContract::maximum_heads);
        for(const auto& row:rows) {
            const auto hash=hash_ple_token(p.hash_contract,row.input_token_id,
                std::span<const std::uint32_t>(previous.data(),context_count));
            st.heads=hash.row_count;
            for(std::uint32_t h=0;h<hash.row_count;++h) st.row_ids.push_back(static_cast<std::uint32_t>(hash.row_ids[h]));
            std::copy_n(hash.next_context.begin(),context_count,previous.begin());
        }
        st.thread=std::thread([&st,destination=w.staging[st.slot],event=w.released[st.slot],
            wait=w.recorded[st.slot],stride=w.stride,table=p.tensors.table] {
            try {
                const auto t0=std::chrono::steady_clock::now();
                if(wait) check_hip(hipEventSynchronize(event),"PLE serve staged slot reuse");
                gather_staged_rows(destination,stride,st.row_ids,st.heads,st.row_bytes,table,&st.majflt,nullptr,{},st.workers);
                st.ns=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-t0).count());
            } catch(...) { st.error=std::current_exception(); }
        });
    } catch(...) { st.error=std::current_exception(); }
}

void HipPleKnownWindow::launch(std::span<const ExecutionDispatchContext> rows) {
    auto& w = *impl_;
    const auto launch_start = std::chrono::steady_clock::now();
    static const bool timing = [] { const char* v = std::getenv("STRIX_PREFILL_TIMING"); return v && v[0] == '1'; }();
    auto mark = std::chrono::steady_clock::now();
    const auto lap = [&](const char* name) {
        if (!timing || !w.prefill) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "prefill ple %-8s %8.2f ms\n", name, std::chrono::duration<double, std::milli>(now - mark).count());
        mark = now;
    };
    const auto prepared = w.prepare(rows);
    lap("prepare");
    const auto prepared_at = std::chrono::steady_clock::now();
    const auto& contract = prepared[0].hash_contract;
    auto& backend = *rows[0].resources->ple_lookup;
    auto* prefetch = rows[0].resources->ple_prefetch;
    const auto context_count = static_cast<std::size_t>(w.config.model.ngram_size - 1ULL);
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1> previous{};
    if (rows[0].position == 0)
        std::fill_n(previous.begin(), context_count, static_cast<std::uint32_t>(w.config.model.ple_eos_token_id));
    else
        std::memcpy(previous.data(), reinterpret_cast<const void*>(prepared[0].ngram_state.address),
            context_count * sizeof(std::uint32_t));
    // Serve lookahead may target the other slot. Keep it running, but join any
    // worker owning our selected slot before hashing can write even row 0.
    const auto needed_table=prepared[0].tensors.table;
    const ple_detail::StageSource needed{rows[0].position,rows.size(),contract.row_bytes,
        needed_table.runtime->allocation_generation,needed_table.runtime,needed_table.manifest,needed_table.runtime->address};
    const auto stage_action=ple_detail::serve_stage_action(w.stage.source,needed,w.stage.slot,w.slot^1U);
    const bool join_stage=w.stage.active && (!w.stage.serve || stage_action!=ple_detail::ServeStageAction::keep);
    ple_reclaim::Prefetcher::GatherReceipt reclaim_receipt;
    auto* const stats_reclaim = w.prefill ? ple_reclaim::active() : nullptr;
    const bool block_stats = stats_reclaim && stats_reclaim->block_stats();
    ple_reclaim::DirectActivity activity;
    const ple_reclaim::DirectReadScope block_read_scope(false, block_stats ? &activity : nullptr);
    if (join_stage) {
        const auto wait0 = std::chrono::steady_clock::now();
        w.join_stage();
        reclaim_receipt.stage_attempted = true;
        reclaim_receipt.stage_wait_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait0).count());
        reclaim_receipt.stage_ns = w.stage.ns;
        if (block_stats) {
            reclaim_receipt.bytes_read = w.stage.activity.bytes;
            reclaim_receipt.read_wait_ns = w.stage.activity.wait_ns;
            reclaim_receipt.copy_ns = w.stage.copy_ns;
        }
        reclaim_receipt.stage_majflt = w.stage.majflt;
        reclaim_receipt.stage_error = static_cast<bool>(w.stage.error);
        if (w.stage.error) {
            if (auto* r = ple_reclaim::active(); r && r->rolling()) std::rethrow_exception(w.stage.error);
            try { std::rethrow_exception(w.stage.error); }
            catch (const std::exception& e) { std::fprintf(stderr, "ple-reclaim: staged gather failed (%s); gathering inline\n", e.what()); }
            catch (...) { std::fprintf(stderr, "ple-reclaim: staged gather failed; gathering inline\n"); }
        }
    }
    const auto* staged_backend = dynamic_cast<FileMappedPleLookupBackend*>(&backend);
    const auto staged_turn = staged_backend ? staged_backend->residency() : nullptr;
    const bool same_stage_turn = staged_turn ? w.stage.turn_id == staged_turn->id() && w.stage.epoch == staged_turn->epoch()
        : w.stage.turn_id == 0U;
    const bool stage_candidate = join_stage && same_stage_turn && !w.stage.error && w.stage.slot == (w.slot ^ 1U) &&
        (!w.stage.serve || stage_action==ple_detail::ServeStageAction::consume) &&
        w.stage.first == rows[0].position && rows.size() <= w.stage.rows;   // any plan block size that fits
    if (join_stage) w.stage.active = false;
    w.slot ^= 1U;
    // The slot was last read by the window before the previous one.
    if (w.recorded[w.slot]) check_hip(hipEventSynchronize(w.released[w.slot]), "known-window PLE staging reuse");
    auto* staging = w.staging[w.slot];
    // Prefill blocks on the file-mapped backend: the n-gram hashes chain row to row, but each row's
    // gather (IQ4_NL decode to BF16) is independent, so the rows are hashed in order on this thread and
    // gathered by up to 8 threads, each with its own stateless file-mapped backend.
    static const std::uint64_t mapped_identity = FileMappedPleLookupBackend{}.implementation_identity();
    auto* const mapped_backend = dynamic_cast<FileMappedPleLookupBackend*>(&backend);
    const auto source_turn = mapped_backend ? mapped_backend->residency() : nullptr;
    const bool leased_prompt = source_turn && w.prefill && source_turn->prompt();
    const bool leased_verify = source_turn && !w.prefill;
    auto* const reclaim = w.prefill && (!source_turn || leased_prompt) ? ple_reclaim::active() : nullptr;
    const bool rolling = (reclaim && reclaim->rolling()) || leased_prompt;
    const bool parallel = w.prefill && (rows.size() >= 32U || rolling) && backend.implementation_identity() == mapped_identity;
    if (rolling && !parallel) throw std::runtime_error("PLE rolling requires the file-mapped gather staging consumer");
    // STRIX_PLE_RECLAIMABLE=1: receipts around this block's hash + gather (faults are process-wide counters).
    const auto reclaim_t0 = std::chrono::steady_clock::now();
    std::uint64_t reclaim_minflt = 0, reclaim_majflt = 0;
#if defined(__linux__)
    if (reclaim || leased_prompt) {
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            reclaim_minflt = static_cast<std::uint64_t>(usage.ru_minflt);
            reclaim_majflt = static_cast<std::uint64_t>(usage.ru_majflt);
        }
    }
#endif
    std::vector<std::size_t> deferred;
    const bool parallel_hash = reclaim && reclaim->rolling() && parallel && !(prefetch && prefetch->active);
    if (parallel_hash) {
        std::vector<std::uint32_t> tokens;
        tokens.reserve(rows.size());
        for (const auto& row : rows) tokens.push_back(row.input_token_id);
        hash_ple_tokens(contract, tokens, std::span<const std::uint32_t>(previous.data(), context_count),
            std::span<PleHashResult>(w.hashes.data(), rows.size()));
    }
    bool first_row_consumed = false;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        auto* destination = staging + r * w.stride;
        PleHashResult hash;
        if ((parallel || leased_verify) && !(r == 0 && prefetch && prefetch->active)) {
            hash = parallel_hash ? w.hashes[r] : hash_ple_token(contract, rows[r].input_token_id,
                std::span<const std::uint32_t>(previous.data(), context_count));
            if (static_cast<std::uint64_t>(hash.row_count) * contract.row_bytes > w.stride)
                throw std::overflow_error("PLE lookup exceeds window staging");
            deferred.push_back(r);
        } else if (r == 0 && prefetch && prefetch->active) {
            if (prefetch->lookup_bytes > w.stride) throw std::overflow_error("PLE prefetched lookup exceeds window staging");
            hash = finish_ple_lookup_prefetch(rows[0].input_token_id, rows[0].position, backend, *prefetch,
                std::span<std::byte>(destination, static_cast<std::size_t>(prefetch->lookup_bytes)));
            first_row_consumed = true;
        } else {
            hash = hash_ple_token(contract, rows[r].input_token_id,
                std::span<const std::uint32_t>(previous.data(), context_count));
            const auto bytes = static_cast<std::uint64_t>(hash.row_count) * contract.row_bytes;
            if (bytes > w.stride) throw std::overflow_error("PLE lookup exceeds window staging");
            PleLookupRequest lookup;
            lookup.row_ids = std::span<const std::uint64_t>(hash.row_ids.data(), hash.row_count);
            lookup.row_bytes = contract.row_bytes;
            lookup.table_manifest = prepared[r].tensors.table.manifest;
            lookup.table_runtime = prepared[r].tensors.table.runtime;
            PleLookupHandle handle = 0;
            try {
                handle = backend.start(lookup);
                backend.finish(handle, std::span<std::byte>(destination, static_cast<std::size_t>(bytes)));
                handle = 0;
            } catch (...) { if (handle) backend.cancel(handle); throw; }
        }
        if (static_cast<std::uint64_t>(hash.row_count) * contract.row_bytes != prepared[r].workspace_layout.gathered_bytes)
            throw std::invalid_argument("PLE window lookup does not match its gathered extent");
        std::copy_n(hash.next_context.begin(), context_count, previous.begin());
        w.hashes[r] = hash;
    }
    if (stage_candidate && (reclaim || w.stage.serve) && parallel &&
        w.stage.row_bytes == contract.row_bytes && w.stage.table_runtime == prepared[0].tensors.table.runtime) {
        const auto hashes=std::span<const PleHashResult>(w.hashes.data(),rows.size());
        if(w.stage.serve) {
            reclaim_receipt.staged_rows=ple_detail::accept_serve_staged_gather(w.stage.source,needed,
                hashes,w.stage.heads,w.stage.row_ids,first_row_consumed,deferred);
        } else reclaim_receipt.staged_rows = ple_detail::accept_staged_gather(
            hashes, w.stage.heads, w.stage.row_ids, first_row_consumed, deferred);
    }
    lap("hash");
    if (!w.prefill) ple_window_warm::window(std::span<const PleHashResult>(w.hashes.data(), rows.size()), launch_start);
    const auto hash_end = block_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (block_stats) reclaim_receipt.hash_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(hash_end - reclaim_t0).count());
    const auto gather_chunk = [&](std::size_t first, std::size_t last) {
        const auto copy0 = block_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        ple_reclaim::Prefetcher::PromptSource source;
        if (leased_prompt) source = {source_turn->prompt_view(rows[0].position + first, last - first),
            prepared[0].tensors.table.runtime->address, prepared[0].tensors.table.manifest->absolute_offset};
        else if (reclaim && reclaim->rolling()) source = reclaim->prompt_source(rows[0].position + first, last - first);
        // Keep verify conversion in its historical row order; prompt helpers
        // remain parallel. Both consume immutable owned bytes on leased turns.
        const auto hardware=std::max(1U,std::thread::hardware_concurrency());
        const auto available=w.stage.active && w.stage.serve ? hardware-w.stage.workers : hardware;
        const auto workers = w.prefill ? static_cast<std::size_t>(std::min(8U,available)) : 1U;
        std::vector<std::exception_ptr> errors(workers);
        const auto gather = [&](std::size_t worker) {
            try {
                FileMappedPleLookupBackend local;
                local.bind_residency(source_turn);
                if (source.view) local.bind_prompt_source(source.view, source.table, source.offset);
                for (std::size_t i = worker; i < deferred.size(); i += workers) {
                    const auto r = deferred[i];
                    if (r < first || r >= last) continue;
                    const auto& hash = w.hashes[r];
                    PleLookupRequest lookup;
                    lookup.row_ids = std::span<const std::uint64_t>(hash.row_ids.data(), hash.row_count);
                    lookup.row_bytes = contract.row_bytes;
                    lookup.table_manifest = prepared[r].tensors.table.manifest;
                    lookup.table_runtime = prepared[r].tensors.table.runtime;
                    const auto handle = local.start(lookup);
                    try {
                        local.finish(handle, std::span<std::byte>(staging + r * w.stride,
                            static_cast<std::size_t>(static_cast<std::uint64_t>(hash.row_count) * contract.row_bytes)));
                    } catch (...) { local.cancel(handle); throw; }
                }
            } catch (...) { errors[worker] = std::current_exception(); }
        };
        std::vector<std::thread> threads;
        struct JoinAll { std::vector<std::thread>& t; ~JoinAll() { for (auto& x : t) if (x.joinable()) x.join(); } } join_all{threads};
        threads.reserve(workers - 1U);
        for (std::size_t t = 1; t < workers; ++t) threads.emplace_back(gather, t);
        gather(0);
        for (auto& thread : threads) thread.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        if (block_stats) reclaim_receipt.copy_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - copy0).count());
    };
    if (leased_prompt && !reclaim_receipt.staged_rows) {
        source_turn->gather_prompt(rows[0].position, rows.size(), [&](auto first, auto count) {
            const auto offset = static_cast<std::size_t>(first - rows[0].position);
            gather_chunk(offset, offset + static_cast<std::size_t>(count));
        });
    } else if (leased_verify) {
        std::vector<std::uint64_t> ids;
        for (std::size_t r = 0; r < rows.size(); ++r)
            ids.insert(ids.end(), w.hashes[r].row_ids.begin(), w.hashes[r].row_ids.begin() + w.hashes[r].row_count);
        const auto* table = prepared[0].tensors.table.runtime;
        source_turn->verify_window(table->address, table->allocation_generation, rows.size(), ids,
            [&] { gather_chunk(0, rows.size()); });
    } else if (rolling && !leased_prompt && !reclaim_receipt.staged_rows) {
        // A consumed c1 hit has already advanced the source frontier. An inline
        // single-row prefetch still needs an empty-copy window to advance it.
        reclaim->gather_range(rows[0].position, rows.size(), [&](auto first, auto count) {
            const auto offset = static_cast<std::size_t>(first - rows[0].position);
            gather_chunk(offset, offset + static_cast<std::size_t>(count));
        });
    } else if (!deferred.empty()) gather_chunk(0, rows.size());
    lap("gather");
    if (reclaim || leased_prompt) {
        std::uint64_t minflt = 0, majflt = 0;
#if defined(__linux__)
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            minflt = static_cast<std::uint64_t>(usage.ru_minflt) - reclaim_minflt;
            majflt = static_cast<std::uint64_t>(usage.ru_majflt) - reclaim_majflt;
        }
#endif
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - reclaim_t0).count();
        reclaim_receipt.gather_ns = static_cast<std::uint64_t>(ns);
        reclaim_receipt.bytes_read += activity.bytes;
        reclaim_receipt.read_wait_ns += activity.wait_ns;
        reclaim_receipt.minflt = minflt;
        reclaim_receipt.majflt = majflt;
        if (source_turn) {
            std::vector<std::uint64_t> ids;
            for (std::size_t r = 0; r < rows.size(); ++r)
                ids.insert(ids.end(), w.hashes[r].row_ids.begin(), w.hashes[r].row_ids.begin() + w.hashes[r].row_count);
            source_turn->observe(rows[0].position, ids, reclaim_receipt.gather_ns, minflt, majflt);
        } else reclaim->on_gather(rows[0].position, std::span<const PleHashResult>(w.hashes.data(), rows.size()), reclaim_receipt);
    }
    if (!w.prefill) {
        // STRIX_CYCLE_TIMING=1: host time of a window's PLE prepare + hash + gather, printed at exit.
        struct Totals { double ms = 0, prep = 0; unsigned long long n = 0; bool on = std::getenv("STRIX_CYCLE_TIMING") != nullptr;
            ~Totals() { if (on && n) std::fprintf(stderr, "window PLE host prepare+gather: %llu x %.4f ms (prepare %.4f ms)\n", n,
                ms / static_cast<double>(n), prep / static_cast<double>(n)); } };
        static Totals totals;
        if (totals.on) {
            totals.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mark).count();
            totals.prep += std::chrono::duration<double, std::milli>(prepared_at - mark).count(); ++totals.n;
        }
    }
    if (reclaim && reclaim->rolling() && parallel)
        w.start_rolling_stage(reclaim, rows[0].position + rows.size(), contract.row_bytes, prepared[0].tensors.table, source_turn);
    void* device = nullptr;
    check_hip(hipHostGetDevicePointer(&device, staging, 0), "known-window PLE staging device view");
    const auto stream = hip_execution_stream();
    bool projected = true;
    if (w.prefill) {
        // 64-row wide projections, then eight-row groups for the rest (the window path's arithmetic
        // per row). STRIX_PREFILL_PLE_WIDE=0 keeps eight-row groups; STRIX_PREFILL_PLE_WIDE_CHECK=1
        // also runs them over the wide rows and compares every projected value bitwise.
        static const bool wide = [] { const char* v = std::getenv("STRIX_PREFILL_PLE_WIDE"); return !v || v[0] != '0'; }();
        static const bool check = [] { const char* v = std::getenv("STRIX_PREFILL_PLE_WIDE_CHECK"); return v && v[0] == '1'; }();
        const auto at = [&](std::uint32_t first) {
            return reinterpret_cast<const std::uint16_t*>(static_cast<std::byte*>(device) + first * w.stride);
        };
        const auto total = static_cast<std::uint32_t>(rows.size());
        std::uint32_t first = 0;
        static const bool batch = [] { const char* v = std::getenv("STRIX_PREFILL_PLE_BATCH"); return v && v[0] == '1' && v[1] == '\0'; }();
        if (wide && batch && !qpf2k::suppressed(qpf2k::Item::ple_batch) &&
            total >= kPleWideRows && w.project_rows_batch(prepared[0], at(0), total / kPleWideRows)) {
            static const bool witness = [] { std::fprintf(stderr, "qpf2k ple row batch active\n"); return true; }();
            (void)witness;
            first = total / kPleWideRows * kPleWideRows;
        }
        if (wide)
            while (total - first >= kPleWideRows && w.project_rows_wide(prepared[first], at(first), first)) first += kPleWideRows;
        if (check && first) {
            const auto expanded = static_cast<std::uint64_t>(prepared[0].expanded), hidden = static_cast<std::uint64_t>(w.config.model.hidden_size);
            const auto read = [&]() {
                std::vector<float> keys(first * expanded), values(first * hidden);
                check_hip(hipMemcpyAsync(keys.data(), w.projections, keys.size() * 4U, hipMemcpyDeviceToHost, stream), "PLE check keys");
                check_hip(hipMemcpyAsync(values.data(), w.projections + w.max_rows * expanded, values.size() * 4U, hipMemcpyDeviceToHost, stream),
                    "PLE check values");
                check_hip(hipStreamSynchronize(stream), "PLE check synchronize");
                return std::pair{keys, values};
            };
            const auto wide_values = read();
            for (std::uint32_t r = 0; r < first; r += 8U) static_cast<void>(w.project_rows(prepared[r], at(r), 8U, r));
            const auto rows_values = read();
            std::size_t differ = 0;
            for (std::size_t i = 0; i < wide_values.first.size(); ++i) differ += std::memcmp(&wide_values.first[i], &rows_values.first[i], 4U) != 0;
            for (std::size_t i = 0; i < wide_values.second.size(); ++i) differ += std::memcmp(&wide_values.second[i], &rows_values.second[i], 4U) != 0;
            std::fprintf(stderr, "PLE wide check: rows %u values %zu differ %zu\n", first, wide_values.first.size() + wide_values.second.size(), differ);
        }
        for (; first < total && projected; first += 8U) {
            const auto count = std::min<std::uint32_t>(8U, total - first);
            projected = w.project_rows(prepared[first], at(first), count, first);
        }
    } else projected = w.project_rows(prepared[0], static_cast<const std::uint16_t*>(device), static_cast<std::uint32_t>(rows.size()));
    const auto expanded = prepared[0].expanded, hidden = w.config.model.hidden_size;
    // Prefill rows run the PLE tail (gate, gated value, norm, convolution) as one rows-form launch
    // sequence instead of four launches per row, with the per-row arithmetic.
    // STRIX_PREFILL_PLE_ROWS=0 keeps the per-row launches; STRIX_PREFILL_PLE_ROWS_CHECK=1 runs both
    // from the same inputs and state and compares every output and the final state bitwise.
    static const bool tail_rows = [] { const char* v = std::getenv("STRIX_PREFILL_PLE_ROWS"); return !v || v[0] != '0'; }();
    static const bool tail_check = [] { const char* v = std::getenv("STRIX_PREFILL_PLE_ROWS_CHECK"); return v && v[0] == '1'; }();
    PleTailRows tail{};
    if (w.prefill && projected && tail_rows && w.tail_rows(prepared, tail)) {
        const auto n = static_cast<std::uint32_t>(rows.size());
        if (!tail_check) enqueue_ple_tail_rows(tail, hip_graph_status(), stream);
        else {
            const auto key_bytes = static_cast<std::size_t>(n) * expanded * 4U;
            const auto in_bytes = static_cast<std::size_t>(n - 1U) * tail.input_stride * 2U + expanded * 2U;
            std::vector<std::byte> keys(key_bytes), conv(w.conv_bytes), input(in_bytes);
            const auto copy = [&](void* to, const void* from, std::size_t bytes, hipMemcpyKind kind) {
                check_hip(hipMemcpyAsync(to, from, bytes, kind, stream), "PLE rows check copy");
            };
            copy(keys.data(), tail.keys, key_bytes, hipMemcpyDeviceToHost);
            copy(conv.data(), tail.conv_state, w.conv_bytes, hipMemcpyDeviceToHost);
            copy(input.data(), tail.input, in_bytes, hipMemcpyDeviceToHost);
            check_hip(hipStreamSynchronize(stream), "PLE rows check save");
            const auto read = [&] {
                std::vector<std::uint16_t> out(static_cast<std::size_t>(n) * expanded + w.conv_bytes / 2U);
                for (std::uint32_t r = 0; r < n; ++r)
                    copy(out.data() + static_cast<std::size_t>(r) * expanded, tail.output + static_cast<std::uint64_t>(r) * tail.output_stride,
                        expanded * 2U, hipMemcpyDeviceToHost);
                copy(out.data() + static_cast<std::size_t>(n) * expanded, tail.conv_state, w.conv_bytes, hipMemcpyDeviceToHost);
                check_hip(hipStreamSynchronize(stream), "PLE rows check read");
                return out;
            };
            enqueue_ple_tail_rows(tail, hip_graph_status(), stream);
            const auto rows_values = read();
            copy(tail.keys, keys.data(), key_bytes, hipMemcpyHostToDevice);
            copy(tail.conv_state, conv.data(), w.conv_bytes, hipMemcpyHostToDevice);
            copy(const_cast<std::uint16_t*>(tail.input), input.data(), in_bytes, hipMemcpyHostToDevice);
            for (std::size_t r = 0; r < rows.size(); ++r)
                enqueue_ple_device(prepared[r], reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(device) + r * w.stride),
                    w.projections + r * expanded, w.projections + w.max_rows * expanded + r * hidden);
            const auto row_values = read();
            std::size_t differ = 0;
            for (std::size_t i = 0; i < rows_values.size(); ++i) differ += rows_values[i] != row_values[i];
            std::fprintf(stderr, "PLE rows check: rows %u values %zu differ %zu\n", n, rows_values.size(), differ);
        }
    } else for (std::size_t r = 0; r < rows.size(); ++r) {
        auto* gathered = reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(device) + r * w.stride);
        if (projected)
            enqueue_ple_device(prepared[r], gathered, w.projections + r * expanded,
                w.projections + w.max_rows * expanded + r * hidden);
        else enqueue_ple_device(prepared[r], gathered);
        if (w.prefill) continue;
        check_hip(hipMemcpyAsync(w.snapshots + r * w.conv_bytes, reinterpret_cast<const void*>(prepared[r].conv_state.address),
            w.conv_bytes, hipMemcpyDeviceToDevice, stream), "known-window PLE convolution snapshot");
    }
    check_hip(hipEventRecord(w.released[w.slot], stream), "known-window PLE staging release");
    w.recorded[w.slot] = true;
    if (reclaim && !reclaim->rolling() && parallel) {
        std::vector<std::uint32_t> ids;
        const auto next = rows[0].position + rows.size();
        // The next block's size is the plan's (T, or a STRIX_QPF2K_SCHED merged/balanced partial block): stage the
        // window's capacity (clipped to the prompt); the next launch uses its first rows.size() rows.
        if (const auto staged = reclaim->stage_rows(next, w.max_rows, ids)) {
            auto& st = w.stage;
            st.first = next; st.rows = staged; st.slot = w.slot ^ 1U; st.heads = reclaim->heads();
            st.row_ids = std::move(ids); st.error = nullptr; st.ns = 0; st.majflt = 0; st.active = true;
            st.row_bytes = contract.row_bytes; st.table_runtime = prepared[0].tensors.table.runtime;
            const bool tracked_stage = reclaim->rolling();
            if (tracked_stage) reclaim->stage_begin();
            try { st.thread = std::thread([&st, destination = w.staging[st.slot], released = w.released[st.slot],
                                     wait = w.recorded[st.slot], stride = w.stride, row_bytes = contract.row_bytes,
                                     table = prepared[0].tensors.table, reclaim, tracked_stage] {
                const ple_reclaim::DirectReadScope read_scope(tracked_stage);
                try {
                    const auto t0 = std::chrono::steady_clock::now();
                    // The slot was last read by the previous block's PLE launches.
                    if (wait) check_hip(hipEventSynchronize(released), "known-window PLE staged slot reuse");
                    if (reclaim->rolling()) {
                        reclaim->gather_range(st.first, st.rows, [&](auto first, auto count) {
                            const auto offset = first - st.first;
                            gather_staged_rows(destination + offset * stride, stride,
                                std::span<const std::uint32_t>(st.row_ids).subspan(offset * st.heads, count * st.heads),
                                st.heads, row_bytes, table, &st.majflt);
                        });
                    } else gather_staged_rows(destination, stride, st.row_ids, st.heads, row_bytes, table, &st.majflt);
                    st.ns = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
                } catch (...) { st.error = std::current_exception(); }
                if (tracked_stage) reclaim->stage_end(st.error);
            }); } catch (...) {
                if (tracked_stage) reclaim->stage_end(std::current_exception());
                throw;
            }
        }
    }
    w.rows = static_cast<std::uint32_t>(rows.size());
    w.conv_address = prepared[0].conv_state.address; w.ngram_address = prepared[0].ngram_state.address;
    if (w.prefill) {
        // Every prefill row is kept: the convolution state already is the last row's; publish its history.
        const auto& hash = w.hashes[rows.size() - 1U];
        std::memcpy(reinterpret_cast<void*>(w.ngram_address), hash.next_context.data(),
            static_cast<std::size_t>(hash.context_count) * sizeof(std::uint32_t));
        ple_window_warm::register_history(hash);
        w.rows = 0;
    } else ple_window_warm::register_source(source_turn, *prepared[0].tensors.table.runtime, contract);
}

void HipPleKnownWindow::commit(std::uint32_t n_keep, void* stream) {
    if (impl_->prefill) throw std::logic_error("prefill PLE rows publish their history in launch()");
    auto& w = *impl_;
    if (!w.rows || !n_keep || n_keep > w.rows) throw std::invalid_argument("known-window PLE commit outside the last window");
    if (n_keep < w.rows)
        check_hip(hipMemcpyAsync(reinterpret_cast<void*>(w.conv_address), w.snapshots + (n_keep - 1U) * w.conv_bytes,
            w.conv_bytes, hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream)), "known-window PLE convolution commit");
    const auto& hash = w.hashes[n_keep - 1U];
    std::memcpy(reinterpret_cast<void*>(w.ngram_address), hash.next_context.data(),
        static_cast<std::size_t>(hash.context_count) * sizeof(std::uint32_t));
    ple_window_warm::register_history(hash);
    w.rows = 0;
}

}  // namespace strix
