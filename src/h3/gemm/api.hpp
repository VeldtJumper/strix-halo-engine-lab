#pragma once
// MiniMax H3 int8 convrot projections: the public device API (consumers include this header only).
//
//   Weights w{codes, scales, n, k};                    // checkpoint `weight` int8 [n][k], `weight_scale` fp32 [n] (device)
//   Workspace ws(max_rows, max_k);                     // owns activation codes, row scales, the flush counter
//   linear(x, x_ld, rows, w, ws, y, y_ld, stream);     // y[rows][n] bf16 = the host reference project() (ref.hpp)
//
//   fc1 + SwiGLU (the fc2 input): permute the fc1 weights ONCE at load time,
//   SwiGLUWeights f = permute_fc1_for_swiglu(host_codes, host_scales, 28672, 5376) -> upload f.codes / f.scales;
//   linear_fc1_swiglu(x, x_ld, rows, {dev codes, dev scales, 28672, 5376}, ws, h, h_ld, stream);
//   h[rows][n / 2] bf16 = bf16(bf16(silu(gate)) * up) of the reference fc1 output halves [gate | up] (project_fc1_swiglu).
//
// Shapes: rows >= 1, k > 0, k % 256 == 0, k <= 16384, n > 0, n % 256 == 0 (SwiGLU: n % 512 == 0). x: device bf16 rows,
// pointer and row pitch (x_ld elements, x_ld >= k) 16-byte aligned. y / h: device, pitch >= the output columns (rows
// never overlap). QKV: one [rows][21504] buffer, q | k | v = columns [0, 7168), [7168, 14336), [14336, 21504).
// Every output row depends only on its own input row and the weights (batch invariant; tile choice never changes bits).
//
// Errors: invalid arguments throw std::invalid_argument before anything is queued; a failed launch (hipGetLastError
// after each kernel) throws std::runtime_error. Execution itself is asynchronous: faults during the kernels surface at the
// caller's next synchronizing call on the stream (hipStreamSynchronize / hipEventSynchronize / a blocking copy), which the
// caller must check. Calls on one stream are ordered; a Workspace must not serve calls in flight on different streams.
//
// Rotation exactness: the rotation is exact within 2^-47 of each 256-group's largest magnitude: nonzero inputs more than
// kFixShift = 47 binades below the group maximum are flushed to zero before the transform and the group is counted in the
// workspace's flush counter (ws.flush_count(stream) synchronizes that stream; ws.reset_flush_count(stream)). This is
// deterministic and batch invariant and far inside the G1 bound (the reference's own fp32-accumulated rotation errs by
// ~gamma_255 * sum|x|, about 2^-16 relative). It does happen on real data: late-block fc2 inputs (SwiGLU outputs of very
// negative gates) flush tens of thousands of groups per forward. The counter is telemetry; callers do not throw on it.
//
// Addressing limit: rows * k < 2^32 and n * k < 2^32 (32-bit operand offsets in the GEMM; K 14336: rows <= 299,593,
// K 16384: rows <= 262,143); larger calls are rejected (split the rows).

#include <hip/hip_runtime.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "h3/gemm/kernels.hpp"

namespace strix::h3::gemm {

struct Weights {
    const std::int8_t* codes = nullptr;  // device, [n][k] row-major (16-byte aligned)
    const float* scales = nullptr;       // device, [n]
    std::uint32_t n = 0, k = 0;
};

namespace detail {
inline void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("h3 gemm: ") + what + ": " + hipGetErrorString(e));
}
inline bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16U == 0U; }
}  // namespace detail

// Owning, move-only workspace (codes [max_rows][max_k], fp32 row scales, one flush counter).
class Workspace {
public:
    Workspace() = default;
    Workspace(std::uint32_t max_rows, std::uint32_t max_k) {
        if (!max_rows || !max_k) throw std::invalid_argument("h3 gemm: empty workspace");
        try {
            detail::hip_check(hipMalloc(&codes_, static_cast<std::size_t>(max_rows) * max_k), "workspace codes");
            detail::hip_check(hipMalloc(&row_scale_, static_cast<std::size_t>(max_rows) * 4U), "workspace scales");
            detail::hip_check(hipMalloc(&flush_, 4U), "workspace flush counter");
            detail::hip_check(hipMemset(flush_, 0, 4U), "workspace flush counter");
            // the default-stream memset must be complete before any (possibly non-blocking) stream uses the counter
            detail::hip_check(hipDeviceSynchronize(), "workspace initialization");
        } catch (...) {
            release();
            throw;
        }
        max_rows_ = max_rows; max_k_ = max_k;
    }
    ~Workspace() { release(); }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&& o) noexcept { swap(o); }
    Workspace& operator=(Workspace&& o) noexcept { if (this != &o) { release(); swap(o); } return *this; }
    void release() noexcept {
        if (codes_) static_cast<void>(hipFree(codes_));
        if (row_scale_) static_cast<void>(hipFree(row_scale_));
        if (flush_) static_cast<void>(hipFree(flush_));
        codes_ = nullptr; row_scale_ = nullptr; flush_ = nullptr; max_rows_ = 0; max_k_ = 0;
    }
    // Groups flushed since the last reset, read in `stream` order: the copy is queued on the producer stream and the
    // stream is synchronized, so every kernel queued before it on that stream has finished (works for non-blocking
    // streams). Work on OTHER streams is not covered: read once per producer stream.
    unsigned flush_count(hipStream_t stream) const {
        unsigned v = 0;
        detail::hip_check(hipMemcpyAsync(&v, flush_, 4U, hipMemcpyDeviceToHost, stream), "flush counter read");
        detail::hip_check(hipStreamSynchronize(stream), "flush counter read");
        return v;
    }
    void reset_flush_count(hipStream_t stream) { detail::hip_check(hipMemsetAsync(flush_, 0, 4U, stream), "flush counter reset"); }
    std::int8_t* codes() const { return codes_; }
    float* row_scale() const { return row_scale_; }
    unsigned* flush() const { return flush_; }
    std::uint32_t max_rows() const { return max_rows_; }
    std::uint32_t max_k() const { return max_k_; }
private:
    void swap(Workspace& o) noexcept {
        std::swap(codes_, o.codes_); std::swap(row_scale_, o.row_scale_); std::swap(flush_, o.flush_);
        std::swap(max_rows_, o.max_rows_); std::swap(max_k_, o.max_k_);
    }
    std::int8_t* codes_ = nullptr;
    float* row_scale_ = nullptr;
    unsigned* flush_ = nullptr;
    std::uint32_t max_rows_ = 0, max_k_ = 0;
};

// fc1 rows reordered for the fused SwiGLU epilogue (host side, once per weight matrix).
struct SwiGLUWeights {
    std::vector<std::int8_t> codes;
    std::vector<float> scales;
};
inline SwiGLUWeights permute_fc1_for_swiglu(const std::int8_t* codes, const float* scales, std::uint32_t n, std::uint32_t k) {
    if (!n || n % 512U || !k) throw std::invalid_argument("h3 gemm: swiglu permutation needs n % 512 == 0 and k > 0");
    SwiGLUWeights out;
    out.codes.resize(static_cast<std::size_t>(n) * k);
    out.scales.resize(n);
    for (std::uint32_t p = 0; p < n; ++p) {
        const std::uint32_t src = swiglu_source_row(p, n);
        std::copy(codes + static_cast<std::size_t>(src) * k, codes + static_cast<std::size_t>(src + 1U) * k, out.codes.begin() + static_cast<std::ptrdiff_t>(p) * k);
        out.scales[p] = scales[src];
    }
    return out;
}

// The exact silu table on the current device (uploaded once per device, 128 KiB, process lifetime).
inline const std::uint16_t* device_silu_table() {
    static std::mutex mutex;
    static std::map<int, void*> tables;
    int device = 0;
    detail::hip_check(hipGetDevice(&device), "hipGetDevice");
    const std::lock_guard lock(mutex);
    auto it = tables.find(device);
    if (it != tables.end()) return static_cast<const std::uint16_t*>(it->second);
    void* p = nullptr;
    detail::hip_check(hipMalloc(&p, 65536U * 2U), "silu table allocation");
    const hipError_t e = hipMemcpy(p, silu_table().data(), 65536U * 2U, hipMemcpyHostToDevice);
    if (e != hipSuccess) { static_cast<void>(hipFree(p)); detail::hip_check(e, "silu table upload"); }
    tables.emplace(device, p);
    return static_cast<const std::uint16_t*>(p);
}

namespace detail {
inline void check(const std::uint16_t* x, std::uint64_t x_ld, std::uint32_t rows, const Weights& w, const Workspace& ws,
    const std::uint16_t* y, std::uint32_t y_ld, bool swiglu) {
    const std::uint32_t cols = swiglu ? w.n / 2U : w.n;
    const char* bad = nullptr;
    if (!rows) bad = "rows == 0";
    else if (!w.k || w.k % 256U || w.k > 16384U) bad = "k must be a positive multiple of 256, at most 16384";
    else if (!w.n || w.n % (swiglu ? 512U : 256U)) bad = swiglu ? "n must be a positive multiple of 512" : "n must be a positive multiple of 256";
    else if (!w.codes || !w.scales || !aligned16(w.codes)) bad = "weights missing or not 16-byte aligned";
    else if (static_cast<std::uint64_t>(rows) * w.k > 0xffffffffULL || static_cast<std::uint64_t>(w.n) * w.k > 0xffffffffULL)
        bad = "rows * k and n * k must stay below 2^32 (split the rows)";
    else if (!ws.codes() || rows > ws.max_rows() || w.k > ws.max_k()) bad = "workspace too small";
    else if (!x || !aligned16(x) || x_ld < w.k || (x_ld * 2U) % 16U) bad = "input pointer / pitch (>= k, 16-byte aligned rows)";
    else if (!y || y_ld < cols) bad = "output pointer / pitch (>= output columns)";
    if (bad) throw std::invalid_argument(std::string("h3 gemm: ") + bad);
}
inline void launched(const char* what) { hip_check(hipGetLastError(), what); }
// Tile width: measured fastest per shape on gfx1151; identical bits either way.
inline bool wide_tiles(std::uint32_t rows, const Weights& w) { return w.n >= 16384U && rows >= 32768U; }
}  // namespace detail

inline void linear(const std::uint16_t* x, std::uint64_t x_ld, std::uint32_t rows, const Weights& w, Workspace& ws, std::uint16_t* y,
    std::uint32_t y_ld, hipStream_t stream) {
    detail::check(x, x_ld, rows, w, ws, y, y_ld, false);
    static_cast<void>(hipGetLastError());  // clear a stale error so the checks below report this call's launches
    launch_quantize(x, x_ld, rows, w.k, ws.codes(), ws.row_scale(), ws.flush(), stream);
    detail::launched("quantize launch");
    dev::GemmArgs a;
    a.a = ws.codes(); a.b = w.codes; a.row_scale = ws.row_scale(); a.ch_scale = w.scales; a.m = rows; a.n = w.n; a.k = w.k;
    a.out.ptr[0] = y; a.out.ld = y_ld; a.out.seg_cols = w.n;
    if (detail::wide_tiles(rows, w)) launch_gemm<256U, dev::Epi::plain>(a, stream);
    else launch_gemm<128U, dev::Epi::plain>(a, stream);
    detail::launched("gemm launch");
}

inline void linear_fc1_swiglu(const std::uint16_t* x, std::uint64_t x_ld, std::uint32_t rows, const Weights& permuted, Workspace& ws,
    std::uint16_t* h, std::uint32_t h_ld, hipStream_t stream) {
    detail::check(x, x_ld, rows, permuted, ws, h, h_ld, true);
    const std::uint16_t* table = device_silu_table();
    static_cast<void>(hipGetLastError());
    launch_quantize(x, x_ld, rows, permuted.k, ws.codes(), ws.row_scale(), ws.flush(), stream);
    detail::launched("quantize launch");
    dev::GemmArgs a;
    a.a = ws.codes(); a.b = permuted.codes; a.row_scale = ws.row_scale(); a.ch_scale = permuted.scales; a.silu = table;
    a.m = rows; a.n = permuted.n; a.k = permuted.k;
    a.out.ptr[0] = h; a.out.ld = h_ld; a.out.seg_cols = permuted.n / 2U;
    launch_gemm<256U, dev::Epi::swiglu>(a, stream);
    detail::launched("gemm launch");
}

}  // namespace strix::h3::gemm
