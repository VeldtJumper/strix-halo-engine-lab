#pragma once
// FirstBlockCache device side (identity h3-fbc-v1, FAST-class, OFF by default). Arithmetic and order: fbc_ref.hpp
// (device chunk sums are bitwise equal to chunk_sums_host). Per model call (rows = local rows, h = the local residual
// stream [rows][width] bf16, contiguous), all calls on ONE stream (bound at begin; another stream throws):
//   fbc.begin(h, rows, s);                          // before block 0: keeps h0
//   <block 0>
//   auto ch = fbc.after_block0(h, rows, s);         // r = bf16(h - h0), keeps h1; chunk sums when ready (syncs s)
//   1 node : Decision d = decide(split_block_sums(ch, rows, cut), rows * width, thr)
//   2 nodes: Sums mine = block_sums(ch, 0, ch.size()); exchange (16 bytes each way); decide({block 0, block 1}, S * width, thr)
//   switch (fbc.resolve(d, h, s)):                  // applies the decision
//     kReuse        -> h holds h_last = bf16(h1 + tail_ref); skip blocks 1..
//     kFullCommit   -> run blocks 1.., then fbc.commit(h_last, rows, s)
//     kFullNoCommit -> (non-finite score: cache cleared) run blocks 1.., no commit
//   reset() per sampling run and before a shape change (reset also abandons a call in progress). A failed launch / copy
//   invalidates the cache (ready() false) and throws. The holder is bound to ONE stream for its lifetime (the first
//   begin); any call on another stream throws, so a pending commit kernel can never race the next call's copies.
//   begin() is accepted only between calls (after construction, reset, commit, a kReuse or a kFullNoCommit resolve). Device memory: 5 x max_rows x width x 2 B (h0, h1, r, r_ref, tail) = 2.04 GB at 38K x 5376.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "h3/levers/fbc/fbc_ref.hpp"

namespace strix::h3::levers::fbc {

namespace dev {
template <int = 0>
__global__ __launch_bounds__(256) void sub_kernel(const std::uint16_t* __restrict__ a, const std::uint16_t* __restrict__ b, std::uint16_t* __restrict__ o,
    std::uint64_t n) {
#pragma clang fp contract(off)
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x; i < n; i += static_cast<std::uint64_t>(gridDim.x) * 256U)
        o[i] = gemm::f32_to_bf16(gemm::bf16_to_f32(a[i]) - gemm::bf16_to_f32(b[i]));
}
template <int = 0>
__global__ __launch_bounds__(256) void add_kernel(std::uint16_t* __restrict__ a, const std::uint16_t* __restrict__ b, std::uint64_t n) {
#pragma clang fp contract(off)
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x; i < n; i += static_cast<std::uint64_t>(gridDim.x) * 256U)
        a[i] = gemm::f32_to_bf16(gemm::bf16_to_f32(a[i]) + gemm::bf16_to_f32(b[i]));
}
// one work-group per 64-row chunk; lane order and tree exactly as chunk_sums_host
template <int = 0>
__global__ __launch_bounds__(256) void chunk_kernel(const std::uint16_t* __restrict__ r, const std::uint16_t* __restrict__ rref, std::uint32_t rows,
    std::uint32_t width, double* __restrict__ out) {
#pragma clang fp contract(off)
    __shared__ double sn[kLanes], sd[kLanes];
    const std::uint32_t c = blockIdx.x, t = threadIdx.x, r0 = c * kChunkRows;
    const std::uint32_t nr = rows - r0 < kChunkRows ? rows - r0 : kChunkRows;
    const std::uint64_t n = static_cast<std::uint64_t>(nr) * width, base = static_cast<std::uint64_t>(r0) * width;
    double an = 0.0, ad = 0.0;
    for (std::uint64_t e = t; e < n; e += kLanes) {
        const float rr = gemm::bf16_to_f32(rref[base + e]);
        const float dv = gemm::bf16_to_f32(gemm::f32_to_bf16(gemm::bf16_to_f32(r[base + e]) - rr));
        an += __builtin_fabs(static_cast<double>(dv));
        ad += __builtin_fabs(static_cast<double>(rr));
    }
    sn[t] = an;
    sd[t] = ad;
    __syncthreads();
    for (std::uint32_t s = kLanes / 2U; s > 0U; s >>= 1U) {
        if (t < s) {
            sn[t] += sn[t + s];
            sd[t] += sd[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        out[2U * c] = sn[0];
        out[2U * c + 1U] = sd[0];
    }
}

// Owning device allocation (move-only).
class Mem {
public:
    Mem() = default;
    explicit Mem(std::size_t bytes) {
        if (hipMalloc(&p_, bytes) != hipSuccess) {
            p_ = nullptr;
            static_cast<void>(hipGetLastError());
            throw std::runtime_error("h3 fbc: hipMalloc failed (" + std::to_string(bytes) + " bytes)");
        }
    }
    Mem(Mem&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    Mem& operator=(Mem&& o) noexcept {
        if (this != &o) {
            release();
            p_ = std::exchange(o.p_, nullptr);
        }
        return *this;
    }
    Mem(const Mem&) = delete;
    Mem& operator=(const Mem&) = delete;
    ~Mem() { release(); }
    template <class T> T* as() const { return static_cast<T*>(p_); }

private:
    void release() {
        if (p_) static_cast<void>(hipFree(p_));
        p_ = nullptr;
    }
    void* p_ = nullptr;
};
}  // namespace dev

class FirstBlockCacheDevice {
public:
    // fail_alloc_at (tests only): allocation number (0-based) that is made to fail, -1 = none.
    FirstBlockCacheDevice(std::uint32_t max_rows, std::uint32_t width, int fail_alloc_at = -1) : max_rows_(max_rows), w_(width) {
        check_rows(max_rows);
        const std::size_t bytes = checked_elems(max_rows, width) * 2U;
        int k = 0;
        const auto alloc = [&](std::size_t b) {
            if (k++ == fail_alloc_at) throw std::runtime_error("h3 fbc: injected allocation failure");
            return dev::Mem(b);
        };
        h0_ = alloc(bytes);
        h1_ = alloc(bytes);
        r_ = alloc(bytes);
        rref_ = alloc(bytes);
        tail_ = alloc(bytes);
        sums_ = alloc((static_cast<std::size_t>(max_rows) / kChunkRows + 1U) * 2U * sizeof(double));
    }
    FirstBlockCacheDevice(const FirstBlockCacheDevice&) = delete;
    FirstBlockCacheDevice& operator=(const FirstBlockCacheDevice&) = delete;
    bool ready() const { return ready_; }
    std::size_t device_bytes() const { return 5U * static_cast<std::size_t>(max_rows_) * w_ * 2U; }
    void reset() {
        ready_ = false;
        phase_ = Phase::kIdle;
    }

    // Tests only: make the next guarded launch / copy report a failure.
    void inject_failure_for_test() { fail_next_ = true; }

    void begin(const std::uint16_t* h, std::uint32_t rows, hipStream_t s) {
        if (phase_ != Phase::kIdle) throw std::logic_error("h3 fbc: begin while a call is in progress (resolve / commit it, or reset())");
        if (!h) throw std::invalid_argument("h3 fbc: null h");
        if (bound_ && s != s_) throw std::invalid_argument("h3 fbc: the holder is bound to the stream of its first begin");
        shape(rows);
        s_ = s;
        bound_ = true;
        guard(hipMemcpyAsync(h0_.as<void>(), h, bytes(rows), hipMemcpyDeviceToDevice, s), "copy h0");
        phase_ = Phase::kBegun;
    }
    // Returns the chunk sums (empty when the cache is not ready: cold start -> resolve gives kFullCommit).
    std::vector<Sums> after_block0(const std::uint16_t* h, std::uint32_t rows, hipStream_t s) {
        if (phase_ != Phase::kBegun) throw std::logic_error("h3 fbc: after_block0 without begin");
        if (!h) throw std::invalid_argument("h3 fbc: null h");
        same(s);
        if (rows != rows_) throw std::invalid_argument("h3 fbc: rows differ from begin");
        const std::uint64_t n = static_cast<std::uint64_t>(rows) * w_;
        static_cast<void>(hipGetLastError());
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::sub_kernel<0>), dim3(grid(n)), dim3(256), 0, s, h, h0_.as<std::uint16_t>(), r_.as<std::uint16_t>(), n);
        guard(hipGetLastError(), "launch r");
        guard(hipMemcpyAsync(h1_.as<void>(), h, bytes(rows), hipMemcpyDeviceToDevice, s), "copy h1");
        std::vector<Sums> out;
        if (ready_) {
            const std::uint32_t chunks = (rows + kChunkRows - 1U) / kChunkRows;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::chunk_kernel<0>), dim3(chunks), dim3(256), 0, s, r_.as<std::uint16_t>(), rref_.as<std::uint16_t>(), rows, w_,
                sums_.as<double>());
            guard(hipGetLastError(), "launch sums");
            std::vector<double> raw(2U * chunks);
            guard(hipMemcpyAsync(raw.data(), sums_.as<void>(), raw.size() * sizeof(double), hipMemcpyDeviceToHost, s), "sums");
            guard(hipStreamSynchronize(s), "sync");
            for (std::uint32_t c = 0; c < chunks; ++c) out.push_back(Sums{raw[2U * c], raw[2U * c + 1U]});
        }
        phase_ = Phase::kDecide;
        return out;
    }
    // Applies the decision (from decide() on the block sums; ignored on a cold start). h = h1 on entry.
    Action resolve(const Decision& d, std::uint16_t* h, hipStream_t s) {
        if (phase_ != Phase::kDecide) throw std::logic_error("h3 fbc: resolve without after_block0");
        same(s);
        const Action a = action_of(ready_, d);
        if (a == Action::kReuse) {
            const std::uint64_t n = static_cast<std::uint64_t>(rows_) * w_;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::add_kernel<0>), dim3(grid(n)), dim3(256), 0, s, h, tail_.as<std::uint16_t>(), n);
            guard(hipGetLastError(), "launch reuse");
        }
        if (a == Action::kFullNoCommit) ready_ = false;
        phase_ = a == Action::kFullCommit ? Phase::kAwaitCommit : Phase::kIdle;
        return a;
    }
    // After the last block of a kFullCommit call: r_ref = r, tail_ref = bf16(h_last - h1).
    void commit(const std::uint16_t* h_last, std::uint32_t rows, hipStream_t s) {
        if (phase_ != Phase::kAwaitCommit) throw std::logic_error("h3 fbc: commit outside a kFullCommit call");
        if (!h_last) throw std::invalid_argument("h3 fbc: null h_last");
        same(s);
        if (rows != rows_) throw std::invalid_argument("h3 fbc: rows differ from begin");
        const std::uint64_t n = static_cast<std::uint64_t>(rows) * w_;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dev::sub_kernel<0>), dim3(grid(n)), dim3(256), 0, s, h_last, h1_.as<std::uint16_t>(), tail_.as<std::uint16_t>(), n);
        guard(hipGetLastError(), "launch tail");  // tail_ may be partly written: guard invalidated the cache
        std::swap(r_, rref_);
        ready_ = true;
        phase_ = Phase::kIdle;
    }

private:
    enum class Phase : std::uint8_t { kIdle, kBegun, kDecide, kAwaitCommit };
    void guard(hipError_t e, const char* what) {
        if (fail_next_) {
            fail_next_ = false;
            e = hipErrorUnknown;
        }
        if (e != hipSuccess) {
            ready_ = false;
            phase_ = Phase::kIdle;
            throw std::runtime_error(std::string("h3 fbc: ") + what + ": " + hipGetErrorString(e));
        }
    }
    void same(hipStream_t s) const {
        if (!bound_ || s != s_) throw std::invalid_argument("h3 fbc: all calls of a model call must use the stream given to begin");
    }
    static std::uint32_t grid(std::uint64_t n) { const std::uint64_t g = (n + 255U) / 256U; return static_cast<std::uint32_t>(g < 4096U ? (g ? g : 1U) : 4096U); }
    void shape(std::uint32_t rows) {
        if (!rows || rows > max_rows_) throw std::invalid_argument("h3 fbc: rows out of range");
        if (ready_ && rows != rows_) throw std::invalid_argument("h3 fbc: row count changed without reset");
        rows_ = rows;
    }
    std::size_t bytes(std::uint32_t rows) const { return static_cast<std::size_t>(rows) * w_ * 2U; }
    std::uint32_t max_rows_, w_, rows_ = 0;
    bool ready_ = false, bound_ = false, fail_next_ = false;
    Phase phase_ = Phase::kIdle;
    hipStream_t s_ = nullptr;
    dev::Mem h0_, h1_, r_, rref_, tail_, sums_;
};

}  // namespace strix::h3::levers::fbc
