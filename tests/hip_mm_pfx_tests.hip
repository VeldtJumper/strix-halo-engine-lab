// strix-hip-mm-pfx-tests: the MiniMax prefill exchange module (src/minimax/pfx) on ONE GPU, two ranks in one process
// over the in-memory transport pair: bytes delivered in place (TX = local, RX = peer) for many sequences (slot
// wraparound at depth 2..4, lag 1..3 sends ahead of the receives), uneven sizes (chunk tails, sub-chunk messages),
// small and large chunk sizes, asymmetric rank timing; failures: a corrupted payload chunk (CRC mismatch, the stream
// is released, drain() throws) and a header mismatch (both ranks throw). Prints PASS/FAIL per case; exit 1 on failure.
#include "minimax/pfx/pf_exchange.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace px = strix::minimax::pfx;
namespace sd = strix::dist;

// Both ranks share ONE GPU here: hipMalloc / hipFree / the exchanger's constructor synchronise the device, which waits
// for the other rank's stream parked on an RX flag that only this rank can satisfy -> deadlock until the transport's
// 30 s timeout. So all allocation happens before a two-rank barrier and all freeing after one (the nodes have one GPU
// per rank and the engine allocates at start-up, so this is a property of the single-GPU test only).
class Barrier {
public:
    void arrive_and_wait() {
        std::unique_lock<std::mutex> g(m_);
        const unsigned gen = gen_;
        if (++count_ == 2) {
            count_ = 0;
            ++gen_;
            cv_.notify_all();
            return;
        }
        cv_.wait(g, [&] { return gen_ != gen; });
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    unsigned count_ = 0, gen_ = 0;
};

// Hang diagnosis: live exchangers register here; a watchdog prints their state every 5 s while a case runs.
std::mutex g_live_mutex;
std::vector<const px::Exchanger*> g_live;
void live(const px::Exchanger* x, bool add) {
    std::lock_guard<std::mutex> g(g_live_mutex);
    if (add) g_live.push_back(x);
    else g_live.erase(std::remove(g_live.begin(), g_live.end(), x), g_live.end());
}

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

__host__ __device__ inline std::uint32_t word(std::uint32_t rank, std::uint32_t msg, std::uint64_t i) {
    std::uint64_t x = (std::uint64_t{rank} << 56U) ^ (std::uint64_t{msg} << 32U) ^ i;
    x ^= x >> 33U;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33U;
    return static_cast<std::uint32_t>(x);
}

__global__ void fill_kernel(std::uint32_t* dst, std::uint32_t rank, std::uint32_t msg, std::size_t words) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        dst[i] = word(rank, msg, i);
}

__global__ void consume_kernel(std::byte* peer_out, std::byte* local_out, const std::byte* peer, const std::byte* local, std::size_t bytes) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        peer_out[i] = peer[i];
        local_out[i] = local[i];
    }
}

// Flips one byte of the `nth` exchange() call's outgoing data (0 = the first call).
class Corrupting final : public sd::Transport {
public:
    Corrupting(sd::Transport& t, int nth) : t_(t), nth_(nth) {}
    void send(std::span<const std::byte> m) override { t_.send(m); }
    void receive(std::span<std::byte> m) override { t_.receive(m); }
    void exchange(std::span<const std::byte> o, std::span<std::byte> i) override {
        if (calls_++ == nth_) {
            std::vector<std::byte> copy(o.begin(), o.end());
            copy[copy.size() / 2] ^= std::byte{0x10};
            t_.exchange(copy, i);
            return;
        }
        t_.exchange(o, i);
    }
    [[nodiscard]] std::string name() const override { return "corrupting(" + t_.name() + ")"; }

private:
    sd::Transport& t_;
    int nth_, calls_ = 0;
};

// wall_clock64(), not clock64(): clock64() does not advance reliably on gfx11 (HIP docs), which made this loop endless and
// stalled the uncached direct wrap/delays case (rank 1's producer never published). Ticks of the constant wall clock.
__global__ void delay_kernel(unsigned ticks) {
    const auto start = wall_clock64();
    while (wall_clock64() - start < ticks) {}
}

struct Case {
    std::string name;
    std::size_t chunk = std::size_t{1} << 20;
    unsigned depth = 4, lag = 2;
    std::vector<std::size_t> sizes;  // message sizes (bytes), the same on both ranks
    unsigned rank1_delay_us = 0;     // rank 1 sleeps up to this long before each enqueue (asymmetry)
    bool uncached = false, direct = false;
    unsigned producer_delay_cycles = 0, consumer_delay_cycles = 0;
};

// One rank's run: returns the number of byte mismatches (or throws).
std::size_t run_rank(std::uint32_t rank, sd::Transport& t, const Case& c, std::size_t capacity, Barrier& bar) {
    hipStream_t s;
    ok(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
    px::Options o;
    o.rank = rank;
    o.capacity = capacity;
    o.depth = c.depth;
    o.chunk = c.chunk;
    o.stream = s;
    o.tx_uncached = c.uncached;
    o.timeout = std::chrono::milliseconds(10000);
    std::size_t bad = 0;
    std::byte* src = nullptr;
    ok(hipMalloc(&src, std::max<std::size_t>(capacity, 8)), "src");
    const std::size_t n = c.sizes.size();
    std::vector<std::byte*> outs(2 * n);
    for (auto& p : outs) ok(hipMalloc(&p, capacity), "out");
    {
        px::Exchanger x(t, o);
        live(&x, true);
        struct Unlive {
            const px::Exchanger* x;
            ~Unlive() { live(x, false); }
        } unlive{&x};
        std::mt19937 rng(1234 + rank);
        std::size_t received = 0;
        bar.arrive_and_wait();  // both ranks allocated and constructed
        const auto receive_one = [&] {
            const auto v = x.recv();
            if (c.consumer_delay_cycles != 0)
                hipLaunchKernelGGL(delay_kernel, dim3(1), dim3(1), 0, s, c.consumer_delay_cycles);
            const std::size_t b = c.sizes[received];
            hipLaunchKernelGGL(consume_kernel, dim3(256), dim3(256), 0, s, outs[2 * received], outs[2 * received + 1],
                               static_cast<const std::byte*>(v.peer), static_cast<const std::byte*>(v.local), b);
            x.retire();
            ++received;
        };
        for (std::size_t i = 0; i < n; ++i) {
            if (rank == 1 && c.rank1_delay_us) std::this_thread::sleep_for(std::chrono::microseconds(rng() % c.rank1_delay_us));
            const std::size_t b = c.sizes[i];
            if (c.producer_delay_cycles != 0)
                hipLaunchKernelGGL(delay_kernel, dim3(1), dim3(1), 0, s, c.producer_delay_cycles);
            auto* dst = c.direct ? static_cast<std::byte*>(x.tx()) : src;
            hipLaunchKernelGGL(fill_kernel, dim3(256), dim3(256), 0, s, reinterpret_cast<std::uint32_t*>(dst), rank, static_cast<std::uint32_t>(i),
                               (b + 3) / 4);
            if (c.direct) x.send(b, px::Tag{1, static_cast<std::uint32_t>(i % 60), i});
            else x.send_copy(src, b, px::Tag{1, static_cast<std::uint32_t>(i % 60), i});
            ok(hipGetLastError(), "producer/send launch");
            if (i == 0) {
                bool rejected = false;
                try { x.set_tx_uncached(!c.uncached); }
                catch (const std::logic_error&) { rejected = true; }
                if (!rejected) throw std::runtime_error("TX memory change accepted an unretired exchange");
            }
            if (i < 3) std::printf("    rank %u sent %zu (%zu B)\n", rank, i, b);
            if (i + 1 == c.depth && c.lag == c.depth) {
                bool rejected = false;
                try { (void)x.tx(); }
                catch (const std::logic_error&) { rejected = true; }
                if (!rejected) throw std::runtime_error("TX slot reused before retire");
            }
            if (i + 1 >= c.lag) receive_one();
        }
        while (received < n) receive_one();
        ok(hipStreamSynchronize(s), "sync");
        x.drain();
        bar.arrive_and_wait();  // both streams idle before any destructor / free
        // Both streams are idle: exercise replacement in both directions after ring wrap.
        x.set_tx_uncached(!c.uncached);
        bar.arrive_and_wait();
        auto* switched = static_cast<std::uint32_t*>(x.tx());
        hipLaunchKernelGGL(fill_kernel, dim3(1), dim3(32), 0, s, switched, rank, 123U, std::size_t{1});
        x.send(4, px::Tag{2, 0, 123});
        const auto view = x.recv();
        hipLaunchKernelGGL(consume_kernel, dim3(1), dim3(32), 0, s, src, src + 4,
                           static_cast<const std::byte*>(view.peer), static_cast<const std::byte*>(view.local), std::size_t{4});
        x.retire();
        ok(hipStreamSynchronize(s), "switched TX sync");
        x.drain();
        bar.arrive_and_wait();
        std::uint32_t switched_words[2]{};
        ok(hipMemcpy(switched_words, src, sizeof(switched_words), hipMemcpyDeviceToHost), "switched TX d2h");
        bad += switched_words[0] != word(1 - rank, 123, 0);
        bad += switched_words[1] != word(rank, 123, 0);
        x.set_tx_uncached(c.uncached);
        bar.arrive_and_wait();
        const auto st = x.stats();
        std::printf("    rank %u: %llu exchanges, %.1f MB, wire %.3f s, tail %.3f s\n", rank, static_cast<unsigned long long>(st.exchanges),
                    static_cast<double>(st.bytes) / 1e6, st.wire, st.tail);
    }
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t b = c.sizes[i];
        std::vector<std::uint32_t> peer((b + 3) / 4), local((b + 3) / 4);
        ok(hipMemcpy(peer.data(), outs[2 * i], b, hipMemcpyDeviceToHost), "d2h");
        ok(hipMemcpy(local.data(), outs[2 * i + 1], b, hipMemcpyDeviceToHost), "d2h");
        for (std::size_t w = 0; w < b / 4; ++w) {
            bad += peer[w] != word(1 - rank, static_cast<std::uint32_t>(i), w);
            bad += local[w] != word(rank, static_cast<std::uint32_t>(i), w);
        }
        if (b % 4) {  // tail bytes
            const std::uint32_t ep = word(1 - rank, static_cast<std::uint32_t>(i), b / 4), el = word(rank, static_cast<std::uint32_t>(i), b / 4);
            bad += std::memcmp(&peer[b / 4], &ep, b % 4) != 0;
            bad += std::memcmp(&local[b / 4], &el, b % 4) != 0;
        }
    }
    for (auto p : outs) (void)hipFree(p);
    (void)hipFree(src);
    (void)hipStreamDestroy(s);
    return bad;
}

bool run_case(const Case& c) {
    auto [t0, t1] = sd::memory_pair();
    std::size_t cap = 0;
    for (auto b : c.sizes) cap = std::max(cap, b);
    std::size_t bad[2] = {0, 0};
    std::string err[2];
    Barrier bar;
    std::thread r1([&] {
        try {
            bad[1] = run_rank(1, *t1, c, cap, bar);
        } catch (const std::exception& e) {
            err[1] = e.what();
        }
    });
    try {
        bad[0] = run_rank(0, *t0, c, cap, bar);
    } catch (const std::exception& e) {
        err[0] = e.what();
    }
    r1.join();
    const bool pass = bad[0] == 0 && bad[1] == 0 && err[0].empty() && err[1].empty();
    std::printf("%s %s (messages %zu, chunk %zu, depth %u, lag %u): mismatches %zu / %zu%s%s\n", pass ? "PASS" : "FAIL", c.name.c_str(),
                c.sizes.size(), c.chunk, c.depth, c.lag, bad[0], bad[1], err[0].empty() ? "" : (" rank0: " + err[0]).c_str(),
                err[1].empty() ? "" : (" rank1: " + err[1]).c_str());
    return pass;
}

// A failure case: rank `who` corrupts its `nth` exchange call (or uses another layer tag when nth < 0); every rank that
// must detect it has to throw with `expect` in the message, and its stream must be released (no hang).
bool run_failure(const std::string& name, int nth, const std::string& expect0, const std::string& expect1, bool uncached = false) {
    auto [t0, t1] = sd::memory_pair();
    const std::size_t b = 3 * (std::size_t{1} << 20) + 4096;
    std::string err[2];
    Barrier bar;
    auto rank_fn = [&](std::uint32_t rank, sd::Transport& t) {
        hipStream_t s;
        ok(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
        std::byte *src = nullptr, *out = nullptr;
        ok(hipMalloc(&src, b), "src");
        ok(hipMalloc(&out, 2 * b), "out");
        try {
            Corrupting ct(t, rank == 0 && nth >= 0 ? nth : -1);
            px::Options o;
            o.rank = rank;
            o.capacity = b;
            o.chunk = std::size_t{1} << 20;  // call indices below (header 0, chunks 1..4, trailer 5) assume 1 MiB chunks
            o.stream = s;
            o.tx_uncached = uncached;
            o.timeout = std::chrono::milliseconds(10000);
            px::Exchanger x(ct, o);
            bar.arrive_and_wait();
            auto* dst = uncached ? static_cast<std::byte*>(x.tx()) : src;
            hipLaunchKernelGGL(fill_kernel, dim3(64), dim3(256), 0, s, reinterpret_cast<std::uint32_t*>(dst), rank, 0U, b / 4);
            if (uncached) x.send(b, px::Tag{1, nth < 0 && rank == 1 ? 7U : 3U, 0});
            else x.send_copy(src, b, px::Tag{1, nth < 0 && rank == 1 ? 7U : 3U, 0});
            const auto v = x.recv();
            hipLaunchKernelGGL(consume_kernel, dim3(64), dim3(256), 0, s, out, out + b, static_cast<const std::byte*>(v.peer),
                               static_cast<const std::byte*>(v.local), b);
            x.retire();
            ok(hipStreamSynchronize(s), "sync (must not hang)");
            x.drain();
        } catch (const std::exception& e) {
            err[rank] = e.what();
        }
        (void)hipStreamSynchronize(s);
        bar.arrive_and_wait();
        (void)hipFree(out);
        (void)hipFree(src);
        (void)hipStreamDestroy(s);
    };
    std::thread r1([&] { rank_fn(1, *t1); });
    rank_fn(0, *t0);
    r1.join();
    const auto match = [](const std::string& e, const std::string& want) { return want.empty() ? e.empty() : e.find(want) != std::string::npos; };
    const bool pass = match(err[0], expect0) && match(err[1], expect1);
    std::printf("%s failure %s: rank0 [%s] rank1 [%s]\n", pass ? "PASS" : "FAIL", name.c_str(), err[0].c_str(), err[1].c_str());
    return pass;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::atomic<bool> quit{false};
    std::thread watchdog([&] {
        for (int t = 0; !quit.load(); ++t) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (t % 50 != 49) continue;
            std::lock_guard<std::mutex> g(g_live_mutex);
            for (auto* x : g_live) std::printf("  [watchdog] %s\n", x->debug_state().c_str());
        }
    });
    struct Join {
        std::atomic<bool>& q;
        std::thread& w;
        ~Join() {
            q = true;
            w.join();
        }
    } join{quit, watchdog};
    try {
        bool all = true;
        const std::size_t MiB = std::size_t{1} << 20, row = 6144 * 4;
        std::printf("crc32c(\"123456789\") = %08x (expect e3069283)\n",
                    px::crc32c(reinterpret_cast<const std::byte*>("123456789"), 9));
        all &= px::crc32c(reinterpret_cast<const std::byte*>("123456789"), 9) == 0xe3069283U;
        {  // the vector path (>= 1 KiB) equals the communicator's
            std::vector<std::byte> big(5 * MiB + 77);
            std::mt19937 rng(7);
            for (auto& b : big) b = static_cast<std::byte>(rng());
            const bool same = px::crc32c(big.data(), big.size()) == sd::crc32c(big);
            std::printf("%s crc32c vector == communicator on 5 MiB + 77\n", same ? "PASS" : "FAIL");
            all &= same;
        }
        std::printf("case a start\n");
        Case a{"prefill-like 1K rows, lag 2", MiB, 4, 2, {}, 0};
        for (int i = 0; i < 24; ++i) a.sizes.push_back(1024 * row);  // 25.2 MB, 24 chunks + tail
        all &= run_case(a);
        Case b{"mixed sizes, small chunk, depth 2, lag 1", 64 * 1024, 2, 1, {}, 0};
        for (std::size_t sz : {std::size_t{64}, std::size_t{4}, row, 7 * row, std::size_t{65536}, std::size_t{65537}, 3 * MiB + 5, 100 * row, std::size_t{1}})
            b.sizes.push_back(sz);
        for (int i = 0; i < 12; ++i) b.sizes.push_back((static_cast<std::size_t>(i) * 977 + 1) * 64 + static_cast<std::size_t>(i));
        all &= run_case(b);
        Case c{"wraparound depth 3 lag 3, asymmetric rank 1", MiB, 3, 3, {}, 3000};
        for (int i = 0; i < 40; ++i) c.sizes.push_back((static_cast<std::size_t>(i % 7) + 1) * 300 * row + static_cast<std::size_t>(i) * 64);
        all &= run_case(c);
        Case d{"8K-like 4096 rows x 6, chunk 4 MiB", 4 * MiB, 4, 2, {}, 0};
        for (int i = 0; i < 6; ++i) d.sizes.push_back(4096 * row);
        all &= run_case(d);
        for (unsigned depth : {2U, 3U, 4U}) {
            Case uc{"uncached direct wrap/delays", MiB, depth, depth, b.sizes, 3000};
            uc.uncached = true;
            uc.direct = true;
            uc.producer_delay_cycles = 1000000;
            uc.consumer_delay_cycles = 2000000;
            all &= run_case(uc);
        }
        for (bool direct : {false, true}) {
            Case uc25 = a;
            uc25.name = direct ? "uncached direct 25 MB" : "uncached copy 25 MB";
            uc25.uncached = true;
            uc25.direct = direct;
            all &= run_case(uc25);
            Case uc100 = d;
            uc100.name = direct ? "uncached direct 100 MB wrap" : "uncached copy 100 MB wrap";
            uc100.uncached = true;
            uc100.direct = direct;
            uc100.sizes.resize(12, 4096 * row);
            all &= run_case(uc100);
        }
        all &= run_failure("uncached corrupt payload", 1, "", "payload CRC mismatch", true);
        all &= run_failure("uncached corrupt trailer", 5, "", "peer trailer differs", true);
        all &= run_failure("uncached header tag", -1, "peer header differs", "peer header differs", true);
        // header (call 0), chunk 0 (call 1): rank 0 corrupts its outgoing chunk 0 -> rank 1 sees a CRC mismatch
        all &= run_failure("corrupt payload chunk 0", 1, "", "payload CRC mismatch");
        all &= run_failure("corrupt payload chunk 3 (last)", 4, "", "payload CRC mismatch");
        all &= run_failure("header mismatch (layer tag)", -1, "peer header differs", "peer header differs");
        std::printf("%s\n", all ? "ALL PASS" : "SOME FAILED");
        return all ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
