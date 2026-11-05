#pragma once
// MiniMax M3.1 TP2 PREFILL exchange module (qualification notes).
//
// The prefill counterpart of the decode Exchanger (minimax/xfer), for messages of up to max_rows FP32
// partial rows (100.7 MB at 4096 rows), with the same no-CPU-in-the-GPU-chain protocol:
//
//   a copy kernel (or the producer itself) writes the message into a host-coherent or uncached TX slot and publishes the
//   TX flag (system-scope release); the progress thread, which already holds the job, sees the flag and moves the
//   message: a 64-byte header round trip (identity and size, so the two ranks agree before any payload), then the
//   payload in `chunk`-byte pieces, each one Transport::exchange (striped over all USB4 streams), while two CRC lanes
//   compute the CRC-32C of every TX chunk and of every RX chunk as it lands; finally a trailer with the per-chunk CRCs
//   is exchanged and compared. Only then the RX flag is written; the stream, which waits for it (hipStreamWaitValue64),
//   then runs the consumer kernels, which read both partials in place (TX slot = local, RX slot = peer).
//
// The CRCs and the identity checks are off the wire's critical path except for the last chunk's CRC. Bytes are moved,
// never changed. Rules as minimax/xfer (both ranks make the same calls in the same order with the same sizes and tags;
// one in-order stream; tx()/send() of sequence s only after retire() of s - depth; a failure releases the stream with
// the RX flag at ~0 and check() / drain() throw with the cause, so results are trusted only after drain()).
// Environment (both ranks the same): STRIX_MM_PFX_DEPTH (default 4), STRIX_MM_PFX_CHUNK (bytes, default 4 MiB,
// multiple of 64 in [64 KiB, 64 MiB]), STRIX_MM_PFX_CRC=0 (off), STRIX_MM_PFX_PROGRESS_CPU / _CRC_CPUS=a,b,
// STRIX_MM_PFX_TIMEOUT_MS (default 120000: producer flag, CRC lanes, drain; a transport call itself is bounded by the
// transport's own timeout, STRIX_DIST_USB4_TIMEOUT_MS).

#include "minimax/pfx/pf_crc.hpp"
#include "strix/dist/communicator.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strix::minimax::pfx {

inline constexpr std::uint64_t kHeaderMagic = 0x3148584650584d4dULL;   // "MMXPFXH1"
inline constexpr std::uint64_t kTrailerMagic = 0x3154584650584d4dULL;  // "MMXPFXT1"

// Identity of one exchange; the header and the trailer both carry it and both ranks must agree on every field
// except `rank`.
struct Header {
    std::uint64_t magic = kHeaderMagic;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
    std::uint64_t sequence = 0;
    std::uint64_t request = 0;   // the engine's call counter (prefill chunk group)
    std::uint64_t bytes = 0;
    std::uint32_t kind = 0;      // 1 = attention sum, 2 = FFN sum
    std::uint32_t layer = 0;
    std::uint32_t chunk = 0;     // wire chunk size | bit 31 = CRC trailer on (both ranks must agree)
    std::uint32_t rank = 0;
};
static_assert(sizeof(Header) == 64);

struct Tag {
    std::uint32_t kind = 0;
    std::uint32_t layer = 0;
    std::uint64_t request = 0;
};

struct Options {
    std::uint32_t rank = 0;
    std::uint64_t epoch = 1;
    std::uint64_t plan = 0x5846504dULL;  // "MPFX"
    std::size_t capacity = 0;            // largest payload in bytes
    unsigned depth = 4;                  // TX / RX slots
    std::size_t chunk = std::size_t{1} << 22;  // 4 MiB: TTFT -1.5 % at 4K, -1.9 % at 1K vs 1 MiB, bytes identical
    hipStream_t stream = nullptr;
    bool crc = true;
    bool tx_uncached = false;
    int progress_cpu = -1, crc_tx_cpu = -1, crc_rx_cpu = -1;
    unsigned idle_spin_us = 3000;
    std::chrono::milliseconds timeout{120000};
    void from_env();  // STRIX_MM_PFX_* over these values
};

struct Stats {
    std::uint64_t exchanges = 0, bytes = 0;
    double producer_wait = 0.0;  // job taken -> TX flag seen (GPU still producing)
    double wire = 0.0;           // header + chunks + trailer exchanges
    double tail = 0.0;           // last chunk landed -> all CRCs compared, RX flag written
};


class Exchanger {
public:
    Exchanger(dist::Transport& borrowed, Options options);
    ~Exchanger();
    Exchanger(const Exchanger&) = delete;
    Exchanger& operator=(const Exchanger&) = delete;

    // Device pointer of the TX slot of the next send() (producers that write the message themselves).
    [[nodiscard]] void* tx();
    // Between complete, retired batches only; waits for the stream before replacing TX storage. Returns false (one-line
    // notice, TX stays host-coherent) when uncached device memory cannot be allocated.
    bool set_tx_uncached(bool on);
    // Copies `bytes` of device memory `local` into the next TX slot (one kernel; its last block publishes) and hands
    // the transfer to the progress thread. Never blocks the CPU.
    void send_copy(const void* local, std::size_t bytes, const Tag& tag);
    // The message of `bytes` was written into tx() by work enqueued so far: publish it (1-block kernel) and send.
    void send(std::size_t bytes, const Tag& tag);
    // The stream waits for the oldest exchange not yet received; returns {local TX slot, peer RX slot} (device).
    struct View {
        const void* local = nullptr;
        const void* peer = nullptr;
    };
    [[nodiscard]] View recv();
    // The last consumer of the oldest received exchange is enqueued: its TX / RX slots may be reused by sequence
    // s + depth (the producer of s + depth runs after that consumer on the stream; its TX flag then also proves that
    // the RX slot is no longer read). tx() / send() of s + depth require it.
    void retire();
    // CPU: waits until every posted exchange is complete, then check(). Call before the transport is used by anyone
    // else (the communicator, mmxfer) and at the end of every prefill.
    void drain();
    void check() const;
    [[nodiscard]] Stats stats() const;
    [[nodiscard]] std::uint64_t sent() const noexcept { return next_; }
    [[nodiscard]] std::uint64_t received() const noexcept { return received_; }
    [[nodiscard]] const Options& options() const noexcept { return options_; }
    [[nodiscard]] std::string name() const;
    // Counters and flags for hang reports (posted / done / sent / received / retired, TX / RX flags, lanes).
    [[nodiscard]] std::string debug_state() const;

private:
    struct Job {
        std::uint64_t sequence = 0;
        std::size_t bytes = 0;
        Tag tag;
    };
    // One CRC lane: computes the CRCs of chunks [0, count) of a buffer as `ready` advances; `done` = chunks finished.
    struct Lane {
        std::thread thread;
        std::atomic<std::uint64_t> generation{0};  // a new job
        std::atomic<std::size_t> ready{0}, done{0};
        const std::byte* data = nullptr;
        std::size_t bytes = 0, chunk = 0, count = 0;
        std::vector<std::uint32_t> crcs;
        std::atomic<bool> stop{false};
        std::mutex mutex;  // sleep / wake of an idle lane (no std::atomic::wait: libstdc++ out-of-line ABI)
        std::condition_variable wake;
    };
    static constexpr std::size_t kQueue = 512;
    void run();
    void lane_run(Lane& lane, int cpu);
    void process(const Job& job);
    void post_lane(Lane& lane, const std::byte* data, std::size_t bytes, std::size_t ready);
    void wait_lane(Lane& lane, std::chrono::steady_clock::time_point deadline);
    [[noreturn]] void fail(const std::string& why);
    std::byte* tx_slot(std::uint64_t sequence) const;
    std::byte* rx_slot(std::uint64_t sequence) const;

    dist::Transport* transport_ = nullptr;
    Options options_;
    std::size_t stride_ = 0, max_chunks_ = 0;
    std::byte* tx_host_ = nullptr;
    std::byte* rx_host_ = nullptr;
    std::byte* tx_dev_ = nullptr;
    std::byte* rx_dev_ = nullptr;
    std::uint64_t* flags_host_ = nullptr;  // [0] TX flag (GPU writes), [8] RX flag (CPU writes)
    std::uint64_t* flags_dev_ = nullptr;
    unsigned* counter_dev_ = nullptr;
    std::uint64_t next_ = 0, received_ = 0, retired_ = 0;
    Job jobs_[kQueue];
    std::atomic<std::uint64_t> posted_{0}, done_{0};
    std::atomic<bool> stop_{false}, failed_{false}, sleeping_{false};
    std::mutex mutex_;
    std::condition_variable wake_;
    std::string failure_;
    mutable std::mutex failure_mutex_;
    Stats stats_;
    mutable std::mutex stats_mutex_;
    Lane tx_lane_, rx_lane_;
    std::thread thread_;
};

}  // namespace strix::minimax::pfx
