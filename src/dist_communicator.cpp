#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/dist/host_copy.hpp"
#include "dist_crc32c_parallel.hpp"
#include "dist_crc32c_vector.hpp"
#include "dist_latency_trace.hpp"

#include "strix/serve/json.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>

namespace strix::dist {
namespace {

void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string("dist communicator: ") + what + ": " + hipGetErrorString(status));
}
[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("dist transport: " + what); }

const std::array<std::uint32_t, 256>& crc_table() {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256U; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1U ? 0x82f63b78U ^ (c >> 1U) : c >> 1U;
            t[i] = c;
        }
        return t;
    }();
    return table;
}
#if defined(__x86_64__)
__attribute__((target("sse4.2"))) std::uint32_t crc32c_sse42(std::span<const std::byte> bytes) noexcept {
    std::uint64_t c = 0xffffffffU;
    std::size_t i = 0;
    for (; i + 8U <= bytes.size(); i += 8U) {
        std::uint64_t word;
        std::memcpy(&word, bytes.data() + i, 8U);
        c = __builtin_ia32_crc32di(c, word);
    }
    auto c32 = static_cast<std::uint32_t>(c);
    for (; i < bytes.size(); ++i) c32 = __builtin_ia32_crc32qi(c32, static_cast<unsigned char>(bytes[i]));
    return c32 ^ 0xffffffffU;
}
#endif
inline void cpu_relax() noexcept {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

// In-process pair: each direction is a queue of whole messages.
struct Channel {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::vector<std::byte>> messages;
};
class MemoryTransport final : public Transport {
public:
    MemoryTransport(std::shared_ptr<Channel> out, std::shared_ptr<Channel> in) : out_(std::move(out)), in_(std::move(in)) {}
    void send(std::span<const std::byte> message) override {
        {
            std::lock_guard lock(out_->mutex);
            out_->messages.emplace_back(message.begin(), message.end());
        }
        out_->ready.notify_one();
    }
    void receive(std::span<std::byte> message) override {
        std::unique_lock lock(in_->mutex);
        if (!in_->ready.wait_for(lock, std::chrono::seconds(30), [&] { return !in_->messages.empty(); }))
            fail("in-process receive timed out");
        auto front = std::move(in_->messages.front());
        in_->messages.pop_front();
        if (front.size() != message.size()) fail("in-process message size differs from the expected size");
        std::memcpy(message.data(), front.data(), front.size());
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        send(out);
        receive(in);
    }
    [[nodiscard]] std::string name() const override { return "memory"; }

private:
    std::shared_ptr<Channel> out_, in_;
};

// Runs one job at a time on a persistent thread (the progress thread of a transport or stream). After a
// job it spins briefly for the next one (the exchanges of consecutive layers come close together), then
// sleeps; the caller spins for completion.
// How long an idle worker spins before it sleeps (STRIX_DIST_SPIN_US, default 300): a job posted within the window
// starts without a wake-up.
std::chrono::microseconds worker_spin() {
    static const auto us = [] {
        const char* v = std::getenv("STRIX_DIST_SPIN_US");
        return std::chrono::microseconds(v && *v ? std::stoll(v) : 300LL);
    }();
    return us;
}

class Worker {
public:
    Worker() : thread_([this] { run(); }) {}
    // Runs this worker on one CPU only (the communicator's progress thread: STRIX_DIST_PROGRESS_CPU).
    void pin(int cpu) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) fail("dist: CPU outside cpu_set_t range");
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (const int e = pthread_setaffinity_np(thread_.native_handle(), sizeof(set), &set); e != 0)
            fail("dist: cannot pin worker to CPU " + std::to_string(cpu) + ": " + std::strerror(e));
    }
    void name(const std::string& value) {
        if (const int e = pthread_setname_np(thread_.native_handle(), value.c_str()); e != 0)
            fail("dist: cannot name worker: " + std::string(std::strerror(e)));
    }
    void idle_spin(std::chrono::microseconds duration) {
        idle_spin_us_.store(duration.count(), std::memory_order_relaxed);
    }
    ~Worker() {
        {
            std::lock_guard lock(mutex_);
            stop_.store(true, std::memory_order_release);
        }
        wake_.notify_all();
        thread_.join();
    }
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    void post(std::function<void()> job) {
        if (latency::enabled()) {
            const auto context = latency::context();
            job_ = [context, job = std::move(job)] {
                latency::context() = context;
                latency::mark("worker_run");
                job();
                latency::mark("worker_done");
            };
            latency::mark("worker_post");
        } else {
            job_ = std::move(job);
        }
        error_ = nullptr;
        {
            std::lock_guard lock(mutex_);
            posted_.fetch_add(1U, std::memory_order_release);
        }
        wake_.notify_one();
    }
    void wait() {
        const auto target = posted_.load(std::memory_order_relaxed);
        while (finished_.load(std::memory_order_acquire) != target) cpu_relax();
        if (error_) std::rethrow_exception(error_);
    }
    // The same with a deadline: false if the job has not finished by then (it keeps running).
    bool wait_until(std::chrono::steady_clock::time_point deadline) {
        const auto target = posted_.load(std::memory_order_relaxed);
        for (unsigned i = 0; finished_.load(std::memory_order_acquire) != target; ++i) {
            if ((i & 1023U) == 0U && std::chrono::steady_clock::now() >= deadline) return false;
            cpu_relax();
        }
        if (error_) std::rethrow_exception(error_);
        return true;
    }
    // The last posted job has finished (wait() then returns at once, or rethrows its failure).
    [[nodiscard]] bool idle() const {
        return finished_.load(std::memory_order_acquire) == posted_.load(std::memory_order_relaxed);
    }

private:
    void run() {
        std::uint64_t seen = 0;
        while (true) {
            const auto spin_end = std::chrono::steady_clock::now() +
                std::chrono::microseconds(idle_spin_us_.load(std::memory_order_relaxed));
            while (posted_.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < spin_end)
                cpu_relax();
            if (posted_.load(std::memory_order_acquire) == seen) {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) || posted_.load(std::memory_order_acquire) != seen; });
            }
            if (stop_.load(std::memory_order_acquire)) return;
            ++seen;
            try { job_(); } catch (...) { error_ = std::current_exception(); }
            finished_.store(seen, std::memory_order_release);
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_;
    std::function<void()> job_;
    std::exception_ptr error_;
    std::atomic<std::uint64_t> posted_{0}, finished_{0};
    std::atomic<bool> stop_{false};
    std::atomic<std::int64_t> idle_spin_us_{worker_spin().count()};
    std::thread thread_;
};

void write_all(int fd, std::span<const std::byte> bytes, const char* what) {
    std::size_t done = 0;
    while (done < bytes.size()) {
        const auto n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) { cpu_relax(); continue; }
        if (n <= 0) fail(std::string(what) + " write failed: " + std::strerror(errno));
        done += static_cast<std::size_t>(n);
    }
}
void read_all(int fd, std::span<std::byte> bytes, const char* what) {
    std::size_t done = 0;
    while (done < bytes.size()) {
        const auto n = ::read(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) { cpu_relax(); continue; }
        if (n == 0) fail(std::string(what) + " closed");
        if (n < 0) fail(std::string(what) + " read failed: " + std::strerror(errno));
        done += static_cast<std::size_t>(n);
    }
}

std::string errno_text(int error) {
    switch (error) {
    case ENODEV: return "ENODEV: the stream device is gone (link unplugged or controller reset)";
    case ETIMEDOUT: return "ETIMEDOUT: the driver timed out (peer or domain gone)";
    case ENXIO: return "ENXIO: no stream attached (peer service gone)";
    case ENOMEM: return "ENOMEM: the driver could not allocate the stream rings (memory pressure)";
    case EPIPE: return "EPIPE: the peer closed the stream (its rank failed or exited)";
    case ECONNRESET: return "ECONNRESET: the peer closed the stream (its rank failed or exited)";
    default: return std::strerror(error);
    }
}

// Host path of USB4 streams. Each stream has a persistent writer and a persistent reader, so a stream's
// part of a message is written and read at the same time and messages of any size go through (a writer
// that ran ahead of its reader blocked once a part exceeded what the two rings buffer). Every transfer
// waits in poll() slices with a deadline and an abort flag: when one stream fails (ENODEV, ETIMEDOUT,
// the peer's CLOSE, the deadline), the other transfers stop, every stream is closed (the peer sees CLOSE
// on its open streams and fails at once instead of waiting) and the transport refuses further use.
// Messages are split in proportion to integer weights at 64-byte boundaries; messages shorter than
// `min_part` bytes per stream use fewer streams (the first k of the order, which alternates cables).
class Usb4HostTransport final : public Transport {
public:
    Usb4HostTransport(std::vector<int> fds, std::vector<std::string> names, std::vector<std::uint32_t> weights,
                      std::size_t min_part = 0, std::chrono::milliseconds timeout = std::chrono::milliseconds(120000))
        : names_(std::move(names)), fds_(std::move(fds)), weights_(std::move(weights)), min_part_(min_part), timeout_(timeout) {
        if (fds_.empty() || fds_.size() > 8U || names_.size() != fds_.size() || weights_.size() != fds_.size()) {
            for (const int fd : fds_) ::close(fd);
            fail("usb4 host transport takes one to eight streams, each with a name and a weight");
        }
        for (const auto w : weights_)
            if (w == 0U) {
                for (const int fd : fds_) ::close(fd);
                fail("usb4 host transport weights must be positive");
            }
        if (const char* t = std::getenv("STRIX_DIST_USB4_TIMEOUT_MS"); t && *t) timeout_ = std::chrono::milliseconds(std::stoll(t));
        // Every read and write is non-blocking (poll() bounds the waits: deadline, abort), whoever opened the streams.
        for (const int fd : fds_)
            if (const int flags = ::fcntl(fd, F_GETFL); flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
                const int error = errno;
                for (const int open_fd : fds_) ::close(open_fd);
                fail(std::string("usb4 host transport: cannot make a stream non-blocking: ") + std::strerror(error));
            }
        // STRIX_DIST_USB4_EAGER=N: messages of at most N bytes are moved on the calling thread, with no hand-off to the
        // per-stream threads. Every direction of every stream progresses in one poll loop, so a part larger than what
        // a stream buffers cannot deadlock two ranks that both write before they read.
        // Default 65536 since P17.3 (deepseek41 pair, layout 1: R1 decode +2.8 %, R6 / batch equal, same bits); =0 off.
        eager_ = std::size_t{65536};
        if (const char* e = std::getenv("STRIX_DIST_USB4_EAGER"); e && *e) eager_ = std::min<std::size_t>(std::stoull(e), std::size_t{1} << 20U);
        // Optional singleton masks per stream, in topology/receipt order. Unspecified: inherited affinity.
        // Validate the complete list before starting any jobs; a partial map is an error, not an unpinned fallback.
        const auto cpus = [this](const char* key) {
            std::vector<int> result;
            if (const char* text = std::getenv(key); text && *text) {
                std::stringstream input(text);
                for (std::string item; std::getline(input, item, ',');) {
                    std::size_t end = 0;
                    const int cpu = std::stoi(item, &end);
                    if (end != item.size() || cpu < 0 || cpu >= CPU_SETSIZE) fail(std::string(key) + ": invalid CPU");
                    result.push_back(cpu);
                }
                if (result.size() != fds_.size()) fail(std::string(key) + ": needs exactly one CPU per stream");
            }
            return result;
        };
        try {
            const auto write_cpus = cpus("STRIX_DIST_USB4_WRITE_CPUS");
            const auto read_cpus = cpus("STRIX_DIST_USB4_READ_CPUS");
            for (std::size_t i = 0; i < fds_.size(); ++i) {
                writers_.push_back(std::make_unique<Worker>());
                readers_.push_back(std::make_unique<Worker>());
                if (!write_cpus.empty()) writers_.back()->pin(write_cpus[i]);
                if (!read_cpus.empty()) readers_.back()->pin(read_cpus[i]);
                if (latency::enabled() || !write_cpus.empty()) writers_.back()->name("u4-w" + std::to_string(i));
                if (latency::enabled() || !read_cpus.empty()) readers_.back()->name("u4-r" + std::to_string(i));
            }
        } catch (...) {
            close_all();
            throw;
        }
    }
    ~Usb4HostTransport() override {
        if (traffic_) std::fprintf(stderr, "%s\n", traffic().c_str());
        aborted_.store(true, std::memory_order_release);
        writers_.clear();
        readers_.clear();
        close_all();
    }
    void send(std::span<const std::byte> message) override { run(&message, nullptr); }
    void receive(std::span<std::byte> message) override { run(nullptr, &message); }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        if (out.size() != in.size()) fail("usb4 exchange: both directions carry the same size");
        run(&out, &in);
    }
    void exchange_overlapped(std::span<const std::byte> out, std::span<std::byte> in, void (*on_wire)(void*), void* context) override {
        if (out.size() != in.size()) fail("usb4 exchange: both directions carry the same size");
        run(&out, &in, on_wire, context);
    }
    // Both ranks the same table (the pairing carries rank 0's in its plan).
    void set_stream_table(const std::array<std::pair<std::size_t, std::size_t>, 4>& table) { table_ = table; }
    [[nodiscard]] std::string name() const override {
        std::string out = "usb4-host:";
        for (std::size_t k = 0; k < names_.size(); ++k)
            out += (k ? "," : "") + names_[k] + (fds_.size() > 1U ? "*" + std::to_string(weights_[k]) : "");
        if (min_part_) out += " (min part " + std::to_string(min_part_) + " B)";
        for (std::size_t i = 0; i < table_.size() && table_[i].second; ++i)
            out += std::string(i ? "," : " (streams ") + (table_[i].first == SIZE_MAX ? std::string("inf") : std::to_string(table_[i].first)) + ":" +
                   std::to_string(table_[i].second) + (i + 1U == table_.size() || !table_[i + 1U].second ? ")" : "");
        if (eager_) out += " (eager up to " + std::to_string(eager_) + " B)";
        return out;
    }

private:
    // Streams used for a message of `size` bytes, and stream k's part of it.
    [[nodiscard]] std::size_t streams_for(std::size_t size) const {
        for (std::size_t i = 0; i < table_.size() && table_[i].second; ++i)
            if (size <= table_[i].first) return std::clamp<std::size_t>(table_[i].second, 1U, fds_.size());
        if (!min_part_) return fds_.size();
        return std::clamp<std::size_t>(size / min_part_, 1U, fds_.size());
    }
    template <class Span> Span part(Span message, std::size_t k, std::size_t used) const {
        std::uint64_t total = 0, before = 0, through = 0;
        for (std::size_t j = 0; j < used; ++j) {
            total += weights_[j];
            if (j < k) before += weights_[j];
            if (j <= k) through += weights_[j];
        }
        const std::uint64_t size = message.size();
        const auto begin = k == 0U ? 0U : (size * before / total) & ~std::uint64_t{63};
        const auto end = k + 1U == used ? size : (size * through / total) & ~std::uint64_t{63};
        return message.subspan(begin, end - begin);
    }
    void close_all() {
        std::lock_guard lock(fds_mutex_); // abort() may be switching them to /dev/null on another thread
        for (auto& fd : fds_)
            if (fd >= 0) { ::close(fd); fd = -1; }
    }
    // Waits until `fd` is ready for `events` in short slices, so an abort or the deadline stops the wait. The
    // descriptors stay non-blocking: no read or write can sleep in the kernel past the deadline or an abort.
    void wait_ready(std::size_t k, short events, std::chrono::steady_clock::time_point deadline, const char* what) {
        if (spin_) return;
        while (true) {
            if (aborted_.load(std::memory_order_acquire)) throw Aborted{};
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) fail_stream(k, std::string(what) + ": no progress within " + std::to_string(timeout_.count()) + " ms");
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            pollfd p{fds_[k], events, 0};
            const int ready = ::poll(&p, 1, static_cast<int>(std::min<long long>(left, 50)));
            if (ready < 0) {
                if (errno == EINTR) continue;
                fail_stream(k, std::string(what) + ": poll: " + errno_text(errno));
            }
            if (ready == 0) continue;
            stats_[k].revents.store(p.revents, std::memory_order_relaxed);
            if (p.revents & POLLNVAL) fail_stream(k, std::string(what) + ": descriptor closed");
            return; // ready, or an error/hang-up that the read or write reports precisely
        }
    }
    void write_part(std::size_t k, std::span<const std::byte> bytes) {
        latency::mark("stream_write_begin", k);
        std::size_t done = 0;
        auto deadline = std::chrono::steady_clock::now() + timeout_;
        auto& st = stats_[k];
        const auto begin_ns = traffic_ ? now_ns() : 0U;
        st.want_w.store(bytes.size(), std::memory_order_relaxed);
        st.done_w.store(0, std::memory_order_relaxed);
        while (done < bytes.size()) {
            wait_ready(k, POLLOUT, deadline, "write");
            latency::mark("write_syscall_begin", bytes.size() - done);
            const auto n = ::write(fds_[k], bytes.data() + done, bytes.size() - done);
            const int call_errno = errno;
            latency::mark("write_syscall_end", n > 0 ? static_cast<std::uint64_t>(n) : 0U);
            errno = call_errno;
            if (n > 0) {
                done += static_cast<std::size_t>(n);
                st.done_w.store(done, std::memory_order_relaxed);
                st.last_w.store(now_ns(), std::memory_order_relaxed);
                deadline = std::chrono::steady_clock::now() + timeout_;
                continue;
            }
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
                if (aborted_.load(std::memory_order_acquire)) throw Aborted{};
                if (spin_) cpu_relax();
                if (std::chrono::steady_clock::now() >= deadline) fail_stream(k, "write: no progress within " + std::to_string(timeout_.count()) + " ms");
                continue;
            }
            fail_stream(k, n == 0 ? std::string("write returned 0") : "write: " + errno_text(errno));
        }
        if (traffic_) {
            const auto end_ns = now_ns();
            st.bytes_w.fetch_add(bytes.size(), std::memory_order_relaxed);
            st.busy_w.fetch_add(end_ns - begin_ns, std::memory_order_relaxed);
            st.fin_w.store(end_ns, std::memory_order_relaxed);
        }
        latency::mark("stream_write_end", k);
    }
    void read_part(std::size_t k, std::span<std::byte> bytes) {
        latency::mark("stream_read_begin", k);
        std::size_t done = 0;
        auto deadline = std::chrono::steady_clock::now() + timeout_;
        auto& st = stats_[k];
        const auto begin_ns = traffic_ ? now_ns() : 0U;
        st.want_r.store(bytes.size(), std::memory_order_relaxed);
        st.done_r.store(0, std::memory_order_relaxed);
        while (done < bytes.size()) {
            wait_ready(k, POLLIN, deadline, "read");
            latency::mark("read_syscall_begin", bytes.size() - done);
            const auto n = ::read(fds_[k], bytes.data() + done, bytes.size() - done);
            const int call_errno = errno;
            latency::mark("read_syscall_end", n > 0 ? static_cast<std::uint64_t>(n) : 0U);
            errno = call_errno;
            if (n > 0) {
                done += static_cast<std::size_t>(n);
                st.done_r.store(done, std::memory_order_relaxed);
                st.last_r.store(now_ns(), std::memory_order_relaxed);
                deadline = std::chrono::steady_clock::now() + timeout_;
                continue;
            }
            if (n == 0) fail_stream(k, "the peer closed the stream (its rank failed or exited)");
            if (errno == EINTR || errno == EAGAIN) {
                if (aborted_.load(std::memory_order_acquire)) throw Aborted{};
                if (spin_) cpu_relax();
                if (std::chrono::steady_clock::now() >= deadline) fail_stream(k, "read: no progress within " + std::to_string(timeout_.count()) + " ms");
                continue;
            }
            fail_stream(k, "read: " + errno_text(errno));
        }
        if (traffic_) {
            const auto end_ns = now_ns();
            st.bytes_r.fetch_add(bytes.size(), std::memory_order_relaxed);
            st.busy_r.fetch_add(end_ns - begin_ns, std::memory_order_relaxed);
            st.fin_r.store(end_ns, std::memory_order_relaxed);
        }
        latency::mark("stream_read_end", k);
    }
    // The eager path: every unfinished direction of every used stream in one loop on the calling thread. Each pass
    // tries the reads and writes (non-blocking); a pass without progress polls them all, in slices so an abort or the
    // deadline (no progress on any of them for the timeout) ends the wait. Partial reads and writes are the norm.
    // on_wire (if any) runs once, in the first pass after which every write is handed to the driver (the reads then wait
    // for the peer), or at the end.
    void eager(const std::span<const std::byte>* out, const std::span<std::byte>* in, std::size_t used, void (*on_wire)(void*) = nullptr,
               void* context = nullptr) {
        std::array<std::span<const std::byte>, 8> wpart{};
        std::array<std::span<std::byte>, 8> rpart{};
        std::array<std::size_t, 8> wdone{}, rdone{};
        std::size_t open = 0, wopen = 0;
        for (std::size_t k = 0; k < used; ++k) {
            auto& st = stats_[k];
            if (out) {
                wpart[k] = part(*out, k, used);
                st.want_w.store(wpart[k].size(), std::memory_order_relaxed);
                st.done_w.store(0, std::memory_order_relaxed);
                open += !wpart[k].empty();
                wopen += !wpart[k].empty();
            }
            if (in) {
                rpart[k] = part(*in, k, used);
                st.want_r.store(rpart[k].size(), std::memory_order_relaxed);
                st.done_r.store(0, std::memory_order_relaxed);
                open += !rpart[k].empty();
            }
        }
        auto deadline = std::chrono::steady_clock::now() + timeout_;
        std::array<pollfd, 16> fds{};
        std::array<std::size_t, 16> stream{};
        while (open) {
            bool moved = false;
            for (std::size_t k = 0; k < used; ++k) {
                auto& st = stats_[k];
                if (wdone[k] < wpart[k].size()) {
                    const auto n = ::write(fds_[k], wpart[k].data() + wdone[k], wpart[k].size() - wdone[k]);
                    if (n > 0) {
                        wdone[k] += static_cast<std::size_t>(n);
                        st.done_w.store(wdone[k], std::memory_order_relaxed);
                        st.last_w.store(now_ns(), std::memory_order_relaxed);
                        moved = true;
                        open -= wdone[k] == wpart[k].size();
                        wopen -= wdone[k] == wpart[k].size();
                    } else if (n == 0 || (errno != EAGAIN && errno != EINTR)) {
                        fail_stream(k, n == 0 ? std::string("write returned 0") : "write: " + errno_text(errno));
                    }
                }
                if (rdone[k] < rpart[k].size()) {
                    const auto n = ::read(fds_[k], rpart[k].data() + rdone[k], rpart[k].size() - rdone[k]);
                    if (n > 0) {
                        rdone[k] += static_cast<std::size_t>(n);
                        st.done_r.store(rdone[k], std::memory_order_relaxed);
                        st.last_r.store(now_ns(), std::memory_order_relaxed);
                        moved = true;
                        open -= rdone[k] == rpart[k].size();
                    } else if (n == 0) {
                        fail_stream(k, "the peer closed the stream (its rank failed or exited)");
                    } else if (errno != EAGAIN && errno != EINTR) {
                        fail_stream(k, "read: " + errno_text(errno));
                    }
                }
            }
            if (on_wire && !wopen && open) {
                on_wire(context);
                on_wire = nullptr;
                continue; // the reads may have progressed meanwhile: try them before polling
            }
            if (!open) break;
            if (aborted_.load(std::memory_order_acquire)) throw Aborted{};
            const auto now = std::chrono::steady_clock::now();
            if (moved) { deadline = now + timeout_; continue; }
            if (now >= deadline) {
                std::size_t k = 0;
                while (wdone[k] == wpart[k].size() && rdone[k] == rpart[k].size()) ++k;
                fail_stream(k, std::string(wdone[k] < wpart[k].size() ? "write" : "read") + ": no progress within " + std::to_string(timeout_.count()) + " ms");
            }
            if (spin_) { cpu_relax(); continue; }
            std::size_t n = 0;
            for (std::size_t k = 0; k < used; ++k) {
                const short events = static_cast<short>((wdone[k] < wpart[k].size() ? POLLOUT : 0) | (rdone[k] < rpart[k].size() ? POLLIN : 0));
                if (events) { fds[n] = pollfd{fds_[k], events, 0}; stream[n++] = k; }
            }
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            const int ready = ::poll(fds.data(), static_cast<nfds_t>(n), static_cast<int>(std::clamp<long long>(left, 0, 50)));
            if (ready < 0 && errno != EINTR) fail_stream(stream[0], "poll: " + errno_text(errno));
            for (std::size_t i = 0; ready > 0 && i < n; ++i) {
                if (!fds[i].revents) continue;
                stats_[stream[i]].revents.store(fds[i].revents, std::memory_order_relaxed);
                if (fds[i].revents & POLLNVAL) fail_stream(stream[i], "descriptor closed");
            }
        }
        if (on_wire) on_wire(context);
    }
    struct Aborted {};
    [[noreturn]] void fail_stream(std::size_t k, const std::string& what) {
        const std::string text = "usb4 stream " + names_[k] + ": " + what;
        {
            std::lock_guard lock(root_mutex_);
            if (root_.empty()) root_ = text; // the first failure is the cause; the others follow from it
        }
        aborted_.store(true, std::memory_order_release);
        fail(text);
    }
    void run(const std::span<const std::byte>* out, const std::span<std::byte>* in, void (*on_wire)(void*) = nullptr, void* context = nullptr) {
        if (poisoned_.load(std::memory_order_acquire)) {
            std::lock_guard lock(root_mutex_);
            fail("usb4 host transport: unusable after an earlier stream failure (" + (poison_.empty() ? root_ : poison_) + ")");
        }
        const std::size_t size = out ? out->size() : in->size();
        const std::size_t used = streams_for(size);
        if (size <= eager_) {
            if (traffic_) {
                ++eager_messages_;
                eager_bytes_ += size;
            }
            try {
                eager(out, in, used, on_wire, context);
                return;
            } catch (...) {
                if (!aborted_.load(std::memory_order_acquire)) throw;
                close_all();
                poisoned_ = true;
                {
                    std::lock_guard lock(root_mutex_);
                    poison_ = root_.empty() ? std::string("aborted") : root_;
                }
                fail(poison_ + " (all streams closed; the peer rank sees them close)");
            }
        }
        for (std::size_t k = 0; k < used; ++k) {
            if (out) {
                const auto piece = part(*out, k, used);
                writers_[k]->post([this, k, piece] { guarded([&] { write_part(k, piece); }); });
            }
            if (in) {
                const auto piece = part(*in, k, used);
                readers_[k]->post([this, k, piece] { guarded([&] { read_part(k, piece); }); });
            }
        }
        if (on_wire) on_wire(context); // the stream threads move the bytes meanwhile
        std::exception_ptr first;
        for (std::size_t k = 0; k < used; ++k) {
            if (out) try { writers_[k]->wait(); } catch (...) { if (!first) first = std::current_exception(); }
            if (in) try { readers_[k]->wait(); } catch (...) { if (!first) first = std::current_exception(); }
        }
        if (aborted_.load(std::memory_order_acquire)) {
            // One stream failed: close every stream now (the peer reads CLOSE on the live ones and fails at
            // once), and keep the reason for any later call.
            close_all();
            poisoned_ = true;
            {
                std::lock_guard lock(root_mutex_);
                poison_ = root_.empty() ? std::string("aborted") : root_;
            }
            fail(poison_ + " (all streams closed; the peer rank sees them close)");
        }
        if (first) std::rethrow_exception(first);
        if (traffic_ && used > 1U) {  // completion skew of this message's parts (the bottleneck stream finished last)
            const auto skew = [&](bool write) {
                std::uint64_t lo = UINT64_MAX, hi = 0;
                std::size_t last = 0;
                for (std::size_t k = 0; k < used; ++k) {
                    const auto t = (write ? stats_[k].fin_w : stats_[k].fin_r).load(std::memory_order_relaxed);
                    lo = std::min(lo, t);
                    if (t >= hi) {
                        hi = t;
                        last = k;
                    }
                }
                (write ? skew_w_ns_ : skew_r_ns_) += hi - lo;
                ++(write ? stats_[last].last_w_count : stats_[last].last_r_count);
            };
            ++striped_messages_;
            striped_bytes_ += size;
            if (out) skew(true);
            if (in) skew(false);
        }
    }
    template <class F> void guarded(F&& f) {
        try { f(); } catch (const Aborted&) {
            throw std::runtime_error("dist transport: usb4 transfer stopped because another stream failed");
        }
    }
    std::vector<std::string> names_;
    std::vector<int> fds_;
    std::vector<std::uint32_t> weights_;
    std::size_t min_part_ = 0, eager_ = 0;
    std::array<std::pair<std::size_t, std::size_t>, 4> table_{}; // (message bytes up to, streams), STRIX_DIST_USB4_STREAMS
    std::chrono::milliseconds timeout_;
    const bool spin_ = [] { const char* v = std::getenv("STRIX_DIST_USB4_SPIN"); return v && v[0] == '1'; }();
    std::atomic<bool> aborted_{false};
    std::atomic<bool> poisoned_{false};
    // Per stream, for failure reports: bytes done / expected of the part in flight, last successful read and write
    // (steady ns), last poll events.
    struct StreamStats {
        std::atomic<std::uint64_t> done_w{0}, want_w{0}, done_r{0}, want_r{0}, last_w{0}, last_r{0};
        std::atomic<int> revents{0};
        // STRIX_DIST_USB4_TRAFFIC=1: striped parts only (eager messages are counted per transport)
        std::atomic<std::uint64_t> bytes_w{0}, bytes_r{0}, busy_w{0}, busy_r{0}, fin_w{0}, fin_r{0};
        std::uint64_t last_w_count = 0, last_r_count = 0;  // calling thread only
    };
    // STRIX_DIST_USB4_TRAFFIC=1: per-stream bytes, busy time and completion skew, printed when the transport closes
    const bool traffic_ = [] { const char* v = std::getenv("STRIX_DIST_USB4_TRAFFIC"); return v && v[0] == '1'; }();
    std::uint64_t eager_messages_ = 0, eager_bytes_ = 0, striped_messages_ = 0, striped_bytes_ = 0, skew_w_ns_ = 0, skew_r_ns_ = 0;
    [[nodiscard]] std::string traffic() const {
        std::ostringstream out;
        const auto d = [](std::uint64_t v) { return static_cast<double>(v); };
        const auto rate = [&](std::uint64_t b, std::uint64_t ns) { return ns ? d(b) * 8.0 / d(ns) : 0.0; };  // Gb/s while busy
        out << "usb4 traffic: eager " << eager_messages_ << " msgs " << d(eager_bytes_) / 1e6 << " MB; striped " << striped_messages_
            << " msgs " << d(striped_bytes_) / 1e6 << " MB, completion skew write " << d(skew_w_ns_) / 1e6 << " ms read " << d(skew_r_ns_) / 1e6
            << " ms";
        for (std::size_t k = 0; k < names_.size(); ++k) {
            const auto& st = stats_[k];
            const std::uint64_t bw = st.bytes_w.load(), tw = st.busy_w.load(), br = st.bytes_r.load(), tr = st.busy_r.load();
            out << "\n  " << names_[k] << " w " << d(bw) / 1e6 << " MB busy " << d(tw) / 1e6 << " ms (" << rate(bw, tw) << " Gb/s) last "
                << st.last_w_count << "x; r " << d(br) / 1e6 << " MB busy " << d(tr) / 1e6 << " ms (" << rate(br, tr) << " Gb/s) last "
                << st.last_r_count << "x";
        }
        return out.str();
    }
    std::array<StreamStats, 8> stats_{};
    static std::uint64_t now_ns() {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }

public:
    void abort(const std::string& why) override {
        {
            std::lock_guard lock(root_mutex_);
            if (root_.empty()) root_ = why;
        }
        aborted_.store(true, std::memory_order_release); // stream threads leave within one poll slice (50 ms)
        // The streams close now (the peer sees CLOSE), but the descriptor numbers stay ours: dup2 puts /dev/null in
        // their place, so a thread still inside a read, write or poll cannot reach an unrelated file that reused the
        // number (an HTTP socket in the server). close_all() releases the numbers later.
        std::lock_guard lock(fds_mutex_);
        const int null_fd = ::open("/dev/null", O_RDWR | O_CLOEXEC);
        for (const int fd : fds_)
            if (fd >= 0) {
                if (null_fd >= 0) (void)::dup2(null_fd, fd);
                else (void)::shutdown(fd, SHUT_RDWR);
            }
        if (null_fd >= 0) ::close(null_fd);
        poisoned_.store(true, std::memory_order_release);
    }
    [[nodiscard]] std::string state() const override {
        std::ostringstream out;
        const auto now = now_ns();
        for (std::size_t k = 0; k < names_.size(); ++k) {
            const auto& st = stats_[k];
            const auto ago = [&](std::uint64_t t) { return t ? std::to_string((now - t) / 1000000U) + " ms ago" : std::string("never"); };
            out << (k ? "; " : "") << names_[k] << ": wrote " << st.done_w.load() << "/" << st.want_w.load() << " (last " << ago(st.last_w.load())
                << "), read " << st.done_r.load() << "/" << st.want_r.load() << " (last " << ago(st.last_r.load()) << "), revents 0x" << std::hex
                << st.revents.load() << std::dec;
        }
        return out.str();
    }

private:
    std::mutex root_mutex_, fds_mutex_;
    std::string poison_, root_;
    std::vector<std::unique_ptr<Worker>> writers_, readers_;
};

void set_blocking(int fd, bool blocking) {
    const int flags = ::fcntl(fd, F_GETFL);
    if (flags < 0 || ::fcntl(fd, F_SETFL, blocking ? flags & ~O_NONBLOCK : flags | O_NONBLOCK) < 0)
        fail(std::string("fcntl failed: ") + std::strerror(errno));
}

// ---- Automatic pairing -------------------------------------------------------------------------------
constexpr std::uint64_t hello_magic = 0x4f4c4548584e5453ULL; // "STNXHELO"
constexpr std::uint64_t plan_magic = 0x4e414c50584e5453ULL;  // "STNXPLAN"
constexpr std::uint16_t pairing_version = 3; // 3: the plan carries the stream table and the calibration mode
constexpr std::size_t max_streams = 8;

struct Hello {
    std::uint64_t magic = hello_magic;
    std::uint64_t nonce = 0;
    std::uint64_t peer_nonce = 0;     // acknowledgement: the peer's nonce
    std::uint16_t version = pairing_version;
    std::uint16_t kind = 0;           // 1 hello, 2 acknowledgement
    std::uint32_t rank = 0;
    std::uint32_t stream = 0;         // the sender's candidate index
    std::uint32_t peer_stream = 0;    // acknowledgement: the peer's candidate index
    std::int32_t device = -1;         // the sender's tbstream number
    std::uint32_t generation = 0;     // the sender's open count of this stream
    std::uint32_t peer_generation = 0; // acknowledgement: the peer's generation
    char host[8] = {};
    std::uint32_t crc = 0;
};
static_assert(sizeof(Hello) == 64);
struct Plan {
    std::uint64_t magic = plan_magic;
    std::uint32_t version = pairing_version;
    std::uint32_t count = 0;
    std::uint32_t index[max_streams] = {};  // rank 0's candidate indices, in striping order
    std::uint32_t weight[max_streams] = {};
    std::uint32_t min_part = 0;  // bytes per stream below which a message uses fewer streams (rank 0's value)
    // STRIX_DIST_USB4_STREAMS (rank 0's): a message of at most streams_up_to[i] bytes (payload + trailer) uses the first
    // streams_count[i] streams of the striping order; the first matching entry wins, 0 entries: min_part decides.
    std::uint32_t streams_up_to[4] = {};
    std::uint8_t streams_count[4] = {};
    std::uint32_t flags = 0;     // plan_joint: the weights are recalibrated with every cable at once (rank 0's choice)
    std::uint32_t crc = 0;
    std::uint32_t pad[3] = {};
};
static_assert(sizeof(Plan) == 128);
constexpr std::uint32_t plan_joint = 1U;

template <class T> std::uint32_t body_crc(const T& value) {
    return crc32c(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&value), offsetof(T, crc)));
}
std::string read_text(const std::string& path) {
    std::ifstream in(path);
    std::string text;
    std::getline(in, text);
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ' || text.back() == '\r')) text.pop_back();
    return text;
}
bool exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}
double leading_number(const std::string& text) {
    try { return text.empty() ? 0.0 : std::stod(text); } catch (...) { return 0.0; }
}
std::int32_t trailing_number(const std::string& text) {
    std::size_t i = text.size();
    while (i > 0U && text[i - 1U] >= '0' && text[i - 1U] <= '9') --i;
    return i == text.size() ? -1 : static_cast<std::int32_t>(std::stol(text.substr(i)));
}
std::string host_name() {
    char name[256] = {};
    if (::gethostname(name, sizeof(name) - 1U) != 0) return "?";
    return name;
}

struct Candidate {
    Usb4StreamPair info;
    std::string service;
    int fd = -1;
    bool dead = false;
    // handshake state of the current open (generation)
    std::uint32_t generation = 0;
    std::chrono::steady_clock::time_point opened_at{}, last_hello{}, reopen_at{};
    std::array<std::byte, 64> in{};
    std::size_t have = 0;
    std::array<std::byte, 128> out{};
    std::size_t out_size = 0, out_done = 0;
    bool hello_ok = false;     // the peer's hello for this generation was received
    bool ack_ok = false;       // the peer acknowledged our hello of this generation
    bool reply_queued = false; // rank 0: hello + acknowledgement queued; rank 1: acknowledgement queued
    bool reply_written = false;
    Hello peer;
    std::size_t garbage = 0, reopens = 0, open_retries = 0;
    std::string last_error;
    std::vector<std::string> trace; // STRIX_DIST_USB4_DEBUG=1
};

std::vector<Candidate> layout_candidates(const Usb4AutoOptions& options, std::vector<std::string>& rejected) {
    std::string path = options.layout;
    if (path.empty()) {
        const char* env = std::getenv("STRIX_DIST_USB4_LAYOUT");
        path = env && *env ? env : "/run/usb4stream-layout.json";
    }
    std::ifstream file(path);
    if (!file) fail("cannot read the USB4 stream layout " + path);
    std::stringstream text;
    text << file.rdbuf();
    const auto layout = strix::serve::parse_json(text.str());
    if (!layout.is_object()) fail("the USB4 stream layout " + path + " is not an object");
    std::vector<std::string> exclude = options.exclude;
    if (const char* env = std::getenv("STRIX_DIST_USB4_EXCLUDE"); env && *env) {
        std::stringstream list(env);
        for (std::string item; std::getline(list, item, ',');) if (!item.empty()) exclude.push_back(item);
    }
    std::vector<Candidate> out;
    for (const auto& [key, entry] : layout.as_object()) {
        const auto slash = key.find('/');
        if (slash == std::string::npos || !entry.is_object()) continue; // "version", "topology"
        const auto* device = entry.find("device");
        if (!device || !device->is_string()) { rejected.push_back(key + ": no device in the layout"); continue; }
        Candidate c;
        c.info.key = key;
        c.info.device = device->as_string();
        c.service = key.substr(0, slash);
        if (std::find(exclude.begin(), exclude.end(), key) != exclude.end() ||
            std::find(exclude.begin(), exclude.end(), c.info.device) != exclude.end()) {
            rejected.push_back(key + " " + c.info.device + ": excluded");
            continue;
        }
        const auto reason = [&](const char* field) {
            const auto* r = entry.find(field);
            return r && r->is_string() ? " (" + r->as_string() + ")" : std::string();
        };
        // A stream in gpu_direct mode serves the GPU-direct path only (host read/write are refused there).
        if (const auto* cfg = entry.find("configfs"); cfg && cfg->is_string() && read_text(cfg->as_string() + "/gpu_direct") == "1") {
            rejected.push_back(key + " " + c.info.device + ": in gpu_direct mode (GPU-direct path)");
            continue;
        }
        // sysfs decides: the layout is written at one moment (boot, a probe) and its attached/openable
        // flags can be stale in either direction. Never open a stream whose service has no peer now
        // (its controller or link may be down).
        if (!exists(options.sysfs + "/" + c.service)) {
            const auto* attached = entry.find("attached");
            const bool layout_says = attached && attached->is_bool() && !attached->as_bool();
            rejected.push_back(key + " " + c.info.device + ": service " + c.service + " is not in " + options.sysfs +
                               " (no peer on that port)" + (layout_says ? "; the layout agrees" + reason("openable_reason") : std::string()));
            continue;
        }
        if (!exists(c.info.device)) { rejected.push_back(key + ": " + c.info.device + " does not exist"); continue; }
        const auto dot = c.service.rfind('.');
        c.info.xdomain = dot == std::string::npos ? c.service : c.service.substr(0, dot);
        const auto dash = c.info.xdomain.find('-');
        c.info.domain = dash == std::string::npos ? -1 : static_cast<std::int32_t>(leading_number(c.info.xdomain.substr(0, dash)));
        if (const auto* cable = entry.find("cable"); cable && cable->is_string()) c.info.cable = cable->as_string();
        if (const auto* local = entry.find("local"); local && local->is_object())
            if (const auto* port = local->find("usb4_port"); port && port->is_number()) c.info.usb4_port = static_cast<std::int32_t>(port->as_int());
        const auto x = options.sysfs + "/" + c.info.xdomain + "/";
        const double rx = leading_number(read_text(x + "rx_speed")) * std::max(1.0, leading_number(read_text(x + "rx_lanes")));
        const double tx = leading_number(read_text(x + "tx_speed")) * std::max(1.0, leading_number(read_text(x + "tx_lanes")));
        c.info.link_gbps = std::min(rx, tx);
        out.push_back(std::move(c));
    }
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) { return a.info.key < b.info.key; });
    if (out.size() > max_streams) fail("more than " + std::to_string(max_streams) + " USB4 streams in the layout");
    return out;
}

bool pairing_debug() {
    static const bool debug = [] { const char* v = std::getenv("STRIX_DIST_USB4_DEBUG"); return v && v[0] == '1'; }();
    return debug;
}
void note(Candidate& c, const std::string& what) {
    if (!pairing_debug()) return;
    static const auto origin = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - origin).count();
    c.trace.push_back(std::to_string(us) + "us " + what);
}
std::string describe(const Hello& h) {
    std::ostringstream out;
    out << (h.kind == 1U ? "hello" : h.kind == 2U ? "ack" : "kind?") << " rank " << h.rank << " stream " << h.stream << " gen "
        << h.generation << " nonce " << std::hex << (h.nonce & 0xffffU) << " peer_nonce " << (h.peer_nonce & 0xffffU) << std::dec
        << " peer_stream " << h.peer_stream << " peer_gen " << h.peer_generation;
    return out.str();
}

// The handshake. With the stream module in use, frames written before the peer has opened its end can
// hold back the later frames of that direction, so rank 0 never writes first: rank 1 sends hellos
// (every 250 ms) and, without an answer within 1.5 s of opening, closes the stream, waits 0.5 s (so
// that the CLOSE rank 0 sends while reopening finds no open end here) and opens it again with a new
// generation. Rank 0 answers each new (nonce, generation) hello with its hello and an acknowledgement
// in one write, and reopens at once when it reads the peer's CLOSE. Rank 1 acknowledges rank 0's hello.
// Every frame either side writes is consumed by the other inside the handshake, so the stream is clean
// afterwards.
struct Handshake {
    std::uint32_t rank;
    std::uint16_t version = pairing_version;
    std::uint64_t nonce;
    std::string host;
    std::chrono::milliseconds answer{1500}, closed_wait{500}, resend{250};
};
Hello make(const Handshake& hs, const Candidate& c, std::uint32_t index, std::uint16_t kind) {
    Hello h;
    h.version = hs.version;
    h.nonce = hs.nonce; h.rank = hs.rank; h.stream = index; h.kind = kind; h.device = trailing_number(c.info.device);
    h.generation = c.generation;
    std::memcpy(h.host, hs.host.data(), std::min(hs.host.size(), sizeof(h.host)));
    if (kind == 2U) { h.peer_nonce = c.peer.nonce; h.peer_stream = c.peer.stream; h.peer_generation = c.peer.generation; }
    h.crc = body_crc(h);
    return h;
}
void queue(Candidate& c, const Hello& first, const Hello* second = nullptr) {
    std::memcpy(c.out.data(), &first, sizeof(first));
    c.out_size = sizeof(first);
    if (second) { std::memcpy(c.out.data() + sizeof(first), second, sizeof(*second)); c.out_size += sizeof(*second); }
    c.out_done = 0;
}
void open_stream(Candidate& c) {
    c.fd = ::open(c.info.device.c_str(), O_RDWR | O_CLOEXEC | O_NONBLOCK);
    if (c.fd < 0) {
        const int e = errno;
        c.last_error = std::string("open failed: ") + errno_text(e);
        // Transient right after the previous session's teardown (the driver's reset after DMA teardown can still be
        // pending, and the rings cannot be allocated yet): try again every 250 ms within the pairing deadline.
        if (e == ENOMEM || e == EBUSY || e == EAGAIN) {
            ++c.open_retries;
            c.reopen_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            return;
        }
        c.dead = true;
        return;
    }
    if (c.open_retries) note(c, "opened after " + std::to_string(c.open_retries) + " transient open failure(s)");
    ++c.generation;
    c.opened_at = std::chrono::steady_clock::now();
    c.last_hello = {};
    c.have = 0; c.out_size = c.out_done = 0;
    c.hello_ok = c.ack_ok = c.reply_queued = c.reply_written = false;
    note(c, "opened generation " + std::to_string(c.generation));
}
void close_stream(Candidate& c, const std::string& why) {
    note(c, "closing: " + why);
    if (c.fd >= 0) ::close(c.fd);
    c.fd = -1;
    ++c.reopens;
}
bool handshake_done(const Candidate& c, std::uint32_t rank) {
    return rank == 0U ? c.reply_written && c.ack_ok : c.hello_ok && c.ack_ok && c.reply_written;
}
// One non-blocking step on one stream; returns true if anything moved.
bool handshake_step(Candidate& c, std::uint32_t index, const Handshake& hs) {
    const auto now = std::chrono::steady_clock::now();
    if (c.fd < 0) {
        if (now >= c.reopen_at) open_stream(c); // rank 0's reopen_at is in the past unless an open failed transiently
        return c.fd >= 0;
    }
    const auto lost = [&](const std::string& why) {
        close_stream(c, why);
        if (hs.rank == 0U) open_stream(c);
        else c.reopen_at = std::chrono::steady_clock::now() + hs.closed_wait;
        return true;
    };
    bool moved = false;
    // Rank 1: hellos until rank 0 answers, then a fresh open if it does not.
    if (hs.rank == 1U && !(c.hello_ok && c.ack_ok)) {
        if (now - c.opened_at > hs.answer) return lost("no answer within " + std::to_string(hs.answer.count()) + " ms");
        if (c.out_done == c.out_size && now - c.last_hello >= hs.resend) {
            queue(c, make(hs, c, index, 1U));
            c.last_hello = now;
        }
    }
    if (c.out_done == c.out_size && c.hello_ok && !c.reply_queued && (hs.rank == 1U ? c.ack_ok : true)) {
        if (hs.rank == 0U) {
            const auto hello = make(hs, c, index, 1U), ack = make(hs, c, index, 2U);
            queue(c, hello, &ack);
        } else {
            queue(c, make(hs, c, index, 2U));
        }
        c.reply_queued = true;
    }
    if (c.out_done < c.out_size) {
        const auto n = ::write(c.fd, c.out.data() + c.out_done, c.out_size - c.out_done);
        if (n > 0) {
            c.out_done += static_cast<std::size_t>(n);
            moved = true;
            if (c.out_done == c.out_size) {
                if (c.reply_queued && !c.reply_written) { c.reply_written = true; note(c, hs.rank == 0U ? "wrote hello + ack" : "wrote ack"); }
                else note(c, "wrote hello");
            }
        } else if (n < 0 && errno != EAGAIN && errno != EINTR) {
            c.last_error = std::string("write: ") + std::strerror(errno);
            return lost(c.last_error);
        }
    }
    const auto n = ::read(c.fd, c.in.data() + c.have, c.in.size() - c.have);
    if (n > 0) { c.have += static_cast<std::size_t>(n); moved = true; }
    else if (n == 0) { c.last_error = "peer closed the stream"; return lost("read end of stream (peer CLOSE)"); }
    else if (errno != EAGAIN && errno != EINTR) { c.last_error = std::string("read: ") + std::strerror(errno); return lost(c.last_error); }
    if (c.have == c.in.size()) {
        c.have = 0;
        Hello h;
        std::memcpy(&h, c.in.data(), sizeof(h));
        if (h.magic != hello_magic || h.crc != body_crc(h) || h.version != hs.version) {
            ++c.garbage; // stale bytes of an earlier session (whole 64-byte units)
            note(c, "read garbage unit");
        } else if (h.kind == 1U) {
            const bool fresh = !c.hello_ok || h.nonce != c.peer.nonce || h.generation != c.peer.generation;
            note(c, "read " + describe(h) + (fresh ? "" : " (repeat)"));
            if (fresh) {
                c.peer = h;
                c.hello_ok = true;
                if (hs.rank == 0U) { c.reply_queued = c.reply_written = c.ack_ok = false; }
            }
        } else if (h.kind == 2U && c.hello_ok && h.nonce == c.peer.nonce && h.generation == c.peer.generation &&
                   h.peer_nonce == hs.nonce && h.peer_stream == index && h.peer_generation == c.generation) {
            note(c, "read " + describe(h) + " (accepted)");
            c.ack_ok = true;
        } else {
            note(c, "read " + describe(h) + " (not for this open)");
            ++c.garbage;
        }
    }
    return moved;
}

// Version 2 used the same 128 bytes, with its CRC immediately after min_part and zero padding thereafter.
// Normalize it to Plan only inside this process; version 3 retains its existing wire representation.
void write_plan(int fd, const Plan& plan) {
    if (plan.version != 2U) {
        write_all(fd, std::span<const std::byte>(reinterpret_cast<const std::byte*>(&plan), sizeof(plan)), "usb4 plan");
        return;
    }
    Plan wire = plan;
    if (plan.version == 2U) {
        constexpr auto prefix = offsetof(Plan, streams_up_to);
        auto* bytes = reinterpret_cast<std::byte*>(&wire);
        const auto crc = crc32c(std::span<const std::byte>(bytes, prefix));
        std::memset(bytes + prefix, 0, sizeof(wire) - prefix);
        std::memcpy(bytes + prefix, &crc, sizeof(crc));
    }
    write_all(fd, std::span<const std::byte>(reinterpret_cast<const std::byte*>(&wire), sizeof(wire)), "usb4 plan");
}
Plan check_plan(Plan plan, std::uint32_t version = pairing_version) {
    bool crc_ok = false;
    if (version == 2U) {
        constexpr auto prefix = offsetof(Plan, streams_up_to);
        auto* bytes = reinterpret_cast<std::byte*>(&plan);
        std::uint32_t crc;
        std::memcpy(&crc, bytes + prefix, sizeof(crc));
        crc_ok = crc == crc32c(std::span<const std::byte>(bytes, prefix));
        std::memset(bytes + prefix, 0, sizeof(plan) - prefix);
        plan.crc = 0;
    } else {
        crc_ok = plan.crc == body_crc(plan);
    }
    if (plan.magic != plan_magic || !crc_ok || plan.version != version || plan.count > max_streams ||
        plan.count == 0U)
        fail("usb4 pairing: malformed plan from the peer");
    return plan;
}
// Pairing reads have a deadline (poll), so a peer that died or a link that went down mid-pairing fails
// the pairing instead of blocking it.
constexpr std::chrono::milliseconds plan_wait{20000};
Plan read_plan(int fd, std::uint32_t version = pairing_version) {
    Plan plan;
    auto* bytes = reinterpret_cast<std::byte*>(&plan);
    std::size_t done = 0;
    const auto deadline = std::chrono::steady_clock::now() + plan_wait;
    const char* spin_env = std::getenv("STRIX_DIST_USB4_SPIN");
    const bool spin = spin_env && std::string(spin_env) == "1";
    if (spin) {
        // busy_poll streams intentionally reject poll(). Keep pairing bounded
        // using the same explicit nonblocking mode as the data path.
        const int flags = ::fcntl(fd, F_GETFL);
        if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
            fail(std::string("usb4 pairing: cannot make plan read nonblocking: ") + std::strerror(errno));
    }
    while (done < sizeof(plan)) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) fail("usb4 pairing: no plan from the peer within " + std::to_string(plan_wait.count()) + " ms");
        if (!spin) {
            pollfd p{fd, POLLIN, 0};
            const int ready = ::poll(&p, 1, static_cast<int>(left));
            if (ready < 0 && errno == EINTR) continue;
            if (ready < 0 || (p.revents & (POLLERR | POLLHUP))) fail("usb4 pairing: the stream failed while waiting for the plan");
            if (ready == 0) continue;
        }
        const auto n = ::read(fd, bytes + done, sizeof(plan) - done);
        if (n == 0) fail("usb4 pairing: the peer closed the stream before the plan");
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) { if (spin) cpu_relax(); continue; }
            fail(std::string("usb4 pairing: plan read failed: ") + std::strerror(errno));
        }
        done += static_cast<std::size_t>(n);
    }
    return check_plan(plan, version);
}
bool same_plan(const Plan& a, const Plan& b) {
    if (a.count != b.count || a.min_part != b.min_part || a.flags != b.flags) return false;
    for (std::uint32_t i = 0; i < a.count; ++i)
        if (a.index[i] != b.index[i] || a.weight[i] != b.weight[i]) return false;
    for (std::size_t i = 0; i < 4U; ++i)
        if (a.streams_up_to[i] != b.streams_up_to[i] || a.streams_count[i] != b.streams_count[i]) return false;
    return true;
}
// Rank 0 sends the plan on its first stream; rank 1 echoes it after checking it.
void agree(int fd, std::uint32_t rank, Plan& plan) {
    plan.crc = body_crc(plan);
    if (rank == 0U) {
        write_plan(fd, plan);
        if (!same_plan(read_plan(fd, plan.version), plan)) fail("usb4 pairing: the peer did not echo the plan");
    } else {
        const auto theirs = read_plan(fd, plan.version);
        bool ok = theirs.count == plan.count;
        for (std::uint32_t i = 0; ok && i < plan.count; ++i) ok = theirs.index[i] == plan.index[i];
        write_plan(fd, theirs);
        if (!ok) fail("usb4 pairing: the two ranks disagree on the streams");
        plan = theirs;
    }
}

double probe_exchange(const std::vector<int>& fds, std::size_t bytes, std::uint32_t rank) {
    const std::size_t count = fds.size();
    std::vector<std::vector<std::byte>> out(count), in(count);
    for (std::size_t k = 0; k < count; ++k) {
        const std::size_t size = (bytes / count) & ~std::size_t{63};
        out[k].resize(size);
        in[k].resize(size);
        for (std::size_t i = 0; i < size; ++i) out[k][i] = static_cast<std::byte>((i * 131U + k * 7U + rank) & 0xffU);
    }
    const auto start = std::chrono::steady_clock::now();
    // A writer and a reader per stream: the probe is larger than what a stream may buffer.
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(2U * count);
    for (std::size_t k = 0; k < count; ++k) {
        threads.emplace_back([&, k] {
            try { write_all(fds[k], out[k], "usb4 probe"); } catch (...) { errors[2U * k] = std::current_exception(); }
        });
        threads.emplace_back([&, k] {
            try { read_all(fds[k], in[k], "usb4 probe"); } catch (...) { errors[2U * k + 1U] = std::current_exception(); }
        });
    }
    for (auto& t : threads) t.join();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (const auto& e : errors)
        if (e) std::rethrow_exception(e);
    for (std::size_t k = 0; k < count; ++k)
        for (std::size_t i = 0; i < in[k].size(); ++i)
            if (in[k][i] != static_cast<std::byte>((i * 131U + k * 7U + (1U - rank)) & 0xffU))
                fail("usb4 probe: corrupted bytes on stream " + std::to_string(k));
    return seconds;
}

// Every stream at once, stream k carrying parts[k] bytes each way: per stream, the seconds until both its write and its
// read are complete (the rate each stream really gets while all of them share the host).
std::vector<double> probe_joint(const std::vector<int>& fds, const std::vector<std::size_t>& parts, std::uint32_t rank) {
    const std::size_t count = fds.size();
    std::vector<std::vector<std::byte>> out(count), in(count);
    for (std::size_t k = 0; k < count; ++k) {
        out[k].resize(parts[k]);
        in[k].resize(parts[k]);
        for (std::size_t i = 0; i < parts[k]; ++i) out[k][i] = static_cast<std::byte>((i * 131U + k * 7U + rank) & 0xffU);
    }
    std::vector<std::chrono::steady_clock::time_point> wend(count), rend(count);
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(2U * count);
    for (std::size_t k = 0; k < count; ++k) {
        threads.emplace_back([&, k] {
            try { write_all(fds[k], out[k], "usb4 joint probe"); } catch (...) { errors[2U * k] = std::current_exception(); }
            wend[k] = std::chrono::steady_clock::now();
        });
        threads.emplace_back([&, k] {
            try { read_all(fds[k], in[k], "usb4 joint probe"); } catch (...) { errors[2U * k + 1U] = std::current_exception(); }
            rend[k] = std::chrono::steady_clock::now();
        });
    }
    for (auto& t : threads) t.join();
    for (const auto& e : errors)
        if (e) std::rethrow_exception(e);
    std::vector<double> seconds(count);
    for (std::size_t k = 0; k < count; ++k) {
        for (std::size_t i = 0; i < in[k].size(); ++i)
            if (in[k][i] != static_cast<std::byte>((i * 131U + k * 7U + (1U - rank)) & 0xffU))
                fail("usb4 joint probe: corrupted bytes on stream " + std::to_string(k));
        seconds[k] = std::chrono::duration<double>(std::max(wend[k], rend[k]) - start).count();
    }
    return seconds;
}

// STRIX_DIST_USB4_STREAMS="UP_TO:N,..." (UP_TO in message bytes incl. the 128-byte trailer, or "inf"; at most 4 entries,
// increasing): a message of at most UP_TO bytes uses the first N streams of the striping order (one per cable first).
void parse_stream_table(const char* text, std::uint32_t (&up_to)[4], std::uint8_t (&count)[4]) {
    std::stringstream input(text);
    std::size_t n = 0;
    std::uint64_t last = 0;
    for (std::string item; std::getline(input, item, ',');) {
        const auto colon = item.find(':');
        if (n == 4U || colon == std::string::npos) fail("STRIX_DIST_USB4_STREAMS: at most 4 UP_TO:N entries");
        const std::string limit = item.substr(0, colon);
        const std::uint64_t bytes = limit == "inf" ? UINT32_MAX : std::stoull(limit);
        const auto streams = std::stoul(item.substr(colon + 1U));
        if (bytes == 0U || bytes > UINT32_MAX || bytes <= last || streams == 0U || streams > max_streams)
            fail("STRIX_DIST_USB4_STREAMS: UP_TO increasing in 1..4294967295 (or inf), N in 1..8");
        up_to[n] = static_cast<std::uint32_t>(bytes);
        count[n] = static_cast<std::uint8_t>(streams);
        last = bytes;
        ++n;
    }
}

class TcpTransport final : public Transport {
public:
    TcpTransport(int fd, std::chrono::milliseconds timeout) : fd_(fd), timeout_(timeout) {
        const int one = 1;
        ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    ~TcpTransport() override {
        sender_.reset();
        ::close(fd_);
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        sender_->post([this, out] { send(out); });
        std::exception_ptr error;
        try { receive(in); } catch (...) { error = std::current_exception(); }
        sender_->wait();
        if (error) std::rethrow_exception(error);
    }
    void send(std::span<const std::byte> message) override {
        std::size_t done = 0;
        while (done < message.size()) {
            wait_for(POLLOUT, "send");
            const auto n = ::send(fd_, message.data() + done, message.size() - done, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) fail(std::string("send failed: ") + std::strerror(errno));
            done += static_cast<std::size_t>(n);
        }
    }
    void receive(std::span<std::byte> message) override {
        std::size_t done = 0;
        while (done < message.size()) {
            wait_for(POLLIN, "receive");
            const auto n = ::recv(fd_, message.data() + done, message.size() - done, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n == 0) fail("peer closed the connection");
            if (n < 0) fail(std::string("receive failed: ") + std::strerror(errno));
            done += static_cast<std::size_t>(n);
        }
    }
    [[nodiscard]] std::string name() const override { return "tcp"; }

private:
    void wait_for(short events, const char* what) const {
        pollfd p{fd_, events, 0};
        const int ready = ::poll(&p, 1, static_cast<int>(timeout_.count()));
        if (ready == 0) fail(std::string(what) + " timed out");
        if (ready < 0 && errno != EINTR) fail(std::string(what) + " poll failed: " + std::strerror(errno));
    }
    int fd_;
    std::chrono::milliseconds timeout_;
    std::unique_ptr<Worker> sender_ = std::make_unique<Worker>();
};

} // namespace

std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept {
#if defined(__x86_64__)
    static const bool hardware = __builtin_cpu_supports("sse4.2");
    if (hardware) {
        static const bool parallel = [] { const char* v = std::getenv("STRIX_DIST_CRC_PARALLEL"); return !v || std::string(v) != "0"; }();  // default on
        if (parallel && bytes.size() >= 1024U && detail::crc32c_vector_available()) return detail::crc32c_vector(bytes);
        if (parallel && bytes.size() >= 12288U) return detail::crc32c_parallel(bytes);
        return crc32c_sse42(bytes);
    }
#endif
    const auto& table = crc_table();
    std::uint32_t c = 0xffffffffU;
    for (const auto b : bytes) c = table[(c ^ static_cast<std::uint8_t>(b)) & 0xffU] ^ (c >> 8U);
    return c ^ 0xffffffffU;
}

std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> memory_pair() {
    auto a = std::make_shared<Channel>(), b = std::make_shared<Channel>();
    return {std::make_unique<MemoryTransport>(a, b), std::make_unique<MemoryTransport>(b, a)};
}

std::unique_ptr<Transport> usb4_host(const std::vector<std::string>& devices, const std::vector<std::uint32_t>& weights) {
    if (devices.empty() || devices.size() > max_streams) fail("usb4 host transport takes one to eight stream devices");
    if (!weights.empty() && weights.size() != devices.size()) fail("usb4 host transport: one weight per device");
    std::vector<int> fds;
    for (const auto& path : devices) {
        // Non-blocking open: a stream whose link is down fails at once (ENXIO) instead of waiting for it.
        const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) {
            for (const int open_fd : fds) ::close(open_fd);
            fail("cannot open " + path + ": " + std::strerror(errno));
        }
        fds.push_back(fd); // stays non-blocking: poll() bounds every wait (deadline, abort)
    }
    return std::make_unique<Usb4HostTransport>(std::move(fds), devices,
                                               weights.empty() ? std::vector<std::uint32_t>(devices.size(), 1U) : weights);
}

std::unique_ptr<Transport> usb4_host_descriptors(std::vector<int> fds, std::vector<std::string> names, std::vector<std::uint32_t> weights,
                                                 std::size_t min_part, std::chrono::milliseconds timeout) {
    return std::make_unique<Usb4HostTransport>(std::move(fds), std::move(names), std::move(weights), min_part, timeout);
}

std::vector<Usb4LayoutStream> usb4_layout_streams(const Usb4AutoOptions& options) {
    std::string path = options.layout;
    if (path.empty()) {
        const char* env = std::getenv("STRIX_DIST_USB4_LAYOUT");
        path = env && *env ? env : "/run/usb4stream-layout.json";
    }
    std::ifstream file(path);
    if (!file) fail("cannot read the USB4 stream layout " + path);
    std::stringstream text;
    text << file.rdbuf();
    const auto layout = strix::serve::parse_json(text.str());
    std::vector<Usb4LayoutStream> out;
    if (!layout.is_object()) return out;
    for (const auto& [key, entry] : layout.as_object()) {
        const auto slash = key.find('/');
        if (slash == std::string::npos || !entry.is_object()) continue;
        Usb4LayoutStream st;
        st.key = key;
        st.service = key.substr(0, slash);
        st.stream = key.substr(slash + 1U);
        if (const auto* d = entry.find("device"); d && d->is_string()) st.device = d->as_string();
        if (const auto* cfg = entry.find("configfs"); cfg && cfg->is_string()) st.configfs = cfg->as_string();
        const auto dot = st.service.rfind('.');
        st.xdomain = dot == std::string::npos ? st.service : st.service.substr(0, dot);
        st.attached = exists(options.sysfs + "/" + st.service);
        st.gpu_direct = !st.configfs.empty() && read_text(st.configfs + "/gpu_direct") == "1";
        out.push_back(std::move(st));
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    return out;
}

std::string Usb4Topology::receipt() const {
    std::ostringstream out;
    std::map<std::string, int> cables;
    for (const auto& s : streams) cables[s.xdomain] += 1;
    out << "usb4 auto-pairing: " << local_host << " <-> " << remote_host;
    if (!remote_unique_id.empty()) out << " (" << remote_unique_id << ")";
    out << ", " << streams.size() << " stream(s) on " << cables.size() << " cable(s)";
    if (min_part_bytes) out << ", messages below " << min_part_bytes << " B per stream use fewer streams";
    for (const auto& s : streams) {
        out << "; " << s.key << ' ' << s.device << " [domain " << s.domain;
        if (s.usb4_port >= 0) out << " port " << s.usb4_port;
        if (!s.cable.empty()) out << " cable " << s.cable;
        out << ", link " << s.link_gbps << " Gb/s] <-> peer tbstream" << s.remote_device << " (index " << s.remote_index
            << "), cable " << s.xdomain << " measured " << s.measured_gbps << " Gb/s, weight " << s.weight;
    }
    for (const auto& r : rejected) out << "; not used: " << r;
    return out.str();
}

std::unique_ptr<Transport> usb4_host_auto(const Usb4AutoOptions& options, Usb4Topology* topology) {
    if (options.rank > 1U) fail("usb4 pairing: rank must be 0 or 1");
    Usb4Topology topo;
    topo.local_host = host_name();
    auto candidates = layout_candidates(options, topo.rejected);
    std::random_device random;
    Handshake hs;
    hs.rank = options.rank;
    hs.version = options.minimax_compat ? 2U : pairing_version;
    hs.host = topo.local_host;
    hs.nonce = (static_cast<std::uint64_t>(random()) << 32U) ^ random() ^
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) ^ static_cast<std::uint64_t>(::getpid());
    for (auto& c : candidates) open_stream(c);
    const auto close_all = [&] { for (auto& c : candidates) if (c.fd >= 0) { ::close(c.fd); c.fd = -1; } };
    const auto start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point first_done{};
    while (true) {
        bool moved = false, all = true;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            auto& c = candidates[i];
            if (c.dead) continue;
            if (!handshake_done(c, hs.rank)) moved = handshake_step(c, static_cast<std::uint32_t>(i), hs) || moved;
            if (c.dead) continue;
            if (handshake_done(c, hs.rank)) { if (first_done == std::chrono::steady_clock::time_point{}) first_done = std::chrono::steady_clock::now(); }
            else all = false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (all) break;
        if (first_done != std::chrono::steady_clock::time_point{} ? now - first_done > options.grace : now - start > options.timeout) break;
        if (!moved) std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (pairing_debug())
        for (const auto& c : candidates) {
            std::fprintf(stderr, "usb4 pairing trace %s %s (rank %u, nonce %llx):\n", c.info.key.c_str(), c.info.device.c_str(),
                         options.rank, static_cast<unsigned long long>(hs.nonce & 0xffffU));
            for (const auto& line : c.trace) std::fprintf(stderr, "  %s\n", line.c_str());
        }
    std::vector<std::size_t> accepted;
    const auto reject = [&](Candidate& c, const std::string& why) {
        std::string text = c.info.key + " " + c.info.device + ": " + why;
        if (c.reopens) text += ", " + std::to_string(c.reopens) + " reopen(s)";
        if (c.garbage) text += ", " + std::to_string(c.garbage) + " stale 64-byte units discarded";
        if (!c.last_error.empty()) text += ", last error " + c.last_error;
        topo.rejected.push_back(text);
        if (c.fd >= 0) ::close(c.fd);
        c.fd = -1;
    };
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        auto& c = candidates[i];
        if (c.fd < 0 && (c.dead || c.open_retries)) { reject(c, "could not be opened"); continue; }
        if (handshake_done(c, hs.rank)) { accepted.push_back(i); continue; }
        reject(c, !c.hello_ok ? "no answer from the peer"
                  : "the peer's hello arrived but the handshake did not complete (one direction only?)");
    }
    const auto fail_all = [&](const std::string& what) {
        close_all();
        std::string all = "usb4 pairing: " + what;
        for (const auto& r : topo.rejected) all += "; " + r;
        fail(all);
    };
    if (accepted.empty()) fail_all("no stream paired with the peer");
    const auto& first_peer = candidates[accepted[0]].peer;
    for (const auto i : accepted) {
        const auto& p = candidates[i].peer;
        if (p.nonce != first_peer.nonce || p.rank != 1U - options.rank || p.nonce == hs.nonce)
            fail_all("stream " + candidates[i].info.key + " leads to " +
                     (p.rank == options.rank ? std::string("a process with the same rank") : std::string("a different peer process")) +
                     "; check the ranks and that only one job uses the streams");
    }
    topo.remote_host = std::string(first_peer.host, strnlen(first_peer.host, sizeof(first_peer.host)));
    topo.remote_unique_id = read_text(options.sysfs + "/" + candidates[accepted[0]].info.xdomain + "/unique_id");
    for (const auto i : accepted) {
        auto& c = candidates[i];
        c.info.rank0_index = options.rank == 0U ? static_cast<std::uint32_t>(i) : c.peer.stream;
        c.info.remote_index = c.peer.stream;
        c.info.remote_device = c.peer.device;
    }
    std::sort(accepted.begin(), accepted.end(), [&](std::size_t a, std::size_t b) {
        return candidates[a].info.rank0_index < candidates[b].info.rank0_index;
    });
    try {
        // Rank 0's accepted streams are a subset of rank 1's (rank 0 completes only on rank 1's
        // acknowledgement, which rank 1 writes when it completes). Rank 0 sends its set on its first
        // stream; rank 1 looks for it on all of its streams, keeps that set and echoes it.
        Plan plan;
        plan.version = hs.version;
        std::size_t plan_stream = accepted[0];
        if (options.rank == 0U) {
            plan.count = static_cast<std::uint32_t>(accepted.size());
            for (std::size_t k = 0; k < accepted.size(); ++k) plan.index[k] = candidates[accepted[k]].info.rank0_index;
            plan.crc = body_crc(plan);
            set_blocking(candidates[plan_stream].fd, true);
            write_plan(candidates[plan_stream].fd, plan);
            if (!same_plan(read_plan(candidates[plan_stream].fd, hs.version), plan)) fail("usb4 pairing: the peer did not echo the plan");
        } else {
            std::map<std::size_t, std::pair<std::array<std::byte, sizeof(Plan)>, std::size_t>> partial;
            const auto deadline = std::chrono::steady_clock::now() + options.timeout;
            bool found = false;
            while (!found) {
                bool moved = false;
                for (const auto i : accepted) {
                    auto& [buffer, have] = partial[i];
                    const auto n = ::read(candidates[i].fd, buffer.data() + have, buffer.size() - have);
                    if (n > 0) { have += static_cast<std::size_t>(n); moved = true; }
                    else if (n == 0) fail("usb4 pairing: the peer closed " + candidates[i].info.key + " before sending the plan");
                    if (have == buffer.size()) {
                        std::memcpy(&plan, buffer.data(), sizeof(plan));
                        plan = check_plan(plan, hs.version);
                        plan_stream = i;
                        found = true;
                        break;
                    }
                }
                if (!found && std::chrono::steady_clock::now() > deadline) fail("usb4 pairing: no plan from rank 0");
                if (!moved) std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            std::vector<std::size_t> kept;
            for (std::uint32_t k = 0; k < plan.count; ++k) {
                const auto it = std::find_if(accepted.begin(), accepted.end(), [&](std::size_t i) { return candidates[i].info.rank0_index == plan.index[k]; });
                if (it == accepted.end()) fail("usb4 pairing: rank 0's plan names a stream this rank did not pair");
                kept.push_back(*it);
            }
            if (kept.empty() || kept[0] != plan_stream) fail("usb4 pairing: the plan arrived on an unexpected stream");
            for (const auto i : accepted)
                if (std::find(kept.begin(), kept.end(), i) == kept.end())
                    reject(candidates[i], "paired here, but rank 0 did not complete its handshake (its acknowledgement was late)");
            accepted = kept;
            set_blocking(candidates[plan_stream].fd, true);
            write_plan(candidates[plan_stream].fd, plan);
        }
        // Striping order: the first stream of every cable, then the second of every cable, ... (cables in
        // the order of their first stream), so a message on k streams spreads over the cables. Both ranks
        // derive it from rank 0's indices and the cable partition, which they share.
        {
            std::vector<std::string> cable_order;
            for (const auto i : accepted)
                if (std::find(cable_order.begin(), cable_order.end(), candidates[i].info.xdomain) == cable_order.end())
                    cable_order.push_back(candidates[i].info.xdomain);
            std::map<std::size_t, std::pair<std::size_t, std::size_t>> key;  // candidate -> (position in cable, cable)
            std::map<std::string, std::size_t> seen;
            for (const auto i : accepted) {
                const auto& x = candidates[i].info.xdomain;
                key[i] = {seen[x]++, static_cast<std::size_t>(std::find(cable_order.begin(), cable_order.end(), x) - cable_order.begin())};
            }
            std::stable_sort(accepted.begin(), accepted.end(), [&](std::size_t a, std::size_t b) { return key[a] < key[b]; });
        }
        // The data path keeps the descriptors non-blocking (poll() bounds every wait, so a deadline or an abort always
        // ends it; a blocking read could outlive both).
        for (const auto i : accepted) set_blocking(candidates[i].fd, false);
        std::vector<int> fds;
        for (const auto i : accepted) fds.push_back(candidates[i].fd);
        // Cables (XDomains) in striping order; each one measured with all its streams at once.
        std::vector<std::string> order;
        for (const auto i : accepted)
            if (std::find(order.begin(), order.end(), candidates[i].info.xdomain) == order.end()) order.push_back(candidates[i].info.xdomain);
        for (const auto& x : order) {
            std::vector<int> cable_fds;
            std::vector<std::size_t> members;
            for (const auto i : accepted)
                if (candidates[i].info.xdomain == x) { cable_fds.push_back(candidates[i].fd); members.push_back(i); }
            double gbps = 0.0;
            if (options.probe_bytes) {
                // A 64-byte barrier on the cable before each repetition: without it the rank that starts
                // later finds the peer's data already buffered and overstates the rate.
                const auto barrier = [&] {
                    std::array<std::byte, 64> token{}, back{};
                    write_all(cable_fds[0], token, "usb4 probe barrier");
                    read_all(cable_fds[0], back, "usb4 probe barrier");
                };
                barrier();
                (void)probe_exchange(cable_fds, std::min<std::size_t>(options.probe_bytes, 256U * 1024U), options.rank); // warm-up
                std::vector<double> times;
                for (int rep = 0; rep < 5; ++rep) {
                    barrier();
                    times.push_back(probe_exchange(cable_fds, options.probe_bytes, options.rank));
                }
                std::sort(times.begin(), times.end());
                gbps = static_cast<double>((options.probe_bytes / cable_fds.size()) & ~std::size_t{63}) * static_cast<double>(cable_fds.size()) * 8.0 / times[2] / 1e9;
            }
            for (const auto i : members) candidates[i].info.measured_gbps = gbps;
        }
        // Weights (rank 0's measurement decides): the cable's rate in Mb/s shared among its streams. The plan
        // lists the streams in striping order now.
        Plan weights = plan;
        for (std::size_t k = 0; k < accepted.size(); ++k) weights.index[k] = candidates[accepted[k]].info.rank0_index;
        weights.min_part = static_cast<std::uint32_t>(options.min_part_bytes);
        if (const char* mp = std::getenv("STRIX_DIST_USB4_MIN_PART"); mp && *mp) weights.min_part = static_cast<std::uint32_t>(std::stoul(mp));
        for (std::size_t k = 0; k < accepted.size(); ++k) {
            const auto& info = candidates[accepted[k]].info;
            const auto members = static_cast<double>(std::count_if(accepted.begin(), accepted.end(), [&](std::size_t i) { return candidates[i].info.xdomain == info.xdomain; }));
            const double rate = options.probe_bytes ? info.measured_gbps : info.link_gbps;
            weights.weight[k] = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(rate * 1000.0 / members));
        }
        // Rank 0's choices (rank 1 adopts its plan): explicit weights, the stream table, the joint calibration.
        // STRIX_DIST_USB4_WEIGHTS=W0,W1,... (striping order) replaces the measured weights. STRIX_DIST_USB4_CALIBRATE=joint
        // re-measures them with every cable at once: each cable alone runs near the same rate, but together the Gen2
        // cable's streams get less than the Gen3 ones, so per-cable weights leave the faster cable idle at the end of a
        // bulk message (transport measurements).
        bool explicit_weights = false;
        if (options.rank == 0U) {
            if (const char* w = std::getenv("STRIX_DIST_USB4_WEIGHTS"); w && *w) {
                std::stringstream input(w);
                std::size_t k = 0;
                for (std::string item; std::getline(input, item, ','); ++k) {
                    if (k == accepted.size()) fail("STRIX_DIST_USB4_WEIGHTS: more weights than streams");
                    const auto value = std::stoul(item);
                    if (value == 0U || value > 1000000U) fail("STRIX_DIST_USB4_WEIGHTS: weights in 1..1000000");
                    weights.weight[k] = static_cast<std::uint32_t>(value);
                }
                if (k != accepted.size()) fail("STRIX_DIST_USB4_WEIGHTS: one weight per stream (" + std::to_string(accepted.size()) + ")");
                explicit_weights = true;
            }
            if (const char* t = std::getenv("STRIX_DIST_USB4_STREAMS"); !options.minimax_compat && t && *t) parse_stream_table(t, weights.streams_up_to, weights.streams_count);
            if (const char* c = std::getenv("STRIX_DIST_USB4_CALIBRATE"); !options.minimax_compat && c && std::string(c) == "joint" && !explicit_weights && accepted.size() > 1U)
                weights.flags |= plan_joint;
        }
        agree(fds[0], options.rank, weights);
        if (weights.flags & plan_joint) {
            const std::size_t total = std::max<std::size_t>(options.probe_bytes, std::size_t{16} << 20U) * order.size();
            const auto barrier = [&] {
                std::array<std::byte, 64> token{}, back{};
                write_all(fds[0], token, "usb4 joint probe barrier");
                read_all(fds[0], back, "usb4 joint probe barrier");
            };
            for (int round = 0; round < 2; ++round) {
                std::uint64_t sum = 0;
                for (std::size_t k = 0; k < accepted.size(); ++k) sum += weights.weight[k];
                std::vector<std::size_t> parts(accepted.size());
                for (std::size_t k = 0; k < accepted.size(); ++k) parts[k] = std::max<std::size_t>(64U, (total * weights.weight[k] / sum) & ~std::size_t{63});
                std::vector<std::vector<double>> times(accepted.size());
                for (int rep = 0; rep < 3; ++rep) {
                    barrier();
                    const auto t = probe_joint(fds, parts, options.rank);
                    for (std::size_t k = 0; k < t.size(); ++k) times[k].push_back(t[k]);
                }
                for (std::size_t k = 0; k < accepted.size(); ++k) {
                    std::sort(times[k].begin(), times[k].end());
                    const double gbps = static_cast<double>(parts[k]) * 8.0 / times[k][1] / 1e9;  // median of 3, one direction
                    weights.weight[k] = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(gbps * 1000.0));
                }
                agree(fds[0], options.rank, weights);  // rank 0's measurement decides
            }
        }
        std::vector<std::string> names;
        std::vector<std::uint32_t> stream_weights;
        for (std::size_t k = 0; k < accepted.size(); ++k) {
            auto& c = candidates[accepted[k]];
            c.info.weight = weights.weight[k];
            names.push_back(c.info.device + ">tbstream" + std::to_string(c.info.remote_device));
            stream_weights.push_back(weights.weight[k]);
            topo.streams.push_back(c.info);
        }
        for (const auto i : accepted) candidates[i].fd = -1; // owned by the transport from here on
        auto transport = std::make_unique<Usb4HostTransport>(std::move(fds), std::move(names), std::move(stream_weights), weights.min_part,
                                                             options.io_timeout);
        std::array<std::pair<std::size_t, std::size_t>, 4> table{};
        for (std::size_t i = 0; i < 4U && weights.streams_count[i]; ++i)
            table[i] = {weights.streams_up_to[i] == UINT32_MAX ? SIZE_MAX : std::size_t{weights.streams_up_to[i]}, std::size_t{weights.streams_count[i]}};
        transport->set_stream_table(table);
        topo.min_part_bytes = weights.min_part;
        if (topology) *topology = std::move(topo);
        return transport;
    } catch (...) {
        close_all();
        throw;
    }
}

std::unique_ptr<Transport> tcp_listen(std::uint16_t port, std::chrono::milliseconds timeout) {
    const int server = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) fail("socket failed");
    const int one = 1;
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || ::listen(server, 1) != 0) {
        ::close(server);
        fail(std::string("bind/listen failed: ") + std::strerror(errno));
    }
    pollfd p{server, POLLIN, 0};
    if (::poll(&p, 1, static_cast<int>(timeout.count())) <= 0) {
        ::close(server);
        fail("no peer connected before the timeout");
    }
    const int fd = ::accept(server, nullptr, nullptr);
    ::close(server);
    if (fd < 0) fail("accept failed");
    return std::make_unique<TcpTransport>(fd, timeout);
}

std::unique_ptr<Transport> tcp_connect(const std::string& host, std::uint16_t port, std::chrono::milliseconds timeout) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found) != 0 || !found) fail("cannot resolve " + host);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0 && ::connect(fd, found->ai_addr, found->ai_addrlen) == 0) {
            ::freeaddrinfo(found);
            return std::make_unique<TcpTransport>(fd, timeout);
        }
        if (fd >= 0) ::close(fd);
        if (std::chrono::steady_clock::now() > deadline) {
            ::freeaddrinfo(found);
            fail("cannot connect to " + host);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

namespace {
std::uint64_t wall_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
} // namespace

class Communicator::Progress {
public:
    Worker worker;
    // STRIX_DIST_CRC_DEFER: sequence + 1 of the last message whose job handed it to finish() (trailer identity
    // checked) before its deferred CRC work; the job keeps running that work afterwards.
    std::atomic<std::uint64_t> delivered{0};
};

// Opt-in full-host pipeline. Both ranks must use the same chunk size. Each chunk
// carries the unchanged payload CRC plus the complete collective identity and its
// offset/total/version; a failed chunk never becomes visible to a consumer kernel.
class Communicator::Pipeline {
public:
    std::size_t chunk = 0;
    std::size_t capacity = 0;  // largest message of the chunked path (larger ones take the plain host path)
    std::byte *send = nullptr, *receive = nullptr;
    hipStream_t tx = nullptr, rx = nullptr;
    hipEvent_t producer = nullptr, consumer = nullptr;
    std::vector<hipEvent_t> ready;
    std::atomic<void*> destination{nullptr};
    bool async = false;
    double stage = 0, wire = 0, check_seconds = 0, reuse_wait = 0;
    std::unique_ptr<Worker> crc_send, crc_receive;
    std::atomic<std::size_t> prepared{0}, received{0};
    std::atomic<bool> failed{false};
    std::mutex failure_mutex;
    std::exception_ptr failure;
    void publish(std::atomic<std::size_t>& counter, std::size_t value) {
        { std::lock_guard lock(failure_mutex);
          if (!failed.load(std::memory_order_relaxed)) counter.store(value, std::memory_order_release); }
        counter.notify_one();
    }
    void fail(std::exception_ptr error = {}) {
        { std::lock_guard lock(failure_mutex);
          if (!failure && error) failure = error;
          failed.store(true, std::memory_order_release);
          prepared.store(SIZE_MAX, std::memory_order_release);
          received.store(SIZE_MAX, std::memory_order_release); }
        prepared.notify_all(); received.notify_all();
    }
    void wait_count(std::atomic<std::size_t>& counter, std::size_t target) {
        for (auto seen = counter.load(std::memory_order_acquire); seen < target; seen = counter.load(std::memory_order_acquire)) {
            if (failed.load(std::memory_order_acquire)) throw std::runtime_error("dist pipeline: a stage failed");
            counter.wait(seen, std::memory_order_acquire);
        }
        if (failed.load(std::memory_order_acquire)) throw std::runtime_error("dist pipeline: a stage failed");
    }
    bool legacy = false;
    Pipeline(std::size_t bytes, std::size_t size, bool compatible) : chunk(size), capacity(bytes), legacy(compatible) {
        const auto count = (capacity - 1U) / chunk + 1U;
        const auto allocation = capacity + count * sizeof(Trailer);
        try {
            check(hipHostMalloc(reinterpret_cast<void**>(&send), allocation, hipHostMallocDefault), "pipeline send");
            check(hipHostMalloc(reinterpret_cast<void**>(&receive), allocation, hipHostMallocDefault), "pipeline receive");
            check(hipStreamCreateWithFlags(&tx, hipStreamNonBlocking), "pipeline tx stream");
            check(hipStreamCreateWithFlags(&rx, hipStreamNonBlocking), "pipeline rx stream");
            check(hipEventCreateWithFlags(&producer, hipEventDisableTiming), "pipeline producer");
            check(hipEventCreateWithFlags(&consumer, hipEventDisableTiming), "pipeline consumer");
            ready.resize(count, nullptr);
            for (auto& e : ready) check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "pipeline ready");
            if (const char* v = std::getenv("STRIX_DIST_HOST_CRC_OVERLAP"); compatible ? (v && std::string(v) == "1") : (!v || std::string(v) != "0")) {  // default on
                crc_send = std::make_unique<Worker>(); crc_receive = std::make_unique<Worker>();
                crc_send->name("u4-crctx"); crc_receive->name("u4-crcrx");
            }
        } catch (...) { cleanup(); throw; }
    }
    ~Pipeline() { cleanup(); }
    void cleanup() {
        fail();
        crc_send.reset(); crc_receive.reset(); // join jobs before releasing their events and pinned memory
        if (tx) (void)hipStreamSynchronize(tx);
        if (rx) (void)hipStreamSynchronize(rx);
        for (auto e : ready) if (e) (void)hipEventDestroy(e);
        if (producer) (void)hipEventDestroy(producer);
        if (consumer) (void)hipEventDestroy(consumer);
        if (tx) (void)hipStreamDestroy(tx);
        if (rx) (void)hipStreamDestroy(rx);
        if (send) (void)hipHostFree(send);
        if (receive) (void)hipHostFree(receive);
    }
};

Communicator::Communicator(Group group, std::unique_ptr<Transport> transport, std::size_t max_bytes)
    : group_(group), transport_(std::move(transport)), max_bytes_(max_bytes), progress_(std::make_unique<Progress>()) {
    if (group_.world != 2U || group_.rank > 1U || !transport_ || max_bytes_ == 0U)
        throw std::invalid_argument("dist communicator needs two ranks, a transport and a size bound");
    if (const char* c = std::getenv("STRIX_DIST_PROGRESS_CPU"); c && *c) {
        progress_->worker.pin(std::stoi(c));
        progress_->worker.name("u4-progress");
    }
    if (latency::enabled()) progress_->worker.name("u4-progress");
    if (const char* c = std::getenv("STRIX_DIST_PROGRESS_SPIN_US"); c && *c) {
        const auto duration = std::stoll(c);
        if (duration < 0 || duration > 10000)
            throw std::invalid_argument("STRIX_DIST_PROGRESS_SPIN_US must be 0..10000");
        progress_->worker.idle_spin(std::chrono::microseconds(duration));
    }
    if (const char* v = std::getenv("STRIX_DIST_ASYNC_START"); v && std::string(v) == "1") async_start_ = true;
    if (const char* t = std::getenv("STRIX_DIST_WAIT_TIMEOUT_MS"); t && *t) wait_timeout_ = std::chrono::milliseconds(std::stoll(t));
    if (const char* t = std::getenv("STRIX_DIST_HOST_D2H_KERNEL_BYTES"); t && *t) {
        d2h_kernel_bytes_ = std::stoull(t);
        if (d2h_kernel_bytes_ > (std::size_t{1} << 20U))
            throw std::invalid_argument("STRIX_DIST_HOST_D2H_KERNEL_BYTES must be 0..1048576");
    }
    // default 262144 (with the lean releases of STRIX_DS41_READY_LEAN); =0: no readiness views
    staged_ready_bytes_ = group_.minimax_compat ? 0U : 262144U;
    if (const char* t = std::getenv("STRIX_DIST_STAGED_READY_BYTES"); t && *t) {
        staged_ready_bytes_ = std::stoull(t);
        if (staged_ready_bytes_ > (std::size_t{1} << 20U))
            throw std::invalid_argument("STRIX_DIST_STAGED_READY_BYTES must be 0..1048576");
    }
    // Fused staging (default; STRIX_DIST_STAGING=copy: off): kernels write and read these buffers directly (send_staging /
    // finish_staged), so they are coherent (no stale GPU-cached lines of a reused receive buffer); copy: the default
    // pinned memory and copies only. The event-based fused path keeps its coherent pools, independently of the separate
    // small readiness pools; "ready" falls back to fused above its limit.
    if (const char* v = std::getenv("STRIX_DIST_STAGING"); v && std::string(v) == "copy") fused_ = false;
    const unsigned staging_flags = fused_ ? hipHostMallocCoherent : hipHostMallocDefault;
    check(hipHostMalloc(reinterpret_cast<void**>(&send_host_), max_bytes_ + sizeof(Trailer), staging_flags), "pinned send buffer");
    check(hipHostMalloc(reinterpret_cast<void**>(&receive_host_), max_bytes_ + sizeof(Trailer), staging_flags), "pinned receive buffer");
    if (d2h_kernel_bytes_) {
        const auto status = hipHostGetDevicePointer(&send_device_alias_, send_host_, 0);
        if (status != hipSuccess) {
            (void)hipHostFree(receive_host_);
            (void)hipHostFree(send_host_);
            receive_host_ = send_host_ = nullptr;
            check(status, "mapped pinned send buffer for copy kernel");
        }
    }
    if (staged_ready_bytes_) {
        try {
            int device = 0, native_atomics = 0;
            check(hipGetDevice(&device), "staged ready device");
            check(hipDeviceGetAttribute(&native_atomics, hipDeviceAttributeHostNativeAtomicSupported, device), "host atomics");
            if (!native_atomics) throw std::runtime_error("staged readiness requires native host atomics");
            const auto count = std::min(max_bytes_, staged_ready_bytes_) + sizeof(Trailer);
            check(hipHostMalloc(reinterpret_cast<void**>(&staged_send_host_), count, hipHostMallocCoherent), "coherent staged send");
            check(hipHostMalloc(reinterpret_cast<void**>(&staged_receive_host_), count, hipHostMallocCoherent), "coherent staged receive");
            check(hipHostMalloc(reinterpret_cast<void**>(&producer_flag_), sizeof(StagedReady), hipHostMallocCoherent), "producer readiness");
            check(hipHostMalloc(reinterpret_cast<void**>(&consumer_flag_), sizeof(StagedReady), hipHostMallocCoherent), "consumer readiness");
            new (producer_flag_) StagedReady{};
            new (consumer_flag_) StagedReady{};
            check(hipMalloc(reinterpret_cast<void**>(&staged_counters_), 2 * sizeof(unsigned)), "staged block counters");
            check(hipMemset(staged_counters_, 0, 2 * sizeof(unsigned)), "staged block counters initialization");
            // Device-memory hipMemset may return before completion. Producers
            // use a nonblocking caller stream, which need not wait on the null
            // stream. Finish this one-time initialization before exposing views.
            check(hipStreamSynchronize(nullptr), "staged block counters initialized");
            producer_flag_->block_counter = staged_counters_;
            consumer_flag_->block_counter = staged_counters_ + 1;
            check(hipHostGetDevicePointer(&staged_send_device_, staged_send_host_, 0), "staged send alias");
            check(hipHostGetDevicePointer(&staged_receive_device_, staged_receive_host_, 0), "staged receive alias");
            check(hipHostGetDevicePointer(reinterpret_cast<void**>(&producer_flag_device_), producer_flag_, 0), "producer readiness alias");
            check(hipHostGetDevicePointer(reinterpret_cast<void**>(&consumer_flag_device_), consumer_flag_, 0), "consumer readiness alias");
        } catch (...) {
            if (staged_counters_) (void)hipFree(staged_counters_);
            if (consumer_flag_) (void)hipHostFree(consumer_flag_);
            if (producer_flag_) (void)hipHostFree(producer_flag_);
            if (staged_receive_host_) (void)hipHostFree(staged_receive_host_);
            if (staged_send_host_) (void)hipHostFree(staged_send_host_);
            (void)hipHostFree(receive_host_);
            (void)hipHostFree(send_host_);
            throw;
        }
    }
    // default on (ds41/xfer t1: R6 -0.7 %, R1/N8 equal or better, exact); =0: inline CRCs
    crc_defer_ = !group_.minimax_compat;
    if (const char* v = std::getenv("STRIX_DIST_CRC_DEFER"); v && v[0] == '0' && v[1] == '\0') crc_defer_ = false;
    {  // default 1 MiB chunks (w5); =0: one piece
        const char* v = std::getenv("STRIX_DIST_HOST_CHUNK_BYTES");
        const auto chunk = v && *v ? static_cast<std::size_t>(std::stoull(v)) : (group_.minimax_compat ? std::size_t{0} : std::size_t{1048576});
        if (chunk && (chunk < 65536U || chunk > (16U << 20U) || chunk % 64U))
            throw std::invalid_argument("STRIX_DIST_HOST_CHUNK_BYTES: 0 or a multiple of 64 in [65536,16777216]");
        if (chunk && max_bytes_ > chunk) pipeline_ = std::make_unique<Pipeline>(max_bytes_, chunk, group_.minimax_compat);
    }
    if (const char* t = std::getenv("STRIX_DIST_TRACE"); t && *t) {
        const auto path = std::string(t) + ".r" + std::to_string(group_.rank);
        trace_ = std::fopen(path.c_str(), "w");
        if (!trace_) throw std::runtime_error("dist trace: cannot open " + path);
        std::fprintf(trace_, "# H seq bytes tag layer entry produced staged peer_staged received validated consumer (host path)\n"
                             "# G seq bytes tag layer rings entry produced tx_submitted rx_done tx_done consumer (GPU-direct; 0: not seen)\n"
                             "# wall-clock ns; produced: the CPU saw the producer done; consumer: the received bytes are in device\n"
                             "# memory for the next kernel (host: H2D copy end, GPU-direct: gather end, both from GPU events)\n");
    }
}
Communicator::Communicator(Group group, std::unique_ptr<Transport> transport, std::unique_ptr<DeviceExchange> direct, std::size_t max_bytes)
    : Communicator(group, std::move(transport), max_bytes) {
    direct_ = std::move(direct);
    if (direct_) configure_routes();
    if (direct_ && trace_) direct_->set_trace(trace_);
}

void Communicator::resize_bytes(std::size_t bytes, bool grow_pipeline) {
    if (bytes == max_bytes_) return;
    if (bytes < staged_ready_bytes_) throw std::invalid_argument("dist resize below staged readiness capacity");
    if (started_ || direct_started_ || staged_prepared_ || staged_receive_held_)
        throw std::logic_error("dist reserve while an exchange is in flight");
    drain_tail();
    check(hipDeviceSynchronize(), "reserve sync");
    const auto old_bytes = max_bytes_;
    const auto chunk = pipeline_ ? pipeline_->chunk : 0;
    const bool legacy = pipeline_ && pipeline_->legacy;
    if (grow_pipeline) pipeline_.reset();
    check(hipHostFree(send_host_), "release send capacity");
    send_host_ = nullptr;
    check(hipHostFree(receive_host_), "release receive capacity");
    receive_host_ = nullptr;
    send_device_alias_ = nullptr;
    const unsigned flags = fused_ ? hipHostMallocCoherent : hipHostMallocDefault;
    check(hipHostMalloc(reinterpret_cast<void**>(&send_host_), bytes + sizeof(Trailer), flags), "grow pinned send buffer");
    check(hipHostMalloc(reinterpret_cast<void**>(&receive_host_), bytes + sizeof(Trailer), flags), "grow pinned receive buffer");
    if (d2h_kernel_bytes_) check(hipHostGetDevicePointer(&send_device_alias_, send_host_, 0), "grow send alias");
    if (chunk && grow_pipeline) pipeline_ = std::make_unique<Pipeline>(bytes, chunk, legacy);
    max_bytes_ = bytes;
    for (auto& r : routes_) if (!r.direct && r.up_to == old_bytes) r.up_to = bytes;
}

const Route& Communicator::route(std::size_t bytes) const {
    for (const auto& r : routes_)
        if (bytes <= r.up_to) return r;
    return routes_.back();
}
std::string Communicator::route_text() const {
    std::ostringstream out;
    for (std::size_t i = 0; i < routes_.size(); ++i) {
        const auto& r = routes_[i];
        out << (i ? "," : "") << (r.up_to == SIZE_MAX ? std::string("inf") : std::to_string(r.up_to)) << ':'
            << (!r.direct ? "host" : r.host_percent ? "mix" : "gpu");
        if (r.host_percent) out << ':' << r.rings << ':' << r.host_percent;
        else if (r.direct && r.rings) out << ':' << r.rings;
    }
    return out.str();
}

namespace {
std::vector<Route> parse_routes(const std::string& text) {
    std::vector<Route> out;
    std::stringstream in(text);
    for (std::string item; std::getline(in, item, ',');) {
        std::stringstream f(item);
        std::string up, path, rings, percent;
        std::getline(f, up, ':');
        std::getline(f, path, ':');
        std::getline(f, rings, ':');
        std::getline(f, percent, ':');
        Route r;
        r.up_to = up == "inf" ? SIZE_MAX : static_cast<std::size_t>(std::stoull(up));
        if (path != "host" && path != "gpu" && path != "mix") throw std::invalid_argument("STRIX_DIST_ROUTE: path must be host, gpu or mix: " + item);
        r.direct = path != "host";
        r.rings = rings.empty() ? 0U : static_cast<unsigned>(std::stoul(rings));
        if (path == "mix") {
            r.host_percent = percent.empty() ? 0U : static_cast<unsigned>(std::stoul(percent));
            if (r.host_percent == 0U || r.host_percent > 99U) throw std::invalid_argument("STRIX_DIST_ROUTE: mix needs RINGS:HOST_PERCENT (1..99): " + item);
        } else if (!percent.empty()) throw std::invalid_argument("STRIX_DIST_ROUTE: only mix takes a host share: " + item);
        if (!out.empty() && r.up_to <= out.back().up_to) throw std::invalid_argument("STRIX_DIST_ROUTE: sizes must increase: " + text);
        out.push_back(r);
    }
    if (out.empty()) throw std::invalid_argument("STRIX_DIST_ROUTE: empty table");
    out.back().up_to = SIZE_MAX;
    return out;
}
} // namespace

// Measures every candidate (host path, direct on 1..rings rings) at a grid of sizes, both ranks in the same order;
// rank 0 picks the fastest p50 per size and both apply its table.
void Communicator::configure_routes() {
    const char* env = std::getenv("STRIX_DIST_ROUTE");
    const std::string mode = env && *env ? env : "table";
    const bool host_allowed = transport_->name().rfind("tcp", 0) != 0;
    if (mode == "table") {
        // The measured defaults (2026-10-21, poll kernel). With host streams (mixed layout, e.g. data0 host + data1
        // gpu_direct): small messages on the host path, two rings up to 1 MB, above that 35 % over the host path and
        // the rest over the rings at once. Without (every stream gpu_direct): one ring up to 32 KB, two up to 256 KB,
        // every ring above.
        const auto all = std::to_string(direct_->rings());
        if (host_allowed && max_bytes_ >= 65536U)
            routes_ = parse_routes("65536:host,1048576:gpu:2,inf:mix:" + all + ":35");
        else
            routes_ = parse_routes("32768:gpu:1,262144:gpu:2,inf:gpu:" + all);
        return;
    }
    if (mode == "gpu") { routes_ = {Route{SIZE_MAX, true, 0}}; return; }
    if (mode == "host") {
        if (!host_allowed) throw std::invalid_argument("STRIX_DIST_ROUTE=host: the transport is only a control channel here");
        routes_ = {Route{max_bytes_, false, 0}, Route{SIZE_MAX, true, 0}};
        return;
    }
    if (mode != "auto") {
        set_routes(mode);
        return;
    }
    routes_ = {Route{SIZE_MAX, true, 0}};
    std::vector<std::size_t> grid;
    for (std::size_t g : {std::size_t{64}, std::size_t{1024}, std::size_t{4096}, std::size_t{20480}, std::size_t{65536}, std::size_t{163840},
                          std::size_t{524288}, std::size_t{2} << 20U, std::size_t{8} << 20U})
        grid.push_back(g);
    const std::size_t largest = grid.back();
    hipStream_t s{};
    void *local = nullptr, *remote = nullptr;
    check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "route calibration stream");
    check(hipMalloc(&local, largest), "route calibration buffer");
    check(hipMalloc(&remote, largest), "route calibration buffer");
    check(hipMemsetAsync(local, 0x5A, largest, s), "route calibration buffer");
    struct Candidate { bool direct; unsigned rings; };
    std::vector<Candidate> candidates;
    for (unsigned r = 1; r <= direct_->rings(); ++r) candidates.push_back({true, r});
    std::ostringstream table;
    std::uint32_t round = 0;
    for (std::size_t i = 0; i < grid.size(); ++i) {
        const auto g = grid[i];
        const unsigned iterations = g <= 65536U ? 40U : g <= (1U << 20U) ? 12U : 4U;
        double best = 1e30;
        Candidate pick{true, 0};
        auto all = candidates;
        if (host_allowed && g <= max_bytes_) all.push_back({false, 0});
        for (const auto& c : all) {
            std::vector<double> t;
            for (unsigned k = 0; k < iterations + 2U; ++k) {
                check(hipStreamSynchronize(s), "route calibration");
                const auto t0 = std::chrono::steady_clock::now();
                const Tag tag{static_cast<std::uint32_t>(TagKind::control), 0xCA11U, round++};
                if (c.direct) {
                    direct_->use_rings(c.rings);
                    // through the router-free path: one direct exchange (chunked above the capacity)
                    routes_ = {Route{SIZE_MAX, true, c.rings}};
                } else {
                    routes_ = {Route{SIZE_MAX, false, 0}};
                }
                exchange(s, local, remote, g, tag);
                check(hipStreamSynchronize(s), "route calibration");
                if (k >= 2U) t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(t.begin(), t.end());
            const double p50 = t[t.size() / 2U];
            if (p50 < best) { best = p50; pick = c; }
        }
        table << (i + 1U == grid.size() ? std::string("inf") : std::to_string(g)) << ':' << (pick.direct ? "gpu" : "host");
        if (pick.direct) table << ':' << pick.rings;
        if (i + 1U < grid.size()) table << ',';
    }
    check(hipStreamSynchronize(s), "route calibration");
    (void)hipFree(local);
    (void)hipFree(remote);
    (void)hipStreamDestroy(s);
    // Rank 0's table, as a fixed-size message (both ranks send the same size).
    std::vector<std::byte> out(1024), in(1024);
    const auto mine = table.str();
    if (mine.size() >= out.size()) throw std::logic_error("route table too long");
    std::memcpy(out.data(), mine.data(), mine.size());
    transport_->exchange(out, in);
    const auto* peer = reinterpret_cast<const char*>(in.data());
    routes_ = parse_routes(group_.rank == 0U ? mine : std::string(peer, strnlen(peer, in.size())));
    for (const auto& r : routes_)
        if (!r.direct && r.up_to > max_bytes_) throw std::runtime_error("route table sends " + std::to_string(r.up_to) + " B over the host path (max " +
                                                                         std::to_string(max_bytes_) + ")");
    stats_ = CommunicatorStats{};
}
void Communicator::set_routes(const std::string& table) {
    if (!direct_) throw std::logic_error("dist routes: no device exchange");
    if (started_ || direct_started_ || staged_ || staged_prepared_ || staged_receive_held_)
        throw std::logic_error("dist routes: a staged view or exchange is in flight");
    auto routes = parse_routes(table);
    const bool host_allowed = transport_->name().rfind("tcp", 0) != 0;
    for (const auto& r : routes)
        if ((!r.direct || r.host_percent) && (!host_allowed || (!r.direct && r.up_to > max_bytes_)))
            throw std::invalid_argument("STRIX_DIST_ROUTE: host path not available for " + table);
    routes_ = std::move(routes);
}

void Communicator::check_exchanges() const {
    if (direct_)
        if (const auto f = direct_->failure(); !f.empty()) throw std::runtime_error("dist exchange: gpu-direct: " + f);
}
std::string Communicator::transport_name() const {
    return direct_ ? direct_->name() + " + control " + transport_->name() + "; routes " + route_text() : transport_->name();
}

void Communicator::cancel_pending(const std::string& reason) noexcept {
    if (terminal_.empty()) terminal_ = reason;
    cancel_.store(true, std::memory_order_release);
    if (pipeline_) pipeline_->fail();
    transport_->abort(reason);
    // Worker destruction joins even after a timeout/error. No host job may submit
    // another HIP operation against caller-owned events after this returns.
    progress_.reset();
    if (pipeline_) {
        pipeline_->crc_send.reset();
        pipeline_->crc_receive.reset();
    }
    started_ = pipeline_started_ = false;
}

Communicator::~Communicator() {
    if (trace_pending_) try { flush_trace(); } catch (...) {}
    direct_.reset(); // waits for its in-flight exchange and releases its streams
    if (started_) {
        cancel_.store(true, std::memory_order_release); // the job's own waits stop
        if (pipeline_started_) {
            pipeline_->fail();
            transport_->abort("dist pipeline: destroyed with an exchange in flight");
        }
        try {
            if (!progress_->worker.wait_until(std::chrono::steady_clock::now() + wait_timeout_)) {
                transport_->abort("dist exchange: destroyed with an exchange in flight");
                (void)progress_->worker.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(5));
            }
        } catch (...) {}
    }
    // Stage jobs use pipeline_ through this communicator, including their failure
    // handler. Join them while the owning pointer is still non-null: unique_ptr
    // reset clears it BEFORE entering Pipeline's destructor.
    if (pipeline_) {
        pipeline_->fail();
    }
    progress_.reset(); // no progress job may read the CRC-worker pointers after this
    if (rx_pending_sequence_ && terminal_.empty())  // a collective sequence should end with an ordinary exchange
        std::fprintf(stderr, "dist exchange: destroyed before the deferred CRC of exchange %llu was checked\n",
                     static_cast<unsigned long long>(rx_pending_sequence_ - 1U));
    if (pipeline_) {
        pipeline_->crc_send.reset();
        pipeline_->crc_receive.reset();
    }
    pipeline_.reset();
    // Teardown only: a forgotten release, a prepared-but-not-started producer or
    // a host timeout cannot free storage still referenced by this stream.
    if (staged_exposed_) (void)hipStreamSynchronize(staged_stream_);
    if (send_ready_) {
        (void)hipEventSynchronize(send_ready_);
        (void)hipEventDestroy(send_ready_);
    }
    if (receive_done_) {
        (void)hipEventSynchronize(receive_done_);
        (void)hipEventDestroy(receive_done_);
    }
    if (consumer_flag_) (void)hipHostFree(consumer_flag_);
    if (staged_counters_) (void)hipFree(staged_counters_);
    if (producer_flag_) (void)hipHostFree(producer_flag_);
    if (staged_receive_host_) (void)hipHostFree(staged_receive_host_);
    if (staged_send_host_) (void)hipHostFree(staged_send_host_);
    (void)hipHostFree(send_host_);
    (void)hipHostFree(receive_host_);
    if (pre_copy_) (void)hipEventDestroy(pre_copy_);
    if (h2d_done_) (void)hipEventDestroy(h2d_done_);
    if (trace_) std::fclose(trace_);
}

void Communicator::quiesce(hipStream_t stream) {
    if (started_ || direct_started_ || staged_ || staged_prepared_ || staged_receive_held_)
        throw std::logic_error("dist exchange: quiesce() while a staged view or exchange is in flight");
    if (direct_) direct_->quiesce(stream);
    if (consumer_pending_token_) {
        try { wait_staged_flag(consumer_flag_, consumer_pending_token_, std::chrono::steady_clock::now() + wait_timeout_); }
        catch (const std::exception& e) { stop(e.what()); }
        consumer_pending_token_ = 0;
    }
}

namespace {
void reset_staged_flag(StagedReady* flag, std::uint64_t token) {
    static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);
    std::atomic_ref<std::uint64_t>(flag->done).store(0, std::memory_order_relaxed);
    std::atomic_ref<std::uint64_t>(flag->ready).store(token, std::memory_order_release);
}
}

void Communicator::wait_staged_flag(StagedReady* flag, std::uint64_t token, std::chrono::steady_clock::time_point deadline) {
    for (unsigned i = 0;; ++i) {
        const auto done = std::atomic_ref<std::uint64_t>(flag->done).load(std::memory_order_acquire);
        if (done == token) return;
        if (done != 0) throw std::runtime_error("staged readiness: wrong generation, expected " + std::to_string(token) + " got " + std::to_string(done));
        if ((i & 1023U) == 0U && (cancel_.load(std::memory_order_acquire) || std::chrono::steady_clock::now() >= deadline))
            throw std::runtime_error("staged readiness: producer/consumer did not release generation " + std::to_string(token) + " before its deadline");
        cpu_relax();
    }
}

ProducerStage Communicator::prepare_staged_ready(hipStream_t stream, std::size_t bytes) {
    if (!staged_ready_bytes_ || !bytes || bytes > max_bytes_ || bytes > staged_ready_bytes_ || trace_ ||
        (direct_ && route(bytes).direct)) return {};
    if (!terminal_.empty()) throw std::runtime_error("staged readiness: communicator stopped: " + terminal_);
    if (started_ || direct_started_ || staged_ || staged_prepared_ || staged_receive_held_)
        throw std::logic_error("staged readiness: previous view or exchange not released");
    if (staged_exposed_ && stream != staged_stream_) throw std::logic_error("staged readiness: one stream per communicator");
    if (sequence_ == UINT64_MAX) throw std::overflow_error("staged readiness: generation exhausted");
    staged_stream_ = stream;
    staged_exposed_ = staged_prepared_ = true;
    staged_prepared_bytes_ = bytes;
    staged_token_ = sequence_ + 1U;
    reset_staged_flag(producer_flag_, staged_token_);
    return ProducerStage{staged_send_device_, producer_flag_device_, staged_token_, staged_counters_};
}

void Communicator::start_staged_ready(hipStream_t stream, std::size_t bytes, const Tag& tag) {
    if (!terminal_.empty()) throw std::runtime_error("staged readiness: communicator stopped: " + terminal_);
    if (!staged_prepared_ || started_ || direct_started_ || bytes != staged_prepared_bytes_ || stream != staged_stream_)
        throw std::logic_error("staged readiness: start without the matching prepared producer");
    tag_ = tag;
    if (consumer_pending_token_) {
        try { wait_staged_flag(consumer_flag_, consumer_pending_token_, std::chrono::steady_clock::now() + wait_timeout_); }
        catch (const std::exception& e) { stop(e.what()); }
        consumer_pending_token_ = 0;
    }
    std::swap(send_host_, staged_send_host_);
    std::swap(receive_host_, staged_receive_host_);
    staged_prepared_ = false;
    staged_started_ = true;
    // start_host records only CPU state on this path; progress waits on producer_flag_.
    start_host(stream, nullptr, bytes);
}

ConsumerStage Communicator::finish_staged_ready() {
    if (!staged_started_ || !started_) throw std::logic_error("staged readiness: finish without start");
    finish_host(staged_stream_, nullptr); // validates the exact same in-band trailer and CRC
    std::swap(send_host_, staged_send_host_);
    std::swap(receive_host_, staged_receive_host_);
    staged_started_ = false;
    staged_receive_held_ = true;
    reset_staged_flag(consumer_flag_, staged_token_); // CPU payload writes happen before GPU's acquire
    latency::mark("consumer_view_ready");
    return ConsumerStage{staged_receive_device_, consumer_flag_device_, staged_token_, staged_counters_ + 1};
}

void Communicator::release_staged_ready(hipStream_t stream) {
    if (!staged_receive_held_ || started_ || stream != staged_stream_)
        throw std::logic_error("staged readiness: release without the matching consumer");
    consumer_pending_token_ = staged_token_;
    staged_receive_held_ = false;
}

void* Communicator::send_staging(std::size_t bytes) const {
    if (!fused_ || !terminal_.empty() || started_ || direct_started_ || staged_ || staged_prepared_ || staged_receive_held_ || bytes == 0U || bytes > max_bytes_) return nullptr;
    if (direct_ && route(bytes).direct) return nullptr;
    void* device = nullptr;
    if (hipHostGetDevicePointer(&device, send_host_, 0) != hipSuccess) return nullptr;
    return device;
}
void Communicator::start_staged(hipStream_t stream, std::size_t bytes, const Tag& tag) {
    if (!terminal_.empty()) throw std::runtime_error("dist exchange: the communicator stopped earlier: " + terminal_);
    if (send_staging(bytes) == nullptr) throw std::logic_error("dist exchange: start_staged() without a host-path staging buffer");
    if (trace_pending_) flush_trace();
    tag_ = tag;
    start_host(stream, nullptr, bytes);
    staged_ = true;
}
const void* Communicator::finish_staged(hipStream_t stream) {
    if (!staged_ || !started_) throw std::logic_error("dist exchange: finish_staged() without start_staged()");
    finish_host(stream, nullptr);
    void* device = nullptr;
    check(hipHostGetDevicePointer(&device, receive_host_, 0), "receive staging");
    return device;
}
void Communicator::release_staged(hipStream_t stream) {
    if (!staged_) throw std::logic_error("dist exchange: release_staged() without start_staged()");
    staged_ = false;
    // the consumers enqueued so far read the receive buffer: it is reused only after them
    if (!receive_done_) check(hipEventCreateWithFlags(&receive_done_, hipEventDisableTiming), "receive event");
    check(hipEventRecord(receive_done_, stream), "receive event");
    receive_pending_ = true;
}

void Communicator::exchange(hipStream_t stream, const void* local, void* remote, std::size_t bytes, const Tag& tag) {
    start(stream, local, bytes, tag);
    finish(stream, remote);
}

void Communicator::start(hipStream_t stream, const void* local, std::size_t bytes, const Tag& tag) {
    if (!terminal_.empty()) throw std::runtime_error("dist exchange: the communicator stopped earlier: " + terminal_);
    if (started_ || direct_started_ || staged_ || staged_prepared_ || staged_receive_held_)
        throw std::logic_error("dist exchange: start() while a staged view or exchange is in flight");
    if (trace_pending_) flush_trace();
    if (direct_ && bytes > 0U && route(bytes).direct) {
        // Device to device. The kernel checks each operation's framing (epoch, sequence, size, slot); the identity
        // of the collective (plan, request, tag, layer) and a checksum of the payload travel in-band and are checked
        // by the receiver's GPU. Messages above the capacity are exchanged in chunks in finish().
        if (!local) throw std::invalid_argument("dist exchange size or buffers");
        const Route& r = route(bytes);
        stats_.last_entry = std::chrono::steady_clock::now();
        tag_ = tag;
        // A mixed route: the head goes over the host path (its own trailer, sequence and CRC), at the same time.
        const std::size_t host = r.host_percent ? std::min(max_bytes_, bytes / 100U * r.host_percent + bytes % 100U * r.host_percent / 100U) & ~std::size_t{63} : 0U;
        direct_offset_ = host;
        direct_bytes_ = bytes - host;
        direct_local_ = static_cast<const std::byte*>(local) + host;
        mix_rings_ = r.rings;
        if (host) start_host(stream, local, host);
        if (direct_bytes_ <= direct_->capacity()) {
            direct_->use_rings(r.rings);
            direct_->expect_next(direct_bytes_, r.rings); // the next message on the rings is likely this size again
            direct_->start(stream, direct_local_, direct_bytes_, ExchangeIdentity{group_.plan, tag.request, tag.kind, tag.layer});
        }
        direct_started_ = true;
        return;
    }
    if (bytes == 0U || bytes > max_bytes_ || !local) throw std::invalid_argument("dist exchange size or buffers");
    tag_ = tag;
    if (pipeline_ && bytes > pipeline_->chunk && bytes <= pipeline_->capacity) start_pipeline(stream, local, bytes);
    else start_host(stream, local, bytes);
}

bool Communicator::start_async(hipStream_t stream, const void* local, void* remote, std::size_t bytes, const Tag& tag,
                               hipEvent_t destination_last_use, hipEvent_t source_free) {
    if (!async_start_ || !pipeline_ || bytes <= pipeline_->chunk || bytes > pipeline_->capacity || (direct_ && route(bytes).direct)) {
        start(stream, local, bytes, tag);
        return false;
    }
    if (!terminal_.empty()) throw std::runtime_error("dist exchange: the communicator stopped earlier: " + terminal_);
    if (started_ || direct_started_ || staged_ || staged_prepared_ || staged_receive_held_)
        throw std::logic_error("dist exchange: start() while a staged view or exchange is in flight");
    if (!bytes || bytes > max_bytes_ || !local || !remote || !destination_last_use || !source_free)
        throw std::invalid_argument("dist async exchange size, buffers or events");
    if (trace_pending_) flush_trace();
    tag_ = tag;
    start_pipeline(stream, local, bytes, remote, destination_last_use, source_free);
    return true;
}

void Communicator::start_pipeline(hipStream_t stream, const void* local, std::size_t bytes, void* remote,
                                  hipEvent_t destination_last_use, hipEvent_t source_free) {
    latency::context().sequence = sequence_;
    latency::mark("pipeline_entry", bytes);
    auto& p = *pipeline_;
    const auto entry = std::chrono::steady_clock::now();
    stats_.last_entry = entry;
    current_ = History{sequence_, bytes, tag_.kind, tag_.layer, tag_.request, 0, 0, 'P'};
    set_phase(Phase::reuse_wait);
    const bool defer_reuse = remote && receive_pending_;
    p.reuse_wait = 0;
    if (receive_pending_ && !remote) {
        const auto reuse_begin = std::chrono::steady_clock::now();
        const auto deadline = entry + wait_timeout_;
        for (;;) {
            const auto e = hipEventQuery(receive_done_);
            if (e == hipSuccess) break;
            if (e != hipErrorNotReady) check(e, "pipeline previous receive");
            if (std::chrono::steady_clock::now() >= deadline) stop("pipeline receive-buffer reuse timeout");
            cpu_relax();
        }
        p.reuse_wait = std::chrono::duration<double>(std::chrono::steady_clock::now() - reuse_begin).count();
    }
    receive_pending_ = false;
    p.async = remote != nullptr;
    p.destination.store(nullptr, std::memory_order_relaxed);
    if (p.async) {
        // Queue the actual last-use dependency before publishing the destination.
        // RX remains ordered after the preceding message's H2D on the same stream.
        check(hipStreamWaitEvent(p.rx, destination_last_use, 0), "pipeline destination lifetime");
        p.destination.store(remote, std::memory_order_release);
    }
    p.stage = p.wire = p.check_seconds = 0;
    p.failure = nullptr;
    p.failed.store(false, std::memory_order_relaxed);
    p.prepared.store(0, std::memory_order_relaxed); p.received.store(0, std::memory_order_relaxed);
    bytes_ = bytes;
    started_at_ = entry;
    if (!send_ready_) check(hipEventCreateWithFlags(&send_ready_, hipEventDisableTiming), "pipeline send event");
    if (!receive_done_) check(hipEventCreateWithFlags(&receive_done_, hipEventDisableTiming), "pipeline receive event");
    check(hipEventRecord(p.producer, stream), "pipeline producer");
    check(hipStreamWaitEvent(p.tx, p.producer, 0), "pipeline producer dependency");
    for (std::size_t at = 0, k = 0; at < bytes; at += p.chunk, ++k) {
        const auto n = std::min(p.chunk, bytes - at);
        const auto slot = at + k * sizeof(Trailer);
        check(hipMemcpyAsync(p.send + slot, static_cast<const std::byte*>(local) + at, n, hipMemcpyDeviceToHost, p.tx), "pipeline D2H");
        check(hipEventRecord(p.ready[k], p.tx), "pipeline chunk ready");
    }
    check(hipEventRecord(send_ready_, p.tx), "pipeline all D2H");
    // Ordinary start preserves its source lifetime contract on the caller stream.
    // Async callers instead fence the next overwrite with their source_free event.
    if (p.async) check(hipEventRecord(source_free, p.tx), "pipeline source free");
    else check(hipStreamWaitEvent(stream, send_ready_, 0), "pipeline source lifetime");
    started_ = pipeline_started_ = true;
    set_phase(Phase::producer);
    const auto prepare = [this](std::size_t at, std::size_t k) {
        auto& q = *pipeline_;
        if (q.failed.load(std::memory_order_acquire)) throw std::runtime_error("dist pipeline: a stage failed");
        const auto n = std::min(q.chunk, bytes_ - at), slot = at + k * sizeof(Trailer);
        const auto deadline = started_at_ + wait_timeout_;
        latency::mark("chunk_d2h_wait", at);
        const auto t = std::chrono::steady_clock::now();
        for (unsigned spin = 0;; ++spin) {
            const auto e = hipEventQuery(q.ready[k]);
            if (e == hipSuccess) break;
            if (e != hipErrorNotReady) check(e, "pipeline event");
            if ((spin & 1023U) == 0U && (q.failed.load(std::memory_order_acquire) || cancel_.load(std::memory_order_acquire) || std::chrono::steady_clock::now() >= deadline))
                throw std::runtime_error("dist pipeline producer timeout");
            cpu_relax();
        }
        latency::mark("chunk_crc_tx_begin", at);
        Trailer sent;
        sent.epoch = group_.epoch; sent.plan = group_.plan; sent.sequence = sequence_; sent.request = tag_.request;
        sent.bytes = n; sent.rank = group_.rank; sent.tag = tag_.kind; sent.layer = tag_.layer;
        sent.reserved[0] = 1U; sent.reserved[1] = at; sent.reserved[2] = bytes_; sent.reserved[3] = q.chunk;
        sent.crc32c = crc32c({q.send + slot, n});
        std::memcpy(q.send + slot + n, &sent, sizeof(sent));
        q.stage += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        latency::mark("chunk_crc_tx_end", at);
    };
    const auto receive = [this](std::size_t at, std::size_t k, void*& destination) {
        auto& q = *pipeline_;
        const auto n = std::min(q.chunk, bytes_ - at), slot = at + k * sizeof(Trailer);
        latency::mark("chunk_crc_rx_begin", at);
        const auto t = std::chrono::steady_clock::now();
        Trailer peer, sent;
        std::memcpy(&sent, q.send + slot + n, sizeof(sent));
        std::memcpy(&peer, q.receive + slot + n, sizeof(peer));
        if (peer.magic != trailer_magic || peer.epoch != sent.epoch || peer.plan != sent.plan || peer.sequence != sent.sequence ||
            peer.request != sent.request || peer.bytes != n || peer.rank != 1U - group_.rank || peer.tag != sent.tag || peer.layer != sent.layer ||
            !std::equal(std::begin(peer.reserved), std::end(peer.reserved), std::begin(sent.reserved)))
            throw std::runtime_error("dist pipeline: peer chunk identity/offset/geometry mismatch");
        if (peer.crc32c != crc32c({q.receive + slot, n})) throw std::runtime_error("dist pipeline: payload CRC mismatch");
        q.check_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        latency::mark("chunk_validated", at);
        if (!destination) {
            const auto deadline = started_at_ + wait_timeout_;
            for (unsigned spin = 0; !(destination = q.destination.load(std::memory_order_acquire)); ++spin) {
                if ((spin & 1023U) == 0U && (q.failed.load(std::memory_order_acquire) || cancel_.load(std::memory_order_acquire) || std::chrono::steady_clock::now() >= deadline))
                    throw std::runtime_error("dist pipeline: finish did not provide a destination before timeout");
                cpu_relax();
            }
            if (!q.async) check(hipStreamWaitEvent(q.rx, q.consumer, 0), "pipeline destination lifetime");
        }
        check(hipMemcpyAsync(static_cast<std::byte*>(destination) + at, q.receive + slot, n, hipMemcpyHostToDevice, q.rx), "pipeline H2D");
        latency::mark("chunk_h2d_enqueued", at);
    };
    const auto guarded = [this](auto job) {
        return [this, job] {
            latency::context().sequence = sequence_;
            try { job(); }
            catch (...) {
                pipeline_->fail(std::current_exception());
                transport_->abort("dist pipeline: a stage failed");
                throw;
            }
        };
    };
    if (p.crc_send) {
        p.crc_send->post(guarded([this, prepare] {
            auto& q = *pipeline_;
            for (std::size_t at = 0, k = 0; at < bytes_; at += q.chunk, ++k) {
                if (q.failed.load(std::memory_order_acquire)) throw std::runtime_error("dist pipeline: a stage failed");
                prepare(at, k);
                q.publish(q.prepared, k + 1U);
            }
        }));
        p.crc_receive->post(guarded([this, receive] {
            auto& q = *pipeline_;
            void* destination = nullptr;
            for (std::size_t at = 0, k = 0; at < bytes_; at += q.chunk, ++k) {
                q.wait_count(q.received, k + 1U);
                receive(at, k, destination);
            }
            check(hipEventRecord(receive_done_, q.rx), "pipeline receive complete");
        }));
    }
    progress_->worker.post(guarded([this, prepare, receive, defer_reuse] {
        auto& q = *pipeline_;
        // finish joined the previous workers, but its H2D may still read pinned
        // receive staging. Guard reuse here, not on the submitting host thread.
        // This query precedes this job's new record of receive_done_.
        if (defer_reuse) {
            const auto reuse_begin = std::chrono::steady_clock::now();
            const auto deadline = started_at_ + wait_timeout_;
            for (unsigned spin = 0;; ++spin) {
                const auto e = hipEventQuery(receive_done_);
                if (e == hipSuccess) break;
                if (e != hipErrorNotReady) check(e, "pipeline previous receive");
                if ((spin & 1023U) == 0U && (q.failed.load(std::memory_order_acquire) || cancel_.load(std::memory_order_acquire) ||
                                           std::chrono::steady_clock::now() >= deadline))
                    throw std::runtime_error("pipeline receive-buffer reuse timeout");
                cpu_relax();
            }
            q.reuse_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - reuse_begin).count();
        }
        void* destination = nullptr;
        for (std::size_t at = 0, k = 0; at < bytes_; at += q.chunk, ++k) {
            const auto n = std::min(q.chunk, bytes_ - at), slot = at + k * sizeof(Trailer);
            if (q.crc_send) q.wait_count(q.prepared, k + 1U);
            else prepare(at, k);
            latency::mark("chunk_transport_begin", at);
            set_phase(Phase::transfer);
            const auto t = std::chrono::steady_clock::now();
            transport_->exchange({q.send + slot, n + sizeof(Trailer)}, {q.receive + slot, n + sizeof(Trailer)});
            q.wire += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
            latency::mark("chunk_transport_end", at);
            if (q.crc_receive) {
                q.publish(q.received, k + 1U);
            } else receive(at, k, destination);
        }
        if (!q.crc_receive) check(hipEventRecord(receive_done_, q.rx), "pipeline receive complete");
    }));
    returned_at_ = std::chrono::steady_clock::now();
}

void Communicator::finish_pipeline(hipStream_t stream, void* remote) {
    latency::mark("pipeline_finish_begin");
    const auto entry = std::chrono::steady_clock::now();
    auto& p = *pipeline_;
    if (!p.async) {
        check(hipEventRecord(p.consumer, stream), "pipeline consumer ready");
        p.destination.store(remote, std::memory_order_release);
    } else if (remote != p.destination.load(std::memory_order_acquire)) {
        throw std::logic_error("dist pipeline: finish destination differs from start");
    }
    const auto deadline = entry + 2 * wait_timeout_ + std::chrono::seconds(5);
    std::exception_ptr error;
    const auto join = [&](Worker& worker) {
        try {
            if (!worker.wait_until(deadline)) {
                p.fail(std::make_exception_ptr(std::runtime_error("pipeline progress timeout")));
                transport_->abort("pipeline progress timeout");
                stop("pipeline progress timeout");
            }
        } catch (...) { if (!error) error = std::current_exception(); }
    };
    join(progress_->worker);
    if (p.crc_send) { join(*p.crc_send); join(*p.crc_receive); }
    { std::lock_guard lock(p.failure_mutex); if (p.failure) error = p.failure; }
    if (error) {
        try { std::rethrow_exception(error); }
        catch (const std::exception& e) { stop(e.what()); }
    }
    check(hipStreamWaitEvent(stream, receive_done_, 0), "pipeline consumer dependency");
    receive_pending_ = true;
    started_ = pipeline_started_ = false;
    set_phase(Phase::idle);
    history_[history_next_++ % history_.size()] = current_;
    ++sequence_;
    ++stats_.exchanges;
    stats_.bytes_sent += bytes_;
    const auto end = std::chrono::steady_clock::now();
    stats_.pipeline_reuse_wait += p.reuse_wait; // worker joined above: no concurrent stats access
    stats_.stage += p.stage;
    stats_.wire += p.wire;
    stats_.check += p.check_seconds;
    stats_.overlap += std::chrono::duration<double>(entry - returned_at_).count();
    stats_.finish_wait += std::chrono::duration<double>(end - entry).count();
    stats_.seconds += std::chrono::duration<double>(end - started_at_).count();
    latency::mark("pipeline_finish_end");
}

void Communicator::start_host(hipStream_t stream, const void* local, std::size_t bytes) {
    drain_tail();
    deferring_ = crc_defer_ && staged_started_ && !check_on_caller();
    latency::context().sequence = sequence_;
    latency::mark("host_entry", bytes);
    const auto entry = std::chrono::steady_clock::now();
    stats_.last_entry = entry;
    // The collective is recorded before any wait (dumped if a wait expires).
    current_ = History{sequence_, bytes, tag_.kind, tag_.layer, tag_.request, 0, 0, 'H'};
    set_phase(Phase::reuse_wait);
    // The receive buffer is reused only after the CPU has seen the previous copy out of it complete.
    if (!staged_started_ && receive_pending_) {
        const auto reuse_begin = std::chrono::steady_clock::now();
        const auto deadline = entry + wait_timeout_;
        for (hipError_t ready; (ready = hipEventQuery(receive_done_)) != hipSuccess;) {
            if (ready != hipErrorNotReady) check(ready, "previous receive copy");
            if (std::chrono::steady_clock::now() >= deadline)
                stop("the previous exchange's copy out of the receive buffer did not complete within " + std::to_string(wait_timeout_.count()) + " ms");
            cpu_relax();
        }
        stats_.host_reuse_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - reuse_begin).count();
    }
    if (!staged_started_) receive_pending_ = false;
    // The copy is ordered after the producer of `local` on the stream; the progress thread waits for it on
    // the CPU before computing the CRC and sending.
    if (trace_) {
        t_entry_ = wall_ns();
        if (!pre_copy_) check(hipEventCreate(&pre_copy_), "trace event");
        check(hipEventRecord(pre_copy_, stream), "trace event");
    }
    if (!staged_started_) {
        latency::mark("d2h_enqueue_begin");
        if (local) {
            if (d2h_kernel_bytes_ && bytes <= d2h_kernel_bytes_)
                enqueue_host_d2h_copy(send_device_alias_, local, bytes, stream);
            else
                check(hipMemcpyAsync(send_host_, local, bytes, hipMemcpyDeviceToHost, stream), "device to host");
        }
        latency::mark("d2h_copy_submitted");
        if (!send_ready_) check(hipEventCreateWithFlags(&send_ready_, hipEventDisableTiming), "send event");
        check(hipEventRecord(send_ready_, stream), "send event");
        latency::mark("d2h_enqueued");
    }
    bytes_ = bytes;
    started_ = true;
    started_at_ = entry;
    set_phase(Phase::producer);
    progress_->worker.post([this] {
        // Spin on the copy's event: a blocking synchronize may sleep and add its wake-up to every exchange. Bounded:
        // a producer that never completes stops the communicator (and wakes the peer) instead of hanging both ranks.
        const auto deadline = started_at_ + wait_timeout_;
        const auto bounded = [&](hipEvent_t event, const char* what) {
            for (unsigned i = 0;; ++i) {
                const hipError_t ready = hipEventQuery(event);
                if (ready == hipSuccess) return;
                if (ready != hipErrorNotReady) check(ready, what);
                if ((i & 1023U) == 0U && (cancel_.load(std::memory_order_acquire) || std::chrono::steady_clock::now() >= deadline))
                    throw std::runtime_error(std::string("dist exchange: ") + what + " did not complete within " + std::to_string(wait_timeout_.count()) + " ms");
                cpu_relax();
            }
        };
        if (trace_) {
            bounded(pre_copy_, "the producer");
            t_produced_ = wall_ns();
        }
        if (staged_started_) {
            latency::mark("producer_flag_wait_begin");
            wait_staged_flag(producer_flag_, staged_token_, deadline);
            latency::mark("producer_flag_ready");
        } else {
            bounded(send_ready_, "the producer's device-to-host copy");
        }
        latency::mark("d2h_observed");
        std::uint64_t cmd = 0;
        std::memcpy(&cmd, send_host_, std::min<std::size_t>(sizeof(cmd), bytes_));
        cmd_out_.store(cmd, std::memory_order_relaxed);
        set_phase(Phase::transfer);
        Trailer trailer;
        trailer.epoch = group_.epoch; trailer.plan = group_.plan; trailer.sequence = sequence_; trailer.request = tag_.request;
        trailer.bytes = bytes_; trailer.rank = group_.rank; trailer.tag = tag_.kind; trailer.layer = tag_.layer;
        // Every message carries the CRC of this rank's previous deferred one (if any), checked by the peer.
        trailer.carried_sequence = tx_carry_sequence_;
        trailer.carried_crc32c = tx_carry_crc_;
        tx_carry_sequence_ = 0;
        // Deferred: this message's own CRC is computed while it is on the wire and goes in the next trailer.
        const bool defer = deferring_;
        const std::uint64_t sequence = sequence_;
        const std::span<const std::byte> payload(send_host_, bytes_);
        const std::span<const std::byte> received(receive_host_, bytes_);
        struct OnWire {
            std::span<const std::byte> payload;
            std::uint32_t crc;
        } on_wire{payload, 0U};
        if (defer) {
            trailer.flags = trailer_crc_deferred;
        } else {
            latency::mark("crc_tx_begin");
            trailer.crc32c = crc32c(payload);
            latency::mark("crc_tx_end");
        }
        if (trace_) trailer.staged_ns = t_staged_ = wall_ns();
        std::memcpy(send_host_ + bytes_, &trailer, sizeof(trailer));
        stats_.last_staged = std::chrono::steady_clock::now();
        latency::mark("transport_begin");
        if (defer)
            transport_->exchange_overlapped(std::span<const std::byte>(send_host_, bytes_ + sizeof(Trailer)),
                                            std::span<std::byte>(receive_host_, bytes_ + sizeof(Trailer)),
                                            [](void* context) {
                                                auto& w = *static_cast<OnWire*>(context);
                                                latency::mark("crc_tx_wire_begin");
                                                w.crc = crc32c(w.payload);
                                                latency::mark("crc_tx_wire_end");
                                            },
                                            &on_wire);
        else
            transport_->exchange(std::span<const std::byte>(send_host_, bytes_ + sizeof(Trailer)),
                                 std::span<std::byte>(receive_host_, bytes_ + sizeof(Trailer)));
        latency::mark("transport_end");
        stats_.last_received = std::chrono::steady_clock::now();
        if (trace_) t_received_ = wall_ns();
        std::uint64_t got = 0;
        std::memcpy(&got, receive_host_, std::min<std::size_t>(sizeof(got), bytes_));
        cmd_in_.store(got, std::memory_order_relaxed);
        set_phase(Phase::check);
        // The peer's message is checked here, as soon as it is complete: on a mixed route this overlaps the device
        // exchange's part instead of following it in finish(). A mismatch is rethrown by finish().
        // STRIX_DIST_CHECK_ON_CALLER=1: checked by finish() on the caller's thread instead (the earlier placement).
        if (!check_on_caller()) {
            latency::mark("crc_rx_begin");
            validate_peer(defer);
            latency::mark("crc_rx_end");
        }
        if (defer) {
            tx_carry_sequence_ = sequence + 1U;
            tx_carry_crc_ = on_wire.crc;
            // finish() returns now; the consumer reads the receive view while its CRC is computed here. The next
            // exchange's job (this worker runs one job at a time) checks it against the peer's next trailer.
            progress_->delivered.store(sequence + 1U, std::memory_order_release);
            latency::mark("crc_rx_deferred_begin");
            rx_pending_crc_ = crc32c(received);
            rx_pending_sequence_ = sequence + 1U;
            latency::mark("crc_rx_deferred_end");
        }
    });
    returned_at_ = std::chrono::steady_clock::now();
}

void Communicator::drain_tail() {
    if (!tail_pending_) return;
    tail_pending_ = false;
    try {
        if (!progress_->worker.wait_until(std::chrono::steady_clock::now() + wait_timeout_))
            stop("the previous exchange's deferred CRC work did not return within " + std::to_string(wait_timeout_.count()) + " ms");
    } catch (const std::exception& e) {
        stop(e.what());
    }
}

void Communicator::finish(hipStream_t stream, void* remote) {
    if (pipeline_started_) {
        if (!remote) throw std::invalid_argument("dist pipeline receive buffer");
        finish_pipeline(stream, remote);
        return;
    }
    if (direct_started_) {
        if (!remote) throw std::invalid_argument("dist exchange receive buffer");
        direct_started_ = false;
        const auto t0 = std::chrono::steady_clock::now();
        auto* const direct_remote = static_cast<std::byte*>(remote) + direct_offset_;
        if (direct_bytes_ <= direct_->capacity()) {
            direct_->finish(stream, direct_remote);
        } else {
            // Chunks of the capacity, one exchange each, in order (no overlap with the caller's work).
            const std::size_t chunk = direct_->capacity() & ~std::size_t{63};
            for (std::size_t offset = 0; offset < direct_bytes_; offset += chunk) {
                const std::size_t n = std::min(chunk, direct_bytes_ - offset);
                // The next chunk exactly, after the last one the first chunk of a message of this size.
                const std::size_t after = offset + n < direct_bytes_ ? std::min(chunk, direct_bytes_ - offset - n) : std::min(chunk, direct_bytes_);
                direct_->use_rings(direct_offset_ ? mix_rings_ : route(n).rings);
                direct_->expect_next(after, direct_offset_ ? mix_rings_ : route(after).rings);
                direct_->start(stream, static_cast<const std::byte*>(direct_local_) + offset, n,
                               ExchangeIdentity{group_.plan, tag_.request, tag_.kind, tag_.layer + static_cast<std::uint32_t>(offset / chunk)});
                direct_->finish(stream, direct_remote + offset);
            }
        }
        stats_.bytes_sent += direct_bytes_;
        if (direct_offset_) { // mixed: the host part counts the exchange and its time
            stats_.check += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            finish_host(stream, remote);
            return;
        }
        ++stats_.exchanges;
        stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - stats_.last_entry).count();
        stats_.check += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
    if (!started_) throw std::logic_error("dist exchange: finish() without start()");
    if (!remote) throw std::invalid_argument("dist exchange receive buffer");
    finish_host(stream, remote);
}

void Communicator::set_phase(Phase p) {
    phase_.store(static_cast<int>(p), std::memory_order_release);
    phase_since_.store(static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()), std::memory_order_release);
}

std::string Communicator::diagnose() const {
    static const char* const names[] = {"idle", "receive-buffer reuse wait", "producer wait", "transfer", "check"};
    std::ostringstream out;
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const int ph = phase_.load(std::memory_order_acquire);
    out << "dist state: rank " << group_.rank << ", phase " << names[std::clamp(ph, 0, 4)] << " for "
        << (now - static_cast<long long>(phase_since_.load(std::memory_order_acquire))) / 1000000 << " ms; in flight: "
        << current_.path << " seq " << current_.sequence << " " << current_.bytes << " B tag " << current_.tag << " layer " << current_.layer
        << " request " << current_.request << " command out 0x" << std::hex << cmd_out_.load() << " in 0x" << cmd_in_.load() << std::dec;
    out << "; last collectives (oldest first):";
    const std::size_t n = std::min<std::size_t>(history_next_, history_.size());
    for (std::size_t i = history_next_ - n; i < history_next_; ++i) {
        const auto& h = history_[i % history_.size()];
        out << " [" << h.path << " " << h.sequence << " " << h.bytes << "B t" << h.tag << " l" << h.layer << " r" << h.request << " c0x" << std::hex
            << h.cmd_out << "/0x" << h.cmd_in << std::dec << "]";
    }
    if (const auto s = transport_->state(); !s.empty()) out << "; streams: " << s;
    return out.str();
}

void Communicator::stop(const std::string& why) {
    const std::string tuple = " [exchange " + std::to_string(current_.sequence) + ": " + std::to_string(current_.bytes) + " B, tag " +
                              std::to_string(current_.tag) + ", layer " + std::to_string(current_.layer) + ", request " +
                              std::to_string(current_.request) + "]";
    const std::string state = diagnose();
    if (terminal_.empty()) {
        terminal_ = why + tuple;
        cancel_.store(true, std::memory_order_release);
        if (pipeline_started_) pipeline_->fail();
        std::fprintf(stderr, "%s\n%s\n", terminal_.c_str(), state.c_str());
        std::fflush(stderr);
        transport_->abort(terminal_); // closes the channel: the peer's pending reads end now
    }
    throw std::runtime_error(why + tuple);
}

bool Communicator::check_on_caller() {
    static const bool on = [] { const char* v = std::getenv("STRIX_DIST_CHECK_ON_CALLER"); return v && v[0] == '1'; }();
    return on;
}

void Communicator::validate_peer(bool deferred) {
    Trailer peer;
    std::memcpy(&peer, receive_host_ + bytes_, sizeof(peer));
    if (peer.magic != trailer_magic || peer.epoch != group_.epoch || peer.plan != group_.plan || peer.sequence != sequence_ ||
        peer.request != tag_.request || peer.bytes != bytes_ || peer.rank != 1U - group_.rank || peer.tag != tag_.kind ||
        peer.layer != tag_.layer)
        throw std::runtime_error("dist exchange: the peer's trailer does not match this collective");
    // The previous deferred message's CRC (computed here after it was delivered) against the one the peer carries now.
    if (peer.carried_sequence != rx_pending_sequence_)
        throw std::runtime_error("dist exchange: deferred CRC carry mismatch (peer carries " + std::to_string(peer.carried_sequence) +
                                 ", expected " + std::to_string(rx_pending_sequence_) + "; STRIX_DIST_CRC_DEFER must match on both ranks)");
    if (rx_pending_sequence_ && peer.carried_crc32c != rx_pending_crc_)
        throw std::runtime_error("dist exchange: payload CRC mismatch (deferred, exchange " + std::to_string(rx_pending_sequence_ - 1U) + ")");
    rx_pending_sequence_ = 0;
    if (((peer.flags & trailer_crc_deferred) != 0U) != deferred)
        throw std::runtime_error("dist exchange: the peer's CRC deferral differs from this rank's (STRIX_DIST_CRC_DEFER must match)");
    if (!deferred && peer.crc32c != crc32c(std::span<const std::byte>(receive_host_, bytes_)))
        throw std::runtime_error("dist exchange: payload CRC mismatch");
    peer_staged_ns_ = peer.staged_ns;
    if (trace_) t_validated_ = wall_ns();
}

void Communicator::finish_host(hipStream_t stream, void* remote) {
    latency::mark("finish_wait_begin");
    const auto entry = std::chrono::steady_clock::now();
    started_ = false;
    try {
        // The job bounds its own waits (producer: wait_timeout_, each stream: the transport's no-progress timeout);
        // this bound is the last resort if it does not return.
        const auto bound = entry + 2 * wait_timeout_ + std::chrono::seconds(5);
        if (deferring_) {
            // Until the job delivers this message (it then computes the deferred CRCs) or ends (a failure: rethrown).
            for (unsigned i = 0; progress_->delivered.load(std::memory_order_acquire) != sequence_ + 1U; ++i) {
                if (progress_->worker.idle()) {
                    progress_->worker.wait();
                    if (progress_->delivered.load(std::memory_order_acquire) != sequence_ + 1U)
                        throw std::logic_error("dist exchange: deferred job ended without delivering");
                    break;
                }
                if ((i & 1023U) == 0U && std::chrono::steady_clock::now() >= bound)
                    stop("the exchange's progress job did not deliver within " + std::to_string((2 * wait_timeout_ + std::chrono::seconds(5)).count()) + " ms");
                cpu_relax();
            }
            tail_pending_ = true;
            latency::mark("delivered");
        } else if (!progress_->worker.wait_until(bound)) {
            stop("the exchange's progress job did not return within " + std::to_string((2 * wait_timeout_ + std::chrono::seconds(5)).count()) + " ms");
        }
    } catch (const std::exception& e) {
        // Which collective was in flight: a stall shows the same one on both ranks, ranks that disagree on the
        // sequence of exchanges show different ones. The communicator stops and closes the channel (the peer wakes).
        stop(e.what());
    }
    set_phase(Phase::idle);
    latency::mark("finish_wait_end");
    const auto received = std::chrono::steady_clock::now();
    if (check_on_caller()) validate_peer();
    const auto bytes = bytes_; // the peer's message was checked (trailer and CRC)
    if (!staged_started_ && remote) {
        latency::mark("h2d_enqueue_begin");
        check(hipMemcpyAsync(remote, receive_host_, bytes, hipMemcpyHostToDevice, stream), "host to device");
        if (!receive_done_) check(hipEventCreateWithFlags(&receive_done_, hipEventDisableTiming), "receive event");
        check(hipEventRecord(receive_done_, stream), "receive event");
        latency::mark("h2d_enqueued");
    }
    latency::mark("h2d_enqueued");
    if (trace_) {
        if (!h2d_done_) check(hipEventCreate(&h2d_done_), "trace event");
        check(hipEventRecord(h2d_done_, stream), "trace event");
    }
    if (!staged_started_ && remote) receive_pending_ = true;
    current_.cmd_out = cmd_out_.load(std::memory_order_relaxed);
    current_.cmd_in = cmd_in_.load(std::memory_order_relaxed);
    history_[history_next_++ % history_.size()] = current_;
    ++sequence_;
    ++stats_.exchanges;
    stats_.bytes_sent += bytes;
    const auto end = std::chrono::steady_clock::now();
    stats_.stage += std::chrono::duration<double>(stats_.last_staged - started_at_).count();
    stats_.wire += std::chrono::duration<double>(stats_.last_received - stats_.last_staged).count();
    stats_.check += std::chrono::duration<double>(end - received).count();
    stats_.overlap += std::chrono::duration<double>(entry - returned_at_).count();
    stats_.finish_wait += std::chrono::duration<double>(received - entry).count();
    stats_.seconds += std::chrono::duration<double>(end - started_at_).count();
    if (trace_) {
        // Written when the next exchange starts: the H2D copy's end is read from the GPU timeline then.
        pending_ = PendingTrace{sequence_ - 1U, bytes, tag_.kind, tag_.layer, t_entry_, t_produced_, t_staged_, peer_staged_ns_, t_received_, t_validated_};
        trace_pending_ = true;
    }
}

void Communicator::flush_trace() {
    trace_pending_ = false;
    float ms = 0.0F;
    std::uint64_t consumer = 0;
    if (hipEventSynchronize(h2d_done_) == hipSuccess && hipEventElapsedTime(&ms, pre_copy_, h2d_done_) == hipSuccess && pending_.produced)
        consumer = pending_.produced + static_cast<std::uint64_t>(static_cast<double>(ms) * 1e6);
    const auto& t = pending_;
    std::fprintf(trace_, "H %llu %zu %u %u %llu %llu %llu %llu %llu %llu %llu\n", static_cast<unsigned long long>(t.sequence), t.bytes, t.tag,
                 t.layer, static_cast<unsigned long long>(t.entry), static_cast<unsigned long long>(t.produced),
                 static_cast<unsigned long long>(t.staged), static_cast<unsigned long long>(t.peer_staged),
                 static_cast<unsigned long long>(t.received), static_cast<unsigned long long>(t.validated),
                 static_cast<unsigned long long>(consumer));
}

void Communicator::exchange_host(std::span<const std::byte> local, std::span<std::byte> remote, const Tag& tag) {
    if (!terminal_.empty()) throw std::runtime_error("dist exchange: the communicator stopped earlier: " + terminal_);
    if (started_ || direct_started_) throw std::logic_error("dist exchange: exchange_host() while an exchange is in flight");
    const std::size_t bytes = local.size();
    if (bytes == 0U || bytes > max_bytes_ || (!remote.empty() && remote.size() != bytes))
        throw std::invalid_argument("dist exchange_host size or buffers");
    const auto entry = std::chrono::steady_clock::now();
    stats_.last_entry = entry;
    // The receive buffer is reused only after the CPU has seen the previous copy out of it (exchange()'s) complete.
    if (receive_pending_)
        for (hipError_t ready; (ready = hipEventQuery(receive_done_)) != hipSuccess;) {
            if (ready != hipErrorNotReady) check(ready, "previous receive copy");
            cpu_relax();
        }
    receive_pending_ = false;
    std::memcpy(send_host_, local.data(), bytes);
    Trailer trailer;
    trailer.epoch = group_.epoch; trailer.plan = group_.plan; trailer.sequence = sequence_; trailer.request = tag.request;
    trailer.bytes = bytes; trailer.rank = group_.rank; trailer.tag = tag.kind; trailer.layer = tag.layer;
    trailer.crc32c = crc32c(std::span<const std::byte>(send_host_, bytes));
    std::memcpy(send_host_ + bytes, &trailer, sizeof(trailer));
    // A failed control exchange poisons the communicator (later calls throw "stopped earlier") and wakes the peer.
    std::chrono::steady_clock::time_point received{};
    try {
        stats_.last_staged = std::chrono::steady_clock::now();
        transport_->exchange(std::span<const std::byte>(send_host_, bytes + sizeof(Trailer)), std::span<std::byte>(receive_host_, bytes + sizeof(Trailer)));
        received = std::chrono::steady_clock::now();
        stats_.last_received = received;
        Trailer peer;
        std::memcpy(&peer, receive_host_ + bytes, sizeof(peer));
        if (peer.magic != trailer_magic || peer.epoch != group_.epoch || peer.plan != group_.plan || peer.sequence != sequence_ ||
            peer.request != tag.request || peer.bytes != bytes || peer.rank != 1U - group_.rank || peer.tag != tag.kind || peer.layer != tag.layer)
            throw std::runtime_error("dist exchange: the peer's trailer does not match this collective");
        if (peer.crc32c != crc32c(std::span<const std::byte>(receive_host_, bytes)))
            throw std::runtime_error("dist exchange: payload CRC mismatch");
    } catch (const std::exception& e) {
        stop(std::string("control exchange: ") + e.what());
    }
    if (!remote.empty()) std::memcpy(remote.data(), receive_host_, bytes);
    ++sequence_;
    ++stats_.exchanges;
    stats_.bytes_sent += bytes;
    const auto end = std::chrono::steady_clock::now();
    stats_.stage += std::chrono::duration<double>(stats_.last_staged - entry).count();
    stats_.wire += std::chrono::duration<double>(received - stats_.last_staged).count();
    stats_.check += std::chrono::duration<double>(end - received).count();
    stats_.seconds += std::chrono::duration<double>(end - entry).count();
}

} // namespace strix::dist
