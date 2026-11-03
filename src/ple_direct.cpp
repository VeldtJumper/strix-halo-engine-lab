#include "strix/ple_direct.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <poll.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strix::ple_reclaim {
namespace {
thread_local bool background_read = false;
thread_local DirectActivity* read_activity = nullptr;
std::uint64_t ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
struct Fd {
    int value;
    ~Fd() {
#if defined(__linux__)
        if (value >= 0) ::close(value);
#endif
    }
};
struct Free { void operator()(std::byte* p) const { std::free(p); } };

#if defined(__linux__)
// One submitter/reaper owns both rings. Acquire/release pairs are shared with
// the kernel, not C++20 blocking atomic operations or a liburing dependency.
struct Ring {
    Fd fd{-1};
    io_uring_params params{};
    void* sq = MAP_FAILED;
    void* cq = MAP_FAILED;
    io_uring_sqe* sqes = nullptr;
    std::size_t sq_size = 0, cq_size = 0;
    explicit Ring(int injected_errno) {
        if (injected_errno) { errno = injected_errno; return; }
        fd.value = static_cast<int>(::syscall(__NR_io_uring_setup, 512U, &params));
        if (fd.value < 0) return;
        sq_size = params.sq_off.array + params.sq_entries * sizeof(unsigned);
        cq_size = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
        if (params.features & IORING_FEAT_SINGLE_MMAP) sq_size = cq_size = std::max(sq_size, cq_size);
        sq = ::mmap(nullptr, sq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd.value, IORING_OFF_SQ_RING);
        if (sq == MAP_FAILED) throw DirectError("PLE io_uring: SQ mmap failed");
        cq = (params.features & IORING_FEAT_SINGLE_MMAP) ? sq :
            ::mmap(nullptr, cq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd.value, IORING_OFF_CQ_RING);
        if (cq == MAP_FAILED) { ::munmap(sq, sq_size); sq = MAP_FAILED; throw DirectError("PLE io_uring: CQ mmap failed"); }
        auto* mapped = ::mmap(nullptr, params.sq_entries * sizeof(io_uring_sqe),
            PROT_READ | PROT_WRITE, MAP_SHARED, fd.value, IORING_OFF_SQES);
        if (mapped == MAP_FAILED) {
            if (cq != sq) ::munmap(cq, cq_size);
            ::munmap(sq, sq_size); sq = cq = MAP_FAILED;
            throw DirectError("PLE io_uring: SQE mmap failed");
        }
        sqes = static_cast<io_uring_sqe*>(mapped);
    }
    ~Ring() {
        if (sqes) ::munmap(sqes, params.sq_entries * sizeof(io_uring_sqe));
        if (cq != MAP_FAILED && cq != sq) ::munmap(cq, cq_size);
        if (sq != MAP_FAILED) ::munmap(sq, sq_size);
    }
    unsigned* sq_word(unsigned offset) { return reinterpret_cast<unsigned*>(static_cast<char*>(sq) + offset); }
    unsigned* cq_word(unsigned offset) { return reinterpret_cast<unsigned*>(static_cast<char*>(cq) + offset); }
    static unsigned acquire(unsigned* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
    static void release(unsigned* p, unsigned n) { __atomic_store_n(p, n, __ATOMIC_RELEASE); }
    unsigned push(int file, void* out, unsigned length, std::uint64_t offset, std::uint64_t tag) {
        const auto tail = acquire(sq_word(params.sq_off.tail));
        const auto index = tail & *sq_word(params.sq_off.ring_mask);
        auto& entry = sqes[index]; entry = {};
        entry.opcode = IORING_OP_READ; entry.fd = file;
        entry.addr = reinterpret_cast<std::uintptr_t>(out); entry.len = length; entry.off = offset; entry.user_data = tag;
        sq_word(params.sq_off.array)[index] = index;
        release(sq_word(params.sq_off.tail), tail + 1U);
        return tail + 1U;
    }
    int enter(unsigned submit, bool wait) {
        return static_cast<int>(::syscall(__NR_io_uring_enter, fd.value, submit, wait ? 1U : 0U,
            wait ? IORING_ENTER_GETEVENTS : 0U, nullptr, 0U));
    }
    bool pop(io_uring_cqe& result) {
        const auto head = acquire(cq_word(params.cq_off.head));
        if (head == acquire(cq_word(params.cq_off.tail))) return false;
        const auto* entries = reinterpret_cast<io_uring_cqe*>(static_cast<char*>(cq) + params.cq_off.cqes);
        result = entries[head & *cq_word(params.cq_off.ring_mask)];
        release(cq_word(params.cq_off.head), head + 1U);
        return true;
    }
};
#endif
}

DirectReadScope::DirectReadScope(bool background, DirectActivity* activity)
    : previous_(background_read), previous_activity_(read_activity) { background_read = background; read_activity = activity; }
DirectReadScope::~DirectReadScope() {
    if (read_activity && previous_activity_ && read_activity != previous_activity_) {
        previous_activity_->bytes += read_activity->bytes; previous_activity_->reads += read_activity->reads;
        previous_activity_->wait_ns += read_activity->wait_ns;
        if (read_activity->begin_ns && (!previous_activity_->begin_ns || read_activity->begin_ns < previous_activity_->begin_ns))
            previous_activity_->begin_ns = read_activity->begin_ns;
        previous_activity_->end_ns = std::max(previous_activity_->end_ns, read_activity->end_ns);
    }
    background_read = previous_; read_activity = previous_activity_;
}
bool DirectReadScope::background() { return background_read; }

struct DirectPageCache::Impl {
    static constexpr std::size_t workers = 32, queue_depth = 512;
    struct Job;
    struct Wait { std::uint64_t remaining = 0; };
    struct Entry { std::uint64_t slot; bool ready = false; std::uint64_t views = 0; Job* job = nullptr; };
    struct Run { std::uint64_t first, slot, count; };
    struct Job {
        Run run;
        std::uint64_t received = 0, begin = 0;
        unsigned retries = 0;
        std::uint64_t first_ns = 0, last_ns = 0, operations = 0;
        unsigned ticket = 0;
        std::vector<std::shared_ptr<Wait>> waiters;
    };
    Fd fd;
    const std::uint64_t bytes, page, capacity;
    Read read;
    std::unique_ptr<std::byte, Free> arena;
    mutable std::mutex mutex;
    std::condition_variable wake, done;
    std::map<std::uint64_t, Entry> entries;
    std::set<std::uint64_t> free_slots;
    std::uint64_t next_slot = 0, critical_waiters = 0, active = 0;
    std::deque<std::shared_ptr<Job>> critical_queue, ahead_queue;
    std::vector<std::thread> pool;
    std::exception_ptr error;
    bool stopping = false, uring = false, emulated = false;
    const bool credit_only;
    int fallback = 0;
    DirectStats counters;
#if defined(__linux__)
    std::unique_ptr<Ring> ring;
#endif

    Impl(int owned_fd, std::uint64_t size, std::uint64_t p, std::uint64_t cap, Read seam, TestRing test, bool verify_only)
        : fd{owned_fd}, bytes(size), page(p), capacity(cap), read(std::move(seam)), credit_only(verify_only) {
        if (page != 4096 || !bytes || capacity < page || capacity % page ||
            capacity > std::numeric_limits<std::size_t>::max() || bytes > static_cast<std::uint64_t>(INT64_MAX))
            throw std::invalid_argument("PLE O_DIRECT: invalid 4 KiB reader geometry");
#if defined(__linux__)
        const bool synthetic = static_cast<bool>(read);
        if (!read) {
            const int flags = ::fcntl(fd.value, F_GETFL);
            if (flags < 0 || !(flags & O_DIRECT))
                throw DirectError("PLE O_DIRECT: descriptor is not O_DIRECT; rolling refused");
            read = [](int f, void* out, std::size_t n, std::uint64_t at) {
                return static_cast<std::int64_t>(::pread(f, out, n, static_cast<off_t>(at)));
            };
        }
        void* storage = nullptr;
        if (::posix_memalign(&storage, static_cast<std::size_t>(page), static_cast<std::size_t>(credit_only ? page : capacity)))
            throw std::bad_alloc();
        arena.reset(static_cast<std::byte*>(storage));
        counters.peak_owned_bytes = credit_only ? page : capacity;
        // The checked pread probe remains before drop/credit, including when a
        // kernel denies io_uring. No fallback changes the O_DIRECT descriptor.
        Job probe; probe.run = {0, 0, 1};
        while (!perform(probe)) {}
        emulated = test.emulate;
        if (emulated) uring = true;
        else if (!synthetic || test.setup_errno) {
            ring = std::make_unique<Ring>(test.setup_errno);
            if (ring->fd.value < 0) {
                fallback = errno;
                if (fallback != ENOSYS && fallback != EPERM && fallback != EINVAL)
                    throw DirectError(std::string("PLE io_uring: setup failed: ") + std::strerror(fallback));
                ring.reset();
            } else {
                // Probe opcode/enter support too, before publishing any source
                // ownership. Completion EINVAL here can safely select pread.
                Job ring_probe; ring_probe.run = {0, 0, 1};
                bool ready = false;
                while (!ready && !fallback) {
                    ring_probe.begin = ns();
                    ring->push(fd.value, arena.get() + ring_probe.received,
                        static_cast<unsigned>(page - ring_probe.received), ring_probe.received, 0);
                    int result;
                    unsigned attempts = 0;
                    do {
                        result = ring->enter(1, true);
                    } while (result < 0 && (errno == EINTR || errno == EAGAIN) && ++attempts < 1024U);
                    if (result < 0) {
                        fallback = errno;
                        if (fallback != ENOSYS && fallback != EPERM && fallback != EINVAL)
                            throw DirectError("PLE io_uring: probe submission failed");
                    } else {
                        io_uring_cqe completion{};
                        while (!ring->pop(completion)) {
                            if (ring->enter(0, true) < 0 && errno != EINTR && errno != EAGAIN)
                                throw DirectError("PLE io_uring: probe completion failed");
                        }
                        if (completion.res == -EINVAL || completion.res == -EPERM || completion.res == -ENOSYS)
                            fallback = -completion.res;
                        else ready = complete(ring_probe, completion.res);
                    }
                }
                if (fallback) ring.reset(); else uring = true;
            }
        }
#else
        (void)test;
        throw DirectUnsupported("PLE O_DIRECT: Linux is required; rolling refused");
#endif
        // All accepted probe I/O completed synchronously above. Preflight owns
        // only a probe page and never publishes a worker or prompt source.
        if (credit_only) return;
        try {
            if (uring) pool.emplace_back([this] { ring_worker(); });
            else for (std::size_t i = 0; i < workers; ++i) pool.emplace_back([this] { worker(); });
        } catch (...) { stop(); throw; }
    }
    ~Impl() { stop(); }
    void stop() {
        { const std::lock_guard lock(mutex); stopping = true; }
        wake.notify_all(); done.notify_all();
        for (auto& t : pool) if (t.joinable()) t.join();
    }
    void check() const {
        if (error) std::rethrow_exception(error);
        if (stopping) throw std::runtime_error("PLE O_DIRECT: reader stopped");
    }
    bool available() const {
        return !critical_queue.empty() || (!ahead_queue.empty() && (!critical_waiters || stopping || error));
    }
    std::shared_ptr<Job> take() {
        auto& queue = critical_queue.empty() ? ahead_queue : critical_queue;
        auto job = queue.front(); queue.pop_front();
        ++active; counters.peak_inflight = std::max(counters.peak_inflight, active);
        return job;
    }
    void finish(const std::shared_ptr<Job>& job, std::exception_ptr failure) {
        // Caller holds mutex. Even failed jobs become drainable, but check()
        // poisons every source/copy certificate before bytes can be consumed.
        if (failure && !error) error = failure;
        for (std::uint64_t i = 0; i < job->run.count; ++i) {
            auto& entry = entries.at(job->run.first + i); entry.ready = true; entry.job = nullptr;
        }
        bool completed_waiter = false;
        for (auto& waiter : job->waiters) { assert(waiter->remaining); completed_waiter |= !--waiter->remaining; }
        job->waiters.clear();
        --active;
        if (completed_waiter) done.notify_all();
        if (failure) wake.notify_all();
    }
    // The raw CQ and injected/pread completions share precisely this alignment,
    // bounded retry and EOF handling. A retry remains active/critical.
    bool complete(Job& job, std::int64_t result) {
        const auto& run = job.run;
        const auto offset = run.first * page, length = run.count * page;
        const auto expected = std::min(length, bytes - offset);
        {
            const std::lock_guard lock(mutex);
            ++counters.reads; job.last_ns = ns(); ++job.operations;
            counters.read_ns += job.last_ns - job.begin;
            if (result > 0) counters.bytes += static_cast<std::uint64_t>(result);
        }
        if (result == -EINTR || result == -EAGAIN) {
            if (++job.retries > 1024U) throw DirectError("PLE O_DIRECT: retry limit; rolling refused");
            return false;
        }
        if (result == -EINVAL || result == -EOPNOTSUPP)
            throw DirectUnsupported("PLE O_DIRECT: unsupported read completion; rolling refused");
        if (result <= 0) throw DirectError("PLE O_DIRECT: read failed/short EOF; rolling refused");
        if (static_cast<std::uint64_t>(result) > length - job.received)
            throw DirectError("PLE O_DIRECT: invalid read completion");
        job.received += static_cast<std::uint64_t>(result);
        if (job.received < expected) {
            if (job.received % page) throw DirectError("PLE O_DIRECT: unaligned short read; rolling refused");
            return false;
        }
        if (job.received < length) std::memset(arena.get() + run.slot * page + job.received, 0,
            static_cast<std::size_t>(length - job.received));
        return true;
    }
    bool perform(Job& job) {
        const auto& run = job.run;
        job.begin = ns();
        if (!job.first_ns) job.first_ns = job.begin;
        const auto n = read(fd.value, arena.get() + run.slot * page + job.received,
            static_cast<std::size_t>(run.count * page - job.received), run.first * page + job.received);
        const int saved = errno;
        return complete(job, n < 0 ? -saved : n);
    }
    void worker() noexcept {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return stopping || available(); });
                if (!available()) return;
                job = take();
            }
            std::exception_ptr failure;
            try { while (!perform(*job)) {} } catch (...) { failure = std::current_exception(); }
            const std::lock_guard lock(mutex); finish(job, failure);
        }
    }
    void ring_worker() noexcept {
#if defined(__linux__)
        std::array<std::shared_ptr<Job>, queue_depth> inflight{};
        std::size_t inflight_count = 0, next = 0;
        unsigned pending = 0, submit_retries = 0, wait_retries = 0;
        try {
            for (;;) {
                {
                    std::unique_lock lock(mutex);
                    if (!inflight_count) {
                        wake.wait(lock, [&] { return stopping || available(); });
                        if (!available()) return;
                    }
                    while (available() && inflight_count < queue_depth) {
                        auto job = take();
                        while (inflight[next]) next = (next + 1U) % queue_depth;
                        const auto tag = next;
                        inflight[tag] = job; ++inflight_count;
                        if (!emulated) {
                            job->begin = ns(); job->first_ns = job->begin;
                            job->ticket = ring->push(fd.value, arena.get() + job->run.slot * page,
                                static_cast<unsigned>(job->run.count * page), job->run.first * page, tag);
                            ++pending;
                        }
                    }
                }
                if (emulated) {
                    // Deterministic CPU CQ seam, not a kernel-throughput test.
                    const auto it = std::find_if(inflight.begin(), inflight.end(), [](const auto& job) { return bool(job); });
                    std::exception_ptr failure;
                    try { while (!perform(**it)) {} } catch (...) { failure = std::current_exception(); }
                    { const std::lock_guard lock(mutex); finish(*it, failure); }
                    it->reset(); --inflight_count; continue;
                }
                // Submit in batches, handling partial accepted batches. No ring
                // mutex is needed: this dedicated thread owns SQ and CQ.
                if (pending) {
                    const auto n = ring->enter(pending, false);
                    if (n < 0 && errno != EINTR && errno != EAGAIN)
                        throw DirectError("PLE io_uring: batch submission failed; rolling refused");
                    if (n > 0 && static_cast<unsigned>(n) > pending)
                        throw DirectError("PLE io_uring: invalid submission count");
                    if (n > 0) { pending -= static_cast<unsigned>(n); submit_retries = 0; }
                    else if (++submit_retries > 1024U)
                        throw DirectError("PLE io_uring: submission retry limit; rolling refused");
                }
                io_uring_cqe cqe{};
                if (!ring->pop(cqe)) {
                    // Reap while a batch is only partly accepted. Waiting for
                    // an unsubmitted SQE could otherwise stall the entire ring.
                    if (pending) { std::this_thread::yield(); continue; }
                    if (ring->enter(0, true) < 0) {
                        if ((errno != EINTR && errno != EAGAIN) || ++wait_retries > 1024U)
                            throw DirectError("PLE io_uring: completion wait failed; rolling refused");
                    }
                    continue;
                }
                wait_retries = 0;
                do {
                    if (cqe.user_data >= queue_depth || !inflight[cqe.user_data]) throw DirectError("PLE io_uring: unknown completion");
                    auto& job = inflight[cqe.user_data];
                    std::exception_ptr failure;
                    bool ready = false;
                    try { ready = complete(*job, cqe.res); } catch (...) { failure = std::current_exception(); }
                    if (!ready && !failure) {
                        job->begin = ns();
                        job->ticket = ring->push(fd.value, arena.get() + job->run.slot * page + job->received,
                            static_cast<unsigned>(job->run.count * page - job->received),
                            job->run.first * page + job->received, cqe.user_data);
                        ++pending;
                    } else {
                        { const std::lock_guard lock(mutex); finish(job, failure); }
                        job.reset(); --inflight_count;
                    }
                } while (ring->pop(cqe));
            }
        } catch (...) {
            const auto failure = std::current_exception();
            // A failed enter may leave submitted reads alive. Keep the ring and
            // arena mapped until every accepted SQE has a CQE. Published but
            // unconsumed SQEs have never reached the device (no SQPOLL is used).
            if (ring) {
                const auto head = Ring::acquire(ring->sq_word(ring->params.sq_off.head));
                for (auto& job : inflight) if (job && static_cast<int>(job->ticket - head) > 0) {
                    const std::lock_guard lock(mutex); finish(job, failure); job.reset(); --inflight_count;
                }
                while (inflight_count) {
                    io_uring_cqe completion{};
                    if (ring->pop(completion)) {
                        if (completion.user_data < queue_depth && inflight[completion.user_data]) {
                            auto& job = inflight[completion.user_data];
                            const std::lock_guard lock(mutex); finish(job, failure); job.reset(); --inflight_count;
                        }
                    } else {
                        pollfd descriptor{ring->fd.value, POLLIN, 0};
                        (void)::poll(&descriptor, 1, -1);
                    }
                }
                ring.reset();
            }
            const std::lock_guard lock(mutex);
            if (!error) error = failure;
            for (auto& job : inflight) if (job) finish(job, failure);
            for (auto* queue : {&critical_queue, &ahead_queue}) {
                for (auto& job : *queue) { ++active; finish(job, failure); }
                queue->clear();
            }
            done.notify_all(); wake.notify_all();
        }
#endif
    }
};

DirectPageCache::DirectPageCache(int fd, std::uint64_t bytes, std::uint64_t page, std::uint64_t cap, Read read, TestRing test, bool credit_only)
    : impl_(std::make_shared<Impl>(fd, bytes, page, cap, std::move(read), test, credit_only)) {}
DirectPageCache::~DirectPageCache() = default;
const char* DirectPageCache::reader() const noexcept { return impl_->uring ? (impl_->emulated ? "io_uring-test" : "io_uring") : "pread-pool"; }
int DirectPageCache::fallback_errno() const noexcept { return impl_->fallback; }

std::uint64_t DirectPageCache::ensure(std::span<const std::uint64_t> requested, bool background, bool critical) {
    auto& s = *impl_;
    if (s.credit_only) throw std::logic_error("PLE O_DIRECT: credit-only reader cannot activate prompt I/O");
    std::vector<std::uint64_t> pages(requested.begin(), requested.end());
    std::sort(pages.begin(), pages.end()); pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    std::unique_lock lock(s.mutex);
    s.check();
    std::size_t missing = 0;
    bool all_ready = true;
    for (auto pg : pages) {
        if (pg > (s.bytes - 1U) / s.page) throw std::out_of_range("PLE O_DIRECT: page outside file");
        const auto it = s.entries.find(pg);
        missing += it == s.entries.end();
        all_ready &= it != s.entries.end() && it->second.ready;
    }
    if (all_ready) { s.counters.hits += pages.size(); return 0; }
    if (missing > s.capacity / s.page - s.entries.size())
        throw std::runtime_error("PLE O_DIRECT: owned buffer cap exhausted before read");
    std::vector<Impl::Run> runs;
    std::vector<Impl::Job*> joined;
    std::vector<std::uint64_t> reserved;
    std::vector<std::shared_ptr<Impl::Job>> owned_jobs;
    runs.reserve(missing); joined.reserve(pages.size()); reserved.reserve(missing); owned_jobs.reserve(missing);
    bool registered_critical = false;
    std::size_t published_jobs = 0;
    const auto begin = ns();
    auto waiter = std::make_shared<Impl::Wait>();
    std::uint64_t issued_bytes = 0;
    try {
        for (auto pg : pages) {
            if (const auto it = s.entries.find(pg); it != s.entries.end()) {
                ++s.counters.hits;
                if (!it->second.ready) { joined.push_back(it->second.job); ++s.counters.inflight_hits; }
                continue;
            }
            std::uint64_t slot;
            if (s.free_slots.empty()) slot = s.next_slot++;
            else { auto it = s.free_slots.begin(); slot = *it; s.free_slots.erase(it); }
            assert(slot < s.capacity / s.page);
            s.entries.emplace(pg, Impl::Entry{slot, false}); reserved.push_back(pg);
            issued_bytes += std::min(s.page, s.bytes - pg * s.page);
            if (!runs.empty() && runs.back().first + runs.back().count == pg &&
                runs.back().slot + runs.back().count == slot && (runs.back().count + 1U) * s.page <= (128U << 10U))
                ++runs.back().count;
            else runs.push_back({pg, slot, 1});
        }
        s.counters.used_bytes = s.entries.size() * s.page;
        s.counters.peak_used_bytes = std::max(s.counters.peak_used_bytes, s.counters.used_bytes);
        // Every reservation is submitted as metadata up front, so the device can
        // maintain 512 outstanding reads even for sparse page sets. No extra data
        // buffers exist beyond the fixed arena. Demand promotes shared queued runs.
        if (critical) {
            ++s.critical_waiters; registered_critical = true;
            for (auto it = s.ahead_queue.begin(); it != s.ahead_queue.end();) {
                const auto& run = (*it)->run;
                const auto pg = std::lower_bound(pages.begin(), pages.end(), run.first);
                if (pg != pages.end() && *pg < run.first + run.count) {
                    s.critical_queue.push_back(*it); it = s.ahead_queue.erase(it);
                } else ++it;
            }
        }
        auto& queue = critical ? s.critical_queue : s.ahead_queue;
        std::sort(joined.begin(), joined.end(), std::less<Impl::Job*>{});
        joined.erase(std::unique(joined.begin(), joined.end()), joined.end());
        for (auto* job : joined) { assert(job); job->waiters.push_back(waiter); ++waiter->remaining; }
        for (const auto& run : runs) {
            auto job = std::make_shared<Impl::Job>(); job->run = run;
            job->waiters.push_back(waiter); ++waiter->remaining;
            for (std::uint64_t i = 0; i < run.count; ++i) s.entries.at(run.first + i).job = job.get();
            owned_jobs.push_back(job); queue.push_back(std::move(job)); ++published_jobs;
        }
    } catch (...) {
        // No new SQE can have been issued while this mutex is held. Roll back
        // partial queue publication, poison credit, and drain pre-existing
        // shared jobs before the caller retires its outer source lease.
        const auto failure = std::current_exception();
        auto& queue = critical ? s.critical_queue : s.ahead_queue;
        while (published_jobs) { queue.pop_back(); --published_jobs; }
        for (auto pg : reserved) { auto& entry = s.entries.at(pg); entry.ready = true; entry.job = nullptr; }
        s.counters.used_bytes = s.entries.size() * s.page;
        s.counters.peak_used_bytes = std::max(s.counters.peak_used_bytes, s.counters.used_bytes);
        if (!s.error) s.error = failure;
        if (registered_critical) --s.critical_waiters;
        s.wake.notify_all();
        s.done.wait(lock, [&] {
            return std::all_of(pages.begin(), pages.end(), [&](auto pg) {
                const auto it = s.entries.find(pg); return it == s.entries.end() || it->second.ready;
            });
        });
        std::rethrow_exception(failure);
    }
    if (runs.size() > 1U || !joined.empty()) s.wake.notify_all(); else s.wake.notify_one();
    // Count completed jobs once, rather than rescanning a block-sized page map
    // on every CQE (which would make a large sparse block quadratic).
    s.done.wait(lock, [&] { return !waiter->remaining; });
    const auto end = ns(), waited = end - begin;
    if (critical) { --s.critical_waiters; if (!s.ahead_queue.empty()) s.wake.notify_all(); }
    (background ? s.counters.hidden_wait_ns : s.counters.exposed_wait_ns) += waited;
    if (read_activity) {
        read_activity->bytes += issued_bytes; read_activity->wait_ns += waited;
        for (const auto& job : owned_jobs) {
            read_activity->reads += job->operations;
            if (job->first_ns && (!read_activity->begin_ns || job->first_ns < read_activity->begin_ns))
                read_activity->begin_ns = job->first_ns;
            read_activity->end_ns = std::max(read_activity->end_ns, job->last_ns);
        }
    }
    s.check();
    return missing;
}
void DirectPageCache::erase(std::span<const std::uint64_t> pages) {
    auto& s = *impl_;
    const std::lock_guard lock(s.mutex);
    for (auto pg : pages) if (const auto it = s.entries.find(pg); it != s.entries.end())
        if (!it->second.ready || it->second.views) throw std::logic_error("PLE O_DIRECT: retiring an in-flight page or pinned view");
    for (auto pg : pages) if (auto it = s.entries.find(pg); it != s.entries.end()) {
        s.free_slots.insert(it->second.slot); s.entries.erase(it);
    }
    s.counters.used_bytes = s.entries.size() * s.page;
}
std::shared_ptr<const DirectPageCache::View> DirectPageCache::view(std::span<const std::uint64_t> pages) const {
    auto result = std::shared_ptr<View>(new View(impl_));
    result->slots_.reserve(pages.size());
    auto& s = *impl_;
    const std::lock_guard lock(s.mutex);
    s.check();
    // Validate before taking pins: refusal leaves no partial ownership.
    for (auto pg : pages) {
        const auto it = s.entries.find(pg);
        if (it == s.entries.end() || !it->second.ready)
            throw std::logic_error("PLE O_DIRECT: view requires completed source pages");
    }
    for (auto pg : pages) result->slots_.emplace_back(pg, s.entries.at(pg).slot);
    std::sort(result->slots_.begin(), result->slots_.end());
    result->slots_.erase(std::unique(result->slots_.begin(), result->slots_.end()), result->slots_.end());
    for (const auto& slot : result->slots_) ++s.entries.at(slot.first).views;
    return result;
}
DirectPageCache::View::~View() {
    const std::lock_guard lock(owner_->mutex);
    for (const auto& slot : slots_) --owner_->entries.at(slot.first).views;
}
void DirectPageCache::View::copy(std::uint64_t offset, std::span<std::byte> out) const {
    const auto& s = *owner_;
    if (offset > s.bytes || out.size() > s.bytes - offset) throw std::out_of_range("PLE O_DIRECT: view row past EOF");
    while (!out.empty()) {
        const auto pg = offset / s.page;
        const auto it = std::lower_bound(slots_.begin(), slots_.end(), pg,
            [](const auto& slot, auto page) { return slot.first < page; });
        if (it == slots_.end() || it->first != pg)
            throw std::logic_error("PLE O_DIRECT: row outside pinned source view");
        const auto within = offset % s.page;
        const auto n = std::min<std::size_t>(out.size(), static_cast<std::size_t>(s.page - within));
        std::memcpy(out.data(), s.arena.get() + it->second * s.page + within, n);
        offset += n; out = out.subspan(n);
    }
}
void DirectPageCache::copy(std::uint64_t offset, std::span<std::byte> out) const {
    const auto& s = *impl_;
    const std::lock_guard lock(s.mutex);
    s.check();
    if (offset > s.bytes || out.size() > s.bytes - offset) throw std::out_of_range("PLE O_DIRECT: row past EOF");
    while (!out.empty()) {
        const auto it = s.entries.find(offset / s.page);
        // This is the only rolling row-byte source. A missing lease is an error,
        // never a reason to dereference the model mapping or use buffered I/O.
        if (it == s.entries.end() || !it->second.ready)
            throw std::logic_error("PLE O_DIRECT: row copy without completed owned-page lease");
        assert(it->second.slot < s.capacity / s.page);
        const auto within = offset % s.page;
        const auto n = std::min<std::size_t>(out.size(), static_cast<std::size_t>(s.page - within));
        std::memcpy(out.data(), s.arena.get() + it->second.slot * s.page + within, n);
        offset += n; out = out.subspan(n);
    }
}
DirectStats DirectPageCache::stats() const {
    const std::lock_guard lock(impl_->mutex);
    auto result = impl_->counters; result.pending_critical = impl_->critical_waiters; return result;
}
bool DirectPageCache::healthy() const { const std::lock_guard lock(impl_->mutex); return !impl_->error; }

} // namespace strix::ple_reclaim
