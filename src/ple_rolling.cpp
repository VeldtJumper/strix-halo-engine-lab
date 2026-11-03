#include "strix/ple_rolling.hpp"
#include "strix/ple_direct.hpp"
#include <algorithm>
#include <charconv>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace strix::ple_reclaim {
std::uint64_t rolling_positions(const char* rolling, const char* exact_memory, const char* positions) {
    if ((rolling && rolling[0] == '0') || (exact_memory && exact_memory[0] == '0')) return 0;
    if (!positions) return 1024;
    const std::string_view text(positions);
    std::uint64_t h = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), h);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || h < 256 || h > 65536 || (h & (h - 1)))
        throw std::invalid_argument("STRIX_PLE_ROLLING_POSITIONS must be a power of two in 256..65536");
    return h;
}
RollingPages::RollingPages(std::uint64_t h, std::uint64_t page, std::uint64_t offset, std::uint64_t row_bytes,
    std::uint32_t heads, std::span<const std::uint32_t> rows, std::uint64_t admitted,
    std::uint64_t decode_allowance, Io io)
    : h_(h), page_(page), offset_(offset), row_bytes_(row_bytes), admitted_(admitted),
      decode_allowance_(decode_allowance), heads_(heads), rows_(rows), io_(std::move(io)) {
    if (!h_ || !page_ || !row_bytes_ || !heads_ || rows_.size() % heads_ ||
        !io_.prefetch || !io_.evict || !io_.resident || !io_.reclaimable)
        throw std::invalid_argument("PLE rolling configuration incomplete");
}
RollingPages::Pages RollingPages::pages(std::span<const std::uint64_t> rows) const {
    Pages result;
    for (const auto row : rows) {
        const auto begin = offset_ + row * row_bytes_;
        for (auto pg = begin / page_; pg <= (begin + row_bytes_ - 1) / page_; ++pg)
            if (io_.reclaimable(pg)) result.push_back(pg);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
RollingPages::Pages RollingPages::window(std::uint64_t first, std::uint64_t count) const {
    const auto positions = rows_.size() / heads_;
    first = std::min<std::uint64_t>(first, positions);
    count = std::min<std::uint64_t>(count, positions - first);
    std::vector<std::uint64_t> ids(rows_.begin() + static_cast<std::ptrdiff_t>(first * heads_),
        rows_.begin() + static_cast<std::ptrdiff_t>((first + count) * heads_));
    return pages(ids);
}
void RollingPages::check() const {
    if (failed_ || finished_) throw std::runtime_error("PLE rolling: stopped after residency/copy failure or turn end");
}
void RollingPages::sample() {
    try {
        const auto resident = io_.resident();
        stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, resident);
        if (resident > admitted_) throw std::runtime_error("PLE rolling: resident bytes exceed admitted bound including slack");
    } catch (...) { failed_ = true; throw; }
}
void RollingPages::retain(const Pages& p) { for (const auto pg : p) ++refs_[pg]; }
void RollingPages::release(const Pages& p) {
    Pages dead;
    for (const auto pg : p) {
        auto it = refs_.find(pg);
        if (it == refs_.end() || !it->second) throw std::logic_error("PLE rolling source reference underflow");
        if (!--it->second) { refs_.erase(it); dead.push_back(pg); }
    }
    if (!dead.empty()) {
        std::sort(dead.begin(), dead.end());
        try { io_.evict(dead); stats_.evicted_pages += dead.size(); }
        catch (...) { failed_ = true; throw; }
    }
}
void RollingPages::gather(std::uint64_t first, std::uint64_t count, const std::function<void()>& copy) {
    {
        const std::lock_guard lock(mutex_);
        check();
        if (gathering_ || first != frontier_ || !count || count > h_ || first > rows_.size() / heads_ ||
            count > rows_.size() / heads_ - first)
            throw std::invalid_argument("PLE rolling gather must advance consecutive bounded source windows");
        gathering_ = true;
        const auto current = window(first, count), next = window(first + count, h_);
        // Retain both replacement windows before releasing the old lookahead:
        // a shared boundary page never transiently reaches zero references.
        retain(current); retain(next);
        release(next_);
        current_ = current; next_ = next;
        Pages live = current_;
        live.insert(live.end(), next_.begin(), next_.end());
        std::sort(live.begin(), live.end()); live.erase(std::unique(live.begin(), live.end()), live.end());
        try { stats_.prefetched_pages += io_.prefetch(live); sample(); }
        catch (...) { failed_ = true; throw; }
    }
    try { copy(); }
    catch (...) {
        const std::lock_guard lock(mutex_);
        failed_ = true; gathering_ = false;
        release(current_); current_.clear();
        throw;
    }
    const std::lock_guard lock(mutex_);
    // No source page is released until the synchronous copy (including joined
    // worker threads) has completed. GPU staging readers need no file lease.
    sample();
    release(current_); current_.clear();
    frontier_ = first + count; gathering_ = false;
    ++stats_.epochs;
    sample();
}
void RollingPages::source(std::span<const std::uint64_t> rows, const std::function<void()>& copy) {
    const auto p = pages(rows);
    {
        const std::lock_guard lock(mutex_);
        check();
        // Reads outside the live prompt windows are charged to this turn's
        // separate decode allowance, including lookups prefetched by decode.
        for (const auto pg : p) if (!refs_.contains(pg) && decode_.insert(pg).second) {
            ++refs_[pg];
            if (decode_.size() > decode_allowance_ / page_) {
                failed_ = true;
                throw std::runtime_error("PLE rolling: turn decode page allowance exceeded");
            }
        }
        retain(p);
        if (io_.missing) {
            try { stats_.faulted_pages += io_.missing(p); }
            catch (...) { failed_ = true; release(p); throw; }
        }
    }
    try { copy(); }
    catch (...) { const std::lock_guard lock(mutex_); failed_ = true; release(p); throw; }
    const std::lock_guard lock(mutex_);
    release(p);
    if (!gathering_ && (++copies_ % 64U == 1U)) sample();
}
void RollingPages::finish() {
    const std::lock_guard lock(mutex_);
    if (finished_) return;
    check();
    if (gathering_) throw std::logic_error("PLE rolling turn ended with an outstanding CPU copy");
    sample();
    release(current_); current_.clear(); release(next_); next_.clear();
    release(Pages(decode_.begin(), decode_.end())); decode_.clear();
    if (!refs_.empty()) throw std::logic_error("PLE rolling turn ended with source readers");
    sample(); finished_ = true;
}
RollingStats RollingPages::stats() const { const std::lock_guard lock(mutex_); return stats_; }
} // namespace strix::ple_reclaim


namespace strix::ple_reclaim {
RollingPipeline::RollingPipeline(std::uint64_t h, std::uint64_t page, std::uint64_t offset,
    std::uint64_t row_bytes, std::uint32_t heads, std::span<const std::uint32_t> rows,
    std::uint64_t admitted, std::uint64_t decode, std::uint64_t cap, std::uint64_t ahead, RollingPages::Io io, std::uint64_t block_rows)
    : h_(h), page_(page), offset_(offset), row_bytes_(row_bytes), admitted_(admitted),
      decode_allowance_(decode), cap_(cap),
      reserve_(page ? std::min(cap / 2U / page * page, h * heads * ((row_bytes + 2U * page - 2U) / page) * page) : 0U),
      ahead_(std::max({h, ahead, block_rows})), block_rows_(block_rows ? block_rows : h), heads_(heads), rows_(rows), io_(std::move(io)) {
    if (!h || !page || !row_bytes || !heads || rows.size() % heads || !reserve_ || cap % page ||
        reserve_ < heads * ((row_bytes + 2U * page - 2U) / page) * page ||
        admitted < cap || decode > admitted - cap ||
        !io_.prefetch || !io_.evict || !io_.resident || !io_.reclaimable)
        throw std::invalid_argument("PLE rolling pipeline configuration incomplete or cap below two source positions");
    target_ = std::min<std::uint64_t>(rows_.size() / heads_, ahead_);
    thread_ = std::thread([this] { helper(); });
}
RollingPipeline::~RollingPipeline() {
    try { finish(); } catch (...) { stop(); }
}
RollingPipeline::Pages RollingPipeline::pages(std::span<const std::uint64_t> rows) const {
    Pages result;
    for (auto row : rows) {
        const auto begin = offset_ + row * row_bytes_;
        for (auto pg = begin / page_; pg <= (begin + row_bytes_ - 1U) / page_; ++pg)
            if (io_.reclaimable(pg)) result.push_back(pg);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
RollingPipeline::Pages RollingPipeline::window(std::uint64_t first, std::uint64_t count) const {
    std::vector<std::uint64_t> ids(rows_.begin() + static_cast<std::ptrdiff_t>(first * heads_),
        rows_.begin() + static_cast<std::ptrdiff_t>((first + count) * heads_));
    return pages(ids);
}
std::uint64_t RollingPipeline::fit(std::uint64_t first, std::uint64_t count) const {
    count = std::min(count, h_);
    while (window(first, count).size() > reserve_ / page_) {
        if (count == 1U) throw std::runtime_error("PLE rolling: one position exceeds demand reserve");
        count = (count + 1U) / 2U;
    }
    return count;
}
void RollingPipeline::check() const {
    if (error_) std::rethrow_exception(error_);
    if (finished_ || stopping_) throw std::runtime_error("PLE rolling: source pipeline stopped");
}
void RollingPipeline::sample() {
    try {
        const auto resident = io_.resident();
        stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, resident);
        if (resident > admitted_) throw std::runtime_error("PLE rolling: resident bytes exceed admitted bound including slack");
    } catch (...) { error_ = std::current_exception(); throw; }
}
bool RollingPipeline::make_room(const Pages& p, std::uint64_t limit) {
    std::uint64_t extra = 0, protected_cold = 0;
    for (auto pg : p) {
        extra += !refs_.contains(pg) && !cold_.contains(pg);
        protected_cold += cold_.contains(pg);
    }
    const auto owned = refs_.size() + cold_.size() - decode_.size();
    const auto slots = limit / page_;
    if (owned + extra <= slots) return true;
    const auto needed = owned + extra - slots;
    if (needed > cold_.size() - protected_cold) return false;
    Pages dead;
    dead.reserve(static_cast<std::size_t>(needed));
    for (auto pg : cold_) if (!std::binary_search(p.begin(), p.end(), pg)) {
        dead.push_back(pg);
        if (dead.size() == needed) break;
    }
    // Only unreferenced pages outside every known future window enter cold_.
    // Drop before giving their reservation to another page, never after.
    try { io_.evict(dead); }
    catch (...) { error_ = std::current_exception(); throw; }
    for (auto pg : dead) cold_.erase(pg);
    stats_.evicted_pages += dead.size();
    return true;
}
void RollingPipeline::retain(const Pages& p) {
    for (auto pg : p) { cold_.erase(pg); ++refs_[pg]; }
    const auto owned = (refs_.size() + cold_.size() - decode_.size()) * page_;
    stats_.peak_owned_bytes = std::max<std::uint64_t>(stats_.peak_owned_bytes, owned);
    if (owned > cap_) throw std::logic_error("PLE rolling: page reservation exceeds cap");
}
void RollingPipeline::release(const Pages& p) {
    for (auto pg : p) {
        auto it = refs_.find(pg);
        if (it == refs_.end() || !it->second) throw std::logic_error("PLE rolling: source reference underflow");
        if (!--it->second) { refs_.erase(it); cold_.insert(pg); }
    }
}
void RollingPipeline::consumed() {
    while (!lookahead_.empty() && lookahead_.front().first + lookahead_.front().count <= frontier_) {
        release(lookahead_.front().pages);
        lookahead_.pop_front();
    }
    planned_ = std::max(planned_, frontier_);
}
void RollingPipeline::helper() noexcept {
    try {
        std::unique_lock lock(mutex_);
        for (;;) {
            wake_.wait(lock, [&] { return stopping_ || error_ || planned_ < target_; });
            if (stopping_ || error_) return;
            // Complete the entire next GPU block before issuing later blocks.
            // A small cap splits this ordered wave; it never changes its order.
            const auto first = planned_;
            const bool critical = first < gpu_end_ + std::min(block_rows_, rows_.size() / heads_ - gpu_end_);
            auto count = std::min(block_rows_ - planned_ % block_rows_, target_ - planned_);
            Pages p;
            for (;;) {
                p = window(planned_, count);
                if (make_room(p, cap_ - reserve_)) break;
                if (count == 1U) { p.clear(); break; }
                count = (count + 1U) / 2U;
                ++stats_.throttles;
            }
            if (p.empty()) {
                // Empty reclaimable windows still advance. Full windows wait for
                // consumption; the reserved demand slot never waits on hints.
                if (!window(planned_, count).empty()) { ++stats_.throttles; wake_.wait(lock); continue; }
            }
            retain(p);
            lookahead_.push_back({planned_, count, p});
            planned_ += count;
            stats_.lookahead_positions = std::max(stats_.lookahead_positions,
                planned_ > gpu_end_ ? planned_ - gpu_end_ : 0U);
            // A separate I/O lease prevents a completed CPU window from dropping
            // a page while a WILLNEED callback is still issuing it.
            retain(p);
            const auto record = io_records_.insert(io_records_.end(), {first, count, {}, false});
            lock.unlock();
            std::uint64_t issued = 0, resident = 0;
            std::exception_ptr failure;
            DirectActivity activity;
            try {
                const DirectReadScope scope(true, &activity);
                issued = critical && io_.critical ? io_.critical(p) : io_.prefetch(p);
                resident = io_.resident();
            }
            catch (...) { failure = std::current_exception(); }
            lock.lock();
            stats_.prefetched_pages += issued;
            record->activity = activity; record->complete = true;
            wake_.notify_all();
            release(p);
            stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, resident);
            if (failure) std::rethrow_exception(failure);
            if (resident > admitted_) throw std::runtime_error("PLE rolling: resident bytes exceed admitted bound including slack");
        }
    } catch (...) {
        const std::lock_guard lock(mutex_);
        error_ = std::current_exception();
        wake_.notify_all();
    }
}
void RollingPipeline::advance(std::uint64_t gpu_end) {
    const std::lock_guard lock(mutex_);
    check();
    const auto positions = static_cast<std::uint64_t>(rows_.size() / heads_);
    gpu_end_ = std::max(gpu_end_, std::min(positions, gpu_end));
    target_ = std::max(target_, gpu_end_ + std::min(ahead_, positions - gpu_end_));
    target_ = std::min(target_, positions);
    // Retain at most the recent GPU block plus live lookahead attribution.
    // Completed historical receipts must not grow with generation lifetime.
    const auto retired = gpu_end_ > block_rows_ ? gpu_end_ - block_rows_ : 0U;
    io_records_.remove_if([&](const auto& record) {
        return record.complete && record.first + record.count <= retired;
    });
    wake_.notify_all();
}
void RollingPipeline::gather_range(std::uint64_t first, std::uint64_t count,
    const std::function<void(std::uint64_t, std::uint64_t)>& copy) {
    const auto positions = rows_.size() / heads_;
    if (!count || first > positions || count > positions - first)
        throw std::invalid_argument("PLE rolling: invalid gather range");
    const auto end = first + count;
    while (first < end) {
        const auto n = fit(first, end - first);
        if (n < std::min(h_, end - first)) { const std::lock_guard lock(mutex_); ++stats_.split_windows; }
        gather(first, n, [&] { copy(first, n); });
        first += n;
    }
}
void RollingPipeline::gather(std::uint64_t first, std::uint64_t count, const std::function<void()>& copy) {
    Pages current;
    {
        std::unique_lock lock(mutex_);
        // Different GPU tail-window owners can have overlapping c1 tasks.
        // Serialize their H-sized CPU copies, never their GPU execution or I/O hints.
        wake_.wait(lock, [&] { return !gathering_ || error_ || stopping_; });
        check();
        if (first > frontier_ || !count || first > rows_.size() / heads_ ||
            count > rows_.size() / heads_ - first || count > h_)
            throw std::invalid_argument("PLE rolling: gather must advance consecutive bounded windows");
        current = window(first, count);
        if (current.size() > reserve_ / page_)
            throw std::invalid_argument("PLE rolling: use split gather_range for this cap");
        if (!make_room(current, cap_)) throw std::logic_error("PLE rolling: demand reserve occupied by outstanding readers");
        retain(current);
        gathering_ = true;
    }
    try {
        // Owned reads complete before the byte-copy callback. Shared in-flight
        // lookahead reads are joined, never issued twice for a live page.
        DirectActivity activity;
        {
            const DirectReadScope scope(DirectReadScope::background(), &activity);
            if (io_.prepare) io_.prepare(current);
        }
        if (activity.bytes) { const std::lock_guard lock(mutex_); io_records_.push_back({first, count, activity}); }
        if (io_.missing) {
            const auto missing = io_.missing(current);
            const std::lock_guard lock(mutex_); stats_.faulted_pages += missing;
        }
        copy();
        const std::lock_guard lock(mutex_);
        frontier_ = std::max(frontier_, first + count);
        consumed();
        release(current);
        gathering_ = false;
        ++stats_.epochs;
        sample();
        wake_.notify_all();
    } catch (...) {
        const std::lock_guard lock(mutex_);
        error_ = std::current_exception();
        if (gathering_) { gathering_ = false; release(current); }
        wake_.notify_all();
        throw;
    }
}
void RollingPipeline::source(std::span<const std::uint64_t> rows, const std::function<void()>& copy) {
    const auto p = pages(rows);
    bool diagnose = false;
    {
        const std::lock_guard lock(mutex_);
        check();
        std::uint64_t extra = 0;
        for (auto pg : p) extra += !refs_.contains(pg);
        if (io_.owned) {
            if (extra > decode_allowance_ / page_)
                throw std::runtime_error("PLE rolling: source wave exceeds owned decode quota");
            while (extra > decode_allowance_ / page_ - decode_.size()) {
                auto victim = decode_lru_.end();
                for (auto it = decode_lru_.end(); it != decode_lru_.begin();) {
                    --it;
                    if (refs_.at(*it) == 1U && !std::binary_search(p.begin(), p.end(), *it)) { victim = it; break; }
                }
                if (victim == decode_lru_.end()) throw std::runtime_error("PLE rolling: owned decode quota held by readers");
                const auto pg = *victim;
                decode_lru_index_.erase(pg); decode_lru_.erase(victim); decode_.erase(pg);
                release(Pages{pg});
                io_.evict(Pages{pg}); cold_.erase(pg); ++stats_.evicted_pages;
            }
        } else if (extra > decode_allowance_ / page_ - decode_.size()) {
            error_ = std::make_exception_ptr(std::runtime_error("PLE rolling: turn decode page allowance exceeded"));
            std::rethrow_exception(error_);
        }
        for (auto pg : p) {
            if (!refs_.contains(pg)) {
                decode_.insert(pg); ++refs_[pg];
                if (io_.owned) { decode_lru_.push_front(pg); decode_lru_index_[pg] = decode_lru_.begin(); }
            } else if (io_.owned && decode_.contains(pg)) {
                decode_lru_.splice(decode_lru_.begin(), decode_lru_, decode_lru_index_.at(pg));
            }
        }
        retain(p);
        diagnose = !gathering_ && static_cast<bool>(io_.missing);
    }
    try {
        if (io_.prepare) io_.prepare(p);
        if (diagnose) {
            const auto missing = io_.missing(p);
            const std::lock_guard lock(mutex_); stats_.faulted_pages += missing;
        }
        copy();
    }
    catch (...) {
        const std::lock_guard lock(mutex_);
        error_ = std::current_exception(); release(p); wake_.notify_all(); throw;
    }
    const std::lock_guard lock(mutex_);
    release(p);
    if (!gathering_ && (++copies_ % 64U == 1U)) sample();
    wake_.notify_all();
}
void RollingPipeline::stop() {
    { const std::lock_guard lock(mutex_); stopping_ = true; }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
}
void RollingPipeline::finish() {
    stop();
    const std::lock_guard lock(mutex_);
    if (finished_) { if (error_) std::rethrow_exception(error_); return; }
    if (gathering_) throw std::logic_error("PLE rolling: turn ended with CPU readers");
    while (!lookahead_.empty()) { release(lookahead_.front().pages); lookahead_.pop_front(); }
    release(Pages(decode_.begin(), decode_.end())); decode_.clear(); decode_lru_.clear(); decode_lru_index_.clear();
    if (!refs_.empty()) throw std::logic_error("PLE rolling: turn ended with source leases");
    if (!cold_.empty()) {
        const Pages dead(cold_.begin(), cold_.end());
        try { io_.evict(dead); }
        catch (...) { error_ = std::current_exception(); throw; }
        stats_.evicted_pages += dead.size(); cold_.clear();
    }
    finished_ = true;
    if (error_) std::rethrow_exception(error_);
    sample();
}
RollingPipeline::BlockIo RollingPipeline::block_io(std::uint64_t first, std::uint64_t count) const {
    std::unique_lock lock(mutex_);
    wake_.wait(lock, [&] {
        return error_ || std::none_of(io_records_.begin(), io_records_.end(), [&](const auto& record) {
            return !record.complete && record.first < first + count && record.first + record.count > first;
        });
    });
    BlockIo result;
    for (const auto& record : io_records_) {
        const auto& a = record.activity;
        if (record.first >= first + count) { result.lookahead_bytes += a.bytes; continue; }
        if (record.first + record.count <= first) continue;
        result.critical.bytes += a.bytes; result.critical.reads += a.reads;
        result.critical.wait_ns += a.wait_ns;
        if (a.begin_ns && (!result.critical.begin_ns || a.begin_ns < result.critical.begin_ns))
            result.critical.begin_ns = a.begin_ns;
        result.critical.end_ns = std::max(result.critical.end_ns, a.end_ns);
    }
    return result;
}
RollingStats RollingPipeline::stats() const { const std::lock_guard lock(mutex_); return stats_; }
bool RollingPipeline::healthy() const { const std::lock_guard lock(mutex_); return !error_; }
} // namespace strix::ple_reclaim


namespace strix::ple_reclaim {
namespace {
std::mutex authorities_mutex;
std::set<std::string> authorities;
}
ResidencyManager::ResidencyManager(std::string backing, std::uintptr_t table, std::uint64_t generation,
    std::uint64_t h, std::uint64_t page, std::uint64_t offset, std::uint64_t row_bytes,
    std::uint64_t row_count, std::uint32_t heads, std::uint64_t admitted,
    std::uint64_t quota, std::uint32_t slots, RollingPages::Io io)
    : backing_(std::move(backing)), table_(table), generation_(generation), h_(h), page_(page), offset_(offset),
      row_bytes_(row_bytes), row_count_(row_count), admitted_(admitted), quota_(quota), heads_(heads), slots_(slots), io_(std::move(io)) {
    if (backing_.empty() || !table_ || !generation_ || !h_ || !page_ || !row_bytes_ || !row_count_ ||
        !heads_ || !slots_ || quota_ < page_ || offset_ > UINT64_MAX - row_bytes_ ||
        row_count_ > (UINT64_MAX - offset_) / row_bytes_ || h_ > UINT64_MAX / heads_ ||
        !io_.evict || !io_.resident || !io_.reclaimable)
        throw std::invalid_argument("PLE residency manager configuration incomplete");
    const std::lock_guard lock(authorities_mutex);
    if (!authorities.insert(backing_).second)
        throw std::runtime_error("PLE residency: backing already has an eviction authority");
}
ResidencyManager::~ResidencyManager() {
    const std::lock_guard lock(authorities_mutex);
    authorities.erase(backing_);
}
void ResidencyManager::invalidate() noexcept { const std::lock_guard lock(mutex_); failed_ = true; }
bool ResidencyManager::healthy() const {
    const std::lock_guard lock(mutex_);
    if (failed_) return false;
    for (const auto& [id, s] : turns_) { (void)id; if (s.pipeline && !s.pipeline->healthy()) return false; }
    return true;
}
namespace {
void merge_prompt_stats(RollingStats& a, const RollingStats& b) {
    a.epochs += b.epochs; a.prefetched_pages += b.prefetched_pages; a.evicted_pages += b.evicted_pages;
    a.faulted_pages += b.faulted_pages; a.throttles += b.throttles; a.split_windows += b.split_windows;
    a.peak_owned_bytes = std::max(a.peak_owned_bytes,b.peak_owned_bytes);
    a.peak_resident_bytes = std::max(a.peak_resident_bytes,b.peak_resident_bytes);
    a.lookahead_positions = std::max(a.lookahead_positions,b.lookahead_positions);
}
}
RollingStats ResidencyManager::stats() const {
    const std::lock_guard lock(mutex_);
    auto result = stats_;
    for (const auto& [id,s] : turns_) { (void)id; if (s.pipeline) merge_prompt_stats(result,s.pipeline->stats()); }
    return result;
}
void ResidencyManager::retire_prompt(State& s) {
    if (!s.pipeline) return;
    std::exception_ptr error;
    try { s.pipeline->finish(); } catch (...) { error = std::current_exception(); }
    merge_prompt_stats(stats_,s.pipeline->stats()); s.pipeline.reset();
    if (error) { failed_ = true; std::rethrow_exception(error); }
}
ResidencyManager::State& ResidencyManager::state(std::uint64_t id) {
    auto it = turns_.find(id);
    if (failed_ || it == turns_.end() || it->second.closing)
        throw std::runtime_error("PLE residency: invalid certificate or retired turn");
    return it->second;
}
std::shared_ptr<ResidencyManager::Turn> ResidencyManager::begin(std::shared_ptr<const void> owner, PromptObserver observer) {
    std::uint64_t id;
    {
        const std::lock_guard lock(mutex_);
        if (failed_ || turns_.size() >= slots_ || sequence_ == UINT64_MAX)
            throw std::runtime_error("PLE residency: no healthy reserved turn slot");
        id = ++sequence_;
        State s; s.observer = std::move(observer);
        turns_.emplace(id, std::move(s));
    }
    try { return std::shared_ptr<Turn>(new Turn(shared_from_this(), id, std::move(owner))); }
    catch (...) { close(id, true); throw; }
}
ResidencyManager::Pages ResidencyManager::pages(std::span<const std::uint64_t> rows) const {
    Pages result;
    for (const auto row : rows) {
        if (row >= row_count_ || row > (UINT64_MAX - offset_ - row_bytes_) / row_bytes_)
            throw std::out_of_range("PLE residency: invalid immutable row");
        const auto first = offset_ + row * row_bytes_;
        for (auto pg = first / page_; pg <= (first + row_bytes_ - 1U) / page_; ++pg)
            if (io_.reclaimable(pg)) result.push_back(pg);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
void ResidencyManager::retain(const Pages& pages) {
    std::size_t acquired = 0;
    try { for (auto pg : pages) { ++refs_[pg]; ++acquired; } }
    catch (...) {
        // No source touch has occurred. Roll back a partially allocated map
        // without allocating another container or dropping a sibling's ref.
        for (std::size_t i = 0; i < acquired; ++i) {
            auto it = refs_.find(pages[i]);
            if (!--it->second) refs_.erase(it);
        }
        failed_ = true; throw;
    }
}
void ResidencyManager::release(const Pages& pages) {
    Pages dead;
    try { dead.reserve(pages.size()); } catch (...) { failed_ = true; throw; }
    for (auto pg : pages) {
        auto it = refs_.find(pg);
        if (it == refs_.end() || !it->second) { failed_ = true; throw std::logic_error("PLE residency: reference underflow"); }
        if (!--it->second) { refs_.erase(it); dead.push_back(pg); }
    }
    if (!dead.empty()) {
        std::sort(dead.begin(), dead.end());
        try { io_.evict(dead); stats_.evicted_pages += dead.size(); }
        catch (...) { failed_ = true; throw; }
    }
}
void ResidencyManager::sample() {
    try {
        auto bytes = io_.resident();
        stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, bytes);
        if (bytes > admitted_) throw std::runtime_error("PLE residency: sampled bound exceeded");
    } catch (...) { failed_ = true; throw; }
}
void ResidencyManager::clear(State& s) {
    Pages cached;
    for (const auto& entry : s.cache) cached.push_back(entry.first);
    s.cache.clear(); s.lru.clear(); // cleanup remains possible after a failed drop/certificate
    auto window = std::move(s.window); s.window.clear();
    std::exception_ptr error;
    try { release(cached); } catch (...) { error = std::current_exception(); }
    try { release(window); } catch (...) { if (!error) error = std::current_exception(); }
    if (error) std::rethrow_exception(error);
}
void ResidencyManager::configure_prompt(PromptConfig config) {
    const std::lock_guard lock(mutex_);
    if (slots_ != 1U || !turns_.empty() || !config.cap || config.rows.empty() || config.rows.size() % heads_)
        throw std::invalid_argument("PLE residency: invalid known prompt pipeline configuration");
    prompt_config_ = config;
}
void ResidencyManager::source(std::uint64_t id, std::uintptr_t table, std::uint64_t generation,
    std::span<const std::uint64_t> rows, const std::function<void()>& copy, std::uint64_t epoch, std::uint64_t* uncached) {
    std::shared_ptr<RollingPipeline> pipeline;
    {
        const std::lock_guard lock(mutex_);
        auto& s = state(id);
        if (table != table_ || generation != generation_ || (epoch && epoch != s.epoch)) {
            failed_ = true;
            throw std::runtime_error("PLE residency: unknown table alias or mapping generation");
        }
        pipeline = s.pipeline;
        if (pipeline) ++s.readers;
    }
    if (pipeline) {
        std::exception_ptr error;
        try { pipeline->source(rows, copy); } catch (...) { error = std::current_exception(); }
        {
            const std::lock_guard lock(mutex_);
            --turns_.at(id).readers;
            if (error) failed_ = true;
            drained_.notify_all();
        }
        if (error) std::rethrow_exception(error);
        return;
    }
    const auto p = pages(rows);
    {
        const std::lock_guard lock(mutex_);
        auto& s = state(id);
        if (table != table_ || generation != generation_ || (epoch && epoch != s.epoch)) {
            failed_ = true;
            throw std::runtime_error("PLE residency: unknown table alias or mapping generation");
        }
        Pages needed;
        for (auto pg : p) if (!s.cache.contains(pg) &&
            !std::binary_search(s.window.begin(), s.window.end(), pg)) needed.push_back(pg);
        const auto capacity = quota_ / page_;
        if (needed.size() > capacity) throw std::runtime_error("PLE residency: source wave exceeds reserved quota");
        if (uncached) *uncached = needed.size();
        const bool refresh = s.cache.size() + needed.size() > capacity;
        while (s.cache.size() + needed.size() > capacity) {
            auto victim = s.lru.end();
            for (auto it = s.lru.end(); it != s.lru.begin();) {
                --it;
                if (!s.sources.contains(*it) && !std::binary_search(p.begin(), p.end(), *it)) { victim = it; break; }
            }
            if (victim == s.lru.end()) throw std::runtime_error("PLE residency: in-flight sources exhaust reserved quota");
            const auto pg = *victim; s.cache.erase(pg); s.lru.erase(victim); release(Pages{pg});
        }
        if (refresh) { ++stats_.epochs; sample(); }
        const auto undo_cache_metadata = [&] {
            for (auto pg : needed) if (auto it = s.cache.find(pg); it != s.cache.end()) {
                s.lru.erase(it->second); s.cache.erase(it);
            }
        };
        try {
            for (auto pg : needed) {
                s.lru.push_front(pg);
                try { s.cache.emplace(pg, s.lru.begin()); }
                catch (...) { s.lru.pop_front(); throw; }
            }
            retain(needed);
        } catch (...) { undo_cache_metadata(); failed_ = true; throw; }
        for (auto pg : p) if (auto it = s.cache.find(pg); it != s.cache.end())
            s.lru.splice(s.lru.begin(), s.lru, it->second);
        std::size_t acquired = 0;
        try {
            for (auto pg : p) { ++s.sources[pg]; ++acquired; }
            retain(p);
        } catch (...) {
            for (std::size_t i = 0; i < acquired; ++i)
                if (!--s.sources.at(p[i])) s.sources.erase(p[i]);
            failed_ = true; throw; // cache refs remain owned and can be drained
        }
        ++s.readers;
        try { if (io_.missing) stats_.faulted_pages += io_.missing(p); }
        catch (...) {
            failed_ = true; --s.readers;
            for (auto pg : p) if (!--s.sources.at(pg)) s.sources.erase(pg);
            drained_.notify_all(); release(p); throw;
        }
    }
    std::exception_ptr error;
    try { if (io_.prepare) io_.prepare(p); copy(); } catch (...) { error = std::current_exception(); }
    {
        const std::lock_guard lock(mutex_);
        auto& s = turns_.at(id); // closing is legal: this reader owns its lease
        if (error) failed_ = true;
        try { release(p); } catch (...) { if (!error) error = std::current_exception(); }
        --s.readers;
        for (auto pg : p) if (!--s.sources.at(pg)) s.sources.erase(pg);
        drained_.notify_all();
        if (!error && (++s.completions % 64U == 1U)) {
            try { sample(); } catch (...) { error = std::current_exception(); }
        }
    }
    if (error) std::rethrow_exception(error);
}
void ResidencyManager::gather(std::uint64_t id, std::span<const std::uint64_t> rows, const std::function<void()>& copy) {
    const auto p = pages(rows);
    {
        const std::lock_guard lock(mutex_);
        auto& s = state(id);
        if (!s.prompt || s.pipeline || permit_ || rows.empty() || rows.size() > h_ * heads_)
            throw std::runtime_error("PLE residency: bounded prefill permit unavailable");
        sample(); s.window = p;
        try { retain(p); } catch (...) { s.window.clear(); throw; }
        permit_ = id;
    }
    std::exception_ptr error;
    try { if (io_.prepare) io_.prepare(p); copy(); } catch (...) { error = std::current_exception(); }
    {
        std::unique_lock lock(mutex_);
        // A caller must join helpers. Also drain retained source callbacks before
        // transferring the single permit, including cancellation/error paths.
        drained_.wait(lock, [&] { return !turns_.at(id).readers; });
        auto& s = turns_.at(id);
        if (error) failed_ = true;
        auto window = std::move(s.window); s.window.clear();
        try { release(window); if (!error) sample(); } catch (...) { if (!error) error = std::current_exception(); }
        permit_ = 0; ++stats_.epochs; drained_.notify_all();
    }
    if (error) std::rethrow_exception(error);
}
void ResidencyManager::phase(std::uint64_t id, bool prompt) {
    std::unique_lock lock(mutex_);
    (void)state(id);
    drained_.wait(lock, [&] {
        const auto it = turns_.find(id);
        return it == turns_.end() || it->second.closing || !it->second.stages;
    });
    auto& s = state(id);
    if (failed_) throw std::runtime_error("PLE residency: invalidated phase transition");
    if (s.readers || permit_ == id) throw std::runtime_error("PLE residency: phase transition requires drained source tasks");
    retire_prompt(s);
    clear(s); sample(); s.prompt = prompt; ++s.epoch; ++stats_.epochs;
    if (prompt && !prompt_config_.rows.empty()) {
        // Calibration pages are retired before a single prompt authority starts.
        // Turn::owner_ retains the row-id storage and mapped backing.
        try { s.pipeline = std::make_shared<RollingPipeline>(h_, page_, offset_, row_bytes_, heads_,
            prompt_config_.rows, admitted_, quota_, prompt_config_.cap, prompt_config_.ahead, io_, prompt_config_.block_rows); }
        catch (...) { failed_ = true; throw; }
    }
}
void ResidencyManager::close(std::uint64_t id, bool cancelled) {
    std::unique_lock lock(mutex_);
    auto it = turns_.find(id);
    if (it == turns_.end()) return;
    if (it->second.closing) {
        drained_.wait(lock, [&] { return !turns_.contains(id); });
        return;
    }
    it->second.closing = true;
    drained_.wait(lock, [&] { return !turns_.at(id).readers && !turns_.at(id).stages && permit_ != id; });
    std::exception_ptr error;
    try { retire_prompt(it->second); }
    catch (...) { error = std::current_exception(); }
    try { clear(it->second); sample(); } catch (...) { if (!error) error = std::current_exception(); }
    lock.unlock();
    try { if (io_.finish) io_.finish(); } catch (...) { if (!error) error = std::current_exception(); }
    lock.lock();
    if (error) failed_ = true;
    turns_.erase(it); ++stats_.epochs; drained_.notify_all();
    if (!cancelled && (error || failed_)) {
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("PLE residency: turn failed; verified credit invalidated");
    }
}
ResidencyManager::Turn::~Turn() { cancel(); }
void ResidencyManager::Turn::cancel() noexcept { try { manager_->close(id_, true); } catch (...) {} }
void ResidencyManager::Turn::finish() { manager_->close(id_, false); }
void ResidencyManager::Turn::observe(std::uint64_t first, std::span<const std::uint64_t> rows,
    std::uint64_t ns, std::uint64_t minflt, std::uint64_t majflt) {
    PromptObserver observer;
    {
        const std::lock_guard lock(manager_->mutex_);
        auto& s = manager_->state(id_);
        if (!s.prompt) throw std::runtime_error("PLE residency: stale prompt observer");
        if (first < manager_->prompt_config_.first_position)
            throw std::out_of_range("PLE observer precedes suffix");
        if (s.pipeline) s.pipeline->advance(first - manager_->prompt_config_.first_position + rows.size() / manager_->heads_);
        observer = s.observer;
    }
    try { if (observer) observer(first, rows, ns, minflt, majflt); }
    catch (...) { manager_->invalidate(); throw; }
}
void ResidencyManager::Turn::phase(bool prompt) { manager_->phase(id_, prompt); }
bool ResidencyManager::Turn::prompt() const {
    const std::lock_guard lock(manager_->mutex_); return manager_->state(id_).prompt;
}
std::uint64_t ResidencyManager::Turn::epoch() const {
    const std::lock_guard lock(manager_->mutex_);
    const auto& s = manager_->turns_.at(id_);
    if (manager_->failed_ || (s.closing && !(s.pipeline && manager_->permit_ == id_)))
        throw std::runtime_error("PLE residency: retired source epoch");
    return s.epoch;
}
void ResidencyManager::Turn::source(std::uintptr_t table, std::uint64_t generation,
    std::span<const std::uint64_t> rows, const std::function<void()>& copy, std::uint64_t epoch) {
    manager_->source(id_, table, generation, rows, copy, epoch);
}
std::uint64_t ResidencyManager::Turn::warm(std::uintptr_t table, std::uint64_t generation, std::span<const std::uint64_t> rows) {
    if (prompt() || rows.empty()) throw std::runtime_error("PLE residency: decode warming requires decode-phase rows");
    std::uint64_t uncached = 0;
    manager_->source(id_, table, generation, rows, [] {}, epoch(), &uncached);
    return uncached;
}
void ResidencyManager::Turn::verify_window(std::uintptr_t table, std::uint64_t generation,
    std::uint64_t positions, std::span<const std::uint64_t> rows, const std::function<void()>& copy) {
    const auto start_epoch = epoch();
    if (prompt() || !positions || positions > maximum_verify_positions || rows.size() != positions * manager_->heads_)
        throw std::runtime_error("PLE residency: verify window exceeds maximum horizon or has incomplete rows");
    // source() admits the entire union against the fixed decode quota before
    // any read. Inner per-row helper leases cannot evict another window row.
    // No absolute-position frontier: rollback may re-read any immutable row.
    source(table, generation, rows, copy, start_epoch);
}
void ResidencyManager::Turn::read_row(std::uint64_t row, std::span<std::byte> out) const {
    if (!manager_->io_.owned || row >= manager_->row_count_ || out.size() != manager_->row_bytes_)
        throw std::runtime_error("PLE O_DIRECT: invalid leased owned row");
    manager_->io_.owned->copy(manager_->offset_ + row * manager_->row_bytes_, out);
}
void ResidencyManager::Turn::gather(std::span<const std::uint64_t> rows, const std::function<void()>& copy) { manager_->gather(id_, rows, copy); }

void ResidencyManager::Turn::stage_begin() {
    const std::lock_guard lock(manager_->mutex_);
    auto& s = manager_->state(id_);
    if (!s.prompt || !s.pipeline) throw std::runtime_error("PLE residency: staging requires prompt phase");
    ++s.stages;
}
void ResidencyManager::Turn::stage_end(std::exception_ptr error) noexcept {
    const std::lock_guard lock(manager_->mutex_);
    auto it = manager_->turns_.find(id_);
    if (it != manager_->turns_.end() && it->second.stages) --it->second.stages;
    if (error) manager_->failed_ = true;
    manager_->drained_.notify_all();
}
void ResidencyManager::Turn::gather_prompt(std::uint64_t first, std::uint64_t count,
    const std::function<void(std::uint64_t, std::uint64_t)>& copy, std::uint64_t staged_epoch) {
    std::shared_ptr<RollingPipeline> pipeline;
    {
        std::unique_lock lock(manager_->mutex_);
        auto& s = manager_->turns_.at(id_);
        const bool registered = staged_epoch && staged_epoch == s.epoch && s.stages;
        if (manager_->failed_ || (s.closing && !registered) || !s.prompt || !s.pipeline ||
            (staged_epoch && !registered))
            throw std::runtime_error("PLE residency: prompt pipeline permit unavailable");
        // Tail owners may have overlapping c1 copies. Serialize their source
        // permits exactly as RollingPipeline does; retirement drains waiters.
        ++s.stages;
        manager_->drained_.wait(lock, [&] { return !manager_->permit_ || manager_->failed_; });
        if (manager_->failed_) {
            --s.stages; manager_->drained_.notify_all();
            throw std::runtime_error("PLE residency: prompt pipeline retired while waiting");
        }
        pipeline = s.pipeline; manager_->permit_ = id_;
    }
    std::exception_ptr error;
    try {
        const auto base = manager_->prompt_config_.first_position;
        if (first < base) throw std::out_of_range("PLE gather precedes suffix");
        pipeline->gather_range(first - base, count, [&](auto p, auto n) { copy(p + base, n); });
    }
    catch (...) { error = std::current_exception(); }
    {
        std::unique_lock lock(manager_->mutex_);
        manager_->drained_.wait(lock, [&] { return !manager_->turns_.at(id_).readers; });
        manager_->permit_ = 0;
        --manager_->turns_.at(id_).stages;
        if (error) manager_->failed_ = true;
        manager_->drained_.notify_all();
    }
    if (error) std::rethrow_exception(error);
}
std::shared_ptr<const DirectPageCache::View> ResidencyManager::Turn::prompt_view(std::uint64_t first, std::uint64_t count) const {
    const std::lock_guard lock(manager_->mutex_);
    auto& s = manager_->turns_.at(id_);
    if (manager_->failed_) throw std::runtime_error("PLE residency: invalidated prompt view");
    const auto base = manager_->prompt_config_.first_position;
    if (first < base) throw std::out_of_range("PLE view precedes suffix");
    first -= base;
    const auto& rows = manager_->prompt_config_.rows;
    const auto heads = manager_->heads_;
    if (!s.prompt || !s.pipeline || manager_->permit_ != id_ || !count || count > manager_->h_ ||
        first > rows.size() / heads || count > rows.size() / heads - first || !manager_->io_.owned)
        throw std::runtime_error("PLE residency: prompt view requires live bounded permit");
    const std::vector<std::uint64_t> ids(rows.begin() + static_cast<std::ptrdiff_t>(first * heads),
        rows.begin() + static_cast<std::ptrdiff_t>((first + count) * heads));
    return manager_->io_.owned->view(manager_->pages(ids));
}
bool ResidencyManager::Turn::owns_prompt_view(const DirectPageCache::View& view, std::uintptr_t table,
    std::uint64_t generation, std::uint64_t epoch, std::uint64_t offset) const {
    const std::lock_guard lock(manager_->mutex_);
    auto& s = manager_->turns_.at(id_);
    return !manager_->failed_ && s.prompt && s.pipeline && manager_->permit_ == id_ && table == manager_->table_ &&
        generation == manager_->generation_ && epoch == s.epoch && offset == manager_->offset_ && manager_->io_.owned && manager_->io_.owned->owns(view);
}
} // namespace strix::ple_reclaim
