#include "strix/startup_markers.hpp"
#include "strix/ple_reclaim.hpp"
#include "strix/qpf2k_defaults.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#if defined(__linux__)
#include <glob.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strix::ple_reclaim {
namespace {
std::atomic<Prefetcher*> active_prefetcher{nullptr};

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("PLE reclaim byte accounting overflow");
    return a + b;
}
[[maybe_unused]] std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::overflow_error("PLE reclaim byte accounting overflow");
    return a * b;
}
[[maybe_unused]] std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
[[maybe_unused]] bool test_bit(const std::vector<std::uint64_t>& bits, std::uint64_t i) noexcept {
    return (bits[static_cast<std::size_t>(i >> 6U)] >> (i & 63U)) & 1U;
}
[[maybe_unused]] void set_bit(std::vector<std::uint64_t>& bits, std::uint64_t i) noexcept {
    bits[static_cast<std::size_t>(i >> 6U)] |= std::uint64_t{1} << (i & 63U);
}
// Sum of this process's KFD queue eviction time over its GPUs, -1 if unreadable (no KFD or no stats).
std::int64_t kfd_evicted_ms() {
#if defined(__linux__)
    const auto pattern = "/sys/class/kfd/kfd/proc/" + std::to_string(::getpid()) + "/stats_*/evicted_ms";
    glob_t found{};
    if (::glob(pattern.c_str(), 0, nullptr, &found) != 0) { ::globfree(&found); return -1; }
    std::int64_t total = 0;
    bool any = false;
    for (std::size_t i = 0; i < found.gl_pathc; ++i) {
        std::ifstream in(found.gl_pathv[i]);
        long long value = 0;
        if (in >> value) { total += value; any = true; }
    }
    ::globfree(&found);
    return any ? total : -1;
#else
    return -1;
#endif
}
}  // namespace

Mode requested() noexcept {
    if (qpf2k::suppressed(qpf2k::Item::ple_reclaim)) return Mode::off;
    const char* v = std::getenv("STRIX_PLE_RECLAIMABLE");
    if (v) {
        if (std::strcmp(v, "1") == 0) return Mode::required;
        return std::strcmp(v, "auto") == 0 ? Mode::automatic : Mode::off;
    }
    // Also honor the defaults policy when no startup injection has run (host callers/tests).
    const char* defaults = std::getenv("STRIX_QPF2K_DEFAULTS");
    return qpf2k::gsq_model || (defaults && defaults[0] == '0') ? Mode::off : Mode::automatic;
}

void set_active(Prefetcher* prefetcher) noexcept { active_prefetcher.store(prefetcher, std::memory_order_release); }
Prefetcher* active() noexcept { return active_prefetcher.load(std::memory_order_acquire); }

std::vector<Interval> reclaim_intervals(std::uint64_t table_offset, std::uint64_t table_bytes, std::uint64_t page,
    std::vector<Interval> registered) {
    if (!page || (page & (page - 1U))) throw std::invalid_argument("PLE reclaim page size must be a power of two");
    const auto end = checked_add(table_offset, table_bytes);
    const auto first = checked_add(table_offset, page - 1U) & ~(page - 1U);
    const auto last = end & ~(page - 1U);
    std::vector<Interval> result;
    if (last <= first) return result;
    for (auto& r : registered) {
        if (r.end < r.begin) throw std::invalid_argument("PLE reclaim registered interval is inverted");
        r.begin &= ~(page - 1U);
        r.end = checked_add(r.end, page - 1U) & ~(page - 1U);
    }
    std::sort(registered.begin(), registered.end(), [](const Interval& a, const Interval& b) { return a.begin < b.begin; });
    auto at = first;
    for (const auto& r : registered) {
        if (r.end <= at || r.begin >= last) continue;
        if (r.begin > at) result.push_back({at, r.begin});
        at = std::max(at, r.end);
        if (at >= last) break;
    }
    if (at < last) result.push_back({at, last});
    return result;
}

std::uint64_t decode_allowance(std::uint64_t generate_tokens, std::uint64_t lookups_per_token,
    std::uint64_t heads, std::uint64_t page, std::uint64_t requested_bytes) {
    const auto lookups = checked_mul(checked_add(generate_tokens, 1U), checked_mul(lookups_per_token, heads));
    return std::max<std::uint64_t>({256ULL << 20U, checked_mul(lookups, checked_mul(2U, page)), requested_bytes});
}

PromptPlan plan_prompt(const Config& config, const PleHashContract& contract,
    std::span<const std::uint32_t> prompt) {
    const auto page = config.page;
    if (!page || (page & (page - 1U)) || !config.stored_row_bytes || !config.table_rows ||
        config.table_rows > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("PLE reclaim configuration is incomplete");
    PromptPlan result;
    const auto in_reclaim = [&](std::uint64_t pg) {
        const auto byte = pg * page;
        const auto it = std::upper_bound(config.reclaim.begin(), config.reclaim.end(), byte,
            [](std::uint64_t value, const Interval& interval) { return value < interval.begin; });
        return it != config.reclaim.begin() && byte < std::prev(it)->end;
    };
    const auto table_bytes = checked_mul(config.table_rows, config.stored_row_bytes);
    if (checked_add(config.table_offset, table_bytes) > config.mapping_bytes)
        throw std::invalid_argument("PLE reclaim table exceeds the model mapping");
    if (prompt.empty()) throw std::invalid_argument("PLE reclaim requires the known prompt tokens");
    if (!contract.head_count || contract.head_count > PleHashContract::maximum_heads || contract.ngram_size < 2 ||
        contract.ngram_size > PleHashContract::maximum_ngram)
        throw std::invalid_argument("PLE reclaim hash contract is invalid");
    std::uint64_t reclaimable = 0;
    for (std::size_t i = 0; i < config.reclaim.size(); ++i) {
        const auto& r = config.reclaim[i];
        if (r.begin >= r.end || r.begin % page || r.end % page || r.end > config.mapping_bytes ||
            r.begin < config.table_offset || r.end > config.table_offset + table_bytes ||
            (i && r.begin < config.reclaim[i - 1].end))
            throw std::invalid_argument("PLE reclaim intervals must be sorted page-aligned table interior ranges");
        reclaimable = checked_add(reclaimable, r.end - r.begin);
    }
    if (config.reclaim.empty()) throw std::invalid_argument("PLE reclaim table has no unregistered interior page");
    const auto first_page = config.reclaim.front().begin / page;
    const auto last_page = config.reclaim.back().end / page;
    const auto span_pages = last_page - first_page;
    std::vector<std::uint64_t> touched(static_cast<std::size_t>((span_pages + 63U) / 64U), 0U);

    // The launcher's hash chain (hip_ple_launcher.hip launch: position 0 starts from EOS context, then next_context).
    const auto heads = contract.head_count;
    const auto positions = prompt.size();
    result.rows.resize(static_cast<std::size_t>(checked_mul(positions, heads)));
    const auto context_count = static_cast<std::size_t>(contract.ngram_size - 1U);
    std::array<std::uint32_t, PleHashContract::maximum_ngram - 1> previous{};
    std::fill_n(previous.begin(), context_count, contract.eos_token_id);
    std::uint64_t touched_pages = 0;
    for (std::uint64_t p = 0; p < positions; ++p) {
        const auto hash = hash_ple_token(contract, prompt[static_cast<std::size_t>(p)],
            std::span<const std::uint32_t>(previous.data(), context_count));
        if (hash.row_count != heads) throw std::logic_error("PLE reclaim hash row count differs from the contract");
        for (std::uint32_t h = 0; h < heads; ++h) {
            const auto row = hash.row_ids[h];
            if (row >= config.table_rows) throw std::out_of_range("PLE reclaim row id exceeds the table");
            result.rows[static_cast<std::size_t>(p * heads + h)] = static_cast<std::uint32_t>(row);
            const auto begin = config.table_offset + row * config.stored_row_bytes;
            for (auto pg = begin / page; pg <= (begin + config.stored_row_bytes - 1U) / page; ++pg) {
                if (pg < first_page || pg >= last_page || !in_reclaim(pg)) continue;
                if (!test_bit(touched, pg - first_page)) { set_bit(touched, pg - first_page); ++touched_pages; }
            }
        }
        std::copy_n(hash.next_context.begin(), context_count, previous.begin());
    }
    if (config.rolling_positions && (config.rolling_positions < 256 || config.rolling_positions > 65536 ||
        (config.rolling_positions & (config.rolling_positions - 1U))))
        throw std::invalid_argument("PLE rolling positions must be a power of two in 256..65536");
    const auto budget = config.rolling_positions ? rolling_budget(config, heads) : RollingBudget{};
    const auto touched_bytes = config.rolling_positions ? budget.cap : checked_mul(touched_pages, page);
    const auto margin = checked_add(64ULL << 20U, touched_bytes / 64U);
    const auto admitted = config.rolling_positions ? budget.admitted
        : checked_add(checked_add(touched_bytes, config.decode_allowance_bytes), margin);
    const auto credit = reclaimable > admitted ? reclaimable - admitted : 0U;
    result.stats.reclaimable_bytes = reclaimable;
    result.stats.touched_bytes = touched_bytes;
    result.stats.admitted_bytes = admitted;
    result.stats.credit_bytes = credit;

    return result;
}

std::uint64_t rolling_cap_bytes(const char* value) {
    if (!value) return 3ULL << 30U;
    std::uint64_t bytes = 0;
    const std::string_view text(value);
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), bytes);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !bytes)
        throw std::invalid_argument("STRIX_PLE_ROLLING_CAP_BYTES must be a positive byte count");
    return bytes;
}
RollingBudget rolling_budget(const Config& c, std::uint64_t heads) {
    if (!c.page || !heads || !c.stored_row_bytes) throw std::invalid_argument("PLE rolling budget geometry incomplete");
    const auto row_pages = checked_add(c.stored_row_bytes, checked_mul(2U, c.page) - 2U) / c.page;
    const auto position = checked_mul(heads, checked_mul(row_pages, c.page));
    const auto cap = c.rolling_cap_bytes ? c.rolling_cap_bytes : rolling_cap_bytes(nullptr);
    if (cap % c.page || cap < checked_mul(2U, position))
        throw std::invalid_argument("PLE rolling cap must be page aligned and hold two worst-case source positions");
    if (c.bounded_turns && cap < checked_mul(c.rolling_positions, position))
        throw std::invalid_argument("PLE rolling MTP cap must hold one worst-case H-sized source wave");
    if (c.decode_allowance_bytes % c.page)
        throw std::invalid_argument("PLE rolling decode allowance must be page aligned");
    const auto margin = checked_add(64ULL << 20U, cap / 64U);
    return {cap, margin, checked_add(checked_add(cap, c.decode_allowance_bytes), margin)};
}

bool Prefetcher::in_reclaim(std::uint64_t page_index) const noexcept {
    const auto byte = page_index * config_.page;
    const auto it = std::upper_bound(config_.reclaim.begin(), config_.reclaim.end(), byte,
        [](std::uint64_t value, const Interval& interval) { return value < interval.begin; });
    return it != config_.reclaim.begin() && byte < std::prev(it)->end;
}

std::uint64_t owner_bytes(const Config& config, std::uint64_t heads,
    std::uint64_t row_capacity, std::uint64_t issued_capacity) {
    auto bytes = checked_add(checked_mul(row_capacity, sizeof(std::uint32_t)),
        checked_mul(issued_capacity, sizeof(std::uint64_t)));
    if (!config.rolling_positions) return bytes;
    const auto budget = rolling_budget(config, heads);
    bytes = checked_add(bytes, checked_add(checked_mul(
        checked_add(budget.cap, config.decode_allowance_bytes) / config.page, 512U), 2U << 20U));
    if (config.bounded_turns) bytes = checked_add(bytes, checked_add(
        checked_mul(budget.admitted / config.page, 256U), 2U << 20U));
    return bytes;
}
std::uint64_t planned_owner_bytes(const Config& config, std::uint64_t heads, std::uint64_t positions) {
    if (config.reclaim.empty() || !config.page) throw std::invalid_argument("PLE owner reclaim geometry incomplete");
    const auto pages = (config.reclaim.back().end - config.reclaim.front().begin) / config.page;
    return owner_bytes(config, heads, checked_mul(positions, heads), checked_add(pages, 63U) / 64U);
}

Prefetcher::Prefetcher(Config config, const PleHashContract& contract, std::span<const std::uint32_t> prompt,
    bool automatic)
    : config_(std::move(config)) {
#if !defined(__linux__)
    static_cast<void>(contract); static_cast<void>(prompt); static_cast<void>(automatic);
    throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1 requires Linux (madvise/mincore/fadvise)");
#else
    const auto page = config_.page;
    if (!config_.mapping_base || !page || (page & (page - 1U)) || config_.mapping_base % page || !config_.stored_row_bytes ||
        !config_.table_rows || config_.table_rows > std::numeric_limits<std::uint32_t>::max() || !config_.drop_page_cache)
        throw std::invalid_argument("PLE reclaim configuration is incomplete");
    if (config_.bounded_turns && !config_.rolling_positions)
        throw std::invalid_argument("PLE bounded turn leases require rolling admission");
    startup::Phase hash_marker(config_.credit_only ? "preflight_ple_hash" : "runtime_ple_hash");
    auto plan = plan_prompt(config_, contract, prompt);
    hash_marker.end();
    heads_ = contract.head_count;
    positions_ = prompt.size();
    rows_ = std::move(plan.rows);
    stats_ = plan.stats;
    admitted_ = stats_.admitted_bytes;
    credit_ = stats_.credit_bytes;
    first_page_ = config_.reclaim.front().begin / page;
    last_page_ = config_.reclaim.back().end / page;
    issued_.assign(static_cast<std::size_t>((last_page_ - first_page_ + 63U) / 64U), 0U);

    if (config_.rolling_positions) {
        if (!config_.open_direct_read_only && !config_.test_direct_read)
            throw std::invalid_argument("PLE O_DIRECT: no backing descriptor; rolling refused");
        const auto budget = rolling_budget(config_, heads_);
        direct_ = std::make_shared<DirectPageCache>(config_.open_direct_read_only ? config_.open_direct_read_only() : -1,
            config_.mapping_bytes, page, budget.cap + config_.decode_allowance_bytes, config_.test_direct_read, DirectPageCache::TestRing{0, false}, config_.credit_only);
    }
    if (direct_) std::fprintf(stderr, "ple-direct reader: backend=%s queue-depth=%u max-run-bytes=131072 fallback-errno=%d\n",
        direct_->reader(), std::strcmp(direct_->reader(), "io_uring") == 0 ? 512U : 32U, direct_->fallback_errno());
    const auto owned_io = [this] {
        RollingPages::Io io;
        // Include both table boundary pages. Protected mapping pages are never
        // touched by these callbacks: the source is a separate anonymous arena.
        io.reclaimable = [](auto) { return true; };
        io.owned = direct_;
        io.resident = [] { return std::uint64_t{0}; }; // file-cache contribution is zero
        io.evict = [cache = direct_](auto pages) { cache->erase(pages); };
        io.prefetch = [cache = direct_](auto pages) { return cache->ensure(pages, true, false); };
        io.critical = [cache = direct_](auto pages) { return cache->ensure(pages, true, true); };
        io.prepare = [cache = direct_](auto pages) { (void)cache->ensure(pages, DirectReadScope::background()); };
        io.finish = [this] {
            bool error = false;
            const auto resident = resident_bytes(&error);
            if (error || resident > config_.drop_residual_bytes)
                throw std::runtime_error("PLE O_DIRECT: end-of-turn mapping residency verification failed");
            if (const auto* v = std::getenv("STRIX_PLE_ROLLING_STATS"); v && std::strcmp(v, "1") == 0)
                std::fprintf(stderr, "ple-direct turn: mapping-resident-pages-end=%llu\n",
                    static_cast<unsigned long long>(resident / config_.page));
        };
        return io;
    };
    // Claim authority before the setup drop. Leases retain this owned reader.
    if (config_.bounded_turns) {
        manager_ = std::make_shared<ResidencyManager>(config_.backing_identity,
            config_.mapping_base + config_.table_offset, config_.mapping_generation,
            config_.rolling_positions, config_.page, config_.table_offset, config_.stored_row_bytes,
            config_.table_rows, heads_, admitted_, config_.decode_allowance_bytes, 1U, owned_io());
    }
    // Drop before the large allocations: never leave the choice of victim pages to kernel reclaim (registered model
    // pages are unpinned userptr pages; reclaiming one evicts every queue of the process).
    startup::Phase drop_marker(config_.credit_only ? "preflight_ple_drop" : "runtime_ple_drop");
    bool failed = false;
    stats_.kfd_evicted_ms_before_drop = kfd_evicted_ms();
    stats_.resident_before_drop_bytes = resident_bytes(&failed);
    for (const auto& r : config_.reclaim) {
        auto* address = reinterpret_cast<void*>(config_.mapping_base + static_cast<std::uintptr_t>(r.begin));
        const auto length = static_cast<std::size_t>(r.end - r.begin);
        (void)madvise(address, length, MADV_RANDOM);
        if (madvise(address, length, MADV_DONTNEED) != 0)
            throw std::runtime_error("PLE reclaim: MADV_DONTNEED on the table interior failed");
        if (const int error = config_.drop_page_cache(r.begin, r.end - r.begin))
            throw std::runtime_error(std::string("PLE reclaim: page-cache drop (posix_fadvise DONTNEED) failed: ") +
                std::strerror(error));
    }
    stats_.resident_after_drop_bytes = resident_bytes(&failed);
    stats_.kfd_evicted_ms_after_drop = kfd_evicted_ms();
    stats_.kfd_evicted_during_drop = stats_.kfd_evicted_ms_before_drop >= 0 &&
        stats_.kfd_evicted_ms_after_drop > stats_.kfd_evicted_ms_before_drop;
    stats_.mincore_failed = failed;
    drop_marker.end();
    startup::Phase credit_marker("credit_verification");
    if (!automatic && !config_.rolling_positions) report_setup();
    // The credit rests on the drop: refuse unless mincore is authoritative and shows the interior (nearly) empty.
    // Linux reports every page resident to a caller that neither owns nor may write the file, and pages another
    // process still maps survive POSIX_FADV_DONTNEED; both land here.
    if (failed)
        throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1: mincore on the PLE table failed; the drop cannot be verified");
    if (stats_.resident_after_drop_bytes > config_.drop_residual_bytes)
        throw std::invalid_argument("STRIX_PLE_RECLAIMABLE=1: " + std::to_string(stats_.resident_after_drop_bytes) +
            " B of the PLE table stay resident after the drop (limit " + std::to_string(config_.drop_residual_bytes) +
            " B): another process maps the model, or mincore is not authoritative (file neither owned nor writable)");
    credit_marker.end();
    if (config_.credit_only) return; // authority/probes/drop verified; runtime activates its fresh owner
    if (manager_) {
        const auto budget = rolling_budget(config_, heads_);
        manager_->configure_prompt({rows_, budget.cap, config_.ahead_rows, config_.critical_rows});
        std::fprintf(stderr, "ple-rolling: activated MTP-CLI H=%llu bound=%llu B verified-credit=%llu B "
            "resident-after-drop=%llu B slots=1 decode-quota=%llu B async-prompt (O_DIRECT owned CPU sources)\n",
            static_cast<unsigned long long>(config_.rolling_positions),
            static_cast<unsigned long long>(admitted_ + config_.table_rows * config_.stored_row_bytes - stats_.reclaimable_bytes),
            static_cast<unsigned long long>(credit_), static_cast<unsigned long long>(stats_.resident_after_drop_bytes),
            static_cast<unsigned long long>(config_.decode_allowance_bytes));
        return;
    }
    if (config_.rolling_positions) {
        auto io = owned_io();
        const auto budget = rolling_budget(config_, heads_);
        rolling_ = std::make_unique<RollingPipeline>(config_.rolling_positions, page, config_.table_offset,
            config_.stored_row_bytes, heads_, rows_, admitted_, config_.decode_allowance_bytes, budget.cap, config_.ahead_rows, std::move(io), config_.critical_rows);
        std::fprintf(stderr, "ple-rolling: activated H=%llu bound=%llu B verified-credit=%llu B "
            "resident-after-drop=%llu B kfd-evicted-during-drop=%u cap=%llu B lookahead=%llu positions (O_DIRECT owned async CPU sources)\n",
            static_cast<unsigned long long>(config_.rolling_positions), static_cast<unsigned long long>(admitted_ +
                config_.table_rows * config_.stored_row_bytes - stats_.reclaimable_bytes),
            static_cast<unsigned long long>(credit_), static_cast<unsigned long long>(stats_.resident_after_drop_bytes),
            stats_.kfd_evicted_during_drop ? 1U : 0U, static_cast<unsigned long long>(budget.cap),
            static_cast<unsigned long long>(std::max({config_.ahead_rows, config_.rolling_positions, config_.critical_rows})));
        return;
    }
    target_ = std::min(positions_, config_.ahead_rows);
    const std::lock_guard<std::mutex> lock(mutex_);
    thread_ = std::thread([this] { helper(); });
    if (automatic) report_setup();
#endif
}

std::shared_ptr<ResidencyManager::Turn> Prefetcher::begin_turn(ResidencyManager::PromptObserver observer) {
    if (config_.credit_only) throw std::logic_error("PLE reclaim: credit-only owner has no prompt turn");
    if (!manager_) return nullptr;
    if (!observer) {
        observer = [this](std::uint64_t first, std::span<const std::uint64_t> ids,
            std::uint64_t ns, std::uint64_t minflt, std::uint64_t majflt) {
            if (ids.size() % heads_) throw std::runtime_error("PLE residency: incomplete prompt source receipt");
            std::vector<PleHashResult> hashes(ids.size() / heads_);
            for (std::size_t i = 0; i < hashes.size(); ++i) {
                hashes[i].row_count = heads_;
                std::copy_n(ids.begin() + static_cast<std::ptrdiff_t>(i * heads_), heads_, hashes[i].row_ids.begin());
            }
            GatherReceipt receipt; receipt.gather_ns = ns; receipt.minflt = minflt; receipt.majflt = majflt;
            on_gather(first, hashes, receipt);
        };
    }
    return manager_->begin(shared_from_this(), std::move(observer));
}

bool Prefetcher::block_stats() const noexcept {
    const auto* v = std::getenv("STRIX_PLE_ROLLING_STATS");
    return rolling_ && v && std::strcmp(v, "1") == 0;
}
Prefetcher::PromptSource Prefetcher::prompt_source(std::uint64_t first, std::uint64_t count) const {
    if (!rolling_ || first > positions_ || !count || count > positions_ - first || count > config_.rolling_positions)
        throw std::logic_error("PLE direct prompt view requires a bounded rolling window");
    std::vector<std::uint64_t> pages;
    pages.reserve(static_cast<std::size_t>(count * heads_ * 2U));
    for (auto i = first * heads_; i < (first + count) * heads_; ++i) {
        const auto begin = config_.table_offset + rows_[i] * config_.stored_row_bytes;
        for (auto pg = begin / config_.page; pg <= (begin + config_.stored_row_bytes - 1U) / config_.page; ++pg)
            pages.push_back(pg);
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    return {direct_->view(pages), config_.mapping_base + config_.table_offset, config_.table_offset};
}
void Prefetcher::gather_window(std::uint64_t first, std::uint64_t count, const std::function<void()>& copy) {
    if (config_.credit_only) throw std::logic_error("PLE reclaim: credit-only owner cannot read prompt sources");
    if (rolling_) rolling_->gather(first, count, copy); else copy();
}
void Prefetcher::gather_range(std::uint64_t first, std::uint64_t count,
    const std::function<void(std::uint64_t, std::uint64_t)>& copy) {
    if (config_.credit_only) throw std::logic_error("PLE reclaim: credit-only owner cannot read prompt sources");
    if (rolling_) rolling_->gather_range(first, count, copy); else copy(first, count);
}
void Prefetcher::source_copy(std::uintptr_t table, std::span<const std::uint64_t> rows, const std::function<void()>& copy) {
    if (config_.credit_only) throw std::logic_error("PLE reclaim: credit-only owner cannot read prompt sources");
    if (manager_) {
        manager_->invalidate();
        throw std::runtime_error("PLE residency: missing explicit source turn binding");
    }
    if (rolling_ && table == config_.mapping_base + config_.table_offset) rolling_->source(rows, copy);
    else if (rolling_) throw std::runtime_error("PLE O_DIRECT: unknown source mapping; rolling refused");
    else copy();
}

void Prefetcher::read_row(std::uintptr_t table, std::uint64_t row, std::span<std::byte> out) const {
    if (!direct_ || table != config_.mapping_base + config_.table_offset || row >= config_.table_rows ||
        out.size() != config_.stored_row_bytes)
        throw std::runtime_error("PLE O_DIRECT: invalid owned source row");
    direct_->copy(config_.table_offset + row * config_.stored_row_bytes, out);
}

void Prefetcher::report_setup() const noexcept {
    if (stats_.kfd_evicted_during_drop)
        std::fprintf(stderr, "ple-reclaim: KFD queue eviction time rose across the PLE drop (%lld -> %lld ms)\n",
            static_cast<long long>(stats_.kfd_evicted_ms_before_drop), static_cast<long long>(stats_.kfd_evicted_ms_after_drop));
    std::fprintf(stderr, "ple-reclaim: reclaimable %llu B, prompt %llu tokens touch %llu pages (%llu B), admitted %llu B, "
        "credit %llu B, resident %llu -> %llu B after drop%s\n",
        static_cast<unsigned long long>(stats_.reclaimable_bytes), static_cast<unsigned long long>(positions_),
        static_cast<unsigned long long>(stats_.touched_bytes / config_.page), static_cast<unsigned long long>(stats_.touched_bytes),
        static_cast<unsigned long long>(admitted_), static_cast<unsigned long long>(credit_),
        static_cast<unsigned long long>(stats_.resident_before_drop_bytes),
        static_cast<unsigned long long>(stats_.resident_after_drop_bytes), stats_.mincore_failed ? " (mincore failed)" : "");
}

Prefetcher::~Prefetcher() {
    (void)drain_stages();
    if (active() == this) set_active(nullptr);
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
}

std::uint64_t Prefetcher::resident_bytes(bool* failed) const {
#if defined(__linux__)
    if (direct_) {
        // One table-wide observation only at setup and request end. Include
        // shared boundary pages, which setup intentionally does not evict.
        const auto first = config_.table_offset / config_.page * config_.page;
        const auto end = config_.table_offset + config_.table_rows * config_.stored_row_bytes;
        const auto pages = (end - first + config_.page - 1U) / config_.page;
        std::vector<unsigned char> vec(static_cast<std::size_t>(pages));
        if (::mincore(reinterpret_cast<void*>(config_.mapping_base + first),
                static_cast<std::size_t>(end - first), vec.data())) {
            if (failed) *failed = true;
            return 0;
        }
        return static_cast<std::uint64_t>(std::count_if(vec.begin(), vec.end(), [](auto v) { return v & 1U; })) * config_.page;
    }
    constexpr std::uint64_t chunk_pages = 1ULL << 20U;
    std::vector<unsigned char> vec(static_cast<std::size_t>(chunk_pages));
    std::uint64_t resident = 0;
    for (const auto& r : config_.reclaim) {
        for (auto at = r.begin; at < r.end; at += chunk_pages * config_.page) {
            const auto bytes = std::min<std::uint64_t>(r.end - at, chunk_pages * config_.page);
            if (mincore(reinterpret_cast<void*>(config_.mapping_base + static_cast<std::uintptr_t>(at)),
                    static_cast<std::size_t>(bytes), vec.data()) != 0) {
                if (failed) *failed = true;
                return 0;
            }
            const auto n = static_cast<std::size_t>(bytes / config_.page);
            for (std::size_t i = 0; i < n; ++i) resident += vec[i] & 1U;
        }
    }
    return resident * config_.page;
#else
    if (failed) *failed = true;
    return 0;
#endif
}

void Prefetcher::prefetch_until(std::uint64_t target) {
#if defined(__linux__)
    const auto t0 = now_ns();
    const auto page = config_.page;
    std::vector<std::uint64_t> pages;
    for (auto p = issued_until_; p < target; ++p)
        for (std::uint32_t h = 0; h < heads_; ++h) {
            const auto begin = config_.table_offset + std::uint64_t{rows_[static_cast<std::size_t>(p * heads_ + h)]} *
                config_.stored_row_bytes;
            for (auto pg = begin / page; pg <= (begin + config_.stored_row_bytes - 1U) / page; ++pg) {
                if (pg < first_page_ || pg >= last_page_ || test_bit(issued_, pg - first_page_) || !in_reclaim(pg)) continue;
                set_bit(issued_, pg - first_page_);
                pages.push_back(pg);
            }
        }
    std::sort(pages.begin(), pages.end());
    std::uint64_t issued = 0;
    std::vector<unsigned char> vec;
    for (std::size_t i = 0; i < pages.size();) {
        auto j = i + 1U;
        while (j < pages.size() && pages[j] == pages[j - 1U] + 1U) ++j;
        const auto count = static_cast<std::size_t>(j - i);
        auto* address = reinterpret_cast<void*>(config_.mapping_base + static_cast<std::uintptr_t>(pages[i] * page));
        const auto length = static_cast<std::size_t>(count * page);
        vec.resize(count);
        std::uint64_t missing = count;
        if (mincore(address, length, vec.data()) == 0) {
            missing = 0;
            for (const auto v : vec) missing += (v & 1U) ? 0U : 1U;
        }
        if (missing) {
            (void)madvise(address, length, MADV_WILLNEED);
            issued += missing;
        }
        i = j;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    stats_.prefetch_pages_considered += pages.size();
    stats_.prefetch_pages_issued += issued;
    stats_.prefetch_ns_total += now_ns() - t0;
#else
    static_cast<void>(target);
#endif
}

void Prefetcher::helper() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return stop_ || target_ > issued_until_ || sample_; });
        if (stop_) return;
        const auto target = target_;
        const bool sample = sample_;
        sample_ = false;
        lock.unlock();
        if (target > issued_until_) prefetch_until(target);   // issued_until_ is written only by this thread
        bool failed = false;
        const auto resident = sample ? resident_bytes(&failed) : 0U;
        lock.lock();
        issued_until_ = std::max(issued_until_, target);
        stats_.prefetched_until = issued_until_;
        if (sample) {
            stats_.mincore_failed = stats_.mincore_failed || failed;
            ++stats_.resident_samples;
            stats_.resident_last_bytes = resident;
            stats_.resident_max_bytes = std::max(stats_.resident_max_bytes, resident);
            if (resident > admitted_ && !stats_.bound_exceeded) {
                stats_.bound_exceeded = true;
                std::fprintf(stderr, "ple-reclaim: resident PLE %llu B exceeds the admitted %llu B\n",
                    static_cast<unsigned long long>(resident), static_cast<unsigned long long>(admitted_));
            }
        }
    }
}

bool Prefetcher::claim_stage_prime() {
    const std::lock_guard lock(mutex_);
    if (!rolling_ || !config_.stage_prefetch || finishing_ || stage_primed_) return false;
    stage_primed_ = true;
    return true;
}
void Prefetcher::stage_begin() {
    const std::lock_guard lock(mutex_);
    if (config_.credit_only) throw std::logic_error("PLE reclaim: credit-only owner cannot stage");
    if (finishing_) throw std::runtime_error("PLE rolling: staged copy started during turn retirement");
    ++stages_;
}
void Prefetcher::stage_end(std::exception_ptr error) noexcept {
    const std::lock_guard lock(mutex_);
    if (error && !stage_error_) stage_error_ = error;
    if (stages_) --stages_;
    wake_.notify_all();
}
std::exception_ptr Prefetcher::drain_stages() {
    std::unique_lock lock(mutex_);
    finishing_ = true;
    wake_.wait(lock, [&] { return !stages_; });
    return stage_error_;
}

std::uint64_t Prefetcher::stage_rows(std::uint64_t first, std::uint64_t max_count, std::vector<std::uint32_t>& out) const {
    if (config_.credit_only || !config_.stage_prefetch || !max_count || first >= positions_) return 0;
    const auto count = std::min(max_count, positions_ - first);
    const auto begin = rows_.begin() + static_cast<std::ptrdiff_t>(first * heads_);
    out.assign(begin, begin + static_cast<std::ptrdiff_t>(count * heads_));
    return count;
}

void Prefetcher::on_gather(std::uint64_t first_position, std::span<const PleHashResult> hashes, const GatherReceipt& receipt) {
    const auto gather_ns = receipt.gather_ns, minflt = receipt.minflt, majflt = receipt.majflt;
    std::uint64_t mismatches = 0, outside = 0;
    for (std::size_t r = 0; r < hashes.size(); ++r) {
        const auto p = first_position + r;
        if (p >= positions_) { ++outside; continue; }
        const auto& hash = hashes[r];
        bool same = hash.row_count == heads_;
        for (std::uint32_t h = 0; same && h < heads_; ++h)
            same = hash.row_ids[h] == rows_[static_cast<std::size_t>(p * heads_ + h)];
        mismatches += same ? 0U : 1U;
    }
    if (rolling_) rolling_->advance(first_position + hashes.size());
    Stats snapshot;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.blocks;
        stats_.gathered_rows += hashes.size();
        stats_.gather_ns_total += gather_ns;
        stats_.gather_ns_max = std::max(stats_.gather_ns_max, gather_ns);
        stats_.minflt_total += minflt;
        stats_.majflt_total += majflt;
        stats_.majflt_max = std::max(stats_.majflt_max, majflt);
        stats_.hash_mismatches += mismatches;
        if (receipt.stage_error) ++stats_.stage_errors;
        if (receipt.stage_attempted) {
            if (receipt.staged_rows) ++stats_.stage_hits; else ++stats_.stage_misses;
            stats_.stage_rows += receipt.staged_rows;
            stats_.stage_ns_total += receipt.stage_ns;
            stats_.stage_wait_ns_total += receipt.stage_wait_ns;
            stats_.stage_majflt_total += receipt.stage_majflt;
        }
        stats_.rows_outside_prompt += outside;
        if (mismatches && !stats_.bound_exceeded) {
            stats_.bound_exceeded = true;
            std::fprintf(stderr, "ple-reclaim: %llu gathered rows differ from the precomputed prompt hashes\n",
                static_cast<unsigned long long>(mismatches));
        }
        const auto end = first_position + hashes.size();
        if (!rolling() ) {
            target_ = std::max(target_, std::min(positions_, end + config_.ahead_rows));
            sample_ = true;
        }
        snapshot = stats_;
    }
    wake_.notify_one();
    if (!rolling()) std::fprintf(stderr, "ple-reclaim block %llu rows %zu staged %llu (worker %.2f ms hidden, wait %.2f ms, majflt %llu) "
        "exposed gather %.2f ms minflt %llu majflt %llu prefetched-until %llu issued %llu pages resident %llu B\n",
        static_cast<unsigned long long>(first_position), hashes.size(), static_cast<unsigned long long>(receipt.staged_rows),
        static_cast<double>(receipt.stage_ns) / 1e6, static_cast<double>(receipt.stage_wait_ns) / 1e6,
        static_cast<unsigned long long>(receipt.stage_majflt),
        static_cast<double>(gather_ns) / 1e6, static_cast<unsigned long long>(minflt), static_cast<unsigned long long>(majflt),
        static_cast<unsigned long long>(snapshot.prefetched_until), static_cast<unsigned long long>(snapshot.prefetch_pages_issued),
        static_cast<unsigned long long>(snapshot.resident_last_bytes));
    if (block_stats()) {
        const auto io = rolling_->block_io(first_position, hashes.size());
        const auto& a = io.critical;
        const auto elapsed = a.end_ns > a.begin_ns ? a.end_ns - a.begin_ns : 0U;
        const double seconds = static_cast<double>(elapsed) / 1e9;
        std::fprintf(stderr,
            "ple-rolling block: block=%llu first=%llu rows=%zu bytes-read=%llu read-wait-ms=%.3f copy-ms=%.3f hash-ms=%.3f join-wait-ms=%.3f "
            "critical-bytes=%llu critical-read-ms=%.3f critical-iops=%.3f critical-MB-s=%.3f lookahead-bytes=%llu\n",
            static_cast<unsigned long long>(snapshot.blocks - 1U), static_cast<unsigned long long>(first_position), hashes.size(),
            static_cast<unsigned long long>(receipt.bytes_read), static_cast<double>(receipt.read_wait_ns) / 1e6,
            static_cast<double>(receipt.copy_ns) / 1e6, static_cast<double>(receipt.hash_ns) / 1e6,
            static_cast<double>(receipt.stage_wait_ns) / 1e6, static_cast<unsigned long long>(a.bytes),
            static_cast<double>(elapsed) / 1e6, seconds ? static_cast<double>(a.reads) / seconds : 0.0,
            seconds ? static_cast<double>(a.bytes) / seconds / 1e6 : 0.0, static_cast<unsigned long long>(io.lookahead_bytes));
    }
    if (manager_ && snapshot.bound_exceeded) manager_->invalidate();
    if ((rolling() || config_.strict) && snapshot.bound_exceeded)
        throw std::runtime_error("STRIX_PLE_RECLAIM_STRICT: PLE residency bound or prompt hash check failed");
}

void Prefetcher::finish() {
    if (stats().final_sampled) return;
    const auto stage_error = drain_stages();
    if (manager_) {
        if (!manager_->healthy()) throw std::runtime_error("PLE residency: verified credit invalidated");
        if (const auto* v = std::getenv("STRIX_PLE_ROLLING_STATS"); v && std::strcmp(v,"1") == 0) {
            const auto r = manager_->stats();
            std::fprintf(stderr, "ple-rolling summary: epochs=%llu pages-faulted=%llu pages-prefetched=%llu pages-evicted=%llu "
                "peak-resident-ple-bytes=%llu gather-wait-ms=%.3f\n",
                static_cast<unsigned long long>(r.epochs), static_cast<unsigned long long>(r.faulted_pages),
                static_cast<unsigned long long>(r.prefetched_pages), static_cast<unsigned long long>(r.evicted_pages), static_cast<unsigned long long>(r.peak_resident_bytes +
                    config_.table_rows * config_.stored_row_bytes - stats_.reclaimable_bytes),
                static_cast<double>(stats().gather_ns_total) / 1e6);
        }
    }
    if (rolling_) {
        rolling_->finish();
        if (stage_error) std::rethrow_exception(stage_error);
        if (const auto* v = std::getenv("STRIX_PLE_ROLLING_STATS"); v && std::strcmp(v, "1") == 0) {
            const auto r = rolling_->stats();
            const auto s = stats();
            std::fprintf(stderr, "ple-rolling summary: epochs=%llu pages-faulted=%llu pages-prefetched=%llu pages-evicted=%llu "
                "peak-resident-ple-bytes=%llu gather-wait-ms=%.3f hidden-gather-ms=%.3f exposed-gather-ms=%.3f "
                "lookahead-depth=%llu throttle-count=%llu cap=%llu peak-owned-bytes=%llu split-windows=%llu\n",
                static_cast<unsigned long long>(r.epochs), static_cast<unsigned long long>(r.faulted_pages),
                static_cast<unsigned long long>(r.prefetched_pages), static_cast<unsigned long long>(r.evicted_pages),
                static_cast<unsigned long long>(r.peak_resident_bytes + config_.table_rows * config_.stored_row_bytes -
                    s.reclaimable_bytes), static_cast<double>(s.gather_ns_total + s.stage_wait_ns_total) / 1e6,
                static_cast<double>(s.stage_ns_total > s.stage_wait_ns_total ? s.stage_ns_total - s.stage_wait_ns_total : 0U) / 1e6,
                static_cast<double>(s.gather_ns_total + s.stage_wait_ns_total) / 1e6,
                static_cast<unsigned long long>(r.lookahead_positions), static_cast<unsigned long long>(r.throttles),
                static_cast<unsigned long long>(rolling_budget(config_, heads_).cap),
                static_cast<unsigned long long>(r.peak_owned_bytes), static_cast<unsigned long long>(r.split_windows));
        }
    }
    bool failed = false;
    const auto resident = resident_bytes(&failed);
    const auto evicted = kfd_evicted_ms();
    bool exceeded = false;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stats_.final_sampled) return;
        stats_.final_sampled = true;
        stats_.kfd_evicted_ms_final = evicted;
        stats_.mincore_failed = stats_.mincore_failed || failed;
        ++stats_.resident_samples;
        stats_.resident_last_bytes = resident;
        stats_.resident_max_bytes = std::max(stats_.resident_max_bytes, resident);
        if (resident > (direct_ ? config_.drop_residual_bytes : admitted_) && !stats_.bound_exceeded) {
            stats_.bound_exceeded = true;
            std::fprintf(stderr, "ple-reclaim: final resident PLE %llu B exceeds the admitted %llu B\n",
                static_cast<unsigned long long>(resident), static_cast<unsigned long long>(admitted_));
        }
        exceeded = stats_.bound_exceeded;
    }
    if (direct_ && (failed || exceeded || !direct_->healthy())) {
        if (manager_) manager_->invalidate();
        throw std::runtime_error("PLE residency: final sample failed or bound exceeded");
    }
    if (direct_) {
        if (const auto* v = std::getenv("STRIX_PLE_ROLLING_STATS"); v && std::strcmp(v, "1") == 0) {
            const auto d = direct_->stats();
            std::fprintf(stderr, "ple-direct summary: reads-issued=%llu bytes-read=%llu cache-hits=%llu "
                "read-wait-hidden-ms=%.3f read-wait-exposed-ms=%.3f peak-owned-bytes=%llu peak-used-bytes=%llu "
                "mapping-resident-pages-end=%llu peak-inflight=%llu page-cache-contribution=0 B hits-inflight=%llu\n",
                static_cast<unsigned long long>(d.reads), static_cast<unsigned long long>(d.bytes),
                static_cast<unsigned long long>(d.hits), static_cast<double>(d.hidden_wait_ns) / 1e6,
                static_cast<double>(d.exposed_wait_ns) / 1e6, static_cast<unsigned long long>(d.peak_owned_bytes),
                static_cast<unsigned long long>(d.peak_used_bytes), static_cast<unsigned long long>(resident / config_.page),
                static_cast<unsigned long long>(d.peak_inflight), static_cast<unsigned long long>(d.inflight_hits));
        }
    }
    if (config_.strict && exceeded) {
        // The run fails before its stdout receipt: leave the evidence on stderr first.
        std::fprintf(stderr, "ple-reclaim receipt: %s\n", receipt_json().c_str());
        throw std::runtime_error("STRIX_PLE_RECLAIM_STRICT: PLE residency bound or prompt hash check failed");
    }
}

Stats Prefetcher::stats() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto result = stats_;
    if (manager_) {
        result.resident_max_bytes = std::max(result.resident_max_bytes, manager_->stats().peak_resident_bytes);
        if (!manager_->healthy()) result.credit_bytes = 0;
    }
    if (rolling_) {
        result.resident_max_bytes = std::max(result.resident_max_bytes, rolling_->stats().peak_resident_bytes);
        if (!rolling_->healthy()) result.credit_bytes = 0;
    }
    if (direct_ && !direct_->healthy()) result.credit_bytes = 0;
    return result;
}

std::string Prefetcher::receipt_json() const {
    const auto s = stats();
    const auto u = [](std::uint64_t v) { return std::to_string(v); };
    const auto mode = rolling() ? ",\"mode\":\"rolling\",\"rolling_positions\":" + u(config_.rolling_positions) +
        ",\"verified_credit_bytes\":" + u(credit_bytes()) : std::string{};
    std::string direct;
    if (direct_) {
        const auto d = direct_->stats();
        direct = ",\"reader_backend\":\"" + std::string(direct_->reader()) + "\",\"reader_fallback_errno\":" + u(direct_->fallback_errno()) + ",\"source_io\":\"O_DIRECT-owned\",\"page_cache_contribution_bytes\":0,\"reads_issued\":" + u(d.reads) +
            ",\"peak_inflight\":" + u(d.peak_inflight) + ",\"bytes_read\":" + u(d.bytes) + ",\"cache_hits\":" + u(d.hits) +
            ",\"read_wait_hidden_ns\":" + u(d.hidden_wait_ns) + ",\"read_wait_exposed_ns\":" + u(d.exposed_wait_ns) +
            ",\"peak_owned_bytes\":" + u(d.peak_owned_bytes) + ",\"peak_used_bytes\":" + u(d.peak_used_bytes) +
            ",\"mapping_resident_pages_end\":" + u(s.resident_last_bytes / config_.page);
    }
    return std::string("{\"scope\":\"STRIX_PLE_RECLAIMABLE=1 prefill PLE pages dropped at setup and prefetched; admission credit\"") + mode + direct +
        ",\"reclaimable_bytes\":" + u(s.reclaimable_bytes) + ",\"touched_bytes\":" + u(s.touched_bytes) +
        ",\"decode_allowance_bytes\":" + u(config_.decode_allowance_bytes) + ",\"admitted_bytes\":" + u(s.admitted_bytes) +
        ",\"credit_bytes\":" + u(s.credit_bytes) + ",\"resident_before_drop_bytes\":" + u(s.resident_before_drop_bytes) +
        ",\"resident_after_drop_bytes\":" + u(s.resident_after_drop_bytes) + ",\"blocks\":" + u(s.blocks) +
        ",\"gathered_rows\":" + u(s.gathered_rows) + ",\"gather_ns_total\":" + u(s.gather_ns_total) +
        ",\"gather_ns_max\":" + u(s.gather_ns_max) + ",\"minflt_total\":" + u(s.minflt_total) +
        ",\"majflt_total\":" + u(s.majflt_total) + ",\"majflt_max\":" + u(s.majflt_max) +
        ",\"prefetch_ahead_rows\":" + u(config_.ahead_rows) + ",\"prefetched_until\":" + u(s.prefetched_until) +
        ",\"prefetch_pages_considered\":" + u(s.prefetch_pages_considered) +
        ",\"prefetch_pages_issued\":" + u(s.prefetch_pages_issued) + ",\"prefetch_ns_total\":" + u(s.prefetch_ns_total) +
        ",\"resident_samples\":" + u(s.resident_samples) + ",\"resident_max_bytes\":" + u(s.resident_max_bytes) +
        ",\"resident_last_bytes\":" + u(s.resident_last_bytes) + ",\"hash_mismatches\":" + u(s.hash_mismatches) +
        ",\"rows_outside_prompt\":" + u(s.rows_outside_prompt) +
        ",\"stage_prefetch\":" + (config_.stage_prefetch ? "true" : "false") + ",\"stage_hits\":" + u(s.stage_hits) +
        ",\"stage_misses\":" + u(s.stage_misses) + ",\"stage_rows\":" + u(s.stage_rows) +
        ",\"stage_ns_total\":" + u(s.stage_ns_total) + ",\"stage_wait_ns_total\":" + u(s.stage_wait_ns_total) +
        ",\"stage_majflt_total\":" + u(s.stage_majflt_total) + ",\"stage_errors\":" + u(s.stage_errors) +
        ",\"strict\":" + (config_.strict ? "true" : "false") +
        ",\"exposed_faults_scope\":\"process-wide (RUSAGE_SELF) around the launch thread's hash+gather\"" +
        ",\"kfd_evicted_ms_before_drop\":" + std::to_string(s.kfd_evicted_ms_before_drop) +
        ",\"kfd_evicted_ms_after_drop\":" + std::to_string(s.kfd_evicted_ms_after_drop) +
        ",\"kfd_evicted_ms_final\":" + std::to_string(s.kfd_evicted_ms_final) +
        ",\"final_sampled\":" + (s.final_sampled ? "true" : "false") +
        ",\"kfd_evicted_during_drop\":" + (s.kfd_evicted_during_drop ? "true" : "false") +
        ",\"mincore_failed\":" + (s.mincore_failed ? "true" : "false") +
        ",\"bound_exceeded\":" + (s.bound_exceeded ? "true" : "false") + "}";
}

}  // namespace strix::ple_reclaim
