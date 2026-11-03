#include "strix/ple_lookup.hpp"
#include "strix/ple_gap_witness.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <mutex>
#include <memory>
#include <limits>
#include <string_view>
#include <thread>
#include <type_traits>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class PrefetchFixtureBackend final
    : public strix::PleLookupBackend {
public:
    std::uint64_t implementation_identity()
        const noexcept override {
        return 0x5052464554434801ULL;
    }

    strix::PleLookupHandle start(
        const strix::PleLookupRequest& request) override {
        if (active_ ||
            request.row_ids.empty() ||
            request.row_bytes == 0) {
            throw std::logic_error(
                "invalid fixture prefetch start");
        }
        active_ = true;
        bytes_ =
            static_cast<std::uint64_t>(request.row_ids.size()) *
            request.row_bytes;
        return 9;
    }

    void finish(
        strix::PleLookupHandle handle,
        std::span<std::byte> destination) override {
        if (!active_ || handle != 9 ||
            destination.size() < bytes_) {
            throw std::invalid_argument(
                "fixture prefetch finish mismatch");
        }
        std::fill_n(
            destination.begin(),
            static_cast<std::size_t>(bytes_),
            std::byte{0x2a});
        active_ = false;
    }

    void cancel(
        strix::PleLookupHandle handle) noexcept override {
        if (handle == 9) active_ = false;
    }

private:
    std::uint64_t bytes_ = 0;
    bool active_ = false;
};

class MemoryIoSource final
    : public strix::SparseLookupIoSource {
public:
    explicit MemoryIoSource(
        std::vector<std::byte> bytes)
        : bytes_(std::move(bytes)) {}

    void read(
        std::uint64_t byte_offset,
        std::span<std::byte> destination) override {
        if (byte_offset >
                bytes_.size() ||
            destination.size() >
                bytes_.size() -
                    static_cast<std::size_t>(
                        byte_offset)) {
            throw std::out_of_range(
                "memory I/O source read exceeds bytes");
        }
        std::copy_n(
            bytes_.data() +
                static_cast<std::size_t>(
                    byte_offset),
            destination.size(),
            destination.data());
    }

private:
    std::vector<std::byte> bytes_;
};

class BudgetPleSource final : public strix::SparseLookupIoSource {
public:
    explicit BudgetPleSource(MemoryIoSource& source, bool blocked = false)
        : source_(source), released_(!blocked) {}
    std::atomic<unsigned> active{0}, peak{0}, calls{0}, caller_reads{0};
    const std::thread::id caller = std::this_thread::get_id();
    void read(std::uint64_t offset, std::span<std::byte> output) override {
        const auto current = active.fetch_add(1) + 1;
        struct Finish { std::atomic<unsigned>& value; ~Finish() { value.fetch_sub(1); } } finish{active};
        auto prior = peak.load();
        while (current > prior && !peak.compare_exchange_weak(prior, current)) {}
        if (std::this_thread::get_id() == caller) caller_reads.fetch_add(1);
        {
            std::unique_lock lock(mutex_);
            calls.fetch_add(1); condition_.notify_all();
            condition_.wait(lock, [&] { return released_; });
        }
        source_.read(offset, output);
    }
    void wait_started() {
        std::unique_lock lock(mutex_);
        if (!condition_.wait_for(lock, std::chrono::seconds(5), [&] { return calls.load() != 0; }))
            throw std::runtime_error("bounded PLE fixture source did not start");
    }
    void release() { std::lock_guard lock(mutex_); released_ = true; condition_.notify_all(); }
private:
    MemoryIoSource& source_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool released_;
};

}  // namespace

class CachePleSource final : public strix::SparseLookupIoSource {
public:
    std::atomic<unsigned> calls{0};
    std::atomic<bool> fail{false};
    void read(std::uint64_t offset, std::span<std::byte> output) override {
        calls.fetch_add(1);
        if (fail.load()) throw std::runtime_error("injected PLE source failure");
        if (offset > 256U || output.size() > 256U - offset)
            throw std::out_of_range("bounded PLE cache source extent");
        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = static_cast<std::byte>((offset + i) & 255U);
    }
};
void qualify_row_cache_source_contract() {
    auto source = std::make_shared<CachePleSource>();
    std::weak_ptr<CachePleSource> weak = source;
    strix::ManifestTensor table;
    table.name = "cache.ngram"; table.role = strix::TensorRole::ngram_table;
    table.ggml_type = 30; table.dimensions = {4, 8}; table.byte_size = 64; table.absolute_offset = 32;
    strix::RuntimeTensorBinding runtime;
    runtime.storage = strix::RuntimeTensorStorage::file_mapped;
    runtime.bytes = 64; runtime.allocation_generation = 7;
    strix::PlannedPleLookupPolicy policy;
    policy.max_rows_per_io = 1; policy.max_inflight_ios = 1;
    policy.cache_capacity_rows = 2; policy.cache_byte_budget = 16;
    policy.cache_model_content.bytes[0] = 0x51;
    const auto reservation = strix::PlannedPleLookupBackend::cache_reservation_bytes(policy);
    expect(reservation > 16U, "row cache reservation includes preallocated tags and complete raw BF16 payloads");
    {
        strix::PlannedPleLookupBackend backend(*source, policy, source);
        expect(backend.cache_stats().reserved_bytes == reservation, "cache reserve uses the public pure calculation");
        const auto gather = [&](std::span<const std::uint64_t> rows) {
            const strix::PleLookupRequest request{rows, 8, &table, &runtime};
            const auto handle = backend.start(request);
            std::vector<std::byte> result(rows.size() * 8U, std::byte{0xcd});
            backend.finish(handle, result);
            for (std::size_t row = 0; row < rows.size(); ++row)
                for (std::size_t byte = 0; byte < 8U; ++byte)
                    expect(result[row * 8U + byte] == static_cast<std::byte>(
                        (table.absolute_offset + rows[row] * 8U + byte) & 255U),
                        "cached and missed rows preserve full payload bits and request duplicate order");
            return result;
        };
        const std::array<std::uint64_t, 3> initial{3, 1, 3};
        const auto reference = gather(initial);
        const auto reads = source->calls.load();
        expect(reads == 2U && backend.cache_stats().inserted == 2U,
            "cold cache submits two real reads for two unique rows, not a planner hit estimate");
        expect(gather(initial) == reference && source->calls.load() == reads && backend.cache_stats().hits == 3U,
            "all-hit lookup performs no source reads and handles an empty I/O future");
        const std::array<std::uint64_t, 1> row4{4}, row1{1};
        (void)gather(row4); // LRU row 1 is evicted; duplicate row 3 was most recently used.
        const auto after4 = source->calls.load();
        (void)gather(row1);
        expect(source->calls.load() == after4 + 1U && backend.cache_stats().resident_rows == 2U &&
            backend.cache_stats().evicted >= 2U, "bounded LRU evicts and rereads the actual least recent payload");
        ++runtime.allocation_generation;
        const auto before_generation = source->calls.load();
        (void)gather(row1);
        expect(source->calls.load() == before_generation + 1U && backend.cache_stats().invalidations == 1U,
            "allocation generation change invalidates cached bytes before any hit");
        table.absolute_offset += 8U;
        const auto before_namespace = source->calls.load();
        (void)gather(row1);
        expect(source->calls.load() == before_namespace + 1U && backend.cache_stats().invalidations == 2U,
            "same row ID with a different file window never returns old content");
        const std::array<std::uint64_t, 1> row6{6};
        const strix::PleLookupRequest cancelled{row6, 8, &table, &runtime};
        const auto insert_before = backend.cache_stats().inserted;
        const auto handle = backend.start(cancelled);
        backend.cancel(handle);
        expect(backend.cache_stats().inserted == insert_before,
            "cancellation joins outstanding reads but does not publish cache entries");
        const auto after_cancel = source->calls.load();
        (void)gather(row6);
        expect(source->calls.load() == after_cancel + 1U,
            "cancelled miss must perform a real source read on retry");
        const std::array<std::uint64_t, 2> mixed{6, 7};
        source->fail = true;
        const auto failure_handle = backend.start({mixed, 8, &table, &runtime});
        std::array<std::byte, 16> untouched;
        untouched.fill(std::byte{0x5a});
        const auto before_failure = backend.cache_stats().inserted;
        bool failed = false;
        try { backend.finish(failure_handle, untouched); }
        catch (const std::runtime_error& error) { failed = std::string_view(error.what()) == "injected PLE source failure"; }
        expect(failed && backend.cache_stats().inserted == before_failure &&
            std::all_of(untouched.begin(), untouched.end(), [](auto byte) { return byte == std::byte{0x5a}; }),
            "failed mixed hit/miss I/O publishes neither cached rows nor a partial destination");
        source->fail = false;
        (void)gather(mixed);
        source.reset();
        expect(!weak.expired(), "backend retains the actual immutable I/O source owner during cached use");
        (void)gather(row6);
    }
    expect(weak.expired(), "source owner is released after backend and its async pending work are destroyed");
    auto dummy = std::make_shared<CachePleSource>();
    bool owner_rejected = false;
    try { strix::PlannedPleLookupBackend invalid(*dummy, policy); }
    catch (const std::invalid_argument&) { owner_rejected = true; }
    expect(owner_rejected, "cache requires a strong immutable source owner");
    auto no_content = policy; no_content.cache_model_content = {};
    bool content_rejected = false;
    try { strix::PlannedPleLookupBackend invalid(*dummy, no_content, dummy); }
    catch (const std::invalid_argument&) { content_rejected = true; }
    expect(content_rejected, "cache requires explicit model content identity, not just address and generation");
    auto partial = policy; partial.cache_byte_budget = 0;
    bool partial_rejected = false;
    try { (void)strix::PlannedPleLookupBackend::cache_reservation_bytes(partial); }
    catch (const std::invalid_argument&) { partial_rejected = true; }
    expect(partial_rejected, "partial cache budgets fail admission");
    auto overflow = policy; overflow.cache_capacity_rows = std::numeric_limits<std::uint64_t>::max();
    bool overflow_rejected = false;
    try { (void)strix::PlannedPleLookupBackend::cache_reservation_bytes(overflow); }
    catch (const std::overflow_error&) { overflow_rejected = true; }
    expect(overflow_rejected, "cache tag reservation overflow fails before allocation");
    auto tiny = policy; tiny.cache_byte_budget = 7;
    strix::PlannedPleLookupBackend too_small(*dummy, tiny, dummy);
    const std::array<std::uint64_t, 1> valid_row{0};
    bool short_rejected = false;
    try { (void)too_small.start({valid_row, 8, &table, &runtime}); }
    catch (const std::invalid_argument&) { short_rejected = true; }
    expect(short_rejected && dummy->calls == 0U, "undersized complete-row cache rejects before I/O");
    auto changed_content = policy; changed_content.cache_model_content.bytes[0] ^= 1U;
    strix::PlannedPleLookupBackend first_content(*dummy, policy, dummy);
    strix::PlannedPleLookupBackend second_content(*dummy, changed_content, dummy);
    expect(first_content.implementation_identity() != second_content.implementation_identity(),
        "source cache identity distinguishes model content even with equal geometry and runtime addresses");
    const std::array<std::uint64_t, 2> late_invalid{0, 8};
    bool row_rejected = false;
    try { (void)first_content.start({late_invalid, 8, &table, &runtime}); }
    catch (const std::out_of_range&) { row_rejected = true; }
    expect(row_rejected && dummy->calls == 0U, "late invalid row rejects before any cache lookup I/O");
    strix::PlannedPleLookupBackend legacy(*dummy);
    expect(legacy.cache_stats().reserved_bytes == 0U, "default backend does not allocate or admit a payload cache");
}

// The pre-issue helper is owned by exactly one backend.
static_assert(!std::is_copy_constructible_v<strix::FileMappedPleLookupBackend> &&
    !std::is_copy_assignable_v<strix::FileMappedPleLookupBackend> &&
    !std::is_move_constructible_v<strix::FileMappedPleLookupBackend>,
    "file-mapped PLE backend owns its pre-issue helper exclusively");

void qualify_preissue_switches() {
    using B = strix::FileMappedPleLookupBackend;
    expect(B::preissue_enabled_from(nullptr, nullptr, nullptr), "pre-issue defaults on");
    expect(!B::preissue_enabled_from("0", nullptr, nullptr), "STRIX_PLE_PREISSUE=0 turns it off");
    expect(B::preissue_enabled_from("1", nullptr, nullptr), "explicit STRIX_PLE_PREISSUE=1 keeps it on");
    expect(!B::preissue_enabled_from("1", nullptr, "1"), "STRIX_BASELINE=1 wins over explicit pre-issue");
    expect(!B::preissue_enabled_from("1", "0", nullptr), "STRIX_QPF2K_DEFAULTS=0 wins over explicit pre-issue");
    expect(B::preissue_enabled_from(nullptr, "1", "0"), "non-disabling master values keep the default");
}

// Host mock of the event API: query() completes once `done` is set; elapsed = b.t - a.t.
struct MockEvents {
    struct Event { double t = -1; bool done = false; };
    using Stream = double*;   // the mock "stream clock"
    static bool create(Event&) noexcept { return true; }
    static void destroy(Event&) noexcept {}
    static bool record(Event& e, Stream s) noexcept { e.t = *s; e.done = true; return true; }
    static bool query(Event& e) noexcept { return e.done; }
    static bool elapsed(float& ms, Event& a, Event& b) noexcept { ms = static_cast<float>(b.t - a.t); return true; }
};

void qualify_gap_witness_slots() {
    double clock = 0;
    strix::ple_gap::Site<MockEvents, 4> site("test", true);
    // Overlapping gathers get distinct slots and both samples are counted.
    const int first = site.begin(&clock);
    const int second = site.begin(&clock);
    expect(first >= 0 && second >= 0 && first != second, "overlapping witness begins reserve distinct slots");
    clock = 2; site.end(first, &clock, 0, 0, 0);
    clock = 5; site.end(second, &clock, 0, 0, 0);
    site.collect();
    expect(site.samples() == 2 && site.gap_ms() == 7.0, "both overlapping witness pairs are read");
    // A begin without end (exception path) is reclaimed when the ring returns to its slot.
    const int lost = site.begin(&clock);
    expect(lost >= 0, "witness slot reserved");
    for (int i = 0; i < 3; ++i) { const int k = site.begin(&clock); site.end(k, &clock, 0, 0, 0); }
    const int again = site.begin(&clock);
    expect(again != lost && again % 4 == lost % 4 && site.abandoned() == 1, "abandoned witness slot is reclaimed and counted");
    site.end(lost, &clock, 0, 0, 0);   // stale end of the reclaimed slot must not record over the new owner
    site.end(again, &clock, 0, 0, 0);
    site.collect();
    expect(site.samples() == 6, "reclaimed slot yields one sample, not two");
    strix::ple_gap::Site<MockEvents, 4> off("off", false);
    expect(off.begin(&clock) == -1, "witness off reserves nothing");
    expect(strix::ple_gap::enabled_from("1", nullptr) && strix::ple_gap::enabled_from(nullptr, "1") &&
        !strix::ple_gap::enabled_from("0", "0") && !strix::ple_gap::enabled_from(nullptr, nullptr), "witness switches");
}

int main() {
    qualify_row_cache_source_contract();
    qualify_preissue_switches();
    qualify_gap_witness_slots();
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto contract =
        strix::build_ple_hash_contract(
            model);

    expect(
        contract.head_count == 16 &&
            contract.head_dim == 160 &&
            contract.row_bytes == 320,
        "PLE hash contract exposes canonical row geometry");
    expect(
        contract.minimum_table_rows >
            model.ngram_vocab_size_base,
        "PLE hash contract builds disjoint head ranges");

    const std::array<std::uint32_t, 2> initial{
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
    };
    const auto first =
        strix::hash_ple_token(
            contract,
            5,
            initial);
    expect(
        first.row_count == 16 &&
            first.context_count == 2 &&
            first.next_context[0] ==
                model.ple_eos_token_id &&
            first.next_context[1] == 5,
        "first PLE token advances EOS-padded history");

    const std::array<std::uint32_t, 2> second_context{
        first.next_context[0],
        first.next_context[1],
    };
    const auto second =
        strix::hash_ple_token(
            contract,
            7,
            second_context);
    expect(
        second.next_context[0] == 5 &&
            second.next_context[1] == 7,
        "second PLE token retains previous token");

    const std::array<std::uint32_t, 2> after_eos_context{
        11,
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
    };
    const auto after_eos =
        strix::hash_ple_token(
            contract,
            13,
            after_eos_context);
    const std::array<std::uint32_t, 2> clean_context{
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
        static_cast<std::uint32_t>(
            model.ple_eos_token_id),
    };
    const auto clean =
        strix::hash_ple_token(
            contract,
            13,
            clean_context);
    expect(
        after_eos.row_ids == clean.row_ids,
        "EOS in immediate history resets older PLE context");

    bool bad_context = false;
    try {
        const std::array<std::uint32_t, 1> short_context{1};
        static_cast<void>(
            strix::hash_ple_token(
                contract,
                2,
                short_context));
    } catch (const std::invalid_argument&) {
        bad_context = true;
    }
    expect(
        bad_context,
        "PLE hashing rejects wrong history length");


    {
        constexpr std::uint64_t row_elements = 160;
        constexpr std::uint64_t rows = 4;
        constexpr std::uint64_t row_bytes =
            row_elements * sizeof(std::uint16_t);

        std::vector<std::uint16_t> table(
            static_cast<std::size_t>(
                row_elements * rows));
        for (std::uint64_t row = 0;
             row < rows;
             ++row) {
            for (std::uint64_t column = 0;
                 column < row_elements;
                 ++column) {
                table[
                    static_cast<std::size_t>(
                        row * row_elements +
                        column)] =
                    static_cast<std::uint16_t>(
                        row * 1000ULL +
                        column);
            }
        }

        strix::ManifestTensor table_manifest;
        table_manifest.name =
            "per_layer_token_embd.weight";
        table_manifest.dimensions = {
            row_elements,
            rows,
        };
        table_manifest.ggml_type = 30;
        table_manifest.byte_size =
            table.size() *
            sizeof(std::uint16_t);
        table_manifest.role =
            strix::TensorRole::ngram_table;

        strix::RuntimeTensorBinding table_runtime;
        table_runtime.storage =
            strix::RuntimeTensorStorage::file_mapped;
        table_runtime.address =
            reinterpret_cast<std::uintptr_t>(
                table.data());
        table_runtime.bytes =
            table_manifest.byte_size;
        table_runtime.allocation_generation = 1;

        const std::array<std::uint64_t, 2>
            requested_rows{2, 0};
        strix::PleLookupRequest lookup;
        lookup.row_ids =
            requested_rows;
        lookup.row_bytes =
            row_bytes;
        lookup.table_manifest =
            &table_manifest;
        lookup.table_runtime =
            &table_runtime;

        strix::FileMappedPleLookupBackend backend;
        expect(
            backend.implementation_identity() != 0,
            "file-mapped PLE backend has stable non-zero identity");

        const auto handle =
            backend.start(lookup);
        std::vector<std::uint16_t> gathered(
            static_cast<std::size_t>(
                row_elements *
                requested_rows.size()));
        backend.finish(
            handle,
            std::as_writable_bytes(
                std::span<std::uint16_t>(
                    gathered)));

        expect(
            gathered[0] == 2000 &&
                gathered[
                    row_elements - 1] ==
                    2159 &&
                gathered[
                    row_elements] == 0 &&
                gathered[
                    2 * row_elements - 1] ==
                    159,
            "file-mapped backend gathers requested rows in order");

        bool concurrent_rejected = false;
        const auto active =
            backend.start(lookup);
        try {
            static_cast<void>(
                backend.start(lookup));
        } catch (const std::logic_error&) {
            concurrent_rejected = true;
        }
        expect(
            concurrent_rejected,
            "file-mapped backend rejects concurrent C1 lookup");
        backend.cancel(active);

        auto resident = table_runtime;
        resident.storage =
            strix::RuntimeTensorStorage::resident;
        lookup.table_runtime =
            &resident;
        bool resident_rejected = false;
        try {
            static_cast<void>(
                backend.start(lookup));
        } catch (const std::invalid_argument&) {
            resident_rejected = true;
        }
        expect(
            resident_rejected,
            "file-mapped backend does not assume resident GPU memory is host-readable");

        lookup.table_runtime =
            &table_runtime;
        const std::array<std::uint64_t, 1>
            bad_row{rows};
        lookup.row_ids = bad_row;
        bool row_rejected = false;
        try {
            static_cast<void>(
                backend.start(lookup));
        } catch (const std::out_of_range&) {
            row_rejected = true;
        }
        expect(
            row_rejected,
            "file-mapped backend rejects row outside table geometry");
    }

    {
        constexpr std::uint64_t row_elements = 160;
        constexpr std::uint64_t rows = 4;
        constexpr std::uint64_t row_bytes =
            row_elements * sizeof(std::uint16_t);
        constexpr std::uint64_t table_offset = 64;

        std::vector<std::byte> file_bytes(
            static_cast<std::size_t>(
                table_offset +
                rows * row_bytes));
        for (std::uint64_t row = 0;
             row < rows;
             ++row) {
            for (std::uint64_t column = 0;
                 column < row_elements;
                 ++column) {
                const auto value =
                    static_cast<std::uint16_t>(
                        row * 1000ULL +
                        column);
                std::memcpy(
                    file_bytes.data() +
                        static_cast<std::size_t>(
                            table_offset +
                            row * row_bytes +
                            column *
                                sizeof(std::uint16_t)),
                    &value,
                    sizeof(value));
            }
        }

        strix::ManifestTensor table_manifest;
        table_manifest.name =
            "per_layer_token_embd.weight";
        table_manifest.dimensions = {
            row_elements,
            rows,
        };
        table_manifest.ggml_type = 30;
        table_manifest.byte_size =
            rows * row_bytes;
        table_manifest.absolute_offset =
            table_offset;
        table_manifest.role =
            strix::TensorRole::ngram_table;

        strix::RuntimeTensorBinding table_runtime;
        table_runtime.storage =
            strix::RuntimeTensorStorage::
                file_mapped;
        table_runtime.address = 1;
        table_runtime.bytes =
            table_manifest.byte_size;
        table_runtime.allocation_generation = 1;

        const std::array<std::uint64_t, 3>
            requested_rows{2, 0, 2};
        strix::PleLookupRequest lookup;
        lookup.row_ids =
            requested_rows;
        lookup.row_bytes =
            row_bytes;
        lookup.table_manifest =
            &table_manifest;
        lookup.table_runtime =
            &table_runtime;

        MemoryIoSource source(
            std::move(file_bytes));
        strix::PlannedPleLookupPolicy policy;
        policy.max_rows_per_io = 2;
        policy.max_inflight_ios = 2;
        strix::PlannedPleLookupBackend backend(
            source,
            policy);

        expect(
            backend.implementation_identity() != 0,
            "planned PLE backend has stable non-zero identity");

        const auto handle =
            backend.start(lookup);
        std::vector<std::uint16_t> gathered(
            static_cast<std::size_t>(
                row_elements *
                requested_rows.size()));
        backend.finish(
            handle,
            std::as_writable_bytes(
                std::span<std::uint16_t>(
                    gathered)));

        expect(
            gathered[0] == 2000 &&
                gathered[
                    row_elements - 1] ==
                    2159 &&
                gathered[
                    row_elements] == 0 &&
                gathered[
                    2 * row_elements - 1] ==
                    159 &&
                gathered[
                    2 * row_elements] ==
                    2000,
            "planned PLE backend reads tensor-relative rows and restores request order");

        auto resident = table_runtime;
        resident.storage =
            strix::RuntimeTensorStorage::
                resident;
        lookup.table_runtime =
            &resident;
        bool resident_rejected = false;
        try {
            static_cast<void>(
                backend.start(lookup));
        } catch (const std::invalid_argument&) {
            resident_rejected = true;
        }
        expect(
            resident_rejected,
            "planned PLE backend requires a file-backed tensor");

        lookup.table_runtime =
            &table_runtime;
        const auto active =
            backend.start(lookup);
        bool concurrent_rejected = false;
        try {
            static_cast<void>(
                backend.start(lookup));
        } catch (const std::logic_error&) {
            concurrent_rejected = true;
        }
        expect(
            concurrent_rejected,
            "planned PLE backend rejects overlapping lookup handles");
        backend.cancel(active);

        const auto legacy_identity = 0x504c45504c414e31ULL ^ (2ULL << 17U) ^ (2ULL << 33U);
        expect(backend.implementation_identity() == legacy_identity,
            "zero CPU helper policy retains the exact incumbent backend identity");
        std::array<std::uint64_t, 4> budget_identities{};
        unsigned identity_index = 0;
        for (const auto helpers : {1U,2U,4U,8U}) {
            BudgetPleSource tracked(source);
            auto bounded_policy = policy; bounded_policy.cpu_helper_threads = helpers;
            bounded_policy.max_rows_per_io = 1; bounded_policy.max_inflight_ios = 8;
            strix::PlannedPleLookupBackend bounded(tracked, bounded_policy);
            budget_identities[identity_index++] = bounded.implementation_identity();
            std::vector<std::uint16_t> bounded_rows(gathered.size(), 0xdead);
            const auto pending = bounded.start(lookup);
            bounded.finish(pending, std::as_writable_bytes(std::span(bounded_rows)));
            expect(bounded_rows == gathered && tracked.active.load() == 0 && tracked.peak.load() <= helpers &&
                tracked.caller_reads.load() == 0 && tracked.calls.load() == 2,
                "planned backend includes its asynchronous coordinator within the explicit CPU reader budget");
            // Cancel while a real source read is blocked. The cancellation
            // thread is test control, not another backend/source worker.
            BudgetPleSource blocked(source, true);
            strix::PlannedPleLookupBackend cancelling(blocked, bounded_policy);
            struct ReleaseOnExit { BudgetPleSource& source; ~ReleaseOnExit() { source.release(); } } release{blocked};
            const auto cancelled_handle = cancelling.start(lookup);
            blocked.wait_started();
            std::promise<void> cancel_entered;
            auto entered = cancel_entered.get_future();
            std::atomic<bool> cancelled_done{false};
            auto cancellation = std::async(std::launch::async, [&] {
                cancel_entered.set_value(); cancelling.cancel(cancelled_handle); cancelled_done.store(true);
            });
            entered.wait();
            expect(!cancelled_done.load() && blocked.active.load() != 0,
                "cancel cannot return or release a source owner while its I/O is blocked");
            blocked.release(); cancellation.get();
            expect(cancelled_done.load() && blocked.active.load() == 0 && blocked.peak.load() <= helpers,
                "cancel joins every submitted bounded reader before returning");
            std::vector<std::uint16_t> retry_rows(gathered.size());
            const auto retry = cancelling.start(lookup);
            cancelling.finish(retry, std::as_writable_bytes(std::span(retry_rows)));
            expect(retry_rows == gathered, "bounded planned backend is reusable after drained cancellation");
        }
        for (std::size_t a = 0; a < budget_identities.size(); ++a) {
            expect(budget_identities[a] != legacy_identity, "explicit CPU policy changes backend organization identity");
            for (std::size_t b = 0; b < a; ++b)
                expect(budget_identities[a] != budget_identities[b], "different CPU helper budgets retain distinct backend identities");
        }
        for (const auto invalid_budget : {9U,UINT32_MAX}) {
            BudgetPleSource untouched(source);
            auto invalid_policy = policy; invalid_policy.cpu_helper_threads = invalid_budget;
            bool rejected = false;
            try { strix::PlannedPleLookupBackend invalid_backend(untouched, invalid_policy); }
            catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected && untouched.calls.load() == 0,
                "invalid helper admission fails before creating a lookup or consuming source I/O");
        }
    }

    {
        strix::ModelManifest manifest;
        manifest.architecture = "prefetch-fixture";
        strix::ManifestTensor table;
        table.name = "per_layer_token_embd.weight";
        table.dimensions = {
            contract.head_dim,
            contract.minimum_table_rows,
        };
        table.ggml_type = 30;
        table.byte_size =
            contract.minimum_table_rows * contract.row_bytes;
        table.role = strix::TensorRole::ngram_table;
        manifest.tensors.push_back(table);
        manifest.total_tensor_bytes = table.byte_size;
        manifest.layout_identity =
            strix::model_manifest_layout_identity(manifest);

        strix::ModelRuntimeBinding weights;
        weights.manifest_layout_identity =
            manifest.layout_identity;
        weights.model_content_identity.bytes[0] = 1;
        strix::RuntimeTensorBinding runtime;
        runtime.tensor_index = 0;
        runtime.storage =
            strix::RuntimeTensorStorage::file_mapped;
        runtime.address = 1;
        runtime.bytes = table.byte_size;
        runtime.allocation_generation = 1;
        weights.tensors.push_back(runtime);

        strix::ExecutionTensorAccessPlan access;
        access.manifest_layout_identity =
            manifest.layout_identity;
        access.plan_identity = 1;
        strix::ExecutionTensorAccess table_access;
        table_access.step_id = 3;
        table_access.op = strix::SerialOp::ple;
        table_access.binding_slot = 0;
        table_access.tensor_indices = {0};
        access.accesses.push_back(table_access);

        strix::SerialSessionStateLayoutRequest layout_request;
        layout_request.context_tokens = 8;
        const auto layout =
            strix::build_serial_session_state_layout(
                model, layout_request);

        std::vector<std::byte> host_state(
            static_cast<std::size_t>(layout.host_total_bytes));
        strix::SessionStateMemoryBinding memory;
        memory.layout_identity = layout.layout_identity;
        memory.host_base =
            reinterpret_cast<std::uintptr_t>(host_state.data());
        memory.host_bytes = layout.host_total_bytes;
        memory.host_allocation_generation = 1;
        memory.device_base =
            static_cast<std::uintptr_t>(0x10000000ULL);
        memory.device_bytes = layout.device_total_bytes;
        memory.device_allocation_generation = 2;

        PrefetchFixtureBackend backend;
        strix::PleLookupPrefetchState prefetch;
        strix::start_ple_lookup_prefetch(
            contract, 3, 42, 0,
            manifest, weights, access,
            layout, memory, backend, prefetch);
        expect(
            prefetch.active &&
                prefetch.input_token_id == 42 &&
                prefetch.position == 0 &&
                prefetch.lookup_bytes ==
                    contract.head_count * contract.row_bytes,
            "PLE prefetch starts lookup before PLE dispatch");

        std::vector<std::byte> staging(
            static_cast<std::size_t>(prefetch.lookup_bytes));
        const auto prefetched_hash =
            strix::finish_ple_lookup_prefetch(
                42, 0, backend, prefetch, staging);
        expect(
            !prefetch.active &&
                prefetched_hash.row_count == contract.head_count &&
                staging.front() == std::byte{0x2a},
            "PLE prefetch finish consumes exact token lookup handle");

        strix::start_ple_lookup_prefetch(
            contract, 3, 43, 0,
            manifest, weights, access,
            layout, memory, backend, prefetch);
        strix::cancel_ple_lookup_prefetch(
            backend, prefetch);
        expect(
            !prefetch.active,
            "PLE prefetch cancellation clears backend lifecycle");
    }

    if (failures == 0) {
        std::cout
            << "PLE lookup tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
