#include "strix/ple_lookup.hpp"
#include "strix/iq4_nl.hpp"
#include "strix/ple_reclaim.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <cstring>
#include <vector>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace strix {
namespace {

constexpr std::uint64_t mask64 = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t splitmix_gamma = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t splitmix_m1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t splitmix_m2 = 0x94D049BB133111EBULL;
constexpr std::uint64_t prime_1 = 10007ULL;

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value = (value + splitmix_gamma) & mask64;
    value = ((value ^ (value >> 30U)) * splitmix_m1) & mask64;
    value = ((value ^ (value >> 27U)) * splitmix_m2) & mask64;
    return (value ^ (value >> 31U)) & mask64;
}

bool is_prime(std::uint64_t value) {
    if (value < 2) {
        return false;
    }
    if ((value % 2ULL) == 0ULL) {
        return value == 2;
    }
    for (std::uint64_t divisor = 3;
         divisor <= value / divisor;
         divisor += 2) {
        if ((value % divisor) == 0ULL) {
            return false;
        }
    }
    return true;
}

std::uint64_t next_prime(std::uint64_t value) {
    if (value == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("PLE prime search overflow");
    }
    ++value;
    while (!is_prime(value)) {
        if (value == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("PLE prime search overflow");
        }
        ++value;
    }
    return value;
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("PLE hash addition overflow");
    }
    return left + right;
}

}  // namespace

PleHashContract build_ple_hash_contract(
    const ModelContract& model) {
    if (const auto error = model.validate();
        !error.empty()) {
        throw std::invalid_argument(error);
    }

    const auto head_count64 =
        model.ple_head_count();
    if (model.ngram_size >
            PleHashContract::maximum_ngram ||
        head_count64 >
            PleHashContract::maximum_heads ||
        model.ngram_size < 2) {
        throw std::invalid_argument(
            "PLE hash geometry exceeds fixed runtime bounds");
    }

    PleHashContract contract;
    contract.ngram_size =
        static_cast<std::uint32_t>(
            model.ngram_size);
    contract.heads_per_ngram =
        static_cast<std::uint32_t>(
            model.heads_per_ngram);
    contract.head_count =
        static_cast<std::uint32_t>(
            head_count64);
    contract.head_dim =
        static_cast<std::uint32_t>(
            model.ple_head_dim());
    contract.eos_token_id =
        static_cast<std::uint32_t>(
            model.ple_eos_token_id);
    contract.row_bytes =
        model.ple_row_bytes_bf16();

    const auto maximum =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
    const auto multiplier_max =
        maximum /
        std::max<std::uint64_t>(
            model.vocab_size,
            1ULL);
    const auto half_bound =
        std::max<std::uint64_t>(
            1ULL,
            multiplier_max / 2ULL);

    const auto base_seed =
        model.ngram_hash_seed;
    for (std::uint32_t index = 0;
         index < contract.ngram_size;
         ++index) {
        const auto value =
            base_seed +
            splitmix_gamma *
                static_cast<std::uint64_t>(
                    index + 1U);
        contract.multipliers[index] =
            2ULL *
                (splitmix64(value) %
                 half_bound) +
            1ULL;
    }

    auto prime =
        model.ngram_vocab_size_base - 1ULL;
    std::uint64_t offset = 0;
    for (std::uint32_t head = 0;
         head < contract.head_count;
         ++head) {
        prime =
            next_prime(prime);
        contract.head_vocab_sizes[head] =
            prime;
        contract.head_offsets[head] =
            offset;
        offset =
            checked_add(
                offset,
                prime);
    }
    contract.minimum_table_rows =
        offset;
    return contract;
}

PleHashResult hash_ple_token(
    const PleHashContract& contract,
    std::uint32_t token_id,
    std::span<const std::uint32_t> previous_context) {
    if (contract.ngram_size < 2 ||
        contract.head_count == 0 ||
        contract.head_count >
            PleHashContract::maximum_heads ||
        previous_context.size() !=
            static_cast<std::size_t>(
                contract.ngram_size - 1U)) {
        throw std::invalid_argument(
            "PLE token hash context does not match contract");
    }

    std::array<std::uint64_t, PleHashContract::maximum_ngram> context{};
    context[0] =
        token_id;

    bool cut = false;
    for (std::uint32_t shift = 1;
         shift < contract.ngram_size;
         ++shift) {
        const auto source_index =
            previous_context.size() -
            static_cast<std::size_t>(
                shift);
        const auto token =
            previous_context[
                source_index];
        cut =
            cut ||
            token ==
                contract.eos_token_id;
        context[shift] =
            cut
                ? contract.eos_token_id
                : token;
    }

    PleHashResult result;
    result.row_count =
        contract.head_count;
    result.context_count =
        contract.ngram_size - 1U;

    for (std::uint32_t ngram = 2;
         ngram <= contract.ngram_size;
         ++ngram) {
        std::uint64_t mixed =
            context[0] *
            contract.multipliers[0];
        for (std::uint32_t position = 1;
             position < ngram;
             ++position) {
            mixed ^=
                context[position] *
                contract.multipliers[position];
        }

        const auto start =
            (ngram - 2U) *
            contract.heads_per_ngram;
        for (std::uint32_t slot = 0;
             slot <
                contract.heads_per_ngram;
             ++slot) {
            const auto head =
                start + slot;
            result.row_ids[head] =
                mixed %
                    contract.head_vocab_sizes[
                        head] +
                contract.head_offsets[
                    head];
        }
    }

    for (std::uint32_t index = 0;
         index + 1U <
            result.context_count;
         ++index) {
        result.next_context[index] =
            previous_context[
                previous_context.size() -
                result.context_count +
                index +
                1U];
    }
    if (result.context_count != 0) {
        result.next_context[
            result.context_count - 1U] =
            token_id;
    }

    return result;
}

void hash_ple_tokens(const PleHashContract& contract, std::span<const std::uint32_t> tokens,
    std::span<const std::uint32_t> previous, std::span<PleHashResult> out) {
    if (contract.ngram_size < 2 || contract.ngram_size > PleHashContract::maximum_ngram ||
        previous.size() != contract.ngram_size - 1U || out.size() != tokens.size())
        throw std::invalid_argument("PLE parallel hash geometry mismatch");
    const auto workers = std::min<std::size_t>(8U, tokens.size());
    std::vector<std::exception_ptr> errors(workers);
    const auto hash = [&](std::size_t worker) {
        try {
            const auto first = tokens.size() * worker / workers, last = tokens.size() * (worker + 1U) / workers;
            std::array<std::uint32_t, PleHashContract::maximum_ngram - 1U> context{};
            for (std::size_t i = 0; i < previous.size(); ++i)
                context[i] = first + i < previous.size() ? previous[first + i] : tokens[first + i - previous.size()];
            for (auto r = first; r < last; ++r) {
                out[r] = hash_ple_token(contract, tokens[r], std::span<const std::uint32_t>(context.data(), previous.size()));
                std::copy_n(out[r].next_context.begin(), previous.size(), context.begin());
            }
        } catch (...) { errors[worker] = std::current_exception(); }
    };
    // The guard also joins partial construction if thread creation refuses (no std::jthread: node libstdc++ lacks its
    // stop-token waits).
    std::vector<std::thread> threads;
    struct JoinAll { std::vector<std::thread>& t; ~JoinAll() { for (auto& x : t) if (x.joinable()) x.join(); } } join_all{threads};
    for (std::size_t t = 1; t < workers; ++t) threads.emplace_back(hash, t);
    if (workers) hash(0);
    for (auto& thread : threads) thread.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

std::uint64_t ple_hash_contract_identity(
    const PleHashContract& contract) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&](std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            hash ^= (value >> shift) & 0xffULL;
            hash *= 1099511628211ULL;
        }
    };
    mix(contract.ngram_size);
    mix(contract.heads_per_ngram);
    mix(contract.head_count);
    mix(contract.head_dim);
    mix(contract.eos_token_id);
    mix(contract.row_bytes);
    mix(contract.minimum_table_rows);
    for (const auto value : contract.multipliers) mix(value);
    for (const auto value : contract.head_offsets) mix(value);
    for (const auto value : contract.head_vocab_sizes) mix(value);
    return hash;
}

void start_ple_lookup_prefetch(
    const PleHashContract& contract,
    std::uint32_t ple_step_id,
    std::uint32_t input_token_id,
    std::uint64_t position,
    const ModelManifest& manifest,
    const ModelRuntimeBinding& weights,
    const ExecutionTensorAccessPlan& tensor_access,
    const SerialSessionStateLayout& state_layout,
    const SessionStateMemoryBinding& state_memory,
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch,
    bool preissue) {
    if (prefetch.active) {
        throw std::logic_error("PLE prefetch already active");
    }
    if (backend.implementation_identity() == 0 ||
        ple_hash_contract_identity(contract) == 0) {
        throw std::invalid_argument(
            "PLE prefetch requires initialized backend/hash contract");
    }

    const auto* access =
        find_execution_tensor_access(tensor_access, ple_step_id, 0);
    if (access == nullptr ||
        access->op != SerialOp::ple ||
        access->tensor_indices.size() != 1) {
        throw std::invalid_argument(
            "PLE prefetch cannot resolve unique n-gram table");
    }
    const auto tensor_index = access->tensor_indices.front();
    if (tensor_index >= manifest.tensors.size()) {
        throw std::out_of_range(
            "PLE prefetch table index exceeds manifest");
    }
    const auto* runtime =
        find_runtime_tensor_binding(weights, tensor_index);
    if (runtime == nullptr) {
        throw std::invalid_argument(
            "PLE prefetch table runtime binding is missing");
    }

    const auto ngram =
        resolve_session_state(
            state_layout,
            state_memory,
            SessionStateComponent::ngram_history);
    const auto context_count =
        static_cast<std::size_t>(contract.ngram_size - 1U);
    if (ngram.storage != SessionStateStorage::host ||
        ngram.address == 0 ||
        ngram.bytes < context_count * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            "PLE prefetch requires host n-gram history");
    }

    std::array<
        std::uint32_t,
        PleHashContract::maximum_ngram - 1> previous{};
    if (position == 0) {
        std::fill_n(
            previous.begin(),
            context_count,
            contract.eos_token_id);
    } else {
        std::memcpy(
            previous.data(),
            reinterpret_cast<const void*>(ngram.address),
            context_count * sizeof(std::uint32_t));
    }

    const auto hashed =
        hash_ple_token(
            contract,
            input_token_id,
            std::span<const std::uint32_t>(
                previous.data(), context_count));

    PleLookupRequest request;
    request.row_ids =
        std::span<const std::uint64_t>(
            hashed.row_ids.data(), hashed.row_count);
    request.row_bytes = contract.row_bytes;
    request.table_manifest = &manifest.tensors[tensor_index];
    request.table_runtime = runtime;

    const auto handle = backend.start(request);
    if (handle == 0) {
        throw std::runtime_error(
            "PLE prefetch backend returned zero handle");
    }
    if (hashed.row_count != 0 &&
        contract.row_bytes >
            std::numeric_limits<std::uint64_t>::max() /
                hashed.row_count) {
        backend.cancel(handle);
        throw std::overflow_error(
            "PLE prefetch byte count overflow");
    }

    prefetch.handle = handle;
    prefetch.backend_identity = backend.implementation_identity();
    prefetch.input_token_id = input_token_id;
    prefetch.position = position;
    prefetch.lookup_bytes =
        static_cast<std::uint64_t>(hashed.row_count) *
        contract.row_bytes;
    prefetch.hash = hashed;
    prefetch.active = true;
    // Pre-issue: start this step's reads now, while the host still enqueues the work ahead of the PLE op.
    // Only reads move earlier; the lookup above and its later finish() are unchanged.
    if (auto* mapped = dynamic_cast<FileMappedPleLookupBackend*>(&backend);
        preissue && mapped && FileMappedPleLookupBackend::preissue_enabled())
        mapped->preissue(std::span<const std::uint64_t>(hashed.row_ids.data(), hashed.row_count));
}

PleHashResult finish_ple_lookup_prefetch(
    std::uint32_t input_token_id,
    std::uint64_t position,
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch,
    std::span<std::byte> destination) {
    if (!prefetch.active ||
        prefetch.handle == 0 ||
        prefetch.backend_identity != backend.implementation_identity() ||
        prefetch.input_token_id != input_token_id ||
        prefetch.position != position ||
        prefetch.lookup_bytes > destination.size()) {
        throw std::invalid_argument(
            "PLE prefetch does not match active token/backend");
    }

    const auto handle = prefetch.handle;
    const auto hash = prefetch.hash;
    try {
        backend.finish(
            handle,
            destination.first(
                static_cast<std::size_t>(prefetch.lookup_bytes)));
    } catch (...) {
        backend.cancel(handle);
        prefetch = {};
        throw;
    }
    prefetch = {};
    return hash;
}

void cancel_ple_lookup_prefetch(
    PleLookupBackend& backend,
    PleLookupPrefetchState& prefetch) noexcept {
    if (prefetch.active && prefetch.handle != 0) {
        backend.cancel(prefetch.handle);
    }
    prefetch = {};
}

}  // namespace strix


namespace strix {
namespace {

constexpr std::uint64_t file_mapped_ple_backend_identity =
    0x504c45464d415002ULL;

std::uint64_t checked_add_lookup(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() -
            left) {
        throw std::overflow_error(
            "PLE lookup addition overflow");
    }
    return left + right;
}

std::uint64_t checked_mul_lookup(
    std::uint64_t left,
    std::uint64_t right) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() /
                left) {
        throw std::overflow_error(
            "PLE lookup multiplication overflow");
    }
    return left * right;
}

}  // namespace

std::uint64_t
FileMappedPleLookupBackend::implementation_identity() const noexcept {
    return file_mapped_ple_backend_identity;
}

PleLookupHandle
FileMappedPleLookupBackend::start(
    const PleLookupRequest& request) {
    if (pending_.active) {
        throw std::logic_error(
            "file-mapped PLE backend already has an active lookup");
    }
    if (request.row_ids.empty() ||
        request.row_ids.size() >
            PleHashContract::maximum_heads ||
        request.row_bytes == 0 ||
        request.table_manifest == nullptr ||
        request.table_runtime == nullptr) {
        throw std::invalid_argument(
            "file-mapped PLE lookup request is incomplete");
    }

    const auto& manifest =
        *request.table_manifest;
    const auto& runtime =
        *request.table_runtime;

    if (manifest.role !=
            TensorRole::ngram_table ||
        manifest.dimensions.size() != 2 ||
        manifest.dimensions[0] == 0 ||
        manifest.dimensions[1] == 0) {
        throw std::invalid_argument(
            "PLE lookup table manifest has invalid geometry");
    }
    if (manifest.ggml_type != 30 && manifest.ggml_type != 20) {
        throw std::invalid_argument(
            "file-mapped PLE lookup supports BF16 or IQ4_NL table rows");
    }
    const auto expected_row_bytes =
        checked_mul_lookup(
            manifest.dimensions[0],
            2ULL);
    if (request.row_bytes !=
        expected_row_bytes) {
        throw std::invalid_argument(
            "PLE lookup output row size does not match BF16 activation geometry");
    }
    std::uint64_t stored_row_bytes = expected_row_bytes;
    if (manifest.ggml_type == 20) {
        if (manifest.dimensions[0] % iq4_nl::block_elements != 0) {
            throw std::invalid_argument(
                "IQ4_NL PLE rows require complete 32-element blocks");
        }
        stored_row_bytes = checked_mul_lookup(
            manifest.dimensions[0] / iq4_nl::block_elements,
            iq4_nl::block_bytes);
    }
    if (runtime.storage !=
            RuntimeTensorStorage::file_mapped ||
        runtime.address == 0 ||
        runtime.bytes < manifest.byte_size) {
        throw std::invalid_argument(
            "file-mapped PLE backend requires a host-readable file mapping");
    }
    if (runtime.bytes > std::numeric_limits<std::uintptr_t>::max() - runtime.address) {
        throw std::overflow_error("PLE mapping address range overflows the host address space");
    }

    const auto minimum_bytes =
        checked_mul_lookup(
            manifest.dimensions[1],
            stored_row_bytes);
    if (runtime.bytes < minimum_bytes || manifest.byte_size < minimum_bytes) {
        throw std::invalid_argument(
            "PLE file mapping is smaller than declared table rows");
    }

    for (const auto row :
         request.row_ids) {
        if (row >=
            manifest.dimensions[1]) {
            throw std::out_of_range(
                "PLE row id exceeds table row count");
        }
    }

    if (next_handle_ == 0) {
        throw std::overflow_error(
            "PLE lookup handle space exhausted");
    }

    pending_.handle =
        next_handle_++;
    pending_.table_address =
        runtime.address;
    pending_.table_bytes =
        runtime.bytes;
    pending_.row_bytes =
        request.row_bytes;
    pending_.stored_row_bytes = stored_row_bytes;
    pending_.ggml_type = manifest.ggml_type;
    pending_.row_count =
        static_cast<std::uint32_t>(
            request.row_ids.size());
    std::copy(
        request.row_ids.begin(),
        request.row_ids.end(),
        pending_.row_ids.begin());
    pending_.mapping_generation = runtime.allocation_generation;
    pending_.residency = residency_;
    const auto* rolling_source = ple_reclaim::active();
    pending_.rolling_source = rolling_source && rolling_source->rolling() ? reinterpret_cast<std::uintptr_t>(rolling_source) : 0U;
    pending_.residency_epoch = residency_ ? residency_->epoch() : 0U;
    pending_.active = true;
    return pending_.handle;
}

void FileMappedPleLookupBackend::finish(
    PleLookupHandle handle,
    std::span<std::byte> destination) {
    // A pre-issue of this lookup never outlives finish(), on every exit path.
    struct JoinPreissue {
        FileMappedPleLookupBackend& backend;
        ~JoinPreissue() { backend.join_preissue(); }
    } join_preissue_on_exit{*this};
    if (!pending_.active ||
        handle == 0 ||
        handle != pending_.handle) {
        throw std::invalid_argument(
            "PLE lookup handle is not active");
    }

    const auto required =
        checked_mul_lookup(
            pending_.row_count,
            pending_.row_bytes);
    if (destination.size() <
        required) {
        throw std::invalid_argument(
            "PLE lookup destination is undersized");
    }

    auto* const row_source = ple_reclaim::active();
    if (pending_.rolling_source && pending_.rolling_source != reinterpret_cast<std::uintptr_t>(row_source))
        throw std::runtime_error("PLE O_DIRECT: lookup outlived its rolling source binding");
    if (row_source && row_source->rolling() && pending_.residency && !pending_.residency->owned_sources())
        throw std::runtime_error("PLE O_DIRECT: unowned turn cannot bypass the rolling source");
    if (prompt_view_) {
        const bool valid = pending_.residency
            ? pending_.residency->owns_prompt_view(*prompt_view_, pending_.table_address,
                pending_.mapping_generation, pending_.residency_epoch, prompt_offset_)
            : row_source && row_source->rolling() && pending_.rolling_source &&
                row_source->owns_prompt_source(*prompt_view_, prompt_table_, prompt_offset_);
        if (!valid || pending_.table_address != prompt_table_)
            throw std::runtime_error("PLE O_DIRECT: prompt view does not match its active source");
    }
    const bool owned = pending_.residency ? pending_.residency->owned_sources() : (row_source && row_source->rolling());
    const auto copy = [&] {
    for (std::uint32_t index = 0;
         index < pending_.row_count;
         ++index) {
        const auto offset =
            checked_mul_lookup(
                pending_.row_ids[index],
                pending_.stored_row_bytes);
        if (offset >
                pending_.table_bytes ||
            pending_.stored_row_bytes >
                pending_.table_bytes -
                    offset) {
            throw std::out_of_range(
                "PLE lookup row range exceeds mapped table");
        }
        auto* output = destination.data() + static_cast<std::size_t>(
            checked_mul_lookup(index, pending_.row_bytes));
        thread_local std::vector<std::byte> stored;
        const std::byte* input;
        if (owned) {
            stored.resize(static_cast<std::size_t>(pending_.stored_row_bytes));
            if (prompt_view_) prompt_view_->copy(prompt_offset_ + offset, stored);
            else if (pending_.residency) pending_.residency->read_row(pending_.row_ids[index], stored);
            else row_source->read_row(pending_.table_address, pending_.row_ids[index], stored);
            input = stored.data();
        } else {
            input = reinterpret_cast<const std::byte*>(pending_.table_address) + static_cast<std::size_t>(offset);
        }
        if (pending_.ggml_type == 30) {
            std::memcpy(output, input, static_cast<std::size_t>(pending_.row_bytes));
            continue;
        }
        // Only selected IQ4_NL rows are decoded. Storage remains quantized;
        // the existing activation contract is BF16, rounded to nearest-even.
        // The row is decoded into host-cached memory and copied once: the destination is
        // often a mapped, uncached staging buffer where 2-byte stores are slow.
        const auto blocks = pending_.stored_row_bytes / iq4_nl::block_bytes;
        thread_local std::vector<std::byte> decoded;
        if (decoded.size() < pending_.row_bytes) decoded.resize(static_cast<std::size_t>(pending_.row_bytes));
        auto* const final_output = output;
        output = decoded.data();
        for (std::uint64_t block = 0; block < blocks; ++block) {
            const std::span<const std::byte, iq4_nl::block_bytes> packed(
                input + static_cast<std::size_t>(block * iq4_nl::block_bytes),
                iq4_nl::block_bytes);
            if (!std::isfinite(iq4_nl::fp16_bits_to_float(iq4_nl::scale_bits(packed)))) {
                throw std::invalid_argument("IQ4_NL PLE scale must be finite");
            }
            for (std::size_t element = 0; element < iq4_nl::block_elements; ++element) {
                const auto value = iq4_nl::dequantized_value(packed, element);
                const auto bits = std::bit_cast<std::uint32_t>(value);
                const auto bf16 = static_cast<std::uint16_t>(
                    (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
                const auto column = block * iq4_nl::block_elements + element;
                std::memcpy(output + static_cast<std::size_t>(column * 2U), &bf16, sizeof(bf16));
            }
        }
        std::memcpy(final_output, decoded.data(), static_cast<std::size_t>(pending_.row_bytes));
    }

    };
    if (prompt_view_) copy();
    else if (pending_.residency)
        pending_.residency->source(pending_.table_address, pending_.mapping_generation,
            std::span<const std::uint64_t>(pending_.row_ids.data(), pending_.row_count), copy, pending_.residency_epoch);
    else if (auto* reclaim = ple_reclaim::active(); reclaim && reclaim->rolling())
        reclaim->source_copy(pending_.table_address,
            std::span<const std::uint64_t>(pending_.row_ids.data(), pending_.row_count), copy);
    else copy();
    pending_ = {};
}

struct FileMappedPleLookupBackend::PreissueWorker {
    std::mutex mutex;
    std::condition_variable wake;
    std::thread thread;
    std::function<void()> job;
    bool stop = false, busy = false;
    ~PreissueWorker() {
        { const std::lock_guard lock(mutex); stop = true; }
        wake.notify_all();
        if (thread.joinable()) thread.join();
    }
    void wait() noexcept {
        std::unique_lock lock(mutex);
        wake.wait(lock, [this] { return !busy; });
    }
    void submit(std::function<void()> work) {
        std::unique_lock lock(mutex);
        wake.wait(lock, [this] { return !busy; });
        if (!thread.joinable()) thread = std::thread([this] { run(); });
        job = std::move(work); busy = true;
        lock.unlock();
        wake.notify_all();
    }
    void run() noexcept {
        std::unique_lock lock(mutex);
        for (;;) {
            wake.wait(lock, [this] { return stop || static_cast<bool>(job); });
            if (stop && !job) return;
            auto work = std::move(job); job = nullptr;
            lock.unlock();
            try { work(); } catch (...) {}
            work = nullptr;   // release the captured source before the caller can observe completion
            lock.lock();
            busy = false;
            wake.notify_all();
        }
    }
};

namespace {
const char* preissue_environment(const char* name) noexcept {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once, before the first lookup
#endif
    return std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}
}  // namespace

bool FileMappedPleLookupBackend::preissue_enabled_from(const char* preissue, const char* defaults, const char* baseline) noexcept {
    if ((baseline && std::strcmp(baseline, "1") == 0) || (defaults && defaults[0] == '0')) return false;
    return !(preissue && std::strcmp(preissue, "0") == 0);
}

bool FileMappedPleLookupBackend::preissue_enabled() noexcept {
    static const bool on = preissue_enabled_from(preissue_environment("STRIX_PLE_PREISSUE"),
        preissue_environment("STRIX_QPF2K_DEFAULTS"), preissue_environment("STRIX_BASELINE"));
    return on;
}

FileMappedPleLookupBackend::FileMappedPleLookupBackend() = default;
FileMappedPleLookupBackend::~FileMappedPleLookupBackend() { join_preissue(); }

void FileMappedPleLookupBackend::preissue(std::span<const std::uint64_t> rows) noexcept {
    try {
        if (!preissue_route_ || !pending_.active || rows.empty() || prompt_view_) return;
        // The same source finish() will use; non-owned (mmap) sources gain nothing and are left alone.
        auto turn = pending_.residency;
        auto* const active = turn ? nullptr : ple_reclaim::active();
        if (turn ? !turn->owned_sources() : !(active && active->rolling() && pending_.rolling_source)) return;
        // The helper owns what it reads through (the turn, or the rolling source), never a borrowed pointer.
        std::shared_ptr<ple_reclaim::Prefetcher> source;
        if (!turn && !(source = active->weak_from_this().lock())) return;
        const auto table = pending_.table_address;
        const auto generation = pending_.mapping_generation;
        std::vector<std::uint64_t> ids(rows.begin(), rows.end());
        if (!preissue_) preissue_ = std::make_unique<PreissueWorker>();
        preissue_->submit([turn = std::move(turn), source = std::move(source), table, generation, ids = std::move(ids)] {
            // Background scope: these joins are not demand waits.
            const ple_reclaim::DirectReadScope scope(true);
            if (turn) (void)turn->warm(table, generation, ids);
            else source->source_copy(table, ids, [] {});
        });
    } catch (...) {}
}

void FileMappedPleLookupBackend::join_preissue() noexcept {
    if (preissue_) preissue_->wait();
}

void FileMappedPleLookupBackend::cancel(
    PleLookupHandle handle) noexcept {
    join_preissue();
    if (pending_.active &&
        handle != 0 &&
        handle == pending_.handle) {
        pending_ = {};
    }
}

std::uint64_t PlannedPleLookupBackend::cache_reservation_bytes(PlannedPleLookupPolicy policy) {
    if ((policy.cache_capacity_rows == 0) != (policy.cache_byte_budget == 0))
        throw std::invalid_argument("PLE row cache requires both capacity and byte budget");
    if (policy.cache_capacity_rows > std::numeric_limits<std::size_t>::max() / sizeof(CacheSlot) ||
        policy.cache_byte_budget > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("PLE row cache reservation exceeds native extent");
    return checked_add_lookup(policy.cache_byte_budget,
        checked_mul_lookup(policy.cache_capacity_rows, sizeof(CacheSlot)));
}
PlannedPleCacheStats PlannedPleLookupBackend::cache_stats() const noexcept { return cache_counters_; }
void PlannedPleLookupBackend::cache_count(std::uint64_t& counter) noexcept {
    if (counter == std::numeric_limits<std::uint64_t>::max()) cache_counters_.saturated = true;
    else ++counter;
}
void PlannedPleLookupBackend::cache_touch(std::size_t slot) noexcept {
    const auto previous = cache_slots_[slot].valid ? cache_slots_[slot].age : cache_rows_;
    for (auto& other : cache_slots_)
        if (other.valid && other.age < previous) ++other.age;
    cache_slots_[slot].age = 0;
}
void PlannedPleLookupBackend::cache_admit(const PleLookupRequest& request) {
    if (!policy_.cache_capacity_rows) return;
    const auto& m = *request.table_manifest;
    const auto& r = *request.table_runtime;
    if (!r.allocation_generation) throw std::invalid_argument("PLE row cache requires a live table allocation generation");
    if (request.row_bytes > policy_.cache_byte_budget)
        throw std::invalid_argument("PLE row cache byte budget cannot retain one complete row");
    const bool same = cache_manifest_ == &m && cache_runtime_ == &r && cache_row_bytes_ == request.row_bytes &&
        cache_offset_ == m.absolute_offset && cache_table_bytes_ == m.byte_size && cache_table_rows_ == m.dimensions[1] &&
        cache_binding_.tensor_index == r.tensor_index && cache_binding_.storage == r.storage &&
        cache_binding_.address == r.address && cache_binding_.bytes == r.bytes &&
        cache_binding_.allocation_generation == r.allocation_generation && cache_binding_.owner_node == r.owner_node;
    if (same) return;
    if (cache_manifest_) cache_count(cache_counters_.invalidations);
    for (auto& slot : cache_slots_) slot = {};
    cache_counters_.resident_rows = 0;
    cache_manifest_ = &m; cache_runtime_ = &r; cache_binding_ = r;
    cache_row_bytes_ = request.row_bytes; cache_offset_ = m.absolute_offset;
    cache_table_bytes_ = m.byte_size; cache_table_rows_ = m.dimensions[1];
    cache_rows_ = std::min(policy_.cache_capacity_rows, policy_.cache_byte_budget / cache_row_bytes_);
}
void PlannedPleLookupBackend::cache_insert(std::uint64_t row, const std::byte* bytes) noexcept {
    std::size_t chosen = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(cache_rows_); ++i) {
        if (cache_slots_[i].valid && cache_slots_[i].row == row) { cache_touch(i); return; }
        if (!cache_slots_[i].valid) { chosen = i; break; }
        if (cache_slots_[i].age > cache_slots_[chosen].age) chosen = i;
    }
    auto& slot = cache_slots_[chosen];
    if (slot.valid) cache_count(cache_counters_.evicted);
    else ++cache_counters_.resident_rows;
    cache_touch(chosen);
    std::memcpy(cache_bytes_.data() + chosen * static_cast<std::size_t>(cache_row_bytes_), bytes,
        static_cast<std::size_t>(cache_row_bytes_));
    slot.row = row; slot.valid = true;
    cache_count(cache_counters_.inserted);
}

PlannedPleLookupBackend::PlannedPleLookupBackend(
    SparseLookupIoSource& source,
    PlannedPleLookupPolicy policy, std::shared_ptr<const void> source_owner)
    : source_(&source),
      policy_(policy), source_owner_(std::move(source_owner)) {
    if (policy_.cpu_helper_threads > 8U)
        throw std::invalid_argument("planned PLE CPU helper budget must be zero or 1..8");
    if (policy_.cache_capacity_rows && (!source_owner_ || !policy_.cache_model_content.valid()))
        throw std::invalid_argument("PLE row cache requires retained immutable source and model content identity");
    const auto reservation = cache_reservation_bytes(policy_);
    cache_slots_.resize(static_cast<std::size_t>(policy_.cache_capacity_rows));
    cache_bytes_.resize(static_cast<std::size_t>(policy_.cache_byte_budget));
    cache_counters_.reserved_bytes = reservation;
    if (policy_.max_inflight_ios == 0) {
        throw std::invalid_argument(
            "planned PLE lookup requires non-zero inflight depth");
    }
}

std::uint64_t
PlannedPleLookupBackend::implementation_identity() const noexcept {
    constexpr std::uint64_t base =
        0x504c45504c414e31ULL;
    const auto legacy = base ^
           (policy_.max_rows_per_io << 17U) ^
           (policy_.max_inflight_ios << 33U);
    if (!policy_.cpu_helper_threads && !policy_.cache_capacity_rows) return legacy;
    auto hash = legacy;
    for (const auto value : {0x504c454350554231ULL, static_cast<unsigned long long>(policy_.cpu_helper_threads)})
        for (unsigned shift = 0; shift < 64; shift += 8)
            hash = (hash ^ ((value >> shift) & 255ULL)) * 1099511628211ULL;
    if (policy_.cache_capacity_rows)
        for (const auto value : {0x504c45524f574331ULL, static_cast<unsigned long long>(policy_.cache_capacity_rows),
                static_cast<unsigned long long>(policy_.cache_byte_budget)})
            for (unsigned shift = 0; shift < 64; shift += 8)
                hash = (hash ^ ((value >> shift) & 255ULL)) * 1099511628211ULL;
    if (policy_.cache_capacity_rows)
        for (const auto byte : policy_.cache_model_content.bytes)
            hash = (hash ^ byte) * 1099511628211ULL;
    return hash ? hash : 1;
}

PleLookupHandle
PlannedPleLookupBackend::start(
    const PleLookupRequest& request) {
    if (pending_.active) {
        throw std::logic_error(
            "planned PLE backend already has an active lookup");
    }
    if (source_ == nullptr ||
        request.row_ids.empty() ||
        request.row_ids.size() >
            PleHashContract::maximum_heads ||
        request.row_bytes == 0 ||
        request.table_manifest == nullptr ||
        request.table_runtime == nullptr) {
        throw std::invalid_argument(
            "planned PLE lookup request is incomplete");
    }

    const auto& manifest =
        *request.table_manifest;
    const auto& runtime =
        *request.table_runtime;

    if (manifest.role !=
            TensorRole::ngram_table ||
        manifest.dimensions.size() < 2 ||
        manifest.dimensions[0] == 0 ||
        manifest.dimensions[1] == 0 ||
        manifest.ggml_type != 30) {
        throw std::invalid_argument(
            "planned PLE lookup requires a BF16 n-gram table");
    }

    const auto expected_row_bytes =
        checked_mul_lookup(
            manifest.dimensions[0],
            2ULL);
    if (request.row_bytes !=
        expected_row_bytes) {
        throw std::invalid_argument(
            "planned PLE row size does not match table geometry");
    }
    if (!runtime_tensor_file_backed(
            runtime.storage) ||
        runtime.bytes <
            manifest.byte_size) {
        throw std::invalid_argument(
            "planned PLE lookup requires a file-backed tensor binding");
    }

    const auto minimum_bytes =
        checked_mul_lookup(
            manifest.dimensions[1],
            request.row_bytes);
    if (manifest.byte_size <
        minimum_bytes) {
        throw std::invalid_argument(
            "planned PLE tensor is smaller than declared rows");
    }

    for (const auto row :
         request.row_ids) {
        if (row >=
            manifest.dimensions[1]) {
            throw std::out_of_range(
                "planned PLE row id exceeds table row count");
        }
    }

    cache_admit(request);
    std::array<std::uint64_t, PleHashContract::maximum_heads> missing_rows{};
    std::array<std::size_t, PleHashContract::maximum_heads> cached_slots{};
    cached_slots.fill(std::numeric_limits<std::size_t>::max());
    std::size_t missing_count = 0;
    for (std::size_t output = 0; output < request.row_ids.size(); ++output) {
        if (policy_.cache_capacity_rows) {
            for (std::size_t slot = 0; slot < static_cast<std::size_t>(cache_rows_); ++slot)
                if (cache_slots_[slot].valid && cache_slots_[slot].row == request.row_ids[output]) {
                    cached_slots[output] = slot;
                    break;
                }
        }
        if (cached_slots[output] == std::numeric_limits<std::size_t>::max()) {
            missing_rows[missing_count++] = request.row_ids[output];
            if (policy_.cache_capacity_rows) cache_count(cache_counters_.misses);
        } else cache_count(cache_counters_.hits);
    }
    SparseLookupRequest sparse_request;
    sparse_request.table_rows =
        manifest.dimensions[1];
    sparse_request.row_bytes =
        request.row_bytes;
    sparse_request.cache_capacity_rows = 0;
    sparse_request.max_rows_per_io =
        policy_.max_rows_per_io;
    sparse_request.max_inflight_ios =
        policy_.max_inflight_ios;
    sparse_request.row_ids = std::span<const std::uint64_t>(missing_rows).first(missing_count);

    SparseLookupPlan plan;
    if (missing_count) plan = build_sparse_lookup_plan(sparse_request);
    for (auto& operation :
         plan.io_plan) {
        if (operation.byte_offset >
                manifest.byte_size ||
            operation.byte_count >
                manifest.byte_size -
                    operation.byte_offset) {
            throw std::out_of_range(
                "planned PLE I/O exceeds tensor range");
        }
        operation.byte_offset =
            checked_add_lookup(
                manifest.absolute_offset,
                operation.byte_offset);
    }

    if (next_handle_ == 0) {
        throw std::overflow_error(
            "planned PLE lookup handle space exhausted");
    }

    const auto handle =
        next_handle_++;
    auto* source =
        source_;
    const auto helper_threads = policy_.cpu_helper_threads;
    std::future<SparseLookupIoExecution> future;
    if (missing_count) future = std::async(
            std::launch::async,
            [plan, source, helper_threads]() mutable {
                if (helper_threads)
                    return execute_sparse_lookup_io_plan(plan, *source, helper_threads);
                return execute_sparse_lookup_io_plan(
                    plan,
                    *source);
            });

    pending_.cached_slots = cached_slots;
    pending_.cache_enabled = policy_.cache_capacity_rows != 0;
    pending_.handle =
        handle;
    pending_.row_bytes =
        request.row_bytes;
    pending_.row_count =
        static_cast<std::uint32_t>(
            request.row_ids.size());
    std::copy(
        request.row_ids.begin(),
        request.row_ids.end(),
        pending_.row_ids.begin());
    pending_.io_plan =
        std::move(plan.io_plan);
    pending_.future =
        std::move(future);
    pending_.active = true;
    return handle;
}

void PlannedPleLookupBackend::finish(
    PleLookupHandle handle,
    std::span<std::byte> destination) {
    if (!pending_.active ||
        handle == 0 ||
        handle != pending_.handle) {
        throw std::invalid_argument(
            "planned PLE lookup handle is not active");
    }

    const auto required =
        checked_mul_lookup(
            pending_.row_count,
            pending_.row_bytes);
    if (destination.size() <
        required) {
        throw std::invalid_argument(
            "planned PLE lookup destination is undersized");
    }

    auto current =
        std::move(pending_);
    pending_ = {};

    const auto execution = current.future.valid() ? current.future.get() : SparseLookupIoExecution{};
    if (execution.operation_payloads.size() !=
        current.io_plan.size()) {
        throw std::logic_error(
            "planned PLE execution payload count mismatch");
    }

    for (std::uint32_t output_row = 0;
         output_row < current.row_count;
         ++output_row) {
        const auto row =
            current.row_ids[output_row];
        const std::byte* source_row = nullptr;
        const auto cached = current.cached_slots[output_row];
        if (current.cache_enabled && cached != std::numeric_limits<std::size_t>::max())
            source_row = cache_bytes_.data() + cached * static_cast<std::size_t>(current.row_bytes);

        for (std::size_t operation_index = 0;
             source_row == nullptr && operation_index <
                current.io_plan.size();
             ++operation_index) {
            const auto& operation =
                current.io_plan[
                    operation_index];
            if (row <
                    operation.first_row ||
                row >=
                    operation.first_row +
                        operation.row_count) {
                continue;
            }

            const auto within_rows =
                row -
                operation.first_row;
            const auto within_bytes =
                checked_mul_lookup(
                    within_rows,
                    current.row_bytes);
            const auto& payload =
                execution.operation_payloads[
                    operation_index];
            if (within_bytes >
                    payload.size() ||
                current.row_bytes >
                    payload.size() -
                        static_cast<std::size_t>(
                            within_bytes)) {
                throw std::logic_error(
                    "planned PLE payload is missing requested row bytes");
            }

            source_row =
                payload.data() +
                static_cast<std::size_t>(
                    within_bytes);
            break;
        }

        if (source_row == nullptr) {
            throw std::logic_error(
                "planned PLE requested row has no owning I/O operation");
        }

        std::memcpy(
            destination.data() +
                static_cast<std::size_t>(
                    checked_mul_lookup(
                        output_row,
                        current.row_bytes)),
            source_row,
            static_cast<std::size_t>(
                current.row_bytes));
    }
    if (current.cache_enabled)
        for (std::uint32_t row = 0; row < current.row_count; ++row)
            cache_insert(current.row_ids[row], destination.data() +
                static_cast<std::size_t>(row * current.row_bytes));
}

void PlannedPleLookupBackend::cancel(
    PleLookupHandle handle) noexcept {
    if (!pending_.active ||
        handle == 0 ||
        handle != pending_.handle) {
        return;
    }

    auto cancelled =
        std::move(pending_);
    pending_ = {};
    // std::future from std::launch::async joins on destruction. The source
    // contract has no cancellation primitive yet, so cancel is fail-closed:
    // results are discarded and no exception escapes this noexcept boundary.
    static_cast<void>(cancelled);
}

}  // namespace strix
