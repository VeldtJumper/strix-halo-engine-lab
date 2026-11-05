#include "strix/startup_markers.hpp"
#include "strix/hip_model_residency.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/registered_backing_lifetime.hpp"
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
#include "strix/hip_model_residency_diagnostic.hpp"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace strix {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
namespace model_residency_diagnostic {
namespace {
thread_local FaultPlan fault_plan;
thread_local Stats counters;
struct Quarantined { std::shared_ptr<const MappedModelFile> mapping; std::vector<std::uintptr_t> registrations; };
// Deliberately no TLS destructor: an unretired failed registration must never
// release its mapping during test failure/process teardown.
thread_local auto* quarantined = new std::vector<Quarantined>;
}
void set_fault_plan(FaultPlan plan) {
    if (!plan.occurrence) throw std::invalid_argument("diagnostic fault occurrence must be positive");
    fault_plan = plan; counters = {};
}
Stats stats() noexcept { return counters; }
void retire_quarantined_mappings() {
    for (auto& entry : *quarantined) {
        for (auto& address : entry.registrations) if (address) {
            if (hipHostUnregister(reinterpret_cast<void*>(address)) != hipSuccess)
                throw std::runtime_error("diagnostic quarantine retirement failed");
            address = 0;
        }
    }
    quarantined->clear();
}
} // namespace model_residency_diagnostic
#endif
namespace {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
bool inject(model_residency_diagnostic::Fault fault, std::uint64_t occurrence) {
    return model_residency_diagnostic::fault_plan.fault == fault &&
        model_residency_diagnostic::fault_plan.occurrence == occurrence;
}
#endif
// Fine-grained (MTYPE UC) model ranges: streamed weights do not displace cached activations;
// coarse-grained was measured slower on the decode window (44.3 vs 45.0 tok/s).
// STRIX_WEIGHTS_COARSE_GRAINED=1 selects it for experiments.
unsigned model_register_flags() {
    static const unsigned flags = std::getenv("STRIX_WEIGHTS_COARSE_GRAINED") ?
        hipHostRegisterMapped | hipExtHostRegisterCoarseGrained : hipHostRegisterMapped;
    return flags;
}
hipError_t selective_register(void* address, std::size_t bytes) {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    ++model_residency_diagnostic::counters.registrations;
#endif
    const auto error = hipHostRegister(address, bytes, model_register_flags());
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    if (error == hipSuccess) ++model_residency_diagnostic::counters.acquired_registrations;
#endif
    return error;
}
hipError_t selective_alias(void** address, void* host) {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    const auto count = ++model_residency_diagnostic::counters.alias_queries;
    if (inject(model_residency_diagnostic::Fault::device_alias, count)) return hipErrorInvalidValue;
#endif
    const auto error = hipHostGetDevicePointer(address, host, 0);
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    if (error == hipSuccess && inject(model_residency_diagnostic::Fault::alias_alignment, count))
        *address = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(*address) + 1U);
#endif
    return error;
}
hipError_t selective_allocate(void** address, std::size_t bytes) {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    const auto count = ++model_residency_diagnostic::counters.allocations;
    if (inject(model_residency_diagnostic::Fault::allocation, count)) return hipErrorOutOfMemory;
#endif
    // STRIX_MODEL_COPY_ALL=fine|uc: the arena's memory type (default hipMalloc).
    const auto mode = hip_model_copy_all_mode();
    const auto error = mode == 2U ? hipExtMallocWithFlags(address, bytes, hipDeviceMallocFinegrained) :
        mode == 3U ? hipExtMallocWithFlags(address, bytes, hipDeviceMallocUncached) : hipMalloc(address, bytes);
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    if (error == hipSuccess) ++model_residency_diagnostic::counters.acquired_allocations;
#endif
    return error;
}
hipError_t selective_copy(void* destination, const void* source, std::size_t bytes) {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    const auto count = ++model_residency_diagnostic::counters.copies;
    if (inject(model_residency_diagnostic::Fault::copy, count)) return hipErrorInvalidValue;
#endif
    return hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice);
}
hipError_t selective_unregister(void* address) {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    const auto count = ++model_residency_diagnostic::counters.unregisters;
    if (inject(model_residency_diagnostic::Fault::unregister, count)) return hipErrorInvalidValue;
#endif
    const auto error = hipHostUnregister(address);
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    if (error == hipSuccess) ++model_residency_diagnostic::counters.released_registrations;
#endif
    return error;
}
void selective_free(void* address) noexcept {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    ++model_residency_diagnostic::counters.frees;
#endif
    const auto error = hipFree(address);
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    if (error == hipSuccess) ++model_residency_diagnostic::counters.released_allocations;
#else
    static_cast<void>(error);
#endif
}

// One setup-only pinned source. The object exists before allocation starts,
// so a successful HIP allocation survives even a later guard exception.
class ExpertCopyStaging final {
public:
    explicit ExpertCopyStaging(const HipSetupQuiescenceGuard& guard) : guard_(guard) {}
    ExpertCopyStaging(const ExpertCopyStaging&) = delete;
    ExpertCopyStaging& operator=(const ExpertCopyStaging&) = delete;
    ~ExpertCopyStaging() noexcept {
        if (!data_) return;
        guard_.recover_or_fail_stop();
        try { retire(); } catch (...) { fail_stop(); }
    }
    void allocate(std::uint64_t bytes) {
        if (!bytes) return;
        guard_.run([&] {
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
            const auto count = ++model_residency_diagnostic::counters.staging_allocations;
            if (inject(model_residency_diagnostic::Fault::staging_allocation, count))
                throw std::runtime_error("selective residency staging allocation: injected failure");
#endif
            void* acquired = nullptr;
            const auto status = hipHostMalloc(&acquired, static_cast<std::size_t>(bytes), hipHostMallocDefault);
            if (status != hipSuccess)
                throw std::runtime_error(std::string("selective residency staging allocation: ") + hipGetErrorString(status));
            data_ = acquired; // Publish ownership before any post-allocation check.
            if (!data_) throw std::runtime_error("selective residency staging allocation returned no storage");
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
            ++model_residency_diagnostic::counters.acquired_staging_allocations;
            if (inject(model_residency_diagnostic::Fault::staging_post_allocate, 1))
                throw std::runtime_error("selective residency staging post-allocation: injected failure");
#endif
        });
    }
    void* data() const noexcept { return data_; }
    void retire() {
        if (!data_) return;
        guard_.run([&] {
            auto* acquired = data_;
            data_ = nullptr; // Never retry an uncertain free, including unwinding.
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
            ++model_residency_diagnostic::counters.staging_frees;
#endif
            if (hipHostFree(acquired) != hipSuccess) fail_stop();
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
            ++model_residency_diagnostic::counters.released_staging_allocations;
#endif
        });
    }
private:
    [[noreturn]] static void fail_stop() noexcept {
        std::fputs("HIP expert staging retirement is unproved; terminating without owner cleanup\n", stderr);
        std::fflush(nullptr);
        std::_Exit(3);
    }
    const HipSetupQuiescenceGuard& guard_;
    void* data_ = nullptr;
};

std::uint64_t expert_staging_capacity(const SelectiveCopyPlan& plan, HipModelCopyTier tier) {
    if (tier != HipModelCopyTier::experts && tier != HipModelCopyTier::all) return 0;
    std::uint64_t cap = 32ULL * 1024ULL * 1024ULL;
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    const auto requested = model_residency_diagnostic::fault_plan.staging_capacity_bytes;
    if (requested > cap) throw std::invalid_argument("diagnostic staging capacity exceeds production cap");
    if (requested) cap = requested;
#endif
    std::uint64_t largest = 0;
    for (const auto& range : plan.ranges) largest = std::max(largest, range.file_end - range.file_begin);
    return std::min(cap, largest);
}

// Process-local allocator lease IDs, never GPU/host virtual addresses.
std::atomic<std::uint64_t> next_residency_identity{1};
std::uint64_t residency_identity() {
    auto value = next_residency_identity.load(std::memory_order_relaxed);
    for (;;) {
        if (!value || value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("HIP residency lease identity exhausted");
        if (next_residency_identity.compare_exchange_weak(value, value + 1U, std::memory_order_relaxed)) return value;
    }
}

void check_hip(
    hipError_t error,
    const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) +
            ": " +
            hipGetErrorString(error));
    }
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() -
            left) {
        throw std::overflow_error(
            "HIP model residency range overflow");
    }
    return left + right;
}

std::uint64_t page_size() {
#if defined(__linux__)
    const auto value =
        ::sysconf(_SC_PAGESIZE);
    if (value <= 0) {
        throw std::runtime_error(
            "cannot determine host page size");
    }
    return static_cast<std::uint64_t>(
        value);
#else
    return 4096;
#endif
}

std::uint64_t align_down(
    std::uint64_t value,
    std::uint64_t alignment) {
    return value -
           value % alignment;
}

std::uint64_t align_up(
    std::uint64_t value,
    std::uint64_t alignment) {
    const auto remainder =
        value % alignment;
    if (remainder == 0) {
        return value;
    }
    return checked_add(
        value,
        alignment - remainder);
}

bool should_promote(
    const ManifestTensor& tensor,
    const HipModelResidencyPolicy& policy) noexcept {
    if (tensor.placement.kind ==
        PlacementKind::remote) {
        return false;
    }
    if (tensor.residency ==
            ResidencyPolicy::file_backed_candidate &&
        !policy.promote_file_backed_candidates) {
        return false;
    }
    return true;
}

bool should_register(const ManifestTensor& tensor,
    const HipModelResidencyPolicy& policy) noexcept {
    return should_promote(tensor, policy) &&
        !((policy.copy_tier == HipModelCopyTier::experts || policy.copy_tier == HipModelCopyTier::all) &&
          selective_copy_eligible(tensor, policy.copy_tier));
}

struct HostRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

std::vector<HostRange> build_registration_ranges(
    const MappedModelFile& mapped,
    const ModelManifest& manifest,
    const HipModelResidencyPolicy& policy) {
    const auto page =
        page_size();
    std::vector<HostRange> merged;
    for (const auto range : build_registered_residency_ranges(manifest,policy,mapped.bytes(),page))
        merged.push_back({range.begin,range.end});
    // STRIX_MODEL_REGISTER_CHUNK_BYTES=N (default 268435456 = 256 MiB; 0 = one registration per merged range): register each merged range as page-aligned
    // pieces of at most N bytes. An invalidated page of a userptr registration makes the driver revalidate the
    // whole registration while every queue of the process is evicted; smaller pieces bound that stall. A tensor
    // may then span pieces: binding requires their device aliases to be contiguous (checked after registering).
    static const std::uint64_t chunk = [] {
        const char* v = std::getenv("STRIX_MODEL_REGISTER_CHUNK_BYTES");
        return v ? std::strtoull(v, nullptr, 10) : 268435456ULL;
    }();
    if (chunk == 0U) return merged;
    const auto piece = std::max<std::uint64_t>(align_down(chunk, page), page);
    std::vector<HostRange> pieces;
    for (const auto range : merged)
        for (auto begin = range.begin; begin < range.end; begin += piece)
            pieces.push_back(HostRange{begin, std::min(range.end, checked_add(begin, piece))});
    return pieces;
}

// Selected source interiors must remain unregistered. Retaining the mapping
// keeps their bytes valid without requiring the source file cache to stay pinned.
void validate_expert_registration(const SelectiveCopyPlan& selected,
    const std::vector<HostRange>& registered, std::uint64_t page) {
    std::size_t first = 0;
    for (const auto& expert : selected.ranges) {
        while (first < registered.size() && registered[first].end <= expert.file_begin) ++first;
        std::uint64_t overlap = 0;
        const auto interior_begin = std::min(checked_add(expert.file_begin, page), expert.file_end);
        const auto interior_end = align_down(expert.file_end - 1U, page);
        for (auto i = first; i < registered.size() && registered[i].begin < expert.file_end; ++i) {
            const auto begin = std::max(expert.file_begin, registered[i].begin);
            const auto end = std::min(expert.file_end, registered[i].end);
            if (end <= begin) continue;
            if (interior_begin < interior_end && begin < interior_end && end > interior_begin)
                throw std::invalid_argument("expert source interior overlaps registered model storage");
            overlap = checked_add(overlap, end - begin);
        }
        if (overlap >= expert.file_end - expert.file_begin)
            throw std::invalid_argument("expert source is fully covered by registered boundary pages");
    }
}


std::vector<HostRange> build_copy_ranges(
    const MappedModelFile& mapped,
    const ModelManifest& manifest,
    const HipModelResidencyPolicy& policy) {
    const auto planned=build_copy_residency_ranges(manifest,policy);
    std::vector<HostRange> ranges;
    for(const auto range:planned){
        if(range.begin>mapped.bytes() || range.end>mapped.bytes())throw std::invalid_argument("manifest tensor exceeds mapped model file");
        ranges.push_back({range.begin,range.end});
    }
    return ranges;
}

}  // namespace

HipModelResidencyReservation plan_hip_model_residency_reservation(const MappedModelFile& mapped,
    const ModelManifest& manifest, HipModelResidencyPolicy policy) {
    if (!mapped.valid() || !mapped.content_identity().valid() || !mapped.allocation_generation() ||
        !manifest.layout_identity || manifest.layout_identity != model_manifest_layout_identity(manifest))
        throw std::invalid_argument("HIP residency reservation requires live mapping and valid manifest");
    if (policy.mode != HipModelResidencyMode::registered_file_mapping && policy.mode != HipModelResidencyMode::resident_copy)
        throw std::invalid_argument("HIP model residency mode is invalid");
    (void)hip_model_copy_tier_name(policy.copy_tier);
    if (policy.copy_tier == HipModelCopyTier::none && policy.copy_payload_bytes)
        throw std::invalid_argument("model copy payload requires a selective tier");
    const auto selective = policy.copy_tier == HipModelCopyTier::none ? SelectiveCopyPlan{} :
        build_selective_copy_plan(manifest, policy, mapped.bytes(), page_size());
    HipModelResidencyReservation result;
    result.copy_payload_bytes = selective.payload_bytes;
    result.copy_tensor_count = selective.tensors.size();
    result.copy_range_count = selective.ranges.size();
    result.copied_reserved_bytes = selective.arena_bytes;
    result.mapped_reserved_bytes = align_up(mapped.bytes(), page_size());
    // Tier all (STRIX_MODEL_COPY_ALL): the source pages of every copied range are paged out right after their copy
    // (MADV_PAGEOUT), so the mapping retains only the uncopied file-backed tensors (the CPU-read PLE table).
    if (policy.copy_tier == HipModelCopyTier::all) {
        std::uint64_t released = 0;
        for (const auto& range : selective.ranges)
            released = checked_add(released, align_up(range.file_end, page_size()) - range.file_begin);
        if (released >= result.mapped_reserved_bytes) throw std::invalid_argument("copy-all leaves no file-backed mapping");
        result.mapped_reserved_bytes -= released;
    }
    const auto ranges = policy.mode == HipModelResidencyMode::registered_file_mapping
        ? build_registration_ranges(mapped, manifest, policy) : build_copy_ranges(mapped, manifest, policy);
    if (policy.copy_tier == HipModelCopyTier::experts)
        validate_expert_registration(selective, ranges, page_size());
    for (const auto range : ranges) {
        if (policy.mode == HipModelResidencyMode::registered_file_mapping)
            result.registered_alias_bytes = checked_add(result.registered_alias_bytes, range.end - range.begin);
        else result.copied_reserved_bytes = checked_add(result.copied_reserved_bytes, range.end - range.begin);
    }
    // Registration aliases the source mapping; only a real copy adds backing.
    result.retained_reserved_bytes = checked_add(result.mapped_reserved_bytes, result.copied_reserved_bytes);
    result.setup_staging_reserved_bytes = expert_staging_capacity(selective, policy.copy_tier);
    // Driver-internal memory and physical-page residency remain unobserved.
    result.setup_peak_reserved_bytes = checked_add(result.retained_reserved_bytes, result.setup_staging_reserved_bytes);
    if (policy.reservation_limits.retained_bytes && result.retained_reserved_bytes > *policy.reservation_limits.retained_bytes)
        throw std::invalid_argument("HIP residency retained reservation exceeds explicit budget");
    if (policy.reservation_limits.setup_peak_bytes && result.setup_peak_reserved_bytes > *policy.reservation_limits.setup_peak_bytes)
        throw std::invalid_argument("HIP residency setup peak reservation exceeds explicit budget");
    return result;
}

HipModelResidency::HipModelResidency(
    HipModelResidency&& other) noexcept
    : mode_(other.mode_),
      registered_ranges_(
          std::move(
              other.registered_ranges_)),
      copied_ranges_(
          std::move(
              other.copied_ranges_)),
      copy_tier_(std::exchange(other.copy_tier_, HipModelCopyTier::none)),
      copy_arena_base_(std::exchange(other.copy_arena_base_, 0)),
      copy_arena_bytes_(std::exchange(other.copy_arena_bytes_, 0)),
      copy_arena_identity_(std::exchange(other.copy_arena_identity_, 0)),
      arena_ranges_(std::move(other.arena_ranges_)), selective_guard_(std::move(other.selective_guard_)),
      registered_bytes_(
          other.registered_bytes_),
      registered_locked_bytes_(std::exchange(other.registered_locked_bytes_, 0)),
      copied_bytes_(
          other.copied_bytes_),
      generation_(other.generation_),
      source_mapping_(other.source_mapping_), source_base_(other.source_base_),
      source_bytes_(other.source_bytes_), source_generation_(other.source_generation_),
      mapping_owner_(std::move(other.mapping_owner_)), reservation_(other.reservation_),
      binding_(
          std::move(
              other.binding_)) {
    other.registered_bytes_ = 0;
    other.copied_bytes_ = 0;
    other.generation_ = 0;
    other.source_mapping_ = nullptr; other.source_base_ = 0;
    other.source_bytes_ = 0; other.source_generation_ = 0; other.reservation_ = {};
    other.registered_ranges_.clear();
    other.copied_ranges_.clear();
    other.arena_ranges_.clear();
}

HipModelResidency&
HipModelResidency::operator=(
    HipModelResidency&& other) noexcept {
    if (this != &other) {
        release();
        mode_ = other.mode_;
        registered_ranges_ =
            std::move(
                other.registered_ranges_);
        copied_ranges_ =
            std::move(
                other.copied_ranges_);
        copy_tier_ = std::exchange(other.copy_tier_, HipModelCopyTier::none);
        copy_arena_base_ = std::exchange(other.copy_arena_base_, 0);
        copy_arena_bytes_ = std::exchange(other.copy_arena_bytes_, 0);
        copy_arena_identity_ = std::exchange(other.copy_arena_identity_, 0);
        arena_ranges_ = std::move(other.arena_ranges_);
        selective_guard_ = std::move(other.selective_guard_);
        registered_bytes_ =
            other.registered_bytes_;
        registered_locked_bytes_ = std::exchange(other.registered_locked_bytes_, 0);
        copied_bytes_ =
            other.copied_bytes_;
        generation_ = other.generation_;
        source_mapping_ = other.source_mapping_; source_base_ = other.source_base_;
        source_bytes_ = other.source_bytes_; source_generation_ = other.source_generation_;
        mapping_owner_ = std::move(other.mapping_owner_); reservation_ = other.reservation_;
        binding_ =
            std::move(
                other.binding_);

        other.registered_bytes_ = 0;
        other.copied_bytes_ = 0;
        other.generation_ = 0;
        other.source_mapping_ = nullptr; other.source_base_ = 0;
        other.source_bytes_ = 0; other.source_generation_ = 0; other.reservation_ = {};
        other.registered_ranges_.clear();
        other.copied_ranges_.clear();
        other.arena_ranges_.clear();
    }
    return *this;
}

HipModelResidency::~HipModelResidency() {
    release();
}

std::vector<ple_reclaim::RegisteredBacking> HipModelResidency::registered_file_backings() const {
    std::vector<ple_reclaim::RegisteredBacking> result;
    if (registered_ranges_.empty()) return result;
    if (!source_mapping_) throw ple_reclaim::DiscardSafetyError("registered model has no file backing witness");
    const auto file = source_mapping_->file_backing_identity();
    for (const auto& r : registered_ranges_) {
        if (r.host_base < source_base_ || !file[1])
            throw ple_reclaim::DiscardSafetyError("registered model file backing is unverifiable");
        result.push_back({"weights.registered-host",r.host_base,r.bytes,file[0],file[1],r.host_base-source_base_});
    }
    return result;
}

HipModelResidency HipModelResidency::create(std::shared_ptr<const MappedModelFile> mapped,
    const ModelManifest& manifest, AllocationGenerationSource& generations, HipModelResidencyPolicy policy) {
    if (!mapped) throw std::invalid_argument("HIP residency requires an actual shared mapping owner");
    // Pin before the first registration, including partial-setup cleanup.
    const auto& source = *mapped;
    return create_impl(source, std::move(mapped), manifest, generations, policy);
}

UmaOwnerInventory HipModelResidency::owner_inventory(std::shared_ptr<const HipModelResidency> actual_owner,
    std::shared_ptr<const MappedModelFile> mapped_owner, UmaMemoryObservation observation) const {
    const auto self = weak_from_this().lock();
    if (!mapped_owner) mapped_owner = mapping_owner_;
    if (!generation_ || !self || actual_owner.get() != this || actual_owner.owner_before(self) || self.owner_before(actual_owner) ||
        !mapping_owner_ || !mapped_owner ||
        mapped_owner.owner_before(mapping_owner_) || mapping_owner_.owner_before(mapped_owner) ||
        mapped_owner.get() != source_mapping_ ||
        !mapped_owner->valid() || mapped_owner->base_address() != source_base_ || mapped_owner->bytes() != source_bytes_ ||
        mapped_owner->allocation_generation() != source_generation_ ||
        mapped_owner->content_identity() != binding_.model_content_identity)
        throw std::invalid_argument("HIP residency inventory requires its unchanged actual mapping and residency owners");
    // Destruction order is explicit: unregister/free residency before unmapping.
    struct Owners {
        std::shared_ptr<const MappedModelFile> mapping;
        std::shared_ptr<const HipModelResidency> residency;
    };
    auto pin = std::make_shared<Owners>(Owners{mapped_owner, std::move(actual_owner)});
    // The allocation component is the retained mapping OBJECT lease, not its
    // CPU address. Its generation and strong pin prohibit reuse while live.
    const UmaBackingIdentity mapping_key{0x4849504d41503031ULL,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(mapped_owner.get())), source_generation_};
    std::vector<UmaOwnerView> views;
    // Tier all declares only the uncopied (file-backed) part of the mapping as retained (see the reservation plan).
    views.push_back({next_uma_owner_view_identity(), mapping_key, UmaBackingKind::file_mapping, "weights.mapping",
        reservation_.mapped_reserved_bytes, 0, copy_tier_ == HipModelCopyTier::all ?
            std::min(source_bytes_, reservation_.mapped_reserved_bytes) : source_bytes_, pin, UmaAddressSpace::host, source_base_});
    for (const auto& range : registered_ranges_) {
        if (range.host_base < source_base_ || range.host_base - source_base_ > reservation_.mapped_reserved_bytes ||
            range.bytes > reservation_.mapped_reserved_bytes - (range.host_base - source_base_) ||
            !covers_device_range(range.device_base, range.bytes))
            throw std::invalid_argument("HIP residency registered alias exceeds its actual mapping owner");
        const auto offset = static_cast<std::uint64_t>(range.host_base - source_base_);
        views.push_back({next_uma_owner_view_identity(), mapping_key, UmaBackingKind::file_mapping, "weights.registered-host",
            reservation_.mapped_reserved_bytes, offset, range.bytes, pin, UmaAddressSpace::host, range.host_base});
        views.push_back({next_uma_owner_view_identity(), mapping_key, UmaBackingKind::file_mapping, "weights.registered-device-alias",
            reservation_.mapped_reserved_bytes, offset, range.bytes, pin, UmaAddressSpace::device, range.device_base});
    }
    for (const auto& range : copied_ranges_) {
        if (!range.allocation_identity || !covers_device_range(range.device_base, range.bytes))
            throw std::invalid_argument("HIP residency copy lacks its actual owned backing lease");
        views.push_back({next_uma_owner_view_identity(), {0x484950434f505931ULL, range.allocation_identity, generation_},
            UmaBackingKind::allocated_storage, "weights.copy", range.bytes, 0, range.bytes, pin,
            UmaAddressSpace::device, range.device_base});
    }
    if (copy_arena_base_) {
        if (!copy_arena_identity_ || !selective_guard_ || copy_arena_bytes_ != reservation_.copied_reserved_bytes)
            throw std::invalid_argument("selective residency arena lacks its actual backing lease");
        selective_guard_->require_original();
        for (const auto& range : arena_ranges_) {
            const auto bytes = range.file_end - range.file_begin;
            if (range.arena_offset > copy_arena_bytes_ || bytes > copy_arena_bytes_ - range.arena_offset)
                throw std::logic_error("selective residency inventory range exceeds arena");
            views.push_back({next_uma_owner_view_identity(), {0x484950434f505931ULL, copy_arena_identity_, generation_},
                UmaBackingKind::allocated_storage, "weights.copy", copy_arena_bytes_, range.arena_offset, bytes, pin,
                UmaAddressSpace::device, copy_arena_base_ + static_cast<std::uintptr_t>(range.arena_offset)});
        }
    }
    auto result = build_uma_owner_inventory(views, std::move(observation));
    if (result.unique_backing_reserved_bytes != reservation_.retained_reserved_bytes)
        throw std::logic_error("HIP residency actual owner inventory differs from its declared reservation");
    return result;
}

bool HipModelResidency::covers_device_range(
    std::uintptr_t address, std::uint64_t bytes) const noexcept {
    constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
    if (!generation_ || !address || !bytes || bytes > maximum - address) return false;
    const auto contains = [&](std::uintptr_t base, std::uint64_t extent) noexcept {
        if (!base || !extent || extent > maximum - base || address < base) return false;
        const auto offset = static_cast<std::uint64_t>(address - base);
        return offset < extent && bytes <= extent - offset;
    };
    for (const auto& range : registered_ranges_)
        if (contains(range.device_base, range.bytes)) return true;
    // Chunked registration: a range may continue through consecutive registrations with contiguous device aliases.
    for (std::size_t i = 0; i < registered_ranges_.size(); ++i) {
        const auto& first = registered_ranges_[i];
        if (!first.device_base || address < first.device_base || address - first.device_base >= first.bytes) continue;
        auto covered = first.bytes - static_cast<std::uint64_t>(address - first.device_base);
        for (auto j = i + 1; covered < bytes && j < registered_ranges_.size(); ++j) {
            const auto& prev = registered_ranges_[j - 1];
            if (registered_ranges_[j].device_base != prev.device_base + prev.bytes) break;
            covered += registered_ranges_[j].bytes;
        }
        return covered >= bytes;
    }
    for (const auto& range : copied_ranges_)
        if (contains(range.device_base, range.bytes)) return true;
    if (contains(copy_arena_base_, copy_arena_bytes_)) return true;
    return false;
}

HipModelResidency HipModelResidency::create(
    const MappedModelFile& mapped,
    const ModelManifest& manifest,
    AllocationGenerationSource& generations,
    HipModelResidencyPolicy policy) {
    return create_impl(mapped, {}, manifest, generations, policy);
}

HipModelResidency HipModelResidency::create_impl(const MappedModelFile& mapped,
    std::shared_ptr<const MappedModelFile> mapping_owner, const ModelManifest& manifest,
    AllocationGenerationSource& generations, HipModelResidencyPolicy policy) {
    if (!mapped.valid() ||
        !mapped.content_identity().valid()) {
        throw std::invalid_argument(
            "HIP model residency requires a valid mapped model");
    }
    if (manifest.layout_identity == 0 ||
        manifest.layout_identity !=
            model_manifest_layout_identity(
                manifest)) {
        throw std::invalid_argument(
            "HIP model residency requires a valid manifest");
    }
    if (mapped.bytes() >
        static_cast<std::uint64_t>(
            std::numeric_limits<
                std::size_t>::max())) {
        throw std::overflow_error(
            "mapped model size does not fit size_t");
    }

    const auto reservation = plan_hip_model_residency_reservation(mapped, manifest, policy);
    const bool selective_enabled = policy.copy_tier != HipModelCopyTier::none;
    SelectiveCopyPlan selective;
    std::shared_ptr<const HipSetupQuiescenceGuard> selective_guard;
    if (selective_enabled) {
        if (!mapping_owner) throw std::invalid_argument("selective residency requires the actual strong mapping owner");
        selective = build_selective_copy_plan(manifest, policy, mapped.bytes(), page_size());
        if (current_hip_launch_context())
            throw std::invalid_argument("selective residency setup requires no active launch scope");
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        check_hip(hipStreamIsCapturing(nullptr, &capture), "selective residency capture admission");
        if (capture != hipStreamCaptureStatusNone)
            throw std::invalid_argument("selective residency cannot be created during capture");
        selective_guard = std::make_shared<const HipSetupQuiescenceGuard>();
    }
    HipModelResidency result;
    result.copy_tier_ = policy.copy_tier;
    result.selective_guard_ = std::move(selective_guard);
    result.mapping_owner_ = std::move(mapping_owner);
    result.reservation_ = reservation;
    result.source_mapping_ = &mapped; result.source_base_ = mapped.base_address();
    result.source_bytes_ = mapped.bytes(); result.source_generation_ = mapped.allocation_generation();
    result.mode_ =
        policy.mode;
    const auto generation =
        generations.next();
    result.generation_ = generation;

    result.binding_ =
        mapped.build_runtime_binding(
            manifest);

    if (policy.mode ==
        HipModelResidencyMode::
            registered_file_mapping) {
        const auto ranges =
            build_registration_ranges(
                mapped,
                manifest,
                policy);
        if (!ranges.empty() && !result.selective_guard_)
            result.selective_guard_ = std::make_shared<const HipSetupQuiescenceGuard>();

#if defined(__linux__) && defined(MLOCK_ONFAULT)
        // Registered pages are kept off the reclaim LRU by default. A reclaimed or migrated page invalidates its
        // userptr registration and the driver evicts every queue of the process until the whole registration is
        // revalidated; under startup memory pressure that restore can stall indefinitely. Needs RLIMIT_MEMLOCK >=
        // the registered bytes. Automatic failures warn as before; explicit =1 must succeed.
        // Off: STRIX_MODEL_REGISTER_MLOCK=0.
        RegisteredPageLockPolicy lock(std::getenv("STRIX_MODEL_REGISTER_MLOCK"));
        if (lock.enabled) {
            std::uint64_t planned = 0;
            for (const auto range : ranges) planned = checked_add(planned, range.end - range.begin);
            rlimit limit{};
            if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0 ||
                (limit.rlim_cur != RLIM_INFINITY && static_cast<std::uint64_t>(limit.rlim_cur) < planned)) {
                lock.failed("model residency: requested registered-page lock exceeds RLIMIT_MEMLOCK; refusing unlocked userptr backing", [&] {
                    std::fprintf(stderr, "model residency: RLIMIT_MEMLOCK below %llu registered bytes; registered pages stay "
                        "unlocked (reclaim can evict GPU queues)\n", static_cast<unsigned long long>(planned));
                });
            }
        }
#endif
        try {
            result.registered_ranges_.
                reserve(
                    ranges.size());

            for (const auto range :
                 ranges) {
                const auto bytes =
                    range.end -
                    range.begin;
                const auto host =
                    mapped.base_address() +
                    static_cast<
                        std::uintptr_t>(
                            range.begin);

                startup::Phase registration_marker("registration_chunk", range.begin);
                acquire_registered_backing([&] {
                    check_hip(selective_enabled ? selective_register(reinterpret_cast<void*>(host), static_cast<std::size_t>(bytes)) :
                        hipHostRegister(reinterpret_cast<void*>(host), static_cast<std::size_t>(bytes), model_register_flags()),
                        "hipHostRegister(model range)");
                }, [&] {
                    // Track before querying the alias so failure retains the backing.
                    result.registered_ranges_.push_back(RegisteredRange{host,0,bytes});
                }, [&] {
                    void* device = nullptr;
                    check_hip(selective_enabled ? selective_alias(&device, reinterpret_cast<void*>(host)) :
                        hipHostGetDevicePointer(&device, reinterpret_cast<void*>(host), 0),
                        "hipHostGetDevicePointer(model range)");
                    result.registered_ranges_.back().device_base = reinterpret_cast<std::uintptr_t>(device);
                });
                result.registered_bytes_ =
                    checked_add(
                        result.registered_bytes_,
                        bytes);
#if defined(__linux__) && defined(MLOCK_ONFAULT)
                // With vm.compact_unevictable_allowed=0 compaction does not migrate locked pages either.
                if (lock.enabled) {
                    if (mlock2(reinterpret_cast<const void*>(host), static_cast<std::size_t>(bytes), MLOCK_ONFAULT) == 0) {
                        result.registered_ranges_.back().locked = true;
                        result.registered_locked_bytes_ = checked_add(result.registered_locked_bytes_, bytes);
                    } else {
                        const int lock_errno = errno;
                        lock.failed(std::string("model residency: requested registered-page lock failed; refusing unlocked userptr backing: ") + std::strerror(lock_errno), [&] {
                            std::fprintf(stderr, "model residency: mlock of %llu registered bytes failed (%s); continuing unlocked\n",
                                static_cast<unsigned long long>(bytes), std::strerror(lock_errno));
                        });
                    }
                }
#endif
            }

            for (std::size_t index = 0;
                 index <
                    manifest.tensors.size();
                 ++index) {
                const auto& tensor =
                    manifest.tensors[index];
                auto& runtime =
                    result.binding_.
                        tensors[index];

                if (!should_register(
                        tensor,
                        policy)) {
                    continue;
                }

                const auto host =
                    mapped.base_address() +
                    static_cast<
                        std::uintptr_t>(
                            tensor.absolute_offset);
                const auto found =
                    std::find_if(
                        result.registered_ranges_.begin(),
                        result.registered_ranges_.end(),
                        [&](const RegisteredRange& range) {
                            if (host < range.host_base) {
                                return false;
                            }
                            const auto offset =
                                static_cast<std::uint64_t>(
                                    host - range.host_base);
                            return offset < range.bytes || (offset == range.bytes && tensor.byte_size == 0U);
                        });
                if (found ==
                    result.registered_ranges_.end()) {
                    throw std::logic_error(
                        "promoted tensor is not covered by a registered range");
                }
                // A tensor that continues past its first range (chunked registration only) must lie in
                // consecutive ranges whose host and device aliases are both contiguous.
                {
                    auto covered = found->bytes - static_cast<std::uint64_t>(host - found->host_base);
                    for (auto next = found + 1; covered < tensor.byte_size; ++next) {
                        const auto prev = next - 1;
                        if (next == result.registered_ranges_.end() || next->host_base != prev->host_base + prev->bytes ||
                            next->device_base != prev->device_base + prev->bytes)
                            throw std::logic_error(
                                "promoted tensor spans registered ranges without contiguous device aliases");
                        covered += next->bytes;
                    }
                }

                const auto offset =
                    static_cast<std::uintptr_t>(
                        host -
                        found->host_base);
                runtime.storage =
                    RuntimeTensorStorage::
                        registered_file_mapped;
                runtime.address =
                    found->device_base +
                    offset;
                runtime.allocation_generation =
                    generation;
            }
            if (selective_enabled) {
                const auto page = page_size();
                for (const auto& range : result.registered_ranges_)
                    if (!range.device_base || range.device_base % page || range.host_base % page ||
                        range.bytes > UINTPTR_MAX - range.device_base)
                        throw std::invalid_argument("selective residency requires page-congruent registered aliases");
                // Tier all: release the page cache of every range about to be copied BEFORE the arena allocation,
                // so the allocation finds free memory instead of reclaiming/compacting ~82 GB of cache under the
                // process's pinned host buffers (measured: 23 M pages scanned and 1.4 M migrated otherwise).
                if (policy.copy_tier == HipModelCopyTier::all)
                    for (const auto& range : selective.ranges) mapped.advise_dontneed(range.file_begin, range.file_end - range.file_begin);
                result.arena_ranges_ = std::move(selective.ranges);
                result.copy_arena_identity_ = residency_identity();
                void* arena = nullptr;
                result.selective_guard_->run([&] {
                    check_hip(selective_allocate(&arena, static_cast<std::size_t>(selective.arena_bytes)),
                        "selective residency arena allocation");
                    result.copy_arena_base_ = reinterpret_cast<std::uintptr_t>(arena);
                    result.copy_arena_bytes_ = selective.arena_bytes;
                });
                bool aligned = result.copy_arena_base_ != 0 && result.copy_arena_base_ % page == 0;
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
                if (inject(model_residency_diagnostic::Fault::arena_alignment, 1)) aligned = false;
#endif
                if (!aligned || result.copy_arena_bytes_ > UINTPTR_MAX - result.copy_arena_base_)
                    throw std::invalid_argument("selective residency requires page-congruent bounded arena");
                ExpertCopyStaging staging(*result.selective_guard_);
                staging.allocate(reservation.setup_staging_reserved_bytes);
                auto source_registration = ranges.begin();
                for (const auto& range : result.arena_ranges_) {
                    auto source_begin = range.file_begin;
                    while (source_begin < range.file_end) {
                        auto source_end = range.file_end;
                        // Keep each transfer within one source registration or
                        // one unregistered span, including shared boundary pages.
                        while (source_registration != ranges.end() && source_registration->end <= source_begin)
                            ++source_registration;
                        if (source_registration != ranges.end())
                            source_end = std::min(source_end,
                                source_registration->begin <= source_begin ?
                                    source_registration->end : source_registration->begin);
                        if (source_end <= source_begin)
                            throw std::logic_error("selective model copy source segment is empty");
                        while (source_begin < source_end) {
                            const auto bytes = reservation.setup_staging_reserved_bytes
                                ? std::min(source_end - source_begin, reservation.setup_staging_reserved_bytes)
                                : source_end - source_begin;
                            const auto* source = reinterpret_cast<const void*>(mapped.base_address() + source_begin);
                            if (staging.data()) {
                                std::memcpy(staging.data(), source, static_cast<std::size_t>(bytes));
                                source = staging.data();
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
                                ++model_residency_diagnostic::counters.staging_copy_chunks;
                                model_residency_diagnostic::counters.maximum_staging_copy_bytes =
                                    std::max(model_residency_diagnostic::counters.maximum_staging_copy_bytes, bytes);
#endif
                            }
                            result.selective_guard_->run([&] {
                                check_hip(selective_copy(reinterpret_cast<void*>(result.copy_arena_base_ +
                                        range.arena_offset + (source_begin - range.file_begin)), source,
                                    static_cast<std::size_t>(bytes)), "selective residency weight copy");
                            }); // Proves the source lease can be reused for the next chunk.
#if defined(__linux__)
                            if (policy.copy_tier == HipModelCopyTier::all) {
#if defined(MADV_PAGEOUT)
                                constexpr int pageout = MADV_PAGEOUT;
#else
                                constexpr int pageout = 21;
#endif
                                // Copied: drop these source pages (clean file pages; a later host read refaults them).
                                const auto first = align_down(source_begin, page_size());
                                const auto last = std::min(align_up(source_begin + bytes, page_size()), align_up(mapped.bytes(), page_size()));
                                (void)::madvise(reinterpret_cast<void*>(mapped.base_address() + first), static_cast<std::size_t>(last - first), pageout);
                            }
#endif
                            source_begin += bytes;
                        }
                    }
                }
                staging.retire(); // No temporary pinned backing escapes into retained ownership.
                for (const auto index : selective.tensors) {
                    const auto& tensor = manifest.tensors[index];
                    const auto range = std::find_if(result.arena_ranges_.begin(), result.arena_ranges_.end(),
                        [&](const auto& r) { return tensor.absolute_offset >= r.file_begin && tensor.absolute_offset <= r.file_end &&
                            tensor.byte_size <= r.file_end - tensor.absolute_offset; });
                    if (range == result.arena_ranges_.end()) throw std::logic_error("selective tensor lacks copied arena range");
                    auto& runtime = result.binding_.tensors[index];
                    runtime.storage = RuntimeTensorStorage::resident;
                    runtime.address = result.copy_arena_base_ + range->arena_offset + (tensor.absolute_offset - range->file_begin);
                    runtime.allocation_generation = generation;
                    if (runtime.address % page != tensor.absolute_offset % page || !result.covers_device_range(runtime.address, runtime.bytes))
                        throw std::logic_error("selective tensor address congruence differs from source");
                }
                result.copied_bytes_ = selective.arena_bytes;
                if (policy.copy_tier == HipModelCopyTier::experts || policy.copy_tier == HipModelCopyTier::all) {
                    std::size_t selected_count = 0;
                    for (std::size_t index = 0; index < manifest.tensors.size(); ++index) {
                        const auto& tensor = manifest.tensors[index];
                        if (!should_promote(tensor, policy)) continue;
                        const auto& runtime = result.binding_.tensors[index];
                        const bool expert = selective_copy_eligible(tensor, policy.copy_tier);
                        if (runtime.storage != (expert ? RuntimeTensorStorage::resident :
                                RuntimeTensorStorage::registered_file_mapped) ||
                            runtime.allocation_generation != generation ||
                            !result.covers_device_range(runtime.address, runtime.bytes))
                            throw std::logic_error("expert model copy has incomplete tensor backing");
                        if (expert) ++selected_count;
                    }
                    if (selected_count != selective.tensors.size())
                        throw std::logic_error("expert model copy binding census differs from plan");
                }
                for (std::size_t index = 0; index < manifest.tensors.size(); ++index)
                    if (manifest.tensors[index].placement.kind == PlacementKind::local &&
                        result.binding_.tensors[index].address % page != manifest.tensors[index].absolute_offset % page)
                        throw std::logic_error("selective residency changed a local tensor page-relative alignment");
                result.selective_guard_->require_original();
            }
        } catch (...) {
            result.release();
            throw;
        }
    } else {
        const auto ranges =
            build_copy_ranges(
                mapped,
                manifest,
                policy);

        const HipSetupQuiescenceGuard setup_guard;

        try {
            result.copied_ranges_.reserve(
                ranges.size());

            for (const auto range : ranges) {
                const auto bytes =
                    range.end - range.begin;
                const auto allocation_identity = residency_identity();
                void* device = nullptr;
                check_hip(
                    hipMalloc(
                        &device,
                        static_cast<std::size_t>(
                            bytes)),
                    "hipMalloc(model range copy)");
                try {
                    setup_guard.run([&] {
                        check_hip(
                            hipMemcpy(
                                device,
                                reinterpret_cast<const void*>(
                                    mapped.base_address() +
                                    static_cast<std::uintptr_t>(range.begin)),
                                static_cast<std::size_t>(bytes),
                                hipMemcpyHostToDevice),
                            "hipMemcpy(model range copy)");
                    });
                } catch (...) {
                    static_cast<void>(
                        hipFree(device));
                    throw;
                }

                result.copied_ranges_.push_back(
                    CopiedRange{
                        range.begin,
                        reinterpret_cast<std::uintptr_t>(
                            device),
                        bytes,
                        allocation_identity,
                    });
                result.copied_bytes_ =
                    checked_add(
                        result.copied_bytes_,
                        bytes);
            }

            for (std::size_t index = 0;
                 index < manifest.tensors.size();
                 ++index) {
                const auto& tensor =
                    manifest.tensors[index];
                auto& runtime =
                    result.binding_.tensors[index];

                if (!should_promote(
                        tensor,
                        policy)) {
                    continue;
                }

                const auto found =
                    std::find_if(
                        result.copied_ranges_.begin(),
                        result.copied_ranges_.end(),
                        [&](const CopiedRange& range) {
                            if (tensor.absolute_offset <
                                range.file_offset) {
                                return false;
                            }
                            const auto offset =
                                tensor.absolute_offset -
                                range.file_offset;
                            return offset <= range.bytes &&
                                   tensor.byte_size <=
                                       range.bytes - offset;
                        });
                if (found ==
                    result.copied_ranges_.end()) {
                    throw std::logic_error(
                        "promoted tensor is not covered by a copied range");
                }

                runtime.storage =
                    RuntimeTensorStorage::resident;
                runtime.address =
                    found->device_base +
                    static_cast<std::uintptr_t>(
                        tensor.absolute_offset -
                        found->file_offset);
                runtime.allocation_generation =
                    generation;
            }
        } catch (...) {
            result.release();
            throw;
        }
    }

    validate_model_runtime_binding(
        manifest,
        result.binding_);
    return result;
}

void HipModelResidency::release() noexcept {
    const auto drain = [&] {
        if (selective_guard_) {
            try { selective_guard_->run([] {}); }
            catch (...) { selective_guard_->recover_or_fail_stop(); }
        } else if (!registered_ranges_.empty()) std::terminate();
    };
    if (registered_ranges_.empty()) drain();
    bool unregister_failed = false;
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
    std::vector<std::uintptr_t> failed_registrations;
#endif
    const bool retired = retire_registered_backings(registered_ranges_, drain, [&](const auto& range) {
        const auto error = copy_tier_ != HipModelCopyTier::none ?
            selective_unregister(reinterpret_cast<void*>(range.host_base)) : hipHostUnregister(reinterpret_cast<void*>(range.host_base));
        if (error != hipSuccess) {
            unregister_failed = true;
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
            if (copy_tier_ != HipModelCopyTier::none) failed_registrations.push_back(range.host_base);
#endif
        }
        return error == hipSuccess;
    }, [](const auto& range) {
#if defined(__linux__) && defined(MLOCK_ONFAULT)
        if (range.locked) (void)munlock(reinterpret_cast<const void*>(range.host_base), static_cast<std::size_t>(range.bytes));
#else
        (void)range;
#endif
    });
    if (!retired) unregister_failed = true;
    if (unregister_failed && !mapping_owner_) {
        std::fputs("model residency: unregister failed on borrowed backing; terminating without unmapping\n",stderr);
        std::_Exit(3);
    }
    if (unregister_failed && mapping_owner_) {
        // Quarantine only this failed strong lease. Do not unmap storage while
        // the runtime may still retain a registration. Allocation failure is
        // fail-stop rather than releasing a possibly live backing.
#if defined(STRIX_BUILD_MODEL_RESIDENCY_DIAGNOSTIC)
        if (copy_tier_ != HipModelCopyTier::none) {
            ++model_residency_diagnostic::counters.quarantined_mappings;
            auto& retained = *model_residency_diagnostic::quarantined;
            retained.reserve(retained.size() + 1U); // Allocate before moving the last mapping pin.
            retained.push_back({std::move(mapping_owner_), std::move(failed_registrations)});
        } else
#endif
        {
            auto* retained = new (std::nothrow) std::shared_ptr<const MappedModelFile>(std::move(mapping_owner_));
            if (!retained) std::terminate();
        }
    }
    registered_ranges_.clear();
    registered_locked_bytes_ = 0;

    for (const auto& range :
         copied_ranges_) {
        static_cast<void>(
            hipFree(
                reinterpret_cast<void*>(
                    range.device_base)));
    }
    copied_ranges_.clear();
    if (copy_arena_base_) selective_free(reinterpret_cast<void*>(copy_arena_base_));
    copy_arena_base_ = 0; copy_arena_bytes_ = 0; copy_arena_identity_ = 0;
    arena_ranges_.clear(); selective_guard_.reset(); copy_tier_ = HipModelCopyTier::none;
    registered_bytes_ = 0;
    copied_bytes_ = 0;
    generation_ = 0;
    binding_ = {};
    source_mapping_ = nullptr; source_base_ = 0; source_bytes_ = 0; source_generation_ = 0;
    reservation_ = {};
    mapping_owner_.reset(); // Only after every unregister/free attempt above.
}

}  // namespace strix
