#pragma once
#include "strix/qwen_target_state_digest.hpp"
#include "qwen_cache_admission.hpp"
#include "prefix_state_cache_identity.hpp"
#include "qwen_tiled_prompt_plan.hpp"
#include "qwen_serial_prompt_certificate.hpp"
// Block-boundary prefix state cache on the tiled known-prefix prefill: session-state capture and
// restore over strix::PrefixStateCache (docs/engine/serving/prefix-state-cache.md).
#include "strix/arithmetic_lineage.hpp"
#include "strix/memory_plan.hpp"
#include "strix/model_contract.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/prefix_state_cache.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include "strix/session_state_cache.hpp"
#include "strix/qpf2k_defaults.hpp"
#include "strix/moe_route_bf16.hpp"
#include "strix/rtx_state_import.hpp"
#include "strix/serial_session.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::decode_cli {

class PrefixCaptureHipError : public PrefixCacheFatalError {
public:
    hipError_t error;
    PrefixCaptureHipError(hipError_t code, const char* what)
        : PrefixCacheFatalError(std::string(what) + ": " + hipGetErrorString(code)), error(code) {}
};

inline PrefixCacheBuffer hip_pinned_prefix_buffer(std::uint64_t bytes) {
    void* memory = nullptr;
    if (bytes && hipHostMalloc(&memory, static_cast<std::size_t>(bytes), hipHostMallocDefault) != hipSuccess)
        throw std::runtime_error("prefix state cache pinned host allocation failed");
    return {std::shared_ptr<std::byte>(static_cast<std::byte*>(memory), [](std::byte* p) { if (p) (void)hipHostFree(p); }), bytes};
}
// Opportunistic prefix capture alone uses the typed allocation failure seam.
inline PrefixCacheBuffer hip_pinned_prefix_capture_buffer(std::uint64_t bytes) {
    void* memory = nullptr;
    if (bytes) {
        const auto error = hipHostMalloc(&memory, static_cast<std::size_t>(bytes), hipHostMallocDefault);
        if (error == hipErrorOutOfMemory) throw PrefixCacheAllocationFailure();
        if (error != hipSuccess) throw PrefixCaptureHipError(error, "prefix state cache pinned host allocation failed");
    }
    return {std::shared_ptr<std::byte>(static_cast<std::byte*>(memory), [](std::byte* p) { if (p) (void)hipHostFree(p); }), bytes};
}

struct PrefixStateCacheRequestStats {
    std::string refusal;
    std::map<std::string,std::uint64_t> certificate_declines;
    std::uint64_t prompt_tokens = 0, restored_position = 0, captured_checkpoints = 0, captured_segment_bytes = 0;
    std::uint64_t captured_fixed_bytes = 0, restored_bytes = 0;
    double lookup_ms = 0.0, restore_ms = 0.0, capture_ms = 0.0;
    // --rtx-state-import: this request started from an imported state (docs/rtx-state-import.md).
    bool rtx_import = false, rtx_sha_ok = false;
    std::uint64_t rtx_position = 0, rtx_bytes = 0, rtx_chunks = 0, rtx_local_rows = 0, rtx_local_bytes = 0;
    double rtx_restore_ms = 0.0, rtx_verify_ms = 0.0;
    std::string rtx_id, rtx_manifest_sha256;
};

// STRIX_RTX_EXPORT_STATE=<file>: write the state of the tiled MTP prompt at position STRIX_RTX_EXPORT_POSITION
// (default: the end of the prompt) in the strix-qwen-rtx-state-v1 file form (a producer for testing the import
// without an RTX service). Opt-in; unset, nothing changes.
struct RtxStateExportRequest {
    std::string path;
    std::uint64_t position = 0;  // 0: the end of every prompt
};
inline const std::optional<RtxStateExportRequest>& rtx_state_export_request() {
    static const std::optional<RtxStateExportRequest> request = []() -> std::optional<RtxStateExportRequest> {
        const char* path = std::getenv("STRIX_RTX_EXPORT_STATE");
        if (!path || !path[0]) return std::nullopt;
        RtxStateExportRequest r;
        r.path = path;
        if (const char* p = std::getenv("STRIX_RTX_EXPORT_POSITION"); p && p[0]) r.position = rtx_parse_u64(p);
        return r;
    }();
    return request;
}

// STRIX_PROMPT_STATE_DIGEST=1 (diagnostic, gates): every tiled MTP prompt gets the state component registry even without a
// prefix cache (no capture, no lookup), so its receipt carries the end-of-prompt state digest (target state, predictor
// rows and held row) on every route; STRIX_PREFIX_CACHE_DIGEST=1 prints it.
inline bool prompt_state_digest_request(bool batch_cert=false) {
    static const bool on = [] { const char* v = std::getenv("STRIX_PROMPT_STATE_DIGEST"); return v && v[0] == '1'; }();
    const char* digest=std::getenv("STRIX_PREFIX_CACHE_DIGEST");
    return on || (batch_cert && digest && digest[0]=='1');
}

// One `qwen-target-state` line. Served requests use it so a failed diagnostic is logged on the line
// instead of failing the request (and with it the server).
template <class Digest>
void print_target_state_digest(std::FILE* out, const char* phase, std::uint64_t position, Digest&& digest) {
    try {
        const auto hex = model_content_identity_hex(digest());
        std::fprintf(out, "qwen-target-state phase=%s position=%llu sha256=%s\n", phase,
            static_cast<unsigned long long>(position), hex.c_str());
    } catch (const std::exception& error) {
        std::fprintf(out, "qwen-target-state phase=%s position=%llu error=%s\n", phase,
            static_cast<unsigned long long>(position), error.what());
    }
}

// What the consumer knows about the state an import must reproduce (computed from the compiled serial plan).
struct RtxStateImportFacts {
    std::array<std::uint64_t, 6> generations_per_row{};  // steps of the serial plan writing flag f (bit f)
    PersistentState valid_union = PersistentState::none;  // union of every step's writes
    ModelContentIdentity model_content;
};

// A failed drain cannot prove DMA quiescence. Preallocate ownership before
// submitting copies, then link it into an intentionally never-destroyed owner.
struct PrefixDmaQuarantine {
    std::shared_ptr<PrefixCacheCheckpoint> checkpoint;
    std::vector<std::shared_ptr<PrefixCacheSegment>> pending;
    std::shared_ptr<const RtxStateStaging> imported_source;
    std::shared_ptr<void> snapshot_owner;
    std::uint64_t staging_bytes = 0;
    PrefixDmaQuarantine* next = nullptr;
    inline static std::atomic<PrefixDmaQuarantine*> head{nullptr};
    inline static std::atomic<std::uint64_t> count{0}, bytes{0};
    static void retain(std::unique_ptr<PrefixDmaQuarantine>& owner) noexcept {
        if (!owner) return;
        std::uint64_t payload = owner->staging_bytes;
        if (owner->checkpoint) {
            payload += owner->checkpoint->fixed.data ? owner->checkpoint->fixed.bytes : 0;
            for (const auto& segment : owner->checkpoint->segments)
                if (segment->payload.data) payload += segment->payload.bytes;
        }
        if (owner->imported_source)
            payload += owner->imported_source->layout().payload_bytes(owner->imported_source->position());
        auto* node = owner.release();
        node->next = head.load();
        while (!head.compare_exchange_weak(node->next, node)) {}
        const auto n = count.fetch_add(1) + 1;
        const auto b = bytes.fetch_add(payload) + payload;
        std::fprintf(stderr, "prefix DMA quarantine: count=%llu bytes=%llu; fail-stop required\n",
            static_cast<unsigned long long>(n), static_cast<unsigned long long>(b));
    }
};

// Positive runtime evidence for the next writers. Unknown ownership refuses capture.
// Predictor sources must remain owned by this stream through snapshot submission;
// any later host/default-stream writer must follow the incumbent priming completion.
struct PrefixCaptureOrder {
    std::optional<hipStream_t> next_target, predictor, next_predictor;
};

// Attach-time HIP resources; a descriptor retains this owner through DMA/quarantine.
struct PrefixCaptureResources {
    void* staging=nullptr;
    std::uint64_t cap=0;
    bool poisoned=false;
    hipStream_t download=nullptr;
    hipEvent_t target_ready=nullptr,mtp_ready=nullptr,done=nullptr;
    ~PrefixCaptureResources() {
        for(auto event:{done,mtp_ready,target_ready}) if(event) (void)hipEventDestroy(event);
        if(download) (void)hipStreamDestroy(download);
        if(staging) (void)hipFree(staging);
    }
};

// One process-wide, nonblocking snapshot slot. Worker submission owns no session pointer.
// Only service-thread methods touch the slot; submitted is the worker release/acquire handoff.
struct PrefixAsyncCapture : std::enable_shared_from_this<PrefixAsyncCapture> {
    struct Slice { const void* source; std::byte* destination; std::uint64_t offset, bytes; bool predictor; };
    std::shared_ptr<PrefixStateCache> store;
    std::shared_ptr<const void> target_owner, predictor_owner;
    std::shared_ptr<PrefixCacheCheckpoint> checkpoint;
    PrefixStateCache::Reservation reservation;
    std::vector<Slice> slices;
    std::shared_ptr<PrefixCaptureResources> resources;
    void* staging = nullptr;
    std::uint64_t bytes = 0, row_bytes = 0;
    std::weak_ptr<PrefixStateCacheRequestStats> request;
    std::chrono::steady_clock::time_point prepared=std::chrono::steady_clock::now();
    hipStream_t download = nullptr;
    hipEvent_t target_ready = nullptr, mtp_ready = nullptr, done = nullptr;
    std::atomic<bool> submitted{false};
    bool committed = false, cancelled = false, started = false, publish = true;
    bool poisoned=false, split_target=false, metadata_complete=true;
    const void* owner_tag=nullptr;
    hipStream_t target_stream=nullptr;
    // Serialized runner thread only. Query is deliberately nonblocking; no producer wait.
    static void before_writer(const void* tag,hipStream_t stream) {
        auto d=slot;
        if(!d || !tag || d->owner_tag!=tag || d->poisoned || stream==d->target_stream) return;
        const auto status=hipEventQuery(d->target_ready);
        if(status==hipSuccess) return;
        if(status!=hipErrorNotReady) {d->fatal();check(status,"prefix writer tripwire query");}
        d->poisoned=true;d->store->record_tripwire_skip();
    }
    using RowChain=std::vector<std::shared_ptr<const PrefixCacheSegment>>;
    std::weak_ptr<RowChain> row_chain;
    std::unique_ptr<PrefixDmaQuarantine> quarantine = std::make_unique<PrefixDmaQuarantine>();
    inline static std::shared_ptr<PrefixAsyncCapture> slot;
    static void check(hipError_t e, const char* what) {
        if(e!=hipSuccess) throw PrefixCaptureHipError(e,what);
    }
    ~PrefixAsyncCapture() {
        if(resources) return;
        if(done) (void)hipEventDestroy(done);
        if(mtp_ready) (void)hipEventDestroy(mtp_ready);
        if(target_ready) (void)hipEventDestroy(target_ready);
        if(download) (void)hipStreamDestroy(download);
        if(staging) (void)hipFree(staging);
    }
    // Checked retirement runs while a shared quarantine owner can still retain backing.
    void retire() {
        if(resources) return;
        try {
            for(auto* event:{&done,&mtp_ready,&target_ready}) if(*event) {
                check(hipEventDestroy(*event),"prefix event retirement"); *event=nullptr;
            }
            if(download) {check(hipStreamDestroy(download),"prefix stream retirement");download=nullptr;}
            if(staging) {check(hipFree(staging),"prefix staging retirement");staging=nullptr;}
        } catch(...) {fatal();throw;}
    }
    void fatal() {
        if(resources) resources->poisoned=true;
        if(!quarantine) return;
        quarantine->checkpoint=checkpoint;
        quarantine->snapshot_owner=shared_from_this();
        quarantine->staging_bytes=bytes;
        PrefixDmaQuarantine::retain(quarantine);
        store->record_capture_failure();
    }
    void snapshot(bool predictor, hipStream_t stream) {
        for(const auto& s:slices) if(s.predictor==predictor)
            check(hipMemcpyAsync(static_cast<std::byte*>(staging)+s.offset,s.source,
                static_cast<std::size_t>(s.bytes),hipMemcpyDeviceToDevice,stream),"prefix snapshot D2D");
        check(hipEventRecord(predictor?mtp_ready:target_ready,stream),"prefix snapshot ready");
    }
    // The split target download has no predictor dependency. The same download stream
    // later joins the predictor group, so done covers both groups and their leases.
    void submit_target_download() {
        check(hipStreamWaitEvent(download,target_ready,0),"prefix target dependency");
        for(const auto& s:slices) if(!s.predictor)
            check(hipMemcpyAsync(s.destination,static_cast<std::byte*>(staging)+s.offset,
                static_cast<std::size_t>(s.bytes),hipMemcpyDeviceToHost,download),"prefix target D2H");
    }
    void cancel_target() {
        if(!split_target || submitted.load(std::memory_order_acquire)) {cancelled=true;return;}
        cancelled=true;
        try {
            check(hipEventRecord(done,download),"prefix cancelled target completion");
            submitted.store(true,std::memory_order_release);
        } catch(...) {fatal();throw;}
    }
    void submit_predictor(hipStream_t stream) {
        try {
            if(split_target && !metadata_complete) throw std::logic_error("prefix split metadata incomplete");
            snapshot(true,stream);
            if(!split_target) check(hipStreamWaitEvent(download,target_ready,0),"prefix target dependency");
            check(hipStreamWaitEvent(download,mtp_ready,0),"prefix predictor dependency");
            for(const auto& s:slices) if(!split_target || s.predictor)
                check(hipMemcpyAsync(s.destination,static_cast<std::byte*>(staging)+s.offset,
                    static_cast<std::size_t>(s.bytes),hipMemcpyDeviceToHost,download),"prefix snapshot D2H");
            check(hipEventRecord(done,download),"prefix snapshot completion event");
            submitted.store(true,std::memory_order_release);
        } catch(...) {fatal();throw;}
    }
    static void reap(bool shutdown=false) {
        auto d=slot;
        if(!d || !d->submitted.load(std::memory_order_acquire)) return;
        const auto e=shutdown?hipEventSynchronize(d->done):hipEventQuery(d->done);
        if(e==hipErrorNotReady && !shutdown) return;
        if(e!=hipSuccess) {d->fatal();slot.reset();check(e,"prefix snapshot completion");}
        if(!d->committed && !d->cancelled && !shutdown) return;
        if(d->committed && !d->cancelled && !d->poisoned) {
            if(auto chain=d->row_chain.lock()) *chain=d->checkpoint->segments;
            bool inserted=false;
            if(d->publish) inserted=d->store->insert(d->checkpoint,&d->reservation);
            else d->store->record_row_prefetch();
            if(auto stats=d->request.lock()) {
                stats->captured_segment_bytes+=d->row_bytes;
                stats->captured_fixed_bytes+=d->checkpoint->fixed.bytes;
                if(inserted) ++stats->captured_checkpoints;
                stats->capture_ms+=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-d->prepared).count();
            }
        }
        d->retire();
        d->store->record_staging(0,d->bytes,false);
        slot.reset();
    }
    void commit(bool valid) noexcept {
        committed=valid && metadata_complete;
        cancelled=cancelled || !committed || poisoned;
    }
    // Called explicitly after serving returns, before runtime owners disappear.
    static void shutdown() {reap(true);}
};

// Shared by legacy setup and post-residency attachment; byte framing is stable.
template<class Cache>
inline void bind_mtp_prefix_cache(Cache& cache, std::uintptr_t kv, std::uint64_t kv_row,
    std::uintptr_t index, std::uint64_t index_row, std::uintptr_t held, std::uint64_t held_bytes,
    const ModelContentIdentity& sidecar, std::uint64_t plan, std::uint64_t profile,
    bool draft_chain, bool draft_exact, bool draft_q4, std::function<void(std::uint64_t)> restore) {
    cache.add_rows(kv,kv_row,1U,"mtp.kv");
    cache.add_rows(index,index_row,1U,"mtp.index");
    cache.add_fixed_device(held,held_bytes,"mtp.held");
    std::vector<std::byte> frame;
    const auto u64=[&](std::uint64_t v){for(unsigned b=0;b<64;b+=8)frame.push_back(static_cast<std::byte>((v>>b)&255U));};
    for(const auto byte:sidecar.bytes)frame.push_back(static_cast<std::byte>(byte));
    u64(plan);u64(profile);u64(draft_chain);u64(draft_exact);u64(draft_q4);
    const char* q4_all=std::getenv("STRIX_MTP_DRAFT_Q4_ALL");u64(!q4_all || q4_all[0]!='0');
    cache.extend_identity(sha256_bytes(frame));
    if(restore) cache.set_restore_hook(std::move(restore));
}
// One live session's view of the store. Capture copies the state after a completed tiled block;
// restore copies a checkpoint into a fresh session and republishes its recorded metadata.
class HipPrefixStateCacheSession final {
public:
    HipPrefixStateCacheSession(std::shared_ptr<PrefixStateCache> store, ModelContentIdentity identity,
        std::uint32_t block_rows, std::uint64_t every, const ModelContract& model,
        const SerialSessionStateLayout& layout, const SessionStateMemoryBinding& memory)
        : store_(std::move(store)), identity_(identity), block_rows_(block_rows), every_(every),
          layout_(layout), memory_(memory), rotary_(model.rotary_identity()) {
        // A null store: the component registry only (--rtx-state-import / STRIX_RTX_EXPORT_STATE without a prefix
        // cache): no capture, no lookup, no disk tier.
        if (!identity_.valid() || !block_rows_ || !every_ || every_ % block_rows_)
            throw std::invalid_argument("prefix state cache session needs an identity and an interval multiple of T");
        validate_session_state_memory_binding(layout_, memory_);
        if (memory_.stage_mapping) throw std::invalid_argument("prefix state cache does not support staged state");
        // Row components: every attention layer's KV rows, then every layer's raw index rows.
        for (const auto component : {SessionStateComponent::attention_kv, SessionStateComponent::sparse_index}) {
            if (!find_session_state_allocation(layout_, component)) continue;
            for (std::uint64_t ordinal = 0; ordinal < model.full_attention_layers; ++ordinal) {
                const auto layer = static_cast<std::uint32_t>((ordinal + 1U) * model.full_attention_interval - 1U);
                const auto address = resolve_indexed_attention_layer_state(layout_, memory_, model, layer, component);
                if (address.storage != SessionStateStorage::device || !layout_.context_tokens ||
                    address.bytes % layout_.context_tokens)
                    throw std::invalid_argument("prefix state cache row component is not a per-position device range");
                rows_.push_back({address.address, address.bytes / layout_.context_tokens, 0U,
                    std::string(component == SessionStateComponent::attention_kv ? "kv.L" : "index.L") + std::to_string(layer)});
                row_bytes_ += address.bytes / layout_.context_tokens;
            }
        }
        // Fixed components: whole allocations, overwritten in place.
        for (const auto component : {SessionStateComponent::recurrent_matrix, SessionStateComponent::gdn_convolution,
                 SessionStateComponent::ple_convolution, SessionStateComponent::ngram_history,
                 SessionStateComponent::position_history}) {
            if (!find_session_state_allocation(layout_, component)) continue;
            const auto address = resolve_session_state(layout_, memory_, component);
            const char* name = component == SessionStateComponent::recurrent_matrix ? "recurrent_matrix"
                : component == SessionStateComponent::gdn_convolution ? "gdn_convolution"
                : component == SessionStateComponent::ple_convolution ? "ple_convolution"
                : component == SessionStateComponent::ngram_history ? "ngram_history" : "position_history";
            fixed_.push_back({address.address, address.bytes, fixed_bytes_, address.storage == SessionStateStorage::device, name});
            fixed_bytes_ += (address.bytes + 255U) & ~std::uint64_t{255};
        }
        for (const auto& allocation : layout_.allocations) {
            const bool known = allocation.component == SessionStateComponent::attention_kv ||
                allocation.component == SessionStateComponent::sparse_index ||
                std::any_of(fixed_.begin(), fixed_.end(), [&](const Fixed& f) {
                    return f.address == resolve_session_state(layout_, memory_, allocation.component).address; });
            if (!known) throw std::invalid_argument("prefix state cache does not cover every session state component");
        }
        target_rows_=rows_.size();target_fixed_=fixed_.size();
        if (hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) != hipSuccess)
            throw std::runtime_error("prefix state cache stream creation failed");
    }
    ~HipPrefixStateCacheSession() { if (stream_) (void)hipStreamDestroy(stream_); }
    HipPrefixStateCacheSession(const HipPrefixStateCacheSession&) = delete;
    HipPrefixStateCacheSession& operator=(const HipPrefixStateCacheSession&) = delete;

    // Extensions, before the first request (MTP predictor state on the MTP route):
    // per-position rows stored `shift` positions late (row j is kept with the block that holds
    // position j + shift: the MTP cache row of pair (j, j+1) needs the token at j + 1) ...
    void add_rows(std::uintptr_t address, std::uint64_t row_bytes, std::uint32_t shift, std::string id = {}) {
        require_unused();
        if (!address || !row_bytes || shift > 1U) throw std::invalid_argument("prefix state cache extra rows are invalid");
        if (id.empty()) id = "extra.rows." + std::to_string(rows_.size());
        rows_.push_back({address, row_bytes, shift, std::move(id)});
        row_bytes_ += row_bytes;
    }
    // ... whole device ranges saved with the fixed state ...
    void add_fixed_device(std::uintptr_t address, std::uint64_t bytes, std::string id = {}) {
        require_unused();
        if (!address || !bytes) throw std::invalid_argument("prefix state cache extra fixed range is invalid");
        if (id.empty()) id = "extra.fixed." + std::to_string(fixed_.size());
        fixed_.push_back({address, bytes, fixed_bytes_, true, std::move(id)});
        fixed_bytes_ += (bytes + 255U) & ~std::uint64_t{255};
    }
    // ... more identity (the consumers of the extra state), and a hook run after a restore.
    void extend_identity(const ModelContentIdentity& extra) {
        require_unused();
        std::vector<std::byte> frame(64);
        std::memcpy(frame.data(), identity_.bytes.data(), 32);
        std::memcpy(frame.data() + 32, extra.bytes.data(), 32);
        identity_ = sha256_bytes(frame);
        identity_extension_ = extra;
    }
    void set_before_restore_hook(std::function<void()> hook) { require_unused(); before_restore_hook_ = std::move(hook); }
    void set_restore_hook(std::function<void(std::uint64_t)> hook) { require_unused(); restore_hook_ = std::move(hook); }
    // Only the serialized request runner may rebind this before begin_request.
    // Slot 0 alternates between a batch runtime and the idle named-session runtime.
    // Keep before_restore_hook_ separate: rebinding must retain M1 invalidation.
    void bind_restore_runtime(std::function<void(std::uint64_t)> hook) {restore_hook_=std::move(hook);}
    [[nodiscard]] const ModelContentIdentity& identity() const noexcept { return identity_; }
    // Bytes of one segment (block k): every row component's rows of that block.
    [[nodiscard]] std::uint64_t segment_bytes(std::uint64_t k) const {
        std::uint64_t total = 0;
        for (const auto& row : rows_) total += row.bytes * row_count(row, k);
        return total;
    }
    // SHA-256 of the state a session holds after `position` prompt rows: every row component's rows
    // [0, position - shift) and every fixed range (diagnostic, STRIX_PREFIX_CACHE_DIGEST=1).
    [[nodiscard]] ModelContentIdentity state_digest(std::uint64_t position,std::optional<std::uint64_t> predictor={}) {
        check(hipDeviceSynchronize(), "prefix digest fence");
        std::vector<std::byte> frame;
        std::vector<std::byte> chunk;
        const auto append = [&](const void* source, std::uint64_t bytes, bool device) {
            chunk.resize(static_cast<std::size_t>(bytes));
            if (device) check(hipMemcpy(chunk.data(), source, static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost), "prefix digest copy");
            else std::memcpy(chunk.data(), source, static_cast<std::size_t>(bytes));
            const auto digest = sha256_bytes(chunk);
            frame.insert(frame.end(), reinterpret_cast<const std::byte*>(digest.bytes.data()),
                reinterpret_cast<const std::byte*>(digest.bytes.data()) + 32);
        };
        for (const auto& row : rows_) {
            const auto count = predictor && row.shift ? *predictor : position > row.shift ? position - row.shift : 0U;
            if (count) append(reinterpret_cast<const void*>(row.address), count * row.bytes, true);
        }
        for (const auto& fixed : fixed_) append(reinterpret_cast<const void*>(fixed.address), fixed.bytes, fixed.device);
        return sha256_bytes(frame);
    }
    // Cross-route numerical digest: frozen target registry, occupied ranges,
    // ordered consumed token history and the unconsumed pending selection.
    // Opaque executor/window lineage remains in complete_session_digest.
    [[nodiscard]] ModelContentIdentity target_state_digest(std::uint64_t position,
        std::span<const std::uint32_t> history,std::optional<std::uint32_t> pending) {
        if(history.size()!=position) throw std::invalid_argument("target digest history/frontier differs");
        check(hipDeviceSynchronize(),"target digest fence");
        QwenTargetStateDigestFrame frame(position,target_rows_,target_fixed_);
        std::vector<std::byte> chunk;
        const auto range=[&](const std::string& id,std::uintptr_t address,std::uint64_t bytes,bool device) {
            chunk.resize(static_cast<std::size_t>(bytes));
            if(bytes) {
                if(device) check(hipMemcpy(chunk.data(),reinterpret_cast<const void*>(address),chunk.size(),hipMemcpyDeviceToHost),"target digest copy");
                else std::memcpy(chunk.data(),reinterpret_cast<const void*>(address),chunk.size());
            }
            frame.component(id,bytes,sha256_bytes(chunk));
        };
        for(std::size_t j=0;j<target_rows_;++j) {
            const auto& r=rows_[j];range(r.id,r.address,(position>r.shift?position-r.shift:0U)*r.bytes,true);
        }
        for(std::size_t j=0;j<target_fixed_;++j) {
            const auto& f=fixed_[j];range(f.id,f.address,f.bytes,f.device);
        }
        return frame.finish(history,pending);
    }
    // STRIX_PREFIX_CACHE_DIGEST_COMPONENTS=1 (with STRIX_PREFIX_CACHE_DIGEST=1): state_digest's per-component hashes as a JSON object (component id -> SHA-256 of its
    // committed rows below `position` or of the whole fixed allocation), to locate a digest difference.
    [[nodiscard]] std::string component_digests(std::uint64_t position) {
        check(hipDeviceSynchronize(), "prefix digest fence");
        std::vector<std::byte> chunk;
        std::string out = "{";
        const auto append = [&](const std::string& id, const void* source, std::uint64_t bytes, bool device) {
            chunk.resize(static_cast<std::size_t>(bytes));
            if (device) check(hipMemcpy(chunk.data(), source, static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost), "prefix digest copy");
            else std::memcpy(chunk.data(), source, static_cast<std::size_t>(bytes));
            out += (out.size() > 1U ? ",\"" : "\"") + id + "\":\"" + model_content_identity_hex(sha256_bytes(chunk)) + '"';
        };
        for (const auto& row : rows_) {
            const auto count = position > row.shift ? position - row.shift : 0U;
            if (count) append(row.id, reinterpret_cast<const void*>(row.address), count * row.bytes, true);
        }
        for (const auto& fixed : fixed_) append(fixed.id, reinterpret_cast<const void*>(fixed.address), fixed.bytes, fixed.device);
        return out + '}';
    }
    [[nodiscard]] ModelContentIdentity complete_session_digest(const SerialSessionContext& session,
        const SessionFrontier& frontier) {
        auto frame = encode_metadata(session);
        const auto u64 = [&](std::uint64_t value) {
            for (unsigned shift = 0; shift < 64; shift += 8)
                frame.push_back(static_cast<std::byte>((value >> shift) & 255U));
        };
        const auto state = state_digest(session.next_position);
        for (const auto byte : state.bytes) frame.push_back(static_cast<std::byte>(byte));
        u64(frontier.consumed); u64(frontier.emitted); u64(frontier.accepted); u64(frontier.predictor);
        u64(frontier.pending ? *frontier.pending : std::uint64_t{UINT32_MAX} + 1U);
        u64(frontier.sampling.size());
        for (const auto c : frontier.sampling) frame.push_back(static_cast<std::byte>(c));
        u64(frontier.tokens.size()); for (const auto token : frontier.tokens) u64(token);
        return sha256_bytes(frame);
    }
    [[nodiscard]] std::uint64_t row_bytes() const noexcept { return row_bytes_; }
    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept { return fixed_bytes_; }
    [[nodiscard]] const PrefixStateCacheRequestStats& request() const noexcept { return *request_; }
    [[nodiscard]] PrefixStateCache& store() noexcept { return *store_; }
    // False: the component registry of --rtx-state-import / STRIX_RTX_EXPORT_STATE alone (no capture, no lookup).
    void disable_store_before_ready() {
        PrefixAsyncCapture::reap(true); segments_.clear(); row_chain_->clear(); store_.reset();
    }
    [[nodiscard]] bool has_store() const noexcept { return store_ != nullptr; }

    // ---- externally produced state (strix-qwen-rtx-state-v1, docs/rtx-state-import.md) ----
    // The consumer's facts, before the first request. Needed by the import, reported by the export.
    void set_rtx_facts(const RtxStateImportFacts& facts) { require_unused(); rtx_facts_ = facts; }
    // The import source: begin_request restores a parked state of `registry` matching the prompt, before the
    // checkpoint lookup. Publishes this session's geometry to the registry (GET /admin/rtx-import/layout).
    void enable_rtx_import(std::shared_ptr<RtxStateImportRegistry> registry, const SerialSessionContext& fresh) {
        require_unused();
        if (!registry || !rtx_facts_) throw std::invalid_argument("rtx state import needs a registry and the consumer facts");
        rtx_registry_ = std::move(registry);
        rtx_fresh_execution_ = fresh.execution_identity;
        rtx_fresh_compatibility_ = fresh.runtime_compatibility_identity;
        rtx_registry_->set_layout(rtx_layout(fresh.state));
    }
    // The geometry of this session's state (row and fixed components in capture order).
    [[nodiscard]] RtxStateLayout rtx_layout(const PersistentStateGenerations& base) const {
        RtxStateLayout l;
        for (const auto& row : rows_) l.rows.push_back({row.id, row.bytes, row.shift});
        for (const auto& f : fixed_) l.fixed.push_back({f.id, f.bytes, f.offset});
        l.row_bytes = row_bytes_;
        l.fixed_bytes = fixed_bytes_;
        l.layout_identity = layout_.layout_identity;
        l.context_tokens = layout_.context_tokens;
        l.rotary = rotary_;
        if (rtx_facts_) {
            l.generations_per_row = rtx_facts_->generations_per_row;
            l.valid_union = static_cast<std::uint32_t>(rtx_facts_->valid_union);
            if (rtx_facts_->model_content.valid()) l.model_content_hex = model_content_identity_hex(rtx_facts_->model_content);
        }
        l.generations_base = base.generations;
        l.valid_base = static_cast<std::uint32_t>(base.valid);
        if (identity_extension_) l.identity_extension_hex = model_content_identity_hex(*identity_extension_);
        l.cache_identity_hex = model_content_identity_hex(identity_);
        l.execution_identity = rtx_fresh_execution_;
        l.runtime_compatibility_identity = rtx_fresh_compatibility_;
        return l;
    }
    // After a unit of the tiled prompt (the session at `position`): writes the STRIX_RTX_EXPORT_STATE file when
    // this is the requested position (default: the end of the prompt). Rows [0, position - shift) of every row
    // component and the fixed state: exactly what an import restores.
    void after_unit(const SerialSessionContext& session, std::span<const std::uint32_t> prompt, std::uint64_t position) {
        PrefixAsyncCapture::reap();
        const auto& request = rtx_state_export_request();
        if (!request || !position || position > prompt.size() || session.next_position != position ||
            (request->position ? request->position != position : position != prompt.size()))
            return;
        if (session.health != SerialSessionHealth::ready || session.arithmetic_lineage.tainted ||
            session.arithmetic_lineage.intervals.empty() || !arithmetic_lineage_valid(session.arithmetic_lineage, position))
            throw std::runtime_error("STRIX_RTX_EXPORT_STATE: the session cannot be exported at this position");
        const auto start = std::chrono::steady_clock::now();
        auto layout = rtx_layout(session.state);
        if (!layout.execution_identity) {
            layout.execution_identity = session.execution_identity;
            layout.runtime_compatibility_identity = session.runtime_compatibility_identity;
        }
        std::vector<std::byte> payload(static_cast<std::size_t>(layout.payload_bytes(position)));
        check(hipDeviceSynchronize(), "rtx export fence");
        std::uint64_t at = 0;
        for (const auto& row : rows_) {
            const auto count = position > row.shift ? position - row.shift : 0U;
            if (count)
                check(hipMemcpy(payload.data() + at, reinterpret_cast<const void*>(row.address), static_cast<std::size_t>(count * row.bytes),
                    hipMemcpyDeviceToHost), "rtx export rows");
            at += count * row.bytes;
        }
        for (const auto& fixed : fixed_) {
            if (fixed.device)
                check(hipMemcpy(payload.data() + at + fixed.offset, reinterpret_cast<const void*>(fixed.address),
                    static_cast<std::size_t>(fixed.bytes), hipMemcpyDeviceToHost), "rtx export fixed state");
            else
                std::memcpy(payload.data() + at + fixed.offset, reinterpret_cast<const void*>(fixed.address), static_cast<std::size_t>(fixed.bytes));
        }
        RtxStateFileSession facts;
        facts.next_position = session.next_position;
        facts.committed_tokens = session.committed_tokens;
        facts.valid = static_cast<std::uint32_t>(session.state.valid);
        facts.generations = session.state.generations;
        for (const auto& i : session.arithmetic_lineage.intervals)
            facts.lineage.push_back({i.begin, i.end, static_cast<std::uint64_t>(i.producer), i.numerical_plan_identity,
                i.reduction_plan_identity, i.state_update_identity});
        // The producer of an AMD export: the first interval's identities (they name this engine's arithmetic).
        const auto& first = session.arithmetic_lineage.intervals.front();
        const RtxStateProducer producer{first.numerical_plan_identity, first.reduction_plan_identity, first.state_update_identity};
        const auto sha = write_rtx_state_file(request->path, layout, prompt.first(static_cast<std::size_t>(position)), position, producer,
            facts, payload);
        std::cerr << "strix rtx export: position " << position << " payload " << payload.size() << " B sha256 "
                  << model_content_identity_hex(sha) << " -> " << request->path << " ("
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << " ms)" << std::endl;
    }

    // Restores the longest cached prefix of `prompt` that leaves at least one prompt token to run
    // into the fresh `session`. Returns the restored position (0: miss, nothing changed).
    // Policy selection state and verify-owner equivalence are not in the prefix certificate.
    void set_policy_selection(bool enabled) { policy_selection_ = enabled; }
    void set_retention(bool retain, std::string conversation = {}) {
        retain_ = retain;
        conversation_ = std::move(conversation);
    }
    // Bind the safe full-T producer family independently of the legacy v1 namespace.
    // Request-local certificates validate the whole cold prefix before any device write.
    std::vector<std::uint64_t> execution_certificate(std::uint64_t position) const {
        auto proof=serial_path_?qwen_serial_certificate(plan_,serial_cold_,plan_n_,position):
            (plan_route_==2?qwen_full_t_certificate_v2(plan_,plan_n_,position):qwen_full_t_certificate(plan_,plan_n_,position));
        if(proof.empty()) return {};
        const auto& binding=plan_route_==2?plan_identity_:identity_;
        for(unsigned i=0;i<binding.bytes.size();i+=8) {
            std::uint64_t word=0;
            for(unsigned j=0;j<8;++j) word|=std::uint64_t(binding.bytes[i+j])<<(8*j);
            proof.push_back(word);
        }
        return proof;
    }
    void set_plan(const QwenTiledPlanConfig& config, std::uint64_t n, bool enabled,
        std::uint64_t route=1,QwenBatchConfig batch={},bool allow_restore=true,bool nesting_possible=false) {
        plan_route_=route;batch_config_=batch;allow_restore_=allow_restore;nesting_possible_=nesting_possible;
        serial_path_=false;
        plan_=config; plan_n_=n; plan_enabled_=enabled; certified_path_=true;
        std::vector<std::byte> frame;
        for(const auto b:identity_.bytes) frame.push_back(static_cast<std::byte>(b));
        const auto put=[&](std::uint64_t v) {for(unsigned k=0;k<8;++k) frame.push_back(static_cast<std::byte>((v>>(k*8U))&255U));};
        put(0x514650554c4c5431ULL); put(config.rows); put(config.schedule); put(config.priming);
        put(config.tiled_start); put(config.suppressed); put(config.force_sync_priming); put(static_cast<std::uint64_t>(config.block.policy));
        put(config.block.execution);put(config.block.kernel);
        put(config.decode_execution);put(config.decode_kernel);put(config.prefill_execution);put(config.prefill_kernel);
        put(config.prefill_profile);put(config.decode_profile);put(config.profile_pair);
        put(config.valid_state);
        for(const auto g:config.initial_generations) put(g);
        for(const auto g:config.generations_per_row) put(g);
        put(config.tails.size());
        for(const auto& t:config.tails) {put(t.rows);put(static_cast<std::uint64_t>(t.policy));put(t.execution);put(t.kernel);}
        put(config.ladder.size()); for(const auto r:config.ladder) put(r);
        if(route==2) {
            put(0x5146554c4c5432ULL);put(route);
            for(const auto value:batch.frame()) put(value);
            put(config.schedule&2U);
        }
        plan_identity_=sha256_bytes(frame);
    }
    void set_serial_plan(const QwenTiledPlanConfig& config,std::uint64_t n,
        std::vector<KnownPromptChunkUnit> trace,std::function<bool(std::uint64_t)> can_resume,bool enabled) {
        set_plan(config,n,enabled);
        serial_path_=true;serial_cold_=std::move(trace);serial_executed_.clear();
        serial_can_resume_=std::move(can_resume);
        std::vector<std::byte> frame;
        for(const auto b:plan_identity_.bytes) frame.push_back(static_cast<std::byte>(b));
        for(unsigned k=0;k<8;++k) frame.push_back(static_cast<std::byte>((qwen_serial_certificate_route>>(8*k))&255U));
        plan_identity_=sha256_bytes(frame);
    }
    void observe_serial_unit(const KnownPromptChunkUnit& unit) {serial_executed_.push_back(unit);}
    void certificate_decline(PrefixCertificateDecline reason) {
        const auto name=prefix_certificate_decline_names[static_cast<std::size_t>(reason)];
        request_->refusal=name; ++request_->certificate_declines[std::string(name)];
        if(store_) store_->record_certificate_decline(reason);
    }
    [[nodiscard]] const std::optional<QwenPlanCursor>& continuation() const {return continuation_;}
    [[nodiscard]] const ModelContentIdentity& active_identity() const {return certified_path_?plan_identity_:identity_;}
    void request_capture(std::uint64_t position) { requested_captures_.insert(position); }
    [[nodiscard]] std::uint64_t cached_boundary(std::span<const std::uint32_t> prompt) const {
        if(!store_ || prompt.empty()) return 0;
        const auto digests=prefix_cache_block_digests(active_identity(),prompt,block_rows_,(prompt.size()-1U)/block_rows_);
        for(auto k=digests.size();k>0;--k) if(store_->contains(digests[k-1U])) return k*block_rows_;
        return 0;
    }
    [[nodiscard]] bool joining_allowed() const {
        static const bool lookup=[] {const char* v=std::getenv("STRIX_PREFIX_CACHE_LOOKUP");return !v || v[0]!='0';}();
        return store_ && !store_->policy().legacy && certified_path_ && plan_enabled_ &&
            plan_route_==2 && allow_restore_ && !batch_config_.groups && !policy_selection_ &&
            lookup && !imported_ && qwen_prefix_join_grid(block_rows_,plan_.rows);
    }
    [[nodiscard]] bool retaining() const {return retain_;}
    [[nodiscard]] std::uint64_t join_boundary(const HipPrefixStateCacheSession& producer,
        std::uint64_t common,std::uint64_t from) const {
        if(!joining_allowed() || !producer.joining_allowed() || !producer.retain_ ||
            (batch_config_.short_rows && plan_n_<=batch_config_.short_rows) ||
            active_identity()!=producer.active_identity()) return 0;
        return qwen_prefix_join_boundary(producer.plan_,producer.plan_n_,plan_,plan_n_,common,from,
            batch_config_,nesting_possible_);
    }
    std::uint64_t begin_request(SerialSessionContext& session, std::span<const std::uint32_t> prompt) {
        PrefixAsyncCapture::reap();
        used_ = true;
        request_ = std::make_shared<PrefixStateCacheRequestStats>();
        continuation_.reset();
        requested_captures_.clear();
        request_->prompt_tokens = prompt.size();
        segments_.clear();
        row_chain_=std::make_shared<PrefixAsyncCapture::RowChain>();
        if (session.next_position || session.committed_tokens || session.health != SerialSessionHealth::ready ||
            !session.arithmetic_lineage.intervals.empty() || session.state_layout_identity != layout_.layout_identity)
            throw std::invalid_argument("prefix state cache restores only into a fresh ready session");
        imported_ = false;
        if (policy_selection_) {
            if (store_ || rtx_registry_) certificate_decline(PrefixCertificateDecline::policy_selection_cache_unqualified);
            return 0;
        }
        if (before_restore_hook_) before_restore_hook_();
        if (rtx_registry_ && prompt.size() >= 2U)
            if (auto state = rtx_registry_->take(prompt)) return restore_rtx_state(session, std::move(state));
        if (!store_) return 0;
        // STRIX_PREFIX_CACHE_LOOKUP=0: capture only (diagnostic; every request prefills from position 0).
        static const bool lookup = [] { const char* v = std::getenv("STRIX_PREFIX_CACHE_LOOKUP"); return !v || v[0] != '0'; }();
        if(serial_path_ && (!plan_enabled_ || block_rows_!=plan_.rows)) {
            certificate_decline(PrefixCertificateDecline::serial_plan_unsupported);return 0;
        }
        if (prompt.size() < 2U || !lookup) {
            certificate_decline(serial_path_?PrefixCertificateDecline::serial_lookup_disabled_or_short:PrefixCertificateDecline::prefix_lookup_disabled_or_short);
            return 0;
        }
        const auto start = std::chrono::steady_clock::now();
        std::function<bool(const PrefixCacheCheckpoint&)> eligible;
        if(certified_path_) eligible=[&](const PrefixCacheCheckpoint& c) {
            if(!allow_restore_) {certificate_decline(PrefixCertificateDecline::prefix_restore_disabled);return false;}
            if(plan_route_==2 &&
                !qwen_suffix_solo_class(plan_,plan_n_,c.position,batch_config_,nesting_possible_)) {
                certificate_decline(PrefixCertificateDecline::batch_suffix_not_solo);return false;
            }
            if(serial_path_ && (!plan_enabled_ || block_rows_!=plan_.rows)) {certificate_decline(PrefixCertificateDecline::serial_plan_unsupported);return false;}
            if(!plan_enabled_ || c.identity!=active_identity() || c.block_rows!=block_rows_ || c.position>=prompt.size() ||
                c.producer_certificate.empty()) {
                certificate_decline(serial_path_?PrefixCertificateDecline::serial_certificate_missing_or_incompatible:PrefixCertificateDecline::prefix_certificate_missing_or_incompatible);
                return false;
            }
            const auto proof=execution_certificate(c.position);
            if(proof.empty() || proof!=c.producer_certificate) {
                certificate_decline(serial_path_?PrefixCertificateDecline::serial_producer_plan_mismatch:PrefixCertificateDecline::prefix_producer_plan_mismatch);
                return false;
            }
            if(serial_path_ && (!serial_can_resume_ || !serial_can_resume_(c.position))) {
                certificate_decline(PrefixCertificateDecline::serial_cursor_continuation_unsupported);return false;
            }
            try { (void)preflight_checkpoint(session,c); }
            catch(const std::runtime_error&) {
                certificate_decline(serial_path_?PrefixCertificateDecline::serial_checkpoint_preflight:PrefixCertificateDecline::prefix_checkpoint_preflight);return false;
            }
            if(serial_path_) return true;
            continuation_=qwen_full_t_continuation(plan_,plan_n_,c.position);
            if(!continuation_) certificate_decline(PrefixCertificateDecline::prefix_continuation_unsupported);
            return continuation_.has_value();
        };
        const auto checkpoint = store_->find_longest(active_identity(), prompt, block_rows_, prompt.size() - 1U, retain_, conversation_,eligible);
        request_->lookup_ms = ms(start);
        if (!checkpoint) {
            if(request_->certificate_declines.empty())
                certificate_decline(serial_path_?PrefixCertificateDecline::serial_no_certified_checkpoint:PrefixCertificateDecline::prefix_no_compatible_checkpoint);
            return 0;
        }
        if(serial_path_) for(const auto& u:serial_cold_) {
            if(u.end>checkpoint->position) break;
            serial_executed_.push_back(u);
        }
        const auto restore_start = std::chrono::steady_clock::now();
        const auto meta = restore_checkpoint(session, *checkpoint);
        session.next_position = meta.position;
        session.committed_tokens = meta.committed_tokens;
        session.state = meta.state;
        session.arithmetic_lineage = meta.lineage;
        if (!arithmetic_lineage_valid(session.arithmetic_lineage, session.next_position)) {
            session.health = SerialSessionHealth::poisoned;
            throw std::runtime_error("prefix state cache restored an invalid arithmetic lineage");
        }
        if (retain_) {segments_ = checkpoint->segments; *row_chain_=checkpoint->segments;}
        if (restore_hook_) restore_hook_(meta.position);
        request_->restored_position = meta.position;
        request_->restore_ms = ms(restore_start);
        return meta.position;
    }

    // Reset callbacks invoke this after their incumbent device fence, before reset writes.
    void before_target_reset() const {PrefixAsyncCapture::before_writer(capture_tag_,nullptr);}
    double attach_capture_pool(bool automatic_admitted=false) {
        if(!automatic_admitted || !store_ || store_->policy().legacy || store_->capture_resources) return 0;
        const auto start=std::chrono::steady_clock::now();
        auto resources=std::make_shared<PrefixCaptureResources>();
        resources->cap=staging_cap_?staging_cap_:row_bytes_*block_rows_+fixed_bytes_;
        check(hipMalloc(&resources->staging,static_cast<std::size_t>(resources->cap)),"prefix pool staging allocation");
        check(hipStreamCreateWithFlags(&resources->download,hipStreamNonBlocking),"prefix pool download stream");
        for(auto* event:{&resources->target_ready,&resources->mtp_ready,&resources->done})
            check(hipEventCreateWithFlags(event,hipEventDisableTiming),"prefix pool event");
        // Reserve bounded metadata inside B; the slab includes every free-list byte.
        const auto budget=store_->policy().host_byte_budget;
        const auto metadata=std::min<std::uint64_t>(budget/8U,UINT64_C(1)<<20);
        // Round the slab down, charging its alignment remainder to metadata too.
        const auto slab=(budget-metadata)&~UINT64_C(255);
        auto pool=std::make_shared<PrefixPayloadPool>(hip_pinned_prefix_capture_buffer(slab));
        store_->attach_payload_pool(std::move(pool),budget-slab);
        store_->set_pool_staging_bytes(resources->cap);
        store_->capture_resources=std::move(resources);
        return ms(start);
    }
    void set_staging_cap(std::uint64_t bytes) { staging_cap_=bytes; }
    using Capture = std::shared_ptr<PrefixAsyncCapture>;
    Capture try_prepare_capture(const SerialSessionContext& session, std::span<const std::uint32_t> prompt,
        hipStream_t producer, bool target_valid, PrefixCaptureOrder order={}, std::uint64_t covered=UINT64_MAX,
        std::shared_ptr<const void> target_owner={},std::shared_ptr<const void> predictor_owner={},const void* owner_tag=nullptr) {
        return prepare_capture(session,prompt,producer,target_valid,order,covered,
            std::move(target_owner),std::move(predictor_owner),owner_tag,false);
    }
    [[nodiscard]] bool pooled_capture() const {return store_ && bool(store_->capture_resources);}
    Capture try_prepare_target_capture(const SerialSessionContext& session, std::span<const std::uint32_t> prompt,
        hipStream_t producer, bool target_valid, PrefixCaptureOrder order={}, std::uint64_t covered=UINT64_MAX,
        std::shared_ptr<const void> target_owner={},std::shared_ptr<const void> predictor_owner={},const void* owner_tag=nullptr) {
        if(!pooled_capture()) return {};
        return prepare_capture(session,prompt,producer,target_valid,order,covered,
            std::move(target_owner),std::move(predictor_owner),owner_tag,true);
    }
    // Called only at the original complete boundary, after priming and held-row update.
    // In particular predictor_complete includes the incumbent predictor-frontier check.
    bool complete_target_capture(const Capture& d,const SerialSessionContext& session,
        bool predictor_complete,hipStream_t predictor) {
        if(!d || !d->split_target || d->metadata_complete) return false;
        const auto p=d->checkpoint->position;
        if(d->cancelled || !predictor_complete || session.next_position!=p || session.committed_tokens!=p ||
            session.health!=SerialSessionHealth::ready || d->checkpoint->identity!=active_identity() ||
            !certified_metadata(session.state,session.arithmetic_lineage,p)) {
            d->cancel_target();return false;
        }
        try {
            d->checkpoint->metadata=encode_metadata(session);
            d->checkpoint->producer_certificate=execution_certificate(p);
            if(d->checkpoint->producer_certificate.empty()) {d->cancel_target();return false;}
            if(d->publish) for(const auto& f:fixed_) if(!f.device)
                std::memcpy(d->checkpoint->fixed.data.get()+f.offset,reinterpret_cast<const void*>(f.address),
                    static_cast<std::size_t>(f.bytes));
            d->metadata_complete=true;
            d->submit_predictor(predictor);
            return true;
        } catch(const PrefixCaptureHipError&) {throw;}
          catch(...) {d->cancel_target();throw;}
    }
private:
    Capture prepare_capture(const SerialSessionContext& session, std::span<const std::uint32_t> prompt,
        hipStream_t producer, bool target_valid, PrefixCaptureOrder order, std::uint64_t covered,
        std::shared_ptr<const void> target_owner,std::shared_ptr<const void> predictor_owner,const void* owner_tag,
        bool split_target) {
        PrefixAsyncCapture::reap();
        if(serial_path_) {certificate_decline(PrefixCertificateDecline::serial_async_capture_unsupported);return {};}
        const auto p=session.next_position;
        if(policy_selection_ || !store_ || store_->policy().legacy || !certified_path_ || !plan_enabled_ || imported_ ||
            !p || p>=prompt.size() || p%block_rows_ || session.committed_tokens!=p ||
            session.health!=SerialSessionHealth::ready || !target_valid ||
            !certified_metadata(session.state,session.arithmetic_lineage,p) ||
            !qwen_full_t_continuation(plan_,plan_n_,p)) return {};
        const bool selected=requested_captures_.contains(p) || p%every_==0 || p+block_rows_>std::min<std::uint64_t>(covered,prompt.size());
        // Unselected boundaries may prefetch rows privately. They never publish checkpoints.
        // This lets a one-block staging cap feed a checkpoint cadence larger than T.
        if(!retain_) {store_->record_no_store_skip();return {};}
        auto cursor=*qwen_full_t_continuation(plan_,plan_n_,p);
        const auto next=qwen_prompt_next(plan_,cursor,p,plan_n_);
        // Even a matching handle cannot authorize a different tail/window/serial owner.
        // An incumbent partial block of this same T owner uses the same ordered stream.
        if(!next || !next->t_owner || next->owner!=plan_.block ||
            !order.next_target || *order.next_target!=producer || !order.predictor ||
            !order.next_predictor || *order.predictor!=*order.next_predictor) {
            store_->record_cross_stream_skip();return {};
        }
        const auto resources=std::static_pointer_cast<PrefixCaptureResources>(store_->capture_resources);
        if(PrefixAsyncCapture::slot) {store_->record_staging_skip(false);if(resources) store_->record_no_slot();return {};}
        if(resources && resources->poisoned) {store_->record_no_slot();return {};}
        Capture d;
        if(prefix_batch_certification_enabled()) store_->record_snapshot_preparation();
        try {
            d=std::make_shared<PrefixAsyncCapture>(); d->store=store_;
            d->split_target=split_target;d->metadata_complete=!split_target;
            d->owner_tag=owner_tag;d->target_stream=producer;
            d->publish=selected; d->row_chain=row_chain_; d->request=request_;
            d->target_owner=std::move(target_owner);d->predictor_owner=std::move(predictor_owner);
            const auto blocks=p/block_rows_;
            const auto digests=prefix_cache_block_digests(active_identity(),prompt,block_rows_,blocks);
            if(digests.size()!=blocks || store_->contains(digests.back())) return {};
            auto c=std::make_shared<PrefixCacheCheckpoint>(); d->checkpoint=c;
            c->identity=active_identity(); c->digest=digests.back(); c->position=p; c->block_rows=block_rows_;
            c->conversation=conversation_;
            if(split_target) {
                // Charge the same metadata/certificate sizes before DMA; freeze values only post-prime.
                c->metadata.resize(encode_metadata(session).size());
                c->producer_certificate.resize(execution_certificate(p).size());
            } else {
                c->metadata=encode_metadata(session);c->producer_certificate=execution_certificate(p);
            }
            c->fixed.bytes=selected?fixed_bytes_:0;
            std::vector<std::shared_ptr<PrefixCacheSegment>> missing;
            for(std::uint64_t k=0;k<blocks;++k) {
                auto retained=k<row_chain_->size()?(*row_chain_)[k]:store_->find_segment(digests[k]);
                if(retained) {c->segments.push_back(std::move(retained));continue;}
                auto segment=std::make_shared<PrefixCacheSegment>();
                segment->identity=c->identity;segment->digest=digests[k];segment->parent=k?digests[k-1U]:c->identity;
                segment->block_rows=block_rows_;segment->block_index=k;
                segment->tokens.assign(prompt.begin()+static_cast<std::ptrdiff_t>(k*block_rows_),
                    prompt.begin()+static_cast<std::ptrdiff_t>((k+1U)*block_rows_));
                segment->payload.bytes=segment_bytes(k); d->bytes+=segment->payload.bytes; d->row_bytes+=segment->payload.bytes;
                c->segments.push_back(segment);missing.push_back(std::move(segment));
            }
            if(selected) for(const auto& f:fixed_) if(f.device) d->bytes+=f.bytes;
            const auto cap=staging_cap_?staging_cap_:row_bytes_*plan_.rows+fixed_bytes_;
            store_->record_staging(cap,0,false);
            if(d->bytes>cap || (resources && d->bytes>resources->cap)) {store_->record_staging_skip(true);return {};}
            d->reservation=store_->reserve(store_->incremental_bytes(*c),c.get());
            if(!d->reservation) {if(resources) store_->record_no_slot();return {};}
            if(selected) c->fixed=store_->allocate(fixed_bytes_,d->reservation);
            for(const auto& segment:missing) segment->payload=store_->allocate(segment->payload.bytes,d->reservation,true);
            if(resources) {
                d->resources=resources;
                d->staging=resources->staging;d->download=resources->download;
                d->target_ready=resources->target_ready;d->mtp_ready=resources->mtp_ready;d->done=resources->done;
            } else {
                auto e=hipMalloc(&d->staging,static_cast<std::size_t>(d->bytes));
                if(e==hipErrorOutOfMemory) {store_->record_staging_skip(true);return {};}
                check(e,"prefix staging allocation");
                check(hipStreamCreateWithFlags(&d->download,hipStreamNonBlocking),"prefix D2H stream");
                for(auto* event:{&d->target_ready,&d->mtp_ready,&d->done})
                    check(hipEventCreateWithFlags(event,hipEventDisableTiming),"prefix capture event");
            }
            std::uint64_t offset=0;
            for(const auto& segment:missing) {
                std::uint64_t dest=0;
                for(const auto& row:rows_) {
                    const auto bytes=row.bytes*row_count(row,segment->block_index);
                    if(bytes) d->slices.push_back({reinterpret_cast<const void*>(row.address+row_first(row,segment->block_index)*row.bytes),
                        segment->payload.data.get()+dest,offset,bytes,row.shift!=0});
                    offset+=bytes;dest+=bytes;
                }
            }
            if(selected) for(const auto& f:fixed_) {
                if(f.device) {
                    d->slices.push_back({reinterpret_cast<const void*>(f.address),c->fixed.data.get()+f.offset,offset,f.bytes,
                        f.id=="mtp.held"}); offset+=f.bytes;
                } else if(!split_target) std::memcpy(c->fixed.data.get()+f.offset,reinterpret_cast<const void*>(f.address),static_cast<std::size_t>(f.bytes));
            }
            // All allocations, host history and credits exist before the first submission.
            capture_tag_=owner_tag;
            PrefixAsyncCapture::slot=d;
            store_->record_staging(cap,d->bytes,true);
            d->started=true;
            try {
                d->snapshot(false,producer);
                if(split_target) d->submit_target_download();
            } catch(...) {d->fatal();throw;}
            return d;
        } catch(const PrefixCacheAllocationFailure&) {if(resources) store_->record_no_slot();else store_->record_capture_failure();return {};}
          catch(const std::bad_alloc&) {if(d && d->started) {d->fatal();throw;} store_->record_capture_failure();return {};}
          catch(const PrefixCaptureHipError& e) {
              if(!d->started && e.error==hipErrorOutOfMemory) {store_->record_capture_failure();return {};}
              throw;
          }
    }

public:
    // After a completed tiled block: the session is at a block boundary of `prompt` (absolute
    // positions). Captures it when the boundary is selected and not cached yet.
    // `covered` is how far complete T-row blocks can run for this prompt (the prompt length, or one
    // less when the last prompt row must run serially): the last boundary below it is always captured.
    void after_block(const SerialSessionContext& session, std::span<const std::uint32_t> prompt,
        std::uint64_t covered = UINT64_MAX) {
        // An imported state was produced elsewhere: nothing derived from it enters the store under this identity.
        if (policy_selection_ || !store_ || imported_) return;
        const auto position = session.next_position;
        covered = std::min<std::uint64_t>(covered, prompt.size());
        if(serial_path_ && (!position || position>=prompt.size() || position%block_rows_ ||
            session.committed_tokens!=position || session.health!=SerialSessionHealth::ready)) {
            certificate_decline(PrefixCertificateDecline::serial_capture_boundary_unsupported);return;
        }
        if (!position || position % block_rows_ || position > prompt.size() || session.health != SerialSessionHealth::ready ||
            session.arithmetic_lineage.tainted || session.committed_tokens != position) {
            if(serial_path_) certificate_decline(PrefixCertificateDecline::serial_capture_metadata_mismatch);
            return;
        }
        if(serial_path_) {
            if(!plan_enabled_ || block_rows_!=plan_.rows) {certificate_decline(PrefixCertificateDecline::serial_plan_unsupported);return;}
            if(!known_prompt_full_t_prefix(serial_executed_,position,plan_.rows) ||
                qwen_serial_certificate(plan_,serial_executed_,plan_n_,position)!=
                qwen_serial_certificate(plan_,serial_cold_,plan_n_,position)) {
                certificate_decline(PrefixCertificateDecline::serial_capture_unit_unsupported);return;
            }
        }
        if(certified_path_ && (!plan_enabled_ || (!serial_path_ && !qwen_full_t_continuation(plan_,plan_n_,position)) ||
            !qwen_capture_boundary(position,prompt.size(),block_rows_,
                certified_metadata(session.state,session.arithmetic_lineage,position),false,true))) {
            if(serial_path_) certificate_decline(PrefixCertificateDecline::serial_capture_metadata_mismatch);
            return;
        }
        if (!prefix_capture_boundary_selected(position,covered,block_rows_,every_)) return;
        if (!retain_) { store_->record_no_store_skip(); return; }
        if(prefix_batch_certification_enabled()) store_->record_snapshot_preparation();
        const auto blocks = position / block_rows_;
        const auto start = std::chrono::steady_clock::now();
        // Own every destination until the capture stream has drained, including on failure.
        auto dma_owner = std::make_unique<PrefixDmaQuarantine>();
        auto& checkpoint = dma_owner->checkpoint;
        auto& pending = dma_owner->pending;
        auto reservation = PrefixStateCache::Reservation{};
        const auto drain = [&] {
            const auto error = hipStreamSynchronize(stream_);
            if (error != hipSuccess) PrefixDmaQuarantine::retain(dma_owner);
            check(error, "prefix capture failure drain");
        };
        const auto fail = [&] {
            drain();
            store_->record_capture_failure();
        };
        try {
            const auto digests = prefix_cache_block_digests(active_identity(), prompt, block_rows_, blocks);
            if (digests.size() != blocks || store_->contains(digests.back())) return;
            checkpoint = std::make_shared<PrefixCacheCheckpoint>();
            checkpoint->conversation = conversation_;
            checkpoint->identity = active_identity();
            if(certified_path_) checkpoint->producer_certificate=execution_certificate(position);
            checkpoint->digest = digests.back();
            checkpoint->position = position;
            checkpoint->block_rows = block_rows_;
            checkpoint->metadata = encode_metadata(session);
            checkpoint->fixed.bytes = fixed_bytes_;
            // Build the insertion estimate without allocating any pinned payload.
            for (std::uint64_t k = 0; k < blocks; ++k) {
                if (k < segments_.size()) { checkpoint->segments.push_back(segments_[k]); continue; }
                if (auto retained = store_->find_segment(digests[k])) {
                    checkpoint->segments.push_back(std::move(retained)); continue;
                }
                auto segment = std::make_shared<PrefixCacheSegment>();
                segment->identity = active_identity(); segment->digest = digests[k];
                segment->parent = k ? digests[k - 1U] : active_identity();
                segment->block_index = k; segment->block_rows = block_rows_;
                segment->tokens.assign(prompt.begin() + static_cast<std::ptrdiff_t>(k * block_rows_),
                    prompt.begin() + static_cast<std::ptrdiff_t>((k + 1U) * block_rows_));
                segment->payload.bytes = segment_bytes(k);
                checkpoint->segments.push_back(segment);
                pending.push_back(std::move(segment));
            }
            if (!store_->policy().legacy) {
                auto credits = store_->reserve(store_->incremental_bytes(*checkpoint), checkpoint.get());
                if (!credits) return;
                reservation = std::move(credits);
            }
            check(hipDeviceSynchronize(), "prefix capture fence");
            std::uint64_t captured_bytes = 0;
            for (const auto& segment : pending) {
                const auto k = segment->block_index;
                segment->payload = store_->allocate(segment->payload.bytes, reservation, true);
                std::uint64_t offset = 0;
                for (const auto& row : rows_) {
                    const auto bytes = row.bytes * row_count(row, k);
                    if (bytes)
                        check(hipMemcpyAsync(segment->payload.data.get() + offset,
                            reinterpret_cast<const void*>(row.address + row_first(row, k) * row.bytes),
                            static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost, stream_), "prefix capture rows");
                    offset += bytes;
                }
                captured_bytes += segment->payload.bytes;
            }
            checkpoint->fixed = store_->allocate(fixed_bytes_, reservation);
            for (const auto& fixed : fixed_) {
                if (fixed.device)
                    check(hipMemcpyAsync(checkpoint->fixed.data.get() + fixed.offset,
                        reinterpret_cast<const void*>(fixed.address), static_cast<std::size_t>(fixed.bytes),
                        hipMemcpyDeviceToHost, stream_), "prefix capture fixed state");
                else
                    std::memcpy(checkpoint->fixed.data.get() + fixed.offset,
                        reinterpret_cast<const void*>(fixed.address), static_cast<std::size_t>(fixed.bytes));
            }
            const auto completion = hipStreamSynchronize(stream_);
            if (completion != hipSuccess) PrefixDmaQuarantine::retain(dma_owner);
            check(completion, "prefix capture completion");
            segments_ = checkpoint->segments;
            request_->captured_segment_bytes += captured_bytes;
            request_->captured_fixed_bytes += fixed_bytes_;
            used_ = true;
            if (store_->insert(std::move(checkpoint), &reservation)) ++request_->captured_checkpoints;
            request_->capture_ms += ms(start);
        } catch (const PrefixCacheAllocationFailure&) {
            if (store_->policy().legacy) { drain(); throw; }
            fail();
        } catch (...) {
            // Drain before stack-owned copy destinations are released; retain fatal error contracts.
            drain();
            throw;
        }
    }

    void enable_fixed_mtp_sessions() {
        if (!identity_extension_) throw std::logic_error("MTP session registry lacks predictor state");
        fixed_mtp_sessions_ = true;
    }

    // Complete live-state capture: no prefill, no producer substitution, no boundary rounding.
    // Fixed MTP sessions normalize transient ticket/controller state at every committed turn.
    SessionCheckpoint capture_session(const SerialSessionContext& session, const SessionCompatibility& compatibility,
        SessionFrontier frontier) {
        if (compatibility.mtp != fixed_mtp_sessions_ || (identity_extension_ && !fixed_mtp_sessions_))
            throw std::runtime_error("session checkpoint refused: MTP session mode/registry mismatch");
        if (session.health != SerialSessionHealth::ready || session.arithmetic_lineage.tainted || imported_ ||
            !session.next_position || session.next_position >= layout_.context_tokens || session.next_position != session.committed_tokens ||
            frontier.consumed != session.next_position || frontier.tokens.size() != session.next_position ||
            frontier.predictor != (fixed_mtp_sessions_ ? session.next_position - 1U : 0U) ||
            !arithmetic_lineage_valid(session.arithmetic_lineage, session.next_position))
            throw std::runtime_error("session checkpoint refused: not an exact committed live frontier");
        check(hipDeviceSynchronize(), "session capture fence");
        SessionCheckpoint out{compatibility, std::move(frontier), {}};
        const auto copy = [&](const std::string& name, std::uintptr_t address, std::uint64_t bytes, bool device) {
            auto payload = hip_pinned_prefix_buffer(bytes);
            if (bytes) {
                if (device) check(hipMemcpy(payload.data.get(), reinterpret_cast<const void*>(address),
                    static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost), "session capture component");
                else std::memcpy(payload.data.get(), reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes));
            }
            out.sections.push_back({name, std::move(payload)});
        };
        for (const auto& row : rows_) {
            const auto count = session.next_position > row.shift ? session.next_position - row.shift : 0U;
            copy(row.id, row.address, count * row.bytes, true);
        }
        for (const auto& fixed : fixed_) copy(fixed.id, fixed.address, fixed.bytes, fixed.device);
        const auto metadata = encode_metadata(session);
        copy("session.metadata", reinterpret_cast<std::uintptr_t>(metadata.data()), metadata.size(), false);
        return out;
    }
    void restore_session(SerialSessionContext& session, const SessionCheckpoint& checkpoint, bool validate_only = false) {
        if (checkpoint.compatibility.mtp != fixed_mtp_sessions_ || (identity_extension_ && !fixed_mtp_sessions_))
            throw std::runtime_error("session restore refused: MTP controller serialization unsupported");
        if (session.next_position || session.committed_tokens || session.health != SerialSessionHealth::ready)
            throw std::runtime_error("session restore requires a fresh ready slot");
        const auto p = checkpoint.frontier.consumed;
        if (!p || p >= layout_.context_tokens || checkpoint.frontier.sampling != "greedy" ||
            checkpoint.frontier.predictor != (fixed_mtp_sessions_ ? p - 1U : 0U) ||
            checkpoint.frontier.tokens.size() != p ||
            checkpoint.sections.size() != rows_.size() + fixed_.size() + 1)
            throw std::runtime_error("session restore frontier or component count differs");
        const auto& last = checkpoint.sections.back();
        if (last.name != "session.metadata") throw std::runtime_error("session restore metadata missing");
        const auto meta = decode_metadata(last.payload.span());
        if (meta.position != p || meta.committed_tokens != p || meta.layout_identity != layout_.layout_identity ||
            meta.row_bytes != row_bytes_ || meta.fixed_bytes != fixed_bytes_ ||
            meta.execution_identity != session.execution_identity ||
            meta.runtime_compatibility_identity != session.runtime_compatibility_identity ||
            meta.lineage.tainted || !arithmetic_lineage_valid(meta.lineage, p))
            throw std::runtime_error("session restore geometry or producer lineage differs");
        std::size_t at = 0;
        for (const auto& row : rows_) {
            const auto count = p > row.shift ? p - row.shift : 0U;
            const auto& s = checkpoint.sections[at++];
            if (s.name != row.id || s.payload.bytes != count * row.bytes)
                throw std::runtime_error("session restore row component differs");
        }
        for (const auto& fixed : fixed_) {
            const auto& s = checkpoint.sections[at++];
            if (s.name != fixed.id || s.payload.bytes != fixed.bytes)
                throw std::runtime_error("session restore fixed component differs");
        }
        // Validation is complete before the first device or host-state write.
        if (validate_only) return;
        check(hipDeviceSynchronize(), "session restore fence");
        try {
            at = 0;
            for (const auto& row : rows_) {
                const auto& s = checkpoint.sections[at++];
                if (s.payload.bytes) check(hipMemcpy(reinterpret_cast<void*>(row.address), s.payload.data.get(),
                    static_cast<std::size_t>(s.payload.bytes), hipMemcpyHostToDevice), "session restore rows");
            }
            for (const auto& fixed : fixed_) {
                const auto& s = checkpoint.sections[at++];
                if (fixed.device) check(hipMemcpy(reinterpret_cast<void*>(fixed.address), s.payload.data.get(),
                    static_cast<std::size_t>(fixed.bytes), hipMemcpyHostToDevice), "session restore fixed");
                else std::memcpy(reinterpret_cast<void*>(fixed.address), s.payload.data.get(), static_cast<std::size_t>(fixed.bytes));
            }
            session.next_position = meta.position; session.committed_tokens = meta.committed_tokens;
            session.state = meta.state; session.arithmetic_lineage = meta.lineage;
        } catch (...) {
            session.health = SerialSessionHealth::poisoned; session.arithmetic_lineage.tainted = true; throw;
        }
        used_ = true;
    }

private:
    bool retain_ = true, policy_selection_ = false;
    std::string conversation_;
    // Restores a committed import at p = state.position() exactly like a checkpoint: rows [0, p - shift) of every
    // row component, the fixed blob, then the counters a run over p rows publishes (generation f advances once per
    // row and per serial-plan step writing f) and one honest prefill_chunk lineage interval [0, p) carrying the
    // producer's identities (cooperative local_rows q: the checkpoint of [0, q), then an interval [q, p)).
    std::uint64_t restore_rtx_state(SerialSessionContext& session, std::shared_ptr<const RtxStateStaging> state_owner) {
        const auto& state = *state_owner;
        const auto start = std::chrono::steady_clock::now();
        const auto& l = state.layout();
        const auto p = state.position();
        if (l.layout_identity != layout_.layout_identity || l.row_bytes != row_bytes_ || l.fixed_bytes != fixed_bytes_ ||
            l.rows.size() != rows_.size() || l.fixed.size() != fixed_.size() || !rtx_facts_ || p >= layout_.context_tokens ||
            session.execution_identity != rtx_fresh_execution_ || session.runtime_compatibility_identity != rtx_fresh_compatibility_)
            throw std::runtime_error("rtx state import does not match the live session geometry");
        for (std::size_t c = 0; c < rows_.size(); ++c)
            if (l.rows[c].bytes_per_row != rows_[c].bytes || l.rows[c].shift != rows_[c].shift)
                throw std::runtime_error("rtx state import row component differs");
        for (std::size_t f = 0; f < fixed_.size(); ++f)
            if (l.fixed[f].bytes != fixed_[f].bytes || l.fixed[f].offset != fixed_[f].offset)
                throw std::runtime_error("rtx state import fixed component differs");
        PersistentStateGenerations generations = session.state;
        generations.valid = generations.valid | rtx_facts_->valid_union;
        for (std::size_t f = 0; f < generations.generations.size(); ++f) {
            const auto per_row = rtx_facts_->generations_per_row[f];
            if (per_row && p > (UINT64_MAX - generations.generations[f]) / per_row)
                throw std::overflow_error("rtx state import generation overflows");
            generations.generations[f] += p * per_row;
        }
        const auto& producer = state.request().producer;
        // Cooperative (local_rows q > 0): rows [0, q) and the facts of prefix [0, q) come from this node's own
        // checkpoint of prompt[0, q); the import adds the rows [q, p) and the fixed state of position p.
        const auto q = state.request().local_rows;
        std::uint64_t local_bytes = 0;
        if (q) {
            if (!store_) throw std::runtime_error("rtx state import with local_rows needs the prefix state cache (--prefix-cache-host-bytes)");
            const auto checkpoint = store_->find_longest(identity_, state.request().tokens, block_rows_, q, retain_);
            if (!checkpoint || checkpoint->position != q)
                throw std::runtime_error("rtx state import local_rows " + std::to_string(q) + ": this node holds no checkpoint of prompt[0, " +
                    std::to_string(q) + ")");
            const auto meta = restore_checkpoint(session, *checkpoint);
            if (meta.position != q || meta.committed_tokens != q) {
                session.health = SerialSessionHealth::poisoned;
                throw std::runtime_error("rtx state import local checkpoint facts differ");
            }
            for (const auto& segment : checkpoint->segments) local_bytes += segment->payload.bytes;
            generations = meta.state;
            generations.valid = generations.valid | rtx_facts_->valid_union;
            for (std::size_t f = 0; f < generations.generations.size(); ++f) {
                const auto per_row = rtx_facts_->generations_per_row[f];
                if (per_row && p - q > (UINT64_MAX - generations.generations[f]) / per_row)
                    throw std::overflow_error("rtx state import generation overflows");
                generations.generations[f] += (p - q) * per_row;
            }
            session.arithmetic_lineage = meta.lineage;
        }
        auto lineage = prepare_prefill_chunk_lineage(session.arithmetic_lineage, q, p, producer.numerical, producer.reduction,
            producer.state_update);
        auto import_owner = std::make_unique<PrefixDmaQuarantine>();
        import_owner->imported_source = std::move(state_owner);
        try {
            for (std::size_t c = 0; c < rows_.size(); ++c) {
                const auto rows = state.component(c);
                const auto skip = static_cast<std::size_t>(rows_[c].bytes * l.component_rows(c, q));
                if (rows.size() > skip)
                    check(hipMemcpyAsync(reinterpret_cast<void*>(rows_[c].address + skip), rows.data() + skip, rows.size() - skip,
                        hipMemcpyHostToDevice, stream_), "rtx import rows");
            }
            const auto blob = state.fixed_blob();
            for (const auto& fixed : fixed_) {
                if (fixed.device)
                    check(hipMemcpyAsync(reinterpret_cast<void*>(fixed.address), blob.data() + fixed.offset, static_cast<std::size_t>(fixed.bytes),
                        hipMemcpyHostToDevice, stream_), "rtx import fixed state");
                else
                    std::memcpy(reinterpret_cast<void*>(fixed.address), blob.data() + fixed.offset, static_cast<std::size_t>(fixed.bytes));
            }
            const auto completion = hipStreamSynchronize(stream_);
            if (completion != hipSuccess) PrefixDmaQuarantine::retain(import_owner);
            check(completion, "rtx import completion");
        } catch (...) {
            // The device state may be partially overwritten: the session cannot be used.
            session.health = SerialSessionHealth::poisoned;
            session.arithmetic_lineage.tainted = true;
            const auto drain_error = hipStreamSynchronize(stream_);
            if (drain_error != hipSuccess) PrefixDmaQuarantine::retain(import_owner);
            check(drain_error, "rtx import failure drain");
            throw;
        }
        session.next_position = p;
        session.committed_tokens = p;
        session.state = generations;
        session.arithmetic_lineage = std::move(lineage);
        if (!arithmetic_lineage_valid(session.arithmetic_lineage, session.next_position)) {
            session.health = SerialSessionHealth::poisoned;
            throw std::runtime_error("rtx state import produced an invalid arithmetic lineage");
        }
        if (restore_hook_) restore_hook_(p);
        imported_ = true;
        request_->restored_position = p;
        request_->restored_bytes = state.payload_bytes();
        request_->restore_ms = ms(start);
        request_->rtx_import = true;
        request_->rtx_sha_ok = state.committed();  // committed = coverage + per-chunk digests + manifest verified
        request_->rtx_position = p;
        request_->rtx_bytes = state.payload_bytes();
        request_->rtx_local_rows = q;
        request_->rtx_local_bytes = local_bytes;
        request_->rtx_chunks = state.chunks();
        request_->rtx_restore_ms = request_->restore_ms;
        request_->rtx_verify_ms = state.commit_ms();
        request_->rtx_id = state.id();
        request_->rtx_manifest_sha256 = model_content_identity_hex(state.manifest_sha256());
        return p;
    }

    struct Row { std::uintptr_t address = 0; std::uint64_t bytes = 0; std::uint32_t shift = 0; std::string id; };
    // Row j of a component with shift s belongs to block floor((j + s) / T).
    std::uint64_t row_first(const Row& row, std::uint64_t k) const {
        const auto first = k * block_rows_;
        return first > row.shift ? first - row.shift : 0U;
    }
    std::uint64_t row_count(const Row& row, std::uint64_t k) const {
        return (k + 1U) * block_rows_ - row.shift - row_first(row, k);
    }
    void require_unused() const {
        if (used_) throw std::logic_error("prefix state cache extensions must precede the first request");
    }
    struct Fixed { std::uintptr_t address = 0; std::uint64_t bytes = 0, offset = 0; bool device = true; std::string id; };
    struct Metadata {
        std::uint64_t position = 0, committed_tokens = 0, layout_identity = 0, row_bytes = 0, fixed_bytes = 0;
        std::uint64_t execution_identity = 0, runtime_compatibility_identity = 0;
        PersistentStateGenerations state;
        ArithmeticLineage lineage;
    };
    static constexpr std::uint64_t metadata_magic = 0x31544d5043585054ULL;

    static void check(hipError_t error, const char* what) {
        if (error != hipSuccess) throw PrefixCaptureHipError(error, what);
    }
    static double ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    std::vector<std::byte> encode_metadata(const SerialSessionContext& session) const {
        std::vector<std::byte> out;
        const auto u64 = [&](std::uint64_t value) {
            for (unsigned shift = 0; shift < 64; shift += 8) out.push_back(static_cast<std::byte>((value >> shift) & 255U));
        };
        u64(metadata_magic);
        u64(session.next_position); u64(session.committed_tokens); u64(layout_.layout_identity);
        u64(row_bytes_); u64(fixed_bytes_); u64(session.execution_identity); u64(session.runtime_compatibility_identity);
        u64(static_cast<std::uint64_t>(session.state.valid));
        for (const auto generation : session.state.generations) u64(generation);
        const auto& lineage = session.arithmetic_lineage;
        u64(lineage.history_identity); u64(lineage.tainted ? 1U : 0U); u64(lineage.intervals.size());
        for (const auto& interval : lineage.intervals) {
            u64(interval.begin); u64(interval.end); u64(static_cast<std::uint64_t>(interval.producer));
            u64(interval.numerical_plan_identity); u64(interval.reduction_plan_identity); u64(interval.state_update_identity);
        }
        return out;
    }
    static Metadata decode_metadata(std::span<const std::byte> in) {
        std::size_t at = 0;
        const auto u64 = [&]() {
            if (at + 8U > in.size()) throw std::runtime_error("prefix state cache metadata is truncated");
            std::uint64_t value = 0;
            for (unsigned shift = 0; shift < 64; shift += 8)
                value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(in[at++])) << shift;
            return value;
        };
        if (u64() != metadata_magic) throw std::runtime_error("prefix state cache metadata has another format");
        Metadata m;
        m.position = u64(); m.committed_tokens = u64(); m.layout_identity = u64(); m.row_bytes = u64(); m.fixed_bytes = u64();
        m.execution_identity = u64(); m.runtime_compatibility_identity = u64();
        m.state.valid = static_cast<PersistentState>(u64());
        for (auto& generation : m.state.generations) generation = u64();
        m.lineage.history_identity = u64();
        m.lineage.tainted = u64() != 0U;
        const auto count = u64();
        if (count > in.size() / 48U) throw std::runtime_error("prefix state cache lineage is malformed");
        for (std::uint64_t i = 0; i < count; ++i) {
            ArithmeticRowInterval interval;
            interval.begin = u64(); interval.end = u64(); interval.producer = static_cast<ArithmeticProducer>(u64());
            interval.numerical_plan_identity = u64(); interval.reduction_plan_identity = u64(); interval.state_update_identity = u64();
            m.lineage.intervals.push_back(interval);
        }
        if (at != in.size()) throw std::runtime_error("prefix state cache metadata has trailing bytes");
        return m;
    }

    bool certified_metadata(const PersistentStateGenerations& state,const ArithmeticLineage& lineage,
        std::uint64_t position) const {
        if(!plan_.rows || !arithmetic_lineage_valid(lineage,position) ||
            static_cast<std::uint32_t>(state.valid)!=plan_.valid_state || lineage.intervals.size()!=position/plan_.rows) return false;
        for(std::size_t f=0;f<state.generations.size();++f) {
            if(plan_.generations_per_row[f] && position>(UINT64_MAX-plan_.initial_generations[f])/plan_.generations_per_row[f]) return false;
            if(state.generations[f]!=plan_.initial_generations[f]+position*plan_.generations_per_row[f]) return false;
        }
        for(std::size_t k=0;k<lineage.intervals.size();++k) {
            const auto& i=lineage.intervals[k];
            if(i.producer!=ArithmeticProducer::prefill_chunk || i.begin!=k*plan_.rows || i.end!=(k+1U)*plan_.rows ||
                i.numerical_plan_identity!=plan_.block.execution || i.reduction_plan_identity!=plan_.block.kernel ||
                i.state_update_identity!=plan_.block.execution) return false;
        }
        return true;
    }

    Metadata preflight_checkpoint(const SerialSessionContext& session,const PrefixCacheCheckpoint& c) const {
        const auto m=decode_metadata(c.metadata);
        if(c.position>layout_.context_tokens || m.position!=c.position || m.committed_tokens!=c.position || !certified_metadata(m.state,m.lineage,c.position) || m.lineage.tainted ||
            !arithmetic_lineage_valid(m.lineage,c.position) || m.layout_identity!=layout_.layout_identity ||
            m.row_bytes!=row_bytes_ || m.fixed_bytes!=fixed_bytes_ || c.fixed.bytes!=fixed_bytes_ ||
            (c.fixed.bytes && !c.fixed.data) || m.execution_identity!=session.execution_identity ||
            m.runtime_compatibility_identity!=session.runtime_compatibility_identity ||
            c.segments.size()!=c.position/block_rows_)
            throw std::runtime_error("prefix certified checkpoint metadata differs");
        auto parent=c.identity;
        for(std::size_t k=0;k<c.segments.size();++k) {
            const auto& seg=c.segments[k];
            if(!seg || seg->identity!=c.identity || seg->parent!=parent || seg->block_index!=k ||
                seg->block_rows!=block_rows_ || seg->payload.bytes!=segment_bytes(k) ||
                (seg->payload.bytes && !seg->payload.data) || seg->tokens.size()!=block_rows_ ||
                seg->digest!=prefix_cache_block_digest(parent,seg->tokens))
                throw std::runtime_error("prefix certified checkpoint section differs");
            parent=seg->digest;
        }
        if(parent!=c.digest) throw std::runtime_error("prefix certified checkpoint chain differs");
        return m;
    }

    // Copies a store checkpoint's rows and fixed state into the session's device buffers (synchronized) after
    // checking it against the live geometry; the caller sets the session facts. Throws (poisoning the session
    // once a copy started) on a mismatch or a failed copy.
    Metadata restore_checkpoint(SerialSessionContext& session, const PrefixCacheCheckpoint& checkpoint_ref) {
        const auto* checkpoint = &checkpoint_ref;
        const auto meta = decode_metadata(checkpoint->metadata);
        if (meta.position != checkpoint->position || meta.layout_identity != layout_.layout_identity ||
            meta.row_bytes != row_bytes_ || meta.fixed_bytes != fixed_bytes_ || checkpoint->fixed.bytes != fixed_bytes_ ||
            meta.execution_identity != session.execution_identity ||
            meta.runtime_compatibility_identity != session.runtime_compatibility_identity)
            throw std::runtime_error("prefix state cache checkpoint does not match the live session geometry");
        for (const auto& segment : checkpoint->segments)
            if (segment->payload.bytes != segment_bytes(segment->block_index))
                throw std::runtime_error("prefix state cache segment size differs");
        auto restore_owner = std::make_unique<PrefixDmaQuarantine>();
        restore_owner->checkpoint = std::make_shared<PrefixCacheCheckpoint>(checkpoint_ref);
        PrefixAsyncCapture::before_writer(capture_tag_,stream_);
        try {
            for (const auto& segment : checkpoint->segments) {
                std::uint64_t offset = 0;
                for (const auto& row : rows_) {
                    const auto bytes = row.bytes * row_count(row, segment->block_index);
                    if (bytes)
                        check(hipMemcpyAsync(reinterpret_cast<void*>(row.address + row_first(row, segment->block_index) * row.bytes),
                            segment->payload.data.get() + offset, static_cast<std::size_t>(bytes), hipMemcpyHostToDevice, stream_),
                            "prefix restore rows");
                    offset += bytes;
                }
                request_->restored_bytes += segment->payload.bytes;
            }
            for (const auto& fixed : fixed_) {
                if (fixed.device)
                    check(hipMemcpyAsync(reinterpret_cast<void*>(fixed.address), checkpoint->fixed.data.get() + fixed.offset,
                        static_cast<std::size_t>(fixed.bytes), hipMemcpyHostToDevice, stream_), "prefix restore fixed state");
                else
                    std::memcpy(reinterpret_cast<void*>(fixed.address), checkpoint->fixed.data.get() + fixed.offset,
                        static_cast<std::size_t>(fixed.bytes));
            }
            request_->restored_bytes += checkpoint->fixed.bytes;
            const auto completion = hipStreamSynchronize(stream_);
            if (completion != hipSuccess) PrefixDmaQuarantine::retain(restore_owner);
            check(completion, "prefix restore completion");
        } catch (...) {
            // The device state may be partially overwritten: the session cannot be used.
            session.health = SerialSessionHealth::poisoned;
            session.arithmetic_lineage.tainted = true;
            const auto drain_error = hipStreamSynchronize(stream_);
            if (drain_error != hipSuccess) PrefixDmaQuarantine::retain(restore_owner);
            check(drain_error, "prefix restore failure drain");
            throw;
        }
        return meta;
    }

    std::shared_ptr<PrefixStateCache> store_;
    ModelContentIdentity identity_;
    ModelContentIdentity plan_identity_;
    QwenTiledPlanConfig plan_;
    std::optional<QwenPlanCursor> continuation_;
    std::uint64_t plan_n_=0;
    const void* capture_tag_=nullptr;
    std::uint64_t plan_route_=1;
    QwenBatchConfig batch_config_;
    bool allow_restore_=true,nesting_possible_=false;
    bool certified_path_=false,plan_enabled_=false,serial_path_=false;
    std::set<std::uint64_t> requested_captures_;
    std::vector<KnownPromptChunkUnit> serial_cold_,serial_executed_;
    std::function<bool(std::uint64_t)> serial_can_resume_;
    std::uint32_t block_rows_ = 0;
    std::uint64_t every_ = 0;
    bool fixed_mtp_sessions_ = false;
    const SerialSessionStateLayout& layout_;
    const SessionStateMemoryBinding& memory_;
    const RotaryIdentity rotary_;
    std::size_t target_rows_=0,target_fixed_=0;
    std::vector<Row> rows_;
    std::vector<Fixed> fixed_;
    std::uint64_t row_bytes_ = 0, fixed_bytes_ = 0;
    std::vector<std::shared_ptr<const PrefixCacheSegment>> segments_;
    std::shared_ptr<PrefixAsyncCapture::RowChain> row_chain_=std::make_shared<PrefixAsyncCapture::RowChain>();
    std::function<void(std::uint64_t)> restore_hook_;
    std::function<void()> before_restore_hook_;
    bool used_ = false, imported_ = false;
    std::optional<ModelContentIdentity> identity_extension_;
    std::optional<RtxStateImportFacts> rtx_facts_;
    std::shared_ptr<RtxStateImportRegistry> rtx_registry_;
    std::uint64_t rtx_fresh_execution_ = 0, rtx_fresh_compatibility_ = 0;
    std::shared_ptr<PrefixStateCacheRequestStats> request_=std::make_shared<PrefixStateCacheRequestStats>();
    std::uint64_t staging_cap_ = 0;
    hipStream_t stream_ = nullptr;
};

// Digest of the running executable, so a disk tier never serves a state another build produced.
inline ModelContentIdentity prefix_state_cache_engine_build() {
    std::ifstream file("/proc/self/exe", std::ios::binary);
    if (!file) return {};
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return sha256_bytes(std::as_bytes(std::span<const char>(bytes)));
}

// Token IDs separated by commas or whitespace.
inline std::vector<std::uint32_t> read_prefix_cache_prompt(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("prefix cache prompt file cannot be read: " + path);
    std::stringstream text;
    text << file.rdbuf();
    auto content = text.str();
    std::replace(content.begin(), content.end(), ',', ' ');
    std::istringstream in(content);
    std::vector<std::uint32_t> tokens;
    unsigned long long value = 0;
    while (in >> value) {
        if (value > UINT32_MAX) throw std::invalid_argument("prefix cache prompt token exceeds uint32");
        tokens.push_back(static_cast<std::uint32_t>(value));
    }
    if (!in.eof()) throw std::invalid_argument("prefix cache prompt file has a non-numeric token: " + path);
    if (tokens.empty()) throw std::invalid_argument("prefix cache prompt file is empty: " + path);
    return tokens;
}

}  // namespace strix::decode_cli
