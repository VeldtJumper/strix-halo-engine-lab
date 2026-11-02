#include "strix/qpf2k_defaults.hpp"
#include "strix/hip_known_prompt_backend.hpp"
#include "strix/known_prefill_limits.hpp"

#include "strix/hip_known_prefix_window.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/serial_decode_replay.hpp"
#include "strix/serial_session.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace strix {
namespace {
void add_prompt_stat(std::uint64_t& total, std::uint64_t amount, bool& saturated) noexcept {
    if (amount > std::numeric_limits<std::uint64_t>::max() - total) {
        total = std::numeric_limits<std::uint64_t>::max();
        saturated = true;
    } else total += amount;
}
void check_prompt_copy(hipError_t status) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string("known prompt terminal logits copy: ") + hipGetErrorString(status));
}
} // namespace

struct HipKnownPromptBackend::Impl {
    const std::shared_ptr<HipKnownPrefixWindow> window;
    const std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    SerialSessionContext& session;
    const ExecutionMemoryBinding scratch;
    const std::uint64_t vocab_size;
    const std::uint64_t logits_bytes;
    bool failed = false;
    bool started = false;
    HipKnownPromptBackendStats observed;
    std::optional<std::pair<SessionExecutor, SessionExecutor>> handoff; // {decode, prefill}
    AllRowsSink all_rows_sink;
    std::function<void()> all_rows_drain;
    float* staging = nullptr; // pinned host, logical_rows x vocab FP32, only with all_rows_sink
    std::uint64_t staging_values = 0;

    Impl(std::shared_ptr<HipKnownPrefixWindow> native_window, SerialSessionContext& live_session,
        ExecutionMemoryBinding memory, std::uint64_t vocabulary,
        std::shared_ptr<const HipSetupQuiescenceGuard> original_guard)
        : window(std::move(native_window)), guard(std::move(original_guard)), session(live_session),
          scratch(memory), vocab_size(vocabulary), logits_bytes(checked_logits_bytes(vocabulary)) {
        if (!window || !guard || !scratch.execution_identity || !scratch.allocation_generation ||
            !scratch.arena_base || !scratch.arena_bytes || logits_bytes > scratch.arena_bytes ||
            scratch.arena_bytes > std::numeric_limits<std::uintptr_t>::max() - scratch.arena_base)
            throw std::invalid_argument("known prompt adapter requires its actual retained window arena and guard");
        if (current_hip_launch_context())
            throw std::invalid_argument("known prompt adapter construction cannot nest a HIP launch scope");
        guard->require_original();
    }

    ~Impl() {
        // No background reader may outlive the staging buffer, even after a failure.
        if (all_rows_drain) try { all_rows_drain(); } catch (...) {}
        if (staging) static_cast<void>(hipHostFree(staging));
    }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // All T rows of a completed block: one strided download (row views are row 0's plus r arena
    // strides) into the pinned staging buffer, then the sink, in row order.
    void deliver_all_rows(const KnownWindowResult& result, std::span<const std::uint32_t> tokens,
        std::uint64_t position, std::vector<float>& terminal) {
        const auto rows = static_cast<std::uint64_t>(tokens.size());
        if (result.logits.size() != rows || result.logit_row_indices.size() != rows || rows * vocab_size != staging_values)
            throw std::logic_error("known prompt all-rows block did not produce every row's logits");
        for (std::uint64_t r = 0; r < rows; ++r) {
            if (result.logit_row_indices[r] != r) throw std::logic_error("known prompt all-rows output order differs");
            validate_terminal(result.logits[r]);
        }
        const auto base = result.logits[0].address;
        const auto pitch = rows > 1U ? result.logits[1].address - base : logits_bytes;
        bool uniform = result.logits[1 % rows].address >= base && pitch >= logits_bytes;
        for (std::uint64_t r = 0; r < rows && uniform; ++r)
            uniform = result.logits[r].address == base + r * pitch;
        all_rows_drain(); // the previous block's readers of the staging buffer are done
        guard->require_original();
        const auto copy_start = std::chrono::steady_clock::now();
        if (uniform) {
            check_prompt_copy(hipMemcpy2D(staging, static_cast<std::size_t>(logits_bytes), reinterpret_cast<const void*>(base),
                static_cast<std::size_t>(pitch), static_cast<std::size_t>(logits_bytes), static_cast<std::size_t>(rows),
                hipMemcpyDeviceToHost));
        } else {
            for (std::uint64_t r = 0; r < rows; ++r)
                check_prompt_copy(hipMemcpy(staging + r * vocab_size, reinterpret_cast<const void*>(result.logits[r].address),
                    static_cast<std::size_t>(logits_bytes), hipMemcpyDeviceToHost));
        }
        guard->require_original();
        // STRIX_PREFILL_TIMING=1: host milliseconds of the block's logits download on stderr.
        static const bool timing = [] { const char* v = std::getenv("STRIX_PREFILL_TIMING"); return v && v[0] == '1'; }();
        if (timing)
            std::fprintf(stderr, "teacher block %llu rows: download %.1f ms\n", static_cast<unsigned long long>(rows),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copy_start).count());
        const std::span<const float> all(staging, static_cast<std::size_t>(staging_values));
        all_rows_sink(tokens, position, all);
        const auto last = all.subspan(static_cast<std::size_t>((rows - 1U) * vocab_size), static_cast<std::size_t>(vocab_size));
        std::copy(last.begin(), last.end(), terminal.begin());
        validate_serial_decode_engine_logits(terminal, vocab_size);
    }

    static std::uint64_t checked_logits_bytes(std::uint64_t vocabulary) {
        if (!vocabulary || vocabulary > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL ||
            vocabulary > std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::invalid_argument("known prompt adapter vocabulary does not fit a complete FP32 row");
        return vocabulary * sizeof(float);
    }

    void validate_terminal(const ExecutionBufferAddress& terminal) const {
        if (terminal.use.buffer.buffer_class != ExecutionBufferClass::logits ||
            !terminal.address || terminal.address % alignof(float) != 0U ||
            terminal.allocation_generation != scratch.allocation_generation || terminal.bytes < logits_bytes ||
            terminal.bytes > std::numeric_limits<std::uintptr_t>::max() - terminal.address ||
            terminal.address < scratch.arena_base || terminal.address - scratch.arena_base > scratch.arena_bytes ||
            terminal.bytes > scratch.arena_bytes - (terminal.address - scratch.arena_base))
            throw std::invalid_argument("known prompt terminal logits are outside the retained window arena");
    }

    std::vector<float> execute(std::span<const std::uint32_t> tokens, std::uint64_t position,bool state_only=false,bool defer=false,
        bool partial=false) {
        if (failed) throw std::logic_error("known prompt adapter cannot be reused after an exception");
        const auto committed_before = session.committed_tokens;
        // STRIX_QPF2K_SCHED bit 4: a partial block (the first tokens.size() rows of the tiled owner's T arena).
        const auto rows = partial ? static_cast<std::uint32_t>(tokens.size()) : window->logical_rows();
        bool window_attempted = false;
        started = true;
        try {
            if (partial && (!window->supports_tiled_prefill() || rows < 16U || rows % 8U || rows > window->logical_rows() ||
                    window->output_policy() == KnownWindowOutputPolicy::all_rows || all_rows_sink))
                throw std::invalid_argument("known prompt partial block needs 16..T rows in multiples of 8 on a terminal-row or state-only tiled owner");
            guard->require_original();
            const auto policy=window->output_policy();
            validate_known_window_output_policy(policy);
            // A tiled prefill backend that produces logits may still run a block whose logits are never read
            // (more prompt follows) without its head.
            const bool headless=state_only&&policy!=KnownWindowOutputPolicy::state_only&&window->supports_tiled_prefill();
            if((policy==KnownWindowOutputPolicy::state_only)!=state_only&&!headless)
                throw std::invalid_argument("known prompt output method differs from native window policy");
            if(all_rows_sink&&state_only)
                throw std::invalid_argument("known prompt all-rows delivery cannot run a block without its heads");
            if (current_hip_launch_context() || session.health != SerialSessionHealth::ready ||
                session.arithmetic_lineage.tainted || position != session.next_position ||
                tokens.size() != rows || (window->supports_tiled_prefill() ? (rows<16U||rows>known_prefill_max_rows) : (rows != 2U && rows != 4U && rows != 8U)) ||
                position > std::numeric_limits<std::uint64_t>::max() - rows ||
                committed_before > std::numeric_limits<std::uint64_t>::max() - rows)
                throw std::invalid_argument("known prompt window differs from the live eager session");
            for (const auto token : tokens)
                if (static_cast<std::uint64_t>(token) >= vocab_size)
                    throw std::invalid_argument("known prompt token exceeds vocabulary");
            // Allocate host return storage before any model-state consumption.
            std::vector<float> logits(state_only?0U:static_cast<std::size_t>(vocab_size));
            // STRIX_PREFILL_TIMING=1: host milliseconds of the hand-offs and the block (stderr).
            static const bool timing=[]{const char* v=std::getenv("STRIX_PREFILL_TIMING");return v&&v[0]=='1';}();
            auto mark=std::chrono::steady_clock::now();
            const auto lap=[&](const char* name) {
                if(!timing) return;
                const auto now=std::chrono::steady_clock::now();
                std::fprintf(stderr,"prefill block %-12s %8.2f ms\n",name,std::chrono::duration<double,std::milli>(now-mark).count());mark=now;
            };
            // STRIX_PREFILL_PIPE bit 8: the model binding is validated once for the hand-offs and the block (each
            // validation inside this scope is skipped for the same manifest/weights pair, which the scope validated).
            static const bool one_validation=[]{const char* v=std::getenv("STRIX_PREFILL_PIPE");return v&&(std::strtoul(v,nullptr,0)&8UL)!=0UL;}();
            std::optional<ValidatedModelBindingScope> binding_scope;
            if (one_validation && !qpf2k::suppressed(qpf2k::Item::pipe, window->gsq_model()) && handoff && handoff->first.resources && !handoff->first.resources->distributed_stage &&
                handoff->first.resources->model_manifest && handoff->first.resources->model_weights)
                binding_scope.emplace(*handoff->first.resources->model_manifest, *handoff->first.resources->model_weights);
            lap("scope");
            if (handoff) hand_off_serial_session(session, handoff->first, handoff->second);
            lap("handoff-in");
            window_attempted = true;
            // STRIX_QPF2K_SCHED bit 2: a deferred headless block (no fence; the next block of this owner checks it).
            const auto result = partial ? window->execute_partial(session, tokens, !state_only, defer) :
                headless ? (defer ? window->execute_without_head_deferred(session, tokens) :
                window->execute_without_head(session, tokens)) : window->execute(session, tokens);
            lap("execute");
            if (handoff) hand_off_serial_session(session, handoff->second, handoff->first);
            lap("handoff-out");
            binding_scope.reset();
            // A window with deferred GDN or per-row PLE keeps every row of a known prompt: commit them all
            // and apply the pending GDN rows now (both are no-ops for windows without those operators).
            if (!window->supports_tiled_prefill()) { window->commit_prefix(rows); window->flush_deferred(); }
            if (session.health != SerialSessionHealth::ready || session.next_position != position + rows ||
                session.committed_tokens != committed_before + rows || result.logits.size()!=result.logit_row_indices.size())
                throw std::logic_error("known prompt native window did not commit its configured causal rows");
            std::size_t outputs=0U;
            for(std::uint32_t r=0;r<rows&&!headless;++r)if(known_window_produces_logits(policy,r,rows)) {
                if(outputs>=result.logit_row_indices.size()||result.logit_row_indices[outputs]!=r)
                    throw std::logic_error("known prompt native output mapping differs from its policy");
                ++outputs;
            }
            if(outputs!=result.logits.size()||result.omitted_lm_heads!=rows-outputs)
                throw std::logic_error("known prompt native output count differs from its policy");
            // execute() has already fenced its owned producer stream and checked
            // D1 before publication. A synchronous full-row download introduces
            // no borrowed stream or additional D1 phase. Output intent was
            // compiled/admitted before execution; never read omitted scratch.
            if(!state_only && all_rows_sink && !headless) {
                deliver_all_rows(result, tokens, position, logits);
            } else if(!state_only) {
                const auto terminal=std::find(result.logit_row_indices.begin(),result.logit_row_indices.end(),rows-1U);
                if(terminal==result.logit_row_indices.end())throw std::logic_error("known prompt terminal output is missing");
                const auto& view=result.logits[static_cast<std::size_t>(terminal-result.logit_row_indices.begin())];
                validate_terminal(view);
                guard->require_original();
                check_prompt_copy(hipMemcpy(logits.data(), reinterpret_cast<const void*>(view.address),
                    static_cast<std::size_t>(logits_bytes), hipMemcpyDeviceToHost));
                guard->require_original();
                validate_serial_decode_engine_logits(logits, vocab_size);
                // STRIX_PREFILL_LOGITS_DIGEST=1: FNV-1a 64 of the terminal row's FP32 logit bytes (stderr), for
                // bitwise comparisons of prefill variants.
                static const bool digest=[]{const char* v=std::getenv("STRIX_PREFILL_LOGITS_DIGEST");return v&&v[0]=='1';}();
                if(digest) {
                    std::uint64_t hash=14695981039346656037ULL;
                    for(const auto byte:std::as_bytes(std::span<const float>(logits))){hash^=static_cast<unsigned char>(byte);hash*=1099511628211ULL;}
                    std::fprintf(stderr,"prefill terminal logits fnv64 %016llx position %llu\n",static_cast<unsigned long long>(hash),
                        static_cast<unsigned long long>(position+rows-1U));
                }
            }
            lap("output");
            // Publish only completed successful adapter returns, from the
            // actual native result. Saturation is telemetry, never admission.
            auto next = observed;
            add_prompt_stat(next.windows, 1U, next.saturated);
            add_prompt_stat(next.omitted_lm_heads,result.omitted_lm_heads,next.saturated);
            if(state_only)add_prompt_stat(next.state_only_windows,1U,next.saturated);
            add_prompt_stat(next.completed_input_rows, rows, next.saturated);
            add_prompt_stat(next.shared_rows_dispatches, result.shared_rows_dispatches, next.saturated);
            add_prompt_stat(next.gdn_rows_dispatches, result.gdn_rows_dispatches, next.saturated);
            add_prompt_stat(next.routed_expert_union_dispatches, result.routed_expert_union_dispatches, next.saturated);
            add_prompt_stat(next.r2_dispatches, result.r2_dispatches, next.saturated);
            add_prompt_stat(next.row_loop_dispatches, result.row_loop_dispatches, next.saturated);
            add_prompt_stat(next.logical_projection_weight_bytes_avoided,
                result.logical_projection_weight_bytes_avoided, next.saturated);
            add_prompt_stat(next.projection_launches_saved, result.projection_launches_saved, next.saturated);
            static_assert(std::tuple_size_v<decltype(result.r2_variant_dispatches)> ==
                std::tuple_size_v<decltype(next.r2_variant_dispatches)>);
            for (std::size_t i = 0; i < next.r2_variant_dispatches.size(); ++i)
                add_prompt_stat(next.r2_variant_dispatches[i], result.r2_variant_dispatches[i], next.saturated);
            observed = next;
            return logits;
        } catch(const KnownPrefillCancelled&) {
            throw;
        } catch (...) {
            failed = true;
            // STRIX_QPF2K_SCHED bit 2: a previous block of this owner still unchecked (any failure of this call, also
            // in admission) poisons the session and drains the owner's stream; nothing of it is published further.
            if (window->completion_pending()) {
                session.health = SerialSessionHealth::poisoned; session.arithmetic_lineage.tainted = true;
                try { window->abandon_pending_completion(); } catch (...) { std::terminate(); }
            }
            // A rejected window leaves a ready session: return it to the decode executor.
            if (handoff && session.health == SerialSessionHealth::ready && handoff->second.compiled &&
                session.execution_identity == handoff->second.compiled->execution_identity) {
                try { hand_off_serial_session(session, handoff->second, handoff->first); }
                catch (...) { session.health = SerialSessionHealth::poisoned; session.arithmetic_lineage.tainted = true; }
            }
            if (session.committed_tokens != committed_before) {
                session.health = SerialSessionHealth::poisoned;
                session.arithmetic_lineage.tainted = true;
            }
            // The native owner also contains its own failure paths. This drain
            // covers output-copy/validation failures after a successful commit.
            // Pure admission has submitted nothing. In particular, rejecting a
            // caller's active scope must not synchronize its live capture.
            if (window_attempted) guard->recover_or_fail_stop();
            throw;
        }
    }
};

HipKnownPromptBackend::HipKnownPromptBackend(std::shared_ptr<HipKnownPrefixWindow> window,
    SerialSessionContext& session, ExecutionMemoryBinding window_scratch, std::uint64_t vocab_size,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard)
    : impl_(std::make_unique<Impl>(std::move(window), session, window_scratch, vocab_size, std::move(guard))) {}
HipKnownPromptBackend::~HipKnownPromptBackend() = default;
std::vector<float> HipKnownPromptBackend::execute_window(std::array<std::uint32_t,2> tokens, std::uint64_t position) {
    return impl_->execute(tokens, position);
}
std::uint32_t HipKnownPromptBackend::logical_rows() const noexcept {
    return impl_->window->logical_rows();
}
bool HipKnownPromptBackend::supports_tiled_prefill() const noexcept {return impl_->window->supports_tiled_prefill();}
bool HipKnownPromptBackend::gsq_model() const noexcept {return impl_->window->gsq_model();}
KnownWindowOutputPolicy HipKnownPromptBackend::output_policy() const noexcept {return impl_->window->output_policy();}
std::vector<float> HipKnownPromptBackend::execute_rows(std::span<const std::uint32_t> tokens, std::uint64_t position) {
    return impl_->execute(tokens, position);
}
void HipKnownPromptBackend::execute_state_rows(std::span<const std::uint32_t> tokens,std::uint64_t position) {
    (void)impl_->execute(tokens,position,true);
}
void HipKnownPromptBackend::execute_state_rows_deferred(std::span<const std::uint32_t> tokens,std::uint64_t position) {
    if(!impl_->window->supports_tiled_prefill()||impl_->window->output_policy()==KnownWindowOutputPolicy::state_only) {
        (void)impl_->execute(tokens,position,true);return;
    }
    (void)impl_->execute(tokens,position,true,true);
}
std::vector<float> HipKnownPromptBackend::execute_rows_partial(std::span<const std::uint32_t> tokens, std::uint64_t position) {
    return impl_->execute(tokens,position,false,false,true);
}
void HipKnownPromptBackend::execute_state_rows_partial(std::span<const std::uint32_t> tokens, std::uint64_t position, bool defer) {
    (void)impl_->execute(tokens,position,true,defer && impl_->window->output_policy()!=KnownWindowOutputPolicy::state_only,true);
}
void HipKnownPromptBackend::finish_deferred() {
    if(!impl_->window->completion_pending()) return;
    try {impl_->window->finish_pending_completion();}
    catch(...) {impl_->failed=true;impl_->session.health=SerialSessionHealth::poisoned;impl_->session.arithmetic_lineage.tainted=true;throw;}
}
HipKnownPromptBackendStats HipKnownPromptBackend::stats() const noexcept {
    return impl_->observed;
}
void HipKnownPromptBackend::set_all_rows_sink(AllRowsSink sink, std::function<void()> drain) {
    auto& self = *impl_;
    if (self.all_rows_sink || self.started || self.failed || !sink || !drain || !self.window->supports_tiled_prefill() ||
        self.window->output_policy() != KnownWindowOutputPolicy::all_rows)
        throw std::invalid_argument("known prompt all-rows sink is set once, before any call, on a tiled all-rows prefill");
    const auto rows = static_cast<std::uint64_t>(self.window->logical_rows());
    if (!rows || self.vocab_size > std::numeric_limits<std::uint64_t>::max() / sizeof(float) / rows ||
        rows * self.vocab_size > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("known prompt all-rows staging size overflows");
    void* memory = nullptr;
    check_prompt_copy(hipHostMalloc(&memory, static_cast<std::size_t>(rows * self.vocab_size * sizeof(float)), hipHostMallocDefault));
    self.staging = static_cast<float*>(memory);
    self.staging_values = rows * self.vocab_size;
    self.all_rows_sink = std::move(sink);
    self.all_rows_drain = std::move(drain);
}
std::vector<std::vector<float>> HipKnownPromptBackend::execute_group(HipKnownPromptBackend& group,
    std::span<const GroupMember> members, bool stage_ple) {
    auto& lead = *group.impl_;
    if (members.empty() || !lead.window->supports_tiled_prefill() || current_hip_launch_context())
        throw std::invalid_argument("known prompt group requires members and a tiled prefill group owner");
    std::vector<Impl*> impls(members.size());
    std::vector<std::uint64_t> committed_before(members.size());
    std::vector<char> handed(members.size(), 0);
    std::vector<HipKnownPrefixWindow::PrefillSegmentRequest> requests(members.size());
    std::vector<std::vector<float>> logits(members.size());
    bool attempted = false;
    for (std::size_t m = 0; m < members.size(); ++m) {
        const auto& member = members[m];
        if (!member.backend) throw std::invalid_argument("known prompt group member has no backend");
        auto& self = *member.backend->impl_;
        impls[m] = &self;
        for (std::size_t o = 0; o < m; ++o)
            if (impls[o] == &self || &impls[o]->session == &self.session)
                throw std::invalid_argument("known prompt group members must be distinct sessions");
    }
    try {
        for (std::size_t m = 0; m < members.size(); ++m) {
            const auto& member = members[m];
            auto& self = *impls[m];
            if (self.failed) throw std::logic_error("known prompt adapter cannot be reused after an exception");
            self.started = true;
            self.guard->require_original();
            const auto rows = self.window->logical_rows();
            const auto policy = self.window->output_policy();
            validate_known_window_output_policy(policy);
            if (!self.window->supports_tiled_prefill() || self.all_rows_sink || policy == KnownWindowOutputPolicy::all_rows ||
                (member.logits && policy == KnownWindowOutputPolicy::state_only))
                throw std::invalid_argument("known prompt group members are terminal-row or state-only tiled blocks");
            auto& session = self.session;
            committed_before[m] = session.committed_tokens;
            if (session.health != SerialSessionHealth::ready || session.arithmetic_lineage.tainted ||
                member.position != session.next_position || member.tokens.size() != rows || rows < 16U || rows > known_prefill_max_rows ||
                member.position > std::numeric_limits<std::uint64_t>::max() - rows ||
                committed_before[m] > std::numeric_limits<std::uint64_t>::max() - rows)
                throw std::invalid_argument("known prompt group block differs from the live eager session");
            for (const auto token : member.tokens)
                if (static_cast<std::uint64_t>(token) >= self.vocab_size)
                    throw std::invalid_argument("known prompt token exceeds vocabulary");
            if (member.logits) logits[m].resize(static_cast<std::size_t>(self.vocab_size));
            requests[m] = {self.window.get(), &session, member.tokens, member.logits, member.observer};
        }
        for (std::size_t m = 0; m < members.size(); ++m)
            if (impls[m]->handoff) { hand_off_serial_session(impls[m]->session, impls[m]->handoff->first, impls[m]->handoff->second); handed[m] = 1; }
        attempted = true;
        const auto results = lead.window->execute_prefill_segments(requests,stage_ple);
        for (std::size_t m = 0; m < members.size(); ++m)
            if (handed[m]) { hand_off_serial_session(impls[m]->session, impls[m]->handoff->second, impls[m]->handoff->first); handed[m] = 0; }
        for (std::size_t m = 0; m < members.size(); ++m) {
            auto& self = *impls[m];
            const auto& result = results[m];
            const auto rows = static_cast<std::uint32_t>(members[m].tokens.size());
            if (self.session.health != SerialSessionHealth::ready || self.session.next_position != members[m].position + rows ||
                self.session.committed_tokens != committed_before[m] + rows || result.logits.size() != result.logit_row_indices.size() ||
                result.logits.size() != (members[m].logits ? 1U : 0U) || result.omitted_lm_heads != rows - result.logits.size() ||
                (members[m].logits && result.logit_row_indices[0] != rows - 1U))
                throw std::logic_error("known prompt group block did not commit its configured causal rows");
            if (members[m].logits) {
                // The segment's terminal row lies in the group owner's arena.
                lead.validate_terminal(result.logits[0]);
                lead.guard->require_original();
                check_prompt_copy(hipMemcpy(logits[m].data(), reinterpret_cast<const void*>(result.logits[0].address),
                    static_cast<std::size_t>(lead.logits_bytes), hipMemcpyDeviceToHost));
                lead.guard->require_original();
                validate_serial_decode_engine_logits(logits[m], self.vocab_size);
            }
            auto next = self.observed;
            add_prompt_stat(next.windows, 1U, next.saturated);
            add_prompt_stat(next.omitted_lm_heads, result.omitted_lm_heads, next.saturated);
            if (!members[m].logits) add_prompt_stat(next.state_only_windows, 1U, next.saturated);
            add_prompt_stat(next.completed_input_rows, rows, next.saturated);
            add_prompt_stat(next.shared_rows_dispatches, result.shared_rows_dispatches, next.saturated);
            add_prompt_stat(next.gdn_rows_dispatches, result.gdn_rows_dispatches, next.saturated);
            add_prompt_stat(next.routed_expert_union_dispatches, result.routed_expert_union_dispatches, next.saturated);
            add_prompt_stat(next.r2_dispatches, result.r2_dispatches, next.saturated);
            add_prompt_stat(next.row_loop_dispatches, result.row_loop_dispatches, next.saturated);
            for (std::size_t i = 0; i < next.r2_variant_dispatches.size(); ++i)
                add_prompt_stat(next.r2_variant_dispatches[i], result.r2_variant_dispatches[i], next.saturated);
            self.observed = next;
        }
        return logits;
    } catch(const KnownPrefillCancelled&) {
        throw;
    } catch (...) {
        for (std::size_t m = 0; m < members.size(); ++m) {
            auto& self = *impls[m];
            self.failed = true;
            auto& session = self.session;
            if (handed[m] && session.health == SerialSessionHealth::ready && session.execution_identity == self.handoff->second.compiled->execution_identity) {
                try { hand_off_serial_session(session, self.handoff->second, self.handoff->first); }
                catch (...) { session.health = SerialSessionHealth::poisoned; session.arithmetic_lineage.tainted = true; }
            }
            if (attempted && session.committed_tokens != committed_before[m]) {
                session.health = SerialSessionHealth::poisoned;
                session.arithmetic_lineage.tainted = true;
            }
        }
        if (attempted) lead.guard->recover_or_fail_stop();
        throw;
    }
}
void HipKnownPromptBackend::set_executor_handoff(SessionExecutor decode, SessionExecutor prefill) {
    if (impl_->handoff || impl_->started || impl_->failed || !decode.compiled || !prefill.compiled ||
        !decode.resources || !prefill.resources || !impl_->window->supports_tiled_prefill())
        throw std::invalid_argument("known prompt executor handoff is set once, before any tiled prefill call");
    validate_sibling_serial_executions(*prefill.compiled, *decode.compiled);
    impl_->handoff.emplace(decode, prefill);
}
std::uint64_t HipKnownPromptBackend::handoff_prefill_execution_identity() const noexcept {
    return impl_->handoff ? impl_->handoff->second.compiled->execution_identity : 0U;
}
} // namespace strix
