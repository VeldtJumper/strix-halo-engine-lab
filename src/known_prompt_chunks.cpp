#include "strix/known_prompt_chunks.hpp"
#include "strix/known_prefill_limits.hpp"
#include "strix/qpf2k_defaults.hpp"

#include "strix/serial_decode_replay.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <span>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
KnownPromptChunkCursor::KnownPromptChunkCursor(KnownPromptChunkRequest request,
    KnownPromptWindowBackend& windows, SerialGreedyStepBackend& serial)
    : request_(request), windows_(windows), serial_(serial) {
    // Reuse the existing complete-request validation without invoking a model.
    const SerialGreedyTokenCursor validated({request.prompt_tokens,request.vocab_size,
        request.context_capacity,request.max_new_tokens,request.eos_token_id});
    (void)validated;
    if(request.tiled_prefill) {
        if(request.logical_rows<16U || request.logical_rows>known_prefill_max_rows ||
            !windows.supports_tiled_prefill())
            throw std::invalid_argument("tiled prompt prefill requires an explicitly capable T"+known_prefill_rows_range()+" backend");
        if(request.output_policy==KnownWindowOutputPolicy::state_only && request.max_new_tokens)
            throw std::invalid_argument("state-only tiled prefill supports zero-emission requests only");
    } else if((request.logical_rows!=2U && request.logical_rows!=4U && request.logical_rows!=8U) ||
        windows.supports_tiled_prefill())
        throw std::invalid_argument("known prompt chunk rows must be closed2/4/8 or explicit tiled prefill");
    if(request.all_row_heads&&(!request.tiled_prefill||request.output_policy!=KnownWindowOutputPolicy::all_rows))
        throw std::invalid_argument("all-row heads require the tiled prefill with the all-rows output policy");
    if(windows.logical_rows()!=request.logical_rows)
        throw std::invalid_argument("known prompt chunk backend row width differs from the request");
    validate_known_window_output_policy(request.output_policy);
    if(windows.output_policy()!=request.output_policy)
        throw std::invalid_argument("known prompt chunk output policy differs from the backend");
    if(request.start_position>request.context_capacity ||
        request.prompt_tokens.size()>request.context_capacity-request.start_position)
        throw std::invalid_argument("known prompt chunks exceed the remaining context capacity");
    if(request.vocab_size>std::numeric_limits<std::size_t>::max()/sizeof(float))
        throw std::length_error("known prompt frontier exceeds host storage capacity");
    // New cursor-owned presentation storage exists before the first pair can
    // consume state. Returned backend vectors follow their own native contract.
    if(request.output_policy!=KnownWindowOutputPolicy::state_only || request.max_new_tokens)
        logits_.resize(static_cast<std::size_t>(request.vocab_size));
    progress_.next_position=request.start_position;
}

void KnownPromptChunkCursor::set_tail_backend(KnownPromptWindowBackend& tail) {
    const auto limit=tails_.empty()?request_.logical_rows:tails_.back()->logical_rows();
    if(progress_.consumed_prompt_tokens || progress_.failed || !request_.tiled_prefill || request_.all_row_heads ||
       !tail.supports_tiled_prefill() || tail.logical_rows()<16U || tail.logical_rows()>=limit ||
       tail.output_policy()!=request_.output_policy || request_.output_policy==KnownWindowOutputPolicy::state_only)
        throw std::invalid_argument("known prompt tail backend must be a smaller tiled backend set before the first chunk");
    tails_.push_back(&tail);
}

bool KnownPromptChunkCursor::advance() {
    if(progress_.failed) throw std::logic_error("failed known-prompt cursor cannot resume");
    if(progress_.cancelled || progress_.prefill_complete) {
        if(!planning_only_ && deferred_) {
            deferred_=false;
            try {windows_.finish_deferred();} catch(...) {progress_.failed=true;has_logits_=false;selection_.reset();throw;}
        }
        return false;
    }
    const auto consumed=static_cast<std::size_t>(progress_.consumed_prompt_tokens);
    const auto remaining=request_.prompt_tokens.size()-consumed;
    const bool state_only=request_.output_policy==KnownWindowOutputPolicy::state_only;
    const bool no_output=request_.max_new_tokens==0U;
    bool window=remaining>=request_.logical_rows&&(!state_only||no_output||remaining>request_.logical_rows);
    KnownPromptWindowBackend* backend=&windows_;
    std::uint32_t block=request_.logical_rows;
    bool tail=false;
    // STRIX_QPF2K_SCHED bit 16: balanced blocks. The prompt's rows down to a multiple of 128 run as k = ceil(remaining / T)
    // blocks of a multiple of 128 rows (at most T), each as the first rows of the T owner's arena, instead of T blocks and
    // the 1024/256 tails; the last remaining mod 128 rows then run as the ladder runs them (64/16 tails, serial rows).
    // Bit 4: the ladder's leading multiple-of-128 tail units merged into one partial block of the T owner.
    static const bool balanced_raw=[]{const char* v=std::getenv("STRIX_QPF2K_SCHED");return v&&(std::strtoul(v,nullptr,0)&16UL)!=0UL;}();
    static const bool merged_raw=[]{const char* v=std::getenv("STRIX_QPF2K_SCHED");return v&&(std::strtoul(v,nullptr,0)&4UL)!=0UL;}();
    const bool defaults_off=qpf2k::suppressed(qpf2k::Item::sched,windows_.gsq_model());
    const bool balanced_env=balanced_raw&&!defaults_off,merged_env=merged_raw&&!defaults_off;
    // Per prompt (decided at its first unit): only where the balanced plan is bitwise equal to the ladder.
    if(balanced_<0) {
        std::vector<std::uint32_t> ladder=request_.ladder_tail_rows;
        if(ladder.empty()) for(auto* t:tails_) ladder.push_back(t->logical_rows());
        balanced_=balanced_env&&known_prefill_balanced_eligible(remaining,request_.logical_rows,ladder)?1:0;
        if(balanced_env && !planning_only_) std::fprintf(stderr,"qpf2k sched balanced %s for %zu rows\n",balanced_?"on":"off (ladder differs)",remaining);
    }
    const bool balanced=balanced_==1&&request_.schedule_extensions&&request_.tiled_prefill&&!request_.all_row_heads&&!state_only&&
        windows_.supports_tiled_prefill()&&windows_.output_policy()!=KnownWindowOutputPolicy::all_rows&&
        known_prefill_balanced_rows(remaining,request_.logical_rows)!=0U;
    bool partial=false,merged_unit=false;
    if(balanced) {
        const auto size=known_prefill_balanced_rows(remaining,request_.logical_rows);
        window=true;tail=false;backend=&windows_;block=static_cast<std::uint32_t>(size);partial=block!=request_.logical_rows;
    }
    // Bit 4 (not balanced): merge the ladder's leading multiple-of-128 tail units into one partial block of the T owner.
    if(!window&&merged_env&&balanced_==0&&request_.schedule_extensions&&request_.tiled_prefill&&!request_.all_row_heads&&!state_only&&
       windows_.supports_tiled_prefill()&&windows_.output_policy()!=KnownWindowOutputPolicy::all_rows) {
        std::vector<std::uint32_t> ladder=request_.ladder_tail_rows;
        if(ladder.empty()) for(auto* t:tails_) ladder.push_back(t->logical_rows());
        if(const auto merged=known_prefill_merged_tail_rows(remaining,request_.logical_rows,ladder)) {
            window=true;tail=false;backend=&windows_;block=static_cast<std::uint32_t>(merged);partial=true;merged_unit=true;
        }
    }
    if(!window) for(auto* candidate:tails_) if(remaining>=candidate->logical_rows()) {
        tail=window=true;backend=candidate;block=candidate->logical_rows();break;
    }
    // STRIX_QPF2K_SCHED bit 8 fail-closed: with a pruned owner list (ladder_tail_rows set), rows the ladder runs in one unit
    // must not fall to a smaller tail owner or to the serial suffix (row classes sit on a 128-row grid per block: other
    // arithmetic, silently). The owners are a subset of the ladder, so a chosen tail owner is never larger than the unit.
    if(!request_.ladder_tail_rows.empty() && remaining<request_.logical_rows) {
        std::uint32_t want=0;
        for(const auto r:request_.ladder_tail_rows) if(r && remaining>=r) {want=r;break;}
        if(want && (!window || (tail && block<want)))
            throw std::logic_error("qpf2k sched: the ladder's "+std::to_string(want)+"-row unit has no tail owner (pruned)");
    }
    // Tiled prefill: a window with more prompt after it is never read, so it runs without its head.
    // STRIX_PREFILL_HEADLESS=0 keeps the head on every block.
    static const bool allow_headless=[]{const char* v=std::getenv("STRIX_PREFILL_HEADLESS");return !v||v[0]!='0';}();
    const bool headless=allow_headless&&!request_.all_row_heads&&request_.tiled_prefill&&!state_only&&window&&remaining>block;
    // A deferred block of windows_ is completed by the next block of windows_; anything else completes it first.
    if(!planning_only_ && deferred_ && (!window || backend!=&windows_)) {
        deferred_=false;
        try {windows_.finish_deferred();} catch(...) {progress_.failed=true;has_logits_=false;selection_.reset();throw;}
    }
    static const bool defer_env=[]{const char* v=std::getenv("STRIX_QPF2K_SCHED");return v&&(std::strtoul(v,nullptr,0)&2UL)!=0UL;}();
    const bool defer=headless&&defer_env&&!defaults_off&&request_.schedule_extensions&&request_.schedule_defer&&!tail&&backend==&windows_&&remaining-block>=(balanced?16U:request_.logical_rows);
    const auto rows=window?block:1U;
    const KnownPromptChunkUnit unit{progress_.next_position,progress_.next_position+rows,
        window?backend->logical_rows():1U,window&&!tail,balanced,merged_unit,defer,headless,state_only};
    // deferred_ stays set until the next block of windows_ (which completes it) succeeded: a failure leaves the
    // cleanup (cancel / destructor) its responsibility.
    try {
        if(!planning_only_) {
            if(headless) {
                // STRIX_QPF2K_SCHED bit 2: when the next unit is another block of this same backend (it completes this one
                // before publishing anything), this block ends without a host fence.
                if(partial) backend->execute_state_rows_partial(request_.prompt_tokens.subspan(consumed,block),progress_.next_position,defer);
                else if(defer) backend->execute_state_rows_deferred(request_.prompt_tokens.subspan(consumed,block),progress_.next_position);
                else backend->execute_state_rows(request_.prompt_tokens.subspan(consumed,block),progress_.next_position);
                deferred_=defer;
                has_logits_=false;selection_.reset();
            } else if(state_only&&(no_output||window||remaining>1U)) {
                if(window) {
                    windows_.execute_state_rows(request_.prompt_tokens.subspan(consumed,request_.logical_rows),progress_.next_position);deferred_=false;
                    ++progress_.state_only_windows;
                } else serial_.execute_state(request_.prompt_tokens[consumed],progress_.next_position);
                has_logits_=false;selection_.reset();
            } else {
                auto values=window ? (partial ? backend->execute_rows_partial(request_.prompt_tokens.subspan(consumed, block),progress_.next_position) :
                        backend->execute_rows(request_.prompt_tokens.subspan(consumed, block),
                        progress_.next_position)) : serial_.execute(request_.prompt_tokens[consumed],progress_.next_position);
                if(window) deferred_=false;
                validate_serial_decode_engine_logits(values,request_.vocab_size);
                // STRIX_PREFILL_LOGITS_DIGEST=1: the terminal row's FP32 logits FNV-1a 64 also when the prompt ends in a serial
                // suffix row (the tiled backend prints it for a block that ends the prompt).
                static const bool digest=[]{const char* v=std::getenv("STRIX_PREFILL_LOGITS_DIGEST");return v&&v[0]=='1';}();
                if(digest&&!window&&consumed+1U==request_.prompt_tokens.size()) {
                    std::uint64_t hash=14695981039346656037ULL;
                    for(const auto byte:std::as_bytes(std::span<const float>(values))){hash^=static_cast<unsigned char>(byte);hash*=1099511628211ULL;}
                    std::fprintf(stderr,"prefill terminal logits fnv64 %016llx position %llu\n",static_cast<unsigned long long>(hash),
                        static_cast<unsigned long long>(progress_.next_position));
                }
                std::copy(values.begin(),values.end(),logits_.begin());
                has_logits_=true;
            }
        }
        deferred_=defer;
        last_unit_=unit;
        progress_.consumed_prompt_tokens+=rows;
        progress_.next_position+=rows;
        if(window) ++progress_.windows_executed;
        else ++progress_.serial_tail_steps;
        if(tail) ++progress_.tail_windows;
        progress_.prefill_complete=progress_.consumed_prompt_tokens==request_.prompt_tokens.size();
        if(!planning_only_ && progress_.prefill_complete && has_logits_) selection_=select_greedy_token(logits_,request_.vocab_size);
        return true;
    } catch(...) {
        progress_.failed=true;has_logits_=false;selection_.reset();
        throw;
    }
}

std::vector<KnownPromptChunkUnit> KnownPromptChunkCursor::cold_plan() const {
    KnownPromptChunkCursor probe(request_,windows_,serial_);
    for(auto* tail:tails_) probe.set_tail_backend(*tail);
    probe.planning_only_=true;
    std::vector<KnownPromptChunkUnit> trace;
    while(probe.advance()) trace.push_back(probe.last_unit());
    return trace;
}
bool KnownPromptChunkCursor::can_resume_at(std::uint64_t boundary) const {
    if(!request_.tiled_prefill || request_.start_position || progress_.consumed_prompt_tokens || progress_.failed ||
        boundary>=request_.prompt_tokens.size()) return false;
    return known_prompt_full_t_prefix(cold_plan(),boundary,request_.logical_rows);
}
void KnownPromptChunkCursor::resume_at(std::uint64_t boundary) {
    if(!can_resume_at(boundary)) throw std::invalid_argument("serial-cursor-continuation-unsupported");
    planning_only_=true;
    while(progress_.next_position<boundary) (void)advance();
    planning_only_=false;
    deferred_=false;
    progress_.restored_prompt_tokens=boundary;
    progress_.windows_executed=progress_.serial_tail_steps=progress_.tail_windows=progress_.state_only_windows=0;
}

KnownPromptChunkCursor::~KnownPromptChunkCursor() {
    // Abandoned with a deferred block: check it (a failed check poisons the session in the backend).
    if(!planning_only_ && deferred_) {deferred_=false;try{windows_.finish_deferred();}catch(...){}}
}

void KnownPromptChunkCursor::cancel() noexcept {
    if(deferred_) {deferred_=false;try{windows_.finish_deferred();}catch(...){progress_.failed=true;}}
    if(!progress_.generation_complete && !progress_.failed) progress_.cancelled=true;
}

const SerialGreedyRequestResult& KnownPromptChunkCursor::finish_greedy() {
    if(progress_.failed || progress_.cancelled || !progress_.prefill_complete ||
        (!has_logits_ && request_.max_new_tokens) || generation_)
        throw std::logic_error("known-prompt generation requires a completed uncancelled fresh frontier");
    if(!has_logits_) {
        // Explicit state-only/maxnew0 consumed all known inputs without a head.
        // There is no selection or pending bonus to manufacture or replay.
        generation_.emplace();
        generation_->final_position=progress_.next_position;
        progress_.generation_complete=true;
        return *generation_;
    }
    const SerialGreedyContinuationRequest continuation{logits_,progress_.next_position,request_.vocab_size,
        request_.context_capacity,request_.max_new_tokens,request_.eos_token_id,request_.emission_hook};
    try {
        auto result=execute_serial_greedy_continuation(continuation,serial_);
        progress_.consumed_generation_tokens=result.steps_executed;
        progress_.next_position=result.final_position;
        progress_.emitted_tokens=result.generated_tokens.size();
        if(!result.generated_tokens.empty()) {
            progress_.bonus_pending=result.generated_tokens.back();
            selection_=progress_.bonus_pending;
        }
        if(result.steps_executed) has_logits_=false;
        generation_=std::move(result);
        progress_.generation_complete=true;
        return *generation_;
    } catch(...) {
        progress_.failed=true;has_logits_=false;selection_.reset();
        throw;
    }
}

KnownPromptChunkFrontier KnownPromptChunkCursor::frontier() const noexcept {
    KnownPromptChunkFrontier value;
    value.valid=!progress_.failed && (progress_.consumed_prompt_tokens || has_logits_ || selection_.has_value());
    value.state_position=progress_.next_position;
    if(progress_.consumed_prompt_tokens || progress_.consumed_generation_tokens)
        value.last_input_position=progress_.next_position-1U;
    if(has_logits_) value.logits=logits_;
    value.unconsumed_selection=selection_;
    value.selection_emitted=progress_.bonus_pending.has_value();
    return value;
}
} // namespace strix
