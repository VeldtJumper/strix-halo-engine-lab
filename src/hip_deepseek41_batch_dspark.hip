#include "strix/deepseek41_batch_dspark.hpp"

#include "strix/ds41_dspark_vocab.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::deepseek41 {
namespace {

namespace ds = ds41::dspark;
using clock_type = std::chrono::steady_clock;
double since(clock_type::time_point t) { return std::chrono::duration<double>(clock_type::now() - t).count(); }

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("deepseek41 batch dspark: ") + what + ": " + hipGetErrorString(e));
}

constexpr std::uint32_t stage_rows = 128;  // kept rows staged per slot before they go to its drafting context
// prompt lookup (the Nemotron selector's constants): shortest match, decay of the evidence, candidates a bucket needs,
// pending candidates per request, prompt tokens indexed per plan
constexpr std::uint32_t lookup_min_match = 2, lookup_evidence = 4, lookup_pending = 16, lookup_chunk = 1024;
constexpr double lookup_decay = 0.98;
std::uint32_t lookup_bucket(std::uint32_t length) { return length < 4U ? 1U : length < 8U ? 2U : length < 16U ? 3U : 4U; }  // 2-3, 4-7, 8-15, 16+

// world 2: the draft round, rank 0 -> rank 1, and rank 1's drafts back (one 512-byte exchange each)
struct DraftRound {
    std::uint32_t magic = BatchSpeculator::round_magic, done = 0, commits = 0, requests = 0;
    std::uint32_t coop = 0;  // 1: one request (slot[0]), drafted by both ranks together (no reply)
    // one entry per request (at most one commit and one draft per request and step), whatever the rows per step
    std::uint32_t commit[ds::BatchWindowController::max_requests]{};  // slot << 16 | kept
    std::uint32_t slot[ds::BatchWindowController::max_requests]{}, token[ds::BatchWindowController::max_requests]{};
};
struct DraftReply {
    std::uint32_t magic = 0x44524632U, count = 0;
    std::uint32_t slot[ds::BatchWindowController::max_requests]{};
    std::uint32_t tokens[ds::BatchWindowController::max_requests][ds::block_size]{};
    float confidence[ds::BatchWindowController::max_requests][ds::block_size]{};
};
constexpr std::size_t round_bytes = BatchSpeculator::round_bytes;
static_assert(sizeof(DraftRound) <= round_bytes && sizeof(DraftReply) <= round_bytes);

// sends `out`, returns what the peer sent as an In (each rank sends its half of the round in the same exchange)
template <class In, class Out>
In swap_packets(const std::function<void(std::span<const std::byte>, std::span<std::byte>)>& exchange, const Out& out) {
    std::array<std::byte, round_bytes> o{}, i{};
    std::memcpy(o.data(), &out, sizeof(Out));
    exchange(o, i);
    In in;
    std::memcpy(&in, i.data(), sizeof(In));
    return in;
}

// cost means: plain running mean for the first 16 samples, then an EMA (the costs drift with the context)
void accumulate(double& mean, std::uint32_t& n, double sample) {
    ++n;
    const double w = n <= 16U ? 1.0 / n : 1.0 / 16.0;
    mean += w * (sample - mean);
}

}  // namespace

struct BatchSpeculator::Slots {
    std::uint32_t count = 0;
    float* staging = nullptr;                          // [count][stage_rows][main_width]
    std::vector<std::uint32_t> staged;                 // rows staged per slot
    std::vector<std::vector<std::uint32_t>> context;   // committed tokens not yet in the slot's draft vocabulary
    ~Slots() {
        if (staging) (void)hipFree(staging);
    }
};

BatchSpeculator::BatchSpeculator(BatchForward& target, ds::DeviceDrafter& drafter, std::uint32_t slots, BatchSpecOptions options)
    : target_(target), drafter_(drafter), options_(std::move(options)), controller_(slots), slots_(std::make_unique<Slots>()) {
    if (options_.fixed_rows == 1U || options_.fixed_rows > ds::block_size + 1U)
        throw std::invalid_argument("deepseek41 batch dspark: a fixed window has 2..6 rows");
    if (options_.max_rows == 0U || options_.max_rows > BatchForward::max_rows || options_.prompt_rows == 0U)
        throw std::invalid_argument("deepseek41 batch dspark: rows per step");
    if (slots == 0U || slots > ds::BatchWindowController::max_requests) throw std::invalid_argument("deepseek41 batch dspark: 1..8 slots");
    if (options_.world == 0U || options_.world > 2U || options_.rank >= options_.world || (options_.world == 2U && !options_.exchange))
        throw std::invalid_argument("deepseek41 batch dspark: drafts on two ranks need rank, world 2 and an exchange");
    calib_ = options_.calib;
    if (const char* v = std::getenv("STRIX_DS41_DSPARK_CALIB"); v != nullptr) calib_ = std::strcmp(v, "0") != 0;
    if (options_.rank == 0U) std::fprintf(stderr, "dspark: cost calibration %s\n", calib_ ? "on" : "off (old estimator)");
    if (const char* v = std::getenv("STRIX_DS41_DSPARK_COSTLOG"); v != nullptr && std::strcmp(v, "1") == 0) cost_log_ = true;
    if (calib_) {
        // The measured acceptance prior (pseudo-observations, capped at 32 per cell; depth x confidence bin, bins
        // < -2, -2, -1, 0, 1, 2, 4, >= 8) from full-window DSpark runs of this model and drafter (held-out route statistics),
        // and the request's own evidence at weight 4 (design notes)
        static constexpr ds::BatchWindowController::BinTable prior_seen = {{{8, 32, 32, 32, 32, 32, 32, 32}, {0, 11, 32, 32, 32, 32, 32, 32},
                                                                           {0, 8, 32, 32, 32, 32, 32, 32}, {0, 9, 31, 32, 32, 32, 32, 32},
                                                                           {0, 4, 31, 32, 22, 32, 32, 31}}};
        static constexpr ds::BatchWindowController::BinTable prior_hits = {{{0, 4, 11, 18, 28, 31, 31, 32}, {0, 3, 16, 20, 29, 32, 32, 32},
                                                                           {0, 2, 14, 22, 28, 31, 32, 32}, {0, 3, 17, 28, 27, 32, 31, 32},
                                                                           {0, 1, 8, 30, 21, 28, 31, 31}}};
        controller_.seed(prior_seen, prior_hits);
        controller_.set_request_weight(4.0);
    }
    prompts_.resize(slots);
    positions_.assign(slots, 0U);
    kept_.fill(0xFFFFFFFFU);
    owners_.assign(slots, 0U);
    coop_ = options_.world == 2U && options_.coop;
    in_step_.assign(slots, false);
    coop_pending_.assign(slots, false);
    (void)target_.captured_streams();  // throws without capture_main_hidden
    drafter_.set_sequences(slots);
    auto& S = *slots_;
    S.count = slots;
    S.staged.assign(slots, 0U);
    lookup_on_ = options_.lookup && options_.rank == 0U;
    if (const char* v = std::getenv("STRIX_DS41_LOOKUP"); v != nullptr && v[0] == '0') lookup_on_ = false;
    if (lookup_on_ && options_.world == 2U && slots > 32U) {  // the step command carries lookup windows as a 32-slot mask
        lookup_on_ = false;
        std::fprintf(stderr, "dspark: prompt lookup off (%u slots; at most 32 with two ranks)\n", slots);
    }
    if (lookup_on_) {
        lookup_slots_.resize(slots);
        if (const char* v = std::getenv("STRIX_DS41_LOOKUP_MARGIN"); v != nullptr && *v) lookup_margin_ = std::clamp(std::atof(v), 1.0, 2.0);
        if (const char* v = std::getenv("STRIX_DS41_LOOKUP_LOG"); v != nullptr && v[0] == '1') lookup_log_ = true;
        if (const char* v = std::getenv("STRIX_DS41_LOOKUP_TOKENS"); v != nullptr && *v)
            lookup_capacity_ = static_cast<std::uint32_t>(std::clamp(std::atof(v), 64.0, 4194304.0));
    }
    S.context.resize(slots);
    check(hipMalloc(reinterpret_cast<void**>(&S.staging), std::size_t{slots} * stage_rows * ds::main_width * sizeof(float)), "staging");
}

BatchSpeculator::~BatchSpeculator() = default;

std::uint64_t BatchSpeculator::staging_bytes(std::uint32_t slots) noexcept {
    return std::uint64_t{slots} * stage_rows * ds::main_width * sizeof(float);
}

float* BatchSpeculator::staging(std::uint32_t slot) const {
    return slots_->staging + (std::size_t{slot} * stage_rows + slots_->staged[slot]) * ds::main_width;
}

void BatchSpeculator::flush(std::uint32_t slot) {
    auto& S = *slots_;
    if (S.staged[slot] == 0U && S.context[slot].empty()) return;
    drafter_.select(slot);
    if (S.staged[slot] != 0U)
        drafter_.append_hidden_device(S.staging + std::size_t{slot} * stage_rows * ds::main_width, S.staged[slot], target_.stream());
    S.staged[slot] = 0;
    if (!S.context[slot].empty()) (void)drafter_.add_vocabulary(S.context[slot]);
    S.context[slot].clear();
}

// rows [first, first + count) of the last step's captured inputs -> the slot's staging (then its drafting context)
void BatchSpeculator::stage(std::uint32_t slot, std::uint32_t first, std::uint32_t count) {
    stage_from(slot, target_.captured_streams(), first, count, target_.stream());
}

void BatchSpeculator::stage_from(std::uint32_t slot, std::array<const float*, 3> streams, std::uint32_t first, std::uint32_t count, void* stream) {
    auto& S = *slots_;
    if (count > stage_rows) throw std::invalid_argument("deepseek41 batch dspark: at most 128 rows staged at once");
    if (S.staged[slot] + count > stage_rows) flush(slot);
    const auto& c = target_.model().config;
    const std::size_t row = std::size_t{c.hc_mult} * c.hidden_size;
    ds::enqueue_main_hidden(streams[0] + first * row, streams[1] + first * row, streams[2] + first * row, count, staging(slot), stream);
    if (stream != target_.stream()) check(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "seed rows");  // flush() orders on the target's
    S.staged[slot] += count;
}

void BatchSpeculator::Moments::add(double tokens, double window_rows) {
    n = n * lookup_decay + 1.0;
    sum = sum * lookup_decay + tokens;
    rows = rows * lookup_decay + window_rows;
}


// a new request in `slot`: its prompt (the last capacity tokens: the index forgets older ones) is indexed a chunk per
// plan, so a long prompt does not hold the loop
void BatchSpeculator::lookup_reset(std::uint32_t slot, std::span<const std::uint32_t> prompt) {
    if (!lookup_on_) return;
    const auto t0 = clock_type::now();
    auto& L = lookup_slots_[slot];
    L = LookupSlot{};
    strix::lookup::IndexConfig config;
    config.capacity = lookup_capacity_;
    L.index = strix::lookup::make_ngram_index(config);
    const std::size_t keep = std::min<std::size_t>(prompt.size(), lookup_capacity_);
    L.backlog.assign(prompt.end() - static_cast<std::ptrdiff_t>(keep), prompt.end());
    L.next_position = prompt.size();
    stats_.lookup_seconds += since(t0);
}

void BatchSpeculator::lookup_catch_up(std::uint32_t slot) {
    auto& L = lookup_slots_[slot];
    const std::size_t end = std::min(L.backlog.size(), L.backlog_at + lookup_chunk);
    for (; L.backlog_at < end; ++L.backlog_at) L.index->append(L.backlog[L.backlog_at]);
    if (L.backlog_at == L.backlog.size()) L.backlog.clear(), L.backlog_at = 0;
}

// the token `slot` committed at stream `position`: scores the pending candidates waiting for this position, then joins
// the index (or its backlog). A position out of sequence (the state changed outside the planner) ends lookup for the
// request.
void BatchSpeculator::lookup_commit(std::uint32_t slot, std::uint32_t token, std::uint64_t position) {
    if (!lookup_on_) return;
    auto& L = lookup_slots_[slot];
    if (!L.index) return;
    const auto t0 = clock_type::now();
    if (position != L.next_position) {
        L = LookupSlot{};
        stats_.lookup_seconds += since(t0);
        return;
    }
    ++L.next_position;
    for (auto it = L.pending.begin(); it != L.pending.end();) {
        auto& c = *it;
        if (c.first + c.done != position) {
            ++it;
            continue;
        }
        L.eligible[c.bucket][c.done] += 1.0;
        const bool hit = c.token[c.done] == token;
        if (hit) L.accepted[c.bucket][c.done] += 1.0;
        ++c.done;
        if (!hit || c.done == c.count) it = L.pending.erase(it);
        else ++it;
    }
    if (!L.backlog.empty()) L.backlog.push_back(token);
    else L.index->append(token);
    stats_.lookup_seconds += since(t0);
}

void BatchSpeculator::start(std::uint32_t slot, std::span<const std::uint32_t> prompt) {
    auto& S = *slots_;
    if (slot >= S.count) throw std::invalid_argument("deepseek41 batch dspark: no such slot");
    drafter_.select(slot);
    drafter_.reset();
    drafter_.set_vocabulary(ds::choose_vocabulary(options_.vocabulary, prompt).vocabulary);
    S.staged[slot] = 0;
    S.context[slot].clear();
    controller_.reset(slot);
    if (++epoch_.at(slot) == 0U) epoch_.at(slot) = 1U;  // a new request (1..255): the cost fits of the steps it joins restart
    last_finish_ok_ = false;  // its prefill / import ran in the loop since the last step: that gap is no per-step cost
    prompts_[slot].assign(prompt.begin(), prompt.end());
    positions_[slot] = 0;
    lookup_reset(slot, prompt);
    owners_[slot] = options_.world == 2U && options_.split_started ? slot % 2U : 0U;
    in_step_[slot] = false;  // rank 1 may not see a prompt read in rows: imported() requests only
}

void BatchSpeculator::imported(std::uint32_t slot, std::span<const std::uint32_t> prompt, std::uint32_t position, std::uint32_t first) {
    imported(slot, prompt, position, first, {}, 0U, nullptr);
}

void BatchSpeculator::imported(std::uint32_t slot, std::span<const std::uint32_t> prompt, std::uint32_t position, std::uint32_t first,
                               std::array<const float*, 3> seed, std::uint32_t rows, void* stream) {
    start(slot, prompt);  // selects the slot's context, empty
    auto& ctx = slots_->context[slot];
    ctx.assign(prompt.begin(), prompt.end());
    ctx.push_back(first);
    lookup_commit(slot, first, position);
    positions_[slot] = position;
    owners_[slot] = options_.world == 2U ? slot % 2U : 0U;
    in_step_[slot] = coop_;  // both ranks import it with the same seed rows
    rows = keeps(slot) && seed[0] != nullptr ? std::min({rows, position, stage_rows}) : 0U;
    drafter_.skip_to(position - rows);  // the seed rows (if any) come first
    if (rows != 0U) {
        stage_from(slot, seed, 0U, rows, stream);
        flush(slot);  // now, not at the first draft: a one-time append the per-draft costs must not carry
    }
}

bool BatchSpeculator::follow_round(std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens, std::uint64_t step) {
    if (options_.world != 2U || options_.rank != 1U) throw std::logic_error("deepseek41 batch dspark: follow_round is rank 1's");
    std::array<std::byte, round_bytes> o{}, i{};
    const DraftReply none;  // rank 1 sends nothing in this half
    std::memcpy(o.data(), &none, sizeof none);
    options_.exchange(o, i);
    return follow_packet(i, last_rows, last_tokens, step);
}

std::vector<std::uint32_t> BatchSpeculator::take_commits() {
    std::vector<std::uint32_t> out;
    out.swap(pending_commits_);
    return out;
}

void BatchSpeculator::follow_commit(std::uint32_t slot, std::uint32_t kept) {
    target_.commit(slot, kept);
    kept_.at(slot) = kept;
}

void BatchSpeculator::follow_stage(std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens, std::uint64_t step) {
    if (step == staged_step_) return;  // once per step: a draft round may have staged it already
    staged_step_ = step;
    auto& S = *slots_;
    // the last step's kept rows: positions, draft vocabulary context, and the hidden rows of this rank's requests
    for (std::uint32_t r0 = 0; r0 < last_rows.size();) {
        const auto slot = last_rows[r0].slot;
        std::uint32_t count = 1;
        while (r0 + count < last_rows.size() && last_rows[r0 + count].slot == slot) ++count;
        const std::uint32_t k = std::min(count, kept_.at(slot));
        const bool prompt = positions_[slot] < prompts_[slot].size();
        const bool lookup_window = slot < 32U && ((lookup_mask_follow_ >> slot) & 1U) != 0U;
        if (count > 1U && !coop_pending_[slot] && !lookup_window) in_step_[slot] = false;  // as finish()
        coop_pending_[slot] = false;
        if (keeps(slot)) {
            stage(slot, r0, k);
            if (prompt) {
                if (positions_[slot] + count >= prompts_[slot].size()) {
                    S.context[slot].insert(S.context[slot].end(), prompts_[slot].begin(), prompts_[slot].end());
                    S.context[slot].push_back(last_tokens[r0 + count - 1U]);
                }
            } else {
                S.context[slot].insert(S.context[slot].end(), last_tokens.begin() + r0, last_tokens.begin() + r0 + k);
            }
        }
        positions_[slot] += k;
        r0 += count;
    }
    kept_.fill(0xFFFFFFFFU);
    lookup_mask_follow_ = 0U;
}

bool BatchSpeculator::follow_packet(std::span<const std::byte> packet, std::span<const BatchRow> last_rows, std::span<const std::uint32_t> last_tokens,
                                    std::uint64_t step, std::vector<std::pair<std::uint32_t, std::uint32_t>>* applied) {
    if (options_.world != 2U || options_.rank != 1U) throw std::logic_error("deepseek41 batch dspark: follow_packet is rank 1's");
    DraftRound round;
    if (packet.size() < sizeof round) throw std::invalid_argument("deepseek41 batch dspark: short draft round");
    std::memcpy(&round, packet.data(), sizeof round);
    if (round.magic != round_magic || round.commits > ds::BatchWindowController::max_requests ||
        round.requests > ds::BatchWindowController::max_requests)
        throw std::runtime_error("deepseek41 batch dspark: bad draft round");
    for (std::uint32_t c = 0; c < round.commits; ++c) {
        follow_commit(round.commit[c] >> 16U, round.commit[c] & 0xFFFFU);
        if (applied) applied->emplace_back(round.commit[c] >> 16U, round.commit[c] & 0xFFFFU);
    }
    follow_stage(last_rows, last_tokens, step);
    if (round.done) return false;
    if (round.coop != 0U) {
        if (round.requests != 1U || round.slot[0] >= in_step_.size() || !in_step_[round.slot[0]])
            throw std::runtime_error("deepseek41 batch dspark: cooperative round for a context not in step on rank 1");
        (void)coop_draft(round.slot[0], round.token[0]);  // rank 0 has the draft: no reply
        coop_pending_[round.slot[0]] = true;
        return true;
    }
    DraftReply reply;
    for (std::uint32_t q = 0; q < round.requests; ++q) {
        const auto slot = round.slot[q];
        in_step_[slot] = false;  // drafted here alone
        flush(slot);
        drafter_.select(slot);
        const auto d = drafter_.draft(round.token[q], false);
        reply.slot[q] = slot;
        for (std::uint32_t j = 0; j < ds::block_size; ++j) reply.tokens[q][j] = d.tokens[j + 1U], reply.confidence[q][j] = d.confidence[j];
    }
    reply.count = round.requests;
    (void)swap_packets<DraftRound>(options_.exchange, reply);
    return true;
}

ds::Draft BatchSpeculator::coop_draft(std::uint32_t slot, std::uint32_t token) {
    flush(slot);
    drafter_.select(slot);
    // each Markov step's candidate (value, id: 8 bytes) in one 512-byte host exchange; the candidate chain's payload
    // (STRIX_DS41_DSPARK_MARKOV=cand, once per draft) in exchanges of at most 64 KiB (the control channel's limit)
    drafter_.set_rank(options_.rank, 2U, [this](void* stream, const void* local, void* remote, std::size_t bytes) {
        constexpr std::size_t chunk = std::size_t{64} << 10U;
        std::vector<std::byte> o(std::max(bytes, round_bytes)), i(o.size());
        check(hipMemcpyAsync(o.data(), local, bytes, hipMemcpyDeviceToHost, static_cast<hipStream_t>(stream)), "coop candidate");
        check(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "coop candidate");
        for (std::size_t off = 0; off < o.size(); off += chunk) {
            const std::size_t n = std::min(chunk, o.size() - off);
            options_.exchange(std::span<const std::byte>(o).subspan(off, n), std::span<std::byte>(i).subspan(off, n));
        }
        check(hipMemcpyAsync(remote, i.data(), bytes, hipMemcpyHostToDevice, static_cast<hipStream_t>(stream)), "coop candidate");
        check(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "coop candidate");  // i is on this frame
    });
    struct OneRank {  // the drafter scans the whole head again however the draft ends
        ds41::dspark::DeviceDrafter& d;
        ~OneRank() { d.set_rank(0U, 1U, {}); }
    } one{drafter_};
    return drafter_.draft(token, false);
}

struct BatchSpeculator::ParkedPlan {
    PlannedStep plan;
    std::vector<ds::Draft> drafts;
    std::vector<std::uint32_t> window_rows;
    std::vector<bool> drafted, lookup;
    std::vector<std::uint32_t> lookup_cap;
    std::uint32_t lookup_mask = 0;
    std::uint32_t planned_drafting = 0, planned_decoders = 0, planned_total = 0;
    bool planned_prompt_rows = false;
    double plan_seconds = 0.0;
    std::uint64_t planned_key = 0;          // calib: the step's requests
    double planned_gap = -1.0, planned_overhead = 0.0;
    bool planned_probe = false;
};

const BatchSpeculator::CostGroup* BatchSpeculator::cost_group(std::uint32_t n) const noexcept {
    const CostGroup* best = nullptr;
    for (const auto& g : groups_)
        if (g.n == n && g.base_n != 0U && (best == nullptr || g.used > best->used)) best = &g;
    return best;
}

std::shared_ptr<BatchSpeculator::ParkedPlan> BatchSpeculator::park() {
    auto p = std::make_shared<ParkedPlan>();
    p->plan = std::move(plan_);
    p->drafts = std::move(drafts_);
    p->window_rows = std::move(window_rows_);
    p->drafted = std::move(drafted_);
    p->lookup = std::move(lookup_);
    p->lookup_cap = std::move(lookup_cap_);
    p->lookup_mask = lookup_mask_;
    p->planned_drafting = planned_drafting_;
    p->planned_decoders = planned_decoders_;
    p->planned_total = planned_total_;
    p->planned_prompt_rows = planned_prompt_rows_;
    p->plan_seconds = plan_seconds_;
    p->planned_key = planned_key_, p->planned_gap = planned_gap_, p->planned_overhead = planned_overhead_, p->planned_probe = planned_probe_;
    plan_ = {};
    drafts_.clear();
    window_rows_.clear();
    drafted_.clear();
    lookup_.clear();
    lookup_cap_.clear();
    return p;
}

void BatchSpeculator::resume(const std::shared_ptr<ParkedPlan>& p) {
    if (!p) throw std::invalid_argument("deepseek41 batch dspark: no parked plan");
    plan_ = p->plan;
    drafts_ = p->drafts;
    window_rows_ = p->window_rows;
    drafted_ = p->drafted;
    lookup_ = p->lookup;
    lookup_cap_ = p->lookup_cap;
    lookup_mask_ = p->lookup_mask;
    planned_drafting_ = p->planned_drafting;
    planned_decoders_ = p->planned_decoders;
    planned_total_ = p->planned_total;
    planned_prompt_rows_ = p->planned_prompt_rows;
    plan_seconds_ = p->plan_seconds;
    planned_key_ = p->planned_key, planned_gap_ = p->planned_gap, planned_overhead_ = p->planned_overhead, planned_probe_ = p->planned_probe;
}

const PlannedStep& BatchSpeculator::plan(std::span<BatchSequence* const> seqs, std::span<const std::uint32_t> remaining) {
    const auto t0 = clock_type::now();
    planned_gap_ = last_finish_ok_ ? std::chrono::duration<double>(t0 - last_finish_end_).count() : -1.0;  // calib: loop time since the last decode step
    last_finish_ok_ = false;
    const auto n = static_cast<std::uint32_t>(seqs.size());
    if (remaining.size() != n) throw std::invalid_argument("deepseek41 batch dspark: one remaining count per sequence");
    plan_ = {};
    drafts_.assign(n, ds::Draft{});
    window_rows_.assign(n, 0U);
    drafted_.assign(n, false);
    lookup_.assign(n, false);
    lookup_cap_.assign(n, 0U);
    lookup_mask_ = 0U;
    if (lookup_log_) lookup_trace_.assign(n, LookupTrace{});
    // decoding requests: one row each at least
    std::vector<std::uint32_t> dec;
    for (std::uint32_t i = 0; i < n; ++i)
        if (!seqs[i]->reading_prompt() && !seqs[i]->generated.empty() && remaining[i] != 0U) dec.push_back(i);
    if (dec.size() > options_.max_rows) throw std::invalid_argument("deepseek41 batch dspark: more decoding requests than rows per step");
    const auto nd = static_cast<std::uint32_t>(dec.size());
    // prompt chunks next (oldest first), as plan_step
    std::uint32_t used = nd;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> chunks;  // (sequence, rows)
    for (std::uint32_t i = 0; i < n && used < options_.max_rows; ++i) {
        const auto& s = *seqs[i];
        if (!s.reading_prompt()) continue;
        std::size_t take = std::min<std::size_t>({options_.prompt_rows, options_.max_rows - used, s.prompt.size() - s.fed});
        if (s.position == 0U) take = std::min<std::size_t>(take, 1U);
        if (take == 0U) continue;
        chunks.emplace_back(i, static_cast<std::uint32_t>(take));
        used += static_cast<std::uint32_t>(take);
    }
    const std::uint32_t budget = options_.max_rows - (used - nd);  // rows for the windows (>= nd)
    // per decoding request: at most 6 rows, its remaining tokens, its room, and one row for every other request
    std::vector<std::uint32_t> slots(nd), limit(nd);
    for (std::uint32_t k = 0; k < nd; ++k) {
        const auto& s = *seqs[dec[k]];
        const std::uint32_t room = target_.capacity(s.slot) > s.position ? target_.capacity(s.slot) - s.position : 1U;
        slots[k] = s.slot;
        limit[k] = std::max(1U, std::min({ds::block_size + 1U, remaining[dec[k]], room, budget - (nd - 1U)}));
        if (options_.align != 0U) limit[k] = std::max(1U, std::min(limit[k], options_.align - s.position % options_.align));
    }
    planned_probe_ = false;
    if (calib_ && nd != 0U && chunks.empty() && ++probe_plans_ % 32U == 0U) probe_due_ = true;  // every 32 decode plans
    // a lone drafting request is drafted cooperatively only when its context is in step: the controller prices it so
    // when every decoding request's is (whichever it picks), else as draft + round
    if (coop_) {
        bool all = nd != 0U;
        for (std::uint32_t k = 0; k < nd; ++k) all = all && in_step_[slots[k]];
        auto c = controller_.costs();
        c.coop = all && coop_n_ != 0U ? static_cast<float>(coop_mean_) : 0.0F;
        controller_.set_costs(c);
    }
    // prompt lookup first: per decoding request the window of the best estimated rate (tokens per priced step time;
    // rows only, no draft) where it beats the mean rate of the request's alternative (its measured DSpark windows
    // while DSpark may draft, else a plain row) by the margin; stage 1 then plans DSpark for the others in the rows left
    std::vector<std::vector<double>> given;
    std::vector<ds::Draft> looked;
    std::uint32_t nlookup = 0, lookup_rows = 0;
    std::vector<std::uint32_t> limit1 = limit;
    if (lookup_on_ && nd != 0U) {
        const auto t_lookup = clock_type::now();
        const auto& c = controller_.costs();
        const double base = controller_.plain_cost(nd), row = c.row;
        given.assign(nd, {});
        looked.assign(nd, ds::Draft{});
        for (std::uint32_t k = 0; k < nd; ++k) {
            const auto slot = slots[k];
            auto& L = lookup_slots_[slot];
            if (!L.index) continue;
            if (!L.backlog.empty()) lookup_catch_up(slot);
            const auto& s = *seqs[dec[k]];
            if (!L.backlog.empty() || L.next_position != std::uint64_t{s.position} + 1U) continue;  // not current
            const auto m = L.index->longest({}, lookup_min_match, ds::block_size);
            if (m.length < lookup_min_match || m.available == 0U) continue;
            const auto b = lookup_bucket(m.length);
            if (lookup_log_) {
                auto& tr = lookup_trace_[dec[k]];
                tr.candidate = true, tr.length = m.length, tr.bucket = b, tr.available = std::min<std::uint32_t>(m.available, ds::block_size);
                for (std::uint32_t j = 0; j < tr.available; ++j) tr.token[j] = m.next[j];
            }
            // the evidence (decayed) that this bucket's candidates survive, scored on what this request commits
            for (auto& x : L.eligible[b]) x *= lookup_decay;
            for (auto& x : L.accepted[b]) x *= lookup_decay;
            const bool enough = L.evidence[b] >= lookup_evidence && base > 0.0 && row > 0.0;
            L.evidence[b] = L.evidence[b] * lookup_decay + 1.0;
            if (L.pending.size() < lookup_pending) {
                LookupSlot::Pending p;
                p.first = std::uint64_t{s.position} + 1U;
                p.count = std::min<std::uint32_t>(m.available, ds::block_size);
                p.bucket = b;
                for (std::uint32_t j = 0; j < p.count; ++j) p.token[j] = m.next[j];
                L.pending.push_back(p);
            }
            if (!enough || budget <= nd + lookup_rows) continue;
            const bool dspark_may = allow_drafts_ && limit[k] > 1U;
            if (dspark_may && L.dspark.n < 3.0) continue;  // the alternative is measured first
            // the alternative at today's prices: its tokens (upper bound) over a step with its mean rows and the cheaper
            // of a draft and a cooperative draft (when the request could draft cooperatively)
            double dspark_rate = 0.0;
            if (dspark_may) {
                // its draft at the price of the route it would take: alone, a cooperative draft while it is in step, else a
                // plain one; beside other decoders (drafts on each rank, cooperative only if it ends up the sole drafter)
                // the cheaper of the two, the conservative bound. A possible route unmeasured (0) is unknown: no lookup.
                // (this request's own eligibility and the cooperative measurement itself, not the controller's price,
                // which is zero as soon as any decoder is out of step)
                const bool coop_route = coop_ && in_step_[slot];
                const double coop_price = coop_n_ != 0U ? coop_mean_ : 0.0, plain_price = c.draft;
                double draft = 0.0;
                if (nd == 1U) {
                    draft = coop_route ? coop_price : plain_price;
                } else if (coop_route) {  // plain beside other drafters, cooperative if it ends up the sole drafter
                    draft = coop_price > 0.0 && plain_price > 0.0 ? std::min(coop_price, plain_price) : 0.0;
                } else {
                    draft = plain_price;
                }
                if (!(draft > 0.0)) continue;
                dspark_rate = L.dspark.mean_tokens() / (base + row * (L.dspark.rows / L.dspark.n) + draft);
            }
            const double alternative = std::max(1.0 / base, dspark_rate) * lookup_margin_;
            std::uint32_t depth = std::min<std::uint32_t>({m.available, ds::block_size, limit[k] - 1U, budget - nd - lookup_rows});
            if (options_.fixed_rows != 0U) depth = std::min(depth, options_.fixed_rows - 1U);
            double best = alternative, survival = 1.0, yield = 1.0;
            std::uint32_t cap = 0;
            std::vector<double> p;
            for (std::uint32_t j = 0; j < depth; ++j) {
                // the Beta(1,2) posterior mean per depth, compared with the alternative's mean rate (both point estimates:
                // a lower bound against an upper one ate the whole draft saving where DSpark accepts ~95 %; the native
                // ABBA qualifies 'never slower')
                const double q = (L.accepted[b][j] + 1.0) / (L.eligible[b][j] + 3.0);
                p.push_back(q);
                survival *= q;
                yield += survival;
                const double rate = yield / (base + row * (j + 1U));
                if (rate > best) best = rate, cap = j + 1U;
            }
            if (lookup_log_) lookup_trace_[dec[k]].rate = best, lookup_trace_[dec[k]].alternative = alternative, lookup_trace_[dec[k]].cap = cap;
            if (cap == 0U) continue;
            p.resize(cap);
            given[k] = std::move(p);
            lookup_cap_[dec[k]] = cap;
            auto& d = looked[k];
            d.tokens[0] = s.generated.back();
            for (std::uint32_t j = 0; j < cap; ++j) d.tokens[j + 1U] = m.next[j];
            limit1[k] = 1U;  // no DSpark draft for it
            lookup_rows += cap;
            ++nlookup;
        }
        if (nlookup == 0U) given.clear();
        stats_.lookup_seconds += since(t_lookup);
    }
    // stage 1: who drafts (the first decode step of every request count is plain: its cost is measured first)
    std::vector<bool> drafting(nd, false);
    const std::uint32_t budget1 = budget - lookup_rows;
    if (nd != 0U && budget1 > nd && allow_drafts_) {
        if (options_.fixed_rows != 0U) {
            for (std::uint32_t k = 0; k < nd; ++k) drafting[k] = limit1[k] > 1U;
        } else if ((calib_ ? measured_n_.at(nd) : plain_n_[nd] != 0U) || !chunks.empty()) {
            // (a step with prompt rows may draft before a measured decode step of nd: those steps are never cost samples)
            drafting = controller_.plan_drafts(slots, limit1, budget1);
        }
    }
    std::vector<ds::Draft> drafts(nd);
    std::uint32_t ndraft = 0;
    // world 2: the draft round -- rank 1's requests out (with the last step's commits), its drafts back after ours
    const auto t_round = clock_type::now();
    std::vector<std::uint32_t> remote;  // k drafted by rank 1
    for (std::uint32_t k = 0; k < nd; ++k)
        if (drafting[k] && owner(slots[k]) == 1U) remote.push_back(k);
    // coop_: the only drafting request, its context in step on both ranks, is drafted by both (a round without reply)
    std::uint32_t coop_k = nd;
    if (coop_) {
        std::uint32_t count = 0;
        for (std::uint32_t k = 0; k < nd; ++k)
            if (drafting[k]) ++count, coop_k = k;
        if (count != 1U || !in_step_[slots[coop_k]]) coop_k = nd;
        else remote.clear();
    }
    const auto t_coop = clock_type::now();  // a cooperative draft's price includes its entry exchange
    if (coop_k != nd) {
        DraftRound round;
        round.coop = 1U;
        round.commits = static_cast<std::uint32_t>(pending_commits_.size());
        std::copy(pending_commits_.begin(), pending_commits_.end(), round.commit);
        pending_commits_.clear();
        round.slot[0] = slots[coop_k];
        round.token[0] = seqs[dec[coop_k]]->generated.back();
        round.requests = 1U;
        (void)swap_packets<DraftReply>(options_.exchange, round);
    }
    const bool split = options_.world == 2U && !remote.empty();  // otherwise no round: the commits go with the step's packet
    if (split) {
        DraftRound round;
        round.commits = static_cast<std::uint32_t>(pending_commits_.size());
        std::copy(pending_commits_.begin(), pending_commits_.end(), round.commit);
        pending_commits_.clear();
        for (const auto k : remote) {
            round.slot[round.requests] = slots[k];
            round.token[round.requests++] = seqs[dec[k]]->generated.back();
        }
        (void)swap_packets<DraftReply>(options_.exchange, round);
    }
    for (std::uint32_t k = 0; k < nd; ++k) {
        if (!drafting[k]) continue;
        if (k != coop_k) in_step_[slots[k]] = false;  // drafted (flushed) on one rank only
        if (options_.world == 2U && k != coop_k && owner(slots[k]) == 1U) {
            drafted_[dec[k]] = true;
            ++ndraft;
            continue;
        }
        const auto td = clock_type::now();
        if (k == coop_k) {
            drafts[k] = coop_draft(slots[k], seqs[dec[k]]->generated.back());
            coop_pending_[slots[k]] = true;
            ++stats_.coop_drafts;
            const double dc = since(t_coop);  // its own price: half the head, the entry and candidate exchanges
            stats_.draft_seconds += since(td);
            if (chunks.empty() && (!calib_ || seen_coop_)) accumulate(coop_mean_, coop_n_, dc);
            seen_coop_ = true;
            drafts_[dec[k]] = drafts[k];
            drafted_[dec[k]] = true;
            ++ndraft;
            continue;
        }
        flush(slots[k]);
        drafter_.select(slots[k]);
        drafts[k] = drafter_.draft(seqs[dec[k]]->generated.back(), false);
        const double ds_ = since(td);
        stats_.draft_seconds += ds_;
        if (chunks.empty() && (!calib_ || seen_draft_)) accumulate(draft_mean_, draft_n_, ds_);  // calib: not the first call
        seen_draft_ = true;
        drafts_[dec[k]] = drafts[k];
        drafted_[dec[k]] = true;
        ++ndraft;
    }
    if (split) {
        const auto rp = swap_packets<DraftReply>(options_.exchange, DraftRound{});  // rank 0 sends nothing in this half
        if (rp.magic != DraftReply{}.magic || rp.count != remote.size()) throw std::runtime_error("deepseek41 batch dspark: bad draft reply");
        for (std::uint32_t q = 0; q < rp.count; ++q) {
            const auto k = remote[q];
            if (rp.slot[q] != slots[k]) throw std::runtime_error("deepseek41 batch dspark: draft reply for another slot");
            auto& d = drafts[k];
            d.tokens[0] = seqs[dec[k]]->generated.back();
            for (std::uint32_t j = 0; j < ds::block_size; ++j) d.tokens[j + 1U] = rp.tokens[q][j], d.confidence[j] = rp.confidence[q][j];
            drafts_[dec[k]] = d;
        }
        if (ndraft != 0U && chunks.empty() && draft_n_ != 0U && (!calib_ || seen_round_))  // the round's cost beyond its parallel drafts
            accumulate(round_mean_, round_n_, since(t_round) - draft_mean_ * ((ndraft + 1U) / 2U));
        seen_round_ = true;  // calib: the first round (rank 1's first drafter call) is first use
    }
    for (std::uint32_t k = 0; k < nd && nlookup != 0U; ++k)
        if (!given[k].empty()) {
            drafts[k] = looked[k];
            drafts_[dec[k]] = drafts[k];
            drafted_[dec[k]] = true;
            lookup_[dec[k]] = true;
            if (slots[k] < 32U) lookup_mask_ |= 1U << slots[k];
        }
    const double draft_span = since(t_round);  // the drafts and the draft round (not per-step overhead)
    // stage 2: rows per request
    std::vector<std::uint32_t> rows(nd, 1U);
    if (ndraft + nlookup != 0U) {
        if (options_.fixed_rows != 0U) {
            for (std::uint32_t k = 0; k < nd; ++k) rows[k] = drafting[k] ? std::min(options_.fixed_rows, limit1[k]) : 1U;
            std::uint32_t total = 0;
            for (const auto r : rows) total += r;
            while (total > budget1) {  // the last of the largest windows gives a row back (lookup rows are reserved)
                std::uint32_t big = 0;
                for (std::uint32_t k = 0; k < nd; ++k)
                    if (rows[k] >= rows[big]) big = k;
                --rows[big];
                --total;
            }
        } else {
            rows = controller_.choose_rows(slots, limit1, budget - lookup_rows, drafting, drafts);
            if (calib_ && probe_due_ && chunks.empty()) {  // a step with prompt rows is no cost sample: keep it due
                std::uint32_t total = 0;
                bool wide = false;
                for (const auto r : rows) total += r, wide = wide || r > 1U;
                if (wide) probe_due_ = false;  // the controller varies windows by itself: no probe needed
                for (std::uint32_t k = 0; k < nd && probe_due_ && total < budget; ++k)
                    if (drafting[k] && rows[k] == 1U && limit[k] > 1U) {
                        rows[k] = 2U;  // the probe row: verified like any window row
                        ++stats_.probe_rows;
                        probe_due_ = false, planned_probe_ = true;
                    }
            }
        }
    }
    for (std::uint32_t k = 0; k < nd && nlookup != 0U; ++k)
        if (!given[k].empty()) rows[k] = lookup_cap_[dec[k]] + 1U;  // as chosen (its rows were reserved before stage 1)
    // the rows: every decoding request's window, then the prompt chunks
    std::uint32_t total = 0;
    for (std::uint32_t k = 0; k < nd; ++k) {
        const auto i = dec[k];
        const auto& s = *seqs[i];
        window_rows_[i] = rows[k];
        plan_.rows.push_back({s.slot, s.generated.back()});
        plan_.owner.push_back(i);
        plan_.emits.push_back(true);
        for (std::uint32_t j = 1; j < rows[k]; ++j) {
            plan_.rows.push_back({s.slot, drafts[k].tokens[j]});
            plan_.owner.push_back(i);
            plan_.emits.push_back(true);
        }
        total += rows[k];
    }
    for (const auto& [i, take] : chunks) {
        const auto& s = *seqs[i];
        for (std::uint32_t k = 0; k < take; ++k) {
            plan_.rows.push_back({s.slot, s.prompt[s.fed + k]});
            plan_.owner.push_back(i);
            plan_.emits.push_back(s.fed + k + 1U == s.prompt.size());
        }
    }
    planned_key_ = 0;
    for (std::uint32_t k = 0; k < nd; ++k) planned_key_ |= std::uint64_t{epoch_.at(slots[k])} << (8U * slots[k]);
    planned_decoders_ = nd;
    planned_drafting_ = ndraft;
    planned_total_ = total;
    planned_prompt_rows_ = !chunks.empty();
    plan_seconds_ = since(t0);
    planned_overhead_ = std::max(plan_seconds_ - draft_span, 0.0);
    stats_.plan_seconds += plan_seconds_;
    return plan_;
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> BatchSpeculator::finish(std::span<BatchSequence* const> seqs, std::span<const std::uint32_t> tokens) {
    last_finish_ok_ = false;  // set again below when this step is a cost sample
    const auto t0 = clock_type::now();
    if (tokens.size() != plan_.rows.size()) throw std::invalid_argument("deepseek41 batch dspark: one token per planned row");
    auto& S = *slots_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> commits;
    std::uint64_t emitted = 0;
    for (std::uint32_t r0 = 0; r0 < plan_.rows.size();) {
        const auto i = plan_.owner[r0];
        std::uint32_t count = 1;
        while (r0 + count < plan_.rows.size() && plan_.owner[r0 + count] == i) ++count;
        auto& s = *seqs[i];
        const auto slot = s.slot;
        const bool coop_window = coop_pending_[slot];
        // rows not from a cooperative draft or a lookup window (as follow_stage): a one-rank draft flushed on one rank
        if (count > 1U && !coop_window && !lookup_[i]) in_step_[slot] = false;
        coop_pending_[slot] = false;
        if (window_rows_[i] != 0U) {  // a window: row j + 1 is kept when draft j + 1 equalled row j's greedy token
            std::uint32_t accepted = 0;
            while (accepted + 1U < count && plan_.rows[r0 + accepted + 1U].token == tokens[r0 + accepted]) ++accepted;
            const std::uint32_t kept = accepted + 1U;
            if (kept < count) {
                target_.commit(slot, kept);
                commits.emplace_back(slot, kept);
                if (options_.world == 2U) pending_commits_.push_back(slot << 16U | kept);
            }
            if (keeps(slot)) stage(slot, r0, kept);
            for (std::uint32_t j = 0; j < kept; ++j) {
                s.generated.push_back(tokens[r0 + j]);
                lookup_commit(slot, tokens[r0 + j], std::uint64_t{s.position} + 1U + j);
                if (keeps(slot)) S.context[slot].push_back(tokens[r0 + j]);
            }
            s.position += kept;
            emitted += kept;
            ++stats_.window_rows[count];
            if (count == 1U) ++stats_.plain_rows;
            else ++stats_.windows;
            if (lookup_log_ && i < lookup_trace_.size() && lookup_trace_[i].candidate) {
                const auto& tr = lookup_trace_[i];
                std::uint32_t agree = 0;  // the lookup candidate's prefix equal to the DSpark draft that ran
                if (drafted_[i] && !lookup_[i])
                    while (agree < tr.available && agree < ds::block_size && drafts_[i].tokens[agree + 1U] == tr.token[agree]) ++agree;
                std::uint32_t truth = 0;  // the candidate's prefix equal to the tokens committed this step
                while (truth < tr.available && truth < kept && tokens[r0 + truth] == tr.token[truth]) ++truth;
                std::fprintf(stderr, "dspark lookup: slot %u pos %u len %u bucket %u avail %u cap %u rate %.2f alt %.2f ran %s rows %u accepted %u agree %u truth %u kept %u\n",
                             slot, s.position, tr.length, tr.bucket, tr.available, tr.cap, tr.rate * 1e-3, tr.alternative * 1e-3,
                             lookup_[i] ? "lookup" : drafted_[i] ? "dspark" : "plain", count, accepted, agree, truth, kept);
            }
            if (lookup_[i]) {  // a prompt-lookup window: its own counts (the DSpark controller observes DSpark drafts only)
                if (count > 1U) {
                    ++stats_.lookup_windows;
                    stats_.lookup_verified += count - 1U;
                    stats_.lookup_accepted += accepted;
                }
            } else if (drafted_[i]) {  // with one row the step's own token still checks draft 1
                const std::uint32_t verified = std::max(count - 1U, 1U);
                const std::uint32_t acc = count > 1U ? accepted : (drafts_[i].tokens[1] == tokens[r0] ? 1U : 0U);
                if (cost_log_) {  // diagnostic: the controller's per-depth acceptance before this observation vs the outcome
                    char line[512];
                    int len = std::snprintf(line, sizeof line, "dspark accept: t %.1f slot %u pos %u rows %u verified %u accepted %u p/conf",
                                            std::chrono::duration<double, std::milli>(clock_type::now().time_since_epoch()).count(), slot,
                                            s.position, count, verified, acc);
                    for (std::uint32_t j = 0; j < ds::block_size && len > 0 && len < 480; ++j)
                        len += std::snprintf(line + len, sizeof line - static_cast<std::size_t>(len), " %.2f/%.2f", controller_.probability(j, drafts_[i].confidence[j]),
                                             drafts_[i].confidence[j]);
                    std::fprintf(stderr, "%s\n", line);
                }
                controller_.observe(slot, drafts_[i], verified, acc);
                if (lookup_on_) lookup_slots_[slot].dspark.add(kept, count - 1U);  // the alternative a lookup window must beat
                stats_.drafts_verified += count - 1U;
                stats_.drafts_accepted += accepted;
                for (std::uint32_t j = 0; j + 1U < count && j <= accepted && j < ds::block_size; ++j) {
                    ++stats_.reached[j];
                    if (j < accepted) ++stats_.hit[j];
                }
            }
        } else {  // prompt rows: every one is kept
            const bool mine = owner(slot) == options_.rank;
            if (mine) stage(slot, r0, count);
            s.position += count;
            s.fed += count;
            if (!s.reading_prompt()) {
                if (mine) S.context[slot].insert(S.context[slot].end(), s.prompt.begin(), s.prompt.end());
                s.generated.push_back(tokens[r0 + count - 1U]);
                lookup_commit(slot, tokens[r0 + count - 1U], s.position);
                if (mine) S.context[slot].push_back(tokens[r0 + count - 1U]);
            }
        }
        r0 += count;
    }
    const double fs = since(t0);
    const double ss = options_.step_seconds ? options_.step_seconds() : target_.last_stats().seconds;
    stats_.finish_seconds += fs;
    stats_.step_seconds += ss;
    ++stats_.steps;
    stats_.rows += plan_.rows.size();
    if (planned_drafting_ != 0U) ++stats_.drafted_steps;
    ++steps_;
    if (cost_log_)  // diagnostic: every step, both estimators (timeline of a request's steps)
        std::fprintf(stderr, "dspark step: t %.1f decoders %u rows %u drafting %u prompt %d plan %.2f step %.2f finish %.2f emitted %llu ms\n",
                     std::chrono::duration<double, std::milli>(clock_type::now().time_since_epoch()).count(), planned_decoders_,
                     planned_total_, planned_drafting_, planned_prompt_rows_ ? 1 : 0, plan_seconds_ * 1e3, ss * 1e3, fs * 1e3,
                     static_cast<unsigned long long>(emitted));
    if (!planned_prompt_rows_ && planned_decoders_ != 0U) {
        ++stats_.decode_steps;
        stats_.decode_tokens += emitted;
        stats_.decode_seconds += plan_seconds_ + ss + fs;
        // costs: plain[n] from steps without drafts, the row slope from steps with windows (n requests known)
        const double wall = ss + fs;
        const auto nd = planned_decoders_;
        if (planned_drafting_ == 0U && planned_total_ == nd) {
            accumulate(plain_mean_[nd], plain_n_[nd], wall);
        } else if (plain_n_[nd] != 0U && planned_total_ > nd) {
            accumulate(row_mean_, row_n_, (wall - plain_mean_[nd]) / (planned_total_ - nd));
        }
        ds::BatchWindowController::Costs c;
        for (std::uint32_t k = 1; k <= ds::BatchWindowController::max_requests; ++k)
            c.plain[k] = plain_n_[k] != 0U ? static_cast<float>(plain_mean_[k]) : 0.0F;
        c.row = row_n_ != 0U ? static_cast<float>(std::max(row_mean_, 1e-4)) : 0.0F;
        if (calib_) {
            const bool first = !seen_rows_.at(planned_total_);  // the first step of this row count: first use, not a sample
            seen_rows_.at(planned_total_) = true;
            const std::uint32_t x = planned_total_ - nd;
            // a row costs at most half a one-request step (caps a pair and the slope: one timing hiccup cannot price rows
            // so high that windows never vary again)
            const CostGroup* g1 = cost_group(1U);
            const double base1 = g1 != nullptr ? g1->base : base1_last_;  // by value: the LRU lookup below may reuse g1's entry
            const double cap = 0.5 * (g1 != nullptr || base1_last_ > 0.0 ? base1 : wall);
            // the row cost in effect: the paired least-squares slope (with its prior), else the controller's default
            const auto row_now = [&] { return pair_xx_ > 0.0 ? std::clamp(pair_xy_ / pair_xx_, 0.0, cap) : 0.13 * base1; };
            // this step's group: the entry of its request set, else the least recently used entry, restarted
            CostGroup* g = nullptr;
            for (auto& e : groups_)
                if (e.n == nd && e.key == planned_key_) g = &e;
            if (g == nullptr) {
                g = &*std::min_element(groups_.begin(), groups_.end(), [](const CostGroup& a, const CostGroup& b) { return a.used < b.used; });
                *g = CostGroup{};
                g->key = planned_key_, g->n = nd;
            }
            if (g->base_n != 0U && g->used + 16U < steps_) g->base_n = 0U, g->prev_ok = false;  // stale: restart
            g->used = steps_;
            if (!first) {
                // a pair: the group's previous sample, at most 2 steps back (the contexts moved by a few positions only)
                if (g->prev_ok && g->prev_step + 2U >= steps_ && g->prev_x != x) {
                    if (pair_xx_ == 0.0) {  // prior: the default (13 % of plain[1]) as 4 pseudo-pairs of dx = 1
                        const double prior = 0.13 * base1;
                        pair_xx_ = 4.0;
                        pair_xy_ = 4.0 * prior;
                    }
                    const double dx = static_cast<double>(x) - static_cast<double>(g->prev_x);
                    const double slope = std::clamp((wall - g->prev_wall) / dx, -cap, cap);
                    const double w = std::min(dx * dx, 4.0);  // least-squares weight, capped: one hiccup outweighs few pairs
                    constexpr double keep = 1.0 - 1.0 / 64.0;  // ~ the last 64 pairs (the slope drifts with the context too)
                    pair_xy_ = keep * pair_xy_ + w * slope, pair_xx_ = keep * pair_xx_ + w;
                    ++pairs_;
                }
                accumulate(g->base, g->base_n, wall - row_now() * x);
                measured_n_.at(nd) = true;
                if (nd == 1U) base1_last_ = g->base;
                g->prev_ok = true, g->prev_wall = wall, g->prev_x = x, g->prev_step = steps_;
                // the per-step overhead outside the wall: the loop's gap before this plan + plan() without its drafts;
                // a sample above 3x the estimate (before 4 samples: above this step's wall) is a prefill / idle gap: dropped
                if (planned_gap_ >= 0.0) {
                    const double o = planned_gap_ + planned_overhead_;
                    if (o <= (overhead_n_ >= 4U ? 3.0 * overhead_ : wall) || overhead_drops_ >= 8U) {
                        accumulate(overhead_, overhead_n_, o);
                        overhead_drops_ = 0U;
                    } else {
                        ++overhead_drops_;
                    }
                }
                last_finish_ok_ = true;
            } else {
                g->prev_ok = false;
            }
            for (std::uint32_t k = 1; k <= ds::BatchWindowController::max_requests; ++k) {
                const CostGroup* e = cost_group(k);
                c.plain[k] = e != nullptr ? static_cast<float>(std::max(e->base, 1e-4) + overhead_) : 0.0F;
            }
            c.row = static_cast<float>(row_now());  // 0 before any measurement: the controller's own default
            if (cost_log_)
                std::fprintf(stderr, "dspark cost: n %u rows %u drafting %u%s wall %.2f ms%s; plain[n] %.2f (%u samples) row %.3f ms (%u pairs) overhead %.2f ms (%u samples) gap %.2f ms\n",
                             nd, planned_total_, planned_drafting_, planned_probe_ ? " probe" : "", wall * 1e3, first ? " (first use, not a sample)" : "", c.plain[nd] * 1e3,
                             g->base_n, c.row * 1e3, pairs_, overhead_ * 1e3, overhead_n_, planned_gap_ * 1e3);
        }
        c.draft = draft_n_ != 0U ? static_cast<float>(draft_mean_) : 0.0F;
        c.extra = 0.0F;  // the appends are inside the draft calls (draft), the commits inside the measured step walls
        c.drafters = options_.world;
        c.round = round_n_ != 0U ? static_cast<float>(std::max(round_mean_, 0.0)) : 0.0F;
        controller_.set_costs(c);
    } else {
        for (auto& g : groups_) g.prev_ok = false;  // a step with prompt rows (or no decoders) is not a sample and breaks every pair
    }
    if (last_finish_ok_) last_finish_end_ = clock_type::now();  // the gap to the next plan() starts here
    return commits;
}

}  // namespace strix::deepseek41
