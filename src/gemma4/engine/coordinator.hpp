#pragma once

#include "gemma4/contracts/g4_types.hpp"
#include "gemma4/dflash/dflash.hpp"
#include "auto_policy.hpp"
#include "strix/serve/json.hpp"
#include <functional>
#include <map>
#include <stdexcept>

// Optional DraftSession hooks for autoregressive drafters are available (e4b-eagle / MTP check these).
#define STRIX_G4_DRAFT_ROWS_API 1
#define STRIX_G4_DRAFT_KV_API 1
// DraftSession::multi_stream_candidate (auto drafting at >1 live stream, window <= k cap).
#define STRIX_G4_DRAFT_MULTI_API 1
// DraftSession::confidence_capable / propose_confident (drafter-confidence-gated C1 window, STRIX_G4_MTP_CONF).
#define STRIX_G4_DRAFT_CONFIDENCE_API 1

namespace strix::gemma4::engine {

// Backends own cache allocation and transaction type. Scheduling knows only the frozen
// Target/RowPlan/KvTxn contract. All assets and the backend outlive the coordinator.
class StreamState {
public:
    virtual ~StreamState() = default;
    virtual std::unique_ptr<KvTxn> begin() = 0;
    [[nodiscard]] virtual std::uint64_t frontier() const = 0;
    [[nodiscard]] virtual std::string digest() const = 0;
    // Optional read-only KV view of a producing target layer (MTP drafters): committed snapshot + optional overlay of
    // `transaction`. The coordinator drafts before any candidate row exists, so it passes nullptr. false = unavailable.
    virtual bool kv_view(std::uint32_t layer, const KvTxn* transaction, KvView& out) const {
        (void)layer; (void)transaction; (void)out; return false;
    }
    // Optional: a new stream admitting `admitted` positions whose committed state equals this one byte for byte
    // (same digest; the same passes give the same outputs). nullptr = unsupported.
    [[nodiscard]] virtual std::unique_ptr<StreamState> fork(std::uint32_t admitted) const { (void)admitted; return nullptr; }
};
// Rows of one exact many-row prefill call (no logits): taps per row and, when
// requested, the final-norm hidden rows that RowPlan::export_hidden returns.
// A backend may return either as a view of its own memory instead (valid until
// its next prefill_rows call), and may omit taps nobody asked for.
struct PrefillRows {
    std::vector<float> taps, hidden;
    std::uint32_t tap_width = 0;
    std::span<const float> tap_view, hidden_view;
    // Prefill levers that ran for this call (receipt counters, prefill::Lever bits).
    std::uint32_t levers = 0;
    double host_ms = 0; // backend host time outside its device work (receipt timings.prefill_host_ms)
    [[nodiscard]] std::span<const float> tap_rows() const { return tap_view.empty() ? std::span<const float>(taps) : tap_view; }
    [[nodiscard]] std::span<const float> hidden_rows() const { return hidden_view.empty() ? std::span<const float>(hidden) : hidden_view; }
};
// One stream's prompt chunk in a batched prefill call.
struct PrefillItem {
    KvTxn* transaction = nullptr;
    std::span<const std::uint32_t> tokens;
};
class Backend : public Target {
public:
    // Optional exact many-row prefill (prefill-exact). prefill_rows evaluates
    // consecutive positions of a fresh transaction; its committed KV, taps and
    // hidden rows equal forward_rows passes over the same rows. 0 = unsupported.
    // taps=false: the caller consumes no taps, which the backend may then omit.
    [[nodiscard]] virtual std::size_t prefill_chunk_rows() const { return 0; }
    // true when prefill_rows writes the same KV, taps and hidden bytes as forward_rows passes over the same rows
    // under this backend's decode identity; then where prompt chunks end cannot change outputs. false (default)
    // for a prefill whose arithmetic differs from decode (an int8-activation decode beside an FP32 prefill, a WMMA
    // prefill identity): the coordinator keeps prompt chunk boundaries independent of scheduling.
    [[nodiscard]] virtual bool prefill_matches_decode() const { return false; }
    virtual PrefillRows prefill_rows(KvTxn& transaction, std::span<const std::uint32_t> tokens, bool hidden, bool taps) {
        (void)transaction; (void)tokens; (void)hidden; (void)taps;
        throw std::logic_error("backend has no many-row prefill");
    }
    // Whether this backend's prefill route has a byte record for batched passes; false keeps one call per stream.
    virtual bool prefill_batch_supported() const { return false; }
    // Several streams' fresh transactions in one many-row pass, at most prefill_chunk_rows() rows in total: one
    // PrefillRows per item, each equal to prefill_rows on that item alone. Empty = unsupported (one call per stream).
    virtual std::vector<PrefillRows> prefill_rows_batch(std::span<const PrefillItem> items, bool hidden, bool taps) {
        (void)items; (void)hidden; (void)taps;
        return {};
    }
    // Capped logits of one final-norm hidden row, equal to forward_rows' logits
    // for that row. false = unsupported.
    virtual bool head_logits(std::span<const float> hidden, std::vector<float>& logits) {
        (void)hidden; (void)logits;
        return false;
    }
    virtual std::unique_ptr<StreamState> create_stream() = 0;
    virtual std::unique_ptr<StreamState> create_stream(std::uint32_t admitted_context) {
        (void)admitted_context;
        return create_stream();
    }
    [[nodiscard]] virtual const ModelConfig& config() const = 0;
    [[nodiscard]] virtual strix::serve::Json identity() const = 0;
    // Returns a resident read-only alias only for this exact host matrix.
    // An empty result requests the explicit copy fallback.
    [[nodiscard]] virtual LinearView resident_embedding(LinearView source) const {
        (void)source;
        return {};
    }
    [[nodiscard]] virtual LinearView resident_head(LinearView source) const {
        return resident_embedding(source);
    }
};
class DraftSession : public gemma4::DraftContext {
public:
    virtual std::array<std::uint32_t, 7> propose(std::uint32_t anchor, std::uint64_t position) = 0;
    // Optional hooks for autoregressive drafters (e4b-eagle). Defaults keep DFlash behaviour and bytes unchanged.
    // The admitted window k is known before drafting, because chained draft cost grows with depth.
    virtual std::array<std::uint32_t, 7> propose_window(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) {
        (void)k;
        return propose(anchor, position);
    }
    // Called once after creation with the request's own stream (drafters that read committed target KV).
    virtual void attach(StreamState& state) { (void)state; }
    // Request RowPlan::export_hidden on this stream's rows (post-final-norm tap, zero cost when false).
    [[nodiscard]] virtual bool wants_hidden() const { return false; }
    // false: stage_rows ignores its taps, so prompt prefill may skip copying them.
    [[nodiscard]] virtual bool wants_taps() const { return true; }
    // true: a drafter whose cost grows per depth from a small base (MTP). Only such sessions are planned, probed
    // or measured by the multi-stream auto policy; it still admits them only on their measured cost. Default
    // false keeps DFlash (fixed block cost) C1-only exactly as before.
    [[nodiscard]] virtual bool multi_stream_candidate() const { return false; }
    // Optional: start from a committed prefix restored without staging its rows. tokens = the committed tokens
    // [0, F) followed by the token after them; hidden = the final hidden row of position F-1 (empty when
    // wants_hidden() is false). Must leave the session as staging and publishing those rows would. false = unsupported.
    virtual bool seed(std::span<const std::uint32_t> tokens, std::span<const float> hidden) {
        (void)tokens; (void)hidden; return false;
    }
    // Drafter-confidence-gated window (ConfidenceConfig): the session drafts in chunks up to kmax depths, stopping
    // once the cumulative top-1 probability falls below tau, and returns the window to verify (<= drafted <= kmax)
    // plus the depths it drafted. Only confidence_capable() sessions are asked; the default keeps propose_window.
    struct Confident {
        std::array<std::uint32_t, 7> tokens{};
        std::uint32_t window = 0, drafted = 0;
        bool rule = true; // false: the window did not come from the confidence rule (prompt-lookup proposal)
    };
    [[nodiscard]] virtual bool confidence_capable() const { return false; }
    // The window configuration this session runs with: an env-explicit config as is, a per-drafter (default ON) one
    // resolved from the drafter's config file / builtin default. Non-capable sessions never run the window.
    [[nodiscard]] virtual ConfidenceConfig resolve_confidence(const ConfidenceConfig& config) const {
        return confidence_capable() ? config : ConfidenceConfig{};
    }
    virtual Confident propose_confident(std::uint32_t anchor, std::uint64_t position, std::uint32_t kmax,
                                        const ConfidenceConfig& config) {
        (void)config;
        return {propose_window(anchor, position, kmax), kmax, kmax};
    }
    // Committed rows plus the token that follows each row (prompt[j+1] in prefill, verdict.emitted[j] in decode)
    // and, when wanted, their hidden rows. Same staging/publication protocol as stage().
    virtual void stage_rows(std::span<const float> taps, std::span<const float> hidden,
                            std::span<const std::uint32_t> next_tokens, std::span<const std::uint64_t> positions) {
        (void)hidden;
        (void)next_tokens;
        stage(taps, positions);
    }
    // The coordinator stages accepted taps before target commit, then publishes
    // draft context. Device sessions override these with C's atomic stage API.
    virtual void stage(std::span<const float> taps, std::span<const std::uint64_t> positions) {
        pending_taps_.assign(taps.begin(),taps.end());
        pending_positions_.assign(positions.begin(),positions.end());
    }
    virtual void publish() { append(pending_taps_,pending_positions_); discard(); }
    virtual void discard() noexcept { pending_taps_.clear(); pending_positions_.clear(); }
    // Session-specific receipt fields (prompt lookup); default adds nothing, so other sessions' receipts are unchanged.
    virtual void report(strix::serve::Json& receipt) const { (void)receipt; }
private:
    std::vector<float> pending_taps_;
    std::vector<std::uint64_t> pending_positions_;
};
using DraftFactory = std::function<std::unique_ptr<DraftSession>()>;

enum class DraftMode { off, on, automatic };
struct Sampling {
    float temperature = 0.0F, top_p = 1.0F;
    std::uint32_t top_k = 0;
    std::uint64_t seed = 0, draws = 0;
    void validate() const;
};
// splitmix64-cdf-v1: state=seed+draws*0x9e3779b97f4a7c15; one 53-bit uniform
// per sampled token. Descending capped logit, low-ID ties; top-k then top-p;
// declared FP32 exp weights, FP64 ordered CDF. Greedy consumes no random draws.
std::uint32_t select_token(std::span<const float> logits, const Sampling& sampling, std::uint64_t& draws);
// Ids in select_token's candidate order (descending capped logit, low-ID ties; the first top_k when set), via
// distinct integer keys instead of a comparator sort over the whole vocabulary.
std::vector<std::uint32_t> candidate_order(std::span<const float> logits, std::uint32_t top_k = 0);

struct Request {
    std::vector<std::uint32_t> prompt, stop_ids{1, 106};
    std::size_t max_tokens = 256;
    Sampling sampling;
    DraftMode dflash = DraftMode::automatic;
    std::uint32_t draft_k = 7;
};
[[nodiscard]] strix::serve::Json effective_config(const Backend& backend, const Request& request,
    bool drafter_available, std::size_t streams = 1);
// Private cost partition: identical model/assets and prompt. This is stricter
// than pooling unrelated languages; the key is never published in receipts.
[[nodiscard]] std::string measurement_key(const Backend& backend, const Request& request);
enum class Status { running, stopped, length, cancelled, failed };
// A committed target state kept for reuse: its tokens [0, F), a fork of the state at F and the final hidden row of
// position F-1 (for drafters that read it; empty otherwise). prefill_written: every row was written by prefill_rows.
struct PrefixSnapshot {
    std::vector<std::uint32_t> tokens;
    std::shared_ptr<const StreamState> state;
    std::vector<float> hidden;
    bool prefill_written = false;
};
struct Result {
    Status status = Status::running;
    std::vector<std::uint32_t> ids;
    std::string error, digest; // digest is final on completion/cancellation; receipt() also supports live inspection
    std::uint64_t frontier = 0, rng_draws = 0;
    dflash::Acceptance acceptance;
    double setup_ms = 0, prefill_ms = 0, decode_ms = 0, draft_ms = 0, context_ms = 0, verify_ms = 0, commit_ms = 0;
    std::uint64_t target_calls = 0, target_rows = 0;
    std::size_t peak_streams = 1;
    std::array<std::uint64_t, 8> windows{};
    std::uint64_t probes = 0;
    std::uint64_t multistream_auto_cycles = 0, multistream_auto_proposals = 0;
    std::uint32_t multistream_auto_max_window = 0; // largest auto window drafted while >1 stream was live
    std::uint64_t cached_tokens = 0; // prompt rows restored from a prefix snapshot instead of prefilled
    std::uint64_t prefill_frontier = 0; // rows [0, prefill_frontier) were all written by prefill_rows
    // Drafter-confidence-gated C1 cycles (STRIX_G4_MTP_CONF): cycles, depths drafted, drafted depths not verified.
    std::uint64_t confidence_cycles = 0, confidence_drafted = 0, confidence_unverified = 0;
    // Cycles that ran the cost-aware long-context rule, and the stream's last measured draft-ms-per-depth / one-row ratio.
    std::uint64_t confidence_long_cycles = 0;
    double confidence_draft_ratio = 0;
    // Prompt prefill chunks per engaged prefill lever (receipt "prefill_levers", only when any ran).
    std::map<std::string, std::uint64_t> prefill_levers;
    // PINNED_TAPS: how the last prompt chunk delivered target taps ("skipped (no consumer)" or "pinned-async").
    std::string prefill_taps;
    // Host time of prompt prefill chunks with the device idle: the backend's own share plus the
    // coordinator's work around each call (checks, drafter staging, commit).
    std::uint64_t prefill_calls = 0;
    double prefill_host_ms = 0;
    // PREFILL_BATCH: this request's chunks that shared a prefill call with other streams, and the widest such call.
    std::uint64_t prefill_batched_calls = 0, prefill_batch_streams = 0;
};
const char* status_name(Status status);

// Evidence is separated by live-stream count and k, never extrapolated from C1
// to C4. Full cycle times include drafting, target, context update and commit.
class CostModel {
public:
    void observe(std::size_t streams, std::uint32_t k, double ms, std::size_t emitted,
        std::uint64_t position = 0, const std::string& workload = {});
    [[nodiscard]] std::uint32_t choose(std::size_t streams, std::uint32_t max_k,
        std::uint64_t position = 0, const std::string& workload = {}) const;
private:
    struct Bin { double n = 0, time = 0, time2 = 0, tokens = 0, tokens2 = 0; };
    std::map<std::string,std::array<std::array<Bin,8>,8>> bins_;
};

class Coordinator {
public:
    explicit Coordinator(Backend& backend, DraftFactory drafts = {}, std::shared_ptr<CostModel> costs = {});
    ~Coordinator();
    // A completed slot may be replaced; generation IDs never repeat after reuse. With `prefix` (its tokens a prefix
    // of the prompt's first prompt.size()-1 tokens) the stream starts from a fork of that state and prefills only
    // the rest; Result::cached_tokens says how many rows it skipped (0 when the backend or drafter cannot reuse it).
    std::size_t submit(Request request, std::shared_ptr<const PrefixSnapshot> prefix = {});
    // The slot's committed state as a snapshot (nullptr when the backend cannot fork). Between steps only.
    [[nodiscard]] std::shared_ptr<PrefixSnapshot> snapshot(std::size_t slot) const;
    void cancel(std::size_t slot);
    // At most one forward_rows call, with <=8 total causal/independent rows.
    bool step();
    // At most `streams` many-row prefill chunks per step, round-robin across slots (0 = every eligible stream,
    // the default). Decoding streams still step every call; outputs are unchanged.
    void set_prefill_limit(std::size_t streams) { prefill_limit_ = streams; }
    // At most `rows` prompt rows per many-row prefill chunk while another stream is decoding (0 = whole chunks, the
    // default), when the backend's prefill rows equal its decode rows (otherwise whole chunks). Outputs are unchanged.
    void set_prefill_budget(std::size_t rows) { prefill_budget_ = rows; }
    void run();
    [[nodiscard]] const Result& result(std::size_t slot) const;
    [[nodiscard]] strix::serve::Json receipt(std::size_t slot) const;
private:
    struct Slot;
    Backend& backend_;
    DraftFactory drafts_;
    std::shared_ptr<CostModel> costs_;
    MultiStreamPolicy multi_;
    ConfidenceConfig confidence_ = ConfidenceConfig::from_environment();
    std::vector<std::unique_ptr<Slot>> slots_;
    std::uint64_t generation_ = 0, steps_ = 0;
    std::size_t prefill_limit_ = 0, prefill_cursor_ = 0, prefill_budget_ = 0;
};

} // namespace strix::gemma4::engine
