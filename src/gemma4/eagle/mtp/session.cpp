#include "session.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix::gemma4::eagle::mtp {
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("mtp session: ") + message);
}
}

std::pair<std::uint32_t, std::uint32_t> shared_kv_layers(const ModelConfig& t) {
    std::uint32_t sliding = UINT32_MAX, full = UINT32_MAX;
    for (std::uint32_t l = 0; l < t.layer.size(); ++l)
        if (t.layer[l].producer == l) (t.layer[l].attention == AttentionKind::global ? full : sliding) = l;
    require(sliding != UINT32_MAX && full != UINT32_MAX, "target has no KV-producing sliding/full layer");
    return {sliding, full};
}

Session::Session(std::unique_ptr<Drafter> drafter, std::pair<std::uint32_t, std::uint32_t> kv_layers, Release release)
    : drafter_(std::move(drafter)), layers_(kv_layers), release_(std::move(release)) {
    require(bool(drafter_), "missing drafter");
}
Session::~Session() {
    if (release_ && drafter_) release_(std::move(drafter_));
}
std::array<std::uint32_t, 7> Session::propose(std::uint32_t anchor, std::uint64_t position) {
    return propose_window(anchor, position, max_depth);
}
bool Session::inputs(std::uint32_t anchor, std::uint64_t position, Inputs& in) const {
    if (!have_ || position != position_ + 1 || anchor != next_ || !source_) return false;
    in.anchor = anchor;
    in.position = position;
    in.hidden = hidden_;
    return source_(layers_.first, in.sliding) && source_(layers_.second, in.full);
}
std::array<std::uint32_t, 7> Session::propose_window(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) {
    std::array<std::uint32_t, 7> out{};
    last_ = {};
    k = std::min(k, max_depth);
    // Exact verification makes any proposal safe; without the previous row's hidden or the KV views, propose zeros.
    Inputs in;
    if (!k || !inputs(anchor, position, in)) return out;
    drafter_->set_statistics(0); // ids only: nothing consumes prob/margin here
    last_ = drafter_->draft(in, k);
    for (std::uint32_t i = 0; i < k; ++i) out[i] = last_.tokens[i];
    return out;
}
#if defined(STRIX_G4_DRAFT_CONFIDENCE_API)
engine::DraftSession::Confident Session::propose_confident(std::uint32_t anchor, std::uint64_t position, std::uint32_t kmax,
                                                           const engine::ConfidenceConfig& config) {
    Confident out;
    last_ = {};
    kmax = std::min(kmax, max_depth);
    Inputs in;
    if (!kmax || !inputs(anchor, position, in)) return out;
    // Numerics are draft(in, d) for the final d whether reached at once or by extension (Drafter::extend contract);
    // the window only selects how many of these tokens the greedy verification sees.
    auto depth = config.next_depth({}, kmax);
    drafter_->set_statistics(statistic_probability); // the window reads prob only
    last_ = drafter_->draft(in, depth);
    for (auto next = config.next_depth({last_.prob.data(), depth}, kmax); next > depth;
         next = config.next_depth({last_.prob.data(), depth}, kmax)) {
        last_ = drafter_->extend(next);
        depth = next;
    }
    out.drafted = depth;
    out.window = std::min(config.window({last_.prob.data(), depth}), depth);
    for (std::uint32_t i = 0; i < out.window; ++i) out.tokens[i] = last_.tokens[i];
    return out;
}
#endif
void Session::stage_rows(std::span<const float>, std::span<const float> hidden, std::span<const std::uint32_t> next_tokens,
                         std::span<const std::uint64_t> positions) {
    require(!staged_, "previous stage not published/discarded");
    if (positions.empty()) return;
    const auto B = drafter_->config().backbone;
    require(hidden.size() == positions.size() * B && next_tokens.size() == positions.size(),
            "final hidden rows required (RowPlan::export_hidden) with one next token per row");
    pending_hidden_.assign(hidden.end() - std::ptrdiff_t(B), hidden.end());
    pending_position_ = positions.back();
    pending_next_ = next_tokens.back();
    staged_ = true;
}
void Session::append(std::span<const float>, std::span<const std::uint64_t>) {
    throw std::runtime_error("mtp session: needs the final hidden row; requires the staged-rows draft API (stage_rows)");
}
void Session::stage(std::span<const float>, std::span<const std::uint64_t> positions) {
    if (positions.empty()) return;
    throw std::runtime_error("mtp session: legacy stage has no hidden rows; requires the staged-rows draft API (stage_rows)");
}
bool Session::seed(std::span<const std::uint32_t> tokens, std::span<const float> hidden) {
    if (staged_ || tokens.size() < 2 || hidden.size() != drafter_->config().backbone) return false;
    hidden_.assign(hidden.begin(), hidden.end());
    position_ = tokens.size() - 2;
    next_ = tokens.back();
    have_ = true;
    return true;
}
void Session::publish() {
    if (!staged_) return;
    hidden_.swap(pending_hidden_);
    position_ = pending_position_;
    next_ = pending_next_;
    have_ = true;
    staged_ = false;
}
void Session::discard() noexcept { staged_ = false; }

#if defined(STRIX_G4_DRAFT_KV_API)
void Session::attach(engine::StreamState& state) {
    source_ = [&state](std::uint32_t layer, KvLayer& out) {
        KvView v; // contracts/g4_types.hpp (A6)
        // Drafting happens before the cycle's candidate rows exist: committed snapshot only (positions < p).
        if (!state.kv_view(layer, nullptr, v)) return false;
        out = {v.k, v.v, v.tags, v.width, v.slots, v.global, v.overlay_k, v.overlay_v, v.frontier, v.candidate_rows};
        return true;
    };
}
#endif

engine::DraftFactory host_drafts(const Weights& weights, Tables tables, LinearView target_embedding, const ModelConfig& target) {
    weights.validate();
    tables.validate();
    const auto kv_layers = shared_kv_layers(target);
    return [&weights, tables, target_embedding, kv_layers] {
        return std::make_unique<Session>(std::make_unique<HostDrafter>(weights, tables, target_embedding), kv_layers);
    };
}

} // namespace strix::gemma4::eagle::mtp
