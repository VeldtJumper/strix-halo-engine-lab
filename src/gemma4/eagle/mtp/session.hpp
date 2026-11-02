#pragma once

// engine::DraftSession adapter for the MTP assistant drafter.
// Needs from the target per cycle: the post-final-norm hidden of the last committed row (A1: wants_hidden +
// stage_rows) and read-only committed KV views of target layers 22 (sliding) and 23 (full) (A6: attach + kv_view).
// The admitted k is passed through A1's propose_window. No drafter state besides that one hidden row.
#include "mtp.hpp"
#include "gemma4/engine/coordinator.hpp"

#include <functional>

#if defined(STRIX_G4_DRAFT_ROWS_API)
#define STRIX_G4_MTP_ROWS override
#else
#define STRIX_G4_MTP_ROWS
#endif
#if defined(STRIX_G4_DRAFT_KV_API)
#define STRIX_G4_MTP_KV override
#else
#define STRIX_G4_MTP_KV
#endif
#if defined(STRIX_G4_DRAFT_MULTI_API)
#define STRIX_G4_MTP_MULTI override
#else
#define STRIX_G4_MTP_MULTI
#endif
#if defined(STRIX_G4_DRAFT_CONFIDENCE_API)
#define STRIX_G4_MTP_CONF override
#else
#define STRIX_G4_MTP_CONF
#endif

namespace strix::gemma4::eagle::mtp {

// Returns false when the layer view is unavailable (then the session proposes nothing useful; output stays exact).
using KvSource = std::function<bool(std::uint32_t layer, KvLayer& out)>;

// Last non-shared target layers of each attention type (E4B: 22 sliding, 23 full).
std::pair<std::uint32_t, std::uint32_t> shared_kv_layers(const ModelConfig& target);

class Session final : public engine::DraftSession {
public:
    using Release = std::function<void(std::unique_ptr<Drafter>)>;
    Session(std::unique_ptr<Drafter> drafter, std::pair<std::uint32_t, std::uint32_t> kv_layers, Release release = {});
    ~Session() override;
    std::array<std::uint32_t, 7> propose(std::uint32_t anchor, std::uint64_t position) override;
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override;
    void stage(std::span<const float> taps, std::span<const std::uint64_t> positions) override;
    void publish() override;
    void discard() noexcept override;
    std::array<std::uint32_t, 7> propose_window(std::uint32_t anchor, std::uint64_t position, std::uint32_t k) STRIX_G4_MTP_ROWS;
    [[nodiscard]] bool wants_hidden() const STRIX_G4_MTP_ROWS { return true; }
    // stage_rows keeps only the last final hidden row; target taps are never read.
    [[nodiscard]] bool wants_taps() const STRIX_G4_MTP_ROWS { return false; }
    // Cost grows per depth from a small base (~0.56 ms/depth on E4B): eligible for multi-stream auto,
    // still admitted only on its measured draft cost and the measured batch step.
    [[nodiscard]] bool multi_stream_candidate() const STRIX_G4_MTP_MULTI { return true; }
#if defined(STRIX_G4_DRAFT_CONFIDENCE_API)
    // Drafter-confidence-gated window (engine::ConfidenceConfig): draft config.first depths, extend by config.step
    // (Drafter::extend, no recomputation on device) while the cumulative top-1 probability >= tau and depth < kmax;
    // window = leading depths with cumulative probability >= tau (>= 1; 0 when prob_1 < min1). Same guards as
    // propose_window, except that an unavailable draft verifies nothing (window 0, drafted 0).
    [[nodiscard]] bool confidence_capable() const STRIX_G4_MTP_CONF { return true; }
    // Per-drafter defaults when the env leaves them open: config.json confidence_* keys, else the builtin tau for
    // these weights (builtin_confidence_tau), else ConfidenceConfig::builtin_tau.
    [[nodiscard]] engine::ConfidenceConfig resolve_confidence(const engine::ConfidenceConfig& config) const STRIX_G4_MTP_CONF {
        const auto& c = drafter_->config();
        return config.resolved(c.confidence_tau, c.confidence_max_k, c.confidence_first, c.confidence_builtin);
    }
    Confident propose_confident(std::uint32_t anchor, std::uint64_t position, std::uint32_t kmax,
                                const engine::ConfidenceConfig& config) STRIX_G4_MTP_CONF;
#endif
    void stage_rows(std::span<const float> taps, std::span<const float> hidden, std::span<const std::uint32_t> next_tokens,
                    std::span<const std::uint64_t> positions) STRIX_G4_MTP_ROWS;
    // The session state keeps only the last committed row's final hidden, its position and the token after it.
    bool seed(std::span<const std::uint32_t> tokens, std::span<const float> hidden) override;
#if defined(STRIX_G4_DRAFT_KV_API)
    void attach(engine::StreamState& state) override;
#endif
    void set_source(KvSource source) { source_ = std::move(source); }
    [[nodiscard]] const Draft& last_draft() const { return last_; }
private:
    // Draft inputs of a cycle at (anchor, position); false when the previous row's hidden or the KV views are missing.
    bool inputs(std::uint32_t anchor, std::uint64_t position, Inputs& in) const;
    std::unique_ptr<Drafter> drafter_;
    std::pair<std::uint32_t, std::uint32_t> layers_;
    Release release_;
    KvSource source_;
    Draft last_;
    std::vector<float> hidden_, pending_hidden_;
    std::uint64_t position_ = 0, pending_position_ = 0; // position of the row whose hidden is kept
    std::uint32_t next_ = 0, pending_next_ = 0;
    bool have_ = false, staged_ = false;
};

// Host reference sessions (KV views from the host target via A6). Weights/tables/embedding outlive the factory.
[[nodiscard]] engine::DraftFactory host_drafts(const Weights& weights, Tables tables, LinearView target_embedding,
                                               const ModelConfig& target);
// Device sessions (strix_gemma4_eagle_device): shared DeviceModel, resident target embedding when available.
[[nodiscard]] engine::DraftFactory device_drafts(const Weights& weights, Tables tables, LinearView host_embedding,
                                                 engine::Backend* backend, bool prepare = true);

} // namespace strix::gemma4::eagle::mtp
