#pragma once
#include "strix/cost_table_cache.hpp"
#include "strix/mtp_window_controller.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace strix {
// The route and configuration part of the Qwen min-p cost-table policy key (the app fills it once the HTTP route is
// resolved). Only requested inputs are keyed, never values a process resolves from what it observes: the automatic
// HTTP slot capacity follows MemAvailable / GTT free at startup and differs between runs of the same command, so a
// key holding it would miss across processes. `observed_slots` is carried for that reason only and is not keyed.
struct MtpCostTableRoute {
    bool tiled=false,http=false;
    bool batched=false;               // the batched HTTP route was taken (any slot count)
    bool batch_explicit=false;        // --http-batch given
    std::uint32_t batch_cap=0;        // its value (0 when automatic)
    std::size_t cohort=0;             // fixed cohort members
    std::uint64_t kernel_plan=0,execution=0,profiles=0;
    std::string optimization,cli;     // selected optimization profile; effective CLI without prompts/outputs
    std::size_t observed_slots=0;     // resolved capacity (memory dependent): NOT part of the key
};
[[nodiscard]] std::string mtp_cost_table_configuration(const MtpCostTableRoute&);

// Persisted costs of the min-p window controller (MtpMinPWindowController) for one verifier, per context bucket
// (cost_table_cache.hpp). Host-only and deterministic in its inputs, so the verifier and the tests share it.
//
// A session starts cold (the controller as without the feature) and is resolved by the first prepare(), which the
// verifier calls before every draft decision and every verify: a sampled session stays cold for its whole length
// and is never written; a greedy one is seeded from its context bucket's table. Every later prepare() checks the
// bucket again: on a crossing the measurements so far are recorded under the old bucket, the controller's cost
// estimates (not its acceptance calibration) are reset and the new bucket's table is seeded. Only `seedable` rows
// (seedable_rows(): serial and the owners, when every owner passes the policy-selection guard) are ever seeded
// or written. exclude() (a cohort cycle: wall times include other sessions) keeps the session from being written.
// flush() (at the next request and at teardown) records the measured medians (>= 3 samples) and stores.
class MtpCostTableSession {
public:
    MtpCostTableSession(std::filesystem::path directory,CostTableKey key,CostTableOptions options,
        std::vector<std::uint32_t> seedable_rows);
    void prepare(MtpMinPWindowController& controller,std::uint64_t position,bool sampled);
    void exclude() noexcept {no_write_=true;}
    void flush(const MtpMinPWindowController& controller) noexcept;
    enum class State {unresolved,cold,warm};
    [[nodiscard]] State state() const noexcept {return state_;}
    [[nodiscard]] std::optional<std::uint32_t> bucket() const noexcept {
        return state_==State::warm?std::optional<std::uint32_t>(bucket_):std::nullopt;}
    // The table of `bucket` (loaded on first use; one stderr line with its load status).
    [[nodiscard]] CostTableCache& table(std::uint32_t bucket);
    [[nodiscard]] static std::string entry(std::uint32_t rows);
    // The rows a verifier may seed and write: 1 (serial) and every owner, but only when EVERY selectable owner is
    // admitted as serial-equivalent by the existing policy-selection guard; otherwise empty (no table at all).
    [[nodiscard]] static std::vector<std::uint32_t> seedable_rows(const std::vector<std::uint32_t>& owner_rows,
        const std::vector<bool>& admitted);
private:
    void seed(MtpMinPWindowController&,std::uint32_t bucket);
    void record(const MtpMinPWindowController&);
    std::filesystem::path directory_;
    CostTableKey key_;
    CostTableOptions options_;
    std::vector<std::uint32_t> seedable_;
    std::map<std::uint32_t,std::unique_ptr<CostTableCache>> tables_;
    State state_=State::unresolved;
    std::uint32_t bucket_=0;
    bool no_write_=false;
};
} // namespace strix
