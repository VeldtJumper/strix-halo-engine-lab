#pragma once
#include "hip_minimax_dspark.hpp"
#include "strix/minimax/engine.hpp"
#include "strix/minimax/session_cache.hpp"
#include <memory>
namespace strix::minimax::session {
struct RunOptions {
  std::string directory, id = "session", save, load, prompt, suffix, output,
                         control_output;
  std::uint64_t budget = 0, disk_budget = UINT64_MAX, max_parked = 8;
  bool disk_enabled = true;
  unsigned generate = 8, suffix_generate = 64;
  bool draft = false;
};
// Engine-thread only. Fixed speculative width one; no wall-clock controller
// state.
class Runtime {
public:
  Runtime(engine::Engine &, dspark::DeviceDrafter *, const engine::Options &,
          const std::string &executable, RunOptions,
          const std::string &draft_dir = {},
          const std::string &draft_exl3 = {},
          const LoaderSnapshot& draft_snapshot = {});
  void start(std::span<const std::uint32_t>);
  void suffix(std::span<const std::uint32_t>);
  std::vector<std::uint32_t> generate(unsigned, const std::string &dump = {},
      std::span<const std::uint32_t> stops = {},
      const std::function<bool(std::span<const std::uint32_t>)>& publish = {});
  void checkpoint(const std::string &, bool disk);
  void park(const std::string &);
  void restore(const std::string &, bool disk);
  void restore_auto(const std::string&);
  void erase(const std::string &id);
  bool begin_request(const std::string&, std::span<const std::uint32_t>);
  // Tokens generate() can still produce in the active session (0 when the context is full or nothing is active).
  std::uint64_t generation_room() const;
  std::vector<Summary> describe() const { return cache_->describe(options_.disk_enabled); }
  std::uint64_t ram_budget() const { return options_.budget; }
  bool in_ram(const std::string& id) const { return cache_->in_ram(id); }
  bool on_disk(const std::string& id) const { return options_.disk_enabled && cache_->on_disk(id); }
  bool disk_enabled() const { return options_.disk_enabled; }
  // A request's state is in the slot and not yet parked.
  bool active() const { return active_; }
  std::vector<std::string> list() const { return cache_->list(); }
  Stats stats() const { return cache_->stats(); }
  std::string digest();

private:
  engine::Engine &eng_;
  dspark::DeviceDrafter *dd_;
  RunOptions options_;
  Exchange exchange_;
  std::unique_ptr<Cache> cache_;
  std::uint32_t slot_ = 0;
  bool active_ = false;
  Frontier frontier_;
  std::vector<std::uint32_t> history_;
  std::vector<float> logits_;
  std::uint64_t windows_ = 0, verified_ = 0, accepted_ = 0;
  Checkpoint capture(const std::string &);
  std::vector<std::uint32_t> step(std::span<const engine::Row>, std::span<const std::uint32_t> stops = {});
  std::uint32_t draft();
};
bool parse_option(RunOptions &, const std::string &,
                  const std::function<std::string()> &);
void run_cli(Runtime &, const RunOptions &);
} // namespace strix::minimax::session
