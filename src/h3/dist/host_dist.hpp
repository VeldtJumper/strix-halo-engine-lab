#pragma once
// Host reference of the two-node schedule (D1): the DiT block stack on one rank's rows, with the per-block K/V
// exchange through a KvExchange and the split-KV attention (local block partial while the exchange is in flight,
// remote block partial after finish, canonical merge). world 1 runs the same code on all rows with both key blocks
// local, so one and two ranks give the same bits by construction (G3).
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "h3/block/host_ref.hpp"
#include "h3/dist/kv_exchange.hpp"
#include "h3/dist/schedule.hpp"

namespace strix::h3::dist {

// Per-forward tables shared by every block (identical on both ranks: computed from the shared layout and plan).
struct ForwardTables {
    std::vector<float> t_emb;           // [n_t][basis]
    std::uint32_t n_t = 0;
    block::RopeTable rope;              // [S][48]
    std::vector<std::uint32_t> mod_row; // [S]
    static ForwardTables build(const block::GlobalWeights& g, const block::PackedLayout& L, const block::ForwardPlan& p);
};

// Per DiT block: FNV-1a of the residual rows of each key block after the block (0 where the rank does not own it).
struct DistTrace {
    std::vector<std::array<std::uint64_t, 2>> hash;
    std::vector<double> block_s, finish_wait_s;
};

// Runs DiT blocks [0, nblocks) in place on x = the rank's rows [rows][5376] (role.row0 .. + role.rows). world 2
// needs `ex`; world 1 ignores it. Double-buffered TX / RX messages (block parity), one exchange in flight.
void dist_blocks_host(block::BlockSource& blocks, std::uint32_t nblocks, const ForwardTables& t, const Split& sp, const Role& role,
    std::vector<std::uint16_t>& x, block::LinearOp& lin, KvExchange* ex, DistTrace* trace = nullptr);

// Thread-safe block source for ranks sharing one process (keeps the last `keep` blocks).
class SharedBlockSource final : public block::BlockSource {
public:
    SharedBlockSource(block::BlockSource& inner, unsigned keep = 2) : inner_(inner), keep_(keep) {}
    std::shared_ptr<const block::BlockWeights> block(std::uint32_t i) override;
    std::uint32_t num_blocks() const override { return inner_.num_blocks(); }

private:
    block::BlockSource& inner_;
    unsigned keep_;
    std::mutex mu_;
    std::vector<std::pair<std::uint32_t, std::shared_ptr<const block::BlockWeights>>> cache_;
};

}  // namespace strix::h3::dist
