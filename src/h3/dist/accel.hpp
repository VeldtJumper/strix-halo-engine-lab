#pragma once
// FAST-class accelerators of the two-node schedule (OFF unless a DistAccel is attached to DeviceDistRank::forward),
// the same objects and rules as the 1-node driver (block DeviceForward accelerated forms):
//   Turbo v4 LoRA (identity h3-turbo-v4-bypass), block::TurboV4 (one per process / rank: its device scratch serves one
//     stream): the rank-64 bf16 bypass branch after the base qkv / out / fc1 linears of every block on the rank's
//     local rows (TurboV4Device::apply), fc1 UNFUSED in checkpoint order ([rows][28672], branch, then the K5 SwiGLU:
//     needs a DeviceModel built with fc1_plain), DiT fc2 from the merged file (the DeviceModel's block source =
//     block::MergedFc2Source), the adaLN E-grid deltas (TurboV4::adaln_deltas, host, identical on both ranks) added in
//     fp32 to every block's and the final layer's adaLN output. The refiner's branches belong to the caller's refiner
//     (rank 0's DeviceForward::refine_text; the refined text is an input of forward()).
//   FirstBlockCache (identity h3-fbc-v1, levers fbc.hpp): one FirstBlockCacheDevice per rank on the rank's stream;
//     after block 0 each rank's chunk sums of its local rows -> (N_b, D_b), exchanged (16 B + state), decide in
//     key-block order on S * 5376 elements (the 1-node rule on the canonical cut); the decision, cold flag and score
//     bits are exchanged and asserted equal; on reuse BOTH ranks skip blocks 1.. and their K/V exchanges, else run and
//     commit. Observe mode: score computed and logged, reuse forced off (anchor refreshed every call).
//   Sparse attention (identity h3-attn-sparse-thr-v1, levers router.hpp + sparse K1, frozen spec in levers router_ref.hpp):
//     needs the rank split Split::sparse(S) (cut 128 floor(S / 256)) in BOTH worlds. Per sparse layer (model call >=
//     dense_calls, layer >= dense_layers, not in dense_layer_mask, non-degenerate shape): K4 -> own key block's kc +
//     (S1, S2) and the local query centroids -> the small stats message (2 x 56 x 128 fp32 + call / layer, asserted
//     equal) -> start the K/V exchange -> mean / var, thresholds -> local LUT part -> local sparse partial -> (RX)
//     remote kc from RX -> remote LUT part -> remote sparse partial -> canonical merge. Other layers: the dense path at
//     the same cut. Model calls are counted per sampling run (begin_run on both ranks); an FBC-reused call does not
//     advance the counter. LUT contract violations count as attention contract failures (pair-wide verdict).
// One DistAccel per DeviceDistRank (its FBC holder is bound to that rank's stream and row count).
// Pair-wide FBC rules (both asserted at every call's decision exchange): reset_fbc() per sampling run on BOTH ranks and
// the same call index on both (run id + call index travel with the block sums; a mismatch throws on both ranks); any
// FBC exception or invalidation on either rank aborts the pair (forward throws after cancelling the communicator, the
// peer fails at its next exchange) and the accelerator is dead: every later forward with it throws (no local-only
// recovery: rebuild the communicator, the ranks and the accelerators together).
#include <cstdint>
#include <memory>
#include <vector>

#include "h3/block/turbo_accel.hpp"
#include "h3/levers/sparse/router_ref.hpp"

namespace strix::h3::dist {

class DistAccel {
public:
    // turbo == nullptr: Turbo off. fbc == false: FirstBlockCache off. Both off: forward() is the dense path, unchanged bits.
    DistAccel(const block::TurboV4* turbo, bool fbc, double fbc_threshold = 0.08, bool fbc_observe = false);
    ~DistAccel();
    DistAccel(const DistAccel&) = delete;
    DistAccel& operator=(const DistAccel&) = delete;

    // Tests: force the FBC decision of the next calls (reuse when the cache is ready / refresh); none = the threshold.
    enum class FbcForce { none, reuse, refresh };
    void set_fbc_force(FbcForce f);
    // Sparse attention on (before the first forward); off by default.
    void set_sparse(const levers::sparse::Params& p);
    // Per sampling run (and before a shape change), on both ranks: drops the FBC cache, next run id, FBC call index 0,
    // sparse model-call counter 0. begin_run() == reset_fbc().
    void reset_fbc();
    void begin_run() { reset_fbc(); }
    // Tests: make the FBC holder's next guarded operation of the next call fail (inside after_block0).
    void inject_fbc_failure_for_test();
    bool failed() const;  // a forward with this accelerator threw: it refuses every later forward

    // Per-forward telemetry (the block package's AccelTaps fields + blocks run).
    struct StepInfo {
        std::vector<std::uint32_t> adaln_idx;  // Turbo: E-grid rows per unique t
        int fbc_action = -1;                   // -1 off, 0 full + commit, 1 reuse, 2 full without commit (non-finite)
        bool fbc_cold = false;
        float fbc_score = 0.0F;
        bool fbc_would_reuse = false;          // observe mode
        std::uint32_t blocks_run = 0;
        std::uint32_t run_id = 0, call_index = 0;  // FBC: sampling run (reset count) and call within it
        // sparse attention (empty / 0 when off)
        bool sparse = false;                       // the forward ran with the sparse identity
        std::uint32_t sparse_call = 0;             // model-call index the dense window used for this forward
        std::uint32_t sparse_layers = 0;           // layers that ran the router + sparse K1
        std::vector<std::uint64_t> lut_digest;     // [block] order-free digest of this rank's LUT lists (sums over ranks == 1 node)
        std::vector<std::uint64_t> lut_entries;    // [block] LUT entries of this rank (sums over ranks == 1 node)
    };
    const StepInfo& last() const;

    struct Impl;
    Impl& impl();

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::h3::dist
