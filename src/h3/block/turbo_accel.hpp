#pragma once
// Accelerated forms of the DiT forward (FAST-class levers, OFF by default; the dense path is unchanged without them).
//
// h3-turbo-v4-bypass (levers turbo-lora.md section 7): the Turbo v4 LoRA at strength s (preset 0.70) as the
// reference's default bypass mode -
//   * rank-64 bf16 side branch y += s up(down(x)) right after the base linear, before anything reads y:
//     DiT qkv / out / fc1 (fc1 UNFUSED: [rows][28672] in checkpoint order, then the K5 SwiGLU) and the token refiner's
//     qkv / out / fc1 / fc2 (levers TurboV4Device::apply);
//   * DiT fc2: no branch, its int8 weights + scales come from the offline merge (merged_fc2_s070.safetensors, strength
//     baked in; MergedFc2Source refuses a strength mismatch);
//   * adaLN injection: for every unique-t row m, idx = nearest adaln_t_table row of t_emb[m] (basis width), st =
//     E-grid row idx (bf16 [2688]), x[m] += Bs (A st) with Bs = bf16(B s) in fp32 (levers adaln_delta), for the 50
//     block adaLN modules and the final layer's.
// The sampler of the identity is flow Euler, 8 steps, "simple" sigmas (runner side).
//
// h3-fbc-v1 (FirstBlockCache, levers fbc.hpp): after block 0 the relative change of the block-0 residual delta
// against the anchor decides reuse (score <= threshold: skip blocks 1..49, h_last = h1 + tail) or a full forward.
// One FirstBlockCacheDevice per stream (bound to its first begin's stream), reset per sampling run. The decision uses
// the canonical key-block split of the rows, so one node and two nodes decide on the same fp64 sums.
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "h3/block/host_ref.hpp"
#include "h3/block/weights.hpp"

namespace strix::h3::levers::lora {
class TurboV4Device;
}

namespace strix::h3::block {

class SparseAccel;  // device_forward.hpp

// Base blocks with blocks.N.mlp.fc2 (int8 codes + per-channel scales) taken from the merged file.
class MergedFc2Source final : public BlockSource {
public:
    // expect_strength: the merged file's metadata "strength" must equal it (the delta is baked in).
    MergedFc2Source(const SafeTensors& base, const SafeTensors& merged, std::uint32_t basis, float expect_strength, LoadPolicy pol = {});
    std::shared_ptr<const BlockWeights> block(std::uint32_t i) override;
    std::uint32_t num_blocks() const override { return n_; }

private:
    const SafeTensors& base_;
    const SafeTensors& merged_;
    std::uint32_t basis_, n_;
    LoadPolicy pol_;
};

// The Turbo v4 bypass identity: resident branches (device) + the adaLN injection data (host). The same holder carries
// the step-distilled LoRA identities (levers lora_file.hpp table: h3-lora-fasth3-6step-v1, h3-lora-turbo-v12-v1):
// rank-generic branches on DiT qkv / out / fc1 / fc2 and the refiner with per-branch scale strength x alpha / rank, no
// adaLN part (has_adaln() false), the base fc2 weights + a run-time fc2 branch (has_fc2()); their full-weight .diff
// deltas are applied to the host weights before upload (LoraDiffs).
class TurboV4 {
public:
    static constexpr const char* kIdentity = "h3-turbo-v4-bypass";
    // lora: the Turbo LoRA safetensors (bf16 lora_A / lora_B); egrid: h3_silu_temb_grid.safetensors.
    TurboV4(const std::string& lora, const std::string& egrid, float strength, std::uint32_t max_rows, std::uint32_t dit_blocks,
        std::uint32_t refiner_blocks);
    // A step-LoRA identity id from file (every DiT block and refiner branch the file holds; a DiT block without its qkv /
    // out / fc1 / fc2 branch throws).
    static std::unique_ptr<TurboV4> step_lora(const std::string& id, const std::string& file, float strength, std::uint32_t max_rows,
        std::uint32_t dit_blocks, std::uint32_t refiner_blocks);
    ~TurboV4();
    TurboV4(const TurboV4&) = delete;
    TurboV4& operator=(const TurboV4&) = delete;

    float strength() const { return strength_; }
    std::size_t device_bytes() const;
    const levers::lora::TurboV4Device& device() const { return *dev_; }
    const std::string& identity() const { return identity_; }
    bool has_adaln() const { return !ada_.empty(); }  // Turbo v4: E-grid adaLN injection
    bool has_fc2() const { return has_fc2_; }          // step LoRAs: run-time DiT fc2 branch after the int8 fc2 GEMM
    std::uint32_t max_rank() const { return max_rank_; }

    // Per forward (host, deterministic; identical on every rank): for the unique-t rows t_emb [nt][basis] the nearest
    // table rows idx [nt] and the adaLN deltas of every block [blocks][nt][96768] and of the final layer [nt][10752].
    struct AdalnDeltas {
        std::vector<std::uint32_t> idx;
        std::vector<float> blocks, final;
        std::uint32_t nt = 0, block_out = 0, final_out = 0;
    };
    AdalnDeltas adaln_deltas(const GlobalWeights& g, const std::vector<float>& t_emb, std::uint32_t nt) const;

private:
    struct Ada {
        std::vector<std::uint16_t> a, bs;  // A [16][2688], Bs = bf16(B s) [out][16]
        std::uint32_t out = 0;
    };
    TurboV4(float strength, std::uint32_t max_rows, std::string identity);
    float strength_;
    std::string identity_;
    bool has_fc2_ = false;
    std::uint32_t max_rank_ = 64;
    std::unique_ptr<levers::lora::TurboV4Device> dev_;
    std::vector<std::uint16_t> egrid_;  // [1025][2688] bf16
    std::vector<Ada> ada_;              // per DiT block
    Ada final_ada_;
};

// One LoRA branch of a step-LoRA file as read on the host (site = levers lora::Site; scale = strength x alpha / rank).
struct HostLoraBranch {
    std::uint8_t site = 0;
    std::uint32_t block = 0, k = 0, n = 0, rank = 0;
    float scale = 0.0F;
    std::vector<std::uint16_t> a, b;  // bf16 [rank][k], [n][rank]
};
// Every branch of the identity's file in upload order: DiT qkv / out / fc1 / fc2 of each block (required), then the
// refiner branches the file holds (host; the identity must be a run-time-fc2 step LoRA).
void for_each_step_lora_branch(const std::string& id, const std::string& file, float strength, std::uint32_t dit_blocks,
    std::uint32_t refiner_blocks, const std::function<void(const HostLoraBranch&)>& fn);

// The full-weight deltas of a step LoRA ("<module>.diff" / ".diff_b", W = bf16(W + strength x diff) on the loaded
// bf16-valued weight): DiT block norm1 / norm2, final_layer.norm, condition_proj, the patch projections and the heads.
// Every delta name must map to one of these (else the constructor throws); sizes are checked when applied.
class LoraDiffs {
public:
    LoraDiffs(const std::string& file, float strength);
    void apply_global(GlobalWeights& g) const;  // everything outside the DiT blocks
    void apply_block(BlockWeights& b) const;    // blocks.<b.index>.norm1 / norm2
    std::size_t count() const { return d_.size(); }
    std::size_t values() const;

private:
    float s_;
    std::map<std::string, std::vector<float>> d_;  // target tensor name ("<module>.weight" / ".bias") -> delta
};

// Base blocks with the step LoRA's block deltas applied (wraps another source; diffs must outlive it).
class DiffBlockSource final : public BlockSource {
public:
    DiffBlockSource(BlockSource& base, const LoraDiffs& diffs) : base_(base), diffs_(diffs) {}
    std::shared_ptr<const BlockWeights> block(std::uint32_t i) override;
    std::uint32_t num_blocks() const override { return base_.num_blocks(); }

private:
    BlockSource& base_;
    const LoraDiffs& diffs_;
};

// Accelerated-form options of a DeviceForward (all off = the dense forward, unchanged bits).
struct AccelOptions {
    const TurboV4* turbo = nullptr;
    bool fbc = false;          // h3-fbc-v1
    double fbc_threshold = 0.08;
    bool fbc_observe = false;  // compute + log the score every call, never reuse
    SparseAccel* sparse = nullptr;  // h3-attn-sparse-thr-v1 (1 node): DiT attention through the threshold router
};

// Per-forward telemetry of the accelerated forms.
struct AccelTaps {
    std::vector<std::uint32_t> adaln_idx;  // Turbo: chosen E-grid rows per unique t
    int fbc_action = -1;                   // -1 off, 0 full + commit, 1 reuse, 2 full without commit (non-finite)
    bool fbc_cold = false;                 // no anchor yet (first call of the run)
    float fbc_score = 0.0F;                // bf16-valued score (0 on a cold call)
    bool fbc_would_reuse = false;          // observe mode: the decision the threshold would take
};

}  // namespace strix::h3::block
