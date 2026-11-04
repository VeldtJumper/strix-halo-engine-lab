#pragma once
// MiniMax H3 single-node device forward driver: the residual stream, the glue kernels (K3-K6), the adaLN, embedding,
// token refiner and final layer run on the GPU; the W8A8 linears and the attention are pluggable device ops
// (GemmDeviceLinear / FlashDeviceAttention wrap the gemm / attn packages; host-reference fallbacks copy through the
// host and are meant for small shapes and bitwise checks).
//
// Interfaces (stable for the two-node driver):
//   DeviceLinearOp::prepare(block) once per block (weights resident from then on), run(block index, kind, ...).
//   DeviceAttentionOp::run(q, k, vt, vt_pitch, rows, keys, split, out): Q, K token-major [rows][56][128], V^T
//   [56][128][vt_pitch] (vt_pitch = vt_pitch_for(keys)), out [rows][7168] bf16. split = true: the canonical key-block
//   split key_cuts(keys) (the cut the two-node run uses); false: one key block (refiner).
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "h3/block/host_ref.hpp"
#include "h3/block/turbo_accel.hpp"

namespace strix::h3::block {

enum class LinearKind { qkv = 0, out = 1, fc1 = 2, fc2 = 3 };

// The canonical key-block cut table (shared by the 1-node and 2-node runs): two blocks split at round_down_64(keys / 2)
// (one block when that is 0). Returned as cuts[0] = 0 .. cuts[n] = keys.
inline std::vector<std::uint32_t> key_cuts(std::uint32_t keys, bool split = true) {
    const std::uint32_t cut = (keys / 2U) / 64U * 64U;
    if (!split || cut == 0U) return {0U, keys};
    return {0U, cut, keys};
}
// V^T row pitch (keys per row): a multiple of 8, >= keys.
inline std::uint64_t vt_pitch_for(std::uint32_t keys) { return (static_cast<std::uint64_t>(keys) + 7U) / 8U * 8U; }

// Device W8A8 linear: d_x [m][k] bf16 -> d_y [m][n] bf16 (device pointers), on `stream`.
struct DeviceLinearOp {
    virtual ~DeviceLinearOp() = default;
    virtual void prepare(const std::shared_ptr<const BlockWeights>& b) = 0;
    virtual bool prepared(std::uint32_t block) const = 0;
    virtual void run(std::uint32_t block, LinearKind kind, const std::uint16_t* d_x, std::uint32_t m, std::uint16_t* d_y, void* stream) = 0;
    // true: fc1 returns SwiGLU(fc1) [m][14336] directly (the driver skips K5); false: [m][28672] and the driver runs K5.
    virtual bool fused_swiglu() const { return false; }
    // Rotation groups outside the exact window since the last reset, read in the order of the producer stream
    // (must be 0 after a forward; synchronizes that stream).
    virtual std::uint64_t flush_count(void* /*stream*/) { return 0; }
    virtual void reset_flush_count(void* /*stream*/) {}
};
struct DeviceAttentionOp {
    virtual ~DeviceAttentionOp() = default;
    virtual void run(const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_vt, std::uint64_t vt_pitch, std::uint32_t rows,
        std::uint32_t keys, bool split, std::uint16_t* d_out, void* stream) = 0;
    // Attention input-bound telemetry (attn package bound_rows: (row, head, block) triples where a 64-key step's max
    // score left the qualified domain) since the last reset, read in stream order (synchronizes). 0 when not tracked.
    virtual std::uint64_t bound_rows(void* /*stream*/) { return 0; }
    virtual void reset_bound_rows(void* /*stream*/) {}
};

// h3-attn-sparse-thr-v1 (levers sparse_attention.hpp, FAST-class, 1 node): one per sampling run. op() is the DiT
// attention (split calls through the levers SparseAttention: dense K1 at the lever cut for calls < dense_calls and the
// dense layers, router + sparse K1 + canonical merge otherwise; non-split calls dense). DeviceForward::run brackets
// each forward with begin_forward(video start) / end_forward (abandon_forward on an FBC reuse: not a counted call).
class SparseAccel {
public:
    SparseAccel(float tau, std::uint32_t dense_calls, std::uint32_t dense_layers, const std::string& extra_dense_layers);
    ~SparseAccel();
    SparseAccel(const SparseAccel&) = delete;
    SparseAccel& operator=(const SparseAccel&) = delete;
    static constexpr const char* kIdentity = "h3-attn-sparse-thr-v1";
    void begin_run();
    DeviceAttentionOp& op();
    void begin_forward(std::uint32_t prefix_rows);
    void end_forward(void* stream);
    void abandon_forward();
    // last forward: sparse layers, mean density over them; run totals
    struct Info {
        std::uint32_t sparse_layers = 0, dense_layers = 0;
        double mean_density = 0.0;
        std::uint64_t run_dense_calls = 0, run_sparse_calls = 0;
        std::uint32_t model_call = 0;
    };
    Info info() const;
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Host-reference fallbacks (synchronous copies through the host).
struct HostFallbackLinear final : DeviceLinearOp {
    void prepare(const std::shared_ptr<const BlockWeights>& b) override { blocks[b->index] = b; }
    bool prepared(std::uint32_t block) const override { return blocks.count(block) != 0; }
    void run(std::uint32_t block, LinearKind kind, const std::uint16_t* d_x, std::uint32_t m, std::uint16_t* d_y, void* stream) override;
    std::uint64_t flush_count(void* /*stream*/) override { return host.inexact_groups(); }
    HostLinear host;
    std::map<std::uint32_t, std::shared_ptr<const BlockWeights>> blocks;
};
struct HostFallbackAttention final : DeviceAttentionOp {
    explicit HostFallbackAttention(std::unique_ptr<AttentionOp> a) : host(std::move(a)) {}
    void run(const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_vt, std::uint64_t vt_pitch, std::uint32_t rows,
        std::uint32_t keys, bool split, std::uint16_t* d_out, void* stream) override;
    std::unique_ptr<AttentionOp> host;
};

// The gemm package's device linear (int8 weights resident, fc1 permuted for the fused SwiGLU epilogue).
class GemmDeviceLinear final : public DeviceLinearOp {
public:
    // fused: fc1 permuted for the SwiGLU epilogue (default); false: fc1 in checkpoint order, [rows][28672] output, the
    // driver runs K5 (needed by the Turbo fc1 branch, which acts on the plain fc1 output)
    GemmDeviceLinear(std::uint32_t max_rows, bool fused = true);
    ~GemmDeviceLinear() override;
    void prepare(const std::shared_ptr<const BlockWeights>& b) override;
    bool prepared(std::uint32_t block) const override;
    void run(std::uint32_t block, LinearKind kind, const std::uint16_t* d_x, std::uint32_t m, std::uint16_t* d_y, void* stream) override;
    bool fused_swiglu() const override { return fused_; }
    std::uint64_t flush_count(void* stream) override;
    void reset_flush_count(void* stream) override;
    std::uint64_t resident_bytes() const;
    bool fused_ = true;
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// The attn package's flash attention (split-KV partials + canonical merge over key_cuts()).
class FlashDeviceAttention final : public DeviceAttentionOp {
public:
    FlashDeviceAttention();
    ~FlashDeviceAttention() override;
    void run(const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_vt, std::uint64_t vt_pitch, std::uint32_t rows,
        std::uint32_t keys, bool split, std::uint16_t* d_out, void* stream) override;
    std::uint64_t bound_rows(void* stream) override;
    void reset_bound_rows(void* stream) override;
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

struct DeviceForwardTaps {
    std::vector<std::vector<std::uint16_t>>* blocks = nullptr;  // residual after every block (downloaded)
    std::vector<std::uint16_t>* h_in = nullptr;
    FinalRows* final_rows = nullptr;
    std::uint32_t stop_after_block = 0xffffffffU;  // run blocks 0 .. stop_after_block only (then no final layer)
    std::vector<double>* block_ms = nullptr;       // per-block wall time (synchronized; enables per-block syncs)
    // teacher forcing (tests): after block i the residual is replaced by teacher[i] ([S][5376] bf16) when present
    const std::vector<std::vector<std::uint16_t>>* teacher = nullptr;
    std::uint64_t flushed_groups = 0;  // out: rotation groups outside the exact fixed-point window in this forward
    double host_prep_ms = 0.0;         // out: host-side preparation (patchify, rope table, adaLN plan, uploads)
    double embed_proj_ms = 0.0;        // out: device patch projections + text copy (synchronized)
    double final_ms = 0.0;             // out: final layer + unpatchify (device + host)
    // out: attention input contract, max over blocks / heads of max||q|| max||k|| log2(e) / sqrt(128) (must be <= 2^14,
    // the forward throws otherwise) and the block that attains it
    double attn_score_max = 0.0;
    std::uint32_t attn_score_block = 0;
    std::uint64_t attn_bound_rows = 0;  // out: attention bound_rows telemetry of this forward (must be 0; throws otherwise)
    AccelTaps* accel = nullptr;         // out: Turbo / FBC telemetry (when those forms are on)
};

class DeviceForward {
public:
    explicit DeviceForward(const GlobalWeights& g);
    ~DeviceForward();
    DeviceForward(const DeviceForward&) = delete;
    DeviceForward& operator=(const DeviceForward&) = delete;

    // Same contract as stream_forward_host (stream inputs -> -velocity as bf16 bits). Blocks are fetched from `blocks`
    // only the first time (lin.prepare + resident norms / adaLN); later forwards do not touch the source.
    StreamOutputs run(BlockSource& blocks, const PackedLayout& L, const ForwardPlan& p, const StreamInputs& in,
        DeviceLinearOp& lin, DeviceAttentionOp& att, DeviceForwardTaps* taps = nullptr, const PddContext* pdd = nullptr);

    // condition_proj + token refiner on the device (bf16 linears resident after the first call); text [L][5120] bf16
    // -> [L][5376] bf16 (bitwise == text_refine_host with the same attention arithmetic).
    std::vector<std::uint16_t> refine_text(const std::uint16_t* text, std::uint32_t L, DeviceAttentionOp& att);
    // attention contract score of the last refine_text (max over its blocks / heads; it throws above 2^14)
    double last_refiner_score() const { return last_refiner_score_; }

    // strict: throw when a forward flushed rotation groups (the gemm package's exact-rotation window);
    // default false: the count is returned in DeviceForwardTaps::flushed_groups.
    void set_strict_flush(bool strict);

    // Accelerated forms (turbo_accel.hpp); default: none (the dense forward). Turbo needs a non-fused DeviceLinearOp
    // whose blocks come from a MergedFc2Source; it also applies to refine_text. reset_accel_run(): start of a sampling
    // run (FirstBlockCache anchor cleared; the next forward is a cold, full one).
    void set_accel(const AccelOptions& o);
    void reset_accel_run();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    double last_refiner_score_ = 0.0;
};

}  // namespace strix::h3::block
