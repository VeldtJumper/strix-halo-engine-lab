#pragma once

#include "g4_identity.hpp"
#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::gemma4 {

enum class WeightFormat { bf16, f32, q8_0, q4_k, q6_k, exl3_mul1 };
enum class AttentionKind { sliding, global };

// Row-major [outputs, inputs], except the explicitly tiled EXL3 format.
// Ordinary strides are bytes; their quantization blocks never cross rows.
// The owning model outlives views and every backend that uses them. Norms are direct multipliers.
struct LinearView {
    const std::uint8_t* data = nullptr;
    std::uint32_t outputs = 0, inputs = 0;
    std::size_t row_bytes = 0;
    WeightFormat format = WeightFormat::bf16;
};
// EXL3 stores signed scale vectors followed by a tiled core, not independent rows.
STRIX_G4_HD inline std::size_t weight_storage_bytes(LinearView v) {
    return std::size_t(v.outputs) * v.row_bytes +
           (v.format == WeightFormat::exl3_mul1 ? 2ULL * (v.inputs + v.outputs) : 0ULL);
}
struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> shape;
    LinearView view;
    std::string source_hash;
};
struct LayerConfig {
    AttentionKind attention = AttentionKind::sliding;
    std::uint32_t head_dim = 256, producer = 0;
};
struct ModelConfig {
    std::uint32_t hidden = gemma4::hidden, layers = gemma4::layers, inter = gemma4::inter;
    std::uint32_t vocab = gemma4::vocab, heads = gemma4::heads, kv_heads = gemma4::kv_heads;
    std::uint32_t ple_dim = gemma4::ple_dim, window = gemma4::window, max_positions = gemma4::max_positions;
    float eps = norm_eps;
    std::vector<LayerConfig> layer;
};
struct ModelManifest {
    std::string identity = arithmetic_id, model_hash, rope_hash, cache_dtype = "F32";
    ModelConfig config;
    std::vector<TensorRecord> tensors;
    std::vector<std::string> ignored_tensors;
    std::int32_t gguf_file_type = -1; // GGUF general.file_type (llama.cpp ftype, 15 = MOSTLY_Q4_K_M); -1: not GGUF / absent
    // Keeps mapped weights alive without exposing backend globals or operating-system handles.
    std::vector<std::shared_ptr<const void>> storage;
};
struct RopeTables {
    // [position, pair, cos/sin], sliding 128 pairs and global 256 (including inactive ones).
    std::span<const float> sliding, global;
    std::uint32_t positions = 0;
    std::string hash;
};

class KvTxn {
public:
    virtual ~KvTxn() = default;
    [[nodiscard]] virtual std::uint64_t frontier() const = 0; // first unprocessed position in immutable snapshot
    // Opt-in multi-pass overlay. Reserve once before evaluation; no publication until commit.
    virtual void reserve_rows(std::size_t count) { (void)count; throw std::runtime_error("transaction append unsupported"); }
    // Publish exactly this processed prefix in position order, once. Unpublished overlay is discarded.
    // Prefix counts refer to target INPUT rows (accepted drafts + anchor), never the correction token.
    virtual void commit(std::size_t prefix_count) = 0;
};
struct RowPlan {
    std::uint64_t stream = 0, generation = 0, position = 0;
    std::uint32_t token = 0;
    KvTxn* transaction = nullptr;
    std::size_t output_row = 0;
    bool active = true; // inactive padding cannot touch state or arithmetic of peers
    bool export_hidden = false; // opt-in post-final-norm copy; no extra arithmetic
    // Opt-out exports: a backend may leave logits (taps) out of the result when
    // no row of the call asks for them. Arithmetic and state never depend on these.
    bool export_logits = true, export_taps = true;
};
struct ForwardResult {
    std::size_t rows = 0;
    std::uint32_t vocab = 0, tap_width = 0;
    std::vector<float> logits; // [output_row, vocab], capped
    std::vector<float> taps;   // [output_row, 5, hidden], block 5/11/17/23/35, after scalar
    std::vector<float> hidden; // [output_row, hidden], empty unless requested; unrequested rows are zero
    // Optional: the greedy winner of each output row by the shared better() tie
    // rule over the capped logits (empty when the backend does not provide it).
    std::vector<std::uint32_t> selected;
    // The capped logits and the taps passed the backend's device finite fence;
    // logits (taps) are empty when no row of the call exported them.
    bool device_checked = false;
};
class Target {
public:
    virtual ~Target() = default;
    virtual ForwardResult forward_rows(std::span<const RowPlan> rows) = 0;
};
class DraftContext {
public:
    virtual ~DraftContext() = default;
    // Row-major taps in the frozen order. Only committed target rows may be appended, once.
    virtual void append(std::span<const float> taps, std::span<const std::uint64_t> positions) = 0;
};
// Read-only view of one KV-producing target layer, addressed by ABSOLUTE position exactly like KvRead:
// position >= frontier -> transaction overlay row (position - frontier) < candidate_rows; otherwise committed
// slot = global ? position : position % slots with tags[slot] == position (else missing). Row = [kv_heads][head_dim]
// FP32. Device memory for device streams, host memory for host streams; valid until the next commit/transaction.
struct KvView {
    const float* k = nullptr;
    const float* v = nullptr;
    const std::uint64_t* tags = nullptr;
    std::uint32_t width = 0, slots = 0;
    bool global = false;
    const float* overlay_k = nullptr;
    const float* overlay_v = nullptr;
    std::uint64_t frontier = UINT64_MAX;
    std::uint32_t candidate_rows = 0;
};
struct Verdict {
    std::size_t accepted = 0;
    std::vector<std::uint32_t> emitted;
};

} // namespace strix::gemma4
