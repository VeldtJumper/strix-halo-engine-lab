#pragma once
#include "prefill.hpp"
#include "attention.hpp"
#include "gemma4/kernels/device.hpp"
#include <map>

namespace strix::gemma4::prefill {
// Per-op precision of WMMA linears (g4-wmma-pf-v2 development); see LinearPolicy.
enum class LinearPrecision { fp16, exact, hilo, hilo_k128 };
[[nodiscard]] bool tiled_enabled(); // startup-cached STRIX_G4_PREFILL_TILING
[[nodiscard]] bool prepared_enabled(); // startup-cached STRIX_G4_PREFILL_PREPARE
[[nodiscard]] unsigned direct_mode(); // startup-cached STRIX_G4_PREFILL_DIRECT, default 3
// Immutable FP16 panels, decoded once with the identity's RNE/overflow rules.
// This owns an additional 2*ceil(N/16)*16*ceil(K/16)*16 bytes, never alters w.
// Construction synchronizes and rejects overflow before returning usable panels.
class PreparedLinear {
  public:
    explicit PreparedLinear(LinearView weights, gpu::Queue queue = nullptr);
    [[nodiscard]] const gpu::Buffer& panels() const { return panels_; }
    [[nodiscard]] LinearView weights() const { return weights_; }
  private:
    LinearView weights_;
    gpu::Buffer panels_;
};
// Inputs/outputs/weights are device-resident. Scratch holds rows*ceil(K/16)*16
// FP16 values. failure and maximum are required device uint32_t scalars; maximum
// receives max(abs(input)) as positive FP32 bits. Calls enqueue on queue.
void linear(LinearView weights, const float* x, unsigned rows, float* out, void* scratch,
            std::uint32_t* failure, std::uint32_t* maximum, gpu::Queue queue = nullptr, bool tiled = true,
            const PreparedLinear* prepared = nullptr, void* converted_event = nullptr, bool converted = false,
            LinearPrecision precision = LinearPrecision::fp16);
// hilo / hilo_k128 (g4-wmma-pf-v2, emulate_linear_v2) need prepared panels and
// twice the scratch: high panels then low panels (rows * stride FP16 each).
// Exact fusion of gate, up, gelu_tanh(gate)*up and the next linear's FP16
// input conversion: product receives rows x outputs FP16 panels as linear's
// scratch would after converting the FP32 product; maximum[0..2] receive the
// gate/up input and product abs-bits maxima. STRIX_G4_PREFILL_FUSE_FFN=0
// keeps the separate steps (identical bits).
[[nodiscard]] bool fused_ffn_enabled();
[[nodiscard]] bool gelu_pair_supported(const PreparedLinear& gate, const PreparedLinear& up);
void gelu_pair(const PreparedLinear& gate, const PreparedLinear& up, const float* x, unsigned rows, void* scratch,
               void* product, std::uint32_t* failure, std::uint32_t* maximum, gpu::Queue queue = nullptr, bool low = false,
               bool split = false, bool blocked = false); // split: gate/up run hilo (scratch holds high+low input panels); blocked: hilo_k128
// EXL3 packs on prepared panels (g4-exl3-wmma-pf-v1): the pack's input
// transform writes the [K16][row][16] input panels, the panel GEMMs above run
// unchanged, then the pack's output transform. Every output keeps one
// increasing K16 WMMA chain on the same FP16 operands, so the bytes equal the
// pack's own 64x64 WMMA kernel; STRIX_G4_EXL3_PREFILL_GEMM=0 restores it.
[[nodiscard]] bool exl3_gemm_enabled();
// EXL3 FFN on the panel route: gate and up GEMMs keep raw core values in
// gate_core/up_core (rows x outputs floats), then one launch applies their
// output transforms, gelu_tanh(gate) * up and down's input transform into
// product (down's FP16 input panels; call linear(down, ..., converted = true)
// with product as scratch). Bytes equal linear(gate), linear(up),
// gpu::gelu_product and linear(down); maximum[0..2] as gelu_pair. scratch holds
// rows * inputs FP16 values. STRIX_G4_PREFILL_FUSE_FFN=0 keeps the separate steps.
[[nodiscard]] bool exl3_gelu_pair_supported(const PreparedLinear& gate, const PreparedLinear& up, LinearView down);
void exl3_gelu_pair(const PreparedLinear& gate, const PreparedLinear& up, LinearView down, const float* x, unsigned rows,
                    void* scratch, float* gate_core, float* up_core, void* product, std::uint32_t* failure,
                    std::uint32_t* maximum, gpu::Queue queue = nullptr);
// Opt-in attention sub-profile. Descriptors use B's immutable snapshot
// and unpublished overlay contract. Queries are consecutive absolute positions
// in one stream, with identical cache/overlay descriptors except position.
// No cache bytes or tags are written here.
void attention(const float* q, const gpu::KvRead* reads, unsigned rows, unsigned heads, unsigned dim,
               float* out, std::uint32_t* failure, gpu::Queue queue = nullptr, void* packed = nullptr,
               unsigned packed_stride = 0, unsigned key_capacity = 0, bool repack = true);
// Bytes required for packed at packed_stride positions of width KV components.
// The default split operand layout (STRIX_G4_PREFILL_ATTN_SPLIT=0 restores the
// original kernel; identical bits) needs more than the original 2*stride*width.
[[nodiscard]] std::size_t attention_packed_bytes(unsigned packed_stride, unsigned width);
[[nodiscard]] bool attention_split_enabled(); // startup-cached STRIX_G4_PREFILL_ATTN_SPLIT
// Canonical (g4-v1) attention for a whole prefill chunk in three launches,
// byte-identical to gpu::attention evaluated row by row on the same
// descriptors (see canonical.hip). reads is the host copy of the chunk's
// descriptors: consecutive positions of one stream, identical otherwise.
// Scores span [base, base + width) per (row, head); scratch needs
// rows * heads * width floats and denominators rows * heads floats.
// Returns false, enqueueing nothing, for head dimensions other than 256/512.
[[nodiscard]] bool canonical_kernel_enabled(); // startup-cached STRIX_G4_PREFILL_CANON_KERNEL, default on
struct CanonicalSpan {
    std::uint64_t base = 0;
    unsigned width = 0;
};
[[nodiscard]] bool canonical_pack_enabled(); // startup-cached STRIX_G4_PREFILL_CANON_PACK, default on
[[nodiscard]] bool canonical_rows_major_enabled(); // startup-cached STRIX_G4_PREFILL_CANON_ROWS_MAJOR, default on
[[nodiscard]] bool canonical_fuse_enabled(); // startup-cached STRIX_G4_PREFILL_CANON_FUSE, default on
[[nodiscard]] CanonicalSpan canonical_span(const gpu::KvRead& first_row, unsigned rows);
// Packed queries/keys (post-order lane layout) and key presence for launch one.
[[nodiscard]] std::size_t canonical_packed_floats(unsigned rows, unsigned heads, unsigned dim, unsigned span_width);
struct CanonicalScratch {
    float* scores = nullptr;       // rows * heads * span width floats
    std::size_t score_floats = 0;
    float* denominators = nullptr; // rows * heads floats
    float* packed = nullptr;       // canonical_packed_floats(), when packing
    std::size_t packed_floats = 0;
};
// pack/fuse: -1 = STRIX_G4_PREFILL_CANON_PACK/_FUSE, 0/1 forced. Fused computes
// row maxima in the score launch and exp/denominators in the value launch (no
// softmax launch). phase_ms (optional, four entries: pack, scores, softmax,
// values) accumulates synchronized event times.
bool canonical_attention(const float* q, std::span<const gpu::KvRead> reads, unsigned heads, unsigned dim, float* out,
                         std::uint32_t* failure, gpu::Queue queue, const CanonicalScratch& scratch, int pack = -1,
                         float* phase_ms = nullptr, int fuse = -1);

// Linear arithmetic of a prefill object. wmma is the g4-wmma-pf identity
// (FP16 operands). exact keeps every g4-v1 linear: gpu::linear_rows, whose
// bytes equal the decode target's eight-row passes, with canonical attention
// and the same norms/RoPE/PLE stages, so the committed state is prefill-exact
// (STRIX_G4_PREFILL_EXACT_MANYROW=0 runs the eight-row linears instead).
// a8: g4-a8-v1 linears (a8_linear.hpp) on every quantized matrix, g4-v1 on BF16/F32 ones.
enum class Linears { wmma, exact, a8 };
// Per-op precision of WMMA linears (g4-wmma-pf-v2 development). fp16 is v1;
// exact runs that op with the g4-v1 linear (diagnostic upper bound); hilo
// splits activations into FP16 high and scaled low panels with independent
// accumulators; hilo_k128 additionally restarts the accumulators every K128
// and merges the block partials with compensated FP32 summation.
// "regex=mode;regex=mode" over op names (ple_model, head, layer.N.{q,k,v,o,
// gate,up,down,ple_gate,ple_projection}); full match, first rule wins, a rule
// without "=mode" means hilo_k128. Throws on syntax errors.
struct LinearPolicy {
    struct Rule { std::string pattern; LinearPrecision mode; };
    std::vector<Rule> rules;
    [[nodiscard]] static LinearPolicy parse(const std::string& spec);
    [[nodiscard]] LinearPrecision lookup(const std::string& op) const;
    [[nodiscard]] bool empty() const { return rules.empty(); }
};
struct Range {
    std::string tensor;
    float input_maximum = 0.0F;
};
struct Stats {
    double linear_ms = 0.0, attention_ms = 0.0, other_ms = 0.0; // cumulative, only when profiling
    struct Kernel {
        double ms = 0.0, conversion_ms = 0.0, matrix_ms = 0.0, flops = 0.0, bytes = 0.0;
        std::uint64_t calls = 0;
    };
    std::map<std::string, Kernel> kernels;
    std::size_t prepared_bytes = 0;
    bool tiled = false, prepared = false;
    unsigned direct = 0;
    AttentionMode attention = AttentionMode::canonical;
    bool attention_packed = false;
    bool canonical_kernel = false; // whole-chunk canonical attention (canonical mode only)
    Linears linears = Linears::wmma;
    std::string linear_policy; // canonical spec text of the active LinearPolicy
};
enum class Residency { host, device };
// One stream's prompt rows in a batched prefill pass.
struct PrefillSegment {
    KvTxn* transaction = nullptr;
    std::span<const std::uint32_t> tokens;
};
class DevicePrefill {
  public:
    // Host residency uploads packed weights once. Device residency borrows immutable
    // views (owner must outlive this object), allowing the coordinator to share them.
    // Linears::exact prepares no FP16 panels. With WMMA attention it is only a
    // diagnostic arm (exact linears + g4-wmma-attn-v1), never an identity.
    DevicePrefill(HostWeights weights, RopeTables tables, Residency residency = Residency::host, bool profile = false,
                  AttentionMode attention = attention_mode(), Linears linears = Linears::wmma);
    ~DevicePrefill();
    DevicePrefill(const DevicePrefill&) = delete;
    DevicePrefill& operator=(const DevicePrefill&) = delete;
    // Evaluate an unpublished target transaction, at most chunk_rows prompt inputs.
    // Padding never writes KV. Failure leaves the transaction uncommittable.
    // logits=false skips the head for a chunk that may not end the prompt; the
    // last row's final-norm output is kept so last_logits() can produce the
    // same logits later if that chunk turns out to be the last one.
    // hidden=true also returns every row's final-norm output (the decode
    // target's RowPlan::export_hidden row).
    // views=true admits PINNED_TAPS: taps (only when taps=true) and hidden rows
    // arrive as ChunkResult views of pinned host memory, copied asynchronously
    // (taps overlap the layers after the last tap layer). With the lever off,
    // or views=false, taps and hidden are vectors as before.
    ChunkResult forward(KvTxn& transaction, std::span<const std::uint32_t> tokens, bool all_logits = false, bool logits = true,
                        bool hidden = false, bool taps = true, bool views = false);
    // Several fresh transactions in one pass without logits, at most chunk_rows rows in total: the linears and
    // epilogues run once over the segments' concatenated rows, while KV reads/writes and attention stay per segment.
    // A many-row linear's output row reads only its own input row, so each segment's KV, taps and hidden rows equal
    // forward() on that segment alone (the WMMA route also needs the same kernel choice: every segment > 8 rows).
    // One result per segment, in order; views point into this object's pinned rows until the next pass. A failure
    // fails every transaction of the pass.
    std::vector<ChunkResult> forward_batch(std::span<const PrefillSegment> segments, bool hidden = false, bool taps = true,
                                           bool views = false);
    [[nodiscard]] std::vector<float> last_logits();
    // Development only: per-op precision of the WMMA linears (see LinearPolicy).
    void set_linear_policy(const LinearPolicy& policy);
    // Commits canonical chunks and returns last-row logits and recent committed taps.
    Result run(gpu::DeviceStream& stream, std::span<const std::uint32_t> tokens, DraftContext* context = nullptr);
    // Cumulative maxima over all chunks evaluated by this object, including failures.
    [[nodiscard]] const std::vector<Range>& ranges() const;
    [[nodiscard]] const Stats& stats() const;
    // Device-address views only, for consumers accepting Residency::device.
    // The owning prefill must outlive every borrower. Never pass these to a
    // host tensor reader or an API that assumes host-address weights.
    [[nodiscard]] const HostWeights& borrow_device_weights() const;
    // A/B switch for the canonical route: whole-chunk kernels or the per-8-row loop.
    void use_canonical_kernel(bool enabled);
    // Turns per-op event timing (Stats::linear_ms/attention_ms/kernels) on or off.
    void set_profile(bool enabled);
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<ChunkResult> pass(std::span<const PrefillSegment> segments, bool all_logits, bool logits, bool hidden, bool taps,
                                  bool views);
};

// Scheduler input fragments are buffered until a canonical chunk is complete.
// finish() is the prompt boundary. No scheduler slice may call finish implicitly.
class Prompt {
  public:
    Prompt(DevicePrefill& prefill, gpu::DeviceStream& stream, DraftContext* context = nullptr);
    void append(std::span<const std::uint32_t> tokens);
    Result finish();
  private:
    void flush(bool final);
    DevicePrefill& prefill_;
    gpu::DeviceStream& stream_;
    DraftContext* context_ = nullptr;
    std::vector<std::uint32_t> pending_;
    Result result_;
    bool finished_ = false, failed_ = false;
};
} // namespace strix::gemma4::prefill
