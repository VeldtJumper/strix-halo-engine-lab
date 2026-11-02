#pragma once

// Device side of the DSpark drafter (gfx1151, wave32): the kernels of ds41_dspark.hpp's declared
// operations and a resident drafter that runs a draft step. Every result equals the host reference
// bitwise (tests/hip_ds41_dspark_tests.hip).

#include "strix/ds41_dspark.hpp"
#include "strix/ds41_dspark_context.hpp"
#include "strix/ds41_dspark_driver.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace strix::ds41::dspark {

// ---- single operations (device pointers; stream = hipStream_t or nullptr) ----

// out[r] = main_hidden_row of layer37/38/39[r] (each [rows][4][5120]).
void enqueue_main_hidden(const float* layer37, const float* layer38, const float* layer39, std::uint32_t rows, float* out, void* stream);
void enqueue_rms_norm(const float* x, std::uint32_t rows, std::uint32_t width, const float* weight, float* out, void* stream);
// mixes[r][24] of streams[r] (hc_mixes); fn is [24][20480] F16.
void enqueue_hc_mixes(const float* streams, std::uint32_t rows, const std::uint16_t* fn, const float* scale, const float* base,
                      float* scratch /* rows * 25 */, float* mixes, void* stream);
void enqueue_hc_pre(const float* streams, std::uint32_t rows, const float* coefficients, std::uint32_t stride, float* out, void* stream);
void enqueue_hc_post(const float* x, const float* residual, std::uint32_t rows, const float* mixes, float* out, void* stream);
// RoPE of the last 64 values of every head_width chunk; table[r] = rope_row(position of row r) as
// 32 cosines then 32 sines.
void enqueue_rope(float* x, std::uint32_t rows, std::uint32_t row_width, std::uint32_t head_width, const float* table, bool inverse,
                  void* stream);
// FP8 entries of rows of 512 values; row r goes to entries[(first + r) % capacity] (capacity 0: entry r).
void enqueue_fp8_entries(const float* x, std::uint32_t rows, std::uint8_t* entries, std::uint32_t capacity, std::uint64_t first,
                         void* stream);
// Block attention before the inverse RoPE of the output (block_attention without its last step):
// keys = ring positions oldest .. oldest + ring_keys - 1 (slot % 128), then the block's `rows` entries.
// key_scratch holds the decoded keys (ring_keys + rows rows of 512 floats).
void enqueue_block_attention(const float* q, std::uint32_t rows, const std::uint8_t* ring, std::uint32_t ring_keys,
                             std::uint64_t oldest, const std::uint8_t* block, const float* sinks, float* key_scratch, float* out, void* stream);
// streams[r][k] = token_embd[tokens[r]] for the four copies.
void enqueue_embed_block(const std::uint16_t* token_embd, const std::uint32_t* tokens, std::uint32_t rows, float* streams, void* stream);
// Markov step i of the head: markov[i] = embed[tokens[i]], logits[i][v] += wave_dot(head[v], markov[i]),
// tokens[i + 1] = argmax(logits[i]).
void enqueue_markov_step(const std::uint16_t* embed, const std::uint16_t* head, std::uint32_t* tokens, std::uint32_t step, float* logits,
                         float* markov, void* stream);
// confidence[r] = wave_dot(weight, concat(x[r], markov[r])).
void enqueue_confidence(const float* weight, const float* x, const float* markov, std::uint32_t rows, float* out, void* stream);

// ---- resident drafter ----

// Exchanges `bytes` of device memory with the other rank, ordered on `stream` (e.g. the communicator's
// exchange): remote receives the other rank's local.
using RankExchange = std::function<void(void* stream, const void* local, void* remote, std::size_t bytes)>;

struct DraftTimes {
    double total_ms = 0.0;              // wall time of draft(), host syncs included
    double stages_ms[stages] = {};
    double head_ms = 0.0;
};

class DeviceDrafter {
public:
    // Uploads every tensor the drafter reads (draft stages, the target's token embedding and output head).
    explicit DeviceDrafter(const Weights& weights);
    ~DeviceDrafter();
    DeviceDrafter(const DeviceDrafter&) = delete;
    DeviceDrafter& operator=(const DeviceDrafter&) = delete;

    // main_x of committed target rows (main_hidden [rows][15360]), downloaded.
    [[nodiscard]] std::vector<float> main_rows(std::span<const float> main_hidden, std::uint32_t rows);
    // Appends main_x rows at the next positions; append_hidden does main_rows + append on the device.
    void append(std::span<const float> main_x, std::uint32_t rows);
    void append_hidden(std::span<const float> main_hidden, std::uint32_t rows);
    // The same from rows already in device memory (e.g. the target window's captured inputs of layers
    // 37-39, reduced by enqueue_main_hidden); the caller's producer must be complete or ordered on
    // `stream` (nullptr: the drafter waits for the whole device).
    void append_hidden_device(const float* main_hidden, std::uint32_t rows, void* stream = nullptr);
    // One draft step from `token` at position next() (as dspark::draft).
    [[nodiscard]] Draft draft(std::uint32_t token, bool keep_logits, DraftTimes* times = nullptr);
    [[nodiscard]] std::uint64_t next() const noexcept;
    // Draft vocabulary (default: all 129,280 ids). Extra ids are gathered from the resident head.
    void set_vocabulary(const Vocab& vocabulary);
    [[nodiscard]] Vocab vocabulary() const;
    // Adds context ids (prompt and committed tokens) not yet in the vocabulary; returns how many.
    std::size_t add_vocabulary(std::span<const std::uint32_t> ids);
    // Head split by vocabulary over two ranks: this rank evaluates its half of the vocabulary rows
    // (core ids, then extras, split by position) and each Markov step exchanges its candidate (value,
    // id; 8 bytes) through `exchange`; both ranks take the better (larger value, lower id), so the tokens
    // equal the one-rank draft. Everything else of the drafter is replicated (same inputs, same bits).
    void set_rank(std::uint32_t rank, std::uint32_t world, RankExchange exchange);
    // Markov chain form, read from the environment at construction (both ranks of a split head must agree):
    // STRIX_DS41_DSPARK_MARKOV=cand (default: every step scores each domain half's top-C head logits of the row plus H,
    // the |H| domain ids of largest Markov-head row norm; one payload exchange per draft instead of one per step with two
    // ranks; the drafts can differ from full, the target verifies them as always) | full (every step scores the whole domain).
    // STRIX_DS41_DSPARK_MARKOV_C / _H: C and |H|. A half smaller than C runs the full chain.
    struct MarkovConfig {
        bool cand = true;
        std::uint32_t C = 4096, H = 8192;
    };
    [[nodiscard]] static MarkovConfig markov_config();
    [[nodiscard]] MarkovConfig markov() const noexcept;
    // Forgets every appended row (new sequence).
    void reset();
    [[nodiscard]] ContextSnapshot export_context() const;
    void import_context(const ContextSnapshot&);
    // An empty context continues a sequence at `position` whose earlier rows it never saw (e.g. a prompt read by the
    // device prefill): appends and drafts start there and the attention reads only the rows appended since.
    void skip_to(std::uint64_t position);
    // Sequence contexts: several requests drafted by one drafter over one copy of its weights (BatchForward). A context
    // is what one request's drafting owns: the stage rings, the position (next()) and the draft vocabulary.
    // set_sequences(n) keeps context 0 (the current one) and adds n - 1 fresh ones (empty, full vocabulary);
    // select(i) makes context i the one append / draft / next / vocabulary / reset act on. Each context drafts
    // exactly as its own DeviceDrafter would (same weights, same kernels, its own state).
    void set_sequences(std::uint32_t count);
    [[nodiscard]] std::uint32_t sequences() const noexcept;
    void select(std::uint32_t sequence);
    [[nodiscard]] std::uint32_t selected() const noexcept;
    [[nodiscard]] std::vector<std::uint8_t> ring(std::uint32_t stage) const;

    // Device bytes a drafter over `weights` with `sequences` contexts allocates (no device needed: the sizes of the
    // constructor's uploads, its scratch and set_sequences' contexts; hipMalloc rounding not included).
    struct Footprint {
        std::uint64_t weights = 0, scratch = 0, contexts = 0;
        [[nodiscard]] std::uint64_t total() const noexcept { return weights + scratch + contexts; }
    };
    [[nodiscard]] static Footprint footprint(const Weights& weights, std::uint32_t sequences);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The device drafter behind the speculative driver's Drafter interface.
class DeviceDrafterAdapter final : public Drafter {
public:
    explicit DeviceDrafterAdapter(DeviceDrafter& drafter) : drafter_(drafter) {}
    void append(std::span<const float> main_hidden, std::uint32_t rows) override {
        for (std::uint32_t r0 = 0; r0 < rows; r0 += 128U) {
            const std::uint32_t n = rows - r0 < 128U ? rows - r0 : 128U;
            drafter_.append_hidden(main_hidden.subspan(static_cast<std::size_t>(r0) * main_width, static_cast<std::size_t>(n) * main_width), n);
        }
    }
    Draft draft(std::uint32_t token) override { return drafter_.draft(token, false); }
    void add_context(std::span<const std::uint32_t> ids) override { (void)drafter_.add_vocabulary(ids); }

private:
    DeviceDrafter& drafter_;
};

} // namespace strix::ds41::dspark
