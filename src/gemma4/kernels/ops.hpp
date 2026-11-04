#pragma once

// g4-v1 device primitives. All pointers are device-resident; calls enqueue on queue.
// Shared-pass linears decode each coefficient once for up to eight independent rows.
#include "gemma4/contracts/g4_types.hpp"
#include "a8.hpp"
#include "gemma4/exl3/format.hpp"
#include <utility>

namespace strix::gemma4::gpu {
using Queue = void*;

class Buffer {
  public:
    Buffer() = default;
    explicit Buffer(std::size_t bytes);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void resize(std::size_t bytes);
    void upload(const void* source, std::size_t bytes, std::size_t offset = 0);
    void download(void* destination, std::size_t bytes, std::size_t offset = 0) const;
    void zero();
    [[nodiscard]] void* data() const { return data_; }
    template <class T> [[nodiscard]] T* as() const { return static_cast<T*>(data_); }
    [[nodiscard]] std::size_t size() const { return size_; }

  private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
};

// g4-a8-v1 quantized rows: codes [rows, width] (row stride width), meta [rows, width / 32].
// A null codes pointer means "no quantized copy requested".
struct A8Rows {
    std::int8_t* codes = nullptr;
    a8::Meta* meta = nullptr;
    std::uint32_t width = 0;
};
struct Candidate {
    float value;
    std::uint32_t id;
};
// Optional fused store/epilogue of one shared linear launch. Each stage is the
// declared separate operation, applied to the stored value in the same order.
struct LinearEpilogue {
    // out = gelu_tanh(dot) * product[row * product_stride + output] (PLE activation).
    const float* product = nullptr;
    std::uint32_t product_stride = 0;
    bool cap = false; // out = softcap(dot) (head)
    // Up to two more matrices with the same format and input share the grid;
    // each output keeps its own wave, visits and store. No norm epilogue then.
    LinearView extra[2];
    float* extra_out[2] = {nullptr, nullptr};
    std::uint32_t extra_block[2] = {0, 0}; // set by linear_fused
    std::uint32_t blocks = 0;               // set by linear_fused
    bool hardware_half = false;             // set by the dispatcher (STRIX_G4_HARDWARE_HALF)
    bool split_tree = false;                // g4-a8-v1 only, set by its dispatcher (STRIX_G4_A8_SPLIT_TREE)
};
// Returns false, enqueueing nothing, when the selected schedule cannot fuse the
// store or shared grid (opt-in tiling/cache paths, scalar decode).
bool linear_fused(LinearView w, const float* x, std::uint32_t rows, float* out, const LinearEpilogue& epilogue,
                  Queue queue, std::uint32_t* failure);
bool store_fusion_enabled();
bool row_prefetch_enabled(std::uint32_t rows);
bool pair_tile_enabled(std::uint32_t rows);
bool staged_input_enabled(std::uint32_t rows);
void synchronize(Queue queue = nullptr);
void require_device(); // gfx1151 / wave32 only; does not silently select another identity
void linear(LinearView w, const float* x, std::uint32_t rows, float* out, bool shared = true, Queue queue = nullptr,
            std::uint32_t* failure = nullptr, void* format_scratch = nullptr);
void gate_up(LinearView gate, LinearView up, const float* x, std::uint32_t rows, float* out, bool shared = true,
             Queue queue = nullptr, std::uint32_t* failure = nullptr);
// With quant.codes set (g4-a8-v1), the default grouped schedule also writes the A8G32 rows of out
// and returns true; false means the caller quantizes separately.
bool norm(const float* x, LinearView weight, float* out, std::uint32_t vectors, std::uint32_t width,
          Queue queue = nullptr, std::uint32_t* failure = nullptr, A8Rows quant = {});
// Declared norm, residual/scalar and optional following norm; all intermediate
// nonfinite checks remain observable through failure before publication.
// quant (g4-a8-v1): A8G32 rows of next, or of the new residual when next is null; returns whether the
// selected schedule wrote them.
// Any row count (prompt chunks): the staged parallel norm_residual schedule, whose
// bytes and finite flags equal norm(branch) + residual() + check_finite(residual)
// [+ norm(residual, next_weight) into next]. false, enqueueing nothing, when the
// selected norm() schedule is another one (STRIX_G4_NORM_GROUP/SEARCH/STAGE).
bool norm_residual_rows(const float* branch, LinearView weight, float* residual, std::uint32_t rows, std::uint32_t width,
                        LinearView scalar, float* next, LinearView next_weight, std::uint32_t* failure, Queue queue = nullptr);
// gelu_product followed by check_finite(gate) in one launch (same bytes and flags).
void gelu_product_finite(float* gate, const float* up, std::uint32_t count, std::uint32_t* failure, Queue queue = nullptr);
bool norm_residual(const float* branch, LinearView weight, float* residual, std::uint32_t rows, std::uint32_t width,
                   LinearView scalar, float* next, LinearView next_weight, std::uint32_t* failure,
                   Queue queue = nullptr, A8Rows quant = {});
// The FFN epilogue (norm_residual without scalar or following norm) fused with
// the PLE gate linear and its activation store (STRIX_G4_FFN_PLE_GATE, default
// on, not for g4-a8-v1 gates): residual and out receive the bytes of the separate launches; ticket is one
// zeroed word, reset by each launch. Returns
// false, enqueueing nothing, outside the admitted geometry (one or two rows).
bool ffn_ple_gate(const float* branch, LinearView weight, float* residual, LinearView gate, const float* ple,
                  std::uint32_t product_stride, std::uint32_t rows, std::uint32_t width, float* out,
                  std::uint32_t* failure, std::uint32_t* ticket, Queue queue = nullptr);
void head_norm_rope(float* x, LinearView weight, std::uint32_t rows, std::uint32_t count, std::uint32_t dim,
                    const std::uint64_t* positions, const float* table, bool global, Queue queue = nullptr,
                    std::uint32_t* failure = nullptr);
void embedding(LinearView table, const std::uint32_t* tokens, std::uint32_t rows, float* out, float scale,
               Queue queue = nullptr);
void ple_combine(float* projection, LinearView table, LinearView weight, const std::uint32_t* tokens,
                 std::uint32_t rows, std::uint32_t layer_count, std::uint32_t width, Queue queue = nullptr,
                 std::uint32_t* failure = nullptr);
void ple_gate(float* gate, const float* ple, std::uint32_t rows, std::uint32_t layer_count, std::uint32_t layer,
              std::uint32_t width, Queue queue = nullptr, std::uint32_t* failure = nullptr);
void residual(float* x, const float* branch, std::uint32_t count, LinearView scalar = {}, Queue queue = nullptr);
void gelu_product(float* gate, const float* up, std::uint32_t count, Queue queue = nullptr);
void cap_logits(float* logits, std::uint32_t count, Queue queue = nullptr);
// Scratch has rows*ceil(vocabulary/256) candidates. Final merge visits blocks in order.
void argmax(const float* capped, std::uint32_t rows, std::uint32_t vocabulary, Candidate* scratch, Candidate* result,
            Queue queue = nullptr);
void check_finite(const float* values, std::size_t count, std::uint32_t* failure, Queue queue = nullptr);

// Device-safe packed lookup; the scalar decoders are shared with the host reference.
STRIX_G4_HD inline float coefficient(LinearView v, std::uint32_t row, std::uint32_t col) {
    const auto* p = v.data + std::size_t(row) * v.row_bytes;
    switch (v.format) {
    case WeightFormat::bf16:
        return bf16_value(load16(p + 2ULL * col));
    case WeightFormat::f32: {
        const auto* b = p + 4ULL * col;
        return bit_cast<float>(std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8U) | (std::uint32_t(b[2]) << 16U) |
                               (std::uint32_t(b[3]) << 24U));
    }
    case WeightFormat::q8_0:
        return q8_0_value(p + (col / 32U) * 34ULL, col % 32U);
    case WeightFormat::q4_k:
        return q4_k_value(p + (col / 256U) * 144ULL, col % 256U);
    case WeightFormat::q6_k:
        return q6_k_value(p + (col / 256U) * 210ULL, col % 256U);
    case WeightFormat::exl3_mul1:
        return exl3::rotated_at(v, col, row); // rotated panels only; linear dispatch applies transforms
    }
    return bit_cast<float>(0x7fc00000U);
}
} // namespace strix::gemma4::gpu
