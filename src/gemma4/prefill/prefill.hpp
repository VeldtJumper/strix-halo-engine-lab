#pragma once

// g4-wmma-pf-v1 is a distinct prefill identity. Decode remains g4-v1 relative
// to its committed state. Admission requires native arithmetic and fidelity gates.
#include "gemma4/contracts/g4_types.hpp"
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::prefill {
inline constexpr char identity[] = "g4-wmma-pf-v1";
inline constexpr unsigned tile = 16, chunk_rows = 512;
// g4-wmma-pf-v2 low-panel scale and accumulator restart length (emulate_linear_v2).
inline constexpr float v2_low_scale = 4096.0F, v2_low_unscale = 1.0F / 4096.0F;
inline constexpr unsigned v2_block = 128;
// The declared g4-wmma-pf-v2 prefill identity: hilo_k128 on every WMMA linear
// except the per-layer embedding model (fp16), with canonical attention.
inline constexpr const char* v2_identity = "g4-wmma-pf-v2";
inline constexpr const char* v2_policy = "ple_model=fp16;.*=hilo_k128";
// Row bound of one prefill call (the identity chunk of plan()/Prompt).
inline constexpr unsigned max_chunk_rows = chunk_rows;
enum class Mode { exact, wmma };
// Empty/auto stays exact unless the coordinator supplies complete qualification
// admission. Numerical-band helpers alone must never set wmma_admitted.
[[nodiscard]] Mode resolve(std::string_view value, bool wmma_admitted = false);

// Pure chunk policy: scheduler fragments never determine numerical boundaries.
struct Chunk {
    std::uint64_t position = 0;
    unsigned rows = 0, padded_rows = 0;
};
[[nodiscard]] std::vector<Chunk> plan(std::uint64_t frontier, std::size_t rows,
                                      std::uint32_t admitted_context = max_positions);
// Host/runtime levers of the exact prefill chunk. Each one leaves every output
// byte unchanged; a set bit in ChunkResult::levers says the lever ran for that
// chunk (receipt engagement counter). Each is default on; STRIX_G4_PREFILL_<NAME>=0
// turns it off. The switches are read per chunk so one process can A/B them.
enum Lever : std::uint32_t {
    lever_pinned_taps = 1U << 0U, // PINNED_TAPS: taps/hidden through pinned asynchronous copies, only when consumed
    lever_device_finite = 1U << 1U, // DEVICE_FINITE: taps checked finite on the device, inside the failure fence
    lever_fuse_epilogue = 1U << 2U, // FUSE_EPILOGUE: norm+residual+finite(+next norm), gelu/PLE/RoPE/gather + finite in one launch
    lever_side_stream = 1U << 3U, // SIDE_STREAM: K/V projections and the chunk's PLE projection on a second queue
    lever_rows_tile_v2 = 1U << 4U, // ROWS_TILE_V2: per-shape many-row linear tiles (two-tile input staging)
    lever_decode24 = 1U << 5U, // DECODE24: Q4_K/Q6_K many-row decoders use the 24-bit integer multiply
    lever_shared_q6 = 1U << 6U, // SHARED_Q6: wide Q6_K many-row linears decode each weight once per workgroup (LDS)
};
struct LeverName {
    Lever bit;
    const char* name;   // receipt counter
    const char* toggle; // OFF switch
};
inline constexpr LeverName lever_names[] = {
    {lever_pinned_taps, "pinned_taps", "STRIX_G4_PREFILL_PINNED_TAPS"},
    {lever_device_finite, "device_finite", "STRIX_G4_PREFILL_DEVICE_FINITE"},
    {lever_fuse_epilogue, "fuse_epilogue", "STRIX_G4_PREFILL_FUSE_EPILOGUE"},
    {lever_side_stream, "side_stream", "STRIX_G4_PREFILL_SIDE_STREAM"},
    {lever_rows_tile_v2, "rows_tile_v2", "STRIX_G4_PREFILL_ROWS_TILE_V2"},
    {lever_decode24, "decode24", "STRIX_G4_PREFILL_DECODE24"},
    {lever_shared_q6, "shared_q6", "STRIX_G4_PREFILL_SHARED_Q6"},
};
[[nodiscard]] inline bool lever_enabled(Lever bit) {
    for (const auto& l : lever_names)
        if (l.bit == bit) {
            const auto* p = std::getenv(l.toggle);
            return !p || std::string_view(p) != "0";
        }
    return false;
}
struct ChunkResult {
    unsigned rows = 0, vocab = 0, tap_width = 0;
    std::vector<float> last_logits, taps;
    std::vector<float> all_logits;
    std::vector<float> hidden; // rows x hidden final-norm rows, only when requested
    // PINNED_TAPS: views of the prefill object's pinned host rows instead of
    // taps/hidden, valid until its next forward(). Use tap_rows()/hidden_rows().
    std::span<const float> pinned_taps, pinned_hidden;
    std::uint32_t levers = 0;
    // Host time of this call outside the device's work: before the first launch and after the final sync.
    double host_ms = 0.0;
    [[nodiscard]] std::span<const float> tap_rows() const { return pinned_taps.empty() ? std::span<const float>(taps) : pinned_taps; }
    [[nodiscard]] std::span<const float> hidden_rows() const { return pinned_hidden.empty() ? std::span<const float>(hidden) : pinned_hidden; }
};
struct Result {
    std::uint64_t frontier = 0;
    std::vector<float> last_logits, tail_taps;
    std::vector<std::uint64_t> tail_positions;
};
// Append a retained committed tail once, in groups admitted by device draft
// context (at most eight rows). Validate the complete tail before publication.
void append_tail(DraftContext& context, const Result& result, unsigned hidden_width);

// FP32 -> FP16, RNE, gradual underflow, signed zero. False means nonfinite or
// rounding overflow; callers must fail the request, never silently clip.
STRIX_G4_HD inline bool half_bits(float x, std::uint16_t& out) {
    const auto u = bit_cast<std::uint32_t>(x), a = u & 0x7fffffffU;
    const auto sign = std::uint16_t((u >> 16U) & 0x8000U);
    if (a >= 0x7f800000U) return false;
    if (a < 0x33000000U) { out = sign; return true; }
    if (a < 0x38800000U) {
        const unsigned shift = 126U - (a >> 23U);
        const auto m = (a & 0x7fffffU) | 0x800000U;
        const auto q = m >> shift, remainder = m & ((1U << shift) - 1U);
        const auto midpoint = 1U << (shift - 1U);
        out = std::uint16_t(sign | (q + unsigned(remainder > midpoint || (remainder == midpoint && (q & 1U)))));
        return true;
    }
    const auto q = ((a - 0x38000000U) + 0xfffU + ((a >> 13U) & 1U)) >> 13U;
    if (q >= 0x7c00U) return false;
    out = std::uint16_t(sign | q);
    return true;
}

struct Fidelity {
    // These are numerical bands only. Corpus coverage, provenance, serial,
    // speculation, overflow and native opcode gates are separate admission gates.
    std::size_t positions = 0, top1_matches = 0, values = 0;
    double sum_kl = 0.0, squared_error = 0.0, maximum_error = 0.0;
    std::vector<double> kl;
    // Full vocabulary, capped logits, identical teacher-forced prefix. KL(ref||candidate), FP64.
    void observe(std::span<const float> reference, std::span<const float> candidate);
    [[nodiscard]] double mean_kl() const;
    [[nodiscard]] double p99_kl() const;
    [[nodiscard]] double top1() const;
    [[nodiscard]] double rmse() const;
    [[nodiscard]] bool same_weight_bands_pass() const;
    [[nodiscard]] bool bf16_bands_pass() const;
};
} // namespace strix::gemma4::prefill
