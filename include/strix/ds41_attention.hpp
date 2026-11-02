#pragma once
// deepseek41 decode attention (CSA2): sliding-window FP8 ring, ratio-2 / ratio-1 KV compression into an FP4 cache,
// the FP4 indexer with ReLU-weighted head scores, top-k and the two-level candidate pool, and D512 multi-query
// attention with a per-head sink. Host references whose arithmetic is the device kernels' declared order (see
// docs/models/deepseek-v4.1/attention.md); the scalar helpers below are shared by host and device code.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "strix/ds41_scalar_codecs.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS41A_HD __host__ __device__ inline
#else
#define DS41A_HD inline
#endif

namespace strix::deepseek41 {
struct LayerAttributes;
struct Config;
}  // namespace strix::deepseek41

namespace strix::ds41::attn {

inline constexpr unsigned kHeadDim = 512, kRopeDim = 64, kRopePairs = 32, kHeads = 64, kWindow = 128;
inline constexpr unsigned kIdxHeads = 32, kIdxDim = 128, kMaxRows = 8, kChunk = 64, kTile = 16;
// Rows of one device call (consecutive positions; the prefill's windows). kMaxRows stays the decode window (it fixes
// the ring capacity kWindow + kMaxRows); a call of more rows needs a ring of capacity >= kWindow + rows.
inline constexpr unsigned kLegacyMaxCallRows = 64, kMaxCallRows = 256;
inline constexpr unsigned kTopK = 512, kCandBlocks = 2048, kCandBlockSize = 8;
inline constexpr unsigned kWindowEntryBytes = 528;  // 512 E4M3 codes + 16 E8M0 scales (one per 32)
inline constexpr unsigned kKvEntryBytes = 288;      // 256 bytes of E2M1 pairs + 32 E4M3 scales (one per 16)
inline constexpr unsigned kIdxEntryBytes = 68;      // 64 bytes of E2M1 pairs + 4 E8M0 scales (one per 32)
inline constexpr float kNormEps = 1e-20F;
inline constexpr float kAttnScale = 0.04419417382415922F;  // 512^-0.5
inline constexpr float kIdxWeightScale = 0.015625F;         // 128^-0.5 * 32^-0.5 (exact)
inline constexpr unsigned kMaxChunks = (kWindow + kTopK + kChunk - 1) / kChunk;

// ---- scalar formats: the shared deepseek41 codecs (ds41_scalar_codecs.hpp), same on host and device ----
using codec::f2u;
using codec::u2f;
using codec::pow2f;
using codec::e4m3_encode;
using codec::e2m1_encode;
DS41A_HD float e8m0_value(std::uint8_t e) { return codec::e8m0(e); }
DS41A_HD float e4m3_value(std::uint8_t c) { return codec::e4m3(c); }
DS41A_HD float e2m1_value(std::uint8_t c) { return codec::e2m1(c); }
DS41A_HD int e2m1_int2(std::uint8_t c) { return codec::e2m1_doubled(c); }
DS41A_HD int log2_ceil(float x) {  // ceil(log2(x)) from the bits, x > 0 (the reference's fast_log2_ceil)
    const std::uint32_t b = f2u(x);
    return static_cast<int>((b >> 23U) & 255U) - 127 + ((b & 0x7FFFFFU) != 0U ? 1 : 0);
}
DS41A_HD float f16_to_float(std::uint16_t h) {  // exact half conversion; backend forms below
#if defined(__CUDA_ARCH__)
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    const bool nan = (h & 0x7c00U) == 0x7c00U && (h & 0x3ffU) != 0U;
    return nan ? __uint_as_float((static_cast<std::uint32_t>(h & 0x8000U) << 16U) | 0x7f800000U | (static_cast<std::uint32_t>(h & 0x3ffU) << 13U)) : f;
#elif defined(__HIPCC__)
    // Keep the qualified HIP implementation; CUDA and host use their own forms.
    const std::uint32_t sgn = static_cast<std::uint32_t>(h >> 15U) << 31U, e = (h >> 10U) & 31U, m = h & 1023U;
    if (e == 0U) { const float v = static_cast<float>(m) * 5.9604644775390625e-08F; return sgn != 0U ? -v : v; }
    if (e == 31U) return u2f(sgn | 0x7F800000U | (m << 13U));
    return u2f(sgn | ((e + 112U) << 23U) | (m << 13U));
#else
    const std::uint32_t sgn = static_cast<std::uint32_t>(h >> 15U) << 31U, e = (h >> 10U) & 31U, m = h & 1023U;
    const float v = static_cast<float>(m) * 5.9604644775390625e-08F;
    const float small = sgn != 0U ? -v : v;
    const float other = u2f(e == 31U ? sgn | 0x7F800000U | (m << 13U) : sgn | ((e + 112U) << 23U) | (m << 13U));
    return e == 0U ? small : other;
#endif
}
DS41A_HD float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
// exp: Cody-Waite reduction and a degree-8 polynomial, all fmaf; 0 below -87, inf above 88 (normal results only,
// so the final power-of-two scaling is exact on host and device).
DS41A_HD float exp_f(float x) {
    if (x < -87.0F) return 0.0F;
    if (x > 88.0F) return __builtin_inff();
    const float n = __builtin_rintf(x * 1.44269504088896341F);
    float r = __builtin_fmaf(n, -0.693145751953125F, x);
    r = __builtin_fmaf(n, -1.42860676533018e-06F, r);
    float p = 2.48015873e-05F;
    p = __builtin_fmaf(p, r, 1.98412698e-04F);
    p = __builtin_fmaf(p, r, 1.38888889e-03F);
    p = __builtin_fmaf(p, r, 8.33333333e-03F);
    p = __builtin_fmaf(p, r, 4.16666667e-02F);
    p = __builtin_fmaf(p, r, 1.66666667e-01F);
    p = __builtin_fmaf(p, r, 0.5F);
    p = __builtin_fmaf(p, r, 1.0F);
    p = __builtin_fmaf(p, r, 1.0F);
    // n clamped to pow2f's range before the conversion (a NaN x gives n = NaN, whose int conversion is undefined): the
    // same n for every x not returned above, and a NaN x still gives the NaN of p. Written as comparisons (fmaxf /
    // fminf semantics: a NaN n takes the lower bound) because MSVC has no __builtin_fmaxf / __builtin_fminf.
    const float nc = !(n >= -126.0F) ? -126.0F : (n > 127.0F ? 127.0F : n);
    return p * pow2f(static_cast<int>(nc));
}
DS41A_HD std::uint32_t order_key(float x) {  // monotone float -> uint32 (for selection)
    const std::uint32_t b = f2u(x);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}

// ---- RoPE (pairs (2i, 2i+1) of the last 64 dims; tables computed in double on the host) ----
struct RopeRow { float c[kRopePairs]; float s[kRopePairs]; };
enum class RopeKind : std::uint8_t { window, compressed };  // theta 10000 / theta 160000 + YaRN (factor 16 over 65536)
RopeRow rope_row(RopeKind kind, std::uint64_t position);
DS41A_HD void rope_pair(float& x0, float& x1, float c, float s, bool inverse) {
    const float a = x0, b = x1;
    if (!inverse) { x0 = a * c - b * s; x1 = a * s + b * c; } else { x0 = a * c + b * s; x1 = b * c - a * s; }
}

// ---- layer plan: KV / index sources and reuse (released config) ----
struct LayerAttention {
    unsigned ratio = 0;            // 0 window only, 2 or 1 compressed
    int kv_source = -1;            // layer whose compressed KV and index keys this layer reads
    int index_source = -1;         // layer whose top-k this layer attends with
    bool owns_kv = false, owns_index = false, candidate_source = false, candidate_user = false;
    RopeKind rope = RopeKind::window;
};
// The plan of the model contract: one entry per target layer from deepseek41::layer_attributes (ratio, KV / index
// sources, candidate source and users). Throws std::invalid_argument for a ratio other than 0, 1 or 2.
std::vector<LayerAttention> plan_from_contract(const std::vector<deepseek41::LayerAttributes>& layers);
// Throws std::invalid_argument naming the first configuration value these kernels do not implement (shapes, window,
// indexer, candidate pool, RoPE bases and YaRN scaling, norm epsilon).
void check_contract(const deepseek41::Config& config);
// The released checkpoint's plan written out by hand: the tests check plan_from_contract on the real configuration
// against it.
std::array<LayerAttention, 40> released_plan();

// ---- host references (R rows of one verify window; rows are consecutive positions) ----
struct CompressState { std::array<float, kHeadDim> kv{}, score{}; bool valid = false; };
struct WindowArgs {
    std::span<const float> kv;            // R x 512, wkv(x)
    std::span<const float> norm;          // 512
    std::span<const std::uint64_t> pos;   // R
    std::uint32_t capacity;               // ring entries (>= 128 + R)
    RopeKind rope;
};
void window_write(const WindowArgs& a, std::span<std::uint8_t> ring);
struct CompressArgs {
    unsigned ratio;                        // 2 or 1
    std::span<const float> kv, score;      // R x 512 (score: ratio 2 only)
    std::span<const float> norm;           // 512
    std::span<const std::uint16_t> wk;     // 128 x 512 F16 bits (index key projection)
    std::span<const float> knorm;          // 128
    std::span<const std::uint64_t> pos;    // R
};
// Writes every completed group: KV entry and index-key entry at position pos / ratio. state carries ratio 2's
// incomplete group across calls (in: before row 0, out: after the last row).
void compress(const CompressArgs& a, CompressState& state, std::span<std::uint8_t> kv_cache, std::span<std::uint8_t> idx_cache);
struct IndexArgs {
    std::span<const float> q;              // R x 32 x 128, indexer wq_b(qr)
    std::span<const float> w;              // R x 32, weights_proj(x)
    std::span<const std::uint64_t> pos;    // R
    unsigned ratio;                        // of the index-key source
    RopeKind rope;
};
// scores[r][t] for t < (pos[r] + 1) / ratio
void index_scores(const IndexArgs& a, std::span<const std::uint8_t> idx_cache, std::vector<std::vector<float>>& scores);
enum class Pool : std::uint8_t { plain, candidate_source, candidate_user };
// top-k positions in ascending order (all of them when count <= topk); candidates: one byte per block of
// block_size positions (written by the source, read by users).
std::vector<std::uint32_t> select_topk(std::span<const float> scores, Pool pool, std::span<std::uint8_t> candidates,
    unsigned topk = kTopK, unsigned cand_blocks = kCandBlocks, unsigned block_size = kCandBlockSize);
struct AttentionArgs {
    std::span<const float> q;               // R x 64 x 512, wq_b(qr) before RoPE
    std::span<const float> sink;            // 64
    std::span<const std::uint64_t> pos;     // R
    std::uint32_t capacity;                 // ring entries
    RopeKind rope;
    std::span<const std::uint32_t> idx;     // R x topk (ascending, count per row in idx_count)
    std::span<const std::uint32_t> idx_count; // R (0 for window-only layers)
    unsigned topk = kTopK;
};
// o: R x 64 x 512, after the inverse RoPE of the output
void attention(const AttentionArgs& a, std::span<const std::uint8_t> ring, std::span<const std::uint8_t> kv_cache, std::span<float> o);

// decoded caches (for checks)
std::array<float, kHeadDim> decode_window_entry(const std::uint8_t* e);
std::array<float, kHeadDim> decode_kv_entry(const std::uint8_t* e);
std::array<float, kIdxDim> decode_idx_entry(const std::uint8_t* e);

}  // namespace strix::ds41::attn
