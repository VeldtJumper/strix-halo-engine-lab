#pragma once
// FirstBlockCache (identity h3-fbc-v1, FAST-class, OFF by default): host mirror + decision arithmetic (no HIP).
//
// Reference (the first-block cache rule of the reference runtime, operating point 0.08), per model call on the packed residual stream
// ([rows][width] bf16 torch tensors): h0 = input of block 0, h1 = output of block 0, r = h1 - h0 (bf16 op):
//   no cache          -> full forward; afterwards r_ref = r, tail_ref = h_last - h1 (bf16 ops)
//   else  N = (r - r_ref).abs().mean()      bf16 tensor: elementwise bf16 ops, mean accumulated wide, rounded ONCE to bf16
//         D = r_ref.abs().mean().clamp(min=1e-8)   (clamp in the op's float math, result bf16)
//         score = N / D                    (bf16 / bf16: fp32 quotient rounded to bf16)
//         score non-finite -> cache cleared, full forward, NO commit (the call is untracked)
//         score <= threshold (python float compare of the bf16 value) -> reuse: h_last = h1 + tail_ref (bf16 op),
//                                     blocks 1.. skipped, anchors unchanged
//         else full forward, then commit (r_ref = r, tail_ref = h_last - h1)
// Ours (bitwise contract between host mirror and device, and between 1 node and 2 nodes):
//   - per 64-row chunk (global chunk index): 256 lanes, lane t sums elements e = t, t + 256, ... (chunk-linear: rows
//     ascending, columns ascending) of |bf16(r - r_ref)| and of |r_ref| in fp64 (no contraction), then a pairwise tree
//     over the lanes (s = 128, 64, .., 1);
//   - per key block b (rows [0, cut) / [cut, rows), cut % 64 == 0): sums of its chunks in chunk order from +0;
//   - N = sum_b N_b, D = sum_b D_b in block order (each rank computes its block's pair; 32 bytes exchanged);
//   - mean = (sum / elements) in fp64, rounded ONCE to bf16 (round-to-odd fp32 then RNE: a single correct rounding);
//     the reference accumulates in fp32 in its library's order, so a mean within a few fp32 ulps of a bf16 rounding
//     boundary can round the other way there (FAST-class; fixtures from the reference cover the boundaries).
//   - D clamp: max(D, bf16(1e-8)) in fp32 (NaN propagates); score = bf16(fp32(N) / fp32(D));
//     decision: finite and double(score) <= threshold.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/gemm/ref.hpp"

namespace strix::h3::levers::fbc {

inline constexpr const char* kFbcIdentity = "h3-fbc-v1";
constexpr std::uint32_t kChunkRows = 64U;
constexpr std::uint32_t kLanes = 256U;

inline float bf(std::uint16_t v) { return gemm::bf16_to_f32(v); }
inline std::uint16_t sub_bf16(std::uint16_t a, std::uint16_t b) { return gemm::f32_to_bf16(bf(a) - bf(b)); }
inline std::uint16_t add_bf16(std::uint16_t a, std::uint16_t b) { return gemm::f32_to_bf16(bf(a) + bf(b)); }

// Correctly rounded fp64 -> bf16 (RNE): round-to-odd to fp32 (16 extra bits), then RNE to bf16.
inline std::uint16_t f64_to_bf16(double x) {
    if (std::isnan(x)) return 0x7fc0U;
    float t = static_cast<float>(x);  // RNE
    if (std::isinf(t) && !std::isinf(x)) t = std::copysign(std::numeric_limits<float>::max(), t);  // overflow: odd max
    if (std::fabs(static_cast<double>(t)) > std::fabs(x)) t = std::nextafter(t, 0.0F);              // truncate
    if (static_cast<double>(t) != x) {
        std::uint32_t u;
        std::memcpy(&u, &t, 4);
        u |= 1U;  // sticky
        std::memcpy(&t, &u, 4);
    }
    return gemm::f32_to_bf16(t);
}

inline void check_threshold(double thr) {
    if (!std::isfinite(thr) || thr < 0.0) throw std::invalid_argument("h3 fbc: threshold must be finite and >= 0");
}
// Row counts are uint32 and chunk counts are computed as (rows + 63) / 64: rows above UINT32_MAX - 63 are rejected.
constexpr std::uint32_t kMaxRows = 0xffffffffU - 63U;
inline void check_rows(std::uint64_t rows) {
    if (!rows || rows > kMaxRows) throw std::invalid_argument("h3 fbc: rows out of range (1 .. UINT32_MAX - 63)");
}
inline std::size_t checked_elems(std::uint64_t rows, std::uint64_t width) {
    if (!rows || !width) throw std::invalid_argument("h3 fbc: empty shape");
    check_rows(rows);
    if (rows > std::numeric_limits<std::uint64_t>::max() / width || rows * width > std::numeric_limits<std::size_t>::max() / 2U)
        throw std::invalid_argument("h3 fbc: shape overflows");
    return static_cast<std::size_t>(rows * width);
}

struct Sums {
    double n = 0.0;  // sum |bf16(r - r_ref)|
    double d = 0.0;  // sum |r_ref|
};

// One chunk: rows_in_chunk (1..64) rows of `width` bf16, row pitch ld >= width.
inline Sums chunk_sums_host(const std::uint16_t* r, const std::uint16_t* r_ref, std::uint32_t rows_in_chunk, std::uint32_t width, std::uint64_t ld) {
    if (!r || !r_ref || !rows_in_chunk || rows_in_chunk > kChunkRows || !width || ld < width) throw std::invalid_argument("h3 fbc: bad chunk");
    const std::uint64_t n = static_cast<std::uint64_t>(rows_in_chunk) * width;
    double ln[kLanes], ldn[kLanes];
    for (std::uint32_t t = 0; t < kLanes; ++t) {
        double sn = 0.0, sd = 0.0;
        for (std::uint64_t e = t; e < n; e += kLanes) {
            const std::uint64_t i = (e / width) * ld + e % width;
            sn += std::fabs(static_cast<double>(bf(sub_bf16(r[i], r_ref[i]))));
            sd += std::fabs(static_cast<double>(bf(r_ref[i])));
        }
        ln[t] = sn;
        ldn[t] = sd;
    }
    for (std::uint32_t s = kLanes / 2U; s > 0U; s >>= 1U)
        for (std::uint32_t t = 0; t < s; ++t) {
            ln[t] += ln[t + s];
            ldn[t] += ldn[t + s];
        }
    return Sums{ln[0], ldn[0]};
}

// All chunk sums of a [rows][width] buffer (pitch width); chunk c = rows [64 c, 64 c + 64).
inline std::vector<Sums> chunk_sums_all(const std::vector<std::uint16_t>& r, const std::vector<std::uint16_t>& r_ref, std::uint32_t rows, std::uint32_t width) {
    const std::size_t n = checked_elems(rows, width);
    if (r.size() != n || r_ref.size() != n) throw std::invalid_argument("h3 fbc: buffer sizes do not match rows * width");
    std::vector<Sums> ch;
    for (std::uint32_t c0 = 0; c0 < rows; c0 += kChunkRows)
        ch.push_back(chunk_sums_host(r.data() + static_cast<std::size_t>(c0) * width, r_ref.data() + static_cast<std::size_t>(c0) * width,
            std::min(kChunkRows, rows - c0), width, width));
    return ch;
}

// Sum of chunks [first, first + count) in chunk order from +0.
inline Sums block_sums(const std::vector<Sums>& chunks, std::size_t first, std::size_t count) {
    if (first > chunks.size() || count > chunks.size() - first) throw std::out_of_range("h3 fbc: chunk range");
    Sums s;
    for (std::size_t c = first; c < first + count; ++c) {
        s.n += chunks[c].n;
        s.d += chunks[c].d;
    }
    return s;
}
// One node: the two key blocks of a [rows] buffer split at cut (multiple of 64, 0 < cut <= rows; cut == rows -> one block).
inline std::vector<Sums> split_block_sums(const std::vector<Sums>& chunks, std::uint32_t rows, std::uint32_t cut) {
    if (!rows || cut % kChunkRows || !cut || cut > rows || chunks.size() != (rows + kChunkRows - 1U) / kChunkRows)
        throw std::invalid_argument("h3 fbc: bad cut");
    const std::size_t cc = cut / kChunkRows;
    if (cc == chunks.size()) return {block_sums(chunks, 0, cc)};
    return {block_sums(chunks, 0, cc), block_sums(chunks, cc, chunks.size() - cc)};
}

struct Decision {
    float score = 0.0F;  // bf16-valued
    bool reuse = false;
    bool finite = true;
};
// blocks: (N_b, D_b) in key-block order; elements = global rows * width.
inline Decision decide(const std::vector<Sums>& blocks, std::uint64_t elements, double threshold) {
    check_threshold(threshold);
    if (!elements || blocks.empty()) throw std::invalid_argument("h3 fbc: no elements / blocks");
    Sums t;
    for (const Sums& b : blocks) {
        t.n += b.n;
        t.d += b.d;
    }
    const double cnt = static_cast<double>(elements);
    const float nb = bf(f64_to_bf16(t.n / cnt));
    float db = bf(f64_to_bf16(t.d / cnt));
    const float floor = bf(gemm::f32_to_bf16(1e-8F));
    if (db < floor) db = floor;  // NaN fails the compare and propagates
    Decision dec;
    dec.score = bf(gemm::f32_to_bf16(nb / db));
    dec.finite = std::isfinite(dec.score);
    dec.reuse = dec.finite && static_cast<double>(dec.score) <= threshold;
    return dec;
}

// What the driver does after block 0.
enum class Action : std::uint8_t {
    kFullCommit,    // run blocks 1.., then commit(h_last)   (cold start or score above the threshold)
    kReuse,         // h already holds h_last; skip blocks 1..
    kFullNoCommit,  // non-finite score: the cache was cleared; run blocks 1.., do NOT commit (next call is cold)
};
inline Action action_of(bool ready, const Decision& d) {
    if (!ready) return Action::kFullCommit;
    if (!d.finite) return Action::kFullNoCommit;
    return d.reuse ? Action::kReuse : Action::kFullCommit;
}

// Host mirror of the per-call state machine (tests / CPU forward). Buffers [rows][width] bf16, contiguous.
class FbcHost {
public:
    FbcHost(double threshold, std::uint32_t width) : thr_(threshold), w_(width) {
        check_threshold(threshold);
        if (!width) throw std::invalid_argument("h3 fbc: width 0");
    }
    void reset() {
        ready_ = false;
        phase_ = Phase::kIdle;
    }
    bool ready() const { return ready_; }
    // After block 0 (h0 = its input, h = its output, modified in place to h_last on kReuse). cut = key-block cut.
    // Rejected while a kFullCommit call awaits its commit (no implicit abandonment: reset() abandons it explicitly).
    Action after_block0(const std::vector<std::uint16_t>& h0, std::vector<std::uint16_t>& h, std::uint32_t rows, std::uint32_t cut, Decision* out) {
        if (phase_ == Phase::kAwaitCommit) throw std::logic_error("h3 fbc: after_block0 while the previous call awaits its commit (reset() to abandon)");
        const std::size_t n = checked_elems(rows, w_);
        if (h0.size() != n || h.size() != n) throw std::invalid_argument("h3 fbc: buffer sizes do not match rows * width");
        if (cut % kChunkRows || !cut || cut > rows) throw std::invalid_argument("h3 fbc: bad cut");
        if (ready_ && rows != rows_) throw std::invalid_argument("h3 fbc: row count changed without reset");
        rows_ = rows;
        r_.resize(n);
        for (std::size_t i = 0; i < n; ++i) r_[i] = sub_bf16(h[i], h0[i]);
        h1_ = h;
        Decision dec;
        if (ready_) dec = decide(split_block_sums(chunk_sums_all(r_, rref_, rows, w_), rows, cut), static_cast<std::uint64_t>(n), thr_);
        const Action a = action_of(ready_, dec);
        if (a == Action::kFullNoCommit) ready_ = false;
        if (a == Action::kReuse)
            for (std::size_t i = 0; i < n; ++i) h[i] = add_bf16(h[i], tail_[i]);
        phase_ = a == Action::kFullCommit ? Phase::kAwaitCommit : Phase::kIdle;
        if (out) *out = dec;
        return a;
    }
    // After the last block of a kFullCommit call.
    void commit(const std::vector<std::uint16_t>& h_last) {
        if (phase_ != Phase::kAwaitCommit) throw std::logic_error("h3 fbc: commit outside a kFullCommit call");
        if (h_last.size() != h1_.size()) throw std::invalid_argument("h3 fbc: h_last size");
        rref_ = r_;
        tail_.resize(h_last.size());
        for (std::size_t i = 0; i < h_last.size(); ++i) tail_[i] = sub_bf16(h_last[i], h1_[i]);
        ready_ = true;
        phase_ = Phase::kIdle;
    }

private:
    enum class Phase : std::uint8_t { kIdle, kAwaitCommit };
    double thr_;
    std::uint32_t w_, rows_ = 0;
    bool ready_ = false;
    Phase phase_ = Phase::kIdle;
    std::vector<std::uint16_t> r_, rref_, tail_, h1_;
};

}  // namespace strix::h3::levers::fbc
