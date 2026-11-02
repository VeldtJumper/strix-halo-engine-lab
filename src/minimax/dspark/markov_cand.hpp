#pragma once
// DSpark Markov chain over a candidate set (drafter numerics change acceptance only; target verify unchanged).
//
// Today (DeviceDrafter::markov_step_host x 7 + Engine::peer2): each step scans the rank's vocabulary half of W2 (51 MB),
// syncs the host, and exchanges 8 bytes with the peer. Here, once per draft:
//   1. each rank selects, per block row k, the canonical SET of the C largest base logits of its half (value descending
//      with -0 == +0, ties by lower id; exactly C; list order unspecified), packs their ids + values and the values of the H
//      members it owns;
//      H = the |H| rows of largest float sum of squares of W2 (lane chains + butterfly; ties: lower id), identical on both
//      ranks (computed at init; the offline evaluation ranked by float32 L2 norm, same order up to rounding at the cutoff);
//   2. one exchange of that payload (two control-channel chunks at C = 1024);
//   3. both ranks run the 7-step chain on device over [own top-C | peer top-C | H], with markov_kernel's per-candidate
//      arithmetic (8 BF16 products per lane in order, xor butterfly 16..1, + base) and its argmax order (`better`),
//      prev chained on device; identical inputs on both ranks -> identical tokens, no agreement needed.
// A nonfinite base logit in either half -> both ranks see the fallback flag in the exchanged counts and return false
// (the caller runs today's full chain). Offline: C = 1024, |H| = 2048 gave the same
// accepted-length histogram as the full chain on 606 anchors.
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <hip/hip_runtime.h>

namespace strix::minimax::dspark {

class CandMarkov {
public:
    static constexpr std::uint32_t kRows = 7, kRank = 256;
    // w1, w2: the drafter's replicated BF16 Markov matrices [vocab][256] (device). [v0, v0 + vn): this rank's vocabulary.
    CandMarkov(const std::uint16_t* w1, const std::uint16_t* w2, std::uint32_t vocab_size, std::uint32_t v0, std::uint32_t vn,
               unsigned C = 1024, unsigned nh = 2048);
    ~CandMarkov();
    CandMarkov(const CandMarkov&) = delete;
    CandMarkov& operator=(const CandMarkov&) = delete;
    // Symmetric host exchange of `bytes` (same size on both ranks): out -> peer, peer's -> in.
    using Exchange = std::function<void(const void* out, void* in, std::size_t bytes)>;
    // logits: this rank's head logits [7][vn] (device). Writes the 7 chain tokens (anchor = previous of row 0).
    // Returns false (both ranks alike) when a nonfinite logit is present: then `out` is untouched.
    bool draft(const float* logits, std::uint32_t anchor, const Exchange& exchange, std::array<std::uint32_t, kRows>& out,
               hipStream_t stream);
    // The same chain with ONE device-to-device exchange of the payload (no host bounce, no 64 KiB chunks): same payload
    // bytes -> same picks as the host form. dev_out/dev_in are device buffers of `bytes` (both ranks, same size).
    using DeviceExchange = std::function<void(const void* dev_out, void* dev_in, std::size_t bytes, hipStream_t stream)>;
    bool draft(const float* logits, std::uint32_t anchor, const DeviceExchange& exchange,
               std::array<std::uint32_t, kRows>& out, hipStream_t stream);
    [[nodiscard]] std::size_t payload_bytes() const noexcept { return payload_; }
    // Top-C selection kernels. v1: one 1024-thread CTA per row (4-pass radix select over the whole row; ids above the
    // threshold in atomic order, then the ties ascending). v2: many CTAs per row (coarse 11-bit histogram -> candidates of
    // the boundary bin -> exact select over the candidates only -> ordered emit; ids ascending). Same canonical SET, same
    // scores per id, same counts / fallback flag, same hval; the chain is independent of the list order (`better` is a
    // total order on (score, id)). STRIX_MM_MARKOV_TOPC=v1 (default) | v2, read at construction.
    enum class TopC : unsigned char { v1, v2 };
    void set_topc(TopC t) noexcept { topc_ = t; }
    [[nodiscard]] TopC topc() const noexcept { return topc_; }
    // Test hook: the select step alone (top-C + scores + counts + owned-H values) into host_out (payload_bytes()), synced.
    void select_payload(const float* logits, unsigned char* host_out, hipStream_t stream);
    [[nodiscard]] unsigned hmax() const noexcept { return hmax_; }
    // Host wall time of draft()'s parts, accumulated over successful drafts (clock reads at the existing syncs only, no
    // extra synchronisation): select = top-C + pack + payload D2H + sync; exchange = the payload chunks; chain = payload
    // H2D + 14 chain kernels + pick D2H + sync. Seconds; exchanges = chunk exchanges issued.
    struct Stats {
        std::size_t drafts = 0, exchanges = 0;
        double select = 0, exchange = 0, chain = 0;
    };
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] unsigned candidates() const noexcept { return 2 * C_ + nh_; }
    static constexpr std::size_t kChunk = std::size_t{64} << 10;  // control-channel limit per exchange

private:
    const std::uint16_t *w1_, *w2_;
    std::uint32_t v0_, vn_;
    unsigned C_, nh_, hmax_;  // hmax_: padded length of the owned-H value arrays (same on both ranks)
    std::size_t payload_ = 0;
    std::uint32_t *hid_own_ = nullptr, *hid_peer_ = nullptr;  // device: H members owned by this rank / by the peer
    std::uint32_t* hpos_own_ = nullptr;                          // device: their positions in this rank's logits
    unsigned nh_own_ = 0, nh_peer_ = 0;
    unsigned char *dpay_own_ = nullptr, *dpay_peer_ = nullptr;
    unsigned char *hsend_ = nullptr, *hrecv_ = nullptr;  // pinned
    std::uint32_t* dprev_ = nullptr;                    // [8]
    void* dcand_ = nullptr;                             // Candidate per block of 8 waves
    void* dpick_ = nullptr;                             // Candidate [7]
    void* hpick_ = nullptr;                             // pinned Candidate [7]
    Stats stats_;
    TopC topc_ = TopC::v1;
    unsigned nb_ = 0;      // v2: 4096-logit chunks per row
    void* ws_ = nullptr;   // v2 workspace (markov_cand.hip: Ws)
    void launch_select(const float* logits, hipStream_t stream);
};

}  // namespace strix::minimax::dspark
