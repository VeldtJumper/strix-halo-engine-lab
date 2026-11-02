// DSpark Markov chain over a candidate set (contract: markov_cand.hpp). Kernels qualified bitwise against CPU replicas in
// the selection benchmark (selection fixtures, canonical top-C sets, pick ids and score bits).
#include "markov_cand.hpp"

#include "../../dspark_common/markov_cand_kernels.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::minimax::dspark {
namespace {

using namespace strix::dspark::markov_cand;
static_assert(CandMarkov::kRank == kRank && CandMarkov::kRows <= kMaxRows);
constexpr unsigned kR = CandMarkov::kRows;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("markov cand ") + what + ": " + hipGetErrorString(e));
}

}  // namespace

CandMarkov::CandMarkov(const std::uint16_t* w1, const std::uint16_t* w2, std::uint32_t vocab_size, std::uint32_t v0, std::uint32_t vn,
                       unsigned C, unsigned nh)
    : w1_(w1), w2_(w2), v0_(v0), vn_(vn), C_(C), nh_(nh) {
    if (C == 0 || C > vn || C > vocab_size - vn || nh == 0 || nh > vocab_size) throw std::invalid_argument("markov cand: bad C / |H|");
    if (vn > (1U << kIdBits)) throw std::invalid_argument("markov cand: vn > 2^22 (topc v2 id width)");
    const char* e = std::getenv("STRIX_MM_MARKOV_TOPC");
    const char* b = std::getenv("STRIX_MM_BASELINE");
    const bool base = b != nullptr && *b != '\0' && std::strcmp(b, "0") != 0;  // as mm_defaults::baseline()
    const std::string tm = e && *e ? e : (base ? "v1" : "v2");  // default v2 (exact, t12); apps export it via mm_defaults
    if (tm != "v1" && tm != "v2") throw std::invalid_argument("STRIX_MM_MARKOV_TOPC must be v1 or v2");
    topc_ = tm == "v2" ? TopC::v2 : TopC::v1;
    // H: identical on both ranks (same kernel, same replicated W2, host sort by (sum desc, id asc)).
    float* dn = nullptr;
    ok(hipMalloc(&dn, std::size_t{vocab_size} * 4), "norm alloc");
    hipLaunchKernelGGL(norm_kernel<Bf16>, dim3((vocab_size + 7) / 8), dim3(256), 0, nullptr, w2, vocab_size, dn);
    std::vector<float> nrm(vocab_size);
    ok(hipMemcpy(nrm.data(), dn, std::size_t{vocab_size} * 4, hipMemcpyDeviceToHost), "norm d2h");
    ok(hipFree(dn), "norm free");
    std::vector<std::uint32_t> order(vocab_size);
    for (std::uint32_t v = 0; v < vocab_size; ++v) order[v] = v;
    std::partial_sort(order.begin(), order.begin() + nh, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) { return nrm[a] > nrm[b] || (nrm[a] == nrm[b] && a < b); });
    std::vector<std::uint32_t> own, peer;
    for (unsigned i = 0; i < nh; ++i) (order[i] >= v0 && order[i] < v0 + vn ? own : peer).push_back(order[i]);
    nh_own_ = static_cast<unsigned>(own.size());
    nh_peer_ = static_cast<unsigned>(peer.size());
    hmax_ = std::max(nh_own_, nh_peer_);
    payload_ = ::strix::dspark::markov_cand::payload_bytes(C, kR, hmax_);  // the member payload_bytes() hides it
    ok(hipMalloc(&hid_own_, std::max<std::size_t>(1, own.size()) * 4), "h alloc");
    ok(hipMalloc(&hid_peer_, std::max<std::size_t>(1, peer.size()) * 4), "h alloc");
    if (!own.empty()) ok(hipMemcpy(hid_own_, own.data(), own.size() * 4, hipMemcpyHostToDevice), "h own");
    std::vector<std::uint32_t> own_pos(own.size());  // positions in this rank's logits
    for (std::size_t i = 0; i < own.size(); ++i) own_pos[i] = own[i] - v0;
    ok(hipMalloc(&hpos_own_, std::max<std::size_t>(1, own.size()) * 4), "h alloc");
    if (!own.empty()) ok(hipMemcpy(hpos_own_, own_pos.data(), own.size() * 4, hipMemcpyHostToDevice), "h own pos");
    if (!peer.empty()) ok(hipMemcpy(hid_peer_, peer.data(), peer.size() * 4, hipMemcpyHostToDevice), "h peer");
    ok(hipMalloc(&dpay_own_, payload_), "payload");
    ok(hipMalloc(&dpay_peer_, payload_), "payload");
    ok(hipHostMalloc(&hsend_, payload_), "pinned");
    ok(hipHostMalloc(&hrecv_, payload_), "pinned");
    ok(hipMalloc(&dprev_, 4 * (kRows + 1)), "prev");
    ok(hipMalloc(&dcand_, std::size_t{(2 * C + nh + 7) / 8} * sizeof(Cand)), "cand");
    ok(hipMalloc(&dpick_, kRows * sizeof(Cand)), "pick");
    ok(hipHostMalloc(&hpick_, kRows * sizeof(Cand)), "pick pinned");
    // topc v2 workspace (hist / flags / counters zeroed here, then reset by v2_refine_kernel at every draft)
    nb_ = (vn + kCh - 1) / kCh;
    ok(hipMalloc(&ws_, ws_bytes(nb_, vn, kR)), "topc ws");
    ok(hipMemset(ws_, 0, ws_words(nb_, kR) * 4), "topc ws zero");
    ok(hipDeviceSynchronize(), "topc ws zero sync");
}

void CandMarkov::launch_select(const float* logits, hipStream_t st) {
    if (topc_ == TopC::v1) {
        hipLaunchKernelGGL(topc_kernel<OneSegment>, dim3(kRows), dim3(1024), 0, st, OneSegment{logits, vn_, v0_}, C_, kR, dpay_own_);
        hipLaunchKernelGGL(pack_kernel, dim3((std::max(C_, hmax_) + 255) / 256, kRows), dim3(256), 0, st, logits, vn_, C_, kR, v0_, hpos_own_,
                           nh_own_, hmax_, dpay_own_);
    } else {
        unsigned char* ws = static_cast<unsigned char*>(ws_);
        const OneSegment lg{logits, vn_, v0_};
        hipLaunchKernelGGL(v2_hist_kernel<OneSegment>, dim3(nb_, kRows), dim3(kV2Threads), 0, st, lg, kR, ws, nb_);
        hipLaunchKernelGGL(v2_compact_kernel<OneSegment>, dim3(nb_, kRows), dim3(kV2Threads), 0, st, lg, C_, kR, ws, nb_);
        hipLaunchKernelGGL(v2_refine_kernel, dim3(kRows), dim3(1024), 0, st, vn_, C_, kR, ws, nb_, dpay_own_);
        hipLaunchKernelGGL(v2_emit_kernel<OneSegment>, dim3(nb_, kRows), dim3(kV2Threads), 0, st, lg, C_, kR, hpos_own_, nh_own_, hmax_, ws,
                           nb_, dpay_own_);
    }
    ok(hipGetLastError(), "select");
}

void CandMarkov::select_payload(const float* logits, unsigned char* host_out, hipStream_t st) {
    launch_select(logits, st);
    ok(hipMemcpyAsync(host_out, dpay_own_, payload_, hipMemcpyDeviceToHost, st), "payload d2h");
    ok(hipStreamSynchronize(st), "payload sync");
}

CandMarkov::~CandMarkov() {
    (void)hipFree(hid_own_);
    (void)hipFree(hpos_own_);
    (void)hipFree(hid_peer_);
    (void)hipFree(dpay_own_);
    (void)hipFree(dpay_peer_);
    (void)hipHostFree(hsend_);
    (void)hipHostFree(hrecv_);
    (void)hipFree(dprev_);
    (void)hipFree(dcand_);
    (void)hipFree(dpick_);
    (void)hipHostFree(hpick_);
    (void)hipFree(ws_);
}

bool CandMarkov::draft(const float* logits, std::uint32_t anchor, const Exchange& exchange, std::array<std::uint32_t, kRows>& out,
                       hipStream_t st) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    launch_select(logits, st);
    ok(hipMemcpyAsync(hsend_, dpay_own_, payload_, hipMemcpyDeviceToHost, st), "send d2h");
    ok(hipStreamSynchronize(st), "send sync");
    const auto t1 = clk::now();
    std::size_t nx = 0;
    for (std::size_t off = 0; off < payload_; off += kChunk, ++nx) exchange(hsend_ + off, hrecv_ + off, std::min(kChunk, payload_ - off));
    const auto t2 = clk::now();
    // fallback decision from both ranks' counts (identical on both)
    const Payload mine = view(hsend_, C_, kR), theirs = view(hrecv_, C_, kR);
    for (unsigned k = 0; k < kRows; ++k)
        if (mine.counts[k] != C_ || theirs.counts[k] != C_) return false;
    ok(hipMemcpyAsync(dpay_peer_, hrecv_, payload_, hipMemcpyHostToDevice, st), "recv h2d");
    ok(hipMemcpyAsync(dprev_, &anchor, 4, hipMemcpyHostToDevice, st), "anchor");
    const unsigned nb = (2 * C_ + nh_own_ + nh_peer_ + 7) / 8;
    for (unsigned k = 0; k < kRows; ++k) {
        hipLaunchKernelGGL((cand_kernel<Bf16>), dim3(nb), dim3(256), 0, st, w1_, w2_, dprev_ + k, k, kR, dpay_own_, dpay_peer_, C_, hid_own_,
                           nh_own_, hid_peer_, nh_peer_, hmax_, static_cast<Cand*>(dcand_));
        hipLaunchKernelGGL(pick_kernel, dim3(1), dim3(256), 0, st, static_cast<Cand*>(dcand_), nb, static_cast<Cand*>(dpick_) + k,
                           dprev_ + k + 1);
    }
    ok(hipGetLastError(), "chain");
    ok(hipMemcpyAsync(hpick_, dpick_, kRows * sizeof(Cand), hipMemcpyDeviceToHost, st), "pick d2h");
    ok(hipStreamSynchronize(st), "chain sync");
    const auto t3 = clk::now();
    for (unsigned k = 0; k < kRows; ++k) out[k] = static_cast<const Cand*>(hpick_)[k].id;
    ++stats_.drafts;
    stats_.exchanges += nx;
    stats_.select += std::chrono::duration<double>(t1 - t0).count();
    stats_.exchange += std::chrono::duration<double>(t2 - t1).count();
    stats_.chain += std::chrono::duration<double>(t3 - t2).count();
    return true;
}

bool CandMarkov::draft(const float* logits, std::uint32_t anchor, const DeviceExchange& exchange,
                       std::array<std::uint32_t, kRows>& out, hipStream_t st) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    launch_select(logits, st);
    ok(hipStreamSynchronize(st), "select sync");  // same attribution point as the host form
    const auto t1 = clk::now();
    exchange(dpay_own_, dpay_peer_, payload_, st);  // one device-to-device exchange; the copy into dpay_peer_ is on st
    // fallback decision from both ranks' counts (identical on both): 2 x 7 words to the host
    const std::size_t coff = std::size_t{kRows} * C_ * 4;
    ok(hipMemcpyAsync(hsend_, dpay_own_ + coff, kRows * 4, hipMemcpyDeviceToHost, st), "counts own");
    ok(hipMemcpyAsync(hrecv_, dpay_peer_ + coff, kRows * 4, hipMemcpyDeviceToHost, st), "counts peer");
    ok(hipStreamSynchronize(st), "counts sync");
    const auto t2 = clk::now();
    const auto* mine = reinterpret_cast<const std::uint32_t*>(hsend_);
    const auto* theirs = reinterpret_cast<const std::uint32_t*>(hrecv_);
    for (unsigned k = 0; k < kRows; ++k)
        if (mine[k] != C_ || theirs[k] != C_) return false;
    ok(hipMemcpyAsync(dprev_, &anchor, 4, hipMemcpyHostToDevice, st), "anchor");
    const unsigned nb = (2 * C_ + nh_own_ + nh_peer_ + 7) / 8;
    for (unsigned k = 0; k < kRows; ++k) {
        hipLaunchKernelGGL((cand_kernel<Bf16>), dim3(nb), dim3(256), 0, st, w1_, w2_, dprev_ + k, k, kR, dpay_own_, dpay_peer_, C_, hid_own_,
                           nh_own_, hid_peer_, nh_peer_, hmax_, static_cast<Cand*>(dcand_));
        hipLaunchKernelGGL(pick_kernel, dim3(1), dim3(256), 0, st, static_cast<Cand*>(dcand_), nb, static_cast<Cand*>(dpick_) + k,
                           dprev_ + k + 1);
    }
    ok(hipGetLastError(), "chain");
    ok(hipMemcpyAsync(hpick_, dpick_, kRows * sizeof(Cand), hipMemcpyDeviceToHost, st), "pick d2h");
    ok(hipStreamSynchronize(st), "chain sync");
    const auto t3 = clk::now();
    for (unsigned k = 0; k < kRows; ++k) out[k] = static_cast<const Cand*>(hpick_)[k].id;
    ++stats_.drafts;
    stats_.exchanges += 1;
    stats_.select += std::chrono::duration<double>(t1 - t0).count();
    stats_.exchange += std::chrono::duration<double>(t2 - t1).count();
    stats_.chain += std::chrono::duration<double>(t3 - t2).count();
    return true;
}

}  // namespace strix::minimax::dspark
