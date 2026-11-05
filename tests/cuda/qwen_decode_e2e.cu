// Decode gate of the CUDA Qwen engine (one model load).
//   qwen_decode_e2e <tables dir> <rope table> <max positions> <tokens file> <prompt N> <generate G> [ref tokens file]
// Prefills rows [0, N-1) with run(), then greedy-decodes G tokens one row at a time (the node's serial route: the
// last prompt row is the first step). Prints each token with the SHA-256 of its FP32 logits row (the node's
// STRIX_SERIAL_LOGITS_DIGEST) and the node's step chain SHA256(prev || consumed position LE8 || row SHA256)
// (STRIX_MTP_STEP_LOGITS_DIGEST, include/strix/mtp_step_logits_digest.hpp).
// Batch-width invariance: replays the same prompt + generated tokens as verify windows of T = 2..8 rows from a fresh
// prefill and requires every row's argmax and logits digest to equal the serial step's. With a ref file (node greedy
// tokens, comma/space separated), the serial tokens must equal its first G entries.
#include "strix/cuda_qwen_prefill.hpp"
#include "strix/model_content_hash.hpp"

#include <array>
#include <chrono>
#include <span>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;

static std::vector<std::uint32_t> tokens_of(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "missing %s\n", path.c_str()); std::exit(2); }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (auto& ch : s) if (ch == ',' || ch == '[' || ch == ']') ch = ' ';
    std::istringstream in(s);
    std::vector<std::uint32_t> t;
    std::uint32_t v;
    while (in >> v) t.push_back(v);
    return t;
}

using Digest = strix::ModelContentIdentity;
static Digest row_sha(const float* row, std::size_t count) {
    return strix::sha256_bytes(std::span<const std::byte>(reinterpret_cast<const std::byte*>(row), count * 4));
}
static std::string hex(const Digest& d) {
    static const char* x = "0123456789abcdef";
    std::string s;
    for (auto b : d.bytes) { s.push_back(x[b >> 4]); s.push_back(x[b & 15]); }
    return s;
}
static Digest chain_step(const Digest& prev, std::uint64_t position, const Digest& row) {
    std::array<std::byte, 72> in{};
    for (std::size_t j = 0; j < 32; ++j) { in[j] = std::byte{prev.bytes[j]}; in[40 + j] = std::byte{row.bytes[j]}; }
    for (unsigned j = 0; j < 8; ++j) in[32 + j] = static_cast<std::byte>((position >> (8U * j)) & 255U);
    return strix::sha256_bytes(in);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: qwen_decode_e2e tables rope max_positions tokens N G [ref]\n");
        return 1;
    }
    qp::Config c;
    c.model_gguf = std::getenv("STRIX_QWEN_GGUF") ? std::getenv("STRIX_QWEN_GGUF") : "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
    if (const char* s = std::getenv("STRIX_QWEN_MTP")) c.mtp_sidecar_gguf = s;
    c.tables_dir = argv[1];
    c.rope_table = argv[2];
    c.max_positions = static_cast<std::uint32_t>(std::atoi(argv[3]));
    c.release_by_layer = false;
    c.decode = true;
    if (const char* v = std::getenv("STRIX_QWEN_DRAFT_VOCAB")) c.draft_vocab = v;
    const auto prompt_all = tokens_of(argv[4]);
    const auto n = static_cast<std::uint64_t>(std::atoll(argv[5]));
    const auto g = static_cast<std::uint32_t>(std::atoi(argv[6]));
    if (n < 2 || n > prompt_all.size() || g == 0) { std::fprintf(stderr, "bad N/G\n"); return 1; }
    std::vector<std::uint32_t> ref;
    if (argc > 7) ref = tokens_of(argv[7]);
    std::printf("loading...\n");
    std::fflush(stdout);
    qp::Prefill p(c);
    std::printf("loaded %s\n", p.describe().c_str());
    std::fflush(stdout);
    const qp::Sink drop = [](const qp::Piece&) {};

    // serial decode
    std::vector<std::uint32_t> seq(prompt_all.begin(), prompt_all.begin() + static_cast<std::ptrdiff_t>(n));
    p.run(seq, n - 1, drop);
    std::vector<std::uint32_t> gen;
    std::vector<Digest> digest;
    Digest chain{};
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < g; ++i) {
        const auto pos = n - 1 + i;
        const auto r = p.step(&seq[pos], 1, pos, true);
        gen.push_back(r.argmax[0]);
        digest.push_back(row_sha(r.logits.data(), r.logits.size()));
        chain = chain_step(chain, pos, digest.back());
        seq.push_back(r.argmax[0]);
        std::printf("serial pos %llu token %u logits_sha %s\n", static_cast<unsigned long long>(pos), r.argmax[0],
                    hex(digest.back()).c_str());
    }
    const double serial_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("serial step chain %s over positions [%llu, %llu)\n", hex(chain).c_str(),
                static_cast<unsigned long long>(n - 1), static_cast<unsigned long long>(n - 1 + g));
    std::printf("serial %u tokens %.3f s (%.1f tok/s, gate build: per-row head + host argmax)\n", g, serial_s, g / serial_s);
    int bad = 0;
    if (const char* e = std::getenv("STRIX_QRTX_EXPECT_CHAIN")) {  // a chain validated earlier (e.g. by the reference kernels)
        const bool ok = hex(chain) == e;
        std::printf("CHAIN %s: expected %s\n", ok ? "PASS" : "FAIL", e);
        bad += !ok;
    }
    if (!ref.empty()) {
        std::uint32_t same = 0;
        for (std::uint32_t i = 0; i < g && i < ref.size(); ++i) same += gen[i] == ref[i];
        const bool ok = ref.size() >= g && same == g;
        std::printf("REF %s: %u/%u tokens equal the reference\n", ok ? "PASS" : "FAIL", same, g);
        bad += !ok;
    }
    // batch-width invariance: windows of T rows over [n-1, n-1+g) (the last window may be shorter)
    for (std::uint32_t t = 2; t <= 8; ++t) {
        std::vector<std::uint32_t> pre(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(n));
        p.run(pre, n - 1, drop);
        std::uint32_t diff = 0;
        for (std::uint32_t i = 0; i < g; i += t) {
            const auto rows = std::min(t, g - i);
            const auto pos = n - 1 + i;
            const auto r = p.step(&seq[pos], rows, pos, true);
            for (std::uint32_t k = 0; k < rows; ++k) {
                const auto d = row_sha(r.logits.data() + std::uint64_t{k} * (r.logits.size() / rows), r.logits.size() / rows);
                if (r.argmax[k] != gen[i + k] || d != digest[i + k]) {
                    if (diff == 0)
                        std::printf("  T=%u first difference at generated row %u: token %u vs %u\n", t, i + k, r.argmax[k], gen[i + k]);
                    ++diff;
                }
            }
        }
        std::printf("T=%u %s (%u differing rows of %u)\n", t, diff == 0 ? "BIT-IDENTICAL" : "DIFFER", diff, g);
        bad += diff != 0;
    }
    // rollback: windows of T rows over the greedy sequence, committing only `keep` rows (cycling 1..T), so the next
    // window recomputes the dropped rows from rolled-back GDN/conv/PLE/n-gram state; every row must equal serial.
    for (const std::uint32_t t : {8U, 3U}) {
        std::vector<std::uint32_t> pre(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(n));
        p.run(pre, n - 1, drop);
        std::uint32_t diff = 0, windows = 0, rows_done = 0;
        for (std::uint32_t i = 0; i < g;) {
            const auto rows = std::min(t, g - i);
            const auto pos = n - 1 + i;
            const auto r = p.verify(&seq[pos], rows, pos, true);
            for (std::uint32_t k = 0; k < rows; ++k) {
                const auto d = row_sha(r.logits.data() + std::uint64_t{k} * (r.logits.size() / rows), r.logits.size() / rows);
                if (r.argmax[k] != gen[i + k] || d != digest[i + k]) {
                    if (diff == 0)
                        std::printf("  rollback T=%u first difference at generated row %u (window %u)\n", t, i + k, windows);
                    ++diff;
                }
            }
            const auto keep = std::min(rows, 1U + windows % t);
            p.commit(keep);
            i += keep;
            rows_done += rows;
            ++windows;
        }
        std::printf("rollback T=%u %s (%u windows, %u rows computed, %u differing)\n", t, diff == 0 ? "BIT-IDENTICAL" : "DIFFER",
                    windows, rows_done, diff);
        bad += diff != 0;
    }
    // spec invariance: MTP draft chain + verify window + commit(accepted); tokens and logits rows == serial (MTP off)
    if (!c.mtp_sidecar_gguf.empty()) {
        const float min_p = std::getenv("STRIX_QWEN_MIN_P") ? static_cast<float>(std::atof(std::getenv("STRIX_QWEN_MIN_P"))) : 0.7F;
        for (const std::uint32_t depth : {7U, 3U}) {
            std::vector<std::uint32_t> pre(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(n));
            p.run(pre, n - 1, drop);
            std::uint32_t pending = seq[n - 1], diff = 0, windows = 0, drafted = 0;
            std::uint64_t pos = n - 1;
            std::vector<std::uint32_t> produced;
            const auto ts = std::chrono::steady_clock::now();
            while (produced.size() < g) {
                const auto d = p.draft(pending, depth, min_p);
                std::vector<std::uint32_t> rows{pending};
                rows.insert(rows.end(), d.tokens.begin(), d.tokens.end());
                const auto r = p.verify(rows.data(), static_cast<std::uint32_t>(rows.size()), pos, true);
                std::uint32_t keep = 1;
                while (keep < rows.size() && rows[keep] == r.argmax[keep - 1]) ++keep;
                const auto vocab = r.logits.size() / rows.size();
                for (std::uint32_t j = 0; j < keep && produced.size() < g; ++j) {
                    const auto i = produced.size();
                    const auto dg = row_sha(r.logits.data() + std::uint64_t{j} * vocab, vocab);
                    if (r.argmax[j] != gen[i] || dg != digest[i]) {
                        if (diff == 0) std::printf("  spec depth %u first difference at generated row %zu\n", depth, i);
                        ++diff;
                    }
                    produced.push_back(r.argmax[j]);
                }
                p.commit(keep);
                pos += keep;
                pending = r.argmax[keep - 1];
                drafted += static_cast<std::uint32_t>(d.tokens.size());
                ++windows;
            }
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            std::printf("spec depth %u min_p %.2f %s (%u windows, %.2f tokens/window, %u drafts, %.1f tok/s gate build)\n", depth,
                        min_p, diff == 0 ? "BIT-IDENTICAL" : "DIFFER", windows, static_cast<double>(g) / windows, drafted, g / sec);
            bad += diff != 0;
        }
    }
    std::printf("%s\n", bad == 0 ? "DECODE GATE PASS" : "DECODE GATE FAIL");
    return bad == 0 ? 0 : 3;
}
