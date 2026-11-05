// Decode timing of the CUDA Qwen engine (one model load; no logits leave the device).
//   qwen_decode_bench <tables dir> <rope table> <max positions> <tokens file> <prompt N> <generate G>
// Prefills rows [0, N-1), then reports: serial greedy steps (T = 1), verify windows of T = 2..8 rows over the serial
// tokens (ms per window), and MTP speculative decode (depth 7, min-p STRIX_QWEN_MIN_P or 0.7) when a sidecar is set.
#include "strix/cuda_qwen_prefill.hpp"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;
using Clock = std::chrono::steady_clock;

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

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: qwen_decode_bench tables rope max_positions tokens N G\n"); return 1; }
    qp::Config c;
    c.model_gguf = std::getenv("STRIX_QWEN_GGUF") ? std::getenv("STRIX_QWEN_GGUF") : "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
    if (const char* s = std::getenv("STRIX_QWEN_MTP")) c.mtp_sidecar_gguf = s;
    if (const char* v = std::getenv("STRIX_QWEN_DRAFT_VOCAB")) c.draft_vocab = v;
    c.tables_dir = argv[1];
    c.rope_table = argv[2];
    c.max_positions = static_cast<std::uint32_t>(std::atoi(argv[3]));
    c.release_by_layer = false;
    c.decode = true;
    const auto prompt = tokens_of(argv[4]);
    const auto n = static_cast<std::uint64_t>(std::atoll(argv[5]));
    const auto g = static_cast<std::uint32_t>(std::atoi(argv[6]));
    if (n < 2 || n > prompt.size() || g < 8) { std::fprintf(stderr, "bad N/G\n"); return 1; }
    qp::Prefill p(c);
    std::printf("loaded %s\n", p.describe().c_str());
    const qp::Sink drop = [](const qp::Piece&) {};
    std::vector<std::uint32_t> seq(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(n));
    p.run(seq, n - 1, drop);
    cudaProfilerStart();  // nsys --capture-range=cudaProfilerApi: decode only, not the upload
    std::vector<double> ms;
    for (std::uint32_t i = 0; i < g; ++i) {
        const auto t0 = Clock::now();
        const auto r = p.step(&seq[n - 1 + i], 1, n - 1 + i);
        ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        seq.push_back(r.argmax[0]);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("serial T=1: median %.2f ms (%.1f tok/s), min %.2f\n", ms[ms.size() / 2], 1000.0 / ms[ms.size() / 2], ms[0]);
    for (std::uint32_t t = 2; t <= 8; ++t) {
        std::vector<std::uint32_t> pre(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(n));
        p.run(pre, n - 1, drop);
        std::vector<double> w;
        for (std::uint32_t i = 0; i + t <= g; i += t) {
            const auto t0 = Clock::now();
            p.step(&seq[n - 1 + i], t, n - 1 + i);
            w.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
        std::sort(w.begin(), w.end());
        std::printf("window T=%u: median %.2f ms\n", t, w[w.size() / 2]);
    }
    if (!c.mtp_sidecar_gguf.empty()) {
        const float min_p = std::getenv("STRIX_QWEN_MIN_P") ? static_cast<float>(std::atof(std::getenv("STRIX_QWEN_MIN_P"))) : 0.7F;
        std::vector<std::uint32_t> pre(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(n));
        p.run(pre, n - 1, drop);
        std::uint32_t pending = seq[n - 1], produced = 0, windows = 0;
        std::uint64_t pos = n - 1;
        double draft_ms = 0, verify_ms = 0;
        const auto ts = Clock::now();
        while (produced < g) {
            const auto t0 = Clock::now();
            const auto d = p.draft(pending, 7, min_p);
            const auto t1 = Clock::now();
            std::vector<std::uint32_t> rows{pending};
            rows.insert(rows.end(), d.tokens.begin(), d.tokens.end());
            const auto r = p.verify(rows.data(), static_cast<std::uint32_t>(rows.size()), pos);
            std::uint32_t keep = 1;
            while (keep < rows.size() && rows[keep] == r.argmax[keep - 1]) ++keep;
            p.commit(keep);
            draft_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            verify_ms += std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
            pos += keep;
            produced += keep;
            pending = r.argmax[keep - 1];
            ++windows;
        }
        const double sec = std::chrono::duration<double>(Clock::now() - ts).count();
        std::printf("spec depth 7: %.1f tok/s, %.2f tokens/window, draft %.2f ms/window, verify+commit %.2f ms/window\n",
                    produced / sec, static_cast<double>(produced) / windows, draft_ms / windows, verify_ms / windows);
    }
    cudaProfilerStop();
    return 0;
}
