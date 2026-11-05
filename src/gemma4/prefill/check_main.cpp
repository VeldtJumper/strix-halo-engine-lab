// Full-model diagnostics are an explicit CLI, never a ctest or admission token.
#include "device.hpp"
#include "qualification.hpp"
#include "gemma4/engine/profile.hpp"
#include "gemma4/ref/npz.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace g4 = strix::gemma4;
namespace pf = g4::prefill;
namespace gpu = g4::gpu;
namespace {
using Clock = std::chrono::steady_clock;
class ClockSamples {
  public:
    ClockSamples() {
#if defined(__linux__)
        std::error_code error;
        for (std::filesystem::directory_iterator i("/sys/class/drm", error), end; !error && i != end; i.increment(error)) {
            const auto candidate = i->path() / "device/pp_dpm_sclk";
            std::error_code status;
            if (std::filesystem::is_regular_file(candidate, status)) { path_ = candidate; break; }
        }
        if (!path_.empty()) worker_ = std::jthread([this](std::stop_token stop) {
            while (!stop.stop_requested()) { sample(); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
        });
#endif
    }
    ~ClockSamples() { finish(); }
    void finish() { if (worker_.joinable()) { worker_.request_stop(); worker_.join(); } }
    void print() const {
        std::cout << "{\"sclk_source\":\"pp_dpm_sclk\",\"sclk_samples\":" << mhz_.size();
        if (!mhz_.empty()) {
            double total = 0;
            for (double value : mhz_) total += value;
            std::cout << ",\"sclk_min_mhz\":" << *std::min_element(mhz_.begin(), mhz_.end())
                      << ",\"sclk_max_mhz\":" << *std::max_element(mhz_.begin(), mhz_.end())
                      << ",\"sclk_mean_mhz\":" << total / double(mhz_.size());
        }
        std::cout << "}\n";
    }
  private:
    void sample() {
        std::ifstream file(path_);
        for (std::string line; std::getline(file, line);) if (line.find('*') != std::string::npos) {
            std::istringstream input(line);
            std::string level, value, unit;
            if (!(input >> level >> value)) continue;
            try {
                std::size_t end = 0;
                double frequency = std::stod(value, &end);
                unit = value.substr(end);
                if (unit.empty()) input >> unit;
                std::transform(unit.begin(), unit.end(), unit.begin(), [](unsigned char c) { return char(std::tolower(c)); });
                if (unit == "ghz") frequency *= 1000;
                else if (unit == "khz") frequency /= 1000;
                else if (unit == "hz") frequency /= 1000000;
                else if (unit != "mhz") continue;
                if (frequency > 0 && std::isfinite(frequency)) mhz_.push_back(frequency);
            } catch (const std::exception&) { }
        }
    }
    std::filesystem::path path_;
    std::vector<double> mhz_;
    std::jthread worker_;
};
unsigned number(const std::string& s) {
    std::size_t end = 0;
    const auto n = std::stoul(s, &end);
    if (end != s.size() || n > g4::max_positions) throw std::invalid_argument("invalid bounded integer");
    return unsigned(n);
}
struct Args {
    std::filesystem::path model, rope, draft_model, draft_rope;
    std::vector<std::filesystem::path> fixtures;
    std::string hf_embedding = "unspecified";
    std::string linear_policy; // --linear-policy SPEC (g4-wmma-pf-v2 development)
    std::vector<unsigned> lengths;
    unsigned repetitions = 1, warmup = 0, split = 0, continuation = 0;
    bool same_weight = false, check_split = false, profile = false, c4 = false, canonical_reference = false;
    bool after_benchmark_checks = false, profile_comparison = false, canonical_ab = false;
    pf::Linears linears = pf::Linears::wmma; // --linears exact|wmma
    bool exact_ab = false;                   // --exact-ab: exact prefill versus eight-row decode passes
    bool a8_gate = false;                    // --a8-gate: every row's logits + KV digest, a8 prefill vs a8 decode
    unsigned gate_chunk = 512, ref_rows = 1; // --gate-chunk N (prefill chunk), --ref-rows N (decode rows per pass; 0 = 1 and 8)
    std::optional<g4::profile::Name> product; // --profile default|fast|quality
    bool allow_pending = false;               // --allow-pending (benchmarking only)
    std::vector<unsigned> beside;             // --batch-beside L: batched prefill beside a second L-token stream
};
Args arguments(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--same-weight") { a.same_weight = true; continue; }
        if (key == "--canonical-reference") { a.canonical_reference = true; continue; }
        if (key == "--after-benchmark-checks") { a.after_benchmark_checks = true; continue; }
        if (key == "--compare-attention-profile") { a.profile_comparison = true; continue; }
        if (g4::profile::take_option(argc, argv, i, a.product)) continue;
        if (key == "--profile" || key == "--kernel-profile") { a.profile = true; continue; }
        if (key == "--canonical-kernel-ab") { a.canonical_ab = true; continue; }
        if (key == "--exact-ab") { a.exact_ab = true; continue; }
        if (key == "--a8-gate") { a.a8_gate = true; continue; }
        if (key == "--check-c4") { a.c4 = true; continue; }
        if (key == "--allow-pending") { a.allow_pending = true; continue; }
        if (++i >= argc) throw std::invalid_argument("missing argument value");
        const std::string value = argv[i];
        if (key == "--model") a.model = value;
        else if (key == "--rope") a.rope = value;
        else if (key == "--fixture") a.fixtures.emplace_back(value);
        else if (key == "--hf-embedding") a.hf_embedding = value;
        else if (key == "--draft-model") a.draft_model = value;
        else if (key == "--draft-rope") a.draft_rope = value;
        else if (key == "--length") a.lengths.push_back(number(value));
        else if (key == "--repetitions") a.repetitions = number(value);
        else if (key == "--warmup") a.warmup = number(value);
        else if (key == "--linear-policy") a.linear_policy = value;
        else if (key == "--split") { a.split = number(value); a.check_split = true; }
        else if (key == "--continuation") a.continuation = number(value);
        else if (key == "--gate-chunk") a.gate_chunk = number(value);
        else if (key == "--ref-rows") a.ref_rows = number(value);
        else if (key == "--batch-beside") a.beside.push_back(number(value));
        else if (key == "--linears") {
            if (value != "exact" && value != "wmma" && value != "a8") throw std::invalid_argument("--linears expects exact, wmma or a8");
            a.linears = value == "exact" ? pf::Linears::exact : value == "a8" ? pf::Linears::a8 : pf::Linears::wmma;
        }
        else throw std::invalid_argument("unknown argument");
    }
    if (a.model.empty() || a.rope.empty() || a.fixtures.empty() || !a.repetitions || a.repetitions > 20 || a.warmup > 20 || a.continuation > 256)
        throw std::invalid_argument("usage: --model MODEL --rope TABLES --fixture INPUT.npz [--length N --repetitions N --split N --same-weight --continuation N]");
    if (a.hf_embedding != "unspecified" && a.hf_embedding != "ordinary-fp32" && a.hf_embedding != "bf16-boundary")
        throw std::invalid_argument("HF embedding boundary must be unspecified, ordinary-fp32 or bf16-boundary");
    if (a.draft_model.empty() != a.draft_rope.empty()) throw std::invalid_argument("DFlash gate needs model and rotary assets");
    if ((a.c4 || !a.draft_model.empty()) && !a.continuation) a.continuation = 32;
    if (!a.beside.empty() && !a.continuation) a.continuation = 64;
    if ((a.after_benchmark_checks || a.profile_comparison) && a.lengths.empty())
        throw std::invalid_argument("benchmark checks/profile comparison require explicit lengths");
    if (a.profile_comparison && a.profile) throw std::invalid_argument("profile comparison requires uninstrumented benchmark runs");
    if (a.profile_comparison && pf::attention_mode() != pf::AttentionMode::wmma)
        throw std::invalid_argument("attention profile comparison requires STRIX_G4_PREFILL_ATTN=wmma");
    if (a.canonical_ab && (a.lengths.empty() || a.profile || a.profile_comparison || pf::attention_mode() != pf::AttentionMode::canonical))
        throw std::invalid_argument("--canonical-kernel-ab needs explicit lengths, canonical attention and no other profile mode");
    if (a.exact_ab && (a.lengths.empty() || a.linears == pf::Linears::wmma || a.canonical_ab || a.profile_comparison))
        throw std::invalid_argument("--exact-ab needs explicit lengths and --linears exact");
    if (a.a8_gate && (a.lengths.empty() || a.linears != pf::Linears::a8 || !a.gate_chunk || a.gate_chunk > pf::max_chunk_rows || a.ref_rows > 8))
        throw std::invalid_argument("--a8-gate needs explicit lengths, --linears a8, --gate-chunk 1..1024 and --ref-rows 0..8");
    return a;
}
void equal(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size_bytes())) throw std::runtime_error("prefill/continuation byte mismatch");
}
unsigned greedy(std::span<const float> x) {
    unsigned best = 0;
    for (unsigned i = 1; i < x.size(); ++i) if (g4::better(x[i], i, x[best], best)) best = i;
    return best;
}
g4::ForwardResult decode(gpu::DeviceTarget& target, gpu::DeviceStream& stream, std::span<const unsigned> tokens) {
    auto tx = stream.begin(); std::vector<g4::RowPlan> rows;
    for (unsigned i = 0; i < tokens.size(); ++i) rows.push_back({0, 1, tx->frontier() + i, tokens[i], tx.get(), i, true});
    auto out = target.forward_rows(rows); tx->commit(tokens.size()); return out;
}
// --batch-beside: the prompt's chunks share forward_batch passes with a second stream's chunks under the coordinator's
// rule (both fit one pass; where prefill differs from decode, both are > 8 rows). Each stream's committed digest, the
// prompt's last logits and both streams' greedy continuations must equal the stream alone.
void batch_beside(pf::DevicePrefill& prefill, gpu::DeviceTarget& decoder, const g4::ModelConfig& config, std::span<const unsigned> mine,
                  std::span<const unsigned> other, unsigned continuation, bool exact) {
    gpu::DeviceStream mine_alone(config, unsigned(mine.size()) + continuation), other_alone(config, unsigned(other.size()) + continuation);
    const auto solo_mine = prefill.run(mine_alone, mine), solo_other = prefill.run(other_alone, other);
    gpu::DeviceStream mine_batched(config, unsigned(mine.size()) + continuation), other_batched(config, unsigned(other.size()) + continuation);
    const auto pm = pf::plan(0, mine.size()), po = pf::plan(0, other.size());
    const auto piece = [](std::span<const unsigned> t, const pf::Chunk& c) { return t.subspan(std::size_t(c.position), c.rows); };
    std::vector<float> mine_last;
    std::size_t batched = 0, i = 0, j = 0;
    // The other stream's next chunk waits for the next prompt chunk it can share a pass with; when the prompt is done
    // (or nothing fits), chunks run alone, each stream in its own chunk order.
    while (i < pm.size() || j < po.size()) {
        const bool join = i < pm.size() && j < po.size() && pm[i].rows + po[j].rows <= pf::chunk_rows &&
                          (exact || (pm[i].rows > 8 && po[j].rows > 8));
        if (join) {
            // Mine goes last: the pass's last row (kept for last_logits) is the prompt's.
            auto to = other_batched.begin(); auto tm = mine_batched.begin();
            const std::vector<pf::PrefillSegment> segments{{to.get(), piece(other, po[j])}, {tm.get(), piece(mine, pm[i])}};
            (void)prefill.forward_batch(segments);
            to->commit(po[j].rows); tm->commit(pm[i].rows); ++batched;
            if (i + 1 == pm.size()) mine_last = prefill.last_logits();
            ++i; ++j;
        } else if (i < pm.size()) {
            auto tx = mine_batched.begin();
            auto out = prefill.forward(*tx, piece(mine, pm[i]), false, i + 1 == pm.size());
            tx->commit(pm[i].rows);
            if (i + 1 == pm.size()) mine_last = std::move(out.last_logits);
            ++i;
        } else {
            auto tx = other_batched.begin(); (void)prefill.forward(*tx, piece(other, po[j]), false, false); tx->commit(po[j].rows); ++j;
        }
    }
    equal(mine_last, solo_mine.last_logits);
    if (mine_batched.digest() != mine_alone.digest() || other_batched.digest() != other_alone.digest())
        throw std::runtime_error("batched prefill committed-state mismatch");
    // Greedy continuations from the same logits: every token, logit row and final state equal.
    const auto follow = [&](gpu::DeviceStream& a, gpu::DeviceStream& b, std::vector<float> logits) {
        for (unsigned r = 0; r < continuation; ++r) {
            const auto token = greedy(logits);
            const auto x = decode(decoder, a, {&token, 1}), y = decode(decoder, b, {&token, 1});
            equal(x.logits, y.logits); logits = x.logits;
        }
        if (a.digest() != b.digest()) throw std::runtime_error("batched prefill continuation state mismatch");
    };
    follow(mine_alone, mine_batched, solo_mine.last_logits);
    follow(other_alone, other_batched, solo_other.last_logits);
    std::uint64_t h = 1469598103934665603ULL;
    for (const float x : mine_last) { h ^= g4::bit_cast<std::uint32_t>(x); h *= 1099511628211ULL; }
    std::cout << "{\"gate\":\"batched-prefill-beside\",\"prefill_rows\":" << mine.size() << ",\"beside_rows\":" << other.size()
              << ",\"batched_passes\":" << batched << ",\"continuation\":" << continuation << ",\"committed_digest\":\"" << mine_batched.digest()
              << "\",\"last_logits_fnv\":\"" << std::hex << h << std::dec << "\",\"verdict\":\"EXACT\"}\n";
}
void metric(const char* gate, const pf::Fidelity& m, bool same_weight = false, bool quantized = false) {
    const bool bands = same_weight ? m.same_weight_bands_pass()
                                  : (quantized ? m.positions && m.top1() >= 0.95 && m.mean_kl() <= 0.02 &&
                                                     m.p99_kl() <= 0.10 && m.rmse() <= 0.30 && m.maximum_error <= 3.0
                                               : m.bf16_bands_pass());
    std::cout << "{\"gate\":\"" << gate << "\",\"positions\":" << m.positions << ",\"top1\":" << m.top1()
              << ",\"mean_kl\":" << m.mean_kl() << ",\"p99_kl\":" << m.p99_kl() << ",\"rmse\":" << m.rmse()
              << ",\"max_abs\":" << m.maximum_error << ",\"bands_pass\":" << (bands ? "true" : "false")
              << ",\"qualification\":\"INCOMPLETE\"}\n";
}
void merge(pf::Fidelity& a, const pf::Fidelity& b) {
    a.positions += b.positions; a.top1_matches += b.top1_matches; a.values += b.values;
    a.sum_kl += b.sum_kl; a.squared_error += b.squared_error; a.maximum_error = std::max(a.maximum_error, b.maximum_error);
    a.kl.insert(a.kl.end(), b.kl.begin(), b.kl.end());
}
void profile_attention(pf::DevicePrefill& target, const g4::ModelConfig& config, std::span<const unsigned> tokens,
                       std::string_view route) {
    gpu::DeviceStream stream(config, unsigned(tokens.size()));
    const auto before = target.stats();
    const auto unix_start = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto start = Clock::now();
    (void)target.run(stream, tokens);
    const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const auto unix_end = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto& now = target.stats();
    std::cout << "{\"attention_profile\":\"" << route << "\",\"rows\":" << tokens.size() << ",\"seconds\":" << seconds
              << ",\"start_unix_s\":" << unix_start << ",\"end_unix_s\":" << unix_end
              << ",\"linear_ms\":" << now.linear_ms - before.linear_ms << ",\"attention_ms\":" << now.attention_ms - before.attention_ms << "}\n";
    for (const auto& [name, total] : now.kernels) {
        const auto it = before.kernels.find(name);
        const auto previous = it == before.kernels.end() ? pf::Stats::Kernel{} : it->second;
        const double ms = total.ms - previous.ms, matrix_ms = total.matrix_ms - previous.matrix_ms;
        if (total.calls == previous.calls) continue;
        const auto flops = total.flops - previous.flops;
        std::cout << "{\"linear_profile\":\"" << name << "\",\"attention_route\":\"" << route << "\",\"rows\":" << tokens.size()
                  << ",\"ms\":" << ms << ",\"matrix_ms\":" << matrix_ms << ",\"conversion_ms\":" << total.conversion_ms - previous.conversion_ms
                  << ",\"tflops\":" << flops / (ms * 1e9) << ",\"matrix_tflops\":" << flops / (matrix_ms * 1e9)
                  << ",\"minimum_gb_s\":" << (total.bytes - previous.bytes) / (ms * 1e6) << "}\n";
    }
}
// Old per-8-row canonical attention versus the whole-chunk kernels on one
// prefill object: interleaved timings, then one profiled run each. Every run
// must reproduce the first old run's committed KV digest, logits and taps.
void canonical_ab(pf::DevicePrefill& prefill, const g4::ModelConfig& config, std::span<const unsigned> tokens, unsigned repetitions) {
    std::string digest;
    std::vector<float> logits, taps;
    const auto once = [&](bool kernel, bool profiled, unsigned iteration) {
        prefill.use_canonical_kernel(kernel);
        prefill.set_profile(profiled);
        gpu::DeviceStream stream(config, unsigned(tokens.size()));
        const auto before = prefill.stats();
        const auto start = Clock::now();
        const auto result = prefill.run(stream, tokens);
        const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
        const auto& now = prefill.stats();
        const auto state = stream.digest();
        if (digest.empty()) { digest = state; logits = result.last_logits; taps = result.tail_taps; }
        else {
            if (state != digest) throw std::runtime_error("canonical kernel A/B committed KV digest mismatch");
            equal(result.last_logits, logits); equal(result.tail_taps, taps);
        }
        std::cout << "{\"canonical_ab\":\"" << (kernel ? "kernel" : "rows8") << "\",\"profiled\":" << (profiled ? "true" : "false")
                  << ",\"rows\":" << tokens.size() << ",\"iteration\":" << iteration << ",\"seconds\":" << seconds
                  << ",\"tokens_per_s\":" << tokens.size() / seconds;
        if (profiled) std::cout << ",\"linear_ms\":" << now.linear_ms - before.linear_ms << ",\"attention_ms\":" << now.attention_ms - before.attention_ms;
        std::cout << "}\n";
    };
    for (unsigned r = 0; r < repetitions; ++r) { once(false, false, r); once(true, false, r); }
    once(false, true, 0); once(true, true, 0);
    prefill.set_profile(false);
    prefill.use_canonical_kernel(pf::canonical_kernel_enabled());
    std::cout << "{\"gate\":\"canonical-kernel-vs-rows8\",\"rows\":" << tokens.size() << ",\"runs\":" << 2 * repetitions + 2
              << ",\"kv_digest\":\"" << digest << "\",\"verdict\":\"EXACT\"}\n";
}
// The exact many-row prefill versus the decode target's eight-row passes over
// the same prompt (the prefill-exact reference): equal committed KV digest,
// last logits and retained taps. The reference runs once, timed.
void exact_ab(pf::DevicePrefill& prefill, gpu::DeviceTarget& target, const g4::ModelConfig& config,
              std::span<const unsigned> tokens, unsigned repetitions) {
    const auto n = tokens.size();
    const auto width = std::size_t(5) * config.hidden, kept = std::min<std::size_t>(n, 1024);
    gpu::DeviceStream reference(config, unsigned(n));
    std::vector<float> logits, taps;
    taps.reserve(kept * width);
    const auto start = Clock::now();
    for (std::size_t p = 0; p < n; p += 8) {
        const auto count = std::min<std::size_t>(8, n - p);
        const auto out = decode(target, reference, tokens.subspan(p, count));
        for (std::size_t r = 0; r < count; ++r)
            if (p + r >= n - kept) taps.insert(taps.end(), out.taps.begin() + std::ptrdiff_t(r * width), out.taps.begin() + std::ptrdiff_t((r + 1) * width));
        if (p + count == n) logits.assign(out.logits.begin() + std::ptrdiff_t((count - 1) * config.vocab), out.logits.begin() + std::ptrdiff_t(count * config.vocab));
    }
    const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const auto digest = reference.digest();
    std::cout << "{\"exact_ab\":\"rows8\",\"rows\":" << n << ",\"seconds\":" << seconds << ",\"tokens_per_s\":" << n / seconds
              << ",\"kv_digest\":\"" << digest << "\"}\n";
    for (unsigned r = 0; r < repetitions; ++r) {
        gpu::DeviceStream stream(config, unsigned(n));
        const auto begin = Clock::now();
        const auto result = prefill.run(stream, tokens);
        const auto elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
        const auto state = stream.digest();
        std::cout << "{\"exact_ab\":\"manyrow\",\"rows\":" << n << ",\"iteration\":" << r << ",\"seconds\":" << elapsed
                  << ",\"tokens_per_s\":" << n / elapsed << ",\"kv_digest\":\"" << state << "\"}\n";
        if (state != digest) throw std::runtime_error("exact prefill committed KV digest differs from the eight-row passes");
        equal(result.last_logits, logits);
        equal(result.tail_taps, taps);
    }
    std::cout << "{\"gate\":\"exact-manyrow-vs-rows8\",\"rows\":" << n << ",\"runs\":" << repetitions << ",\"tap_rows\":" << kept
              << ",\"kv_digest\":\"" << digest << "\",\"verdict\":\"EXACT\"}\n";
}
std::uint64_t row_hash(const float* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* b = reinterpret_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n * sizeof(float); ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
// The NF2 byte gate: every prompt row's full-vocabulary logits (bytes, via a 64-bit FNV-1a of the row) and
// the committed KV digest of the a8 prefill route, chunked by `chunk`, against the g4-a8-v1 decode target of
// the same build in passes of `ref_rows` rows. Any differing row = not the a8 identity.
// ref_rows = 0 runs the decode target twice, one-row serial passes and 8-row passes, and first requires those
// two to agree on every row and on the KV digest (the decode gate 'R1..8 == serial'); the prefill is then
// compared with them.
void a8_gate(pf::DevicePrefill& prefill, gpu::DeviceTarget& target, const g4::ModelConfig& config,
             std::span<const unsigned> tokens, unsigned chunk, unsigned ref_rows) {
    const auto n = tokens.size();
    std::vector<std::uint64_t> expected(n), actual(n);
    const auto reference_pass = [&](unsigned rows, std::vector<std::uint64_t>& hashes) {
        gpu::DeviceStream reference(config, unsigned(n));
        for (std::size_t p = 0; p < n; p += rows) {
            const auto count = std::min<std::size_t>(rows, n - p);
            const auto out = decode(target, reference, tokens.subspan(p, count));
            for (std::size_t r = 0; r < count; ++r) hashes[p + r] = row_hash(out.logits.data() + r * config.vocab, config.vocab);
        }
        return reference.digest();
    };
    auto start = Clock::now();
    const auto digest = reference_pass(ref_rows ? ref_rows : 1U, expected);
    const auto decode_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (!ref_rows) {
        std::vector<std::uint64_t> batched(n);
        const auto batched_digest = reference_pass(8U, batched);
        std::size_t differing = 0;
        for (std::size_t r = 0; r < n; ++r) differing += batched[r] != expected[r];
        const bool same = differing == 0 && batched_digest == digest;
        std::cout << "{\"gate\":\"a8-decode-rows8-vs-serial\",\"rows\":" << n << ",\"rows_differing\":" << differing
                  << ",\"kv_digest_serial\":\"" << digest << "\",\"kv_digest_rows8\":\"" << batched_digest
                  << "\",\"verdict\":\"" << (same ? "BYTE-IDENTICAL" : "DIFFERENT") << "\"}\n";
        if (!same) throw std::runtime_error("a8 decode: 8-row passes differ from one-row serial passes");
    }
    gpu::DeviceStream stream(config, unsigned(n));
    start = Clock::now();
    for (std::size_t p = 0; p < n; p += chunk) {
        const auto count = std::min<std::size_t>(chunk, n - p);
        auto tx = stream.begin();
        const auto out = prefill.forward(*tx, tokens.subspan(p, count), true);
        tx->commit(count);
        for (std::size_t r = 0; r < count; ++r) actual[p + r] = row_hash(out.all_logits.data() + r * config.vocab, config.vocab);
    }
    const auto prefill_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    const auto state = stream.digest();
    std::size_t differing = 0, first = n;
    for (std::size_t r = 0; r < n; ++r)
        if (expected[r] != actual[r]) { ++differing; first = std::min(first, r); }
    const bool same = differing == 0 && state == digest;
    std::cout << "{\"gate\":\"a8-prefill-vs-a8-decode\",\"rows\":" << n << ",\"chunk\":" << chunk << ",\"ref_rows\":" << ref_rows
              << ",\"rows_compared\":" << n << ",\"rows_differing\":" << differing
              << ",\"first_differing_row\":" << (differing ? std::to_string(first) : std::string("null"))
              << ",\"kv_digest_decode\":\"" << digest << "\",\"kv_digest_prefill\":\"" << state
              << "\",\"decode_s\":" << decode_seconds << ",\"prefill_all_logits_s\":" << prefill_seconds
              << ",\"verdict\":\"" << (same ? "BYTE-IDENTICAL" : "DIFFERENT") << "\"}\n";
    if (!same) throw std::runtime_error("a8 prefill differs from the a8 decode target");
}
void telemetry() {
#if defined(__linux__)
    utsname info{};
    if (uname(&info) == 0) std::cout << "kernel " << info.release << '\n';
    std::ifstream parameter("/sys/module/amdgpu/parameters/iommu_perfopt");
    std::string value;
    if (parameter >> value) std::cout << "amdgpu.iommu_perfopt " << value << '\n';
    for (const auto& card : std::filesystem::directory_iterator("/sys/class/drm")) {
        if (!card.path().filename().string().starts_with("card")) continue;
        for (const char* field : {"pp_dpm_sclk", "pp_dpm_mclk"}) {
            std::ifstream f(card.path() / "device" / field);
            if (!f) continue;
            std::cout << field << '\n';
            for (std::string line; std::getline(f, line);) std::cout << line << '\n';
        }
    }
#endif
}
} // namespace
int main(int argc, char** argv) try {
    const auto args = arguments(argc, argv);
    // Product profile: exported before the cached prefill switches are first read.
    const auto product = g4::profile::resolve(args.product.value_or(g4::profile::Name::standard), g4::profile::App::prefill_bench,
                                              {.allow_pending = args.allow_pending});
    g4::profile::apply(product);
    // DevicePrefill implements only g4-wmma-pf-v1 linears; never report another route than the one that runs.
    if (const auto* linears = product.find("prefill_linears"); linears && linears->value != "wmma")
        throw std::invalid_argument("strix-gemma4-prefill-check runs only g4-wmma-pf-v1 linears; requested prefill route '" +
                                    linears->value + "' (measure prefill-exact through strix-gemma4-decode)");
    std::cout << "{\"profile\":" << product.json() << "}\n";
    std::cout << std::setprecision(12);
    telemetry();
    auto model = std::filesystem::is_directory(args.model) ? g4::load_safetensors_model(args.model) : g4::load_gguf_model(args.model);
    const auto weights = g4::bind_host_weights(model);
    const bool quantized = std::any_of(model.tensors.begin(), model.tensors.end(), [](const auto& tensor) {
        return tensor.view.format == g4::WeightFormat::q4_k || tensor.view.format == g4::WeightFormat::q6_k || tensor.view.format == g4::WeightFormat::q8_0;
    });
    auto tables = g4::load_rope_tables(args.rope);
    pf::DevicePrefill prefill(weights, tables.view, pf::Residency::host, args.profile, pf::attention_mode(), args.linears);
    if (!args.linear_policy.empty()) prefill.set_linear_policy(pf::LinearPolicy::parse(args.linear_policy));
    std::unique_ptr<pf::DevicePrefill> canonical_reference;
    if (args.canonical_reference)
        canonical_reference = std::make_unique<pf::DevicePrefill>(prefill.borrow_device_weights(), tables.view, pf::Residency::device, false, pf::AttentionMode::canonical);
    std::unique_ptr<gpu::DeviceTarget> decoder;
    // --linears a8 compares against the decode target running g4-a8-v1 (eight-row passes).
    gpu::DeviceOptions decoder_options;
    if (args.linears == pf::Linears::a8) decoder_options.identity = g4::a8::arithmetic_id;
    if (args.same_weight || args.continuation || args.c4 || args.exact_ab || args.a8_gate || !args.draft_model.empty() || !args.beside.empty())
        decoder = std::make_unique<gpu::DeviceTarget>(weights, tables.view, decoder_options);
    std::unique_ptr<pf::DevicePrefill> profiled, canonical_profile;
    if (args.profile_comparison) {
        profiled = std::make_unique<pf::DevicePrefill>(prefill.borrow_device_weights(), tables.view, pf::Residency::device, true, prefill.stats().attention);
        canonical_profile = std::make_unique<pf::DevicePrefill>(prefill.borrow_device_weights(), tables.view, pf::Residency::device, true, pf::AttentionMode::canonical);
    }
    std::cout << "{\"decode_identity\":\"" << (args.linears == pf::Linears::a8 ? g4::a8::arithmetic_id : "g4-v1") << "\",\"prefill_identity\":\""
              << (args.linears == pf::Linears::exact ? "prefill-exact" : args.linears == pf::Linears::a8 ? "prefill-a8" : pf::identity) << "\",\"model_hash\":\"" << model.model_hash
              << "\",\"rope_hash\":\"" << tables.view.hash << "\",\"cache_dtype\":\"F32\",\"chunk_rows\":512,\"shared_tiling\":"
              << (prefill.stats().tiled ? "true" : "false") << ",\"prepared_weights\":" << (prefill.stats().prepared ? "true" : "false")
              << ",\"direct_mode\":" << prefill.stats().direct
              << ",\"prepared_bytes\":" << prefill.stats().prepared_bytes << ",\"attention_identity\":\""
              << (prefill.stats().attention == pf::AttentionMode::wmma ? pf::attention_identity : "g4-v1")
              << "\",\"attention_packed\":" << (prefill.stats().attention_packed ? "true" : "false")
              << ",\"canonical_kernel\":" << (prefill.stats().canonical_kernel ? "true" : "false")
              << ",\"hf_embedding\":\"" << args.hf_embedding << "\"}\n";
    pf::Fidelity aggregate_hf, aggregate_same_weight;
    unsigned fixture_index = 0;
    for (const auto& path : args.fixtures) {
        std::cout << "{\"fixture_index\":" << fixture_index++ << "}\n";
        const auto fixture = g4::read_npz(path);
        const auto ids = fixture.at("prompt_ids").values<std::int32_t>("<i4");
        if (ids.empty()) throw std::invalid_argument("empty prompt fixture");
        auto lengths = args.lengths.empty() ? std::vector<unsigned>{0} : args.lengths;
        if (args.after_benchmark_checks && std::find(lengths.begin(), lengths.end(), 0) == lengths.end()) lengths.push_back(0);
        for (const auto length : lengths) {
            const bool checks = !args.after_benchmark_checks || !length;
            const auto count = length ? length : unsigned(ids.size());
            if (args.split > count || count + args.continuation > g4::max_positions) throw std::invalid_argument("prompt or continuation bounds");
            std::vector<unsigned> tokens(count);
            for (unsigned i = 0; i < count; ++i) {
                if (ids[i % ids.size()] < 0) throw std::invalid_argument("negative fixture token");
                tokens[i] = unsigned(ids[i % ids.size()]);
            }
            if (args.canonical_ab) { canonical_ab(prefill, model.config, tokens, args.repetitions); continue; }
            if (args.exact_ab) { exact_ab(prefill, *decoder, model.config, tokens, args.repetitions); continue; }
            if (args.a8_gate) { a8_gate(prefill, *decoder, model.config, tokens, args.gate_chunk, args.ref_rows); continue; }
            pf::Result result;
            auto stream = std::make_unique<gpu::DeviceStream>(model.config, count + args.continuation);
            std::vector<double> timings;
            // Untimed repetitions first so short prompts are
            // not measured on the clock ramp; they compute the same state.
            for (unsigned r = 0; r < args.warmup; ++r) {
                gpu::DeviceStream warm(model.config, count + args.continuation);
                (void)prefill.run(warm, tokens);
            }
            for (unsigned r = 0; r < args.repetitions; ++r) {
                if (r) stream = std::make_unique<gpu::DeviceStream>(model.config, count + args.continuation);
                ClockSamples clocks;
                const auto unix_start = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
                const auto start = Clock::now();
                const auto before = prefill.stats();
                result = prefill.run(*stream, tokens);
                const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
                const auto unix_end = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
                clocks.finish();
                timings.push_back(seconds);
                std::cout << "{\"prefill_rows\":" << count << ",\"iteration\":" << r << ",\"seconds\":" << seconds << ",\"tokens_per_s\":" << count / seconds
                          << ",\"start_unix_s\":" << unix_start << ",\"end_unix_s\":" << unix_end << "}\n";
                clocks.print();
                if (args.profile)
                    std::cout << "{\"profiled\":true,\"linear_ms\":" << prefill.stats().linear_ms - before.linear_ms
                              << ",\"attention_ms\":" << prefill.stats().attention_ms - before.attention_ms << "}\n";
                if (args.profile) for (const auto& [name, total] : prefill.stats().kernels) {
                    const auto it = before.kernels.find(name);
                    const auto previous = it == before.kernels.end() ? pf::Stats::Kernel{} : it->second;
                    const double ms = total.ms - previous.ms, flops = total.flops - previous.flops, bytes = total.bytes - previous.bytes;
                    const double matrix_ms = total.matrix_ms - previous.matrix_ms;
                    std::cout << "{\"linear_profile\":\"" << name << "\",\"ms\":" << ms << ",\"calls\":" << total.calls - previous.calls
                              << ",\"conversion_ms\":" << total.conversion_ms - previous.conversion_ms << ",\"matrix_ms\":" << matrix_ms
                              << ",\"tflops\":" << flops / (ms * 1e9) << ",\"matrix_tflops\":" << flops / (matrix_ms * 1e9)
                              << ",\"minimum_gb_s\":" << bytes / (ms * 1e6) << "}\n";
                }
            }
            std::sort(timings.begin(), timings.end());
            const auto middle = timings.size() / 2;
            const double median = timings.size() % 2 ? timings[middle] : 0.5 * (timings[middle - 1] + timings[middle]);
            std::cout << "{\"median_tokens_per_s\":" << count / median << ",\"samples\":" << timings.size() << "}\n";
            {
                std::uint64_t h = 1469598103934665603ULL;
                for (const float x : result.last_logits) { h ^= g4::bit_cast<std::uint32_t>(x); h *= 1099511628211ULL; }
                std::cout << "{\"prefill_rows\":" << count << ",\"committed_digest\":\"" << stream->digest()
                          << "\",\"last_logits_fnv\":\"" << std::hex << h << std::dec << "\"}\n";
            }
            for (const auto beside : args.beside) {
                // The second stream reads the fixture from another offset, so its rows differ from the prompt's.
                std::vector<unsigned> other(beside);
                for (unsigned i = 0; i < beside; ++i) other[i] = unsigned(ids[(i + 7U) % ids.size()]);
                batch_beside(prefill, *decoder, model.config, tokens, other, args.continuation, args.linears == pf::Linears::exact);
            }
            if (args.profile_comparison && length && length >= 2048) {
                profile_attention(*canonical_profile, model.config, tokens, "canonical");
                profile_attention(*profiled, model.config, tokens, "wmma");
            }
            for (const auto& range : prefill.ranges()) std::cout << "{\"linear\":\"" << range.tensor << "\",\"input_abs_max\":" << range.input_maximum << "}\n";
            if (count == ids.size() && !length && fixture.contains("last_logits")) {
                pf::Fidelity m; m.observe(fixture.at("last_logits").values<float>("<f4"), result.last_logits); metric("HF-last-prompt-row", m, false, quantized);
                merge(aggregate_hf, m);
            } else std::cout << "{\"gate\":\"HF-fidelity\",\"verdict\":\"NO-REFERENCE\"}\n";
            if (args.same_weight && checks) {
                gpu::DeviceStream wm(model.config, count), exact(model.config, count);
                pf::Fidelity m;
                for (const auto& chunk : pf::plan(0, count)) {
                    const auto piece = std::span(tokens).subspan(std::size_t(chunk.position), chunk.rows);
                    auto tx = wm.begin(); auto out = prefill.forward(*tx, piece, true); tx->commit(chunk.rows);
                    for (unsigned r = 0; r < chunk.rows; r += 8) {
                        const auto n = std::min(8U, chunk.rows - r);
                        const auto reference = decode(*decoder, exact, piece.subspan(r, n));
                        for (unsigned i = 0; i < n; ++i)
                            m.observe(std::span(reference.logits).subspan(std::size_t(i) * model.config.vocab, model.config.vocab),
                                      std::span(out.all_logits).subspan(std::size_t(r + i) * model.config.vocab, model.config.vocab));
                    }
                }
                metric("WMMA-same-weight-exact-path", m, true);
                merge(aggregate_same_weight, m);
                std::cout << "{\"same_weight_bands_pass\":" << (m.same_weight_bands_pass() ? "true" : "false") << "}\n";
            }
            if (canonical_reference && checks) {
                gpu::DeviceStream wm(model.config, count), canonical(model.config, count);
                pf::Fidelity m;
                for (const auto& chunk : pf::plan(0, count)) {
                    const auto piece = std::span(tokens).subspan(std::size_t(chunk.position), chunk.rows);
                    auto a = wm.begin(), b = canonical.begin();
                    const auto actual = prefill.forward(*a, piece, true), reference = canonical_reference->forward(*b, piece, true);
                    a->commit(chunk.rows); b->commit(chunk.rows);
                    for (unsigned r = 0; r < chunk.rows; ++r)
                        m.observe(std::span(reference.all_logits).subspan(std::size_t(r) * model.config.vocab, model.config.vocab),
                                  std::span(actual.all_logits).subspan(std::size_t(r) * model.config.vocab, model.config.vocab));
                }
                metric("WMMA-attention-versus-canonical-prefill", m, true);
            }
            if (checks && (args.check_split || args.continuation)) {
                gpu::DeviceStream other(model.config, count + args.continuation);
                pf::Prompt prompt(prefill, other);
                prompt.append(std::span(tokens).first(args.split)); prompt.append(std::span(tokens).subspan(args.split));
                const auto split = prompt.finish();
                equal(split.last_logits, result.last_logits); equal(split.tail_taps, result.tail_taps);
                if (other.digest() != stream->digest()) throw std::runtime_error("split committed-state mismatch");
                std::cout << "{\"gate\":\"scheduler-split\",\"split\":" << args.split << ",\"verdict\":\"EXACT\"}\n";
                auto logits = result.last_logits;
                std::vector<unsigned> generated;
                for (unsigned first = 0; first < args.continuation; first += 4) {
                    const auto n = std::min(4U, args.continuation - first);
                    std::vector<unsigned> greedy_tokens;
                    std::vector<g4::ForwardResult> serial;
                    for (unsigned r = 0; r < n; ++r) {
                        const auto token = greedy(logits); greedy_tokens.push_back(token); generated.push_back(token);
                        serial.push_back(decode(*decoder, *stream, {&token, 1})); logits = serial.back().logits;
                    }
                    const auto batch = decode(*decoder, other, greedy_tokens);
                    for (unsigned r = 0; r < n; ++r) {
                        equal(serial[r].logits, std::span(batch.logits).subspan(std::size_t(r) * model.config.vocab, model.config.vocab));
                        equal(serial[r].taps, std::span(batch.taps).subspan(std::size_t(r) * batch.tap_width, batch.tap_width));
                    }
                    if (other.digest() != stream->digest()) throw std::runtime_error("post-prefill decode state mismatch");
                }
                std::cout << "{\"gate\":\"post-WMMA-serial-verify\",\"rows\":" << args.continuation << ",\"verdict\":\"EXACT\"}\n";
                if (!length && !generated.empty() && fixture.contains("gen_ids")) {
                    const auto expected = fixture.at("gen_ids").values<std::int32_t>("<i4");
                    const auto n = std::min(expected.size(), generated.size());
                    std::size_t prefix = 0, matches = 0;
                    while (prefix < n && expected[prefix] >= 0 && generated[prefix] == unsigned(expected[prefix])) ++prefix;
                    for (std::size_t i = 0; i < n; ++i) matches += unsigned(expected[i] >= 0 && generated[i] == unsigned(expected[i]));
                    std::cout << "{\"gate\":\"HF-greedy-continuation\",\"tokens\":" << n << ",\"matches\":" << matches
                              << ",\"matching_prefix\":" << prefix << ",\"qualification\":\"INCOMPLETE\"}\n";
                }
            }
            if (args.c4 && checks) pf::check_c4(prefill, *decoder, model.config, tokens, args.continuation);
            if (!args.draft_model.empty() && checks) pf::check_dflash(prefill, *decoder, weights, tokens, args.continuation, args.draft_model, args.draft_rope);
        }
    }
    if (aggregate_hf.positions) metric("HF-last-prompt-row-aggregate", aggregate_hf, false, quantized);
    if (aggregate_same_weight.positions) metric("WMMA-same-weight-aggregate", aggregate_same_weight, true);
    telemetry();
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
