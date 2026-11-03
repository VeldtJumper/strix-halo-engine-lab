// strix-glm5-rank: GLM-5.3-Flash engine runner (one node or one rank of the TP2 pair).
//
//   strix-glm5-rank MODEL_DIR --world 1                      [modes...]   one node (all heads / full I / full vocab)
//   strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto|usb4gpu|tcp-listen:P|tcp-connect:H:P [modes...]
// Modes (any order, run in this order):
//   --g3 TOKENS.u32 [--layers N]     teacher-forced first N layers (default 6), slot 0: writes the residual after every
//                                    token to --out (raw f32 [T][4][4096]) and prints its FNV-1a + the state hash
//   --decode N --prompt TOKENS.u32   prefill (default auto = exact chunked gemv identity when admissible; --prefill ref = decode row by row)
//                                    then N greedy steps; prints tokens, ms/token
//   --batch R --steps N              R slots decoding together (one row each) for N steps; aggregate tok/s
//   --spec-oracle K --pattern M,M,.. --decode N --prompt T.u32   gate SP1a (needs --spec 1): plain greedy run of N tokens,
//                                    then the speculative loop (verify / accept_greedy / commit) with ORACLE drafts = the
//                                    plain tokens, the draft at depth M_c (cycle c, cyclic) corrupted (M = K: none):
//                                    emitted tokens, every committed row's logits and the flushed frontier-bounded
//                                    state must equal the plain run's bit for bit
//   --spec-run K[,K,...] --decode N --prompt A.u32[,B.u32,...]   paired fixed-count MTP benchmark; requires
//                                    --spec 1 --mtp 1, identity r1b|r1, automatic decode Xfer; SPV1 records
//   --mtp-prefill ref|chunked       spec-run bootstrap (default ref; chunked uses exact gemv, rejects WMMA)
//   --mtp-prefill-gate PROMPT.u32   one-load bitwise bootstrap/state + K=3 continuation gate; --decode N (default 32)
//   --g4 RMAX --steps N --prompt T.u32   G4 mandatory-bitwise half: the probe stream (prompt prefill + N greedy steps) runs
//                                    inside batches R = 1..RMAX, at row/slot 0 and at row/slot R-1, next to synthetic
//                                    streams (token 1000 + r from position 0); its per-step logits bits and tokens must
//                                    equal the R = 1 run's in every placement, and a second R = 1 pass must equal the first
//   --tf TOKENS.u32 --score N --out PREFIX   G4 teacher forcing (needs --spec 1): consumes the file's first T - 1 tokens
//                                    in verify groups of <= 8 rows + full commits (SP1a: bitwise to row-by-row) and
//                                    writes the rank's logits slice of the last N positions (they predict tokens
//                                    T - N .. T - 1) to PREFIX-r<rank>.f32 (f32 [N][vocab_slice]); prints its FNV
//     --tf-stop P --state-out FILE   stop after input P - 1 (flushed), save the frontier-bounded state + P to FILE
//     --state-in FILE                restore that state (P inside) and continue from position P (gate SR: the scored
//                                    rows must equal an uninterrupted run's bit for bit)
//     --tf-check                     then the same positions row by row (step, one row per call): the N scored logits
//                                    rows and the flushed state up to T must be equal bit for bit (the --tf identity)
//   --mem-plan                       memory plan only (design notes): dry-run plan of every device allocation,
//                                    calibration of one hipMalloc's physical cost, admission check; loads no weight, no peer
// Options: --mtp-spec 0|1 (STRIX_GLM_MTP_SPEC; CLI wins), --mtp-k 0..7, --mtp-adaptive 0|1
//   --spec-run also accepts a for adaptive Kmax=3; normal eligible decode defaults to adaptive Kmax=3.
// Options: --slots S --capacity C --mtp 0|1 --raw-linears 0|1 --out FILE --identity r1|r1b|r1b-all
// Defaults (glm_rank_defaults.hpp): every measured optimization is ON and resolves to the production receipt
// (R1b, exact chunked prefill, MTP self-speculation on eligible decode when the model has a complete MTP head, hp host
// path, ...); each has its own off switch (--mtp 0 / --mtp-spec 0: plain decode), --baseline / STRIX_GLM_BASELINE=1
// switches all of them off (explicit settings win), --print-effective-config prints the resolution and exits (no GPU).
// Identity default: r1b = R1 prompts + R1b generated forwards.
// r1 = R1 throughout; r1b-all = R1b throughout (reference prefill required).
// --idcheck N --prompt T.u32: R=1, slot 0; compare r1b with R1 prefill then r1b-all steps.
// Memory (engine, glm_mem.hpp): the load is planned and admitted before the first weight (fails closed with every term
// named); STRIX_GLM_MEMTRACE=FILE writes one memory line per second; the watchdog (STRIX_GLM_WATCHDOG=0 off) aborts the
// rank with exit code 86 near the OOM line.
#include "glm_rank_defaults.hpp"
#include "glm_rank_startup.hpp"
#include "glm_rank_transport.hpp"

#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/glm/engine.hpp"

#include "glm/attn/glm_attn_fast.hpp"
#include "glm/attn/glm_attn_tierb.hpp"
#include "glm/common/glm_math.hpp"
#include "glm/engine/glm_mem.hpp"
#include "glm/engine/glm_r1b.hpp"
#include "glm/engine/glm_fast_moe.hpp"
#include "glm/prefill/glm_pf_path.hpp"
#include "glm/spec/glm_spec_host.hpp"
#include "glm/spec/glm_spec_run.hpp"
#include "glm/xfer/glm_xfer_exchange.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
namespace ge = strix::glm::engine;

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

std::vector<std::uint32_t> read_u32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("read " + path);
    std::vector<std::uint32_t> v(static_cast<std::size_t>(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    return v;
}

using strix::glm::rank_transport::open_transport;

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered: a killed run still leaves every printed line in its log
    std::setvbuf(stderr, nullptr, _IONBF, 0);  // unbuffered (the C default, made explicit): progress and errors land at once
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: %s MODEL_DIR --world 1|2 [--rank R --transport T] [modes] [--identity r1|r1b|r1b-all] [--idcheck N --prompt T.u32]\n"
                                 "identity: r1 throughout; r1b (default) R1 prompts/R1b generation; "
                                 "r1b-all throughout (reference prefill)\n", argv[0]);
            return 2;
        }
        const auto t_start = Clock::now();
        ge::Options opt;
        opt.model_dir = argv[1];
        std::string transport_spec, g3_tokens, prompt, out_path, tf_tokens, state_out, state_in;
        std::string mtp_prefill = "ref", mtp_prefill_gate;
        std::string prefill_mode = "auto", pf_gate, pf_ttft, pf_ttft_mode = "gemv", pf_null_study, pf_nulls;
        bool reference_gate = false;
        bool pf_before_load = false, pf_swap_moe = false;
        std::uint64_t tf_stop = 0;
        unsigned idcheck_n = 0;
        unsigned prompt_max = 0, ab_n = 0, aba_n = 0, aba3_n = 0, aba_xch_n = 0;
        std::string aba_xch_rows = "1";
        bool g3_sweep = false;
        std::string logits_out;
        unsigned g3_layers = 6, decode_n = 0, batch_r = 0, batch_steps = 0, spec_k = 0, spec_run = 0, g4_rmax = 0, tf_score = 0;
        std::vector<unsigned> pattern;
        bool mem_plan = false, profile = false, tf_check = false, print_config = false;
        std::string xch_kind = "comm";  // comm = CommExchange (Communicator) | xfer = XferExchange
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };
            if (a == "--ab" || a == "--aba" || a == "--g3-sweep" || a == "--aba-attn3" || a == "--aba-xch" ||
                a == "--g4" || a == "--tf" || a == "--spec-oracle" || a == "--spec-run" || a == "--state-in" || a == "--state-out" ||
                a == "--mtp-prefill-gate" || a == "--pf-gate" || a == "--pf-ttft" || a == "--pf-null-study")  // the pf modes install their own paths
                reference_gate = true;
            if (a == "--world") opt.world = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--rank") opt.rank = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--transport") transport_spec = next();
            else if (a == "--g3") g3_tokens = next();
            else if (a == "--layers") g3_layers = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--decode") decode_n = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--prompt") prompt = next();
            else if (a == "--prompt-max") prompt_max = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--ab") ab_n = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--g3-sweep") g3_sweep = true;
            else if (a == "--aba-xch") aba_xch_n = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--aba-xch-rows") aba_xch_rows = next();
            else if (a == "--aba-attn3") aba3_n = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--idcheck") {
                idcheck_n = static_cast<unsigned>(std::stoul(next()));
                if (idcheck_n == 0) throw std::invalid_argument("--idcheck needs N >= 1");
            }
            else if (a == "--aba") aba_n = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--logits-out") logits_out = next();
            else if (a == "--batch") batch_r = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--steps") batch_steps = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--slots") opt.slots = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--capacity") opt.capacity = std::stoull(next());
            else if (a == "--mtp-prefill") mtp_prefill = next();
            else if (a == "--mtp-prefill-gate") mtp_prefill_gate = next();
            else if (a == "--prefill") prefill_mode = next();  // auto (default) | ref | gemv (exact) | wmma (D10)
            else if (a == "--pf-install-before-load") pf_before_load = true;
            else if (a == "--pf-swap-moe") pf_swap_moe = true;  // --pf-gate test (world 2): a fresh fast MoE seam set right before the gemv run  // test: --prefill path set before load() (installed at its end)
            else if (a == "--pf-gate") pf_gate = next();  // prompt file: ref vs gemv (bitwise) vs wmma (D10) prefill, one process
            else if (a == "--pf-ttft") pf_ttft = next();  // comma-separated prompt files: chunked prefill TTFT
            else if (a == "--pf-ttft-mode") pf_ttft_mode = next();  // gemv (default, exact identity) | wmma | wmma,gemv (each prompt in both, one load)
            else if (a == "--pf-null-study") pf_null_study = next();  // prompt file: exact (gemv) vs wmma vs 1-ulp nulls
            else if (a == "--pf-null") pf_nulls = next();  // comma-separated LAYER:POS:ELEM 1-ulp residual flips (test hook)
            else if (a == "--max-rows") opt.max_rows = static_cast<unsigned>(std::stoul(next()));  // chunk rows of --prefill gemv|wmma
            else if (a == "--mtp") opt.mtp = std::stoul(next()) != 0;
            else if (a == "--raw-linears") opt.raw_linears = std::stoul(next()) != 0;
            else if (a == "--spec") opt.spec = std::stoul(next()) != 0;
            else if (a == "--spec-oracle") spec_k = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--spec-run") { (void)next(); spec_run = 1; }
            else if (a == "--mtp-spec" || a == "--mtp-k" || a == "--mtp-adaptive") (void)next();
            else if (a == "--pattern") {
                std::string v = next();
                for (std::size_t p = 0; p < v.size();) {
                    const std::size_t q = v.find(',', p);
                    pattern.push_back(static_cast<unsigned>(std::stoul(v.substr(p, q - p))));
                    p = q == std::string::npos ? v.size() : q + 1;
                }
            }
            else if (a == "--out") out_path = next();
            else if (a == "--mem-plan") mem_plan = true;
            else if (a == "--profile") profile = true;
            else if (a == "--xch") xch_kind = next();
            else if (a == "--decode-xch") {
                const std::string v = next();
                if (v != "comm" && v != "xfer") throw std::invalid_argument("--decode-xch comm|xfer");
                opt.decode_xfer = v == "xfer";
            }
            else if (a == "--identity") {
                const std::string v = next();
                if (v != "r1" && v != "r1b" && v != "r1b-all") throw std::invalid_argument("--identity r1|r1b|r1b-all");
                opt.identity = v == "r1" ? ge::IdentityMode::kR1 : v == "r1b" ? ge::IdentityMode::kR1b : ge::IdentityMode::kR1bAll;
            }
            else if (a == "--g4") g4_rmax = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--tf") tf_tokens = next();
            else if (a == "--score") tf_score = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--tf-check") tf_check = true;
            else if (a == "--tf-stop") tf_stop = std::stoull(next());
            else if (a == "--state-out") state_out = next();
            else if (a == "--state-in") state_in = next();
            else if (a == "--baseline") {}  // resolved by rank_defaults (old path of every default-on item)
            else if (a == "--print-effective-config") print_config = true;
            else throw std::invalid_argument("unknown option " + a);
        }
        if (mtp_prefill != "ref" && mtp_prefill != "chunked")
            throw std::invalid_argument("--mtp-prefill must be ref or chunked");
        namespace rd = strix::glm::rank_defaults;
        const auto request = rd::scan(std::vector<std::string>(argv + 2, argv + argc));
        auto eff = rd::resolve(request, rd::process_env(), rd::probe_model_mtp(argv[1]));
        rd::fallback_without_handshake(eff, request, rd::process_env());
        if (print_config) {  // dry run: no environment change, no GPU, no peer
            std::fputs(rd::print(eff).c_str(), stdout);
            return 0;
        }
        rd::apply_env(eff, [](const char* k, const char* v) { (void)::setenv(k, v, 0); });
        opt.mtp = eff.mtp.value == "1";
        opt.max_rows = static_cast<unsigned>(std::stoul(eff.max_rows.value));
        opt.auto_rows = rd::max_rows_auto(eff);
        opt.spec = eff.spec.value == "1";
        bool mtp_spec_on = eff.mtp_spec.value == "1";
        if (!mtp_prefill_gate.empty() && (!opt.spec || !opt.mtp))
            throw std::invalid_argument("--mtp-prefill-gate requires --spec 1 --mtp 1");
        if (eff.identity.source == "baseline") opt.identity = ge::IdentityMode::kR1;
        if (eff.decode_xch.source == "baseline") opt.decode_xfer = false;
        if (eff.baseline) std::printf("baseline: default-on optimizations off (old path), explicit settings kept\n");
        prefill_mode = eff.prefill_requested.value;
        if (prefill_mode != "ref" && prefill_mode != "gemv" && prefill_mode != "wmma" && prefill_mode != "auto")
            throw std::invalid_argument("--prefill / STRIX_GLM_PREFILL must be ref, gemv, wmma or auto");
        const std::string prefill_requested = prefill_mode;
        // r1b-all runs prompts in R1b (reference prefill); r1 and the default r1b run Engine::prefill in R1, so the exact
        // chunked prefill (gemv identity) keeps its bits there. STRIX_GLM_IDENTITY overrides --identity (as in the engine).
        const char* id_env = std::getenv("STRIX_GLM_IDENTITY");
        const bool r1b_all = id_env != nullptr ? std::string(id_env) == "r1b-all" : opt.identity == ge::IdentityMode::kR1bAll;
        const char* smoke = std::getenv("STRIX_GLM_SMOKE_HEAD");
        const bool smoke_head = smoke != nullptr && std::string(smoke) == "1";
        const bool g3_only = !g3_tokens.empty() && decode_n == 0 && batch_r == 0 && idcheck_n == 0;
        if (g3_only) opt.layers = g3_layers;  // the G3 run loads only the layers it needs (no head)
        else if (smoke_head)
            opt.layers = g3_layers;  // test only: truncated stack + head; the engine refuses it unless the model is synthetic
        if (pf_before_load && prefill_mode != "gemv" && prefill_mode != "wmma")
            throw std::invalid_argument("--pf-install-before-load needs --prefill gemv|wmma");
        if (prefill_mode == "auto")
            prefill_mode = rd::resolve_prefill_auto(opt.mtp, opt.spec, g3_only, smoke_head, mem_plan, r1b_all, idcheck_n, reference_gate, mtp_spec_on);
        if (prefill_mode != eff.prefill) throw std::logic_error("prefill resolution differs from --print-effective-config");
        if (batch_r > opt.slots) opt.slots = batch_r;
        if (g4_rmax > opt.slots) opt.slots = g4_rmax;
        if (mem_plan) {  // no peer, no weights: the engine stops after the admission check
            (void)::setenv("STRIX_GLM_LOAD_TEST", "1", 1);
            (void)::setenv("STRIX_GLM_MEM_PLAN_ONLY", "1", 1);
        }
        // The kernel OOM killer cannot see GTT (load measurement killed sshd and the desktop, never the rank): make the rank its first
        // victim (STRIX_GLM_OOM_SCORE_ADJ, default 1000; 'keep' leaves it).
        const char* oom = std::getenv("STRIX_GLM_OOM_SCORE_ADJ");
        if (!(oom != nullptr && std::string(oom) == "keep") && !strix::glm::engine::mem::set_oom_score_adj(oom ? std::atoi(oom) : 1000))
            std::fprintf(stderr, "warning: could not set oom_score_adj\n");
        const char* frag = std::getenv("HSA_DISABLE_FRAGMENT_ALLOCATOR");
        const char* trace = std::getenv("STRIX_GLM_MEMTRACE");
        std::printf("rank %u: pid %d, HSA_DISABLE_FRAGMENT_ALLOCATOR=%s, memtrace %s\n", opt.rank, static_cast<int>(::getpid()),
                    frag ? frag : "(unset)", trace ? trace : "off");
        if (opt.max_rows == 0 || opt.max_rows > 65536) throw std::invalid_argument("--max-rows must be 1..65536");

        // Cache exactly the truncated token lists consumed by normal decode, including their order.
        std::vector<std::vector<std::uint32_t>> decode_prompts;
        std::uint32_t prompt_hash = 2166136261U;
        // Gate modes (--mtp-prefill-gate, ...) return before the decode loop and may
        // carry --decode without --prompt.
        if (decode_n > 0 && spec_k == 0 && spec_run == 0 && !prompt.empty()) {
            for (std::size_t a = 0; a <= prompt.size();) {
                const auto b = std::min(prompt.find(',', a), prompt.size());
                auto toks = read_u32(prompt.substr(a, b - a));
                if (prompt_max > 0 && toks.size() > prompt_max) toks.resize(prompt_max);
                // boundaries are part of the agreement: each prompt's length, then its tokens, then the prompt count
                auto mix = [&](std::uint32_t w) {
                    for (unsigned byte = 0; byte < 4; ++byte) prompt_hash = (prompt_hash ^ ((w >> (8 * byte)) & 255U)) * 16777619U;
                };
                mix(static_cast<std::uint32_t>(toks.size()));
                for (const auto token : toks) mix(token);
                decode_prompts.push_back(std::move(toks));
                a = b + 1;
            }
            for (unsigned byte = 0; byte < 4; ++byte)
                prompt_hash = (prompt_hash ^ ((static_cast<std::uint32_t>(decode_prompts.size()) >> (8 * byte)) & 255U)) * 16777619U;
        }
        std::unique_ptr<strix::dist::Communicator> comm;
        std::shared_ptr<strix::glm::xfer::XferExchange> xfer_exchange;
        const char* lt = std::getenv("STRIX_GLM_LOAD_TEST");
        if (opt.world == 2 && !(lt != nullptr && std::string(lt) == "1")) {
            if (transport_spec.empty()) throw std::invalid_argument("--world 2 needs --transport");
            (void)::setenv("STRIX_DIST_USB4_TIMEOUT_MS", "600000", 0);
            // the DS/MiniMax 'hp' host path (STRIX_DIST_CRC_PARALLEL / HOST_CRC_OVERLAP / HOST_CHUNK_BYTES) is set by rank_defaults
            (void)::setenv("STRIX_DIST_WAIT_TIMEOUT_MS", "600000", 0);
            std::string receipt;
            auto transport = open_transport(transport_spec, opt.rank, receipt);
            if (!receipt.empty()) std::printf("transport: %s\n", receipt.c_str());
            if (xch_kind == "xfer") {  // the module owns the transport; no Communicator (Options::external_exchange)
                if (opt.spec) throw std::invalid_argument("--xch xfer has no speculative merge yet: use --xch comm with --spec 1");
                xfer_exchange = std::make_shared<strix::glm::xfer::XferExchange>(std::move(transport), opt.rank, 8U);
                opt.external_exchange = true;
                std::printf("connected: rank %u exchange xfer %.1f s\n", opt.rank, secs(t_start, Clock::now()));
            } else {
            // one exchange carries up to max(8, max_rows) rows of 4096 floats (a chunked prefill exchanges whole chunks)
            opt.exchange_rows = rd::max_rows_auto(eff) ? rd::kFallbackMaxRows : opt.max_rows;
            comm = strix::glm::rank_transport::make_communicator(opt.rank, std::move(transport), opt.exchange_rows);
            std::printf("connected: rank %u transport %s %.1f s\n", opt.rank, comm->transport_name().c_str(), secs(t_start, Clock::now()));
            }
        }
        if (pf_swap_moe && (opt.world != 2 || pf_gate.empty())) throw std::invalid_argument("--pf-swap-moe needs --world 2 and --pf-gate");
        const bool startup_handshake = rd::has_startup_handshake(opt.world, xch_kind, decode_n > 0, reference_gate);
        ge::Engine eng(opt, comm.get());
        if (xfer_exchange) eng.set_exchange(xfer_exchange);
        bool startup_agreed = false;
        auto startup_agree = [&]() {
            startup_agreed = true;
            // External Xfer must never bind its stream through a host control merge.
            if (!startup_handshake) return;
            const bool eligible = mtp_spec_on && opt.mtp && opt.spec && opt.layers == 45 &&
                                  (prefill_mode == "ref" || prefill_mode == "gemv");
            const std::uint32_t packed = eff.kmax | (eff.adaptive ? 1U << 4 : 0U) |
                ((prefill_mode == "ref" ? 1U : prefill_mode == "gemv" ? 2U : 3U) << 5) |
                (static_cast<std::uint32_t>(eng.identity_mode()) << 7) |
                (eff.env.at("STRIX_GLM_SPEC_XCH").value == "xfer" ? 1U << 9 : 0U) |
                (eng.decode_xfer_enabled() ? 1U << 10 : 0U);
            const auto result = rd::startup_agreement(
                {eligible ? 1U : 0U, packed, opt.max_rows, decode_n,
                 static_cast<std::uint32_t>(opt.capacity), static_cast<std::uint32_t>(opt.capacity >> 32), opt.slots, prompt_hash},
                rd::max_rows_auto(eff), startup_handshake, [&](auto* rows, unsigned n, auto& ctl) { eng.exchange()->merge(rows, n, ctl, nullptr); });
            rd::agree_max_rows(eff, result.rows);
            opt.max_rows = static_cast<unsigned>(std::stoul(eff.max_rows.value));
            mtp_spec_on = result.eligible;
            if (!mtp_spec_on)
                std::printf("rank %u mtp-spec: unavailable on at least one rank (head or precondition); using plain decode\n", opt.rank);
        };
        const auto lr = eng.load([&](const ge::mem::LoadAdmission& admission) {
            namespace pf = strix::glm::pf;
            namespace mem = strix::glm::engine::mem;
            if (rd::max_rows_auto(eff)) {
                const auto b = pf::prefill_budget(64 / opt.world, 64 / opt.world, opt.capacity, rd::kAutoMaxRows, opt.mtp);
                auto terms = admission.terms;
                // Account for all later owners, including the full grown communication pools. This is conservative:
                // the current 4096 pools are already charged to the snapshot; growth frees them first.
                const auto comm_bytes = comm && !admission.recheck ? std::uint64_t{rd::kAutoMaxRows} * (3 * 4096 * 4 + 16) + 16384 : 0;
                terms.push_back({"communication growth and chunk staging", 3 * comm_bytes, 0});
                if (prefill_mode != "ref") {
                    // Bound per-allocation physical rounding too; the path checks its allocation-count bound.
                    const auto quantum = admission.model.physical(1);
                    const auto path = admission.model.physical(b.path) + pf::kPrefillAllocationBound * quantum;
                    const auto workspace = admission.model.physical(b.workspace) + 4 * quantum;
                    const auto install = admission.model.physical(admission.prefill_install) + 256 * quantum;
                    terms.push_back({"prefill path (physical bound)", path, 0});
                    terms.push_back({"prefill workspace and install", workspace + install, 0});
                }
                auto margins = admission.margins;
                margins.ram = std::max({margins.ram, rd::kMaxRowsSafetyBytes, pf::admission_margin_bytes()});
                rd::select_max_rows(eff, [&] { return mem::fits(terms, admission.snapshot, margins, admission.device_free); });
                opt.max_rows = static_cast<unsigned>(std::stoul(eff.max_rows.value));
                std::fprintf(stderr, "rank %u max-rows admission: %s; candidate %u, path %.2f GiB, workspace %.2f GiB, install %.2f GiB\n",
                             opt.rank, mem::describe(admission.snapshot).c_str(), opt.max_rows,
                             mem::gib(b.path), mem::gib(b.workspace), mem::gib(admission.prefill_install));
                if (!admission.recheck) startup_agree();
                else {
                    const auto rows = rd::recheck_rows(opt.max_rows, comm != nullptr,
                        [&](auto* proposal, unsigned n, auto& ctl) { eng.exchange()->merge(proposal, n, ctl, nullptr); });
                    rd::agree_max_rows(eff, rows);
                    opt.max_rows = static_cast<unsigned>(std::stoul(eff.max_rows.value));
                }
                if (comm) comm->resize_bytes(std::size_t{opt.max_rows} * 4096 * sizeof(float) + 4096);
            }
            std::fprintf(stderr, "rank %u max-rows=%u (%s)\n", opt.rank, opt.max_rows, eff.max_rows.source.c_str());
            if (pf_before_load && (!opt.auto_rows || admission.recheck)) {
                eng.set_prefill(pf::make_prefill_path(64 / opt.world, 64 / opt.world, opt.capacity, opt.max_rows,
                                                      prefill_mode == "gemv" ? pf::TierB::gemv : pf::TierB::wmma));
                std::printf("rank %u prefill %s set before load\n", opt.rank, prefill_mode.c_str());
            }
            return opt.max_rows;
        });
        const auto load_identity = eng.identity_mode();
        std::printf("rank %u: load %.1f s, %.1f GiB read, device %.1f GiB (layers %u, mtp %d) identity %s decode-xch %s%s%s%s%s", opt.rank, lr.seconds,
                    lr.gib_read, lr.device_gib, opt.layers, opt.mtp ? 1 : 0, ge::identity_mode_name(load_identity),
                    xfer_exchange ? "xfer (external)" : opt.world == 1 ? "local" : eng.decode_xfer_enabled() ? "xfer" : "comm",
                    std::getenv("STRIX_GLM_HC_NORM") ? " hc-norm=" : "",
                    std::getenv("STRIX_GLM_HC_NORM") ? std::getenv("STRIX_GLM_HC_NORM") : "",
                    std::getenv("STRIX_GLM_MOE_PIPE") ? " moe-pipe=" : "",
                    std::getenv("STRIX_GLM_MOE_PIPE") ? std::getenv("STRIX_GLM_MOE_PIPE") : "");
        if (load_identity != ge::IdentityMode::kR1) {
            const char* hc = std::getenv("STRIX_GLM_R1B_HC");
            std::printf(" r1b-dense %d r1b-hc %d", strix::glm::r1b::dense_variant(), hc ? std::atoi(hc) : 4);
        }
        std::printf("\n");
        std::fflush(stdout);
        if (mem_plan) {
            std::printf("rank %u prefill %s (requested %s), pipeline %u\n", opt.rank, prefill_mode.c_str(), prefill_requested.c_str(), eng.prefill_pipeline());
            std::printf("rank %u: mem-plan ADMITTED (nothing loaded)\n", opt.rank);
            return 0;
        }
        if (!startup_agreed) startup_agree();
        if (mtp_spec_on && profile)
            std::printf("rank %u mtp-spec: --profile covers plain steps only; skipping decode profile\n", opt.rank);
        if (prefill_mode != "ref" && !pf_before_load) {  // chunked prefill, installed after the load so its admission check sees real free memory
            namespace pf = strix::glm::pf;
            const unsigned heads = 64 / opt.world;
            const auto b = pf::prefill_budget(heads, heads, opt.capacity, opt.max_rows, opt.mtp);
            std::shared_ptr<ge::PrefillPath> path;
            try {
                path = pf::make_prefill_path(heads, heads, opt.capacity, opt.max_rows,
                                             prefill_mode == "gemv" ? pf::TierB::gemv : pf::TierB::wmma);
            } catch (const std::exception& e) {
                // world 2: the ranks must run the same prefill (chunk vs row exchanges pair up), so no per-rank fallback
                if (prefill_requested != "auto" || opt.world != 1) throw;
                std::printf("rank %u prefill auto: exact chunked path not admitted (%s) -> ref (same bits, slower)\n", opt.rank, e.what());
                prefill_mode = "ref";
            }
            if (path) {
                eng.set_prefill(std::move(path));
                std::printf("rank %u prefill %s: max_rows %u, pipeline %u, path %.2f GiB + workspace %.2f GiB (engine rows %.2f GiB)\n", opt.rank,
                            prefill_mode.c_str(), opt.max_rows, eng.prefill_pipeline(), static_cast<double>(b.path) / double(1ULL << 30),
                            static_cast<double>(b.workspace) / double(1ULL << 30), static_cast<double>(b.engine_rows) / double(1ULL << 30));
            }
        }

        std::printf("rank %u prefill %s (requested %s), pipeline %u\n", opt.rank, prefill_mode.c_str(), prefill_requested.c_str(), eng.prefill_pipeline());

        if (!g3_tokens.empty() && g3_sweep) {  // per-layer residuals in both identities, one load: OUT-<r1|r1b>.f32 [T][L][4][4096]
            const auto tokens = read_u32(g3_tokens);
            for (int b = 0; b < 2; ++b) {
                eng.set_identity_mode(b == 1 ? ge::IdentityMode::kR1bAll : ge::IdentityMode::kR1);
                eng.reset(0);
                const auto res = eng.gate_residual_layers(tokens, g3_layers);
                const std::string path = out_path + (b ? "-r1b.f32" : "-r1.f32");
                std::ofstream f(path, std::ios::binary);
                f.write(reinterpret_cast<const char*>(res.data()), static_cast<std::streamsize>(res.size() * 4));
                std::printf("rank %u g3-sweep %s: layers %u T %zu residuals fnv %016llx -> %s\n", opt.rank, b ? "r1b" : "r1", g3_layers,
                            tokens.size(), static_cast<unsigned long long>(fnv(res.data(), res.size() * 4)), path.c_str());
            }
            eng.set_identity_mode(load_identity);
        } else if (!g3_tokens.empty()) {
            const auto tokens = read_u32(g3_tokens);
            const auto t0 = Clock::now();
            const auto res = eng.gate_residual(tokens, g3_layers);
            const auto st = eng.state_bytes(0);
            std::printf("rank %u g3: layers %u T %zu residual fnv %016llx state fnv %016llx (%zu B) %.1f s\n", opt.rank, g3_layers,
                        tokens.size(), static_cast<unsigned long long>(fnv(res.data(), res.size() * 4)),
                        static_cast<unsigned long long>(fnv(st.data(), st.size())), st.size(), secs(t0, Clock::now()));
            if (!out_path.empty()) {
                std::ofstream f(out_path, std::ios::binary);
                f.write(reinterpret_cast<const char*>(res.data()), static_cast<std::streamsize>(res.size() * 4));
                std::ofstream fs(out_path + ".state", std::ios::binary);
                fs.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size()));
            }
        }
        if (spec_k > 0) {
            if (!opt.spec || decode_n < 2 || prompt.empty() || spec_k > 7) throw std::invalid_argument("--spec-oracle needs --spec 1 --decode N --prompt, K <= 7");
            if (pattern.empty()) pattern.push_back(spec_k);
            const auto toks = read_u32(prompt);
            const std::uint32_t V = eng.vocab_slice();
            auto logits_fnv = [&](unsigned rows) {  // FNV of each row's rank logits slice
                std::vector<float> lg(static_cast<std::size_t>(rows) * V);
                if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) throw std::runtime_error("logits copy");
                std::vector<std::uint64_t> h(rows);
                for (unsigned r = 0; r < rows; ++r) h[r] = fnv(lg.data() + static_cast<std::size_t>(r) * V, static_cast<std::size_t>(V) * 4);
                return h;
            };
            // plain run: T[0] = token after the prompt, T[i] after position L + i - 1; plain_lfnv[i] = logits producing T[i + 1]
            eng.reset(0);
            std::vector<std::uint32_t> T{eng.prefill(0, toks, 0)};
            std::vector<std::uint64_t> plain_lfnv;
            const std::uint64_t L = toks.size();
            for (unsigned i = 1; i < decode_n + 8; ++i) {
                const ge::Row r{0, T.back(), L + i - 1};
                T.push_back(eng.step({&r, 1})[0]);
                plain_lfnv.push_back(logits_fnv(1)[0]);
            }
            // speculative run
            eng.reset(0);
            std::vector<std::uint32_t> out{eng.prefill(0, toks, 0)};
            std::uint64_t p = L;
            unsigned cycle = 0, accepted = 0, mism = 0;
            const auto t0 = Clock::now();
            while (out.size() < decode_n) {
                const unsigned K = static_cast<unsigned>(std::min<std::uint64_t>(spec_k, opt.capacity - 1 - p));
                const std::size_t i = p - L;  // out[i] = x0 = T[i]
                std::vector<std::uint32_t> rows{out.back()}, drafts;
                const unsigned depth = pattern[cycle % pattern.size()];
                for (unsigned j = 1; j <= K; ++j) {
                    std::uint32_t d = T[i + j];
                    if (j == depth + 1) d = (d + 1) % (V * opt.world);  // corrupt the draft at depth `depth`
                    drafts.push_back(d);
                    rows.push_back(d);
                }
                ge::Exchange::MergeCtl ctl;
                ctl.cycle = cycle;
                ctl.kind = 1;
                const auto v = eng.verify(0, rows, p, ctl);
                const auto lf = logits_fnv(K + 1);
                unsigned first_flag = strix::glm::spec::kNoFlag;
                for (unsigned j = 0; j <= K; ++j)
                    if (v.flag[j] != 0 && first_flag == strix::glm::spec::kNoFlag) first_flag = j;
                if (v.replay_flag) throw std::runtime_error("replay row flagged");
                const auto acc = strix::glm::spec::accept_greedy(drafts, v.id, first_flag, {});
                if (acc.abort) throw std::runtime_error("spec abort");
                for (unsigned j = 0; j < acc.rows; ++j)  // committed rows: logits must equal the plain run's at p + j
                    if (lf[j] != plain_lfnv.at(i + j)) ++mism;
                for (unsigned j = 0; j < acc.m; ++j) out.push_back(drafts[j]);
                if (acc.has_bonus) out.push_back(acc.bonus);
                eng.commit(0, acc.rows, p);
                p += acc.rows;
                accepted += acc.m;
                ++cycle;
            }
            const double dt = secs(t0, Clock::now());
            out.resize(decode_n);
            bool tok_eq = true;
            for (unsigned i = 0; i < decode_n; ++i) tok_eq = tok_eq && out[i] == T[i];
            // flushed spec state at its frontier p vs a plain re-run to the same frontier
            eng.flush(0);
            const std::uint64_t P_end = p;
            const auto spec_state = eng.state_bytes_upto(0, P_end);
            eng.reset(0);
            eng.prefill(0, toks, 0);
            for (std::uint64_t q = L; q < P_end; ++q) {
                const ge::Row r{0, T.at(q - L), q};
                (void)eng.step({&r, 1});
            }
            const bool state_eq = eng.state_bytes_upto(0, P_end) == spec_state;
            std::printf("rank %u spec-oracle: K %u cycles %u accepted %u tokens %u (%.2f tok/cycle) %.2f ms/cycle | tokens %s, committed-row logits mismatches %u, flushed state %s (frontier %llu)\n",
                        opt.rank, spec_k, cycle, accepted, decode_n, static_cast<double>(decode_n - 1) / cycle, 1000.0 * dt / cycle,
                        tok_eq ? "EQUAL" : "DIFFER", mism, state_eq ? "EQUAL" : "DIFFER", static_cast<unsigned long long>(P_end));
            if (!tok_eq || mism != 0 || !state_eq) throw std::runtime_error("SP1a FAIL");
            std::printf("rank %u SP1a PASS\n", opt.rank);
        }
        if (!mtp_prefill_gate.empty()) {
            namespace pf = strix::glm::pf;
            namespace sp = strix::glm::spec;
            if (eng.identity_mode() == ge::IdentityMode::kR1bAll || prefill_mode == "wmma")
                throw std::invalid_argument("MTP prefill gate requires R1 prompts and the exact gemv identity");
            const auto toks = read_u32(mtp_prefill_gate);
            const unsigned N = decode_n == 0 ? 32U : decode_n;
            if (toks.empty() || toks.size() > opt.capacity || N + 8ULL > opt.capacity - toks.size())
                throw std::invalid_argument("MTP prefill gate needs a nonempty prompt and L + N + 8 <= capacity");
            for (auto t : toks)
                if (t >= 154880U) throw std::invalid_argument("MTP prefill gate token outside vocabulary");
            struct Result {
                std::uint32_t y0 = 0;
                ge::Engine::MtpOut mtp;
                std::vector<std::uint8_t> hp, state;
                std::vector<std::uint32_t> tokens;
                double bootstrap_ms = 0, decode_ms = 0;
            };
            auto run = [&](bool chunked) {
                eng.reset(0);
                const auto identity = eng.identity_mode();
                std::uint32_t seq = 0;
                auto ctl = [&](unsigned kind) {
                    ge::Exchange::MergeCtl c;
                    c.kind = kind;
                    c.step = seq++;
                    return c;
                };
                Result r;
                const auto t0 = Clock::now();
                if (chunked) {
                    auto c = ctl(2);
                    const auto b = eng.prefill_mtp(0, toks, 0, c);
                    r.y0 = b.y0;
                    r.mtp = b.mtp;
                } else {
                    struct Scope {
                        ge::Engine& e;
                        ge::IdentityMode saved;
                        explicit Scope(ge::Engine& engine) : e(engine), saved(e.identity_mode()) {
                            e.set_identity_mode(ge::IdentityMode::kR1);
                        }
                        ~Scope() { e.set_identity_mode(saved); }
                    } scope(eng);
                    for (std::size_t j = 0; j < toks.size(); ++j) {
                        const ge::Row row{0, toks[j], j};
                        r.y0 = eng.step({&row, 1})[0];
                        const auto t = j + 1 < toks.size() ? toks[j + 1] : r.y0;
                        auto c = ctl(2);
                        r.mtp = eng.mtp(0, eng.last_hidden(), {&t, 1}, j, c);
                    }
                }
                const auto t1 = Clock::now();
                r.bootstrap_ms = 1000.0 * secs(t0, t1);
                r.hp.resize(4096 * sizeof(float));
                if (hipMemcpy(r.hp.data(), eng.mtp_hidden(), r.hp.size(), hipMemcpyDeviceToHost) != hipSuccess)
                    throw std::runtime_error("MTP prefill gate hidden copy");
                r.state = eng.state_bytes(0);  // deliberately includes every unused DSA/indexer suffix
                r.tokens.push_back(r.y0);
                const float* hp = eng.mtp_hidden();
                std::vector<std::uint32_t> drafts;
                if (!r.mtp.flagged) drafts.push_back(r.mtp.id[0]);
                std::uint64_t p = toks.size();
                const auto td = Clock::now();
                while (r.tokens.size() < N) {
                    const auto remaining = N - static_cast<unsigned>(r.tokens.size());
                    const unsigned limit = std::min(3U, remaining - 1);
                    if (drafts.size() > limit) drafts.resize(limit);
                    while (!drafts.empty() && drafts.size() < limit) {
                        const auto t = drafts.back();
                        auto c = ctl(3);
                        const auto m = eng.mtp(0, hp, {&t, 1}, p + drafts.size() - 1, c);
                        if (m.flagged) break;
                        drafts.push_back(m.id[0]);
                        hp = eng.mtp_hidden();
                    }
                    std::vector<std::uint32_t> rows{r.tokens.back()};
                    rows.insert(rows.end(), drafts.begin(), drafts.end());
                    auto c = ctl(1);
                    const auto v = eng.verify(0, rows, p, c);
                    unsigned first_flag = sp::kNoFlag;
                    for (unsigned j = 0; j < v.flag.size(); ++j)
                        if (v.flag[j] != 0 && first_flag == sp::kNoFlag) first_flag = j;
                    if (v.replay_flag) throw std::runtime_error("MTP prefill gate replay flagged");
                    const auto a = sp::accept_greedy(drafts, v.id, first_flag, sp::Terminal{sp::kNoFlag, remaining});
                    if (a.abort) throw std::runtime_error("MTP prefill gate required row flagged");
                    r.tokens.insert(r.tokens.end(), drafts.begin(), drafts.begin() + a.m);
                    if (a.has_bonus) r.tokens.push_back(a.bonus);
                    eng.commit(0, a.rows, p);
                    if (!a.stop) {
                        if (!a.has_bonus) throw std::logic_error("MTP prefill gate missing bonus");
                        std::vector<std::uint32_t> cu(drafts.begin(), drafts.begin() + a.m);
                        cu.push_back(a.bonus);
                        drafts.clear();
                        auto cm = ctl(2);
                        const auto m = eng.mtp(0, eng.last_hidden(), cu, p, cm);
                        if (!m.flagged) {
                            drafts.push_back(m.id[a.m]);
                            hp = eng.mtp_hidden() + static_cast<std::size_t>(a.m) * 4096;
                        }
                    }
                    p += a.rows;
                }
                eng.flush(0);
                r.decode_ms = 1000.0 * secs(td, Clock::now());
                if (eng.identity_mode() != identity) throw std::logic_error("MTP bootstrap identity was not restored");
                return r;
            };
            eng.set_prefill(ge::reference_prefill());
            const auto ref = run(false);
            const unsigned heads = 64 / opt.world;
            eng.set_prefill(pf::make_prefill_path(heads, heads, opt.capacity, opt.max_rows, pf::TierB::gemv));
            const auto chunk = run(true);
            const bool y = ref.y0 == chunk.y0;
            const bool m = ref.mtp.id == chunk.mtp.id && ref.mtp.flagged == chunk.mtp.flagged &&
                           std::memcmp(&ref.mtp.confidence, &chunk.mtp.confidence, sizeof(float)) == 0;
            const bool h = ref.hp == chunk.hp, s = ref.state == chunk.state, t = ref.tokens == chunk.tokens;
            const bool ok = y && m && h && s && t;
            std::printf("rank %u %s mtp-prefill chunked == bootstrap (L=%zu C=%u N=%u y0=%u mtp=%u hp=%u state=%u tokens=%u state_bytes=%zu ref_ms=%.3f chunked_ms=%.3f ref_decode_ms=%.3f chunked_decode_ms=%.3f)\n",
                        opt.rank, ok ? "OK" : "FAIL", toks.size(), opt.max_rows, N, y, m, h, s, t, ref.state.size(),
                        ref.bootstrap_ms, chunk.bootstrap_ms, ref.decode_ms, chunk.decode_ms);
            if (!ok) throw std::runtime_error("MTP prefill bitwise gate failed");
            return 0;
        }
        if (spec_run > 0) {
            namespace pf = strix::glm::pf;
            namespace sp = strix::glm::spec;
            if (!opt.spec || !opt.mtp || decode_n < 2 || prompt.empty())
                throw std::invalid_argument("--spec-run needs --spec 1 --mtp 1 --decode N (N >= 2) and nonempty prompts");
            if (mtp_prefill == "ref" && (prefill_mode != "ref" || pf_before_load))
                throw std::invalid_argument("--spec-run with --mtp-prefill ref requires --prefill ref");
            if (mtp_prefill == "chunked" && prefill_mode == "wmma")
                throw std::invalid_argument("--mtp-prefill chunked requires the exact gemv identity");
            std::shared_ptr<ge::PrefillPath> mtp_path;
            if (mtp_prefill == "chunked") {
                eng.set_prefill(ge::reference_prefill());
                const unsigned heads = 64 / opt.world;
                mtp_path = pf::make_prefill_path(heads, heads, opt.capacity, opt.max_rows, pf::TierB::gemv);
                eng.set_prefill(mtp_path);  // admit/warm once before the timed bootstrap
                eng.set_prefill(ge::reference_prefill());
            }
            const auto identity = eng.identity_mode();
            if (identity != ge::IdentityMode::kR1b && identity != ge::IdentityMode::kR1)
                throw std::invalid_argument("--spec-run requires actual identity r1b or r1");
            if (opt.world == 2 && !eng.decode_xfer_enabled())
                throw std::invalid_argument("--spec-run requires automatic decode Xfer routing (TP2)");
            std::string raw_ks;
            for (int i = 2; i + 1 < argc; ++i)
                if (std::string(argv[i]) == "--spec-run") raw_ks = argv[i + 1];
            std::vector<unsigned> ks;
            for (std::size_t a = 0; a <= raw_ks.size();) {
                const auto b = std::min(raw_ks.find(',', a), raw_ks.size());
                // Each element is exactly one decimal digit: reject signs, whitespace and suffixes.
                if (b != a + 1 || (raw_ks[a] != 'a' && (raw_ks[a] < '1' || raw_ks[a] > '7')))
                    throw std::invalid_argument("--spec-run expects a comma list of integers 1..7 or a");
                ks.push_back(raw_ks[a] == 'a' ? 0U : static_cast<unsigned>(raw_ks[a] - '0'));
                a = b + 1;
            }
            struct Prompt {
                std::string name, category;
                std::vector<std::uint32_t> tokens;
            };
            std::vector<Prompt> prompts;
            for (std::size_t a = 0; a <= prompt.size();) {
                const auto b = std::min(prompt.find(',', a), prompt.size());
                const auto path = prompt.substr(a, b - a);
                if (path.empty()) throw std::invalid_argument("--spec-run prompt paths must be nonempty");
                Prompt pr;
                pr.name = path.substr(path.find_last_of("/\\") + 1);
                if (pr.name.size() >= 4 && pr.name.substr(pr.name.size() - 4) == ".u32") pr.name.resize(pr.name.size() - 4);
                pr.category = pr.name;
                if (pr.name.size() > 4 && pr.name[0] == 'p' && pr.name[1] >= '0' && pr.name[1] <= '9' &&
                    pr.name[2] >= '0' && pr.name[2] <= '9' && pr.name[3] == '-') {
                    const auto end = pr.name.find('-', 4);
                    pr.category = pr.name.substr(4, end == std::string::npos ? end : end - 4);
                }
                pr.tokens = read_u32(path);
                if (pr.tokens.empty()) throw std::invalid_argument("--spec-run prompt is empty: " + path);
                for (auto t : pr.tokens)
                    if (t >= std::uint64_t{eng.vocab_slice()} * opt.world)
                        throw std::invalid_argument("--spec-run prompt token outside vocabulary: " + path);
                if (std::uint64_t{pr.tokens.size()} + decode_n + 8 > opt.capacity)
                    throw std::invalid_argument("--spec-run requires L + decode_n + 8 <= capacity: " + path);
                prompts.push_back(std::move(pr));
                a = b + 1;
            }
            auto ms = [](Clock::time_point a, Clock::time_point b) { return 1000.0 * secs(a, b); };
            auto equal = [&](const Prompt& pr, unsigned k, const auto& expected, const auto& actual) {
                for (std::size_t j = 0; j < std::max(expected.size(), actual.size()); ++j) {
                    const auto e = j < expected.size() ? expected[j] : sp::kNoFlag;
                    const auto v = j < actual.size() ? actual[j] : sp::kNoFlag;
                    if (e != v) {
                        std::printf("SPV1_FAIL rank=%u prompt=%s K=%u index=%zu expected=%u actual=%u\n",
                                    opt.rank, pr.name.c_str(), k, j, e, v);
                        std::fflush(stdout);
                        throw std::runtime_error("SPV1 token equality failed");
                    }
                }
            };
            using Sample = sp::TimingSample;  // 1 verify, 2 extension, 3 catch-up
            struct Pool { std::string category; std::vector<Sample> samples; };
            std::vector<Pool> pools;
            auto curve = [&](const char* name, unsigned k, const char* scope, const std::vector<Sample>& samples) {
                for (unsigned kind = 1; kind <= 3; ++kind)
                    for (unsigned r = 1; r <= 8; ++r)
                        for (unsigned replay = 0; replay <= 8; ++replay) {
                            std::vector<double> cell;
                            double sum = 0;
                            for (const auto& s : samples)
                                if (s.kind == kind && s.r == r && s.replay == replay) {
                                    cell.push_back(s.elapsed);
                                    sum += s.elapsed;
                                }
                            if (cell.empty()) continue;
                            std::sort(cell.begin(), cell.end());
                            const auto n = cell.size();
                            const double median = (cell[(n - 1) / 2] + cell[n / 2]) / 2;
                            std::printf("SPV1_VCURVE rank=%u prompt=%s K=%u scope=%s kind=%s R=%u replay=%u samples=%zu mean_ms=%.6f median_ms=%.6f\n",
                                        opt.rank, name, k, scope, kind == 1 ? "verify" : kind == 2 ? "extend" : "catchup",
                                        r, replay, n, sum / static_cast<double>(n), median);
                        }
            };
            std::printf("SPV1_BEGIN rank=%u identity=%s prompt_xch=xfer plain_xch=xfer spec_xch=comm N=%u Ks=%s\n",
                        opt.rank, ge::identity_mode_name(identity), decode_n, raw_ks.c_str());
            std::fflush(stdout);
            // Untimed warm-up. Verify supplies R initialized trunk hidden rows for each MTP call.
            eng.reset(0);
            const auto& warm = prompts.front().tokens;
            auto token = eng.prefill(0, warm, 0);
            for (unsigned j = 0; j < 3; ++j) {
                const ge::Row row{0, token, warm.size() + j};
                token = eng.step({&row, 1})[0];
            }
            eng.reset(0);
            for (unsigned r = 1; r <= 4; ++r) {
                std::vector<std::uint32_t> rows(r, warm.front());
                ge::Exchange::MergeCtl cv;
                cv.cycle = r;
                cv.kind = 1;
                const auto v = eng.verify(0, rows, 0, cv);
                if (v.replay_flag || std::any_of(v.flag.begin(), v.flag.end(), [](auto f) { return f != 0; }))
                    throw std::runtime_error("--spec-run warm-up verify flagged");
                ge::Exchange::MergeCtl cm;
                cm.cycle = r;
                cm.kind = 2;
                (void)eng.mtp(0, eng.last_hidden(), rows, 0, cm);
                eng.reset(0);
            }
            unsigned arms = 0;
            for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
                const auto& pr = prompts[pi];
                const auto& toks = pr.tokens;
                const std::uint64_t L = toks.size();
                auto plain_cost = sp::seeded_cost_table();
                auto plain = [&](const char* pass) {
                    eng.reset(0);
                    eng.set_identity_mode(identity);
                    std::vector<std::uint32_t> out;
                    out.reserve(decode_n);
                    const auto t0 = Clock::now();
                    out.push_back(eng.prefill(0, toks, 0));
                    const auto t1 = Clock::now();
                    for (unsigned j = 1; j < decode_n; ++j) {
                        const ge::Row row{0, out.back(), L + j - 1};
                        const auto ts = Clock::now();
                        out.push_back(eng.step({&row, 1})[0]);
                        if (opt.rank == 0) sp::observe_cost(plain_cost, 0, ms(ts, Clock::now()));
                    }
                    eng.flush(0);
                    const auto t2 = Clock::now();
                    std::printf("SPV1_PLAIN rank=%u prompt=%s pass=%s gen=%u timed_tokens=%u prefill_ms=%.6f decode_ms=%.6f ms_token=%.6f hash=%016llx\n",
                                opt.rank, pr.name.c_str(), pass, decode_n, decode_n - 1, ms(t0, t1), ms(t1, t2),
                                ms(t1, t2) / (decode_n - 1), static_cast<unsigned long long>(fnv(out.data(), out.size() * 4)));
                    return out;
                };
                const auto T = plain("A0");
                for (std::size_t ki = 0; ki < ks.size(); ++ki) {
                    const unsigned selected = ks[(ki + pi % 3) % ks.size()];
                    const bool adaptive = selected == 0;
                    const unsigned kmax = adaptive ? 3U : selected;
                    eng.reset(0);
                    eng.set_identity_mode(identity);
                    sp::ControlState control;
                    control.rank = opt.rank;
                    control.publisher = plain_cost;
                    std::vector<sp::CycleRecord> records;
                    std::vector<Sample> samples;
                    control.on_cycle = [&](const sp::CycleRecord& rec) { records.push_back(rec); };
                    control.on_sample = [&](const Sample& sample) { samples.push_back(sample); };
                    const auto tb0 = Clock::now();
                    if (mtp_path) eng.set_prefill(mtp_path);
                    const auto seed = sp::bootstrap(eng, 0, toks, mtp_path ? sp::BootstrapPolicy::Chunked :
                                                                         sp::BootstrapPolicy::Reference, control);
                    if (mtp_path) eng.set_prefill(ge::reference_prefill());
                    equal(pr, kmax, std::vector<std::uint32_t>{T[0]}, std::vector<std::uint32_t>{seed.y0});
                    const auto tb1 = Clock::now();
                    const auto result = sp::decode_greedy(eng, 0, seed, {kmax, adaptive, kmax, opt.capacity}, decode_n, control);
                    const auto& out = result.tokens;
                    const auto& depths = result.depths;
                    const auto cycle = result.cycles, offered = result.offered, accepted = result.accepted;
                    equal(pr, kmax, T, out);
                    const double bootstrap_ms = ms(tb0, tb1), loop_ms = result.loop_ms, tail_ms = result.tail_ms;
                    for (std::size_t ci = 0; ci < records.size(); ++ci) {
                        const auto& r = records[ci];
                        std::printf("SPV1_CYCLE rank=%u prompt=%s Kmax=%u cycle=%zu p=%llu K=%u replay=%u accepted=%u emitted=%u rows=%u extend_ms=%.6f verify_ms=%.6f commit_ms=%.6f catchup_ms=%.6f other_ms=%.6f\n",
                                    opt.rank, pr.name.c_str(), kmax, ci, static_cast<unsigned long long>(r.p), r.k, r.replay,
                                    r.accepted, r.emitted, r.rows, r.extend, r.verify, r.commit, r.catchup, r.other);
                        if (adaptive)
                            std::printf("SPV1_DECISION rank=%u cycle=%zu K=%u table_version=%u checksum=%08x\n",
                                        opt.rank, ci, r.k, r.table_version, r.checksum);
                    }
                    for (unsigned d = 1; d <= kmax; ++d) {
                        const auto& x = depths[d];
                        const std::string cond = x.reached ? std::to_string(double(x.accepted) / x.reached) : "NA";
                        const std::string off = x.offered ? std::to_string(double(x.accepted) / x.offered) : "NA";
                        std::printf("SPV1_DEPTH rank=%u prompt=%s Kmax=%u d=%u offered=%u reached=%u accepted=%u a_cond=%s a_offered=%s\n",
                                    opt.rank, pr.name.c_str(), kmax, d, x.offered, x.reached, x.accepted, cond.c_str(), off.c_str());
                    }
                    std::printf("SPV1_RESULT rank=%u prompt=%s K=%u gen=%u cycles=%u offered=%u accepted=%u tok_cycle=%.6f bootstrap_ms=%.6f loop_ms=%.6f tail_ms=%.6f decode_ms=%.6f ms_token=%.6f request_ms=%.6f equal=1 hash=%016llx\n",
                                opt.rank, pr.name.c_str(), kmax, decode_n, cycle, offered, accepted, double(decode_n - 1) / cycle,
                                bootstrap_ms, loop_ms, tail_ms, loop_ms + tail_ms, (loop_ms + tail_ms) / (decode_n - 1),
                                bootstrap_ms + loop_ms + tail_ms, static_cast<unsigned long long>(fnv(out.data(), out.size() * 4)));
                    curve(pr.name.c_str(), kmax, "arm", samples);
                    auto pool = std::find_if(pools.begin(), pools.end(), [&](const Pool& x) { return x.category == pr.category; });
                    if (pool == pools.end()) {
                        pools.push_back({pr.category, {}});
                        pool = pools.end() - 1;
                    }
                    pool->samples.insert(pool->samples.end(), samples.begin(), samples.end());
                    ++arms;
                    std::fflush(stdout);
                }
                const auto A1 = plain("A1");
                equal(pr, 0, T, A1);
                std::fflush(stdout);
            }
            for (const auto& pool : pools) curve(pool.category.c_str(), 0, "category", pool.samples);
            eng.reset(0);
            eng.set_identity_mode(identity);
            std::printf("SPV1_DONE rank=%u status=PASS prompts=%zu arms=%u\n", opt.rank, prompts.size(), arms);
            std::printf("rank %u SP1 PASS\n", opt.rank);
            std::fflush(stdout);
        }
        // decode mode: --prompt A[,B,...] runs each prompt in turn (slot 0 reset per prompt); --prompt-max N keeps the
        // first N tokens of each; --logits-out PREFIX writes the rank's logits slice after each prefill
        // (PREFIX-p<i>-r<rank>.f32, the first generated token's distribution) for identity comparisons.
        namespace sp = strix::glm::spec;
        sp::ControlState decode_control;
        decode_control.rank = opt.rank;
        unsigned pi = 0;
        for (const auto& toks : decode_prompts) {
            eng.reset(0);
            std::size_t free0 = 0, free1 = 0, total = 0;
            if (hipMemGetInfo(&free0, &total) != hipSuccess) throw std::runtime_error("mem info");
            const auto t0 = Clock::now();
            auto& control = decode_control;
            sp::Seed seed;
            std::uint32_t next = 0;
            if (mtp_spec_on) {
                seed = sp::bootstrap(eng, 0, toks, prefill_mode == "gemv" ? sp::BootstrapPolicy::Chunked :
                                                                         sp::BootstrapPolicy::Reference, control);
                next = seed.y0;
            } else next = eng.prefill(0, toks, 0);
            const auto t1 = Clock::now();
            if (hipMemGetInfo(&free1, &total) != hipSuccess) throw std::runtime_error("mem info");
            std::printf("rank %u pf dev free delta %+.1f MiB (%zu rows)\n", opt.rank,
                        (static_cast<double>(free1) - static_cast<double>(free0)) / double(1ULL << 20), toks.size());
            {  // prefill digest (compare ref / gemv / wmma runs): this rank's last-row logits slice + state up to the prompt end
                std::vector<float> lg(eng.vocab_slice());
                if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) throw std::runtime_error("logits copy");
                const auto st = eng.state_bytes_upto(0, toks.size());
                std::printf("rank %u prefill %s: %zu tokens %.3f s (%.3f ms/token), first token %u, logits fnv %016llx, state fnv %016llx (%zu B)\n",
                            opt.rank, prefill_mode.c_str(), toks.size(), secs(t0, t1), 1000.0 * secs(t0, t1) / static_cast<double>(toks.size()), next,
                            static_cast<unsigned long long>(fnv(lg.data(), lg.size() * 4)), static_cast<unsigned long long>(fnv(st.data(), st.size())),
                            st.size());
            }
            if (!logits_out.empty()) {
                std::vector<float> lg(eng.vocab_slice());
                if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
                    throw std::runtime_error("logits copy");
                const std::string path = logits_out + "-p" + std::to_string(pi) + "-r" + std::to_string(opt.rank) + ".f32";
                std::ofstream f(path, std::ios::binary);
                f.write(reinterpret_cast<const char*>(lg.data()), static_cast<std::streamsize>(lg.size() * 4));
                if (!f) throw std::runtime_error("write " + path);
                std::printf("rank %u prompt %u logits fnv %016llx -> %s\n", opt.rank, pi,
                            static_cast<unsigned long long>(fnv(lg.data(), lg.size() * 4)), path.c_str());
            }
            ++pi;
            const auto t1d = Clock::now();  // decode timing starts after the digests / logits dump
            std::vector<std::uint32_t> out{next};
            std::uint64_t pos = toks.size();
            const bool decode_profile = profile && !mtp_spec_on;
            if (decode_profile) eng.set_profiling(true);  // decode steps only (after the prefill)
            const ge::Profile p0 = eng.profile();
            sp::RunResult result;
            if (mtp_spec_on) {
                result = sp::decode_greedy(eng, 0, seed, {eff.kmax, eff.adaptive, eff.kmax, opt.capacity}, decode_n, control);
                out = result.tokens;
            } else for (unsigned i = 1; i < decode_n; ++i) {
                const ge::Row r{0, next, pos++};
                next = eng.step({&r, 1})[0];
                out.push_back(next);
            }
            const auto t2 = Clock::now();
            if (decode_profile) {
                eng.set_profiling(false);
                const ge::Profile& p = eng.profile();
                const double n = static_cast<double>(p.calls - p0.calls);
                auto d = [&](double a, double b) { return (a - b) / n; };
                std::printf("rank %u profile (ms/token, device events, %llu decode steps): embed %.3f hc %.3f kda %.3f dsa %.3f "
                            "exchange %.3f moe %.3f dense %.3f head %.3f | sum %.3f\n",
                            opt.rank, static_cast<unsigned long long>(p.calls - p0.calls), d(p.embed, p0.embed), d(p.hc, p0.hc),
                            d(p.kda, p0.kda), d(p.dsa, p0.dsa), d(p.exchange, p0.exchange), d(p.moe, p0.moe), d(p.dense, p0.dense),
                            d(p.head, p0.head),
                            d(p.embed + p.hc + p.kda + p.dsa + p.exchange + p.moe + p.dense + p.head,
                              p0.embed + p0.hc + p0.kda + p0.dsa + p0.exchange + p0.moe + p0.dense + p0.head));
            }
            const double ms = decode_n > 1 ? 1000.0 * secs(t1d, t2) / (decode_n - 1) : 0.0;
            std::printf("rank %u decode: prompt %zu tokens prefill %.2f s, %u tokens, %.2f ms/token (%.2f tok/s), tokens fnv %016llx\n",
                        opt.rank, toks.size(), secs(t0, t1), decode_n, ms, ms > 0 ? 1000.0 / ms : 0.0,
                        static_cast<unsigned long long>(fnv(out.data(), out.size() * 4)));
            if (mtp_spec_on) {
                std::printf("rank %u mtp-spec: cycles %u offered %u accepted %u (d1 %u/%u, d2 %u/%u, d3 %u/%u) K hist k1 %u k2 %u k3 %u tok/cycle %.6f\n",
                            opt.rank, result.cycles, result.offered, result.accepted,
                            result.depths[1].accepted, result.depths[1].offered,
                            result.depths[2].accepted, result.depths[2].offered,
                            result.depths[3].accepted, result.depths[3].offered,
                            result.histogram[1], result.histogram[2], result.histogram[3],
                            result.cycles ? double(decode_n - 1) / result.cycles : 0.0);
            }
            std::printf("rank %u tokens:", opt.rank);
            for (auto t : out) std::printf(" %u", t);
            std::printf("\n");
        }
        // --aba-attn3 N: in-process A B A B A B A of attention defaults, A = legacy (Tier B table off, MLA scores lane),
        // B = new defaults (table on, scores auto; KDA lanes auto = q2 at one row, unchanged at R = 1), first --prompt,
        // reset + reference prefill + N greedy tokens per run: ms/token, tokens fnv, last logits fnv, state fnv (must be
        // equal across all 7 runs: the switches select gated bit-identical forms).
        if (aba3_n > 0) {
            namespace TB = strix::glm::attn_tierb;
            namespace FA = strix::glm::fast;
            const std::size_t pe = std::min(prompt.find(','), prompt.size());
            const auto toks = read_u32(prompt.substr(0, pe));
            for (int k = 0; k < 7; ++k) {
                const bool b = k % 2 == 1;
                TB::set_table_enabled(b);
                if (b) FA::set_scores_auto();
                else FA::set_scores_schedule(FA::DenseSchedule::kLane);
                eng.reset(0);
                std::uint32_t nx = eng.prefill(0, toks, 0);
                std::vector<std::uint32_t> out{nx};
                std::uint64_t pos = toks.size();
                const auto t0 = Clock::now();
                for (unsigned i = 1; i < aba3_n; ++i) {
                    const ge::Row r{0, nx, pos++};
                    nx = eng.step({&r, 1})[0];
                    out.push_back(nx);
                }
                const double ms = 1000.0 * secs(t0, Clock::now()) / (aba3_n - 1);
                std::vector<float> lg(eng.vocab_slice());
                if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
                    throw std::runtime_error("logits copy");
                const auto st = eng.state_bytes_upto(0, pos);
                std::printf("aba3 rank %u run %d %s: %.3f ms/token tokens fnv %016llx logits fnv %016llx state fnv %016llx\n", opt.rank, k,
                            b ? "new   " : "legacy", ms, static_cast<unsigned long long>(fnv(out.data(), out.size() * 4)),
                            static_cast<unsigned long long>(fnv(lg.data(), lg.size() * 4)),
                            static_cast<unsigned long long>(fnv(st.data(), st.size())));
                std::fflush(stdout);
            }
            TB::set_table_enabled(true);
            FA::set_scores_auto();
        }
        // --aba-xch N: in-process A B A B A B A of the decode exchange, A = CommExchange (the Communicator), B =
        // XferExchange borrowing the communicator's transport (one load, same session). Per run: reset slots 0..R-1 and
        // prefill each with the first --prompt over A (prefill is not compared), then N-1 greedy steps of R rows
        // (--aba-xch-rows R1,R2,..: one A/B/A series per R, default 1) over the run's exchange: ms/step, tokens fnv, last logits fnv, state fnv of slot 0
        // (all must be equal across the 7 runs: the two exchanges are gated bit-identical).
        if (aba_xch_n > 0) {
            if (comm == nullptr) throw std::invalid_argument("--aba-xch needs --world 2 --xch comm");
            const std::size_t pe = std::min(prompt.find(','), prompt.size());
            const auto toks = read_u32(prompt.substr(0, pe));
            const auto xa = eng.exchange();
            auto xb = std::make_shared<strix::glm::xfer::XferExchange>(comm->transport(), opt.rank, 8U);
            std::vector<unsigned> rlist;
            for (std::size_t q = 0; q <= aba_xch_rows.size();) {
                const std::size_t e = std::min(aba_xch_rows.find(',', q), aba_xch_rows.size());
                rlist.push_back(static_cast<unsigned>(std::stoul(aba_xch_rows.substr(q, e - q))));
                if (rlist.back() == 0 || rlist.back() > 8 || rlist.back() > opt.slots) throw std::invalid_argument("--aba-xch-rows: each R must be 1..min(8, slots)");
                q = e + 1;
            }
            for (const unsigned aba_rows : rlist)
            for (int k = 0; k < 7; ++k) {
                const bool b = k % 2 == 1;
                eng.set_exchange(xa);
                std::vector<std::uint32_t> nx(aba_rows);
                for (unsigned j = 0; j < aba_rows; ++j) {
                    eng.reset(j);
                    nx[j] = eng.prefill(j, toks, 0);
                }
                if (b) eng.set_exchange(xb);
                std::vector<std::uint32_t> out(nx);
                std::uint64_t pos = toks.size();
                std::vector<ge::Row> rows(aba_rows);
                const auto t0 = Clock::now();
                for (unsigned i = 1; i < aba_xch_n; ++i, ++pos) {
                    for (unsigned j = 0; j < aba_rows; ++j) rows[j] = ge::Row{j, nx[j], pos};
                    const auto o = eng.step(rows);
                    for (unsigned j = 0; j < aba_rows; ++j) nx[j] = o[j];
                    out.insert(out.end(), nx.begin(), nx.end());
                }
                const double ms = 1000.0 * secs(t0, Clock::now()) / (aba_xch_n - 1);
                if (b) xb->complete();
                std::vector<float> lg(eng.vocab_slice());
                if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
                    throw std::runtime_error("logits copy");
                const auto st = eng.state_bytes_upto(0, pos);
                std::printf("abax rank %u run %d %s R %u: %.3f ms/step tokens fnv %016llx logits fnv %016llx state fnv %016llx\n", opt.rank, k,
                            b ? "xfer" : "comm", aba_rows, ms, static_cast<unsigned long long>(fnv(out.data(), out.size() * 4)),
                            static_cast<unsigned long long>(fnv(lg.data(), lg.size() * 4)),
                            static_cast<unsigned long long>(fnv(st.data(), st.size())));
                std::fflush(stdout);
            }
            eng.set_exchange(xa);
        }
        if (idcheck_n > 0) {
            if (prompt.empty()) throw std::invalid_argument("--idcheck needs --prompt");
            const auto toks = read_u32(prompt);
            if (toks.empty()) throw std::invalid_argument("--idcheck needs a nonempty prompt");
            struct Digest {
                std::uint64_t tokens, logits, state;
            };
            auto run = [&](bool independent) {
                eng.set_identity_mode(independent ? ge::IdentityMode::kR1 : ge::IdentityMode::kR1b);
                eng.reset(0);
                std::uint32_t nx = eng.prefill(0, toks, 0);
                if (independent) eng.set_identity_mode(ge::IdentityMode::kR1bAll);
                std::vector<std::uint32_t> out{nx};
                std::uint64_t pos = toks.size();
                for (unsigned i = 1; i < idcheck_n; ++i) {
                    const ge::Row row{0, nx, pos++};
                    nx = eng.step({&row, 1})[0];
                    out.push_back(nx);
                }
                std::vector<float> lg(eng.vocab_slice());
                if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
                    throw std::runtime_error("logits copy");
                const auto state = eng.state_bytes_upto(0, pos);
                return Digest{fnv(out.data(), out.size() * 4), fnv(lg.data(), lg.size() * 4), fnv(state.data(), state.size())};
            };
            const auto a = run(false), b = run(true);
            eng.set_identity_mode(load_identity);
            const bool equal = a.tokens == b.tokens && a.logits == b.logits && a.state == b.state;
            std::printf("idcheck rank %u: a tokens fnv %016llx logits fnv %016llx state fnv %016llx | "
                        "b tokens fnv %016llx logits fnv %016llx state fnv %016llx -> %s\n", opt.rank,
                        static_cast<unsigned long long>(a.tokens), static_cast<unsigned long long>(a.logits),
                        static_cast<unsigned long long>(a.state), static_cast<unsigned long long>(b.tokens),
                        static_cast<unsigned long long>(b.logits), static_cast<unsigned long long>(b.state), equal ? "EQUAL" : "DIFF");
            if (!equal) return 1;
        }
        // --aba N: in-process timing of the two identities on the first --prompt, A B A B A B A (A = R1, B = R1b-all),
        // each = reset + reference prefill + N greedy tokens; prints ms/token per run and the tokens fnv.
        if (aba_n > 0) {
            const std::size_t pe = std::min(prompt.find(','), prompt.size());
            const auto toks = read_u32(prompt.substr(0, pe));
            for (int k = 0; k < 7; ++k) {
                const bool b = k % 2 == 1;
                eng.set_identity_mode(b ? ge::IdentityMode::kR1bAll : ge::IdentityMode::kR1);
                eng.reset(0);
                std::uint32_t nx = eng.prefill(0, toks, 0);
                std::vector<std::uint32_t> out{nx};
                std::uint64_t pos = toks.size();
                const auto t0 = Clock::now();
                for (unsigned i = 1; i < aba_n; ++i) {
                    const ge::Row r{0, nx, pos++};
                    nx = eng.step({&r, 1})[0];
                    out.push_back(nx);
                }
                const double ms = 1000.0 * secs(t0, Clock::now()) / (aba_n - 1);
                std::printf("aba rank %u run %d %s: %.3f ms/token tokens fnv %016llx\n", opt.rank, k, b ? "R1b" : "R1 ", ms,
                            static_cast<unsigned long long>(fnv(out.data(), out.size() * 4)));
                std::fflush(stdout);
            }
            eng.set_identity_mode(load_identity);
        }
        // --ab N (identity comparison, R = 1): per prompt of --prompt A,B,.. (each cut to --prompt-max): (A) R1 greedy N
        // tokens, keeping every position's logits slice + router ids; (B) R1b teacher-forced on A's tokens: per position
        // the softmax partials of both identities on this rank's vocab slice (max, sum exp, sum exp_A * (lA - lB)), top-5
        // of each, and the layers whose top-8 route differs; (C) R1b greedy N tokens (first divergence vs A).
        // Lines 'ab ...' are combined across ranks by tools/glm/omath/ab_stats.py. Both ranks run the same sequence.
        if (ab_n > 0) {
            const std::uint32_t V = eng.vocab_slice();
            eng.set_route_log(true);
            auto logits_host = [&](std::vector<float>& out) {
                out.resize(V);
                if (hipMemcpy(out.data(), eng.last_logits(), static_cast<std::size_t>(V) * 4, hipMemcpyDeviceToHost) != hipSuccess)
                    throw std::runtime_error("logits copy");
            };
            auto top5 = [&](const std::vector<float>& l) {
                std::vector<std::uint32_t> idx(V);
                for (std::uint32_t i = 0; i < V; ++i) idx[i] = i;
                std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](std::uint32_t a, std::uint32_t b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
                std::string s;
                for (int k = 0; k < 5; ++k) s += " " + std::to_string(idx[static_cast<std::size_t>(k)] + opt.rank * V) + ":" + std::to_string(l[idx[static_cast<std::size_t>(k)]]);
                return s;
            };
            std::size_t pa2 = 0;
            unsigned pidx = 0;
            const auto tab = Clock::now();
            while (pa2 <= prompt.size()) {
                const std::size_t pb2 = std::min(prompt.find(',', pa2), prompt.size());
                auto toks = read_u32(prompt.substr(pa2, pb2 - pa2));
                pa2 = pb2 + 1;
                if (prompt_max > 0 && toks.size() > prompt_max) toks.resize(prompt_max);
                if (toks.size() + 2 * ab_n + 2 > opt.capacity) throw std::invalid_argument("--ab: prompt + N exceeds --capacity");
                // (A) R1 greedy
                eng.set_identity_mode(ge::IdentityMode::kR1);
                eng.reset(0);
                std::vector<std::vector<float>> la(ab_n);
                std::vector<std::vector<std::uint32_t>> ra(ab_n);
                std::vector<std::uint32_t> ga;
                std::uint32_t nx = eng.prefill(0, toks, 0);
                logits_host(la[0]);
                ra[0] = eng.route_log();
                ga.push_back(nx);
                std::uint64_t pos = toks.size();
                for (unsigned i = 1; i < ab_n; ++i) {
                    const ge::Row r{0, nx, pos++};
                    nx = eng.step({&r, 1})[0];
                    logits_host(la[i]);
                    ra[i] = eng.route_log();
                    ga.push_back(nx);
                }
                // (B) R1b teacher-forced on A's tokens
                eng.set_identity_mode(ge::IdentityMode::kR1bAll);
                eng.reset(0);
                std::vector<float> lb;
                unsigned tf_top1 = 0;
                for (unsigned i = 0; i < ab_n; ++i) {
                    std::uint32_t own = 0;
                    if (i == 0) {
                        own = eng.prefill(0, toks, 0);
                    } else {
                        const ge::Row r{0, ga[i - 1], toks.size() + i - 1};
                        own = eng.step({&r, 1})[0];
                    }
                    logits_host(lb);
                    if (i == 0 && !out_path.empty()) {  // first-token logits of both identities (float64 O-math comparison)
                        for (int w = 0; w < 2; ++w) {
                            const std::vector<float>& v = w == 0 ? la[0] : lb;
                            const std::string pth = out_path + (w == 0 ? "-A" : "-B") + "-p" + std::to_string(pidx) + "-r" + std::to_string(opt.rank) + ".f32";
                            std::ofstream f(pth, std::ios::binary);
                            f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
                        }
                    }
                    const auto rb = eng.route_log();
                    tf_top1 += own == ga[i] ? 1U : 0U;
                    const std::vector<float>& a = la[i];
                    float ma = a[0], mb = lb[0];
                    for (std::uint32_t k = 1; k < V; ++k) {
                        ma = std::max(ma, a[k]);
                        mb = std::max(mb, lb[k]);
                    }
                    double sa = 0, sb = 0, t = 0;
                    for (std::uint32_t k = 0; k < V; ++k) {
                        const double ea = std::exp(double(a[k]) - ma);
                        sa += ea;
                        sb += std::exp(double(lb[k]) - mb);
                        t += ea * (double(a[k]) - double(lb[k]));
                    }
                    unsigned flips = 0, moe = 0;
                    std::string fl;
                    for (unsigned L = 0; L < strix::glm::kLayers; ++L) {
                        const std::uint32_t* x = ra[i].data() + static_cast<std::size_t>(L) * 8;
                        const std::uint32_t* y = rb.data() + static_cast<std::size_t>(L) * 8;
                        if (x[0] == 0xFFFFFFFFU) continue;
                        ++moe;
                        if (!std::equal(x, x + 8, y)) {
                            ++flips;
                            fl += (fl.empty() ? "" : ",") + std::to_string(L);
                        }
                    }
                    std::printf("ab rank %u p %u i %u tokA %u tokB %u ma %.9g sa %.17g mb %.9g sb %.17g t %.17g flips %u/%u [%s] topA%s topB%s\n",
                                opt.rank, pidx, i, ga[i], own, double(ma), sa, double(mb), sb, t, flips, moe, fl.c_str(), top5(a).c_str(),
                                top5(lb).c_str());
                }
                // (C) R1b greedy
                eng.reset(0);
                std::vector<std::uint32_t> gb{eng.prefill(0, toks, 0)};
                pos = toks.size();
                for (unsigned i = 1; i < ab_n; ++i) {
                    const ge::Row r{0, gb.back(), pos++};
                    gb.push_back(eng.step({&r, 1})[0]);
                }
                unsigned first = ab_n, same = 0;
                for (unsigned i = 0; i < ab_n; ++i) {
                    if (ga[i] == gb[i]) ++same;
                    else if (first == ab_n) first = i;
                }
                std::printf("ab-summary rank %u p %u prompt %zu tokens, N %u: TF top-1 (R1b argmax == R1 token) %u/%u, free-run equal %u/%u, "
                            "first divergence %s; fnv A %016llx B %016llx; %.1f s so far\n",
                            opt.rank, pidx, toks.size(), ab_n, tf_top1, ab_n, same, ab_n, first == ab_n ? "none" : std::to_string(first).c_str(),
                            static_cast<unsigned long long>(fnv(ga.data(), ga.size() * 4)), static_cast<unsigned long long>(fnv(gb.data(), gb.size() * 4)),
                            secs(tab, Clock::now()));
                std::fflush(stdout);
                ++pidx;
            }
            eng.set_identity_mode(load_identity);
        }
        if (batch_r > 0) {
            std::vector<ge::Row> rows(batch_r);
            std::vector<std::uint32_t> all;  // every step's tokens, row-major per step
            for (unsigned r = 0; r < batch_r; ++r) {
                eng.reset(r);
                rows[r] = ge::Row{r, 1000 + r, 0};
            }
            const auto t0 = Clock::now();
            for (unsigned s = 0; s < batch_steps; ++s) {
                const auto nx = eng.step(rows);
                all.insert(all.end(), nx.begin(), nx.end());
                for (unsigned r = 0; r < batch_r; ++r) rows[r] = ge::Row{r, nx[r], rows[r].pos + 1};
            }
            const double dt = secs(t0, Clock::now());
            std::vector<std::uint32_t> row0;  // slot 0's sequence (batch invariance: equal for every R)
            for (std::size_t i = 0; i < all.size(); i += batch_r) row0.push_back(all[i]);
            std::printf("rank %u batch: R %u steps %u %.2f ms/step %.2f tok/s aggregate, tokens fnv %016llx, row0 fnv %016llx\n",
                        opt.rank, batch_r, batch_steps, 1000.0 * dt / batch_steps, batch_r * batch_steps / dt,
                        static_cast<unsigned long long>(fnv(all.data(), all.size() * 4)),
                        static_cast<unsigned long long>(fnv(row0.data(), row0.size() * 4)));
        }
        if (!tf_tokens.empty()) {
            const auto toks = read_u32(tf_tokens);
            const std::uint64_t T = toks.size() - 1;  // inputs 0..T-1; token T is only scored
            if (!opt.spec || out_path.empty() || tf_score == 0 || toks.size() < 2 || tf_score > T || T > opt.capacity)
                throw std::invalid_argument("--tf needs --spec 1 --score N --out PREFIX, N < tokens, tokens - 1 <= capacity");
            const std::uint32_t V = eng.vocab_slice();
            std::vector<float> keep(static_cast<std::size_t>(tf_score) * V), lg;
            if (tf_stop != 0 && (state_out.empty() || tf_stop >= T || tf_check))
                throw std::invalid_argument("--tf-stop P needs --state-out, P < T, no --tf-check");
            std::uint64_t p0 = 0;
            if (!state_in.empty()) {  // file: u64 frontier, u64 T, then state_bytes_upto
                std::ifstream f(state_in, std::ios::binary);
                std::uint64_t hdr[2] = {};
                f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
                std::vector<std::uint8_t> st((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                if (!f.eof() || hdr[1] != T || hdr[0] == 0 || hdr[0] >= T) throw std::runtime_error("--state-in: bad file " + state_in);
                eng.restore_state(0, st, hdr[0]);
                p0 = hdr[0];
                if (tf_check) throw std::invalid_argument("--state-in with --tf-check: run the check uninterrupted");
            } else {
                eng.reset(0);
            }
            const std::uint64_t p_end = tf_stop != 0 ? tf_stop : T;
            if (p_end > T - tf_score && p_end != T) throw std::invalid_argument("--tf-stop must be before the scored rows");
            ge::Exchange::MergeCtl ctl;
            ctl.kind = 1;
            const auto t0 = Clock::now();
            for (std::uint64_t p = p0; p < p_end;) {
                const unsigned n = static_cast<unsigned>(std::min<std::uint64_t>(8, p_end - p));
                const auto v = eng.verify(0, std::span(toks).subspan(p, n), p, ctl);
                if (v.replay_flag) throw std::runtime_error("tf: replay row flagged");
                for (unsigned j = 0; j < n; ++j)
                    if (v.flag[j] != 0) throw std::runtime_error("tf: row flag at position " + std::to_string(p + j));
                if (p + n > T - tf_score) {  // some rows of this group are scored
                    lg.resize(static_cast<std::size_t>(n) * V);
                    if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
                        throw std::runtime_error("logits copy");
                    for (unsigned j = 0; j < n; ++j)
                        if (p + j >= T - tf_score)
                            std::copy_n(lg.data() + static_cast<std::size_t>(j) * V, V,
                                        keep.data() + static_cast<std::size_t>(p + j - (T - tf_score)) * V);
                }
                eng.commit(0, n, p);
                ++ctl.cycle;
                p += n;
            }
            eng.flush(0);
            const double dt = secs(t0, Clock::now());
            if (tf_stop != 0) {
                const auto st = eng.state_bytes_upto(0, tf_stop);
                const std::uint64_t hdr[2] = {tf_stop, T};
                {
                    std::ofstream f(state_out + ".tmp", std::ios::binary);
                    f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
                    f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size()));
                    if (!f) throw std::runtime_error("tf: write " + state_out);
                }
                if (std::rename((state_out + ".tmp").c_str(), state_out.c_str()) != 0) throw std::runtime_error("tf: rename " + state_out);
                std::printf("rank %u tf: stopped at %llu, %.1f s (%.2f ms/position), state %zu B fnv %016llx -> %s\n", opt.rank,
                            static_cast<unsigned long long>(tf_stop), dt, 1000.0 * dt / static_cast<double>(tf_stop), st.size(),
                            static_cast<unsigned long long>(fnv(st.data(), st.size())), state_out.c_str());
            } else {
                const std::string path = out_path + "-r" + std::to_string(opt.rank) + ".f32";
                {
                    std::ofstream f(path + ".tmp", std::ios::binary);
                    f.write(reinterpret_cast<const char*>(keep.data()), static_cast<std::streamsize>(keep.size() * 4));
                    if (!f) throw std::runtime_error("tf: write " + path);
                }
                if (std::rename((path + ".tmp").c_str(), path.c_str()) != 0) throw std::runtime_error("tf: rename " + path);
                std::printf("rank %u tf: T %llu scored %u vocab slice %u %.1f s (%.2f ms/position) logits fnv %016llx -> %s\n", opt.rank,
                            static_cast<unsigned long long>(T), tf_score, V, dt, 1000.0 * dt / static_cast<double>(T),
                            static_cast<unsigned long long>(fnv(keep.data(), keep.size() * 4)), path.c_str());
                if (tf_check) {
                    const auto tf_state = eng.state_bytes_upto(0, T);
                    eng.reset(0);
                    unsigned bad = 0;
                    const auto t1 = Clock::now();
                    for (std::uint64_t p = 0; p < T; ++p) {
                        const ge::Row r{0, toks[p], p};
                        (void)eng.step({&r, 1});
                        if (p >= T - tf_score) {
                            lg.resize(V);
                            if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
                                throw std::runtime_error("logits copy");
                            if (std::memcmp(lg.data(), keep.data() + static_cast<std::size_t>(p - (T - tf_score)) * V, V * 4) != 0) ++bad;
                        }
                    }
                    const bool state_eq = eng.state_bytes_upto(0, T) == tf_state;
                    std::printf("rank %u tf-check: row-by-row %.1f s (%.2f ms/position): scored logits rows differing %u of %u, "
                                "state %s -> %s\n", opt.rank, secs(t1, Clock::now()), 1000.0 * secs(t1, Clock::now()) / static_cast<double>(T),
                                bad, tf_score, state_eq ? "EQUAL" : "DIFFER", bad == 0 && state_eq ? "PASS" : "FAIL");
                    if (bad != 0 || !state_eq) return 1;
                }
            }
        }
        if (g4_rmax > 0) {
            if (prompt.empty() || batch_steps == 0) throw std::invalid_argument("--g4 needs --prompt and --steps");
            const auto toks = read_u32(prompt);
            const std::uint32_t V = eng.vocab_slice();
            std::vector<float> lg;
            // one run: R rows, probe at row/slot `at`; returns {logits chain fnv, tokens fnv} of the probe stream
            auto run = [&](unsigned R, unsigned at) {
                std::vector<ge::Row> rows(R);
                for (unsigned r = 0; r < R; ++r) {
                    eng.reset(r);
                    rows[r] = ge::Row{r, 1000 + r, 0};
                }
                rows[at] = ge::Row{at, eng.prefill(at, toks, 0), toks.size()};
                std::uint64_t chain = 14695981039346656037ULL;
                std::vector<std::uint32_t> ptok{rows[at].token};
                for (unsigned s = 0; s < batch_steps; ++s) {
                    const auto nx = eng.step(rows);
                    lg.resize(static_cast<std::size_t>(R) * V);
                    if (hipMemcpy(lg.data(), eng.last_logits(), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
                        throw std::runtime_error("logits copy");
                    chain = fnv(lg.data() + static_cast<std::size_t>(at) * V, static_cast<std::size_t>(V) * 4, chain);
                    ptok.push_back(nx[at]);
                    for (unsigned r = 0; r < R; ++r) rows[r] = ge::Row{r, nx[r], rows[r].pos + 1};
                }
                return std::pair{chain, fnv(ptok.data(), ptok.size() * 4)};
            };
            const auto ref = run(1, 0);
            unsigned runs = 1, bad = 0;
            for (unsigned R = 2; R <= g4_rmax; ++R)
                for (unsigned at : {0U, R - 1}) {
                    const auto got = run(R, at);
                    ++runs;
                    const bool ok = got == ref;
                    bad += ok ? 0 : 1;
                    std::printf("rank %u g4: R %u probe row %u logits-chain %016llx tokens %016llx %s\n", opt.rank, R, at,
                                static_cast<unsigned long long>(got.first), static_cast<unsigned long long>(got.second),
                                ok ? "==" : "DIFFERS");
                }
            const auto again = run(1, 0);
            ++runs;
            bad += again == ref ? 0 : 1;
            std::printf("rank %u g4: %s (%u runs, %u differ; R1 logits-chain %016llx tokens %016llx; second R1 pass %s)\n",
                        opt.rank, bad == 0 ? "PASS" : "FAIL", runs, bad, static_cast<unsigned long long>(ref.first),
                        static_cast<unsigned long long>(ref.second), again == ref ? "==" : "DIFFERS");
            if (bad != 0) return 1;
        }
        if (!pf_gate.empty() || !pf_ttft.empty() || !pf_null_study.empty()) {
            // prefill real-model gate. Every rank runs the same sequence (the exchanges pair up). Paths are swapped through
            // the reference first, so only one chunked path holds memory at a time.
            namespace pf = strix::glm::pf;
            const unsigned heads = 64 / opt.world;
            struct Run {
                std::uint32_t token = 0;
                std::vector<float> logits;
                std::vector<std::uint8_t> state;
                double s = 0;
            };
            auto install = [&](const std::string& m) {
                eng.set_prefill(ge::reference_prefill());
                if (m != "ref") eng.set_prefill(pf::make_prefill_path(heads, heads, opt.capacity, opt.max_rows, m == "gemv" ? pf::TierB::gemv : pf::TierB::wmma));
            };
            auto run = [&](const std::vector<std::uint32_t>& toks, bool keep_state) {
                Run r;
                // the MoE seam's tables were built by the install / set_moe, so the prefill allocates none of them
                if (const std::uint64_t cold = eng.moe_cold_bytes(); cold != 0)
                    throw std::runtime_error("MoE seam still cold before a prefill (" + std::to_string(cold) + " bytes to allocate)");
                eng.reset(0);
                std::size_t free0 = 0, free1 = 0, total = 0;
                if (hipMemGetInfo(&free0, &total) != hipSuccess) throw std::runtime_error("mem info");
                const auto t0 = Clock::now();
                r.token = eng.prefill(0, toks, 0);
                r.s = secs(t0, Clock::now());
                if (hipMemGetInfo(&free1, &total) != hipSuccess) throw std::runtime_error("mem info");
                // no device allocation inside a prefill (the path, its workspace and the seams' tables exist before it)
                std::printf("rank %u pf dev free delta %+.1f MiB (%zu rows)\n", opt.rank,
                            (static_cast<double>(free1) - static_cast<double>(free0)) / double(1ULL << 20), toks.size());
                r.logits.resize(eng.vocab_slice());
                if (hipMemcpy(r.logits.data(), eng.last_logits(), r.logits.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
                    throw std::runtime_error("logits copy");
                if (keep_state) r.state = eng.state_bytes_upto(0, toks.size());
                return r;
            };
            // Phase consensus: both ranks exchange their phase status before every phase, so a rank that failed locally
            // (admission, a local check) makes both stop together instead of leaving the peer inside a prefill exchange.
            std::uint32_t* ph_d = nullptr;
            if (comm && hipMalloc(&ph_d, 16) != hipSuccess) throw std::runtime_error("phase buffer");
            unsigned phase_no = 0;
            auto phase = [&](bool ok_local, const char* what) {
                if (!comm) {  // one rank (world 1 or the load test): local status only
                    if (!ok_local) throw std::runtime_error(std::string("pf phase failed: ") + what);
                    return;
                }
                const std::uint32_t mine = ok_local ? 1U : 0U;
                std::uint32_t peer = 0;
                if (hipMemcpy(ph_d, &mine, 4, hipMemcpyHostToDevice) != hipSuccess) throw std::runtime_error("phase up");
                strix::dist::Tag tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0x9F00U + phase_no++, 0};
                comm->exchange(nullptr, ph_d, ph_d + 2, 4, tag);
                if (hipMemcpy(&peer, ph_d + 2, 4, hipMemcpyDeviceToHost) != hipSuccess) throw std::runtime_error("phase down");
                if (!ok_local || peer != 1U)
                    throw std::runtime_error(std::string("pf phase ") + what + ": local " + (ok_local ? "ok" : "FAIL") + ", peer " + (peer == 1U ? "ok" : "FAIL"));
            };
            auto guarded_install = [&](const std::string& m) {  // a local admission failure is reported, then both stop
                bool okl = true;
                std::string err;
                try {
                    install(m);
                } catch (const std::exception& e) {
                    okl = false;
                    err = e.what();
                    std::fprintf(stderr, "error: %s\n", e.what());
                }
                phase(okl, ("install " + m + (err.empty() ? "" : ": " + err)).c_str());
            };
            auto guarded_run = [&](const std::vector<std::uint32_t>& toks, bool keep_state, const char* what) {  // run + consensus
                Run res;
                bool okl = true;
                try {
                    res = run(toks, keep_state);
                } catch (const std::exception& e) {
                    okl = false;
                    std::fprintf(stderr, "error: %s: %s\n", what, e.what());
                }
                phase(okl, what);
                return res;
            };
            auto bytes_eq = [](const auto& a, const auto& b) {
                return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
            };
            auto digest = [&](const char* m, const Run& r, std::size_t n) {
                std::printf("rank %u pf %s: %zu tokens %.3f s (%.3f ms/token), token %u, logits fnv %016llx, state fnv %016llx\n", opt.rank, m, n, r.s,
                            1000.0 * r.s / static_cast<double>(n), r.token, static_cast<unsigned long long>(fnv(r.logits.data(), r.logits.size() * 4)),
                            static_cast<unsigned long long>(fnv(r.state.data(), r.state.size())));
            };
            if (!pf_gate.empty()) {
                const auto toks = read_u32(pf_gate);
                guarded_install("ref");
                const Run ref = guarded_run(toks, true, "ref prefill");
                digest("ref", ref, toks.size());
                guarded_install("gemv");
                if (pf_swap_moe) {  // review regression: set_moe with a prefill installed must build the new seam's tables
                    bool ok = true;
                    try {
                        eng.set_moe(ge::fast_moe());
                        std::printf("rank %u moe seam replaced after the gemv install, cold bytes %llu\n", opt.rank,
                                    static_cast<unsigned long long>(eng.moe_cold_bytes()));
                    } catch (const std::exception& e) {
                        std::printf("rank %u FAIL moe seam replacement: %s\n", opt.rank, e.what());
                        ok = false;
                    }
                    phase(ok, "moe seam replacement");
                }
                const Run gv = guarded_run(toks, true, "gemv prefill");
                digest("gemv", gv, toks.size());
                const bool same = gv.token == ref.token && bytes_eq(gv.logits, ref.logits) && bytes_eq(gv.state, ref.state);
                std::printf("rank %u %s   chunked prefill Tier B gemv == reference prefill, bitwise (token, rank logits bytes, state bytes to the prompt end)\n",
                            opt.rank, same ? "OK" : "FAIL");
                phase(same, "gemv == reference");
                // STRIX_GLM_PF_GATE_WMMA=0: exact gate only (the D10 wmma arm needs prepared attention linears)
                const char* gw = std::getenv("STRIX_GLM_PF_GATE_WMMA");
                if (gw == nullptr || std::string(gw) != "0") {
                guarded_install("wmma");
                const Run w1 = guarded_run(toks, true, "wmma prefill 1");
                digest("wmma", w1, toks.size());
                const Run w2 = guarded_run(toks, true, "wmma prefill 2");
                const bool det = w2.token == w1.token && bytes_eq(w2.logits, w1.logits) && bytes_eq(w2.state, w1.state);
                std::printf("rank %u %s   wmma two runs bitwise equal\n", opt.rank, det ? "OK" : "FAIL");
                phase(det, "wmma determinism");
                double num = 0, den = 0, mx = 0;
                for (std::size_t i = 0; i < ref.logits.size(); ++i) {
                    const double d = double(w1.logits[i]) - double(ref.logits[i]);
                    num += d * d;
                    den += double(ref.logits[i]) * ref.logits[i];
                    mx = std::max(mx, std::abs(d));
                }
                std::size_t eq = 0;
                for (std::size_t i = 0; i < ref.state.size(); ++i) eq += w1.state[i] == ref.state[i];
                std::printf("rank %u DEV wmma vs ref: token %u vs %u, rank logits rel-L2 %.3e max |d| %.3e, state bytes equal %.4f %%; ref %.1f s, gemv %.1f s, wmma %.1f s\n",
                            opt.rank, w1.token, ref.token, den > 0 ? std::sqrt(num / den) : 0.0, mx,
                            ref.state.empty() ? 0.0 : 100.0 * double(eq) / double(ref.state.size()), ref.s, gv.s, w1.s);
                } else {
                    std::printf("rank %u pf-gate: wmma arm skipped (STRIX_GLM_PF_GATE_WMMA=0); ref %.1f s, gemv %.1f s\n", opt.rank, ref.s, gv.s);
                }
            }
            if (!pf_ttft.empty()) {
                std::size_t mb = 0;
                while (mb <= pf_ttft_mode.size()) {  // each mode: install once, then every prompt
                    const std::size_t me = std::min(pf_ttft_mode.find(',', mb), pf_ttft_mode.size());
                    const std::string mode = pf_ttft_mode.substr(mb, me - mb);
                    if (mode != "wmma" && mode != "gemv") throw std::invalid_argument("--pf-ttft-mode: wmma | gemv | wmma,gemv");
                    guarded_install(mode.c_str());
                    std::size_t a = 0;
                    while (a <= pf_ttft.size()) {
                        const std::size_t b = std::min(pf_ttft.find(',', a), pf_ttft.size());
                        const auto toks = read_u32(pf_ttft.substr(a, b - a));
                        if (toks.size() + 1 > opt.capacity) throw std::invalid_argument("--pf-ttft prompt longer than --capacity");
                        const Run r = guarded_run(toks, false, (mode + " ttft").c_str());
                        digest((mode + "-ttft").c_str(), r, toks.size());
                        a = b + 1;
                    }
                    mb = me + 1;
                }
            }
            if (!pf_null_study.empty()) {
                // D10 noise reference (measurement test 1 shape, informational): the exact path (gemv-chunked, == the
                // reference prefill bitwise) vs the wmma candidate and vs exact runs with one 1-ulp residual flip.
                const auto toks = read_u32(pf_null_study);
                if (toks.size() + 1 > opt.capacity) throw std::invalid_argument("--pf-null-study prompt longer than --capacity");
                auto dev = [&](const char* what, const Run& x, const Run& ex) {
                    double num = 0, den = 0, mx = 0;
                    for (std::size_t i = 0; i < ex.logits.size(); ++i) {
                        const double d = double(x.logits[i]) - double(ex.logits[i]);
                        num += d * d;
                        den += double(ex.logits[i]) * ex.logits[i];
                        mx = std::max(mx, std::abs(d));
                    }
                    std::size_t eq = 0;
                    for (std::size_t i = 0; i < ex.state.size(); ++i) eq += x.state[i] == ex.state[i];
                    std::printf("rank %u DEV %s vs exact: token %u vs %u, rank logits rel-L2 %.3e max |d| %.3e, state bytes equal %.4f %%, %.1f s\n",
                                opt.rank, what, x.token, ex.token, den > 0 ? std::sqrt(num / den) : 0.0, mx,
                                ex.state.empty() ? 0.0 : 100.0 * double(eq) / double(ex.state.size()), x.s);
                };
                guarded_install("gemv");
                const Run ex = guarded_run(toks, true, "null study exact");
                digest("exact", ex, toks.size());
                guarded_install("wmma");
                const Run cand = guarded_run(toks, true, "null study wmma");
                dev("wmma", cand, ex);
                guarded_install("gemv");
                std::size_t a = 0;
                while (!pf_nulls.empty() && a <= pf_nulls.size()) {
                    const std::size_t b = std::min(pf_nulls.find(',', a), pf_nulls.size());
                    const std::string spec = pf_nulls.substr(a, b - a);
                    const std::size_t c1 = spec.find(':'), c2 = spec.find(':', c1 + 1);
                    if (c1 == std::string::npos || c2 == std::string::npos) throw std::invalid_argument("--pf-null LAYER:POS:ELEM");
                    const int layer = std::stoi(spec.substr(0, c1));
                    const auto pos = static_cast<std::uint64_t>(std::stoull(spec.substr(c1 + 1, c2 - c1 - 1)));
                    const auto elem = static_cast<unsigned>(std::stoul(spec.substr(c2 + 1)));
                    if (pos >= toks.size()) throw std::invalid_argument("--pf-null position past the prompt");
                    eng.set_null_flip(layer, pos, elem);
                    const Run nr = guarded_run(toks, true, "null study null");
                    eng.set_null_flip(-1, 0, 0);
                    dev(("null L" + std::to_string(layer) + " pos " + std::to_string(pos) + " elem " + std::to_string(elem)).c_str(), nr, ex);
                    a = b + 1;
                }
            }
            if (ph_d) (void)hipFree(ph_d);
            std::printf("rank %u pf modes complete\n", opt.rank);
        }
        std::printf("rank %u: done %.1f s\n", opt.rank, secs(t_start, Clock::now()));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
