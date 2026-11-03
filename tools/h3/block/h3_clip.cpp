// strix-h3-clip: single-node MiniMax H3 text-to-video+audio clip runner (gfx1151).
//
//   prompt -> tokenizer + presentation -> text encoder (GPU, FP32 hidden states after layer 50)
//          -> bf16 cast -> condition_proj + token refiner (GPU) -> seeded noise (torch CPU randn port)
//          -> DiT sampler (simple schedule, shift 12 / audio 3, res_multistep or Euler, N steps; GPU forward)
//          -> finalize -> video VAE decode (36-layer, tiled + chunked) + audio decode (GPU)
//          -> raw frames (uint8 RGB [F][H][W][3]) + 16-bit PCM stereo WAV (32 kHz) + latents + meta.json.
//
// Shapes follow the reference's EmptyMiniMaxH3LatentAV: the frame count snaps up to the 17k + 5 grid at 24 fps,
// latent T = ((frames - 5) / 17) * 5 + 2, audio latent length = round(frames / 24 * 40), H, W = height / 16, width / 16.
// No CFG (the H3 workflows run cfg 1). Stages run one after another; each model is freed before the next loads.
//
// Resumable (a 5 s 768x1344 clip is ~100 s per step): with --budget SECONDS the run stops before a stage or step that
// would not fit and exits with code 3; the output directory keeps the text encoder output, the sampler carry after the
// last finished step (dit_state.bin) and the final latents, and the same command continues from there (same bits as
// one uninterrupted run). The run's identity (prompt, shape, steps, sampler, seed, shifts) is checked on resume.
//
// DiT backends (--dit): single = the single-node DeviceForward; dist = the two-node schedule's DeviceDistRank (world 1:
// both key blocks local; world 2: one key block per node, per-block K/V exchange, velocity all-gather). The token
// refiner always runs once on rank 0 (DeviceForward::refine_text) and its output is sent to rank 1, so every form uses
// the same refined text bits.
// Two nodes (--world 2 --rank R --transport usb4auto, or both ranks as threads of one process: --transport emulate):
// rank 0 owns the prompt, text encoder, refiner, noise, sampler state and checkpoints, the audio VAE, the VAE finish and
// the outputs; rank 1 holds the DiT weights and serves the requests of rank 0 in lockstep (BEGIN: layout + refined
// text; STEP: sigma + stream inputs, then both ranks run the forward; VAE: normalized latent, rank 1 decodes the calls
// [n/2 + shift, n) of the canonical order and streams each piece (chunk canvas or a cut chunk's tiles) to rank 0's frame
// assembler as soon as it is ready; END). Rank 1 keeps no state across invocations, so the
// resume is rank 0's alone.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "h3/block/device_forward.hpp"
#include "h3/block/turbo_accel.hpp"
#include "h3/levers/lora/lora_file.hpp"
#include "h3/block/carry_io.hpp"
#include "h3/block/sampler.hpp"
#include "h3/block/torch_rng.hpp"
#include "h3/te/te_gpu.hpp"
#include "h3/te/te_sha256.hpp"
#include "h3/te/te_tokenizer.hpp"
#include "h3/vae/audio_decoder.hpp"
#include "h3/vae/vae_decoder.hpp"
#include "h3/vae/vae_shard.hpp"
#include "h3/vae/fast_decoder.hpp"
#ifdef H3_CLIP_DIST
#include "h3/dist/accel.hpp"
#include "h3/dist/device_dist.hpp"
#include "h3/dist/host_dist.hpp"
#include "strix/dist/communicator.hpp"
#endif

namespace blk = strix::h3::block;
namespace te = strix::h3::te;
namespace vae = strix::h3::vae;
#ifdef H3_CLIP_DIST
namespace dst = strix::h3::dist;
namespace sd = strix::dist;
#endif

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Args {
    std::string prompt = "A red fox trots through fresh snow in a quiet birch forest at dawn, soft light, its breath visible.";
    int width = 256, height = 256, length = 17, steps = 8;
    std::uint64_t seed = 42;
    std::string sampler = "res_multistep";
    float shift = 12.0F, shift_audio = 3.0F;
    std::string out = "clip-out";
    double budget = 1e30;      // seconds for this invocation
    double vae_s_per_call = 0.6;  // budget estimate of the video VAE (per decoder call, + 20 s)
    std::string tok, te_pack, ckpt, vae_video, vae_audio;
    std::string dit = "single";   // single | dist
    int world = 1, rank = 0;
    std::string transport = "usb4auto";  // world 2: usb4auto | emulate
    int vae_shift = 1;            // world 2: decoder calls moved from rank 1 to rank 0 (split at n / 2 + shift)
    bool fast_vae = false;        // h3-vae-fast24-v1 (FAST-class): the distilled 24-layer decoder, whole clip in one pass
    std::string fast_vae_weights;
    std::string decode_only;      // dir with latent_video_f32.bin + latent_audio_f32.bin: VAE + outputs only (1 node)
    // accelerated forms (FAST-class, default off = dense): h3-turbo-v4-bypass and h3-fbc-v1
    bool turbo = false;
    float strength = 0.70F;
    std::string lora, merged_fc2, egrid;
    // step-distilled LoRA identity (--lora <id>, levers lora_file.hpp table): rank-generic bypass incl. a run-time fc2
    // branch + full-weight deltas, no adaLN part; preset strength / steps / Euler sampler from the identity
    std::string step_lora, step_lora_file;
    float lora_strength = -1.0F;  // < 0: the identity's preset
    bool steps_set = false;
    bool fbc = false, fbc_observe = false;
    double fbc_threshold = 0.08;
    bool sampler_set = false;
    bool dump_velocity = false;   // write every step's model output (bf16) + sigma to <out>/velocity/
    bool sparse = false;
    float sparse_tau = 0.75F;
    std::uint32_t sparse_dense_calls = 3;
    std::string sparse_dense_layers;  // extra dense DiT layers, e.g. "39-42"
};

std::string env_or(const char* k, const std::string& d) {
    const char* v = std::getenv(k);
    return v && *v ? std::string(v) : d;
}

void usage() {
    std::puts(
        "strix-h3-clip [--prompt TEXT | --prompt-file F] [--width 256] [--height 256] [--length 17] [--steps 8]\n"
        "              [--sampler res_multistep|euler] [--seed 42] [--shift 12] [--shift-audio 3] [--out DIR] [--budget S]\n"
        "              [--tok DIR] [--te-pack DIR] [--ckpt F] [--vae F] [--audio-vae F]\n"
        "              [--turbo [--strength 0.70] [--lora F] [--merged-fc2 F] [--egrid F]] (FAST-class, no numeric gate)\n"
        "              [--lora h3-lora-fasth3-6step-v1|h3-lora-turbo-v12-v1 [--lora-strength x] [--lora-file F]] (step LoRAs,\n"
        "               FAST-class: preset strength / steps from the identity, Euler simple; file from H3_MODELS by default)\n"
        "              [--fbc 0.08 | --fbc-observe]\n"
        "              [--sparse 0.75 [--sparse-dense-layers 39-42] [--sparse-dense-calls 3]]\n"
        "              [--fast-vae [--fast-vae-weights F]] (h3-vae-fast24-v1, FAST-class) [--decode-only LATENT_DIR (1 node)]\n"
        "              [--dump-velocity] (per step: <out>/velocity/step_NN_{video,audio}_neg_v_bf16.bin + index.txt)\n"
        "              [--dit single|dist] [--world 1|2 --rank R --transport usb4auto|emulate] [--vae-shift 1]\n"
        "asset paths: flags, or env H3_TOKENIZER_DIR, H3_TE_PACK, H3_CKPT, H3_VAE_WEIGHTS, H3_AUDIO_VAE_WEIGHTS, H3_TURBO_LORA,\n"
        "H3_TURBO_MERGED_FC2, H3_TURBO_EGRID, or H3_MODELS (a directory holding the default file names); no built-in location.\n"
        "--fbc (reusing) and --sparse runs cannot be resumed across invocations (their state is not persisted).");
}

Args parse(int argc, char** argv) {
    Args a;
    const std::string models = env_or("H3_MODELS", "");
    auto def = [&](const char* env, const char* file) { return env_or(env, models.empty() ? std::string() : models + "/" + file); };
    a.tok = env_or("H3_TOKENIZER_DIR", "");
    a.te_pack = def("H3_TE_PACK", "te-pack");
    a.ckpt = def("H3_CKPT", "minimax_h3_fl2va_pruned_int8_convrot.safetensors");
    a.vae_video = def("H3_VAE_WEIGHTS", "minimax_h3_video_vae_fp16.safetensors");
    a.vae_audio = def("H3_AUDIO_VAE_WEIGHTS", "minimax_h3_audio_vae_fp32.safetensors");
    a.lora = def("H3_TURBO_LORA", "minimax_h3_turbo_v4_step600_ema_comfyui.safetensors");
    a.merged_fc2 = def("H3_TURBO_MERGED_FC2", "merged_fc2_s070.safetensors");
    a.egrid = def("H3_TURBO_EGRID", "h3_silu_temb_grid.safetensors");
    a.fast_vae_weights = def("H3_FAST_VAE_WEIGHTS", "fast_h3_vae_noncausal_24l.safetensors");
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--prompt") a.prompt = val();
        else if (k == "--prompt-file") {
            const std::string pf = val();
            std::ifstream f(pf, std::ios::binary);
            if (!f) throw std::runtime_error("cannot read the prompt file " + pf);
            std::stringstream ss;
            ss << f.rdbuf();
            if (f.bad()) throw std::runtime_error("cannot read the prompt file " + pf);
            a.prompt = ss.str();
            while (!a.prompt.empty() && (a.prompt.back() == '\n' || a.prompt.back() == '\r')) a.prompt.pop_back();
        } else if (k == "--width") a.width = std::stoi(val());
        else if (k == "--height") a.height = std::stoi(val());
        else if (k == "--length") a.length = std::stoi(val());
        else if (k == "--steps") {
            a.steps = std::stoi(val());
            a.steps_set = true;
        }
        else if (k == "--seed") a.seed = std::stoull(val());
        else if (k == "--sampler") {
            a.sampler = val();
            a.sampler_set = true;
        }
        else if (k == "--shift") a.shift = std::stof(val());
        else if (k == "--shift-audio") a.shift_audio = std::stof(val());
        else if (k == "--out") a.out = val();
        else if (k == "--budget") a.budget = std::stod(val());
        else if (k == "--tok") a.tok = val();
        else if (k == "--te-pack") a.te_pack = val();
        else if (k == "--ckpt") a.ckpt = val();
        else if (k == "--vae") a.vae_video = val();
        else if (k == "--fast-vae") a.fast_vae = true;
        else if (k == "--fast-vae-weights") a.fast_vae_weights = val();
        else if (k == "--decode-only") a.decode_only = val();
        else if (k == "--dump-velocity") a.dump_velocity = true;
        else if (k == "--audio-vae") a.vae_audio = val();
        else if (k == "--dit") a.dit = val();
        else if (k == "--world") a.world = std::stoi(val());
        else if (k == "--rank") a.rank = std::stoi(val());
        else if (k == "--transport") a.transport = val();
        else if (k == "--vae-shift") a.vae_shift = std::stoi(val());
        else if (k == "--turbo") a.turbo = true;
        else if (k == "--strength") a.strength = std::stof(val());
        else if (k == "--lora") {  // a step-LoRA identity id, else the Turbo v4 LoRA file
            const std::string v = val();
            if (v.rfind("h3-lora-", 0) == 0) a.step_lora = v;
            else a.lora = v;
        } else if (k == "--lora-strength") a.lora_strength = std::stof(val());
        else if (k == "--lora-file") a.step_lora_file = val();
        else if (k == "--merged-fc2") a.merged_fc2 = val();
        else if (k == "--egrid") a.egrid = val();
        else if (k == "--sparse") {
            a.sparse = true;
            a.sparse_tau = std::stof(val());
        } else if (k == "--sparse-dense-layers") a.sparse_dense_layers = val();
        else if (k == "--sparse-dense-calls") a.sparse_dense_calls = static_cast<std::uint32_t>(std::stoul(val()));
        else if (k == "--fbc") {
            a.fbc = true;
            a.fbc_threshold = std::stod(val());
        } else if (k == "--fbc-observe") {
            a.fbc = true;
            a.fbc_observe = true;
        }
        else if (k == "--pdd" || k.rfind("--pdd", 0) == 0) {
            throw std::runtime_error("PDD head banks are not wired in the runner (no PDD checkpoint / gate): " + k + " refused");
        } else if (k == "-h" || k == "--help") {
            usage();
            std::exit(0);
        } else throw std::runtime_error("unknown argument " + k);
    }
    auto finite_pos = [](double v) { return std::isfinite(v) && v > 0.0; };
    if (!finite_pos(a.shift) || !finite_pos(a.shift_audio)) throw std::runtime_error("--shift / --shift-audio: finite and > 0");
    if (a.steps < 1 || a.steps > 1000) throw std::runtime_error("--steps: 1 .. 1000 (the sigma table has 1000 entries)");
    if (!(a.budget > 0.0) || std::isnan(a.budget)) throw std::runtime_error("--budget: > 0");
    if (a.length < 1 || a.length > 4096) throw std::runtime_error("--length: 1 .. 4096 frames");
    if (a.width > 4096 || a.height > 4096) throw std::runtime_error("--width / --height: <= 4096");
    if (a.turbo && !(std::isfinite(a.strength) && a.strength > 0.0F && a.strength <= 4.0F)) throw std::runtime_error("--strength: finite, (0, 4]");
    if (a.fbc && !std::isfinite(a.fbc_threshold)) throw std::runtime_error("--fbc threshold: finite");
    if (a.sparse && !(std::isfinite(a.sparse_tau) && a.sparse_tau >= 0.0F)) throw std::runtime_error("--sparse tau: finite, >= 0");
    if (a.vae_shift < -100000 || a.vae_shift > 100000) throw std::runtime_error("--vae-shift out of range");
    if (a.world != 1 && a.world != 2) throw std::runtime_error("--world 1 | 2");
    if (a.rank < 0 || a.rank >= a.world) throw std::runtime_error("--rank out of range");
    if (a.world == 2) a.dit = "dist";
    if (a.turbo && !a.sampler_set) a.sampler = "euler";  // the identity's sampler: flow Euler, "simple" sigmas
    if (!a.step_lora.empty()) {
        if (a.turbo) throw std::runtime_error("--turbo and --lora <step-LoRA id> are exclusive");
        const strix::h3::levers::lora::LoraIdentity& li = strix::h3::levers::lora::lora_identity(a.step_lora);
        if (!li.fc2_bypass) throw std::runtime_error("--lora " + a.step_lora + ": not a step-LoRA identity");
        if (a.lora_strength < 0.0F) a.lora_strength = li.strength;
        if (!(std::isfinite(a.lora_strength) && a.lora_strength > 0.0F && a.lora_strength <= 4.0F))
            throw std::runtime_error("--lora-strength: finite, (0, 4]");
        if (!a.steps_set) a.steps = static_cast<int>(li.steps);
        if (!a.sampler_set) a.sampler = "euler";
        if (a.sampler != "euler") throw std::runtime_error("--lora " + a.step_lora + " runs the Euler sampler (" + li.sampler + ")");
        if (a.step_lora_file.empty()) a.step_lora_file = models.empty() ? std::string() : models + "/" + li.file;
    } else if (a.lora_strength >= 0.0F || !a.step_lora_file.empty()) {
        throw std::runtime_error("--lora-strength / --lora-file need --lora <step-LoRA id>");
    }
    if (a.turbo && a.sampler != "euler") throw std::runtime_error("--turbo runs the Euler sampler (h3-turbo-v4-bypass)");
    if (a.fbc && !(a.fbc_threshold >= 0.0)) throw std::runtime_error("--fbc threshold >= 0");
    if (a.dit != "single" && a.dit != "dist") throw std::runtime_error("--dit single | dist");
#ifndef H3_CLIP_DIST
    if (a.dit == "dist") throw std::runtime_error("built without the two-node schedule (dist package)");
#endif
    if (a.transport != "usb4auto" && a.transport != "emulate") throw std::runtime_error("--transport usb4auto | emulate");
    if (!a.decode_only.empty() && a.world != 1) throw std::runtime_error("--decode-only: one node");
    const bool needs_prompt = a.decode_only.empty() && (a.rank == 0 || (a.world == 2 && a.transport == "emulate"));
    if (needs_prompt && a.tok.empty()) throw std::runtime_error("tokenizer dir required (--tok or H3_TOKENIZER_DIR)");
    if (a.width % 32 || a.height % 32 || a.width < 32 || a.height < 32) throw std::runtime_error("width / height: multiples of 32");
    if (needs_prompt && a.prompt.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error("empty prompt");
    // every asset this rank opens must exist before any model is loaded
    auto need_file = [](const std::string& path, const char* what) {
        if (path.empty()) throw std::runtime_error(std::string(what) + ": no path (flag, its env variable or H3_MODELS)");
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) throw std::runtime_error(std::string(what) + " not found: " + path);
    };
    if (a.decode_only.empty()) need_file(a.ckpt, "DiT checkpoint");
    if (!a.fast_vae) need_file(a.vae_video, "video VAE (36L)");
    if (a.fast_vae) need_file(a.fast_vae_weights, "fast VAE (24L) weights");
    if (!a.decode_only.empty()) {
        need_file(a.decode_only + "/latent_video_f32.bin", "decode-only video latents");
        need_file(a.decode_only + "/latent_audio_f32.bin", "decode-only audio latents");
        need_file(a.vae_audio, "audio VAE");
    }
    if (needs_prompt) {
        need_file(a.tok, "tokenizer dir");
        need_file(a.te_pack, "text encoder pack");
        need_file(a.vae_audio, "audio VAE");
    }
    if (!a.step_lora.empty()) need_file(a.step_lora_file, "step LoRA file");
    if (a.turbo) {
        need_file(a.lora, "Turbo LoRA");
        need_file(a.merged_fc2, "Turbo merged fc2");
        need_file(a.egrid, "Turbo E-grid");
    }
    if (a.sampler != "res_multistep" && a.sampler != "euler") throw std::runtime_error("sampler: res_multistep | euler");
    return a;
}

// Reference temporal shape (nodes_minimax_h3.temporal_shape).
struct Temporal {
    int frames = 0, latent_t = 0, audio_t = 0;
};
Temporal temporal_shape(int length) {
    int n = std::max(5, length);
    while (n % 17 != 5) ++n;
    Temporal t;
    t.frames = n;
    t.latent_t = n <= 5 ? 2 : ((n - 5) / 17) * 5 + 2;
    // round(duration * 40) with Python's round-half-even on the binary value
    t.audio_t = static_cast<int>(std::nearbyint(static_cast<double>(n) / 24.0 * 40.0));
    return t;
}

std::vector<std::uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<std::uint16_t> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) o[i] = blk::tob(v[i]);
    return o;
}
std::vector<float> widen(const std::vector<std::uint16_t>& v) {
    std::vector<float> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) o[i] = blk::bf(v[i]);
    return o;
}

// Test hooks (strix-h3-clip-testhooks only: compiled with H3_CLIP_TEST_HOOKS; the production binary has none).
std::string test_hook(const char* name) {
#ifdef H3_CLIP_TEST_HOOKS
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
#else
    static_cast<void>(name);
    return std::string();
#endif
}

void write_bin(const std::filesystem::path& p, const void* data, std::size_t bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    f.flush();
    if (!f) throw std::runtime_error("write " + p.string());
}
// Appends one line; the stream state is checked AFTER the flush (a failed write throws).
void append_line(const std::filesystem::path& p, const std::string& line) {
    if (test_hook("H3_CLIP_TEST_FAIL") == "write-outputs" && p.filename() == "hashes.txt" && line.rfind("final ", 0) == 0)
        throw std::runtime_error("injected output-write failure (H3_CLIP_TEST_FAIL=write-outputs)");
    std::ofstream f(p, std::ios::app);
    f << line;
    f.flush();
    if (!f) throw std::runtime_error("write " + p.string());
}

void write_wav(const std::filesystem::path& p, const std::vector<float>& wave, int channels, int rate) {
    const std::size_t n = wave.size() / static_cast<std::size_t>(channels);
    std::vector<std::int16_t> pcm(n * static_cast<std::size_t>(channels));
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < channels; ++c) {
            const float v = std::clamp(wave[static_cast<std::size_t>(c) * n + i], -1.0F, 1.0F);  // planar -> interleaved
            pcm[i * static_cast<std::size_t>(channels) + static_cast<std::size_t>(c)] = static_cast<std::int16_t>(std::lrint(v * 32767.0F));
        }
    std::ofstream f(p, std::ios::binary);
    auto u32 = [&](std::uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](std::uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(pcm.size() * 2);
    f.write("RIFF", 4);
    u32(36 + data_bytes);
    f.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(static_cast<std::uint16_t>(channels));
    u32(static_cast<std::uint32_t>(rate));
    u32(static_cast<std::uint32_t>(rate * channels * 2));
    u16(static_cast<std::uint16_t>(channels * 2));
    u16(16);
    f.write("data", 4);
    u32(data_bytes);
    f.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(data_bytes));
    f.flush();
    if (!f) throw std::runtime_error("write " + p.string());
}

std::vector<char> read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
template <class T>
std::vector<T> read_vec(const std::filesystem::path& p) {
    const auto b = read_file(p);
    std::vector<T> v(b.size() / sizeof(T));
    if (!v.empty()) std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
    return v;
}

// ---------------------------------------------------------------- run identity (exact bits + asset content identities)
// --dump-velocity coverage: index.txt holds exactly the steps 1..n once each in order, and each step's three files exist
// with the expected sizes; throws otherwise.
void check_dump(const std::filesystem::path& od, std::size_t n, std::size_t nv, std::size_t na) {
    const std::filesystem::path vd = od / "velocity";
    std::ifstream ix(vd / "index.txt");
    std::size_t k = 0;
    for (std::string l; std::getline(ix, l);) {
        if (l.empty()) continue;
        ++k;
        if (l.rfind("step " + std::to_string(k) + " ", 0) != 0)
            throw std::runtime_error("--dump-velocity: velocity/index.txt line " + std::to_string(k) + " is not step " + std::to_string(k));
    }
    if (k != n)
        throw std::runtime_error("--dump-velocity: velocity/index.txt covers " + std::to_string(k) + " of the " + std::to_string(n) + " finished steps");
    for (std::size_t i = 1; i <= n; ++i) {
        char nb[32];
        std::snprintf(nb, sizeof nb, "step_%02zu", i);
        const std::pair<const char*, std::size_t> files[3] = {{"_video_neg_v_bf16.bin", nv * 2}, {"_audio_neg_v_bf16.bin", na * 2},
            {"_audio_carried_v_bf16.bin", na * 2}};
        for (const auto& f : files) {
            std::error_code ec;
            const auto fp = vd / (std::string(nb) + f.first);
            const auto sz = std::filesystem::file_size(fp, ec);
            if (ec || sz != f.second) throw std::runtime_error("--dump-velocity: " + fp.filename().string() + " missing or of the wrong size");
        }
    }
}

std::string hexf(double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%a", v);
    return b;
}
// Content identity of an asset: SHA-256 of the whole file (directories: of their sorted regular files, name + digest),
// cached per (canonical path, size, mtime) in H3_ASSET_ID_CACHE (default ./.h3-asset-ids) so a 21 GB checkpoint is
// hashed once per node.
std::string content_id(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path cp = fs::weakly_canonical(path, ec);
    if (ec) throw std::runtime_error("asset not accessible: " + path);
    if (fs::is_directory(cp)) {
        std::vector<std::string> names;
        for (const auto& e : fs::directory_iterator(cp))
            if (e.is_regular_file()) names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());
        std::string acc;
        for (const auto& n : names) acc += n + "=" + content_id((cp / n).string()) + ";";
        return te::sha256_hex(acc.data(), acc.size());
    }
    const auto size = fs::file_size(cp, ec);
    if (ec) throw std::runtime_error("asset not readable: " + path);
    const auto mt = fs::last_write_time(cp, ec).time_since_epoch().count();
    const std::string key = cp.string() + " " + std::to_string(size) + " " + std::to_string(static_cast<long long>(mt));
    const std::string cache = env_or("H3_ASSET_ID_CACHE", ".h3-asset-ids");
    {
        std::ifstream f(cache);
        for (std::string line; std::getline(f, line);)
            if (line.size() > 65 && line.compare(65, std::string::npos, key) == 0) return line.substr(0, 64);
    }
    const std::string h = te::sha256_file_hex(cp.string());
    std::ofstream(cache, std::ios::app) << h << " " << key << "\n";
    return h;
}
std::string build_id() {
    std::error_code ec;
    if (std::filesystem::exists("/proc/self/exe", ec)) return content_id(std::filesystem::read_symlink("/proc/self/exe", ec).string());
    return "unknown";
}
// The assets rank 1 also opens (DiT, video VAE, Turbo files) + the binary: both ranks must agree (HELLO handshake).
std::string shared_assets(const Args& a) {
    std::ostringstream o;
    o << "build=" << build_id() << "\nckpt=" << content_id(a.ckpt) << "\nvae_video="
      << (a.fast_vae ? std::string("unused (fast VAE)") : content_id(a.vae_video)) << "\n";
    if (a.turbo) o << "lora=" << content_id(a.lora) << "\nmerged_fc2=" << content_id(a.merged_fc2) << "\negrid=" << content_id(a.egrid) << "\n";
    if (!a.step_lora.empty())
        o << "step_lora=" << a.step_lora << " " << content_id(a.step_lora_file) << " strength=" << hexf(a.lora_strength) << "\n";
    o << "accel turbo=" << a.turbo << " strength=" << hexf(a.strength) << " fbc=" << a.fbc << " observe=" << a.fbc_observe
      << " threshold=" << hexf(a.fbc_threshold) << "\n";
    if (a.sparse) o << "sparse tau=" << hexf(a.sparse_tau) << " dense_calls=" << a.sparse_dense_calls << " dense_layers=" << a.sparse_dense_layers << "\n";
    return o.str();
}

// ---------------------------------------------------------------- recorded timings (budget admission)
struct Timings {
    double setup_s = 0, step_s = 0, vae_s = 0;
};
Timings read_timings(const std::filesystem::path& p) {
    Timings t;
    std::ifstream f(p);
    if (f) f >> t.setup_s >> t.step_s >> t.vae_s;
    if (!f || !(t.setup_s >= 0) || !(t.step_s >= 0) || !(t.vae_s >= 0)) return Timings{};
    return t;
}
void write_timings(const std::filesystem::path& p, const Timings& t) {
    std::ofstream f(p);
    f << t.setup_s << " " << t.step_s << " " << t.vae_s << "\n";
    f.flush();
    if (!f) throw std::runtime_error("write " + p.string());
}
// Fast 24L decoder (whole clip, one GPU), conservative: weights load + ~0.25 ms per latent token (x 2 margin).
double fast_vae_model_s(int T, int H, int W) { return 30.0 + 2.5e-4 * static_cast<double>(T) * H * W * 2.0; }
// Conservative per-forward estimate when nothing was measured yet (fit of the measured 1-node dense forwards: 3.9 s at
// S = 3226, 107 s at S = 37734; x 1.3 margin; two nodes ~ half).
double model_step_s(std::uint32_t S, int world) {
    const double s = static_cast<double>(S);
    return (9.2e-4 * s + 5.07e-8 * s * s) * 1.3 / static_cast<double>(world) + 1.0;
}

// Minimal PNG writer (RGB8, zlib stream of stored deflate blocks: no compression library needed).
std::uint32_t crc32_png(const unsigned char* p, std::size_t n, std::uint32_t c = 0xffffffffU) {
    static std::uint32_t tab[256];
    static bool init = false;
    if (!init) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t v = i;
            for (int k = 0; k < 8; ++k) v = (v & 1U) ? 0xedb88320U ^ (v >> 1U) : v >> 1U;
            tab[i] = v;
        }
        init = true;
    }
    for (std::size_t i = 0; i < n; ++i) c = tab[(c ^ p[i]) & 0xffU] ^ (c >> 8U);
    return c;
}
void write_png(const std::filesystem::path& path, const std::uint8_t* rgb, int w, int h) {
    std::vector<unsigned char> raw;
    raw.reserve(static_cast<std::size_t>(h) * (static_cast<std::size_t>(w) * 3 + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb + static_cast<std::size_t>(y) * w * 3, rgb + static_cast<std::size_t>(y + 1) * w * 3);
    }
    std::vector<unsigned char> z = {0x78, 0x01};
    std::uint32_t a1 = 1, a2 = 0;
    for (unsigned char c : raw) {
        a1 = (a1 + c) % 65521U;
        a2 = (a2 + a1) % 65521U;
    }
    for (std::size_t off = 0; off < raw.size() || off == 0; off += 65535) {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - off);
        z.push_back(off + n >= raw.size() ? 1 : 0);
        z.push_back(static_cast<unsigned char>(n & 0xffU));
        z.push_back(static_cast<unsigned char>(n >> 8U));
        z.push_back(static_cast<unsigned char>(~n & 0xffU));
        z.push_back(static_cast<unsigned char>((~n >> 8U) & 0xffU));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(off), raw.begin() + static_cast<std::ptrdiff_t>(off + n));
        if (raw.empty()) break;
    }
    const std::uint32_t ad = (a2 << 16U) | a1;
    for (int k = 3; k >= 0; --k) z.push_back(static_cast<unsigned char>(ad >> (8 * k)));
    std::ofstream f(path, std::ios::binary);
    auto be32 = [&](std::uint32_t v) {
        const unsigned char b[4] = {static_cast<unsigned char>(v >> 24U), static_cast<unsigned char>(v >> 16U),
            static_cast<unsigned char>(v >> 8U), static_cast<unsigned char>(v)};
        f.write(reinterpret_cast<const char*>(b), 4);
    };
    auto chunk = [&](const char* type, const std::vector<unsigned char>& d) {
        be32(static_cast<std::uint32_t>(d.size()));
        std::vector<unsigned char> td(type, type + 4);
        td.insert(td.end(), d.begin(), d.end());
        f.write(reinterpret_cast<const char*>(td.data()), static_cast<std::streamsize>(td.size()));
        be32(crc32_png(td.data(), td.size()) ^ 0xffffffffU);
    };
    f.write("\x89PNG\r\n\x1a\n", 8);
    std::vector<unsigned char> ihdr(13, 0);
    for (int k = 0; k < 4; ++k) {
        ihdr[static_cast<std::size_t>(k)] = static_cast<unsigned char>(static_cast<std::uint32_t>(w) >> (8 * (3 - k)));
        ihdr[static_cast<std::size_t>(4 + k)] = static_cast<unsigned char>(static_cast<std::uint32_t>(h) >> (8 * (3 - k)));
    }
    ihdr[8] = 8;
    ihdr[9] = 2;
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    f.flush();
    if (!f) throw std::runtime_error("write " + path.string());
}

// Preview: frames 0, F/3, 2F/3, F-1 as full-size PNGs plus one strip of them box-downscaled to <= 448 px wide each.
void write_previews(const std::filesystem::path& dir, const std::vector<std::uint8_t>& rgb, std::size_t F, int w, int h) {
    if (F == 0) return;
    std::vector<std::size_t> pick = {0, F / 3, 2 * F / 3, F - 1};
    pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
    const int ds = std::max(1, (w + 447) / 448);
    const int sw = w / ds, sh = h / ds;
    std::vector<std::uint8_t> strip(static_cast<std::size_t>(sw) * pick.size() * sh * 3);
    const std::size_t plane = static_cast<std::size_t>(w) * h * 3;
    for (std::size_t n = 0; n < pick.size(); ++n) {
        const std::uint8_t* fr = rgb.data() + pick[n] * plane;
        char name[64];
        std::snprintf(name, sizeof name, "frame_%04zu.png", pick[n]);
        write_png(dir / name, fr, w, h);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x)
                for (int c = 0; c < 3; ++c) {
                    unsigned sum = 0;
                    for (int yy = 0; yy < ds; ++yy)
                        for (int xx = 0; xx < ds; ++xx)
                            sum += fr[(static_cast<std::size_t>(y * ds + yy) * w + static_cast<std::size_t>(x * ds + xx)) * 3 + c];
                    strip[(static_cast<std::size_t>(y) * sw * pick.size() + n * sw + static_cast<std::size_t>(x)) * 3 + c] =
                        static_cast<std::uint8_t>((sum + static_cast<unsigned>(ds * ds) / 2) / static_cast<unsigned>(ds * ds));
                }
    }
    write_png(dir / "preview_strip.png", strip.data(), sw * static_cast<int>(pick.size()), sh);
}

double rms(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += static_cast<double>(x) * x;
    return v.empty() ? 0.0 : std::sqrt(s / static_cast<double>(v.size()));
}

std::uint64_t fnv64(const void* data, std::size_t n, std::uint64_t h = 0xcbf29ce484222325ULL) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
template <class T>
std::uint64_t fnv_vec(const std::vector<T>& v) {
    return fnv64(v.data(), v.size() * sizeof(T));
}

std::uint32_t f2bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
float bits2f(std::uint64_t u64) {
    const auto u = static_cast<std::uint32_t>(u64);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Per-step telemetry of a DiT forward (summed / maxed over the ranks).
struct StepInfo {
    std::uint64_t flushed = 0, bound_rows = 0;
    double score = 0, host_prep_ms = 0, final_ms = 0;
    std::string extra;
};

class DitBackend {
public:
    virtual ~DitBackend() = default;
    // -velocity (bf16 bits) of the model at `sigma` (p = ForwardPlan::build(L, sigma, {}, cfg)).
    virtual blk::StreamOutputs forward(float sigma, const blk::ForwardPlan& p, const blk::StreamInputs& in, StepInfo& si) = 0;
};

// world 1, the single-node driver (block DeviceForward: gemm + flash, weights resident after the first forward).
class SingleBackend final : public DitBackend {
public:
    // fused = false for Turbo (the fc1 branch needs the plain fc1 output)
    SingleBackend(std::unique_ptr<blk::DeviceForward> dev, blk::BlockSource& blocks, const blk::PackedLayout& L, bool fused = true,
        const blk::SparseAccel* sparse = nullptr)
        : dev_(std::move(dev)), blocks_(blocks), L_(L), lin_(L.seq_len, fused), sparse_(sparse) {}
    blk::StreamOutputs forward(float, const blk::ForwardPlan& p, const blk::StreamInputs& in, StepInfo& si) override {
        blk::DeviceForwardTaps taps;
        blk::AccelTaps acc;
        taps.accel = &acc;
        blk::StreamOutputs so = dev_->run(blocks_, L_, p, in, lin_, fa_, &taps);
        si.flushed = taps.flushed_groups;
        si.bound_rows = taps.attn_bound_rows;
        si.score = taps.attn_score_max;
        si.host_prep_ms = taps.host_prep_ms;
        si.final_ms = taps.final_ms;
        std::ostringstream e;
        if (!acc.adaln_idx.empty()) {
            e << "turbo adaLN rows";
            for (auto v : acc.adaln_idx) e << " " << v;
        }
        if (acc.fbc_action >= 0) {
            static const char* names[] = {"full+commit", "REUSE", "full-nocommit"};
            if (!e.str().empty()) e << "; ";
            e << "fbc " << (acc.fbc_cold ? "cold" : "score ") ;
            if (!acc.fbc_cold) e << acc.fbc_score << (acc.fbc_would_reuse ? " (<= thr)" : " (> thr)");
            e << " -> " << names[acc.fbc_action];
            fbc_reuses_ += acc.fbc_action == 1 ? 1 : 0;
        }
        if (sparse_) {
            const blk::SparseAccel::Info sp = sparse_->info();
            if (!e.str().empty()) e << "; ";
            e << "sparse call " << sp.model_call << ": " << sp.sparse_layers << " sparse / " << sp.dense_layers << " dense layers, mean density "
              << sp.mean_density << " (run: " << sp.run_sparse_calls << " sparse / " << sp.run_dense_calls << " dense attention calls)";
        }
        si.extra = e.str();
        return so;
    }
    int fbc_reuses() const { return fbc_reuses_; }

private:
    std::unique_ptr<blk::DeviceForward> dev_;
    blk::BlockSource& blocks_;
    const blk::PackedLayout& L_;
    blk::GemmDeviceLinear lin_;
    blk::FlashDeviceAttention fa_;
    int fbc_reuses_ = 0;
    const blk::SparseAccel* sparse_ = nullptr;
};

#ifdef H3_CLIP_DIST
// ---- two-node control protocol over the communicator's symmetric host exchange (both ranks call in lockstep)
enum Cmd : std::uint32_t { kReady = 1, kBegin = 2, kStep = 3, kVae = 4, kEnd = 5, kHello = 6 };
constexpr std::uint64_t kCtlMagic = 0x48334c4950ULL;  // "H3CLIP"
struct Ctl {
    std::uint64_t magic = kCtlMagic;
    std::uint32_t cmd = kReady, step = 0;
    std::uint64_t a[10] = {};
};
// After every distributed forward: both ranks hold the full velocity (all-gather), so the hashes must agree.
struct PeerStep {
    std::uint64_t flushed = 0, bound_rows = 0, vh_video = 0, vh_audio = 0;
    double admission = 0, wall_s = 0, blocks_s = 0, gather_s = 0;
    std::uint64_t sparse_call = 0, sparse_layers = 0, lut_entries = 0, lut_slots = 0;  // --sparse telemetry (0 when off)
};

class Link {
public:
    explicit Link(sd::Communicator& c) : c_(c) {}
    sd::Communicator& comm() { return c_; }
    Ctl ctl(const Ctl& mine) {
        Ctl peer;
        c_.exchange_host(std::as_bytes(std::span<const Ctl>(&mine, 1)), std::as_writable_bytes(std::span<Ctl>(&peer, 1)),
            tag(sd::TagKind::control, 0xC0U));
        if (peer.magic != kCtlMagic) throw std::runtime_error("h3 clip: peer control frame magic");
        return peer;
    }
    template <class T>
    T swap(const T& mine) {
        T peer{};
        c_.exchange_host(std::as_bytes(std::span<const T>(&mine, 1)), std::as_writable_bytes(std::span<T>(&peer, 1)),
            tag(sd::TagKind::control, 0xC2U));
        return peer;
    }
    // `bytes` from the sender into the receiver's buffer (the receiver sends zeros), in chunks of the capacity.
    void bcast(void* data, std::size_t bytes, bool sender) {
        const std::size_t chunk = std::min<std::size_t>(c_.max_bytes(), std::size_t{64} << 20);
        std::vector<std::byte> zeros;
        for (std::size_t off = 0; off < bytes; off += chunk) {
            const std::size_t n = std::min(chunk, bytes - off);
            std::byte* p = static_cast<std::byte*>(data) + off;
            if (sender) {
                c_.exchange_host(std::span<const std::byte>(p, n), std::span<std::byte>(), tag(sd::TagKind::raw_slots, 0xC1U));
            } else {
                zeros.assign(n, std::byte{0});
                c_.exchange_host(std::span<const std::byte>(zeros), std::span<std::byte>(p, n), tag(sd::TagKind::raw_slots, 0xC1U));
            }
        }
    }

private:
    sd::Tag tag(sd::TagKind k, std::uint32_t layer) { return sd::Tag{static_cast<std::uint32_t>(k), layer, req_++}; }
    sd::Communicator& c_;
    std::uint64_t req_ = 1;
};

std::uint64_t accel_flags(const Args& a) {
    return (a.turbo ? 1ULL : 0ULL) | (a.fbc ? 2ULL : 0ULL) | (a.fbc_observe ? 4ULL : 0ULL) | (a.sparse ? 8ULL : 0ULL) |
           (a.step_lora.empty() ? 0ULL : 16ULL) | (static_cast<std::uint64_t>(f2bits(a.step_lora.empty() ? a.strength : a.lora_strength)) << 32);
}
// The LoRA bypass holder of a run: Turbo v4 (E-grid adaLN + merged fc2) or a step-LoRA identity (run-time fc2).
bool lora_bypass(const Args& a) { return a.turbo || !a.step_lora.empty(); }
std::unique_ptr<blk::TurboV4> make_bypass(const Args& a, std::uint32_t max_rows, std::uint32_t dit_blocks, std::uint32_t refiner_blocks) {
    if (a.turbo) return std::make_unique<blk::TurboV4>(a.lora, a.egrid, a.strength, max_rows, dit_blocks, refiner_blocks);
    if (!a.step_lora.empty()) return blk::TurboV4::step_lora(a.step_lora, a.step_lora_file, a.lora_strength, max_rows, dit_blocks, refiner_blocks);
    return nullptr;
}
// The sparse identity's parameters (the same Params the 1-node adapter builds: dense layers < 2 + the extra list).
strix::h3::levers::sparse::Params sparse_params(const Args& a) {
    strix::h3::levers::sparse::Params p;
    p.tau = a.sparse_tau;
    p.dense_calls = a.sparse_dense_calls;
    p.dense_layers = 2U;
    p.dense_layer_mask = a.sparse_dense_layers.empty() ? 0 : strix::h3::levers::sparse::layer_mask(a.sparse_dense_layers);
    return p;
}
// The rank split of a run: the sparse identity runs on its own cut (128 floor(S / 256)) in both worlds.
dst::Split clip_split(const Args& a, std::uint32_t S) { return a.sparse ? dst::Split::sparse(S) : dst::Split::of(S); }
// One accelerator per rank and sampling run: Turbo / FBC / sparse as flagged, begin_run() on both ranks.
std::unique_ptr<dst::DistAccel> make_dist_accel(const Args& a, const blk::TurboV4* turbo) {
    if (!(turbo || a.fbc || a.sparse)) return nullptr;
    auto acc = std::make_unique<dst::DistAccel>(turbo, a.fbc, a.fbc_threshold, a.fbc_observe);
    if (a.sparse) acc->set_sparse(sparse_params(a));  // before the first forward
    acc->begin_run();                                 // per sampling run, on both ranks
    return acc;
}
// Sparse telemetry of one forward: model call, sparse layers, this rank's LUT entries and its (query tile x key tile x
// head) slots over the sparse layers (density = entries / slots).
void sparse_counts(const dst::DistAccel* acc, std::uint32_t S, std::uint32_t qrows, std::uint64_t& layers, std::uint64_t& entries,
    std::uint64_t& slots) {
    layers = entries = slots = 0;
    if (!acc || !acc->last().sparse) return;
    const auto& l = acc->last();
    layers = l.sparse_layers;
    for (auto e : l.lut_entries) entries += e;
    slots = static_cast<std::uint64_t>(l.sparse_layers) * ((qrows + 127U) / 128U) * 56U * ((S + 63U) / 64U);
}
std::uint64_t f64bits(double d) {
    std::uint64_t u;
    std::memcpy(&u, &d, 8);
    return u;
}
std::string accel_info(const dst::DistAccel* acc) {
    if (!acc) return "";
    const auto& l = acc->last();
    std::ostringstream e;
    if (l.sparse) e << "sparse call " << l.sparse_call << " (" << l.sparse_layers << " sparse layers)";
    if (!l.adaln_idx.empty()) {
        e << (e.str().empty() ? "" : "; ") << "turbo adaLN rows";
        for (auto v : l.adaln_idx) e << " " << v;
    }
    if (l.fbc_action >= 0) {
        static const char* names[] = {"full+commit", "REUSE", "full-nocommit"};
        e << (e.str().empty() ? "" : "; ") << "fbc " << (l.fbc_cold ? "cold" : "score ");
        if (!l.fbc_cold) e << l.fbc_score << (l.fbc_would_reuse ? " (<= thr)" : " (> thr)");
        e << " -> " << names[l.fbc_action] << " (run " << l.run_id << " call " << l.call_index << ", blocks " << l.blocks_run << ")";
    }
    return e.str();
}

void ensure_capacity(sd::Communicator& c, const dst::Split& sp) {
    const std::size_t need = sp.message_bytes();
    if (c.max_bytes() < need) c.resize_bytes(need);
}

// emulate: rank 0 hands its model to the rank 1 thread together with the GlobalWeights the model borrows (kept alive
// for the model's lifetime on whichever rank drops it last).
struct ModelSlot {
    std::shared_ptr<const blk::GlobalWeights> g;
    std::shared_ptr<dst::DeviceModel> model;
};

// The two-node schedule (dist DeviceDistRank). world 1: comm == nullptr, both key blocks local. world 2 on rank 0:
// sends STEP (sigma + stream inputs) to rank 1, both run the forward, then the per-step telemetry is exchanged and
// checked (rank 1's attention contract rows, and the gathered velocity must hash the same on both ranks).
class DistBackend final : public DitBackend {
public:
    DistBackend(const blk::GlobalWeights& g, std::shared_ptr<dst::DeviceModel> m, const blk::PackedLayout& L, const dst::Split& sp, Link* link,
        dst::DistAccel* accel = nullptr)
        : g_(g), m_(std::move(m)), L_(L), sp_(sp), link_(link), accel_(accel),
          R_(*m_, sp, dst::Role{0, link ? 2U : 1U, false}, link ? &link->comm() : nullptr) {}
    blk::StreamOutputs forward(float sigma, const blk::ForwardPlan& p, const blk::StreamInputs& in, StepInfo& si) override {
        if (link_) {
            Ctl c;
            c.cmd = kStep;
            c.step = step_;
            c.a[0] = f2bits(sigma);
            c.a[1] = in.video_x.size();
            c.a[2] = in.audio_x.size();
            link_->ctl(c);
            std::vector<std::uint16_t> buf(in.video_x);
            buf.insert(buf.end(), in.audio_x.begin(), in.audio_x.end());
            link_->bcast(buf.data(), buf.size() * 2, true);
        }
        ++step_;
        const dst::ForwardTables t = dst::ForwardTables::build(g_, L_, p);
        dst::DeviceRunOptions opt;
        opt.hashes = false;
        opt.timing = false;
        auto f = R_.forward(L_, p, in, t, m_->num_blocks(), opt, accel_);
        si.flushed = f.run.flushed_groups;
        si.bound_rows = f.run.attn_bound_rows;
        si.score = f.run.admission_bound;
        if (!link_ && f.run.attn_bound_rows) throw std::runtime_error("h3 clip: attention bound rows != 0");
        si.final_ms = f.final_s * 1e3;
        char b[256];
        std::snprintf(b, sizeof b, "embed %.2f + blocks %.2f + final %.2f + gather %.3f s", f.embed_s, f.blocks_s, f.final_s, f.gather_s);
        si.extra = b;
        const std::string ai = accel_info(accel_);
        if (!ai.empty()) si.extra += "; " + ai;
        PeerStep mine;
        // this rank's query rows: world 1 holds both key blocks (density over all rows), world 2 rank 0 the first
        if (link_) sparse_counts(accel_, sp_.S, sp_.rows(0), mine.sparse_layers, mine.lut_entries, mine.lut_slots);
        else {
            std::uint64_t l0 = 0, e0 = 0, s0 = 0, l1 = 0, e1 = 0, s1 = 0;
            sparse_counts(accel_, sp_.S, sp_.rows(0), l0, e0, s0);
            sparse_counts(accel_, sp_.S, sp_.rows(1), l1, e1, s1);
            mine.sparse_layers = l0;
            mine.lut_entries = e0;
            mine.lut_slots = s0 + s1;
        }
        if (accel_ && accel_->last().sparse) mine.sparse_call = accel_->last().sparse_call;
        PeerStep peer;
        if (link_) {
            mine.flushed = f.run.flushed_groups;
            mine.bound_rows = f.run.attn_bound_rows;
            mine.vh_video = fnv_vec(f.out.video_neg_v);
            mine.vh_audio = fnv_vec(f.out.audio_neg_v);
            mine.admission = f.run.admission_bound;
            mine.wall_s = f.wall_s;
            peer = link_->swap(mine);
            // both ranks see both counts / hashes after the swap and fail together; the communicator is dead after it
            if (mine.bound_rows || peer.bound_rows) {
                link_->comm().cancel_pending("h3 clip: attention bound rows != 0");
                throw std::runtime_error("h3 clip: attention bound rows != 0 (rank 0 " + std::to_string(mine.bound_rows) + ", rank 1 " +
                    std::to_string(peer.bound_rows) + ")");
            }
            if (peer.vh_video != mine.vh_video || peer.vh_audio != mine.vh_audio) {
                link_->comm().cancel_pending("h3 clip: gathered velocity differs between the ranks");
                throw std::runtime_error("h3 clip: the gathered velocity differs between the ranks");
            }
            if (peer.sparse_call != mine.sparse_call || peer.sparse_layers != mine.sparse_layers) {
                link_->comm().cancel_pending("h3 clip: sparse call / layers differ between the ranks");
                throw std::runtime_error("h3 clip: the sparse model call / layer count differs between the ranks");
            }
            si.flushed += peer.flushed;
            si.bound_rows += peer.bound_rows;
            si.score = std::max(si.score, peer.admission);
            std::snprintf(b, sizeof b, "; rank 1 forward %.2f s, flushed r0 %llu + r1 %llu", peer.wall_s,
                static_cast<unsigned long long>(mine.flushed), static_cast<unsigned long long>(peer.flushed));
            si.extra += b;
        }
        if (mine.sparse_layers) {
            const std::uint64_t ent = mine.lut_entries + peer.lut_entries, slots = mine.lut_slots + peer.lut_slots;
            std::snprintf(b, sizeof b, "; sparse density %.4f (LUT entries r0 %llu + r1 %llu over %llu slots)",
                slots ? static_cast<double>(ent) / static_cast<double>(slots) : 0.0, static_cast<unsigned long long>(mine.lut_entries),
                static_cast<unsigned long long>(peer.lut_entries), static_cast<unsigned long long>(slots));
            si.extra += b;
        }
        return std::move(f.out);
    }

private:
    const blk::GlobalWeights& g_;
    std::shared_ptr<dst::DeviceModel> m_;
    const blk::PackedLayout& L_;
    dst::Split sp_;
    Link* link_;
    dst::DistAccel* accel_;
    dst::DeviceDistRank R_;
    std::uint32_t step_ = 0;
};

// Rank 1: DiT weights (loaded at start in the background unless `shared` hands over rank 0's model: emulate) and the
// video VAE; serves rank 0 until END.
int serve_rank1(const Args& a, sd::Communicator& comm, ModelSlot* shared) {
    auto say = [](const char* fmt, auto... v) {
        std::printf("[rank1] ");
        std::printf(fmt, v...);
        std::printf("\n");
        std::fflush(stdout);
    };
    const double t_all = now_s();
    Link link(comm);
    blk::SafeTensors st(a.ckpt);
    std::unique_ptr<blk::LoraDiffs> diffs;  // step LoRA: full-weight deltas on the host weights before upload
    if (!a.step_lora.empty()) diffs = std::make_unique<blk::LoraDiffs>(a.step_lora_file, a.lora_strength);
    const blk::GlobalWeights g = [&]() {
        blk::GlobalWeights w = blk::load_global(st);
        if (diffs) diffs->apply_global(w);
        return w;
    }();
    std::unique_ptr<blk::SafeTensors> merged_st;
    if (a.turbo) merged_st = std::make_unique<blk::SafeTensors>(a.merged_fc2);
    std::future<std::shared_ptr<dst::DeviceModel>> pre;
    if (!shared)
        pre = std::async(std::launch::async, [&]() {
            const double t0 = now_s();
            std::shared_ptr<dst::DeviceModel> m;
            if (a.turbo) {
                blk::MergedFc2Source src(st, *merged_st, g.basis, a.strength);
                m = std::make_shared<dst::DeviceModel>(g, src, blk::count_blocks(st), true);
            } else if (diffs) {
                blk::FileBlockSource base(st, g.basis, false);
                blk::DiffBlockSource src(base, *diffs);
                m = std::make_shared<dst::DeviceModel>(g, src, blk::count_blocks(st), true);
            } else {
                blk::FileBlockSource src(st, g.basis, false);
                m = std::make_shared<dst::DeviceModel>(g, src, blk::count_blocks(st));
            }
            say("DiT weights resident: %.1f GB in %.1f s", m->resident_bytes() / 1e9, now_s() - t0);
            return m;
        });
    std::shared_ptr<const blk::GlobalWeights> gkeep;  // emulate: rank 0's weights the shared model borrows (outlives model)
    std::shared_ptr<dst::DeviceModel> model;
    std::unique_ptr<blk::PackedLayout> lay;
    std::unique_ptr<blk::TurboV4> turbo;
    std::unique_ptr<dst::DistAccel> acc;
    std::unique_ptr<dst::DeviceDistRank> R;
    dst::Split r1_split{};
    blk::ForwardPlanConfig pcfg;
    std::vector<std::uint16_t> text;
    Ctl reply;
    bool hello = false;
    for (;;) {
        const Ctl c = link.ctl(reply);
        reply = Ctl{};
        if (c.cmd == kHello) {
            const std::string assets = shared_assets(a) + test_hook("H3_CLIP_TEST_RANK1_SALT");
            std::array<std::uint8_t, 32> mine{};
            te::Sha256 h;
            h.update(assets.data(), assets.size());
            mine = h.finish();
            const auto peer = link.swap(mine);
            if (peer != mine) {
                comm.cancel_pending("h3 clip rank 1: assets / binary / flags differ");
                throw std::runtime_error("h3 clip rank 1: rank 0 runs a different binary, assets or accelerator flags (HELLO digest mismatch)");
            }
            hello = true;
            say("HELLO ok (same binary, DiT / VAE / Turbo assets and flags as rank 0)");
            continue;
        }
        if (!hello && c.cmd != kEnd) throw std::runtime_error("h3 clip rank 1: command before HELLO");
        if (c.cmd == kEnd) {
            if (pre.valid()) pre.wait();
            say("END after %.1f s", now_s() - t_all);
            return 0;
        }
        if (c.cmd == kBegin) {
            text.resize(c.a[7] / 2);
            link.bcast(text.data(), c.a[7], false);
            lay = std::make_unique<blk::PackedLayout>(blk::PackedLayout::build(static_cast<std::uint32_t>(c.a[0]), static_cast<std::uint32_t>(c.a[1]),
                static_cast<std::uint32_t>(c.a[2]), static_cast<std::uint32_t>(c.a[3]), static_cast<std::uint32_t>(c.a[4])));
            pcfg.shift_video = bits2f(c.a[5]);
            pcfg.shift_audio = bits2f(c.a[6]);
            if (c.a[8] != accel_flags(a) || c.a[9] != f64bits(a.fbc_threshold)) {
                comm.cancel_pending("h3 clip rank 1: accelerator flags differ from rank 0");
                throw std::runtime_error("h3 clip rank 1: --turbo / --strength / --fbc flags differ from rank 0's");
            }
            const double t0 = now_s();
            if (shared) {
                gkeep = shared->g;
                model = shared->model;
            } else {
                model = pre.get();
            }
            if (!model) throw std::runtime_error("h3 clip rank 1: no DiT model");
            const dst::Split split = clip_split(a, lay->seq_len);
            ensure_capacity(comm, split);
            R = std::make_unique<dst::DeviceDistRank>(*model, split, dst::Role{1, 2, false}, &comm);
            r1_split = split;
            if (lora_bypass(a)) {
                if (!model->fc1_plain()) throw std::runtime_error("h3 clip rank 1: the LoRA bypass needs the fc1-plain model");
                turbo = make_bypass(a, lay->seq_len, blk::count_blocks(st), 0);
            }
            acc = make_dist_accel(a, turbo.get());
            say("BEGIN S=%u text %u rows, cut %u (waited %.1f s for the weights)%s%s%s", lay->seq_len, lay->text_len, split.cut, now_s() - t0,
                turbo ? (", " + turbo->identity()).c_str() : "", a.fbc ? (a.fbc_observe ? ", fbc observe" : ", fbc") : "", a.sparse ? ", sparse" : "");
        } else if (c.cmd == kStep) {
            if (!R) throw std::runtime_error("h3 clip rank 1: STEP before BEGIN");
            const float sigma = bits2f(c.a[0]);
            const std::size_t nv = c.a[1], na = c.a[2];
            std::vector<std::uint16_t> buf(nv + na);
            link.bcast(buf.data(), buf.size() * 2, false);
            const double t0 = now_s();
            const blk::ForwardPlan p = blk::ForwardPlan::build(*lay, sigma, {}, pcfg);
            blk::StreamInputs in;
            in.T = lay->latent_t;
            in.H = lay->latent_h;
            in.W = lay->latent_w;
            in.audio_t = lay->audio_t;
            in.video_x.assign(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(nv));
            in.audio_x.assign(buf.begin() + static_cast<std::ptrdiff_t>(nv), buf.end());
            in.text = text;
            const dst::ForwardTables t = dst::ForwardTables::build(g, *lay, p);
            dst::DeviceRunOptions opt;
            opt.hashes = false;
            opt.timing = false;
            const auto f = R->forward(*lay, p, in, t, model->num_blocks(), opt, acc.get());
            PeerStep mine;
            mine.flushed = f.run.flushed_groups;
            mine.bound_rows = f.run.attn_bound_rows;
            mine.vh_video = fnv_vec(f.out.video_neg_v);
            mine.vh_audio = fnv_vec(f.out.audio_neg_v);
            mine.admission = f.run.admission_bound;
            mine.wall_s = f.wall_s;
            sparse_counts(acc.get(), r1_split.S, r1_split.rows(1), mine.sparse_layers, mine.lut_entries, mine.lut_slots);
            if (acc && acc->last().sparse) mine.sparse_call = acc->last().sparse_call;
            const PeerStep peer = link.swap(mine);
            const std::string ai = accel_info(acc.get());
            say("step %u sigma %.6f: %.2f s (embed %.2f + blocks %.2f + final %.2f + gather %.3f), flushed %llu, bound_rows %llu, vhash %016llx %016llx%s%s%s",
                c.step + 1, sigma, now_s() - t0, f.embed_s, f.blocks_s, f.final_s, f.gather_s, static_cast<unsigned long long>(mine.flushed),
                static_cast<unsigned long long>(mine.bound_rows), static_cast<unsigned long long>(mine.vh_video),
                static_cast<unsigned long long>(mine.vh_audio), peer.vh_video == mine.vh_video && peer.vh_audio == mine.vh_audio ? "" : " MISMATCH",
                ai.empty() ? "" : " | ", ai.c_str());
            if (mine.bound_rows || peer.bound_rows) {
                comm.cancel_pending("h3 clip rank 1: attention bound rows != 0");
                throw std::runtime_error("h3 clip rank 1: attention bound rows != 0");
            }
            if (peer.vh_video != mine.vh_video || peer.vh_audio != mine.vh_audio) {
                comm.cancel_pending("h3 clip rank 1: gathered velocity differs");
                throw std::runtime_error("h3 clip rank 1: the gathered velocity differs between the ranks");
            }
            if (peer.sparse_call != mine.sparse_call || peer.sparse_layers != mine.sparse_layers) {
                comm.cancel_pending("h3 clip rank 1: sparse call / layers differ");
                throw std::runtime_error("h3 clip rank 1: the sparse model call / layer count differs between the ranks");
            }
        } else if (c.cmd == kVae) {
            R.reset();
            acc.reset();
            turbo.reset();
            model.reset();
            if (pre.valid()) pre.get().reset();
            const int T = static_cast<int>(c.a[0]), H = static_cast<int>(c.a[1]), W = static_cast<int>(c.a[2]);
            const int first = static_cast<int>(c.a[3]), last = static_cast<int>(c.a[4]);
            vae::H16 z(c.a[5] / 2);
            link.bcast(z.data(), c.a[5], false);
            const double t0 = now_s();
            double send_s = 0;
            std::size_t sent = 0, pieces = 0;
            if (first < last) {
                vae::GpuDecoder vd(a.vae_video);
                const double tl = now_s() - t0;
                const vae::DecodePlan plan = vae::make_plan(T, H, W);
                const vae::H16 zp = vae::denormalize_and_pad(plan, z, vd.head().latents_mean.data(), vd.head().latents_std.data());
                // stream every piece as soon as it is ready: size, then the bytes (CRC-checked chunks of the communicator)
                vd.decode_pieces(plan, zp, vae::CallRange{first, last}, [&](vae::DecodePiece&& x) {
                    const double ts = now_s();
                    std::vector<std::uint8_t> b = vae::pack_piece(x);
                    const std::uint64_t n = b.size();
                    static_cast<void>(link.swap(n));
                    link.bcast(b.data(), b.size(), true);
                    send_s += now_s() - ts;
                    sent += b.size();
                    ++pieces;
                });
                say("VAE calls [%d, %d): load %.1f s, decode %.1f s; %zu pieces %.1f MB streamed (%.2f s in the sink)", first, last, tl,
                    now_s() - t0 - tl, pieces, sent / 1e6, send_s);
            }
            static_cast<void>(link.swap(std::uint64_t{0}));  // end of the piece stream
        } else {
            throw std::runtime_error("h3 clip rank 1: unknown command");
        }
    }
}
#endif

struct RankCtx {
#ifdef H3_CLIP_DIST
    Link* link = nullptr;                             // world 2: the control link to rank 1
    ModelSlot* slot = nullptr;                          // emulate: hands rank 0's model to the rank 1 thread
#endif
};

// --decode-only: VAE (36L or --fast-vae) + audio VAE on given final latents, outputs as a clip run (1 node). Quality
// probe of the decoders on real latents; its own identity (latent content ids + decoder).
int run_decode_only(const Args& a) {
    const double t_start = now_s();
    const Temporal tm = temporal_shape(a.length);
    const int lh = a.height / 16, lw = a.width / 16;
    const std::filesystem::path od(a.out), src(a.decode_only);
    std::filesystem::create_directories(od);
    std::ostringstream id;
    id << "decode-only video=" << content_id((src / "latent_video_f32.bin").string()) << " audio=" << content_id((src / "latent_audio_f32.bin").string())
       << "\nwidth=" << a.width << " height=" << a.height << " frames=" << tm.frames << "\nbuild=" << build_id() << "\nvae="
       << (a.fast_vae ? std::string(vae::fast::kIdentity) + " " + content_id(a.fast_vae_weights) : "36L " + content_id(a.vae_video))
       << "\nvae_audio=" << content_id(a.vae_audio) << "\n";
    std::error_code ec;
    if (std::filesystem::exists(od / "run_identity.txt", ec)) {
        const auto prev = read_file(od / "run_identity.txt");
        if (std::string(prev.begin(), prev.end()) != id.str()) throw std::runtime_error("output dir holds a different run");
    } else {
        for (const char* f : {"frames_rgb24.raw", "hashes.txt"})
            if (std::filesystem::exists(od / f, ec)) throw std::runtime_error("output dir holds outputs without an identity");
        const std::string t = id.str();
        write_bin(od / "run_identity.txt", t.data(), t.size());
    }
    const std::size_t nv = static_cast<std::size_t>(24) * tm.latent_t * lh * lw, na = static_cast<std::size_t>(32) * 2 * tm.audio_t;
    const std::vector<float> lv = read_vec<float>(src / "latent_video_f32.bin"), la = read_vec<float>(src / "latent_audio_f32.bin");
    if (lv.size() != nv || la.size() != na) throw std::runtime_error("decode-only: latent sizes do not match --width / --height / --length");
    // tests only: time spent before the admission (identity hashing / latent load stand-in) counts against the budget
    if (const std::string d = test_hook("H3_CLIP_TEST_DECODE_DELAY_S"); !d.empty())
        std::this_thread::sleep_for(std::chrono::duration<double>(std::stod(d)));
    {  // admission against what is LEFT after identity hashing + latent load: the whole stage (video decode + audio decode +
       // outputs; recorded for this out dir and route, else the route model + a fixed audio / output allowance)
        const double rec = read_timings(od / "timing.txt").vae_s;
        const double est = rec > 0 ? rec * 1.1 + 10.0
                                   : (a.fast_vae ? fast_vae_model_s(tm.latent_t, lh, lw)
                                                 : vae::make_plan(tm.latent_t, lh, lw).calls() * a.vae_s_per_call + 20.0) + 15.0;
        const double left = a.budget - (now_s() - t_start);
        if (left < est) {
            std::printf("[budget] decode stage (~%.0f s, %s) does not fit the %.0f s left: nothing done\n", est,
                a.fast_vae ? vae::fast::kIdentity : "36L", left);
            return 3;
        }
    }
    vae::H16 z(nv);
    for (std::size_t i = 0; i < nv; ++i) z[i] = vae::f2h(lv[i]);
    const double t0 = now_s();
    double vdec_s = 0, adec_ms = 0;
    std::vector<float> frames;
    if (a.fast_vae) {
        vae::fast::GpuDecoder vd(a.fast_vae_weights);
        frames = vd.decode(z, tm.latent_t, lh, lw, &vdec_s);
    } else {
        vae::GpuDecoder vd(a.vae_video);
        frames = vd.decode(z, tm.latent_t, lh, lw, &vdec_s);
    }
    const double vtot = now_s() - t0;
    std::vector<float> wave;
    {
        vae::audio::GpuDecoder ad(a.vae_audio);
        wave = ad.decode(la, tm.audio_t, &adec_ms);
    }
    const std::size_t plane = static_cast<std::size_t>(a.height) * a.width;
    const std::size_t F = frames.size() / (3 * plane);
    if (F * 3 * plane != frames.size() || F == 0) throw std::runtime_error("decode-only: decoder frame count");
    std::vector<std::uint8_t> rgb(F * plane * 3);
    for (std::size_t f = 0; f < F; ++f)
        for (std::size_t i = 0; i < plane; ++i)
            for (std::size_t c = 0; c < 3; ++c)
                rgb[(f * plane + i) * 3 + c] = static_cast<std::uint8_t>(std::clamp(frames[(c * F + f) * plane + i] * 255.0F, 0.0F, 255.0F));
    write_bin(od / "frames_rgb24.raw", rgb.data(), rgb.size());
    write_wav(od / "audio.wav", wave, 2, 32000);
    write_previews(od, rgb, F, a.width, a.height);
    char b[200];
    std::snprintf(b, sizeof b, "final latent video %016llx audio %016llx frames %016llx wave %016llx\n", static_cast<unsigned long long>(fnv_vec(lv)),
        static_cast<unsigned long long>(fnv_vec(la)), static_cast<unsigned long long>(fnv_vec(rgb)), static_cast<unsigned long long>(fnv_vec(wave)));
    append_line(od / "hashes.txt", b);
    {  // the whole stage (video + audio + outputs) for the next admission of this route
        Timings t = read_timings(od / "timing.txt");
        t.vae_s = now_s() - t0;
        write_timings(od / "timing.txt", t);
    }
    std::printf("[decode-only] %s: %zu frames %dx%d, video decode %.1f s (%.1f s incl. load), audio %.2f s\n[hash] %s",
        a.fast_vae ? vae::fast::kIdentity : "36L", F, a.width, a.height, vdec_s, vtot, adec_ms / 1e3, b);
    return 0;
}

// Rank 0 (and world 1): the whole clip.
int run_rank0(const Args& a, RankCtx& ctx) {
    {
#ifdef H3_CLIP_DIST
        bool peer_open = ctx.link != nullptr;
        auto end_peer = [&]() {
            if (!peer_open) return;
            Ctl c;
            c.cmd = kEnd;
            ctx.link->ctl(c);
            peer_open = false;
        };
#else
        auto end_peer = []() {};
#endif
        const Temporal tm = temporal_shape(a.length);
        const int lh = a.height / 16, lw = a.width / 16;
        std::filesystem::create_directories(a.out);
        std::printf("[clip] %dx%d, %d frames (requested %d) -> latent [24][%d][%d][%d], audio [32][2][%d]; %s %d steps, seed %llu, "
                    "shift %.1f / %.1f\n",
            a.width, a.height, tm.frames, a.length, tm.latent_t, lh, lw, tm.audio_t, a.sampler.c_str(), a.steps,
            static_cast<unsigned long long>(a.seed), a.shift, a.shift_audio);
        std::fflush(stdout);
        const double t_all = now_s();
        const std::filesystem::path od(a.out);
        auto left = [&]() { return a.budget - (now_s() - t_all); };
        const std::string assets = shared_assets(a);
#ifdef H3_CLIP_DIST
        if (ctx.link) {  // HELLO: rank 1 must run the same binary on the same assets and flags (else both exit)
            Ctl c;
            c.cmd = kHello;
            ctx.link->ctl(c);
            std::array<std::uint8_t, 32> mine{}, peer{};
            {
                te::Sha256 h;
                h.update(assets.data(), assets.size());
                mine = h.finish();
            }
            peer = ctx.link->swap(mine);
            if (peer != mine) {
                ctx.link->comm().cancel_pending("h3 clip: assets / binary / flags differ between the ranks");
                throw std::runtime_error("rank 1 runs a different binary, assets or accelerator flags (HELLO digest mismatch)");
            }
        }
#endif
        {
            std::ostringstream id;
            id << "prompt=" << a.prompt << "\nwidth=" << a.width << " height=" << a.height << " frames=" << tm.frames << " steps=" << a.steps
               << " sampler=" << a.sampler << " seed=" << a.seed << " shift=" << hexf(a.shift) << " shift_audio=" << hexf(a.shift_audio) << "\n";
            if (a.turbo) id << "accel=" << blk::TurboV4::kIdentity << " strength=" << hexf(a.strength) << " (FAST-class)\n";
            if (!a.step_lora.empty())
                id << "accel=" << a.step_lora << " file=" << content_id(a.step_lora_file) << " strength=" << hexf(a.lora_strength) << " (FAST-class)\n";
            if (a.fbc) id << "accel=h3-fbc-v1 threshold=" << hexf(a.fbc_threshold) << (a.fbc_observe ? " observe" : "") << "\n";
            if (a.sparse) id << "accel=" << blk::SparseAccel::kIdentity << " tau=" << hexf(a.sparse_tau) << " dense_calls=" << a.sparse_dense_calls
                             << " dense_layers=" << a.sparse_dense_layers << "\n";
            if (a.fast_vae) id << "accel=" << vae::fast::kIdentity << " weights=" << content_id(a.fast_vae_weights) << "\n";
            if (a.dump_velocity) id << "dump_velocity=1 (every step's model output in velocity/)\n";
            id << "tokenizer=" << content_id(a.tok) << "\nte_pack_manifest=" << content_id(a.te_pack + "/te-pack.json")
               << "\nvae_audio=" << content_id(a.vae_audio) << "\n" << assets;
            std::error_code ec;
            const bool have_id = std::filesystem::exists(od / "run_identity.txt", ec);
            const auto prev = read_file(od / "run_identity.txt");
            if (have_id && std::string(prev.begin(), prev.end()) != id.str())
                throw std::runtime_error("output dir holds a different run (run_identity.txt: prompt, shape, parameter bits, assets or build "
                                         "differ); use another --out");
            if (!have_id) {
                for (const char* f : {"te_hidden_f32.bin", "dit_state.bin", "latent_video_f32.bin", "latent_audio_f32.bin", "hashes.txt",
                         "timing.txt", "frames_rgb24.raw"})
                    if (std::filesystem::exists(od / f, ec))
                        throw std::runtime_error(std::string("output dir holds ") + f + " without a run identity: use an empty --out");
                const std::string t = id.str();
                write_bin(od / "run_identity.txt", t.data(), t.size());
            }
        }
        const Timings rec = read_timings(od / "timing.txt");

        // ---- 1. text encoder (skipped when te_hidden_f32.bin exists)
        double t0 = now_s();
        const auto tok = te::Tokenizer::load(a.tok);
        const te::Presented pres = te::present(*tok, a.prompt, 0, {});
        for (auto id : pres.ids)
            if (id < 0) throw std::runtime_error("prompt presentation has vision slots / embedding rows (t2va only)");
        const double tok_ms = (now_s() - t0) * 1e3;
        const std::uint32_t L = static_cast<std::uint32_t>(pres.ids.size());
        const std::size_t nv = static_cast<std::size_t>(24) * tm.latent_t * lh * lw, na = static_cast<std::size_t>(32) * 2 * tm.audio_t;
        std::error_code fec;
        auto fsize = [&](const char* f) -> std::int64_t {
            return std::filesystem::exists(od / f, fec) ? static_cast<std::int64_t>(std::filesystem::file_size(od / f, fec)) : -1;
        };
        if (fsize("te_hidden_f32.bin") >= 0 && fsize("te_hidden_f32.bin") != static_cast<std::int64_t>(L) * blk::kTextDim * 4)
            throw std::runtime_error("te_hidden_f32.bin has the wrong size for this prompt");
        const std::int64_t lv = fsize("latent_video_f32.bin"), la = fsize("latent_audio_f32.bin");
        if ((lv >= 0) != (la >= 0) || (lv >= 0 && (lv != static_cast<std::int64_t>(nv * 4) || la != static_cast<std::int64_t>(na * 4))))
            throw std::runtime_error("final latents incomplete or of the wrong size");
        const bool dit_done = lv >= 0;
        if (dit_done && a.dump_velocity) check_dump(od, static_cast<std::size_t>(a.steps), nv, na);
        // admission before any setup: the setup + the first step must fit (recorded timings, else a size model)
        const std::uint32_t S_est = blk::PackedLayout::build(L, static_cast<std::uint32_t>(tm.latent_t), static_cast<std::uint32_t>(lh),
            static_cast<std::uint32_t>(lw), static_cast<std::uint32_t>(tm.audio_t)).seq_len;
        const double step_est = rec.step_s > 0 ? rec.step_s : model_step_s(S_est, a.world);
        const double setup_est = rec.setup_s > 0 ? rec.setup_s : 45.0;
        if (!dit_done && left() < setup_est + step_est * 1.15 + 5.0) {
            std::printf("[budget] setup (~%.0f s) + one step (~%.0f s) do not fit the %.0f s left: nothing done\n", setup_est, step_est, left());
            end_peer();
            return 3;
        }
        t0 = now_s();
        std::vector<float> hidden = read_vec<float>(od / "te_hidden_f32.bin");
        te::EncodeTimings et;
        double te_load_s = 0, te_ms = 0;
        if (hidden.size() == static_cast<std::size_t>(L) * blk::kTextDim) {
            std::printf("[te] %u tokens; reused te_hidden_f32.bin\n", L);
        } else {
            {
                auto enc = te::TextEncoder::load(a.te_pack);
                te_load_s = now_s() - t0;
                t0 = now_s();
                hidden = enc->encode(pres.ids, &et);
            }
            te_ms = (now_s() - t0) * 1e3;
            std::printf("[te] %u tokens; tokenize %.1f ms, load %.1f s, encode %.1f ms (forward %.1f), hidden rms %.3f\n", L, tok_ms,
                te_load_s, te_ms, et.forward_ms, rms(hidden));
            if (hidden.size() != static_cast<std::size_t>(L) * blk::kTextDim) throw std::runtime_error("text encoder output size");
            write_bin(od / "te_hidden_f32.bin", hidden.data(), hidden.size() * 4);
        }
        std::fflush(stdout);
        const std::vector<std::uint16_t> text_bf16 = to_bf16(hidden);  // the DiT's inference dtype
        hidden.clear();
        hidden.shrink_to_fit();

        // ---- 2. DiT: refiner + sampler
        t0 = now_s();
        blk::SafeTensors st(a.ckpt);
        std::unique_ptr<blk::LoraDiffs> diffs;  // step LoRA: full-weight deltas on the host weights before any upload
        if (!a.step_lora.empty()) {
            diffs = std::make_unique<blk::LoraDiffs>(a.step_lora_file, a.lora_strength);
            std::printf("[lora] %s: %zu full-weight deltas (%zu values) at strength %.3f\n", a.step_lora.c_str(), diffs->count(), diffs->values(),
                static_cast<double>(a.lora_strength));
        }
        const auto g_owned = std::make_shared<const blk::GlobalWeights>([&]() {
            blk::GlobalWeights w = blk::load_global(st);
            if (diffs) diffs->apply_global(w);
            return w;
        }());
        const blk::GlobalWeights& g = *g_owned;
        blk::FileBlockSource base_blocks(st, g.basis, false);
        std::unique_ptr<blk::SafeTensors> merged_st;
        std::unique_ptr<blk::MergedFc2Source> merged_blocks;
        std::unique_ptr<blk::DiffBlockSource> diff_blocks;
        if (a.turbo) {
            merged_st = std::make_unique<blk::SafeTensors>(a.merged_fc2);
            merged_blocks = std::make_unique<blk::MergedFc2Source>(st, *merged_st, g.basis, a.strength);
        }
        if (diffs) diff_blocks = std::make_unique<blk::DiffBlockSource>(base_blocks, *diffs);
        blk::BlockSource& blocks = a.turbo  ? static_cast<blk::BlockSource&>(*merged_blocks)
                                   : diffs ? static_cast<blk::BlockSource&>(*diff_blocks)
                                           : static_cast<blk::BlockSource&>(base_blocks);
        const blk::PackedLayout lay = blk::PackedLayout::build(L, static_cast<std::uint32_t>(tm.latent_t), static_cast<std::uint32_t>(lh),
            static_cast<std::uint32_t>(lw), static_cast<std::uint32_t>(tm.audio_t));
        const std::uint32_t S = lay.seq_len;
        std::vector<float> final_video, final_audio;
        if (dit_done) {
            final_video = read_vec<float>(od / "latent_video_f32.bin");
            final_audio = read_vec<float>(od / "latent_audio_f32.bin");
            if (final_video.size() != nv || final_audio.size() != na) throw std::runtime_error("final latents unreadable");
        }
        double refiner_ms = 0, dit_load_s = 0, steps_s = 0;
        std::vector<double> step_ms;
        std::uint64_t flushed_total = 0;
        double attn_score = 0;
        bool stopped = false;
        std::size_t first_step = 0;
        if (dit_done) {
            std::printf("[dit] reused the final latents\n");
        } else {
            // The token refiner: once, on rank 0, on the device (the same bits for every DiT form and world).
            auto dev = std::make_unique<blk::DeviceForward>(g);
            std::unique_ptr<blk::TurboV4> turbo;
            std::unique_ptr<blk::SparseAccel> sparse;
            if (lora_bypass(a) || a.fbc || a.sparse) {
                blk::AccelOptions ao;
                if (lora_bypass(a)) {
                    const double tt = now_s();
                    turbo = make_bypass(a, std::max(S, L), blk::count_blocks(st), static_cast<std::uint32_t>(g.refiner.size()));
                    if (a.turbo)
                        std::printf("[turbo] %s strength %.2f: LoRA branches %.2f GB resident, merged fc2 %s, E-grid loaded (%.1f s)\n",
                            blk::TurboV4::kIdentity, a.strength, turbo->device_bytes() / 1e9, a.merged_fc2.c_str(), now_s() - tt);
                    else
                        std::printf("[lora] %s strength %.3f: branches %.2f GB resident (max rank %u, run-time fc2 %s), %s (%.1f s)\n",
                            turbo->identity().c_str(), static_cast<double>(a.lora_strength), turbo->device_bytes() / 1e9, turbo->max_rank(),
                            turbo->has_fc2() ? "yes" : "no", a.step_lora_file.c_str(), now_s() - tt);
                    ao.turbo = turbo.get();
                }
                ao.fbc = a.fbc;
                ao.fbc_threshold = a.fbc_threshold;
                ao.fbc_observe = a.fbc_observe;
                if (a.sparse) {
                    if (a.dit == "single") {
                        sparse = std::make_unique<blk::SparseAccel>(a.sparse_tau, a.sparse_dense_calls, 2U, a.sparse_dense_layers);
                        sparse->begin_run();
                        ao.sparse = sparse.get();
                    }
                    std::printf("[sparse] %s tau %.3f, dense calls %u, dense layers < 2 + {%s} (%s)\n", blk::SparseAccel::kIdentity, a.sparse_tau,
                        a.sparse_dense_calls, a.sparse_dense_layers.c_str(), a.dit == "single" ? "1 node, DeviceForward" : "two-node schedule");
                }
                if (a.fbc) std::printf("[fbc] h3-fbc-v1 threshold %.4f%s (anchor cleared at the start of this invocation)\n", a.fbc_threshold,
                    a.fbc_observe ? ", observe only" : "");
                dev->set_accel(ao);
                dev->reset_accel_run();
            }
            std::vector<std::uint16_t> refined;
            {
                blk::FlashDeviceAttention rfa;
                const double tr0 = now_s();
                refined = dev->refine_text(text_bf16.data(), L, rfa);
                refiner_ms = (now_s() - tr0) * 1e3;
            }
            std::printf("[refiner] L=%u %.1f ms (incl. first-call weight upload), contract %.1f, refined text %016llx\n", L, refiner_ms,
                dev->last_refiner_score(), static_cast<unsigned long long>(fnv_vec(refined)));
            std::printf("[dit] S=%u rows (text %u, audio %d, video %d), backend %s, world %d\n", S, L, 2 * tm.audio_t,
                tm.latent_t * (lh / 2) * (lw / 2), a.dit.c_str(), a.world);
            std::fflush(stdout);
#ifdef H3_CLIP_DIST
            std::unique_ptr<dst::DistAccel> dacc;  // before backend: outlives it
#endif
            std::unique_ptr<DitBackend> backend;
            if (a.dit == "single") {
                backend = std::make_unique<SingleBackend>(std::move(dev), blocks, lay, !lora_bypass(a), sparse.get());
            } else {
#ifdef H3_CLIP_DIST
                dev.reset();
                const double tm0 = now_s();
                auto model = std::make_shared<dst::DeviceModel>(g, blocks, blk::count_blocks(st), lora_bypass(a));
                std::printf("[dit] dist model: %u blocks, %.1f GB resident in %.1f s\n", model->num_blocks(), model->resident_bytes() / 1e9,
                    now_s() - tm0);
                if (ctx.slot) {
                    ctx.slot->g = g_owned;
                    ctx.slot->model = model;
                }
                if (ctx.link) {
                    Ctl c;
                    c.cmd = kBegin;
                    c.a[0] = L;
                    c.a[1] = static_cast<std::uint64_t>(tm.latent_t);
                    c.a[2] = static_cast<std::uint64_t>(lh);
                    c.a[3] = static_cast<std::uint64_t>(lw);
                    c.a[4] = static_cast<std::uint64_t>(tm.audio_t);
                    c.a[5] = f2bits(a.shift);
                    c.a[6] = f2bits(a.shift_audio);
                    c.a[7] = refined.size() * 2;
                    c.a[8] = accel_flags(a);
                    c.a[9] = f64bits(a.fbc_threshold);
                    const double tb = now_s();
                    ctx.link->ctl(c);
                    ctx.link->bcast(refined.data(), refined.size() * 2, true);
                    ensure_capacity(ctx.link->comm(), clip_split(a, S));
                    std::printf("[dit] BEGIN sent to rank 1 (%.2f s incl. waiting for its weights)\n", now_s() - tb);
                }
                dacc = make_dist_accel(a, turbo.get());
                const dst::Split split = clip_split(a, S);
                if (a.sparse) std::printf("[sparse] %s on the two-node schedule: split cut %u of S=%u (both worlds)\n", blk::SparseAccel::kIdentity, split.cut, S);
                backend = std::make_unique<DistBackend>(g, std::move(model), lay, split, ctx.link, dacc.get());
#endif
            }
            std::fflush(stdout);
            const std::vector<float> sig = blk::simple_sigmas(a.shift, static_cast<std::uint32_t>(a.steps));
            const float audio_scale = a.shift / a.shift_audio;
            blk::ForwardPlanConfig pcfg;
            pcfg.shift_video = a.shift;
            pcfg.shift_audio = a.shift_audio;
            blk::SamplerCarry carry;
            if (blk::load_carry(od / "dit_state.bin", carry, nv, na, sig)) {
                if ((a.fbc && !a.fbc_observe) || a.sparse)
                    throw std::runtime_error("a --fbc (reusing) / --sparse run cannot resume: the FBC anchor / the sparse call counter are not "
                                             "persisted; rerun from scratch with a budget for the whole clip");
                if (a.dump_velocity) check_dump(od, carry.next, nv, na);  // every finished step already dumped (the identity pins the flag)
                std::printf("[dit] resuming at step %zu/%d from dit_state.bin\n", carry.next + 1, a.steps);
            } else {
                blk::TorchCpuGenerator gen(a.seed);  // video noise, then audio noise (one generator, prepare_noise)
                const std::vector<float> noise_v = gen.randn(nv), noise_a = gen.randn(na);
                carry.x = blk::initial_state(sig[0], noise_v, noise_a);
            }
            first_step = carry.next;
            int step = static_cast<int>(carry.next);
            blk::AvModelFn model = [&](const blk::AvState& x, float sigma) {
                const double ts = now_s();
                const blk::ForwardPlan p = blk::ForwardPlan::build(lay, sigma, {}, pcfg);
                blk::StreamInputs in;
                in.T = static_cast<std::uint32_t>(tm.latent_t);
                in.H = static_cast<std::uint32_t>(lh);
                in.W = static_cast<std::uint32_t>(lw);
                in.audio_t = static_cast<std::uint32_t>(tm.audio_t);
                in.video_x = to_bf16(x.video);
                const std::vector<std::uint16_t> a_carried = to_bf16(x.audio);
                in.audio_x = blk::uncarry_audio(a_carried, p);
                in.text = refined;
                StepInfo si;
                const blk::StreamOutputs so = backend->forward(sigma, p, in, si);
                if (si.bound_rows) throw std::runtime_error("attention bound rows != 0 (contract)");
                const auto va = blk::carry_audio_velocity(a_carried, so.audio_neg_v, p, audio_scale);
                const double ms = (now_s() - ts) * 1e3;
                step_ms.push_back(ms);
                flushed_total += si.flushed;
                attn_score = std::max(attn_score, si.score);
                ++step;
                std::printf("[step %d/%d] sigma %.6f: %.1f ms%s (host prep %.1f ms, final %.1f ms, flushed %llu, contract %.0f, bound_rows %llu) "
                            "vhash %016llx %016llx%s%s\n",
                    step, a.steps, sigma, ms, step_ms.size() == 1 && a.dit == "single" ? " incl. DiT weight upload" : "", si.host_prep_ms,
                    si.final_ms, static_cast<unsigned long long>(si.flushed), si.score, static_cast<unsigned long long>(si.bound_rows),
                    static_cast<unsigned long long>(fnv_vec(so.video_neg_v)), static_cast<unsigned long long>(fnv_vec(so.audio_neg_v)),
                    si.extra.empty() ? "" : " | ", si.extra.c_str());
                std::fflush(stdout);
                {
                    char b[160];
                    std::snprintf(b, sizeof b, "step %d sigma %08x vhash %016llx %016llx\n", step, f2bits(sigma),
                        static_cast<unsigned long long>(fnv_vec(so.video_neg_v)), static_cast<unsigned long long>(fnv_vec(so.audio_neg_v)));
                    append_line(od / "hashes.txt", b);
                }
                if (a.dump_velocity) {
                    // the model output as the sampler receives it: -velocity bf16 (video [24][T][H][W], audio [32][2][T],
                    // audio before the carry), plus the carried audio velocity the sampler integrates (bf16)
                    const std::filesystem::path vd = od / "velocity";
                    std::filesystem::create_directories(vd);
                    char nb[64];
                    std::snprintf(nb, sizeof nb, "step_%02d", step);
                    write_bin(vd / (std::string(nb) + "_video_neg_v_bf16.bin"), so.video_neg_v.data(), so.video_neg_v.size() * 2);
                    write_bin(vd / (std::string(nb) + "_audio_neg_v_bf16.bin"), so.audio_neg_v.data(), so.audio_neg_v.size() * 2);
                    write_bin(vd / (std::string(nb) + "_audio_carried_v_bf16.bin"), va.data(), va.size() * 2);
                    char line[200];
                    std::snprintf(line, sizeof line, "step %d sigma %a sigma_bits %08x video [24][%d][%d][%d] audio [32][2][%d]\n", step,
                        static_cast<double>(sigma), f2bits(sigma), tm.latent_t, lh, lw, tm.audio_t);
                    std::ofstream ix(vd / "index.txt", std::ios::app);
                    ix << line;
                    ix.flush();
                    if (!ix) throw std::runtime_error("write velocity/index.txt");
                }
                return blk::AvState{widen(so.video_neg_v), widen(va)};
            };
            dit_load_s = now_s() - t0;
            const double ts0 = now_s();
            const blk::SamplerKind kind = a.sampler == "euler" ? blk::SamplerKind::euler : blk::SamplerKind::res_multistep;
            const std::size_t nsteps = sig.size() - 1;
            const double setup_s = now_s() - t_all;
            while (carry.next < nsteps) {
                // one step at a time; checkpoint after each; stop when the next step would not fit (every step, also
                // the first: last measured step, else the recorded / modelled one + 15 s for the first call's uploads)
                const double est = step_ms.empty() ? step_est * 1.15 + 15.0 : step_ms.back() * 1.15e-3 + 5.0;
                // tests only: H3_CLIP_TEST_STOP_AFTER=N stops this invocation after N steps exactly as a budget stop
                const std::string stop_after = test_hook("H3_CLIP_TEST_STOP_AFTER");
                if (left() < est || (!stop_after.empty() && step_ms.size() >= std::stoul(stop_after))) {
                    stopped = true;
                    break;
                }
                blk::sample_steps(model, carry, sig, kind, carry.next + 1);
                blk::save_carry(od / "dit_state.bin", carry);
            }
            steps_s = now_s() - ts0;
            if (!step_ms.empty()) {  // record for the next invocation's admission
                Timings t = rec;
                t.setup_s = setup_s;
                t.step_s = step_ms.size() > 1 ? step_ms.back() / 1e3 : std::max(rec.step_s, step_ms.back() / 1e3);
                write_timings(od / "timing.txt", t);
            }
            if (!stopped) {
                const blk::AvState fin = blk::finalize(carry.x, sig.back(), audio_scale);
                final_video = fin.video;
                final_audio = fin.audio;
            }
        }
#ifdef H3_CLIP_DIST
        if (ctx.slot) {
            ctx.slot->model.reset();
            ctx.slot->g.reset();
        }  // emulate: the DiT weights go with the last owner (rank 1 drops its copy on VAE / END)
#endif
        if (stopped) {
            std::printf("[dit] stopped after step %d/%d (budget); rerun the same command to continue\n",
                static_cast<int>(first_step + step_ms.size()), a.steps);
            end_peer();
            return 3;
        }
        if (!dit_done) {
            std::printf("[dit] sampling %.1f s (steps %zu..%d this run), latent rms video %.4f audio %.4f\n", steps_s, first_step + 1, a.steps,
                rms(final_video), rms(final_audio));
            write_bin(od / "latent_video_f32.bin", final_video.data(), final_video.size() * 4);
            write_bin(od / "latent_audio_f32.bin", final_audio.data(), final_audio.size() * 4);
        }
        std::fflush(stdout);
        {
            const int calls = vae::make_plan(tm.latent_t, lh, lw).calls();
            // the VAE is not resumable inside: start it only when the measured (else modelled) time of THIS route fits
            // (36L: calls split over the ranks; fast 24L: the whole clip on rank 0 alone)
            const double rec_vae = read_timings(od / "timing.txt").vae_s;
            const double est = rec_vae > 0 ? rec_vae * 1.1 + 10.0
                                           : (a.fast_vae ? fast_vae_model_s(tm.latent_t, lh, lw)
                                                         : calls * a.vae_s_per_call / static_cast<double>(a.world) + 20.0) + 15.0;
            if (left() < est) {
                std::printf("[vae] %d decoder calls (~%.0f s) do not fit the budget (%.0f s left); rerun to decode\n", calls, est, left());
                end_peer();
                return 3;
            }
        }

        // ---- 3. VAE decode (world 2: rank 1 decodes the upper temporal chunks, rank 0 the lower ones + the audio, then
        // finishes all canvases in canonical order: the same bits as one node)
        t0 = now_s();
        const double t_vae0 = t0;
        std::vector<float> frames;
        double vdec_s = 0, vload_s = 0, vgather_s = 0;
        std::vector<float> wave;
        double adec_ms = 0, aload_s = 0;
        auto decode_audio = [&]() {
            const double ta = now_s();
            vae::audio::GpuDecoder ad(a.vae_audio);
            aload_s = now_s() - ta;
            wave = ad.decode(final_audio, tm.audio_t, &adec_ms);
        };
        {
            vae::H16 z(final_video.size());
            for (std::size_t i = 0; i < z.size(); ++i) z[i] = vae::f2h(final_video[i]);  // the VAE's fp16 input
#ifdef H3_CLIP_DIST
            if (ctx.link && !a.fast_vae) {
                // calls split at n/2 + shift (shift = calls moved to rank 0; rank 0 also runs the audio VAE); rank 1
                // streams its pieces (chunk canvases / the cut chunk's tiles) while it decodes, a receive thread here
                // pushes them into the assembler next to rank 0's own pieces: the same frames as one node, any cut.
                const vae::DecodePlan plan = vae::make_plan(tm.latent_t, lh, lw);
                const vae::CallRange r0 = vae::split_calls(plan, 0, 2, a.vae_shift), r1 = vae::split_calls(plan, 1, 2, a.vae_shift);
                Ctl c;
                c.cmd = kVae;
                c.a[0] = static_cast<std::uint64_t>(tm.latent_t);
                c.a[1] = static_cast<std::uint64_t>(lh);
                c.a[2] = static_cast<std::uint64_t>(lw);
                c.a[3] = static_cast<std::uint64_t>(r1.first);
                c.a[4] = static_cast<std::uint64_t>(r1.last);
                c.a[5] = z.size() * 2;
                ctx.link->ctl(c);
                ctx.link->bcast(z.data(), z.size() * 2, true);
                vae::FrameAssembler fa(plan);
                std::exception_ptr rx_err;
                std::size_t rx_bytes = 0, rx_pieces = 0;
                double rx_last = 0;
                std::thread rx([&]() {
                    try {
                        for (;;) {
                            const std::uint64_t n = ctx.link->swap(std::uint64_t{0});
                            if (n == 0) break;
                            std::vector<std::uint8_t> b(n);
                            ctx.link->bcast(b.data(), b.size(), false);
                            fa.push(vae::unpack_piece(plan, b.data(), b.size()));
                            rx_bytes += n;
                            ++rx_pieces;
                            rx_last = now_s();
                        }
                    } catch (...) {
                        rx_err = std::current_exception();
                    }
                });
                double own_s = 0, ta0 = 0, ta = 0;
                try {
                    if (test_hook("H3_CLIP_TEST_FAIL") == "vae-rank0") throw std::runtime_error("injected failure (H3_CLIP_TEST_FAIL=vae-rank0)");
                    {
                        vae::GpuDecoder vd(a.vae_video);
                        vload_s = now_s() - t0;
                        const vae::H16 zp = vae::denormalize_and_pad(plan, z, vd.head().latents_mean.data(), vd.head().latents_std.data());
                        vd.decode_pieces(plan, zp, r0, [&](vae::DecodePiece&& x) { fa.push(std::move(x)); });
                        own_s = now_s() - t0 - vload_s;
                    }
                    ta0 = now_s();
                    decode_audio();
                    ta = now_s() - ta0;
                } catch (...) {
                    // cancel FIRST (the receive thread may be blocked in an exchange), then join: never unwind past a
                    // joinable thread; the peer fails at its next exchange
                    ctx.link->comm().cancel_pending("h3 clip rank 0: VAE stage failed");
                    rx.join();
                    throw;
                }
                rx.join();
                if (rx_err) std::rethrow_exception(rx_err);
                const double tj = now_s();
                frames = fa.take();
                vgather_s = now_s() - ta0 - ta;
                vdec_s = now_s() - t0 - vload_s - ta;
                std::printf("[vae] two nodes: %d calls, rank 0 [%d, %d) %.1f s (+ audio %.2f s), rank 1 [%d, %d): %zu pieces %.1f MB, last "
                            "piece in %.1f s after the start, wait after own decode + audio %.2f s, take %.2f s\n",
                    plan.calls(), r0.first, r0.last, own_s, ta, r1.first, r1.last, rx_pieces, rx_bytes / 1e6, rx_pieces ? rx_last - t0 : 0.0, vgather_s,
                    now_s() - tj);
            } else
#endif
            if (a.fast_vae) {  // h3-vae-fast24-v1: the whole clip in one pass on rank 0 (world 2: rank 1 only gets END)
                vae::fast::GpuDecoder vd(a.fast_vae_weights);
                vload_s = now_s() - t0;
                frames = vd.decode(z, tm.latent_t, lh, lw, &vdec_s);
                std::printf("[vae] fast 24L decoder (%s, FAST-class)\n", vae::fast::kIdentity);
            } else {
                vae::GpuDecoder vd(a.vae_video);
                vload_s = now_s() - t0;
                frames = vd.decode(z, tm.latent_t, lh, lw, &vdec_s);
            }
        }
        const std::size_t plane = static_cast<std::size_t>(a.height) * a.width;
        const std::size_t F = frames.size() / (3 * plane);
        std::printf("[vae] video: load %.1f s, decode %.1f s -> %zu frames %dx%d\n", vload_s, vdec_s, F, a.width, a.height);
        std::fflush(stdout);
        if (wave.empty()) decode_audio();
        const std::size_t samples = wave.size() / 2;
        std::printf("[vae] audio: load %.1f s, decode %.1f ms -> 2 x %zu samples (%.3f s at 32 kHz)\n", aload_s, adec_ms, samples,
            static_cast<double>(samples) / 32000.0);

        // ---- 4. dump: uint8 RGB as the reference's video save ((x * 255).clamp(0, 255).byte(): truncation)
        t0 = now_s();
        std::vector<std::uint8_t> rgb(F * plane * 3);
        for (std::size_t f = 0; f < F; ++f)
            for (std::size_t i = 0; i < plane; ++i)
                for (std::size_t c = 0; c < 3; ++c) {
                    const float v = std::clamp(frames[(c * F + f) * plane + i] * 255.0F, 0.0F, 255.0F);
                    rgb[(f * plane + i) * 3 + c] = static_cast<std::uint8_t>(v);
                }
        write_bin(std::filesystem::path(a.out) / "frames_rgb24.raw", rgb.data(), rgb.size());
        write_wav(std::filesystem::path(a.out) / "audio.wav", wave, 2, 32000);
        write_previews(od, rgb, F, a.width, a.height);
        const double dump_ms = (now_s() - t0) * 1e3;
        const double total_s = now_s() - t_all;
        double steady = 0;
        for (std::size_t i = 1; i < step_ms.size(); ++i) steady += step_ms[i];
        if (step_ms.size() > 1) steady /= static_cast<double>(step_ms.size() - 1);
        if (step_ms.empty()) step_ms.push_back(0.0);
        {
            std::ofstream m(od / "meta.json");
            m << "{\n  \"width\": " << a.width << ", \"height\": " << a.height << ", \"frames\": " << F << ", \"fps\": 24,\n"
              << "  \"rgb\": \"frames_rgb24.raw [frames][height][width][3] uint8\",\n"
              << "  \"audio\": \"audio.wav 2 ch 16-bit 32000 Hz\", \"audio_samples\": " << samples << ",\n"
              << "  \"latent_video\": [24, " << tm.latent_t << ", " << lh << ", " << lw << "], \"latent_audio\": [32, 2, " << tm.audio_t << "],\n"
              << "  \"tokens\": " << L << ", \"seq_len\": " << S << ", \"steps\": " << a.steps << ", \"sampler\": \"" << a.sampler
              << "\", \"seed\": " << a.seed << ", \"shift\": [" << a.shift << ", " << a.shift_audio << "],\n"
              << "  \"timings_note\": \"this invocation only (a resumed run reports its last part)\",\n"
              << "  \"timings_s\": {\"te_load\": " << te_load_s << ", \"te_encode\": " << te_ms / 1e3 << ", \"refiner\": " << refiner_ms / 1e3
              << ", \"dit_setup\": " << dit_load_s << ", \"sampling\": " << steps_s << ", \"step_first\": " << step_ms.front() / 1e3
              << ", \"step_steady\": " << steady / 1e3 << ", \"vae_video_load\": " << vload_s << ", \"vae_video\": " << vdec_s
              << ", \"vae_audio_load\": " << aload_s << ", \"vae_audio\": " << adec_ms / 1e3 << ", \"dump\": " << dump_ms / 1e3
              << ", \"total\": " << total_s << "},\n"
              << "  \"flushed_groups_total\": " << flushed_total << ", \"attn_contract_max\": " << attn_score << "\n}\n";
            m.flush();
            if (!m) throw std::runtime_error("write meta.json");
        }
        std::printf("[clip] done in %.1f s: te %.1f s (+load %.1f), refiner %.2f s, sampling %.1f s (first step %.1f s, steady %.2f s/step), "
                    "video VAE %.1f s (+load %.1f), audio VAE %.2f s, dump %.2f s -> %s\n",
            total_s, te_ms / 1e3, te_load_s, refiner_ms / 1e3, steps_s, step_ms.front() / 1e3, steady / 1e3, vdec_s, vload_s, adec_ms / 1e3,
            dump_ms / 1e3, a.out.c_str());
        {
            char b[200];
            std::snprintf(b, sizeof b, "final latent video %016llx audio %016llx frames %016llx wave %016llx\n",
                static_cast<unsigned long long>(fnv_vec(final_video)), static_cast<unsigned long long>(fnv_vec(final_audio)),
                static_cast<unsigned long long>(fnv_vec(rgb)), static_cast<unsigned long long>(fnv_vec(wave)));
            std::printf("[hash] %s", b);
            append_line(od / "hashes.txt", b);
        }
        {  // the whole VAE stage (video + audio + outputs) for the next admission of this route
            Timings t = read_timings(od / "timing.txt");
            t.vae_s = now_s() - t_vae0;
            write_timings(od / "timing.txt", t);
        }
        end_peer();  // END only after every output is written: rank 1's success then means a complete clip
        return 0;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args a = parse(argc, argv);
        if (!a.decode_only.empty()) return run_decode_only(a);
        if (a.world == 1) {
            RankCtx ctx;
            return run_rank0(a, ctx);
        }
#ifdef H3_CLIP_DIST
        const sd::Group group0{0, 2, 0x4833U, kCtlMagic}, group1{1, 2, 0x4833U, kCtlMagic};
        const std::size_t initial_bytes = std::size_t{64} << 20;  // grown to the K/V message size at BEGIN
        if (a.transport == "emulate") {
            // both ranks as threads of this process on one GPU (memory transport, same communicator code)
            auto [tp0, tp1] = sd::memory_pair();
            ModelSlot slot;
            int rc1 = 0;
            std::exception_ptr err1;
            std::thread th([&, t = std::move(tp1)]() mutable {
                try {
                    sd::Communicator c1(group1, std::move(t), initial_bytes);
                    rc1 = serve_rank1(a, c1, &slot);
                } catch (...) {
                    err1 = std::current_exception();
                }
            });
            int rc0 = 0;
            try {
                sd::Communicator c0(group0, std::move(tp0), initial_bytes);
                Link link(c0);
                RankCtx ctx;
                ctx.link = &link;
                ctx.slot = &slot;
                rc0 = run_rank0(a, ctx);
            } catch (const std::exception& e) {
                // rank 1 would wait for the next command until its timeout: fail the whole process at once
                std::fprintf(stderr, "strix-h3-clip rank 0: %s\n", e.what());
                std::fflush(stdout);
                std::fflush(stderr);
                std::_Exit(1);
            }
            th.join();
            if (err1) std::rethrow_exception(err1);
            return rc0 ? rc0 : rc1;
        }
        sd::Usb4AutoOptions o;
        o.rank = static_cast<std::uint32_t>(a.rank);
        o.timeout = std::chrono::milliseconds(240000);  // the ranks reach the rendezvous at different times
        sd::Usb4Topology topo;
        const double t0 = now_s();
        auto t = sd::usb4_host_auto(o, &topo);
        std::printf("[rank%d] transport %s (%.1f s)\n%s\n", a.rank, t->name().c_str(), now_s() - t0, topo.receipt().c_str());
        std::fflush(stdout);
        sd::Communicator comm(a.rank ? group1 : group0, std::move(t), initial_bytes);
        if (a.rank == 1) return serve_rank1(a, comm, nullptr);
        Link link(comm);
        RankCtx ctx;
        ctx.link = &link;
        return run_rank0(a, ctx);
#else
        throw std::runtime_error("world 2 needs the two-node schedule (dist package) in the build");
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-h3-clip: %s\n", e.what());
        return 1;
    }
}
