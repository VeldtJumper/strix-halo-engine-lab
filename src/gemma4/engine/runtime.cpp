#include "runtime.hpp"
#include "lookup.hpp"
#include "gemma4/exl3/pack.hpp"
#include "gemma4/exl3/format.hpp"
#include "gemma4/features/features.hpp"

#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace strix::gemma4::engine {
namespace {
std::uint64_t unsigned_number(const std::string& value) {
    std::uint64_t n = 0;
    const auto [end,error] = std::from_chars(value.data(),value.data()+value.size(),n);
    if (error != std::errc{} || end != value.data()+value.size()) throw std::invalid_argument("invalid unsigned option value");
    return n;
}
class MetadataBackend final : public Backend {
public:
    MetadataBackend(std::unique_ptr<Backend> backend, strix::serve::Json metadata)
        : backend_(std::move(backend)), metadata_(std::move(metadata)) {}
    std::unique_ptr<StreamState> create_stream() override { return backend_->create_stream(); }
    std::unique_ptr<StreamState> create_stream(std::uint32_t admitted) override { return backend_->create_stream(admitted); }
    ForwardResult forward_rows(std::span<const RowPlan> rows) override { return backend_->forward_rows(rows); }
    const ModelConfig& config() const override { return backend_->config(); }
    strix::serve::Json identity() const override {
        auto result = backend_->identity();
        for (const auto& [key,value] : metadata_.as_object()) result.set(key,value);
        return result;
    }
    LinearView resident_embedding(LinearView source) const override { return backend_->resident_embedding(source); }
    LinearView resident_head(LinearView source) const override { return backend_->resident_head(source); }
    std::size_t prefill_chunk_rows() const override { return backend_->prefill_chunk_rows(); }
    bool prefill_matches_decode() const override { return backend_->prefill_matches_decode(); }
    PrefillRows prefill_rows(KvTxn& transaction, std::span<const std::uint32_t> tokens, bool hidden, bool taps) override {
        return backend_->prefill_rows(transaction,tokens,hidden,taps);
    }
    bool prefill_batch_supported() const override { return backend_->prefill_batch_supported(); }
    std::vector<PrefillRows> prefill_rows_batch(std::span<const PrefillItem> items, bool hidden, bool taps) override {
        return backend_->prefill_rows_batch(items,hidden,taps);
    }
    bool head_logits(std::span<const float> hidden, std::vector<float>& logits) override {
        return backend_->head_logits(hidden,logits);
    }
private:
    std::unique_ptr<Backend> backend_;
    strix::serve::Json metadata_;
};
std::string file_hash(const std::filesystem::path& path) { const MappedFile f(path); return hash_bytes(f.data(),f.size()); }
}
Options parse_options(int argc, char** argv) {
    Options o;
    auto selected = profile::Name::standard;
    bool dflash_given = false, draft_off = false, allow_pending = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string { if (++i == argc) throw std::invalid_argument("missing value for "+arg); return argv[i]; };
        const auto number = [&](std::uint64_t maximum) { const auto n = unsigned_number(value()); if (n > maximum) throw std::invalid_argument("option outside range: "+arg); return n; };
        if (arg == "--model") o.model = value();
        else if (arg == "--exl3-arm") o.exl3_arm = value();
        else if (arg == "--rope-tables") o.rope = value();
        else if (arg == "--tokenizer") o.tokenizer = value();
        else if (arg == "--draft") { o.draft = value(); if (o.draft == "off") { o.draft.clear(); draft_off = true; } }
        else if (arg == "--draft-rope") o.draft_rope = value();
        else if (arg == "--draft-eagle") o.draft_eagle = value();
        else if (arg == "--draft-mtp") o.draft_mtp = value();
        else if (arg == "--chat") o.chat = value();
        else if (arg == "--ids") o.ids = value();
        else if (arg == "--fixture") o.fixtures.emplace_back(value());
        else if (arg == "--prompt") o.prompt = value();
        else if (arg == "--output") o.output = value();
        else if (arg == "--backend") o.backend = value();
        else if (arg == "--bind") o.bind = value();
        else if (arg == "--port") o.port = std::uint16_t(number(65535));
        else if (arg == "--threads") o.threads = std::uint32_t(number(16));
        else if (arg == "--streams") o.streams = std::uint32_t(number(8));
        else if (arg == "--max-tokens" || arg == "--generate") o.request.max_tokens = std::size_t(number(65536));
        else if (arg == "--temperature") o.request.sampling.temperature = std::stof(value());
        else if (arg == "--top-p") o.request.sampling.top_p = std::stof(value());
        else if (arg == "--top-k") o.request.sampling.top_k = std::uint32_t(number(UINT32_MAX));
        else if (arg == "--seed") o.request.sampling.seed = number(UINT64_MAX);
        else if (arg == "--draft-k") o.request.draft_k = std::uint32_t(number(7));
        else if (arg == "--ignore-eos") o.request.stop_ids.clear();
        else if (arg == "--profile") selected = profile::parse(value());
        else if (arg == "--allow-pending") allow_pending = true; // benchmarking only: receipts say unqualified
        else if (arg == "--dflash") {
            dflash_given = true;
            const auto v = value();
            if (v != "on" && v != "off" && v != "auto") throw std::invalid_argument("dflash must be on/off/auto");
            o.request.dflash = v == "on" ? DraftMode::on : v == "off" ? DraftMode::off : DraftMode::automatic;
        }
        else if (arg == "--print-effective-config") o.inspect = true;
        else if (arg == "--help") o.help = true;
        else if (!arg.starts_with("--") && o.model.empty()) o.model = arg;
        else throw std::invalid_argument("unknown option: "+arg);
    }
    if (o.help) return o;
    o.request.sampling.validate();
    if (!o.threads || !o.streams || !o.request.max_tokens || o.model.empty() || o.rope.empty()) throw std::invalid_argument("model, --rope-tables and positive limits required; use --help");
    if (o.output != "json" && o.output != "ids" && o.output != "text") throw std::invalid_argument("output must be json/ids/text");
    if (o.backend != "host" && o.backend != "device") throw std::invalid_argument("backend must be host/device");
    // Explicit options win over the profile; `--draft off` wins over every drafter path.
    profile::Inputs inputs;
    inputs.allow_pending = allow_pending;
    if (!o.exl3_arm.empty()) inputs.cli.emplace_back("exl3_route",o.exl3_arm.generic_string());
    for (const auto* path : {&o.draft, &o.draft_eagle, &o.draft_mtp}) if (!path->empty()) inputs.cli.emplace_back("draft",path->generic_string());
    if (draft_off) inputs.cli.emplace_back("draft","off");
    if (dflash_given) inputs.cli.emplace_back("draft_mode",o.request.dflash == DraftMode::on ? "on" : o.request.dflash == DraftMode::off ? "off" : "auto");
    // Drafter discovery: <model dir>/drafter-mtp.
    std::error_code error;
    const auto drafter = (std::filesystem::is_directory(o.model,error) ? o.model : o.model.parent_path()) / "drafter-mtp";
    if (std::filesystem::is_directory(drafter,error)) inputs.assets.emplace_back("draft",drafter.generic_string());
    o.profile = profile::resolve(selected,profile::App::engine,inputs);
    profile::apply(o.profile);
    if (const auto* draft = o.profile.find("draft")) {
        if (draft->status == profile::Status::off) { o.draft.clear(); o.draft_eagle.clear(); o.draft_mtp.clear(); }
        else if (o.draft.empty() && o.draft_eagle.empty() && o.draft_mtp.empty() &&
                 (draft->status == profile::Status::set || draft->status == profile::Status::explicit_env)) o.draft_mtp = draft->value;
    }
    return o;
}
std::string read_text(const std::filesystem::path& path) {
    if (path == "-") return {std::istreambuf_iterator<char>(std::cin),std::istreambuf_iterator<char>()};
    std::ifstream f(path,std::ios::binary);
    if (!f) throw std::runtime_error("cannot read input file");
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
Runtime::Runtime(const Options& o) {
    // EXL3 core identity: fixed once for this session from the pack and --streams
    // (STRIX_G4_EXL3_IDENTITY forces it); recorded below and in the model identity.
    if (!o.exl3_arm.empty()) (void)exl3::configure_identity(exl3::pack_arm(o.exl3_arm), o.streams);
    manifest = !o.exl3_arm.empty() ? exl3::load_arm(o.exl3_arm, o.model) :
        o.model.extension() == ".gguf" ? load_gguf_model(o.model) : load_safetensors_model(o.model);
    rope = load_rope_tables(o.rope); manifest.rope_hash = rope.view.hash;
    if (o.backend == "host") backend = std::make_unique<HostBackend>(manifest,rope.view,o.threads);
    else {
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
        backend = make_device_backend(manifest,rope.view,o.inspect);
#else
        throw std::invalid_argument("device backend is not built");
#endif
    }
    auto metadata = strix::serve::Json::object();
    if (!o.exl3_arm.empty()) {
        metadata.set("exl3_identity", exl3::active_identity());
        metadata.set("exl3_identity_selection", exl3::forced_version() ? "forced" : "auto");
        metadata.set("exl3_max_streams", std::uint64_t(o.streams));
    }
    if (!o.tokenizer.empty()) {
        tokenizer = gemma4::serve::Tokenizer::load(o.tokenizer);
        metadata.set("tokenizer_hash",file_hash(o.tokenizer));
        metadata.set("template_identity","gemma4-native-template-v1");
        const auto* setting = std::getenv("STRIX_G4_TOKENIZER_HEAP");
        const auto identity = backend->identity();
        auto switches = identity.find("off_switches") ?
            strix::serve::Json(identity.find("off_switches")->as_object()) : strix::serve::Json::object();
        switches.set("STRIX_G4_TOKENIZER_HEAP",!setting || std::string(setting) != "0");
        metadata.set("off_switches",switches);
    }
    const auto* kind = std::getenv("STRIX_G4_DRAFT_KIND");
    const bool mtp_kind = !o.draft_mtp.empty() || (!o.draft.empty() && kind && std::string(kind) == "mtp");
    if (mtp_kind) {
        // MTP assistant: target sliding/global RoPE tables, committed layer 22/23 KV (A6), final hidden row (A1),
        // target token_embd. STRIX_G4_MTP_Q8=1 repacks the assistant matrices to Q8_0 (default: checkpoint BF16).
        const auto path = o.draft_mtp.empty() ? o.draft : o.draft_mtp;
        const auto* q8 = std::getenv("STRIX_G4_MTP_Q8");
        const bool q8_on = q8 && std::string(q8) == "1";
        mtp_draft = std::make_unique<eagle::mtp::Weights>(eagle::mtp::Weights::load(path, q8_on ? WeightFormat::q8_0 : WeightFormat::bf16));
        const auto weights = bind_host_weights(manifest);
        const eagle::mtp::Tables tables{rope.view.sliding, rope.view.global, rope.view.positions};
        if (o.backend == "host") drafts = eagle::mtp::host_drafts(*mtp_draft,tables,weights.embedding,manifest.config);
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
        else drafts = eagle::mtp::device_drafts(*mtp_draft,tables,weights.embedding,backend.get(),
            !o.inspect && dflash::enabled_from_environment());
#endif
        metadata.set("draft_kind","mtp-assistant");
        metadata.set("draft_backend",o.backend == "device" ? "resident-device" : "host-reference");
        metadata.set("draft_identity",eagle::mtp::arithmetic_id); metadata.set("draft_hash",mtp_draft->source);
        metadata.set("draft_matrices",q8_on ? "q8_0" : "bf16");
        metadata.set("draft_topk",std::uint64_t(mtp_draft->config.top_k)); // STRIX_G4_MTP_TOPK, default 64
    }
    const bool eagle_kind = !mtp_kind && (!o.draft_eagle.empty() ||
        (!o.draft.empty() && eagle::kind_from_environment() == eagle::Kind::eagle));
    if (eagle_kind) {
        // e4b-eagle: target-feature AR drafter; RoPE = the target's own sliding tables, embedding = target table.
        if (!o.draft.empty() && !o.draft_eagle.empty()) throw std::invalid_argument("use either --draft or --draft-eagle");
        const auto path = o.draft_eagle.empty() ? o.draft : o.draft_eagle;
        eagle_draft = std::make_unique<eagle::Weights>(eagle::Weights::load(path));
        if (eagle_draft->config.target_hidden != manifest.config.hidden || eagle_draft->config.target_vocab != manifest.config.vocab)
            throw std::invalid_argument("e4b-eagle drafter does not match the target width/vocabulary");
        const auto weights = bind_host_weights(manifest);
        const eagle::RopeView table{rope.view.sliding, rope.view.positions};
        if (o.backend == "host") drafts = eagle::host_drafts(*eagle_draft,table,weights.embedding);
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
        else drafts = eagle::device_drafts(*eagle_draft,table,weights.embedding,backend.get(),
            !o.inspect && dflash::enabled_from_environment() && eagle::enabled_from_environment());
#endif
        metadata.set("draft_kind","eagle");
        metadata.set("draft_backend",o.backend == "device" ? "resident-device" : "host-reference");
        metadata.set("draft_identity",eagle::arithmetic_id); metadata.set("draft_hash",file_hash(path));
    } else if (!mtp_kind && !o.draft.empty()) {
        if (o.draft_rope.empty()) throw std::invalid_argument("--draft requires frozen --draft-rope tables");
        draft = std::make_unique<dflash::DraftFile>(dflash::DraftFile::open(o.draft));
        draft_rope = dflash::RopeFile::open(o.draft_rope);
        const auto weights = bind_host_weights(manifest);
        if (o.backend == "device") {
            const auto enabled = [](const char* name) { const auto* v = std::getenv(name); return !v || std::string(v) != "0"; };
            metadata.set("draft_model_shared",enabled("STRIX_G4_DRAFT_MODEL"));
            metadata.set("draft_session_reuse",enabled("STRIX_G4_DRAFT_REUSE"));
            metadata.set("draft_head_source",enabled("STRIX_G4_DRAFT_HEAD_SHARE") && backend->resident_head(weights.head.data ? weights.head : weights.embedding).data ?
                "target-resident-original-bytes" : o.inspect ? "deferred-inspection" : "immutable-copy");
            const auto identity = backend->identity();
            auto switches = metadata.find("off_switches") ? strix::serve::Json(metadata.find("off_switches")->as_object()) :
                strix::serve::Json(identity.find("off_switches")->as_object());
            for (const auto* name : {"STRIX_G4_DRAFT_MODEL","STRIX_G4_DRAFT_REUSE","STRIX_G4_DRAFT_HEAD_SHARE"}) switches.set(name,enabled(name));
            metadata.set("off_switches",switches);
        }
        if (o.backend == "host") drafts = host_drafts(draft->weights(),draft_rope.view(),weights.embedding,weights.head.data ? weights.head : weights.embedding);
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
        else drafts = device_drafts(draft->weights(),draft_rope.view(),weights.embedding,backend.get(),
            !o.inspect && dflash::enabled_from_environment(),weights.head);
#endif
        metadata.set("draft_backend",o.backend == "device" ? "resident-device" : "host-reference");
        metadata.set("draft_identity",dflash::arithmetic_id); metadata.set("draft_hash",file_hash(o.draft));
        metadata.set("draft_rope_hash",file_hash(o.draft_rope));
    }
    // Prompt lookup (STRIX_G4_LOOKUP, default auto): wraps the MTP drafter with the per-request selector, or stands
    // alone when no drafter is loaded; =0 off. DFlash and e4b-eagle sessions are left unchanged. Drafting off (--dflash off or
    // STRIX_G4_DFLASH=0) leaves it out entirely, so those receipts carry no prompt_lookup.
    const bool drafting_off = o.request.dflash == DraftMode::off || !dflash::enabled_from_environment();
    if (const auto lookup = lookup::LookupConfig::from_environment();
        lookup::wraps(lookup, bool(drafts), mtp_kind, drafting_off)) {
        drafts = lookup::hybrid_factory(std::move(drafts),lookup);
        metadata.set("prompt_lookup",lookup.json());
        if (!metadata.find("draft_kind")) metadata.set("draft_kind","prompt-lookup");
    }
    metadata.set("profile",strix::serve::parse_json_unique(o.profile.json()));
    metadata.set("qualification",o.profile.qualification());
    metadata.set("feature_identity",features::freeze(manifest,
        o.tokenizer.empty() ? std::string{} : metadata.find("tokenizer_hash")->as_string()).json());
    backend = std::make_unique<MetadataBackend>(std::move(backend),std::move(metadata));
}
} // namespace strix::gemma4::engine
