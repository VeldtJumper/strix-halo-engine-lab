#include "runtime.hpp"
#include "gemma4/kernels/device.hpp"
#include "gemma4/dflash/device.hpp"
#include "gemma4/prefill/device.hpp"
#include "gemma4/exl3/device.hpp"
#include "gemma4/kernels/a8.hpp"

#include <cstdlib>
#include <stdexcept>
#include <mutex>

namespace strix::gemma4::engine {
namespace {
bool enabled(const char* name) {
    const auto* value = std::getenv(name);
    return !value || std::string(value) != "0";
}
class DeviceState final : public StreamState {
public:
    DeviceState(const ModelConfig& config, std::uint32_t admitted) : owned_(std::make_unique<gpu::DeviceStream>(config,admitted)), state_(*owned_) {}
    explicit DeviceState(std::unique_ptr<gpu::DeviceStream> stream) : owned_(std::move(stream)), state_(*owned_) {}
    std::unique_ptr<StreamState> fork(std::uint32_t admitted) const override {
        return std::make_unique<DeviceState>(state_.fork(admitted));
    }
    std::unique_ptr<KvTxn> begin() override { return state_.begin(); }
    std::uint64_t frontier() const override { return state_.frontier(); }
    std::string digest() const override { return state_.digest(); }
    bool kv_view(std::uint32_t layer, const KvTxn* transaction, KvView& out) const override {
        // "No KV on this layer"/poisoned = unavailable; a foreign transaction (std::invalid_argument) propagates.
        try { out = state_.kv_view(layer,transaction); return true; } catch (const std::runtime_error&) { return false; }
    }
private:
    std::unique_ptr<gpu::DeviceStream> owned_;
    gpu::DeviceStream& state_;
};
class DeviceBackend final : public Backend {
public:
    DeviceBackend(const ModelManifest& model, RopeTables tables, bool inspect)
        : config_(model.config), host_embedding_(bind_host_weights(model).embedding), host_head_(bind_host_weights(model).head), metadata_(strix::serve::Json::object()) {
        metadata_.set("backend","gfx1151-device"); metadata_.set("identity",model.identity);
        metadata_.set("model_hash",model.model_hash); metadata_.set("rope_hash",tables.hash);
        metadata_.set("cache_dtype",model.cache_dtype);
        auto switches = strix::serve::Json::object();
        for (const auto* name : {"STRIX_G4_SHARED_WEIGHTS","STRIX_G4_FUSION","STRIX_G4_GRAPHS",
                "STRIX_G4_NORM_GROUP","STRIX_G4_CHUNK_DECODE","STRIX_G4_NORM_SEARCH",
                "STRIX_G4_KV_POOL","STRIX_G4_OUTPUT_TILE","STRIX_G4_KV_PUBLISH",
                "STRIX_G4_LINEAR_FINITE","STRIX_G4_NORM_PREFETCH","STRIX_G4_NORM_FINITE",
                "STRIX_G4_EXL3_OUTPUT_TILE","STRIX_G4_EXL3_COLUMN_CACHE",
                "STRIX_G4_PREFILL_EXACT_MANYROW","STRIX_G4_ATTN_KV_SHARED","STRIX_G4_EXL3_FFN_FUSE"}) switches.set(name,enabled(name));
        const auto* tiling = std::getenv("STRIX_G4_OUTPUT_TILE");
        switches.set("STRIX_G4_OUTPUT_TILE",enabled("STRIX_G4_CHUNK_DECODE") && tiling && std::string(tiling) == "1");
        switches.set("STRIX_G4_HOST_PARALLEL","not used by device target or device drafter");
        const auto* depth = std::getenv("STRIX_G4_EXL3_COOP_DEPTH");
        switches.set("STRIX_G4_EXL3_COOP_DEPTH",std::string(depth && *depth ? depth : "4"));
        metadata_.set("off_switches",switches);
        // Prompt prefill: exact many-row chunks (same bytes as the eight-row
        // passes; EXL3 packs keep their identity's core order);
        // STRIX_G4_PREFILL_EXACT_MANYROW=0 keeps the passes.
        const bool many_row = enabled("STRIX_G4_PREFILL_EXACT_MANYROW");
        // EXL3 packs: STRIX_G4_EXL3_PREFILL=wmma selects the pack's qualified held-out
        // prefill identity g4-exl3-wmma-pf-v1 (FP16 panels decoded once, WMMA
        // GEMMs, canonical attention); bit-changing against the exact route.
        const auto* exl3_route = std::getenv("STRIX_G4_EXL3_PREFILL");
        if (exl3_route && *exl3_route && std::string(exl3_route) != "exact" && std::string(exl3_route) != "wmma")
            throw std::invalid_argument("STRIX_G4_EXL3_PREFILL must be exact or wmma");
        const bool exl3_wmma = many_row && exl3_route && std::string(exl3_route) == "wmma" &&
                               bind_host_weights(model).layer.front().gate.format == WeightFormat::exl3_mul1;
        // Non-EXL3 packs: STRIX_G4_PREFILL_IDENTITY=v1|v2 runs a WMMA prefill identity with canonical
        // attention: g4-wmma-pf-v1 (FP16 operands) or g4-wmma-pf-v2 (the declared hilo policy); "fast"
        // (the fast profile's value) runs v1 on Q4_K_M packs, the pack its qualification covers, and
        // prefill-exact on others; exact (or unset) keeps prefill-exact.
        const auto* identity = std::getenv("STRIX_G4_PREFILL_IDENTITY");
        const std::string route{profile::prefill_route(identity ? identity : "",model.gguf_file_type,
                                                       bind_host_weights(model).layer.front().gate.format == WeightFormat::exl3_mul1)};
        const bool wmma_route = many_row && !exl3_wmma && route != "exact";
        const bool v2 = wmma_route && route == "v2";
        metadata_.set("prefill_route",exl3_wmma ? exl3::prefill_identity : v2 ? prefill::v2_identity : wmma_route ? prefill::identity
                                       : many_row ? "many-row" : "rows8");
        // The prefill linears decide whether prefill rows equal decode rows: any WMMA route never does.
        const auto linears = exl3_wmma || wmma_route ? prefill::Linears::wmma : prefill::Linears::exact;
        // Read-only inspection must never initialize HIP or upload weights.
        if (!inspect) {
            target_ = std::make_unique<gpu::DeviceTarget>(bind_host_weights(model),tables);
            if (many_row)
                prefill_ = std::make_unique<prefill::DevicePrefill>(target_->borrow_device_weights(),tables,
                    prefill::Residency::device,false,prefill::AttentionMode::canonical,linears);
            // v1: no policy (every WMMA linear on FP16 operands), also over a STRIX_G4_PREFILL_LINEAR_POLICY.
            if (wmma_route) prefill_->set_linear_policy(v2 ? prefill::LinearPolicy::parse(prefill::v2_policy) : prefill::LinearPolicy{});
        }
        // Report the identity the device resolved (STRIX_G4_IDENTITY may select g4-a8-v1), not the static model one.
        if (target_) metadata_.set("identity",target_->stats().identity);
        else if (const auto* selected = std::getenv("STRIX_G4_IDENTITY"); selected && *selected)
            metadata_.set("identity",std::string(selected));
        const auto* resolved = metadata_.find("identity");
        prefill_matches_decode_ = linears == prefill::Linears::exact && !(resolved && resolved->as_string() == a8::arithmetic_id);
        if (prefill_ && (prefill_->stats().linears == prefill::Linears::exact) != (linears == prefill::Linears::exact))
            throw std::logic_error("prefill linears differ from the declared prefill_matches_decode");
        metadata_.set("prefill_matches_decode",prefill_matches_decode_);
    }
    std::size_t prefill_chunk_rows() const override { return prefill_ ? prefill::chunk_rows : 0; }
    // The exact many-row prefill (g4-v1 linears, or the EXL3 pack's own core order) equals the decode passes; a
    // g4-a8-v1 decode (int8 activations) or the WMMA prefill identity does not.
    bool prefill_matches_decode() const override { return prefill_matches_decode_; }
    PrefillRows prefill_rows(KvTxn& transaction, std::span<const std::uint32_t> tokens, bool hidden, bool taps) override {
        if (!prefill_) throw std::logic_error("many-row prefill is off");
        auto chunk = prefill_->forward(transaction,tokens,false,false,hidden,taps,true);
        PrefillRows out;
        out.taps = std::move(chunk.taps); out.hidden = std::move(chunk.hidden); out.tap_width = chunk.tap_width;
        out.tap_view = chunk.pinned_taps; out.hidden_view = chunk.pinned_hidden; out.levers = chunk.levers; out.host_ms = chunk.host_ms;
        return out;
    }
    bool head_logits(std::span<const float> hidden, std::vector<float>& logits) override {
        if (!target_ || hidden.size() != config_.hidden) return false;
        const auto head = target_->resident_head();
        // The decode target's head on one row: the same linear and softcap bytes.
        if (head_input_.size() < hidden.size_bytes()) head_input_.resize(hidden.size_bytes());
        if (head_output_.size() < std::size_t(config_.vocab) * sizeof(float)) head_output_.resize(std::size_t(config_.vocab) * sizeof(float));
        if (!head_failure_.size()) head_failure_.resize(sizeof(std::uint32_t));
        head_input_.upload(hidden.data(),hidden.size_bytes());
        head_failure_.zero();
        if (head.format == WeightFormat::exl3_mul1) {
            if (head_scratch_.size() < exl3::device_scratch_bytes) head_scratch_.resize(exl3::device_scratch_bytes);
            exl3::device_linear(head,head_input_.as<float>(),1,head_output_.as<float>(),head_scratch_.data(),
                head_failure_.as<std::uint32_t>());
        } else
            gpu::linear(head,head_input_.as<float>(),1,head_output_.as<float>(),true,nullptr,head_failure_.as<std::uint32_t>());
        gpu::cap_logits(head_output_.as<float>(),config_.vocab);
        gpu::synchronize();
        std::uint32_t failure = 0;
        head_failure_.download(&failure,sizeof(failure));
        if (failure) throw std::runtime_error("nonfinite head logits");
        logits.resize(config_.vocab);
        head_output_.download(logits.data(),logits.size() * sizeof(float));
        return true;
    }
    std::unique_ptr<StreamState> create_stream() override { return create_stream(config_.max_positions); }
    std::unique_ptr<StreamState> create_stream(std::uint32_t admitted) override {
        if (!target_) throw std::logic_error("inspection backend cannot execute");
        return std::make_unique<DeviceState>(config_,admitted);
    }
    ForwardResult forward_rows(std::span<const RowPlan> rows) override {
        if (!target_) throw std::logic_error("inspection backend cannot execute");
        return target_->forward_rows(rows);
    }
    LinearView resident_embedding(LinearView source) const override {
        if (!target_ || source.data != host_embedding_.data || source.inputs != host_embedding_.inputs ||
            source.outputs != host_embedding_.outputs || source.row_bytes != host_embedding_.row_bytes ||
            source.format != host_embedding_.format) return {};
        return target_->resident_embedding();
    }
    const ModelConfig& config() const override { return config_; }
    LinearView resident_head(LinearView source) const override {
        if (!host_head_.data) return resident_embedding(source);
        if (!target_ || source.data != host_head_.data || source.inputs != host_head_.inputs ||
            source.outputs != host_head_.outputs || source.row_bytes != host_head_.row_bytes || source.format != host_head_.format) return {};
        return target_->resident_head();
    }
    strix::serve::Json identity() const override {
        auto result = strix::serve::Json(metadata_.as_object());
        if (target_) result.set("resident_weight_bytes",std::uint64_t(target_->stats().resident_weight_bytes));
        return result;
    }
private:
    ModelConfig config_;
    LinearView host_embedding_;
    LinearView host_head_;
    strix::serve::Json metadata_;
    std::unique_ptr<gpu::DeviceTarget> target_;
    std::unique_ptr<prefill::DevicePrefill> prefill_;
    bool prefill_matches_decode_ = false;
    gpu::Buffer head_input_, head_output_, head_failure_, head_scratch_;
};
struct DraftAssets {
    const dflash::Weights& weights;
    dflash::RopeView rope;
    gpu::Buffer storage;
    LinearView embedding;
    gpu::Buffer head_storage;
    LinearView head;
    std::shared_ptr<const dflash::DraftModel> model;
    std::vector<std::unique_ptr<dflash::DeviceDrafter>> spare;
    std::mutex mutex;
    bool shared_model = enabled("STRIX_G4_DRAFT_MODEL"), reuse = enabled("STRIX_G4_DRAFT_REUSE");
    bool resident = false, head_resident = false, tied_head = true;
    DraftAssets(const dflash::Weights& w, dflash::RopeView r, LinearView host, Backend* backend, LinearView host_head)
        : weights(w), rope(r), embedding(host), head(host_head.data ? host_head : host),
          tied_head(!host_head.data || host_head.data == host.data) {
        if (backend && enabled("STRIX_G4_DRAFT_HEAD_SHARE")) {
            const auto view = backend->resident_embedding(host);
            if (view.data) { embedding = view; resident = true; }
            const auto h = backend->resident_head(head);
            if (h.data) { head = h; head_resident = true; }
        }
    }
    void load() {
        if (!resident && !storage.size()) {
            storage.resize(std::size_t(embedding.outputs)*embedding.row_bytes);
            try { storage.upload(embedding.data,storage.size()); }
            catch (...) { storage.resize(0); throw; }
            embedding.data = storage.as<std::uint8_t>();
        }
        if (!head_resident) {
            if (tied_head) head = embedding;
            else if (!head_storage.size()) {
                head_storage.resize(weight_storage_bytes(head));
                try { head_storage.upload(head.data, head_storage.size()); }
                catch (...) { head_storage.resize(0); throw; }
                head.data = head_storage.as<std::uint8_t>();
            }
        }
        if (shared_model && !model)
            model = std::make_shared<dflash::DraftModel>(weights,rope,embedding,head);
    }
    std::unique_ptr<dflash::DeviceDrafter> acquire() {
        std::lock_guard lock(mutex);
        load();
        if (reuse && !spare.empty()) {
            auto result = std::move(spare.back()); spare.pop_back();
            result->reset();
            return result;
        }
        if (shared_model) return std::make_unique<dflash::DeviceDrafter>(model);
        return std::make_unique<dflash::DeviceDrafter>(weights,rope,embedding,head);
    }
    void release(std::unique_ptr<dflash::DeviceDrafter> session) noexcept {
        if (!reuse || !session->reusable()) return;
        try {
            std::lock_guard lock(mutex);
            if (spare.size() < 8) spare.push_back(std::move(session));
        } catch (...) {} // A cache allocation failure only drops the spare session.
    }
};
class DeviceDraft final : public DraftSession {
public:
    explicit DeviceDraft(std::shared_ptr<DraftAssets> assets)
        : assets_(std::move(assets)), drafter_(assets_->acquire()) {}
    ~DeviceDraft() override { assets_->release(std::move(drafter_)); }
    std::array<std::uint32_t,7> propose(std::uint32_t anchor, std::uint64_t position) override {
        return drafter_->draft(anchor,position);
    }
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override {
        drafter_->append(taps,positions);
    }
    void stage(std::span<const float> taps, std::span<const std::uint64_t> positions) override {
        drafter_->stage_append(taps,positions);
    }
    void publish() override { drafter_->publish_append(); }
    void discard() noexcept override { drafter_->discard_append(); }
private:
    std::shared_ptr<DraftAssets> assets_;
    std::unique_ptr<dflash::DeviceDrafter> drafter_;
};
}
std::unique_ptr<Backend> make_device_backend(const ModelManifest& model, RopeTables tables, bool inspect) {
    return std::make_unique<DeviceBackend>(model,tables,inspect);
}
DraftFactory device_drafts(const dflash::Weights& weights, dflash::RopeView rope, LinearView embedding,
    Backend* backend, bool prepare, LinearView head) {
    // Runtime prepares the shared model and one spare outside request timing.
    // Inspection/OFF performs no HIP work. All model bytes are validated/uploaded
    // once for this loaded model; subsequent leases clear only context rings.
    auto assets = std::make_shared<DraftAssets>(weights,rope,embedding,backend,head);
    if (prepare && assets->shared_model) {
        auto first = assets->acquire(); assets->release(std::move(first));
    }
    return [assets] { return std::make_unique<DeviceDraft>(assets); };
}
} // namespace strix::gemma4::engine
