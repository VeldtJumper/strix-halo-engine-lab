#include "gemma4/engine/runtime.hpp"
#include "gemma4/engine/lookup.hpp"
#include "gemma4/prefill/prefill.hpp"
#include "gemma4/kernels/capture_guard.hpp"
#include "../gemma4_dflash/fixture.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#ifndef _WIN32
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#endif

namespace g4 = strix::gemma4;
namespace e = g4::engine;
using Json = strix::serve::Json;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ")+#x+" line "+std::to_string(__LINE__)); } while (false)
template<class F> void refuses(F f) { bool caught = false; try { f(); } catch (const std::exception&) { caught = true; } CHECK(caught); }
namespace {
// Synthetic geometry and coefficient construction follow the host row gate.
g4::ModelManifest tiny(std::uint32_t hidden = 16) {
    g4::ModelManifest m; auto& c = m.config;
    c.hidden = hidden; c.layers = 2; c.inter = 24; c.vocab = 37; c.heads = 4; c.kv_heads = 1;
    c.ple_dim = 8; c.window = 8; c.max_positions = 96;
    c.layer = {{g4::AttentionKind::sliding,8,0},{g4::AttentionKind::global,8,1}};
    std::mt19937 rng(471);
    for (const auto& [name,shape] : g4::required_tensors(c)) {
        const auto outputs = shape.size() == 2 ? std::uint32_t(shape[0]) : 1U, inputs = std::uint32_t(shape.back());
        auto values = std::make_shared<std::vector<float>>(std::size_t(outputs)*inputs);
        for (auto& x : *values) {
            x = float(int(rng()%127)-63)*0.0009765625F;
            if (name.find("norm") != std::string::npos) x += 0.5F;
            if (name.ends_with("layer_scalar")) x = 0.875F;
        }
        m.tensors.push_back({name,shape,{reinterpret_cast<const std::uint8_t*>(values->data()),outputs,inputs,inputs*4ULL,g4::WeightFormat::f32},{}});
        m.storage.push_back(values);
    }
    return m;
}
struct Tables {
    std::vector<float> sliding = std::vector<float>(96*256), global = std::vector<float>(96*512);
    Tables() { for (std::size_t i = 0; i < sliding.size(); i += 2) sliding[i] = 1; for (std::size_t i = 0; i < global.size(); i += 2) global[i] = 1; }
    g4::RopeTables view() const { return {sliding,global,96,"synthetic-identity-rotation"}; }
};
class CountBackend final : public e::Backend {
public:
    CountBackend(const g4::ModelManifest& m, g4::RopeTables t) : host(m,t,1) {}
    std::unique_ptr<e::StreamState> create_stream() override { return host.create_stream(); }
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan> rows) override {
        CHECK(!rows.empty() && rows.size() <= 8); ++calls; max_rows = std::max(max_rows,rows.size());
        if (fail) throw std::runtime_error("injected backend failure");
        for (const auto& r : rows) generations.push_back(r.generation);
        return host.forward_rows(rows);
    }
    const g4::ModelConfig& config() const override { return host.config(); }
    Json identity() const override { return host.identity(); }
    e::HostBackend host;
    std::size_t calls = 0, max_rows = 0;
    bool fail = false;
    std::vector<std::uint64_t> generations;
};
e::Result serial(e::Backend& backend, const e::Request& request) {
    auto state = backend.create_stream();
    e::Result out; out.rng_draws = request.sampling.draws;
    auto anchor = request.prompt.front();
    for (std::size_t p = 0; p < request.prompt.size()-1+request.max_tokens; ++p) {
        if (p < request.prompt.size()) anchor = request.prompt[p];
        auto tx = state->begin(); const g4::RowPlan row{0,0,p,anchor,tx.get(),0,true};
        const auto result = backend.forward_rows({&row,1}); tx->commit(1);
        if (p+1 < request.prompt.size()) continue;
        anchor = e::select_token(result.logits,request.sampling,out.rng_draws); out.ids.push_back(anchor);
        if (std::find(request.stop_ids.begin(),request.stop_ids.end(),anchor) != request.stop_ids.end()) { out.status = e::Status::stopped; break; }
    }
    if (out.status == e::Status::running) out.status = e::Status::length;
    out.frontier = state->frontier(); out.digest = state->digest(); return out;
}
struct DraftLog { std::uint64_t frontier = 0, proposals = 0; std::vector<float> taps; };
class FakeDraft final : public e::DraftSession {
public:
    FakeDraft(std::vector<std::uint32_t> gold, std::size_t base, unsigned seed, int mode, std::shared_ptr<DraftLog> log)
        : gold_(std::move(gold)), base_(base), rng_(seed), mode_(mode), log_(std::move(log)) {}
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override {
        CHECK(!positions.empty());
        for (auto p : positions) CHECK(p == log_->frontier++);
        log_->taps.insert(log_->taps.end(),taps.begin(),taps.end());
    }
    std::array<std::uint32_t,7> propose(std::uint32_t, std::uint64_t position) override {
        CHECK(position == log_->frontier); ++log_->proposals;
        std::array<std::uint32_t,7> result{};
        for (std::size_t d = 0; d < 7; ++d) {
            const auto index = std::size_t(position)-base_+d;
            const auto expected = index < gold_.size() ? gold_[index] : 0;
            result[d] = mode_ == 0 ? expected : mode_ == 1 ? (expected+1)%37 :
                mode_ == 2 ? std::uint32_t(rng_()%37) : d < std::size_t(mode_-3) ? expected : (expected+1)%37;
        }
        return result;
    }
private:
    std::vector<std::uint32_t> gold_;
    std::size_t base_;
    std::mt19937 rng_;
    int mode_;
    std::shared_ptr<DraftLog> log_;
};
e::Request request(std::size_t length = 13) {
    e::Request r; r.stop_ids.clear(); r.max_tokens = 16; r.dflash = e::DraftMode::off;
    for (std::size_t i = 0; i < length; ++i) r.prompt.push_back(std::uint32_t((i*7+2)%37));
    return r;
}
void check_equal(const e::Result& a, const e::Result& b) {
    CHECK(a.status == b.status); CHECK(a.ids == b.ids); CHECK(a.frontier == b.frontier); CHECK(a.digest == b.digest); CHECK(a.rng_draws == b.rng_draws);
}
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
void resident_formats() {
    for (const auto format : {g4::WeightFormat::bf16, g4::WeightFormat::q8_0}) {
        auto model = tiny(32); const Tables tables;
        const auto source = g4::bind_host_weights(model).embedding;
        const auto stride = g4::packed_row_bytes(format, source.inputs);
        auto packed = std::make_shared<std::vector<std::uint8_t>>(source.outputs * stride);
        for (unsigned r = 0; r < source.outputs; ++r) {
            auto* row = packed->data() + r * stride;
            if (format == g4::WeightFormat::q8_0) {
                row[0] = 0; row[1] = 0x20; // FP16 1/128, one 32-element block.
                for (unsigned i = 0; i < source.inputs; ++i) row[2+i] = std::uint8_t(int((r+i)%15)-7);
            } else for (unsigned i = 0; i < source.inputs; ++i) {
                const auto bits = g4::bf16_bits(g4::weight_at(source,r,i));
                row[2*i] = std::uint8_t(bits & 255U); row[2*i+1] = std::uint8_t(bits >> 8U);
            }
        }
        for (auto& tensor : model.tensors) if (tensor.view.data == source.data)
            tensor.view = {packed->data(),source.outputs,source.inputs,stride,format};
        model.storage.push_back(packed);
        auto device = e::make_device_backend(model,tables.view());
        const auto host_view = g4::bind_host_weights(model).embedding;
        const auto resident = device->resident_embedding(host_view);
        CHECK(resident.data && resident.format == format && resident.row_bytes == stride);
        auto shape = fixture::shape(); shape.hidden = 32;
        const auto weights = fixture::weights(shape); const fixture::Tables draft_tables(96,8);
        const auto drafts = e::device_drafts(weights,draft_tables.view(),host_view,device.get());
        e::HostBackend host(model,tables.view(),1);
        auto req = request(3); req.max_tokens = 9; req.dflash = e::DraftMode::on;
        e::Coordinator coordinator(*device,drafts);
        const auto slot = coordinator.submit(req); coordinator.run();
        check_equal(coordinator.result(slot),serial(host,req));
        CHECK(coordinator.submit(req) == slot); coordinator.run();
        check_equal(coordinator.result(slot),serial(host,req));
    }
}
void device_tests() {
    resident_formats();
    const auto model = tiny(); const Tables tables;
    e::HostBackend host(model,tables.view(),1);
    auto device = e::make_device_backend(model,tables.view());
    for (const unsigned count : {1U,4U,8U}) {
        e::Coordinator batch(*device);
        std::vector<e::Result> gold;
        for (unsigned s = 0; s < count; ++s) {
            auto req = request(s+1); req.max_tokens = 5;
            gold.push_back(serial(host,req)); batch.submit(req);
        }
        batch.run();
        for (unsigned s = 0; s < count; ++s) check_equal(batch.result(s),gold[s]);
    }
    auto req = request(3); req.max_tokens = 9;
    const auto gold = serial(host,req);
    for (unsigned k : {0U,3U,7U}) for (int mode : {0,1}) {
        auto log = std::make_shared<DraftLog>();
        e::Coordinator spec(*device,[=] { return std::make_unique<FakeDraft>(gold.ids,req.prompt.size()-1,0,mode,log); });
        req.dflash = e::DraftMode::on; req.draft_k = k;
        const auto slot = spec.submit(req); spec.run(); check_equal(spec.result(slot),gold);
    }
    // Concrete resident device sessions use C's staged context protocol in the
    // product coordinator, not a host drafter hidden behind a device target.
    auto weights = fixture::weights(); const fixture::Tables draft_tables(96,8);
    const auto target_weights = g4::bind_host_weights(model);
    const auto resident = device->resident_embedding(target_weights.embedding);
    CHECK(resident.data && resident.data != target_weights.embedding.data);
    CHECK(resident.format == target_weights.embedding.format && resident.row_bytes == target_weights.embedding.row_bytes);
    auto other = target_weights.embedding; ++other.inputs;
    CHECK(!device->resident_embedding(other).data);
    const auto drafts = e::device_drafts(weights,draft_tables.view(),target_weights.embedding,device.get());
    for (unsigned k = 0; k <= 7; ++k) {
        req.dflash = e::DraftMode::on; req.draft_k = k;
        e::Coordinator native(*device,drafts);
        const auto slot = native.submit(req); native.run(); check_equal(native.result(slot),gold);
        if (k) CHECK(native.result(slot).acceptance.attempted > 0);
    }
    for (unsigned n : {1U,3U,7U}) {
        auto stopped = req; stopped.max_tokens = n; stopped.stop_ids = {gold.ids[n-1]};
        e::Coordinator native(*device,drafts);
        const auto slot = native.submit(stopped); native.run(); check_equal(native.result(slot),serial(host,stopped));
    }
    e::Coordinator reuse(*device,drafts);
    auto automatic = req; automatic.dflash = e::DraftMode::automatic;
    for (unsigned s = 0; s < 4; ++s) reuse.submit(automatic);
    CHECK(reuse.step()); reuse.cancel(1);
    CHECK(reuse.submit(automatic) == 1); reuse.run();
    for (unsigned s = 0; s < 4; ++s) {
        check_equal(reuse.result(s),gold); CHECK(reuse.result(s).acceptance.attempted == 0);
        CHECK(reuse.result(s).multistream_auto_cycles > 0 && reuse.result(s).multistream_auto_proposals == 0);
    }
    std::cout << "device: coordinator C1/C4/C8 and verify all-accept/reject match host IDs and state bytes\n";
}
#endif
void prefill_taps_tests();
void prefill_batch_tests();
void decode_export_tests();
void capture_guard_tests();
void host_tests(bool off) {
    const auto model = tiny(); const Tables tables; CountBackend backend(model,tables.view());
    auto r = request(); const auto reference = serial(backend,r);
    e::Coordinator c1(backend); const auto slot = c1.submit(r); c1.run(); check_equal(c1.result(slot),reference);
    CHECK(backend.max_rows == 8); CHECK(c1.receipt(slot).find("decode_identity")->as_string() == "g4-v1");
    e::Coordinator inspected(backend); const auto inspected_slot = inspected.submit(r);
    CHECK(inspected.step());
    auto prefix = request(9); prefix.max_tokens = 0;
    CHECK(inspected.receipt(inspected_slot).find("state_digest")->as_string() == serial(backend,prefix).digest);
    inspected.cancel(inspected_slot);
    for (unsigned count = 1; count <= 8; ++count) {
        e::Coordinator batch(backend); std::vector<e::Result> gold;
        for (unsigned s = 0; s < count; ++s) { const auto req = request(s*3+1); gold.push_back(serial(backend,req)); CHECK(batch.submit(req) == s); }
        while (batch.result(0).status == e::Status::running) {
            const auto before = backend.calls; CHECK(batch.step()); CHECK(backend.calls == before+1);
        }
        batch.run(); for (unsigned s = 0; s < count; ++s) check_equal(batch.result(s),gold[s]);
    }
    e::Coordinator reuse(backend); std::vector<std::size_t> slots;
    for (unsigned s = 0; s < 4; ++s) slots.push_back(reuse.submit(request(4+s*3)));
    CHECK(reuse.step()); reuse.cancel(slots[1]); const auto calls = backend.calls;
    const auto reused = reuse.submit(request(2)); CHECK(reused == slots[1]);
    reuse.run(); CHECK(backend.calls > calls);
    check_equal(reuse.result(reused),serial(backend,request(2)));
    for (unsigned s : {0U,2U,3U}) check_equal(reuse.result(s),serial(backend,request(4+s*3)));
    CHECK(*std::max_element(backend.generations.begin(),backend.generations.end()) >= 5);
    for (unsigned seed = 0; seed < 40; ++seed) for (unsigned k = 0; k <= 7; ++k) {
        const auto mode = seed < 10 ? int(seed) : 2;
        auto log = std::make_shared<DraftLog>();
        e::DraftFactory factory = [=] { return std::make_unique<FakeDraft>(reference.ids,r.prompt.size()-1,seed,mode,log); };
        auto req = r; req.dflash = e::DraftMode::on; req.draft_k = k;
        e::Coordinator spec(backend,factory); const auto s = spec.submit(req); spec.run(); check_equal(spec.result(s),reference);
        if (off || !k) CHECK(log->proposals == 0);
        else { CHECK(log->proposals > 0); CHECK(log->frontier == reference.frontier); CHECK(spec.result(s).acceptance.attempted > 0); }
        if (!off && k && mode == 0) CHECK(spec.result(s).acceptance.accepted > 0);
        if (!off && mode == 1) CHECK(spec.result(s).acceptance.accepted == 0);
    }
    // Explicit speculation across peers shares the same pass and row budget.
    std::size_t next_draft = 0;
    std::vector<e::Request> requests;
    std::vector<e::Result> greedy;
    for (unsigned s = 0; s < 4; ++s) { requests.push_back(request(2+s*3)); greedy.push_back(serial(backend,requests.back())); }
    e::Coordinator spec_batch(backend,[&] {
        const auto s = next_draft++;
        return std::make_unique<FakeDraft>(greedy[s].ids,requests[s].prompt.size()-1,0,0,std::make_shared<DraftLog>());
    });
    for (auto req : requests) { req.dflash = e::DraftMode::on; spec_batch.submit(req); }
    spec_batch.run(); for (unsigned s = 0; s < 4; ++s) check_equal(spec_batch.result(s),greedy[s]);
    // Budget and EOS truncate at every location inside an accepted verify window.
    for (std::size_t budget = 1; budget <= 9; ++budget) for (unsigned mode = 0; mode < 2; ++mode) {
        auto req = r; req.max_tokens = budget; req.dflash = e::DraftMode::on;
        if (mode) req.stop_ids = {reference.ids[budget-1]};
        auto log = std::make_shared<DraftLog>();
        e::Coordinator spec(backend,[=] { return std::make_unique<FakeDraft>(reference.ids,r.prompt.size()-1,0,0,log); });
        const auto s = spec.submit(req); spec.run(); check_equal(spec.result(s),serial(backend,req));
    }
    // Exercise the concrete host session, including its embedding/head adapters,
    // full block and committed-context projection, with bounded synthetic assets.
    const auto target_weights = g4::bind_host_weights(model);
    const fixture::Tables draft_tables(96,8);
    for (unsigned seed = 0; seed < 4; ++seed) {
        auto weights = fixture::weights();
        weights.layer[0].q = fixture::matrix(32,16,seed+2);
        auto req = r; req.dflash = e::DraftMode::on;
        e::Coordinator native_draft(backend,e::host_drafts(weights,draft_tables.view(),target_weights.embedding,target_weights.embedding));
        const auto s = native_draft.submit(req); native_draft.run(); check_equal(native_draft.result(s),reference);
        if (!off) CHECK(native_draft.result(s).acceptance.attempted > 0);
    }
    // Sampling RNG is per stream and disables speculative selection.
    auto sampled = r; sampled.sampling = {0.8F,0.9F,12,123,0};
    const auto sampled_gold = serial(backend,sampled);
    auto log = std::make_shared<DraftLog>(); sampled.dflash = e::DraftMode::on;
    e::Coordinator sampling(backend,[=] { return std::make_unique<FakeDraft>(reference.ids,r.prompt.size()-1,0,0,log); });
    for (unsigned s = 0; s < 4; ++s) sampling.submit(sampled);
    sampling.run(); for (unsigned s = 0; s < 4; ++s) check_equal(sampling.result(s),sampled_gold);
    CHECK(log->proposals == 0); CHECK(sampled_gold.rng_draws == sampled.max_tokens);
    e::CostModel costs; CHECK(costs.choose(1,7) == 0);
    for (unsigned n = 0; n < 8; ++n) { costs.observe(1,0,10,1); costs.observe(1,3,20,4); costs.observe(1,7,100,8); }
    CHECK(costs.choose(1,7) == 3); CHECK(costs.choose(4,7) == 0);
    // Seed measured evidence and exercise automatic admission and its OFF fallback.
    auto measured = std::make_shared<e::CostModel>();
    const auto key = e::measurement_key(backend,r);
    for (unsigned n = 0; n < 8; ++n) { measured->observe(1,0,10,1,0,key); measured->observe(1,3,20,4,0,key); }
    CHECK(measured->choose(1,7,8192,key) == 0); CHECK(measured->choose(1,7,0,key+"different") == 0);
    auto auto_log = std::make_shared<DraftLog>();
    e::Coordinator automatic(backend,[=] { return std::make_unique<FakeDraft>(reference.ids,r.prompt.size()-1,0,0,auto_log); },measured);
    auto auto_req = r; auto_req.dflash = e::DraftMode::automatic;
    const auto auto_slot = automatic.submit(auto_req); automatic.run(); check_equal(automatic.result(auto_slot),reference);
    CHECK(off ? auto_log->proposals == 0 : auto_log->proposals > 0);
    if (automatic.receipt(auto_slot).find("dflash_environment")->as_string() == "on") {
        CHECK(automatic.receipt(auto_slot).find("dflash_admission")->as_string() == "explicit-bounded-window");
        CHECK(automatic.result(auto_slot).windows[7] > 0 && automatic.result(auto_slot).probes == 0);
    }
    refuses([&] { auto bad = r; bad.max_tokens = 100; (void)c1.submit(bad); });
    refuses([&] { auto bad = r; bad.sampling.top_p = 0; (void)c1.submit(bad); });
    std::uint64_t draws = 0;
    const float tied[] = {2,2,1}; CHECK(e::select_token(tied,{},draws) == 0); CHECK(draws == 0);
    refuses([&] { const float bad[] = {0,std::numeric_limits<float>::quiet_NaN()}; (void)e::select_token(bad,{},draws); });
    const float uniform[] = {1,1,1,1}; const unsigned known[] = {3,1,0,3,0,1,0,3};
    e::Sampling policy; policy.temperature = 1;
    for (const auto expected : known) CHECK(e::select_token(uniform,policy,draws) == expected);
    // candidate_order == the comparator sort it replaced (descending logit, low id on ties, -0 == +0), full and top-k.
    {
        std::uint64_t z = 7;
        const auto next = [&] { z = z*6364136223846793005ULL+1442695040888963407ULL; return std::uint32_t(z >> 33U); };
        for (const std::size_t n : {std::size_t(1),std::size_t(3),std::size_t(1000),std::size_t(65537),std::size_t(262144)}) {
            std::vector<float> l(n);
            for (auto& x : l) {
                const auto kind = next() % 8;
                x = kind == 0 ? 0.0F : kind == 1 ? -0.0F : kind == 2 ? float(int(next() % 5)) - 2.0F : // heavy ties
                    (float(next()) / 4294967296.0F - 0.5F) * 60.0F;
            }
            std::vector<std::uint32_t> comparator(n);
            std::iota(comparator.begin(),comparator.end(),0U);
            const auto t0 = std::chrono::steady_clock::now();
            std::sort(comparator.begin(),comparator.end(),[&](auto a, auto b) { return g4::better(l[a],a,l[b],b); });
            const auto t1 = std::chrono::steady_clock::now();
            CHECK(e::candidate_order(l) == comparator);
            const auto t2 = std::chrono::steady_clock::now();
            for (const std::uint32_t k : {1U,40U,1000U}) {
                const auto top = e::candidate_order(l,k);
                CHECK(top.size() == std::min<std::size_t>(k,n) && std::equal(top.begin(),top.end(),comparator.begin()));
            }
            if (n == 262144)
                std::cout << "candidate order over 262144 ids: comparator sort "
                          << std::chrono::duration<double,std::milli>(t1-t0).count() << " ms, keyed radix "
                          << std::chrono::duration<double,std::milli>(t2-t1).count() << " ms\n";
        }
    }
    e::Coordinator failed(backend); const auto failure_slot = failed.submit(r);
    const auto before_failure = failed.result(failure_slot).digest;
    backend.fail = true; CHECK(failed.step()); backend.fail = false;
    CHECK(failed.result(failure_slot).status == e::Status::failed); CHECK(failed.result(failure_slot).frontier == 0);
    CHECK(failed.result(failure_slot).digest == before_failure); CHECK(!failed.step());
    struct BrokenContext final : e::DraftSession {
        void append(std::span<const float>, std::span<const std::uint64_t>) override { throw std::runtime_error("injected context failure"); }
        std::array<std::uint32_t,7> propose(std::uint32_t, std::uint64_t) override { return {}; }
    };
    if (!off) {
        auto req = r; req.dflash = e::DraftMode::on;
        e::Coordinator broken(backend,[] { return std::make_unique<BrokenContext>(); });
        const auto s = broken.submit(req); CHECK(broken.step());
        CHECK(broken.result(s).status == e::Status::failed); CHECK(broken.result(s).frontier == 8);
        CHECK(!broken.step()); CHECK(broken.result(s).ids.empty());
        struct BrokenStage final : e::DraftSession {
            void append(std::span<const float>, std::span<const std::uint64_t>) override {}
            void stage(std::span<const float>, std::span<const std::uint64_t>) override { throw std::runtime_error("injected stage failure"); }
            std::array<std::uint32_t,7> propose(std::uint32_t, std::uint64_t) override { return {}; }
        };
        e::Coordinator staged(backend,[] { return std::make_unique<BrokenStage>(); });
        const auto stage_slot = staged.submit(req); CHECK(staged.step());
        CHECK(staged.result(stage_slot).status == e::Status::failed);
        CHECK(staged.result(stage_slot).frontier == 0 && staged.result(stage_slot).acceptance.cycles == 0);
    }
    e::Coordinator cancelled(backend); const auto cancelled_slot = cancelled.submit(r); const auto before_cancel = backend.calls;
    cancelled.cancel(cancelled_slot); cancelled.run(); CHECK(backend.calls == before_cancel);
    CHECK(cancelled.result(cancelled_slot).status == e::Status::cancelled);
    if (!off) { prefill_taps_tests(); prefill_batch_tests(); decode_export_tests(); capture_guard_tests(); }
    std::cout << "host: C1..C8, shared calls, unequal prompts, cancellation/reuse, DFlash windows/seeds/stops, sampling, costs and prompt taps passed\n";
}

// Many-row prompt prefill as a backend may implement it (eight-row chunks here,
// the host target's bytes). With PINNED_TAPS on, taps that the session does not
// consume are omitted and consumed taps arrive as a view of backend memory, like
// the device backend; off, every call returns taps as a vector.
class PrefillBackend final : public e::Backend {
public:
    PrefillBackend(const g4::ModelManifest& m, g4::RopeTables t) : host(m,t,1) {}
    std::unique_ptr<e::StreamState> create_stream() override { return host.create_stream(); }
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan> rows) override { return host.forward_rows(rows); }
    const g4::ModelConfig& config() const override { return host.config(); }
    Json identity() const override { return host.identity(); }
    std::size_t prefill_chunk_rows() const override { return chunk; }
    bool prefill_matches_decode() const override { return matches; }
    bool prefill_batch_supported() const override { return supported; }
    e::PrefillRows prefill_rows(g4::KvTxn& tx, std::span<const std::uint32_t> tokens, bool hidden, bool taps) override {
        requested.push_back(taps);
        std::vector<g4::RowPlan> rows;
        for (std::size_t i = 0; i < tokens.size(); ++i) rows.push_back({0,0,tx.frontier()+i,tokens[i],&tx,i,true,hidden});
        auto pass = host.forward_rows(rows);
        e::PrefillRows out; out.tap_width = pass.tap_width;
        if (hidden) out.hidden = std::move(pass.hidden);
        if (!g4::prefill::lever_enabled(g4::prefill::lever_pinned_taps)) { out.taps = std::move(pass.taps); return out; }
        out.levers = g4::prefill::lever_pinned_taps;
        if (taps) { held = std::move(pass.taps); out.tap_view = held; }
        return out;
    }
    // Batched calls as the device backend makes them: one result per stream, views valid until the next call.
    std::vector<e::PrefillRows> prefill_rows_batch(std::span<const e::PrefillItem> items, bool hidden, bool taps) override {
        ++batch_calls; batch_width = std::max(batch_width,items.size());
        std::size_t total = 0;
        for (const auto& item : items) total += item.tokens.size();
        CHECK(total <= chunk && items.size() > 1);
        held_batch.clear(); held_batch.reserve(items.size());
        std::vector<e::PrefillRows> out;
        for (const auto& item : items) {
            out.push_back(prefill_rows(*item.transaction,item.tokens,hidden,taps));
            if (!out.back().tap_view.empty()) { held_batch.push_back(held); out.back().tap_view = held_batch.back(); }
        }
        return out;
    }
    e::HostBackend host;
    std::vector<bool> requested;
    std::vector<float> held;
    std::vector<std::vector<float>> held_batch;
    std::size_t chunk = 8, batch_calls = 0, batch_width = 0;
    bool matches = false, supported = true;
};
struct StageLog { std::vector<float> taps, hidden; std::vector<std::uint64_t> positions; std::size_t empty_taps = 0; };
// Session shapes of the three drafter families: DFlash consumes taps through
// stage(); EAGLE consumes taps and final hidden rows through stage_rows(); MTP
// keeps only final hidden rows and declares wants_taps() false.
class ShapeDraft final : public e::DraftSession {
public:
    ShapeDraft(int family, std::vector<std::uint32_t> gold, std::size_t base, std::shared_ptr<StageLog> log)
        : family_(family), gold_(std::move(gold)), base_(base), log_(std::move(log)) {}
    bool wants_hidden() const override { return family_ != 0; }
    bool wants_taps() const override { return family_ != 2; }
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override {
        log_->taps.insert(log_->taps.end(),taps.begin(),taps.end());
        log_->positions.insert(log_->positions.end(),positions.begin(),positions.end());
    }
    void stage_rows(std::span<const float> taps, std::span<const float> hidden, std::span<const std::uint32_t> next,
                    std::span<const std::uint64_t> positions) override {
        if (family_ == 0) { e::DraftSession::stage_rows(taps,hidden,next,positions); return; }
        CHECK(hidden.size() == positions.size()*16);
        if (family_ == 2 && taps.empty()) ++log_->empty_taps;
        else CHECK(taps.size() == positions.size()*80);
        log_->hidden.insert(log_->hidden.end(),hidden.begin(),hidden.end());
        e::DraftSession::stage(taps,positions);
    }
    std::array<std::uint32_t,7> propose(std::uint32_t, std::uint64_t position) override {
        std::array<std::uint32_t,7> result{};
        for (std::size_t d = 0; d < 7; ++d) { const auto i = std::size_t(position)-base_+d; result[d] = i < gold_.size() ? gold_[i] : 0; }
        return result;
    }
private:
    int family_;
    std::vector<std::uint32_t> gold_;
    std::size_t base_;
    std::shared_ptr<StageLog> log_;
};
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    (void)_putenv_s(name, value ? value : "");
#else
    if (value) (void)setenv(name, value, 1); else (void)unsetenv(name);
#endif
}
// PREFILL_BATCH: streams with prompt chunks pending share one prefill call; every stream's ids, frontier and state
// digest equal the stream alone. Two short prompts decode while three long ones prefill 4-row chunks (the row budget),
// so two chunks share an 8-row call. A route whose prefill differs from decode joins only chunks of more than
// 8 rows (here: never, the host admits 8-row passes; the device test covers longer chunks).
void prefill_batch_tests() {
    const auto model = tiny(); const Tables tables;
    PrefillBackend backend(model,tables.view());
    const auto run = [&](std::size_t chunk, bool matches, std::size_t limit) {
        backend.chunk = chunk; backend.matches = matches; backend.batch_calls = backend.batch_width = 0;
        e::Coordinator c(backend); c.set_prefill_limit(limit); c.set_prefill_budget(4);
        std::vector<e::Request> requests{request(5),request(6)};
        for (unsigned s = 0; s < 3; ++s) requests.push_back(request(30+s*5));
        std::vector<e::Result> gold;
        for (std::size_t s = 0; s < requests.size(); ++s) { gold.push_back(serial(backend,requests[s])); CHECK(c.submit(requests[s]) == s); }
        c.run();
        std::uint64_t batched = 0;
        for (std::size_t s = 0; s < requests.size(); ++s) {
            check_equal(c.result(s),gold[s]);
            batched += c.result(s).prefill_batched_calls;
            if (c.result(s).prefill_batched_calls) CHECK(c.receipt(s).find("prefill_batch_streams")->as_int() >= 2);
        }
        CHECK((backend.batch_calls > 0) == (batched > 0));
        return backend.batch_calls;
    };
    const auto* off = std::getenv("STRIX_G4_PREFILL_BATCH");
    const bool batching = !off || std::string(off) != "0";  // off: the same outputs, one call per stream
    CHECK((run(8,true,1) > 0) == batching); CHECK(!batching || backend.batch_width >= 2);
    CHECK((run(8,true,0) > 0) == batching);
    CHECK(run(8,false,1) == 0);   // <= 8-row chunks on such a route never share a call
    backend.supported = false;    // a route without a batched byte record: one call per stream
    CHECK(run(8,true,1) == 0);
    backend.supported = true;
    std::cout << "host: batched prompt prefill (shared calls, exact and other-arithmetic routes) passed\n";
}
void prefill_taps_tests() {
    const auto model = tiny(); const Tables tables;
    PrefillBackend backend(model,tables.view());
    auto r = request(29); const auto reference = serial(backend,r);
    for (int family = 0; family < 3; ++family) {
        std::array<std::shared_ptr<StageLog>,2> logs;
        std::array<e::Result,2> results;
        std::array<Json,2> receipts;
        for (int on = 0; on < 2; ++on) {
            set_env("STRIX_G4_PREFILL_PINNED_TAPS", on ? nullptr : "0");
            backend.requested.clear();
            auto log = logs[on] = std::make_shared<StageLog>();
            e::DraftFactory factory = [=] { return std::make_unique<ShapeDraft>(family,reference.ids,r.prompt.size()-1,log); };
            auto req = r; req.dflash = e::DraftMode::on;
            e::Coordinator c(backend,factory); const auto s = c.submit(req); c.run();
            check_equal(c.result(s),reference);
            results[on] = c.result(s); receipts[on] = c.receipt(s);
            CHECK(!backend.requested.empty());
            for (const bool asked : backend.requested) CHECK(asked == (family != 2));
            const auto* taps = receipts[on].find("prefill_taps");
            if (!on) CHECK(!taps);
            else CHECK(taps && taps->as_string() == (family == 2 ? "skipped (no consumer)" : "pinned-async"));
        }
        // Same outputs and the same staged drafter inputs with the lever on and off.
        CHECK(results[0].ids == results[1].ids && results[0].digest == results[1].digest);
        CHECK(results[0].acceptance.accepted == results[1].acceptance.accepted);
        CHECK(logs[0]->hidden == logs[1]->hidden && logs[0]->positions == logs[1]->positions);
        if (family != 2) CHECK(logs[0]->taps == logs[1]->taps); // MTP never reads them
        if (family != 2) CHECK(!logs[1]->taps.empty());
        if (family == 2) CHECK(logs[1]->empty_taps > 0);
    }
    // The shipped default wraps the MTP drafter in prompt lookup: the wrapper forwards wants_taps() (lookup reads
    // committed tokens only), so prompt prefill still skips the tap copy; same outputs as the MTP shape alone.
    {
        set_env("STRIX_G4_PREFILL_PINNED_TAPS", nullptr);
        backend.requested.clear();
        auto log = std::make_shared<StageLog>();
        const e::DraftFactory mtp = [=] { return std::make_unique<ShapeDraft>(2,reference.ids,r.prompt.size()-1,log); };
        auto req = r; req.dflash = e::DraftMode::on;
        e::Coordinator c(backend,e::lookup::hybrid_factory(mtp,e::lookup::LookupConfig{})); const auto s = c.submit(req); c.run();
        check_equal(c.result(s),reference);
        CHECK(!backend.requested.empty());
        for (const bool asked : backend.requested) CHECK(!asked);
        const auto receipt = c.receipt(s);
        const auto* taps = receipt.find("prefill_taps");
        CHECK(taps && taps->as_string() == "skipped (no consumer)");
        CHECK(receipt.find("prompt_lookup"));
        CHECK(log->empty_taps > 0);
    }
    set_env("STRIX_G4_PREFILL_PINNED_TAPS", nullptr);
}

// Decode passes as the device backend returns them with DECODE_EXPORT on:
// device greedy winners (the shared better() rule), outputs checked before the
// failure fence, and logits/taps left out when no row of the call exports them.
class TrimBackend final : public e::Backend {
public:
    TrimBackend(const g4::ModelManifest& m, g4::RopeTables t) : host(m,t,1) {}
    std::unique_ptr<e::StreamState> create_stream() override { return host.create_stream(); }
    const g4::ModelConfig& config() const override { return host.config(); }
    Json identity() const override { return host.identity(); }
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan> rows) override {
        auto out = host.forward_rows(rows);
        out.selected.assign(out.rows,0U);
        for (std::size_t r = 0; r < out.rows; ++r) {
            std::uint32_t best = 0;
            for (std::uint32_t i = 1; i < out.vocab; ++i)
                if (g4::better(out.logits[r*out.vocab+i],i,out.logits[r*out.vocab+best],best)) best = i;
            out.selected[r] = best;
        }
        const bool logits = std::any_of(rows.begin(),rows.end(),[](const g4::RowPlan& p) { return p.export_logits; });
        const bool taps = std::any_of(rows.begin(),rows.end(),[](const g4::RowPlan& p) { return p.export_taps; });
        if (!logits) { out.logits.clear(); ++logits_skipped; } else ++logits_kept;
        if (!taps) { out.taps.clear(); ++taps_skipped; }
        out.device_checked = true;
        return out;
    }
    e::HostBackend host;
    std::size_t logits_skipped = 0, logits_kept = 0, taps_skipped = 0;
};
void decode_export_tests() {
    const auto model = tiny(); const Tables tables;
    e::HostBackend plain(model,tables.view(),1);
    TrimBackend trimmed(model,tables.view());
    const auto r = request(29);
    auto sampled = r; sampled.sampling = {0.8F,0.9F,12,123,0};
    for (const auto& req : {r, sampled}) {
        const auto reference = serial(plain,req);
        // No drafter: greedy passes need neither logits nor taps; sampled ones need logits.
        trimmed.logits_skipped = trimmed.logits_kept = trimmed.taps_skipped = 0;
        { e::Coordinator c(trimmed); const auto s = c.submit(req); c.run(); check_equal(c.result(s),reference); }
        CHECK(trimmed.taps_skipped > 0);
        // Greedy: every pass skips the logits. Sampled: prompt windows still skip
        // them, decode passes must carry them (select_token would refuse otherwise).
        CHECK(trimmed.logits_skipped > 0 && (trimmed.logits_kept > 0) == (req.sampling.temperature != 0));
        // DFlash-, EAGLE- and MTP-shaped sessions: same ids, digest, acceptance and staged inputs as the full exports.
        for (int family = 0; family < 3; ++family) {
            std::array<std::shared_ptr<StageLog>,2> logs;
            std::array<e::Result,2> results;
            for (std::size_t trim = 0; trim < 2; ++trim) {
                auto log = logs[trim] = std::make_shared<StageLog>();
                e::DraftFactory factory = [=] { return std::make_unique<ShapeDraft>(family,reference.ids,req.prompt.size()-1,log); };
                auto q = req; q.dflash = e::DraftMode::on;
                trimmed.logits_skipped = trimmed.taps_skipped = 0;
                e::Backend& backend = trim ? static_cast<e::Backend&>(trimmed) : static_cast<e::Backend&>(plain);
                e::Coordinator c(backend,factory); const auto s = c.submit(q); c.run();
                check_equal(c.result(s),reference);
                results[trim] = c.result(s);
                // Greedy drafting: only the MTP shape (no tap consumer) skips taps.
                // Sampled requests run without a drafter, so every pass may skip them.
                if (trim && req.sampling.temperature == 0) CHECK((trimmed.taps_skipped > 0) == (family == 2));
            }
            CHECK(results[0].ids == results[1].ids && results[0].digest == results[1].digest);
            CHECK(results[0].acceptance.accepted == results[1].acceptance.accepted);
            CHECK(logs[0]->hidden == logs[1]->hidden && logs[0]->positions == logs[1]->positions);
            if (family != 2) CHECK(logs[0]->taps == logs[1]->taps);
        }
    }
}

// A graph capture is granted only while no prefill pass is in flight, and a prefill pass waits for a running capture
// to end; STRIX_G4_CAPTURE_GUARD=0 grants every capture.
void capture_guard_tests() {
    namespace gpu = strix::gemma4::gpu;
    auto& state = gpu::capture_guard_state();
    const auto eager = state.eager_passes.load();
    { gpu::CaptureSlot slot; CHECK(slot.granted()); }
    {
        const gpu::PrefillInFlight prefill;
        { gpu::CaptureSlot slot; CHECK(!slot.granted()); }
        CHECK(state.eager_passes.load() == eager + 1);
        set_env("STRIX_G4_CAPTURE_GUARD", "0");
        { gpu::CaptureSlot slot; CHECK(slot.granted()); }
        set_env("STRIX_G4_CAPTURE_GUARD", nullptr);
    }
    { gpu::CaptureSlot slot; CHECK(slot.granted()); }
    // A prefill pass started on another thread during a granted capture waits until the capture is released.
    std::atomic<bool> entered{false};
    std::thread prefill;
    {
        gpu::CaptureSlot slot;
        CHECK(slot.granted());
        prefill = std::thread([&] { const gpu::PrefillInFlight p; entered = true; });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(!entered.load());
        slot.release();
        prefill.join();
        CHECK(entered.load());
    }
    CHECK(state.prefills.load() == 0);
}

class ScriptState final : public e::StreamState {
public:
    struct Tx final : g4::KvTxn {
        ScriptState& s; std::uint64_t start; std::size_t rows = 0; bool committed = false;
        explicit Tx(ScriptState& state) : s(state), start(state.p) {}
        std::uint64_t frontier() const override { return start; }
        void commit(std::size_t n) override { CHECK(!committed && n <= rows); s.p += n; committed = true; }
    };
    std::size_t index; std::uint64_t p = 0;
    explicit ScriptState(std::size_t i) : index(i) {}
    std::unique_ptr<g4::KvTxn> begin() override { return std::make_unique<Tx>(*this); }
    std::uint64_t frontier() const override { return p; }
    std::string digest() const override { return std::to_string(p); }
};
class ScriptBackend final : public e::Backend {
public:
    struct Script { std::size_t prompt; std::vector<std::uint32_t> tokens; };
    g4::ModelConfig c;
    std::vector<Script> scripts;
    std::size_t next = 0;
    ScriptBackend() { c.hidden = 1; c.vocab = 262144; c.max_positions = 65536; }
    std::unique_ptr<e::StreamState> create_stream() override { CHECK(next < scripts.size()); return std::make_unique<ScriptState>(next++); }
    const g4::ModelConfig& config() const override { return c; }
    Json identity() const override { return Json::object(); }
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan> rows) override {
        g4::ForwardResult out; out.rows = rows.size(); out.vocab = c.vocab; out.tap_width = 5;
        out.logits.resize(rows.size()*c.vocab,-1); out.taps.resize(rows.size()*5);
        for (const auto& row : rows) {
            auto* tx = dynamic_cast<ScriptState::Tx*>(row.transaction); CHECK(tx);
            const auto& s = scripts.at(tx->s.index);
            const auto i = row.position+1 >= s.prompt ? row.position+1-s.prompt : UINT64_MAX;
            const auto token = i < s.tokens.size() ? s.tokens[std::size_t(i)] : 1U;
            out.logits[row.output_row*c.vocab+token] = 1; ++tx->rows;
        }
        return out;
    }
};
class Writer final : public strix::serve::HttpResponseWriter {
public:
    int status = 0; bool ended = false; std::string body;
    void send(int s, std::string_view, std::string_view b, const std::vector<std::pair<std::string,std::string>>&) override { status = s; body = b; }
    bool begin_stream(int s, std::string_view, const std::vector<std::pair<std::string,std::string>>&) override { status = s; return true; }
    bool write(std::string_view b) override { body += b; return true; }
    void end_stream() override { ended = true; }
    bool client_gone() const override { return false; }
};
#ifndef _WIN32
void socket_test(e::Api& api, const std::string& body) {
    strix::serve::HttpServerConfig config; config.port = 0;
    strix::serve::HttpServer server(config,[&](const auto& req, auto& writer) { api.handle(req,writer); });
    server.listen(); std::thread listener([&] { server.serve(); });
    const int fd = socket(AF_INET,SOCK_STREAM,0);
    try {
        CHECK(fd >= 0); sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(server.port());
        CHECK(inet_pton(AF_INET,"127.0.0.1",&address.sin_addr) == 1);
        CHECK(connect(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address)) == 0);
        const auto request = "POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body;
        std::size_t sent = 0;
        while (sent < request.size()) { const auto n = send(fd,request.data()+sent,request.size()-sent,0); CHECK(n > 0); sent += std::size_t(n); }
        std::string response; char bytes[4096];
        for (;;) { const auto n = recv(fd,bytes,sizeof(bytes),0); CHECK(n >= 0); if (!n) break; response.append(bytes,std::size_t(n)); }
        CHECK(response.starts_with("HTTP/1.1 200")); CHECK(response.find("[DONE]") != std::string::npos);
    } catch (...) { if (fd >= 0) close(fd); server.stop(); listener.join(); throw; }
    close(fd); server.stop(); listener.join();
}
#endif
void chat_tests(const std::filesystem::path& tokenizer_path) {
    const auto tokenizer = g4::serve::Tokenizer::load(tokenizer_path); const g4::serve::Adapter adapter(tokenizer);
    e::ChatRequest r;
    r.input = g4::serve::chat_input(strix::serve::parse_json_unique(R"([{"role":"user","content":"Explain."}])"),
        Json(Json::Object{{"enable_thinking",true}}));
    r.generation.max_tokens = 12; r.generation.dflash = e::DraftMode::off;
    const auto thought = tokenizer->encode("<|channel>thought\nChecking"); r.thinking_budget = std::uint32_t(thought.size());
    g4::serve::OutputParser partial; for (auto id : thought) (void)partial.push(tokenizer->token_bytes(id)); (void)partial.finish();
    const auto continuation = g4::serve::continue_after_budget(r.input,partial.output(),r.thinking_budget,r.thinking_budget,12);
    const auto answer = tokenizer->encode("Answer.<turn|>");
    ScriptBackend backend; backend.scripts = {{adapter.prepare(r.input).ids.size(),thought},{tokenizer->encode(continuation.prompt).size(),answer}};
    std::string content, reasoning;
    const auto result = e::run_chat(backend,adapter,r,{},[&](const auto& d) { content += d.content; reasoning += d.reasoning; return true; });
    CHECK(result.budget_continued && result.receipts.size() == 2); CHECK(result.finish == "stop");
    CHECK(result.receipts[1].find("chat_policy")->find("selection")->as_string() == "budget-answer-continuation");
    CHECK(result.output.reasoning == "Checking"); CHECK(result.output.content == "Answer."); CHECK(content == result.output.content && reasoning == result.output.reasoning);
    if (g4::dflash::enabled_from_environment()) {
        ScriptBackend speculative; speculative.scripts = backend.scripts;
        std::size_t phase = 0;
        auto thinking = r; thinking.generation.dflash = e::DraftMode::on;
        const auto spec_result = e::run_chat(speculative,adapter,thinking,[&] {
            const auto& script = speculative.scripts.at(phase++);
            return std::make_unique<FakeDraft>(script.tokens,script.prompt-1,0,0,std::make_shared<DraftLog>());
        });
        CHECK(spec_result.ids == result.ids && spec_result.budget_continued);
        CHECK(spec_result.output.content == result.output.content && spec_result.output.reasoning == result.output.reasoning);
        CHECK(spec_result.receipts.front().find("acceptance")->find("attempted")->as_int() > 0);
    }
    // Natural stop inside the first budget does not execute a continuation.
    ScriptBackend stopped; stopped.scripts = {{adapter.prepare(r.input).ids.size(),{1}}};
    const auto natural = e::run_chat(stopped,adapter,r); CHECK(!natural.budget_continued && stopped.next == 1);
    const auto request_json = strix::serve::parse_json_unique(R"({"messages":[{"role":"user","content":"Weather?"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":64})");
    const auto parsed = e::parse_chat_request(request_json,e::DraftMode::off);
    const auto tool_frame = std::string("<|tool_call>call:weather{city:<|\"|>Madrid<|\"|>}<tool_call|><|tool_response>");
    const auto tool_ids = tokenizer->encode(tool_frame+"ignored");
    ScriptBackend tool; for (unsigned i = 0; i < 3; ++i) tool.scripts.push_back({adapter.prepare(parsed.input).ids.size(),tool_ids});
    e::Api api(tool,tokenizer,{},e::DraftMode::off);
    strix::serve::HttpRequest req; req.method = "POST"; req.path = "/v1/chat/completions"; req.body = request_json.dump();
    Writer writer; api.handle(req,writer); CHECK(writer.status == 200);
    const auto response = strix::serve::parse_json_unique(writer.body);
    CHECK(response.find("usage")->find("completion_tokens")->as_int() == std::int64_t(tokenizer->encode(tool_frame).size()));
    const auto& wire_ids = response.find("strix")->find("generated_ids")->as_array();
    const auto expected_ids = tokenizer->encode(tool_frame);
    CHECK(wire_ids.size() == expected_ids.size());
    for (std::size_t i = 0; i < wire_ids.size(); ++i) CHECK(wire_ids[i].as_int() == expected_ids[i]);
    const auto& choice = response.find("choices")->as_array()[0]; CHECK(choice.find("finish_reason")->as_string() == "tool_calls");
    const auto& message = *choice.find("message"); const auto& call = message.find("tool_calls")->as_array()[0];
    CHECK(call.find("function")->find("arguments")->as_string() == "{\"city\":\"Madrid\"}");
    auto round_trip = Json(Json::Object{{"messages",Json(Json::Array{request_json.find("messages")->as_array()[0],message,
        Json(Json::Object{{"role","tool"},{"tool_call_id",*call.find("id")},{"content","sunny"}})})}});
    CHECK(adapter.prepare(e::parse_chat_request(round_trip).input).rendered.find("sunny") != std::string::npos);
    auto streaming = Json(request_json.as_object()); streaming.set("stream",true); req.body = streaming.dump();
    Writer sse; api.handle(req,sse); CHECK(sse.status == 200 && sse.ended); CHECK(sse.body.find("[DONE]") != std::string::npos); CHECK(sse.body.find("tool_calls") != std::string::npos);
#ifndef _WIN32
    socket_test(api,req.body);
#endif
    for (const auto* path : {"/health","/v1/models","/v1/capabilities"}) { req.method = "GET"; req.path = path; Writer w; api.handle(req,w); CHECK(w.status == 200); }
    refuses([&] { auto bad = Json(request_json.as_object()); bad.set("logprobs",true); (void)e::parse_chat_request(bad); });
    // Sampled requests force serial selection even when DFlash was requested
    // and no drafter is installed. Keep API admission aligned with Coordinator.
    const auto sampled_json = strix::serve::parse_json_unique(R"({"messages":[{"role":"user","content":"OK"}],"temperature":1,"top_k":1,"dflash":"on","max_tokens":4})");
    const auto sampled_request = e::parse_chat_request(sampled_json);
    ScriptBackend sampled; sampled.scripts = {{adapter.prepare(sampled_request.input).ids.size(),tokenizer->encode("OK<eos>")}};
    e::Api sample_api(sampled,tokenizer);
    strix::serve::HttpRequest sample_req; sample_req.method = "POST"; sample_req.path = "/v1/chat/completions"; sample_req.body = sampled_json.dump();
    Writer sample_writer; sample_api.handle(sample_req,sample_writer); CHECK(sample_writer.status == 200);
    CHECK(sample_writer.body.find("serial-sampling") != std::string::npos);
    std::cout << "chat: checked budget continuation/natural stop, streamed parsing, queued API, tools round trip and SSE passed\n";
}
}
int main(int argc, char** argv) {
    try {
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
        if (argc == 2 && std::string(argv[1]) == "--device") { device_tests(); return 0; }
#endif
        if (argc == 3 && std::string(argv[1]) == "--chat") chat_tests(argv[2]);
        else host_tests(argc == 2 && std::string(argv[1]) == "--off");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
