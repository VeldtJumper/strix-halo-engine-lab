#include "fixture.hpp"
#include "gemma4/ref/npz.hpp"
#include <cstring>
#include <bit>
#include <limits>
#include <iostream>
#include <map>
#include <thread>
#ifndef _WIN32
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace g = strix::gemma4;
namespace e = g::engine;
namespace f = g::features;
using Json = strix::serve::Json;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ")+#x+" line "+std::to_string(__LINE__)); } while (false)
template<class F> void refuses(F call) { bool failed = false; try { call(); } catch (const std::exception&) { failed = true; } CHECK(failed); }
void bytes(const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size()); CHECK(a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float)) == 0);
}
void equal(const f::Result& a, const f::Result& b) {
    CHECK(a.before == b.before && a.after == b.after && a.frontier == b.frontier && a.branches.size() == b.branches.size());
    for (std::size_t i = 0; i < a.branches.size(); ++i) {
        const auto& x = a.branches[i]; const auto& y = b.branches[i];
        CHECK(x.id == y.id && x.processed_rows == y.processed_rows && x.options.size() == y.options.size());
        CHECK(x.read.branch_position == y.read.branch_position && x.read.absolute_position == y.read.absolute_position);
        bytes(x.read.values,y.read.values); bytes(x.code_logits,y.code_logits);
        for (std::size_t p = 0; p < x.options.size(); ++p) bytes(x.options[p].values,y.options[p].values);
    }
}
g::ForwardResult step(e::Backend& backend, e::StreamState& state, std::uint32_t token, bool capture) {
    auto tx = state.begin(); const g::RowPlan row{0,0,state.frontier(),token,tx.get(),0,true,capture};
    auto pass = backend.forward_rows({&row,1}); tx->commit(1); return pass;
}
void ordinary(e::Backend& backend, const f::Request& request, const f::Result& result) {
    for (std::size_t b = 0; b < request.branches.size(); ++b) {
        auto stream = backend.create_stream(96);
        for (auto token : *request.prefix) (void)step(backend,*stream,token,false);
        const auto& branch = request.branches[b];
        for (std::size_t p = 0; p < branch.tokens.size(); ++p) {
            const auto pass = step(backend,*stream,branch.tokens[p],true);
            if (p == branch.read_position) {
                bytes(pass.hidden,result.branches[b].read.values);
                std::vector<float> codes; for (auto id : request.code_ids) codes.push_back(pass.logits[id]);
                bytes(codes,result.branches[b].code_logits);
            }
            for (std::size_t i = 0; i < branch.option_close_positions.size(); ++i)
                if (p == branch.option_close_positions[i]) bytes(pass.hidden,result.branches[b].options[i].values);
        }
    }
}
void isolation(e::Backend& backend, f::Exporter& exporter, const f::Request& r) {
    auto control = backend.create_stream(96), inspected = backend.create_stream(96);
    for (auto token : *r.prefix) {
        const auto a = step(backend,*control,token,false), b = step(backend,*inspected,token,true);
        CHECK(a.hidden.empty()); bytes(a.logits,b.logits); bytes(a.taps,b.taps);
    }
    const auto digest = inspected->digest();
    (void)exporter.export_features(r,*inspected);
    CHECK(inspected->digest() == digest && inspected->digest() == control->digest());
    for (auto token : feature_fixture::tokens(4,5)) {
        bytes(step(backend,*control,token,true).hidden,step(backend,*inspected,token,true).hidden);
        CHECK(inspected->digest() == control->digest());
    }
}
Json wire(const f::Request& r) {
    auto prefix = Json::array(); for (auto id : *r.prefix) prefix.push(std::uint64_t(id));
    auto branches = Json::array();
    for (const auto& b : r.branches) {
        auto ids = Json::array(); for (auto id : b.tokens) ids.push(std::uint64_t(id));
        auto positions = Json::array(); for (auto p : b.option_close_positions) positions.push(std::uint64_t(p));
        branches.push(Json(Json::Object{{"id",b.id},{"token_ids",ids},{"read_position",std::uint64_t(b.read_position)},{"option_close_positions",positions}}));
    }
    auto codes = Json::array(); for (auto id : r.code_ids) codes.push(std::uint64_t(id));
    return Json(Json::Object{{"identity",r.identity.json()},{"prefix_ids",prefix},{"branches",branches},{"code_ids",codes}});
}
class FailureBackend final : public e::Backend {
public:
    explicit FailureBackend(e::Backend& b) : backend(b) {}
    std::unique_ptr<e::StreamState> create_stream() override { return backend.create_stream(); }
    const g::ModelConfig& config() const override { return backend.config(); }
    Json identity() const override { return backend.identity(); }
    g::ForwardResult forward_rows(std::span<const g::RowPlan> rows) override {
        auto result = backend.forward_rows(rows);
        if (++calls == 2) throw std::runtime_error("injected failure after overlay write");
        return result;
    }
    e::Backend& backend; unsigned calls = 0;
};
class MetadataBackend final : public e::Backend {
public:
    MetadataBackend(e::Backend& b, f::Identity i) : backend(b), frozen(std::move(i)) {}
    std::unique_ptr<e::StreamState> create_stream() override { return backend.create_stream(); }
    const g::ModelConfig& config() const override { return backend.config(); }
    Json identity() const override { auto j = backend.identity(); j.set("feature_exporter_identity",frozen.json()); return j; }
    g::ForwardResult forward_rows(std::span<const g::RowPlan> rows) override { return backend.forward_rows(rows); }
    e::Backend& backend; f::Identity frozen;
};
// A backend with a many-row prefill (emulated with exact forward_rows passes) whose prefill_matches_decode is
// `matches`: the exporter may take its many-row path only when prompt rows equal decode rows (prefill-exact);
// a v1/v2/a8 prompt route (matches = false) must never reach the exported features.
class ManyRowBackend final : public e::Backend {
public:
    ManyRowBackend(e::Backend& b, bool m) : backend(b), matches(m) {}
    std::unique_ptr<e::StreamState> create_stream() override { return backend.create_stream(); }
    std::unique_ptr<e::StreamState> create_stream(std::uint32_t admitted) override { return backend.create_stream(admitted); }
    const g::ModelConfig& config() const override { return backend.config(); }
    Json identity() const override { return backend.identity(); }
    g::ForwardResult forward_rows(std::span<const g::RowPlan> rows) override { return backend.forward_rows(rows); }
    std::size_t prefill_chunk_rows() const override { return 16; }
    bool prefill_matches_decode() const override { return matches; }
    e::PrefillRows prefill_rows(g::KvTxn& tx, std::span<const std::uint32_t> tokens, bool hidden, bool) override {
        ++calls; e::PrefillRows out; const auto width = backend.config().hidden; tx.reserve_rows(tokens.size());
        for (std::size_t p = 0; p < tokens.size(); p += 8) {
            std::vector<g::RowPlan> rows;
            for (std::size_t k = 0; k < std::min<std::size_t>(8,tokens.size()-p); ++k)
                rows.push_back({0,0,tx.frontier()+p+k,tokens[p+k],&tx,k,true,true});
            const auto pass = backend.forward_rows(rows);
            for (std::size_t k = 0; k < rows.size(); ++k) {
                const auto first = pass.hidden.begin()+std::ptrdiff_t(k*width);
                std::vector<float> h(first,first+std::ptrdiff_t(width));
                const auto logits = pass.logits.begin()+std::ptrdiff_t(k*pass.vocab);
                heads[std::string(reinterpret_cast<const char*>(h.data()),h.size()*sizeof(float))].assign(logits,logits+pass.vocab);
                if (hidden) out.hidden.insert(out.hidden.end(),h.begin(),h.end());
            }
        }
        return out;
    }
    bool head_logits(std::span<const float> hidden, std::vector<float>& logits) override {
        const auto it = heads.find(std::string(reinterpret_cast<const char*>(hidden.data()),hidden.size_bytes()));
        if (it == heads.end()) return false;
        logits = it->second; return true;
    }
    e::Backend& backend; bool matches; unsigned calls = 0; std::map<std::string,std::vector<float>> heads;
};
void many_row_gate(e::Backend& host, const f::Identity& identity) {
    const auto request = feature_fixture::request(identity);
    const auto expected = f::Exporter(host,identity).export_features(request);
    ManyRowBackend other(host,false), exact(host,true);
    equal(expected,f::Exporter(other,identity).export_features(request));
    CHECK(other.calls == 0);
    equal(expected,f::Exporter(exact,identity).export_features(request));
    CHECK(exact.calls > 0);
    std::cout << "PASS many-row feature rows only when prefill matches decode\n";
}
class Writer final : public strix::serve::HttpResponseWriter {
public:
    int status = 0; std::string body;
    void send(int s, std::string_view, std::string_view b, const std::vector<std::pair<std::string,std::string>>&) override { status = s; body = b; }
    bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string,std::string>>&) override { return false; }
    bool write(std::string_view) override { return false; }
    void end_stream() override {}
    bool client_gone() const override { return false; }
};
void http(e::Backend& backend, f::Identity identity, const std::filesystem::path& path) {
    const g::MappedFile asset(path); identity.tokenizer = g::hash_bytes(asset.data(),asset.size());
    const auto tokenizer = g::serve::Tokenizer::load(path); const g::serve::Adapter adapter(tokenizer);
    MetadataBackend wrapped(backend,identity); e::Api api(wrapped,tokenizer);
    auto request = feature_fixture::request(identity);
    strix::serve::HttpRequest req; req.method = "POST"; req.path = "/v1/gemma4/features"; req.body = wire(request).dump();
    Writer first, second;
    std::thread worker([&] { api.handle(req,first); }); api.handle(req,second); worker.join();
    CHECK(first.status == 200 && second.status == 200);
    const auto result = strix::serve::parse_json_unique(first.body);
    CHECK(result.find("branch_isolation")->as_bool());
    CHECK(result.find("branches")->dump() == strix::serve::parse_json_unique(second.body).find("branches")->dump());
    request.identity.weights += "wrong"; req.body = wire(request).dump(); Writer bad; api.handle(req,bad); CHECK(bad.status == 400);
    auto chat = wire(feature_fixture::request(identity));
    Json::Object object;
    for (const auto& [key,value] : chat.as_object()) if (key != "prefix_ids") object.emplace_back(key,value);
    chat = Json(object); const auto messages = strix::serve::parse_json_unique(R"([{"role":"user","content":"A test."}])");
    chat.set("messages",messages);
    CHECK(*f::parse_request(chat,&adapter).prefix == adapter.prepare(g::serve::chat_input(messages)).ids);
    std::cout << "PASS HTTP route, concurrent serialization, identity refusal, chat rendering\n";
}
void gates(e::Backend& backend, const f::Identity& identity) {
    f::Exporter exporter(backend,identity,1); auto request = feature_fixture::request(identity);
    const auto result = exporter.export_features(request);
    CHECK(result.before == result.after && result.frontier == 13);
    equal(result,exporter.export_features(request));
    auto serial = request; serial.chunk_rows = 1; equal(result,exporter.export_features(serial));
    ordinary(backend,request,result); isolation(backend,exporter,request);
    for (std::size_t b = 0; b < request.branches.size(); ++b) {
        auto solo = request; solo.branches = {request.branches[b]};
        auto expected = result; expected.branches = {result.branches[b]};
        equal(expected,exporter.export_features(solo));
    }
    request.retain_prefix = true; const auto retained = exporter.export_features(request);
    CHECK(!retained.stream_handle.empty());
    refuses([&] { (void)exporter.export_features(request); });
    auto reuse = request; reuse.prefix.reset(); reuse.stream_handle = retained.stream_handle; reuse.retain_prefix = false;
    equal(result,exporter.export_features(reuse));
    auto bad = reuse; bad.identity.route += "wrong"; refuses([&] { (void)exporter.export_features(bad); });
    bad = reuse; bad.branches[0].option_close_positions = {99}; refuses([&] { (void)exporter.export_features(bad); });
    equal(result,exporter.export_features(reuse));
    reuse.release_prefix = true; (void)exporter.export_features(reuse);
    refuses([&] { (void)exporter.export_features(reuse); });
    request.retain_prefix = false;
    const auto parsed = f::parse_request(wire(request)); equal(result,exporter.export_features(parsed));
    auto invalid = wire(request); invalid.set("unknown",true); refuses([&] { (void)f::parse_request(invalid); });
    auto state = backend.create_stream(96);
    for (auto token : *request.prefix) (void)step(backend,*state,token,false);
    const auto digest = state->digest(); FailureBackend failing(backend); f::Exporter failed(failing,identity);
    refuses([&] { (void)failed.export_features(request,*state); }); CHECK(state->digest() == digest);
    (void)step(backend,*state,7,false); // A failed branch releases the exclusive transaction lease.
    const auto archive = std::filesystem::temp_directory_path()/"gemma4-feature-test.npz";
    f::write_npz(archive,result); const auto arrays = g::read_npz(archive); std::filesystem::remove(archive);
    bytes(arrays.at("branch_0_read").values<float>("<f4"),result.branches[0].read.values);
    CHECK(arrays.at("branch_0_options").shape == std::vector<std::size_t>({5,16}));
    const auto json = strix::serve::parse_json_unique(result.json().dump()); CHECK(json.find("branch_isolation")->as_bool());
    auto edge = result;
    edge.branches[0].read.values = {-0.0F,0.0F,std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::max()};
    const auto encoded = strix::serve::parse_json_unique(edge.json().dump());
    const auto& values = encoded.find("branches")->as_array()[0].find("read")->find("hidden")->as_array();
    for (std::size_t i = 0; i < values.size(); ++i)
        CHECK(std::bit_cast<std::uint32_t>(float(values[i].as_double())) == std::bit_cast<std::uint32_t>(edge.branches[0].read.values[i]));
    CHECK(values[0].number_source() == "-0.0");
    std::cout << "PASS features: serial/chunk, ordinary rows, aliases 22/23, isolation/continuation, groups/solo, repeatability, handles, failures, JSON/NPZ\n";
}
int main(int argc, char** argv) {
    try {
        const auto model = feature_fixture::model(); const feature_fixture::Tables tables;
        e::HostBackend host(model,tables.view(),1); const auto identity = f::freeze(model);
        gates(host,identity); many_row_gate(host,identity);
        if (argc > 2 && std::string(argv[1]) == "--tokenizer") http(host,identity,argv[2]);
        if (argc > 1 && std::string(argv[1]) == "--device") {
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
            auto device = e::make_device_backend(model,tables.view()); gates(*device,identity);
            f::Exporter h(host,identity), d(*device,identity); const auto request = feature_fixture::request(identity);
            equal(h.export_features(request),d.export_features(request));
            std::cout << "PASS host/device feature bytes\n";
#else
            throw std::runtime_error("device backend not built");
#endif
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
