#include "gemma4/engine/coordinator.hpp"
#include "gemma4/exl3/format.hpp"
#include <iostream>

namespace g4 = strix::gemma4;
namespace {
class Backend final : public g4::engine::Backend {
public:
    bool exl3 = true;
    std::string_view prefill_route;
    std::unique_ptr<g4::engine::StreamState> create_stream() override { throw std::logic_error("metadata-only fixture"); }
    g4::ForwardResult forward_rows(std::span<const g4::RowPlan>) override { throw std::logic_error("metadata-only fixture"); }
    const g4::ModelConfig& config() const override { return config_; }
    strix::serve::Json identity() const override {
        auto out = strix::serve::Json::object();
        if (exl3) out.set("identity", g4::exl3::identity);
        if (!prefill_route.empty()) out.set("prefill_route", prefill_route);
        return out;
    }
private:
    g4::ModelConfig config_;
};
}
int main() {
    try {
        Backend backend; g4::engine::Request request;
        auto record = g4::engine::effective_config(backend, request, false);
        if (record.find("decode_identity")->as_string() != g4::exl3::identity) throw std::runtime_error("EXL3 receipt identity");
        if (record.find("prefill_identity")->as_string() != "prefill-exact") throw std::runtime_error("absent prefill route fallback");
        for (const auto& [route, identity] : {std::pair<std::string_view, std::string_view>{g4::prefill::identity, "g4-wmma-pf-v1"},
                {g4::prefill::v2_identity, "g4-wmma-pf-v2"}, {g4::exl3::prefill_identity, "g4-exl3-wmma-pf-v1"},
                {"a8-many-row", "prefill-a8"}, {"many-row", "prefill-exact"}, {"rows8", "prefill-exact"}, {"unknown", "prefill-exact"}}) {
            backend.prefill_route = route;
            record = g4::engine::effective_config(backend, request, false);
            if (record.find("prefill_identity")->as_string() != identity) throw std::runtime_error("prefill receipt identity");
        }
        backend.exl3 = false;
        record = g4::engine::effective_config(backend, request, false);
        if (record.find("decode_identity")->as_string() != g4::arithmetic_id) throw std::runtime_error("legacy metadata fallback");
        std::cout << "EXL3 and legacy receipt identities PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
