#include "features.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

namespace strix::gemma4::features {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
const Json& required(const Json& j, const char* key) {
    const auto* p = j.find(key);
    if (!p) throw std::invalid_argument(std::string("missing field: ")+key);
    return *p;
}
void fields(const Json& j, std::initializer_list<const char*> names) {
    for (const auto& [key,value] : j.as_object()) {
        (void)value;
        if (std::none_of(names.begin(),names.end(),[&](const char* n) { return key == n; }))
            throw std::invalid_argument("unsupported feature field: "+key);
    }
}
std::size_t number(const Json& j) {
    const auto n = j.as_int();
    if (n < 0 || n > 65536) throw std::invalid_argument("feature index outside context range");
    return std::size_t(n);
}
std::vector<std::uint32_t> ids(const Json& j) {
    std::vector<std::uint32_t> out;
    for (const auto& n : j.as_array()) {
        const auto v = n.as_int();
        if (v < 0 || v > UINT32_MAX) throw std::invalid_argument("invalid feature token id");
        out.push_back(std::uint32_t(v));
    }
    return out;
}
Identity identity_json(const Json& j) {
    fields(j,{"weights_hash","route_hash","tokenizer_hash","template_hash","rope_hash","identity","prefill_identity","cache_dtype"});
    Identity i;
    i.weights = required(j,"weights_hash").as_string(); i.route = required(j,"route_hash").as_string();
    i.tokenizer = required(j,"tokenizer_hash").as_string(); i.chat_template = required(j,"template_hash").as_string();
    i.rope = required(j,"rope_hash").as_string(); i.decode = required(j,"identity").as_string();
    i.prefill = required(j,"prefill_identity").as_string(); i.cache = required(j,"cache_dtype").as_string();
    return i;
}
Json floats(std::span<const float> values) {
    auto out = Json::array();
    for (const float x : values) {
        if (!std::isfinite(x)) throw std::runtime_error("nonfinite exported feature");
        // Keep negative zero a floating number for JSON consumers with distinct integer parsing.
        out.push(x == 0.0F && std::signbit(x) ? Json::number_text("-0.0") : Json(double(x)));
    }
    return out;
}
Json row_json(const HiddenRow& row, bool data) {
    Json j(Json::Object{{"branch_position",std::uint64_t(row.branch_position)},{"absolute_position",row.absolute_position}});
    if (data) j.set("hidden",floats(row.values));
    return j;
}
}
Json Identity::json() const {
    return Json(Json::Object{{"weights_hash",weights},{"route_hash",route},{"tokenizer_hash",tokenizer},
        {"template_hash",chat_template},{"rope_hash",rope},{"identity",decode},{"prefill_identity",prefill},{"cache_dtype",cache}});
}
Identity freeze(const ModelManifest& model, std::string tokenizer_hash) {
    Identity i;
    i.weights = model.model_hash; i.rope = model.rope_hash; i.decode = model.identity; i.cache = model.cache_dtype;
    i.tokenizer = std::move(tokenizer_hash);
    i.chat_template = STRIX_G4_FEATURE_TEMPLATE_HASH;
    std::string route;
    for (const auto& t : model.tensors) {
        route += t.name+":"+std::to_string(int(t.view.format))+":"+std::to_string(t.view.row_bytes);
        for (auto d : t.shape) route += ":"+std::to_string(d);
        route += "\n";
    }
    i.route = hash_bytes(route.data(),route.size());
    return i;
}
Identity backend_identity(const engine::Backend& backend) {
    return identity_json(required(backend.identity(),"feature_exporter_identity"));
}
Json Result::json(bool data) const {
    Json j(Json::Object{{"identity",identity.json()},{"stream_handle",stream_handle},{"prefix_frontier",frontier},
        {"committed_digest_before",before},{"committed_digest_after",after},{"branch_isolation",before == after},
        {"transaction_policy","uncommitted-overlay-discard"},{"prefill_ms",prefix_ms},{"branch_ms",branch_ms}});
    auto out = Json::array();
    for (const auto& b : branches) {
        auto options = Json::array(); for (const auto& r : b.options) options.push(row_json(r,data));
        Json item(Json::Object{{"id",b.id},{"processed_rows",std::uint64_t(b.processed_rows)},
            {"committed_rows",0},{"read",row_json(b.read,data)},{"options",options}});
        if (data) item.set("code_logits",floats(b.code_logits));
        out.push(item);
    }
    j.set("branches",out);
    auto codes = Json::array(); for (auto id : code_ids) codes.push(std::uint64_t(id));
    j.set("code_ids",codes); return j;
}
Exporter::Exporter(engine::Backend& backend, Identity identity, std::size_t limit)
    : backend_(backend), identity_(std::move(identity)), limit_(limit) {
    const auto b = backend.identity();
    if (identity_.decode != arithmetic_id || identity_.prefill != "prefill-exact" || identity_.cache != "F32" ||
        identity_.weights.empty() || identity_.route.empty() || identity_.rope.empty() || identity_.chat_template.empty() ||
        required(b,"identity").as_string() != identity_.decode || required(b,"model_hash").as_string() != identity_.weights ||
        required(b,"rope_hash").as_string() != identity_.rope || required(b,"cache_dtype").as_string() != identity_.cache)
        throw std::invalid_argument("feature identity does not match frozen backend");
}
void Exporter::release(const std::string& handle) {
    if (!prefixes_.erase(handle)) throw std::invalid_argument("unknown feature stream handle");
}
Result Exporter::export_features(const Request& r) {
    const auto& c = backend_.config();
    if (r.identity != identity_) throw std::invalid_argument("feature identity mismatch");
    if (r.prefix.has_value() == !r.stream_handle.empty()) throw std::invalid_argument("choose prefix_ids/messages or stream_handle");
    if (!r.chunk_rows || r.chunk_rows > 8 || r.branches.empty()) throw std::invalid_argument("branches and chunk_rows in 1..8 required");
    if (r.retain_prefix && (!r.prefix || prefixes_.size() >= limit_)) throw std::invalid_argument("feature prefix cache full or handle already retained");
    if (r.release_prefix && r.stream_handle.empty()) throw std::invalid_argument("release_prefix requires stream_handle");
    Prefix temporary;
    Prefix* prefix = &temporary;
    if (!r.stream_handle.empty()) {
        const auto found = prefixes_.find(r.stream_handle);
        if (found == prefixes_.end()) throw std::invalid_argument("unknown feature stream handle");
        prefix = &found->second;
    }
    const auto base = r.prefix ? std::uint64_t(r.prefix->size()) : prefix->state->frontier();
    std::size_t longest = 0;
    const auto validate_ids = [&](const auto& tokens) { for (auto id : tokens) if (id >= c.vocab) throw std::invalid_argument("feature token outside vocabulary"); };
    if (r.prefix) validate_ids(*r.prefix);
    validate_ids(r.code_ids);
    std::set<std::string> branch_ids;
    for (const auto& b : r.branches) {
        validate_ids(b.tokens);
        if (b.id.empty() || !branch_ids.insert(b.id).second || b.tokens.empty() || b.read_position >= b.tokens.size())
            throw std::invalid_argument("invalid feature branch id/read position");
        for (auto p : b.option_close_positions) if (p >= b.tokens.size()) throw std::invalid_argument("option close outside branch");
        longest = std::max(longest,b.tokens.size());
    }
    if (base > c.max_positions || longest > c.max_positions - base ||
        (!r.prefix && longest > prefix->admitted - base)) throw std::invalid_argument("feature branch exceeds admitted context");
    Result result; result.identity = identity_; result.frontier = base;
    auto start = Clock::now();
    if (r.prefix) {
        temporary.admitted = r.retain_prefix ? c.max_positions : std::uint32_t(base+longest);
        temporary.state = backend_.create_stream(temporary.admitted);
        // Many-row chunks only when the backend's prompt rows equal its decode
        // rows (prefill-exact, same committed bytes): a v1/v2/a8 prompt route
        // would change the frozen identity. chunk_rows < 8 keeps explicit small passes.
        const auto many = r.chunk_rows == 8 && backend_.prefill_matches_decode() ? backend_.prefill_chunk_rows() : 0;
        for (std::size_t p = 0; p < r.prefix->size();) {
            if (many && r.prefix->size()-p > 8) {
                const auto n = std::min<std::size_t>(r.prefix->size()-p,many-p%many);
                auto tx = temporary.state->begin();
                (void)backend_.prefill_rows(*tx,{r.prefix->data()+p,n},false,false);
                tx->commit(n); p += n;
                continue;
            }
            const auto n = std::min(r.chunk_rows,r.prefix->size()-p);
            auto tx = temporary.state->begin(); std::vector<RowPlan> rows;
            for (std::size_t k = 0; k < n; ++k) rows.push_back({0,0,p+k,(*r.prefix)[p+k],tx.get(),k,true});
            (void)backend_.forward_rows(rows); tx->commit(n); p += n;
        }
    }
    const auto prefix_ms = elapsed(start);
    result = export_features(r,*prefix->state); result.prefix_ms = prefix_ms;
    if (r.retain_prefix) {
        result.stream_handle = "g4-prefix-"+std::to_string(++sequence_);
        prefixes_.emplace(result.stream_handle,std::move(temporary));
    } else if (!r.release_prefix) result.stream_handle = r.stream_handle;
    if (r.release_prefix) release(r.stream_handle);
    return result;
}
Result Exporter::export_features(const Request& r, engine::StreamState& prefix) {
    const auto& c = backend_.config();
    if (r.identity != identity_) throw std::invalid_argument("feature identity mismatch");
    if (!r.chunk_rows || r.chunk_rows > 8 || r.branches.empty()) throw std::invalid_argument("branches and chunk_rows in 1..8 required");
    const auto base = prefix.frontier();
    std::size_t longest = 0;
    const auto validate_ids = [&](const auto& tokens) { for (auto id : tokens) if (id >= c.vocab) throw std::invalid_argument("feature token outside vocabulary"); };
    validate_ids(r.code_ids);
    std::set<std::string> branch_ids;
    for (const auto& b : r.branches) {
        validate_ids(b.tokens);
        if (b.id.empty() || !branch_ids.insert(b.id).second || b.tokens.empty() || b.read_position >= b.tokens.size())
            throw std::invalid_argument("invalid feature branch id/read position");
        for (auto p : b.option_close_positions) if (p >= b.tokens.size()) throw std::invalid_argument("option close outside branch");
        longest = std::max(longest,b.tokens.size());
    }
    if (base > c.max_positions || longest > c.max_positions - base) throw std::invalid_argument("feature branch exceeds admitted context");
    Result result; result.identity = identity_; result.frontier = base; result.code_ids = r.code_ids;
    result.before = prefix.digest();
    auto start = Clock::now();
    for (const auto& b : r.branches) {
        BranchResult out; out.id = b.id;
        auto tx = prefix.begin(); tx->reserve_rows(b.tokens.size());
        const auto capture = [&](std::size_t position, std::size_t offset, const ForwardResult& pass) {
            HiddenRow row; row.branch_position = position; row.absolute_position = base+position;
            const auto first = pass.hidden.begin()+std::ptrdiff_t(offset*c.hidden);
            row.values.assign(first,first+c.hidden);
            for (auto x : row.values) if (!std::isfinite(x)) throw std::runtime_error("nonfinite exported hidden row");
            return row;
        };
        out.options.resize(b.option_close_positions.size());
        // One exact many-row pass over the whole branch (only when prompt rows
        // equal decode rows): hidden rows from the same final norm, code logits
        // from the same head and softcap.
        const auto many = r.chunk_rows == 8 && backend_.prefill_matches_decode() ? backend_.prefill_chunk_rows() : 0;
        if (many && b.tokens.size() > 8 && b.tokens.size() <= many) {
            tx.reset(); tx = prefix.begin();
            const auto rows = backend_.prefill_rows(*tx,b.tokens,true,false);
            const auto hidden_rows = rows.hidden_rows();
            if (hidden_rows.size() != b.tokens.size()*c.hidden) throw std::runtime_error("backend returned incomplete feature rows");
            const auto row = [&](std::size_t position) {
                HiddenRow h; h.branch_position = position; h.absolute_position = base+position;
                const auto first = hidden_rows.begin()+std::ptrdiff_t(position*c.hidden);
                h.values.assign(first,first+c.hidden);
                for (auto x : h.values) if (!std::isfinite(x)) throw std::runtime_error("nonfinite exported hidden row");
                return h;
            };
            out.read = row(b.read_position);
            if (!r.code_ids.empty()) {
                std::vector<float> logits;
                if (!backend_.head_logits(out.read.values,logits) || logits.size() != c.vocab)
                    throw std::runtime_error("backend has no head for many-row feature rows");
                for (auto id : r.code_ids) {
                    if (!std::isfinite(logits[id])) throw std::runtime_error("nonfinite exported code logit");
                    out.code_logits.push_back(logits[id]);
                }
            }
            for (std::size_t i = 0; i < b.option_close_positions.size(); ++i) out.options[i] = row(b.option_close_positions[i]);
            out.processed_rows = b.tokens.size();
            tx.reset(); // Never commit a question branch, including its delimiter.
            result.branches.push_back(std::move(out));
            continue;
        }
        for (std::size_t p = 0; p < b.tokens.size();) {
            const auto n = std::min(r.chunk_rows,b.tokens.size()-p);
            std::vector<RowPlan> rows;
            for (std::size_t k = 0; k < n; ++k) {
                const auto pos = p+k;
                const bool wanted = pos == b.read_position || std::find(b.option_close_positions.begin(),b.option_close_positions.end(),pos) != b.option_close_positions.end();
                rows.push_back({0,0,base+pos,b.tokens[pos],tx.get(),k,true,wanted});
            }
            const auto pass = backend_.forward_rows(rows);
            if (pass.rows != n || pass.vocab != c.vocab || pass.logits.size() != n*c.vocab ||
                (std::any_of(rows.begin(),rows.end(),[](const RowPlan& row) { return row.export_hidden; }) && pass.hidden.size() != n*c.hidden))
                throw std::runtime_error("backend returned incomplete feature rows");
            if (b.read_position >= p && b.read_position < p+n) {
                const auto k = b.read_position-p; out.read = capture(b.read_position,k,pass);
                for (auto id : r.code_ids) {
                    const auto value = pass.logits[k*c.vocab+id];
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite exported code logit");
                    out.code_logits.push_back(value);
                }
            }
            for (std::size_t i = 0; i < b.option_close_positions.size(); ++i) {
                const auto pos = b.option_close_positions[i];
                if (pos >= p && pos < p+n) out.options[i] = capture(pos,pos-p,pass);
            }
            p += n; out.processed_rows += n;
        }
        tx.reset(); // Never commit a question branch, including its delimiter.
        result.branches.push_back(std::move(out));
    }
    result.branch_ms = elapsed(start); result.after = prefix.digest();
    if (result.before != result.after || prefix.frontier() != base) throw std::runtime_error("feature branch changed committed prefix");
    return result;
}
Request parse_request(const Json& j, const gemma4::serve::Adapter* adapter) {
    fields(j,{"identity","prefix_ids","messages","chat_template_kwargs","tools","stream_handle","branches","code_ids","retain_prefix","release_prefix","chunk_rows"});
    Request r; r.identity = identity_json(required(j,"identity"));
    if (const auto* p = j.find("prefix_ids")) r.prefix = ids(*p);
    if (const auto* p = j.find("messages")) {
        if (r.prefix || !adapter) throw std::invalid_argument("messages require tokenizer and no prefix_ids");
        auto input = gemma4::serve::chat_input(*p,j.find("chat_template_kwargs") ? *j.find("chat_template_kwargs") : Json::object());
        if (j.find("tools")) input.tools = *j.find("tools");
        r.prefix = adapter->prepare(input).ids;
    } else if (j.find("tools") || j.find("chat_template_kwargs")) throw std::invalid_argument("template options require messages");
    if (j.find("stream_handle")) r.stream_handle = required(j,"stream_handle").as_string();
    for (const auto& b : required(j,"branches").as_array()) {
        fields(b,{"id","token_ids","read_position","option_close_positions"});
        Branch branch; branch.id = required(b,"id").as_string(); branch.tokens = ids(required(b,"token_ids"));
        branch.read_position = number(required(b,"read_position"));
        for (const auto& p : required(b,"option_close_positions").as_array()) branch.option_close_positions.push_back(number(p));
        r.branches.push_back(std::move(branch));
    }
    if (j.find("code_ids")) r.code_ids = ids(required(j,"code_ids"));
    if (j.find("retain_prefix")) r.retain_prefix = required(j,"retain_prefix").as_bool();
    if (j.find("release_prefix")) r.release_prefix = required(j,"release_prefix").as_bool();
    if (j.find("chunk_rows")) r.chunk_rows = number(required(j,"chunk_rows"));
    return r;
}
} // namespace strix::gemma4::features
