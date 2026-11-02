#pragma once
// --http-serve: the OpenAI/Anthropic-compatible HTTP front end (strix_serve) over the MTP bridge's
// request runner. The bridge thread executes the generation queue; HTTP connections run on their
// own threads and only exchange tokens with it through tickets (docs/engine/serving/serving.md).
#include "hip_decode_options.hpp"
#include "http_session_completion.hpp"
#include "qwen_effective_config.hpp"
#include "hip_mtp_decode_bridge.hpp"
#include "rtx_state_import_routes.hpp"
#include "strix/serve/api.hpp"
#include "qwen_service_health.hpp"
#include "qwen_rotary_capabilities.hpp"
#include "qwen_moe_route_bf16.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::decode_cli {

inline void append_mtp_pool_receipt(serve::GenerationStats& stats, const MtpServeResult& result) {
    if (!result.qsa_mtp_pool_active) return;
    stats.engine_extra.emplace_back("mtp_paid_drafts",std::to_string(result.paid_drafts));
    std::string rows="{";
    for(std::size_t r=1;r<result.chosen_window_rows.size();++r)
        rows+=(r>1?",":"")+std::string("\"")+std::to_string(r)+"\":"+std::to_string(result.chosen_window_rows[r]);
    stats.engine_extra.emplace_back("mtp_chosen_window_rows_histogram",rows+"}");
}

inline void add_prefill_capabilities(serve::Json& out) {
    const auto fields = serve::parse_json("{" + prefill_identity_fields(prefill_options_active()) + "," + http_exact_prefill_fields() + "}");
    for (const auto& [key, value] : fields.as_object()) out.set(key, value);
    out.set("prefill_override_scope", serve::Json("server-level only"));
}

inline void add_prefill_stats(serve::GenerationStats& stats) {
    const auto fields = serve::parse_json("{" + prefill_identity_fields(prefill_options_active()) + "," + http_exact_prefill_fields() + "}");
    for (const auto& [key, value] : fields.as_object()) stats.engine_extra.emplace_back(key, value.dump());
}

// Used only on the bridge thread. The tokenizer-wide trie is built lazily once
// per server; mutable selection frontiers are created afresh for each request.
class HttpConstraintRequests {
public:
    HttpConstraintRequests(std::shared_ptr<const serve::BpeTokenizer> tokenizer,bool argmax_first)
        : tokenizer_(std::move(tokenizer)),argmax_first_(argmax_first) {}
    bool available(std::span<const std::uint32_t> stops) const {
        const auto id=tokenizer_->find("</think>");
        return id!=UINT32_MAX && tokenizer_->token_bytes(id)=="</think>" &&
            (tokenizer_->type(id)==serve::BpeTokenizer::TokenType::control ||
             tokenizer_->type(id)==serve::BpeTokenizer::TokenType::user_defined) &&
            std::find(stops.begin(),stops.end(),id)==stops.end();
    }
    std::shared_ptr<MtpConstraintRow> make(const std::shared_ptr<const serve::ConstraintSpec>& spec) {
        if(!spec) return {};
        if(!trie_) trie_=std::make_shared<serve::TokenByteTrie>(*tokenizer_,spec->eos_ids);
        auto selection=std::make_shared<serve::ConstraintSelection>(spec,trie_,tokenizer_);
        selection->set_argmax_first(argmax_first_);
        return std::make_shared<MtpConstraintRow>(std::move(selection));
    }
private:
    std::shared_ptr<const serve::BpeTokenizer> tokenizer_;
    std::shared_ptr<const serve::TokenByteTrie> trie_;
    bool argmax_first_;
};
// Execution failures stop the server; constraint refusals carry request-local receipts.
class MtpBridgeEngine final : public serve::GenerationEngine {
public:
    MtpBridgeEngine(const MtpServeRunner& run, std::uint64_t context_tokens,
        HttpConstraintRequests& constraints, bool enabled, bool thinking_enabled, bool selection_mtp_available,
        serve::BackendProbeSlot* probe = nullptr, strix::qwen::profile::Resolution profile = {})
        : run_(run), context_(context_tokens), constraints_(constraints), constraints_enabled_(enabled),
          thinking_enabled_(thinking_enabled), selection_mtp_available_(selection_mtp_available), probe_(probe), profile_(std::move(profile)) {}
    serve::GenerationStats generate(serve::GenerationJob& job) override {
        if (failed_.load()) throw std::runtime_error("the engine failed during an earlier request; restart the server");
        if (job.prompt.empty() || job.prompt.size() >= context_) throw std::invalid_argument("prompt is empty or fills the context");
        MtpServeRequest request;
        request.backend_probe = probe_;
        request.constraint = constraints_.make(job.constraint);
        request.on_admitted = job.on_admitted;
        request.session_id = job.session_id;
        request.cache_retention = job.cache_retention;
        request.prompt = job.prompt;
        request.max_tokens = job.max_tokens;
        request.stop_tokens = job.stop_tokens;
        request.ngram_draft = job.ngram_draft;
        request.on_tokens = job.on_tokens;
        request.on_progress = job.on_progress;
        MtpServeResult result;
        try {
            result = run_(request);
        } catch (const SessionApiError& e) {
            serve::GenerationStats stats;
            stats.engine_extra.emplace_back("profile", profile_.json());
            stats.prompt_tokens=job.prompt.size(); stats.finish="admission_refused";
            stats.error_status=e.status; stats.error_code=e.code;
            throw serve::GenerationRequestError(std::move(stats),e.what());
        } catch (...) {
            failed_.store(true);
            if (probe_) probe_->fail();
            if (!request.constraint) throw;
            serve::GenerationStats stats;
            stats.engine_extra.emplace_back("profile", profile_.json());
            add_prefill_stats(stats);
            stats.prompt_tokens = job.prompt.size();
            stats.finish = "execution_failure";
            stats.constraint = request.constraint->verdict(stats.finish);
            try { throw; }
            catch (const std::exception& e) { throw serve::GenerationRequestError(std::move(stats), e.what()); }
            catch (...) { throw serve::GenerationRequestError(std::move(stats), "engine failure"); }
        }
        serve::GenerationStats stats;
        stats.engine_extra.emplace_back("profile", profile_.json());
        add_prefill_stats(stats);
        stats.prompt_tokens = job.prompt.size();
        stats.cached_tokens = result.restored_tokens;
        stats.generated_tokens = result.generated_tokens;
        stats.cycles = result.cycles;
        stats.proposed = result.proposed;
        stats.accepted = result.accepted;
        append_mtp_pool_receipt(stats,result);
        stats.ngram_rows = result.ngram_rows;
        stats.ngram_accepted = result.ngram_accepted;
        stats.prefill_ms = result.prompt_ms;
        stats.decode_ms = result.decode_ms;
        stats.constraint = result.constraint;
        stats.finish = result.finish;
        stats.sampling = job.constraint && job.constraint->transform.active() ? "greedy-logit-transform-v1" : "greedy";
        if (!result.final_state_digest.empty()) {
            stats.engine_extra.push_back({"final_state_digest", "\"" + result.final_state_digest + "\""});
            stats.engine_extra.push_back({"final_state_position", std::to_string(result.final_state_position)});
            if (!result.final_state_components.empty()) stats.engine_extra.push_back({"final_state_components", result.final_state_components});
        }
        if (!result.prompt_state_digest.empty())
            stats.engine_extra.push_back({"prompt_state_digest", "\"" + result.prompt_state_digest + "\""});
        if (!result.cache_refusal.empty()) stats.engine_extra.emplace_back("state_cache_refusal", serve::Json(result.cache_refusal).dump());
        if (!result.session_json.empty()) stats.engine_extra.emplace_back("session", result.session_json);
        if (!result.rtx_import_json.empty()) stats.engine_extra.emplace_back("rtx_import", result.rtx_import_json);
        if(!result.error.empty()) throw serve::GenerationRequestError(std::move(stats),result.error);
        return stats;
    }
    [[nodiscard]] std::uint64_t context_tokens() const override { return context_; }
    [[nodiscard]] bool supports_thinking_budget() const override {return thinking_enabled_;}
    [[nodiscard]] bool supports_constraints() const override {return constraints_enabled_;}
    [[nodiscard]] std::vector<std::string> constraint_execution_routes() const override {return selection_mtp_available_?std::vector<std::string>{"mtp-host","serial-host"}:std::vector<std::string>{"serial-host"};}
    [[nodiscard]] bool failed() const noexcept { return failed_.load(); }
private:
    const MtpServeRunner& run_;
    std::uint64_t context_;
    HttpConstraintRequests& constraints_;
    bool constraints_enabled_;
    bool thinking_enabled_;
    bool selection_mtp_available_;
    std::atomic<bool> failed_{false};
    serve::BackendProbeSlot* probe_;
    strix::qwen::profile::Resolution profile_;
};

inline std::atomic<bool>& http_serve_stop_requested() {
    static std::atomic<bool> requested{false};
    return requested;
}
extern "C" inline void http_serve_signal(int) { http_serve_stop_requested().store(true); }

inline std::string read_http_api_key(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read --http-api-key-file");
    std::string key((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' ')) key.pop_back();
    if (key.empty()) throw std::invalid_argument("--http-api-key-file is empty");
    return key;
}

inline bool handle_session_route(bool enabled, const MtpServeRunner& run,
    const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
    if (enabled && (request.path == "/v1/sessions" || request.path == "/v1/cache/stats" || request.path.starts_with("/v1/sessions/"))) {
        try {
            MtpServeRequest command;
            if (request.method == "GET" && request.path == "/v1/sessions") command.session_operation = "list";
            else if (request.method == "GET" && request.path == "/v1/cache/stats") command.session_operation = "stats";
            else if (request.path.starts_with("/v1/sessions/")) {
                const auto rest = request.path.substr(13);
                const auto slash = rest.find('/');
                command.session_id = rest.substr(0, slash);
                validate_http_session_id(command.session_id);
                if (request.method == "DELETE" && slash == std::string::npos) command.session_operation = "delete";
                else if (request.method == "POST" && slash != std::string::npos) {
                    command.session_operation = rest.substr(slash + 1U);
                    if (command.session_operation != "checkpoint" && command.session_operation != "park" && command.session_operation != "resume")
                        throw std::invalid_argument("unknown session operation");
                    if (command.session_operation == "checkpoint") {
                        const auto body = serve::parse_request_json(request.body);
                        const auto* durability = body.find("durability");
                        if (!durability || !durability->is_string()) throw std::invalid_argument("durability must be ram|disk");
                        command.session_durability = durability->as_string();
                    }
                }
            }
            if (command.session_operation.empty()) { response.send(405, "application/json", "{}"); return true; }
            response.send(200, "application/json", run(command).session_json);
        } catch (const SessionApiError& error) {
            response.send(error.status, "application/json", session_error_json(error).dump());
        } catch (const std::invalid_argument& error) {
            response.send(400, "application/json", session_error_json(SessionApiError(400, "invalid_request", error.what())).dump());
        } catch (const std::exception& error) {
            const bool incompatible = std::string_view(error.what()).find("compatibility mismatch") != std::string_view::npos;
            SessionApiError classified(incompatible ? 409 : 422,
                incompatible ? "session_incompatible" : "session_corrupt", error.what());
            response.send(classified.status, "application/json", session_error_json(classified).dump());
        }
        return true;
    }
    return false;
}


// Serves until SIGINT/SIGTERM or an engine failure (rethrown after the server stops).
// rtx_import (--rtx-state-import): the /admin/rtx-import/* routes over these imports; null: no route.
inline MtpDecodeServe make_http_serve(const Options& options, std::string engine_description,
    std::shared_ptr<RtxStateImportRegistry> rtx_import = {},
    std::optional<StartupPreflight> startup = {}, serve::Json assets = {}, std::function<void()> before_listen = {}) {
    return [&options, engine_description = std::move(engine_description), rtx_import, startup, assets, before_listen](const MtpServeRunner& run,bool selection_mtp_available) {
        if(options.cache_readiness_recheck) options.cache_readiness_recheck();
        const auto tokenizer = serve::BpeTokenizer::load_gguf(options.model_path);
        serve::ApiConfig config;
        config.model_id = options.http_model_id;
        config.http_slots = options.http_batch ? options.http_batch : 1U;
        config.default_max_tokens = options.http_default_max_tokens;
        config.api_key = read_http_api_key(options.http_api_key_file);
        config.sampling_policy = options.http_sampling;
        config.default_temperature = options.http_default_temperature;
        config.default_top_k = options.http_default_top_k;
        config.default_top_p = options.http_default_top_p;
        config.default_min_p = options.http_default_min_p;
        config.responses = options.http_responses;
        config.capabilities = options.http_capabilities;
        config.capabilities_extra = [modes=session_modes_json(options), policy=cache_activation_json(options), &options](serve::Json& out) {
            out.set("profile", serve::parse_json(options.product_profile.json()));
            qwen_rotary_capabilities(out);
            out.set("cache_policy",policy);
            prefix_cache_capabilities(out,options);
            moe_route_bf16_capabilities(out);
            add_prefill_capabilities(out);
            out.set("session_numerical_modes",modes);
        };
        config.stream_receipts = options.http_stream_receipts;
        config.sse_keepalive_seconds = options.http_sse_keepalive;
        config.sse_progress_seconds = options.http_sse_progress_interval;
        config.engine_progress = true;
        config.engine_description = engine_description;
        config.sessions_enabled = options.session_management_requested;
        if (options.session_management_requested || options.prefix_cache_host_bytes) {
            if (options.session_management_requested) {
                MtpServeRequest stats; stats.session_operation="stats"; (void)run(stats);
            }
            config.state_cache_stats=[mirror=options.state_cache_stats,policy=cache_activation_json(options)] {
                auto out=serve::parse_json(mirror->read());out.set("cache_policy",policy);return out.dump();
            };
        }
        config.stop_tokens.push_back(tokenizer->eos_token());
        if (const auto end = tokenizer->find("<|endoftext|>"); end != UINT32_MAX) config.stop_tokens.push_back(end);
        if (options.eos_token) config.stop_tokens.push_back(*options.eos_token);
        std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
        config.stop_tokens.erase(std::unique(config.stop_tokens.begin(), config.stop_tokens.end()), config.stop_tokens.end());
        HttpConstraintRequests constraints(tokenizer,options.http_selection_fast);
        config.engine_logprobs = options.http_logprobs;
        config.engine_logit_transforms = options.http_logit_transforms;
        config.engine_constraints = options.http_constraints && constraints.available(config.stop_tokens);
        config.engine_thinking_budget = options.http_thinking_budget && constraints.available(config.stop_tokens);
        config.enforce_anthropic_thinking_budget = options.http_anthropic_thinking_budget_enforce;
        config.constraint_execution_routes=selection_mtp_available?std::vector<std::string>{"mtp-host","serial-host"}:std::vector<std::string>{"serial-host"};
        config.engine_routes = [rtx_import, &run, enabled = options.session_management_requested](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
            if (handle_session_route(enabled, run, request, response)) return true;
            return rtx_import && handle_rtx_state_import_route(*rtx_import, request, response);
        };

        if (serve::serve_health_record_enabled())
            config.health_record = http_health_record(options, 1, startup, assets);
        serve::GenerationQueue queue(options.http_max_queue);
        serve::BackendProbeSlot probe([&] { queue.wake_control(); });
        const bool probing = config.health_record && config.health_record->backend_probe_enabled;
        if (probing) {
            queue.set_idle_control([&] { probe.boundary(serve::BackendProof::control_only); });
            config.health_provider = [&] { return probe.snapshot(); };
        }
        MtpBridgeEngine engine(run, options.context_tokens, constraints, config.engine_constraints, config.engine_thinking_budget,
            selection_mtp_available, probing ? &probe : nullptr, options.product_profile);
        serve::ApiService api(config, tokenizer, queue, options.context_tokens, false);
        serve::HttpServerConfig http;
        http.bind_address = options.http_bind;
        http.port = static_cast<std::uint16_t>(options.http_port);
        http.max_connections = options.http_max_connections;
        // Import chunks of up to 256 MiB (opt-in route only; the default limit stays 64 MiB).
        if (rtx_import) http.max_body_bytes = std::max<std::uint64_t>(http.max_body_bytes, 256ULL << 20U);
        serve::HttpServer server(http, [&api](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
            api.handle(request, response);
        });
        if(before_listen) before_listen();
        server.listen();
        api.set_state("ready");
        std::cerr << "strix serve: listening on " << options.http_bind << ':' << server.port() << " model " << config.model_id << std::endl;

        http_serve_stop_requested().store(false);
        const auto previous_int = std::signal(SIGINT, http_serve_signal);
        const auto previous_term = std::signal(SIGTERM, http_serve_signal);
        std::atomic<bool> finished{false};
        std::thread http_thread([&server] { server.serve(); });
        std::thread watcher([&] {
            while (!finished.load() && !http_serve_stop_requested().load() && !engine.failed())
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (engine.failed()) api.set_state("failed");
            server.stop();
            queue.shutdown();
        });
        queue.run(engine);
        finished.store(true);
        watcher.join();
        http_thread.join();
        std::signal(SIGINT, previous_int);
        std::signal(SIGTERM, previous_term);
        if (engine.failed()) throw std::runtime_error("strix serve stopped after an engine failure");
        std::cerr << "strix serve: stopped" << std::endl;
    };
}

// --http-serve --http-batch N: the same front end over the batched runner. The bridge thread admits
// queued requests into free slots between decode cycles (each prompt runs as a single request's would)
// and decodes every active request together; a request leaves when it ends, without stopping the others.
// With --http-sampling sample, a request with temperature > 0 samples (sampling-v1) with its own policy
// and seed.
inline MtpDecodeBatchServe make_http_batch_serve(const Options& options, std::string engine_description,
    std::optional<StartupPreflight> startup = {}, serve::Json assets = {},
    std::shared_ptr<const serve::BpeTokenizer> supplied_tokenizer = {}, std::function<void()> before_listen = {}) {
    return [&options, engine_description = std::move(engine_description), startup, assets,
        supplied_tokenizer = std::move(supplied_tokenizer), before_listen](MtpBatchRunner& runner) {
        if(options.cache_readiness_recheck) options.cache_readiness_recheck();
        const auto tokenizer = supplied_tokenizer ? supplied_tokenizer : serve::BpeTokenizer::load_gguf(options.model_path);
        serve::ApiConfig config;
        config.model_id = options.http_model_id;
        config.http_slots = options.http_batch ? options.http_batch : 1U;
        config.default_max_tokens = options.http_default_max_tokens;
        config.api_key = read_http_api_key(options.http_api_key_file);
        config.sampling_policy = options.http_sampling;
        config.default_temperature = options.http_default_temperature;
        config.default_top_k = options.http_default_top_k;
        config.default_top_p = options.http_default_top_p;
        config.default_min_p = options.http_default_min_p;
        config.responses = options.http_responses;
        config.capabilities = options.http_capabilities;
        auto cert=serve::Json::object();
        prefix_batch_cert_capabilities(cert,options);
        cert.set("cache_policy",cache_activation_json(options));
        prefix_cache_capabilities(cert,options);
        config.capabilities_extra = [modes=session_modes_json(options), cert=std::move(cert), profile=options.product_profile](serve::Json& out) {
            out.set("profile", serve::parse_json(profile.json()));
            qwen_rotary_capabilities(out);
            moe_route_bf16_capabilities(out);
            add_prefill_capabilities(out);
            out.set("session_numerical_modes",modes);
            for(const auto& [key,value]:cert.as_object()) out.set(key,value);
        };
        config.stream_receipts = options.http_stream_receipts;
        config.sse_keepalive_seconds = options.http_sse_keepalive;
        config.sse_progress_seconds = options.http_sse_progress_interval;
        config.engine_progress = true;
        config.engine_description = engine_description;
        config.sessions_enabled = options.session_management_requested;
        if (options.session_management_requested || options.prefix_cache_host_bytes) {
            if (options.session_management_requested) {
                MtpServeRequest stats; stats.session_operation="stats"; (void)runner.session_run(stats);
            }
            config.state_cache_stats=[mirror=options.state_cache_stats,policy=cache_activation_json(options)] {
                auto out=serve::parse_json(mirror->read());out.set("cache_policy",policy);return out.dump();
            };
        }
        config.engine_routes = [&runner, enabled = options.session_management_requested](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
            return handle_session_route(enabled, runner.session_run, request, response);
        };
        config.stop_tokens.push_back(tokenizer->eos_token());
        if (const auto end = tokenizer->find("<|endoftext|>"); end != UINT32_MAX) config.stop_tokens.push_back(end);
        if (options.eos_token) config.stop_tokens.push_back(*options.eos_token);
        std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
        config.stop_tokens.erase(std::unique(config.stop_tokens.begin(), config.stop_tokens.end()), config.stop_tokens.end());
        HttpConstraintRequests constraints(tokenizer,options.http_selection_fast);
        config.engine_logprobs = options.http_logprobs;
        config.engine_logit_transforms = options.http_logit_transforms;
        config.engine_constraints = runner.supports_constraints() && constraints.available(config.stop_tokens);
        config.engine_thinking_budget = runner.supports_thinking_budget() && constraints.available(config.stop_tokens);
        config.enforce_anthropic_thinking_budget = options.http_anthropic_thinking_budget_enforce;
        config.constraint_execution_routes=runner.constraint_execution_routes();
        const bool samples = options.http_sampling == "sample";
        if (serve::serve_health_record_enabled())
            config.health_record = http_health_record(options, runner.slots(), startup, assets);
        // Strict tool-call frame proposals (batch default on, STRIX_BATCH_TOOL_DRAFT=0 off) need the template's frame tokens.
        if (const auto open = tokenizer->find("<tool_call>"), close = tokenizer->find("</tool_call>");
            open != UINT32_MAX && close != UINT32_MAX && open != close)
            runner.frame_tokens = {open, close};

        serve::GenerationQueue queue(options.http_max_queue);
        serve::BackendProbeSlot probe([&] { queue.wake_control(); });
        const bool probing = config.health_record && config.health_record->backend_probe_enabled;
        if (probing) {
            queue.set_idle_control([&] { probe.boundary(serve::BackendProof::control_only); });
            config.health_provider = [&] { return probe.snapshot(); };
        }
        serve::ApiService api(config, tokenizer, queue, options.context_tokens, samples);
        serve::HttpServerConfig http;
        http.bind_address = options.http_bind;
        http.port = static_cast<std::uint16_t>(options.http_port);
        http.max_connections = options.http_max_connections;
        serve::HttpServer server(http, [&api](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
            api.handle(request, response);
        });
        if(before_listen) before_listen();
        server.listen();
        api.set_state("ready");
        std::cerr << "strix serve: listening on " << options.http_bind << ':' << server.port() << " model " << config.model_id
                  << " batch " << runner.slots() << std::endl;

        http_serve_stop_requested().store(false);
        const auto previous_int = std::signal(SIGINT, http_serve_signal);
        const auto previous_term = std::signal(SIGTERM, http_serve_signal);
        std::atomic<bool> finished{false}, failed{false};
        std::thread http_thread([&server] { server.serve(); });
        std::thread watcher([&] {
            while (!finished.load() && !http_serve_stop_requested().load() && !failed.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (failed.load()) api.set_state("failed");
            server.stop();
            queue.shutdown();
        });
        // Jobs in the slots, finished by the runner (on_finish) or, after an engine failure, here.
        std::vector<std::shared_ptr<serve::TakenJob>> live;
        const auto release = [&](const std::shared_ptr<serve::TakenJob>& taken) {
            live.erase(std::remove(live.begin(), live.end(), taken), live.end());
        };
        std::unique_lock<std::mutex> batch_lock;
        if (runner.session_mutex) batch_lock = std::unique_lock<std::mutex>(*runner.session_mutex, std::defer_lock);
        const auto admit = [&](serve::TakenJob taken_job,const MtpBatchRunner::AdmissionSink& submit) {
            auto taken = std::make_shared<serve::TakenJob>(std::move(taken_job));
            live.push_back(taken);
            auto& job = taken->job;
            if (job.prompt.empty() || job.prompt.size() >= options.context_tokens) {
                queue.finish(*taken, {}, "prompt is empty or fills the context");
                release(taken);
                return;
            }
            if (!job.session_id.empty()) {
                if (!runner.session_run || runner.active()) throw std::logic_error("named session requires an idle session runner");
                if (batch_lock.owns_lock()) batch_lock.unlock();
                // Named sessions run the fixed session protocol: no constrained decoding on that path.
                MtpBridgeEngine engine(runner.session_run, options.context_tokens, constraints, false, false, false, probing ? &probe : nullptr, options.product_profile);
                finish_http_session_job(queue, *taken, engine, [&] { release(taken); });
                return;
            }
            MtpBatchRequest request;
            request.constraint = constraints.make(job.constraint);
            request.cache_retention = job.cache_retention;
            request.prompt = job.prompt;
            request.max_tokens = job.max_tokens;
            request.stop_tokens = job.stop_tokens;
            request.on_tokens = job.on_tokens;
            request.on_progress = job.on_progress;
            request.cancelled = [ticket=taken->ticket] {return ticket->cancelled() || http_serve_stop_requested().load();};
            request.ngram_draft = job.ngram_draft;
            request.draft_depth_max = job.draft_depth_max;
            request.tool_draft = job.tool_draft;
            request.ngram_peers = job.ngram_peers;
            std::string sampling = job.constraint && job.constraint->transform.active() ? "greedy-logit-transform-v1" : "greedy";
            if (samples && job.sampling.temperature > 0.0) {
                TokenSamplingPolicy policy;
                policy.temperature = static_cast<float>(job.sampling.temperature);
                policy.top_k = job.sampling.top_k;
                policy.top_p = static_cast<float>(job.sampling.top_p);
                policy.min_p = static_cast<float>(job.sampling.min_p);
                policy.seed = job.sampling.seed;
                try {
                    validate_token_sampling_policy(policy);
                    request.sampler = std::make_shared<TokenSampler>(policy);
                } catch (const std::exception& e) {  // this request fails alone; the server keeps serving
                    queue.finish(*taken, {}, std::string("sampling policy refused: ") + e.what());
                    release(taken);
                    return;
                }
                sampling = "sampling-v1";
            }
            const auto prompt_tokens = job.prompt.size();
            request.on_finish = [&queue, &release, taken, prompt_tokens, sampling, profile=options.product_profile](const MtpServeResult& result) {
                serve::GenerationStats stats;
                stats.engine_extra.emplace_back("profile", profile.json());
                add_prefill_stats(stats);
                stats.prompt_tokens = prompt_tokens;
                stats.cached_tokens = result.restored_tokens;
                stats.generated_tokens = result.generated_tokens;
                stats.cycles = result.cycles;
                stats.proposed = result.proposed;
                stats.accepted = result.accepted;
                append_mtp_pool_receipt(stats,result);
                stats.prefill_ms = result.prompt_ms;
                stats.decode_ms = result.decode_ms;
                stats.ngram_rows = result.ngram_rows;
                stats.ngram_accepted = result.ngram_accepted;
                stats.tool_rows = result.tool_rows;
                stats.tool_accepted = result.tool_accepted;
                stats.source_shadow = result.source_shadow;
                stats.source_scored = result.source_scored;
                stats.source_right = result.source_right;
                stats.frame_cycles = result.frame_cycles;
                stats.frame_shadow = result.frame_shadow;
                stats.constraint = result.constraint;
                if (!result.cache_refusal.empty()) stats.engine_extra.emplace_back("state_cache_refusal", serve::Json(result.cache_refusal).dump());
                stats.engine_extra.push_back({"batch_service","{"+batch_service_receipt(result.service)+"}"});
                stats.engine_extra.push_back({"shared_execution",result.shared_execution.json()});
                if(!result.final_state_digest.empty()) {
                    stats.engine_extra.push_back({"final_state_digest","\""+result.final_state_digest+"\""});
                    if(!result.final_state_components.empty()) stats.engine_extra.push_back({"final_state_components",result.final_state_components});
                    stats.engine_extra.push_back({"final_state_position",std::to_string(result.final_state_position)});
                }
                if(!result.prompt_state_digest.empty()) stats.engine_extra.push_back({"prompt_state_digest","\""+result.prompt_state_digest+"\""});
                stats.finish = result.finish;
                stats.sampling = sampling;
                queue.finish(*taken, std::move(stats), result.error);
                release(taken);
            };
            if (runner.session_mutex && !batch_lock.owns_lock()) batch_lock.lock();
            submit(std::move(request));
        };
        std::optional<serve::TakenJob> deferred;
        runner.set_service_admission([&]{return !deferred && queue.has_queued_work();},[&](const MtpBatchRunner::AdmissionSink& prepare,const std::function<bool()>& can_prepare) {
            while(!deferred && can_prepare()) {
                auto job=queue.take(false);
                if(!job) break;
                // A named turn must wait until every ordinary slot is idle, including nested work.
                if (!job->job.session_id.empty()) { deferred = std::move(job); break; }
                admit(std::move(*job),prepare);
            }
        });
        struct ClearAdmission {
            MtpBatchRunner& runner;
            ~ClearAdmission(){runner.set_service_admission({},{});}
        } clear_admission{runner};
        std::string failure;
        try {
            for (;;) {
                runner.add_slot(queue.snapshot().queued != 0U);
                while (runner.has_free_slot()) {
                    if (deferred && runner.active()) break;
                    auto job = deferred ? std::move(deferred) : queue.take(runner.active() == 0U);
                    deferred.reset();
                    if (!job) break;
                    if (!job->job.session_id.empty() && runner.active()) { deferred = std::move(job); break; }
                    admit(std::move(*job),[&](MtpBatchRequest request){runner.admit(std::move(request));});
                }
                // The admitted prompt's first unit runs in this step: build the spare slot now so a request arriving
                // during that unit finds a free slot (nested admission).
                runner.add_slot(queue.snapshot().queued != 0U);
                if (!runner.active()) {
                    if (queue.stopping()) break;
                    continue;
                }
                const auto generation = probing ? probe.pending_generation() : 0;
                if (generation) {
                    const bool completed = runner.step<true>();
                    if (completed) probe.complete(generation, serve::BackendProof::work_completion);
                } else runner.step();
                if (!runner.active() && batch_lock.owns_lock()) batch_lock.unlock();
            }
        } catch (const std::exception& e) {
            failure = e.what()[0] ? e.what() : "engine failure";
        } catch (...) {
            failure = "engine failure";
        }
        if (!failure.empty()) {
            // The target state of every slot is unknown: fail the requests in flight and stop.
            failed.store(true);
            if (probing) probe.fail();
            runner.fail_requests(failure);
            for (auto& taken : std::vector<std::shared_ptr<serve::TakenJob>>(live)) queue.finish(*taken, {}, failure);
            live.clear();
            if (deferred) { queue.finish(*deferred, {}, failure); deferred.reset(); }
            runner.abandon();
        }
        if (batch_lock.owns_lock()) batch_lock.unlock();
        finished.store(true);
        watcher.join();
        http_thread.join();
        std::signal(SIGINT, previous_int);
        std::signal(SIGTERM, previous_term);
        if (!failure.empty()) throw std::runtime_error("strix serve stopped after an engine failure: " + failure);
        std::cerr << "strix serve: stopped" << std::endl;
    };
}

}  // namespace strix::decode_cli
