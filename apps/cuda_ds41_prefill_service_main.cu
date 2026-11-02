// strix-cuda-ds41-prefill-service: the deepseek41 CUDA prefill (cuda_ds41_prefill.hpp) resident behind a small HTTP/1.1
// server. The model loads once; every request is one prefill of a fresh sequence, its attention state exported and,
// when asked, handed to a decode server (POST /admin/import-state of strix-deepseek41-serve).
//
//   strix-cuda-ds41-prefill-service MODEL.gguf [--host 127.0.0.1] [--port 8095] [--positions N]
//       [--vram-cap-gib G] [--vram-reserve-gib G] [--no-auto-budget [--device-gib G | --tiers DHF..]] [--no-expert-residency]
//       [--expert-rank FILE] [--xfer-group N] [--xfer-piece-mib N] [--host-gib G | --host-gib-auto G (opportunistic, e.g. 32)] [--overlay OVERLAY.gguf] [--host-pin-order layer|deadline|early]
//       [--host-commit-floor-gib F (20 with --host-gib-auto)] [--host-pin-wait-s S] [--slots S] [--chunk C] [--moe-chunk M]
//       [--no-host-promote | --host-promote] [--host-promote-max-gib G] [--baseline]
//       [--window W] [--io-threads N] [--engram-threads N] [--buffered] [--engram-upload mapped|stream]
//       [--engram-cache-mib N] [--no-gpu-probe] [--device D] [--layers L]
//       [--exact | --no-fast-dense | --no-fast-moe | --no-fast-attention] [--fast-attention-min-rows N] [--fast-moe-signed-lut] [--fast-moe-wide-tiles]
//       [--replay-rows 0|256]  (approximate bounded decoder replay; default 0)
//       [--max-queue N] [--api-key KEY] [--handoff-api-key KEY] [--handoff-timeout S]
//       [--timing] [--pipeline [--handoff-queue N] | --fast-export [--export-check] [--export-sha-after]]
//       [--batch-max B [--batch-sum-tokens S] [--batch-alone-rows N] [--batch-ctx-positions P] [--batch-max-overtakes K]]
//
// Fast-arm policy (the qualified one): --fast-dense and --fast-moe on every request, --fast-attention on prompts of at
// least --fast-attention-min-rows rows (default 8192; its buffers exist only when --positions reaches it). --exact:
// none (the reference bits).
//
// Residency (defaults: --expert-residency --auto-budget): the hottest experts of every layer stay on the device, the
// rest stream; the device budget comes from the free memory after the dense weights and buffers, less
// --vram-reserve-gib (default 1), and under --vram-cap-gib G when given (everything the object allocates stays under
// G). --no-auto-budget: --device-gib / --tiers as given; --no-expert-residency: whole-layer tiers.
// Lazy host promotion (default ON; CudaPrefillOptions::host_promote): a file-tier expert layer read for a request is
// kept in a new pinned buffer and streams from host memory in the next requests (the same bytes), while the host keeps
// max(--host-commit-floor-gib, 20) GiB free and, with --host-promote-max-gib G, the promoted layers stay under G GiB;
// below the floor at a request start the least recently used promoted layers go back to the file. Off:
// --no-host-promote, STRIX_DS41_HOST_PROMOTE=0, --baseline / STRIX_DS41_RTX_BASELINE=1 (the service's default-ON
// levers off; strix/ds41_rtx_profile.hpp, app prefill-service; the last of a lever's options and --baseline decides).
// Streamer transport only (not with --no-expert-residency, --pass-rows).
// --layers L: layers [0, L) only (tests: no head, no cone; the next token is not the model's).
//
// Closed batches (--batch-max B > 1; default 1 = one prompt at a time): when the GPU frees, the worker takes the queue
// front and, if it is shorter than --batch-alone-rows (default 8192: 8K / 32K run alone), adds waiting prompts in arrival
// order while the batch stays within B prompts and --batch-sum-tokens rows (default 4096). A prompt may be passed by
// at most --batch-max-overtakes (default 2) later arrivals. No deliberate wait to form a batch. The batch runs
// layer-major (CudaPrefill::run_batch: one pass of the streamed expert weights for every member; each member's bits are
// its solo run's), then each member is exported and handed off in order. B - 1 extra contexts of
// --batch-ctx-positions rows (default sum / 2) are allocated at load (out of the resident experts).
//
// Endpoints (JSON; one request per connection, Connection: close):
//   GET  /health        {"status":"ok"|"failed","busy","queued","served","max_positions"}
//   GET  /v1/describe   {"describe":{tiers, budgets, device accounting, GPU},"service":{policy, queue, totals},
//                        "device_free_gb","device_total_gb"}
//   POST /v1/prefill    {"tokens":[...], "handoff":"http://HOST:PORT/admin/import-state" (optional), "session":any,
//                        "api_key":"..." (handoff bearer, else --handoff-api-key), "reserve_tokens":N (optional,
//                        X-State-Reserve-Tokens of the import)}
//       -> {"session","tokens","next_token","logits_fnv","state_bytes","sha256","digest","fast":{dense,moe,attention},
//           "seconds":{"queue","prefill","export","sha256","transfer","import","handoff","total"},
//           "prefill":{engram_s, expert_io_s, expert_gb_streamed, file_gb_streamed, host_promoted_layers,
//                      host_promoted_total, host_promoted_gb, host_demoted_layers, row_layers}, "import":<the import reply>|null,
//           "batch":{"size","index","prefill_s" (the batch's wall)}}  (seconds.prefill: until this member's logits)
//       --fast-export services take "export":"legacy"|"fast"|"check" per request (legacy: the old export, idle preload released first).
//       The prefills run one at a time in arrival order (a queue of at most --max-queue waiting requests; 503 when
//       full). transfer = the import POST's wall time minus the server's own time (its "timing.server_s" = import).
//       A failed handoff answers 502 with the prefill's fields and "error"; the state stays for /v1/handoff.
//   POST /v1/handoff    {"handoff":URL, "api_key", "reserve_tokens"}: posts the last exported state again.
//
// --timing: the prefill runs with device events (no per-op events) and every reply adds "accounting": the request's
//   wall time split into non-overlapping parts (run: host setup, device span = embed + per-layer compute + exposed wait
//   for expert copies + gaps + head, host tail; capture (export_state: D2H + re-slot), serialize, digest, sha256, post,
//   handoff queue), "explained_frac" = their sum / wall, the expert DMA of the run plus the between-runs preload it
//   consumed, and per-layer rows (compute, wait, copy). One "rtxq-acct" line per request on stderr.
// --pipeline: the GPU worker only runs the prefill and captures the state (export_state, host copies it owns); a
//   handoff worker serializes, hashes and POSTs it, fed by a FIFO of at most --handoff-queue (default 2) captured states
//   (each with its own id; the GPU worker blocks while the FIFO is full). The next prefill starts right after the capture.
//   Replies and bytes are the same as without it; seconds add "engine" (run + capture) and "handoff_queue".
//
// The import POST carries Content-Type application/octet-stream, X-State-Format strix/ds41-attention-state-v1 and
// X-State-Sha256 (the SHA-256 of the body); the reply must echo the same sha256 and the prompt's position.
// Events on stdout (one JSON line each): {"event":"loaded",...}, {"event":"READY",...}, {"event":"request",...},
// {"event":"STOPPED"}. SIGINT / SIGTERM stop accepting, finish the running prefill and exit.
#include "strix/cuda_ds41_prefill.hpp"
#include "strix/cuda_ds41_replay.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/json.hpp"

#include <cuda_runtime.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include "strix/ds41_memory_budget.hpp"
#include "strix/ds41_rtx_profile.hpp"
#include "strix/ds41_state_transfer.hpp"
#ifdef _WIN32
#include <bcrypt.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <charconv>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strix::ds41::cuda;
using strix::serve::Json;
using Clock = std::chrono::steady_clock;

namespace {

// ---------------------------------------------------------------------------------------------------- sockets
#ifdef _WIN32
using Sock = SOCKET;
const Sock kNoSock = INVALID_SOCKET;
void close_sock(Sock s) { closesocket(s); }
void set_timeouts(Sock s, int seconds) {
    const DWORD ms = static_cast<DWORD>(seconds) * 1000U;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
}
#else
using Sock = int;
const Sock kNoSock = -1;
void close_sock(Sock s) { ::close(s); }
void set_timeouts(Sock s, int seconds) {
    timeval tv{};
    tv.tv_sec = seconds;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
#endif

bool send_all(Sock s, const char* p, std::size_t n) {
    while (n > 0) {
        const int piece = static_cast<int>(std::min<std::size_t>(n, 1U << 20));
#ifdef _WIN32
        const int k = ::send(s, p, piece, 0);
#else
        const auto k = static_cast<int>(::send(s, p, static_cast<std::size_t>(piece), MSG_NOSIGNAL));
#endif
        if (k <= 0) return false;
        p += k;
        n -= static_cast<std::size_t>(k);
    }
    return true;
}

int recv_some(Sock s, char* p, std::size_t n) {
    return static_cast<int>(::recv(s, p, static_cast<int>(std::min<std::size_t>(n, 1U << 20)), 0));
}

std::string lower(std::string v) {
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return v;
}

std::string trim(const std::string& v) {
    std::size_t a = 0, b = v.size();
    while (a < b && (v[a] == ' ' || v[a] == '\t')) ++a;
    while (b > a && (v[b - 1] == ' ' || v[b - 1] == '\t' || v[b - 1] == '\r')) --b;
    return v.substr(a, b - a);
}

// One HTTP/1.1 message (request or response): start line, lower-case headers, Content-Length body.
struct Message {
    std::string start;
    std::map<std::string, std::string> headers;
    std::string body;
    [[nodiscard]] std::string header(const std::string& k) const {
        const auto it = headers.find(k);
        return it == headers.end() ? std::string() : it->second;
    }
};

// Reads one message; without Content-Length a response body runs to the connection's end (requests: no body).
// Returns an error text, empty on success.
std::string read_message(Sock s, Message& m, std::size_t max_body, bool is_response) {
    std::string buf;
    char tmp[65536];
    std::size_t head_end = std::string::npos;
    while ((head_end = buf.find("\r\n\r\n")) == std::string::npos) {
        if (buf.size() > 65536) return "headers too large";
        const int k = recv_some(s, tmp, sizeof(tmp));
        if (k <= 0) return buf.empty() ? "connection closed" : "truncated headers";
        buf.append(tmp, static_cast<std::size_t>(k));
    }
    const std::string head = buf.substr(0, head_end);
    std::size_t pos = head.find("\r\n");
    m.start = head.substr(0, pos);
    while (pos != std::string::npos) {
        const std::size_t next = head.find("\r\n", pos + 2);
        const std::string line = head.substr(pos + 2, next == std::string::npos ? std::string::npos : next - pos - 2);
        if (const auto colon = line.find(':'); colon != std::string::npos) m.headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
        pos = next;
    }
    m.body = buf.substr(head_end + 4);
    if (!m.header("transfer-encoding").empty() && lower(m.header("transfer-encoding")) != "identity") return "chunked bodies are not supported";
    const std::string cl = m.header("content-length");
    if (cl.empty()) {
        if (!is_response) return {};
        for (;;) {
            const int k = recv_some(s, tmp, sizeof(tmp));
            if (k <= 0) break;
            m.body.append(tmp, static_cast<std::size_t>(k));
            if (m.body.size() > max_body) return "body too large";
        }
        return {};
    }
    std::size_t want = 0;
    try {
        want = static_cast<std::size_t>(std::stoull(cl));
    } catch (const std::exception&) {
        return "bad Content-Length";
    }
    if (want > max_body) return "body too large";
    m.body.reserve(want);
    while (m.body.size() < want) {
        const int k = recv_some(s, tmp, std::min(sizeof(tmp), want - m.body.size()));
        if (k <= 0) return "truncated body";
        m.body.append(tmp, static_cast<std::size_t>(k));
    }
    m.body.resize(want);
    return {};
}

const char* reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

void respond(Sock s, int status, const std::string& body, const char* type = "application/json") {
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\nContent-Type: " + type +
                       "\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
    if (send_all(s, head.data(), head.size())) send_all(s, body.data(), body.size());
}

Json error_json(const std::string& why) {
    Json e = Json::object();
    e.set("error", Json(why));
    return e;
}

// ---------------------------------------------------------------------------------------------------- client
struct Url {
    std::string host, port = "80", path = "/";
};

Url parse_url(const std::string& u) {
    if (u.rfind("http://", 0) != 0) throw std::invalid_argument("handoff URL must start with http://");
    Url r;
    const std::string rest = u.substr(7);
    const auto slash = rest.find('/');
    std::string hostport = rest.substr(0, slash);
    if (slash != std::string::npos) r.path = rest.substr(slash);
    if (!hostport.empty() && hostport[0] == '[') {  // [v6]:port
        const auto close = hostport.find(']');
        if (close == std::string::npos) throw std::invalid_argument("bad handoff URL host");
        r.host = hostport.substr(1, close - 1);
        if (close + 1 < hostport.size() && hostport[close + 1] == ':') r.port = hostport.substr(close + 2);
    } else if (const auto colon = hostport.rfind(':'); colon != std::string::npos) {
        r.host = hostport.substr(0, colon);
        r.port = hostport.substr(colon + 1);
    } else {
        r.host = hostport;
    }
    if (r.host.empty()) throw std::invalid_argument("handoff URL without a host");
    return r;
}

Sock connect_to(const Url& u, int timeout_s) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(u.host.c_str(), u.port.c_str(), &hints, &res) != 0 || res == nullptr) throw std::runtime_error("cannot resolve " + u.host);
    Sock s = kNoSock;
    for (auto* a = res; a != nullptr; a = a->ai_next) {
        s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == kNoSock) continue;
        if (::connect(s, a->ai_addr, static_cast<int>(a->ai_addrlen)) == 0) break;
        close_sock(s);
        s = kNoSock;
    }
    freeaddrinfo(res);
    if (s == kNoSock) throw std::runtime_error("cannot connect to " + u.host + ":" + u.port);
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
    set_timeouts(s, timeout_s);
    return s;
}

// POST body to url; returns (status, reply body).
std::pair<int, std::string> http_post(const std::string& url, const std::vector<std::pair<std::string, std::string>>& headers,
                                      std::span<const std::uint8_t> body, int timeout_s) {
    const Url u = parse_url(url);
    const Sock s = connect_to(u, timeout_s);
    std::string head = "POST " + u.path + " HTTP/1.1\r\nHost: " + u.host + ":" + u.port + "\r\nConnection: close\r\nContent-Length: " +
                       std::to_string(body.size()) + "\r\n";
    for (const auto& [k, v] : headers) head += k + ": " + v + "\r\n";
    head += "\r\n";
    const bool sent = send_all(s, head.data(), head.size()) && send_all(s, reinterpret_cast<const char*>(body.data()), body.size());
    Message m;
    const std::string err = read_message(s, m, std::size_t{64} << 20, true);
    close_sock(s);
    if (!err.empty()) throw std::runtime_error(std::string(sent ? "" : "send failed; ") + "reply: " + err);
    const auto sp = m.start.find(' ');
    if (sp == std::string::npos) throw std::runtime_error("bad status line: " + m.start);
    return {std::atoi(m.start.c_str() + sp + 1), m.body};
}

// ---------------------------------------------------------------------------------------------------- service
std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

std::string hex16(std::uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
    return b;
}

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

struct Config {
    std::string host = "127.0.0.1";
    int port = 8095;
    bool fast_dense = true, fast_moe = true, fast_attention = true;
    std::uint32_t fast_attention_min_rows = 8192;
    std::size_t max_queue = 16;
    std::uint32_t batch_max = 1, batch_sum_tokens = 4096, batch_alone_rows = 8192, batch_ctx_positions = 0, batch_max_overtakes = 2;
    std::string api_key, handoff_api_key;
    int handoff_timeout = 600;
    bool timing = false;          // accounting in every reply (CudaPrefillOptions::timing without per-op events)
    bool pipeline = false;        // serialize + SHA + POST on a handoff worker
    std::size_t handoff_queue = 2;  // captured states waiting for the handoff worker
    // --fast-export: the state captured into a reusable pinned buffer already serialized (CudaPrefill::capture_state,
    // the streamer's idle preload held until then), sealed and hashed with the fast SHA-256 provider; same bytes.
    // --export-check: also the old export_state + serialize on every request, compared byte for byte (and the portable
    // SHA-256 against the fast one); reply "export_check".
    // --export-sha-after: the SHA-256 after the seal (each piece timed alone) instead of overlapped with it.
    bool fast_export = false, export_check = false, export_sha_after = false;
    int v2_timeout = 30;
    bool handoff_v2 = false;
    std::string state_identity, handoff_only;
    std::string levers_json = "null";  // the default-ON service levers as resolved (profile receipt)
};

struct Handoff {
    std::string url, api_key;
    std::optional<std::uint64_t> reserve_tokens;
};

// Only the opt-in v2 path uses the platform provider. v1 retains its existing seal path.
strix::ModelContentIdentity transfer_sha256(std::span<const std::byte> bytes) {
#ifdef _WIN32
    if (bytes.size() <= 0xffffffffULL) {
        strix::ModelContentIdentity result;
        std::uint8_t empty = 0;
        auto* input = bytes.empty() ? &empty : const_cast<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(bytes.data()));
        const auto status = BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0U, input, static_cast<ULONG>(bytes.size()),
                                       result.bytes.data(), static_cast<ULONG>(result.bytes.size()));
        if (status < 0) throw std::runtime_error("v2 platform SHA256 provider failed");
        return result;
    }
#endif
    return strix::sha256_bytes(bytes);
}

void check_transfer_sha256() {
    constexpr char abc[] = "abc";
    const auto empty = strix::model_content_identity_hex(transfer_sha256({}));
    const auto sample = strix::model_content_identity_hex(transfer_sha256(std::as_bytes(std::span(abc, 3))));
    if (empty != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ||
        sample != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        throw std::runtime_error("v2 SHA256 known-answer check failed");
}

class V2StateStream final : public CudaPrefillStateStream {
public:
    V2StateStream(Handoff handoff, const Config& config) : handoff_(std::move(handoff)), config_(config) {
        std::random_device random;
        id_ = hex16((std::uint64_t{random()} << 32U) | random()) + hex16((std::uint64_t{random()} << 32U) | random());
        base_ = handoff_.url;
        while (!base_.empty() && base_.back() == '/') base_.pop_back();
        base_ += "/v2/" + id_ + "/";
        worker_ = std::thread([this] { loop(); });
    }
    ~V2StateStream() override {
        { const std::lock_guard lock(mu_); stopping_ = true; }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    void begin(std::uint32_t position, std::span<const CudaPrefillStateSection> sections) override {
        position_ = position;
        Json manifest = Json::object(), list = Json::array();
        manifest.set("position", Json(std::uint64_t{position}));
        manifest.set("identity", Json(config_.state_identity));
        if (handoff_.reserve_tokens) manifest.set("reserve_tokens", Json(std::uint64_t{*handoff_.reserve_tokens}));
        for (const auto& section : sections) {
            expected_[section.name] = section.bytes;
            Json item = Json::object(); item.set("name", Json(section.name)); item.set("bytes", Json(section.bytes));
            list.push(std::move(item));
        }
        Json final = Json::object(); final.set("name", Json("final"));
        final.set("bytes", Json(std::uint64_t{position} * 4U + (4ULL << 20U)));
        list.push(std::move(final));
        manifest.set("sections", std::move(list));
        begun_ = true;  // even an unacknowledged begin may exist at the receiver and need aborting
        (void)json_request("begin", manifest);
    }
    std::future<void> chunk(std::string_view section, std::uint64_t offset, std::span<const std::uint8_t> bytes) override {
        auto task = std::make_unique<Task>();
        task->section = section; task->offset = offset; task->bytes = bytes;
        auto result = task->done.get_future();
        {
            const std::lock_guard lock(mu_);
            if (stopping_ || queue_.size() >= 2U) throw std::runtime_error("v2 transfer queue exceeds two buffers");
            queue_.push_back(std::move(task));
        }
        cv_.notify_one();
        return result;
    }
    Json finish(std::span<const std::uint8_t> bytes, const std::string& sha) {
        const auto plan = strix::ds41::plan_state_transfer_v1(bytes);
        if (plan.position != position_) throw std::runtime_error("v2 position changed");
        std::map<std::string, std::uint64_t> actual;
        std::vector<std::uint8_t> final;
        for (const auto& section : plan.sections) {
            if (section.name == "final") final.resize(static_cast<std::size_t>(section.bytes));
            else actual[section.name] = section.bytes;
        }
        // Empty caches generate no v1 payload slice; the begin manifest may declare zero capacity.
        for (const auto& [name, count] : expected_) if (!count) actual.try_emplace(name, 0U);
        if (actual != expected_) throw std::runtime_error("v2 immutable sections differ from final state");
        Json assembly = Json::object(), segments = Json::array();
        assembly.set("sha256", Json(sha));
        for (const auto& slice : plan.slices) {
            Json segment = Json::object();
            segment.set("section", Json(slice.section)); segment.set("offset", Json(slice.section_offset));
            segment.set("bytes", Json(slice.bytes)); segments.push(std::move(segment));
            if (slice.section == "final") std::memcpy(final.data() + slice.section_offset, bytes.data() + slice.file_offset,
                                                     static_cast<std::size_t>(slice.bytes));
        }
        for (std::size_t at = 0; at < final.size(); at += chunk_bytes) {
            const auto count = std::min(chunk_bytes, final.size() - at);
            chunk("final", at, {final.data() + at, count}).get();
        }
        assembly.set("segments", std::move(segments));
        (void)json_request("finalize", assembly);
        const auto valid = json_request("validate", Json::object());
        check_receipt(valid, sha);
        auto receipt = json_request("commit", Json::object());
        check_receipt(receipt, sha);
        const auto* committed = receipt.find("committed");
        if (!committed || !committed->is_bool() || !committed->as_bool()) throw std::runtime_error("v2 commit not acknowledged");
        return receipt;
    }
    Json upload_full(std::span<const std::uint8_t> bytes, const std::string& sha) {
        const auto plan = strix::ds41::plan_state_transfer_v1(bytes);
        std::vector<CudaPrefillStateSection> sections;
        for (const auto& section : plan.sections) if (section.name != "final") sections.push_back({section.name, section.bytes});
        begin(plan.position, sections);
        for (const auto& slice : plan.slices) if (slice.section != "final") {
            for (std::uint64_t at = 0; at < slice.bytes; at += chunk_bytes) {
                const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(chunk_bytes, slice.bytes - at));
                chunk(slice.section, slice.section_offset + at, bytes.subspan(static_cast<std::size_t>(slice.file_offset + at), count)).get();
            }
        }
        return finish(bytes, sha);
    }
    void abort() noexcept {
        if (begun_) { try { (void)json_request("abort", Json::object()); } catch (...) {} }
    }
    std::uint64_t acknowledged_bytes() const noexcept { return acknowledged_bytes_.load(); }
    double network_seconds() const noexcept { return double(network_ns_.load()) * 1e-9; }
private:
    struct Task {
        std::string section;
        std::uint64_t offset = 0;
        std::span<const std::uint8_t> bytes;
        std::promise<void> done;
    };
    static constexpr std::size_t chunk_bytes = 4U << 20U;
    Json request(const std::string& action, std::span<const std::uint8_t> bytes,
                 std::vector<std::pair<std::string, std::string>> headers) {
        const auto& key = handoff_.api_key.empty() ? config_.handoff_api_key : handoff_.api_key;
        if (!key.empty()) headers.emplace_back("Authorization", "Bearer " + key);
        std::pair<int, std::string> response;
        const auto started = Clock::now();
        for (unsigned attempt = 0;; ++attempt) {
            try { response = http_post(base_ + action, headers, bytes,
                      action == "abort" ? std::min(config_.v2_timeout, 5) : config_.v2_timeout); break; }
            catch (...) { if (attempt >= (action == "abort" ? 0U : 2U)) throw; }  // same id and bytes on retry
        }
        if (response.first != 200) throw std::runtime_error("v2 " + action + " HTTP " + std::to_string(response.first) + ": " + response.second.substr(0, 256));
        network_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count());
        return strix::serve::parse_json(response.second);
    }
    Json json_request(const std::string& action, const Json& value) {
        const auto body = value.dump();
        return request(action, {reinterpret_cast<const std::uint8_t*>(body.data()), body.size()}, {{"Content-Type", "application/json"}});
    }
    void check_receipt(const Json& receipt, const std::string& sha) {
        const auto* got = receipt.find("sha256"); const auto* position = receipt.find("position");
        if (!got || !got->is_string() || got->as_string() != sha || !position || !position->is_number() ||
            position->as_int() != position_) throw std::runtime_error("v2 receipt SHA/position differs");
    }
    void loop() {
        for (;;) {
            std::unique_ptr<Task> task;
            {
                std::unique_lock lock(mu_);
                cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) return;
                task = std::move(queue_.front()); queue_.pop_front();
            }
            try {
                const auto sha = strix::model_content_identity_hex(transfer_sha256(std::as_bytes(task->bytes)));
                (void)request("chunk", task->bytes, {{"Content-Type", "application/octet-stream"}, {"X-State-Section", task->section},
                    {"X-State-Offset", std::to_string(task->offset)}, {"X-State-Sha256", sha}});
                acknowledged_bytes_.fetch_add(task->bytes.size());
                task->done.set_value();
            } catch (...) { task->done.set_exception(std::current_exception()); }
        }
    }
    Handoff handoff_;
    const Config config_;
    std::string id_, base_;
    std::uint32_t position_ = 0;
    std::map<std::string, std::uint64_t> expected_;
    bool begun_ = false, stopping_ = false;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<Task>> queue_;
    std::thread worker_;
    std::atomic<std::uint64_t> acknowledged_bytes_{0};
    std::atomic<std::int64_t> network_ns_{0};
};

struct Job {
    std::vector<std::uint32_t> tokens;
    std::optional<Handoff> handoff;
    Json session;
    std::string export_mode;  // "legacy" | "fast" | "check" (--fast-export services; empty: the service's flags)
    Clock::time_point queued;
    std::uint32_t overtaken = 0;  // later arrivals batched ahead of it
    bool cancel_requested = false;  // protected by Service::mu_
    std::promise<std::pair<int, std::string>> done;
};

// A prefill's state captured on the host (it owns every byte; nothing on the device refers to it): what the GPU
// worker hands to the handoff stage. Immutable once queued.
struct Captured {
    std::uint64_t id = 0;              // capture order (1, 2, ..): its own identity, not the last state's
    std::unique_ptr<Job> job;
    CudaPrefillResult res;
    strix::deepseek41::AttentionState state;
    bool fa = false, want_fa = false;
    Clock::time_point t_start, t_run_end, t_captured, t_dequeued;
    double run_s = 0.0, capture_s = 0.0;
    // --fast-export: "fast" / "check" leave the bytes in the service's capture buffer (state stays empty for "fast");
    // "check" also keeps the old export (state) and its serialization (legacy)
    std::string mode = "legacy";
    std::vector<std::uint8_t> legacy;
    double legacy_export_s = 0.0, legacy_serialize_s = 0.0;
    // closed batches (--batch-max): this member's index and the batch's size (1: a solo prefill)
    std::size_t index = 0, batch_size = 1;
    // --handoff-v2 (--pass-rows): the request's state stream, fed during the run, finished by the handoff
    std::shared_ptr<V2StateStream> streaming;
};

std::atomic<bool> g_stop{false};
extern "C" void on_signal(int) { g_stop.store(true); }

class Service {
public:
    Service(CudaPrefill& p, const CudaPrefillOptions& o, Config c) : p_(p), o_(o), c_(std::move(c)) {
        if (c_.handoff_queue == 0U) c_.handoff_queue = 1U;
        capture_ = p_.export_capture();
        pinned_snapshot_.store(p_.pinned_bytes() + (capture_ ? capture_->pinned_bytes() : 0));
        worker_ = std::thread([this] { loop(); });
        if (c_.pipeline) handoff_worker_ = std::thread([this] { handoff_loop(); });
    }
    ~Service() {
        {
            const std::lock_guard lock(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();  // the GPU worker drains its queue first
        {
            const std::lock_guard lock(mu_);
            handoff_quit_ = true;
        }
        hcv_.notify_all();
        if (handoff_worker_.joinable()) handoff_worker_.join();
    }

    // Queues the job; empty optional when the queue is full.
    std::optional<std::future<std::pair<int, std::string>>> submit(std::unique_ptr<Job> job, std::string& refused) {
        auto f = job->done.get_future();
        {
            const std::lock_guard lock(mu_);
            if (failed_ || quit_) { refused = "the service failed or is stopping"; return std::nullopt; }
            if (queue_.size() >= c_.max_queue) { refused = "the queue is full"; return std::nullopt; }
            if (o_.pass_rows && !strix::ds41::host_commit_fits(strix::ds41::host_memory_room(),
                    strix::ds41::long_request_host_reserve(static_cast<std::uint32_t>(job->tokens.size())),
                    static_cast<std::uint64_t>(o_.host_commit_floor_gib * double(1ULL << 30)))) {
                refused = "insufficient free commit for request reserve";
                return std::nullopt;
            }
            queue_.push_back(std::move(job));
        }
        cv_.notify_one();
        return f;
    }

    Json host_memory_snapshot() const {
        Json h = Json::object();
        const auto& pins = p_.host_pin_stats(); // immutable after construction
        h.set("policy", Json(o_.host_tier_implicit ? "implicit" : "explicit"));
        h.set("budget_bytes", Json(o_.host_expert_gib * double(1ULL << 30)));
        h.set("requested_bytes", Json(pins.requested_bytes));
        h.set("effective_bytes", Json(pins.effective_bytes));
        h.set("floor_gib", Json(o_.host_commit_floor_gib));
        h.set("wait_s", Json(o_.host_pin_wait_s));
        h.set("effective_wait_s", Json(o_.host_tier_implicit ? 0.0 : o_.host_pin_wait_s));
        h.set("total_service_pinned_bytes", Json(pinned_snapshot_.load()));
        Json layers = Json::array(), demotions = Json::array();
        for (auto l : pins.layers) layers.push(Json(std::uint64_t{l}));
        for (auto d : pins.demotions) {
            Json v = Json::object();
            v.set("layer", Json(std::uint64_t{d.layer}));
            v.set("reason", Json(strix::ds41::host_pin_reason_name(d.reason)));
            demotions.push(std::move(v));
        }
        h.set("host_layers", std::move(layers));
        h.set("demotion_count", Json(std::uint64_t{pins.demotions.size()}));
        h.set("demotions", std::move(demotions));
        const auto room = strix::ds41::host_memory_room();
        h.set("free_physical_bytes", room.known ? Json(room.free_physical) : Json("unknown"));
        h.set("free_commit_bytes", room.known ? Json(room.free_commit) : Json("unknown"));
        const double floor = o_.host_commit_floor_gib * double(1ULL << 30);
        h.set("pressure", Json(!room.known ? "unknown" :
            (double(room.free_commit) < floor || double(room.free_physical) < floor + double(4ULL << 30)) ? "low" : "ok"));
        return h;
    }

    Json health() {
        const std::lock_guard lock(mu_);
        Json h = Json::object();
        h.set("host_tier", host_memory_snapshot());
        h.set("status", Json(failed_ ? "failed" : "ok"));
        h.set("busy", Json(busy_));
        h.set("queued", Json(static_cast<std::uint64_t>(queue_.size())));
        h.set("handoff_pending", Json(static_cast<std::uint64_t>(handoff_q_.size() + (handoff_busy_ ? 1U : 0U))));
        h.set("served", Json(served_));
        h.set("batches", Json(batches_));
        h.set("max_positions", Json(static_cast<std::uint64_t>(o_.max_positions)));
        if (o_.pass_rows) {
            const auto p = p_.progress();
            h.set("position", Json(static_cast<std::uint64_t>(p.position)));
            h.set("total", Json(static_cast<std::uint64_t>(p.total)));
            h.set("elapsed_s", Json(p.elapsed_seconds));
            h.set("cancel_pending", Json(active_ && active_->cancel_requested));
            h.set("eta_s", p.position && p.elapsed_seconds > 0.0 ?
                  Json(p.elapsed_seconds * double(p.total - p.position) / p.position) : Json(nullptr));
            const auto room = strix::ds41::host_memory_room();
            h.set("host_free_commit_bytes", room.known ? Json(room.free_commit) : Json(nullptr));
        }
        return h;
    }

    std::pair<int, std::string> cancel(const Json& session) {
        const std::lock_guard lock(mu_);
        if (!o_.pass_rows) return {409, error_json("cancellation requires --pass-rows").dump()};
        if (session.is_null()) return {400, error_json("a non-null session is required").dump()};
        const auto key = session.dump();
        for (auto it = queue_.begin(); it != queue_.end(); ++it) {
            if ((*it)->session.dump() != key) continue;
            (*it)->done.set_value({409, error_json("prefill cancelled while queued").dump()});
            queue_.erase(it);
            return {200, "{\"cancelled\":\"queued\"}"};
        }
        if (active_ && active_->session.dump() == key) {
            if (!active_->cancel_requested) {
                active_->cancel_requested = true;
                p_.request_cancel();
            }
            return {200, "{\"cancelled\":\"pending_pass_boundary\"}"};
        }
        return {404, error_json("session is not queued or active").dump()};
    }

    Json describe() {
        Json out = Json::object();
        std::string d;
        std::size_t device_bytes = 0, contexts = 0;
        double load_s = 0;
        {
            const std::lock_guard lock(run_mu_);  // describe() reads the object the worker runs
            d = p_.describe();
            device_bytes = p_.device_bytes();
            contexts = p_.batch_contexts();
            load_s = p_.load_seconds();
        }
        try {
            out.set("describe", strix::serve::parse_json(d));
        } catch (const std::exception&) {
            std::replace(d.begin(), d.end(), '\\', '/');
            out.set("describe", strix::serve::parse_json(d));
        }
        out.set("host_tier", host_memory_snapshot());
        Json s = Json::object();
        s.set("fast_dense", Json(c_.fast_dense));
        s.set("fast_moe", Json(c_.fast_moe));
        s.set("fast_attention", Json(c_.fast_attention));
        s.set("fast_attention_min_rows", Json(static_cast<std::uint64_t>(c_.fast_attention_min_rows)));
        s.set("fast_attention_available", Json(o_.fast_attention));
        s.set("max_positions", Json(static_cast<std::uint64_t>(o_.max_positions)));
        s.set("expert_residency", Json(o_.expert_residency));
        s.set("auto_budget", Json(o_.auto_budget));
        s.set("vram_cap_gib", Json(o_.vram_cap_gib));
        s.set("host_promote", Json(o_.host_promote));
        s.set("host_promote_max_gib", Json(o_.host_promote_max_gib));
        s.set("levers", strix::serve::parse_json(c_.levers_json));
        s.set("max_queue", Json(static_cast<std::uint64_t>(c_.max_queue)));
        s.set("timing", Json(c_.timing));
        s.set("pipeline", Json(c_.pipeline));
        s.set("handoff_queue", Json(static_cast<std::uint64_t>(c_.handoff_queue)));
        s.set("batch_max", Json(static_cast<std::uint64_t>(c_.batch_max)));
        s.set("batch_sum_tokens", Json(static_cast<std::uint64_t>(c_.batch_sum_tokens)));
        s.set("batch_alone_rows", Json(static_cast<std::uint64_t>(c_.batch_alone_rows)));
        s.set("batch_max_overtakes", Json(static_cast<std::uint64_t>(c_.batch_max_overtakes)));
        s.set("batch_contexts", Json(static_cast<std::uint64_t>(contexts)));
        s.set("load_s", Json(load_s));
        s.set("device_gb", Json(double(device_bytes) / 1e9));
        s.set("pinned_gb", Json(double(pinned_snapshot_.load()) / 1e9));
        {
            const std::lock_guard lock(mu_);
            s.set("served", Json(served_));
            s.set("failed", Json(failed_));
            s.set("busy", Json(busy_));
            s.set("queued", Json(static_cast<std::uint64_t>(queue_.size())));
            s.set("prefill_seconds_total", Json(prefill_total_));
            s.set("batches", Json(batches_));
        }
        out.set("service", std::move(s));
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            out.set("device_free_gb", Json(double(free_b) / 1e9));
            out.set("device_total_gb", Json(double(total_b) / 1e9));
        }
        return out;
    }

    // POST of the last exported state (a retry of a failed handoff, or a second decode group).
    std::pair<int, std::string> handoff_last(const Handoff& h) {
        std::shared_ptr<const std::vector<std::uint8_t>> bytes;
        std::string sha;
        std::uint32_t position = 0;
        {
            const std::lock_guard lock(mu_);
            bytes = last_state_;
            sha = last_sha_;
            position = last_position_;
        }
        if (!bytes) return {409, error_json("no exported state yet").dump()};
        Json out = Json::object();
        out.set("sha256", Json(sha));
        out.set("state_bytes", Json(static_cast<std::uint64_t>(bytes->size())));
        Json sec = Json::object();
        const int status = post_state(h, *bytes, sha, position, out, sec);
        out.set("seconds", std::move(sec));
        return {status, out.dump()};
    }

private:
    // The import POST; fills out["import"] / out["error"] and seconds transfer / import / handoff. Returns 200 or 502.
    int post_state(const Handoff& h, const std::vector<std::uint8_t>& bytes, const std::string& sha, std::uint32_t position, Json& out,
                   Json& sec, V2StateStream* streaming = nullptr) {
        if (c_.handoff_v2) {
            const auto started = Clock::now();
            std::unique_ptr<V2StateStream> local;
            try {
                const auto before = streaming ? streaming->acknowledged_bytes() : 0U;
                if (!streaming) { local = std::make_unique<V2StateStream>(h, c_); streaming = local.get(); }
                auto reply = local ? streaming->upload_full(bytes, sha) : streaming->finish(bytes, sha);
                out.set("import", std::move(reply));
                Json details = Json::object();
                details.set("streamed_before_prefill_end_bytes", Json(before));
                details.set("acknowledged_payload_bytes", Json(streaming->acknowledged_bytes()));
                details.set("network_s", Json(streaming->network_seconds()));
                out.set("v2", std::move(details));
                sec.set("handoff", Json(secs(started, Clock::now())));
                return 200;
            } catch (const std::exception& e) {
                if (streaming) streaming->abort();
                sec.set("handoff", Json(secs(started, Clock::now())));
                out.set("error", Json(std::string("v2 handoff: ") + e.what()));
                return 502;
            }
        }
        std::vector<std::pair<std::string, std::string>> headers = {{"Content-Type", "application/octet-stream"},
                                                                    {"X-State-Format", "strix/ds41-attention-state-v1"},
                                                                    {"X-State-Sha256", sha}};
        const std::string key = !h.api_key.empty() ? h.api_key : c_.handoff_api_key;
        if (!key.empty()) headers.emplace_back("Authorization", "Bearer " + key);
        if (h.reserve_tokens) headers.emplace_back("X-State-Reserve-Tokens", std::to_string(*h.reserve_tokens));
        const auto t0 = Clock::now();
        try {
            const auto [status, body] = http_post(h.url, headers, bytes, c_.handoff_timeout);
            const double post_s = secs(t0, Clock::now());
            sec.set("handoff", Json(post_s));
            Json reply;
            try {
                reply = strix::serve::parse_json(body);
            } catch (const std::exception&) {
                reply = Json(body.substr(0, 400));
            }
            out.set("import", reply);
            if (status != 200) throw std::runtime_error("import-state answered HTTP " + std::to_string(status));
            const Json* rsha = reply.find("sha256");
            const Json* rpos = reply.find("position");
            if (rsha == nullptr || !rsha->is_string() || rsha->as_string() != sha) throw std::runtime_error("the import reply's sha256 differs");
            if (rpos == nullptr || !rpos->is_number() || static_cast<std::uint64_t>(rpos->as_int()) != position)
                throw std::runtime_error("the import reply's position differs");
            double server_s = 0.0;
            if (const Json* t = reply.find("timing"); t != nullptr)
                if (const Json* ss = t->find("server_s"); ss != nullptr && ss->is_number()) server_s = ss->as_double();
            sec.set("transfer", Json(post_s - server_s));
            sec.set("import", Json(server_s));
            return 200;
        } catch (const std::exception& e) {
            if (sec.find("handoff") == nullptr) sec.set("handoff", Json(secs(t0, Clock::now())));
            out.set("error", Json(std::string("handoff: ") + e.what()));
            return 502;
        }
    }

    // Closed-batch admission (under mu_): the queue front, then later waiting prompts in arrival order while the batch
    // stays within batch_max prompts and batch_sum_tokens rows, no member reaches batch_alone_rows, and no skipped
    // prompt has been passed batch_max_overtakes times.
    std::vector<std::unique_ptr<Job>> take_batch() {
        std::vector<std::unique_ptr<Job>> b;
        b.push_back(std::move(queue_.front()));
        queue_.pop_front();
        const auto fits_alone = [&](const Job& j) { return j.tokens.size() < c_.batch_alone_rows; };
        if (c_.batch_max <= 1U || !fits_alone(*b[0]) || b[0]->tokens.size() > c_.batch_sum_tokens) return b;
        std::size_t sum = b[0]->tokens.size();
        for (std::size_t i = 0; i < queue_.size() && b.size() < std::min<std::size_t>(c_.batch_max, p_.batch_contexts());) {
            const Job& j = *queue_[i];
            const bool ok = fits_alone(j) && sum + j.tokens.size() <= c_.batch_sum_tokens;
            bool fair = true;
            for (std::size_t k = 0; k < i; ++k) fair = fair && queue_[k]->overtaken < c_.batch_max_overtakes;
            if (!ok || !fair) {
                ++i;
                continue;
            }
            for (std::size_t k = 0; k < i; ++k) ++queue_[k]->overtaken;
            sum += j.tokens.size();
            b.push_back(std::move(queue_[i]));
            queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(i));
        }
        return b;
    }

    void loop() {
        for (;;) {
            std::vector<std::unique_ptr<Job>> jobs;
            {
                std::unique_lock lock(mu_);
                cv_.wait(lock, [&] { return quit_ || !queue_.empty(); });
                if (queue_.empty()) return;
                jobs = take_batch();
                if (o_.pass_rows) { p_.clear_cancel(); active_ = jobs[0].get(); }
                busy_ = true;
            }
            std::vector<std::unique_ptr<Captured>> caps;
            bool ok = true;
            int error_status = 500;
            std::string error;
            try {
                caps = gpu_stage(jobs);
            } catch (const CudaPrefillInterrupted& e) {  // bounded passes: cancelled, or no host room (the service stays up)
                ok = false;
                error_status = 503;
                error = error_json(e.what()).dump();
            } catch (const CudaPrefillTransferError& e) {  // the v2 state stream failed (the service stays up)
                ok = false;
                error_status = 502;
                error = error_json(e.what()).dump();
            } catch (const std::exception& e) {
                // the device or the object is in an unknown state: fail this and every later request
                const std::lock_guard lock(mu_);
                failed_ = true;
                ok = false;
                error = error_json(std::string("prefill failed (the service needs a restart): ") + e.what()).dump();
            }
            {
                const std::lock_guard lock(mu_);
                busy_ = false;
                if (o_.pass_rows) { active_ = nullptr; p_.clear_cancel(); }
                if (failed_) {
                    for (auto& j : queue_) j->done.set_value({503, error_json("the service failed").dump()});
                    queue_.clear();
                }
            }
            if (!ok) {
                for (auto& j : jobs) j->done.set_value({error_status, error});  // failed before the capture: the jobs are still ours
                continue;
            }
            for (std::size_t k = 0; k < caps.size(); ++k) caps[k]->job = std::move(jobs[k]);
            if (!c_.pipeline) {
                for (auto& cap : caps) finish(*cap);  // members in batch order
                continue;
            }
            // hand the captured states to the handoff worker in batch order (bounded FIFO: wait while it is full)
            for (auto& cap : caps) {
                std::unique_lock lock(mu_);
                hcv_full_.wait(lock, [&] { return handoff_q_.size() < c_.handoff_queue; });
                handoff_q_.push_back(std::move(cap));
                lock.unlock();
                hcv_.notify_one();
            }
        }
    }

    void handoff_loop() {
        for (;;) {
            std::unique_ptr<Captured> cap;
            {
                std::unique_lock lock(mu_);
                hcv_.wait(lock, [&] { return handoff_quit_ || !handoff_q_.empty(); });
                if (handoff_q_.empty()) return;
                cap = std::move(handoff_q_.front());
                handoff_q_.pop_front();
                handoff_busy_ = true;
            }
            hcv_full_.notify_all();
            cap->t_dequeued = Clock::now();
            finish(*cap);
            {
                const std::lock_guard lock(mu_);
                handoff_busy_ = false;
            }
        }
    }

    // GPU stage: the prefill (one prompt, or a closed batch through run_batch) and each member's state captured into
    // host memory it owns (or, --fast-export solo, into the service's capture buffer). Throws on a device failure.
    std::vector<std::unique_ptr<Captured>> gpu_stage(const std::vector<std::unique_ptr<Job>>& jobs) {
        const std::size_t B = jobs.size();
        std::vector<std::unique_ptr<Captured>> caps(B);
        const auto t_start = Clock::now();
        for (std::size_t k = 0; k < B; ++k) {
            auto& cap = caps[k];
            cap = std::make_unique<Captured>();
            cap->t_start = t_start;
            cap->index = k;
            cap->batch_size = B;
            const auto T = static_cast<std::uint32_t>(jobs[k]->tokens.size());
            cap->want_fa = c_.fast_attention && T >= c_.fast_attention_min_rows;
            cap->mode = c_.export_check ? "check" : c_.fast_export ? "fast" : "legacy";
            if (c_.fast_export && !jobs[k]->export_mode.empty()) cap->mode = jobs[k]->export_mode;
        }
        {
            const std::lock_guard lock(run_mu_);
            struct PublishPins {
                Service& s;
                ~PublishPins() { s.pinned_snapshot_.store(s.p_.pinned_bytes() + (s.capture_ ? s.capture_->pinned_bytes() : 0)); }
            } publish{*this};
            if (B == 1U) {
                auto& cap = *caps[0];
                const Job& job = *jobs[0];
                Clock::time_point t_capture;
                cap.fa = p_.set_fast_attention(cap.want_fa);
                if (c_.handoff_v2 && job.handoff) {
                    cap.streaming = std::make_shared<V2StateStream>(*job.handoff, c_);
                    p_.set_state_stream(cap.streaming);
                }
                try {
                    cap.res = p_.run(job.tokens);
                } catch (...) {
                    p_.set_state_stream(nullptr);
                    if (cap.streaming) cap.streaming->abort();
                    throw;
                }
                p_.set_state_stream(nullptr);
                cap.t_run_end = Clock::now();
                if (o_.pass_rows) {
                    const std::lock_guard lock(mu_);
                    active_ = nullptr;  // export and handoff are already committed to finishing
                    p_.clear_cancel();
                }
                if (cap.mode == "legacy") {
                    p_.release_idle_preload();  // a --fast-export service asked for the old path: as before (preload running)
                    t_capture = Clock::now();
                    cap.state = p_.export_state();
                } else {
                    if (cap.mode == "check") {  // the old path first (the idle preload is still held: no queueing behind it)
                        const auto t0 = Clock::now();
                        cap.state = p_.export_state();
                        const auto t1 = Clock::now();
                        cap.legacy = strix::deepseek41::serialize_attention_state(cap.state);
                        cap.legacy_export_s = secs(t0, t1);
                        cap.legacy_serialize_s = secs(t1, Clock::now());
                    }
                    t_capture = Clock::now();
                    p_.capture_state(*capture_);  // synchronized; releases the idle preload
                }
                cap.t_captured = Clock::now();
                cap.capture_s = secs(t_capture, cap.t_captured);
            } else {
                // a closed batch (--batch-max; never with --fast-export: refused at start): every member's state through
                // export_state(k), host copies each Captured owns
                std::vector<CudaBatchItem> items(B);
                for (std::size_t k = 0; k < B; ++k) {
                    caps[k]->fa = c_.fast_attention && o_.fast_attention && jobs[k]->tokens.size() >= c_.fast_attention_min_rows;
                    items[k] = {jobs[k]->tokens, caps[k]->fa};
                }
                auto rs = p_.run_batch(items);
                const auto t_run_end = Clock::now();
                for (std::size_t k = 0; k < B; ++k) {
                    auto& cap = *caps[k];
                    cap.res = std::move(rs[k]);
                    cap.t_run_end = t_run_end;
                    const auto t0 = Clock::now();
                    cap.state = p_.export_state(k);
                    cap.t_captured = Clock::now();
                    cap.capture_s = secs(t0, cap.t_captured);
                }
            }
        }
        const std::lock_guard lock(mu_);
        if (B > 1U) ++batches_;
        for (auto& cap : caps) {
            cap->run_s = secs(cap->t_start, cap->t_run_end);
            cap->t_dequeued = cap->t_captured;
            cap->id = ++captured_;
        }
        return caps;
    }

    // Accounting of one request (timing): non-overlapping parts of its wall time.
    Json accounting(const Captured& cap, double wall, double serialize_s, double digest_s, double sha_s, double post_s, double handoff_queue_s,
                    double reply_s) const {
        const auto& r = cap.res;
        Json a = Json::object();
        double compute_ms = 0.0, wait_ms = 0.0, gap_ms = 0.0, acquire_ms = 0.0, copy_ms = 0.0;
        std::uint64_t copy_bytes = 0;
        Json layers = Json::array();
        for (std::uint32_t l = o_.first_layer; l < o_.end_layer && l < r.layers.size(); ++l) {
            const auto& t = r.layers[l];
            const double w = double(t.range_wait_ms) + double(t.expert_wait_device_ms);
            const double comp = double(t.total_ms) - w;
            compute_ms += comp;
            wait_ms += w;
            acquire_ms += t.expert_wait_device_ms;
            gap_ms += t.gap_before_ms;
            copy_ms += t.copy_ms;
            copy_bytes += t.copy_bytes;
            Json L = Json::object();
            L.set("layer", Json(static_cast<std::uint64_t>(l)));
            L.set("rows", Json(static_cast<std::uint64_t>(t.rows)));
            L.set("total_ms", Json(double(t.total_ms)));
            L.set("attention_ms", Json(double(t.attention_ms)));
            L.set("moe_ms", Json(double(t.moe_ms)));
            L.set("compute_ms", Json(comp));
            L.set("wait_ms", Json(w));
            L.set("gap_ms", Json(double(t.gap_before_ms)));
            L.set("copy_ms", Json(double(t.copy_ms)));
            L.set("copy_gb", Json(double(t.copy_bytes) / 1e9));
            layers.push(std::move(L));
        }
        const double device_s = r.device_ms * 1e-3;
        const double launch_s = cap.run_s - r.host_setup_s - device_s - r.host_tail_s;  // launch lag + sync latency (not a named part)
        Json run = Json::object();
        run.set("wall_s", Json(cap.run_s));
        run.set("host_setup_s", Json(r.host_setup_s));
        run.set("device_s", Json(device_s));
        run.set("embed_s", Json(r.embed_ms * 1e-3));
        run.set("compute_s", Json(compute_ms * 1e-3));
        run.set("exposed_wait_s", Json(wait_ms * 1e-3));
        run.set("acquire_wait_s", Json(acquire_ms * 1e-3));
        run.set("gaps_s", Json(gap_ms * 1e-3));
        run.set("head_s", Json(r.head_ms * 1e-3));
        run.set("host_tail_s", Json(r.host_tail_s));
        run.set("engram_join_s", Json(r.engram_join_s));
        run.set("unattributed_s", Json(launch_s));
        a.set("run", std::move(run));
        Json dma = Json::object();
        const double run_gb = double(r.expert_bytes_streamed) / 1e9, pre_gb = double(r.preload_bytes) / 1e9;
        dma.set("run_gb", Json(run_gb));
        dma.set("preload_gb", Json(pre_gb));
        dma.set("total_gb", Json(run_gb + pre_gb));
        dma.set("copy_busy_s", Json(r.expert_copy_busy_s));
        dma.set("run_gbps", Json(r.expert_copy_busy_s > 0.0 ? run_gb / r.expert_copy_busy_s : 0.0));
        dma.set("preload_s", Json(r.preload_s));
        dma.set("preload_after_begin_s", Json(r.preload_after_begin_s));
        dma.set("total_dma_s", Json(r.expert_copy_busy_s + r.preload_s));
        // copy time the compute did not wait for (hidden behind compute) vs exposed
        dma.set("hidden_s", Json(std::max(0.0, r.expert_copy_busy_s - wait_ms * 1e-3)));
        dma.set("layer_copy_gb", Json(double(copy_bytes) / 1e9));
        a.set("dma", std::move(dma));
        Json host = Json::object();
        host.set("capture_s", Json(cap.capture_s));
        host.set("handoff_queue_s", Json(handoff_queue_s));
        host.set("serialize_s", Json(serialize_s));
        host.set("digest_s", Json(digest_s));
        host.set("sha256_s", Json(sha_s));
        host.set("post_s", Json(post_s));
        host.set("reply_s", Json(reply_s));
        a.set("host", std::move(host));
        const double named = r.host_setup_s + device_s + r.host_tail_s + cap.capture_s + handoff_queue_s + serialize_s + digest_s + sha_s + post_s + reply_s;
        a.set("wall_s", Json(wall));
        a.set("explained_s", Json(named));
        a.set("explained_frac", Json(wall > 0.0 ? named / wall : 0.0));
        a.set("engine_s", Json(cap.run_s + cap.capture_s));
        a.set("host_tail_after_capture_s", Json(serialize_s + digest_s + sha_s + post_s));
        Json tl = Json::object();  // seconds since the service started (steady clock): GPU idle between requests
        tl.set("start", Json(secs(t0_, cap.t_start)));
        tl.set("run_end", Json(secs(t0_, cap.t_run_end)));
        tl.set("captured", Json(secs(t0_, cap.t_captured)));
        tl.set("handoff_start", Json(secs(t0_, cap.t_dequeued)));
        tl.set("done", Json(secs(t0_, Clock::now())));
        a.set("timeline", std::move(tl));
        a.set("layers", std::move(layers));
        std::fprintf(stderr,
                     "rtxq-acct id=%llu T=%u wall=%.4f run=%.4f [setup %.4f dev %.4f = embed %.4f + compute %.4f + wait %.4f + gaps %.4f + head %.4f; "
                     "tail %.4f (engram join %.4f); unattr %.4f] capture=%.4f hq=%.4f ser=%.4f digest=%.4f sha=%.4f post=%.4f reply=%.4f explained=%.1f%% "
                     "dma run %.2f GB + preload %.2f GB (busy %.3f s, preload %.3f s, after-begin %.3f s)\n",
                     static_cast<unsigned long long>(cap.id), static_cast<unsigned>(cap.job->tokens.size()), wall, cap.run_s, r.host_setup_s, device_s,
                     r.embed_ms * 1e-3, compute_ms * 1e-3, wait_ms * 1e-3, gap_ms * 1e-3, r.head_ms * 1e-3, r.host_tail_s, r.engram_join_s, launch_s, cap.capture_s,
                     handoff_queue_s, serialize_s, digest_s, sha_s, post_s, reply_s, wall > 0.0 ? 100.0 * named / wall : 0.0, run_gb, pre_gb,
                     r.expert_copy_busy_s, r.preload_s, r.preload_after_begin_s);
        return a;
    }

    // Handoff stage: serialize, hash, POST and answer the request (on the GPU worker without --pipeline).
    void finish(Captured& cap) {
        std::pair<int, std::string> reply;
        try {
            reply = finish_reply(cap);
        } catch (const std::exception& e) {
            reply = {500, error_json(std::string("handoff stage failed: ") + e.what()).dump()};
        }
        cap.job->done.set_value(std::move(reply));
    }

    std::pair<int, std::string> finish_reply(Captured& cap) {
        Job& job = *cap.job;
        const auto& res = cap.res;
        const auto T = static_cast<std::uint32_t>(job.tokens.size());
        Json out = Json::object();
        Json sec = Json::object();
        sec.set("queue", Json(secs(job.queued, cap.t_start)));
        const double handoff_queue_s = secs(cap.t_captured, cap.t_dequeued);
        std::vector<std::uint8_t> bytes;
        std::string sha, digest;
        double serialize_s = 0.0, sha_s = 0.0, digest_s = 0.0;
        const bool check_mode = cap.mode == "check";
        Json detail = Json::object(), check = Json::object();
        if (cap.mode == "legacy") {
            const auto t0 = Clock::now();
            bytes = strix::deepseek41::serialize_attention_state(cap.state);
            const auto t1 = Clock::now();
            serialize_s = secs(t0, t1);
            const auto view = std::as_bytes(std::span(bytes.data(), bytes.size()));
            sha = strix::model_content_identity_hex(c_.handoff_v2 ? transfer_sha256(view) : strix::sha256_bytes(view));
            const auto t2 = Clock::now();
            sha_s = secs(t1, t2);
            digest = hex16(cap.state.digest());
            digest_s = secs(t2, Clock::now());
        } else {
            // --fast-export: the service's one capture buffer, filled by this request's gpu_stage (never with --pipeline:
            // refused at start), so the GPU worker seals it here before it captures the next state
            check.set("legacy_export_s", Json(cap.legacy_export_s));
            check.set("legacy_serialize_s", Json(cap.legacy_serialize_s));
            const auto t1 = Clock::now();
            capture_->seal(0, !c_.export_sha_after);  // the SHA-256 overlapped with the digest and checksum
            serialize_s = secs(t1, Clock::now());     // the seal (the bytes were written in place by the capture)
            const auto view = capture_->bytes();
            if (c_.export_sha_after) {
                const auto t2 = Clock::now();
                sha = strix::model_content_identity_hex(strix::sha256_bytes_fast(std::as_bytes(view)));
                sha_s = secs(t2, Clock::now());
            } else {
                sha = capture_->sha256();
                sha_s = capture_->timing().sha_ms / 1e3;  // inside seal (export), on its own thread
            }
            digest = hex16(capture_->digest());  // computed by the seal
            const auto t3 = Clock::now();
            bytes.assign(view.begin(), view.end());  // the POST / retry copy (a pipelined sender posts from the pinned buffer)
            const auto& tm = capture_->timing();
            detail.set("capture_ms", Json(tm.capture_ms));
            detail.set("meta_ms", Json(tm.meta_ms));
            detail.set("pack_ms", Json(tm.pack_ms));
            detail.set("d2h_ms", Json(tm.d2h_ms));
            detail.set("frame_ms", Json(tm.frame_ms));
            detail.set("seal_ms", Json(tm.seal_ms));
            detail.set("sha_ms", Json(sha_s * 1e3));
            detail.set("sha_overlapped", Json(!c_.export_sha_after));
            detail.set("copy_out_ms", Json(secs(t3, Clock::now()) * 1e3));
            detail.set("segments", Json(static_cast<std::uint64_t>(tm.segments)));
            detail.set("copies", Json(static_cast<std::uint64_t>(tm.copies)));
            detail.set("sha_provider", Json(strix::sha256_fast_provider()));
            if (check_mode) {
                const auto t4 = Clock::now();
                const std::string sha_portable = strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(view)));
                check.set("legacy_sha_s", Json(secs(t4, Clock::now())));
                const auto t5 = Clock::now();
                const std::uint64_t legacy_digest = cap.state.digest();
                check.set("legacy_digest_s", Json(secs(t5, Clock::now())));
                check.set("bytes_equal", Json(cap.legacy == bytes));
                check.set("sha_equal", Json(sha_portable == sha));
                check.set("digest_equal", Json(hex16(legacy_digest) == digest));
            }
        }
        out.set("session", job.session);
        out.set("tokens", Json(static_cast<std::uint64_t>(T)));
        out.set("next_token", Json(static_cast<std::uint64_t>(res.next_token)));
        out.set("logits_fnv", Json(hex16(fnv(res.logits))));
        out.set("state_bytes", Json(static_cast<std::uint64_t>(bytes.size())));
        out.set("sha256", Json(sha));
        out.set("digest", Json(digest));
        out.set("capture_id", Json(cap.id));
        out.set("export_mode", Json(cap.mode));
        if (cap.mode != "legacy") out.set("export_detail", std::move(detail));
        if (check_mode) out.set("export_check", std::move(check));
        Json fast = Json::object();
        fast.set("dense", Json(o_.fast_dense));
        fast.set("moe", Json(o_.fast_moe));
        fast.set("attention", Json(cap.fa));
        fast.set("replay_rows", Json(static_cast<std::uint64_t>(o_.replay_rows)));
        if (cap.want_fa && !cap.fa) fast.set("attention_note", Json("not available: start the service with --positions >= the threshold"));
        out.set("fast", std::move(fast));
        sec.set("prefill", Json(res.seconds));
        sec.set("capture", Json(cap.capture_s));
        sec.set("serialize", Json(serialize_s));
        sec.set("export", Json(cap.capture_s + serialize_s));
        sec.set("sha256", Json(sha_s));
        sec.set("digest", Json(digest_s));
        if (c_.pipeline) {
            sec.set("engine", Json(cap.run_s + cap.capture_s));
            sec.set("handoff_queue", Json(handoff_queue_s));
        }
        Json pre = Json::object();
        pre.set("host_pin_order", Json(strix::ds41::host_pin_order_name(o_.host_pin_order)));
        Json host_layers = Json::array();
        // Placement is immutable after load, including any pin-allocation demotions.
        for (std::uint32_t l = o_.first_layer; l < o_.end_layer; ++l)
            if (p_.tiers()[l] == ExpertTier::host) host_layers.push(Json(static_cast<std::uint64_t>(l)));
        pre.set("host_layers", std::move(host_layers));
        pre.set("engram_s", Json(res.engram_seconds));
        pre.set("expert_io_s", Json(res.expert_io_seconds));
        const auto weight_io = p_.weight_io_json();
        if (weight_io != "null") pre.set("weight_io", strix::serve::parse_json(weight_io));
        pre.set("expert_gb_streamed", Json(double(res.expert_bytes_streamed) / 1e9));
        pre.set("file_gb_streamed", Json(double(res.file_bytes_streamed) / 1e9));
        pre.set("host_promoted_layers", Json(static_cast<std::uint64_t>(res.host_promoted_layers)));
        pre.set("host_promoted_total", Json(static_cast<std::uint64_t>(res.host_promoted_total)));
        pre.set("host_promoted_gb", Json(double(res.host_promoted_bytes) / 1e9));
        pre.set("host_demoted_layers", Json(static_cast<std::uint64_t>(res.host_demoted_layers)));
        pre.set("row_layers", Json(res.row_layers));
        pre.set("expert_copy_busy_s", Json(res.expert_copy_busy_s));
        pre.set("expert_stall_ms", Json(res.expert_stall_ms));
        pre.set("engram_records", Json(res.engram_records));
        pre.set("engram_pages", Json(res.engram_pages));
        pre.set("engram_page_hits", Json(res.engram_page_hits));
        pre.set("preload_gb", Json(double(res.preload_bytes) / 1e9));
        out.set("prefill", std::move(pre));
        Json bj = Json::object();
        bj.set("size", Json(static_cast<std::uint64_t>(cap.batch_size)));
        bj.set("index", Json(static_cast<std::uint64_t>(cap.index)));
        bj.set("prefill_s", Json(res.batch_seconds));
        bj.set("overtaken", Json(static_cast<std::uint64_t>(job.overtaken)));
        out.set("batch", std::move(bj));
        auto shared = std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
        {
            const std::lock_guard lock(mu_);
            last_state_ = shared;
            last_sha_ = sha;
            last_position_ = T;
            ++served_;
            if (cap.index == 0U) prefill_total_ += res.batch_seconds;
            out.set("request", Json(served_));
        }
        int status = 200;
        const auto t3 = Clock::now();
        if (job.handoff) status = post_state(*job.handoff, *shared, sha, T, out, sec, cap.streaming.get());
        else out.set("import", Json(nullptr));
        const auto t4 = Clock::now();
        const double post_s = secs(t3, t4);
        if (c_.timing) {
            // the reply's own build is the last part: measured up to here, estimated as the time since t4 at dump
            const double wall_pre = secs(cap.t_start, Clock::now());
            Json acct = accounting(cap, wall_pre, serialize_s, digest_s, sha_s, post_s, handoff_queue_s, secs(t4, Clock::now()));
            out.set("accounting", std::move(acct));
        }
        sec.set("total", Json(secs(cap.t_start, Clock::now())));
        out.set("seconds", std::move(sec));
        return {status, out.dump()};
    }

    CudaPrefill& p_;
    const CudaPrefillOptions o_;
    Config c_;
    std::atomic<std::uint64_t> pinned_snapshot_{0};
    std::mutex mu_, run_mu_;
    std::condition_variable cv_, hcv_, hcv_full_;
    std::deque<std::unique_ptr<Job>> queue_;
    std::deque<std::unique_ptr<Captured>> handoff_q_;
    Job* active_ = nullptr;  // protected by mu_ (--pass-rows cancellation): the job whose run is in progress
    bool quit_ = false, busy_ = false, failed_ = false, handoff_quit_ = false, handoff_busy_ = false;
    std::uint64_t captured_ = 0;
    const Clock::time_point t0_ = Clock::now();
    std::uint64_t served_ = 0, batches_ = 0;
    double prefill_total_ = 0.0;
    std::shared_ptr<const std::vector<std::uint8_t>> last_state_;
    std::shared_ptr<attn::ExportCapture> capture_;  // --fast-export (one; the worker runs one job at a time)
    std::string last_sha_;
    std::uint32_t last_position_ = 0;
    std::thread worker_, handoff_worker_;
};

Handoff handoff_from(const Json& body, const Json& url) {
    if (!url.is_string() || url.as_string().empty()) throw std::invalid_argument("handoff must be a URL string");
    Handoff h;
    h.url = url.as_string();
    (void)parse_url(h.url);
    if (const Json* k = body.find("api_key"); k != nullptr && k->is_string()) h.api_key = k->as_string();
    if (const Json* r = body.find("reserve_tokens"); r != nullptr && !r->is_null()) {
        if (!r->is_number() || r->as_int() < 0) throw std::invalid_argument("reserve_tokens must be a number");
        h.reserve_tokens = static_cast<std::uint64_t>(r->as_int());
    }
    return h;
}

void handle(Sock s, Service& svc, const Config& c, std::uint32_t max_positions, bool bounded_passes) {
    Message req;
    const auto max_body = bounded_passes ? std::size_t{max_positions} * 16U + 65536U : std::size_t{256} << 20;
    const std::string err = read_message(s, req, max_body, false);
    if (!err.empty()) {
        if (err != "connection closed") respond(s, 400, error_json(err).dump());
        return;
    }
    const auto sp1 = req.start.find(' ');
    const auto sp2 = req.start.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return respond(s, 400, error_json("bad request line").dump());
    const std::string method = req.start.substr(0, sp1);
    std::string path = req.start.substr(sp1 + 1, sp2 - sp1 - 1);
    if (const auto q = path.find('?'); q != std::string::npos) path.resize(q);
    if (path == "/health") {
        const Json h = svc.health();
        const Json* st = h.find("status");
        return respond(s, st != nullptr && st->as_string() == "ok" ? 200 : 503, h.dump());
    }
    if (!c.api_key.empty() && req.header("authorization") != "Bearer " + c.api_key && req.header("x-api-key") != c.api_key)
        return respond(s, 401, error_json("invalid or missing API key").dump());
    if (path == "/v1/describe") {
        if (method != "GET") return respond(s, 405, error_json("GET").dump());
        return respond(s, 200, svc.describe().dump());
    }
    if (path != "/v1/prefill" && path != "/v1/handoff" && path != "/v1/cancel") return respond(s, 404, error_json("unknown path " + path).dump());
    if (method != "POST") return respond(s, 405, error_json("POST a JSON body").dump());
    Json body;
    try {
        body = strix::serve::parse_json(req.body);
        if (!body.is_object()) throw std::invalid_argument("the body must be a JSON object");
        if (path == "/v1/cancel") {
            const Json* session = body.find("session");
            const auto [status, reply] = svc.cancel(session ? *session : Json(nullptr));
            return respond(s, status, reply);
        }
        if (path == "/v1/handoff") {
            const Json* url = body.find("handoff");
            if (url == nullptr) throw std::invalid_argument("handoff URL missing");
            const auto [status, reply] = svc.handoff_last(handoff_from(body, *url));
            return respond(s, status, reply);
        }
        auto job = std::make_unique<Job>();
        const Json* toks = body.find("tokens");
        if (toks == nullptr || !toks->is_array() || toks->size() == 0) throw std::invalid_argument("tokens must be a non-empty array of token ids");
        if (toks->size() > max_positions)
            return respond(s, 413, error_json("the prompt has " + std::to_string(toks->size()) + " tokens; the service holds " + std::to_string(max_positions)).dump());
        job->tokens.reserve(toks->size());
        for (const auto& t : toks->as_array()) {
            if (!t.is_number() || t.as_int() < 0 || t.as_int() > 0xffffffffLL) throw std::invalid_argument("token ids must be non-negative integers");
            job->tokens.push_back(static_cast<std::uint32_t>(t.as_int()));
        }
        if (const Json* h = body.find("handoff"); h != nullptr && !h->is_null()) job->handoff = handoff_from(body, *h);
        if (const Json* se = body.find("session"); se != nullptr) job->session = *se;
        if (const Json* ex = body.find("export"); ex != nullptr && ex->is_string()) {  // --fast-export services: per request
            job->export_mode = ex->as_string();
            if (job->export_mode != "legacy" && job->export_mode != "fast" && job->export_mode != "check")
                throw std::invalid_argument("export must be legacy, fast or check");
        }
        job->queued = Clock::now();
        if (bounded_passes) {
            body = Json(nullptr);  // do not retain a parsed million-element array while waiting in the queue
            std::string{}.swap(req.body);
        }
        std::string refused;
        auto f = svc.submit(std::move(job), refused);
        if (!f) return respond(s, 503, error_json(refused).dump());
        const auto [status, reply] = f->get();
        return respond(s, status, reply);
    } catch (const std::exception& e) {
        return respond(s, 400, error_json(e.what()).dump());
    }
}

[[noreturn]] void usage(const char* why) {
    std::fprintf(stderr, "strix-cuda-ds41-prefill-service: %s\n", why);
    std::fprintf(stderr, "  --weight-dirs LIST (; separated; empty disables) --drive-readers N (4) --weight-verify cached|always (cached)\n  Environment when absent: STRIX_WEIGHT_DIRS, STRIX_WEIGHT_VERIFY\n");
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) usage("MODEL.gguf [options]");
    const std::string model = argv[1];
    CudaPrefillOptions o;
    bool weight_dirs_given = false, weight_verify_given = false;
    o.max_positions = 8192;
    o.expert_residency = true;
    o.auto_budget = true;
    bool host_gib_given = false, host_floor_given = false, host_wait_given = false, host_gib_auto_given = false;
    double host_gib_auto = 0.0;
    Config c;
    std::string tiers;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(("missing value of " + a).c_str());
            return argv[++i];
        };
        if (a == "--host") c.host = next();
        else if (a == "--port") c.port = std::stoi(next());
        else if (a == "--pass-rows") {
            const auto value = next();
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), o.pass_rows);
            if (error != std::errc{} || end != value.data() + value.size() || o.pass_rows > 131072U)
                usage("--pass-rows must be an integer in 0..131072");
        }
        else if (a == "--positions" || a == "--context") o.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--device-gib") o.device_expert_gib = std::stod(next());
        else if (a == "--vram-cap-gib" || a == "--vram-cap") o.vram_cap_gib = std::stod(next());
        else if (a == "--vram-reserve-gib") o.vram_reserve_gib = std::stod(next());
        else if (a == "--no-auto-budget") o.auto_budget = false;
        else if (a == "--no-expert-residency") o.expert_residency = false;
        else if (a == "--expert-residency") o.expert_residency = true;
        else if (a == "--auto-budget") o.auto_budget = true;
        else if (a == "--expert-rank") o.expert_rank = next();
        else if (a == "--xfer-group") o.xfer_group_experts = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--xfer-piece-mib") o.xfer_piece_mib = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--engram-upload") o.engram_upload = next();
        else if (a == "--engram-cache-mib") o.engram_cache_mib = std::stoul(next());
        else if (a == "--no-gpu-probe") o.gpu_probe = false;
        else if (a == "--host-gib") { o.host_expert_gib = std::stod(next()); host_gib_given = true; }
        else if (a == "--host-gib-auto") { host_gib_auto = std::stod(next()); host_gib_auto_given = true; }
        else if (a == "--overlay") o.weights_overlay = next();  // dense weights overlay (e.g. c-u6.gguf: the pair's prefill identity)
        else if (a == "--host-pin-order") {
            const auto value = next();
            try { o.host_pin_order = strix::ds41::parse_host_pin_order(value); }
            catch (const std::invalid_argument& e) { usage(e.what()); }
        }
        else if (a == "--host-commit-floor-gib") { o.host_commit_floor_gib = std::stod(next()); host_floor_given = true; }
        else if (a == "--host-pin-wait-s") { o.host_pin_wait_s = std::stod(next()); host_wait_given = true; }
        else if (a == "--host-promote" || a == "--no-host-promote" || a == "--baseline") {}  // resolved below
        else if (a == "--host-promote-max-gib") o.host_promote_max_gib = std::stod(next());
        else if (a == "--tiers") tiers = next();
        else if (a == "--slots") o.slots = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--chunk") o.chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--moe-chunk") o.moe_chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--window") o.window_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--io-threads") o.io_threads = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--engram-threads") o.engram_threads = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--weight-dirs") { o.weight_read.weight_dirs = next(); weight_dirs_given = true; }
        else if (a == "--drive-readers") {
            try { o.weight_read.set_drive_readers(next()); }
            catch (const std::exception& e) { usage(e.what()); }
        }
        else if (a == "--weight-verify") { o.weight_read.weight_verify = next(); weight_verify_given = true; }
        else if (a == "--buffered") o.engram_direct = false;
        else if (a == "--device") o.device = std::stoi(next());
        else if (a == "--layers") o.end_layer = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--replay-rows") {
            const auto v = next();
            if (v != "0" && v != "256") usage("--replay-rows must be 0 or 256");
            o.replay_rows = v == "256" ? 256U : 0U;
        }
        else if (a == "--exact") c.fast_dense = c.fast_moe = c.fast_attention = false;
        else if (a == "--no-fast-dense") c.fast_dense = false;
        else if (a == "--no-fast-moe") c.fast_moe = false;
        else if (a == "--fast-moe-signed-lut") o.fast_moe_signed_lut = true;
        else if (a == "--fast-moe-wide-tiles") o.fast_moe_wide_tiles = true;
        else if (a == "--no-fast-attention") c.fast_attention = false;
        else if (a == "--fast-attention-min-rows") c.fast_attention_min_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--max-queue") c.max_queue = static_cast<std::size_t>(std::stoul(next()));
        else if (a == "--batch-max") c.batch_max = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--batch-sum-tokens") c.batch_sum_tokens = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--batch-alone-rows") c.batch_alone_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--batch-ctx-positions") c.batch_ctx_positions = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--batch-max-overtakes") c.batch_max_overtakes = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--api-key") c.api_key = next();
        else if (a == "--handoff-api-key") c.handoff_api_key = next();
        else if (a == "--handoff-timeout") c.handoff_timeout = std::stoi(next());
        else if (a == "--timing") c.timing = true;
        else if (a == "--pipeline") c.pipeline = true;
        else if (a == "--handoff-queue") c.handoff_queue = static_cast<std::size_t>(std::stoul(next()));
        else if (a == "--fast-export") c.fast_export = true;
        else if (a == "--export-check") c.export_check = true;
        else if (a == "--export-sha-after") c.export_sha_after = true;
        else if (a == "--handoff-v2") c.handoff_v2 = true;
        else if (a == "--state-identity") c.state_identity = next();
        else if (a == "--handoff-only") c.handoff_only = next();
        else if (a == "--v2-timeout") c.v2_timeout = std::stoi(next());
        else usage(("unknown option " + a).c_str());
    }
    try {
        o.weight_read.environment(weight_dirs_given, weight_verify_given, model);
        o.weight_read.require_supported(model);
    } catch (const std::exception& e) { usage(e.what()); }

    // --host-gib-auto G: opportunistic pinned host expert tier capped at G GiB (admitted once per layer against free
    // physical memory and commit, no waits; a layer that does not fit streams from the file with the same bits).
    // Opt-in until coexistence with the other PRO services is qualified; omitted = file-only as before.
    if (host_gib_auto_given && host_gib_given) usage("--host-gib-auto and --host-gib are exclusive");
    if (host_gib_auto_given) {
        if (o.pass_rows != 0U || !tiers.empty()) usage("--host-gib-auto is not available with --pass-rows or --tiers");
        if (!(host_gib_auto >= 0.0 && host_gib_auto <= 1024.0)) usage("--host-gib-auto must be within 0..1024");
        o.host_tier_implicit = true;
        o.host_expert_gib = host_gib_auto;
        if (!host_floor_given) o.host_commit_floor_gib = 20.0;
        if (!host_wait_given) o.host_pin_wait_s = 90.0;
    }
    if (!strix::ds41::valid_host_pin_policy(o.host_commit_floor_gib, o.host_pin_wait_s))
        usage("host floor/wait must be finite and nonnegative");
    if (!(o.host_promote_max_gib >= 0.0 && o.host_promote_max_gib <= 1024.0)) usage("--host-promote-max-gib must be within 0..1024");
    {
        // the default-ON service levers: the last of a lever's own options and --baseline, else STRIX_DS41_RTX_BASELINE=1,
        // else the lever's variable, else on (the service does not take --profile yet: the default column)
        namespace pf = strix::ds41::rtx::profile;
        const auto levers = pf::resolve(pf::Name::standard, false, pf::App::prefill_service,
                                        pf::Inputs{std::vector<std::string>(argv + 2, argv + argc), {}});
        const auto* hp = levers.find("host_promote");
        o.host_promote = hp != nullptr && hp->value == "on";
        c.levers_json = levers.json();
    }
    if (!tiers.empty()) {
        o.tiers.assign(40, ExpertTier::file);
        for (std::size_t l = 0; l < tiers.size() && l < 40; ++l)
            o.tiers[l] = tiers[l] == 'D' ? ExpertTier::device : tiers[l] == 'H' ? ExpertTier::host : ExpertTier::file;
    }
    if (c.timing) {
        o.timing = true;       // per-layer device events + copy events (accounting)
        o.timing_ops = false;  // no per-op events
    }
    if (c.export_check) c.fast_export = true;
    // --fast-export captures into ONE reusable pinned buffer that the reply seals and hashes before the next capture;
    // --pipeline queues several captured states for its handoff worker. Combining them needs a buffer per queued state
    // (not done yet), so the two are refused together.
    if (c.fast_export && c.pipeline) usage("--pipeline cannot be combined with --fast-export / --export-check (yet)");
    o.service_host_buffers = true;
    o.reserve_export_capture = c.fast_export;
    o.defer_idle_preload = c.fast_export;
    o.fast_dense = c.fast_dense;
    if (o.pass_rows) {
        if (o.vram_cap_gib == 0.0) o.vram_cap_gib = kDefaultLongVramCapGiB;
        o.host_commit_floor_gib = std::max(4.0, o.host_commit_floor_gib);
        if (c.batch_max > 1U) usage("--pass-rows requires --batch-max 1");
        if (c.max_queue > 16U) usage("--pass-rows supports --max-queue <= 16 for its host memory budget");
        // the long-context host budget (ds41_memory_budget.hpp) counts one captured state per request on the GPU
        // worker; --pipeline queues more, --fast-export pins a max_positions capture buffer and --timing's accounting
        // does not cover bounded passes: not combined (yet)
        if (c.pipeline || c.fast_export || c.export_check) usage("--pass-rows cannot be combined with --pipeline / --fast-export (yet)");
        if (c.timing) usage("--pass-rows cannot be combined with --timing (yet)");
    }
    o.fast_moe = c.fast_moe;
    o.fast_attention = c.fast_attention && o.max_positions >= c.fast_attention_min_rows;  // its buffers only when reachable
    if (c.handoff_v2 && (c.state_identity.empty() || (!o.pass_rows && c.handoff_only.empty())))
        usage("--handoff-v2 needs --state-identity and --pass-rows (or --handoff-only URL)");
    if (c.handoff_v2 && (c.state_identity.size() > 256U || c.v2_timeout < 1 || c.v2_timeout > 600))
        usage("v2 identity <= 256 bytes and v2 timeout 1..600 seconds");
    if (!c.handoff_only.empty() && !c.handoff_v2) usage("--handoff-only needs --handoff-v2");
    if (c.batch_max > 1U) {
        // B - 1 extra contexts of sum / 2 rows: only the longest member of a batch can exceed sum / 2, and it takes
        // context 0 (max_positions)
        if (c.batch_max > 4U) usage("--batch-max 1..4");
        const std::uint32_t cap = c.batch_ctx_positions != 0U ? c.batch_ctx_positions : c.batch_sum_tokens / 2U;
        if (cap > o.max_positions) usage("--batch-ctx-positions above --positions");
        o.batch_positions.assign(c.batch_max - 1U, cap);
        if (o.engram_upload != "mapped") usage("batches need --engram-upload mapped");
        // --fast-export captures one prompt into the service's single capture buffer; a batch exports every member
        // through export_state(k) (each reply owns its bytes). Not combined (yet).
        if (c.fast_export) usage("--batch-max > 1 cannot be combined with --fast-export / --export-check (yet)");
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) usage("WSAStartup failed");
#else
    std::signal(SIGPIPE, SIG_IGN);
#endif
    try {
        validate_bounded_replay(o.replay_rows, o.cone, o.first_layer, o.end_layer,
                                o.head_rows, o.draft_rows, o.fast_dense, o.fast_moe);
        if (c.handoff_v2) check_transfer_sha256();
        if (!c.handoff_only.empty()) {
            const auto started = Clock::now();
            const auto size = std::filesystem::file_size(model);
            if (size > (1536ULL << 20U) || !strix::ds41::host_commit_fits(strix::ds41::host_memory_room(),
                    size * 3U + (3ULL << 30U), 4ULL << 30U)) throw std::runtime_error("handoff-only host budget exceeded");
            auto state = strix::deepseek41::load_attention_state(model);
            const auto loaded = Clock::now();
            auto bytes = strix::deepseek41::serialize_attention_state(state);
            const auto serialized = Clock::now();
            const auto sha = strix::model_content_identity_hex(transfer_sha256(std::as_bytes(std::span(bytes))));
            const auto sealed = Clock::now();
            Handoff handoff; handoff.url = c.handoff_only;
            V2StateStream transfer(std::move(handoff), c);
            Json receipt;
            try { receipt = transfer.upload_full(bytes, sha); }
            catch (...) { transfer.abort(); throw; }
            Json timings = Json::object();
            timings.set("read_s", Json(secs(started, loaded)));
            timings.set("serialize_s", Json(secs(loaded, serialized)));
            timings.set("sha256_s", Json(secs(serialized, sealed)));
            timings.set("transfer_s", Json(secs(sealed, Clock::now())));
            timings.set("total_s", Json(secs(started, Clock::now())));
            receipt.set("cpu_only_seconds", std::move(timings));
            std::printf("%s\n", receipt.dump().c_str());
            return 0;  // explicit CPU-only transport mode: no CUDA query, allocation or engine construction
        }
        // the listener first: a busy port fails before the load
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;
        addrinfo* res = nullptr;
        const std::string port = std::to_string(c.port);
        if (getaddrinfo(c.host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) throw std::runtime_error("cannot resolve " + c.host);
        Sock ls = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (ls == kNoSock) throw std::runtime_error("socket failed");
#ifndef _WIN32
        int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
        if (::bind(ls, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0 || ::listen(ls, 64) != 0) {
            freeaddrinfo(res);
            throw std::runtime_error("cannot listen on " + c.host + ":" + port);
        }
        freeaddrinfo(res);

        CudaPrefill p(model, o);
        std::printf("{\"event\":\"loaded\",\"describe\":%s}\n", p.describe().c_str());
        std::fflush(stdout);
        Service svc(p, o, c);
        std::printf("{\"event\":\"READY\",\"http\":\"%s:%d\",\"max_positions\":%u,\"fast_dense\":%s,\"fast_moe\":%s,\"fast_attention_min_rows\":%u,"
                    "\"timing\":%s,\"pipeline\":%s,\"handoff_queue\":%zu,\"host_promote\":%s,\"levers\":%s%s}\n",
                    c.host.c_str(), c.port, o.max_positions, c.fast_dense ? "true" : "false", c.fast_moe ? "true" : "false",
                    o.fast_attention ? c.fast_attention_min_rows : 0U, c.timing ? "true" : "false", c.pipeline ? "true" : "false", c.handoff_queue,
                    o.host_promote ? "true" : "false", c.levers_json.c_str(), p.weight_io_fields().c_str());
        std::fflush(stdout);
        std::vector<std::thread> conns;
        std::atomic<int> live{0};
        while (!g_stop.load()) {
            fd_set rd;
            FD_ZERO(&rd);
            FD_SET(ls, &rd);
            timeval tv{};
            tv.tv_usec = 200000;
            const int k = ::select(static_cast<int>(ls + 1), &rd, nullptr, nullptr, &tv);
            if (k <= 0) continue;
            const Sock cs = ::accept(ls, nullptr, nullptr);
            if (cs == kNoSock) continue;
            if (o.pass_rows && live.load() >= static_cast<int>(c.max_queue + 4U)) {
                respond(cs, 503, error_json("too many live connections for bounded host memory").dump());
                close_sock(cs);
                continue;
            }
            int on = 1;
            setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&on), sizeof(on));
            set_timeouts(cs, 3600);
            ++live;
            std::thread([cs, &svc, &c, &live, &o] {
                try {
                    handle(cs, svc, c, o.max_positions, o.pass_rows != 0U);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "strix-cuda-ds41-prefill-service: connection: %s\n", e.what());
                }
                close_sock(cs);
                --live;
            }).detach();
        }
        close_sock(ls);
        while (live.load() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(50));  // the running request ends
        std::printf("{\"event\":\"STOPPED\"}\n");
        std::fflush(stdout);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-ds41-prefill-service: %s\n", e.what());
        return 1;
    }
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
