#include "strix/serve/http_server.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <optional>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
// Winsock: the same calls under their Windows names (WSAPoll, closesocket) on SOCKET handles. No SIGPIPE on Windows.
// Polls for peer data ask for normal data only (POLLRDNORM: WSAPoll's POLLIN also reports out-of-band bytes), so a
// recv after a ready poll cannot block without MSG_DONTWAIT. Socket timeouts are DWORD milliseconds and report
// WSAETIMEDOUT through WSAGetLastError(); TCP_USER_TIMEOUT has no Windows equivalent and is skipped.
#include <winsock2.h>
#include <ws2tcpip.h>
#define poll WSAPoll
#define close(fd) closesocket(fd)
#define MSG_NOSIGNAL 0
#define MSG_DONTWAIT 0
#define SHUT_RDWR SD_BOTH
#define SHUT_WR SD_SEND
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace strix::serve {

std::string HttpRequest::header(std::string_view name) const {
    for (const auto& [key, value] : headers) if (key == name) return value;
    return {};
}

std::string_view http_status_text(int status) {
    switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

namespace {
#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket kNoSocket = INVALID_SOCKET;
constexpr short kPeerData = POLLRDNORM;
using IoSize = int;
bool interrupted() { return WSAGetLastError() == WSAEINTR; }
bool timed_out() { const int e = WSAGetLastError(); return e == WSAETIMEDOUT || e == WSAEWOULDBLOCK; }
// Receive/send timeout in microseconds (0 = none).
void set_timeout(Socket fd, int option, long long micros) {
    const DWORD ms = micros <= 0 ? 0 : static_cast<DWORD>(std::max<long long>(1, (micros + 999) / 1000));
    ::setsockopt(fd, SOL_SOCKET, option, reinterpret_cast<const char*>(&ms), sizeof(ms));
}
std::string socket_error() { return "Winsock error " + std::to_string(WSAGetLastError()); }
#else
using Socket = int;
constexpr Socket kNoSocket = -1;
constexpr short kPeerData = POLLIN;
using IoSize = std::size_t;
bool interrupted() { return errno == EINTR; }
bool timed_out() { return errno == EAGAIN || errno == EWOULDBLOCK; }
void set_timeout(Socket fd, int option, long long micros) {
    const timeval limit{static_cast<time_t>(micros / 1000000), static_cast<suseconds_t>(micros % 1000000)};
    ::setsockopt(fd, SOL_SOCKET, option, &limit, sizeof(limit));
}
std::string socket_error() { return std::strerror(errno); }
#endif
template <class T> void set_option(Socket fd, int level, int name, const T& value) {
    ::setsockopt(fd, level, name, reinterpret_cast<const char*>(&value), sizeof(value));
}
std::string lower(std::string_view text) {
    std::string out(text);
    for (auto& c : out) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return out;
}
// RFC 9110 token: one or more tchar.
bool is_token(std::string_view text) {
    static constexpr std::string_view extra = "!#$%&'*+-.^_`|~";
    return !text.empty() && std::all_of(text.begin(), text.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || extra.find(c) != std::string_view::npos;
    });
}
// Closes a connection without discarding the response just sent: unread request bytes would turn the close into a
// reset that can destroy it, so the write side is shut first and what the client still sends is drained (bounded:
// 1 s, 1 MiB) until it closes too. A stopping server does not wait for clients.
void close_gracefully(Socket fd, const std::atomic<bool>& stopping) {
    ::shutdown(fd, SHUT_WR);
    char sink[16384];
    constexpr std::size_t cap = 1U << 20U;
    std::size_t drained = 0;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (drained < cap && !stopping.load()) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(until - std::chrono::steady_clock::now()).count();
        if (left <= 0) break;
        pollfd p{fd, kPeerData, 0};
        const int ready = ::poll(&p, 1, static_cast<int>(std::min<long long>(left, 50)));  // short slices: stop() is seen
        if (ready < 0) break;
        if (ready == 0) continue;
        const auto n = ::recv(fd, sink, static_cast<IoSize>(std::min(sizeof(sink), cap - drained)), MSG_DONTWAIT);
        if (n <= 0) break;
        drained += static_cast<std::size_t>(n);
    }
    ::close(fd);
}
std::string trim(std::string_view text) {
    std::size_t a = 0, b = text.size();
    while (a < b && (text[a] == ' ' || text[a] == '\t')) ++a;
    while (b > a && (text[b - 1U] == ' ' || text[b - 1U] == '\t' || text[b - 1U] == '\r')) --b;
    return std::string(text.substr(a, b - a));
}

class Connection final : public HttpResponseWriter {
public:
    Connection(Socket fd, bool cors) : fd_(fd), cors_(cors) {}
    bool send_all(std::string_view data) {
        while (!data.empty()) {
            const auto n = ::send(fd_, data.data(), static_cast<IoSize>(std::min<std::size_t>(data.size(), 1U << 30U)), MSG_NOSIGNAL);
            if (n < 0) { if (interrupted()) continue; gone_ = true; return false; }
            data.remove_prefix(static_cast<std::size_t>(n));
        }
        return true;
    }
    std::string head(int status, std::string_view content_type, const std::vector<std::pair<std::string, std::string>>& headers) const {
        std::string out = "HTTP/1.1 " + std::to_string(status) + " " + std::string(http_status_text(status)) + "\r\n";
        if (!content_type.empty()) out += "Content-Type: " + std::string(content_type) + "\r\n";
        if (cors_) out += "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        for (const auto& [name, value] : headers) out += name + ": " + value + "\r\n";
        for (const auto& [name, value] : extra_) out += name + ": " + value + "\r\n";
        out += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
        return out;
    }
    void send(int status, std::string_view content_type, std::string_view body,
        const std::vector<std::pair<std::string, std::string>>& headers = {}) override {
        if (responded_) return;
        responded_ = true;
        auto out = head(status, content_type, headers);
        out += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
        out += body;
        send_all(out);
    }
    bool begin_stream(int status, std::string_view content_type, const std::vector<std::pair<std::string, std::string>>& headers = {}) override {
        if (responded_) return false;
        responded_ = streaming_ = true;
        auto out = head(status, content_type, headers);
        out += "Transfer-Encoding: chunked\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\n\r\n";
        return send_all(out);
    }
    bool write(std::string_view data) override {
        if (!streaming_ || gone_) return false;
        if (data.empty()) return !peer_closed();
        char size[32];
        const auto n = std::snprintf(size, sizeof(size), "%zx\r\n", data.size());
        std::string chunk(size, static_cast<std::size_t>(n));
        chunk.append(data);
        chunk += "\r\n";
        return send_all(chunk) && !peer_closed();
    }
    void end_stream() override {
        if (!streaming_) return;
        streaming_ = false;
        send_all("0\r\n\r\n");
    }
    bool client_gone() const override { return gone_ || const_cast<Connection*>(this)->peer_closed(); }
    // A readable socket with no data pending (orderly shutdown) or an error means the client left.
    bool peer_closed() {
        if (gone_) return true;
        pollfd p{fd_, kPeerData, 0};
        if (::poll(&p, 1, 0) > 0) {
            if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) { gone_ = true; return true; }
            char probe;
            const auto n = ::recv(fd_, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
            if (n == 0) { gone_ = true; return true; }
        }
        return false;
    }
    void add_response_header(std::string name, std::string value) override { extra_.emplace_back(std::move(name), std::move(value)); }
    void reset() {
        responded_ = streaming_ = false;
        extra_.clear();
    }
    bool responded() const { return responded_; }
    bool keep_alive = true;
private:
    Socket fd_;
    bool cors_;
    bool responded_ = false, streaming_ = false, gone_ = false;
    std::vector<std::pair<std::string, std::string>> extra_;  // add_response_header() of the current request
};
}  // namespace

struct HttpServer::Impl {
    HttpServerConfig config;
    HttpHandler handler;
    Socket listen_fd = kNoSocket;
    std::uint16_t bound_port = 0;
    std::atomic<bool> stopping{false};
    std::atomic<unsigned> connections{0};
    std::atomic<std::uint64_t> inflight_body{0};  // Content-Length of requests being received or handled
    std::mutex mutex;
    std::condition_variable idle;
    std::set<Socket> open_fds; // under mutex; stop() shuts them down so blocked reads return

    void handle(Socket fd, std::string peer) {
        const int one = 1;
        set_option(fd, IPPROTO_TCP, TCP_NODELAY, one);
        if (config.write_timeout_seconds > 0) {
            set_timeout(fd, SO_SNDTIMEO, static_cast<long long>(config.write_timeout_seconds) * 1000000);
#ifdef TCP_USER_TIMEOUT
            const unsigned user_timeout_ms = static_cast<unsigned>(config.write_timeout_seconds) * 1000U;
            set_option(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, user_timeout_ms);
#endif
        }
        // A peer that vanished without closing is noticed in about a minute instead of the retransmit timeout.
        const int keep_idle = 30, keep_interval = 10, keep_count = 3;
        set_option(fd, SOL_SOCKET, SO_KEEPALIVE, one);
        set_option(fd, IPPROTO_TCP, TCP_KEEPIDLE, keep_idle);
        set_option(fd, IPPROTO_TCP, TCP_KEEPINTVL, keep_interval);
        set_option(fd, IPPROTO_TCP, TCP_KEEPCNT, keep_count);
        Connection connection(fd, config.cors);
        // Declared after the connection: unregisters the descriptor before the connection closes it.
        struct Registration {
            Impl& impl; Socket fd;
            Registration(Impl& i, Socket f) : impl(i), fd(f) {
                const std::lock_guard lock(impl.mutex);
                impl.open_fds.insert(fd);
                if (impl.stopping.load()) ::shutdown(fd, SHUT_RDWR);
            }
            ~Registration() { const std::lock_guard lock(impl.mutex); impl.open_fds.erase(fd); }
        } registration(*this, fd);
        std::string buffer;
        char chunk[16384];
        using Clock = std::chrono::steady_clock;
        // Waiting for a request uses the idle timeout; once a request has started, only the remaining request deadline.
        std::optional<Clock::time_point> deadline;
        enum class Read { data, closed, late };
        const auto expired = [&] { return deadline && Clock::now() >= *deadline; };
        const auto receive = [&]() {
            for (;;) {
                long long limit = static_cast<long long>(config.idle_timeout_seconds) * 1000000;
                if (deadline) {
                    const auto left = std::chrono::duration_cast<std::chrono::microseconds>(*deadline - Clock::now()).count();
                    if (left <= 0) return Read::late;
                    limit = left;
                }
                set_timeout(fd, SO_RCVTIMEO, limit);
                const auto n = ::recv(fd, chunk, static_cast<IoSize>(sizeof(chunk)), 0);
                if (n > 0) { buffer.append(chunk, static_cast<std::size_t>(n)); return Read::data; }
                if (n < 0 && interrupted()) continue;  // the remaining time is recomputed
                return deadline && n < 0 && timed_out() ? Read::late : Read::closed;
            }
        };
        const auto start_request = [&] {
            if (!deadline && config.request_timeout_seconds > 0)
                deadline = Clock::now() + std::chrono::seconds(config.request_timeout_seconds);
        };
        const auto too_late = [&] {
            connection.keep_alive = false;
            connection.send(408, "application/json", R"({"error":{"message":"request not received in time","type":"invalid_request_error"}})");
        };
        while (!stopping.load()) {
            connection.reset();  // a refusal of this request (408/431/413/501) is its first response
            deadline.reset();
            if (!buffer.empty()) start_request();  // a pipelined request is already arriving
            // Header.
            std::size_t end = std::string::npos;
            while ((end = buffer.find("\r\n\r\n")) == std::string::npos) {
                // Without a terminator the header is at least size - 3 bytes long: the same limit as `end` below,
                // whatever the read boundaries.
                if (buffer.size() > config.max_header_bytes + 3U) {
                    connection.keep_alive = false;
                    connection.send(431, "application/json", R"({"error":{"message":"request header too large","type":"invalid_request_error"}})");
                    return;
                }
                const auto read = receive();
                if (read == Read::late) { too_late(); return; }
                if (read == Read::closed) return;
                start_request();
            }
            if (expired()) { too_late(); return; }  // the last read may have completed the header after the deadline
            // The header limit holds even when the terminator arrived in the same read as the bytes beyond the limit.
            if (end > config.max_header_bytes) {
                connection.keep_alive = false;
                connection.send(431, "application/json", R"({"error":{"message":"request header too large","type":"invalid_request_error"}})");
                return;
            }
            // Framing the server cannot read unambiguously is refused and the connection closed, never guessed.
            const auto bad_request = [&](const std::string& message) {
                connection.keep_alive = false;
                connection.send(400, "application/json", R"({"error":{"message":")" + message + R"(","type":"invalid_request_error"}})");
            };
            HttpRequest request;
            request.peer = peer;
            const std::string_view head(buffer.data(), end);
            const auto line_end = head.find("\r\n");
            const auto first = head.substr(0, line_end);
            const auto s1 = first.find(' '), s2 = first.rfind(' ');
            const auto target = s2 > s1 ? first.substr(s1 + 1U, s2 - s1 - 1U) : std::string_view{};
            const bool visible_target = !target.empty() && std::all_of(target.begin(), target.end(), [](char c) {
                return static_cast<unsigned char>(c) > 0x20U && c != 0x7f;  // no whitespace or control bytes
            });
            if (s1 == std::string_view::npos || s2 == s1 || !is_token(first.substr(0, s1)) || !visible_target) {
                bad_request("malformed request line");
                return;
            }
            request.method = std::string(first.substr(0, s1));
            request.target = std::string(first.substr(s1 + 1U, s2 - s1 - 1U));
            request.version = std::string(first.substr(s2 + 1U));
            if (request.version != "HTTP/1.1" && request.version != "HTTP/1.0") { bad_request("unsupported HTTP version"); return; }
            const auto q = request.target.find('?');
            request.path = request.target.substr(0, q);
            if (q != std::string::npos) request.query = request.target.substr(q + 1U);
            for (std::size_t at = line_end == std::string_view::npos ? head.size() : line_end + 2U; at < head.size();) {
                auto next = head.find("\r\n", at);
                if (next == std::string_view::npos) next = head.size();
                const auto line = head.substr(at, next - at);
                const auto colon = line.find(':');
                // A field name is a token (RFC 9110 tchar): no whitespace or control bytes (this also refuses obsolete
                // line folding) and never empty.
                if (colon == std::string_view::npos || !is_token(line.substr(0, colon))) {
                    bad_request("malformed header field");
                    return;
                }
                request.headers.emplace_back(lower(line.substr(0, colon)), trim(line.substr(colon + 1U)));
                at = next + 2U;
            }
            buffer.erase(0, end + 4U);
            if (std::any_of(request.headers.begin(), request.headers.end(), [](const auto& h) { return h.first == "transfer-encoding"; })) {
                connection.keep_alive = false;
                connection.send(501, "application/json", R"({"error":{"message":"transfer-encoded request bodies are not supported","type":"invalid_request_error"}})");
                return;
            }
            // Content-Length: plain decimal digits only; repeated fields must agree.
            std::optional<std::uint64_t> length_field;
            for (const auto& [key, value] : request.headers) {
                if (key != "content-length") continue;
                std::uint64_t parsed = 0;
                const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || result.ec != std::errc{} ||
                    (length_field && *length_field != parsed)) {
                    bad_request("invalid or conflicting Content-Length");
                    return;
                }
                length_field = parsed;
            }
            const std::uint64_t length = length_field.value_or(0);
            if (length > config.max_body_bytes) {
                connection.keep_alive = false;
                connection.send(413, "application/json", R"({"error":{"message":"request body too large","type":"invalid_request_error"}})");
                return;
            }
            // Reserved from the shared body budget until the handler has returned (the reservation ends with this
            // request's iteration).
            struct BodyReservation {
                std::atomic<std::uint64_t>& total;
                std::uint64_t bytes = 0;
                ~BodyReservation() { total.fetch_sub(bytes); }
            } reservation{inflight_body};
            if (length > 0U && config.max_inflight_body_bytes > 0U) {
                if (inflight_body.fetch_add(length) + length > config.max_inflight_body_bytes) {
                    inflight_body.fetch_sub(length);
                    connection.keep_alive = false;
                    connection.add_response_header("Retry-After", "1");
                    connection.send(503, "application/json", R"({"error":{"message":"server busy: too many request bytes in flight","type":"server_error"}})");
                    return;
                }
                reservation.bytes = length;
            }
            // The body may take one more second per started MiB on top of the request deadline (>= 1 MiB/s).
            if (deadline) *deadline += std::chrono::seconds(static_cast<long>((length + (1U << 20U) - 1U) >> 20U));
            while (buffer.size() < length) {
                const auto read = receive();
                if (read == Read::late) { too_late(); return; }
                if (read == Read::closed) return;
            }
            if (expired()) { too_late(); return; }
            if (buffer.size() == length) {
                request.body = std::move(buffer);  // the usual case: no second copy of a large body
                buffer.clear();
            } else {
                request.body = buffer.substr(0, static_cast<std::size_t>(length));
                buffer.erase(0, static_cast<std::size_t>(length));
            }
            const auto connection_header = lower(request.header("connection"));
            connection.keep_alive = request.version == "HTTP/1.1" ? connection_header != "close" : connection_header == "keep-alive";
            connection.reset();
            if (request.method == "OPTIONS" && config.cors) connection.send(204, "", "");
            else {
                try { handler(request, connection); }
                catch (const std::exception& error) {
                    // A failed handler may have left a response half written (e.g. an open chunked stream): never
                    // reuse this connection for another request.
                    connection.keep_alive = false;
                    if (!connection.responded()) {
                        std::string body = R"({"error":{"message":")";
                        for (const char c : std::string(error.what())) { if (c == '"' || c == '\\') body.push_back('\\'); body.push_back(c >= 0x20 ? c : ' '); }
                        body += R"(","type":"server_error"}})";
                        connection.send(500, "application/json", body);
                    }
                } catch (...) {
                    // Any thrown type is a failed request, never a dead server.
                    connection.keep_alive = false;
                    if (!connection.responded())
                        connection.send(500, "application/json", R"({"error":{"message":"internal error","type":"server_error"}})");
                }
                if (!connection.responded()) connection.send(500, "application/json", R"({"error":{"message":"no response","type":"server_error"}})");
            }
            if (!connection.keep_alive || connection.client_gone()) break;
        }
    }
};

HttpServer::HttpServer(HttpServerConfig config, HttpHandler handler) : impl_(std::make_shared<Impl>()) {
    impl_->config = std::move(config);
    impl_->handler = std::move(handler);
}
HttpServer::~HttpServer() {
    stop();
    // Connection threads use the handler (and what it references): wait for every one of them.
    std::unique_lock lock(impl_->mutex);
    impl_->idle.wait(lock, [&] { return impl_->connections.load() == 0U; });
}

void HttpServer::listen() {
#ifdef _WIN32
    static const bool started = [] { WSADATA data; return WSAStartup(MAKEWORD(2, 2), &data) == 0; }();
    if (!started) throw std::runtime_error("WSAStartup failed");
#endif
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICHOST;
    addrinfo* found = nullptr;
    const auto service = std::to_string(impl_->config.port);
    if (const int rc = ::getaddrinfo(impl_->config.bind_address.c_str(), service.c_str(), &hints, &found); rc != 0)
        throw std::runtime_error("HTTP bind address " + impl_->config.bind_address + ": " + ::gai_strerror(rc));
    const Socket fd = ::socket(found->ai_family, found->ai_socktype, found->ai_protocol);
    if (fd == kNoSocket) { ::freeaddrinfo(found); throw std::runtime_error("HTTP socket failed"); }
    const int one = 1;
    set_option(fd, SOL_SOCKET, SO_REUSEADDR, one);
    if (::bind(fd, found->ai_addr, static_cast<socklen_t>(found->ai_addrlen)) != 0 || ::listen(fd, 128) != 0) {
        const auto error = socket_error();
        ::freeaddrinfo(found); ::close(fd);
        throw std::runtime_error("HTTP listen on " + impl_->config.bind_address + ":" + service + " failed: " + error);
    }
    ::freeaddrinfo(found);
    sockaddr_storage bound{};
    socklen_t size = sizeof(bound);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &size);
    impl_->bound_port = ntohs(bound.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6*>(&bound)->sin6_port :
                                                            reinterpret_cast<sockaddr_in*>(&bound)->sin_port);
    impl_->listen_fd = fd;
}

void HttpServer::serve() {
    auto& impl = *impl_;
    if (impl.listen_fd == kNoSocket) listen();
    while (!impl.stopping.load()) {
        pollfd p{impl.listen_fd, POLLIN, 0};
        if (::poll(&p, 1, 200) <= 0) continue;
        sockaddr_storage peer{};
        socklen_t size = sizeof(peer);
        const Socket fd = ::accept(impl.listen_fd, reinterpret_cast<sockaddr*>(&peer), &size);
        if (fd == kNoSocket) continue;
        char host[NI_MAXHOST] = "";
        ::getnameinfo(reinterpret_cast<sockaddr*>(&peer), size, host, sizeof(host), nullptr, 0, NI_NUMERICHOST);
        if (impl.connections.load() >= impl.config.max_connections) {
            // Closed at once, not drained: the accept loop never waits on one client (the 503 is best effort).
            const char* busy = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            (void)::send(fd, busy, static_cast<IoSize>(std::strlen(busy)), MSG_NOSIGNAL);
            ::close(fd);
            continue;
        }
        ++impl.connections;
        const auto retire = [](Impl& shared) {
            {
                const std::lock_guard lock(shared.mutex);
                --shared.connections;
            }
            shared.idle.notify_all();
        };
        try {
            std::thread([shared = impl_, fd, peer_name = std::string(host), retire] {
                // A failure inside one connection ends that connection only; handle() has unregistered fd on exit.
                try { shared->handle(fd, peer_name); } catch (...) {}
                close_gracefully(fd, shared->stopping);
                retire(*shared);
            }).detach();
        } catch (...) {
            // No thread for this connection (e.g. EAGAIN): refuse it like a full server and keep serving.
            const char* busy = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            (void)::send(fd, busy, static_cast<IoSize>(std::strlen(busy)), MSG_NOSIGNAL);
            ::close(fd);
            retire(impl);
        }
    }
}

void HttpServer::stop() {
    impl_->stopping.store(true);
    if (impl_->listen_fd != kNoSocket) {
        ::shutdown(impl_->listen_fd, SHUT_RDWR);
        ::close(impl_->listen_fd);
        impl_->listen_fd = kNoSocket;
    }
    const std::lock_guard lock(impl_->mutex);
    for (const Socket fd : impl_->open_fds) ::shutdown(fd, SHUT_RDWR);
}
std::uint16_t HttpServer::port() const noexcept { return impl_->bound_port; }
unsigned HttpServer::active_connections() const noexcept { return impl_->connections.load(); }

}  // namespace strix::serve
