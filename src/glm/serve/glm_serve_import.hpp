#pragma once
// Per-rank staging of RTX prefill hand-offs for the GLM pair server. Off the decode path: import_expect starts a
// receive on this rank's endpoint (the RTX producer connects to it, wire v1, src/glm/rtxpf/import), the payload is
// verified (header == expected: model, prompt hash, frontier, layout, rank, node; extent == the layout sum) and kept
// in memory; import_state then restores the slot from it. Both ranks report their status in every reply block; the
// lead asks the RTX only once both listen, commits (import_state) only when both are ready, and falls back to its own
// prefill on any failure. Nothing here blocks the caller: a dropped receive is cancelled and reaped later.
//
// Commit (import_transaction, run by BOTH ranks inside the same import_state replay step): a prepare vote (each rank's
// local checks, host-memory admission and the authenticated hand-off binding: equal on both ranks or both abort), the
// 'DSA once' relay if asked (its own header vote, buffers allocated before it), the restore, a restored vote (both
// restored or both abort, their slots reset), and only then the suffix (last prompt token / MTP bootstrap), which runs
// engine collectives. A failure before the restored vote is an import failure on both ranks (fallback to the pair
// prefill). A failure inside the suffix follows the engine's contract: ImportFatal, the server stops (the peer may be
// inside a collective that can no longer pair up). A broken pair link is fatal in any phase (the communicator poisons
// itself).
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::glm::serve {

// pending: announced, the receiver not bound yet; listening: bound, the RTX may connect; ready: received and verified;
// failed: refused or broken; restored: import_state succeeded on this rank (reported after the commit).
enum class ImportStatus : std::uint32_t { none = 0, pending = 1, ready = 2, failed = 3, listening = 4, restored = 5 };

// One rank's view of the import in flight (sent in every follower reply): its id, ImportStatus, and after import_state
// the first token it produced.
struct ImportReport {
    std::uint32_t id = 0, status = 0, token = 0;
};

// The hello's hash of the attested --import-model-sha: the 64 hex digits parsed to 32 bytes (either case; anything else
// throws), then FNV-1a 64 over the bytes, so equal SHAs spelled differently agree.
inline std::uint64_t attested_model_hash(const std::string& hex) {
    if (hex.size() != 64) throw std::invalid_argument("--import-model-sha: expected 64 hex digits");
    auto nib = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
        throw std::invalid_argument("--import-model-sha: not a hex digit");
    };
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < 64; i += 2) {
        const auto byte = static_cast<std::uint8_t>((nib(hex[i]) << 4) | nib(hex[i + 1]));
        h = (h ^ byte) * 1099511628211ULL;
    }
    return h;
}

// A failure inside the import's engine collectives (after both ranks restored): not an import failure but the engine's
// fatal error (the control replay re-throws it; both ranks stop).
struct ImportFatal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The authenticated hand-off binding: a digest of the producer's Begin common fields (version, world, layers,
// frontier, total, model, prompt, transaction, producer, node; rank excluded), the same on both ranks for one hand-off.
using Binding = std::array<std::uint8_t, 32>;

// One rank's staged hand-off: its restore payload (frames in order) and the binding it was received under.
struct StagedPayload {
    std::vector<std::uint8_t> payload;
    Binding binding{};
    StagedPayload() = default;
    StagedPayload(std::vector<std::uint8_t> p, const Binding& b = {}) : payload(std::move(p)), binding(b) {}  // NOLINT: implicit
};

class ImportStager {
public:
    // Receives one hand-off for `id`; returns the rank's restore payload (frames applied in order) or throws. It calls
    // listening() once the endpoint is bound and gives up soon after `cancel` turns true.
    using Receive = std::function<StagedPayload(std::uint32_t id, std::span<const std::uint32_t> tokens,
                                                const std::atomic<bool>& cancel, const std::function<void()>& listening)>;
    explicit ImportStager(Receive receive) : receive_(std::move(receive)) {}
    ~ImportStager() {
        std::vector<std::thread> t;
        {
            std::lock_guard<std::mutex> l(m_);
            for (auto& [id, e] : entries_) {
                e.cancel->store(true);
                if (e.worker.joinable()) t.push_back(std::move(e.worker));
            }
            for (auto& r : retired_) {
                r.cancel->store(true);
                t.push_back(std::move(r.worker));
            }
            retired_.clear();
        }
        for (auto& x : t)
            if (x.joinable()) x.join();
    }
    ImportStager(const ImportStager&) = delete;
    ImportStager& operator=(const ImportStager&) = delete;

    // Starts receiving `id` (a new id; an id already known is refused).
    void expect(std::uint32_t id, std::vector<std::uint32_t> tokens) {
        reap();
        std::lock_guard<std::mutex> l(m_);
        if (entries_.count(id) != 0) throw std::runtime_error("import: id " + std::to_string(id) + " already expected");
        if (!retired_.empty()) {  // a dropped receive may still hold the endpoint: this import falls back (both ranks vote)
            auto& e = entries_[id];
            e.status = ImportStatus::failed;
            e.error = "a dropped receive is still closing the endpoint";
            e.cancel = std::make_shared<std::atomic<bool>>(true);
            e.done = std::make_shared<std::atomic<bool>>(true);
            return;
        }
        auto& e = entries_[id];
        e.status = ImportStatus::pending;
        e.cancel = std::make_shared<std::atomic<bool>>(false);
        e.done = std::make_shared<std::atomic<bool>>(false);
        e.worker = std::thread([this, id, t = std::move(tokens), cancel = e.cancel, done = e.done] {
            StagedPayload bytes;
            std::string why;
            const std::function<void()> listening = [this, id, cancel] {
                std::lock_guard<std::mutex> g(m_);
                const auto it = entries_.find(id);
                if (it != entries_.end() && it->second.cancel == cancel && it->second.status == ImportStatus::pending)
                    it->second.status = ImportStatus::listening;
            };
            try {
                bytes = receive_(id, t, *cancel, listening);
            } catch (const std::exception& ex) {
                why = ex.what();
            } catch (...) {
                why = "receive failed";
            }
            {
                std::lock_guard<std::mutex> g(m_);
                const auto it = entries_.find(id);
                if (it != entries_.end() && it->second.cancel == cancel) {  // still wanted (not dropped)
                    if (why.empty()) {
                        it->second.payload = std::move(bytes.payload);
                        it->second.binding = bytes.binding;
                        it->second.status = ImportStatus::ready;
                    } else {
                        it->second.error = why;
                        it->second.status = ImportStatus::failed;
                    }
                }
            }
            done->store(true);
        });
    }
    ImportStatus status(std::uint32_t id) const {
        std::lock_guard<std::mutex> l(m_);
        const auto it = entries_.find(id);
        return it == entries_.end() ? ImportStatus::none : it->second.status;
    }
    std::string error(std::uint32_t id) const {
        std::lock_guard<std::mutex> l(m_);
        const auto it = entries_.find(id);
        return it == entries_.end() ? std::string() : it->second.error;
    }
    // The ready payload of `id` (valid until drop(id); a ready entry is never written again).
    std::span<const std::uint8_t> payload(std::uint32_t id) const {
        std::lock_guard<std::mutex> l(m_);
        const auto it = entries_.find(id);
        if (it == entries_.end() || it->second.status != ImportStatus::ready)
            throw std::runtime_error("import: nothing staged under id " + std::to_string(id));
        return it->second.payload;
    }
    // The binding the ready payload of `id` was received under.
    Binding binding(std::uint32_t id) const {
        std::lock_guard<std::mutex> l(m_);
        const auto it = entries_.find(id);
        if (it == entries_.end() || it->second.status != ImportStatus::ready)
            throw std::runtime_error("import: nothing staged under id " + std::to_string(id));
        return it->second.binding;
    }
    // Forget `id` once restored or abandoned. Never waits (it runs inside the control replay, beside decoding requests):
    // a running receive is cancelled and retired; expect() refuses new imports until every retired receive has ended
    // (its listener closed), so no new receive races it for the endpoint.
    void drop(std::uint32_t id) {
        {
            std::lock_guard<std::mutex> l(m_);
            const auto it = entries_.find(id);
            if (it != entries_.end()) {
                it->second.cancel->store(true);
                if (it->second.worker.joinable())
                    retired_.push_back(Retired{std::move(it->second.worker), it->second.cancel, it->second.done});
                entries_.erase(it);
            }
        }
        reap();
    }
    // Receives cancelled but not finished yet (tests).
    std::size_t retired() const {
        std::lock_guard<std::mutex> l(m_);
        return retired_.size();
    }

private:
    struct Entry {
        ImportStatus status = ImportStatus::none;
        std::vector<std::uint8_t> payload;
        Binding binding{};
        std::string error;
        std::thread worker;
        std::shared_ptr<std::atomic<bool>> cancel, done;
    };
    struct Retired {
        std::thread worker;
        std::shared_ptr<std::atomic<bool>> cancel, done;
    };
    // Joins the cancelled receives that have ended (their join is immediate).
    void reap() {
        std::vector<std::thread> t;
        {
            std::lock_guard<std::mutex> l(m_);
            for (auto it = retired_.begin(); it != retired_.end();) {
                if (it->done->load()) {
                    t.push_back(std::move(it->worker));
                    it = retired_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto& x : t) x.join();
    }
    Receive receive_;
    mutable std::mutex m_;
    std::map<std::uint32_t, Entry> entries_;
    std::vector<Retired> retired_;
};

// ---- 'DSA once' (hand-off Begin.version 2): the RTX producer sends the DSA components (the same bytes for both ranks)
// to rank 0 only; rank 1 receives its KDA components, and rank 0 relays the DSA ones to it over the pair link before
// both restore. A rank's restore payload is its components in layout order (Engine::restore_state bytes).
struct StateComponent {
    bool dsa = false;         // DSA latent / index key / gate / pooled (else KDA S / conv: per rank)
    std::uint64_t bytes = 0;  // rows x stride
};
struct RelayStats {
    std::uint64_t bytes = 0, chunks = 0;
    double ms = 0;
};
inline std::uint64_t component_bytes(std::span<const StateComponent> c, bool dsa) {
    std::uint64_t n = 0;
    for (const auto& x : c)
        if (x.dsa == dsa) n += x.bytes;
    return n;
}
// rank 0: its DSA components, in layout order, out of its full payload
inline std::vector<std::uint8_t> gather_dsa(std::span<const std::uint8_t> full, std::span<const StateComponent> c) {
    if (full.size() != component_bytes(c, true) + component_bytes(c, false)) throw std::runtime_error("import relay: rank 0 payload size");
    std::vector<std::uint8_t> out;
    out.reserve(static_cast<std::size_t>(component_bytes(c, true)));
    std::size_t o = 0;
    for (const auto& x : c) {
        if (x.dsa) out.insert(out.end(), full.begin() + static_cast<std::ptrdiff_t>(o), full.begin() + static_cast<std::ptrdiff_t>(o + x.bytes));
        o += x.bytes;
    }
    return out;
}
// rank 1: its KDA-only payload with the relayed DSA components put back in layout order
inline std::vector<std::uint8_t> merge_dsa(std::span<const std::uint8_t> kda, std::span<const std::uint8_t> dsa,
                                           std::span<const StateComponent> c) {
    if (kda.size() != component_bytes(c, false) || dsa.size() != component_bytes(c, true))
        throw std::runtime_error("import relay: rank 1 payload sizes");
    std::vector<std::uint8_t> out;
    out.reserve(kda.size() + dsa.size());
    std::size_t k = 0, d = 0;
    for (const auto& x : c) {
        auto& src = x.dsa ? dsa : kda;
        auto& o = x.dsa ? d : k;
        out.insert(out.end(), src.begin() + static_cast<std::ptrdiff_t>(o), src.begin() + static_cast<std::ptrdiff_t>(o + x.bytes));
        o += x.bytes;
    }
    return out;
}
// rank 1: merges into `out` (already reserved: no allocation after the relay)
inline void merge_dsa_into(std::vector<std::uint8_t>& out, std::span<const std::uint8_t> kda, std::span<const std::uint8_t> dsa,
                           std::span<const StateComponent> c) {
    if (kda.size() != component_bytes(c, false) || dsa.size() != component_bytes(c, true))
        throw std::runtime_error("import relay: rank 1 payload sizes");
    out.clear();
    std::size_t k = 0, d = 0;
    for (const auto& x : c) {
        auto& src = x.dsa ? dsa : kda;
        auto& o = x.dsa ? d : k;
        out.insert(out.end(), src.begin() + static_cast<std::ptrdiff_t>(o), src.begin() + static_cast<std::ptrdiff_t>(o + x.bytes));
        o += x.bytes;
    }
}
// The relay round of import_state, run by BOTH ranks at the same point. Each rank first allocates what the relay needs
// (rank 0: its DSA components gathered; rank 1: the DSA buffer and the merged payload), then a header (ok flag, DSA
// bytes) goes both ways: a rank that cannot take part (nothing staged, wrong size, no memory) makes both throw together
// (import_abort on both, the lead falls back). Then the DSA bytes in chunks of at most `chunk` bytes, rank 0 -> rank 1
// (an exchange is symmetric: rank 1 sends a same-size block back, ignored). xchg(local, remote, seq): one pair exchange,
// remote empty = not wanted. Returns rank 1's merged payload; rank 0 returns an empty vector (its staged payload is
// restored as is).
template <class Xchg>
std::vector<std::uint8_t> dsa_relay(unsigned rank, bool ok, std::span<const std::uint8_t> mine, std::span<const StateComponent> c,
                                    std::size_t chunk, Xchg&& xchg, RelayStats* stats = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t dsa_bytes = component_bytes(c, true);
    const std::uint64_t expect = rank == 0 ? dsa_bytes + component_bytes(c, false) : component_bytes(c, false);
    const bool sized = mine.size() == expect && chunk > 0;
    std::vector<std::uint8_t> dsa, merged;
    bool prepared = ok && sized;
    if (prepared) {
        try {
            if (rank == 0) {
                dsa = gather_dsa(mine, c);
            } else {
                dsa.resize(static_cast<std::size_t>(dsa_bytes));
                merged.reserve(mine.size() + dsa.size());
            }
        } catch (const std::bad_alloc&) {
            prepared = false;
        }
    }
    const std::uint64_t hdr[2] = {prepared ? 1U : 0U, dsa_bytes};
    std::uint64_t peer[2] = {};
    xchg(std::as_bytes(std::span(hdr)), std::as_writable_bytes(std::span(peer)), 0);
    if (!ok) throw std::runtime_error("import relay: nothing usable is staged on this rank for the relay");
    if (!sized) throw std::runtime_error("import relay: this rank's staged state has " + std::to_string(mine.size()) + " bytes, expected " +
                                         std::to_string(expect));
    if (!prepared) throw std::runtime_error("import relay: this rank cannot allocate the relay buffers");
    if (peer[0] != 1U || peer[1] != dsa_bytes) throw std::runtime_error("import relay: the peer cannot relay / receive");
    std::uint64_t seq = 1;
    for (std::uint64_t o = 0; o < dsa_bytes; o += chunk, ++seq) {
        const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(chunk, dsa_bytes - o));
        const std::span<std::uint8_t> part(dsa.data() + o, n);
        if (rank == 0) xchg(std::as_bytes(part), std::span<std::byte>(), seq);
        else xchg(std::as_bytes(part), std::as_writable_bytes(part), seq);  // the block it sends back is copied out first
    }
    if (stats) {
        stats->bytes = dsa_bytes;
        stats->chunks = seq - 1;
        stats->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    if (rank == 0) return {};
    merge_dsa_into(merged, mine, dsa, c);
    return merged;
}

// One status vote of the import commit, both ranks at the same point: {ok, frontier, binding} both ways. Any rank not
// ok, or a binding / frontier that differs, makes BOTH throw (std::runtime_error: an import failure, never fatal).
// xchg(local, remote, seq) as for the relay; an empty xchg (one rank, world 1) checks only the local flag.
template <class Xchg>
void import_vote(const char* phase, bool ok, const std::string& why, const Binding& binding, std::uint64_t frontier, Xchg&& xchg,
                 std::uint64_t seq) {
    std::array<std::uint8_t, 48> mine{}, peer{};
    const std::uint64_t head[2] = {ok ? 1U : 0U, frontier};
    std::memcpy(mine.data(), head, sizeof head);
    std::memcpy(mine.data() + 16, binding.data(), binding.size());
    bool paired = false;
    if constexpr (requires { static_cast<bool>(xchg); }) paired = static_cast<bool>(xchg);
    else paired = true;
    if (paired) xchg(std::as_bytes(std::span(mine)), std::as_writable_bytes(std::span(peer)), seq);
    if (!ok) throw std::runtime_error(std::string("import ") + phase + ": this rank: " + (why.empty() ? "not ready" : why));
    if (!paired) return;
    std::uint64_t ph[2] = {};
    std::memcpy(ph, peer.data(), sizeof ph);
    if (ph[0] != 1U) throw std::runtime_error(std::string("import ") + phase + ": the peer rank failed");
    if (ph[1] != frontier || std::memcmp(peer.data() + 16, binding.data(), binding.size()) != 0)
        throw std::runtime_error(std::string("import ") + phase + ": the ranks staged different hand-offs (binding / frontier mismatch)");
}

// The import commit on one rank (see the header comment): prepared = this rank's local checks passed (why: else);
// relay() runs the 'DSA once' relay when asked (may be empty); restore() puts the state into the slot; suffix() runs
// the last prompt token / MTP bootstrap (engine collectives) and returns the first token.
template <class Xchg, class Relay, class Restore, class Suffix>
std::uint32_t import_transaction(bool prepared, const std::string& why, const Binding& binding, std::uint64_t frontier, Xchg&& vote,
                                 Relay&& relay, Restore&& restore, Suffix&& suffix) {
    import_vote("prepare", prepared, why, binding, frontier, vote, 1);
    relay();  // its own header vote: both throw together
    bool restored = true;
    std::string err;
    try {
        restore();
    } catch (const std::exception& e) {
        restored = false;
        err = e.what();
    }
    import_vote("restore", restored, err, binding, frontier, vote, 2);
    try {
        return suffix();
    } catch (const ImportFatal&) {
        throw;
    } catch (const std::exception& e) {
        throw ImportFatal(std::string("import: failure inside the engine collectives after both ranks restored (fatal): ") + e.what());
    }
}

}  // namespace strix::glm::serve
