#pragma once
// GLM TP2 server control channel (host-only, no HIP): rank 0 (lead) -> rank 1 (follower) command blocks.
//
// One fixed-size 512-byte block per round, both ranks exchange at the same time (the follower answers with its own
// block: an acknowledgement plus the echo of the tokens its last command produced). Prompt tokens of a prefill follow
// the prefill block in messages of kTokensPerMessage tokens. Both ranks then make the same engine calls (replay), so
// every engine collective pairs up.
//
// Checks on every round (both ranks): magic, version, the round number (no duplicate / stale round), the peer's
// acknowledgement of the previous round, and the echo: the round, count, last token and FNV-1a of the tokens the
// peer's last command produced must equal this rank's own (both ranks compute every token; a disagreement aborts both).
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "glm/serve/glm_serve_import.hpp"

namespace strix::glm::serve {

inline constexpr std::uint32_t kControlMagic = 0x434D4C47U;  // "GLMC" (little-endian bytes)
inline constexpr std::uint32_t kControlVersion = 6;  // 3: chunked prefill (ppos); 4: batched spec (spec_verify / spec_draft); 5: import_expect / import_state / state_digest; 6: import settings in Hello (moves every later field by 16 bytes), import_drop
inline constexpr std::uint32_t kPrefillMtp = 1U;   // prefill pflags: exact chunked MTP bootstrap (prefill_mtp)
inline constexpr std::uint32_t kPrefillLast = 2U;  // prefill pflags: the prompt's last chunk (MTP: its seed is kept)
inline constexpr std::uint32_t kMaxMtpCycles = 64;
inline constexpr std::uint32_t kEnterSeed = 4U;   // mtp_enter pflags: open the C1 decoder from the slot's batched-spec seed
inline constexpr std::uint32_t kImportFailed = ~0U;  // import_state's token on a rank whose restore failed
inline constexpr std::uint32_t kImportDsaOnce = 8U;  // import_state pflags: hand-off v2, rank 0 relays the DSA components to rank 1
inline constexpr std::size_t kControlBytes = 512;
inline constexpr unsigned kMaxRows = 8;                       // Engine::step takes 1..8 rows
inline constexpr std::size_t kTokensPerMessage = 16384;       // 64 KiB per prompt message
inline constexpr std::uint64_t kNoRound = ~std::uint64_t{0};

// mtp_enter: seed the slot's drafter from the last step's row pslot (row[0] = {slot, token it produced, its position}).
// mtp_run: up to `cycles` verify cycles of slot pslot (the first run opens the decoder from the seed with `budget`).
// mtp_leave: close the slot's decoder (its deferred KDA rows are applied: plain steps may follow).
// spec_verify: batched self-speculation, ONE verify of every decoding slot's window (row[] sorted by slot: anchor, then
//   drafts at consecutive positions; sbudget[i] = tokens window i may still emit), then per slot greedy acceptance,
//   commit and the drafter catch-up (both ranks, from the merged results). Produces per window: count, tokens.
// spec_draft: one more draft for slot pslot (an MTP extension step of its drafter chain).
enum class Cmd : std::uint32_t { reply = 0, hello = 1, prefill = 2, step = 3, heartbeat = 4, shutdown = 5,
                                 mtp_enter = 6, mtp_run = 7, mtp_leave = 8, spec_verify = 9, spec_draft = 10,
                                 import_state = 11, import_expect = 12, state_digest = 13, import_drop = 14 };
// state_digest (import gate): both ranks digest slot pslot's state up to frontier ppos (Engine::state_bytes_upto, per
// component: fnv64 + size, as u32 words); rank 1 sends its digests to rank 0. No engine state changes.
// import_expect: every rank starts receiving the RTX hand-off for import_id (plen prompt tokens follow); its replies then
// carry that id and its staging status (import_id / import_status: ImportStatus). import_state commits only once both
// ranks report ready; a failure on either rank (import_abort: the slot reset, kImportFailed) is reported in the next
// reply (import_status restored / failed, import_token: the first token) and both fall back to the prefill.
// import_drop: every rank forgets what it staged (or is still receiving) under import_id (a fallback or a cancelled
// request), so its receiver and memory are freed at once.
// import_state: slot pslot restored from the payload each rank staged under import_id (the RTX producer's hand-off,
// verified on that rank), then its last prompt token (plen tokens follow, as for prefill) through prefill / prefill_mtp.

inline const char* cmd_name(Cmd c) {
    switch (c) {
        case Cmd::reply: return "reply";
        case Cmd::hello: return "hello";
        case Cmd::import_state: return "import_state";
        case Cmd::import_expect: return "import_expect";
        case Cmd::import_drop: return "import_drop";
        case Cmd::state_digest: return "state_digest";
        case Cmd::prefill: return "prefill";
        case Cmd::step: return "step";
        case Cmd::heartbeat: return "heartbeat";
        case Cmd::shutdown: return "shutdown";
        case Cmd::mtp_enter: return "mtp_enter";
        case Cmd::mtp_run: return "mtp_run";
        case Cmd::mtp_leave: return "mtp_leave";
        case Cmd::spec_verify: return "spec_verify";
        case Cmd::spec_draft: return "spec_draft";
    }
    return "?";
}

// Engine::Row on the wire (same field order; this header does not include the HIP engine header).
struct WireRow {
    std::uint32_t slot = 0, token = 0;
    std::uint64_t pos = 0;
    bool operator==(const WireRow&) const = default;
};

// hello: what both ranks must agree on before the first engine call.
struct Hello {
    std::uint64_t model_hash = 0;     // FNV-1a of the model directory's metadata (config, index, shard names and sizes)
    std::uint64_t identity_hash = 0;  // FNV-1a of the resolved engine configuration (identity, prefill, exchange, env)
    std::uint64_t capacity = 0;       // positions per slot
    std::uint32_t slots = 0;          // engine slots
    std::uint32_t max_rows = 0;       // prefill chunk rows (Options::max_rows)
    std::uint32_t step_rows = 0;      // rows per step (kMaxRows)
    std::uint32_t world = 0;
    std::uint32_t mtp = 0;            // served MTP self-speculation on (both ranks load and run it)
    std::uint32_t mtp_kmax = 0;       // drafts per cycle (adaptive below it)
    std::uint32_t g2 = 0;             // batched self-speculation (multi-slot verify) on
    // RTX import settings every rank must share (0 = import off): bit 0 receiver on, bit 1 'DSA once'; the expected
    // hand-off wire version; FNV-1a of the --import-model-sha attestation (operator-supplied, see the README).
    std::uint32_t import_flags = 0;
    std::uint32_t import_wire = 0;
    std::uint32_t import_pad = 0;
    std::uint64_t import_model_hash = 0;
    bool operator==(const Hello&) const = default;
};

struct Control {
    std::uint32_t magic = kControlMagic;
    std::uint32_t version = kControlVersion;
    std::uint64_t round = 0;
    std::uint64_t ack_round = kNoRound;  // the last round this rank completed (kNoRound: none)
    std::uint32_t cmd = 0;               // Cmd
    std::uint32_t resets = 0;            // bit s: reset slot s before the command (a prefill always sets its slot)
    std::uint32_t pslot = 0, plen = 0;   // prefill: slot, prompt tokens (they follow in kTokensPerMessage messages)
    std::uint32_t nrows = 0, rank = 0;   // step: rows used in row[]; sender's rank
    WireRow row[kMaxRows]{};
    // echo of the sender's last executed command that produced tokens
    std::uint64_t echo_round = kNoRound;
    std::uint32_t echo_count = 0, echo_token = 0, echo_hash = 0, pad0 = 0;
    Hello hello{};
    std::uint32_t pflags = 0;            // prefill: kPrefillMtp
    std::uint32_t budget = 0, cycles = 0, pad1 = 0;  // mtp_run
    std::uint64_t ppos = 0;              // prefill: position of the chunk's first token (0: the slot is reset first)
    std::uint32_t sbudget[kMaxRows]{};   // spec_verify: per window
    std::uint32_t import_id = 0, import_status = 0;  // import_state / import_expect; replies: this rank's staging status
    std::uint32_t import_token = 0, pad2 = 0;          // replies: the first token of this rank's import_state
};
static_assert(sizeof(Control) <= kControlBytes);
static_assert(sizeof(WireRow) == 16);
// The block is a raw memcpy of Control: any change of these offsets is a wire-layout change and needs a new
// kControlVersion (decode checks magic and version before reading any other field, so a peer with another layout is
// refused there instead of misreading commands).
static_assert(offsetof(Control, magic) == 0 && offsetof(Control, version) == 4, "GLMC: magic / version lead the block");
static_assert(offsetof(Control, hello) == 200 && sizeof(Hello) == 72, "GLMC v6 layout: Hello");
static_assert(offsetof(Control, pflags) == 272 && offsetof(Control, budget) == 276 && offsetof(Control, cycles) == 280 &&
                  offsetof(Control, ppos) == 288 && offsetof(Control, sbudget) == 296 && offsetof(Control, import_id) == 328 &&
                  offsetof(Control, import_status) == 332 && offsetof(Control, import_token) == 336 && sizeof(Control) == 344,
              "GLMC v6 layout changed: bump kControlVersion and update these offsets");
static_assert(offsetof(Control, pad1) == 284 && offsetof(Control, pad2) == 340, "GLMC v6 layout: reserved fields");
static_assert(offsetof(Hello, model_hash) == 0 && offsetof(Hello, identity_hash) == 8 && offsetof(Hello, capacity) == 16 &&
                  offsetof(Hello, slots) == 24 && offsetof(Hello, max_rows) == 28 && offsetof(Hello, step_rows) == 32 &&
                  offsetof(Hello, world) == 36 && offsetof(Hello, mtp) == 40 && offsetof(Hello, mtp_kmax) == 44 &&
                  offsetof(Hello, g2) == 48 && offsetof(Hello, import_flags) == 52 && offsetof(Hello, import_wire) == 56 &&
                  offsetof(Hello, import_pad) == 60 && offsetof(Hello, import_model_hash) == 64,
              "GLMC v6 Hello layout changed: bump kControlVersion and update these offsets");

inline std::uint32_t fnv1a32(std::span<const std::uint32_t> tokens, std::uint32_t h = 2166136261U) {
    for (const std::uint32_t t : tokens)
        for (unsigned b = 0; b < 4; ++b) h = (h ^ ((t >> (8 * b)) & 255U)) * 16777619U;
    return h;
}
inline std::uint64_t fnv1a64(const void* p, std::size_t n, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
inline std::uint64_t fnv1a64(const std::string& s, std::uint64_t h = 14695981039346656037ULL) { return fnv1a64(s.data(), s.size(), h); }

class ProtocolError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using Block = std::array<std::byte, kControlBytes>;

inline Block encode(const Control& c) {
    Block b{};
    std::memcpy(b.data(), &c, sizeof(Control));
    return b;
}

// Structural validation of a received block (independent of the round state).
inline Control decode(const Block& b) {
    Control c;
    std::memcpy(&c, b.data(), sizeof(Control));
    if (c.magic != kControlMagic) throw ProtocolError("control channel: bad magic from the peer");
    if (c.version != kControlVersion)
        throw ProtocolError("control channel: protocol version " + std::to_string(c.version) + " from the peer, expected " +
                            std::to_string(kControlVersion));
    if (c.cmd > static_cast<std::uint32_t>(Cmd::import_drop)) throw ProtocolError("control channel: unknown command");
    if (c.cycles > kMaxMtpCycles) throw ProtocolError("control channel: too many MTP cycles");
    if (c.nrows > kMaxRows) throw ProtocolError("control channel: more than 8 rows");
    if (c.pslot >= 32) throw ProtocolError("control channel: prefill slot out of range");
    for (std::size_t b2 = sizeof(Control); b2 < kControlBytes; ++b2)
        if (b[b2] != std::byte{0}) throw ProtocolError("control channel: non-zero padding");
    return c;
}

// Empty when equal, otherwise the differing fields (local vs peer).
inline std::string hello_mismatch(const Hello& mine, const Hello& peer) {
    std::string s;
    auto f = [&](const char* name, std::uint64_t a, std::uint64_t b) {
        if (a == b) return;
        if (!s.empty()) s += ", ";
        s += std::string(name) + " " + std::to_string(a) + " vs peer " + std::to_string(b);
    };
    f("model_hash", mine.model_hash, peer.model_hash);
    f("identity_hash", mine.identity_hash, peer.identity_hash);
    f("capacity", mine.capacity, peer.capacity);
    f("slots", mine.slots, peer.slots);
    f("max_rows", mine.max_rows, peer.max_rows);
    f("step_rows", mine.step_rows, peer.step_rows);
    f("world", mine.world, peer.world);
    f("mtp", mine.mtp, peer.mtp);
    f("mtp_kmax", mine.mtp_kmax, peer.mtp_kmax);
    f("g2", mine.g2, peer.g2);
    f("import_flags", mine.import_flags, peer.import_flags);
    f("import_wire", mine.import_wire, peer.import_wire);
    f("import_model_hash", mine.import_model_hash, peer.import_model_hash);
    return s;
}

// Step rows as the lead sends them: 1..8 rows, strictly ascending slots (one row per slot), each slot < slots, each
// position < capacity. The follower refuses anything else before calling the engine.
inline void check_rows(std::span<const WireRow> rows, std::uint32_t slots, std::uint64_t capacity) {
    if (rows.empty() || rows.size() > kMaxRows) throw ProtocolError("control channel: step needs 1..8 rows");
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (rows[r].slot >= slots) throw ProtocolError("control channel: row slot out of range");
        if (rows[r].pos >= capacity) throw ProtocolError("control channel: row position beyond the capacity");
        if (r > 0 && rows[r].slot <= rows[r - 1].slot) throw ProtocolError("control channel: rows not in ascending slot order");
    }
}

// Windows of a spec_verify call: rows grouped by slot (sorted, one group per slot), positions consecutive per group.
struct Window {
    std::uint32_t slot = 0, row0 = 0, n = 0;
};
inline std::vector<Window> windows_of(std::span<const WireRow> rows, std::uint32_t slots, std::uint64_t capacity) {
    if (rows.empty() || rows.size() > kMaxRows) throw ProtocolError("control channel: spec_verify needs 1..8 rows");
    std::vector<Window> w;
    for (std::uint32_t r = 0; r < rows.size(); ++r) {
        if (rows[r].slot >= slots || rows[r].pos >= capacity) throw ProtocolError("control channel: spec row out of range");
        if (!w.empty() && rows[r].slot == w.back().slot) {
            if (rows[r].pos != rows[r - 1].pos + 1) throw ProtocolError("control channel: window positions not consecutive");
            ++w.back().n;
            continue;
        }
        if (!w.empty() && rows[r].slot < w.back().slot) throw ProtocolError("control channel: windows not in ascending slot order");
        w.push_back({rows[r].slot, r, 1});
    }
    return w;
}

// A slot's drafter state in batched self-speculation (both ranks hold the same; rank 0's lead reads it to size the
// windows): the drafts proposed for its next window and their merged confidences; `p` = the anchor position they follow.
struct SlotSpec {
    bool seeded = false, disabled = false;
    std::uint64_t p = 0;
    std::vector<std::uint32_t> drafts;
    std::vector<float> conf;
};

// KDA batches one call only when rows[r].slot == rows[0].slot + r (engine glm_fast_attn); otherwise per-row launches.
inline bool consecutive_slots(std::span<const WireRow> rows) {
    for (std::size_t r = 1; r < rows.size(); ++r)
        if (rows[r].slot != rows[0].slot + r) return false;
    return true;
}

// The two ranks' channel. `wire(out, in, layer, request)` moves one message of out.size() bytes each way (both ranks
// pass the same size and tag); world 1 passes an empty wire (the peer is this rank's own mirror).
class Channel {
public:
    using Wire = std::function<void(std::span<const std::byte>, std::span<std::byte>, std::uint32_t layer, std::uint64_t request)>;
    Channel(Wire wire, std::uint32_t rank) : wire_(std::move(wire)), rank_(rank) {}

    // One round: fills the bookkeeping fields of `mine` and returns the peer's block after every check.
    Control exchange(Control mine) {
        mine.magic = kControlMagic;
        mine.version = kControlVersion;
        mine.round = round_;
        mine.ack_round = round_ == 0 ? kNoRound : round_ - 1;
        mine.rank = rank_;
        mine.echo_round = echo_round_;
        mine.echo_count = echo_count_;
        mine.echo_token = echo_token_;
        mine.echo_hash = echo_hash_;
        Control peer;
        if (!wire_) {
            peer = mine;
            peer.rank = rank_ ^ 1U;
        } else {
            const Block out = encode(mine);
            Block in{};
            wire_(out, in, 0, round_);
            peer = decode(in);
        }
        if (peer.round != round_) throw ProtocolError("control channel: duplicate / stale round " + std::to_string(peer.round) +
                                                      ", expected " + std::to_string(round_));
        if (peer.ack_round != mine.ack_round) throw ProtocolError("control channel: preceding round not acknowledged");
        if (peer.rank == rank_) throw ProtocolError("control channel: peer reports the same rank");
        if (peer.echo_round != echo_round_ || peer.echo_count != echo_count_ || peer.echo_token != echo_token_ ||
            peer.echo_hash != echo_hash_)
            throw ProtocolError("cross-rank token mismatch: round " + std::to_string(echo_round_) + " local count " +
                                std::to_string(echo_count_) + " last " + std::to_string(echo_token_) + " hash " +
                                std::to_string(echo_hash_) + ", peer round " + std::to_string(peer.echo_round) + " count " +
                                std::to_string(peer.echo_count) + " last " + std::to_string(peer.echo_token) + " hash " +
                                std::to_string(peer.echo_hash));
        ++round_;
        return peer;
    }

    // Prompt tokens after a prefill block: the lead sends `tokens`, the follower receives `n` of them.
    void tokens(std::vector<std::uint32_t>& tokens, std::size_t n, bool sender) {
        if (sender && tokens.size() != n) throw std::logic_error("control channel: prompt size");
        if (!wire_) return;
        if (!sender) tokens.assign(n, 0);
        std::vector<std::uint32_t> out, in;
        for (std::size_t c0 = 0, k = 0; c0 < n; c0 += kTokensPerMessage, ++k) {
            const std::size_t m = std::min(kTokensPerMessage, n - c0);
            out.assign(m, 0);
            in.assign(m, 0);
            if (sender) std::memcpy(out.data(), tokens.data() + c0, m * 4);
            wire_(std::as_bytes(std::span<const std::uint32_t>(out)), std::as_writable_bytes(std::span<std::uint32_t>(in)),
                  1, (round_ << 20) | k);
            if (!sender) std::memcpy(tokens.data() + c0, in.data(), m * 4);
        }
    }

    // Tokens produced by the command of round `round` (the round already exchanged: round() - 1).
    void record(std::uint64_t round, std::span<const std::uint32_t> produced) {
        echo_round_ = round;
        echo_count_ = static_cast<std::uint32_t>(produced.size());
        echo_token_ = produced.empty() ? 0U : produced.back();
        echo_hash_ = fnv1a32(produced);
    }

    // hello on both ranks (each sends Cmd::hello with its own fields): throws on any mismatch.
    void hello(const Hello& mine) {
        Control c;
        c.cmd = static_cast<std::uint32_t>(Cmd::hello);
        c.hello = mine;
        const Control peer = exchange(c);
        if (peer.cmd != static_cast<std::uint32_t>(Cmd::hello)) throw ProtocolError("control channel: peer did not open with hello");
        const std::string m = hello_mismatch(mine, peer.hello);
        if (!m.empty()) throw ProtocolError("hello mismatch between the ranks: " + m);
    }

    [[nodiscard]] std::uint64_t round() const noexcept { return round_; }
    [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }

private:
    Wire wire_;
    std::uint32_t rank_ = 0;
    std::uint64_t round_ = 0;
    std::uint64_t echo_round_ = kNoRound;
    std::uint32_t echo_count_ = 0, echo_token_ = 0, echo_hash_ = 2166136261U;
};

// The engine calls of one command, made identically by both ranks (the lead after sending it, the follower after
// receiving it). Eng: reset(slot); prefill(slot, tokens, pos0) -> next; step(span<const WireRow>) -> tokens;
// prefill_mtp(slot, tokens, pos0, last) -> next (last: seed kept); mtp_enter(row index, WireRow); mtp_run(slot, budget, cycles) -> tokens;
// mtp_leave(slot); spec_verify(rows, budgets) -> per window [count, tokens..]; spec_draft(slot) -> draft. `prompt`: the lead's prompt (sent), or empty on the follower (received). Returns the produced tokens.
template<class Eng>
std::vector<std::uint32_t> replay(Eng& eng, Channel& ch, const Control& c, std::vector<std::uint32_t>& prompt, bool sender,
                                  std::uint32_t slots, std::uint64_t capacity) {
    const auto cmd = static_cast<Cmd>(c.cmd);
    if (slots < 32 && (c.resets >> slots) != 0) throw ProtocolError("control channel: reset of a slot out of range");
    for (std::uint32_t s = 0; s < slots && s < 32; ++s)
        if ((c.resets >> s) & 1U) eng.reset(s);
    std::vector<std::uint32_t> out;
    switch (cmd) {
        case Cmd::prefill: {
            if (c.pslot >= slots || (c.ppos == 0 && ((c.resets >> c.pslot) & 1U) == 0))
                throw ProtocolError("control channel: prefill without its slot reset");
            if (c.plen == 0 || c.ppos >= capacity || c.plen >= capacity - c.ppos)
                throw ProtocolError("control channel: prompt chunk out of range");
            ch.tokens(prompt, c.plen, sender);
            const std::span<const std::uint32_t> p(prompt);
            out.push_back((c.pflags & kPrefillMtp) != 0 ? eng.prefill_mtp(c.pslot, p, c.ppos, (c.pflags & kPrefillLast) != 0)
                                                        : eng.prefill(c.pslot, p, c.ppos));
            break;
        }
        case Cmd::import_state: {
            if (c.pslot >= slots || ((c.resets >> c.pslot) & 1U) == 0) throw ProtocolError("control channel: import without its slot reset");
            if (c.plen < 2 || c.plen >= capacity) throw ProtocolError("control channel: import prompt length out of range");
            ch.tokens(prompt, c.plen, sender);
            // A failed restore / last-token call is local to this rank: the slot is reset (import_abort) and the lead
            // learns both ranks' outcome from the next reply. No echo: the ranks may differ here by design.
            std::uint32_t y = kImportFailed;
            try {
                y = eng.import_state(c.pslot, c.import_id, std::span<const std::uint32_t>(prompt), (c.pflags & kPrefillMtp) != 0,
                                     (c.pflags & kImportDsaOnce) != 0);
            } catch (const ImportFatal&) {
                throw;  // inside the engine collectives after both ranks restored: the engine's fatal contract
            } catch (const std::exception& e) {
                eng.import_abort(c.pslot, c.import_id, e.what());
                y = kImportFailed;
            }
            out.push_back(y);
            return out;
        }
        case Cmd::import_drop: {
            if constexpr (requires { eng.import_drop(c.import_id); }) eng.import_drop(c.import_id);
            return out;
        }
        case Cmd::import_expect: {
            if (c.plen < 2 || c.plen >= capacity) throw ProtocolError("control channel: import prompt length out of range");
            ch.tokens(prompt, c.plen, sender);
            eng.import_expect(c.import_id, std::span<const std::uint32_t>(prompt));
            return out;
        }
        case Cmd::step: {
            const std::span<const WireRow> rows(c.row, c.nrows);
            check_rows(rows, slots, capacity);
            out = eng.step(rows);
            if (out.size() != rows.size()) throw std::runtime_error("engine step returned a wrong row count");
            break;
        }
        case Cmd::mtp_enter: {
            if (c.pslot >= kMaxRows || c.row[0].slot >= slots || c.row[0].pos + 1 >= capacity)
                throw ProtocolError("control channel: mtp_enter out of range");
            eng.mtp_enter(c.pslot, c.row[0], (c.pflags & kEnterSeed) != 0);
            return out;
        }
        case Cmd::mtp_run: {
            if (c.pslot >= slots || c.cycles == 0 || c.budget == 0) throw ProtocolError("control channel: mtp_run out of range");
            out = eng.mtp_run(c.pslot, c.budget, c.cycles);
            break;
        }
        case Cmd::spec_verify: {
            const std::span<const WireRow> rows(c.row, c.nrows);
            const auto w = windows_of(rows, slots, capacity);
            for (std::size_t i = 0; i < w.size(); ++i)
                if (c.sbudget[i] == 0) throw ProtocolError("control channel: spec window without budget");
            out = eng.spec_verify(rows, std::span<const std::uint32_t>(c.sbudget, w.size()));
            break;
        }
        case Cmd::spec_draft: {
            if (c.pslot >= slots) throw ProtocolError("control channel: spec_draft out of range");
            out.push_back(eng.spec_draft(c.pslot));
            break;
        }
        case Cmd::mtp_leave: {
            if (c.pslot >= slots) throw ProtocolError("control channel: mtp_leave out of range");
            eng.mtp_leave(c.pslot);
            return out;
        }
        case Cmd::state_digest: {
            if (c.pslot >= slots || c.ppos == 0 || c.ppos >= capacity) throw ProtocolError("control channel: state_digest out of range");
            if constexpr (requires { eng.state_digest(c.pslot, c.ppos); }) {
                std::vector<std::uint32_t> mine = eng.state_digest(c.pslot, c.ppos), peer;
                // the same layout on both ranks (hello): rank 1 sends its words, rank 0 receives them
                if (sender) ch.tokens(peer, mine.size(), false);
                else ch.tokens(mine, mine.size(), true);
                out = std::move(mine);  // rank 0: its digests, then rank 1's
                out.insert(out.end(), peer.begin(), peer.end());
                return out;
            } else {
                throw ProtocolError("control channel: state_digest not supported by this engine");
            }
        }
        case Cmd::heartbeat:
        case Cmd::shutdown: return out;
        case Cmd::hello:
        case Cmd::reply: throw ProtocolError(std::string("control channel: unexpected ") + cmd_name(cmd));
    }
    ch.record(ch.round() - 1, out);
    return out;
}

}  // namespace strix::glm::serve
