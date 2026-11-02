// Two-node gather of the VAE decode: call split, pieces, wire format, and the incremental frame assembler.
#include "h3/vae/vae_shard.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "h3/vae/vae_threads.hpp"

namespace strix::h3::vae {

namespace {
int calls_per_chunk(const DecodePlan& p) { return int(p.ty.start.size() * p.tx.start.size()); }

Pix16 crop_frames(const Pix16& s, int f0, int nf) {
    Pix16 o;
    o.C = s.C; o.F = nf; o.Hh = s.Hh; o.Ww = s.Ww;
    const std::size_t plane = std::size_t(s.Hh) * s.Ww;
    o.v.resize(std::size_t(o.C) * nf * plane);
    for (int c = 0; c < s.C; ++c)
        for (int f = 0; f < nf; ++f)
            std::memcpy(o.v.data() + (std::size_t(c) * nf + f) * plane, s.v.data() + (std::size_t(c) * s.F + f0 + f) * plane, plane * 2);
    return o;
}

bool shape_is(const Pix16& t, int C, int F, int H, int W) {
    return t.C == C && t.F == F && t.Hh == H && t.Ww == W && t.v.size() == std::size_t(C) * F * H * W;
}

// little-endian byte writer / reader (the nodes and the PC are little-endian; values are copied byte-wise)
void put32(std::vector<std::uint8_t>& b, std::int32_t v) {
    std::uint8_t x[4];
    std::memcpy(x, &v, 4);
    b.insert(b.end(), x, x + 4);
}
struct Reader {
    const std::uint8_t* p;
    std::size_t n, at = 0;
    std::int32_t i32() {
        if (n - at < 4) throw std::invalid_argument("h3 vae piece: truncated");
        std::int32_t v;
        std::memcpy(&v, p + at, 4);
        at += 4;
        return v;
    }
};
constexpr std::int32_t kMagic = 0x50563348;  // 'H3VP'
}  // namespace

CallRange split_calls(const DecodePlan& p, int rank, int world, int shift) {
    const int n = p.calls();
    if (world == 1 && rank == 0) return {0, n};
    if (world != 2 || rank < 0 || rank > 1) throw std::invalid_argument("h3 vae split_calls: world must be 1 or 2");
    const int cut = std::max(0, std::min(n, n / 2 + shift));
    return rank == 0 ? CallRange{0, cut} : CallRange{cut, n};
}

std::vector<PieceSpan> piece_spans(const DecodePlan& p, CallRange r) {
    const int per = calls_per_chunk(p);
    if (r.first < 0 || r.last > p.calls() || r.first > r.last) throw std::invalid_argument("h3 vae piece_spans: call range");
    std::vector<PieceSpan> out;
    for (int a = r.first; a < r.last;) {
        const int c = a / per, b = std::min(r.last, (c + 1) * per);
        out.push_back({c, a, b, a == c * per && b == (c + 1) * per});
        a = b;
    }
    return out;
}

std::vector<std::uint8_t> pack_piece(const DecodePiece& piece) {
    std::vector<const Pix16*> ts;
    if (piece.whole) ts.push_back(&piece.canvas);
    else for (const auto& t : piece.tiles) ts.push_back(&t);
    std::size_t bytes = 7 * 4;
    for (const Pix16* t : ts) bytes += 16 + t->v.size() * 2;
    std::vector<std::uint8_t> b;
    b.reserve(bytes);
    put32(b, kMagic); put32(b, 1); put32(b, piece.chunk); put32(b, piece.call_first); put32(b, piece.call_last);
    put32(b, piece.whole ? 1 : 0); put32(b, std::int32_t(ts.size()));
    for (const Pix16* t : ts) {
        put32(b, t->C); put32(b, t->F); put32(b, t->Hh); put32(b, t->Ww);
        const auto* d = reinterpret_cast<const std::uint8_t*>(t->v.data());
        b.insert(b.end(), d, d + t->v.size() * 2);
    }
    return b;
}

DecodePiece unpack_piece(const DecodePlan& p, const std::uint8_t* data, std::size_t size) {
    Reader r{data, size};
    if (r.i32() != kMagic || r.i32() != 1) throw std::invalid_argument("h3 vae piece: bad magic or version");
    DecodePiece x;
    x.chunk = r.i32(); x.call_first = r.i32(); x.call_last = r.i32();
    const std::int32_t whole = r.i32(), n = r.i32();
    const int per = calls_per_chunk(p);
    if (x.chunk < 0 || x.chunk >= p.num_chunks || x.call_first < x.chunk * per || x.call_last > (x.chunk + 1) * per ||
        x.call_first >= x.call_last || (whole != 0 && whole != 1))
        throw std::invalid_argument("h3 vae piece: header outside the plan");
    x.whole = whole == 1;
    if (x.whole && (x.call_first != x.chunk * per || x.call_last != (x.chunk + 1) * per || n != 1))
        throw std::invalid_argument("h3 vae piece: whole piece must cover its chunk with one canvas");
    if (!x.whole && n != x.call_last - x.call_first) throw std::invalid_argument("h3 vae piece: tile count != call count");
    const auto calls = decoder_calls(p);
    for (std::int32_t k = 0; k < n; ++k) {
        Pix16 t;
        t.C = r.i32(); t.F = r.i32(); t.Hh = r.i32(); t.Ww = r.i32();
        const DecoderCall& c = calls[std::size_t(x.call_first + (x.whole ? 0 : k))];
        const bool ok = x.whole ? (t.C == 3 && t.F == 4 * c.nt && t.Hh == p.out_h && t.Ww == p.out_w)
                                : (t.C == 3 && t.F == 4 * c.nt && t.Hh == 16 * c.ny && t.Ww == 16 * c.nx);
        if (!ok) throw std::invalid_argument("h3 vae piece: tensor shape does not match the plan");
        const std::size_t cnt = std::size_t(t.C) * t.F * t.Hh * t.Ww;
        if ((r.n - r.at) / 2 < cnt) throw std::invalid_argument("h3 vae piece: truncated payload");
        t.v.resize(cnt);
        std::memcpy(t.v.data(), r.p + r.at, cnt * 2);
        r.at += cnt * 2;
        if (x.whole) x.canvas = std::move(t);
        else x.tiles.push_back(std::move(t));
    }
    if (r.at != r.n) throw std::invalid_argument("h3 vae piece: trailing bytes");
    return x;
}

struct FrameAssembler::State {
    DecodePlan p;
    int per = 0, done = 0;
    std::vector<float> out;
    std::vector<std::vector<Pix16>> tiles;      // per chunk: decoder outputs of a cut chunk (slot per call)
    std::vector<std::vector<char>> have_tile;
    std::vector<int> tiles_in;
    std::vector<char> have_canvas, written;     // canvas delivered (whole or assembled); frames written
    std::vector<Pix16> canvas;                  // held until the predecessor's tail is present
    std::vector<Pix16> tail;                    // part 1 (frames [23, 28)) of each chunk once its canvas is in
    std::vector<int> pos;                       // first output frame of each chunk's part 0
    mutable std::mutex mu;

    void finalize(const Pix16& part, int write_pos) {  // the finalize of assemble_video, same arithmetic
        const float mean16[3] = {rh(0.485F), rh(0.456F), rh(0.406F)};
        const float std16[3] = {rh(0.229F), rh(0.224F), rh(0.225F)};
        const std::size_t plane = std::size_t(p.out_h) * p.out_w;
        const int copy = std::min(part.F, std::max(0, p.out_frames - write_pos));
        const int jobs = 3 * copy;
        if (jobs <= 0) return;
        const int nt = std::max(1, std::min<int>(int(std::thread::hardware_concurrency()), jobs));
        Workers pool;
        for (int t = 0; t < nt; ++t)
            pool.run([&, t] {
                for (int j = t; j < jobs; j += nt) {
                    const int c = j / copy, f = j % copy;
                    for (std::size_t q = 0; q < plane; ++q) {
                        float v = h2f(part.v[(std::size_t(c) * part.F + f) * plane + q]) * std16[c];
                        v = v + mean16[c];
                        v = v < 0.0F ? 0.0F : (v > 1.0F ? 1.0F : v);
                        out[(std::size_t(c) * p.out_frames + write_pos + f) * plane + q] = v;
                    }
                }
            });
        pool.join();
    }
    static int part0_frames(int F) {
        constexpr int chunk_dec = DecodePlan::tokens_chunk_size * DecodePlan::ratio_t;
        const int fe = std::min(chunk_dec, F);
        return fe - std::min(DecodePlan::frame_pre_padding, fe);
    }
    // temporal step of chunk i (needs canvas i and, for i > 0, tail i - 1): mirrors assemble_video's loop body
    void temporal(int i) {
        constexpr int chunk_dec = DecodePlan::tokens_chunk_size * DecodePlan::ratio_t;
        const Pix16 dec = std::move(canvas[std::size_t(i)]);
        const int fe = std::min(chunk_dec, dec.F), f0 = std::min(DecodePlan::frame_pre_padding, fe);
        Pix16 part = crop_frames(dec, f0, fe - f0);
        if (i > 0) {
            part = blend(tail[std::size_t(i - 1)], part, DecodePlan::frame_overlap, 0);
            Pix16().v.swap(tail[std::size_t(i - 1)].v);
        }
        if (part.F > 0) finalize(part, pos[std::size_t(i)]);
        if (i == p.num_chunks - 1 && tail[std::size_t(i)].F > 0) finalize(tail[std::size_t(i)], pos[std::size_t(i)] + part.F);
        written[std::size_t(i)] = 1;
        ++done;
    }
    void canvas_in(int i, Pix16&& cv) {
        constexpr int chunk_dec = DecodePlan::tokens_chunk_size * DecodePlan::ratio_t;
        const int fs = chunk_dec, fe = std::min(fs + chunk_dec, cv.F), f0 = std::min(fs + DecodePlan::frame_pre_padding, fe);
        tail[std::size_t(i)] = crop_frames(cv, f0, fe - f0);
        canvas[std::size_t(i)] = std::move(cv);
        have_canvas[std::size_t(i)] = 1;
        if (i == 0 || have_canvas[std::size_t(i - 1)]) temporal(i);
        if (i + 1 < p.num_chunks && have_canvas[std::size_t(i + 1)] && !written[std::size_t(i + 1)]) temporal(i + 1);
    }
};

FrameAssembler::FrameAssembler(const DecodePlan& p) : s_(std::make_unique<State>()) {
    State& s = *s_;
    s.p = p;
    s.per = calls_per_chunk(p);
    const std::size_t n = std::size_t(p.num_chunks);
    s.out.assign(std::size_t(3) * p.out_frames * std::size_t(p.out_h) * p.out_w, 0.0F);
    s.tiles.resize(n); s.have_tile.resize(n); s.tiles_in.assign(n, 0);
    s.have_canvas.assign(n, 0); s.written.assign(n, 0); s.canvas.resize(n); s.tail.resize(n); s.pos.assign(n, 0);
    int at = 0;
    for (std::size_t i = 0; i < n; ++i) {
        s.pos[i] = at;
        at += State::part0_frames(4 * (p.chunks[i].t1 - p.chunks[i].t0));
    }
}
FrameAssembler::~FrameAssembler() = default;

void FrameAssembler::push(DecodePiece&& x) {
    State& s = *s_;
    const std::lock_guard<std::mutex> lock(s.mu);
    const DecodePlan& p = s.p;
    if (x.chunk < 0 || x.chunk >= p.num_chunks || x.call_first < x.chunk * s.per || x.call_last > (x.chunk + 1) * s.per ||
        x.call_first >= x.call_last)
        throw std::invalid_argument("h3 vae assembler: piece outside the plan");
    const std::size_t c = std::size_t(x.chunk);
    if (s.have_canvas[c]) throw std::invalid_argument("h3 vae assembler: chunk delivered twice");
    const int F = 4 * (p.chunks[c].t1 - p.chunks[c].t0);
    if (x.whole) {
        if (x.call_first != x.chunk * s.per || x.call_last != (x.chunk + 1) * s.per || s.tiles_in[c] != 0 ||
            !shape_is(x.canvas, 3, F, p.out_h, p.out_w))
            throw std::invalid_argument("h3 vae assembler: whole piece shape / coverage");
        s.canvas_in(x.chunk, std::move(x.canvas));
        return;
    }
    if (int(x.tiles.size()) != x.call_last - x.call_first) throw std::invalid_argument("h3 vae assembler: tile count != call count");
    auto& slot = s.tiles[c];
    auto& have = s.have_tile[c];
    if (slot.empty()) { slot.resize(std::size_t(s.per)); have.assign(std::size_t(s.per), 0); }
    const int nx = int(p.tx.start.size());
    for (int k = x.call_first; k < x.call_last; ++k) {
        const int j = k - x.chunk * s.per, ti = j / nx, tj = j % nx;
        Pix16& t = x.tiles[std::size_t(k - x.call_first)];
        if (have[std::size_t(j)]) throw std::invalid_argument("h3 vae assembler: call delivered twice");
        if (!shape_is(t, 3, F, p.ty.len[std::size_t(ti)], p.tx.len[std::size_t(tj)]))
            throw std::invalid_argument("h3 vae assembler: tile shape does not match the plan");
    }
    for (int k = x.call_first; k < x.call_last; ++k) {
        const int j = k - x.chunk * s.per;
        slot[std::size_t(j)] = std::move(x.tiles[std::size_t(k - x.call_first)]);
        have[std::size_t(j)] = 1;
        ++s.tiles_in[c];
    }
    if (s.tiles_in[c] == s.per) {
        Pix16 cv = assemble_chunk(p, [&](int i, int j) { return slot[std::size_t(i * nx + j)]; });
        std::vector<Pix16>().swap(slot);
        s.canvas_in(x.chunk, std::move(cv));
    }
}

bool FrameAssembler::complete() const {
    const std::lock_guard<std::mutex> lock(s_->mu);
    return s_->done == s_->p.num_chunks;
}
int FrameAssembler::chunks_done() const {
    const std::lock_guard<std::mutex> lock(s_->mu);
    return s_->done;
}
std::vector<float> FrameAssembler::take() {
    const std::lock_guard<std::mutex> lock(s_->mu);
    if (s_->done != s_->p.num_chunks) throw std::logic_error("h3 vae assembler: frames taken before every chunk arrived");
    return std::move(s_->out);
}

}  // namespace strix::h3::vae
