#include "h3/vae/vae_plan.hpp"
#include "h3/vae/vae_threads.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace strix::h3::vae {

Tiles1D split_tiles(int input_len) {
    constexpr int tile = DecodePlan::tile_size, omin = DecodePlan::tile_overlap_min, ratio = DecodePlan::ratio;
    Tiles1D t;
    if (tile >= input_len) {
        t.start = {0};
        t.len = {input_len};
        return t;
    }
    int n = (input_len + tile - 1) / tile;
    int remaining = 0;
    for (;;) {
        remaining = tile * n - omin * (n - 1) - input_len;
        if (remaining < 0) ++n;
        else break;
    }
    t.overlap.assign(std::size_t(n - 1), omin);
    for (int i = 0; i < remaining / ratio; ++i) t.overlap[std::size_t(i % (n - 1))] += ratio;
    t.start = {0};
    for (int i = 0; i < n - 1; ++i) t.start.push_back(t.start.back() + tile - t.overlap[std::size_t(i)]);
    t.len.assign(std::size_t(n), tile);
    return t;
}

namespace {
int pad_frames(int z_len, int pad_tokens) {
    if (pad_tokens <= 0) return 0;
    constexpr int intra_tail = DecodePlan::clip_length % DecodePlan::ratio_t;
    if (intra_tail == 0) return pad_tokens * DecodePlan::ratio_t;
    const int before = z_len - pad_tokens;
    int s = 0;
    for (int k = 0; k < pad_tokens; ++k) s += (before + k) % DecodePlan::tokens_chunk_size == 0 ? intra_tail : DecodePlan::ratio_t;
    return s;
}
int frame_plan(int z_len, int num_chunks, int pad_tokens) {
    constexpr int chunk_dec = DecodePlan::tokens_chunk_size * DecodePlan::ratio_t;
    constexpr int split_count = (DecodePlan::token_drop > 0 ? 1 : 0) + 1;
    int total = 0, final_overlap = 0;
    for (int i = 0; i < num_chunks; ++i) {
        const int ts = i * DecodePlan::tokens_chunk_size, te = ts + DecodePlan::tokens_chunk_size + DecodePlan::token_overlap;
        const int clip_tokens = std::max(0, std::min(te, z_len) - std::min(ts, z_len));
        const int clip_frames = clip_tokens * DecodePlan::ratio_t;
        for (int j = 0; j < split_count; ++j) {
            const int fs = j * chunk_dec, fe = std::min(fs + chunk_dec, clip_frames);
            const int frames = std::max(0, fe - fs - DecodePlan::frame_pre_padding);
            if (j == 0) total += frames;
            else final_overlap = frames;
        }
    }
    total += final_overlap;
    return total - pad_frames(z_len, pad_tokens);
}
}  // namespace

DecodePlan make_plan(int T, int H, int W) {
    if (T < 2 || H < 1 || W < 1) throw std::invalid_argument("h3 vae plan: need T >= 2");
    DecodePlan p;
    p.T = T; p.H = H; p.W = W;
    int pseudo = T + DecodePlan::token_drop;
    p.pad_tokens = (DecodePlan::tokens_chunk_size - pseudo % DecodePlan::tokens_chunk_size) % DecodePlan::tokens_chunk_size;
    pseudo += p.pad_tokens;
    p.num_chunks = pseudo / DecodePlan::tokens_chunk_size - (DecodePlan::token_drop > 0 ? 1 : 0);
    if (p.num_chunks < 1) {
        p.pad_tokens += DecodePlan::tokens_chunk_size;
        p.num_chunks += 1;
    }
    const int zlen = T + p.pad_tokens;
    for (int i = 0; i < p.num_chunks; ++i) {
        const int t0 = i * DecodePlan::tokens_chunk_size;
        p.chunks.push_back({std::min(t0, zlen), std::min(t0 + DecodePlan::tokens_chunk_size + DecodePlan::token_overlap, zlen)});
    }
    p.ty = split_tiles(H * DecodePlan::ratio);
    p.tx = split_tiles(W * DecodePlan::ratio);
    p.out_frames = frame_plan(zlen, p.num_chunks, p.pad_tokens);
    p.out_h = H * DecodePlan::ratio;
    p.out_w = W * DecodePlan::ratio;
    return p;
}

std::vector<DecoderCall> decoder_calls(const DecodePlan& p) {
    std::vector<DecoderCall> out;
    for (int c = 0; c < p.num_chunks; ++c)
        for (std::size_t i = 0; i < p.ty.start.size(); ++i)
            for (std::size_t j = 0; j < p.tx.start.size(); ++j) {
                DecoderCall d;
                d.index = int(out.size());
                d.chunk = c; d.ti = int(i); d.tj = int(j);
                d.t0 = p.chunks[std::size_t(c)].t0; d.nt = p.chunks[std::size_t(c)].t1 - d.t0;
                d.y0 = p.ty.start[i] / DecodePlan::ratio; d.ny = p.ty.len[i] / DecodePlan::ratio;
                d.x0 = p.tx.start[j] / DecodePlan::ratio; d.nx = p.tx.len[j] / DecodePlan::ratio;
                out.push_back(d);
            }
    return out;
}

CallRange partition(const DecodePlan& p, int rank, int world) {
    const int n = p.calls();
    return {int(std::int64_t(n) * rank / world), int(std::int64_t(n) * (rank + 1) / world)};
}

Pix16 blend(const Pix16& a, const Pix16& b, int extent, int axis) {
    const int la = axis == 0 ? a.F : axis == 1 ? a.Hh : a.Ww;
    const int lb = axis == 0 ? b.F : axis == 1 ? b.Hh : b.Ww;
    const int e = std::min({la, lb, extent});
    Pix16 out = b;  // shape of b; the first e slices along `axis` are replaced
    if (e <= 0) return out;
    std::vector<float> wa(static_cast<std::size_t>(e)), wb(static_cast<std::size_t>(e));
    for (int k = 0; k < e; ++k) {
        const float pos = h2f(f2h(float(k)));
        const float q = rh(pos / float(e));
        wb[std::size_t(k)] = q;
        wa[std::size_t(k)] = rh(1.0F - q);
    }
    for (int c = 0; c < b.C; ++c)
        for (int f = 0; f < b.F; ++f)
            for (int y = 0; y < b.Hh; ++y)
                for (int x = 0; x < b.Ww; ++x) {
                    const int k = axis == 0 ? f : axis == 1 ? y : x;
                    if (k >= e) continue;
                    const int fa = axis == 0 ? a.F - e + f : f, ya = axis == 1 ? a.Hh - e + y : y, xa = axis == 2 ? a.Ww - e + x : x;
                    const float pa = rh(h2f(a.at(c, fa, ya, xa)) * wa[std::size_t(k)]);
                    const float pb = rh(h2f(b.at(c, f, y, x)) * wb[std::size_t(k)]);
                    out.at(c, f, y, x) = f2h(pa + pb);
                }
    return out;
}

namespace {
Pix16 crop(const Pix16& s, int f0, int nf, int y0, int ny, int x0, int nx) {
    Pix16 o;
    o.C = s.C; o.F = nf; o.Hh = ny; o.Ww = nx;
    o.v.resize(std::size_t(o.C) * nf * ny * nx);
    for (int c = 0; c < s.C; ++c)
        for (int f = 0; f < nf; ++f)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) o.at(c, f, y, x) = s.at(c, f0 + f, y0 + y, x0 + x);
    return o;
}
void paste(Pix16& dst, const Pix16& s, int y0, int x0) {
    for (int c = 0; c < s.C; ++c)
        for (int f = 0; f < s.F; ++f)
            for (int y = 0; y < s.Hh; ++y)
                for (int x = 0; x < s.Ww; ++x) dst.at(c, f, y0 + y, x0 + x) = s.at(c, f, y, x);
}
}  // namespace

Pix16 assemble_chunk(const DecodePlan& p, const std::function<Pix16(int, int)>& tile_fn) {
    const int height = p.out_h, width = p.out_w;
    const auto& yi = p.ty;
    const auto& xi = p.tx;
    const int ny = int(yi.start.size()), nx = int(xi.start.size());
    Pix16 canvas, strip;
    bool have_canvas = false;
    int out_y = 0;
    for (int i = 0; i < ny; ++i) {
        Pix16 new_strip, left_tail;
        bool have_new_strip = false;
        int out_x = 0, tile_height = 0;
        for (int j = 0; j < nx; ++j) {
            Pix16 tile = tile_fn(i, j);
            if (i > 0) {
                const Pix16 above = crop(strip, 0, strip.F, 0, strip.Hh, xi.start[std::size_t(j)],
                                         std::min(xi.len[std::size_t(j)], strip.Ww - xi.start[std::size_t(j)]));
                tile = blend(above, tile, yi.overlap[std::size_t(i - 1)], 1);
            }
            if (j > 0) tile = blend(left_tail, tile, xi.overlap[std::size_t(j - 1)], 2);
            if (j < nx - 1) {
                const int ov = xi.overlap[std::size_t(j)];
                left_tail = crop(tile, 0, tile.F, 0, tile.Hh, tile.Ww - ov, ov);
                tile = crop(tile, 0, tile.F, 0, tile.Hh, 0, tile.Ww - ov);
            }
            if (!have_canvas) {
                canvas.C = tile.C; canvas.F = tile.F; canvas.Hh = height; canvas.Ww = width;
                canvas.v.assign(std::size_t(canvas.C) * canvas.F * height * width, 0);
                have_canvas = true;
            }
            if (i < ny - 1) {
                const int ov = yi.overlap[std::size_t(i)];
                if (!have_new_strip) {
                    new_strip.C = tile.C; new_strip.F = tile.F; new_strip.Hh = ov; new_strip.Ww = width;
                    new_strip.v.assign(std::size_t(new_strip.C) * new_strip.F * ov * width, 0);
                    have_new_strip = true;
                }
                paste(new_strip, crop(tile, 0, tile.F, tile.Hh - ov, ov, 0, tile.Ww), 0, out_x);
                tile = crop(tile, 0, tile.F, 0, tile.Hh - ov, 0, tile.Ww);
            }
            paste(canvas, tile, out_y, out_x);
            tile_height = tile.Hh;
            out_x += tile.Ww;
        }
        strip = std::move(new_strip);
        out_y += tile_height;
    }
    return canvas;
}

std::vector<float> assemble_video(const DecodePlan& p, const std::function<Pix16(int)>& chunk_canvas) {
    constexpr int chunk_dec = DecodePlan::tokens_chunk_size * DecodePlan::ratio_t;
    // ImageNet constants as FP16 buffers (the module is cast to the working dtype), upcast to FP32.
    const float mean16[3] = {rh(0.485F), rh(0.456F), rh(0.406F)};
    const float std16[3] = {rh(0.229F), rh(0.224F), rh(0.225F)};
    const std::size_t plane = std::size_t(p.out_h) * p.out_w;
    std::vector<float> out(std::size_t(3) * p.out_frames * plane, 0.0F);
    int write_pos = 0;
    const auto write_part = [&](const Pix16& part) {
        const int copy = std::min(part.F, std::max(0, p.out_frames - write_pos));
        // elementwise: (channel, frame) planes on worker threads (independent outputs)
        const int jobs = 3 * copy;
        const int nt = std::max(1, std::min<int>(int(std::thread::hardware_concurrency()), jobs));
        Workers pool;
        for (int t = 0; t < nt; ++t)
            pool.run([&, t] {
                for (int j = t; j < jobs; j += nt) {
                    const int c = j / copy, f = j % copy;
                    for (std::size_t q = 0; q < plane; ++q) {
                        float v = h2f(part.v[(std::size_t(c) * part.F + f) * plane + q]) * std16[c];
                        v = v + mean16[c];
                        v = v < 0.0F ? 0.0F : (v > 1.0F ? 1.0F : v);  // clamp to [0, 1]; a NaN input stays NaN (the reference clamp keeps NaN)
                        out[(std::size_t(c) * p.out_frames + write_pos + f) * plane + q] = v;
                    }
                }
            });
        pool.join();
        write_pos += copy;
    };
    Pix16 overlap;
    bool have_overlap = false;
    for (int i = 0; i < p.num_chunks; ++i) {
        const Pix16 dec = chunk_canvas(i);
        for (int j = 0; j < 2; ++j) {
            const int fs = j * chunk_dec, fe = std::min(fs + chunk_dec, dec.F);
            const int f0 = std::min(fs + DecodePlan::frame_pre_padding, fe);
            Pix16 part = crop(dec, f0, fe - f0, 0, dec.Hh, 0, dec.Ww);
            if (j == 0) {
                if (have_overlap) {
                    part = blend(overlap, part, DecodePlan::frame_overlap, 0);
                    have_overlap = false;
                }
                if (part.F > 0) write_part(part);
            } else {
                overlap = std::move(part);
                have_overlap = true;
            }
        }
        if (i == p.num_chunks - 1 && have_overlap) {
            if (overlap.F > 0) write_part(overlap);
            have_overlap = false;
        }
    }
    return out;
}

std::vector<half_bits> denormalize_and_pad(const DecodePlan& p, const std::vector<half_bits>& z,
                                           const half_bits* latents_mean, const half_bits* latents_std) {
    const std::size_t hw = std::size_t(p.H) * p.W;
    if (z.size() != std::size_t(24) * std::size_t(p.T) * hw) throw std::invalid_argument("h3 vae: latent length != 24 x T x H x W");
    const int tp = p.T + p.pad_tokens;
    std::vector<half_bits> out(std::size_t(24) * tp * hw);
    for (int c = 0; c < 24; ++c) {
        const float s = h2f(latents_std[c]), m = h2f(latents_mean[c]);
        for (int t = 0; t < tp; ++t) {
            const int ts = std::min(t, p.T - 1);
            for (std::size_t q = 0; q < hw; ++q) {
                const float v = rh(h2f(z[(std::size_t(c) * p.T + ts) * hw + q]) * s);
                out[(std::size_t(c) * tp + t) * hw + q] = f2h(v + m);
            }
        }
    }
    return out;
}

}  // namespace strix::h3::vae
