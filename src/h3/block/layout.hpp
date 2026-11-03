#pragma once
// MiniMax H3 packed sequence layout and per-forward timestep plan (host; port of the reference PackedLayout and
// the _forward segment/timestep logic). Packed rows: [text | cond rows (keyframes) or reference blocks | audio | video].
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/block/ops_ref.hpp"

namespace strix::h3::block {

inline constexpr double kFrameRescale = 5.0 / 3.0;
inline constexpr unsigned kFramePerToken[5] = {1, 4, 4, 4, 4};
inline constexpr double kVisualCondT = 0.999;
inline constexpr double kAudioCondT = 1.0;

enum class SegKind { text, cond, cond_audio, ref_img, ref_audio, audio, video };
inline const char* seg_kind_name(SegKind k) {
    switch (k) {
        case SegKind::text: return "text";
        case SegKind::cond: return "cond";
        case SegKind::cond_audio: return "cond_audio";
        case SegKind::ref_img: return "ref_img";
        case SegKind::ref_audio: return "ref_audio";
        case SegKind::audio: return "audio";
        case SegKind::video: return "video";
    }
    return "?";
}
inline unsigned seg_tag(SegKind k) {  // adaLN modality tag: video 0, text 1, audio 2
    switch (k) {
        case SegKind::text: return 1;
        case SegKind::cond_audio: case SegKind::ref_audio: case SegKind::audio: return 2;
        default: return 0;
    }
}
inline bool seg_is_video_like(SegKind k) { return k == SegKind::cond || k == SegKind::ref_img || k == SegKind::video; }

struct Segment {
    std::uint32_t start = 0, stop = 0;
    SegKind kind = SegKind::text;
};

struct Keyframe {
    double resolved_frame_index = 0.0;
    std::uint32_t video_t = 0;  // latent frames of the keyframe's video latent (0: none)
    std::uint32_t audio_t = 0;  // audio latent frames (0: none)
};
struct RefBlock {
    enum Kind { image, audio, video, video_audio } kind = image;
    std::uint32_t latent_t = 0, latent_h = 0, latent_w = 0, ref_audio_t = 0;
};

namespace detail {
inline std::vector<double> axis_from_sqrt_area(std::uint32_t dim, std::uint32_t patch, double sqrt_area) {
    const double ratio = static_cast<double>(dim) / sqrt_area;
    const std::uint32_t n = dim / patch;
    std::vector<double> a(n);
    for (std::uint32_t i = 0; i < n; ++i) a[i] = (static_cast<double>(i) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
    return a;
}
struct FrameGrid {
    std::vector<double> hw;  // [rows][2]
    std::vector<double> w_axis;
    std::uint32_t rows() const { return static_cast<std::uint32_t>(hw.size() / 2); }
};
inline FrameGrid frame_grid(std::uint32_t h, std::uint32_t w) {
    const double area = std::sqrt(static_cast<double>(h) * static_cast<double>(w));
    FrameGrid g;
    const auto hh = axis_from_sqrt_area(h, 2, area);
    g.w_axis = axis_from_sqrt_area(w, 2, area);
    for (double a : hh)
        for (double b : g.w_axis) {
            g.hw.push_back(a);
            g.hw.push_back(b);
        }
    return g;
}
inline double video_t_span(std::uint32_t k) { return kFrameRescale * kFramePerToken[k % 5]; }
inline std::vector<double> video_t_grid(std::uint32_t n, double origin) {
    std::vector<double> g(n);
    double cum = 0.0;  // sequential cumsum (f64)
    for (std::uint32_t k = 0; k < n; ++k) {
        g[k] = origin + (k == 0 ? 0.0 : cum);
        cum += video_t_span(k);
    }
    // reference: origin + cat([0], spans[:-1].cumsum()) -> element k = origin + sum_{j<k} spans[j]
    return g;
}
inline double sum_video_spans(std::uint32_t n) {
    double s = 0.0;  // python sum() over a list: sequential
    for (std::uint32_t k = 0; k < n; ++k) s += video_t_span(k);
    return s;
}
inline double ref_t_span(const RefBlock& b) {
    switch (b.kind) {
        case RefBlock::image: return 1.0;
        case RefBlock::audio: return static_cast<double>(b.ref_audio_t);
        default: return std::max(static_cast<double>(b.ref_audio_t), sum_video_spans(b.latent_t));
    }
}
inline void push_audio_grid(std::vector<double>& pos, double cursor, std::uint32_t t, double wl, double wh) {
    for (std::uint32_t c = 0; c < 2; ++c)
        for (std::uint32_t i = 0; i < t; ++i) {
            pos.push_back(cursor + static_cast<double>(i));
            pos.push_back(0.0);
            pos.push_back(c == 0 ? wl : wh);
        }
}
inline void push_video_grid(std::vector<double>& pos, std::uint32_t vt, const FrameGrid& f, double cursor) {
    const auto tg = video_t_grid(vt, cursor);
    for (std::uint32_t t = 0; t < vt; ++t)
        for (std::uint32_t r = 0; r < f.rows(); ++r) {
            pos.push_back(tg[t]);
            pos.push_back(f.hw[2 * r]);
            pos.push_back(f.hw[2 * r + 1]);
        }
}
}  // namespace detail

struct PackedLayout {
    std::uint32_t text_len = 0, latent_t = 0, latent_h = 0, latent_w = 0, audio_t = 0;
    std::uint32_t seq_len = 0;
    std::vector<double> position_ids;  // [S][3] (t, h, w)
    std::vector<Segment> segments;
    std::vector<std::int64_t> img_pos, audio_pos;  // packed rows of video-like / audio-like rows
    std::vector<std::uint8_t> img_update, audio_update;

    const Segment& find(SegKind k) const {
        for (const auto& s : segments)
            if (s.kind == k) return s;
        throw std::runtime_error("layout: no segment of that kind");
    }

    static PackedLayout build(std::uint32_t text_len, std::uint32_t latent_t, std::uint32_t latent_h, std::uint32_t latent_w,
        std::uint32_t audio_t, const std::vector<Keyframe>& keyframes = {}, const std::vector<RefBlock>& refs = {}) {
        using namespace detail;
        PackedLayout L;
        L.text_len = text_len;
        L.latent_t = latent_t;
        L.latent_h = latent_h;
        L.latent_w = latent_w;
        L.audio_t = audio_t;
        const FrameGrid frame = frame_grid(latent_h, latent_w);
        std::vector<std::pair<SegKind, std::uint32_t>> segs = {{SegKind::text, text_len}};
        for (std::uint32_t i = 0; i < text_len; ++i) {
            L.position_ids.push_back(static_cast<double>(i));
            L.position_ids.push_back(0.0);
            L.position_ids.push_back(0.0);
        }
        std::uint32_t row = text_len;
        const double wl = frame.w_axis.front(), wh = frame.w_axis.back();
        double cursor = static_cast<double>(text_len);
        for (const auto& b : refs) cursor += ref_t_span(b);
        auto add_img = [&](std::uint32_t n, bool upd) {
            for (std::uint32_t i = 0; i < n; ++i) {
                L.img_pos.push_back(row + i);
                L.img_update.push_back(upd ? 1 : 0);
            }
        };
        auto add_aud = [&](std::uint32_t n, bool upd) {
            for (std::uint32_t i = 0; i < n; ++i) {
                L.audio_pos.push_back(row + i);
                L.audio_update.push_back(upd ? 1 : 0);
            }
        };
        for (const auto& kf : keyframes) {
            const double cond_t = cursor + kFrameRescale * kf.resolved_frame_index;
            if (kf.video_t) {
                const std::uint32_t n = kf.video_t * frame.rows();
                segs.push_back({SegKind::cond, n});
                push_video_grid(L.position_ids, kf.video_t, frame, cond_t);
                add_img(n, false);
                row += n;
            }
            if (kf.audio_t) {
                segs.push_back({SegKind::cond_audio, kf.audio_t * 2});
                push_audio_grid(L.position_ids, cond_t, kf.audio_t, wl, wh);
                add_aud(kf.audio_t * 2, false);
                row += kf.audio_t * 2;
            }
        }
        if (!refs.empty()) {
            double cur = static_cast<double>(text_len);
            for (const auto& b : refs) {
                if (b.kind == RefBlock::image) {
                    const FrameGrid rf = frame_grid(b.latent_h, b.latent_w);
                    const std::uint32_t n = rf.rows();
                    segs.push_back({SegKind::ref_img, n});
                    for (std::uint32_t r = 0; r < n; ++r) {
                        L.position_ids.push_back(cur);
                        L.position_ids.push_back(rf.hw[2 * r]);
                        L.position_ids.push_back(rf.hw[2 * r + 1]);
                    }
                    add_img(n, false);
                    row += n;
                    cur += 1.0;
                } else if (b.kind == RefBlock::audio) {
                    if (b.ref_audio_t > 0) {
                        segs.push_back({SegKind::ref_audio, b.ref_audio_t * 2});
                        push_audio_grid(L.position_ids, cur, b.ref_audio_t, wl, wh);
                        add_aud(b.ref_audio_t * 2, false);
                        row += b.ref_audio_t * 2;
                    }
                    cur += static_cast<double>(b.ref_audio_t);
                } else {
                    const FrameGrid rf = frame_grid(b.latent_h, b.latent_w);
                    if (b.ref_audio_t > 0) {
                        segs.push_back({SegKind::ref_audio, b.ref_audio_t * 2});
                        push_audio_grid(L.position_ids, cur, b.ref_audio_t, rf.w_axis.front(), rf.w_axis.back());
                        add_aud(b.ref_audio_t * 2, false);
                        row += b.ref_audio_t * 2;
                    }
                    const std::uint32_t n = b.latent_t * rf.rows();
                    segs.push_back({SegKind::ref_img, n});
                    push_video_grid(L.position_ids, b.latent_t, rf, cur);
                    add_img(n, false);
                    row += n;
                    cur += std::max(static_cast<double>(b.ref_audio_t), sum_video_spans(b.latent_t));
                }
            }
        }
        segs.push_back({SegKind::audio, audio_t * 2});
        push_audio_grid(L.position_ids, cursor, audio_t, wl, wh);
        add_aud(audio_t * 2, true);
        row += audio_t * 2;
        const std::uint32_t nv = latent_t * frame.rows();
        segs.push_back({SegKind::video, nv});
        push_video_grid(L.position_ids, latent_t, frame, cursor);
        add_img(nv, true);
        row += nv;
        L.seq_len = row;
        std::uint32_t off = 0;
        for (const auto& [k, n] : segs) {
            L.segments.push_back({off, off + n, k});
            off += n;
        }
        if (L.position_ids.size() != static_cast<std::size_t>(row) * 3) throw std::runtime_error("layout: position size");
        return L;
    }
};

// A contiguous run of rows sharing one adaLN mod row (t_index * 3 + tag).
struct ModSegment {
    std::uint32_t start = 0, stop = 0, row = 0;
};

struct ForwardPlanConfig {
    float shift_video = 12.0F, shift_audio = 3.0F;
    double visual_cond_noise_aug = kVisualCondT, audio_cond_noise_aug = kAudioCondT;
};

// Per-forward timestep plan (no denoise masks).
struct ForwardPlan {
    float sigma_v = 0.0F, sigma_a = 0.0F;  // fp32 as the model computes them
    double t_v = 0.0, t_a = 0.0;
    std::vector<double> unique_t;  // sorted
    std::vector<float> t_vals;     // fp32 of unique_t (curve lookup input)
    std::vector<ModSegment> mod_segments;
    std::uint32_t video_t_row = 0, audio_t_row = 0;  // final-layer rows (t index)
    std::uint32_t n_t() const { return static_cast<std::uint32_t>(unique_t.size()); }
    std::uint32_t n_mod_rows() const { return n_t() * 3; }

    // sigma = the sampler's video sigma (fp32); text_tags: per text row tag (empty: all text = 1).
    static ForwardPlan build(const PackedLayout& L, float sigma, const std::vector<std::int64_t>& text_tags,
        const ForwardPlanConfig& cfg = {}) {
        ForwardPlan p;
        p.sigma_v = model_sigma(sigma);
        p.sigma_a = time_shift_sigma_f32(p.sigma_v, cfg.shift_video, cfg.shift_audio);
        p.t_v = static_cast<double>(1.0F - p.sigma_v);
        p.t_a = static_cast<double>(1.0F - p.sigma_a);
        auto seg_t = [&](SegKind k) -> double {
            switch (k) {
                case SegKind::text: case SegKind::video: return p.t_v;
                case SegKind::audio: return p.t_a;
                case SegKind::cond: case SegKind::ref_img: return std::max(p.t_v, cfg.visual_cond_noise_aug);
                default: return std::max(p.t_a, cfg.audio_cond_noise_aug);
            }
        };
        std::vector<double> ts = {p.t_v, p.t_a};
        for (const auto& s : L.segments) ts.push_back(seg_t(s.kind));
        std::sort(ts.begin(), ts.end());
        ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
        p.unique_t = ts;
        for (double t : ts) p.t_vals.push_back(static_cast<float>(t));
        auto t_row = [&](double t) {
            return static_cast<std::uint32_t>(std::lower_bound(ts.begin(), ts.end(), t) - ts.begin());
        };
        for (const auto& s : L.segments) {
            const std::uint32_t base = t_row(seg_t(s.kind)) * 3;
            if (s.kind == SegKind::text && !text_tags.empty()) {
                if (text_tags.size() != s.stop - s.start) throw std::runtime_error("plan: text tag count");
                std::uint32_t run = 0;
                const std::uint32_t n = s.stop - s.start;
                for (std::uint32_t i = 1; i <= n; ++i)
                    if (i == n || text_tags[i] != text_tags[run]) {
                        p.mod_segments.push_back({s.start + run, s.start + i, base + static_cast<std::uint32_t>(text_tags[run])});
                        run = i;
                    }
            } else {
                p.mod_segments.push_back({s.start, s.stop, base + seg_tag(s.kind)});
            }
        }
        p.video_t_row = t_row(seg_t(SegKind::video));
        p.audio_t_row = t_row(seg_t(SegKind::audio));
        return p;
    }
};

}  // namespace strix::h3::block
