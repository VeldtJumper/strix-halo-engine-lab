#include "h3/te/te_vision_host.hpp"

#include "h3/te/te_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strix::h3::te {
namespace {

constexpr int kPatch = 16, kMerge = 2, kTemporal = 2, kFactor = kPatch * kMerge;
constexpr double kMinPixels = 3136, kMaxPixels = 12845056;

struct Taps { std::vector<int> i0, i1; std::vector<float> w0, w1; };

// Source indices/weights of one axis: half-pixel centers, clamped at 0, scale = in / out (fp32).
Taps taps(int in, int out) {
    Taps t;
    const float scale = float(in) / float(out);
    for (int o = 0; o < out; ++o) {
        float src = scale * (float(o) + 0.5F) - 0.5F;
        if (src < 0.0F) src = 0.0F;
        const int i0 = static_cast<int>(src);
        const float l1 = std::min(std::max(src - float(i0), 0.0F), 1.0F);
        t.i0.push_back(i0);
        t.i1.push_back(i0 < in - 1 ? i0 + 1 : i0);
        t.w0.push_back(1.0F - l1);
        t.w1.push_back(l1);
    }
    return t;
}

}  // namespace

void smart_resize(int h, int w, int& h_bar, int& w_bar) {
    // Python round(): half to even on the double quotient
    h_bar = int(std::nearbyint(double(h) / kFactor)) * kFactor;
    w_bar = int(std::nearbyint(double(w) / kFactor)) * kFactor;
    if (double(h_bar) * w_bar > kMaxPixels) {
        const double beta = std::sqrt(double(h) * w / kMaxPixels);
        h_bar = std::max(kFactor, int(std::floor(h / beta / kFactor)) * kFactor);
        w_bar = std::max(kFactor, int(std::floor(w / beta / kFactor)) * kFactor);
    } else if (double(h_bar) * w_bar < kMinPixels) {
        const double beta = std::sqrt(kMinPixels / (double(h) * w));
        h_bar = int(std::ceil(h * beta / kFactor)) * kFactor;
        w_bar = int(std::ceil(w * beta / kFactor)) * kFactor;
    }
}

std::vector<float> resize_bilinear(const float* rgb, int h, int w, int oh, int ow) {
    const auto th = taps(h, oh), tw = taps(w, ow);
    std::vector<float> out(std::size_t(oh) * ow * 3);
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](int yy, int xx) { return rgb[(std::size_t(yy) * w + xx) * 3 + c]; };
                const float r0 = at(th.i0[y], tw.i0[x]) * tw.w0[x] + at(th.i0[y], tw.i1[x]) * tw.w1[x];
                const float r1 = at(th.i1[y], tw.i0[x]) * tw.w0[x] + at(th.i1[y], tw.i1[x]) * tw.w1[x];
                out[(std::size_t(y) * ow + x) * 3 + c] = r0 * th.w0[y] + r1 * th.w1[y];
            }
    return out;
}

Patches make_patches(const Frames& f) {
    if ((f.t != 1 && f.t != 2) || f.h <= 0 || f.w <= 0 || f.rgb.size() != std::size_t(f.t) * f.h * f.w * 3)
        throw std::invalid_argument("make_patches: frames");
    int hb = 0, wb = 0;
    smart_resize(f.h, f.w, hb, wb);
    std::vector<std::vector<float>> frames;
    for (int t = 0; t < f.t; ++t) {
        auto r = resize_bilinear(f.rgb.data() + std::size_t(t) * f.h * f.w * 3, f.h, f.w, hb, wb);
        for (auto& v : r) v = (v - 0.5F) / 0.5F;
        frames.push_back(std::move(r));
    }
    Patches p;
    p.grid_h = hb / kPatch;
    p.grid_w = wb / kPatch;
    constexpr int feat = 3 * kTemporal * kPatch * kPatch;
    p.data.resize(std::size_t(p.grid_h) * p.grid_w * feat);
    std::size_t row = 0;
    for (int bh = 0; bh < p.grid_h / kMerge; ++bh)
        for (int bw = 0; bw < p.grid_w / kMerge; ++bw)
            for (int mh = 0; mh < kMerge; ++mh)
                for (int mw = 0; mw < kMerge; ++mw, ++row) {
                    float* dst = p.data.data() + row * feat;
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < kTemporal; ++t) {
                            const auto& fr = frames[f.t == 1 ? 0 : t];
                            for (int py = 0; py < kPatch; ++py)
                                for (int px = 0; px < kPatch; ++px) {
                                    const int y = (bh * kMerge + mh) * kPatch + py, x = (bw * kMerge + mw) * kPatch + px;
                                    *dst++ = fr[(std::size_t(y) * wb + x) * 3 + c];
                                }
                        }
                }
    return p;
}

Layout expand_layout(const std::vector<std::int32_t>& presented, const std::vector<Patches>& blocks) {
    Layout l;
    std::size_t b = 0;
    for (const auto id : presented) {
        if (id != kVisionSlot) { l.ids.push_back(id); continue; }
        if (b >= blocks.size()) throw std::invalid_argument("expand_layout: more slots than blocks");
        const auto& p = blocks[b++];
        VisionSpan s;
        s.index = std::uint32_t(l.ids.size());
        s.size = std::uint32_t(p.merged_tokens());
        s.grid_h = p.grid_h;
        s.grid_w = p.grid_w;
        l.spans.push_back(s);
        l.ids.insert(l.ids.end(), s.size, kVisionSlot);
    }
    if (b != blocks.size()) throw std::invalid_argument("expand_layout: unused blocks");
    const std::size_t S = l.ids.size();
    l.tags.assign(S, 1);
    for (const auto& s : l.spans) {
        const std::size_t lo = s.index > 0 ? s.index - 1 : 0, hi = std::min<std::size_t>(S, s.index + s.size + 1);
        for (std::size_t i = lo; i < hi; ++i) l.tags[i] = 0;
    }
    if (l.spans.empty()) return l;
    // MRoPE positions, following the reference construction step by step (float32 values)
    l.positions.assign(3 * S, 0.0F);
    auto pos = [&](int axis, std::size_t i) -> float& { return l.positions[std::size_t(axis) * S + i]; };
    long long offset = 0;
    bool first = true;
    for (const auto& s : l.spans) {
        const std::size_t start = s.index, end = s.index + s.size;
        if (first) {
            for (std::size_t i = 0; i < start; ++i) for (int a = 0; a < 3; ++a) pos(a, i) = float(i);
            first = false;
        }
        const long long len_max = std::max({1, s.grid_h, s.grid_w}) / 2;
        const long long start_next = len_max + (long long)start;
        for (std::size_t i = end; i < S; ++i)
            for (int a = 0; a < 3; ++a) pos(a, i) = float(start_next + offset + (long long)(i - end));
        const long long mh = s.grid_h / 2, mw = s.grid_w / 2;
        const long long n_h = (s.size + mh - 1) / mh;  // ceil(size / max_d) repeats of each H value
        for (std::size_t j = 0; j < s.size; ++j) {
            pos(0, start + j) = float((long long)start + offset);
            pos(1, start + j) = float((long long)start + offset + (long long)j / n_h);
            pos(2, start + j) = float((long long)start + offset + (long long)j % mw);
        }
        offset += len_max - (long long)s.size;
    }
    return l;
}

}  // namespace strix::h3::te
