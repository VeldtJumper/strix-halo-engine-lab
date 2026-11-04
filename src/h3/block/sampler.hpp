#pragma once
// MiniMax H3 sampler (host): flow sigma schedules, ModelSamplingAV packing (audio carried at audio_scale =
// shift_video / shift_audio), flow Euler and res_multistep (eta 0, second order from step 1) in the reference's fp32
// operation order (checked bitwise on the dumped sampler steps), and the final unscaling.
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace strix::h3::block {

// ModelSamplingDiscreteFlow table: sigma_i = shift * t / (1 + (shift - 1) * t), t = fp32(i) / 1000, i = 1..1000 (fp32).
inline std::vector<float> flow_sigma_table(float shift, std::uint32_t timesteps = 1000, float multiplier = 1000.0F) {
    std::vector<float> s(timesteps);
    for (std::uint32_t i = 1; i <= timesteps; ++i) {
        const float ts = (static_cast<float>(i) / static_cast<float>(timesteps)) * multiplier;
        const float t = ts / multiplier;
        s[i - 1] = shift == 1.0F ? t : shift * t / (1.0F + (shift - 1.0F) * t);
    }
    return s;
}
// "simple" scheduler: steps sigmas from the top of the table, then 0.
inline std::vector<float> simple_sigmas(float shift, std::uint32_t steps) {
    const std::vector<float> tab = flow_sigma_table(shift);
    const double ss = static_cast<double>(tab.size()) / steps;
    std::vector<float> o;
    for (std::uint32_t x = 0; x < steps; ++x) o.push_back(tab[tab.size() - 1 - static_cast<std::size_t>(x * ss)]);
    o.push_back(0.0F);
    return o;
}
// Uniform base grid t_i = 1 - i / n shifted: sigma_i = shift t / (1 + (shift - 1) t) (the PDD grid, e.g. 12t/(1+11t)),
// evaluated in fp64 and rounded to fp32; last entry 0.
inline std::vector<float> shifted_linear_sigmas(float shift, std::uint32_t steps) {
    std::vector<float> o;
    for (std::uint32_t i = 0; i <= steps; ++i) {
        const double t = 1.0 - static_cast<double>(i) / steps;
        o.push_back(static_cast<float>(shift * t / (1.0 + (shift - 1.0) * t)));
    }
    o.back() = 0.0F;
    return o;
}

// Flat sampler state: [video | audio(carried)] fp32.
struct AvState {
    std::vector<float> video, audio;
};

// Initial state: x = sigma0 * noise + (1 - sigma0) * latent (latent already carried; empty latent = zeros).
inline AvState initial_state(float sigma0, const std::vector<float>& noise_v, const std::vector<float>& noise_a,
    const std::vector<float>* latent_v = nullptr, const std::vector<float>* latent_a = nullptr) {
    AvState s{noise_v, noise_a};
    auto f = [&](std::vector<float>& x, const std::vector<float>* lat) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            const float l = lat ? (*lat)[i] : 0.0F;
            x[i] = sigma0 * (1.0F * x[i]) + (1.0F - sigma0) * l;
        }
    };
    f(s.video, latent_v);
    f(s.audio, latent_a);
    return s;
}

// The model as the sampler sees it: velocity (bf16 values, carried audio) of a bf16 input at sigma. The caller casts
// x to bf16 (model input), runs the DiT and returns its outputs widened to fp32.
using AvModelFn = std::function<AvState(const AvState& x, float sigma)>;

// denoised = x - out * sigma (CONST flow), fp32.
inline AvState denoised_from(const AvState& x, const AvState& out, float sigma) {
    AvState d = x;
    for (std::size_t i = 0; i < d.video.size(); ++i) d.video[i] = x.video[i] - out.video[i] * sigma;
    for (std::size_t i = 0; i < d.audio.size(); ++i) d.audio[i] = x.audio[i] - out.audio[i] * sigma;
    return d;
}

struct SamplerTrace {
    std::vector<AvState> x, denoised;  // per step: state entering the step and its x0 prediction
};

enum class SamplerKind { euler, res_multistep };

// Resumable sampler state: the state entering step `next` plus the multistep history.
struct SamplerCarry {
    AvState x;
    std::size_t next = 0;  // index of the next step (0 .. sigmas.size() - 1)
    bool have_old = false;
    AvState old;           // previous step's x0 prediction
    float old_sigma_down = 0.0F;
};

// Runs steps carry.next .. stop - 1 (stop <= sigmas.size() - 1; eta 0, no noise is drawn). Running the steps in one call
// or in several (with the carry kept) gives the same bits.
inline void sample_steps(const AvModelFn& model, SamplerCarry& c, const std::vector<float>& sigmas, SamplerKind kind, std::size_t stop,
    SamplerTrace* trace = nullptr) {
    auto each = [](AvState& a, const std::function<float(float, std::size_t, bool)>& f) {
        for (std::size_t i = 0; i < a.video.size(); ++i) a.video[i] = f(a.video[i], i, true);
        for (std::size_t i = 0; i < a.audio.size(); ++i) a.audio[i] = f(a.audio[i], i, false);
    };
    AvState& x = c.x;
    for (std::size_t i = c.next; i < stop && i + 1 < sigmas.size(); ++i) {
        const float s = sigmas[i], sn = sigmas[i + 1];
        const AvState out = model(x, s);
        const AvState den = denoised_from(x, out, s);
        if (trace) {
            trace->x.push_back(x);
            trace->denoised.push_back(den);
        }
        const float sigma_down = sn;  // eta 0: sigma_up = 0
        if (kind == SamplerKind::euler || sigma_down == 0.0F || !c.have_old) {
            const float dt = sigma_down - s;
            each(x, [&](float v, std::size_t j, bool vid) {
                const float dd = (v - (vid ? den.video[j] : den.audio[j])) / s;
                return v + dd * dt;
            });
        } else {
            auto t_fn = [](float sg) { return -std::log(sg); };
            const float t = t_fn(s), t_old = t_fn(c.old_sigma_down), t_next = t_fn(sigma_down), t_prev = t_fn(sigmas[i - 1]);
            const float h = t_next - t;
            const float c2 = (t_prev - t_old) / h;
            auto phi1 = [](float v) { return std::expm1(v) / v; };
            auto phi2 = [&](float v) { return (phi1(v) - 1.0F) / v; };
            const float p1 = phi1(-h), p2 = phi2(-h);
            auto nan0 = [](float v) {
                if (std::isnan(v)) return 0.0F;
                if (std::isinf(v)) return v > 0 ? 3.4028234663852886e38F : -3.4028234663852886e38F;
                return v;
            };
            const float b1 = nan0(p1 - p2 / c2), b2 = nan0(p2 / c2);
            const float eh = std::exp(-h);
            const AvState& old = c.old;
            each(x, [&](float v, std::size_t j, bool vid) {
                const float d = vid ? den.video[j] : den.audio[j];
                const float o = vid ? old.video[j] : old.audio[j];
                return eh * v + h * (b1 * d + b2 * o);
            });
        }
        c.old = den;
        c.have_old = true;
        c.old_sigma_down = sigma_down;
        c.next = i + 1;
    }
}

// Runs sigmas.size() - 1 steps (eta 0, no noise is drawn).
inline AvState sample(const AvModelFn& model, AvState x, const std::vector<float>& sigmas, SamplerKind kind,
    SamplerTrace* trace = nullptr) {
    SamplerCarry c;
    c.x = std::move(x);
    sample_steps(model, c, sigmas, kind, sigmas.empty() ? 0 : sigmas.size() - 1, trace);
    return std::move(c.x);
}

// Output latent: x / (1 - sigma_last) (inverse noise scaling), then the audio carry undone (* fp32(1 / audio_scale)).
inline AvState finalize(const AvState& x, float sigma_last, float audio_scale) {
    AvState o = x;
    const float inv = static_cast<float>(1.0 / static_cast<double>(audio_scale));
    for (auto& v : o.video) v = v / (1.0F - sigma_last);
    for (auto& v : o.audio) v = (v / (1.0F - sigma_last)) * inv;
    return o;
}

}  // namespace strix::h3::block
