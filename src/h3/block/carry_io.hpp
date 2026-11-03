#pragma once
// Sampler checkpoint (dit_state.bin) of the clip runner: the SamplerCarry after a finished step, serialized and loaded
// fail-closed. Layout (little endian): u64 magic, u64 next, u64 have_old, u64 |video|, u64 |audio|, f32 old_sigma_down,
// then f32 x.video, x.audio and (have_old) old.video, old.audio.
// Invariant of every carry written after a step (both samplers: sample_steps records the history every step):
// 1 <= next <= steps, have_old == 1, old_sigma_down == sigmas[next] (bit for bit), every value finite.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/block/sampler.hpp"

namespace strix::h3::block {

inline constexpr std::uint64_t kCarryMagic = 0x48334341525259ULL;

inline void save_carry(const std::filesystem::path& p, const SamplerCarry& c) {
    const std::filesystem::path tmp = p.string() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        const std::uint64_t hdr[5] = {kCarryMagic, c.next, c.have_old ? 1ULL : 0ULL, c.x.video.size(), c.x.audio.size()};
        f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
        f.write(reinterpret_cast<const char*>(&c.old_sigma_down), 4);
        auto w = [&](const std::vector<float>& v, std::size_t n) {
            if (v.size() != n) throw std::runtime_error("carry: state size");
            f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * 4));
        };
        w(c.x.video, hdr[3]);
        w(c.x.audio, hdr[4]);
        if (c.have_old) {
            w(c.old.video, hdr[3]);
            w(c.old.audio, hdr[4]);
        }
        f.flush();
        if (!f) throw std::runtime_error("carry: write " + tmp.string());
    }
    std::filesystem::rename(tmp, p);
}

// false: no checkpoint file. Throws on anything that is not a carry of THIS run (sizes, step range, history, sigma, values).
inline bool load_carry(const std::filesystem::path& p, SamplerCarry& c, std::size_t nv, std::size_t na, const std::vector<float>& sigmas) {
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) return false;
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("dit_state.bin unreadable");
    const std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 44) throw std::runtime_error("dit_state.bin malformed (short header, " + std::to_string(b.size()) + " bytes)");
    std::uint64_t hdr[5];
    std::memcpy(hdr, b.data(), sizeof hdr);
    const std::size_t nsteps = sigmas.empty() ? 0 : sigmas.size() - 1;
    if (hdr[0] != kCarryMagic) throw std::runtime_error("dit_state.bin: bad magic");
    if (hdr[3] != nv || hdr[4] != na) throw std::runtime_error("dit_state.bin does not match this run (latent sizes)");
    if (hdr[1] < 1 || hdr[1] > nsteps) throw std::runtime_error("dit_state.bin: step index " + std::to_string(hdr[1]) + " outside 1.." + std::to_string(nsteps));
    if (hdr[2] != 1) throw std::runtime_error("dit_state.bin: missing / bad history flag (every step records the history)");
    const std::size_t want = 44 + 2 * (nv + na) * 4;
    if (b.size() != want) throw std::runtime_error("dit_state.bin truncated / oversized");
    SamplerCarry r;
    r.next = hdr[1];
    r.have_old = true;
    std::memcpy(&r.old_sigma_down, b.data() + 40, 4);
    std::uint32_t got = 0, exp = 0;
    std::memcpy(&got, &r.old_sigma_down, 4);
    std::memcpy(&exp, &sigmas[r.next], 4);
    if (got != exp) throw std::runtime_error("dit_state.bin: history sigma does not match the schedule at step " + std::to_string(r.next));
    const char* q = b.data() + 44;
    auto rd = [&](std::vector<float>& v, std::size_t n, const char* what) {
        v.resize(n);
        std::memcpy(v.data(), q, n * 4);
        q += n * 4;
        for (float x : v)
            if (!std::isfinite(x)) throw std::runtime_error(std::string("dit_state.bin: non-finite ") + what);
    };
    rd(r.x.video, nv, "state");
    rd(r.x.audio, na, "state");
    rd(r.old.video, nv, "history");
    rd(r.old.audio, na, "history");
    c = std::move(r);
    return true;
}

}  // namespace strix::h3::block
