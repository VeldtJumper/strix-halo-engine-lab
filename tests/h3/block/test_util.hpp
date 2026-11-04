#pragma once
// Shared helpers of the H3 block tests: fixture access, comparisons, report lines.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "h3/block/ops_ref.hpp"
#include "h3/common/fixture.hpp"

namespace h3t {

namespace fx = strix::h3::fixture;
using strix::h3::block::bf;

inline int g_failures = 0;
inline void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    if (!ok) ++g_failures;
}

inline std::string env_or(const char* name, const char* dflt) {
    const char* e = std::getenv(name);
    return e && *e ? std::string(e) : std::string(dflt);
}
// Assets come only from the environment (no built-in location): H3_FIXTURES = the fixture root (tools/h3/fixtures/FORMAT.md),
// H3_CKPT = the DiT checkpoint. Modes that need them report SKIPPED (exit kSkipped) when they are absent.
constexpr int kSkipped = 77;
inline std::string fixtures_root() {
    const std::string r = env_or("H3_FIXTURES", "");
    if (r.empty()) throw std::runtime_error("H3_FIXTURES is not set");
    return r;
}
inline std::string ckpt_path() {
    const std::string r = env_or("H3_CKPT", "");
    if (r.empty()) throw std::runtime_error("H3_CKPT is not set");
    return r;
}
inline bool assets_present(bool need_ckpt) {
    const std::string f = env_or("H3_FIXTURES", ""), c = env_or("H3_CKPT", "");
    if (f.empty() || (need_ckpt && c.empty())) return false;
    FILE* fp = std::fopen((f + "/FORMAT.md").c_str(), "rb");
    bool ok = fp != nullptr;
    if (fp) std::fclose(fp);
    if (ok && need_ckpt) {
        fp = std::fopen(c.c_str(), "rb");
        ok = fp != nullptr;
        if (fp) std::fclose(fp);
    }
    return ok;
}

template <class T> std::vector<T> as_vec(const fx::Tensor& t) {
    std::vector<T> v(t.bytes.size() / sizeof(T));
    std::memcpy(v.data(), t.bytes.data(), v.size() * sizeof(T));
    return v;
}
inline std::vector<std::uint16_t> bf16v(const fx::Manifest& m, const std::string& n) { return as_vec<std::uint16_t>(m.read(n)); }
inline std::vector<float> f32v(const fx::Manifest& m, const std::string& n) { return as_vec<float>(m.read(n)); }

struct Cmp {
    std::size_t n = 0, mism = 0;
    double max_abs = 0, rel_l2 = 0, cosine = 0, ref_max = 0;
    std::string str() const {
        char b[256];
        std::snprintf(b, sizeof b, "n=%zu mism=%zu max_abs=%.4g (ref max %.4g) rel_l2=%.3e cos=%.8f", n, mism, max_abs, ref_max,
            rel_l2, cosine);
        return b;
    }
};
inline Cmp compare_f(const std::vector<double>& got, const std::vector<double>& ref, const std::vector<bool>* differ = nullptr) {
    Cmp c;
    c.n = got.size();
    double se = 0, sr = 0, sg = 0, dot = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = got[i] - ref[i];
        c.max_abs = std::max(c.max_abs, std::fabs(d));
        c.ref_max = std::max(c.ref_max, std::fabs(ref[i]));
        se += d * d;
        sr += ref[i] * ref[i];
        sg += got[i] * got[i];
        dot += got[i] * ref[i];
        if (differ ? (*differ)[i] : got[i] != ref[i]) ++c.mism;
    }
    c.rel_l2 = sr > 0 ? std::sqrt(se / sr) : std::sqrt(se);
    c.cosine = (sr > 0 && sg > 0) ? dot / std::sqrt(sr * sg) : 1.0;
    return c;
}
inline Cmp compare_bf16(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& ref, std::size_t off = 0,
    std::size_t len = SIZE_MAX) {
    if (len == SIZE_MAX) len = ref.size() - off;
    std::vector<double> g(len), r(len);
    std::vector<bool> d(len);
    for (std::size_t i = 0; i < len; ++i) {
        g[i] = bf(got[off + i]);
        r[i] = bf(ref[off + i]);
        d[i] = got[off + i] != ref[off + i];
    }
    return compare_f(g, r, &d);
}
inline Cmp compare_f32(const std::vector<float>& got, const std::vector<float>& ref) {
    std::vector<double> g(got.begin(), got.end()), r(ref.begin(), ref.end());
    return compare_f(g, r);
}

}  // namespace h3t
