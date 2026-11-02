#include "h3/te/te_sha256.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(__HIP_DEVICE_COMPILE__)
#define STRIX_H3_TE_SHA_NI 1
#include <immintrin.h>
#endif

namespace strix::h3::te {
namespace {

constexpr std::uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

#ifdef STRIX_H3_TE_SHA_NI
// x86 SHA extensions (run-time detected): 4 rounds per sha256rnds2 pair, message schedule by sha256msg1/2.
__attribute__((target("sha,sse4.1,ssse3"))) void blocks_ni(std::uint32_t state[8], const std::uint8_t* data,
                                                          std::size_t n) {
    const __m128i mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
    __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
    tmp = _mm_shuffle_epi32(tmp, 0xB1);          // CDAB
    s1 = _mm_shuffle_epi32(s1, 0x1B);            // EFGH
    __m128i s0 = _mm_alignr_epi8(tmp, s1, 8);    // ABEF
    s1 = _mm_blend_epi16(s1, tmp, 0xF0);         // CDGH
    for (; n; --n, data += 64) {
        const __m128i abef = s0, cdgh = s1;
        __m128i w[4];
        for (int g = 0; g < 16; ++g) {
            __m128i m;
            if (g < 4) {
                m = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * g)), mask);
            } else {  // W[g] = msg2(msg1(W[g-4], W[g-3]) + (W[g-2]:W[g-1] >> 32), W[g-1])
                const __m128i x = _mm_add_epi32(_mm_sha256msg1_epu32(w[g & 3], w[(g + 1) & 3]),
                                                _mm_alignr_epi8(w[(g + 3) & 3], w[(g + 2) & 3], 4));
                m = _mm_sha256msg2_epu32(x, w[(g + 3) & 3]);
            }
            w[g & 3] = m;
            __m128i t = _mm_add_epi32(m, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kK[4 * g])));
            s1 = _mm_sha256rnds2_epu32(s1, s0, t);
            t = _mm_shuffle_epi32(t, 0x0E);
            s0 = _mm_sha256rnds2_epu32(s0, s1, t);
        }
        s0 = _mm_add_epi32(s0, abef);
        s1 = _mm_add_epi32(s1, cdgh);
    }
    tmp = _mm_shuffle_epi32(s0, 0x1B);           // FEBA
    s1 = _mm_shuffle_epi32(s1, 0xB1);            // DCHG
    s0 = _mm_blend_epi16(tmp, s1, 0xF0);         // DCBA
    s1 = _mm_alignr_epi8(s1, tmp, 8);            // HGFE
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), s0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), s1);
}

bool cpu_has_sha_ni() {
    static const bool has = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("sha") && __builtin_cpu_supports("sse4.1");
    }();
    return has;
}
#endif

}  // namespace

bool sha256_uses_sha_ni() {
#ifdef STRIX_H3_TE_SHA_NI
    return cpu_has_sha_ni();
#else
    return false;
#endif
}

Sha256::Sha256(bool portable) : ni_(!portable && sha256_uses_sha_ni()) {
    static constexpr std::uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h_, init, sizeof h_);
}

void Sha256::blocks(const std::uint8_t* p, std::size_t n) {
#ifdef STRIX_H3_TE_SHA_NI
    if (ni_) {
        blocks_ni(h_, p, n);
        return;
    }
#endif
    for (; n; --n, p += 64) block(p);
}

void Sha256::block(const std::uint8_t* p) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = std::uint32_t(p[4 * i]) << 24 | std::uint32_t(p[4 * i + 1]) << 16 | std::uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
        const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

void Sha256::update(const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_ += n;
    if (fill_) {
        const std::size_t t = std::min<std::size_t>(64 - fill_, n);
        std::memcpy(buf_ + fill_, p, t);
        fill_ += t; p += t; n -= t;
        if (fill_ < 64) return;
        blocks(buf_, 1);
        fill_ = 0;
    }
    const std::size_t nb = n / 64;
    if (nb) blocks(p, nb);
    p += nb * 64;
    n -= nb * 64;
    std::memcpy(buf_, p, n);
    fill_ = n;
}

std::array<std::uint8_t, 32> Sha256::finish() {
    const std::uint64_t bits = total_ * 8;
    const std::uint8_t pad = 0x80;
    update(&pad, 1);
    const std::uint8_t zero[64] = {};
    update(zero, (fill_ <= 56 ? 56 - fill_ : 120 - fill_));
    std::uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = std::uint8_t(bits >> (56 - 8 * i));
    update(len, 8);
    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) out[4 * i + j] = std::uint8_t(h_[i] >> (24 - 8 * j));
    return out;
}

std::string Sha256::finish_hex() {
    static const char* hx = "0123456789abcdef";
    const auto d = finish();
    std::string s(64, '0');
    for (int i = 0; i < 32; ++i) { s[2 * i] = hx[d[i] >> 4]; s[2 * i + 1] = hx[d[i] & 15]; }
    return s;
}

std::string sha256_hex(const void* data, std::size_t n) {
    Sha256 h;
    h.update(data, n);
    return h.finish_hex();
}

std::string sha256_file_hex(const std::string& path) {
    struct Closer { void operator()(std::FILE* f) const { if (f) std::fclose(f); } };
    std::unique_ptr<std::FILE, Closer> f(std::fopen(path.c_str(), "rb"));
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<unsigned char> buf(std::size_t(8) << 20);
    Sha256 h;
    for (;;) {
        const std::size_t n = std::fread(buf.data(), 1, buf.size(), f.get());
        if (n) h.update(buf.data(), n);
        if (n < buf.size()) {
            if (std::ferror(f.get())) throw std::runtime_error("read error in " + path);
            break;
        }
    }
    return h.finish_hex();
}

}  // namespace strix::h3::te
