// Byte gate of the long-span decode attention with shared key/value chunks
// (STRIX_G4_ATTN_KV_SHARED=1 and 2) against the chunk-fastest launches: outputs, a8
// codes/metadata and failure words for sliding and global layers, 1..8 rows,
// spans up to 64K positions, rows of one stream and of two streams. With
// "bench" it times both schedules per context.
#include <hip/hip_runtime.h>
#include "gemma4/kv/storage.hpp"
#include "gemma4/kernels/a8.hpp"
#include "gemma4/kernels/internal.hpp"
#include "fixture.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <string_view>

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
// Finite mixed-magnitude values (both zero signs), cheap enough for 64K-row caches.
void fill(std::vector<float>& x, std::uint64_t seed) {
    for (auto& a : x) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const auto r = unsigned(seed >> 33U);
        a = r % 61U == 0 ? ((r & 1U) ? -0.0F : 0.0F) : std::ldexp(float(int(r % 2000001U) - 1000000), -21 - int(r >> 28U));
    }
}
template <class T> Buffer device(const std::vector<T>& values) {
    Buffer b(values.size() * sizeof(T));
    b.upload(values.data(), b.size());
    return b;
}
struct Cache {
    Buffer k, v, tags, ok, ov;
    std::uint64_t frontier = 0;
    unsigned width = 0, slots = 0;
    bool global = false;
};
// One stream's cache: committed positions [0, frontier) in the slots, the next
// eight positions in the overlay (the current pass's rows).
Cache cache(std::uint64_t frontier, bool global, std::uint64_t seed) {
    Cache c;
    c.frontier = frontier; c.global = global;
    const unsigned dim = global ? 512U : 256U;
    c.width = kv_heads * dim;
    c.slots = global ? unsigned(frontier) + 8U : window;
    std::vector<float> k(std::size_t(c.slots) * c.width), v(k.size()), ok(std::size_t(8) * c.width), ov(ok.size());
    fill(k, seed); fill(v, seed + 1); fill(ok, seed + 2); fill(ov, seed + 3);
    std::vector<std::uint64_t> tags(c.slots, UINT64_MAX);
    const std::uint64_t first = !global && frontier > window ? frontier - window : 0U;
    for (auto p = first; p < frontier; ++p) tags[global ? p : p % c.slots] = p;
    c.k = device(k); c.v = device(v); c.tags = device(tags); c.ok = device(ok); c.ov = device(ov);
    return c;
}
KvRead read(const Cache& c, unsigned row) {
    return {c.k.as<float>(), c.v.as<float>(), c.tags.as<std::uint64_t>(), c.ok.as<float>(), c.ov.as<float>(),
            c.frontier, c.frontier + row, c.width, c.slots, 8, window, c.global};
}
struct Case {
    std::uint64_t frontier;
    bool global;
    unsigned rows;
    bool two_streams, quant;
};
std::size_t cases = 0;
// FNV-1a over every case's outputs (and a8 codes/metadata): compared across processes run with different
// switches (e.g. STRIX_G4_ATTN_MERGE) to prove those launches byte-identical.
std::uint64_t digest = 1469598103934665603ULL;
void absorb(const Buffer& x) {
    std::vector<std::uint8_t> u(x.size());
    x.download(u.data(), u.size());
    for (const auto c : u) digest = (digest ^ c) * 1099511628211ULL;
}
double run(const Case& t, bool bench) {
    const unsigned dim = t.global ? 512U : 256U, hcount = heads, width = hcount * dim;
    const auto a = cache(t.frontier, t.global, 11 + t.frontier), b = t.two_streams ? cache(t.frontier + 3, t.global, 7) : Cache{};
    std::vector<KvRead> reads;
    std::vector<std::uint64_t> positions;
    for (unsigned r = 0; r < t.rows; ++r) {
        const bool second = t.two_streams && (r & 1U);
        reads.push_back(read(second ? b : a, r / (t.two_streams ? 2U : 1U)));
        positions.push_back(reads.back().position);
    }
    const unsigned stride = unsigned(t.frontier) + 3U + 8U;
    const unsigned chunks = (stride + attention_chunk - 1U) / attention_chunk;
    std::vector<float> q(std::size_t(t.rows) * width), norm(dim), table(std::size_t(stride) * dim);
    fill(q, 5); fill(table, 9);
    for (auto& w : norm) w = 1.0F + std::ldexp(float(int(&w - norm.data()) % 17 - 8), -6);
    auto dq = device(q), dnorm = device(norm), dtable = device(table), dreads = device(reads), dpos = device(positions);
    const LinearView q_norm{dnorm.as<std::uint8_t>(), 1, dim, dim * 4U, WeightFormat::f32};
    Buffer scores(std::size_t(t.rows) * hcount * stride * 4), maxima(std::size_t(t.rows) * hcount * (chunks + 1) * 4),
        partials(std::size_t(t.rows) * hcount * chunks * (dim + 1) * 4), counters(std::size_t(t.rows) * hcount * 4);
    Buffer out[3] = {Buffer(std::size_t(t.rows) * width * 4), Buffer(std::size_t(t.rows) * width * 4),
                     Buffer(std::size_t(t.rows) * width * 4)};
    Buffer codes[3] = {Buffer(std::size_t(t.rows) * width), Buffer(std::size_t(t.rows) * width), Buffer(std::size_t(t.rows) * width)};
    Buffer meta[3] = {Buffer(std::size_t(t.rows) * (width / 32U) * sizeof(a8::Meta)),
                      Buffer(std::size_t(t.rows) * (width / 32U) * sizeof(a8::Meta)),
                      Buffer(std::size_t(t.rows) * (width / 32U) * sizeof(a8::Meta))};
    Buffer failure(12);
    counters.zero(); failure.zero();
    double ms[3] = {};
    for (unsigned shared = 0; shared < 3; ++shared) {
        out[shared].zero(); codes[shared].zero(); meta[shared].zero();
        const A8Rows quant = t.quant ? A8Rows{codes[shared].as<std::int8_t>(), meta[shared].as<a8::Meta>(), width} : A8Rows{};
        const auto launch = [&] {
            CHECK(fused_attention(dq.as<float>(), q_norm, dpos.as<std::uint64_t>(), dtable.as<float>(), dreads.as<KvRead>(),
                                  t.rows, hcount, dim, stride, t.global, out[shared].as<float>(), failure.as<unsigned>() + shared,
                                  nullptr, true, false, false, scores.as<float>(), maxima.as<float>(), partials.as<float>(),
                                  counters.as<unsigned>(), quant, int(shared)));
        };
        launch();
        synchronize();
        if (bench) {
            const auto start = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < 5; ++i) launch();
            synchronize();
            ms[shared] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 5;
        }
    }
    unsigned flags[3] = {};
    failure.download(flags, 12);
    CHECK(flags[0] == flags[1] && flags[0] == flags[2]);
    const auto same = [](const Buffer& x, const Buffer& y) {
        std::vector<std::uint8_t> u(x.size()), w(y.size());
        x.download(u.data(), u.size()); y.download(w.data(), w.size());
        return u == w;
    };
    bool equal = true;
    for (unsigned m = 1; m < 3; ++m)
        equal = equal && same(out[0], out[m]) && (!t.quant || (same(codes[0], codes[m]) && same(meta[0], meta[m])));
    if (!equal) {
        std::cerr << "mismatch frontier " << t.frontier << " global " << t.global << " rows " << t.rows << " two " << t.two_streams
                  << " quant " << t.quant << '\n';
        throw std::runtime_error("shared key/value attention byte mismatch");
    }
    absorb(out[0]);
    // Failure words too: a switch that set an extra bit in every mode of one process would otherwise pass.
    for (const auto c : std::string_view(reinterpret_cast<const char*>(flags), 4)) digest = (digest ^ std::uint8_t(c)) * 1099511628211ULL;
    if (t.quant) { absorb(codes[0]); absorb(meta[0]); }
    ++cases;
    if (bench)
        std::cout << std::setprecision(4) << "{\"bench\":\"attention\",\"global\":" << t.global << ",\"context\":" << t.frontier
                  << ",\"rows\":" << t.rows << ",\"two_streams\":" << t.two_streams << ",\"chunk_fastest_ms\":" << ms[0]
                  << ",\"shared_ms\":" << ms[1] << ",\"group_ms\":" << ms[2] << "}\n";
    return ms[1];
}
} // namespace
int main(int argc, char** argv) {
    try {
        require_device();
        if (argc == 2 && std::string_view(argv[1]) == "bench") {
            for (const std::uint64_t context : {8192ULL, 32768ULL, 65528ULL})
                for (const bool global : {true, false})
                    for (const unsigned rows : {1U, 4U, 8U}) run({context, global, rows, rows == 4U, false}, true);
            return 0;
        }
        // Spans that admit the long-span launches (more than two chunks), chunk
        // edges, ragged rows, both streams layouts, a8 on and off.
        for (const std::uint64_t frontier : {190ULL, 191ULL, 192ULL, 500ULL, 513ULL, 1000ULL, 4095ULL, 8192ULL, 16383ULL, 32768ULL, 65528ULL})
            for (const bool global : {false, true})
                for (const unsigned rows : {1U, 3U, 8U}) {
                    if (frontier > 16383U && rows == 3U) continue;
                    run({frontier, global, rows, false, rows == 8U}, false);
                    if (rows > 1U) run({frontier, global, rows, true, rows == 3U}, false);
                }
        std::cout << "{\"gate\":\"attention-kv-shared-vs-chunk-fastest\",\"cases\":" << cases << ",\"digest\":\"" << std::hex << digest << std::dec
                  << "\",\"verdict\":\"EXACT\"}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
