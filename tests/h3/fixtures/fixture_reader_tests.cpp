// CPU test for the h3fx fixture reader: SHA-256 vectors, JSON parsing, a self-written fixture
// set round trip, and (optional) a real fixture set passed as argv[1].
#include "h3/common/fixture.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fx = strix::h3::fixture;

static int failures = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

static std::string sha(const std::string& s) {
    return fx::sha256_hex(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
}

static void test_sha256() {
    CHECK(sha("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(sha(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

static void test_json() {
    auto j = fx::JsonParser(R"({"a": [1, -2.5e1, true, null, "xé\n"], "b": {"c": "d"}})").parse();
    CHECK(j.at("a").arr().size() == 5);
    CHECK(j.at("a").arr()[1].num() == -25.0);
    CHECK(j.at("a").arr()[4].str() == "x\xc3\xa9\n");
    CHECK(j.at("b").at("c").str() == "d");
}

static void test_conversions() {
    CHECK(fx::bf16_to_f32(0x3F80) == 1.0F);
    CHECK(fx::bf16_to_f32(0xC040) == -3.0F);
    CHECK(fx::f16_to_f32(0x3C00) == 1.0F);
    CHECK(fx::f16_to_f32(0x0001) == std::ldexp(1.0F, -24));
    CHECK(fx::f16_to_f32(0xFBFF) == -65504.0F);
}

static void test_roundtrip() {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "h3fx_reader_test";
    fs::create_directories(dir / "sub");
    const std::uint16_t vals[6] = {0x3F80, 0x4000, 0x4040, 0xBF80, 0x0000, 0x7F80};  // bf16 1 2 3 -1 0 inf
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(vals);
    {
        std::ofstream f(dir / "sub" / "t.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }
    std::string man = std::string(R"({"format": "h3fx", "version": 1, "set": "rt", "meta": {"k": 3},
 "tensors": [{"name": "sub/t", "file": "sub/t.bin", "dtype": "bf16", "shape": [2, 3], "nbytes": 12,
 "sha256": ")") + fx::sha256_hex(bytes, sizeof(vals)) + R"(", "desc": ""}]})";
    {
        std::ofstream f(dir / "manifest.json", std::ios::binary);
        f << man;
    }
    auto m = fx::Manifest::load(dir.string());
    CHECK(m.set_name == "rt");
    CHECK(m.meta.at("k").num() == 3);
    auto t = m.read("sub/t");
    CHECK(t.info.shape.size() == 2 && t.info.shape[1] == 3);
    auto f = t.to_f32();
    CHECK(f[0] == 1.0F && f[2] == 3.0F && f[3] == -1.0F && std::isinf(f[5]));
    {  // corrupt one byte -> sha mismatch must throw
        std::fstream g(dir / "sub" / "t.bin", std::ios::in | std::ios::out | std::ios::binary);
        g.seekp(0);
        g.put('\x01');
    }
    bool threw = false;
    try {
        (void)m.read("sub/t");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
    fs::remove_all(dir);
}

static void test_real_set(const std::string& dir) {
    auto m = fx::Manifest::load(dir);
    std::uint64_t total = 0;
    for (const auto& ti : m.tensors) {
        auto t = m.read(ti.name, true);
        total += t.bytes.size();
    }
    std::printf("real set %s: %zu tensors, %llu bytes, all sha256 ok\n", m.set_name.c_str(), m.tensors.size(),
                static_cast<unsigned long long>(total));
}

int main(int argc, char** argv) {
    test_sha256();
    test_json();
    test_conversions();
    test_roundtrip();
    for (int i = 1; i < argc; ++i) {
        try {
            test_real_set(argv[i]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL real set %s: %s\n", argv[i], e.what());
            ++failures;
        }
    }
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
