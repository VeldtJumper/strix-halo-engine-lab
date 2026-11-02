// H3 text-encoder pack reader: corrupt-pack rejection (host only).
//   strix-h3-te-pack-tests [TMPDIR]
#include "h3/te/te_pack_reader.hpp"
#include "h3/te/te_sha256.hpp"
#include "te_fixture_io.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using namespace strix::h3::te;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}

std::string g_dir = ".";

// Writes a safetensors file: header JSON (as given) + data bytes; returns its path.
std::string write_file(const std::string& name, const std::string& header, const std::vector<unsigned char>& data,
                       std::uint64_t header_len_override = 0) {
    const std::string path = g_dir + "/" + name;
    std::ofstream f(path, std::ios::binary);
    const std::uint64_t n = header_len_override ? header_len_override : header.size();
    for (int i = 0; i < 8; ++i) f.put(static_cast<char>((n >> (8 * i)) & 0xFFU));
    f.write(header.data(), std::streamsize(header.size()));
    f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    return path;
}

bool rejects(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const PackError&) {
        return true;
    } catch (const std::exception& e) {
        std::printf("  (non-PackError exception: %s)\n", e.what());
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    const std::vector<unsigned char> data(4 + 8 + 2 * 3, 0);  // a: F32[1], b: U8[2,4], c: BF16[3]
    const std::string ok_header =
        R"({"__metadata__":{"format":"x"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
        R"("b":{"dtype":"U8","shape":[2,4],"data_offsets":[4,12]},"c":{"dtype":"BF16","shape":[3],"data_offsets":[12,18]}})";
    {
        std::vector<unsigned char> d = data;
        const float one = 1.5F;
        std::memcpy(d.data(), &one, 4);
        const auto p = write_file("ok.safetensors", ok_header, d);
        bool ok = false;
        try {
            SafetensorsFile f(p);
            ok = f.read_f32_scalar("a") == 1.5F && f.read_vector<std::uint8_t>("b", "U8", {2, 4}).size() == 8 &&
                 f.read_vector<std::uint16_t>("c", "BF16", {3}).size() == 3;
            check(rejects([&] { (void)f.read_vector<std::uint8_t>("b", "U8", {4, 2}); }), "shape mismatch rejected");
            check(rejects([&] { (void)f.read_vector<std::uint16_t>("b", "U8", {2, 4}); }), "element size mismatch rejected");
            check(rejects([&] { (void)f.read_vector<std::uint8_t>("b", "I8", {2, 4}); }), "dtype mismatch rejected");
            check(rejects([&] { (void)f.read_f32_scalar("c"); }), "non-scalar tscale rejected");
            check(rejects([&] { (void)f.read_f32_scalar("b"); }), "U8 tscale rejected");
            check(rejects([&] { (void)f.info("missing"); }), "missing tensor rejected");
            std::vector<unsigned char> small(4);
            check(rejects([&] { f.read("b", small.data(), small.size()); }), "short destination rejected");
        } catch (const std::exception& e) {
            std::printf("  valid file threw: %s\n", e.what());
        }
        check(ok, "valid file reads");
    }
    const auto bad = [&](const char* name, const std::string& header, const std::vector<unsigned char>& d,
                         std::uint64_t hl = 0) {
        const auto p = write_file(name, header, d, hl);
        check(rejects([&] { SafetensorsFile f(p); }), std::string("rejects ") + name);
    };
    bad("bytes_vs_shape.safetensors",
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", std::vector<unsigned char>(4));
    bad("past_end.safetensors", R"({"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}})", std::vector<unsigned char>(4));
    bad("overlap.safetensors",
        R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}})",
        std::vector<unsigned char>(8));
    bad("unknown_dtype.safetensors", R"({"a":{"dtype":"Q4","shape":[4],"data_offsets":[0,4]}})", std::vector<unsigned char>(4));
    bad("negative_dim.safetensors", R"({"a":{"dtype":"U8","shape":[-4],"data_offsets":[0,4]}})", std::vector<unsigned char>(4));
    bad("reversed_offsets.safetensors", R"({"a":{"dtype":"U8","shape":[0],"data_offsets":[4,0]}})", std::vector<unsigned char>(4));
    bad("missing_fields.safetensors", R"({"a":{"dtype":"U8","shape":[4]}})", std::vector<unsigned char>(4));
    bad("bad_json.safetensors", R"({"a":{"dtype":"U8",)", std::vector<unsigned char>(4));
    bad("not_object.safetensors", R"([1,2,3])", std::vector<unsigned char>(4));
    bad("header_too_long.safetensors", R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})", std::vector<unsigned char>(4),
        1U << 20U);
    check(rejects([&] { SafetensorsFile f(g_dir + "/does_not_exist.safetensors"); }), "rejects missing file");
    bad("duplicate_key.safetensors",
        R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
        std::vector<unsigned char>(4));
    bad("duplicate_inner_key.safetensors", R"({"a":{"dtype":"U8","dtype":"I8","shape":[4],"data_offsets":[0,4]}})",
        std::vector<unsigned char>(4));

    // SHA-256 known answers (FIPS 180-2 examples) and the streaming split
    check(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
    check(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
    {
        const std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        check(sha256_hex(m.data(), m.size()) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
              "sha256 two-block message");
        std::string mil(1000000, 'a');
        Sha256 h;
        for (std::size_t i = 0; i < mil.size(); i += 777) h.update(mil.data() + i, std::min<std::size_t>(777, mil.size() - i));
        check(h.finish_hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "sha256 million a, streamed");
        Sha256 hp(true);
        hp.update(mil.data(), mil.size());
        check(hp.finish_hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "sha256 million a, portable");
        std::uint32_t st = 7;
        std::vector<unsigned char> rnd(100003);
        for (auto& c : rnd) { st = st * 1664525U + 1013904223U; c = static_cast<unsigned char>(st >> 24); }
        for (const std::size_t len : {0UL, 1UL, 55UL, 56UL, 63UL, 64UL, 65UL, 119UL, 128UL, 4097UL, 100003UL}) {
            Sha256 a, b(true);
            a.update(rnd.data(), len);
            b.update(rnd.data(), len);
            check(a.finish_hex() == b.finish_hex(), "sha256 accelerated == portable, len " + std::to_string(len));
        }
        std::printf("sha256 provider: %s\n", sha256_uses_sha_ni() ? "x86 SHA extensions" : "portable");
    }

    // te-pack.json: manifest parsing, byte counts, hashes, duplicate keys
    {
        const std::string dir = g_dir + "/manifest_pack";
        std::filesystem::create_directories(dir);
        const std::string payload = "pack payload bytes";
        std::ofstream(dir + "/w.safetensors", std::ios::binary) << payload;
        const std::string sha = sha256_hex(payload.data(), payload.size());
        const auto manifest = [&](const std::string& files, const std::string& extra = "") {
            return std::string(R"({"format":"h3-te-pack-1","n_layers":50,"hidden":5120,"ffn":25600,"n_heads":64,)") +
                   R"("n_kv_heads":8,"head_dim":128,"vocab":151936,"rms_eps":1e-06,"rope_theta":5000000.0,)" +
                   R"("final_norm":false,"mrope_sections":[24,20,20],"mrope_interleaved":true,)" + extra +
                   R"("files":{)" + files + "}}";
        };
        const auto entry = [](const std::string& name, std::size_t bytes, const std::string& h) {
            return "\"" + name + "\":{\"bytes\":" + std::to_string(bytes) + ",\"sha256\":\"" + h + "\"}";
        };
        const auto write_manifest = [&](const std::string& text) { std::ofstream(dir + "/te-pack.json", std::ios::binary) << text; };
        write_manifest(manifest(entry("w.safetensors", payload.size(), sha)));
        bool ok = false;
        try {
            const auto m = read_pack_manifest(dir);
            verify_pack_files(dir, m, {"w.safetensors"}, true, 2);
            ok = m.hidden == 5120 && m.mrope_sections.size() == 3 && m.files.size() == 1 && m.json_sha256.size() == 64;
        } catch (const std::exception& e) {
            std::printf("  valid manifest threw: %s\n", e.what());
        }
        check(ok, "valid manifest + hash verifies");
        {
            const auto m = read_pack_manifest(dir);
            check(rejects([&] { verify_pack_files(dir, m, {"other.safetensors"}, false); }), "unlisted file rejected");
        }
        std::string wrong = sha;
        wrong[0] = wrong[0] == '0' ? '1' : '0';
        write_manifest(manifest(entry("w.safetensors", payload.size(), wrong)));
        {
            const auto m = read_pack_manifest(dir);
            check(!rejects([&] { verify_pack_files(dir, m, {"w.safetensors"}, false); }), "hash skipped when disabled");
            check(rejects([&] { verify_pack_files(dir, m, {"w.safetensors"}, true); }), "wrong sha256 rejected");
        }
        write_manifest(manifest(entry("w.safetensors", payload.size() + 1, sha)));
        {
            const auto m = read_pack_manifest(dir);
            check(rejects([&] { verify_pack_files(dir, m, {"w.safetensors"}, false); }), "wrong byte count rejected");
        }
        write_manifest(manifest(entry("w.safetensors", payload.size(), sha), R"("hidden":4096,)"));
        check(rejects([&] { (void)read_pack_manifest(dir); }), "duplicate top-level key rejected");
        write_manifest(manifest(entry("w.safetensors", payload.size(), sha) + "," + entry("w.safetensors", payload.size(), sha)));
        check(rejects([&] { (void)read_pack_manifest(dir); }), "duplicate file entry rejected");
        write_manifest(manifest(entry("w.safetensors", payload.size(), "ABC")));
        check(rejects([&] { (void)read_pack_manifest(dir); }), "malformed sha256 rejected");
        write_manifest(manifest(entry("../w.safetensors", payload.size(), sha)));
        check(rejects([&] { (void)read_pack_manifest(dir); }), "path in file name rejected");
        write_manifest(R"({"format":"h3-te-pack-1","files":{}})");
        check(rejects([&] { (void)read_pack_manifest(dir); }), "missing geometry rejected");
    }
    {  // GPU-gate fixture reads and comparison shapes
        namespace fio = fixture_io;
        const auto raw = [](const std::string& name, std::size_t bytes) {
            const std::string path = g_dir + "/" + name;
            std::ofstream f(path, std::ios::binary);
            for (std::size_t i = 0; i < bytes; ++i) f.put(static_cast<char>(i + 1));
            return path;
        };
        check(fio::read_bin<float>(raw("fx_ok.f32", 16)).size() == 4, "fixture: 16 bytes -> 4 floats");
        check(fio::read_bin<float>(raw("fx_empty.f32", 0)).empty(), "fixture: empty file -> 0 floats");
        check(rejects([&] { (void)fio::read_bin<float>(raw("fx_odd.f32", 18)); }), "fixture: 18 bytes as f32 rejected");
        check(rejects([&] { (void)fio::read_bin<std::int32_t>(raw("fx_odd.i32", 7)); }), "fixture: 7 bytes as i32 rejected");
        check(rejects([&] { (void)fio::read_bin<float>(g_dir + "/fx_missing.f32"); }), "fixture: missing file rejected");
        check(!rejects([&] { fio::check_dims(10, 10, 2, 5); }), "compare: exact [2][5] accepted");
        check(rejects([&] { fio::check_dims(0, 0, 0, 5120); }), "compare: empty reference rejected");
        check(rejects([&] { fio::check_dims(10, 10, 2, 0); }), "compare: zero width rejected");
        check(rejects([&] { fio::check_dims(10, 9, 2, 5); }), "compare: short reference rejected");
        check(rejects([&] { fio::check_dims(9, 10, 2, 5); }), "compare: short output rejected");
        check(rejects([&] { fio::check_dims(11, 10, 2, 5); }), "compare: long output rejected");
    }
    std::printf("pack reader: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
