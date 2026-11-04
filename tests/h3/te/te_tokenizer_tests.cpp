// H3 text encoder host tests.
//   strix-h3-te-tests host                       NFC, pre-tokenizer, presentation helpers (no files)
//   strix-h3-te-tests tokenizer CASES.json DIR   token ids vs the reference dump (DIR: vocab.json,
//                                                merges.txt, tokenizer_config.json); skipped if DIR is empty
//   strix-h3-te-tests vision DIR                 patches / MRoPE positions / tags vs the vision fixtures
//                                                (DIR with manifest.json + <case>.*.f32); skipped if empty
#include "h3/te/te_tokenizer.hpp"
#include "h3/te/te_vision_host.hpp"

#include "strix/serve/json.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::h3::te;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}

std::string hex(std::string_view s) {
    std::string out;
    char buf[8];
    for (const unsigned char c : s) { std::snprintf(buf, sizeof buf, "%02X ", c); out += buf; }
    return out;
}

void check_nfc(const char* in, const char* want, const char* name) {
    const auto got = Tokenizer::nfc(in);
    check(got == want, std::string("nfc ") + name + ": got " + hex(got) + " want " + hex(want));
}

void check_split(const char* in, std::vector<std::string> want, const char* name) {
    const auto got = Tokenizer::pre_tokenize(in);
    std::string g;
    for (const auto& p : got) g += "[" + p + "]";
    std::string w;
    for (const auto& p : want) w += "[" + p + "]";
    check(got == want, std::string("split ") + name + ": got " + g + " want " + w);
}

int host() {
    check_nfc("caf\x65\xCC\x81", "caf\xC3\xA9", "e+acute");
    check_nfc("\xE2\x84\xAB", "\xC3\x85", "angstrom singleton");
    check_nfc("\xE1\x84\x80\xE1\x85\xA1", "\xEA\xB0\x80", "hangul LV");
    check_nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", "\xEA\xB0\x81", "hangul LVT");
    // a + dot below (220) + grave (230) in either order -> a-with-dot-below + grave
    check_nfc("a\xCC\x80\xCC\xA3", "\xE1\xBA\xA1\xCC\x80", "reorder+compose");
    check_nfc("\xEF\xAC\x81", "\xEF\xAC\x81", "ligature fi is compat only");
    check_nfc("\xFF\xFE", "\xFF\xFE", "invalid bytes pass through");
    check_split("Hello world", {"Hello", " world"}, "words");
    check_split("I'm it's", {"I", "'m", " it", "'s"}, "contractions");
    check_split("a  b", {"a", " ", " b"}, "double space");
    check_split("x\n\n y", {"x", "\n\n", " y"}, "newlines");
    check_split("12345", {"1", "2", "3", "4", "5"}, "digits");
    check_split("ok!!!\n", {"ok", "!!!\n"}, "punct+newline");
    check_split("end   ", {"end", "   "}, "trailing ws");
    check_split("e\xCC\x81t\xC3\xA9", {"e", "\xCC\x81t\xC3\xA9"}, "mark is not a letter");
    check(format_seconds(0.25) == "0.2", "format 0.25");
    check(format_seconds(1.25) == "1.2", "format 1.25");
    check(format_seconds(0.75) == "0.8", "format 0.75");
    check(format_seconds(2.0) == "2.0", "format 2.0");
    Presented p;
    p.ids = {5, kVisionStart, kVisionSlot, kVisionEnd, 6};
    const auto tags = token_tags(p, {3});
    check(tags == std::vector<std::int32_t>{1, 0, 0, 0, 0, 0, 1}, "token tags");
    std::printf("host: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::int32_t> ints(const strix::serve::Json& a) {
    std::vector<std::int32_t> out;
    for (const auto& v : a.as_array()) out.push_back(static_cast<std::int32_t>(v.as_int()));
    return out;
}

std::string show(const std::vector<std::int32_t>& v) {
    std::string s;
    for (const auto x : v) s += std::to_string(x) + " ";
    return s;
}

int tokenizer(const std::string& cases_path, const std::string& dir) {
    if (dir.empty()) { std::printf("tokenizer: SKIPPED (STRIX_H3_TE_TOKENIZER_DIR not set)\n"); return 0; }
    const auto tok = Tokenizer::load(dir);
    const auto doc = strix::serve::parse_json(read_file(cases_path));
    // the dump's embedding directory holds one 3-row embedding named "fx"
    const EmbeddingLookup lookup = [](std::string_view name) { return name == "fx" ? 3 : -1; };
    int n = 0;
    for (const auto& c : doc.find("cases")->as_array()) {
        const auto& text = c.find("text")->as_string();
        const auto run = [&](const EmbeddingLookup* lk, const char* key) {
            auto got = present(*tok, text, 0, {}, lk).ids;
            const auto want = ints(*c.find(key));
            check(got == want, std::string(key) + " case \"" + text + "\": got " + show(got) + " want " + show(want));
            ++n;
        };
        run(&lookup, "ids");
        if (const auto* nd = c.find("ids_no_embedding_dir"); nd != nullptr) run(nullptr, "ids_no_embedding_dir");
    }
    for (const auto& c : doc.find("presentation")->as_array()) {
        std::vector<RefItem> items;
        for (const auto& it : c.find("items")->as_array()) {
            RefItem r;
            const auto& type = it.find("type")->as_string();
            r.kind = type == "image" ? RefItem::Kind::image : type == "audio" ? RefItem::Kind::audio : RefItem::Kind::video;
            if (r.kind == RefItem::Kind::video) {
                const auto* ts = it.find("timestamps");
                if (ts != nullptr && ts->is_array()) for (const auto& t : ts->as_array()) r.timestamps.push_back(t.as_double());
                r.frames = static_cast<int>(it.find("frames")->as_int());  // used when timestamps are absent
            }
            items.push_back(std::move(r));
        }
        const auto& prompt = c.find("prompt")->as_string();
        const auto got = present(*tok, prompt, static_cast<int>(c.find("images")->as_int()), items, &lookup).ids;
        const auto want = ints(*c.find("ids"));
        check(got == want, "presentation \"" + prompt + "\": got " + show(got) + " want " + show(want));
        ++n;
    }
    std::printf("tokenizer: %d cases, %s\n", n, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}

template <typename T>
std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<std::size_t>(f.tellg());
    std::vector<T> v(n / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    return v;
}

int vision(const std::string& dir) {
    if (dir.empty()) { std::printf("vision: SKIPPED (no fixture dir)\n"); return 0; }
    const auto man = strix::serve::parse_json(read_file(dir + "/manifest.json"));
    int n = 0;
    for (const auto& c : man.find("cases")->as_array()) {
        const auto name = c.find("name")->as_string();
        std::vector<Patches> blocks;
        int b = 0;
        for (const auto& v : c.find("vision")->as_array()) {
            const auto base = dir + "/" + name + ".vis" + std::to_string(b);
            const auto& shape = c.find("files")->find("vis" + std::to_string(b) + ".image.f32")->find("shape")->as_array();
            Frames f;
            f.t = static_cast<int>(shape.at(0).as_int());
            f.h = static_cast<int>(shape.at(1).as_int());
            f.w = static_cast<int>(shape.at(2).as_int());
            f.rgb = read_bin<float>(base + ".image.f32");
            auto p = make_patches(f);
            const auto want = read_bin<float>(base + ".patches.f32");
            const auto& g = v.find("grid_thw")->as_array().at(0).as_array();
            check(p.grid_h == g.at(1).as_int() && p.grid_w == g.at(2).as_int(), name + " vis" + std::to_string(b) + " grid");
            check(p.data == want, name + " vis" + std::to_string(b) + " patches bitwise");
            blocks.push_back(std::move(p));
            ++b;
            ++n;
        }
        const auto ids = read_bin<std::int32_t>(dir + "/" + name + ".ids.i32");
        const auto lay = expand_layout(ids, blocks);
        const auto tags = read_bin<std::int32_t>(dir + "/" + name + ".tags.i32");
        const auto pos = read_bin<float>(dir + "/" + name + ".positions.f32");
        check(lay.tags == tags, name + " tags");
        check(lay.positions == pos, name + " MRoPE positions bitwise");
        ++n;
    }
    std::printf("vision: %d checks, %s\n", n, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "host";
    try {
        if (mode == "host") return host();
        if (mode == "tokenizer" && argc > 2) return tokenizer(argv[2], argc > 3 ? argv[3] : "");
        if (mode == "vision") return vision(argc > 2 ? argv[2] : "");
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("usage: strix-h3-te-tests host | tokenizer CASES.json DIR\n");
    return 2;
}
