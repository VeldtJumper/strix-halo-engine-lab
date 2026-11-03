// GLM-4/5 tokenizer: BpeTokenizer::load_hf_json (pre-tokenizer glm4, ignore_merges) equals HuggingFace `tokenizers` on
// the fixtures written by tools/glm/make_tokenizer_fixtures.py.
//   strix-glm-tokenizer-tests FIXTURES.json TOKENIZER_JSON
// Without both files the test only checks the glm4 pre-tokenizer on built-in cases.
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/json.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using strix::serve::BpeTokenizer;
using strix::serve::Json;

struct Tally { std::size_t pass = 0, fail = 0; };
std::map<std::string, Tally> tallies;
int printed = 0;

void check(const std::string& group, bool ok, const std::string& what) {
    auto& t = tallies[group];
    if (ok) { ++t.pass; return; }
    ++t.fail;
    if (t.fail <= 12U && printed++ < 60) std::cout << "FAIL " << group << ": " << what << "\n";
}
std::string escape(const std::string& s) {
    std::string out;
    strix::serve::append_json_string(out, s);
    return out.size() > 300U ? out.substr(0, 300U) + "..." : out;
}
std::string ids_text(const std::vector<std::uint32_t>& ids) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < ids.size() && i < 40U; ++i) os << (i ? "," : "") << ids[i];
    if (ids.size() > 40U) os << ",...(" << ids.size() << ")";
    os << ']';
    return os.str();
}
std::vector<std::uint32_t> ids_of(const Json& a) {
    std::vector<std::uint32_t> out;
    for (const auto& v : a.as_array()) out.push_back(static_cast<std::uint32_t>(v.as_int()));
    return out;
}
// Rust String::from_utf8_lossy (what `tokenizers` decode returns): each maximal invalid subpart becomes U+FFFD.
std::string utf8_lossy(const std::string& s) {
    std::string out;
    std::size_t i = 0;
    const auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    while (i < s.size()) {
        const unsigned c = b(i);
        std::size_t need = 0;
        unsigned lo = 0x80, hi = 0xBF;
        if (c < 0x80) { out.push_back(static_cast<char>(c)); ++i; continue; }
        if (c >= 0xC2 && c <= 0xDF) need = 1;
        else if (c >= 0xE0 && c <= 0xEF) { need = 2; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
        else if (c >= 0xF0 && c <= 0xF4) { need = 3; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
        else { out += "\xEF\xBF\xBD"; ++i; continue; }
        std::size_t k = 1;
        for (; k <= need; ++k) {
            if (i + k >= s.size()) break;
            const unsigned d = b(i + k);
            const unsigned l = k == 1 ? lo : 0x80, h = k == 1 ? hi : 0xBF;
            if (d < l || d > h) break;
        }
        if (k == need + 1) { out.append(s, i, need + 1); i += need + 1; }
        else { out += "\xEF\xBF\xBD"; i += k; }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    // glm4 pre-tokenizer on built-in cases (\p{L}-only words, numbers of up to three digits)
    const auto pieces = [](const char* text) { return BpeTokenizer::pre_tokenize(text, "glm4"); };
    check("pre", pieces("1234567") == std::vector<std::string>{"123", "456", "7"}, "digits");
    check("pre", pieces("abc123def") == std::vector<std::string>{"abc", "123", "def"}, "letters digits");
    check("pre", pieces("e\xCC\x81") == std::vector<std::string>{"e", "\xCC\x81"}, "mark is not a letter");
    check("pre", pieces("I'm") == std::vector<std::string>{"I", "'m"}, "contraction");
    if (argc >= 3 && std::filesystem::exists(argv[1]) && std::filesystem::exists(argv[2])) {
        std::ifstream in(argv[1], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const Json fixtures = strix::serve::parse_json(text);
        const auto tok = BpeTokenizer::load_hf_json(argv[2]);
        check("vocab", tok->vocab_size() == static_cast<std::size_t>(fixtures.find("vocab_size")->as_int()),
              std::to_string(tok->vocab_size()));
        check("eos", tok->eos_token() == static_cast<std::uint32_t>(fixtures.find("eos")->as_int()), std::to_string(tok->eos_token()));
        for (const auto& c : fixtures.find("encode")->as_array()) {
            const auto& t = c.find("text")->as_string();
            const auto want = ids_of(*c.find("ids")), want_plain = ids_of(*c.find("ids_plain"));
            const auto got = tok->encode(t, true), got_plain = tok->encode(t, false);
            check("encode", got == want, escape(t) + " -> " + ids_text(got) + " want " + ids_text(want));
            check("encode-plain", got_plain == want_plain, escape(t) + " -> " + ids_text(got_plain) + " want " + ids_text(want_plain));
            check("decode", utf8_lossy(tok->decode(want, false)) == c.find("decoded")->as_string(), ids_text(want));
            check("decode-skip", utf8_lossy(tok->decode(want, true)) == c.find("decoded_skip")->as_string(), ids_text(want));
        }
        for (const auto& c : fixtures.find("decode")->as_array()) {
            const auto ids = ids_of(*c.find("ids"));
            const auto dec = utf8_lossy(tok->decode(ids, false));
            check("decode-ids", dec == c.find("text")->as_string(), ids_text(ids) + " -> " + escape(dec) + " want " + escape(c.find("text")->as_string()));
            check("decode-ids-skip", utf8_lossy(tok->decode(ids, true)) == c.find("text_skip")->as_string(), ids_text(ids));
        }
    } else {
        std::cout << "fixtures / tokenizer.json not given: pre-tokenizer checks only\n";
    }
    std::size_t fail = 0;
    for (const auto& [name, t] : tallies) {
        std::cout << name << ": " << t.pass << " pass, " << t.fail << " fail\n";
        fail += t.fail;
    }
    std::cout << (fail ? "FAILED" : "OK") << "\n";
    return fail ? 1 : 0;
}
