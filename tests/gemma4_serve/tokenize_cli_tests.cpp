// Fixture check of strix-gemma4-tokenize: the CLI binary must print the serve tokenizer's IDs for golden
// texts (ASCII, Spanish accents, Japanese, code, a chat prompt with <bos>/<|turn>/<turn|> special tokens),
// equal both to the reference goldens and to an in-process g4::Tokenizer, in encode, plain and decode modes.
#include "gemma4/serve/tokenizer.hpp"
#include "strix/serve/json.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace g4 = strix::gemma4::serve;
using strix::serve::Json;
namespace fs = std::filesystem;
namespace {
void check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
std::string slurp(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    check(static_cast<bool>(f), "cannot read " + path.string());
    return std::string(std::istreambuf_iterator<char>(f), {});
}
const Json& get(const Json& j, std::string_view k) {
    const auto* p = j.find(k); if (!p) throw std::runtime_error("missing fixture key " + std::string(k)); return *p;
}
std::vector<std::uint32_t> ids(const Json& j) {
    std::vector<std::uint32_t> v;
    for (const auto& i : j.as_array()) v.push_back(static_cast<std::uint32_t>(i.as_int()));
    return v;
}
std::string csv(const std::vector<std::uint32_t>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}
// Runs the CLI on `input` written to a file; returns its stdout bytes.
std::string run(const fs::path& cli, const fs::path& tok, const std::string& mode, const std::string& input, const fs::path& work) {
    const auto in = work / ("in-" + mode), out = work / ("out-" + mode);
    { std::ofstream f(in, std::ios::binary); f << input; }
    std::string cmd = "\"" + cli.string() + "\" \"" + tok.string() + "\" " + mode + " \"" + in.string() + "\" > \"" + out.string() + "\"";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";  // cmd.exe strips one outer quote pair
#endif
    check(std::system(cmd.c_str()) == 0, "CLI failed: " + cmd);
    return slurp(out);
}
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc >= 4, "usage: tokenize-cli-tests CLI FIXTURE_DIR WORK_DIR [TOKENIZER_JSON]");
        if (argc < 5 || !*argv[4]) { std::cout << "NO-REFERENCE: official tokenizer asset not configured\n"; return 77; }
        const fs::path cli(argv[1]), dir(argv[2]), work(argv[3]), tok_path(argv[4]);
        fs::create_directories(work);
        const auto tok = g4::Tokenizer::load(tok_path);
        check(tok->find("<|turn>") == 105 && tok->find("<turn|>") == 106, "chat special token IDs");
        const auto goldens = strix::serve::parse_json(slurp(dir / "tok_goldens.json"));
        const auto extended = strix::serve::parse_json(slurp(dir / "extended.json"));
        struct Case { std::string name, mode; const Json* c; };
        const auto& enc = get(goldens, "encode").as_array();
        const std::vector<Case> cases{
            {"ascii", "encode", &enc.at(2)}, {"spanish", "encode", &enc.at(6)}, {"japanese", "encode", &enc.at(10)},
            {"code", "encode", &enc.at(17)}, {"chat-special", "encode", &get(goldens, "chat").as_array().at(0)},
            {"plain-special-text", "plain", &get(extended, "encode_no_special").as_array().at(1)}};
        for (const auto& c : cases) {
            const auto& text = c.c->find("rendered") ? get(*c.c, "rendered").as_string() : get(*c.c, "text").as_string();
            const auto expected = ids(get(*c.c, "ids"));
            check(tok->encode(text, c.mode == "encode") == expected, c.name + ": serve tokenizer differs from golden");
            const auto got = run(cli, tok_path, c.mode, text, work);
            check(got == csv(expected) + "\n", c.name + ": CLI " + c.mode + " IDs\nexpected: " + csv(expected) + "\ngot:      " + got);
            check(run(cli, tok_path, "decode", got, work) == text, c.name + ": CLI decode round trip");
            std::cout << "ok " << c.name << " " << c.mode << " " << expected.size() << " ids\n";
        }
        // A model directory resolves to its tokenizer.json.
        if (tok_path.filename() == "tokenizer.json") check(run(cli, tok_path.parent_path(), "encode", "Hello, world!", work) == csv(ids(get(enc.at(2), "ids"))) + "\n", "MODEL_DIR form");
        std::cout << "PASS gemma4 tokenize cli\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
