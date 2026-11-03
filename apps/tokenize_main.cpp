// strix-tokenize: the server's tokenizer from a GGUF file, for prompts and checks.
//   MODEL may also be a Hugging Face tokenizer.json, or a model directory that holds one (EXL3 models such as GLM-5.3).
//   strix-tokenize MODEL.gguf encode FILE   token IDs of FILE's bytes (special tokens parsed), comma-separated
//   strix-tokenize MODEL.gguf plain FILE    same without special-token parsing
//   strix-tokenize MODEL.gguf decode FILE   text of the comma/space-separated token IDs in FILE
//   strix-tokenize MODEL.gguf time FILE [K] the tokenizer's load time and the encode time of FILE (special tokens parsed),
//                                           K runs (default 5): first, min and median, and the token count
#include "strix/serve/bpe_tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::shared_ptr<const strix::serve::BpeTokenizer> load_tokenizer(const std::string& model) {
    namespace fs = std::filesystem;
    if (fs::is_directory(model)) return strix::serve::BpeTokenizer::load_hf_json((fs::path(model) / "tokenizer.json").string());
    if (fs::path(model).extension() == ".json") return strix::serve::BpeTokenizer::load_hf_json(model);
    return strix::serve::BpeTokenizer::load_gguf(model);
}

}  // namespace

int main(int argc, char** argv) try {
    const std::string mode = argc > 2 ? argv[2] : "";
    if (argc != 4 && !(argc == 5 && mode == "time")) {
        std::cerr << "usage: strix-tokenize MODEL.gguf|tokenizer.json|MODEL_DIR encode|plain|decode|time FILE [K]\n";
        return 2;
    }
    std::ifstream file(argv[3], std::ios::binary);
    if (!file) throw std::runtime_error(std::string("cannot read ") + argv[3]);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto t_load = std::chrono::steady_clock::now();
    const auto tokenizer = load_tokenizer(argv[1]);
    if (mode == "time") {
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        const double load_ms = ms(std::chrono::steady_clock::now() - t_load);
        const int runs = argc == 5 ? std::max(1, std::stoi(argv[4])) : 5;
        std::vector<double> t;
        std::size_t count = 0;
        for (int k = 0; k < runs; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            count = tokenizer->encode(text, true).size();
            t.push_back(ms(std::chrono::steady_clock::now() - t0));
        }
        const double first = t.front();
        std::sort(t.begin(), t.end());
        std::printf("tokenize: %zu bytes -> %zu tokens; load %.3f ms; encode first %.3f ms, min %.3f ms, median %.3f ms (%d runs)\n", text.size(),
                    count, load_ms, first, t.front(), t[t.size() / 2U], runs);
        return 0;
    }
    if (mode == "encode" || mode == "plain") {
        const auto ids = tokenizer->encode(text, mode == "encode");
        for (std::size_t i = 0; i < ids.size(); ++i) std::cout << (i ? "," : "") << ids[i];
        std::cout << '\n';
        return 0;
    }
    if (mode == "decode") {
        std::string spaced = text;
        for (auto& c : spaced) if (c == ',') c = ' ';
        std::istringstream words(spaced);
        std::vector<std::uint32_t> ids;
        for (std::uint64_t id = 0; words >> id;) ids.push_back(static_cast<std::uint32_t>(id));
        std::cout << tokenizer->decode(ids);
        return 0;
    }
    std::cerr << "unknown mode " << mode << '\n';
    return 2;
} catch (const std::exception& error) {
    std::cerr << "strix-tokenize: " << error.what() << '\n';
    return 1;
}
