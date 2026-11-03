// strix-gemma4-tokenize: the gemma4 (Nemotron 3.5 Bolt E4B) engine's own tokenizer, for prompts and checks.
//   It loads the official tokenizer.json through strix::gemma4::serve::Tokenizer, the tokenizer the engine serves with;
//   strix-tokenize uses the generic BPE path instead. Encoding never inserts BOS/EOS (the chat renderer owns those).
//   strix-gemma4-tokenize MODEL_DIR|tokenizer.json encode FILE   token IDs of FILE's bytes (special tokens parsed), comma-separated
//   strix-gemma4-tokenize MODEL_DIR|tokenizer.json plain FILE    same without special-token parsing
//   strix-gemma4-tokenize MODEL_DIR|tokenizer.json decode FILE   text of the comma/space-separated token IDs in FILE
#include "gemma4/serve/tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv) try {
    namespace fs = std::filesystem;
    if (argc != 4) {
        std::cerr << "usage: strix-gemma4-tokenize MODEL_DIR|tokenizer.json encode|plain|decode FILE\n";
        return 2;
    }
#ifdef _WIN32
    // Decoded text is raw UTF-8 bytes: no CRLF translation, so the output matches the Linux build byte for byte.
    (void)_setmode(_fileno(stdout), _O_BINARY);
#endif
    const std::string mode = argv[2];
    if (mode != "encode" && mode != "plain" && mode != "decode") {
        std::cerr << "unknown mode " << mode << '\n';
        return 2;
    }
    std::ifstream file(argv[3], std::ios::binary);
    if (!file) throw std::runtime_error(std::string("cannot read ") + argv[3]);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const fs::path model(argv[1]);
    const auto tokenizer = strix::gemma4::serve::Tokenizer::load(fs::is_directory(model) ? model / "tokenizer.json" : model);
    if (mode == "decode") {
        std::string spaced = text;
        for (auto& c : spaced) if (c == ',') c = ' ';
        std::istringstream words(spaced);
        std::vector<std::uint32_t> ids;
        for (std::uint64_t id = 0; words >> id;) ids.push_back(static_cast<std::uint32_t>(id));
        std::cout << tokenizer->decode(ids);
        return 0;
    }
    const auto ids = tokenizer->encode(text, mode == "encode");
    for (std::size_t i = 0; i < ids.size(); ++i) std::cout << (i ? "," : "") << ids[i];
    std::cout << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-gemma4-tokenize: " << error.what() << '\n';
    return 1;
}
