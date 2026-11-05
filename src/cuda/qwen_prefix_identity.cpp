#include "strix/cuda_qwen_prefix_identity.hpp"
#include "strix/cuda_qwen_cohort.hpp"
#include "strix/cuda_qwen_prefix_plan.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace strix::cuda::qwen_prefix {

std::string canonical_identity(const IdentityFields& fields) {
    std::string out;
    for (const auto& [name, value] : fields) {
        for (const auto* part : {&name, &value}) out += std::to_string(part->size()) + ":" + *part;
    }
    return out;
}

ModelContentIdentity identity_key(const IdentityFields& fields) {
    const auto text = canonical_identity(fields);
    return sha256_bytes(std::as_bytes(std::span(text.data(), text.size())));
}

IdentityFields identity_fields(const qwen_prefill::Config& c, std::uint32_t block) {
    validate_block(block);
    IdentityFields f;
    const auto put = [&](std::string name, std::string value) { f.emplace_back(std::move(name), std::move(value)); };
    const auto file = [&](const std::string& name, const std::string& path) {
        put(name + ".path", path.empty() ? "" : std::filesystem::weakly_canonical(path).generic_string());
        if (path.empty()) return;
        put(name + ".size", std::to_string(std::filesystem::file_size(path)));
        put(name + ".mtime", std::to_string(std::filesystem::last_write_time(path).time_since_epoch().count()));
    };
    const auto model = [&](const std::string& name, const std::string& path) {
        file(name, path);
        const auto at = path.rfind("-00001-of-");
        if (at == std::string::npos || path.size() < at + 20) return;
        const auto total = std::stoul(path.substr(at + 10, 5));
        for (unsigned i = 2; i <= total; ++i) {
            const auto number = std::to_string(i);
            file(name + ".part" + number, path.substr(0, at + 1) + std::string(5 - number.size(), '0') + number + path.substr(at + 6));
        }
    };
    put("layout", "qwen-rtx-prefix-v1");
    model("model", c.model_gguf);
    model("mtp", c.mtp_sidecar_gguf);
    file("draft_vocab", c.draft_vocab);
    file("rope", c.rope_table);
    put("tables.path", std::filesystem::weakly_canonical(c.tables_dir).generic_string());
    // Include the files themselves: a directory's timestamp need not change on overwrite.
    std::vector<std::filesystem::path> tables;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(c.tables_dir))
        if (entry.is_regular_file()) tables.push_back(entry.path());
    std::sort(tables.begin(), tables.end());
    for (const auto& path : tables) file("table." + path.lexically_relative(c.tables_dir).generic_string(), path.string());
    put("max_positions", std::to_string(c.max_positions));
    put("segment_rows", std::to_string(block));
    put("device", std::to_string(c.device));
    put("ple_host_pinned", c.ple_host_pinned ? "1" : "0");
    put("tail_large", c.tail_large ? "1" : "0");
    put("final_block_rows", std::to_string(c.final_block_rows));
    put("decode", c.decode ? "1" : "0");
    put("export_state", c.export_state ? "1" : "0");
    put("release_by_layer", c.release_by_layer ? "1" : "0");
    for (const auto rows : c.block_schedule) put("block_rows", std::to_string(rows));
    // All getenv readers in the Qwen CUDA translation units, including scheduling
    // selectors whose arithmetic equivalence is currently expected by the engine.
    for (const auto* name : {"STRIX_QRTX_DECODE_KERNELS", "STRIX_QRTX_SIDE_SHARED", "STRIX_QRTX_GRAPHS",
                            "STRIX_QRTX_GRAPH_TABLE", "STRIX_QRTX_MULTISEG", "STRIX_QRTX_HEAD1",
                            "STRIX_QRTX_DRAFT_SYNC", "STRIX_QSA_SCORE_WAVE", "STRIX_QRTX_NVTX"}) {
        const char* raw = std::getenv(name);
        const bool default_on = std::string(name) == "STRIX_QRTX_DECODE_KERNELS" || std::string(name) == "STRIX_QRTX_SIDE_SHARED" ||
            std::string(name) == "STRIX_QRTX_GRAPHS" || std::string(name) == "STRIX_QRTX_GRAPH_TABLE" || std::string(name) == "STRIX_QRTX_MULTISEG";
        put(name, (default_on ? !(raw && raw[0] == '0') : raw && raw[0] == '1') ? "1" : "0");
    }
    // Cohort readers in include/strix/cuda_qwen_cohort.hpp (effective values).
    put("STRIX_QRTX_WINDOW_ROWS", std::to_string(strix::cuda::qwen_cohort::window_rows_default()));
    put("STRIX_QRTX_DRAFT_BATCH", strix::cuda::qwen_cohort::draft_batch_default() ? "1" : "0");
    const auto* cap = std::getenv("STRIX_QRTX_GRAPH_CACHE");
    put("STRIX_QRTX_GRAPH_CACHE", std::to_string(cap ? std::max<unsigned long>(1, std::strtoul(cap, nullptr, 10)) : 512));
    return f;
}

std::uint64_t default_host_budget() {
    std::uint64_t available = 0;
#ifdef _WIN32
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) available = memory.ullAvailPhys;
#else
    const auto pages = sysconf(_SC_AVPHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) available = static_cast<std::uint64_t>(pages) * page_size;
#endif
    return std::min<std::uint64_t>(8192ULL << 20, available / 4);
}

}  // namespace strix::cuda::qwen_prefix
