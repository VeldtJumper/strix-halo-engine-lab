#include "strix/cuda_qwen_prefill.hpp"
#include "strix/cuda_qwen_prefix_plan.hpp"
#include "strix/cuda_qwen_prefix_store.hpp"
#include "strix/cuda_vram_guard.hpp"
#include "strix/model_content_hash.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace qp = strix::cuda::qwen_prefill;
namespace px = strix::cuda::qwen_prefix;
using Clock = std::chrono::steady_clock;

namespace {
struct Buffer {
    std::byte* data = nullptr;
    explicit Buffer(std::uint64_t bytes) {
        const auto rc = cudaMallocHost(&data, bytes);
        if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
    }
    ~Buffer() { if (data) cudaFreeHost(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

double ms(Clock::time_point from) { return std::chrono::duration<double, std::milli>(Clock::now() - from).count(); }

double prefill(qp::Prefill& e, qp::SessionId session, const std::vector<std::uint32_t>& prompt,
               std::uint64_t from, std::uint64_t end, std::uint32_t chunk, bool overlap) {
    const auto begin = Clock::now();
    while (from < end) {
        const auto rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(chunk, end - from));
        if (overlap) {
            e.prefill_chunk_async(session, prompt, prompt.size() - 1, from, rows);
            from = e.prefill_wait().length;
        } else from = e.prefill_chunk(session, prompt, prompt.size() - 1, from, rows);
    }
    return ms(begin);
}

strix::ModelContentIdentity digest(qp::Prefill& e, qp::SessionId session, bool mtp, std::uint32_t block) {
    strix::Sha256Fast hash;
    const auto length = e.length(session);
    Buffer rows(block * e.row_bytes()), fixed(e.fixed_bytes());
    for (std::uint64_t first = 0; first < length; first += block) {
        const auto count = std::min<std::uint64_t>(block, length - first);
        e.export_rows(session, first, count, rows.data);
        const auto qsa_bytes = count * 12 * 2304;
        hash.update({rows.data, static_cast<std::size_t>(qsa_bytes)});
        if (mtp) {
            // The final teacher pair is absent until decode supplies the next token.
            // Hash every defined byte; allocator residue is not session state.
            const auto pairs = std::min<std::uint64_t>(count, length - 1 - first);
            hash.update({rows.data + qsa_bytes, static_cast<std::size_t>(pairs * 2048)});
            hash.update({rows.data + qsa_bytes + count * 2048, static_cast<std::size_t>(pairs * 256)});
        }
    }
    e.export_fixed(session, fixed.data);
    hash.update({fixed.data, static_cast<std::size_t>(e.fixed_bytes())});
    return hash.finalize();
}

struct Decode {
    std::vector<std::uint32_t> ids;
    std::vector<strix::ModelContentIdentity> logits;
    bool operator==(const Decode&) const = default;
};

Decode decode(qp::Prefill& e, qp::SessionId session, std::uint32_t pending, bool mtp, std::uint32_t depth, float min_p) {
    Decode out;
    auto position = e.length(session);
    while (out.ids.size() < 64) {
        std::vector<std::uint32_t> rows{pending};
        if (mtp && depth) {
            const auto drafts = e.draft(session, pending, depth, min_p);
            rows.insert(rows.end(), drafts.tokens.begin(), drafts.tokens.end());
        }
        // Single-session verify/commit uses exactly the same loop as the serialized server.
        const auto r = e.verify_cohort({{session, rows.data(), static_cast<std::uint32_t>(rows.size()), position}}, true).at(0);
        if (!r.ok) { e.commit_cohort({0}); throw std::runtime_error(r.error); }
        std::uint32_t keep = 1;
        while (keep < rows.size() && rows[keep] == r.argmax[keep - 1]) ++keep;
        e.commit_cohort({keep});
        position += keep;
        pending = r.argmax[keep - 1];
        const auto vocab = r.logits.size() / rows.size();
        for (std::uint32_t j = 0; j < keep && out.ids.size() < 64; ++j) {
            out.ids.push_back(r.argmax[j]);
            out.logits.push_back(strix::sha256_bytes_fast(std::as_bytes(std::span(r.logits.data() + j * vocab, vocab))));
        }
    }
    return out;
}

std::vector<std::uint32_t> read_tokens(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open --prompt-tokens");
    const auto bytes = in.tellg();
    if (bytes < 0 || bytes % 4) throw std::runtime_error("prompt tokens must be uint32 LE");
    std::vector<std::uint32_t> tokens(static_cast<std::size_t>(bytes) / 4);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(tokens.data()), bytes)) throw std::runtime_error("short prompt tokens");
    for (auto token : tokens) if (token >= 248320) throw std::invalid_argument("prompt token out of vocabulary");
    return tokens;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        qp::Config c;
        c.decode = true;
        c.export_state = false;
        c.release_by_layer = false;
        std::string token_path, positions;
        std::uint32_t block = 256, chunk = 1024, depth = 7;
        float min_p = 0.7F;
        bool overlap = false, cache_vram = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 == argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--model") c.model_gguf = next();
            else if (a == "--mtp-sidecar") c.mtp_sidecar_gguf = next();
            else if (a == "--draft-vocab") c.draft_vocab = next();
            else if (a == "--tables") c.tables_dir = next();
            else if (a == "--rope-table") c.rope_table = next();
            else if (a == "--max-positions") c.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--block-rows") c.block_schedule = {static_cast<std::uint32_t>(std::stoul(next()))};
            else if (a == "--prefix-cache-block") block = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--prefill-chunk") chunk = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--mtp-depth") depth = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--min-p") min_p = std::stof(next());
            else if (a == "--prompt-tokens") token_path = next();
            else if (a == "--P") positions = next();
            else if (a == "--overlap") overlap = true;
            else if (a == "--cache-vram") cache_vram = true;
            else throw std::invalid_argument("unknown option " + a);
        }
        px::validate_block(block);
        if (!chunk || depth > 7 || c.model_gguf.empty() || c.tables_dir.empty() || c.rope_table.empty() || positions.empty())
            throw std::invalid_argument("required: --model --tables --rope-table --prompt-tokens --P; depth <= 7, chunk > 0");
        const auto a = read_tokens(token_path);
        if (a.size() < 3 || a.size() + 80 >= c.max_positions) throw std::invalid_argument("prompt must leave 80 decode positions");
        std::vector<std::uint64_t> checkpoints;
        std::istringstream list(positions);
        for (std::string word; std::getline(list, word, ',');) {
            const auto p = std::stoull(word);
            if (!p || p % block || p > px::lookup_limit(a.size())) throw std::invalid_argument("P must be a positive block multiple < n-1");
            checkpoints.push_back(p);
        }
        qp::Prefill e(c);
        if (overlap) e.enable_prefill_overlap(chunk);
        const bool mtp = !c.mtp_sidecar_gguf.empty();
        const auto x = e.create_session();
        const auto whole_ms = prefill(e, 0, a, 0, a.size() - 1, chunk, overlap);
        const auto reference = digest(e, 0, mtp, block);
        const auto reference_ids = decode(e, 0, a.back(), mtp, depth, min_p);
        const auto identity = strix::sha256_bytes(std::as_bytes(std::span("prefix-gate-v1", 14)));
        unsigned failures = 0;
        const auto report = [&](const char* name, std::uint64_t p, bool ok, double restore_ms, double suffix_ms) {
            failures += !ok;
            std::cout << (ok ? "PASS " : "FAIL ") << name << " P=" << p << " restore_ms=" << restore_ms
                      << " fresh_suffix_ms=" << suffix_ms << " fresh_whole_ms=" << whole_ms
                      << " rows_bytes=" << p * e.row_bytes() << " fixed_bytes=" << e.fixed_bytes() << '\n';
        };
        for (const auto p : checkpoints) {
            strix::PrefixStateCachePolicy policy;
            policy.host_byte_budget = p * e.row_bytes() + 2 * e.fixed_bytes() + 1024;
            policy.retention = strix::PrefixCacheRetention::lru;
            strix::PrefixStateCache store(policy, [cache_vram](std::uint64_t bytes) -> strix::PrefixCacheBuffer {
                void* ptr = nullptr;
                if (cache_vram) strix::cuda::require_vram(bytes, "prefix gate checkpoint");
                const auto rc = cache_vram ? cudaMalloc(&ptr, bytes) : cudaMallocHost(&ptr, bytes);
                if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
                return {std::shared_ptr<std::byte>(static_cast<std::byte*>(ptr), [cache_vram](std::byte* p) {
                    if (cache_vram) cudaFree(p); else cudaFreeHost(p);
                }), bytes};
            });
            const auto capture = [&](std::uint64_t position) {
                if (!px::capture(store, identity, a, block, position, mtp, e.row_bytes(), e.fixed_bytes(),
                    [&](auto first, auto count, void* dst) { e.export_rows(x, first, count, dst); },
                    [&](void* dst) { e.export_fixed(x, dst); })) throw std::runtime_error("checkpoint capture declined");
            };
            const auto restore = [&](const strix::PrefixCacheCheckpoint& cp) {
                std::vector<qp::RowSegment> segments;
                for (const auto& seg : cp.segments)
                    segments.push_back({seg->block_index * seg->block_rows, seg->block_rows, seg->payload.data.get()});
                const auto begin = Clock::now();
                e.import_state(0, cp.position, segments, cp.fixed.data.get());
                return ms(begin);
            };
            const auto earlier = p > block ? p - block : 0;
            if (earlier) { prefill(e, x, a, 0, earlier, chunk, overlap); capture(earlier); }
            prefill(e, x, a, earlier, p, chunk, overlap);
            capture(p);
            const auto cp = store.find_longest(identity, a, block, p);
            if (!cp || cp->position != p) throw std::runtime_error("checkpoint lookup missed");
            const auto suffix_ms = prefill(e, x, a, p, a.size() - 1, chunk, overlap);
            report("continued", p, digest(e, x, mtp, block) == reference, 0, suffix_ms);

            // Session 0 has decoded a prior prompt, so both stale rows and GDN parity differ.
            const auto restore_ms = restore(*cp);
            prefill(e, 0, a, p, a.size() - 1, chunk, overlap);
            const auto restored = digest(e, 0, mtp, block);
            const auto ids = decode(e, 0, a.back(), mtp, depth, min_p);
            report("different-session", p, restored == reference && ids == reference_ids, restore_ms, suffix_ms);

            auto b = a;
            // With MTP, token P is part of pair P-1. Preserve that boundary token;
            // a changed boundary must be refused by the host-store eligibility gate.
            const auto change_from = p + (mtp ? 1 : 0);
            for (std::size_t i = change_from; i < b.size(); ++i) b[i] = (b[i] + 1) % 248320;
            const auto b_whole_ms = prefill(e, x, b, 0, b.size() - 1, chunk, overlap);
            const auto b_reference = digest(e, x, mtp, block);
            const auto b_ids = decode(e, x, b.back(), mtp, depth, min_p);
            const auto b_restore_ms = restore(*cp);
            const auto b_suffix_ms = prefill(e, 0, b, p, b.size() - 1, chunk, overlap);
            const auto b_restored = digest(e, 0, mtp, block);
            const auto b_restored_ids = decode(e, 0, b.back(), mtp, depth, min_p);
            report("dropped-reasoning", p, b_restored == b_reference && b_restored_ids == b_ids, b_restore_ms, b_suffix_ms);
            std::cout << "timing dropped_whole_ms=" << b_whole_ms << '\n';

            // Also change the first suffix token. The real store must decline P for
            // MTP, then restore the earlier checkpoint (or prefill fresh at P == T).
            b[p] = (a[p] + 1) % 248320;
            const auto branch_whole_ms = prefill(e, x, b, 0, b.size() - 1, chunk, overlap);
            const auto branch_reference = digest(e, x, mtp, block);
            const auto branch_ids = decode(e, x, b.back(), mtp, depth, min_p);
            const auto branch_cp = store.find_longest(identity, b, block, px::lookup_limit(b.size()), true, {},
                [&](const strix::PrefixCacheCheckpoint& candidate) { return px::boundary_matches(candidate, b, mtp); });
            const auto restored_p = branch_cp ? branch_cp->position : 0;
            const auto branch_restore_ms = branch_cp ? restore(*branch_cp) : 0;
            const auto branch_suffix_ms = prefill(e, 0, b, restored_p, b.size() - 1, chunk, overlap);
            const auto branch_restored = digest(e, 0, mtp, block);
            const auto branch_restored_ids = decode(e, 0, b.back(), mtp, depth, min_p);
            report("changed-boundary", p, restored_p == (mtp ? earlier : p) && branch_restored == branch_reference &&
                branch_restored_ids == branch_ids, branch_restore_ms, branch_suffix_ms);
            std::cout << "timing branch_whole_ms=" << branch_whole_ms << " restored_P=" << restored_p << '\n';
        }
        e.destroy_session(x);
        return failures ? 1 : 0;
    } catch (const std::exception& ex) { std::cerr << "FAIL " << ex.what() << '\n'; return 1; }
}
