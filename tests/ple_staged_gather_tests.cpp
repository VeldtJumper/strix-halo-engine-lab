#include "strix/ple_staged_gather.hpp"

#include <array>
#include <iostream>

namespace {

int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

constexpr std::size_t rows = 4;
constexpr std::uint32_t heads = 2;
using Hashes = std::array<strix::PleHashResult, rows>;
using Ids = std::array<std::uint32_t, rows * heads>;

// CPU copy oracle for the caller's deferred gather: one distinct byte per head.
using Bytes = std::array<std::uint8_t, rows * heads>;
void gather(const Hashes& hashes, std::span<const std::size_t> deferred, Bytes& bytes) {
    for (const auto r : deferred)
        for (std::uint32_t h = 0; h < heads; ++h)
            bytes[r * heads + h] = static_cast<std::uint8_t>(hashes[r].row_ids[h]);
}

}  // namespace

int main() {
    Hashes hashes{};
    Ids ids{};
    for (std::size_t r = 0; r < rows; ++r) {
        hashes[r].row_count = heads;
        for (std::uint32_t h = 0; h < heads; ++h) {
            ids[r * heads + h] = static_cast<std::uint32_t>(r * heads + h + 1U);
            hashes[r].row_ids[h] = ids[r * heads + h];
        }
    }
    const std::vector<std::size_t> all{0, 1, 2, 3}, tail{1, 2, 3};
    using strix::ple_detail::accept_staged_gather;
    for (const bool consumed : {false, true}) {
        auto deferred = consumed ? tail : all;
        Bytes bytes{};
        gather(hashes, all, bytes);  // completed staging worker
        if (consumed) gather(hashes, std::array<std::size_t, 1>{0}, bytes);
        expect(accept_staged_gather(hashes, heads, ids, consumed, deferred) == rows,
            "accepted receipt counts the entire staged block, including consumed row 0");
        expect(deferred.empty(), "accepted block clears all remaining deferred entries");
        gather(hashes, deferred, bytes);
        Bytes expected{};
        gather(hashes, all, expected);
        expect(bytes == expected, "staged hit preserves inline gather bytes");

        // Every head of every row must match, including independently consumed row 0.
        for (std::size_t i = 0; i < ids.size(); ++i) {
            auto wrong = ids;
            ++wrong[i];
            deferred = consumed ? tail : all;
            const auto original = deferred;
            expect(accept_staged_gather(hashes, heads, wrong, consumed, deferred) == 0,
                "any mismatching row/head rejects staging");
            expect(deferred == original, "hash mismatch retains the inline fallback work list");
            Bytes fallback{};
            if (consumed) gather(hashes, std::array<std::size_t, 1>{0}, fallback);
            gather(hashes, deferred, fallback);
            expect(fallback == expected, "rejected stage falls back to identical copied bytes");
        }
        for (std::size_t r = 0; r < rows; ++r) {
            auto wrong = hashes;
            --wrong[r].row_count;
            deferred = consumed ? tail : all;
            const auto original = deferred;
            expect(accept_staged_gather(wrong, heads, ids, consumed, deferred) == 0 && deferred == original,
                "any row head-count mismatch rejects without changing fallback");
        }
    }

    auto deferred = tail;
    expect(accept_staged_gather(hashes, heads, ids, false, deferred) == 0 && deferred == tail,
        "row 0 cannot be omitted without explicit successful consumption");
    for (const auto& invalid : {std::vector<std::size_t>{0, 2, 3},
             std::vector<std::size_t>{1, 1, 3}, std::vector<std::size_t>{1, 2}, all}) {
        deferred = invalid;
        expect(accept_staged_gather(hashes, heads, ids, true, deferred) == 0 && deferred == invalid,
            "consumption permits only the ordered tail, not arbitrary missing or duplicate rows");
    }
    deferred = tail;
    expect(accept_staged_gather(hashes, heads, std::span(ids).first(ids.size() - 1U), true, deferred) == 0 &&
        deferred == tail, "short staged extent rejects without changing fallback");
    expect(accept_staged_gather(hashes, 0, ids, true, deferred) == 0 && deferred == tail,
        "invalid head geometry rejects without changing fallback");
    expect(accept_staged_gather(hashes, strix::PleHashContract::maximum_heads + 1U, ids, true, deferred) == 0 &&
        deferred == tail, "oversized head geometry rejects without changing fallback");
    deferred = {1};
    expect(accept_staged_gather(std::span(hashes).first(2), heads, ids, true, deferred) == 2 && deferred.empty(),
        "a smaller launch may accept a matching prefix of staged rows");
    deferred.clear();
    expect(accept_staged_gather({}, heads, ids, true, deferred) == 0,
        "empty launch cannot be accepted");

    using strix::ple_detail::StageSource;
    using strix::ple_detail::accept_serve_staged_gather;
    using strix::ple_detail::ServeStageAction;
    using strix::ple_detail::serve_stage_action;
    using strix::ple_detail::serve_stage_workers;
    const int runtime=0,manifest=0,other=0;
    const StageSource source{17,rows,128,9,&runtime,&manifest,4096};
    for(unsigned hardware=0;hardware<=256U;++hardware) {
        const auto workers=serve_stage_workers(hardware);
        expect(workers<=8U,"serve stage uses at most eight gather workers");
        if(hardware<2U) expect(!workers,"unknown or single-thread hosts decline serve overlap");
        else {
            const auto remaining=hardware-workers;
            const auto inline_workers=remaining<8U?remaining:8U;
            expect(workers && inline_workers && workers+inline_workers<=hardware,
                "stage coordinator and inline caller are included in the hardware worker budget");
        }
    }
    for(std::uint32_t slot=0;slot<2U;++slot) {
        // Pre-launch lookahead fills the slot after the current launch's slot.
        // Keeping it active prevents the next pre-launch hook replacing it.
        const auto staged_slot=slot;
        const auto inline_slot=slot^1U;
        auto predecessor=source;predecessor.first+=rows;
        expect(serve_stage_action(source,predecessor,staged_slot,inline_slot)==ServeStageAction::keep,
            "predecessor leaves the successor's other-slot gather running");
        expect(serve_stage_action(source,source,staged_slot,inline_slot)==ServeStageAction::keep,
            "equal segment positions and sources do not steal another slot's stage");
        expect(serve_stage_action(source,source,staged_slot,inline_slot^1U)==ServeStageAction::consume,
            "successor joins its selected slot before validating every staged row");
        expect(serve_stage_action(source,predecessor,staged_slot,inline_slot^1U)==ServeStageAction::discard,
            "selected-slot mismatch must join before inline fallback reuses storage");
        // Distinct sessions can have identical positions and table sources.
        auto different=hashes;++different[0].row_ids[0];
        deferred=all;
        expect(accept_serve_staged_gather(source,source,different,heads,ids,false,deferred)==0 && deferred==all,
            "selected-slot source match still rejects another session's row ids after joining");
    }
    for(bool consumed:{false,true}) {
        deferred=consumed?tail:all;
        expect(accept_serve_staged_gather(source,source,hashes,heads,ids,consumed,deferred)==rows && deferred.empty(),
            "matching serve positions, table and row ids accept");
        for(unsigned mismatch=0;mismatch<8;++mismatch) {
            auto wrong=source;
            switch(mismatch) {
            case 0: ++wrong.first; break;
            case 1: --wrong.rows; break;
            case 2: ++wrong.row_bytes; break;
            case 3: ++wrong.generation; break;
            case 4: wrong.runtime=&other; break;
            case 5: wrong.manifest=&other; break;
            case 6: ++wrong.address; break;
            case 7: wrong.runtime=nullptr; break;
            }
            deferred=consumed?tail:all;
            const auto original=deferred;
            expect(serve_stage_action(wrong,source,0,1)==ServeStageAction::keep &&
                serve_stage_action(wrong,source,0,0)==ServeStageAction::discard,
                "all source mismatches preserve other-slot work but join selected-slot work");
            expect(accept_serve_staged_gather(wrong,source,hashes,heads,ids,consumed,deferred)==0 && deferred==original,
                "any serve source mismatch retains inline work");
        }
        for(std::size_t i=0;i<ids.size();++i) {
            auto wrong=ids;++wrong[i];
            deferred=consumed?tail:all;
            const auto original=deferred;
            expect(accept_serve_staged_gather(source,source,hashes,heads,wrong,consumed,deferred)==0 && deferred==original,
                "serve staging compares every row id, including consumed row zero");
            Bytes fallback{},expected{};
            if(consumed) gather(hashes,std::array<std::size_t,1>{0},fallback);
            gather(hashes,deferred,fallback);gather(hashes,all,expected);
            expect(fallback==expected,"serve rejection gathers the same bytes inline");
        }
    }
    auto wrong=source;++wrong.rows;
    deferred=all;
    expect(accept_serve_staged_gather(wrong,wrong,hashes,heads,ids,false,deferred)==0 && deferred==all,
        "serve hash count must match requested positions");
    if (!failures) std::cout << "PLE staged gather acceptance and fallback tests passed\n";
    return failures ? 1 : 0;
}
