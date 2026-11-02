#include "strix/known_prompt_chunks.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
using namespace strix;
namespace {
void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
struct Call {
    std::uint64_t begin,end;
    std::uint32_t owner;
    bool headless,deferred;
    bool operator==(const Call&) const = default;
};
class Backend final : public KnownPromptWindowBackend,public SerialGreedyStepBackend {
    std::uint32_t rows_;
    std::vector<Call>& calls_;
    std::vector<float> record(std::span<const std::uint32_t> tokens,std::uint64_t p,bool headless,bool deferred) {
        calls_.push_back({p,p+tokens.size(),rows_,headless,deferred});
        return std::vector<float>(16,1.0F);
    }
public:
    Backend(std::uint32_t rows,std::vector<Call>& calls):rows_(rows),calls_(calls) {}
    bool supports_tiled_prefill() const noexcept override {return true;}
    std::uint32_t logical_rows() const noexcept override {return rows_;}
    KnownWindowOutputPolicy output_policy() const noexcept override {return KnownWindowOutputPolicy::terminal_row;}
    std::vector<float> execute(std::uint32_t token,std::uint64_t p) override {return record(std::span(&token,1),p,false,false);}
    std::vector<float> execute_window(std::array<std::uint32_t,2>,std::uint64_t) override {throw std::logic_error("no pairs");}
    std::vector<float> execute_rows(std::span<const std::uint32_t> t,std::uint64_t p) override {return record(t,p,false,false);}
    std::vector<float> execute_rows_partial(std::span<const std::uint32_t> t,std::uint64_t p) override {return record(t,p,false,false);}
    void execute_state_rows(std::span<const std::uint32_t> t,std::uint64_t p) override {(void)record(t,p,true,false);}
    void execute_state_rows_partial(std::span<const std::uint32_t> t,std::uint64_t p,bool d) override {(void)record(t,p,true,d);}
    void execute_state_rows_deferred(std::span<const std::uint32_t> t,std::uint64_t p) override {(void)record(t,p,true,true);}
    void finish_deferred() override {}
};
}
int main(int argc,char** argv) {
    require(argc==2,"schedule argument required");
    ::setenv("STRIX_QPF2K_SCHED",argv[1],1);
    const auto schedule=std::strtoul(argv[1],nullptr,0);
    for(const std::size_t n:{1280U,3328U,5376U,6145U,8193U}) {
        const std::vector<std::uint32_t> prompt(n,1);
        KnownPromptChunkRequest r;
        r.prompt_tokens=prompt;r.vocab_size=16;r.context_capacity=16384;r.eos_token_id=15;
        r.logical_rows=2048;r.tiled_prefill=true;r.output_policy=KnownWindowOutputPolicy::terminal_row;
        r.schedule_extensions=true;r.ladder_tail_rows={1024,256};
        std::vector<Call> calls;
        Backend blocks(2048,calls),tail1(1024,calls),tail2(256,calls),serial(1,calls);
        const auto install=[&](KnownPromptChunkCursor& c) {c.set_tail_backend(tail1);c.set_tail_backend(tail2);};
        KnownPromptChunkCursor cold(r,blocks,serial);install(cold);
        const auto expected=cold.cold_plan();
        require(calls.empty(),"cold plan performs zero backend calls");
        std::vector<KnownPromptChunkUnit> actual;
        while(cold.advance()) actual.push_back(cold.last_unit());
        require(actual==expected,"execution equals whole-prompt preflight");
        require(calls.size()==actual.size(),"one receipt for each physical execution");
        for(std::size_t i=0;i<calls.size();++i) {
            const auto& u=actual[i];
            require(calls[i]==Call{u.begin,u.end,u.owner_rows,u.headless,u.deferred},"receipt describes actual backend call");
        }
        if(schedule==4 && n==1280) require(actual.size()==1 && actual[0].merged &&
            actual[0].owner_rows==2048 && actual[0].end==1280,"1280 rows merge on T owner");
        const auto cold_calls=calls;calls.clear();
        // A cache observer uses exactly the request used when no cache is configured.
        KnownPromptChunkCursor observed(r,blocks,serial);install(observed);
        require(observed.cold_plan()==expected && calls.empty(),"passive cache planner leaves cold plan unchanged");
        std::vector<KnownPromptChunkUnit> observed_trace;
        while(observed.advance()) observed_trace.push_back(observed.last_unit());
        require(observed_trace==actual && calls==cold_calls,"cache miss has identical units and physical calls");
        for(const auto& u:expected) {
            KnownPromptChunkCursor warm(r,blocks,serial);install(warm);
            const bool eligible=u.end<n && known_prompt_full_t_prefix(expected,u.end,2048);
            require(warm.can_resume_at(u.end)==eligible,"resume accepts only solo full-T prefix");
            require(!warm.can_resume_at(u.begin+1),"interior boundary refused");
            const auto count=calls.size();
            if(eligible) {
                warm.resume_at(u.end);
                require(calls.size()==count,"resume preflight has zero writes");
                std::vector<KnownPromptChunkUnit> suffix,wanted;
                while(warm.advance()) suffix.push_back(warm.last_unit());
                for(const auto& v:expected) if(v.begin>=u.end) wanted.push_back(v);
                require(suffix==wanted,"resumed suffix is whole-prompt cold suffix");
            } else {
                bool refused=false;
                try {warm.resume_at(u.end);} catch(const std::invalid_argument&) {refused=true;}
                require(refused && calls.size()==count,"unsupported continuation refuses before writes");
            }
        }
    }
    if(schedule&2U) {
        // A CLI prefix-state cache with a defaulted schedule drops bit 2 (qpf2k::cli_sched_defers): the same units
        // without the deferral, so the full-T boundaries stay save/resume points. With bit 2 they are not.
        const std::vector<std::uint32_t> prompt(8193,1);
        KnownPromptChunkRequest r;
        r.prompt_tokens=prompt;r.vocab_size=16;r.context_capacity=16384;r.eos_token_id=15;
        r.logical_rows=2048;r.tiled_prefill=true;r.output_policy=KnownWindowOutputPolicy::terminal_row;
        r.schedule_extensions=true;r.ladder_tail_rows={1024,256};
        std::vector<Call> calls;
        Backend blocks(2048,calls),tail1(1024,calls),tail2(256,calls),serial(1,calls);
        const auto plan=[&](bool defer) {
            r.schedule_defer=defer;
            KnownPromptChunkCursor c(r,blocks,serial);c.set_tail_backend(tail1);c.set_tail_backend(tail2);
            return c.cold_plan();
        };
        const auto deferred=plan(true),kept=plan(false);
        require(std::any_of(deferred.begin(),deferred.end(),[](const auto& u){return u.deferred;}),"bit 2 defers blocks");
        require(std::none_of(kept.begin(),kept.end(),[](const auto& u){return u.deferred;}),"schedule_defer=false defers nothing");
        require(deferred.size()==kept.size(),"dropping bit 2 keeps the units");
        for(std::size_t i=0;i<kept.size();++i) {auto u=deferred[i];u.deferred=false;require(u==kept[i],"dropping bit 2 keeps each unit");}
        for(const std::uint64_t b:{2048U,4096U,6144U}) {
            require(!known_prompt_full_t_prefix(deferred,b,2048),"deferred blocks are no cache boundaries");
            require(known_prompt_full_t_prefix(kept,b,2048),"without bit 2 every full-T boundary is a cache boundary");
        }
        r.schedule_defer=false;
        KnownPromptChunkCursor run(r,blocks,serial);run.set_tail_backend(tail1);run.set_tail_backend(tail2);
        require(run.can_resume_at(4096),"a cached run without bit 2 can resume at a full-T boundary");
        while(run.advance()) {}
        require(std::none_of(calls.begin(),calls.end(),[](const Call& c){return c.deferred;}),"no backend call is deferred");
    }
    std::cout<<"serial prompt plan PASS schedule "<<schedule<<'\n';
}
