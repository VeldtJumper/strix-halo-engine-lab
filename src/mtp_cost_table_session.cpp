#include "strix/mtp_cost_table_session.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

namespace strix {
std::string mtp_cost_table_configuration(const MtpCostTableRoute& r) {
    return "qwen-decode-cost-v3;tiled="+std::to_string(r.tiled)+";http="+std::to_string(r.http)+
        ";batched="+std::to_string(r.batched)+";batch="+(r.batch_explicit?std::to_string(r.batch_cap):std::string("auto"))+
        ";cohort="+std::to_string(r.cohort)+";kernel_plan="+std::to_string(r.kernel_plan)+
        ";execution="+std::to_string(r.execution)+";profiles="+std::to_string(r.profiles)+
        ";optimization="+r.optimization+";cli="+r.cli;
}
MtpCostTableSession::MtpCostTableSession(std::filesystem::path directory,CostTableKey key,CostTableOptions options,
    std::vector<std::uint32_t> seedable_rows)
    :directory_(std::move(directory)),key_(std::move(key)),options_(options),seedable_(std::move(seedable_rows)) {}
std::string MtpCostTableSession::entry(std::uint32_t rows) {return "r"+std::to_string(rows);}
std::vector<std::uint32_t> MtpCostTableSession::seedable_rows(const std::vector<std::uint32_t>& owner_rows,
    const std::vector<bool>& admitted) {
    if(owner_rows.empty() || owner_rows.size()!=admitted.size() ||
        std::find(admitted.begin(),admitted.end(),false)!=admitted.end()) return {};
    std::vector<std::uint32_t> out{1U};
    out.insert(out.end(),owner_rows.begin(),owner_rows.end());
    return out;
}
CostTableCache& MtpCostTableSession::table(std::uint32_t bucket) {
    auto& t=tables_[bucket];
    if(!t) {
        auto k=key_;k.context_bucket=bucket;
        t=std::make_unique<CostTableCache>(directory_,std::move(k),options_);
        const auto status=t->load();
        std::string recheck;
        for(const auto& n:t->recheck_entries()) recheck+=(recheck.empty()?"":",")+n;
        std::fprintf(stderr,"cost-table: %s bucket %u %s entries=%zu recheck=%s file=%s\n",key_.engine.c_str(),bucket,
            cost_table_load_name(status),t->loaded_entries(),recheck.empty()?"-":recheck.c_str(),
            t->file().filename().string().c_str());
    }
    return *t;
}
void MtpCostTableSession::seed(MtpMinPWindowController& controller,std::uint32_t bucket) {
    bucket_=bucket;state_=State::warm;
    auto& t=table(bucket);
    std::vector<MtpMinPWindowController::CostPrior> priors;
    for(const auto rows:seedable_) if(const auto e=t.warm(entry(rows))) priors.push_back({rows,e->ms,e->samples});
    controller.seed_costs(priors);
}
void MtpCostTableSession::record(const MtpMinPWindowController& controller) {
    auto& t=table(bucket_);
    for(const auto& m:controller.measured_costs(3U))
        if(std::find(seedable_.begin(),seedable_.end(),m.rows)!=seedable_.end()) t.record(entry(m.rows),{m.ms,m.samples});
}
void MtpCostTableSession::prepare(MtpMinPWindowController& controller,std::uint64_t position,bool sampled) {
    if(state_==State::cold) return;
    if(state_==State::unresolved) {
        // Only a controller that has not decided or measured anything yet this session takes a prior.
        if(sampled || !controller.fresh()) {state_=State::cold;return;}
        seed(controller,cost_table_context_bucket(position));
        return;
    }
    if(sampled) { // not expected (the sampler is fixed per request); never let a sampled cycle use a prior
        controller.reset_costs();state_=State::cold;no_write_=true;return;
    }
    const auto bucket=cost_table_context_bucket(position);
    if(bucket==bucket_) return;
    if(!no_write_) record(controller);
    controller.reset_costs();
    seed(controller,bucket);
}
void MtpCostTableSession::flush(const MtpMinPWindowController& controller) noexcept {
    try {
        if(state_==State::warm && !no_write_) record(controller);
        for(auto& [bucket,t]:tables_) if(t->dirty()) (void)t->store();
    } catch(...) {} // a persisted prior is optional: never fail a request over it
    state_=State::unresolved;no_write_=false;
}
} // namespace strix
