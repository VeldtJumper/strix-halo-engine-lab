#include "../apps/qwen_tiled_prompt_plan.hpp"
#include <iostream>
#include <memory>
#include <set>
using namespace strix;
using namespace strix::decode_cli;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
// Adapter lets the frozen old algorithm retain its backend/owner selection expressions.
using KnownWindowOutputPolicy=QwenPlanOutput;
struct Backend { QwenPlanOutput policy; QwenPlanOutput output_policy() const {return policy;} };
struct Owner {};
struct Incumbent {
    using Phase=QwenPlanPhase;
    struct Tail {std::uint32_t rows;std::shared_ptr<Backend> backend;std::shared_ptr<Owner> owner;};
    struct Tiled {std::uint32_t rows;std::shared_ptr<Backend> blocks;std::shared_ptr<Owner> block_owner;
        std::vector<Tail> tails;std::vector<std::uint32_t> ladder_tail_rows;} tiled_;
    struct Block {Backend* backend;Owner* owner;std::uint32_t rows;bool capture,final;};
    QwenTiledPlanConfig config;
    Phase phase_=Phase::blocks;std::size_t tail_=0;int balanced_=-1;
    std::uint64_t pos_=0,n_=0;bool small_=false,adopted_=false;
    Incumbent(const QwenTiledPlanConfig& c,std::uint64_t n):config(c),n_(n) {
        tiled_.rows=c.rows;tiled_.blocks=std::make_shared<Backend>(Backend{c.block.policy});
        tiled_.block_owner=std::make_shared<Owner>();tiled_.ladder_tail_rows=c.ladder;
        for(const auto& t:c.tails) tiled_.tails.push_back({t.rows,std::make_shared<Backend>(Backend{t.policy}),std::make_shared<Owner>()});
    }
    std::optional<Block> next() {
#include "qwen_tiled_incumbent.inc"
    }
};
std::uint64_t shapes=0,units=0;
void compare(const QwenTiledPlanConfig& c,std::uint64_t n,bool small=false) {
    Incumbent old(c,n);old.small_=small;QwenPlanCursor cursor;std::uint64_t pos=0;
    for(;;) {
        std::optional<Incumbent::Block> a;std::optional<QwenPlanUnit> b;
        std::string ae,be;
        try {a=old.next();} catch(const std::logic_error& e) {ae=e.what();}
        try {b=qwen_tiled_next(c,cursor,pos,n,small);} catch(const std::logic_error& e) {be=e.what();}
        require(ae==be,"missing owner/error differs");
        if(!ae.empty()) break;
        require(bool(a)==bool(b),"block/serial decision differs");
        require(cursor==QwenPlanCursor{old.phase_,old.tail_,old.balanced_},"continuation cursor differs");
        if(!a) break;
        require(b->begin==pos && b->end==pos+a->rows && b->capture==a->capture && b->final==a->final,"unit differs");
        const auto* backend=b->t_owner?old.tiled_.blocks.get():old.tiled_.tails[b->tail].backend.get();
        require(backend==a->backend,"owner/profile selection differs");
        ++units;pos=b->end;old.pos_=pos;
        if(pos==n) break;
    }
    ++shapes;
}
}
int main() {
try {
    for(const std::uint32_t t:{1024U,2048U,4096U,8192U})
    for(unsigned mask=0;mask<16;++mask)
    for(const std::uint32_t sched:{0U,4U,12U,16U,20U})
    for(bool start:{false,true})
    for(const auto policy:{QwenPlanOutput::state_only,QwenPlanOutput::terminal,QwenPlanOutput::all_rows}) {
        QwenTiledPlanConfig c;c.rows=t;c.block={t,policy};c.schedule=sched;c.tiled_start=start;
        unsigned bit=0;for(const auto r:{1024U,256U,64U,16U}) {if(mask&(1U<<bit)) {c.tails.push_back({r,policy});c.ladder.push_back(r);}++bit;}
        // Exhaust all residues through two T boundaries and cadence edges 4T/8T/16T.
        for(std::uint64_t n=1;n<=2U*t+1U;++n) compare(c,n);
        for(std::uint64_t multiple=3;multiple<=32;++multiple) for(int delta=-1;delta<=1;++delta) {
            compare(c,static_cast<std::uint64_t>(static_cast<std::int64_t>(multiple*t)+delta));
            for(const auto& tail:c.tails)
                compare(c,static_cast<std::uint64_t>(static_cast<std::int64_t>(multiple*t+tail.rows)+delta));
        }
        compare(c,1280);compare(c,3U*t+17U,true);
        auto mixed=c;
        for(std::size_t k=0;k<mixed.tails.size();++k) mixed.tails[k].policy=static_cast<QwenPlanOutput>(k%3U);
        compare(mixed,3U*t+1280U);mixed.ladder.clear();compare(mixed,3U*t+1280U);
        c.suppressed=true;compare(c,1280);compare(c,3U*t+17U);
        if(!c.tails.empty()) {c.suppressed=false;c.tails.erase(c.tails.begin());compare(c,3U*t+1280U);}
    }
    QwenTiledPlanConfig c;c.rows=2048;c.block={2048,QwenPlanOutput::terminal};c.schedule=12;
    c.ladder={1024,256,64,16};for(const auto r:c.ladder)c.tails.push_back({r,QwenPlanOutput::terminal});
    QwenPlanCursor cursor;const auto merged=qwen_tiled_next(c,cursor,0,1280);
    require(merged && merged->merged && merged->end==1280,"1280 regression");
    require(qwen_full_t_certificate(c,1280,1024).empty(),"internal merged checkpoint");
    const auto proof=qwen_full_t_certificate(c,8192,4096,&cursor);
    require(!proof.empty(),"full-T certificate");
    const auto suffix=qwen_tiled_next(c,cursor,4096,8192);
    require(suffix && !suffix->heads && suffix->begin==4096 && suffix->end==6144,"whole prompt continuation");
    require(proof==qwen_full_t_certificate(c,32768,4096),"invariant full-T family");
    c.schedule=20;require(qwen_full_t_certificate(c,5120,4096).empty(),"different n/balanced producer refusal");
    c.schedule=12;c.profile_pair=17;
    require(proof!=qwen_full_t_certificate(c,8192,4096),"profile certificate");
    c.profile_pair=0;c.block.execution=5;
    require(proof!=qwen_full_t_certificate(c,8192,4096),"producer identity certificate");
    c.block.execution=0;c.generations_per_row[0]=1;
    require(proof!=qwen_full_t_certificate(c,8192,4096),"generation certificate");
    c.generations_per_row[0]=0;c.priming=255;require(proof!=qwen_full_t_certificate(c,8192,4096),"priming certificate");
    for(bool complete:{false,true}) for(bool pending:{false,true}) for(bool certified:{false,true}) {
        require(qwen_capture_boundary(2048,8192,2048,complete,pending,certified)==(complete&&!pending&&certified),"capture readiness");
        require(!qwen_capture_boundary(8192,8192,2048,complete,pending,certified),"no final capture");
        require(!qwen_capture_boundary(1280,8192,2048,complete,pending,certified),"no nongrid capture");
    }
    // Suffix serial/window decisions, including odd-first and final handoff.
    for(std::uint64_t n=1;n<65;++n) for(std::uint64_t p=0;p<n;++p) for(bool windows:{false,true}) {
        const auto u=qwen_suffix_unit(p,n,windows);
        std::uint32_t r=1;
        if(windows && (n-p)%2U==0) {r=8;while(r>n-p) r/=2U;}
        require(u.begin==p && u.end==p+r && u.final==(p+r==n),"suffix serial/window plan");
    }
    for(const auto t:{2048U,8192U}) for(const auto n:{8192U,32768U}) {
        c.rows=t;c.block.rows=t;c.priming=255;
        const auto every=std::max(4096U,t);
        unsigned internal=0,selected=0,captures=0;
        QwenPlanCursor walk;std::uint64_t p=0;
        while(const auto u=qwen_tiled_next(c,walk,p,n)) {
            p=u->end;
            if(p<n && !qwen_full_t_certificate(c,n,p).empty()) {
                ++internal;
                if(p%every==0 || p+t>n-1U) {++selected;if(!u->async_priming)++captures;}
            }
            if(p==n) break;
        }
        require(captures==0 && internal==n/t-1U,"default pipeline capture count");
        require(selected==(t==2048U?(n==8192U?2U:8U):(n==8192U?0U:3U)),"default cadence count");
    }
    require(qwen_batch_staging_cap(0,27648,116380160,2304,20480,8192)==361767424,"staging includes future predictor and held state");
    require(qwen_batch_staging_cap(123,UINT64_MAX,UINT64_MAX,1,1,8192)==123,"explicit staging cap controls peak");
    bool overflow=false;
    try {(void)qwen_batch_staging_cap(0,UINT64_MAX,0,1,0,8192);} catch(const std::overflow_error&) {overflow=true;}
    require(overflow,"staging reservation fails closed on overflow");
    // Batch certification never changes the incumbent planner; execution evidence is separate.
    for(const auto t:{2048U,8192U}) {
        QwenTiledPlanConfig batch_plan;batch_plan.rows=t;batch_plan.block={t,QwenPlanOutput::terminal};
        batch_plan.schedule=12;batch_plan.tails={{1024,QwenPlanOutput::terminal},{256,QwenPlanOutput::terminal},
            {64,QwenPlanOutput::terminal},{16,QwenPlanOutput::terminal}};
        const std::uint64_t n=3U*t+1192U,p=2U*t;
        const auto v1=qwen_full_t_certificate(batch_plan,n,p);
        const auto v2=qwen_full_t_certificate_v2(batch_plan,n,p);
        require(!v2.empty() && v1!=v2 && v2!=qwen_full_t_certificate_v2(batch_plan,n,p,1),"route/version separation");
        std::vector<std::uint64_t> stripped{v1[0],v2[2],v2[3]};
        std::size_t at=4;
        for(unsigned k=0;k<2;++k,at+=8) {
            stripped.insert(stripped.end(),v2.begin()+static_cast<std::ptrdiff_t>(at),v2.begin()+static_cast<std::ptrdiff_t>(at+7));
            require(v2[at+7]==1,"per-unit execution certificate");
        }
        stripped.insert(stripped.end(),v2.begin()+static_cast<std::ptrdiff_t>(at),v2.end());
        require(stripped==v1,"v2 preserves the v1 producer fields");
        std::vector<QwenUnitExec> trace{{0,t,t,true,true},{t,p,t,true,true}};
        require(qwen_producer_trace_ok(batch_plan,n,p,trace),"actual solo full-T trace");
        for(unsigned bit=0;bit<7;++bit) {
            auto bad=trace;
            switch(bit) {case 0:bad[0].grouped=true;break;case 1:bad[0].small=true;break;
                case 2:bad[0].forced=true;break;case 3:bad[0].adopted=true;break;
                case 4:bad[0].own_owner=false;break;case 5:bad[0].end--;break;default:bad.erase(bad.begin());break;}
            require(!qwen_producer_trace_ok(batch_plan,n,p,bad),"trace must be observed and complete");
        }
        QwenBatchConfig cfg;
        auto owner=cfg;owner.group_owner=true;
        for(const bool groups:{false,true}) {
            owner.groups=groups;owner.group_rows=2U*t;
            require(!owner.allowed(12) && std::string_view(owner.decline(12))=="group-owner:prompt-batch-rows",
                "prompt-batch-rows owner declines even when grouping is off and names the input");
        }
        for(unsigned mask=0;mask<256;++mask) for(bool partial:{false,true}) for(bool nest:{false,true}) {
            auto unit=trace.back();auto facts=cfg;
            unit.grouped=mask&1;unit.small=mask&2;unit.forced=mask&4;unit.adopted=mask&8;
            unit.own_owner=!(mask&16);facts.group_owner=mask&32;facts.small=mask&64;
            const auto schedule=12U|((mask&128)?2U:0U);
            require(qwen_batch_capture_eligible(facts,schedule,unit,t,true,true,partial?128U:t,nest)==
                (!mask && (!partial || !nest)),"batch eligibility truth table");
        }
        // HTTP/MTP bridge: bit 2 is inert there, so the default 14 (and an explicit 14) plans, certifies and captures
        // as 12; only the CLI cursor's explicit bit 2 keeps the sched-bit-2 decline.
        require(qwen_bridge_plan_schedule("14")==12U && qwen_bridge_plan_schedule("12")==12U &&
            qwen_bridge_plan_schedule("13")==13U && qwen_bridge_plan_schedule("0x1e")==28U && qwen_bridge_plan_schedule(nullptr)==0U,
            "bridge plan schedule masks only bit 2");
        require(std::string_view(cfg.decline(14))=="sched-bit-2" && cfg.allowed(qwen_bridge_plan_schedule("14")) &&
            qwen_batch_capture_eligible(cfg,qwen_bridge_plan_schedule("14"),trace.back(),t,true,true,t,false),
            "HTTP batch certification admits the default schedule");
        require(!qwen_batch_capture_eligible(cfg,12,trace.back(),t,false,true,t,false),"unobserved prefix refused");
        require(!qwen_batch_capture_eligible(cfg,12,trace.back(),t,true,false,128,false),"foreign next owner refused");
        // Actual T=8192 nesting hazard; T=2048 has no nested row class.
        require(qwen_batch_capture_eligible(cfg,12,trace.back(),t,true,true,128,t==8192)==(t!=8192),"partial-next nesting hazard");
        for(std::uint64_t length=t+1U;length<2U*t+260U;++length)
            for(bool groups:{false,true}) for(bool cohort:{false,true}) for(bool windows:{false,true}) {
                cfg.groups=groups;cfg.cohort=cohort;cfg.windows=windows;
                auto suffix_cursor=qwen_full_t_continuation(batch_plan,length,t);
                bool expected=bool(suffix_cursor);auto pos=std::uint64_t(t);
                if(suffix_cursor) while(const auto u=qwen_prompt_next(batch_plan,*suffix_cursor,pos,length,windows)) {
                    if(u->kind==QwenPlanKind::tail_owner && groups) expected=false;
                    if(u->kind==QwenPlanKind::window && cohort) expected=false;
                    pos=u->end;
                }
                require(qwen_suffix_solo_class(batch_plan,length,t,cfg,false)==expected,"L0 exhaustive suffix classes");
                require(!qwen_suffix_solo_class(batch_plan,length,t,cfg,true),"nesting excludes L0");
            }
    }
    {
        QwenTiledPlanConfig plan;plan.rows=1024;plan.block={1024,QwenPlanOutput::terminal};plan.schedule=12;
        QwenBatchConfig batch;batch.groups=false;batch.cohort=false;batch.windows=false;
        const auto p=qwen_prefix_join_boundary(plan,11392,plan,10192,9692,1024,batch,false);
        require(p==9216,"join selects deepest common cold full-T boundary");
        auto changed=plan;changed.block.kernel=123;
        require(!qwen_prefix_join_boundary(plan,11392,changed,10192,9692,1024,batch,false),
            "join refuses changed arithmetic identity");
        require(!qwen_prefix_join_boundary(plan,11392,plan,10192,9692,1024,batch,true),
            "join refuses possible nested suffix");
        require(!qwen_prefix_join_boundary(plan,11392,plan,10192,9692,9216,batch,false),
            "join cannot wait for already passed work");
        require(!qwen_prefix_join_boundary(plan,11392,plan,10192,1023,0,batch,false),
            "join cannot split a full-T block");
        require(qwen_prefix_join_grid(8192,8192) && qwen_prefix_join_grid(1024,8192) && qwen_prefix_join_grid(1024,2048),
            "join accepts a cache grid dividing T, including the HTTP 1024-row checkpoint default");
        require(!qwen_prefix_join_grid(0,8192) && !qwen_prefix_join_grid(3072,8192) && !qwen_prefix_join_grid(1024,0),
            "join refuses a missing or non-dividing cache grid");
        QwenPrefixJoinWait wait{42,9216};
        require(wait.pending(42,true,true,1024) && wait.pending(42,true,true,9216),"wait keeps incomplete producer");
        require(!wait.pending(43,true,true,0),"ABA slot generation releases waiter");
        require(!wait.pending(42,false,true,1024) && !wait.pending(42,true,false,1024),"producer cancellation/completion releases waiter");
        require(!wait.pending(42,true,true,10240),"passed boundary releases waiter");
    }
    std::cout<<"planner passed: "<<shapes<<" shapes, "<<units<<" units; certificates and capture eligibility passed\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
