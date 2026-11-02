// CPU-only decode contract tests: no CUDA runtime, device or model data.
#include "glm/rtxpf/producer/runtime.hpp"
#include "glm/rtxpf/producer/session.hpp"
#include "glm/rtxpf/producer/head_select.hpp"
#include "glm/rtxpf/producer/expert_tier.hpp"
#include "glm/rtxpf/producer/decode_timeline.hpp"
#include "glm/rtxpf/c1_schedule.hpp"
#include "glm/rtxpf/producer/schedule.hpp"
#include "glm/rtxpf/producer/profile.hpp"
#include "glm/rtxpf/producer/generate_request.hpp"
#include "glm/rtxpf/producer/shards.hpp"
#include "glm/rtxpf/producer/state_slices.hpp"
#include "glm/rtxpf/producer/unit_file.hpp"
#include "glm/rtxpf/producer/tierstream_binding.hpp"
#include "glm/rtxpf/producer/unit_auth.hpp"
#include <winioctl.h>
#include <algorithm>
#include <functional>
#include <cmath>
#include <tuple>
#include <numeric>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <chrono>
using namespace strix::glm::rtxpf::producer;
using strix::serve::Json;
unsigned checks=0;
void check(bool b,const char* what){++checks;if(!b)throw std::runtime_error(std::string("decode host check failed: ")+what);}
template<class F> void rejects(F f,const char* what){bool threw=false;try{f();}catch(const std::exception&){threw=true;}check(threw,what);}
template<class F> int rejects_code(F f){try{f();}catch(const RequestError& e){return e.code;}catch(const std::exception&){return -1;}return 0;}

void session_tests(){
    SessionFrontier s(16);std::vector<std::uint32_t> q{5,6,7};
    rejects([&]{(void)s.position();},"closed session has no position");
    s.begin(q);check(s.frontier()==2&&s.pending()==7&&s.position()==2,"pending token convention P=N-1");
    s.commit(9);check(s.frontier()==3&&s.pending()==9&&s.tokens().size()==4,"commit appends pending");
    // After t steps the layout equals a fresh prefill of prompt+generated[:t].
    SessionFrontier fresh(16);auto seq=q;seq.push_back(9);fresh.begin(seq);check(fresh.frontier()==s.frontier()&&fresh.pending()==s.pending(),"incremental == fresh frontier");
    rejects([&]{s.commit(vocab_size);},"token outside vocabulary");
    s.fail();check(!s.can_step(),"failed session cannot step");rejects([&]{(void)s.position();},"failed session position");s.reset();check(!s.open(),"reset closes");
    rejects([&]{SessionFrontier::validate_prompt(std::vector<std::uint32_t>{1},16);},"single-token prompt");
    rejects([&]{SessionFrontier::validate_prompt(std::vector<std::uint32_t>(17,1),16);},"prompt above capacity");
    SessionFrontier::validate_generation(10,6,16);rejects([&]{SessionFrontier::validate_generation(10,7,16);},"prompt+max_new above capacity");rejects([&]{SessionFrontier::validate_generation(10,0,16);},"zero max_new");
    // Full session: the final step consumes position capacity-2; a fresh prefill of the result fits.
    SessionFrontier f(8);f.begin(std::vector<std::uint32_t>{1,2});unsigned steps=0;while(f.can_step()){f.commit(3);++steps;}check(steps==6&&f.tokens().size()==8,"steps up to capacity");
    // Byte accounting (design section 1 tables).
    check(session_state_bytes(1024).total==173080576ULL&&session_state_bytes(8192).total==316211200ULL&&session_state_bytes(32768).total==806944768ULL,"session state bytes");
    check(session_state_bytes(1024).kda==152633344ULL&&session_state_bytes(8192).dsa==163577856ULL,"KDA/DSA split");
    check(session_state_bytes(1).replay==170459136ULL,"replay checkpoint bytes");
    check(2*export_rank_bytes(1024)==193527808ULL&&2*export_rank_bytes(8192)==479789056ULL&&2*export_rank_bytes(32768)==1461256192ULL,"two-rank export bytes");
    unsigned kda=0,dsa=0;std::vector<bool> ks(34),ds(12);
    for(unsigned l=0;l<46;++l){if(is_kda_layer(l)){auto k=kda_slot(l);check(k<34&&!ks[k],"unique KDA slot");ks[k]=true;++kda;}else{auto d=dsa_slot(l);check(d<12&&!ds[d],"unique DSA slot");ds[d]=true;++dsa;}}
    check(kda==34&&dsa==12&&dsa_slot(45)==11,"34 KDA + 12 DSA incl MTP");
    // KDA rank slicing of the resident state (export path).
    std::vector<std::byte> st(64*65536),cv(64*4608);for(unsigned h=0;h<64;++h)st[h*65536]=std::byte(h);
    check(kda_tp_state(st,1)[0]==std::byte(32)&&kda_tp_conv(cv,0).size()==32*4608,"KDA TP slices");
    // DSA pool boundaries for one-row appends.
    for(unsigned first:{0u,1u,2u,3u,4u,7u,2047u,2048u,2051u,8191u}){check(dsa_completes_pool(first)==((first+1)%4==0),"pool completion");check(dsa_scored_pools(first)==(first+1)/4,"scored pools");check(dsa_attended_tokens(first)<=dsa_attended_bound,"attended bound");}
    check(dsa_attended_tokens(2046)==2047&&dsa_attended_tokens(8190)==2048+3,"attention window below/above threshold");
}
// B32 arithmetic model: a block's result depends on its composition through
// half rounding (as the WMMA operands do). Replay from the block checkpoint
// reproduces fresh prefill; one-row blocks do not.
// Round to an 11-bit significand (FP16 precision model, ties to even).
float h(float x){if(x==0)return x;int e=0;double m=std::frexp(double(x),&e);return float(std::ldexp(std::nearbyint(std::ldexp(m,11)),e-11));}
Json& mut(Json& j,const std::string& k){for(auto& [key,v]:j.as_object())if(key==k)return v;throw std::runtime_error("missing test member");}
struct Toy {float s=0;};
float block_out(float s0,const std::vector<float>& rows,unsigned t){float acc=0;for(unsigned u=0;u<=t;++u)acc=h(acc+h(rows[u]*0.3f));return h(s0*0.1f+acc);}
float block_state(float s0,const std::vector<float>& rows){float acc=0;for(auto x:rows)acc=h(acc+h(x*0.3f));return s0*0.5f+acc;}
float row(unsigned i){return 1.f+float(i%7)*0.137f+float(i%3)*1.e-3f;}
void kda_schedule_tests(){
    for(unsigned P:{0u,1u,30u,31u,32u,33u,62u,63u,64u,65u,96u}){
        check(kda_block_start(P)%32==0&&kda_block_start(P)+kda_open_rows(P)==P&&kda_replay_rows(P)==kda_open_rows(P)+1,"open-block arithmetic");
        check(kda_promotes(P)==(P%32==31),"promotion at the block's 32nd row");
    }
    bool naive_differs=false;
    for(unsigned N:{2u,31u,32u,33u,34u,63u,64u,65u,66u}){
        // Fresh prefill of rows [0,P): absolute B32 blocks; checkpoint = start of the open block.
        unsigned P=N-1;float s=0;for(unsigned b=0;b+32<=P;b+=32){std::vector<float> r;for(unsigned i=b;i<b+32;++i)r.push_back(row(i));s=block_state(s,r);}
        float ck=s;std::vector<float> tail;for(unsigned i=kda_block_start(P);i<P;++i)tail.push_back(row(i));
        float naive=0;for(unsigned i=0;i<P;++i)naive=block_state(naive,{row(i)});
        // Incremental: ten steps, each replaying the open block from its checkpoint.
        for(unsigned step=0;step<10;++step,++P){
            tail.push_back(row(P));check(tail.size()==kda_replay_rows(P),"replay rows");
            float out=block_out(ck,tail,unsigned(tail.size()-1)),now=block_state(ck,tail);
            // Fresh reference over rows [0,P]: complete blocks, then the partial block containing P.
            float f=0;std::vector<float> r;unsigned b=0;for(;b+32<=P+1;b+=32){r.clear();for(unsigned i=b;i<b+32;++i)r.push_back(row(i));if(b+32<=P)f=block_state(f,r);}
            float fs=f;r.clear();for(unsigned i=kda_block_start(P);i<=P;++i)r.push_back(row(i));float fo=block_out(fs,r,unsigned(r.size()-1));float fstate=block_state(fs,r);
            check(out==fo&&now==fstate,"replay output/state equal fresh prefill");
            naive=block_state(naive,{row(P)});if(naive!=fstate)naive_differs=true;
            if(kda_promotes(P)){ck=now;tail.clear();}
        }
    }
    check(naive_differs,"one-row recurrence diverges from B32 prefill (exactness trap)");
}
void head_tests(){
    std::vector<float> x{1.f,3.f,3.f,-1.f};check(head_argmax(x).token==1,"tie selects lower ID");
    std::vector<float> z{-0.f,0.f,-1.f};check(head_argmax(z).token==0,"+0 == -0 tie selects lower ID");
    std::vector<float> z2{-1.f,0.f,-0.f};check(head_argmax(z2).token==1,"signed zero tie");
    std::vector<float> n{1.f,std::numeric_limits<float>::quiet_NaN(),2.f};auto c=head_argmax(n);check(c.token==2&&c.nonfinite==1,"NaN never wins");rejects([&]{accept_choice(c);},"nonfinite rejects token");
    std::vector<float> inf{1.f,std::numeric_limits<float>::infinity()};rejects([&]{accept_choice(head_argmax(inf));},"infinite logit rejects");
    rejects([&]{accept_choice(head_argmax(std::vector<float>{}));},"empty logits");
    std::vector<float> neg(154880,-3.f);neg[154879]=-2.f;neg[7]=-2.f;check(accept_choice(head_argmax(neg))==7,"full vocabulary lowest maximal ID");
    check(head_better(1.f,5,1.f,no_token)&&!head_better(1.f,no_token,0.f,3),"sentinel order");
}
// Synthetic small-geometry routed layers, packed exactly like Shards::pack.
struct Synthetic {ExpertGeometry g{256,128,288,3,4};Json tensors=Json::object();std::map<std::string,std::vector<std::byte>> blobs;};
unsigned rate_of(unsigned l,unsigned e,unsigned m){return 3+(l+e+m)%4;}
Synthetic synthetic(){
    Synthetic s;
    struct T{std::string dtype;Json shape;std::uint64_t bytes;std::uint32_t fill;};std::map<std::string,T> all;
    auto arr=[](std::initializer_list<unsigned> d){auto a=Json::array();for(auto v:d)a.push(int(v));return a;};
    for(unsigned l=3;l<=4;++l)for(unsigned e=0;e<288;++e)for(unsigned m=0;m<3;++m){auto b=ExpertCatalog::prefix(l,e)+matrix_names[m]+".";unsigned in=s.g.in(m),out=s.g.out(m),r=rate_of(l,e,m);std::uint32_t tag=(l<<24)|(e<<8)|(m<<4);
        all[b+"trellis"]={"I16",arr({in/16,out/16,16*r}),trellis_bytes(in,out,r),tag|1};all[b+"suh"]={"F16",arr({in}),std::uint64_t(in)*2,0x3c003c00};all[b+"svh"]={"F16",arr({out}),std::uint64_t(out)*2,0x40004000};all[b+"mul1"]={"I32",Json::array(),4,0x83dcd12d};}
    for(auto& [name,t]:all){auto l=name.find(".layers.4.")!=std::string::npos?4u:3u;auto& blob=s.blobs[ExpertCatalog::blob_name(l)];auto off=align_up(blob.size(),256);blob.resize(off+t.bytes);
        for(std::uint64_t i=0;i<t.bytes;++i)blob[off+i]=std::byte((t.fill>>((i%4)*8))&255);if(t.dtype=="I16")for(std::uint64_t i=0;i<t.bytes;i+=4)blob[off+i]=std::byte((i/4)&255);
        auto v=Json::object();v.set("blob",ExpertCatalog::blob_name(l));v.set("offset",off);v.set("bytes",t.bytes);v.set("dtype",t.dtype);v.set("shape",t.shape);s.tensors.set(name,v);}
    return s;
}
#ifdef GLM_TIERSTREAM_PARSER
// Compiled from the read-only tierstream source tree (src/desc.cpp + src/json.cpp) in a separate TU.
std::string tierstream_desc_check(const std::string& text,std::uint64_t* layers,std::uint64_t* units,std::uint64_t* max_unit);
std::string tierstream_source_read(const std::string& desc_text,const char* raw_path,const std::string& sha256,std::uint64_t bytes,std::string* data);
#endif
#ifdef GLM_TIERSTREAM_PARSER
std::vector<std::vector<std::uint8_t>> tierstream_dg_steps(unsigned gated,const std::vector<std::pair<char,unsigned>>& ops);
#endif
// Predicted prefetch never targets a layer the step skips or holds: GLM's StepPlan
// (the binding's export/promise rule) replayed over GLM's own call order (two prefills
// through the cache = 43 single-layer held steps each, then decode tokens), a layer
// held during an all-layer step, and reset_open; every step's export set must equal
// the library's DgSteps (src/dg_steps.hpp, compiled from GLM_TIERSTREAM_SRC).
void step_plan_tests(){
    constexpr unsigned G=43,through=41,n=routed_topk;std::vector<std::pair<char,unsigned>> ops;
    for(int prefill=0;prefill<2;++prefill)for(unsigned g=0;g<G;++g){ops.push_back({'H',g});ops.push_back({'L',g});ops.push_back({'R',g});}
    for(int t=0;t<3;++t)ops.push_back({'A',0});
    ops.push_back({'H',5});ops.push_back({'A',0});ops.push_back({'L',5});ops.push_back({'L',6});ops.push_back({'X',0});ops.push_back({'A',0});
    StepPlan plan;plan.init(G);std::vector<std::vector<std::uint8_t>> mine;unsigned promised=0,decode_steps=0;
    for(auto [op,g]:ops){
        if(op=='H')plan.hold(g);else if(op=='R')plan.release(g);else if(op=='X')plan.reset();
        else{if(op=='A')plan.begin_all();else plan.begin_layer(g);mine.push_back(plan.run);
            bool all=true;for(unsigned h=0;h<G;++h){const unsigned p=plan.promise(h,n,through);
                check(!p||(plan.exports(h)&&plan.exports(h+1)),"a promise only when the step exports the layer and the next one");
                check(plan.promise(h,0,through)==0,"no promise without predicted ids");promised+=p?1:0;all=all&&plan.exports(h);}
            if(op=='L')for(unsigned h=0;h<G;++h)check(plan.promise(h,n,through)==0,"a single-layer step promises nothing");
            if(op=='A'&&all){++decode_steps;for(unsigned h=0;h<through;++h)check(plan.promise(h,n,through)==n,"a decode token promises every layer before `through`");
                check(plan.promise(through,n,through)==0&&plan.promise(G-1,n,through)==0,"no promise at or past `through`");}}
    }
    check(mine.size()==2*G+3+4&&decode_steps==4,"step count of the replayed sequence");
    // the held layer 5 in an all-layer step: neither 5 nor 4 (whose next is 5) promise; 5 is not exported
    const auto& held_step=mine[2*G+3];check(!held_step[5]&&held_step[4]&&held_step[6],"a held layer is not exported in an all-layer step");
    {StepPlan p;p.init(G);p.hold(5);p.begin_all();check(p.promise(4,n,through)==0&&p.promise(5,n,through)==0&&p.promise(6,n,through)==n,"no prediction toward or from a held layer");
     p.begin_layer(5);check(!p.exports(5),"a held layer's own single-layer step exports nothing");}
#ifdef GLM_TIERSTREAM_PARSER
    const auto lib=tierstream_dg_steps(G,ops);check(lib.size()==mine.size(),"library step count");
    for(std::size_t i=0;i<mine.size()&&i<lib.size();++i)check(lib[i]==mine[i],("step "+std::to_string(i)+": GLM's export set == the library's DgSteps").c_str());
    std::cout<<"INFO step plan == tierstream DgSteps over "<<mine.size()<<" steps ("<<promised<<" promises)\n";
#else
    std::cout<<"SKIP tierstream DgSteps comparison not configured (GLM_TIERSTREAM_SRC)\n";
#endif
}
// Decode timeline (decode_timeline.hpp): device intervals summed per phase and per
// layer, host enqueue time kept apart, totals equal to the sum of the intervals, and
// the per-arm aggregation over the steps that carry a timeline.
void decode_timeline_tests(){
    std::vector<TimelineMark> marks{{-1,DecodePhase::embed,0.01},{3,DecodePhase::router,0.02},{3,DecodePhase::export_wait,0.03},{3,DecodePhase::routed_moe,0.04},
                                    {4,DecodePhase::router,0.02},{4,DecodePhase::routed_moe,0.05},{-1,DecodePhase::head,0.01},{-1,DecodePhase::host_gap,1.5}};
    std::vector<double> gpu{0.1,0.2,2.0,3.0,0.25,3.5,1.0,1.4};
    const auto t=decode_timeline_json(marks,gpu,11.45);
    const auto* ph=t.find("phases");
    check(std::abs(ph->find("routed_moe")->find("gpu_ms")->as_double()-6.5)<1e-12&&ph->find("routed_moe")->find("count")->as_int()==2,"timeline: per-phase device sum and count");
    check(std::abs(ph->find("host_gap")->find("host_ms")->as_double()-1.5)<1e-12,"timeline: host time per phase");
    check(std::abs(t.find("gpu_ms_total")->as_double()-11.45)<1e-9&&std::abs(t.find("span_ms")->as_double()-11.45)<1e-12,"timeline: interval total == span");
    const auto& layers=t.find("layers")->as_array();
    check(layers.size()==2&&layers[0].find("layer")->as_int()==3&&std::abs(layers[0].find("export_wait")->as_double()-2.0)<1e-12&&!layers[1].find("export_wait"),"timeline: per-layer phases");
    check(!ph->find("kda")&&!ph->find("predict"),"timeline: phases never marked are omitted");
    bool refused=false;try{(void)decode_timeline_json(marks,{1.0},1.0);}catch(const std::invalid_argument&){refused=true;}check(refused,"timeline: one interval per mark");
    auto steps=Json::array();for(int i=0;i<2;++i){auto s=Json::object();s.set("token",i);s.set("timeline",t);steps.push(s);}
    auto plain=Json::object();plain.set("token",9);steps.push(plain);
    const auto agg=aggregate_decode_timelines(steps);
    check(agg.find("tokens")->as_int()==2&&std::abs(agg.find("span_ms_per_token")->as_double()-11.45)<1e-9
          &&std::abs(agg.find("phases")->find("routed_moe")->find("gpu_ms_per_token")->as_double()-6.5)<1e-9
          &&std::abs(agg.find("phases")->find("routed_moe")->find("gpu_ms_total")->as_double()-13.0)<1e-9,"timeline: per-arm totals over the steps that carry one");
    check(!aggregate_decode_timelines(Json::array()).find("tokens"),"timeline: no timeline, no totals");
}
// skip_empty_expert_groups: the C1 group schedule (c1_schedule.hpp, the same functions the
// kernels call). Group counts; the device's slot -> expert list is a permutation with the
// routed experts first (ascending); a model of the launch loop (groups 0.. with double-buffer
// parity g % 2 and the retired[g - 2] guard, each slot running its expert only when that expert
// has rows) runs exactly the routed experts once each, as the full schedule does.
void c1_schedule_tests(){
    namespace r=strix::glm::rtxpf;
    check(r::c1_groups(288,1,8,true)==1&&r::c1_groups(288,1,8,false)==18&&r::c1_groups(288,2,8,true)==1&&r::c1_groups(288,3,8,true)==2,"groups: decode rows");
    check(r::c1_groups(288,36,8,true)==18&&r::c1_groups(288,8192,8,true)==18&&r::c1_groups(1,1,1,true)==1&&r::c1_groups(1,4096,1,false)==1,"groups: prefill rows and one expert");
    std::mt19937 rng(87);
    for(unsigned rows:{1u,2u,3u,5u,40u}){
        for(int trial=0;trial<200;++trial){
            std::vector<std::uint32_t> counts(288,0);
            for(unsigned row=0;row<rows;++row){std::vector<unsigned> e(288);std::iota(e.begin(),e.end(),0u);std::shuffle(e.begin(),e.end(),rng);for(unsigned k=0;k<8;++k)++counts[e[k]];}
            std::vector<std::uint32_t> act(288);r::c1_compact(counts.data(),act.data(),288);
            auto sorted=act;std::sort(sorted.begin(),sorted.end());bool perm=true;for(unsigned j=0;j<288;++j)perm=perm&&sorted[j]==j;
            unsigned active=0;for(auto c:counts)active+=c?1:0;
            bool order=true;for(unsigned j=0;j+1<288;++j)if((counts[act[j]]>0)==(counts[act[j+1]]>0))order=order&&act[j]<act[j+1];
            for(unsigned j=0;j<288;++j)order=order&&((j<active)==(counts[act[j]]>0));
            const unsigned groups=r::c1_groups(288,rows,8,true);
            // the launch loop model: which experts run, how often, and the group numbering
            auto run=[&](const std::uint32_t* list,unsigned ngroups){std::vector<unsigned> ran(288,0);unsigned executed=0;bool consecutive=true;
                for(unsigned first=0;first<288&&first/16<ngroups;first+=16){const unsigned g=first/16;consecutive=consecutive&&g==executed;++executed;
                    for(unsigned local=0;local<std::min(16u,288-first);++local){const unsigned e=r::c1_expert(list,first+local);if(counts[e])++ran[e];}}
                return std::make_tuple(ran,executed,consecutive);};
            auto [ran_c,exec_c,cons_c]=run(act.data(),groups);auto [ran_f,exec_f,cons_f]=run(nullptr,r::c1_groups(288,rows,8,false));
            check(perm&&order,"compaction: a permutation, routed experts first and ascending");
            check(active<=16*groups&&exec_c==groups&&cons_c&&cons_f&&exec_f==18,"every routed expert lies in an executed group; groups numbered 0.. consecutively");
            check(ran_c==ran_f,"the compacted schedule runs exactly the experts the full one runs");
            bool once=true;for(unsigned e=0;e<288;++e)once=once&&ran_c[e]==(counts[e]?1u:0u);check(once,"each routed expert runs once, no empty expert runs");
        }
    }
    std::vector<std::uint32_t> all(288,1),act(288);r::c1_compact(all.data(),act.data(),288);bool id=true;for(unsigned j=0;j<288;++j)id=id&&act[j]==j;
    check(id,"every expert routed: the list is the identity (the full schedule)");
}
// Slot census (expert_tier.hpp): table homes per gated layer (persistent slots first, then the
// bank aliases), decode misses per exported layer, free persistent slots from the table.
void slot_census_tests(){
    constexpr std::uint32_t persistent=587,units=288;
    std::vector<std::int32_t> layer(units,-1);for(unsigned u=0;u<13;++u)layer[u]=std::int32_t(u*40);layer[200]=588;layer[201]=603-1;
    const auto c=census_layer(layer.data(),units,persistent);
    check(c.persistent==13&&c.bank==2&&c.absent==units-15,"census: persistent / bank / absent homes");
    std::vector<LayerCensus> all(43,c);std::vector<std::uint32_t> misses(43,0xFFFFFFFFu);misses[0]=8;misses[38]=8;misses[39]=0;
    const auto j=census_json(all,misses,persistent);
    check(j.find("persistent_used")->as_int()==43*13&&j.find("persistent_free")->as_int()==int(persistent-43*13)&&j.find("bank_used")->as_int()==86,"census: totals and free persistent slots");
    const auto& a=j.find("layers")->as_array();
    check(a.size()==43&&a[0].find("misses")->as_int()==8&&a[38].find("misses")->as_int()==8&&a[39].find("misses")->as_int()==0&&!a[1].find("misses")&&j.find("misses")->as_int()==16,
          "census: misses only for exported layers");
    std::vector<LayerCensus> full(43,LayerCensus{14,0,274});check(census_json(full,{},persistent).find("persistent_free")->as_int()==0,"census: a full arena has no free slot");
}
// restore_decode_residency: the open placement per gated layer from the tier array (resident
// keys, ascending units), sized like the plan (residents + spares = persistent slots).
void planned_residents_tests(){
    TierBudget b;b.resident=557;b.pinned=2495;b.disk=43ULL*288-557-2495;b.slots=587;
    const auto order=round_robin_order();const auto tiers=tier_assignment(b,43ULL*288,order);
    const auto p=planned_residents(tiers,0,43,288);std::uint64_t n=0;bool ascending=true,resident=true;
    for(unsigned g=0;g<43;++g){n+=p[g].size();for(std::size_t i=0;i<p[g].size();++i){ascending=ascending&&(i==0||p[g][i-1]<p[g][i]);resident=resident&&tiers[std::size_t(g)*288+std::size_t(p[g][i])]==0;}}
    check(n==557&&ascending&&resident,"planned residents: every resident key once, per layer ascending");
    check(p[0].size()==13&&p[40].size()==13&&p[41].size()==12&&p[42].size()==12,"planned residents: the round-robin open placement (13 per layer, 12 for 41/42)");
    check(planned_residents(std::vector<std::uint8_t>(43*288,1),0,43,288)[5].empty(),"planned residents: none without device residents");
}
// MTP prefill pieces (schedule.hpp): at most 2048 rows whatever the trunk tile, 4-aligned (absolute DSA
// positions pool every 4 rows), never larger than the tile itself; every row covered once.
void mtp_tile_tests(){
    check(mtp_row_tile(8192)==2048&&mtp_row_tile(4096)==2048&&mtp_row_tile(2048)==2048&&mtp_row_tile(1024)==1024&&mtp_row_tile(32)==32,"MTP piece = min(tile, 2048)");
    for(unsigned P:{31u,2047u,2048u,8191u,31999u}){unsigned covered=0,pieces=0;bool aligned=true;
        for(unsigned first=0;first<P;first+=mtp_row_tile(8192)){auto c=row_tile(first,P,mtp_row_tile(8192));aligned=aligned&&first%4==0;covered+=c.rows;++pieces;}
        check(covered==P&&aligned&&pieces==(P+2047)/2048,"MTP pieces cover every prefill row once, 4-aligned");}
}
// Per-request prefill tile (schedule.hpp select_prefill_tile): below the crossover min(4096, allocated),
// at/above it the allocated tile, crossover 0 = always allocated; both tiles cover the same rows once
// (exact: only the chunking changes); the two-slot graph template cache keyed by prefill_binding never
// replays a template captured for another shape when tiles alternate between requests.
// Plan-sized arena spares under a file tier (expert_tier.hpp prefill_spares_needed): one prefill layer's
// 288 units fit in free slots when spares >= 288 - residents per layer (for example 5955 residents -> 150).
void prefill_spares_tests(){
    check(prefill_spares_needed(5955)==150&&prefill_spares_needed(5493)==161&&prefill_spares_needed(0)==288,"spares needed = 288 - residents/43");
    check(prefill_spares_needed(43ULL*288)==0&&prefill_spares_needed(43ULL*300)==0,"all units resident: no spares needed");
    // the planner's fixed point: x spares leave (slots - x) residents; x >= 288 - (slots - x)/43
    for(std::uint64_t slots:{5988ull,5800ull,4000ull}){unsigned x=16;for(int it=0;it<8;++it){const unsigned want=std::max(16u,prefill_spares_needed(slots-x));if(want<=x)break;x=want;}
        check(x>=prefill_spares_needed(slots-x)&&x<slots,"plan-sized spares converge to a fixed point");}
}
void prefill_tile_selection_tests(){
    check(default_prefill_tile_crossover==0&&small_prefill_tile==4096,"crossover default 0 (always the allocated tile), small tile 4096");
    check(select_prefill_tile(8191,8192,default_prefill_tile_crossover)==8192&&select_prefill_tile(31999,8192,default_prefill_tile_crossover)==8192,"default: every request at the allocation");
    check(select_prefill_tile(8191,8192,16384)==4096&&select_prefill_tile(16383,8192,16384)==4096,"rows below the crossover: 4096");
    check(select_prefill_tile(16384,8192,16384)==8192&&select_prefill_tile(31999,8192,16384)==8192,"rows at/above the crossover: the allocated tile");
    check(select_prefill_tile(1023,2048,16384)==2048&&select_prefill_tile(1023,4096,16384)==4096,"an allocation <= 4096 is never exceeded");
    check(select_prefill_tile(1023,8192,0)==8192&&select_prefill_tile(31999,8192,0)==8192,"crossover 0 (off): always the allocated tile");
    check(select_prefill_tile(100,8192,1)==8192&&select_prefill_tile(0,8192,1)==4096,"crossover 1: only an empty prefix takes the small tile");
    check(std::string(prefill_tile_reason(1023,16384)).find("below")!=std::string::npos&&std::string(prefill_tile_reason(16384,16384)).find("at or above")!=std::string::npos
          &&std::string(prefill_tile_reason(5,0)).find("off")!=std::string::npos,"tile choice reasons");
    for(unsigned P:{1023u,4095u,4096u,8191u,16383u,16384u,31999u})for(unsigned t:{4096u,8192u}){
        std::vector<unsigned> hit(P,0);for(unsigned first=0;first<P;first+=t){auto c=row_tile(first,P,t);check(c.bucket==(c.rows==t?0u:1u)&&c.rows>0,"bucket 0 = full tile, 1 = tail");for(unsigned r=0;r<c.rows;++r)++hit[c.first+r];}
        check(std::all_of(hit.begin(),hit.end(),[](unsigned h){return h==1;}),"every prefill row in exactly one chunk at either tile");}
    // Model of graph[layer][bucket]: a slot replays only when its stored key equals the wanted key; the
    // captured shape must then be the wanted one (first, rows, session block), whatever the tile history.
    struct Slot{std::string key;unsigned first=0,rows=0,P=0;bool live=false;};
    for(bool session:{false,true}){Slot slot[2];unsigned replays=0,captures=0;
        const std::vector<std::pair<unsigned,unsigned>> requests{{8192,8192},{4096,8192},{8192,8192},{4096,4096},{8192,32000},{4096,32000},{8192,16384},{4096,16384},{4096,8192},{8192,8192},{8192,1024},{4096,1024}};
        for(auto [t,n]:requests){const unsigned P=n-1;const std::string sk=session?":s"+std::to_string(P/32*32)+"/"+std::to_string(P%32):"";
            for(unsigned first=0;first<P;first+=t){auto c=row_tile(first,P,t);auto& s=slot[c.bucket];const auto key=prefill_binding(first,c.rows,"glm-pf-rtx-native-v1",false,sk);
                if(s.live&&s.key==key){++replays;check(s.first==first&&s.rows==c.rows&&(!session||(s.P/32==P/32&&s.P%32==P%32)),"a replayed template has the wanted shape (no stale capture)");}
                else{++captures;s={key,first,c.rows,P,true};}}}
        check(replays>0&&captures>0,"the alternating sequence both replays and recaptures");}
    check(prefill_binding(0,4096,"glm-pf-rtx-native-v1",false,"")!=prefill_binding(0,8192,"glm-pf-rtx-native-v1",false,""),"a tile change changes the full-tile key");
    check(prefill_binding(0,8191,"x",true,":s8160/31")=="0:8191:x:1:s8160/31","binding key layout unchanged");
}
void tierstream_parser_tests(const ExpertCatalog& c,const Json& d){
#ifdef GLM_TIERSTREAM_PARSER
    std::uint64_t layers=0,units=0,max_unit=0;auto error=tierstream_desc_check(d.dump(),&layers,&units,&max_unit);
    check(error.empty()&&layers==2&&units==2*288&&max_unit==c.unit_stride(),"tierstream desc parser accepts the GLM unit descriptor");
    auto variant=[&](const std::string& from,const std::string& to){auto text=d.dump();auto at=text.find(from);check(at!=std::string::npos,"descriptor mutation anchor");text.replace(at,from.size(),to);return tierstream_desc_check(text,&layers,&units,&max_unit);};
    check(!variant("\"sources\"","\"source\"").empty(),"parser rejects singular source");
    check(!variant("\"index\":0","\"index\":3").empty(),"parser rejects model-layer indices");
    check(!variant(tierstream_unit_format,"glm-expert-unit-v1").empty(),"parser rejects unknown format");
    std::cout<<"INFO tierstream desc parser validated the GLM descriptor\n";
#else
    (void)c;(void)d;std::cout<<"SKIP tierstream desc parser not configured (GLM_TIERSTREAM_SRC)\n";
#endif
}
void catalog_tests(){
    auto s=synthetic();ExpertCatalog c(s.tensors,s.g);(void)0;check(c.size()==2*288,"catalog entries");
    // Lexicographic placement: 1 < 10 < 100 < 101 < 11 < 2 < 287 physically.
    auto b=[&](unsigned e){return c.at(3,e).span_begin;};
    check(b(0)<b(1)&&b(1)<b(10)&&b(10)<b(100)&&b(100)<b(101)&&b(101)<b(11)&&b(11)<b(2)&&b(2)<b(287),"lexicographic expert placement");
    check(b(2)!=2*(c.at(3,1).span_end-c.at(3,1).span_begin),"no id*average addressing");
    for(unsigned e:{0u,1u,2u,10u,100u,287u}){auto& x=c.at(3,e);
        for(unsigned m=0;m<3;++m){check(x.unit.rate[m]==rate_of(3,e,m),"mixed rates");check(x.t[m][0].bytes==trellis_bytes(s.g.in(m),s.g.out(m),x.unit.rate[m]),"trellis bytes");}
        check(x.span_begin%256==0&&x.unit.bytes<=c.unit_stride()&&c.unit_stride()%unit_alignment==0,"unit stride alignment");
        // Assemble the canonical unit from the layer blob (single in-memory source).
        auto& blob=s.blobs.at(x.blob);std::vector<std::byte> unit(c.unit_stride());AlignedMemory scratch(4*unit_alignment+c.max_span_bytes());
        auto io=read_extents(unit_extents(x,blob_locator),{blob.size()},[&](unsigned,std::uint64_t begin,std::uint64_t n,void* dst){std::memcpy(dst,blob.data()+begin,n);},unit.data(),scratch.p,4*unit_alignment+c.max_span_bytes());
        check(io.logical==x.unit.logical&&io.requests==1&&io.physical%unit_alignment==0&&io.physical>=x.span_end-x.span_begin,"one coalesced aligned read per expert");
        validate_unit_image(x,unit,c.unit_stride());
        for(unsigned m=0;m<3;++m)for(unsigned r=0;r<4;++r)check(std::memcmp(unit.data()+x.unit.offset[m][r],blob.data()+x.t[m][r].offset,x.t[m][r].bytes)==0,"compact rebased bytes");
        auto bad=unit;bad[c.unit_stride()-1]=std::byte{1};rejects([&]{validate_unit_image(x,bad,c.unit_stride());},"nonzero unit padding");
        bad=unit;bad[x.unit.offset[1][3]]=std::byte{0};rejects([&]{validate_unit_image(x,bad,c.unit_stride());},"unit mul1 marker");
        check(c.unit_offset(4,e)==(std::uint64_t(288)+e)*c.unit_stride(),"numeric unit order offset");
        auto u=unit_offsets(x);check(u.off[2][0]==x.unit.offset[2][0]&&u.rate[1]==x.unit.rate[1],"device unit offsets");
    }
    // Unit production: numeric order, per-unit digests, nothing written to disk.
    std::vector<std::string> blob_order{ExpertCatalog::blob_name(3),ExpertCatalog::blob_name(4)};std::uint64_t next=0;
    auto locate=[&](const ExpertTensor& t){return SourceExtent{t.blob==blob_order[0]?0u:1u,t.offset,t.bytes,0,0};};
    auto produced=produce_units(c,locate,{s.blobs.at(blob_order[0]).size(),s.blobs.at(blob_order[1]).size()},
        [&](unsigned f,std::uint64_t begin,std::uint64_t n,void* dst){std::memcpy(dst,s.blobs.at(blob_order[f]).data()+begin,n);},
        [&](unsigned l,unsigned e,std::span<const std::byte> image){check(c.unit_offset(l,e)==next*c.unit_stride()&&image.size()==c.unit_stride(),"numeric unit stream order");++next;});
    check(next==c.size()&&produced.size()==c.size(),"all units produced");
    std::vector<std::string> digests=produced;auto m=unit_manifest(c,std::string(64,'b'),digests,std::string(64,'c'));validate_unit_manifest(m,c,std::string(64,'b'));
    check(digests[0]!=digests[1],"distinct unit digests");
    rejects([&]{validate_unit_manifest(m,c,std::string(64,'c'));},"unit manifest model identity");
    // Missing or mistyped nested fields: controlled rejection, never a null dereference.
    auto erase=[](Json& o,const std::string& k){auto& v=o.as_object();v.erase(std::remove_if(v.begin(),v.end(),[&](auto& p){return p.first==k;}),v.end());};
    for(const char* k:{"layer","expert","rates","bytes","sha256"}){auto t=strix::serve::parse_json(m.dump());erase(mut(t,"units").as_array()[3],k);rejects([&]{validate_unit_manifest(t,c,std::string(64,'b'));},"missing unit field");}
    for(auto [k,v]:std::initializer_list<std::pair<const char*,Json>>{{"layer",Json("3")},{"expert",Json(1.5)},{"rates",Json(int(4))},{"bytes",Json(-1.0)},{"sha256",Json(int(7))}}){
        auto t=strix::serve::parse_json(m.dump());mut(mut(t,"units").as_array()[0],k)=v;rejects([&]{validate_unit_manifest(t,c,std::string(64,'b'));},"mistyped unit field");}
    {auto t=strix::serve::parse_json(m.dump());mut(mut(t,"units").as_array()[0],"rates").as_array()[1]=Json("x");rejects([&]{validate_unit_manifest(t,c,std::string(64,'b'));},"mistyped rate");}
    {auto t=strix::serve::parse_json(m.dump());mut(t,"units").as_array()[2]=Json(int(1));rejects([&]{validate_unit_manifest(t,c,std::string(64,'b'));},"unit entry not an object");}
    for(const char* k:{"units","unit_stride","format","hidden"}){auto t=strix::serve::parse_json(m.dump());erase(t,k);rejects([&]{validate_unit_manifest(t,c,std::string(64,'b'));},"missing root field");}
    {auto t=strix::serve::parse_json(m.dump());mut(t,"units")=Json::object();rejects([&]{validate_unit_manifest(t,c,std::string(64,'b'));},"units not an array");}
    rejects([&]{validate_unit_manifest(Json(int(3)),c,std::string(64,'b'));},"manifest not an object");
    {auto broken=strix::serve::parse_json(s.tensors.dump());mut(mut(broken,ExpertCatalog::prefix(3,9)+"gate_proj.trellis"),"shape").as_array()[0]=Json("16");rejects([&]{ExpertCatalog x(broken,s.g);},"mistyped shape dimension");}
    auto tampered=strix::serve::parse_json(m.dump());mut(mut(tampered,"units").as_array()[5],"rates").as_array()[0]=Json(int(9));rejects([&]{validate_unit_manifest(tampered,c,std::string(64,'b'));},"unit manifest rate");
    auto d=tierstream_descriptor(c,"units.bin",std::string(64,'b'));auto& layers=d.find("layers")->as_array();check(layers.size()==2&&d.find("sources")&&!d.find("source"),"descriptor sources/layers");
    for(std::size_t i=0;i<layers.size();++i){auto& L=layers[i];auto& parts=L.find("parts")->as_array();check(L.find("index")->as_int()==std::int64_t(i)&&L.find("x-glm-layer")->as_int()==std::int64_t(3+i),"gated ordinal indices 0..n-1");
        check(parts.size()==1&&std::uint64_t(parts[0].find("stride")->as_int())==c.unit_stride()&&std::uint64_t(parts[0].find("bytes")->as_int())==c.unit_stride()&&parts[0].find("format")->as_string()==tierstream_unit_format,"one opaque part, stride==bytes, x- format");}
    tierstream_parser_tests(c,d);
    // Interleaved or overlapping extents are rejected.
    auto broken=strix::serve::parse_json(s.tensors.dump());mut(broken,ExpertCatalog::prefix(3,7)+"up_proj.suh").set("offset",std::uint64_t(0));rejects([&]{ExpertCatalog x(broken,s.g);},"noncontiguous/overlapping expert");
    auto missing=strix::serve::parse_json(s.tensors.dump());missing.as_object().erase(std::remove_if(missing.as_object().begin(),missing.as_object().end(),[](auto& p){return p.first==ExpertCatalog::prefix(4,287)+"down_proj.mul1";}),missing.as_object().end());rejects([&]{ExpertCatalog x(missing,s.g);},"missing expert tensor");
}
void extent_tests(const std::filesystem::path& build){
    check(aligned_range(70000,10,1<<20).begin==65536&&aligned_range(70000,10,1<<20).skip==70000-65536&&aligned_range(70000,10,1<<20).length==65536,"aligned range");
    check(aligned_range(1000,100,1200).length==1200,"clamped to end of file");rejects([&]{aligned_range(1000,300,1200);},"range outside file");
    auto runs=coalesce({{0,100,10,0,0},{0,120,10,10,0},{0,1000000,10,20,0},{1,5,4,30,0}},65536,1<<20);check(runs.size()==3&&runs[0].pieces.size()==2,"coalesce by gap and file");
    rejects([&]{coalesce({{0,100,10,0,0},{0,105,10,10,0}},65536,1<<20);},"overlapping extents");rejects([&]{coalesce({{0,0,2<<20,0,0}},65536,1<<20);},"bounded staging");
    // Real retained-handle reads: an expert split across two source shards, unaligned offsets.
    auto root=build/"decode-host-shards";std::filesystem::remove_all(root);std::filesystem::create_directories(root);
    std::vector<std::vector<std::byte>> payload(2);
    for(unsigned f=0;f<2;++f){auto header=Json::object();std::size_t n=f?200000:131113;payload[f].resize(n);for(std::size_t i=0;i<n;++i)payload[f][i]=std::byte((i*7+f)&255);
        for(std::size_t i=0;i+1<n;i+=2){payload[f][i+1]=std::byte(0x3c);}if(!f){payload[f][10]=std::byte{0};payload[f][11]=std::byte{0x7c};}
        auto t=Json::object(),o=Json::array();o.push(0);o.push(std::uint64_t(n));t.set("dtype","I16");auto sh=Json::array();sh.push(std::uint64_t(n/2));t.set("shape",sh);t.set("data_offsets",o);header.set("x",t);
        auto raw=header.dump();std::uint64_t hn=raw.size();std::ofstream out(root/(f?"b.safetensors":"a.safetensors"),std::ios::binary);out.write(reinterpret_cast<const char*>(&hn),8);out.write(raw.data(),raw.size());out.write(reinterpret_cast<const char*>(payload[f].data()),n);}
    std::uint64_t base[2];for(unsigned f=0;f<2;++f){std::ifstream in(root/(f?"b.safetensors":"a.safetensors"),std::ios::binary);std::uint64_t hn=0;in.read(reinterpret_cast<char*>(&hn),8);base[f]=8+hn;}
    {ShardSources src(root);std::vector<std::uint64_t> sizes{src.files[0].bytes,src.files[1].bytes};
        std::vector<SourceExtent> ex{{0,base[0]+3,77,0,0},{0,base[0]+70000,4096,256,'h'},{0,base[0]+90001,33,4400,0},{1,base[1]+65535,9,8192,0},{1,base[1]+199990,10,8448,0}};
        std::vector<std::byte> dst(9000);(void)0;AlignedMemory scratch(1<<20);
        auto io=read_extents(ex,sizes,[&](unsigned f,std::uint64_t b,std::uint64_t n,void* d){ShardSources::read_span(src.files[f],b,n,d);},dst.data(),scratch.p,1<<20);
        check(io.logical==77+4096+33+9+10&&io.requests==4&&io.physical%65536==0,"cross-shard logical/physical accounting");
        for(auto& e:ex){auto f=e.file;check(std::memcmp(dst.data()+e.dst,payload[f].data()+(e.offset-base[f]),e.bytes)==0,"cross-shard unaligned bytes");}
        // A nonfinite F16 scale is rejected exactly as the streaming loader rejects it.
        std::vector<SourceExtent> bad{{0,base[0]+10,2,0,'h'}};rejects([&]{read_extents(bad,sizes,[&](unsigned f,std::uint64_t b,std::uint64_t n,void* d){ShardSources::read_span(src.files[f],b,n,d);},dst.data(),scratch.p,1<<20);},"nonfinite scale rejected");
    }
    std::filesystem::remove_all(root);
}
// ---- tierstream binding: budgets, tier assignment, slot -> descriptor mapping,
// miss accounting, unit file, and (with the CPU library) real admission.
void tier_budget_tests(){
    const std::uint64_t unit=12681216,keys=43ULL*288;TierBudgetIn in;in.keys=keys;in.unit_bytes=unit;
    in.device_bytes=40ULL<<30;in.host_bytes=400ULL<<30;auto two=tier_budget(in);
    check(two.refuse.empty()&&!two.three_tier&&two.disk==0&&two.resident+two.pinned==keys&&two.slots==two.resident+16&&two.ring==17,"two-tier when RAM holds every non-resident unit");
    check((two.slots+two.ring)*unit<=in.device_bytes&&(two.slots+two.ring+1)*unit>in.device_bytes,"arena fills the VRAM budget exactly by whole units");
    in.host_bytes=48ULL<<30;auto three=tier_budget(in);
    check(three.refuse.empty()&&three.three_tier&&three.staging==8&&three.disk>0&&three.resident+three.pinned+three.disk==keys,"three-tier when RAM is short");
    check((three.pinned+three.host_spares+three.staging)*unit<=in.host_bytes&&(three.pinned+three.host_spares+three.staging+1)*unit>in.host_bytes,"pinned tier fills the RAM budget");
    auto lib=library_budget(three,unit,keys,8);check(lib.ring_slots==17&&lib.device_bytes>=(three.slots+17)*unit&&lib.host_bytes>=three.host_bytes(unit),"library budget covers arena/ring and pinned blocks");
    // Defaults: no new option changes today's behaviour (queue 1 = no extra bounce buffers, no stripes, no clean evict).
    {const DecodeOptions d{};check(disk_bounce_readers(d)==0&&d.weight_dirs.empty()&&d.drive_readers==0&&d.tier_disk_queue==1&&!d.tier_clean_evict&&!d.routes_prefill&&TierBudgetIn{}.disk_readers==0,"new options default off");
     auto q=d;q.tier_disk_queue=8;check(disk_bounce_readers(q)==8,"queue 8: eight reader bounce buffers");
     bool bad=false;for(const char* v:{"",";","D:/s;;C:/s","D:/s;d:\\S\\"," ;D:/s"}){try{check_weight_dirs(v);}catch(...){continue;}bad=true;}
     check(!bad,"--weight-dirs: empty entries and duplicates refused");try{check_weight_dirs("D:/a;C:/b");}catch(...){check(false,"--weight-dirs: two distinct directories accepted");}}
    // dg_disk_queue > 1: one bounce buffer per disk reader, billed in the plan, host bytes and library budget.
    {auto q=three;q.disk_readers=8;const auto lq=library_budget(q,unit,keys,8);const auto stride=(unit+4095)/4096*4096;
     check(q.host_bytes(unit)==three.host_bytes(unit)+8*unit&&lq.host_bytes==lib.host_bytes+8*stride&&lq.device_bytes==lib.device_bytes,"disk-reader bounce buffers are billed to pinned host memory");}
    in.device_bytes=10*unit;check(!tier_budget(in).refuse.empty(),"VRAM below ring + spares refused");
    in.device_bytes=40ULL<<30;in.host_bytes=10*unit;check(!tier_budget(in).refuse.empty(),"host below spares + staging refused");
    in.host_bytes=48ULL<<30;in.file_tier=false;check(!tier_budget(in).refuse.empty(),"no file tier and short RAM refused");
    in.file_tier=true;in.device_bytes=std::uint64_t(keys+100)*unit;auto all=tier_budget(in);check(all.resident==keys&&all.pinned==0&&all.disk==0,"everything resident");
    auto t=tier_assignment(three,keys,{5000,7,5000,12000});std::uint64_t n[3]{};for(auto v:t)++n[v];
    check(n[0]==three.resident&&n[1]==three.pinned&&n[2]==three.disk&&t[5000]==0&&t[7]==0&&t[12000]==0,"priority order first, counts exact");
}
// Prefill through the cache, GLM side, against a CPU mock of the reader/resident
// contract: make_resident between tokens (kept until released,
// SS_E_CAPACITY -> no change), kept units resolve through the slot table in any
// step, single-layer steps. Every one of a layer's 288 slot-bound descriptors
// must address bytes identical to the source tensors (what the CUTLASS prefill reads).
void prefill_cache_tests(){
    auto s=synthetic();ExpertCatalog c(s.tensors,s.g);const std::uint64_t stride=c.unit_stride();
    struct Mock final:TierCacheApi {
        std::uint64_t stride;unsigned slots,spares,gated;std::vector<std::byte> arena;std::vector<std::int32_t> slot_of;std::vector<std::uint32_t> unit_base;
        std::vector<int> owner;std::vector<bool> kept;std::uint32_t abort=0,absent=0;bool in_step=false,fail=false;std::vector<unsigned> steps;
        std::function<void(unsigned,unsigned,std::byte*)> fill;
        Mock(std::uint64_t st,unsigned n,unsigned sp,std::function<void(unsigned,unsigned,std::byte*)> f,unsigned gl=2):stride(st),slots(n),spares(sp),gated(gl),arena(std::size_t(n)*st),slot_of(std::size_t(gl)*288,-1),unit_base(gl+1),owner(n,-1),kept(std::size_t(gl)*288,false),fill(std::move(f)){
            for(unsigned g=0;g<=gl;++g)unit_base[g]=g*288;}
        unsigned kept_n()const{return unsigned(std::count(kept.begin(),kept.end(),true));}
        bool supports_make_resident()const override{return true;}
        bool make_resident(unsigned g,const std::int32_t* ids,unsigned n)override{
            if(in_step)throw std::runtime_error("make_resident inside a step");if(fail)throw std::runtime_error("injected make_resident failure");
            if(n+kept_n()+spares+1>slots)return false; // SS_E_CAPACITY: nothing changed
            for(unsigned i=0;i<n;++i){const unsigned key=g*288+unsigned(ids[i]);kept[key]=true;if(slot_of[key]>=0)continue;
                int slot=-1;for(unsigned k=0;k<slots&&slot<0;++k)if(owner[k]<0)slot=int(k);
                for(unsigned k=0;k<slots&&slot<0;++k)if(!kept[unsigned(owner[k])]){slot_of[unsigned(owner[k])]=-1;slot=int(k);} // evict a non-kept unit
                if(slot<0)throw std::runtime_error("mock arena full");owner[unsigned(slot)]=int(key);slot_of[key]=slot;fill(g,unsigned(ids[i]),arena.data()+std::size_t(slot)*stride);}
            return true;}
        void release_resident(unsigned g)override{for(unsigned e=0;e<288;++e)kept[g*288+e]=false;}
        void begin_layer_step(unsigned g)override{if(in_step)throw std::runtime_error("nested step");in_step=true;steps.push_back(g);}
        void begin()override{in_step=true;}void enqueue_begin(void*)override{}void enqueue_export(unsigned,const std::int32_t*,unsigned,void*)override{}void enqueue_wait(unsigned,void*)override{}void enqueue_end(void*)override{}
        void end()override{in_step=false;}void reset_open()override{}Json stats()override{return Json::object();}
        TierView view()const override{TierView v;v.slot_of=slot_of.data();v.unit_base=unit_base.data();v.abort=&abort;v.absent_err=const_cast<std::uint32_t*>(&absent);v.guard_slot=slots;v.n_gated=gated;v.units=288;v.base=arena.data();v.stride=stride;return v;}
    };
    auto fill=[&](unsigned g,unsigned e,std::byte* dst){auto& x=c.at(g+3,e);auto& blob=s.blobs.at(x.blob);for(unsigned m=0;m<3;++m)for(unsigned r=0;r<4;++r)std::memcpy(dst+x.unit.offset[m][r],blob.data()+x.t[m][r].offset,x.t[m][r].bytes);};
    // Runtime init by mode (the decisions the runtime uses at its ring, layer
    // admission and descriptor sites): cache mode touches none of them.
    const auto cache_init=prefill_init(true),ring_init=prefill_init(false);
    check(!cache_init.rings&&!cache_init.admit_layers&&!cache_init.startup_descriptors,"cache mode: no rings, no layer admission, no startup routed descriptors");
    check(ring_init.rings&&ring_init.admit_layers&&ring_init.startup_descriptors,"interim mode keeps rings, admission and startup descriptors");
    Mock mock(stride,300,8,fill);CacheTokenSite site(mock);PrefillResidency residency;
    TierBudget b;b.slots=296;b.ring=17;b.spare_slots=8;check(PrefillResidency::refusal(b).empty(),"296 persistent slots keep a layer + 8 spares");
    b.slots=295;check(!PrefillResidency::refusal(b).empty(),"ring/bank slots never count as keepable (295 persistent slots refused)");
    check(PrefillResidency::min_resident()==288,"planner floor: 288 residents (the planner adds the spares)");
    b.slots=200;check(!PrefillResidency::refusal(b).empty(),"a smaller arena is refused (expert batches not implemented)");
    // Exact prefill protocol, both gated layers in order (layer 0's units are evicted for layer 1).
    for(unsigned g=0;g<2;++g){
        residency.acquire(site,g);site.begin_layer(g);check(mock.in_step&&mock.kept_n()==288,"one single-layer step over 288 kept units");
        const auto v=mock.view();const auto base=reinterpret_cast<std::uint64_t>(v.base);std::vector<bool> used(mock.slots,false);
        for(unsigned e=0;e<288;++e){bool absent=false;const auto slot=guarded_slot(v.slot_of,v.unit_base,*v.abort,v.guard_slot,g,e,&absent);
            check(!absent&&slot>=0&&unsigned(slot)<mock.slots&&!used[unsigned(slot)],"every expert resolves to its own resident slot");used[unsigned(slot)]=true;
            auto u=unit_offsets(c.at(g+3,e));auto& x=c.at(g+3,e);auto& blob=s.blobs.at(x.blob);
            for(unsigned m=0;m<3;++m)for(unsigned r=0;r<3;++r)check(std::memcmp(reinterpret_cast<const void*>(unit_address(base,slot,stride,u,m,r)),blob.data()+x.t[m][r].offset,x.t[m][r].bytes)==0,"prefill descriptor bytes == source tensor bytes");}
        site.end();site.release_resident(g);check(!mock.in_step&&mock.kept_n()==0,"step ended, layer released");
    }
    check(mock.steps==std::vector<unsigned>{0,1},"one single-layer step per routed layer, in order");
    site.begin();site.end();check(site.unusable().empty(),"decode tokens continue after prefill");
    // Residency is refused inside a token; a capacity refusal changes nothing but retires the worker.
    site.begin();rejects([&]{residency.acquire(site,0);},"residency inside a token");
    check(!site.unusable().empty(),"a residency precondition failure retires the worker (original error kept)");
    Mock tight(stride,200,8,fill);CacheTokenSite tsite(tight);
    bool capacity=false;try{residency.acquire(tsite,0);}catch(const std::exception& e){capacity=std::string(e.what()).find("SS_E_CAPACITY")!=std::string::npos;}
    check(capacity&&!tsite.unusable().empty()&&tight.kept_n()==0,"capacity refusal: nothing changed, the worker is retired irreversibly");
    bool reported=false;try{tsite.begin();}catch(const std::exception& e){reported=std::string(e.what()).find("SS_E_CAPACITY")!=std::string::npos;}
    check(reported,"later work reports the capacity refusal");
    // Any other failure retires the cache: later requests report the original error.
    Mock broken(stride,300,8,fill);broken.fail=true;CacheTokenSite bsite(broken);
    rejects([&]{residency.acquire(bsite,0);},"injected failure");bool original=false;try{bsite.begin();}catch(const std::exception& e){original=std::string(e.what()).find("injected make_resident failure")!=std::string::npos;}
    check(!bsite.unusable().empty()&&original,"a failed residency retires the cache with the original error");
    // Opt-in prefill overlap (PrefillOverlap), three gated layers in an arena that keeps exactly two: no token is
    // opened; layer k+1 is made resident while k's kernels are in flight and bound (enqueued) BEFORE the wait on the
    // previous layer's event; a layer is released only after its wait; never more than two held, none at the end; a
    // held layer's units never move, so its descriptors still address the source bytes when it is waited.
    b.slots=584;b.spare_slots=8;check(PrefillResidency::two_layers(b),"584 persistent slots keep two layers + 8 spares");
    b.slots=583;check(!PrefillResidency::two_layers(b)&&PrefillResidency::refusal(b).empty(),"583: one layer only (overlap off, the one-layer path runs)");
    {auto fill3=[&](unsigned g,unsigned e,std::byte* dst){fill(g%2,e,dst);};
     Mock om(stride,585,8,fill3,3);CacheTokenSite osite(om);PrefillOverlap ov;std::vector<std::string> log;unsigned most=0;std::vector<std::vector<std::int32_t>> slots(3);
     auto bound=[&](unsigned g){ // the slots a bind of g resolves (ss_dg_slot), their bytes checked against the source
        const auto v=om.view();const auto base=reinterpret_cast<std::uint64_t>(v.base);std::vector<std::int32_t> slots(288);
        for(unsigned e=0;e<288;++e){bool absent=false;slots[e]=guarded_slot(v.slot_of,v.unit_base,*v.abort,v.guard_slot,g,e,&absent);check(!absent&&slots[e]>=0&&unsigned(slots[e])<om.slots,"overlap: every expert of a held layer resolves");
            auto u=unit_offsets(c.at(g%2+3,e));auto& x=c.at(g%2+3,e);auto& blob=s.blobs.at(x.blob);
            for(unsigned m=0;m<3;++m)for(unsigned r=0;r<3;++r)check(std::memcmp(reinterpret_cast<const void*>(unit_address(base,slots[e],stride,u,m,r)),blob.data()+x.t[m][r].offset,x.t[m][r].bytes)==0,"overlap: descriptor bytes == source tensor bytes");}
        return slots;};
     auto acquire=[&](unsigned g){check(!osite.open()&&!om.in_step,"overlap: make_resident never inside a token");residency.acquire(osite,g);most=std::max(most,om.kept_n()/288);log.push_back("acquire "+std::to_string(g));};
     for(unsigned g=0;g<3;++g){ov.begin(g,acquire);slots[g]=bound(g);log.push_back("bind "+std::to_string(g)); // bind, then the layer's kernels (in flight)
        ov.end(g,g+1<3?int(g+1):-1,acquire,[&](unsigned h){check(bound(h)==slots[h],"overlap: a held layer never moved before its wait");log.push_back("wait "+std::to_string(h));},
               [&](unsigned h){osite.release_resident(h);log.push_back("release "+std::to_string(h));});}
     check(log==std::vector<std::string>{"acquire 0","bind 0","acquire 1","bind 1","wait 0","release 0","acquire 2","bind 2","wait 1","release 1","wait 2","release 2"},
           "overlap order: the next layer bound before the wait on the previous one, a layer released only after its wait");
     check(most==2&&om.kept_n()==0&&ov.held().empty()&&om.steps.empty()&&osite.unusable().empty(),"overlap: at most two held, none left, no step, worker usable");
     osite.begin();osite.end();check(osite.unusable().empty(),"decode tokens continue after an overlapped prefill");
     PrefillOverlap bad;bad.begin(0,[](unsigned){});rejects([&]{bad.begin(1,[](unsigned){});},"overlap: begin while an earlier layer is held");
     rejects([&]{bad.end(1,-1,[](unsigned){},[](unsigned){},[](unsigned){});},"overlap: end of a layer that is not held");
     // An arena that keeps one layer only: the overlapped make_resident is refused (SS_E_CAPACITY) and retires the
     // worker; the held layer is not released before its wait.
     Mock one(stride,584,8,fill3,3);CacheTokenSite one_site(one);PrefillOverlap o1;auto one_acquire=[&](unsigned g){residency.acquire(one_site,g);};
     o1.begin(0,one_acquire);rejects([&]{o1.end(0,1,one_acquire,[](unsigned){},[&](unsigned h){one_site.release_resident(h);});},"overlap without room refused");
     check(!one_site.unusable().empty()&&one.kept_n()==288,"overlap without room: the worker is retired, nothing released before the wait");
     // With try_acquire (the runtime's form) the same arena runs serially instead: the next layer is refused (nothing
     // changed), the current one is waited and released, then the next is acquired alone; the worker stays usable.
     Mock sm(stride,584,8,fill3,3);CacheTokenSite ssite(sm);PrefillOverlap so;std::vector<std::string> slog;unsigned serial=0;
     auto sacq=[&](unsigned g){residency.acquire(ssite,g);slog.push_back("acquire "+std::to_string(g));};
     auto stry=[&](unsigned g){const bool ok=residency.try_acquire(ssite,g);slog.push_back(std::string(ok?"try ":"refused ")+std::to_string(g));return ok;};
     for(unsigned g=0;g<3;++g){so.begin(g,sacq);
        if(!so.end(g,g+1<3?int(g+1):-1,stry,sacq,[&](unsigned h){slog.push_back("wait "+std::to_string(h));},[&](unsigned h){ssite.release_resident(h);slog.push_back("release "+std::to_string(h));}))++serial;}
     check(slog==std::vector<std::string>{"acquire 0","refused 1","wait 0","release 0","acquire 1","refused 2","wait 1","release 1","acquire 2","wait 2","release 2"},
           "overlap without room, try form: serial steps, a layer released only after its wait");
     check(serial==2&&ssite.unusable().empty()&&sm.kept_n()==0&&so.held().empty(),"overlap without room, try form: two serial steps, worker usable, nothing held");}
    // Prefill overlap's spare target (PrefillSpareTarget): lowered only when the spares alone block two held layers,
    // restored once at the prefill end or abort; a failed lowering keeps overlap off, a failed restore retires.
    {struct SpareMock final:TierCacheApi {unsigned target=31,open=31;int lowers=0,restores=0;bool fail_lower=false,fail_restore=false;
        void begin()override{}void enqueue_begin(void*)override{}void enqueue_export(unsigned,const std::int32_t*,unsigned,void*)override{}void enqueue_wait(unsigned,void*)override{}
        void enqueue_end(void*)override{}void end()override{}TierView view()const override{return TierView{};}void reset_open()override{}Json stats()override{return Json::object();}
        bool lower_spare_target(unsigned n)override{++lowers;if(fail_lower)throw std::runtime_error("injected lower failure");target=n;return true;}
        void restore_spare_target()override{++restores;if(fail_restore)throw std::runtime_error("injected restore failure");target=open;}};
     TierBudget tb;tb.slots=606;tb.spare_slots=31;std::string why;
     {SpareMock m;CacheTokenSite st(m);PrefillSpareTarget sp;
      check(sp.lower(m,st,tb,true,why)&&sp.lowered()==30&&m.target==30&&why.empty(),"spare target: 606 slots, 31 spares -> 30 for the prefill");
      sp.restore(m,st,true);sp.restore(m,st,true);check(m.target==31&&m.restores==1&&sp.lowered()==0&&st.unusable().empty(),"spare target: restored once at the prefill end");}
     {SpareMock m;CacheTokenSite st(m);PrefillSpareTarget sp;TierBudget roomy=tb,short_arena=tb;roomy.slots=700;short_arena.slots=576;
      check(!sp.lower(m,st,tb,false,why)&&!sp.lower(m,st,roomy,true,why)&&!sp.lower(m,st,short_arena,true,why)&&m.lowers==0,
            "spare target untouched: --prefill-overlap 0, two layers fit already, or two layers do not fit even without spares");}
     {SpareMock m;CacheTokenSite st(m);PrefillSpareTarget sp;check(sp.lower(m,st,tb,true,why),"spare target lowered");
      try{throw std::runtime_error("mid-prefill failure");}catch(const std::exception&){sp.restore(m,st,false);}
      check(m.target==31&&st.unusable().empty(),"abort mid-prefill: the open-time target restored");}
     {SpareMock m;m.fail_lower=true;CacheTokenSite st(m);PrefillSpareTarget sp;
      check(!sp.lower(m,st,tb,true,why)&&why.find("injected lower failure")!=std::string::npos&&m.restores==1&&m.target==31&&sp.lowered()==0&&st.unusable().empty(),
            "a failed lowering: overlap off, the open-time target set again, worker usable");}
     {SpareMock m;CacheTokenSite st(m);PrefillSpareTarget sp;(void)sp.lower(m,st,tb,true,why);m.fail_restore=true;
      rejects([&]{sp.restore(m,st,true);},"a failed restore at the prefill end fails the request");check(!st.unusable().empty(),"a failed restore retires the worker");}
     {SpareMock m;CacheTokenSite st(m);PrefillSpareTarget sp;(void)sp.lower(m,st,tb,true,why);m.fail_restore=true;
      sp.restore(m,st,false);check(!st.unusable().empty(),"abort path: a failed restore retires without a second throw");}
     {SpareMock m;m.fail_lower=true;m.fail_restore=true;CacheTokenSite st(m);PrefillSpareTarget sp;
      rejects([&]{(void)sp.lower(m,st,tb,true,why);},"a failed lowering whose reset fails throws");check(!st.unusable().empty(),"and retires the worker");}}
    std::cout<<"INFO prefill through the cache (mock reader/resident): 2 layers x 288 slot-bound descriptors == source bytes; overlap: 3 layers, two held at most\n";
}
void tier_mapping_tests(){
    auto s=synthetic();ExpertCatalog c(s.tensors,s.g);const auto stride=c.unit_stride();
    // A host "arena": slots hold unit images; slot_of maps (gated,unit) keys, -1 = absent.
    std::vector<std::byte> arena(5*stride);std::vector<std::int32_t> slot_of(2*288,-1);std::uint32_t unit_base[3]{0,288,576};
    const std::pair<unsigned,unsigned> placed[4]={{3,0},{3,287},{4,100},{4,1}};
    for(std::int32_t k=0;k<4;++k){auto [l,e]=placed[k];auto& x=c.at(l,e);auto& blob=s.blobs.at(x.blob);
        for(unsigned m=0;m<3;++m)for(unsigned r=0;r<4;++r)std::memcpy(arena.data()+k*stride+x.unit.offset[m][r],blob.data()+x.t[m][r].offset,x.t[m][r].bytes);
        slot_of[unit_base[gated_ordinal(l)]+e]=k;}
    const auto base=reinterpret_cast<std::uint64_t>(arena.data());
    for(auto [l,e]:placed){bool absent=false;auto slot=guarded_slot(slot_of.data(),unit_base,0,4,gated_ordinal(l),e,&absent);check(!absent&&slot>=0&&slot<4,"resident slot");
        auto u=unit_offsets(c.at(l,e));auto& x=c.at(l,e);auto& blob=s.blobs.at(x.blob);
        for(unsigned m=0;m<3;++m)for(unsigned r=0;r<3;++r)check(std::memcmp(reinterpret_cast<const void*>(unit_address(base,slot,stride,u,m,r)),blob.data()+x.t[m][r].offset,x.t[m][r].bytes)==0,"descriptor bytes == source tensor bytes");}
    bool absent=false;check(guarded_slot(slot_of.data(),unit_base,0,4,0,5,&absent)==4&&absent,"absent unit -> guard slot + absent_err");
    // Miss accounting from the library's copy/read counters (no misses - file_reads).
    CacheCounters a,b;a.lookups=100;a.misses=10;a.fetched=8;a.file_reads=2;b=a;b.lookups+=344;b.misses+=40;b.resurrected=5;b.covered=3;b.fetched+=30;b.prefetched=4;b.file_reads+=12;b.file_bytes=12*stride;
    auto d=cache_delta(a,b,stride);
    check(d.find("device_hits")->as_int()==304&&d.find("resurrected")->as_int()==5&&d.find("covered")->as_int()==3,"device hits, resurrections and covered misses reported separately");
    check(d.find("ram_copies")->as_int()==30+4-12&&d.find("file_reads")->as_int()==12&&!d.find("pinned_hits"),"RAM copies = copies - file reads (copy counters)");
    check(std::uint64_t(d.find("h2d_bytes")->as_int())==34*stride&&std::uint64_t(d.find("file_bytes")->as_int())==12*stride,"copy/file bytes");
    auto back=b;back.misses=a.misses-1;rejects([&]{cache_delta(a,back,stride);},"counters backwards");auto timeout=b;timeout.timeouts=1;rejects([&]{cache_delta(a,timeout,stride);},"device timeout fails the token");
    auto bad=b;bad.file_reads=a.file_reads+40;rejects([&]{cache_delta(a,bad,stride);},"file reads beyond copies are inconsistent");
    // Failed-token teardown: an injected failure after begin retires the cache; the
    // next begin reports the ORIGINAL failure, never "already open".
    TokenGuard g;g.begin();g.end();g.begin();g.fail("injected: CUTLASS launch failed at layer 17");
    bool reported=false;try{g.begin();}catch(const std::exception& e){std::string m=e.what();reported=m.find("injected: CUTLASS launch failed at layer 17")!=std::string::npos&&m.find("already open")==std::string::npos;}
    check(reported,"original failure reported after a failed token");rejects([&]{g.end();},"no end after a failed token");check(!g.failure().empty()&&!g.open(),"cache retired");
    TokenGuard idle;idle.fail("not inside a token");idle.begin();idle.end();check(idle.failure().empty(),"failure outside a token does not retire the cache");
    TokenGuard twice;twice.begin();rejects([&]{twice.begin();},"nested begin refused");
    // CacheTokenSite over a mock library: an injected failure from TierCacheApi::end()
    // keeps the token owned until fail() retires it with the ORIGINAL error.
    struct Mock final:TierCacheApi {int begins=0,ends=0,resets=0;bool fail_end=false;
        void begin()override{++begins;}void enqueue_begin(void*)override{}void enqueue_export(unsigned,const std::int32_t*,unsigned,void*)override{}void enqueue_wait(unsigned,void*)override{}void enqueue_end(void*)override{}
        void end()override{++ends;if(fail_end)throw std::runtime_error("injected end failure: token not accepted (device error word 4)");}
        TierView view()const override{return {};}void reset_open()override{++resets;}Json stats()override{return Json::object();}};
    Mock mock;CacheTokenSite site(mock);site.begin();site.end();site.reset_open();check(mock.begins==1&&mock.ends==1&&mock.resets==1&&site.unusable().empty(),"accepted token");
    site.begin();mock.fail_end=true;std::string original;try{site.end();}catch(const std::exception& e){original=e.what();}
    check(!original.empty()&&site.open(),"failing end keeps the token owned");site.fail(original);
    bool later=false;try{site.begin();}catch(const std::exception& e){std::string m=e.what();later=m.find("injected end failure")!=std::string::npos&&m.find("already open")==std::string::npos;}
    check(later&&mock.begins==2,"next token reports the original end failure without touching the library");rejects([&]{site.reset_open();},"no reset after a failed token");
    // Backend selection: one exclusive chain; tierstream always authenticates.
    auto ref=backend_plan("reference",false);check(ref.reference&&!ref.tierstream&&!ref.authenticate_unit_file,"reference backend");
    auto ts=backend_plan("tierstream",false);check(ts.tierstream&&ts.authenticate_unit_file&&!ts.reference_compare&&!ts.reference,"tierstream authenticates");
    auto both=backend_plan("tierstream",true);check(both.tierstream&&both.authenticate_unit_file&&both.reference_compare,"tierstream + comparison still authenticates");
    rejects([&]{backend_plan("reference",true);},"comparison needs tierstream");rejects([&]{backend_plan("other",false);},"unknown backend");
}
#ifdef GLM_TIERSTREAM_LIBRARY
// Full 43 x 288 catalog at production geometry (metadata only, lexicographic
// packing, K4 with some K3/K5/K6 experts), admitted by the real CPU library.
ExpertCatalog production_catalog(){
    struct T{const char* dtype;Json shape;std::uint64_t bytes;};std::map<std::string,T> all;
    auto arr=[](std::initializer_list<unsigned> d){auto a=Json::array();for(auto v:d)a.push(int(v));return a;};ExpertGeometry g;
    for(unsigned l=3;l<=45;++l)for(unsigned e=0;e<288;++e)for(unsigned m=0;m<3;++m){auto b=ExpertCatalog::prefix(l,e)+matrix_names[m]+".";unsigned in=g.in(m),out=g.out(m),r=e%97==5?3+m:4;
        all.emplace(b+"trellis",T{"I16",arr({in/16,out/16,16*r}),trellis_bytes(in,out,r)});all.emplace(b+"suh",T{"F16",arr({in}),std::uint64_t(in)*2});all.emplace(b+"svh",T{"F16",arr({out}),std::uint64_t(out)*2});all.emplace(b+"mul1",T{"I32",Json::array(),4});}
    Json::Object members;members.reserve(all.size());std::map<std::string,std::uint64_t> at;
    for(auto& [name,t]:all){auto l=unsigned(std::stoul(name.substr(name.find(".layers.")+8)));auto blob=ExpertCatalog::blob_name(l);auto& off=at[blob];off=align_up(off,256);
        auto v=Json::object();v.set("blob",blob);v.set("offset",off);v.set("bytes",t.bytes);v.set("dtype",t.dtype);v.set("shape",t.shape);members.emplace_back(name,v);off+=t.bytes;}
    return ExpertCatalog(Json(std::move(members)));
}
void production_admission_tests(){
    auto c=production_catalog();const auto stride=c.unit_stride(),keys=c.size();check(keys==43ULL*288&&stride%65536==0,"production catalog");
    auto probe=[&](std::uint64_t residents,std::uint64_t host_bytes,std::uint32_t dynamic){TierBudgetIn in;in.keys=keys;in.unit_bytes=stride;in.device_bytes=(residents+16+17)*stride;in.host_bytes=host_bytes;
        TierOpenConfig cfg;cfg.budget=tier_budget(in);check(cfg.budget.refuse.empty()&&cfg.budget.resident==std::min(residents,keys),"production budget");
        cfg.descriptor_json=tierstream_descriptor(c,"OUT_DIR/glm-units.bin",std::string(64,'b')).dump();cfg.tiers=tier_assignment(cfg.budget,keys,{});cfg.unit_bytes=stride;cfg.cuda=false;cfg.dynamic_per_layer=dynamic;
        return std::make_pair(tierstream_admission_probe(cfg),cfg.budget);};
    for(auto [residents,host]:std::initializer_list<std::pair<std::uint64_t,std::uint64_t>>{{3353,48ULL<<30},{3353,200ULL<<30},{1,48ULL<<30},{keys-20,48ULL<<30},{6000,16ULL<<30}}){
        auto [r,b]=probe(residents,host,0);
        check(r.first==8,("tierstream admits the full 43x288 open: residents "+std::to_string(b.resident)+" pinned "+std::to_string(b.pinned)+" file "+std::to_string(b.disk)+" (got "+std::to_string(r.first)+": "+r.second+")").c_str());}
    // The reviewed defect: two billed dynamic slots per layer against the 16 global
    // spares exceeds the device budget at 3,353 residents; zero per-layer billing fits.
    auto [old_billing,ob]=probe(3353,48ULL<<30,2);(void)ob;check(old_billing.first!=8&&old_billing.first!=0,("per-layer dynamic billing is refused (got "+std::to_string(old_billing.first)+")").c_str());
    std::cout<<"INFO full 43x288 admission at stride "<<stride<<": ok; per-layer billing refused: "<<old_billing.second<<"\n";
    // Full-card shape with the DS placement: 5% spares, round-robin seed (dg_seed +
    // matching tier array), each boundary policy; all admitted by the library.
    {TierBudgetIn in;in.keys=keys;in.unit_bytes=stride;in.device_bytes=(6262ULL+17)*stride;in.host_bytes=~0ULL>>1;in.spare_permille=50;
        TierOpenConfig cfg;cfg.budget=tier_budget(in);check(cfg.budget.refuse.empty()&&cfg.budget.spare_slots==(6262ULL*50+999)/1000,"full-card 5% spares");
        cfg.descriptor_json=tierstream_descriptor(c,"OUT_DIR/glm-units.bin",std::string(64,'b')).dump();cfg.seed=seed_order({},{});cfg.tiers=tier_assignment(cfg.budget,keys,cfg.seed);cfg.unit_bytes=stride;cfg.cuda=false;
        for(int policy:{0,1,2}){cfg.policy=policy;auto r=tierstream_admission_probe(cfg);check(r.first==8,("seeded open admitted, policy "+std::to_string(policy)+" (got "+std::to_string(r.first)+": "+r.second+")").c_str());}
        auto wrong=cfg;std::swap(wrong.seed[0],wrong.seed.back());auto rw=tierstream_admission_probe(wrong);
        check(rw.first==-1&&rw.second.find("seed head")!=std::string::npos,"a seed whose head is not the resident tier is refused before the library");
        std::cout<<"INFO seeded full-card admission: resident "<<cfg.budget.resident<<" + spares "<<cfg.budget.spare_slots<<", round-robin dg_seed, lru/scored/auto admitted\n";}
}
// The earlier GLM tier rules, kept ONLY as the oracle proving that tier_budget
// (now a thin call into tierstream's ss_budget_plan) gives identical results.
TierBudget tier_budget_oracle(const TierBudgetIn& in){
    TierBudget p;p.ring=2ULL*in.max_ids+1;const std::uint64_t arena=in.device_bytes/in.unit_bytes;
    if(in.spare_slots<1){p.refuse="spare";return p;}
    if(arena<p.ring+in.spare_slots+1){p.refuse="vram";return p;}
    p.spare_slots=in.spare_slots;p.resident=std::min<std::uint64_t>(arena-p.ring-p.spare_slots,in.keys);p.slots=p.resident+p.spare_slots;
    const std::uint64_t nonres=in.keys-p.resident,cap=in.host_bytes/in.unit_bytes;p.host_spares=in.host_spares;
    if(nonres+in.host_spares<=cap){p.pinned=nonres;if(p.spare_slots>nonres){p.spare_slots=nonres;p.slots=p.resident+p.spare_slots;}if(!p.spare_slots)p.refuse="resident";return p;}
    if(!in.file_tier){p.refuse="nofile";return p;}
    if(cap<std::uint64_t(in.host_spares)+in.staging+1){p.refuse="host";return p;}
    p.three_tier=true;p.staging=in.staging;p.pinned=std::min(nonres,cap-in.host_spares-in.staging);p.disk=nonres-p.pinned;
    if(!p.disk){p.three_tier=false;p.staging=0;}
    if(p.spare_slots>p.pinned+p.disk){p.spare_slots=p.pinned+p.disk;p.slots=p.resident+p.spare_slots;}
    return p;
}
// Whole-device budgets through tierstream's ss_budget_plan / ss_budget_ledger at the
// real 43x288 geometry. GLM's parts are representative here (the exact ones need
// the CUDA workspace layouts; --plan-budget prints them from the shard headers).
void decode_budget_tests(){
    // (1) Uncapped arena/pinned budgets: identical to the earlier rules on a grid.
    std::uint64_t cases=0,refusals=0;
    for(std::uint64_t keys:{1ULL,7ULL,64ULL,43ULL*288})for(std::uint64_t unit:{65536ULL,12648448ULL})for(std::uint32_t spares:{0u,1u,16u})for(std::uint32_t hs:{1u,8u})for(std::uint32_t st:{1u,8u})for(bool file:{true,false})
    for(std::uint64_t du:{0ULL,10ULL,17ULL+16+1,40ULL,keys/2+40,keys+33,keys+200})for(std::uint64_t hu:{0ULL,5ULL,9ULL,17ULL,keys/3,keys,keys*2})for(std::uint64_t extra:{0ULL,1ULL,65535ULL}){
        TierBudgetIn in;in.keys=keys;in.unit_bytes=unit;in.spare_slots=spares;in.host_spares=hs;in.staging=st;in.file_tier=file;in.device_bytes=du*unit+extra;in.host_bytes=hu*unit+extra;
        const auto a=tier_budget_oracle(in),b=tier_budget(in);++cases;refusals+=!a.refuse.empty();
        const bool same=a.refuse.empty()==b.refuse.empty()&&(!a.refuse.empty()||(a.slots==b.slots&&a.resident==b.resident&&a.pinned==b.pinned&&a.disk==b.disk&&a.spare_slots==b.spare_slots&&a.host_spares==b.host_spares&&a.staging==b.staging&&a.ring==b.ring&&a.three_tier==b.three_tier));
        if(!same)check(false,("tier_budget == earlier rules: keys "+std::to_string(keys)+" unit "+std::to_string(unit)+" spares "+std::to_string(spares)+" hs "+std::to_string(hs)+" st "+std::to_string(st)+" file "+std::to_string(file)+" du "+std::to_string(du)+" hu "+std::to_string(hu)+" extra "+std::to_string(extra)+" ("+a.refuse+" | "+b.refuse+")").c_str());
    }
    check(cases>10000&&refusals>0,"oracle grid size");
    // (1b) Spares as a share of the arena (retained misses): the full-card arena
    // (6262 slots of 12648448 B) keeps its bytes; 5% of it becomes spares.
    {TierBudgetIn in;in.keys=43ULL*288;in.unit_bytes=12648448ULL;in.device_bytes=(6262ULL+17)*in.unit_bytes;in.host_bytes=~0ULL>>1;
        const auto base=tier_budget(in);in.spare_permille=50;const auto share=tier_budget(in);
        const std::uint64_t arena=base.resident+base.spare_slots,want=(arena*50+999)/1000;
        check(base.refuse.empty()&&share.refuse.empty()&&base.spare_slots==16,"spare share: both plans admitted, base 16 spares");
        check(share.spare_slots==want&&share.resident+share.spare_slots==arena&&share.slots==arena,"spare share: 5% of the same arena, residents give way");
        in.spare_permille=1;const auto tiny=tier_budget(in);check(tiny.spare_slots==16,"a share below the explicit minimum keeps the minimum");
        std::cout<<"INFO spare share 5%: arena "<<arena<<" slots -> resident "<<share.resident<<" + spares "<<share.spare_slots<<" (was "<<base.resident<<" + 16)\n";}
    std::cout<<"INFO tier_budget via ss_budget_plan == earlier GLM rules on "<<cases<<" cases ("<<refusals<<" refusals)\n";
    // (2) Whole-device 24 GiB VRAM + 32 GiB RAM at 43x288.
    auto c=production_catalog();const std::uint64_t gib=1ULL<<30,stride=c.unit_stride(),layer=288*c.max_span_bytes();
    auto glm=[&]{WholeDeviceIn w;w.vram_total=24*gib;w.ram_total=32*gib;w.context=640ULL<<20;w.host_avail=100*gib;
        w.device={{"trunk + dense + shared experts (base.bin)",9*gib},{"activations (capacity, row tile)",3*gib},{"decode KV/state + replay (capacity)",gib},{"workspaces",gib/2},{"prefill expert rings (2 x largest layer)",2*layer}};w.vram_margin=gib;w.ram_margin=gib;
        w.pinned={{"prefill expert rings (2 x largest layer)",2*layer}};w.pageable={{"process private bytes at startup (measured)",2*gib}};
        w.tier.keys=c.size();w.tier.unit_bytes=stride;return w;};
    auto in=glm();auto p=plan_whole_device(in);const auto j=whole_device_json(in,p);
    check(p.refuse.empty()&&p.device_planned<=24*gib&&p.host_planned<=32*gib&&p.tier.resident>0&&p.tier.resident+p.tier.pinned+p.tier.disk==c.size(),("whole-device 24/32 admitted within the caps: "+p.refuse).c_str());
    check(p.device_planned+in.vram_margin<=24*gib&&p.device_planned+in.vram_margin+stride>24*gib,"the arena is everything the VRAM cap leaves after the margin (one more slot would not fit)");
    check(p.min_vram_total<=24*gib&&p.min_ram_total<=32*gib,"minimum caps below the requested ones");
    {   // host growth after open (session snapshot + driver headroom) comes out of the pinned tier, not the cap
        auto grown=glm();const std::uint64_t snap=489406976;add_host_growth_parts(grown.pageable,snap);auto g=plan_whole_device(grown);
        check(g.refuse.empty()&&g.host_planned<=32*gib&&g.tier.pinned<p.tier.pinned,"host growth parts admitted under the RAM cap, fewer pinned units");
        const std::uint64_t freed=(p.tier.pinned-g.tier.pinned)*stride;
        check(freed+stride>snap+host_growth_headroom&&freed<=snap+host_growth_headroom+2*stride,"the pinned tier gives up exactly the snapshot + headroom (to a unit)");
        check(grown.pageable.size()==in.pageable.size()+2&&grown.pageable.back().name==std::string(host_growth_part),"named parts for the plan JSON");
    }
    TierOpenConfig cfg;cfg.budget=p.tier;cfg.descriptor_json=tierstream_descriptor(c,"OUT_DIR/glm-units.bin",std::string(64,'b')).dump();cfg.tiers=tier_assignment(p.tier,c.size(),{});cfg.unit_bytes=stride;cfg.cuda=false;
    auto lib=tierstream_admission_probe(cfg);check(lib.first==8,("ts library admits the whole-device 24/32 open (got "+std::to_string(lib.first)+": "+lib.second+")").c_str());
    std::cout<<"INFO whole-device 24/32 at 43x288 (representative parts): "<<j.find("summary")->as_string()<<"\n";
    // (2b) Default prefill row tile: the largest tile <= 8192 the whole-device plan admits
    // whose activation + workspace growth over tile 2048 is <= 5% of the expert arena
    // planned at 2048 (the tile must not eat the arena). Prefill through the cache (no
    // rings, 288 kept + spares). Workspace representative of the routed CUTLASS layout:
    // 1.6 GiB of weight staging + 600 KiB per row (8 routes of operands and scratch).
    {auto plan=[&](unsigned cap,unsigned t,std::uint64_t vram){auto w=glm();w.vram_total=vram;w.ram_total=0;w.min_resident=PrefillResidency::min_resident();
            std::erase_if(w.device,[](const BudgetPart& d){return d.name.find("rings")!=std::string::npos;});w.pinned.clear();
            for(auto& d:w.device)if(d.name.rfind("activations",0)==0)d.bytes=activation_bytes(cap,t);
            w.device.push_back({"workspaces (tile-scaled, representative)",(1600ULL<<20)+std::uint64_t(t)*(600ULL<<10)});return plan_whole_device(w);};
        auto cost=[](unsigned cap,unsigned t){return activation_bytes(cap,t)+(1600ULL<<20)+std::uint64_t(t)*(600ULL<<10);};
        auto choose=[&](unsigned cap,std::uint64_t vram,std::uint64_t* arena_out){const auto base=plan(cap,2048,vram);const std::uint64_t arena=base.refuse.empty()?base.tier.slots*stride:0;*arena_out=arena;
            auto ok=[&](unsigned t){const auto p=plan(cap,t,vram);const auto c=cost(cap,t),c0=cost(cap,2048);return p.refuse.empty()&&(t<=2048||(arena&&tile_growth_within(c>c0?c-c0:0,arena)));};
            const auto t=largest_fitting_tile(cap,ok,default_max_row_tile);
            check(t&&t<=default_max_row_tile&&valid_schedule(cap,t)&&ok(t),("chosen tile obeys the rule (capacity "+std::to_string(cap)+", VRAM "+std::to_string(vram>>30)+" GiB, tile "+std::to_string(t)+")").c_str());
            for(unsigned x=first_tile_candidate(cap,default_max_row_tile);x>t;x=next_tile_candidate(x))check(!ok(x),"every larger candidate breaks the plan or the 5% rule");return t;};
        std::uint64_t a_full=0,a24=0;const unsigned full=choose(32768,93*gib,&a_full),t24=choose(8320,24*gib,&a24);
        check(full==4096&&t24==2048,"full card 4096 (8192 would cost over 5% of the arena), 24/32 2048 (a small arena)");
        check(!tile_growth_within(cost(32768,8192)-cost(32768,2048),a_full)&&tile_growth_within(cost(32768,4096)-cost(32768,2048),a_full),"the 5% boundary lies between 4096 and 8192 on the full card");
        check(tile_growth_within(0,0)&&!tile_growth_within(6,100)&&tile_growth_within(5,100),"5% rule arithmetic");
        std::cout<<"INFO default row tile (<= 8192, growth <= 5% of the arena): full card cap 32768 -> "<<full<<" (arena "<<gib_text(a_full)<<", 8192 would cost "<<gib_text(cost(32768,8192)-cost(32768,2048))
                 <<"), 24/32 cap 8320 -> "<<t24<<" (arena "<<gib_text(a24)<<", 4096 would cost "<<gib_text(cost(8320,4096)-cost(8320,2048))<<")\n";}
    // (3) A cap that cannot hold the fixed parts is refused naming them and the minimum N.
    auto tight=glm();tight.vram_total=12*gib;auto t=plan_whole_device(tight);
    check(!t.refuse.empty()&&t.refuse.find("--vram-total-gib")!=std::string::npos&&t.refuse.find("trunk")!=std::string::npos&&t.min_vram_total>12*gib,("too small a VRAM cap refused with the minimum: "+t.refuse).c_str());
    auto small_ram=glm();small_ram.ram_total=4*gib;auto r=plan_whole_device(small_ram);check(!r.refuse.empty()&&r.refuse.find("--ram-total-gib")!=std::string::npos,("too small a RAM cap refused: "+r.refuse).c_str());
    std::cout<<"INFO refusal: "<<t.refuse<<"\n";
    // Keepable boundary through the planner AND the residency check (8 spares):
    // a cap of exactly the minimum gives 288 residents = 296 persistent slots (admitted), one byte less is refused.
    auto edge=glm();edge.ram_total=0;edge.tier.spare_slots=8;edge.min_resident=PrefillResidency::min_resident();edge.vram_total=0;
    const auto probe=plan_whole_device(edge);edge.vram_total=probe.min_vram_total;auto at=plan_whole_device(edge);
    check(at.refuse.empty()&&at.tier.resident==288&&at.tier.slots==296&&PrefillResidency::refusal(at.tier).empty(),("296 persistent slots admitted by planner and residency: "+at.refuse).c_str());
    edge.vram_total=probe.min_vram_total-1;auto below=plan_whole_device(edge);
    check(!below.refuse.empty(),"one byte below the keepable minimum is refused by the planner");
    TierBudget b295;b295.slots=295;b295.spare_slots=8;check(!PrefillResidency::refusal(b295).empty(),"295 persistent slots refused by the residency check");
    // (4) Without caps the arena-only behaviour stays: every expert resident when the device allows.
    auto open=glm();open.vram_total=0;open.ram_total=0;auto u=plan_whole_device(open);check(u.tier.resident==c.size()||!u.refuse.empty(),"uncapped: the planner places every expert (device room is applied by the caller)");
    // (5) Ledger: charges by name against the caps, refusal names the component, peaks.
    BudgetLedger l(24*gib,32*gib);std::string why;
    check(l.add(MemKind::device,"trunk",20*gib)&&!l.add(MemKind::device,"arena",5*gib,&why)&&why.find("arena")!=std::string::npos,("over-cap device charge refused naming it: "+why).c_str());
    check(l.add(MemKind::pinned,"rings",8*gib)&&l.add(MemKind::pageable,"process",2*gib)&&!l.add(MemKind::pinned,"pinned experts",23*gib),"host cap covers pinned + pageable");
    l.sub(MemKind::device,"trunk",20*gib);l.observe("load",21*gib,~0ULL);l.observe("request:qualify",3*gib,30*gib);auto s=l.stats();
    check(s.find("device")->find("peak_bytes")->as_int()==std::int64_t(20*gib)&&s.find("measured_device_peak_bytes")->as_int()==std::int64_t(21*gib)&&s.find("refusals")->as_int()==2&&l.current(MemKind::device)==0,"ledger peaks, measured peak and refusals");
    check(s.find("measured_device_peak_at")->as_string()=="load"&&s.find("measured_host_peak_bytes")->as_int()==std::int64_t(30*gib)&&l.within_caps(),"measured peaks by stage, within the caps");
    l.observe("request:bench",25*gib,~0ULL);check(!l.within_caps()&&l.stats().find("device_samples_over_cap")->as_int()==1,"a measured sample above the VRAM cap fails within_caps");
}
#endif
// Placement, the DS way (seed_order): round-robin fallback, residency file,
// routes census, priority and the per-layer balance of the resident tier.
void placement_tests(){
    auto rr=round_robin_order();check(rr.size()==43ULL*288&&rr[0]==0&&rr[1]==288&&rr[42]==42*288&&rr[43]==1&&rr.back()==42*288+287,"expert-major round-robin across all 43 routed layers");
    {std::vector<bool> seen(rr.size());for(auto k:rr){check(!seen[k],"round-robin: every key once");seen[k]=true;}}
    // Full-card residents (6229 of 12384): layer order put layers 3..24 on the device; round-robin balances every layer.
    TierBudget b;b.resident=6229;b.pinned=43ULL*288-6229;auto layer_order=tier_assignment(b,43ULL*288,{}),balanced=tier_assignment(b,43ULL*288,rr);
    auto per_layer=[](const std::vector<std::uint8_t>& t){std::uint64_t lo=~0ULL,hi=0;for(unsigned g=0;g<43;++g){std::uint64_t n=0;for(unsigned u=0;u<288;++u)n+=t[g*288+u]==0;lo=std::min(lo,n);hi=std::max(hi,n);}return std::pair{lo,hi};};
    auto [lo0,hi0]=per_layer(layer_order);auto [lo1,hi1]=per_layer(balanced);
    check(lo0==0&&hi0==288,"layer order: some layers fully resident, others not at all");check(hi1-lo1<=1&&lo1==6229/43,"round-robin: residents per layer differ by at most one");
    std::cout<<"INFO full-card residents per layer: layer order "<<lo0<<".."<<hi0<<", round-robin "<<lo1<<".."<<hi1<<"\n";
    // Residency file (DS layout, glm tag): round trip, foreign model / geometry / repeats refused.
    const std::string sha(64,'c');std::vector<std::uint32_t> keys{5,300,12383,0};for(std::uint32_t k=1000;k<1100;++k)keys.push_back(k);
    auto text=residency_text(sha,keys);check(text.rfind("glm-residency-v1 sha256="+sha+" layers=43 experts=288 n=104\n",0)==0,"residency header");
    check(parse_residency(text,sha)==keys,"residency round trip");
    auto refused=[&](const std::string& s,const std::string& m){try{parse_residency(s,m);}catch(...){return true;}return false;};
    check(refused(text,std::string(64,'d'))&&refused(residency_text(sha,keys,42,288),sha)&&refused("glm-residency-v1 sha256="+sha+" layers=43 experts=288 n=2\n7 7\n",sha)&&refused("glm-residency-v1 sha256="+sha+" layers=43 experts=288 n=1\n12384\n",sha)&&refused("glm-residency-v1 sha256="+sha+" layers=43 experts=288 n=3\n1 2\n",sha),
          "residency: other model, geometry, repeated, out-of-range or short key lists refused");
    // Routes census (DS routes format, model layers): one line per token.
    std::vector<std::uint32_t> ids(43*8);for(unsigned g=0;g<43;++g)for(unsigned k=0;k<8;++k)ids[g*8+k]=(g*7+k*31)%288;
    auto line=routes_line(ids.data());check(line.rfind("3:0,31,62,93,124,155,186,217 4:7,",0)==0&&line.find("45:")!=std::string::npos,"routes line: model layers 3..45");
    std::vector<std::uint64_t> counts;auto tokens=count_routes("# seq a rows 5\n"+line+"\n"+line+"\n# seq b rows 2\n3:9\n",counts);
    check(tokens==3&&counts[0*288+0]==2&&counts[0*288+9]==1&&counts[42*288+ids[42*8]]==2,"routes census counts by key");
    bool bad=false;try{count_routes("2:1\n",counts);}catch(...){bad=true;}check(bad,"routes: dense layer refused");
    // Prefill census rows: the unset MTP layer (45) is left out, never counted as expert 0.
    {auto pre=ids;for(unsigned k=0;k<8;++k)pre[42*8+k]=~0u;const auto pl=routes_line_present(pre.data());
     check(pl==line.substr(0,line.find(" 45:")),"prefill census line: layers 3..44 only");
     std::vector<std::uint64_t> pc;check(count_routes(pl+"\n",pc)==1&&pc[42*288+ids[42*8]]==0&&pc[0*288+0]==1,"prefill census: no MTP-layer counts");}
    // A layer the census never saw (the MTP layer of a prefill census) gets the observed mean per unit,
    // so it interleaves with the observed layers instead of sorting last with no device slot.
    {std::vector<std::uint64_t> u(43*288);for(unsigned g=0;g<42;++g)for(unsigned e=0;e<288;++e)u[g*288+e]=(e%4==0)?8:0; // mean 2
     check(fill_unobserved_layers(u)==1&&u[42*288+0]==2&&u[42*288+287]==2&&u[0]==8&&u[1]==0,"unobserved layer filled at the mean per-unit rate");
     const auto order=seed_order({},u);std::size_t first42=order.size();for(std::size_t i=0;i<order.size();++i)if(order[i]/288==42){first42=i;break;}
     check(first42<42*72+288,"unobserved layer interleaves with the observed ones");
     std::vector<std::uint64_t> none(43*288),all(43*288,1);check(fill_unobserved_layers(none)==0&&fill_unobserved_layers(all)==0,"no census or a full census: unchanged");
     // barely sampled (a few decode tokens vs thousands of prefill rows) counts as unobserved too
     auto t=u;for(unsigned e=0;e<288;++e)t[42*288+e]=(e<8)?1:0;check(fill_unobserved_layers(t)==1&&t[42*288+100]==2,"under-sampled layer filled at the mean");}
    // Priority: residency first, then most routed (ties: round-robin order), then the rest.
    std::vector<std::uint64_t> c(43*288);c[5*288+3]=9;c[0*288+200]=4;c[7]=4;
    auto o=seed_order({100,5*288+3},c);check(o.size()==43ULL*288&&o[0]==100&&o[1]==5*288+3&&o[2]==7&&o[3]==200&&o[4]==0&&o[5]==288,"seed order: residency, counts (ties round-robin), then round-robin");
    {std::vector<bool> seen(o.size());for(auto k:o){check(!seen[k],"seed order: every key once");seen[k]=true;}}
    auto t=tier_assignment(b,43ULL*288,o);for(std::size_t i=0;i<b.resident;++i)check(t[o[i]]==0,"seed head = resident tier keys");
}
// --profile default|fast|quality (profile.hpp, the DS RTX resolver's structure): default = today, every exact
// lever on; predicted prefetch pending everywhere until proven; fast adds pending bit-changing rows; quality = default;
// CLI > environment > profile > builtin; a pending column never refuses; the JSON is the receipt's "profile" object.
void profile_tests(){
    namespace rp=strix::glm::rtxpf::profile;
    auto lookup=[](std::vector<std::pair<std::string,std::string>> vars){return rp::Lookup([vars](const char* k)->const char*{
        for(auto& [n,v]:vars)if(n==k)return v.c_str();return nullptr;});};
    auto run=[&](std::vector<std::string> args,std::vector<std::pair<std::string,std::string>> env={}){rp::Inputs in;in.args=std::move(args);in.env=lookup(std::move(env));return rp::resolve(rp::App::decode_service,in);};
    const auto def=run({});
    check(def.name==rp::Name::standard&&def.qualification()=="qualified","no --profile = default, qualified");
    for(const char* l:{"expert_tier","placement","eviction_policy","spare_share","row_tile","prefill_tile_crossover","cuda_graphs","restore_decode_residency","flush_ready","prefill_overlap","decode_row_kernels"}){
        auto* e=def.find(l);check(e&&e->status==rp::Status::profile&&e->lever->kind==rp::Kind::exact,(std::string("default: exact lever on: ")+l).c_str());}
    check(def.find("expert_tier")->value=="tierstream"&&def.find("eviction_policy")->value=="auto"&&def.find("spare_share")->value=="5"&&def.find("row_tile")->value=="auto",
          "default values = today's service defaults");
    check(def.find("prefill_tile_crossover")->value=="0"&&run({"--profile","fast"}).find("prefill_tile_crossover")->value=="0","crossover 0 (allocated tile) in every profile");
    {const auto c=run({"--prefill-tile-crossover","8192"});check(c.find("prefill_tile_crossover")->status==rp::Status::explicit_cli&&c.find("prefill_tile_crossover")->value=="8192","--prefill-tile-crossover explicit");}
    auto* pp=def.find("predicted_prefetch");
    check(pp&&pp->status==rp::Status::pending&&pp->value=="0"&&pp->wanted=="1","predicted prefetch pending in default: its builtin (off) runs");
    check(!def.find("prefill_tile_large")&&!def.find("dsa_index_mma"),"default leaves the fast-only rows alone");
    check(def.pending()==std::vector<std::string>{"predicted_prefetch","skip_empty_expert_groups","persistent_dense_fp16","mtp_self_spec","lookup_drafts"},"default pending list");
    const auto fast=run({"--profile","fast"});
    check(fast.pending()==std::vector<std::string>{"predicted_prefetch","skip_empty_expert_groups","persistent_dense_fp16","mtp_self_spec","lookup_drafts","prefill_tile_large","dsa_index_mma"},"fast: every fast-only row pending (none qualified)");
    check(fast.find("prefill_tile_large")->value=="off"&&fast.find("dsa_index_mma")->value=="fp32"&&fast.find("dsa_index_mma")->lever->kind==rp::Kind::bit_changing,
          "fast pending rows run their builtin form");
    const auto quality=run({"--profile","quality"});
    check(quality.entries.size()==def.entries.size(),"quality = default rows");
    for(std::size_t i=0;i<def.entries.size();++i)
        check(quality.entries[i].lever==def.entries[i].lever&&quality.entries[i].value==def.entries[i].value&&quality.entries[i].status==def.entries[i].status,
              "quality = default values and statuses");
    check(!quality.notices().empty()&&quality.notices().front().find("no higher-precision expert pack")!=std::string::npos,"quality states it runs default");
    // explicit CLI overrides (the last of a lever's options wins; a lever option's value is never read as a flag)
    const auto cli=run({"--profile","fast","--tier-policy","scored","--row-tile","4096","--no-graphs","--tier-policy","lru","--tier-spare-slots","96"});
    check(cli.find("eviction_policy")->status==rp::Status::explicit_cli&&cli.find("eviction_policy")->value=="lru","--tier-policy explicit");
    check(cli.find("row_tile")->value=="4096"&&cli.find("cuda_graphs")->value=="off"&&cli.find("cuda_graphs")->status==rp::Status::explicit_cli,"--row-tile / --no-graphs explicit");
    check(cli.find("spare_share")->status==rp::Status::explicit_cli&&cli.find("spare_share")->value=="96","--tier-spare-slots explicit (overrides the 5% share)");
    check(run({"--tier-predict","1"}).find("predicted_prefetch")->status==rp::Status::explicit_cli,"explicit --tier-predict is an expert override, not pending");
    // the exact decode levers: pending everywhere (builtin 0 runs), explicit by CLI or environment, run by --allow-pending
    {const auto allowed=run({"--allow-pending"});
     for(const char* l:{"skip_empty_expert_groups","persistent_dense_fp16","mtp_self_spec","lookup_drafts"}){auto* e=def.find(l);
        check(e&&e->status==rp::Status::pending&&e->value=="0"&&e->wanted=="1"&&e->lever->kind==rp::Kind::exact,(std::string("pending exact lever: ")+l).c_str());
        check(allowed.find(l)->status==rp::Status::unqualified&&allowed.find(l)->value=="1",(std::string("--allow-pending runs ")+l).c_str());}}
    {const auto c=run({"--skip-empty-expert-groups","1"});check(c.find("skip_empty_expert_groups")->status==rp::Status::explicit_cli,"--skip-empty-expert-groups explicit");
     check(def.find("prefill_overlap")->status==rp::Status::profile&&def.find("prefill_overlap")->value=="1"&&fast.find("prefill_overlap")->value=="1"&&
           quality.find("prefill_overlap")->value=="1","prefill_overlap on in every profile");
     const auto o=run({"--prefill-overlap","0"});check(o.find("prefill_overlap")->status==rp::Status::explicit_cli&&o.find("prefill_overlap")->value=="0","--prefill-overlap 0 turns it off");
     const auto e=run({},{{"STRIX_GLM_RTX_PERSISTENT_DENSE_FP16","1"}});check(e.find("persistent_dense_fp16")->status==rp::Status::explicit_env&&e.find("persistent_dense_fp16")->value=="1","STRIX_GLM_RTX_PERSISTENT_DENSE_FP16=1 explicit");}
    // prompt-lookup drafts: longest earlier suffix match (most recent first), up to k following tokens
    {const std::vector<std::uint32_t> s{5,1,2,3,9,9,1,2,3,7,4,1,2,3};
     check(lookup_draft(s,3)==std::vector<std::uint32_t>{7,4,1},"lookup: most recent match of the longest suffix");
     check(lookup_draft(s,8,3,3)==std::vector<std::uint32_t>{7,4,1,2,3},"lookup: drafts stop at the end of the history");
     check(lookup_draft(std::vector<std::uint32_t>{1,2,3,4,5},3).empty(),"lookup: no repeated suffix, no draft");
     check(lookup_draft(std::vector<std::uint32_t>{8,8,8,8,8},2)==std::vector<std::uint32_t>{8},"lookup: overlapping match");}
    // decode_row_kernels: on in every profile; 0, 1 or a '+' subset; the runtime switches follow the value
    check(def.find("decode_row_kernels")->value=="1"&&fast.find("decode_row_kernels")->value=="1","decode_row_kernels on by default");
    check(run({"--mtp-self-spec","0"}).find("mtp_self_spec")->status==rp::Status::explicit_cli&&run({"--mtp-self-spec","0"}).find("mtp_self_spec")->value=="0","--mtp-self-spec 0 turns it off");
    check(run({},{{"STRIX_GLM_RTX_ROW_KERNELS","0"}}).find("decode_row_kernels")->value=="off","STRIX_GLM_RTX_ROW_KERNELS=0 turns it off");
    {const auto c=run({"--decode-row-kernels","moe+router"});check(c.find("decode_row_kernels")->status==rp::Status::explicit_cli&&c.find("decode_row_kernels")->value=="moe+router","--decode-row-kernels explicit");
     DecodeOptions d;set_row_kernels_value(d,"moe+router");check(d.row_moe&&!d.row_hc&&d.row_router&&row_kernels_name(d)=="moe+router","row kernel subset");
     set_row_kernels_value(d,"1");check(d.row_moe&&d.row_hc&&d.row_router&&row_kernels_name(d)=="1","row kernels all");
     set_row_kernels_value(d,"0");check(!d.row_moe&&!d.row_hc&&!d.row_router&&row_kernels_name(d)=="0","row kernels none");
     set_row_kernels_value(d,"1");check(!d.row_r2,"row kernels 1 = round one");
     set_row_kernels_value(d,"2");check(d.row_moe&&d.row_dsa&&d.row_r2&&row_kernels_name(d)=="2","row kernels 2 = round two");
     set_row_kernels_value(d,"moe+r2");check(d.row_moe&&!d.row_kda&&d.row_r2&&row_kernels_name(d)=="moe+r2","row kernel subset with r2");
     bool threw=false;try{set_row_kernels_value(d,"moe+gemm");}catch(const std::exception&){threw=true;}check(threw,"unknown row kernel refused");}
    // restore_decode_residency: placement only, on in every profile; CLI / environment override
    check(def.find("restore_decode_residency")->value=="on"&&fast.find("restore_decode_residency")->value=="on"&&def.find("restore_decode_residency")->status==rp::Status::profile,"restore_decode_residency on");
    {const auto off=run({},{{"STRIX_GLM_RTX_RESTORE_RESIDENCY","0"}});check(off.find("restore_decode_residency")->status==rp::Status::explicit_env&&off.find("restore_decode_residency")->value=="off","STRIX_GLM_RTX_RESTORE_RESIDENCY=0");}
    // flush_ready: on in every profile; --tier-flush-ready (CLI) beats STRIX_GLM_RTX_FLUSH_READY (env)
    check(def.find("flush_ready")->value=="on"&&fast.find("flush_ready")->value=="on"&&quality.find("flush_ready")->value=="on","flush_ready on in all profiles");
    {const auto env_off=run({},{{"STRIX_GLM_RTX_FLUSH_READY","0"}});auto* e=env_off.find("flush_ready");
     check(e&&e->status==rp::Status::explicit_env&&e->value=="off","STRIX_GLM_RTX_FLUSH_READY=0 turns it off");
     const auto cli_on=run({"--tier-flush-ready","1"},{{"STRIX_GLM_RTX_FLUSH_READY","0"}});auto* c=cli_on.find("flush_ready");
     check(c&&c->status==rp::Status::explicit_cli&&c->value=="1","--tier-flush-ready wins over the environment");}
    // environment selection, refusal of bad names, --allow-pending
    check(run({},{{"STRIX_GLM_RTX_PROFILE","fast"}}).name==rp::Name::fast&&run({"--profile","quality"},{{"STRIX_GLM_RTX_PROFILE","fast"}}).name==rp::Name::quality,
          "STRIX_GLM_RTX_PROFILE selects; --profile wins");
    for(auto args:{std::vector<std::string>{"--profile","turbo"},std::vector<std::string>{"--profile"}}){
        bool refused=false;try{(void)run(args);}catch(const std::invalid_argument&){refused=true;}check(refused,"bad or missing --profile refused");}
    {bool refused=false;try{(void)run({},{{"STRIX_GLM_RTX_PROFILE","x"}});}catch(const std::invalid_argument&){refused=true;}check(refused,"bad STRIX_GLM_RTX_PROFILE refused");}
    const auto bench=run({"--profile","fast","--allow-pending"});
    check(bench.find("prefill_tile_large")->status==rp::Status::unqualified&&bench.find("prefill_tile_large")->value=="8192"&&
          bench.find("predicted_prefetch")->status==rp::Status::unqualified&&bench.find("dsa_index_mma")->status==rp::Status::pending,
          "--allow-pending runs pending rows that are in this build only (dsa_index_mma stays pending)");
    check(bench.qualification()=="unqualified: predicted_prefetch,skip_empty_expert_groups,persistent_dense_fp16,mtp_self_spec,lookup_drafts,prefill_tile_large","the receipt is stamped unqualified");
    check(run({},{{"STRIX_GLM_RTX_PROFILE_ALLOW_PENDING","1"}}).allow_pending,"STRIX_GLM_RTX_PROFILE_ALLOW_PENDING=1");
    // stderr notices: one line per pending row, one for an unqualified run
    auto n=fast.notices();check(n.size()==7&&n[0].rfind("profile fast: predicted_prefetch pending, not used (",0)==0,"pending stderr lines");
    check(bench.notices().back()=="profile fast: unqualified: predicted_prefetch,skip_empty_expert_groups,persistent_dense_fp16,mtp_self_spec,lookup_drafts,prefill_tile_large (--allow-pending: benchmarks only)","unqualified stderr line");
    // the JSON (--print-effective-config / the first receipt event's "profile")
    auto j=strix::serve::parse_json(fast.json([](std::string_view l){return l=="row_tile"?std::string("auto (planner rule)"):std::string();}));
    check(j.find("profile")->as_string()=="fast"&&j.find("app")->as_string()=="decode-service"&&j.find("qualification")->as_string()=="qualified"&&!j.find("allow_pending")->as_bool(),
          "JSON header fields");
    auto* levers=j.find("levers");check(levers&&levers->as_object().size()==fast.entries.size(),"JSON: every row");
    auto* row=levers->find("row_tile");check(row->find("effective")->as_string()=="auto (planner rule)"&&row->find("switch")->as_string()=="--row-tile"&&
          row->find("kind")->as_string()=="exact"&&row->find("status")->as_string()=="profile","JSON row fields");
    check(levers->find("dsa_index_mma")->find("wanted")->as_string()=="mma"&&levers->find("dsa_index_mma")->find("kind")->as_string()=="bit-changing","JSON pending row");
    check(j.find("pending")->as_array().size()==7&&j.find("unqualified")->as_array().empty(),"JSON pending / unqualified lists");
    check(strix::serve::parse_json(quality.json()).find("note")!=nullptr,"quality JSON states quality = default");
    std::cout<<"INFO profile default: "<<def.json().size()<<" B JSON, pending "<<def.pending().size()<<"; fast pending "<<fast.pending().size()<<"\n";
}
// WDDM host commit split, on the measured full-card receipt (health):
// raw private 175587450880 B, ledger device 98701794188 B of which the measured
// context + other processes part is 2663841792 B; tracked pinned + pageable
// 78093165120 + 940605440 B. Our host commit must be the tracked host bytes
// within 1 GB (residual), under the 111 GiB cap the raw value failed.
void host_commit_tests(){
    const auto c=host_commit(175587450880ULL,98701794188ULL,2663841792ULL);
    check(c.device_backing==98701794188ULL-2663841792ULL&&c.host==175587450880ULL-c.device_backing,"host commit = private - our device bytes");
    const std::int64_t residual=std::int64_t(c.host)-std::int64_t(78093165120ULL+940605440ULL);
    check(residual>=0&&residual<(1LL<<30),"full-card residual (untracked host commit) below 1 GiB");
    check(c.host<=111ULL<<30&&c.raw_private>111ULL<<30,"honest host commit fits 111 GiB; the raw private bytes do not");
    std::cout<<"INFO full-card WDDM split: raw private "<<c.raw_private<<" B = host "<<c.host<<" B + device backing "<<c.device_backing<<" B; residual over tracked host "<<residual<<" B\n";
    auto z=host_commit(1000,0,0);check(z.device_backing==0&&z.host==1000,"no ledger device bytes: raw");
    auto ctx=host_commit(1000,500,700);check(ctx.device_backing==0&&ctx.host==1000,"context part above the ledger total: nothing subtracted");
    auto big=host_commit(100,500,0);check(big.host==0,"backing above private saturates at 0");
    // Gate: max(sampled, OS peak commit - peak attributable backing). Full card: the OS
    // term (79.55 GB) is the gate; a 40 GB temporary between samples is caught.
    const std::uint64_t sampled=79151440448ULL,os=175587450880ULL;
    check(host_gate_peak(sampled,os,c.device_backing)==c.host&&c.host<=111ULL<<30,"full-card gate peak fits 111 GiB");
    check(host_gate_peak(sampled,os+(40ULL<<30),c.device_backing)>111ULL<<30,"an untracked 40 GiB temporary fails the gate");
    check(host_gate_peak(sampled,10,c.device_backing)==sampled,"the sampled peak is never lowered");
    // A device-wide graph estimate (may include another process) is not discounted:
    // the caller passes the ledger device total minus context AND graph estimates.
    const std::uint64_t graph=2ULL<<30;auto g=host_commit(175587450880ULL,98701794188ULL-graph,2663841792ULL);
    check(g.host==c.host+graph,"graph estimates stay on the host side (conservative)");
    // Receipt history: an earlier OS-derived gate peak never enters the sampled peak.
    HostPeakHistory h;check(!h.known,"empty history unknown");
    h.add_sample(79ULL<<30);h.add_os_bound(200ULL<<30,96ULL<<30);check(h.known&&h.sampled==79ULL<<30&&h.gate==104ULL<<30,"OS bound raises the gate only");
    h.add_sample(80ULL<<30);h.add_os_bound(150ULL<<30,96ULL<<30);check(h.sampled==80ULL<<30&&h.gate==104ULL<<30,"sampled stays sampled; the gate keeps its history");
    h.add_sample(110ULL<<30);check(h.sampled==110ULL<<30&&h.gate==110ULL<<30,"a larger sample raises both");
    HostPeakHistory os_only;os_only.add_os_bound(100,40);check(os_only.known&&os_only.sampled==0&&os_only.gate==60,"OS-only history: gate known, nothing sampled");
}
// ---- Identity-bound opening contract: junction swap between authentication and cache open.
void make_junction(const std::filesystem::path& link,const std::filesystem::path& target){
    std::filesystem::create_directory(link);
    struct Reparse {DWORD tag;USHORT length,reserved,sub_offset,sub_length,print_offset,print_length;wchar_t paths[4096];} data{};
    auto native=std::filesystem::absolute(target).make_preferred().wstring();auto substitute=L"\\??\\"+native;auto print=native;
    data.tag=IO_REPARSE_TAG_MOUNT_POINT;data.sub_length=USHORT(substitute.size()*2);data.print_offset=data.sub_length+2;data.print_length=USHORT(print.size()*2);
    data.length=USHORT(8+data.print_offset+data.print_length+2);check(data.length+8<=sizeof data,"junction buffer");
    std::memcpy(data.paths,substitute.c_str(),data.sub_length+2);std::memcpy(reinterpret_cast<char*>(data.paths)+data.print_offset,print.c_str(),data.print_length+2);
    WinHandle h;h.h=CreateFileW(link.c_str(),GENERIC_WRITE,0,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT|FILE_FLAG_BACKUP_SEMANTICS,nullptr);
    DWORD bytes=0;check(h.h!=INVALID_HANDLE_VALUE&&DeviceIoControl(h.h,FSCTL_SET_REPARSE_POINT,&data,data.length+8,nullptr,0,&bytes,nullptr),"create junction");
}
void binding_tests(const std::filesystem::path& build){
    auto root=build/"decode-host-binding";std::filesystem::remove_all(root);std::filesystem::create_directories(root/"real");std::filesystem::create_directories(root/"other");
    const std::size_t n=1<<20;std::vector<char> genuine(n,'G'),impostor(n,'X');
    std::ofstream(root/"real"/"units.bin",std::ios::binary).write(genuine.data(),n);std::ofstream(root/"other"/"units.bin",std::ios::binary).write(impostor.data(),n);
    make_junction(root/"alias",root/"real");
    auto read_first=[&](const std::filesystem::path& p){std::ifstream in(p,std::ios::binary);char c=0;in.read(&c,1);return c;};
    {   // Authentication through the alias (the operator's path), then binding.
        VerifiedFile held(root/"alias"/"units.bin",n);BoundUnitFile bound(held);
        check(bound.path.string().find("alias")==std::string::npos&&bound.path.filename()=="units.bin"&&bound.ancestors()>=3,"bound to the canonical final path, not the alias");
        // Swap the junction between authentication and the cache open (allowed: the alias is not bound).
        check(RemoveDirectoryW((root/"alias").c_str())!=0,"remove alias junction");make_junction(root/"alias",root/"other");
        check(read_first(root/"alias"/"units.bin")=='X',"the operator's name now resolves to same-size impostor bytes");
        bool redirected=false;{WinHandle h;h.h=CreateFileW((root/"alias"/"units.bin").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,0,nullptr);
            try{same_file_object(held.held_handle(),h.h);}catch(const std::exception&){redirected=true;}}
        check(redirected,"an unbound reopen would consume a different object");
        // The cache opens the bound name: same object, authenticated bytes.
        bound.verify_serving(held);check(read_first(bound.path)=='G',"bound name serves the authenticated bytes after the swap");
        // The bound namespace cannot change while held: ancestors and the file refuse rename/replacement.
        check(!MoveFileExW((root/"real").c_str(),(root/"real-moved").c_str(),0),"bound ancestor rename refused");
        check(!MoveFileExW(root.c_str(),(build/"decode-host-binding-moved").c_str(),0),"higher bound ancestor rename refused");
        check(!MoveFileExW((root/"other"/"units.bin").c_str(),bound.path.c_str(),MOVEFILE_REPLACE_EXISTING),"bound file replacement refused");
        check(!DeleteFileW(bound.path.c_str()),"bound file deletion refused");
        check(std::string(bound.describe()).find("ancestor directories held")!=std::string::npos,"health binding statement");
    }
    // Released: the same rename now succeeds, so the refusals above came from the binding.
    check(MoveFileExW((root/"real").c_str(),(root/"real-moved").c_str(),0)!=0,"rename succeeds once the binding is released");
    RemoveDirectoryW((root/"alias").c_str());std::filesystem::remove_all(root);
}
// ---- Encoding: the cache receives the bound name as lossless UTF-8. Two sibling
// directories whose names collide under ANSI-vs-UTF-8 confusion: "caf\u00C3\u00A9"
// (genuine) and "caf\u00E9" (same-size impostor). Under code page 1252 the ANSI
// bytes of the genuine name, decoded as UTF-8 by the library, name the impostor.
void encoding_tests(const std::filesystem::path& build){
    auto root=build/"decode-host-encoding";std::filesystem::remove_all(root);
    const auto mojibake=root/std::wstring(L"caf\u00C3\u00A9"),accented=root/std::wstring(L"caf\u00E9");
    std::filesystem::create_directories(mojibake);std::filesystem::create_directories(accented);
    const std::size_t n=1<<20;std::string genuine(n,'G'),impostor(n,'X');
    std::ofstream(mojibake/"units.bin",std::ios::binary).write(genuine.data(),n);std::ofstream(accented/"units.bin",std::ios::binary).write(impostor.data(),n);
    {
        VerifiedFile held(mojibake/"units.bin",n);BoundUnitFile bound(held);const auto u=bound.utf8();
        check(u.find("caf\xC3\x83\xC2\xA9\\units.bin")!=std::string::npos,"bound name serialized as UTF-8 with native backslashes");
        check(u.find('/')==std::string::npos,"no generic separators");
        const int m=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,u.data(),int(u.size()),nullptr,0);std::wstring back(std::size_t(m>0?m:0),L'\0');
        check(m>0&&MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,u.data(),int(u.size()),back.data(),m)==m&&back==bound.path.native(),"UTF-8 decodes back to the identical bound path");
        // The negative case uses explicit CP1252 bytes (what path.string() yields on a
        // cp1252 system), independent of this machine's active code page.
        const auto& w=bound.path.native();BOOL lossy=FALSE;
        const int an=WideCharToMultiByte(1252,WC_NO_BEST_FIT_CHARS,w.data(),int(w.size()),nullptr,0,nullptr,&lossy);std::string ansi(std::size_t(an>0?an:0),'\0');
        check(an>0&&!lossy&&WideCharToMultiByte(1252,WC_NO_BEST_FIT_CHARS,w.data(),int(w.size()),ansi.data(),an,nullptr,&lossy)==an&&!lossy,"explicit CP1252 form of the bound name");
        check(ansi!=u&&ansi.find("caf\xC3\xA9\\units.bin")!=std::string::npos,"the CP1252 form differs from the UTF-8 form for this name");
#ifdef GLM_TIERSTREAM_PARSER
        // tierstream's own public reader (ss_mfile_open: its UTF-8 decode and open)
        // given the descriptor's source path, compared with the authenticated bytes.
        auto s=synthetic();ExpertCatalog c(s.tensors,s.g);const auto desc=tierstream_descriptor(c,u,std::string(64,'b')).dump();
        const auto sha=unit_sha256(std::as_bytes(std::span(genuine)));std::string got;
        auto error=tierstream_source_read(desc,nullptr,sha,4096,&got);
        check(error.empty()&&got==genuine.substr(0,4096),("tierstream's reader opens the bound UTF-8 name and reads the authenticated bytes ("+error+")").c_str());
        bound.verify_serving(held);
        {   // tierstream decodes as UTF-8 on every code page, so the CP1252 bytes name the impostor.
            std::string alt;auto e2=tierstream_source_read({},ansi.c_str(),sha,4096,&alt);
            check(!e2.empty()||alt!=genuine.substr(0,4096),"the CP1252 name would not have served the authenticated bytes");
            std::cout<<"INFO CP1252 form of the bound name (active code page "<<GetACP()<<"), via tierstream's reader: "<<(e2.empty()?(alt[0]=='X'?"opened the impostor":"opened other bytes"):"refused: "+e2)<<"\n";
        }
#else
        std::cout<<"SKIP tierstream reader encoding check not configured (GLM_TIERSTREAM_SRC)\n";
#endif
    }
    // Alternate data streams are refused (operator path and held final path).
    const auto plain=root/"ads.bin";std::ofstream(plain,std::ios::binary).write(impostor.data(),n);
    const auto stream=std::filesystem::path(plain.native()+L":alt");std::ofstream(stream,std::ios::binary).write(genuine.data(),n);
    rejects([&]{refuse_stream_path(stream);},"operator path naming an alternate data stream refused");
    rejects([&]{refuse_stream_path(std::filesystem::path(L"\\\\?\\")+=stream.native());},"prefixed alternate data stream path refused");
    refuse_stream_path(plain);refuse_stream_path(std::filesystem::path(L"\\\\?\\")+=plain.native());
    bool bound_stream=true;try{VerifiedFile held(stream,n);BoundUnitFile bound(held);}catch(const std::exception&){bound_stream=false;}
    std::cout<<"INFO alternate data stream handle: "<<(bound_stream?"final path omits the stream (refused by the operator-path check)":"refused by the held final-path check")<<"\n";
    std::filesystem::remove_all(root);
}
// ---- Lifetime: the cache always closes before its binding is released, also
// when the owning backend's constructor unwinds after the cache was installed.
struct NullCache:TierCacheApi {
    void begin()override{}void enqueue_begin(void*)override{}void enqueue_export(unsigned,const std::int32_t*,unsigned,void*)override{}void enqueue_wait(unsigned,void*)override{}void enqueue_end(void*)override{}
    void end()override{}TierView view()const override{return {};}void reset_open()override{}Json stats()override{return Json::object();}};
void lifetime_tests(const std::filesystem::path& build){
    std::vector<std::string> log;
    struct LogBinding {std::vector<std::string>* log;explicit LogBinding(std::vector<std::string>* l):log(l){}~LogBinding(){log->push_back("binding released");}};
    struct LogCache final:NullCache {std::vector<std::string>* log;explicit LogCache(std::vector<std::string>* l):log(l){}~LogCache()override{log->push_back("cache closed");}};
    // Mirrors TierstreamTier: the BoundCache is the first member, the body may throw.
    struct Owner {BoundCache<LogBinding> owned;TierCacheApi* api;Owner(BoundCache<LogBinding>&& o,bool fail):owned(std::move(o)),api(&owned.cache()){if(fail)throw std::runtime_error("injected failure after cache install");}};
    auto make=[&]{return BoundCache<LogBinding>(std::make_unique<LogBinding>(&log),[&](LogBinding&){log.push_back("cache opened");return std::unique_ptr<TierCacheApi>(std::make_unique<LogCache>(&log));});};
    bool thrown=false;try{Owner o(make(),true);}catch(const std::runtime_error&){thrown=true;}
    check(thrown&&log==std::vector<std::string>{"cache opened","cache closed","binding released"},"constructor unwinding closes the cache before releasing the binding");
    log.clear();{Owner o(make(),false);(void)o;}
    check(log==std::vector<std::string>{"cache opened","cache closed","binding released"},"normal teardown closes the cache first");
    log.clear();rejects([&]{BoundCache<LogBinding> b(std::make_unique<LogBinding>(&log),[&](LogBinding&)->std::unique_ptr<TierCacheApi>{throw std::runtime_error("injected open failure");});},"failing open");
    check(log==std::vector<std::string>{"binding released"},"a failing open releases the binding with no cache");
    // Real binding: while the cache is being closed the bound ancestors are still held.
    auto root=build/"decode-host-lifetime";std::filesystem::remove_all(root);std::filesystem::create_directories(root/"real");
    const std::size_t n=1<<16;std::string bytes(n,'G');std::ofstream(root/"real"/"units.bin",std::ios::binary).write(bytes.data(),n);
    struct HeldCheckCache final:NullCache {std::filesystem::path dir;bool* held_at_close;HeldCheckCache(std::filesystem::path d,bool* h):dir(std::move(d)),held_at_close(h){}
        ~HeldCheckCache()override{*held_at_close=!MoveFileExW(dir.c_str(),(dir.native()+L"-moved").c_str(),0);}};
    struct RealOwner {BoundCache<UnitBinding> owned;RealOwner(BoundCache<UnitBinding>&& o):owned(std::move(o)){throw std::runtime_error("injected failure after cache install");}};
    bool held_at_close=false;thrown=false;
    try{auto u=std::make_unique<UnitBinding>();u->lock=std::make_unique<VerifiedFile>(root/"real"/"units.bin",n);u->bound=std::make_unique<BoundUnitFile>(*u->lock);
        const auto dir=u->bound->path.parent_path();
        RealOwner o(BoundCache<UnitBinding>(std::move(u),[&](UnitBinding&){return std::unique_ptr<TierCacheApi>(std::make_unique<HeldCheckCache>(dir,&held_at_close));}));}
    catch(const std::runtime_error&){thrown=true;}
    check(thrown&&held_at_close,"bound ancestors still held while the cache closes on unwinding");
    check(MoveFileExW((root/"real").c_str(),(root/"real-moved").c_str(),0)!=0,"binding released after the cache closed");
    std::filesystem::remove_all(root);
}
void receipt_tests(const std::filesystem::path& receipt){
    const std::uint64_t unit=12648448;CacheCounters a,b,c2;b.lookups=344;b.misses=60;b.resurrected=6;b.covered=4;b.fetched=40;b.prefetched=10;b.bank_serves=2;b.file_reads=12;b.file_bytes=12*unit;
    c2=b;c2.lookups+=344;c2.misses+=20;c2.resurrected+=1;c2.covered+=2;c2.fetched+=15;c2.file_reads+=3;c2.file_bytes+=3*unit;
    auto steps=Json::array();for(auto d:{cache_delta(a,b,unit),cache_delta(b,c2,unit)}){auto st=Json::object();st.set("token",3);st.set("experts",d);steps.push(st);}
    auto t=aggregate_expert_totals(steps);auto n=[&](const char* k){auto* v=t.find(k);return v?std::uint64_t(v->as_double()):~0ULL;};
    check(n("lookups")==688&&n("misses")==80&&n("device_hits")==608&&n("resurrected")==7&&n("covered")==6&&n("prefetched")==10&&n("bank_serves")==2,"aggregated counters include every new field");
    check(n("ram_copies")==(40+10-12)+(15-3)&&n("file_reads")==15&&n("file_bytes")==15*unit&&n("h2d_bytes")==65*unit&&!t.find("free_slots"),"copy-source totals; gauges excluded");
    // Exact integers: no double round trip, overflow refused.
    auto one=[](Json v){auto st=Json::object();auto e=Json::object();e.set("lookups",v);st.set("experts",e);return st;};
    auto sum=[&](std::initializer_list<Json> vs){auto a=Json::array();for(auto& v:vs)a.push(one(v));return aggregate_expert_totals(a).find("lookups")->number_source();};
    check(sum({Json(std::uint64_t(9007199254740993ULL)),Json(std::uint64_t(1))})=="9007199254740994","exact above 2^53");
    check(sum({Json::number_text("9007199254740993")})=="9007199254740993","2^53+1 preserved");
    check(sum({Json(~std::uint64_t(0))})=="18446744073709551615","UINT64_MAX preserved");
    rejects([&]{sum({Json(~std::uint64_t(0)),Json(std::uint64_t(1))});},"UINT64_MAX + 1 overflow refused");
    for(const char* bad:{"18446744073709551616","1.5","-1","1e3","+2"})rejects([&]{sum({Json::number_text(bad)});},"non-uint64 counter text refused");
    if(receipt.empty())return;
    // Fixture publication: remove any previous artifact, then publish this run's.
    std::filesystem::create_directories(receipt.parent_path());std::filesystem::remove(receipt);
    auto arm=Json::object();arm.set("arm","cold");arm.set("steps",steps);arm.set("expert_totals",t);arm.set("published_by","glm_native_decode_host");
    arm.set("written_unix_s",std::uint64_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()));
    std::random_device rd;arm.set("run_nonce",std::to_string(rd())+std::to_string(rd()));
    auto tmp=receipt;tmp+=".partial";{std::ofstream out(tmp,std::ios::binary);out<<arm.dump();check(bool(out),"receipt artifact written");}
    std::filesystem::rename(tmp,receipt);check(std::filesystem::exists(receipt),"receipt artifact published");
}
void unit_file_tests(const std::filesystem::path& build){
    auto s=synthetic();ExpertCatalog c(s.tensors,s.g);auto dir=build/"decode-host-units";std::filesystem::remove_all(dir);std::filesystem::create_directories(dir);
    auto plan=plan_unit_file(c,dir/"glm-units.bin",{parse_stripe((dir/"d").string()+":4"),parse_stripe((dir/"e").string()+":even:4")},1ULL<<20);
    check(plan.bytes==2*288*c.unit_stride()&&plan.stride%4096==0,"primary bytes and 4 KiB alignment");
    std::uint64_t odd=0,even=0,chunks=(plan.bytes+(1<<20)-1)>>20;for(std::uint64_t k=0;k<chunks;++k)((k/4)%2?odd:even)+=std::min<std::uint64_t>(1<<20,plan.bytes-(k<<20));
    check(plan.stripes[0].bytes==odd&&plan.stripes[1].bytes==even&&odd+even==plan.bytes,"blocks:K / blocks-even:K stripe bytes");
    auto r=stripe_ranges(plan,plan.stripes[0]);check(r.find("schema")->as_string()=="tierstream/stripe-ranges-v1"&&r.find("ranges")->as_array()[0].find("policy")->as_string()=="blocks:4","stripe ranges JSON");
    auto pj=plan_json(plan,std::string(64,'b'));check(std::uint64_t(pj.find("total_write_bytes")->as_int())==plan.bytes+odd+even,"dry-run total bytes");
    rejects([&]{parse_stripe("D:4");},"stripe needs a directory");rejects([&]{parse_stripe("x:0");},"stripe K >= 1");
    std::vector<std::string> names{ExpertCatalog::blob_name(3),ExpertCatalog::blob_name(4)};
    auto locate=[&](const ExpertTensor& t){return SourceExtent{t.blob==names[0]?0u:1u,t.offset,t.bytes,0,0};};
    auto read=[&](unsigned f,std::uint64_t b,std::uint64_t n,void* d){std::memcpy(d,s.blobs.at(names[f]).data()+b,n);};
    std::vector<std::uint64_t> sizes{s.blobs.at(names[0]).size(),s.blobs.at(names[1]).size()};
    // Exclusive reservation: any existing output or .partial refuses the copy before any write.
    check(unit_file_outputs(plan).size()==6,"finals and partials are all outputs");
    {std::ofstream(partial_path(plan.manifest))<<"stale";rejects([&]{write_unit_file(c,plan,std::string(64,'b'),locate,sizes,read);},"stale sidecar partial refuses the copy");
        check(!std::filesystem::exists(partial_path(plan.primary))&&!std::filesystem::exists(plan.primary),"nothing written when refused");std::filesystem::remove(partial_path(plan.manifest));}
    {auto x=dir/"x.bin";std::ofstream(dir/"taken.bin")<<"keep";{ExclusiveFile e(x);rejects([&]{ExclusiveFile again(x);},"second reservation refused");e.write("abc",3);char got[3]{};e.read_at(0,got,3);check(std::memcmp(got,"abc",3)==0,"owned read-back");
            rejects([&]{e.publish(dir/"taken.bin");},"publish never replaces an existing file");}
        check(!std::filesystem::exists(x),"unpublished reservation removed");std::ifstream t(dir/"taken.bin");std::string keep;t>>keep;check(keep=="keep","existing file untouched");}
    auto manifest=write_unit_file(c,plan,std::string(64,'b'),locate,sizes,read);
    {std::ifstream mf(plan.manifest,std::ios::binary);std::string text((std::istreambuf_iterator<char>(mf)),{});
        check(sha256_hex(manifest.find("file_sha256")->as_string())&&manifest.find("manifest_sha256")->as_string()==strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(text)))),"manifest digest pin and whole-file digest");
        check(!std::filesystem::exists(partial_path(plan.primary))&&!std::filesystem::exists(partial_path(plan.manifest))&&!std::filesystem::exists(partial_path(plan.descriptor)),"partials published, none left");}
    check(std::filesystem::file_size(plan.primary)==plan.bytes&&std::filesystem::exists(plan.manifest)&&std::filesystem::exists(plan.descriptor),"unit file, manifest, descriptor written");
    check(verify_unit_file(c,plan.primary,manifest,std::string(64,'b'))==c.size(),"unit file verified");
    rejects([&]{write_unit_file(c,plan,std::string(64,'b'),locate,sizes,read);},"existing unit file never overwritten");
    {   // Startup authentication (the runtime's function): handle first, pinned manifest, contents.
        auto cache=build/"decode-host-hash-authority";std::filesystem::remove_all(cache);const auto pin=manifest.find("manifest_sha256")->as_string();
        bool journal=false;std::string first_how;
        {auto a1=authenticate_unit_file(plan.primary,c,std::string(64,'b'),pin,cache);check(a1.lock&&!a1.recalled&&a1.file_sha256==manifest.find("file_sha256")->as_string(),"first start hashes the whole file");
            first_how=a1.how();journal=a1.lock->journal_available();}
        check(first_how=="hashed through the held handle","health reports the hash truthfully");
        // Warm start recalls: the USN record with a journal, else the mtime-stamped no-USN record (weaker; named in health).
        {auto a2=authenticate_unit_file(plan.primary,c,std::string(64,'b'),pin,cache);
            check(a2.recalled&&a2.basis==(journal?"usn record":"no-usn record (mtime-stamped)")&&std::string(a2.how())==(journal?"authenticated record recalled (USN)":"authenticated record recalled (no USN: mtime-stamped record)"),"warm start recalls the authenticated record (USN, or the no-USN mtime-stamped record)");
            std::cout<<"INFO unit file warm start: "<<a2.how()<<" in "<<a2.ms<<" ms\n";}
        rejects([&]{authenticate_unit_file(plan.primary,c,std::string(64,'b'),std::string(64,'0'),cache);},"wrong manifest pin");
        rejects([&]{authenticate_unit_file(plan.primary,c,std::string(64,'d'),pin,cache);},"manifest bound to another model");
        rejects([&]{authenticate_unit_file(plan.primary,c,std::string(64,'b'),"",cache);},"pin required");
        {std::fstream f(plan.primary,std::ios::in|std::ios::out|std::ios::binary);f.seekp(std::streamoff(c.unit_offset(3,7)+c.at(3,7).unit.offset[1][0]+1));char x=0x11;f.write(&x,1);}
        rejects([&]{authenticate_unit_file(plan.primary,c,std::string(64,'b'),pin,cache);},"same-size corrupted unit file rejected at startup");
        std::filesystem::remove_all(cache);
    }
    {std::fstream f(plan.primary,std::ios::in|std::ios::out|std::ios::binary);f.seekp(std::streamoff(c.unit_offset(4,1)+c.at(4,1).unit.offset[0][0]+3));char x=0x55;f.write(&x,1);}
    rejects([&]{verify_unit_file(c,plan.primary,manifest,std::string(64,'b'));},"tampered unit fails verification");
#ifdef GLM_TIERSTREAM_LIBRARY
    // Real tierstream library, CPU backend: every admission rule runs, then "needs the CUDA backend".
    std::filesystem::remove(plan.primary);{std::ofstream f(plan.primary,std::ios::binary);std::vector<char> z(c.unit_stride());for(std::uint64_t k=0;k<c.size();++k)f.write(z.data(),std::streamsize(z.size()));}
    TierBudgetIn in;in.keys=c.size();in.unit_bytes=c.unit_stride();in.device_bytes=60*c.unit_stride();in.host_bytes=200*c.unit_stride();
    TierOpenConfig cfg;cfg.budget=tier_budget(in);check(cfg.budget.three_tier&&cfg.budget.disk>0,"probe exercises the disk tier");
    cfg.descriptor_json=tierstream_descriptor(c,utf8_path(plan.primary),std::string(64,'b')).dump();cfg.tiers=tier_assignment(cfg.budget,c.size(),{});cfg.unit_bytes=c.unit_stride();cfg.expected_gated=2;cfg.cuda=false;
    auto ok=tierstream_admission_probe(cfg);check(ok.first==8,("tierstream admits the GLM three-tier open (got "+std::to_string(ok.first)+": "+ok.second+")").c_str());
    auto no_disk=cfg;no_disk.budget.three_tier=false;no_disk.budget.staging=0;auto refused=tierstream_admission_probe(no_disk);check(refused.first!=8&&refused.first!=0,"file-homed units without dg_disk_tier refused");
    auto bad=cfg;bad.tiers[0]=2;auto r2=tierstream_admission_probe(bad);check(r2.first==-1,"tier counts must match the budget");
    std::cout<<"INFO tierstream CPU admission: "<<ok.second<<"\n";
    production_admission_tests();decode_budget_tests();
#else
    std::cout<<"SKIP tierstream library not configured\n";
#endif
    std::filesystem::remove_all(dir);
}
void slot_tests(){
    std::int32_t slot_of[2*288];std::uint32_t unit_base[3]{0,288,576};for(auto& v:slot_of)v=-1;slot_of[5]=7;slot_of[288+287]=3;bool absent=false;
    check(guarded_slot(slot_of,unit_base,0,99,0,5,&absent)==7&&!absent,"resident slot");
    check(guarded_slot(slot_of,unit_base,0,99,1,287,&absent)==3&&!absent,"second gated layer");
    check(guarded_slot(slot_of,unit_base,0,99,0,6,&absent)==99&&absent,"absent -> guard + flag");absent=false;
    check(guarded_slot(slot_of,unit_base,1,99,0,5,&absent)==99&&!absent,"abort -> guard");
    check(guarded_slot(slot_of,unit_base,0,99,0,288,&absent)==99&&!absent,"out of range -> guard");
    check(gated_ordinal(3)==0&&gated_ordinal(44)==41&&gated_ordinal(45)==42,"gated ordinals");rejects([&]{gated_ordinal(2);},"dense layer not gated");
}
void request_tests(){
    auto key=std::string(64,'a');auto body=[&](const std::string& extra){return strix::serve::parse_json("{\"tokens\":[1,2,3],\"max_new_tokens\":4,\"key\":\""+key+"\""+extra+"}");};
    auto r=parse_generate(body(",\"identity\":\"glm-dec-rtx-native-v1\",\"stop_token_ids\":[2]"),16);check(r.tokens.size()==3&&r.max_new==4&&r.stop.size()==1,"valid generate");
    check(rejects_code([&]{parse_generate(body(",\"temperature\":0.7"),16);})==400,"sampling field 400");
    try{parse_generate(body(",\"top_p\":0.9"),16);}catch(const RequestError& e){check(std::string(e.what()).find("top_p")!=std::string::npos,"field-specific message");}
    check(rejects_code([&]{parse_generate(body(",\"handoff\":{}"),16);})==400,"unknown field");
    check(rejects_code([&]{parse_generate(strix::serve::parse_json("{\"tokens\":[1,2],\"max_new_tokens\":1}"),16);})==401,"missing key");
    check(rejects_code([&]{parse_generate(body(",\"identity\":\"glm-pf-rtx-native-v1\""),16);})==400,"identity mismatch");
    check(rejects_code([&]{parse_generate(body(""),6);})==400,"prompt+max_new above capacity");parse_generate(body(""),7);
    check(rejects_code([&]{parse_generate(strix::serve::parse_json("{\"tokens\":[1,154880],\"max_new_tokens\":1,\"key\":\""+key+"\"}"),16);})==400,"token outside vocabulary");
    check(rejects_code([&]{parse_generate(strix::serve::parse_json("{\"tokens\":[1,2.5],\"max_new_tokens\":1,\"key\":\""+key+"\"}"),16);})==400,"non-integer token");
    check(rejects_code([&]{parse_generate(strix::serve::parse_json("{\"tokens\":[1,2],\"max_new_tokens\":0,\"key\":\""+key+"\"}"),16);})==400,"zero max_new");
    std::string many=",\"stop_token_ids\":[";for(int i=0;i<65;++i)many+=(i?",":"")+std::to_string(i);many+="]";check(rejects_code([&]{parse_generate(body(many),16);})==400,"stop list bound");
    check(rejects_code([&]{parse_generate(strix::serve::parse_json("[1,2]"),16);})==400,"non-object body");
}
int main(int argc,char** argv){try{
    if(argc<2||argc>3)throw std::runtime_error("usage: decode_host BUILD_DIR [RECEIPT_ARTIFACT]");
    session_tests();kda_schedule_tests();head_tests();catalog_tests();extent_tests(argv[1]);slot_tests();
#ifdef GLM_TIERSTREAM_LIBRARY
    tier_budget_tests();
#endif
    tier_mapping_tests();prefill_cache_tests();step_plan_tests();decode_timeline_tests();c1_schedule_tests();slot_census_tests();planned_residents_tests();mtp_tile_tests();prefill_tile_selection_tests();prefill_spares_tests();host_commit_tests();placement_tests();profile_tests();unit_file_tests(argv[1]);binding_tests(argv[1]);encoding_tests(argv[1]);lifetime_tests(argv[1]);receipt_tests(argc>2?std::filesystem::path(argv[2]):std::filesystem::path());request_tests();
    std::cout<<"PASS "<<checks<<" decode host checks: session/frontier, state bytes, KDA open-block replay model, DSA pools, argmax, expert catalog/units, cross-shard reads, guarded slots, generate requests\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
