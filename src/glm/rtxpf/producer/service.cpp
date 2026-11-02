// Resident Windows PRO worker. Build never launches this executable.
#include "sender.hpp"
#include "runtime.hpp"
#include "verified_io.hpp"
#include "policy.hpp"
#include "generate_request.hpp"
#include "expert_tier.hpp"
#include "decode_timeline.hpp"
#include "profile.hpp"
#include "schedule.hpp"
#include "decode_command.hpp"
#include "crash_report.hpp"
#include "strix/serve/json.hpp"
#include <atomic>
#include <cuda.h>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace p=strix::glm::rtxpf::producer;
namespace w=strix::glm::import;
using strix::serve::Json;
const Json& field(const Json& j,const char* name){auto* v=j.find(name);w::require(v!=nullptr,"missing request field");return *v;}
void fields(const Json& j,std::initializer_list<std::string> allowed){w::require(j.is_object(),"expected object");for(auto& [key,value]:j.as_object())w::require(std::find(allowed.begin(),allowed.end(),key)!=allowed.end(),"unsupported request field");}
struct AsyncSender {
    struct Job {unsigned layer,component;w::Bytes bytes;};
    std::unique_ptr<p::Sender> sender;std::mutex mutex;std::condition_variable cv;
    std::deque<Job> jobs;bool closed=false;std::exception_ptr error;std::future<void> worker;
    explicit AsyncSender(std::unique_ptr<p::Sender> s):sender(std::move(s)),worker(std::async(std::launch::async,[this]{
        try {while(true){Job job;{std::unique_lock lock(mutex);cv.wait(lock,[&]{return closed||!jobs.empty();});if(jobs.empty())break;job=std::move(jobs.front());jobs.pop_front();cv.notify_all();}sender->frame(job.layer,job.component,job.bytes);}}
        catch(...){std::lock_guard lock(mutex);error=std::current_exception();closed=true;jobs.clear();cv.notify_all();}
    })){}
    void submit(unsigned l,unsigned c,std::span<const std::byte> b){std::unique_lock lock(mutex);cv.wait(lock,[&]{return jobs.size()<2||closed;});if(error)std::rethrow_exception(error);w::require(!closed,"sender closed");jobs.push_back({l,c,{b.begin(),b.end()}});cv.notify_all();}
    void drain(){{std::lock_guard lock(mutex);closed=true;cv.notify_all();}if(worker.valid())worker.get();if(error)std::rethrow_exception(error);}
    ~AsyncSender(){try{drain();}catch(...){} }
};
struct Http {
    std::string method,path,body;
};
// One bounded HTTP request per connection. No chunked bodies, pipelining,
// duplicate Content-Length, implicit tokenization or unsupported generation fields.
Http request(w::Socket fd){
    std::string data;char c;while(data.find("\r\n\r\n")==std::string::npos){w::require(data.size()<16384,"HTTP header too large");w::require(recv(fd,&c,1,0)==1,"HTTP header EOF");data+=c;}
    auto end=data.find("\r\n");auto line=data.substr(0,end);auto a=line.find(' '),b=line.rfind(' ');
    w::require(a!=std::string::npos&&a<b&&line.substr(b+1)=="HTTP/1.1","HTTP request line");
    Http r{line.substr(0,a),line.substr(a+1,b-a-1),{}};bool seen=false;std::size_t size=0;
    for(auto off=end+2;off+2<data.size();){auto next=data.find("\r\n",off);if(next==off)break;auto header=data.substr(off,next-off);auto colon=header.find(':');w::require(colon!=std::string::npos,"HTTP header");auto key=header.substr(0,colon);for(char& x:key)if(x>='A'&&x<='Z')x+=32;
        if(key=="content-length"){w::require(!seen,"duplicate length");seen=true;auto value=header.substr(colon+1);auto first=value.find_first_not_of(" \t");w::require(first!=std::string::npos,"empty length");value=value.substr(first);w::require(value.find_first_not_of("0123456789")==std::string::npos&&value.size()<10,"invalid length");size=std::stoull(value);w::require(size<=4*1024*1024,"HTTP body too large");}
        w::require(key!="transfer-encoding","transfer encoding unsupported");off=next+2;
    }
    if(r.method=="POST")w::require(seen,"Content-Length required");
    r.body.resize(size);if(size)w::io(fd,std::as_writable_bytes(std::span(r.body)),false);return r;
}
std::vector<std::uint32_t> read_prompt(const std::string& file,unsigned capacity){
    std::ifstream in(file,std::ios::binary);w::require(bool(in),"open prompt file");std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});
    w::require(!in.bad()&&bytes.size()%4==0&&bytes.size()<=std::size_t(capacity)*4,"prompt file extent");std::vector<std::uint32_t> t(bytes.size()/4);std::memcpy(t.data(),bytes.data(),bytes.size());return t;
}
// Per-rank payload SHA-256 in release order (no wire/file headers) and the
// component extents/order, optionally also writing state-r{0,1}.bin exports.
Json session_digest(p::Runtime& runtime,const std::vector<std::uint32_t>& tokens,const std::string& dir={}){
    strix::Sha256Fast sha[2];auto components=Json::array();std::ofstream out[2];std::uint64_t bytes[2]{};
    if(!dir.empty()){std::filesystem::create_directories(dir);for(unsigned r=0;r<2;++r){out[r].open(std::filesystem::path(dir)/("state-r"+std::to_string(r)+".bin"),std::ios::binary);w::Bytes header;w::number(header,tokens.size()-1);w::number(header,tokens.size());out[r].write(reinterpret_cast<const char*>(header.data()),header.size());}}
    runtime.export_session([&](unsigned rank,unsigned layer,unsigned component,std::span<const std::byte> b){sha[rank].update(b);bytes[rank]+=b.size();
        if(rank==0){auto c=Json::array();c.push(int(layer));c.push(int(component));c.push(std::uint64_t(b.size()));components.push(c);}
        if(out[rank].is_open()){out[rank].write(reinterpret_cast<const char*>(b.data()),b.size());w::require(bool(out[rank]),"state export write");}});
    auto j=Json::object();for(unsigned r=0;r<2;++r){j.set("rank"+std::to_string(r)+"_sha256",strix::model_content_identity_hex(sha[r].finalize()));j.set("rank"+std::to_string(r)+"_bytes",bytes[r]);}
    j.set("frontier",std::uint64_t(tokens.size()-1));j.set("components",components);j.set("replay_sha256",runtime.replay_digest());return j;
}
// Qualification arms over one load. Incremental: generate steps+1 tokens,
// exporting at every tested frontier; fresh: prefill prompt+generated[:t],
// export, then consume the pending token. The harness compares raw digests.
Json qualify(p::Runtime& runtime,const std::vector<std::uint32_t>& prompt,unsigned steps,const std::string& export_dir){
    auto rec=Json::object();rec.set("prompt_tokens",std::uint64_t(prompt.size()));rec.set("steps",int(steps));
    strix::Sha256Fast legacy[2];runtime.prefill(prompt,"glm-pf-rtx-native-v1",[&](unsigned rank,unsigned,unsigned,std::span<const std::byte> b){legacy[rank].update(b);});
    auto parity=Json::object();for(unsigned r=0;r<2;++r)parity.set("rank"+std::to_string(r)+"_sha256",strix::model_content_identity_hex(legacy[r].finalize()));rec.set("legacy_prefill",parity);rec.set("legacy_prefill_row_tile",int(runtime.prefill_row_tile()));
    auto begin=std::chrono::steady_clock::now();runtime.begin_session(prompt);rec.set("session_row_tile",int(runtime.prefill_row_tile()));auto inc=Json::array();auto first=session_digest(runtime,prompt);first.set("t",0);inc.push(first);
    std::vector<std::uint32_t> generated;auto stepped=Json::array();
    for(unsigned t=1;t<=steps+1;++t){auto r=strix::serve::parse_json(runtime.decode_one(true));generated.push_back(std::uint32_t(r.find("token")->as_int()));stepped.push(r);
        if(t<=steps){auto seq=runtime.session_tokens();auto d=session_digest(runtime,seq,t==steps?export_dir:std::string{});d.set("t",int(t));inc.push(d);}}
    rec.set("incremental_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());rec.set("incremental_exports",inc);rec.set("incremental_steps",stepped);
    auto gen=Json::array();for(auto y:generated)gen.push(std::uint64_t(y));rec.set("generated",gen);
    auto fresh=Json::array();
    for(unsigned t=1;t<=steps;++t){auto seq=prompt;seq.insert(seq.end(),generated.begin(),generated.begin()+t);runtime.begin_session(seq);auto d=session_digest(runtime,seq);d.set("t",int(t));d.set("row_tile",int(runtime.prefill_row_tile()));
        auto r=strix::serve::parse_json(runtime.decode_one(true));d.set("next",r);fresh.push(d);}
    rec.set("fresh",fresh);runtime.reset_session();return rec;
}
// Backend comparison: the identical continuation decoded by the uncached
// reference tier and by the tierstream cache (cold) must agree on every token,
// every complete logits digest and the final state/replay digests.
Json compare(p::Runtime& runtime,const std::vector<std::uint32_t>& prompt,unsigned steps){
    auto rec=Json::object();rec.set("prompt_tokens",std::uint64_t(prompt.size()));auto arms=Json::array();
    for(const char* tier:{"reference","tierstream"}){runtime.select_expert_tier(tier);if(std::string(tier)=="tierstream")runtime.reset_expert_cache(true);
        runtime.begin_session(prompt);auto tokens=Json::array(),logits=Json::array(),steps_out=Json::array();
        for(unsigned i=0;i<steps;++i){auto r=strix::serve::parse_json(runtime.decode_one(true));tokens.push(r.find("token")->as_int());logits.push(r.find("logits_sha256")->as_string());steps_out.push(r);}
        auto a=Json::object();a.set("arm",tier);a.set("backend",runtime.expert_tier());a.set("tokens",tokens);a.set("logits_sha256",logits);a.set("steps",steps_out);a.set("final_state",session_digest(runtime,runtime.session_tokens()));arms.push(a);}
    runtime.select_expert_tier("tierstream");rec.set("arms",arms);runtime.reset_session();return rec;
}
// Time to first token on the cached tierstream path, cold cache: prefill of the
// whole prompt (session) + the first decoded token, host wall time.
// row_tile (optional, <= the allocated tile) runs this prefill at that tile and
// restores the per-request rule after (0 = auto: rows vs --prefill-tile-crossover). The prefill state digest (both ranks' exported
// state) is taken untimed between the prefill and the first token, so tiles can be
// compared bitwise; ttft_s = prefill_s + first_token_s.
Json ttft(p::Runtime& runtime,const std::vector<std::uint32_t>& prompt,unsigned row_tile=0){
    w::require(runtime.expert_tier()==std::string("tierstream"),"TTFT runs only on the cached tierstream backend");
    struct Restore {p::Runtime& r;unsigned t;~Restore(){try{r.use_prefill_row_tile(t);}catch(...){}}} restore{runtime,runtime.requested_prefill_row_tile()};
    if(row_tile)runtime.use_prefill_row_tile(row_tile);
    runtime.reset_expert_cache(true);const auto begin=std::chrono::steady_clock::now();
    auto prefill=strix::serve::parse_json(runtime.begin_session(prompt));const auto prefilled=std::chrono::steady_clock::now();
    auto state=session_digest(runtime,runtime.session_tokens());
    // The deferred decode residency restore runs here, untimed (a prefill-only request never pays it);
    // its own time, units/bytes copied and cache counters are reported separately.
    const auto decode_restore=strix::serve::parse_json(runtime.restore_decode_residency());
    const auto first_begin=std::chrono::steady_clock::now();auto first=strix::serve::parse_json(runtime.decode_one(false));const auto done=std::chrono::steady_clock::now();
    auto rec=Json::object();rec.set("prompt_tokens",std::uint64_t(prompt.size()));rec.set("backend",runtime.expert_tier());
    rec.set("row_tile",int(runtime.prefill_row_tile()));rec.set("allocated_row_tile",int(runtime.allocated_prefill_row_tile()));rec.set("row_tile_requested",int(row_tile));
    rec.set("prefill_tile_crossover",int(runtime.prefill_tile_crossover()));if(auto* c=prefill.find("row_tile_choice"))rec.set("row_tile_choice",*c);
    rec.set("decode_restore",decode_restore);
    if(auto* t=prefill.find("prefill_timings")){for(const char* k:{"prefill_cache","make_resident_ms_total","make_resident_ms_per_layer","prefill_overlap","prefill_overlap_gpu_idle_ms_total","prefill_overlap_gpu_idle_ms_per_layer","prefill_overlap_host_wait_ms_total","prefill_overlap_host_wait_ms_per_layer"})if(auto* v=t->find(k))rec.set(k,*v);}
    // the prefill's own expert cache traffic (file units read during this prefill; cheap: a counter delta)
    if(auto* c=prefill.find("prefill_expert_cache"))for(const char* k:{"file_reads","file_bytes","file_read_ns","misses","fetched","h2d_bytes"})if(auto* v=c->find(k))rec.set(std::string("prefill_")+k,*v);
    const double prefill_s=std::chrono::duration<double>(prefilled-begin).count(),first_s=std::chrono::duration<double>(done-first_begin).count();
    rec.set("prefill_s",prefill_s);rec.set("first_token_s",first_s);rec.set("ttft_s",prefill_s+first_s);rec.set("prefill_state",state);
    rec.set("token",first.find("token")->as_int());rec.set("prefill",prefill);runtime.reset_session();return rec;
}
// Timing arms from one restored initial session (the reference tier has no
// fetched-expert cache). Each arm also yields complete per-step logits digests
// and its final state/replay digests; digest work is excluded from the timed loop.
// One decode call of `left` wanted tokens: plain (one token) or a self-speculation cycle (1..2 tokens).
Json step(p::Runtime& runtime,bool spec,bool q,unsigned left){return strix::serve::parse_json(spec?runtime.decode_spec(q,left):runtime.decode_one(q));}
// Kernel trace (decode command "trace"; run under nsys --capture-range=cudaProfilerApi): from one restored
// session, a warm untimed pass, then profiler ranges around `steps` plain tokens and (when the service was opened
// with mtp_self_spec) `steps` tokens of self-speculation cycles. Outputs are not compared here (bench does that).
Json trace(p::Runtime& runtime,const std::vector<std::uint32_t>& prompt,unsigned steps,bool spec){
    auto rec=Json::object();rec.set("prompt_tokens",std::uint64_t(prompt.size()));runtime.begin_session(prompt);runtime.snapshot_session();
    for(unsigned i=0;i<steps;++i)(void)runtime.decode_one(false);runtime.restore_session();
    auto begin=std::chrono::steady_clock::now();runtime.profiler(true);for(unsigned i=0;i<steps;++i)(void)runtime.decode_one(false);runtime.profiler(false);
    rec.set("plain_s",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
    if(spec){runtime.restore_session();for(unsigned i=0;i<steps;){auto r=step(runtime,true,false,steps-i);i+=unsigned(r.find("tokens")->as_array().size());}runtime.restore_session();
        begin=std::chrono::steady_clock::now();runtime.profiler(true);unsigned calls=0;for(unsigned i=0;i<steps;++calls){auto r=step(runtime,true,false,steps-i);i+=unsigned(r.find("tokens")->as_array().size());}runtime.profiler(false);
        rec.set("spec_s",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());rec.set("spec_calls",int(calls));}
    runtime.reset_session();return rec;
}
Json bench(p::Runtime& runtime,const std::vector<std::uint32_t>& prompt,unsigned steps,const Json& order,bool spec=false){
    w::require(order.is_array()&&order.as_array().size()==2,"bench order: two arms");
    std::vector<std::string> names;for(auto& n:order.as_array()){w::require(n.is_string()&&(n.as_string()=="cold"||n.as_string()=="warm"),"bench arm label");names.push_back(n.as_string());}
    w::require(names[0]!=names[1],"bench arms must be cold and warm");
    // Timed arms run only on the cached tierstream product path.
    w::require(runtime.expert_tier()==std::string("tierstream"),"timed arms run only on the cached tierstream backend");
    auto rec=Json::object();rec.set("prompt_tokens",std::uint64_t(prompt.size()));rec.set("prefill",strix::serve::parse_json(runtime.begin_session(prompt)));runtime.snapshot_session();auto arms=Json::array();
    // cold: the fetched-expert cache reset to its open state; warm: an untimed
    // warm-up pass over the same continuation first (then the same restore).
    for(auto& name:names){runtime.restore_session();
        if(name=="cold")runtime.reset_expert_cache(true);
        else{for(unsigned i=0;i<steps;){auto r=step(runtime,spec,false,steps-i);i+=spec?unsigned(r.find("tokens")->as_array().size()):1;}runtime.restore_session();}
        auto tokens=Json::array(),logits=Json::array(),steps_out=Json::array();double digest_ms=0;std::uint64_t drafted=0,accepted=0;auto begin=std::chrono::steady_clock::now();
        for(unsigned i=0;i<steps;){auto r=step(runtime,spec,true,steps-i);digest_ms+=r.find("digest_ms")->as_double();
            if(spec){for(auto& t:r.find("tokens")->as_array())tokens.push(t.as_int());for(auto& g:r.find("logits")->as_array())logits.push(g.as_string());i+=unsigned(r.find("tokens")->as_array().size());drafted+=std::uint64_t(r.find("drafted")->as_int());accepted+=std::uint64_t(r.find("accepted")->as_int());}
            else{tokens.push(r.find("token")->as_int());logits.push(r.find("logits_sha256")->as_string());++i;}steps_out.push(r);}
        auto wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count(),loop=wall-digest_ms/1000;
        auto a=Json::object();a.set("arm",name);a.set("tokens",tokens);a.set("logits_sha256",logits);a.set("generation_loop_s",loop);a.set("digest_s",digest_ms/1000);a.set("tokens_per_s",loop>0?double(steps)/loop:0.0);a.set("steps",steps_out);if(spec){a.set("spec",true);a.set("drafted",drafted);a.set("accepted",accepted);a.set("calls",std::uint64_t(steps_out.as_array().size()));}
        // Every integer counter of every step (tierstream and reference alike).
        a.set("expert_totals",p::aggregate_expert_totals(steps_out));
        // --decode-profile: per phase device/host ms per token over the arm (timings include the events' cost).
        if(auto t=p::aggregate_decode_timelines(steps_out);t.find("tokens"))a.set("timeline_totals",t);
        a.set("final_state",session_digest(runtime,runtime.session_tokens()));arms.push(a);}
    rec.set("arms",arms);runtime.reset_session();return rec;
}
// Every generation receipt ends with the cumulative memory report (ledger peaks,
// allocation-time and request samples, process peak commit); a peak above a
// whole-device cap fails the receipt (CLI exit 5, HTTP 507) instead of a pass.
// The cache residency (most recent first) is written back after every request
// (HTTP prefill, generation, decode commands; DS writes it at shutdown; a killed service keeps the last request's). It seeds
// the next start's placement (--expert-residency). Best effort: placement only.
void persist_residency(p::Runtime& runtime,Json& receipt){
    try{const auto path=runtime.save_residency();if(!path.empty())receipt.set("residency_file",path);}catch(const std::exception& e){receipt.set("residency_error",std::string(e.what()));}
}
bool finalize_memory(p::Runtime& runtime,Json& receipt,const std::string& stage){
    persist_residency(runtime,receipt);
    auto m=strix::serve::parse_json(runtime.memory_report(stage));const bool within=m.find("within_caps")->as_bool();
    receipt.set("memory",m);if(!within){receipt.set("status","FAILED");receipt.set("failure","a measured memory peak exceeds the whole-device cap");}
    return within;
}
// A generation that threw still yields a FAILED receipt: the original error and
// the measured peaks (best effort: the report itself may fail after a device error).
Json failed_receipt(p::Runtime& runtime,const std::string& error,const std::string& stage){
    auto j=Json::object();j.set("status","FAILED");j.set("error",error);
    try{j.set("memory",strix::serve::parse_json(runtime.memory_report(stage)));}catch(const std::exception& e){j.set("memory_report_error",std::string(e.what()));}
    // Which layer/flag/token failed and the cache state (device_timeouts, stalls, free slots, flush_ready).
    try{j.set("expert_cache",strix::serve::parse_json(runtime.expert_diagnostics()));}catch(const std::exception& e){j.set("expert_cache_error",std::string(e.what()));}
    return j;
}
void response(w::Socket fd,int code,const std::string& body){std::string text="HTTP/1.1 "+std::to_string(code)+" Result\r\nContent-Type: application/json\r\nContent-Length: "+std::to_string(body.size())+"\r\nConnection: close\r\n\r\n"+body;w::io(fd,std::as_writable_bytes(std::span(text)),true);}

int main(int argc,char** argv){p::install_crash_report();try{
    // Lazy module loading + a kernel spinning on a device flag (tierstream's wait kernel) can
    // stall the first launch of another kernel until the wait deadline (as the DS replica and
    // tierstream's own preload guard against): CUDA modules load eagerly, set here before any
    // CUDA call, unless CUDA_MODULE_LOADING is set explicitly (the tierstream decode service
    // then still refuses a lazy driver below).
    std::string module_loading;{const char* v=std::getenv("CUDA_MODULE_LOADING");
        if(v&&*v)module_loading=std::string(v)+" (environment)";else{_putenv_s("CUDA_MODULE_LOADING","EAGER");module_loading="EAGER (set by the service)";}}
    std::string command_file,command_out; // --decode-commands from a file (e.g. under a profiler that keeps stdin), results also to a file
    std::string cache,model,manifest_hash,key_file,export_dir,prompt_file,identity="glm-pf-rtx-native-v1",bind="127.0.0.1";unsigned port=8097,capacity=8192,tile=0,host_floor_gib=8,max_device_layers=43,max_pinned_layers=43;bool require_file_tier=false,allow=false,graphs=true,self_check=false,legacy_index=false,serial_stage=false,enable_generate=false,decode_commands=false,decode_profile=false;std::vector<std::string> check_prompts;unsigned decode_tokens=0;p::DecodeOptions dopts;bool plan_budget=false,print_effective=false,check_commands=false;double device_total_gib=-1,device_free_gib=-1;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--require-file-tier")require_file_tier=true;else if(a=="--legacy-fp32-index")legacy_index=true;else if(a=="--serial-stage")serial_stage=true;else if(a=="--self-check-schedule")self_check=true;else if(a=="--allow-gpu")allow=true;else if(a=="--no-graphs")graphs=false;else if(a=="--enable-generate")enable_generate=true;else if(a=="--decode-commands")decode_commands=true;else if(a=="--reference-compare")dopts.reference_compare=true;else if(a=="--plan-budget")plan_budget=true;else if(a=="--allow-pending"){}else if(a=="--print-effective-config")print_effective=true;else if(a=="--check-decode-commands")check_commands=true;else if(a=="--decode-profile")decode_profile=true;else {w::require(i+1<argc,"missing option argument");std::string value=argv[++i];if(a=="--profile"){}else if(a=="--cache")cache=value;else if(a=="--model")model=value;else if(a=="--cache-manifest-sha256")manifest_hash=value;else if(a=="--key-file")key_file=value;else if(a=="--export-state")export_dir=value;else if(a=="--prompt"){prompt_file=value;check_prompts.push_back(value);}else if(a=="--identity")identity=value;else if(a=="--decode"){auto n=std::stoul(value);w::require(n>=1&&n<=131072,"decode token count");decode_tokens=unsigned(n);}else if(a=="--expert-tier")dopts.expert_tier=value;else if(a=="--unit-file")dopts.unit_file=value;else if(a=="--vram-total-gib")dopts.vram_total_gib=std::stod(value);else if(a=="--ram-total-gib")dopts.ram_total_gib=std::stod(value);else if(a=="--device-total-gib")device_total_gib=std::stod(value);else if(a=="--device-free-gib")device_free_gib=std::stod(value);else if(a=="--tier-vram-gib")dopts.tier_vram_gib=std::stod(value);else if(a=="--tier-ram-gib")dopts.tier_ram_gib=std::stod(value);
        else if(a=="--tier-spare-slots"){dopts.tier_spare_slots=unsigned(std::stoul(value));dopts.tier_spare_permille=0;}else if(a=="--tier-spare-percent"){const double v=std::stod(value);if(!(v>=0&&v<=50))throw std::runtime_error("--tier-spare-percent must be in [0, 50]");dopts.tier_spare_permille=unsigned(v*10+0.5);}else if(a=="--expert-residency")dopts.expert_residency=value;else if(a=="--expert-routes")dopts.expert_routes=value;else if(a=="--routes-out")dopts.routes_out=value;else if(a=="--routes-prefill"){w::require(value=="0"||value=="1","--routes-prefill 0|1");dopts.routes_prefill=value=="1";}
        else if(a=="--tier-policy"){w::require(value=="lru"||value=="scored"||value=="auto","--tier-policy must be lru, scored or auto");dopts.tier_policy=value;}else if(a=="--tier-predict"){w::require(value=="0"||value=="1","--tier-predict 0|1");dopts.tier_predict=value=="1";}
        else if(a=="--skip-empty-expert-groups"){w::require(value=="0"||value=="1","--skip-empty-expert-groups 0|1");dopts.skip_empty_expert_groups=value=="1";}
        else if(a=="--persistent-dense-fp16"){w::require(value=="0"||value=="1","--persistent-dense-fp16 0|1");dopts.persistent_dense_fp16=value=="1";}
        else if(a=="--decode-row-kernels")p::set_row_kernels_value(dopts,value);
        else if(a=="--command-file")command_file=value;else if(a=="--command-out")command_out=value;
        else if(a=="--mtp-self-spec"){w::require(value=="0"||value=="1"||value=="always","--mtp-self-spec 0|1|always");dopts.mtp_spec=value!="0";dopts.spec_always=value=="always";}
        else if(a=="--spec-lookup"){w::require(value=="0"||value=="1","--spec-lookup 0|1");dopts.spec_lookup=value=="1";}
        else if(a=="--restore-decode-residency"){w::require(value=="0"||value=="1","--restore-decode-residency 0|1");dopts.restore_decode_residency=value=="1";}
        else if(a=="--prefill-overlap"){w::require(value=="0"||value=="1","--prefill-overlap 0|1");dopts.prefill_overlap=value=="1";}
        else if(a=="--tier-flush-ready"){w::require(value=="0"||value=="1","--tier-flush-ready 0|1");dopts.tier_flush_ready=value=="1";}
        else if(a=="--tier-host-spares")dopts.tier_host_spares=unsigned(std::stoul(value));else if(a=="--tier-staging"){w::require(!value.empty()&&value.size()<=2&&value.find_first_not_of("0123456789")==std::string::npos&&std::stoul(value)>=1&&std::stoul(value)<=64,"--tier-staging 1..64");dopts.tier_staging=unsigned(std::stoul(value));}
        else if(a=="--prefill-tile-crossover"){w::require(value=="off"||(!value.empty()&&value.find_first_not_of("0123456789")==std::string::npos&&value.size()<=6&&std::stoul(value)<=131072),"--prefill-tile-crossover: rows 0..131072 or off");}
        else if(a=="--weight-dirs"){w::require(value.find('\0')==std::string::npos,"--weight-dirs: embedded NUL");p::check_weight_dirs(value);dopts.weight_dirs=value;}
        else if(a=="--drive-readers"){w::require(!value.empty()&&value.size()<=2&&value.find_first_not_of("0123456789")==std::string::npos&&std::stoul(value)>=1&&std::stoul(value)<=32,"--drive-readers 1..32");dopts.drive_readers=unsigned(std::stoul(value));}
        else if(a=="--tier-disk-queue"){w::require(!value.empty()&&value.size()<=2&&value.find_first_not_of("0123456789")==std::string::npos&&std::stoul(value)>=1&&std::stoul(value)<=64,"--tier-disk-queue 1..64");dopts.tier_disk_queue=unsigned(std::stoul(value));}
        else if(a=="--tier-clean-evict"){w::require(value=="0"||value=="1","--tier-clean-evict 0|1");dopts.tier_clean_evict=value=="1";}
        else if(a=="--tier-landing-blocks"){w::require(!value.empty()&&value.size()<=4&&value.find_first_not_of("0123456789")==std::string::npos&&std::stoul(value)<=4096,"--tier-landing-blocks 0..4096");dopts.tier_landing_blocks=unsigned(std::stoul(value));}
        else if(a=="--tier-landing-max-queued"){w::require(!value.empty()&&value.size()<=2&&value.find_first_not_of("0123456789")==std::string::npos&&std::stoul(value)<=64,"--tier-landing-max-queued 0..64");dopts.tier_landing_max_queued=unsigned(std::stoul(value));}
        else if(a=="--tier-landing-ttl"){w::require(!value.empty()&&value.size()<=4&&value.find_first_not_of("0123456789")==std::string::npos&&std::stoul(value)<=1024,"--tier-landing-ttl 0..1024");dopts.tier_landing_ttl=unsigned(std::stoul(value));}
        else if(a=="--tier-landing-chunk"){w::require(value=="whole"||(value.find_first_not_of("0123456789")==std::string::npos&&!value.empty()&&value.size()<=10),"--tier-landing-chunk BYTES|whole");
            const auto n=value=="whole"?0xFFFFFFFFull:std::stoull(value);w::require(n<=0xFFFFFFFFull&&(n==0xFFFFFFFFull||n%4096==0),"--tier-landing-chunk BYTES (0 or a multiple of 4096)|whole");dopts.tier_landing_chunk=std::uint32_t(n);}
        else if(a=="--disk-prefetch")dopts.disk_prefetch=value=="1";else if(a=="--unit-manifest-sha256")dopts.unit_manifest_sha256=value;else if(a=="--bind")bind=value;else if(a=="--port")port=unsigned(std::stoul(value));else if(a=="--capacity")capacity=unsigned(std::stoul(value));else if(a=="--row-tile")tile=value=="auto"?0:unsigned(std::stoul(value));else if(a=="--max-device-layers")max_device_layers=p::layer_limit(value);else if(a=="--max-pinned-layers")max_pinned_layers=p::layer_limit(value);else if(a=="--host-floor-gib"){auto n=std::stoul(value);w::require(n<=1024,"host floor range");host_floor_gib=unsigned(n);}else throw std::runtime_error("unknown service option");}}
    // --profile default|fast|quality (profile.hpp, the DS RTX resolver's structure): one switch selects every lever;
    // per-lever options stay expert overrides (CLI > environment > profile > builtin). A pending row never refuses the
    // run: its builtin form runs and it is listed (receipt + one stderr line). --print-effective-config prints the
    // resolution (the first receipt event's "profile" object) and exits here, before any model read or CUDA call.
    namespace rp=strix::glm::rtxpf::profile;
    rp::Inputs profile_inputs;for(int i=1;i<argc;++i)profile_inputs.args.emplace_back(argv[i]);
    const auto prof=rp::resolve(rp::App::decode_service,profile_inputs);
    auto use=[&](const char* lever)->const rp::Entry*{auto* e=prof.find(lever);return e&&(e->status==rp::Status::profile||e->status==rp::Status::unqualified)?e:nullptr;};
    if(auto* e=use("expert_tier"))dopts.expert_tier=e->value;
    if(auto* e=use("eviction_policy"))dopts.tier_policy=e->value;
    if(auto* e=use("spare_share"))dopts.tier_spare_permille=unsigned(std::stoul(e->value)*10);
    if(auto* e=use("row_tile"))tile=e->value=="auto"?0u:unsigned(std::stoul(e->value));
    if(auto* e=use("cuda_graphs"))graphs=e->value!="off";
    // Per-request prefill tile crossover (rows; exact): the profile value or --prefill-tile-crossover (0/off = always the allocated tile).
    unsigned tile_crossover=p::default_prefill_tile_crossover;
    if(auto* e=prof.find("prefill_tile_crossover"))tile_crossover=e->value=="off"||e->value.empty()?0u:unsigned(std::stoul(e->value));
    // flush_ready: the profile value, or STRIX_GLM_RTX_FLUSH_READY=0|1 (explicit env); --tier-flush-ready was applied above.
    // The exact decode levers: pending (their builtin, off) until qualified; --allow-pending runs them; the environment
    // (STRIX_GLM_RTX_SKIP_EMPTY_GROUPS / _PERSISTENT_DENSE_FP16) sets them explicitly. prefill_overlap: on in every profile.
    for(auto [lever,flag]:{std::pair<const char*,bool*>{"skip_empty_expert_groups",&dopts.skip_empty_expert_groups},{"persistent_dense_fp16",&dopts.persistent_dense_fp16},{"prefill_overlap",&dopts.prefill_overlap}})
        if(auto* e=prof.find(lever);e&&e->status!=rp::Status::explicit_cli&&e->status!=rp::Status::pending)*flag=e->value!="off"&&e->value!="0";
    if(auto* e=prof.find("decode_row_kernels");e&&e->status!=rp::Status::explicit_cli&&e->status!=rp::Status::pending)p::set_row_kernels_value(dopts,e->value);
    // mtp_self_spec from the profile applies where it can run (the tierstream cache, no --routes-out); --mtp-self-spec 1 elsewhere is an error.
    if(auto* e=prof.find("mtp_self_spec");e&&e->status!=rp::Status::explicit_cli&&e->status!=rp::Status::pending)
        {dopts.mtp_spec=e->value!="off"&&e->value!="0"&&dopts.expert_tier=="tierstream"&&dopts.routes_out.empty();dopts.spec_always=e->value=="always";}
    if(auto* e=prof.find("lookup_drafts");e&&e->status!=rp::Status::explicit_cli&&e->status!=rp::Status::pending)dopts.spec_lookup=e->value!="off"&&e->value!="0";
    if(auto* e=prof.find("restore_decode_residency");e&&e->status!=rp::Status::explicit_cli)dopts.restore_decode_residency=e->value!="off"&&e->value!="0";
    if(auto* e=prof.find("flush_ready");e&&e->status!=rp::Status::explicit_cli)dopts.tier_flush_ready=e->value!="off"&&e->value!="0";
    if(auto* e=prof.find("predicted_prefetch");e&&(e->status==rp::Status::pending||e->status==rp::Status::profile||e->status==rp::Status::unqualified))dopts.tier_predict=e->value=="1";
    // After every profile-derived value: self-speculation exports two rows per layer through the tierstream cache.
    w::require(!dopts.spec_lookup||dopts.mtp_spec,"--spec-lookup drafts for the self-speculation verify (--mtp-self-spec 1)");
    // Landing (tierstream dg_landing_*): the sub-options need the blocks, and the landing reads run on the read pool.
    w::require(dopts.tier_landing_blocks||(!dopts.tier_landing_max_queued&&!dopts.tier_landing_ttl&&!dopts.tier_landing_chunk),"--tier-landing-max-queued / -ttl / -chunk need --tier-landing-blocks N > 0");
    w::require(!dopts.tier_landing_blocks||(dopts.expert_tier=="tierstream"&&dopts.tier_disk_queue>1),"--tier-landing-blocks needs the tierstream cache and --tier-disk-queue > 1");
    w::require(!dopts.mtp_spec||(dopts.routes_out.empty()&&dopts.expert_tier=="tierstream"),
               "--mtp-self-spec: tierstream expert tier only, not with --routes-out (it records one row per layer)");
    if(auto* e=use("prefill_tile_large");e&&e->status==rp::Status::unqualified){auto* rt=prof.find("row_tile");if(!rt||rt->status!=rp::Status::explicit_cli)tile=std::min(8192u,capacity/32*32);}
    auto effective=[&](std::string_view lever)->std::string{
        if(lever=="expert_tier")return dopts.expert_tier;
        if(lever=="placement")return "residency "+(dopts.expert_residency.empty()?std::string("<unit-file>.residency"):dopts.expert_residency)+(dopts.expert_routes.empty()?std::string():", routes "+dopts.expert_routes)+", round-robin fallback";
        if(lever=="eviction_policy")return dopts.tier_policy;
        if(lever=="spare_share")return dopts.tier_spare_permille?std::to_string(dopts.tier_spare_permille)+" permille (min "+std::to_string(dopts.tier_spare_slots)+" slots)":std::to_string(dopts.tier_spare_slots)+" slots (exact)";
        if(lever=="row_tile")return tile?std::to_string(tile):std::string("auto (planner rule)");
        if(lever=="cuda_graphs")return graphs?"on":"off";
        if(lever=="flush_ready")return dopts.tier_flush_ready?"on":"off";
        if(lever=="restore_decode_residency")return dopts.restore_decode_residency?"on":"off";
        if(lever=="skip_empty_expert_groups")return dopts.skip_empty_expert_groups?"1":"0";
        if(lever=="persistent_dense_fp16")return dopts.persistent_dense_fp16?"1 (dropped at startup if the plan cannot afford it)":"0";
        if(lever=="prefill_overlap")return dopts.prefill_overlap?"1 (off when the arena cannot keep two layers)":"0";
        if(lever=="decode_row_kernels")return p::row_kernels_name(dopts);
        if(lever=="mtp_self_spec")return dopts.mtp_spec?"1":"0";
        if(lever=="lookup_drafts")return dopts.spec_lookup?"1":"0";
        if(lever=="predicted_prefetch")return dopts.tier_predict?"1":"0";
        if(lever=="prefill_tile_large")return tile>=8192?"8192":"off";
        if(lever=="prefill_tile_crossover")return tile_crossover?std::to_string(tile_crossover)+" rows (below: min(4096, allocated tile); at/above: the allocated tile)":std::string("off (always the allocated tile)");
        if(lever=="dsa_index_mma")return "fp32";
        return {};};
    const std::string profile_json=prof.json(effective);
    // --check-decode-commands (CPU only, before any model read or CUDA call): every stdin line through the
    // service's own decode-command parser; one {"ok":true,...} or {"ok":false,"error":...} per line (the suite
    // feeds it every request shape the harness plans).
    if(check_commands){for(std::string line;std::getline(std::cin,line);){auto r=Json::object();
            try{auto c=p::parse_decode_command(strix::serve::parse_json(line));if(!c.row_kernels.empty()){p::DecodeOptions d;p::set_row_kernels_value(d,c.row_kernels);r.set("row_kernels",p::row_kernels_name(d));}
                if(c.decode_profile>=0)r.set("decode_profile",c.decode_profile==1);r.set("ok",true);r.set("op",c.op);r.set("steps",int(c.steps));r.set("row_tile",int(c.row_tile));r.set("export_dir",!c.export_dir.empty());r.set("order",c.order);}
            catch(const std::exception& e){r.set("ok",false);r.set("error",std::string(e.what()));}
            std::cout<<r.dump()<<"\n"<<std::flush;}
        return 0;}
    if(print_effective){auto pj=strix::serve::parse_json(profile_json);pj.set("cuda_module_loading",module_loading);std::cout<<pj.dump()<<"\n"<<std::flush;return 0;}
    for(const auto& line:prof.notices())std::cerr<<line<<"\n";
    if(plan_budget){
        // Dry plan (no --allow-gpu, no CUDA calls): sizes from shard headers, host
        // figures measured, device figures given or an idle PRO at the 99.86 GB cap.
        w::require(!model.empty()&&cache.empty(),"--plan-budget needs --model (shard headers)");
        const double gib=double(1ULL<<30);const std::uint64_t cap_bytes=99860000000ULL;
        // Default: an idle PRO at the 99.86 GB cap with an assumed 640 MiB CUDA context
        // (the service measures it at startup); measured figures can be passed.
        const std::uint64_t total=device_total_gib<0?cap_bytes:std::uint64_t(device_total_gib*gib),free=device_free_gib<0?total-(640ULL<<20):std::uint64_t(device_free_gib*gib);
        w::require(free<=total,"--device-free-gib must not exceed --device-total-gib");
        auto j=strix::serve::parse_json(p::Runtime::plan_budget(model,capacity,tile,host_floor_gib,dopts,total,free));
        j.set("device_figures",device_free_gib<0?"assumed idle PRO (99.86 GB cap, 640 MiB CUDA context assumed); pass --device-total-gib/--device-free-gib for measured figures":"given on the command line");
        std::cout<<j.dump()<<"\n";return j.find("verdict")->as_string()=="ADMITTED"?0:3;
    }
    w::require(allow,"requires explicit --allow-gpu during an authorized PRO window");
    const char* visible=std::getenv("CUDA_VISIBLE_DEVICES");p::require_pro_uuid(visible?visible:"");
    w::require((cache.empty()!=model.empty())&&(!cache.empty()? !manifest_hash.empty():manifest_hash.empty())&&(!key_file.empty()||!export_dir.empty()||self_check||decode_tokens||decode_commands)&&port>0&&port<=65535,"exactly one of --model or --cache (with pinned manifest SHA), key-file (HTTP) and valid port required");
    if(const char* off=std::getenv("STRIX_GLM_RTX_PREFILL"))w::require(std::string(off)!="0","native prefill disabled");
    const bool decode=decode_tokens||decode_commands||enable_generate;
    // Every generation and timing entry point runs on the cached tierstream
    // backend; the reference backend only inside the untimed compare (--reference-compare).
    w::require(!decode||dopts.expert_tier=="tierstream","decode, generation and timing run only on the cached tierstream backend (the reference backend only inside the untimed compare)");
    w::require(!decode||(!self_check&&export_dir.empty()&&identity=="glm-pf-rtx-native-v1"),"decode modes exclude self-check, export and alternate index identities");
    w::require(!(decode_tokens&&decode_commands),"choose one of --decode or --decode-commands");
    std::unique_ptr<p::VerifiedManifest> manifest_file;
    if(!cache.empty())manifest_file=std::make_unique<p::VerifiedManifest>(std::filesystem::path(cache)/"manifest.json",manifest_hash);
    auto startup_begin=std::chrono::steady_clock::now();
    p::Runtime runtime(model.empty()?cache:model,capacity,tile,!model.empty(),graphs,host_floor_gib,legacy_index,serial_stage,manifest_file.get(),max_device_layers,max_pinned_layers,require_file_tier,decode,dopts);if(decode_profile)runtime.set_decode_profile(true);runtime.set_prefill_tile_crossover(tile_crossover);auto health_value=strix::serve::parse_json(runtime.health());
    {CUmoduleLoadingMode mode{};const bool eager=cuModuleGetLoadingMode(&mode)==CUDA_SUCCESS&&mode==CU_MODULE_EAGER_LOADING;
     health_value.set("cuda_module_loading",module_loading+(eager?"; driver: eager":"; driver: lazy"));
     w::require(eager||!decode||dopts.expert_tier!="tierstream","the tierstream decode service needs eager CUDA module loading (unset CUDA_MODULE_LOADING or set it to EAGER): a lazily loaded kernel launched while the cache's wait kernel spins can stall until the wait deadline");}health_value.set("decode_profile",decode_profile);
    health_value.set("startup_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-startup_begin).count());const auto health=health_value.dump();
    std::wstring module(32768,L'\0');auto module_length=GetModuleFileNameW(nullptr,module.data(),DWORD(module.size()));w::require(module_length>0&&module_length<module.size(),"executable path");module.resize(module_length);
    const auto executable_hash=strix::model_content_identity_hex(strix::sha256_file_content(std::filesystem::path(module)));
    auto identity_fields=[&](Json& j){j.set("profile",strix::serve::parse_json(profile_json));j.set("producer_executable_sha256",executable_hash);j.set("model_sha256",runtime.model_sha256());j.set("cache_manifest_sha256",model.empty()?Json(manifest_hash):Json());j.set("weight_source",model.empty()?"cache":"shards");j.set("capacity",int(capacity));j.set("health",health_value);};
    if(decode_tokens){
        w::require(!prompt_file.empty(),"--decode needs --prompt little-endian u32 file");auto tokens=read_prompt(prompt_file,capacity);
        // Generation AND its finalization (identity, memory report) inside one receipt
        // boundary: any failure still prints a FAILED receipt with the error and the peaks.
        Json receipt;bool within=false;
        try{receipt=strix::serve::parse_json(runtime.generate(tokens,decode_tokens,{},false));identity_fields(receipt);within=finalize_memory(runtime,receipt,"generate");}
        catch(const std::exception& e){receipt=failed_receipt(runtime,e.what(),"generate (failed)");try{identity_fields(receipt);}catch(...){}std::cout<<receipt.dump()<<"\n";return 5;}
        catch(...){receipt=failed_receipt(runtime,"generate: non-standard exception","generate (failed)");std::cout<<receipt.dump()<<"\n";return 5;}
        std::cout<<receipt.dump()<<"\n";return within?0:5;
    }
    if(decode_commands){
        // One JSON command per stdin line, one JSON result per stdout line; the
        // harness decides (deadline) whether to send another arm.
        std::ifstream cmd_in;std::ofstream cmd_out;if(!command_file.empty()){cmd_in.open(command_file);w::require(bool(cmd_in),"--command-file unreadable");}
        if(!command_out.empty()){cmd_out.open(command_out,std::ios::app);w::require(bool(cmd_out),"--command-out unwritable");}
        std::istream& commands=command_file.empty()?static_cast<std::istream&>(std::cin):cmd_in;
        auto emit=[&](const Json& j){std::cout<<j.dump()<<"\n"<<std::flush;if(cmd_out.is_open())cmd_out<<j.dump()<<"\n"<<std::flush;};
        auto ready=Json::object();ready.set("ready",true);identity_fields(ready);emit(ready);
        for(std::string line;std::getline(commands,line);){if(line.empty())continue;
            const auto cmd=p::parse_decode_command(strix::serve::parse_json(line));const auto& op=cmd.op;if(op=="quit")break;
            // Optional per-command decode_row_kernels / decode timeline (in-process A/B over one load); they stay set.
            if(!cmd.row_kernels.empty())runtime.set_row_kernels(cmd.row_kernels);
            if(cmd.decode_profile>=0)runtime.set_decode_profile(cmd.decode_profile==1);
            if(cmd.lookup>=0)runtime.set_spec_lookup(cmd.lookup==1);
            auto tokens=read_prompt(cmd.prompt,capacity);const auto steps=cmd.steps;w::require(steps>=1&&steps<=64&&tokens.size()+steps+1<=capacity,"command steps/capacity");
            auto begin=std::chrono::steady_clock::now();Json out;
            try{
            if(op=="qualify")out=qualify(runtime,tokens,steps,cmd.export_dir);
            else if(op=="bench")out=bench(runtime,tokens,steps,cmd.order,cmd.spec);
            else if(op=="compare")out=compare(runtime,tokens,steps);else if(op=="ttft")out=ttft(runtime,tokens,cmd.row_tile);
            else if(op=="trace")out=trace(runtime,tokens,steps,cmd.spec);else throw std::runtime_error("unknown decode command");
            }catch(const std::exception& e){
                // The worker exits (its cache may be retired); the failure receipt (error, peaks,
                // expert cache diagnostics) goes to stderr, which the harness keeps as the service log.
                auto f=failed_receipt(runtime,e.what(),"request:"+op+" (failed)");f.set("op",op);f.set("prompt",cmd.prompt);
                std::cerr<<"FAILED-RECEIPT "<<f.dump()<<"\n"<<std::flush;throw;}
            persist_residency(runtime,out);
            out.set("memory",strix::serve::parse_json(runtime.memory_report("request:"+op))); // measured peaks vs the whole-device caps
            out.set("op",op);out.set("row_kernels",runtime.row_kernels());out.set("prompt",cmd.prompt);out.set("wall_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());emit(out);
        }
        return 0;
    }
    if(self_check){
        w::require(!check_prompts.empty(),"self-check requires --prompt");
        auto record=Json::object();record.set("health",health_value);std::cout<<record.dump()<<"\n"<<std::flush;
        for(const auto& file:check_prompts){
            std::ifstream in(file,std::ios::binary);w::require(bool(in),"open self-check prompt");
            std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});w::require(!in.bad()&&bytes.size()%4==0&&bytes.size()<=std::size_t(capacity)*4,"self-check prompt extent");
            std::vector<std::uint32_t> tokens(bytes.size()/4);std::memcpy(tokens.data(),bytes.data(),bytes.size());
            auto result=runtime.self_check(tokens,identity);std::cout<<result<<"\n"<<std::flush;
            if(strix::serve::parse_json(result).find("verdict")->as_string()!="EXACT")return 3;
        }
        return 0;
    }
    if(!export_dir.empty()){
        w::require(!prompt_file.empty(),"export needs --prompt little-endian u32 file");
        std::ifstream input(prompt_file,std::ios::binary);w::require(bool(input),"open prompt file");std::vector<char> bytes((std::istreambuf_iterator<char>(input)),{});w::require(!input.bad()&&bytes.size()%4==0,"prompt file");std::vector<std::uint32_t> tokens(bytes.size()/4);std::memcpy(tokens.data(),bytes.data(),bytes.size());w::require(tokens.size()>=2&&tokens.size()<=capacity,"export prompt capacity");
        std::filesystem::create_directories(export_dir);std::ofstream out[2];for(unsigned rank=0;rank<2;++rank){out[rank].open(std::filesystem::path(export_dir)/("state-r"+std::to_string(rank)+".bin.partial"),std::ios::binary);w::Bytes header;w::number(header,tokens.size()-1);w::number(header,tokens.size());out[rank].write(reinterpret_cast<const char*>(header.data()),header.size());}
        runtime.prefill(tokens,identity,[&](unsigned rank,unsigned,unsigned,std::span<const std::byte> data){out[rank].write(reinterpret_cast<const char*>(data.data()),data.size());w::require(bool(out[rank]),"export state write failed");});
        auto receipt=Json::object();receipt.set("producer_identity",identity);receipt.set("qualification","DRAFT");receipt.set("frontier",std::uint64_t(tokens.size()-1));receipt.set("producer_executable_sha256",executable_hash);receipt.set("model_sha256",runtime.model_sha256());
        for(unsigned rank=0;rank<2;++rank){out[rank].close();w::require(bool(out[rank]),"export close failed");auto path=std::filesystem::path(export_dir)/("state-r"+std::to_string(rank)+".bin");std::filesystem::rename(path.string()+".partial",path);receipt.set("rank"+std::to_string(rank)+"_sha256",strix::model_content_identity_hex(strix::sha256_file_content(path)));}
        std::cout<<receipt.dump()<<"\n";return 0;
    }
    w::require(identity=="glm-pf-rtx-native-v1","alternate identities require --export-state; wire v1 cannot accept them");
    auto expected_key=w::read_key(key_file);p::SocketRuntime sockets;
    w::SocketOwner listener;listener.fd=socket(AF_INET,SOCK_STREAM,0);w::require(listener.fd!=w::bad_socket,"HTTP socket");sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(static_cast<u_short>(port));w::require(inet_pton(AF_INET,bind.c_str(),&address.sin_addr)==1,"numeric HTTP bind");
    w::require(::bind(listener.fd,reinterpret_cast<sockaddr*>(&address),sizeof address)==0&&listen(listener.fd,8)==0,"HTTP bind/listen");
    std::mutex busy;std::atomic<unsigned> connections=0;
    for(;;){auto fd=accept(listener.fd,nullptr,nullptr);if(fd==w::bad_socket)continue;if(connections.fetch_add(1)>=8){--connections;w::close_socket(fd);continue;}
        std::thread([&,fd]{w::SocketOwner client;client.fd=fd;DWORD timeout=120000;setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,reinterpret_cast<char*>(&timeout),sizeof timeout);setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,reinterpret_cast<char*>(&timeout),sizeof timeout);
            try{auto req=request(fd);if(req.method=="GET"&&req.path=="/health"){response(fd,200,health);}
                else if(req.method=="POST"&&req.path=="/v1/prefill"){
                    auto body=strix::serve::parse_json(req.body);fields(body,{"tokens","handoff","identity"});auto& handoff=field(body,"handoff");fields(handoff,{"rank0","rank1","key","node_identity"});
                    // The request supplies the raw shared key as 64 hex digits;
                    // compare to the startup key before connecting to either rank.
                    auto supplied=w::unhex(field(handoff,"key").as_string());auto expected=w::hash(expected_key);
                    auto key_bytes=std::as_bytes(std::span(supplied.bytes));w::require(w::secure_equal(w::hash(key_bytes),expected),"handoff key authentication failed");
                    auto id=body.find("identity")?body.find("identity")->as_string():"glm-pf-rtx-native-v1";
                    w::require(id=="glm-pf-rtx-native-v1","receiver wire v1 accepts only glm-pf-rtx-native-v1; alternate index identities cannot hand off");
                    auto node=handoff.find("node_identity")?handoff.find("node_identity")->as_string():"r1b";w::require(node=="r1"||node=="r1b"||node=="r1b-all","node identity");
                    std::vector<std::uint32_t> tokens;for(auto& t:field(body,"tokens").as_array()){auto n=t.as_int();w::require(n>=0&&n<154880,"token out of vocabulary");tokens.push_back(std::uint32_t(n));}w::require(tokens.size()>=2&&tokens.size()<=capacity,"prompt capacity");
                    std::unique_lock lock(busy,std::try_to_lock);if(!lock.owns_lock()){response(fd,503,"{\"error\":\"producer busy\"}");--connections;return;}
                    auto started=std::chrono::steady_clock::now();w::Begin b;b.layers=46;b.frontier=tokens.size()-1;b.total=tokens.size();b.node=node;b.producer=id;b.model=w::unhex(runtime.model_sha256());b.prompt=w::hash(std::as_bytes(std::span(tokens)));std::random_device random;for(auto& x:b.transaction.bytes)x=std::uint8_t(random());
                    auto connect_rank=[&](unsigned rank){auto begin=b;begin.rank=rank;auto endpoint=field(handoff,rank?"rank1":"rank0").as_string();return std::make_unique<p::Sender>(endpoint,expected_key,begin);};
                    auto c0=std::async(std::launch::async,connect_rank,0),c1=std::async(std::launch::async,connect_rank,1);
                    AsyncSender rank0(c0.get()),rank1(c1.get());
                    runtime.prefill(tokens,id,[&](unsigned rank,unsigned layer,unsigned component,std::span<const std::byte> bytes){(rank?rank1:rank0).submit(layer,component,bytes);});
                    const auto produced=std::chrono::steady_clock::now();rank0.drain();rank1.drain();auto s0=std::async(std::launch::async,[&]{rank0.sender->commit();}),s1=std::async(std::launch::async,[&]{rank1.sender->commit();});s0.get();s1.get();
                    auto receipt=Json::object();receipt.set("status","LOCALLY_SEALED_BOTH");receipt.set("global_commit","requires both node receipts");receipt.set("qualification","DRAFT");receipt.set("requested_route","rtx-native");receipt.set("effective_route","rtx-native");receipt.set("producer_identity",id);receipt.set("node_prefill_identity",node=="r1b"?"r1":node);receipt.set("node_decode_identity",node);receipt.set("composed_identity",id+"+"+(node=="r1b"?"r1":node)+"+"+node);receipt.set("model_sha256",runtime.model_sha256());receipt.set("producer_executable_sha256",executable_hash);receipt.set("cache_manifest_sha256",model.empty()?Json(manifest_hash):Json());receipt.set("weight_source",model.empty()?"cache":"shards");receipt.set("qualification_receipt",Json());receipt.set("node_executable_hashes",Json());receipt.set("phase_timings",strix::serve::parse_json(runtime.timings()));receipt.set("fallback_reason",Json());receipt.set("frontier",std::uint64_t(b.frontier));receipt.set("mtp","prefix pairs complete; node seed pending");receipt.set("raw_bytes",rank0.sender->raw_bytes()+rank1.sender->raw_bytes());receipt.set("wire_bytes",rank0.sender->wire_bytes()+rank1.sender->wire_bytes());receipt.set("producer_ms",std::chrono::duration<double,std::milli>(produced-started).count());receipt.set("total_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());receipt.set("capacity_cache_profile",strix::serve::parse_json(health));persist_residency(runtime,receipt);response(fd,200,receipt.dump());
                }else if(req.method=="POST"&&req.path=="/v1/generate"){
                    if(!enable_generate){response(fd,404,"{\"error\":\"generation not enabled (--enable-generate)\"}");--connections;return;}
                    p::GenerateRequest g;
                    try{g=p::parse_generate(strix::serve::parse_json(req.body),capacity);
                        auto supplied=w::unhex(g.key_hex);if(!w::secure_equal(w::hash(std::as_bytes(std::span(supplied.bytes))),w::hash(expected_key)))throw p::RequestError(401,"generate key authentication failed");}
                    catch(const p::RequestError& e){auto j=Json::object();j.set("error",e.what());response(fd,e.code,j.dump());--connections;return;}
                    catch(const std::exception&){response(fd,400,"{\"error\":\"malformed generate request\"}");--connections;return;}
                    std::unique_lock lock(busy,std::try_to_lock);if(!lock.owns_lock()){response(fd,503,"{\"error\":\"producer busy\"}");--connections;return;}
                    try{auto receipt=strix::serve::parse_json(runtime.generate(g.tokens,g.max_new,g.stop,false));identity_fields(receipt);runtime.reset_session();
                        const bool within=finalize_memory(runtime,receipt,"request:generate");response(fd,within?200:507,receipt.dump());} // finalization inside the boundary
                    catch(const std::exception& e){auto j=failed_receipt(runtime,std::string("generation failed; session invalidated: ")+e.what(),"request:generate (failed)");
                        try{runtime.reset_session();}catch(...){}identity_fields(j);response(fd,500,j.dump());}
                }else response(fd,404,"{\"error\":\"unknown endpoint\"}");
            }catch(const std::exception&){try{response(fd,400,"{\"error\":\"prefill rejected or aborted; no global commit receipt\"}");}catch(...){} }
            --connections;
        }).detach();
    }
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
