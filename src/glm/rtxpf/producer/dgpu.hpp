#pragma once
// --dgpu auto|off|<uuid>: which discrete GPU the GLM RTX service runs on, resolved before any CUDA call by setting
// CUDA_VISIBLE_DEVICES (CUDA then sees only that card). The service runs only on an sm_120 RTX PRO (the rule of
// Runtime's check_device), so a card is eligible only when NVML shows compute capability 12.0 and "PRO" in its name;
// that is checked here, through NVML (no CUDA context on any card), before the runtime creates its streams.
// auto = the eligible card with the most memory (ties: the lower UUID); <uuid> must be an eligible card; off = refuse
// (this service has no CPU/iGPU path). An explicit CUDA_VISIBLE_DEVICES must be the same full UUID or is refused.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace strix::glm::rtxpf::producer::dgpu {

// One NVML device: the UUID always (a device without one is not listed); memory and eligibility only when NVML
// answered (a failed query leaves *_known false, never a guess).
struct Card {std::string uuid,name;std::uint64_t memory=0;int cc_major=-1,cc_minor=-1;bool memory_known=false,eligible_known=false,eligible=false;};
// The enumeration: ok = NVML loaded, initialised and counted; error says why not; failed = devices NVML counted
// but could not identify (handle/UUID query failed).
struct Inventory {bool ok=false;std::string error;std::vector<Card> cards;unsigned failed=0;};

inline bool is_uuid(const std::string& v){
    return std::regex_match(v,std::regex("GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"));}
// The canonical spelling NVML reports: "GPU-" + lowercase hex.
inline std::string canonical(std::string v){for(std::size_t i=4;i<v.size();++i)v[i]=char(std::tolower(static_cast<unsigned char>(v[i])));return v;}
// Runtime::check_device's rule, from NVML: sm_120 and a PRO name.
inline bool eligible_card(int major,int minor,const std::string& name){return major==12&&minor==0&&name.find("PRO")!=std::string::npos;}

// The card for a --dgpu value, given the inventory and CUDA_VISIBLE_DEVICES (nullptr = unset). Pure: no NVML, no
// environment access. Throws with the reason on every refusal.
inline Card resolve(const std::string& value,const Inventory& inv,const char* visible){
    if(value=="off")throw std::runtime_error("--dgpu off: the GLM RTX service runs only on a discrete GPU (an sm_120 RTX PRO)");
    if(value!="auto"&&!is_uuid(value))throw std::runtime_error("--dgpu must be auto, off or a full GPU UUID (GPU-xxxxxxxx-...) from nvidia-smi -L");
    if(!inv.ok)throw std::runtime_error("--dgpu "+value+": cannot check the card before CUDA starts (NVML: "+inv.error+")");
    Card chosen;
    if(value=="auto"){
        if(inv.failed)throw std::runtime_error("--dgpu auto: NVML could not identify "+std::to_string(inv.failed)+" device(s); pass --dgpu <uuid>");
        std::vector<const Card*> pool;
        for(const auto& c:inv.cards){
            if(!c.eligible_known||!c.memory_known)throw std::runtime_error("--dgpu auto: NVML did not report the compute capability/memory of "+c.uuid+"; pass --dgpu <uuid>");
            if(c.eligible)pool.push_back(&c);}
        if(pool.empty())throw std::runtime_error("--dgpu auto: no sm_120 RTX PRO among the "+std::to_string(inv.cards.size())+" NVIDIA device(s)");
        chosen=**std::min_element(pool.begin(),pool.end(),[](const Card* a,const Card* b){return a->memory!=b->memory?a->memory>b->memory:a->uuid<b->uuid;});
    }else{
        const auto want=canonical(value);const Card* hit=nullptr;
        for(const auto& c:inv.cards)if(canonical(c.uuid)==want)hit=&c;
        if(!hit)throw std::runtime_error("--dgpu "+value+": no such GPU"+(inv.failed?" among the devices NVML could identify ("+std::to_string(inv.failed)+" could not be)":"")+" (nvidia-smi -L lists the UUIDs)");
        if(!hit->eligible_known)throw std::runtime_error("--dgpu "+value+": NVML did not report its compute capability; cannot check it is an sm_120 RTX PRO");
        if(!hit->eligible)throw std::runtime_error("--dgpu "+value+" ("+hit->name+", sm_"+std::to_string(hit->cc_major)+std::to_string(hit->cc_minor)+"): the GLM RTX service runs only on an sm_120 RTX PRO");
        chosen=*hit;}
    chosen.uuid=canonical(chosen.uuid);
    if(visible){
        if(!*visible)throw std::runtime_error("--dgpu "+value+": CUDA_VISIBLE_DEVICES is set but empty (hides every device); unset it or set it to "+chosen.uuid);
        if(canonical(visible)!=chosen.uuid)
            throw std::runtime_error("--dgpu "+value+" chose "+chosen.uuid+" but CUDA_VISIBLE_DEVICES="+visible+"; with both set they must name the same card by its full UUID (no index, list or MIG id)");}
    return chosen;}

// NVIDIA devices through NVML (nvml.dll / libnvidia-ml.so.1): no CUDA context. NVML failures come back in the
// Inventory (ok / error / failed / *_known), never as an exception; only allocation failures can propagate.
inline Inventory enumerate(){
    Inventory inv;
    using Init=int(*)();using Count=int(*)(unsigned*);using Handle=int(*)(unsigned,void**);using Text=int(*)(void*,char*,unsigned);
    struct Mem{unsigned long long total,free,used;};using MemInfo=int(*)(void*,Mem*);using Cc=int(*)(void*,int*,int*);using Shut=int(*)();
#ifdef _WIN32
    HMODULE lib=LoadLibraryA("nvml.dll");if(!lib){inv.error="nvml.dll not found";return inv;}
    auto sym=[&](const char* n){return reinterpret_cast<void*>(GetProcAddress(lib,n));};
    struct Unload{HMODULE l;~Unload(){FreeLibrary(l);}} unload{lib};
#else
    void* lib=dlopen("libnvidia-ml.so.1",RTLD_NOW);if(!lib){inv.error="libnvidia-ml.so.1 not found";return inv;}
    auto sym=[&](const char* n){return dlsym(lib,n);};
    struct Unload{void* l;~Unload(){dlclose(l);}} unload{lib};
#endif
    auto init=reinterpret_cast<Init>(sym("nvmlInit_v2"));auto count=reinterpret_cast<Count>(sym("nvmlDeviceGetCount_v2"));
    auto handle=reinterpret_cast<Handle>(sym("nvmlDeviceGetHandleByIndex_v2"));auto uuid=reinterpret_cast<Text>(sym("nvmlDeviceGetUUID"));
    auto name=reinterpret_cast<Text>(sym("nvmlDeviceGetName"));auto mem=reinterpret_cast<MemInfo>(sym("nvmlDeviceGetMemoryInfo"));
    auto cc=reinterpret_cast<Cc>(sym("nvmlDeviceGetCudaComputeCapability"));auto shut=reinterpret_cast<Shut>(sym("nvmlShutdown"));
    if(!init||!count||!handle||!uuid||!shut){inv.error="NVML lacks the device query entry points";return inv;}
    if(const int s=init();s!=0){inv.error="nvmlInit_v2 failed ("+std::to_string(s)+")";return inv;}
    struct Shutdown{Shut f;~Shutdown(){f();}} shutdown{shut};
    unsigned n=0;if(const int s=count(&n);s!=0){inv.error="nvmlDeviceGetCount_v2 failed ("+std::to_string(s)+")";return inv;}
    inv.ok=true;
    for(unsigned i=0;i<n;++i){void* h=nullptr;char u[96]{},nm[96]{};
        if(handle(i,&h)!=0||uuid(h,u,sizeof u)!=0){++inv.failed;continue;}
        Card c;c.uuid=u;if(name&&name(h,nm,sizeof nm)==0)c.name=nm;
        Mem m{};if(mem&&mem(h,&m)==0){c.memory=m.total;c.memory_known=true;}
        int major=-1,minor=-1;if(cc&&!c.name.empty()&&cc(h,&major,&minor)==0){c.cc_major=major;c.cc_minor=minor;c.eligible_known=true;c.eligible=eligible_card(major,minor,c.name);}
        inv.cards.push_back(c);}
    return inv;}

// Sets CUDA_VISIBLE_DEVICES for this process; false when the environment write fails.
inline bool set_visible(const char* uuid){
#ifdef _WIN32
    return _putenv_s("CUDA_VISIBLE_DEVICES",uuid)==0;
#else
    return setenv("CUDA_VISIBLE_DEVICES",uuid,1)==0;
#endif
}
// Resolves the card and writes it with set_env (tests inject the inventory, the environment and the writer).
inline Card apply(const std::string& value,const Inventory& inv,const char* visible,bool(*set_env)(const char*)){
    const Card c=resolve(value,inv,visible);
    if(!set_env(c.uuid.c_str()))throw std::runtime_error("--dgpu: could not set CUDA_VISIBLE_DEVICES="+c.uuid);
    return c;}
// Applies --dgpu before any CUDA call: resolves the card and sets CUDA_VISIBLE_DEVICES to its UUID. Returns the card
// for the startup line.
inline Card apply(const std::string& value){
    return apply(value,value=="off"?Inventory{}:enumerate(),std::getenv("CUDA_VISIBLE_DEVICES"),set_visible);}

// Without --dgpu: the card CUDA_VISIBLE_DEVICES names must be an sm_120 RTX PRO when NVML can tell (refused here,
// before the runtime creates any CUDA resource on it); when NVML cannot tell, the runtime's own check decides.
inline void check_visible(const std::string& visible,const Inventory& inv){
    if(!inv.ok)return;
    for(const auto& c:inv.cards)if(canonical(c.uuid)==canonical(visible)&&c.eligible_known&&!c.eligible)
        throw std::runtime_error("CUDA_VISIBLE_DEVICES="+visible+" ("+c.name+", sm_"+std::to_string(c.cc_major)+std::to_string(c.cc_minor)+"): the GLM RTX service runs only on an sm_120 RTX PRO");}

}  // namespace strix::glm::rtxpf::producer::dgpu
