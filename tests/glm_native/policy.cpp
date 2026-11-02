#include "glm/rtxpf/producer/policy.hpp"
#include "glm/rtxpf/producer/dgpu.hpp"
#include "glm/rtxpf/producer/schedule.hpp"
#include "glm/rtxpf/producer/state_slices.hpp"
#include <iostream>
#include <vector>
using namespace strix::glm::rtxpf::producer;
void check(bool b){if(!b)throw std::runtime_error("policy check failed");}
int main(){
    for(auto invalid:{"", "0", "0,1", "GPU-123", "MIG-123", "GPU-00000000-0000-0000-0000-000000000000,GPU-11111111-1111-1111-1111-111111111111"}){
        bool rejected=false;try{require_pro_uuid(invalid);}catch(...){rejected=true;}check(rejected);
    }
    check(!valid_schedule(131076,256)&&!valid_schedule(131072,33)&&!valid_schedule(131072,65536));
    check(effective_row_tile(8192,0)==256&&effective_row_tile(32768,0)==2048&&effective_row_tile(131072,256)==256);
    // Planner-chosen default tile candidates: largest first, every one a valid schedule.
    check(first_tile_candidate(8320)==8320&&first_tile_candidate(32768)==32768&&first_tile_candidate(131072)==32768&&first_tile_candidate(8200)==8192);
    {std::vector<unsigned> chain;for(unsigned t=first_tile_candidate(8320);t;t=next_tile_candidate(t))chain.push_back(t);
        check((chain==std::vector<unsigned>{8320,4160,2080,1024,512,256,128,64,32}));}
    for(unsigned capacity:{32u,1024u,8320u,32768u,65536u,131072u})for(unsigned t=first_tile_candidate(capacity);t;t=next_tile_candidate(t))check(valid_schedule(capacity,t));
    check(largest_fitting_tile(32768,[](unsigned t){return t<=5000;})==4096&&largest_fitting_tile(8320,[](unsigned){return true;})==8320&&largest_fitting_tile(8320,[](unsigned){return false;})==0);
    for(unsigned tile:{32u,256u,1024u,2048u})for(unsigned capacity:{8192u,32768u,131072u}){
        check(valid_schedule(capacity,tile));
        // Exercise partial pools, exact/tail graph buckets, B32 carry boundaries
        // and the maximum prefix, without allocating a prefix-sized test buffer.
        for(unsigned prefix:{1u,3u,31u,32u,33u,tile-1,tile,tile+1,capacity-1}){
            unsigned covered=0,pools=0;
            for(unsigned first=0;first<prefix;first+=tile){auto c=row_tile(first,prefix,tile);check(c.first==covered&&c.first%32==0&&c.rows>0&&c.rows<=tile&&c.first+c.rows<=prefix);check(c.bucket==(c.rows==tile?0u:1u));pools+=(c.first+c.rows)/4-c.first/4;covered+=c.rows;}
            check(covered==prefix&&pools==prefix/4);
        }
        // A larger capacity adds only residual/hidden rows and token IDs;
        // changing tile size affects scratch without multiplying by capacity.
        if(capacity<131072)check(activation_bytes(capacity*2,tile)-activation_bytes(capacity,tile)==std::uint64_t(capacity)*(4096*4*5+4));
        check(activation_bytes(capacity,tile)>std::uint64_t(capacity)*4096*4*5);
    }
    check(activation_bytes(131072,256)<11ULL*1024*1024*1024);
    require_pro_uuid("GPU-00000000-0000-0000-0000-000000000000");
    // --dgpu (dgpu.hpp): only an sm_120 RTX PRO is eligible (checked from NVML before CUDA); auto = the eligible card
    // with the most memory (ties: the lower UUID); off, malformed values, unknown/ineligible cards, NVML gaps and a
    // conflicting or empty CUDA_VISIBLE_DEVICES are refused.
    {auto card=[](const char* u,const char* n,std::uint64_t gib,int major,int minor){dgpu::Card c;c.uuid=u;c.name=n;c.memory=gib<<30;c.memory_known=true;
         c.cc_major=major;c.cc_minor=minor;c.eligible_known=true;c.eligible=dgpu::eligible_card(major,minor,n);return c;};
     const char* pro="GPU-afb1d028-f002-6b1c-3c0f-3b55bae23884";const char* g4090="GPU-d55f5e5a-80f9-ef74-342b-8baaa4e7f101";
     dgpu::Inventory inv;inv.ok=true;inv.cards={card(g4090,"NVIDIA GeForce RTX 4090",24,8,9),card(pro,"NVIDIA RTX PRO 6000 Blackwell Workstation Edition",96,12,0)};
     auto refused=[](auto f){try{(void)f();}catch(const std::exception&){return true;}return false;};
     check(dgpu::resolve("auto",inv,nullptr).uuid==pro&&dgpu::resolve(pro,inv,nullptr).uuid==pro);
     check(refused([&]{return dgpu::resolve(g4090,inv,nullptr);}));                        // the 4090 is never chosen, even by name
     {auto big=inv;big.cards[0]=card(g4090,"NVIDIA GeForce RTX 4090",192,8,9);check(dgpu::resolve("auto",big,nullptr).uuid==pro);} // auto ignores larger ineligible cards
     {auto only=inv;only.cards.erase(only.cards.begin()+1);check(refused([&]{return dgpu::resolve("auto",only,nullptr);}));}
     {dgpu::Inventory tie;tie.ok=true;tie.cards={card("GPU-00000000-0000-0000-0000-000000000002","RTX PRO b",8,12,0),card("GPU-00000000-0000-0000-0000-000000000001","RTX PRO a",8,12,0)};
      check(dgpu::resolve("auto",tie,nullptr).name=="RTX PRO a");}                        // ties: the lower UUID, whatever the order
     check(dgpu::resolve("GPU-AFB1D028-F002-6B1C-3C0F-3B55BAE23884",inv,nullptr).uuid==pro); // hex case is normalized
     for(auto bad:{"off","","0","GPU-123","GPU-00000000-0000-0000-0000-00000000000z","gpu-afb1d028-f002-6b1c-3c0f-3b55bae23884"})check(refused([&]{return dgpu::resolve(bad,inv,nullptr);}));
     check(refused([&]{return dgpu::resolve("GPU-00000000-0000-0000-0000-000000000000",inv,nullptr);})); // not listed
     // NVML gaps: unavailable, a device it could not identify (auto only), unknown capability/memory
     {dgpu::Inventory none;none.error="nvml.dll not found";check(refused([&]{return dgpu::resolve(pro,none,nullptr);})&&refused([&]{return dgpu::resolve("auto",none,nullptr);}));}
     {auto part=inv;part.failed=1;check(refused([&]{return dgpu::resolve("auto",part,nullptr);})&&dgpu::resolve(pro,part,nullptr).uuid==pro);}
     {auto cc=inv;cc.cards[1].eligible_known=false;check(refused([&]{return dgpu::resolve(pro,cc,nullptr);})&&refused([&]{return dgpu::resolve("auto",cc,nullptr);}));}
     {auto mem=inv;mem.cards[0].memory_known=false;check(refused([&]{return dgpu::resolve("auto",mem,nullptr);}));
      mem=inv;mem.cards[1].memory_known=false;check(refused([&]{return dgpu::resolve("auto",mem,nullptr);})&&dgpu::resolve(pro,mem,nullptr).uuid==pro);} // explicit: memory not needed
     // apply(): the chosen UUID is written; a failed environment write is an error, never a silent success
     {static std::string written;written.clear();
      check(dgpu::apply("auto",inv,nullptr,[](const char* u){written=u;return true;}).uuid==pro&&written==pro);
      check(refused([&]{return dgpu::apply("auto",inv,nullptr,[](const char*){return false;});}));
      written.clear();check(refused([&]{return dgpu::apply(g4090,inv,nullptr,[](const char* u){written=u;return true;});})&&written.empty());} // refused before any write
     // CUDA_VISIBLE_DEVICES with --dgpu: absent or the same UUID (any hex case) pass; empty, another card, an index, a list or a MIG id are refused
     check(dgpu::resolve("auto",inv,"GPU-AFB1D028-f002-6b1c-3c0f-3b55bae23884").uuid==pro);
     for(const char* env:{"",g4090,"0","GPU-afb1d028-f002-6b1c-3c0f-3b55bae23884,GPU-d55f5e5a-80f9-ef74-342b-8baaa4e7f101","MIG-afb1d028-f002-6b1c-3c0f-3b55bae23884"})
         check(refused([&]{return dgpu::resolve("auto",inv,env);}));
     // Without --dgpu: an ineligible card named by CUDA_VISIBLE_DEVICES is refused before CUDA; unknown or NVML-less passes to the runtime's check
     check(refused([&]{dgpu::check_visible(g4090,inv);return 0;}));
     dgpu::check_visible(pro,inv);dgpu::check_visible("GPU-00000000-0000-0000-0000-000000000000",inv);dgpu::check_visible(g4090,dgpu::Inventory{});}
    check(targets(8192).device==20&&targets(32768).device==19&&targets(131072).device==14);
    check(layer_limit("0")==0&&layer_limit("43")==43);
    check(!caps_require_file_tier(32768,43,43)&&caps_require_file_tier(32768,43,0));
    check(caps_require_file_tier(32768,19,22)&&!caps_require_file_tier(32768,19,24));
    for(auto bad:{"-1","44","1x","9999999999999999",""}){bool rejected=false;try{layer_limit(bad);}catch(...){rejected=true;}check(rejected);}
    check(!file_tier_covered(0,0,0,false)&&!file_tier_covered(2,0,3,true));
    check(!file_tier_covered(1,1024,1,true)&&!file_tier_covered(2,1024,1,true));
    check(file_tier_covered(2,1024,3,true)&&file_tier_covered(1,1024,1,false));
    check(!admit(1,0,0)&&!admit(1,10,10)&&admit(10,20,10));
    check(host_budget(60ULL<<30,50ULL<<30)==42ULL<<30);
    check(host_budget(60ULL<<30,7ULL<<30)==0);
    check(host_budget(0,~0ULL)==0);
    check(host_budget(60ULL<<30,50ULL<<30,16)==34ULL<<30);
    check(host_budget(60ULL<<30,15ULL<<30,16)==0);
    check(host_budget(60ULL<<30,50ULL<<30,0)==50ULL<<30);
    std::vector<std::byte> conv(64*4608),state(64*65536);
    for(unsigned part=0;part<3;++part)for(unsigned head=0;head<64;++head)
        std::fill_n(conv.begin()+(part*64+head)*1536,1536,std::byte(part*64+head));
    for(unsigned head=0;head<64;++head)std::fill_n(state.begin()+head*65536,65536,std::byte(head));
    for(unsigned rank=0;rank<2;++rank){auto c=kda_tp_conv(conv,rank);auto s=kda_tp_state(state,rank);
        for(unsigned part=0;part<3;++part)for(unsigned head=0;head<32;++head){auto b=c.begin()+(part*32+head)*1536;check(std::all_of(b,b+1536,[&](auto v){return v==std::byte(part*64+rank*32+head);}));}
        for(unsigned head=0;head<32;++head)check(s[head*65536]==std::byte(rank*32+head));
    }
    std::cout<<"PASS 128K row schedule/pool coverage, bounded activation accounting, UUID, --dgpu choice, tier targets, VRAM reserve, physical/commit admission, TP2 state/convolution slices\n";
}
