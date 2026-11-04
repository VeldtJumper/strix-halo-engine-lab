#pragma once
// The --decode-commands protocol: one JSON object per stdin line (the decode
// qualification harness, decode_qualification_window.py decode_command()). Every
// field the harness or an in-process A/B script may send is listed here, per op (row_kernels / decode_profile / dsa_short_skip:
// any work op, they stay set for later commands); the service loop and the
// CPU-only --check-decode-commands mode both parse with this function, so the
// suite drives the real binary's parser with every request shape the harness plans.
#include "strix/serve/json.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace strix::glm::rtxpf::producer {
struct DecodeCommand {
    std::string op,prompt,export_dir;
    unsigned steps=0,row_tile=0;   // row_tile 0 = the service's per-request rule
    std::string row_kernels;       // decode_row_kernels for this and later commands ("" = unchanged; the runtime validates)
    int dsa_short_skip=-1;         // 0 / 1 for this and later commands; -1 = unchanged
    int decode_profile=-1;         // decode timeline on (1) / off (0) for this and later commands; -1 = unchanged
    strix::serve::Json order;      // bench: the two arm labels
    bool spec=false;               // bench / trace: decode with self-speculation (needs --mtp-self-spec 1)
    int lookup=-1;                 // prompt-lookup drafts on (1) / off (0) for this and later commands; -1 = unchanged
};
inline constexpr const char* decode_command_fields[]={"op","prompt","steps","order","export_dir","row_tile","row_kernels","decode_profile","dsa_short_skip","spec","lookup"};
inline DecodeCommand parse_decode_command(const strix::serve::Json& j){
    auto fail=[](const std::string& why){throw std::runtime_error("decode command: "+why);};
    if(!j.is_object())fail("expected a JSON object");
    for(const auto& [key,value]:j.as_object()){(void)value;
        if(std::find_if(std::begin(decode_command_fields),std::end(decode_command_fields),[&](const char* f){return key==f;})==std::end(decode_command_fields))
            fail("unsupported request field: "+key);}
    auto* op=j.find("op");if(!op||!op->is_string())fail("missing op");
    DecodeCommand c;c.op=op->as_string();
    if(c.op!="quit"&&c.op!="qualify"&&c.op!="bench"&&c.op!="compare"&&c.op!="ttft"&&c.op!="trace")fail("unknown op: "+c.op);
    auto only=[&](const char* f,const char* ops){if(j.find(f)&&std::string(ops).find(" "+c.op+" ")==std::string::npos)fail(std::string(f)+" is not a field of op "+c.op);};
    only("order"," bench ");only("export_dir"," qualify ");only("row_tile"," ttft ");
    only("row_kernels"," qualify bench compare ttft trace ");only("decode_profile"," qualify bench compare ttft trace ");
    only("dsa_short_skip"," qualify bench compare ttft trace ");
    only("spec"," bench trace ");only("lookup"," qualify bench compare ttft trace ");
    if(c.op=="quit"){if(j.as_object().size()!=1)fail("quit takes no fields");return c;}
    auto* prompt=j.find("prompt");if(!prompt||!prompt->is_string()||prompt->as_string().empty())fail("missing prompt");c.prompt=prompt->as_string();
    auto* steps=j.find("steps");if(!steps||!steps->is_number())fail("missing steps");
    const auto s=steps->as_int();if(s<1||s>64)fail("steps must be 1..64");c.steps=unsigned(s);
    if(c.op=="bench"){auto* o=j.find("order");if(!o||!o->is_array()||o->as_array().size()!=2)fail("bench needs order: two arm labels");c.order=*o;}
    if(auto* e=j.find("export_dir")){if(!e->is_string())fail("export_dir must be a string");c.export_dir=e->as_string();}
    if(auto* k=j.find("row_kernels")){if(!k->is_string()||k->as_string().empty())fail("row_kernels must be a non-empty string (0, 1 or a + subset of moe, hc, router, kda, dsa)");c.row_kernels=k->as_string();}
    if(auto* d=j.find("dsa_short_skip")){if(!d->is_number()||(d->as_double()!=0&&d->as_double()!=1))fail("dsa_short_skip must be 0 or 1");c.dsa_short_skip=int(d->as_int());}
    if(auto* d=j.find("decode_profile")){if(!d->is_bool())fail("decode_profile must be true or false");c.decode_profile=d->as_bool()?1:0;}
    if(auto* p=j.find("spec")){if(!p->is_bool())fail("spec must be true or false");c.spec=p->as_bool();}
    if(auto* l=j.find("lookup")){if(!l->is_bool())fail("lookup must be true or false");c.lookup=l->as_bool()?1:0;}
    if(auto* t=j.find("row_tile")){if(!t->is_number())fail("row_tile must be an integer");const auto v=t->as_int();
        if(v<32||v>32768||v%32)fail("row_tile must be a multiple of 32 in 32..32768 (0/absent = the per-request rule)");c.row_tile=unsigned(v);}
    return c;
}
}
