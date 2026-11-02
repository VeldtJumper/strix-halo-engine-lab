#pragma once
// Host-only single-session decode contract (draft identity glm-dec-rtx-native-v1).
// One pending token: for logical sequence S of length N, trunk state holds rows
// S[0..N-2], MTP holds pairs (hidden[j],S[j+1]) for j<N-1, S[N-1] is pending and
// both frontiers equal P=N-1. A step consumes the pending token at absolute
// position P, chooses y, advances MTP with (hidden[P],y), appends y as pending.
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace strix::glm::rtxpf::producer {
inline constexpr const char* decode_identity="glm-dec-rtx-native-v1";
inline constexpr const char* head_arithmetic="lm_head-k6-fp16-h128-rn16-f32-ascending-fmaf";
inline constexpr const char* kda_block_policy="b32-open-block-checkpoint-replay";
inline constexpr unsigned vocab_size=154880,hidden_size=4096,kda_block=32,kda_layer_count=34,dsa_layer_count=12;
inline constexpr std::uint64_t kda_state_bytes=64ULL*128*128*4,kda_conv_bytes=3ULL*64*128*3*4,kda_row_bytes=4096ULL*4;
inline void session_need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
inline bool is_kda_layer(unsigned l){return l<45&&l%4!=3;}
// Dense slot index for per-layer state arrays: KDA 0..33, DSA 0..10 trunk and 11 = MTP.
inline unsigned kda_slot(unsigned l){session_need(is_kda_layer(l),"not a KDA layer");return l-(l+1)/4;}
inline unsigned dsa_slot(unsigned l){session_need(l==45||(l<45&&l%4==3),"not a DSA layer");return l==45?11:l/4;}
inline std::uint64_t dsa_layer_bytes(std::uint64_t capacity){return capacity*1024+2*capacity*256+(capacity/4)*512;}
struct StateBytes {std::uint64_t kda,dsa,replay,total;};
// Capacity allocations only (no weights, workspaces or head).
inline StateBytes session_state_bytes(unsigned capacity){
    StateBytes b{kda_layer_count*(kda_state_bytes+kda_conv_bytes),dsa_layer_count*dsa_layer_bytes(capacity),
                 kda_layer_count*(kda_state_bytes+kda_conv_bytes+kda_block*kda_row_bytes),0};
    b.total=b.kda+b.dsa;return b;
}
// One rank's serialized payload at frontier P (KDA partitioned, DSA replicated).
inline std::uint64_t export_rank_bytes(std::uint64_t P){
    return kda_layer_count*(32*65536ULL+32*4608ULL)+dsa_layer_count*(P*1024+2*P*256+(P/4)*512);
}
// KDA open block. Fresh prefill of P rows processes absolute B32 blocks; the
// last (possibly empty) block [P-P%32,P) is the open block. Incremental decode
// restores the block-start checkpoint and replays open rows plus the new one.
inline unsigned kda_block_start(unsigned P){return P-P%kda_block;}
inline unsigned kda_open_rows(unsigned P){return P%kda_block;}
inline unsigned kda_replay_rows(unsigned P){return P%kda_block+1;}
inline bool kda_promotes(unsigned P){return P%kda_block==kda_block-1;}
// DSA one-row append at absolute position `first`.
inline bool dsa_completes_pool(unsigned first){return (first+1)%4==0;}
inline unsigned dsa_scored_pools(unsigned first){return (first+1)/4;}
inline constexpr unsigned dsa_attended_bound=2051;
inline unsigned dsa_attended_tokens(unsigned first){auto pools=dsa_scored_pools(first);return pools<=512?first+1:2048+(first+1)%4;}
// Prompt-lookup drafts (local stand-in for the shared n-gram component): the longest suffix of `tokens`
// (max_match down to min_match tokens) that occurred earlier, most recent occurrence first; returns up to k
// tokens that followed it (empty: no match). Drafts only propose; the greedy verify decides every token.
inline std::vector<std::uint32_t> lookup_draft(std::span<const std::uint32_t> tokens,unsigned k,unsigned min_match=3,unsigned max_match=8){
    const std::size_t L=tokens.size();
    for(std::size_t n=std::min<std::size_t>(max_match,L>0?L-1:0);n>=min_match&&n>0;--n)
        for(std::size_t i=L-n;i-->0;){
            bool same=true;for(std::size_t j=0;j<n&&same;++j)same=tokens[i+j]==tokens[L-n+j];
            if(same)return std::vector<std::uint32_t>(tokens.begin()+std::ptrdiff_t(i+n),tokens.begin()+std::ptrdiff_t(std::min(i+n+k,L)));
        }
    return {};
}
class SessionFrontier {
    unsigned capacity_=0;std::vector<std::uint32_t> tokens_;bool open_=false,failed_=false;
public:
    explicit SessionFrontier(unsigned capacity=0):capacity_(capacity){}
    unsigned capacity()const{return capacity_;}
    bool open()const{return open_&&!failed_;}
    bool failed()const{return failed_;}
    unsigned frontier()const{session_need(open_,"no open session");return unsigned(tokens_.size()-1);}
    std::uint32_t pending()const{session_need(open_,"no open session");return tokens_.back();}
    const std::vector<std::uint32_t>& tokens()const{return tokens_;}
    static void validate_prompt(std::span<const std::uint32_t> prompt,unsigned capacity){
        session_need(prompt.size()>=2&&prompt.size()<=capacity,"prompt length must be 2..capacity");
        for(auto t:prompt)session_need(t<vocab_size,"token outside vocabulary");
    }
    // Generation never exceeds what a fresh prefill of prompt+generated could admit.
    static void validate_generation(std::size_t prompt_length,std::uint64_t max_new,unsigned capacity){
        session_need(max_new>=1&&prompt_length+max_new<=capacity,"prompt_length + max_new_tokens exceeds capacity");
    }
    void begin(std::span<const std::uint32_t> prompt){validate_prompt(prompt,capacity_);tokens_.assign(prompt.begin(),prompt.end());open_=true;failed_=false;}
    bool can_step()const{return open()&&tokens_.size()<capacity_;}
    // Position consumed by the next step.
    unsigned position()const{session_need(can_step(),"session cannot step (closed, failed or full)");return frontier();}
    void commit(std::uint32_t y){session_need(can_step(),"commit on non-steppable session");session_need(y<vocab_size,"token outside vocabulary");tokens_.push_back(y);}
    void fail(){failed_=true;}
    void reset(){tokens_.clear();open_=false;failed_=false;}
};
}
