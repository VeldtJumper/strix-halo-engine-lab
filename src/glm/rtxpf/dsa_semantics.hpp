#pragma once
#include "dsa_selection.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace strix::glm::rtxpf::dsa_host {
// Independent stable sort oracle for pool selection. The partial pool's tokens
// are always trailing; future and partial pools never participate in ranking.
inline std::vector<unsigned> select(const std::vector<float>& scores,unsigned t) {
    unsigned pools=(t+1)/4;
    if(scores.size()<pools)throw std::invalid_argument("short scores");
    std::vector<unsigned> order(pools);std::iota(order.begin(),order.end(),0);
    for(unsigned p=0;p<pools;++p)if(!std::isfinite(scores[p]))throw std::invalid_argument("nonfinite score");
    std::stable_sort(order.begin(),order.end(),[&](unsigned a,unsigned b){return scores[a]>scores[b];});
    order.resize(std::min(dsa_selection_budget,pools));std::sort(order.begin(),order.end());
    std::vector<unsigned> tokens;for(auto p:order)for(unsigned j=0;j<4;++j)tokens.push_back(4*p+j);
    for(unsigned j=4*pools;j<=t;++j)tokens.push_back(j);return tokens;
}
// Score-free selection while every complete pool fits the budget (the decode
// short-context skip): every causal token, as select() returns for any finite scores.
inline std::vector<unsigned> select_causal(unsigned t) {
    if((std::uint64_t(t)+1)/4>dsa_selection_budget)throw std::invalid_argument("pools exceed the selection budget");
    std::vector<unsigned> tokens(t+1);std::iota(tokens.begin(),tokens.end(),0);return tokens;
}
inline bool near_tie(float score,float cutoff,float abs=1.e-3f,float rel=1.e-3f){return std::fabs(score-cutoff)<=abs+rel*std::max(std::fabs(score),std::fabs(cutoff));}
}
