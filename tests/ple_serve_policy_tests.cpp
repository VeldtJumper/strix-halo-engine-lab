#include "strix/ple_serve_policy.hpp"

#include <cstdint>
#include <iostream>

int main() {
    int failures=0;
    const auto check=[&](std::uint64_t available,std::uint64_t registered,std::uint64_t table,
        bool reserve,std::uint64_t expected) {
        if(strix::ple_detail::serve_available(available,registered,table,reserve)!=expected) {
            std::cerr << "FAIL: PLE serve available subtraction\n";
            ++failures;
        }
    };
    check(100,20,30,true,50);
    check(100,20,30,false,80);
    check(100,20,0,true,80); // not warmed
    check(0,20,30,true,0);
    check(100,100,30,true,0);
    check(100,101,30,false,0);
    check(100,20,80,true,0);
    check(100,20,81,true,0);
    check(100,0,30,true,70);
    check(UINT64_MAX,UINT64_MAX-1U,UINT64_MAX,true,0);
    check(UINT64_MAX,1,UINT64_MAX,false,UINT64_MAX-1U);
    return failures ? 1 : 0;
}
