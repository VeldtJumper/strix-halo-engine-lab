// CPU-only client exercising the production sender against the actual receiver.
#include "glm/rtxpf/producer/sender.hpp"
#include <cstdlib>
#include <iostream>
using namespace strix::glm::rtxpf::producer;
int main(int argc,char** argv){try{
    wire::require(argc==6||argc==7,"sender ENDPOINT KEY PROMPT_SHA MODEL_SHA CASE [RANK]");SocketRuntime runtime;
    wire::Begin b;b.rank=0;b.layers=46;b.frontier=5;b.total=6;b.node="r1";b.producer="glm-pf-rtx-native-v1";b.prompt=wire::unhex(argv[3]);b.model=wire::unhex(argv[4]);b.transaction=wire::unhex(std::string(64,'3'));
    if(argc==7)b.rank=unsigned(std::stoul(argv[6]));
    if(const char* v=std::getenv("GLM_TEST_WIRE_VERSION"))b.version=unsigned(std::stoul(v)); // 2 = DSA once
    Sender sender(argv[1],wire::read_key(argv[2]),b);std::string mode=argv[5];
    if(mode=="abort"){sender.abort();return 0;}
    // Raw packets past the sender's own checks: a DSA frame (layer 3, id 2) as the first frame, or a commit before any frame.
    if(mode=="extra_dsa"){wire::Bytes payload(static_cast<std::size_t>(b.frontier*1024),std::byte{7});
        wire::Bytes frame;wire::number(frame,2,4);wire::number(frame,3,4);wire::number(frame,2,4);wire::number(frame,0);wire::number(frame,b.frontier);wire::number(frame,payload.size());wire::digest(frame,wire::hash(payload));wire::append(frame,payload);
        sender.packet(frame,0);sender.packet(frame,0);return 2;}
    if(mode=="premature"){wire::Bytes commit;wire::number(commit,3,4);wire::number(commit,0);wire::number(commit,0);sender.packet(commit,1);return 2;}
    for(auto c:wire::layout(b)){
        wire::Bytes payload(static_cast<std::size_t>(c.rows*c.stride),std::byte((c.layer*7+c.id)%256));
        if(mode=="corrupt"||mode=="duplicate"){
            wire::Bytes frame;wire::number(frame,2,4);wire::number(frame,c.layer,4);wire::number(frame,c.id,4);wire::number(frame,0);wire::number(frame,c.rows);wire::number(frame,payload.size());wire::digest(frame,wire::hash(payload));wire::append(frame,payload);
            if(mode=="corrupt")frame.back()^=std::byte{1};
            sender.packet(frame,0);sender.packet(frame,0);return 2;
        }
        sender.frame(c.layer,c.id,payload);
    }
    sender.commit();std::cout<<"LOCALLY_SEALED raw="<<sender.raw_bytes()<<" wire="<<sender.wire_bytes()<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
