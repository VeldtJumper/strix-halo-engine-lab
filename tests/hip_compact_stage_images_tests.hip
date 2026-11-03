#include "strix/hip_compact_qsa_diagnostic.hpp"
#include <hip/hip_runtime.h>
#include "../src/hip_rounded_stage_image.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
struct Buffer {
    void* data=nullptr;
    explicit Buffer(std::size_t bytes) {check(hipMalloc(&data,bytes));}
    ~Buffer() {if(data) (void)hipFree(data);}
    template<class T> T* as() const {return static_cast<T*>(data);}
};
constexpr unsigned width=6144U, hidden=2560U, maximum=2052U;
__global__ void initialize(float* query, std::uint16_t* kv, std::uint32_t* selected,
    unsigned char* weights, unsigned rows) {
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<static_cast<std::uint64_t>(rows)*width*2U) {
        // BF16-rounded queries and gates, with both signs and zeros.
        const auto b=static_cast<unsigned>((i*17U+3U)%129U);
        const float x=(static_cast<float>(b)-64.0F)/256.0F;
        query[i]=__uint_as_float(__float_as_uint(x)&0xffff0000U);
    }
    if(i<32U*1024U) kv[i]=static_cast<std::uint16_t>(0x3c00U+((i*13U)%256U)+(i%2U?0x8000U:0U));
    if(i<static_cast<std::uint64_t>(rows)*maximum) selected[i]=static_cast<unsigned>(i%maximum)%32U;
    if(i<hidden*(width/32U)*34U) {
        const auto at=i%34U;
        weights[i]=at==0U?0U:at==1U?0x28U:static_cast<unsigned char>(static_cast<int>((i*7U)%17U)-8);
    }
}
__global__ void initialize_float_weights(float* weights) {
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<static_cast<std::uint64_t>(width)*hidden) weights[i]=(static_cast<float>(i%17U)-8.0F)/256.0F;
}
__global__ void initialize_scores(float* scores, float* query, std::uint64_t elements, bool special) {
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<elements) scores[i]=(i%maximum<32U)?1.0F/32.0F:0.0F;
    if(special && i<16U) {
        const unsigned words[16]={0U,0x80000000U,0x7f800000U,0xff800000U,0x7fc10000U,0xffc10000U,
            0x7f810000U,0xff810000U,0x00010000U,0x80010000U,0x7fff0000U,0xffff0000U,
            0x47800000U,0xc7800000U,0x3f800000U,0xbf800000U};
        query[256U+i]=__uint_as_float(words[i]);
        if(i==0U) scores[0]=__uint_as_float(0x7fc10000U);
    }
}
__global__ void initialize_int8(std::uint16_t* kv) {
    const unsigned token=blockIdx.x,feature=threadIdx.x;
    auto* row=reinterpret_cast<unsigned char*>(kv)+token*1040U;
    for(unsigned i=feature;i<1024U;i+=256U) reinterpret_cast<std::int8_t*>(row)[i]=static_cast<std::int8_t>(static_cast<int>(i%17U)-8);
    if(feature<4U) reinterpret_cast<float*>(row+1024U)[feature]=1.0F/64.0F;
}
__global__ void special_bits(std::uint32_t* fp32, std::uint16_t* compact) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=65536U) return;
    // Exercise exactly the producer's integer rounding, then both production stores.
    for(unsigned low=0;low<5U;++low) {
        const unsigned tails[5]={0U,1U,0x7fffU,0x8000U,0xffffU};
        const unsigned raw=(i<<16U)|tails[low];
        const float rounded=__uint_as_float((raw+0x7fffU+((raw>>16U)&1U))&0xffff0000U);
        strix::RoundedStageImage a(reinterpret_cast<float*>(fp32)),b(reinterpret_cast<float*>(compact),true);
        a[i*5U+low]=rounded;b[i*5U+low]=rounded;
    }
}
__global__ void read_special_bits(const std::uint16_t* compact, std::uint32_t* widened) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=65536U*5U) return;
    strix::RoundedStageImage image(reinterpret_cast<float*>(const_cast<std::uint16_t*>(compact)),true);
    widened[i]=__float_as_uint(static_cast<float>(image[i]));
}
void special_fixture(hipStream_t stream) {
    Buffer reference(65536U*5U*4U),compact(65536U*5U*2U),widened(65536U*5U*4U);
    hipLaunchKernelGGL(special_bits,dim3(256),dim3(256),0,stream,reference.as<std::uint32_t>(),compact.as<std::uint16_t>());
    hipLaunchKernelGGL(read_special_bits,dim3(1280),dim3(256),0,stream,compact.as<std::uint16_t>(),widened.as<std::uint32_t>());
    std::vector<std::uint32_t> a(65536U*5U),wide(a.size());std::vector<std::uint16_t> b(a.size());
    check(hipStreamSynchronize(stream));
    check(hipMemcpy(a.data(),reference.data,a.size()*4U,hipMemcpyDeviceToHost));
    check(hipMemcpy(b.data(),compact.data,b.size()*2U,hipMemcpyDeviceToHost));
    check(hipMemcpy(wide.data(),widened.data,wide.size()*4U,hipMemcpyDeviceToHost));
    if(a!=wide) throw std::runtime_error("device widening changed special bits");
    for(std::size_t i=0;i<a.size();++i) if(a[i]!=(static_cast<unsigned>(b[i])<<16U))
        throw std::runtime_error("special value bits differ");
}
void run(unsigned rows, hipStream_t stream) {
    const auto padded=(rows+63U)/64U*64U;
    const auto image_elements=static_cast<std::size_t>(padded)*width;
    const auto score_elements=static_cast<std::size_t>(padded)*24U*maximum;
    Buffer query(image_elements*8U),kv(32U*1024U*2U),selected(static_cast<std::size_t>(padded)*maximum*4U),
        scores(score_elements*4U),weights(hidden*(width/32U)*34U),float_weights(static_cast<std::size_t>(hidden)*width*4U),
        ref_image(image_elements*4U+128U),small_image(image_elements*2U+128U),
        ref_output(static_cast<std::size_t>(padded)*hidden*2U),small_output(static_cast<std::size_t>(padded)*hidden*2U),status(8U),quantized(8U*width+192U*8U*4U);
    const auto blocks=static_cast<unsigned>((image_elements*2U+255U)/256U);
    hipLaunchKernelGGL(initialize,dim3(blocks),dim3(256),0,stream,query.as<float>(),kv.as<std::uint16_t>(),
        selected.as<std::uint32_t>(),weights.as<unsigned char>(),padded);
    hipLaunchKernelGGL(initialize_float_weights,dim3((hidden*width+255U)/256U),dim3(256),0,stream,float_weights.as<float>());
    std::vector<std::uint32_t> ref(static_cast<std::size_t>(rows)*width);
    std::vector<std::uint16_t> small(ref.size()),a(static_cast<std::size_t>(padded)*hidden),b(a.size());
    for(unsigned mode=0;mode<13U;++mode) for(unsigned projection=0;projection<5U;++projection) {
        unsigned statuses[2][2]{};
        std::vector<unsigned char> quantized_a(8U*width+192U*8U*4U),quantized_b(quantized_a.size());
        if(mode>=10U) hipLaunchKernelGGL(initialize_int8,dim3(32),dim3(256),0,stream,kv.as<std::uint16_t>());
        for(unsigned arm=0;arm<2U;++arm) {
            auto& image=arm?small_image:ref_image;auto& output=arm?small_output:ref_output;
            check(hipMemsetAsync(image.data,0xa5,image_elements*(arm?2U:4U)+128U,stream));
            check(hipMemsetAsync(output.data,0xa5,a.size()*2U,stream));check(hipMemsetAsync(status.data,0,8U,stream));
            hipLaunchKernelGGL(initialize_scores,dim3(static_cast<unsigned>((score_elements+255U)/256U)),dim3(256),0,stream,
                scores.as<float>(),query.as<float>(),score_elements,mode==9U);
            strix::HipCompactQsaProbe p;
            p.query=query.as<float>();p.kv=kv.as<std::uint16_t>();p.selected=selected.as<std::uint32_t>();p.scores=scores.as<float>();
            p.image=image.as<unsigned char>()+64U;p.weights=projection==3U?float_weights.data:weights.data;p.projected=output.as<std::uint16_t>();p.status=status.as<std::uint32_t>();
            p.quantized=quantized.data;
            p.rows=rows;p.mode=mode;p.compact=arm!=0U;p.tiled_projection=projection==0U;p.scalar_projection=projection==3U;
            p.dense_a8_projection=projection==4U;p.fast_f16_projection=projection!=2U;
            strix::launch_hip_compact_qsa_probe(p,stream);check(hipStreamSynchronize(stream));
            check(hipMemcpy(statuses[arm],status.data,8U,hipMemcpyDeviceToHost));
            if(mode==8U) {auto& target=arm?quantized_b:quantized_a;check(hipMemcpy(target.data(),quantized.data,target.size(),hipMemcpyDeviceToHost));}
            if(arm) check(hipMemcpy(small.data(),p.image,small.size()*2U,hipMemcpyDeviceToHost));
            else check(hipMemcpy(ref.data(),p.image,ref.size()*4U,hipMemcpyDeviceToHost));
            auto& target=arm?b:a;check(hipMemcpy(target.data(),output.data,target.size()*2U,hipMemcpyDeviceToHost));
            std::uint32_t guards[16];
            for(auto offset:{std::size_t{0},64U+image_elements*(arm?2U:4U)}) {
                check(hipMemcpy(guards,image.as<unsigned char>()+offset,64U,hipMemcpyDeviceToHost));
                for(auto word:guards) if(word!=0xa5a5a5a5U) throw std::runtime_error("image guard overwritten");
            }
        }
        for(std::size_t i=0;i<ref.size();++i) if(ref[i]!=(static_cast<unsigned>(small[i])<<16U))
            throw std::runtime_error("attention image bits differ");
        if(quantized_a!=quantized_b) throw std::runtime_error("tail quantizer bytes differ");
        if(a!=b || std::memcmp(statuses[0],statuses[1],sizeof(statuses[0]))!=0) throw std::runtime_error("projection or status bytes differ");
        std::printf("T%u mode%u projection%u EXACT image/projection/status/range/guards\n",rows,mode,projection);
    }
}
}
// Verify-window value/gate: the window group kernel (G = 2|3|4|6|12) against the per-head kernel, bitwise
// (image, dense A8 codes and deltas), over a 16384-token BF16 cache with scattered selections; also times both.
__global__ void initialize_window(float* query, std::uint16_t* kv, std::uint32_t* selected, float* scores,
    std::uint64_t query_words, std::uint64_t kv_words, std::uint64_t selected_words, std::uint64_t score_words) {
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto hash=[](std::uint64_t x) {x^=x>>33U;x*=0xff51afd7ed558ccdULL;x^=x>>33U;x*=0xc4ceb9fe1a85ec53ULL;return x^(x>>33U);};
    if(i<query_words) query[i]=(static_cast<float>(hash(i)%2001U)-1000.0F)/256.0F;
    if(i<kv_words) {
        const auto h=hash(i+0x9e37U);
        kv[i]=static_cast<std::uint16_t>((h%3U==0U?0x8000U:0U)|(0x3a00U+(h>>8U)%0x0c00U));
    }
    if(i<selected_words) selected[i]=static_cast<std::uint32_t>(hash(i+0x51U)%16384U);
    if(i<score_words) scores[i]=static_cast<float>(hash(i+0x77U)%100000U+1U)*0x1p-27F;
}
void window_out_group_fixture(hipStream_t stream) {
    constexpr unsigned heads=24U,dim=256U,tokens=16384U,rows=8U;
    const std::uint64_t query_words=rows*heads*dim*2ULL,kv_words=tokens*2ULL*2ULL*dim,
        selected_words=rows*static_cast<std::uint64_t>(maximum),score_words=rows*heads*static_cast<std::uint64_t>(maximum);
    Buffer query(query_words*4U),kv(kv_words*2U),selected(selected_words*4U),scores(score_words*4U),
        image(rows*heads*dim*4U),codes(rows*heads*dim),deltas(heads*dim/32U*64U*4U);
    const auto words=std::max(std::max(query_words,kv_words),std::max(selected_words,score_words));
    hipLaunchKernelGGL(initialize_window,dim3(static_cast<unsigned>((words+255U)/256U)),dim3(256),0,stream,query.as<float>(),
        kv.as<std::uint16_t>(),selected.as<std::uint32_t>(),scores.as<float>(),query_words,kv_words,selected_words,score_words);
    check(hipStreamSynchronize(stream));
    hipEvent_t start=nullptr,stop=nullptr;check(hipEventCreate(&start));check(hipEventCreate(&stop));
    const std::uint32_t counts[][8]={{2052U,2052U,2052U,2052U,2052U,2052U,2052U,2052U},{1031U,1032U,1033U,1034U,1035U,1036U,1037U,1038U},
        {37U,1U,8U,0U,15U,2051U,16U,9U}};
    for(const unsigned window:{2U,4U,8U}) for(const auto& count:counts) for(const bool a8:{true,false}) {
        std::vector<std::uint32_t> ref_image(rows*heads*dim);std::vector<std::int8_t> ref_codes(rows*heads*dim);
        std::vector<float> ref_deltas(heads*dim/32U*64U);
        for(const unsigned group:{0U,2U,3U,4U,6U,12U}) {
            strix::HipQsaWindowOutGroupProbe p;
            p.query=query.as<float>();p.kv=kv.as<std::uint16_t>();p.selected=selected.as<std::uint32_t>();p.scores=scores.as<float>();
            p.image=image.as<float>();p.codes=a8?codes.as<std::int8_t>():nullptr;p.deltas=a8?deltas.as<float>():nullptr;
            p.rows=window;p.group=group;p.maximum_selected=maximum;std::memcpy(p.count,count,sizeof(p.count));
            check(hipMemsetAsync(image.data,0xa5,rows*heads*dim*4U,stream));check(hipMemsetAsync(codes.data,0x5a,rows*heads*dim,stream));
            check(hipMemsetAsync(deltas.data,0xa5,heads*dim/32U*64U*4U,stream));
            strix::launch_hip_qsa_window_out_group_probe(p,stream);check(hipStreamSynchronize(stream));
            std::vector<std::uint32_t> got_image(ref_image.size());std::vector<std::int8_t> got_codes(ref_codes.size());
            std::vector<float> got_deltas(ref_deltas.size());
            check(hipMemcpy(got_image.data(),image.data,got_image.size()*4U,hipMemcpyDeviceToHost));
            check(hipMemcpy(got_codes.data(),codes.data,got_codes.size(),hipMemcpyDeviceToHost));
            check(hipMemcpy(got_deltas.data(),deltas.data,got_deltas.size()*4U,hipMemcpyDeviceToHost));
            if(group==0U) {ref_image=got_image;ref_codes=got_codes;ref_deltas=got_deltas;}
            else if(got_image!=ref_image || got_codes!=ref_codes || std::memcmp(got_deltas.data(),ref_deltas.data(),got_deltas.size()*4U)!=0)
                throw std::runtime_error("window out-group bytes differ");
            float ms=0.0F;constexpr int reps=50;
            check(hipEventRecord(start,stream));
            for(int i=0;i<reps;++i) strix::launch_hip_qsa_window_out_group_probe(p,stream);
            check(hipEventRecord(stop,stream));check(hipEventSynchronize(stop));check(hipEventElapsedTime(&ms,start,stop));
            std::printf("window R%u count %u a8 %u G%u %s %.1f us\n",window,count[0],a8?1U:0U,group,group?"EXACT":"ref",ms*1000.0F/reps);
        }
    }
    check(hipEventDestroy(start));check(hipEventDestroy(stop));
}
int main() {
    int count=0;if(hipGetDeviceCount(&count)!=hipSuccess || !count) return 77;
    hipStream_t stream=nullptr;
    try {
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        special_fixture(stream);window_out_group_fixture(stream);run(8192U,stream);run(40U,stream);
        check(hipStreamDestroy(stream));return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"compact image fixture: %s\n",e.what());
        if(stream) (void)hipStreamDestroy(stream);return 1;
    }
}
