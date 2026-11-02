// GPU exactness check for the few-row trellis kernels (row_mma.cuh): random packed tiles of every
// rate, random FP16 operands; the row strips must equal the pinned CUTLASS projection GEMM and the
// C1 grouped MoE (routed and single expert) bit for bit. Exit 0 = every case equal.
#include "../row_mma.cuh"
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace strix::glm::rtxpf {
namespace {
#include "../projection.inl"
struct Arena {U16 *operand,*decoded;float* raw;Problem* problem;};
}
}
using namespace strix::glm::rtxpf;

static void ck(cudaError_t e,const char* what){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s: %s\n",what,cudaGetErrorString(e));std::exit(2);}}
template<class T> T* upload(const std::vector<T>& h){T* d=nullptr;ck(cudaMalloc(&d,h.size()*sizeof(T)),"malloc");ck(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice),"h2d");return d;}
static std::mt19937 rng(0x5eed1234u);
static std::vector<U16> random_halves(std::size_t n,float lo,float hi){std::uniform_real_distribution<float> u(lo,hi);std::vector<U16> v(n);for(auto& x:v)x=fh(u(rng));return v;}
static std::vector<U16> random_bits(std::size_t n){std::vector<U16> v(n);for(auto& x:v)x=U16(rng());return v;}
static Matrix random_matrix(unsigned in,unsigned out,unsigned bits){
    Matrix m{};m.input=in;m.output=out;m.bits=bits;
    m.packed=upload(random_bits(std::size_t(in/16)*(out/16)*16*bits));
    // suh/svh: +-1 sign vectors with a random scale, as the H128 transforms expect
    m.suh=upload(random_halves(in,-1.5f,1.5f));m.svh=upload(random_halves(out,-1.5f,1.5f));
    return m;
}
static unsigned failures=0,cases=0;
static void verdict(const char* what,const std::vector<float>& a,const std::vector<float>& b){
    ++cases;std::size_t diff=0;for(std::size_t j=0;j<a.size();++j)diff+=std::memcmp(&a[j],&b[j],4)!=0;
    if(diff){++failures;std::printf("DIFFERENT %s: %zu of %zu\n",what,diff,a.size());}else std::printf("EQUAL %s (%zu values)\n",what,a.size());
}

static void projection_case(unsigned in,unsigned out,unsigned bits,unsigned rows){
    Matrix w=random_matrix(in,out,bits);
    auto op=upload(random_halves(std::size_t(rows)*in,-2.f,2.f));
    U16* decoded;float *raw1,*raw2;Problem* prob;
    ck(cudaMalloc(&decoded,std::size_t(in)*out*2),"m");ck(cudaMalloc(&raw1,std::size_t(rows)*out*4),"m");ck(cudaMalloc(&raw2,std::size_t(rows)*out*4),"m");ck(cudaMalloc(&prob,sizeof(Problem)),"m");
    layer_decode<<<in/16*(out/16),256>>>(w,decoded);
    ck(gemm<Row>(op,decoded,raw1,rows,in,out,prob,nullptr),"gemm");
    std::vector<float> a(std::size_t(rows)*out),b(a.size());
    ck(cudaMemcpy(a.data(),raw1,a.size()*4,cudaMemcpyDeviceToHost),"d2h");
    for(bool r2:{false,true}){
        launch_row_project(w,op,raw2,rows,nullptr,r2);ck(cudaDeviceSynchronize(),"sync");
        ck(cudaMemcpy(b.data(),raw2,b.size()*4,cudaMemcpyDeviceToHost),"d2h");
        char name[128];std::snprintf(name,sizeof name,"projection%s K%u %ux%u rows %u",r2?" r2":"",bits,in,out,rows);verdict(name,a,b);}
    cudaFree(decoded);cudaFree(raw1);cudaFree(raw2);cudaFree(prob);cudaFree(op);
    cudaFree((void*)w.packed);cudaFree((void*)w.suh);cudaFree((void*)w.svh);
}

// Dense FP16 [n][k] weights: row_dense == the CUTLASS column-major projection GEMM (KDA ordinary projections).
static void dense_case(unsigned in,unsigned out,unsigned rows){
    auto w=upload(random_halves(std::size_t(in)*out,-1.f,1.f));auto op=upload(random_halves(std::size_t(rows)*in,-2.f,2.f));
    float *y1,*y2;Problem* prob;ck(cudaMalloc(&y1,std::size_t(rows)*out*4),"m");ck(cudaMalloc(&y2,std::size_t(rows)*out*4),"m");ck(cudaMalloc(&prob,sizeof(Problem)),"m");
    ck(gemm<Col>(op,w,y1,rows,in,out,prob,nullptr),"gemm col");launch_row_dense(w,op,y2,rows,in,out,nullptr);ck(cudaDeviceSynchronize(),"sync");
    std::vector<float> p(std::size_t(rows)*out),q(p.size());ck(cudaMemcpy(p.data(),y1,p.size()*4,cudaMemcpyDeviceToHost),"d2h");ck(cudaMemcpy(q.data(),y2,q.size()*4,cudaMemcpyDeviceToHost),"d2h");
    char name[128];std::snprintf(name,sizeof name,"dense fp16 %ux%u rows %u",in,out,rows);verdict(name,p,q);
    cudaFree(w);cudaFree(op);cudaFree(y1);cudaFree(y2);cudaFree(prob);
}
// FP32 index projections: projection_fp32_row (rows <= 4) == projection_fp32_tiled, trellis (FP16 operand, [K][N]
// weights) and ordinary (FP32 x, [N][K] weights) operands.
static void fp32_case(unsigned in,unsigned out,unsigned rows,bool ord){
    auto w=upload(random_halves(std::size_t(in)*out,-1.f,1.f));
    std::uniform_real_distribution<float> u(-2.f,2.f);std::vector<float> xs(std::size_t(rows)*in);for(auto& v:xs)v=u(rng);
    auto x=upload(xs);auto a=upload(random_halves(std::size_t(rows)*in,-2.f,2.f));
    float *y1,*y2;ck(cudaMalloc(&y1,std::size_t(rows)*out*4),"m");ck(cudaMalloc(&y2,std::size_t(rows)*out*4),"m");
    projection_fp32_tiled<<<dim3((out+31)/32,(rows+7)/8),128>>>(ord?x:nullptr,ord?nullptr:a,w,y1,rows,in,out,ord);
    projection_fp32_row<<<dim3((out+255)/256,rows),256>>>(ord?x:nullptr,ord?nullptr:a,w,y2,in,out,ord);ck(cudaDeviceSynchronize(),"sync");
    std::vector<float> p(std::size_t(rows)*out),q(p.size());ck(cudaMemcpy(p.data(),y1,p.size()*4,cudaMemcpyDeviceToHost),"d2h");ck(cudaMemcpy(q.data(),y2,q.size()*4,cudaMemcpyDeviceToHost),"d2h");
    char name[128];std::snprintf(name,sizeof name,"fp32 %s %ux%u rows %u",ord?"ordinary":"trellis",in,out,rows);verdict(name,p,q);
    cudaFree(w);cudaFree(x);cudaFree(a);cudaFree(y1);cudaFree(y2);
}

static void moe_case(unsigned experts,unsigned hidden,unsigned inter,unsigned rows,CutlassExecution& exec,cudaStream_t s){
    std::vector<Expert> ex(experts);
    for(unsigned e=0;e<experts;++e){unsigned b=3+(e+rows)%4;ex[e]={random_matrix(hidden,inter,b),random_matrix(hidden,inter,b==6?3:b+1),random_matrix(inter,hidden,b)};}
    auto dex=upload(ex);
    std::uniform_real_distribution<float> u(-1.f,1.f);std::vector<float> x(std::size_t(rows)*hidden);for(auto& v:x)v=u(rng);
    std::vector<U32> ids;std::vector<float> wts;
    for(unsigned r=0;r<rows&&experts>1;++r){std::vector<U32> all(experts);for(unsigned j=0;j<experts;++j)all[j]=j;std::shuffle(all.begin(),all.end(),rng);
        for(unsigned j=0;j<8;++j){ids.push_back(all[j]);wts.push_back(0.05f+0.3f*std::abs(u(rng)));}}
    auto dx=upload(x);U32* dids=experts>1?upload(ids):nullptr;float* dw=experts>1?upload(wts):nullptr;
    std::size_t bytes=workspace_bytes_cutlass(rows,experts,hidden,inter);void* ws;ck(cudaMalloc(&ws,bytes),"ws");
    float *y1,*y2;ck(cudaMalloc(&y1,x.size()*4),"y");ck(cudaMalloc(&y2,x.size()*4),"y");
    Call v{rows,experts,hidden,inter,dx,dids,dw,dex,y1};
    ck(enqueue_cutlass(v,ws,bytes,s,exec),"grouped");ck(cudaStreamSynchronize(s),"sync");
    std::vector<float> a(x.size()),b(x.size());ck(cudaMemcpy(a.data(),y1,a.size()*4,cudaMemcpyDeviceToHost),"d2h");
    for(bool r2:{false,true}){
        v.y=y2;v.row_gemv=true;v.row_r2=r2;ck(enqueue_cutlass(v,ws,bytes,s,exec),"row");ck(cudaStreamSynchronize(s),"sync");
        ck(cudaMemcpy(b.data(),y2,b.size()*4,cudaMemcpyDeviceToHost),"d2h");
        char name[128];std::snprintf(name,sizeof name,"moe%s experts %u %ux%u rows %u",r2?" r2":"",experts,hidden,inter,rows);verdict(name,a,b);}
    for(auto& e:ex)for(const Matrix* m:{&e.gate,&e.up,&e.down}){cudaFree((void*)m->packed);cudaFree((void*)m->suh);cudaFree((void*)m->svh);}
    cudaFree(dex);cudaFree(dx);if(dids)cudaFree(dids);if(dw)cudaFree(dw);cudaFree(ws);cudaFree(y1);cudaFree(y2);
}

int main(){
    if(Projection<Row>::sufficient()<=0||Projection<Col>::sufficient()<=0){std::fprintf(stderr,"projection GEMM setup failed\n");return 2;} // smem attribute, as prepare_kda
    // decode shapes: KDA qkv/out, DSA kva/qb/out, MTP eh; replay rows 1..32
    for(unsigned bits=3;bits<=6;++bits)
        for(unsigned rows:{1u,2u,7u,16u,17u,31u,32u}){projection_case(4096,512,bits,rows);projection_case(1536,4096,bits,rows);}
    projection_case(4096,24576,4,32);projection_case(8192,4096,5,1);projection_case(16384,4096,3,1);
    for(unsigned rows:{1u,16u,17u,32u}){dense_case(4096,128,rows);dense_case(128,8192,rows);dense_case(4096,64,rows);}
    for(unsigned rows=1;rows<=4;++rows){fp32_case(4096,1536,rows,false);fp32_case(1536,4096,rows,false);fp32_case(4096,128,rows,true);fp32_case(4096,32,rows,true);}
    cudaStream_t s,d;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");ck(cudaStreamCreateWithFlags(&d,cudaStreamNonBlocking),"stream");
    CutlassExecution exec{};exec.decode_stream=d;
    auto ev=[](){cudaEvent_t e;ck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming),"event");return e;};
    exec.dispatch=ev();for(unsigned i=0;i<18;++i){exec.ready[i]=ev();exec.retired[i]=ev();}
    ck(prepare_cutlass(exec),"prepare");
    for(unsigned rows=1;rows<=4;++rows){moe_case(16,512,256,rows,exec,s);moe_case(1,512,256,rows,exec,s);}
    moe_case(24,4096,2048,1,exec,s);moe_case(1,4096,12288,1,exec,s);moe_case(1,4096,2048,1,exec,s);
    std::printf("%s: %u of %u cases equal\n",failures?"FAIL":"PASS",cases-failures,cases);
    return failures?1:0;
}
