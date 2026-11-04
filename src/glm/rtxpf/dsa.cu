#include "dsa.cuh"
#include "paired_index_model.hpp"
#include <cuda_bf16.h>
#include <math_constants.h>
#include "row_mma.cuh"
#include <mma.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <algorithm>
#include <vector>
#include <cstring>
namespace strix::glm::rtxpf {
namespace {
#include "projection.inl"
// CUTLASS FastF32 decomposes both operands into TF32 high/residual terms,
// accumulating high*high, high*low and low*high on tensor cores.
template<class Layout>
using AccurateKernel=typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float,Row,cutlass::ComplexTransform::kNone,4,
    float,Layout,cutlass::ComplexTransform::kNone,4,
    float,Row,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64,64,16>,cutlass::gemm::GemmShape<32,32,16>,
    cutlass::gemm::GemmShape<16,8,8>,
    cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3,cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    cutlass::arch::OpMultiplyAddFastF32>::GemmKernel;
template<class L>using Accurate=cutlass::gemm::device::GemmGrouped<AccurateKernel<L>>;
struct AccurateProblem {cutlass::gemm::GemmCoord size;float *a,*b,*d;int64_t lda,ldb,ldd;};
__global__ void accurate_problem(AccurateProblem* p,const float* a,const float* b,float* d,unsigned r,unsigned in,unsigned out,bool row) {
    if(threadIdx.x==0){p->size={int(r),int(out),int(in)};p->a=(float*)a;p->b=(float*)b;p->d=d;p->lda=in;p->ldb=row?out:in;p->ldd=out;}
}
__global__ void widen(const U16* a,float* b,std::size_t n){auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j<n)b[j]=hf(a[j]);}
template<class L>
cudaError_t accurate_gemm(const float* x,const float* w,float* y,unsigned r,unsigned in,unsigned out,AccurateProblem* p,cudaStream_t s) {
    accurate_problem<<<1,1,0,s>>>(p,x,w,y,r,in,out,std::is_same_v<L,Row>);
    using G=Accurate<L>;using K=typename G::GemmKernel;
    typename G::Arguments args(&p->size,1,256,{1.f,0.f},&p->a,&p->b,&p->d,&p->d,&p->lda,&p->ldb,&p->ldd,&p->ldd);
    typename K::Params params(args);
    cutlass::Kernel<K><<<256,K::kThreadCount,sizeof(typename K::SharedStorage),s>>>(params);
    return cudaGetLastError();
}
// Independent IEEE paired-half model: binary16 products are exact in binary64.
// One RN32 after the three-term sum; no claim about AMD DOT2 truncation.
__global__ void paired_projection(const U16* a,const U16* w,float* y,unsigned rows,unsigned in,unsigned out) {
    auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j>=std::size_t(rows)*out)return;
    unsigned r=j/out,n=j%out,tiles=in/16,splits=(out<=1024?32:(out<=4096&&in>=4096?16:4));
    float total=0;
    for(unsigned sp=0;sp<splits;++sp){float v[4]={0,0,0,0};
        unsigned begin=tiles*sp/splits,end=tiles*(sp+1)/splits;
        for(unsigned kt=begin;kt<end;++kt)for(unsigned lane=0;lane<4;++lane){
            unsigned k=kt*16+lane*2;
            v[lane]=paired_half_model(hf(a[std::size_t(r)*in+k]),hf(w[std::size_t(k)*out+n]),
                hf(a[std::size_t(r)*in+k+1]),hf(w[std::size_t(k+1)*out+n]),v[lane]);
            k+=8;
            v[lane]=paired_half_model(hf(a[std::size_t(r)*in+k]),hf(w[std::size_t(k)*out+n]),
                hf(a[std::size_t(r)*in+k+1]),hf(w[std::size_t(k+1)*out+n]),v[lane]);
        }
        total=__fadd_rn(total,__fadd_rn(__fadd_rn(v[0],v[2]),__fadd_rn(v[1],v[3])));
    }
    y[j]=total;
}
using BF=__nv_bfloat16;
using namespace nvcuda;
struct Arena {
    AccurateProblem* accurate_problem; float *accurate_operand,*accurate_weights[5];
    Problem* problem; U16 *decoded,*operand;float* raw;
    U16* weights[5];
    float *qa,*qr,*lat,*key,*gate,*wh,*iq,*q,*qt,*ob,*o,*scores,*ref_scores;
    U16* qt_bf; float *partial,*stats;
    U32 *keys,*tokens,*counts;
    std::size_t size;
    Arena(void* p,unsigned cap) {
        std::size_t off=0;auto take=[&](std::size_t n){auto old=off;off+=(n+255)&~std::size_t(255);return p?static_cast<char*>(p)+old:nullptr;};
        problem=(Problem*)take(sizeof(Problem));
        accurate_problem=(AccurateProblem*)take(sizeof(AccurateProblem));
        accurate_operand=(float*)take(std::size_t(dsa_tile)*4096*4);
        const unsigned wide_sizes[5]={4096*1536,1536*4096,128*4096,128*4096,32*4096};
        for(unsigned j=0;j<5;++j)accurate_weights[j]=(float*)take(std::size_t(wide_sizes[j])*4);
        const unsigned ins[5]={4096,1536,4096,16384,1536},outs[5]={1536,16384,512,4096,4096};
        for(unsigned j=0;j<5;++j)weights[j]=(U16*)take(std::size_t(ins[j])*outs[j]*2);
        decoded=nullptr;operand=(U16*)take(std::size_t(dsa_tile)*16384*2);raw=(float*)take(std::size_t(dsa_tile)*16384*4);
        auto floats=[&](std::size_t n){return (float*)take(n*4);};
        qa=floats(dsa_tile*1536);qr=floats(dsa_tile*1536);lat=floats(dsa_tile*512);
        key=floats(dsa_tile*128);gate=floats(dsa_tile*128);wh=floats(dsa_tile*32);iq=floats(dsa_tile*4096);
        q=floats(dsa_tile*16384);qt=floats(dsa_tile*64*512);ob=floats(dsa_tile*64*512);o=floats(dsa_tile*16384);
        scores=floats(std::size_t(dsa_tile)*(cap/4));keys=(U32*)take(std::size_t(dsa_tile)*(cap/4)*4);
        qt_bf=(U16*)take(std::size_t(dsa_tile)*64*512*2);
        partial=floats(std::size_t(dsa_tile)*dsa_splits*64*512);
        stats=floats(std::size_t(dsa_tile)*dsa_splits*64*2);
        ref_scores=floats(std::size_t(dsa_tile)*64*dsa_max_tokens);
        tokens=(U32*)take(std::size_t(dsa_tile)*dsa_max_tokens*4);counts=(U32*)take(dsa_tile*4);size=off;
    }
};
__device__ U16 tobf(float f){return __bfloat16_as_ushort(__float2bfloat16_rn(f));}
__device__ U32 ordered(float f){U32 b=__float_as_uint(f);if((b&0x7fffffff)==0)b=0;return b&0x80000000?~b:b|0x80000000;}
__device__ void finite(float f,unsigned* flag){if(!isfinite(f))atomicOr(flag,1u);}
__global__ void rms(const float* x,const U16* w,float* y,unsigned width,bool mimic=false) {
    __shared__ float sums[256];unsigned r=blockIdx.x,l=threadIdx.x;float sum=0;
    for(unsigned j=l;j<width;j+=256)sum=__fmaf_rn(x[std::size_t(r)*width+j],x[std::size_t(r)*width+j],sum);
    sums[l]=sum;__syncthreads();for(unsigned k=128;k;k/=2){if(l<k)sums[l]+=sums[l+k];__syncthreads();}
    float arg=sums[0]/width+1.e-5f;
    float scale=1.f/(mimic?float(__dsqrt_rn(double(arg))):sqrtf(arg));
    for(unsigned j=l;j<width;j+=256)y[std::size_t(r)*width+j]=x[std::size_t(r)*width+j]*scale*bf(w[j]);
}
__global__ void store(DsaState st,unsigned first,const float* lat,const float* key,const float* gate,const U16* nw,const U16* nb,unsigned* flag,bool mimic,bool skip_latent) {
    __shared__ float sums[128];unsigned r=blockIdx.x,i=threadIdx.x;auto at=std::size_t(r)*128+i;float k=key[at];sums[i]=k;__syncthreads();
    for(unsigned off=64;off;off/=2){if(i<off)sums[i]+=sums[i+off];__syncthreads();}float mean=sums[0]/128;__syncthreads();
    float d=k-mean;sums[i]=d*d;__syncthreads();for(unsigned off=64;off;off/=2){if(i<off)sums[i]+=sums[i+off];__syncthreads();}
    float arg=sums[0]/128+1.e-6f;
    float norm=mimic?__fmul_rn(d,__fdiv_rn(1.f,float(__dsqrt_rn(double(arg))))):d/sqrtf(arg);
    U16 kb=tobf(norm*hf(nw[i])+hf(nb[i])),gb=tobf(gate[at]);
    finite(bf(kb),flag);finite(bf(gb),flag);auto dst=std::size_t(first+r)*128+i;st.key[dst]=kb;st.gate[dst]=gb;
    for(unsigned j=i;!skip_latent&&j<512;j+=128){U16 lb=tobf(lat[std::size_t(r)*512+j]);finite(bf(lb),flag);st.latent[std::size_t(first+r)*512+j]=lb;}
}
__global__ void pool(DsaState st,unsigned base,const float* ape,unsigned* flag) {
    unsigned p=base+blockIdx.x,i=threadIdx.x;float g[4],k[4],e[4];
    for(unsigned j=0;j<4;++j){auto at=std::size_t(p*4+j)*128+i;g[j]=bf(st.gate[at])+ape[j*128+i];k[j]=bf(st.key[at]);}
    float m=g[0];for(unsigned j=1;j<4;++j)m=fmaxf(m,g[j]);for(unsigned j=0;j<4;++j)e[j]=declared_exp(g[j]-m);
    float den=((e[0]+e[1])+e[2])+e[3];for(unsigned j=0;j<4;++j)e[j]/=den;
    float v=((e[0]*k[0]+e[1]*k[1])+e[2]*k[2])+e[3]*k[3];finite(v,flag);st.pooled[std::size_t(p)*128+i]=v;
}
// Batched head-wise q*K / ob*V, using original F16 kv_b rows [head,2,256,512].
// Query contraction is over 256; value contraction over 512. One warp/tile.
template<bool Value,bool Ref>
__global__ void absorb(const float* x,const U16* w,float* y,unsigned rows) {
    unsigned h=blockIdx.z,r0=blockIdx.x*16,n0=blockIdx.y*16,l=threadIdx.x;
    constexpr unsigned in=Value?512:256,out=Value?256:512;
    if constexpr(Ref){for(unsigned j=l;j<256;j+=32){unsigned r=r0+j/16,n=n0+j%16;if(r>=rows)continue;float v=0;
        for(unsigned k=0;k<in;++k){auto wi=(std::size_t(h)*512+(Value?256+n:k))*512+(Value?k:n);v=__fmaf_rn(x[(std::size_t(r)*64+h)*in+k],hf(w[wi]),v);}y[(std::size_t(r)*64+h)*out+n]=v;} }
    else {
        __shared__ __align__(32) __half a[256],b[256];__shared__ __align__(32) float c[256];
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf0;
        wmma::fragment<wmma::accumulator,16,16,16,float> cf;wmma::fill_fragment(cf,0.f);
        for(unsigned k0=0;k0<in;k0+=16){for(unsigned j=l;j<256;j+=32){unsigned r=r0+j/16,k=k0+j%16;
            a[j]=__float2half_rn(r<rows?x[(std::size_t(r)*64+h)*in+k]:0);unsigned kk=k0+j/16,n=n0+j%16;
            b[j]=__ushort_as_half(w[(std::size_t(h)*512+(Value?256+n:kk))*512+(Value?kk:n)]);}
            __syncwarp();wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bf0,b,16);wmma::mma_sync(cf,af,bf0,cf);__syncwarp();}
        wmma::store_matrix_sync(c,cf,16,wmma::mem_row_major);__syncwarp();for(unsigned j=l;j<256;j+=32){unsigned r=r0+j/16;if(r<rows)y[(std::size_t(r)*64+h)*out+n0+j%16]=c[j];}
    }
}
// Bounded [64,capacity/4] folded scores only, never [C,C].
// Default: FP32 operands and ascending-k FMA, then RN32 scale/ReLU and
// ascending-h FMA fold, matching the node formula. BF16 MMA is opt-in.
// Each warp scores 16 pools, with fused scale/ReLU/ascending-head fold.
// BF16x3 omits only residual*residual; the FP32 reference stays independent.
template<int Kind>
__device__ void index_dots(const float* pooled,const float* qi,unsigned p0,unsigned P,float* dots,BF* operands) {
    unsigned l=threadIdx.x%32;
    if constexpr(Kind==1){for(unsigned j=l;j<512;j+=32){unsigned h=j/16,p=p0+j%16;float v=0;
        if(p<P)for(unsigned k=0;k<128;++k)v=__fmaf_rn(qi[h*128+k],pooled[std::size_t(p)*128+k],v);dots[j]=v;}}
    else {
        BF *a=operands,*b=a+256,*al=b+256,*bl=al+256;
        wmma::fragment<wmma::matrix_a,16,16,16,BF,wmma::row_major> af,alf;
        wmma::fragment<wmma::matrix_b,16,16,16,BF,wmma::row_major> bf0,blf;
        for(unsigned h0=0;h0<32;h0+=16){wmma::fragment<wmma::accumulator,16,16,16,float> cf;wmma::fill_fragment(cf,0.f);
            for(unsigned k0=0;k0<128;k0+=16){for(unsigned j=l;j<256;j+=32){
                float av=qi[(h0+j/16)*128+k0+j%16];unsigned p=p0+j%16;
                float bv=p<P?pooled[std::size_t(p)*128+k0+j/16]:0;
                a[j]=__float2bfloat16_rn(av);b[j]=__float2bfloat16_rn(bv);
                if constexpr(Kind==2){al[j]=__float2bfloat16_rn(__fsub_rn(av,__bfloat162float(a[j])));bl[j]=__float2bfloat16_rn(__fsub_rn(bv,__bfloat162float(b[j])));}
            }
            __syncwarp();wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bf0,b,16);
            if constexpr(Kind==2){
                wmma::load_matrix_sync(alf,al,16);wmma::load_matrix_sync(blf,bl,16);
                wmma::mma_sync(cf,alf,bf0,cf);wmma::mma_sync(cf,af,blf,cf);
            }
            wmma::mma_sync(cf,af,bf0,cf);__syncwarp();}
            wmma::store_matrix_sync(dots+h0*16,cf,16,wmma::mem_row_major);__syncwarp();}
    }
    __syncwarp();
}
__device__ float fold_index(const float* dots,const float* wh,unsigned l){
    float v=0;for(unsigned h=0;h<32;++h)v=__fmaf_rn(wh[h]*0.17677669529663687f,fmaxf(0.f,dots[h*16+l]*0.088388347648318f),v);return v;
}
template<int Kind>
__global__ void index_scores(const float* pooled,const float* qi,const float* wh,float* scores,U32* keys,unsigned cap,unsigned first,unsigned* flag) {
    unsigned r=blockIdx.y,p0=blockIdx.x*16,P=(first+r+1)/4,l=threadIdx.x;
    if(p0>=P||P<=dsa_selection_budget)return; // Entirely future tiles execute no loads or MMA.
    __shared__ __align__(32) float dots[512];
    __shared__ __align__(32) BF operands[1024];
    index_dots<Kind>(pooled,qi+std::size_t(r)*4096,p0,P,dots,operands);
    if(l<16&&p0+l<P){float v=fold_index(dots,wh+std::size_t(r)*32,l);finite(v,flag);
        scores[std::size_t(r)*cap+p0+l]=v;keys[std::size_t(r)*cap+p0+l]=ordered(v);}
}
// Two queries x 16 pools, FP32 only. All eight accumulators per thread retain
// ascending K=0..127. The final head fold is the unchanged ascending 32-chain.
__global__ __launch_bounds__(128) void index_fp32_tiled(const float* pooled,const float* qi,const float* wh,float* scores,U32* keys,
                                                       unsigned cap,unsigned first,unsigned rows,unsigned* flag) {
    unsigned r0=blockIdx.y*2,p0=blockIdx.x*16,tid=threadIdx.x,h0=tid/16,p=tid%16;
    unsigned Pmax=(first+min(r0+1,rows-1)+1)/4;if(p0>=Pmax||Pmax<=dsa_selection_budget)return;
    __shared__ float query[2][32][32],pool_tile[16][33],dots[2][32][16];
    float acc[2][4]{};
    for(unsigned k0=0;k0<128;k0+=32){
        for(unsigned j=tid;j<2*32*32;j+=128){unsigned r=r0+j/1024,h=j/32%32,k=k0+j%32;
            query[j/1024][h][j%32]=r<rows?qi[std::size_t(r)*4096+h*128+k]:0;}
        for(unsigned j=tid;j<16*32;j+=128){unsigned pp=p0+j/32;pool_tile[j/32][j%32]=pp<Pmax?pooled[std::size_t(pp)*128+k0+j%32]:0;}
        __syncthreads();
        #pragma unroll
        for(unsigned k=0;k<32;++k){
            #pragma unroll
            for(unsigned r=0;r<2;++r){
                #pragma unroll
                for(unsigned hh=0;hh<4;++hh)acc[r][hh]=__fmaf_rn(query[r][h0+hh*8][k],pool_tile[p][k],acc[r][hh]);
            }
        }
        __syncthreads();
    }
    for(unsigned r=0;r<2;++r)for(unsigned hh=0;hh<4;++hh)dots[r][h0+hh*8][p]=acc[r][hh];
    __syncthreads();
    if(tid<32){unsigned rr=tid/16,r=r0+rr,pp=p0+tid%16,P=(first+r+1)/4;
        if(r<rows&&P>dsa_selection_budget&&pp<P){float v=fold_index(&dots[rr][0][0],wh+std::size_t(r)*32,tid%16);finite(v,flag);
            scores[std::size_t(r)*cap+pp]=v;keys[std::size_t(r)*cap+pp]=ordered(v);}}
}
// Two queries x 32 pools. Each staged operand is reused across head/pool
// blocks; no matrix of per-head scores leaves this CTA. Future tiles return
// before staging. BF16 residual expansion keeps the v2 three-product identity.
struct __align__(32) TiledIndexShared {
    BF a[64*128],al[64*128],b[128*32],bl[128*32];
    float dots[64*32];
};
__global__ __launch_bounds__(128) void tiled_index(const float* pooled,const float* qi,const float* wh,float* scores,U32* keys,
                                                unsigned cap,unsigned first,unsigned rows,unsigned* flag) {
    unsigned r0=blockIdx.y*2,p0=blockIdx.x*32,l=threadIdx.x,warp=l/32;
    unsigned last=min(r0+1,rows-1),Pmax=(first+last+1)/4;
    if(p0>=Pmax||Pmax<=dsa_selection_budget)return;
    extern __shared__ __align__(32) unsigned char tiled_storage[];
    auto& sh=*reinterpret_cast<TiledIndexShared*>(tiled_storage);
    for(unsigned j=l;j<64*128;j+=128){unsigned r=r0+j/(32*128);float v=r<rows?qi[std::size_t(r)*4096+j%4096]:0;
        sh.a[j]=__float2bfloat16_rn(v);sh.al[j]=__float2bfloat16_rn(__fsub_rn(v,__bfloat162float(sh.a[j])));}
    for(unsigned j=l;j<128*32;j+=128){unsigned p=p0+j%32;float v=p<Pmax?pooled[std::size_t(p)*128+j/32]:0;
        sh.b[j]=__float2bfloat16_rn(v);sh.bl[j]=__float2bfloat16_rn(__fsub_rn(v,__bfloat162float(sh.b[j])));}
    __syncthreads();
    for(unsigned col=0;col<32;col+=16){
        wmma::fragment<wmma::accumulator,16,16,16,float> c;wmma::fill_fragment(c,0.f);
        for(unsigned k=0;k<128;k+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,BF,wmma::row_major> a,al;
            wmma::fragment<wmma::matrix_b,16,16,16,BF,wmma::row_major> b,bl;
            wmma::load_matrix_sync(a,sh.a+warp*16*128+k,128);wmma::load_matrix_sync(al,sh.al+warp*16*128+k,128);
            wmma::load_matrix_sync(b,sh.b+k*32+col,32);wmma::load_matrix_sync(bl,sh.bl+k*32+col,32);
            wmma::mma_sync(c,al,b,c);wmma::mma_sync(c,a,bl,c);wmma::mma_sync(c,a,b,c);
        }
        wmma::store_matrix_sync(sh.dots+warp*16*32+col,c,32,wmma::mem_row_major);
    }
    __syncthreads();
    if(l<64){unsigned r=r0+l/32,p=p0+l%32,P=(first+r+1)/4;
        if(r<rows&&P>dsa_selection_budget&&p<P){float v=0;for(unsigned h=0;h<32;++h)
            v=__fmaf_rn(wh[std::size_t(r)*32+h]*0.17677669529663687f,fmaxf(0.f,sh.dots[(l/32*32+h)*32+l%32]*0.088388347648318f),v);
            finite(v,flag);scores[std::size_t(r)*cap+p]=v;keys[std::size_t(r)*cap+p]=ordered(v);}
    }
}
// Streaming top-512: at most 512 survivors + 512 new pools per row.
// Consumes bounded folded keys; no score arithmetic in the selector. Equal-score
// survivors are visited in ascending pool ID (survivors precede newer pools).
// Stable compaction retains canonical chronological token order.
struct __align__(32) IndexShared {
    U32 keys[1024],ids[1024],saved_keys[512],saved_ids[512];
    unsigned hist[256],warp_counts[8],prefix,mask,remaining,carry;
};
__global__ __launch_bounds__(256) void streaming_index(const U32* staged,unsigned stride,unsigned first,U32* tokens,U32* counts) {
    __shared__ IndexShared sh;
    unsigned r=blockIdx.x,t=first+r,P=(t+1)/4,l=threadIdx.x,lane=l%32,warp=l/32;
    U32* tok=tokens+std::size_t(r)*dsa_max_tokens;
    if(P<=dsa_selection_budget){for(unsigned j=l;j<=t;j+=256)tok[j]=j;if(l==0)counts[r]=t+1;return;}
    unsigned kept=0;
    for(unsigned base=0;base<P;base+=512){unsigned added=min(512u,P-base),n=kept+added;
        for(unsigned j=l;j<added;j+=256){sh.keys[kept+j]=staged[std::size_t(r)*stride+base+j];sh.ids[kept+j]=base+j;}
        __syncthreads();
        if(n>512){
            if(l==0){sh.prefix=0;sh.mask=0;sh.remaining=512;}__syncthreads();
            for(int shift=24;shift>=0;shift-=8){sh.hist[l]=0;__syncthreads();
                for(unsigned j=l;j<n;j+=256)if((sh.keys[j]&sh.mask)==sh.prefix)atomicAdd(sh.hist+((sh.keys[j]>>shift)&255),1u);
                __syncthreads();if(l==0){unsigned rem=sh.remaining;for(int d=255;d>=0;--d){if(sh.hist[d]>=rem){sh.prefix|=unsigned(d)<<shift;break;}rem-=sh.hist[d];}sh.remaining=rem;sh.mask|=255u<<shift;}__syncthreads();
            }
            // Stable compaction preserves ascending IDs inductively: survivors
            // are ordered, and every newly appended ID is larger than them.
            if(l==0)sh.carry=0;__syncthreads();unsigned equal_before=0;
            for(unsigned b=0;b<n;b+=256){unsigned i=b+l,key=i<n?sh.keys[i]:0;bool eq=i<n&&key==sh.prefix;
                unsigned eqbits=__ballot_sync(0xffffffff,eq);if(lane==0)sh.warp_counts[warp]=__popc(eqbits);__syncthreads();
                unsigned eqrank=equal_before;for(unsigned w=0;w<warp;++w)eqrank+=sh.warp_counts[w];eqrank+=__popc(eqbits&((1u<<lane)-1));
                unsigned all_eq=0;for(unsigned w=0;w<8;++w)all_eq+=sh.warp_counts[w];
                bool take=i<n&&(key>sh.prefix||(eq&&eqrank<sh.remaining));__syncthreads();
                unsigned bits=__ballot_sync(0xffffffff,take);if(lane==0)sh.warp_counts[warp]=__popc(bits);__syncthreads();
                unsigned at=sh.carry;for(unsigned w=0;w<warp;++w)at+=sh.warp_counts[w];at+=__popc(bits&((1u<<lane)-1));
                if(take){sh.saved_keys[at]=key;sh.saved_ids[at]=sh.ids[i];}
                __syncthreads();if(l==0)for(unsigned w=0;w<8;++w)sh.carry+=sh.warp_counts[w];__syncthreads();equal_before+=all_eq;
            }
            for(unsigned i=l;i<512;i+=256){sh.keys[i]=sh.saved_keys[i];sh.ids[i]=sh.saved_ids[i];}__syncthreads();
        }
        kept=min(n,512u);
    }
    // Survivors are sorted already after each compaction.
    for(unsigned i=l;i<512;i+=256)for(unsigned j=0;j<4;++j)tok[i*4+j]=sh.ids[i]*4+j;
    if(l==0){unsigned n=2048;for(unsigned j=P*4;j<=t;++j)tok[n++]=j;counts[r]=n;}
}
// Adapted from project node prefill radix selection; same ordered-key and lower-ID tie contract.
__global__ __launch_bounds__(1024) void select_rows(const U32* keys_all, unsigned cap, unsigned t0,
                                                               U32* tokens_all, U32* n_all) {
    __shared__ unsigned hist[256];
    __shared__ unsigned sh_prefix, sh_mask, sh_remaining, sh_carry_eq, sh_carry_out;
    __shared__ unsigned scan_eq[1024], scan_sel[1024];
    const unsigned l = threadIdx.x, row = blockIdx.x, t = t0 + row;
    const unsigned P = (t + 1) / 4;
    U32* tokens = tokens_all + static_cast<std::size_t>(row) * dsa_max_tokens;
    if (P <= dsa_selection_budget) {
        for (unsigned pos = l; pos <= t; pos += 1024) tokens[pos] = pos;
        if (l == 0) n_all[row] = t + 1;
        return;
    }
    const U32* keys = keys_all + static_cast<std::size_t>(row) * cap;
    const unsigned k = dsa_selection_budget;
    if (l == 0) {
        sh_prefix = 0;
        sh_mask = 0;
        sh_remaining = k;
        sh_carry_eq = 0;
        sh_carry_out = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned d = l; d < 256; d += 1024) hist[d] = 0;
        __syncthreads();
        const unsigned prefix = sh_prefix, mask = sh_mask;
        for (unsigned p = l; p < P; p += 1024)
            if ((keys[p] & mask) == prefix) atomicAdd(&hist[(keys[p] >> shift) & 255U], 1U);
        __syncthreads();
        if (l == 0) {
            unsigned rem = sh_remaining;
            for (int d = 255; d >= 0; --d) {
                if (hist[d] >= rem) {
                    sh_prefix = prefix | (static_cast<unsigned>(d) << shift);
                    break;
                }
                rem -= hist[d];
            }
            sh_remaining = rem;
            sh_mask = mask | (255U << shift);
        }
        __syncthreads();
    }
    const unsigned kstar = sh_prefix, take_eq = sh_remaining;
    for (unsigned base = 0; base < P; base += 1024) {
        const unsigned p = base + l;
        const unsigned key = p < P ? keys[p] : 0U;
        const unsigned eq = (p < P && key == kstar) ? 1U : 0U;
        scan_eq[l] = eq;
        __syncthreads();
        for (unsigned off = 1; off < 1024; off <<= 1U) {
            const unsigned v = l >= off ? scan_eq[l - off] : 0U;
            __syncthreads();
            scan_eq[l] += v;
            __syncthreads();
        }
        const unsigned eq_rank = sh_carry_eq + scan_eq[l] - eq;
        const unsigned sel = (p < P && (key > kstar || (eq && eq_rank < take_eq))) ? 1U : 0U;
        scan_sel[l] = sel;
        __syncthreads();
        for (unsigned off = 1; off < 1024; off <<= 1U) {
            const unsigned v = l >= off ? scan_sel[l - off] : 0U;
            __syncthreads();
            scan_sel[l] += v;
            __syncthreads();
        }
        if (sel) {
            const unsigned at = sh_carry_out + scan_sel[l] - 1;
            for (unsigned j = 0; j < 4; ++j) tokens[at * 4 + j] = p * 4 + j;
        }
        __syncthreads();
        if (l == 1023) {
            sh_carry_eq += scan_eq[1023];
            sh_carry_out += scan_sel[1023];
        }
        __syncthreads();
    }
    if (l == 0) {
        unsigned n = sh_carry_out * 4;
        for (unsigned pos = P * 4; pos <= t; ++pos) tokens[n++] = pos;
        n_all[row] = n;
    }
}

// One CTA per query/chunk, all 64 heads share each gathered latent tile.
// QK uses eight warp-level 16x16x16 products; PV uses sixteen warps,
// each retaining 16x128 FP32 outputs in registers across the online loop.
// Selected keys are split into fixed 256-key chunks; combine visits chunks
// in ascending order. No atomics, whole-prefix scores or expanded K/V.
struct __align__(32) AttentionShared {
    // Padding avoids shared-bank conflicts in both HMMA operand layouts.
    BF latent[32*520],prob[64*40];
    float score[68*32],m[64],den[64],alpha[64];
};
__global__ void attention_queries(const float* qt,U16* out,unsigned rows) {
    auto i=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(i<std::size_t(rows)*64*512)out[i]=tobf(qt[i]);
}
__device__ unsigned pair_bf(const BF* p,unsigned stride=1) {
    return unsigned(__bfloat16_as_ushort(p[0]))|(unsigned(__bfloat16_as_ushort(p[stride]))<<16);
}
__device__ void mma_pv(float (&c)[4],unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__global__ __launch_bounds__(512) void sparse_mma(DsaState st,const U32* tokens,const U32* counts,const U16* qt,
                                                float* partial,float* stats) {
    extern __shared__ __align__(32) unsigned char storage[];
    auto& sh=*reinterpret_cast<AttentionShared*>(storage);
    unsigned r=blockIdx.x,split=blockIdx.y,tid=threadIdx.x,l=tid%32,warp=tid/32;
    unsigned begin=split*dsa_chunk,end=min(begin+dsa_chunk,counts[r]);
    // Empty splits are never read by combine.
    if(begin>=end)return;
    unsigned hg=warp/4,dc=(warp%4)*128,g=l/4,t=l%4;
    float out[16][4]{};
    if(tid<64){sh.m[tid]=-CUDART_INF_F;sh.den[tid]=0;}__syncthreads();
    for(unsigned base=begin;base<end;base+=32){
        // Direct 16-byte async copies: BF16 state is already the MMA operand.
        // src-size=0 zero-fills masked keys without accessing a future row.
        for(unsigned j=tid;j<32*64;j+=512){
            unsigned key=base+j/64,d=(j%64)*8;
            unsigned tok=key<end?tokens[std::size_t(r)*dsa_max_tokens+key]:0;
            unsigned dst=static_cast<unsigned>(__cvta_generic_to_shared(sh.latent+(j/64)*520+d));
            const U16* src=st.latent+std::size_t(tok)*512+d;
            unsigned bytes=key<end?16:0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(bytes));
        }
        asm volatile("cp.async.commit_group;"::);
        asm volatile("cp.async.wait_group 0;"::);
        __syncthreads();
        if(warp<8){
            wmma::fragment<wmma::matrix_a,16,16,16,BF,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,BF,wmma::col_major> bf0;
            wmma::fragment<wmma::accumulator,16,16,16,float> cf;
            wmma::fill_fragment(cf,0.f);
            unsigned head=(warp/2)*16,key=(warp%2)*16;
            for(unsigned k=0;k<512;k+=16){
                wmma::load_matrix_sync(af,reinterpret_cast<const BF*>(qt)+(std::size_t(r)*64+head)*512+k,512);
                wmma::load_matrix_sync(bf0,sh.latent+key*520+k,520);
                wmma::mma_sync(cf,af,bf0,cf);
            }
            wmma::store_matrix_sync(sh.score+head+key*68,cf,68,wmma::mem_col_major);
        }
        __syncthreads();
        // Eight lanes/head; four exponents/lane. The XOR max/sum trees
        // have fixed offsets 4,2,1 and never cross a head boundary.
        unsigned h=tid/8,l8=tid%8;float mx=-CUDART_INF_F;
        #pragma unroll
        for(unsigned j=0;j<4;++j){unsigned key=l8+j*8;
            if(base+key<end)mx=fmaxf(mx,sh.score[key*68+h]*0.0625f);
        }
        for(unsigned off=4;off;off/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,off,8));
        float m=fmaxf(sh.m[h],mx),alpha=declared_exp(sh.m[h]-m),sum=0;
        #pragma unroll
        for(unsigned j=0;j<4;++j){unsigned key=l8+j*8;
            float p=base+key<end?declared_exp(sh.score[key*68+h]*0.0625f-m):0;
            sum=__fadd_rn(sum,p);sh.prob[h*40+key]=__float2bfloat16_rn(p);
        }
        for(unsigned off=4;off;off/=2)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,off,8));
        if(l8==0){sh.m[h]=m;sh.den[h]=__fmaf_rn(sh.den[h],alpha,sum);sh.alpha[h]=alpha;}
        __syncthreads();
        float a0=sh.alpha[hg*16+g],a1=sh.alpha[hg*16+g+8];
        #pragma unroll
        for(unsigned col=0;col<16;++col){
            out[col][0]*=a0;out[col][1]*=a0;out[col][2]*=a1;out[col][3]*=a1;
        }
        #pragma unroll
        for(unsigned k=0;k<32;k+=16){
            const BF* pa=sh.prob+hg*16*40+g*40+k+2*t;
            unsigned aa0=pair_bf(pa),aa1=pair_bf(pa+8*40),aa2=pair_bf(pa+8),aa3=pair_bf(pa+8*40+8);
            #pragma unroll
            for(unsigned col=0;col<16;++col){
                const BF* pb=sh.latent+(k+2*t)*520+dc+col*8+g;
                mma_pv(out[col],aa0,aa1,aa2,aa3,pair_bf(pb,520),pair_bf(pb+8*520,520));
            }
        }
        __syncthreads();
    }
    auto slot=(std::size_t(r)*dsa_splits+split)*64;
    if(tid<64){stats[(slot+tid)*2]=sh.m[tid];stats[(slot+tid)*2+1]=sh.den[tid];}
    #pragma unroll
    for(unsigned col=0;col<16;++col){
        unsigned d=dc+col*8+2*t;
        partial[(slot+hg*16+g)*512+d]=out[col][0];
        partial[(slot+hg*16+g)*512+d+1]=out[col][1];
        partial[(slot+hg*16+g+8)*512+d]=out[col][2];
        partial[(slot+hg*16+g+8)*512+d+1]=out[col][3];
    }
}
__global__ void sparse_combine(const U32* counts,const float* partial,const float* stats,float* ob,unsigned* flag) {
    __shared__ float alpha[dsa_splits][16],beta[dsa_splits][16],denom[16];
    unsigned r=blockIdx.x,h0=blockIdx.y*16,tid=threadIdx.x,splits=(counts[r]+dsa_chunk-1)/dsa_chunk;
    // Softmax statistics are shared across all 512 value channels of a head.
    if(tid<16){float m=-CUDART_INF_F,den=0;
        for(unsigned s=0;s<splits;++s){auto at=(std::size_t(r)*dsa_splits+s)*64+h0+tid;
            float sm=stats[at*2],next=fmaxf(m,sm),a=declared_exp(m-next),b=declared_exp(sm-next);
            alpha[s][tid]=a;beta[s][tid]=b;den=__fmaf_rn(stats[at*2+1],b,den*a);m=next;
        }
        denom[tid]=den;
    }
    __syncthreads();
    for(unsigned j=tid;j<16*512;j+=256){unsigned h=j/512,d=j%512;float v=0;
        for(unsigned s=0;s<splits;++s){auto at=(std::size_t(r)*dsa_splits+s)*64+h0+h;
            v=__fmaf_rn(partial[at*512+d],beta[s][h],v*alpha[s][h]);
        }
        v/=denom[h];finite(v,flag);ob[(std::size_t(r)*64+h0+h)*512+d]=v;
    }
}
// Independent scalar-FMA reference: chronological score chain and online
// softmax. First stage one lane/key, second stage one lane/latent channel.
__global__ void sparse_reference_scores(DsaState st,const U32* tokens,const U32* counts,const float* qt,float* scores) {
    unsigned r=blockIdx.x,h=blockIdx.y,l=threadIdx.x,n=counts[r];
    for(unsigned j=l;j<n;j+=256){float v=0;unsigned tok=tokens[std::size_t(r)*dsa_max_tokens+j];
        for(unsigned k=0;k<512;++k)v=__fmaf_rn(qt[(std::size_t(r)*64+h)*512+k],bf(st.latent[std::size_t(tok)*512+k]),v);
        scores[(std::size_t(r)*64+h)*dsa_max_tokens+j]=v*0.0625f;}
}
__global__ void sparse_reference_pv(DsaState st,const U32* tokens,const U32* counts,const float* scores,float* ob,unsigned* flag) {
    unsigned r=blockIdx.x,h=blockIdx.y,d=threadIdx.x,n=counts[r];float m=-CUDART_INF_F,den=0,v=0;
    for(unsigned j=0;j<n;++j){float score=scores[(std::size_t(r)*64+h)*dsa_max_tokens+j],next=fmaxf(m,score),alpha=declared_exp(m-next),p=declared_exp(score-next);
        v=__fmaf_rn(p,bf(st.latent[std::size_t(tokens[std::size_t(r)*dsa_max_tokens+j])*512+d]),v*alpha);den=__fmaf_rn(den,alpha,p);m=next;}
    v/=den;finite(v,flag);ob[(std::size_t(r)*64+h)*512+d]=v;
}
__device__ bool has_pool(const U32* tok,unsigned n,unsigned p){unsigned lo=0,hi=n/4;while(lo<hi){unsigned mid=(lo+hi)/2,v=tok[mid*4]/4;if(v<p)lo=mid+1;else hi=mid;}return lo<n/4&&tok[lo*4]/4==p;}
__global__ void compare_sets(const float* scores,unsigned cap,unsigned first,const U32* tokens,const U32* counts,const U32* native,const U32* native_counts,DsaDiagnostics* diag,float abs,float rel) {
    unsigned r=blockIdx.x,l=threadIdx.x,P=(first+r+1)/4;
    const U32* ref=tokens+std::size_t(r)*dsa_max_tokens;const U32* act=native+std::size_t(r)*dsa_max_tokens;
    __shared__ float cutoff;__shared__ unsigned differs;
    if(l==0){cutoff=CUDART_INF_F;differs=counts[r]!=native_counts[r];if(P>dsa_selection_budget)for(unsigned j=0;j<512;++j)cutoff=fminf(cutoff,scores[std::size_t(r)*cap+ref[j*4]/4]);}__syncthreads();
    for(unsigned p=l;p<P;p+=256)if(has_pool(ref,counts[r],p)!=has_pool(act,native_counts[r],p)){
        atomicExch(&differs,1);atomicAdd(&diag->differing_pools,1ull);float score=scores[std::size_t(r)*cap+p];
        if(P>dsa_selection_budget&&fabsf(score-cutoff)<=abs+rel*fmaxf(fabsf(score),fabsf(cutoff)))atomicAdd(&diag->near_tie_pools,1ull);else atomicAdd(&diag->outside_tolerance_pools,1ull);}
    // Trailing token membership and exact cardinality are never near-tie exempt.
    if(l==0){unsigned nr=counts[r],na=native_counts[r];if(nr!=na)atomicAdd(&diag->outside_tolerance_pools,1ull);
        for(unsigned j=(nr/4)*4;j<nr;++j)if(j>=na||ref[j]!=act[j]){atomicExch(&differs,1);atomicAdd(&diag->outside_tolerance_pools,1ull);}}
    __syncthreads();if(l==0&&differs)atomicAdd(&diag->differing_rows,1ull);
}
bool valid(unsigned r,unsigned cap){return r>0&&r<=32768&&cap>=4&&cap<=131072&&cap%4==0;}
}
std::size_t dsa_export_bytes(unsigned frontier){return frontier<=131072?std::size_t(frontier)*1536+std::size_t(frontier/4)*512:0;}
cudaError_t enqueue_dsa_export(DsaState st,unsigned frontier,void* output,std::size_t bytes,cudaStream_t s){
    if(st.capacity<4||st.capacity>131072||st.capacity%4||frontier>st.capacity||bytes<dsa_export_bytes(frontier))return cudaErrorInvalidValue;
    if(!frontier)return cudaSuccess;
    if(!output||!st.latent||!st.key||!st.gate||!st.pooled)return cudaErrorInvalidValue;
    char* dst=static_cast<char*>(output);
    const void* src[4]={st.latent,st.key,st.gate,st.pooled};
    std::size_t sizes[4]={std::size_t(frontier)*1024,std::size_t(frontier)*256,std::size_t(frontier)*256,std::size_t(frontier/4)*512};
    for(unsigned j=0;j<4;++j){if(sizes[j]){auto e=cudaMemcpyAsync(dst,src[j],sizes[j],cudaMemcpyDeviceToDevice,s);if(e!=cudaSuccess)return e;}dst+=sizes[j];}
    return cudaSuccess;
}
cudaError_t prepare_dsa(){
    if(Projection<Row>::sufficient()<=0||Projection<Col>::sufficient()<=0)return cudaErrorInvalidConfiguration;
    auto e=cudaFuncSetAttribute(tiled_index,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(TiledIndexShared));
    if(e!=cudaSuccess)return e;
    return cudaFuncSetAttribute(sparse_mma,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(AttentionShared));
}
std::size_t dsa_workspace_bytes(unsigned r,unsigned cap){return valid(r,cap)?Arena(nullptr,cap).size:0;}
cudaError_t enqueue_dsa(const DsaCall& v,void* workspace,std::size_t bytes,cudaStream_t s,bool ref,DsaProfile* prof) {
    const auto& w=v.w;unsigned cap=v.state.capacity;
    if(!valid(v.rows,cap)||v.first>cap||v.rows>cap-v.first||!workspace||(reinterpret_cast<std::uintptr_t>(workspace)&255)||!v.x||!v.y||!v.nonfinite||!v.state.latent||!v.state.key||!v.state.gate||!v.state.pooled)return cudaErrorInvalidValue;
    if(!w.qa_norm||!w.kv_norm||!w.kvb||!w.key||!w.gate||!w.head_weight||!w.key_norm||!w.key_bias||!w.ape)return cudaErrorInvalidValue;
    Matrix matrices[5]={w.qa,w.qb,w.kva,w.out,w.iq};const unsigned ins[5]={4096,1536,4096,16384,1536},outs[5]={1536,16384,512,4096,4096};
    for(unsigned j=0;j<5;++j){auto m=matrices[j];if(m.input!=ins[j]||m.output!=outs[j]||!m.packed||!m.suh||!m.svh||m.bits<3||m.bits>6)return cudaErrorInvalidValue;}
    if((v.tokens==nullptr)!=(v.counts==nullptr)||(v.compare_tokens==nullptr)!=(v.compare_counts==nullptr)||(v.compare_tokens&&(!ref||!v.diagnostics))||(v.compare2_tokens&&(!ref||!v.diagnostics2||!v.compare2_counts))||!std::isfinite(v.near_tie_abs)||!std::isfinite(v.near_tie_rel)||v.near_tie_abs<0||v.near_tie_rel<0)return cudaErrorInvalidValue;
    if((v.node_mimic&&(!ref||!v.index_only))||(v.index_only&&!ref)||(v.compare2_tokens==nullptr)!=(v.compare2_counts==nullptr))return cudaErrorInvalidValue;
    Arena a(workspace,cap);if(bytes<a.size)return cudaErrorInvalidValue;
    const auto mode=v.mma_index?DsaIndex::mma:v.index;
    const bool accurate=!ref&&mode==DsaIndex::tf32x3;
    const bool index_fp32=ref||mode==DsaIndex::fp32||mode==DsaIndex::fp32_proj_mma_score;
    const bool score_fp32=ref||mode==DsaIndex::fp32||mode==DsaIndex::mma_proj_fp32_score;
    auto mark=[&](unsigned phase,unsigned window,bool end){return prof?cudaEventRecord((end?prof->end:prof->begin)[phase*dsa_max_windows+window],s):cudaSuccess;};
    #define DSA_TRY(expr) do {auto status=(expr);if(status!=cudaSuccess)return status;} while(0)
    // Only production one-row FP32 decode opts in. Diagnostic compare_sets can
    // read scores on mismatched membership, so diagnostics retain the old work.
    const bool skip_index=dsa_short_skip_eligible(v.short_skip,v.rows,v.first)&&!ref&&mode==DsaIndex::fp32
        &&!v.node_mimic&&!v.index_only&&!v.compare_tokens&&!v.compare2_tokens&&!v.diagnostics&&!v.diagnostics2;
    auto iqmark=[&](unsigned part,bool end){return v.index_query_profile?cudaEventRecord(
        (end?v.index_query_profile->end:v.index_query_profile->begin)[part],s):cudaSuccess;};
    const bool rowk=v.row_kernels&&!ref&&!accurate&&!v.node_mimic&&v.rows<=32;
    DSA_TRY(mark(0,0,false));for(unsigned j=0;j<5;++j){
        if(j==4)DSA_TRY(iqmark(0,false));
        if((!rowk||j==0||j==4)&&!(skip_index&&j==4))layer_decode<<<ins[j]/16*(outs[j]/16),256,0,s>>>(matrices[j],a.weights[j]);
        if(j==4)DSA_TRY(iqmark(0,true));
    }DSA_TRY(cudaGetLastError());if(accurate){
        const U16* src[5]={a.weights[0],a.weights[4],w.key,w.gate,w.head_weight};
        const unsigned sizes[5]={4096*1536,1536*4096,128*4096,128*4096,32*4096};
        for(unsigned j=0;j<5;++j)widen<<<(sizes[j]+255)/256,256,0,s>>>(src[j],a.accurate_weights[j],sizes[j]);
    }
    DSA_TRY(mark(0,0,true));
    auto proj=[&](unsigned j,const float* x,float* y,unsigned rows,bool f32=false){a.decoded=a.weights[j];return project(matrices[j],x,y,rows,a,s,ref||f32,!ref&&f32&&!v.legacy_fp32_index,rowk&&(!f32||(v.row_r2&r2_idx)),v.row_r2);}; // rowk: strips (non-FP32), with r2_idx also the FP32 row chains
    auto index_project=[&](unsigned j,const float* x,float* y,unsigned n){
        if(v.node_mimic){
            input_transform<<<(n*(matrices[j].input/128)+7)/8,256,0,s>>>(x,a.operand,matrices[j].suh,n,matrices[j].input);
            paired_projection<<<(n*matrices[j].output+255)/256,256,0,s>>>(a.operand,a.weights[j],a.raw,n,matrices[j].input,matrices[j].output);
            output_transform<<<(n*(matrices[j].output/128)+7)/8,256,0,s>>>(a.raw,y,matrices[j].svh,n,matrices[j].output);
            return cudaGetLastError();
        }
        if(!accurate)return proj(j,x,y,n,index_fp32);
        input_transform<<<(n*(matrices[j].input/128)+7)/8,256,0,s>>>(x,a.operand,matrices[j].suh,n,matrices[j].input);
        widen<<<(n*matrices[j].input+255)/256,256,0,s>>>(a.operand,a.accurate_operand,std::size_t(n)*matrices[j].input);
        auto e=accurate_gemm<Row>(a.accurate_operand,a.accurate_weights[j==0?0:1],a.raw,n,matrices[j].input,matrices[j].output,a.accurate_problem,s);
        if(e!=cudaSuccess)return e;
        output_transform<<<(n*(matrices[j].output/128)+7)/8,256,0,s>>>(a.raw,y,matrices[j].svh,n,matrices[j].output);
        return cudaGetLastError();
    };
    auto index_ordinary=[&](const float* x,const U16* w,float* y,unsigned n,unsigned out,unsigned slot){
        return accurate?accurate_gemm<Col>(x,a.accurate_weights[slot],y,n,4096,out,a.accurate_problem,s):ordinary(x,w,y,n,4096,out,a,s,index_fp32,!ref&&index_fp32&&!v.legacy_fp32_index,rowk&&(v.row_r2&(index_fp32?r2_idx:r2_dense)));
    };
    for(unsigned base=0,window=0;base<v.rows;base+=dsa_tile,++window){unsigned n=std::min(dsa_tile,v.rows-base);const float* x=v.x+std::size_t(base)*4096;
        DSA_TRY(mark(1,window,false));if(!v.index_only){DSA_TRY(proj(2,x,a.lat,n));rms<<<n,256,0,s>>>(a.lat,w.kv_norm,a.lat,512);}
        DSA_TRY(index_ordinary(x,w.key,a.key,n,128,2));DSA_TRY(index_ordinary(x,w.gate,a.gate,n,128,3));
        store<<<n,128,0,s>>>(v.state,v.first+base,a.lat,a.key,a.gate,w.key_norm,w.key_bias,v.nonfinite,v.node_mimic,v.index_only);DSA_TRY(cudaGetLastError());DSA_TRY(mark(1,window,true));
    }
    DSA_TRY(mark(2,0,false));unsigned p0=v.first/4,p1=(v.first+v.rows)/4;if(p1>p0)pool<<<p1-p0,128,0,s>>>(v.state,p0,w.ape,v.nonfinite);DSA_TRY(cudaGetLastError());DSA_TRY(mark(2,0,true));
    for(unsigned base=0,window=0;base<v.rows;base+=dsa_tile,++window){unsigned n=std::min(dsa_tile,v.rows-base);const float* x=v.x+std::size_t(base)*4096;
        DSA_TRY(mark(3,window,false));
        // V2 attention q-A stays on CUTLASS FP16; preserve historical shared
        // scalar q-A for the v1 FP32 identity and split the v2 index branch.
        if(v.node_mimic){DSA_TRY(index_project(0,x,a.qa,n));}else {DSA_TRY(proj(0,x,a.qa,n,mode==DsaIndex::fp32&&index_fp32));}rms<<<n,256,0,s>>>(a.qa,w.qa_norm,a.qr,1536,v.node_mimic);
        if(!v.index_only){DSA_TRY(proj(1,a.qr,a.q,n));}
        if(!ref&&((index_fp32&&mode!=DsaIndex::fp32)||accurate)){DSA_TRY(index_project(0,x,a.qa,n));rms<<<n,256,0,s>>>(a.qa,w.qa_norm,a.qr,1536,v.node_mimic);}
        DSA_TRY(iqmark(1,false));if(!skip_index){DSA_TRY(index_project(4,a.qr,a.iq,n));DSA_TRY(index_ordinary(x,w.head_weight,a.wh,n,32,4));}DSA_TRY(iqmark(1,true));DSA_TRY(mark(3,window,true));
        DSA_TRY(mark(4,window,false));if(v.index_only){}else if(ref)absorb<false,true><<<dim3((n+15)/16,32,64),32,0,s>>>(a.q,w.kvb,a.qt,n);else absorb<false,false><<<dim3((n+15)/16,32,64),32,0,s>>>(a.q,w.kvb,a.qt,n);DSA_TRY(cudaGetLastError());DSA_TRY(mark(4,window,true));
        DSA_TRY(mark(5,window,false));DSA_TRY(iqmark(2,false));if(skip_index){}else if(accurate){
            tiled_index<<<dim3(std::max(1u,((v.first+base+n)/4+31)/32),(n+1)/2),128,sizeof(TiledIndexShared),s>>>(v.state.pooled,a.iq,a.wh,a.scores,a.keys,cap/4,v.first+base,n,v.nonfinite);
            streaming_index<<<n,256,0,s>>>(a.keys,cap/4,v.first+base,a.tokens,a.counts);
        }
        else if(score_fp32&&!ref&&!v.legacy_fp32_index)index_fp32_tiled<<<dim3(std::max(1u,((v.first+base+n)/4+15)/16),(n+1)/2),128,0,s>>>(v.state.pooled,a.iq,a.wh,a.scores,a.keys,cap/4,v.first+base,n,v.nonfinite);
        else if(score_fp32)index_scores<1><<<dim3(std::max(1u,((v.first+base+n)/4+15)/16),n),32,0,s>>>(v.state.pooled,a.iq,a.wh,a.scores,a.keys,cap/4,v.first+base,v.nonfinite);
        else index_scores<0><<<dim3(std::max(1u,((v.first+base+n)/4+15)/16),n),32,0,s>>>(v.state.pooled,a.iq,a.wh,a.scores,a.keys,cap/4,v.first+base,v.nonfinite);DSA_TRY(cudaGetLastError());DSA_TRY(iqmark(2,true));DSA_TRY(mark(5,window,true));
        DSA_TRY(mark(6,window,false));if(!accurate)select_rows<<<n,1024,0,s>>>(a.keys,cap/4,v.first+base,a.tokens,a.counts);
        if(v.tokens){DSA_TRY(cudaMemcpyAsync(v.tokens+std::size_t(base)*dsa_max_tokens,a.tokens,std::size_t(n)*dsa_max_tokens*4,cudaMemcpyDeviceToDevice,s));DSA_TRY(cudaMemcpyAsync(v.counts+base,a.counts,n*4,cudaMemcpyDeviceToDevice,s));}
        if(v.compare_tokens)compare_sets<<<n,256,0,s>>>(a.scores,cap/4,v.first+base,a.tokens,a.counts,v.compare_tokens+std::size_t(base)*dsa_max_tokens,v.compare_counts+base,v.diagnostics,v.near_tie_abs,v.near_tie_rel);
        if(v.compare2_tokens)compare_sets<<<n,256,0,s>>>(a.scores,cap/4,v.first+base,a.tokens,a.counts,v.compare2_tokens+std::size_t(base)*dsa_max_tokens,v.compare2_counts+base,v.diagnostics2,v.near_tie_abs,v.near_tie_rel);
        DSA_TRY(cudaGetLastError());DSA_TRY(mark(6,window,true));
        if(v.index_only)continue;
        DSA_TRY(mark(7,window,false));if(ref){sparse_reference_scores<<<dim3(n,64),256,0,s>>>(v.state,a.tokens,a.counts,a.qt,a.ref_scores);sparse_reference_pv<<<dim3(n,64),512,0,s>>>(v.state,a.tokens,a.counts,a.ref_scores,a.ob,v.nonfinite);}
        else {attention_queries<<<(n*64*512+255)/256,256,0,s>>>(a.qt,a.qt_bf,n);
            sparse_mma<<<dim3(n,dsa_splits),512,sizeof(AttentionShared),s>>>(v.state,a.tokens,a.counts,a.qt_bf,a.partial,a.stats);
            sparse_combine<<<dim3(n,4),256,0,s>>>(a.counts,a.partial,a.stats,a.ob,v.nonfinite);}DSA_TRY(cudaGetLastError());DSA_TRY(mark(7,window,true));
        DSA_TRY(mark(8,window,false));if(ref)absorb<true,true><<<dim3((n+15)/16,16,64),32,0,s>>>(a.ob,w.kvb,a.o,n);else absorb<true,false><<<dim3((n+15)/16,16,64),32,0,s>>>(a.ob,w.kvb,a.o,n);DSA_TRY(cudaGetLastError());DSA_TRY(mark(8,window,true));
        DSA_TRY(mark(9,window,false));DSA_TRY(proj(3,a.o,v.y+std::size_t(base)*4096,n));DSA_TRY(mark(9,window,true));
    }
    #undef DSA_TRY
    return cudaGetLastError();
}
cudaError_t self_check_dsa_index(const DsaCall& v,void* workspace,std::size_t bytes,cudaStream_t s,DsaExactReport& report){
    const unsigned cap=v.state.capacity;
    if(!valid(v.rows,cap)||bytes<dsa_workspace_bytes(v.rows,cap)||v.first+v.rows>cap)return cudaErrorInvalidValue;
    Arena a(workspace,cap);report={};
    #define EXACT_TRY(expr) do{auto e=(expr);if(e!=cudaSuccess)return e;}while(0)
    auto compare=[&](auto launch_old,auto launch_new,const void* device,std::size_t count,std::uint64_t& differing)->cudaError_t{
        std::vector<U32> old(count),now(count);launch_old();EXACT_TRY(cudaGetLastError());
        EXACT_TRY(cudaMemcpyAsync(old.data(),device,count*4,cudaMemcpyDeviceToHost,s));EXACT_TRY(cudaStreamSynchronize(s));
        // Never let an omitted NEW store pass by inheriting OLD's output.
        for(std::size_t j=0;j<count;++j)now[j]=~old[j];
        EXACT_TRY(cudaMemcpyAsync(const_cast<void*>(device),now.data(),count*4,cudaMemcpyHostToDevice,s));
        launch_new();EXACT_TRY(cudaGetLastError());EXACT_TRY(cudaMemcpyAsync(now.data(),device,count*4,cudaMemcpyDeviceToHost,s));EXACT_TRY(cudaStreamSynchronize(s));
        report.memcmp_failures+=std::memcmp(old.data(),now.data(),count*4)!=0;
        for(std::size_t j=0;j<count;++j)differing+=old[j]!=now[j];return cudaSuccess;
    };
    for(auto item:{std::pair<Matrix,U16*>{v.w.qa,a.weights[0]},{v.w.iq,a.weights[4]}})
        layer_decode<<<item.first.input/16*(item.first.output/16),256,0,s>>>(item.first,item.second);
    for(unsigned base=0;base<v.rows;base+=64){unsigned n=std::min(64u,v.rows-base);const float* x=v.x+std::size_t(base)*4096;
        auto projection=[&](Matrix m,U16* decoded,const float* src,float* dst)->cudaError_t{
            input_transform<<<(n*(m.input/128)+7)/8,256,0,s>>>(src,a.operand,m.suh,n,m.input);
            EXACT_TRY(compare([&]{projection_reference<<<(n*m.output+255)/256,256,0,s>>>(nullptr,a.operand,decoded,a.raw,n,m.input,m.output,false);},
                [&]{projection_fp32_tiled<<<dim3((m.output+31)/32,(n+7)/8),128,0,s>>>(nullptr,a.operand,decoded,a.raw,n,m.input,m.output,false);},
                a.raw,std::size_t(n)*m.output,report.projection_differing));
            report.projection_elements+=std::size_t(n)*m.output;
            output_transform<<<(n*(m.output/128)+7)/8,256,0,s>>>(a.raw,dst,m.svh,n,m.output);return cudaSuccess;
        };
        EXACT_TRY(projection(v.w.qa,a.weights[0],x,a.qa));rms<<<n,256,0,s>>>(a.qa,v.w.qa_norm,a.qr,1536);
        EXACT_TRY(projection(v.w.iq,a.weights[4],a.qr,a.iq));
        for(auto item:{std::pair<const U16*,unsigned>{v.w.key,128},{v.w.gate,128},{v.w.head_weight,32}}){
            float* dst=item.second==32?a.wh:a.raw;
            EXACT_TRY(compare([&]{projection_reference<<<(n*item.second+255)/256,256,0,s>>>(x,nullptr,item.first,dst,n,4096,item.second,true);},
                [&]{projection_fp32_tiled<<<dim3((item.second+31)/32,(n+7)/8),128,0,s>>>(x,nullptr,item.first,dst,n,4096,item.second,true);},
                dst,std::size_t(n)*item.second,report.projection_differing));
            report.projection_elements+=std::size_t(n)*item.second;
        }
        // Scores use the final 32-column head weights in wh.
        auto score_bytes=std::size_t(64)*(cap/4)*4;
        EXACT_TRY(cudaMemsetAsync(a.scores,0,score_bytes,s));EXACT_TRY(cudaMemsetAsync(a.keys,0,score_bytes,s));
        auto grid=dim3(std::max(1u,((v.first+base+n)/4+15)/16),n);
        EXACT_TRY(compare([&]{cudaMemsetAsync(a.scores,0,score_bytes,s);cudaMemsetAsync(a.keys,0,score_bytes,s);index_scores<1><<<grid,32,0,s>>>(v.state.pooled,a.iq,a.wh,a.scores,a.keys,cap/4,v.first+base,v.nonfinite);},
            [&]{cudaMemsetAsync(a.scores,0,score_bytes,s);cudaMemsetAsync(a.keys,0,score_bytes,s);index_fp32_tiled<<<dim3(grid.x,(n+1)/2),128,0,s>>>(v.state.pooled,a.iq,a.wh,a.scores,a.keys,cap/4,v.first+base,n,v.nonfinite);},
            a.scores,std::size_t(n)*(cap/4),report.score_differing));
        EXACT_TRY(compare([&]{cudaMemsetAsync(a.scores,0,score_bytes,s);cudaMemsetAsync(a.keys,0,score_bytes,s);index_scores<1><<<grid,32,0,s>>>(v.state.pooled,a.iq,a.wh,a.scores,a.keys,cap/4,v.first+base,v.nonfinite);},
            [&]{cudaMemsetAsync(a.scores,0,score_bytes,s);cudaMemsetAsync(a.keys,0,score_bytes,s);index_fp32_tiled<<<dim3(grid.x,(n+1)/2),128,0,s>>>(v.state.pooled,a.iq,a.wh,a.scores,a.keys,cap/4,v.first+base,n,v.nonfinite);},
            a.keys,std::size_t(n)*(cap/4),report.key_differing));
        report.score_elements+=std::size_t(n)*(cap/4);
    }
    #undef EXACT_TRY
    return cudaGetLastError();
}
}

// Layer-45 MTP composition, full heads/experts. No vocabulary head or trunk MHC.
#include "mtp.cuh"
namespace strix::glm::rtxpf {
namespace {
struct MtpArena {
    Problem* problem;U16 *decoded,*operand;float* raw;
    float *cat,*z,*xn,*part,*residual,*routed,*shared,*logits,*rates;
    U32* ids;void *dsa,*moe;std::size_t dsa_bytes,moe_bytes,size;
    MtpArena(void* p,unsigned rows,unsigned cap){std::size_t off=0;auto take=[&](std::size_t n){auto at=off;off+=(n+255)&~std::size_t(255);return p?static_cast<char*>(p)+at:nullptr;};
        problem=(Problem*)take(sizeof(Problem));decoded=(U16*)take(std::size_t(8192)*4096*2);operand=(U16*)take(std::size_t(rows)*8192*2);raw=(float*)take(std::size_t(rows)*4096*4);
        cat=(float*)take(std::size_t(rows)*8192*4);
        auto hidden=[&](){return (float*)take(std::size_t(rows)*4096*4);};
        z=hidden();xn=hidden();part=hidden();residual=hidden();routed=hidden();shared=hidden();
        logits=(float*)take(std::size_t(rows)*288*4);rates=(float*)take(std::size_t(rows)*8*4);ids=(U32*)take(std::size_t(rows)*8*4);
        dsa_bytes=dsa_workspace_bytes(rows,cap);dsa=take(dsa_bytes);
        moe_bytes=std::max({workspace_bytes_cutlass(rows,288,4096,2048),workspace_bytes_cutlass(rows,1,4096,2048),workspace_bytes(rows,288,4096,2048),workspace_bytes_dense(rows,4096,2048)});moe=take(moe_bytes);size=off;
    }
};
__global__ void mtp_cat(const float* e,const float* h,float* cat,unsigned rows) {auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j<std::size_t(rows)*4096){unsigned r=j/4096,d=j%4096;cat[std::size_t(r)*8192+d]=e[j];cat[std::size_t(r)*8192+4096+d]=h[j];}}
__global__ void mtp_add(const float* a,const float* b,float* y,unsigned rows,unsigned* flag){auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j<std::size_t(rows)*4096){float v=a[j]+b[j];finite(v,flag);y[j]=v;}}
__global__ void router_logits(const float* x,const U16* w,float* logits,unsigned* flag) {
    unsigned r=blockIdx.x,e=threadIdx.x;if(e>=288)return;float v=0;for(unsigned k=0;k<4096;++k)v=__fmaf_rn(x[std::size_t(r)*4096+k],hf(w[std::size_t(e)*4096+k]),v);finite(v,flag);logits[std::size_t(r)*288+e]=v;
}
__global__ void router_select(const float* logits,const float* bias,U32* ids,float* rates) {
    __shared__ float sg[288];__shared__ U32 keys[288];unsigned r=blockIdx.x,e=threadIdx.x;
    if(e<288){float v=sig(logits[std::size_t(r)*288+e]);sg[e]=v;keys[e]=ordered(v+bias[e]);}__syncthreads();if(e!=0)return;
    bool used[288]{};unsigned pick[8];for(unsigned j=0;j<8;++j){unsigned best=288;for(unsigned k=0;k<288;++k)if(!used[k]&&(best==288||keys[k]>keys[best]))best=k;pick[j]=best;used[best]=true;}
    for(unsigned j=1;j<8;++j)for(unsigned k=j;k&&pick[k-1]>pick[k];--k){unsigned t=pick[k-1];pick[k-1]=pick[k];pick[k]=t;}
    float den=0;for(unsigned j=0;j<8;++j)den+=sg[pick[j]];den+=1.e-20f;
    for(unsigned j=0;j<8;++j){ids[std::size_t(r)*8+j]=pick[j];rates[std::size_t(r)*8+j]=sg[pick[j]]/den*2.5f;}
}
}
std::size_t mtp_workspace_bytes(unsigned rows,unsigned cap){return valid(rows,cap)?MtpArena(nullptr,rows,cap).size:0;}
cudaError_t enqueue_mtp(const MtpCall& v,void* workspace,std::size_t bytes,cudaStream_t s,const CutlassExecution& exec,bool ref){
    const auto& w=v.w;unsigned rows=v.rows;
    if(!valid(rows,v.state.capacity)||v.first>v.state.capacity||rows>v.state.capacity-v.first||w.layer!=45||!workspace||(reinterpret_cast<std::uintptr_t>(workspace)&255)||!v.embedding||!v.hidden||!v.y||!v.nonfinite||!w.enorm||!w.hnorm||!w.input_norm||!w.post_norm||!w.head_norm||!w.router||!w.router_bias||!w.experts||!w.shared||!w.eh.packed||!w.eh.suh||!w.eh.svh||w.eh.input!=8192||w.eh.output!=4096||w.eh.bits<3||w.eh.bits>6)return cudaErrorInvalidValue;
    MtpArena a(workspace,rows,v.state.capacity);if(bytes<a.size)return cudaErrorInvalidValue;
    #define MTP_TRY(expr) do {auto status=(expr);if(status!=cudaSuccess)return status;} while(0)
    if(v.phase>2||(v.phase==1&&!v.route_ids))return cudaErrorInvalidValue;
    unsigned blocks=unsigned((std::size_t(rows)*4096+255)/256);
    if(v.phase!=2){
    rms<<<rows,256,0,s>>>(v.embedding,w.enorm,a.xn,4096);rms<<<rows,256,0,s>>>(v.hidden,w.hnorm,a.part,4096);
    mtp_cat<<<unsigned((std::size_t(rows)*4096+255)/256),256,0,s>>>(a.xn,a.part,a.cat,rows);
    const bool rowk=v.row_proj&&!ref&&row_project_fits(w.eh,rows);
    if(!rowk)layer_decode<<<8192/16*(4096/16),256,0,s>>>(w.eh,a.decoded);MTP_TRY(project(w.eh,a.cat,a.z,rows,a,s,ref,false,rowk,v.row_r2));
    rms<<<rows,256,0,s>>>(a.z,w.input_norm,a.xn,4096);
    DsaCall d{};d.rows=rows;d.first=v.first;d.w=w.attention;d.state=v.state;d.x=a.xn;d.y=a.part;d.nonfinite=v.nonfinite;d.index=v.index;d.legacy_fp32_index=v.legacy_fp32_index;d.row_kernels=v.row_proj;d.row_r2=v.row_r2;d.short_skip=v.dsa_short_skip;d.index_query_profile=v.index_query_profile;
    MTP_TRY(enqueue_dsa(d,a.dsa,a.dsa_bytes,s,ref));
    mtp_add<<<blocks,256,0,s>>>(a.z,a.part,a.residual,rows,v.nonfinite);rms<<<rows,256,0,s>>>(a.residual,w.post_norm,a.xn,4096);
    router_logits<<<rows,320,0,s>>>(a.xn,w.router,a.logits,v.nonfinite);router_select<<<rows,320,0,s>>>(a.logits,w.router_bias,a.ids,a.rates);
    if(v.phase==1){MTP_TRY(cudaMemcpyAsync(v.route_ids,a.ids,std::size_t(rows)*8*4,cudaMemcpyDeviceToDevice,s));return cudaGetLastError();}
    }
    Call moe{rows,288,4096,2048,a.xn,a.ids,a.rates,w.experts,a.routed};moe.row_gemv=v.row_moe;moe.row_r2=v.row_r2;
    if(ref)MTP_TRY(enqueue_reference(moe,a.moe,a.moe_bytes,s));else MTP_TRY(enqueue_cutlass(moe,a.moe,a.moe_bytes,s,exec));
    Call shared{rows,1,4096,2048,a.xn,nullptr,nullptr,w.shared,a.shared};shared.row_gemv=v.row_moe;shared.row_r2=v.row_r2;
    if(ref)MTP_TRY(enqueue_reference(shared,a.moe,a.moe_bytes,s));else MTP_TRY(enqueue_cutlass(shared,a.moe,a.moe_bytes,s,exec));
    mtp_add<<<blocks,256,0,s>>>(a.routed,a.shared,a.part,rows,v.nonfinite);mtp_add<<<blocks,256,0,s>>>(a.residual,a.part,a.z,rows,v.nonfinite);rms<<<rows,256,0,s>>>(a.z,w.head_norm,v.y,4096);
    #undef MTP_TRY
    return cudaGetLastError();
}
}
