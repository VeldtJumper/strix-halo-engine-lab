#include "kda.cuh"
#include "row_mma.cuh"
#include <mma.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <algorithm>

namespace strix::glm::rtxpf {
namespace {
constexpr unsigned D=128,B=32;
#include "projection.inl"
struct Arena {
    Problem* problem;
    U16 *decoded,*operand;
    float *raw,*qkv,*f,*g,*beta,*lo,*hi,*prep,*o,*z,*prefix,*ck,*cq,*bk,*bq,*delta;
    std::size_t size;
    Arena(void* p,unsigned r,unsigned h,unsigned heads) {
        std::size_t off=0,c=heads*D;
        auto take=[&](std::size_t n){auto old=off;off+=(n+255)&~std::size_t(255);return p?static_cast<char*>(p)+old:nullptr;};
        // Reuse decoded storage only after that projection has consumed it.
        problem=(Problem*)take(sizeof(Problem));
        decoded=(U16*)take(std::size_t(h)*3*c*2);
        operand=(U16*)take(std::size_t(r)*std::max<std::size_t>(h,c)*2);
        raw=(float*)take(std::size_t(r)*std::max<std::size_t>(h,3*c)*4);
        qkv=(float*)take(std::size_t(r)*3*c*4);
        f=(float*)take(std::size_t(r)*c*4);g=(float*)take(std::size_t(r)*c*4);
        beta=(float*)take(std::size_t(r)*heads*4);
        lo=(float*)take(std::size_t(r)*D*4);hi=(float*)take(std::size_t(r)*D*4);
        prep=(float*)take(std::size_t(r)*heads*4*D*4);
        o=(float*)take(std::size_t(r)*c*4);z=(float*)take(std::size_t(r)*c*4);
        prefix=(float*)take(std::size_t(heads)*B*D*4);
        ck=(float*)take(std::size_t(heads)*B*B*4);cq=(float*)take(std::size_t(heads)*B*B*4);
        bk=(float*)take(std::size_t(heads)*B*D*4);bq=(float*)take(std::size_t(heads)*B*D*4);
        delta=(float*)take(std::size_t(heads)*B*D*4);
        size=off;
    }
};
__global__ void prepare(KdaWeights w,const float* qkv,const float* f,float* beta,const float* cv,float* prep) {
    __shared__ float sums[128];unsigned h=blockIdx.x,r=blockIdx.y,i=threadIdx.x,c=w.heads*D;
    float v[3];
    for(unsigned p=0;p<3;++p){unsigned ch=p*c+h*D+i;float acc=0;
        for(unsigned j=0;j<4;++j){int t=int(r)+int(j)-3;float x=t<0?cv[std::size_t(ch)*3+3+t]:qkv[std::size_t(t)*3*c+ch];acc=__fmaf_rn(bf(w.conv[ch*4+j]),x,acc);}v[p]=acc*sig(acc);}
    float norm[2];for(unsigned p=0;p<2;++p){sums[i]=v[p]*v[p];__syncthreads();for(unsigned k=64;k;k/=2){if(i<k)sums[i]+=sums[i+k];__syncthreads();}norm[p]=sqrtf(sums[0]+1.e-6f);__syncthreads();}
    std::size_t rh=std::size_t(r)*w.heads+h;float* sc=prep+rh*4*D;
    sc[i]=v[0]/norm[0]*0x1.6a09e6p-4f;sc[D+i]=v[1]/norm[1];sc[2*D+i]=v[2];
    sc[3*D+i]=-5.f*sig(declared_exp(w.a_log[h])*(f[std::size_t(r)*c+h*D+i]+w.dt[h*D+i]));
    if(i==0)beta[rh]=sig(beta[rh]);
}
__global__ void commit_conv(const float* qkv,float* cv,unsigned rows,unsigned channels) {
    unsigned ch=blockIdx.x*256+threadIdx.x;if(ch>=channels)return;float old[3];for(unsigned j=0;j<3;++j)old[j]=cv[std::size_t(ch)*3+j];
    for(unsigned j=0;j<3;++j){int t=int(rows)+int(j)-3;cv[std::size_t(ch)*3+j]=t<0?old[3+t]:qkv[std::size_t(t)*channels+ch];}
}
__global__ void snapshot_conv(const float* qkv,const float* cv,float* out,unsigned row,unsigned channels) {
    unsigned ch=blockIdx.x*256+threadIdx.x;if(ch>=channels)return;
    for(unsigned j=0;j<3;++j){int t=int(row)+int(j)-3;out[std::size_t(ch)*3+j]=t<0?cv[std::size_t(ch)*3+3+t]:qkv[std::size_t(t)*channels+ch];}
}
__global__ void sequential(const float* p,const float* beta,float* state,float* out,unsigned rows,unsigned heads,unsigned first) {
    unsigned h=blockIdx.x,v=threadIdx.x;float st[D];
    #pragma unroll
    for(unsigned k=0;k<D;++k)st[k]=state[(std::size_t(h)*D+k)*D+v];
    for(unsigned t=0;t<rows;++t){auto rh=std::size_t(first+t)*heads+h;const float* sc=p+rh*4*D;float pred=0;
        #pragma unroll
        for(unsigned k=0;k<D;++k){st[k]*=declared_exp(sc[3*D+k]);pred=__fmaf_rn(st[k],sc[D+k],pred);}
        float delta=(sc[2*D+v]-pred)*beta[rh],o=0;
        #pragma unroll
        for(unsigned k=0;k<D;++k){st[k]=__fmaf_rn(sc[D+k],delta,st[k]);o=__fmaf_rn(st[k],sc[k],o);}out[rh*D+v]=o;
    }
    #pragma unroll
    for(unsigned k=0;k<D;++k)state[(std::size_t(h)*D+k)*D+v]=st[k];
}
// Prefix log-decays are local to 32 tokens. We only exponentiate NONPOSITIVE
// differences, never divide by a decayed prefix or form inverse decays. This
// remains bounded for -5 vector gates (exp(-160) may safely underflow to zero).
__global__ void prefixes(const float* p,float* pre,unsigned heads,unsigned first,unsigned n) {
    unsigned h=blockIdx.x,k=threadIdx.x;float sum=0;
    for(unsigned t=0;t<n;++t){sum+=p[(std::size_t(first+t)*heads+h)*4*D+3*D+k];pre[(std::size_t(h)*B+t)*D+k]=sum;}
}
using namespace nvcuda;
// One warp/CTA, 16x16x16 WMMA; all shared operands are 32-byte aligned.
// Coefficients K[t,s]=k_t^T D(t,s) k_s, Q[t,s]=q_t^T D(t,s) k_s.
// Query t is fixed; rows are preceding s and B columns repeat k_t/q_t.
__global__ void triangular(const float* p,const float* pre,float* ck,float* cq,unsigned heads,unsigned first,unsigned n) {
    __shared__ __align__(32) __half a[16*16],b[16*16];
    __shared__ __align__(32) float c[16*16];
    unsigned h=blockIdx.x,t=blockIdx.y,s0=blockIdx.z*16,l=threadIdx.x;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf0;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;wmma::fill_fragment(cf,0.f);
    for(unsigned k0=0;k0<D;k0+=16){
        for(unsigned j=l;j<256;j+=32){unsigned s=s0+j/16,k=k0+j%16;float decay=s<=t&&s<n?declared_exp(pre[(std::size_t(h)*B+t)*D+k]-pre[(std::size_t(h)*B+s)*D+k]):0;
            a[j]=__float2half_rn(s<=t&&s<n?p[(std::size_t(first+s)*heads+h)*4*D+D+k]*decay:0);
            unsigned kk=k0+j/16,col=j%16;b[j]=__float2half_rn(p[(std::size_t(first+t)*heads+h)*4*D+(col<8?D:0)+kk]);}
        __syncwarp();wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bf0,b,16);wmma::mma_sync(cf,af,bf0,cf);__syncwarp();
    }
    wmma::store_matrix_sync(c,cf,16,wmma::mem_row_major);__syncwarp();
    for(unsigned s=l;s<16;s+=32)if(s0+s<n){auto ix=(std::size_t(h)*B+t)*B+s0+s;ck[ix]=s0+s<t?c[s*16]:0;cq[ix]=s0+s<=t?c[s*16+8]:0;}
}
// Base products against incoming state. Rows 0..31 use k, 32..63 use q.
__global__ void base_products(const float* p,const float* pre,const float* state,float* bk,float* bq,unsigned heads,unsigned first,unsigned n) {
    __shared__ __align__(32) __half a[256],b[256];__shared__ __align__(32) float c[256];
    unsigned h=blockIdx.x,row0=blockIdx.y*16,v0=blockIdx.z*16,l=threadIdx.x;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf0;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;wmma::fill_fragment(cf,0.f);
    for(unsigned k0=0;k0<D;k0+=16){for(unsigned j=l;j<256;j+=32){unsigned t=(row0+j/16)%B,k=k0+j%16;
        a[j]=__float2half_rn(t<n?p[(std::size_t(first+t)*heads+h)*4*D+(row0<B?D:0)+k]*declared_exp(pre[(std::size_t(h)*B+t)*D+k]):0);
        b[j]=__float2half_rn(state[(std::size_t(h)*D+k0+j/16)*D+v0+j%16]);}
        __syncwarp();wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bf0,b,16);wmma::mma_sync(cf,af,bf0,cf);__syncwarp();}
    wmma::store_matrix_sync(c,cf,16,wmma::mem_row_major);__syncwarp();
    for(unsigned j=l;j<256;j+=32){unsigned t=(row0+j/16)%B;if(t<n)(row0<B?bk:bq)[(std::size_t(h)*B+t)*D+v0+j%16]=c[j];}
}
__global__ void solve(const float* p,const float* beta,const float* ck,const float* bk,float* delta,unsigned heads,unsigned first,unsigned n) {
    unsigned h=blockIdx.x,v=threadIdx.x;
    for(unsigned t=0;t<n;++t){float pred=bk[(std::size_t(h)*B+t)*D+v];
        for(unsigned s=0;s<t;++s)pred=__fmaf_rn(ck[(std::size_t(h)*B+t)*B+s],delta[(std::size_t(h)*B+s)*D+v],pred);
        delta[(std::size_t(h)*B+t)*D+v]=(p[(std::size_t(first+t)*heads+h)*4*D+2*D+v]-pred)*beta[std::size_t(first+t)*heads+h];}
}
// O = baseQ + causal Q*delta; final S = D_end*S_in + weighted K^T*delta.
// Persistent state remains FP32. FP16 rounding is restricted to block operands.
template<bool Carry>
__global__ void block_products(const float* p,const float* pre,const float* cq,const float* bq,const float* delta,float* state,float* out,unsigned heads,unsigned first,unsigned n) {
    __shared__ __align__(32) __half a[256],b[256];__shared__ __align__(32) float c[256];
    unsigned h=blockIdx.x,row0=blockIdx.y*16,v0=blockIdx.z*16,l=threadIdx.x;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf0;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;wmma::fill_fragment(cf,0.f);
    for(unsigned s0=0;s0<B;s0+=16){for(unsigned j=l;j<256;j+=32){unsigned r=row0+j/16,s=s0+j%16;float av=0;
        if constexpr(Carry){if(s<n)av=p[(std::size_t(first+s)*heads+h)*4*D+D+r]*declared_exp(pre[(std::size_t(h)*B+n-1)*D+r]-pre[(std::size_t(h)*B+s)*D+r]);}
        else {if(r<n&&s<n)av=cq[(std::size_t(h)*B+r)*B+s];}
        a[j]=__float2half_rn(av);unsigned ds=s0+j/16;b[j]=__float2half_rn(ds<n?delta[(std::size_t(h)*B+ds)*D+v0+j%16]:0);}
        __syncwarp();wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bf0,b,16);wmma::mma_sync(cf,af,bf0,cf);__syncwarp();}
    wmma::store_matrix_sync(c,cf,16,wmma::mem_row_major);__syncwarp();
    for(unsigned j=l;j<256;j+=32){unsigned r=row0+j/16,v=v0+j%16;
        if constexpr(Carry){auto at=(std::size_t(h)*D+r)*D+v;state[at]=__fmaf_rn(declared_exp(pre[(std::size_t(h)*B+n-1)*D+r]),state[at],c[j]);}
        else if(r<n)out[(std::size_t(first+r)*heads+h)*D+v]=bq[(std::size_t(h)*B+r)*D+v]+c[j];}
}
__global__ void output_gate(KdaWeights w,const float* out,const float* gate,float* z) {
    __shared__ float sum[128];unsigned h=blockIdx.x,r=blockIdx.y,i=threadIdx.x;auto at=(std::size_t(r)*w.heads+h)*D+i;float v=out[at];sum[i]=v*v;__syncthreads();
    for(unsigned k=64;k;k/=2){if(i<k)sum[i]+=sum[i+k];__syncthreads();}
    z[at]=v/sqrtf(sum[0]/D+1.e-5f)*bf(w.norm[i])*sig(gate[at]);
}
bool valid(unsigned r,unsigned h,unsigned heads){return r>0&&r<=32768&&h>=128&&h<=16384&&h%128==0&&heads>0&&heads<=64&&heads%8==0;}
}
cudaError_t prepare_kda(){return Projection<Row>::sufficient()>0&&Projection<Col>::sufficient()>0?cudaSuccess:cudaErrorInvalidConfiguration;}
std::size_t kda_workspace_bytes(unsigned r,unsigned h,unsigned heads){return valid(r,h,heads)?Arena(nullptr,r,h,heads).size:0;}
cudaError_t enqueue_kda(const KdaCall& v,void* workspace,std::size_t bytes,cudaStream_t stream,bool ref,const KdaCheckpoint* cp) {
    const auto& w=v.w;unsigned r=v.rows,c=w.heads*D;
    if(!valid(r,w.hidden,w.heads)||!workspace||(reinterpret_cast<std::uintptr_t>(workspace)&255)||!v.x||!v.y||!v.state||!v.conv||!w.b||!w.fa||!w.fb||!w.ga||!w.gb||!w.conv||!w.norm||!w.a_log||!w.dt)return cudaErrorInvalidValue;
    for(auto m:{w.qkv,w.out})if(!m.packed||!m.suh||!m.svh||m.bits<3||m.bits>6)return cudaErrorInvalidValue;
    if(w.qkv.input!=w.hidden||w.qkv.output!=3*c||w.out.input!=c||w.out.output!=w.hidden)return cudaErrorInvalidValue;
    Arena a(workspace,r,w.hidden,w.heads);if(bytes<a.size)return cudaErrorInvalidValue;
    if(cp&&(cp->row%B||cp->row>=r||!cp->state||!cp->conv))return cudaErrorInvalidValue;
    auto s=stream;cudaError_t e;
    const bool rq=v.row_kernels&&!ref&&row_project_fits(w.qkv,r),ro=v.row_kernels&&!ref&&row_project_fits(w.out,r);
    if(!rq)layer_decode<<<w.qkv.input/16*(w.qkv.output/16),256,0,s>>>(w.qkv,a.decoded);
    const bool r2=v.row_r2;
    if((e=project(w.qkv,v.x,a.qkv,r,a,s,ref,false,rq,r2))!=cudaSuccess)return e;
    if((e=ordinary(v.x,w.fa,a.lo,r,w.hidden,D,a,s,ref,false,ro&&r2))!=cudaSuccess)return e;
    if((e=ordinary(v.x,w.ga,a.hi,r,w.hidden,D,a,s,ref,false,ro&&r2))!=cudaSuccess)return e;
    if((e=ordinary(a.lo,w.fb,a.f,r,D,c,a,s,ref,false,ro&&r2))!=cudaSuccess)return e;
    if((e=ordinary(a.hi,w.gb,a.g,r,D,c,a,s,ref,false,ro&&r2))!=cudaSuccess)return e;
    if((e=ordinary(v.x,w.b,a.beta,r,w.hidden,w.heads,a,s,ref,false,ro&&r2))!=cudaSuccess)return e;
    prepare<<<dim3(w.heads,r),D,0,s>>>(w,a.qkv,a.f,a.beta,v.conv,a.prep);
    if(cp)snapshot_conv<<<(3*c+255)/256,256,0,s>>>(a.qkv,v.conv,cp->conv,cp->row,3*c);
    commit_conv<<<(3*c+255)/256,256,0,s>>>(a.qkv,v.conv,r,3*c);
    auto checkpoint=[&](unsigned first){return cp&&first==cp->row?cudaMemcpyAsync(cp->state,v.state,std::size_t(w.heads)*D*D*4,cudaMemcpyDeviceToDevice,s):cudaSuccess;};
    if(ref)for(unsigned first=0;first<r;first+=B){if((e=checkpoint(first))!=cudaSuccess)return e;sequential<<<w.heads,D,0,s>>>(a.prep,a.beta,v.state,a.o,std::min(B,r-first),w.heads,first);}
    else for(unsigned first=0;first<r;first+=B){unsigned n=std::min(B,r-first);if((e=checkpoint(first))!=cudaSuccess)return e;
        prefixes<<<w.heads,D,0,s>>>(a.prep,a.prefix,w.heads,first,n);
        triangular<<<dim3(w.heads,n,2),32,0,s>>>(a.prep,a.prefix,a.ck,a.cq,w.heads,first,n);
        base_products<<<dim3(w.heads,4,8),32,0,s>>>(a.prep,a.prefix,v.state,a.bk,a.bq,w.heads,first,n);
        solve<<<w.heads,D,0,s>>>(a.prep,a.beta,a.ck,a.bk,a.delta,w.heads,first,n);
        block_products<false><<<dim3(w.heads,2,8),32,0,s>>>(a.prep,a.prefix,a.cq,a.bq,a.delta,v.state,a.o,w.heads,first,n);
        block_products<true><<<dim3(w.heads,8,8),32,0,s>>>(a.prep,a.prefix,a.cq,a.bq,a.delta,v.state,a.o,w.heads,first,n);
    }
    output_gate<<<dim3(w.heads,r),D,0,s>>>(w,a.o,a.g,a.z);
    if(!ro)layer_decode<<<w.out.input/16*(w.out.output/16),256,0,s>>>(w.out,a.decoded);
    return project(w.out,a.z,v.y,r,a,s,ref,false,ro,r2);
}
}
