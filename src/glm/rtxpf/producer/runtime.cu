// CUDA port of project-owned node mHC kernels; retain ascending chains and FP32 nonlinear arithmetic.
#include "../kda.cuh"
#include "../mtp.cuh"
#include "../head.cuh"
#include "session.hpp"
#include "expert_tier.hpp"
#include "tierstream_binding.hpp"
#include "unit_file.hpp"
#include "unit_auth.hpp"
#include "head_select.hpp"
#include "runtime.hpp"
#include "decode_timeline.hpp"
#include <cuda_profiler_api.h>
#include "policy.hpp"
#include "state_slices.hpp"
#include "strix/serve/json.hpp"
#include "strix/model_content_hash.hpp"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <fstream>
#include <map>
#include <future>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <chrono>
#include <iostream>
#include "shards.hpp"
#include "schedule.hpp"
#include "verified_io.hpp"

namespace strix::glm::rtxpf::producer {
namespace {
constexpr unsigned kHc=4,kHidden=4096,kHcMix=24,kHcIters=20;
constexpr float kEps=1.e-5f,kHcEps=1.e-6f;
using strix::serve::Json;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
__device__ float bf16_to_float(U16 v){return __uint_as_float(U32(v)<<16);}
__device__ float inv_sqrt(float x){return 1.f/sqrtf(x);}
__device__ float sigmoid(float x){return 1.f/(1.f+declared_exp(-x));}
__device__ U32 ordered(float v){U32 bits=__float_as_uint(v);if((bits&0x7fffffffu)==0)bits=0;return bits&0x80000000u?~bits:bits^0x80000000u;}
__device__ void flag_if(unsigned* f,float x){if(f&&!isfinite(x))atomicOr(f,1u);}
template<unsigned L> __device__ float lds_tree(float* s,float a){unsigned l=threadIdx.x;s[l]=a;__syncthreads();for(unsigned h=L/2;h;h/=2){if(l<h)s[l]+=s[l+h];__syncthreads();}float v=s[0];__syncthreads();return v;}
// The mix (lane 0, R6 in the host hc_mix order) and the collapsed row, shared by both hc_pre kernels.
__device__ void hc_mix_finish(const float* z, float* m, const float* base, const float* scale, const float* Xr, float* mix, float* xc,
                              unsigned r, unsigned l) {
    if (l == 0) {  // R6 on lane 0 (host hc_mix order)
        for (unsigned j = 0; j < kHc; ++j) m[j] = sigmoid(z[j] * scale[0] + base[j]) + kHcEps;
        for (unsigned j = 0; j < kHc; ++j) m[kHc + j] = 2.0F * sigmoid(z[kHc + j] * scale[1] + base[kHc + j]);
        float* c = m + 2 * kHc;
        for (unsigned k = 0; k < kHc * kHc; ++k) c[k] = z[2 * kHc + k] * scale[2] + base[2 * kHc + k];
        for (unsigned j = 0; j < kHc; ++j) {
            float mx = c[j * kHc];
            for (unsigned k = 1; k < kHc; ++k) mx = c[j * kHc + k] > mx ? c[j * kHc + k] : mx;
            float e[kHc];
            for (unsigned k = 0; k < kHc; ++k) e[k] = declared_exp(c[j * kHc + k] - mx);
            const float sum = ((e[0] + e[1]) + e[2]) + e[3];
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = e[k] / sum + kHcEps;
        }
        for (unsigned it = 0; it < kHcIters; ++it) {
            if (it > 0) {  // rows
                float sr[kHc];
                for (unsigned j = 0; j < kHc; ++j)
                    sr[j] = (((c[j * kHc + 0] + c[j * kHc + 1]) + c[j * kHc + 2]) + c[j * kHc + 3]) + kHcEps;
                for (unsigned j = 0; j < kHc; ++j)
                    for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sr[j];
            }
            float sc[kHc];  // columns
            for (unsigned k = 0; k < kHc; ++k)
                sc[k] = (((c[0 * kHc + k] + c[1 * kHc + k]) + c[2 * kHc + k]) + c[3 * kHc + k]) + kHcEps;
            for (unsigned j = 0; j < kHc; ++j)
                for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sc[k];
        }
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(r) * kHcMix + k] = m[k];
    }
    __syncthreads();
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
}
__global__ __launch_bounds__(256) void hc_pre_kernel(const float* X, const float* fn, const float* base,
                                                     const float* scale, float* F, float* mix, float* xc) {
    constexpr unsigned n = kHc * kHidden;
    __shared__ float s[256];
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Fr = F + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __fmaf_rn(Xr[i], Xr[i], a);
    const float ss = lds_tree<256>(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(n) + kEps);
    for (unsigned i = l; i < n; i += 256) Fr[i] = Xr[i] * rr;
    __syncthreads();
    if (l < kHcMix) {  // 24 chains over the normalized streams (each lane reads the F values it did not write: the
                       // __syncthreads above orders global writes within the block)
        float acc = 0.0F;
        const float* row = fn + static_cast<std::size_t>(l) * n;
        for (unsigned i = 0; i < n; ++i) acc = __fmaf_rn(Fr[i], row[i], acc);
        z[l] = acc;
    }
    __syncthreads();
    hc_mix_finish(z, m, base, scale, Xr, mix, xc, r, l);
}

// hc_pre_kernel with the 24 chains fed from shared memory: all 256 threads stage F (recomputed as
// Xr[i]*rr, the value F holds) and the 24 rows in 256-column chunks, the next chunk in registers
// while lanes 0..23 run the same ascending __fmaf_rn chains. Every value is hc_pre_kernel's. V4 (row-kernels
// round 2, and the 2-4 row verify): rows padded to 260 floats, four operands per shared-memory read.
template <bool V4>
__global__ __launch_bounds__(256) void hc_pre_staged(const float* X, const float* fn, const float* base,
                                                     const float* scale, float* F, float* mix, float* xc) {
    constexpr unsigned n = kHc * kHidden, C = 256;
    __shared__ float s[256];
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    __shared__ __align__(16) float sf[C];
    __shared__ __align__(16) float srow[kHcMix][V4 ? C + 4 : C + 1];  // V4 stride 260: 16-byte reads, conflict-free per quarter warp
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Fr = F + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __fmaf_rn(Xr[i], Xr[i], a);
    const float ss = lds_tree<256>(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(n) + kEps);
    for (unsigned i = l; i < n; i += 256) Fr[i] = Xr[i] * rr;
    float pf = Xr[l] * rr, pr[kHcMix];
#pragma unroll
    for (unsigned q = 0; q < kHcMix; ++q) pr[q] = fn[static_cast<std::size_t>(q) * n + l];
    float acc = 0.0F;
    for (unsigned c = 0; c < n; c += C) {
        sf[l] = pf;
#pragma unroll
        for (unsigned q = 0; q < kHcMix; ++q) srow[q][l] = pr[q];
        __syncthreads();
        if (c + C < n) {
            pf = Xr[c + C + l] * rr;
#pragma unroll
            for (unsigned q = 0; q < kHcMix; ++q) pr[q] = fn[static_cast<std::size_t>(q) * n + c + C + l];
        }
        if (l < kHcMix) {  // the same ascending chain
            if constexpr (V4) {
                const float4* f4 = reinterpret_cast<const float4*>(sf);
                const float4* w4 = reinterpret_cast<const float4*>(srow[l]);
#pragma unroll 8
                for (unsigned i = 0; i < C / 4; ++i) {
                    const float4 f = f4[i], w = w4[i];
                    acc = __fmaf_rn(f.x, w.x, acc); acc = __fmaf_rn(f.y, w.y, acc);
                    acc = __fmaf_rn(f.z, w.z, acc); acc = __fmaf_rn(f.w, w.w, acc);
                }
            } else {
#pragma unroll 16
                for (unsigned i = 0; i < C; ++i) acc = __fmaf_rn(sf[i], srow[l][i], acc);
            }
        }
        __syncthreads();
    }
    if (l < kHcMix) z[l] = acc;
    __syncthreads();
    hc_mix_finish(z, m, base, scale, Xr, mix, xc, r, l);
}

__global__ void hc_expand_kernel(const float* X, const float* mix, const float* y, float* Xo, unsigned* flag, unsigned flag_stride) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    if (flag) flag += static_cast<std::size_t>(r) * flag_stride;  // stride 0: one flag for the call; 1: per-row flags (spec)
    const float* m = mix + static_cast<std::size_t>(r) * kHcMix;
    const float* c = m + 2 * kHc;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    const float yd = y[static_cast<std::size_t>(r) * kHidden + d];
    for (unsigned i = 0; i < kHc; ++i) {
        const float mx = ((c[0 * kHc + i] * Xr[d] + c[1 * kHc + i] * Xr[kHidden + d]) + c[2 * kHc + i] * Xr[2 * kHidden + d]) +
                         c[3 * kHc + i] * Xr[3 * kHidden + d];
        const float v = m[kHc + i] * yd + mx;
        flag_if(flag, v);
        Xo[static_cast<std::size_t>(r) * kHc * kHidden + i * kHidden + d] = v;
    }
}

__global__ void hc_final_kernel(const float* X, float* h) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    h[static_cast<std::size_t>(r) * kHidden + d] = (((Xr[d] + Xr[kHidden + d]) + Xr[2 * kHidden + d]) + Xr[3 * kHidden + d]) * 0.25F;
}


__global__ void norm(const float* x,const U16* w,float* y,unsigned rows){
    __shared__ float s[256];unsigned r=blockIdx.x,l=threadIdx.x;float a=0;
    for(unsigned d=l;d<4096;d+=256){float v=x[std::size_t(r)*4096+d];a=__fmaf_rn(v,v,a);}
    float rr=inv_sqrt(lds_tree<256>(s,a)/4096+1.e-5f);
    for(unsigned d=l;d<4096;d+=256)y[std::size_t(r)*4096+d]=(x[std::size_t(r)*4096+d]*rr)*bf16_to_float(w[d]);
}
__global__ void embed(const U16* e,const U32* ids,float* x,unsigned rows,unsigned streams){
    auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j>=std::size_t(rows)*streams*4096)return;
    unsigned r=j/(streams*4096),d=j%4096;x[j]=bf16_to_float(e[std::size_t(ids[r])*4096+d]);
}
__global__ void add(const float* a,const float* b,float* y,std::size_t n,unsigned* flag){auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j<n){float v=a[j]+b[j];flag_if(flag,v);y[j]=v;}}
__global__ void router(const float* x,const U16* w,const float* bias,U32* ids,float* rates,unsigned* flag){
    __shared__ float sg[288];__shared__ U32 keys[288];unsigned r=blockIdx.x,e=threadIdx.x;
    if(e<288){float v=0;for(unsigned k=0;k<4096;++k)v=__fmaf_rn(x[std::size_t(r)*4096+k],hf(w[std::size_t(e)*4096+k]),v);flag_if(flag,v);sg[e]=sigmoid(v);keys[e]=ordered(sg[e]+bias[e]);}__syncthreads();if(e)return;
    bool used[288]{};unsigned pick[8];for(unsigned j=0;j<8;++j){unsigned best=288;for(unsigned k=0;k<288;++k)if(!used[k]&&(best==288||keys[k]>keys[best]))best=k;used[best]=true;pick[j]=best;}
    for(unsigned j=1;j<8;++j)for(unsigned k=j;k&&pick[k-1]>pick[k];--k){unsigned t=pick[k-1];pick[k-1]=pick[k];pick[k]=t;}
    float den=0;for(unsigned j=0;j<8;++j)den+=sg[pick[j]];den+=1.e-20f;
    for(unsigned j=0;j<8;++j){ids[std::size_t(r)*8+j]=pick[j];rates[std::size_t(r)*8+j]=sg[pick[j]]/den*2.5f;}
}
// router, same values per row (blockIdx.y; rows <= 4): 9 CTAs of 32 experts; each expert's ascending __fmaf_rn chain over
// 4096 inputs reads 128-column chunks staged in shared memory (the next chunk in registers), then
// router_select picks the same top 8 as router's greedy scan + index sort (the eight largest keys,
// ties to the lower index, ascending) and normalizes them in the same order.
__global__ __launch_bounds__(128) void router_scores(const float* x,const U16* w,const float* bias,float* sg,U32* keys,unsigned* flag){
    constexpr unsigned C=128,E=32,W=C/2+1; // row stride W words: lane e reads its row conflict-free
    __shared__ U32 sw[E*W];__shared__ float sx[C];
    const unsigned t=threadIdx.x,e0=blockIdx.x*E,r=blockIdx.y;x+=std::size_t(r)*4096;sg+=std::size_t(r)*288;keys+=std::size_t(r)*288;
    const U32* rows=reinterpret_cast<const U32*>(w+std::size_t(e0)*4096);
    U32 pw[16];float px=x[t];
    #pragma unroll
    for(unsigned q=0;q<16;++q){unsigned j=t+128*q;pw[q]=rows[std::size_t(j/64)*2048+j%64];}
    float v=0;
    for(unsigned c=0;c<4096;c+=C){
        #pragma unroll
        for(unsigned q=0;q<16;++q){unsigned j=t+128*q;sw[j/64*W+j%64]=pw[q];}
        sx[t]=px;__syncthreads();
        if(c+C<4096){px=x[c+C+t];
            #pragma unroll
            for(unsigned q=0;q<16;++q){unsigned j=t+128*q;pw[q]=rows[std::size_t(j/64)*2048+(c+C)/2+j%64];}}
        if(t<E){const U16* wr=reinterpret_cast<const U16*>(sw+t*W);
            #pragma unroll 16
            for(unsigned k=0;k<C;++k)v=__fmaf_rn(sx[k],hf(wr[k]),v);}
        __syncthreads();
    }
    if(t<E){const unsigned e=e0+t;flag_if(flag,v);const float g=sigmoid(v);sg[e]=g;keys[e]=ordered(g+bias[e]);}
}
__global__ __launch_bounds__(288) void router_select(const float* sg,const U32* keys,U32* ids,float* rates){
    __shared__ U32 k[288];__shared__ unsigned chosen[288];__shared__ unsigned pick[8];
    const unsigned e=threadIdx.x,r=blockIdx.x;sg+=std::size_t(r)*288;keys+=std::size_t(r)*288;ids+=std::size_t(r)*8;rates+=std::size_t(r)*8;k[e]=keys[e];__syncthreads();
    unsigned rank=0;for(unsigned j=0;j<288;++j)rank+=k[j]>k[e]||(k[j]==k[e]&&j<e);
    chosen[e]=rank<8;__syncthreads();
    if(chosen[e]){unsigned pos=0;for(unsigned j=0;j<e;++j)pos+=chosen[j];pick[pos]=e;}
    __syncthreads();if(e)return;
    float den=0;for(unsigned j=0;j<8;++j)den+=sg[pick[j]];den+=1.e-20f;
    for(unsigned j=0;j<8;++j){ids[j]=pick[j];rates[j]=sg[pick[j]]/den*2.5f;}
}
__global__ void check_finite(const float* x,std::size_t n,unsigned* flag){auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j<n)flag_if(flag,x[j]);}
// Tierstream adapter: guarded slot -> GLM descriptor for each selected ID,
// indexed by original expert ID. Unselected entries stay zero (never read).
// Census (diagnostics, read-only): the layer's exported ids absent from the slot table at
// export time (its decode misses), and per gated layer the table's homes.
__global__ void count_absent(TierView v,unsigned g,const U32* ids,unsigned n,U32* misses){
    if(threadIdx.x==0){const volatile std::int32_t* t=v.slot_of+v.unit_base[g];unsigned c=0;for(unsigned j=0;j<n;++j)c+=t[ids[j]]<0?1u:0u;misses[g]=c;}}
__global__ void slot_census(TierView v,unsigned persistent,U32* out){
    const unsigned g=blockIdx.x;if(g>=v.n_gated)return;const volatile std::int32_t* t=v.slot_of+v.unit_base[g];unsigned a=0,b=0,c=0;
    for(unsigned u=threadIdx.x;u<v.units;u+=blockDim.x){const std::int32_t s=t[u];if(s<0)++c;else if(unsigned(s)<persistent)++a;else ++b;}
    atomicAdd(out+3*g,a);atomicAdd(out+3*g+1,b);atomicAdd(out+3*g+2,c);}
__global__ void bind_units(TierView v,unsigned g,const U32* ids,unsigned n,const UnitOffsets* offsets,Expert* out){
    unsigned j=threadIdx.x;if(j>=n)return;U32 id=ids[j];bool absent=false;
    auto slot=guarded_slot(v.slot_of,v.unit_base,v.abort?*v.abort:0u,v.guard_slot,g,id,&absent);
    if(absent)atomicOr(v.absent_err,1u);if(id>=v.units)return;
    const auto& u=offsets[std::size_t(g)*v.units+id];Matrix m[3];
    const auto b=reinterpret_cast<std::uint64_t>(v.base);auto at=[&](unsigned k,unsigned r){return reinterpret_cast<const U16*>(unit_address(b,slot,v.stride,u,k,r));};
    for(unsigned k=0;k<3;++k)m[k]={at(k,0),at(k,1),at(k,2),k<2?4096u:2048u,k<2?2048u:4096u,u.rate[k]};
    out[id]={m[0],m[1],m[2]};
}
std::string hex_sha(const void* p,std::size_t n){return strix::model_content_identity_hex(strix::sha256_bytes(std::span(static_cast<const std::byte*>(p),n)));}
// Process allocation ledger (tierstream ss_budget_ledger, owned by Runtime::Impl):
// every Buffer charges its device or pinned bytes to the component named by the
// innermost Charge scope, against the whole-device caps; an over-cap allocation
// fails naming the component (an optional allocation just reports false).
BudgetLedger* g_ledger=nullptr;
thread_local const char* g_component="unattributed";
struct Charge {const char* prev;explicit Charge(const char* name):prev(g_component){g_component=name;}~Charge(){g_component=prev;}Charge(const Charge&)=delete;Charge& operator=(const Charge&)=delete;};
// Host commit (expert_tier.hpp host_commit): raw PrivateUsage minus our device
// bytes, which WDDM also charges to the process commit. g_context_bytes is the
// measured context + other processes part charged at startup (not ours).
std::uint64_t g_context_bytes=0;
std::uint64_t process_private_raw(){PROCESS_MEMORY_COUNTERS_EX m{};m.cb=sizeof m;return GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof m)?std::uint64_t(m.PrivateUsage):~0ULL;}
// Only ATTRIBUTABLE device bytes are discounted: exact allocation sizes (our
// Buffers, the library's arena/tables/controls charged by name). The measured
// context + other processes part and the CUDA graph estimates (device-wide free
// deltas, which can include another process's allocation) are excluded: graph
// backing therefore stays on the host side (conservative).
std::uint64_t g_graph_device_bytes=0,g_backing_peak=0;
std::uint64_t attributable_device_bytes(){if(!g_ledger)return 0;const auto d=g_ledger->current(MemKind::device),x=g_context_bytes+g_graph_device_bytes;return d>x?d-x:0;}
HostCommit host_commit_now(std::uint64_t raw){if(raw==~0ULL)return {raw,0,raw};auto c=host_commit(raw,attributable_device_bytes(),0);g_backing_peak=std::max(g_backing_peak,c.device_backing);return c;}
HostCommit process_host_commit(){return host_commit_now(process_private_raw());}
std::uint64_t process_commit_bytes(){return process_host_commit().host;}
struct Buffer {
    void* p=nullptr;std::size_t bytes=0;bool pinned=false;std::string charged;std::uint64_t charged_bytes=0;
    Buffer()=default;Buffer(const Buffer&)=delete;
    ~Buffer(){release();}
    // Frees the allocation and leaves an empty, reusable Buffer (no lifetime games).
    void release(){if(p){if(pinned){cudaHostUnregister(p);VirtualFree(p,0,MEM_RELEASE);}else cudaFree(p);}uncharge();p=nullptr;bytes=0;pinned=false;}
    void uncharge(){if(!charged.empty()&&g_ledger)g_ledger->sub(pinned?MemKind::pinned:MemKind::device,charged,charged_bytes);charged.clear();charged_bytes=0;}
    bool allocate(std::size_t n,bool host=false,bool optional=false){need(!p,"double allocation");pinned=host;bytes=n;if(!n)return true;
        if(g_ledger){std::string why;const std::uint64_t c=host?io_round(n):n;
            if(!g_ledger->add(host?MemKind::pinned:MemKind::device,g_component,c,&why)){bytes=0;pinned=false;if(optional)return false;throw std::runtime_error(why);}
            charged=g_component;charged_bytes=c;}
        cudaError_t e;
        if(host){p=VirtualAlloc(nullptr,io_round(n),MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE);e=p?cudaHostRegister(p,io_round(n),cudaHostRegisterDefault):cudaErrorMemoryAllocation;}
        else e=cudaMalloc(&p,n);
        if(e!=cudaSuccess){cudaGetLastError();if(host&&p)VirtualFree(p,0,MEM_RELEASE);p=nullptr;uncharge();bytes=0;if(!optional)ck(e);return false;}
        observe_allocation();return true;}
    // Allocation-time sample (device-wide use and process commit) at the component's stage.
    static void observe_allocation(const char* stage=nullptr){if(!g_ledger)return;std::size_t f=0,t=0;if(cudaMemGetInfo(&f,&t)!=cudaSuccess){cudaGetLastError();return;}
        g_ledger->observe(stage?stage:g_component,t-f,process_commit_bytes());}
    template<class T>T* as()const{return static_cast<T*>(p);}
    void reset(){release();}
};
// A host temporary (pageable) charged to the ledger for its lifetime against the
// RAM cap (an over-cap temporary fails, naming it); the caller samples while the
// bytes are alive (Buffer::observe_allocation(name)).
struct HostTemp {
    const char* name;std::uint64_t bytes=0;
    HostTemp(const char* n,std::uint64_t b):name(n){if(!g_ledger||!b)return;std::string why;if(!g_ledger->add(MemKind::pageable,name,b,&why))throw std::runtime_error(why);bytes=b;}
    ~HostTemp(){if(bytes&&g_ledger)g_ledger->sub(MemKind::pageable,name,bytes);}
    HostTemp(const HostTemp&)=delete;HostTemp& operator=(const HostTemp&)=delete;
};
struct Stream {cudaStream_t s{};Stream(){ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s)cudaStreamDestroy(s);}};
struct Event {cudaEvent_t e{};Event(){ck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));}~Event(){if(e)cudaEventDestroy(e);}};
std::uint64_t integer(const Json& j){auto n=j.as_int();need(n>=0,"negative cache integer");return std::uint64_t(n);}
const Json& at(const Json& j,const std::string& key){auto* v=j.find(key);need(v!=nullptr,"missing manifest field");return *v;}
std::string str(const Json& j,const std::string& key){return at(j,key).as_string();}
std::pair<std::uint64_t,std::uint64_t> host_free(){MEMORYSTATUSEX m{};m.dwLength=sizeof m;need(GlobalMemoryStatusEx(&m)!=0,"host memory status");return {m.ullAvailPhys,m.ullAvailPageFile};}
struct Hc {const float *fn,*base,*scale;};
struct Layer {
    KdaWeights k{};DsaWeights d{};MtpWeights m{};Hc attn{},ffn{};
    const U16 *input=nullptr,*post=nullptr,*router=nullptr;
    const float* bias=nullptr;
    Expert shared{},dense{};Buffer descriptors,host_descriptors,shared_descriptor;Event ready;
    Buffer fp16_g,fp16_u,fp16_d; // persistent_dense_fp16: this layer's shared (or dense) FP16 B operands
};
struct Blob {std::string name,sha;std::size_t bytes;Buffer device,host;std::unique_ptr<VerifiedFile> file;};
struct Ring {double read_ms=0;Buffer device,host;Event loaded,retired;bool used=false;std::future<void> read;int layer=-1;};
// Two templates per layer: complete row tile and exact (unpadded) tail.
// Rebinding recaptures host launch parameters, then updates the existing graph
// executable. This bounds graph memory independently of prefix length.
// Effective device cap: the whole-device --vram-total-gib when set, else the PRO's 99.86 GB.
std::uint64_t g_device_cap=99860000000ULL;
// An irreversible process-level failure (e.g. a graph executable that could not be
// released): every later graph launch, hence every request, reports it.
std::string g_fatal;
struct GraphTemplate {
    cudaGraphExec_t executable=nullptr;std::string binding;double binding_ms=0;std::uint64_t charged=0;
    static constexpr const char* graph_part="CUDA graph executables (measured at instantiate)";
    // Only a confirmed release drops the ledger charge; a failed one retires the worker.
    // The executable stays OWNED (and charged) until a release is confirmed: a failed
    // release keeps it for the next cleanup attempt and retires the process.
    void destroy()noexcept{if(!executable){release_charge();return;}const auto e=cudaGraphExecDestroy(executable);
        if(e==cudaSuccess){executable=nullptr;release_charge();return;}cudaGetLastError();
        if(g_fatal.empty())g_fatal=std::string("CUDA graph executable release failed (")+cudaGetErrorString(e)+"): it stays owned and charged; worker retired";}
    void release_charge()noexcept{if(charged&&g_ledger)g_ledger->sub(MemKind::device,graph_part,charged);g_graph_device_bytes-=std::min(g_graph_device_bytes,charged);charged=0;}
    ~GraphTemplate(){destroy();}
    static std::uint64_t device_free(){std::size_t f=0,tt=0;ck(cudaMemGetInfo(&f,&tt));return f;}
    // A measured sample on an error path (recorded, never throws).
    static void record(const char* stage)noexcept{std::size_t f=0,tt=0;if(cudaMemGetInfo(&f,&tt)!=cudaSuccess){cudaGetLastError();return;}
        if(g_ledger)try{g_ledger->observe(stage,tt-f,process_commit_bytes());}catch(...){}}
    // A measured sample at a graph stage; the effective device cap is enforced here.
    static void sample(const char* stage){std::size_t f=0,tt=0;ck(cudaMemGetInfo(&f,&tt));const std::uint64_t used=tt-f;
        if(g_ledger)g_ledger->observe(stage,used,process_commit_bytes());
        if(used>g_device_cap)throw std::runtime_error(std::string(stage)+": device use "+std::to_string(used)+" B exceeds the effective cap "+std::to_string(g_device_cap)+" B");}
    // Charge (or release) the net change of this template's persistent device memory.
    void settle(std::uint64_t free0){const std::uint64_t free1=device_free();
        if(free0>free1){const std::uint64_t grow=free0-free1;if(g_ledger){std::string why;if(!g_ledger->add(MemKind::device,graph_part,grow,&why))throw std::runtime_error(why);}charged+=grow;g_graph_device_bytes+=grow;}
        else{const std::uint64_t shrink=std::min<std::uint64_t>(free1-free0,charged);if(shrink&&g_ledger)g_ledger->sub(MemKind::device,graph_part,shrink);charged-=shrink;g_graph_device_bytes-=std::min(g_graph_device_bytes,shrink);}}
    template<class F> void launch(cudaStream_t stream,const std::string& key,F enqueue){
        binding_ms=0;auto begin=std::chrono::steady_clock::now();
        if(!g_fatal.empty())throw std::runtime_error(g_fatal);
        if(!executable||binding!=key){
            // Every operation's result is sampled BEFORE any cleanup (capture, update,
            // instantiate; success or failure) and gated against the effective cap on
            // the success path; the net persistent change is charged to the ledger.
            std::uint64_t free0=device_free();
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            cudaGraph_t graph=nullptr;
            try{enqueue();ck(cudaStreamEndCapture(stream,&graph));}
            catch(...){cudaStreamEndCapture(stream,&graph);record("graph capture failed");if(graph)cudaGraphDestroy(graph);throw;}
            try{
                sample("graph capture");
                bool instantiate=!executable;
                if(executable){cudaGraphExecUpdateResultInfo info{};auto status=cudaGraphExecUpdate(executable,graph,&info);instantiate=status!=cudaSuccess||info.result!=cudaGraphExecUpdateSuccess;
                    if(instantiate){cudaGetLastError();sample("graph update failed"); // gated before the replacement
                        const auto before=device_free();destroy();
                        if(!g_fatal.empty())throw std::runtime_error(g_fatal);free0+=device_free()-before;} // the replaced executable's memory, rebased
                    else sample("graph update");}
                if(instantiate){const auto e=cudaGraphInstantiate(&executable,graph,0);if(e!=cudaSuccess){record("graph instantiate failed");executable=nullptr;ck(e);}sample("graph instantiate");}
                binding=key;cudaGraphDestroy(graph);graph=nullptr;
                settle(free0);
                binding_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            }catch(...){record("graph error");if(graph)cudaGraphDestroy(graph);destroy();binding.clear();throw;}
        }
        ck(cudaGraphLaunch(executable,stream));
    }
};

}
struct Runtime::Impl {
    // Declared first: destroyed last, after every Buffer released its charge.
    struct LedgerOwner {std::unique_ptr<BudgetLedger> l;~LedgerOwner(){if(g_ledger==l.get())g_ledger=nullptr;}} ledger;
    std::filesystem::path cache;std::unique_ptr<Shards> shards;bool graphs;GraphTemplate graph[46][2];std::map<std::string,const Json*> tensor_index;Json manifest;std::string model_hash;unsigned capacity,tile,allocated_tile=0;
    // Per-request prefill tile (schedule.hpp select_prefill_tile): requested_tile 0 = auto (rows vs the crossover);
    // a nonzero request (ttft row_tile, at most the allocation) is used as is; the self-check pins its own tiles.
    unsigned requested_tile=0,tile_crossover=default_prefill_tile_crossover;std::string tile_choice="allocated (startup)";bool pin_tile=false;
    void choose_tile(std::size_t tokens){if(pin_tile||tokens<2||!allocated_tile)return; /* before the allocation is recorded: the startup tile */const unsigned P=unsigned(tokens-1);
        if(requested_tile){tile=requested_tile;tile_choice="request override (row_tile)";}else{tile=select_prefill_tile(P,allocated_tile,tile_crossover);tile_choice=prefill_tile_reason(P,tile_crossover);}}
    void record_prefill_cache(){if(!prefill_via_cache||!ts_tier)return;last_timings.set("prefill_cache",counter_delta(prefill_counters0,cache_counters()));
        auto a=Json::array();double total=0;for(double x:ts_tier->acquire_ms){a.push(x);total+=x;}last_timings.set("make_resident_ms_per_layer",a);last_timings.set("make_resident_ms_total",total);
        last_timings.set("prefill_overlap",ts_tier->overlap_note);
        if(ts_tier->overlapping)for(auto [k,v]:{std::pair<const char*,const std::vector<double>*>{"prefill_overlap_gpu_idle_ms",&ts_tier->gpu_idle_ms},{"prefill_overlap_host_wait_ms",&ts_tier->host_wait_ms}}){
            auto w=Json::array();double wt=0;for(double x:*v){w.push(x);wt+=x;}last_timings.set(std::string(k)+"_per_layer",w);last_timings.set(std::string(k)+"_total",wt);}
        last_timings.set("decode_restore",restore_pending?"deferred to the next decode token":"off");}
    void record_tile(){last_timings.set("row_tile",int(tile));last_timings.set("allocated_row_tile",int(allocated_tile));last_timings.set("row_tile_choice",tile_choice);last_timings.set("prefill_tile_crossover",int(tile_crossover));}
    Stream compute,h2d,decode;CutlassExecution exec{};std::vector<cudaEvent_t> exec_events;
    std::map<std::string,std::unique_ptr<Blob>> blobs;std::vector<std::unique_ptr<Layer>> layers;
    Ring ring[2];unsigned device_count=0,host_count=0,file_count=0;bool staged[46]{};bool serial_stage=false,legacy_index=false,require_file_tier=false,both_file_rings=false;unsigned file_ring_mask=0;std::uint64_t file_ring_bytes[2]{};
    Buffer X,X2,F,mix,collapsed,normalized,part,routed,shared,hidden,embedding,tokens,ids,rates,workspace,flag;
    Buffer state,conv,latent,key,gate,pooled;std::size_t ws_bytes=0;Json profile;
    std::chrono::steady_clock::time_point request_start;Json releases=Json::array(),last_timings=Json::object();double staging_ms=0,file_read_ms=0,source_load_ms=0,graph_binding_ms=0;std::uint64_t file_bytes=0,source_load_bytes=0;
    Impl(const std::string& path,unsigned cap,unsigned rows,bool from_shards,bool use_graphs,unsigned floor_gib,const VerifiedManifest* verified,unsigned max_device,unsigned max_pinned,bool require_file,bool decode_on=false,const DecodeOptions& dopts={}):cache(path),graphs(use_graphs),capacity(cap),tile(effective_row_tile(cap,rows)),decode_mode(decode_on),sess(cap){
        const unsigned requested_rows=rows;rows=tile;
        need(max_device<=43&&max_pinned<=43,"layer limits must be 0..43");
        // Tierstream mode (stopgap): the decode cache owns the whole expert budget
        // (--tier-vram-gib / --tier-ram-gib); prefill keeps NO resident or pinned
        // expert copies and streams every routed layer through its two rings.
        const bool ts_mode=decode_on&&dopts.expert_tier=="tierstream";
        whole_device=dopts.vram_total_gib>=0||dopts.ram_total_gib>=0;
        need(!whole_device||ts_mode,"--vram-total-gib/--ram-total-gib apply to the tierstream decode service");
        prefill_via_cache=ts_mode&&tierstream_supports_make_resident(); // else the interim prefill rings
        if(budget_library_available()){ledger.l=std::make_unique<BudgetLedger>(gib_bytes(dopts.vram_total_gib),gib_bytes(dopts.ram_total_gib));g_ledger=ledger.l.get();}
        g_device_cap=dopts.vram_total_gib>=0?std::min<std::uint64_t>(gib_bytes(dopts.vram_total_gib),99860000000ULL):99860000000ULL;
        requested_device_layers=max_device;requested_pinned_layers=max_pinned;if(ts_mode){max_device=0;max_pinned=0;}
        require_file_tier=require_file||caps_require_file_tier(cap,max_device,max_pinned);both_file_rings=require_file;
        need(valid_schedule(cap,rows),"capacity 4-aligned <=131072; tile 32-aligned <=min(capacity,32768)");
        // Retain deny-write/delete handles before verification and all reads.
        // Persistent hits require the complete filesystem identity, not size.
        auto source_begin=std::chrono::steady_clock::now();
        if(from_shards){shards=std::make_unique<Shards>(path,!ts_mode,decode_on);manifest=shards->manifest;}else {need(verified!=nullptr,"cache requires held verified manifest");manifest=verified->parsed;}
        if(ts_mode){
            // Whole-device budget from sizes only (tierstream ss_budget_plan), BEFORE
            // source hashing, unit-file authentication and every load: a bad budget
            // fails in seconds naming the parts and the smallest caps that fit.
            check_device();std::size_t free0=0,total0=0;ck(cudaMemGetInfo(&free0,&total0));
            const std::uint64_t used0=total0-free0;auto [physical0,commit0]=host_free();
            // Without --vram-total-gib (also with only --ram-total-gib) the usable device is the VRAM cap.
            const std::uint64_t hard=std::min<std::uint64_t>(total0,99860000000ULL);
            auto tp=plan_tile(manifest,cap,requested_rows,dopts,used0,host_budget(physical0,commit0,floor_gib),process_host_bytes(),!prefill_via_cache,hard);
            // persistent_dense_fp16 is dropped (and recorded) when the plan cannot afford its copies.
            // It is also dropped when its copies would push expert units to the file tier that the
            // plan without them keeps in device or pinned memory (decode misses would then read files).
            if(dopts.persistent_dense_fp16&&(!tp.refused.empty()||tp.plan.tier.disk>0)){auto d2=dopts;d2.persistent_dense_fp16=false;
                auto tp2=plan_tile(manifest,cap,requested_rows,d2,used0,host_budget(physical0,commit0,floor_gib),process_host_bytes(),!prefill_via_cache,hard);
                if(tp2.refused.empty()&&(!tp.refused.empty()||tp2.plan.tier.disk<tp.plan.tier.disk)){persist_refused=true;
                    persist_note=!tp.refused.empty()?"off: the whole-device plan cannot afford the persistent FP16 copies ("+tp.refused+")"
                        :"off: the persistent FP16 copies would move "+std::to_string(tp.plan.tier.disk-tp2.plan.tier.disk)+" expert units to the file tier";
                    tp=std::move(tp2);}}
            spares_reduced=tp.spares_reduced;planned_spare_slots=tp.spare_slots;tile=rows=tp.tile;budget_in=tp.in;const auto& planned=tp.plan;startup_budget=tp.json;
            need(valid_schedule(cap,rows),"capacity 4-aligned <=131072; tile 32-aligned <=min(capacity,32768)");
            startup_budget.set("mode",whole_device?"whole-device caps (--vram-total-gib/--ram-total-gib)":"device room (no whole-device cap)");
            std::cerr<<"INFO decode budget plan: "<<startup_budget.dump()<<"\n";
            const auto& refused=tp.refused;(void)planned;
            need(refused.empty(),("decode budget refused before startup: "+refused).c_str());
            if(ledger.l){ledger.l->add(MemKind::device,context_part,used0);g_context_bytes=used0;ledger.l->add(MemKind::pageable,process_part,budget_in.pageable.front().bytes);ledger.l->observe("startup",used0,process_host_bytes());}
            if(shards){shards->authenticate_sources();manifest=shards->manifest;}
        }
        need(str(manifest,"format")=="glm-native-cache-v1"&&str(manifest,"layout")=="k16-n16-msb-mul1"&&str(manifest,"decoder")=="mul1-8tiles-v2"&&str(manifest,"tile")=="cutlass-c1-16experts-v1"&&str(manifest,"transform")=="h128-rn16-v1","cache version mismatch");
        model_hash=str(manifest,"model_sha256");need(model_hash.size()==64,"model hash");
        auto key_contract=Json::object();const bool head=manifest.find("head")!=nullptr;
        if(head)need(str(manifest,"head")==head_contract,"cache head contract");
        need(!decode_on||head,"decode requires a head-capable cache/inventory (lm_head); this one is prefill-only");
        // Alphabetical (canonical) order; "head" exists only for decode-capable inventories.
        for(auto name:{"decoder","format","head","layout","model_sha256","tile","transform"})if(std::string(name)!="head"||head)key_contract.set(name,at(manifest,name));
        auto key_text=key_contract.dump();need(strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(key_text))))==str(manifest,"cache_key"),"cache key mismatch");
        auto source_text=at(manifest,"sources").dump();need(strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(source_text))))==model_hash,"source manifest identity mismatch");
        for(const auto& [name,j]:at(manifest,"blobs").as_object()){
            need(std::filesystem::path(name).filename()==name,"cache path traversal");auto b=std::make_unique<Blob>();b->name=name;b->sha=str(j,"sha256");b->bytes=integer(at(j,"bytes"));
            if(!shards)b->file=std::make_unique<VerifiedFile>(cache/name,b->bytes);blobs.emplace(name,std::move(b));
        }
        need(blobs.size()==(head?45u:44u)&&blobs.contains("base.bin")&&(!head||blobs.contains("head.bin")),"full-model cache requires base +43 expert layers (+head)");
        std::atomic<unsigned> hash_hits=shards?shards->verified_hash_hits:0;
        if(!shards){std::vector<Blob*> jobs;for(auto& [name,b]:blobs)jobs.push_back(b.get());auto hash_cache=default_hash_cache();
            parallel_hash_files(jobs.size(),[&](std::size_t j){auto& b=*jobs[j];if(b.file->verify(b.sha,hash_cache))++hash_hits;});}
        auto source_inventory_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-source_begin).count();
        for(const auto& [name,t]:at(manifest,"tensors").as_object()){
            tensor_index.emplace(name,&t);auto bi=blobs.find(str(t,"blob"));need(bi!=blobs.end(),"tensor blob");auto off=integer(at(t,"offset")),n=integer(at(t,"bytes"));need(off%256==0&&off<=bi->second->bytes&&n<=bi->second->bytes-off,"tensor bounds/alignment");
        }
        check_device();
        ws_bytes=workspace_for(rows,cap);
        need(ws_bytes>0,"workspace shape");
        std::size_t largest=0;for(auto& [name,b]:blobs)if(name!="base.bin"&&name!="head.bin")largest=std::max(largest,b->bytes);
        if(decode_on)catalog=std::make_unique<ExpertCatalog>(at(manifest,"tensors"));
        const std::uint64_t decode_fixed=decode_on?decode_device_bytes(cap)+blobs.at("head.bin")->bytes:0;
        std::size_t initial_free,initial_total;ck(cudaMemGetInfo(&initial_free,&initial_total));
        const auto fixed=prefill_fixed_bytes(cap,tile,ws_bytes,blobs.at("base.bin")->bytes,largest,decode_fixed);
        auto existing=initial_total-initial_free;
        need(admit(fixed,initial_free,2ULL<<30)&&existing+fixed+(2ULL<<30)<=99860000000ULL,"fixed allocations exceed free VRAM/99.86 GB cap");
        auto [physical,commit]=host_free();need(2*io_round(largest)+(16ULL<<20)<=host_budget(physical,commit,floor_gib),"insufficient physical RAM/commit for bounded pinned rings");
        // Only residual streams and final hidden span the prefix; attention,
        // HC, FFN dispatch and MTP scratch use the declared row tile.
        auto full=std::size_t(cap)*4096*4,row_bytes=std::size_t(tile)*4096*4;
        {Charge c(activation_part);X.allocate(full*4);X2.allocate(row_bytes*4);F.allocate(row_bytes*4);mix.allocate(std::size_t(tile)*24*4);
        collapsed.allocate(row_bytes);normalized.allocate(row_bytes);part.allocate(row_bytes);routed.allocate(row_bytes);shared.allocate(row_bytes);hidden.allocate(full);embedding.allocate(row_bytes);
        tokens.allocate(std::size_t(cap+1)*4);ids.allocate(std::size_t(tile)*8*4);rates.allocate(std::size_t(tile)*8*4);flag.allocate(4);}
        {Charge c(attention_part);state.allocate(64*128*128*4);conv.allocate(3*64*128*3*4);latent.allocate(std::size_t(cap)*1024);key.allocate(std::size_t(cap)*256);gate.allocate(std::size_t(cap)*256);pooled.allocate(std::size_t(cap/4)*512);}
        {Charge c(workspace_part);workspace.allocate(ws_bytes);}
        auto& base=*blobs.at("base.bin");{Charge c(trunk_part);base.device.allocate(base.bytes);}
        // Bounded scratch transfer for nonexpert startup, using the one H2D stream.
        Buffer staging;{Charge c(staging_part);staging.allocate(8<<20,true);}
        if(shards){auto begin=std::chrono::steady_clock::now();shards->stream("base.bin",staging.p,staging.bytes,[&](std::uint64_t off,const void* p,std::size_t n){ck(cudaMemcpyAsync(base.device.as<char>()+off,p,n,cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));});source_load_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();source_load_bytes+=base.bytes;}
        else {for(std::size_t off=0;off<base.bytes;){auto n=std::min<std::size_t>(staging.bytes,base.bytes-off);base.file->read_at(off,staging.p,n);ck(cudaMemcpyAsync(base.device.as<char>()+off,staging.p,n,cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));off+=n;}}
        // Decode allocations precede expert admission: the device budget below
        // is measured after them (session, replay, head, bounded expert slots).
        dopt=dopts;if(persist_refused)dopt.persistent_dense_fp16=false;if(spares_reduced){dopt.tier_spare_permille=0;if(planned_spare_slots)dopt.tier_spare_slots=planned_spare_slots;}if(decode_on)prepare_decode(staging,dopt.expert_tier,floor_gib);
        const auto init=prefill_init(prefill_via_cache);
        if(init.rings){Charge c(ring_part);for(auto& r:ring){r.device.allocate(largest);r.host.allocate(largest,true);}}
        Charge layer_charge("prefill expert layers (resident/pinned) + descriptors");
        auto target=targets(cap);std::size_t free,total;ck(cudaMemGetInfo(&free,&total));
        std::uint64_t used=total-free,cap_bytes=99860000000ULL;
        auto device_budget=std::min<std::uint64_t>(free,used<cap_bytes?cap_bytes-used:0);
        for(unsigned l=3;l<=45&&init.admit_layers;++l){auto& b=*blobs.at(expert_name(l));bool admitted=false;
            if(device_count<std::min(target.device,max_device)&&admit(b.bytes,device_budget,2ULL<<30)&&b.device.allocate(b.bytes,false,true)){
                load(b,ring[0].host.p);ck(cudaMemcpyAsync(b.device.p,ring[0].host.p,b.bytes,cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));device_budget-=b.bytes;++device_count;admitted=true;
            }
            if(!admitted){auto [physical,commit]=host_free();if(host_count<max_pinned&&io_round(b.bytes)<=host_budget(physical,commit,floor_gib)&&b.host.allocate(b.bytes,true,true)){load(b,b.host.p);++host_count;admitted=true;}}
            if(!admitted){++file_count;file_ring_mask|=1u<<(l%2);}
        }
        need(!init.admit_layers||!require_file_tier||file_tier_covered(file_count,1,file_ring_mask,both_file_rings),"requested file tier requires admitted file layers and both ring parities");
        exec.decode_stream=decode.s;ck(prepare_cutlass(exec));
        auto ev=[&](){cudaEvent_t e{};ck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));exec_events.push_back(e);return e;};
        exec.dispatch=ev();for(unsigned i=0;i<18;++i){exec.ready[i]=ev();exec.retired[i]=ev();}
        ck(prepare_kda());ck(prepare_dsa());
        for(unsigned l=0;l<46;++l){auto w=std::make_unique<Layer>();auto p=layer_name(l);w->input=u16(p+"input_layernorm.weight");w->post=u16(p+"post_attention_layernorm.weight");
            if(l<45){auto hc=[&](const char* site){auto n=p+site;return Hc{f32(n+"fn"),f32(n+"base"),f32(n+"scale")};};w->attn=hc("hc_attn_");w->ffn=hc("hc_ffn_");}
            auto a=p+"self_attn.";
            if(l<45&&l%4!=3)w->k={4096,64,matrix(a+"qkv_proj",4096,24576),matrix(a+"o_proj",8192,4096),u16(a+"b_proj.weight"),u16(a+"f_a_proj.weight"),u16(a+"f_b_proj.weight"),u16(a+"g_a_proj.weight"),u16(a+"g_b_proj.weight"),u16(a+"conv1d.weight"),u16(a+"o_norm.weight"),f32(a+"A_log"),f32(a+"dt_bias")};
            else w->d={matrix(a+"q_a_proj",4096,1536),matrix(a+"q_b_proj",1536,16384),matrix(a+"kv_a_proj_with_mqa",4096,512),matrix(a+"o_proj",16384,4096),matrix(a+"indexer.wq_b",1536,4096),u16(a+"q_a_layernorm.weight"),u16(a+"kv_a_layernorm.weight"),u16(a+"kv_b_proj.weight"),u16(a+"indexer.wk.weight"),u16(a+"indexer.index_kpool_compress_gate"),u16(a+"indexer.weights_proj.weight"),u16(a+"indexer.k_norm.weight"),u16(a+"indexer.k_norm.bias"),f32(a+"indexer.index_kpool_compress_ape")};
            if(l<3)w->dense=expert(p+"mlp.",12288);else {w->router=u16(p+"mlp.gate.weight");w->bias=f32(p+"mlp.gate.e_score_correction_bias");w->shared=expert(p+"mlp.shared_experts.",2048);w->descriptors.allocate(288*sizeof(Expert));}
            w->shared_descriptor.allocate(sizeof(Expert));auto descriptor=l<3?w->dense:w->shared;
            ck(cudaMemcpyAsync(w->shared_descriptor.p,&descriptor,sizeof descriptor,cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));
            if(l==45){w->m.layer=45;w->m.eh=matrix(p+"eh_proj",8192,4096);w->m.attention=w->d;w->m.enorm=u16(p+"enorm.weight");w->m.hnorm=u16(p+"hnorm.weight");w->m.input_norm=w->input;w->m.post_norm=w->post;w->m.head_norm=u16(p+"shared_head.norm.weight");w->m.router=w->router;w->m.router_bias=w->bias;w->m.experts=w->descriptors.as<Expert>();w->m.shared=w->shared_descriptor.as<Expert>();}
            if(l>=3&&!init.startup_descriptors)ck(cudaMemsetAsync(w->descriptors.p,0,288*sizeof(Expert),h2d.s)); // bound per layer by prefill_begin
            else if(l>=3){auto& b=*blobs.at(expert_name(l));const void* base=b.device.p?b.device.p:ring[l%2].device.p;
                w->host_descriptors.allocate(288*sizeof(Expert),true);auto* desc=w->host_descriptors.as<Expert>();
                for(unsigned e=0;e<288;++e)desc[e]=expert(p+"mlp.experts."+std::to_string(e)+".",2048,base);
                // Every layer has a fixed resident address or fixed ring parity.
                // These descriptors are immutable for the lifetime of the service.
                ck(cudaMemcpyAsync(w->descriptors.p,desc,288*sizeof(Expert),cudaMemcpyHostToDevice,h2d.s));}
            layers.push_back(std::move(w));
        }
        // persistent_dense_fp16 (decode): each shared expert and dense FFN decoded once with the C1
        // decode window (the same FP16 operands every call would decode), charged by name, before
        // the cache opens so its arena is planned from what is left.
        if(decode_on&&dopt.persistent_dense_fp16){Charge c(persist_part);Buffer one;one.allocate(4);const U32 h1=1;ck(cudaMemcpy(one.p,&h1,4,cudaMemcpyHostToDevice));
            for(unsigned l=0;l<45;++l){auto& w=*layers[l];const std::size_t inter=l<3?12288:2048,n=std::size_t(4096)*inter;
                w.fp16_g.allocate(n*2);w.fp16_u.allocate(n*2);w.fp16_d.allocate(n*2);
                ck(decode_expert_fp16(w.shared_descriptor.as<Expert>(),4096,unsigned(inter),w.fp16_g.as<U16>(),w.fp16_u.as<U16>(),w.fp16_d.as<U16>(),one.as<U32>(),exec,compute.s));}
            ck(cudaStreamSynchronize(compute.s));ck(cudaGetLastError());}
        // Tierstream opens last: its VRAM arena is what remains after prefill
        // residency, base weights, workspaces, head and session; RAM from the
        // free host memory above the floor; the unit file is the disk tier.
        if(decode_on&&dopt.expert_tier=="tierstream")open_tier_cache(floor_gib);
        profile=Json::object();profile.set("max_device_layers",int(max_device));profile.set("max_pinned_layers",int(max_pinned));profile.set("require_file_tier",require_file_tier);profile.set("require_both_file_rings",both_file_rings);profile.set("file_ring_mask",int(file_ring_mask));profile.set("weight_source",shards?"shards":"cache");profile.set("source_inventory_hash_ms",source_inventory_ms);profile.set("source_load_ms",source_load_ms);profile.set("source_load_bytes",source_load_bytes);profile.set("graph_templates",graphs);profile.set("graph_template_limit",92);profile.set("workspace_rows",int(tile));
        profile.set("verified_hash_hits",int(hash_hits.load()));profile.set("hash_workers",int(std::max(1u,std::thread::hardware_concurrency())));profile.set("host_floor_gib",int(floor_gib));profile.set("file_io_chunk_bytes",std::uint64_t(io_chunk));profile.set("file_io_inflight",2);
        if(ts_mode){profile.set("prefill_experts",prefill_via_cache?"through the tierstream cache: per layer make_resident(288) + one-layer step + slot-bound descriptors (one copy of every expert)":
            "INTERIM file tier only: tierstream mode keeps no prefill resident/pinned expert copies and streams every layer through two rings (the linked tierstream has no make_resident)");
            profile.set("requested_max_device_layers",int(requested_device_layers));profile.set("requested_max_pinned_layers",int(requested_pinned_layers));}
        std::uint64_t request_bytes=0;for(auto& [name,b]:blobs)if(name!="base.bin"&&name!="head.bin"&&!b->device.p&&!b->host.p)request_bytes+=b->bytes;
        if(decode_on){auto d=Json::object();d.set("identity",decode_identity);d.set("qualification","DRAFT");d.set("expert_tier",tier->backend());d.set("head_arithmetic",head_arithmetic);d.set("head_identity",head_identity);d.set("kda_block_policy",kda_block_policy);
            auto sb=session_state_bytes(cap);d.set("session_state_bytes",sb.total);d.set("replay_state_bytes",sb.replay);d.set("head_decoded_bytes",std::uint64_t(head_decoded_bytes()));d.set("expert_unit_stride",catalog->unit_stride());d.set("fetched_expert_cache",tier->backend()==std::string("reference")?"none (uncached reference reads; test backend)":"tierstream");if(!tier_plan.is_null())d.set("tier_plan",tier_plan);profile.set("decode",d);}profile.set("file_tier_bytes_per_request",request_bytes);profile.set("estimated_file_tier_host_ms",source_load_bytes?double(request_bytes)*source_load_ms/double(source_load_bytes):0.0);profile.set("cost_estimate_basis","resident load throughput; excludes overlapped GPU compute; cold storage may differ");profile.set("device_layers",int(device_count));profile.set("pinned_layers",int(host_count));profile.set("file_layers",int(file_count));profile.set("compressed_rings",init.rings?2:0);profile.set("decoded_buffers",init.rings?2:0);profile.set("prefill_mode",prefill_via_cache?"cache":"rings");profile.set("workspace_bytes",std::uint64_t(ws_bytes));profile.set("read_ahead_layers",init.rings?1:0);profile.set("h2d_streams",1);
        ck(cudaMemGetInfo(&free,&total));profile.set("device_free_bytes",std::uint64_t(free));profile.set("device_used_bytes",std::uint64_t(total-free));
        if(ts_mode)profile.set("memory",memory_json("startup"));
        auto [remaining_physical,remaining_commit]=host_free();profile.set("host_available_physical_bytes",remaining_physical);profile.set("host_available_commit_bytes",remaining_commit);
        ck(cudaStreamSynchronize(h2d.s));
    }
    ~Impl(){for(auto& r:ring)if(r.read.valid())r.read.wait();cudaStreamSynchronize(compute.s);cudaStreamSynchronize(h2d.s);cudaStreamSynchronize(decode.s);
        tier.reset();reference_tier.reset(); // close the cache while the unit file binding handles are still held
        for(auto e:exec_events)cudaEventDestroy(e);}
    static std::string layer_name(unsigned l){return "model.language_model.layers."+std::to_string(l)+".";}
    static std::string expert_name(unsigned l){return std::string("experts-")+(l<10?"0":"")+std::to_string(l)+".bin";}
    const Json& tensor(const std::string& name)const{auto it=tensor_index.find(name);need(it!=tensor_index.end(),"missing model tensor");return *it->second;}
    void* pointer(const std::string& name,const void* expert_base=nullptr)const{
        const auto& t=tensor(name);auto& b=*blobs.at(str(t,"blob"));const void* base=b.device.p?b.device.p:expert_base;need(base!=nullptr,"nonresident tensor without ring");return const_cast<char*>(static_cast<const char*>(base))+integer(at(t,"offset"));
    }
    const U16* u16(const std::string& n)const{auto dt=str(tensor(n),"dtype");need(dt=="F16"||dt=="BF16","expected 16-bit tensor");return static_cast<U16*>(pointer(n));}
    const float* f32(const std::string& n)const{need(str(tensor(n),"dtype")=="F32","expected FP32 tensor");return static_cast<float*>(pointer(n));}
    Matrix matrix(const std::string& n,unsigned in,unsigned out,const void* ring_base=nullptr)const{
        auto& t=tensor(n+".trellis");auto& sh=at(t,"shape").as_array();need(str(t,"dtype")=="I16"&&sh.size()==3&&integer(sh[0])==in/16&&integer(sh[1])==out/16,"matrix geometry");unsigned rate=unsigned(integer(sh[2])/16);need(integer(sh[2])==rate*16&&rate>=3&&rate<=6,"matrix rate");
        return {static_cast<U16*>(pointer(n+".trellis",ring_base)),static_cast<U16*>(pointer(n+".suh",ring_base)),static_cast<U16*>(pointer(n+".svh",ring_base)),in,out,rate};
    }
    Expert expert(const std::string& n,unsigned inter,const void* base=nullptr)const{return {matrix(n+"gate_proj",4096,inter,base),matrix(n+"up_proj",4096,inter,base),matrix(n+"down_proj",inter,4096,base)};}
    void load(Blob& b,void* out){
        auto begin=std::chrono::steady_clock::now();
        if(shards)shards->read_into(b.name,out);
        else b.file->read_into(out);
        source_load_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();source_load_bytes+=b.bytes;
    }
    void read_file(Blob& b,Ring& r){auto begin=std::chrono::steady_clock::now();if(shards)shards->read_into(b.name,r.host.p);else b.file->read_into(r.host.p);r.read_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
    void readahead(unsigned l){if(l<3||l>45)return;auto& b=*blobs.at(expert_name(l));if(b.device.p||b.host.p)return;auto& r=ring[l%2];if(r.layer==int(l))return;
        if(r.read.valid())r.read.get();if(r.used)ck(cudaEventSynchronize(r.loaded.e)); // host ring is free after its last H2D
        r.layer=l;r.read=std::async(std::launch::async,[this,&r,&b](){read_file(b,r);});
    }
    void stage(unsigned l,bool acquire=true){if(l<3||l>45)return;
        // Prefill through the cache: the layer's units are made resident and its
        // descriptors bound at the layer's start (no rings, no pre-staging).
        if(prefill_via_cache){if(acquire&&!staged[l]){prefill_begin(l);staged[l]=true;}return;}
        if(staged[l]){if(acquire)ck(cudaStreamWaitEvent(compute.s,layers[l]->ready.e,0));return;}
        auto& b=*blobs.at(expert_name(l));void* base=b.device.p;auto& r=ring[l%2];
        if(!base){if(r.used)ck(cudaStreamWaitEvent(h2d.s,r.retired.e,0));if(!b.host.p){readahead(l);if(r.read.valid()){r.read.get();file_read_ms+=r.read_ms;file_bytes+=b.bytes;file_ring_bytes[l%2]+=b.bytes;}}void* host=b.host.p?b.host.p:r.host.p;
            ck(cudaMemcpyAsync(r.device.p,host,b.bytes,cudaMemcpyHostToDevice,h2d.s));ck(cudaEventRecord(r.loaded.e,h2d.s));base=r.device.p;r.used=true;r.layer=l;
        }
        ck(cudaEventRecord(layers[l]->ready.e,h2d.s));staged[l]=true;
        if(serial_stage)ck(cudaEventSynchronize(layers[l]->ready.e));
        if(acquire)ck(cudaStreamWaitEvent(compute.s,layers[l]->ready.e,0));
        // Only one future layer may be reading; parity reuse is event protected.
        readahead(l+1);
    }
    DsaState dstate(){return {latent.as<U16>(),key.as<U16>(),gate.as<U16>(),pooled.as<float>(),capacity};}
    // Prefill writes either the shared release buffers (legacy) or one resident
    // per-layer state set (session). Legacy pointers/launches are unchanged.
    bool in_session=false;
    float* kst(unsigned l){return in_session?kda_s.as<float>()+std::size_t(kda_slot(l))*(kda_state_bytes/4):state.as<float>();}
    float* kcv(unsigned l){return in_session?kda_c.as<float>()+std::size_t(kda_slot(l))*(kda_conv_bytes/4):conv.as<float>();}
    float* cks(unsigned l){return ck_s.as<float>()+std::size_t(kda_slot(l))*(kda_state_bytes/4);}
    float* ckc(unsigned l){return ck_c.as<float>()+std::size_t(kda_slot(l))*(kda_conv_bytes/4);}
    float* tl(unsigned l){return tail.as<float>()+std::size_t(kda_slot(l))*kda_block*4096;}
    DsaState dstate(unsigned l){return in_session?sdstate(l):dstate();}
    void zero_layer(unsigned l){
        if(!in_session){zero_state();return;}
        if(is_kda_layer(l)){ck(cudaMemsetAsync(kst(l),0,kda_state_bytes,compute.s));ck(cudaMemsetAsync(kcv(l),0,kda_conv_bytes,compute.s));return;}
        auto d=dstate(l);std::size_t c=capacity;ck(cudaMemsetAsync(d.latent,0,c*1024,compute.s));ck(cudaMemsetAsync(d.key,0,c*256,compute.s));ck(cudaMemsetAsync(d.gate,0,c*256,compute.s));ck(cudaMemsetAsync(d.pooled,0,(c/4)*512,compute.s));
    }
    std::string session_key(unsigned P)const{return in_session?":s"+std::to_string(kda_block_start(P))+"/"+std::to_string(kda_open_rows(P)):"";}
    void zero_state(){for(Buffer* b:{&state,&conv,&latent,&key,&gate,&pooled})ck(cudaMemsetAsync(b->p,0,b->bytes,compute.s));}
    void pre(const Hc& hc,const U16* weight,const float* x,float* normalized_out,unsigned n){
        if(n==1&&dopt.row_hc&&!dopt.row_r2)hc_pre_staged<false><<<1,256,0,compute.s>>>(x,hc.fn,hc.base,hc.scale,F.as<float>(),mix.as<float>(),collapsed.as<float>());
        else if(n<=4&&dopt.row_hc)hc_pre_staged<true><<<n,256,0,compute.s>>>(x,hc.fn,hc.base,hc.scale,F.as<float>(),mix.as<float>(),collapsed.as<float>());
        else hc_pre_kernel<<<n,256,0,compute.s>>>(x,hc.fn,hc.base,hc.scale,F.as<float>(),mix.as<float>(),collapsed.as<float>());
        norm<<<n,256,0,compute.s>>>(collapsed.as<float>(),weight,normalized_out,n);
    }
    void post(const float* x,const float* y,float* out,unsigned n){hc_expand_kernel<<<dim3(16,n),256,0,compute.s>>>(x,mix.as<float>(),y,out,flag.as<unsigned>(),0);}
    void export_state(unsigned l,unsigned P,const StateSink& sink){
        ck(cudaStreamSynchronize(compute.s));unsigned bad=0;ck(cudaMemcpy(&bad,flag.p,4,cudaMemcpyDeviceToHost));need(!bad,"nonfinite prefix; abort handoff");
        auto release=Json::object();release.set("layer",int(l));release.set("state_ready_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-request_start).count());releases.push(release);
        // Host temporaries of this layer's export, charged for their lifetime: KDA
        // state + conv downloads + one rank's conv slice; DSA: one component at a time.
        static constexpr const char* export_part="state export downloads/slices (temporary)";
        HostTemp temporaries(export_part,(l<45&&l%4!=3)?std::uint64_t(kda_state_bytes)+2ULL*kda_conv_bytes:std::uint64_t(P)*1024);
        auto download=[&](const void* p,std::size_t bytes){std::vector<std::byte> out(bytes);ck(cudaMemcpy(out.data(),p,bytes,cudaMemcpyDeviceToHost));Buffer::observe_allocation(export_part);return out;};
        if(l<45&&l%4!=3){
            auto s=download(kst(l),kda_state_bytes),c=download(kcv(l),kda_conv_bytes);
            for(unsigned rank=0;rank<2;++rank){sink(rank,l,0,kda_tp_state(s,rank));
                // Node serialization is q/k/v-major. Select 32 heads separately
                // inside each part, then concatenate (never interleave heads).
                auto sliced=kda_tp_conv(c,rank);Buffer::observe_allocation(export_part);
                sink(rank,l,1,sliced);
            }
        }else{
            auto d=dstate(l);
            for(auto pair:{std::pair<const void*,unsigned>{d.latent,2},{d.key,3},{d.gate,4},{d.pooled,5}}){auto bytes=pair.second==2?std::size_t(P)*1024:pair.second==5?std::size_t(P/4)*512:std::size_t(P)*256;if(!bytes)continue;auto out=download(pair.first,bytes);for(unsigned rank=0;rank<2;++rank)sink(rank,l,pair.second,out);}
        }
    }
    DsaIndex index(const std::string& id){if(id=="glm-pf-rtx-native-v1"||id==dsa_identity)return DsaIndex::fp32;for(auto v:{DsaIndex::mma,DsaIndex::mma_proj_fp32_score,DsaIndex::fp32_proj_mma_score,DsaIndex::tf32x3})if(id==dsa_index_identity(v))return v;throw std::runtime_error("unsupported producer identity");}
    void run(std::span<const U32> input,const std::string& identity,const StateSink& sink){
        choose_tile(input.size());
        try{run_prefill(input,identity,sink);record_tile();record_prefill_cache();write_census_rows();}
        catch(const std::exception& e){if(prefill_via_cache&&ts_tier)ts_tier->abort_prefill(e.what());throw;}
        catch(...){if(prefill_via_cache&&ts_tier)ts_tier->abort_prefill("prefill: non-standard exception");throw;}
    }
    void run_prefill(std::span<const U32> input,const std::string& identity,const StateSink& sink){
        prefill_counters0=cache_counters();if(ts_tier)ts_tier->acquire_ms.clear();
        if(prefill_via_cache&&ts_tier)ts_tier->choose_overlap(dopt.prefill_overlap); // once per prefill: held layers, no token
        request_start=std::chrono::steady_clock::now();releases=Json::array();staging_ms=0;file_read_ms=0;file_bytes=0;file_ring_bytes[0]=file_ring_bytes[1]=0;graph_binding_ms=0;
        // Request-scoped reads: do not silently reuse a previous request's last
        // two host rings, which would leave a forced-file repeat unexercised.
        for(auto& r:ring){if(r.read.valid())r.read.get();r.layer=-1;}
        std::fill(std::begin(staged),std::end(staged),false);
        need(input.size()>=2&&input.size()<=capacity,"prompt exceeds resident capacity");for(auto t:input)need(t<154880,"token outside vocabulary");unsigned P=unsigned(input.size()-1);auto mode=index(identity);
        census_rows.clear();if(dopt.routes_prefill&&routes_file.is_open())census_rows.assign(std::size_t(P)*routed_gated*routed_topk,~U32(0));
        ck(cudaMemsetAsync(flag.p,0,4,compute.s));ck(cudaMemcpyAsync(tokens.p,input.data(),input.size_bytes(),cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));
        embed<<<unsigned((std::size_t(P)*4*4096+255)/256),256,0,compute.s>>>(u16("model.language_model.embed_tokens.weight"),tokens.as<U32>(),X.as<float>(),P,4);
        // Entire prefix for each layer. No state from a later layer becomes
        // visible before the preceding layer's last prefix row is complete.
        for(unsigned l=0;l<45;++l){auto& w=*layers[l];zero_layer(l);auto stage_start=std::chrono::steady_clock::now();stage(l);staging_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-stage_start).count();
            for(unsigned first=0;first<P;first+=tile){auto chunk=row_tile(first,P,tile);unsigned n=chunk.rows;auto off=std::size_t(first)*4096;
                auto sequence=[&]{
                    pre(w.attn,w.input,X.as<float>()+off*4,normalized.as<float>(),n);
                    if(l%4!=3){KdaCall v{n,w.k,normalized.as<float>(),part.as<float>(),kst(l),kcv(l)};
                        // Session prefill only: capture the open B32 block's start
                        // state/convolution and retain its normalized input rows.
                        const auto b0=kda_block_start(P);KdaCheckpoint cp{};const KdaCheckpoint* checkpoint=nullptr;
                        if(in_session&&kda_open_rows(P)&&first<=b0&&b0<first+n){cp={b0-first,cks(l),ckc(l)};checkpoint=&cp;
                            ck(cudaMemcpyAsync(tl(l),normalized.as<float>()+std::size_t(cp.row)*4096,std::size_t(n-cp.row)*kda_row_bytes,cudaMemcpyDeviceToDevice,compute.s));}
                        ck(enqueue_kda(v,workspace.p,ws_bytes,compute.s,false,checkpoint));}
                    else {DsaCall v{};v.rows=n;v.first=first;v.w=w.d;v.state=dstate(l);v.x=normalized.as<float>();v.y=part.as<float>();v.nonfinite=flag.as<unsigned>();v.index=mode;v.legacy_fp32_index=legacy_index;ck(enqueue_dsa(v,workspace.p,ws_bytes,compute.s));}
                    post(X.as<float>()+off*4,part.as<float>(),X2.as<float>(),n);
                    pre(w.ffn,w.post,X2.as<float>(),normalized.as<float>(),n);
                    if(l<3){Call v{n,1,4096,12288,normalized.as<float>(),nullptr,nullptr,w.shared_descriptor.as<Expert>(),part.as<float>()};predecoded(v,w);ck(enqueue_cutlass(v,workspace.p,ws_bytes,compute.s,exec));}
                    else {router<<<n,320,0,compute.s>>>(normalized.as<float>(),w.router,w.bias,ids.as<U32>(),rates.as<float>(),flag.as<unsigned>());
                        Call v{n,288,4096,2048,normalized.as<float>(),ids.as<U32>(),rates.as<float>(),w.descriptors.as<Expert>(),routed.as<float>()};ck(enqueue_cutlass(v,workspace.p,ws_bytes,compute.s,exec));
                        Call sh{n,1,4096,2048,normalized.as<float>(),nullptr,nullptr,w.shared_descriptor.as<Expert>(),shared.as<float>()};predecoded(sh,w);ck(enqueue_cutlass(sh,workspace.p,ws_bytes,compute.s,exec));
                        add<<<unsigned((std::size_t(n)*4096+255)/256),256,0,compute.s>>>(routed.as<float>(),shared.as<float>(),part.as<float>(),std::size_t(n)*4096,flag.as<unsigned>());
                    }
                    post(X2.as<float>(),part.as<float>(),X.as<float>()+off*4,n);
                };
                auto binding=prefill_binding(first,n,identity,legacy_index,session_key(P));
                if(graphs){auto& g=graph[l][chunk.bucket];g.launch(compute.s,binding,sequence);graph_binding_ms+=g.binding_ms;}else sequence();
                // Census: this tile's router ids, in stream order before the next tile's router overwrites them.
                if(!census_rows.empty()&&l>=3)ck(cudaMemcpy2DAsync(census_rows.data()+std::size_t(first)*routed_gated*routed_topk+std::size_t(l-3)*routed_topk,routed_gated*routed_topk*4,ids.p,routed_topk*4,routed_topk*4,n,cudaMemcpyDeviceToHost,compute.s));
            }
            if(prefill_via_cache&&l>=3)prefill_end(l); // close the layer's step, release its units
            // H2D for N+1 runs while N's queued kernels execute. Acquisition
            // occurs only at N+1; exporting N never waits for N+1's ready event.
            if(!serial_stage){auto begin=std::chrono::steady_clock::now();stage(l+1,false);staging_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
            if(in_session&&is_kda_layer(l)&&!kda_open_rows(P)){ck(cudaMemcpyAsync(cks(l),kst(l),kda_state_bytes,cudaMemcpyDeviceToDevice,compute.s));ck(cudaMemcpyAsync(ckc(l),kcv(l),kda_conv_bytes,cudaMemcpyDeviceToDevice,compute.s));}
            if(!in_session){check_finite<<<unsigned((state.bytes/4+255)/256),256,0,compute.s>>>(state.as<float>(),state.bytes/4,flag.as<unsigned>());export_state(l,P,sink);}
            else if(is_kda_layer(l))check_finite<<<unsigned((kda_state_bytes/4+255)/256),256,0,compute.s>>>(kst(l),kda_state_bytes/4,flag.as<unsigned>());
            if(l>=3)ck(cudaEventRecord(ring[l%2].retired.e,compute.s));
        }
        for(unsigned first=0;first<P;first+=tile){auto chunk=row_tile(first,P,tile);unsigned n=chunk.rows;auto off=std::size_t(first)*4096;hc_final_kernel<<<dim3(16,n),256,0,compute.s>>>(X.as<float>()+off*4,part.as<float>());norm<<<n,256,0,compute.s>>>(part.as<float>(),u16("model.language_model.norm.weight"),hidden.as<float>()+off,n);}
        zero_layer(45);auto stage_start=std::chrono::steady_clock::now();stage(45);staging_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-stage_start).count();auto& w=*layers[45];
        // Each exact row tile pairs hidden[t] with input[t+1]. Absolute DSA
        // positions and complete pools survive between calls; no padded tokens.
        const unsigned mtp_tile=mtp_row_tile(tile); // MTP in its own (<= 2048-row) pieces
        for(unsigned first=0;first<P;first+=mtp_tile){auto chunk=row_tile(first,P,mtp_tile);unsigned n=chunk.rows;auto off=std::size_t(first)*4096;
            auto sequence=[&]{embed<<<unsigned((std::size_t(n)*4096+255)/256),256,0,compute.s>>>(u16("model.language_model.embed_tokens.weight"),tokens.as<U32>()+first+1,embedding.as<float>(),n,1);
                MtpCall v{n,first,w.m,dstate(45),embedding.as<float>(),hidden.as<float>()+off,part.as<float>(),flag.as<unsigned>(),mode};v.legacy_fp32_index=legacy_index;ck(enqueue_mtp(v,workspace.p,ws_bytes,compute.s,exec));};
            auto binding=prefill_binding(first,n,identity,legacy_index,session_key(P));if(graphs){auto& g=graph[45][chunk.bucket];g.launch(compute.s,binding,sequence);graph_binding_ms+=g.binding_ms;}else sequence();
        }
        if(prefill_via_cache)prefill_end(45);
        if(prefill_via_cache&&ts_tier)ts_tier->restore_spares(true); // the decode spare target again, before the residency restore (a failure fails the request)
        // The decode placement again (the prefill's whole-layer sets displaced it); placement only.
        // The decode residency restore is deferred (restore_pending): it runs before the next decode
        // token (or explicitly, untimed, after a TTFT prefill); a prefill-only request never pays it.
        if(prefill_via_cache&&ts_tier&&ts_tier->restore)restore_pending=true;
        ck(cudaEventRecord(ring[1].retired.e,compute.s));if(!in_session)export_state(45,P,sink);ck(cudaStreamSynchronize(compute.s));ck(cudaGetLastError());
        if(in_session){unsigned bad=0;ck(cudaMemcpy(&bad,flag.p,4,cudaMemcpyDeviceToHost));need(!bad,"nonfinite session prefill");}
        unsigned read_mask=(file_ring_bytes[0]?1u:0u)|(file_ring_bytes[1]?2u:0u);
        need(prefill_via_cache||!require_file_tier||file_tier_covered(file_count,file_bytes,read_mask,both_file_rings),"requested file tier performed no reads or missed a ring");
        last_timings=Json::object();last_timings.set("file_ring0_bytes",file_ring_bytes[0]);last_timings.set("file_ring1_bytes",file_ring_bytes[1]);last_timings.set("layer_state_releases",releases);last_timings.set("trunk_staging_host_ms",staging_ms);last_timings.set("graph_binding_host_ms",graph_binding_ms);last_timings.set("file_tier_host_read_ms",file_read_ms);last_timings.set("file_tier_bytes",file_bytes);last_timings.set("runtime_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-request_start).count());
    }
    // ======================= decode session (draft) =======================
    // The authenticated unit binding is declared BEFORE `tier` (destroyed after it);
    // at cache open it moves into the cache's BoundCache owner, which always closes
    // the cache before releasing the binding, including on constructor unwinding.
    // Opt-in decode timeline (decode_timeline.hpp): one event per enqueued phase on the
    // compute stream, read after the token's synchronization. Events only.
    struct DecodeTimelineRec {
        bool on=false;std::vector<cudaEvent_t> ev;std::vector<TimelineMark> marks;std::size_t n=0;std::chrono::steady_clock::time_point last;
        ~DecodeTimelineRec(){for(auto e:ev)if(e)cudaEventDestroy(e);}
        void enable(){if(ev.empty()){ev.resize(640,nullptr);for(auto& e:ev)ck(cudaEventCreate(&e));}on=true;}
        void start(cudaStream_t s){if(!on)return;marks.clear();n=0;ck(cudaEventRecord(ev[n++],s));last=std::chrono::steady_clock::now();}
        void mark(int layer,DecodePhase phase,cudaStream_t s){if(!on)return;need(n<ev.size(),"decode timeline: event pool exhausted");ck(cudaEventRecord(ev[n++],s));
            const auto t=std::chrono::steady_clock::now();marks.push_back({layer,phase,std::chrono::duration<double,std::milli>(t-last).count()});last=t;}
        // After the stream synchronized: device intervals between consecutive events.
        Json collect(){std::vector<double> gpu(marks.size());for(std::size_t i=0;i<marks.size();++i){float ms=0;ck(cudaEventElapsedTime(&ms,ev[i],ev[i+1]));gpu[i]=ms;}
            float span=0;if(n>1)ck(cudaEventElapsedTime(&span,ev[0],ev[n-1]));return decode_timeline_json(marks,gpu,span);}
        // A failed token, after the streams were synchronized (best effort, never throws): the
        // intervals up to the first event that did not complete; the longest interval names the
        // layer and phase where the device stalled (a timed-out cache wait shows as ~the timeout).
        Json collect_failed()noexcept{try{auto j=Json::object();std::vector<TimelineMark> done;std::vector<double> gpu;double longest=-1;std::size_t at=0;
                for(std::size_t i=0;i<marks.size();++i){if(cudaEventQuery(ev[i+1])!=cudaSuccess){cudaGetLastError();
                        auto p=Json::object();p.set("layer",marks[i].layer);p.set("phase",decode_phase_names[std::size_t(marks[i].phase)]);j.set("first_incomplete",p);break;}
                    float ms=0;if(cudaEventElapsedTime(&ms,ev[i],ev[i+1])!=cudaSuccess){cudaGetLastError();break;}
                    done.push_back(marks[i]);gpu.push_back(ms);if(ms>longest){longest=ms;at=i;}}
                j.set("timeline",decode_timeline_json(done,gpu,0));
                if(longest>=0){auto p=Json::object();p.set("layer",done[at].layer);p.set("phase",decode_phase_names[std::size_t(done[at].phase)]);p.set("gpu_ms",longest);j.set("longest_interval",p);}
                return j;}catch(...){return Json::object();}}
    } timeline;
    Json failed_timeline; // the last failed decode token's partial timeline (--decode-profile)
    Json residency_restore; // the last restore of the decode placement (deferred: before the next decode token)
    bool restore_pending=false;Json prefill_counters0;
    Json cache_counters(){return prefill_via_cache&&ts_tier?ts_tier->api->counters():Json::object();}
    static Json counter_delta(const Json& a,const Json& b){auto j=Json::object();if(!a.is_object()||!b.is_object())return j;
        for(const auto& [k,v]:b.as_object()){auto* x=a.find(k);if(x&&v.is_number()&&x->is_number()){const auto d=v.as_int()-x->as_int();j.set(k,std::uint64_t(d>0?d:0));}}return j;}
    // The pending restore (placement only), timed and counted on its own.
    Json run_pending_restore(){if(!restore_pending||!ts_tier)return Json();restore_pending=false;
        const auto c0=cache_counters();const auto t0=std::chrono::steady_clock::now();auto r=ts_tier->restore_residency(compute.s);
        r.set("wall_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());r.set("cache",counter_delta(c0,cache_counters()));
        residency_restore=r;return r;}
    Json failed_slots;    // the slot census after the last failed decode token (always; best effort)
    void failed_census()noexcept{try{if(ts_tier&&tier.get()==ts_tier)failed_slots=ts_tier->census(compute.s);}catch(...){cudaGetLastError();}}
    bool decode_mode=false;SessionFrontier sess;std::unique_ptr<ExpertCatalog> catalog;std::unique_ptr<UnitBinding> unit_binding;std::unique_ptr<ExpertTier> tier;
    Buffer kda_s,kda_c,ck_s,ck_c,tail,dlat,dkey,dgate,dpool,head_dec,head_suh,head_svh,head_ws,logits,argres,rx,rx2,hrow,emb,route,mtp_out,host_small,route_sg,route_keys,dlogits;
    bool have_draft=false;U32 draft_host=0; // mtp_self_spec: the device draft (argres + 16) is valid
    // Adaptive self-speculation (mtp_self_spec 1; "always" drafts every step): which path emits tokens faster is
    // measured, as ms per emitted token (EMA), for verify cycles and for plain steps. A session starts with 8
    // verify cycles and 4 plain steps; then a run of 32 steps on the faster path is followed by a probe of 4 on the
    // other. Both paths emit the plain greedy tokens, so the choice changes speed only.
    struct SpecControl {bool plain=false;unsigned left=8;double ms[2]{0,0};bool known[2]{false,false};bool probe=false;};
    SpecControl spec_ctl;
    void spec_observe(bool plain,double ms_per_token){auto& m=spec_ctl.ms[plain];m=spec_ctl.known[plain]?0.75*m+0.25*ms_per_token:ms_per_token;spec_ctl.known[plain]=true;}
    void spec_next(){if(--spec_ctl.left)return;auto& c=spec_ctl;
        const bool best_plain=c.known[0]&&c.known[1]&&c.ms[1]<c.ms[0];
        if(c.probe||!c.known[!c.plain]){if(c.probe){c.probe=false;c.plain=best_plain;c.left=32;}else{c.probe=true;c.plain=!c.plain;c.left=4;}}
        else{c.probe=true;c.plain=!best_plain;c.left=4;}}
    // Routed ids exported per gated layer and step: rows x top-k (2 rows under mtp_self_spec).
    static constexpr unsigned spec_drafts=3; // drafts per verify (rows <= 4: the row MoE path, one 16-row MMA tile)
    unsigned tier_ids()const{return dopt.mtp_spec?(spec_drafts+1)*routed_topk:routed_topk;}
    std::vector<std::vector<std::byte>> snapshot;std::vector<std::uint32_t> snapshot_tokens;
    static void check_device(){
        int count=0;ck(cudaGetDeviceCount(&count));need(count==1,"one visible device required");cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
        need(prop.major==12&&prop.minor==0&&std::string(prop.name).find("PRO")!=std::string::npos,"requires sm_120 PRO");}
    static std::uint64_t workspace_for(unsigned rows,unsigned cap){return std::max({workspace_bytes_cutlass(rows,288,4096,2048),workspace_bytes_cutlass(rows,1,4096,12288),kda_workspace_bytes(rows,4096,64),dsa_workspace_bytes(rows,cap),mtp_workspace_bytes(mtp_row_tile(rows),cap)});}
    static std::uint64_t prefill_fixed_bytes(unsigned cap,unsigned tile,std::uint64_t ws,std::uint64_t base,std::uint64_t largest,std::uint64_t decode_fixed){
        return activation_bytes(cap,tile)+64*128*128*4+3*64*128*3*4+std::uint64_t(cap)*1664+ws+base+2*largest+(4ULL<<20)+decode_fixed;}
    // Component names: the budget plan's fixed parts and the ledger's Charge labels.
    static constexpr const char* context_part="CUDA context + other processes (measured)";
    static constexpr const char* process_part="process private bytes at startup (measured)";
    static constexpr const char* trunk_part="trunk + dense + shared experts (base.bin)";
    static constexpr const char* activation_part="activations (capacity, row tile)";
    static constexpr const char* attention_part="prefill KDA/DSA state (capacity)";
    static constexpr const char* workspace_part="workspaces (CUTLASS, KDA, DSA, MTP)";
    static constexpr const char* ring_part="prefill expert rings (2 x largest layer)";
    static constexpr const char* kv_part="decode KV/state + replay (capacity)";
    static constexpr const char* head_part="lm_head (decoded FP16, scales, logits, head workspace)";
    static constexpr const char* head_blob_part="lm_head compressed (startup transient)";
    static constexpr const char* compare_part="comparison backend slots";
    static constexpr const char* table_part="GLM unit tables";
    static constexpr const char* persist_part="persistent FP16 shared + dense experts (decode)";
    // Shared experts of layers 3..44 (H x 2048 gate/up, 2048 x H down) and the dense FFNs of 0..2 (H x 12288).
    static constexpr std::uint64_t persist_bytes=42ULL*3*4096*2048*2+3ULL*3*4096*12288*2;
    bool persist_refused=false;std::string persist_note;bool spares_reduced=false;unsigned planned_spare_slots=0; // the startup plan dropped the spare share (file tier)
    static constexpr const char* staging_part="startup staging + small host buffers";
    // Our host commit (process private bytes minus our WDDM device backing).
    static std::uint64_t process_host_bytes(){const auto c=process_host_commit();need(c.raw_private!=~0ULL,"process memory counters");return c.host;}
    // Budget inputs from blob/tensor sizes only (no hashing): GLM's fixed parts by
    // name for tierstream's ss_budget_plan. Tierstream mode keeps no prefill
    // resident/pinned expert layers (rings only). No budget arithmetic here.
    static WholeDeviceIn budget_inputs(const Json& m,unsigned cap,unsigned tile,const DecodeOptions& d,std::uint64_t context,std::uint64_t host_avail,std::uint64_t process_private,bool rings){
        ExpertCatalog c(at(m,"tensors"));std::uint64_t base=0,head=0,largest=0;
        for(const auto& [name,j]:at(m,"blobs").as_object()){auto b=integer(at(j,"bytes"));if(name=="base.bin")base=b;else if(name=="head.bin")head=b;else largest=std::max(largest,b);}
        need(base&&head&&largest,"budget plan needs base, head and expert blob sizes");
        const auto span=std::max(c.max_span_bytes(),c.max_unit_bytes());const std::uint64_t dstride=align_up(span,256),hstride=io_round(span+3*io_alignment);
        const auto sb=session_state_bytes(cap);const std::uint64_t kv=sb.total+sb.replay;const bool whole=d.vram_total_gib>=0||d.ram_total_gib>=0;
        WholeDeviceIn in;in.vram_total=gib_bytes(d.vram_total_gib);in.ram_total=gib_bytes(d.ram_total_gib);in.context=context;in.host_avail=host_avail;
        in.device={{trunk_part,base},{activation_part,activation_bytes(cap,tile)+(4ULL<<20)},{attention_part,64*128*128*4+3*64*128*3*4+std::uint64_t(cap)*1664},{workspace_part,workspace_for(tile,cap)},
                   {kv_part,kv},{head_part,decode_device_bytes(cap)-kv},{head_blob_part,head},{table_part,c.size()*sizeof(UnitOffsets)+288*sizeof(Expert)+288*4}};
        if(rings)in.device.push_back({ring_part,2*largest}); // interim: prefill streams layers through its own rings
        else in.min_resident=PrefillResidency::min_resident(); // one whole layer kept (the planner adds the spares), refused before loading
        if(d.reference_compare)in.device.push_back({compare_part,8*dstride+288*sizeof(Expert)});
        if(d.persistent_dense_fp16)in.device.push_back({persist_part,persist_bytes});
        // Margins kept free under the caps (graphs, CUTLASS, untracked, process growth): today's 4 GiB without a whole-device cap.
        in.vram_margin=whole?(1ULL<<30):(4ULL<<30);in.ram_margin=1ULL<<30;
        in.pinned={{staging_part,(8ULL<<20)+65536+(16ULL<<20)}};if(rings)in.pinned.push_back({ring_part,2*io_round(largest)});
        if(d.reference_compare)in.pinned.push_back({compare_part,8*hstride+288*sizeof(Expert)});
        in.pageable={{process_part,process_private}};if(d.reference_compare)in.pageable.push_back({compare_part,hstride});
        add_host_growth_parts(in.pageable,kv); // the bench snapshot (= session state + replay) and driver growth
        in.tier.keys=c.size();in.tier.unit_bytes=c.unit_stride();in.tier.max_ids=d.mtp_spec?4*routed_topk:routed_topk;in.tier.spare_slots=d.tier_spare_slots;in.tier.spare_permille=d.tier_spare_permille;in.tier.host_spares=d.tier_host_spares;in.tier.staging=d.tier_staging;in.tier.disk_readers=disk_bounce_readers(d);in.tier.landing_blocks=d.tier_landing_blocks;
        if(d.tier_ram_gib>=0)in.ram_expert_bytes=gib_bytes(d.tier_ram_gib);
        if(d.tier_vram_gib>=0){TierBudgetIn t=in.tier;const auto req=gib_bytes(d.tier_vram_gib);t.device_bytes=req>(256ULL<<20)?req-(256ULL<<20):0;t.host_bytes=~0ULL>>1;
            in.arena_units=tier_budget(t).resident;} // the explicit arena, planned (and checked) before any load
        return in;}
    // One check for the startup plan and the dry plan: the planner's refusal, an explicit
    // arena the device cannot give, or an arena below the keepable minimum.
    // The startup plan at a row tile. Default (requested 0): the LARGEST tile the
    // whole-device plan admits (schedule.hpp candidates; no small prefill pieces
    // by default; activations and workspaces grow with the tile and come out of
    // the expert arena); --row-tile is planned exactly. The refusal of the
    // smallest candidate is returned when none fits.
    struct TilePlan {unsigned tile=0;WholeDeviceIn in;WholeDevicePlan plan;Json json;std::string refused;bool spares_reduced=false;std::uint64_t file_units_kept_off=0;unsigned spare_slots=0;};
    static TilePlan plan_tile(const Json& m,unsigned cap,unsigned requested,const DecodeOptions& d,std::uint64_t context,std::uint64_t host_avail,std::uint64_t proc,bool rings,std::uint64_t hard){
        auto with=[&](unsigned t,const DecodeOptions& dd){TilePlan p;p.tile=t;p.in=budget_inputs(m,cap,t,dd,context,host_avail,proc,rings);if(!p.in.vram_total)p.in.vram_total=hard;
            need(p.in.vram_total<=hard,("--vram-total-gib "+gib_text(p.in.vram_total)+" exceeds this device's usable "+gib_text(hard)+" (99.86 GB cap)").c_str());
            p.plan=plan_whole_device(p.in);p.json=whole_device_json(p.in,p.plan);p.refused=plan_refusal(p.in,p.plan,p.json.find("summary")->as_string());return p;};
        // Under a file tier the arena spares are sized from the plan (placement only): enough free
        // slots that a prefill layer's make_resident(288) does not evict other layers' decode residents
        // (prefill_spares_needed: 288 - residents per layer, re-planned until it holds), instead of the
        // percent share (more file units) or the minimum (every prefill layer evicts, and the decode
        // residency restore copies them back). The trade (file units at share / plan-sized / minimum)
        // is recorded in the plan.
        auto one=[&](unsigned t){auto p=with(t,d);
            if(p.refused.empty()&&p.plan.tier.disk>0&&d.tier_spare_permille>0){auto d2=d;d2.tier_spare_permille=0;
                auto pmin=with(t,d2);unsigned x=std::max<unsigned>(d.tier_spare_slots,prefill_spares_needed(p.plan.tier.resident));TilePlan p2;
                for(int it=0;it<8;++it){d2.tier_spare_slots=x;p2=with(t,d2);if(!p2.refused.empty())break;
                    const unsigned want=std::max<unsigned>(d.tier_spare_slots,prefill_spares_needed(p2.plan.tier.resident));if(want<=x)break;x=want;}
                if(p2.refused.empty()&&p2.plan.tier.spare_slots<p.plan.tier.spare_slots){p2.spares_reduced=true;p2.spare_slots=x;
                    p2.file_units_kept_off=p.plan.tier.disk>p2.plan.tier.disk?p.plan.tier.disk-p2.plan.tier.disk:0;
                    auto tr=Json::object();tr.set("spares_needed_per_prefill_layer",int(prefill_spares_needed(p2.plan.tier.resident)));
                    auto arm=[&](const TilePlan& q,const char* role){auto a=Json::object();a.set("role",role);a.set("spare_slots",q.plan.tier.spare_slots);
                        a.set("resident_units",q.plan.tier.resident);a.set("pinned_units",q.plan.tier.pinned);a.set("file_units",q.plan.tier.disk);a.set("admitted",q.refused.empty());return a;};
                    auto arms=Json::array();arms.push(arm(p,"percent share"));arms.push(arm(p2,"plan-sized (chosen)"));arms.push(arm(pmin,"minimum"));tr.set("arms",arms);
                    p2.json.set("spare_policy",tr);
                    p2.json.set("spare_share_note","file tier: "+std::to_string(x)+" plan-sized arena spares (a prefill layer's 288 units fit without evicting decode residents; "
                        +std::to_string(p.plan.tier.spare_slots)+" at the "+std::to_string(d.tier_spare_permille/10)+"% share, "+std::to_string(pmin.plan.tier.spare_slots)+" minimum)");return p2;}}
            return p;};
        // The trade at a tile: expert arena, residents and the other tiers, as planned.
        auto trade=[&](const TilePlan& p,const char* role){auto x=Json::object();x.set("row_tile",int(p.tile));x.set("role",role);x.set("verdict",p.refused.empty()?"ADMITTED":"REFUSED");
            x.set("tile_dependent_bytes",tile_cost(cap,p.tile));x.set("expert_arena_bytes",p.refused.empty()?p.plan.tier.slots*p.in.tier.unit_bytes:0);x.set("expert_arena_gib",p.refused.empty()?double(p.plan.tier.slots*p.in.tier.unit_bytes)/double(1ULL<<30):0.0);
            x.set("resident_units",p.plan.tier.resident);x.set("pinned_units",p.plan.tier.pinned);x.set("file_units",p.plan.tier.disk);x.set("spare_slots",p.plan.tier.spare_slots);return x;};
        // Tiles 2048 / chosen / 8192 side by side (those <= capacity), whatever chose the tile.
        auto trades=[&](TilePlan& chosen){auto a=Json::array();
            for(unsigned t:{baseline_row_tile,default_max_row_tile})if(t<=cap&&t!=chosen.tile){auto p=one(t);a.push(trade(p,t==baseline_row_tile?"baseline 2048":"8192 (whole 8K prompt in one piece)"));}
            a.push(trade(chosen,"chosen"));chosen.json.set("row_tile_trade",a);};
        if(requested){need(valid_schedule(cap,requested),"capacity 4-aligned <=131072; tile 32-aligned <=min(capacity,32768)");auto p=one(requested);
            p.json.set("row_tile",int(requested));p.json.set("row_tile_choice","--row-tile (explicit)");p.json.set("row_tile_components",tile_components(cap));trades(p);return p;}
        // The rule (schedule.hpp): largest tile <= 8192 the plan admits whose activation +
        // workspace growth over tile 2048 costs <= 5% of the expert arena planned at 2048.
        const unsigned base_tile=std::min(baseline_row_tile,first_tile_candidate(cap));const auto base=one(base_tile);
        const std::uint64_t base_arena=base.refused.empty()?base.plan.tier.slots*base.in.tier.unit_bytes:0,base_cost=tile_cost(cap,base_tile);
        auto tried=Json::array();TilePlan last;
        for(unsigned t=first_tile_candidate(cap,default_max_row_tile);t;t=next_tile_candidate(t)){need(valid_schedule(cap,t),"tile candidate");auto p=one(t);
            const std::uint64_t cost=tile_cost(cap,t),growth=cost>base_cost?cost-base_cost:0;
            // A larger tile must also keep the file tier no larger than tile 2048 does.
            const bool off_file=!base.refused.empty()||!p.refused.empty()||p.plan.tier.disk<=base.plan.tier.disk;
            const bool within=t<=base_tile||(base_arena&&tile_growth_within(growth,base_arena)&&off_file);
            auto x=Json::object();x.set("row_tile",int(t));x.set("activation_plus_workspace_bytes",cost);x.set("growth_over_2048_bytes",growth);
            x.set("growth_share_of_arena_2048",base_arena?double(growth)/double(base_arena):-1.0);
            x.set("expert_arena_bytes",p.refused.empty()?p.plan.tier.slots*p.in.tier.unit_bytes:0);x.set("resident_units",p.plan.tier.resident);
            x.set("file_units",p.plan.tier.disk);
            x.set("verdict",!p.refused.empty()?"REFUSED (plan)":!off_file?"REFUSED (more expert units on the file tier than tile 2048)":!within?"REFUSED (growth over 5% of the expert arena)":"CHOSEN");tried.push(x);
            if(p.refused.empty()&&within){p.json.set("row_tile",int(t));
                p.json.set("row_tile_choice","default: the largest tile <= 8192 the whole-device plan admits whose activation + workspace growth over tile 2048 is <= 5% of the expert arena planned at 2048");
                p.json.set("expert_arena_at_2048_bytes",base_arena);p.json.set("row_tiles_tried",tried);p.json.set("row_tile_components",tile_components(cap));trades(p);return p;}
            last=std::move(p);}
        last.json.set("row_tiles_tried",tried);last.json.set("row_tile_components",tile_components(cap));return last;
    }
    // Device bytes that grow with the prefill tile: activations + the shared workspace.
    static std::uint64_t tile_cost(unsigned cap,unsigned t){return activation_bytes(cap,t)+workspace_for(t,cap);}
    // The footprint by component at tiles 2048 / 8192 / 32768 (those <= capacity).
    static Json tile_components(unsigned cap){auto a=Json::array();
        for(unsigned t:{2048u,4096u,8192u,32768u}){if(t>cap)continue;auto j=Json::object();j.set("row_tile",int(t));
            j.set("activations_residual_bytes",std::uint64_t(cap)*4096*4*5);j.set("activations_tile_bytes",activation_bytes(cap,t)-std::uint64_t(cap)*4096*4*5);
            j.set("workspace_cutlass_routed_bytes",std::uint64_t(workspace_bytes_cutlass(t,288,4096,2048)));j.set("workspace_cutlass_dense_bytes",std::uint64_t(workspace_bytes_cutlass(t,1,4096,12288)));
            j.set("workspace_kda_bytes",std::uint64_t(kda_workspace_bytes(t,4096,64)));j.set("workspace_dsa_bytes",std::uint64_t(dsa_workspace_bytes(t,cap)));j.set("workspace_mtp_bytes",std::uint64_t(mtp_workspace_bytes(mtp_row_tile(t),cap)));j.set("mtp_row_tile",int(mtp_row_tile(t)));
            j.set("workspace_bytes (max of the above)",workspace_for(t,cap));j.set("tile_dependent_total_bytes",tile_cost(cap,t));a.push(j);}
        return a;}
    static std::string plan_refusal(const WholeDeviceIn& in,const WholeDevicePlan& p,const std::string& summary){
        if(!p.refuse.empty())return p.refuse;
        if(in.arena_units!=~0ULL&&p.tier.resident<in.arena_units)return "the explicit --tier-vram-gib arena ("+std::to_string(in.arena_units)+" resident units) exceeds the "+std::to_string(p.tier.resident)+" units left after the fixed parts: "+summary;
        if(p.tier.resident<in.min_resident)return "the arena keeps "+std::to_string(p.tier.resident)+" resident units, below the "+std::to_string(in.min_resident)+" prefill through the cache needs: "+summary;
        return {};}
    // Prefill through the cache (reader/resident): no prefill rings or expert layers;
    // each routed layer's descriptors are bound from the cache's slots (TierstreamTier).
    bool prefill_via_cache=false;struct TierstreamTier;TierstreamTier* ts_tier=nullptr;
    void prefill_begin(unsigned l);void prefill_end(unsigned l);
    unsigned requested_device_layers=0,requested_pinned_layers=0;WholeDeviceIn budget_in;Json startup_budget;bool unit_journal=false,whole_device=false;
    static std::uint64_t decode_device_bytes(unsigned cap){auto b=session_state_bytes(cap);return b.total+b.replay+head_decoded_bytes()+8192+309760+head_workspace_bytes(1)+std::uint64_t(vocab_size)*4+2*65536+4*16384+(64ULL<<20)+8*16ULL*1024*1024;}
    DsaState sdstate(unsigned l){std::size_t j=dsa_slot(l),c=capacity;return {dlat.as<U16>()+j*c*512,dkey.as<U16>()+j*c*128,dgate.as<U16>()+j*c*128,dpool.as<float>()+j*(c/4)*128,capacity};}
    void load_blob_device(Blob& b,Buffer& staging){
        b.device.allocate(b.bytes);
        if(shards){shards->stream(b.name,staging.p,staging.bytes,[&](std::uint64_t off,const void* p,std::size_t n){ck(cudaMemcpyAsync(b.device.as<char>()+off,p,n,cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));});}
        else for(std::size_t off=0;off<b.bytes;){auto n=std::min<std::size_t>(staging.bytes,b.bytes-off);b.file->read_at(off,staging.p,n);ck(cudaMemcpyAsync(b.device.as<char>()+off,staging.p,n,cudaMemcpyHostToDevice,h2d.s));ck(cudaStreamSynchronize(h2d.s));off+=n;}
        source_load_bytes+=b.bytes;
    }
    void prepare_decode(Buffer& staging,const std::string& backend,unsigned floor_gib){
        const std::size_t c=capacity;Charge kv_charge(kv_part);
        kda_s.allocate(kda_layer_count*kda_state_bytes);kda_c.allocate(kda_layer_count*kda_conv_bytes);ck_s.allocate(kda_layer_count*kda_state_bytes);ck_c.allocate(kda_layer_count*kda_conv_bytes);tail.allocate(kda_layer_count*kda_block*kda_row_bytes);
        dlat.allocate(dsa_layer_count*c*1024);dkey.allocate(dsa_layer_count*c*256);dgate.allocate(dsa_layer_count*c*256);dpool.allocate(dsa_layer_count*(c/4)*512);
        const std::size_t sr=dopt.mtp_spec?spec_drafts+1:1; // rows of a decode step (up to spec_drafts+1: the verify)
        rx.allocate(65536*sr);rx2.allocate(65536*sr);hrow.allocate(16384*sr);emb.allocate(16384*sr);mtp_out.allocate(16384*sr);route.allocate(256);route_sg.allocate(4*288*4);route_keys.allocate(4*288*4);argres.allocate(256);host_small.allocate(65536,true); // router scratch: up to 4 rows (verify)
        // Head: stream head.bin to the device once, decode K6 -> FP16, keep scales, free the compressed blob.
        auto& hb=*blobs.at("head.bin");{Charge h(head_blob_part);load_blob_device(hb,staging);}
        auto hm=matrix("lm_head",head_hidden,head_vocab);need(hm.bits==6,"lm_head must be K6");
        Charge head_charge(head_part);head_dec.allocate(head_decoded_bytes());head_suh.allocate(8192);head_svh.allocate(309760);head_ws.allocate(head_workspace_bytes(dopt.mtp_spec?spec_drafts+1:1));logits.allocate(std::size_t(vocab_size)*4*(dopt.mtp_spec?spec_drafts+1:1));if(dopt.mtp_spec)dlogits.allocate(std::size_t(vocab_size)*4);
        ck(decode_head(hm,head_dec.as<U16>(),h2d.s));ck(cudaMemcpyAsync(head_suh.p,hm.suh,8192,cudaMemcpyDeviceToDevice,h2d.s));ck(cudaMemcpyAsync(head_svh.p,hm.svh,309760,cudaMemcpyDeviceToDevice,h2d.s));
        ck(cudaStreamSynchronize(h2d.s));ck(cudaGetLastError());hb.device.reset();
        // One mutually exclusive selection. Tierstream ALWAYS authenticates its
        // unit file (unit_auth.hpp: held handle first, pinned manifest, contents
        // via recalled authenticated record or full hash) before the cache opens
        // with per-fetch hashing off; the comparison backend is only an addition.
        const auto plan=backend_plan(backend,dopt.reference_compare);
        if(plan.reference)tier=std::make_unique<ReferenceTier>(*this,floor_gib);
        else{
            need(plan.tierstream&&plan.authenticate_unit_file,"tierstream backend plan");need(!dopt.unit_file.empty(),"--expert-tier tierstream needs --unit-file");
            auto auth=authenticate_unit_file(dopt.unit_file,*catalog,model_hash,dopt.unit_manifest_sha256);
            auto binding=std::make_unique<UnitBinding>();binding->lock=std::move(auth.lock);binding->bound=std::make_unique<BoundUnitFile>(*binding->lock);unit_binding=std::move(binding);unit_file_sha=auth.file_sha256;unit_hash_recalled=auth.recalled;unit_hash_ms=auth.ms;unit_auth_how=auth.how();unit_authenticated=true;
            unit_journal=file_journal_identity(unit_binding->lock->held_handle()).has_value();
            std::cerr<<"INFO unit file "<<unit_auth_how<<" in "<<unit_hash_ms<<" ms; record accelerator "<<(unit_journal?"available (USN record: the next start recalls it)":"no USN: mtime-stamped record (engine file-hash stamp v3: volume serial, file index, size, write and creation times (ChangeTime excluded); manifest pin; DPAPI-keyed MAC; held deny-write from open). Weaker than USN: a same-user rewrite restoring every timestamp between runs is not detected")<<"\n";
            if(plan.reference_compare)reference_tier=std::make_unique<ReferenceTier>(*this,floor_gib); // same-window comparison backend
        }
    }
    DecodeOptions dopt;Json tier_plan;std::string unit_file_sha,unit_auth_how="not authenticated";bool unit_hash_recalled=false,unit_authenticated=false;double unit_hash_ms=0;std::unique_ptr<ExpertTier> reference_tier;ExpertTier* product_tier=nullptr;
    // Active backend for the next session (qualification compares identical continuations).
    void select_tier(const std::string& name){
        if(name==std::string(tier->backend()))return;need(reference_tier!=nullptr,"no comparison backend (start with --reference-compare)");std::swap(tier,reference_tier);}
    void open_tier_cache(unsigned floor_gib){
        need(unit_authenticated&&unit_binding!=nullptr&&unit_binding->lock!=nullptr&&unit_binding->bound!=nullptr,"tierstream cache refused: unit file not authenticated and bound");
        const auto& c=*catalog;const std::uint64_t keys=c.size(),stride=c.unit_stride();
        std::size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total));const std::uint64_t cap=99860000000ULL,used=total-free,reserve=4ULL<<30;
        if(ledger.l)ledger.l->observe("cache open (before)",used,process_host_bytes());
        auto [physical,commit]=host_free();const std::uint64_t host_avail=host_budget(physical,commit,floor_gib);
        std::uint64_t avail=0,dev=0,host=host_avail;TierBudget b;Json open_plan;
        // Arena-only requests (--tier-vram-gib / --tier-ram-gib) are granted exactly or refused with the numbers.
        auto arena_units=[&](double gib){TierBudgetIn t;t.keys=keys;t.max_ids=tier_ids();t.unit_bytes=stride;t.spare_slots=dopt.tier_spare_slots;t.spare_permille=dopt.tier_spare_permille;t.host_spares=dopt.tier_host_spares;t.staging=dopt.tier_staging;t.disk_readers=disk_bounce_readers(dopt);t.landing_blocks=dopt.tier_landing_blocks;
            const auto req=gib_bytes(gib);t.device_bytes=req>(256ULL<<20)?req-(256ULL<<20):0;t.host_bytes=~0ULL>>1;return tier_budget(t).resident;};
        if(whole_device){
            // Re-plan from what is allocated now (measured device use; process
            // commit, pinned included): the tiers get the rest of the caps.
            WholeDeviceIn w=budget_in;w.context=used;w.device.clear();w.pinned.clear();w.vram_margin=1ULL<<30;w.ram_margin=1ULL<<30;
            w.pageable={{"process host commit at cache open (measured: private bytes minus our WDDM device backing; pinned included)",process_host_bytes()}};w.host_avail=host_avail;
            {const auto sb=session_state_bytes(capacity);add_host_growth_parts(w.pageable,sb.total+sb.replay);} // still to come after open
            if(dopt.tier_vram_gib>=0)w.arena_units=arena_units(dopt.tier_vram_gib);
            const auto p=plan_whole_device(w);open_plan=whole_device_json(w,p);open_plan.set("mode","whole-device caps at cache open (measured)");
            std::cerr<<"INFO decode budget at cache open: "<<open_plan.dump()<<"\n";
            need(p.refuse.empty(),("whole-device budget refused at cache open: "+p.refuse).c_str());b=p.tier;
            need(dopt.tier_vram_gib<0||b.resident==w.arena_units,("requested tierstream arena "+gib_text(gib_bytes(dopt.tier_vram_gib))+" does not fit the whole-device cap: "+open_plan.find("summary")->as_string()).c_str());
            need(dopt.tier_ram_gib<0||!b.three_tier||(b.pinned+b.host_spares+b.staging)*stride+stride>gib_bytes(dopt.tier_ram_gib),("requested tierstream RAM tier "+gib_text(gib_bytes(dopt.tier_ram_gib))+" does not fit the whole-device cap: "+open_plan.find("summary")->as_string()).c_str());
            avail=dev=(b.slots+b.ring)*stride;host=b.host_bytes(stride);
        }else{
            avail=std::min<std::uint64_t>(free,used<cap?cap-used:0);avail=avail>reserve?avail-reserve:0;dev=avail;
            if(dopt.tier_vram_gib>=0){dev=gib_bytes(dopt.tier_vram_gib);
                need(dev<=avail,("requested tierstream VRAM arena "+gib_text(dev)+" exceeds the "+gib_text(avail)+" available at cache open (device free "+gib_text(free)+", used "+gib_text(used)+", reserve "+gib_text(reserve)+")").c_str());}
            if(dopt.tier_ram_gib>=0){host=gib_bytes(dopt.tier_ram_gib);
                need(host<=host_avail,("requested tierstream RAM tier "+gib_text(host)+" exceeds the "+gib_text(host_avail)+" host memory available above the floor at cache open").c_str());}
            TierBudgetIn in;in.keys=keys;in.unit_bytes=stride;in.device_bytes=dev>(256ULL<<20)?dev-(256ULL<<20):0;in.host_bytes=host>(256ULL<<20)?host-(256ULL<<20):0;
            in.max_ids=tier_ids();in.spare_slots=dopt.tier_spare_slots;in.spare_permille=dopt.tier_spare_permille;in.host_spares=dopt.tier_host_spares;in.staging=dopt.tier_staging;in.disk_readers=disk_bounce_readers(dopt);in.landing_blocks=dopt.tier_landing_blocks;b=tier_budget(in);
        }
        need(b.refuse.empty(),b.refuse.empty()?"":b.refuse.c_str());
        TierOpenConfig cfg;cfg.max_ids=tier_ids();// The cache opens the canonical held path, never the operator's (possibly aliased) name.
        const auto order=placement_order();cfg.tiers=tier_assignment(b,keys,order);cfg.seed=order;cfg.policy=policy_code(dopt.tier_policy);cfg.budget=b;cfg.unit_bytes=stride;cfg.disk_prefetch=dopt.disk_prefetch;cfg.flush_ready=dopt.tier_flush_ready;cfg.ledger=ledger.l.get();
        cfg.weight_dirs=dopt.weight_dirs;cfg.drive_readers=dopt.drive_readers;cfg.disk_queue=dopt.tier_disk_queue;cfg.clean_evict=dopt.tier_clean_evict;
        // Landing options as one group, only when the plan kept the blocks (a file tier): none on a two-tier plan.
        if(b.landing_blocks){cfg.landing_blocks=std::uint32_t(b.landing_blocks);cfg.landing_max_queued=dopt.tier_landing_max_queued;cfg.landing_ttl=dopt.tier_landing_ttl;cfg.landing_chunk=dopt.tier_landing_chunk;}
        // The source path is the bound native path as lossless UTF-8 (tierstream decodes UTF-8);
        // the library opens it through held ancestor handles and requires the authenticated identity.
        BoundCache<UnitBinding> owned(std::move(unit_binding),[&](UnitBinding& u){cfg.descriptor_json=tierstream_descriptor(c,u.bound->utf8(),model_hash).dump();
            BY_HANDLE_FILE_INFORMATION fi{};need(GetFileInformationByHandle(u.lock->held_handle(),&fi)!=0,"unit file identity");
            cfg.source_volume=fi.dwVolumeSerialNumber;cfg.source_index=(std::uint64_t(fi.nFileIndexHigh)<<32)|fi.nFileIndexLow;cfg.source_size=(std::uint64_t(fi.nFileSizeHigh)<<32)|fi.nFileSizeLow;
            return open_tierstream(cfg);});
        if(ledger.l){std::size_t f2=0,t2=0;ck(cudaMemGetInfo(&f2,&t2));ledger.l->observe("cache open",t2-f2,process_host_bytes());}
        auto& ub=owned.binding();ub.bound->verify_serving(*ub.lock); // the bound name still resolves to the authenticated object after the open
        const auto binding_text=ub.bound->describe(),bound_path=ub.bound->path.string();
        {auto t=std::make_unique<TierstreamTier>(*this,std::move(owned));ts_tier=t.get();tier=std::move(t);}product_tier=tier.get();
        ts_tier->planned=planned_residents(cfg.tiers,std::uint8_t(0),routed_gated,routed_experts);ts_tier->restore=dopt.restore_decode_residency;
        if(prefill_via_cache)need(PrefillResidency::refusal(b).empty(),PrefillResidency::refusal(b).c_str());
        ts_tier->budget=b; // prefill overlap room (PrefillResidency::two_layers), decided per prefill
        {std::uint64_t lo=~0ULL,hi=0;for(unsigned g=0;g<routed_gated;++g){std::uint64_t n=0;for(unsigned u=0;u<routed_experts;++u)n+=cfg.tiers[g*routed_experts+u]==0;lo=std::min(lo,n);hi=std::max(hi,n);}
            placement.set("resident_per_layer_min",lo);placement.set("resident_per_layer_max",hi);placement.set("policy",dopt.tier_policy);}
        if(dopt.tier_predict)ts_tier->enable_predict();
        if(!dopt.routes_out.empty()){ts_tier->enable_routes();routes_file.open(dopt.routes_out,std::ios::binary|std::ios::app);need(routes_file.is_open(),("open --routes-out "+dopt.routes_out).c_str());}
        need(!dopt.routes_prefill||routes_file.is_open(),"--routes-prefill needs --routes-out on the tierstream expert tier");
        tier_plan=Json::object();tier_plan.set("placement",placement);tier_plan.set("skip_empty_expert_groups",dopt.skip_empty_expert_groups?"on: routed MoE launches only the expert groups holding routed rows (device compaction)":"off");tier_plan.set("persistent_dense_fp16",dopt.persistent_dense_fp16?"on: shared experts and dense FFNs decoded once at startup":persist_note.empty()?std::string("off"):persist_note);tier_plan.set("flush_ready",dopt.tier_flush_ready?"on: dg_flush_ready, the demand stream is flushed after each ready signal (WDDM submission)":"off");tier_plan.set("predicted_prefetch",dopt.tier_predict?"on: next layer's router on this layer's MoE input, ss_dg_enqueue_predict on a side stream (8 ids, layers 3..43 predict 4..44)":"off");tier_plan.set("routes_recording",dopt.routes_out.empty()?"off":dopt.routes_out);
        tier_plan.set("unit_file",dopt.unit_file);tier_plan.set("unit_file_sha256",unit_file_sha);
        tier_plan.set("weight_dirs",dopt.weight_dirs.empty()?std::string("none: the primary only"):dopt.weight_dirs);if(dopt.weight_dirs.empty())tier_plan.set("drive_readers",std::string("n/a: no --weight-dirs"));else tier_plan.set("drive_readers",std::uint64_t(dopt.drive_readers?dopt.drive_readers:4));
        tier_plan.set("disk_queue",std::uint64_t(b.three_tier?dopt.tier_disk_queue:0));tier_plan.set("disk_bounce_units",std::uint64_t(b.disk_readers));tier_plan.set("landing_blocks_requested",std::uint64_t(dopt.tier_landing_blocks));tier_plan.set("landing_blocks",b.landing_blocks);tier_plan.set("landing_max_queued",std::uint64_t(b.landing_blocks?dopt.tier_landing_max_queued:0));tier_plan.set("landing_ttl",std::uint64_t(b.landing_blocks?dopt.tier_landing_ttl:0));tier_plan.set("landing_chunk",std::uint64_t(b.landing_blocks?dopt.tier_landing_chunk:0));tier_plan.set("clean_evict",b.three_tier&&dopt.tier_clean_evict?"on: device victims with no free pinned block go to their file (no D2H)":"off");tier_plan.set("unit_manifest_sha256",dopt.unit_manifest_sha256);
        {auto c=ts_tier->census(compute.s);for(const auto& l:c.find("layers")->as_array())ts_tier->open_persistent.push_back(std::uint32_t(l.find("persistent")->as_int()));
         tier_plan.set("slot_census_at_open",c);} // homes per gated layer as the cache opened
        if(auto diag=ts_tier->api->diagnostics();diag.find("cuda_module_loading"))tier_plan.set("tierstream_cuda_module_loading",diag.find("cuda_module_loading")->as_string());
        tier_plan.set("restore_decode_residency",dopt.restore_decode_residency?"on: the open placement is made resident again after every prefill through the cache":"off");
        tier_plan.set("unit_file_authenticated",unit_auth_how);tier_plan.set("unit_file_binding",binding_text);tier_plan.set("unit_file_bound_path",bound_path);tier_plan.set("unit_file_hash_ms",unit_hash_ms);tier_plan.set("reference_compare",reference_tier!=nullptr);tier_plan.set("unit_stride",stride);tier_plan.set("resident_units",b.resident);tier_plan.set("pinned_units",b.pinned);tier_plan.set("file_units",b.disk);
        tier_plan.set("arena_slots",b.slots+b.ring);tier_plan.set("spare_slots",b.spare_slots);tier_plan.set("host_spares",b.host_spares);tier_plan.set("disk_staging",b.staging);tier_plan.set("three_tier",b.three_tier);
        tier_plan.set("vram_bytes",(b.slots+b.ring)*stride);tier_plan.set("ram_bytes",b.host_bytes(stride));tier_plan.set("device_budget_bytes",dev);tier_plan.set("host_budget_bytes",host);
        tier_plan.set("device_available_at_open_bytes",avail);tier_plan.set("host_available_at_open_bytes",host_avail);tier_plan.set("startup_budget_plan",startup_budget);
        tier_plan.set("budget_mode",whole_device?"whole-device caps":"device room");if(!open_plan.is_null())tier_plan.set("open_budget_plan",open_plan);
        tier_plan.set("tierstream_commit",tierstream_commit());
        tier_plan.set("unit_file_expected_identity",std::to_string(cfg.source_volume)+":"+std::to_string(cfg.source_index)+":"+std::to_string(cfg.source_size)+" (volume serial, file index, size; required by the library's bound open)");
        tier_plan.set("unit_file_record_accelerator",unit_journal?"USN record (journal id + file USN)":"no USN: mtime-stamped record (engine file-hash stamp v3: volume serial, file index, size, write and creation times (ChangeTime excluded); manifest pin; DPAPI-keyed MAC; held deny-write from open). Weaker than USN: a same-user rewrite restoring every timestamp between runs is not detected");
    }
    // Measured memory for receipts: device-wide cudaMemGetInfo samples (startup,
    // cache open, every report; they include other processes, so they bound ours
    // from above), the ledger's charged peaks by component, and the process's peak
    // commit (pinned included). NVML per-process figures are not linked here.
    // ---- Placement, the DS way (expert_tier.hpp seed_order) ----
    Json placement=Json::object();std::filesystem::path residency_path;std::ofstream routes_file;std::uint64_t route_sessions=0;
    // DS routes census header per decode session ("# seq NAME rows N"; rows = prompt tokens).
    std::vector<U32> census_rows;std::uint64_t census_prompts=0; // --routes-prefill: [row][gated layer][k] of the current prefill
    void write_census_rows(){
        if(census_rows.empty())return;ck(cudaStreamSynchronize(compute.s));const std::size_t per=std::size_t(routed_gated)*routed_topk,rows=census_rows.size()/per;
        routes_file<<"# seq prefill-"<<++census_prompts<<" rows "<<rows<<'\n';
        // Only the layers the prefill trunk routed (3..44); the MTP layer's slots stay unset and are skipped.
        for(std::size_t r=0;r<rows;++r)routes_file<<routes_line_present(census_rows.data()+r*per)<<'\n';
        routes_file.flush();need(bool(routes_file),"write --routes-out (prefill census)");census_rows.clear();}
    void routes_header(const std::string& name,std::size_t rows){if(routes_file.is_open()){routes_file<<"# seq "<<name<<" rows "<<rows<<'\n';routes_file.flush();}}
    static int policy_code(const std::string& p){if(p=="lru")return 0;if(p=="scored")return 1;need(p=="auto","--tier-policy must be lru, scored or auto");return 2;}
    static std::string read_small(const std::filesystem::path& f){std::error_code ec;const auto n=std::filesystem::file_size(f,ec);need(!ec&&n<=(64ULL<<20),("placement input missing or over 64 MiB: "+f.string()).c_str());
        std::ifstream in(f,std::ios::binary);std::string s(std::size_t(n),'\0');in.read(s.data(),std::streamsize(n));need(bool(in),("cannot read "+f.string()).c_str());return s;}
    std::vector<std::uint32_t> placement_order(){
        placement=Json::object();std::vector<std::uint32_t> res;std::vector<std::uint64_t> counts;std::uint64_t tokens=0;
        if(dopt.expert_residency!="none"){residency_path=dopt.expert_residency.empty()?std::filesystem::path(dopt.unit_file+".residency"):std::filesystem::path(dopt.expert_residency);
            std::error_code ec;if(std::filesystem::exists(residency_path,ec))try{res=parse_residency(read_small(residency_path),model_hash);}
                catch(const std::exception& e){placement.set("residency_ignored",std::string(e.what()));res.clear();}}
        for(std::size_t pos=0;!dopt.expert_routes.empty();){const auto next=dopt.expert_routes.find(',',pos);
            const auto one=dopt.expert_routes.substr(pos,next==std::string::npos?std::string::npos:next-pos);if(!one.empty())tokens+=count_routes(read_small(one),counts);
            if(next==std::string::npos)break;pos=next+1;}
        const unsigned uniform_layers=tokens?fill_unobserved_layers(counts):0u; // e.g. the MTP layer of a prefill census
        auto order=seed_order(res,counts);
        placement.set("residency_file",residency_path.string());placement.set("residency_keys",std::uint64_t(res.size()));placement.set("routes_files",dopt.expert_routes);placement.set("routes_tokens",tokens);placement.set("routes_uniform_layers",std::uint64_t(uniform_layers));
        placement.set("order",std::string(res.empty()?"":"residency of the previous run, then ")+(tokens?"route counts, then ":"")+"expert-major round-robin across all routed layers");
        return order;}
    std::string save_residency(){
        if(!ts_tier||residency_path.empty())return "";const auto keys=ts_tier->api->residency();if(keys.empty())return "";
        auto tmp=residency_path;tmp+=".partial";{std::ofstream out(tmp,std::ios::binary|std::ios::trunc);out<<residency_text(model_hash,keys);out.close();need(bool(out),"write residency");}
        std::error_code ec;std::filesystem::rename(tmp,residency_path,ec);need(!ec,("replace residency file: "+ec.message()).c_str());return residency_path.string();}
    std::uint64_t sampled_device_peak=0;HostPeakHistory host_hist;
    Json memory_json(const std::string& stage){
        // Historical peaks (sampled, ledger, process) are always serialized; a fresh
        // sample that fails is reported as such and never discards them.
        auto j=Json::object();const std::uint64_t vcap=gib_bytes(dopt.vram_total_gib),rcap=gib_bytes(dopt.ram_total_gib);
        j.set("vram_total_cap_bytes",vcap);j.set("ram_total_cap_bytes",rcap);
        std::size_t f=0,t=0;const auto e=cudaMemGetInfo(&f,&t);
        if(e==cudaSuccess){const std::uint64_t used=t-f;sampled_device_peak=std::max(sampled_device_peak,used);j.set("device_total_bytes",std::uint64_t(t));j.set("measured_device_used_bytes",used);
            if(ledger.l)try{ledger.l->observe(stage,used,process_host_bytes());}catch(const std::exception& x){j.set("ledger_observe_error",std::string(x.what()));}}
        else{cudaGetLastError();j.set("device_sample_error",std::string(cudaGetErrorString(e)));}
        bool ledger_within=true;
        if(ledger.l)try{auto s=ledger.l->stats();sampled_device_peak=std::max<std::uint64_t>(sampled_device_peak,std::uint64_t(s.find("measured_device_peak_bytes")->as_int()));ledger_within=s.find("within_caps")->as_bool();j.set("ledger",s);}
            catch(const std::exception& x){j.set("ledger_error",std::string(x.what()));ledger_within=false;}
        j.set("measured_device_peak_bytes",sampled_device_peak);
        PROCESS_MEMORY_COUNTERS_EX m{};m.cb=sizeof m;bool host_ok=GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof m)!=0;
        // Host: OUR commit = raw private minus our ATTRIBUTABLE device backing
        // (host_commit_now). Gate peak = max(sampled peak: every ledger host
        // observation, i.e. each allocation, graph operation, charged host
        // temporary, cache open and report; OS bound: the process peak commit
        // (PeakPagefileUsage, every byte ever committed at once) minus the peak
        // attributable backing). The OS bound covers untracked temporaries; it
        // assumes the commit peak coincides with the backing peak (the arena and
        // weights stay allocated from cache open on). History is serialized even
        // when the current sample fails (within_caps is then false).
        if(ledger.l)if(auto* s=j.find("ledger"))if(auto* v=s->find("measured_host_peak_bytes"))host_hist.add_sample(std::uint64_t(v->as_int()));
        if(host_ok){const auto c=host_commit_now(std::uint64_t(m.PrivateUsage));const std::uint64_t os_peak=std::uint64_t(m.PeakPagefileUsage);
            host_hist.add_sample(c.host);host_hist.add_os_bound(os_peak,g_backing_peak);
            j.set("host_commit_bytes",c.host);j.set("host_peak_os_bound_bytes",host_gate_peak(0,os_peak,g_backing_peak));
            j.set("process_private_bytes_raw",c.raw_private);j.set("process_peak_commit_bytes_raw",os_peak);
            j.set("wddm_device_backing_bytes",c.device_backing);j.set("wddm_device_backing_peak_bytes",g_backing_peak);j.set("graph_device_estimate_bytes",g_graph_device_bytes);
            j.set("host_peak_working_set_bytes",std::uint64_t(m.PeakWorkingSetSize));
            if(ledger.l){const std::uint64_t tracked=ledger.l->current(MemKind::pinned)+ledger.l->current(MemKind::pageable);
                j.set("host_tracked_bytes",tracked);j.set("host_commit_residual_bytes",std::int64_t(c.host)-std::int64_t(tracked));}}
        else j.set("host_sample_error","GetProcessMemoryInfo failed");
        if(host_hist.known){j.set("host_peak_sampled_bytes",host_hist.sampled);j.set("host_peak_commit_bytes",host_hist.gate);}
        j.set("host_commit_method","gate host_peak_commit_bytes = max(sampled host peak, OS peak commit - peak attributable device backing). Host = PrivateUsage minus attributable device bytes (exact allocation sizes: our buffers, the cache arena/tables; not the measured context + other processes part, not the device-wide CUDA graph estimates, which stay on the host side). WDDM charges every device allocation to the process commit as eviction backing without making it resident. The OS bound assumes the commit peak coincides with the backing peak.");
        // Unknown is not within: a failed sample or ledger read cannot prove the caps held.
        j.set("within_caps",e==cudaSuccess&&host_ok&&host_hist.known&&(!vcap||sampled_device_peak<=vcap)&&(!rcap||host_hist.gate<=rcap)&&ledger_within);
        if(!g_fatal.empty())j.set("fatal",g_fatal);
        j.set("method","cudaMemGetInfo samples (startup, allocations, graph operations, cache open, after every request; device-wide) + tierstream ss_budget_ledger charges + host commit net of WDDM device backing (raw private/peak kept); NVML per-process not linked");
        return j;}
    HeadWeights head(){return {head_dec.as<U16>(),head_suh.as<U16>(),head_svh.as<U16>()};}
    static Expert rebase(const ExpertEntry& x,const char* base,bool canonical){
        auto mat=[&](unsigned m){auto o=[&](unsigned role){return canonical?x.unit.offset[m][role]:x.t[m][role].offset-x.span_begin;};
            return Matrix{reinterpret_cast<const U16*>(base+o(0)),reinterpret_cast<const U16*>(base+o(1)),reinterpret_cast<const U16*>(base+o(2)),m<2?4096u:2048u,m<2?2048u:4096u,x.unit.rate[m]};};
        return {mat(0),mat(1),mat(2)};
    }
    // Reference backend (exactness/test only; NOT the product cache): per-token
    // uncached per-expert reads from the admitted layer tiers. Device layers use
    // resident descriptors; pinned layers copy only selected expert spans; file
    // layers read only selected ranges through the retained verified handles.
    // Host route readback makes this backend eager (no whole-token graph).
    struct ReferenceTier final:ExpertTier {
        Impl& r;Buffer dslots,hslots,ddesc,hdesc;std::unique_ptr<AlignedMemory> scratch;std::uint64_t dstride=0,hstride=0;U32 ids[8]{};const void* bound=nullptr;unsigned position=0;Json layers=Json::array();
        std::uint64_t logical=0,h2d=0,file_physical=0,selections[3]{};double read_ms=0;
        ReferenceTier(Impl& impl,unsigned floor_gib):r(impl){auto span=std::max(r.catalog->max_span_bytes(),r.catalog->max_unit_bytes());dstride=align_up(span,256);hstride=io_round(span+3*io_alignment);
            auto [physical,commit]=host_free();need(9*hstride+(16ULL<<20)<=host_budget(physical,commit,floor_gib),"insufficient RAM for bounded expert staging");
            Charge charge(compare_part);dslots.allocate(8*dstride);hslots.allocate(8*hstride,true);ddesc.allocate(288*sizeof(Expert));hdesc.allocate(288*sizeof(Expert),true);scratch=std::make_unique<AlignedMemory>(hstride);}
        const char* backend()const override{return "reference";}
        void begin_token(unsigned p)override{position=p;layers=Json::array();logical=h2d=file_physical=0;selections[0]=selections[1]=selections[2]=0;read_ms=0;}
        void acquire(unsigned l,const std::uint32_t* ids_device,unsigned count,void* stream)override{
            need(count==8,"eight routed experts per row");auto s=static_cast<cudaStream_t>(stream);auto* host=r.host_small.as<U32>()+64;
            ck(cudaMemcpyAsync(host,ids_device,32,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));std::memcpy(ids,host,32);
            auto& b=*r.blobs.at(expert_name(l));unsigned kind=b.device.p?0:b.host.p?1:2;auto rec=Json::object();rec.set("layer",int(l));rec.set("tier",kind==0?"device":kind==1?"pinned":"file");auto list=Json::array();
            for(unsigned j=0;j<8;++j){need(ids[j]<288,"router expert ID");list.push(int(ids[j]));logical+=r.catalog->at(l,ids[j]).unit.logical;}rec.set("experts",list);selections[kind]+=8;
            if(kind==0){bound=r.layers[l]->descriptors.p;layers.push(rec);return;}
            auto* desc=hdesc.as<Expert>();std::memset(static_cast<void*>(desc),0,288*sizeof(Expert));std::uint64_t layer_h2d=0,layer_file=0;auto begin=std::chrono::steady_clock::now();
            for(unsigned j=0;j<8;++j){auto& x=r.catalog->at(l,ids[j]);char* dst=dslots.as<char>()+j*dstride;char* hs=hslots.as<char>()+j*hstride;const auto span=x.span_end-x.span_begin;
                if(kind==1){ck(cudaMemcpyAsync(dst,b.host.as<char>()+x.span_begin,span,cudaMemcpyHostToDevice,s));desc[ids[j]]=rebase(x,dst,false);layer_h2d+=span;}
                else if(r.shards){std::vector<std::pair<std::string,std::uint64_t>> want;for(unsigned m=0;m<3;++m)for(unsigned q=0;q<4;++q)want.emplace_back(x.t[m][q].name,x.unit.offset[m][q]);
                    auto io=r.shards->read_ranges(want,hs,scratch->p,hstride);layer_file+=io.physical;ck(cudaMemcpyAsync(dst,hs,x.unit.bytes,cudaMemcpyHostToDevice,s));desc[ids[j]]=rebase(x,dst,true);layer_h2d+=x.unit.bytes;}
                else{auto a=aligned_range(x.span_begin,span,b.bytes,io_alignment);b.file->read_span(a.begin,a.length,hs);layer_file+=io_round(a.length);
                    ck(cudaMemcpyAsync(dst,hs+a.skip,span,cudaMemcpyHostToDevice,s));desc[ids[j]]=rebase(x,dst,false);layer_h2d+=span;}
            }
            ck(cudaMemcpyAsync(ddesc.p,desc,288*sizeof(Expert),cudaMemcpyHostToDevice,s));bound=ddesc.p;
            read_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            h2d+=layer_h2d;file_physical+=layer_file;rec.set("h2d_bytes",layer_h2d);rec.set("file_bytes",layer_file);layers.push(rec);
        }
        const void* bind_selected_descriptors(unsigned,void*)override{need(bound!=nullptr,"bind before acquire");return bound;}
        // Slots are reused only after the next acquire's stream synchronization.
        void release(unsigned,void*)override{bound=nullptr;}
        void end_token()override{}
        Json collect_stats()override{auto j=Json::object();j.set("backend","reference");j.set("position",int(position));j.set("layers",layers);j.set("logical_bytes",logical);j.set("h2d_bytes",h2d);j.set("file_bytes",file_physical);
            j.set("device_selections",selections[0]);j.set("pinned_selections",selections[1]);j.set("file_selections",selections[2]);j.set("acquire_host_ms",read_ms);return j;}
        void reset_cache(bool)override{}
        void fail_token(const std::string&)override{} // stateless: nothing outlives the token
    };
    // Product backend over the shared tierstream cache (GLM expert units). No
    // host readback: export + wait + device bind, capturable into a token graph.
    struct TierstreamTier final:ExpertTier {
        // `owned` is the first member: if this constructor unwinds, or on teardown,
        // it is destroyed last, closing the cache and then releasing the binding.
        BoundCache<UnitBinding> owned;Impl& r;TierCacheApi* api;CacheTokenSite site;Buffer offsets,desc;
        TierstreamTier(Impl& impl,BoundCache<UnitBinding>&& o):owned(std::move(o)),r(impl),api(&owned.cache()),site(*api){auto& c=*r.catalog;auto v=api->view();
            need(v.base&&v.slot_of&&v.unit_base&&v.absent_err&&v.n_gated==43&&v.units==288&&v.stride==c.unit_stride(),"tierstream view does not match GLM unit layout");
            std::vector<UnitOffsets> host(c.size());for(unsigned l=3;l<=45;++l)for(unsigned e=0;e<288;++e)host[std::size_t(l-3)*288+e]=unit_offsets(c.at(l,e));
            Charge charge(table_part);offsets.allocate(host.size()*sizeof(UnitOffsets));ck(cudaMemcpy(offsets.p,host.data(),offsets.bytes,cudaMemcpyHostToDevice));desc.allocate(288*sizeof(Expert));}
        const char* backend()const override{return "tierstream";}
        void begin_token(unsigned)override{site.begin();}
        void acquire(unsigned l,const std::uint32_t* ids,unsigned n,void* s)override{auto st=static_cast<cudaStream_t>(s);
            if(join_pending){ck(cudaStreamWaitEvent(st,join.e,0));join_pending=false;} // the previous layer's predict branch, joined before this export
            if(l==routed_first_layer)api->enqueue_begin(s);
            if(census_on){count_absent<<<1,32,0,st>>>(api->view(),gated_ordinal(l),ids,n,miss_dev.as<U32>());ck(cudaGetLastError());}
            api->enqueue_export(gated_ordinal(l),reinterpret_cast<const std::int32_t*>(ids),n,s);api->enqueue_wait(gated_ordinal(l),s);
            ck(cudaMemsetAsync(desc.p,0,desc.bytes,st));bind_units<<<1,32,0,st>>>(api->view(),gated_ordinal(l),ids,n,offsets.as<UnitOffsets>(),desc.as<Expert>());ck(cudaGetLastError());
            if(route_log.p)ck(cudaMemcpyAsync(route_log.as<U32>()+gated_ordinal(l)*routed_topk,ids,routed_topk*4,cudaMemcpyDeviceToDevice,st));}
        // Overlapped next-layer prefetch (ss_dg_enqueue_predict). Layer l+1's router
        // runs on layer l's MoE input on a side stream into its OWN buffers (ids,
        // rates, flag: never read by the token's math); the library only prefetches
        // those units. The branch forks after l's export+wait and joins before l+1's
        // export. Layer 44 does not predict 45 (MTP: its router is inside enqueue_mtp).
        // Rows <= 4 (a self-speculation verify predicts every row): the promise is set per step (8 ids per row).
        bool predict=false,join_pending=false;Buffer pred_in,pred_ids,pred_rates,pred_flag,pred_sg,pred_keys;Stream side;Event fork,join;unsigned pred_rows=1;
        void enable_predict(){Charge c(table_part);pred_in.allocate(4*4096*4);pred_ids.allocate(4*routed_topk*4);pred_rates.allocate(4*routed_topk*4);pred_flag.allocate(4);
            pred_sg.allocate(4*288*4);pred_keys.allocate(4*288*4);api->set_predict_ids(routed_topk,gated_ordinal(routed_last_layer-1));predict=true;}
        // Between steps (host): the promise of the coming step's exports. Quiet cache: after 16 tokens without a
        // demand miss the step promises (and predicts) nothing, so prefetches stop evicting a warm cache; the
        // next miss turns prediction back on. Outputs never depend on it.
        bool pred_live=true,promised_live=true;unsigned pred_quiet=0;
        void observe_misses(std::uint64_t misses){if(misses){pred_quiet=0;pred_live=true;}else if(++pred_quiet>=16)pred_live=false;}
        void predict_rows(unsigned rows){if(predict&&(rows!=pred_rows||pred_live!=promised_live)){
            api->set_predict_ids(pred_live?routed_topk*rows:0,gated_ordinal(routed_last_layer-1));pred_rows=rows;promised_live=pred_live;}}
        void predict_next(unsigned l,const float* x,cudaStream_t st,unsigned rows=1){
            // Never for a layer the step skips or holds: its export promised nothing (StepPlan).
            if(!predict||!promised_live||l+1>=routed_last_layer||!api->step_exports(gated_ordinal(l+1)))return;auto& w=*r.layers[l+1];
            // The branch reads its OWN copy of the input: the next layer overwrites `x`
            // (normalized) before the join. The copy is ordered on the main stream.
            need(rows==pred_rows,"prediction rows differ from the step's promise");
            ck(cudaMemcpyAsync(pred_in.p,x,std::size_t(rows)*4096*4,cudaMemcpyDeviceToDevice,st));
            ck(cudaEventRecord(fork.e,st));ck(cudaStreamWaitEvent(side.s,fork.e,0));
            if(r.dopt.row_router){router_scores<<<dim3(9,rows),128,0,side.s>>>(pred_in.as<float>(),w.router,w.bias,pred_sg.as<float>(),pred_keys.as<U32>(),pred_flag.as<unsigned>());
                router_select<<<rows,288,0,side.s>>>(pred_sg.as<float>(),pred_keys.as<U32>(),pred_ids.as<U32>(),pred_rates.as<float>());}
            else router<<<rows,320,0,side.s>>>(pred_in.as<float>(),w.router,w.bias,pred_ids.as<U32>(),pred_rates.as<float>(),pred_flag.as<unsigned>());
            ck(cudaGetLastError());
            api->enqueue_predict(gated_ordinal(l),reinterpret_cast<const std::int32_t*>(pred_ids.p),routed_topk*rows,side.s);
            ck(cudaEventRecord(join.e,side.s));join_pending=true;}
        // Slot census (diagnostics): per-layer decode misses (counted before each export) and the
        // table's homes per layer; with --decode-profile per token, at a failure, and at open.
        Buffer miss_dev,census_dev;bool census_on=false;
        unsigned persistent_slots()const{auto v=api->view();const unsigned b=2*r.tier_ids();return v.guard_slot>=b?v.guard_slot-b:0;}
        void enable_census(){if(!miss_dev.p){Charge c(table_part);miss_dev.allocate(routed_gated*4);census_dev.allocate(routed_gated*3*4);}census_on=true;}
        void census_begin(cudaStream_t s){if(census_on)ck(cudaMemsetAsync(miss_dev.p,0xFF,miss_dev.bytes,s));}
        // After the stream synchronized (best effort for a failed token's caller).
        Json census(cudaStream_t s){if(!census_dev.p){Charge c(table_part);miss_dev.allocate(routed_gated*4);ck(cudaMemsetAsync(miss_dev.p,0xFF,miss_dev.bytes,s));census_dev.allocate(routed_gated*3*4);}
            const unsigned persistent=persistent_slots();ck(cudaMemsetAsync(census_dev.p,0,census_dev.bytes,s));slot_census<<<routed_gated,256,0,s>>>(api->view(),persistent,census_dev.as<U32>());ck(cudaGetLastError());
            std::vector<U32> raw(routed_gated*3),miss(routed_gated);ck(cudaMemcpyAsync(raw.data(),census_dev.p,census_dev.bytes,cudaMemcpyDeviceToHost,s));
            ck(cudaMemcpyAsync(miss.data(),miss_dev.p,miss_dev.bytes,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));
            std::vector<LayerCensus> layers(routed_gated);for(unsigned g=0;g<routed_gated;++g)layers[g]={raw[3*g],raw[3*g+1],raw[3*g+2]};
            return census_json(layers,census_on?miss:std::vector<U32>{},persistent);}
        // Routes census (DS routes format): every gated layer's 8 ids of the token.
        Buffer route_log;
        void enable_routes(){Charge c(table_part);route_log.allocate(std::size_t(routed_gated)*routed_topk*4);ck(cudaMemset(route_log.p,0xFF,route_log.bytes));}
        const void* bind_selected_descriptors(unsigned,void*)override{return desc.p;}
        void release(unsigned l,void* s)override{if(l==routed_last_layer)api->enqueue_end(s);}
        // The token stays owned until the library accepted it; a failing end is retired by fail_token.
        void end_token()override{site.end();}
        Json collect_stats()override{return api->stats();}
        void reset_cache(bool open)override{if(open)site.reset_open();}
        // Prefill through the cache (reader/resident contract; PrefillResidency).
        PrefillResidency residency;Buffer all_ids;bool prefill_active=false;
        // An exception while a layer is held or its step is open retires the worker (original error).
        void abort_prefill(const std::string& why){restore_spares(false);if(prefill_active){site.retire(why);prefill_active=false;}}
        void upload_ids(){if(!all_ids.p){Charge c(table_part);all_ids.allocate(PrefillResidency::units*4);ck(cudaMemcpy(all_ids.p,residency.ids.data(),all_ids.bytes,cudaMemcpyHostToDevice));}}
        // Opt-in prefill overlap (PrefillOverlap), decided once per prefill: no token; prefill_active stays set
        // from the first make_resident to the last release (a failure in between retires the worker as above).
        PrefillOverlap overlap;bool overlapping=false;TierBudget budget;std::string overlap_note="off: option";
        std::vector<double> host_wait_ms,gpu_idle_ms; // per layer: host wait on its completion event; GPU gap before its start
        // Overlap mode only (timing events; the serial path records none): per gated layer at its start (before the
        // descriptor bind) and after its last kernel. GPU idle of a layer = end of the previous routed layer -> its
        // start (an upper bound: includes the small inter-layer kernels, e.g. check_finite, zero_layer, the final norm).
        struct LayerEvents {cudaEvent_t start[routed_gated]{},end[routed_gated]{};bool made=false;
            void create(){if(made)return;for(unsigned g=0;g<routed_gated;++g){ck(cudaEventCreateWithFlags(&start[g],cudaEventDefault));ck(cudaEventCreateWithFlags(&end[g],cudaEventDefault));}made=true;}
            ~LayerEvents(){for(unsigned g=0;g<routed_gated;++g){if(start[g])cudaEventDestroy(start[g]);if(end[g])cudaEventDestroy(end[g]);}}
        } ev;
        // Two held layers need slots >= 2 x 288 + the spare target; when only the spares stand in the way the prefill
        // lowers the target (PrefillSpareTarget) and restore_spares() puts it back once the prefill ends or aborts
        // (placement only; a prefill evicts decode residents there anyway). rethrow: the normal end fails the request.
        PrefillSpareTarget spares;unsigned serial_steps=0;
        void restore_spares(bool rethrow){spares.restore(*api,site,rethrow);}
        void choose_overlap(bool want){overlap.reset();host_wait_ms.clear();gpu_idle_ms.clear();serial_steps=0;restore_spares(true);overlapping=want&&PrefillResidency::two_layers(budget);
            std::string lower_failed;if(!overlapping&&spares.lower(*api,site,budget,want,lower_failed))overlapping=true;
            const unsigned spare_lowered=spares.lowered();
            if(overlapping)ev.create();
            if(!lower_failed.empty()){overlap_note="off: spare target not lowered ("+lower_failed+")";return;}
            overlap_note=spare_lowered?"on (spare target "+std::to_string(budget.spare_slots)+" -> "+std::to_string(spare_lowered)+" for the prefill)":overlapping?"on":!want?"off: option":"off: arena "+std::to_string(budget.slots)+" persistent slots < 2 x "+std::to_string(PrefillResidency::units)+" + "+std::to_string(budget.spare_slots)+" spares";}
        void timed_acquire(unsigned g){const auto a0=std::chrono::steady_clock::now();residency.acquire(site,g);acquire_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a0).count());}
        void overlap_begin(unsigned l,Buffer& d,cudaStream_t s){
            prefill_active=true;const unsigned g=gated_ordinal(l);
            overlap.begin(g,[&](unsigned h){ck(cudaStreamSynchronize(s));timed_acquire(h);}); // the first routed layer only; later ones were acquired by the previous end
            ck(cudaEventRecord(ev.start[g],s));upload_ids();ck(cudaMemsetAsync(d.p,0,d.bytes,s));
            bind_units<<<1,PrefillResidency::units,0,s>>>(api->view(),g,all_ids.as<U32>(),PrefillResidency::units,offsets.as<UnitOffsets>(),d.as<Expert>());ck(cudaGetLastError());}
        // l's end event is NOT waited here: the previous layer is waited and released, the next made resident (host;
        // overlaps l's kernels), and the caller enqueues the next layer behind l. After the last layer every set is waited and released.
        void overlap_end(unsigned l,cudaStream_t s){const unsigned g=gated_ordinal(l);ck(cudaEventRecord(ev.end[g],s));
            const bool overlapped=overlap.end(g,l<routed_last_layer?int(gated_ordinal(l+1)):-1,
                [&](unsigned h){const auto a0=std::chrono::steady_clock::now();const bool ok=residency.try_acquire(site,h);
                    if(ok)acquire_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a0).count());return ok;},
                [&](unsigned h){timed_acquire(h);},
                [&](unsigned h){const auto w0=std::chrono::steady_clock::now();ck(cudaEventSynchronize(ev.end[h]));host_wait_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-w0).count());
                    // No ss_end reports an absent unit here: read the library's absent flag (read only) after the kernels.
                    std::uint32_t absent=0;ck(cudaMemcpy(&absent,api->view().absent_err,4,cudaMemcpyDeviceToHost));need(absent==0,"prefill overlap: a held layer's unit was absent from the slot table");},
                [&](unsigned h){site.release_resident(h);});
            if(!overlapped)++serial_steps; // the next layer did not fit beside this one (SS_E_CAPACITY): this step ran serially
            if(!overlap.held().empty())return;
            if(serial_steps)overlap_note+="; "+std::to_string(serial_steps)+" serial steps (SS_E_CAPACITY)";
            prefill_active=false;gpu_idle_ms.assign(routed_gated,0.0); // after the last wait: every layer event completed
            for(unsigned h=1;h<routed_gated;++h){float ms=0;ck(cudaEventElapsedTime(&ms,ev.end[h-1],ev.start[h]));gpu_idle_ms[h]=ms;}}
        void prefill_begin(unsigned l,Buffer& d,cudaStream_t s){
            if(overlapping){overlap_begin(l,d,s);return;}
            ck(cudaStreamSynchronize(s)); // the previous layer's step ended and nothing reads its units any more
            prefill_active=true;
            const unsigned g=gated_ordinal(l);timed_acquire(g);upload_ids();
            site.begin_layer(g);
            try{api->enqueue_begin(s);ck(cudaMemsetAsync(d.p,0,d.bytes,s));
                bind_units<<<1,PrefillResidency::units,0,s>>>(api->view(),g,all_ids.as<U32>(),PrefillResidency::units,offsets.as<UnitOffsets>(),d.as<Expert>());ck(cudaGetLastError());}
            catch(const std::exception& e){site.fail(e.what());throw;}catch(...){site.fail("prefill step: non-standard exception");throw;}}
        void prefill_end(unsigned l,cudaStream_t s){
            if(overlapping){overlap_end(l,s);return;}
            try{api->enqueue_end(s);}catch(const std::exception& e){site.fail(e.what());throw;}catch(...){site.fail("prefill step end: non-standard exception");throw;}
            try{site.end();}catch(const std::exception& e){site.fail(e.what());throw;}catch(...){site.fail("prefill step end: non-standard exception");throw;} // a refused step retires the worker
            site.release_resident(gated_ordinal(l));prefill_active=false;}
        void fail_token(const std::string& why)override{site.fail(why);}
        std::string unusable()const override{return site.unusable();}
        // restore_decode_residency: after a prefill through the cache, the open placement again
        // (planned_residents). Placement only: outputs unchanged. Every held set is released
        // before returning, also when a set is refused (the restore stops there, recorded).
        std::vector<std::vector<std::int32_t>> planned;bool restore=false;std::vector<std::uint32_t> open_persistent;std::vector<double> acquire_ms; // per prefill layer make_resident host ms
        Json restore_residency(cudaStream_t s){
            auto j=Json::object();const auto begin=std::chrono::steady_clock::now();ck(cudaStreamSynchronize(s));
            // Only what the prefill displaced: the slot table says which planned units are still on the
            // device. Pass 1 holds those (no copies) so pass 2 can only evict prefill leftovers; pass 2
            // makes the full planned set resident on the layers that lost units (copies the missing ones).
            const auto v=api->view();std::vector<std::int32_t> base(v.n_gated),slot(std::size_t(v.n_gated)*v.units);
            ck(cudaMemcpy(base.data(),v.unit_base,base.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(slot.data(),v.slot_of,slot.size()*4,cudaMemcpyDeviceToHost));
            std::vector<std::vector<std::int32_t>> present(planned.size());std::vector<unsigned> missing(planned.size(),0);std::uint64_t displaced=0;
            for(unsigned g=0;g<planned.size()&&g<v.n_gated;++g)for(auto id:planned[g]){const auto k=std::size_t(base[g])+std::size_t(id);
                if(k<slot.size()&&slot[k]>=0)present[g].push_back(id);else{++missing[g];++displaced;}}
            std::vector<unsigned> held;std::uint64_t units=0,layers_copied=0;std::string refused;
            try{if(displaced){
                    for(unsigned g=0;g<planned.size()&&refused.empty();++g){if(present[g].empty())continue;
                        if(!site.try_make_resident(g,present[g].data(),unsigned(present[g].size()))){refused="gated layer "+std::to_string(g)+" hold (SS_E_CAPACITY)";break;}held.push_back(g);}
                    for(unsigned g=0;g<planned.size()&&refused.empty();++g){if(!missing[g])continue;
                        if(!site.try_make_resident(g,planned[g].data(),unsigned(planned[g].size()))){refused="gated layer "+std::to_string(g)+" (SS_E_CAPACITY)";break;}
                        if(present[g].empty())held.push_back(g);units+=missing[g];++layers_copied;}}}
            catch(...){for(auto g:held)try{site.release_resident(g);}catch(...){}throw;}
            for(auto g:held)site.release_resident(g);
            j.set("displaced_units",displaced);j.set("units_copied",units);j.set("bytes_copied",units*std::uint64_t(v.stride));j.set("layers_copied",layers_copied);
            j.set("ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());j.set("layers_held",std::uint64_t(held.size()));
            if(!refused.empty())j.set("refused",refused);
            // Every layer holds at least its open count again. (make_resident fills the free slots
            // first and evicts only what it needs, so up to the spare target of prefill units stay
            // and free is 0 until the next boundary defers its victims, as after any prefill.)
            auto c=census(s);bool restored=!open_persistent.empty();const auto& ls=c.find("layers")->as_array();
            for(std::size_t g=0;g<ls.size()&&g<open_persistent.size();++g)restored=restored&&std::uint32_t(ls[g].find("persistent")->as_int())>=open_persistent[g];
            j.set("persistent_at_least_open",restored);j.set("persistent_used",c.find("persistent_used")->as_int());j.set("persistent_free",c.find("persistent_free")->as_int());return j;}
    };
    void session_failed(){sess.fail();cudaStreamSynchronize(compute.s);cudaStreamSynchronize(decode.s);cudaStreamSynchronize(h2d.s);}
    Json begin_session(std::span<const U32> input){
        need(decode_mode,"service started without decode (head/session) support");SessionFrontier::validate_prompt(input,capacity);
        if(auto why=tier->unusable();!why.empty())throw std::runtime_error("expert cache unusable after a failed token (restart the worker): "+why);sess.reset();
        auto begin=std::chrono::steady_clock::now();in_session=true;
        // Same drain/ring reset as a failed prefill before any reuse.
        try{run(input,"glm-pf-rtx-native-v1",StateSink{});}catch(...){in_session=false;session_failed();
            for(auto& r:ring){r.layer=-1;if(r.read.valid())try{r.read.get();}catch(...){}cudaEventRecord(r.retired.e,compute.s);}cudaStreamSynchronize(compute.s);throw;}
        in_session=false;sess.begin(input);have_draft=false;spec_ctl=SpecControl{};routes_header("session-"+std::to_string(++route_sessions),input.size());auto j=Json::object();j.set("frontier",int(sess.frontier()));j.set("prefill_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());j.set("prefill_timings",last_timings);j.set("row_tile",int(tile));j.set("row_tile_choice",tile_choice);
        // The prefill's own expert cache traffic (counter delta since the last read: reset_open or the last token), so
        // file units read during prefill are visible per request and the first decode token counts only its own traffic.
        if(prefill_via_cache&&ts_tier&&tier.get()==ts_tier){try{j.set("prefill_expert_cache",ts_tier->collect_stats());}catch(const std::exception& e){j.set("prefill_expert_cache_error",std::string(e.what()));}}
        return j;
    }
    static void predecoded(Call& v,Layer& w){if(w.fp16_g.p){v.fp16_gate=w.fp16_g.as<U16>();v.fp16_up=w.fp16_u.as<U16>();v.fp16_down=w.fp16_d.as<U16>();}}
    void moe(unsigned l,Layer& w,const float* x,float* y,unsigned rows=1){
        if(rows<=4&&dopt.row_router){router_scores<<<dim3(9,rows),128,0,compute.s>>>(x,w.router,w.bias,route_sg.as<float>(),route_keys.as<U32>(),flag.as<unsigned>());router_select<<<rows,288,0,compute.s>>>(route_sg.as<float>(),route_keys.as<U32>(),ids.as<U32>(),rates.as<float>());}
        else router<<<rows,320,0,compute.s>>>(x,w.router,w.bias,ids.as<U32>(),rates.as<float>(),flag.as<unsigned>());
        timeline.mark(int(l),DecodePhase::router,compute.s);
        tier->acquire(l,ids.as<U32>(),8*rows,compute.s);timeline.mark(int(l),DecodePhase::export_wait,compute.s);
        if(ts_tier&&tier.get()==ts_tier&&ts_tier->predict){ts_tier->predict_next(l,x,compute.s,rows);timeline.mark(int(l),DecodePhase::predict,compute.s);}
        auto* d=static_cast<const Expert*>(tier->bind_selected_descriptors(l,compute.s));
        Call v{rows,288,4096,2048,x,ids.as<U32>(),rates.as<float>(),d,routed.as<float>()};v.compact_groups=dopt.skip_empty_expert_groups;v.row_gemv=dopt.row_moe;v.row_r2=dopt.row_r2;ck(enqueue_cutlass(v,workspace.p,ws_bytes,compute.s,exec));timeline.mark(int(l),DecodePhase::routed_moe,compute.s);
        Call sh{rows,1,4096,2048,x,nullptr,nullptr,w.shared_descriptor.as<Expert>(),shared.as<float>()};predecoded(sh,w);sh.row_gemv=dopt.row_moe;sh.row_r2=dopt.row_r2;ck(enqueue_cutlass(sh,workspace.p,ws_bytes,compute.s,exec));timeline.mark(int(l),DecodePhase::shared_expert,compute.s);
        add<<<16*rows,256,0,compute.s>>>(routed.as<float>(),shared.as<float>(),y,std::size_t(4096)*rows,flag.as<unsigned>());tier->release(l,compute.s);timeline.mark(int(l),DecodePhase::combine,compute.s);
    }
    // Session working KDA state (export/replay target); the prefill writes it through kst/kcv.
    float* kst2(unsigned l){return kda_s.as<float>()+std::size_t(kda_slot(l))*(kda_state_bytes/4);}
    float* kcv2(unsigned l){return kda_c.as<float>()+std::size_t(kda_slot(l))*(kda_conv_bytes/4);}
    // The trunk over `rows` consecutive tokens at positions P.. (already in `tokens`): embed, 45 layers, the
    // final norm, the head over every row and one argmax per row (argres + 4*j). rows = 1 is the plain step
    // (KDA promotes its checkpoint inline); rows > 1 is a verify: KDA replays the open block plus every row,
    // and the promotion is the caller's, after acceptance. Begins the tier token.
    void trunk(unsigned P,unsigned rows){
        embed<<<64*rows,256,0,compute.s>>>(u16("model.language_model.embed_tokens.weight"),tokens.as<U32>()+P,rx.as<float>(),rows,4);timeline.mark(-1,DecodePhase::embed,compute.s);
        if(ts_tier&&tier.get()==ts_tier){ts_tier->census_begin(compute.s);ts_tier->predict_rows(rows);}
        tier->begin_token(P);const unsigned n=kda_replay_rows(P)+rows-1;
        for(unsigned l=0;l<45;++l){auto& w=*layers[l];const float* yrow=part.as<float>();
            pre(w.attn,w.input,rx.as<float>(),normalized.as<float>(),rows);timeline.mark(int(l),DecodePhase::hc_pre_attn,compute.s);
            if(is_kda_layer(l)){
                ck(cudaMemcpyAsync(tl(l)+std::size_t(n-rows)*4096,normalized.p,rows*kda_row_bytes,cudaMemcpyDeviceToDevice,compute.s));
                ck(cudaMemcpyAsync(kst2(l),cks(l),kda_state_bytes,cudaMemcpyDeviceToDevice,compute.s));ck(cudaMemcpyAsync(kcv2(l),ckc(l),kda_conv_bytes,cudaMemcpyDeviceToDevice,compute.s));
                KdaCall v{n,w.k,tl(l),part.as<float>(),kst2(l),kcv2(l)};v.row_kernels=dopt.row_kda;v.row_r2=dopt.row_r2;ck(enqueue_kda(v,workspace.p,ws_bytes,compute.s));yrow=part.as<float>()+std::size_t(n-rows)*4096;
                if(rows==1&&kda_promotes(P))promote_kda(l);
                timeline.mark(int(l),DecodePhase::kda,compute.s);
            }else{DsaCall v{};v.rows=rows;v.first=P;v.w=w.d;v.state=sdstate(l);v.x=normalized.as<float>();v.y=part.as<float>();v.nonfinite=flag.as<unsigned>();v.index=DsaIndex::fp32;v.row_kernels=dopt.row_dsa;v.row_r2=dopt.row_r2;ck(enqueue_dsa(v,workspace.p,ws_bytes,compute.s));timeline.mark(int(l),DecodePhase::dsa,compute.s);}
            post(rx.as<float>(),yrow,rx2.as<float>(),rows);timeline.mark(int(l),DecodePhase::hc_post_attn,compute.s);
            pre(w.ffn,w.post,rx2.as<float>(),normalized.as<float>(),rows);timeline.mark(int(l),DecodePhase::hc_pre_ffn,compute.s);
            if(l<3){Call v{rows,1,4096,12288,normalized.as<float>(),nullptr,nullptr,w.shared_descriptor.as<Expert>(),part.as<float>()};predecoded(v,w);v.row_gemv=dopt.row_moe;v.row_r2=dopt.row_r2;ck(enqueue_cutlass(v,workspace.p,ws_bytes,compute.s,exec));timeline.mark(int(l),DecodePhase::dense_ffn,compute.s);}
            else moe(l,w,normalized.as<float>(),part.as<float>(),rows);
            post(rx2.as<float>(),part.as<float>(),rx.as<float>(),rows);timeline.mark(int(l),DecodePhase::hc_post_ffn,compute.s);
        }
        hc_final_kernel<<<dim3(16,rows),256,0,compute.s>>>(rx.as<float>(),part.as<float>());norm<<<rows,256,0,compute.s>>>(part.as<float>(),u16("model.language_model.norm.weight"),hrow.as<float>(),rows);
        ck(enqueue_head(head(),hrow.as<float>(),logits.as<float>(),rows,head_ws.p,head_ws.bytes,compute.s));
        for(unsigned j=0;j<rows;++j)ck(enqueue_argmax(logits.as<float>()+std::size_t(j)*vocab_size,vocab_size,argres.as<U32>()+4*j,compute.s));
    }
    void promote_kda(unsigned l){ck(cudaMemcpyAsync(cks(l),kst2(l),kda_state_bytes,cudaMemcpyDeviceToDevice,compute.s));ck(cudaMemcpyAsync(ckc(l),kcv2(l),kda_conv_bytes,cudaMemcpyDeviceToDevice,compute.s));}
    // One generation step. Same layer helpers, kernels and order as prefill:
    // KDA replays the open B32 block from its checkpoint; DSA appends rows=1 at
    // the absolute position; MTP pairs (hidden[P],y) on its own state.
    Json decode_one(bool qualification,bool draft=false){
        Json restored;if(restore_pending&&decode_mode&&sess.open())restored=run_pending_restore();
        auto out=decode_step(qualification,draft);if(restored.is_object())out.set("deferred_restore",restored);observe(out);return out;}
    void observe(const Json& out){if(!ts_tier||tier.get()!=ts_tier)return;auto* e=out.find("experts");auto* m=e?e->find("misses"):nullptr;if(m)ts_tier->observe_misses(std::uint64_t(m->as_int()));}
    Json decode_step(bool qualification,bool draft){
        need(decode_mode,"decode not enabled");const unsigned P=sess.position();const U32 pending=sess.pending();
        try{
            auto begin=std::chrono::steady_clock::now();timeline.start(compute.s);ck(cudaMemsetAsync(flag.p,0,4,compute.s));
            auto* hs=host_small.as<U32>();hs[0]=pending;ck(cudaMemcpyAsync(tokens.as<U32>()+P,hs,4,cudaMemcpyHostToDevice,compute.s));
            trunk(P,1);
            ck(cudaMemcpyAsync(hs,argres.p,12,cudaMemcpyDeviceToHost,compute.s));ck(cudaMemcpyAsync(hs+4,flag.p,4,cudaMemcpyDeviceToHost,compute.s));timeline.mark(-1,DecodePhase::head,compute.s);ck(cudaStreamSynchronize(compute.s));
            HeadChoice choice{hs[0],hs[1],0};std::memcpy(&choice.logit,hs+2,4);need(!hs[4],"nonfinite trunk activation; session invalidated");const U32 y=accept_choice(choice);
            auto record=Json::object();
            double digest_ms=0;
            if(qualification){auto digest_begin=std::chrono::steady_clock::now();std::vector<float> host(vocab_size);ck(cudaMemcpy(host.data(),logits.p,host.size()*4,cudaMemcpyDeviceToHost));auto again=head_argmax(host);
                need(again.token==choice.token&&again.nonfinite==0,"device/host argmax disagree");record.set("logits_sha256",hex_sha(host.data(),host.size()*4));
                std::vector<float> h(4096);ck(cudaMemcpy(h.data(),hrow.p,16384,cudaMemcpyDeviceToHost));record.set("hidden_sha256",hex_sha(h.data(),16384));
                digest_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-digest_begin).count();record.set("digest_ms",digest_ms);}
            auto trunk_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            // MTP maintenance (qualification state; does not choose the token).
            timeline.mark(-1,DecodePhase::host_gap,compute.s); // device idle while the host synchronized, checked and digested
            hs[8]=y;ck(cudaMemcpyAsync(tokens.as<U32>()+P+1,hs+8,4,cudaMemcpyHostToDevice,compute.s));
            embed<<<16,256,0,compute.s>>>(u16("model.language_model.embed_tokens.weight"),tokens.as<U32>()+P+1,emb.as<float>(),1,1);
            auto& w=*layers[45];
            MtpCall m1{1,P,w.m,sdstate(45),emb.as<float>(),hrow.as<float>(),mtp_out.as<float>(),flag.as<unsigned>(),DsaIndex::fp32};m1.phase=1;m1.route_ids=route.as<U32>();m1.row_proj=dopt.row_dsa;m1.row_moe=dopt.row_moe;m1.row_r2=dopt.row_r2;ck(enqueue_mtp(m1,workspace.p,ws_bytes,compute.s,exec));
            tier->acquire(45,route.as<U32>(),8,compute.s);auto m2=m1;m2.phase=2;m2.route_ids=nullptr;m2.w.experts=static_cast<const Expert*>(tier->bind_selected_descriptors(45,compute.s));ck(enqueue_mtp(m2,workspace.p,ws_bytes,compute.s,exec));
            tier->release(45,compute.s);if(draft)enqueue_draft(0,hs+12);ck(cudaMemcpyAsync(hs+4,flag.p,4,cudaMemcpyDeviceToHost,compute.s));timeline.mark(-1,DecodePhase::mtp,compute.s);ck(cudaStreamSynchronize(compute.s));ck(cudaGetLastError());need(!hs[4],"nonfinite MTP activation; session invalidated");
            if(draft)take_draft(hs+12);else have_draft=false;
            tier->end_token();sess.commit(y);
            std::string routes;
            if(ts_tier&&tier.get()==ts_tier&&ts_tier->route_log.p){std::vector<U32> h(std::size_t(routed_gated)*routed_topk);ck(cudaMemcpy(h.data(),ts_tier->route_log.p,h.size()*4,cudaMemcpyDeviceToHost));routes=routes_line(h.data());
                if(routes_file.is_open()){routes_file<<routes<<'\n';routes_file.flush();}}
            record.set("position",int(P));record.set("token",std::uint64_t(y));record.set("trunk_ms",trunk_ms);record.set("step_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());record.set("experts",tier->collect_stats());if(!routes.empty())record.set("routes",routes);
            if(timeline.on){record.set("timeline",timeline.collect());if(ts_tier&&tier.get()==ts_tier)record.set("slot_census",ts_tier->census(compute.s));}
            return record;
        }catch(const std::exception& e){tier->fail_token(e.what());session_failed();if(timeline.on)failed_timeline=timeline.collect_failed();failed_census();throw;}
        catch(...){tier->fail_token("non-standard exception");session_failed();if(timeline.on)failed_timeline=timeline.collect_failed();failed_census();throw;}
    }
    // MTP self-speculation (mtp_self_spec, K = 1). The draft for the next position is the shared head's argmax
    // over the MTP output of the last committed row (dlogits; the trunk's logits stay the digest source); it
    // stays on the device (argres + 16) and is read back with the cycle's last synchronization.
    void enqueue_draft(unsigned row,U32* host){
        ck(enqueue_head(head(),mtp_out.as<float>()+std::size_t(row)*4096,dlogits.as<float>(),1,head_ws.p,head_ws.bytes,compute.s));
        ck(enqueue_argmax(dlogits.as<float>(),vocab_size,argres.as<U32>()+16,compute.s));ck(cudaMemcpyAsync(host,argres.as<U32>()+16,12,cudaMemcpyDeviceToHost,compute.s));
    }
    void take_draft(const U32* h){have_draft=h[0]<vocab_size&&h[1]==0;draft_host=h[0];}
    // One cycle: verify [pending, d1..dK] at P..P+K with the plain step's per-row ops (trunk rows = K+1 <= 4),
    // accept greedily (the longest prefix of drafts equal to the targets, then the target after it), commit
    // m+1 tokens, MTP over the committed rows, next MTP draft. Drafts: the prompt lookup (spec_lookup, up to
    // spec_drafts tokens) when the session's suffix occurred before, else the MTP draft (one token). The rows stay
    // inside one KDA block. A plain step instead when no draft exists or one token is wanted. Rejected rows need no
    // undo: the KDA working state and the tail are rebuilt from the block checkpoint every step (promotion only
    // when the last committed row ends the block), and their DSA rows (and pools) are rewritten by later steps.
    // A non-finite verify flag falls back to the plain step at P (the flag cannot tell the rows apart).
    Json decode_spec(bool qualification,unsigned max_tokens){
        need(decode_mode&&dopt.mtp_spec,"decode_spec needs the service opened with mtp_self_spec");
        Json restored;if(restore_pending&&sess.open())restored=run_pending_restore();
        const bool plain=!dopt.spec_always&&spec_ctl.plain;const auto begin=std::chrono::steady_clock::now();Json out;
        if(plain){out=decode_step(qualification,spec_ctl.left==1); // the last plain step drafts for the next cycle
            auto t=Json::array();t.push(out.find("token")->as_int());out.set("tokens",t);
            if(qualification){auto g=Json::array();g.push(out.find("logits_sha256")->as_string());out.set("logits",g);}out.set("drafted",0);out.set("accepted",0);}
        else out=decode_spec_step(qualification,max_tokens);
        const auto n=out.find("tokens")->as_array().size();
        if(!dopt.spec_always){if(!(plain&&spec_ctl.left==1))spec_observe(plain,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/double(n));
            out.set("spec_path",plain?"plain":"verify");spec_next();}
        if(restored.is_object())out.set("deferred_restore",restored);observe(out);return out;}
    Json decode_spec_step(bool qualification,unsigned max_tokens){
        const unsigned P=sess.position();
        std::vector<U32> drafts;bool from_mtp=false;
        if(dopt.spec_lookup)drafts=lookup_draft(sess.tokens(),spec_drafts);
        if(drafts.empty()&&have_draft){drafts.assign(1,draft_host);from_mtp=true;}
        const unsigned R=std::min({unsigned(drafts.size())+1,kda_block-P%kda_block,max_tokens,capacity-1-P});
        if(R<2){auto r=decode_step(qualification,true);
            auto t=Json::array();t.push(r.find("token")->as_int());r.set("tokens",t);
            if(qualification){auto g=Json::array();g.push(r.find("logits_sha256")->as_string());r.set("logits",g);}r.set("drafted",0);r.set("accepted",0);return r;}
        drafts.resize(R-1);
        try{
            auto begin=std::chrono::steady_clock::now();timeline.start(compute.s);ck(cudaMemsetAsync(flag.p,0,4,compute.s));
            // host_small: argres rows 0..15, flag 24, verify tokens 32.., MTP tokens 40.., draft 48..50
            auto* hs=host_small.as<U32>();hs[32]=sess.pending();for(unsigned j=1;j<R;++j)hs[32+j]=drafts[j-1];
            if(from_mtp){ck(cudaMemcpyAsync(tokens.as<U32>()+P,hs+32,4,cudaMemcpyHostToDevice,compute.s));ck(cudaMemcpyAsync(tokens.as<U32>()+P+1,argres.as<U32>()+16,4,cudaMemcpyDeviceToDevice,compute.s));}
            else ck(cudaMemcpyAsync(tokens.as<U32>()+P,hs+32,4*R,cudaMemcpyHostToDevice,compute.s));
            trunk(P,R);
            ck(cudaMemcpyAsync(hs,argres.p,16*R,cudaMemcpyDeviceToHost,compute.s));ck(cudaMemcpyAsync(hs+24,flag.p,4,cudaMemcpyDeviceToHost,compute.s));timeline.mark(-1,DecodePhase::head,compute.s);ck(cudaStreamSynchronize(compute.s));
            if(hs[24]){tier->end_token();have_draft=false;return decode_spec_step(qualification,1);}
            auto choice=[&](unsigned j){HeadChoice c{hs[4*j],hs[4*j+1],0};std::memcpy(&c.logit,hs+4*j+2,4);return c;};
            std::vector<U32> out;unsigned m=0;U32 t=accept_choice(choice(0));
            while(m+1<R&&drafts[m]==t){out.push_back(drafts[m]);++m;t=accept_choice(choice(m));}
            out.push_back(t);const unsigned c=m+1; // committed rows P..P+m, emitted d1..dm then t
            auto record=Json::object();auto tok=Json::array(),dig=Json::array();for(auto y:out)tok.push(std::uint64_t(y));
            double digest_ms=0;
            if(qualification){auto digest_begin=std::chrono::steady_clock::now();std::vector<float> host(vocab_size);
                for(unsigned j=0;j<c;++j){ck(cudaMemcpy(host.data(),logits.as<float>()+std::size_t(j)*vocab_size,host.size()*4,cudaMemcpyDeviceToHost));auto again=head_argmax(host);
                    need(again.token==choice(j).token&&again.nonfinite==0,"device/host argmax disagree");dig.push(hex_sha(host.data(),host.size()*4));}
                record.set("logits",dig);digest_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-digest_begin).count();record.set("digest_ms",digest_ms);}
            auto trunk_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            timeline.mark(-1,DecodePhase::host_gap,compute.s);
            if(c==R&&kda_promotes(P+c-1))for(unsigned l=0;l<45;++l)if(is_kda_layer(l))promote_kda(l);
            // MTP over the committed rows: pairs (hidden[P+j], token P+1+j).
            for(unsigned j=0;j<c;++j)hs[40+j]=out[j];ck(cudaMemcpyAsync(tokens.as<U32>()+P+1,hs+40,4*c,cudaMemcpyHostToDevice,compute.s));
            embed<<<16*c,256,0,compute.s>>>(u16("model.language_model.embed_tokens.weight"),tokens.as<U32>()+P+1,emb.as<float>(),c,1);
            auto& w=*layers[45];
            MtpCall m1{c,P,w.m,sdstate(45),emb.as<float>(),hrow.as<float>(),mtp_out.as<float>(),flag.as<unsigned>(),DsaIndex::fp32};m1.phase=1;m1.route_ids=route.as<U32>();m1.row_proj=dopt.row_dsa;m1.row_moe=dopt.row_moe;m1.row_r2=dopt.row_r2;ck(enqueue_mtp(m1,workspace.p,ws_bytes,compute.s,exec));
            tier->acquire(45,route.as<U32>(),8*c,compute.s);auto m2=m1;m2.phase=2;m2.route_ids=nullptr;m2.w.experts=static_cast<const Expert*>(tier->bind_selected_descriptors(45,compute.s));ck(enqueue_mtp(m2,workspace.p,ws_bytes,compute.s,exec));
            tier->release(45,compute.s);enqueue_draft(c-1,hs+48);ck(cudaMemcpyAsync(hs+24,flag.p,4,cudaMemcpyDeviceToHost,compute.s));timeline.mark(-1,DecodePhase::mtp,compute.s);ck(cudaStreamSynchronize(compute.s));ck(cudaGetLastError());need(!hs[24],"nonfinite MTP activation; session invalidated");
            take_draft(hs+48);tier->end_token();for(auto y:out)sess.commit(y);
            record.set("position",int(P));record.set("tokens",tok);record.set("token",std::uint64_t(out[0]));record.set("drafted",int(R-1));record.set("accepted",int(m));record.set("draft_source",from_mtp?"mtp":"lookup");
            record.set("trunk_ms",trunk_ms);record.set("step_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());record.set("experts",tier->collect_stats());
            if(timeline.on){record.set("timeline",timeline.collect());if(ts_tier&&tier.get()==ts_tier)record.set("slot_census",ts_tier->census(compute.s));}
            return record;
        }catch(const std::exception& e){have_draft=false;tier->fail_token(e.what());session_failed();if(timeline.on)failed_timeline=timeline.collect_failed();failed_census();throw;}
        catch(...){have_draft=false;tier->fail_token("non-standard exception");session_failed();if(timeline.on)failed_timeline=timeline.collect_failed();failed_census();throw;}
    }
    void export_session(const StateSink& sink){
        need(decode_mode&&sess.open(),"no open session to export");const unsigned P=sess.frontier();ck(cudaStreamSynchronize(compute.s));
        ck(cudaMemsetAsync(flag.p,0,4,compute.s));in_session=true;try{for(unsigned l=0;l<46;++l)export_state(l,P,sink);}catch(...){in_session=false;throw;}in_session=false;
    }
    std::string replay_digest(){
        need(decode_mode&&sess.open(),"no open session");const unsigned P=sess.frontier(),open=kda_open_rows(P);strix::Sha256Fast sha;std::vector<std::byte> buf(kda_state_bytes);
        for(unsigned l=0;l<45;++l)if(is_kda_layer(l))for(auto [p,n]:{std::pair<const void*,std::size_t>{cks(l),kda_state_bytes},{ckc(l),kda_conv_bytes},{tl(l),open*kda_row_bytes}}){
            if(!n)continue;ck(cudaMemcpy(buf.data(),p,n,cudaMemcpyDeviceToHost));sha.update(std::span(buf.data(),n));}
        return strix::model_content_identity_hex(sha.finalize());
    }
    std::vector<Buffer*> session_buffers(){return {&kda_s,&kda_c,&ck_s,&ck_c,&tail,&dlat,&dkey,&dgate,&dpool};}
    void snapshot_session(){need(decode_mode&&sess.open(),"no open session");ck(cudaStreamSynchronize(compute.s));snapshot.clear();
        for(auto* b:session_buffers()){snapshot.emplace_back(b->bytes);ck(cudaMemcpy(snapshot.back().data(),b->p,b->bytes,cudaMemcpyDeviceToHost));}snapshot_tokens=sess.tokens();}
    void restore_session(){need(decode_mode&&!snapshot.empty(),"no session snapshot");have_draft=false;ck(cudaStreamSynchronize(compute.s));std::size_t j=0;
        for(auto* b:session_buffers())ck(cudaMemcpy(b->p,snapshot[j++].data(),b->bytes,cudaMemcpyHostToDevice));sess.reset();sess.begin(snapshot_tokens);}
};
Runtime::Runtime(const std::string& path,unsigned cap,unsigned rows,bool source,bool graphs,unsigned floor_gib,bool legacy,bool serial,const VerifiedManifest* verified,unsigned max_device,unsigned max_pinned,bool require_file,bool decode,const DecodeOptions& dopts):p_(std::make_unique<Impl>(path,cap,rows,source,graphs,floor_gib,verified,max_device,max_pinned,require_file,decode,dopts)){
    p_->allocated_tile=p_->tile;
    if(decode)need(!legacy,"decode identity uses the base FP32 index; --legacy-fp32-index is a prefill diagnostic only");p_->legacy_index=legacy;p_->serial_stage=serial;p_->profile.set("legacy_fp32_index",legacy);p_->profile.set("serial_stage",serial);}
Runtime::~Runtime()=default;
unsigned Runtime::prefill_row_tile()const{return p_->tile;}
unsigned Runtime::allocated_prefill_row_tile()const{return p_->allocated_tile;}
void Runtime::use_prefill_row_tile(unsigned t){if(!t){p_->requested_tile=0;return;}
    need(valid_schedule(p_->capacity,t)&&t<=p_->allocated_tile,"row tile must be 32-aligned and at most the allocated tile");p_->requested_tile=t;p_->tile=t;}
unsigned Runtime::requested_prefill_row_tile()const{return p_->requested_tile;}
void Runtime::set_prefill_tile_crossover(unsigned rows){need(rows<=max_capacity,"prefill tile crossover: 0 (off) .. 131072 rows");p_->tile_crossover=rows;}
unsigned Runtime::prefill_tile_crossover()const{return p_->tile_crossover;}
void Runtime::Impl::prefill_begin(unsigned l){need(ts_tier!=nullptr,"prefill through the cache needs the open tierstream cache");ts_tier->prefill_begin(l,layers[l]->descriptors,compute.s);}
void Runtime::Impl::prefill_end(unsigned l){need(ts_tier!=nullptr,"prefill through the cache needs the open tierstream cache");ts_tier->prefill_end(l,compute.s);}
std::string Runtime::plan_budget(const std::string& model,unsigned cap,unsigned rows,unsigned floor_gib,const DecodeOptions& d,std::uint64_t device_total,std::uint64_t device_free){
    need(d.expert_tier=="tierstream","--plan-budget plans the tierstream decode budget");
    need(device_free<=device_total,"device figures");
    Shards shards(model,false,true); // headers and sizes only: no source hashing
    auto [physical,commit]=host_free();
    // Context = the given device use (total - free); the process baseline is this
    // planning process's own private bytes (a lower bound for the service's).
    const bool whole=d.vram_total_gib>=0||d.ram_total_gib>=0;const std::uint64_t hard=std::min<std::uint64_t>(device_total,99860000000ULL);
    // The same tile choice as the service: the largest admitted tile unless --row-tile.
    auto tp=Impl::plan_tile(shards.manifest,cap,rows,d,device_total-device_free,host_budget(physical,commit,floor_gib),Impl::process_host_bytes(),!tierstream_supports_make_resident(),hard);
    const unsigned tile=tp.tile;auto& j=tp.json;const auto& refused=tp.refused;
    if(!refused.empty()){j.set("verdict","REFUSED");j.set("refuse",refused);}
    j.set("mode",std::string("plan only: shard headers and sizes (no hashing, no unit-file read, no CUDA calls); ")+(whole?"whole-device caps":"device room (no whole-device cap)"));j.set("tierstream_commit",tierstream_commit());j.set("capacity",int(cap));j.set("row_tile",int(tile));j.set("host_floor_gib",int(floor_gib));
    j.set("prefill_experts",tierstream_supports_make_resident()?"through the tierstream cache (make_resident per layer; no prefill rings; one copy of every expert)":
        "INTERIM file tier only (two prefill rings; the linked tierstream has no make_resident)");
    j.set("cuda_driver_loaded",GetModuleHandleW(L"nvcuda.dll")!=nullptr);return j.dump();
}
std::string Runtime::memory_report(const std::string& stage){return p_->memory_json(stage).dump();}
void Runtime::set_decode_profile(bool on){if(on)p_->timeline.enable();else p_->timeline.on=false;if(p_->ts_tier){if(on)p_->ts_tier->enable_census();else p_->ts_tier->census_on=false;}}
void Runtime::set_row_kernels(const std::string& v){ck(cudaStreamSynchronize(p_->compute.s));set_row_kernels_value(p_->dopt,v);}
std::string Runtime::row_kernels()const{return row_kernels_name(p_->dopt);}
std::string Runtime::expert_diagnostics(){auto j=Json::object();if(!p_->ts_tier)return j.dump();j=p_->ts_tier->api->diagnostics();
    if(p_->failed_timeline.find("timeline"))j.set("decode_timeline_at_failure",p_->failed_timeline);
    if(p_->failed_slots.find("layers"))j.set("slot_census_at_failure",p_->failed_slots);
    if(auto why=p_->ts_tier->unusable();!why.empty())j.set("retired",why);return j.dump();}
std::string Runtime::save_residency(){return p_->save_residency();}
std::string Runtime::model_sha256()const{return p_->model_hash;}
std::string Runtime::timings()const{return p_->last_timings.dump();}
std::string Runtime::self_check(std::span<const U32> input,const std::string& identity,unsigned reference_tile){
    if(p_->allocated_tile)p_->tile=p_->allocated_tile; // the candidate is the allocation, whatever the last request chose
    need(valid_schedule(p_->capacity,reference_tile)&&reference_tile<=p_->tile,"self-check baseline must fit allocated tile");
    struct Snapshot{unsigned rank,layer,component;std::vector<std::byte> bytes;};std::vector<Snapshot> old;
    struct Pin{Impl& i;Pin(Impl& x):i(x){i.pin_tile=true;}~Pin(){i.pin_tile=false;}} pin{*p_};const unsigned candidate_tile=p_->tile;const bool candidate_serial=p_->serial_stage,candidate_legacy=p_->legacy_index;auto report=Json::object();report.set("residency",p_->profile);
    strix::Sha256Fast old_hash[2],new_hash[2];
    try{
        p_->tile=reference_tile;p_->serial_stage=true;p_->legacy_index=true;
        prefill(input,identity,[&](unsigned rank,unsigned layer,unsigned component,std::span<const std::byte> bytes){old_hash[rank].update(bytes);old.push_back({rank,layer,component,{bytes.begin(),bytes.end()}});});
        report.set("old_timings",p_->last_timings);
        std::vector<std::byte> old_hidden(std::size_t(input.size()-1)*4096*4);ck(cudaMemcpy(old_hidden.data(),p_->hidden.p,old_hidden.size(),cudaMemcpyDeviceToHost));
        p_->tile=candidate_tile;p_->serial_stage=candidate_serial;p_->legacy_index=candidate_legacy;
        std::size_t next=0;std::uint64_t different=0,compared=0,memcmp_failures=0;
        auto compare=[&](std::span<const std::byte> a,std::span<const std::byte> b){need(a.size()==b.size(),"self-check state extent");memcmp_failures+=std::memcmp(a.data(),b.data(),a.size())!=0;
            for(std::size_t j=0;j<a.size();++j)different+=a[j]!=b[j];compared+=a.size();};
        prefill(input,identity,[&](unsigned rank,unsigned layer,unsigned component,std::span<const std::byte> bytes){new_hash[rank].update(bytes);need(next<old.size(),"extra self-check component");auto& ref=old[next++];need(ref.rank==rank&&ref.layer==layer&&ref.component==component,"self-check state order");compare(ref.bytes,bytes);});
        need(next==old.size(),"missing self-check component");
        std::vector<std::byte> new_hidden(old_hidden.size());ck(cudaMemcpy(new_hidden.data(),p_->hidden.p,new_hidden.size(),cudaMemcpyDeviceToHost));compare(old_hidden,new_hidden);
        for(unsigned rank=0;rank<2;++rank){report.set("old_rank"+std::to_string(rank)+"_sha256",strix::model_content_identity_hex(old_hash[rank].finalize()));report.set("new_rank"+std::to_string(rank)+"_sha256",strix::model_content_identity_hex(new_hash[rank].finalize()));}
        report.set("digest_layout","rank-local state payloads in release order, no wire headers");report.set("tokens",std::uint64_t(input.size()));
        report.set("new_timings",p_->last_timings);report.set("old_row_tile",int(reference_tile));report.set("new_row_tile",int(candidate_tile));report.set("compared_bytes",compared);report.set("differing_bytes",different);report.set("memcmp_failures",memcmp_failures);report.set("verdict",different?"MISMATCH":"EXACT");report.set("qualification","DRAFT");
        return report.dump();
    }catch(...){p_->tile=candidate_tile;p_->serial_stage=candidate_serial;p_->legacy_index=candidate_legacy;throw;}
}
std::string Runtime::health()const{auto j=Json::object();j.set("status","ready");j.set("identity","glm-pf-rtx-native-v1");j.set("index_identity",dsa_identity);j.set("qualification","DRAFT");j.set("capacity",int(p_->capacity));j.set("row_tile",int(p_->allocated_tile?p_->allocated_tile:p_->tile));j.set("prefill_tile_crossover",int(p_->tile_crossover));j.set("small_prefill_tile",int(std::min(small_prefill_tile,p_->allocated_tile?p_->allocated_tile:p_->tile)));j.set("model_sha256",p_->model_hash);j.set("cache_key",str(p_->manifest,"cache_key"));j.set("residency",p_->profile);
    if(p_->decode_mode){j.set("decode_identity",decode_identity);j.set("generation","greedy only (no sampling fields)");j.set("max_new_tokens_rule","prompt_length + max_new_tokens <= capacity");}
    return j.dump();}
bool Runtime::decode_capable()const{return p_->decode_mode;}
std::string Runtime::begin_session(std::span<const U32> t){return p_->begin_session(t).dump();}
std::string Runtime::decode_one(bool q){return p_->decode_one(q).dump();}
std::string Runtime::decode_spec(bool q,unsigned max_tokens){return p_->decode_spec(q,max_tokens).dump();}
void Runtime::profiler(bool on){ck(cudaStreamSynchronize(p_->compute.s));ck(on?cudaProfilerStart():cudaProfilerStop());}
void Runtime::set_spec_lookup(bool on){need(!on||p_->dopt.mtp_spec,"lookup drafts need the service opened with mtp_self_spec");p_->dopt.spec_lookup=on;}
void Runtime::export_session(const StateSink& sink){p_->export_session(sink);}
std::string Runtime::replay_digest(){return p_->replay_digest();}
std::vector<std::uint32_t> Runtime::session_tokens()const{return p_->sess.tokens();}
void Runtime::snapshot_session(){p_->snapshot_session();}
void Runtime::restore_session(){p_->restore_session();}
void Runtime::reset_session(){p_->sess.reset();}
void Runtime::select_expert_tier(const std::string& name){need(p_->decode_mode,"decode not enabled");p_->select_tier(name);p_->sess.reset();}
std::string Runtime::expert_tier()const{return p_->tier?p_->tier->backend():"none";}
void Runtime::reset_expert_cache(bool open){need(p_->decode_mode,"decode not enabled");p_->tier->reset_cache(open);if(open)p_->restore_pending=false;}
std::string Runtime::restore_decode_residency(){auto r=p_->run_pending_restore();return r.is_object()?r.dump():std::string("{}");}
bool Runtime::decode_restore_pending()const{return p_->restore_pending;}
std::string Runtime::generate(std::span<const U32> prompt,unsigned max_new,std::span<const U32> stop,bool qualification){
    need(p_->decode_mode,"decode not enabled");SessionFrontier::validate_prompt(prompt,p_->capacity);SessionFrontier::validate_generation(prompt.size(),max_new,p_->capacity);
    for(auto t:stop)need(t<vocab_size,"stop token outside vocabulary");
    auto receipt=Json::object();receipt.set("identity",decode_identity);receipt.set("qualification","DRAFT");receipt.set("generation","greedy");receipt.set("prompt_tokens",std::uint64_t(prompt.size()));receipt.set("max_new_tokens",int(max_new));
    auto start=std::chrono::steady_clock::now();auto prefill=p_->begin_session(prompt);receipt.set("prefill",prefill);
    auto loop=std::chrono::steady_clock::now();auto steps=Json::array(),generated=Json::array();std::string reason="max_new_tokens";double first_ms=0;
    std::uint64_t logical=0,h2d=0,file=0,sel[3]{},lookups=0,misses=0,file_reads=0,pinned_hits=0,device_hits=0;
    // mtp_self_spec: each step is one verify cycle emitting 1..4 tokens (the plain greedy tokens); tokens after a stop
    // token are not emitted (the session frontier may lie past it).
    const bool spec=p_->dopt.mtp_spec;bool stopped=false;
    for(unsigned i=0;i<max_new&&!stopped;){auto r=spec?p_->decode_spec(qualification,max_new-i):p_->decode_one(qualification);
        if(i==0)first_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const auto& e=*r.find("experts");auto num=[&](const char* k){auto* v=e.find(k);return v?std::uint64_t(v->as_int()):0ULL;};
        logical+=num("logical_bytes");h2d+=num("h2d_bytes");file+=num("file_bytes");sel[0]+=num("device_selections");sel[1]+=num("pinned_selections");sel[2]+=num("file_selections");lookups+=num("lookups");misses+=num("misses");file_reads+=num("file_reads");pinned_hits+=num("pinned_hits");device_hits+=num("device_hits");
        std::vector<std::uint32_t> ys;if(spec)for(const auto& t:r.find("tokens")->as_array())ys.push_back(std::uint32_t(t.as_int()));else ys.push_back(std::uint32_t(r.find("token")->as_int()));
        steps.push(r);
        for(auto y:ys){generated.push(std::uint64_t(y));++i;if(std::find(stop.begin(),stop.end(),y)!=stop.end()){reason="stop_token";stopped=true;break;}}}
    auto end=std::chrono::steady_clock::now();double loop_s=std::chrono::duration<double>(end-loop).count();auto n=generated.as_array().size();
    receipt.set("generated_token_ids",generated);receipt.set("stop_reason",reason);receipt.set("emitted_tokens",std::uint64_t(n));
    receipt.set("first_token_ms",first_ms);receipt.set("generation_loop_s",loop_s);receipt.set("tokens_per_s",loop_s>0?double(n)/loop_s:0.0);receipt.set("total_ms",std::chrono::duration<double,std::milli>(end-start).count());
    auto t=Json::object();t.set("logical_compressed_bytes",logical);t.set("h2d_bytes",h2d);t.set("file_read_bytes",file);t.set("device_selections",sel[0]);t.set("pinned_selections",sel[1]);t.set("file_selections",sel[2]);
    t.set("logical_bytes_per_token",n?double(logical)/double(n):0.0);t.set("h2d_bytes_per_token",n?double(h2d)/double(n):0.0);t.set("file_bytes_per_token",n?double(file)/double(n):0.0);t.set("scope","generation loop only; startup/prefill traffic excluded");
    if(lookups){t.set("cache_lookups",lookups);t.set("cache_misses",misses);t.set("misses_per_token",n?double(misses)/double(n):0.0);t.set("device_hits",device_hits);t.set("pinned_hits",pinned_hits);t.set("file_reads",file_reads);}
    receipt.set("expert_traffic",t);receipt.set("steps",steps);receipt.set("expert_tier",p_->tier->backend());receipt.set("fetched_expert_cache",std::string(p_->tier->backend())=="reference"?"none":"tierstream");
    receipt.set("final_frontier",int(p_->sess.frontier()));return receipt.dump();
}
void Runtime::prefill(std::span<const U32> t,const std::string& id,const StateSink& sink){try{p_->run(t,id,sink);}catch(...){
    // A failed request can leave work on either stream. Drain all producers
    // before any persistent ring/event/workspace is reused by the next request.
    cudaStreamSynchronize(p_->compute.s);cudaStreamSynchronize(p_->decode.s);cudaStreamSynchronize(p_->h2d.s);
    for(auto& r:p_->ring){r.layer=-1;if(r.read.valid())try{r.read.get();}catch(...){}cudaEventRecord(r.retired.e,p_->compute.s);}
    cudaStreamSynchronize(p_->compute.s);throw;
}}
}
