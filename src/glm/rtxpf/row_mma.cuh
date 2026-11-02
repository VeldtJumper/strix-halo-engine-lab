#pragma once
// Few-row trellis projections without an FP16 weight buffer. The packed 16x16 tile is stored in
// m16n8k16 B-fragment order (c1_position: lane = n%8*4 + k%8/2, element = k%2 + 2*(k/8) + 4*(n/8)),
// so each lane decodes its eight weights straight into the two B fragments of a 16-column strip.
// One warp owns a strip; every output is one m16n8k16 chain in ascending K from +0 with no split-K,
// which is how the pinned CUTLASS TensorOp kernels (16x8x16 instructions, K tiles of 32, no split-K)
// accumulate it, so the FP32 results carry the same bits for any row count.
#include "native.cuh"
#include "c1_layout.cuh"

namespace strix::glm::rtxpf {
__device__ __forceinline__ void row_mma(float (&c)[4],U32 a0,U32 a1,U32 a2,U32 a3,U32 b0,U32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
// a: FP16 rows [rows][k] (row stride k), d: FP32 [rows][n]; MT 16-row tiles cover rows (<= 16*MT).
// words: this warp's 48-word staging. k/16 must be a multiple of 4. Pre (row-kernels round 2, opt-in): the A
// fragments are loaded before the tile's weight decode instead of after it (same values).
template<unsigned Bits,unsigned MT,bool Pre>
__device__ __forceinline__ void strip_rows(const U16* packed,const U16* a,float* d,unsigned rows,unsigned k,unsigned n,
    unsigned strip,U32* words) {
    constexpr unsigned PF=4; // tiles in flight per warp
    const unsigned lane=threadIdx.x%32,g=lane/4,tig=lane%4,kts=k/16;
    const uint4* base=reinterpret_cast<const uint4*>(packed)+std::size_t(strip)*2*Bits;
    const std::size_t step=std::size_t(n/16)*2*Bits; // next K tile of the same strip
    float c[MT][2][4]={};
    uint4 buf[PF];
    #pragma unroll
    for(unsigned p=0;p<PF;++p) if(lane<2*Bits) buf[p]=__ldg(base+p*step+lane);
    for(unsigned kt=0;kt<kts;kt+=PF) {
        #pragma unroll
        for(unsigned p=0;p<PF;++p) {
            const unsigned k0=(kt+p)*16+2*tig;
            U32 af[MT][4];
            auto load_a=[&]{
                #pragma unroll
                for(unsigned m=0;m<MT;++m) {
                    const unsigned r0=16*m+g,r1=r0+8;
                    const U32* x0=reinterpret_cast<const U32*>(a+std::size_t(r0)*k+k0);
                    const U32* x1=reinterpret_cast<const U32*>(a+std::size_t(r1)*k+k0);
                    if constexpr(Pre){af[m][0]=r0<rows?__ldg(x0):0u;af[m][2]=r0<rows?__ldg(x0+4):0u;af[m][1]=r1<rows?__ldg(x1):0u;af[m][3]=r1<rows?__ldg(x1+4):0u;}
                    else {af[m][0]=r0<rows?x0[0]:0u;af[m][2]=r0<rows?x0[4]:0u;af[m][1]=r1<rows?x1[0]:0u;af[m][3]=r1<rows?x1[4]:0u;}
                }
            };
            if constexpr(Pre)load_a();
            if(lane<2*Bits) {
                reinterpret_cast<uint4*>(words)[lane]=buf[p];
                if(kt+p+PF<kts) buf[p]=__ldg(base+(kt+p+PF)*step+lane);
            }
            __syncwarp();
            U32 b[4];
            #pragma unroll
            for(unsigned r=0;r<4;++r)
                b[r]=U32(book(c1_state<Bits>(words,lane*8+2*r)))|U32(book(c1_state<Bits>(words,lane*8+2*r+1)))<<16;
            __syncwarp(); // every lane decoded before the staging is reused
            if constexpr(!Pre)load_a();
            #pragma unroll
            for(unsigned m=0;m<MT;++m) {
                row_mma(c[m][0],af[m][0],af[m][1],af[m][2],af[m][3],b[0],b[1]);
                row_mma(c[m][1],af[m][0],af[m][1],af[m][2],af[m][3],b[2],b[3]);
            }
        }
    }
    #pragma unroll
    for(unsigned m=0;m<MT;++m)
        #pragma unroll
        for(unsigned h=0;h<2;++h) {
            const unsigned col=strip*16+8*h+2*tig,r0=16*m+g,r1=r0+8;
            if(r0<rows) *reinterpret_cast<float2*>(d+std::size_t(r0)*n+col)=make_float2(c[m][h][0],c[m][h][1]);
            if(r1<rows) *reinterpret_cast<float2*>(d+std::size_t(r1)*n+col)=make_float2(c[m][h][2],c[m][h][3]);
        }
}
template<unsigned MT,bool Pre>
__device__ __forceinline__ void strip_rows_any(const Matrix& w,const U16* a,float* d,unsigned rows,unsigned k,unsigned n,unsigned strip,U32* words) {
    switch(w.bits) {
        case 3:strip_rows<3,MT,Pre>(w.packed,a,d,rows,k,n,strip,words);break;
        case 4:strip_rows<4,MT,Pre>(w.packed,a,d,rows,k,n,strip,words);break;
        case 5:strip_rows<5,MT,Pre>(w.packed,a,d,rows,k,n,strip,words);break;
        case 6:strip_rows<6,MT,Pre>(w.packed,a,d,rows,k,n,strip,words);break;
    }
}
// One matrix, rows <= 32: y[rows][w.output] = a[rows][w.input] x W (raw, before the output transform).
template<unsigned MT,bool Pre>
__global__ __launch_bounds__(128) void row_project(Matrix w,const U16* a,float* y,unsigned rows) {
    __shared__ __align__(16) U32 staging[4][48];
    const unsigned strip=blockIdx.x*4+threadIdx.x/32;
    if(strip>=w.output/16) return;
    strip_rows_any<MT,Pre>(w,a,y,rows,w.input,w.output,strip,staging[threadIdx.x/32]);
}
// Dense FP16 weights [n][k] (the CUTLASS projection's column-major B): y[rows][n] = a[rows][k] x W^T as the same
// ascending-K m16n8k16 chains; each B fragment pair is one 32-bit load (k, k+1 adjacent in a weight row).
template<unsigned MT>
__global__ __launch_bounds__(128) void row_dense(const U16* w,const U16* a,float* y,unsigned rows,unsigned k,unsigned n) {
    const unsigned strip=blockIdx.x*4+threadIdx.x/32;
    if(strip>=n/16) return;
    const unsigned lane=threadIdx.x%32,g=lane/4,tig=lane%4;
    float c[MT][2][4]={};
    #pragma unroll 4
    for(unsigned kt=0;kt<k/16;++kt) {
        const unsigned k0=kt*16+2*tig;
        U32 b[4];
        #pragma unroll
        for(unsigned h=0;h<2;++h) {
            const U32* wn=reinterpret_cast<const U32*>(w+std::size_t(strip*16+8*h+g)*k+k0);
            b[2*h]=__ldg(wn);b[2*h+1]=__ldg(wn+4);
        }
        #pragma unroll
        for(unsigned m=0;m<MT;++m) {
            const unsigned r0=16*m+g,r1=r0+8;
            const U32* x0=reinterpret_cast<const U32*>(a+std::size_t(r0)*k+k0);
            const U32* x1=reinterpret_cast<const U32*>(a+std::size_t(r1)*k+k0);
            const U32 a0=r0<rows?__ldg(x0):0u,a2=r0<rows?__ldg(x0+4):0u,a1=r1<rows?__ldg(x1):0u,a3=r1<rows?__ldg(x1+4):0u;
            row_mma(c[m][0],a0,a1,a2,a3,b[0],b[1]);
            row_mma(c[m][1],a0,a1,a2,a3,b[2],b[3]);
        }
    }
    #pragma unroll
    for(unsigned m=0;m<MT;++m)
        #pragma unroll
        for(unsigned h=0;h<2;++h) {
            const unsigned col=strip*16+8*h+2*tig,r0=16*m+g,r1=r0+8;
            if(r0<rows) *reinterpret_cast<float2*>(y+std::size_t(r0)*n+col)=make_float2(c[m][h][0],c[m][h][1]);
            if(r1<rows) *reinterpret_cast<float2*>(y+std::size_t(r1)*n+col)=make_float2(c[m][h][2],c[m][h][3]);
        }
}
inline bool row_dense_fits(unsigned rows,unsigned k,unsigned n){return rows>=1&&rows<=32&&k%16==0&&n%16==0;}
inline void launch_row_dense(const U16* w,const U16* a,float* y,unsigned rows,unsigned k,unsigned n,cudaStream_t s) {
    const unsigned ctas=(n/16+3)/4;
    if(rows<=16) row_dense<1><<<ctas,128,0,s>>>(w,a,y,rows,k,n);
    else row_dense<2><<<ctas,128,0,s>>>(w,a,y,rows,k,n);
}
inline bool row_project_fits(const Matrix& w,unsigned rows){return rows>=1&&rows<=32&&w.input%64==0&&w.output%16==0;}
inline void launch_row_project(const Matrix& w,const U16* a,float* y,unsigned rows,cudaStream_t s,bool r2=false) {
    const unsigned ctas=(w.output/16+3)/4;
    if(r2){if(rows<=16) row_project<1,true><<<ctas,128,0,s>>>(w,a,y,rows);else row_project<2,true><<<ctas,128,0,s>>>(w,a,y,rows);}
    else if(rows<=16) row_project<1,false><<<ctas,128,0,s>>>(w,a,y,rows);
    else row_project<2,false><<<ctas,128,0,s>>>(w,a,y,rows);
}
}  // namespace strix::glm::rtxpf
