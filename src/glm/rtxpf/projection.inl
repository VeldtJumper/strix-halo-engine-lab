// Shared by KDA, DSA and MTP: pinned CUTLASS TensorOp trellis projections.
using Half=cutlass::half_t;
using Row=cutlass::layout::RowMajor;
using Col=cutlass::layout::ColumnMajor;
// Same pinned TensorOp family/tile/precision as the MoE CUTLASS hot path.
template<class Layout>
using ProjectionKernel=typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Half,Row,cutlass::ComplexTransform::kNone,8,
    Half,Layout,cutlass::ComplexTransform::kNone,8,
    float,Row,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<16,8,16>,
    cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
template<class Layout>using Projection=cutlass::gemm::device::GemmGrouped<ProjectionKernel<Layout>>;
struct Problem {
    cutlass::gemm::GemmCoord size;
    Half *a,*b;float *d;
    int64_t lda,ldb,ldd;
};
__device__ float bf(U16 u){return __uint_as_float(unsigned(u)<<16);}
__device__ float sig(float x){return 1.f/(1.f+declared_exp(-x));}
__device__ void h128(float (&v)[4],unsigned lane) {
    for(unsigned s=1;s<32;s*=2)for(unsigned j=0;j<4;++j){float p=__shfl_xor_sync(0xffffffff,v[j],s);v[j]=(lane&s)?__fsub_rn(p,v[j]):__fadd_rn(v[j],p);}
    for(unsigned s=1;s<4;s*=2)for(unsigned j=0;j<4;++j)if(!(j&s)){float a=v[j],b=v[j+s];v[j]=a+b;v[j+s]=a-b;}
    for(unsigned j=0;j<4;++j)v[j]*=0x1.6a09e6p-4f;
}
__global__ void input_transform(const float* x,U16* a,const U16* suh,unsigned rows,unsigned width) {
    unsigned block=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(block>=rows*(width/128))return;
    unsigned col=(block%(width/128))*128;std::size_t at=std::size_t(block/(width/128))*width+col;
    float v[4];for(unsigned j=0;j<4;++j)v[j]=hf(fh(x[at+lane+32*j]*hf(suh[col+lane+32*j])));
    h128(v,lane);for(unsigned j=0;j<4;++j)a[at+lane+32*j]=fh(v[j]);
}
__global__ void output_transform(const float* x,float* y,const U16* svh,unsigned rows,unsigned width) {
    unsigned block=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(block>=rows*(width/128))return;
    unsigned col=(block%(width/128))*128;std::size_t at=std::size_t(block/(width/128))*width+col;
    float v[4];for(unsigned j=0;j<4;++j)v[j]=x[at+lane+32*j];h128(v,lane);
    for(unsigned j=0;j<4;++j)y[at+lane+32*j]=v[j]*hf(svh[col+lane+32*j]);
}
__global__ void round_input(const float* x,U16* a,std::size_t n){auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j<n)a[j]=fh(x[j]);}
// Reference uses original FP32 input for ordinary matrices. Trellis operands
// retain their declared transform/RN16 boundaries in both arms.
__global__ void projection_reference(const float* x,const U16* a,const U16* w,float* y,unsigned rows,unsigned in,unsigned out,bool ordinary) {
    auto j=std::size_t(blockIdx.x)*256+threadIdx.x;if(j>=std::size_t(rows)*out)return;
    unsigned r=unsigned(j/out),n=unsigned(j%out);float v=0;
    for(unsigned k=0;k<in;++k)v=__fmaf_rn(ordinary?x[std::size_t(r)*in+k]:hf(a[std::size_t(r)*in+k]),hf(w[ordinary?std::size_t(n)*in+k:std::size_t(k)*out+n]),v);
    y[j]=v;
}
// 8 rows x 32 columns. Each thread owns two complete, independent K chains.
// Shared operands preserve their exact FP32/FP16 values; no partial reductions,
// tensor products, or rounding points are introduced by this scheduling change.
__global__ __launch_bounds__(128) void projection_fp32_tiled(const float* x,const U16* a,const U16* w,float* y,unsigned rows,unsigned in,unsigned out,bool ordinary) {
    __shared__ float as[8][32],bs[32][33];
    unsigned tid=threadIdx.x,r0=blockIdx.y*8,n0=blockIdx.x*32,r=tid/32,n=tid%32;
    float v0=0,v1=0;
    for(unsigned k0=0;k0<in;k0+=32){
        for(unsigned j=tid;j<8*32;j+=128){unsigned rr=r0+j/32,k=k0+j%32;
            as[j/32][j%32]=rr<rows?(ordinary?x[std::size_t(rr)*in+k]:hf(a[std::size_t(rr)*in+k])):0;}
        for(unsigned j=tid;j<32*32;j+=128){unsigned nn=n0+j/32,k=k0+j%32;
            bs[j%32][j/32]=nn<out?hf(w[ordinary?std::size_t(nn)*in+k:std::size_t(k)*out+nn]):0;}
        __syncthreads();
        #pragma unroll
        for(unsigned k=0;k<32;++k){v0=__fmaf_rn(as[r][k],bs[k][n],v0);v1=__fmaf_rn(as[r+4][k],bs[k][n],v1);}
        __syncthreads();
    }
    if(n0+n<out){if(r0+r<rows)y[std::size_t(r0+r)*out+n0+n]=v0;if(r0+r+4<rows)y[std::size_t(r0+r+4)*out+n0+n]=v1;}
}
// projection_fp32_tiled for rows <= 4 (decode): one thread per (row, output) runs the same ascending __fmaf_rn
// chain over K from 0 on the same operand values (x or hf(a), hf(w)); the row is staged in 256-wide chunks.
// in must be a multiple of 256.
__global__ __launch_bounds__(256) void projection_fp32_row(const float* x,const U16* a,const U16* w,float* y,unsigned in,unsigned out,bool ordinary) {
    __shared__ float sx[256];
    const unsigned t=threadIdx.x,n=blockIdx.x*256+t,r=blockIdx.y;
    float v=0;
    for(unsigned c=0;c<in;c+=256) {
        sx[t]=ordinary?x[std::size_t(r)*in+c+t]:hf(a[std::size_t(r)*in+c+t]);
        __syncthreads();
        if(n<out) {
            if(ordinary) {
                const U16* wr=w+std::size_t(n)*in+c;
                #pragma unroll 8
                for(unsigned k=0;k<256;++k)v=__fmaf_rn(sx[k],hf(wr[k]),v);
            } else {
                const U16* wc=w+std::size_t(c)*out+n;
                #pragma unroll 8
                for(unsigned k=0;k<256;++k)v=__fmaf_rn(sx[k],hf(wc[std::size_t(k)*out]),v);
            }
        }
        __syncthreads();
    }
    if(n<out)y[std::size_t(r)*out+n]=v;
}
__global__ void problem_list(Problem* p,const U16* a,const U16* w,float* y,unsigned rows,unsigned in,unsigned out,bool row) {
    if(threadIdx.x==0){p->size={int(rows),int(out),int(in)};p->a=(Half*)a;p->b=(Half*)w;p->d=y;p->lda=in;p->ldb=row?out:in;p->ldd=out;}
}
template<class Layout>
cudaError_t gemm(const U16* a,const U16* w,float* y,unsigned rows,unsigned in,unsigned out,Problem* p,cudaStream_t s) {
    problem_list<<<1,1,0,s>>>(p,a,w,y,rows,in,out,std::is_same_v<Layout,Row>);
    using G=Projection<Layout>;using Kernel=typename G::GemmKernel;
    typename G::Arguments args(&p->size,1,256,{1.f,0.f},&p->a,&p->b,&p->d,&p->d,&p->lda,&p->ldb,&p->ldd,&p->ldd);
    typename Kernel::Params params(args);
    cutlass::Kernel<Kernel><<<256,Kernel::kThreadCount,sizeof(typename Kernel::SharedStorage),s>>>(params);
    return cudaGetLastError();
}
template<class Arena>
// row: the few-row strips (or, with r2 & r2_idx, the FP32 row chains for rows <= 4); r2 & r2_pre: early A loads.
cudaError_t project(Matrix w,const float* x,float* y,unsigned rows,Arena& a,cudaStream_t s,bool ref,bool tiled=false,bool row=false,unsigned r2=0) {
    // qkv has 24576 columns, beyond the FFN diagnostic decoder's public bound.
    // Its format and decode kernel are identical; use the layer-local decoder below.
    input_transform<<<(rows*(w.input/128)+7)/8,256,0,s>>>(x,a.operand,w.suh,rows,w.input);
    if(ref&&tiled&&row&&(r2&r2_idx)&&rows<=4&&w.input%256==0)projection_fp32_row<<<dim3((w.output+255)/256,rows),256,0,s>>>(nullptr,a.operand,a.decoded,a.raw,w.input,w.output,false);
    else if(ref&&tiled)projection_fp32_tiled<<<dim3((w.output+31)/32,(rows+7)/8),128,0,s>>>(nullptr,a.operand,a.decoded,a.raw,rows,w.input,w.output,false);
    else if(ref)projection_reference<<<unsigned((std::size_t(rows)*w.output+255)/256),256,0,s>>>(nullptr,a.operand,a.decoded,a.raw,rows,w.input,w.output,false);
    else if(row)launch_row_project(w,a.operand,a.raw,rows,s,(r2&r2_pre)!=0); // the caller skipped the layer decode
    else {auto e=gemm<Row>(a.operand,a.decoded,a.raw,rows,w.input,w.output,a.problem,s);if(e!=cudaSuccess)return e;}
    output_transform<<<(rows*(w.output/128)+7)/8,256,0,s>>>(a.raw,y,w.svh,rows,w.output);
    return cudaGetLastError();
}
__global__ void layer_decode(Matrix w,U16* q) {
    unsigned p=threadIdx.x;std::size_t tile=blockIdx.x;
    q[(tile/(w.output/16)*16+qi(p))*w.output+tile%(w.output/16)*16+qo(p)]=book(state(w.packed+tile*16*w.bits,p,w.bits));
}
template<class Arena>
cudaError_t ordinary(const float* x,const U16* w,float* y,unsigned r,unsigned in,unsigned out,Arena& a,cudaStream_t s,bool ref,bool tiled=false,bool row=false) {
    if(ref&&tiled&&row&&r<=4&&in%256==0)projection_fp32_row<<<dim3((out+255)/256,r),256,0,s>>>(x,nullptr,w,y,in,out,true);
    else if(ref&&tiled)projection_fp32_tiled<<<dim3((out+31)/32,(r+7)/8),128,0,s>>>(x,nullptr,w,y,r,in,out,true);
    else if(ref)projection_reference<<<unsigned((std::size_t(r)*out+255)/256),256,0,s>>>(x,nullptr,w,y,r,in,out,true);
    else {round_input<<<unsigned((std::size_t(r)*in+255)/256),256,0,s>>>(x,a.operand,std::size_t(r)*in);
        if(row&&row_dense_fits(r,in,out)){launch_row_dense(w,a.operand,y,r,in,out,s);return cudaGetLastError();} // the same chains
        return gemm<Col>(a.operand,w,y,r,in,out,a.problem,s);}
    return cudaGetLastError();
}
