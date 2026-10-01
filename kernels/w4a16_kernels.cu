// w4a16 GEMV kernels for A100 (sm_80). Weight-only int4, fp16 activations.
//
// This file is the full optimization ladder, not just the final kernel. Each
// kernel is a measured rung in taking int4 decode-GEMV from 52% -> ~80% of the
// A100's 1555 GB/s peak, at full fp32 accuracy. Kept together so the whole
// story (including the rejected experiments) is reproducible via the benchmark.
//
// Measured on A100-SXM4-40GB, large decode shapes (see benchmarks/):
//   byte        one 8-bit load / lane, warp-per-row, fp32       ~52% peak   (starting point)
//   u32         32-bit load / lane (8 int4)                      ~64% peak
//   u32_v4      + 128-bit float4 vector load (cuts L1 pressure)  ~67% peak
//   champ       + fp16 bit-trick dequant, fp32 accumulate        ~72% peak
//   marlin      + lop3 dequant, interleaved packing (SHIPPED)    ~80% peak
//
// Rejected experiments, kept for reproducibility (see README "what didn't work"):
//   uint4       128-bit weight load (32 int4 / lane)   regressed: too little work/warp
//   h2f32       fp16 multiply-accumulate               ~82% but maxerr 6e-2 (1000x worse)
//
// TWO PACKINGS: the ladder up to champ uses simple hi/lo packing (pack_int4).
// marlin requires interleaved packing (pack_int4_interleaved), slot order
// [0,2,4,6,1,3,5,7], so that four lop3 ops decode nibbles back into column order.
// Each kernel below documents which packing it expects.

#include <torch/extension.h>
#include <cuda_fp16.h>

// (a & b) | c  in one instruction; immLut 0xEA. Used by the marlin dequant.
__device__ __forceinline__ unsigned lop3_ea(unsigned a, unsigned b, unsigned c){
    unsigned r;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// fp16 bit-trick: half(0x6400 | n) - 1032.0 == n - 8, for n in 0..15.
// Decodes 8 nibbles (simple hi/lo packing) into four half2 (weights, zero-point removed).
__device__ __forceinline__ void dq4_simple(unsigned val, half2& w0, half2& w1, half2& w2, half2& w3){
    const half2 bias = __float2half2_rn(1032.0f);
    unsigned h0=0x64006400u|((val>>4)&0xF)|((val&0xF)<<16);
    unsigned h1=0x64006400u|((val>>12)&0xF)|(((val>>8)&0xF)<<16);
    unsigned h2=0x64006400u|((val>>20)&0xF)|(((val>>16)&0xF)<<16);
    unsigned h3=0x64006400u|((val>>28)&0xF)|(((val>>24)&0xF)<<16);
    w0=__hsub2(*reinterpret_cast<half2*>(&h0),bias); w1=__hsub2(*reinterpret_cast<half2*>(&h1),bias);
    w2=__hsub2(*reinterpret_cast<half2*>(&h2),bias); w3=__hsub2(*reinterpret_cast<half2*>(&h3),bias);
}

// ================= LADDER =================

// rung 1: byte load / lane (simple packing)
__global__ void k_byte(const uint8_t* packed,const float* scale,const half* vec,float* out,int rows,int cols){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32; if(wid>=rows)return;
    float s=scale[wid];float p=0.f;int c2=cols/2; const uint8_t* rp=packed+wid*c2;
    for(int c=lane;c<c2;c+=32){uint8_t b=rp[c];int hi=(b>>4)&0xF,lo=b&0xF;
        p+=(hi-8)*s*__half2float(vec[c*2])+(lo-8)*s*__half2float(vec[c*2+1]);}
    for(int o=16;o>0;o/=2)p+=__shfl_down_sync(0xffffffff,p,o); if(lane==0)out[wid]=p;
}

// rung 2: u32 load / lane, 8 int4 (simple packing), int dequant, 4 half2 vector loads
__global__ void k_u32(const uint8_t* packed,const float* scale,const half* vec,float* out,int rows,int cols){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32; if(wid>=rows)return;
    float s=scale[wid];float p=0.f;int groups=cols/8;
    const uint32_t* rp=reinterpret_cast<const uint32_t*>(packed+wid*(cols/2));
    const half2* v2=reinterpret_cast<const half2*>(vec);
    for(int g=lane;g<groups;g+=32){unsigned val=rp[g];int b=g*4;
        int n0=(val>>4)&0xF,n1=val&0xF,n2=(val>>12)&0xF,n3=(val>>8)&0xF,n4=(val>>20)&0xF,n5=(val>>16)&0xF,n6=(val>>28)&0xF,n7=(val>>24)&0xF;
        float2 a=__half22float2(v2[b]),bb=__half22float2(v2[b+1]),c=__half22float2(v2[b+2]),d=__half22float2(v2[b+3]);
        p+=(n0-8)*a.x+(n1-8)*a.y+(n2-8)*bb.x+(n3-8)*bb.y+(n4-8)*c.x+(n5-8)*c.y+(n6-8)*d.x+(n7-8)*d.y;}
    p*=s; for(int o=16;o>0;o/=2)p+=__shfl_down_sync(0xffffffff,p,o); if(lane==0)out[wid]=p;
}

// rung 3: u32 + float4 vector load (simple packing, int dequant)
__global__ void k_u32_v4(const uint8_t* packed,const float* scale,const half* vec,float* out,int rows,int cols){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32; if(wid>=rows)return;
    float s=scale[wid];float p=0.f;int groups=cols/8;
    const uint32_t* rp=reinterpret_cast<const uint32_t*>(packed+wid*(cols/2));
    const float4* v4=reinterpret_cast<const float4*>(vec);
    for(int g=lane;g<groups;g+=32){unsigned val=rp[g];float4 raw=v4[g];half2* hv=reinterpret_cast<half2*>(&raw);
        int n0=(val>>4)&0xF,n1=val&0xF,n2=(val>>12)&0xF,n3=(val>>8)&0xF,n4=(val>>20)&0xF,n5=(val>>16)&0xF,n6=(val>>28)&0xF,n7=(val>>24)&0xF;
        float2 a=__half22float2(hv[0]),bb=__half22float2(hv[1]),c=__half22float2(hv[2]),d=__half22float2(hv[3]);
        p+=(n0-8)*a.x+(n1-8)*a.y+(n2-8)*bb.x+(n3-8)*bb.y+(n4-8)*c.x+(n5-8)*c.y+(n6-8)*d.x+(n7-8)*d.y;}
    p*=s; for(int o=16;o>0;o/=2)p+=__shfl_down_sync(0xffffffff,p,o); if(lane==0)out[wid]=p;
}

// rung 4: champ, fp16 bit-trick dequant + float4 vector + fp32 accumulate (simple packing)
__global__ void k_champ(const uint8_t* packed,const float* scale,const half* vec,float* out,int rows,int cols){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32; if(wid>=rows)return;
    float s=scale[wid];float p=0.f;int groups=cols/8;
    const uint32_t* rp=reinterpret_cast<const uint32_t*>(packed+wid*(cols/2));
    const float4* v4=reinterpret_cast<const float4*>(vec);
    for(int g=lane;g<groups;g+=32){unsigned val=rp[g];float4 raw=v4[g];half2* hv=reinterpret_cast<half2*>(&raw);
        half2 w0,w1,w2,w3; dq4_simple(val,w0,w1,w2,w3);
        float2 f0=__half22float2(w0),f1=__half22float2(w1),f2=__half22float2(w2),f3=__half22float2(w3);
        float2 a=__half22float2(hv[0]),b=__half22float2(hv[1]),c=__half22float2(hv[2]),d=__half22float2(hv[3]);
        p+=f0.x*a.x+f0.y*a.y+f1.x*b.x+f1.y*b.y+f2.x*c.x+f2.y*c.y+f3.x*d.x+f3.y*d.y;}
    p*=s; for(int o=16;o>0;o/=2)p+=__shfl_down_sync(0xffffffff,p,o); if(lane==0)out[wid]=p;
}

// rung 5: marlin (SHIPPED), lop3 dequant + interleaved packing + float4 vector + fp32 accumulate
// REQUIRES pack_int4_interleaved (slot order [0,2,4,6,1,3,5,7]).
__global__ void k_marlin(const uint8_t* packed,const float* scale,const half* vec,float* out,int rows,int cols){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32; if(wid>=rows)return;
    float s=scale[wid];float p=0.f;int groups=cols/8;
    const uint32_t* rp=reinterpret_cast<const uint32_t*>(packed+wid*(cols/2));
    const float4* v4=reinterpret_cast<const float4*>(vec);
    const half2 bias=__float2half2_rn(1032.0f); const unsigned MASK=0x000f000fu,EXP=0x64006400u;
    for(int g=lane;g<groups;g+=32){unsigned val=rp[g];float4 raw=v4[g];half2* hv=reinterpret_cast<half2*>(&raw);
        unsigned o0=lop3_ea(val,MASK,EXP),o1=lop3_ea(val>>4,MASK,EXP),o2=lop3_ea(val>>8,MASK,EXP),o3=lop3_ea(val>>12,MASK,EXP);
        half2 w0=__hsub2(*reinterpret_cast<half2*>(&o0),bias),w1=__hsub2(*reinterpret_cast<half2*>(&o1),bias);
        half2 w2=__hsub2(*reinterpret_cast<half2*>(&o2),bias),w3=__hsub2(*reinterpret_cast<half2*>(&o3),bias);
        float2 f0=__half22float2(w0),f1=__half22float2(w1),f2=__half22float2(w2),f3=__half22float2(w3);
        float2 a=__half22float2(hv[0]),b=__half22float2(hv[1]),c=__half22float2(hv[2]),d=__half22float2(hv[3]);
        p+=f0.x*a.x+f0.y*a.y+f1.x*b.x+f1.y*b.y+f2.x*c.x+f2.y*c.y+f3.x*d.x+f3.y*d.y;}
    p*=s; for(int o=16;o>0;o/=2)p+=__shfl_down_sync(0xffffffff,p,o); if(lane==0)out[wid]=p;
}

// ============== REJECTED (kept for reproducibility) ==============

// uint4: 128-bit weight load, 32 int4 / lane (simple packing). Regressed: too little work per warp.
__device__ __forceinline__ float dot8_int(unsigned val, float2 a, float2 b, float2 c, float2 d){
    int n0=(val>>4)&0xF,n1=val&0xF,n2=(val>>12)&0xF,n3=(val>>8)&0xF,n4=(val>>20)&0xF,n5=(val>>16)&0xF,n6=(val>>28)&0xF,n7=(val>>24)&0xF;
    return (n0-8)*a.x+(n1-8)*a.y+(n2-8)*b.x+(n3-8)*b.y+(n4-8)*c.x+(n5-8)*c.y+(n6-8)*d.x+(n7-8)*d.y;
}
__global__ void k_uint4(const uint8_t* packed,const float* scale,const half* vec,float* out,int rows,int cols){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32; if(wid>=rows)return;
    float s=scale[wid];float p=0.f;int groups=cols/32;
    const uint4* rp=reinterpret_cast<const uint4*>(packed+wid*(cols/2));
    const half2* v2=reinterpret_cast<const half2*>(vec);
    for(int g=lane;g<groups;g+=32){uint4 pk=rp[g];const half2* vg=v2+g*16;
        unsigned w[4]={pk.x,pk.y,pk.z,pk.w};
        for(int k=0;k<4;k++){int b=k*4;
            p+=dot8_int(w[k],__half22float2(vg[b]),__half22float2(vg[b+1]),__half22float2(vg[b+2]),__half22float2(vg[b+3]));}}
    p*=s; for(int o=16;o>0;o/=2)p+=__shfl_down_sync(0xffffffff,p,o); if(lane==0)out[wid]=p;
}

// h2f32: fp16 multiply-accumulate (simple packing). ~82% peak but maxerr ~6e-2 (fp16 multiply
// rounds each product before summing). Documented speed/accuracy knob, not the default.
__global__ void k_h2f32(const uint8_t* packed,const float* scale,const half* vec,float* out,int rows,int cols){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32; if(wid>=rows)return;
    float s=scale[wid];float p=0.f;int groups=cols/8;
    const uint32_t* rp=reinterpret_cast<const uint32_t*>(packed+wid*(cols/2));
    const float4* v4=reinterpret_cast<const float4*>(vec);
    for(int g=lane;g<groups;g+=32){unsigned val=rp[g];float4 raw=v4[g];half2* hv=reinterpret_cast<half2*>(&raw);
        half2 w0,w1,w2,w3; dq4_simple(val,w0,w1,w2,w3);
        half2 t=__hmul2(w0,hv[0]); t=__hfma2(w1,hv[1],t); t=__hfma2(w2,hv[2],t); t=__hfma2(w3,hv[3],t);
        p+=__low2float(t)+__high2float(t);}
    p*=s; for(int o=16;o>0;o/=2)p+=__shfl_down_sync(0xffffffff,p,o); if(lane==0)out[wid]=p;
}

// fp16 reference (dense half matvec, vectorized), the efficiency baseline
__global__ void k_fp16(const half* mat,const half* vec,float* out,int rows,int cols){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(wid<rows){float p=0.f;int c2=cols/2;
        const half2* mr=reinterpret_cast<const half2*>(mat+wid*cols);const half2* v2=reinterpret_cast<const half2*>(vec);
        for(int c=lane;c<c2;c+=32){float2 mf=__half22float2(mr[c]),vf=__half22float2(v2[c]);p+=mf.x*vf.x+mf.y*vf.y;}
        for(int o=16;o>0;o/=2)p+=__shfl_down_sync(0xffffffff,p,o); if(lane==0)out[wid]=p;}
}

// ============== host wrappers ==============
#define WQ(NM,K) torch::Tensor NM(torch::Tensor pk,torch::Tensor sc,torch::Tensor v){int rows=pk.size(0),cols=pk.size(1)*2;auto o=torch::empty({rows},torch::dtype(torch::kFloat32).device(pk.device()));int t=256,w=t/32,b=(rows+w-1)/w;K<<<b,t>>>(pk.data_ptr<uint8_t>(),sc.data_ptr<float>(),reinterpret_cast<const half*>(v.data_ptr<at::Half>()),o.data_ptr<float>(),rows,cols);return o;}
WQ(byte_cuda,k_byte) WQ(u32_cuda,k_u32) WQ(u32_v4_cuda,k_u32_v4) WQ(champ_cuda,k_champ)
WQ(marlin_cuda,k_marlin) WQ(uint4_cuda,k_uint4) WQ(h2f32_cuda,k_h2f32)

torch::Tensor fp16_cuda(torch::Tensor m,torch::Tensor v){int rows=m.size(0),cols=m.size(1);auto o=torch::empty({rows},m.options().dtype(torch::kFloat32));int t=256,w=t/32,b=(rows+w-1)/w;k_fp16<<<b,t>>>(reinterpret_cast<const half*>(m.data_ptr<at::Half>()),reinterpret_cast<const half*>(v.data_ptr<at::Half>()),o.data_ptr<float>(),rows,cols);return o;}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("byte_cuda",&byte_cuda);       m.def("u32_cuda",&u32_cuda);
    m.def("u32_v4_cuda",&u32_v4_cuda);   m.def("champ_cuda",&champ_cuda);
    m.def("marlin_cuda",&marlin_cuda);   m.def("uint4_cuda",&uint4_cuda);
    m.def("h2f32_cuda",&h2f32_cuda);     m.def("fp16_cuda",&fp16_cuda);
}
