// Pure tensor-core throughput: register-only mma.sync loops, no memory traffic.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CH 8
template<int KIND> __global__ void k(int iters, int* out){
  uint32_t a0=threadIdx.x, a1=a0*3, a2=a0*5, a3=a0*7, b0=a0*11, b1=a0*13;
  float  fc[CH][4]={}; int ic[CH][4]={};
  for(int i=0;i<iters;i++){
#pragma unroll
    for(int c=0;c<CH;c++){
      if(KIND==0) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(ic[c][0]),"+r"(ic[c][1]),"+r"(ic[c][2]),"+r"(ic[c][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
      if(KIND==1) asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(fc[c][0]),"+f"(fc[c][1]),"+f"(fc[c][2]),"+f"(fc[c][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
      if(KIND==2) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(fc[c][0]),"+f"(fc[c][1]),"+f"(fc[c][2]),"+f"(fc[c][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
      if(KIND==4) asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
        :"+f"(fc[c][0]),"+f"(fc[c][1]),"+f"(fc[c][2]),"+f"(fc[c][3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0x38383838u),"r"(0x38383838u));
      if(KIND==3) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
        :"+r"(ic[c][0]),"+r"(ic[c][1]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
  }
  int s=0; for(int c=0;c<CH;c++) for(int j=0;j<4;j++) s+=ic[c][j]+(int)fc[c][j];
  if(s==0x7fffffff) out[0]=s;
}
template<int KIND> void run(const char* name, double flops_per_mma, int sms){
  int* d; cudaMalloc(&d,4); int iters=400000, blocks=sms*8, threads=256;
  k<KIND><<<blocks,threads>>>(100,d); cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); k<KIND><<<blocks,threads>>>(iters,d); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b);
  double ops=(double)blocks*(threads/32)*iters*CH*flops_per_mma;
  printf("%-28s %8.1f T/s   (%.0f ms)\n",name,ops/ms/1e9,ms);
}
int main(){
  int sms; cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
  run<4>("nvfp4 m16n8k64 block_scale",2.0*16*8*64,sms);
  for(int r=0;r<1;r++){
  run<0>("int8 m16n8k32 s32acc",2.0*16*8*32,sms);
  run<1>("fp8 e4m3 m16n8k32 f32acc",2.0*16*8*32,sms);
  run<2>("bf16 m16n8k16 f32acc",2.0*16*8*16,sms);
  run<3>("fp16 m16n8k16 f16acc",2.0*16*8*16,sms);
  }
}
