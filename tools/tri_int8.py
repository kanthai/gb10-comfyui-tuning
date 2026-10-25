import torch, triton, triton.language as tl, sys
from comfy_kitchen.backends import cuda as ckc
dev="cuda"
cfgs=[triton.Config({"BM":bm,"BN":bn,"BK":bk,"G":8},num_warps=w,num_stages=s)
      for bm,bn,bk,w,s in [(128,128,64,4,3),(128,128,128,4,3),(128,256,64,8,3),(256,128,64,8,3),(128,128,64,8,4),
                           (128,256,128,8,3),(256,128,128,8,3),(64,128,128,4,4),(128,64,128,4,4),(256,256,64,8,3)]]
@triton.autotune(configs=cfgs,key=["M","N","K"])
@triton.jit
def k(a,b,xs,ws,c,M,N,K,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,G:tl.constexpr):
    pid=tl.program_id(0); npm=tl.cdiv(M,BM); npn=tl.cdiv(N,BN)
    gid=pid//(G*npn); fm=gid*G; gs=min(npm-fm,G)
    pm=fm+(pid%gs); pn=(pid%(G*npn))//gs
    rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN); rk=tl.arange(0,BK)
    A=a+rm[:,None]*K+rk[None,:]; B=b+rn[None,:]*K+rk[:,None]
    acc=tl.zeros((BM,BN),dtype=tl.int32)
    for kk in range(0,K,BK):
        acc+=tl.dot(tl.load(A,mask=rm[:,None]<M,other=0),tl.load(B,mask=rn[None,:]<N,other=0))
        A+=BK; B+=BK
    out=acc.to(tl.float32)*tl.load(xs+rm,mask=rm<M,other=0)[:,None]*tl.load(ws+rn,mask=rn<N,other=0)[None,:]
    tl.store(c+rm[:,None]*N+rn[None,:],out.to(tl.bfloat16),mask=(rm[:,None]<M)&(rn[None,:]<N))
def tri(x8,w8,xs,ws):
    M,K=x8.shape; N=w8.shape[0]; c=torch.empty(M,N,device=dev,dtype=torch.bfloat16)
    k[(lambda m: (triton.cdiv(M,m["BM"])*triton.cdiv(N,m["BN"]),))](x8,w8,xs,ws,c,M,N,K); return c
def bench(fn,it=10,warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it/1e3
M=int(sys.argv[1]); shapes={"qkv":(5376,21504),"out":(7168,5376),"fc1":(5376,28672),"fc2":(14336,5376)}
T={"int_mm":0,"tri_fused":0,"ck_cublas_gemm+deq":0}
for name,(K,N) in shapes.items():
    x8=torch.randint(-127,127,(M,K),device=dev,dtype=torch.int8); w8=torch.randint(-127,127,(N,K),device=dev,dtype=torch.int8)
    xs=torch.rand(M,device=dev)*1e-3; ws=torch.rand(N,device=dev)*1e-3
    ref=(torch._int_mm(x8,w8.t()).float()*xs[:,None]*ws[None,:])
    y=tri(x8,w8,xs,ws).float(); err=((y-ref).abs().max()/ref.abs().max()).item()
    fl=2*M*K*N
    r={"int_mm":bench(lambda: torch._int_mm(x8,w8.t())),"tri_fused":bench(lambda: tri(x8,w8,xs,ws))}
    def ckpath():
        o32=torch.empty(M,N,dtype=torch.int32,device=dev); out=torch.empty(M,N,dtype=torch.bfloat16,device=dev)
        ckc._C.cublas_gemm_int8(ckc._wrap_for_dlpack(x8),ckc._wrap_for_dlpack(w8),ckc._wrap_for_dlpack(o32),ckc._wrap_for_dlpack(ckc.get_cublas_workspace()),torch.cuda.current_stream().cuda_stream)
        ckc._C.dequantize_int8_linear(ckc._wrap_for_dlpack(o32),ckc._wrap_for_dlpack(xs[:,None].contiguous()),ckc._wrap_for_dlpack(ws[:,None].contiguous()),ckc._wrap_for_dlpack(ckc._empty_cuda_tensor(x8.device,torch.bfloat16)),ckc._wrap_for_dlpack(out),ckc.DTYPE_TO_CODE[torch.bfloat16],torch.cuda.current_stream().cuda_stream)
    try: r["ck_cublas_gemm+deq"]=bench(ckpath)
    except Exception as e: print("ckpath fail",str(e)[:150]); r["ck_cublas_gemm+deq"]=0
    for kk,v in r.items(): T[kk]+=v
    print(f"{name}: "+"  ".join(f"{kk}={v*1e3:.1f}ms({fl/v/1e12 if v else 0:.0f}T)" for kk,v in r.items())+f"  tri_err={err:.1e} cfg={k.best_config}")
print("x50 blocks, s:",{kk:round(v*50,2) for kk,v in T.items()})
