import ctypes, mmap, os, resource, sys, time, torch
N = int(float(os.environ.get("GB", "1.2")) * 2**30)
dev = torch.device("cuda")
src = torch.ones(N // 2, dtype=torch.float16, device=dev)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
MADV_POPULATE_WRITE = 23
def faults(): return resource.getrusage(resource.RUSAGE_SELF).ru_minflt
def timed(name, make, direction="d2h"):
    buf = make()
    torch.cuda.synchronize(); f0 = faults(); t = time.perf_counter()
    if direction == "d2h": buf.copy_(src)
    else: src.copy_(buf)
    torch.cuda.synchronize(); dt = time.perf_counter() - t; df = faults() - f0
    print(f"{direction} {name:28s} {dt*1e3:9.1f} ms  {N/dt/1e9:7.2f} GB/s  minflt {df}", flush=True)
    del buf
def fresh(): return torch.empty(N // 2, dtype=torch.float16)
def zeroed():
    b = fresh(); b.zero_(); return b
def readtouch():
    b = fresh(); step = os.sysconf("SC_PAGESIZE") // 2; float(b[::step].float().sum()); return b
def popwrite():
    b = fresh(); p = b.data_ptr(); ps = os.sysconf("SC_PAGESIZE"); a = p - p % ps
    r = libc.madvise(ctypes.c_void_p(a), ctypes.c_size_t(N + (p - a)), MADV_POPULATE_WRITE)
    if r != 0: print("madvise errno", ctypes.get_errno())
    return b
def registered():
    b = fresh(); cr = torch.cuda.cudart(); r = cr.cudaHostRegister(b.data_ptr(), N, 0); assert int(r) == 0, r; return b
def pinned(): return torch.empty(N // 2, dtype=torch.float16, pin_memory=True)
print("pagesize", os.sysconf("SC_PAGESIZE"), "GB", N / 2**30, "env", os.environ.get("CUDA_BUFFER_PAGE_IN_THRESHOLD_MS"), flush=True)
p = torch.cuda.get_device_properties(0); print("integrated", p.is_integrated, flush=True)
cases = [("fresh", fresh), ("zero_ (cpu write touch)", zeroed), ("read touch", readtouch), ("MADV_POPULATE_WRITE", popwrite), ("cudaHostRegister", registered), ("pin_memory", pinned)]
only = os.environ.get("ONLY")
for name, mk in cases:
    if only and name != only: continue
    timed(name, mk)
if not only:
    for name, mk in (("fresh", fresh), ("zero_ (cpu write touch)", zeroed), ("pin_memory", pinned)):
        timed(name, mk, "h2d")
