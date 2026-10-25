import time, torch
n = 3 * 192 * 768 * 1344        # 8 s H3 clip, fp32 pixels (2.4 GB)
g = torch.randn(n, device="cuda"); torch.cuda.synchronize()
def t(label, dst):
    torch.cuda.synchronize(); s = time.time(); dst.copy_(g); torch.cuda.synchronize(); d = time.time() - s
    print(f"{label:34s} {d:6.2f} s  {g.numel()*4/d/1e9:6.1f} GB/s", flush=True)
p = torch.empty(n); t("pageable fresh (first touch)", p); t("pageable again (pages resident)", p)
s = time.time(); q = torch.empty(n, pin_memory=True); print(f"{'pin_memory alloc':34s} {time.time()-s:6.2f} s")
t("pinned", q); t("pinned again", q)
c = g.cpu(); s = time.time(); c2 = g.cpu(); torch.cuda.synchronize(); print(f"{'tensor.cpu() second call':34s} {time.time()-s:6.2f} s")
