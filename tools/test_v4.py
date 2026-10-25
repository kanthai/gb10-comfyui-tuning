import importlib.util, sys, torch, comfy_kitchen as ck
from comfy_kitchen.backends import cuda as ckc
ver = sys.argv[1]
sp = importlib.util.spec_from_file_location("ckp_" + ver, f"ck_int8_patch_{ver}.py"); P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
real = ckc._C._real if hasattr(ckc._C, "_real") else ckc._C   # undo the deployed proxy so v3/v4 install fresh
ckc._C = real; ck.int8_linear = getattr(ck.int8_linear, "__wrapped__", ck.int8_linear)
P._apply(ckc, "triton")
if hasattr(P, "_compile_safe_int8_linear"): P._compile_safe_int8_linear(ck)
torch.manual_seed(0); dev = "cuda"
M, K, N = 4096, 10752, 5376                     # swiglu: input 2*K_act -> K_act = 5376
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randint(-127, 127, (N, K // 2), device=dev, dtype=torch.int8); ws = torch.rand(N, 1, device=dev) * 1e-3
def f(x):
    y = ck.int8_linear(x, w, ws, None, torch.bfloat16, True, 256, input_act="swiglu")
    return torch.nn.functional.silu(y) * 2.0 + 1.0
eager = f(x)
try:
    comp = torch.compile(f, fullgraph=False)(x)
    print(ver, "compiled OK; identical to eager:", torch.equal(comp, eager), " max diff:", (comp.float() - eager.float()).abs().max().item())
except Exception as e:
    print(ver, "compile FAILED:", type(e).__name__, str(e).split("\n")[0][:160])
