"""Per-layer NVFP4 error: plain RTN vs a CTQ (convert_to_quant) learned-rounding checkpoint.
Usage: qerr.py <bf16 source> <ctq mixed nvfp4 file>"""
import json, sys, torch
from safetensors import safe_open
from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as F4
import comfy_kitchen as ck
bf = safe_open(sys.argv[1], "pt", device="cuda")
cq = safe_open(sys.argv[2], "pt", device="cuda")
P = "model.diffusion_model.blocks."
rows = []
for b in (5, 8, 12, 15, 18, 22):
    for l in ("attn.wq", "attn.wo", "mlp.up", "mlp.down"):
        k = f"{P}{b}.{l}"
        meta = json.loads(bytes(cq.get_tensor(k + ".comfy_quant").tolist()).decode())
        if meta.get("format") != "nvfp4":
            continue
        w = bf.get_tensor(k + ".weight").to(torch.bfloat16)
        q, p = F4.quantize(w)
        rtn = ck.dequantize_nvfp4(q, p.scale, p.block_scale, torch.bfloat16).float()
        ctq = ck.dequantize_nvfp4(cq.get_tensor(k + ".weight"), cq.get_tensor(k + ".weight_scale_2"), cq.get_tensor(k + ".weight_scale"), torch.bfloat16).float()
        x = torch.randn(2048, w.shape[1], device="cuda")
        y = x @ w.float().t()
        e = lambda d: ((d - w.float()).norm() / w.float().norm()).item()
        o = lambda d: ((x @ d.t() - y).norm() / y.norm()).item()
        rows.append((k.split("blocks.")[1], e(rtn), e(ctq), o(rtn), o(ctq)))
print(f"{'layer':16s} {'weight err RTN':>14s} {'CTQ':>7s}  {'output err RTN':>14s} {'CTQ':>7s}")
for r in rows: print(f"{r[0]:16s} {r[1]:14.4f} {r[2]:7.4f}  {r[3]:14.4f} {r[4]:7.4f}")
m = lambda i: sum(r[i] for r in rows) / len(rows)
print(f"{'MEAN':16s} {m(1):14.4f} {m(2):7.4f}  {m(3):14.4f} {m(4):7.4f}   -> CTQ error = {m(2)/m(1):.0%} of RTN (weights), {m(4)/m(3):.0%} (outputs)")
