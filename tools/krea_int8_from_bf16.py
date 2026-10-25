"""INT8 tensorwise + ConvRot-256 (per-channel) Krea 2 DiT straight from a BF16 checkpoint (no FP8 step).
Key names are matched to an existing FP8 file of the same model so it is a drop-in replacement.
Usage: krea_int8_from_bf16.py <bf16 src> <fp8 reference (for key names)> <out>"""
import json, os, struct, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as I8
with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f: f.write("1000")
src, ref, out_path = sys.argv[1:4]
PFX = "model.diffusion_model."
def keys(p):
    with open(p, "rb") as f: n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    h.pop("__metadata__", None); return set(h)
ref_keys = keys(ref)
CQ = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
out, errs, nq = {}, [], 0
with safe_open(src, "pt", device="cuda") as f:
    for k in f.keys():
        name = k[len(PFX):] if k.startswith(PFX) else k
        t = f.get_tensor(k)
        base = name[:-len(".weight")] if name.endswith(".weight") else None
        if base and base.startswith("blocks.") and t.dim() == 2 and t.shape[1] % 256 == 0 and t.dtype == torch.bfloat16:
            q, p = I8.quantize(t, per_channel=True, convrot=True, convrot_groupsize=256)
            if len(errs) < 40:
                d = I8.dequantize(q, p).float(); errs.append(((d - t.float()).norm() / t.float().norm()).item())
            out[name], out[f"{base}.weight_scale"] = q.cpu(), p.scale.float().cpu()
            out[f"{base}.comfy_quant"] = torch.frombuffer(bytearray(CQ), dtype=torch.uint8); nq += 1
        else:
            out[name] = t.cpu()
        del t
missing = sorted(k for k in ref_keys if k not in out)
print(f"quantized {nq} linears from bf16; mean weight rel err vs bf16 {sum(errs)/len(errs):.4f}; keys missing vs fp8 file: {len(missing)} {missing[:3]}", flush=True)
save_file(out, out_path + ".tmp"); os.replace(out_path + ".tmp", out_path)
print("wrote", out_path, os.path.getsize(out_path) // 2**20, "MiB", flush=True)
