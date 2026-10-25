import sys, time, torch
sys.path.insert(0, "/opt/ComfyUI")
import comfy.sd, comfy.utils
sd = comfy.utils.load_torch_file("/opt/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors")
vae = comfy.sd.VAE(sd=sd)
m = vae.first_stage_model.to("cuda").to(vae.vae_dtype)
print("vae class", type(m).__name__, "dtype", vae.vae_dtype, "tile", m.tile_size, "overlap", m.tile_overlap_min, flush=True)
torch.manual_seed(0)
z = torch.randn(1, 24, 57, 48, 84, device="cuda", dtype=vae.vae_dtype)
def run(tiling, tile):
    m.tiling, m.tile_size = tiling, tile
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    with torch.inference_mode():
        out = m.decode(z.clone())
    torch.cuda.synchronize()
    return time.time() - t, out.float().cpu(), torch.cuda.max_memory_allocated() / 2**30
res = {}
for name, tiling, tile in (("default tile 256", True, 256), ("tile 512", True, 512), ("tile 768", True, 768), ("untiled", False, 256)):
    try:
        dt, out, mem = run(tiling, tile); res[name] = out
        print(f"{name:16s}: {dt:6.1f} s  peak {mem:5.1f} GiB  frames {tuple(out.shape)}", flush=True)
    except Exception as e:
        print(f"{name:16s}: FAILED {type(e).__name__} {str(e)[:120]}", flush=True)
if "untiled" in res:
    u = (res["untiled"].clamp(0, 1) * 255).round()
    for name, o in res.items():
        if name == "untiled": continue
        mse = (((o.clamp(0, 1) * 255).round() - u) ** 2).mean().item()
        import math; print(f"{name:16s} vs untiled: PSNR {10*math.log10(255**2/max(mse,1e-9)):.1f} dB")
