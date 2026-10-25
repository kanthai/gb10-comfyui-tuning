# GB10 (DGX Spark) ComfyUI tuning notes

Two days of measured, A/B-tested performance work on a 4-node **NVIDIA DGX Spark (GB10)** cluster
running **ComfyUI 0.41 + DGX Monarch** (multi-node split rendering), plus one **RTX PRO 2000
Blackwell** box. Everything here was measured on real production graphs (MiniMax H3 video,
Krea 2 image, Qwen Image 2.1) with same-seed comparisons and, where claimed, hash-level
bit-exactness checks.

The goal was *faster without changing results*. Most wins below are bit-identical; the rest are
quantization swaps with measured fidelity.

> These are field notes and drop-in patches for a specific stack (ComfyUI 0.41.x,
> comfy-kitchen 0.2.37, DGX Monarch 1.3.0, PyTorch 2.14, CUDA 13, driver 580/590). Each patch
> is version-guarded and falls back to stock behaviour when its assumptions don't hold. Read the
> doc for a patch before installing it.

## Results at a glance

| Workload (GB10 unless noted) | Before | After | Change | Exact? |
|---|---|---|---|---|
| MiniMax H3 ref2v, 8 s 768p clip, 2-node Monarch split | 515.5 s | **415 s** cold (386 s warm, measured before the VAE split) | **-20..25%** | latents + frames bit-identical |
| H3 sampling (8 steps, INT8 ConvRot DiT) | 369.7 s | 302.9 s | -18% | bit-identical |
| H3 VAE decode (192 frames) | 78.2 s | 58.3 s (1 node) / ~33 s (split over 2 nodes) | -25% / -58% | bit-identical |
| Krea 2 turbo fine-tune, 1344x768, 8 steps, 2 LoRAs, Monarch Fleet | 17.8 s/img (FP8) | **9.6 s/img** | -46% | different quant: INT8 from BF16 is *closer* to BF16 than the FP8 file (21.7 vs 17.0 dB) |
| Krea 2, stock ComfyUI single node | 18.2 s | 10.0 s | -45% | same as above |
| Qwen Image 2.1, 1024², 25 steps | 16.3 s | 11.1 s | -32% | official INT8 ConvRot, 31-40 dB vs BF16 |
| Qwen Image 2.1 turbo, 8 steps | 5.1 s | 3.6 s | -30% | 35-38 dB vs BF16 (NVFP4 turbo: 25-26 dB) |
| Krea 2 on RTX PRO 2000 (60 W, 16 GB) | 36.4 s | 18.6 s | -49% | same quant change as above |

## What was done

| # | Topic | Doc | Code |
|---|---|---|---|
| 1 | Hardware ceilings, measurement method | [docs/01-hardware-and-method.md](docs/01-hardware-and-method.md) | `tools/mma_peak2.cu`, `tools/ws_timing.py` |
| 2 | **INT8 GEMM on sm_121**: comfy-kitchen's CUTLASS path runs at 37% of peak; a Triton fused-dequant GEMM reaches 88% | [docs/02-int8-triton-gemm.md](docs/02-int8-triton-gemm.md) | `patches/ck_int8_patch.py` |
| 3 | **torch.compile + INT8**: why H3 crashed under compile and the one-line root cause | [docs/03-torch-compile-int8.md](docs/03-torch-compile-int8.md) | `patches/ck_int8_patch.py` (v4 part), `tools/test_v4.py` |
| 4 | **GB10 pageable-memory trap**: GPU→fresh host memory at 0.1 GB/s, and the H3 VAE fix | [docs/04-pageable-memory-and-h3-vae.md](docs/04-pageable-memory-and-h3-vae.md) | `patches/gb10_comfy_patch.py`, `tools/d2h_matrix.py`, `tools/copy_bench.py` |
| 5 | **Split the H3 VAE decode across two nodes** (bit-identical) | [docs/05-split-vae-decode.md](docs/05-split-vae-decode.md) | `patches/gb10_comfy_patch.py`, `patches/h3_vae_peer.py`, `systemd/h3-vae-peer.service` |
| 6 | DGX Monarch: memory preflights, warm back-to-back renders, recycle cost | [docs/06-monarch-memory-and-warm-renders.md](docs/06-monarch-memory-and-warm-renders.md) | — |
| 7 | Persistent torch.compile cache for Monarch actors | [docs/07-persistent-inductor-cache.md](docs/07-persistent-inductor-cache.md) | `systemd/inductor-cache-link.*` |
| 8 | Quantization: INT8 ConvRot from BF16 vs FP8, NVFP4 (RTN / learned rounding), Qwen | [docs/08-quantization.md](docs/08-quantization.md) | `tools/krea_int8_from_bf16.py`, `tools/qerr.py` |
| 9 | ComfyUI flags, attention kernels, compile, clocks: what did **not** help | [docs/09-what-did-not-help.md](docs/09-what-did-not-help.md) | `tools/clock_sustain.sh`, `tools/clock_bench.sh` |
| 10 | RTX PRO 2000 Blackwell (sm_120, 60 W) | [docs/10-rtx-pro-2000.md](docs/10-rtx-pro-2000.md) | — |
| 11 | Scaling: 4-node split vs two 2-node pairs | [docs/11-scaling-4-nodes.md](docs/11-scaling-4-nodes.md) | — |
| — | **Gotchas** (the list you actually want) | [GOTCHAS.md](GOTCHAS.md) | — |

## The three biggest lessons

1. **ComfyUI is not tuned for GB10, but it is not "wrong" either.** Its memory defaults
   (DynamicVRAM / async offload) are the right choice on unified memory; every memory flag we tried
   made things slower. The losses were in *kernels* (INT8 routed to a slow CUTLASS config on sm_121)
   and in *host copies* (GB10's first-touch pageable-memory behaviour).
2. **On GB10, never let the GPU be the first writer of fresh pageable host memory.** Pre-touch it
   from the CPU (`zero_()`), pin it, or keep the data on the GPU. 2.4 GB took 19 s instead of 0.05 s.
3. **Measure fidelity against the true source, not against what you shipped yesterday.** An INT8
   model quantized from the official BF16 weights was both the fastest *and* the most faithful
   option — while looking "more different" from the old FP8 outputs.

## Installing the patches (short version)

Patches are loaded through `.pth` files in the ComfyUI venv's `site-packages`, so they apply to the
ComfyUI process **and** to DGX Monarch actor processes (which import site-packages too). Each one has
a mode file to turn it off without uninstalling.

```bash
SP=$(/opt/comfyui-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
cp patches/ck_int8_patch.py patches/gb10_comfy_patch.py "$SP"/
echo 'import ck_int8_patch'   > "$SP"/zz_ck_int8_patch.pth
echo 'import gb10_comfy_patch' > "$SP"/zz_gb10_comfy_patch.pth
echo triton > /etc/comfy-kitchen-int8.mode       # 'off' = stock comfy-kitchen
# restart ComfyUI; with Monarch also recycle the mesh so actors re-import
```

See each doc for prerequisites (Triton needs `Python.h`; the split VAE needs a peer service).

## License

MIT. No warranty: these patches monkey-patch third-party code at import time. They were validated
on the versions listed above only.
