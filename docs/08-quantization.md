# 8. Quantization: what is fast *and* faithful on GB10

## Summary

| Format | Speed on GB10 | Fidelity | Verdict |
|---|---|---|---|
| BF16 | baseline | reference | too slow for production |
| FP8 (raw, no scales) | runs as "manual cast": FP8→BF16 every step, BF16 GEMM | ≈ BF16 | slow unless `--fast fp8_matrix_mult` (stock ComfyUI only) |
| **INT8 tensorwise + ConvRot-256, per-channel** | **fastest** with the Triton GEMM (doc 02) | closest to BF16 when made from BF16 | **default** |
| NVFP4 (W4A4) | similar to INT8+Triton on GB10 | "sibling" images: composition/props change | rejected for production |
| GGUF | not tested in depth | — | — |

## Krea 2 turbo fine-tune (1344x768, 8 steps, 2 LoRAs)

Fidelity measured as PSNR vs renders from the **official BF16** checkpoint, 8 images, same seeds:

| Checkpoint | s/img (2-node Fleet) | PSNR vs BF16 |
|---|---|---|
| official BF16 | 16.1 | — |
| FP8 (the file we were using) | 17.8 | 17.0 dB |
| FP8 + compile | 14.4 | 16.6 dB |
| INT8 ConvRot quantized **from the FP8 file** | 9.7 | 18.3 dB |
| **INT8 ConvRot quantized from the official BF16** | **9.6** | **21.7 dB** |
| CTQ mixed NVFP4/MXFP8 (learned rounding) | 13.4 | 12-16 dB ("sibling") |

The INT8-from-BF16 file is the fastest **and** the most faithful. Compared with the old FP8
outputs it looks *more* different (17.4 dB vs 19.3 dB for INT8-from-FP8), because it is closer to
the true model than the FP8 file was. Measure against the source, not against yesterday's output.

`tools/krea_int8_from_bf16.py` makes it: every 2-D linear weight in the transformer blocks whose
input dim is a multiple of 256 → `TensorWiseINT8Layout.quantize(per_channel=True, convrot=True,
convrot_groupsize=256)` from comfy-kitchen, with a `comfy_quant` metadata tensor so ComfyUI picks
the INT8 path; everything else stays BF16. Key names are matched to the existing FP8 file (strip
the `model.diffusion_model.` prefix) so it is a drop-in replacement. Mean weight relative error
~1%.

## Qwen Image 2.1 (official Comfy-Org INT8 ConvRot files, sha-verified)

| Preset | BF16 / NVFP4 | INT8 ConvRot | Fidelity |
|---|---|---|---|
| base, 1024², 25 steps | 16.3 s (BF16) | **11.1 s** | 31-40 dB vs BF16, visually identical |
| base + LoRA, cfg 4 | 30.9 s | **22.4 s** | same composition |
| turbo, 8 steps | 3.59 s (community NVFP4) | 3.56 s | INT8 35-38 dB vs BF16; NVFP4 only 25-26 dB |

## MiniMax H3

Comfy-Org ships and recommends INT8 ConvRot for H3. A self-made NVFP4 (RTN from the official BF16
ref2v checkpoint) sampled at 315.7 s vs 302.9 s for INT8+Triton — **slower**, with visible
content drift (a lost garment detail, an extra hand). Rejected.

## Can NVFP4 be made more faithful?

- Learned rounding (convert_to_quant / CTQ 1.3.4, a "mixed" recipe with 184 NVFP4 + 61 MXFP8
  layers) gave **zero** gain on the NVFP4 layers: relative weight error 0.0944, identical to plain
  round-to-nearest (`tools/qerr.py`). Its optimizer stops early on these layers.
- The drift is dominated by **4-bit activations** (W4A4), not weight rounding. Only
  activation-aware methods (SVDQuant-style low-rank outliers, AWQ-style scaling) can narrow it.
- Published results and community reports agree: W4A4 hurts image-to-video and reference
  conditioning the most — exactly the H3 ref2v / i2v use case.

### CTQ gotchas (if you try it anyway)

- It processes tensors in **lexicographic** block order (`0, 1, 10..19, 2, 20..27, 3..9`) and writes
  its output **only at the end**. To split work across two machines, split by that order and merge
  (a merge script is trivial with safetensors); never stop a run midway.
- `--custom-layers` overrides `--exclude-layers`.
- A 12B DiT took hours per half on GB10.

## RTX PRO 2000 (16 GB) numbers for the same Krea model

| | s/img |
|---|---|
| FP8 eager | 36.4 |
| FP8 + `--fast fp8_matrix_mult` | 21.3 |
| INT8 ConvRot (stock CUTLASS is fine on sm_120) | 20.7 |
| **INT8 ConvRot from BF16** | **18.6-18.9** |
| RTN NVFP4 / CTQ NVFP4 | 15.8 / 15.1 (sibling images) |
| any `torch.compile` | unusable: INT8 OOM at 16 GB, NVFP4 dynamo SymInt assert, FP8mm+compile 65-79 s |
