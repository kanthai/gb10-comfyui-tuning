# 9. What did not help (and why)

Negative results are results. Everything below was measured on the same production graphs.

## ComfyUI memory flags on GB10

Hypothesis: "ComfyUI isn't tuned for unified memory; `--gpu-only` / `--highvram` /
`--disable-dynamic-vram` should help." Measured on H3 (outside-sampling time per render):

| Flags | Non-sampling time |
|---|---|
| defaults (DynamicVRAM via comfy-aimdo + async offload) | **142 s** |
| `--gpu-only`, `--highvram`, `--disable-dynamic-vram` and combos | 172-240 s |

The defaults are right for GB10: on unified memory "offload" is nearly free and DynamicVRAM avoids
duplicate copies. `--fast autotune` and `--disable-comfy-compiler`: no effect. Tested on stock
ComfyUI as well as through Monarch, with the same conclusion. Note that ComfyUI command-line flags
reach only the ComfyUI process — Monarch actors start without ComfyUI's arguments.

## Attention kernels for H3

| Kernel | Sampling | Error / quality | Verdict |
|---|---|---|---|
| SageAttention 2 (INT8 QK, FP8 PV) | 302.9 s | baseline | **keep** |
| Sol-Attn tau 0.6 | 305.7 s | — | no gain |
| Sol-Attn tau 1.0 | 286.5 s (-5%) | visible motion drift (27 dB) | rejected |
| SageAttention 3 (FP4, Blackwell) | 358 ms/block vs 336 | rel. error 19% vs 4% | rejected: slower **and** worse at H3's shape |

SageAttention 3 had to be built for sm_121 with a C++17→C++20 change in `setup.py` and its
submodules initialized; install it into an isolated target to avoid touching the production venv.

## torch.compile on H3

Works after the fix in doc 03, but warm 315.2 s vs eager 302.9 s (cold 359.9 s): H3 at this
resolution is attention-bound and compile only adds overhead. Keep H3 eager.

## Monarch communication knobs

`sync_ulysses` is ignored by the pinned xfuser 0.7.0 (it never reaches `SeqAllToAll4D`);
`pipeline_depth` only overlaps consecutive renders inside the seed-sweep sampler node. Neither
changes a normal single render.

## Clock cap

The cluster runs with `nvidia-smi -lgc 0,2000`. Is it leaving performance on the table?

60-second burst (one node, room air-conditioned):

| | Clock | BF16 GEMM | INT8 GEMM | Power | SoC temp |
|---|---|---|---|---|---|
| capped | 1982 MHz | 84.6 TFLOPS | 136.5 TOPS | 61 W | 49→80 °C |
| uncapped | 2275 MHz (+15%) | 88.1 (+4%) | 131.3 (-4%) | 89 W (+45%) | 59→91 °C |

6-minute sustained mixed INT8+BF16 load (`tools/clock_sustain.sh`, watchdog-guarded):

| Cap | Throughput | SoC temp | Power |
|---|---|---|---|
| 2000 MHz | 105.2 T/s | 85 °C | 64 W |
| 2200 MHz | 107.2 (+2%) | 92 °C | 80 W |
| uncapped | 110.0 (+4.6%) | **96 °C**, clock sagging 2275→2186 | 87 W |

The GEMMs are bandwidth/power-bound: +45% power for +0-5%. Without air conditioning the uncapped
node throttles and ends up slower overall. **Keep the cap.**

## fp16 accumulation (`--fast fp16_accumulation`) for the H3 VAE

With fp16 accumulation enabled, ComfyUI routes the H3 VAE's 3-D convolutions to comfy-kitchen's
fp16-accumulate CUTLASS conv instead of cuDNN. On GB10 that is **48% slower** (56 frames: 20.9 s vs
14.1 s) at 63 dB vs the default. Without the flag the convolutions already run through cuDNN (the
"sm80 cutlass" kernel name in profiles is cuDNN's implicit-GEMM kernel), which is the faster path.

## torch.compile for Qwen Image 2.1 (INT8)

Monarch's `compile_dit` never compiles Qwen (it looks for `.blocks`). Stock `TorchCompileModel` on a
single node: **17.8 s compiled vs 17.5 s eager** per 1024² image (25 steps) — no gain. It also first
crashed: with DynamicVRAM, ComfyUI's malloc-graph recorder received a generic `torch.Stream`
(`'torch.Stream' object has no attribute 'cuda_stream'` in `comfy_aimdo/malloc_graph.py`) when the
model was wrapped by `TorchCompileModel`; `--disable-comfy-compiler` avoids that path.

## Monarch RDMA latent return

`rdma_latent_return` would return the sampled latent over one-sided RDMA. An H3 8 s latent is
~22 MB; actor messaging already returns it in well under a second, and Monarch documents native
RDMA as qualified only in a limited scope with failure recovery untested. Not worth it.

## SageAttention 2 variants (H3 per-rank shape: 28 heads, 29k tokens, head dim 128)

| Variant | ms | rel. error vs fp32 SDPA |
|---|---|---|
| PyTorch SDPA bf16 (flash) | 164.2 | 0.002 |
| Sage auto on sm_121: INT8 QK, FP8 PV, per-warp, `fp32+fp16` | 90.1 | 0.040 |
| same with `pv_accum_dtype="fp32+fp32"` | **86.5** (-4%, consistent at 36k tokens too) | 0.040 |
| per-thread QK quantization | 91.7 | 0.040 |
| FP16 PV, fp32 accumulate | 118.6 | **0.013** |
| FP16 PV, fp16+fp32 / Triton | 138.5 / 143.9 | 0.013 |

**Adopted.** Real 2-node H3 render, same seed: sampling 290.5 → **282.1 s** (−2.9%), total
405.8 → 398.0 s. Not bit-identical: same shots, composition and lighting, but motion takes a
slightly different path (per-frame PSNR 15-24 dB in fast action, 22 dB on a face close-up,
34-41 dB in a slow shot) — a "sibling" render of the same seed, judged visually indistinguishable
in quality. Enabled with `/etc/gb10-sage-pv.conf` = `fp32+fp32` (opt-in hook in
`patches/gb10_comfy_patch.py`).

Two traps on the way to a valid A/B:

- Under DGX Monarch's Ulysses split, attention does **not** go through ComfyUI's `attention.py`:
  xfuser/yunchang's `SAGE_AUTO` calls `sageattention.sageattn` itself. Patch the package
  attribute, not only ComfyUI's imported name. Verify with the latent hash — the first two
  "A/B" renders were bit-identical to the baseline, i.e. the variant never ran.
- `sageattention.sageattn` accepts and silently ignores unknown kwargs (yunchang passes
  `dropout_p`, `softmax_scale`, `causal`, `window_size`; ComfyUI passes `smooth_k=False`, which
  is therefore ignored too — production runs `smooth_k=True`). A wrapper must do the same or it
  silently falls back. Monarch's built-in `SAGE_FP8` option also uses `fp32+fp32` but with
  per-thread QK quantization and is only ~2.5% faster than auto; per-warp `fp32+fp32` is ~5.5%. The FP16-PV kernels are 3x more accurate but 31%
slower — an option if attention error is ever suspected in a quality problem.

## NVFP4 for speed

See doc 08: on GB10 with the Triton INT8 GEMM, NVFP4 is not faster for these models and it
changes the image.
