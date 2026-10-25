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

## NVFP4 for speed

See doc 08: on GB10 with the Triton INT8 GEMM, NVFP4 is not faster for these models and it
changes the image.
