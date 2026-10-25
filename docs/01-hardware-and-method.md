# 1. Hardware ceilings and measurement method

## The cluster

- 4x NVIDIA DGX Spark: GB10 Grace-Blackwell, **sm_121, 48 SMs**, 128 GB unified LPDDR5x
  (~273 GB/s), integrated GPU (`torch.cuda.get_device_properties(0).is_integrated == True`),
  ConnectX-7 200 GbE RoCE between nodes. DGX OS 7.x (Ubuntu 24.04, aarch64).
- Software: ComfyUI 0.41.3, comfy-kitchen 0.2.37, comfy-aimdo 0.5.6 (DynamicVRAM),
  DGX Monarch 1.3.0, PyTorch 2.14 + CUDA 13, SageAttention 2 built for sm_121.
- Nodes run as **pairs**: one ComfyUI "driver" (also Monarch rank 0) + one worker (rank 1).
  Monarch splits one render across both GPUs (Ulysses sequence parallel for H3, data-parallel
  "Fleet" for image batches).
- The GPU clock is deliberately capped at `nvidia-smi -lgc 0,2000` (see
  [09](09-what-did-not-help.md#clock-cap) for why it stays).

## Measured tensor-core ceilings (mma.sync microbenchmark, `tools/mma_peak2.cu`)

At 1989 MHz on GB10:

| Precision | Dense TOPS | Ops/clk/SM |
|---|---|---|
| NVFP4 (needs `-gencode arch=compute_121a,code=sm_121a`) | 391 | 4096 |
| INT8 | 196 | 2048 |
| FP8 | 196 | 2048 |
| BF16 | 98 | 1024 |

Sustained large BF16 GEMM through cuBLAS: ~85 TFLOPS. For reference, an A100 does 624 dense INT8
TOPS, so GB10 is roughly 1/3 of an A100 on INT8 — but with 128 GB of memory and no PCIe hop.

**Takeaway:** INT8 and FP8 are 2x BF16 on paper. Anything quantized that runs *slower* than BF16
is a kernel problem, not a hardware limit — which is exactly what we found (doc 02).

## Method (what made results trustworthy)

- **Use the real production graph** pulled from ComfyUI `/history`, swap only the variable under
  test, write outputs to a separate folder.
- **Same seed, same graph.** ComfyUI caches identical prompts: a resubmission with only a new
  filename prefix "renders" in 0.45 s. A timed second pass must change the seed (or clear the cache).
- **1 warm-up + 2 timed passes.** The first image after a model change includes loading, Monarch's
  first-use identity gate and Triton/inductor compilation.
- **Exactness = hashes, not eyeballs.** Monarch logs a sha256 of the sampled latent; decoded video
  frames are hashed after decoding the MP4 to RGB (`tools/frame_md5.py`; MP4 file hashes differ by
  container metadata). "Bit-identical" in these notes always means equal hashes.
- **Quality for non-exact changes** = PSNR vs a reference render **plus** a contact sheet looked at
  by a human. Same-seed PSNR alone cannot tell "worse" from "a sibling image" (different
  composition of equal quality). With LoRAs, merely toggling ComfyUI's DynamicVRAM moves Krea
  output by ~19 dB, so ~17-22 dB between two *valid* configurations is normal there.
- **Per-node timing** via ComfyUI's websocket (`tools/ws_timing.py`): gives seconds per node
  (text encode, sampler, VAE decode, save) instead of one total.
- **Profile before guessing:** `torch.profiler` / nsys on a standalone repro of the hot op.
