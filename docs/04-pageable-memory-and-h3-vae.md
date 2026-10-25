# 4. The GB10 pageable-memory trap, and the H3 VAE fix

## The trap

On GB10, a GPU→host copy **into freshly allocated pageable memory** (never touched by the CPU)
runs at about **0.1 GB/s**. The same copy into memory the CPU has already written runs at
**~49 GB/s**. Pinned memory is also fast.

```python
gpu = torch.empty(2_400_000_000 // 2, dtype=torch.float16, device="cuda")
host = torch.empty_like(gpu, device="cpu")
host.copy_(gpu)            # ~19 s   (fresh pages)

host = torch.empty_like(gpu, device="cpu")
host.zero_()               # ~0.03 s (CPU first touch)
host.copy_(gpu)            # ~0.05 s
```

`tensor.cpu()` always allocates fresh memory, so it **always** hits the slow path for large
tensors. Transparent huge pages (`always`) did not help.

### Why (current best explanation)

GB10 is cache-coherent over NVLink-C2C and the GPU uses ATS (address translation through the CPU's
page tables). For pageable memory the copy engine can write straight into the process's virtual
addresses instead of bouncing through a pinned staging buffer. A page that was never touched has
no page-table entry yet, so each GPU write to it takes a translation fault that the CPU services
before the copy engine retries: ~30 µs per 4 KiB page is about what 2.4 GB in 19 s works out to.
Once the pages exist, the copy runs at full speed.

Related public reports (same mechanism class, other directions):

- NVIDIA driver 535.129.03 release notes (GH200, issue 4308011): pageable copies offloaded to copy
  engines were slow on non-resident pages; the driver now pages them in first
  (`CUDA_BUFFER_PAGE_IN_THRESHOLD_MS`).
- vLLM issue #58726 / PR #58868 (GB10): host→device loads from cold mmapped pages were ~170x slower
  per tensor; pre-faulting one byte per page cut a model load from 443 s to 26 s. The fix is gated
  on integrated GPUs.
- NVIDIA developer forum: "DGX Spark ARM64 CUDA 13: pathological slowdown for many small H2D
  copies from pageable CPU memory (50x vs pinned)".

`tools/d2h_matrix.py` is a micro-benchmark matrix to confirm this on your system: fresh vs
`zero_()` vs read-touch vs `MADV_POPULATE_WRITE` vs `cudaHostRegister` vs pinned, both directions,
with minor-fault counts and a `CUDA_BUFFER_PAGE_IN_THRESHOLD_MS` sweep. (If a read-touch stays
slow, the zero-page theory holds: a CPU *read* of untouched anonymous memory maps the shared zero
page read-only, so the GPU write still faults.)

### Rules that follow

- Pre-touch host buffers from the CPU (`buf.zero_()`), or allocate pinned memory once and reuse it.
- Prefer one bulk copy at the end over many per-chunk copies into fresh memory.
- Don't call `.cpu()` on big tensors in hot paths; copy into a pre-touched buffer instead.
- This also applies to services you write: our split-VAE peer (doc 05) pre-touches its reply buffer.

## Where it hit: the MiniMax H3 video VAE

`MiniMaxH3VideoVAE.decode_temporal` decodes the latent in temporal chunks (17-frame clips with an
overlap that is blended) and streams each finished chunk into a CPU output buffer, "so the full
video never sits on the GPU". On a discrete GPU that is a sensible VRAM saving. On GB10 the "CPU
buffer" is the same physical memory, and each per-chunk copy lands in fresh pages.

Profile of the 8 s / 192-frame decode: 47% one fp16 convolution GEMM, **26% per-chunk
device→host copies stalling the GPU**, the rest elementwise. Spatial tiling did not matter
(256 / 512 / 768 px tiles or untiled: 60-68 s, compute-bound).

### Fix (`patches/gb10_comfy_patch.py`)

On integrated GPUs only, when the caller passes a CPU output buffer:

1. decode into a **GPU** buffer of the same shape (unified memory, so this costs nothing extra),
2. `output_buffer.zero_()` (CPU first touch, ~0.03 s),
3. one `output_buffer.copy_(gpu_buffer)` (~0.05 s for 2.4 GB).

Discrete GPUs and every other call path are untouched. Disable with
`echo off > /etc/gb10-comfy-patch.mode`.

| | VAEDecode node |
|---|---|
| stock | 78.2 s |
| patched | **58.3 s** |

Frames are bit-identical (same chunk math, only the copy destination changed).

## Warning: benchmarking the VAE

Never call ComfyUI's full `comfy.sd.VAE.decode()` from a standalone script on a Spark that also
runs ComfyUI: it sized its working memory for the whole machine and drove the node into a global
OOM that killed ComfyUI, the user's systemd manager (and with it the Monarch worker) and unrelated
daemons. Benchmark through `vae.first_stage_model.decode(...)` (or `decode_temporal`) instead —
peak ~6 GB — and run tests under `systemd-run -p MemoryMax=... -p OOMScoreAdjust=900`.
