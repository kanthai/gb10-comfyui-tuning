# Gotchas

Things that cost us time, roughly in order of how much.

## GB10 / CUDA

- **GPU→fresh pageable host memory runs at ~0.1 GB/s.** Pre-touch from the CPU (`zero_()`) or use
  pinned memory; `tensor.cpu()` on big tensors always hits it. THP does not help. (doc 04)
- **INT8 on sm_121 is slower than BF16 out of the box** (comfy-kitchen routes to a CUTLASS config at
  37% of peak). Fixed with a Triton GEMM. (doc 02)
- **NVFP4 PTX needs `compute_121a` / `sm_121a`**, not `sm_121`.
- **Triton needs `Python.h`** to build its launcher. Missing `python3.x-dev` = Triton import fails;
  point `CPATH` at a header copy rather than upgrading the system Python on a live box.
- **Use int64 offsets in custom GEMMs**: `M*N >= 2^31` is reachable (long clips, CFG batches).
- **Warm reboot can wedge the Spark's embedded controller**; cold power-cycle after firmware work.
- **Clock cap:** uncapped buys ~+4.6% for +45% power at 96 °C; keep a cap. (doc 09)

## ComfyUI

- **Identical prompts are cached.** A timed rerun needs a new seed (or `/free` with
  `free_memory`).
- **`/free` is deferred** until the prompt worker loops: after the current job, and after any job
  already queued. Free on an idle queue and wait a few seconds.
- **Never call `comfy.sd.VAE.decode()` from a standalone script on a busy Spark**: it caused a
  global OOM that killed ComfyUI and the user's systemd manager. Use
  `vae.first_stage_model.decode_temporal(...)` under `systemd-run -p MemoryMax=...`.
- **ComfyUI flags don't reach DGX Monarch actors** (they start without ComfyUI's args); `.pth`
  import hooks in the venv do.
- **`comfy/ops.py` `linear_input_act_` calls the Python `comfy_kitchen.int8_linear`, not the
  custom op** → torch.compile dies on FakeTensor `__dlpack__` for H3. (doc 03)
- **DynamicVRAM stages weights**: reading `model.parameters()` inside a live ComfyUI does not give
  you the real weights. Use a functional probe if you need to fingerprint a model.
- **Memory flags make GB10 slower**; keep the defaults. (doc 09)
- **With LoRAs, toggling DynamicVRAM alone moves Krea output ~19 dB.** Calibrate your "same
  quality" threshold before judging quantization.

## DGX Monarch

- **Back-to-back H3 renders get refused** by the activation preflight; the driver node carries
  ComfyUI's models *and* rank 0. `/free` alone sometimes suffices (saves ~50 s), a recycle always
  does. `empty_cache` in the actors does nothing useful. (doc 06)
- **Actor `TMPDIR` is tmpfs** → torch.compile cache lost on reboot; symlink it to disk. (doc 07)
- **`compile_dit` only compiles models with a `.blocks` list** (not Qwen's `transformer_blocks`).
- **`POST /interrupt` does not stop a running split render**; remove bad jobs from the queue
  before they start.
- **Two job queues feeding one pair deadlock-ish**: the second queue's jobs keep getting refused
  while the first keeps the pair busy. Feed each pair from exactly one sequential queue.
- **The recycle endpoint requires a CSRF-style header** (`X-DGXM-Action: recycle` plus `Origin`);
  without it you get 403.
- **Never silently fall back to one node** on a refused split: it is ~6x slower and you won't
  notice until the queue is hours late.

## Quantization

- **Quantize from the original BF16**, not from an FP8 derivative: faster *and* +4.7 dB more
  faithful in our Krea test. (doc 08)
- **Measure fidelity against the source model**, not against the outputs you are used to.
- **NVFP4 (W4A4) gives sibling images**; learned weight rounding does not fix activation error.
- **CTQ processes blocks in lexicographic order and writes only at the end.**

## Process / tooling

- **`pgrep -f myscript.sh` inside a waiting script matches itself** → waits forever.
- **`ssh host 'nohup cmd &'` keeps the ssh session open** and hangs your tool; use
  `systemd-run --unit=... -p StandardOutput=file:...`.
- **Don't put tokens on command lines** (visible in `ps`, and `pgrep -a` will print them); use a
  0600 header file.
- **Background auditors/agents need an explicit GPU window**, or they will run a crash test
  during a production render.
