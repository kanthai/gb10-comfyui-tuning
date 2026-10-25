# 2. INT8 GEMM on sm_121: from 37% to 88% of peak

## Symptom

INT8 (ConvRot) checkpoints — the format Comfy-Org recommends for MiniMax H3 and ships for
Qwen Image 2.1 — were **slower than BF16** on GB10, although INT8 tensor cores are 2x BF16.

## Root cause

comfy-kitchen 0.2.37 (also 0.2.40) routes INT8 linears on sm_121 to
`_C.cutlass_int8_dequant`: an INT8 GEMM with the per-row activation scale, per-channel weight
scale and bias applied in the epilogue. On GB10 that CUTLASS configuration reaches only
**~72 TOPS = 37% of the 196 TOPS INT8 peak** on H3 shapes (M = 29k-36k tokens, K/N = 5-28k).

Alternatives inside comfy-kitchen:

| Route | TOPS on H3 shapes | Notes |
|---|---|---|
| CUTLASS `int8_dequant` (default) | ~72 | slower than BF16 |
| cuBLAS (`COMFY_KITCHEN_DISABLE_CUTLASS=1`) | ~105 | int32 output round-trip costs ~3 GB scratch and bandwidth; LPDDR5x at 273 GB/s makes that expensive |
| **Triton fused-dequant GEMM (this patch)** | **~170** | 88% of peak, scales/bias fused in the epilogue |

On the RTX PRO 2000 (sm_120, GDDR7) the stock CUTLASS path is fine (1308 ms vs Triton 1509 ms per
forward), so this is GB10/sm_121-specific — install the patch only where it helps.

## The kernel

`patches/ck_int8_patch.py`: a standard Triton GEMM with grouped tile ordering:

- `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64`, `num_warps=8`, `num_stages=4`, `GROUP_M=8`
- `int8 x int8 -> int32` accumulate, then `acc * x_scale[row] * w_scale[col] + bias` and cast to
  the output dtype in the epilogue
- **int64 offsets** (`rm.to(tl.int64)`): `M*N >= 2^31` happens for real (M = 75k tokens at
  N = 28672, e.g. CFG batches or long clips). The first version used int32 offsets and an audit
  caught it.
- Config chosen by a sweep on H3 shapes (`tools/tri_int8.py`); larger tiles spilled, smaller ones
  lost ~15%.

### How it is hooked in

The patch replaces the extension module object `comfy_kitchen.backends.cuda._C` with a thin proxy:
`cutlass_int8_dequant` goes to Triton (same signature, all three call sites), every other
attribute falls through to the real extension. Calls with a **fused residual** and calls with
**M < 128** go to the real CUTLASS kernel (not worth a Triton launch, and keeps exotic paths stock).

Safety rails (each was added after something went wrong or an audit asked for it):

- **Version guard:** only activates on comfy-kitchen `0.2.37`; any other version = stock.
- **Mode file** `/etc/comfy-kitchen-int8.mode`: `off` | `cublas` | `triton`, read at process start.
  A typo logs a warning instead of silently doing nothing.
- **Fail-open:** if Triton cannot import or compile, the patch logs the traceback and leaves
  comfy-kitchen untouched. An early version broke `import comfy_kitchen` entirely on an import
  failure — never let a performance patch take down the import.
- **Shape/dtype checks** before dispatching to Triton; anything unexpected goes to CUTLASS.
- **`Python.h` fallback:** Triton JIT-compiles a small C launcher and needs Python headers. One
  node had no `python3.12-dev` and installing it would have bumped the system Python, so the
  patch points `CPATH` at a copied header directory when the venv's include dir lacks `Python.h`.

### Why a `.pth` file

DGX Monarch runs the DiT in separate actor processes (`python -P -s -B -m
monarch._src.actor.bootstrap_main`). `-s` skips *user* site-packages but the venv's site-packages
(and its `.pth` files) still load, so a `.pth` import hook patches both ComfyUI and every actor
with no changes to either. Change the mode file on **both** nodes of a pair, restart ComfyUI and
recycle the Monarch mesh so the actors re-import.

## Exactness

Integer accumulation is exact, and the epilogue applies the same float operations in the same
order as CUTLASS (`int32 -> fp32 * x_scale * w_scale + bias -> cast`). Verified:

- unit tests over odd shapes, bias/no bias, fp16/bf16 outputs: `torch.equal` with CUTLASS
- a full H3 render: sampled latent sha256, all 192 decoded frames and the audio identical.

## Results

| Workload | CUTLASS | Triton | |
|---|---|---|---|
| H3 8 s clip, sampling (8 steps, 2 nodes) | 369.7 s | 302.9 s | -18%, bit-identical |
| H3 10 s clip, total | 11:26 | 9:47 | |
| Qwen Image 2.1 INT8, 25 steps | (slower than BF16) | 11.1 s vs BF16 16.3 s | |
| Krea 2 INT8 ConvRot, 8 steps | 13.2 s single-node | (with compile: 10.0 s) | |

Per H3 step (~38 s after the patch, ~45 s before): linears ~15 s -> ~8 s, attention ~17 s (Sage 2
at ~73% of FP8 peak), everything else ~13 s.

## Upstream

The right fix is in comfy-kitchen: either a retuned CUTLASS config for sm_12x or a Triton
fallback there. A draft issue is in progress.
