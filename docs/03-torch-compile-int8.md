# 3. torch.compile + INT8: the one-call root cause

## Symptom

`torch.compile` on an INT8 MiniMax H3 model (DGX Monarch's `compile_dit`, or ComfyUI's
`TorchCompileModel`) crashed during tracing:

```
torch._dynamo ... FakeTensor ... __dlpack__
```

Krea 2 INT8 compiled fine, which made this look like an H3- or Monarch-specific problem. It isn't.

## Root cause

comfy-kitchen registers its INT8 linear as a proper PyTorch custom op,
`torch.ops.comfy_kitchen.int8_linear`, **with a fake (meta) kernel** (`register_fake`), so dynamo
can trace it without running it.

But ComfyUI's `comfy/ops.py` `linear_input_act_` path — used by H3's fused SwiGLU MLP
(`comfy/ldm/minimax/model.py`, the `fc2` input-activation linear) — calls the **plain Python
function** `comfy_kitchen.int8_linear(...)`. That function dispatches into the CUDA backend and
converts tensors with DLPack. Under dynamo the tensors are FakeTensors, and `__dlpack__` on a
FakeTensor fails.

Krea never calls `linear_input_act`, so it always went through the custom op and compiled fine.

## Fix

Upstream fix (one call site): `comfy/ops.py` should call `torch.ops.comfy_kitchen.int8_linear`
instead of the Python wrapper.

Runtime workaround (in `patches/ck_int8_patch.py`, "v4" part): wrap `comfy_kitchen.int8_linear`;
when `torch.compiler.is_compiling()` is true, route the call to the custom op with every argument
mapped explicitly (dtype -> the op's integer dtype code, convrot, groupsize, input activation,
residual...). Eager calls are untouched.

`tools/test_v4.py` compiles a SwiGLU INT8 linear and checks `torch.equal(compiled, eager)`.

## Did compile help?

| Model | Eager | Compiled | Verdict |
|---|---|---|---|
| Krea 2 INT8 ConvRot (Monarch Fleet, per image) | 12.5-12.6 s | **9.6-9.7 s** | keep compile |
| Krea 2 INT8 ConvRot (stock ComfyUI `TorchCompileModel`) | 13.2 s | **10.0 s** | keep compile |
| Krea 2 FP8 (Monarch) | 17.8 s | 14.4 s | keep compile |
| MiniMax H3 INT8 (2-node, warm) | 302.9 s | 315.2 s (cold 359.9 s) | **eager** — attention-bound, compile adds overhead |
| Qwen Image 2.1 | — | no effect | Monarch's `compile_dit` looks for a `.blocks` list; Qwen's are `transformer_blocks`, so nothing is compiled |
| NVFP4 / GGUF checkpoints | — | — | excluded: dynamo SymInt asserts / no benefit |

So the policy is: compile image DiTs (FP8/BF16/INT8), never H3, never NVFP4/GGUF.

Fidelity: compile changes kernel fusion, so outputs are *not* bit-identical; Krea images moved
19-26 dB vs eager (the same size as LoRA-path noise) and looked identical on contact sheets.
