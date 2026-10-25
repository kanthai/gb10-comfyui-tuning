"""Optional INT8 GEMM override for comfy-kitchen on GB10 (sm_121).

Loaded from zz_ck_int8_patch.pth in the venv so every process (ComfyUI and the
DGX Monarch actors) gets the same kernels. Inert unless /etc/comfy-kitchen-int8.mode
says otherwise. Modes:
  off     - stock comfy-kitchen (CUTLASS int8_dequant)
  cublas  - stock cuBLAS fallback (same as COMFY_KITCHEN_DISABLE_CUTLASS=1)
  triton  - stock CUTLASS route, but _C.cutlass_int8_dequant (int8 GEMM with the
            row/col scaling and bias in its epilogue) runs as a Triton kernel tuned
            for GB10; ~170 TOPS vs ~72 on MiniMax H3 shapes. Fused-residual calls
            and small M stay on the real CUTLASS kernel.
Only active on the comfy-kitchen version it was validated against.

In cublas/triton mode it also makes the top-level comfy_kitchen.int8_linear compile-safe:
ComfyUI's linear_input_act (H3's swiglu MLP) calls that plain Python function, so torch.compile
traced into the CUDA backend and died on __dlpack__ of a FakeTensor. While dynamo is tracing,
the call is routed through the registered custom op comfy_kitchen::int8_linear (which has a
fake kernel); eager calls are untouched.
"""
import importlib.abc
import importlib.machinery
import os
import sys
import sysconfig

MODE_FILE = "/etc/comfy-kitchen-int8.mode"
TARGET = "comfy_kitchen.backends.cuda"
TOP = "comfy_kitchen"
VALIDATED_KITCHEN = ("0.2.37",)
MIN_M = 128
PY_HEADERS_FALLBACK = "/opt/py312-include"  # dgxspark2 has no python3.12-dev; Triton needs Python.h


def _read_mode():
    try:
        with open(MODE_FILE) as f:
            return f.read().strip().lower() or "off"
    except OSError:
        return "off"


def _kitchen_version():
    try:
        from importlib.metadata import version
        return version("comfy-kitchen")
    except Exception:
        return None


def _ensure_python_headers():
    inc = sysconfig.get_paths().get("include", "")
    if os.path.exists(os.path.join(inc, "Python.h")):
        return
    alt = os.path.join(PY_HEADERS_FALLBACK, "python3.12")
    if os.path.exists(os.path.join(alt, "Python.h")):
        os.environ["CPATH"] = os.pathsep.join(p for p in (alt, PY_HEADERS_FALLBACK, os.environ.get("CPATH")) if p)


def _build_triton_gemm():
    _ensure_python_headers()
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(a, b, xs, ws, bias, c, M, N, K,
                sam, sak, sbn, sbk, scm, scn, ws_n,
                HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, G: tl.constexpr):
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        gid = pid // (G * npn)
        fm = gid * G
        gs = min(npm - fm, G)
        pm = fm + (pid % gs)
        pn = (pid % (G * npn)) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        # int64 offsets: M*N or M*K past 2^31 elements (CFG batches, long clips) would wrap in int32
        rm64 = rm.to(tl.int64)
        rn64 = rn.to(tl.int64)
        A = a + rm64[:, None] * sam + rk[None, :] * sak
        B = b + rn64[None, :] * sbn + rk[:, None] * sbk
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, K, BK):
            km = (k0 + rk) < K
            acc += tl.dot(tl.load(A, mask=(rm[:, None] < M) & km[None, :], other=0),
                          tl.load(B, mask=(rn[None, :] < N) & km[:, None], other=0))
            A += BK * sak
            B += BK * sbk
        out = acc.to(tl.float32) * tl.load(xs + rm, mask=rm < M, other=0.0)[:, None]
        out = out * tl.load(ws + rn * ws_n, mask=rn < N, other=0.0)[None, :]
        if HAS_BIAS:
            out += tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
        C = c + rm64[:, None] * scm + rn64[None, :] * scn
        tl.store(C, out.to(c.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

    def gemm(x8, w8, x_scale, w_scale, bias, out):
        M, K = x8.shape
        N = w8.shape[0]
        xs = x_scale.reshape(-1)
        ws = w_scale.reshape(-1)
        ws_n = 0 if ws.numel() == 1 else 1
        has_bias = bias.numel() > 0
        # 128x128x64 / 8 warps / 4 stages / group 8: best or within 1% on every
        # H3 shape at 29k and 36k tokens (sweep 2026-10-25).
        grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
        _kernel[grid](x8, w8, xs, ws, bias if has_bias else out, out, M, N, K,
                      x8.stride(0), x8.stride(1), w8.stride(0), w8.stride(1),
                      out.stride(0), out.stride(1), ws_n,
                      HAS_BIAS=has_bias, BM=128, BN=128, BK=64, G=8,
                      num_warps=8, num_stages=4)

    return gemm


class _CProxy:
    """Stands in for comfy_kitchen.backends.cuda._C and takes over cutlass_int8_dequant."""

    def __init__(self, real, gemm):
        self._real = real
        self._gemm = gemm
        self._broken = False

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._real, name)

    def cutlass_int8_dequant(self, x, w, x_scale, w_scale, bias, out, dtype_code, stream):
        import torch
        tx, tw, txs, tws, tb, tout = (torch.from_dlpack(t) for t in (x, w, x_scale, w_scale, bias, out))
        M, K = tx.shape if tx.dim() == 2 else (0, 0)
        N = tw.shape[0] if tw.dim() == 2 else 0
        # Mirror CUTLASS eligibility (K % 16, contiguous) so anything CUTLASS would decline (and ck then
        # routes to cuBLAS) keeps the stock route; the Triton kernel itself would handle those too.
        if (self._broken or M < MIN_M or K % 16 or not (tx.is_contiguous() and tw.is_contiguous() and tout.is_contiguous())
                or tx.dtype != torch.int8 or tw.dtype != torch.int8
                or tw.dim() != 2 or tw.shape[1] != K or tuple(tout.shape) != (M, N)
                or tout.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or txs.dtype != torch.float32 or txs.numel() != M
                or tws.dtype != torch.float32 or tws.numel() not in (1, N)
                or tb.numel() not in (0, N) or (tb.numel() and tb.dtype != tout.dtype)):
            return self._real.cutlass_int8_dequant(*(t.__dlpack__(stream=-1) for t in (tx, tw, txs, tws, tb, tout)),
                                                   dtype_code, stream)
        try:
            self._gemm(tx, tw, txs, tws, tb, tout)
        except Exception as exc:  # e.g. Triton cannot build its launcher: stay on stock CUTLASS
            import traceback
            self._broken = True
            print(f"[ck-int8-patch] Triton GEMM failed ({exc!r:.200}); stock CUTLASS for the rest of this process\n"
                  + traceback.format_exc(), file=sys.stderr, flush=True)
            return self._real.cutlass_int8_dequant(*(t.__dlpack__(stream=-1) for t in (tx, tw, txs, tws, tb, tout)),
                                                   dtype_code, stream)
        return True


def _apply(mod, mode):
    try:
        _apply_unguarded(mod, mode)
    except Exception as exc:
        print(f"[ck-int8-patch] NOT applied ({exc!r:.200}); running stock", file=sys.stderr, flush=True)


def _apply_unguarded(mod, mode):
    ver = _kitchen_version()
    if ver not in VALIDATED_KITCHEN:
        print(f"[ck-int8-patch] NOT applied: comfy-kitchen {ver} is not validated "
              f"({', '.join(VALIDATED_KITCHEN)}); running stock", file=sys.stderr, flush=True)
        return
    if mode == "cublas":
        mod._DISABLE_CUTLASS_INT8 = True
    elif mode == "triton":
        mod._C = _CProxy(mod._C, _build_triton_gemm())
    print(f"[ck-int8-patch] comfy_kitchen {ver} int8 mode={mode}", file=sys.stderr, flush=True)


def _compile_safe_int8_linear(top):
    """Route comfy_kitchen.int8_linear through its custom op while torch.compile traces."""
    try:
        import torch
        from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
        if _kitchen_version() not in VALIDATED_KITCHEN or getattr(top.int8_linear, "_ck_compile_safe", False):
            return
        code_of = {dt: code for code, dt in DTYPE_CODE_TO_DTYPE.items()}
        eager = top.int8_linear

        def int8_linear(x, weight, weight_scale, bias=None, out_dtype=None, convrot=False,
                        convrot_groupsize=256, input_act=None, input_act_weight=None,
                        input_act_eps=0.0, residual=None, residual_scale=None):
            if torch.compiler.is_compiling():
                return torch.ops.comfy_kitchen.int8_linear(
                    x, weight, weight_scale, bias, code_of[out_dtype or torch.bfloat16],
                    convrot, convrot_groupsize, input_act, input_act_weight, input_act_eps,
                    residual, residual_scale)
            return eager(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                         input_act, input_act_weight, input_act_eps, residual, residual_scale)

        int8_linear._ck_compile_safe = True
        int8_linear.__wrapped__ = eager
        top.int8_linear = int8_linear
    except Exception as exc:
        print(f"[ck-int8-patch] compile-safe int8_linear NOT applied ({exc!r:.200})", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, mode):
        self.mode = mode

    def find_spec(self, name, path, target=None):
        if name not in (TARGET, TOP):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        loader, mode = spec.loader, self.mode
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            if name == TARGET:
                _apply(module, mode)
            else:
                _compile_safe_int8_linear(module)

        loader.exec_module = exec_module
        return spec


def install():
    mode = _read_mode()
    if mode not in ("off", "cublas", "triton"):
        print(f"[ck-int8-patch] unknown mode {mode!r} in {MODE_FILE}; running stock", file=sys.stderr, flush=True)
    if mode not in ("cublas", "triton"):
        return
    if TARGET in sys.modules:
        _apply(sys.modules[TARGET], mode)
    if TOP in sys.modules and hasattr(sys.modules[TOP], "int8_linear"):
        _compile_safe_int8_linear(sys.modules[TOP])
    if TARGET not in sys.modules or TOP not in sys.modules:
        sys.meta_path.insert(0, _Finder(mode))


try:
    install()
except Exception as exc:  # never break interpreter start-up
    print(f"[ck-int8-patch] disabled: {exc!r}", file=sys.stderr)
