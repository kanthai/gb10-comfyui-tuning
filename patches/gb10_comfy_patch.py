"""GB10 (integrated, unified-memory) fixes for ComfyUI, loaded from zz_gb10_comfy_patch.pth.

MiniMax H3 video VAE: decode_temporal streams each finished chunk into a CPU output buffer
("so the full video never sits on the GPU"). On GB10 that memory is the same physical RAM,
and the per-chunk GPU->host copies stall the decode: 8 s clip 63.5 s vs 52.4 s when decoding
into a GPU buffer (bit-identical, measured 2026-10-25). Here: decode into a GPU buffer, pre-touch
the CPU buffer from the CPU (GPU-driven first touch of fresh pageable pages runs at ~0.1 GB/s on
GB10), then one bulk copy (~0.05 s for 2.4 GB). Only on integrated CUDA devices with a CPU
output buffer; everything else is untouched. Disable with `echo off > /etc/gb10-comfy-patch.mode`.

Split decode (v3): when /etc/gb10-vae-split.conf names a peer ("host:port", running
h3_vae_peer.py on the other Spark), the last half of the temporal chunks is decoded there while
this host decodes the first half; blending and writing stay here in the original order. Each
chunk's decode is independent, so frames are bit-identical to a local decode. Used only when the
upstream decode_temporal source still matches the version this was written against and both
sides decode a fixed probe chunk to the same bytes; any error falls back to the local decode.
"""
import importlib.abc
import importlib.machinery
import sys

MODE_FILE = "/etc/gb10-comfy-patch.mode"
SPLIT_FILE = "/etc/gb10-vae-split.conf"
SAGE_FILE = "/etc/gb10-sage-pv.conf"     # "fp32+fp32" = opt-in SageAttention PV accumulation on sm_121
TARGET = "comfy.ldm.minimax.vae"
ATTN_TARGET = "comfy.ldm.modules.attention"
SAGE_PKG = "sageattention"   # Monarch's split path (xfuser/yunchang SAGE_AUTO) calls sageattention.sageattn
# sha256 of the dedented upstream MiniMaxH3VideoVAE.decode_temporal that _split_decode mirrors
SPLIT_RETRY_AFTER = 600   # seconds to skip the split after a peer failure
_split_state = {"skip_until": 0.0}
VALIDATED_DECODE_TEMPORAL = ("9ca4d63079a186e5a74e25fb6ae072c4ed1845b4dde29ca3f1c11c355045e2af",)


def _enabled():
    try:
        with open(MODE_FILE) as f:
            return f.read().strip().lower() != "off"
    except OSError:
        return True


def _split_peer():
    try:
        with open(SPLIT_FILE) as f:
            value = f.read().strip()
    except OSError:
        return None
    if not value or value.lower() == "off" or ":" not in value:
        return None
    host, port = value.rsplit(":", 1)
    return host, int(port)


def _probe_hash(model, torch, device):
    """Functional identity of a VAE decoder: hash of decoding one fixed small chunk.

    Reading parameters directly is not reliable inside ComfyUI (dynamic VRAM stages weights), and
    this also proves both sides compute bit-identical results with their current settings.
    """
    import hashlib
    t = model.tokens_chunk_size + model.token_overlap
    n = 24 * t * 4 * 4
    z = (torch.arange(n, dtype=torch.float32, device=device).reshape(1, 24, t, 4, 4) % 7 - 3) / 3
    dtype = next(model.decoder.parameters()).dtype
    with torch.inference_mode():
        out = model._adaptive_decode(z.to(dtype)).float().contiguous()
    host = torch.empty(out.shape, dtype=out.dtype)
    host.zero_()
    host.copy_(out)
    return hashlib.sha256(host.numpy().tobytes()).hexdigest()[:32]


def _pack(header, tensors, torch):
    import json
    import struct
    header = dict(header, tensors=[])
    blobs = []
    for t in tensors:
        t = t.contiguous()
        header["tensors"].append({"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")})
        blobs.append(t.view(-1).view(torch.uint8).cpu().numpy().tobytes())
    head = json.dumps(header).encode()
    return struct.pack("<Q", len(head)) + head + b"".join(blobs)


def _unpack(body, torch):
    import json
    import struct
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    (n,) = struct.unpack_from("<Q", body, 0)
    header = json.loads(body[8:8 + n])
    pos, tensors = 8 + n, []
    for spec in header["tensors"]:
        dtype = dtypes[spec["dtype"]]
        count = 1
        for d in spec["shape"]:
            count *= d
        nbytes = count * torch.empty((), dtype=dtype).element_size()
        tensors.append(torch.frombuffer(bytearray(body[pos:pos + nbytes]), dtype=dtype).view(spec["shape"]))
        pos += nbytes
    if pos != len(body):
        raise ValueError("peer reply length does not match header")
    return header, tensors


def _remote_decode(peer, payload, timeout):
    import http.client
    conn = http.client.HTTPConnection(peer[0], peer[1], timeout=timeout)
    try:
        conn.request("POST", "/decode", body=payload, headers={"Content-Type": "application/octet-stream"})
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"peer HTTP {resp.status}: {body[:300]!r}")
        return body
    finally:
        conn.close()


def _split_decode(self, z, output_buffer, peer, torch):
    """Mirror of upstream decode_temporal with the chunk decodes shared with a peer."""
    import threading
    import time
    t0 = time.time()
    chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
    split_count = int(self.token_drop > 0) + 1
    pad_tokens, num_chunks = self._decode_temporal_chunks(z.shape[2])
    if num_chunks < 2:
        raise ValueError("single chunk")
    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)
    clips = []
    for i in range(num_chunks):
        t_start_idx = i * self.tokens_chunk_size
        t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
        clips.append(z[:, :, t_start_idx:t_end_idx, :, :])
    n_local = num_chunks // 2            # peer gets the larger half: the driver also probes, blends and writes
    remote_ids = list(range(n_local, num_chunks))
    header = {"probe": _probe_hash(self, torch, z.device), "tiling": bool(self.tiling),
              "tile_size": int(self.tile_size), "tile_overlap_min": int(self.tile_overlap_min)}
    payload = _pack(header, [clips[i] for i in remote_ids], torch)
    box = {}

    def worker():
        try:
            box["body"] = _remote_decode(peer, payload, timeout=60 + 15 * len(remote_ids))
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    decoded = {}
    for i in range(n_local):
        decoded[i] = self._adaptive_decode(clips[i])
    t_local = time.time() - t0
    thread.join()
    rhead = {}
    try:
        if "error" in box:
            raise box["error"]
        rhead, rtensors = _unpack(box["body"], torch)
        if len(rtensors) != len(remote_ids):
            raise ValueError("peer returned the wrong number of chunks")
        for i, t in zip(remote_ids, rtensors):
            if t.dtype != decoded[0].dtype or t.shape[:2] != decoded[0].shape[:2] or t.shape[3:] != decoded[0].shape[3:]:
                raise ValueError(f"peer chunk {i} has shape {tuple(t.shape)} {t.dtype}")
            decoded[i] = t.to(z.device)
    except Exception as exc:
        # keep the local half; decode the peer's chunks here (same result as a local decode)
        _split_state["skip_until"] = time.time() + SPLIT_RETRY_AFTER
        print(f"[gb10-comfy-patch] split peer failed ({exc!r:.300}); decoding its {len(remote_ids)} chunks locally, "
              f"split paused {SPLIT_RETRY_AFTER} s", file=sys.stderr, flush=True)
        for i in remote_ids:
            decoded[i] = self._adaptive_decode(clips[i])
    t_wait = time.time() - t0

    dec = output_buffer
    dec_overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal write_pos
        part_frames = part.shape[2]
        if part_frames <= 0:
            return
        part = self._finalize_pixels(part)
        copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
        if copy_frames > 0:
            dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(
                part[:, :, :copy_frames, :, :]
            )
            write_pos += copy_frames

    for i in range(num_chunks):
        clip_dec = decoded.pop(i)
        for j in range(split_count):
            f_start_idx = j * chunk_dec
            f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
            clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
            clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding:, :, :]

            if j == 0:
                if dec_overlap is not None:
                    clip_dec_chunk = self.blend(
                        dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                    )
                    dec_overlap = None
                write_part(clip_dec_chunk)
            else:
                dec_overlap = clip_dec_chunk.contiguous()

        if i == num_chunks - 1 and dec_overlap is not None:
            write_part(dec_overlap)
            dec_overlap = None

        del clip_dec, clip_dec_chunk
    print(f"[gb10-comfy-patch] H3 VAE split decode: {n_local} local + {len(remote_ids)} on {peer[0]} "
          f"(local {t_local:.1f} s, peer {rhead.get('decode_s')} s, joined at {t_wait:.1f} s, total {time.time() - t0:.1f} s)",
          file=sys.stderr, flush=True)
    return dec


def _apply(mod):
    try:
        import torch
        cls = getattr(mod, "MiniMaxH3VideoVAE", None)
        if cls is None or getattr(cls.decode_temporal, "_gb10_gpu_buffer", False):
            return
        if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).is_integrated:
            return
        orig = cls.decode_temporal
        import hashlib
        import inspect
        import textwrap
        src_hash = hashlib.sha256(textwrap.dedent(inspect.getsource(orig)).encode()).hexdigest()
        split_ok = src_hash in VALIDATED_DECODE_TEMPORAL

        def decode_into(self, z, buffer):
            import time
            peer = _split_peer() if split_ok and time.time() >= _split_state["skip_until"] else None
            if peer is not None:
                try:
                    return _split_decode(self, z, buffer, peer, torch)
                except Exception as exc:
                    print(f"[gb10-comfy-patch] split decode failed, decoding locally ({exc!r:.300})",
                          file=sys.stderr, flush=True)
            return orig(self, z, buffer)

        def decode_temporal(self, z, output_buffer=None):
            if output_buffer is None or output_buffer.device.type != "cpu" or not z.is_cuda:
                if output_buffer is None or not z.is_cuda:
                    return orig(self, z, output_buffer)
                return decode_into(self, z, output_buffer)
            gpu = torch.empty(output_buffer.shape, dtype=output_buffer.dtype, device=z.device)
            decode_into(self, z, gpu)
            output_buffer.zero_()            # CPU first touch: fast; GPU-driven first touch is not
            output_buffer.copy_(gpu)
            del gpu
            return output_buffer

        decode_temporal._gb10_gpu_buffer = True
        cls.decode_temporal = decode_temporal
        print(f"[gb10-comfy-patch] H3 VAE decode: GPU buffer + one bulk copy; split decode "
              f"{'available' if split_ok else 'DISABLED (upstream decode_temporal changed)'}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[gb10-comfy-patch] NOT applied ({exc!r:.200})", file=sys.stderr, flush=True)


def _apply_sage(mod):
    """Opt-in: SageAttention 2 on sm_121 with fp32+fp32 PV accumulation instead of the auto fp32+fp16.

    Same INT8-QK / FP8-PV per-warp kernel, ~4% faster at H3 shapes (86.5 vs 90.1 ms at 29k tokens)
    with the same error vs fp32 SDPA (0.040). Not bit-identical to the default, hence opt-in.
    """
    try:
        try:
            with open(SAGE_FILE) as f:
                want = f.read().strip()
        except OSError:
            return
        if want != "fp32+fp32" or getattr(getattr(mod, "sageattn", None), "_gb10_sage", False):
            return
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1):
            return
        from sageattention import core
        orig = mod.sageattn

        def sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False,
                     smooth_k=True, **kwargs):
            # like sageattention.sageattn, extra kwargs (yunchang passes dropout_p, softmax_scale, causal,
            # window_size, ...; ComfyUI passes smooth_k) are accepted and ignored
            if q.dtype not in (torch.float16, torch.bfloat16):
                return orig(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale,
                            return_lse=return_lse, smooth_k=smooth_k, **kwargs)
            # mirror sageattn()'s sm_121 branch exactly except the accumulator: that branch swallows
            # smooth_k in **kwargs (ComfyUI passes smooth_k=False), so the kernel default (True) applies
            return core.sageattn_qk_int8_pv_fp8_cuda(
                q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp",
                sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp32")

        sageattn._gb10_sage = True
        mod.sageattn = sageattn
        print(f"[gb10-comfy-patch] SageAttention sm_121: pv_accum fp32+fp32 (opt-in) in {mod.__name__}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[gb10-comfy-patch] Sage hook NOT applied ({exc!r:.200})", file=sys.stderr, flush=True)


_HOOKS = {TARGET: _apply, ATTN_TARGET: _apply_sage, SAGE_PKG: _apply_sage}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _HOOKS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _HOOKS[name](module)

        spec.loader.exec_module = exec_module
        return spec


try:
    if _enabled():
        for _name, _hook in _HOOKS.items():
            if _name in sys.modules:
                _hook(sys.modules[_name])
        sys.meta_path.insert(0, _Finder())
except Exception as exc:  # never break interpreter start-up
    print(f"[gb10-comfy-patch] disabled: {exc!r}", file=sys.stderr)
