"""H3 video-VAE decode peer: decodes temporal chunks for the ComfyUI driver on the other Spark.

The driver (gb10_comfy_patch, MiniMaxH3VideoVAE.decode_temporal) sends some latent chunks here
while it decodes the rest; each chunk's decode is independent, so the result is the same as a
local decode. Wire format (no pickle): 8-byte little-endian header length, JSON header, raw
tensor bytes back to back. Only accepts the driver's fabric address. The VAE is loaded on the
first request and dropped after IDLE_UNLOAD seconds so it does not hold 5 GB between renders.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/opt/ComfyUI")

BIND = os.environ["H3_VAE_PEER_BIND"]                          # this node's fabric/LAN address
PORT = int(os.environ.get("H3_VAE_PEER_PORT", "8290"))
ALLOWED = set(os.environ["H3_VAE_PEER_ALLOW"].split(","))      # driver address(es), comma-separated
VAE_PATH = os.environ.get("H3_VAE_PATH", "/opt/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors")
IDLE_UNLOAD = float(os.environ.get("H3_VAE_IDLE_UNLOAD", "1800"))
MAX_BODY = 1 << 30

_lock = threading.Lock()
_state = {"model": None, "fingerprint": None, "probe_key": None, "last": 0.0}
DTYPES = {}


def _torch():
    import torch
    if not DTYPES:
        DTYPES.update({"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32})
    return torch


def probe_hash(model, torch, device):
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


def _load():
    torch = _torch()
    import comfy.sd
    import comfy.utils
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE_PATH))
    model = vae.first_stage_model.to("cuda").to(vae.vae_dtype).eval()
    _state.update(model=model, fingerprint=None)
    print(f"[h3-vae-peer] loaded {os.path.basename(VAE_PATH)} dtype={vae.vae_dtype}", flush=True)


def _unloader():
    while True:
        time.sleep(30)
        with _lock:
            if _state["model"] is not None and time.time() - _state["last"] > IDLE_UNLOAD:
                _state.update(model=None, fingerprint=None, probe_key=None)
                import gc
                gc.collect()
                _torch().cuda.empty_cache()
                print("[h3-vae-peer] idle: VAE unloaded", flush=True)


def pack(header, tensors):
    blobs = []
    header = dict(header, tensors=[])
    for t in tensors:
        t = t.contiguous()
        header["tensors"].append({"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")})
        if t.device.type != "cpu":
            host = _torch().empty(t.shape, dtype=t.dtype)
            host.zero_()        # CPU first touch; a GPU->fresh-pageable copy runs at ~0.1 GB/s on GB10
            host.copy_(t)
            t = host
        blobs.append(t.view(-1).view(_torch().uint8).numpy().tobytes())
    head = json.dumps(header).encode()
    return struct.pack("<Q", len(head)) + head + b"".join(blobs)


def unpack(body):
    torch = _torch()
    (n,) = struct.unpack_from("<Q", body, 0)
    header = json.loads(body[8:8 + n])
    pos, tensors = 8 + n, []
    for spec in header["tensors"]:
        dtype = DTYPES[spec["dtype"]]
        count = 1
        for d in spec["shape"]:
            count *= d
        nbytes = count * torch.empty((), dtype=dtype).element_size()
        tensors.append(torch.frombuffer(bytearray(body[pos:pos + nbytes]), dtype=dtype).view(spec["shape"]))
        pos += nbytes
    if pos != len(body):
        raise ValueError("body length does not match header")
    return header, tensors


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.client_address[0] not in ALLOWED:
            return self._reply(403, b"forbidden", "text/plain")
        try:   # lets the driver skip the split while this GPU renders its own job (solo mode)
            util = int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=3).stdout.split()[0])
        except Exception:
            util = -1
        busy = util >= 50 and not _lock.locked()
        info = {"loaded": _state["model"] is not None, "fingerprint": _state["fingerprint"], "vae": VAE_PATH,
                "gpu_util": util, "busy": busy}
        self._reply(200, json.dumps(info).encode(), "application/json")

    def do_POST(self):
        if self.client_address[0] not in ALLOWED or self.path != "/decode":
            return self._reply(403, b"forbidden", "text/plain")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_BODY:
            return self._reply(400, b"bad length", "text/plain")
        try:
            header, chunks = unpack(self.rfile.read(length))
            torch = _torch()
            with _lock:
                if _state["model"] is None:
                    _load()
                model = _state["model"]
                model.tiling = bool(header["tiling"])
                model.tile_size = int(header["tile_size"])
                model.tile_overlap_min = int(header["tile_overlap_min"])
                key = (model.tiling, model.tile_size, model.tile_overlap_min)
                if _state.get("probe_key") != key:
                    _state.update(fingerprint=probe_hash(model, torch, "cuda"), probe_key=key)
                if header.get("probe") != _state["fingerprint"]:
                    return self._reply(409, f"vae probe mismatch (peer {_state['fingerprint']})".encode(), "text/plain")
                t0 = time.time()
                out = []
                with torch.inference_mode():
                    for clip_z in chunks:
                        out.append(model._adaptive_decode(clip_z.to("cuda")))
                torch.cuda.synchronize()
                dt = time.time() - t0
                body = pack({"decode_s": round(dt, 3)}, out)
                del out
                _state["last"] = time.time()
            self._reply(200, body)
        except Exception as exc:  # report, never crash the server
            self._reply(500, f"{type(exc).__name__}: {exc}"[:500].encode(), "text/plain")

    def log_message(self, fmt, *args):
        print("[h3-vae-peer] " + fmt % args, flush=True)


if __name__ == "__main__":
    threading.Thread(target=_unloader, daemon=True).start()
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
