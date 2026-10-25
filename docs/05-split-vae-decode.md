# 5. Splitting the H3 VAE decode across two nodes (bit-identical)

## Why this is possible

DGX Monarch deliberately keeps text encoders and VAEs on the driver node (its design doc explains
why: those stages are short, and splitting them adds complexity and cross-node failure modes). For
a 2-node H3 render that leaves the second GPU idle for the whole ~55 s VAE decode.

H3's `decode_temporal` is embarrassingly parallel: the latent is cut into temporal chunks
(`tokens_chunk_size + token_overlap` latent frames each) and **every chunk is decoded
independently**. Only the final step — trimming padding, blending each chunk's overlap with the
previous one, writing frames in order — is sequential, and it is cheap.

So: decode some chunks on the other node, ship the decoded pixels back over the 200 GbE fabric,
and do the blend/write on the driver exactly as upstream does.

## Design

- **Peer service** (`patches/h3_vae_peer.py`, `systemd/h3-vae-peer.service`) on the worker node:
  - plain `http.server`, bound to the fabric address only, with an allow-list of driver addresses
  - wire format: 8-byte header length + JSON header (shapes, dtypes, tiling settings, probe hash)
    + raw tensor bytes. **No pickle** — `torch.load` on network input would be remote code execution.
  - loads the VAE lazily on the first request and unloads it after 300 s idle, so it does not hold
    5 GB on the worker between renders; runs as the unprivileged ComfyUI user with `MemoryMax`
  - copies results GPU→host into a **pre-touched** buffer (see doc 04, otherwise the reply alone
    would take seconds)
- **Driver side** (`patches/gb10_comfy_patch.py`, `_split_decode`): a line-for-line mirror of
  upstream `decode_temporal`, except that the per-chunk `_adaptive_decode` results come from a
  dict filled by local decodes (first half of the chunks) and one HTTP request to the peer (second
  half), which run concurrently. Blending and writing are unchanged and in the original order.
- Enabled per driver by `/etc/gb10-vae-split.conf` containing `peer-host:port` (read on every
  decode, so deleting the file disables it without a restart).

## Guards (so it can never silently change output)

1. **Upstream source hash.** The patch hashes the source of the installed
   `MiniMaxH3VideoVAE.decode_temporal` (`inspect.getsource` + `textwrap.dedent`, sha256) and only
   enables splitting when it matches the version the mirror was written against. If ComfyUI
   changes that function, the patch falls back to the (still useful) GPU-buffer path from doc 04.
2. **Functional probe.** Both sides decode one fixed, small synthetic chunk with the current tiling
   settings and compare sha256 of the output bytes. A mismatch (different VAE file, dtype, tiling,
   kernels) refuses the split. This replaced a first attempt that hashed parameter bytes:
   inside ComfyUI, DynamicVRAM stages weights, so reading parameters directly does not give the
   real weights, and the hash never matched (the split refused and fell back — safe, but useless).
3. **Any error → local decode.** Peer down, timeout, HTTP error, wrong number of chunks: log one
   line and decode everything locally.

## Results

Offline (standalone, full 8 s latent of 192 frames = 11 chunks, 6 local + 5 remote):

| | Decode time |
|---|---|
| one node | 53.5 s |
| split | **31.0 s** |
| max abs difference | **0.0** |

In a real 2-node render the VAE step went from ~58 s to **32.7 s**, the whole job from ~431 s
to **415 s cold**, and the decoded frames and audio were hash-identical to a reference render of the
same seed without the split.

## Notes

- The chunk decode on the worker is slightly faster than on the driver (no ComfyUI process
  competing), so ~half/half is close to balanced; the join point is within ~1.5 s.
- Transfer is small relative to compute: ~1.3 GB of fp16 pixels over 200 GbE.
- Splitting across more than two nodes works the same way, but the sequential blend and the
  fixed per-request overhead make it pointless for an 11-chunk clip.
