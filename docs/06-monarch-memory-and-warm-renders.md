# 6. DGX Monarch: memory preflights, warm renders, and the cost of recycling

## The refusals

Back-to-back H3 renders on a 2-node pair were often refused before sampling with:

```
[dgxm:C guard=activation_footprint_preflight] minimax_h3 activation preflight:
35.6 GiB needed per rank, 33.3 GiB usable (38.3 GiB MemAvailable minus a 5.0 GiB reserve)
```

The usual remedy — free ComfyUI's memory and **recycle the Monarch mesh** — always works but costs
~38 s per render: actor respawn (~8 s), weight load (~9 s), LoRA bake, conditioning transfer.

## Where the memory actually is

On unified memory, Monarch checks `MemAvailable` per node. After one H3 render the driver node
(ComfyUI driver **and** Monarch rank 0) had 38 GiB available; the worker node (rank 1 only) had 63.

| Holder on the driver node | Size | Freed by ComfyUI's `/free`? |
|---|---|---|
| ComfyUI: H3 text encoder (~15 GB), VAE (~5 GB), cached node outputs (decoded frames, conditioning) | ~28 GB | yes |
| Monarch rank-0 actor: resident H3 DiT + LoRA original-weight backup + allocator pool | ~34 GB RSS | **no** — only a recycle or worker restart |

Things that do **not** help:

- `torch.cuda.empty_cache()` in the actors after each render (tested via a hook): the cached-free
  pool is small; the resident model is the bulk.
- Monarch's "Clear VRAM" node in `soft` mode — same thing.

A second, opposite refusal exists too: the *loader* preflight on the driver
(`driver-side footprint preflight ... 19.5 GiB to place on the driver host, 18.3 GiB usable`)
fires when the actors still hold their memory and ComfyUI wants to load the text encoder + VAE.
Only a recycle fixes that one.

## ComfyUI's `/free` is deferred

`POST /free {"unload_models": true, "free_memory": true}` only **sets flags**. ComfyUI's prompt
worker applies them in its loop *after* the current job finishes — and if another job is already
queued, that job runs first and the free happens after it. "Clear, then immediately queue" or
"clear while something is queued" frees nothing for that job. When the queue is idle the flag is
applied within a moment; wait a few seconds before submitting.

## Warm renders

Freeing only ComfyUI's side (no recycle) and keeping the H3 DiT resident in the actors:

| | Sampling | Total |
|---|---|---|
| after recycle | 305.2 s (incl. 9 s load) | 440.1 s |
| warm (driver `/free` only) | 290.5 s (`reuse`) | **386.1 s** |

Latent hashes of a warm render and a recycled render of the same seed were identical.

**But the margin is thin.** On the driver node the admitted figure was 45.4 GiB usable vs
40.6 GiB needed. In a later run the rank-0 actor held ~8 GB more GPU memory after the previous
render and the same procedure was refused (35.9 GiB). Our job client therefore does:

1. if the queue is idle: `/free` (driver models + cache), wait 5 s, submit — no recycle;
2. on a capacity refusal: wait for an idle queue, `/free` + recycle, retry (max 3 tries);
3. never silently fall back to a single node (~6x slower) unless explicitly asked.

Keeping ComfyUI's node cache (`/free` with only `unload_models`) was also admitted once (53 GiB
available) and a seed-only change then took 330 s, because the text and reference encodes were
cached. In real use the prompt changes per shot, so the realistic gain is just the empty negative
encode (~5.6 s), for ~8 GiB less headroom — not worth it while the margin is this thin.

The structural fix belongs in Monarch: the driver could release its own TE/VAE/cache before the
activation check, or account for it as reclaimable.

## Other Monarch knobs we checked

- `lora_low_rss` (default `auto` = on for unified memory) and `slab_weights` (experimental
  zero-copy slab) — already in effect / not a capacity win for H3.
- `reserve_vram_gb` (default 8 GiB on unified memory) and the 5 GiB driver floor — lowering
  reserves to squeeze a render through is not worth an OOM.
- `sync_ulysses`: ignored by the pinned xfuser version. `pipeline_depth`: only overlaps consecutive
  renders inside the seed-sweep sampler node, not normal single renders.
