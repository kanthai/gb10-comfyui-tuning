# 11. Scaling: one 4-node split or two 2-node pairs?

DGX Monarch's design target is 2-8 nodes, but its recorded timings cover two. H3's 56 attention
heads divide by 4, so a 4-way Ulysses split is feasible in principle.

## The right comparison

A 4-way split does not have to beat a 2-way split; it has to beat **two independent 2-way pairs
rendering two clips at the same time** — that is the alternative use of the same four nodes.

| Setup | One 8 s clip | Clips/hour |
|---|---|---|
| one pair, 2-way | ~415 s (≈360-385 s warm) | ~9 |
| **two pairs, 2-way each** | ~415 s each, two at a time | **~17-18** |
| one 4-way split (estimate) | ~250-310 s | ~12-14 |

About 135 s of every H3 job is not sampling (text encoder + reference encoding, VAE decode, muxing,
Monarch overhead) and does not shrink with more ranks. To match two pairs a 4-way clip would need
~210 s, i.e. ~75 s of sampling — impossible when 2-way sampling is ~300 s and a perfect 4-way split
would still be ~150 s plus more all-to-all communication.

**4-way only wins on latency for a single clip** (iterating on one shot). For throughput, run two
pairs.

## Measured: one clip per Spark vs a split pair (2026-10-27)

Same production H3 clip (ref2v, 8 s, 1344x768, 8 steps, 4 LoRAs, Sage), same stack:

| | 1 Spark (stock ComfyUI) | 2-node split |
|---|---|---|
| cold | 610 s | ~400 s |
| warm | 555 s | ~350 s |

The split is only ~1.6x for 2x the hardware, so **two Sparks rendering separately give ~26% more
clips per hour**; the split wins only latency for a single clip. One Spark held the clip comfortably
(lowest MemAvailable 60-67 GiB, no swap growth). An independent 4-node owner measured the same
ratio on Monarch's benchmark cells (1 -> 2 nodes 1.64x, 2 -> 4 nodes uly4 1.65x).

## Monarch's own benchmark cell, our stack vs stock

"MiniMax H3 guide" (Monarch example workflow, converted with `benchmark/sweep/convert.py`): fl2va
INT8 ConvRot, 1344x768, 124 frames, 20 steps `res_multistep`, `TORCH_FLASH`, no LoRAs, 2 nodes, warm:

| | Warm wall time |
|---|---|
| Monarch's published table (2100 MHz cap, stock stack) | 636.6 s |
| independent 4-Spark owner (uncapped ~2.4 GHz, stock stack) | 640.6 s |
| **this repo's stack (2000 MHz cap)** | **499.4 s** (sampling 464 s, split VAE 20.6 s) |

## Topology note

Ulysses needs an all-to-all between every pair of ranks each layer. A ring of direct 200 GbE links
(1-2, 2-3, 3-4, 4-1) lacks the 1-3 and 2-4 links; routing through a middle node costs CPU
forwarding and most of the gain. A switch is a prerequisite for a fair 4-way test.
