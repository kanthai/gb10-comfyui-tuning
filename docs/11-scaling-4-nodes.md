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

## Topology note

Ulysses needs an all-to-all between every pair of ranks each layer. A ring of direct 200 GbE links
(1-2, 2-3, 3-4, 4-1) lacks the 1-3 and 2-4 links; routing through a middle node costs CPU
forwarding and most of the gain. A switch is a prerequisite for a fair 4-way test.
