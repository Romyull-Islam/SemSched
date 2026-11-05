# Changelog

## Unreleased (2026-11-05)

Every reported number changes; `REPRODUCE.md` regenerates all of them.

**Timing engine**
- Buses take each request in the earliest idle stretch at or after it can
  start (`BUS_ORDER=fill`), so a cache hit no longer waits behind a NAND fill
  queued before it. The old strict issue order is `BUS_ORDER=fifo`. Under the
  old order, splitting the same bytes into smaller transfers could make a step
  up to 33% slower.
- Units can move in blocks (`blocks=`, `PIPELINE_BLOCKS`), each issued when its
  unit would be and computed as soon as it lands.
- The engine can record its schedule (`trace=`); Fig. 4 is drawn from it.
- The differential test checks the new order against a reference written
  independently from its specification, and the old order against the old one.

**TierSched**
- Stripes every sub-layer, and optionally the KV cache, across the tiers in
  the proportions of a capacity fill; moves each sub-layer in eight blocks.
- Searches the layout (contiguous, contiguous with the NAND residue spread,
  striped), the KV placement (one tier or striped) and its reservation (the
  generation mean or the peak), and a reserve given as a share of host plus
  device DRAM split three ways. Finalists are timed at nine steps.

**Baselines**
- FlexGen applies its weight fractions to every layer, as its code does.
- ITME added.
- Every baseline reserves its KV cache at the generation mean or at its peak,
  whichever is faster. Growth past the reservation sits on the cache's last
  tier, in free room first, for every policy.
- The dynamic baseline also runs with unbounded lookahead (Cao et al.'s
  Aggressive); tuned baselines may use block transfers.

**Analysis**
- `bound.py`: a lower bound on decode step time that holds for any policy.
  Table II reports each policy against it, and `verify_results.py` checks that
  no policy beats it.

## v2.0-bigdata2026 (2026-08-17)

Artifact of the IEEE BigData 2026 submission, *Trading Memory Capacity for
Prefetch Bandwidth in LLM Inference on Hybrid CXL Devices*.

**What the system claims changed.** v1 claimed that decomposing decoder blocks
into attention and MLP sub-layers and ordering them semantically was the
mechanism. v2 measures that claim at 0.96x against ordering the same
sub-layers by size and retires it. The mechanism v2 defends instead: fast
memory has two uses, holding weights and staging prefetched bytes, and the
split between them is searched per configuration, jointly with the tier that
holds the KV cache and with device capacity itself, which the search may
decline so that bytes deliberately left on NAND ride an otherwise idle bus.

**Added**
- CXLAimPod as a fourth baseline; all four now run under uniform byte accounting.
- Two-stage placement search: estimate over the candidate grid, then exact
  timing of per-group finalists on the same engine that runs decode, so the
  plan validated is the plan realized.
- `verify_results.py`: 122 invariant checks over the full result grid.
- Cycle-level validation of tier bandwidths and the bus-independence assumption
  in gem5 with SimCXL; configuration and driver under validation/.
- ShareGPT trace evaluation on 50 real prompt/response length pairs.
- Model-scale coverage from 7B to 405B parameters.

**Retired, and reported as measured nulls**
- Semantic sub-layer ordering: 0.96x against ordering by size, losing in all twelve.
- Prefill staging: loses to overlap in all twelve configurations.
- Duplex write scheduling: moves latency, not throughput.

**Repository**
- Tree reduced to the 35 files the paper stands on. The pre-revision working
  set remains in history; `mascots2026-submitted` is the v1 state.
- Every figure regenerates from measured JSON produced by the same harness that
  produces the tables, so a figure cannot drift from a table.

## v1.0 (tag `mascots2026-submitted`, 2026-06-03)

*Semantic Sub-Layer-Aware Scheduling for Hybrid CXL-Based LLM Inference*, the
MASCOTS 2026 submission. 82 tracked files. Rejected; superseded by v2.0.
