# Changelog

## v2.0-bigdata2026 — 2026-08-17

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

## v1.0 — tag `mascots2026-submitted`, 2026-06-03

*Semantic Sub-Layer-Aware Scheduling for Hybrid CXL-Based LLM Inference*, the
MASCOTS 2026 submission. 82 tracked files. Rejected; superseded by v2.0.
