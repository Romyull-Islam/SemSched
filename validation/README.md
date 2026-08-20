# Cycle-level validation on SimCXL

**This directory shares no code with the simulators above it.** Nothing here
is imported by the paper's simulator and nothing here imports from it. It
exists to test the claims the analytical model cannot check itself.

## Why

The main artifact is our own trace-driven simulator. Its compute term is
calibrated against LLMCompass (ISCA 2024) and its DRAM bandwidths against
DRAMsim3, but three things remain unvalidated by anything external:

1. **The full-duplex CXL link model**: that KV writes injected on the idle Tx
   lane are absorbed under concurrent Rx reads rather than adding to latency.
   This is the paper's core mechanism and nothing checks it.
2. **The DRAM-cache-over-NAND hybrid**: CMM-H's device-side caching, which we
   model as a hit/miss bandwidth step (27 GB/s → 5 GB/s) rather than as a cache.
3. **The placement policy under a real memory controller**, with queueing,
   row-buffer conflicts, and refresh, none of which our tier model has.

## What is in this directory

The CMM-H hybrid device configuration (`configs/cmmh_hybrid.py`), the sweep
driver (`sweep.py`), and this README. SimCXL and gem5 are **not** redistributed
here; they are upstream projects with their own licences, and this directory
holds only the configuration and driver written for this validation.

### Setup

```bash
git clone https://github.com/TianheMICALab/SimCXL.git
cd SimCXL && scons build/X86/gem5.opt -j$(nproc)     # see SimCXL's own build docs
cd .. && cp -r configs sweep.py SimCXL/              # drop this repo's files in
```

Every command below assumes that layout, with `SimCXL/build/X86/gem5.opt` built.

## Tool

[SimCXL](https://github.com/TianheMICALab/SimCXL): gem5-based, full-system,
cycle-level, modelling CXL.io/.cache/.mem and Type 1/2/3 devices. Backed by
CXL-DMSim (TCAD 2025, silicon-validated) and Cohet (HPCA 2026).

Chosen over [CXLMemSim](https://github.com/SlugLab/CXLMemSim), which samples a
real running binary via PMU counters and injects emulated CXL latency. That
answers "how would my existing app behave on CXL"; we have no such binary, and
it does not model link contention.

The property that makes SimCXL usable here: `src/mem/cxl_bridge.hh` defines
separate `BridgeRequestPort` and `BridgeResponsePort`, each with an independent
deferred-packet queue and delay. That is structurally the two-direction link our
duplex model asserts, so the claim can be tested rather than assumed.

## Plan

| Step | What | Status |
|---|---|---|
| 1 | Build gem5 with SimCXL's CXL device models | done |
| 2 | Run the stock Type 3 expander config, confirm baseline | done |
| 3 | Extend Type 3 into a **CMM-H hybrid**: DDR cache tier + 1 TB NAND tier | done (SimpleMemory tiers) |
| 4 | Calibrate: measured 27.05 / 5.00 / 38.47 GB/s vs targets 27 / 5 / 38.4 | **done, <0.2% each** |
| 5 | Configure our hierarchy: 16 GB host DDR5, 48 GB CXL DRAM cache, 1 TB NAND | done |
| 6 | **Bus-independence test** (`--mode concurrent`): cxl + host driven simultaneously hold 27.02 and 38.47 GB/s, each within 0.2% of solo | **done; validates the paper's core timing assumption** |
| 7 | **Within-tier duplex** (`--mode duplex`): 50/50 read-write on device DRAM sums to the tier ceiling; reads yield to writes | done; matches the engine's within-tier serialization |
| 8 | Link-level Rx/Tx duplexity via SimCXL's `cxl_bridge` | open; the one modelled-not-validated claim, stated in the paper |

Repro, from the SimCXL tree with this repository's files copied in:
`build/X86/gem5.opt --outdir=out configs/cmmh_hybrid.py --mode {stream,concurrent,duplex} --target {host,cxl,nand} --duration 2e6`

## Parameterised, not fixed

No tier size is hard-coded. The main artifact's result is a **surface** over
(host DRAM, CXL DRAM), not a point: the advantage peaks where NAND residency
divided by cache capacity approaches unity, and inverts at both extremes. A
validation that pinned one configuration would test the least interesting part
of the claim. `sweep.py` drives the same grid the main artifact uses.

```bash
./sweep.py --hosts 8 16 32 64 --cxls 32 48 64 96     # the full surface
./sweep.py --hosts 16 --cxls 48 --gpu 28             # the headline point
./sweep.py --hosts 16 --cxls 48 --duplex             # read/write concurrency
```

Host DRAM, CXL DRAM cache, NAND backend and accelerator memory are all
arguments; each cell writes to `results/<host>H_<cxl>C[_<gpu>G]/` and the grid
summary carries `overflow_over_cache` so the surface can be checked against the
analytical model directly.

## Configuration under test

The default cell, matching the main artifact's headline:

| Tier | Capacity | Bandwidth | Latency |
|---|---|---|---|
| Host DRAM | 16 GB DDR5-4800 | 38.4 GB/s | 200 ns |
| CXL device DRAM cache | 48 GB DDR4-2666 | 27 GB/s | 505 ns |
| CXL NAND backend | 1 TB | 5 GB/s | 1547 ns |

Model: Qwen2.5 72B FP16, 145 GB, so 81 GB is NAND-resident and must be staged.

## What this can and cannot settle

It can test the **mechanism**: whether duplex injection works, whether staging
into a device-side DRAM cache behaves as modeled, and whether a real memory
controller changes the picture.

It cannot re-run the **comparison**. The baselines are policies, not workloads;
reproducing four papers inside gem5 is out of scope. Ratios remain a property of
the main artifact.
