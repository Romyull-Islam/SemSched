# SemSched

**SemSched: Trading Memory Capacity for Prefetch Bandwidth in LLM Inference on Hybrid CXL Devices**

This repository contains the simulator, the four baselines, the experiment
harness, and the plotting code that produce every number and figure in the
SemSched paper. One harness produces all of them; nothing is read from a
saved result unless a command below says so.

SemSched targets **CMM-H**-style hybrid CXL devices that pair an on-card DRAM
cache with NAND flash behind a single host link. Fast memory on such a device
has two uses: holding weights, and staging prefetched bytes before they are
needed. Every published policy spends all of it on the first. SemSched
searches the split per configuration, how much host and device DRAM to hold
back as prefetch staging, jointly with the tier that holds the KV cache and
with the device capacity itself, which the search may decline so that bytes
deliberately left on NAND ride an otherwise idle bus. Finalist plans are
timed exactly on the same engine that runs decode, so the plan validated is
the plan realized.

## Version

This is **v2.0**, the artifact of the IEEE BigData 2026 submission. It
contains only the components the revised paper stands on. Mechanisms the
paper measures and retires: semantic sub-layer placement (0.96x against
ordering by size), prefill staging, duplex write scheduling as a throughput
mechanism, ship as the measured nulls inside the harness, not as systems.
The working tree is exactly the layout below and nothing else. The full
pre-revision history, including the retired components, remains in this
repository's git history; tag `mascots2026-submitted` is the v1 state.

## Repository layout

```
.
├── semduplex_scheduler.py       # SemSched: two-stage placement search + decode engine
├── flexgen_baseline.py          # FlexGen: linear program over per-tier percentages
├── lia_baseline.py              # LIA: parameters in device DRAM, host reserved for KV
├── cxlaimpod_baseline.py        # CXLAimPod: pools device memory in capacity order
├── llmflash_baseline.py         # LLM-in-a-Flash: sliding window of active neurons
│
├── sim_cfg.py                   # Hardware parameters (bandwidths, latencies, capacities)
├── model_cfg.py                 # Model definitions, 7B to 405B, FP16 and INT8
├── tiers.py                     # Memory tiers and transfer-time helpers
├── cxl_link.py                  # Two-queue CXL link model
├── pipeline.py                  # Shared overlap/timing engine used by all five simulators
│
├── run_paper_tables.py          # The harness: every table and sweep in the paper
├── verify_results.py            # 122 invariant checks over the full result grid
│
├── plot_reserve_curve.py        # Motivation figure: the reserve trade, measured
├── plot_evaluation.py           # Evaluation panels: capacity, write stall, ShareGPT
├── plot_sweeps.py               # Batch-size and model-scale figures
├── evaluation_data.json         # Measured data behind the evaluation panels
├── reserve_curve.json           # Measured data behind the motivation figure
├── sweep_figs.json              # Measured data behind the sweep figures
│
├── test_pipeline_differential.py # Timing engine vs an independent model + bounds
├── test_cxl_link.py             # Unit tests for the link model (23 checks)
├── test_bandwidth_conservation.py  # Bandwidth accounting tests (pytest)
│
├── trace_workload/
│   ├── sharegpt_lens.json       # (prefill, decode) length pairs from ShareGPT V3
│   └── download_sharegpt.py     # regenerates the pairs from the public dataset
│
├── figures/                     # the generated figures, overwritten by the plot scripts
├── REPRODUCE.md                 # command-to-number map for every reported cell
├── RESULTS.md                   # the measured result set, with the commands that made it
├── CHANGELOG.md                 # what changed between v1.0 and v2.0
└── LICENSE                      # MIT
```

## Requirements

- Python 3.10+. The simulators use only the standard library.
- `matplotlib` for the plotting scripts.
- Optional: `pytest` for `test_bandwidth_conservation.py`;
  `datasets` and `tiktoken` only to re-download ShareGPT.

## Regenerating every number

Each cell runs in a private temporary tree with the configuration rewritten
for that run, so no sweep can contaminate another or the repository.

```bash
python run_paper_tables.py               # decode throughput + wall clock, both platforms
python run_paper_tables.py --prefill     # prefill tables
python run_paper_tables.py --kv-tier     # KV-tier placement table
python run_paper_tables.py --ablation    # search-disabled ablation
python run_paper_tables.py --batch-sweep # B = 1..128 grid behind the batch figure
python run_paper_tables.py --models      # 7B..405B grid behind the model-scale figure
python run_paper_tables.py --sharegpt    # 50 real ShareGPT prompts, both platforms

python verify_results.py                 # 122 invariant checks; exits nonzero on failure
```

## Regenerating the figures

```bash
python plot_reserve_curve.py --measure   # motivation figure (re-runs the sweep)
python plot_evaluation.py                # evaluation panels (measures, then plots)
python plot_sweeps.py                    # batch and model figures from sweep_figs.json
```

Without `--measure`, `plot_reserve_curve.py` and `plot_evaluation.py --cached`
replot from the checked-in JSON.

The paper itself is not in this repository. It will be linked here once it is
published; this repository is the artifact that produces its numbers.

## Cycle-level validation

The tier bandwidths and the timing engine's bus-independence assumption are
reproduced at cycle level in gem5 with SimCXL's CXL device models, in the
companion repository **SemSched-CXLSim**, which shares no code with this
simulator. Its README documents the calibration and the three validated
properties.

## Citation

If you use this code, please cite the paper:

```
SemSched: Trading Memory Capacity for Prefetch Bandwidth in LLM Inference
on Hybrid CXL Devices. Md Romyull Islam, Tu N. Nguyen, Selena He, Yong Shi,
and Kun Suo. IEEE BigData 2026 (under submission).
```
