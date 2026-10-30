# TierSched

**TierSched: Trading Memory Capacity for Prefetch Bandwidth in LLM Inference on Hybrid CXL Devices**

This repository contains the simulator, the four baselines, the experiment
harness, and the plotting code that produce every number and figure in the
TierSched paper. One harness produces all of them; nothing is read from a
saved result unless a command below says so.

TierSched targets **CMM-H**-style hybrid CXL devices that pair an on-card DRAM
cache with NAND flash behind a single host link. Fast memory on such a device
has two uses: holding weights, and staging prefetched bytes before they are
needed. The offloading policies evaluated here spend all of it on the first.
TierSched searches the split per configuration, how much host and device DRAM
to hold back as prefetch staging, jointly with the tier that holds the KV cache
and the device capacity to use. It spreads NAND-resident sub-layers across the
execution order and decodes with a lookahead bounded by the reserve. Finalist
plans are timed exactly on the same engine that runs decode, so the plan the
search evaluates is the plan decode executes.

## Version

This artifact contains only the components the paper stands on. Mechanisms
the paper measures and retires (semantic sub-layer placement, prefill staging,
and duplex write scheduling as a throughput mechanism) ship as measured nulls
inside the harness, not as systems. The working tree is exactly the layout
below and nothing else.

## Repository layout

```
.
├── tiersched.py                 # TierSched: two-stage placement search + decode engine
├── flexgen_baseline.py          # FlexGen: linear program over per-tier percentages
├── lia_baseline.py              # LIA: parameters in device DRAM, host reserved for KV
├── cxlaimpod_baseline.py        # CXLAimPod: pools device memory in capacity order
├── llmflash_baseline.py         # LLM-in-a-Flash: sliding window of active neurons
├── dynamic_baseline.py          # Belady's rule over device DRAM, or CMM-H's default LRU
│
├── sim_cfg.py                   # Hardware parameters (bandwidths, latencies, capacities)
├── model_cfg.py                 # Model definitions, 7B to 405B, FP16 and INT8
├── tiers.py                     # Memory tiers and transfer-time helpers
├── cxl_link.py                  # Two-queue CXL link model
├── pipeline.py                  # Shared overlap/timing engine used by all five simulators
│
├── run_paper_tables.py          # The harness: every table and sweep in the paper
├── verify_results.py            # 122 invariant checks over the full result grid
├── sensitivity.py               # Table V: hints ignored, NAND rate and latency, long prompts
├── realistic_runs.py            # Realistic host and decode-length experiments
├── ablation_passthrough.py      # Table VI: mechanism ablation
├── fixed_reserve.py             # Per-configuration search against one fixed reserve
├── replan_cost.py               # Cost of re-planning when the context length changes
├── parse_sweeps.py              # Harness output -> sweep_figs.json
│
├── plot_reserve_curve.py        # Motivation figure: the reserve trade, measured
├── plot_evaluation.py           # Evaluation panels: capacity, write stall, ShareGPT
├── plot_sweeps.py               # Batch-size and model-scale figures
├── plot_overlap.py              # One decode step in time: filled vs reserved
├── plot_depth.py                # Lookahead depth against throughput, every policy
├── evaluation_data.json         # Measured data behind the evaluation panels
├── reserve_curve.json           # Measured data behind the motivation figure
├── sweep_figs.json              # Measured data behind the sweep figures
├── overlap_data.json            # Measured schedules behind the timeline figure
├── depth_sweep.json             # Measured data behind the lookahead figure
├── sensitivity.json             # Measured data behind Table V
│
├── test_pipeline_differential.py # Timing engine vs an independent model + bounds
├── test_cxl_link.py             # Unit tests for the link model (23 checks)
├── test_bandwidth_conservation.py  # Bandwidth accounting tests (pytest)
│
├── trace_workload/
│   ├── sharegpt_lens.json       # (prefill, decode) length pairs from ShareGPT V3
│   └── download_sharegpt.py     # regenerates the pairs from the public dataset
│
├── hardware/                    # real-server benchmark, FlexGen runs, measured results
│
├── validation/                  # cycle-level check of the timing model on gem5/SimCXL
│   ├── configs/cmmh_hybrid.py   #   the CMM-H hybrid device configuration
│   └── sweep.py                 #   drives the same (host, device) grid
│
├── figures/                     # the generated figures, overwritten by the plot scripts
├── REPRODUCE.md                 # command-to-number map for every reported cell
├── RESULTS.md                   # the measured result set, with the commands that made it
├── CHANGELOG.md                 # what changed between versions
└── LICENSE                      # MIT
```

## Requirements

- Python 3.10+. The simulators use only the standard library.
- `matplotlib` for the plotting scripts.
- Optional: `pytest` for `test_bandwidth_conservation.py`;
  `datasets` and `tiktoken` only to re-download ShareGPT.
- `hardware/` only: a CUDA GPU with PyTorch, an NVMe SSD, and FlexGen
  (FMInference/FlexLLMGen) for the real-engine runs.

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

`REPRODUCE.md` maps every table, figure and section of the paper to the
command that produces it, including the robustness, ablation, equal-footing
and real-hardware experiments.

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
reproduced at cycle level in gem5 with SimCXL's CXL device models. The
configuration and sweep driver live in `validation/` and share no code with
the simulators here; `validation/README.md` documents the calibration, the
three validated properties, and how to obtain and build the upstream tools.

## Citation

If you use this code, please cite the paper:

```
TierSched: Trading Memory Capacity for Prefetch Bandwidth in LLM Inference
on Hybrid CXL Devices. Under review.
```
