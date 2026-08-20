# Reproducing every reported number

Tag `v2.0-bigdata2026` is the submitted state, and the tables regenerate from
it. Nothing below reads a saved file; every command re-runs the simulators.

```bash
git checkout v2.0-bigdata2026        # exact state of the reported tables
git checkout main                    # back to the branch tip
```

Each cell runs in a private temporary tree with `sim_cfg.py` and `model_cfg.py`
rewritten for that configuration, so a sweep cannot leave the repository
configured for one experiment. Runtimes are for the whole command.

| What | Command | ~time |
|---|---|---|
| Decode throughput, both platforms | `python run_paper_tables.py` | 1 min |
| CPU-only only / accelerator only | `python run_paper_tables.py --cpu-only` (or `--gpu-only`) | 30 s |
| Prefill throughput | `python run_paper_tables.py --prefill` | 1 min |
| **Warmup / prefill / decode / wall clock** | `python run_paper_tables.py --detail` | 2 min |
| Batch sweep, B = 1…128, all 96 cells | `python run_paper_tables.py --batch-sweep` | 15 min |
| Which tier should hold the KV cache | `python run_paper_tables.py --kv-tier` | 2 min |
| Phase-1 warmup ablation | `python run_paper_tables.py --ablation` | 1 min |
| Model scale, 7B to 405B | `python run_paper_tables.py --models` | 6 min |
| 50 real ShareGPT prompts | `python run_paper_tables.py --sharegpt` | 8 min |
| Decode-step timeline, filled vs reserved | `python plot_overlap.py` | 2 min |
| **Timing engine vs an independent model** | `python test_pipeline_differential.py` | 20 s |
| **All correctness invariants** | `python verify_results.py` | 25 min |
| Invariants without the batch sweep | `python verify_results.py --quick` | 6 min |

## Ablating the contributions

Placement order was this project's original hypothesis, and it is retired:
`size-desc` is the default and the shipped behaviour. The orders below are
nulls with the same tiers, the same capacities and the same bytes — only the
sequence in which sub-layers claim fast memory changes.

```bash
SEMSCHED_PLACEMENT_ORDER=size-desc  python run_paper_tables.py   # default, shipped
SEMSCHED_PLACEMENT_ORDER=semantic   python run_paper_tables.py   # the retired hypothesis
SEMSCHED_PLACEMENT_ORDER=sequential python run_paper_tables.py   # model order
SEMSCHED_PLACEMENT_ORDER=random     python run_paper_tables.py   # seeded shuffle
```

Measured over the twelve reported configurations: `size-desc` beats `semantic`
in all twelve, median 0.96x, worst 0.86x (accelerated INT8 16H+32C). Against
`sequential` the semantic order is near-neutral, median 0.997x over the same
grid. `size-desc` ships because it is the only ordering that stays above every
baseline in all twelve cells; `semantic` and `sequential` each fall below the
strongest baseline in two accelerated cells.

The joint capacity search is what actually produces the result. There is no
flag for it -- it is the search in `semduplex_scheduler.py` -- so ablate it by
pinning the grid to a single point (reserve nothing, KV on the device, full
device capacity):

```python
_fr = (0.0,)                              # the reserve fractions
_kv_opts = ["cxl"]                        # the KV tier choice
_dev_caps = [cxl_dev_dram_capacity_bytes] # declining capacity disabled
```

Measured: SemSched then loses 9 of 12 and surrenders the entire INT8 margin.
`plot_reserve_curve.py` automates exactly this pin, one reserve fraction at a
time, and asserts each patch anchor still matches the file.

## What each file is for

| File | |
|---|---|
| `run_paper_tables.py` | the only sanctioned way to produce a number for the paper |
| `verify_results.py` | checks invariants, not a previous run -- every defect this project hit reproduced perfectly |
| `pipeline.py` | the timing engine, shared by all five policies |
| `semduplex_scheduler.py` | SemSched |
| `{flexgen,lia,cxlaimpod,llmflash}_baseline.py` | one per baseline, each implementing its own paper |
| `tiers.py` | device bandwidths and the interface ceilings they are asserted against |
| `RESULTS.md` | the measured tables in prose |

## Configuration

Qwen2.5 72B, 512-token prompt, 16 decode steps, B=128, on a CMM-H hybrid CXL
device: host DDR5-4800 at 38.4 GB/s, CXL device DRAM at 27 GB/s, CXL NAND at
5.0 GB/s behind PCIe Gen4 x4, the two device tiers sharing a Gen5 x8 host link
at 31.5 GB/s. Platforms are 2x EPYC 9454 (14.4 TFLOPS, no accelerator) and
+RTX 5090 (146.9 TFLOPS, 28 GB for weights). Change any of these in
`run_paper_tables.py` -- `GRID`, `CXLS`, `QUANTS`, `BATCH`, `DECODE`, `ENGINES`.

## What this simulator does not model

- The CXL link's Rx/Tx duplexity is modelled, not validated. Tier bandwidths
  and the independence of the tier buses are reproduced at cycle level in gem5
  with SimCXL; the bridge model that would validate duplexity is not run. No
  reported headline number depends on it.
- Baselines implement their published memory policies, not their whole systems:
  CXLAimPod's eBPF duplex co-scheduler and LIA's AMX compute offload are out of
  scope. FlexGen's linear program optimizes under its own serial cost model, so
  its rows are a lower bound on what its policy could achieve here.
- Decoder blocks are assumed uniform, and sparsity comes from published
  profiles rather than per-model measurement.
- No physical CMM-H hardware. Every number is trace-driven simulation against
  documented device specifications.
