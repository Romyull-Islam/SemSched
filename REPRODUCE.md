# Reproducing every reported number

Tag `v2026-08-14-ledger` is the state the results tables were measured on.
Nothing below reads a saved file; every command re-runs the simulators.

```bash
git checkout v2026-08-14-ledger      # exact state of the reported tables
git checkout revision-bigdata2026    # back to the branch tip
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
| **All correctness invariants** | `python verify_results.py` | 25 min |
| Invariants without the batch sweep | `python verify_results.py --quick` | 6 min |

## Ablating the contributions

The placement order is the paper's named contribution. `semantic` is the
default and the shipped behaviour; the others are nulls with the same tiers,
the same capacities and the same bytes — only the sequence in which sub-layers
claim fast memory changes.

```bash
SEMSCHED_PLACEMENT_ORDER=semantic   python run_paper_tables.py   # default
SEMSCHED_PLACEMENT_ORDER=sequential python run_paper_tables.py   # model order
SEMSCHED_PLACEMENT_ORDER=size-desc  python run_paper_tables.py   # largest first
SEMSCHED_PLACEMENT_ORDER=random     python run_paper_tables.py   # seeded shuffle
```

Measured: semantic vs sequential is 0.983x-1.108x (median 0.995x), and
`size-desc` beats `semantic` in 11 of 12 cells.

The staging reservation is what actually produces the result. There is no flag
for it -- it is the search at `semduplex_scheduler.py` -- so ablate it by
pinning the grid to a single point:

```python
_fr = (0.0,)                # was (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
_kv_opts = ["cxl"]          # was ["cxl", "host"] + (["gpu"] if ... )
```

Measured: SemSched then loses 11 of 12, at 0.85x-1.00x.

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

## Open, as of this tag

- The staging reservation has no baseline equivalent, and FlexGen's paper
  specifies an LP policy search that this tree replaced with a fixed cascade.
  Until that is symmetric the 12/12 is not defensible.
- Four baseline fidelity defects still favour SemSched (LIA's prefetch depth,
  LLM-in-a-Flash's rewrite penalty, CXLAimPod's missing accelerator tier,
  FlexGen's LP). Being repaired on the branch after this tag.
- `SemSched.tex` carries numbers from before all of this and contradicts the
  tree in 28 places.
