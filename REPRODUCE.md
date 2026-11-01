# Reproducing every reported number

The branch tip regenerates every table and figure in the paper. Nothing below
reads a saved result unless it says so; every command re-runs the simulators.

Each cell runs in a private temporary tree with `sim_cfg.py` and `model_cfg.py`
rewritten for that configuration, so a sweep cannot leave the repository
configured for one experiment. Runtimes are for the whole command on a 32-core
machine. All commands use the default pass-through NAND model
(`NAND_PASS_THROUGH=1`): every NAND read also occupies the device DRAM.

## Simulation results

| Paper item | Command | ~time |
|---|---|---|
| Table II, decode throughput, both platforms | `python run_paper_tables.py` | 1 min |
| CPU-only only / accelerator only | `python run_paper_tables.py --cpu-only` (or `--gpu-only`) | 30 s |
| Table III, end-to-end time (prefill + decode) | `python run_paper_tables.py --detail` | 2 min |
| Prefill throughput | `python run_paper_tables.py --prefill` | 1 min |
| Table IV, decode length 16 to 2048 | `python realistic_runs.py decode` | 5 min |
| Sec. V-D, equal-footing baselines (tuned K and reserve, Belady) | `python run_paper_tables.py --equal-footing` | 2 min |
| Fig. 1, reserve curve | `python plot_reserve_curve.py --measure` | 2 min |
| Fig. 4, one decode step, filled vs reserved | `python plot_overlap.py` | 2 min |
| Fig. 5, lookahead depth | `python plot_depth.py --measure` | 2 min |
| Table V, robustness | `python sensitivity.py all` (or one of `hints`, `nandbw`, `nandlat`, `context`, `passthrough`, `streaming`, `gpulink`, `mismatch`, `headroom`) | 12 min |
| Table VI, mechanism ablation | `python ablation_passthrough.py` | 2 min |
| Sec. V-G, per-configuration search against the best single reserve | `python fixed_reserve.py` | 6 min |
| Sec. V-H, KV-cache tier | `python run_paper_tables.py --kv-tier` | 2 min |
| Fig. 7, batch size, B = 1 to 128 | `python run_paper_tables.py --batch-sweep` | 15 min |
| Fig. 8, model scale, 7B to 405B | `python run_paper_tables.py --models` | 6 min |
| Sec. V-K, realistic host | `python realistic_runs.py host` | 5 min |
| Sec. V-L, 50 ShareGPT prompts | `python run_paper_tables.py --sharegpt` | 8 min |
| Sec. VII, a plan served at another context length, and plan headroom | `python sensitivity.py mismatch headroom` | 3 min |
| Sec. VII, re-planning cost under changing context | `python replan_cost.py` | 2 min |
| All correctness invariants (122 checks) | `python verify_results.py` | 25 min |
| Invariants without the batch sweep | `python verify_results.py --quick` | 6 min |
| Timing engine vs an independent model | `python test_pipeline_differential.py` | 20 s |

Figs. 7 and 8 are plotted from the harness output:

```bash
python run_paper_tables.py --batch-sweep > batch.txt
python run_paper_tables.py --models      > models.txt
python parse_sweeps.py batch.txt models.txt   # writes sweep_figs.json
python plot_sweeps.py                         # writes fig_batch.pdf, fig_models.pdf
```

`plot_overlap.py --cached`, `plot_depth.py` (without `--measure`) and
`plot_reserve_curve.py` (without `--measure`) replot from the checked-in JSON.

## Real-hardware results (Sec. V-E)

These need a GPU server with an NVMe SSD; the scripts are in `hardware/`.

| Paper item | Command |
|---|---|
| Fig. 6 benchmark, reserve against throughput | `python hardware/reserve_bench.py` on each machine, writing `hardware/results/<machine>/reserve_bench.csv` |
| Timing-engine prediction of the benchmark | `python hardware/predict.py hardware/results/<machine>/reserve_bench.csv` |
| Fig. 6 | `python hardware/plot_hw.py` |
| FlexGen with an external prefetcher, one point | `hardware/flexgen_point.sh CPU_PCT BUDGET_GB DEPTH OUT` |
| The full FlexGen sweep, with GPU-sharing exclusion rules | `hardware/flexgen_overnight.sh`, then `hardware/flexgen_retry.sh` |
| Plan search for FlexGen before any run | `python hardware/plan_search.py` (scores plans with `io_replay.py`) |

FlexGen itself is run unmodified (FMInference/FlexLLMGen); `flexgen_reserve.py`
adds the prefetcher. The measured logs and CSVs behind the paper are in
`hardware/results/`.

## Switches

| Variable | Effect |
|---|---|
| `NAND_PASS_THROUGH=0` | NAND reads are not charged to the device DRAM (Table V, "NAND bypasses device DRAM") |
| `STREAM_COMPUTE=1` | a sub-layer computes alongside its own reads instead of after them, as a CPU computing on weights in place would (Table V, "Compute overlaps its own reads") |
| `GPU_LINK_BPS` | caps the accelerator's PCIe link, which host DRAM and the device share, e.g. `63e9`; 0 (default) leaves it uncapped; no effect without an accelerator (Table V, "Accelerator link") |
| `TIERSCHED_PLACEMENT_ORDER` | `size-desc` (default), `semantic`, `sequential` or `random` (Sec. V-G) |
| `TIERSCHED_HINT_DROP` | fraction of prefetch hints the device ignores; ignored hints become demand reads (Table V) |
| `TIERSCHED_PLAN_PREFILL` | plan for this prompt length and serve the configured one (Sec. VII) |
| `TIERSCHED_STAGE_FRAC` | share of the reserve that Phase-1 warmup may pin; 0 by default, because staging loses this split |
| `DYN_POLICY` | `belady` (default) or `lru`, CMM-H's default host-agnostic cache (Sec. V-D) |
| `DYN_PREFETCH_DEPTH`, `DYN_KV_TIER` | lookahead and KV tier of the dynamic baseline |

## Ablating the contributions

Placement order was this project's original hypothesis, and it is retired:
`size-desc` is the default. The orders below place the same bytes in the same
tiers; only the sequence in which sub-layers claim fast memory changes.

```bash
TIERSCHED_PLACEMENT_ORDER=size-desc  python run_paper_tables.py   # default
TIERSCHED_PLACEMENT_ORDER=semantic   python run_paper_tables.py   # the retired hypothesis
TIERSCHED_PLACEMENT_ORDER=sequential python run_paper_tables.py   # model order
TIERSCHED_PLACEMENT_ORDER=random     python run_paper_tables.py   # seeded shuffle
```

Measured over the twelve configurations: `size-desc` beats `semantic` in all
twelve, by a median of 2% and at most 11% (accelerated INT8 16H+32C). Every
order, the random one included, beats every baseline in all twelve, so the gain
comes from the reserve and residue spreading rather than from the order.

The reserve search has no flag, because it is the search in `tiersched.py`
itself. Ablate it by pinning the grid to a single point (reserve nothing, KV on
the device, full device capacity):

```python
_fr = (0.0,)                              # the reserve fractions
_kv_opts = ["cxl"]                        # the KV tier choice
_dev_caps = [cxl_dev_dram_capacity_bytes] # the device sizes
```

TierSched then loses 9 of 12 and gives up the entire INT8 margin.
`ablation_passthrough.py` applies this pin and the other Table VI variants, and
`plot_reserve_curve.py` sweeps the reserve one fraction at a time; both assert
that each patch anchor still matches the file.

## What each file is for

| File | |
|---|---|
| `run_paper_tables.py` | the harness: the only sanctioned way to produce a number for the paper |
| `verify_results.py` | checks invariants, not a previous run |
| `pipeline.py` | the timing engine, shared by every policy |
| `tiersched.py` | TierSched: two-stage placement search and bounded decode |
| `{flexgen,lia,cxlaimpod,llmflash}_baseline.py` | one per baseline, each implementing its own paper |
| `dynamic_baseline.py` | dynamic replacement by Belady's rule, or CMM-H's default LRU cache |
| `sensitivity.py`, `realistic_runs.py`, `ablation_passthrough.py`, `fixed_reserve.py`, `replan_cost.py` | the experiments listed above |
| `tiers.py` | device bandwidths and the interface ceilings they are asserted against |
| `RESULTS.md` | the measured tables in prose |

## Configuration

Qwen2.5 72B, 512-token prompt, 256 decode tokens, B=128, on a CMM-H hybrid CXL
device: host DDR5-4800 at 38.4 GB/s, CXL device DRAM at 27 GB/s, CXL NAND at
5.0 GB/s behind PCIe Gen4 x4, the two device tiers sharing a Gen5 x8 host link
at 31.5 GB/s. Platforms are 2x EPYC 9454 (14.4 TFLOPS, no accelerator) and
+RTX 5090 (146.9 TFLOPS, 28 GB for weights). Change any of these in
`run_paper_tables.py`: `GRID`, `CXLS`, `QUANTS`, `BATCH`, `DECODE`, `ENGINES`.

## What this simulator does not model

- The CXL link's Rx/Tx duplexity is modelled, not validated. Tier bandwidths
  and the independence of the tier buses are reproduced at cycle level in gem5
  with SimCXL; the bridge model that would validate duplexity is not run. No
  reported headline number depends on it.
- Baselines implement their published memory policies, not their whole systems:
  CXLAimPod's eBPF duplex co-scheduler and LIA's AMX compute offload are out of
  scope. FlexGen's weights are placed fastest tier first, the optimum of its
  linear program when capacity binds, and its schedule is timed on the shared
  engine.
- Decoder blocks are assumed uniform, and sparsity comes from published
  profiles rather than per-model measurement.
- NAND latency is charged serially per 256 KB read, with no SSD queue
  parallelism; `sensitivity.py nandlat` varies it from 0.5 to 100 us.
- A sub-layer computes once all its bytes have landed, as on a GPU that copies
  weights into its own memory. A CPU computing on weights in place overlaps the
  two; `sensitivity.py streaming` runs that variant, which lowers the CPU-only
  gains (to 1.01-1.27x) and keeps all twelve leads.
- The accelerator's PCIe link is uncapped by default. `sensitivity.py gpulink`
  caps it at 63 or 55 GB/s; that slows the baselines more than TierSched and
  keeps all twelve leads.
- No physical CMM-H device. The CMM-H numbers are trace-driven simulation
  against documented device specifications; the mechanism and the timing
  engine are checked on real DRAM-over-NVMe servers (`hardware/`).
