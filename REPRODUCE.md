# Reproducing every reported number

The branch tip regenerates every table and figure in the paper. Nothing below
reads a saved result unless it says so; every command re-runs the simulators.

Each cell runs in a private temporary tree with `sim_cfg.py` and `model_cfg.py`
rewritten for that configuration, so a sweep cannot leave the repository
configured for one experiment. Runtimes are for the whole command on a 32-core
machine with `OPENBLAS_NUM_THREADS=1`: every cell is its own process, and
without it each one starts a full set of BLAS threads. All commands use the default pass-through NAND model
(`NAND_PASS_THROUGH=1`): every NAND read also occupies the device DRAM.

## Simulation results

| Paper item | Command | ~time |
|---|---|---|
| Table II, decode throughput and the bound, both platforms | `python run_paper_tables.py` | 8 min |
| Sec. V-B, lower bound on decode for any policy, one configured tree | `python bound.py` (the harness runs it per cell) | 1 s |
| CPU-only only / accelerator only | `python run_paper_tables.py --cpu-only` (or `--gpu-only`) | 30 s |
| Table II, end-to-end column (prefill + decode) | `python run_paper_tables.py --detail` | 2 min |
| Prefill throughput | `python run_paper_tables.py --prefill` | 1 min |
| Sec. V-A, decode length 16 to 2048 | `python realistic_runs.py decode` | 2 min |
| Sec. V-C, equal-footing baselines (tuned K, blocks and reserve; Belady up to unbounded lookahead); writes `equal_footing.json` | `python run_paper_tables.py --equal-footing` | 50 min |
| Sec. V-C, where the lead over the tuned baselines comes from (needs `equal_footing.json`) | `python attribution.py` | 1 min |
| Fig. 1, reserve curve | `python plot_reserve_curve.py --measure` | 2 min |
| Fig. 4, one decode step, filled vs reserved | `python plot_overlap.py` | 2 min |
| Sec. V-C, throughput against lookahead depth (no longer plotted in the paper) | `python plot_depth.py --measure` | 4 min |
| Table III, robustness | `python sensitivity.py all` (or one of `hints`, `nandbw`, `nandlat`, `context`, `passthrough`, `streaming`, `gpulink`, `device`, `staging`, `busorder`, `hostlink`, `hostbw`, `mismatch`, `headroom`) | 20 min |
| Table IV, mechanism ablation | `python ablation_passthrough.py` | 2 min |
| Sec. V-F, per-configuration search against the best single reserve | `python fixed_reserve.py` | 11 min |
| Sec. V-F, KV-cache tier | `python run_paper_tables.py --kv-tier` | 2 min |
| Sec. V-G, batch size, B = 1 to 128 (`fig_batch.pdf`, not in the paper) | `python run_paper_tables.py --batch-sweep` | 5 min |
| Fig. 6, model scale, 7B to 405B | `python run_paper_tables.py --models` | 8 min |
| Sec. V-H, realistic host | `python realistic_runs.py host` | 5 min |
| Sec. V-H, 50 ShareGPT prompts | `python run_paper_tables.py --sharegpt` | 8 min |
| Sec. VII, a plan served at another context length, and plan headroom | `python sensitivity.py mismatch headroom` | 3 min |
| Sec. VII, re-planning cost under changing context | `python replan_cost.py` | 2 min |
| All correctness invariants (211 checks, including the bound) | `python verify_results.py` | 30 min |
| Invariants without the batch sweep | `python verify_results.py --quick` | 6 min |
| Timing engine vs an independent model | `python test_pipeline_differential.py` | 20 s |

Fig. 6 and `fig_batch.pdf` are plotted from the harness output:

```bash
python run_paper_tables.py --batch-sweep > batch.txt
python run_paper_tables.py --models      > models.txt
python parse_sweeps.py batch.txt models.txt   # writes sweep_figs.json
python plot_sweeps.py                         # writes fig_batch.pdf, fig_models.pdf
```

`plot_overlap.py --cached`, `plot_depth.py` (without `--measure`) and
`plot_reserve_curve.py` (without `--measure`) replot from the checked-in JSON.

## Real-hardware results (Sec. V-D)

These need a GPU server with an NVMe SSD; the scripts are in `hardware/`.

| Paper item | Command |
|---|---|
| Fig. 5 benchmark, reserve against throughput | `python hardware/reserve_bench.py` on each machine, writing `hardware/results/<machine>/reserve_bench.csv` |
| Timing-engine prediction of the benchmark | `python hardware/predict.py hardware/results/<machine>/reserve_bench.csv` |
| Fig. 5 | `python hardware/plot_hw.py` |
| FlexGen with an external prefetcher, one point | `hardware/flexgen_point.sh CPU_PCT BUDGET_GB DEPTH OUT` |
| The full FlexGen sweep, with GPU-sharing exclusion rules | `hardware/flexgen_overnight.sh`, then `hardware/flexgen_retry.sh` |
| Plan search for FlexGen before any run | `python hardware/plan_search.py` (scores plans with `io_replay.py`) |

FlexGen itself is run unmodified (FMInference/FlexLLMGen); `flexgen_reserve.py`
adds the prefetcher. The measured logs and CSVs behind the paper are in
`hardware/results/`.

## Switches

| Variable | Effect |
|---|---|
| `NAND_PASS_THROUGH=0` or `2` | NAND bytes cross the device DRAM zero times or twice instead of once (Table III, "NAND crosses device DRAM 0/2 times") |
| `STAGE_NAND_ONLY=1` | only NAND bytes count against the in-flight budget, instead of every byte read ahead (Table III, "Only NAND bytes need staging room") |
| `DEVICE_SERIAL=1` | the device serves one request at a time instead of serving its DRAM while it reads NAND (Table III, "Device serves one request at a time") |
| `STREAM_COMPUTE=1` | a sub-layer computes alongside its own reads instead of after them, as a CPU computing on weights in place would (Table III, "Compute overlaps its own reads") |
| `GPU_LINK_BPS` | caps the accelerator's PCIe link, which host DRAM and the device share, e.g. `63e9`; 0 (default) leaves it uncapped; no effect without an accelerator (Table III, "Accelerator link") |
| `BUS_ORDER=fifo` | every bus serves requests strictly in issue order, so a cache hit waits behind a NAND fill queued before it; the default `fill` books each request into the earliest idle stretch of its bus (Table III) |
| `PIPELINE_BLOCKS` | transfer blocks per unit for policies that do not choose their own; 1 by default, 8 for the tuned baselines' block option |
| `TIERSCHED_BLOCKS` | TierSched's transfer blocks per sub-layer (default 8; 1 for the ablation) |
| `TIERSCHED_PLACEMENT_ORDER` | `size-desc` (default), `semantic`, `sequential` or `random` (Sec. V-F) |
| `TIERSCHED_HINT_DROP` | fraction of prefetch hints the device ignores; ignored hints become demand reads (Table III) |
| `TIERSCHED_PLAN_PREFILL` | plan for this prompt length and serve the configured one (Sec. VII) |
| `TIERSCHED_PLAN_DEVICE=default` | plan for the default device while decode runs on the one the switches above describe (Sec. V-E) |
| `TIERSCHED_STAGE_FRAC` | share of the reserve that Phase-1 warmup may pin; 0 by default, because staging loses this split |
| `DYN_POLICY` | `belady` (default) or `lru`, CMM-H's default host-agnostic cache (Sec. V-C) |
| `DYN_PREFETCH_DEPTH`, `DYN_KV_TIER` | lookahead and KV tier of the dynamic baseline; a depth of 4096 is unbounded lookahead, Cao et al.'s Aggressive |
| `ITME_BUFFER_LAYERS`, `ITME_HOST_STAGING_FRAC` | ITME's ring depth in device DRAM (3) and the share of host DRAM its staging buffers hold (30/256) |

## Ablating the contributions

`ablation_passthrough.py` runs every Table IV variant. Each is a pin of the
search in `tiersched.py`, asserted against its anchor, or a switch:

```python
_rf = (0.0,)                                   # no reserve (tiers filled)
_kv_opts = ["cxl"]                             # KV on the device only
_dev_caps = [cxl_dev_dram_capacity_bytes]      # full device size
_modes = ("spread", "contig")                  # no striping
_splits = ("even",)                            # reserve split in proportion only
PREFETCH_QUEUE_DEPTH  = 0                      # demand paging
```

and `TIERSCHED_BLOCKS=1` moves each sub-layer whole. Measured over the twelve
configurations against the full scheduler: filling the tiers (no reserve) is
up to 33% slower (median 7.5%) and loses 9 of 12 to the best baseline, with
K=32 and K=0 identical over the filled device; no striping costs up to 11%;
whole sub-layers instead of eight blocks cost up to 7%; the reserve split and
the smaller device sizes change at most 1%.

Placement order (`TIERSCHED_PLACEMENT_ORDER`: `size-desc`, the default,
`semantic`, `sequential`, `random`) only matters for the contiguous layouts:
with a contiguous fill, `semantic` beats every baseline in 1 of 12 and
`random` in 6, against 12 for `size-desc`. Striping gives every sub-layer the
same split, so the order no longer changes the plan.

`plot_reserve_curve.py` sweeps the reserve one share at a time and
`fixed_reserve.py` pins the best single one; both assert their anchors.

## What each file is for

| File | |
|---|---|
| `run_paper_tables.py` | the harness: the only sanctioned way to produce a number for the paper |
| `verify_results.py` | checks invariants, not a previous run |
| `pipeline.py` | the timing engine, shared by every policy |
| `tiersched.py` | TierSched: two-stage plan search and block-wise bounded decode |
| `{flexgen,lia,cxlaimpod,llmflash,itme}_baseline.py` | one per baseline, each implementing its own paper |
| `bound.py` | the lower bound on step time that no policy can beat |
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
  and the concurrency of host DRAM and device DRAM are reproduced at cycle level
  in gem5 with SimCXL; the bridge model that would validate duplexity is not run.
  No reported headline number depends on it.
- The device's internal data path is not in the gem5 model: there, device DRAM
  and NAND are separate memories picked by address. By default the engine lets
  the device serve its DRAM while it reads NAND, and charges each NAND byte to
  the device DRAM once. `sensitivity.py device` charges it twice (12/12 leads)
  and makes the device serve one request at a time (12/12, median 1.04x, when
  TierSched plans for that device; 8/12, worst 0.58x, when it plans for the
  default one).
- Baselines implement their published memory policies, not their whole systems:
  CXLAimPod's eBPF duplex co-scheduler and LIA's AMX compute offload are out of
  scope. FlexGen's weights are placed fastest tier first, the optimum of its
  linear program when capacity binds, and its schedule is timed on the shared
  engine.
- Decoder blocks are assumed uniform, and sparsity comes from published
  profiles rather than per-model measurement.
- NAND latency is charged serially per 256 KB read, with no SSD queue
  parallelism; `sensitivity.py nandlat` varies it from 0.5 to 100 us.
- A bus serves each request in the earliest idle stretch at or after the time
  it can start (`BUS_ORDER=fill`), so a request ready early can use a gap that
  a request issued before it, still waiting for another resource, left open.
  This is non-preemptive list scheduling, and like any such schedule it can be
  slightly slower with more lookahead, budget or bandwidth in rare cases (one or
  two of 20,000 random cases in `test_pipeline_differential.py`, which reports
  them). A transfer's share of the host link is booked in its own idle stretch,
  so it can end after the transfer's bus interval; the finish time takes the
  later of the two. `BUS_ORDER=fifo` keeps strict issue order.
- A sub-layer computes once all its bytes have landed, as on a GPU that copies
  weights into its own memory. A CPU computing on weights in place overlaps the
  two; `sensitivity.py streaming` runs that variant, which keeps all twelve
  leads (worst 1.01x). TierSched's eight blocks per sub-layer already let
  compute start before a sub-layer has fully arrived.
- The accelerator's PCIe link is uncapped by default. `sensitivity.py gpulink`
  caps it at 63 or 55 GB/s; all twelve leads remain.
- No physical CMM-H device. The CMM-H numbers are simulated from
  documented device specifications; the mechanism and the timing
  engine are checked on real DRAM-over-NVMe servers (`hardware/`).
