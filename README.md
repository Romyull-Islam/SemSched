# SemSched

**SemSched: Semantic Sub-Layer-Aware Scheduling for Hybrid CXL-Based LLM Inference**

This repository contains the simulator, baselines, experiment driver, and
plotting code that produce the figures and tables in the SemSched paper.

SemSched targets **CMM-H** style hybrid CXL devices that pair on-card DRAM
with NAND flash on a single PCIe card. It splits each Transformer decoder
block into attention and MLP sub-layers and routes them across Host DRAM,
CXL device DRAM, and CXL NAND according to sparsity and latency
sensitivity. A two-queue full-duplex link hides KV-cache writes inside the
read stream during decode.

## Repository layout

```
.
├── semduplex_scheduler.py            # SemSched simulator (the proposed system)
├── flexgen_baseline.py               # FlexGen-style demand-paging baseline
├── lia_baseline.py                   # LIA CXL-tiering baseline
├── llmflash_baseline.py              # LLM-in-a-Flash sparsity-streaming baseline
├── sim_cfg.py                        # Hardware parameters (bandwidths, latencies, capacities)
├── model_cfg.py                      # Model definitions (Mistral 7B, Llama 13B, Qwen3 20B, Qwen2.5 72B)
├── tiers.py                          # Memory-tier abstractions and transfer-time helpers
├── cxl_link.py                       # Two-queue CXL link model used by SemSched
│
├── run_experiments.py                # Top-level driver: sweeps simulators × models × quant × mem × batch
├── final_results_with_coldload.csv   # Canonical result set used by paper figures
│
├── plot_paper_figures.py             # Generates the main paper figures from the CSV
├── cache_ablation_study.py           # Generates Fig. 1(b) caching-policy comparison
├── plot_timeline.py                  # Generates the per-token bus timeline figure
│
├── trace_workload/                   # Real-trace driver (ShareGPT) and per-mechanism ablation
│   ├── trace_runner.py               #   shared runner: patches sim parameters in a tempdir
│   ├── download_sharegpt.py          #   extracts (prefill, decode) pairs from ShareGPT V3
│   ├── ablation.py                   #   per-mechanism ablation (warmup, placement, link, eviction)
│   ├── matrix_explore.py             #   (mem × quant × batch × prefill) sweep
│   ├── plot_sharegpt.py              #   Fig. 9: ShareGPT speedup CDF and scatter
│   └── plot_matrix.py                #   heatmap + win-rate over the sweep
│
└── figures/                          # Generated PDFs (overwritten by the plotting scripts)
```

## Requirements

- Python 3.10+
- `pip install numpy pandas matplotlib seaborn tiktoken datasets`

A virtual environment is recommended; the included `.gitignore` excludes
`CXL_venv/`, `.venv/`, `venv/`, and `gem5-py310/`.

## Reproducing the paper results

### 1. Main result CSV

```bash
python run_experiments.py
```

This sweeps all four simulators across the model, quantization, memory,
and batch grids described in Table II of the paper. It writes
`final_results_with_coldload.csv`, which is the canonical input for every
main-paper figure.

### 2. Paper figures

```bash
python plot_paper_figures.py     # Figs. 4-8 and supporting plots
python cache_ablation_study.py   # Fig. 1(b) caching-policy comparison
```

Generated PDFs land in `figures/` and are overwritten on each run.

### 3. Real-workload validation on ShareGPT (Fig. 9)

```bash
python trace_workload/download_sharegpt.py     # writes sharegpt_lens.json
python trace_workload/trace_runner.py \
    --pairs trace_workload/sharegpt_lens.json \
    --n 50 --batch 128 --quant fp16 --mem 16H+64C \
    --decode 64 \
    --out trace_workload/sharegpt_n50.csv
python trace_workload/plot_sharegpt.py
```

### 4. Per-mechanism ablation (Table IV)

```bash
python trace_workload/ablation.py
```

This patches the SemSched source in a tempdir to disable one mechanism at
a time (parallel warmup, semantic placement, two-queue link,
sparsity-aware eviction), reruns the headline (FP16 16H+64C) and
tight-memory (INT8 16H+32C) configurations, and writes
`trace_workload/ablation_results.csv`.

## Notes on the simulator

- All four simulators consume the same `sim_cfg.py` and `model_cfg.py`,
  so the only thing that varies between runs is the scheduling policy.
- `trace_runner.py` patches simulator parameters by copying the sources
  into a tempdir before each run, so the originals stay byte-identical.
- Memory configurations use the `<host>H+<cxl>C` shorthand throughout
  (e.g. `16H+32C` = 16 GB Host DRAM + 32 GB CXL device DRAM).
- Hardware parameters in `sim_cfg.py` follow the CMM-H characterization
  in Soltaniyeh et al., HotStorage 2025.

## Citation

If you use this code, please cite the paper:

```
SemSched: Semantic Sub-Layer-Aware Scheduling for Hybrid CXL-Based LLM Inference.
Md Romyull Islam and Kun Suo. MASCOTS 2026 (under submission).
```
