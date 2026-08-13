# SemSched — measured results

All figures regenerated from this working tree with

```bash
python run_paper_tables.py               # decode, both platforms, B=128
python run_paper_tables.py --prefill     # prefill, both platforms, B=128
python run_paper_tables.py --batch-sweep # B = 1..128
python run_paper_tables.py --ablation    # Phase-1 warmup on/off
```

Nothing here comes from an uncommitted script. If a number in the paper is not
in this file, it has no source.

**Workload.** Qwen2.5 72B, 512-token prompt, 16 decode steps, 1 TB CMM-H NAND
backend. Host DRAM DDR5-4800 at 38.4 GB/s; CXL device DRAM at 27 GB/s; NAND
backend at 5.0 GB/s, bounded by the PCIe Gen4 x4 link the prototype documents
(7.88 GB/s theoretical).

**Platforms.** Both the memory tier *and* the compute engine change between
them; attaching one without the other understates every accelerated cell.

| Platform | Engine | TFLOPS | Weights in HBM |
|---|---|---|---|
| CPU-only | 2x EPYC 9454, AVX-512, 96c @ 2.75 GHz, 0.85 | 14.4 | — |
| +RTX 5090 | 170 SMs, 512 FLOP/cyc, 2.41 GHz, 0.70 | 146.9 | 28 GB |

---

## 1. Decode throughput, B=128 (t/s)

### CPU-only

| Quant | Memory | FlexGen | LIA | AimPod | LLMFlash | SemSched | ratio |
|---|---|---|---|---|---|---|---|
| FP16 | 16H+32C | 4.03 | **5.01** | 4.89 | 4.77 | 4.76 | 0.95x |
| FP16 | 16H+48C | 4.03 | **5.62** | 5.51 | 5.33 | 5.32 | 0.95x |
| FP16 | 16H+64C | 4.03 | **6.48** | 6.22 | 6.00 | 6.04 | 0.93x |
| INT8 | 16H+32C | 8.68 | 12.97 | **14.45** | 13.92 | 14.10 | 0.98x |
| INT8 | 16H+48C | 8.68 | 18.36 | **21.12** | 19.74 | 20.58 | 0.97x |
| INT8 | 16H+64C | 8.68 | 30.31 | **38.76** | 33.86 | 38.57 | 0.99x |

SemSched leads 0/6. Parity, and the byte-accounting floor shows parity is the
ceiling here: with no accelerator there is no fast tier to place into, so every
policy moves near-identical bytes across identical tiers.

### +RTX 5090

| Quant | Memory | FlexGen | LIA | AimPod | LLMFlash | SemSched | ratio |
|---|---|---|---|---|---|---|---|
| FP16 | 16H+32C | 5.01 | **6.31** | 4.89 | 4.89 | 6.14 | 0.97x |
| FP16 | 16H+48C | 5.01 | **7.42** | 5.51 | 5.50 | 7.12 | 0.96x |
| FP16 | 16H+64C | 5.01 | **8.83** | 6.22 | 6.25 | 8.47 | 0.96x |
| INT8 | 16H+32C | 14.94 | 27.66 | 14.45 | 15.11 | **43.44** | **1.57x** |
| INT8 | 16H+48C | 14.94 | 49.47 | 21.12 | 22.69 | **63.17** | **1.28x** |
| INT8 | 16H+64C | 14.94 | 49.47 | 38.76 | 44.92 | **63.17** | **1.28x** |

SemSched leads 3/6 — the INT8 rows. At FP16 the model is 145 GB against ~92 GB
of fast memory, so NAND traffic dominates whatever the policy does.

B=128 is the *weakest* point of the INT8 surface, not the strongest: see §3.

---

## 2. Prefill throughput, B=128 (t/s per sequence)

| Platform | Quant | FlexGen | LIA | AimPod | LLMFlash | SemSched | ratio |
|---|---|---|---|---|---|---|---|
| CPU-only | FP16 | 0.77 | 0.77 | 0.77 | 0.77 | 0.76 | 0.99–1.00x |
| CPU-only | INT8 | 0.77 | 0.77 | 0.77 | 0.77 | 0.70 | 0.91x |
| +RTX 5090 | FP16 | 7.84 | 7.77 | 7.82 | 7.84 | 7.54–7.62 | 0.96–0.97x |
| +RTX 5090 | INT8 | 7.84 | 7.81–7.83 | 7.83 | 7.84 | 7.12–7.13 | 0.91x |

Flat across all five policies and all three cache sizes: prefill is compute-bound
at the `2*P*S*B` floor, so placement cannot move it. SemSched pays 3–9% for
Phase-1 staging traffic. **No policy prefills anomalously slowly** — the 5120 s
figure previously attributed to LLM-in-a-Flash was a `%.1f` rounding artifact
(0.065402 printed as 0.1, then 512/0.1) and does not exist.

---

## 3. Batch sweep, 16H+{32,48,64}C — where the advantage lives

`active_frac = 1 - (1 - 0.46)^B` is LLM-in-a-Flash's own model of how much of an
MLP sub-layer fires. It saturates at 1.000 by B=16, after which every policy
reads every byte once per step.

### +RTX 5090, INT8 — SemSched leads 24/24

| B | active_frac | FlexGen | LIA | AimPod | LLMFlash | SemSched | 32C | 48C | 64C |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.460 | 0.16 | 0.22–0.39 | 0.15–0.36 | 0.71–0.87 | 0.95 | 1.09x | 1.24x | 1.34x |
| 2 | 0.708 | 0.31 | 0.43–0.77 | 0.29–0.73 | 0.68–1.15 | 1.49 | 2.19x | 1.30x | 1.40x |
| 4 | 0.915 | 0.63 | 0.86–1.55 | 0.59–1.45 | 0.74–1.75 | 2.48 | **2.87x** | 1.58x | 1.42x |
| 8 | 0.993 | 1.24 | 1.73–3.09 | 1.14–2.89 | 1.26–3.27 | 4.92 | 2.85x | 1.59x | 1.51x |
| 16 | 1.000 | 2.42 | 3.46–6.18 | 2.28–5.72 | 2.44–6.46 | 9.69 | 2.80x | 1.57x | 1.50x |
| 32 | 1.000 | 4.65 | 6.92–12.37 | 4.36–11.23 | 4.68–12.72 | 18.78 | 2.71x | 1.52x | 1.48x |
| 64 | 1.000 | 8.59 | 13.83–24.74 | 8.14–21.65 | 8.67–24.71 | 35.33 | 2.55x | 1.43x | 1.43x |
| 128 | 1.000 | 14.94 | 27.66–49.47 | 14.45–38.76 | 15.11–44.92 | 43.44 / 63.17 | 1.57x | 1.28x | 1.28x |

SemSched's own throughput is identical across the three cache sizes up to B=64
because nothing spills: 26.8 GB of weights fit the device budget at every size.
The ratio varies only because the *baselines* differ.

Peak advantage is at **B=4–16**, not B=128.

### +RTX 5090, FP16

| B | 16H+32C | 16H+48C | 16H+64C |
|---|---|---|---|
| 1 | 0.70x | 0.57x | 0.40x |
| 2 | 0.98x | 0.95x | 0.93x |
| 4 | **1.20x** | **1.25x** | **1.33x** |
| 8 | 1.22x | 1.25x | 1.33x |
| 16 | 1.20x | 1.23x | 1.29x |
| 32 | 1.16x | 1.18x | 1.24x |
| 64 | 1.09x | 1.10x | 1.13x |
| 128 | 0.97x | 0.96x | 0.96x |

Wins B=4–64, loses at B=1–2 (LLM-in-a-Flash's sparsity regime) and at B=128
(KV grows to 20.3 GB and crowds weights onto NAND for everyone).

### CPU-only

Never exceeds 1.01x at any batch or cache size. At B=1 LLM-in-a-Flash leads by
2–3x on its own published mechanism; from B=8 upward all policies converge to
0.93–1.01x.

---

## 4. Phase-1 warmup ablation — CPU-only

| Quant | Memory | best baseline | warmup ON | warmup OFF | contributes |
|---|---|---|---|---|---|
| FP16 | 16H+32C | 5.01 | 4.76 | 4.76 | **0.0%** |
| FP16 | 16H+48C | 5.62 | 5.32 | 5.32 | **0.0%** |
| FP16 | 16H+64C | 6.48 | 6.04 | 6.04 | **0.0%** |
| INT8 | 16H+32C | 14.45 | 14.10 | 14.10 | **0.0%** |
| INT8 | 16H+48C | 21.12 | 20.58 | 20.58 | **0.0%** |
| INT8 | 16H+64C | 38.76 | 38.57 | 38.57 | **0.0%** |

Phase 1 stages **0.00 GB** in every cell, because placement and staging draw on
the same device DRAM and placement fills it first:

| Quant | Memory | staged | staging budget | KV reserved | placed in device |
|---|---|---|---|---|---|
| FP16 | 16H+32C | 0.00 G | 0.24 G | 20.31 G | 11.45 G |
| FP16 | 16H+48C | 0.00 G | 0.26 G | 20.31 G | 27.42 G |
| FP16 | 16H+64C | 0.00 G | 0.36 G | 20.31 G | 43.33 G |
| INT8 | 16H+32C | 0.00 G | 0.14 G | 10.16 G | 21.70 G |
| INT8 | 16H+48C | 0.00 G | 0.20 G | 10.16 G | 37.64 G |
| INT8 | 16H+64C | 0.00 G | 0.02 G | 10.16 G | 53.82 G |

Sub-layers are 0.4–0.9 GB, so nothing fits in a 0.02–0.36 GB window. Phase-1
staging cannot be listed as a contribution on the evidence available.

---

## 5. Corrections behind these numbers

Five defects were found after the previously reported tables were produced. Four
are in the simulator or a baseline; one was in the measurement harness.

| # | Defect | Direction | Effect |
|---|---|---|---|
| 1 | Staging cache did not reserve KV capacity, though placement did — the same budget applied at one of two sites | favoured SemSched | SemSched over-reported in 4 of 6 accelerated cells and all 6 CPU-only cells |
| 2 | `ENABLE_PREFILL_WARMUP` declared but never read | neither | every warmup ablation compared a run to itself |
| 3 | LLM-in-a-Flash charged `dram_window_frac * active_frac` for resident FFN, discounting the same sparsity twice | favoured LLM-in-a-Flash | 0.212 of the FFN read at B=1 where 0.460 fires; inert at B=128 |
| 4 | Harness set `gpu_hbm_capacity_bytes` but left the compute engine at the EPYC's 14.4 TFLOPS | against SemSched | understated every accelerated cell |
| 5 | Prefill throughput was never reported by the harness | neither | the prefill claims had no path back to the simulator |

Defect 1 was found by asking why INT8 16H+32C, 48C and 64C returned identical
throughput when only 32C has NAND-resident weights. A result that does not move
when the quantity it depends on moves is the signature to chase.
