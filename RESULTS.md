# SemSched — measured results

All figures regenerated from this working tree with

```bash
python run_paper_tables.py               # decode, both platforms, B=128
python run_paper_tables.py --prefill     # prefill, both platforms, B=128
python run_paper_tables.py --batch-sweep # B = 1..128
python run_paper_tables.py --ablation    # Phase-1 warmup on/off
python run_paper_tables.py --kv-tier     # which tier holds the KV cache
```

Nothing here comes from an uncommitted script. If a number in the paper is not
in this file, it has no source.

**Workload.** Qwen2.5 72B, 512-token prompt, 16 decode steps, 1 TB CMM-H NAND
backend. Host DRAM DDR5-4800 at 38.4 GB/s; CXL device DRAM at 27 GB/s; NAND
backend at 5.0 GB/s, bounded by the PCIe Gen4 x4 link the prototype documents.
CXL device DRAM and NAND share the Gen5 x8 host link at 31.5 GB/s.

**Timing engine.** `pipeline.py`, shared by all five policies. Memory tiers are
independent buses and run concurrently; transfers within a tier serialize; the
two device tiers are capped by the host link they share. How much of that
concurrency a policy captures is set by its prefetch depth AND by the free
memory it has to stage into -- bytes fetched ahead of use have to live
somewhere, so a deep queue over a full device is worth nothing.

| policy | prefetch depth | source |
|---|---|---|
| LIA | 0 | no weight prefetch described; CXL read on demand |
| FlexGen | 1 | Algorithm 1, "the weights load of the next layer" |
| CXLAimPod | 4 | its own PREFETCH_WINDOW |
| LLM-in-a-Flash | 5 | sliding window, k=5 |
| SemSched | 32 | PREFETCH_QUEUE_DEPTH |

**Platforms.** Both the memory tier *and* the compute engine change between
them; attaching one without the other understates every accelerated row.

| Platform | Engine | TFLOPS | Weights in HBM |
|---|---|---|---|
| CPU-only | 2x EPYC 9454, AVX-512, 96c @ 2.75 GHz, 0.85 | 14.4 | — |
| +RTX 5090 | 170 SMs, 512 FLOP/cyc, 2.41 GHz, 0.70 | 146.9 | 28 GB |

---

## 1. Decode throughput, B=128 (t/s)

### CPU-only — SemSched leads 6/6

| Quant | Memory | FlexGen | LIA | AimPod | LLMFlash | SemSched | ratio |
|---|---|---|---|---|---|---|---|
| FP16 | 16H+32C | 4.63 | 4.60 | 4.63 | 4.72 | **4.96** | 1.05x |
| FP16 | 16H+48C | 5.16 | 5.17 | 5.18 | 5.12 | **5.74** | 1.11x |
| FP16 | 16H+64C | 5.84 | 5.80 | 5.81 | 5.96 | **6.64** | 1.11x |
| INT8 | 16H+32C | 12.76 | 11.45 | 12.55 | 13.37 | **15.15** | 1.13x |
| INT8 | 16H+48C | 17.86 | 15.45 | 17.31 | 21.10 | **21.93** | 1.04x |
| INT8 | 16H+64C | 29.76 | 23.12 | 27.84 | 25.81 | **40.24** | 1.35x |

### +RTX 5090 — SemSched leads 5/6

| Quant | Memory | FlexGen | LIA | AimPod | LLMFlash | SemSched | ratio |
|---|---|---|---|---|---|---|---|
| FP16 | 16H+32C | 6.29 | 6.28 | 4.84 | 4.99 | **6.49** | 1.03x |
| FP16 | 16H+48C | 7.32 | 7.25 | 5.44 | 5.53 | **7.47** | 1.02x |
| FP16 | 16H+64C | 8.76 | 8.76 | 6.14 | 6.48 | **8.90** | 1.02x |
| INT8 | 16H+32C | **45.83** | 31.35 | 14.19 | 15.97 | 44.93 | 0.98x |
| INT8 | 16H+48C | 70.38 | 62.63 | 20.58 | 28.03 | **77.56** | 1.10x |
| INT8 | 16H+64C | 70.38 | 62.63 | 37.42 | 44.88 | **81.62** | 1.16x |

At INT8 16H+32C the device is too tight for a reserve to pay for itself, and
SemSched trails FlexGen by 2%. Reported, not hidden.

## 1b. What the advantage rests on

Two mechanisms, and they are not equally robust.

**Adaptive prefetch reservation.** Every baseline fills each tier to capacity --
FlexGen's LP maximises residency by construction, the other three fill greedily
-- so none has staging room and none can prefetch deeply whatever its queue
depth. SemSched instead searches how much host and device DRAM to hold back. The
reserved bytes stop being resident, costing NAND traffic, and start being
staging room, letting NAND transfers overlap host and accelerator reads. The
optimum is interior and configuration-dependent: 1.09 GB at INT8 16H+32C, worth
+3.3% there.

**Prefetch depth**, K=32 against 0/1/4/5. This is the larger effect at INT8 and
the more contestable one. Sensitivity to FlexGen's depth, +RTX 5090 16H+48C:

| FlexGen K | INT8 ratio | FP16 ratio |
|---|---|---|
| 1 (their Algorithm 1) | 1.10x | 1.02x |
| 2 | 1.08x | 1.02x |
| 4 | 1.05x | 1.02x |
| 8 | 1.00x | 1.02x |
| 16 | 0.99x | 1.02x |

The INT8 lead is gone if FlexGen's block schedule is read as pipelining eight
layers rather than one. Their text says "the next layer". The FP16 lead is flat
across the whole range, because FlexGen has no staging room there and depth
cannot help it -- that part rests on the reservation mechanism alone.

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
