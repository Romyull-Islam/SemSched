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

## 3. Batch sweep — every configuration, B = 1..128

`active_frac = 1 - (1 - 0.46)^B` is LLM-in-a-Flash's own model of how much of an
MLP sub-layer fires; it saturates at 1.000 by B=16. Regenerate with
`python run_paper_tables.py --batch-sweep`.

```
==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.04      0.04      0.04      0.11      0.07    0.62x         0.460
   2        0.09      0.08      0.09      0.13      0.11    0.83x         0.708
   4        0.18      0.16      0.18      0.20      0.19    0.94x         0.915
   8        0.35      0.31      0.35      0.36      0.37    1.03x         0.993
  16        0.70      0.62      0.69      0.70      0.74    1.04x         1.000
  32        1.35      1.24      1.34      1.36      1.43    1.05x         1.000
  64        2.57      2.44      2.55      2.54      2.72    1.06x         1.000
 128        4.63      4.60      4.63      4.72      4.96    1.05x         1.000

==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.05      0.04      0.05      0.17      0.08    0.47x         0.460
   2        0.10      0.09      0.10      0.17      0.13    0.75x         0.708
   4        0.20      0.18      0.20      0.24      0.22    0.90x         0.915
   8        0.40      0.35      0.40      0.43      0.43    1.00x         0.993
  16        0.80      0.70      0.79      0.83      0.85    1.02x         1.000
  32        1.54      1.38      1.54      1.60      1.66    1.04x         1.000
  64        2.90      2.73      2.89      2.95      3.18    1.08x         1.000
 128        5.16      5.17      5.18      5.12      5.74    1.11x         1.000

==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.06      0.05      0.06      0.28      0.10    0.35x         0.460
   2        0.12      0.10      0.12      0.24      0.15    0.62x         0.708
   4        0.24      0.20      0.24      0.31      0.25    0.82x         0.915
   8        0.47      0.40      0.47      0.53      0.50    0.94x         0.993
  16        0.93      0.80      0.92      1.02      0.98    0.96x         1.000
  32        1.79      1.59      1.77      1.94      1.91    0.99x         1.000
  64        3.33      3.14      3.29      3.52      3.63    1.03x         1.000
 128        5.84      5.80      5.81      5.96      6.64    1.11x         1.000

==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.15      0.10      0.14      0.55      0.25    0.45x         0.460
   2        0.29      0.20      0.29      0.87      0.38    0.44x         0.708
   4        0.58      0.40      0.57      0.84      0.60    0.72x         0.915
   8        1.15      0.80      1.12      1.38      1.19    0.86x         0.993
  16        2.23      1.59      2.20      2.61      2.35    0.90x         1.000
  32        4.22      3.14      4.13      4.83      4.53    0.94x         1.000
  64        7.61      6.08      7.46      8.42      8.42    1.00x         1.000
 128       12.76     11.45     12.55     13.37     15.15    1.13x         1.000

==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.26      0.14      0.25      0.54      0.39    0.73x         0.460
   2        0.51      0.29      0.49      0.80      0.62    0.77x         0.708
   4        1.00      0.57      0.95      1.36      1.04    0.76x         0.915
   8        1.96      1.13      1.87      2.68      2.04    0.76x         0.993
  16        3.73      2.24      3.54      5.22      3.93    0.75x         1.000
  32        6.78      4.38      6.56      9.95      7.38    0.74x         1.000
  64       11.55      8.39     11.18     15.64     13.16    0.84x         1.000
 128       17.86     15.45     17.31     21.10     21.93    1.04x         1.000

==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.37      0.24      0.38      0.53      0.59    1.11x         0.460
   2        0.74      0.47      0.75      0.79      0.99    1.25x         0.708
   4        1.47      0.94      1.51      1.30      1.75    1.16x         0.915
   8        2.94      1.86      3.00      2.38      3.47    1.16x         0.993
  16        5.85      3.65      5.95      4.58      6.86    1.15x         1.000
  32       11.57      7.03     11.70      8.64     13.40    1.14x         1.000
  64       22.74     13.12     22.47     15.48     25.61    1.13x         1.000
 128       29.76     23.12     27.84     25.81     40.24    1.35x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.06      0.05      0.04      0.11      0.10    0.86x         0.460
   2        0.12      0.10      0.09      0.14      0.15    1.10x         0.708
   4        0.25      0.20      0.18      0.20      0.25    1.00x         0.915
   8        0.49      0.41      0.35      0.37      0.49    1.00x         0.993
  16        0.96      0.82      0.70      0.73      0.96    1.00x         1.000
  32        1.86      1.64      1.36      1.41      1.87    1.01x         1.000
  64        3.51      3.26      2.61      2.67      3.56    1.01x         1.000
 128        6.29      6.28      4.84      4.99      6.49    1.03x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.08      0.06      0.05      0.18      0.13    0.69x         0.460
   2        0.15      0.12      0.10      0.18      0.19    1.06x         0.708
   4        0.30      0.24      0.20      0.25      0.30    1.00x         0.915
   8        0.59      0.49      0.40      0.44      0.59    1.00x         0.993
  16        1.16      0.97      0.79      0.86      1.17    1.00x         1.000
  32        2.23      1.94      1.56      1.66      2.25    1.01x         1.000
  64        4.17      3.86      2.97      3.11      4.23    1.01x         1.000
 128        7.32      7.25      5.44      5.53      7.47    1.02x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.10      0.07      0.06      0.38      0.16    0.44x         0.460
   2        0.19      0.15      0.12      0.26      0.25    0.97x         0.708
   4        0.38      0.29      0.24      0.32      0.38    1.00x         0.915
   8        0.75      0.58      0.47      0.55      0.75    1.00x         0.993
  16        1.47      1.17      0.93      1.06      1.46    1.00x         1.000
  32        2.80      2.33      1.79      2.02      2.79    1.00x         1.000
  64        5.13      4.63      3.39      3.74      5.13    1.00x         1.000
 128        8.76      8.76      6.14      6.48      8.90    1.02x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.64      0.26      0.14      0.87      1.03    1.18x         0.460
   2        1.28      0.52      0.29      0.98      1.58    1.24x         0.708
   4        2.56      1.04      0.58      0.89      2.62    1.02x         0.915
   8        5.11      2.07      1.13      1.46      5.20    1.02x         0.993
  16       10.14      4.14      2.25      2.80     10.31    1.02x         1.000
  32       19.75      8.21      4.29      5.29     20.02    1.01x         1.000
  64       35.94     16.17      8.01      9.55     34.75    0.97x         1.000
 128       45.83     31.35     14.19     15.97     44.93    0.98x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.64      0.55      0.25      0.77      1.26    1.64x         0.460
   2        1.28      1.10      0.50      1.15      1.90    1.48x         0.708
   4        2.56      2.20      0.96      1.96      3.05    1.19x         0.915
   8        5.11      4.40      1.91      3.91      6.07    1.19x         0.993
  16       10.14      8.75      3.66      7.14     11.99    1.18x         1.000
  32       19.75     17.23      6.98     12.43     23.39    1.18x         1.000
  64       37.83     33.34     12.46     19.77     44.26    1.17x         1.000
 128       70.38     62.63     20.58     28.03     77.56    1.10x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.64      0.55      0.38      0.71      1.26    1.77x         0.460
   2        1.28      1.10      0.75      1.06      1.90    1.48x         0.708
   4        2.56      2.20      1.51      1.75      3.05    1.19x         0.915
   8        5.11      4.40      3.00      3.26      6.07    1.19x         0.993
  16       10.14      8.75      5.95      6.41     11.99    1.18x         1.000
  32       19.75     17.23     11.72     12.55     23.39    1.18x         1.000
  64       37.83     33.34     22.52     24.08     44.60    1.18x         1.000
 128       70.38     62.63     37.42     44.88     81.62    1.16x         1.000
```

### Where the advantage is strongest, and where it is not

| region | result |
|---|---|
| **INT8 16H+48C and 16H+64C, +RTX 5090** | **8/8 each**, 1.10-1.77x. The most consistent region in the study. |
| **INT8 16H+64C, CPU-only** | **8/8**, 1.11-1.35x. The device's native platform. |
| INT8 16H+32C, +RTX 5090 | 6/8; loses at B=64 and 128 (0.97x, 0.98x) -- 32 GB is too tight for a reserve to pay for itself. |
| FP16, both platforms | wins concentrate at B>=8; at B=1-4 LLM-in-a-Flash leads on its own sparsity mechanism, where its active fraction is 0.46-0.92. |
| INT8 16H+48C, CPU-only | loses to LLM-in-a-Flash from B=1 to 64 (0.73-0.84x) and leads only at B=128. |

Two boundaries worth stating plainly. Below B=4 LLM-in-a-Flash wins wherever its
sliding window has room, which is its published design point and not a defect in
our accounting. And a device small relative to the model leaves nothing to
reserve, so the mechanism that carries the result switches off exactly where
memory is tightest -- the regime the paper otherwise argues it targets.

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
