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
| FP16 | 16H+32C | 4.73 | 4.60 | 4.63 | 4.65 | **4.98** | 1.05x |
| FP16 | 16H+48C | 5.28 | 5.17 | 5.18 | 5.17 | **5.75** | 1.09x |
| FP16 | 16H+64C | 5.99 | 5.80 | 5.81 | 5.87 | **6.82** | 1.14x |
| INT8 | 16H+32C | 13.96 | 12.94 | 12.55 | 13.12 | **16.83** | 1.21x |
| INT8 | 16H+48C | 20.31 | 18.31 | 17.31 | 18.87 | **30.18** | 1.49x |
| INT8 | 16H+64C | 37.26 | 30.18 | 27.84 | 32.08 | **49.66** | 1.33x |

### +RTX 5090 — SemSched leads 6/6

| Quant | Memory | FlexGen | LIA | AimPod | LLMFlash | SemSched | ratio |
|---|---|---|---|---|---|---|---|
| FP16 | 16H+32C | 6.06 | 6.28 | 6.22 | 6.32 | **6.63** | 1.05x |
| FP16 | 16H+48C | 7.00 | 7.25 | 7.26 | 7.40 | **8.11** | 1.10x |
| FP16 | 16H+64C | 8.29 | 8.76 | 8.55 | 8.92 | **10.43** | 1.17x |
| INT8 | 16H+32C | 40.82 | 32.42 | 40.62 | 54.79 | **71.08** | 1.30x |
| INT8 | 16H+48C | 67.33 | 66.84 | 70.53 | 83.11 | **87.51** | 1.05x |
| INT8 | 16H+64C | 67.33 | 66.84 | 70.53 | 83.11 | **87.51** | 1.05x |

The accelerated INT8 rows at 48 and 64 GB are identical because the search
converges to the same allocation once the device is large enough: it declines
the extra 16 GB and leaves those bytes on NAND, where they ride an otherwise
idle internal bus. That is the declining-capacity result, visible as a flat
pair rather than a rising one.

An earlier revision of this file reported the accelerated INT8 16H+32C cell as
a 2% loss to FlexGen. That predates the pipelined engine, the exact evaluation
of shortlisted finalists and the declining-capacity search; the cell is now
1.30x. Numbers here are regenerated from the working tree, not carried forward.

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
| CPU-only | FP16 | 0.77 | 0.77 | 0.74 | 0.77 | 0.76 | 0.99–1.00x |
| CPU-only | INT8 | 0.77 | 0.77 | 0.76 | 0.77 | 0.77 | 1.00x |
| +RTX 5090 | FP16 | 7.84 | 7.77 | 5.99–6.42 | 7.84 | 7.55–7.63 | 0.96–0.97x |
| +RTX 5090 | INT8 | 7.84 | 7.81–7.83 | 7.53–7.84 | 7.84 | 7.80 | 0.99–1.00x |

Flat across all five policies and all three cache sizes: prefill is compute-bound
at the `2*P*S*B` floor, so placement cannot move it. SemSched pays at most 4% for staging traffic, and nothing at INT8: an
earlier revision of this file showed a 9% INT8 penalty, which was the phantom
staging budget since removed. **No policy prefills anomalously slowly** — the 5120 s
figure previously attributed to LLM-in-a-Flash was a `%.1f` rounding artifact
(0.065402 printed as 0.1, then 512/0.1) and does not exist.

---

## 3. Batch sweep — every configuration, B = 1..128

`active_frac = 1 - (1 - 0.46)^B` is LLM-in-a-Flash's own model of how much of an
MLP sub-layer fires; it saturates at 1.000 by B=16. Regenerate with
`python run_paper_tables.py --batch-sweep`. SemSched wins every cell at B >= 8
and loses every cell at B <= 4, all of them to LLM-in-a-Flash's sparsity
discount; Fig. 4 in the paper plots the medians and the min-max band.

```
==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.04      0.04      0.04      0.11      0.05    0.44x         0.460
   2        0.09      0.08      0.09      0.13      0.09    0.71x         0.708
   4        0.17      0.16      0.18      0.20      0.19    0.94x         0.915
   8        0.34      0.31      0.35      0.36      0.37    1.03x         0.993
  16        0.68      0.63      0.69      0.71      0.74    1.04x         1.000
  32        1.32      1.25      1.34      1.37      1.44    1.05x         1.000
  64        2.54      2.51      2.55      2.58      2.73    1.06x         1.000
 128        4.73      4.60      4.63      4.65      4.98    1.05x         1.000

==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.05      0.04      0.05      0.16      0.06    0.36x         0.460
   2        0.10      0.09      0.10      0.16      0.11    0.69x         0.708
   4        0.20      0.18      0.20      0.23      0.23    0.96x         0.915
   8        0.39      0.35      0.40      0.42      0.45    1.07x         0.993
  16        0.77      0.70      0.79      0.82      0.88    1.08x         1.000
  32        1.50      1.40      1.54      1.57      1.71    1.09x         1.000
  64        2.87      2.81      2.89      2.93      3.21    1.10x         1.000
 128        5.28      5.17      5.18      5.17      5.75    1.09x         1.000

==============================================================================
CPU-only   Qwen2.5 72B FP16  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.06      0.05      0.06      0.30      0.07    0.24x         0.460
   2        0.11      0.10      0.12      0.22      0.14    0.65x         0.708
   4        0.23      0.20      0.24      0.28      0.28    0.99x         0.915
   8        0.45      0.41      0.47      0.50      0.56    1.12x         0.993
  16        0.89      0.81      0.92      0.96      1.10    1.14x         1.000
  32        1.74      1.62      1.77      1.84      2.10    1.14x         1.000
  64        3.30      3.24      3.29      3.40      3.90    1.15x         1.000
 128        5.99      5.80      5.81      5.87      6.82    1.14x         1.000

==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.14      0.10      0.14      0.85      0.19    0.22x         0.460
   2        0.28      0.20      0.29      0.71      0.38    0.53x         0.708
   4        0.55      0.41      0.57      0.76      0.75    0.99x         0.915
   8        1.10      0.81      1.12      1.27      1.48    1.16x         0.993
  16        2.15      1.62      2.20      2.43      2.88    1.19x         1.000
  32        4.17      3.24      4.13      4.54      5.45    1.20x         1.000
  64        7.82      6.48      7.46      8.05      9.93    1.23x         1.000
 128       13.96     12.94     12.55     13.12     16.83    1.21x         1.000

==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.23      0.14      0.25      0.76      0.41    0.54x         0.460
   2        0.46      0.29      0.49      1.15      0.81    0.70x         0.708
   4        0.91      0.57      0.95      1.75      1.62    0.93x         0.915
   8        1.81      1.15      1.87      2.42      3.24    1.34x         0.993
  16        3.50      2.30      3.54      4.42      6.36    1.44x         1.000
  32        6.65      4.59      6.56      7.86     12.41    1.58x         1.000
  64       12.04      9.17     11.18     12.87     20.56    1.60x         1.000
 128       20.31     18.31     17.31     18.87     30.18    1.49x         1.000

==============================================================================
CPU-only   Qwen2.5 72B INT8  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.36      0.24      0.38      0.73      0.45    0.62x         0.460
   2        0.73      0.47      0.75      1.12      0.90    0.81x         0.708
   4        1.45      0.95      1.51      1.86      1.80    0.97x         0.915
   8        2.91      1.90      3.00      3.49      3.60    1.03x         0.993
  16        5.78      3.79      5.95      6.86      7.10    1.03x         1.000
  32       11.47      7.58     11.70     13.41     13.80    1.03x         1.000
  64       22.56     15.13     22.47     25.59     26.54    1.04x         1.000
 128       37.26     30.18     27.84     32.08     49.66    1.33x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.06      0.05      0.06      0.33      0.07    0.21x         0.460
   2        0.12      0.10      0.12      0.23      0.13    0.60x         0.708
   4        0.23      0.21      0.24      0.29      0.27    0.92x         0.915
   8        0.46      0.41      0.48      0.51      0.53    1.04x         0.993
  16        0.91      0.82      0.95      0.99      1.05    1.05x         1.000
  32        1.77      1.64      1.82      1.91      2.01    1.05x         1.000
  64        3.35      3.28      3.44      3.57      3.76    1.05x         1.000
 128        6.06      6.28      6.22      6.32      6.63    1.05x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.07      0.06      0.07      0.54      0.09    0.16x         0.460
   2        0.14      0.12      0.14      0.34      0.18    0.52x         0.708
   4        0.28      0.24      0.29      0.37      0.35    0.95x         0.915
   8        0.55      0.49      0.58      0.63      0.70    1.11x         0.993
  16        1.08      0.97      1.13      1.21      1.36    1.12x         1.000
  32        2.10      1.94      2.20      2.32      2.59    1.11x         1.000
  64        3.94      3.88      4.10      4.28      4.74    1.11x         1.000
 128        7.00      7.25      7.26      7.40      8.11    1.10x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B FP16  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.09      0.07      0.09      0.48      0.12    0.26x         0.460
   2        0.17      0.15      0.19      0.68      0.24    0.36x         0.708
   4        0.35      0.29      0.37      0.51      0.49    0.95x         0.915
   8        0.69      0.58      0.72      0.82      0.96    1.17x         0.993
  16        1.35      1.17      1.44      1.57      1.85    1.18x         1.000
  32        2.59      2.33      2.71      2.96      3.51    1.18x         1.000
  64        4.78      4.66      4.96      5.34      6.42    1.20x         1.000
 128        8.29      8.76      8.55      8.92     10.43    1.17x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+32C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.61      0.26      0.66      1.51      0.93    0.62x         0.460
   2        1.21      0.52      1.32      2.29      1.87    0.82x         0.708
   4        2.41      1.04      2.64      3.81      3.72    0.98x         0.915
   8        4.81      2.08      5.26      7.10      7.37    1.04x         0.993
  16        9.54      4.16     10.25     13.79     14.50    1.05x         1.000
  32       18.73      8.30     19.66     26.33     27.63    1.05x         1.000
  64       30.44     16.46     31.45     46.63     50.63    1.09x         1.000
 128       40.82     32.42     40.62     54.79     71.08    1.30x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+48C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.61      0.55      0.66      1.51      0.93    0.62x         0.460
   2        1.21      1.10      1.32      2.29      1.87    0.82x         0.708
   4        2.41      2.21      2.64      3.81      3.72    0.98x         0.915
   8        4.81      4.42      5.26      7.10      7.37    1.04x         0.993
  16        9.54      8.83     10.42     13.79     14.50    1.05x         1.000
  32       18.73     17.54     20.31     26.34     27.72    1.05x         1.000
  64       36.01     34.51     38.66     48.37     50.63    1.05x         1.000
 128       67.33     66.84     70.53     83.11     87.51    1.05x         1.000

==============================================================================
+RTX 5090 (28 GB)   Qwen2.5 72B INT8  16H+64C   batch sweep
==============================================================================
   B     FlexGen       LIA    AimPod  LLMFlash  SemSched    ratio   active_frac
------------------------------------------------------------------------------
   1        0.61      0.55      0.66      1.51      0.93    0.62x         0.460
   2        1.21      1.10      1.32      2.29      1.87    0.82x         0.708
   4        2.41      2.21      2.64      3.81      3.72    0.98x         0.915
   8        4.81      4.42      5.26      7.10      7.37    1.04x         0.993
  16        9.54      8.83     10.42     13.79     14.50    1.05x         1.000
  32       18.73     17.54     20.31     26.34     27.72    1.05x         1.000
  64       36.01     34.51     38.66     48.37     50.63    1.05x         1.000
 128       67.33     66.84     70.53     83.11     87.51    1.05x         1.000
```

## 4. Phase-1 warmup ablation — CPU-only

| Quant | Memory | best baseline | warmup ON | warmup OFF | contributes | ratio |
|---|---|---|---|---|---|---|
| FP16 | 16H+32C | 4.73 | 4.98 | 4.98 | **0.0%** | 1.05x |
| FP16 | 16H+48C | 5.28 | 5.75 | 5.75 | **0.0%** | 1.09x |
| FP16 | 16H+64C | 5.99 | 6.82 | 6.82 | **0.0%** | 1.14x |
| INT8 | 16H+32C | 13.96 | 16.83 | 16.83 | **0.0%** | 1.21x |
| INT8 | 16H+48C | 20.31 | 30.18 | 30.18 | **0.0%** | 1.49x |
| INT8 | 16H+64C | 37.26 | 49.66 | 49.66 | **0.0%** | 1.33x |

Turning Phase-1 warmup off changes nothing, in every cell, to three decimal
places. The shipped scheduler stages nothing before decode unless staging is
granted room explicitly, and when it is granted room the paper measures it
losing to overlap in all twelve configurations. Warmup is not part of the
result; the reservation and the residue spreading are.

Regenerate with `python run_paper_tables.py --ablation`.

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
