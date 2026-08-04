# Independent validation

This directory is **deliberately separate** from the main artifact. Nothing here
is imported by the paper's simulator, and nothing here imports from it. It exists
to answer one question a reviewer is entitled to ask:

> The headline result comes from a simulator the authors wrote. How much of it is
> a property of the system, and how much is a property of the simulator?

## Method

`independent_model.py` is a clean-room re-implementation of the same comparison.
It shares **no code** with the main artifact:

- imports nothing from the repository root — not `tiers.py`, not `model_cfg.py`,
  not `sim_cfg.py`, not `semduplex_scheduler.py`
- restates every hardware constant directly from Soltaniyeh et al. (HotStorage
  2025) rather than reading ours
- re-derives Qwen2.5 72B geometry from the model architecture
- implements each placement policy from its own paper's description
- takes the **compute term from LLMCompass** (ISCA 2024), a third-party model
  validated to within 4.1% of real GPUs, instead of from any expression of ours

```bash
python independent_model.py --llmcompass /path/to/LLMCompass
```

## Result

| Config | FlexGen | LIA | CXLAimPod | LLMFlash | **SemSched** | ratio |
|---|---|---|---|---|---|---|
| FP16 16H+64C | 4.15 | 6.30 | 7.32 | 7.29 | **19.59** | 2.68× |
| INT8 16H+32C | 8.30 | 12.59 | 17.69 | 17.43 | **40.44** | 2.29× |
| FP32 32H+64C | 2.08 | 2.46 | 2.76 | 2.75 | **9.79** | 3.55× |
| INT4 16H+32C | 16.60 | 58.03 | 86.57 | 81.16 | 86.56 | 1.00× |

## What this establishes, and what it does not

**Established — SemSched's absolute throughput reproduces exactly.**

| Config | main simulator | independent model | agreement |
|---|---|---|---|
| FP16 16H+64C | 19.58 | 19.59 | **0.05%** |
| INT8 16H+32C | 40.43 | 40.44 | **0.02%** |

Two implementations sharing no code, with compute taken from a third-party
validated model, land on the same number. SemSched's throughput is therefore a
consequence of the tier bandwidths and the placement policy, not of our
simulator's internals.

**Not established — the ratios.** The clean-room baselines are cruder than the
tuned ones in the main artifact, which implement each system's published
mechanisms in detail: LLMFlash's k=5 sliding window, row-column bundling at
1.8×, DRAM-window overflow accounting, and its unified DRAM pool. The naive
reimplementation here reaches only 7.29 t/s at FP16 against the tuned baseline's
11.73.

The direction matters and is worth stating plainly: **the independent model makes
SemSched look better than we report** — 2.68× against our published 1.67×. Our
main simulator is therefore *harder* on SemSched than a straightforward
independent implementation would be, because we invested effort in making the
baselines strong. The published ratios are the conservative ones.

**Also not established — FP32.** The independent model assumes staging always
succeeds, which is equivalent to σ=1 in the main simulator's terms, and it
returns 9.79 t/s — exactly the σ=1 value from the sensitivity sweep in §V-E.
It therefore does not test the re-fetch regime, which is the FP32 case the paper
reports as a band rather than a point estimate.

## Honest summary

This validates the **SemSched path** of the main simulator and rules out the
possibility that its absolute numbers are an artifact of that code. It does not
independently validate the **comparison**, because a clean-room baseline is not a
faithful baseline. Anyone wishing to check the ratios should read the baseline
implementations in the repository root, where each deviation from a source paper
is documented in §IV-B of the paper.
