# Memory-model validation against DRAMsim3

`independent_model.py` validates the **compute** half of the model against
LLMCompass. This file validates the **memory** half against
[DRAMsim3](https://github.com/umd-memsys/DRAMsim3), a cycle-accurate DRAM
simulator validated against real DRAM devices.

## What was run

```bash
git clone https://github.com/umd-memsys/DRAMsim3.git && cd DRAMsim3
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
./build/dramsim3main configs/DDR4_8Gb_x8_2666.ini -c 200000 -s stream -o out
```

DDR4-2666 is the device-DRAM generation used by the CMM-H prototype
(Soltaniyeh et al., Table 1: "48GB DDR4"). The `stream` generator matches our
access pattern: large sequential weight-tensor reads.

## Result

| Quantity | Value |
|---|---|
| Theoretical peak (2666 MT/s × 8 B) | 21.33 GB/s |
| DRAMsim3 achieved, streaming | **15.10 GB/s** |
| Efficiency | **70.8%** |
| Average read latency | 375.7 cycles |

## What this exposed

A methodological inconsistency in our tier table:

| Tier | Value | Kind |
|---|---|---|
| CXL DRAM | 27.0 GB/s | **measured achieved** (CMM-H Table 3) |
| CXL NAND | 5.0 GB/s | **measured achieved** (CMM-H Fig. 2) |
| Host DRAM | 38.4 GB/s | **JEDEC theoretical peak** for DDR5-4800 |

The two CXL tiers are achieved numbers from silicon; the host tier was a spec
peak. Since SemSched pins latency-sensitive attention sub-layers in host DRAM,
an inflated host tier flatters SemSched. That had to be tested, not argued.

## Sensitivity test

Re-running with the host tier derated by the DRAMsim3-measured efficiency
(38.4 × 0.708 = 27.2 GB/s):

| Config | Host BW | Best baseline | SemSched | Ratio |
|---|---|---|---|---|
| FP16 16H+64C | 38.4 (peak) | 11.73 | 19.58 | **1.67×** |
| FP16 16H+64C | 27.2 (achieved) | 11.44 | 19.05 | **1.66×** |
| INT8 16H+32C | 38.4 (peak) | 27.79 | 40.43 | **1.45×** |
| INT8 16H+32C | 27.2 (achieved) | 26.39 | 38.23 | **1.45×** |

Absolute throughput falls about 3% for every policy, because the derating
applies to whichever policy uses host DRAM. **The ratio moves by 0.01× at FP16
and not at all at INT8.** The reported speedups are therefore insensitive to
whether the host tier is expressed as a peak or an achieved figure, and we
report the peak value with this sensitivity disclosed rather than silently
mixing conventions.

## Coverage summary

| Half of the model | Validated against | Outcome |
|---|---|---|
| Compute term | LLMCompass (ISCA 2024, 4.1% error) | regime reproduced; utilization calibrated; SemSched throughput reproduced to 0.05% |
| Memory term | DRAMsim3 (cycle-accurate, validated) | DRAM efficiency measured at 70.8%; headline ratios move ≤0.01× when applied |

---

## Accelerator tier (added 2026-08-05)

The tier model gained an accelerator level, initially set from a vendor spec
sheet with nothing checking it. DRAMsim3 ships GDDR and HBM configurations, so
the same method used for DDR4 applies directly.

```bash
./build/dramsim3main configs/GDDR6_8Gb_x16.ini  -c 200000 -s stream -o out
./build/dramsim3main configs/HBM2_8Gb_x128.ini  -c 200000 -s stream -o out
```

| Config | Peak | DRAMsim3 achieved | Efficiency | Usable? |
|---|---|---|---|---|
| DDR4-2666 (CXL cache) | 21.33 GB/s | 15.10 GB/s | **70.8%** | yes |
| GDDR6 x16 (accelerator) | 193.9 GB/s | 115.98 GB/s | **59.8%** | yes |
| HBM2 x128 | 256.0 GB/s | 22.06 GB/s | 8.6% | **no — discarded** |

GDDR6 peak is $(1/0.66\,\text{ns}) \times 8 \times 16$ B: GDDR6 clocks WCK at
$4\times$ CK and transfers on both edges, so eight transfers per command clock,
giving 12.1 Gbps/pin — the correct figure for real GDDR6. Computing it as plain
DDR gives 48.5 GB/s and an impossible 239% efficiency, which is how the error
was caught.

**The HBM2 number is discarded rather than reported.** That configuration has 8
independent channels and the `stream` generator drives a single sequential
address stream, which cannot fill them. Its 8.6% measures the trace, not the
device. DDR4 and GDDR6 are single-channel configs, so `stream` saturates them
and their numbers stand.

### Sensitivity

Re-running with the accelerator tier derated by the measured GDDR6 efficiency
($1792 \times 0.598 = 1072$ GB/s), RTX 5090 at 16H+48C FP16:

| Batch | 1792 GB/s (spec) | 1072 GB/s (achieved) | Δ |
|---|---|---|---|
| 8 | 2.05× | 2.06× | +0.01 |
| 32 | 1.83× | 1.84× | +0.01 |
| 64 | 1.92× | 1.93× | +0.01 |
| 128 | 1.64× | 1.65× | +0.01 |

**A 40% cut in accelerator bandwidth moves every ratio by +0.01×.** The
accelerator is roughly 40–66× faster than the CXL DRAM tier either way, so it is
never the constraint; the CXL and NAND tiers set the step time. The reported
accelerator results are therefore insensitive to whether that tier is expressed
as a spec peak or a measured achieved figure, in the same way and for the same
reason as the host DRAM tier above.
