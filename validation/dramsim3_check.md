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
