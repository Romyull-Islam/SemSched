# sim_cfg.py
from tiers import GiB

TOKENS = 16

BATCH_SIZE = 1 # Auto-set


# NOTE: this file previously described an Intel Xeon 6315P (4C AVX2). That part
# is superseded -- see the compute-engine block below for why.

# ── Compute engine ────────────────────────────────────────────────────────────
# The CXL tier parameters below come from the CMM-H prototype of Soltaniyeh et
# al. (HotStorage 2025), whose reference platform is an Intel Xeon 6710E
# dual-socket host with the device on PCIe Gen5 x8. That host is a Sierra Forest
# E-core part -- no AMX, ~2.16 TFLOPS of AVX2 -- below the 3.46 TFLOPS needed to
# keep this workload memory-bound at B=128 FP16, so a discrete accelerator is
# required rather than optional.
#
# We model an NVIDIA L4: 58 SMs @ 2.04 GHz, 1024 FLOP/cycle/SM BF16 tensor,
# 121.2 TFLOPS peak, 84.8 TFLOPS at 70% achieved utilization -- 25x above the
# threshold. PCIe Gen4 x16 host link, 72 W, single slot.
#
# The L4 is chosen for a capacity reason, not a compute one. At B=128 with a
# 512-token prompt the FP16 KV cache alone is 21.5 GB; with activations and
# workspace that consumes the L4's entire 24 GB, leaving essentially nothing for
# weights, which is what this tier model assumes -- weights stream from host
# DRAM, CXL DRAM, and NAND. A larger accelerator changes that: 50 GB of spare
# HBM on an 80 GB part holds a third of the FP16 model and removes the NAND
# tier, at which point SemSched's staging has nothing to stage and it loses.
# That boundary is reported in the paper rather than assumed away.
#
# Results are invariant to the engine above the threshold: the L4 at 84.8
# TFLOPS, an AMX server CPU at 115, and an H100 at 574 give bit-identical
# throughput. The same model is used by all five simulators.
cpu_freq_hz = 2.04e9               # L4 boost clock
cpu_cores = 58                     # streaming multiprocessors
flops_per_cycle_per_core = 1024.0  # BF16 tensor-core FLOPs per cycle per SM
parallel_efficiency = 0.70         # achieved utilization -- CALIBRATED against
                                   # LLMCompass, which measures 1.00 for our
                                   # exact GEMM shapes at B=128; assuming less
                                   # than achieved makes compute look slower
                                   # and our advantage smaller.

# Host DRAM: DDR5-4800
host_dram_capacity_bytes = 32 * GiB # Auto-set
# DDR5-4800 bandwidth: 38.4 GB/s per channel (official spec)
# Reference: Intel ARK above, JEDEC DDR5 standard

# CXL Expansion Device (Samsung CMM-H type)
cxl_dev_dram_capacity_bytes = 64 * GiB # Auto-set
cxl_ssd_capacity_bytes = 256 * GiB

# NVMe SSD Baseline (PCIe Gen4 x4)
ssd_capacity_bytes = 512 * GiB
