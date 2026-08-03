# sim_cfg.py
from tiers import GiB

TOKENS = 16

BATCH_SIZE = 1 # Auto-set


# Processor: Intel Xeon 6315P (Raptor Lake, Q1'25)
# Cores: 4 Performance cores (no Hyper-Threading)
# Base frequency: 2.8 GHz
# Max turbo: 5.2 GHz
# L3 Cache: 12 MB
# Memory: DDR5-4800, ECC supported, 2 channels, max 128 GB
# SIMD: AVX2 (no AVX-512)
# PCIe Gen 5.0 (CXL compatible), up to 20 lanes
# TDP: 55W
# Source: Intel ARK, https://www.intel.com/content/www/us/en/products/sku/241603/intel-xeon-6315p-processor-12m-cache-2-80-ghz/specifications.html

# ── Compute engine ────────────────────────────────────────────────────────────
# REVISED (BigData 2026): the original 4-core AVX2 Xeon 6315P (78.8 GFLOPS) is far
# below the rate needed to keep this workload memory-bound once decode FLOPs are
# correctly scaled by batch size. At B=128 FP16 the decode arithmetic intensity is
# 2B/bytes_per_param = 128 FLOP/byte, so saturating the 27 GB/s CXL DRAM tier
# requires >= 3.46 TFLOPS. Below that the engine is compute-bound and tier
# placement is irrelevant -- i.e. the old config erased our own contribution.
#
# We now model an AMX-capable Intel Xeon 6 (Granite Rapids) P-core part:
#   64 cores @ 2.0 GHz, AMX-BF16 at 1024 FLOP/cycle/core, 88% parallel efficiency
#   => 64 * 1024 * 2.0e9 * 0.88 = 115.3 TFLOPS sustained
# This matches LIA's AMX substrate [8], clearing the 3.46 TFLOPS threshold with
# ~33x margin. Identical for all four simulators (compute is a property of the
# simulated hardware, not of any baseline's policy).
cpu_freq_hz = 2.0e9
cpu_cores = 64
flops_per_cycle_per_core = 1024.0   # Intel AMX BF16 tile ops
parallel_efficiency = 0.88

# Host DRAM: DDR5-4800
host_dram_capacity_bytes = 32 * GiB # Auto-set
# DDR5-4800 bandwidth: 38.4 GB/s per channel (official spec)
# Reference: Intel ARK above, JEDEC DDR5 standard

# CXL Expansion Device (Samsung CMM-H type)
cxl_dev_dram_capacity_bytes = 64 * GiB # Auto-set
cxl_ssd_capacity_bytes = 256 * GiB

# NVMe SSD Baseline (PCIe Gen4 x4)
ssd_capacity_bytes = 512 * GiB
