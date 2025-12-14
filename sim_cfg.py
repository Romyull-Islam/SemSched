# sim_cfg.py
from tiers import GiB

TOKENS = 16

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

cpu_freq_hz = 2.8e9
cpu_cores = 4
flops_per_cycle_per_core = 8.0   # AVX2, 8-wide single precision
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
