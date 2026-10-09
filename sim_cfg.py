from tiers import GiB

# run_paper_tables.py patches TOKENS, BATCH_SIZE, the compute engine and the
# GPU, host and CXL DRAM capacities per run; the values here are defaults for
# running a simulator directly.
TOKENS = 16

BATCH_SIZE = 1


# Compute engine: dual AMD EPYC 9454 (Zen 4, Genoa), 2 sockets x 48 cores at
# 2.75 GHz, AVX-512 at 64 FLOP/cycle/core, 85% achieved -> 14.4 TFLOPS, no
# accelerator. Zeng et al. characterized CMM-H on this host.
#
# Above the 3.46 TFLOPS needed at B=128 FP16 (this engine has 4.2x that) the
# result is unchanged, 13.36 t/s and 1.29x at 16H+48C, on this part, LIA's
# 40-core Sapphire Rapids with AMX (61.4 TF), CXLAimPod's 86-core Granite Rapids
# with AMX (132.1 TF) and an RTX 4090 (115.6 TF). Below it, the Xeon 6710E of the
# CMM-H prototype paper (Sierra Forest, no AMX or AVX-512, 2.2 TF) gives 1.08x.
cpu_freq_hz = 2.75e9                # EPYC 9454 base clock
cpu_cores = 96                      # 2 sockets x 48 cores
flops_per_cycle_per_core = 64.0     # AVX-512 FMA, Zen 4
parallel_efficiency = 0.85          # across 2 sockets; LLMCompass gives 1.00
                                    # for these GEMM shapes at B=128, and a
                                    # lower value shrinks SemSched's advantage

# Accelerator memory for weights (tiers.GPU_HBM); 0 means no accelerator, as in
# both CMM-H papers. When nonzero it is a separate tier above host DRAM, not
# extra host capacity.
gpu_hbm_capacity_bytes = 0 * GiB

# Host DRAM: DDR5-4800, 38.4 GB/s per channel (JEDEC DDR5)
host_dram_capacity_bytes = 32 * GiB

# CXL expansion device (Samsung CMM-H)
cxl_dev_dram_capacity_bytes = 64 * GiB
cxl_ssd_capacity_bytes = 1024 * GiB   # 1 TB, the CMM-H prototype (Soltaniyeh
                                      # et al.; Table II); holds a 405B model's
                                      # NAND share

# Host NVMe SSD (PCIe Gen4 x4)
ssd_capacity_bytes = 512 * GiB
