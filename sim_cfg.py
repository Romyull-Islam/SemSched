# sim_cfg.py
from tiers import GiB

# Workload shape
TOKENS = 16

# CPU model (shared by all sims)
cpu_freq_hz = 2.4e9
cpu_cores   = 4
flops_per_cycle_per_core = 4.0
parallel_efficiency       = 0.90

# Machine capacities (shared by all sims)
host_dram_capacity_bytes    = 4 * GiB
cxl_dev_dram_capacity_bytes = 8 * GiB
cxl_ssd_capacity_bytes      = 64 * GiB
ssd_capacity_bytes          = 512 * GiB