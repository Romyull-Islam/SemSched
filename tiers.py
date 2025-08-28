# tiers.py
from dataclasses import dataclass

GiB = 1024 ** 3
IO_CHUNK_BYTES = 256 * 1024  # 256 KiB

@dataclass(frozen=True)
class Tier:
    name: str
    bw_Bps: float            # sustained bandwidth (bytes/sec)
    chunk_latency_s: float   # fixed per-IO (for IO_CHUNK_BYTES)


# Host memory (my assumption)
# -----------------------------
# DDR5-4800 ≈ 38.4 GB/s per module.
HOST_DRAM = Tier("Host DRAM", 38.4e9, 0.2e-6)


# CXL hybrid device (CMM-H): DRAM cache + NAND SSD
# (values taken from the uploaded CMM-H paper)
# -------------------------------------------------
# Peak host-visible BW when hitting device DRAM cache: ~27 GB/s (Table 3).
# Latency (host-visible, median) ≈ 505 ns (Table 3) — used as per-IO fixed term.
CXL_DEVICE_DRAM = Tier("CXL Device DRAM (CMM-H cache)", 27.0e9, 0.505e-6)

# For large working sets (beyond cache), BW asymptotes to ~5 GB/s (Fig. 2).
# We set a conservative per-IO fixed term to 1.547 µs (Table 3 p99.9 latency)
# as a simple way to reflect higher tail costs on misses.
CXL_DEVICE_NAND = Tier("CXL Device NAND (CMM-H)",        5.0e9, 1.547e-6)

# Backward-compat aliases (so existing imports don’t break)
CXL_DRAM     = CXL_DEVICE_DRAM
CXL_SSD_NAND = CXL_DEVICE_NAND


# Host NVMe SSD (no-CXL sim)

# Keep these as-is unless we have host SSD microbenchmarks.
NVME_STREAM_BW      = 7.6e9
NVME_STREAM_LAT_S   = 20e-6
NVME_THRASH_BW      = 300e6
NVME_THRASH_LAT_S   = 80e-6
NVME_FAULT_OVERHEAD = 8e-6

def transfer_time_s(bytes_amt: int, tier: Tier, chunk_bytes: int = IO_CHUNK_BYTES) -> float:
    if bytes_amt <= 0: return 0.0
    import math
    chunks = math.ceil(bytes_amt / chunk_bytes)
    return (bytes_amt / tier.bw_Bps) + chunks * tier.chunk_latency_s

def chunk_us(tier: Tier, chunk_bytes: int = IO_CHUNK_BYTES) -> float:
    return 1e6 * ((chunk_bytes / tier.bw_Bps) + tier.chunk_latency_s)
