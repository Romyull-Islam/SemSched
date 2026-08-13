# tiers.py
from dataclasses import dataclass

GiB = 1024 ** 3
IO_CHUNK_BYTES = 256 * 1024  # 256 KiB for easier calculations.

@dataclass(frozen=True)
class Tier:
    name: str
    bw_Bps: float            # sustained bandwidth (bytes/sec)
    chunk_latency_s: float   # fixed per-IO (for IO_CHUNK_BYTES)

# Host memory (DDR5-4800 ≈ 38.4 GB/s per module)
HOST_DRAM = Tier("Host DRAM", 38.4e9, 0.2e-6)

# CXL hybrid device (CMM-H): DRAM cache + NAND SSD
# Peak host-visible BW when hitting device DRAM cache: ~27 GB/s (Table 3)
CXL_DEVICE_DRAM = Tier("CXL Device DRAM (CMM-H cache)", 27.0e9, 0.505e-6)

# NAND Backend: ~5 GB/s (Fig 2) with conservative latency
CXL_DEVICE_NAND = Tier("CXL Device NAND (CMM-H)", 5.0e9, 1.547e-6)

# Hard ceiling on the NAND backend, for every policy without exception.
# Zeng et al.: "The CMM-H device employs a PCIe Gen 4 x4 NVMe SSD" -> 7.88 GB/s
# theoretical. The measured 5.0 GB/s above is 63% of it and already reflects
# streaming access. Any simulator claiming more than the ceiling is asking for
# bandwidth the bus cannot supply; assert it rather than trust review to notice.
NAND_LINK_CEILING_BPS = 7.88e9
assert CXL_DEVICE_NAND.bw_Bps <= NAND_LINK_CEILING_BPS, (
    "NAND tier exceeds PCIe Gen4 x4 capacity")

# Accelerator memory, when one is attached. Defaults to the RTX 5090's GDDR7,
# not HBM: 32 GB at 1792 GB/s on a 512-bit bus. Consumer parts do not use HBM,
# and modeling one at HBM2e's 2039 GB/s overstated it by 14%.
#   RTX 5090   32 GB GDDR7   1792 GB/s     RTX 4090  24 GB GDDR6X  1008 GB/s
#   A100 80GB  80 GB HBM2e   2039 GB/s     H100 80GB 80 GB HBM3    3350 GB/s
# Chunk latency is ~1 us at the DMA granularity we model, two orders below the
# transfer time for a whole sub-layer, and is charged as zero rather than
# guessed -- a simplification that flatters whichever policy uses the tier most.
GPU_MEM_BW_BPS = 1792e9
GPU_HBM = Tier("Accelerator memory", GPU_MEM_BW_BPS, 0.0)

# Backward-compat aliases
CXL_DRAM     = CXL_DEVICE_DRAM
CXL_SSD_NAND = CXL_DEVICE_NAND

# Host NVMe SSD (Gen4 x4)
NVME_STREAM_BW      = 7.6e9     # ~7.6 GB/s
# EQUALISED. 20 us per 256 KiB chunk, charged serially, was used by FlexGen
# alone and has no counterpart in its paper -- their cost model is pure
# bandwidth (Sec 4.3: dtoc_g = bytes / disk_to_cpu_bandwidth, no additive
# per-I/O term). It dropped FlexGen's effective NVMe from 7.6 to 4.81 GB/s and
# accounted for 93% of its excess over its own byte-accounting floor. Set to
# the CMM-H NAND chunk latency so every simulator pays the same per-chunk cost
# on its slow tier.
NVME_STREAM_LAT_S   = 1.547e-6
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