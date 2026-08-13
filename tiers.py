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
NAND_LINK_CEILING_BPS  = 7.88e9    # PCIe Gen4 x4 to the backend SSD
CXL_HOST_LINK_CEILING  = 31.5e9    # PCIe Gen5 x8, Soltaniyeh's CMM-H host link
HOST_DRAM_CEILING      = 38.4e9    # DDR5-4800, one channel
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


# Every tier must stay inside the interface its bytes physically cross. Five of
# today's six accounting defects were terms with no such check -- 9 GB/s over a
# 7.88 GB/s link, 136 GB into a 64 GB part, a transfer billed twice. Assert the
# physics; a float expression will otherwise return any number you ask it for.
assert CXL_DEVICE_NAND.bw_Bps <= NAND_LINK_CEILING_BPS,  "NAND exceeds PCIe Gen4 x4"
assert CXL_DEVICE_DRAM.bw_Bps <= CXL_HOST_LINK_CEILING,  "CXL DRAM exceeds Gen5 x8"
assert HOST_DRAM.bw_Bps       <= HOST_DRAM_CEILING,      "host DRAM exceeds DDR5-4800"
assert NVME_STREAM_BW         <= NAND_LINK_CEILING_BPS,  "NVMe exceeds PCIe Gen4 x4"


def kv_growth_spill_time_s(kv_now, kv_reserved, kv_tier, nand=CXL_DEVICE_NAND):
    """Extra time per decode step once the KV cache outgrows what was reserved.

    Placement reserves KV at the generation mean, PREFILL + TOKENS/2, which is
    exact only at the midpoint: past it the cache is larger than its reservation
    and the excess has to displace weights out of that tier. The displaced bytes
    are then read from NAND on every subsequent step, so the cost is not the
    one-off eviction but the bandwidth difference, charged for the rest of the
    generation.

    At 16 decode steps this is 0.31 GB at FP16 B=128 and rounds away. At 512 it
    is 10 GB, and a model that ignores it reports a device holding more than it
    has -- for every policy, since all five reserve at the mean. Applied
    uniformly, it is what makes a generation-length sweep mean anything.
    """
    over = max(0.0, kv_now - kv_reserved)
    if over <= 0 or kv_tier.bw_Bps <= 0:
        return 0.0
    return over * (1.0 / nand.bw_Bps - 1.0 / kv_tier.bw_Bps)
