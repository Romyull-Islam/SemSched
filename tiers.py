from dataclasses import dataclass

GiB = 1024 ** 3
IO_CHUNK_BYTES = 256 * 1024  # each chunk pays the tier's chunk_latency_s

@dataclass(frozen=True)
class Tier:
    name: str
    bw_Bps: float            # sustained bandwidth (bytes/sec)
    chunk_latency_s: float   # fixed latency per IO_CHUNK_BYTES chunk

# Host memory: DDR5-4800, ~38.4 GB/s per channel
HOST_DRAM = Tier("Host DRAM", 38.4e9, 0.2e-6)

# CXL hybrid device (CMM-H): DRAM cache + NAND SSD
# Peak host-visible BW when hitting device DRAM cache: ~27 GB/s (Table 3)
CXL_DEVICE_DRAM = Tier("CXL Device DRAM (CMM-H cache)", 27.0e9, 0.505e-6)

# NAND backend: ~5 GB/s (Fig 2) with conservative latency
CXL_DEVICE_NAND = Tier("CXL Device NAND (CMM-H)", 5.0e9, 1.547e-6)

# Physical link ceilings. Zeng et al.: "The CMM-H device employs a PCIe Gen 4 x4
# NVMe SSD", i.e. 7.88 GB/s theoretical; the measured 5.0 GB/s above is 63% of
# that and already reflects streaming access.
NAND_LINK_CEILING_BPS  = 7.88e9    # PCIe Gen4 x4 to the backend SSD
CXL_HOST_LINK_CEILING  = 31.5e9    # PCIe Gen5 x8, Soltaniyeh's CMM-H host link
HOST_DRAM_CEILING      = 38.4e9    # DDR5-4800, one channel
assert CXL_DEVICE_NAND.bw_Bps <= NAND_LINK_CEILING_BPS, (
    "NAND tier exceeds PCIe Gen4 x4 capacity")

# Accelerator memory, when one is attached: the RTX 5090's GDDR7, 32 GB at
# 1792 GB/s on a 512-bit bus (consumer parts do not use HBM).
#   RTX 5090   32 GB GDDR7   1792 GB/s     RTX 4090  24 GB GDDR6X  1008 GB/s
#   A100 80GB  80 GB HBM2e   2039 GB/s     H100 80GB 80 GB HBM3    3350 GB/s
# Chunk latency (~1 us at our DMA granularity, two orders below a sub-layer
# transfer) is charged as zero, which favors whichever policy uses this tier
# most.
GPU_MEM_BW_BPS = 1792e9
GPU_HBM = Tier("Accelerator memory", GPU_MEM_BW_BPS, 0.0)

# Short aliases
CXL_DRAM     = CXL_DEVICE_DRAM
CXL_SSD_NAND = CXL_DEVICE_NAND

# Host NVMe SSD (Gen4 x4)
NVME_STREAM_BW      = 7.6e9
# Same per-chunk latency as the CMM-H NAND tier, so every simulator pays the
# same per-chunk cost on its slow tier. FlexGen's own cost model is pure
# bandwidth (Sec 4.3: dtoc_g = bytes / disk_to_cpu_bandwidth, no per-I/O term).
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


# Every tier must stay within the bandwidth of the interface its bytes cross.
assert CXL_DEVICE_NAND.bw_Bps <= NAND_LINK_CEILING_BPS,  "NAND exceeds PCIe Gen4 x4"
assert CXL_DEVICE_DRAM.bw_Bps <= CXL_HOST_LINK_CEILING,  "CXL DRAM exceeds Gen5 x8"
assert HOST_DRAM.bw_Bps       <= HOST_DRAM_CEILING,      "host DRAM exceeds DDR5-4800"
assert NVME_STREAM_BW         <= NAND_LINK_CEILING_BPS,  "NVMe exceeds PCIe Gen4 x4"


def kv_growth_spill_time_s(kv_now, kv_reserved, kv_tier, nand=CXL_DEVICE_NAND):
    """Extra time per decode step once the KV cache outgrows its reservation.

    Placement reserves KV at the generation mean, PREFILL + TOKENS/2, so past the
    midpoint the excess displaces weights from that tier. Those weights are then
    read from NAND on every later step, so the cost charged is the bandwidth
    difference on the excess bytes.

    The excess is 0.31 GB at FP16 B=128 with 16 decode steps, but 10 GB at 512,
    so generation-length sweeps need it. All five policies call it.
    """
    over = max(0.0, kv_now - kv_reserved)
    if over <= 0 or kv_tier.bw_Bps <= 0:
        return 0.0
    return over * (1.0 / nand.bw_Bps - 1.0 / kv_tier.bw_Bps)
