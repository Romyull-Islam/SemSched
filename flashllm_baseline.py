# flashllm_baseline.py
# Mimics: FlashLLM (VLDB 2024) / FlexGen (ICML 2023)
# Logic: Aggressive Sequential Prefetching (Pipelining) + Blind LRU Eviction
# Limitation: Treats layers as monolithic blobs (Semantic-Blind)

import math
from collections import deque
from tiers import HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import TOKENS, cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes

PL_HOST_DRAM = "Host DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"
PL_CXL_DEV_DRAM = "CXL Device DRAM"

# FlashLLM Specs
# Aggressive Lookahead to saturate bandwidth
PREFETCH_WINDOW = 4 

def compute_time_s(flops): 
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency) if flops > 0 else 0

def cxl_time_s(n): return transfer_time_s(n, CXL_DRAM)
def nand_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)

# Build Model
layers = build_layers(sequence_length=512)
placement = [None]*len(layers)
host_free = host_dram_capacity_bytes
cxl_free = cxl_dev_dram_capacity_bytes

# Tiered Placement (Hot Data First - Initial Load)
for i, L in enumerate(layers):
    if L["bytes"] <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= L["bytes"]
    elif L["bytes"] <= cxl_free:
        placement[i] = PL_CXL_DEV_DRAM
        cxl_free -= L["bytes"]
    else:
        placement[i] = PL_CXL_DEV_NAND

class PipelinedPrefetcher:
    """Mimics FlashLLM's Sequential Pipelining"""
    def __init__(self):
        self.link_busy_until = 0.0
        self.on_chip_mem = set()
        self.mem_used = 0
        
    def schedule(self, idx, size, current_time):
        if idx not in self.on_chip_mem:
            # Calculate transfer time (NAND -> CXL DRAM)
            start = max(current_time, self.link_busy_until)
            duration = nand_time_s(size)
            finish = start + duration
            self.link_busy_until = finish
            
            # BLIND LRU EVICTION (The Critical Flaw)
            # FlashLLM blindly makes space for the new block
            while (self.mem_used + size) > cxl_dev_dram_capacity_bytes and self.on_chip_mem:
                # Evict oldest (simulated via pop)
                # In real FlashLLM, this swaps to disk. Here we just drop from CXL cache.
                rem = self.on_chip_mem.pop() 
                self.mem_used -= layers[rem]["bytes"]
            
            self.on_chip_mem.add(idx)
            self.mem_used += size
            return finish
        return 0.0 # Already there

    def get_arrival_time(self, current_time):
        return self.link_busy_until

def simulate_flashllm(layers, is_prefill):
    # Instantiate Prefetcher INSIDE to reset state for Decode
    prefetcher = PipelinedPrefetcher() 
    
    elapsed = 0.0
    for i, L in enumerate(layers):
        # 1. Pipeline: Prefetch Future Layers
        for k in range(1, PREFETCH_WINDOW + 1):
            if i + k < len(layers) and placement[i+k] == PL_CXL_DEV_NAND:
                prefetcher.schedule(i+k, layers[i+k]["bytes"], elapsed)
        
        # 2. Compute Current Layer
        flops = L["flops"] * (15.0 if is_prefill else 1.0)
        compute = compute_time_s(flops)
        
        mem_wait = 0.0
        if placement[i] == PL_CXL_DEV_NAND:
            arrival = prefetcher.get_arrival_time(elapsed)
            
            # Check if prefetch landed in time
            if i in prefetcher.on_chip_mem and arrival <= elapsed:
                # HIT (Hidden Latency)
                mem_wait = cxl_time_s(L["bytes"])
            else:
                # STALL (Bandwidth Wall)
                if i not in prefetcher.on_chip_mem:
                    finish = prefetcher.schedule(i, L["bytes"], elapsed)
                    arrival = finish
                wait = max(0, arrival - elapsed)
                mem_wait = wait + cxl_time_s(L["bytes"])
                
        elif placement[i] == PL_HOST_DRAM:
            mem_wait = dram_time_s(L["bytes"])
            
        elapsed += max(compute, mem_wait)
    return elapsed

# Cold Load (Full Model Copy)
cxl_bytes = sum(L["bytes"] for i,L in enumerate(layers) if placement[i] in [PL_CXL_DEV_NAND, PL_CXL_DEV_DRAM])
from tiers import NVME_STREAM_BW, NVME_STREAM_LAT_S, Tier
cold_load = transfer_time_s(cxl_bytes, Tier("NVMe", NVME_STREAM_BW, NVME_STREAM_LAT_S))

pf_time = simulate_flashllm(layers, True)
dec_time = simulate_flashllm(layers, False)

# Throughput Calculation (Tokens/sec)
tps = 1.0 / dec_time 
pf_tps = 512.0 / pf_time

print(f"Decode throughput: {tps:.6f}")
print(f"Prefill throughput: {pf_tps:.3f}")
print(f"Overall throughput: {(512+TOKENS)/(cold_load+pf_time+dec_time):.3f}")