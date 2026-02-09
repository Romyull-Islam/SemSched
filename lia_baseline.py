# lia_baseline.py
import math
import pandas as pd
from collections import OrderedDict
from tiers import GiB, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import TOKENS, cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes

PL_HOST_DRAM = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"
PREFILL_TOKENS = 512

# ** LIA/TPP RIGOR **
# Linux Kernel HugePage Fault Overhead (~8us) + Context Switch
OS_FAULT_PENALTY_S = 8e-6 
# Linux Standard Readahead (2MB)
OS_READAHEAD_BYTES = 2 * 1024 * 1024  

class OSPagingManager:
    """Mimics OS Kernel (TPP/LIA) Page Management"""
    def __init__(self, cap_bytes):
        self.cap = cap_bytes
        self.used = 0
        self.lru = OrderedDict() # Mimic Active/Inactive LRU lists
    
    def access_page(self, layer_id, size):
        # On access, OS checks residency
        if layer_id in self.lru:
            self.lru.move_to_end(layer_id)
            return True # Hit (No fault)
        else:
            # Miss: Page Fault & Promote
            self._allocate(layer_id, size)
            return False
            
    def _allocate(self, layer_id, size):
        # Evict cold pages if needed (Kernel reclaim)
        while (self.used + size) > self.cap and self.lru:
            lid, sz = self.lru.popitem(last=False)
            self.used -= sz
        
        if (self.used + size) <= self.cap:
            self.lru[layer_id] = size
            self.used += size
            self.lru.move_to_end(layer_id)

def compute_time_s(flops):
    if flops <= 0: return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)

def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)
def nand_time_s(n):   return transfer_time_s(n, CXL_SSD_NAND)
def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)

# Build Model
layers = build_layers(sequence_length=PREFILL_TOKENS)
kv_inc = {L["name"]: (L["kv_cache_bytes"] // PREFILL_TOKENS) for L in layers}

# LIA Topology-Aware Placement: Host First, then spill to CXL (OS NUMA tiering)
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
for i, L in enumerate(layers):
    if L["bytes"] <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= L["bytes"]
    else:
        placement[i] = PL_CXL_DEV_NAND # Cold start in NAND (Tier 3)

os_cache = OSPagingManager(cxl_dev_dram_capacity_bytes)

# Phase 2: PREFILL (Reactive Paging)
prefill_latency = 0.0
for i, L in enumerate(layers):
    place = placement[i]
    comp_s = compute_time_s(L["flops"] * 15.0)
    
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(L["bytes"])
    else:
        # LIA: Reactive Page Faults. 
        # Check if page is in CXL DRAM (Tier 2). If not, fetch from NAND (Tier 3).
        if os_cache.access_page(i, L["bytes"]):
            mem_time = cxl_time_s(L["bytes"])
        else:
            # Fault: Transfer + Kernel Overhead
            mem_time = nand_time_s(L["bytes"]) + OS_FAULT_PENALTY_S
            
    prefill_latency += max(comp_s, mem_time)

# Phase 3: DECODE (OS Readahead)
per_token_latency = 0.0
for i, L in enumerate(layers):
    place = placement[i]
    sz = L["bytes"] + kv_inc[L["name"]]
    comp_s = compute_time_s(L["flops"])
    
    mem_time = 0.0
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(sz)
    else:
        # LIA/TPP Logic: Access triggers fault/promotion
        if os_cache.access_page(i, sz):
            mem_time = cxl_time_s(sz)
        else:
            # Miss: Fetch + Readahead
            # OS Readahead optimization: Fetch slightly more than needed (reduces latency for sequential access)
            t_miss = nand_time_s(sz) + OS_FAULT_PENALTY_S
            mem_time = t_miss 
            
            # Simple OS Readahead simulation: 
            # If we missed here, assume OS triggers prefetch for next layer if contiguous
            if i + 1 < len(layers):
                os_cache.access_page(i+1, layers[i+1]["bytes"]) # Speculative promotion
                
    per_token_latency += max(comp_s, mem_time)

print(f"Decode throughput: {1.0 / per_token_latency:.6f}")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f}")