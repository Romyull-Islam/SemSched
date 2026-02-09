# flashneuron_baseline.py
import math
from collections import deque
from tiers import HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import TOKENS, cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency, host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes

PL_HOST_DRAM = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

# ** FlashNeuron RIGOR **
HW_CHANNELS = 12 # Dedicated HW I/O Lanes
HW_QUEUE_DEPTH = 32 # Fixed hardware FIFO buffer size

def compute_time_s(flops): 
    if flops <= 0: return 0.0
    return flops / (cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency)

def cxl_time_s(n): return transfer_time_s(n, CXL_DRAM)
def nand_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def dram_time_s(n): return transfer_time_s(n, HOST_DRAM)

# Build Model & Placement
layers = build_layers(sequence_length=512)
kv_inc = {L["name"]: (L["kv_cache_bytes"] // 512) for L in layers}
placement = [None]*len(layers)
host_free = host_dram_capacity_bytes
cxl_free = cxl_dev_dram_capacity_bytes

# Strict Hierarchy Placement (FlashNeuron style)
for i, L in enumerate(layers):
    if L["bytes"] <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= L["bytes"]
    elif L["bytes"] <= cxl_free:
        placement[i] = PL_CXL_DEV_DRAM
        cxl_free -= L["bytes"]
    else:
        placement[i] = PL_CXL_DEV_NAND

class HardwareOffloader:
    """Mimics FPGA/ASIC offload engine (FlashNeuron)"""
    def __init__(self):
        self.channels = [0.0] * HW_CHANNELS
        self.nand_link_busy_until = 0.0
        self.fifo_queue = deque() # Strict FIFO (No semantic reordering)
        self.on_chip_mem = set()
        self.mem_used = 0
    
    def enqueue(self, idx):
        # Hardware logic: Add to FIFO if not already present or queued
        if idx not in self.on_chip_mem and idx not in self.fifo_queue:
            if len(self.fifo_queue) < HW_QUEUE_DEPTH:
                self.fifo_queue.append(idx)
            
    def tick(self, current_time, layer_sizes):
        # Process Hardware Queue
        for i in range(len(self.channels)):
            if self.channels[i] <= current_time and self.fifo_queue:
                idx = self.fifo_queue.popleft() # BLIND FIFO POP
                sz = layer_sizes[idx]
                
                # Hardware Eviction (Simple LRU / Blind Eviction)
                while (self.mem_used + sz) > cxl_dev_dram_capacity_bytes and self.on_chip_mem:
                    # FlashNeuron doesn't know "Attention" vs "MLP", it evicts blindly
                    rem = self.on_chip_mem.pop() 
                    self.mem_used -= layer_sizes[rem]
                
                # Schedule DMA Transfer
                dur = transfer_time_s(sz, CXL_SSD_NAND)
                start = max(current_time, self.channels[i], self.nand_link_busy_until)
                finish = start + dur
                
                self.channels[i] = finish
                self.nand_link_busy_until = finish
                self.on_chip_mem.add(idx)
                self.mem_used += sz

offloader = HardwareOffloader()
layer_sizes = {i: L["bytes"] for i,L in enumerate(layers)}
lat = 0.0

# Decode Loop
for i, L in enumerate(layers):
    # Hardware Prefetcher: Blindly queue next K layers
    for k in range(1, HW_QUEUE_DEPTH):
        if i+k < len(layers) and placement[i+k] == PL_CXL_DEV_NAND:
            offloader.enqueue(i+k)
            
    offloader.tick(lat, layer_sizes)
    
    comp = compute_time_s(L["flops"])
    
    if placement[i] == PL_CXL_DEV_NAND:
        if i in offloader.on_chip_mem:
            mem = cxl_time_s(L["bytes"]) # DMA Hit
        else:
            # DMA Miss: Stall until hardware fetches
            # Find earliest available channel or wait for NAND
            arrival = min([t for t in offloader.channels if t > lat], default=lat+nand_time_s(L["bytes"]))
            lat = arrival
            mem = cxl_time_s(L["bytes"])
    else:
        mem = dram_time_s(L["bytes"]) if placement[i] == PL_HOST_DRAM else cxl_time_s(L["bytes"])
        
    # Execution (Compute overlaps with DMA if hit, else adds to latency)
    lat += max(comp, mem)

# Corrected Throughput Calculation:
# lat is the total time to process ONE token across ALL layers.
# Therefore, 1.0 / lat is Tokens Per Second.
print(f"Decode throughput: {1.0 / lat:.6f}") 

# Corrected Prefill Approximation:
# Assuming prefill takes ~15x compute due to sequence length (512 vs 1) + linear memory scan
print(f"Prefill throughput: {512.0 / (lat * 15.0):.3f}")