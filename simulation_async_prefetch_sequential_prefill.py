# simulation_async_prefetch_sequential_prefill.py
# ------------------------------------------------------------
# SIMULATOR: Sequential Execution with Asynchronous I/O Prefetch
# ------------------------------------------------------------
import math
import pandas as pd
from collections import OrderedDict, deque

from tiers import (
    HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s,
    Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
)
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, cxl_ssd_capacity_bytes
)

GiB = 1024**3

# Canonical labels
PL_HOST_DRAM = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

# Tunables
IO_THREAD_POOL_SIZE = 12
PREFETCH_QUEUE_DEPTH = 24
DEVICE_DRAM_HINT_PIN_FIRST_K = 4
DEVICE_DRAM_PIN_STRICT = True

# Prefill configuration
PREFILL_TOKENS = 512
PREFILL_FLOP_MULTIPLIER = 15.0

# --- Helpers ---
def compute_time_s(flops, cores=cpu_cores):
    if flops <= 0 or cores <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)
def cxlssd_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def nand_time_s(n):   return transfer_time_s(n, CXL_SSD_NAND)
def ssd_cold_time_s(n):
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

# ---------------------------
# Build model & KV cache sizing
# ---------------------------
sequence_length = PREFILL_TOKENS
layers = build_layers(sequence_length=sequence_length)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

kv_cache_increment = {}
total_kv_cache_increment = 0
layer_full_kv_size = {}
layer_total_size = {}

for L in layers:
    layer_name = L["name"]
    if L["kind"] == "DecoderBlock":
        head_dim = L.get("head_dim", 128)
        kv_heads = L.get("kv_heads", 40 if len(layers) > 35 else 8)
        kv_inc = 2 * kv_heads * head_dim * 1 * BYTES_PER_PARAM
        kv_cache_increment[layer_name] = kv_inc
        total_kv_cache_increment += kv_inc
        full_kv = kv_inc * sequence_length
        layer_full_kv_size[layer_name] = full_kv
        layer_total_size[layer_name] = L["bytes"] + full_kv
    else:
        kv_cache_increment[layer_name] = 0
        layer_full_kv_size[layer_name] = 0
        layer_total_size[layer_name] = L["bytes"]

# ---------------------------
# Placement Strategy
# ---------------------------
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
cxl_dram_free = cxl_dev_dram_capacity_bytes

# 1. Prioritize hot layers in Host DRAM
hot_indices = [name_to_idx.get("final_norm"), name_to_idx.get("lm_head")]
for idx in hot_indices:
    if idx is not None:
        sz = layer_total_size[layers[idx]["name"]]
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= sz

# 2. Fill remaining Host DRAM greedily
for idx, L in enumerate(layers):
    if placement[idx] is None:
        sz = layer_total_size[L["name"]]
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= sz

# 3. Fill CXL DRAM
for idx, L in enumerate(layers):
    if placement[idx] is None:
        sz = layer_total_size[L["name"]]
        if sz <= cxl_dram_free:
            placement[idx] = PL_CXL_DEV_DRAM
            cxl_dram_free -= sz

# 4. Spill to NAND
for idx in range(len(layers)):
    if placement[idx] is None:
        placement[idx] = PL_CXL_DEV_NAND

# ---------------------------
# Device DRAM Pool
# ---------------------------
class DeviceDRAMPool:
    def __init__(self, cap_bytes):
        self.cap = max(0, int(cap_bytes))
        self.used_cache = 0
        self.lru = OrderedDict()
        self.pinned = set()
    
    @property
    def free_bytes(self): return max(0, self.cap - self.used_cache)
    
    def cached_bytes(self, layer_id): return self.lru.get(layer_id, 0)
    
    def _evict_until(self, need_extra):
        if need_extra <= 0: return
        to_delete = []
        for lid, sz in self.lru.items():
            if lid in self.pinned and DEVICE_DRAM_PIN_STRICT: continue
            to_delete.append(lid)
            if sum(self.lru[d] for d in to_delete) >= need_extra: break
        for lid in to_delete:
            self.used_cache -= self.lru.pop(lid)
    
    def add_cache_bytes(self, layer_id, add_b):
        if add_b <= 0:
            if layer_id in self.lru:
                self.lru.move_to_end(layer_id)
            return True
        add_b = int(add_b)
        needed = max(0, (self.used_cache + add_b) - self.cap)
        if needed > 0:
            self._evict_until(needed)
        if (self.used_cache + add_b) > self.cap:
            return False
        self.used_cache += add_b
        self.lru[layer_id] = self.lru.get(layer_id, 0) + add_b
        self.lru.move_to_end(layer_id)
        return True

class IOThread:
    def __init__(self, thread_id):
        self.id = thread_id
        self.busy_until = 0.0
        self.current_task = None

# ---------------------------
# Phase 1: Cold Load
# ---------------------------
total_model_bytes = sum(L["bytes"] for L in layers)
total_kv_cache_bytes = sum(layer_full_kv_size[L["name"]] for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

# ---------------------------
# Initialize Cache & Threads
# ---------------------------
dev_pool = DeviceDRAMPool(cxl_dev_dram_capacity_bytes)
io_threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]

# ---------------------------
# Phase 2: STANDARD PREFILL (Demand Paging)
# ---------------------------
# NOTE: No warmup loop here.
print("\n" + "="*80)
print(f"PREFILL PHASE ({PREFILL_TOKENS} tokens, standard batch processing)")
print("="*80)

prefill_latency = 0.0
prefill_rows = []

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement[exec_idx]
    param_bytes = L["bytes"]
    comp_s = compute_time_s(L["flops"] * PREFILL_FLOP_MULTIPLIER)
    
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(param_bytes)
        served_from = PL_HOST_DRAM
    elif place == PL_CXL_DEV_DRAM:
        mem_time = cxl_time_s(param_bytes)
        served_from = "CXL DRAM (resident)"
    else:
        # Standard demand paging for Async
        if dev_pool.cached_bytes(exec_idx) >= param_bytes:
            mem_time = cxl_time_s(param_bytes)
            served_from = "CXL DRAM (Hit)"
        else:
            mem_time = nand_time_s(param_bytes)
            served_from = "CXL NAND"
            dev_pool.add_cache_bytes(exec_idx, param_bytes)
    
    layer_time = max(comp_s, mem_time)
    prefill_latency += layer_time
    
    prefill_rows.append({
        "Layer": exec_idx+1, "Name": L["name"], "Placement": place,
        "Served_From": served_from, "Layer_Time_s": layer_time
    })

prefill_df = pd.DataFrame(prefill_rows)
print(prefill_df.to_string())
print(f"\nPrefill total time: {prefill_latency:.6f}s")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f} tokens/sec")

# ---------------------------
# Phase 3: DECODE with Async Prefetch
# ---------------------------
print("\n" + "="*80)
print(f"DECODE PHASE ({TOKENS} tokens, autoregressive + async prefetch)")
print("="*80)

decode_rows = []
per_token_latency = 0.0
fetch_queue = deque()
fetched_or_queued = set()

# --- GLOBAL LOCK (FIX) ---
nand_link_free_at = 0.0  

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    layer_name = L["name"]
    sz = L["bytes"] + kv_cache_increment[layer_name]
    
    # 1. Populate Queue
    for i in range(1, PREFETCH_QUEUE_DEPTH + 1):
        future_idx = exec_idx + i
        if (future_idx < len(layers) and 
            placement[future_idx] == PL_CXL_DEV_NAND and 
            future_idx not in fetched_or_queued):
            fetch_queue.append(future_idx)
            fetched_or_queued.add(future_idx)
            
    # 2. Dispatch prefetch tasks to I/O threads
    for thread in io_threads:
        if thread.busy_until <= per_token_latency and fetch_queue:
            layer_to_fetch_idx = fetch_queue.popleft()
            layer_to_fetch = layers[layer_to_fetch_idx]
            fetch_size = layer_total_size[layer_to_fetch["name"]]
            
            # Start when: compute ready, thread ready, AND LINK FREE
            start_time = max(per_token_latency, thread.busy_until, nand_link_free_at)
            fetch_duration = cxlssd_time_s(fetch_size)
            finish_time = start_time + fetch_duration
            
            thread.busy_until = finish_time
            thread.current_task = (layer_to_fetch_idx, fetch_size)
            nand_link_free_at = finish_time # Lock link
            
    # 3. Execute current layer
    layer_time = 0.0
    served_from = ""
    
    if placement[exec_idx] == PL_HOST_DRAM:
        layer_time = max(compute_time_s(L["flops"]), dram_time_s(sz))
        served_from = "Host DRAM"
    elif placement[exec_idx] == PL_CXL_DEV_DRAM:
        layer_time = max(compute_time_s(L["flops"]), cxl_time_s(sz))
        served_from = "CXL DRAM (resident)"
    else:  # NAND layer
        full_layer_size = layer_total_size[layer_name]
        if dev_pool.cached_bytes(exec_idx) >= full_layer_size:
            layer_time = max(compute_time_s(L["flops"]), cxl_time_s(sz))
            served_from = "CXL DRAM (prefetched hit)"
        else:
            served_from = "CXL NAND (stall)"
            stall_until = float('inf')
            
            # Find thread handling this task
            for thread in io_threads:
                if thread.current_task and thread.current_task[0] == exec_idx:
                    stall_until = thread.busy_until
                    break
            
            # Panic fetch (rare)
            if stall_until == float('inf'):
                idle_thread = min(io_threads, key=lambda th: th.busy_until)
                fetch_duration = cxlssd_time_s(full_layer_size)
                start_time = max(per_token_latency, idle_thread.busy_until, nand_link_free_at)
                stall_until = start_time + fetch_duration
                idle_thread.busy_until = stall_until
                nand_link_free_at = stall_until
            
            stall_time = max(0, stall_until - per_token_latency)
            per_token_latency += stall_time
            layer_time = max(compute_time_s(L["flops"]), cxl_time_s(sz))
    
    per_token_latency += layer_time
    
    # 4. Complete prefetch tasks
    for thread in io_threads:
        if thread.current_task and thread.busy_until <= per_token_latency:
            task_idx, task_bytes = thread.current_task
            dev_pool.add_cache_bytes(task_idx, task_bytes)
            thread.current_task = None
    
    decode_rows.append({
        "Layer": exec_idx + 1, "Name": layer_name, "Placement": placement[exec_idx],
        "Served_From": served_from, "Layer_Time_s": layer_time
    })

decode_df = pd.DataFrame(decode_rows)
print(decode_df.to_string())
print(f"\nSingle-token decode latency: {per_token_latency:.6f}s")
print(f"Decode throughput: {1.0 / per_token_latency:.6f} tokens/sec")

total_time_s = cold_load_s + prefill_latency + (TOKENS * per_token_latency)
print(f"Overall throughput: {(PREFILL_TOKENS + TOKENS) / total_time_s:.3f} tokens/sec")

# Save CSVs
prefill_df.to_csv("sim_async_prefill.csv", index=False)
decode_df.to_csv("sim_async_decode.csv", index=False)