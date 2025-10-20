# filename: simulation_async_prefetch_sequential.py
# ------------------------------------------------------------
# SIMULATOR: Sequential Execution with an Asynchronous I/O Prefetch Pool
#
# DESCRIPTION:
# This simulation models a high-performance, sequential inference process that
# uses a dedicated, asynchronous I/O thread pool to hide CXL NAND latency.
# It is designed to be more robust and powerful than simple prefetching or
# imbalanced pipelining.
#
# MAIN COMPUTE ENGINE:
# - A single execution engine uses ALL available CPU cores to process one
#   layer at a time, maximizing computational throughput.
# - Before starting a layer, it checks if the data is in the CXL DRAM cache.
#   If not, the engine stalls until the I/O threads deliver the data.
#
# ASYNCHRONOUS I/O POOL:
# - A dedicated pool of background I/O threads runs in parallel to the
#   compute engine. Their only job is to prefetch data.
# - The compute engine looks several layers ahead (Prefetch Queue Depth)
#   and adds layers that reside on NAND to a "fetch request" queue.
# - The I/O threads concurrently pull requests from this queue and begin
#   fetching layers from CXL NAND into the CXL DRAM cache.
#
# THE GOAL:
# - By using multiple I/O threads and a deep fetch-ahead queue, this model aims
#   to match the I/O throughput of the NAND device to the data consumption
#   rate of the CPU, ensuring the compute engine rarely has to stall.
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
PL_HOST_DRAM        = "Host DRAM"
PL_CXL_DEV_DRAM     = "CXL Device DRAM"
PL_CXL_DEV_NAND     = "CXL Device NAND"

# ---------------------------
# Tunables
# ---------------------------
IO_THREAD_POOL_SIZE = 12      # Number of parallel I/O threads fetching from NAND
PREFETCH_QUEUE_DEPTH = 24     # How many layers ahead the compute engine looks
DEVICE_DRAM_HINT_PIN_FIRST_K = 4
DEVICE_DRAM_PIN_STRICT       = True

# --- Helpers ---
def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)
def cxlssd_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def fmt_bytes(n): return f"{n/(1024**3):.3f} GiB"
def ssd_cold_time_s(n):
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

# ---------------------------
# Build model & Placement
# ---------------------------
layers = build_layers(sequence_length=512)  # Must match sequence_length
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

# --- NEW: Calculate incremental KV cache size per token ---
sequence_length = 512
kv_cache_increment = {}  # Per-layer incremental KV cache size
total_kv_cache_increment = 0
for L in layers:
    if L["kind"] == "DecoderBlock":
        head_dim = L.get("head_dim", 128)
        kv_heads = L.get("kv_heads", 40 if len(layers) > 35 else 8)
        kv_cache_increment[L["name"]] = 2 * kv_heads * head_dim * 1 * BYTES_PER_PARAM
        total_kv_cache_increment += kv_cache_increment[L["name"]]
    else:
        kv_cache_increment[L["name"]] = 0

placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
cxl_dram_free = cxl_dev_dram_capacity_bytes

# 1. Prioritize "hot" layers in Host DRAM
hot_indices = [name_to_idx.get("final_norm"), name_to_idx.get("lm_head")]
for idx in hot_indices:
    if idx is not None:
        sz = layers[idx]["bytes"] + (kv_cache_increment[layers[idx]["name"]] * sequence_length)
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM; host_free -= sz

# 2. Fill remaining Host DRAM greedily with earliest layers
for idx, L in enumerate(layers):
    if placement[idx] is None:
        sz = L["bytes"] + (kv_cache_increment[L["name"]] * sequence_length)
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM; host_free -= sz

# 3. Fill CXL DRAM with the next earliest available layers
for idx, L in enumerate(layers):
    if placement[idx] is None:
        sz = L["bytes"] + (kv_cache_increment[L["name"]] * sequence_length)
        if sz <= cxl_dram_free:
            placement[idx] = PL_CXL_DEV_DRAM; cxl_dram_free -= sz

# 4. Spill everything else to NAND
for idx in range(len(layers)):
    if placement[idx] is None:
        placement[idx] = PL_CXL_DEV_NAND

# ---------------------------
# Simulation State Classes
# ---------------------------
class DeviceDRAMPool:
    def __init__(self, cap_bytes):
        self.cap = max(0, int(cap_bytes)); self.used_cache = 0
        self.lru = OrderedDict(); self.pinned = set()
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
        for lid in to_delete: self.used_cache -= self.lru.pop(lid)
    def add_cache_bytes(self, layer_id, add_b):
        if add_b <= 0: return
        add_b = int(add_b)
        needed = max(0, (self.used_cache + add_b) - self.cap)
        if needed > 0: self._evict_until(needed)
        if (self.used_cache + add_b) > self.cap: return False
        self.used_cache += add_b
        self.lru[layer_id] = self.lru.get(layer_id, 0) + add_b
        self.lru.move_to_end(layer_id)
        return True

class IOThread:
    def __init__(self, thread_id):
        self.id = thread_id
        self.busy_until = 0.0
        self.current_task = None

# --- Main Simulation Logic ---
rows = []
per_token_latency = 0.0
acct = {"compute_stall_s": 0, "bytes_prefetched": 0, "bytes_from_nand_miss": 0}

dev_pool = DeviceDRAMPool(cxl_dev_dram_capacity_bytes)
io_threads = [IOThread(i) for i in range(IO_THREAD_POOL_SIZE)]
fetch_queue = deque()
fetched_or_queued = set()

# Main loop over layers
for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    # Include KV cache increment for per-token transfer
    sz = L["bytes"] + kv_cache_increment[L["name"]]
    
    for i in range(1, PREFETCH_QUEUE_DEPTH + 1):
        future_idx = exec_idx + i
        if future_idx < len(layers) and placement[future_idx] == PL_CXL_DEV_NAND and future_idx not in fetched_or_queued:
            fetch_queue.append(future_idx)
            fetched_or_queued.add(future_idx)

    for thread in io_threads:
        if thread.busy_until <= per_token_latency and fetch_queue:
            layer_to_fetch_idx = fetch_queue.popleft()
            layer_to_fetch = layers[layer_to_fetch_idx]
            fetch_size = layer_to_fetch["bytes"] + kv_cache_increment[layer_to_fetch["name"]]
            
            fetch_time = cxlssd_time_s(fetch_size)
            thread.busy_until = per_token_latency + fetch_time
            thread.current_task = (layer_to_fetch_idx, fetch_size)

    layer_time = 0.0
    served_from = ""
    
    if placement[exec_idx] == PL_HOST_DRAM:
        layer_time = max(compute_time_s(L["flops"], cpu_cores), dram_time_s(sz))
        served_from = "Host DRAM"
    elif placement[exec_idx] == PL_CXL_DEV_DRAM:
        layer_time = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))
        served_from = "CXL DRAM (resident)"
    else: # Layer is on CXL NAND
        if dev_pool.cached_bytes(exec_idx) >= sz:
            layer_time = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))
            served_from = "CXL DRAM (prefetched hit)"
        else:
            served_from = "CXL NAND (stall)"
            stall_until = float('inf')
            for thread in io_threads:
                if thread.current_task and thread.current_task[0] == exec_idx:
                    stall_until = thread.busy_until
                    break
            
            if stall_until == float('inf'):
                idle_thread = min(io_threads, key=lambda th: th.busy_until)
                fetch_time = cxlssd_time_s(sz)
                stall_until = idle_thread.busy_until + fetch_time
                acct["bytes_from_nand_miss"] += sz
            
            stall_time = max(0, stall_until - per_token_latency)
            acct["compute_stall_s"] += stall_time
            per_token_latency += stall_time
            
            layer_time = max(compute_time_s(L["flops"], cpu_cores), cxl_time_s(sz))

    per_token_latency += layer_time

    for thread in io_threads:
        if thread.current_task and thread.busy_until <= per_token_latency:
            task_idx, task_bytes = thread.current_task
            if dev_pool.add_cache_bytes(task_idx, task_bytes):
                 acct["bytes_prefetched"] += task_bytes
            thread.current_task = None
            
    rows.append({
        "Layer": exec_idx + 1, "Name": L["name"], "Placement": placement[exec_idx],
        "Served_From": served_from, "Layer_Time_s": layer_time
    })

# --- Output ---
df = pd.DataFrame(rows)
df = df.reindex(columns=['Layer', 'Name', 'Placement', 'Served_From', 'Layer_Time_s'])
print(df.to_string())

total_model_bytes = sum(L["bytes"] for L in layers)
total_kv_cache_bytes = sum(L.get("kv_cache_bytes", 0) for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

throughput = 1.0 / per_token_latency if per_token_latency > 0 else 0.0
total_time_for_all_tokens = cold_load_s + (TOKENS * per_token_latency)

model_dtype_bits = int(BYTES_PER_PARAM * 8)
host_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i, p in enumerate(placement) if p == PL_HOST_DRAM)
cxl_dram_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i, p in enumerate(placement) if p == PL_CXL_DEV_DRAM)
cxl_nand_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i, p in enumerate(placement) if p == PL_CXL_DEV_NAND)

print(f"\nSummary (Sequential Execution with Asynchronous I/O Pool):")
print(f"One-time cold SSD load: {cold_load_s:.6f} s")
print(f"Single-token Latency: {per_token_latency:.6f} s")
print(f"Estimated Tokens/sec: {throughput:.6f}")
print(f"Total time for T={TOKENS}: {total_time_for_all_tokens:.6f} s")

print("\nModel Size & Placement:")
print(f"  Dtype: FP{model_dtype_bits}, Cores: {cpu_cores}, Host DRAM: {host_dram_capacity_bytes/GiB:.1f} GiB")
print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")
print(f"  Total KV cache size: {total_kv_cache_bytes:,} bytes ({fmt_bytes(total_kv_cache_bytes)})")
print(f"  Per-token KV cache update: {total_kv_cache_increment:,} bytes ({fmt_bytes(total_kv_cache_increment)})")
print(f"  Host DRAM layers: {len([p for p in placement if p == PL_HOST_DRAM])}")
print(f"  CXL DRAM layers: {len([p for p in placement if p == PL_CXL_DEV_DRAM])}")
print(f"  CXL NAND layers: {len([p for p in placement if p == PL_CXL_DEV_NAND])}")
print(f"  Host DRAM: {host_bytes:,} bytes ({fmt_bytes(host_bytes)})")
print(f"  CXL Device DRAM: {cxl_dram_bytes:,} bytes ({fmt_bytes(cxl_dram_bytes)})")
print(f"  CXL Device NAND: {cxl_nand_bytes:,} bytes ({fmt_bytes(cxl_nand_bytes)})")

print("\nPerformance Breakdown:")
print(f"  Total compute stall time (waiting for I/O): {acct['compute_stall_s']:.6f} s")
print(f"  Total bytes successfully prefetched: {acct['bytes_prefetched']:,} bytes")
print(f"  Total bytes read from NAND on a miss (stall): {acct['bytes_from_nand_miss']:,} bytes")