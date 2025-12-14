# simulation_async_prefetch_sequential_prefill.py
# ------------------------------------------------------------
# SIMULATOR: Sequential Execution with Asynchronous I/O Prefetch + STANDARD PREFILL
#
# DESCRIPTION:
# This simulation models a high-performance, sequential inference process with:
# 1. Standard prefill phase (512 tokens batch processing)
# 2. Asynchronous I/O thread pool for aggressive NAND prefetching during decode
# 3. CXL DRAM caching with LRU eviction
# 4. Cache hit rate and prefetch effectiveness metrics
#
# PHASES:
# 1. Cold load: Load model from SSD into memory
# 2. Prefill: Process 512-token prompt in one batch pass (loads NAND layers into cache)
# 3. Decode: Generate 16 tokens autoregressively with async prefetch
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
def fmt_bytes(n): return f"{n/(1024**3):.3f} GiB"
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
# Phase 2: PREFILL (512 tokens)
# ---------------------------
print("\n" + "="*80)
print(f"PREFILL PHASE ({PREFILL_TOKENS} tokens, standard batch processing)")
print("="*80)

prefill_latency = 0.0
prefill_rows = []

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    layer_name = L["name"]
    place = placement[exec_idx]
    
    # Memory: Load parameters ONLY (KV writes overlap with compute)
    param_bytes = L["bytes"]
    
    # Prefill FLOPs: 15× decode
    decode_flops = L["flops"]
    prefill_flops = decode_flops * PREFILL_FLOP_MULTIPLIER
    comp_s = compute_time_s(prefill_flops)
    
    # Memory timing
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(param_bytes)
        served_from = PL_HOST_DRAM
    elif place == PL_CXL_DEV_DRAM:
        mem_time = cxl_time_s(param_bytes)
        served_from = "CXL DRAM (resident)"
    else:  # PL_CXL_DEV_NAND
        full_layer_size = layer_total_size[layer_name]
        staged_b = dev_pool.cached_bytes(exec_idx)
        
        if staged_b >= full_layer_size:
            mem_time = cxl_time_s(param_bytes)
            served_from = f"CXL DRAM (cache {staged_b/1e6:.1f}MB)"
        elif staged_b > 0:
            read_from_cache = min(staged_b, param_bytes)
            rem_b = param_bytes - read_from_cache
            mem_time = cxl_time_s(read_from_cache) + cxlssd_time_s(rem_b)
            served_from = f"CXL DRAM ({read_from_cache/1e6:.1f}MB) + NAND ({rem_b/1e6:.1f}MB)"
            dev_pool.add_cache_bytes(exec_idx, full_layer_size)
        else:
            mem_time = cxlssd_time_s(param_bytes)
            served_from = f"CXL NAND ({param_bytes/1e6:.1f}MB)"
            dev_pool.add_cache_bytes(exec_idx, full_layer_size)
    
    layer_time = max(comp_s, mem_time)
    prefill_latency += layer_time
    
    kv_write_bytes = kv_cache_increment[layer_name] * PREFILL_TOKENS
    
    prefill_rows.append({
        "Layer": exec_idx + 1,
        "Name": layer_name,
        "Kind": L["kind"],
        "Placement": place,
        "Served_From": served_from,
        "Param_Bytes": param_bytes,
        "KV_Write_Bytes": kv_write_bytes,
        "Compute_s": comp_s,
        "Mem_s": mem_time,
        "Layer_Time_s": layer_time
    })

prefill_df = pd.DataFrame(prefill_rows)
print(prefill_df.to_string())
print(f"\nPrefill total time: {prefill_latency:.6f}s")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f} tokens/sec")
print(f"Total KV cache written: {sum(r['KV_Write_Bytes'] for r in prefill_rows):,} bytes (overlapped)")

# ---------------------------
# Phase 3: DECODE with Async Prefetch
# ---------------------------
print("\n" + "="*80)
print(f"DECODE PHASE ({TOKENS} tokens, autoregressive + async prefetch)")
print("="*80)

decode_rows = []
per_token_latency = 0.0
acct = {"compute_stall_s": 0, "bytes_prefetched": 0, "bytes_from_nand_miss": 0}

fetch_queue = deque()
fetched_or_queued = set()

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    layer_name = L["name"]
    sz = L["bytes"] + kv_cache_increment[layer_name]
    
    # Prefetch lookahead
    for i in range(1, PREFETCH_QUEUE_DEPTH + 1):
        future_idx = exec_idx + i
        if (future_idx < len(layers) and 
            placement[future_idx] == PL_CXL_DEV_NAND and 
            future_idx not in fetched_or_queued):
            fetch_queue.append(future_idx)
            fetched_or_queued.add(future_idx)
    
    # Dispatch prefetch tasks to I/O threads
    for thread in io_threads:
        if thread.busy_until <= per_token_latency and fetch_queue:
            layer_to_fetch_idx = fetch_queue.popleft()
            layer_to_fetch = layers[layer_to_fetch_idx]
            fetch_size = layer_total_size[layer_to_fetch["name"]]
            
            fetch_time = cxlssd_time_s(fetch_size)
            thread.busy_until = per_token_latency + fetch_time
            thread.current_task = (layer_to_fetch_idx, fetch_size)
    
    # Execute current layer
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
            for thread in io_threads:
                if thread.current_task and thread.current_task[0] == exec_idx:
                    stall_until = thread.busy_until
                    break
            
            if stall_until == float('inf'):
                idle_thread = min(io_threads, key=lambda th: th.busy_until)
                fetch_time = cxlssd_time_s(full_layer_size)
                stall_until = idle_thread.busy_until + fetch_time
                acct["bytes_from_nand_miss"] += sz
            
            stall_time = max(0, stall_until - per_token_latency)
            acct["compute_stall_s"] += stall_time
            per_token_latency += stall_time
            
            layer_time = max(compute_time_s(L["flops"]), cxl_time_s(sz))
    
    per_token_latency += layer_time
    
    # Complete prefetch tasks
    for thread in io_threads:
        if thread.current_task and thread.busy_until <= per_token_latency:
            task_idx, task_bytes = thread.current_task
            if dev_pool.add_cache_bytes(task_idx, task_bytes):
                acct["bytes_prefetched"] += task_bytes
            thread.current_task = None
    
    decode_rows.append({
        "Layer": exec_idx + 1,
        "Name": layer_name,
        "Placement": placement[exec_idx],
        "Served_From": served_from,
        "Layer_Time_s": layer_time
    })

decode_df = pd.DataFrame(decode_rows)
print(decode_df.to_string())
print(f"\nSingle-token decode latency: {per_token_latency:.6f}s")
print(f"Decode throughput: {1.0 / per_token_latency:.6f} tokens/sec")

# ---------------------------
# NEW: Calculate Cache Hit Rate
# ---------------------------
cache_hits = sum(1 for r in decode_rows if "prefetched hit" in r["Served_From"])
cache_misses = sum(1 for r in decode_rows if "stall" in r["Served_From"])
total_nand_accesses = cache_hits + cache_misses

if total_nand_accesses > 0:
    cache_hit_rate = cache_hits / total_nand_accesses
else:
    cache_hit_rate = 0.0

# ---------------------------
# Summary
# ---------------------------
total_time_s = cold_load_s + prefill_latency + (TOKENS * per_token_latency)
total_tokens_processed = PREFILL_TOKENS + TOKENS

model_dtype_bits = int(BYTES_PER_PARAM * 8)
host_bytes = sum(layer_total_size[layers[i]["name"]] for i, p in enumerate(placement) if p == PL_HOST_DRAM)
cxl_dram_bytes = sum(layer_total_size[layers[i]["name"]] for i, p in enumerate(placement) if p == PL_CXL_DEV_DRAM)
cxl_nand_bytes = sum(layer_total_size[layers[i]["name"]] for i, p in enumerate(placement) if p == PL_CXL_DEV_NAND)

# NEW: Calculate Prefetch Effectiveness
if cxl_dev_dram_capacity_bytes > 0:
    prefetch_utilization = acct["bytes_prefetched"] / cxl_dev_dram_capacity_bytes
else:
    prefetch_utilization = 0.0

print("\n" + "="*80)
print("Summary (Async Prefetch + Standard Prefill)")
print("="*80)
print(f"\nPhase Breakdown:")
print(f"  Cold load (SSD→memory): {cold_load_s:.6f}s")
print(f"  Prefill ({PREFILL_TOKENS} tokens): {prefill_latency:.6f}s")
print(f"  Decode ({TOKENS} tokens): {TOKENS * per_token_latency:.6f}s ({per_token_latency:.6f}s per token)")
print(f"  TOTAL time: {total_time_s:.6f}s")
print(f"\nOverall throughput: {total_tokens_processed / total_time_s:.3f} tokens/sec")

print("\nModel Size & Placement:")
print(f"  Dtype: FP{model_dtype_bits}, Cores: {cpu_cores}, Host DRAM: {host_dram_capacity_bytes/GiB:.1f} GiB")
print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")
print(f"  Total KV cache size: {total_kv_cache_bytes:,} bytes ({fmt_bytes(total_kv_cache_bytes)})")
print(f"  Per-token KV cache update: {total_kv_cache_increment:,} bytes ({fmt_bytes(total_kv_cache_increment)})")
print(f"  Host DRAM: {host_bytes:,} bytes ({fmt_bytes(host_bytes)})")
print(f"  CXL Device DRAM: {cxl_dram_bytes:,} bytes ({fmt_bytes(cxl_dram_bytes)})")
print(f"  CXL Device NAND: {cxl_nand_bytes:,} bytes ({fmt_bytes(cxl_nand_bytes)})")

print("\nDecode Phase Performance:")
print(f"  Compute stall time (waiting for I/O): {acct['compute_stall_s']:.6f}s")
print(f"  Bytes successfully prefetched: {acct['bytes_prefetched']:,} bytes")
print(f"  Bytes read from NAND on miss (stall): {acct['bytes_from_nand_miss']:,} bytes")

# NEW: Cache Metrics
print("\nCache Effectiveness Metrics:")
print(f"  Cache hit rate (NAND layers only): {cache_hit_rate:.1%} ({cache_hits}/{total_nand_accesses} accesses)")
print(f"  Cache misses (stalls): {cache_misses}")
print(f"  Prefetch utilization: {prefetch_utilization:.1%} of CXL DRAM capacity")
print(f"  Prefetched data size: {acct['bytes_prefetched']:,} bytes ({acct['bytes_prefetched']/GiB:.2f} GiB)")
print(f"  CXL DRAM capacity: {cxl_dev_dram_capacity_bytes:,} bytes ({cxl_dev_dram_capacity_bytes/GiB:.2f} GiB)")

# Save CSVs
prefill_df.to_csv("sim_async_prefill.csv", index=False)
decode_df.to_csv("sim_async_decode.csv", index=False)

print("\nCSV files saved: sim_async_prefill.csv, sim_async_decode.csv")






