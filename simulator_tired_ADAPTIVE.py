# filename: simulator_tired_ADAPTIVE.py
# Simulation with TRUE ADAPTIVE PREFETCHING
# ================================================================

import math
import pandas as pd
from collections import OrderedDict

from tiers import (
    GiB, IO_CHUNK_BYTES, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND,
    transfer_time_s, Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
)
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, cxl_ssd_capacity_bytes
)

# Labels
PL_HOST_DRAM     = "Host DRAM"
PL_CXL_DEV_DRAM  = "CXL Device DRAM"
PL_CXL_DEV_NAND  = "CXL Device NAND"

# ==============================
# ADAPTIVE PREFETCHER CLASS
# ==============================
class AdaptivePrefetcher:
    """Dynamically adapts prefetch strategy based on runtime metrics"""
    
    def __init__(self, initial_lookahead=8, initial_chunk_mb=4):
        self.prefetch_lookahead = initial_lookahead
        self.prefetch_gran_bytes = initial_chunk_mb * 1024 * 1024
        
        # Runtime tracking
        self.prefetch_successes = 0
        self.prefetch_failures = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.adapt_history = []
    
    def get_prefetch_efficiency(self):
        """Calculate prefetch hit rate (0.0 to 1.0)"""
        total = self.prefetch_successes + self.prefetch_failures
        if total == 0:
            return 0.5
        return self.prefetch_successes / total
    
    def get_cache_hit_rate(self):
        """Calculate overall cache hit rate"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.5
        return self.cache_hits / total
    
    def adapt_lookahead_depth(self, layer_idx):
        """
        ADAPTATION 1: Dynamically adjust lookahead based on effectiveness
        
        High efficiency (>60%): Increase lookahead (prefetch further)
        Low efficiency (<30%): Decrease lookahead (save resources)
        """
        efficiency = self.get_prefetch_efficiency()
        old_lookahead = self.prefetch_lookahead
        
        if efficiency > 0.60:
            self.prefetch_lookahead = min(32, self.prefetch_lookahead + 1)
            if self.prefetch_lookahead != old_lookahead:
                msg = f"L{layer_idx}: Prefetch eff {efficiency:.1%} → lookahead ↑ to {self.prefetch_lookahead}"
                self.adapt_history.append(msg)
        elif efficiency < 0.30:
            self.prefetch_lookahead = max(4, self.prefetch_lookahead - 1)
            if self.prefetch_lookahead != old_lookahead:
                msg = f"L{layer_idx}: Prefetch eff {efficiency:.1%} → lookahead ↓ to {self.prefetch_lookahead}"
                self.adapt_history.append(msg)
    
    def adapt_chunk_size(self, layer_idx):
        """
        ADAPTATION 2: Adjust chunk granularity based on cache effectiveness
        
        High hit rate (>65%): Larger chunks (aggressive prefetch)
        Low hit rate (<35%): Smaller chunks (conservative prefetch)
        """
        hit_rate = self.get_cache_hit_rate()
        old_chunk = self.prefetch_gran_bytes
        
        if hit_rate > 0.65:
            self.prefetch_gran_bytes = min(16 * 1024 * 1024, self.prefetch_gran_bytes * 2)
        elif hit_rate < 0.35:
            self.prefetch_gran_bytes = max(1 * 1024 * 1024, self.prefetch_gran_bytes // 2)
        
        if self.prefetch_gran_bytes != old_chunk:
            msg = f"L{layer_idx}: Cache hit {hit_rate:.1%} → chunk {old_chunk/1e6:.0f}MB → {self.prefetch_gran_bytes/1e6:.0f}MB"
            self.adapt_history.append(msg)
    
    def get_adaptive_prefetch_budget(self, base_budget, layer_idx, total_layers, nand_remaining):
        """
        ADAPTATION 3: Scale prefetch budget based on urgency
        
        More NAND layers remaining → higher budget
        Few NAND layers left → lower budget
        """
        urgency = nand_remaining / max(1, total_layers)
        adaptive_budget = base_budget * (0.5 + 0.5 * urgency)
        return adaptive_budget
    
    def get_aggressiveness(self, layer_idx, total_layers):
        """
        ADAPTATION 4: Layer-position-aware aggressiveness
        
        Early (0-15%): 0.5 (normal)
        Middle (15-75%): 0.8 (aggressive)
        Late (75-100%): 1.0 (maximum)
        """
        pos = layer_idx / max(1, total_layers)
        if pos < 0.15:
            return 0.5
        elif pos < 0.75:
            return 0.8
        else:
            return 1.0
    
    def record_prefetch_success(self, bytes_prefetched):
        """Record successful prefetch"""
        self.prefetch_successes += bytes_prefetched
    
    def record_prefetch_failure(self, bytes_attempted):
        """Record failed/partial prefetch"""
        self.prefetch_failures += bytes_attempted
    
    def record_cache_hit(self, bytes_hit):
        """Record cache hit"""
        self.cache_hits += bytes_hit
    
    def record_cache_miss(self, bytes_miss):
        """Record cache miss"""
        self.cache_misses += bytes_miss


# ==============================
# Helpers
# ==============================
def compute_time_s(flops: float) -> float:
    if flops <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)
def nand_time_s(n):   return transfer_time_s(n, CXL_SSD_NAND)
def ssd_cold_time_s(n):
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

# ==============================
# Device DRAM Pool Class
# ==============================
class DeviceDramPool:
    def __init__(self, cap_bytes):
        self.cap = max(0, int(cap_bytes))
        self.used = 0
        self.lru = OrderedDict()
    
    def bytes_cached(self, layer_id):
        return self.lru.get(layer_id, 0)
    
    def add_bytes(self, layer_id, add_b):
        if add_b <= 0:
            return 0
        add_b = int(add_b)
        needed = max(0, (self.used + add_b) - self.cap)
        while needed > 0 and self.lru:
            lid, sz = self.lru.popitem(last=False)
            self.used -= sz
            needed -= sz
        if (self.used + add_b) > self.cap:
            return 0
        self.used += add_b
        self.lru[layer_id] = self.lru.get(layer_id, 0) + add_b
        self.lru.move_to_end(layer_id)
        return add_b

# ==============================
# Build layers & Placement
# ==============================
layers = build_layers(sequence_length=512)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

sequence_length = 512
kv_cache_increment = {}
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
cxl_free  = cxl_dev_dram_capacity_bytes

hot_indices = [name_to_idx.get("final_norm"), name_to_idx.get("lm_head")]
for idx in hot_indices:
    if idx is not None:
        sz = layers[idx]["bytes"] + (kv_cache_increment[layers[idx]["name"]] * sequence_length)
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= sz
for idx, L in enumerate(layers):
    if placement[idx] is not None: continue
    sz = L["bytes"] + (kv_cache_increment[L["name"]] * sequence_length)
    if sz <= host_free:
        placement[idx] = PL_HOST_DRAM
        host_free -= sz
for idx, L in enumerate(layers):
    if placement[idx] is not None: continue
    sz = L["bytes"] + (kv_cache_increment[L["name"]] * sequence_length)
    if sz <= cxl_free:
        placement[idx] = PL_CXL_DEV_DRAM
        cxl_free -= sz
for idx in range(len(layers)):
    if placement[idx] is None:
        placement[idx] = PL_CXL_DEV_NAND

pinned_cxl = {i for i,p in enumerate(placement) if p == PL_CXL_DEV_DRAM}
pinned_cxl_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i in pinned_cxl)
dyn_cxl_capacity = max(0, cxl_dev_dram_capacity_bytes - pinned_cxl_bytes)
pool = DeviceDramPool(dyn_cxl_capacity)

# ==============================
# Initialize Adaptive Prefetcher
# ==============================
prefetcher = AdaptivePrefetcher(initial_lookahead=8, initial_chunk_mb=4)
nand_layer_indices = [i for i in range(len(layers)) if placement[i] == PL_CXL_DEV_NAND]

# ==============================
# Execute with Adaptive Prefetch
# ==============================
rows = []
per_token_latency = 0.0
cxl_hit_bytes_cum, cxl_miss_bytes_cum = 0, 0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    sz = L["bytes"] + kv_cache_increment[L["name"]]
    comp_time = compute_time_s(L["flops"])
    mem_time, served_from_parts = 0.0, []

    if placement[exec_idx] == PL_HOST_DRAM:
        mem_time = dram_time_s(sz)
        served_from_parts.append(PL_HOST_DRAM)
    elif placement[exec_idx] == PL_CXL_DEV_DRAM:
        mem_time = cxl_time_s(sz)
        cxl_hit_bytes_cum += sz
        served_from_parts.append(f"{PL_CXL_DEV_DRAM} (resident)")
        prefetcher.record_cache_hit(sz)
    else:  # PL_CXL_DEV_NAND
        staged_b = pool.bytes_cached(exec_idx)
        if staged_b > 0:
            mem_time += cxl_time_s(staged_b)
            cxl_hit_bytes_cum += staged_b
            served_from_parts.append(f"CXL DRAM (cache {staged_b/1e6:.1f}MB)")
            prefetcher.record_cache_hit(staged_b)
        rem_b = sz - staged_b
        if rem_b > 0:
            mem_time += nand_time_s(rem_b)
            cxl_miss_bytes_cum += rem_b
            pool.add_bytes(exec_idx, rem_b)
            served_from_parts.append(f"CXL NAND ({rem_b/1e6:.1f}MB)")
            prefetcher.record_cache_miss(rem_b)

    layer_time = max(comp_time, mem_time)
    per_token_latency += layer_time

    # Count remaining NAND layers
    nand_remaining = sum(1 for i in range(exec_idx + 1, len(layers)) if placement[i] == PL_CXL_DEV_NAND)
    
    # ADAPTIVE PREFETCH with dynamic parameters
    pf_bytes, pf_time, pf_targets = 0, 0.0, []
    
    # Get adaptive budget based on urgency
    adaptive_budget = prefetcher.get_adaptive_prefetch_budget(
        layer_time, exec_idx, len(layers), nand_remaining
    )
    
    # Get layer-specific aggressiveness
    aggressiveness = prefetcher.get_aggressiveness(exec_idx, len(layers))
    
    budget_s = adaptive_budget
    if budget_s > 0:
        for k in range(1, prefetcher.prefetch_lookahead + 1):
            if budget_s <= 0: break
            next_idx = exec_idx + k
            if next_idx >= len(layers) or placement[next_idx] != PL_CXL_DEV_NAND:
                continue
            
            need_b = (layers[next_idx]["bytes"] + kv_cache_increment[layers[next_idx]["name"]]) - pool.bytes_cached(next_idx)
            if need_b <= 0:
                continue
            pf_targets.append(next_idx + 1)
            
            # Apply aggressiveness to chunk size
            effective_chunk = int(prefetcher.prefetch_gran_bytes * aggressiveness)
            
            while need_b > 0 and budget_s > 0:
                chunk = min(effective_chunk, need_b)
                t_chunk = nand_time_s(chunk)
                
                if t_chunk <= budget_s:
                    bytes_added = pool.add_bytes(next_idx, chunk)
                    if bytes_added == 0:
                        break
                    prefetcher.record_prefetch_success(bytes_added)
                    need_b -= bytes_added
                    pf_bytes += bytes_added
                    pf_time += t_chunk
                    budget_s -= t_chunk
                else:
                    partial_bytes = max(0, int(chunk * (budget_s / t_chunk)))
                    if partial_bytes > 0:
                        bytes_added = pool.add_bytes(next_idx, partial_bytes)
                        prefetcher.record_prefetch_success(bytes_added)
                        pf_bytes += bytes_added
                        pf_time += budget_s
                    else:
                        prefetcher.record_prefetch_failure(chunk)
                    budget_s = 0

    rows.append({
        "Layer": exec_idx + 1, "Name": L["name"], "Placement": placement[exec_idx],
        "Served_From": " + ".join(served_from_parts), "Bytes": sz,
        "Compute_s": comp_time, "Mem_s": mem_time, "Layer_Time_s": layer_time,
        "Prefetch_Bytes": pf_bytes, "Prefetch_Time_s": pf_time,
        "Prefetch_Lookahead": prefetcher.prefetch_lookahead,
        "Prefetch_Chunk_MB": prefetcher.prefetch_gran_bytes / 1e6,
        "DeviceDRAM_Cache_GB": round(pool.used / GiB, 3),
    })

    # Adapt every 5 layers
    if (exec_idx + 1) % 5 == 0:
        prefetcher.adapt_lookahead_depth(exec_idx + 1)
        prefetcher.adapt_chunk_size(exec_idx + 1)

# ==============================
# Final Summary
# ==============================
df = pd.DataFrame(rows)
df.to_csv("sim_adaptive_prefetch.csv", index=False)
print(df.to_string())

print("\n" + "="*100)
print("ADAPTIVE PREFETCHER TUNING HISTORY")
print("="*100)
if prefetcher.adapt_history:
    for msg in prefetcher.adapt_history:
        print(f"  {msg}")
else:
    print("  No adaptations needed (metrics stable)")

total_model_bytes = sum(L["bytes"] for L in layers)
total_kv_cache_bytes = sum(kv_cache_increment[L["name"]] * sequence_length for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

throughput_tokens_per_sec = 1.0 / per_token_latency if per_token_latency > 0 else 0.0
total_time_s_all_tokens = cold_load_s + (TOKENS * per_token_latency)

model_dtype_bits = int(BYTES_PER_PARAM * 8)
host_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i,p in enumerate(placement) if p == PL_HOST_DRAM)
cxl_bytes  = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i,p in enumerate(placement) if p == PL_CXL_DEV_DRAM)
ssd_bytes  = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i,p in enumerate(placement) if p == PL_CXL_DEV_NAND)
def fmt_bytes(n): return f"{n / GiB:.3f} GiB"

print(f"\nSummary (Sequential Execution with ADAPTIVE Prefetching):")
print(f"One-time cold SSD load: {cold_load_s:.6f} s")
print(f"Single-token Latency: {per_token_latency:.6f} s")
print(f"Estimated Tokens/sec (sequential): {throughput_tokens_per_sec:.6f}")
print(f"Total time for T={TOKENS}: {total_time_s_all_tokens:.6f} s")

print("\nModel Size:")
print(f"  Dtype: FP{model_dtype_bits} ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")
print(f"  Total KV cache size: {total_kv_cache_bytes:,} bytes ({fmt_bytes(total_kv_cache_bytes)})")
print(f"  Per-token KV cache update: {total_kv_cache_increment:,} bytes ({fmt_bytes(total_kv_cache_increment)})")

print("\nPlacement Breakdown (by bytes):")
print(f"  Host DRAM: {host_bytes:,} ({fmt_bytes(host_bytes)})")
print(f"  CXL Device DRAM (pinned): {cxl_bytes:,} ({fmt_bytes(cxl_bytes)})")
print(f"  CXL Device NAND: {ssd_bytes:,} ({fmt_bytes(ssd_bytes)})")
print(f"  Dynamic CXL DRAM Pool Capacity: {dyn_cxl_capacity/GiB:.3f} GiB")

print("\nRuntime Traffic Served (first token):")
print(f"  From CXL Device DRAM (Hits: resident + cache): {cxl_hit_bytes_cum:,} bytes")
print(f"  From CXL Device NAND (Misses): {cxl_miss_bytes_cum:,} bytes")
print(f"  Total Prefetched Bytes: {df['Prefetch_Bytes'].sum():,} bytes")

print("\nAdaptive Prefetcher Metrics:")
print(f"  Final Prefetch Efficiency: {prefetcher.get_prefetch_efficiency():.1%}")
print(f"  Final Cache Hit Rate: {prefetcher.get_cache_hit_rate():.1%}")
print(f"  Final Lookahead Depth: {prefetcher.prefetch_lookahead}")
print(f"  Final Chunk Size: {prefetcher.prefetch_gran_bytes/1e6:.0f}MB")
print(f"  Prefetch Successes: {prefetcher.prefetch_successes:,} bytes")
print(f"  Prefetch Failures: {prefetcher.prefetch_failures:,} bytes")
