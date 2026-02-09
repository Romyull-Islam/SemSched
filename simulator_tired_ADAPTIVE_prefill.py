# simulator_tired_ADAPTIVE_prefill.py
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
PL_HOST_DRAM    = "Host DRAM"
PL_CXL_DEV_DRAM = "CXL Device DRAM"
PL_CXL_DEV_NAND = "CXL Device NAND"

# Prefill configuration
PREFILL_TOKENS = 512
PREFILL_FLOP_MULTIPLIER = 15.0

# ==============================
# ADAPTIVE PREFETCHER CLASS
# ==============================
class AdaptivePrefetcher:
    """Dynamically adapts prefetch strategy based on runtime metrics"""
    def __init__(self, initial_lookahead=8, initial_chunk_mb=4):
        self.prefetch_lookahead = initial_lookahead
        self.prefetch_gran_bytes = initial_chunk_mb * 1024 * 1024
        self.prefetch_successes = 0
        self.prefetch_failures = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.adapt_history = []

    def get_prefetch_efficiency(self):
        total = self.prefetch_successes + self.prefetch_failures
        if total == 0:
            return 0.5
        return self.prefetch_successes / total

    def get_cache_hit_rate(self):
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.5
        return self.cache_hits / total

    def adapt_lookahead_depth(self, layer_idx):
        """ADAPTATION 1: Dynamically adjust lookahead based on effectiveness"""
        efficiency = self.get_prefetch_efficiency()
        old_lookahead = self.prefetch_lookahead
        if efficiency > 0.60:
            self.prefetch_lookahead = min(32, self.prefetch_lookahead + 1)
            if self.prefetch_lookahead != old_lookahead:
                msg = f"L{layer_idx}: Prefetch eff {efficiency:.1%} -> lookahead UP to {self.prefetch_lookahead}"
                self.adapt_history.append(msg)
        elif efficiency < 0.30:
            self.prefetch_lookahead = max(4, self.prefetch_lookahead - 1)
            if self.prefetch_lookahead != old_lookahead:
                msg = f"L{layer_idx}: Prefetch eff {efficiency:.1%} -> lookahead DOWN to {self.prefetch_lookahead}"
                self.adapt_history.append(msg)

    def adapt_chunk_size(self, layer_idx):
        """ADAPTATION 2: Adjust chunk granularity based on cache effectiveness"""
        hit_rate = self.get_cache_hit_rate()
        old_chunk = self.prefetch_gran_bytes
        if hit_rate > 0.65:
            self.prefetch_gran_bytes = min(16 * 1024 * 1024, self.prefetch_gran_bytes * 2)
        elif hit_rate < 0.35:
            self.prefetch_gran_bytes = max(1 * 1024 * 1024, self.prefetch_gran_bytes // 2)
        if self.prefetch_gran_bytes != old_chunk:
            msg = f"L{layer_idx}: Cache hit {hit_rate:.1%} -> chunk {old_chunk/1e6:.0f}MB -> {self.prefetch_gran_bytes/1e6:.0f}MB"
            self.adapt_history.append(msg)

    def get_adaptive_prefetch_budget(self, base_budget, layer_idx, total_layers, nand_remaining):
        """ADAPTATION 3: Scale prefetch budget based on urgency"""
        urgency = nand_remaining / max(1, total_layers)
        adaptive_budget = base_budget * (0.5 + 0.5 * urgency)
        return adaptive_budget

    def get_aggressiveness(self, layer_idx, total_layers):
        """ADAPTATION 4: Layer-position-aware aggressiveness"""
        pos = layer_idx / max(1, total_layers)
        if pos < 0.15:
            return 0.5
        elif pos < 0.75:
            return 0.8
        else:
            return 1.0

    def record_prefetch_success(self, bytes_prefetched):
        self.prefetch_successes += bytes_prefetched

    def record_prefetch_failure(self, bytes_attempted):
        self.prefetch_failures += bytes_attempted

    def record_cache_hit(self, bytes_hit):
        self.cache_hits += bytes_hit

    def record_cache_miss(self, bytes_miss):
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
            if layer_id in self.lru:
                self.lru.move_to_end(layer_id)
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
        self.lru[layer_id] = add_b
        self.lru.move_to_end(layer_id)
        return add_b

# ==============================
# Build layers & Sizing
# ==============================
sequence_length = PREFILL_TOKENS
layers = build_layers(sequence_length=sequence_length)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

kv_cache_increment = {}
layer_total_size = {}

for i, L in enumerate(layers):
    layer_name = L["name"]
    if L["kind"] == "DecoderBlock":
        head_dim = L.get("head_dim", 128)
        kv_heads = L.get("kv_heads", 40 if len(layers) > 35 else 8)
        kv_inc = 2 * kv_heads * head_dim * 1 * BYTES_PER_PARAM
        kv_cache_increment[layer_name] = kv_inc
        layer_total_size[layer_name] = L["bytes"] + (kv_inc * sequence_length)
    else:
        kv_cache_increment[layer_name] = 0
        layer_total_size[layer_name] = L["bytes"]

# ==============================
# Placement (Knapsack Strategy)
# ==============================
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
unpinned_layers = []

layers_to_place = []
for i, L in enumerate(layers):
    size = layer_total_size[L["name"]]
    priority = 1000
    if L["name"] == "final_norm": priority = 0
    elif L["name"] == "embed_tokens": priority = 10
    elif L["kind"] == "DecoderBlock": priority = 20 + i
    elif L["name"] == "lm_head": priority = 100
    layers_to_place.append((priority, size, i, L["name"]))

layers_to_place.sort()

for priority, size, idx, name in layers_to_place:
    if size <= host_free:
        placement[idx] = PL_HOST_DRAM
        host_free -= size
    else:
        unpinned_layers.append((size, idx, name))

pool = DeviceDramPool(cxl_dev_dram_capacity_bytes)

for idx in range(len(layers)):
    if placement[idx] is None: placement[idx] = PL_CXL_DEV_NAND

total_model_bytes = sum(L["bytes"] for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

# ==============================
# Phase 2: STANDARD PREFILL (Demand Paging)
# ==============================
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
    else:
        # Standard demand paging logic for Adaptive (No Warmup)
        # Check cache (which starts empty), if miss fetch from NAND
        if pool.bytes_cached(exec_idx) >= param_bytes:
            mem_time = cxl_time_s(param_bytes)
            served_from = "CXL DRAM (Hit)"
        else:
            mem_time = nand_time_s(param_bytes)
            served_from = "CXL NAND"
            pool.add_bytes(exec_idx, param_bytes)
    
    layer_time = max(comp_s, mem_time)
    prefill_latency += layer_time
    
    prefill_rows.append({
        "Layer": exec_idx + 1, "Name": L["name"], "Placement": place,
        "Served_From": served_from, "Layer_Time_s": layer_time
    })

prefill_df = pd.DataFrame(prefill_rows)
print(prefill_df.to_string())
print(f"\nPrefill total time: {prefill_latency:.6f}s")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f} tokens/sec")

# ==============================
# Phase 3: DECODE with Adaptive Prefetch
# ==============================
print("\n" + "="*80)
print(f"DECODE PHASE ({TOKENS} tokens, autoregressive + adaptive prefetch)")
print("="*80)

prefetcher = AdaptivePrefetcher(initial_lookahead=8, initial_chunk_mb=4)
decode_rows = []
per_token_latency = 0.0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    sz = L["bytes"] + kv_cache_increment[L["name"]]
    comp_time = compute_time_s(L["flops"])
    
    mem_time = 0.0
    nand_demand_time_s = 0.0
    served_from_parts = []

    if placement[exec_idx] == PL_HOST_DRAM:
        mem_time = dram_time_s(sz)
        served_from_parts.append(PL_HOST_DRAM)
    else:
        full_layer_size = layer_total_size[L["name"]]
        staged_b = pool.bytes_cached(exec_idx)

        if staged_b >= full_layer_size:
            mem_time += cxl_time_s(sz)
            served_from_parts.append("CXL DRAM (Hit)")
            prefetcher.record_cache_hit(sz)
            pool.add_bytes(exec_idx, sz) # Update LRU
        else:
            if staged_b > 0:
                mem_time += cxl_time_s(staged_b)
                prefetcher.record_cache_hit(staged_b)
            
            rem_b = sz - staged_b
            if rem_b > 0:
                t_nand = nand_time_s(rem_b)
                mem_time += t_nand
                nand_demand_time_s = t_nand 
                served_from_parts.append("CXL NAND (Miss)")
                prefetcher.record_cache_miss(rem_b)
            
            pool.add_bytes(exec_idx, full_layer_size)

    layer_time = max(comp_time, mem_time)
    per_token_latency += layer_time

    # Adaptive prefetch with budget correction
    nand_remaining = sum(1 for i in range(exec_idx + 1, len(layers)) if placement[i] == PL_CXL_DEV_NAND)
    raw_budget = prefetcher.get_adaptive_prefetch_budget(layer_time, exec_idx, len(layers), nand_remaining)
    budget_s = max(0.0, raw_budget - nand_demand_time_s)
    
    pf_bytes = 0
    if budget_s > 0:
        for k in range(1, prefetcher.prefetch_lookahead + 1):
            next_idx = exec_idx + k
            if next_idx >= len(layers) or placement[next_idx] != PL_CXL_DEV_NAND: continue
            
            need_b = layer_total_size[layers[next_idx]["name"]] - pool.bytes_cached(next_idx)
            if need_b > 0:
                chunk = min(need_b, int(prefetcher.prefetch_gran_bytes * prefetcher.get_aggressiveness(exec_idx, len(layers))))
                cost = nand_time_s(chunk)
                if cost <= budget_s:
                    if pool.add_bytes(next_idx, chunk) > 0:
                        prefetcher.record_prefetch_success(chunk)
                        pf_bytes += chunk
                        budget_s -= cost
                    else: break
                else: break

    if (exec_idx + 1) % 5 == 0:
        prefetcher.adapt_lookahead_depth(exec_idx + 1)
        prefetcher.adapt_chunk_size(exec_idx + 1)

    decode_rows.append({
        "Layer": exec_idx + 1, "Name": L["name"], "Placement": placement[exec_idx],
        "Served_From": "+".join(served_from_parts), "Layer_Time_s": layer_time,
        "Prefetch_Bytes": pf_bytes
    })

decode_df = pd.DataFrame(decode_rows)
print(decode_df.to_string())
print(f"\nSingle-token decode latency: {per_token_latency:.6f}s")
print(f"Decode throughput: {1.0 / per_token_latency:.6f} tokens/sec")

total_time_s = cold_load_s + prefill_latency + (TOKENS * per_token_latency)
print(f"Overall throughput: {(PREFILL_TOKENS + TOKENS) / total_time_s:.3f} tokens/sec")

# Save CSVs
prefill_df.to_csv("sim_adaptive_prefill.csv", index=False)
decode_df.to_csv("sim_adaptive_decode.csv", index=False)