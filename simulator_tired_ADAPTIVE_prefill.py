# simulator_tired_ADAPTIVE_prefill.py
# ================================================================
# Simulation with TRUE ADAPTIVE PREFETCHING + STANDARD PREFILL
# ================================================================
# This version includes:
# 1. Knapsack placement strategy (v9) with post-greedy lm_head swap
# 2. Four runtime adaptation mechanisms (lookahead, chunk size, budget, aggressiveness)
# 3. STANDARD PREFILL MODELING (512-token batch processing)
# 4. DECODE phase with adaptive prefetching (16 tokens autoregressive)
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
                msg = f"L{layer_idx}: Prefetch eff {efficiency:.1%} → lookahead ↑ to {self.prefetch_lookahead}"
                self.adapt_history.append(msg)
        elif efficiency < 0.30:
            self.prefetch_lookahead = max(4, self.prefetch_lookahead - 1)
            if self.prefetch_lookahead != old_lookahead:
                msg = f"L{layer_idx}: Prefetch eff {efficiency:.1%} → lookahead ↓ to {self.prefetch_lookahead}"
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
            msg = f"L{layer_idx}: Cache hit {hit_rate:.1%} → chunk {old_chunk/1e6:.0f}MB → {self.prefetch_gran_bytes/1e6:.0f}MB"
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
        self.lru[layer_id] = self.lru.get(layer_id, 0) + add_b
        self.lru.move_to_end(layer_id)
        return add_b

# ==============================
# Build layers & Sizing
# ==============================
sequence_length = PREFILL_TOKENS
layers = build_layers(sequence_length=sequence_length)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

kv_cache_increment = {}
total_kv_cache_increment = 0
layer_full_kv_size = {}
layer_total_size = {}

for i, L in enumerate(layers):
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

# ==============================
# Placement (Knapsack Strategy)
# ==============================
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
unpinned_layers = []

print("--- Starting Efficient Placement (v9 - Knapsack, User Priority) ---")
print(f"Host DRAM Capacity: {host_free / GiB:.2f} GiB")

layers_to_place = []
for i, L in enumerate(layers):
    size = layer_total_size[L["name"]]
    if L["name"] == "final_norm":
        priority = 0
    elif L["name"] == "embed_tokens":
        priority = 10
    elif L["kind"] == "DecoderBlock":
        priority = 20 + i
    elif L["name"] == "lm_head":
        priority = 100
    else:
        priority = 1000
    layers_to_place.append((priority, size, i, L["name"]))

layers_to_place.sort()

print(f"Packing Host DRAM...")
last_pinned_decoder = None
lm_head_idx = name_to_idx.get("lm_head", None)
lm_head_size = layer_total_size.get("lm_head", float("inf"))

for priority, size, idx, name in layers_to_place:
    if size <= host_free:
        placement[idx] = PL_HOST_DRAM
        host_free -= size
        print(f"  Pinned {name} (Prio {priority}, {size / GiB:.2f} GiB). Host free: {host_free / GiB:.2f} GiB")
        if layers[idx]["kind"] == "DecoderBlock":
            last_pinned_decoder = (idx, size)
    else:
        unpinned_layers.append((size, idx, name))

# Post-greedy swap for lm_head
if lm_head_idx is not None and placement[lm_head_idx] is None and last_pinned_decoder is not None:
    dec_idx, dec_size = last_pinned_decoder
    epsilon = int(0.01 * GiB)
    if host_free < lm_head_size and (host_free + dec_size + epsilon) >= lm_head_size:
        placement[dec_idx] = None
        host_free += dec_size
        placement[lm_head_idx] = PL_HOST_DRAM
        host_free -= lm_head_size
        unpinned_layers.append((dec_size, dec_idx, layers[dec_idx]["name"]))
        print(f"  Swapped out {layers[dec_idx]['name']} to pin lm_head ({lm_head_size / GiB:.2f} GiB). Host free: {host_free / GiB:.2f} GiB")

# Backfill wasted space
unpinned_layers.sort()
if host_free > 0 and unpinned_layers:
    print(f"Backfilling {host_free / GiB:.2f} GiB of remaining Host DRAM...")
    for size, idx, name in unpinned_layers:
        if placement[idx] is None and size <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= size
            print(f"  Backfilled {name} ({size / GiB:.2f} GiB). Host free: {host_free / GiB:.2f} GiB")

print(f"Host DRAM placement complete. Wasted space: {host_free / GiB:.2f} GiB")

# CXL DRAM is 100% dynamic cache
dyn_cxl_capacity = cxl_dev_dram_capacity_bytes
pool = DeviceDramPool(dyn_cxl_capacity)
print(f"CXL DRAM Pool (Dynamic Cache) size: {dyn_cxl_capacity / GiB:.2f} GiB")

# Remaining layers to NAND
for idx in range(len(layers)):
    if placement[idx] is None:
        placement[idx] = PL_CXL_DEV_NAND
print("--- Placement Complete ---")

# ==============================
# Phase 1: Cold Load
# ==============================
total_model_bytes = sum(L["bytes"] for L in layers)
total_kv_cache_bytes = sum(layer_full_kv_size[L["name"]] for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

# ==============================
# Phase 2: PREFILL (512 tokens)
# ==============================
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
    else:  # PL_CXL_DEV_NAND
        full_layer_size = layer_total_size[layer_name]
        staged_b = pool.bytes_cached(exec_idx)
        
        if staged_b >= full_layer_size:
            mem_time = cxl_time_s(param_bytes)
            served_from = f"CXL DRAM (cache {staged_b/1e6:.1f}MB)"
        elif staged_b > 0:
            read_from_cache = min(staged_b, param_bytes)
            rem_b = param_bytes - read_from_cache
            mem_time = cxl_time_s(read_from_cache) + nand_time_s(rem_b)
            served_from = f"CXL DRAM ({read_from_cache/1e6:.1f}MB) + NAND ({rem_b/1e6:.1f}MB)"
            pool.add_bytes(exec_idx, full_layer_size)
        else:
            mem_time = nand_time_s(param_bytes)
            served_from = f"CXL NAND ({param_bytes/1e6:.1f}MB)"
            pool.add_bytes(exec_idx, full_layer_size)
    
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

# ==============================
# Phase 3: DECODE with Adaptive Prefetch
# ==============================
print("\n" + "="*80)
print(f"DECODE PHASE ({TOKENS} tokens, autoregressive + adaptive prefetch)")
print("="*80)

prefetcher = AdaptivePrefetcher(initial_lookahead=8, initial_chunk_mb=4)
decode_rows = []
per_token_latency = 0.0
cxl_hit_bytes_cum, cxl_miss_bytes_cum = 0, 0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    layer_name = L["name"]
    sz = L["bytes"] + kv_cache_increment[layer_name]
    comp_time = compute_time_s(L["flops"])
    mem_time, served_from_parts = 0.0, []

    if placement[exec_idx] == PL_HOST_DRAM:
        mem_time = dram_time_s(sz)
        served_from_parts.append(PL_HOST_DRAM)
    else:  # PL_CXL_DEV_NAND
        full_layer_size = layer_total_size[layer_name]
        staged_b = pool.bytes_cached(exec_idx)

        if staged_b >= full_layer_size:
            mem_time += cxl_time_s(sz)
            cxl_hit_bytes_cum += sz
            served_from_parts.append(f"CXL DRAM (cache {staged_b/1e6:.1f}MB)")
            prefetcher.record_cache_hit(sz)
            pool.add_bytes(exec_idx, 0)
        else:
            if staged_b > 0:
                read_from_cache = min(staged_b, sz)
                mem_time += cxl_time_s(read_from_cache)
                cxl_hit_bytes_cum += read_from_cache
                served_from_parts.append(f"CXL DRAM (cache {read_from_cache/1e6:.1f}MB)")
                prefetcher.record_cache_hit(read_from_cache)
            else:
                read_from_cache = 0
            rem_b = sz - read_from_cache
            if rem_b > 0:
                mem_time += nand_time_s(rem_b)
                cxl_miss_bytes_cum += rem_b
                served_from_parts.append(f"CXL NAND ({rem_b/1e6:.1f}MB)")
                prefetcher.record_cache_miss(rem_b)
            pool.add_bytes(exec_idx, full_layer_size)

    layer_time = max(comp_time, mem_time)
    per_token_latency += layer_time

    nand_remaining = sum(1 for i in range(exec_idx + 1, len(layers)) if placement[i] == PL_CXL_DEV_NAND)

    # Adaptive prefetch
    pf_bytes, pf_time, pf_targets = 0, 0.0, []
    adaptive_budget = prefetcher.get_adaptive_prefetch_budget(
        layer_time, exec_idx, len(layers), nand_remaining
    )
    aggressiveness = prefetcher.get_aggressiveness(exec_idx, len(layers))

    budget_s = adaptive_budget
    if budget_s > 0:
        for k in range(1, prefetcher.prefetch_lookahead + 1):
            if budget_s <= 0: break
            next_idx = exec_idx + k
            if next_idx >= len(layers) or placement[next_idx] != PL_CXL_DEV_NAND:
                continue
            L_next = layers[next_idx]
            full_layer_size_next = layer_total_size[L_next["name"]]
            need_b = full_layer_size_next - pool.bytes_cached(next_idx)
            if need_b <= 0:
                continue
            pf_targets.append(next_idx + 1)
            effective_chunk = int(prefetcher.prefetch_gran_bytes * aggressiveness)

            while need_b > 0 and budget_s > 0:
                chunk = min(effective_chunk, need_b)
                t_chunk = nand_time_s(chunk)
                if t_chunk <= budget_s and t_chunk > 0:
                    bytes_added = pool.add_bytes(next_idx, chunk)
                    if bytes_added == 0:
                        prefetcher.record_prefetch_failure(chunk)
                        break
                    prefetcher.record_prefetch_success(bytes_added)
                    need_b -= bytes_added
                    pf_bytes += bytes_added
                    pf_time += t_chunk
                    budget_s -= t_chunk
                else:
                    if t_chunk == 0:
                        budget_s = 0
                        break
                    partial_bytes = max(0, int(chunk * (budget_s / t_chunk)))
                    if partial_bytes > 0:
                        bytes_added = pool.add_bytes(next_idx, partial_bytes)
                        if bytes_added == 0:
                            prefetcher.record_prefetch_failure(partial_bytes)
                            break
                        prefetcher.record_prefetch_success(bytes_added)
                        pf_bytes += bytes_added
                        pf_time += budget_s
                    else:
                        prefetcher.record_prefetch_failure(chunk)
                    budget_s = 0

    decode_rows.append({
        "Layer": exec_idx + 1, 
        "Name": layer_name, 
        "Placement": placement[exec_idx],
        "Served_From": " + ".join(served_from_parts), 
        "Bytes": sz,
        "Compute_s": comp_time, 
        "Mem_s": mem_time, 
        "Layer_Time_s": layer_time,
        "Prefetch_Bytes": pf_bytes, 
        "Prefetch_Time_s": pf_time,
        "Prefetch_Lookahead": prefetcher.prefetch_lookahead,
        "Prefetch_Chunk_MB": prefetcher.prefetch_gran_bytes / 1e6,
        "DeviceDRAM_Cache_GB": round(pool.used / GiB, 3),
    })

    if (exec_idx + 1) % 5 == 0:
        prefetcher.adapt_lookahead_depth(exec_idx + 1)
        prefetcher.adapt_chunk_size(exec_idx + 1)

decode_df = pd.DataFrame(decode_rows)
print(decode_df.to_string())
print(f"\nSingle-token decode latency: {per_token_latency:.6f}s")
print(f"Decode throughput: {1.0 / per_token_latency:.6f} tokens/sec")

# ==============================
# Summary
# ==============================
total_time_s = cold_load_s + prefill_latency + (TOKENS * per_token_latency)
total_tokens_processed = PREFILL_TOKENS + TOKENS

def fmt_bytes(n): return f"{n / GiB:.3f} GiB"
model_dtype_bits = int(BYTES_PER_PARAM * 8)
host_bytes = sum(layer_total_size[layers[i]["name"]] for i,p in enumerate(placement) if p == PL_HOST_DRAM)
ssd_bytes = sum(layer_total_size[layers[i]["name"]] for i,p in enumerate(placement) if p == PL_CXL_DEV_NAND)

print("\n" + "="*100)
print("ADAPTIVE PREFETCHER TUNING HISTORY")
print("="*100)
if prefetcher.adapt_history:
    for msg in prefetcher.adapt_history:
        print(f"  {msg}")
else:
    print("  No adaptations needed (metrics stable)")

print("\n" + "="*80)
print("Summary (Adaptive Prefetch + Standard Prefill)")
print("="*80)
print(f"\nPhase Breakdown:")
print(f"  Cold load (SSD→memory): {cold_load_s:.6f}s")
print(f"  Prefill ({PREFILL_TOKENS} tokens): {prefill_latency:.6f}s")
print(f"  Decode ({TOKENS} tokens): {TOKENS * per_token_latency:.6f}s ({per_token_latency:.6f}s per token)")
print(f"  TOTAL time: {total_time_s:.6f}s")
print(f"\nOverall throughput: {total_tokens_processed / total_time_s:.3f} tokens/sec")

print("\nModel Size:")
print(f"  Dtype: FP{model_dtype_bits} ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")
print(f"  Total KV cache size: {total_kv_cache_bytes:,} bytes ({fmt_bytes(total_kv_cache_bytes)})")
print(f"  Per-token KV cache update: {total_kv_cache_increment:,} bytes ({fmt_bytes(total_kv_cache_increment)})")

print("\nPlacement Breakdown:")
print(f"  Host DRAM (Pinned): {host_bytes:,} ({fmt_bytes(host_bytes)})")
print(f"  CXL Device NAND (Cold): {ssd_bytes:,} ({fmt_bytes(ssd_bytes)})")
print(f"  Dynamic CXL DRAM Pool Capacity: {dyn_cxl_capacity/GiB:.3f} GiB")

print("\nDecode Phase Traffic:")
print(f"  From CXL Device DRAM (Hits): {cxl_hit_bytes_cum:,} bytes")
print(f"  From CXL Device NAND (Misses): {cxl_miss_bytes_cum:,} bytes")
print(f"  Total Prefetched Bytes: {decode_df['Prefetch_Bytes'].sum():,} bytes")

print("\nAdaptive Prefetcher Metrics:")
print(f"  Final Prefetch Efficiency: {prefetcher.get_prefetch_efficiency():.1%}")
print(f"  Final Cache Hit Rate: {prefetcher.get_cache_hit_rate():.1%}")
print(f"  Final Lookahead Depth: {prefetcher.prefetch_lookahead}")
print(f"  Final Chunk Size: {prefetcher.prefetch_gran_bytes/1e6:.0f}MB")
print(f"  Prefetch Successes: {prefetcher.prefetch_successes:,} bytes")
print(f"  Prefetch Failures: {prefetcher.prefetch_failures:,} bytes")

# Save CSVs
prefill_df.to_csv("sim_adaptive_prefill.csv", index=False)
decode_df.to_csv("sim_adaptive_decode.csv", index=False)





