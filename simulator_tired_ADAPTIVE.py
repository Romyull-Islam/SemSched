# filename: simulator_tired_ADAPTIVE_v9.py
# Simulation with TRUE ADAPTIVE PREFETCHING
# ================================================================
# This version includes the "Knapsack" placement strategy (v9):
# 1. This logic tests the user's hypothesis to maximize utilization
#    and performance by prioritizing embed_tokens.
# 2. Priority Order:
#    P1: final_norm (tiny)
#    P2: embed_tokens (critical start-of-loop)
#    P3: decoder_0, decoder_1, ... (compute-heavy layers)
#    P4: lm_head (critical end-of-loop, but lowest priority)
# 3. Packs Host DRAM greedily based on this priority.
# 4. Dedicates 100% of CXL Device DRAM to the dynamic cache pool.
# 5. NEW: Post-greedy swap — if (last pinned decoder size + remaining host_free) >= lm_head,
#    then evict the last pinned decoder and pin lm_head.
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
PL_CXL_DEV_DRAM = "CXL Device DRAM"  # Defined for logging, but NOT used for placement
PL_CXL_DEV_NAND = "CXL Device NAND"

# ==============================
# ADAPTIVE PREFETCHER CLASS
# (Unchanged)
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
        """
        urgency = nand_remaining / max(1, total_layers)
        adaptive_budget = base_budget * (0.5 + 0.5 * urgency)
        return adaptive_budget

    def get_aggressiveness(self, layer_idx, total_layers):
        """
        ADAPTATION 4: Layer-position-aware aggressiveness
        """
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
# (Unchanged)
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
# (Unchanged)
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
            lid, sz = self.lru.popitem(last=False) # Evict oldest
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
# (Unchanged)
# ==============================
layers = build_layers(sequence_length=512)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

sequence_length = 512
kv_cache_increment = {}
total_kv_cache_increment = 0
layer_full_kv_size = {} # Store full KV cache size per layer
layer_total_size = {}   # Store params + full KV cache

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

# ========================================================
# NEW EFFICIENT PLACEMENT LOGIC (v9 - Knapsack, User Priority)
# with post-greedy lm_head swap
# ========================================================
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
unpinned_layers = [] # Keep track of layers not yet pinned

print("--- Starting Efficient Placement (v9 - Knapsack, User Priority) ---")
print(f"Host DRAM Capacity: {host_free / GiB:.2f} GiB")

# --- 1. Define Priority Lists ---
# Create (priority, size, index, name) tuples
# Lower priority number = HIGHER priority
layers_to_place = []
for i, L in enumerate(layers):
    size = layer_total_size[L["name"]]
    # Priority mapping
    if L["name"] == "final_norm":
        priority = 0 # Tiny, always pin
    elif L["name"] == "embed_tokens":
        priority = 10 # Prio 10
    elif L["kind"] == "DecoderBlock":
        priority = 20 + i # Prio 20, 21, 22...
    elif L["name"] == "lm_head":
        priority = 100 # Lowest priority in greedy
    else:
        priority = 1000 # Should not happen
    layers_to_place.append((priority, size, i, L["name"]))

# Sort by priority (lowest number first)
layers_to_place.sort()

# --- 2. Pin Priority Layers to Host DRAM (Greedy Knapsack) ---
print(f"Packing Host DRAM...")
last_pinned_decoder = None  # (idx, size) of most recently pinned DecoderBlock
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
        # Layer is too big for remaining space
        unpinned_layers.append((size, idx, name))

# --- 2b. Post-greedy swap to include lm_head if near-miss ---
# Condition: lm_head not pinned, doesn't currently fit, but would fit if last pinned decoder is evicted.
if lm_head_idx is not None and placement[lm_head_idx] is None and last_pinned_decoder is not None:
    dec_idx, dec_size = last_pinned_decoder
    # Use a tiny epsilon to avoid unit rounding issues
    epsilon = int(0.01 * GiB)  # 10 MiB slack
    if host_free < lm_head_size and (host_free + dec_size + epsilon) >= lm_head_size:
        # Swap: evict last decoder, pin lm_head
        placement[dec_idx] = None
        host_free += dec_size
        placement[lm_head_idx] = PL_HOST_DRAM
        host_free -= lm_head_size
        # Make the evicted decoder available for backfill
        unpinned_layers.append((dec_size, dec_idx, layers[dec_idx]["name"]))
        print(f"  Swapped out {layers[dec_idx]['name']} to pin lm_head ({lm_head_size / GiB:.2f} GiB). Host free: {host_free / GiB:.2f} GiB")

# --- 3. "Backfill" Wasted Space ---
# Try to fill the remaining space with any layers that fit,
# sorting by smallest size first to maximize utilization.
unpinned_layers.sort() # Sort by size (smallest first)
if host_free > 0 and unpinned_layers:
    print(f"Backfilling {host_free / GiB:.2f} GiB of remaining Host DRAM...")
    for size, idx, name in unpinned_layers:
        if placement[idx] is None and size <= host_free: # Check if not already placed
            placement[idx] = PL_HOST_DRAM
            host_free -= size
            print(f"  Backfilled {name} ({size / GiB:.2f} GiB). Host free: {host_free / GiB:.2f} GiB")

print(f"Host DRAM placement complete. Wasted space: {host_free / GiB:.2f} GiB")

# --- 4. CXL DRAM is 100% DYNAMIC CACHE ---
dyn_cxl_capacity = cxl_dev_dram_capacity_bytes  # Use ALL CXL DRAM as cache
pool = DeviceDramPool(dyn_cxl_capacity)
print(f"CXL DRAM Pool (Dynamic Cache) size: {dyn_cxl_capacity / GiB:.2f} GiB")

# --- 5. All remaining layers go to NAND ---
for idx in range(len(layers)):
    if placement[idx] is None:
        placement[idx] = PL_CXL_DEV_NAND
print("--- Placement Complete ---")

# --- Calculate pinned CXL bytes (should be 0 now, for logging) ---
pinned_cxl_bytes = 0.0
for i, p in enumerate(placement):
    if p == PL_CXL_DEV_DRAM: # This should not happen
        L = layers[i]
        pinned_cxl_bytes += L["bytes"] + layer_full_kv_size[L["name"]]

# ==============================
# Initialize Adaptive Prefetcher
# (Unchanged)
# ==============================
prefetcher = AdaptivePrefetcher(initial_lookahead=8, initial_chunk_mb=4)
nand_layer_indices = [i for i in range(len(layers)) if placement[i] == PL_CXL_DEV_NAND]

# ==============================
# Execute with Adaptive Prefetch
# (Unchanged)
# ==============================
rows = []
per_token_latency = 0.0
cxl_hit_bytes_cum, cxl_miss_bytes_cum = 0, 0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    layer_name = L["name"]
    # Runtime access size = params + ONE token's worth of KV cache
    sz = L["bytes"] + kv_cache_increment[layer_name]
    comp_time = compute_time_s(L["flops"])
    mem_time, served_from_parts = 0.0, []

    if placement[exec_idx] == PL_HOST_DRAM:
        mem_time = dram_time_s(sz)
        served_from_parts.append(PL_HOST_DRAM)
    else:  # PL_CXL_DEV_NAND
        # Full size of layer (params + full KV) is needed for caching
        full_layer_size = layer_total_size[layer_name]
        staged_b = pool.bytes_cached(exec_idx)

        # Check if the entire layer is cached
        if staged_b >= full_layer_size:
            # CACHE HIT (Full Layer)
            mem_time += cxl_time_s(sz)
            cxl_hit_bytes_cum += sz
            served_from_parts.append(f"CXL DRAM (cache {staged_b/1e6:.1f}MB)")
            prefetcher.record_cache_hit(sz)
            pool.add_bytes(exec_idx, 0) # Move to end of LRU
        else:
            # CACHE MISS (or partial hit)
            # 1. Read from cache (if any)
            if staged_b > 0:
                read_from_cache = min(staged_b, sz)
                mem_time += cxl_time_s(read_from_cache)
                cxl_hit_bytes_cum += read_from_cache
                served_from_parts.append(f"CXL DRAM (cache {read_from_cache/1e6:.1f}MB)")
                prefetcher.record_cache_hit(read_from_cache)
            else:
                read_from_cache = 0
            # 2. Stall to read remaining bytes from NAND
            rem_b = sz - read_from_cache
            if rem_b > 0:
                mem_time += nand_time_s(rem_b) # Stall
                cxl_miss_bytes_cum += rem_b
                served_from_parts.append(f"CXL NAND ({rem_b/1e6:.1f}MB)")
                prefetcher.record_cache_miss(rem_b)
            # 3. Load the entire layer into cache on miss
            pool.add_bytes(exec_idx, full_layer_size)

    layer_time = max(comp_time, mem_time)
    per_token_latency += layer_time

    # Count remaining NAND layers
    nand_remaining = sum(1 for i in range(exec_idx + 1, len(layers)) if placement[i] == PL_CXL_DEV_NAND)

    # ADAPTIVE PREFETCH with dynamic parameters
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
                        break # Cache is full
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
# (Unchanged)
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
total_kv_cache_bytes = sum(layer_full_kv_size[L["name"]] for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

throughput_tokens_per_sec = 1.0 / per_token_latency if per_token_latency > 0 else 0.0
total_time_s_all_tokens = cold_load_s + (TOKENS * per_token_latency)

model_dtype_bits = int(BYTES_PER_PARAM * 8)
host_bytes = sum(layer_total_size[layers[i]["name"]] for i,p in enumerate(placement) if p == PL_HOST_DRAM)
cxl_bytes  = pinned_cxl_bytes # Should be 0
ssd_bytes  = sum(layer_total_size[layers[i]["name"]] for i,p in enumerate(placement) if p == PL_CXL_DEV_NAND)
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
print(f"  Host DRAM (Pinned): {host_bytes:,} ({fmt_bytes(host_bytes)})")
print(f"  CXL Device DRAM (Pinned): {cxl_bytes:,} ({fmt_bytes(cxl_bytes)})")
print(f"  CXL Device NAND (Cold): {ssd_bytes:,} ({fmt_bytes(ssd_bytes)})")
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
