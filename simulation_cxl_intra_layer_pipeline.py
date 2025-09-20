# filename: simulation_cxl_intra_layer_pipeline.py
# ------------------------------------------------------------
# SIMULATOR: Two-Stage Pipeline with Intra-Layer Tiling (Double-Buffering)
#
# DESCRIPTION:
# This simulation models a two-stage pipeline for LLM inference, specifically
# designed to mitigate CXL NAND latency through advanced I/O hiding techniques.
#
# STAGE 1 (HOST):
# - Executes all model layers that fit entirely within the fast Host DRAM.
# - The number of cores for this stage is determined by the adaptive planner.
#
# STAGE 2 (CXL DEVICE):
# - Executes all remaining layers that are stored on the CXL device.
# - For layers already in the CXL DRAM cache, it performs a fast memory read.
# - For layers on CXL NAND, it uses INTRA-LAYER TILING (Double-Buffering):
#   - A large layer is broken into smaller tiles (e.g., 4 MiB).
#   - While the CPU computes on the current tile, the system fetches the *next*
#     tile from NAND in parallel, effectively hiding most of the I/O latency.
#
# ADAPTIVE CORE SPLITTING:
# - Before execution, an "Adaptive Planner" estimates the workload of both stages
#   and splits the total CPU cores between them to achieve the most balanced
#   pipeline and maximize throughput.
# ------------------------------------------------------------
import math
import pandas as pd
from collections import OrderedDict

from tiers import (
    HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, IO_CHUNK_BYTES, transfer_time_s
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
PL_CXL_DEVICE_DRAM  = "CXL Device DRAM (cache)"
PL_CXL_DEVICE_NAND  = "CXL Device NAND"

# ---------------------------
# Tunables
# ---------------------------
TILE_BYTES_REQ = 4 * 1024 * 1024  # 4 MiB tile size for double-buffering
DEVICE_DRAM_HINT_PIN_FIRST_K = 4
DEVICE_DRAM_PIN_STRICT       = True


# ---------------------------
# Helpers
# ---------------------------
def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)
def cxlssd_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def fmt_bytes(n): return f"{n/(1024**3):.3f} GiB"

# ---------------------------
# Build model & placement (Global scope)
# ---------------------------
layers = build_layers()
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
# Simple greedy prefix fill for stage split
order_prefix = [0] + list(range(1, sum(1 for L in layers if L["kind"] == "DecoderBlock") + 1))
for idx in order_prefix:
    if placement[idx] is None and layers[idx]["bytes"] <= host_free:
        placement[idx] = PL_HOST_DRAM
        host_free -= layers[idx]["bytes"]
tail_idx = [len(layers)-2, len(layers)-1] # Tail layers can also go in host DRAM but are part of Stage 2
for idx in tail_idx:
    if placement[idx] is None and layers[idx]["bytes"] <= host_free:
        placement[idx] = PL_HOST_DRAM
        host_free -= layers[idx]["bytes"]
for i in range(len(layers)):
    if placement[i] is None:
        placement[i] = PL_CXL_DEVICE_NAND

split = 0
while split < len(layers) and placement[split] == PL_HOST_DRAM:
    split += 1
s1_idx = list(range(0, split))
s2_idx = list(range(split, len(layers)))

# ---------------------------
# Device-DRAM Pool Class
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
        if (self.used_cache + add_b) > self.cap: return
        self.used_cache += add_b
        self.lru[layer_id] = self.lru.get(layer_id, 0) + add_b
        self.lru.move_to_end(layer_id)
    def pin_layer(self, layer_id, size_b):
        needed = max(0, size_b - self.cached_bytes(layer_id))
        self.add_cache_bytes(layer_id, needed)
        if self.cached_bytes(layer_id) >= size_b: self.pinned.add(layer_id)

# ---------------------------
# Simulation Functions
# ---------------------------
def get_stage1_details(cores):
    total = 0.0; rows = []
    for k in s1_idx:
        L = layers[k]
        t = max(compute_time_s(L["flops"], cores), dram_time_s(L["bytes"]))
        total += t
        rows.append({ "Layer": k+1, "Name": L["name"], "Placement": PL_HOST_DRAM, "Stage": 1, "Bytes": L["bytes"], "Stage1_Layer_Time_s": t})
    return total, rows

def get_stage2_details(cores, run_tokens):
    dev_pool = DeviceDRAMPool(cxl_dev_dram_capacity_bytes)
    if DEVICE_DRAM_HINT_PIN_FIRST_K > 0:
        to_pin = [k for k in s2_idx if placement[k] == PL_CXL_DEVICE_NAND][:DEVICE_DRAM_HINT_PIN_FIRST_K]
        for k in to_pin: dev_pool.pin_layer(k, layers[k]["bytes"])

    per_token_s2, rows_token1 = [], []
    acct = {"bytes_devdram_hits": 0, "bytes_nand": 0, "bytes_hostdram_in_s2": 0}

    for t in range(run_tokens):
        token_time = 0.0
        row_builder = []
        for k in s2_idx:
            L, sz, fl, plc = layers[k], layers[k]["bytes"], layers[k]["flops"], placement[k]
            layer_time = 0.0
            served_from = ""

            if plc == PL_HOST_DRAM:
                layer_time = max(compute_time_s(fl, cores), dram_time_s(sz))
                acct["bytes_hostdram_in_s2"] += sz if t==0 else 0
                served_from = "Host DRAM"
            else: # Layer is on CXL Device (NAND or cached in DRAM)
                # Check if the entire layer is already cached
                if dev_pool.cached_bytes(k) >= sz:
                    layer_time = max(compute_time_s(fl, cores), cxl_time_s(sz))
                    acct["bytes_devdram_hits"] += sz
                    dev_pool.add_cache_bytes(k, 0) # Update LRU
                    served_from = "CXL DRAM (cache hit)"
                else:
                    # Implement tile-based double buffering for layers fetched from NAND
                    served_from = "CXL NAND (Tiled)"
                    num_tiles = math.ceil(sz / TILE_BYTES_REQ)
                    flops_per_tile = fl / num_tiles if num_tiles > 0 else 0
                    bytes_per_tile = TILE_BYTES_REQ

                    # Time to fetch and compute the first tile (pipeline fill)
                    t_fetch_1 = cxlssd_time_s(bytes_per_tile)
                    t_compute_1 = compute_time_s(flops_per_tile, cores)
                    layer_time += t_fetch_1 + t_compute_1
                    acct["bytes_nand"] += bytes_per_tile
                    dev_pool.add_cache_bytes(k, bytes_per_tile)

                    # For remaining tiles, overlap fetch and compute
                    for i in range(1, num_tiles):
                        t_fetch_i = cxlssd_time_s(bytes_per_tile)
                        t_compute_i_minus_1 = compute_time_s(flops_per_tile, cores)
                        layer_time += max(t_fetch_i, t_compute_i_minus_1)
                        acct["bytes_nand"] += bytes_per_tile
                        dev_pool.add_cache_bytes(k, bytes_per_tile)

                    # Add compute time for the very last tile
                    layer_time += compute_time_s(flops_per_tile, cores)

            token_time += layer_time
            if t == 0: row_builder.append({"Layer": k+1, "Name": L["name"], "Placement": plc, "Stage": 2, "Bytes": sz, "Served_From": served_from, "Layer_Time_s": layer_time})

        per_token_s2.append(token_time)
        if t == 0: rows_token1 = row_builder

    acct["devdram_cache_used_GB"] = sum(dev_pool.lru.values()) / GiB
    return per_token_s2, rows_token1, acct

def find_optimal_pipeline_config(P):
    print("\nFinding optimal pipeline configuration...")
    best_config = None; min_bottleneck = float('inf')

    def estimate_s1(cores):
        return sum(max(compute_time_s(L["flops"], cores), dram_time_s(L["bytes"])) for i, L in enumerate(layers) if i in s1_idx)

    def estimate_s2_tiled(cores):
        total_s2_time = 0
        for i, L in enumerate(layers):
            if i not in s2_idx: continue

            if placement[i] == PL_HOST_DRAM:
                total_s2_time += max(compute_time_s(L["flops"], cores), dram_time_s(L["bytes"]))
            else: # CXL NAND Layer - estimate tiled execution
                num_tiles = math.ceil(L["bytes"] / TILE_BYTES_REQ)
                if num_tiles == 0: continue

                flops_per_tile = L["flops"] / num_tiles
                t_fetch = cxlssd_time_s(TILE_BYTES_REQ)
                t_compute = compute_time_s(flops_per_tile, cores)

                # Simplified estimation: 1 full fetch + N compute + (N-1) overlaps
                layer_time = t_fetch + (num_tiles * t_compute) + (num_tiles - 1) * max(0, t_fetch - t_compute)
                total_s2_time += layer_time
        return total_s2_time

    for p1 in range(1, P):
        p2 = P - p1
        s1_lat, s2_lat = estimate_s1(p1), estimate_s2_tiled(p2)
        bottleneck = max(s1_lat, s2_lat)
        print(f"  Testing split p1={p1}, p2={p2} -> S1_est={s1_lat:.4f}s, S2_est={s2_lat:.4f}s -> Bottleneck={bottleneck:.4f}s")
        if bottleneck < min_bottleneck:
            min_bottleneck = bottleneck
            best_config = {'p1': p1, 'p2': p2}

    print(f"==> Optimal split found: p1={best_config['p1']}, p2={best_config['p2']}")
    return best_config

# ---------------------------
# Run
# ---------------------------
P = cpu_cores
optimal_config = find_optimal_pipeline_config(P)
p1, p2 = optimal_config['p1'], optimal_config['p2']

S1_time, s1_rows = get_stage1_details(p1)
per_token_s2, s2_rows, acct = get_stage2_details(p2, TOKENS)

S2_first = per_token_s2[0]
S2_steady = (sorted(per_token_s2[1:])[len(per_token_s2[1:])//2] if TOKENS > 1 else S2_first)
bottleneck = max(S1_time, S2_steady)
pipeline_total_time = S1_time + S2_first + (TOKENS - 1) * bottleneck if TOKENS > 1 else S1_time + S2_first
throughput_steady = 1.0 / bottleneck if bottleneck > 0 else 0.0

# ---------------------------
# Output
# ---------------------------
s2_df = pd.DataFrame(s2_rows)
s1_df = pd.DataFrame(s1_rows)
if not s1_df.empty and not s2_df.empty:
    df = pd.concat([s1_df, s2_df], ignore_index=True).sort_values(by="Layer")
elif not s1_df.empty:
    df = s1_df
else:
    df = s2_df

# Reorder columns for clarity
df = df.reindex(columns=['Layer', 'Name', 'Placement', 'Stage', 'Bytes', 'Stage1_Layer_Time_s', 'Layer_Time_s', 'Served_From'])

print(df.to_string())
print(f"\nSummary (Pipeline with Tiling/Double-Buffering):")
print(f"Core split over P={P}: p1={p1} (Stage-1), p2={p2} (Stage-2)")
print(f"Stage-1 per token S1={S1_time:.6f} s")
print(f"Stage-2 per token S2 (token1 cold)={S2_first:.6f} s, S2 (steady median)={S2_steady:.6f} s")
print(f"Steady bottleneck per token = {bottleneck:.6f} s -> throughput ≈ {throughput_steady:.6f} tok/s")
print(f"Total time for T={TOKENS}: {pipeline_total_time:.6f} s")

# Model and placement info
total_model_bytes = sum(L["bytes"] for L in layers)
model_dtype_bits = int(BYTES_PER_PARAM * 8)
print("\nModel Size & Placement:")
print(f"  Dtype: FP{model_dtype_bits}, Cores: {P}, Host DRAM: {host_dram_capacity_bytes/GiB:.1f} GiB")
print(f"  Host DRAM layers: {len(s1_idx)}, CXL/NAND layers: {len(s2_idx)}")
print("\nStage-2 memory service accounting (across all tokens):")
print(f"  From CXL DRAM (hits): {acct['bytes_devdram_hits']:,} bytes")
print(f"  From CXL NAND (misses): {acct['bytes_nand']:,} bytes")
print(f"  Peak devDRAM cache used: {acct['devdram_cache_used_GB']:.3f} GB")