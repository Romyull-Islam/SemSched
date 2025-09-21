# filename: simulator_tired_update.py
# Simulation of transformer inference with tiered memory (Host DRAM + CXL Device DRAM + CXL Device NAND)
# Sequential access pattern with CXL device DRAM caching and adaptive prefetching from NAND to DRAM

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
# Prefetch knobs
# ==============================
prefetch_lookahead = 5
prefetch_gran_bytes = 4 * 1024 * 1024

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
# Build layers & Placement
# ==============================
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
cxl_free  = cxl_dev_dram_capacity_bytes

hot_indices = [name_to_idx.get("final_norm"), name_to_idx.get("lm_head")]
for idx in hot_indices:
    if idx is not None:
        # Include KV cache for placement
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
# Execute with Adaptive Prefetch
# ==============================
rows = []
per_token_latency = 0.0
cxl_hit_bytes_cum, cxl_miss_bytes_cum = 0, 0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    # Include KV cache increment for per-token transfer
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
    else: # PL_CXL_DEV_NAND
        staged_b = pool.bytes_cached(exec_idx)
        if staged_b > 0:
            mem_time += cxl_time_s(staged_b)
            cxl_hit_bytes_cum += staged_b
            served_from_parts.append(f"CXL DRAM (cache {staged_b/1e6:.1f}MB)")
        rem_b = sz - staged_b
        if rem_b > 0:
            mem_time += nand_time_s(rem_b)
            cxl_miss_bytes_cum += rem_b
            pool.add_bytes(exec_idx, rem_b)
            served_from_parts.append(f"CXL NAND ({rem_b/1e6:.1f}MB)")

    layer_time = max(comp_time, mem_time)
    per_token_latency += layer_time

    pf_bytes, pf_time, pf_targets = 0, 0.0, []
    budget_s = layer_time
    if budget_s > 0:
        for k in range(1, prefetch_lookahead + 1):
            if budget_s <= 0: break
            next_idx = exec_idx + k
            if next_idx >= len(layers) or placement[next_idx] != PL_CXL_DEV_NAND: continue
            
            # Include KV cache for prefetch
            need_b = (layers[next_idx]["bytes"] + kv_cache_increment[layers[next_idx]["name"]]) - pool.bytes_cached(next_idx)
            if need_b <= 0: continue
            pf_targets.append(next_idx + 1)
            
            while need_b > 0 and budget_s > 0:
                chunk = min(prefetch_gran_bytes, need_b)
                t_chunk = nand_time_s(chunk)
                
                if t_chunk <= budget_s:
                    bytes_added = pool.add_bytes(next_idx, chunk)
                    if bytes_added == 0: break
                    need_b -= bytes_added; pf_bytes += bytes_added
                    pf_time += t_chunk; budget_s -= t_chunk
                else:
                    partial_bytes = max(0, int(chunk * (budget_s / t_chunk)))
                    if partial_bytes > 0:
                        bytes_added = pool.add_bytes(next_idx, partial_bytes)
                        pf_bytes += bytes_added; pf_time += budget_s
                    budget_s = 0

    rows.append({
        "Layer": exec_idx + 1, "Name": L["name"], "Placement": placement[exec_idx],
        "Served_From": " + ".join(served_from_parts), "Bytes": sz,
        "Compute_s_all_cores": comp_time, "Mem_s_total": mem_time, "Layer_Time_s": layer_time,
        "CXL_Hit_Bytes_cum": cxl_hit_bytes_cum, "CXL_Miss_Bytes_cum": cxl_miss_bytes_cum,
        "Prefetch_Source": "NAND->Device DRAM" if pf_bytes > 0 else "None",
        "Prefetch_Next_Layer": ",".join(map(str, sorted(list(set(pf_targets))))) if pf_targets else "None",
        "Prefetch_Bytes": pf_bytes, "Prefetch_Time_s": pf_time,
        "DeviceDRAM_Cache_Used_GB": round(pool.used / GiB, 3),
    })

# ==============================
# Final Summary
# ==============================
df = pd.DataFrame(rows)
df.to_csv("sim_adaptive_prefetch.csv", index=False)
print(df.to_string())

total_model_bytes = sum(L["bytes"] for L in layers)
total_kv_cache_bytes = sum(L.get("kv_cache_bytes", 0) for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

throughput_tokens_per_sec = 1.0 / per_token_latency if per_token_latency > 0 else 0.0
total_time_s_all_tokens = cold_load_s + (TOKENS * per_token_latency)

model_dtype_bits = int(BYTES_PER_PARAM * 8)
host_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i,p in enumerate(placement) if p == PL_HOST_DRAM)
cxl_bytes  = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i,p in enumerate(placement) if p == PL_CXL_DEV_DRAM)
ssd_bytes  = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) for i,p in enumerate(placement) if p == PL_CXL_DEV_NAND)
def fmt_bytes(n): return f"{n / GiB:.3f} GiB"

print(f"\nSummary (Sequential Execution with Adaptive Prefetching):")
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