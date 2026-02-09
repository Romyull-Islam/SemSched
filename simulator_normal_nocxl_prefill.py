# simulator_normal_nocxl_prefill.py
import math
import pandas as pd

from tiers import (
    GiB, IO_CHUNK_BYTES, Tier, HOST_DRAM, transfer_time_s,
    NVME_STREAM_BW, NVME_STREAM_LAT_S,
    NVME_THRASH_BW, NVME_THRASH_LAT_S, NVME_FAULT_OVERHEAD
)
from model_cfg import build_layers, BYTES_PER_PARAM, HOT_LAYERS_BY_NAME
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes
)

# Labels
PL_HOST_DRAM = "Host DRAM"
PL_HOST_SSD  = "Host SSD (NVMe)"

# Prefill configuration
PREFILL_TOKENS = 512
PREFILL_FLOP_MULTIPLIER = 15.0

# ==============================
# Helpers
# ==============================
def compute_time_s(flops):
    if flops <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):  return transfer_time_s(n, HOST_DRAM)

def ssd_cold_time_s(n):
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

def ssd_sequential_time_s(n):
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

def fmt_bytes(n): return f"{n / (1024**3):.3f} GiB"

# ==============================
# Build layers
# ==============================
sequence_length = PREFILL_TOKENS
layers = build_layers(sequence_length=sequence_length)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

# ==============================
# KV Cache Sizing
# ==============================
kv_cache_increment = {}
total_kv_cache_increment = 0
for L in layers:
    if L["kind"] == "DecoderBlock":
        head_dim = L.get("head_dim", 128)
        kv_heads = L.get("kv_heads", 40 if len(layers) > 35 else 8)
        kv_inc = 2 * kv_heads * head_dim * 1 * BYTES_PER_PARAM
        kv_cache_increment[L["name"]] = kv_inc
        total_kv_cache_increment += kv_inc
    else:
        kv_cache_increment[L["name"]] = 0

# ==============================
# Placement (Host DRAM + Host SSD)
# ==============================
placement = [PL_HOST_SSD] * len(layers)
host_free = host_dram_capacity_bytes

for n in HOT_LAYERS_BY_NAME:
    idx = name_to_idx.get(n)
    if idx is not None:
        sz = layers[idx]["bytes"] + kv_cache_increment[layers[idx]["name"]] * sequence_length
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= sz

for i, L in enumerate(layers):
    if placement[i] == PL_HOST_DRAM:
        continue
    sz = L["bytes"] + kv_cache_increment[L["name"]] * sequence_length
    if sz <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= sz
    else:
        break

# ==============================
# Phase 1: Cold Load
# ==============================
total_model_bytes = sum(L["bytes"] for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

# ==============================
# Phase 2: PREFILL
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
    decode_flops = L["flops"]
    prefill_flops = decode_flops * PREFILL_FLOP_MULTIPLIER
    comp_s = compute_time_s(prefill_flops)
    
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(param_bytes)
        served_from = PL_HOST_DRAM
    else:
        mem_time = ssd_sequential_time_s(param_bytes)
        served_from = "Host SSD (NVMe, sequential)"
    
    layer_time = max(comp_s, mem_time)
    prefill_latency += layer_time
    
    kv_write_bytes = kv_cache_increment[L["name"]] * PREFILL_TOKENS
    
    prefill_rows.append({
        "Layer": exec_idx + 1, "Name": L["name"], "Kind": L["kind"],
        "Placement": place, "Served_From": served_from,
        "Param_Bytes": param_bytes, "KV_Write_Bytes": kv_write_bytes,
        "Compute_s": comp_s, "Mem_s": mem_time, "Layer_Time_s": layer_time
    })

prefill_df = pd.DataFrame(prefill_rows)
print(prefill_df.to_string())
print(f"\nPrefill total time: {prefill_latency:.6f}s")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f} tokens/sec")

# ==============================
# Phase 3: DECODE
# ==============================
print("\n" + "="*80)
print(f"DECODE PHASE ({TOKENS} tokens, autoregressive generation)")
print("="*80)

decode_rows = []
per_token_latency = 0.0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement[exec_idx]
    
    # Memory: Load parameters + READ KV cache + WRITE 1 new token
    param_bytes = L["bytes"]
    kv_read_bytes = kv_cache_increment[L["name"]] * PREFILL_TOKENS
    kv_write_bytes = kv_cache_increment[L["name"]] 
    total_mem_bytes = param_bytes + kv_read_bytes + kv_write_bytes
    
    comp_s = compute_time_s(L["flops"])
    
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(total_mem_bytes)
        served_from = PL_HOST_DRAM
    else:
        mem_time = ssd_sequential_time_s(total_mem_bytes)
        served_from = "Host SSD (NVMe, sequential)"
    
    layer_time = max(comp_s, mem_time)
    per_token_latency += layer_time
    
    decode_rows.append({
        "Layer": exec_idx + 1, "Name": L["name"], "Kind": L["kind"],
        "Placement": place, "Served_From": served_from,
        "Bytes": total_mem_bytes, "Compute_s": comp_s,
        "Mem_s": mem_time, "Layer_Time_s": layer_time
    })

decode_df = pd.DataFrame(decode_rows)
print(decode_df.to_string())
print(f"\nSingle-token decode latency: {per_token_latency:.6f}s")
print(f"Decode throughput: {1.0 / per_token_latency:.6f} tokens/sec")

# ==============================
# Summary
# ==============================
total_time_s = cold_load_s + prefill_latency + (TOKENS * per_token_latency)
total_tokens_processed = PREFILL_TOKENS + TOKENS

print("\n" + "="*80)
print("Summary (Baseline with Host Swap + Standard Prefill)")
print("="*80)
print(f"\nPhase Breakdown:")
print(f"  Cold load (SSD->DRAM): {cold_load_s:.6f}s")
print(f"  Prefill ({PREFILL_TOKENS} tokens): {prefill_latency:.6f}s")
print(f"  Decode ({TOKENS} tokens): {TOKENS * per_token_latency:.6f}s")
print(f"  TOTAL time: {total_time_s:.6f}s")
print(f"\nOverall throughput: {total_tokens_processed / total_time_s:.3f} tokens/sec")

# Save CSVs
prefill_df.to_csv("sim_baseline_prefill.csv", index=False)
decode_df.to_csv("sim_baseline_decode.csv", index=False)