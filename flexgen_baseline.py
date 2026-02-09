# flexgen_baseline.py
import math
import pandas as pd
from tiers import GiB, HOST_DRAM, Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S, transfer_time_s
from model_cfg import build_layers, BYTES_PER_PARAM, HOT_LAYERS_BY_NAME
from sim_cfg import TOKENS, cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency, host_dram_capacity_bytes

PL_HOST_DRAM = "Host DRAM"
PL_HOST_SSD  = "Host SSD (NVMe)"
PREFILL_TOKENS = 512
PREFILL_FLOP_MULTIPLIER = 15.0

def compute_time_s(flops):
    if flops <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):  return transfer_time_s(n, HOST_DRAM)
def ssd_time_s(n):   return transfer_time_s(n, Tier("Host SSD", NVME_STREAM_BW, NVME_STREAM_LAT_S))

# Build Model Layers
sequence_length = PREFILL_TOKENS
layers = build_layers(sequence_length=sequence_length)
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

# Calculate KV Cache increments
kv_cache_increment = {}
for L in layers:
    if L["kind"] == "DecoderBlock":
        head_dim = L.get("head_dim", 128)
        kv_heads = L.get("kv_heads", 8)
        kv_cache_increment[L["name"]] = 2 * kv_heads * head_dim * BYTES_PER_PARAM
    else:
        kv_cache_increment[L["name"]] = 0

# FlexGen Logic: Strict Block-Diagonal Offloading
placement = [PL_HOST_SSD] * len(layers)
host_free = host_dram_capacity_bytes

# 1. Pin Hot Layers (Embed/Head)
for n in HOT_LAYERS_BY_NAME:
    idx = name_to_idx.get(n)
    if idx is not None:
        sz = layers[idx]["bytes"]
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= sz

# 2. Opportunistic Caching (Fill remaining DRAM linearly)
for i, L in enumerate(layers):
    if placement[i] == PL_HOST_DRAM: continue
    sz = L["bytes"] + kv_cache_increment[L["name"]] * sequence_length
    if sz <= host_free:
        placement[i] = PL_HOST_DRAM
        host_free -= sz
    else:
        break # Fill until full, then spill everything else

# Phase 2: PREFILL
prefill_latency = 0.0
for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement[exec_idx]
    comp_s = compute_time_s(L["flops"] * PREFILL_FLOP_MULTIPLIER)
    
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(L["bytes"])
    else:
        mem_time = ssd_time_s(L["bytes"]) # FlexGen demand fetch
    
    # FlexGen overlaps I/O and compute
    prefill_latency += max(comp_s, mem_time)

# Phase 3: DECODE
per_token_latency = 0.0
for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement[exec_idx]
    
    # I/O Demand: Weights + KV Read + KV Write
    # FlexGen loads the full diagonal block
    total_io = L["bytes"] + kv_cache_increment[L["name"]] * (PREFILL_TOKENS + 1)
    
    comp_s = compute_time_s(L["flops"])
    
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(total_io)
    else:
        mem_time = ssd_time_s(total_io) # Offload penalty
    
    # FlexGen Overlap:
    per_token_latency += max(comp_s, mem_time)

print(f"Decode throughput: {1.0 / per_token_latency:.6f}")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f}")