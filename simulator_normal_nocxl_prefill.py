# simulator_normal_nocxl_prefill.py
# ------------------------------------------------------------
# SIMULATOR: Baseline Performance (No CXL, Host SSD Swapping with Manual Offloading)
#            NOW WITH STANDARD PREFILL MODELING
#
# DESCRIPTION:
# Models realistic LLM inference workflow:
# 1. Cold load: Load model from SSD into memory
# 2. Prefill: Process 512-token prompt in one batch pass
# 3. Decode: Generate 16 tokens autoregressively (one at a time)
# ------------------------------------------------------------
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

MiB = 1024 ** 2

# Labels
PL_HOST_DRAM = "Host DRAM"
PL_HOST_SSD  = "Host SSD (NVMe)"

# Prefill configuration
PREFILL_TOKENS = 512  # Prompt length

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
def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

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
        kv_cache_increment[L["name"]] = 2 * kv_heads * head_dim * 1 * BYTES_PER_PARAM
        total_kv_cache_increment += kv_cache_increment[L["name"]]
    else:
        kv_cache_increment[L["name"]] = 0

# ==============================
# Placement (Host DRAM + Host SSD)
# ==============================
placement = [PL_HOST_SSD] * len(layers)
host_free = host_dram_capacity_bytes

# Place hot layers first
for n in HOT_LAYERS_BY_NAME:
    idx = name_to_idx.get(n)
    if idx is not None:
        layer_kv_cache_size = kv_cache_increment[layers[idx]["name"]] * sequence_length
        sz = layers[idx]["bytes"] + layer_kv_cache_size
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM
            host_free -= sz

# Fill remaining DRAM with earliest decoder blocks
num_blocks = sum(1 for L in layers if L["kind"] == "DecoderBlock")
first_decoder_idx = next((i for i, L in enumerate(layers) if L["kind"] == "DecoderBlock"), -1)

if first_decoder_idx != -1:
    for i in range(first_decoder_idx, first_decoder_idx + num_blocks):
        if placement[i] == PL_HOST_DRAM:
            continue
        layer_kv_cache_size = kv_cache_increment[layers[i]["name"]] * sequence_length
        sz = layers[i]["bytes"] + layer_kv_cache_size
        if sz <= host_free:
            placement[i] = PL_HOST_DRAM
            host_free -= sz
        else:
            break

# ==============================
# Model totals
# ==============================
total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"] for L in layers)
total_kv_cache_bytes = sum(L.get("kv_cache_bytes", 0) for L in layers)
model_dtype_bits = int(BYTES_PER_PARAM * 8)

model_fits_in_dram = (total_model_bytes + total_kv_cache_bytes) <= host_dram_capacity_bytes

host_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) 
                 for i in range(len(layers)) if placement[i] == PL_HOST_DRAM)
ssd_bytes = sum(layers[i]["bytes"] + (kv_cache_increment[layers[i]["name"]] * sequence_length) 
                for i in range(len(layers)) if placement[i] == PL_HOST_SSD)

# ==============================
# Phase 1: Cold Load
# ==============================
cold_load_s = ssd_cold_time_s(total_model_bytes)

# ==============================
# Phase 2: PREFILL (CORRECTED FLOPS)
# ==============================
print("\n" + "="*80)
print(f"PREFILL PHASE ({PREFILL_TOKENS} tokens, standard batch processing)")
print("="*80)

# Empirical multiplier: prefill is ~10-20× decode FLOPs per layer
# Conservative estimate for CPU inference without FlashAttention
# Empirical multiplier: prefill is ~10-20× decode FLOPs per layer
# Conservative estimate for CPU inference without FlashAttention
# Based on: Attention scales O(seq²) but optimized kernels amortize cost
# Reference: FlexGen (Chen et al., 2023), LLM-in-a-Flash (Alizadeh et al., 2023)
PREFILL_FLOP_MULTIPLIER = 15.0

prefill_latency = 0.0
prefill_rows = []

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement[exec_idx]
    
    # Memory: Load parameters ONLY (KV writes overlap with compute)
    param_bytes = L["bytes"]
    
    # CORRECTED: Prefill FLOPs are ~15× decode, not 512×
    decode_flops = L["flops"]
    prefill_flops = decode_flops * PREFILL_FLOP_MULTIPLIER
    comp_s = compute_time_s(prefill_flops)
    
    # Memory timing (parameters only)
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(param_bytes)
        served_from = PL_HOST_DRAM
    else:
        mem_time = ssd_sequential_time_s(param_bytes)
        served_from = "Host SSD (NVMe, sequential)"
    
    # Layer time = max(compute, memory_read)
    layer_time = max(comp_s, mem_time)
    prefill_latency += layer_time
    
    # Track KV bytes for reporting (overlapped)
    kv_write_bytes = kv_cache_increment[L["name"]] * PREFILL_TOKENS
    
    prefill_rows.append({
        "Layer": exec_idx + 1,
        "Name": L["name"],
        "Kind": L["kind"],
        "Placement": place,
        "Served_From": served_from,
        "Param_Bytes": param_bytes,
        "KV_Write_Bytes": kv_write_bytes,
        "Compute_s": comp_s,
        "Mem_s": mem_time,
        "Layer_Time_s": layer_time
    })

prefill_df = pd.DataFrame(prefill_rows)  # FIX: Add this line
print(prefill_df.to_string())
print(f"\nPrefill total time: {prefill_latency:.6f}s")
print(f"Prefill throughput: {PREFILL_TOKENS / prefill_latency:.3f} tokens/sec")
print(f"Total KV cache written: {sum(r['KV_Write_Bytes'] for r in prefill_rows):,} bytes (overlapped)")


# ==============================
# Phase 3: DECODE (16 tokens, autoregressive)
# ==============================
print("\n" + "="*80)
print(f"DECODE PHASE ({TOKENS} tokens, autoregressive generation)")
print("="*80)

decode_rows = []
per_token_latency = 0.0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement[exec_idx]
    
    # Memory: Load parameters + READ 512-token KV cache + WRITE 1 new token
    param_bytes = L["bytes"]
    kv_read_bytes = kv_cache_increment[L["name"]] * PREFILL_TOKENS  # Read existing cache
    kv_write_bytes = kv_cache_increment[L["name"]]  # Write 1 new token
    total_mem_bytes = param_bytes + kv_read_bytes + kv_write_bytes
    
    # Compute: Process ONE token (standard decode FLOPs)
    comp_s = compute_time_s(L["flops"])
    
    # Memory timing
    if place == PL_HOST_DRAM:
        mem_time = dram_time_s(total_mem_bytes)
        served_from = PL_HOST_DRAM
    else:
        mem_time = ssd_sequential_time_s(total_mem_bytes)
        served_from = "Host SSD (NVMe, sequential)"
    
    layer_time = max(comp_s, mem_time)
    per_token_latency += layer_time
    
    decode_rows.append({
        "Layer": exec_idx + 1,
        "Name": L["name"],
        "Kind": L["kind"],
        "Placement": place,
        "Served_From": served_from,
        "Bytes": total_mem_bytes,
        "Compute_s": comp_s,
        "Mem_s": mem_time,
        "Layer_Time_s": layer_time
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
print(f"Model + KV Cache fits in DRAM: {model_fits_in_dram}")
print(f"\nPhase Breakdown:")
print(f"  Cold load (SSD→DRAM): {cold_load_s:.6f}s")
print(f"  Prefill ({PREFILL_TOKENS} tokens): {prefill_latency:.6f}s")
print(f"  Decode ({TOKENS} tokens): {TOKENS * per_token_latency:.6f}s ({per_token_latency:.6f}s per token)")
print(f"  TOTAL time: {total_time_s:.6f}s")
print(f"\nOverall throughput: {total_tokens_processed / total_time_s:.3f} tokens/sec")

print("\nModel Size:")
print(f"  Total parameters: {total_params:,} ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits} ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total parameter size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")
print(f"  Total KV cache size: {total_kv_cache_bytes:,} bytes ({fmt_bytes(total_kv_cache_bytes)})")
print(f"  Per-token KV cache update: {total_kv_cache_increment:,} bytes ({fmt_bytes(total_kv_cache_increment)})")

print("\nPlacement Breakdown:")
print(f"  Host DRAM: {host_bytes:,} ({fmt_bytes(host_bytes)})")
print(f"  Host SSD: {ssd_bytes:,} ({fmt_bytes(ssd_bytes)})")

print("\nCapacities:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  DRAM-resident layers: {[i+1 for i in range(len(layers)) if placement[i]==PL_HOST_DRAM]}")
print(f"  SSD-resident layers: {[i+1 for i in range(len(layers)) if placement[i]==PL_HOST_SSD]}")

# Save CSVs
prefill_df.to_csv("sim_baseline_prefill.csv", index=False)
decode_df.to_csv("sim_baseline_decode.csv", index=False)





