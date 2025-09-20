# filename: simulation_baseline_host_swap.py
# ------------------------------------------------------------
# SIMULATOR: Baseline Performance (No CXL, Host SSD Swapping)
#
# DESCRIPTION:
# This simulation models the baseline scenario where a model is too large
# for the available Host DRAM. It relies on the operating system's virtual
# memory to swap pages to a standard Host NVMe SSD, resulting in significant
# performance degradation known as "page-fault thrashing."
#
# All system parameters (DRAM size, CPU cores) are correctly read from
# the central sim_cfg.py file for fair comparison.
# ------------------------------------------------------------
import math
import pandas as pd

from tiers import (
    GiB, IO_CHUNK_BYTES, Tier, HOST_DRAM, transfer_time_s,
    NVME_STREAM_BW, NVME_STREAM_LAT_S,
    NVME_THRASH_BW, NVME_THRASH_LAT_S, NVME_FAULT_OVERHEAD
)
from model_cfg import build_layers, HOT_LAYERS_BY_NAME, BYTES_PER_PARAM
# --- MODIFIED: Import system configuration from the central file ---
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes
)

MiB = 1024 ** 2

# Labels
PL_HOST_DRAM = "Host DRAM"
PL_HOST_SSD  = "Host SSD (NVMe)"

# Page-fault readahead window (OS-level knob; sim-specific)
SSD_PF_READAHEAD_BYTES   = 1 * MiB

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

def ssd_token_time_s(n):
    if n <= 0: return 0.0
    windows = math.ceil(n / SSD_PF_READAHEAD_BYTES)
    transfer = n / NVME_THRASH_BW
    latency  = windows * (NVME_THRASH_LAT_S + NVME_FAULT_OVERHEAD)
    return transfer + latency

def fmt_bytes(n): return f"{n / (1024**3):.3f} GiB"
def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

# ==============================
# Build layers (honors QUANT)
# ==============================
layers = build_layers()
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

# ==============================
# Placement (Host DRAM + Host SSD only)
# ==============================
placement = [PL_HOST_SSD] * len(layers)
host_free = host_dram_capacity_bytes

for n in ("lm_head", "final_norm", "embed_tokens"):
    idx = name_to_idx.get(n)
    if idx is not None:
        sz = layers[idx]["bytes"]
        if sz <= host_free:
            placement[idx] = PL_HOST_DRAM; host_free -= sz

num_blocks = sum(1 for L in layers if L["kind"] == "DecoderBlock")
first_decoder_idx = -1
for i, L in enumerate(layers):
    if L["kind"] == "DecoderBlock":
        first_decoder_idx = i
        break

if first_decoder_idx != -1:
    for i in range(first_decoder_idx, first_decoder_idx + num_blocks):
        sz = layers[i]["bytes"]
        if sz <= host_free:
            placement[i] = PL_HOST_DRAM; host_free -= sz
        else:
            break

# Totals & fit check
total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"]  for L in layers)
model_dtype_bits = int(BYTES_PER_PARAM * 8)

host_bytes = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == PL_HOST_DRAM)
ssd_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == PL_HOST_SSD)
model_fits_in_dram = (total_model_bytes <= host_dram_capacity_bytes)

# ==============================
# Execute (sequential; no prefetch)
# ==============================
rows = []
# --- CORRECTED LOGIC: Cold load is for the ENTIRE model, always ---
cold_load_s = ssd_cold_time_s(total_model_bytes)

per_token_latency = 0.0
for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    sz = L["bytes"]; flops = L["flops"]; place = placement[exec_idx]

    comp_s = compute_time_s(flops)
    if model_fits_in_dram:
        mem_dram = dram_time_s(sz); mem_ssd = 0.0
        served_from = PL_HOST_DRAM
        layer_time = max(comp_s, mem_dram)
    else:
        if place == PL_HOST_DRAM:
            mem_dram = dram_time_s(sz); mem_ssd = 0.0
            served_from = PL_HOST_DRAM
            layer_time = max(comp_s, mem_dram)
        else:
            mem_dram = 0.0; mem_ssd = ssd_token_time_s(sz)
            served_from = "Host SSD (NVMe, page-fault thrash)"
            layer_time = max(comp_s, mem_ssd)

    per_token_latency += layer_time
    rows.append({
        "Layer": exec_idx + 1, "Name": L["name"], "Kind": L["kind"],
        "Placement": place, "Served_From": served_from, "Bytes": sz,
        "Compute_s_all_cores": comp_s, "Mem_s_dram": mem_dram,
        "Mem_s_ssd (stall)": mem_ssd, "Mem_s_total": mem_dram + mem_ssd,
        "Layer_Time_s": layer_time,
    })

total_time_s = cold_load_s + TOKENS * per_token_latency
throughput_tokens_per_sec = 1.0 / per_token_latency if per_token_latency > 0 else 0.0

df = pd.DataFrame(rows)
df.to_csv("sim_normal_nocxl.csv", index=False)
print(df.to_string())

print("\nSummary (Baseline with Host Swap):")
print("Model fits in DRAM." if model_fits_in_dram else "Model does NOT fit in DRAM → page-fault thrash for spill layers.")
print(f"One-time cold SSD load: {cold_load_s:.6f} s")
print(f"Single-token Latency: {per_token_latency:.6f} s -> Throughput: {throughput_tokens_per_sec:.6f} tok/s")
print(f"Total time for T={TOKENS}: {total_time_s:.6f} s")

print("\nModel Size:")
print(f"  Total parameters: {total_params:,}  ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits}  ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes  ({fmt_bytes(total_model_bytes)})")

print("\nPlacement Breakdown (by bytes):")
print(f"  Host DRAM: {host_bytes:,}  ({fmt_bytes(host_bytes)})")
print(f"  Host SSD (NVMe) : {ssd_bytes:,}  ({fmt_bytes(ssd_bytes)})")

# --- CORRECTED LOGIC: Use placement data directly for summary ---
print("\nRuntime Traffic Served (steady-state per token):")
print(f"  Host DRAM bytes served: {host_bytes:,}")
print(f"  Host SSD (thrash) bytes served: {ssd_bytes:,}")

print("\nCapacities:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  DRAM-resident layers: {[i+1 for i in range(len(layers)) if placement[i]==PL_HOST_DRAM]}")
print(f"  Host SSD-resident layers : {[i+1 for i in range(len(layers)) if placement[i]==PL_HOST_SSD]}")

