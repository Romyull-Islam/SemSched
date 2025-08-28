import math
import pandas as pd

from tiers import (
    GiB, IO_CHUNK_BYTES, Tier, HOST_DRAM, transfer_time_s,
    NVME_STREAM_BW, NVME_STREAM_LAT_S,
    NVME_THRASH_BW, NVME_THRASH_LAT_S, NVME_FAULT_OVERHEAD
)
from model_cfg import build_layers, HOT_LAYERS_BY_NAME, BYTES_PER_PARAM

MiB = 1024 ** 2


# Global knobs

TOKENS = 16

cpu_freq_hz = 2.4e9
cpu_cores   = 4
flops_per_cycle_per_core = 4.0
parallel_efficiency       = 0.90

# Machine: Host DRAM + regular Host SSD (NO CXL)
host_dram_capacity_bytes = 4 * GiB

# Page-fault readahead window (OS-level knob; keep sim-specific)
SSD_PF_READAHEAD_BYTES   = 1 * MiB

# One-time cold load: charge SSD time ONCE to bring DRAM-resident tensors into RAM.
CHARGE_COLD_LOAD_FOR_DRAM = True


# Helpers

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


# Build layers (honors QUANT)

layers = build_layers()
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}


# Placement (Host DRAM + Host SSD only)

placement = ["Host SSD (NVMe)"] * len(layers)  # default spill
host_free = host_dram_capacity_bytes

# Pin hot: lm_head then final_norm; also try to keep embed in Host DRAM if there’s room
for n in ("lm_head", "final_norm", "embed_tokens"):
    idx = name_to_idx[n]
    sz  = layers[idx]["bytes"]
    if sz <= host_free:
        placement[idx] = "Host DRAM"; host_free -= sz

# Fill with as many early decoders as fit
num_blocks = sum(1 for L in layers if L["kind"] == "DecoderBlock")
for i in range(1, 1+num_blocks):
    sz = layers[i]["bytes"]
    if sz <= host_free:
        placement[i] = "Host DRAM"; host_free -= sz
    else:
        break

# Totals & fit check
total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"]  for L in layers)
model_dtype_bits = int(BYTES_PER_PARAM * 8)

host_bytes = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "Host DRAM")
ssd_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "Host SSD (NVMe)")
model_fits_in_dram = (total_model_bytes <= host_dram_capacity_bytes)


# Execute (sequential; no prefetch)

rows = []
cold_load_s = 0.0
if CHARGE_COLD_LOAD_FOR_DRAM:
    cold_load_s = sum(ssd_cold_time_s(layers[i]["bytes"]) for i in range(len(layers)) if placement[i]=="Host DRAM")

per_token_latency = 0.0
for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    sz = L["bytes"]; flops = L["flops"]; place = placement[exec_idx]

    comp_s = compute_time_s(flops)
    if model_fits_in_dram:
        mem_dram = dram_time_s(sz); mem_ssd = 0.0
        served_from = "Host DRAM"
        layer_time = max(comp_s, mem_dram)
    else:
        if place == "Host DRAM":
            mem_dram = dram_time_s(sz); mem_ssd = 0.0
            served_from = "Host DRAM"
            layer_time = max(comp_s, mem_dram)
        else:
            mem_dram = 0.0; mem_ssd = ssd_token_time_s(sz)
            served_from = "Host SSD (NVMe, page-fault thrash)"
            layer_time = max(comp_s, mem_ssd)

    per_token_latency += layer_time

    rows.append({
        "Layer": exec_idx + 1,
        "Name": L["name"],
        "Kind": L["kind"],
        "Placement": place,
        "Served_From": served_from,
        "Bytes": sz,
        "Compute_s_all_cores": comp_s,
        "Mem_s_dram": mem_dram,
        "Mem_s_ssd (stall)": mem_ssd,
        "Mem_s_total": mem_dram + mem_ssd,
        "Layer_Time_s": layer_time,
        # parity with other sims
        "CXL_Hit_Bytes": 0, "CXL_Miss_Bytes": 0,
        "Prefetch_Source": "None", "Prefetch_Next_Layer": None,
        "Prefetch_Bytes": 0, "Prefetch_Time_s": 0.0, "CXL_Cache_Used_GB": 0.0,
    })

total_time_s = cold_load_s + TOKENS * per_token_latency
throughput_tokens_per_sec = 1.0 / per_token_latency if per_token_latency > 0 else 0.0

df = pd.DataFrame(rows)
df.to_csv("sim_normal_nocxl.csv", index=False)
print(df)

print("\nSummary (Host DRAM + Host SSD, no CXL):")
print("Model fits in DRAM." if model_fits_in_dram else "Model does NOT fit in DRAM → page-fault thrash for spill layers.")
print(f"One-time cold SSD load (streaming): {cold_load_s:.6f} s")
print(f"Steady-state single-token latency (thrash model): {per_token_latency:.6f} s")
print(f"Estimated tokens/sec (steady-state): {throughput_tokens_per_sec:.6f}")
print(f"Total time for T={TOKENS}: {total_time_s:.6f} s")

print("\nModel Size:")
print(f"  Total parameters: {total_params:,}  ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits}  ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes  ({fmt_bytes(total_model_bytes)})")

print("\nPlacement Breakdown (by bytes):")
print(f"  Host DRAM: {host_bytes:,}  ({fmt_bytes(host_bytes)})")
print(f"  Host SSD (NVMe) : {ssd_bytes:,}  ({fmt_bytes(ssd_bytes)})")

print("\nRuntime Traffic Served (steady-state per token):")
print(f"  Host DRAM bytes served: {sum(df[df.Placement=='Host DRAM']['Bytes']):,}")
print(f"  Host SSD (thrash) bytes served: {sum(df[df.Placement=='Host SSD (NVMe)']['Bytes']):,}")

print("\nCapacities:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  DRAM-resident layers: {[i+1 for i in range(len(layers)) if placement[i]=='Host DRAM']}")
print(f"  Host SSD-resident layers : {[i+1 for i in range(len(layers)) if placement[i]=='Host SSD (NVMe)']}")
