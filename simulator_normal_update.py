import math
import pandas as pd
from collections import OrderedDict

from tiers import (
    GiB, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s, chunk_us
)
from model_cfg import build_layers, HOT_LAYERS_BY_NAME, BYTES_PER_PARAM

# ==============================
# Global knobs
# ==============================
TOKENS = 16

cpu_freq_hz = 2.4e9
cpu_cores   = 4
flops_per_cycle_per_core = 4.0
parallel_efficiency       = 0.90

# Capacities
host_dram_capacity_bytes    = 4 * GiB
cxl_dev_dram_capacity_bytes = 4 * GiB
cxl_ssd_capacity_bytes      = 64 * GiB

def compute_time_s(flops):
    if flops <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):    return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):     return transfer_time_s(n, CXL_DRAM)
def cxlssd_time_s(n):  return transfer_time_s(n, CXL_SSD_NAND)

# Device-side DRAM cache in CXL-SSD (LRU)
class DeviceCache:
    def __init__(self, capacity_bytes):
        self.cap = max(0, int(capacity_bytes))
        self.used = 0
        self.entries = OrderedDict()  # key = layer index, value = bytes cached (full layer)
    def has(self, idx, need_bytes): 
        return self.entries.get(idx, 0) >= need_bytes
    def touch(self, idx, size_b):
        if self.cap <= 0 or size_b <= 0: return 0
        while self.used + size_b > self.cap and self.entries:
            _, old = self.entries.popitem(last=False)
            self.used -= old
        if self.used + size_b > self.cap:
            return 0
        self.entries[idx] = size_b
        self.entries.move_to_end(idx, last=True)
        self.used += size_b
        return size_b

# Build layers (honors QUANT)
layers = build_layers()
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

# Placement: Host DRAM -> CXL Device DRAM -> CXL Device NAND, then ensure hot layers in Host DRAM
placement = [""] * len(layers)
host_free = host_dram_capacity_bytes
cxl_free  = cxl_dev_dram_capacity_bytes
for idx, L in enumerate(layers):
    sz = L["bytes"]
    if sz <= host_free:
        placement[idx] = "Host DRAM";       host_free -= sz
    elif sz <= cxl_free:
        placement[idx] = "CXL Device DRAM"; cxl_free  -= sz
    else:
        placement[idx] = "CXL Device NAND"

# Ensure hot layers (lm_head, then final_norm) are in Host DRAM by evicting biggest non-hot Host DRAM layers if needed
def pin_hot_layers_in_dram(placement, layers, host_cap_bytes):
    used = sum(l["bytes"] for i,l in enumerate(layers) if placement[i] == "Host DRAM")
    hot_order = ["lm_head", "final_norm"]
    for hot in hot_order:
        i = name_to_idx[hot]
        if placement[i] == "Host DRAM": 
            continue
        need = layers[i]["bytes"] - max(0, host_cap_bytes - used)
        if need > 0:
            victims = sorted(
                [j for j in range(len(layers)) if placement[j]=="Host DRAM" and layers[j]["name"] not in hot_order],
                key=lambda j: layers[j]["bytes"], reverse=True
            )
            for v in victims:
                placement[v] = "CXL Device DRAM" if layers[v]["bytes"] <= (
                    cxl_dev_dram_capacity_bytes - sum(layers[k]["bytes"] for k,p in enumerate(placement) if p=="CXL Device DRAM")
                ) else "CXL Device NAND"
                used -= layers[v]["bytes"]
                need -= layers[v]["bytes"]
                if need <= 0: break
        placement[i] = "Host DRAM"; used += layers[i]["bytes"]
    return placement

placement = pin_hot_layers_in_dram(placement, layers, host_dram_capacity_bytes)

pinned_host = {i for i,p in enumerate(placement) if p=="Host DRAM"}
pinned_cxl  = {i for i,p in enumerate(placement) if p=="CXL Device DRAM"}

# Execute (sequential, NO host-side prefetch; with device-side cache)
rows = []
total_time_s = 0.0
dev_cache = DeviceCache(2 * GiB)

dev_hit_B = 0
dev_miss_B = 0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement[exec_idx]
    sz = L["bytes"]; flops = L["flops"]

    comp_s = compute_time_s(flops)
    mem_s_dram = mem_s_cxl = mem_s_cxlssd = 0.0
    served_from = ""

    if place == "Host DRAM":
        mem_s_dram = dram_time_s(sz)
        served_from = "Host DRAM"

    elif place == "CXL Device DRAM":
        mem_s_cxl  = cxl_time_s(sz)
        served_from = "CXL Device DRAM"

    else:  # CXL Device NAND with device-side DRAM cache
        if dev_cache.has(exec_idx, sz):
            dev_hit_B += sz
            mem_s_cxlssd = cxl_time_s(sz)      # served via device DRAM over CXL
            served_from  = "CXL Device NAND (device DRAM hit)"
        else:
            dev_miss_B += sz
            mem_s_cxlssd = cxlssd_time_s(sz)   # NAND path
            served_from  = "CXL Device NAND (NAND miss)"
            dev_cache.touch(exec_idx, sz)

    mem_total = mem_s_dram + mem_s_cxl + mem_s_cxlssd
    layer_time = max(comp_s, mem_total)
    total_time_s += layer_time

    rows.append({
        "Layer": exec_idx + 1,
        "Name": L["name"],
        "Kind": L["kind"],
        "Placement": place,
        "Served_From": served_from,
        "Bytes": sz,
        "Compute_s_all_cores": comp_s,
        "Mem_s_dram": mem_s_dram,
        "Mem_s_cxl": mem_s_cxl,
        "Mem_s_cxlssd": mem_s_cxlssd,
        "Mem_s_total": mem_total,
        "Layer_Time_s": layer_time,
        "DevCache_Hit_Bytes": dev_hit_B,
        "DevCache_Miss_Bytes": dev_miss_B,
        "Prefetch_Source": "None",
        "Prefetch_Next_Layer": None,
        "Prefetch_Bytes": 0,
        "Prefetch_Time_s": 0.0,
        "CXL_Cache_Used_GB": 0.0,
    })

df = pd.DataFrame(rows)

throughput_tokens_per_sec = 1.0 / total_time_s if total_time_s > 0 else 0.0

total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"]  for L in layers)
model_dtype_bits = int(BYTES_PER_PARAM * 8)

host_bytes = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "Host DRAM")
cxl_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "CXL Device DRAM")
ssd_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "CXL Device NAND")

def fmt_bytes(n): return f"{n / (1024**3):.3f} GiB"
def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

df.to_csv("sim_normal_noprefetch.csv", index=False)
print(df)
print("\nSummary:")
print(f"Single-token Latency: {total_time_s:.6f} s")
print(f"Estimated Tokens/sec (sequential): {throughput_tokens_per_sec:.6f}")
print(f"Total time for T={TOKENS}: {TOKENS*total_time_s:.6f} s")

print("\nModel Size:")
print(f"  Total parameters: {total_params:,}  ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits}  ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes  ({fmt_bytes(total_model_bytes)})")

print("\nPlacement Breakdown (by bytes):")
print(f"  Host DRAM: {host_bytes:,}  ({fmt_bytes(host_bytes)})")
print(f"  CXL Device DRAM : {cxl_bytes:,}  ({fmt_bytes(cxl_bytes)})")
print(f"  CXL Device NAND  : {ssd_bytes:,}  ({fmt_bytes(ssd_bytes)})")

print("\nRuntime Traffic Served (this run):")
print(f"  Device-cache HIT bytes: {dev_hit_B:,}")
print(f"  Device-cache MISS bytes (NAND path): {dev_miss_B:,}")
print(f"  Peak device cache cap: {2:.3f} GB")

print("\nCapacities & Tiers:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL Device DRAM cap : {cxl_dev_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL Device NAND cap  : {cxl_ssd_capacity_bytes/(1024**3):.3f} GB")
print(f"  Host DRAM layers: {[i+1 for i in sorted(list(pinned_host))]}")
print(f"  CXL Device DRAM layers : {[i+1 for i in sorted(list(pinned_cxl))]}")
print(f"  CXL Device NAND layers  : {[i+1 for i in range(len(layers)) if i not in pinned_host and i not in pinned_cxl]}")
