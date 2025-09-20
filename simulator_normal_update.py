import math
import pandas as pd
from collections import OrderedDict

from tiers import (
    GiB, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s,
    Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
)
from model_cfg import build_layers, BYTES_PER_PARAM
from sim_cfg import (
    TOKENS,
    cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency,
    host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes, cxl_ssd_capacity_bytes
)

# Canonical placement labels
PL_HOST_DRAM      = "Host DRAM"
PL_CXL_DEV_DRAM   = "CXL Device DRAM"
PL_CXL_DEV_NAND   = "CXL Device NAND"

def compute_time_s(flops):
    if flops <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):     return transfer_time_s(n, HOST_DRAM)
def cxl_dram_time_s(n): return transfer_time_s(n, CXL_DRAM)
def cxl_nand_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)
def ssd_cold_time_s(n):
    return transfer_time_s(n, Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S))

class DeviceDramPool:
    def __init__(self, capacity_bytes):
        self.cap = max(0, int(capacity_bytes)); self.used = 0
        self.entries = OrderedDict()
    def bytes_cached(self, idx): return self.entries.get(idx, 0)
    def add_bytes(self, idx, add_b):
        add_b = int(add_b)
        if self.cap <= 0 or add_b <= 0: return 0
        while self.used + add_b > self.cap and self.entries:
            _, old = self.entries.popitem(last=False); self.used -= old
        if self.used + add_b > self.cap: return 0
        self.entries[idx] = self.entries.get(idx, 0) + add_b
        self.entries.move_to_end(idx, last=True); self.used += add_b
        return add_b

layers = build_layers()
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}

# --- PLACEMENT LOGIC ---
placement = [None] * len(layers)
host_free = host_dram_capacity_bytes
cxl_free  = cxl_dev_dram_capacity_bytes
hot_indices = [name_to_idx.get("final_norm"), name_to_idx.get("lm_head")]
for idx in hot_indices:
    if idx is not None and layers[idx]["bytes"] <= host_free:
        placement[idx] = PL_HOST_DRAM; host_free -= layers[idx]["bytes"]
for idx, L in enumerate(layers):
    if placement[idx] is None and L["bytes"] <= host_free:
        placement[idx] = PL_HOST_DRAM; host_free -= L["bytes"]
for idx, L in enumerate(layers):
    if placement[idx] is None and L["bytes"] <= cxl_free:
        placement[idx] = PL_CXL_DEV_DRAM; cxl_free -= L["bytes"]
for idx in range(len(layers)):
    if placement[idx] is None: placement[idx] = PL_CXL_DEV_NAND

pinned_cxl = {i for i,p in enumerate(placement) if p==PL_CXL_DEV_DRAM}
pinned_cxl_bytes = sum(layers[i]["bytes"] for i in pinned_cxl)
pool_capacity = max(0, cxl_dev_dram_capacity_bytes - pinned_cxl_bytes)
devpool = DeviceDramPool(pool_capacity)

rows = []
per_token_latency, cxl_hit_served_B, nand_served_B = 0.0, 0, 0
for exec_idx, L in enumerate(layers):
    place, sz, flops = placement[exec_idx], L["bytes"], L["flops"]
    comp_s = compute_time_s(flops)
    mem_s_dram = mem_s_cxl = mem_s_nand = 0.0; served_from = ""

    if place == PL_HOST_DRAM:
        mem_s_dram = dram_time_s(sz); served_from = PL_HOST_DRAM
    elif place == PL_CXL_DEV_DRAM:
        mem_s_cxl = cxl_dram_time_s(sz); cxl_hit_served_B += sz
        served_from = f"{PL_CXL_DEV_DRAM} (resident)"
    else:
        cached = devpool.bytes_cached(exec_idx); rem = sz - cached
        if cached > 0: mem_s_cxl = cxl_dram_time_s(cached); cxl_hit_served_B += cached
        if rem > 0: mem_s_nand = cxl_nand_time_s(rem); nand_served_B += rem; devpool.add_bytes(exec_idx, rem)
        served_from = f"Cache({cached/1e6:.1f}MB)+NAND({rem/1e6:.1f}MB)" if cached > 0 else f"{PL_CXL_DEV_NAND} (NAND)"

    mem_total = mem_s_dram + mem_s_cxl + mem_s_nand
    layer_time = max(comp_s, mem_total)
    per_token_latency += layer_time
    rows.append({"Layer": exec_idx + 1, "Name": L["name"], "Placement": place, "Served_From": served_from, "Bytes": sz, "Compute_s_all_cores": comp_s, "Mem_s_total": mem_total, "Layer_Time_s": layer_time, "CXL_Hit_Bytes_cum": cxl_hit_served_B, "NAND_Bytes_cum": nand_served_B, "CXL_Device_DRAM_Cache_Used_GB": round(devpool.used / GiB, 3)})

df = pd.DataFrame(rows)
df.to_csv("sim_normal_corrected.csv", index=False)
print(df.to_string())


# --- Summary ---
total_model_bytes = sum(L["bytes"] for L in layers)
cold_load_s = ssd_cold_time_s(total_model_bytes)

throughput = 1.0 / per_token_latency if per_token_latency > 0 else 0
total_time_for_all_tokens = cold_load_s + (TOKENS * per_token_latency)
total_params = sum(L["params"] for L in layers)
model_dtype_bits = int(BYTES_PER_PARAM * 8)

host_bytes = sum(layers[i]["bytes"] for i, p in enumerate(placement) if p == PL_HOST_DRAM)
cxl_dram_bytes = sum(layers[i]["bytes"] for i, p in enumerate(placement) if p == PL_CXL_DEV_DRAM)
cxl_nand_bytes = sum(layers[i]["bytes"] for i, p in enumerate(placement) if p == PL_CXL_DEV_NAND)

host_traffic = sum(df[df.Placement==PL_HOST_DRAM]['Bytes'])
cxl_dram_traffic = cxl_hit_served_B
cxl_nand_traffic = nand_served_B

def fmt_bytes(n): return f"{n / GiB:.3f} GiB"
def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    return str(n)

# Create lists of layer numbers for each placement
host_layers = sorted([i+1 for i, p in enumerate(placement) if p == PL_HOST_DRAM])
cxl_dram_layers = sorted([i+1 for i, p in enumerate(placement) if p == PL_CXL_DEV_DRAM])
cxl_nand_layers = sorted([i+1 for i, p in enumerate(placement) if p == PL_CXL_DEV_NAND])

print(f"\nSummary (CXL Tiered Memory, Sequential):")
print(f"One-time cold SSD load: {cold_load_s:.6f} s")
print(f"Single-token Latency: {per_token_latency:.6f} s -> Throughput: {throughput:.6f} tok/s")
print(f"Total time for T={TOKENS}: {total_time_for_all_tokens:.6f} s")

print("\nModel Size:")
print(f"  Total parameters: {total_params:,} ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits} ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes ({fmt_bytes(total_model_bytes)})")

print("\nPlacement Breakdown:")
print(f"  Host DRAM ({fmt_bytes(host_bytes)}): Layers {host_layers}")
print(f"  CXL Device DRAM ({fmt_bytes(cxl_dram_bytes)}): Layers {cxl_dram_layers}")
print(f"  CXL Device NAND ({fmt_bytes(cxl_nand_bytes)}): Layers {cxl_nand_layers}")

print("\nRuntime Traffic Served (per token):")
print(f"  From Host DRAM: {host_traffic:,} bytes")
print(f"  From CXL Device DRAM (Hits): {cxl_dram_traffic:,} bytes")
print(f"  From CXL Device NAND (Misses): {cxl_nand_traffic:,} bytes")

print("\nCapacities:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/GiB:.3f} GB")
print(f"  CXL Device DRAM cap: {cxl_dev_dram_capacity_bytes/GiB:.3f} GB")

