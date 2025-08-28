# simulator_cxl_pipeline_parallel_update.py
import math
import pandas as pd
from collections import OrderedDict

from tiers import (
    GiB, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND, transfer_time_s
)
from model_cfg import build_layers, BYTES_PER_PARAM


# Canonical placement labels (use these everywhere)

PL_HOST_DRAM        = "Host DRAM"
PL_CXL_DEVICE_DRAM  = "CXL Device DRAM"
PL_CXL_DEVICE_NAND  = "CXL Device NAND"


# Global knobs

TOKENS = 16         # for total-time reporting
P = 4               # total CPU cores

cpu_freq_hz = 2.4e9
flops_per_cycle_per_core = 4.0
parallel_efficiency       = 0.90

def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)        # CXL Device DRAM path
def cxlssd_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)    # CXL Device NAND path


# Capacities

host_dram_capacity_bytes    = 4 * GiB
cxl_dev_dram_capacity_bytes = 4 * GiB
cxl_ssd_capacity_bytes      = 64 * GiB

# Device-side DRAM capacity inside CXL-SSD (used by warm pipeline)
DEVICE_CACHE_BYTES = 2 * GiB


# Simple device-side DRAM cache (LRU, full-layer granularity)

class DeviceCache:
    def __init__(self, cap_bytes):
        self.cap = max(0, int(cap_bytes))
        self.used = 0
        self.entries = OrderedDict()  # idx -> bytes cached (full or partial ok)
    def has(self, idx, size_b): 
        return self.entries.get(idx, 0) >= size_b
    def add(self, idx, add_b):
        if self.cap <= 0 or add_b <= 0: return 0
        while self.used + add_b > self.cap and self.entries:
            _, old = self.entries.popitem(last=False)  # evict LRU
            self.used -= old
        if self.used + add_b > self.cap: 
            return 0
        cur = self.entries.get(idx, 0)
        self.entries[idx] = cur + add_b
        self.entries.move_to_end(idx, last=True)
        self.used += add_b
        return add_b
    def amount(self, idx):
        return self.entries.get(idx, 0)


# Build layers (honors QUANT preset in model_cfg.py)

layers = build_layers()


# Placement: strict Host -> CXL Device DRAM -> CXL Device NAND

placement = [""] * len(layers)
host_free = host_dram_capacity_bytes
cxl_free  = cxl_dev_dram_capacity_bytes
for idx, L in enumerate(layers):
    sz = L["bytes"]
    if sz <= host_free:
        placement[idx] = PL_HOST_DRAM;       host_free -= sz
    elif sz <= cxl_free:
        placement[idx] = PL_CXL_DEVICE_DRAM; cxl_free  -= sz
    else:
        placement[idx] = PL_CXL_DEVICE_NAND

# Stage split: Stage1 = Host DRAM-resident layers; Stage2 = everything else
s1_idx = [i for i,p in enumerate(placement) if p == PL_HOST_DRAM]
s2_idx = [i for i,p in enumerate(placement) if p != PL_HOST_DRAM]


# Helper: Stage-1 sequential time

def stage1_time(cores):
    total = 0.0
    for k in s1_idx:
        L = layers[k]; sz=L["bytes"]; fl=L["flops"]
        comp_s = compute_time_s(fl, cores)
        mem_s  = dram_time_s(sz)  # Stage-1 layers are in Host DRAM by construction
        total += max(comp_s, mem_s)
    return total


# Device-side WARM PIPELINE (parallel NAND→device DRAM warming)
#   - Overlaps NAND reads for upcoming Stage-2 layers while current
#     Stage-2 layer is computing/serving.
#   - If a layer is fully warmed on arrival, serve it as a device-DRAM
#     hit (CXL path) instead of NAND.


# Warming knobs
DEV_WARM_CHANNELS = 2                 # simulate N parallel NAND "threads"
WARM_LOOKAHEAD    = 5                 # how many future Stage-2 layers to warm
WARM_CHUNK_BYTES  = 4 * 1024 * 1024   # 4 MiB warm chunk

def nand_warm_time_s(bytes_amt, channels=1):
    """Time to warm 'bytes_amt' from NAND into device DRAM with 'channels' overlap."""
    if bytes_amt <= 0: return 0.0
    eff_bw = CXL_SSD_NAND.bw_Bps * max(1, int(channels))
    chunk_bytes = getattr(CXL_SSD_NAND, "io_chunk_bytes", 256 * 1024)
    chunks = math.ceil(bytes_amt / chunk_bytes)
    # tiers API uses .chunk_latency_s
    return (bytes_amt / eff_bw) + chunks * CXL_SSD_NAND.chunk_latency_s

def stage2_time_with_device_warm(cores, collect_rows=False):
    """
    Sequential Stage-2 time with overlapped device-side warming of upcoming SSD layers.
    If collect_rows=True, also emit a per-layer table describing where bytes were served from.
    """
    total = 0.0
    rows = []
    dev_cache = DeviceCache(DEVICE_CACHE_BYTES)
    warm_progress = {k: dev_cache.amount(k) for k in s2_idx if placement[k] == PL_CXL_DEVICE_NAND}

    for pos, k in enumerate(s2_idx):
        L = layers[k]; sz=L["bytes"]; fl=L["flops"]; plc=placement[k]
        comp_s = compute_time_s(fl, cores)

        # Current layer memory service
        if plc == PL_CXL_DEVICE_DRAM:
            mem_s = cxl_time_s(sz); served_from = PL_CXL_DEVICE_DRAM
        elif plc == PL_CXL_DEVICE_NAND:
            warmed = warm_progress.get(k, 0)
            if warmed >= sz:
                mem_s = cxl_time_s(sz); served_from = f"{PL_CXL_DEVICE_NAND} (device DRAM hit)"
            else:
                mem_s = cxlssd_time_s(sz); served_from = f"{PL_CXL_DEVICE_NAND} (NAND miss)"
                # After serving, assume it is now resident (best-effort model)
                added = dev_cache.add(k, sz - warmed)
                warm_progress[k] = warmed + added
        else:  # Host DRAM shouldn't appear in Stage-2, but handle gracefully
            mem_s = dram_time_s(sz); served_from = PL_HOST_DRAM

        layer_time = max(comp_s, mem_s)
        total += layer_time

        # Use the layer's time budget to warm future SSD layers within lookahead
        budget = layer_time
        if budget > 0:
            for j in s2_idx[pos+1 : pos+1+WARM_LOOKAHEAD]:
                if budget <= 0: break
                if placement[j] != PL_CXL_DEVICE_NAND:
                    continue
                need = layers[j]["bytes"] - warm_progress.get(j, 0)
                if need <= 0:
                    continue
                while need > 0 and budget > 0:
                    chunk = min(WARM_CHUNK_BYTES, need)
                    t = nand_warm_time_s(chunk, channels=DEV_WARM_CHANNELS)
                    if t <= budget:
                        got = dev_cache.add(j, chunk)
                        warm_progress[j] = warm_progress.get(j, 0) + got
                        need   -= got
                        budget -= t
                        if got <= 0:
                            need = 0
                            break
                    else:
                        # partial warm within remaining budget (minus one chunk latency)
                        possible = int(max(0.0, budget - CXL_SSD_NAND.chunk_latency_s)
                                       * (CXL_SSD_NAND.bw_Bps * max(1, DEV_WARM_CHANNELS)))
                        if possible > 0:
                            got = dev_cache.add(j, possible)
                            warm_progress[j] = warm_progress.get(j, 0) + got
                        budget = 0.0
                        break

        if collect_rows:
            rows.append({
                "Layer": k+1, "Name": L["name"], "Kind": L["kind"],
                "Placement": plc, "Stage": 2, "Served_From": served_from,
                "Bytes": sz, "Compute_s (on stage cores)": comp_s,
                "Mem_s_dram": 0.0,
                "Mem_s_cxl": cxl_time_s(sz) if PL_CXL_DEVICE_DRAM in served_from else 0.0,
                "Mem_s_cxlssd": mem_s if "NAND miss" in served_from else 0.0,
                "Mem_s_total": mem_s, "Layer_Time_s": layer_time,
                "DevCache_Used_GB": round(dev_cache.used / GiB, 3),
                "Prefetch_Source": "Device warm (NAND→device DRAM)",
                "Prefetch_Next_Layer": None,
                "Prefetch_Bytes": 0, "Prefetch_Time_s": 0.0,
                "Scenario": "device-warm",
            })

    return (total, rows) if collect_rows else (total, None)


# Search core split for warmed pipeline

best = None
for p1 in range(1, P):
    p2 = P - p1
    S1 = stage1_time(p1)                               # Stage-1 unchanged (Host DRAM)
    S2, _ = stage2_time_with_device_warm(p2, False)    # Stage-2 warmed
    cand = (max(S1, S2), p1, p2, S1, S2)
    if best is None or cand < best:
        best = cand

bottleneck, p1, p2, S1, S2 = best
throughput_steady = 1.0 / bottleneck
pipeline_total_time = S1 + S2 + (TOKENS - 1) * bottleneck

# Build a detailed per-layer table for the warmed scenario using the chosen split
_, rows_warm = stage2_time_with_device_warm(p2, collect_rows=True)

# Compose full table (Stage-1 rows + Stage-2 warmed rows)
rows_all = []
# Stage-1 rows
for k in s1_idx:
    L = layers[k]; sz=L["bytes"]; fl=L["flops"]
    comp_s = compute_time_s(fl, p1)
    mem_s  = dram_time_s(sz)
    layer_time = max(comp_s, mem_s)
    rows_all.append({
        "Layer": k+1, "Name": L["name"], "Kind": L["kind"],
        "Placement": PL_HOST_DRAM, "Stage": 1, "Served_From": PL_HOST_DRAM,
        "Bytes": sz, "Compute_s (on stage cores)": comp_s,
        "Mem_s_dram": mem_s, "Mem_s_cxl": 0.0, "Mem_s_cxlssd": 0.0,
        "Mem_s_total": mem_s, "Layer_Time_s": layer_time,
        "DevCache_Used_GB": 0.0, "Prefetch_Source": "None",
        "Prefetch_Next_Layer": None, "Prefetch_Bytes": 0, "Prefetch_Time_s": 0.0,
        "Scenario": "device-warm",
    })
# Stage-2 rows
rows_all.extend(rows_warm or [])

df = pd.DataFrame(rows_all)
df.to_csv("sim_pipeline_2stage_device_warm.csv", index=False)
print(df)

print("\nSummary (Pipeline 2-Stage + device-side warm (NAND→device DRAM)):")
print(f"Core split search over P={P} → p1={p1} (Stage1/Host DRAM), p2={p2} (Stage2/CXL)")
print(f"Per-token S1={S1:.6f} s, S2(warm)={S2:.6f} s, bottleneck={bottleneck:.6f} s, throughput={throughput_steady:.6f} tok/s")
print(f"Total time for T={TOKENS}: {pipeline_total_time:.6f} s")


# Final model / placement info

total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"]  for L in layers)
model_dtype_bits = int(BYTES_PER_PARAM * 8)

host_bytes = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == PL_HOST_DRAM)
cxl_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == PL_CXL_DEVICE_DRAM)
ssd_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == PL_CXL_DEVICE_NAND)

def fmt_bytes(n): return f"{n/(1024**3):.3f} GiB"
def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

print("\nModel Size:")
print(f"  Total parameters: {total_params:,}  ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits}  ({BYTES_PER_PARAM} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes  ({fmt_bytes(total_model_bytes)})")

print("\nPlacement Breakdown (by bytes):")
print(f"  {PL_HOST_DRAM}: {host_bytes:,}  ({fmt_bytes(host_bytes)})")
print(f"  {PL_CXL_DEVICE_DRAM}: {cxl_bytes:,}  ({fmt_bytes(cxl_bytes)})")
print(f"  {PL_CXL_DEVICE_NAND}: {ssd_bytes:,}  ({fmt_bytes(ssd_bytes)})")

print("\nCapacities & Tiers:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL Device DRAM cap : {cxl_dev_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL Device NAND cap  : {cxl_ssd_capacity_bytes/(1024**3):.3f} GB")
print(f"  Host DRAM layers: {[i+1 for i in s1_idx]}")
print(f"  CXL Device DRAM layers : {[i+1 for i in range(len(layers)) if placement[i] == PL_CXL_DEVICE_DRAM]}")
print(f"  CXL Device NAND layers  : {[i+1 for i in s2_idx if placement[i] == PL_CXL_DEVICE_NAND]}")
