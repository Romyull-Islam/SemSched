import math
import pandas as pd
from collections import OrderedDict

# shared tiers & model
from tiers import (
    GiB, IO_CHUNK_BYTES, HOST_DRAM, CXL_DRAM, CXL_SSD_NAND,
    transfer_time_s, chunk_us
)
from model_cfg import build_layers, HOT_LAYERS_BY_NAME, BYTES_PER_PARAM


# Global knobs

TOKENS = 16

# CPU (all cores work on one layer at a time for sequential)
cpu_freq_hz = 2.4e9
cpu_cores   = 4
flops_per_cycle_per_core = 4.0
parallel_efficiency       = 0.90


# Capacities (machine shape)

host_dram_capacity_bytes    = 4 * GiB
cxl_dev_dram_capacity_bytes = 4 * GiB
cxl_ssd_capacity_bytes      = 64 * GiB


# Prefetch knobs (CXL Device NAND -> CXL Device DRAM)

# aggressive so prefetch "bites"
prefetch_lookahead = 5                   # how many future NAND layers to stage
prefetch_gran_bytes = 4 * 1024 * 1024   # 4 MiB chunks
prefetch_share_when_current_is_cxl = 1.0  # share BW while current layer uses CXL (optimistic)

# Keep CXL Device DRAM mostly free for staging (dynamic cache)
MAX_PERM_CXL_DRAM_DECODERS = 1


# Helpers

def compute_time_s(flops: float) -> float:
    if flops <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def dram_time_s(n):   return transfer_time_s(n, HOST_DRAM)
def cxl_time_s(n):    return transfer_time_s(n, CXL_DRAM)       # CXL Device DRAM path
def cxlssd_time_s(n): return transfer_time_s(n, CXL_SSD_NAND)   # CXL Device NAND path


# Build execution order (uses global QUANT from model_cfg)

layers = build_layers()


# Placement
# - Pin hot layers in Host DRAM: lm_head first, then final_norm.
# - Also try to keep embed in Host DRAM (cheap + used at start).
# - Keep at most one decoder permanently in CXL Device DRAM; rest spill to CXL Device NAND so CXL can act as staging cache.
# ==============================
placement = [""] * len(layers)
host_free = host_dram_capacity_bytes
cxl_free  = cxl_dev_dram_capacity_bytes

def try_pin(idx, tier_name="Host DRAM"):
    global host_free, cxl_free
    sz = layers[idx]["bytes"]
    if tier_name == "Host DRAM" and sz <= host_free:
        placement[idx] = "Host DRAM"; host_free -= sz; return True
    if tier_name == "CXL Device DRAM" and sz <= cxl_free:
        placement[idx] = "CXL Device DRAM"; cxl_free -= sz; return True
    return False

# 1) Pin hot (lm_head then final_norm). Also try to keep embed in Host DRAM if room.
name_to_idx = {L["name"]: i for i, L in enumerate(layers)}
for hot in ("lm_head", "final_norm"):
    try_pin(name_to_idx[hot], "Host DRAM")
try_pin(name_to_idx["embed_tokens"], "Host DRAM")

# 2) Fill Host DRAM with earliest decoders that fit
num_blocks = sum(1 for L in layers if L["kind"] == "DecoderBlock")
for i in range(1, 1+num_blocks):
    if placement[i]: continue
    if not try_pin(i, "Host DRAM"):
        break

# 3) Put at most one decoder in CXL Device DRAM; the rest to CXL Device NAND
cxl_residents = 0
for i in range(1, 1+num_blocks):
    if placement[i]: continue
    if cxl_residents < MAX_PERM_CXL_DRAM_DECODERS and try_pin(i, "CXL Device DRAM"):
        cxl_residents += 1
    else:
        placement[i] = "CXL Device NAND"

# Any remaining non-decoder tail (unlikely)
for idx in range(1+num_blocks, len(layers)-2):
    if not placement[idx]:
        placement[idx] = "CXL Device NAND"

# Treat “Host DRAM/CXL Device DRAM” as “pinned”
pinned_host = {i for i,p in enumerate(placement) if p == "Host DRAM"}
pinned_cxl  = {i for i,p in enumerate(placement) if p == "CXL Device DRAM"}


# Caches

class CxlCache:
    """Host-visible CXL Device DRAM staging cache (for prefetch)."""
    def __init__(self, capacity_bytes):
        self.capacity = max(0, capacity_bytes)
        self.used = 0
        self.entries = OrderedDict()  # idx -> bytes
    def cached_bytes(self, idx): return self.entries.get(idx, 0)
    def add_bytes(self, idx, add_b):
        if add_b <= 0 or self.capacity <= 0: return 0
        # evict LRU until room
        while self.used + add_b > self.capacity and self.entries:
            _, old_b = self.entries.popitem(last=False); self.used -= old_b
        if self.used + add_b > self.capacity: return 0
        cur = self.entries.get(idx, 0)
        new = cur + add_b
        self.entries[idx] = new
        self.entries.move_to_end(idx, last=True)
        self.used += add_b
        return add_b
    def consume(self, idx, need_b=None):
        if idx not in self.entries: return 0
        have = self.entries[idx]
        take = have if need_b is None else min(have, need_b)
        remain = have - take
        if remain > 0: self.entries[idx] = remain
        else: self.entries.pop(idx)
        self.used -= take
        return take

class DeviceCache:
    """Device-side DRAM cache inside the CXL Device NAND (LRU, full-layer granularity)."""
    def __init__(self, capacity_bytes):
        self.cap = max(0, int(capacity_bytes))
        self.used = 0
        self.entries = OrderedDict()  # idx -> full layer size
    def has(self, idx, size_b): return self.entries.get(idx, 0) >= size_b
    def touch(self, idx, size_b):
        if self.cap <= 0 or size_b <= 0: return 0
        while self.used + size_b > self.cap and self.entries:
            _, old = self.entries.popitem(last=False); self.used -= old
        if self.used + size_b > self.cap: return 0
        self.entries[idx] = size_b
        self.entries.move_to_end(idx, last=True)
        self.used += size_b
        return size_b

pinned_cxl_bytes = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "CXL Device DRAM")
dyn_cxl_capacity = max(0, cxl_dev_dram_capacity_bytes - pinned_cxl_bytes)
cxl_cache = CxlCache(dyn_cxl_capacity)
# device-side cache size: use e.g., 2 GiB
device_cache = DeviceCache(2 * GiB)


# Execute (sequential with prefetch + device cache, allowing PARTIAL hits)

rows = []
total_time_s = 0.0
dram_served_B = 0
cxl_hit_served_B = 0           # bytes served from CXL Device DRAM (resident + prefetched)
nand_served_B = 0              # bytes served from NAND
devdram_served_B = 0           # bytes served from device-side DRAM (inside NAND device)

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement[exec_idx]
    L_bytes = L["bytes"]
    flops = L["flops"]

    comp_time = compute_time_s(flops)

    mem_s_dram = mem_s_cxl = mem_s_cxlssd = 0.0
    served_from = ""

    if place == "Host DRAM":
        mem_s_dram = dram_time_s(L_bytes)
        dram_served_B += L_bytes
        served_from = "Host DRAM"

    elif place == "CXL Device DRAM":
        mem_s_cxl = cxl_time_s(L_bytes)
        cxl_hit_served_B += L_bytes
        served_from = "CXL Device DRAM (resident hit)"

    else:  # CXL Device NAND resident
        # --- PARTIAL CXL Device DRAM consumption first ---
        staged_b = cxl_cache.cached_bytes(exec_idx)
        take_b   = min(L_bytes, staged_b)
        if take_b > 0:
            cxl_cache.consume(exec_idx, take_b)
            cxl_hit_served_B += take_b
            mem_s_cxl += cxl_time_s(take_b)

        rem_b = L_bytes - take_b
        if rem_b > 0:
            if device_cache.has(exec_idx, rem_b):
                devdram_served_B += rem_b
                mem_s_cxlssd += cxl_time_s(rem_b)     # served from device DRAM via CXL
                served_from = "CXL Device DRAM (partial) + CXL Device NAND (device DRAM)"
            else:
                nand_served_B += rem_b
                mem_s_cxlssd += cxlssd_time_s(rem_b)  # NAND for remainder
                served_from = "CXL Device DRAM (partial) + CXL Device NAND (NAND)"
                device_cache.touch(exec_idx, rem_b)
        else:
            served_from = "CXL Device DRAM (prefetch FULL hit)"

    t_mem = mem_s_dram + mem_s_cxl + mem_s_cxlssd
    layer_time_if_no_overlap = max(comp_time, t_mem)

    # Prefetch next up to 'prefetch_lookahead' NAND-resident layers (NAND -> CXL Device DRAM)
    pf_bytes = 0; pf_time  = 0.0; pf_targets = []
    budget_s = layer_time_if_no_overlap
    if budget_s > 0 and prefetch_lookahead > 0:
        prefetch_bw_Bps = CXL_SSD_NAND.bw_Bps * (prefetch_share_when_current_is_cxl if place != "Host DRAM" else 1.0)
        for k in range(1, prefetch_lookahead + 1):
            if budget_s <= 0: break
            next_idx = exec_idx + k
            if next_idx >= len(layers): break
            if placement[next_idx] != "CXL Device NAND": continue
            need = layers[next_idx]["bytes"] - cxl_cache.cached_bytes(next_idx)
            if need <= 0: continue
            pf_targets.append(next_idx + 1)  # 1-based for readability
            while need > 0 and budget_s > 0:
                chunk = min(prefetch_gran_bytes, need)
                # time to pull 'chunk' from NAND into CXL Device DRAM
                t_chunk = (chunk / prefetch_bw_Bps) + CXL_SSD_NAND.chunk_latency_s
                if t_chunk <= budget_s:
                    got = cxl_cache.add_bytes(next_idx, chunk)
                    if got == 0: break
                    need     -= got
                    pf_bytes += got
                    pf_time  += t_chunk
                    budget_s -= t_chunk
                else:
                    possible = max(0, int((budget_s - CXL_SSD_NAND.chunk_latency_s) * prefetch_bw_Bps))
                    if possible > 0:
                        got = cxl_cache.add_bytes(next_idx, possible)
                        pf_bytes += got
                        pf_time  += budget_s
                        budget_s  = 0.0
                    break

    layer_time = layer_time_if_no_overlap
    total_time_s += layer_time

    rows.append({
        "Layer": exec_idx + 1,
        "Name": L["name"],
        "Kind": L["kind"],
        "Placement": place,
        "Served_From": served_from,
        "Bytes": L_bytes,
        "Compute_s_all_cores": comp_time,
        "Mem_s_dram": mem_s_dram,
        "Mem_s_cxl": mem_s_cxl,
        "Mem_s_cxlssd": mem_s_cxlssd,
        "Mem_s_total": t_mem,
        "Layer_Time_s": layer_time,
        "CXL_Hit_Bytes_cum": cxl_hit_served_B,
        "DeviceDRAM_Bytes_cum": devdram_served_B,
        "NAND_Bytes_cum": nand_served_B,
        "Prefetch_Source": "CXL Device NAND→CXL Device DRAM" if pf_bytes > 0 else "None",
        "Prefetch_Next_Layer": ",".join(map(str, pf_targets)) if pf_targets else None,
        "Prefetch_Bytes": pf_bytes,
        "Prefetch_Time_s": pf_time,
        "CXL_Cache_Used_GB": round(cxl_cache.used / GiB, 3),
        "DeviceCache_Used_GB": round(device_cache.used / GiB, 3),
    })


# Final Summary (uniform)

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

df.to_csv("sim_tired_prefetch_devicecache.csv", index=False)
print(df)

print("\nLatency model per 256 KiB chunk:")
print(f"  Host DRAM         : {chunk_us(HOST_DRAM):.2f} µs (BW={HOST_DRAM.bw_Bps/1e9:.1f} GB/s, L0={HOST_DRAM.chunk_latency_s*1e6:.1f} µs)")
print(f"  CXL Device DRAM   : {chunk_us(CXL_DRAM):.2f} µs (BW={CXL_DRAM.bw_Bps/1e9:.1f} GB/s, L0={CXL_DRAM.chunk_latency_s*1e6:.1f} µs)")
print(f"  CXL Device NAND   : {chunk_us(CXL_SSD_NAND):.2f} µs (BW={CXL_SSD_NAND.bw_Bps/1e9:.1f} GB/s, L0={CXL_SSD_NAND.chunk_latency_s*1e6:.1f} µs)")

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
print(f"  Host DRAM bytes served: {sum(df[df.Placement=='Host DRAM']['Bytes']):,}")
print(f"  CXL Device DRAM bytes served (resident + prefetched): {int(df['CXL_Hit_Bytes_cum'].max()):,}")
print(f"  Device-DRAM bytes served: {int(df['DeviceDRAM_Bytes_cum'].max()):,}")
print(f"  NAND bytes served: {int(df['NAND_Bytes_cum'].max()):,}")
print(f"  Peak CXL Device DRAM cache use: {max(df['CXL_Cache_Used_GB']):.3f} GB / {dyn_cxl_capacity/(1024**3):.3f} GB")
print(f"  Peak device cache use : {max(df['DeviceCache_Used_GB']):.3f} GB / {2:.3f} GB")

print("\nCapacities & Tiers:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL Device DRAM cap : {cxl_dev_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL Device NAND cap  : {cxl_ssd_capacity_bytes/(1024**3):.3f} GB")
print(f"  Host DRAM layers: {[i+1 for i in sorted(list(pinned_host))]}")
print(f"  CXL Device DRAM layers : {[i+1 for i in sorted(list(pinned_cxl))]}")
print(f"  CXL Device NAND layers  : {[i+1 for i in range(len(layers)) if i not in pinned_host and i not in pinned_cxl]}")
