import math
import pandas as pd
from collections import OrderedDict

GiB = 1024 ** 3


# CPU (all cores work on one layer at a time)
# ==============================
cpu_freq_hz = 2.4e9
cpu_cores   = 4
flops_per_cycle_per_core = 4.0
parallel_efficiency       = 0.90

# Memory tiers
# ==============================
host_dram_capacity_bytes = 4 * GiB      # Host DRAM: 4 GB
host_dram_peak_bw = 17e9                # 17 GB/s

cxl_dev_dram_capacity_bytes = 4 * GiB   # CXL device DRAM cache: 4 GB
cxl_ssd_capacity_bytes      = 64 * GiB  # CXL SSD: 64 GB

cxl_link_peak_bw = 12e9                 # CXL-DRAM hits
ssd_effective_bw = 1.2e9                # SSD misses

# Latency model (per 256KB chunk)
io_chunk_bytes = 256 * 1024
cxl_chunk_latency_s = 0.5e-6            # 0.5 us per chunk (hit path)
ssd_chunk_latency_s = 10e-6             # 10 us per chunk (miss path)
dram_chunk_latency_s = 0.2e-6           # tiny, near-negligible

# Prefetch
prefetch_lookahead = 1
prefetch_gran_bytes = io_chunk_bytes
prefetch_share_when_current_is_cxl = 0.30


# Model (Mistral-7B v0.2 core specs)
# ==============================
num_blocks = 32               # decoder blocks
vocab_size = 32000
embedding_dim = 4096
mlp_hidden_dim = 14336
q_heads = 32
kv_heads = 8
head_dim = embedding_dim // q_heads       # 128
kv_total = kv_heads * head_dim            # 1024
bytes_per_param = 4  # FP32


# Per-layer sizes (CORRECT for Mistral)
# ==============================
d = embedding_dim

# Attention params per block:
# q: dxd, k: dx(kv_total), v: dx(kv_total), o: dxd  -> 2*d*d + 2*d*kv_total
attn_params = (2 * d * d) + (2 * d * kv_total)

# MLP (SwiGLU): gate/up: dxh each, down: hxd -> 2*d*h + h*d
mlp_params  = (2 * d * mlp_hidden_dim) + (mlp_hidden_dim * d)

block_params = attn_params + mlp_params
block_bytes  = block_params * bytes_per_param

# Special layers
embed_params = vocab_size * d
embed_bytes  = embed_params * bytes_per_param

lmhead_params = vocab_size * d      # v0.2 untied
lmhead_bytes  = lmhead_params * bytes_per_param

finalnorm_params = d                # tiny
finalnorm_bytes  = finalnorm_params * bytes_per_param


# Helpers (bandwidth + per-chunk latency)
# ==============================
def compute_time_s(flops):
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s if flops > 0 else 0.0

def transfer_time_s(bytes_amt, bw_Bps, chunk_latency_s):
    if bytes_amt <= 0: return 0.0
    chunks = math.ceil(bytes_amt / io_chunk_bytes)
    return (bytes_amt / bw_Bps) + chunks * chunk_latency_s

def dram_time_s(bytes_amt): return transfer_time_s(bytes_amt, host_dram_peak_bw, dram_chunk_latency_s)
def cxl_time_s(bytes_amt):  return transfer_time_s(bytes_amt, cxl_link_peak_bw, cxl_chunk_latency_s)
def ssd_time_s(bytes_amt):  return transfer_time_s(bytes_amt, ssd_effective_bw, ssd_chunk_latency_s)

# ==============================
# Build execution order as real layers
# 1: Embedding
# 2..33: 32 decoder blocks
# 34: Final RMSNorm
# 35: LM Head
# ==============================
layers = []
layers.append({"name":"embed_tokens","kind":"Embedding","params":embed_params,"bytes":embed_bytes,"flops":0,"label":"Token Embedding"})
for i in range(num_blocks):
    layers.append({"name":f"decoder_{i}","kind":"DecoderBlock","params":block_params,"bytes":block_bytes,"flops":2*block_params,"label":"Transformer Block"})
layers.append({"name":"final_norm","kind":"RMSNorm","params":finalnorm_params,"bytes":finalnorm_bytes,"flops":0,"label":"Final RMSNorm"})
layers.append({"name":"lm_head","kind":"LMHead","params":lmhead_params,"bytes":lmhead_bytes,"flops":0,"label":"LM Head"})

# ==============================
# Placement policy:
# - Pin Embedding + LM Head in Host DRAM first
# - Then place remaining sequentially: Host DRAM -> CXL-DRAM (pinned) -> CXL-SSD
# ==============================
pinned_host = set()
pinned_cxl  = set()
host_free = host_dram_capacity_bytes
cxl_free  = cxl_dev_dram_capacity_bytes

def try_pin_host(idx, bytes_amt):
    global host_free
    if bytes_amt <= host_free:
        pinned_host.add(idx); host_free -= bytes_amt
        return True
    return False

def try_pin_cxl(idx, bytes_amt):
    global cxl_free
    if bytes_amt <= cxl_free:
        pinned_cxl.add(idx); cxl_free -= bytes_amt
        return True
    return False

# Pin hot specials in Host DRAM
assert try_pin_host(0, layers[0]["bytes"]), "Embedding does not fit in host DRAM"
assert try_pin_host(len(layers)-1, layers[-1]["bytes"]), "LM Head does not fit in host DRAM"

# Place remaining layers sequentially
for idx in range(1, len(layers)-1):
    b = layers[idx]["bytes"]
    if idx not in pinned_host and idx not in pinned_cxl:
        if try_pin_host(idx, b): continue
        if try_pin_cxl(idx, b):  continue
        # else falls to SSD

def placement_of(idx):
    if idx in pinned_host: return "DRAM(pinned)"
    if idx in pinned_cxl:  return "CXL-DRAM(pinned)"
    return "CXL-SSD"

# ==============================
# CXL cache for prefetch (SSD -> CXL-DRAM)
# ==============================
class CxlCache:
    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.used = 0
        self.entries = OrderedDict()  # idx -> bytes_cached
    def cached_bytes(self, idx): return self.entries.get(idx, 0)
    def add_bytes(self, idx, add_b):
        if add_b <= 0: return 0
        while self.used + add_b > self.capacity and self.entries:
            _, old_b = self.entries.popitem(last=False)
            self.used -= old_b
        cur = self.entries.get(idx, 0)
        new = cur + add_b
        self.entries[idx] = new
        self.entries.move_to_end(idx, last=True)
        self.used += add_b
        return add_b
    def reserve_exact(self, idx, bytes_amt):
        have = self.cached_bytes(idx)
        need = max(0, bytes_amt - have)
        if need > 0:
            self.add_bytes(idx, need)
    def consume(self, idx):
        if idx in self.entries:
            b = self.entries.pop(idx)
            self.used -= b
            return b
        return 0

cxl_cache = CxlCache(cxl_dev_dram_capacity_bytes)

# Preload pinned CXL layers fully
for idx in range(len(layers)):
    if placement_of(idx) == "CXL-DRAM(pinned)":
        cxl_cache.reserve_exact(idx, layers[idx]["bytes"])

# ==============================
# Execute (sequential)
# ==============================
rows = []
total_time_s = 0.0
dram_served_B = 0
cxl_hit_served_B = 0
cxl_miss_served_B = 0

for exec_idx in range(len(layers)):
    L = layers[exec_idx]
    place = placement_of(exec_idx)
    L_bytes = L["bytes"]
    flops = L["flops"]

    # Compute time
    comp_time = compute_time_s(flops)

    # Memory service breakdown
    mem_s_dram = mem_s_cxl = mem_s_ssd = 0.0
    cxl_hit_b = cxl_miss_b = 0

    if place == "DRAM(pinned)":
        mem_s_dram = dram_time_s(L_bytes)
        dram_served_B += L_bytes
        served_from = "Host DRAM"

    elif place == "CXL-DRAM(pinned)":
        hit_b  = min(L_bytes, cxl_cache.cached_bytes(exec_idx))
        miss_b = max(0, L_bytes - hit_b)  # should be 0
        mem_s_cxl = cxl_time_s(hit_b)
        mem_s_ssd = ssd_time_s(miss_b)
        cxl_hit_served_B  += hit_b
        cxl_miss_served_B += miss_b
        cxl_hit_b, cxl_miss_b = hit_b, miss_b
        served_from = "CXL DRAM (pinned hit)"

    else:  # CXL-SSD resident
        cached = cxl_cache.cached_bytes(exec_idx)
        hit_b  = min(L_bytes, cached)
        miss_b = max(0, L_bytes - hit_b)
        mem_s_cxl = cxl_time_s(hit_b)
        mem_s_ssd = ssd_time_s(miss_b)   # <-- this is your **stall** if prefetch didn't finish
        cxl_hit_served_B  += hit_b
        cxl_miss_served_B += miss_b
        if hit_b > 0: cxl_cache.consume(exec_idx)
        cxl_hit_b, cxl_miss_b = hit_b, miss_b
        served_from = "CXL (hit+miss)"

    t_mem = mem_s_dram + mem_s_cxl + mem_s_ssd
    layer_time_if_no_overlap = max(comp_time, t_mem)

    # ---------- Prefetch NEXT SSD-resident layer (SSD -> CXL-DRAM) ----------
    next_idx = exec_idx + 1
    prefetch_bytes = 0
    prefetch_time_s = 0.0
    prefetch_src = "None"
    prefetch_bw = ssd_effective_bw if place.startswith("DRAM") else ssd_effective_bw * prefetch_share_when_current_is_cxl
    budget_s = layer_time_if_no_overlap

    if next_idx < len(layers) and placement_of(next_idx) == "CXL-SSD" and budget_s > 0:
        prefetch_src = "SSD→CXL-DRAM"
        need = layers[next_idx]["bytes"] - cxl_cache.cached_bytes(next_idx)
        while need > 0 and budget_s > 0:
            chunk = min(prefetch_gran_bytes, need)
            # time for one chunk from SSD into CXL (bandwidth + latency)
            t_chunk = (chunk / prefetch_bw) + ssd_chunk_latency_s
            if t_chunk <= budget_s:
                cxl_cache.add_bytes(next_idx, chunk)
                prefetch_bytes += chunk
                prefetch_time_s += t_chunk
                need -= chunk
                budget_s -= t_chunk
            else:
                # partial fill with remaining time (bandwidth only approx)
                possible = max(0, int((budget_s - ssd_chunk_latency_s) * prefetch_bw))
                if possible > 0:
                    cxl_cache.add_bytes(next_idx, possible)
                    prefetch_bytes += possible
                    prefetch_time_s += budget_s
                break

    # Final overlapped layer time
    layer_time = layer_time_if_no_overlap
    total_time_s += layer_time

    rows.append({
        "Layer": exec_idx + 1,                 # 1-based index
        "Name": L["name"],
        "Kind": L["kind"],
        "Placement": place,
        "Served_From": served_from,
        "Label": L["label"],
        "Bytes": L_bytes,
        "Compute_s_all_cores": comp_time,
        "Mem_s_dram": mem_s_dram,
        "Mem_s_cxl": mem_s_cxl,
        "Mem_s_ssd (stall)": mem_s_ssd,       # SSD part is the stall if prefetch fell short
        "Mem_s_total": t_mem,
        "Layer_Time_s": layer_time,
        "CXL_Hit_Bytes": cxl_hit_b,
        "CXL_Miss_Bytes": cxl_miss_b,
        "Prefetch_Source": prefetch_src,
        "Prefetch_Next_Layer": (next_idx + 1) if prefetch_src != "None" else None,
        "Prefetch_Bytes": prefetch_bytes,
        "Prefetch_Time_s": prefetch_time_s,
        "CXL_Cache_Used_GB": round(cxl_cache.used / GiB, 3),
    })

# ==============================
# Final Summary (enhanced: includes model size & dtype)
# ==============================
df = pd.DataFrame(rows)
total_cycles = total_time_s * cpu_freq_hz
throughput_tokens_per_sec_proxy = (cpu_freq_hz * cpu_cores) / total_cycles if total_cycles > 0 else 0.0

# Model-wide totals
total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"]  for L in layers)
model_dtype_bits = bytes_per_param * 8

# Placement totals
pinned_host_total = sum(layers[i]["bytes"] for i in pinned_host)
pinned_cxl_total  = sum(layers[i]["bytes"] for i in pinned_cxl)
ssd_total         = total_model_bytes - pinned_host_total - pinned_cxl_total

def fmt_bytes(n):
    return f"{n/ (1024**3):.3f} GiB"

def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

df.to_csv("cxl_simulator_embed_lmhead_host_with_prefetch_latency.csv", index=False)

print(df)
print("\nSummary:")
print(f"Total Time: {total_time_s:.4f} s")
print(f"Estimated Tokens/sec (proxy, {cpu_cores} cores): {throughput_tokens_per_sec_proxy:.3f}")

# Model size & dtype
print("\nModel Size:")
print(f"  Total parameters: {total_params:,}  ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits}  ({bytes_per_param} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes  ({fmt_bytes(total_model_bytes)})")

# Per-tier placement breakdown
print("\nPlacement Breakdown (by bytes):")
print(f"  Host DRAM pinned: {pinned_host_total:,}  ({fmt_bytes(pinned_host_total)})")
print(f"  CXL-DRAM pinned : {pinned_cxl_total:,}  ({fmt_bytes(pinned_cxl_total)})")
print(f"  CXL-SSD         : {ssd_total:,}  ({fmt_bytes(ssd_total)})")

# Runtime traffic breakdown actually served
print("\nRuntime Traffic Served (actual during this run):")
print(f"  Host DRAM bytes served: {sum(df[df.Placement=='DRAM(pinned)']['Bytes']):,}")
print(f"  CXL DRAM (hit) bytes served: {int(df['CXL_Hit_Bytes'].sum()):,}")
print(f"  CXL SSD (miss) bytes served: {int(df['CXL_Miss_Bytes'].sum()):,}")
print(f"  Peak CXL cache use: {max([r['CXL_Cache_Used_GB'] for r in rows]):.3f} GB / {cxl_dev_dram_capacity_bytes/ (1024**3):.3f} GB")

# Capacities and which layers landed where (1-based indices as printed in the table)
print("\nCapacities & Pins:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/ (1024**3):.3f} GB   | Pinned: {pinned_host_total/ (1024**3):.3f} GB")
print(f"  CXL-DRAM cap : {cxl_dev_dram_capacity_bytes/ (1024**3):.3f} GB | Pinned: {pinned_cxl_total/ (1024**3):.3f} GB")
print(f"  Host DRAM pinned layers: {[i+1 for i in sorted(list(pinned_host))]}")
print(f"  CXL DRAM pinned layers : {[i+1 for i in sorted(list(pinned_cxl))]}")
print(f"  CXL SSD layers         : {[i+1 for i in range(len(layers)) if i not in pinned_host and i not in pinned_cxl]}")
