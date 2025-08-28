import math
import pandas as pd

GiB = 1024 ** 3

# ==============================
# CPU (all cores work on one layer at a time)
# ==============================
cpu_freq_hz = 2.4e9
cpu_cores   = 4
flops_per_cycle_per_core = 4.0
parallel_efficiency       = 0.90

# ==============================
# Memory tiers
# ==============================
host_dram_capacity_bytes = 4 * GiB      # Host DRAM: 4 GB
host_dram_peak_bw        = 17e9         # 17 GB/s

cxl_dev_dram_capacity_bytes = 4 * GiB   # CXL device DRAM cache: 4 GB
cxl_ssd_capacity_bytes      = 64 * GiB  # CXL SSD: 64 GB

cxl_link_peak_bw = 12e9                 # CXL-DRAM
ssd_effective_bw = 1.2e9                # SSD

# Latency model (per 256KB chunk)
io_chunk_bytes       = 256 * 1024
dram_chunk_latency_s = 0.2e-6           # 0.2 µs/chunk
cxl_chunk_latency_s  = 0.5e-6           # 0.5 µs/chunk
ssd_chunk_latency_s  = 10e-6            # 10 µs/chunk

# ==============================
# Model (Mistral-7B v0.2 core specs)
# ==============================
num_blocks    = 32
vocab_size    = 32000
embedding_dim = 4096
mlp_hidden_dim = 14336
q_heads = 32
kv_heads = 8
head_dim = embedding_dim // q_heads       # 128
kv_total = kv_heads * head_dim            # 1024
bytes_per_param = 4  # FP32

# ==============================
# Per-layer sizes (Mistral-7B v0.2)
# ==============================
d = embedding_dim

# Attention params: q:dxd, k:dx(kv_total), v:dx(kv_total), o:dxd -> 2*d*d + 2*d*kv_total
attn_params = (2 * d * d) + (2 * d * kv_total)
# MLP (SwiGLU): gate/up: dxh each, down: hxd -> 3*d*h
mlp_params  = (2 * d * mlp_hidden_dim) + (mlp_hidden_dim * d)

block_params = attn_params + mlp_params
block_bytes  = block_params * bytes_per_param

# Specials
embed_params = vocab_size * d
embed_bytes  = embed_params * bytes_per_param

lmhead_params = vocab_size * d          # untied in v0.2
lmhead_bytes  = lmhead_params * bytes_per_param

finalnorm_params = d
finalnorm_bytes  = finalnorm_params * bytes_per_param

# ==============================
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
def cxl_time_s(bytes_amt):  return transfer_time_s(bytes_amt, cxl_link_peak_bw,  cxl_chunk_latency_s)
def ssd_time_s(bytes_amt):  return transfer_time_s(bytes_amt, ssd_effective_bw,  ssd_chunk_latency_s)

# ==============================
# Build execution order (35 layers):
# 1: Embedding, 2..33: 32 DecoderBlocks, 34: Final RMSNorm, 35: LM Head
# ==============================
layers = []
layers.append({"name":"embed_tokens","kind":"Embedding",   "params":embed_params,   "bytes":embed_bytes,    "flops":0,               "label":"Token Embedding"})
for i in range(num_blocks):
    layers.append({"name":f"decoder_{i}","kind":"DecoderBlock","params":block_params,"bytes":block_bytes, "flops":2*block_params,"label":"Transformer Block"})
layers.append({"name":"final_norm","kind":"RMSNorm",       "params":finalnorm_params,"bytes":finalnorm_bytes,"flops":0,             "label":"Final RMSNorm"})
layers.append({"name":"lm_head","kind":"LMHead",           "params":lmhead_params,  "bytes":lmhead_bytes,   "flops":0,               "label":"LM Head"})

# ==============================
# Placement: STRICT SEQUENTIAL (Host -> CXL-DRAM -> CXL-SSD)
# ==============================
placement = [""] * len(layers)
host_free = host_dram_capacity_bytes
cxl_free  = cxl_dev_dram_capacity_bytes

def try_place_host(idx, sz):
    global host_free
    if sz <= host_free:
        placement[idx] = "DRAM"
        host_free -= sz
        return True
    return False

def try_place_cxl(idx, sz):
    global cxl_free
    if sz <= cxl_free:
        placement[idx] = "CXL-DRAM"
        cxl_free -= sz
        return True
    return False

for idx in range(len(layers)):
    sz = layers[idx]["bytes"]
    if try_place_host(idx, sz): 
        continue
    if try_place_cxl(idx, sz): 
        continue
    placement[idx] = "CXL-SSD"

def served_from(place):
    if place == "DRAM": return "Host DRAM"
    if place == "CXL-DRAM": return "CXL DRAM"
    return "CXL SSD"

# ==============================
# Execute (no prefetch, no cache)
# ==============================
rows = []
total_time_s = 0.0

dram_served_B = 0
cxl_served_B  = 0
ssd_served_B  = 0

for i, L in enumerate(layers):
    place = placement[i]
    sz = L["bytes"]
    flops = L["flops"]

    # Compute
    comp_s = compute_time_s(flops)

    # Memory (exactly one tier)
    mem_s_dram = mem_s_cxl = mem_s_ssd = 0.0
    if place == "DRAM":
        mem_s_dram = dram_time_s(sz); dram_served_B += sz
    elif place == "CXL-DRAM":
        mem_s_cxl  = cxl_time_s(sz);  cxl_served_B  += sz
    else:  # CXL-SSD
        mem_s_ssd  = ssd_time_s(sz);  ssd_served_B  += sz

    mem_total = mem_s_dram + mem_s_cxl + mem_s_ssd
    layer_time = max(comp_s, mem_total)
    total_time_s += layer_time

    rows.append({
        "Layer": i+1,
        "Name": L["name"],
        "Kind": L["kind"],
        "Placement": place,
        "Served_From": served_from(place),
        "Label": L["label"],
        "Bytes": sz,
        "Compute_s_all_cores": comp_s,
        "Mem_s_dram": mem_s_dram,
        "Mem_s_cxl":  mem_s_cxl,
        "Mem_s_ssd":  mem_s_ssd,
        "Mem_s_total": mem_total,
        "Layer_Time_s": layer_time,
    })

# ==============================
# Final Summary (includes model size & dtype)
# ==============================
df = pd.DataFrame(rows)

total_cycles = total_time_s * cpu_freq_hz
throughput_tokens_per_sec_proxy = (cpu_freq_hz * cpu_cores) / total_cycles if total_cycles > 0 else 0.0

# Model-wide totals
total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"]  for L in layers)
model_dtype_bits = bytes_per_param * 8

# Per-tier placement totals
host_bytes = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "DRAM")
cxl_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "CXL-DRAM")
ssd_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i] == "CXL-SSD")

def fmt_bytes(n): return f"{n / (1024**3):.3f} GiB"
def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

df.to_csv("cxl_simulator_no_prefetch_sequential_strict.csv", index=False)

print(df)
print("\nSummary:")
print(f"Total Time: {total_time_s:.4f} s")
print(f"Estimated Tokens/sec (proxy, {cpu_cores} cores): {throughput_tokens_per_sec_proxy:.3f}")

print("\nModel Size:")
print(f"  Total parameters: {total_params:,}  ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits}  ({bytes_per_param} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes  ({fmt_bytes(total_model_bytes)})")

print("\nPlacement Breakdown (by bytes):")
print(f"  Host DRAM: {host_bytes:,}  ({fmt_bytes(host_bytes)})")
print(f"  CXL-DRAM : {cxl_bytes:,}  ({fmt_bytes(cxl_bytes)})")
print(f"  CXL-SSD  : {ssd_bytes:,}  ({fmt_bytes(ssd_bytes)})")

print("\nRuntime Traffic Served (actual this run):")
print(f"  Host DRAM bytes served: {dram_served_B:,}")
print(f"  CXL DRAM bytes served : {cxl_served_B:,}")
print(f"  CXL SSD bytes served  : {ssd_served_B:,}")

print("\nCapacities:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL-DRAM cap : {cxl_dev_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL-SSD cap  : {cxl_ssd_capacity_bytes/(1024**3):.3f} GB")
