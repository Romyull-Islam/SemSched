# simulator_cxl_pipeline_parallel.py
import math
import pandas as pd

GiB = 1024 ** 3


# Global knobs (identical)
# ==============================
TOKENS = 16     # for total-time reporting
P = 4           # total CPU cores

cpu_freq_hz = 2.4e9
flops_per_cycle_per_core = 4.0
parallel_efficiency       = 0.90

def compute_time_s(flops, cores):
    if flops <= 0 or cores <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

# Memory tiers
host_dram_capacity_bytes = 4 * GiB
host_dram_peak_bw        = 17e9
cxl_dev_dram_capacity_bytes = 4 * GiB
cxl_ssd_capacity_bytes      = 64 * GiB
cxl_link_peak_bw = 12e9
ssd_effective_bw = 1.2e9

io_chunk_bytes       = 256 * 1024
dram_chunk_latency_s = 0.2e-6
cxl_chunk_latency_s  = 0.5e-6
ssd_chunk_latency_s  = 10e-6

def transfer_time_s(nbytes, bw_Bps, chunk_lat_s):
    if nbytes <= 0: return 0.0
    chunks = math.ceil(nbytes / io_chunk_bytes)
    return (nbytes / bw_Bps) + chunks * chunk_lat_s

def dram_time_s(n): return transfer_time_s(n, host_dram_peak_bw, dram_chunk_latency_s)
def cxl_time_s(n):  return transfer_time_s(n, cxl_link_peak_bw,  cxl_chunk_latency_s)
def ssd_time_s(n):  return transfer_time_s(n, ssd_effective_bw,  ssd_chunk_latency_s)


# Model (Mistral-7B v0.2 FP32)
# ==============================
num_blocks    = 32
vocab_size    = 32000
embedding_dim = 4096
mlp_hidden_dim = 14336
q_heads = 32
kv_heads = 8
head_dim = embedding_dim // q_heads
kv_total = kv_heads * head_dim
bytes_per_param = 4

d = embedding_dim
attn_params = (2 * d * d) + (2 * d * kv_total)
mlp_params  = (2 * d * mlp_hidden_dim) + (mlp_hidden_dim * d)
block_params = attn_params + mlp_params
block_bytes  = block_params * bytes_per_param

embed_params = vocab_size * d
embed_bytes  = embed_params * bytes_per_param
lmhead_params = vocab_size * d
lmhead_bytes  = lmhead_params * bytes_per_param
finalnorm_params = d
finalnorm_bytes  = finalnorm_params * bytes_per_param

# Build layers
layers = []
layers.append({"name":"embed_tokens","kind":"Embedding",   "params":embed_params,    "bytes":embed_bytes,     "flops":0})
for i in range(num_blocks):
    layers.append({"name":f"decoder_{i}","kind":"DecoderBlock","params":block_params,"bytes":block_bytes,     "flops":2*block_params})
layers.append({"name":"final_norm","kind":"RMSNorm",       "params":finalnorm_params,"bytes":finalnorm_bytes, "flops":0})
layers.append({"name":"lm_head","kind":"LMHead",           "params":lmhead_params,   "bytes":lmhead_bytes,    "flops":0})

# Placement: strict Host -> CXL-DRAM -> CXL-SSD (identical policy)
placement = [""] * len(layers)
host_free = host_dram_capacity_bytes
cxl_free  = cxl_dev_dram_capacity_bytes
for idx, L in enumerate(layers):
    sz = L["bytes"]
    if sz <= host_free:
        placement[idx] = "DRAM";       host_free -= sz
    elif sz <= cxl_free:
        placement[idx] = "CXL-DRAM";   cxl_free  -= sz
    else:
        placement[idx] = "CXL-SSD"

# Stage split: Stage1 = DRAM-resident layers; Stage2 = everything else
stage = []
s1_idx = []; s2_idx = []
for i,p in enumerate(placement):
    if p == "DRAM":
        stage.append(1); s1_idx.append(i)
    else:
        stage.append(2); s2_idx.append(i)

# Per-token time for a stage (sum over its layers)
def stage_time(idx_list, cores):
    total = 0.0
    for k in idx_list:
        L = layers[k]; sz=L["bytes"]; fl=L["flops"]; plc=placement[k]
        comp_s = compute_time_s(fl, cores)
        if plc == "DRAM":
            mem_s = dram_time_s(sz)
        elif plc == "CXL-DRAM":
            mem_s = cxl_time_s(sz)
        else:
            mem_s = ssd_time_s(sz)
        total += max(comp_s, mem_s)
    return total

# Search p1/p2 to minimize bottleneck
best = None
for p1 in range(1, P):
    p2 = P - p1
    S1 = stage_time(s1_idx, p1)
    S2 = stage_time(s2_idx, p2)
    cand = (max(S1, S2), p1, p2, S1, S2)
    if best is None or cand < best:
        best = cand
bottleneck, p1, p2, S1, S2 = best

# --------- Timeline simulator (confirms overlap & utilization) ---------
def run_pipeline_timeline(S1, S2, p1, p2, tokens=16, lanes2=1):
    """
    Deterministic 2-stage pipeline with Stage-1 (one lane) and Stage-2 (lanes2 lanes).
    Produces makespan, per-stage busy time, overall CPU utilization, and key timestamps.
    """
    # Stage-1: sequential (one lane)
    s1_start = [0.0]*tokens
    s1_end   = [0.0]*tokens
    for t in range(tokens):
        s1_start[t] = s1_end[t-1] if t>0 else 0.0
        s1_end[t]   = s1_start[t] + S1

    # Stage-2: parallel servers (lanes2)
    lane_free = [0.0]*lanes2
    s2_start = [0.0]*tokens
    s2_end   = [0.0]*tokens
    for t in range(tokens):
        lane = min(range(lanes2), key=lambda i: lane_free[i])
        s2_start[t] = max(s1_end[t], lane_free[lane])
        s2_end[t]   = s2_start[t] + S2
        lane_free[lane] = s2_end[t]

    makespan = s2_end[-1]
    s1_busy = tokens * S1
    s2_busy = tokens * S2

    # Core-seconds used vs available
    core_seconds_used = (p1 * s1_busy) + (p2 * s2_busy)
    core_seconds_cap  = (p1 + p2) * makespan
    overall_util      = core_seconds_used / core_seconds_cap if core_seconds_cap>0 else 0.0

    return {
        "makespan": makespan,
        "s1_busy": s1_busy,
        "s2_busy": s2_busy,
        "overall_util": overall_util,
        "s1_finish_all_tokens": s1_end[-1],
        "s2_finish_last_token": s2_end[-1],
        "s1_starts": s1_start, "s1_ends": s1_end,
        "s2_starts": s2_start, "s2_ends": s2_end
    }

# Steady-state metrics (classic 2-stage pipe)
throughput_steady = 1.0 / bottleneck
pipeline_total_time = S1 + S2 + (TOKENS - 1) * bottleneck

# Timeline (lanes2=1 for a fair 2-stage comparison)
timeline = run_pipeline_timeline(S1, S2, p1, p2, tokens=TOKENS, lanes2=1)

# --------- Per-layer table (tired-style columns, prefetch fields zeroed) ---------
rows = []
for i,L in enumerate(layers):
    sz=L["bytes"]; fl=L["flops"]; plc=placement[i]
    st = 1 if i in s1_idx else 2
    cores = p1 if st==1 else p2
    comp_s = compute_time_s(fl, cores)
    if plc == "DRAM":
        mem_d, mem_c, mem_s = dram_time_s(sz), 0.0, 0.0
        served_from = "Host DRAM"
    elif plc == "CXL-DRAM":
        mem_d, mem_c, mem_s = 0.0, cxl_time_s(sz), 0.0
        served_from = "CXL DRAM"
    else:
        mem_d, mem_c, mem_s = 0.0, 0.0, ssd_time_s(sz)
        served_from = "CXL SSD"
    layer_time = max(comp_s, (mem_d + mem_c + mem_s))
    rows.append({
        "Layer": i+1,
        "Name": L["name"],
        "Kind": L["kind"],
        "Placement": plc,
        "Stage": st,
        "Served_From": served_from,
        "Bytes": sz,
        "Compute_s (on stage cores)": comp_s,
        "Mem_s_dram": mem_d,
        "Mem_s_cxl":  mem_c,
        "Mem_s_ssd (stall)": mem_s,
        "Mem_s_total": mem_d + mem_c + mem_s,
        "Layer_Time_s": layer_time,
        # Prefetch-related columns kept for reporting parity; zeroed here
        "CXL_Hit_Bytes": 0,
        "CXL_Miss_Bytes": 0,
        "Prefetch_Source": "None",
        "Prefetch_Next_Layer": None,
        "Prefetch_Bytes": 0,
        "Prefetch_Time_s": 0.0,
        "CXL_Cache_Used_GB": 0.0,
    })

df = pd.DataFrame(rows)
df.to_csv("sim_pipeline_2stage.csv", index=False)
print(df)

# Uniform “tired-style” summary + pipeline & utilization metrics
total_params = sum(L["params"] for L in layers)
total_model_bytes = sum(L["bytes"]  for L in layers)
model_dtype_bits = bytes_per_param * 8

host_bytes = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i]=="DRAM")
cxl_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i]=="CXL-DRAM")
ssd_bytes  = sum(layers[i]["bytes"] for i in range(len(layers)) if placement[i]=="CXL-SSD")

def fmt_bytes(n): return f"{n/(1024**3):.3f} GiB"
def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

print("\nSummary (Pipeline 2-Stage):")
print(f"Core split search over P={P} → p1={p1} (Stage1/DRAM), p2={p2} (Stage2/CXL)")
print(f"Per-token S1={S1:.6f} s, S2={S2:.6f} s, bottleneck={bottleneck:.6f} s, throughput={throughput_steady:.6f} tok/s")
print(f"Total time for T={TOKENS}: {pipeline_total_time:.6f} s")

print("\n=== Timeline & CPU Utilization ===")
print(f"Stage-1 finishes all tokens at: {timeline['s1_finish_all_tokens']:.6f} s")
print(f"Stage-2 finishes last token at: {timeline['s2_finish_last_token']:.6f} s")
print(f"Makespan: {timeline['makespan']:.6f} s")
print(f"Stage-1 busy time: {timeline['s1_busy']:.6f} s on {p1} core(s) "
      f"→ util {(timeline['s1_busy']/timeline['makespan'])*100:.1f}%")
print(f"Stage-2 busy time: {timeline['s2_busy']:.6f} s on {p2} core(s) "
      f"→ util {(timeline['s2_busy']/timeline['makespan'])*100:.1f}%")
print(f"Overall CPU utilization: {timeline['overall_util']*100:.1f}%")

print("\nModel Size:")
print(f"  Total parameters: {total_params:,}  ({fmt_params(total_params)})")
print(f"  Dtype: FP{model_dtype_bits}  ({bytes_per_param} bytes/param)")
print(f"  Total model size: {total_model_bytes:,} bytes  ({fmt_bytes(total_model_bytes)})")

print("\nPlacement Breakdown (by bytes):")
print(f"  Host DRAM: {host_bytes:,}  ({fmt_bytes(host_bytes)})")
print(f"  CXL-DRAM : {cxl_bytes:,}  ({fmt_bytes(cxl_bytes)})")
print(f"  CXL-SSD  : {ssd_bytes:,}  ({fmt_bytes(ssd_bytes)})")

print("\nRuntime Traffic Served (per token, by residence):")
print(f"  Host DRAM bytes served: {sum(df[df.Placement=='DRAM']['Bytes']):,}")
print(f"  CXL DRAM (hit) bytes served: {int(df['CXL_Hit_Bytes'].sum()):,}")   # 0 here
print(f"  CXL SSD (miss) bytes served: {int(df['CXL_Miss_Bytes'].sum()):,}")  # 0 here
print(f"  Peak CXL cache use: 0.000 GB / {cxl_dev_dram_capacity_bytes/(1024**3):.3f} GB")

print("\nCapacities & Tiers:")
print(f"  Host DRAM cap: {host_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL-DRAM cap : {cxl_dev_dram_capacity_bytes/(1024**3):.3f} GB")
print(f"  CXL-SSD cap  : {cxl_ssd_capacity_bytes/(1024**3):.3f} GB")
print(f"  Host DRAM layers: {[i+1 for i in s1_idx]}")
print(f"  CXL DRAM layers : {[i+1 for i in range(len(layers)) if placement[i]=='CXL-DRAM']}")
print(f"  CXL SSD layers  : {[i+1 for i in range(len(layers)) if placement[i]=='CXL-SSD']}")
