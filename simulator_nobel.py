import pandas as pd
import random

# ==============================
# System Parameters (RPi 5)
# ==============================
cpu_freq_hz = 2.5e9                      # 2.5 GHz
dram_peak_bw_bytes_per_sec = 17e9       # 17 GB/s peak bandwidth
dram_efficiency = 0.8                   # 80% realistic efficiency

# Derived DRAM bandwidth in bytes per CPU cycle
dram_bandwidth_bytes_per_cycle = (dram_peak_bw_bytes_per_sec * dram_efficiency) / cpu_freq_hz

# ==============================
# Model & Predictor Parameters
# ==============================
num_layers = 32
sparsity_ratio = 0.2
active_ratio = 1 - sparsity_ratio
embedding_dim = 4096
mlp_hidden_dim = 14336
attn_proj_dim = 1024

# Predictor-related latencies
predictor_cycles = 270_000_000       # Only once at Layer 0
clustering_cycles = 2_200_000        # Only once at Layer 0
dma_setup_cycles = 50_000            # One-time prefetch initiation overhead


# ==============================
# Layer Weight Calculations
# ==============================
attn_weights = (
    embedding_dim * embedding_dim +
    embedding_dim * attn_proj_dim * 2 +
    attn_proj_dim * embedding_dim
)
mlp_weights_full = (
    embedding_dim * mlp_hidden_dim * 2 +
    mlp_hidden_dim * embedding_dim
)
mlp_weights_sparse = (
    embedding_dim * mlp_hidden_dim * 2 * active_ratio +
    mlp_hidden_dim * embedding_dim
)

# ==============================
# Simulation Loop
# ==============================
layer_data = []
dram_hits = 0
dram_misses = 0
total_cycles = 0

for i in range(num_layers):
    if i == 0:
        # Layer 0: no prefetch, clustering & prediction triggered
        attn_params = attn_weights
        mlp_params = mlp_weights_full
        hit = 0
        miss = attn_params + mlp_params
        compute_flops = 2 * (attn_params + mlp_params)
        dram_miss_cycles = (miss * 4) / dram_bandwidth_bytes_per_cycle
        total_layer_cycles = compute_flops + dram_miss_cycles + clustering_cycles + predictor_cycles + dma_setup_cycles
        layer_type = "Baseline (No Prefetch)"
    elif i == 1:
        # Layer 1: prefetch was issued in Layer 0, assumed successful
        attn_params = attn_weights
        mlp_params = mlp_weights_sparse
        hit = mlp_params
        miss = attn_params
        compute_flops = 2 * (attn_params + mlp_params)
        dram_miss_cycles = (miss * 4) / dram_bandwidth_bytes_per_cycle
        total_layer_cycles = compute_flops + dram_miss_cycles
        layer_type = "Attn Miss, MLP Prefetched"
    else:
        attn_params = attn_weights
        mlp_params = mlp_weights_sparse
        hit = attn_params + mlp_params
        miss = 0
        compute_flops = 2 * (attn_params + mlp_params)
        total_layer_cycles = compute_flops  # all prefetched, no penalty
        layer_type = "Prefetch Success"
        

    dram_hits += hit
    dram_misses += miss
    total_cycles += total_layer_cycles

    layer_data.append({
        'Layer': i,
        'Execution Type': layer_type,
        'Attention Params': attn_params,
        'MLP Params': mlp_params,
        'DRAM Hit Params': hit,
        'DRAM Miss Params': miss,
        'FLOPs': compute_flops,
        'Estimated Cycles': int(total_layer_cycles)
    })

# ==============================
# Summary & Output
# ==============================
throughput_tokens_per_sec = cpu_freq_hz / total_cycles
dram_hit_ratio = dram_hits / (dram_hits + dram_misses)
dram_miss_ratio = dram_misses / (dram_hits + dram_misses)

summary = {
    'Layer': 'Total',
    'Execution Type': '---',
    'Attention Params': sum(d['Attention Params'] for d in layer_data),
    'MLP Params': sum(d['MLP Params'] for d in layer_data),
    'DRAM Hit Params': dram_hits,
    'DRAM Miss Params': dram_misses,
    'FLOPs': sum(d['FLOPs'] for d in layer_data),
    'Estimated Cycles': int(total_cycles)
}
# layer_data.append(summary)

df = pd.DataFrame(layer_data)
print(df)

summary_metrics = {
    'Layer': 'Total',
    'Execution Type': '---',
    'Attention Params': sum(d['Attention Params'] for d in layer_data),
    'MLP Params': sum(d['MLP Params'] for d in layer_data),
    'DRAM Hit Params': dram_hits,
    'DRAM Miss Params': dram_misses,
    'FLOPs': sum(d['FLOPs'] for d in layer_data),
    'Estimated Cycles': int(total_cycles),
    "DRAM BW (Bytes/Cycle)": dram_bandwidth_bytes_per_cycle,
    "DRAM Hit Ratio": dram_hit_ratio,
    "DRAM Miss Ratio": dram_miss_ratio,
    "Throughput (tokens/sec)": throughput_tokens_per_sec
}

print(summary_metrics)

import pandas as pd

# Save DataFrame and metrics to CSV
df.to_csv("NeuroCache.csv", index=False)
# OR for other scripts:
# df.to_csv("results/NeuroCache.csv", index=False)
# df.to_csv("results/FullLayerPrefetch.csv", index=False)

# Save metadata dictionary
with open("NeuroCache_20_metrics.txt", "w") as f:
    for key, value in summary_metrics.items():
        f.write(f"{key}: {value}\n")
