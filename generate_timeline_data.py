import pandas as pd
import os
from tiers import transfer_time_s, CXL_DRAM, HOST_DRAM
from model_cfg import build_layers, Qwen2_5_72BCfg
from sim_cfg import cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency

def compute_time_s(flops):
    if flops <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

# Mimics your decomposed_build_layers from semduplex_scheduler
def get_decomposed_layers(bpp):
    standard_layers = build_layers(Qwen2_5_72BCfg(), sequence_length=512)
    new_layers = []
    for i, L in enumerate(standard_layers):
        if L["kind"] == "DecoderBlock":
            raw_bytes = int((L["bytes"] / 4.0) * bpp)
            raw_flops = L["flops"]
            
            # Attention Sub-layer (Pinned in Host DRAM)
            new_layers.append({
                "name": f"Layer_{i}_Attn",
                "kind": "Attention",
                "bytes": int(raw_bytes * 0.17 * 2),
                "flops": int(raw_flops * 0.33),
                "kv_bytes": 2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * bpp,
                "sparsity": 0.05,
                "placement": "Host_DRAM" # SemDuplex pins Attention to Host
            })
            
            # MLP Sub-layer (Streamed from CXL DRAM)
            new_layers.append({
                "name": f"Layer_{i}_MLP",
                "kind": "MLP",
                "bytes": int(raw_bytes * 0.83 * 2),
                "flops": int(raw_flops * 0.67),
                "kv_bytes": 0,
                "sparsity": 0.54, # Dense MLP
                "placement": "CXL_DRAM" # SemDuplex streams dense MLP from CXL
            })
    return new_layers

def generate_timeline_data():
    B = 128
    NUM_TOKENS = 3 # Generate 3 tokens for the zoomed plot
    QUANT = "int8" # Use INT8 to trigger the Duplex regime
    bpp = 1.0 
    
    cxl_bw_gbps = CXL_DRAM.bw_Bps / 1e9
    timeline_records = []

    # -----------------------------------------------------------------
    # 1. BASELINE SIMPLEX (Monolithic blocks, CXL only)
    # -----------------------------------------------------------------
    standard_layers = build_layers(Qwen2_5_72BCfg(), sequence_length=512)
    current_time = 0.0
    
    for token in range(1, NUM_TOKENS + 1):
        for idx, L in enumerate(standard_layers):
            if L["kind"] != "DecoderBlock": continue
                
            layer_name = f"Layer_{idx}"
            layer_bytes = int((L["bytes"] / 4.0) * bpp)
            kv_bytes = 2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * bpp * B
            
            read_time = transfer_time_s(layer_bytes, CXL_DRAM)
            timeline_records.append({
                "Simulator": "Baseline_Simplex", "Token": token, "Layer": layer_name,
                "Event": "Weight_Read", "Start_Time_s": current_time,
                "End_Time_s": current_time + read_time, "Bandwidth_GBps": cxl_bw_gbps
            })
            current_time += read_time
            
            write_time = transfer_time_s(kv_bytes, CXL_DRAM)
            timeline_records.append({
                "Simulator": "Baseline_Simplex", "Token": token, "Layer": layer_name,
                "Event": "KV_Write_Stall", "Start_Time_s": current_time,
                "End_Time_s": current_time + write_time, "Bandwidth_GBps": 0.0
            })
            current_time += write_time

    # -----------------------------------------------------------------
    # 2. SEMDUPLEX (Decomposed layers, Host vs CXL, Duplexing)
    # -----------------------------------------------------------------
    decomposed = get_decomposed_layers(bpp)
    current_time = 0.0
    DUPLEX_PENALTY = 1.15
    
    for token in range(1, NUM_TOKENS + 1):
        for L in decomposed:
            sz = L["bytes"]
            kv_total = L["kv_bytes"] * B
            eff_flops = int(L["flops"] * (1.0 - L["sparsity"]))
            comp_time = compute_time_s(eff_flops)
            
            # Determine Read source and time
            if L["placement"] == "Host_DRAM":
                mem_time = transfer_time_s(sz, HOST_DRAM)
                active_cxl_read_bw = 0.0 # Reading from DDR5, CXL link is IDLE!
            else:
                mem_time = transfer_time_s(sz, CXL_DRAM)
                active_cxl_read_bw = cxl_bw_gbps # Reading from CXL
                
            ltime_base = max(comp_time, mem_time)
            
            # Apply duplex penalty ONLY if writing to CXL while reading from CXL
            actual_time = ltime_base * DUPLEX_PENALTY if (kv_total > 0 and active_cxl_read_bw > 0) else ltime_base
            
            # 1. READ EVENT (CXL BW will drop to 0 when reading from Host DRAM)
            timeline_records.append({
                "Simulator": "SemDuplex", "Token": token, "Layer": L["name"],
                "Event": f"{L['kind']}_Read", "Start_Time_s": current_time,
                "End_Time_s": current_time + actual_time, "Bandwidth_GBps": active_cxl_read_bw
            })
            
            # 2. WRITE EVENT (Negative CXL BW, happens during Attention layers)
            if kv_total > 0:
                write_time = transfer_time_s(kv_total, CXL_DRAM)
                timeline_records.append({
                    "Simulator": "SemDuplex", "Token": token, "Layer": L["name"],
                    "Event": "KV_Write_Background", "Start_Time_s": current_time, 
                    "End_Time_s": current_time + write_time, "Bandwidth_GBps": -cxl_bw_gbps
                })
            
            current_time += actual_time

    df = pd.DataFrame(timeline_records)
    df.to_csv("duplex_timeline_data.csv", index=False)
    print(">>> True SemDuplex timeline data generated with sub-layers and Host/CXL awareness.")

if __name__ == "__main__":
    generate_timeline_data()

    