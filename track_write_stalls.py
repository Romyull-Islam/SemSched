import pandas as pd
import os
from tiers import transfer_time_s, Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S, CXL_DRAM, CXL_SSD_NAND
from model_cfg import build_layers, Qwen2_5_72BCfg
from sim_cfg import cpu_freq_hz, cpu_cores, flops_per_cycle_per_core, parallel_efficiency

def compute_time_s(flops):
    if flops <= 0: return 0.0
    flops_per_s = cpu_freq_hz * cpu_cores * flops_per_cycle_per_core * parallel_efficiency
    return flops / flops_per_s

def track_write_stalls():
    print("\n" + "=" * 70)
    print("TRACKING EXACT WRITE STALLS PER TOKEN (Qwen 72B)")
    print("=" * 70)

    results = []
    batch_sizes = [1, 4, 16, 64, 128]
    quants = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}
    
    ssd_tier = Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S)

    for q_name, bpp in quants.items():
        layers = build_layers(Qwen2_5_72BCfg(), sequence_length=512)
        
        for b in batch_sizes:
            flexgen_ws = 0.0
            lia_ws = 0.0
            llmflash_ws = 0.0
            semduplex_ws = 0.0
            
            is_duplex = q_name in ["int4", "int8"]

            for L in layers:
                if L["kind"] == "DecoderBlock":
                    comp_s = compute_time_s(L["flops"])
                    layer_bytes = int((L["bytes"] / 4.0) * bpp)
                    
                    # 1. KV Cache base size
                    kv_per_seq = 2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * bpp
                    
                    # --------------------------------------------------
                    # 1. FlexGen Write Stall (SSD Demand Paging)
                    # FlexGen writes full historical KV cache + new token
                    # --------------------------------------------------
                    flexgen_kv_io = kv_per_seq * (512 + 1) * b
                    mem_no_kv = transfer_time_s(layer_bytes, ssd_tier)
                    mem_with_kv = transfer_time_s(layer_bytes + flexgen_kv_io, ssd_tier)
                    flexgen_ws += max(comp_s, mem_with_kv) - max(comp_s, mem_no_kv)
                    
                    # --------------------------------------------------
                    # 2. LIA Write Stall (NUMA Tiering to NAND)
                    # LIA writes single token KV to NAND
                    # --------------------------------------------------
                    lia_kv_io = kv_per_seq * b
                    mem_no_kv_lia = transfer_time_s(layer_bytes, CXL_SSD_NAND)
                    mem_with_kv_lia = transfer_time_s(layer_bytes + lia_kv_io, CXL_SSD_NAND)
                    lia_ws += max(comp_s, mem_with_kv_lia) - max(comp_s, mem_no_kv_lia)
                    
                    # --------------------------------------------------
                    # 3. LLMFlash Write Stall (Streaming to CXL DRAM)
                    # Synchronous KV write per sequence in batch
                    # --------------------------------------------------
                    llmflash_ws += b * transfer_time_s(kv_per_seq, CXL_DRAM)
                    
                    # --------------------------------------------------
                    # 4. SemDuplex Write Stall (Duplex Masking)
                    # Opportunistic writes to idle lanes
                    # --------------------------------------------------
                    attn_bytes = layer_bytes * 0.17 * 2
                    attn_comp = compute_time_s(L["flops"] * 0.33)
                    attn_mem = transfer_time_s(attn_bytes, CXL_DRAM)
                    ltime_base = max(attn_comp, attn_mem)
                    
                    if is_duplex:
                        # Write is hidden behind read, but incurs hardware turnaround penalty (1.15x)
                        # The stall is ONLY the penalty difference
                        semduplex_ws += ltime_base * (1.15 - 1.0)
                    else:
                        # Bus is saturated (FP32/FP16), writes cannot be hidden
                        extra_kv_t = (b - 1) * transfer_time_s(kv_per_seq, CXL_DRAM)
                        semduplex_ws += extra_kv_t

            results.append({
                "Model": "72B",
                "Quant": q_name.upper(),
                "BatchSize": b,
                "FlexGen_Write_Stall_s": flexgen_ws,
                "LIA_Write_Stall_s": lia_ws,
                "LLMFlash_Write_Stall_s": llmflash_ws,
                "SemDuplex_Write_Stall_s": semduplex_ws
            })
            
            if b == 128:
                print(f"Quant: {q_name.upper():<5} | B=128 | SemDuplex WS: {semduplex_ws:.4f}s | LLMFlash WS: {llmflash_ws:.4f}s")

    df = pd.DataFrame(results)
    csv_file = "exact_write_stalls.csv"
    df.to_csv(csv_file, index=False)
    
    print("=" * 70)
    print(f">>> Successfully calculated isolated write stalls and saved to {csv_file}")
    print("=" * 70)


if __name__ == "__main__":
    track_write_stalls()