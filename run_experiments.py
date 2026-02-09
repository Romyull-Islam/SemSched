import os
import subprocess
import re
import pandas as pd
import time

# ==========================================
# CONFIGURATION
# ==========================================
SIMULATORS = [
    "flexgen_baseline.py",      # No-CXL Baseline
    "lia_baseline.py",          # OS-CXL SOTA
    "flashllm_baseline.py",     # Modern CXL SOTA (Pipelined Prefetching)
    "semduplex_scheduler.py"    # Ours
]

MODELS = [
    ("Mistral7BCfg", 7),
    ("Llama13BCfg", 13),
    ("Qwen3_20BCfg", 20),
    ("Qwen2_5_72BCfg", 72)
]

CONFIG_FILE = "model_cfg.py"
SIM_CONFIG_FILE = "sim_cfg.py"

# Standard NVMe Bandwidth for Cold Load Calculation (3.5 GiB/s)
NVME_BW_BYTES_PER_SEC = 3.5 * (1024**3)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def update_model_config(model_cls, quant):
    """Updates model_cfg.py with the requested Model Class and Quantization."""
    try:
        with open(CONFIG_FILE, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith("DEFAULT_MODEL_CFG ="):
                new_lines.append(f"DEFAULT_MODEL_CFG = {model_cls} # Auto-set\n")
            elif line.strip().startswith("QUANT ="):
                new_lines.append(f'QUANT = "{quant}" # Auto-set\n')
            else:
                new_lines.append(line)
                
        with open(CONFIG_FILE, "w") as f:
            f.writelines(new_lines)
            f.flush()
            os.fsync(f.fileno())
        time.sleep(0.1) 
    except Exception as e:
        print(f"Error updating model config: {e}")

def update_memory_config(host_gb, cxl_gb):
    """Updates sim_cfg.py with the requested Host and CXL DRAM capacities."""
    try:
        with open(SIM_CONFIG_FILE, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith("host_dram_capacity_bytes ="):
                new_lines.append(f"host_dram_capacity_bytes = {host_gb} * GiB # Auto-set\n")
            elif line.strip().startswith("cxl_dev_dram_capacity_bytes ="):
                new_lines.append(f"cxl_dev_dram_capacity_bytes = {cxl_gb} * GiB # Auto-set\n")
            else:
                new_lines.append(line)
                
        with open(SIM_CONFIG_FILE, "w") as f:
            f.writelines(new_lines)
            f.flush()
            os.fsync(f.fileno())
        time.sleep(0.1)
    except Exception as e:
        print(f"Error updating memory config: {e}")

def clean_sim_name(name):
    if "flexgen" in name: return "FlexGen"
    if "lia" in name: return "LIA"
    if "flashllm" in name: return "FlashLLM"
    if "semduplex" in name: return "SemDuplex"
    return name

def calculate_cold_load(model_size_b, quant):
    """
    Estimates Cold Load time based on model size and quantization.
    Time = Total Bytes / NVMe Bandwidth
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }.get(quant, 4)
    
    # 1 Billion params * 1e9 * bytes_per_param
    total_bytes = model_size_b * 1e9 * bytes_per_param
    time_s = total_bytes / NVME_BW_BYTES_PER_SEC
    return time_s

def parse_metrics(output, sim_name):
    """Parses output for TPS, Prefill, and calculates Write Utilization."""
    metrics = {
        "TPS": 0.0,
        "Prefill_TPS": 0.0,
        "Write_Util_Pct": 0.0
    }
    
    # Decode TPS
    tps_match = re.search(r"(?:Decode throughput|Decode tokens/sec): ([\d\.]+)", output)
    if tps_match: metrics["TPS"] = float(tps_match.group(1))

    # Prefill TPS
    pf_tps_match = re.search(r"(?:Prefill throughput): ([\d\.]+)", output)
    if pf_tps_match: metrics["Prefill_TPS"] = float(pf_tps_match.group(1))

    # Write Channel Utilization
    # SemDuplex prints "Read_Ratio". Write Util = 100 - Read Ratio.
    # Others are Simplex (Read-Only), so Write Util is 0.
    if "semduplex" in sim_name.lower():
        duplex_match = re.search(r"Read_Ratio.*: ([\d\.]+)%", output)
        if duplex_match: 
            read_ratio = float(duplex_match.group(1))
            metrics["Write_Util_Pct"] = 100.0 - read_ratio
    else:
        metrics["Write_Util_Pct"] = 0.0

    return metrics

# ==========================================
# EXPERIMENT RUNNER
# ==========================================
def run_experiments():
    results = []
    print(">>> STARTING RIGOROUS SOTA COMPARISON SWEEP <<<")
    print("(Including Cold Load Times & Write Channel Utilization)\n")
    
    # Defaults
    update_memory_config(32, 64)

    # ---------------------------------------------------------
    # EXPERIMENT 1: Scalability (FP32)
    # ---------------------------------------------------------
    print("[Exp 1] Scalability (FP32)")
    for model_cls, size in MODELS:
        print(f"  Model: {model_cls} ({size}B)...")
        update_model_config(model_cls, "fp32")
        
        # Calc Cold Load for this config
        cold_load_s = calculate_cold_load(size, "fp32")
        
        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=900)
                mets = parse_metrics(ret.stdout, sim)
                
                print(f"    {sim_name}: Decode={mets['TPS']:.4f} | Prefill={mets['Prefill_TPS']:.1f} | "
                      f"ColdLoad={cold_load_s:.2f}s | WriteUtil={mets['Write_Util_Pct']:.2f}%")
                
                results.append({
                    "Experiment": "Scalability",
                    "Model": size,
                    "Simulator": sim_name,
                    "Quant": "fp32",
                    "MemConfig": "32H+64C",
                    "Cold_Load_s": cold_load_s,
                    **mets
                })
            except Exception as e:
                print(f"    {sim_name}: ERROR {e}")

    # ---------------------------------------------------------
    # EXPERIMENT 2: Quantization (72B Only)
    # ---------------------------------------------------------
    print("\n[Exp 2] Quantization (72B)")
    update_memory_config(32, 32)
    print("  (Setting Memory to 32GB Host + 32GB CXL)")

    for q in ["fp32", "fp16", "int8", "int4"]:
        print(f"  Quantization: {q}...")
        update_model_config("Qwen2_5_72BCfg", q)
        
        cold_load_s = calculate_cold_load(72, q)

        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=900)
                mets = parse_metrics(ret.stdout, sim)
                
                print(f"    {sim_name}: Decode={mets['TPS']:.4f} | Prefill={mets['Prefill_TPS']:.1f} | "
                      f"ColdLoad={cold_load_s:.2f}s | WriteUtil={mets['Write_Util_Pct']:.2f}%")
                
                results.append({
                    "Experiment": "Quantization",
                    "Model": 72,
                    "Simulator": sim_name,
                    "Quant": q,
                    "MemConfig": "32H+32C",
                    "Cold_Load_s": cold_load_s,
                    **mets
                })
            except: pass


# ---------------------------------------------------------
    # EXPERIMENT 3: Memory Sensitivity (72B) - FULL SWEEP
    # ---------------------------------------------------------
    print("\n[Exp 3] Memory Config Sweep (72B) - ALL QUANTS")
    
    # ADDED "int4" and "int8" here
    for q_mem in ["int4", "int8", "fp16", "fp32"]:
        print(f"  --> Testing Quantization: {q_mem}")
        update_model_config("Qwen2_5_72BCfg", q_mem)
        cold_load_s = calculate_cold_load(72, q_mem)

        for h, c in [(16, 32), (16, 64), (32, 32), (32, 64)]:
            cfg_label = f"{h}GB_Host+{c}GB_CXL"
            print(f"    Config: {cfg_label}...")
            update_memory_config(h, c)
            
            for sim in SIMULATORS:
                sim_name = clean_sim_name(sim)
                try:
                    ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=900)
                    mets = parse_metrics(ret.stdout, sim)
                    
                    if "semduplex" in sim.lower() or "flashllm" in sim.lower():
                         print(f"      {sim_name}: Decode={mets['TPS']:.4f}")

                    results.append({
                        "Experiment": "Memory",
                        "Model": 72,
                        "Simulator": sim_name,
                        "Quant": q_mem,
                        "MemConfig": cfg_label,
                        "Cold_Load_s": cold_load_s,
                        **mets
                    })
                except: pass

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    
    # Clean Column Order
    target_cols = ["Experiment", "Model", "Simulator", "Quant", "MemConfig", 
                   "TPS", "Prefill_TPS", "Cold_Load_s", "Write_Util_Pct"]
    
    # Keep only columns that actually exist (in case of empty results)
    final_cols = [c for c in target_cols if c in df.columns]
    df = df[final_cols]
    
    filename = "final_results_with_coldload.csv"
    df.to_csv(filename, index=False)
    print(f"\n>>> Data saved to {filename}")
    
    # Restore
    update_memory_config(32, 64)
    update_model_config("Qwen2_5_72BCfg", "fp32")

if __name__ == "__main__":
    run_experiments()