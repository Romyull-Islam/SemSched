import os
import subprocess
import re
import pandas as pd
import time
import sys

# ==========================================
# CONFIGURATION
# ==========================================
SIMULATORS = [
    "simulator_normal_nocxl_prefill.py",              # Baseline
    "simulator_tired_ADAPTIVE_prefill.py",            # Adaptive
    "simulation_async_prefetch_sequential_prefill.py",# Async
    "simulation_semantic_duplex_prefill.py"           # SemDuplex
]

MODELS = [
    ("Mistral7BCfg", 7),
    ("Llama13BCfg", 13),
    ("Qwen3_20BCfg", 20),
    ("Qwen2_5_72BCfg", 72)
]

CONFIG_FILE = "model_cfg.py"
SIM_CONFIG_FILE = "sim_cfg.py"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def update_model_config(model_cls, quant):
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
        time.sleep(0.2) 
    except Exception as e:
        print(f"Error updating model config: {e}")

def update_memory_config(host_gb, cxl_gb):
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
        time.sleep(0.2)
    except Exception as e:
        print(f"Error updating memory config: {e}")

def clean_sim_name(name):
    if "normal" in name: return "Baseline"
    if "ADAPTIVE" in name: return "Adaptive"
    if "async" in name: return "Async"
    if "semantic" in name: return "SemDuplex"
    return name

def parse_metrics(output):
    """
    Robust parser: Captures Decode TPS, Stalls, Hit Rate, AND Prefill Metrics.
    FIXED: Case-insensitive prefill matching + Auto-calc TPS from time.
    """
    metrics = {
        "TPS": 0.0,             # Decode Tokens/sec
        "Stall_s": 0.0,         # Decode Stalls
        "Hit_Rate": 0.0,        # Cache Hit Rate
        "Prefill_Time_s": 0.0,  # NEW: Time to process prompt
        "Prefill_TPS": 0.0,     # NEW: Prompt tokens/sec
        "Duplex_Read_Ratio": 0.0,
        "Sparsity_Savings_FLOPs": 0,
        "Injected_Ops": 0,
        "NAND_Miss_Bytes": 0
    }
    
    # --- DECODE METRICS ---
    tps_match = re.search(r"(?:Decode throughput|Decode tokens/sec|Overall throughput): ([\d\.]+)", output)
    if tps_match: metrics["TPS"] = float(tps_match.group(1))

    stall_match = re.search(r"(?:Compute stall time|Decode compute stalls).*: ([\d\.]+)s", output)
    if stall_match: metrics["Stall_s"] = float(stall_match.group(1))
            
    hit_match = re.search(r"Cache hit rate.*: ([\d\.]+)%", output)
    if hit_match: metrics["Hit_Rate"] = float(hit_match.group(1))
    
    # --- PREFILL METRICS (FIXED) ---
    pf_time_match = re.search(r"(?i)prefill.*time.*: ([\d\.]+)s", output)
    if pf_time_match: 
        metrics["Prefill_Time_s"] = float(pf_time_match.group(1))

    pf_tps_match = re.search(r"(?i)prefill throughput: ([\d\.]+)", output)
    if pf_tps_match: 
        metrics["Prefill_TPS"] = float(pf_tps_match.group(1))
    elif metrics["Prefill_Time_s"] > 0:
        metrics["Prefill_TPS"] = 512.0 / metrics["Prefill_Time_s"]

    # --- DUPLEX SPECIFICS ---
    duplex_match = re.search(r"Final Duplex read ratio: ([\d\.]+)%", output)
    if duplex_match: metrics["Duplex_Read_Ratio"] = float(duplex_match.group(1))

    sparsity_match = re.search(r"Sparsity-based FLOP savings: ([\d,]+)", output)
    if sparsity_match: metrics["Sparsity_Savings_FLOPs"] = int(sparsity_match.group(1).replace(',', ''))

    ops_match = re.search(r"Complementary ops injected: (\d+)", output)
    if ops_match: metrics["Injected_Ops"] = int(ops_match.group(1))

    miss_match = re.search(r"Bytes from NAND miss.*: ([\d,]+)", output)
    if miss_match: metrics["NAND_Miss_Bytes"] = int(miss_match.group(1).replace(',', ''))

    return metrics

def run_experiments():
    results = []
    print(">>> STARTING COMPREHENSIVE SWEEP (V6 - WITH MEMORY FP32) <<<")
    
    # Set Default Memory
    update_memory_config(32, 64)

    # 1. Scalability Sweep (FP32)
    print("\n[Exp 1] Scalability (FP32)")
    for model_cls, size in MODELS:
        print(f"  Model: {model_cls} ({size}B)...")
        update_model_config(model_cls, "fp32")
        
        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=600)
                mets = parse_metrics(ret.stdout)
                print(f"    {sim_name}: Decode={mets['TPS']:.4f} t/s | Prefill={mets['Prefill_TPS']:.2f} t/s")
                results.append({
                    "Experiment": "Scalability",
                    "Model": size,
                    "Simulator": sim_name,
                    "Quant": "fp32",
                    "MemConfig": "32H+64C",
                    **mets
                })
            except Exception as e:
                print(f"    {sim_name}: ERROR {e}")

    # 2. Quantization Sweep (72B Only)
    print("\n[Exp 2] Quantization (72B)")
    model_72b = "Qwen2_5_72BCfg"
    for q in ["fp32", "fp16", "int8", "int4"]:
        print(f"  Quantization: {q}...")
        update_model_config(model_72b, q)
        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=600)
                mets = parse_metrics(ret.stdout)
                results.append({
                    "Experiment": "Quantization",
                    "Model": 72,
                    "Simulator": sim_name,
                    "Quant": q,
                    "MemConfig": "32H+64C",
                    **mets
                })
            except: pass

    # 3. Memory Sensitivity (72B, FP16 AND FP32)
    print("\n[Exp 3] Memory Config Sweep (72B)")
    mem_configs = [(16, 32), (16, 64), (32, 32), (32, 64)]
    # Loop over BOTH fp16 and fp32
    for q_mem in ["fp16", "fp32"]:
        print(f"  --> Testing Quantization: {q_mem}")
        update_model_config(model_72b, q_mem)
        
        for h, c in mem_configs:
            cfg_label = f"{h}GB_Host+{c}GB_CXL"
            print(f"    Config: {cfg_label}...")
            update_memory_config(h, c)
            
            for sim in SIMULATORS:
                sim_name = clean_sim_name(sim)
                try:
                    ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=600)
                    mets = parse_metrics(ret.stdout)
                    results.append({
                        "Experiment": "Memory",
                        "Model": 72,
                        "Simulator": sim_name,
                        "Quant": q_mem,  # Dynamic quantization label
                        "MemConfig": cfg_label,
                        **mets
                    })
                except: pass

    # Save Results
    filename = "final_results_with_prefill.csv"
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\n>>> Data saved to {filename}")
    
    # Restore Defaults
    update_memory_config(32, 64)
    update_model_config("Qwen2_5_72BCfg", "fp32")

if __name__ == "__main__":
    run_experiments()