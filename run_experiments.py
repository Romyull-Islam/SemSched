import os
import subprocess
import re
import pandas as pd
import time

# ==========================================
# CONFIGURATION
# ==========================================
# Filenames of the 4 NEW simulators
SIMULATORS = [
    "flexgen_baseline.py",      # No-CXL Baseline
    "lia_baseline.py",          # OS-CXL SOTA
    "flashllm_baseline.py",    # Modern CXL SOTA (Replaces FlashNeuron)
    "semduplex_scheduler.py"    # Ours
]

# Models to test in Experiment 1
MODELS = [
    ("Mistral7BCfg", 7),
    ("Llama13BCfg", 13),
    ("Qwen3_20BCfg", 20),
    ("Qwen2_5_72BCfg", 72)
]

# Define filenames explicitly (Fixes ImportError)
CONFIG_FILE = "model_cfg.py"
SIM_CONFIG_FILE = "sim_cfg.py"

# ==========================================
# HELPER FUNCTIONS (Config Rewriting)
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
        time.sleep(0.1) # Brief pause to ensure file write settles
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
    """Shortens simulator filenames for cleaner CSV output."""
    if "flexgen" in name: return "FlexGen (Baseline)"
    if "lia" in name: return "LIA (OS-CXL)"
    if "flashneuron" in name: return "FlashNeuron (HW)"
    if "semduplex" in name: return "SemDuplex (Ours)"
    return name

def parse_metrics(output):
    """Parses standard output from simulators to extract key metrics."""
    metrics = {
        "TPS": 0.0,             # Decode Tokens/sec
        "Stall_s": 0.0,         # Decode Stalls
        "Hit_Rate": 0.0,        # Cache Hit Rate
        "Prefill_Time_s": 0.0,  # Prefill Latency
        "Prefill_TPS": 0.0,     # Prefill Throughput
        "Duplex_Read_Ratio": 0.0,
        "Injected_Ops": 0
    }
    
    # Regex patterns matching the new files
    tps_match = re.search(r"(?:Decode throughput|Decode tokens/sec): ([\d\.]+)", output)
    if tps_match: metrics["TPS"] = float(tps_match.group(1))

    pf_tps_match = re.search(r"(?:Prefill throughput): ([\d\.]+)", output)
    if pf_tps_match: metrics["Prefill_TPS"] = float(pf_tps_match.group(1))

    duplex_match = re.search(r"Read_Ratio.*: ([\d\.]+)%", output)
    if duplex_match: metrics["Duplex_Read_Ratio"] = float(duplex_match.group(1))

    return metrics

# ==========================================
# EXPERIMENT RUNNER
# ==========================================
def run_experiments():
    results = []
    print(">>> STARTING RIGOROUS SOTA COMPARISON SWEEP <<<")
    
    # Set Default Memory for Scalability (32GB Host / 64GB CXL)
    update_memory_config(32, 64)

    # ---------------------------------------------------------
    # EXPERIMENT 1: Scalability (Model Sizes in FP32)
    # ---------------------------------------------------------
    print("\n[Exp 1] Scalability (FP32)")
    for model_cls, size in MODELS:
        print(f"  Model: {model_cls} ({size}B)...")
        update_model_config(model_cls, "fp32")
        
        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=900)
                if ret.returncode != 0:
                    print(f"    {sim_name}: FAILED")
                    continue

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

    # ---------------------------------------------------------
    # EXPERIMENT 2: Quantization (72B Model Only)
    # ---------------------------------------------------------
    print("\n[Exp 2] Quantization (72B)")
    model_72b = "Qwen2_5_72BCfg"
    
    # We use 32GB CXL for this experiment to highlight INT4 caching benefits
    update_memory_config(32, 32)
    print("  (Setting Memory to 32GB Host + 32GB CXL to stress INT4 capacity)")

    for q in ["fp32", "fp16", "int8", "int4"]:
        print(f"  Quantization: {q}...")
        update_model_config(model_72b, q)
        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=900)
                mets = parse_metrics(ret.stdout)
                print(f"    {sim_name}: Decode={mets['TPS']:.4f} t/s | Prefill={mets['Prefill_TPS']:.2f} t/s")
                
                results.append({
                    "Experiment": "Quantization",
                    "Model": 72,
                    "Simulator": sim_name,
                    "Quant": q,
                    "MemConfig": "32H+32C",
                    **mets
                })
            except: pass

    # ---------------------------------------------------------
    # EXPERIMENT 3: Memory Sensitivity (72B, FP16 & FP32)
    # ---------------------------------------------------------
    print("\n[Exp 3] Memory Config Sweep (72B)")
    mem_configs = [(16, 32), (16, 64), (32, 32), (32, 64)]
    
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
                    ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=900)
                    mets = parse_metrics(ret.stdout)
                    print(f"      {sim_name}: Decode={mets['TPS']:.4f} t/s | Prefill={mets['Prefill_TPS']:.2f} t/s")
                    
                    results.append({
                        "Experiment": "Memory",
                        "Model": 72,
                        "Simulator": sim_name,
                        "Quant": q_mem,
                        "MemConfig": cfg_label,
                        **mets
                    })
                except: pass

    # ---------------------------------------------------------
    # FINALIZE
    # ---------------------------------------------------------
    filename = "final_results_all_combinations.csv"
    df = pd.DataFrame(results)
    
    cols = ["Experiment", "Model", "Simulator", "Quant", "MemConfig", "TPS", "Stall_s", "Hit_Rate", 
            "Prefill_TPS", "Duplex_Read_Ratio", "Injected_Ops"]
    # Add any extra columns that might appear
    cols += [c for c in df.columns if c not in cols]
    
    # Reorder if columns exist, otherwise ignore
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]
    
    df.to_csv(filename, index=False)
    print(f"\n>>> Data saved to {filename}")
    
    # Restore Defaults
    update_memory_config(32, 64)
    update_model_config("Qwen2_5_72BCfg", "fp32")

if __name__ == "__main__":
    run_experiments()