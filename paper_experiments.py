# paper_experiments.py
import os
import subprocess
import re
import pandas as pd
import time
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Exact filenames of your simulators
SIMULATORS = [
    "simulator_normal_nocxl_prefill.py",              # Baseline
    "simulator_tired_ADAPTIVE_prefill.py",            # Adaptive
    "simulation_async_prefetch_sequential_prefill.py",# Async
    "simulation_semantic_duplex_prefill.py"           # DuplexGen
]

# Models to test (ClassName, SizeLabel)
MODELS = [
    ("Mistral7BCfg", 7),
    ("Llama13BCfg", 13),
    ("Qwen3_20BCfg", 20),
    ("Qwen2_5_72BCfg", 72)
]

CONFIG_FILE = "model_cfg.py"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def update_config_file(model_cls, quant):
    """Safely rewrites model_cfg.py with new settings"""
    try:
        with open(CONFIG_FILE, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            # Replace Model Config
            if line.strip().startswith("DEFAULT_MODEL_CFG ="):
                new_lines.append(f"DEFAULT_MODEL_CFG = {model_cls} # Auto-set\n")
            # Replace Quantization
            elif line.strip().startswith("QUANT ="):
                new_lines.append(f'QUANT = "{quant}" # Auto-set\n')
            else:
                new_lines.append(line)
                
        with open(CONFIG_FILE, "w") as f:
            f.writelines(new_lines)
            f.flush()           # Flush buffer
            os.fsync(f.fileno()) # Sync to disk (INDENTATION FIXED)
        
        time.sleep(0.2) # Safety pause for file system
        
    except Exception as e:
        print(f"Error updating config: {e}")
        sys.exit(1)

def clean_sim_name(name):
    if "normal" in name: return "Baseline"
    if "ADAPTIVE" in name: return "Adaptive"
    if "async" in name: return "Async"
    if "semantic" in name: return "DuplexGen"
    return name

def parse_throughput(output):
    """Extracts throughput number from various log formats"""
    # Regex patterns to match your different print statements
    patterns = [
        r"Overall throughput: ([\d\.]+) tokens/sec", # Baseline / Async / Adaptive
        r"Decode tokens/sec: ([\d\.]+)",              # DuplexGen
        r"Decode throughput: ([\d\.]+) tokens/sec"    # Alternative
    ]
    
    for line in output.split('\n'):
        for pat in patterns:
            match = re.search(pat, line)
            if match:
                return float(match.group(1))
    return 0.0

def run_experiments():
    results = []
    
    print(">>> STARTING ROBUST EXPERIMENT SWEEP <<<")
    
    # -------------------------------------------------
    # 1. Scalability Sweep (FP32)
    # -------------------------------------------------
    print("\n[Experiment 1] Scalability (FP32)")
    for model_cls, size in MODELS:
        print(f"  Model: {model_cls} ({size}B)...")
        update_config_file(model_cls, "fp32")
        
        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                # Run simulator and capture stdout
                ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=600)
                
                if ret.returncode != 0:
                    print(f"    {sim_name}: FAILED (Exit Code {ret.returncode})")
                    # print(ret.stderr) # Uncomment to debug
                    tps = 0.0
                else:
                    tps = parse_throughput(ret.stdout)
                    print(f"    {sim_name}: {tps:.4f} t/s")
                    
                results.append({
                    "Experiment": "Scalability",
                    "Model": size,
                    "Simulator": sim_name,
                    "Quant": "fp32",
                    "TPS": tps
                })
            except Exception as e:
                print(f"    {sim_name}: ERROR ({e})")

    # -------------------------------------------------
    # 2. Quantization Sweep (70B Only)
    # -------------------------------------------------
    print("\n[Experiment 2] Quantization (70B)")
    model_70b = "Qwen2_5_72BCfg"
    
    for q in ["fp32", "fp16", "int8", "int4"]:
        print(f"  Quantization: {q}...")
        update_config_file(model_70b, q)
        
        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(["python", sim], capture_output=True, text=True, timeout=600)
                tps = parse_throughput(ret.stdout) if ret.returncode == 0 else 0.0
                print(f"    {sim_name}: {tps:.4f} t/s")
                
                results.append({
                    "Experiment": "Quantization",
                    "Model": 72,
                    "Simulator": sim_name,
                    "Quant": q,
                    "TPS": tps
                })
            except:
                pass

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("final_results.csv", index=False)
    print("\n>>> Data saved to final_results.csv")

if __name__ == "__main__":
    run_experiments()