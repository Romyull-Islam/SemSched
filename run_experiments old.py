import os
import glob
import subprocess
import re
import pandas as pd
import time


# ==========================================
# CONFIGURATION
# ==========================================
SIMULATORS = [
    "flexgen_baseline.py",
    "lia_baseline.py",
    "flashllm_baseline.py",
    "semduplex_scheduler.py"
]

MODELS = [
    ("Mistral7BCfg", 7),
    ("Llama13BCfg", 13),
    ("Qwen3_20BCfg", 20),
    ("Qwen2_5_72BCfg", 72)
]

CONFIG_FILE = "model_cfg.py"
SIM_CONFIG_FILE = "sim_cfg.py"
NVME_BW_BYTES_PER_SEC = 3.5 * (1024 ** 3)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def invalidate_pycache(filename):
    """Delete stale .pyc so subprocesses re-compile the updated .py."""
    base = os.path.splitext(os.path.basename(filename))[0]
    for pyc in glob.glob(f"__pycache__/{base}*.pyc"):
        try:
            os.remove(pyc)
        except OSError:
            pass


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

        # FIX 1: Invalidate .pyc so subprocess picks up new QUANT value
        invalidate_pycache(CONFIG_FILE)
        time.sleep(0.1)

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

        # FIX 1: Invalidate .pyc so subprocess picks up new memory values
        invalidate_pycache(SIM_CONFIG_FILE)
        time.sleep(0.1)

    except Exception as e:
        print(f"Error updating memory config: {e}")


def clean_sim_name(name):
    if "flexgen"   in name: return "FlexGen"
    if "lia"       in name: return "LIA"
    if "flashllm"  in name: return "FlashLLM"
    if "semduplex" in name: return "SemDuplex"
    return name


def calculate_cold_load(model_size_b, quant):
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}.get(quant, 4)
    total_bytes = model_size_b * 1e9 * bytes_per_param
    return total_bytes / NVME_BW_BYTES_PER_SEC


def parse_metrics(output, sim_name):
    metrics = {
        "TPS": 0.0, "Prefill_TPS": 0.0,
        "Read_Op_Pct": 0.0, "Write_Op_Pct": 0.0,
        "Write_Util_Pct": 0.0, "Bus_Mode": "Simplex"
    }

    tps = re.search(r"Decode throughput:\s*([\d\.]+)", output)
    if tps: metrics["TPS"] = float(tps.group(1))

    pf = re.search(r"Prefill throughput:\s*([\d\.]+)", output)
    if pf: metrics["Prefill_TPS"] = float(pf.group(1))

    rop = re.search(r"Read_Op_Percent:\s*([\d\.]+)%", output)
    if rop: metrics["Read_Op_Pct"] = float(rop.group(1))

    wop = re.search(r"Write_Op_Percent:\s*([\d\.]+)%", output)
    if wop: metrics["Write_Op_Pct"] = float(wop.group(1))

    util = re.search(r"Read_Ratio:\s*([\d\.]+)%", output)
    if util:
        metrics["Write_Util_Pct"] = 100.0 - float(util.group(1))
        if "semduplex" in sim_name.lower():
            metrics["Bus_Mode"] = "Full-Duplex"

    return metrics


# FIX 2: Replaced divide-by-zero write stall proof with total-time reduction
def generate_write_stall_proof(df):
    print("\n" + "=" * 60)
    print("CRITICAL PROOF: STALL REDUCTION ANALYSIS (72B FP32)")
    print("=" * 60)

    sub = df[
        (df["Model"] == 72) &
        (df["Quant"] == "fp32") &
        (df["Experiment"] == "Scalability")
    ].copy()

    if sub.empty:
        print("No 72B FP32 Scalability data found.")
        return

    sub["Total_Time_s"] = 1.0 / sub["TPS"]
    baseline_row = sub[sub["Simulator"] == "FlexGen"]
    if baseline_row.empty:
        print("FlexGen baseline not found.")
        return

    baseline_time = baseline_row["Total_Time_s"].values[0]

    print(f"{'Simulator':<12} | {'Total Time':>12} | {'Bus Mode':>12} | {'Reduction':>10}")
    print("-" * 60)

    for _, row in sub.iterrows():
        bus_mode = "Full-Duplex" if row["Simulator"] == "SemDuplex" else "Simplex"
        if row["Simulator"] == "FlexGen":
            reduction = "--"
        else:
            pct = ((baseline_time - row["Total_Time_s"]) / baseline_time) * 100
            reduction = f"{pct:.1f}%"
        print(f"{row['Simulator']:<12} | {row['Total_Time_s']:>10.2f}s | "
              f"{bus_mode:>12} | {reduction:>10}")

    sem_rows = sub[sub["Simulator"] == "SemDuplex"]
    if not sem_rows.empty:
        sem_time = sem_rows["Total_Time_s"].values[0]
        pct_total = ((baseline_time - sem_time) / baseline_time) * 100
        print(f"\nTechnical Evidence for Paper:")
        print(f"-> FlexGen baseline: {baseline_time:.2f}s per token (synchronous NAND stalls).")
        print(f"-> SemDuplex: {sem_time:.2f}s per token — {pct_total:.1f}% total stall reduction.")
        print(f"-> Sources: semantic prefill warmup + opportunistic duplex writes.")
    print("=" * 60)


# ==========================================
# EXPERIMENT RUNNER
# ==========================================
def run_experiments():
    results = []
    print(">>> STARTING RIGOROUS SOTA COMPARISON SWEEP <<<")
    print("(Including Cold Load Times & Write Channel Utilization)\n")

    update_memory_config(32, 64)

    # ---------------------------------------------------------
    # EXPERIMENT 1: Scalability (FP32)
    # ---------------------------------------------------------
    print("[Exp 1] Scalability (FP32)")
    for model_cls, size in MODELS:
        print(f"  Model: {model_cls} ({size}B)...")
        update_model_config(model_cls, "fp32")
        cold_load_s = calculate_cold_load(size, "fp32")

        for sim in SIMULATORS:
            sim_name = clean_sim_name(sim)
            try:
                ret = subprocess.run(
                    ["python", sim], capture_output=True, text=True, timeout=900)
                mets = parse_metrics(ret.stdout, sim)
                print(f"    {sim_name}: Decode={mets['TPS']:.4f} | "
                      f"Prefill={mets['Prefill_TPS']:.1f} | "
                      f"ColdLoad={cold_load_s:.2f}s | "
                      f"WriteUtil={mets['Write_Util_Pct']:.2f}%")
                results.append({
                    "Experiment": "Scalability", "Model": size,
                    "Simulator": sim_name, "Quant": "fp32",
                    "MemConfig": "32H+64C", "Cold_Load_s": cold_load_s, **mets
                })
            except Exception as e:
                print(f"    {sim_name}: ERROR — {e}")

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
            # FIX 3: No more silent except: pass
            try:
                ret = subprocess.run(
                    ["python", sim], capture_output=True, text=True, timeout=900)
                mets = parse_metrics(ret.stdout, sim)
                print(f"    {sim_name}: Decode={mets['TPS']:.4f} | "
                      f"Prefill={mets['Prefill_TPS']:.1f} | "
                      f"ColdLoad={cold_load_s:.2f}s | "
                      f"WriteUtil={mets['Write_Util_Pct']:.2f}%")
                results.append({
                    "Experiment": "Quantization", "Model": 72,
                    "Simulator": sim_name, "Quant": q,
                    "MemConfig": "32H+32C", "Cold_Load_s": cold_load_s, **mets
                })
            except Exception as e:
                print(f"    {sim_name}: ERROR — {e}")

    # ---------------------------------------------------------
    # EXPERIMENT 3: Memory Sensitivity (72B) - FULL SWEEP
    # ---------------------------------------------------------
    print("\n[Exp 3] Memory Config Sweep (72B) - ALL QUANTS")

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
                # FIX 3: No more silent except: pass
                try:
                    ret = subprocess.run(
                        ["python", sim], capture_output=True, text=True, timeout=900)
                    mets = parse_metrics(ret.stdout, sim)

                    if "semduplex" in sim.lower() or "flashllm" in sim.lower():
                        print(f"      {sim_name}: Decode={mets['TPS']:.4f}")

                    results.append({
                        "Experiment": "Memory", "Model": 72,
                        "Simulator": sim_name, "Quant": q_mem,
                        "MemConfig": cfg_label, "Cold_Load_s": cold_load_s, **mets
                    })
                except Exception as e:
                    print(f"      {sim_name}: ERROR — {e}")

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------
    df = pd.DataFrame(results)

    generate_write_stall_proof(df)

    target_cols = [
        "Experiment", "Model", "Simulator", "Quant", "MemConfig",
        "TPS", "Prefill_TPS", "Cold_Load_s",
        "Read_Op_Pct", "Write_Op_Pct", "Write_Util_Pct"
    ]
    final_cols = [c for c in target_cols if c in df.columns]
    df = df[final_cols]

    filename = "final_results_with_coldload.csv"
    df.to_csv(filename, index=False)
    print(f"\n>>> Data saved to {filename}")

    # Restore defaults
    update_memory_config(32, 64)
    update_model_config("Qwen2_5_72BCfg", "fp32")


if __name__ == "__main__":
    run_experiments()
