import os
import glob
import subprocess
import re
import pandas as pd
import time


# ==========================================
# SIMULATORS & MODELS
# ==========================================
SIMULATORS = [
    "flexgen_baseline.py",
    "lia_baseline.py",
    "llmflash_baseline.py",
    "semduplex_scheduler.py"
]

MODELS = [
    ("Mistral7BCfg",    7),
    ("Llama13BCfg",    13),
    ("Qwen3_20BCfg",   20),
    ("Qwen2_5_72BCfg", 72)
]

CONFIG_FILE           = "model_cfg.py"
SIM_CONFIG_FILE       = "sim_cfg.py"
NVME_BW_BYTES_PER_SEC = 3.5 * (1024 ** 3)

# ==========================================
# BATCH SIZE CONSTANTS  (all in one place)
# ==========================================
BATCH_SIZES_SCALE = [1, 128]              # Exp 1: B=1 baseline + B=128 serving
BATCH_SIZES_QUANT = [1, 16, 64, 128]      # Exp 2: quantization multi-batch
BATCH_SIZES_MEM   = [1, 128]              # Exp 3: memory config sweep
BATCH_SIZES       = [1, 4, 8, 16, 32,
                     64, 128, 256]         # Exp 4: full batch sweep

# Exp 4 memory: tight config to expose NAND overflow at large batch
EXP4_HOST_GB = 16
EXP4_CXL_GB  = 32


# ==========================================
# CONFIG HELPERS
# ==========================================
def invalidate_pycache(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    for pyc in glob.glob(f"__pycache__/{base}*.pyc"):
        try: os.remove(pyc)
        except OSError: pass


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
                new_lines.append(
                    f"host_dram_capacity_bytes = {host_gb} * GiB # Auto-set\n")
            elif line.strip().startswith("cxl_dev_dram_capacity_bytes ="):
                new_lines.append(
                    f"cxl_dev_dram_capacity_bytes = {cxl_gb} * GiB # Auto-set\n")
            else:
                new_lines.append(line)
        with open(SIM_CONFIG_FILE, "w") as f:
            f.writelines(new_lines)
            f.flush()
            os.fsync(f.fileno())
        invalidate_pycache(SIM_CONFIG_FILE)
        time.sleep(0.1)
    except Exception as e:
        print(f"Error updating memory config: {e}")


def update_batch_config(batch_size):
    try:
        with open(SIM_CONFIG_FILE, "r") as f:
            lines = f.readlines()
        new_lines = []
        found = False
        for line in lines:
            if line.strip().startswith("BATCH_SIZE ="):
                new_lines.append(f"BATCH_SIZE = {batch_size} # Auto-set\n")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"BATCH_SIZE = {batch_size} # Auto-set\n")
        with open(SIM_CONFIG_FILE, "w") as f:
            f.writelines(new_lines)
            f.flush()
            os.fsync(f.fileno())
        invalidate_pycache(SIM_CONFIG_FILE)
        time.sleep(0.1)
    except Exception as e:
        print(f"Error updating batch config: {e}")


def clean_sim_name(name):
    if "flexgen"   in name: return "FlexGen"
    if "lia"       in name: return "LIA"
    if "llmflash"  in name: return "LLMFlash"
    if "semduplex" in name: return "SemDuplex"
    return name


def calculate_cold_load(model_size_b, quant):
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}.get(quant, 4)
    return (model_size_b * 1e9 * bytes_per_param) / NVME_BW_BYTES_PER_SEC


def parse_metrics(output, sim_name):
    metrics = {
        "TPS": 0.0, "Prefill_TPS": 0.0,
        "Read_Op_Pct": 0.0, "Write_Op_Pct": 0.0,
        "Write_Util_Pct": 0.0, "Bus_Mode": "Simplex"
    }
    tps = re.search(r"Decode throughput:\s*([\d\.]+)", output)
    if tps: metrics["TPS"] = float(tps.group(1))
    pf  = re.search(r"Prefill throughput:\s*([\d\.]+)", output)
    if pf:  metrics["Prefill_TPS"] = float(pf.group(1))
    rop = re.search(r"Read_Op_Percent:\s*([\d\.]+)%", output)
    if rop: metrics["Read_Op_Pct"] = float(rop.group(1))
    wop = re.search(r"Write_Op_Percent:\s*([\d\.]+)%", output)
    if wop: metrics["Write_Op_Pct"] = float(wop.group(1))
    wutil = re.search(r"Write_Util_Pct:\s*([\d\.]+)%", output)
    if wutil:
        metrics["Write_Util_Pct"] = float(wutil.group(1))

    stall = re.search(r"Read_Stall_s:\s*([\d\.]+)", output)
    if stall:
        metrics["Read_Stall_s"] = float(stall.group(1))

    if "semduplex" in sim_name.lower():
        metrics["Bus_Mode"] = "Full-Duplex"

    return metrics


# ==========================================
# REPORTING HELPERS
# ==========================================
def generate_write_stall_proof(df):
    print("\n" + "=" * 60)
    print("CRITICAL PROOF: STALL REDUCTION ANALYSIS (72B FP32)")
    print("(Tight memory: 16GB Host + 32GB CXL)")
    print("=" * 60)

    sub = df[
        (df["Model"]      == 72) &
        (df["Quant"]      == "fp32") &
        (df["Experiment"] == "Memory") &
        (df["MemConfig"]  == "16GB_Host+32GB_CXL")
    ].copy()

    if sub.empty:
        print("  [Note] Falling back to Scalability experiment data")
        sub = df[
            (df["Model"]      == 72) &
            (df["Quant"]      == "fp32") &
            (df["Experiment"] == "Scalability")
        ].copy()

    if sub.empty:
        print("No 72B FP32 data found.")
        return

    use_measured = "Read_Stall_s" in sub.columns and sub["Read_Stall_s"].sum() > 0
    if use_measured:
        sub["Total_Time_s"] = sub["Read_Stall_s"]
        print("  [Using directly measured Read_Stall_s]")
    else:
        sub["Total_Time_s"] = 1.0 / sub["TPS"].replace(0, float("nan"))
        print("  [Warning: Read_Stall_s not found — using 1/TPS proxy]")

    baseline_row = sub[sub["Simulator"] == "FlexGen"]
    if baseline_row.empty:
        print("FlexGen baseline not found.")
        return

    baseline_time = baseline_row["Total_Time_s"].values[0]
    print(f"{'Simulator':<12} | {'Total Time':>12} | {'Bus Mode':>12} | {'Reduction':>10}")
    print("-" * 60)

    for _, row in sub.iterrows():
        bus_mode  = "Full-Duplex" if row["Simulator"] == "SemDuplex" else "Simplex"
        reduction = "--"
        if row["Simulator"] != "FlexGen":
            pct       = ((baseline_time - row["Total_Time_s"]) / baseline_time) * 100
            reduction = f"{pct:.1f}%"
        print(f"{row['Simulator']:<12} | {row['Total_Time_s']:>10.2f}s | "
              f"{bus_mode:>12} | {reduction:>10}")

    sem_rows = sub[sub["Simulator"] == "SemDuplex"]
    if not sem_rows.empty:
        sem_time  = sem_rows["Total_Time_s"].values[0]
        pct_total = ((baseline_time - sem_time) / baseline_time) * 100
        llm_rows  = sub[sub["Simulator"] == "LLMFlash"]
        llm_time  = llm_rows["Total_Time_s"].values[0] if not llm_rows.empty else 0
        llm_pct   = ((baseline_time - llm_time) / baseline_time) * 100 \
                    if llm_time > 0 else 0
        print(f"\nTechnical Evidence for Paper:")
        print(f"-> FlexGen   : {baseline_time:.2f}s per token (synchronous NAND stalls).")
        print(f"-> LLMFlash  : {llm_time:.2f}s — {llm_pct:.1f}% reduction.")
        print(f"-> SemDuplex : {sem_time:.2f}s — {pct_total:.1f}% reduction.")
        print(f"-> SemDuplex advantage: generality + prefill + B≥4 serving.")
    print("=" * 60)


def print_batch_analysis(df):
    print("\n" + "=" * 70)
    print("BATCH SCALING: LLMFlash Sparsity Collapse vs SemDuplex")
    print(f"(Memory: {EXP4_HOST_GB}GB Host + {EXP4_CXL_GB}GB CXL — tight)")
    print("=" * 70)
    sub = df[df["Experiment"] == "BatchSweep"].copy()
    if sub.empty:
        return
    print(f"\n{'Batch':<8} | {'LLMFlash':>12} | {'SemDuplex':>12} | "
          f"{'LIA':>10} | {'FlexGen':>10} | {'Winner':>10}")
    print("-" * 75)
    for b in sorted(sub["BatchSize"].unique()):
        b_df  = sub[sub["BatchSize"] == b]
        llm_v = b_df[b_df["Simulator"] == "LLMFlash"]["TPS"].values
        sem_v = b_df[b_df["Simulator"] == "SemDuplex"]["TPS"].values
        lia_v = b_df[b_df["Simulator"] == "LIA"]["TPS"].values
        flx_v = b_df[b_df["Simulator"] == "FlexGen"]["TPS"].values
        llm_v = llm_v[0] if len(llm_v) else 0
        sem_v = sem_v[0] if len(sem_v) else 0
        lia_v = lia_v[0] if len(lia_v) else 0
        flx_v = flx_v[0] if len(flx_v) else 0
        winner = "LLMFlash" if llm_v > sem_v else "SemDuplex"
        print(f"B={b:<6} | {llm_v:>12.4f} | {sem_v:>12.4f} | "
              f"{lia_v:>10.4f} | {flx_v:>10.4f} | {winner:>10}")
    print("=" * 70)


# ==========================================
# MAIN EXPERIMENT RUNNER
# ==========================================
def run_experiments():
    results = []
    print(">>> STARTING RIGOROUS SOTA COMPARISON SWEEP <<<")
    print("(Including Cold Load Times & Write Channel Utilization)\n")

    # Default starting state
    update_memory_config(32, 64)
    update_batch_config(1)

    # ---------------------------------------------------------
    # EXPERIMENT 1: Scalability — FP32, all models
    # Runs at BOTH B=1 (baseline) and B=128 (serving)
    # so motivation figure panel (a) has B=128 line data
    # ---------------------------------------------------------
    print(f"[Exp 1] Scalability (FP32) — batches: {BATCH_SIZES_SCALE}")
    update_memory_config(32, 64)

    for batch in BATCH_SIZES_SCALE:
        print(f"\n  -- B={batch} --")
        update_batch_config(batch)
        for model_cls, size in MODELS:
            print(f"  Model: {model_cls} ({size}B)...")
            update_model_config(model_cls, "fp32")
            cold_load_s = calculate_cold_load(size, "fp32")
            for sim in SIMULATORS:
                sim_name = clean_sim_name(sim)
                try:
                    ret  = subprocess.run(
                        ["python", sim], capture_output=True,
                        text=True, timeout=900)
                    mets = parse_metrics(ret.stdout, sim)
                    print(f"    {sim_name}: Decode={mets['TPS']:.4f} | "
                          f"Prefill={mets['Prefill_TPS']:.1f} | "
                          f"ColdLoad={cold_load_s:.2f}s | "
                          f"WriteUtil={mets['Write_Util_Pct']:.2f}%")
                    results.append({
                        "Experiment": "Scalability", "Model": size,
                        "Simulator": sim_name, "Quant": "fp32",
                        "MemConfig": "32H+64C", "BatchSize": batch,
                        "Cold_Load_s": cold_load_s, **mets
                    })
                except Exception as e:
                    print(f"    {sim_name}: ERROR — {e}")

    # ---------------------------------------------------------
    # EXPERIMENT 2: Quantization — 72B, multi-batch
    # ---------------------------------------------------------
    print(f"\n[Exp 2] Quantization (72B) — batches: {BATCH_SIZES_QUANT}")
    print(f"  (Memory: 32GB Host + 32GB CXL)")
    update_memory_config(32, 32)

    for quant in ["fp32", "fp16", "int8", "int4"]:
        cold_load_s = calculate_cold_load(72, quant)
        update_model_config("Qwen2_5_72BCfg", quant)
        for batch in BATCH_SIZES_QUANT:
            print(f"  Quant={quant}, B={batch}...")
            update_batch_config(batch)
            for sim in SIMULATORS:
                sim_name = clean_sim_name(sim)
                try:
                    ret  = subprocess.run(
                        ["python", sim], capture_output=True,
                        text=True, timeout=900)
                    mets = parse_metrics(ret.stdout, sim)
                    print(f"    {sim_name}: Decode={mets['TPS']:.4f} | "
                          f"Prefill={mets['Prefill_TPS']:.1f} | "
                          f"ColdLoad={cold_load_s:.2f}s | "
                          f"WriteUtil={mets['Write_Util_Pct']:.2f}%")
                    results.append({
                        "Experiment": "Quantization", "Model": 72,
                        "Simulator": sim_name, "Quant": quant,
                        "MemConfig": "32H+32C", "BatchSize": batch,
                        "Cold_Load_s": cold_load_s, **mets
                    })
                except Exception as e:
                    print(f"    {sim_name}: ERROR — {e}")

    # ---------------------------------------------------------
    # EXPERIMENT 3: Memory Config Sweep — 72B, all quants
    # Runs at BOTH B=1 and B=128
    # ---------------------------------------------------------
    print(f"\n[Exp 3] Memory Config Sweep (72B) — batches: {BATCH_SIZES_MEM}")

    for q_mem in ["int4", "int8", "fp16", "fp32"]:
        print(f"\n  --> Quant: {q_mem}")
        update_model_config("Qwen2_5_72BCfg", q_mem)
        cold_load_s = calculate_cold_load(72, q_mem)
        for h, c in [(16, 32), (16, 64), (32, 32), (32, 64)]:
            cfg_label = f"{h}GB_Host+{c}GB_CXL"
            print(f"    Config: {cfg_label}...")
            update_memory_config(h, c)
            for batch in BATCH_SIZES_MEM:
                update_batch_config(batch)
                for sim in SIMULATORS:
                    sim_name = clean_sim_name(sim)
                    try:
                        ret  = subprocess.run(
                            ["python", sim], capture_output=True,
                            text=True, timeout=900)
                        mets = parse_metrics(ret.stdout, sim)
                        if "semduplex" in sim.lower() or "llmflash" in sim.lower():
                            print(f"      {sim_name} B={batch}: "
                                  f"Decode={mets['TPS']:.4f}")
                        results.append({
                            "Experiment": "Memory", "Model": 72,
                            "Simulator": sim_name, "Quant": q_mem,
                            "MemConfig": cfg_label, "BatchSize": batch,
                            "Cold_Load_s": cold_load_s, **mets
                        })
                    except Exception as e:
                        print(f"      {sim_name}: ERROR — {e}")

# ---------------------------------------------------------
    # EXPERIMENT 4: Batch Size Sweep — 72B ALL QUANTS, tight memory
    # ---------------------------------------------------------
    print(f"\n[Exp 4] Batch Sweep (72B All Quants | "
          f"{EXP4_HOST_GB}H+{EXP4_CXL_GB}C — tight memory)")
          
    update_memory_config(EXP4_HOST_GB, EXP4_CXL_GB)

    for quant in ["fp32", "fp16", "int8", "int4"]:
        print(f"\n  >>> Starting Quantization: {quant.upper()} <<<")
        update_model_config("Qwen2_5_72BCfg", quant)
        cold_load_s = calculate_cold_load(72, quant)

        for batch in BATCH_SIZES:
            print(f"  Batch={batch}...")
            update_batch_config(batch)
            for sim in SIMULATORS:
                sim_name = clean_sim_name(sim)
                try:
                    ret  = subprocess.run(
                        ["python", sim], capture_output=True,
                        text=True, timeout=900)
                    mets = parse_metrics(ret.stdout, sim)
                    print(f"    {sim_name}: Decode={mets['TPS']:.4f}")
                    results.append({
                        "Experiment": "BatchSweep", "Model": 72,
                        "Simulator": sim_name, "Quant": quant,  # Fixed hardcoded "int8" here!
                        "MemConfig": f"{EXP4_HOST_GB}H+{EXP4_CXL_GB}C",
                        "BatchSize": batch,
                        "Cold_Load_s": cold_load_s, **mets
                    })
                except Exception as e:
                    print(f"    {sim_name}: ERROR — {e}")

    # ---------------------------------------------------------
    # EXP 5: BatchSweep ALL QUANTS at B=128 (Serving Focus)
    # ---------------------------------------------------------
    print("\n[Exp 5] BatchSweep ALL QUANTS — B=128 Serving")
    for quant in ["fp32", "fp16", "int8", "int4"]:
        print(f"  Quant={quant} B=128...")
        update_model_config("Qwen2_5_72BCfg", quant)
        update_memory_config(EXP4_HOST_GB, EXP4_CXL_GB)  # 16H+32C tight
        #update_batch_config(128)
        cold_load_s = calculate_cold_load(72, quant)
        for batch in BATCH_SIZES:
            print(f"  Batch={batch}...")
            update_batch_config(batch)
            for sim in SIMULATORS:
                sim_name = clean_sim_name(sim)
                try:
                    ret  = subprocess.run(
                        ["python", sim], capture_output=True,
                        text=True, timeout=900)
                    mets = parse_metrics(ret.stdout, sim)
                    print(f"    {sim_name}: Decode={mets['TPS']:.4f}")
                    results.append({
                        "Experiment": "BatchSweepAllQuants", "Model": 72,
                        "Simulator": sim_name, "Quant": quant,
                        "MemConfig": f"{EXP4_HOST_GB}H+{EXP4_CXL_GB}C",
                        "BatchSize": 128,
                        "Cold_Load_s": cold_load_s, **mets
                    })
                except Exception as e:
                    print(f"    {sim_name}: ERROR — {e}")






    # ---------------------------------------------------------
    # SAVE & REPORT
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    generate_write_stall_proof(df)
    print_batch_analysis(df)

    target_cols = [
        "Experiment", "Model", "Simulator", "Quant", "MemConfig", "BatchSize",
        "TPS", "Prefill_TPS", "Cold_Load_s",
        "Read_Op_Pct", "Write_Op_Pct", "Write_Util_Pct"
    ]
    final_cols = [c for c in target_cols if c in df.columns]
    df[final_cols].to_csv("final_results_with_coldload.csv", index=False)
    print(f"\n>>> Data saved to final_results_with_coldload.csv")

    # Restore defaults
    update_memory_config(32, 64)
    update_model_config("Qwen2_5_72BCfg", "fp32")
    update_batch_config(1)


# ==========================================
# TIMELINE TRACE TRACKER (Layer-by-Layer)
# ==========================================
def track_timeline_for_plot():
    from tiers import transfer_time_s, CXL_DRAM, HOST_DRAM, CXL_SSD_NAND, Tier, NVME_STREAM_BW, NVME_STREAM_LAT_S
    from model_cfg import build_layers, Qwen2_5_72BCfg
    from sim_cfg import host_dram_capacity_bytes, cxl_dev_dram_capacity_bytes

    print("\n" + "=" * 70)
    print("GENERATING EXACT LAYER-BY-LAYER TIMELINE DATA")
    print("=" * 70)

    B = 128
    NUM_TOKENS_TO_TRACK = 3
    QUANT = "int8"
    bpp = 1.0
    DUPLEX_PENALTY = 1.15

    ssd_tier   = Tier("Host SSD (stream)", NVME_STREAM_BW, NVME_STREAM_LAT_S)
    cxl_bw_gbps  = CXL_DRAM.bw_Bps / 1e9
    host_bw_gbps = HOST_DRAM.bw_Bps / 1e9
    ssd_bw_gbps  = ssd_tier.bw_Bps / 1e9

    standard_layers = build_layers(Qwen2_5_72BCfg(), sequence_length=512)

    decomposed_layers = []
    for i, L in enumerate(standard_layers):
        if L["kind"] == "DecoderBlock":
            raw_bytes = int((L["bytes"] / 4.0) * bpp)
            raw_flops = L["flops"]
            decomposed_layers.append({
                "name": f"Layer_{i}_Attn", "kind": "Attention",
                "bytes": int(raw_bytes * 0.17),
                "flops": int(raw_flops * 0.33),
                "kv_bytes": 2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * bpp,
                "sparsity": 0.05, "placement": "Host_DRAM"
            })
            decomposed_layers.append({
                "name": f"Layer_{i}_MLP", "kind": "MLP",
                "bytes": int(raw_bytes * 0.83),
                "flops": int(raw_flops * 0.67),
                "kv_bytes": 0, "sparsity": 0.54, "placement": "CXL_DRAM"
            })

    timeline_records = []

    # --------------------------------------------------
    # FLEXGEN: SSD demand paging
    # --------------------------------------------------
    current_time = 0.0
    for token in range(1, NUM_TOKENS_TO_TRACK + 1):
        for idx, L in enumerate(standard_layers):
            if L["kind"] != "DecoderBlock": continue
            layer_name      = f"Layer_{idx}"
            layer_bytes     = int((L["bytes"] / 4.0) * bpp)
            kv_bytes_per_seq = 2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * bpp

            read_time = transfer_time_s(layer_bytes, ssd_tier)
            timeline_records.append({
                "Simulator": "FlexGen", "Token": token, "Layer": layer_name,
                "Event": "Weight_Read", "Start_Time_s": current_time,
                "End_Time_s": current_time + read_time,
                "Bandwidth_GBps": ssd_bw_gbps          # positive = read
            })
            current_time += read_time

            kv_write_bytes = kv_bytes_per_seq * (512 + token) * B
            write_time     = transfer_time_s(kv_write_bytes, ssd_tier)
            timeline_records.append({
                "Simulator": "FlexGen", "Token": token, "Layer": layer_name,
                "Event": "KV_Write_Stall", "Start_Time_s": current_time,
                "End_Time_s": current_time + write_time,
                "Bandwidth_GBps": -ssd_bw_gbps          # negative = write/stall
            })
            current_time += write_time

    # --------------------------------------------------
    # LLMFLASH: CXL streaming simplex
    # --------------------------------------------------
    current_time = 0.0
    for token in range(1, NUM_TOKENS_TO_TRACK + 1):
        for idx, L in enumerate(standard_layers):
            if L["kind"] != "DecoderBlock": continue
            layer_name  = f"Layer_{idx}"
            layer_bytes = int((L["bytes"] / 4.0) * bpp)
            kv_bytes    = 2 * L.get("kv_heads", 8) * L.get("head_dim", 128) * bpp * B

            read_time = transfer_time_s(layer_bytes, CXL_DRAM)
            timeline_records.append({
                "Simulator": "LLMFlash", "Token": token, "Layer": layer_name,
                "Event": "Weight_Read", "Start_Time_s": current_time,
                "End_Time_s": current_time + read_time,
                "Bandwidth_GBps": cxl_bw_gbps           # positive = read
            })
            current_time += read_time

            if kv_bytes > 0:
                write_time = transfer_time_s(kv_bytes, CXL_DRAM)
                timeline_records.append({
                    "Simulator": "LLMFlash", "Token": token, "Layer": layer_name,
                    "Event": "KV_Write_Stall", "Start_Time_s": current_time,
                    "End_Time_s": current_time + write_time,
                    "Bandwidth_GBps": -cxl_bw_gbps       # negative = write/stall
                })
                current_time += write_time

    # --------------------------------------------------
    # SEMDUPLEX: Host + CXL full-duplex
    # --------------------------------------------------
    current_time = 0.0
    for token in range(1, NUM_TOKENS_TO_TRACK + 1):
        for L in decomposed_layers:
            sz       = L["bytes"]
            kv_total = L["kv_bytes"] * B
            is_cxl   = L["placement"] == "CXL_DRAM"

            # Duplex active on any CXL read when KV writes are in flight
            duplex_active = is_cxl and (kv_total > 0)

            if is_cxl:
                actual_read_time = transfer_time_s(sz, CXL_DRAM)
                if duplex_active:
                    actual_read_time *= DUPLEX_PENALTY
                read_bw = cxl_bw_gbps
            else:
                actual_read_time = transfer_time_s(sz, HOST_DRAM)
                read_bw = host_bw_gbps                  # actual host BW, not 0.0

            timeline_records.append({
                "Simulator": "SemDuplex", "Token": token, "Layer": L["name"],
                "Event": f"{L['kind']}_Read", "Start_Time_s": current_time,
                "End_Time_s": current_time + actual_read_time,
                "Bandwidth_GBps": read_bw
            })

            # Background write on idle CXL lane
            if kv_total > 0:
                write_time = transfer_time_s(kv_total, CXL_DRAM)
                timeline_records.append({
                    "Simulator": "SemDuplex", "Token": token, "Layer": L["name"],
                    "Event": "KV_Write_Background", "Start_Time_s": current_time,
                    "End_Time_s": current_time + write_time,
                    "Bandwidth_GBps": -cxl_bw_gbps       # negative = write lane
                })

            current_time += actual_read_time

    df = pd.DataFrame(timeline_records)
    csv_file = "all_simulators_timeline.csv"
    df.to_csv(csv_file, index=False)
    print(f">>> Successfully generated layer-by-layer trace for all simulators!")
    print(f">>> Saved to {csv_file}")


# Update your standard execution block
if __name__ == "__main__":
    # 1. Run your standard experiments
    run_experiments()
    
    # 2. Run the newly added timeline tracker
    track_timeline_for_plot()

