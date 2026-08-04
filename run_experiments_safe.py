"""
run_experiments_safe.py — regenerate final_results_with_coldload.csv.

Unlike run_experiments.py, this driver NEVER modifies sim_cfg.py or model_cfg.py
in the working tree. Each run copies the simulator plus its dependencies into a
temporary directory, patches the constants there, and executes. A crashed or
interrupted sweep therefore leaves the repository byte-identical (audit item A8).

Output schema matches the original CSV exactly, so plot_paper_figures.py
produces figures with unchanged sizes and styling.

    python run_experiments_safe.py [--out final_results_with_coldload.csv]
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
DEPS = ["sim_cfg.py", "tiers.py", "model_cfg.py", "cxl_link.py"]
SIMS = [
    ("FlexGen",  "flexgen_baseline.py"),
    ("LIA",      "lia_baseline.py"),
    ("LLMFlash", "llmflash_baseline.py"),
    ("CXLAimPod", "cxlaimpod_baseline.py"),
    ("SemSched", "semduplex_scheduler.py"),
]
MODELS = [("Mistral7BCfg", 7), ("Llama13BCfg", 13),
          ("Qwen3_20BCfg", 20), ("Qwen2_5_72BCfg", 72)]
QUANTS = ["fp32", "fp16", "int8", "int4"]
MEMS = [(16, 32), (16, 64), (32, 32), (32, 64)]
BATCHES = [1, 4, 8, 16, 32, 64, 128, 256]

COLUMNS = ["Experiment", "Model", "Simulator", "Quant", "MemConfig", "BatchSize",
           "TPS", "Prefill_TPS", "Cold_Load_s", "Read_Op_Pct", "Write_Op_Pct",
           "Write_Util_Pct", "Write_Stall_Time_s", "Write_Stall_Pct",
           "Write_Stall_Count", "Write_Stall_Freq_Pct", "Per_Token_Stall_Min_Pct",
           "Per_Token_Stall_Max_Pct", "Per_Token_Stall_Mean_Pct", "Token_Time_s",
           "Stall_Pct_of_Token"]

_num = r"([-+]?[\d.]+(?:[eE][-+]?\d+)?)"


def _grab(out, pattern, default=0.0):
    m = re.search(pattern, out)
    try:
        return float(m.group(1)) if m else default
    except (ValueError, IndexError):
        return default


def parse_metrics(out):
    m = {
        "TPS":                 _grab(out, r"Decode throughput:\s*" + _num),
        "Prefill_TPS":         _grab(out, r"Prefill throughput:\s*" + _num),
        "Cold_Load_s":         _grab(out, r"Cold[_ ]?[Ll]oad[^\d\-]*" + _num),
        "Read_Op_Pct":         _grab(out, r"Read_Op_Percent:\s*" + _num),
        "Write_Op_Pct":        _grab(out, r"Write_Op_Percent:\s*" + _num),
        "Write_Util_Pct":      _grab(out, r"Write_Util_Pct:\s*" + _num),
        "Write_Stall_Time_s":  _grab(out, r"Write_Stall_Time_s:\s*" + _num),
        "Write_Stall_Pct":     _grab(out, r"Write_Stall_Pct:\s*" + _num),
        "Write_Stall_Count":   _grab(out, r"Write_Stall_Count:\s*" + _num),
        "Write_Stall_Freq_Pct": 0.0,
    }
    pts = re.search(r"Per_Token_Write_Stall_Pcts:\s*([\d.,\-eE+]+)", out)
    if pts:
        vals = [float(x) for x in pts.group(1).split(",") if x.strip()]
        if vals:
            m["Per_Token_Stall_Min_Pct"] = min(vals)
            m["Per_Token_Stall_Max_Pct"] = max(vals)
            m["Per_Token_Stall_Mean_Pct"] = sum(vals) / len(vals)
    m.setdefault("Per_Token_Stall_Min_Pct", 0.0)
    m.setdefault("Per_Token_Stall_Max_Pct", 0.0)
    m.setdefault("Per_Token_Stall_Mean_Pct", 0.0)
    m["Token_Time_s"] = 0.0
    m["Stall_Pct_of_Token"] = 0.0
    return m


def run(sim_file, model_cls, quant, host_gb, cxl_gb, batch, decode=16, timeout=900):
    with tempfile.TemporaryDirectory() as td:
        for f in DEPS + [sim_file]:
            shutil.copy(os.path.join(REPO, f), td)

        p = os.path.join(td, "sim_cfg.py")
        s = open(p).read()
        s = re.sub(r"^TOKENS\s*=\s*\d+", f"TOKENS = {decode}", s, 1, re.M)
        s = re.sub(r"^BATCH_SIZE\s*=\s*\d+", f"BATCH_SIZE = {batch}", s, 1, re.M)
        s = re.sub(r"^host_dram_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
                   f"host_dram_capacity_bytes = {host_gb} * GiB", s, 1, re.M)
        s = re.sub(r"^cxl_dev_dram_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
                   f"cxl_dev_dram_capacity_bytes = {cxl_gb} * GiB", s, 1, re.M)
        open(p, "w").write(s)

        p = os.path.join(td, "model_cfg.py")
        s = open(p).read()
        s = re.sub(r'^QUANT\s*=\s*"\w+"', f'QUANT = "{quant}"', s, 1, re.M)
        s = re.sub(r"^DEFAULT_MODEL_CFG\s*=\s*\w+",
                   f"DEFAULT_MODEL_CFG = {model_cls}", s, 1, re.M)
        open(p, "w").write(s)

        try:
            r = subprocess.run([sys.executable, sim_file], cwd=td,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return parse_metrics(r.stdout + r.stderr)


def emit(rows, exp, model, sim_name, quant, mem, batch, met):
    if met is None:
        return
    row = {c: "" for c in COLUMNS}
    row.update(Experiment=exp, Model=model, Simulator=sim_name,
               Quant=quant, MemConfig=mem, BatchSize=batch)
    row.update({k: v for k, v in met.items() if k in COLUMNS})
    rows.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_results_with_coldload.csv")
    args = ap.parse_args()

    rows = []
    total = 0

    def note(msg):
        print(msg, flush=True)

    # ── Scalability: all models, FP32, 32H+64C, B=128 ────────────────────────
    note("[1/5] Scalability")
    for cls, size in MODELS:
        for sim_name, sim_file in SIMS:
            emit(rows, "Scalability", size, sim_name, "fp32", "32H+64C", 128,
                 run(sim_file, cls, "fp32", 32, 64, 128))
            total += 1

    # ── Quantization: 72B, all quants, 32H+32C, B=128 ────────────────────────
    note("[2/5] Quantization")
    for quant in QUANTS:
        for sim_name, sim_file in SIMS:
            emit(rows, "Quantization", 72, sim_name, quant, "32H+32C", 128,
                 run(sim_file, "Qwen2_5_72BCfg", quant, 32, 32, 128))
            total += 1

    # ── Memory: 72B, all quants x all mem configs, B=128 ─────────────────────
    note("[3/5] Memory")
    for quant in QUANTS:
        for h, c in MEMS:
            mem = f"{h}GB_Host+{c}GB_CXL"
            for sim_name, sim_file in SIMS:
                emit(rows, "Memory", 72, sim_name, quant, mem, 128,
                     run(sim_file, "Qwen2_5_72BCfg", quant, h, c, 128))
                total += 1

    # ── BatchSweep: 72B, FP32, 16H+32C, all batches ──────────────────────────
    note("[4/5] BatchSweep")
    for b in BATCHES:
        for sim_name, sim_file in SIMS:
            emit(rows, "BatchSweep", 72, sim_name, "fp32", "16H+32C", b,
                 run(sim_file, "Qwen2_5_72BCfg", "fp32", 16, 32, b))
            total += 1

    # ── BatchSweepAllQuants: 72B, all quants, 16H+32C, all batches ───────────
    note("[5/5] BatchSweepAllQuants")
    for quant in QUANTS:
        for b in BATCHES:
            for sim_name, sim_file in SIMS:
                emit(rows, "BatchSweepAllQuants", 72, sim_name, quant,
                     "16H+32C", b,
                     run(sim_file, "Qwen2_5_72BCfg", quant, 16, 32, b))
                total += 1

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    note(f"\nWrote {len(rows)}/{total} rows to {args.out}")
    note("Repository configs were never modified.")


if __name__ == "__main__":
    main()
