"""
trace_runner.py — drives the 4 simulators with custom (prefill, decode) pairs
from a real workload trace (e.g., ShareGPT).

Originals stay byte-identical: for each run we copy the simulator + its deps
into a tempdir, patch the constants there, and execute. Nothing in the main
repo is modified.

Usage:
    python trace_workload/trace_runner.py \
        --pairs trace_workload/sharegpt_lens.json \
        --n 20 \
        --batch 128 \
        --quant fp16 \
        --mem 16H+64C \
        --out trace_workload/trace_results.csv

The script writes one row per (pair_idx, simulator) with Decode_TPS,
Prefill_TPS, prefill_len, decode_len.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIM_FILES = {
    "FlexGen":   "flexgen_baseline.py",
    "LIA":       "lia_baseline.py",
    "LLMFlash":  "llmflash_baseline.py",
    "SemSched":  "semduplex_scheduler.py",
}

DEPS = ["sim_cfg.py", "tiers.py", "model_cfg.py", "cxl_link.py"]


MEM_MAP = {
    "16H+32C": (16, 32), "16H+64C": (16, 64),
    "32H+32C": (32, 32), "32H+64C": (32, 64),
}


def patch_sim_cfg(src: str, decode: int, batch: int,
                  host_gb: int, cxl_gb: int) -> str:
    """Replace TOKENS, BATCH_SIZE, host_dram, cxl_dev_dram in sim_cfg.py."""
    src = re.sub(r"^TOKENS\s*=\s*\d+",
                 f"TOKENS = {decode}", src, count=1, flags=re.MULTILINE)
    src = re.sub(r"^BATCH_SIZE\s*=\s*\d+",
                 f"BATCH_SIZE = {batch}", src, count=1, flags=re.MULTILINE)
    src = re.sub(r"^host_dram_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
                 f"host_dram_capacity_bytes = {host_gb} * GiB",
                 src, count=1, flags=re.MULTILINE)
    src = re.sub(r"^cxl_dev_dram_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
                 f"cxl_dev_dram_capacity_bytes = {cxl_gb} * GiB",
                 src, count=1, flags=re.MULTILINE)
    return src


def patch_model_cfg(src: str, quant: str, model_cls: str) -> str:
    """Replace QUANT and DEFAULT_MODEL_CFG in model_cfg.py."""
    src = re.sub(r'^QUANT\s*=\s*"[^"]+"',
                 f'QUANT = "{quant}"', src, count=1, flags=re.MULTILINE)
    src = re.sub(r"^DEFAULT_MODEL_CFG\s*=\s*\w+",
                 f"DEFAULT_MODEL_CFG = {model_cls}",
                 src, count=1, flags=re.MULTILINE)
    return src


def patch_simulator(src: str, prefill: int) -> str:
    """Replace PREFILL_TOKENS (or NUM_PREFILL_TOKENS) in a simulator source."""
    src = re.sub(r"NUM_PREFILL_TOKENS(\s*=\s*)\d+",
                 f"NUM_PREFILL_TOKENS\\g<1>{prefill}", src)
    src = re.sub(r"^PREFILL_TOKENS(\s*=\s*)\d+",
                 f"PREFILL_TOKENS\\g<1>{prefill}", src, flags=re.MULTILINE)
    return src


def parse_output(stdout: str) -> dict:
    """Pick Decode throughput and Prefill throughput from simulator stdout."""
    out = {"Decode_TPS": None, "Prefill_TPS": None}
    for line in stdout.splitlines():
        m = re.match(r"Decode throughput:\s*([\d.]+)", line)
        if m: out["Decode_TPS"] = float(m.group(1))
        m = re.match(r"Prefill throughput:\s*([\d.]+)", line)
        if m: out["Prefill_TPS"] = float(m.group(1))
    return out


def run_one(sim_name: str, prefill: int, decode: int, batch: int,
            quant: str, mem: str, timeout_s: int = 180) -> dict:
    """Run one simulator with the given config, return parsed metrics."""
    sim_file = SIM_FILES[sim_name]

    host_gb, cxl_gb = MEM_MAP.get(mem, (16, 64))
    model_cls = "Qwen2_5_72BCfg"
    env = os.environ.copy()

    with tempfile.TemporaryDirectory(prefix="cxl_trace_") as tmp:
        for dep in DEPS:
            shutil.copy(os.path.join(REPO, dep), tmp)

        # Patch sim_cfg.py in tempdir
        cfg_path = os.path.join(tmp, "sim_cfg.py")
        with open(cfg_path) as f:
            cfg_src = f.read()
        with open(cfg_path, "w") as f:
            f.write(patch_sim_cfg(cfg_src, decode=decode, batch=batch,
                                  host_gb=host_gb, cxl_gb=cxl_gb))

        # Patch model_cfg.py in tempdir
        mcfg_path = os.path.join(tmp, "model_cfg.py")
        with open(mcfg_path) as f:
            mcfg_src = f.read()
        with open(mcfg_path, "w") as f:
            f.write(patch_model_cfg(mcfg_src, quant=quant, model_cls=model_cls))

        # Copy + patch the simulator file
        sim_src_path = os.path.join(REPO, sim_file)
        sim_tgt_path = os.path.join(tmp, sim_file)
        with open(sim_src_path) as f:
            sim_src = f.read()
        with open(sim_tgt_path, "w") as f:
            f.write(patch_simulator(sim_src, prefill=prefill))

        try:
            result = subprocess.run(
                [sys.executable, sim_file],
                cwd=tmp, env=env, capture_output=True, text=True,
                timeout=timeout_s,
            )
            if result.returncode != 0:
                return {"Decode_TPS": None, "Prefill_TPS": None,
                        "err": result.stderr[-500:]}
            return parse_output(result.stdout)
        except subprocess.TimeoutExpired:
            return {"Decode_TPS": None, "Prefill_TPS": None, "err": "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="trace_workload/sharegpt_lens.json")
    ap.add_argument("--n", type=int, default=20,
                    help="number of (prefill, decode) pairs to sample")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--quant", default="fp16")
    ap.add_argument("--mem", default="16H+64C")
    ap.add_argument("--decode-cap", type=int, default=64,
                    help="cap decode length for smoke runs (default 64)")
    ap.add_argument("--out", default="trace_workload/trace_results.csv")
    ap.add_argument("--sims", nargs="+",
                    default=list(SIM_FILES.keys()))
    args = ap.parse_args()

    pairs_path = os.path.join(REPO, args.pairs) if not os.path.isabs(args.pairs) else args.pairs
    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out

    with open(pairs_path) as f:
        all_pairs = json.load(f)
    # Stable sample (first N) so re-runs are reproducible
    pairs = all_pairs[: args.n]

    print(f"Driving {len(pairs)} pairs × {len(args.sims)} sims "
          f"= {len(pairs)*len(args.sims)} runs")
    print(f"Config: batch={args.batch} quant={args.quant} mem={args.mem} "
          f"decode-cap={args.decode_cap}")

    rows = []
    for idx, (pf, dc) in enumerate(pairs):
        dc_capped = min(dc, args.decode_cap)
        for sim in args.sims:
            print(f"  [{idx+1}/{len(pairs)}] {sim:10s} pf={pf:5d} dc={dc_capped:4d} ...",
                  end="", flush=True)
            res = run_one(sim, prefill=pf, decode=dc_capped,
                          batch=args.batch, quant=args.quant, mem=args.mem)
            ok = res.get("Decode_TPS") is not None
            print(f" {'OK ' if ok else 'FAIL'} "
                  f"dec={res.get('Decode_TPS')} pf={res.get('Prefill_TPS')}")
            rows.append({
                "pair_idx": idx,
                "Simulator": sim,
                "prefill_len": pf,
                "decode_len": dc_capped,
                "Decode_TPS": res.get("Decode_TPS"),
                "Prefill_TPS": res.get("Prefill_TPS"),
                "err": res.get("err", ""),
            })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["Decode_TPS"] is not None)
    print(f"\nWrote {len(rows)} rows to {out_path} ({ok} OK, {len(rows)-ok} failed)")


if __name__ == "__main__":
    main()
