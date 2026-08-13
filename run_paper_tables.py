"""
run_paper_tables.py — regenerate the Section V tables from whatever is in the
working tree, so no reported number is traceable only to a shell history.

Every table in the paper had been produced by an ad-hoc script that was not
committed, which meant a fix to the simulator did not invalidate anything
visibly: the tables kept saying what they said. This file closes that gap. It
is the only sanctioned way to produce a number that goes into the paper.

    python run_paper_tables.py                # both platforms, both quants
    python run_paper_tables.py --cpu-only
    python run_paper_tables.py --diag         # + placement and staging dump

Each cell runs in a temporary directory holding a private copy of the whole
simulator, with sim_cfg.py and model_cfg.py rewritten for that cell. Nothing
mutates the repository, so a failed sweep cannot leave the tree configured for
one experiment.
"""
import argparse
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
    ("AimPod",   "cxlaimpod_baseline.py"),
    ("LLMFlash", "llmflash_baseline.py"),
    ("SemSched", "semduplex_scheduler.py"),
]

# RTX 5090: 32 GB GDDR7, of which 28 GB is usable for weights after activations
# and the runtime's own footprint. 0 = the CPU-only platform, which is what both
# CMM-H papers actually characterise.
GPU_GB = 28
GRID = [16]                       # host DRAM
CXLS = [32, 48, 64]               # CXL device DRAM
QUANTS = ["fp16", "int8"]
BATCH, DECODE = 128, 16


def patch(td, quant, h, c, gpu, warmup=True):
    p = os.path.join(td, "sim_cfg.py")
    s = open(p).read()
    s = re.sub(r"^TOKENS\s*=\s*\d+", f"TOKENS = {DECODE}", s, 1, re.M)
    s = re.sub(r"^BATCH_SIZE\s*=\s*\d+", f"BATCH_SIZE = {BATCH}", s, 1, re.M)
    s = re.sub(r"^host_dram_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
               f"host_dram_capacity_bytes = {h} * GiB", s, 1, re.M)
    s = re.sub(r"^cxl_dev_dram_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
               f"cxl_dev_dram_capacity_bytes = {c} * GiB", s, 1, re.M)
    s = re.sub(r"^gpu_hbm_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
               f"gpu_hbm_capacity_bytes = {gpu} * GiB", s, 1, re.M)
    open(p, "w").write(s)

    p = os.path.join(td, "model_cfg.py")
    s = open(p).read()
    s = re.sub(r'^QUANT\s*=\s*"\w+"', f'QUANT = "{quant}"', s, 1, re.M)
    s = re.sub(r"^DEFAULT_MODEL_CFG\s*=\s*\w+",
               "DEFAULT_MODEL_CFG = Qwen2_5_72BCfg", s, 1, re.M)
    open(p, "w").write(s)

    if not warmup:
        p = os.path.join(td, "semduplex_scheduler.py")
        s = open(p).read()
        s = re.sub(r"^ENABLE_PREFILL_WARMUP\s*=\s*True",
                   "ENABLE_PREFILL_WARMUP = False", s, 1, re.M)
        open(p, "w").write(s)


def run(sim, quant, h, c, gpu, warmup=True, diag=False):
    """One simulator, one cell. Returns (decode_tps, prefill_tps, stdout)."""
    with tempfile.TemporaryDirectory() as td:
        for f in DEPS + [s for _, s in SIMS]:
            shutil.copy(os.path.join(REPO, f), td)
        patch(td, quant, h, c, gpu, warmup)
        r = subprocess.run([sys.executable, sim], cwd=td,
                           capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        d = re.search(r"Decode throughput:\s*([\d.]+)", out)
        p = re.search(r"Prefill throughput:\s*([\d.]+)", out)
        if d is None and diag:
            print(out[-1500:], file=sys.stderr)
        return (float(d.group(1)) if d else None,
                float(p.group(1)) if p else None, out)


def comparison(gpu, diag=False):
    tag = f"+RTX 5090 ({GPU_GB} GB)" if gpu else "CPU-only (no accelerator)"
    print(f"\n{'=' * 78}\n{tag}   Qwen2.5 72B, B={BATCH}, {DECODE} decode steps"
          f"\n{'=' * 78}")
    names = [n for n, _ in SIMS]
    print(f"{'Quant':<6}{'Memory':<11}" + "".join(f"{n:>10}" for n in names)
          + f"{'ratio':>9}")
    print("-" * 78)
    wins = total = 0
    for quant in QUANTS:
        for c in CXLS:
            for h in GRID:
                tps = {}
                for name, sim in SIMS:
                    d, _, _ = run(sim, quant, h, c, gpu, diag=diag)
                    tps[name] = d
                ours = tps["SemSched"]
                best = max(v for k, v in tps.items()
                           if k != "SemSched" and v is not None)
                total += 1
                wins += ours > best
                print(f"{quant.upper():<6}{f'{h}H+{c}C':<11}"
                      + "".join(f"{tps[n]:>10.2f}" if tps[n] is not None
                                else f"{'--':>10}" for n in names)
                      + f"{ours / best:>8.2f}x")
    print("-" * 78)
    print(f"SemSched leads {wins}/{total}")


def warmup_ablation(gpu=0):
    """Phase-1 staging is the mechanism the paper claims. Measure what it is
    worth by turning it off, on exactly the configurations we report."""
    print(f"\n{'=' * 78}\nPhase-1 warmup ablation"
          f"{'  (+RTX 5090)' if gpu else '  (CPU-only)'}\n{'=' * 78}")
    print(f"{'Quant':<6}{'Memory':<11}{'best base':>11}{'warmup ON':>11}"
          f"{'warmup OFF':>12}{'contributes':>13}{'ratio ON':>10}{'ratio OFF':>11}")
    print("-" * 78)
    for quant in QUANTS:
        for c in CXLS:
            for h in GRID:
                base = max(run(s, quant, h, c, gpu)[0] or 0
                           for n, s in SIMS if n != "SemSched")
                on, _, _ = run("semduplex_scheduler.py", quant, h, c, gpu, True)
                off, _, _ = run("semduplex_scheduler.py", quant, h, c, gpu, False)
                print(f"{quant.upper():<6}{f'{h}H+{c}C':<11}{base:>11.2f}"
                      f"{on:>11.2f}{off:>12.2f}{(on - off) / off * 100:>12.1f}%"
                      f"{on / base:>9.2f}x{off / base:>10.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                    help="warmup ablation only")
    ap.add_argument("--diag", action="store_true")
    a = ap.parse_args()

    if a.ablation:
        warmup_ablation(0)
        return
    if not a.gpu_only:
        comparison(0, a.diag)
    if not a.cpu_only:
        comparison(GPU_GB, a.diag)


if __name__ == "__main__":
    main()
