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
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
DEPS = ["sim_cfg.py", "tiers.py", "model_cfg.py", "cxl_link.py", "pipeline.py"]

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

# An accelerator is two things, and attaching only one of them is a modelling
# error that quietly understates every accelerated row: it is HBM capacity AND a
# compute engine. Setting gpu_hbm_capacity_bytes alone runs GPU memory behind
# the EPYC's 14.4 TFLOPS, so any layer where compute exceeds transfer is timed
# on the wrong engine. Both are set together here.
#   EPYC 9454 x2 : 96 cores, AVX-512 (64 FLOP/cyc), 2.75 GHz, 0.85 ->  14.4 TF
#   RTX 5090     : 170 SMs, 512 FLOP/cyc/SM, 2.41 GHz, 0.70        -> 146.9 TF
ENGINES = {
    0:      (96,  64.0,  2.75e9, 0.85),
    GPU_GB: (170, 512.0, 2.41e9, 0.70),
}
GRID = [16]                       # host DRAM
CXLS = [32, 48, 64]               # CXL device DRAM
QUANTS = ["fp16", "int8"]
BATCH, DECODE = 128, 16


def patch(td, quant, h, c, gpu, warmup=True, batch=BATCH, kv_tier="cxl"):
    p = os.path.join(td, "sim_cfg.py")
    s = open(p).read()
    s = re.sub(r"^TOKENS\s*=\s*\d+", f"TOKENS = {DECODE}", s, 1, re.M)
    s = re.sub(r"^BATCH_SIZE\s*=\s*\d+", f"BATCH_SIZE = {batch}", s, 1, re.M)
    s = re.sub(r"^host_dram_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
               f"host_dram_capacity_bytes = {h} * GiB", s, 1, re.M)
    s = re.sub(r"^cxl_dev_dram_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
               f"cxl_dev_dram_capacity_bytes = {c} * GiB", s, 1, re.M)
    s = re.sub(r"^gpu_hbm_capacity_bytes\s*=\s*\S+\s*\*\s*GiB",
               f"gpu_hbm_capacity_bytes = {gpu} * GiB", s, 1, re.M)
    cores, fpc, freq, eff = ENGINES[gpu]
    s = re.sub(r"^cpu_cores\s*=\s*\d+", f"cpu_cores = {cores}", s, 1, re.M)
    s = re.sub(r"^flops_per_cycle_per_core\s*=\s*[\d.]+",
               f"flops_per_cycle_per_core = {fpc}", s, 1, re.M)
    s = re.sub(r"^cpu_freq_hz\s*=\s*[\d.e+]+", f"cpu_freq_hz = {freq}", s, 1, re.M)
    s = re.sub(r"^parallel_efficiency\s*=\s*[\d.]+",
               f"parallel_efficiency = {eff}", s, 1, re.M)
    open(p, "w").write(s)

    p = os.path.join(td, "model_cfg.py")
    s = open(p).read()
    s = re.sub(r'^QUANT\s*=\s*"\w+"', f'QUANT = "{quant}"', s, 1, re.M)
    s = re.sub(r"^DEFAULT_MODEL_CFG\s*=\s*\w+",
               "DEFAULT_MODEL_CFG = Qwen2_5_72BCfg", s, 1, re.M)
    open(p, "w").write(s)

    if not warmup or kv_tier != "cxl":
        p = os.path.join(td, "semduplex_scheduler.py")
        s = open(p).read()
        if not warmup:
            s = re.sub(r"^ENABLE_PREFILL_WARMUP\s*=\s*True",
                       "ENABLE_PREFILL_WARMUP = False", s, 1, re.M)
        s = re.sub(r'^KV_TIER\s*=\s*"\w+"', f'KV_TIER = "{kv_tier}"', s, 1, re.M)
        open(p, "w").write(s)


def run(sim, quant, h, c, gpu, warmup=True, diag=False, batch=BATCH,
        kv_tier="cxl"):
    """One simulator, one cell. Returns (decode_tps, prefill_tps, stdout)."""
    with tempfile.TemporaryDirectory() as td:
        for f in DEPS + [s for _, s in SIMS]:
            shutil.copy(os.path.join(REPO, f), td)
        patch(td, quant, h, c, gpu, warmup, batch, kv_tier)
        r = subprocess.run([sys.executable, sim], cwd=td,
                           capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        d = re.search(r"Decode throughput:\s*([\d.]+)", out)
        p = re.search(r"Prefill throughput:\s*([\d.]+)", out)
        if d is None and diag:
            print(out[-1500:], file=sys.stderr)
        return (float(d.group(1)) if d else None,
                float(p.group(1)) if p else None, out)


# ── ShareGPT replay ──────────────────────────────────────────────────────────
# Real prompt/response lengths rather than a fixed 512/16, which is the one thing
# a reviewer can check against a workload they know. The distribution is the only
# thing taken from the corpus: per-layer compute, transfer and KV costs still come
# from the model geometry, so this strengthens an input without changing how
# timing is computed.
def _patch_lengths(td, prefill, decode):
    for f, pat, rep in (
            ("semduplex_scheduler.py", r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}"),
            ("flexgen_baseline.py",    r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}"),
            ("lia_baseline.py",        r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}"),
            ("cxlaimpod_baseline.py",  r"^PREFILL_TOKENS = \d+",      f"PREFILL_TOKENS = {prefill}"),
            ("llmflash_baseline.py",   r"^NUM_PREFILL_TOKENS = \d+",  f"NUM_PREFILL_TOKENS = {prefill}"),
            ("llmflash_baseline.py",   r"^NUM_DECODE_TOKENS(\s*)= \d+", f"NUM_DECODE_TOKENS\\1= {decode}")):
        q = os.path.join(td, f)
        t = open(q).read()
        open(q, "w").write(re.sub(pat, rep, t, 1, re.M))
    q = os.path.join(td, "sim_cfg.py")
    t = open(q).read()
    open(q, "w").write(re.sub(r"^TOKENS\s*=\s*\d+", f"TOKENS = {decode}", t, 1, re.M))


def run_trace(sim, quant, h, c, gpu, prefill, decode):
    with tempfile.TemporaryDirectory() as td:
        for f in DEPS + [s for _, s in SIMS]:
            shutil.copy(os.path.join(REPO, f), td)
        patch(td, quant, h, c, gpu)
        _patch_lengths(td, prefill, decode)
        r = subprocess.run([sys.executable, sim], cwd=td, capture_output=True,
                           text=True, timeout=900)
        m = re.search(r"Decode throughput:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None


def sharegpt(gpu, quant="fp16", c=64, n=50, seed=20260815):
    import json, random, statistics as st
    pairs = json.load(open(os.path.join(REPO, "trace_workload/sharegpt_lens.json")))
    pairs = [(int(a), int(b)) for a, b in pairs if a > 0 and b > 0]
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:n]
    tag = f"+RTX 5090" if gpu else "CPU-only"
    print(f"\n{'=' * 78}\nShareGPT replay, {n} prompts, {quant.upper()} 16H+{c}C, "
          f"{tag}, B={BATCH}\n{'=' * 78}")
    pl = sorted(p for p, _ in pairs)
    print(f"prefill: median {st.median(pl):.0f}  P90 {pl[int(.9*len(pl))]}  max {max(pl)}")
    ratios, wins = [], 0
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        fut = {ex.submit(run_trace, sim, quant, 16, c, gpu, p, d): (i, name)
               for i, (p, d) in enumerate(pairs) for name, sim in SIMS}
        got = {}
        for f in cf.as_completed(fut):
            got[fut[f]] = f.result()
    for i, (p, d) in enumerate(pairs):
        v = {n_: got.get((i, n_)) for n_, _ in SIMS}
        if v.get("SemSched") is None:
            continue
        base = max((x for k, x in v.items() if k != "SemSched" and x), default=None)
        if not base:
            continue
        ratios.append(v["SemSched"] / base)
        wins += v["SemSched"] > base
    ratios.sort()
    q = lambda f: ratios[min(len(ratios) - 1, int(f * len(ratios)))]
    print(f"\n  prompts scored     {len(ratios)}")
    print(f"  win rate           {wins}/{len(ratios)} ({100*wins/len(ratios):.0f}%)")
    print(f"  median speedup     {st.median(ratios):.2f}x")
    print(f"  P10 / P90          {q(.10):.2f}x / {q(.90):.2f}x")
    print(f"  min / max          {ratios[0]:.2f}x / {ratios[-1]:.2f}x")
    return ratios


def _metrics(out):
    """Everything the simulators report, plus the derived wall-clock terms."""
    def g(pat, default=None):
        m = re.search(pat, out)
        return float(m.group(1)) if m else default
    dec = g(r"Decode throughput:\s*([\d.]+)")
    pre = g(r"Prefill throughput:\s*([\d.]+)")
    m = {
        "warmup_s":   g(r"Warmup time:\s*([\d.]+)", 0.0),
        "staged_B":   g(r"Warmup staged bytes:\s*([\d.]+)", 0.0),
        "prefill_tps": pre,
        "decode_tps":  dec,
        "wstall_ms":  (g(r"Write_Stall_Time_s:\s*([\d.]+)", 0.0) or 0.0) * 1e3,
    }
    # A run is one prompt then DECODE tokens for each of BATCH sequences.
    m["prefill_s"] = (512.0 / pre) if pre else None
    m["step_s"]    = (BATCH / dec) if dec else None
    m["decode_s"]  = (DECODE * BATCH / dec) if dec else None
    m["total_s"]   = ((m["prefill_s"] or 0) + (m["decode_s"] or 0)) or None
    return m


def detail(gpu):
    """Per-policy breakdown: staging, prefill, decode and wall clock."""
    tag = f"+RTX 5090 ({GPU_GB} GB)" if gpu else "CPU-only (no accelerator)"
    print(f"\n{'=' * 110}\n{tag}   Qwen2.5 72B, B={BATCH}, 512-token prompt, "
          f"{DECODE} decode steps\n{'=' * 110}")
    print(f"{'Quant':<5}{'Memory':<9}{'Policy':<10}{'warmup s':>9}{'staged':>8}"
          f"{'prefill t/s':>12}{'prefill s':>10}{'decode t/s':>11}{'step s':>9}"
          f"{'decode s':>10}{'total s':>9}{'w-stall ms':>11}")
    print("-" * 110)
    for quant in QUANTS:
        for c in CXLS:
            for h in GRID:
                for name, sim in SIMS:
                    m = _metrics(run(sim, quant, h, c, gpu)[2])
                    st = (f"{m['staged_B']/1024**3:.1f}G" if m["staged_B"] else "-")
                    print(f"{quant.upper():<5}{f'{h}H+{c}C':<9}{name:<10}"
                          f"{m['warmup_s']:>9.3f}{st:>8}"
                          f"{m['prefill_tps']:>12.3f}{m['prefill_s']:>10.1f}"
                          f"{m['decode_tps']:>11.2f}{m['step_s']:>9.3f}"
                          f"{m['decode_s']:>10.1f}{m['total_s']:>9.1f}"
                          f"{m['wstall_ms']:>11.3f}")
                print("-" * 110)


def comparison(gpu, diag=False, metric="decode"):
    tag = f"+RTX 5090 ({GPU_GB} GB)" if gpu else "CPU-only (no accelerator)"
    print(f"\n{'=' * 78}\n{tag}   Qwen2.5 72B, B={BATCH}, {DECODE} decode steps"
          f"   [{metric} t/s]\n{'=' * 78}")
    names = [n for n, _ in SIMS]
    print(f"{'Quant':<6}{'Memory':<11}" + "".join(f"{n:>10}" for n in names)
          + f"{'ratio':>9}")
    print("-" * 78)
    wins = total = 0
    idx = 1 if metric == "prefill" else 0   # run() returns (decode, prefill, out)
    for quant in QUANTS:
        for c in CXLS:
            for h in GRID:
                tps = {}
                for name, sim in SIMS:
                    r = run(sim, quant, h, c, gpu, diag=diag)
                    tps[name] = r[idx]
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


def kv_tier_sweep(gpu):
    """Where should the KV cache live?

    Each tier is charged both ways -- capacity reserved before placement, and
    every KV access timed at its bandwidth -- so the answer is not simply "the
    fastest tier". Putting KV in HBM reads it 66x faster than the device but
    evicts an equal mass of weights down to a slower tier, and the weights are
    read every step too. The question is which of the two exchanges is cheaper,
    and it is a measurement, not an argument.
    """
    tag = f"+RTX 5090 ({GPU_GB} GB)" if gpu else "CPU-only"
    tiers = ["cxl", "host"] + (["gpu"] if gpu else [])
    print(f"\n{'=' * 78}\nKV cache tier — SemSched, {tag}, B={BATCH}\n{'=' * 78}")
    print(f"{'Quant':<6}{'Memory':<11}" + "".join(f"{'KV in ' + t:>12}" for t in tiers)
          + f"{'best':>8}")
    print("-" * 78)
    for quant in QUANTS:
        for c in CXLS:
            for h in GRID:
                got = {t: run("semduplex_scheduler.py", quant, h, c, gpu,
                              kv_tier=t)[0] for t in tiers}
                best = max(got, key=lambda t: got[t] or 0)
                print(f"{quant.upper():<6}{f'{h}H+{c}C':<11}"
                      + "".join(f"{got[t]:>12.2f}" if got[t] is not None
                                else f"{'--':>12}" for t in tiers)
                      + f"{best:>8}")


BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]


def batch_sweep(gpu, quant, c, h=16):
    """Locate the sparsity-collapse crossover.

    The active fraction of an MLP sub-layer under LLM-in-a-Flash's own model is
    1 - (1 - p)^B with p = 0.46, which saturates at 1.000 by B = 16. Above that
    every policy reads every byte once per step and the step time is pinned to
    sum(bytes(tier) / bw(tier)) -- the byte-accounting floor -- so placement
    capacity is the only thing left that can differ and all five converge.
    Below it, semantic placement has something to act on. This sweep measures
    where that transition actually falls rather than asserting it.
    """
    names = [n for n, _ in SIMS]
    rows = []
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        fut = {ex.submit(run, sim, quant, h, c, gpu, True, False, b): (b, name)
               for b in BATCHES for name, sim in SIMS}
        got = {}
        for f in cf.as_completed(fut):
            b, name = fut[f]
            got[(b, name)] = f.result()[0]
    for b in BATCHES:
        tps = {n: got[(b, n)] for n in names}
        ours = tps["SemSched"]
        best = max(v for k, v in tps.items() if k != "SemSched" and v is not None)
        rows.append((b, tps, ours / best if best else 0.0))
    return rows


def batch_table(gpu):
    tag = f"+RTX 5090 ({GPU_GB} GB)" if gpu else "CPU-only"
    names = [n for n, _ in SIMS]
    for quant in QUANTS:
        for c in CXLS:
            print(f"\n{'=' * 78}\n{tag}   Qwen2.5 72B {quant.upper()}  16H+{c}C"
                  f"   batch sweep\n{'=' * 78}")
            print(f"{'B':>4}  " + "".join(f"{n:>10}" for n in names)
                  + f"{'ratio':>9}   active_frac")
            print("-" * 78)
            for b, tps, ratio in batch_sweep(gpu, quant, c):
                af = 1.0 - (1.0 - 0.46) ** b
                print(f"{b:>4}  " + "".join(
                    f"{tps[n]:>10.2f}" if tps[n] is not None else f"{'--':>10}"
                    for n in names)
                    + f"{ratio:>8.2f}x{af:>14.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sweep", action="store_true",
                    help="sweep B to locate the sparsity-collapse crossover")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                    help="warmup ablation only")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--prefill", action="store_true",
                    help="report prefill throughput instead of decode")
    ap.add_argument("--kv-tier", action="store_true",
                    help="sweep which tier holds the KV cache")
    ap.add_argument("--sharegpt", action="store_true",
                    help="replay real ShareGPT prompt/response lengths")
    ap.add_argument("--detail", action="store_true",
                    help="per-policy warmup / prefill / decode / wall-clock table")
    a = ap.parse_args()

    if a.sharegpt:
        if not a.gpu_only:
            sharegpt(0)
        if not a.cpu_only:
            sharegpt(GPU_GB)
        return
    if a.detail:
        if not a.gpu_only:
            detail(0)
        if not a.cpu_only:
            detail(GPU_GB)
        return
    if a.kv_tier:
        if not a.gpu_only:
            kv_tier_sweep(0)
        if not a.cpu_only:
            kv_tier_sweep(GPU_GB)
        return
    if a.batch_sweep:
        if not a.gpu_only:
            batch_table(0)
        if not a.cpu_only:
            batch_table(GPU_GB)
        return
    if a.ablation:
        warmup_ablation(0)
        return
    metric = "prefill" if a.prefill else "decode"
    if not a.gpu_only:
        comparison(0, a.diag, metric)
    if not a.cpu_only:
        comparison(GPU_GB, a.diag, metric)


if __name__ == "__main__":
    main()
