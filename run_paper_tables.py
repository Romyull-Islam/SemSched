"""
Regenerate the Section V tables from the current working tree. Every number
reported in the paper comes from this script.

    python run_paper_tables.py                # both platforms, both quants
    python run_paper_tables.py --cpu-only
    python run_paper_tables.py --diag         # print simulator output for failed cells

Each cell runs in a temporary directory holding a private copy of the
simulator, with sim_cfg.py and model_cfg.py (and tiersched.py for
some sweeps) rewritten for that cell. The repository itself is never modified.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
DEPS = ["sim_cfg.py", "tiers.py", "model_cfg.py", "cxl_link.py", "pipeline.py",
        "bound.py"]

SIMS = [
    ("FlexGen",  "flexgen_baseline.py"),
    ("LIA",      "lia_baseline.py"),
    ("AimPod",   "cxlaimpod_baseline.py"),
    ("LLMFlash", "llmflash_baseline.py"),
    ("ITME",     "itme_baseline.py"),
    ("TierSched", "tiersched.py"),
]

# RTX 5090: 32 GB GDDR7, of which 28 GB is usable for weights after activations
# and the runtime's own footprint. 0 = the CPU-only platform, which is what both
# CMM-H papers characterise.
GPU_GB = 28

# An accelerator adds both memory capacity and a compute engine, so both are set
# together. Setting gpu_hbm_capacity_bytes alone would time compute-bound layers
# on the EPYC's 14.4 TFLOPS.
#   EPYC 9454 x2 : 96 cores, AVX-512 (64 FLOP/cyc), 2.75 GHz, 0.85 ->  14.4 TF
#   RTX 5090     : 170 SMs, 512 FLOP/cyc/SM, 2.41 GHz, 0.70        -> 146.9 TF
ENGINES = {
    0:      (96,  64.0,  2.75e9, 0.85),
    GPU_GB: (170, 512.0, 2.41e9, 0.70),
}
GRID = [16]                       # host DRAM
CXLS = [32, 48, 64]               # CXL device DRAM
QUANTS = ["fp16", "int8"]
BATCH, DECODE = 128, 256


def patch(td, quant, h, c, gpu, warmup=True, batch=BATCH, kv_tier=None,
          model="Qwen2_5_72BCfg"):
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

    # LLM-in-a-Flash keeps its own decode count rather than reading sim_cfg.
    p = os.path.join(td, "llmflash_baseline.py")
    s = open(p).read()
    s, n = re.subn(r"^NUM_DECODE_TOKENS(\s*)= \d+", f"NUM_DECODE_TOKENS\\1= {DECODE}", s, 1, re.M)
    assert n == 1, "llmflash decode anchor drifted"
    open(p, "w").write(s)

    p = os.path.join(td, "model_cfg.py")
    s = open(p).read()
    s = re.sub(r'^QUANT\s*=\s*"\w+"', f'QUANT = "{quant}"', s, 1, re.M)
    s = re.sub(r"^DEFAULT_MODEL_CFG\s*=\s*\w+",
               f"DEFAULT_MODEL_CFG = {model}", s, 1, re.M)
    open(p, "w").write(s)

    if not warmup or kv_tier is not None:
        p = os.path.join(td, "tiersched.py")
        s = open(p).read()
        if not warmup:
            s = re.sub(r"^ENABLE_PREFILL_WARMUP\s*=\s*True",
                       "ENABLE_PREFILL_WARMUP = False", s, 1, re.M)
        if kv_tier is not None:
            s = re.sub(r'^KV_TIER\s*=\s*"\w+"', f'KV_TIER = "{kv_tier}"', s, 1, re.M)
            # The joint search iterates _kv_opts and overwrites _best_kv, so
            # patching KV_TIER alone has no effect. Pin the search list instead,
            # as plot_reserve_curve.py does for the reserve grid.
            anchor = ('_kv_opts = ["cxl", "host", "stripe"] + (["gpu"] '
                      'if gpu_hbm_capacity_bytes else [])')
            assert anchor in s, "kv-tier pin anchor drifted; --kv-tier would no-op"
            s = s.replace(anchor, f'_kv_opts = ["{kv_tier}"]', 1)
        open(p, "w").write(s)


def run(sim, quant, h, c, gpu, warmup=True, diag=False, batch=BATCH,
        kv_tier=None, model="Qwen2_5_72BCfg"):
    """One simulator, one cell. Returns (decode_tps, prefill_tps, stdout)."""
    with tempfile.TemporaryDirectory() as td:
        for f in DEPS + [s for _, s in SIMS]:
            shutil.copy(os.path.join(REPO, f), td)
        patch(td, quant, h, c, gpu, warmup, batch, kv_tier, model)
        r = subprocess.run([sys.executable, sim], cwd=td,
                           capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        d = re.search(r"Decode throughput:\s*([\d.]+)", out)
        p = re.search(r"Prefill throughput:\s*([\d.]+)", out)
        if d is None and diag:
            print(out[-1500:], file=sys.stderr)
        return (float(d.group(1)) if d else None,
                float(p.group(1)) if p else None, out)


def bound(quant, h, c, gpu, batch=BATCH, model="Qwen2_5_72BCfg", prefill=None,
          env=None):
    """Upper bound on any policy's decode tokens/s in one cell (bound.py)."""
    with tempfile.TemporaryDirectory() as td:
        for f in DEPS + [s for _, s in SIMS]:
            shutil.copy(os.path.join(REPO, f), td)
        patch(td, quant, h, c, gpu, batch=batch, model=model)
        if prefill:
            _patch_lengths(td, prefill, DECODE)
        r = subprocess.run([sys.executable, "bound.py"], cwd=td, capture_output=True,
                           text=True, timeout=300, env={**os.environ, **(env or {})})
        m = re.search(r"Decode bound:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None


# ShareGPT replay: real prompt/response lengths instead of a fixed 512/16. Only
# the length distribution comes from the corpus; per-layer compute, transfer
# and KV costs still come from the model geometry.
def _patch_lengths(td, prefill, decode):
    for f, pat, rep in (
            ("tiersched.py", r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}"),
            ("flexgen_baseline.py",    r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}"),
            ("lia_baseline.py",        r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}"),
            ("cxlaimpod_baseline.py",  r"^PREFILL_TOKENS = \d+",      f"PREFILL_TOKENS = {prefill}"),
            ("llmflash_baseline.py",   r"^NUM_PREFILL_TOKENS = \d+",  f"NUM_PREFILL_TOKENS = {prefill}"),
            ("llmflash_baseline.py",   r"^NUM_DECODE_TOKENS(\s*)= \d+", f"NUM_DECODE_TOKENS\\1= {decode}"),
            ("itme_baseline.py",       r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}"),
            ("dynamic_baseline.py",    r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}"),
            ("bound.py",               r"^PREFILL_TOKENS(\s*)= \d+", f"PREFILL_TOKENS\\1= {prefill}")):
        q = os.path.join(td, f)
        if not os.path.exists(q):
            continue
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
        if v.get("TierSched") is None:
            continue
        base = max((x for k, x in v.items() if k != "TierSched" and x), default=None)
        if not base:
            continue
        ratios.append(v["TierSched"] / base)
        wins += v["TierSched"] > base
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
          + f"{'ratio':>9}" + (f"{'bound':>9}{'of bound':>9}" if metric == "decode" else ""))
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
                ours = tps["TierSched"]
                best = max(v for k, v in tps.items()
                           if k != "TierSched" and v is not None)
                total += 1
                wins += ours > best
                ub = bound(quant, h, c, gpu) if metric == "decode" else None
                print(f"{quant.upper():<6}{f'{h}H+{c}C':<11}"
                      + "".join(f"{tps[n]:>10.2f}" if tps[n] is not None
                                else f"{'--':>10}" for n in names)
                      + f"{ours / best:>8.3f}x"
                      + (f"{ub:>9.2f}{ours / ub:>8.1%}" if ub else ""))
    print("-" * 78)
    print(f"TierSched leads {wins}/{total}")


def warmup_ablation(gpu=0):
    """TierSched with Phase-1 warmup staging on and off, on the reported cells."""
    print(f"\n{'=' * 78}\nPhase-1 warmup ablation"
          f"{'  (+RTX 5090)' if gpu else '  (CPU-only)'}\n{'=' * 78}")
    print(f"{'Quant':<6}{'Memory':<11}{'best base':>11}{'warmup ON':>11}"
          f"{'warmup OFF':>12}{'contributes':>13}{'ratio ON':>10}{'ratio OFF':>11}")
    print("-" * 78)
    for quant in QUANTS:
        for c in CXLS:
            for h in GRID:
                base = max(run(s, quant, h, c, gpu)[0] or 0
                           for n, s in SIMS if n != "TierSched")
                on, _, _ = run("tiersched.py", quant, h, c, gpu, True)
                off, _, _ = run("tiersched.py", quant, h, c, gpu, False)
                print(f"{quant.upper():<6}{f'{h}H+{c}C':<11}{base:>11.2f}"
                      f"{on:>11.2f}{off:>12.2f}{(on - off) / off * 100:>12.1f}%"
                      f"{on / base:>9.2f}x{off / base:>10.2f}x")


MODELS = [
    ("Mistral7BCfg",     "Mistral 7B"),
    ("Llama13BCfg",      "Llama 13B"),
    ("Qwen3_20BCfg",     "Qwen3 22B"),
    ("Qwen2_5_72BCfg",   "Qwen2.5 72B"),
    ("Llama3_1_405BCfg", "Llama 405B"),
]


def model_sweep(gpu):
    """Model scale, 7B to 405B, at the documented 16H+48C module.

    The gain should be bounded by the non-dominant share of a step: parity when
    the model fits in fast memory, largest when a second tier holds a large
    share, and narrowing again when NAND dominates.
    """
    tag = f"+RTX 5090 ({GPU_GB} GB)" if gpu else "CPU-only"
    names = [n for n, _ in SIMS]
    for quant in QUANTS:
        print(f"\n{'=' * 78}\nModel scale, {quant.upper()} 16H+48C, {tag}, "
              f"B={BATCH}\n{'=' * 78}")
        print(f"{'Model':<13}" + "".join(f"{n:>10}" for n in names) + f"{'ratio':>9}")
        print("-" * 78)
        for cfg, label in MODELS:
            with cf.ThreadPoolExecutor(max_workers=len(SIMS)) as ex:
                fut = {ex.submit(run, sim, quant, 16, 48, gpu, model=cfg): n
                       for n, sim in SIMS}
                tps = {}
                for f in cf.as_completed(fut):
                    tps[fut[f]] = f.result()[0]
            ours = tps.get("TierSched")
            base = max((v for k, v in tps.items() if k != "TierSched" and v), default=None)
            print(f"{label:<13}" + "".join(
                f"{tps[n]:>10.2f}" if tps.get(n) else f"{'--':>10}" for n in names)
                + (f"{ours/base:>8.2f}x" if ours and base else f"{'--':>9}"))


def kv_tier_sweep(gpu):
    """Decode throughput with the KV cache pinned to each tier.

    Each tier is charged for the capacity KV reserves before placement and for
    every KV access at its bandwidth. KV in HBM is read 66x faster than from the
    device, but it evicts an equal mass of weights to a slower tier, and those
    are read every step too.
    """
    tag = f"+RTX 5090 ({GPU_GB} GB)" if gpu else "CPU-only"
    tiers = ["cxl", "host", "stripe"] + (["gpu"] if gpu else [])
    print(f"\n{'=' * 78}\nKV cache tier — TierSched, {tag}, B={BATCH}\n{'=' * 78}")
    print(f"{'Quant':<6}{'Memory':<11}" + "".join(f"{'KV in ' + t:>12}" for t in tiers)
          + f"{'best':>8}")
    print("-" * 78)
    for quant in QUANTS:
        for c in CXLS:
            for h in GRID:
                got = {t: run("tiersched.py", quant, h, c, gpu,
                              kv_tier=t)[0] for t in tiers}
                best = max(got, key=lambda t: got[t] or 0)
                print(f"{quant.upper():<6}{f'{h}H+{c}C':<11}"
                      + "".join(f"{got[t]:>12.2f}" if got[t] is not None
                                else f"{'--':>12}" for t in tiers)
                      + f"{best:>8}")


BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]


def batch_sweep(gpu, quant, c, h=16):
    """Locate the sparsity-collapse crossover.

    Under LLM-in-a-Flash's model the active fraction of an MLP sub-layer is
    1 - (1 - p)^B with p = 0.46, which reaches 1.000 by B = 16. Beyond that every
    policy reads every byte once per step, the step time is pinned to the
    byte-accounting floor sum(bytes(tier) / bw(tier)), and all the policies
    converge. Returns (B, {policy: tps}, TierSched / best baseline) per batch.
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
        ours = tps["TierSched"]
        best = max(v for k, v in tps.items() if k != "TierSched" and v is not None)
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


# Equal footing. Each baseline gets lookahead K and
# a fixed reserve of a fraction of host and device DRAM, taken off the capacity
# it places into and added to its staging budget. The rewrites below are the
# only changes to each baseline: its depth constant or call, and its budget.
EQ_DEPTHS = [1, 4, 8, 16, 32]
# TierSched's reserve grid (tiersched.py): a share of host plus device DRAM,
# split host first, device first or in proportion to their sizes.
EQ_SHARES = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10,
             0.13, 0.16, 0.20, 0.30, 0.40, 0.50]
# Transfer blocks per unit the tuned baselines may use, as TierSched does.
EQ_BLOCKS = [1, 8]


def eq_reserves(h, c):
    """(host, device) reserve fractions of TierSched's grid for h GB of host
    and c GB of device DRAM."""
    out = set()
    for f in EQ_SHARES:
        r = f * (h + c)
        out.add((f, f))
        hr = min(r, h)
        out.add((hr / h, min(r - hr, c) / c))
        dr = min(r, c)
        out.add((min(r - dr, h) / h, dr / c))
    return sorted(out)


EQ_PATCH = {
    "flexgen_baseline.py": [(r"^PREFETCH_DEPTH = 1$", "PREFETCH_DEPTH = {k}"),
                            ("inflight_budget=max(0.0, _FLEXGEN_STAGING)",
                             "inflight_budget=max(0.0, _FLEXGEN_STAGING) + EQ_EXTRA")],
    "lia_baseline.py": [(r"^PREFETCH_DEPTH = 1$", "PREFETCH_DEPTH = {k}"),
                        ("inflight_budget=max(0.0, _LIA_STAGING)",
                         "inflight_budget=max(0.0, _LIA_STAGING) + EQ_EXTRA")],
    "cxlaimpod_baseline.py": [("elapsed = pipelined_time_s(\n        units, PREFETCH_WINDOW,",
                               "elapsed = pipelined_time_s(\n        units, {k},"),
                              ("inflight_budget=max(0.0, gpu_free + cxl_free + host_free))",
                               "inflight_budget=max(0.0, gpu_free + cxl_free + host_free) + EQ_EXTRA)")],
    "llmflash_baseline.py": [("t = pipelined_time_s(_units, WINDOW_SIZE_K,",
                              "t = pipelined_time_s(_units, {k},"),
                             ("inflight_budget=_window) + step_kv",
                              "inflight_budget=_window + EQ_EXTRA) + step_kv")],
    "itme_baseline.py": [(r"^PREFETCH_DEPTH = 2 \* BUFFER_LAYERS - 1$", "PREFETCH_DEPTH = {k}"),
                         ("inflight_budget=_ITME_STAGING)",
                          "inflight_budget=_ITME_STAGING + EQ_EXTRA)")],
}
# 4096 sub-layers ahead is unbounded lookahead: Belady's rule with no window
# is Cao et al.'s Aggressive integrated prefetching and caching.
DYN_DEPTHS = [1, 2, 4, 8, 16, 32, 4096]


def _run_variant(sim, quant, h, c, gpu, rewrites=(), env=None):
    """One cell with `rewrites` applied to the simulator's private copy."""
    with tempfile.TemporaryDirectory() as td:
        for f in DEPS + [s for _, s in SIMS] + ["dynamic_baseline.py"]:
            shutil.copy(os.path.join(REPO, f), td)
        patch(td, quant, h, c, gpu)
        p = os.path.join(td, sim)
        s = open(p).read()
        for old, new in rewrites:
            if old.startswith("^"):
                s, n = re.subn(old, new, s, 1, re.M)
            else:
                n = s.count(old)
                s = s.replace(old, new, 1)
            assert n == 1, (sim, old)
        open(p, "w").write(s)
        r = subprocess.run([sys.executable, sim], cwd=td, capture_output=True,
                           text=True, timeout=900, env={**os.environ, **(env or {})})
        m = re.search(r"Decode throughput:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None


def _eq_job(job):
    kind, sim, quant, c, gpu, k, x = job
    h = 16
    if kind == "base":
        fh, fd, blocks = x
        rw = [(o, n.format(k=k)) for o, n in EQ_PATCH[sim]]
        return job, _run_variant_scaled(sim, quant, h, c, gpu, (fh, fd), rw,
                                        env={"PIPELINE_BLOCKS": str(blocks)})
    if kind == "dyn":
        kv, blocks = x
        return job, _run_variant(sim, quant, h, c, gpu,
                                 env={"DYN_PREFETCH_DEPTH": str(k), "DYN_KV_TIER": kv,
                                      "PIPELINE_BLOCKS": str(blocks)})
    return job, _run_variant(sim, quant, h, c, gpu,
                             rewrites=[(r"^PREFETCH_QUEUE_DEPTH  = 32$",
                                        f"PREFETCH_QUEUE_DEPTH  = {k}")])


def _run_variant_scaled(sim, quant, h, c, gpu, frac, rewrites, env=None):
    """A baseline with a fixed reserve: it places into (1 - frac) of host and
    device DRAM, and the reserved bytes are added to its staging budget. frac is
    one fraction for both tiers, or a (host, device) pair as TierSched searches."""
    if isinstance(frac, tuple):
        fh, fd = frac
        extra = fh * h + fd * c
    else:
        fh = fd = frac
        extra = frac * (h + c)
    with tempfile.TemporaryDirectory() as td:
        for f in DEPS + [s for _, s in SIMS]:
            shutil.copy(os.path.join(REPO, f), td)
        patch(td, quant, h * (1 - fh), c * (1 - fd), gpu)
        p = os.path.join(td, sim)
        s = f"EQ_EXTRA = {extra * 1024 ** 3!r}\n" + open(p).read()
        for old, new in rewrites:
            if old.startswith("^"):
                s, n = re.subn(old, new, s, 1, re.M)
            else:
                n = s.count(old)
                s = s.replace(old, new, 1)
            assert n == 1, (sim, old)
        open(p, "w").write(s)
        r = subprocess.run([sys.executable, sim], cwd=td, capture_output=True,
                           text=True, timeout=900, env={**os.environ, **(env or {})})
        m = re.search(r"Decode throughput:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None


def equal_footing(gpu):
    """Every policy at its best lookahead: baselines also at their best fixed
    reserve, with host and device fractions chosen separately from the same 15
    values TierSched searches, and the dynamic-replacement baseline at its best
    KV tier. Returns the per-configuration results."""
    jobs = []
    for quant in QUANTS:
        for c in CXLS:
            for sim in EQ_PATCH:
                for k in EQ_DEPTHS:
                    for fh, fd in eq_reserves(16, c):
                        for b in EQ_BLOCKS:
                            jobs.append(("base", sim, quant, c, gpu, k, (fh, fd, b)))
            for k in DYN_DEPTHS:
                for kv in ("cxl", "host"):
                    for b in EQ_BLOCKS:
                        jobs.append(("dyn", "dynamic_baseline.py", quant, c, gpu, k, (kv, b)))
            for k in EQ_DEPTHS:
                jobs.append(("sem", "tiersched.py", quant, c, gpu, k, None))
    res = {}
    with cf.ProcessPoolExecutor() as ex:
        for job, tps in ex.map(_eq_job, jobs):
            res[job] = tps or 0.0

    def best(pred):
        cand = [(v, j) for j, v in res.items() if pred(j)]
        return max(cand) if cand else (0.0, None)

    names = dict((s, n) for n, s in SIMS)
    print(f"\n{'=' * 100}\nEqual footing: every policy at its best setting"
          f"{'  (+RTX 5090)' if gpu else '  (CPU-only)'}\n{'=' * 100}")
    print(f"{'Quant':<6}{'Memory':<10}{'best static baseline':>34}{'dynamic (Belady)':>24}"
          f"{'TierSched':>18}{'vs static':>11}{'vs dynamic':>11}")
    print("-" * 100)
    rows = {}
    for quant in QUANTS:
        for c in CXLS:
            cell = lambda j: j[2] == quant and j[3] == c
            bv, bj = best(lambda j: j[0] == "base" and cell(j))
            dv, dj = best(lambda j: j[0] == "dyn" and cell(j))
            sv, sj = best(lambda j: j[0] == "sem" and cell(j))
            bdesc = (f"{names[bj[1]]} K={bj[5]} r={bj[6][0]:.1%}/{bj[6][1]:.1%}"
                     f" b={bj[6][2]}")
            ddesc = f"K={dj[5]} KV={dj[6][0]} b={dj[6][1]}"
            print(f"{quant.upper():<6}{f'16H+{c}C':<10}{bv:>8.2f} ({bdesc:<26}){dv:>9.2f} ({ddesc:<11})"
                  f"{sv:>8.2f} (K={sj[5]:<2}){sv / bv:>10.2f}x{sv / dv:>10.2f}x")
            rows[f"{'GPU' if gpu else 'CPU'} {quant} 16H+{c}C"] = {
                "static": bv, "static_policy": names[bj[1]], "static_K": bj[5],
                "static_host_reserve": bj[6][0], "static_device_reserve": bj[6][1],
                "static_blocks": bj[6][2],
                "dynamic": dv, "dynamic_K": dj[5], "dynamic_kv": dj[6][0],
                "dynamic_blocks": dj[6][1],
                "tiersched": sv, "tiersched_K": sj[5]}
    return rows


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
    ap.add_argument("--models", action="store_true",
                    help="sweep model scale, 7B to 405B, at 16H+48C")
    ap.add_argument("--sharegpt", action="store_true",
                    help="replay real ShareGPT prompt/response lengths")
    ap.add_argument("--equal-footing", action="store_true",
                    help="every policy at its best lookahead and reserve, plus a "
                         "Belady dynamic-replacement baseline")
    ap.add_argument("--detail", action="store_true",
                    help="per-policy warmup / prefill / decode / wall-clock table")
    a = ap.parse_args()

    if a.equal_footing:
        out = os.path.join(REPO, "equal_footing.json")
        rows = json.load(open(out)) if os.path.exists(out) else {}
        if not a.gpu_only:
            rows.update(equal_footing(0))
        if not a.cpu_only:
            rows.update(equal_footing(GPU_GB))
        json.dump(rows, open(out, "w"), indent=1)
        print(f"wrote {out}")
        return
    if a.models:
        if not a.gpu_only:
            model_sweep(0)
        if not a.cpu_only:
            model_sweep(GPU_GB)
        return
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
