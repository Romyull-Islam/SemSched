"""
Mechanism ablation for Table V, under the pass-through NAND model. Each
variant patches a private copy of the scheduler; the patches are the ones
documented in REPRODUCE.md, and each asserts that its anchor still matches.

    NAND_PASS_THROUGH=1 python ablation_passthrough.py
"""
import concurrent.futures as cf
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

import run_paper_tables as R

PIN = [("""        _rf = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10,
               0.13, 0.16, 0.20, 0.30, 0.40, 0.50)""", "        _rf = (0.0,)"),
       ('_kv_opts = ["cxl", "host", "stripe"] + (["gpu"] if gpu_hbm_capacity_bytes else [])',
        '_kv_opts = ["cxl"]'),
       ("""        _dev_caps = sorted({cxl_dev_dram_capacity_bytes}
                           | {c * GiB for c in (32, 48)
                              if c * GiB < cxl_dev_dram_capacity_bytes})""",
        "        _dev_caps = [cxl_dev_dram_capacity_bytes]")]
NO_STRIPE = [('        _modes = ("spread", "contig", "stripe")', '        _modes = ("spread", "contig")')]
CONTIG = [('        _modes = ("spread", "contig", "stripe")', '        _modes = ("contig",)')]
EVEN = [('        _splits = ("host", "device", "even")', '        _splits = ("even",)')]
ONE_BLOCK = {"TIERSCHED_BLOCKS": "1"}
NO_SMALLER = [("if c * GiB < cxl_dev_dram_capacity_bytes})", "if False})")]
DEPTH0 = [("PREFETCH_QUEUE_DEPTH  = 32", "PREFETCH_QUEUE_DEPTH  = 0")]

VARIANTS = {
    "full": ([], {}),
    "one block per sub-layer": ([], ONE_BLOCK),
    "no striping": (NO_STRIPE, {}),
    "no striping, one block": (NO_STRIPE, ONE_BLOCK),
    "no striping or spreading": (CONTIG, {}),
    "reserve split evenly only": (EVEN, {}),
    "no smaller device": (NO_SMALLER, {}),
    "search disabled (filled)": (PIN, {}),
    "filled, K=0": (PIN + DEPTH0, {}),
    "filled, K=0, one block": (PIN + DEPTH0, ONE_BLOCK),
    "contiguous, semantic order": (CONTIG, {"TIERSCHED_PLACEMENT_ORDER": "semantic"}),
    "contiguous, random order": (CONTIG, {"TIERSCHED_PLACEMENT_ORDER": "random"}),
}


def run_sem(quant, c, gpu, rewrites, env):
    with tempfile.TemporaryDirectory() as td:
        for f in R.DEPS + [s for _, s in R.SIMS]:
            shutil.copy(os.path.join(R.REPO, f), td)
        R.patch(td, quant, 16, c, gpu)
        p = os.path.join(td, "tiersched.py")
        s = open(p).read()
        for a, b in rewrites:
            assert a in s, f"patch anchor drifted: {a[:50]!r}"
            s = s.replace(a, b, 1)
        open(p, "w").write(s)
        r = subprocess.run([sys.executable, "tiersched.py"], cwd=td,
                           capture_output=True, text=True, timeout=900,
                           env={**os.environ, **env})
        m = re.search(r"Decode throughput:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None


def job(j):
    kind, quant, c, gpu, name = j
    if kind == "base":
        return j, R.run(name, quant, 16, c, gpu)[0]
    rw, env = VARIANTS[name]
    return j, run_sem(quant, c, gpu, rw, env)


cells = [(q, c, g) for g in (0, R.GPU_GB) for q in R.QUANTS for c in R.CXLS]
jobs = [("base", q, c, g, s) for q, c, g in cells for n, s in R.SIMS if n != "TierSched"]
jobs += [("sem", q, c, g, v) for q, c, g in cells for v in VARIANTS]
with cf.ProcessPoolExecutor() as ex:
    res = dict(ex.map(job, jobs))

base = {cell: max(res[("base", *cell, s)] or 0 for n, s in R.SIMS if n != "TierSched")
        for cell in cells}
print(f"{'cell':<18}{'base':>8}" + "".join(f"{v[:14]:>16}" for v in VARIANTS))
for cell in cells:
    q, c, g = cell
    print(f"{('GPU ' if g else 'CPU ') + q + f' {c}C':<18}{base[cell]:>8.2f}"
          + "".join(f"{res[('sem', *cell, v)] or 0:>16.2f}" for v in VARIANTS))
print()
for v in VARIANTS:
    vs_full = [res[("sem", *cell, v)] / res[("sem", *cell, "full")] for cell in cells]
    vs_base = [res[("sem", *cell, v)] / base[cell] for cell in cells]
    wins = sum(x > 1.0 for x in vs_base)
    print(f"{v:<26} vs full: median {statistics.median(vs_full):.3f} min {min(vs_full):.3f} "
          f"max {max(vs_full):.3f} | beats best baseline in {wins}/12")
