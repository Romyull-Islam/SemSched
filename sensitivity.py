"""
Robustness of the main result to the assumptions a CMM-H deployment could
break. Every sweep reruns the twelve configurations of Table III and reports
TierSched's speedup over the best of the four baselines in each one.

    python sensitivity.py hints      # device ignores a fraction of prefetch hints
    python sensitivity.py nandbw     # NAND bandwidth 2.5 / 5 / 7.5 GB/s
    python sensitivity.py nandlat    # NAND latency 0.5 / 1.547 / 5 / 20 / 50 / 100 us
    python sensitivity.py context    # prompts of 2K / 8K / 32K tokens at B=16
    python sensitivity.py mismatch   # plan for 512-token prompts, serve others
    python sensitivity.py passthrough  # NAND reads charged to device DRAM, or not
    python sensitivity.py all

Results go to sensitivity.json, keyed by sweep and setting.
"""
import concurrent.futures as cf
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

import run_paper_tables as R

OUT = os.path.join(R.REPO, "sensitivity.json")
CONFIGS = [(q, c, g) for g in (0, R.GPU_GB) for q in R.QUANTS for c in R.CXLS]


def cell(sim, quant, cxl, gpu, nand_bw=None, nand_lat=None, prefill=None,
         batch=R.BATCH, env=None):
    """Decode tokens/s of one simulator in one configuration, or None."""
    with tempfile.TemporaryDirectory() as td:
        for f in R.DEPS + [s for _, s in R.SIMS]:
            shutil.copy(os.path.join(R.REPO, f), td)
        R.patch(td, quant, 16, cxl, gpu, batch=batch)
        if prefill:
            R._patch_lengths(td, prefill, R.DECODE)
        if nand_bw or nand_lat:
            p = os.path.join(td, "tiers.py")
            s = open(p).read()
            s, n = re.subn(r'CXL_DEVICE_NAND = Tier\("CXL Device NAND \(CMM-H\)", [\d.e]+, [\d.e-]+\)',
                           f'CXL_DEVICE_NAND = Tier("CXL Device NAND (CMM-H)", '
                           f'{nand_bw or 5.0e9}, {nand_lat or 1.547e-6})', s, 1)
            assert n == 1, "NAND tier anchor drifted"
            open(p, "w").write(s)
        r = subprocess.run([sys.executable, sim], cwd=td, capture_output=True,
                           text=True, timeout=1800, env={**os.environ, **(env or {})})
        m = re.search(r"Decode throughput:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None


def speedups(points, workers=24, **kw):
    """{setting: [speedup per config]} for settings given as (label, kwargs)."""
    jobs = {}
    with cf.ProcessPoolExecutor(workers) as ex:
        for label, extra in points:
            for q, c, g in CONFIGS:
                for name, sim in R.SIMS:
                    a = {**kw, **extra}
                    sem = a.pop("sem", {})
                    if name == "TierSched":
                        a.update(sem)
                    jobs[ex.submit(cell, sim, q, c, g, **a)] = (label, q, c, g, name)
        res = {}
        for f in cf.as_completed(jobs):
            res[jobs[f]] = f.result()
    out = {}
    for label, _ in points:
        rows = []
        for q, c, g in CONFIGS:
            base = [res[(label, q, c, g, n)] for n, _ in R.SIMS if n != "TierSched"]
            sem = res[(label, q, c, g, "TierSched")]
            if sem is None or None in base:
                rows.append(None)
                continue
            rows.append({"config": f"{'GPU' if g else 'CPU'} {q} 16H+{c}C",
                         "tiersched": sem, "best_baseline": max(base),
                         "speedup": sem / max(base)})
        out[label] = rows
        sp = [r["speedup"] for r in rows if r]
        print(f"{label:>14}: speedup min {min(sp):.2f} median {statistics.median(sp):.2f} "
              f"max {max(sp):.2f}, wins {sum(s > 1.005 for s in sp)}/{len(sp)}, "
              f"failed {rows.count(None)}", flush=True)
    return out


SWEEPS = {
    "hints": lambda: speedups([(f"drop {p:.0%}", {"sem": {"env": {"TIERSCHED_HINT_DROP": str(p)}}})
                               for p in (0.0, 0.1, 0.25, 0.5, 1.0)]),
    "nandbw": lambda: speedups([(f"{b / 1e9:g} GB/s", {"nand_bw": b})
                                for b in (2.5e9, 5.0e9, 7.5e9)]),
    "nandlat": lambda: speedups([(f"{l * 1e6:g} us", {"nand_lat": l})
                                 for l in (0.5e-6, 1.547e-6, 5e-6, 20e-6, 50e-6, 100e-6)]),
    "context": lambda: speedups([(f"prompt {p}", {"prefill": p})
                                 for p in (2048, 8192, 32768)], batch=16),
    "passthrough": lambda: speedups([(f"pass-through {v}", {"env": {"NAND_PASS_THROUGH": v}})
                                     for v in ("1", "0")]),
    # Plan once for 512-token prompts; serve prompts of another length. The
    # baselines need no plan, so they run at the served length either way.
    "mismatch": lambda: {
        **speedups([(f"served {p}, planned 512",
                     {"prefill": p, "sem": {"env": {"TIERSCHED_PLAN_PREFILL": "512"}}})
                    for p in (128, 256, 1024, 2048)]),
        **speedups([(f"served {p}, replanned", {"prefill": p}) for p in (128, 256, 1024, 2048)]),
    },
}


def main():
    which = sys.argv[1:] or ["all"]
    names = list(SWEEPS) if which == ["all"] else which
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for n in names:
        print(f"== {n}", flush=True)
        data[n] = SWEEPS[n]()
        json.dump(data, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
