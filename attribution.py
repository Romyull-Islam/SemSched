"""
Where TierSched's lead over the tuned baselines comes from (Sec. V-D). Starting
from TierSched's own placement with the tuned baselines' kind of reserve, add
its mechanisms one at a time and compare each step with the best tuned static
baseline of equal_footing.json:

  1. size-descending contiguous placement, one sub-layer per transfer, the
     reserve split evenly between host and device, KV cache on the device,
     full device size, no residue spreading
  2. + transfer blocks
  3. + striping
  4. + reserve split host or device first
  5. + the KV-tier and device-size search, residue spreading (the full scheduler)

    python run_paper_tables.py --equal-footing   # writes equal_footing.json
    python attribution.py                        # writes attribution.json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_paper_tables as R

CONTIG = [('        _modes = ("spread", "contig", "stripe")', '        _modes = ("contig",)')]
NO_SPREAD_STRIPE = [('        _modes = ("spread", "contig", "stripe")', '        _modes = ("contig", "stripe")')]
ONE_BLOCK = [("TRANSFER_BLOCKS = int(__import__(\"os\").environ.get(\"TIERSCHED_BLOCKS\", \"8\"))",
              "TRANSFER_BLOCKS = 1")]
KV_DEVICE = [('_kv_opts = ["cxl", "host", "stripe"] + (["gpu"] if gpu_hbm_capacity_bytes else [])',
              '_kv_opts = ["cxl"]')]
FULL_DEVICE = [("""        _dev_caps = sorted({cxl_dev_dram_capacity_bytes}
                           | {c * GiB for c in (32, 48)
                              if c * GiB < cxl_dev_dram_capacity_bytes})""",
                "        _dev_caps = [cxl_dev_dram_capacity_bytes]")]
EVEN = [('        _splits = ("host", "device", "even")', '        _splits = ("even",)')]

STEPS = [
    ("contiguous, one block, even split", CONTIG + ONE_BLOCK + EVEN + KV_DEVICE + FULL_DEVICE),
    ("+ transfer blocks", CONTIG + EVEN + KV_DEVICE + FULL_DEVICE),
    ("+ striping", NO_SPREAD_STRIPE + EVEN + KV_DEVICE + FULL_DEVICE),
    ("+ host or device first", NO_SPREAD_STRIPE + KV_DEVICE + FULL_DEVICE),
    ("+ KV tier, device size, spreading (full)", []),
]
CELLS = [(q, c, g) for g in (0, R.GPU_GB) for q in R.QUANTS for c in R.CXLS]


def run(quant, c, gpu, rewrites):
    with tempfile.TemporaryDirectory() as td:
        for f in R.DEPS + [s for _, s in R.SIMS]:
            shutil.copy(os.path.join(R.REPO, f), td)
        R.patch(td, quant, 16, c, gpu)
        p = os.path.join(td, "tiersched.py")
        s = open(p).read()
        for a, b in rewrites:
            assert s.count(a) == 1, f"patch anchor drifted: {a[:50]!r}"
            s = s.replace(a, b, 1)
        open(p, "w").write(s)
        r = subprocess.run([sys.executable, "tiersched.py"], cwd=td,
                           capture_output=True, text=True, timeout=900)
        m = re.search(r"Decode throughput:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None


def job(j):
    q, c, g, i = j
    return j, run(q, c, g, STEPS[i][1])


def main():
    eq = json.load(open(os.path.join(R.REPO, "equal_footing.json")))
    jobs = [(q, c, g, i) for q, c, g in CELLS for i in range(len(STEPS))]
    with cf.ProcessPoolExecutor(24) as ex:
        res = dict(ex.map(job, jobs))
    out = {}
    print(f"{'cell':<20}{'tuned':>8}" + "".join(f"{n[:22]:>24}" for n, _ in STEPS))
    for q, c, g in CELLS:
        key = f"{'GPU' if g else 'CPU'} {q} 16H+{c}C"
        base = eq[key]["static"]
        tps = [res[(q, c, g, i)] for i in range(len(STEPS))]
        out[key] = {"tuned_static": base, **{n: t for (n, _), t in zip(STEPS, tps)}}
        print(f"{key:<20}{base:>8.2f}" + "".join(f"{t:>14.2f} ({t / base:.3f}x)" for t in tps))
    print()
    for i, (n, _) in enumerate(STEPS):
        sp = [out[k][n] / out[k]["tuned_static"] for k in out]
        print(f"{n:<34} vs tuned: median {statistics.median(sp):.3f} min {min(sp):.3f} "
              f"max {max(sp):.3f}, leads {sum(s > 1.005 for s in sp)}/12")
    json.dump(out, open(os.path.join(R.REPO, "attribution.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
