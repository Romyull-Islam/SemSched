"""
Cost of re-planning when the context length doubles or halves (Sec. VII): the
bytes that must newly arrive in host DRAM or the device cache, plus the KV-cache
move, at the CMM-H rates, plus the search itself, compared with the decode time
of one generation.

    python replan_cost.py
"""
import concurrent.futures as cf, os, re, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_paper_tables as R

def plan(q, c, g, p):
    with tempfile.TemporaryDirectory() as td:
        for f in R.DEPS + [s for _, s in R.SIMS]:
            shutil.copy(os.path.join(R.REPO, f), td)
        R.patch(td, q, 16, c, g); R._patch_lengths(td, p, R.DECODE)
        o = subprocess.run([sys.executable, "tiersched.py"], cwd=td,
                           capture_output=True, text=True).stdout
    m = re.search(r"Placement: (.*); KV (\w+) ([\d.]+) GB", o)
    tiers = {k.strip(): float(v) for v, k in
             [(x.split(" GB")[0].split()[-1], " ".join(x.split(" GB")[0].split()[:-1])) for x in m.group(1).split(", ")]}
    dec = float(re.search(r"Decode throughput:\s*([\d.]+)", o).group(1))
    st = float(re.search(r"Search time: ([\d.]+)", o).group(1))
    return tiers, m.group(2), float(m.group(3)), dec, st

CFG = [(q, c, g) for g in (0, R.GPU_GB) for q in R.QUANTS for c in R.CXLS]
LENS = [256, 512, 1024, 2048]
with cf.ProcessPoolExecutor(4) as ex:
    P = {k: f.result() for k, f in {(q, c, g, p): ex.submit(plan, q, c, g, p)
                                     for q, c, g in CFG for p in LENS}.items()}
worst = (0, None)
for q, c, g in CFG:
    for a, b in zip(LENS, LENS[1:]):
        for x, y in ((a, b), (b, a)):
            t0, kv0, kvb0, _, _ = P[(q, c, g, x)]
            t1, kv1, kvb1, dec, st = P[(q, c, g, y)]
            into_host = max(0, t1.get("Host DRAM", 0) - t0.get("Host DRAM", 0))
            into_dev = max(0, t1.get("CXL Device DRAM", 0) - t0.get("CXL Device DRAM", 0))
            kv_move = kvb1 if kv1 != kv0 else 0.0
            # host arrivals come from the device at its 27 GB/s; device-cache
            # arrivals come from NAND at 5 GB/s; a moved KV cache crosses the link
            s = into_host / 27 + into_dev / 5 + kv_move / 27 + st
            # decode time of one generation; throughput is tokens/s over the batch
            gen_s = R.DECODE * R.BATCH / dec
            frac = s / gen_s
            if frac > worst[0]:
                worst = (frac, (q, c, g, x, y, round(s, 1), round(gen_s)))
print("worst re-plan cost as a share of one generation:", round(worst[0] * 100, 2), "%", worst[1])
