"""
Is the reserve worth searching per configuration (Sec. V-G)? Pin TierSched's
reserve to the one (device, host) pair that does best over the twelve main
configurations, with the KV tier and device size still searched, then run that
pinned setting on the model, batch, prompt-length and NAND sweeps and compare
it with the full search and with the best of the baselines. The same
sweeps also run the minimum reserve of Sec. III-B: two of the largest
sub-layers in device DRAM, enough to keep one NAND read in flight.

    python fixed_reserve.py
"""
import concurrent.futures as cf
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_paper_tables as R

RF = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10,
      0.13, 0.16, 0.20, 0.30, 0.40, 0.50)
SPLITS = ("host", "device", "even")
MAIN = [(q, c, g) for g in (0, R.GPU_GB) for q in R.QUANTS for c in R.CXLS]
BASELINES = [s for n, s in R.SIMS if n != "TierSched"]


def pin(f, split):
    return [("for _f in _rf:", f"for _f in ({f},):"),
            ("for _s in _splits:", f"for _s in ({split!r},):")]


# Two blocks of the largest sub-layer in device DRAM and no host reserve:
# enough to keep one NAND read in flight.
MINIMUM = pin(0.0, "even") + [
    ("_dr, _hr = _split(_f, _s, _host_total, _dev_total)",
     "_dr, _hr = min(_dev_total, 2 * max(_L['bytes'] for _L in layers)"
     " / TRANSFER_BLOCKS), 0.0")]


def cell(sim, quant, c, gpu, model="Qwen2_5_72BCfg", batch=R.BATCH,
         prefill=None, nand_bw=None, nand_lat=None, rewrites=()):
    """Decode tokens/s of one simulator in one configuration, or None."""
    with tempfile.TemporaryDirectory() as td:
        for f in R.DEPS + [s for _, s in R.SIMS]:
            shutil.copy(os.path.join(R.REPO, f), td)
        R.patch(td, quant, 16, c, gpu, batch=batch, model=model)
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
        if rewrites:
            p = os.path.join(td, "tiersched.py")
            s = open(p).read()
            for a, b in rewrites:
                assert a in s, f"patch anchor drifted: {a!r}"
                s = s.replace(a, b, 1)
            open(p, "w").write(s)
        r = subprocess.run([sys.executable, sim], cwd=td, capture_output=True,
                           text=True, timeout=1800)
        m = re.search(r"Decode throughput:\s*([\d.]+)", r.stdout + r.stderr)
        return float(m.group(1)) if m else None


def _job(j):
    key, sim, kw, rewrites = j
    return key, cell(sim, rewrites=rewrites, **kw)


def run_all(jobs):
    with cf.ProcessPoolExecutor() as ex:
        return dict(ex.map(_job, jobs))


def sweeps():
    """(sweep, label, kwargs) for every cell of the scaling and robustness sweeps."""
    out = []
    for g in (0, R.GPU_GB):
        tag = "GPU" if g else "CPU"
        for q in R.QUANTS:
            for mdl, lab in R.MODELS:
                if mdl != "Qwen2_5_72BCfg":
                    out.append(("model scale", f"{tag} {q} {lab}",
                                dict(quant=q, c=48, gpu=g, model=mdl)))
            for c in R.CXLS:
                for b in (1, 2, 4, 8, 16, 32, 64):
                    out.append(("batch", f"{tag} {q} {c}C B={b}",
                                dict(quant=q, c=c, gpu=g, batch=b)))
                for p in (2048, 8192, 32768):
                    out.append(("prompt", f"{tag} {q} {c}C {p} tokens B=16",
                                dict(quant=q, c=c, gpu=g, batch=16, prefill=p)))
                for bw in (2.5e9, 7.5e9):
                    out.append(("NAND rate", f"{tag} {q} {c}C {bw / 1e9} GB/s",
                                dict(quant=q, c=c, gpu=g, nand_bw=bw)))
                for lat in (20e-6, 50e-6, 100e-6):
                    out.append(("NAND latency", f"{tag} {q} {c}C {lat * 1e6:.0f} us",
                                dict(quant=q, c=c, gpu=g, nand_lat=lat)))
    return out


def main():
    # Step 1: the best single reserve over the twelve main configurations,
    # ranked by its worst case and then its median against the search.
    kws = {cfg: dict(quant=cfg[0], c=cfg[1], gpu=cfg[2]) for cfg in MAIN}
    jobs = [((cfg, "search"), "tiersched.py", kws[cfg], ()) for cfg in MAIN]
    jobs += [((cfg, (f, sp)), "tiersched.py", kws[cfg], pin(f, sp))
             for cfg in MAIN for f in RF for sp in SPLITS]
    res = run_all(jobs)
    ranked = []
    for f in RF:
        for sp in SPLITS:
            r = [res[(cfg, (f, sp))] / res[(cfg, "search")] for cfg in MAIN]
            ranked.append((min(r), statistics.median(r), f, sp))
    ranked.sort(reverse=True)
    lo, med, df, hf = ranked[0]
    filled = [res[(cfg, (0.0, "even"))] / res[(cfg, "search")] for cfg in MAIN]
    print(f"Best single reserve over the twelve: {df:.2%} of host + device, split {hf}; "
          f"against the search: worst {lo:.3f}, median {med:.3f}")
    print(f"Filled (no reserve) against the search: worst {min(filled):.3f}, "
          f"median {statistics.median(filled):.3f}")

    # Step 2: that pinned setting on every other sweep.
    cells = sweeps()
    jobs = []
    for i, (_, _, kw) in enumerate(cells):
        jobs.append(((i, "search"), "tiersched.py", kw, ()))
        jobs.append(((i, "pinned"), "tiersched.py", kw, pin(df, hf)))
        jobs.append(((i, "minimum"), "tiersched.py", kw, MINIMUM))
        jobs += [((i, b), b, kw, ()) for b in BASELINES]
    res = run_all(jobs)
    by = {}
    for i, (sw, lab, _) in enumerate(cells):
        best = max(res[(i, b)] or 0 for b in BASELINES)
        by.setdefault(sw, []).append((lab, res[(i, "search")], res[(i, "pinned")],
                                      res[(i, "minimum")], best))
    every = [row for rows in by.values() for row in rows]
    print(f"\n{'sweep':<14}{'cells':>6}{'worst':>8}{'median':>8}"
          f"{'wins search':>13}{'wins pinned':>13}{'wins minimum':>14}"
          f"{'losses search':>15}{'losses pinned':>15}")
    for sw, rows in list(by.items()) + [("all", every)]:
        r = [p / s for _, s, p, _, _ in rows]
        print(f"{sw:<14}{len(rows):>6}{min(r):>8.3f}{statistics.median(r):>8.3f}"
              f"{sum(s / b > 1.005 for _, s, _, _, b in rows):>13}"
              f"{sum(p / b > 1.005 for _, _, p, _, b in rows):>13}"
              f"{sum(m / b > 1.005 for _, _, _, m, b in rows):>14}"
              f"{sum(s / b < 0.995 for _, s, _, _, b in rows):>15}"
              f"{sum(p / b < 0.995 for _, _, p, _, b in rows):>15}")
    print("\nLargest losses of the pinned reserve against the search:")
    for lab, s, p, _, b in sorted(every, key=lambda x: x[2] / x[1])[:6]:
        print(f"  {lab:<32} search {s:9.2f}  pinned {p:9.2f} ({p / s:.3f})  best baseline {b:9.2f}")


if __name__ == "__main__":
    main()
