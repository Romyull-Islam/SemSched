#!/usr/bin/env python3
"""
sweep.py — drive SimCXL across the same memory matrix as the main artifact.

The point of this file is that no tier size is hard-coded anywhere. The main
artifact's results are a *surface* over (host DRAM, CXL DRAM), not a point:
the advantage peaks where NAND residency divided by cache capacity is near
unity, so any validation that fixes one configuration cannot test the claim
that matters. This sweeps the same grid.

    ./sweep.py --hosts 8 16 32 64 --cxls 32 48 64 96
    ./sweep.py --hosts 16 --cxls 48 --gpu 28        # the headline point
    ./sweep.py --duplex                             # read/write concurrency test

Each cell writes to results/<host>H_<cxl>C[_<gpu>G]/ and the summary lands in
results/summary.csv.
"""
import argparse
import csv
import itertools
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
GEM5 = os.path.join(REPO, "SimCXL", "build", "X86", "gem5.opt")
CONFIG = os.path.join(REPO, "configs", "cmmh_hybrid.py")

# CMM-H prototype, Soltaniyeh et al. HotStorage 2025. These are the calibration
# targets: whatever we build inside gem5 has to reproduce them before any
# result from it means anything.
CMMH = {
    "dram_hit_bw_GBps":   27.0,
    "dram_hit_lat_ns":    505,
    "nand_miss_bw_GBps":   5.0,
    "nand_miss_lat_ns":  1547,
}
HOST_DDR5 = {"bw_GBps": 38.4, "lat_ns": 200}


def cell_name(h, c, g):
    return f"{h}H_{c}C" + (f"_{g}G" if g else "")


def run_cell(host_gb, cxl_gb, nand_gb, gpu_gb, mode, extra):
    """One gem5 invocation. Returns the outdir, or None if it failed."""
    out = os.path.join(REPO, "results", cell_name(host_gb, cxl_gb, gpu_gb))
    os.makedirs(out, exist_ok=True)
    cmd = [GEM5, f"--outdir={out}", CONFIG,
           f"--host-dram={host_gb}GB",
           f"--cxl-dram={cxl_gb}GB",
           f"--nand={nand_gb}GB",
           f"--mode={mode}"]
    if gpu_gb:
        cmd.append(f"--accel-mem={gpu_gb}GB")
    cmd += extra
    print("  " + " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED rc={r.returncode}\n{r.stderr[-800:]}", file=sys.stderr)
        return None
    return out


def read_stats(outdir):
    """Pull bandwidth and latency out of gem5's stats.txt."""
    path = os.path.join(outdir, "stats.txt")
    if not os.path.exists(path):
        return {}
    want = ("bwTotal", "avgMemAccLat", "readReqs", "writeReqs",
            "bwRead", "bwWrite", "cxl", "hits", "misses")
    got = {}
    with open(path) as f:
        for line in f:
            if line.startswith(("#", "-")) or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2 and any(w.lower() in parts[0].lower() for w in want):
                got[parts[0]] = parts[1]
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", type=int, nargs="+", default=[16],
                    help="host DRAM sizes in GB")
    ap.add_argument("--cxls", type=int, nargs="+", default=[48],
                    help="CXL device DRAM cache sizes in GB")
    ap.add_argument("--nand", type=int, default=1024,
                    help="CXL NAND backend in GB (prototype: 1 TB)")
    ap.add_argument("--gpu", type=int, default=0,
                    help="accelerator memory in GB usable for weights (0 = none)")
    ap.add_argument("--mode", default="stream",
                    choices=["stream", "duplex", "trace"],
                    help="stream = calibration; duplex = concurrent r/w; "
                         "trace = replay a decode trace")
    ap.add_argument("--duplex", action="store_true",
                    help="shorthand for --mode duplex")
    ap.add_argument("--out", default=None)
    args, extra = ap.parse_known_args()
    mode = "duplex" if args.duplex else args.mode

    if not os.path.exists(GEM5):
        sys.exit(f"gem5 not built yet: {GEM5}")
    if not os.path.exists(CONFIG):
        sys.exit(f"config not written yet: {CONFIG}")

    rows = []
    grid = list(itertools.product(args.hosts, args.cxls))
    print(f"{len(grid)} cells, mode={mode}, nand={args.nand}GB, gpu={args.gpu}GB")
    for h, c in grid:
        print(f"[{h}H+{c}C]", flush=True)
        out = run_cell(h, c, args.nand, args.gpu, mode, extra)
        if out is None:
            continue
        st = read_stats(out)
        # 145 GB is Qwen2.5 72B in FP16; overflow over cache capacity is the
        # predictor the main artifact's surface follows.
        overflow = max(0, 145 - h - c - args.gpu)
        rows.append({"host_GB": h, "cxl_GB": c, "nand_GB": args.nand,
                     "gpu_GB": args.gpu, "mode": mode,
                     "overflow_GB": overflow,
                     "overflow_over_cache": round(overflow / c, 3) if c else "",
                     **st})

    dest = args.out or os.path.join(REPO, "results", "summary.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {dest}")
    else:
        print("\nno cells completed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
