"""
plot_overlap.py -- one decode step in time, measured both ways.

The paper's claim is temporal: a NAND read over a filled device enters the
critical path, and the same read over a searched reserve is issued ahead and
completes behind fast-tier work. This figure shows that on the real schedule,
not a schematic: the unit list is dumped from an actual decode step of the
shipped scheduler and of the same scheduler with the search pinned to a filled
device, and the schedule is reconstructed by a replica of the timing engine
whose final time is asserted equal to pipelined_time_s on exactly these units.

    python plot_overlap.py            # run both schedulers, then plot
    python plot_overlap.py --cached   # replot from overlap_data.json

Output: figures/fig_overlap.pdf, one column wide.  Cell: INT8 16H+48C, CPU.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "overlap_data.json")
OUT = os.path.join(REPO, "figures", "fig_overlap.pdf")
sys.path.insert(0, REPO)

from pipeline import pipelined_time_s, SHARED_LINK           # noqa: E402
from tiers import CXL_HOST_LINK_CEILING                     # noqa: E402

CHUNK = 256 * 1024

# The dump is injected immediately before the decode loop's engine call, so the
# schedule plotted is the one the reported number is made of.
ANCHOR = """            lat = pipelined_time_s(units, PREFETCH_QUEUE_DEPTH, TIER_BW,
                                   TIER_LAT,
                                   inflight_budget=max(0.0, _prefetch_reserve
                                                       - pf_stats["bytes_prefetched"]))"""
DUMP = """            if token_step == 0 and not __import__("os").path.exists("overlap_dump.json"):
                __import__("json").dump(
                    {"units": [[bt, c] for bt, c in units],
                     "depth": PREFETCH_QUEUE_DEPTH,
                     "budget": max(0.0, _prefetch_reserve
                                   - pf_stats["bytes_prefetched"]),
                     "bw": TIER_BW, "lat": TIER_LAT},
                    open("overlap_dump.json", "w"))
""" + ANCHOR

# The filled variant is the paper's own ablation pin, identical to
# plot_reserve_curve.py: reserve nothing, KV on the device, full capacity.
PINS = [
    ("""        _fr = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20,
               0.25, 0.30, 0.35, 0.40, 0.45, 0.50)""",
     "        _fr = (0.0,)"),
    ('_kv_opts = ["cxl", "host"] + (["gpu"] if gpu_hbm_capacity_bytes else [])',
     '_kv_opts = ["cxl"]'),
    ("""        _dev_caps = sorted({cxl_dev_dram_capacity_bytes}
                           | {c * GiB for c in (32, 48)
                              if c * GiB < cxl_dev_dram_capacity_bytes})""",
     "        _dev_caps = [cxl_dev_dram_capacity_bytes]"),
]


def measure(pin):
    import run_paper_tables as R
    with tempfile.TemporaryDirectory() as td:
        for f in R.DEPS + [s for _, s in R.SIMS]:
            shutil.copy(os.path.join(R.REPO, f), td)
        R.patch(td, "int8", 16, 48, 0)                 # the 1.49x cell, CPU-only
        p = os.path.join(td, "semduplex_scheduler.py")
        s = open(p).read()
        assert ANCHOR in s, "dump anchor drifted"
        s = s.replace(ANCHOR, DUMP, 1)
        if pin:
            for a, b in PINS:
                assert a in s, f"pin anchor drifted: {a[:50]!r}"
                s = s.replace(a, b, 1)
        open(p, "w").write(s)
        subprocess.run([sys.executable, "semduplex_scheduler.py"], cwd=td,
                       capture_output=True, text=True, timeout=900)
        return json.load(open(os.path.join(td, "overlap_dump.json")))


def schedule(d):
    """Replica of pipelined_time_s that records events; asserted against it."""
    units = [(bt, c) for bt, c in d["units"]]
    depth, budget, bw, lat = d["depth"], d["budget"], d["bw"], d["lat"]
    tier_free, link_free, prev_end = {}, 0.0, 0.0
    comp_done = [0.0] * len(units)
    xfer, comp = [], []                     # (i, tier, start, end), (i, start, end)
    for i, (by, c) in enumerate(units):
        j = i - depth - 1
        issue = comp_done[j] if j >= 0 else 0.0
        if budget is not None:
            m, acc = i - 1, sum(by.values())
            while m > j and m >= 0:
                nxt = sum(units[m][0].values())
                if acc + nxt > budget:
                    break
                acc += nxt
                m -= 1
            issue = max(issue, comp_done[m] if m >= 0 else 0.0)
        ready = issue
        for tier, nb in by.items():
            if nb <= 0 or not bw.get(tier):
                continue
            dur = nb / bw[tier]
            if lat.get(tier, 0.0):
                dur += math.ceil(nb / CHUNK) * lat[tier]
            start = max(issue, tier_free.get(tier, 0.0))
            if tier in SHARED_LINK:
                start = max(start, link_free)
                link_free = start + nb / CXL_HOST_LINK_CEILING
            end = start + dur
            tier_free[tier] = end
            ready = max(ready, end)
            xfer.append((i, tier, start, end))
        s0 = max(ready, prev_end)
        prev_end = s0 + c
        comp_done[i] = prev_end
        comp.append((i, s0, prev_end))
    truth = pipelined_time_s(units, depth, bw, lat, inflight_budget=budget)
    assert abs(prev_end - truth) <= 1e-12 * max(1.0, truth), \
        f"trace diverged from the engine: {prev_end} vs {truth}"
    return xfer, comp, prev_end


ROWS = ["Compute", "Host DRAM", "CXL DRAM", "NAND"]
KEY = {"Host DRAM": "Host DRAM", "CXL Device DRAM": "CXL DRAM",
       "CXL Device NAND": "NAND"}
COL = {"Compute": "#3D4C5C", "Host DRAM": "#8FB4D9",
       "CXL DRAM": "#4A7BA7", "NAND": "#B4503C"}


def panel(ax, d, i0, i1, label, step_s):
    xfer, comp, _ = schedule(d)
    win = lambda ev: [e for e in ev if i0 <= e[0] <= i1]
    t0 = min([s for _, s, _ in win(comp)] +
             [s for _, _, s, _ in win(xfer)])
    for i, s, e in win(comp):
        ax.barh(3, (e - s) * 1e3, left=(s - t0) * 1e3, height=0.62,
                color=COL["Compute"], edgecolor="white", linewidth=0.3)
    for i, tier, s, e in win(xfer):
        r = KEY.get(tier)
        if r is None:
            continue
        y = {"Host DRAM": 2, "CXL DRAM": 1, "NAND": 0}[r]
        ax.barh(y, (e - s) * 1e3, left=(s - t0) * 1e3, height=0.62,
                color=COL[r], edgecolor="white", linewidth=0.3)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(ROWS, fontsize=6.6)
    ax.set_ylim(-0.55, 3.55)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.set_title(f"{label} (full step {step_s:.2f} s)",
                 fontsize=7.4, pad=3, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    return max([e for _, _, e in win(comp)]) - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    a = ap.parse_args()
    if a.cached and os.path.exists(DATA):
        data = json.load(open(DATA))
    else:
        data = {"filled": measure(pin=True), "reserved": measure(pin=False)}
        json.dump(data, open(DATA, "w"))

    # Window: centred on NAND-bearing units of the SHIPPED schedule, and the
    # same unit indices are shown for the filled one, so the panels compare the
    # identical sub-layers.
    xr, cr, t_res = schedule(data["reserved"])
    _, _, t_fil = schedule(data["filled"])
    nand_units = sorted({i for i, tier, *_ in xr if tier == "CXL Device NAND"})
    mid = nand_units[min(2, len(nand_units) - 1)]
    i0, i1 = max(0, mid - 4), mid + 7

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "DejaVu Serif"],
        "font.size": 7.5, "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.35), sharex=True)
    end_a = panel(axes[0], data["filled"], i0, i1,
                  "(a) tiers filled: NAND on the critical path", t_fil)
    end_b = panel(axes[1], data["reserved"], i0, i1,
                  "(b) searched reserve: same reads issued ahead", t_res)
    lim = max(end_a, end_b) * 1e3 * 1.03
    for ax in axes:
        ax.set_xlim(0, lim)
    # the identical twelve units, on one clock: (b) finishes earlier, and the
    # emptiness to the right of its last bar IS the saving.
    axes[1].axvline(end_b * 1e3, color="#3D4C5C", linewidth=0.7,
                    linestyle=(0, (4, 3)))
    axes[1].annotate(f"same units,\n{(1 - end_b / end_a) * 100:.0f}% sooner",
                     xy=(end_b * 1e3, 2.6), xytext=(end_b * 1e3 + lim * 0.04, 2.35),
                     fontsize=6.4, color="#3D4C5C",
                     arrowprops=dict(arrowstyle="->", color="#3D4C5C",
                                     linewidth=0.7))
    axes[1].set_xlabel("time within the step (ms)", fontsize=7.2)
    fig.tight_layout(pad=0.4, h_pad=1.1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.015)
    print(f"wrote {OUT}   units {i0}..{i1}   step {t_fil:.2f}s vs {t_res:.2f}s")


if __name__ == "__main__":
    main()
