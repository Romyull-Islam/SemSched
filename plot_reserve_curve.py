"""
plot_reserve_curve.py -- the motivation figure.

One plot carries the paper's claim: every published policy fills its fast tiers
completely, which is the leftmost point on every curve here, and every curve
peaks somewhere else. The peak is interior, it is worth 15-55%, and it moves
with quantization, capacity and platform -- 5%, 8%, 12%, 20% across four
configurations -- which is why it has to be searched rather than set.

Data comes from reserve_curve.json, produced by pinning the scheduler's reserve
grid to a single fraction and sweeping it. Regenerate with:

    python plot_reserve_curve.py --measure     # re-run the sweep, then plot
    python plot_reserve_curve.py               # plot from the saved JSON

Output: figures/fig_reserve_curve.pdf, sized for one IEEE column.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "reserve_curve.json")
OUT = os.path.join(REPO, "figures", "fig_reserve_curve.pdf")

FRACS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40]
CASES = [
    ("INT8, 16H+48C, CPU",   "int8", 48, 0),
    ("FP16, 16H+64C, CPU",   "fp16", 64, 0),
    ("INT8, 16H+32C, +GPU",  "int8", 32, 28),
    ("INT8, 16H+48C, +GPU",  "int8", 48, 28),
]

# Okabe-Ito, colour-blind safe, and distinguishable in greyscale by marker.
STYLE = [("#0072B2", "o", "-"), ("#D55E00", "s", "--"),
         ("#009E73", "^", "-."), ("#CC79A7", "D", ":")]


def measure():
    sys.path.insert(0, REPO)
    import run_paper_tables as R

    def one(frac, quant, c, gpu):
        with tempfile.TemporaryDirectory() as td:
            for f in R.DEPS + [s for _, s in R.SIMS]:
                shutil.copy(os.path.join(R.REPO, f), td)
            p = os.path.join(td, "semduplex_scheduler.py")
            s = open(p).read()
            # Pin the search to ONE plan: the swept reserve fraction, KV on the
            # device, full device capacity (no declining). Each replace asserts,
            # because a silent no-op here produced a perfectly flat curve when
            # the search was rewritten under this script -- every point was the
            # default search re-run, identical by construction.
            reps = [
                ("""        _fr = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20,
               0.25, 0.30, 0.35, 0.40, 0.45, 0.50)""",
                 f"        _fr = ({frac},)"),
                ('_kv_opts = ["cxl", "host"] + (["gpu"] if gpu_hbm_capacity_bytes else [])',
                 '_kv_opts = ["cxl"]'),
                ("""        _dev_caps = sorted({cxl_dev_dram_capacity_bytes}
                           | {c * GiB for c in (32, 48)
                              if c * GiB < cxl_dev_dram_capacity_bytes})""",
                 "        _dev_caps = [cxl_dev_dram_capacity_bytes]"),
            ]
            for a, b in reps:
                assert a in s, f"patch anchor drifted: {a[:60]!r}"
                s = s.replace(a, b, 1)
            open(p, "w").write(s)
            R.patch(td, quant, 16, c, gpu)
            r = subprocess.run([sys.executable, "semduplex_scheduler.py"], cwd=td,
                               capture_output=True, text=True, timeout=900)
            m = re.search(r"Decode throughput:\s*([\d.]+)", r.stdout + r.stderr)
            return float(m.group(1)) if m else None

    out = {}
    for label, quant, c, gpu in CASES:
        tps = [one(f, quant, c, gpu) for f in FRACS]
        base = max(R.run(s, quant, 16, c, gpu)[0]
                   for n, s in R.SIMS if n != "SemSched")
        if max(tps) - min(tps) < 1e-9:
            raise SystemExit(f"flat curve for {label}: the pin no-oped")
        out[label] = {"frac": FRACS, "tps": tps, "base": base}
        print(f"  {label:<22}{tps[0]:7.2f} -> {max(tps):7.2f} t/s   base {base:6.2f}")
    json.dump(out, open(DATA, "w"), indent=1)
    return out


def plot(data):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "lines.linewidth": 1.25, "lines.markersize": 3.4,
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.45))

    for (label, (colour, marker, dash)) in zip(data, STYLE):
        d = data[label]
        # Normalise to the strongest baseline, so 1.0 is the parity line and the
        # four configurations are comparable on one axis.
        y = [t / d["base"] for t in d["tps"]]
        x = [f * 100 for f in d["frac"]]
        ax.plot(x, y, color=colour, marker=marker, linestyle=dash, label=label,
                markerfacecolor="white", markeredgewidth=0.9, clip_on=False, zorder=3)
        k = max(range(len(y)), key=lambda i: y[i])
        ax.plot(x[k], y[k], marker=marker, color=colour, markersize=5.6,
                markerfacecolor=colour, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4, clip_on=False)

    ax.axhline(1.0, color="#555555", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.text(40.2, 1.005, "parity with best baseline", ha="right", va="bottom",
            fontsize=6.6, color="#555555")

    # Every published policy fills its tiers; that is x = 0.
    ax.axvline(0, color="#B00020", linewidth=0.8, zorder=2)
    ax.annotate("every published\npolicy sits here",
                xy=(0, 0.70), xytext=(6.5, 0.635),
                fontsize=6.8, color="#B00020", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#B00020",
                                linewidth=0.7, shrinkA=0, shrinkB=2))

    ax.set_xlabel("fast memory held back as prefetch staging (%)")
    ax.set_ylabel("decode throughput\nvs. best baseline")
    ax.set_xlim(-1, 40.5)
    ax.set_ylim(0.60, 1.55)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}$\\times$"))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=2.1,
              borderaxespad=0.3, labelspacing=0.32)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.015)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="re-run the sweep instead of plotting the saved JSON")
    a = ap.parse_args()
    plot(measure() if a.measure or not os.path.exists(DATA) else json.load(open(DATA)))
