"""
Lookahead sensitivity: decode throughput against prefetch depth K for every
policy, at the configuration where TierSched leads most (CPU INT8 16H+48C) and
one where it ties (+GPU INT8 16H+64C).

Each static baseline is shown at its best fixed reserve for that K, out of the
search's own 15 reserve fractions (0-50%) for host and device DRAM, chosen
separately; the dynamic-replacement baseline at
its best KV tier; and TierSched with its own search. Runs under the pass-through NAND model. Regenerate with:

    python plot_depth.py --measure     # re-run the sweep, then plot
    python plot_depth.py               # plot from the saved JSON

Output: figures/fig_depth.pdf, sized for one IEEE column.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "depth_sweep.json")
OUT = os.path.join(REPO, "figures", "fig_depth.pdf")

DEPTHS = [1, 2, 4, 8, 16, 32]
CASES = [("CPU, INT8, 16H+48C", "int8", 48, 0),
         ("+GPU, INT8, 16H+64C", "int8", 64, 28)]
POLICIES = ["FlexGen", "LIA", "AimPod", "LLMFlash", "Dynamic", "TierSched"]
STYLE = {"FlexGen":  ("#A9B3BB", "v", "-"),
         "LIA":      ("#8C98A2", "<", "-"),
         "AimPod":   ("#6F7C87", ">", "-"),
         "LLMFlash": ("#4F5B66", "s", "-"),
         "Dynamic":  ("#D55E00", "D", "--"),
         "TierSched": ("#0072B2", "o", "-")}


def measure():
    os.environ["NAND_PASS_THROUGH"] = "1"
    sys.path.insert(0, REPO)
    import run_paper_tables as R
    files = {n: s for n, s in R.SIMS}
    jobs = []
    for label, quant, c, gpu in CASES:
        for k in DEPTHS:
            for name in POLICIES[:4]:
                for fh in R.EQ_RESERVES:
                    for fd in R.EQ_RESERVES:
                        jobs.append(("base", files[name], quant, c, gpu, k, (fh, fd)))
            for kv in ("cxl", "host"):
                jobs.append(("dyn", "dynamic_baseline.py", quant, c, gpu, k, kv))
            jobs.append(("sem", "tiersched.py", quant, c, gpu, k, None))
    with cf.ProcessPoolExecutor() as ex:
        res = dict(ex.map(R._eq_job, jobs))
    name_of = {s: n for n, s in R.SIMS}
    out = {}
    for label, quant, c, gpu in CASES:
        out[label] = {"K": DEPTHS}
        for name in POLICIES:
            ys = []
            for k in DEPTHS:
                def match(j):
                    if j[2:6] != (quant, c, gpu, k):
                        return False
                    if name == "Dynamic":
                        return j[0] == "dyn"
                    if name == "TierSched":
                        return j[0] == "sem"
                    return j[0] == "base" and name_of[j[1]] == name
                ys.append(max(v or 0.0 for j, v in res.items() if match(j)))
            out[label][name] = ys
    json.dump(out, open(DATA, "w"), indent=1)
    return out


def plot(data):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "lines.linewidth": 1.25, "lines.markersize": 3.4,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2))
    for ax, label in zip(axes, data):
        d = data[label]
        for name in POLICIES:
            colour, marker, dash = STYLE[name]
            ax.plot(d["K"], d[name], color=colour, marker=marker, linestyle=dash,
                    label={"Dynamic": "Dynamic (Belady)"}.get(name, name),
                    markerfacecolor="white" if name != "TierSched" else colour,
                    markeredgewidth=0.9, zorder=4 if name == "TierSched" else 3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(d["K"])
        ax.set_xticklabels([str(k) for k in d["K"]])
        ax.set_xlabel("lookahead depth $K$")
        ax.set_title(label, pad=3)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("decode throughput (tok/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout(pad=0.35, w_pad=0.8)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.0)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.015)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="re-run the sweep instead of plotting the saved JSON")
    a = ap.parse_args()
    plot(measure() if a.measure or not os.path.exists(DATA) else json.load(open(DATA)))
