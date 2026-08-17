"""
plot_evaluation.py -- the three evaluation panels, each measured.

  (a) decode throughput across memory capacity, which is the headline claim
  (b) per-token write stall, which is what the duplex lane actually buys
  (c) ShareGPT speedup distribution, on real prompt and response lengths

Every value is produced by the same harness that produces the tables, so a
figure cannot drift from a table. Nothing is read from a saved PDF.

    python plot_evaluation.py            # measure and plot
    python plot_evaluation.py --cached   # replot from evaluation_data.json

Output: figures/fig_evaluation.pdf, sized for a two-column IEEE float.
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "evaluation_data.json")
OUT = os.path.join(REPO, "figures", "fig_evaluation.pdf")

sys.path.insert(0, REPO)

POLICIES = ["FlexGen", "LIA", "AimPod", "LLMFlash", "SemSched"]
# Okabe-Ito; SemSched is the only saturated colour, the baselines are muted.
COL = {"FlexGen": "#8C9BA5", "LIA": "#A9B4BC", "AimPod": "#C2CBD1",
       "LLMFlash": "#7E8B94", "SemSched": "#0072B2"}
HATCH = {"SemSched": ""}


def measure():
    import run_paper_tables as R
    out = {"decode": {}, "stall": {}, "sharegpt": {}}

    # (a) decode across capacity, INT8 where the mechanism has room to act
    for c in R.CXLS:
        col = {}
        for name, sim in R.SIMS:
            col[name] = R.run(sim, "int8", 16, c, R.GPU_GB)[0]
        out["decode"][str(c)] = col
        print(f"  decode 16H+{c}C: " + "  ".join(f"{k} {v:.1f}" for k, v in col.items()))

    # (b) write stall per policy, at the documented 48 GB module
    for name, sim in R.SIMS:
        txt = R.run(sim, "int8", 16, 48, R.GPU_GB)[2]
        m = re.search(r"Write_Stall_Time_s:\s*([\d.]+)", txt)
        out["stall"][name] = (float(m.group(1)) * 1e3) if m else 0.0
    print("  write stall (ms): " + "  ".join(f"{k} {v:.3f}" for k, v in out["stall"].items()))

    # (c) ShareGPT, both platforms
    for gpu, key in ((0, "cpu"), (R.GPU_GB, "gpu")):
        out["sharegpt"][key] = R.sharegpt(gpu)
    json.dump(out, open(DATA, "w"), indent=1)
    return out


def plot(d):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.05))

    # ── (a) decode across capacity ──────────────────────────────────────────
    ax = axes[0]
    caps = sorted(d["decode"], key=int)
    w, n = 0.16, len(POLICIES)
    for j, pol in enumerate(POLICIES):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(caps))]
        ys = [d["decode"][c][pol] for c in caps]
        ax.bar(xs, ys, width=w, color=COL[pol], label=pol,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels([f"16H+{c}C" for c in caps])
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_title("(a) capacity sensitivity, INT8", fontsize=8, pad=4)
    # One legend for the whole figure, above all three panels: panels (a) and
    # (b) share the policy colours, and an in-panel legend either sits on the
    # bars or collides with the panel title.
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 1.035),
               handlelength=1.1, columnspacing=1.3, fontsize=7)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.55, zorder=0)

    # ── (b) write stall ─────────────────────────────────────────────────────
    ax = axes[1]
    ys = [d["stall"][p] for p in POLICIES]
    ax.bar(range(len(POLICIES)), ys, width=0.62,
           color=[COL[p] for p in POLICIES], edgecolor="white",
           linewidth=0.4, zorder=3)
    for i, v in enumerate(ys):
        ax.text(i, v + max(ys) * 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6.4)
    ax.set_xticks(range(len(POLICIES)))
    ax.set_xticklabels(POLICIES, rotation=28, ha="right")
    ax.set_ylabel("write stall per token (ms)")
    ax.set_title("(b) KV write stall, INT8 16H+48C", fontsize=8, pad=4)
    ax.set_ylim(0, max(ys) * 1.22)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.55, zorder=0)

    # ── (c) ShareGPT distribution ───────────────────────────────────────────
    ax = axes[2]
    for key, lbl, col, dash in (("gpu", "$+$RTX 5090", "#0072B2", "-"),
                                ("cpu", "CPU-only", "#D55E00", "--")):
        r = sorted(d["sharegpt"][key])
        ax.plot(r, [i / len(r) for i in range(1, len(r) + 1)],
                color=col, linestyle=dash, linewidth=1.3, label=lbl, zorder=3)
    ax.axvline(1.0, color="#555555", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.text(1.006, 0.06, "parity", fontsize=6.4, color="#555555")
    ax.set_xlabel("speedup vs. best baseline")
    ax.set_ylabel("fraction of prompts")
    ax.set_title("(c) 50 real ShareGPT prompts", fontsize=8, pad=4)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.3)
    ax.grid(color="#DDDDDD", linewidth=0.55, zorder=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}$\\times$"))

    for a in axes:
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout(pad=0.4, w_pad=1.4, rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true",
                    help="replot from evaluation_data.json instead of measuring")
    a = ap.parse_args()
    plot(json.load(open(DATA)) if a.cached and os.path.exists(DATA) else measure())
