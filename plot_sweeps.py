"""
plot_sweeps.py -- the batch-size and model-scale results as figures.

Both were 8-plus-column tables; a reviewer absorbs a crossover from a curve in
seconds and from a 96-cell grid never, and the venue's papers are figure-led.
The full grids still regenerate from run_paper_tables.py --batch-sweep and
--models, so nothing is lost to the artifact by plotting the shape.

Data comes from sweep_figs.json, parsed from the harness's own sweep output.

    python plot_sweeps.py

Outputs figures/fig_batch.pdf and figures/fig_models.pdf, one column wide.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(REPO, "sweep_figs.json")))

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.25, "lines.markersize": 3.2,
})

STYLE = {("CPU", "FP16"): ("#0072B2", "o", "-"),
         ("CPU", "INT8"): ("#0072B2", "s", "--"),
         ("GPU", "FP16"): ("#D55E00", "^", "-"),
         ("GPU", "INT8"): ("#D55E00", "D", "--")}
LBL = {("CPU", "FP16"): "CPU, FP16", ("CPU", "INT8"): "CPU, INT8",
       ("GPU", "FP16"): "$+$5090, FP16", ("GPU", "INT8"): "$+$5090, INT8"}


def deco(ax):
    ax.axhline(1.0, color="#555555", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}$\\times$"))


# ── batch: median line + min-max band per (platform, quant), 48C shown solid ──
fig, ax = plt.subplots(figsize=(3.45, 2.15))
BS = [1, 2, 4, 8, 16, 32, 64, 128]
for (plat, quant), (c, m, ls) in STYLE.items():
    rows = [v for k, v in D["batch"].items() if k.startswith(f"{plat}|{quant}|")]
    med = [sorted(r[b] if isinstance(list(r)[0], int) else r[str(b)]
                  for r in rows)[len(rows) // 2] for b in BS]
    lo = [min(r[b] if isinstance(list(r)[0], int) else r[str(b)] for r in rows) for b in BS]
    hi = [max(r[b] if isinstance(list(r)[0], int) else r[str(b)] for r in rows) for b in BS]
    ax.fill_between(BS, lo, hi, color=c, alpha=0.10 if quant == "FP16" else 0.14,
                    linewidth=0, zorder=2)
    ax.plot(BS, med, color=c, marker=m, linestyle=ls, label=LBL[(plat, quant)],
            markerfacecolor="white", markeredgewidth=0.9, zorder=3, clip_on=False)
ax.set_xscale("log", base=2)
ax.set_xticks(BS)
ax.set_xticklabels([str(b) for b in BS])
ax.set_xlabel("batch size $B$")
ax.set_ylabel("speedup vs. best baseline")
ax.axvspan(0.9, 5.7, color="#B00020", alpha=0.05, zorder=0)
ax.text(2.35, 1.52, "LLM-in-a-Flash's\nsparsity regime", fontsize=6.4,
        color="#B00020", ha="center", va="top")
deco(ax)
ax.set_xlim(0.93, 138)
ax.legend(frameon=False, loc="lower right", handlelength=1.9,
          borderaxespad=0.3, labelspacing=0.3)
fig.tight_layout(pad=0.35)
fig.savefig(os.path.join(REPO, "figures", "fig_batch.pdf"),
            bbox_inches="tight", pad_inches=0.015)
print("wrote figures/fig_batch.pdf")

# ── models: ratio per model, categorical x, both platforms and quants ─────────
fig, ax = plt.subplots(figsize=(3.45, 2.05))
MODELS = ["Mistral 7B", "Llama 13B", "Qwen3 20B", "Qwen2.5 72B", "Llama 405B"]
X = range(len(MODELS))
for (plat, quant), (c, m, ls) in STYLE.items():
    r = D["models"][f"{plat}|{quant}"]
    ax.plot(X, [r[mm] for mm in MODELS], color=c, marker=m, linestyle=ls,
            label=LBL[(plat, quant)], markerfacecolor="white",
            markeredgewidth=0.9, zorder=3, clip_on=False)
ax.set_xticks(list(X))
ax.set_xticklabels(["7B", "13B", "20B", "72B", "405B"])
ax.set_xlabel("model (16H$+$48C)")
ax.set_ylabel("speedup vs. best baseline")
deco(ax)
ax.set_ylim(0.93, 1.55)
ax.annotate("fits in\nfast memory", xy=(0, 1.005), xytext=(0.05, 1.18),
            fontsize=6.4, color="#555555", ha="left",
            arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.7))
ax.annotate("NAND carries\n>80% of step", xy=(4, 1.12), xytext=(3.35, 1.30),
            fontsize=6.4, color="#555555", ha="left",
            arrowprops=dict(arrowstyle="->", color="#555555", linewidth=0.7))
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, handlelength=1.9, borderaxespad=0.1, labelspacing=0.3,
          columnspacing=1.2)
fig.tight_layout(pad=0.35)
fig.savefig(os.path.join(REPO, "figures", "fig_models.pdf"),
            bbox_inches="tight", pad_inches=0.015)
print("wrote figures/fig_models.pdf")
