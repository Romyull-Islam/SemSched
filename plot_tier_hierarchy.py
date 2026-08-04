"""
plot_tier_hierarchy.py — Fig. "memory tier capacity vs bandwidth", including the
GPU tier.

Purpose: the paper models the CXL-to-host memory path, not a GPU end-to-end. This
figure makes the positioning explicit and supports the §VI limitation and the
compute-engine argument in §V-E, by showing (a) that no single tier has both the
capacity and the bandwidth a 72B model needs, and (b) that attaching a GPU does
not introduce a new bottleneck, because the PCIe feed link is faster than the CXL
device it is fed from. Tier ordering is preserved, so SemSched's placement
decisions carry over unchanged.

These are published/spec bandwidths, NOT simulated results. Labeled as such.

Style matches plot_paper_figures.py so the figure sits with the others.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# (label, capacity GB, bandwidth GB/s, group, annotation offset)
TIERS = [
    ("GPU HBM\n(H100 PCIe 80GB)", 80, 2039.0, "gpu",  (66,  -6)),
    ("Host DRAM\n(DDR5-5600)",  32,     38.4, "host", (-6, -40)),
    ("CXL device DRAM\n(CMM-H cache)", 48, 27.0, "cxl", (86, 24)),
    ("CXL NAND\n(CMM-H)",    1024,      5.0, "cxl",  (-4,  26)),
    ("NVMe SSD\n(Gen4)",      512,      7.6, "ssd",  (-14, 26)),
]

# Categorical hues assigned by tier identity, in fixed order — not cycled.
# Distinct in hue AND lightness so the figure survives grayscale printing.
CMAP = {"gpu": "#4d4d4d", "host": "#377eb8", "cxl": "#e41a1c", "ssd": "#7f7f7f"}
MARK = {"gpu": "s", "host": "o", "cxl": "D", "ssd": "^"}

MODEL_GB = 151.3   # Qwen2.5 72B at FP16
PCIE_GEN5_X16 = 63.0

fig, ax = plt.subplots(figsize=(8, 4.4))

# Working-set band: the model must be resident somewhere to the right of this.
ax.axvspan(MODEL_GB, 2000, color="#e41a1c", alpha=0.05, zorder=0)
ax.axvline(MODEL_GB, color="#e41a1c", ls=":", lw=2, zorder=1)
ax.text(MODEL_GB * 1.14, 3000, "72B FP16 working set\n151 GB — exceeds every\ntier fast enough to serve it",
        fontsize=11.5, color="#e41a1c", va="top", ha="left")

# The GPU feed link. Above CXL DRAM, so it is never the binding constraint.
ax.axhline(PCIE_GEN5_X16, color="#4d4d4d", ls="--", lw=1.8, alpha=0.8, zorder=1)
ax.text(9.3, PCIE_GEN5_X16 * 1.25, "PCIe Gen5 $\\times$16 GPU feed — 63 GB/s",
        fontsize=11.5, color="#4d4d4d", ha="left")

for label, cap, bw, grp, (dx, dy) in TIERS:
    ax.scatter(cap, bw, s=190, c=CMAP[grp], marker=MARK[grp],
               edgecolors="white", linewidths=1.8, zorder=5)
    arrow = dict(arrowstyle="-", color=CMAP[grp], lw=1.1,
                 shrinkA=2, shrinkB=9) if abs(dx) > 60 else None
    ax.annotate(label, (cap, bw), textcoords="offset points", xytext=(dx, dy),
                ha="center", fontsize=12.5, zorder=6, arrowprops=arrow)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Capacity (GB, log scale)")
ax.set_ylabel("Bandwidth (GB/s, log scale)")
ax.set_xlim(8, 2000)
ax.set_ylim(2.2, 9000)

handles = [
    Line2D([], [], marker="s", ls="", color=CMAP["gpu"],  markersize=11, label="GPU memory"),
    Line2D([], [], marker="o", ls="", color=CMAP["host"], markersize=11, label="Host memory"),
    Line2D([], [], marker="D", ls="", color=CMAP["cxl"],  markersize=11, label="CXL device (SemSched)"),
    Line2D([], [], marker="^", ls="", color=CMAP["ssd"],  markersize=11, label="Host storage"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
          ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.2)

fig.tight_layout()
fig.savefig("figures/fig_tier_hierarchy.pdf", bbox_inches="tight")
print("Saved figures/fig_tier_hierarchy.pdf")

for label, cap, bw, grp, _ in TIERS:
    fits = "fits" if cap >= MODEL_GB else f"{MODEL_GB/cap:.1f}x too small"
    print(f"  {label.replace(chr(10),' '):<34} {cap:>6} GB {bw:>8.1f} GB/s   {fits}")
print(f"\n  PCIe Gen5 x16 feed = {PCIE_GEN5_X16} GB/s "
      f"= {PCIE_GEN5_X16/27.0:.1f}x the CXL device DRAM tier")
print("  => the CXL device stays the bottleneck when a GPU is attached;")
print("     tier ordering, and therefore SemSched's placement, is unchanged.")
