"""
Real-hardware figure: measured decode throughput against the reserved share of
the host-memory budget, on two machines, with the timing engine's prediction
from each machine's measured rates.

    python hardware/plot_hw.py

Reads hardware/results/{rtx4500,a100}/reserve_bench.csv and writes
figures/fig_hardware.pdf, one IEEE column wide.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predict import predict                 # noqa: E402

MACHINES = [("rtx4500", "RTX 4500 Ada + NVMe"), ("a100", "A100 80 GB + NVMe")]
OUT = os.path.join(HERE, "..", "figures", "fig_hardware.pdf")


def load(name):
    d = os.path.join(HERE, "results", name)
    rows = list(csv.DictReader(open(os.path.join(d, "reserve_bench.csv"))))
    meta = json.load(open(os.path.join(d, "reserve_bench_meta.json")))
    return rows, meta


def main():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.2,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 3.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))
    for ax, (name, title) in zip(axes, MACHINES):
        rows, meta = load(name)
        reserves = sorted({float(r["reserve"]) for r in rows})
        x = [r * 100 for r in reserves]

        def pick(k_test):
            out = []
            for res in reserves:
                cand = [r for r in rows if float(r["reserve"]) == res and k_test(int(r["K"]))]
                out.append(min(cand, key=lambda r: float(r["step_s"])))
            return out

        deep, shallow = pick(lambda k: k >= 4), pick(lambda k: k == 1)
        b = meta["batch"]
        ax.plot(x, [b / float(r["step_s"]) for r in shallow], color="#8C98A2",
                marker="v", markerfacecolor="white", label="measured, $K{=}1$")
        ax.plot(x, [b / float(r["step_s"]) for r in deep], color="#0072B2",
                marker="o", label="measured, best $K{\\geq}4$")
        ax.plot(x, [b / predict(meta, r) for r in deep], color="#D55E00",
                linestyle="--", linewidth=1.1, label="engine prediction")
        ax.set_title(title, pad=3)
        ax.set_xlabel("reserved share of budget (%)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("decode throughput (tok/s)")
    fig.tight_layout(pad=0.35, w_pad=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=0.9)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.015)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
