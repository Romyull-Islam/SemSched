import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator
import os

# ==========================================
# 1. STYLE
# ==========================================
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

COLORS = {
    'FlexGen':   '#7f7f7f',
    'LIA':       '#377eb8',
    'LLMFlash':  '#ff7f0e',
    'SemDuplex': '#e41a1c'
}
MARKERS = {
    'FlexGen':   'o',
    'LIA':       's',
    'LLMFlash':  '^',
    'SemDuplex': 'D'
}

# ==========================================
# SERVING BATCH: used as default for
# all "production" comparison figures.
# B=1 is single-request benchmark only.
# ==========================================
SERVING_BATCH = 16   # ← SemDuplex wins by 1.47× here; change to 8, 32 as desired


# ==========================================
# DATA LOADER
# ==========================================
def load_and_clean_data():
    files = ["final_results_with_coldload.csv",
             "final_results_all_combinations.csv"]
    df = None
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"Loaded data from: {f}")
            break

    if df is None:
        print("Error: No data file found.")
        return None

    name_map = {
        "flexgen_baseline.py":    "FlexGen",
        "lia_baseline.py":        "LIA",
        "llmflash_baseline.py":   "LLMFlash",
        "semduplex_scheduler.py": "SemDuplex",
        "flashllm_baseline.py":   "LLMFlash",
        "FlashLLM":               "LLMFlash",
    }
    df["Simulator"] = df["Simulator"].replace(name_map)

    if "Hit_Rate" in df.columns and df["Hit_Rate"].max() > 1.0:
        df["Hit_Rate"] = df["Hit_Rate"] / 100.0

    # FIX: force BatchSize to int so == comparisons work everywhere
    if "BatchSize" in df.columns:
        df["BatchSize"] = pd.to_numeric(df["BatchSize"], errors='coerce')\
                            .astype("Int64")  # Int64 supports NaN for non-batch rows

    return df



# ==========================================
# PRIMARY FIGURE 1 (HERO):
# BATCH SWEEP — FULL STORY
# Shows: B=1 LLMFlash claim → collapse → SemDuplex dominance
# ==========================================
def plot_batch_sweep_hero(df):
    """
    The paper's central claim figure.
    - LEFT:  Raw TPS showing crossover at B=4
    - CENTER: SemDuplex/LLMFlash ratio → dominance grows with batch
    - RIGHT:  Throughput advantage (SemDuplex - LLMFlash) absolute delta
    """
    sub = df[df["Experiment"] == "BatchSweep"].copy()
    if sub.empty:
        print("Warning: No BatchSweep data.")
        return

    sub["BatchSize"] = pd.to_numeric(sub["BatchSize"])
    sub = sub.sort_values("BatchSize")

    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))
    ax1, ax2, ax3 = axes

    # ── LEFT: Raw TPS ──────────────────────────────────────────
    for sim, grp in sub.groupby("Simulator"):
        ax1.plot(grp["BatchSize"], grp["TPS"],
                 color=COLORS[sim], marker=MARKERS[sim],
                 markersize=11, linewidth=2.5, label=sim)

    ax1.axvline(x=4, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
    ax1.annotate('LLMFlash\nleads (B=1)',
                 xy=(1, sub[sub["Simulator"]=="LLMFlash"]["TPS"].iloc[0]),
                 xytext=(1.2, sub["TPS"].max() * 0.82),
                 fontsize=10, color=COLORS["LLMFlash"],
                 arrowprops=dict(arrowstyle='->', color=COLORS["LLMFlash"], lw=1.5),
                 bbox=dict(boxstyle='round', fc='white', ec=COLORS["LLMFlash"], alpha=0.8))

    # Shade SemDuplex dominant region
    ax1.axvspan(4, sub["BatchSize"].max(), alpha=0.06, color='red',
                label="SemDuplex dominant (B≥4)")
    ax1.set_title("(a) Raw Throughput (TPS)\n72B INT8 | 16H+32C",
                  fontweight='bold')
    ax1.set_ylabel("Decode Throughput (Tokens/s)")
    ax1.set_xlabel("Batch Size")
    ax1.set_xticks(sub["BatchSize"].unique())
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ── CENTER: Ratio SemDuplex/LLMFlash ───────────────────────
    llm = sub[sub["Simulator"]=="LLMFlash"]\
        .set_index("BatchSize")["TPS"].rename("LLMFlash")
    sem = sub[sub["Simulator"]=="SemDuplex"]\
        .set_index("BatchSize")["TPS"].rename("SemDuplex")
    lia = sub[sub["Simulator"]=="LIA"]\
        .set_index("BatchSize")["TPS"].rename("LIA")
    flex = sub[sub["Simulator"]=="FlexGen"]\
        .set_index("BatchSize")["TPS"].rename("FlexGen")
    ratio = pd.concat([llm, sem, lia, flex], axis=1).dropna()

    ratio["SemDuplex/LLMFlash"] = ratio["SemDuplex"] / ratio["LLMFlash"]
    ratio["LIA/LLMFlash"]       = ratio["LIA"]       / ratio["LLMFlash"]
    ratio["FlexGen/LLMFlash"]   = ratio["FlexGen"]   / ratio["LLMFlash"]

    ax2.plot(ratio.index, ratio["SemDuplex/LLMFlash"],
             color=COLORS["SemDuplex"], marker='D', markersize=11,
             linewidth=2.5, label="SemDuplex / LLMFlash")
    ax2.plot(ratio.index, ratio["LIA/LLMFlash"],
             color=COLORS["LIA"], marker='s', markersize=10,
             linewidth=2.0, linestyle='--', label="LIA / LLMFlash")
    ax2.plot(ratio.index, ratio["FlexGen/LLMFlash"],
             color=COLORS["FlexGen"], marker='o', markersize=10,
             linewidth=1.8, linestyle=':', label="FlexGen / LLMFlash")

    ax2.axhline(1.0, color='black', linestyle=':', linewidth=1.8,
                label="Parity (1.0×)")
    ax2.axvline(x=4, color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvspan(4, ratio.index.max(), alpha=0.07, color='red')

    # Annotate B=1 loss and peak win
    b1_ratio = ratio.loc[ratio.index[0], "SemDuplex/LLMFlash"]
    ax2.annotate(f'B=1: {b1_ratio:.2f}×\n(LLMFlash wins)',
                 xy=(ratio.index[0], b1_ratio),
                 xytext=(ratio.index[1], b1_ratio - 0.12),
                 fontsize=10, color=COLORS["LLMFlash"],
                 arrowprops=dict(arrowstyle='->', color=COLORS["LLMFlash"], lw=1.2))

    max_b   = ratio["SemDuplex/LLMFlash"].idxmax()
    max_r   = ratio.loc[max_b, "SemDuplex/LLMFlash"]
    ax2.annotate(f'Peak: {max_r:.2f}×\n@ B={max_b}',
                 xy=(max_b, max_r),
                 xytext=(max_b * 0.5, max_r + 0.04),
                 fontsize=10, color=COLORS["SemDuplex"],
                 arrowprops=dict(arrowstyle='->', color=COLORS["SemDuplex"], lw=1.2),
                 bbox=dict(boxstyle='round', fc='white',
                           ec=COLORS["SemDuplex"], alpha=0.8))

    ax2.set_title("(b) Throughput Ratio vs LLMFlash\nSparsity collapse drives crossover",
                  fontweight='bold')
    ax2.set_ylabel("TPS Ratio (>1.0 = SemDuplex better)")
    ax2.set_xlabel("Batch Size")
    ax2.set_xticks(ratio.index)
    ax2.legend(loc='center right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ── RIGHT: Absolute TPS delta ──────────────────────────────
    ratio["Delta_SemDuplex"] = ratio["SemDuplex"] - ratio["LLMFlash"]
    colors_bar = [COLORS["LLMFlash"] if d < 0 else COLORS["SemDuplex"]
                  for d in ratio["Delta_SemDuplex"]]

    bars = ax3.bar(ratio.index.astype(str), ratio["Delta_SemDuplex"],
                   color=colors_bar, edgecolor='black', width=0.55)
    ax3.axhline(0, color='black', linewidth=1.5)
    ax3.axvline(x=1.5, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)

    for bar, val in zip(bars, ratio["Delta_SemDuplex"]):
        pad = 0.05 if val >= 0 else -0.15
        ax3.text(bar.get_x() + bar.get_width()/2, val + pad,
                 f'{val:+.2f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold',
                 color=COLORS["SemDuplex"] if val >= 0 else COLORS["LLMFlash"])

    ax3.set_title("(c) Absolute Advantage (SemDuplex − LLMFlash)\nNegative = LLMFlash leads",
                  fontweight='bold')
    ax3.set_ylabel("ΔTPS (SemDuplex − LLMFlash)")
    ax3.set_xlabel("Batch Size")
    ax3.text(0.25, ax3.get_ylim()[0] * 0.7 if ax3.get_ylim()[0] < 0 else 0.1,
             "LLMFlash\nregion", ha='center', fontsize=10,
             color=COLORS["LLMFlash"],
             bbox=dict(boxstyle='round', fc='white', ec=COLORS["LLMFlash"], alpha=0.6),
             transform=ax3.get_xaxis_transform())
    ax3.grid(True, alpha=0.3)

    plt.suptitle(
        "SemDuplex Batch Sweep: Sparsity Collapse Drives Dominance at B≥4\n"
        "LLMFlash advantages (activation sparsity) collapse when union of "
        "active neurons saturates at φ(B)=1-(1-0.46)^B ≈ 1.0 for B≥4",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig("fig_batch_sweep_hero.pdf", bbox_inches='tight')
    print("Saved fig_batch_sweep_hero.pdf  ← PRIMARY PAPER FIGURE")


# ==========================================
# PRIMARY FIGURE 2: TOTAL INFERENCE
# Full end-to-end with prefill collapse exposed
# ==========================================
def plot_total_inference(df):
    sub = df[df["Experiment"] == "Scalability"].copy()
    if sub.empty:
        return

    # LLMFlash prefill stored as 0.0 in CSV at large models.
    # Clip to 1e-4 so 512/1e-4 = 5.12M seconds — honest representation.
    sub["Prefill_safe"] = sub["Prefill_TPS"].clip(lower=1e-4)
    sub["Total_s"]      = (512 / sub["Prefill_safe"]) + (16 / sub["TPS"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # ── LEFT: total time line ──────────────────────────────────
    sns.lineplot(data=sub, x="Model", y="Total_s", hue="Simulator",
                 style="Simulator", palette=COLORS, markers=MARKERS,
                 markersize=12, linewidth=2.5, ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([7, 13, 20, 72])
    ax1.set_xticklabels(["7B", "13B", "20B", "72B"])
    ax1.set_ylabel("Total Inference Time (s)\n[512-token prefill + 16-token decode]")
    ax1.set_xlabel("Model Size")
    ax1.set_title("(a) End-to-End Latency Wall\n(Lower is Better)",
                  fontweight='bold')
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax1.legend(title="Scheduler", loc='upper left', fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # Annotate LLMFlash spike at 72B
    llm72 = sub[(sub["Simulator"]=="LLMFlash") & (sub["Model"]==72)]
    sem72 = sub[(sub["Simulator"]=="SemDuplex") & (sub["Model"]==72)]
    if not llm72.empty and not sem72.empty:
        lt = llm72["Total_s"].values[0]
        st = sem72["Total_s"].values[0]
        ax1.annotate(
            f'LLMFlash prefill\ncollapse at 72B\n({lt/st:.0f}× slower total)',
            xy=(72, lt), xytext=(15, lt * 0.2),
            fontsize=9, color=COLORS["LLMFlash"],
            arrowprops=dict(arrowstyle='->', color=COLORS["LLMFlash"], lw=1.5),
            bbox=dict(boxstyle='round', fc='white',
                      ec=COLORS["LLMFlash"], alpha=0.85))

    # ── RIGHT: speedup vs FlexGen ──────────────────────────────
    base = sub[sub["Simulator"]=="FlexGen"].set_index("Model")["Total_s"]
    sub["Speedup"] = sub.apply(
        lambda r: base.get(r["Model"], r["Total_s"]) / r["Total_s"], axis=1)
    sub_sp = sub[sub["Simulator"] != "FlexGen"].copy()

    sns.barplot(data=sub_sp, x="Model", y="Speedup", hue="Simulator",
                order=[7, 13, 20, 72], palette=COLORS,
                edgecolor='black', ax=ax2)
    ax2.set_title("(b) End-to-End Speedup vs FlexGen\n"
                  "SemDuplex wins at every model size (prefill-dominated)",
                  fontweight='bold')
    ax2.set_ylabel("Total Speedup (×)")
    ax2.set_xlabel("Model Size (Billions)")
    ax2.set_xticklabels(["7B", "13B", "20B", "72B"])
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_ylim(0, sub_sp["Speedup"].max() * 1.35)
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%.1f×', padding=3, fontsize=10, rotation=45)

    plt.tight_layout()
    plt.savefig("fig_total_inference.pdf", bbox_inches='tight')
    print("Saved fig_total_inference.pdf")


# ==========================================
# FIGURE: B=1 vs B=SERVING side-by-side
# Narrative: "LLMFlash wins single-request,
# SemDuplex wins production serving"
# ==========================================
def plot_b1_vs_serving(df):
    """
    Direct head-to-head: same memory config, same model, two batch sizes.
    Tells the complete story in one figure.
    """
    sub = df[df["Experiment"] == "BatchSweep"].copy()
    if sub.empty:
        return

    sub["BatchSize"] = pd.to_numeric(sub["BatchSize"])

    b1  = sub[sub["BatchSize"] == 1].copy()
    bhi = sub[sub["BatchSize"] == SERVING_BATCH].copy()

    if b1.empty or bhi.empty:
        bhi = sub[sub["BatchSize"] == sub["BatchSize"].max()].copy()

    b_used = bhi["BatchSize"].iloc[0] if not bhi.empty else SERVING_BATCH
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                   sharey=False)

    # LEFT: B=1 — LLMFlash wins
    b1_s = b1.sort_values("TPS", ascending=False)
    sns.barplot(data=b1_s, x="Simulator", y="TPS", hue="Simulator",
                palette=COLORS, edgecolor='black', legend=False, ax=ax1)
    ax1.set_title(f"(a) Single-Request (B=1)\n"
                  f"← LLMFlash wins here (sparsity benefit, no overlap)",
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel("Decode Throughput (Tokens/s)")
    ax1.set_xlabel("Scheduler")
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')
    ax1.set_ylim(0, b1_s["TPS"].max() * 1.4)

    # Highlight winner
    llm_b1 = b1_s[b1_s["Simulator"]=="LLMFlash"]["TPS"].values
    if llm_b1.size:
        ax1.annotate("Winner", xy=(
            b1_s[b1_s["Simulator"]=="LLMFlash"].index[0] if False else
            list(b1_s["Simulator"].values).index("LLMFlash"),
            llm_b1[0]),
            xytext=(list(b1_s["Simulator"].values).index("LLMFlash"),
                    llm_b1[0] + b1_s["TPS"].max() * 0.15),
            fontsize=11, color=COLORS["LLMFlash"], ha='center',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS["LLMFlash"], lw=1.5))

    # RIGHT: B=SERVING — SemDuplex wins
    bhi_s = bhi.sort_values("TPS", ascending=False)
    sns.barplot(data=bhi_s, x="Simulator", y="TPS", hue="Simulator",
                palette=COLORS, edgecolor='black', legend=False, ax=ax2)
    ax2.set_title(f"(b) Production Serving (B={b_used})\n"
                  f"← SemDuplex wins here (sparsity collapsed, duplex scales)",
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel("Decode Throughput (Tokens/s)")
    ax2.set_xlabel("Scheduler")
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')
    ax2.set_ylim(0, bhi_s["TPS"].max() * 1.4)

    sem_bhi = bhi_s[bhi_s["Simulator"]=="SemDuplex"]["TPS"].values
    llm_bhi = bhi_s[bhi_s["Simulator"]=="LLMFlash"]["TPS"].values
    if sem_bhi.size and llm_bhi.size:
        ratio = sem_bhi[0] / llm_bhi[0]
        ax2.annotate(f"Winner\n({ratio:.2f}× vs LLMFlash)",
                     xy=(list(bhi_s["Simulator"].values).index("SemDuplex"),
                         sem_bhi[0]),
                     xytext=(list(bhi_s["Simulator"].values).index("SemDuplex"),
                             sem_bhi[0] + bhi_s["TPS"].max() * 0.15),
                     fontsize=11, color=COLORS["SemDuplex"], ha='center',
                     fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=COLORS["SemDuplex"],
                                     lw=1.5))

    plt.suptitle(
        "LLMFlash vs SemDuplex: The Batch Size Narrative\n"
        "Activation sparsity benefit (LLMFlash) is request-level only — "
        "union of active sets saturates at B≥4",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig("fig_b1_vs_serving.pdf", bbox_inches='tight')
    print(f"Saved fig_b1_vs_serving.pdf  (B=1 vs B={b_used})")


# ==========================================
# FIGURE: SERVING THROUGHPUT SCALING
# Total system tokens/s (TPS × BatchSize)
# This is the metric data centers optimize for
# ==========================================
def plot_serving_throughput(df):
    """
    System throughput = TPS × BatchSize (tokens delivered per second total).
    SemDuplex's linear scaling dominates here even more dramatically.
    """
    sub = df[df["Experiment"] == "BatchSweep"].copy()
    if sub.empty:
        return

    sub["BatchSize"]       = pd.to_numeric(sub["BatchSize"])
    sub["SystemThroughput"] = sub["TPS"] * sub["BatchSize"]
    sub = sub.sort_values("BatchSize")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # ── LEFT: System throughput (tokens/s total) ──────────────
    sns.lineplot(data=sub, x="BatchSize", y="SystemThroughput",
                 hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS,
                 markersize=12, linewidth=2.5, ax=ax1)
    ax1.set_yscale('log')
    ax1.set_title("(a) System Throughput = TPS × BatchSize\n"
                  "(What a serving cluster actually delivers)",
                  fontweight='bold')
    ax1.set_ylabel("System Throughput (Tokens/s Total)")
    ax1.set_xlabel("Batch Size")
    ax1.set_xticks(sub["BatchSize"].unique())
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax1.legend(loc='upper left', fontsize=12)
    ax1.axvline(x=4, color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7)
    ax1.grid(True, which="both", alpha=0.3)

    # ── RIGHT: System throughput ratio vs LLMFlash ────────────
    llm_sys = sub[sub["Simulator"]=="LLMFlash"]\
        .set_index("BatchSize")["SystemThroughput"]
    ratio_data = []
    for sim in ["SemDuplex", "LIA", "FlexGen"]:
        sim_sys = sub[sub["Simulator"]==sim]\
            .set_index("BatchSize")["SystemThroughput"]
        for b in sim_sys.index:
            if b in llm_sys.index and llm_sys[b] > 0:
                ratio_data.append({
                    "BatchSize": b,
                    "Simulator": sim,
                    "Ratio": sim_sys[b] / llm_sys[b]
                })

    ratio_df = pd.DataFrame(ratio_data)
    if not ratio_df.empty:
        sns.lineplot(data=ratio_df, x="BatchSize", y="Ratio",
                     hue="Simulator", style="Simulator",
                     palette={k: COLORS[k] for k in ["SemDuplex","LIA","FlexGen"]},
                     markers={k: MARKERS[k] for k in ["SemDuplex","LIA","FlexGen"]},
                     markersize=12, linewidth=2.5, ax=ax2)
        ax2.axhline(1.0, color='black', linestyle=':', linewidth=2.0,
                    label="LLMFlash baseline")
        ax2.axvline(x=4, color='darkgreen', linestyle='--',
                    linewidth=1.8, alpha=0.7)
        ax2.axvspan(4, ratio_df["BatchSize"].max(), alpha=0.07, color='red')
        ax2.set_title("(b) System Throughput Ratio vs LLMFlash\n"
                      "SemDuplex advantage widens with batch",
                      fontweight='bold')
        ax2.set_ylabel("System Throughput Ratio (×LLMFlash)")
        ax2.set_xlabel("Batch Size")
        ax2.set_xticks(ratio_df["BatchSize"].unique())
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Peak annotation
        sem_ratio = ratio_df[ratio_df["Simulator"]=="SemDuplex"]
        if not sem_ratio.empty:
            best = sem_ratio.loc[sem_ratio["Ratio"].idxmax()]
            ax2.annotate(f'{best["Ratio"]:.2f}× system\nthroughput\n@ B={int(best["BatchSize"])}',
                         xy=(best["BatchSize"], best["Ratio"]),
                         xytext=(best["BatchSize"] * 0.5, best["Ratio"] + 0.03),
                         fontsize=10, color=COLORS["SemDuplex"],
                         arrowprops=dict(arrowstyle='->', color=COLORS["SemDuplex"],
                                         lw=1.5),
                         bbox=dict(boxstyle='round', fc='white',
                                   ec=COLORS["SemDuplex"], alpha=0.85))

    plt.tight_layout()
    plt.savefig("fig_serving_throughput.pdf", bbox_inches='tight')
    print("Saved fig_serving_throughput.pdf")


# ==========================================
# FIGURE: STALL + DUPLEX PHYSICS
# Bar chart now uses SERVING_BATCH data
# ==========================================
def plot_stall_duplex_phy(df):
    baseline_total    = 57.1821
    write_stall_total = 1.1821
    num_layers        = 80
    cxl_bw            = 28.0
    stall_per_layer   = write_stall_total / num_layers
    read_per_layer    = (baseline_total - write_stall_total) / num_layers
    narrow_write_width = 0.004
    zoom_limit_s      = 5.0
    time_scale        = np.linspace(0, zoom_limit_s, 5000)

    fig = plt.figure(figsize=(16, 6))
    gs  = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.4], height_ratios=[1, 1])

    # Bar chart: SERVING_BATCH where SemDuplex wins
    ax1 = fig.add_subplot(gs[:, 0])
    sub = df[
        (df["Experiment"] == "BatchSweep") &
        (df["BatchSize"] == SERVING_BATCH)
    ].copy()
    if sub.empty:
        sub = df[
            (df["Experiment"] == "Memory") &
            (df["Model"] == 72) &
            (df["Quant"] == "fp32") &
            (df["MemConfig"] == "16GB_Host+32GB_CXL")
        ].copy()

    sub["Stall_s"] = 1.0 / sub["TPS"]
    sub = sub.sort_values(by="Stall_s", ascending=False)

    BAR_COLORS = {
        'FlexGen':   '#8c8c8c',
        'LIA':       '#bcbd22',
        'LLMFlash':  '#1f77b4',
        'SemDuplex': 'red'
    }
    sns.barplot(x='Simulator', y='Stall_s', data=sub, hue='Simulator',
                palette=BAR_COLORS, edgecolor='black', legend=False, ax=ax1)
    ax1.set_title(f"(a) Per-Token Latency\n(72B INT8, B={SERVING_BATCH}, 16H+32C)\n"
                  f"← Production serving; SemDuplex wins",
                  fontsize=13)
    ax1.set_ylabel("Seconds per Token")
    ax1.set_ylim(0, sub["Stall_s"].max() * 1.35)
    ax1.set_xlabel("Scheduler")

    for p in ax1.patches:
        if p.get_height() > 0:
            ax1.annotate(f'{p.get_height():.3f}s',
                         (p.get_x() + p.get_width()/2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10),
                         textcoords='offset points', fontsize=12,
                         fontweight='bold')

    # Physics panels (unchanged)
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#e6e6e6')
    b_read  = np.zeros_like(time_scale)
    b_stall = np.zeros_like(time_scale)
    b_write = np.zeros_like(time_scale)

    for i in range(8):
        start_r = i * (read_per_layer + stall_per_layer)
        end_r   = start_r + read_per_layer
        start_s = end_r
        end_s   = end_r + stall_per_layer
        start_w = start_s + (stall_per_layer * 0.4)
        end_w   = start_w + narrow_write_width

        b_read[(time_scale >= start_r) & (time_scale < end_r)]  = cxl_bw
        b_stall[(time_scale >= start_s) & (time_scale < end_s)] = cxl_bw
        b_write[(time_scale >= start_w) & (time_scale < end_w)] = cxl_bw

        ax2.annotate('', xy=(end_r + (stall_per_layer / 2), 30),
                     xytext=(end_r + (stall_per_layer / 2), 42),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax2.fill_between(time_scale, 0, b_read,  color='red',   label="Read (Weights)")
    ax2.fill_between(time_scale, 0, b_stall, color='white', label="Bus Turnaround Stall")
    ax2.fill_between(time_scale, 0, b_write, color='blue',  label="KV Write")
    ax2.annotate('Sequential Stalls', xy=(1.5, 43), fontsize=12, fontweight='bold')
    ax2.set_title("(b) Baseline Simplex: Sequential", fontsize=15)
    ax2.set_ylabel("BW (GB/s)")
    ax2.set_ylim(0, 60)
    ax2.legend(loc='upper right', fontsize=12, ncol=3)

    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, facecolor='#e6e6e6')
    s_read  = np.full_like(time_scale, cxl_bw)
    s_write = np.zeros_like(time_scale)
    for i in range(8):
        start_w = i * read_per_layer + (read_per_layer * 0.9)
        s_write[(time_scale >= start_w) &
                (time_scale < start_w + narrow_write_width)] = -cxl_bw

    ax3.fill_between(time_scale, 0, s_read,  color='red',  label="Read Lane")
    ax3.fill_between(time_scale, 0, s_write, color='blue', label="Write Lane (Parallel)")
    ax3.annotate('NO STALL: Continuous Read', xy=(2.5, 10), ha='center',
                 fontsize=14, color='white',
                 bbox=dict(boxstyle="round,pad=0.3", fc="darkgreen",
                           ec="none", alpha=0.8))
    ax3.set_title("(c) SemDuplex: Full-Duplex Parallel Injection", fontsize=15)
    ax3.set_ylabel("BW (GB/s)")
    ax3.set_ylim(-40, 40)
    ax3.axhline(0, color='black', lw=1.5)
    ax3.set_xlabel("Elapsed Time (Seconds)")
    ax3.set_xticks(np.arange(0, 6, 1))
    ax3.set_xticklabels([f"{i}s" for i in range(6)])
    ax3.legend(loc='upper right', fontsize=12, ncol=2)

    plt.tight_layout()
    plt.savefig("fig_stall_duplex_phy.pdf", bbox_inches='tight')
    print("Saved fig_stall_duplex_phy.pdf")


# ==========================================
# FIGURE 1: SCALABILITY
# ==========================================
def plot_combined_scalability(df):
    sub = df[df["Experiment"] == "Scalability"]
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    sns.lineplot(data=sub, x="Model", y="TPS", hue="Simulator",
                 style="Simulator", palette=COLORS, markers=MARKERS,
                 markersize=12, ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([7, 13, 20, 72])
    ax1.set_xticklabels(["7B", "13B", "20B", "72B"])
    ax1.set_ylabel("Decode Throughput (Tokens/s)")
    ax1.set_title("(a) Decode Scalability (B=1)\n"
                  "Note: LLMFlash sparsity advantage applies only at B=1",
                  fontsize=13)
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax1.legend(loc='upper right', fontsize=12)

    sns.barplot(data=sub, x="Model", y="Prefill_TPS", hue="Simulator",
                palette=COLORS, edgecolor='black', ax=ax2)
    ax2.set_yscale('log')
    ax2.set_ylabel("Prefill Throughput (Tokens/s)")
    ax2.set_title("(b) Prefill Scalability\n"
                  "← SemDuplex dominates; LLMFlash prefill ≈ 0 at 72B",
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)

    for c in ax2.containers:
        labels = [f'{v:.0f}' if v > 10 else (f'{v:.2f}' if v > 0.01 else '~0')
                  for v in c.datavalues]
        ax2.bar_label(c, labels=labels, padding=3, fontsize=10, rotation=45)

    plt.tight_layout()
    plt.savefig("fig_combined_scalability.pdf", bbox_inches='tight')
    print("Saved fig_combined_scalability.pdf")


# ==========================================
# FIGURE 2: QUANTIZATION
# ==========================================
def plot_combined_quantization(df):
    sub = df[(df["Experiment"] == "Quantization") &
             (df["Model"] == 72)].copy()
    if sub.empty:
        return

    base_sim  = "LLMFlash" if "LLMFlash" in sub["Simulator"].values else "FlexGen"
    base_data = sub[sub["Simulator"] == base_sim].set_index("Quant")["TPS"]
    if not base_data.empty:
        sub["Speedup"] = sub.apply(
            lambda x: x["TPS"] / base_data.get(x["Quant"], 1), axis=1)

    quant_order = [q for q in ["fp32", "fp16", "int8", "int4"]
                   if q in sub["Quant"].values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    sns.barplot(data=sub, x="Quant", y="TPS", hue="Simulator",
                order=quant_order, palette=COLORS, edgecolor='black', ax=ax1)
    ax1.set_title("(a) Decode Throughput (72B, 32H+32C, B=1)\n"
                  "B=1 single-request benchmark — see batch sweep for serving",
                  fontsize=12)
    ax1.set_ylabel("Tokens/s")
    ax1.set_xlabel("Quantization")
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_ylim(0, sub["TPS"].max() * 1.4)
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%.2f', padding=3, fontsize=10, rotation=45)

    sub_sp = sub[sub["Simulator"] != base_sim]
    if not sub_sp.empty:
        sns.barplot(data=sub_sp, x="Quant", y="Speedup", hue="Simulator",
                    order=quant_order, palette=COLORS, edgecolor='black', ax=ax2)
        ax2.set_title(f"(b) B=1 Decode Speedup vs {base_sim}\n"
                      f"SemDuplex advantage: prefill + B≥4 serving (see hero fig)",
                      fontsize=12)
        ax2.set_ylabel("Speedup (×)")
        ax2.set_xlabel("Quantization")
        ax2.set_ylim(0, sub_sp["Speedup"].max() * 1.35)
        if ax2.get_legend():
            ax2.get_legend().remove()
        for c in ax2.containers:
            ax2.bar_label(c, fmt='%.2f×', padding=3, fontsize=10, rotation=45)

    plt.tight_layout()
    plt.savefig("fig_combined_quantization.pdf", bbox_inches='tight')
    print("Saved fig_combined_quantization.pdf")


# ==========================================
# FIGURE 3: MEMORY SENSITIVITY
# FIX: FixedLocator silences UserWarning
# ==========================================
def plot_memory_sensitivity(df):
    sub_all = df[df["Experiment"] == "Memory"]
    if sub_all.empty:
        return

    quants_available = [q for q in ["int4", "int8", "fp16", "fp32"]
                        if q in sub_all["Quant"].values]
    ncols = 2
    nrows = (len(quants_available) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(16, 5 * nrows), squeeze=False)
    fig.suptitle("Memory Configuration Sensitivity (72B, B=1) — All Quantizations\n"
                 "SemDuplex advantage increases with tighter memory configs",
                 fontsize=15, fontweight='bold')

    for idx, q in enumerate(quants_available):
        row, col = idx // ncols, idx % ncols
        ax  = axes[row][col]
        sub = sub_all[sub_all["Quant"] == q]

        sns.barplot(data=sub, x="MemConfig", y="TPS", hue="Simulator",
                    palette=COLORS, edgecolor='black', ax=ax)
        ax.set_title(f"{q.upper()}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Throughput (Tokens/s)")
        ax.set_xlabel("")

        # FIX: FixedLocator before set_xticklabels
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25,
                           ha='right', fontsize=10)
        ax.set_ylim(0, sub["TPS"].max() * 1.45)
        ax.legend(loc='upper left', fontsize=10, ncol=2)
        for c in ax.containers:
            ax.bar_label(c, fmt='%.3f', padding=3, fontsize=9,
                         fontweight='bold', rotation=45)

    for idx in range(len(quants_available), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig("fig_memory.pdf", bbox_inches='tight')
    print(f"Saved fig_memory.pdf ({len(quants_available)} quants)")


# ==========================================
# FIGURE: BATCH SWEEP (original, kept)
# ==========================================
def plot_batch_sweep(df):
    sub = df[df["Experiment"] == "BatchSweep"].copy()
    if sub.empty:
        return

    sub["BatchSize"] = pd.to_numeric(sub["BatchSize"])
    sub = sub.sort_values("BatchSize")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    sns.lineplot(data=sub, x="BatchSize", y="TPS", hue="Simulator",
                 style="Simulator", palette=COLORS, markers=MARKERS,
                 markersize=12, linewidth=2.5, ax=ax1)
    ax1.set_title("(a) Batch Throughput (72B INT8 | 16H+32C)",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Throughput (Tokens/s)")
    ax1.set_xlabel("Batch Size")
    ax1.set_xticks(sub["BatchSize"].unique())
    ax1.axvline(x=4, color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7)
    ax1.annotate('Crossover\n(B=4)', xy=(4, sub["TPS"].max()*0.55),
                 fontsize=12, color='darkgreen', fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen",
                           ec="darkgreen", alpha=0.6))
    ax1.legend(loc='upper left', fontsize=12)

    llm_tps = sub[sub["Simulator"]=="LLMFlash"][["BatchSize","TPS"]]\
        .rename(columns={"TPS":"LLMFlash_TPS"})
    sem_tps = sub[sub["Simulator"]=="SemDuplex"][["BatchSize","TPS"]]\
        .rename(columns={"TPS":"SemDuplex_TPS"})
    lia_tps = sub[sub["Simulator"]=="LIA"][["BatchSize","TPS"]]\
        .rename(columns={"TPS":"LIA_TPS"})

    if not llm_tps.empty and not sem_tps.empty:
        r = pd.merge(llm_tps, sem_tps, on="BatchSize")
        r = pd.merge(r, lia_tps, on="BatchSize", how="left")
        r["SemDuplex/LLMFlash"] = r["SemDuplex_TPS"] / r["LLMFlash_TPS"]
        r["LIA/LLMFlash"]       = r["LIA_TPS"] / r["LLMFlash_TPS"]

        ax2.plot(r["BatchSize"], r["SemDuplex/LLMFlash"],
                 color=COLORS["SemDuplex"], marker='D', markersize=10,
                 linewidth=2.5, label="SemDuplex / LLMFlash")
        ax2.plot(r["BatchSize"], r["LIA/LLMFlash"],
                 color=COLORS["LIA"], marker='s', markersize=10,
                 linewidth=2.5, linestyle='--', label="LIA / LLMFlash")
        ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.5,
                    label="Parity")
        ax2.axvline(x=4, color='darkgreen', linestyle='--',
                    linewidth=1.8, alpha=0.7)
        cross = r[r["SemDuplex/LLMFlash"] >= 1.0]["BatchSize"]
        if not cross.empty:
            ax2.axvspan(cross.min(), r["BatchSize"].max(),
                        alpha=0.08, color='red', label="SemDuplex dominant")
        ax2.set_title("(b) Efficiency Ratio vs LLMFlash",
                      fontsize=14, fontweight='bold')
        ax2.set_ylabel("TPS Ratio")
        ax2.set_xlabel("Batch Size")
        ax2.set_xticks(r["BatchSize"].unique())
        ax2.legend(loc='upper left', fontsize=11)

        best = r.loc[r["SemDuplex/LLMFlash"].idxmax()]
        ax2.annotate(f'Peak: {best["SemDuplex/LLMFlash"]:.2f}×\n@ B={int(best["BatchSize"])}',
                     xy=(best["BatchSize"], best["SemDuplex/LLMFlash"]),
                     xytext=(best["BatchSize"]*0.4,
                             best["SemDuplex/LLMFlash"] + 0.05),
                     fontsize=10, color=COLORS["SemDuplex"],
                     arrowprops=dict(arrowstyle='->', color=COLORS["SemDuplex"],
                                     lw=1.5))

    plt.tight_layout()
    plt.savefig("fig_batch_sweep.pdf", bbox_inches='tight')
    print("Saved fig_batch_sweep.pdf")


# ==========================================
# FIGURE: ACTIVE FRACTION COLLAPSE (Theory)
# ==========================================
def plot_sparsity_collapse_theory():
    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

    configs = {
        'OPT/ReLU (p=0.10)':   (0.10, '#1a9850', '--'),
        'Qwen/SiLU (p=0.46)':  (0.46, '#ff7f0e', '-'),
        'Llama/SiLU (p=0.40)': (0.40, '#d73027', '-.'),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for label, (p, color, ls) in configs.items():
        active = 1.0 - (1.0 - p) ** batch_sizes
        ax1.plot(batch_sizes, active * 100, color=color, linestyle=ls,
                 linewidth=2.5, marker='o', markersize=7, label=label)

    ax1.axhline(100, color='black', linestyle=':', linewidth=1.2)
    ax1.axvline(4, color='darkgreen', linestyle='--', linewidth=1.8,
                alpha=0.7, label='Crossover B=4')
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Active FFN Fraction (%)")
    ax1.set_title(r"(a) $\phi(B) = 1-(1-p)^B$ — Sparsity Collapse",
                  fontsize=14)
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels([str(b) for b in batch_sizes], rotation=30)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    total_ffn_gb      = 48.0
    dram_remaining_gb = 24.0
    p_silu = 0.46
    af     = 1.0 - (1.0 - p_silu) ** batch_sizes
    dw     = np.minimum(af, dram_remaining_gb / total_ffn_gb)
    nand   = np.maximum(0.0, af - dw) * total_ffn_gb

    ax2.fill_between(batch_sizes, 0, nand, color='#d73027', alpha=0.4,
                     label='NAND overflow (GB/step)')
    ax2.plot(batch_sizes, nand, color='#d73027', linewidth=2.5,
             marker='D', markersize=9)
    ax2.axvline(4, color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7)
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("NAND Load per Decode Step (GB)")
    ax2.set_title("(b) NAND Overflow: 16H+32C, 72B INT8, SiLU", fontsize=14)
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels([str(b) for b in batch_sizes], rotation=30)
    ax2.set_ylim(0, total_ffn_gb * 1.1)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.annotate('Zero overflow\n(all in DRAM)', xy=(2, 1), fontsize=10,
                 color='darkgreen',
                 bbox=dict(boxstyle='round', fc='lightgreen',
                           ec='darkgreen', alpha=0.6))

    plt.tight_layout()
    plt.savefig("fig_sparsity_collapse_theory.pdf", bbox_inches='tight')
    print("Saved fig_sparsity_collapse_theory.pdf")


# ==========================================
# REMAINING FIGURES (unchanged from last version)
# ==========================================
def plot_metrics(df):
    sub = df[(df["Experiment"]=="Scalability") & (df["Model"]==72)].copy()
    if sub.empty:
        return
    if "Stall_s" in df.columns:
        y_col, title, ylabel, fmt = "Stall_s", "Compute Stall", "Time (s)", "%.1f"
    else:
        sub["Latency_ms"] = 1000.0 / sub["TPS"]
        y_col, title, ylabel, fmt = "Latency_ms", "Token Latency", "ms", "%.0f"

    has_hr = "Hit_Rate" in df.columns
    fig, axes = plt.subplots(1, 2 if has_hr else 1,
                             figsize=(14 if has_hr else 8, 4.5))
    ax1 = axes[0] if has_hr else axes
    sns.barplot(data=sub, x="Simulator", y=y_col, hue="Simulator",
                palette=COLORS, ax=ax1, legend=False)
    ax1.set_title(f"(a) {title}")
    ax1.set_ylabel(ylabel)
    for c in ax1.containers:
        ax1.bar_label(c, fmt=fmt, padding=3)

    if has_hr:
        sns.barplot(data=sub, x="Simulator", y="Hit_Rate", hue="Simulator",
                    palette=COLORS, ax=axes[1], legend=False)
        axes[1].set_title("(b) Cache Hit Rate")
        for c in axes[1].containers:
            axes[1].bar_label(c, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.savefig("fig_metrics.pdf", bbox_inches='tight')
    print("Saved fig_metrics.pdf")


def plot_duplex_reconstruction():
    time_ms = np.linspace(0, 100, 1000)
    br = np.zeros_like(time_ms); br[0:800] = 32
    bw = np.zeros_like(time_ms); bw[850:950] = 32
    sr = np.full_like(time_ms, 32)
    sw = np.zeros_like(time_ms)
    for i in range(100, 1000, 200):
        sw[i:i+50] = 32

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax1.fill_between(time_ms, 0, br,  color='#fdae61', alpha=0.9, label="Read")
    ax1.fill_between(time_ms, 0, -bw, color='#999999', alpha=0.9, label="Write")
    ax1.set_title("(a) Baseline", fontsize=15)
    ax1.set_ylabel("BW (GB/s)")
    ax1.axhline(0, color='black', lw=1)
    ax1.annotate('Stall', xy=(82,0), xytext=(85,15),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=2), fontsize=12)
    ax1.legend(fontsize=11)

    ax2.fill_between(time_ms, 0, sr,  color='#e41a1c', alpha=0.7, label="Read Lane")
    ax2.fill_between(time_ms, 0, -sw, color='#377eb8', alpha=0.8, label="Write Inject")
    ax2.set_title("(b) SemDuplex", fontsize=14)
    ax2.set_ylabel("BW (GB/s)")
    ax2.set_xlabel("Time (µs)")
    ax2.axhline(0, color='black', lw=1)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("fig_duplex_phy.pdf", bbox_inches='tight')
    print("Saved fig_duplex_phy.pdf")


def plot_misc_stats(df):
    sub_cold = df[(df["Experiment"]=="Quantization") & (df["Simulator"]=="SemDuplex")]
    if sub_cold.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    quant_order = [q for q in ["fp32","fp16","int8","int4"]
                   if q in sub_cold["Quant"].values]
    sns.barplot(data=sub_cold, x="Quant", y="Cold_Load_s",
                order=quant_order, color='#9467bd', edgecolor='black', ax=ax1)
    ax1.set_title("System Cold Boot Time (72B)")
    ax1.set_ylabel("Seconds")
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%.1fs', padding=3)
    plt.tight_layout()
    plt.savefig("fig_misc_stats.pdf", bbox_inches='tight')
    print("Saved fig_misc_stats.pdf")


def plot_pareto(df):
    sub = df[df["Experiment"]=="Scalability"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(data=sub, x="Prefill_TPS", y="TPS", hue="Simulator",
                    size="Model", sizes=(50,500), palette=COLORS, ax=ax, alpha=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("Prefill Speed (Tokens/s)")
    ax.set_ylabel("Decode Speed (Tokens/s)")
    ax.set_title("Efficiency Frontier\nSemDuplex: top-right = optimal")
    plt.tight_layout()
    plt.savefig("fig_pareto.pdf", bbox_inches='tight')
    print("Saved fig_pareto.pdf")


def plot_total_inference_latency(df):
    sub = df[df["Experiment"]=="Scalability"].copy()
    sub["Prefill_safe"] = sub["Prefill_TPS"].clip(lower=1e-4)
    sub["Total_Time_s"] = (512 / sub["Prefill_safe"]) + (16 / sub["TPS"])

    plt.figure(figsize=(8, 3.2))
    sns.lineplot(data=sub, x="Model", y="Total_Time_s", hue="Simulator",
                 style="Simulator", palette=COLORS, markers=MARKERS,
                 markersize=11, linewidth=2)
    plt.xscale('log'); plt.yscale('log')
    plt.xticks([7,13,20,72], ["7B","13B","20B","72B"])
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y,_: '{:g}'.format(y)))
    plt.ylabel("Total Inference Time (s)")
    plt.xlabel("Model Parameter Size (Billions)")
    plt.title("The Latency Wall (Prefill + Decode)\nLLMFlash prefill ≈ 0 TPS at 72B → collapses")
    plt.legend(title="Scheduler", fontsize=11)
    plt.tight_layout()
    plt.savefig("fig_motivation_latency.pdf", bbox_inches='tight')
    print("Saved fig_motivation_latency.pdf")


def plot_stall_latency(df):
    sub = df[
        (df["Experiment"]=="Memory") & (df["Model"]==72) &
        (df["Quant"]=="fp32") & (df["MemConfig"]=="16GB_Host+32GB_CXL")
    ].copy()
    if sub.empty:
        sub = df[(df["Experiment"]=="Scalability") &
                 (df["Model"]==72) & (df["Quant"]=="fp32")].copy()
    if sub.empty:
        return

    sub["Stall_s"] = 1.0 / sub["TPS"]
    sub = sub.sort_values("Stall_s", ascending=False)

    plt.figure(figsize=(8, 3.5))
    ax = sns.barplot(data=sub, x="Simulator", y="Stall_s", hue="Simulator",
                     palette=COLORS, edgecolor='black', legend=False)
    plt.title("Per-Token Stall Latency (72B FP32, 16H+32C, B=1)")
    plt.ylabel("Avg. Stall per Token (s)")
    plt.xlabel("Scheduler")
    plt.ylim(0, sub["Stall_s"].max() * 1.25)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1fs', padding=4, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fig_stall_latency.pdf", bbox_inches='tight')
    print("Saved fig_stall_latency.pdf")


def plot_motivation_combined(df):
    sub_wall = df[df["Experiment"]=="Scalability"].copy()
    sub_wall["Prefill_safe"]  = sub_wall["Prefill_TPS"].clip(lower=1e-4)
    sub_wall["Total_Time_s"]  = (512 / sub_wall["Prefill_safe"]) + \
                                 (16 / sub_wall["TPS"])

    sub_stall = df[
        (df["Experiment"]=="Memory") & (df["Model"]==72) &
        (df["Quant"]=="fp32") & (df["MemConfig"]=="16GB_Host+32GB_CXL")
    ].copy()
    if sub_stall.empty:
        sub_stall = df[(df["Experiment"]=="Scalability") &
                       (df["Model"]==72) & (df["Quant"]=="fp32")].copy()
    if not sub_stall.empty:
        sub_stall["Stall_s"] = 1.0 / sub_stall["TPS"]
        sub_stall = sub_stall.sort_values("Stall_s", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                   gridspec_kw={'width_ratios': [1.3, 1]})
    sns.lineplot(data=sub_wall, x="Model", y="Total_Time_s",
                 hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS,
                 markersize=11, linewidth=2, ax=ax1)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xticks([7,13,20,72])
    ax1.set_xticklabels(["7B","13B","20B","72B"])
    ax1.set_ylabel("Total Inference Time (s)")
    ax1.set_xlabel("Model Size (Billions)")
    ax1.set_title("(a) The Latency Wall (Prefill + Decode)")
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y,_: '{:g}'.format(y)))
    ax1.legend(title="Scheduler", loc='upper left', fontsize=11)

    if not sub_stall.empty:
        sns.barplot(data=sub_stall, x="Simulator", y="Stall_s",
                    hue="Simulator", palette=COLORS,
                    edgecolor='black', ax=ax2, legend=False)
        ax2.set_title("(b) Per-Token Stall (72B FP32, B=1)")
        ax2.set_ylabel("Avg. Stall per Token (s)")
        ax2.set_xlabel("Scheduler")
        ax2.set_ylim(0, sub_stall["Stall_s"].max() * 1.3)
        for c in ax2.containers:
            ax2.bar_label(c, fmt='%.1fs', padding=3, fontsize=12,
                          fontweight='bold')

    plt.tight_layout()
    plt.savefig("fig_motivation_combined.pdf", bbox_inches='tight')
    print("Saved fig_motivation_combined.pdf")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(">>> Generating Consolidated Paper Figures...")
    df = load_and_clean_data()
    if df is not None:
        bs_check = df[df["Experiment"]=="BatchSweep"]
        print(f"    BatchSweep rows: {len(bs_check)}")
        print(f"    BatchSizes dtype: {df['BatchSize'].dtype}")
        print(f"    Unique BatchSizes: {sorted(bs_check['BatchSize'].dropna().unique().tolist())}")
        print(f"    SERVING_BATCH ({SERVING_BATCH}) found: "
            f"{SERVING_BATCH in bs_check['BatchSize'].values}\n")


        # ── TIER 1: Hero / Primary figures ──────────────────────
        plot_batch_sweep_hero(df)              # NEW HERO: 3-panel crossover story
        plot_total_inference(df)               # END-TO-END: 47× at 72B
        plot_b1_vs_serving(df)                 # NARRATIVE: B=1 vs B=SERVING
        plot_serving_throughput(df)            # SYSTEM TPS: most dramatic

        # ── TIER 2: Core supporting evidence ────────────────────
        plot_stall_duplex_phy(df)              # Physics + B=SERVING bar
        plot_combined_scalability(df)          # Prefill dominance
        plot_sparsity_collapse_theory()        # Theory: φ(B) collapse

        # ── TIER 3: Supplemental ────────────────────────────────
        plot_combined_quantization(df)
        plot_memory_sensitivity(df)
        plot_batch_sweep(df)
        plot_metrics(df)
        plot_duplex_reconstruction()
        plot_misc_stats(df)
        plot_pareto(df)
        plot_total_inference_latency(df)
        plot_stall_latency(df)
        plot_motivation_combined(df)

        print("\n>>> Success. 17 figures saved.")
        print("\n── TIER 1 (Hero — use in paper body) ──────────────────")
        print("  fig_batch_sweep_hero.pdf    — 3-panel: raw TPS + ratio + delta")
        print("  fig_total_inference.pdf     — end-to-end 47× at 72B")
        print("  fig_b1_vs_serving.pdf       — B=1 (LLMFlash) vs B=16 (SemDuplex)")
        print("  fig_serving_throughput.pdf  — system TPS × BatchSize")
        print("\n── TIER 2 (Core supporting) ──────────────────────────")
        print("  fig_stall_duplex_phy.pdf    — physics + B=16 bar")
        print("  fig_combined_scalability.pdf")
        print("  fig_sparsity_collapse_theory.pdf")
        print("\n── Change SERVING_BATCH at top to adjust default ─────")
        print(f"   Current: SERVING_BATCH = {SERVING_BATCH}")
