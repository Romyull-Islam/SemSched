"""
plot_paper_figures.py — primary paper-figure generator.

Input:
    final_results_with_coldload.csv  (produced by run_experiments.py)

Output figures (all in figures/):
    fig_total_inference.pdf          — end-to-end latency wall
    fig_motivation_latency.pdf       — Fig 1(a) latency wall (motivation panel)
    fig_motivation_combined.pdf      — Fig 1 combined motivation
    fig_stall_duplex_phy.pdf         — Fig 8 (a)(b)(c) write-stall, FP32 batch sweep
    fig_combined_scalability.pdf     — Fig 4 (a) decode + (b) prefill, 32H+64C
    fig_combined_quantization.pdf    — Fig 6 (a) decode + (b) speedup, 32H+32C
    fig_memory.pdf                   — Fig 7 (a) INT4 + (b) FP32 memory sensitivity
    fig_batch_sweep.pdf              — Fig 5 batch sweep, 16H+32C
    fig_sparsity_collapse_theory.pdf — illustrative sparsity-collapse plot (no CSV)
    fig_duplex_phy.pdf               — illustrative duplex schematic (no CSV)
    fig_metrics.pdf                  — supporting per-token latency bars
    fig_misc_stats.pdf               — supporting bandwidth utilization
    fig_pareto.pdf                   — supporting Pareto plot
    fig_stall_latency.pdf            — supporting stall vs latency
    scaling_trends_dynamic.pdf       — batch-sweep line plot across all precisions

Sibling scripts (separate purposes — do NOT overwrite these PDFs):
    cache_ablation_study.py — generates Fig 1(b) fig_motivation_comprehensive.pdf
    plot_timeline.py        — generates supplementary fig_duplex_timeline.pdf
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator
import os

# Force TrueType font embedding (Required for IEEE/ACM PDF compliance)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---------------------------
# Global style and constants
# ---------------------------
# Use a robust seaborn style name that exists across Matplotlib versions
available_styles = plt.style.available
if 'seaborn-v0_8-paper' in available_styles:
    plt.style.use('seaborn-v0_8-paper')
elif 'seaborn-paper' in available_styles:
    plt.style.use('seaborn-paper')
elif 'seaborn-v0_8' in available_styles:
    plt.style.use('seaborn-v0_8')
else:
    plt.style.use('seaborn')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
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

# Changed key to SemSched
COLORS  = {'FlexGen': '#7f7f7f', 'LIA': '#377eb8', 'LLMFlash': '#ff7f0e', 'SemSched': '#e41a1c'}
MARKERS = {'FlexGen': 'o',       'LIA': 's',        'LLMFlash': '^',        'SemSched': 'D'}

SERVING_BATCH = 128  # desired serving batch for paper figures

# ---------------------------
# Data loading & cleaning
# ---------------------------
def load_and_clean_data():
    files = ["final_results_with_coldload.csv", "final_results_all_combinations.csv"]
    df = None
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"Loaded data from {f}")
            break
    if df is None:
        print("Error: No data file found.")
        return None

    name_map = {
        'flexgen_baseline.py':   'FlexGen',
        'lia_baseline.py':       'LIA',
        'llmflash_baseline.py':  'LLMFlash',
        'semduplex_scheduler.py':'SemSched', # Changed to SemSched
        'flashllm_baseline.py':  'LLMFlash',
        'FlashLLM':              'LLMFlash',
        'SemDuplex':             'SemSched', # Changed to SemSched
    }
    df['Simulator'] = df['Simulator'].replace(name_map)

    memconfig_map = {
        '16H32C': '16GB_Host+32GB_CXL',
        '16H64C': '16GB_Host+64GB_CXL',
        '32H32C': '32GB_Host+32GB_CXL',
        '32H64C': '32GB_Host+64GB_CXL',
    }
    if 'MemConfig' in df.columns:
        df['MemConfig'] = df['MemConfig'].replace(memconfig_map)

    if 'BatchSize' in df.columns:
        df['BatchSize'] = pd.to_numeric(df['BatchSize'], errors='coerce').astype('Int64')

    if 'HitRate' in df.columns and df['HitRate'].max() <= 1.0:
        df['HitRate'] = df['HitRate'] * 100.0

    return df

# ---------------------------
# 1. Total inference figure
# ---------------------------
def plot_total_inference(df):
    sub = df[df["Experiment"] == "Scalability"].copy()
    if sub.empty:
        return
    sub = sub[sub["Prefill_TPS"] > 0].copy()
    if sub.empty:
        return

    # Use a consistent batch
    if "BatchSize" in sub.columns:
        batches = sub["BatchSize"].dropna().unique()
        b_use = SERVING_BATCH if SERVING_BATCH in batches else batches.min()
        sub = sub[sub["BatchSize"] == b_use].copy()
        print(f"[TotalInference] Using B={b_use} (available: {batches})")
    else:
        b_use = None

    sub["Totals"] = 512 / sub["Prefill_TPS"] + 16 / sub["TPS"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    sns.lineplot(data=sub, x="Model", y="Totals", hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS, markersize=12, linewidth=2.5, ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([7, 13, 20, 72])
    ax1.set_xticklabels(['7B', '13B', '20B', '72B'])
    title_suffix = f"(B={b_use})" if b_use is not None else ""
    ax1.set_ylabel("Total Inference Time (s)\n512-tok prefill + 16-tok decode", fontsize=13)
    ax1.set_xlabel("Model Size (Parameters)")
    ax1.set_title(f"(a) End-to-End Latency Wall\n{title_suffix}", fontweight='bold')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}'))
    ax1.legend(title="Scheduler", loc='upper left', fontsize=12)
    ax1.grid(True, which='both', ls='--', alpha=0.3)

    llm72 = sub[(sub["Simulator"] == "LLMFlash") & (sub["Model"] == 72)]
    sem72 = sub[(sub["Simulator"] == "SemSched") & (sub["Model"] == 72)] # Changed name
    if not llm72.empty and not sem72.empty:
        llmt  = llm72["Totals"].values[0]
        semt  = sem72["Totals"].values[0]
        ratio = llmt / semt
        ax1.annotate(f'LLMFlash stall\n{ratio:.0f}× slower',
                     xy=(72, llmt), xytext=(20, llmt * 0.3), fontsize=10,
                     color=COLORS['LLMFlash'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['LLMFlash'], lw=1.5),
                     bbox=dict(boxstyle='round', fc='white', ec=COLORS['LLMFlash'], alpha=0.8))

    # Safe speedup calc (avoid duplicate index)
    base_df = sub[sub["Simulator"] == "FlexGen"][["Model", "Totals"]].rename(columns={"Totals": "BaseTotals"})
    base_df = base_df.groupby("Model", as_index=False)["BaseTotals"].mean()
    subspeed = sub[sub["Simulator"] != "FlexGen"].copy()
    subspeed = subspeed.merge(base_df, on="Model", how="left")
    subspeed["Speeduptotal"] = subspeed["BaseTotals"] / subspeed["Totals"]

    model_order = [7, 13, 20, 72]
    sns.barplot(data=subspeed, x="Model", y="Speeduptotal", hue="Simulator",
                order=model_order, palette=COLORS, edgecolor='black', ax=ax2)
    ax2.set_title("(b) End-to-End Speedup vs FlexGen\n(Higher is Better)", fontweight='bold')
    ax2.set_ylabel("Total Speedup ×", fontsize=14)
    ax2.set_xlabel("Model Size (Billions)")
    ax2.xaxis.set_major_locator(FixedLocator(ax2.get_xticks()))
    ax2.set_xticklabels(['7B', '13B', '20B', '72B'])
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_ylim(0, subspeed["Speeduptotal"].max() * 1.35)
    for c in ax2.containers:
        ax2.bar_label(c, fmt='.1f', padding=3, fontsize=10, rotation=45)

    semspeedup72 = subspeed[(subspeed["Simulator"] == "SemSched") & # Changed name
                             (subspeed["Model"] == 72)]["Speeduptotal"]
    if not semspeedup72.empty:
        ax2.annotate(f'SemSched\n{semspeedup72.values[0]:.0f}× speedup\n(prefill-driven)', # Changed name
                     xy=(3, semspeedup72.values[0]),
                     xytext=(1.5, semspeedup72.values[0] + 0.6), fontsize=10,
                     color=COLORS['SemSched'], # Updated key
                     arrowprops=dict(arrowstyle='->', color=COLORS['SemSched'], lw=1.5), # Updated key
                     bbox=dict(boxstyle='round', fc='white', ec=COLORS['SemSched'], alpha=0.8)) # Updated key

    plt.tight_layout()
    plt.savefig("figures/fig_total_inference.pdf", bbox_inches='tight')
    print("Saved figures/fig_total_inference.pdf  [PRIMARY PAPER FIGURE]")


# ==========================================
# 2. STALL DUPLEX PHY — FP32, B=1..128
# ==========================================
def plot_stall_duplex_phy(df):

    fig = plt.figure(figsize=(16, 3.8))
    gs  = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.3], height_ratios=[1, 1],
                            hspace=0.19)
    ax1 = fig.add_subplot(gs[:, 0])

    # ── Shared sweep data (used by all 3 panels) ──────────────────────────
    sweep = df[
        (df["Experiment"] == "BatchSweep") &
        (df["Model"]      == 72) &
        (df["Quant"]      == "fp32")
    ].copy()

    if "MemConfig" in sweep.columns and "16GB_Host+32GB_CXL" in sweep["MemConfig"].values:
        sweep = sweep[sweep["MemConfig"] == "16GB_Host+32GB_CXL"].copy()

    sweep["BatchSize"] = pd.to_numeric(sweep["BatchSize"], errors="coerce")
    sweep = sweep[sweep["BatchSize"] <= 128].copy()
    sweep = sweep.sort_values("BatchSize")

    if "Write_Stall_Time_s" in sweep.columns:
        sweep["stall_ms"] = sweep["Write_Stall_Time_s"] * 1000.0
    elif "WriteStall_s" in sweep.columns:
        sweep["stall_ms"] = sweep["WriteStall_s"] * 1000.0
    else:
        sweep["token_time_s"] = sweep["BatchSize"] / sweep["TPS"]
        sweep["stall_ms"]     = (sweep.get("Write_Stall_Pct", 0) / 100.0) \
                                  * sweep["token_time_s"] * 1000.0

    sweep["token_time_ms"] = sweep["BatchSize"] / sweep["TPS"] * 1000.0
    sweep["stall_pct"]     = (sweep["stall_ms"] / sweep["token_time_ms"] * 100.0).fillna(0)

    if sweep.empty:
        print("[StallDuplex] No BatchSweep FP32 72B data found. Skipping.")
        return

    batch_order = sorted(sweep["BatchSize"].dropna().unique().tolist())
    sims_order  = ["FlexGen", "LIA", "LLMFlash", "SemSched"] # Changed name

    # ── Panel (a): Write Stall bar at B=SERVING_BATCH ─────────────────────
    bar_data = sweep[sweep["BatchSize"] == SERVING_BATCH].copy()

    if bar_data.empty:
        # fallback to max available batch
        max_b    = sweep["BatchSize"].max()
        bar_data = sweep[sweep["BatchSize"] == max_b].copy()
        print(f"[StallDuplex] B={SERVING_BATCH} not in sweep, fallback B={max_b}")
    else:
        print(f"[StallDuplex] Panel (a) using B={SERVING_BATCH} ✓")

    bar_data = bar_data.sort_values("stall_ms", ascending=False)
    b_label  = int(bar_data["BatchSize"].iloc[0])

    sns.barplot(x='Simulator', y='stall_ms', data=bar_data, hue='Simulator',
                palette=COLORS, edgecolor='black', legend=False, ax=ax1,
                order=sims_order)
    ax1.set_title(f"(a) Write Stall / Token (72B FP32, B={b_label}, 16H+32C)",
                  fontsize=16)
    ax1.set_ylabel("Write Stall per Token (ms)", fontsize=14)
    ax1.set_ylim(0, bar_data["stall_ms"].max() * 1.35)
    ax1.set_xlabel("Scheduler")
    for p in ax1.patches:
        if p.get_height() > 0:
            ax1.annotate(f'{p.get_height():.2f}ms',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10),
                         textcoords='offset points', fontsize=12, fontweight='bold')

    # ── Panels (b) & (c): Stall vs batch ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

    for sim in sims_order:
        s = sweep[sweep["Simulator"] == sim].sort_values("BatchSize")
        if s.empty:
            continue
        ax2.plot(s["BatchSize"], s["stall_ms"],
                 marker=MARKERS[sim], color=COLORS[sim], label=sim,
                 linewidth=2.5, markersize=9)
        ax3.plot(s["BatchSize"], s["stall_pct"],
                 marker=MARKERS[sim], color=COLORS[sim], label=sim,
                 linewidth=2.5, markersize=9)

    # ── Axis styling ──────────────────────────────────────────────────────
    ax2.set_title("(b) Per-Token Write Stall Time (72B FP32, 16H+32C)",
                  fontsize=16)
    ax2.set_ylabel("Write Stall / Token (ms)", fontsize=14)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(batch_order)
    ax2.set_xticklabels(batch_order)
    ax2.legend(loc='upper left', fontsize=12, frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    

    ax3.set_title("(c) Write Stall % of Token Time (72B FP32, 16H+32C)",
                  fontsize=16)
    ax3.set_ylabel("Stall / Token Time (%)", fontsize=14)
    ax3.set_xlabel("Batch Size", fontsize=14)
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(batch_order)
    ax3.set_xticklabels(batch_order)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax3.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.savefig("figures/fig_stall_duplex_phy.pdf", bbox_inches='tight')
    print("Saved figures/fig_stall_duplex_phy.pdf  [FP32, B=1–128, CSV-driven]")





# ==========================================
# 3. COMBINED SCALABILITY — STRICT B=128
# ==========================================
def plot_combined_scalability(df):
    sub = df[df["Experiment"] == "Scalability"].copy()
    if sub.empty:
        print("ERROR: No Scalability data")
        return

    available_batches = sub["BatchSize"].dropna().unique()
    print(f"[Scalability] Available batches: {sorted(available_batches)}")

    if SERVING_BATCH in available_batches:
        sub   = sub[sub["BatchSize"] == SERVING_BATCH].copy()
        b_use = SERVING_BATCH
        print(f"[Scalability] Using B={b_use} ✓")
    else:
        print(f"WARNING: B={SERVING_BATCH} not found in Scalability. Available: {sorted(available_batches)}")
        print("Stopping — please add B=128 Scalability runs to your CSV.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 3.5))
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7.5))

    # ── Line plot ──────────────────────────────────────────────
    sns.lineplot(data=sub, x="Model", y="TPS", hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS, markersize=12, ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([7, 13, 20, 72])
    ax1.set_xticklabels(['7B', '13B', '20B', '72B'])
    ax1.set_xlabel("Model Size", fontsize=14)
    ax1.set_ylabel("Decode Throughput (Tokens/s)", fontsize=14)
    ax1.set_title(f"(a) Decode Scalability, 32H+64C, (B={b_use})", fontsize=16)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax1.legend(loc='upper right', fontsize=12, frameon=True)
    ax1.margins(y=0.2)

    # ── Add non-overlapping value labels on line plot ──────────
    MODELS = sorted(sub["Model"].unique())
    SIMS   = ['FlexGen', 'LIA', 'LLMFlash', 'SemSched'] # Updated list

    #OFFSET_LADDER = [-18, -15, +13, +27]
    OFFSET_LADDER = [(-4, -16), (15, -15), (15, 10), (0, 16)]

    for model in MODELS:
        model_data = sub[sub["Model"] == model].copy()
        model_data = model_data.sort_values("TPS")
        ranked_sims = model_data["Simulator"].tolist()

        for rank, sim in enumerate(ranked_sims):
            row = model_data[model_data["Simulator"] == sim]
            if row.empty:
                continue
            tps   = row["TPS"].values[0]
            color = COLORS.get(sim, 'black')
            xoff, yoff = OFFSET_LADDER[rank]

            ax1.annotate(
                f'{tps:.1f}',
                xy=(model, tps),
                xytext=(xoff, yoff),
                textcoords='offset points',
                ha='center',
                va='center',
                fontsize=8.5,
                color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec=color, alpha=0.75, lw=0.8),
            )

    # ── Bar plot ───────────────────────────────────────────────
    sns.barplot(data=sub, x="Model", y="Prefill_TPS", hue="Simulator",
                palette=COLORS, edgecolor='black', ax=ax2)
    ax2.set_yscale('log')
    ax2.set_xticklabels(['7B', '13B', '20B', '72B'])
    ax2.set_ylabel("Prefill Throughput (Tokens/s)", fontsize=14)
    ax2.set_title(f"(b) Prefill Scalability, 32H+64C, (B={b_use})", fontsize=16)
    ax2.set_xlabel("Model Size", fontsize=14)
    ax2.legend(loc='upper right', fontsize=12, frameon=True, ncol=2)
    ax2.margins(y=0.2)
    for c in ax2.containers:
        labels = [f'{v:.0f}' if v > 10 else f'{v:.2f}' if v > 0.01 else '<0.1'
                  for v in c.datavalues]
        ax2.bar_label(c, labels=labels, padding=3, fontsize=11, rotation=0)

    plt.tight_layout()
    plt.savefig("figures/fig_combined_scalability.pdf", bbox_inches='tight')
    print(f"Saved figures/fig_combined_scalability.pdf  [B={b_use}]")



# ---------------------------
# 4. Combined quantization
# ---------------------------
def plot_combined_quantization(df):
    quant_df = df[(df["Experiment"] == "Quantization") & (df["Model"] == 72)].copy()
    if quant_df.empty:
        return

    available_batches = sorted(quant_df["BatchSize"].dropna().unique().tolist())
    b_use = SERVING_BATCH if SERVING_BATCH in available_batches else available_batches[0]
    sub   = quant_df[quant_df["BatchSize"] == b_use].copy()
    print(f"[Quantization] Using B={b_use} (available: {available_batches})")

    base_sim  = 'LLMFlash' if 'LLMFlash' in sub["Simulator"].values else 'FlexGen'
    base_data = sub[sub["Simulator"] == base_sim][["Quant", "TPS"]].rename(columns={"TPS": "BaseTPS"})
    sub       = sub.merge(base_data, on="Quant", how="left")
    sub["Speedup"] = (sub["TPS"] / sub["BaseTPS"]).replace(0, float('nan'))

    quant_order = [q for q in ["fp32", "fp16", "int8", "int4"] if q in sub["Quant"].values]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 3.6))

    sns.barplot(data=sub, x="Quant", y="TPS", hue="Simulator",
                order=quant_order, palette=COLORS, edgecolor='black', ax=ax1)
    ax1.set_title(f"(a) Decode Throughput, 72B, 32H+32C, B={b_use}", fontsize=16)
    ax1.set_ylabel("Tokens/s", fontsize=14)
    ax1.set_xlabel("Quantization", fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_ylim(0, sub["TPS"].max() * 1.4)
    for c in ax1.containers:
        labels = [f'{v:.2f}' for v in c.datavalues]
        ax1.bar_label(c, labels=labels, padding=3, fontsize=14, rotation=45)

    subsp = sub[sub["Simulator"] != base_sim]
    if not subsp.empty:
        sns.barplot(data=subsp, x="Quant", y="Speedup", hue="Simulator",
                    order=quant_order, palette=COLORS, edgecolor='black', ax=ax2)
        ax2.set_title(f"(b) Speedup vs {base_sim}, 72B, 32H+32C, B={b_use}", fontsize=16)
        ax2.set_ylabel("Speedup (x)", fontsize=14)
        ax2.set_xlabel("Quantization", fontsize=14)
        ax2.set_ylim(0, subsp["Speedup"].max() * 1.35)
        if ax2.get_legend():
            ax2.get_legend().remove()
        for i, c in enumerate(ax2.containers):
            labels = [f'{v:.2f}' for v in c.datavalues]
            ax2.bar_label(c, labels=labels, padding=3, fontsize=14, rotation=45)
    else:
        ax2.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                 transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig("figures/fig_combined_quantization.pdf", bbox_inches='tight')
    print(f"Saved figures/fig_combined_quantization.pdf  [B={b_use}]")

# ==========================================
# 5. MEMORY SENSITIVITY — STRICTLY 2 SUBPLOTS (INT4 + FP32), B=128
# ==========================================
def plot_memory_sensitivity(df):
    suball = df[
        (df["Experiment"] == "Memory") &
        (df["Model"]      == 72) &
        (df["BatchSize"]  == SERVING_BATCH) 
    ].copy()

    if suball.empty:
        print(f"[Memory] No B={SERVING_BATCH} Memory data at 72B.")
        return

    PAPER_QUANTS = ["int4", "fp32"]
    quants_available = [q for q in PAPER_QUANTS if q in suball["Quant"].values]

    if not quants_available:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 3.9))

    for col_idx, q in enumerate(PAPER_QUANTS):
        ax  = axes[col_idx]
        sub = suball[suball["Quant"] == q].copy()

        if sub.empty:
            ax.text(0.5, 0.5, f"No {q.upper()} data\nat B={SERVING_BATCH}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue

        mem_order = ["16GB_Host+32GB_CXL", "16GB_Host+64GB_CXL",
                     "32GB_Host+32GB_CXL", "32GB_Host+64GB_CXL"]
        mem_order = [m for m in mem_order if m in sub["MemConfig"].values]

        sns.barplot(data=sub, x="MemConfig", y="TPS", hue="Simulator",
                    order=mem_order, palette=COLORS, edgecolor='black', ax=ax)
        panel_tag = "(a)" if col_idx == 0 else "(b)"
        ax.set_title(f"{panel_tag} Memory Configuration Sensitivity (72B, {q.upper()}, B={SERVING_BATCH})", fontsize=16)
        ax.set_ylabel("Throughput (Tokens/s)", fontsize=14)
        ax.set_xlabel("Memory Config", fontsize=14)
        ax.xaxis.set_major_locator(FixedLocator(range(len(mem_order))))
        ax.set_xticklabels(
            [m.replace("GB_Host+", "H+").replace("GB_CXL", "C")
             for m in mem_order],
            rotation=0, ha='center', fontsize=14
        )
        ax.set_ylim(0, sub["TPS"].max() * 1.45)
        #ax.legend(loc='upper left', fontsize=14, ncol=2)
        if col_idx == 0: 
            # Remove seaborn's auto-generated legend for the left plot
            if ax.get_legend():
                ax.get_legend().remove()
        else:
            # Keep and format the legend for the right plot
            ax.legend(loc='upper left', fontsize=14, ncol=2)

        for i, c in enumerate(ax.containers):
            labels = [f'{v:.1f}' for v in c.datavalues]
            ax.bar_label(c, labels=labels, padding=3, fontsize=11,
                          rotation=0)

    plt.tight_layout()
    plt.savefig("figures/fig_memory.pdf", bbox_inches='tight')
    print(f"Saved figures/fig_memory.pdf  [INT4+FP32 only, B={SERVING_BATCH}]")


# ==========================================
# 5b. MEMORY SENSITIVITY (all 4 quants) — separate combined figure
# ==========================================
def plot_memory_sensitivity_int8_fp16(df):
    suball = df[
        (df["Experiment"] == "Memory") &
        (df["Model"]      == 72) &
        (df["BatchSize"]  == SERVING_BATCH)
    ].copy()

    if suball.empty:
        print(f"[Memory ALL] No B={SERVING_BATCH} Memory data at 72B.")
        return

    PAPER_QUANTS = ["fp32", "fp16", "int8", "int4"]
    quants_available = [q for q in PAPER_QUANTS if q in suball["Quant"].values]
    if not quants_available:
        print("[Memory ALL] No quant rows in data.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(24, 4.15))
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    for idx, q in enumerate(PAPER_QUANTS):
        ax  = axes[idx]
        sub = suball[suball["Quant"] == q].copy()

        if sub.empty:
            ax.text(0.5, 0.5, f"No {q.upper()} data\nat B={SERVING_BATCH}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=16)
            continue

        mem_order = ["16GB_Host+32GB_CXL", "16GB_Host+64GB_CXL",
                     "32GB_Host+32GB_CXL", "32GB_Host+64GB_CXL"]
        mem_order = [m for m in mem_order if m in sub["MemConfig"].values]

        sns.barplot(data=sub, x="MemConfig", y="TPS", hue="Simulator",
                    order=mem_order, palette=COLORS, edgecolor='black', ax=ax)
        ax.set_title(f"{panel_tags[idx]} 72B, {q.upper()}, B={SERVING_BATCH}", fontsize=17)
        ax.set_ylabel("Throughput (Tokens/s)", fontsize=16)
        ax.set_xlabel("Memory Config", fontsize=16)
        ax.xaxis.set_major_locator(FixedLocator(range(len(mem_order))))
        ax.set_xticklabels(
            [m.replace("GB_Host+", "H+").replace("GB_CXL", "C")
             for m in mem_order],
            rotation=0, ha='center', fontsize=15
        )
        ax.set_ylim(0, sub["TPS"].max() * 1.45)

        if idx == 1:
            ax.legend(loc='upper left', fontsize=14, ncol=1)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

        for i, ct in enumerate(ax.containers):
            labels = [f'{v:.2f}' for v in ct.datavalues]
            ax.bar_label(ct, labels=labels, padding=1, fontsize=14, rotation=58)

    plt.tight_layout()
    plt.savefig("figures/fig_memory_all.pdf", bbox_inches='tight')
    print(f"Saved figures/fig_memory_all.pdf  [all 4 quants, B={SERVING_BATCH}]")


def plot_batch_sweep(df):
    sub = df[df["Experiment"] == "BatchSweep"].copy()
    if sub.empty:
        return
    sub["BatchSize"] = pd.to_numeric(sub["BatchSize"])
    sub = sub.sort_values("BatchSize")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

    sns.lineplot(data=sub, x="BatchSize", y="TPS", hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS, markersize=12, linewidth=2.5, ax=ax1)
    ax1.set_title("(a) Batch Throughput\n72B INT8 16H+32C (tight memory)",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Throughput (Tokens/s)", fontsize=14)
    ax1.set_xlabel("Batch Size", fontsize=14)
    ax1.set_xticks([1, 4, 8, 16, 32, 64])
    ax1.axvline(x=4, color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7)
    ax1.legend(loc='upper left', fontsize=12, frameon=True)
    ax1.grid(True, alpha=0.3)

    llm_tps = sub[sub["Simulator"] == "LLMFlash"][["BatchSize", "TPS"]].rename(columns={"TPS": "LLMFlashTPS"})
    sem_tps = sub[sub["Simulator"] == "SemSched"][["BatchSize", "TPS"]].rename(columns={"TPS": "SemSchedTPS"}) # Updated name
    lia_tps = sub[sub["Simulator"] == "LIA"][["BatchSize", "TPS"]].rename(columns={"TPS": "LIATPS"})

    if not llm_tps.empty and not sem_tps.empty:
        ratio_df = pd.merge(llm_tps, sem_tps, on="BatchSize")
        ratio_df = pd.merge(ratio_df, lia_tps, on="BatchSize", how="left")
        ratio_df["SemSched/LLMFlash"] = ratio_df["SemSchedTPS"] / ratio_df["LLMFlashTPS"] # Updated ratio
        ratio_df["LIA/LLMFlash"]       = ratio_df["LIATPS"]       / ratio_df["LLMFlashTPS"]

        ax2.plot(ratio_df["BatchSize"], ratio_df["SemSched/LLMFlash"], # Updated name
                 color=COLORS['SemSched'], marker='D', markersize=10, # Updated color key
                 linewidth=2.5, label="SemSched / LLMFlash")
        ax2.plot(ratio_df["BatchSize"], ratio_df["LIA/LLMFlash"],
                 color=COLORS['LIA'], marker='s', markersize=10,
                 linewidth=2.5, linestyle='--', label="LIA / LLMFlash")
        ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label="Parity ratio=1")
        ax2.axvline(x=4, color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7)

        crossvals = ratio_df[ratio_df["SemSched/LLMFlash"] > 1.0]["BatchSize"] # Updated check
        if not crossvals.empty:
            ax2.axvspan(crossvals.min(), ratio_df["BatchSize"].max(),
                        alpha=0.08, color='red', label="SemSched dominant") # Updated label

        ax2.set_title("(b) Relative Efficiency vs LLMFlash\nSemSched advantage at B≥4", # Updated title
                      fontsize=13, fontweight='bold')
        ax2.set_ylabel("TPS Ratio (>1 = SemSched better)", fontsize=13)
        ax2.set_xlabel("Batch Size", fontsize=14)
        ax2.set_xticks([1, 4, 8, 16, 32, 64])
        ax2.legend(loc='upper left', fontsize=11, frameon=True)

        max_row = ratio_df.loc[ratio_df["SemSched/LLMFlash"].idxmax()]
        ax2.annotate(f'Peak {max_row["SemSched/LLMFlash"]:.2f}×\nB={int(max_row["BatchSize"])}',
                     xy=(max_row["BatchSize"], max_row["SemSched/LLMFlash"]),
                     xytext=(max_row["BatchSize"] - 20, max_row["SemSched/LLMFlash"] + 0.05),
                     fontsize=11, color=COLORS['SemSched'], # Updated color key
                     arrowprops=dict(arrowstyle='->', color=COLORS['SemSched'], lw=1.5)) # Updated key
    else:
        ax2.text(0.5, 0.5, "Need both LLMFlash and SemSched data", ha='center', va='center')

    plt.tight_layout()
    plt.savefig("figures/fig_batch_sweep.pdf", bbox_inches='tight')
    print("Saved figures/fig_batch_sweep.pdf")


def plot_sparsity_collapse_theory():
    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
    configs = {
        'OPT-ReLU (p=0.10)':   (0.10, '#1a9850', '--'),
        'Qwen-SiLU (p=0.46)':  (0.46, '#ff7f0e', '-'),
        'Llama-SiLU (p=0.40)': (0.40, '#d73027', '-.'),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

    for label, (p, color, ls) in configs.items():
        active = 1.0 - (1.0 - p) ** batch_sizes
        ax1.plot(batch_sizes, active * 100, color=color, linestyle=ls,
                 linewidth=2.5, marker='o', markersize=8, label=label)
    ax1.axhline(100, color='black', linestyle=':', linewidth=1.2, label="Full load (100%)")
    ax1.axvline(4,   color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7,
                label="Crossover B=4 (SiLU)")
    ax1.set_xlabel("Batch Size", fontsize=14)
    ax1.set_ylabel("Active FFN Fraction (%)", fontsize=14)
    ax1.set_title(r"(a) Sparsity Collapse: $B \mapsto 1-(1-p)^B$", fontsize=14)
    ax1.set_xticks(batch_sizes)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3)

    total_ffn_gb    = 48.0
    dram_remaining  = 24.0
    p_silu          = 0.46
    active_fracs    = 1.0 - (1.0 - p_silu) ** batch_sizes
    dram_window     = np.minimum(active_fracs, dram_remaining / total_ffn_gb)
    nand_load       = np.maximum(0.0, active_fracs - dram_window) * total_ffn_gb

    ax2.fill_between(batch_sizes, 0, nand_load, color='#d73027', alpha=0.4, label="NAND overflow (GB/step)")
    ax2.plot(batch_sizes, nand_load, color='#d73027', linewidth=2.5, marker='D', markersize=10)
    ax2.axvline(4, color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7)
    ax2.set_xlabel("Batch Size", fontsize=14)
    ax2.set_ylabel("NAND Load per Decode Step (GB)", fontsize=13)
    ax2.set_title("(b) NAND Overflow at 16H+32C\n72B INT8, SiLU p=0.46", fontsize=14)
    ax2.set_xticks(batch_sizes)
    ax2.set_ylim(0, total_ffn_gb * 1.1)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig_sparsity_collapse_theory.pdf", bbox_inches='tight')
    print("Saved figures/fig_sparsity_collapse_theory.pdf")


def plot_metrics(df):
    sub = df[(df["Experiment"] == "Scalability") & (df["Model"] == 72)].copy()
    if sub.empty:
        return

    sub["Latency_ms"] = 1000.0 / sub["TPS"]
    y_col, title, ylabel, fmt_str = "Latency_ms", "Avg Token Latency (1/TPS)", "Latency (ms)", ".0f"

    fig, ax_metric = plt.subplots(1, 1, figsize=(8, 5))

    sns.barplot(data=sub, x="Simulator", y=y_col, hue="Simulator",
                palette=COLORS, ax=ax_metric, legend=False)
    ax_metric.set_title(f"(a) {title}")
    ax_metric.set_ylabel(ylabel)
    for c in ax_metric.containers:
        ax_metric.bar_label(c, fmt=fmt_str, padding=3)

    plt.tight_layout()
    plt.savefig("figures/fig_metrics.pdf", bbox_inches='tight')
    print("Saved figures/fig_metrics.pdf")


def plot_duplex_reconstruction():
    time_ms   = np.linspace(0, 100, 1000)
    base_read = np.zeros_like(time_ms)
    base_read[0:800] = 32
    base_write = np.zeros_like(time_ms)
    base_write[850:950] = 32
    sem_read  = np.full_like(time_ms, 32)
    sem_write = np.zeros_like(time_ms)
    for i in range(100, 1000, 200):
        sem_write[i:i+50] = 32

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
    ax1.fill_between(time_ms, 0,  base_read,  color='#fdae61', alpha=0.9, label="Read Weights")
    ax1.fill_between(time_ms, 0, -base_write, color='#999999', alpha=0.9, label="Write KV-Cache")
    ax1.set_title("(a) Baseline Architecture", fontsize=15)
    ax1.set_ylabel("BW (GB/s)", fontsize=14)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.legend(fontsize=11)

    ax2.fill_between(time_ms, 0,  sem_read,  color='#e41a1c', alpha=0.7, label="Read Lane")
    ax2.fill_between(time_ms, 0, -sem_write, color='#377eb8', alpha=0.8, label="Write Injection")
    ax2.set_title("(b) SemSched Architecture", fontsize=14) # Changed name
    ax2.set_ylabel("BW (GB/s)", fontsize=14)
    ax2.set_xlabel("Time (microseconds)", fontsize=14)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("figures/fig_duplex_phy.pdf", bbox_inches='tight')
    print("Saved figures/fig_duplex_phy.pdf")


def plot_misc_stats(df):
    sub_cold  = df[(df["Experiment"] == "Quantization") & (df["Simulator"] == "SemSched")] # Updated name
    
    if sub_cold.empty:
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    quant_order = [q for q in ["fp32", "fp16", "int8", "int4"]
                   if q in sub_cold["Quant"].values]
    sns.barplot(data=sub_cold, x="Quant", y="Cold_Load_s",
                order=quant_order, color='#9467bd', edgecolor='black', ax=ax1)
    ax1.set_title("System Cold Boot Time (72B)")
    ax1.set_ylabel("Seconds")
    for c in ax1.containers:
        ax1.bar_label(c, fmt='.1f', padding=3)

    plt.tight_layout()
    plt.savefig("figures/fig_misc_stats.pdf", bbox_inches='tight')
    print("Saved figures/fig_misc_stats.pdf")


def plot_pareto(df):
    sub = df[df["Experiment"] == "Scalability"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.scatterplot(data=sub, x="Prefill_TPS", y="TPS", hue="Simulator",
                    size="Model", sizes=(50, 500), palette=COLORS, ax=ax, alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Prefill Speed (Tokens/s)")
    ax.set_ylabel("Decode Speed (Tokens/s)")
    ax.set_title("Efficiency Frontier\n(SemSched: top-right corner = optimal)") # Updated title
    plt.tight_layout()
    plt.savefig("figures/fig_pareto.pdf", bbox_inches='tight')
    print("Saved figures/fig_pareto.pdf")


def plot_total_inference_latency(df):
    sub = df[df["Experiment"] == "Scalability"].copy()
    sub = sub[sub["Prefill_TPS"] > 0].copy()
    if sub.empty:
        return
    sub["TotalTimes"] = 512 / sub["Prefill_TPS"] + 16 / sub["TPS"]

    plt.figure(figsize=(8, 3))
    sns.lineplot(data=sub, x="Model", y="TotalTimes", hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS, markersize=11, linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([7, 13, 20, 72], ['7B', '13B', '20B', '72B'], fontsize=14)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}'))
    plt.ylabel("Total Inference Time (s)", fontsize=14)
    plt.xlabel("Model Parameter Size (Billions)", fontsize=14)
    plt.title("The Latency Wall (Prefill + Decode)", fontsize=15)
    plt.grid(True, which='major', ls='-', alpha=0.2)
    plt.tight_layout()
    plt.savefig("figures/fig_motivation_latency.pdf", bbox_inches='tight')
    print("Saved figures/fig_motivation_latency.pdf")


def plot_stall_latency(df):
    sub = df[(df["Experiment"] == "Memory") & (df["Model"] == 72) &
             (df["Quant"] == "fp32") & (df["MemConfig"] == "16GB_Host+32GB_CXL")].copy()
    if sub.empty:
        sub = df[(df["Experiment"] == "Scalability") &
                 (df["Model"] == 72) & (df["Quant"] == "fp32")].copy()
    if sub.empty:
        return
    sub["Stalls"] = 1.0 / sub["TPS"]
    sub = sub.sort_values(by="Stalls", ascending=False)

    plt.figure(figsize=(8, 3))
    ax = sns.barplot(data=sub, x="Simulator", y="Stalls", hue="Simulator",
                     palette=COLORS, edgecolor='black', legend=False)
    plt.title("Per-Token Stall Latency\n72B FP32, 16H+32C", fontsize=16)
    plt.ylabel("Avg. Stall Time per Token (s)", fontsize=16)
    plt.xlabel("Scheduler", fontsize=14)
    plt.ylim(0, sub["Stalls"].max() * 1.25)
    for c in ax.containers:
        ax.bar_label(c, fmt='.1f', padding=4, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/fig_stall_latency.pdf", bbox_inches='tight')
    print("Saved figures/fig_stall_latency.pdf")


def plot_motivation_combined(df):
    subwall = df[df["Experiment"] == "Scalability"].copy()
    subwall = subwall[subwall["Prefill_TPS"] > 0].copy()
    if subwall.empty:
        return
    subwall["TotalTimes"] = 512 / subwall["Prefill_TPS"] + 16 / subwall["TPS"]

    substall = df[(df["Experiment"] == "Memory") & (df["Model"] == 72) &
                  (df["Quant"] == "fp32") & (df["MemConfig"] == "16GB_Host+32GB_CXL")].copy()
    if substall.empty:
        substall = df[(df["Experiment"] == "Scalability") &
                      (df["Model"] == 72) & (df["Quant"] == "fp32")].copy()
    if not substall.empty:
        substall["Stalls"] = 1.0 / substall["TPS"]
        substall = substall.sort_values(by="Stalls", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                   gridspec_kw={'width_ratios': [1.3, 1]})
    sns.lineplot(data=subwall, x="Model", y="TotalTimes", hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS, markersize=11, linewidth=2, ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([7, 13, 20, 72])
    ax1.set_xticklabels(['7B', '13B', '20B', '72B'])
    ax1.set_ylabel("Total Inference Time (s)")
    ax1.set_xlabel("Model Parameter Size (Billions)")
    ax1.set_title("(a) The Latency Wall\n(prefill + decode combined)")
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}'))
    ax1.legend(title="Scheduler", loc='upper left', fontsize=11)

    if not substall.empty:
        sns.barplot(data=substall, x="Simulator", y="Stalls", hue="Simulator",
                    palette=COLORS, edgecolor='black', ax=ax2, legend=False)
        ax2.set_title("(b) Per-Token Decode Stall\n72B FP32, 16H+32C, B=1")
        ax2.set_ylabel("Avg. Stall per Token (s)")
        ax2.set_xlabel("Scheduler")
        ax2.set_ylim(0, substall["Stalls"].max() * 1.3)
        for c in ax2.containers:
            ax2.bar_label(c, fmt='.1f', padding=3, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/fig_motivation_combined.pdf", bbox_inches='tight')
    print("Saved figures/fig_motivation_combined.pdf")


def plot_scaling_trends_from_csv(df):
    sub = df[df["Experiment"].isin(["BatchSweep", "BatchSweepAllQuants"])].copy()
    if sub.empty:
        return

    precisions = ['fp32', 'fp16', 'int8', 'int4']
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))

    for i, prec in enumerate(precisions):
        ax = axes[i]
        data_p = sub[sub["Quant"] == prec].sort_values("BatchSize")
        
        sns.lineplot(data=data_p, x="BatchSize", y="TPS", hue="Simulator", 
                     style="Simulator", palette=COLORS, markers=MARKERS, 
                     markersize=10, linewidth=2, ax=ax)
        
        ax.set_title(f"Precision: {prec.upper()}", fontsize=18)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Batch Size (B)", fontsize=16)
        if i == 0:
            ax.set_ylabel("Throughput (TPS)", fontsize=16)
        else:
            ax.set_ylabel("")
            ax.get_legend().remove()
        
        ax.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("figures/scaling_trends_dynamic.pdf", bbox_inches='tight')

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("=" * 55)
    print(f"Generating Consolidated Paper Figures (SERVING_BATCH={SERVING_BATCH})")
    print("=" * 55)

    df = load_and_clean_data()
    if df is not None:
        plot_total_inference(df)
        plot_stall_duplex_phy(df)
        plot_combined_scalability(df)
        plot_combined_quantization(df)
        plot_memory_sensitivity(df)
        plot_memory_sensitivity_int8_fp16(df)
        plot_batch_sweep(df)
        plot_sparsity_collapse_theory()
        plot_metrics(df)
        plot_duplex_reconstruction()
        plot_misc_stats(df)
        plot_pareto(df)
        plot_total_inference_latency(df)
        plot_stall_latency(df)
        plot_motivation_combined(df)
        plot_scaling_trends_from_csv(df)

        print("\n✓ All figures saved in ./figures/")