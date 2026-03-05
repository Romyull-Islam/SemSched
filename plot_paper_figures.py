import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator
import os

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

COLORS  = {'FlexGen': '#7f7f7f', 'LIA': '#377eb8', 'LLMFlash': '#ff7f0e', 'SemDuplex': '#e41a1c'}
MARKERS = {'FlexGen': 'o',       'LIA': 's',        'LLMFlash': '^',        'SemDuplex': 'D'}

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
        'semduplex_scheduler.py':'SemDuplex',
        'flashllm_baseline.py':  'LLMFlash',
        'FlashLLM':              'LLMFlash',
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

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
    sem72 = sub[(sub["Simulator"] == "SemDuplex") & (sub["Model"] == 72)]
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

    semspeedup72 = subspeed[(subspeed["Simulator"] == "SemDuplex") &
                             (subspeed["Model"] == 72)]["Speeduptotal"]
    if not semspeedup72.empty:
        ax2.annotate(f'SemDuplex\n{semspeedup72.values[0]:.0f}× speedup\n(prefill-driven)',
                     xy=(3, semspeedup72.values[0]),
                     xytext=(1.5, semspeedup72.values[0] + 0.6), fontsize=10,
                     color=COLORS['SemDuplex'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['SemDuplex'], lw=1.5),
                     bbox=dict(boxstyle='round', fc='white', ec=COLORS['SemDuplex'], alpha=0.8))

    plt.tight_layout()
    plt.savefig("figures/fig_total_inference.pdf", bbox_inches='tight')
    print("Saved figures/fig_total_inference.pdf  [PRIMARY PAPER FIGURE]")


# ==========================================
# 2. STALL DUPLEX PHY — FP32, B=128
# ==========================================
def plot_stall_duplex_phy(df):
    baseline_total     = 57.1821
    write_stall_total  = 1.1821
    num_layers         = 80
    cxl_bw             = 28.0
    stall_per_layer    = write_stall_total / num_layers
    read_per_layer     = (baseline_total - write_stall_total) / num_layers
    narrow_write_width = 0.004
    zoom_limit_s       = 5.0
    time_scale         = np.linspace(0, zoom_limit_s, 5000)

    fig = plt.figure(figsize=(16, 6))
    gs  = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.4], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[:, 0])

    # ← FIXED: strictly FP32 + B=128 from Memory experiment
    sub = df[
        (df["Experiment"] == "Memory") &
        (df["Model"]      == 72) &
        (df["Quant"]      == "fp32") &
        (df["BatchSize"]  == SERVING_BATCH)
    ].copy()

    if sub.empty:
        print(f"[StallDuplex] No FP32 B={SERVING_BATCH} Memory data. Available:")
        mem = df[(df["Experiment"] == "Memory") & (df["Model"] == 72)]
        print(mem[["Quant", "BatchSize"]].drop_duplicates().to_string())
        # fallback: any FP32 at 72B
        sub = df[
            (df["Experiment"] == "Memory") &
            (df["Model"]      == 72) &
            (df["Quant"]      == "fp32")
        ].copy()
        if sub.empty:
            print("[StallDuplex] No FP32 72B data at all. Skipping bar chart.")
        else:
            b_actual = sub["BatchSize"].mode().iloc[0]
            sub      = sub[sub["BatchSize"] == b_actual].copy()
            print(f"[StallDuplex] Fallback: FP32 B={b_actual}")
    else:
        b_actual = SERVING_BATCH
        print(f"[StallDuplex] Using FP32 B={b_actual} ✓")

    if not sub.empty:
        # Use MemConfig 16H+32C if available, else all configs averaged
        if "MemConfig" in sub.columns and "16GB_Host+32GB_CXL" in sub["MemConfig"].values:
            sub = sub[sub["MemConfig"] == "16GB_Host+32GB_CXL"].copy()

        sub["Stall_s"] = 1.0 / sub["TPS"]
        sub = sub.sort_values(by="Stall_s", ascending=False)

        b_label = sub["BatchSize"].iloc[0] if "BatchSize" in sub.columns else "?"
        q_label = sub["Quant"].iloc[0].upper() if "Quant" in sub.columns else "FP32"

        BAR_COLORS = {'FlexGen': '#8c8c8c', 'LIA': '#bcbd22',
                      'LLMFlash': '#1f77b4', 'SemDuplex': 'red'}
        sns.barplot(x='Simulator', y='Stall_s', data=sub, hue='Simulator',
                    palette=BAR_COLORS, edgecolor='black', legend=False, ax=ax1)
        ax1.set_title(f"(a) Per-Token Latency (72B {q_label}, B={b_label}, 16H+32C)",
                      fontsize=16)
        ax1.set_ylabel("Seconds per Token", fontsize=14)
        ax1.set_ylim(0, sub["Stall_s"].max() * 1.35)
        ax1.set_xlabel("Scheduler")
        for p in ax1.patches:
            if p.get_height() > 0:
                ax1.annotate(f'{p.get_height():.2f}s',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 10),
                             textcoords='offset points', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, "No FP32 data available", ha='center', va='center',
                 transform=ax1.transAxes, fontsize=13)

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

        b_read [(time_scale >= start_r) & (time_scale < end_r)]  = cxl_bw
        b_stall[(time_scale >= start_s) & (time_scale < end_s)]  = cxl_bw
        b_write[(time_scale >= start_w) & (time_scale < end_w)]  = cxl_bw

        ax2.annotate('', xy=(end_r + (stall_per_layer / 2), 30),
                     xytext=(end_r + (stall_per_layer / 2), 42),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax2.fill_between(time_scale, 0, b_read,  color='red',   label="Read (Weights)")
    ax2.fill_between(time_scale, 0, b_stall, color='white', label="Bus Turnaround Stall")
    ax2.fill_between(time_scale, 0, b_write, color='blue',  label="KV Write")
    ax2.annotate('Sequential Stalls', xy=(1.5, 43), fontsize=12, fontweight='bold')
    ax2.set_title("(b) Baseline Simplex: Sequential", fontsize=16)
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

    ax3.fill_between(time_scale, 0, s_read,  color='red',  label="Read Lane (Solid)")
    ax3.fill_between(time_scale, 0, s_write, color='blue', label="Write Lane (Parallel)")
    ax3.annotate('NO STALL: Continuous Read', xy=(2.5, 10), ha='center',
                 fontsize=14, color='white',
                 bbox=dict(boxstyle="round,pad=0.3", fc="darkgreen", ec="none", alpha=0.8))
    ax3.set_title("(c) SemDuplex Full-Duplex: Write Utilization", fontsize=16)
    ax3.set_ylabel("BW (GB/s)", fontsize=14)
    ax3.set_ylim(-40, 40)
    ax3.axhline(0, color='black', lw=1.5)
    ax3.set_xlabel("Elapsed Time (Seconds)", fontsize=14)
    ax3.set_xticks(np.arange(0, 6, 1))
    ax3.set_xticklabels([f"{i}s" for i in range(6)])
    #ax3.legend(loc='upper right', fontsize=12, ncol=2)

    plt.tight_layout()
    plt.savefig("figures/fig_stall_duplex_phy.pdf", bbox_inches='tight')
    print("Saved figures/fig_stall_duplex_phy.pdf  [FP32, B=128]")


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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

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
    # At each model size, sort simulators by TPS and assign
    # alternating above/below offsets to prevent overlap
    MODELS = sorted(sub["Model"].unique())
    SIMS   = ['FlexGen', 'LIA', 'LLMFlash', 'SemDuplex']

    # Pixel offset ladder: rank 0→bottom, rank 3→top
    # Offsets in points (positive = above, negative = below)
    #OFFSET_LADDER = [-22, -11, +11, +22]   # 4 simulators → 4 rungs
    # Increased pixel offset ladder to move labels further from lines
    OFFSET_LADDER = [-17, -15, +19, +30]

    for model in MODELS:
        model_data = sub[sub["Model"] == model].copy()
        # Sort by TPS ascending so rank 0 = lowest value
        model_data = model_data.sort_values("TPS")
        ranked_sims = model_data["Simulator"].tolist()

        for rank, sim in enumerate(ranked_sims):
            row = model_data[model_data["Simulator"] == sim]
            if row.empty:
                continue
            tps   = row["TPS"].values[0]
            color = COLORS.get(sim, 'black')
            yoff  = OFFSET_LADDER[rank]

            ax1.annotate(
                f'{tps:.1f}',
                xy=(model, tps),
                xytext=(0, yoff),
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
        ax2.bar_label(c, labels=labels, padding=3, fontsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig("figures/fig_combined_scalability.pdf", bbox_inches='tight')
    print(f"Saved figures/fig_combined_scalability.pdf  [B={b_use}]")


"""
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    sns.barplot(data=sub, x="Quant", y="TPS", hue="Simulator",
                order=quant_order, palette=COLORS, edgecolor='black', ax=ax1)
    ax1.set_title(f"(a) Decode Throughput, 72B, 32H+32C, B={b_use}", fontsize=16)
    ax1.set_ylabel("Tokens/s", fontsize=14)
    ax1.set_xlabel("Quantization", fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_ylim(0, sub["TPS"].max() * 1.4)
    for c in ax1.containers:
        labels = [f'{v:.2f}' for v in c.datavalues]
        ax1.bar_label(c, labels=labels, padding=3, fontsize=12, rotation=45)

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
        #for c in ax2.containers:
            #ax2.bar_label(c, fmt='.2f', padding=3, fontsize=10, rotation=45)
        for i, c in enumerate(ax2.containers):
            labels = [f'{v:.2f}' for v in c.datavalues]
            ax2.bar_label(c, labels=labels, padding=4, fontsize=11,rotation=45)
    else:
        ax2.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                 transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig("figures/fig_combined_quantization.pdf", bbox_inches='tight')
    print(f"Saved figures/fig_combined_quantization.pdf  [B={b_use}]")"""

# ---------------------------
# 4. Combined quantization (1x3 Layout)
# ---------------------------
def plot_combined_quantization(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Colors fallback
    COLORS = {'SemDuplex': '#1f77b4', 'LLMFlash': '#d62728',  'LIA': '#2ca02c', 'FlexGen': '#7f7f7f'}
    
    # Isolate 72B data
    df_72 = df[df["Model"] == 72].copy()
    if df_72.empty:
        return

    # Find target batch size (prefer 128 for serving)
    available_batches = sorted(df_72["BatchSize"].dropna().unique().tolist())
    b_use = 128 if 128 in available_batches else available_batches[-1]
    print(f"[Quantization] Using B={b_use} (available: {available_batches})")

    # Extract the two memory configurations
    df_32 = df_72[(df_72["MemConfig"].isin(["32H+32C", "32GB_Host+32GB_CXL"])) & (df_72["BatchSize"] == b_use)].copy()
    df_16 = df_72[(df_72["MemConfig"].isin(["16H+32C", "16GB_Host+32GB_CXL"])) & (df_72["BatchSize"] == b_use)].copy()

    # Calculate speedup for the 16H+32C tight configuration
    base_sim = 'LLMFlash' if 'LLMFlash' in df_16["Simulator"].values else 'FlexGen'
    if not df_16.empty:
        base_data = df_16[df_16["Simulator"] == base_sim][["Quant", "TPS"]].rename(columns={"TPS": "BaseTPS"})
        df_16 = df_16.merge(base_data, on="Quant", how="left")
        df_16["Speedup"] = (df_16["TPS"] / df_16["BaseTPS"]).replace(0, float('nan'))

    quant_order = ["fp32", "fp16", "int8", "int4"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))

    # --- Subplot 1: TPS (32H + 32C) ---
    if not df_32.empty:
        sns.barplot(data=df_32, x="Quant", y="TPS", hue="Simulator", order=quant_order, palette=COLORS, edgecolor='black', ax=ax1)
        ax1.set_title(f"(a) Decode Throughput (32H+32C, B={b_use})", fontsize=16)
        ax1.set_ylabel("Tokens/s", fontsize=14)
        ax1.set_xlabel("Quantization", fontsize=14)
        ax1.set_ylim(0, df_32["TPS"].max() * 1.35)
        ax1.legend(loc='upper left', fontsize=12)
        for c in ax1.containers:
            labels = [f'{v:.2f}' if pd.notnull(v) and v > 0 else '' for v in c.datavalues]
            ax1.bar_label(c, labels=labels, padding=3, fontsize=11, rotation=45)

    # --- Subplot 2: TPS (16H + 32C) ---
    if not df_16.empty:
        sns.barplot(data=df_16, x="Quant", y="TPS", hue="Simulator", order=quant_order, palette=COLORS, edgecolor='black', ax=ax2)
        ax2.set_title(f"(b) Decode Throughput (16H+32C, B={b_use})", fontsize=16)
        ax2.set_ylabel("Tokens/s", fontsize=14)
        ax2.set_xlabel("Quantization", fontsize=14)
        ax2.set_ylim(0, df_16["TPS"].max() * 1.35)
        if ax2.get_legend(): ax2.get_legend().remove()
        for c in ax2.containers:
            labels = [f'{v:.2f}' if pd.notnull(v) and v > 0 else '' for v in c.datavalues]
            ax2.bar_label(c, labels=labels, padding=3, fontsize=11, rotation=45)

    # --- Subplot 3: Speedup (16H + 32C) ---
    if not df_16.empty:
        subsp_16 = df_16[df_16["Simulator"] != base_sim]
        sns.barplot(data=subsp_16, x="Quant", y="Speedup", hue="Simulator", order=quant_order, palette=COLORS, edgecolor='black', ax=ax3)
        ax3.set_title(f"(c) Speedup vs {base_sim} (16H+32C, B={b_use})", fontsize=16)
        ax3.set_ylabel("Speedup (x)", fontsize=14)
        ax3.set_xlabel("Quantization", fontsize=14)
        ax3.set_ylim(0, subsp_16["Speedup"].max() * 1.35)
        if ax3.get_legend(): ax3.get_legend().remove()
        for c in ax3.containers:
            labels = [f'{v:.2f}x' if pd.notnull(v) and v > 0 else '' for v in c.datavalues]
            ax3.bar_label(c, labels=labels, padding=4, fontsize=11, rotation=45)

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
        (df["BatchSize"]  == SERVING_BATCH)   # ← STRICT B=128
    ].copy()

    if suball.empty:
        print(f"[Memory] No B={SERVING_BATCH} Memory data at 72B. Available:")
        mem = df[(df["Experiment"] == "Memory") & (df["Model"] == 72)]
        print(mem[["Quant", "BatchSize"]].drop_duplicates().to_string())
        return

    # ← STRICTLY only INT4 and FP32, nothing else
    PAPER_QUANTS = ["int4", "fp32"]

    # Check which of the two are actually in the data
    quants_available = [q for q in PAPER_QUANTS if q in suball["Quant"].values]
    print(f"[Memory] B={SERVING_BATCH} quants available: {suball['Quant'].unique()}")
    print(f"[Memory] Plotting only: {quants_available}")

    if not quants_available:
        print("[Memory] Neither INT4 nor FP32 found at B=128. Skipping.")
        return

    # Always create exactly 2 subplots regardless
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    """fig.suptitle(
        f"Memory Configuration Sensitivity (72B, B={SERVING_BATCH})",
        fontsize=16, fontweight='bold'
    )"""

    for col_idx, q in enumerate(PAPER_QUANTS):
        ax  = axes[col_idx]
        sub = suball[suball["Quant"] == q].copy()

        if sub.empty:
            ax.text(0.5, 0.5, f"No {q.upper()} data\nat B={SERVING_BATCH}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"(a) Memory Configuration Sensitivity (72B, B={SERVING_BATCH}),{q.upper()}) (no data)", fontsize=16)
            ax.set_xlabel("Memory Config", fontsize=14)
            ax.set_ylabel("Throughput (Tokens/s)", fontsize=14)
            continue

        mem_order = ["16GB_Host+32GB_CXL", "16GB_Host+64GB_CXL",
                     "32GB_Host+32GB_CXL", "32GB_Host+64GB_CXL"]
        mem_order = [m for m in mem_order if m in sub["MemConfig"].values]

        sns.barplot(data=sub, x="MemConfig", y="TPS", hue="Simulator",
                    order=mem_order, palette=COLORS, edgecolor='black', ax=ax)
        ax.set_title(f"(b) Memory Configuration Sensitivity (72B, {q.upper()}, B={SERVING_BATCH})", fontsize=16)
        ax.set_ylabel("Throughput (Tokens/s)", fontsize=14)
        ax.set_xlabel("Memory Config", fontsize=14)
        ax.xaxis.set_major_locator(FixedLocator(range(len(mem_order))))
        ax.set_xticklabels(
            [m.replace("GB_Host+", "H+").replace("GB_CXL", "C")
             for m in mem_order],
            rotation=0, ha='center', fontsize=14
        )
        ax.set_ylim(0, sub["TPS"].max() * 1.45)
        ax.legend(loc='upper left', fontsize=14, ncol=2)
        # Bar labels with larger font + 1 decimal for readability
        # Bold black labels (no box)
        for i, c in enumerate(ax.containers):
            labels = [f'{v:.1f}' for v in c.datavalues]
            ax.bar_label(c, labels=labels, padding=4, fontsize=11,
                         fontweight='bold', rotation=0)



    plt.tight_layout()
    plt.savefig("figures/fig_memory.pdf", bbox_inches='tight')
    print(f"Saved figures/fig_memory.pdf  [INT4+FP32 only, B={SERVING_BATCH}]")


# ---------------------------
# 6–14: Remaining figures
# (unchanged from your last working version)
# ---------------------------

def plot_batch_sweep(df):
    sub = df[df["Experiment"] == "BatchSweep"].copy()
    if sub.empty:
        print("Warning: No BatchSweep data found.")
        return
    sub["BatchSize"] = pd.to_numeric(sub["BatchSize"])
    sub = sub.sort_values("BatchSize")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    sns.lineplot(data=sub, x="BatchSize", y="TPS", hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS, markersize=12, linewidth=2.5, ax=ax1)
    ax1.set_title("(a) Batch Throughput\n72B INT8 16H+32C (tight memory)",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Throughput (Tokens/s)", fontsize=14)
    ax1.set_xlabel("Batch Size", fontsize=14)
    ax1.set_xticks([1, 4, 8, 16, 32, 64])
    ax1.axvline(x=4, color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7)
    ax1.annotate('Crossover\nB=4', xy=(4, sub["TPS"].max() * 0.55), fontsize=12,
                 color='darkgreen', fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', ec='darkgreen', alpha=0.6))
    ax1.legend(loc='upper left', fontsize=12, frameon=True)
    ax1.grid(True, alpha=0.3)

    llm_tps = sub[sub["Simulator"] == "LLMFlash"][["BatchSize", "TPS"]].rename(columns={"TPS": "LLMFlashTPS"})
    sem_tps = sub[sub["Simulator"] == "SemDuplex"][["BatchSize", "TPS"]].rename(columns={"TPS": "SemDuplexTPS"})
    lia_tps = sub[sub["Simulator"] == "LIA"][["BatchSize", "TPS"]].rename(columns={"TPS": "LIATPS"})

    if not llm_tps.empty and not sem_tps.empty:
        ratio_df = pd.merge(llm_tps, sem_tps, on="BatchSize")
        ratio_df = pd.merge(ratio_df, lia_tps, on="BatchSize", how="left")
        ratio_df["SemDuplex/LLMFlash"] = ratio_df["SemDuplexTPS"] / ratio_df["LLMFlashTPS"]
        ratio_df["LIA/LLMFlash"]       = ratio_df["LIATPS"]       / ratio_df["LLMFlashTPS"]

        ax2.plot(ratio_df["BatchSize"], ratio_df["SemDuplex/LLMFlash"],
                 color=COLORS['SemDuplex'], marker='D', markersize=10,
                 linewidth=2.5, label="SemDuplex / LLMFlash")
        ax2.plot(ratio_df["BatchSize"], ratio_df["LIA/LLMFlash"],
                 color=COLORS['LIA'], marker='s', markersize=10,
                 linewidth=2.5, linestyle='--', label="LIA / LLMFlash")
        ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label="Parity ratio=1")
        ax2.axvline(x=4, color='darkgreen', linestyle='--', linewidth=1.8, alpha=0.7)

        crossvals = ratio_df[ratio_df["SemDuplex/LLMFlash"] > 1.0]["BatchSize"]
        if not crossvals.empty:
            ax2.axvspan(crossvals.min(), ratio_df["BatchSize"].max(),
                        alpha=0.08, color='red', label="SemDuplex dominant")

        ax2.set_title("(b) Relative Efficiency vs LLMFlash\nSemDuplex advantage at B≥4",
                      fontsize=13, fontweight='bold')
        ax2.set_ylabel("TPS Ratio (>1 = SemDuplex better)", fontsize=13)
        ax2.set_xlabel("Batch Size", fontsize=14)
        ax2.set_xticks([1, 4, 8, 16, 32, 64])
        ax2.legend(loc='upper left', fontsize=11, frameon=True)
        ax2.grid(True, alpha=0.3)

        max_row = ratio_df.loc[ratio_df["SemDuplex/LLMFlash"].idxmax()]
        ax2.annotate(f'Peak {max_row["SemDuplex/LLMFlash"]:.2f}×\nB={int(max_row["BatchSize"])}',
                     xy=(max_row["BatchSize"], max_row["SemDuplex/LLMFlash"]),
                     xytext=(max_row["BatchSize"] - 20, max_row["SemDuplex/LLMFlash"] + 0.05),
                     fontsize=11, color=COLORS['SemDuplex'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['SemDuplex'], lw=1.5))
    else:
        ax2.text(0.5, 0.5, "Need both LLMFlash and SemDuplex data",
                 ha='center', va='center')

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

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
    ax2.annotate("Zero overflow\n(fits in DRAM)", xy=(2, 1), fontsize=11,
                 color='darkgreen',
                 bbox=dict(boxstyle='round', fc='lightgreen', ec='darkgreen', alpha=0.6))

    plt.tight_layout()
    plt.savefig("figures/fig_sparsity_collapse_theory.pdf", bbox_inches='tight')
    print("Saved figures/fig_sparsity_collapse_theory.pdf")


def plot_metrics(df):
    sub = df[(df["Experiment"] == "Scalability") & (df["Model"] == 72)].copy()
    if sub.empty:
        return

    if "Stalls" in df.columns:
        y_col, title, ylabel, fmt_str = "Stalls", "Compute Stall Latency", "Time (s)", ".1f"
    else:
        sub["Latency_ms"] = 1000.0 / sub["TPS"]
        y_col, title, ylabel, fmt_str = "Latency_ms", "Avg Token Latency (1/TPS)", "Latency (ms)", ".0f"

    has_hitrate = "HitRate" in df.columns
    if has_hitrate:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        ax_metric, ax_hit = axes[0], axes[1]
    else:
        fig, ax_metric = plt.subplots(1, 1, figsize=(8, 5))
        ax_hit = None

    sns.barplot(data=sub, x="Simulator", y=y_col, hue="Simulator",
                palette=COLORS, ax=ax_metric, legend=False)
    ax_metric.set_title(f"(a) {title}")
    ax_metric.set_ylabel(ylabel)
    for c in ax_metric.containers:
        ax_metric.bar_label(c, fmt=fmt_str, padding=3)

    if has_hitrate and ax_hit is not None:
        sns.barplot(data=sub, x="Simulator", y="HitRate", hue="Simulator",
                    palette=COLORS, ax=ax_hit, legend=False)
        ax_hit.set_title("(b) Cache Hit Rate")
        ax_hit.set_ylabel("Hit Rate (%)")
        for c in ax_hit.containers:
            ax_hit.bar_label(c, fmt='.2f', padding=3)

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax1.fill_between(time_ms, 0,  base_read,  color='#fdae61', alpha=0.9, label="Read Weights")
    ax1.fill_between(time_ms, 0, -base_write, color='#999999', alpha=0.9, label="Write KV-Cache")
    ax1.set_title("(a) Baseline Architecture", fontsize=15)
    ax1.set_ylabel("BW (GB/s)", fontsize=14)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.annotate("Stall", xy=(82, 0), xytext=(85, 15),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=2), fontsize=12)
    ax1.legend(fontsize=11)

    ax2.fill_between(time_ms, 0,  sem_read,  color='#e41a1c', alpha=0.7, label="Read Lane")
    ax2.fill_between(time_ms, 0, -sem_write, color='#377eb8', alpha=0.8, label="Write Injection")
    ax2.set_title("(b) SemDuplex Architecture", fontsize=14)
    ax2.set_ylabel("BW (GB/s)", fontsize=14)
    ax2.set_xlabel("Time (microseconds)", fontsize=14)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.annotate("Injection\nHidden", xy=(32.5, -15), xytext=(40, -30),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=2), fontsize=12)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("figures/fig_duplex_phy.pdf", bbox_inches='tight')
    print("Saved figures/fig_duplex_phy.pdf")


def plot_misc_stats(df):
    sub_cold  = df[(df["Experiment"] == "Quantization") & (df["Simulator"] == "SemDuplex")]
    has_sparse = "SparsitySavingsFLOPs" in df.columns
    sub_sparse = pd.DataFrame()
    if has_sparse:
        sub_sparse = df[(df["Experiment"] == "Scalability") &
                        (df["Model"] == 72) & (df["Simulator"] == "SemDuplex")]

    if sub_cold.empty and sub_sparse.empty:
        return

    if not sub_cold.empty and not sub_sparse.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        ax2 = None

    if not sub_cold.empty:
        quant_order = [q for q in ["fp32", "fp16", "int8", "int4"]
                       if q in sub_cold["Quant"].values]
        sns.barplot(data=sub_cold, x="Quant", y="Cold_Load_s",
                    order=quant_order, color='#9467bd', edgecolor='black', ax=ax1)
        ax1.set_title("System Cold Boot Time (72B)")
        ax1.set_ylabel("Seconds")
        for c in ax1.containers:
            ax1.bar_label(c, fmt='.1f', padding=3)

    if not sub_sparse.empty and ax2 is not None:
        val = sub_sparse["SparsitySavingsFLOPs"] / 1e12
        sns.barplot(x=["SemDuplex"], y=val, color=COLORS['SemDuplex'], ax=ax2, width=0.4)
        ax2.set_title("Compute Skipped (72B)")
        ax2.set_ylabel("TFLOPs")
        for c in ax2.containers:
            ax2.bar_label(c, fmt='.1f', padding=3)

    plt.tight_layout()
    plt.savefig("figures/fig_misc_stats.pdf", bbox_inches='tight')
    print("Saved figures/fig_misc_stats.pdf")


def plot_pareto(df):
    sub = df[df["Experiment"] == "Scalability"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(data=sub, x="Prefill_TPS", y="TPS", hue="Simulator",
                    size="Model", sizes=(50, 500), palette=COLORS, ax=ax, alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Prefill Speed (Tokens/s)")
    ax.set_ylabel("Decode Speed (Tokens/s)")
    ax.set_title("Efficiency Frontier\n(SemDuplex: top-right corner = optimal)")
    plt.tight_layout()
    plt.savefig("figures/fig_pareto.pdf", bbox_inches='tight')
    print("Saved figures/fig_pareto.pdf")


def plot_total_inference_latency(df):
    sub = df[df["Experiment"] == "Scalability"].copy()
    sub = sub[sub["Prefill_TPS"] > 0].copy()
    if sub.empty:
        return
    sub["TotalTimes"] = 512 / sub["Prefill_TPS"] + 16 / sub["TPS"]

    plt.figure(figsize=(8, 3.2))
    sns.lineplot(data=sub, x="Model", y="TotalTimes", hue="Simulator", style="Simulator",
                 palette=COLORS, markers=MARKERS, markersize=11, linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([7, 13, 20, 72], ['7B', '13B', '20B', '72B'], fontsize=14)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}'))
    plt.ylabel("Total Inference Time (s)", fontsize=14)
    plt.xlabel("Model Parameter Size (Billions)", fontsize=14)
    plt.title("The Latency Wall (Prefill + Decode)", fontsize=15)
    plt.legend(title="Scheduler", title_fontsize=12, fontsize=11)
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
        print("Warning: No 72B FP32 data for stall latency plot.")
        return
    sub["Stalls"] = 1.0 / sub["TPS"]
    sub = sub.sort_values(by="Stalls", ascending=False)

    plt.figure(figsize=(8, 3.5))
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
        ax2.tick_params(axis='x', rotation=0)
        for c in ax2.containers:
            ax2.bar_label(c, fmt='.1f', padding=3, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figures/fig_motivation_combined.pdf", bbox_inches='tight')
    print("Saved figures/fig_motivation_combined.pdf")


def plot_scaling_all():
        # Data definitions
    batches = [1, 4, 8, 16, 32, 64, 128]
    data = {
        'FP32': {
            'LLMFlash': [0.0595, 0.1402, 0.2620, 0.5208, 1.0416, 2.0831, 4.1659, ],
            'SemDuplex': [0.0423, 0.1693, 0.3386, 0.6772, 1.3545, 2.7089, 5.4178],
            'LIA': [0.0276, 0.1104, 0.2208, 0.4415, 0.8830, 1.7658, 3.5311 ],
            'FlexGen': [0.0165, 0.0659, 0.1312, 0.2600, 0.5109, 0.9869, 1.8477]
        },
        'FP16': {
            'LLMFlash': [0.1581, 0.3280, 0.6062, 1.2039, 2.4076, 4.8151, 9.6295],
            'SemDuplex': [0.0930, 0.3719, 0.7439, 1.4877, 2.9754, 5.9507, 11.9006],
            'LIA': [0.0585, 0.2340, 0.4680, 0.9360, 1.8720, 3.7438, 7.4866],
            'FlexGen': [0.0350, 0.1394, 0.2776, 0.5502, 1.0809, 2.0882, 3.9098]
        },
        'INT8': {
            'LLMFlash': [0.6629, 1.0250, 1.8165, 3.5952, 7.1896, 14.3785, 28.7543],
            'SemDuplex': [0.3315, 1.3258, 2.6514, 5.3021, 10.6015, 21.1926, 42.3435],
            'LIA': [0.1276, 0.5105, 1.0209, 2.0418, 4.0835, 8.1665, 16.3308],
            'FlexGen': [0.0791, 0.3154, 0.6279, 1.2446, 2.4453, 4.7241, 8.8458]
        },
        'INT4': {
            'LLMFlash': [1.3258, 3.5429, 6.7051, 13.3438, 26.6853, 53.3656, 106.7111],
            'SemDuplex': [0.6058, 2.4232, 4.8458, 9.6895, 19.3709, 38.7093, 77.2887],
            'LIA': [0.5273, 2.1093, 4.2185, 8.4370, 16.8741, 33.7482, 67.4963],
            'FlexGen': [0.1923, 0.7671, 1.5283, 3.0333, 5.9756, 11.6032, 21.9291]
        }
    }

    # Aesthetics
    precisions = ['FP32', 'FP16', 'INT8', 'INT4']
    markers = {'SemDuplex': 's', 'LLMFlash': 'o',  'LIA': '^', 'FlexGen': 'D'}
    colors = {'SemDuplex': '#1f77b4', 'LLMFlash': '#d62728',  'LIA': '#2ca02c', 'FlexGen': '#7f7f7f'}

    # Create Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for i, prec in enumerate(precisions):
        ax = axes[i]
        for scheduler in [ 'SemDuplex', 'LLMFlash', 'LIA', 'FlexGen']:
            ax.plot(batches, data[prec][scheduler], 
                    label=scheduler, 
                    marker=markers[scheduler], 
                    color=colors[scheduler], 
                    linewidth=1.8, 
                    markersize=6)
        
        ax.set_title(f'{prec}', fontsize=16)
        ax.set_xlabel('Batch Size (B)', fontsize=14)
        if i == 0:
            ax.set_ylabel('Throughput (TPS)', fontsize=14)
            ax.legend(fontsize=14, loc='upper left')
        
        # Log scaling
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(batches)
        ax.set_xticklabels(batches)
        ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig('scaling_trends.pdf', bbox_inches='tight') # Better for LaTeX
    plt.show()



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
        print(f"  Simulators : {sorted(df['Simulator'].unique())}")
        print(f"  Experiments: {sorted(df['Experiment'].unique())}")
        print()

        plot_total_inference(df)
        plot_stall_duplex_phy(df)
        plot_combined_scalability(df)
        plot_combined_quantization(df)
        plot_memory_sensitivity(df)
        plot_batch_sweep(df)
        plot_sparsity_collapse_theory()
        plot_metrics(df)
        plot_duplex_reconstruction()
        plot_misc_stats(df)
        plot_pareto(df)
        plot_total_inference_latency(df)
        plot_stall_latency(df)
        plot_motivation_combined(df)
        plot_scaling_all()

        print("\n✓ All figures saved in ./figures/")
        print("  Key B=128-sensitive figures:")
        print("    fig_stall_duplex_phy.pdf    (uses B=128 if available; else annotated fallback)")
        print("    fig_combined_scalability.pdf (filtered to single B, reported in title)")
        print("    fig_combined_quantization.pdf (uses SERVING_BATCH if available)")
        print("    fig_memory.pdf  (INT4+FP32, B=128 only)")
