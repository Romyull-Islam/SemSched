import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import os

# ==========================================
# 1. PLOTTING STYLE CONFIGURATION
# ==========================================
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Professional Color Palette
COLORS = {
    'FlexGen':   '#7f7f7f',  # Gray (Baseline)
    'LIA':       '#377eb8',  # Blue (OS SOTA)
    'FlashLLM':  '#ff7f00',  # Orange (Prefetch SOTA)
    'SemDuplex': '#e41a1c'   # Red (Ours)
}

MARKERS = {
    'FlexGen': 'o',
    'LIA': 's',
    'FlashLLM': '^',
    'SemDuplex': 'D'
}

def load_and_clean_data():
    files = ["final_results_with_coldload.csv", "final_results_all_combinations.csv"]
    df = None
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"Loaded data from: {f}")
            break
    
    if df is None:
        print("Error: No data file found. Please run run_experiments.py first.")
        return None

    # Cleanup Simulator Names
    name_map = {
        "flexgen_baseline.py": "FlexGen",
        "lia_baseline.py": "LIA",
        "flashllm_baseline.py": "FlashLLM",
        "semduplex_scheduler.py": "SemDuplex"
    }
    df["Simulator"] = df["Simulator"].replace(name_map)
    
    if "Hit_Rate" in df.columns and df["Hit_Rate"].max() > 1.0:
        df["Hit_Rate"] = df["Hit_Rate"] / 100.0
        
    return df

# ==========================================
# FIGURE 1: SCALABILITY (Line Plot)
# ==========================================
def plot_scalability(df):
    sub = df[df["Experiment"] == "Scalability"]
    if sub.empty: return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=sub, x="Model", y="TPS", hue="Simulator", style="Simulator", 
                 palette=COLORS, markers=MARKERS, markersize=12, ax=ax)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([7, 13, 20, 72])
    ax.set_xticklabels(["7B", "13B", "20B", "72B"])
    ax.set_ylabel("Decode Throughput (Tokens/s)")
    ax.set_title("Scalability across Model Sizes")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    plt.legend(title="Method", loc='upper right')
    plt.tight_layout()
    plt.savefig("fig1_scale.pdf")
    print("Saved fig1_scale.pdf")

# ==========================================
# FIGURE 2: QUANTIZATION (Bar Plot)
# ==========================================
def plot_quantization(df):
    sub = df[(df["Experiment"] == "Quantization") & (df["Model"] == 72)]
    if sub.empty: return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sub, x="Quant", y="TPS", hue="Simulator", palette=COLORS, 
                edgecolor='black', ax=ax)
    
    ax.set_title("Quantization Impact (72B Model)")
    ax.set_ylabel("Throughput (Tokens/s)")
    
    # Add values
    if not sub["TPS"].empty:
        ax.set_ylim(0, sub["TPS"].max() * 1.15)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
        
    plt.tight_layout()
    plt.savefig("fig2_quant.pdf")
    print("Saved fig2_quant.pdf")

# ==========================================
# FIGURE 3: MEMORY SENSITIVITY (Bar Plot)
# ==========================================
def plot_memory_sensitivity(df):
    sub = df[df["Experiment"] == "Memory"]
    if sub.empty: return
    
    # Filter to one quantization to keep it clean
    if "int4" in sub["Quant"].values:
        sub = sub[sub["Quant"] == "int4"]
        title_suf = "(INT4)"
    elif "fp16" in sub["Quant"].values:
        sub = sub[sub["Quant"] == "fp16"]
        title_suf = "(FP16)"
    else:
        title_suf = "(Mixed)"

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=sub, x="MemConfig", y="TPS", hue="Simulator", palette=COLORS, 
                edgecolor='black', ax=ax)
    
    ax.set_title(f"Memory Configuration Sensitivity {title_suf}")
    ax.set_ylabel("Throughput (Tokens/s)")
    
    # Add values
    ax.set_ylim(0, sub["TPS"].max() * 1.15)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10, rotation=0)

    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("fig3_memory.pdf")
    print("Saved fig3_memory.pdf")

# ==========================================
# FIGURE 4: METRICS (Side-by-Side)
# ==========================================
def plot_metrics(df):
    if "Stall_s" not in df.columns: return
    sub = df[(df["Experiment"] == "Scalability") & (df["Model"] == 72)]
    if sub.empty: return
    
    # 1 Row, 2 Columns (Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Stalls
    sns.barplot(data=sub, x="Simulator", y="Stall_s", hue="Simulator", palette=COLORS, ax=axes[0], legend=False)
    axes[0].set_title("Compute Stall Latency")
    axes[0].set_ylabel("Time (s)")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt='%.1f', padding=3)
    
    # Plot Hit Rate
    if "Hit_Rate" in df.columns:
        sns.barplot(data=sub, x="Simulator", y="Hit_Rate", hue="Simulator", palette=COLORS, ax=axes[1], legend=False)
        axes[1].set_title("Cache Hit Rate")
        axes[1].set_ylabel("Hit Rate")
        for c in axes[1].containers:
            axes[1].bar_label(c, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    plt.savefig("fig4_metrics.pdf")
    print("Saved fig4_metrics.pdf")

# ==========================================
# FIGURE 5: DUPLEX PHYSICS (Side-by-Side)
# ==========================================
def plot_duplex_reconstruction():
    time_ms = np.linspace(0, 100, 1000)
    
    # Baseline
    base_read = np.zeros_like(time_ms); base_read[0:800] = 32; base_read[1000:] = 32
    base_write = np.zeros_like(time_ms); base_write[850:950] = 32
    
    # SemDuplex
    sem_read = np.zeros_like(time_ms); sem_read[:] = 32
    sem_write = np.zeros_like(time_ms); sem_write[200:300] = 32 

    # 1 Row, 2 Columns (Side-by-Side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Plot Baseline
    ax1.fill_between(time_ms, 0, base_read, color=COLORS['FlashLLM'], alpha=0.6, label="Read")
    ax1.fill_between(time_ms, 0, -base_write, color=COLORS['FlexGen'], alpha=0.6, label="Write")
    ax1.set_title("Baseline (Simplex)\nReads Pause for Writes")
    ax1.set_ylabel("BW (GB/s)")
    ax1.set_xlabel("Time (microseconds)")
    ax1.axhline(0, color='black')

    # Plot SemDuplex
    ax2.fill_between(time_ms, 0, sem_read, color=COLORS['SemDuplex'], alpha=0.6, label="Read")
    ax2.fill_between(time_ms, 0, -sem_write, color='#1f77b4', alpha=0.6, label="Write Injection")
    ax2.set_title("SemDuplex (Ours)\nWrite Injection")
    ax2.set_xlabel("Time (microseconds)")
    ax2.axhline(0, color='black')
    
    plt.tight_layout()
    plt.savefig("fig5_duplex_phy.pdf")
    print("Saved fig5_duplex_phy.pdf")

# ==========================================
# FIGURE 6: SPARSITY SAVINGS
# ==========================================
def plot_sparsity(df):
    if "Sparsity_Savings_FLOPs" not in df.columns: return
    sub = df[(df["Experiment"] == "Scalability") & (df["Model"] == 72) & (df["Simulator"] == "SemDuplex")]
    if sub.empty: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    val = sub["Sparsity_Savings_FLOPs"] / 1e12
    sns.barplot(x=["SemDuplex"], y=val, color=COLORS['SemDuplex'], ax=ax, width=0.4)
    ax.set_title("Compute Skipped via Sparsity (72B)")
    ax.set_ylabel("Skipped TFLOPs")
    for c in ax.containers: ax.bar_label(c, fmt='%.1f T', padding=3)
    plt.tight_layout()
    plt.savefig("fig6_sparsity.pdf")
    print("Saved fig6_sparsity.pdf")

# ==========================================
# FIGURE 7: PREFILL THROUGHPUT
# ==========================================
def plot_prefill(df):
    sub = df[df["Experiment"] == "Scalability"]
    if sub.empty: return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sub, x="Model", y="Prefill_TPS", hue="Simulator", palette=COLORS, 
                edgecolor='black', ax=ax)
    ax.set_yscale('log')
    ax.set_title("Prefill Throughput (Parallel Warmup)")
    ax.set_ylabel("Tokens/s")
    
    # Add values (formatted for log scale readability)
    for c in ax.containers:
        # Custom formatting for log scale
        labels = [f'{v:.0f}' if v > 0 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, padding=3, fontsize=9)

    plt.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig("fig7_prefill.pdf")
    print("Saved fig7_prefill.pdf")

# ==========================================
# FIGURE 8: SPEEDUP FACTOR
# ==========================================
def plot_speedup(df):
    sub = df[df["Experiment"] == "Quantization"].copy()
    if sub.empty: return
    
    base_sim = "FlashLLM" if "FlashLLM" in sub["Simulator"].values else "FlexGen"
    base_data = sub[sub["Simulator"] == base_sim].set_index("Quant")["TPS"]
    
    if base_data.empty: return
    sub["Speedup"] = sub.apply(lambda x: x["TPS"] / base_data.get(x["Quant"], 1), axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sub[sub["Simulator"] != base_sim], x="Quant", y="Speedup", hue="Simulator", 
                palette=COLORS, ax=ax, edgecolor='black')
    
    ax.axhline(1, color='black', linestyle='--')
    ax.set_title(f"Speedup vs {base_sim} (72B)")
    ax.set_ylabel("Speedup (x)")
    
    for c in ax.containers: ax.bar_label(c, fmt='%.1fx', padding=3)
    plt.tight_layout()
    plt.savefig("fig8_speedup.pdf")
    print("Saved fig8_speedup.pdf")

# ==========================================
# FIGURE 9: PARETO FRONTIER
# ==========================================
def plot_pareto(df):
    sub = df[df["Experiment"] == "Scalability"]
    if sub.empty: return

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=sub, x="Prefill_TPS", y="TPS", hue="Simulator", size="Model", 
                    sizes=(50, 500), palette=COLORS, ax=ax, alpha=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("Prefill Speed (Tokens/s)")
    ax.set_ylabel("Decode Speed (Tokens/s)")
    ax.set_title("Efficiency Frontier")
    plt.tight_layout()
    plt.savefig("fig9_pareto.pdf")
    print("Saved fig9_pareto.pdf")

# ==========================================
# FIGURE 10: COLD LOAD
# ==========================================
def plot_cold_load(df):
    if "Cold_Load_s" not in df.columns: return
    sub = df[(df["Experiment"] == "Quantization") & (df["Simulator"] == "SemDuplex")]
    if sub.empty: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=sub, x="Quant", y="Cold_Load_s", color='#9467bd', edgecolor='black', ax=ax)
    ax.set_title("System Cold Boot Time (72B)")
    ax.set_ylabel("Time (seconds)")
    for c in ax.containers: ax.bar_label(c, fmt='%.1fs', padding=3)
    plt.tight_layout()
    plt.savefig("fig10_coldload.pdf")
    print("Saved fig10_coldload.pdf")

if __name__ == "__main__":
    print(">>> Generating ALL 10 Paper Figures...")
    df = load_and_clean_data()
    if df is not None:
        plot_scalability(df)
        plot_quantization(df)
        plot_memory_sensitivity(df)
        plot_metrics(df)
        plot_duplex_reconstruction()
        plot_sparsity(df)
        plot_prefill(df)
        plot_speedup(df)
        plot_pareto(df)
        plot_cold_load(df)
        print("\n>>> Success. All figures saved.")