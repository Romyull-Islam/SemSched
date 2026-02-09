import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# PLOTTING STYLE CONFIGURATION
# ==========================================
plt.style.use('seaborn-v0_8-paper')

# Safe font fallback to stop findfont warnings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5
})

# GLOBAL PALETTES
colors = {'Baseline': '#7f7f7f', 'Adaptive': '#377eb8', 'Async': '#ff7f00', 'SemDuplex': '#e41a1c'}
# Comparison Palette: Side-by-side comparison colors
comp_colors = {'fp32': '#9ecae1', 'int4': '#ef3b2c'} 

def plot_all():
    filename = "final_results_all_combinations.csv"
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {filename}")
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    df["Simulator"] = df["Simulator"].replace("DuplexGen", "SemDuplex")

    # ==========================================
    # 7. Prefill Throughput: FP32 vs. INT4 Side-by-Side (Across Sizes)
    # ==========================================
    # Filter for the two target precisions
    sub_pf = df[(df["Quant"].isin(["fp32", "int4"]))].copy()
    
    if not sub_pf.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        # 'hue' puts the precisions side-by-side
        sns.barplot(data=sub_pf, x="Model", y="Prefill_TPS", hue="Quant", 
                    palette=comp_colors, edgecolor='black', ax=ax)
        
        ax.set_yscale('log')
        ax.set_title("Prefill Throughput Comparison: FP32 vs. INT4")
        ax.set_ylabel("Throughput (Tokens/s) - Log Scale")
        ax.set_xlabel("Model Size (Parameters)")
        
        # Proper tick handling to avoid UserWarnings
        models = sorted(sub_pf["Model"].unique())
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([f"{m}B" for m in models])
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=11)
            
        plt.legend(title="Precision", frameon=True, loc='upper right')
        plt.tight_layout()
        plt.savefig("fig7_prefill_side_by_side.pdf")
        print("Saved fig7_prefill_side_by_side.pdf")

    # ==========================================
    # 11. 72B Prefill Specific Jump (The "Trigger" Evidence)
    # ==========================================
    sub_72 = df[(df["Model"] == 72) & (df["Quant"].isin(["fp32", "int4"]))]
    if not sub_72.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=sub_72, x="Simulator", y="Prefill_TPS", hue="Quant", 
                    palette=comp_colors, edgecolor='black', ax=ax)
        
        ax.set_yscale('log')
        ax.set_title("Prefill Efficiency Leap: 72B Model")
        ax.set_ylabel("Prefill TPS (Log Scale)")
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig("fig11_72b_prefill_leap.pdf")
        print("Saved fig11_72b_prefill_leap.pdf")

    # ==========================================
    # 10. Summary Performance Leap (Decode)
    # ==========================================
    sub_q = df[(df["Experiment"] == "Quantization") & (df["Quant"].isin(["fp32", "int4"]))]
    if not sub_q.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=sub_q, x="Simulator", y="TPS", hue="Quant", 
                    palette=comp_colors, edgecolor='black', ax=ax)
        ax.set_yscale('log')
        ax.set_title("System Throughput Leap: FP32 vs. INT4")
        ax.set_ylabel("Decode Tokens/sec (Log Scale)")
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=12, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig("fig10_decode_leap.pdf")
        print("Saved fig10_decode_leap.pdf")

if __name__ == "__main__":
    plot_all()