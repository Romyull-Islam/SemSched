import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# PLOTTING STYLE CONFIGURATION
# ==========================================
plt.style.use('seaborn-v0_8-paper')

# Global Font Settings for Paper Quality (IEEE/ACM Standard)
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

def plot_all():
    # Load the data - Updated to match your experiment runner filename
    filename = "final_results_all_combinations.csv"
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {filename}")
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    # Normalize Hit_Rate
    if df["Hit_Rate"].max() > 1.0:
        df["Hit_Rate"] = df["Hit_Rate"] / 100.0

    # Rename 'DuplexGen' to 'SemDuplex'
    df["Simulator"] = df["Simulator"].replace("DuplexGen", "SemDuplex")

    # PROFESSIONAL COLOR PALETTE
    colors = {
        'Baseline':  '#7f7f7f',  # Gray
        'Adaptive':  '#377eb8',  # Blue
        'Async':     '#ff7f00',  # Orange
        'SemDuplex': '#e41a1c'   # Red
    }

    # ==========================================
    # 1. Scalability Line Plot (Decode) - FIXED Y-SCALE
    # ==========================================
    sub = df[df["Experiment"] == "Scalability"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.lineplot(data=sub, x="Model", y="TPS", hue="Simulator", style="Simulator", 
                      markers=True, palette=colors, linewidth=3, markersize=12, ax=ax)
        ax.set_xscale('log')
        ax.set_yscale('log') # Added log scale so 72B isn't a flat line
        ax.set_xticks([7, 13, 20, 72])
        ax.set_xticklabels(["7B", "13B", "20B", "72B"])
        ax.set_ylabel("Decode Throughput (Tokens/s)")
        ax.set_title("Scalability across Model Sizes (Decode)")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        
        plt.legend(title="Method", title_fontsize=14, loc='upper right', frameon=True)
        plt.tight_layout()
        plt.savefig("fig1_scale.pdf", bbox_inches='tight')
        print("Saved fig1_scale.pdf")

    # ==========================================
    # 2. Quantization Bar Plot (FIXED OVERLAP)
    # ==========================================
    sub = df[df["Experiment"] == "Quantization"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(data=sub, x="Quant", y="TPS", hue="Simulator", palette=colors, 
                    ax=ax, edgecolor='black', linewidth=1)
        ax.set_title("Quantization Impact (72B Model)")
        ax.set_ylabel("Throughput (Tokens/s)")
        ax.set_xlabel("Quantization Level")
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=5, fontsize=12, rotation=45)
            
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig("fig2_quant.pdf", bbox_inches='tight')
        print("Saved fig2_quant.pdf")

    # ==========================================
    # 3. Memory Sensitivity Plot
    # ==========================================
    sub = df[(df["Experiment"] == "Memory") & (df["Quant"] == "fp32")]
    if sub.empty: sub = df[df["Experiment"] == "Memory"]

    if not sub.empty:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        sns.barplot(data=sub, x="MemConfig", y="TPS", hue="Simulator", palette=colors, 
                    ax=ax, edgecolor='black', linewidth=1)
        ax.set_title("Memory Configuration Sensitivity (72B FP32)")
        ax.set_ylabel("Throughput (Tokens/s)")
        ax.set_xlabel("Host DRAM + CXL DRAM Configuration")
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=11, rotation=45)

        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        plt.legend(loc='center left', frameon=True, ncol=2)
        plt.tight_layout()
        plt.savefig("fig3_memory.pdf", bbox_inches='tight')
        print("Saved fig3_memory.pdf")

    # ==========================================
    # 4. Hit Rate/Stall Analysis
    # ==========================================
    sub = df[(df["Experiment"] == "Scalability") & (df["Model"] == 72)].copy()
    if not sub.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        sns.barplot(data=sub, x="Simulator", y="Stall_s", hue="Simulator", ax=ax1, 
                    palette=colors, edgecolor='black', linewidth=1, legend=False)
        ax1.set_title("Compute Stall Latency")
        ax1.set_ylabel("Stall Time (s)")
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.1f', padding=3, fontsize=12)
        
        sns.barplot(data=sub, x="Simulator", y="Hit_Rate", hue="Simulator", ax=ax2, 
                    palette=colors, edgecolor='black', linewidth=1, legend=False)
        ax2.set_title("Effective Cache Hit Rate")
        ax2.set_ylabel("Hit Rate (Ratio)")
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.2f', padding=3, fontsize=12)
        
        plt.tight_layout()
        plt.savefig("fig4_metrics.pdf", bbox_inches='tight')
        print("Saved fig4_metrics.pdf")

    # ==========================================
    # 5. Duplex Mechanics Analysis (NATURAL ONLY)
    # ==========================================
    sub = df[(df["Experiment"] == "Scalability") & (df["Simulator"] == "SemDuplex")]
    if not sub.empty:
        sub = sub.sort_values("Model")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Read Ratio (Should be high for FP32, lower for INT4)
        sns.barplot(data=sub, x="Model", y="Duplex_Read_Ratio", color='#3182bd', ax=ax1, edgecolor='black')
        ax1.set_title("Natural Duplex Traffic Balance")
        ax1.set_ylabel("Read Ratio (%)")
        ax1.set_xticklabels(["7B", "13B", "20B", "72B"])

        # Plot 2: Count of Natural Operations
        sns.barplot(data=sub, x="Model", y="Injected_Ops", color='#e6550d', ax=ax2, edgecolor='black')
        ax2.set_title("Natural Write Activity (KV + Eviction)")
        ax2.set_ylabel("Count of Write Ops")
        ax2.set_xticklabels(["7B", "13B", "20B", "72B"])
        
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%d', padding=3, fontsize=12)

        plt.tight_layout()
        plt.savefig("fig5_natural_duplex.pdf")
        print("Saved fig5_natural_duplex.pdf")

    # ==========================================
    # 6. Sparsity Savings
    # ==========================================
    sub = df[(df["Experiment"] == "Scalability") & (df["Model"] == 72) & (df["Simulator"] == "SemDuplex")]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        savings_tflops = sub["Sparsity_Savings_FLOPs"] / 1e12
        sns.barplot(x=["SemDuplex"], y=savings_tflops, color=colors['SemDuplex'], 
                    edgecolor='black', linewidth=1, ax=ax, width=0.4)
        ax.set_title("Compute Skipped via Sparsity (72B)")
        ax.set_ylabel("Skipped Compute (TFLOPs)")
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f TFLOPs', fontsize=14, padding=3)
        plt.tight_layout()
        plt.savefig("fig6_sparsity.pdf", bbox_inches='tight')
        print("Saved fig6_sparsity.pdf")

    # ==========================================
    # 7. Prefill Throughput Comparison
    # ==========================================
    sub = df[df["Experiment"] == "Scalability"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=sub, x="Model", y="Prefill_TPS", hue="Simulator", 
                    palette=colors, ax=ax, edgecolor='black', linewidth=1)
        ax.set_title("Prefill Throughput Scalability")
        ax.set_ylabel("Prefill Speed (Tokens/s)")
        ax.set_xlabel("Model Size (Parameters)")
        ax.set_xticklabels(["7B", "13B", "20B", "72B"])
        ax.set_yscale('log')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=12)
        plt.legend(loc='upper right', ncol=2, frameon=True)
        plt.tight_layout()
        plt.savefig("fig7_prefill.pdf", bbox_inches='tight')
        print("Saved fig7_prefill.pdf")

    # ==========================================
    # NEW 8. Speedup Factor over Baseline (PRO PLOT)
    # ==========================================
    sub = df[df["Experiment"] == "Quantization"].copy()
    if not sub.empty:
        # Calculate speedup relative to Baseline for each Quant level
        baselines = sub[sub["Simulator"] == "Baseline"].set_index("Quant")["TPS"]
        sub["Speedup"] = sub.apply(lambda x: x["TPS"] / baselines[x["Quant"]], axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sub[sub["Simulator"] != "Baseline"], x="Quant", y="Speedup", 
                    hue="Simulator", palette=colors, ax=ax, edgecolor='black')
        ax.axhline(1, color='black', linestyle='--', label="Baseline (1.0x)")
        ax.set_title("SemDuplex Speedup Factor (72B Model)")
        ax.set_ylabel("Speedup (x-times faster)")
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2fx', padding=3, fontsize=12)
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig("fig8_speedup.pdf", bbox_inches='tight')
        print("Saved fig8_speedup.pdf")

    # ==========================================
    # NEW 9. Pareto Frontier: Prefill vs Decode Efficiency
    # ==========================================
    sub = df[df["Experiment"] == "Scalability"].copy()
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(data=sub, x="Prefill_TPS", y="TPS", hue="Simulator", 
                        size="Model", sizes=(100, 600), palette=colors, ax=ax, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Prefill Throughput (Tokens/s)")
        ax.set_ylabel("Decode Throughput (Tokens/s)")
        ax.set_title("System Efficiency Pareto Frontier")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig("fig9_pareto.pdf", bbox_inches='tight')
        print("Saved fig9_pareto.pdf")

if __name__ == "__main__":
    plot_all()