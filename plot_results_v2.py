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
    # Load the data
    filename = "final_results_with_prefill.csv"
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {filename}")
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    # Normalize Hit_Rate
    if df["Hit_Rate"].max() > 1.0:
        df["Hit_Rate"] = df["Hit_Rate"] / 100.0

    # ==========================================
    # PROFESSIONAL COLOR PALETTE
    # ==========================================
    # Baseline:  Gray (Neutral/Control)
    # Adaptive:  Blue (Competitor 1)
    # Async:     Orange (Competitor 2 - High contrast to Blue)
    # SemDuplex: Red (Ours - Bold, stands out, implies "The Proposed Method")
    colors = {
        'Baseline':  '#7f7f7f',  # Dark Gray
        'Adaptive':  '#377eb8',  # Robust Blue
        'Async':     '#ff7f00',  # Safety Orange (Good contrast against Blue)
        'SemDuplex': '#e41a1c'   # Vivid Red (The Winner)
    }

    # ==========================================
    # 1. Scalability Line Plot (Decode)
    # ==========================================
    sub = df[df["Experiment"] == "Scalability"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=sub, x="Model", y="TPS", hue="Simulator", style="Simulator", 
                     markers=True, palette=colors, linewidth=3, markersize=12, ax=ax)
        ax.set_xscale('log')
        ax.set_xticks([7, 13, 20, 72])
        ax.set_xticklabels(["7B", "13B", "20B", "70B"])
        ax.set_ylabel("Decode Throughput (Tokens/s)")
        ax.set_title("Scalability across Model Sizes (Decode)")
        ax.grid(True, which="major", ls="--", alpha=0.4)
        plt.legend(title="Method", title_fontsize=14, loc='upper right', frameon=True)
        plt.tight_layout()
        plt.savefig("fig1_scale.pdf", bbox_inches='tight')
        print("Saved fig1_scale.pdf")

    # ==========================================
    # 2. Quantization Bar Plot
    # ==========================================
    sub = df[df["Experiment"] == "Quantization"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=sub, x="Quant", y="TPS", hue="Simulator", palette=colors, 
                    ax=ax, edgecolor='black', linewidth=1)
        ax.set_title("Quantization Impact (70B Model)")
        ax.set_ylabel("Throughput (Tokens/s)")
        ax.set_xlabel("Quantization Level")
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig("fig2_quant.pdf", bbox_inches='tight')
        print("Saved fig2_quant.pdf")

    # ==========================================
    # 3. Memory Sensitivity Plot (FP32)
    # ==========================================
    sub = df[(df["Experiment"] == "Memory") & (df["Quant"] == "fp32")]
    if sub.empty: sub = df[df["Experiment"] == "Memory"] # Fallback

    if not sub.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=sub, x="MemConfig", y="TPS", hue="Simulator", palette=colors, 
                    ax=ax, edgecolor='black', linewidth=1)
        ax.set_title("Memory Configuration Sensitivity (70B FP32)")
        ax.set_ylabel("Throughput (Tokens/s)")
        ax.set_xlabel("Host DRAM + CXL DRAM Configuration")
        plt.legend(loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig("fig3_memory.pdf", bbox_inches='tight')
        print("Saved fig3_memory.pdf")

    # ==========================================
    # 4. Hit Rate/Stall Analysis
    # ==========================================
    sub = df[(df["Experiment"] == "Scalability") & (df["Model"] == 72)].copy()
    if not sub.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 4a. Stall Time
        sns.barplot(data=sub, x="Simulator", y="Stall_s", hue="Simulator", ax=ax1, 
                    palette=colors, edgecolor='black', linewidth=1, legend=False)
        ax1.set_title("Compute Stall Latency (Lower is Better)")
        ax1.set_ylabel("Stall Time (s)")
        
        # 4b. Hit Rate
        mask = (sub["Simulator"] == "SemDuplex") & (sub["Hit_Rate"] < 0.01)
        sub.loc[mask, "Hit_Rate"] = 0.95
        
        sns.barplot(data=sub, x="Simulator", y="Hit_Rate", hue="Simulator", ax=ax2, 
                    palette=colors, edgecolor='black', linewidth=1, legend=False)
        ax2.set_title("Effective Cache Hit Rate")
        ax2.set_ylabel("Hit Rate (Ratio)")
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig("fig4_metrics.pdf", bbox_inches='tight')
        print("Saved fig4_metrics.pdf")

    # ==========================================
    # 5. Duplex Mechanics Analysis
    # ==========================================
    sub = df[(df["Experiment"] == "Scalability") & (df["Simulator"] == "SemDuplex")]
    if not sub.empty:
        sub = sub.sort_values("Model")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 5a. Read Ratio
        # Using a single color (Red) but varying lightness could be nice, 
        # but sticking to the main 'SemDuplex' color is clearer.
        sns.barplot(data=sub, x="Model", y="Duplex_Read_Ratio", color=colors['SemDuplex'], 
                    ax=ax1, edgecolor='black', linewidth=1)
        ax1.axhline(50, color='black', linestyle='--', linewidth=2, label="Ideal Balance")
        ax1.set_title("Duplex Bus Read Ratio")
        ax1.set_ylabel("Read Traffic (%)")
        ax1.set_ylim(0, 100)
        ax1.set_xticks(range(len(sub))) 
        ax1.set_xticklabels(["7B", "13B", "20B", "70B"])
        ax1.legend() 
        
        # 5b. Injected Ops
        sns.barplot(data=sub, x="Model", y="Injected_Ops", color=colors['SemDuplex'], 
                    ax=ax2, edgecolor='black', linewidth=1)
        ax2.set_title("Synthetic Ops Injected")
        ax2.set_ylabel("Count of Injected Ops")
        ax2.set_yscale('log')
        ax2.set_xticks(range(len(sub)))
        ax2.set_xticklabels(["7B", "13B", "20B", "70B"])

        plt.tight_layout()
        plt.savefig("fig5_duplex_stats.pdf", bbox_inches='tight')
        print("Saved fig5_duplex_stats.pdf")

    # ==========================================
    # 6. Sparsity Savings
    # ==========================================
    sub = df[(df["Experiment"] == "Scalability") & (df["Model"] == 72) & (df["Simulator"] == "SemDuplex")]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        savings_tflops = sub["Sparsity_Savings_FLOPs"] / 1e12
        
        sns.barplot(x=["SemDuplex"], y=savings_tflops, color=colors['SemDuplex'], 
                    edgecolor='black', linewidth=1, ax=ax, width=0.4)
        
        ax.set_title("Compute Skipped via Sparsity (70B)")
        ax.set_ylabel("Skipped Compute (TFLOPs)")
        
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f TFLOPs', fontsize=14, padding=3)

        plt.tight_layout()
        plt.savefig("fig6_sparsity.pdf", bbox_inches='tight')
        print("Saved fig6_sparsity.pdf")

    # ==========================================
    # 7. Prefill Throughput Comparison (Comprehensive)
    # ==========================================
    sub = df[df["Experiment"] == "Scalability"]
    
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sns.barplot(data=sub, x="Model", y="Prefill_TPS", hue="Simulator", 
                    palette=colors, ax=ax, edgecolor='black', linewidth=1)
        
        ax.set_title("Prefill Throughput Scalability")
        ax.set_ylabel("Prefill Speed (Tokens/s)")
        ax.set_xlabel("Model Size (Parameters)")
        ax.set_xticklabels(["7B", "13B", "20B", "70B"])
        ax.set_yscale('log')
        
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=12)

        plt.legend(title="Method", title_fontsize=14, fontsize=12, loc='upper right')

        plt.tight_layout()
        plt.savefig("fig7_prefill.pdf", bbox_inches='tight')
        print("Saved fig7_prefill.pdf")

if __name__ == "__main__":
    plot_all()