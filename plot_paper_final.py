import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("deep")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

def plot_all():
    try:
        df = pd.read_csv("final_results.csv")
    except FileNotFoundError:
        print("final_results.csv not found. Run paper_experiments.py first.")
        return

    # 1. Scalability Line Chart
    sub = df[df["Experiment"] == "Scalability"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Define markers and colors for consistency
        styles = {
            "Baseline": {"marker": "o", "color": "#7f8c8d", "ls": "--"},
            "Adaptive": {"marker": "s", "color": "#f1c40f", "ls": "-."},
            "Async":    {"marker": "D", "color": "#3498db", "ls": "-"},
            "DuplexGen": {"marker": "^", "color": "#c0392b", "ls": "-"}
        }
        
        for name, style in styles.items():
            data = sub[sub["Simulator"] == name]
            if not data.empty:
                ax.plot(data["Model"], data["TPS"], label=name, linewidth=2, **style)
        
        ax.set_xscale('log')
        ax.set_xticks([7, 13, 20, 72])
        ax.set_xticklabels(["7B", "13B", "20B", "70B"])
        ax.set_xlabel("Model Size (Billions of Parameters)", fontweight='bold')
        ax.set_ylabel("Throughput (Tokens/sec)", fontweight='bold')
        ax.set_title("Fig 1. Inference Scalability (FP32)", fontweight='bold')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("fig1_scalability.pdf")
        print("Saved fig1_scalability.pdf")

    # 2. Quantization Bar Chart
    sub = df[df["Experiment"] == "Quantization"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.barplot(data=sub, x="Quant", y="TPS", hue="Simulator", 
                    hue_order=["Baseline", "Adaptive", "Async", "DuplexGen"],
                    palette=['#95a5a6', '#f1c40f', '#3498db', '#c0392b'],
                    edgecolor="black", ax=ax)
        
        ax.set_title("Fig 2. Quantization Impact (70B Model)", fontweight='bold')
        ax.set_ylabel("Throughput (Tokens/sec)", fontweight='bold')
        ax.set_xlabel("Precision", fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("fig2_quantization.pdf")
        print("Saved fig2_quantization.pdf")

if __name__ == "__main__":
    plot_all()