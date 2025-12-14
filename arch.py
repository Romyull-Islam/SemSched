import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # --- Colors ---
    host_color = "#e3f2fd"   # Light Blue
    logic_color = "#fce4ec"  # Pink
    dev_color = "#e8f5e9"    # Light Green
    dram_color = "#c8e6c9"   # Green
    nand_color = "#ffe0b2"   # Orange
    read_arrow = "#1565c0"   # Dark Blue
    write_arrow = "#c62828"  # Dark Red

    # ==========================================
    # 1. HOST SYSTEM (Top Section)
    # ==========================================
    # Host Container
    ax.add_patch(patches.FancyBboxPatch((1, 6.5), 10, 3, boxstyle="round,pad=0.1", 
                                        linewidth=2, edgecolor='black', facecolor=host_color, alpha=0.3))
    ax.text(1.2, 9.2, "Host System (Intel Xeon 6315P)", fontsize=12, fontweight='bold')

    # CPU (Using FancyBboxPatch for rounded corners)
    ax.add_patch(patches.FancyBboxPatch((4.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1", fc="white", ec="black"))
    ax.text(6, 8.25, "CPU Cores\n(AVX2 Compute)\n4 Cores / 2.8 GHz", ha="center", va="center", fontsize=10)

    # Host DRAM
    ax.add_patch(patches.Rectangle((1.5, 7.5), 2, 1.5, fc="#bbdefb", ec="black"))
    ax.text(2.5, 8.25, "Host DRAM\n(32 GB)\n\nTarget:\nAttention Layers", ha="center", va="center", fontsize=9)

    # Duplex Logic (Software Scheduler)
    ax.add_patch(patches.Rectangle((8.0, 7.0), 2.5, 2.0, fc=logic_color, ec="black", linestyle="--"))
    ax.text(9.25, 8.7, "Duplex Scheduler", ha="center", fontweight='bold', fontsize=9)
    ax.text(9.25, 8.1, "• Traffic Monitor\n• Synth Injector\n• IO Thread Pool", ha="center", fontsize=8.5)

    # ==========================================
    # 2. CXL INTERFACE (Middle Section)
    # ==========================================
    # Read Channel Arrow (Up)
    ax.arrow(6.2, 6.4, 0, -2.4, width=0.15, head_width=0.4, head_length=0.3, fc=read_arrow, ec="black")
    ax.text(6.5, 5.2, "Read Channel\n(Param Fetch ~27 GB/s)", color=read_arrow, fontsize=9, fontweight='bold')

    # Write Channel Arrow (Down) - Offset slightly left
    ax.arrow(5.8, 4.0, 0, 2.4, width=0.15, head_width=0.4, head_length=0.3, fc=write_arrow, ec="black")
    ax.text(5.5, 5.2, "Write Channel\n(KV + Synthetic)", color=write_arrow, fontsize=9, fontweight='bold', ha='right')
    
    # Bus Label
    ax.text(9.25, 6.0, "CXL 2.0/3.0 Type 3\nFull Duplex Bus", ha="center", 
            bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

    # ==========================================
    # 3. CXL DEVICE (Bottom Section)
    # ==========================================
    # Device Container
    ax.add_patch(patches.FancyBboxPatch((1, 0.5), 10, 3.5, boxstyle="round,pad=0.1", 
                                        linewidth=2, edgecolor='black', facecolor=dev_color, alpha=0.3))
    ax.text(1.2, 3.7, "CXL Hybrid Device (Samsung CMM-H)", fontsize=12, fontweight='bold')

    # Device Controller
    ax.add_patch(patches.Rectangle((4, 2.8), 4, 0.8, fc="#eeeeee", ec="black"))
    ax.text(6, 3.2, "Device Controller\n(Attn-Guided Cache Logic)", ha="center", va="center", fontsize=9)

    # Device DRAM
    ax.add_patch(patches.Rectangle((2, 1.0), 3, 1.5, fc=dram_color, ec="black"))
    ax.text(3.5, 1.75, "Device DRAM\n(64 GB Cache)\n\nTarget:\nHigh-Sparsity MLPs", ha="center", va="center", fontsize=9)

    # NAND Flash
    ax.add_patch(patches.Rectangle((7, 1.0), 3, 1.5, fc=nand_color, ec="black"))
    ax.text(8.5, 1.75, "NAND Flash\n(256 GB Cold)\n\nTarget:\nCold Layers", ha="center", va="center", fontsize=9)

    # Internal Connectors (Controller to Memory)
    ax.plot([6, 3.5], [2.8, 2.5], color="black", linewidth=2) # Ctrl to DRAM
    ax.plot([6, 8.5], [2.8, 2.5], color="black", linewidth=2) # Ctrl to NAND
    
    # Prefetch Arrow (NAND -> DRAM)
    ax.annotate("", xy=(5.1, 1.75), xytext=(6.9, 1.75), 
                arrowprops=dict(arrowstyle="->", lw=3, color="purple", ls="--"))
    ax.text(6.0, 1.9, "Prefill / Warmup", color="purple", ha="center", fontsize=9, fontweight='bold')

    # ==========================================
    # 4. LOGIC CONNECTIONS (Curved Lines)
    # ==========================================
    # Logic to Bus
    # Draw a line from Scheduler to the Write Channel to show control
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="gray", alpha=0.5, linestyle="--")
    
    # Scheduler -> Write Channel control
    a1 = patches.FancyArrowPatch((8.0, 8.0), (5.9, 5.5), connectionstyle="arc3,rad=-0.3", **kw)
    ax.add_patch(a1)
    
    # Title
    plt.title("Semantic-Aware Duplex CXL Simulator Architecture", fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300)
    print("Successfully generated 'architecture_diagram.png'")
    # plt.show() # Uncomment if you have a display connected

if __name__ == "__main__":
    draw_architecture()