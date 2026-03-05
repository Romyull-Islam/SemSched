import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# ==========================================
# CONSTANTS & STYLING
# ==========================================
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

COLOR_READ  = '#377eb8'
COLOR_WRITE = '#e41a1c'
MIN_VISUAL_WIDTH_S = 0.005

# ==========================================
# PLOTTING SCRIPT
# ==========================================
def plot_timeline_comparison():
    csv_file = "duplex_timeline_data.csv"
    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}'.")
        return

    # FIX 1: Removed redundant pd.DataFrame() wrapper
    df = pd.read_csv(csv_file)
    df = df[df["Token"] == 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ---------------------------------------------------------
    # PANEL (b) - BASELINE SIMPLEX (Top Plot)
    # ---------------------------------------------------------
    df_base = df[df["Simulator"] == "Baseline_Simplex"]

    # FIX 2: Use sorted read-only rows directly with step='pre' — no manual doubling
    df_base_read = df_base[df_base["Bandwidth_GBps"] > 0].sort_values("Start_Time_s")

    ax1.fill_between(
        df_base_read["Start_Time_s"],
        0,
        df_base_read["Bandwidth_GBps"],
        step='pre',
        color=COLOR_READ, alpha=0.8,
        label="Weight Read (CXL Rx Lane)"
    )

    # FIX 3 & 4: Use actual BW magnitude from data instead of hardcoded 27.0
    for _, row in df_base[df_base["Event"] == "KV_Write_Stall"].iterrows():
        vis_width = max(row["End_Time_s"] - row["Start_Time_s"], MIN_VISUAL_WIDTH_S)
        bw_height = abs(row["Bandwidth_GBps"]) if row["Bandwidth_GBps"] != 0 else 27.0
        ax1.add_patch(patches.Rectangle(
            (row["Start_Time_s"], 0), vis_width, bw_height,
            color=COLOR_WRITE, alpha=0.9
        ))

    patch_write_stall = patches.Patch(
        color=COLOR_WRITE, alpha=0.9, label='KV Write Stall (Lane Blocked)'
    )

    ax1.set_title(
        "(b) Baseline Simplex: Read Stream Fragmented by Sequential KV-Cache Write Stalls",
        loc='left'
    )
    ax1.set_ylabel("CXL Bandwidth\n(GB/s)")
    ax1.set_ylim(-35, 35)
    ax1.axhline(0, color='black', linewidth=1.5)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    handles1, _ = ax1.get_legend_handles_labels()
    handles1.append(patch_write_stall)
    ax1.legend(handles=handles1, loc="lower right", framealpha=1)

    # ---------------------------------------------------------
    # PANEL (c) - SEMDUPLEX (Bottom Plot)
    # ---------------------------------------------------------
    df_sem = df[df["Simulator"] == "SemDuplex"]

    # FIX 5: Use sorted read rows directly with step='pre' — no manual doubling
    df_sem_read = df_sem[df_sem["Event"].str.contains("Read")].sort_values("Start_Time_s")

    ax2.fill_between(
        df_sem_read["Start_Time_s"],
        0,
        df_sem_read["Bandwidth_GBps"],
        step='pre',
        color=COLOR_READ, alpha=0.8,
        label="Weight Read (Continuous)"
    )

    # FIX 6: Use actual BW magnitude from data instead of hardcoded -27.0
    df_sem_write = df_sem[df_sem["Event"] == "KV_Write_Background"].sort_values("Start_Time_s")
    for _, row in df_sem_write.iterrows():
        vis_width = max(row["End_Time_s"] - row["Start_Time_s"], MIN_VISUAL_WIDTH_S)
        bw_depth = abs(row["Bandwidth_GBps"]) if row["Bandwidth_GBps"] != 0 else 27.0
        ax2.add_patch(patches.Rectangle(
            (row["Start_Time_s"], -bw_depth), vis_width, bw_depth,
            color=COLOR_WRITE, alpha=0.9
        ))

    patch_write_bg = patches.Patch(
        color=COLOR_WRITE, alpha=0.9, label='KV Write (Parallel Tx Lane)'
    )

    ax2.set_title(
        "(c) SemDuplex: Continuous Read Stream via Parallel Write Lane Injection",
        loc='left'
    )
    ax2.set_ylabel("CXL Bandwidth\n(GB/s)")
    ax2.set_xlabel("Inference Timeline (Seconds) — Zoomed in on Token 1 (First 5 Layers)")
    ax2.set_ylim(-35, 35)
    ax2.axhline(0, color='black', linewidth=1.5)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    handles2, _ = ax2.get_legend_handles_labels()
    handles2.append(patch_write_bg)
    ax2.legend(handles=handles2, loc="lower right", framealpha=1)

    # ---------------------------------------------------------
    # FINAL FORMATTING & SAVE
    # ---------------------------------------------------------
    # FIX 7: Set xlim once; sharex propagates it to ax2 automatically
    ax1.set_xlim(0, 0.75)

    # FIX 8: Use tight_layout instead of manual subplots_adjust for robust spacing
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    save_path = "figures/fig_stall_duplex_phy.pdf"
    plt.savefig(save_path, bbox_inches='tight')

    # FIX 10: Close figure to free memory
    plt.close(fig)

    print(f"\n>>> Successfully generated Figure 6 (Zoomed View)!")
    print(f">>> Saved to: {save_path}")

if __name__ == "__main__":
    plot_timeline_comparison()
