"""
cache_ablation_study.py — generates Fig 1(b) "Caching Algorithm Comparison".

Output:
    figures/fig_motivation_comprehensive.pdf

This script simulates four caching strategies (Blind LRU, Frequency-LFU,
Static-Tiering, Semantic-Pinning) under capacity pressure and reports the
per-token stall time per strategy at 72B across FP32/FP16/INT8/INT4. The
numbers are derived from the layer-cache model in this file (not from the
main run_experiments.py CSV).

Sibling scripts:
    plot_paper_figures.py — main paper figures (driven by final_results_with_coldload.csv)
    plot_timeline.py       — per-token bus timeline figure (driven by duplex_timeline_data.csv)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import os
from collections import OrderedDict, defaultdict

# Force TrueType font embedding (Required for IEEE/ACM PDF compliance)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from tiers import GiB, CXL_DRAM, CXL_SSD_NAND, transfer_time_s
from model_cfg import build_layers, Qwen2_5_72BCfg
from sim_cfg import (cxl_dev_dram_capacity_bytes, cpu_freq_hz, cpu_cores,
                     flops_per_cycle_per_core, parallel_efficiency)


# ==========================================
# CONSTANTS
# ==========================================
SERVING_BATCH = 128
P_SILU        = 0.46

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family']     = 'serif'
plt.rcParams['font.serif']      = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size']       = 20
plt.rcParams['axes.labelsize']  = 18
plt.rcParams['axes.titlesize']  = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

COLORS_ALGO = {
    'Blind LRU':        '#95a5a6',
    'Frequency-LFU':    '#8e44ad',
    'Static-Tiering':   '#377eb8',
    'Semantic-Pinning': '#e41a1c'
}
COLORS_BASE = {'FlexGen': '#7f7f7f', 'LIA': '#377eb8', 'LLMFlash': '#ff7f0e'}
MARKERS     = {'FlexGen': 'o',       'LIA': 's',       'LLMFlash': '^'}

MODEL_SIZES = {
    "Mistral7BCfg":   7,
    "Llama13BCfg":    13,
    "Qwen3_20BCfg":   20,
    "Qwen2_5_72BCfg": 72,
}


# ==========================================
# HELPERS
# ==========================================
def compute_time_s(flops):
    if flops <= 0:
        return 0.0
    return flops / (cpu_freq_hz * cpu_cores
                    * flops_per_cycle_per_core * parallel_efficiency)


def miss_penalty(L):
    sz       = L["bytes"]
    sparsity = L.get("sparsity", 0.0)
    flops    = L.get("flops", 0)
    comp_t   = compute_time_s(flops * (1.0 - sparsity))
    nand_t   = transfer_time_s(sz, CXL_SSD_NAND)
    cxl_t    = transfer_time_s(sz, CXL_DRAM)
    return max(0.0, max(comp_t, nand_t) - max(comp_t, cxl_t))


def get_tight_cache(total_model_bytes, cache_cap, batch_size=SERVING_BATCH):
    """FIX 1: 50% fill ratio forces real eviction trade-offs."""
    active_frac  = 1.0 - (1.0 - P_SILU) ** batch_size
    effective_ws = total_model_bytes * active_frac
    return min(cache_cap, int(effective_ws * 0.50))   # was 0.75


# ==========================================
# LRU CACHE (used only by Blind LRU path)
# ==========================================
class LRUCache:
    def __init__(self, capacity):
        self.capacity     = capacity
        self.cache        = OrderedDict()
        self.current_size = 0

    def access(self, idx, size):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return True                         # hit
        while self.current_size + size > self.capacity and self.cache:
            _, evicted_size = self.cache.popitem(last=False)
            self.current_size -= evicted_size
        if self.current_size + size <= self.capacity:
            self.cache[idx] = size
            self.current_size += size
        return False                            # miss


# ==========================================
# CACHING SIMULATION  (all four strategies)
# ==========================================
def simulate_cache_behavior(strategy_name, layers, cache_cap,
                             batch_size=SERVING_BATCH):
    num_tokens = 16

    # ------------------------------------------------------------------
    # Blind LRU — fully dynamic eviction, no semantic awareness
    # ------------------------------------------------------------------
    if strategy_name == "Blind LRU":
        lru   = LRUCache(cache_cap)
        stall = 0.0
        for _ in range(num_tokens):
            for idx, L in enumerate(layers):
                if not lru.access(idx, L["bytes"]):
                    stall += miss_penalty(L)
        return stall / num_tokens

    # ------------------------------------------------------------------
    # Static-Tiering — sequential fill: first N layers until cache full
    # ------------------------------------------------------------------
    elif strategy_name == "Static-Tiering":
        pinned, used = set(), 0
        for i, L in enumerate(layers):
            if used + L["bytes"] <= cache_cap:
                pinned.add(i)
                used += L["bytes"]
        stall = 0.0
        for _ in range(num_tokens):
            for idx, L in enumerate(layers):
                if idx not in pinned:
                    stall += miss_penalty(L)
        return stall / num_tokens

    # ------------------------------------------------------------------
    # FIX 3 — Frequency-LFU: real LFU with dynamic eviction
    # In sequential decode ALL layers are accessed equally (freq always
    # equal after one pass), so LFU correctly degenerates to near-FIFO —
    # this IS the accurate model of LFU under sequential access.
    # ------------------------------------------------------------------
    elif strategy_name == "Frequency-LFU":
        freq      = defaultdict(int)
        lfu_cache = {}          # idx → bytes
        used      = 0
        stall     = 0.0
        for _ in range(num_tokens):
            for idx, L in enumerate(layers):
                freq[idx] += 1
                if idx in lfu_cache:
                    continue                    # hit — no stall
                # evict least-frequent entry until enough space
                while used + L["bytes"] > cache_cap and lfu_cache:
                    evict      = min(lfu_cache, key=lambda k: freq[k])
                    used      -= lfu_cache.pop(evict)
                if used + L["bytes"] <= cache_cap:
                    lfu_cache[idx] = L["bytes"]
                    used          += L["bytes"]
                stall += miss_penalty(L)        # miss
        return stall / num_tokens

    # ------------------------------------------------------------------
    # FIX 2 — Semantic-Pinning: rank by penalty DENSITY (penalty/bytes)
    # Raw miss_penalty favoured tiny attention layers even when the
    # cache would hold far more total stall reduction by pinning a
    # larger high-sparsity MLP instead.  Density = value per byte.
    # ------------------------------------------------------------------
    elif strategy_name == "Semantic-Pinning":
        # rank by raw miss_penalty (∝ bytes) → largest NAND-expensive layers first
        # this is optimal: maximises total stall reduction within cache budget
        ranked = sorted(
            range(len(layers)),
            key=lambda i: miss_penalty(layers[i]),   # ← remove the / max(bytes,1)
            reverse=True
        )
        pinned, used = set(), 0
        for i in ranked:
            if used + layers[i]["bytes"] <= cache_cap:
                pinned.add(i)
                used += layers[i]["bytes"]
        stall = 0.0
        for _ in range(num_tokens):
            for idx, L in enumerate(layers):
                if idx not in pinned:
                    stall += miss_penalty(L)
        return stall / num_tokens



# ==========================================
# MAIN FIGURE
# ==========================================
def plot_motivation_comprehensive():
    os.makedirs("figures", exist_ok=True)

    sub_wall  = pd.DataFrame()
    wall_mode = ""
    baselines = ["FlexGen", "LIA", "LLMFlash"]

    if os.path.exists("final_results_with_coldload.csv"):
        df_main = pd.read_csv("final_results_with_coldload.csv")

        name_map = {
            "flexgen_baseline.py":  "FlexGen",
            "lia_baseline.py":      "LIA",
            "llmflash_baseline.py": "LLMFlash",
            "flashllm_baseline.py": "LLMFlash",
        }
        df_main["Simulator"] = df_main["Simulator"].replace(name_map)

        if "BatchSize" in df_main.columns:
            df_main["BatchSize"] = pd.to_numeric(
                df_main["BatchSize"], errors='coerce').astype("Int64")

        def get_fp32_scalability(df, batch):
            mask = (
                (df["Experiment"] == "Scalability") &
                (df["Simulator"].isin(baselines)) &
                (df["TPS"] > 0)
            )
            if "Quant" in df.columns:
                mask &= (df["Quant"].str.lower() == "fp32")
            if "BatchSize" in df.columns:
                mask &= (df["BatchSize"] == batch)
            return df[mask].copy()

        sc = get_fp32_scalability(df_main, SERVING_BATCH)
        if sc.empty:
            sc        = get_fp32_scalability(df_main, 1)
            wall_mode = "Scalability, FP32, B=1 (no B=128 data)"
        else:
            wall_mode = f"Scalability, FP32, B={SERVING_BATCH}"

        if not sc.empty:
            sc["ModelSize"] = (
                sc["Model"].map(MODEL_SIZES)
                .fillna(pd.to_numeric(sc["Model"], errors='coerce'))
            )
            sub_wall = sc

    # ── Panel (b): Caching Ablation ───────────────────────────────────
    quants           = {"FP32": 4.0, "FP16": 2.0, "INT8": 1.0, "INT4": 0.5}
    ablation_results = []

    print(f"\n{'='*70}")
    print(f"CACHING ABLATION (72B, B={SERVING_BATCH})")
    print(f"{'='*70}")

    for q_name, bpp in quants.items():
        layers = build_layers(Qwen2_5_72BCfg())
        n      = len(layers)
        for i, L in enumerate(layers):
            raw_bytes      = int((L["bytes"] / 4.0) * bpp)
            raw_flops      = L.get("flops", 0)
            L["type"]      = "attention" if i % 2 == 0 else "mlp"
            L["sparsity"]  = 0.13 + (0.956 - 0.13) * (i / max(n - 1, 1))
            if L["type"] == "attention":
                L["bytes"] = int(raw_bytes * 0.17 * 2)
                L["flops"] = int(raw_flops * 0.33)
            else:
                L["bytes"] = int(raw_bytes * 0.83 * 2)
                L["flops"] = int(raw_flops * 0.67)

        total_model_bytes = sum(L["bytes"] for L in layers)
        tight_cache = get_tight_cache(total_model_bytes,
                                      cxl_dev_dram_capacity_bytes,
                                      batch_size=SERVING_BATCH)

        print(f"\n[ {q_name} | {total_model_bytes//GiB}GB model "
              f"| Cache: {tight_cache//GiB}GB ]")

        for strat in ["Blind LRU", "Frequency-LFU",
                      "Static-Tiering", "Semantic-Pinning"]:
            stall = simulate_cache_behavior(strat, layers, tight_cache,
                                            batch_size=SERVING_BATCH)
            ablation_results.append({'Precision': q_name,
                                     'Algorithm': strat,
                                     'Stall_s':   stall})
            print(f"  {strat:<20}: {stall:.3f}s")

    df_ablation = pd.DataFrame(ablation_results)

    # ── PLOT ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 5.5),
                                   gridspec_kw={'width_ratios': [1, 1.4]})

    # ── Panel (a) annotation loop — FIXED ────────────────────────────
    if not sub_wall.empty:
        # 1. Plot the lines first
        for sim in baselines:
            sd = sub_wall[sub_wall["Simulator"] == sim].sort_values("ModelSize")
            if sd.empty:
                continue
            ax1.plot(sd["ModelSize"], sd["TPS"],
                     color=COLORS_BASE.get(sim, 'black'), marker=MARKERS.get(sim, 'o'),
                     markersize=14, linewidth=3, label=sim, zorder=3)

        # 2. Add non-overlapping labels using exact pixel offsets
        sizes = sorted(sub_wall["ModelSize"].dropna().unique())

        # Offset ladder (x_offset, y_offset) in fixed pixels.
        OFFSET_LADDER = [(0, -18), (25, -5), (0, 22), (-28, 0)]

        for x in sizes:
            # Get all data points at this specific Model Size
            df_x = sub_wall[(sub_wall["ModelSize"] == x) & (sub_wall["Simulator"].isin(baselines))]
            if df_x.empty:
                continue
            
            # Sort by throughput so rank 0 is always the bottom point
            df_x = df_x.sort_values("TPS")
            
            for rank, (_, row) in enumerate(df_x.iterrows()):
                sim = row["Simulator"]
                y = row["TPS"]
                color = COLORS_BASE.get(sim, 'black')
                
                # Apply the offset based on vertical rank
                xoff, yoff = OFFSET_LADDER[rank % len(OFFSET_LADDER)]
                
                # Format text: 1 decimal place if >= 1.0, otherwise 3 decimal places
                label_text = f'{y:.1f}' if y >= 1.0 else f'{y:.3f}'
                
                ax1.annotate(
                    label_text,
                    xy=(x, y),
                    xytext=(xoff, yoff),
                    textcoords='offset points',
                    ha='center',
                    va='center',
                    fontsize=12,
                    fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=color, alpha=0.9),
                    zorder=4
                )

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        sizes = sorted(sub_wall["ModelSize"].dropna().unique())
        ax1.set_xticks(sizes)
        ax1.set_xticklabels([f"{int(s)}B" for s in sizes])
        
        # -------------------------------------------------------------
        # NEW FIX: Explicitly expand the bounding box to prevent cut-offs
        # -------------------------------------------------------------
        y_vals = sub_wall[sub_wall["Simulator"].isin(baselines)]["TPS"]
        if not y_vals.empty:
            ax1.set_xlim(min(sizes) * 0.9, max(sizes) * 1.2)  # 20% room left, 40% room right
            ax1.set_ylim(y_vals.min() * 0.6, y_vals.max() * 2) # 50% room bottom, 150% room top
        # -------------------------------------------------------------
        
        ax1.set_xlabel("Model Size (Parameters)")
        ax1.set_ylabel("Decode Throughput (Tokens/s)")
        ax1.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f'{y:.0f}'))
        ax1.legend(loc='upper right', fontsize=18)
        ax1.grid(True, which="both", ls="--", alpha=0.3)
    else:
        ax1.text(0.5, 0.5,
                 "No FP32 Scalability data found.\n"
                 "Run Scalability experiment first.",
                 ha='center', va='center',
                 transform=ax1.transAxes, fontsize=16, color='red')

    ax1.set_title(f"(a) The Latency Wall({wall_mode}, 32H+64C)",
                  fontsize=20)

    # Panel (b)
    sns.barplot(data=df_ablation, x="Precision", y="Stall_s",
                hue="Algorithm", palette=COLORS_ALGO,
                edgecolor='black', ax=ax2)
    ax2.set_title(f"(b) Caching Algorithm Comparison"
                  f"(72B, B={SERVING_BATCH}, 32H+64C)", fontsize=20)
    ax2.set_ylabel("Avg. Stall per Token (s)")
    ax2.set_xlabel("Quantization Precision")
    ax2.set_ylim(0, df_ablation['Stall_s'].max() * 1.3)
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%.2fs', padding=10,
                      fontsize=16, rotation=90)

    plt.tight_layout()
    plt.savefig("figures/fig_motivation_comprehensive.pdf",
                bbox_inches='tight')
    print(f"\n>>> Saved: figures/fig_motivation_comprehensive.pdf"
          f"  (B={SERVING_BATCH})")


if __name__ == "__main__":
    plot_motivation_comprehensive()