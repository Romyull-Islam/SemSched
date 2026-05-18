"""
plot_matrix.py — visualize the SemSched vs LLMFlash matrix across
(mem × quant × batch × prefill).

Generates one heatmap per (mem, quant) showing SemSched/LLMFlash speedup,
plus a summary "win region" plot.

Output:
    trace_workload/fig_matrix_heatmap.pdf
    trace_workload/fig_matrix_summary.pdf
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "trace_workload/matrix_full.csv")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['pdf.fonttype'] = 42


def load():
    df = pd.read_csv(CSV)
    # Pivot: speedup of SemSched vs LLMFlash per (mem, quant, batch, prefill)
    sem = df[df.Simulator=='SemSched'].set_index(['mem','quant','batch','prefill'])
    llm = df[df.Simulator=='LLMFlash'].set_index(['mem','quant','batch','prefill'])
    speedup = (sem['Decode_TPS'] / llm['Decode_TPS']).reset_index()
    speedup.rename(columns={'Decode_TPS':'speedup'}, inplace=True)
    speedup.columns = ['mem','quant','batch','prefill','speedup']
    # Also compute speedup over FlexGen (demand paging)
    fg = df[df.Simulator=='FlexGen'].set_index(['mem','quant','batch','prefill'])
    speedup_fg = (sem['Decode_TPS'] / fg['Decode_TPS']).reset_index()
    speedup_fg.columns = ['mem','quant','batch','prefill','speedup_vs_FG']
    return df, speedup, speedup_fg


def heatmap_grid(speedup):
    """One heatmap per (mem, quant). Rows = prefills, cols = batches."""
    mems   = sorted(speedup['mem'].unique())
    quants = ['fp32', 'fp16', 'int8', 'int4']
    quants = [q for q in quants if q in speedup['quant'].unique()]
    fig, axes = plt.subplots(len(mems), len(quants),
                              figsize=(3.5*len(quants), 3.0*len(mems)),
                              squeeze=False)
    for i, mem in enumerate(mems):
        for j, q in enumerate(quants):
            ax = axes[i][j]
            sub = speedup[(speedup.mem==mem) & (speedup.quant==q)]
            pivot = sub.pivot_table(index='prefill', columns='batch', values='speedup')
            im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0.5, vmax=2.0,
                           aspect='auto')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            for ii in range(len(pivot.index)):
                for jj in range(len(pivot.columns)):
                    val = pivot.values[ii, jj]
                    color = 'black' if 0.85 < val < 1.4 else 'white'
                    ax.text(jj, ii, f"{val:.2f}", ha='center', va='center',
                            color=color, fontsize=9)
            ax.set_title(f"{mem}, {q.upper()}", fontsize=11)
            if i == len(mems)-1:
                ax.set_xlabel("Batch size")
            if j == 0:
                ax.set_ylabel("Prefill tokens")
    fig.suptitle("SemSched / LLMFlash decode-TPS ratio (green = win, red = loss)",
                 fontsize=13, y=1.0)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label='Ratio')
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])
    out = os.path.join(REPO, "trace_workload/fig_matrix_heatmap.pdf")
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")


def summary_winrate(speedup):
    """Bar chart: % of cells where SemSched > 1.0× LLMFlash per quant."""
    fig, ax = plt.subplots(figsize=(6, 4))
    summary = speedup.groupby(['mem','quant']).apply(
        lambda g: (g['speedup'] > 1.0).sum() / len(g) * 100
    ).unstack()
    summary = summary.reindex(columns=['fp32','fp16','int8','int4'])
    summary.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
    ax.set_ylabel("% of configurations where SemSched wins")
    ax.set_xlabel("Memory configuration")
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.legend(title="Quant", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xticklabels(summary.index, rotation=0)
    plt.tight_layout()
    out = os.path.join(REPO, "trace_workload/fig_matrix_winrate.pdf")
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved {out}")


def main():
    if not os.path.exists(CSV):
        print(f"No data yet at {CSV}. Wait for matrix_explore.py to finish.")
        return
    df, speedup, speedup_fg = load()
    print(f"Loaded {len(df)} rows")
    print(f"Speedup grid: {len(speedup)} cells")
    print(f"Win count (vs LLMFlash): {(speedup['speedup']>1.0).sum()}/{len(speedup)}")
    print(f"Win count (vs FlexGen):  {(speedup_fg['speedup_vs_FG']>1.0).sum()}/{len(speedup_fg)}")
    print(f"Median speedup vs LLMFlash: {speedup['speedup'].median():.2f}x")
    print(f"P90 speedup vs LLMFlash:    {speedup['speedup'].quantile(0.9):.2f}x")
    heatmap_grid(speedup)
    summary_winrate(speedup)


if __name__ == "__main__":
    main()
