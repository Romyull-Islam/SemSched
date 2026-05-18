"""
plot_sharegpt.py — generate the ShareGPT validation figure for §V-F.
Two panels:
  (a) CDF of SemSched/LLMFlash speedup across 50 ShareGPT prompts
  (b) Scatter of speedup vs prefill length
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "trace_workload/sharegpt_n50.csv")
OUT = os.path.join(REPO, "figures/fig_sharegpt.pdf")

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['font.size'] = 13


def main():
    df = pd.read_csv(CSV)
    piv = df.pivot_table(index='pair_idx', columns='Simulator', values='Decode_TPS')
    piv['prefill'] = df[df.Simulator=='LLMFlash'].set_index('pair_idx')['prefill_len']
    piv['speedup'] = piv['SemSched'] / piv['LLMFlash']
    piv = piv.dropna().sort_values('prefill')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 2.82))

    # Panel (a): CDF
    sorted_su = np.sort(piv['speedup'].values)
    cdf_y = np.arange(1, len(sorted_su)+1) / len(sorted_su) * 100
    ax1.plot(sorted_su, cdf_y, color='#e41a1c', linewidth=2.5)
    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.6, label='Parity')
    ax1.axvline(sorted_su[len(sorted_su)//2], color='#377eb8',
                linestyle=':', alpha=0.8,
                label=f'Median = {np.median(sorted_su):.2f}$\\times$')
    ax1.set_xlabel("SemSched / LLMFlash decode speedup", fontsize=13)
    ax1.set_ylabel("Cumulative % of prompts", fontsize=13)
    ax1.set_title("(a) Speedup CDF over 50 ShareGPT prompts", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.set_xlim(0.9, 2.2)
    ax1.set_ylim(0, 105)

    # Panel (b): Scatter vs prefill
    ax2.scatter(piv['prefill'], piv['speedup'], c='#e41a1c',
                edgecolor='black', s=50, alpha=0.7)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.6)
    ax2.set_xlabel("Prefill length (tokens)", fontsize=13)
    ax2.set_ylabel("SemSched / LLMFlash speedup", fontsize=13)
    ax2.set_title("(b) Speedup vs. prompt length", fontsize=13)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches='tight')
    print(f"Saved {OUT}")
    print(f"  Median speedup: {piv['speedup'].median():.2f}x")
    print(f"  P10: {piv['speedup'].quantile(0.1):.2f}x, "
          f"P90: {piv['speedup'].quantile(0.9):.2f}x")
    print(f"  Win-rate: {(piv['speedup']>1.0).sum()}/{len(piv)}")


if __name__ == "__main__":
    main()
