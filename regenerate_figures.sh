#!/usr/bin/env bash
# regenerate_figures.sh — rebuilds every paper figure from current data.
#
# Run order matters: each script writes to a unique output PDF, so order does
# not matter for correctness, but the order below mirrors the paper's figure
# numbering for log readability.
#
# Inputs:
#   final_results_with_coldload.csv  (produced by run_experiments.py)
#   duplex_timeline_data.csv         (produced by generate_timeline_data.py)
#
# Outputs: figures/*.pdf

set -euo pipefail
cd "$(dirname "$0")"

# Activate the project venv if it exists
if [[ -f CXL_venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source CXL_venv/bin/activate
fi

echo "=========================================="
echo "[1/3] cache_ablation_study.py  -> Fig 1(b)"
echo "=========================================="
python cache_ablation_study.py

echo
echo "=========================================="
echo "[2/3] plot_paper_figures.py    -> Fig 1(a), 4, 5, 6, 7, 8 + supporting"
echo "=========================================="
python plot_paper_figures.py

echo
echo "=========================================="
echo "[3/3] plot_timeline.py         -> supplementary timeline"
echo "=========================================="
if [[ -f duplex_timeline_data.csv ]]; then
    python plot_timeline.py
else
    echo "  Skipped: duplex_timeline_data.csv not found."
    echo "  Run 'python generate_timeline_data.py' first if needed."
fi

echo
echo "✓ All figures regenerated in ./figures/"
ls -la figures/*.pdf
