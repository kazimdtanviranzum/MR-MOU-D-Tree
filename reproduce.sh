#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# reproduce.sh - one command reproduction of every result in the manuscript.
#
#   bash reproduce.sh
#
# Regenerates, in order:
#   equations/   eq01..eq30 LaTeX + preview images
#   data/        scenario centroids, simulation outcomes, population objectives
#   figures/     fig01..fig18
#   tables/      all result and parameter tables (CSV)
#   results/     results_summary.json (machine readable headline numbers)
#   manuscript/  MR_MOU_D_Tree_manuscript.docx (native editable equations)
#
# Everything is seeded (GLOBAL_SEED and SPLIT_SEED in src/config.py), so a rerun
# on the same package versions reproduces the published numbers exactly.
#
# Set MRMOU_QUICK=1 to run a reduced-size smoke test in seconds instead of the
# full design (used for development only; it does NOT reproduce the manuscript).
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/src"

echo "[1/3] rendering equations"
python render_equations.py

echo "[2/3] running full pipeline (simulation, metamodel, decisions, bootstraps)"
python run_pipeline.py

echo "[3/3] building manuscript"
python build_manuscript.py

echo
echo "Done. Key outputs:"
echo "  results/results_summary.json"
echo "  manuscript/MR_MOU_D_Tree_manuscript.docx"
