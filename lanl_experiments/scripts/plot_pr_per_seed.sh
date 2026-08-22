#!/usr/bin/env bash
# Plot one PR curve per seed NPZ (DAPS applied at plot time).
set -euo pipefail
cd "$(dirname "$0")/.."

FIG_DIR="${FIG_DIR:-figures}"
ALPHA="${EDGE_DEV_ALPHA:-0.1}"
SEEDS="${SEEDS:-1 2 3 4 5}"
PY="${PY:-python}"

BASE_ARGS=()
for s in $SEEDS; do
  f="$FIG_DIR/baseline_scores_seed${s}.npz"
  [[ -f "$f" ]] || { echo "missing $f"; exit 1; }
  BASE_ARGS+=(--base-scores "$f")
done

$PY -m loaders.degree_check --no-figures --pr-curve --pr-per-seed \
  --edge-dev-alpha "$ALPHA" \
  --fig-dir "$FIG_DIR" \
  "${BASE_ARGS[@]}"

echo "Wrote $FIG_DIR/fig3_pr_curve_per_seed.png"
