#!/usr/bin/env bash
# Train seeds 1-5 (no DAPS), save one NPZ per seed, then one PR figure with 5 curves.
# DAPS (α) is applied at plot time from raw existence scores.
set -euo pipefail
cd "$(dirname "$0")/.."

FIG_DIR="${FIG_DIR:-figures}"
ALPHA="${EDGE_DEV_ALPHA:-0.1}"
SEEDS="${SEEDS:-1 2 3 4 5}"
PY="${PY:-python}"

mkdir -p "$FIG_DIR"

for s in $SEEDS; do
  echo "========== seed $s =========="
  $PY -m run -t 1 --seed "$s" -d 0.5 -e GCN -r GRU -i DETECT -w 3 -g 1 \
    --patience 10 --lr 0.005 --fpweight 0.6 \
    --save-test-scores "$FIG_DIR/baseline_scores_seed${s}.npz"
  if [[ -f model_save.pkl ]]; then
    cp -f model_save.pkl "$FIG_DIR/model_save_seed${s}.pkl"
  fi
done

echo "========== plotting 5 PR curves =========="
BASE_ARGS=()
for s in $SEEDS; do
  BASE_ARGS+=(--base-scores "$FIG_DIR/baseline_scores_seed${s}.npz")
done

$PY -m loaders.degree_check --no-figures --pr-curve --pr-per-seed \
  --edge-dev-alpha "$ALPHA" \
  --fig-dir "$FIG_DIR" \
  "${BASE_ARGS[@]}"

echo "Done."
echo "  NPZs: $FIG_DIR/baseline_scores_seed{1..5}.npz"
echo "  Figure: $FIG_DIR/fig3_pr_curve_per_seed.png"
