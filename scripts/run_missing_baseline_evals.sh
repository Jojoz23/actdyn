#!/usr/bin/env bash
set -euo pipefail

# Run baseline rollout evals with the correct resolved configs / output dirs.
# This avoids terminal paste corruption and prevents writing into wrong folders.
#
# Usage:
#   bash /virtual/csc415user/actdyn/scripts/run_missing_baseline_evals.sh
#
# Optional:
#   INCLUDE_LIFT=0 bash ...   # skip Lift re-eval if you don't want to refresh Lift baseline eval files

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout"

INCLUDE_LIFT="${INCLUDE_LIFT:-1}"

run_eval() {
  local cfg="$1"
  local data="$2"
  local ckpt="$3"
  echo ""
  echo "=== EVAL $(basename "$(dirname "$cfg")") / $(basename "$(dirname "$(dirname "$ckpt")")") ==="
  uv run --extra rollout python -m actdyn.eval \
    --config "$cfg" \
    --dataset "$data" \
    --checkpoint "$ckpt" \
    --no-sample-latent
}

if [[ "$INCLUDE_LIFT" == "1" ]]; then
  echo "Running Lift baseline evals (restore clean Lift files)..."
  run_eval "$ROOT_DIR/runs/lift_bc_mlp/resolved_config.yaml" "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/lift_bc_mlp/checkpoints/best.pt"
  run_eval "$ROOT_DIR/runs/lift_knn_bc/resolved_config.yaml" "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/lift_knn_bc/checkpoints/best.pt"
  run_eval "$ROOT_DIR/runs/lift_bet_style/resolved_config.yaml" "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/lift_bet_style/checkpoints/best.pt"
else
  echo "Skipping Lift baseline evals (INCLUDE_LIFT=$INCLUDE_LIFT)."
fi

echo "Running Can baseline evals..."
run_eval "$ROOT_DIR/runs/can_bc_mlp/resolved_config.yaml" "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/can_bc_mlp/checkpoints/best.pt"
run_eval "$ROOT_DIR/runs/can_knn_bc/resolved_config.yaml" "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/can_knn_bc/checkpoints/best.pt"
run_eval "$ROOT_DIR/runs/can_bet_style/resolved_config.yaml" "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/can_bet_style/checkpoints/best.pt"

echo "Running Square baseline evals..."
run_eval "$ROOT_DIR/runs/square_bc_mlp/resolved_config.yaml" "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/square_bc_mlp/checkpoints/best.pt"
run_eval "$ROOT_DIR/runs/square_knn_bc/resolved_config.yaml" "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/square_knn_bc/checkpoints/best.pt"
run_eval "$ROOT_DIR/runs/square_bet_style/resolved_config.yaml" "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" "$ROOT_DIR/runs/square_bet_style/checkpoints/best.pt"

echo ""
echo "Done. Quick check:"
echo "  ls $ROOT_DIR/runs/lift_bc_mlp/eval $ROOT_DIR/runs/lift_knn_bc/eval $ROOT_DIR/runs/lift_bet_style/eval"
echo "  ls $ROOT_DIR/runs/can_bc_mlp/eval  $ROOT_DIR/runs/can_knn_bc/eval  $ROOT_DIR/runs/can_bet_style/eval"
echo "  ls $ROOT_DIR/runs/square_bc_mlp/eval $ROOT_DIR/runs/square_knn_bc/eval $ROOT_DIR/runs/square_bet_style/eval"
