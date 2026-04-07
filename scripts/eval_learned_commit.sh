#!/usr/bin/env bash
# Roll out learned_commit mode with the CORRECT task dataset (avoids env/dataset mismatch).
#
# Usage:
#   ./scripts/eval_learned_commit.sh lift
#   ./scripts/eval_learned_commit.sh can
#   ./scripts/eval_learned_commit.sh square
#
# Requires: checkpoint at runs/actdyn_{task}_learned/checkpoints/best.pt
#           (train first with configs/lift_lowdim_learned.yaml + matching --dataset)

set -euo pipefail

TASK="${1:?usage: $0 lift|can|square}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="$ROOT/configs/lift_lowdim_learned.yaml"

case "$TASK" in
  lift)
    DATA="$ROOT/datasets/lift/ph/low_dim_v15.hdf5"
    RUN_ROOT="$ROOT/runs/actdyn_lift_learned"
    ;;
  can)
    DATA="$ROOT/datasets/can/ph/low_dim_v15.hdf5"
    RUN_ROOT="$ROOT/runs/actdyn_can_learned"
    ;;
  square)
    DATA="$ROOT/datasets/square/ph/low_dim_v15.hdf5"
    RUN_ROOT="$ROOT/runs/actdyn_square_learned"
    ;;
  *)
    echo "Unknown task: $TASK (use lift, can, or square)" >&2
    exit 1
    ;;
esac

CKPT="$RUN_ROOT/checkpoints/best.pt"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  echo "Train first: uv run actdyn-train --config $CFG --dataset $DATA --out-dir $RUN_ROOT" >&2
  exit 1
fi
if [[ ! -f "$DATA" ]]; then
  echo "Missing dataset: $DATA" >&2
  exit 1
fi

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  if [[ -d "$ROOT/.venv-rollout" ]]; then
    export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-rollout"
  else
    echo "No UV_PROJECT_ENVIRONMENT set and $ROOT/.venv-rollout not found." >&2
    echo "Create a rollout venv for MuJoCo, for example:" >&2
    echo "  cd $ROOT && uv venv .venv-rollout --python 3.13" >&2
    echo "  UV_PROJECT_ENVIRONMENT=$ROOT/.venv-rollout uv sync --extra rollout" >&2
    exit 1
  fi
fi

# Config YAML hardcodes logging.out_dir to lift; override so each task writes under its own RUN_ROOT.
uv run --extra rollout python -m actdyn.eval \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --dataset "$DATA" \
  --out-dir "$RUN_ROOT" \
  --no-sample-latent \
  --allow-partial-checkpoint

echo "Done. See $RUN_ROOT/eval/learned_commit/summary.json"
