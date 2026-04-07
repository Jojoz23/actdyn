#!/usr/bin/env bash
set -euo pipefail

# Run a compact eval matrix for runs/actdyn_square_strong and snapshot outputs.
# Usage:
#   ./scripts/eval_square_matrix.sh [CHECKPOINT] [CONFIG]
#
# Defaults:
#   CHECKPOINT = runs/actdyn_square_strong/checkpoints/best.pt
#   CONFIG     = configs/square_lowdim.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CKPT="${1:-runs/actdyn_square_strong/checkpoints/best.pt}"
CFG="${2:-configs/square_lowdim.yaml}"
TE_CFG="configs/square_lowdim_te_only.yaml"
RUN_DIR="runs/actdyn_square_strong"
EVAL_DIR="$RUN_DIR/eval"
AB_DIR="$RUN_DIR/eval_ablations"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "Config not found: $CFG" >&2
  exit 1
fi
if [[ ! -f "$TE_CFG" ]]; then
  echo "TE-only config not found: $TE_CFG" >&2
  exit 1
fi

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  if [[ -d "$ROOT_DIR/.venv-rollout" ]]; then
    export UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout"
  else
    echo "No UV_PROJECT_ENVIRONMENT set and $ROOT_DIR/.venv-rollout not found." >&2
    echo "Create a rollout env first: uv venv .venv-rollout --python 3.13 && UV_PROJECT_ENVIRONMENT=.venv-rollout uv sync --extra rollout" >&2
    exit 1
  fi
fi

mkdir -p "$AB_DIR"

run_case() {
  local tag="$1"
  local cfg_path="$2"
  shift 2
  echo ""
  echo "==== Running: $tag ===="
  uv run --extra rollout python -m actdyn.eval --config "$cfg_path" --checkpoint "$CKPT" "$@"
  rm -rf "$AB_DIR/$tag"
  cp -r "$EVAL_DIR" "$AB_DIR/$tag"
  echo "Saved snapshot: $AB_DIR/$tag"
}

# 1) Paper-style ACT deployment default: z=0 (deterministic prior)
run_case "z0_default_modes" "$CFG" --no-sample-latent

# 2) Stochastic latent at rollout (tests posterior/prior mismatch sensitivity)
run_case "sampled_latent_default_modes" "$CFG"

# 3) Temporal ensembling only, z=0 (ACT Algorithm 2-style comparator)
run_case "z0_temporal_ensemble_only" "$TE_CFG" --no-sample-latent

echo ""
echo "All snapshots written under: $AB_DIR"
