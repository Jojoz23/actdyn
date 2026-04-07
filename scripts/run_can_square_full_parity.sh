#!/usr/bin/env bash
set -euo pipefail

# Full Lift + Can + Square pipeline (matched parity per task):
# - Main ACT train+eval (strong / lowdim configs)
# - ACT ablation matrix (z0 / sampled / temporal-ensemble-only) via eval_*_matrix.sh
# - Baselines train+eval (bc_mlp / knn_bc / bet_style)
# - Learned-commit train+eval (must pass matching --dataset to actdyn.eval)
# - 3-seed train+eval
#
# Usage:
#   bash scripts/run_can_square_full_parity.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG="$ROOT_DIR/runs/overnight_full_lift_can_square_parity_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== START $(date) ==="
echo "Log: $LOG"

# Environments
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv sync
if [[ ! -d "$ROOT_DIR/.venv-rollout" ]]; then
  uv venv "$ROOT_DIR/.venv-rollout" --python 3.13
fi
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv sync --extra rollout

# Ensure matrix scripts are executable
chmod +x "$ROOT_DIR/scripts/eval_can_matrix.sh"
chmod +x "$ROOT_DIR/scripts/eval_square_matrix.sh"
chmod +x "$ROOT_DIR/scripts/eval_lift_strong_matrix.sh"

###############################################################################
# LIFT: main ACT + matrix + baselines + learned + seeds
###############################################################################
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_lowdim_strong.yaml" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/actdyn_lift_strong"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_lowdim_strong.yaml" \
  --checkpoint "$ROOT_DIR/runs/actdyn_lift_strong/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" "$ROOT_DIR/scripts/eval_lift_strong_matrix.sh" \
  "$ROOT_DIR/runs/actdyn_lift_strong/checkpoints/best.pt" \
  "$ROOT_DIR/configs/lift_lowdim_strong.yaml"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_bc_mlp.yaml" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/lift_bc_mlp"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_knn_bc.yaml" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/lift_knn_bc"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_bet_style.yaml" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/lift_bet_style"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_bc_mlp.yaml" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/lift_bc_mlp/checkpoints/best.pt" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_knn_bc.yaml" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/lift_knn_bc/checkpoints/best.pt" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_bet_style.yaml" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/lift_bet_style/checkpoints/best.pt" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_lowdim_learned.yaml" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/actdyn_lift_learned"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_lowdim_learned.yaml" \
  --checkpoint "$ROOT_DIR/runs/actdyn_lift_learned/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  --no-sample-latent \
  --allow-partial-checkpoint

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run "$ROOT_DIR/scripts/train_seeds.sh" \
  "$ROOT_DIR/configs/lift_lowdim_strong.yaml" \
  "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
  0 1 2

for s in 0 1 2; do
  UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
    --config "$ROOT_DIR/runs/lift_lowdim_strong_seed${s}/resolved_config.yaml" \
    --checkpoint "$ROOT_DIR/runs/lift_lowdim_strong_seed${s}/checkpoints/best.pt" \
    --dataset "$ROOT_DIR/datasets/lift/ph/low_dim_v15.hdf5" \
    --no-sample-latent \
    --allow-partial-checkpoint
done

###############################################################################
# CAN: main ACT + matrix
###############################################################################
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/can_lowdim.yaml" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/actdyn_can_strong"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/can_lowdim.yaml" \
  --checkpoint "$ROOT_DIR/runs/actdyn_can_strong/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" "$ROOT_DIR/scripts/eval_can_matrix.sh" \
  "$ROOT_DIR/runs/actdyn_can_strong/checkpoints/best.pt" \
  "$ROOT_DIR/configs/can_lowdim.yaml"

###############################################################################
# CAN: baselines
###############################################################################
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_bc_mlp.yaml" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/can_bc_mlp"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_knn_bc.yaml" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/can_knn_bc"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_bet_style.yaml" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/can_bet_style"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_bc_mlp.yaml" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/can_bc_mlp/checkpoints/best.pt" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_knn_bc.yaml" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/can_knn_bc/checkpoints/best.pt" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_bet_style.yaml" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/can_bet_style/checkpoints/best.pt" \
  --no-sample-latent

###############################################################################
# CAN: learned-commit + seeds
###############################################################################
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_lowdim_learned.yaml" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/actdyn_can_learned"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_lowdim_learned.yaml" \
  --checkpoint "$ROOT_DIR/runs/actdyn_can_learned/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/actdyn_can_learned" \
  --no-sample-latent \
  --allow-partial-checkpoint

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run "$ROOT_DIR/scripts/train_seeds.sh" \
  "$ROOT_DIR/configs/can_lowdim.yaml" \
  "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  0 1 2

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/runs/can_lowdim_seed0/resolved_config.yaml" \
  --checkpoint "$ROOT_DIR/runs/can_lowdim_seed0/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --no-sample-latent \
  --allow-partial-checkpoint

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/runs/can_lowdim_seed1/resolved_config.yaml" \
  --checkpoint "$ROOT_DIR/runs/can_lowdim_seed1/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --no-sample-latent \
  --allow-partial-checkpoint

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/runs/can_lowdim_seed2/resolved_config.yaml" \
  --checkpoint "$ROOT_DIR/runs/can_lowdim_seed2/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/can/ph/low_dim_v15.hdf5" \
  --no-sample-latent \
  --allow-partial-checkpoint

###############################################################################
# SQUARE: main ACT + matrix
###############################################################################
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/square_lowdim.yaml" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/actdyn_square_strong"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/square_lowdim.yaml" \
  --checkpoint "$ROOT_DIR/runs/actdyn_square_strong/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" "$ROOT_DIR/scripts/eval_square_matrix.sh" \
  "$ROOT_DIR/runs/actdyn_square_strong/checkpoints/best.pt" \
  "$ROOT_DIR/configs/square_lowdim.yaml"

###############################################################################
# SQUARE: baselines
###############################################################################
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_bc_mlp.yaml" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/square_bc_mlp"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_knn_bc.yaml" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/square_knn_bc"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_bet_style.yaml" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/square_bet_style"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_bc_mlp.yaml" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/square_bc_mlp/checkpoints/best.pt" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_knn_bc.yaml" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/square_knn_bc/checkpoints/best.pt" \
  --no-sample-latent

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_bet_style.yaml" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --checkpoint "$ROOT_DIR/runs/square_bet_style/checkpoints/best.pt" \
  --no-sample-latent

###############################################################################
# SQUARE: learned-commit + seeds
###############################################################################
UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run actdyn-train \
  --config "$ROOT_DIR/configs/lift_lowdim_learned.yaml" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/actdyn_square_learned"

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/configs/lift_lowdim_learned.yaml" \
  --checkpoint "$ROOT_DIR/runs/actdyn_square_learned/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --out-dir "$ROOT_DIR/runs/actdyn_square_learned" \
  --no-sample-latent \
  --allow-partial-checkpoint

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv" uv run "$ROOT_DIR/scripts/train_seeds.sh" \
  "$ROOT_DIR/configs/square_lowdim.yaml" \
  "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  0 1 2

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/runs/square_lowdim_seed0/resolved_config.yaml" \
  --checkpoint "$ROOT_DIR/runs/square_lowdim_seed0/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --no-sample-latent \
  --allow-partial-checkpoint

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/runs/square_lowdim_seed1/resolved_config.yaml" \
  --checkpoint "$ROOT_DIR/runs/square_lowdim_seed1/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --no-sample-latent \
  --allow-partial-checkpoint

UV_PROJECT_ENVIRONMENT="$ROOT_DIR/.venv-rollout" uv run --extra rollout python -m actdyn.eval \
  --config "$ROOT_DIR/runs/square_lowdim_seed2/resolved_config.yaml" \
  --checkpoint "$ROOT_DIR/runs/square_lowdim_seed2/checkpoints/best.pt" \
  --dataset "$ROOT_DIR/datasets/square/ph/low_dim_v15.hdf5" \
  --no-sample-latent \
  --allow-partial-checkpoint

echo "=== DONE $(date) ==="
echo "Log: $LOG"
