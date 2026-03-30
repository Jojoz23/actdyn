#!/usr/bin/env bash
# Multi-seed training (course proposal: 3–5 seeds). Example:
#   ./scripts/train_seeds.sh configs/lift_lowdim.yaml /path/to/low_dim_v15.hdf5 0 1 2
set -euo pipefail
CFG="${1:?config yaml}"
DATA="${2:?path to low_dim_v15.hdf5}"
shift 2
SEEDS=("$@")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH=src
for s in "${SEEDS[@]}"; do
  echo "=== seed $s ==="
  python -m actdyn.train --config "$CFG" --dataset "$DATA" --seed "$s" --out-dir "runs/$(basename "$CFG" .yaml)_seed${s}"
done
