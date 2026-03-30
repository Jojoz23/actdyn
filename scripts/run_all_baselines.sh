#!/usr/bin/env bash
# Train every empirical policy type in the repo (then eval each with its own config).
# Usage: ./scripts/run_all_baselines.sh /path/to/low_dim_v15.hdf5
set -euo pipefail
DATA="${1:?path to low_dim_v15.hdf5}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
RUNNER=(uv run actdyn-train)
if ! command -v uv >/dev/null 2>&1; then
  export PYTHONPATH=src
  RUNNER=(python -m actdyn.train)
fi

for cfg in \
  configs/lift_bc_mlp.yaml \
  configs/lift_knn_bc.yaml \
  configs/lift_bet_style.yaml \
  configs/lift_lowdim.yaml \
  configs/lift_lowdim_learned.yaml
do
  echo "======== TRAIN $cfg ========"
  "${RUNNER[@]}" --config "$cfg" --dataset "$DATA"
done
