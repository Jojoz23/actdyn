# actdyn — CSC415: adaptive execution for transformer action chunking

RoboMimic **low-dim** imitation (Lift / Can / Square), ACT-style chunk CVAE, and **inference-time** execution modes (full chunk, receding, dynamic overlap, temporal ensemble, learned commit).

## Quick start (recommended: `uv`)

From the repo root:

```bash
# Training stack (Python 3.10–3.14; project default often 3.14)
uv sync

# Rollout eval needs MuJoCo: use a second venv on Python 3.10–3.13 (prebuilt wheels)
uv venv .venv-rollout --python 3.13
UV_PROJECT_ENVIRONMENT=.venv-rollout uv sync --extra rollout
```

### Pip fallback (if `uv` is unavailable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Put HDF5s under `datasets/<task>/ph/low_dim_v15.hdf5` (paths match configs), or override with `--dataset` / `ACTDYN_DATASET_PATH`.

**Train (example — Lift strong):**

```bash
uv run actdyn-train \
  --config configs/lift_lowdim_strong.yaml \
  --dataset datasets/lift/ph/low_dim_v15.hdf5 \
  --out-dir runs/actdyn_lift_strong
```

**Eval (z0 latent, all modes in config):**

```bash
UV_PROJECT_ENVIRONMENT=.venv-rollout uv run --extra rollout python -m actdyn.eval \
  --config configs/lift_lowdim_strong.yaml \
  --checkpoint runs/actdyn_lift_strong/checkpoints/best.pt \
  --dataset datasets/lift/ph/low_dim_v15.hdf5 \
  --no-sample-latent
```

**Offline-only loss (no simulator):**

```bash
uv run python -m actdyn.eval --config configs/lift_lowdim_strong.yaml \
  --checkpoint runs/actdyn_lift_strong/checkpoints/best.pt --offline-only
```

More detail, Can/Square configs, and troubleshooting: **[`docs/README.md`](docs/README.md)**.

## Dependencies

| File | Role |
|------|------|
| [`pyproject.toml`](pyproject.toml) | Canonical deps + `[rollout]` optional group (robomimic / robosuite). |
| [`requirements.txt`](requirements.txt) | Course-style `pip install -r requirements.txt` → editable install with `[rollout]`. Prefer `uv` if possible. |
| [`uv.lock`](uv.lock) | Locked dependency versions for `uv sync` (reproducible installs). |

Under `runs/`, only small eval **summary JSON** files are tracked (`rollout_summary.json`, per-mode `summary.json`, including learned-commit). Checkpoints, logs, and the rest of `runs/` stay gitignored.

Reproduce environments with **`uv sync`** (and the rollout venv in Quick start)—do **not** commit `.venv` / `.venv-rollout`; they are machine-local.

## Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `train_seeds.sh` | Retrain same config with seeds `0 1 2 …` into `runs/<config_basename>_seed<N>/`. |
| `eval_lift_strong_matrix.sh` | z0 / sampled-latent / TE-only ablation snapshots → `runs/actdyn_lift_strong/eval_ablations/`. |
| `eval_can_matrix.sh` | Same pattern for Can strong checkpoint. |
| `eval_square_matrix.sh` | Same pattern for Square strong checkpoint. |
| `eval_learned_commit.sh` | **Learned commit** rollouts with **correct** `--dataset` per task (`lift` \| `can` \| `square`). |
| `run_all_baselines.sh` | Train several baseline configs on one HDF5 path. |
| `run_missing_baseline_evals.sh` | Batch eval for baseline checkpoints (if present). |
| `run_can_square_full_parity.sh` | Long pipeline: **Lift + Can + Square** (ACT, matrices, baselines, learned, seeds). |

Make executable once: `chmod +x scripts/*.sh`

## Learned-commit eval (while you are running these)

Train (per task):

```bash
uv run actdyn-train \
  --config configs/lift_lowdim_learned.yaml \
  --dataset datasets/<TASK>/ph/low_dim_v15.hdf5 \
  --out-dir runs/actdyn_<TASK>_learned
```

Eval (forces matching dataset):

```bash
chmod +x scripts/eval_learned_commit.sh
./scripts/eval_learned_commit.sh lift   # or can | square
```

Read metrics: `runs/actdyn_<task>_learned/eval/learned_commit/summary.json`.

## Reproduce key report tables

```bash
# 1) Main ACT z0 table (Lift/Can/Square, four execution modes)
UV_PROJECT_ENVIRONMENT=.venv-rollout ./scripts/eval_lift_strong_matrix.sh
UV_PROJECT_ENVIRONMENT=.venv-rollout ./scripts/eval_can_matrix.sh
UV_PROJECT_ENVIRONMENT=.venv-rollout ./scripts/eval_square_matrix.sh

# 2) Learned-commit appendix table
./scripts/eval_learned_commit.sh lift
./scripts/eval_learned_commit.sh can
./scripts/eval_learned_commit.sh square

# 3) Seed table (example: 3 seeds)
./scripts/train_seeds.sh configs/lift_lowdim_strong.yaml datasets/lift/ph/low_dim_v15.hdf5 0 1 2
./scripts/train_seeds.sh configs/can_lowdim.yaml datasets/can/ph/low_dim_v15.hdf5 0 1 2
./scripts/train_seeds.sh configs/square_lowdim.yaml datasets/square/ph/low_dim_v15.hdf5 0 1 2
```

