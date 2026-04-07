# actdyn (CSC415)

**Quick start, script table, and submission checklist:** see the repo root [`README.md`](../README.md).

Offline BC chunk policy (CVAE) + **inference-time** execution modes for RoboMimic low-dim (Lift / Can / Square).

## Train (GPU, Python 3.14 via uv)

```bash
cd actdyn
uv sync
uv run actdyn-train --config configs/lift_lowdim.yaml --dataset /path/to/low_dim_v15.hdf5
```

RoboMimic **Square** / **Can** low-dim PH: `configs/square_lowdim.yaml` and `configs/can_lowdim.yaml` (same hyperparameters as Lift; `obs_dim` / `act_dim` are read from each HDF5). Rename odd download names (e.g. `low_dim_v15 (1).hdf5`) to `low_dim_v15.hdf5` or override with `--dataset`.

Options: `--seed N`, `--out-dir runs/my_run`.

**Learned commit head** (proposal alignment): `configs/lift_lowdim_learned.yaml` — trains auxiliary head with expert-delta labels; eval includes `learned_commit`.

**Safe learned-commit rollout (dataset must match task):** after training `runs/actdyn_{lift,can,square}_learned`, run:

```bash
chmod +x scripts/eval_learned_commit.sh
./scripts/eval_learned_commit.sh lift   # or can / square
```

Do **not** point `--dataset` at the wrong task’s HDF5—the simulator is built from that path.

**Multi-seed:** `./scripts/train_seeds.sh configs/lift_lowdim.yaml /path/to/low_dim_v15.hdf5 0 1 2`

## Rollout eval (Python 3.13 side env + robomimic / robosuite)

Training can stay on **Python 3.14** in `.venv`. Rollouts need **3.10–3.13** (prebuilt `mujoco` wheels). Create a **second** venv and sync **into that directory** — not `uv sync --python .venv-rollout/bin/python`, which only changes the default `.venv` interpreter and leaves `.venv-rollout` empty.

```bash
cd actdyn
uv venv .venv-rollout --python 3.13
UV_PROJECT_ENVIRONMENT=.venv-rollout uv sync --extra rollout
.venv-rollout/bin/python -m actdyn.eval --config configs/lift_lowdim.yaml \
  --checkpoint runs/actdyn_lift/checkpoints/best.pt
```

Equivalent: `source .venv-rollout/bin/activate` then `uv sync --extra rollout --active`.

If you previously ran `uv sync --python .venv-rollout/bin/python`, uv may have **recreated `.venv`** with 3.13. Restore the train env with `uv venv --python 3.14` (or your usual interpreter) then `uv sync` without rollout.

- `eval.obs_noise_std` — Gaussian noise on the observation vector (robustness).
- Rollout summary includes `avg_policy_queries`, `avg_replans_per_episode`, commit-length stats.
- Offline eval for ACT now also reports `loss_action_prior_z0` (prior-only `z=0` chunk error), which better matches deployment behavior than teacher-forced posterior loss alone.

### Lift strong eval matrix (recommended)

Run three ablations and archive each output under `runs/actdyn_lift_strong/eval_ablations/`:

```bash
cd actdyn
chmod +x scripts/eval_lift_strong_matrix.sh
./scripts/eval_lift_strong_matrix.sh
```

Optional args:

```bash
./scripts/eval_lift_strong_matrix.sh runs/actdyn_lift_strong/checkpoints/best.pt configs/lift_lowdim_strong.yaml
```

Cases:
- `z0_default_modes`: default modes with `--no-sample-latent`.
- `sampled_latent_default_modes`: default modes with stochastic latent sampling.
- `z0_temporal_ensemble_only`: temporal-ensemble-only config (`configs/lift_lowdim_strong_te_only.yaml`).

Environment: `ACTDYN_DATASET_PATH` overrides YAML dataset path.

## Baselines vs proposal (see `latex/main.tex` Table)

| Name | Config | `policy_type` |
|------|--------|----------------|
| BC-ConvMLP (low-dim) | `configs/lift_bc_mlp.yaml` | `bc_mlp` |
| VINN-style kNN | `configs/lift_knn_bc.yaml` | `knn_bc` |
| BeT-style discrete BC | `configs/lift_bet_style.yaml` | `bet_dbc` |
| ACT-style chunk CVAE | `configs/lift_lowdim.yaml` | `act_cvae` (default) |
| + learned commit head | `configs/lift_lowdim_learned.yaml` | `act_cvae` + `use_commit_head` |

Train all: `./scripts/run_all_baselines.sh /path/to/low_dim_v15.hdf5`

**RT-1** is not instantiated (state-only Lift); cite as related work. See `src/actdyn/baselines/__init__.py`.

**“Proper” paper code (ACT, BeT, RoboMimic, RT-1):** see `docs/upstream_baselines.md`.
