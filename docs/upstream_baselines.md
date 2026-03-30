# “Proper” baselines: upstream code vs this repo

Your **course project is imitation learning**: policies are fit from **offline demonstrations**, then tested with **RL-style rollouts** (return, success) in simulation. That is standard for RoboMimic-style work. It is **not** the same as training from online RL reward.

What lives **here** (`actdyn`) are **self-contained comparators** that share your dataloader, checkpoint format, and rollout script. They are **not** byte-for-byte copies of every paper’s public codebase.

If you want the **authors’ implementations** (“from the flesh”), start from these upstream projects and **integrate or cite** them explicitly:

| Idea (proposal) | Upstream (starting point) | Notes |
|-----------------|---------------------------|--------|
| **ACT** (chunk transformer / CVAE-style chunking) | [tonyzhaozh/act](https://github.com/tonyzhaozh/act) · [LeRobot ACT docs](https://huggingface.co/docs/lerobot/main/en/act) | Often vision + sim/real; different config from our low-dim Lift-only stack. |
| **BeT** (discrete behavior tokens + transformer) | [notmahi/bet](https://github.com/notmahi/bet) · [notmahi/miniBET](https://github.com/notmahi/miniBET) | Full BeT uses transformer + offsets; our `bet_dbc` is **k-means + MLP** (BeT-*style*, smaller scope). |
| **RoboMimic BC / RNN / etc.** | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | Official baselines on the **same** HDF5 format you use. Strong option for a “real” BC baseline without maintaining your own `bc_mlp`. |
| **RT-1** | [Project / paper](https://robotics-transformer1.github.io/) (and linked Google Research materials) | **Vision + language + massive data.** Not a drop-in for state-only Lift; reasonable to **cite, not re-run**. |
| **VINN** | Depends which paper you meant in the proposal | Our `knn_bc` is **kNN regression in obs space** (a common *nonparametric* BC baseline). It is **not** a claim to reproduce a specific “VINN” paper unless you name that paper and use their code. |

You do **not** need the full PDF of every cited paper in the repo. You **do** need: correct citations + honest wording (what you ran vs what you cite).

## Training everything in `actdyn`

1. Put `low_dim_v15.hdf5` where your YAML expects it (or use `--dataset` / `ACTDYN_DATASET_PATH`).
2. From repo root (with `uv`):

```bash
./scripts/run_all_baselines.sh /path/to/low_dim_v15.hdf5
```

Or train configs individually with `uv run actdyn-train --config configs/<name>.yaml --dataset /path/to/low_dim_v15.hdf5`.

3. Rollout eval still uses your **Python 3.10 + robomimic** environment (as before).

**This workspace** often has **no HDF5** mounted; training must run on a machine that has the dataset (e.g. lab PC).

## Implementation caveats (review)

- **`knn_bc`**: stores the **entire train split** in GPU buffers at train time → can **OOM** on huge datasets; consider subsampling or CPU distance (future improvement).
- **`bet_dbc`**: simplified discrete BC; not full BeT transformer + offset head.
- **Synthetic smoke test** (no HDF5): run `uv run python -c "..."` from the agent log in development, or add a proper `pytest` later.
