# Final Experiment Plan (Lift + Can + Square)

Use this as the single source of truth for finishing experiments, paper tables, and slides.

## 0) Workspace

```bash
cd /virtual/csc415user/actdyn
```

---

## 1) Lift: ACT ablations (must do first)

```bash
cd /virtual/csc415user/actdyn
chmod +x /virtual/csc415user/actdyn/scripts/eval_lift_strong_matrix.sh
/virtual/csc415user/actdyn/scripts/eval_lift_strong_matrix.sh \
  /virtual/csc415user/actdyn/runs/actdyn_lift_strong/checkpoints/best.pt \
  /virtual/csc415user/actdyn/configs/lift_lowdim_strong.yaml
```

Expected snapshots:
- `runs/actdyn_lift_strong/eval_ablations/z0_default_modes`
- `runs/actdyn_lift_strong/eval_ablations/sampled_latent_default_modes`
- `runs/actdyn_lift_strong/eval_ablations/z0_temporal_ensemble_only`

---

## 2) Lift: baseline train/eval

### 2.1 Train all baselines

```bash
cd /virtual/csc415user/actdyn
/virtual/csc415user/actdyn/scripts/run_all_baselines.sh \
  /virtual/csc415user/actdyn/datasets/lift/ph/low_dim_v15.hdf5
```

### 2.2 Optional explicit baseline eval commands

```bash
cd /virtual/csc415user/actdyn
uv run --extra rollout python -m actdyn.eval \
  --config /virtual/csc415user/actdyn/configs/lift_bc_mlp.yaml \
  --checkpoint /virtual/csc415user/actdyn/runs/lift_bc_mlp/checkpoints/best.pt \
  --dataset /virtual/csc415user/actdyn/datasets/lift/ph/low_dim_v15.hdf5 \
  --no-sample-latent
```

```bash
cd /virtual/csc415user/actdyn
uv run --extra rollout python -m actdyn.eval \
  --config /virtual/csc415user/actdyn/configs/lift_knn_bc.yaml \
  --checkpoint /virtual/csc415user/actdyn/runs/lift_knn_bc/checkpoints/best.pt \
  --dataset /virtual/csc415user/actdyn/datasets/lift/ph/low_dim_v15.hdf5 \
  --no-sample-latent
```

```bash
cd /virtual/csc415user/actdyn
uv run --extra rollout python -m actdyn.eval \
  --config /virtual/csc415user/actdyn/configs/lift_bet_style.yaml \
  --checkpoint /virtual/csc415user/actdyn/runs/lift_bet_style/checkpoints/best.pt \
  --dataset /virtual/csc415user/actdyn/datasets/lift/ph/low_dim_v15.hdf5 \
  --no-sample-latent
```

---

## 3) Lift: multi-seed ACT robustness

```bash
cd /virtual/csc415user/actdyn
/virtual/csc415user/actdyn/scripts/train_seeds.sh \
  /virtual/csc415user/actdyn/configs/lift_lowdim_strong.yaml \
  /virtual/csc415user/actdyn/datasets/lift/ph/low_dim_v15.hdf5 \
  0 1 2
```

---

## 4) Can task: ACT + baselines

### 4.1 ACT train

```bash
cd /virtual/csc415user/actdyn
uv run actdyn-train \
  --config /virtual/csc415user/actdyn/configs/can_lowdim.yaml \
  --dataset /virtual/csc415user/actdyn/datasets/can/ph/low_dim_v15.hdf5 \
  --out-dir /virtual/csc415user/actdyn/runs/actdyn_can
```

### 4.2 ACT eval

```bash
cd /virtual/csc415user/actdyn
uv run --extra rollout python -m actdyn.eval \
  --config /virtual/csc415user/actdyn/configs/can_lowdim.yaml \
  --checkpoint /virtual/csc415user/actdyn/runs/actdyn_can/checkpoints/best.pt \
  --dataset /virtual/csc415user/actdyn/datasets/can/ph/low_dim_v15.hdf5 \
  --no-sample-latent
```

### 4.3 Baseline train

```bash
cd /virtual/csc415user/actdyn
uv run actdyn-train --config /virtual/csc415user/actdyn/configs/lift_bc_mlp.yaml --dataset /virtual/csc415user/actdyn/datasets/can/ph/low_dim_v15.hdf5 --out-dir /virtual/csc415user/actdyn/runs/can_bc_mlp
uv run actdyn-train --config /virtual/csc415user/actdyn/configs/lift_knn_bc.yaml --dataset /virtual/csc415user/actdyn/datasets/can/ph/low_dim_v15.hdf5 --out-dir /virtual/csc415user/actdyn/runs/can_knn_bc
uv run actdyn-train --config /virtual/csc415user/actdyn/configs/lift_bet_style.yaml --dataset /virtual/csc415user/actdyn/datasets/can/ph/low_dim_v15.hdf5 --out-dir /virtual/csc415user/actdyn/runs/can_bet_style
```

### 4.4 Baseline eval

```bash
cd /virtual/csc415user/actdyn
uv run --extra rollout python -m actdyn.eval --config /virtual/csc415user/actdyn/configs/lift_bc_mlp.yaml --dataset /virtual/csc415user/actdyn/datasets/can/ph/low_dim_v15.hdf5 --checkpoint /virtual/csc415user/actdyn/runs/can_bc_mlp/checkpoints/best.pt --no-sample-latent
uv run --extra rollout python -m actdyn.eval --config /virtual/csc415user/actdyn/configs/lift_knn_bc.yaml --dataset /virtual/csc415user/actdyn/datasets/can/ph/low_dim_v15.hdf5 --checkpoint /virtual/csc415user/actdyn/runs/can_knn_bc/checkpoints/best.pt --no-sample-latent
uv run --extra rollout python -m actdyn.eval --config /virtual/csc415user/actdyn/configs/lift_bet_style.yaml --dataset /virtual/csc415user/actdyn/datasets/can/ph/low_dim_v15.hdf5 --checkpoint /virtual/csc415user/actdyn/runs/can_bet_style/checkpoints/best.pt --no-sample-latent
```

---

## 5) Square task: ACT + baselines

### 5.1 ACT train

```bash
cd /virtual/csc415user/actdyn
uv run actdyn-train \
  --config /virtual/csc415user/actdyn/configs/square_lowdim.yaml \
  --dataset /virtual/csc415user/actdyn/datasets/square/ph/low_dim_v15.hdf5 \
  --out-dir /virtual/csc415user/actdyn/runs/actdyn_square
```

### 5.2 ACT eval

```bash
cd /virtual/csc415user/actdyn
uv run --extra rollout python -m actdyn.eval \
  --config /virtual/csc415user/actdyn/configs/square_lowdim.yaml \
  --checkpoint /virtual/csc415user/actdyn/runs/actdyn_square/checkpoints/best.pt \
  --dataset /virtual/csc415user/actdyn/datasets/square/ph/low_dim_v15.hdf5 \
  --no-sample-latent
```

### 5.3 Baseline train

```bash
cd /virtual/csc415user/actdyn
uv run actdyn-train --config /virtual/csc415user/actdyn/configs/lift_bc_mlp.yaml --dataset /virtual/csc415user/actdyn/datasets/square/ph/low_dim_v15.hdf5 --out-dir /virtual/csc415user/actdyn/runs/square_bc_mlp
uv run actdyn-train --config /virtual/csc415user/actdyn/configs/lift_knn_bc.yaml --dataset /virtual/csc415user/actdyn/datasets/square/ph/low_dim_v15.hdf5 --out-dir /virtual/csc415user/actdyn/runs/square_knn_bc
uv run actdyn-train --config /virtual/csc415user/actdyn/configs/lift_bet_style.yaml --dataset /virtual/csc415user/actdyn/datasets/square/ph/low_dim_v15.hdf5 --out-dir /virtual/csc415user/actdyn/runs/square_bet_style
```

### 5.4 Baseline eval

```bash
cd /virtual/csc415user/actdyn
uv run --extra rollout python -m actdyn.eval --config /virtual/csc415user/actdyn/configs/lift_bc_mlp.yaml --dataset /virtual/csc415user/actdyn/datasets/square/ph/low_dim_v15.hdf5 --checkpoint /virtual/csc415user/actdyn/runs/square_bc_mlp/checkpoints/best.pt --no-sample-latent
uv run --extra rollout python -m actdyn.eval --config /virtual/csc415user/actdyn/configs/lift_knn_bc.yaml --dataset /virtual/csc415user/actdyn/datasets/square/ph/low_dim_v15.hdf5 --checkpoint /virtual/csc415user/actdyn/runs/square_knn_bc/checkpoints/best.pt --no-sample-latent
uv run --extra rollout python -m actdyn.eval --config /virtual/csc415user/actdyn/configs/lift_bet_style.yaml --dataset /virtual/csc415user/actdyn/datasets/square/ph/low_dim_v15.hdf5 --checkpoint /virtual/csc415user/actdyn/runs/square_bet_style/checkpoints/best.pt --no-sample-latent
```

---

## 6) Optional: noise robustness (Lift ACT)

Run eval multiple times after editing `eval.obs_noise_std` in `configs/lift_lowdim_strong.yaml` to:
- `0.00`
- `0.01`
- `0.02`

Command:

```bash
cd /virtual/csc415user/actdyn
uv run --extra rollout python -m actdyn.eval \
  --config /virtual/csc415user/actdyn/configs/lift_lowdim_strong.yaml \
  --checkpoint /virtual/csc415user/actdyn/runs/actdyn_lift_strong/checkpoints/best.pt \
  --dataset /virtual/csc415user/actdyn/datasets/lift/ph/low_dim_v15.hdf5 \
  --no-sample-latent
```

---

## 7) Required artifacts for paper / presentation

For each run folder:
- `eval/rollout_summary.json`
- `eval/offline_metrics.json`
- `plots/train_val_loss.png`

Core metrics to report:
- `success_rate`
- `avg_return`
- `avg_policy_queries`
- `avg_replans_per_episode`
- commit stats (`commit_k_mean/std/min/max`) for ACT modes

---

## 8) Scope note for writeup (important)

Baselines implemented and compared:
- BC-MLP (`bc_mlp`)
- VINN-style kNN (`knn_bc`)
- BeT-style discrete BC (`bet_dbc`)

RT-1 is related work only (not reimplemented in this pipeline).
