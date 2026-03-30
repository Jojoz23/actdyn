from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from actdyn.builders import (
    build_bet_training_datasets,
    build_datasets,
    build_policy,
    build_step_datasets,
    build_train_loader,
    build_val_loader,
    device_from_config,
    materialize_step_tensors,
)
from actdyn.models.bet_style import BeTStyleDiscretePolicy
from actdyn.models.knn_bc import KNNBCPolicy
from actdyn.config import load_config, prepare_output_dirs, save_config
from actdyn.utils.checkpoint import save_checkpoint
from actdyn.utils.logging import ExperimentLogger
from actdyn.utils.misc import count_parameters, maybe_detach_dict, set_seed


def lr_cosine_warmup(
    epoch: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    max_epochs: int,
) -> float:
    """Linear warmup, then cosine decay to min_lr (epoch is 1-based)."""
    if warmup_epochs <= 0:
        warmup_epochs = 1
    if epoch <= warmup_epochs:
        return base_lr * float(epoch) / float(warmup_epochs)
    denom = max(max_epochs - warmup_epochs, 1)
    t = float(epoch - warmup_epochs) / float(denom)
    t = min(max(t, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def apply_scheduler_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    *,
    base_lr: float,
    sched_cfg: dict | None,
    max_epochs: int,
) -> float:
    if not sched_cfg or sched_cfg.get("type") not in ("cosine_warmup", "cosine"):
        return float(optimizer.param_groups[0]["lr"])
    warmup = int(sched_cfg.get("warmup_epochs", 10))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    lr = lr_cosine_warmup(
        epoch,
        base_lr=base_lr,
        min_lr=min_lr,
        warmup_epochs=warmup,
        max_epochs=max_epochs,
    )
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def run_epoch(
    model,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device,
    grad_clip_norm: float,
    train: bool,
    *,
    is_bc_mlp: bool = False,
    is_bet_dbc: bool = False,
    k_max: int = 16,
    act_dim: int = 7,
) -> dict[str, float]:
    model.train(mode=train)
    total = {"loss": 0.0, "loss_action": 0.0, "loss_kl": 0.0}
    if getattr(model, "use_commit_head", False):
        total["loss_commit"] = 0.0
    if float(getattr(model, "deploy_loss_weight", 0.0)) != 0.0:
        total["loss_action_deploy"] = 0.0
    num_batches = 0

    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in iterator:
        if is_bet_dbc:
            obs = batch["obs"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            act_c = batch["action"].to(device, non_blocking=True)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss_from_labels(obs, labels, act_c)
            if train and optimizer is not None:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        elif is_bc_mlp:
            obs = batch["obs"].to(device, non_blocking=True)
            act_single = batch["action"].to(device, non_blocking=True)
            bsz = obs.shape[0]
            actions = torch.zeros(bsz, k_max, act_dim, device=device, dtype=act_single.dtype)
            actions[:, 0] = act_single
            is_pad = torch.ones(bsz, k_max, dtype=torch.bool, device=device)
            is_pad[:, 0] = False
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss(obs=obs, actions=actions, is_pad=is_pad, deterministic=not train)
            if train and optimizer is not None:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        else:
            obs = batch["obs"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            is_pad = batch["is_pad"].to(device, non_blocking=True)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss(obs=obs, actions=actions, is_pad=is_pad, deterministic=not train)
            if train and optimizer is not None:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        metrics_f = maybe_detach_dict(metrics)
        for key in total:
            total[key] += metrics_f.get(key, 0.0)
        num_batches += 1
        iterator.set_postfix(loss=f"{metrics_f['loss']:.4f}")

    if num_batches == 0:
        return {k: float("nan") for k in total}
    return {k: v / num_batches for k, v in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ACT dynamic execution prototype.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset HDF5 path (also supported: ACTDYN_DATASET_PATH env).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed (multi-seed sweeps).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override logging.out_dir (e.g. runs/actdyn_lift_seed1).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.out_dir:
        config.setdefault("logging", {})
        config["logging"]["out_dir"] = args.out_dir
    if args.dataset:
        config.setdefault("dataset", {})
        config["dataset"]["path"] = os.path.expanduser(args.dataset)
    set_seed(int(config.get("seed", 0)))
    dirs = prepare_output_dirs(config)
    save_config(config, dirs["root"] / "resolved_config.yaml")

    policy_type = str(config["model"].get("policy_type", "act_cvae")).lower()
    is_bc_mlp = policy_type == "bc_mlp"
    is_bet_dbc = policy_type == "bet_dbc"
    is_knn = policy_type == "knn_bc"

    if is_knn or is_bc_mlp:
        train_dataset, val_dataset, metadata = build_step_datasets(config)
    elif is_bet_dbc:
        train_dataset, val_dataset, metadata, cent_t = build_bet_training_datasets(config)
    else:
        train_dataset, val_dataset, metadata = build_datasets(config)
    device = device_from_config(config)
    k_max = int(config["model"]["k_max"])
    act_dim = int(metadata["act_dim"])

    if is_knn:
        obs_t, act_t = materialize_step_tensors(train_dataset)
        model = KNNBCPolicy(
            obs_bank=obs_t.to(device),
            act_bank=act_t.to(device),
            k_neighbors=int(config["model"].get("knn_k", 5)),
            k_max=k_max,
            act_dim=act_dim,
            action_loss=str(config["model"].get("action_loss", "l1")),
        ).to(device)
    elif is_bet_dbc:
        model = BeTStyleDiscretePolicy(
            obs_dim=int(metadata["obs_dim"]),
            act_dim=act_dim,
            k_max=k_max,
            num_bins=int(cent_t.shape[0]),
            centroids=cent_t.to(device).float(),
            hidden_dims=tuple(int(x) for x in config["model"].get("bc_hidden_dims", [512, 512, 256])),
            dropout=float(config["model"].get("dropout", 0.1)),
        ).to(device)
    else:
        model = build_policy(config, obs_dim=metadata["obs_dim"], act_dim=metadata["act_dim"]).to(device)

    train_loader = build_train_loader(config, train_dataset)
    val_loader = build_val_loader(config, val_dataset)

    base_lr = float(config["optim"].get("lr", 1e-4))
    if is_knn:
        optimizer = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=float(config["optim"].get("weight_decay", 0.0)),
        )
    sched_cfg = None if is_knn else config["optim"].get("scheduler")
    num_epochs = 1 if is_knn else int(config["train"].get("epochs", 50))

    logger = ExperimentLogger(out_dir=dirs["root"], tensorboard_dir=dirs["tensorboard"])
    best_val = math.inf
    best_extra = None
    global_step = 0

    print(f"Device: {device} policy_type={policy_type}")
    print(f"obs_dim={metadata['obs_dim']} act_dim={metadata['act_dim']} params={count_parameters(model):,}")
    print(f"Train demos={len(metadata['train_demo_keys'])} Val demos={len(metadata['val_demo_keys'])}")
    print(f"Obs keys={metadata['obs_keys']}")

    grad_clip_norm = float(config["optim"].get("grad_clip_norm", 1.0))

    for epoch in range(1, num_epochs + 1):
        if optimizer is not None:
            current_lr = apply_scheduler_lr(
                optimizer,
                epoch,
                base_lr=base_lr,
                sched_cfg=sched_cfg if isinstance(sched_cfg, dict) else None,
                max_epochs=num_epochs,
            )
        else:
            current_lr = 0.0
        start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
            train=True,
            is_bc_mlp=is_bc_mlp or is_knn,
            is_bet_dbc=is_bet_dbc,
            k_max=k_max,
            act_dim=act_dim,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
            train=False,
            is_bc_mlp=is_bc_mlp or is_knn,
            is_bet_dbc=is_bet_dbc,
            k_max=k_max,
            act_dim=act_dim,
        )
        elapsed = time.time() - start

        tb_metrics: dict[str, float] = {
            "train/loss": train_metrics["loss"],
            "train/loss_action": train_metrics["loss_action"],
            "train/loss_kl": train_metrics["loss_kl"],
            "val/loss": val_metrics["loss"],
            "val/loss_action": val_metrics["loss_action"],
            "val/loss_kl": val_metrics["loss_kl"],
            "train/lr": current_lr,
            "time/epoch_sec": elapsed,
        }
        if "loss_commit" in train_metrics:
            tb_metrics["train/loss_commit"] = train_metrics["loss_commit"]
            tb_metrics["val/loss_commit"] = val_metrics["loss_commit"]
        logger.log_scalars(step=epoch, metrics=tb_metrics)

        extra = {
            "dataset_metadata": metadata,
        }

        if config["checkpoint"].get("save_last", True):
            save_checkpoint(
                path=dirs["checkpoints"] / "last.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                global_step=global_step,
                extra=extra,
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_extra = extra
            if config["checkpoint"].get("save_best", True):
                save_checkpoint(
                    path=dirs["checkpoints"] / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    extra=extra,
                )

        logger.log_jsonl(
            "epoch_metrics.jsonl",
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "best_val_loss": best_val,
                "epoch_sec": elapsed,
            },
        )
        logger.plot_train_val_loss(dirs["plots"] / "train_val_loss.png")
        print(
            f"[epoch {epoch:03d}] train={train_metrics['loss']:.5f} "
            f"val={val_metrics['loss']:.5f} best={best_val:.5f}"
        )
        global_step += len(train_loader)

    logger.close()
    print(f"Training complete. Best val loss: {best_val:.6f}")
    print(f"Outputs written to: {dirs['root']}")


if __name__ == "__main__":
    main()
