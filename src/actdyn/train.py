from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from actdyn.builders import build_datasets, build_policy, build_train_loader, build_val_loader, device_from_config
from actdyn.config import load_config, prepare_output_dirs, save_config
from actdyn.utils.checkpoint import save_checkpoint
from actdyn.utils.logging import ExperimentLogger
from actdyn.utils.misc import count_parameters, maybe_detach_dict, set_seed


def run_epoch(model, loader, optimizer, device, grad_clip_norm: float, train: bool) -> dict[str, float]:
    model.train(mode=train)
    total = {"loss": 0.0, "loss_action": 0.0, "loss_kl": 0.0}
    num_batches = 0

    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in iterator:
        obs = batch["obs"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        loss, metrics = model.loss(obs=obs, actions=actions, is_pad=is_pad, deterministic=not train)

        if train:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        metrics_f = maybe_detach_dict(metrics)
        for key in total:
            total[key] += metrics_f[key]
        num_batches += 1
        iterator.set_postfix(loss=f"{metrics_f['loss']:.4f}")

    if num_batches == 0:
        return {k: float("nan") for k in total}
    return {k: v / num_batches for k, v in total.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ACT dynamic execution prototype.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 0)))
    dirs = prepare_output_dirs(config)
    save_config(config, dirs["root"] / "resolved_config.yaml")

    train_dataset, val_dataset, metadata = build_datasets(config)
    device = device_from_config(config)
    model = build_policy(config, obs_dim=metadata["obs_dim"], act_dim=metadata["act_dim"]).to(device)

    train_loader = build_train_loader(config, train_dataset)
    val_loader = build_val_loader(config, val_dataset)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"].get("lr", 1e-4)),
        weight_decay=float(config["optim"].get("weight_decay", 0.0)),
    )

    logger = ExperimentLogger(out_dir=dirs["root"], tensorboard_dir=dirs["tensorboard"])
    best_val = math.inf
    best_extra = None
    global_step = 0

    print(f"Device: {device}")
    print(f"obs_dim={metadata['obs_dim']} act_dim={metadata['act_dim']} params={count_parameters(model):,}")
    print(f"Train demos={len(metadata['train_demo_keys'])} Val demos={len(metadata['val_demo_keys'])}")
    print(f"Obs keys={metadata['obs_keys']}")

    num_epochs = int(config["train"].get("epochs", 50))
    grad_clip_norm = float(config["optim"].get("grad_clip_norm", 1.0))

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, grad_clip_norm=grad_clip_norm, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, grad_clip_norm=grad_clip_norm, train=False)
        elapsed = time.time() - start

        logger.log_scalars(
            step=epoch,
            metrics={
                "train/loss": train_metrics["loss"],
                "train/loss_action": train_metrics["loss_action"],
                "train/loss_kl": train_metrics["loss_kl"],
                "val/loss": val_metrics["loss"],
                "val/loss_action": val_metrics["loss_action"],
                "val/loss_kl": val_metrics["loss_kl"],
                "train/lr": optimizer.param_groups[0]["lr"],
                "time/epoch_sec": elapsed,
            },
        )

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
