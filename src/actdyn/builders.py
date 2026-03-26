from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from actdyn.data.robomimic_lowdim import (
    RoboMimicLowDimDataset,
    auto_detect_obs_keys,
    split_demo_keys,
)
from actdyn.models.act_cvae import ACTCVAEPolicy


def resolve_obs_keys(config: dict[str, Any]) -> list[str]:
    obs_keys = list(config["dataset"].get("obs_keys", []))
    if obs_keys:
        return obs_keys
    return auto_detect_obs_keys(config["dataset"]["path"])


def build_datasets(config: dict[str, Any]):
    dataset_cfg = config["dataset"]
    obs_keys = resolve_obs_keys(config)

    train_keys, val_keys = split_demo_keys(
        dataset_path=dataset_cfg["path"],
        filter_key_train=dataset_cfg.get("filter_key_train"),
        filter_key_val=dataset_cfg.get("filter_key_val"),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        seed=int(config.get("seed", 0)),
    )

    base_train = RoboMimicLowDimDataset(
        dataset_path=dataset_cfg["path"],
        demo_keys=train_keys,
        obs_keys=obs_keys,
        k_max=int(config["model"]["k_max"]),
        normalize_obs=False,
        normalize_actions=False,
    )
    obs_norm, act_norm = base_train.compute_normalizers()

    train_dataset = RoboMimicLowDimDataset(
        dataset_path=dataset_cfg["path"],
        demo_keys=train_keys,
        obs_keys=obs_keys,
        k_max=int(config["model"]["k_max"]),
        obs_normalizer=obs_norm,
        action_normalizer=act_norm,
        normalize_obs=bool(dataset_cfg.get("normalize_obs", True)),
        normalize_actions=bool(dataset_cfg.get("normalize_actions", False)),
    )
    val_dataset = RoboMimicLowDimDataset(
        dataset_path=dataset_cfg["path"],
        demo_keys=val_keys,
        obs_keys=obs_keys,
        k_max=int(config["model"]["k_max"]),
        obs_normalizer=obs_norm,
        action_normalizer=act_norm,
        normalize_obs=bool(dataset_cfg.get("normalize_obs", True)),
        normalize_actions=bool(dataset_cfg.get("normalize_actions", False)),
    )
    metadata = {
        "obs_keys": obs_keys,
        "train_demo_keys": train_keys,
        "val_demo_keys": val_keys,
        "obs_normalizer": obs_norm.state_dict(),
        "action_normalizer": act_norm.state_dict(),
        "obs_dim": train_dataset.obs_dim,
        "act_dim": train_dataset.act_dim,
    }
    return train_dataset, val_dataset, metadata


def build_policy(config: dict[str, Any], obs_dim: int, act_dim: int) -> ACTCVAEPolicy:
    model_cfg = config["model"]
    return ACTCVAEPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        k_max=int(model_cfg["k_max"]),
        d_model=int(model_cfg.get("d_model", 256)),
        nhead=int(model_cfg.get("nhead", 8)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        action_loss=str(model_cfg.get("action_loss", "l1")),
        kl_beta=float(model_cfg.get("kl_beta", 1e-4)),
    )


def build_train_loader(config: dict[str, Any], dataset) -> DataLoader:
    train_cfg = config["train"]
    num_workers = int(train_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 8)),
    )


def build_val_loader(config: dict[str, Any], dataset) -> DataLoader:
    train_cfg = config["train"]
    num_workers = int(train_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("val_batch_size", train_cfg.get("batch_size", 256))),
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 8)),
    )


def device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config["train"].get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)
