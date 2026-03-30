from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from actdyn.data.robomimic_lowdim import (
    RoboMimicLowDimDataset,
    RoboMimicStepDataset,
    auto_detect_obs_keys,
    split_demo_keys,
)
from actdyn.models.act_cvae import ACTCVAEPolicy
from actdyn.models.bc_mlp import BCMLPPolicy
from actdyn.models.bet_style import BeTStyleDiscretePolicy
from actdyn.models.knn_bc import KNNBCPolicy
from actdyn.utils.action_kmeans import action_kmeans, assign_actions_to_centroids


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


def build_step_datasets(config: dict[str, Any]):
    """Train/val datasets of (obs_t, a_t) for BC-MLP."""
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

    train_dataset = RoboMimicStepDataset(
        dataset_path=dataset_cfg["path"],
        demo_keys=train_keys,
        obs_keys=obs_keys,
        obs_normalizer=obs_norm,
        action_normalizer=act_norm,
        normalize_obs=bool(dataset_cfg.get("normalize_obs", True)),
        normalize_actions=bool(dataset_cfg.get("normalize_actions", False)),
    )
    val_dataset = RoboMimicStepDataset(
        dataset_path=dataset_cfg["path"],
        demo_keys=val_keys,
        obs_keys=obs_keys,
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


class LabeledStepDataset(Dataset):
    """Step dataset with k-means behavior-token labels (BeT-style training / eval)."""

    def __init__(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.obs = obs
        self.actions = actions
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.obs[index],
            "action": self.actions[index],
            "label": self.labels[index],
        }


def materialize_step_tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)
    obs_parts: list[torch.Tensor] = []
    act_parts: list[torch.Tensor] = []
    for batch in loader:
        obs_parts.append(batch["obs"])
        act_parts.append(batch["action"])
    return torch.cat(obs_parts, dim=0), torch.cat(act_parts, dim=0)


def build_bet_training_datasets(
    config: dict[str, Any],
) -> tuple[LabeledStepDataset, LabeledStepDataset, dict[str, Any], torch.Tensor]:
    train_step, val_step, meta = build_step_datasets(config)
    train_obs, train_act = materialize_step_tensors(train_step)
    val_obs, val_act = materialize_step_tensors(val_step)
    num_bins = int(config["model"].get("bet_num_bins", 64))
    seed = int(config.get("seed", 0))
    centroids_np, train_lab = action_kmeans(train_act.detach().cpu().numpy(), num_bins, seed)
    val_lab = assign_actions_to_centroids(val_act.detach().cpu().numpy(), centroids_np)
    cent_t = torch.from_numpy(centroids_np)
    train_ds = LabeledStepDataset(train_obs, train_act, torch.from_numpy(train_lab))
    val_ds = LabeledStepDataset(val_obs, val_act, torch.from_numpy(val_lab))
    return train_ds, val_ds, meta, cent_t


def build_bet_val_dataset_for_eval(config: dict[str, Any], centroids_np: np.ndarray) -> LabeledStepDataset:
    _tr, val_step, _meta = build_step_datasets(config)
    val_obs, val_act = materialize_step_tensors(val_step)
    val_lab = assign_actions_to_centroids(val_act.detach().cpu().numpy(), centroids_np)
    return LabeledStepDataset(val_obs, val_act, torch.from_numpy(val_lab))


def build_policy(config: dict[str, Any], obs_dim: int, act_dim: int) -> nn.Module:
    model_cfg = config["model"]
    ptype = str(model_cfg.get("policy_type", "act_cvae")).lower()
    k_max = int(model_cfg["k_max"])
    if ptype == "bc_mlp":
        hidden = model_cfg.get("bc_hidden_dims", [512, 512, 256])
        return BCMLPPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            k_max=k_max,
            hidden_dims=tuple(int(x) for x in hidden),
            dropout=float(model_cfg.get("dropout", 0.1)),
            action_loss=str(model_cfg.get("action_loss", "l1")),
        )
    if ptype == "knn_bc":
        return KNNBCPolicy(
            obs_bank=torch.zeros(1, obs_dim),
            act_bank=torch.zeros(1, act_dim),
            k_neighbors=int(model_cfg.get("knn_k", 5)),
            k_max=k_max,
            act_dim=act_dim,
            action_loss=str(model_cfg.get("action_loss", "l1")),
        )
    if ptype == "bet_dbc":
        nb = int(model_cfg.get("bet_num_bins", 64))
        hidden = model_cfg.get("bc_hidden_dims", [512, 512, 256])
        return BeTStyleDiscretePolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            k_max=k_max,
            num_bins=nb,
            centroids=torch.zeros(nb, act_dim),
            hidden_dims=tuple(int(x) for x in hidden),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    dyn = config.get("dynamic", {})
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
        use_commit_head=bool(model_cfg.get("use_commit_head", False)),
        commit_loss_weight=float(model_cfg.get("commit_loss_weight", 0.0)),
        commit_label_delta_threshold=float(
            model_cfg.get("commit_label_delta_threshold", dyn.get("delta_threshold", 0.12))
        ),
        commit_k_min=int(model_cfg.get("commit_k_min", dyn.get("k_min", 1))),
        deploy_loss_weight=float(model_cfg.get("deploy_loss_weight", 1.0)),
    )


def build_train_loader(config: dict[str, Any], dataset) -> DataLoader:
    train_cfg = config["train"]
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", 256)),
        "shuffle": True,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 8))
    return DataLoader(dataset, **kwargs)


def build_val_loader(config: dict[str, Any], dataset) -> DataLoader:
    train_cfg = config["train"]
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg.get("val_batch_size", train_cfg.get("batch_size", 256))),
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 8))
    return DataLoader(dataset, **kwargs)


def device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config["train"].get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)
