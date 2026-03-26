from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from actdyn.utils.misc import safe_std


@dataclass
class RunningNormalizer:
    mean: np.ndarray
    std: np.ndarray
    enabled: bool = True

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x.astype(np.float32)
        return (x * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "RunningNormalizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            enabled=bool(state.get("enabled", True)),
        )


def _flatten_obs_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _is_image_like(dataset: h5py.Dataset) -> bool:
    shape = dataset.shape
    if len(shape) < 2:
        return False
    if dataset.dtype == np.uint8 and any(dim >= 32 for dim in shape[1:]):
        return True
    return False


def list_demo_keys(dataset_path: str | Path) -> list[str]:
    with h5py.File(dataset_path, "r") as f:
        return sorted(list(f["data"].keys()))


def get_filter_keys(dataset_path: str | Path) -> list[str]:
    with h5py.File(dataset_path, "r") as f:
        if "mask" not in f:
            return []
        return sorted(list(f["mask"].keys()))


def get_env_metadata(dataset_path: str | Path) -> dict[str, Any] | None:
    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        attr_keys = ["env_args", "env_meta", "env_metadata"]
        for key in attr_keys:
            if key in data_group.attrs:
                raw = data_group.attrs[key]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                if isinstance(raw, str):
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return {"raw": raw}
                return raw
    return None


def auto_detect_obs_keys(dataset_path: str | Path) -> list[str]:
    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(list(f["data"].keys()))
        if not demo_keys:
            raise ValueError("No demos found in dataset.")
        obs_group = f["data"][demo_keys[0]]["obs"]
        keys: list[str] = []
        for key in sorted(obs_group.keys()):
            ds = obs_group[key]
            if _is_image_like(ds):
                continue
            keys.append(key)
        if not keys:
            raise ValueError("Could not auto-detect any low-dimensional observation keys.")
        return keys


def split_demo_keys(
    dataset_path: str | Path,
    filter_key_train: str | None,
    filter_key_val: str | None,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    with h5py.File(dataset_path, "r") as f:
        all_demo_keys = sorted(list(f["data"].keys()))
        if "mask" in f and filter_key_train and filter_key_val and filter_key_train in f["mask"] and filter_key_val in f["mask"]:
            train_keys = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["mask"][filter_key_train][()]]
            val_keys = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["mask"][filter_key_val][()]]
            return sorted(train_keys), sorted(val_keys)

        rng = np.random.default_rng(seed)
        shuffled = list(all_demo_keys)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_keys = sorted(shuffled[:n_val])
        train_keys = sorted(shuffled[n_val:])
        if not train_keys:
            train_keys = val_keys[:]
        return train_keys, val_keys


def summarize_dataset(dataset_path: str | Path) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(list(f["data"].keys()))
        if not demo_keys:
            raise ValueError("No demos found in dataset.")
        first_demo = f["data"][demo_keys[0]]
        obs_group = first_demo["obs"]
        summary = {
            "dataset_path": str(dataset_path),
            "num_demos": len(demo_keys),
            "demo_keys_preview": demo_keys[:5],
            "action_shape_first_demo": tuple(first_demo["actions"].shape),
            "obs_keys": sorted(list(obs_group.keys())),
            "filter_keys": sorted(list(f["mask"].keys())) if "mask" in f else [],
            "env_metadata": get_env_metadata(dataset_path),
        }
        return summary


class RoboMimicLowDimDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        demo_keys: list[str],
        obs_keys: list[str],
        k_max: int,
        obs_normalizer: RunningNormalizer | None = None,
        action_normalizer: RunningNormalizer | None = None,
        normalize_obs: bool = True,
        normalize_actions: bool = False,
    ) -> None:
        super().__init__()
        self._h5_file = None
        self.dataset_path = str(dataset_path)
        self.demo_keys = list(demo_keys)
        self.obs_keys = list(obs_keys)
        self.k_max = int(k_max)
        self.normalize_obs = bool(normalize_obs)
        self.normalize_actions = bool(normalize_actions)
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self._index: list[tuple[str, int]] = []
        self._demo_lengths: dict[str, int] = {}
        self.obs_dim: int | None = None
        self.act_dim: int | None = None

        with h5py.File(self.dataset_path, "r") as f:
            for demo_key in self.demo_keys:
                demo = f["data"][demo_key]
                actions = demo["actions"]
                self._demo_lengths[demo_key] = int(actions.shape[0])
                for t in range(actions.shape[0]):
                    self._index.append((demo_key, t))
                if self.obs_dim is None:
                    obs_dim = 0
                    for key in self.obs_keys:
                        obs_dim += int(np.prod(demo["obs"][key].shape[1:]))
                    self.obs_dim = obs_dim
                    self.act_dim = int(actions.shape[-1])

        if self.obs_dim is None or self.act_dim is None:
            raise ValueError("Failed to infer obs_dim or act_dim from dataset.")

    def __len__(self) -> int:
        return len(self._index)

    def _get_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.dataset_path, "r")
        return self._h5_file

    def _read_obs(self, demo: h5py.Group, t: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key in self.obs_keys:
            parts.append(_flatten_obs_array(demo["obs"][key][t]))
        obs = np.concatenate(parts, axis=0).astype(np.float32)
        if self.normalize_obs and self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
        return obs

    def _read_actions_chunk(self, demo: h5py.Group, t: int) -> tuple[np.ndarray, np.ndarray]:
        actions = np.asarray(demo["actions"], dtype=np.float32)
        act_dim = actions.shape[-1]
        end = min(t + self.k_max, actions.shape[0])
        chunk = np.zeros((self.k_max, act_dim), dtype=np.float32)
        is_pad = np.ones((self.k_max,), dtype=bool)
        valid = actions[t:end]
        chunk[: valid.shape[0]] = valid
        is_pad[: valid.shape[0]] = False
        if self.normalize_actions and self.action_normalizer is not None:
            chunk = self.action_normalizer.normalize(chunk)
        return chunk, is_pad

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        demo_key, t = self._index[index]
        f = self._get_file()
        demo = f["data"][demo_key]
        obs = self._read_obs(demo, t)
        action_chunk, is_pad = self._read_actions_chunk(demo, t)

        return {
            "obs": torch.from_numpy(obs),
            "actions": torch.from_numpy(action_chunk),
            "is_pad": torch.from_numpy(is_pad),
            "timestep": torch.tensor(t, dtype=torch.long),
        }

    def compute_normalizers(self) -> tuple[RunningNormalizer, RunningNormalizer]:
        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []
        with h5py.File(self.dataset_path, "r") as f:
            for demo_key in self.demo_keys:
                demo = f["data"][demo_key]
                obs_parts = []
                for key in self.obs_keys:
                    x = np.asarray(demo["obs"][key], dtype=np.float32).reshape(demo["obs"][key].shape[0], -1)
                    obs_parts.append(x)
                demo_obs = np.concatenate(obs_parts, axis=1)
                obs_list.append(demo_obs)
                action_list.append(np.asarray(demo["actions"], dtype=np.float32))

        obs_all = np.concatenate(obs_list, axis=0)
        act_all = np.concatenate(action_list, axis=0)
        obs_norm = RunningNormalizer(
            mean=obs_all.mean(axis=0).astype(np.float32),
            std=safe_std(obs_all),
            enabled=True,
        )
        act_norm = RunningNormalizer(
            mean=act_all.mean(axis=0).astype(np.float32),
            std=safe_std(act_all),
            enabled=True,
        )
        return obs_norm, act_norm
