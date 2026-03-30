from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file {path} is empty.")
    override = os.environ.get("ACTDYN_DATASET_PATH")
    if override:
        cfg.setdefault("dataset", {})
        cfg["dataset"]["path"] = override
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def prepare_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    out_dir = Path(config["logging"]["out_dir"])
    dirs = {
        "root": out_dir,
        "checkpoints": out_dir / "checkpoints",
        "tensorboard": out_dir / "tensorboard",
        "plots": out_dir / "plots",
        "eval": out_dir / "eval",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def dump_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def get_device(config: dict[str, Any]) -> str:
    return str(config.get("train", {}).get("device", "cuda"))
