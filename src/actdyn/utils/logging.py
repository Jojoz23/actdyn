from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - fallback when tensorboard is absent
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass


class ExperimentLogger:
    def __init__(self, out_dir: str | Path, tensorboard_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.tb = SummaryWriter(log_dir=str(tensorboard_dir))
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.history: dict[str, list[tuple[int, float]]] = {}

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.tb.add_scalar(key, value, step)
            self.history.setdefault(key, []).append((step, float(value)))
        row = {
            "type": "scalars",
            "step": int(step),
            "time": time.time(),
            **{k: float(v) for k, v in metrics.items()},
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def log_histogram(self, step: int, name: str, values: list[float] | Any) -> None:
        self.tb.add_histogram(name, values, step)

    def log_jsonl(self, filename: str, row: dict[str, Any]) -> None:
        path = self.out_dir / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def close(self) -> None:
        self.tb.flush()
        self.tb.close()

    def plot_train_val_loss(self, path: str | Path) -> None:
        path = Path(path)
        train = self.history.get("train/loss", [])
        val = self.history.get("val/loss", [])
        if not train and not val:
            return

        plt.figure(figsize=(8, 5))
        if train:
            plt.plot([x for x, _ in train], [y for _, y in train], label="train/loss")
        if val:
            plt.plot([x for x, _ in val], [y for _, y in val], label="val/loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Train / Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
