from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from actdyn.builders import build_datasets, build_policy, build_val_loader, device_from_config
from actdyn.config import load_config, prepare_output_dirs
from actdyn.data.robomimic_lowdim import RunningNormalizer
from actdyn.envs.robomimic_env import build_obs_vector, extract_success, load_env_from_dataset, reset_env, step_env
from actdyn.execution.executor import DynamicChunkExecutor
from actdyn.utils.checkpoint import load_checkpoint
from actdyn.utils.logging import write_csv
from actdyn.utils.misc import maybe_detach_dict, set_seed


def evaluate_offline(model, loader, device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_action": 0.0, "loss_kl": 0.0}
    count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="offline-eval", leave=False):
            obs = batch["obs"].to(device)
            actions = batch["actions"].to(device)
            is_pad = batch["is_pad"].to(device)
            _, metrics = model.loss(obs=obs, actions=actions, is_pad=is_pad, deterministic=True)
            metrics_f = maybe_detach_dict(metrics)
            for key in totals:
                totals[key] += metrics_f[key]
            count += 1
    if count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / count for k, v in totals.items()}


def make_executor(config: dict[str, Any], model, act_dim: int, device: str, mode: str) -> DynamicChunkExecutor:
    dyn = config["dynamic"]
    return DynamicChunkExecutor(
        policy=model,
        act_dim=act_dim,
        k_max=int(config["model"]["k_max"]),
        mode=mode,
        rule=str(dyn["rule"]),
        k_min=int(dyn.get("k_min", 1)),
        overlap_window=int(dyn.get("overlap_window", 4)),
        overlap_threshold=float(dyn.get("overlap_threshold", 0.08)),
        delta_threshold=float(dyn.get("delta_threshold", 0.12)),
        uncertainty_threshold=float(dyn.get("uncertainty_threshold", 0.10)),
        num_uncertainty_samples=int(dyn.get("num_uncertainty_samples", 8)),
        device=device,
    )


def summarize_commit_events(commit_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not commit_rows:
        return {
            "commit_count": 0,
            "commit_k_mean": float("nan"),
            "commit_k_std": float("nan"),
            "commit_k_min": float("nan"),
            "commit_k_max": float("nan"),
        }
    realized = np.asarray([float(r["realized_length"]) for r in commit_rows], dtype=np.float32)
    return {
        "commit_count": int(len(commit_rows)),
        "commit_k_mean": float(realized.mean()),
        "commit_k_std": float(realized.std()),
        "commit_k_min": float(realized.min()),
        "commit_k_max": float(realized.max()),
    }


def evaluate_rollouts(
    config: dict[str, Any],
    model,
    metadata: dict[str, Any],
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    env, env_meta = load_env_from_dataset(config["dataset"]["path"], use_image_obs=False)
    obs_norm = RunningNormalizer.from_state_dict(metadata["obs_normalizer"]) if config["dataset"].get("normalize_obs", True) else None
    obs_keys = metadata["obs_keys"]
    max_horizon = int(config["eval"].get("max_horizon", 400))
    num_episodes = int(config["eval"].get("num_episodes", 20))

    all_mode_summaries: dict[str, Any] = {}

    for mode in config["eval"].get("modes", ["full_chunk", "receding_horizon", "dynamic"]):
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        executor = make_executor(config, model=model, act_dim=int(metadata["act_dim"]), device=device, mode=mode)

        episode_rows: list[dict[str, Any]] = []
        commit_rows: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []

        success_values: list[float] = []
        returns: list[float] = []
        lengths: list[int] = []

        for episode_id in tqdm(range(num_episodes), desc=f"rollout-{mode}", leave=False):
            obs_dict = reset_env(env)
            executor.reset_episode()
            done = False
            ep_return = 0.0
            final_success = False
            t = 0

            while not done and t < max_horizon:
                obs_vec = build_obs_vector(obs_dict=obs_dict, obs_keys=obs_keys, obs_normalizer=obs_norm)
                result = executor.act(obs=obs_vec, episode_id=episode_id, timestep=t)
                obs_dict, reward, done, info = step_env(env, result.action)
                ep_return += reward
                final_success = final_success or extract_success(env, info, reward)
                t += 1

            executor.finish_episode()
            commit_rows.extend(executor.commit_events)
            step_rows.extend(executor.step_records)

            ep_row = {
                "episode_id": episode_id,
                "mode": mode,
                "rule": config["dynamic"]["rule"],
                "success": bool(final_success),
                "return": float(ep_return),
                "length": int(t),
                "num_commit_events": int(len(executor.commit_events)),
            }
            episode_rows.append(ep_row)
            success_values.append(float(final_success))
            returns.append(float(ep_return))
            lengths.append(int(t))

        commit_summary = summarize_commit_events(commit_rows)
        mode_summary = {
            "mode": mode,
            "rule": config["dynamic"]["rule"],
            "num_episodes": num_episodes,
            "success_rate": float(mean(success_values)) if success_values else float("nan"),
            "avg_return": float(mean(returns)) if returns else float("nan"),
            "avg_length": float(mean(lengths)) if lengths else float("nan"),
            **commit_summary,
        }

        (mode_dir / "summary.json").write_text(json.dumps(mode_summary, indent=2), encoding="utf-8")
        write_csv(mode_dir / "episode_metrics.csv", episode_rows)
        write_csv(mode_dir / "commit_events.csv", commit_rows)
        write_csv(mode_dir / "step_records.csv", step_rows)
        all_mode_summaries[mode] = mode_summary

    return {
        "env_meta": env_meta,
        "modes": all_mode_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ACT dynamic execution prototype.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint.")
    parser.add_argument("--offline-only", action="store_true", help="Only compute offline validation loss.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 0)))
    dirs = prepare_output_dirs(config)

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    metadata = ckpt.get("extra", {}).get("dataset_metadata")
    if metadata is None:
        raise RuntimeError("Checkpoint is missing dataset_metadata in the extra payload.")

    config = dict(config)
    config['dataset'] = dict(config['dataset'])
    config['dataset']['obs_keys'] = list(metadata['obs_keys'])
    _train_dataset, val_dataset, _unused = build_datasets(config)
    device = device_from_config(config)
    model = build_policy(config, obs_dim=int(metadata["obs_dim"]), act_dim=int(metadata["act_dim"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_loader = build_val_loader(config, val_dataset)
    offline = evaluate_offline(model=model, loader=val_loader, device=device)
    out_dir = dirs["eval"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "offline_metrics.json").write_text(json.dumps(offline, indent=2), encoding="utf-8")
    print("Offline metrics:", json.dumps(offline, indent=2))

    if args.offline_only:
        return

    rollout_summary = evaluate_rollouts(
        config=config,
        model=model,
        metadata=metadata,
        device=str(device),
        out_dir=out_dir,
    )
    (out_dir / "rollout_summary.json").write_text(json.dumps(rollout_summary, indent=2, default=str), encoding="utf-8")
    print("Rollout summary written to:", out_dir / "rollout_summary.json")


if __name__ == "__main__":
    main()
