from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from actdyn.builders import (
    build_bet_val_dataset_for_eval,
    build_datasets,
    build_policy,
    build_step_datasets,
    build_val_loader,
    device_from_config,
)
from actdyn.models.bet_style import BeTStyleDiscretePolicy
from actdyn.config import load_config, prepare_output_dirs
from actdyn.data.robomimic_lowdim import RunningNormalizer
from actdyn.envs.robomimic_env import build_obs_vector, extract_success, load_env_from_dataset, reset_env, step_env
from actdyn.execution.executor import DynamicChunkExecutor
from actdyn.utils.checkpoint import load_checkpoint
from actdyn.models.act_cvae import ACTCVAEPolicy
from actdyn.models.knn_bc import KNNBCPolicy
from actdyn.utils.logging import write_csv
from actdyn.utils.misc import maybe_detach_dict, set_seed


def evaluate_offline(model, loader, device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_action": 0.0, "loss_kl": 0.0}
    if getattr(model, "use_commit_head", False):
        totals["loss_commit"] = 0.0
    if float(getattr(model, "deploy_loss_weight", 0.0)) != 0.0:
        totals["loss_action_deploy"] = 0.0
    if isinstance(model, ACTCVAEPolicy):
        # z=0 / prior-only chunk error (matches rollout — teacher loss uses q(z|o,a)).
        totals["loss_action_prior_z0"] = 0.0
    count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="offline-eval", leave=False):
            if isinstance(model, BeTStyleDiscretePolicy):
                obs = batch["obs"].to(device)
                labels = batch["label"].to(device)
                act_c = batch["action"].to(device)
                _, metrics = model.loss_from_labels(obs, labels, act_c)
            elif "actions" in batch:
                obs = batch["obs"].to(device)
                actions = batch["actions"].to(device)
                is_pad = batch["is_pad"].to(device)
                _, metrics = model.loss(obs=obs, actions=actions, is_pad=is_pad, deterministic=True)
                if isinstance(model, ACTCVAEPolicy):
                    out_p = model.forward(obs=obs, actions=None, is_pad=None, deterministic=True)
                    metrics["loss_action_prior_z0"] = model._action_reconstruction_loss(
                        out_p.pred_actions, actions, is_pad
                    )
            elif "action" in batch:
                # Step dataset (bc_mlp, knn_bc): match train.py packing into a 1-hot chunk.
                obs = batch["obs"].to(device)
                act_single = batch["action"].to(device)
                bsz = obs.shape[0]
                k_max = int(getattr(model, "k_max", 16))
                act_dim = int(act_single.shape[-1])
                actions = torch.zeros(bsz, k_max, act_dim, device=device, dtype=act_single.dtype)
                actions[:, 0] = act_single
                is_pad = torch.ones(bsz, k_max, dtype=torch.bool, device=device)
                is_pad[:, 0] = False
                _, metrics = model.loss(obs=obs, actions=actions, is_pad=is_pad, deterministic=True)
            else:
                raise KeyError(f"offline-eval: expected batch with actions or action; got keys={sorted(batch.keys())}")
            metrics_f = maybe_detach_dict(metrics)
            for key in totals:
                totals[key] += metrics_f.get(key, 0.0)
            count += 1
    if count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / count for k, v in totals.items()}


def make_executor(config: dict[str, Any], model, act_dim: int, device: str, mode: str) -> DynamicChunkExecutor:
    dyn = config["dynamic"]
    ev = config.get("eval", {}) or {}
    action_clip = ev.get("action_clip", 1.0)
    if action_clip is not None:
        action_clip = float(action_clip)
    use_te = mode == "temporal_ensemble" or bool(ev.get("use_act_temporal_ensemble", False))
    te_m = float(ev.get("temporal_ensemble_m", 0.01))
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
        latent_deterministic=not bool(ev.get("sample_latent", False)),
        action_clip=action_clip,
        use_act_temporal_ensemble=use_te,
        temporal_ensemble_m=te_m,
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


def _maybe_add_obs_noise(obs_vec: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    if noise_std <= 0.0:
        return obs_vec
    return obs_vec + rng.normal(0.0, noise_std, size=obs_vec.shape).astype(np.float32)


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
    obs_noise_std = float(config["eval"].get("obs_noise_std", 0.0))
    noise_seed = int(config["eval"].get("obs_noise_seed", int(config.get("seed", 0))))

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
        policy_queries_eps: list[int] = []
        replans_eps: list[int] = []

        for episode_id in tqdm(range(num_episodes), desc=f"rollout-{mode}", leave=False):
            rng = np.random.default_rng(noise_seed + 100_003 * episode_id + sum(ord(c) for c in mode))
            obs_dict = reset_env(env)
            executor.reset_episode()
            done = False
            ep_return = 0.0
            final_success = False
            t = 0

            while not done and t < max_horizon:
                obs_vec = build_obs_vector(obs_dict=obs_dict, obs_keys=obs_keys, obs_normalizer=obs_norm)
                obs_vec = _maybe_add_obs_noise(obs_vec, obs_noise_std, rng)
                result = executor.act(obs=obs_vec, episode_id=episode_id, timestep=t)
                obs_dict, reward, done, info = step_env(env, result.action)
                ep_return += reward
                final_success = final_success or extract_success(env, info, reward)
                t += 1

            executor.finish_episode()
            commit_rows.extend(executor.commit_events)
            step_rows.extend(executor.step_records)

            n_commits = int(len(executor.commit_events))
            replans = max(0, n_commits - 1)
            rule_tag = "learned_commit" if mode == "learned_commit" else str(config["dynamic"]["rule"])
            ep_row = {
                "episode_id": episode_id,
                "mode": mode,
                "rule": rule_tag,
                "success": bool(final_success),
                "return": float(ep_return),
                "length": int(t),
                "num_commit_events": n_commits,
                "replans_per_episode": replans,
                "policy_queries": int(executor.policy_queries),
            }
            episode_rows.append(ep_row)
            success_values.append(float(final_success))
            returns.append(float(ep_return))
            lengths.append(int(t))
            policy_queries_eps.append(int(executor.policy_queries))
            replans_eps.append(replans)

        commit_summary = summarize_commit_events(commit_rows)
        rule_summary = "learned_commit" if mode == "learned_commit" else str(config["dynamic"]["rule"])
        mode_summary = {
            "mode": mode,
            "rule": rule_summary,
            "num_episodes": num_episodes,
            "success_rate": float(mean(success_values)) if success_values else float("nan"),
            "avg_return": float(mean(returns)) if returns else float("nan"),
            "avg_length": float(mean(lengths)) if lengths else float("nan"),
            "avg_policy_queries": float(mean(policy_queries_eps)) if policy_queries_eps else float("nan"),
            "avg_replans_per_episode": float(mean(replans_eps)) if replans_eps else float("nan"),
            "obs_noise_std": obs_noise_std,
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
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset HDF5 path for env load / dataloaders (must match training data).",
    )
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Load weights with strict=False (e.g. old ckpt missing commit_head).",
    )
    parser.add_argument(
        "--no-sample-latent",
        action="store_true",
        help="ACT-style inference: use z=0 (overrides eval.sample_latent in the config).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_sample_latent:
        config.setdefault("eval", {})
        config["eval"]["sample_latent"] = False
    if args.dataset:
        config.setdefault("dataset", {})
        config["dataset"]["path"] = os.path.expanduser(args.dataset)
    set_seed(int(config.get("seed", 0)))
    dirs = prepare_output_dirs(config)

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    metadata = ckpt.get("extra", {}).get("dataset_metadata")
    if metadata is None:
        raise RuntimeError("Checkpoint is missing dataset_metadata in the extra payload.")

    config = dict(config)
    if "config" in ckpt and isinstance(ckpt["config"], dict) and "model" in ckpt["config"]:
        merged_m = dict(config.get("model", {}))
        merged_m.update(ckpt["config"]["model"])
        config["model"] = merged_m
    config["dataset"] = dict(config["dataset"])
    config["dataset"]["obs_keys"] = list(metadata["obs_keys"])
    device = device_from_config(config)
    ckpt_policy = str(config.get("model", {}).get("policy_type", "act_cvae")).lower()
    if ckpt_policy == "knn_bc":
        state = ckpt["model_state_dict"]
        obs_bank = state.get("obs_bank")
        act_bank = state.get("act_bank")
        if obs_bank is None or act_bank is None:
            raise RuntimeError("kNN checkpoint is missing obs_bank or act_bank tensors.")
        model = KNNBCPolicy(
            obs_bank=obs_bank,
            act_bank=act_bank,
            k_neighbors=int(config["model"].get("knn_k", 5)),
            k_max=int(config["model"]["k_max"]),
            act_dim=int(metadata["act_dim"]),
            action_loss=str(config["model"].get("action_loss", "l1")),
        ).to(device)
    else:
        model = build_policy(config, obs_dim=int(metadata["obs_dim"]), act_dim=int(metadata["act_dim"])).to(device)
    load_res = model.load_state_dict(ckpt["model_state_dict"], strict=not args.allow_partial_checkpoint)
    if args.allow_partial_checkpoint and (load_res.missing_keys or load_res.unexpected_keys):
        print("Partial load:", load_res)
    model.eval()

    if ckpt_policy == "bet_dbc":
        cent_np = model.centroids.detach().cpu().numpy()
        val_dataset = build_bet_val_dataset_for_eval(config, cent_np)
    elif ckpt_policy in ("bc_mlp", "knn_bc"):
        _train_dataset, val_dataset, _unused = build_step_datasets(config)
    else:
        _train_dataset, val_dataset, _unused = build_datasets(config)

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
