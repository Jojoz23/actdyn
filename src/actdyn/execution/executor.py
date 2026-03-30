from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from actdyn.execution.heuristics import (
    action_delta_scores,
    dispersion_scores,
    overlap_disagreement_score,
    plan_commit_length_from_deltas,
    plan_commit_length_from_dispersion,
    plan_commit_length_from_overlap,
    torch_chunk_to_numpy,
)
from actdyn.utils.misc import clamp_int


@dataclass
class CommitEvent:
    episode_id: int
    mode: str
    rule: str
    start_timestep: int
    planned_length: int
    realized_length: int = 0
    score: float = 0.0
    score_name: str = ""
    replan_reason: str = "queue_empty"


@dataclass
class StepRecord:
    episode_id: int
    timestep: int
    mode: str
    rule: str
    queue_len_before: int
    queue_len_after: int
    replanned: bool
    score: float
    score_name: str
    action_norm: float


@dataclass
class ExecutionResult:
    action: np.ndarray
    replanned: bool
    score: float
    score_name: str


@dataclass
class DynamicChunkExecutor:
    policy: Any
    act_dim: int
    k_max: int
    mode: str
    rule: str
    k_min: int
    overlap_window: int
    overlap_threshold: float
    delta_threshold: float
    uncertainty_threshold: float
    num_uncertainty_samples: int
    device: str = "cuda"
    # CVAE inference: deterministic=True uses z=0 in the prior; False samples z~N(0,I) each query.
    latent_deterministic: bool = True
    # RoboMimic low-dim actions live in [-1, 1]; unbounded MLP heads can hurt sim if not clipped.
    action_clip: float | None = 1.0
    # True when eval mode is temporal_ensemble: ACT-style query every step + bin blend (Algorithm 2).
    use_act_temporal_ensemble: bool = False
    temporal_ensemble_m: float = 0.01

    queue: list[np.ndarray] = field(default_factory=list)
    current_commit: CommitEvent | None = None
    commit_events: list[dict[str, Any]] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)
    policy_queries: int = 0
    _te_buffers: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))

    def reset_episode(self) -> None:
        self.queue.clear()
        self.current_commit = None
        self.commit_events.clear()
        self.step_records.clear()
        self.policy_queries = 0
        self._te_buffers = defaultdict(list)

    def _finalize_current_commit(self, reason: str | None = None) -> None:
        if self.current_commit is None:
            return
        if reason is not None:
            self.current_commit.replan_reason = reason
        self.commit_events.append(self.current_commit.__dict__.copy())
        self.current_commit = None

    def _predict_deterministic_chunk(self, obs: np.ndarray) -> np.ndarray:
        self.policy_queries += 1
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        chunk = self.policy.predict_chunk(
            obs_t, deterministic=self.latent_deterministic, num_samples=1
        )
        return torch_chunk_to_numpy(chunk)

    def _predict_sampled_chunks(self, obs: np.ndarray, num_samples: int) -> np.ndarray:
        self.policy_queries += int(num_samples)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        sampled = self.policy.predict_chunk(obs_t, deterministic=False, num_samples=num_samples)
        return sampled[:, 0].detach().cpu().numpy().astype(np.float32)

    def _predict_commit_length(self, obs: np.ndarray) -> int:
        """One forward on the commit head (proposal: learned \\hat m)."""
        if not hasattr(self.policy, "predict_commit_length"):
            raise RuntimeError("Policy has no predict_commit_length; train with model.use_commit_head=true.")
        self.policy_queries += 1
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.policy.predict_commit_length(obs_t))

    def _remaining_queue_array(self) -> np.ndarray:
        if not self.queue:
            return np.zeros((0, self.act_dim), dtype=np.float32)
        return np.stack(self.queue, axis=0).astype(np.float32)

    def _start_new_commit(
        self,
        episode_id: int,
        timestep: int,
        planned_length: int,
        score: float,
        score_name: str,
        reason: str,
    ) -> None:
        self.current_commit = CommitEvent(
            episode_id=episode_id,
            mode=self.mode,
            rule=self.rule,
            start_timestep=timestep,
            planned_length=int(planned_length),
            realized_length=0,
            score=float(score),
            score_name=score_name,
            replan_reason=reason,
        )

    def _plan_new_queue(
        self,
        obs: np.ndarray,
        episode_id: int,
        timestep: int,
        reason: str,
    ) -> tuple[bool, float, str]:
        score = 0.0
        score_name = ""

        if self.mode == "full_chunk":
            chunk = self._predict_deterministic_chunk(obs)
            planned_length = self.k_max

        elif self.mode == "receding_horizon":
            chunk = self._predict_deterministic_chunk(obs)
            planned_length = 1

        elif self.mode == "learned_commit":
            chunk = self._predict_deterministic_chunk(obs)
            planned_length = self._predict_commit_length(obs)
            planned_length = clamp_int(planned_length, self.k_min, self.k_max)
            score = float(planned_length)
            score_name = "learned_m"
            self.queue = [chunk[i].astype(np.float32) for i in range(planned_length)]
            self._start_new_commit(
                episode_id=episode_id,
                timestep=timestep,
                planned_length=planned_length,
                score=score,
                score_name=score_name,
                reason=reason,
            )
            return True, score, score_name

        else:
            if self.rule == "action_change_magnitude":
                chunk = self._predict_deterministic_chunk(obs)
                delta_scores = action_delta_scores(chunk)
                score = float(delta_scores.max()) if delta_scores.size > 0 else 0.0
                score_name = "delta_max"
                planned_length = plan_commit_length_from_deltas(
                    delta_scores=delta_scores,
                    threshold=self.delta_threshold,
                    k_min=self.k_min,
                    k_max=self.k_max,
                )
            elif self.rule == "stochastic_uncertainty":
                chunk = self._predict_deterministic_chunk(obs)
                sampled = self._predict_sampled_chunks(obs, self.num_uncertainty_samples)
                scores = dispersion_scores(sampled)
                score = float(scores.max()) if scores.size > 0 else 0.0
                score_name = "dispersion_max"
                planned_length = plan_commit_length_from_dispersion(
                    per_step_dispersion=scores,
                    threshold=self.uncertainty_threshold,
                    k_min=self.k_min,
                    k_max=self.k_max,
                )
            elif self.rule == "overlap_disagreement":
                chunk = self._predict_deterministic_chunk(obs)
                planned_length = self.k_max
                score = 0.0
                score_name = "overlap_disagreement"
            else:
                raise ValueError(f"Unsupported dynamic rule: {self.rule}")

        self.queue = [chunk[i].astype(np.float32) for i in range(planned_length)]
        self._start_new_commit(
            episode_id=episode_id,
            timestep=timestep,
            planned_length=planned_length,
            score=score,
            score_name=score_name,
            reason=reason,
        )
        return True, score, score_name

    def _act_temporal_ensemble(
        self,
        obs: np.ndarray,
        episode_id: int,
        timestep: int,
    ) -> ExecutionResult:
        """ACT Algorithm 2 style: deposit full chunk into per-timestep bins; blend with exp(-m*i) (oldest i=0)."""
        chunk = self._predict_deterministic_chunk(obs)
        t = int(timestep)
        for j in range(self.k_max):
            self._te_buffers[t + j].append((t, chunk[j].astype(np.float32, copy=True)))

        entries = self._te_buffers.pop(t, [])
        if not entries:
            raise RuntimeError(f"temporal_ensemble: empty buffer at timestep {t}")

        entries.sort(key=lambda x: x[0])
        actions = np.stack([a for _, a in entries], axis=0).astype(np.float64)
        m = float(self.temporal_ensemble_m)
        w = np.exp(-m * np.arange(len(entries), dtype=np.float64))
        w = w / (w.sum() + 1e-12)
        action = (w[:, None] * actions).sum(axis=0).astype(np.float32)

        if self.action_clip is not None and float(self.action_clip) > 0.0:
            c = float(self.action_clip)
            action = np.clip(action, -c, c).astype(np.float32, copy=False)

        score_name = "temporal_ensemble"
        self.step_records.append(
            StepRecord(
                episode_id=episode_id,
                timestep=t,
                mode=self.mode,
                rule=score_name,
                queue_len_before=len(entries),
                queue_len_after=0,
                replanned=True,
                score=float(len(entries)),
                score_name=score_name,
                action_norm=float(np.linalg.norm(action)),
            ).__dict__.copy()
        )
        return ExecutionResult(action=action, replanned=True, score=float(len(entries)), score_name=score_name)

    def _maybe_preempt_overlap(
        self,
        obs: np.ndarray,
        timestep: int,
        episode_id: int,
    ) -> tuple[bool, float, str]:
        if self.mode != "dynamic" or self.rule != "overlap_disagreement" or not self.queue:
            return False, 0.0, ""
        candidate = self._predict_deterministic_chunk(obs)
        remaining = self._remaining_queue_array()
        score = overlap_disagreement_score(
            candidate_chunk=candidate,
            remaining_queue=remaining,
            overlap_window=self.overlap_window,
        )
        score_name = "overlap_disagreement"
        if score > self.overlap_threshold:
            self._finalize_current_commit(reason="overlap_preempt")
            planned_length = plan_commit_length_from_overlap(
                score=score,
                threshold=self.overlap_threshold,
                k_min=self.k_min,
                k_max=self.k_max,
            )
            self.queue = [candidate[i].astype(np.float32) for i in range(planned_length)]
            self._start_new_commit(
                episode_id=episode_id,
                timestep=timestep,
                planned_length=planned_length,
                score=score,
                score_name=score_name,
                reason="overlap_preempt",
            )
            return True, score, score_name
        return False, score, score_name

    def act(self, obs: np.ndarray, episode_id: int, timestep: int) -> ExecutionResult:
        if self.use_act_temporal_ensemble:
            return self._act_temporal_ensemble(obs, episode_id, timestep)
        replanned = False
        score = 0.0
        score_name = ""

        if not self.queue:
            replanned, score, score_name = self._plan_new_queue(obs, episode_id, timestep, reason="queue_empty")
        else:
            replanned, score, score_name = self._maybe_preempt_overlap(obs, timestep, episode_id)

        queue_len_before = len(self.queue)
        action = self.queue.pop(0)
        if self.action_clip is not None and float(self.action_clip) > 0.0:
            c = float(self.action_clip)
            action = np.clip(action, -c, c).astype(np.float32, copy=False)
        if self.current_commit is not None:
            self.current_commit.realized_length += 1
        queue_len_after = len(self.queue)
        if queue_len_after == 0:
            self._finalize_current_commit(reason="queue_drained")

        self.step_records.append(
            StepRecord(
                episode_id=episode_id,
                timestep=timestep,
                mode=self.mode,
                rule=self.rule,
                queue_len_before=queue_len_before,
                queue_len_after=queue_len_after,
                replanned=replanned,
                score=float(score),
                score_name=score_name,
                action_norm=float(np.linalg.norm(action)),
            ).__dict__.copy()
        )
        return ExecutionResult(action=action, replanned=replanned, score=float(score), score_name=score_name)

    def finish_episode(self) -> None:
        self._finalize_current_commit(reason="episode_end")
