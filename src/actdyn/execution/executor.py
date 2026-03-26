from __future__ import annotations

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

    queue: list[np.ndarray] = field(default_factory=list)
    current_commit: CommitEvent | None = None
    commit_events: list[dict[str, Any]] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)

    def reset_episode(self) -> None:
        self.queue.clear()
        self.current_commit = None
        self.commit_events.clear()
        self.step_records.clear()

    def _finalize_current_commit(self, reason: str | None = None) -> None:
        if self.current_commit is None:
            return
        if reason is not None:
            self.current_commit.replan_reason = reason
        self.commit_events.append(self.current_commit.__dict__.copy())
        self.current_commit = None

    def _predict_deterministic_chunk(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        chunk = self.policy.predict_chunk(obs_t, deterministic=True, num_samples=1)
        return torch_chunk_to_numpy(chunk)

    def _predict_sampled_chunks(self, obs: np.ndarray, num_samples: int) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        sampled = self.policy.predict_chunk(obs_t, deterministic=False, num_samples=num_samples)
        return sampled[:, 0].detach().cpu().numpy().astype(np.float32)

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
        replanned = False
        score = 0.0
        score_name = ""

        if not self.queue:
            replanned, score, score_name = self._plan_new_queue(obs, episode_id, timestep, reason="queue_empty")
        else:
            replanned, score, score_name = self._maybe_preempt_overlap(obs, timestep, episode_id)

        queue_len_before = len(self.queue)
        action = self.queue.pop(0)
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
