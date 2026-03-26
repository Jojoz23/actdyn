from __future__ import annotations

import numpy as np
import torch

from actdyn.utils.misc import clamp_int, linear_interpolate


def overlap_disagreement_score(candidate_chunk: np.ndarray, remaining_queue: np.ndarray, overlap_window: int) -> float:
    if candidate_chunk.ndim != 2 or remaining_queue.ndim != 2:
        raise ValueError("candidate_chunk and remaining_queue must have shape (T, act_dim).")
    m = min(int(overlap_window), candidate_chunk.shape[0], remaining_queue.shape[0])
    if m <= 0:
        return 0.0
    diffs = candidate_chunk[:m] - remaining_queue[:m]
    norms = np.linalg.norm(diffs, axis=-1)
    return float(norms.mean())


def plan_commit_length_from_overlap(score: float, threshold: float, k_min: int, k_max: int) -> int:
    if score <= threshold:
        return int(k_max)
    ratio = min(score / max(threshold, 1e-8), 4.0)
    scaled = round(linear_interpolate(ratio, 1.0, 4.0, float(k_max), float(k_min)))
    return clamp_int(int(scaled), k_min, k_max)


def action_delta_scores(chunk: np.ndarray) -> np.ndarray:
    if chunk.shape[0] <= 1:
        return np.zeros((chunk.shape[0],), dtype=np.float32)
    deltas = np.linalg.norm(np.diff(chunk, axis=0), axis=-1)
    scores = np.zeros((chunk.shape[0],), dtype=np.float32)
    scores[0] = 0.0
    scores[1:] = deltas.astype(np.float32)
    return scores


def plan_commit_length_from_deltas(delta_scores: np.ndarray, threshold: float, k_min: int, k_max: int) -> int:
    if delta_scores.ndim != 1:
        raise ValueError("delta_scores must be 1D.")
    commit = k_max
    for step in range(1, min(k_max, delta_scores.shape[0])):
        if delta_scores[step] > threshold:
            commit = step
            break
    return clamp_int(int(commit), k_min, k_max)


def dispersion_scores(sampled_chunks: np.ndarray) -> np.ndarray:
    """
    sampled_chunks: (S, K, A)
    returns: (K,)
    """
    if sampled_chunks.ndim != 3:
        raise ValueError("sampled_chunks must have shape (S, K, A).")
    std = sampled_chunks.std(axis=0)  # (K, A)
    return np.linalg.norm(std, axis=-1).astype(np.float32)


def plan_commit_length_from_dispersion(
    per_step_dispersion: np.ndarray,
    threshold: float,
    k_min: int,
    k_max: int,
) -> int:
    commit = k_max
    for step in range(1, min(k_max, per_step_dispersion.shape[0]) + 1):
        if per_step_dispersion[step - 1] > threshold:
            commit = step - 1
            break
    return clamp_int(int(max(commit, k_min)), k_min, k_max)


def torch_chunk_to_numpy(chunk: torch.Tensor) -> np.ndarray:
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    return chunk.detach().cpu().numpy().astype(np.float32)
