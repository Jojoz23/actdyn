from __future__ import annotations

import numpy as np


def action_kmeans(
    actions: np.ndarray,
    num_bins: int,
    seed: int,
    num_iters: int = 35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lloyd's algorithm on continuous actions. actions: (N, D).
    Returns centroids (K, D), assign (N,) int64 in [0, K).
    """
    n, d = actions.shape
    k = min(int(num_bins), n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    centroids = actions[idx].astype(np.float32).copy()
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(num_iters):
        dist = np.sum((actions[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assign = dist.argmin(axis=1)
        for j in range(k):
            mask = assign == j
            if np.any(mask):
                centroids[j] = actions[mask].mean(axis=0)
            else:
                centroids[j] = actions[rng.integers(0, n)]
    return centroids.astype(np.float32), assign


def assign_actions_to_centroids(actions: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dist = np.sum((actions[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return dist.argmin(axis=1).astype(np.int64)
