from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class KNNBCPolicy(nn.Module):
    """
    kNN over stored (obs, action) pairs from the train split — a lightweight
    **VINN-style** nonparametric imitation baseline (not the full VINN system).
    """

    def __init__(
        self,
        obs_bank: torch.Tensor,
        act_bank: torch.Tensor,
        k_neighbors: int,
        k_max: int,
        act_dim: int,
        action_loss: str = "l1",
    ) -> None:
        super().__init__()
        self.register_buffer("obs_bank", obs_bank.float().contiguous())
        self.register_buffer("act_bank", act_bank.float().contiguous())
        self.k_neighbors = max(1, int(k_neighbors))
        self.k_max = int(k_max)
        self.act_dim = int(act_dim)
        self.action_loss = str(action_loss)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        dists = torch.cdist(obs, self.obs_bank, p=2.0)
        k = min(self.k_neighbors, self.obs_bank.shape[0])
        _, idx = dists.topk(k, largest=False, dim=1)
        acts = self.act_bank[idx].mean(dim=1)
        return acts

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del deterministic
        target = actions[:, 0, :]
        pred = self.forward(obs)
        if self.action_loss == "mse":
            per = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        else:
            per = F.l1_loss(pred, target, reduction="none").mean(dim=-1)
        valid = (~is_pad[:, 0]).float()
        denom = valid.sum().clamp_min(1.0)
        action_loss = (per * valid).sum() / denom
        metrics = {
            "loss": action_loss,
            "loss_action": action_loss,
            "loss_kl": torch.zeros((), device=obs.device, dtype=obs.dtype),
        }
        return action_loss, metrics

    @torch.no_grad()
    def predict_chunk(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> torch.Tensor:
        del deterministic
        if num_samples != 1:
            raise ValueError("KNNBCPolicy only supports num_samples=1")
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        a = self.forward(obs)
        chunk = a.unsqueeze(1).expand(-1, self.k_max, -1)
        if chunk.shape[0] == 1:
            return chunk[0]
        return chunk

    @property
    def use_commit_head(self) -> bool:
        return False
