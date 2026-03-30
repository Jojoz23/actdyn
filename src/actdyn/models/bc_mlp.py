from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from actdyn.utils.misc import maybe_detach_dict


class BCMLPPolicy(nn.Module):
    """
    Single-step behavioral cloning MLP (proposal BC-ConvMLP analogue on low-dim state: no conv).
    Intended for receding-horizon rollouts: one action per replan. predict_chunk repeats the same
    action k_max times for API compatibility; only index 0 is used when planned_length=1.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        k_max: int,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        dropout: float = 0.1,
        action_loss: str = "l1",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.k_max = int(k_max)
        self.action_loss = str(action_loss)
        dims = [obs_dim] + list(hidden_dims) + [act_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

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
            raise ValueError("BCMLPPolicy only supports num_samples=1")
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
