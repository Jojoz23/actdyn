from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(0, num_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeTStyleDiscretePolicy(nn.Module):
    """
    Discrete behavior token (k-means cluster) from state, decode with frozen centroids — BeT-style baseline.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        k_max: int,
        num_bins: int,
        centroids: torch.Tensor,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.k_max = int(k_max)
        self.num_bins = int(num_bins)
        self.register_buffer("centroids", centroids.float().contiguous())
        self.mlp = MLP(obs_dim, hidden_dims[0], num_bins, num_layers=len(hidden_dims) + 1, dropout=dropout)

    def forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)

    def decode_actions(self, logits: torch.Tensor) -> torch.Tensor:
        idx = logits.argmax(dim=-1)
        return self.centroids[idx]

    def loss_from_labels(
        self,
        obs: torch.Tensor,
        labels: torch.Tensor,
        action_continuous: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.forward_logits(obs)
        loss = F.cross_entropy(logits, labels)
        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_action": loss.detach(),
            "loss_kl": torch.zeros((), device=obs.device, dtype=obs.dtype),
        }
        if action_continuous is not None:
            pred_a = self.decode_actions(logits)
            metrics["loss_action"] = F.l1_loss(pred_a, action_continuous)
        return loss, metrics

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del deterministic, is_pad
        labels = actions[:, 0, 0].long()
        act_c = actions[:, 0, 1:]
        return self.loss_from_labels(obs, labels, act_c)

    @torch.no_grad()
    def predict_chunk(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> torch.Tensor:
        del deterministic
        if num_samples != 1:
            raise ValueError("BeTStyleDiscretePolicy only supports num_samples=1")
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        logits = self.forward_logits(obs)
        a = self.decode_actions(logits)
        chunk = a.unsqueeze(1).expand(-1, self.k_max, -1)
        if chunk.shape[0] == 1:
            return chunk[0]
        return chunk

    @property
    def use_commit_head(self) -> bool:
        return False
