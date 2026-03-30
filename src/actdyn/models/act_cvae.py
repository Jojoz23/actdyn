from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from actdyn.execution.heuristics import expert_commit_length_from_chunk
from actdyn.utils.misc import clamp_int, maybe_detach_dict


@dataclass
class PolicyOutput:
    pred_actions: torch.Tensor
    mu: torch.Tensor | None
    logvar: torch.Tensor | None
    z: torch.Tensor | None
    metrics: dict[str, float] | dict[str, torch.Tensor]


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


class ACTCVAEPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        k_max: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
        action_loss: str = "l1",
        kl_beta: float = 1e-4,
        use_commit_head: bool = False,
        commit_loss_weight: float = 0.0,
        commit_label_delta_threshold: float = 0.12,
        commit_k_min: int = 1,
        # Train decoder with z~p(z) (no future-action encoder). Matches rollout/predict_chunk; avoids collapsing into q(z|o,a).
        deploy_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.k_max = int(k_max)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.kl_beta = float(kl_beta)
        self.action_loss = str(action_loss)
        self.use_commit_head = bool(use_commit_head)
        self.commit_loss_weight = float(commit_loss_weight)
        self.commit_label_delta_threshold = float(commit_label_delta_threshold)
        self.commit_k_min = int(commit_k_min)
        self.deploy_loss_weight = float(deploy_loss_weight)

        self.obs_encoder = MLP(obs_dim, d_model, d_model, num_layers=2, dropout=dropout)
        self.action_encoder = nn.Linear(act_dim, d_model)
        self.query_embed = nn.Parameter(torch.randn(k_max, d_model) * 0.02)
        self.output_pos_embed = nn.Parameter(torch.randn(k_max, d_model) * 0.02)
        self.obs_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.prior_proj = nn.Linear(latent_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.inference_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, act_dim),
        )
        self.commit_head: nn.Module | None
        if self.use_commit_head:
            self.commit_head = nn.Linear(d_model, k_max)
        else:
            self.commit_head = None

    def encode_posterior(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        obs_token = self.obs_token.expand(batch_size, -1, -1) + self.obs_encoder(obs).unsqueeze(1)
        action_tokens = self.action_encoder(actions) + self.output_pos_embed.unsqueeze(0)
        enc_in = torch.cat([obs_token, action_tokens], dim=1)
        padding_mask = torch.cat(
            [torch.zeros((batch_size, 1), dtype=torch.bool, device=is_pad.device), is_pad],
            dim=1,
        )
        enc_out = self.inference_encoder(enc_in, src_key_padding_mask=padding_mask)
        pooled = enc_out[:, 0]
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(min=-10.0, max=10.0)
        return mu, logvar

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        obs_context = self.obs_encoder(obs)
        z_context = self.prior_proj(z)
        decoder_tokens = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        decoder_tokens = decoder_tokens + obs_context.unsqueeze(1) + z_context.unsqueeze(1)
        dec_out = self.decoder(decoder_tokens)
        pred_actions = self.action_head(dec_out)
        return pred_actions

    def sample_prior(self, batch_size: int, device: torch.device, deterministic: bool) -> torch.Tensor:
        if deterministic:
            return torch.zeros((batch_size, self.latent_dim), device=device)
        return torch.randn((batch_size, self.latent_dim), device=device)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if actions is not None and is_pad is not None:
            mu, logvar = self.encode_posterior(obs, actions, is_pad)
            z = self.reparameterize(mu, logvar, deterministic=deterministic)
        else:
            mu = None
            logvar = None
            z = self.sample_prior(obs.shape[0], obs.device, deterministic=deterministic)

        pred_actions = self.decode(obs, z)
        return PolicyOutput(
            pred_actions=pred_actions,
            mu=mu,
            logvar=logvar,
            z=z,
            metrics={},
        )

    def _action_reconstruction_loss(
        self,
        pred_logits: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Same squashing as predict_chunk (tanh); train/deploy target space match."""
        pred = torch.tanh(pred_logits)
        if self.action_loss == "mse":
            per_elem = F.mse_loss(pred, actions, reduction="none")
        else:
            per_elem = F.l1_loss(pred, actions, reduction="none")
        mask = (~is_pad).float().unsqueeze(-1)
        return (per_elem * mask).sum() / mask.sum().clamp_min(1.0)

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self.forward(obs=obs, actions=actions, is_pad=is_pad, deterministic=deterministic)
        action_loss = self._action_reconstruction_loss(out.pred_actions, actions, is_pad)

        assert out.mu is not None and out.logvar is not None
        kl = -0.5 * (1 + out.logvar - out.mu.pow(2) - out.logvar.exp())
        kl = kl.mean()
        total = action_loss + self.kl_beta * kl
        metrics: dict[str, torch.Tensor] = {
            "loss": total,
            "loss_action": action_loss,
            "loss_kl": kl,
        }

        w = float(self.deploy_loss_weight)
        if w != 0.0:
            out_prior = self.forward(obs=obs, actions=None, is_pad=None, deterministic=deterministic)
            action_loss_deploy = self._action_reconstruction_loss(out_prior.pred_actions, actions, is_pad)
            total = total + w * action_loss_deploy
            metrics["loss_action_deploy"] = action_loss_deploy
            metrics["loss"] = total

        if self.use_commit_head and self.commit_head is not None:
            h = self.obs_encoder(obs)
            logits = self.commit_head(h)
            targets = self._commit_class_targets(actions, is_pad)
            commit_loss = F.cross_entropy(logits, targets)
            total = total + self.commit_loss_weight * commit_loss
            metrics["loss_commit"] = commit_loss
            metrics["loss"] = total

        return total, metrics

    def _commit_class_targets(self, actions: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        b = actions.shape[0]
        labels: list[int] = []
        for i in range(b):
            chunk = actions[i].detach().float().cpu().numpy()
            pad = is_pad[i].detach().cpu().numpy().astype(bool)
            m = expert_commit_length_from_chunk(
                chunk,
                pad,
                self.commit_label_delta_threshold,
                self.commit_k_min,
                self.k_max,
            )
            labels.append(m - 1)
        return torch.tensor(labels, device=actions.device, dtype=torch.long)

    @torch.no_grad()
    def predict_chunk(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if num_samples == 1:
            out = self.forward(obs=obs, deterministic=deterministic)
            # Demos use normalized actions in [-1, 1]; tanh keeps rollouts in-range (linear head can extrapolate).
            return torch.tanh(out.pred_actions)

        chunks = []
        for _ in range(num_samples):
            out = self.forward(obs=obs, deterministic=False)
            chunks.append(torch.tanh(out.pred_actions).unsqueeze(0))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def predict_commit_length(self, obs: torch.Tensor) -> int:
        """Argmax commit class -> length in [commit_k_min, k_max]. Requires use_commit_head=True."""
        if not self.use_commit_head or self.commit_head is None:
            raise RuntimeError("predict_commit_length requires model.use_commit_head=True.")
        self.eval()
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        h = self.obs_encoder(obs)
        logits = self.commit_head(h)
        m = int(logits.argmax(dim=-1).item()) + 1
        return int(clamp_int(m, self.commit_k_min, self.k_max))

    @torch.no_grad()
    def eval_loss_dict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> dict[str, float]:
        _, metrics = self.loss(obs=obs, actions=actions, is_pad=is_pad)
        return maybe_detach_dict(metrics)
