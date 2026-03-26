from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from actdyn.utils.misc import maybe_detach_dict


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
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.k_max = int(k_max)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.kl_beta = float(kl_beta)
        self.action_loss = str(action_loss)

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

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self.forward(obs=obs, actions=actions, is_pad=is_pad, deterministic=deterministic)
        if self.action_loss == "mse":
            per_elem = F.mse_loss(out.pred_actions, actions, reduction="none")
        else:
            per_elem = F.l1_loss(out.pred_actions, actions, reduction="none")
        mask = (~is_pad).float().unsqueeze(-1)
        action_loss = (per_elem * mask).sum() / mask.sum().clamp_min(1.0)

        assert out.mu is not None and out.logvar is not None
        kl = -0.5 * (1 + out.logvar - out.mu.pow(2) - out.logvar.exp())
        kl = kl.mean()
        total = action_loss + self.kl_beta * kl
        metrics = {
            "loss": total,
            "loss_action": action_loss,
            "loss_kl": kl,
        }
        return total, metrics

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
            return out.pred_actions

        chunks = []
        for _ in range(num_samples):
            out = self.forward(obs=obs, deterministic=False)
            chunks.append(out.pred_actions.unsqueeze(0))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def eval_loss_dict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> dict[str, float]:
        _, metrics = self.loss(obs=obs, actions=actions, is_pad=is_pad)
        return maybe_detach_dict(metrics)
