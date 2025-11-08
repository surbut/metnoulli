"""
MetNoulli Latent Pathway Model
==============================

Lightweight PyTorch implementation inspired by the Aladynoulli architecture.
Provides:
    • Latent pathway mixture (lambda → theta) per patient.
    • Site/time-specific pathway deviations (phi) plus hot-start ψ offsets.
    • Baseline logit curve per site supplied externally.
    • Loss function matching the discrete-time metastasis likelihood.

Intended for experimentation on simulated data before wiring into the full pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class MetNoulliBatch:
    Y: torch.Tensor  # (N, D, T)
    event_times: torch.Tensor  # (N, D)
    covariates: torch.Tensor  # (N, P)


class MetNoulliLatentModel(nn.Module):
    def __init__(
        self,
        n_patients: int,
        n_sites: int,
        n_time: int,
        n_pathways: int,
        n_covariates: int,
        baseline_logit: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        psi_init: Optional[np.ndarray] = None,
        gamma_scale: float = 0.05,
    ):
        super().__init__()
        self.N = n_patients
        self.D = n_sites
        self.T = n_time
        self.K = n_pathways
        self.P = n_covariates

        self.register_buffer("baseline_logit", baseline_logit.clone())  # (D, T)

        if covariates is None:
            covariates = torch.zeros(n_patients, n_covariates)
        self.register_buffer("covariates", covariates.clone())

        self.lambda_ = nn.Parameter(torch.randn(n_patients, n_pathways, n_time) * 0.05)
        self.phi = nn.Parameter(torch.randn(n_pathways, n_sites, n_time) * 0.05)
        self.psi = nn.Parameter(
            torch.tensor(psi_init, dtype=torch.float32)
            if psi_init is not None
            else torch.zeros(n_pathways, n_sites)
        )
        self.gamma = nn.Parameter(torch.randn(n_covariates, n_pathways) * gamma_scale)
        self.register_buffer("eps", torch.tensor(1e-8))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cov_effect = torch.matmul(self.covariates, self.gamma).unsqueeze(-1)  # (N,K,1)
        lambda_total = self.lambda_ + cov_effect

        theta = torch.softmax(lambda_total, dim=1)  # (N,K,T)
        phi_logits = self.phi + self.psi.unsqueeze(-1)  # (K,D,T)

        pathway_effect = torch.einsum("nkt,kdt->ndt", theta, phi_logits)
        logits = self.baseline_logit.unsqueeze(0) + pathway_effect
        probs = torch.sigmoid(logits)
        return probs, theta

    def compute_loss(self, batch: MetNoulliBatch) -> torch.Tensor:
        probs, _ = self.forward()
        probs = torch.clamp(probs, self.eps, 1 - self.eps)

        N, D, T = probs.shape
        event_times = batch.event_times.long()
        time_grid = torch.arange(T, device=probs.device).view(1, 1, T)

        mask_before = (time_grid < event_times.unsqueeze(-1)).float()
        mask_at = (time_grid == event_times.unsqueeze(-1)).float()

        loss_censored = -(torch.log(1 - probs) * mask_before).sum()
        loss_event = -(torch.log(probs) * mask_at * batch.Y).sum()
        loss_no_event = -(torch.log(1 - probs) * mask_at * (1 - batch.Y)).sum()

        total_loss = (loss_censored + loss_event + loss_no_event) / N
        reg = 0.5 * (self.psi ** 2).mean() + 0.5 * (self.phi ** 2).mean()
        return total_loss + 1e-3 * reg


def train_metnoulli_model(
    model: MetNoulliLatentModel,
    batch: MetNoulliBatch,
    n_epochs: int = 200,
    learning_rate: float = 1e-2,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == n_epochs):
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")

    with torch.no_grad():
        probs, theta = model.forward()

    return {
        "losses": np.array(losses),
        "probs": probs.cpu().numpy(),
        "theta": theta.cpu().numpy(),
        "lambda": model.lambda_.detach().cpu().numpy(),
        "phi": model.phi.detach().cpu().numpy(),
        "psi": model.psi.detach().cpu().numpy(),
    }

