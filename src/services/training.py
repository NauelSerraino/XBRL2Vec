"""
Training loop for the masked autoencoder.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from services.config import DEVICE
from services.data import ModelType, TrainConfig


# ---------------------------------------------------------------------------
# Epoch metric container
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    epoch: int
    mse: float
    mae: float
    smooth: float


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MaskedAETrainer:
    """Encapsulates one training run for a contextual or blind AE model."""

    def __init__(self, config: TrainConfig, model_type: ModelType):
        self.config = config
        self.model_type = model_type

    def _seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

    def train(
        self,
        model: torch.nn.Module,
        X_fin: torch.Tensor,
        X_macro: torch.Tensor,
        *,
        alpha: float = 0.0,
        repeats: int = 10,
        device: torch.device = DEVICE,
    ) -> tuple[torch.nn.Module, list[EpochMetrics]]:
        """
        Train the model and return (trained_model, per-epoch metrics).

        Args:
            alpha:   Weight for masked vs full reconstruction loss.
                     0 = standard AE (no masking penalty).
            repeats: Number of augmented forward passes per batch.
        """
        self._seed()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        g = torch.Generator(device="cpu").manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(X_fin, X_macro),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
        )

        history: list[EpochMetrics] = []

        for epoch in range(self.config.epochs):
            model.train()
            total_mse = total_mae = total_smooth = 0.0
            n = 0

            for x_fin_b, x_mac_b in loader:
                x_fin_b = x_fin_b.to(device)
                x_mac_b = x_mac_b.to(device)

                for r in range(repeats):
                    optimizer.zero_grad()

                    if self.config.use_mask and self.config.mask_prob > 0:
                        mask_gen = torch.Generator(device="cpu").manual_seed(
                            self.config.seed + r + n
                        )
                        mask = (
                            torch.rand(x_fin_b.shape, generator=mask_gen).to(device)
                            < self.config.mask_prob
                        )
                        x_fin_input = x_fin_b.clone()
                        x_fin_input[mask] = 0.0

                        _, x_hat = model(x_fin_input, x_mac_b)
                        loss = (
                            alpha * F.mse_loss(x_hat[mask], x_fin_b[mask])
                            + (1 - alpha) * F.mse_loss(x_hat, x_fin_b)
                        )
                    else:
                        _, x_hat = model(x_fin_b, x_mac_b)
                        loss = F.mse_loss(x_hat, x_fin_b)

                    loss.backward()
                    optimizer.step()

                    bs_eff = x_fin_b.size(0)
                    n += bs_eff
                    total_mse += loss.item() * bs_eff

                    with torch.no_grad():
                        total_mae    += F.l1_loss(x_hat, x_fin_b).item() * bs_eff
                        total_smooth += F.smooth_l1_loss(x_hat, x_fin_b).item() * bs_eff

            metrics = EpochMetrics(
                epoch  = epoch,
                mse    = total_mse / n,
                mae    = total_mae / n,
                smooth = total_smooth / n,
            )
            history.append(metrics)

            mlflow.log_metrics(
                {
                    f"mse_{self.model_type.value}":      metrics.mse,
                    f"mae_{self.model_type.value}":      metrics.mae,
                    f"smooth_l1_{self.model_type.value}": metrics.smooth,
                },
                step=epoch,
            )

            print(
                f"[{self.model_type.value.upper()}] "
                f"Epoch {epoch+1}/{self.config.epochs} "
                f"MSE={metrics.mse:.6f}  MAE={metrics.mae:.6f}  Smooth={metrics.smooth:.6f}"
            )

        return model, history