"""
The encoder of XBRL2Vec: one firm's window of T quarters x F ratios -> one embedding z of `latent_dim` numbers.

    x [B, T, F]                           B windows, T = 12 quarters, F = 32 ratios (the fingerprint of each quarter)
      |  SeriesDecomp                     each ratio's series split into a trend (5-quarter moving average) and the
      |                                   remainder ("seasonal" in DLinear's vocabulary: what the trend leaves)
      |  two linear maps over time        trend -> 1 number, remainder -> 1 number, per ratio; the 2 x (T + 1) weights
      |                                   are shared by all ratios (they say which quarters of the window count)
    h [B, 2F]                             remainder summaries, then trend summaries
      |  MLP  2F -> hidden -> hidden -> latent_dim, ReLU
    z [B, latent_dim]

The first block is the encoder of DLinear (Zeng et al. 2023, "Are Transformers Effective for Time Series
Forecasting?", AAAI) without its forecasting head. The MLP lets the embedding combine ratios non-linearly. About 29,000
parameters with T 12, F 32, hidden 128, latent_dim 32. Similarity between firms is the cosine of their z.
"""
import numpy as np
import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average over time of each series, same length as the input (the ends are padded with the end values)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # x [B, T, F]
        pad = (self.kernel_size - 1) // 2
        x = torch.cat([x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, pad, 1)], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)             # AvgPool1d averages over the last axis


class SeriesDecomp(nn.Module):
    """x -> (x - trend, trend), the trend being the moving average of x."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        return x - trend, trend


class DLinearEncoder(nn.Module):
    """[B, T, F] -> [B, 2F]: for each ratio, a weighted sum over the T quarters of its remainder and of its trend."""

    def __init__(self, T: int, F: int):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=5)
        self.trend_proj = nn.Linear(T, 1)                                # the same T weights for every ratio
        self.seas_proj = nn.Linear(T, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seas, trend = self.decomp(x)
        seas = self.seas_proj(seas.permute(0, 2, 1)).squeeze(-1)         # [B, F, T] -> [B, F]
        trend = self.trend_proj(trend.permute(0, 2, 1)).squeeze(-1)
        return torch.cat([seas, trend], dim=-1)


class XBRL2VecEncoder(nn.Module):
    """Window [B, T, F] -> embedding [B, latent_dim] (see the module docstring)."""

    def __init__(self, T: int, F: int, latent_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.fin_encoder = DLinearEncoder(T, F)
        self.to_latent = nn.Sequential(
            nn.Linear(2 * F, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.fin_encoder(x))


@torch.no_grad()
def embed(model: XBRL2VecEncoder, X: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    """The embeddings of every window of X [N, T, F] as a numpy array [N, latent_dim] (evaluation mode, on the CPU)."""
    model.cpu().eval()
    return np.concatenate([model(X[i:i + batch_size]).numpy() for i in range(0, len(X), batch_size)])
