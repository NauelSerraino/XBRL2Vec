import torch
import torch.nn as nn


# --- Legend ----
# B: Batch size
# T_in:  input time steps
# T_out: output time steps (= 1 for forecasting, = T_in for reconstruction)
# F: Financial features


# --- DLinear decomposition blocks ---
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend


class DLinearEncoder(nn.Module):
    def __init__(self, T, F):
        super().__init__()
        self.decomp     = SeriesDecomp(kernel_size=5)
        self.trend_proj = nn.Linear(T, 1)
        self.seas_proj  = nn.Linear(T, 1)

    def forward(self, x):
        seas, trend = self.decomp(x)
        seas  = self.seas_proj(seas.permute(0, 2, 1)).squeeze(-1)
        trend = self.trend_proj(trend.permute(0, 2, 1)).squeeze(-1)
        return torch.cat([seas, trend], dim=-1)  # [B, 2F]


class FinancialDecoder(nn.Module):
    def __init__(self, latent_dim, T, F):
        super().__init__()
        self.T  = T
        self.F  = F
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, T * F),
        )

    def forward(self, z):
        return self.fc(z).view(-1, self.T, self.F)


# --- Ablation: financial-only forecaster (no macro branch, no FiLM) ---
class FinancialOnlyAE(nn.Module):
    """
    Pure financial forecaster for the ablation baseline.
    No macro encoder, no FiLM conditioner.

    Accepts x_macro=None so it can be used as a drop-in replacement
    in the existing training / evaluation pipeline.
    """

    def __init__(self, T_in: int, T_out: int, F: int, latent_dim: int):
        super().__init__()
        self.fin_encoder = DLinearEncoder(T_in, F)
        self.to_latent   = nn.Linear(2 * F, latent_dim)
        self.decoder     = FinancialDecoder(latent_dim, T_out, F)

    def forward(self, x_fin, x_macro=None):
        h_fin = self.fin_encoder(x_fin)
        z     = self.to_latent(h_fin)
        x_hat = self.decoder(z)
        return z, x_hat
