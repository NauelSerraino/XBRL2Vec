import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Legend ----
# B: Batch size
# T_in:  input time steps  (= SEQ_LEN - 1)
# T_out: forecast horizon  (= 1)
# F: Financial features
# M: Macro features


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


class MacroConditioner(nn.Module):
    def __init__(self, M, fin_dim):
        super().__init__()
        self.net = nn.Linear(M, 2 * fin_dim)

    def forward(self, h_macro):
        gamma, beta = self.net(h_macro).chunk(2, dim=-1)
        return gamma, beta


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


# --- Forecasting Autoencoder (with FiLM macro conditioning) ---
class ForecastingAE(nn.Module):
    """
    Encoder receives T_in past quarters + macro context.
    Decoder predicts the next T_out quarters.

    This gives the macro branch a genuine informational advantage:
    predicting future financials benefits from knowing the macro state.
    """

    def __init__(self, T_in: int, T_out: int, F: int, M: int, latent_dim: int):
        super().__init__()
        self.fin_encoder   = DLinearEncoder(T_in, F)
        self.macro_encoder = DLinearEncoder(T_in, M)
        self.conditioner   = MacroConditioner(2 * M, 2 * F)
        self.to_latent     = nn.Linear(2 * F, latent_dim)
        self.decoder       = FinancialDecoder(latent_dim, T_out, F)

    def forward(self, x_fin, x_macro=None):
        h_fin = self.fin_encoder(x_fin)  # [B, 2F]

        if x_macro is not None:
            h_macro           = self.macro_encoder(x_macro)  # [B, 2M]
            gamma, beta       = self.conditioner(h_macro)
            h_fin             = gamma * h_fin + beta

        z     = self.to_latent(h_fin)   # [B, latent_dim]
        x_hat = self.decoder(z)         # [B, T_out, F]
        return z, x_hat
