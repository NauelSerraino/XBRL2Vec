import torch
import torch.nn as nn
import torch.nn.functional as F


# AutoEncoder inspired by the original D-Linear model: 
# https://github.com/vivva/DLinear/blob/main/models/DLinear.py

# --- Legend ----
# B: Batch size
# T: Time steps (quarters)
# F: Financial features (variables per quarter)
# M: Macro features (variables per quarter)


# --- DLinear decomposition blocks ---
# This 2 blocks capture the seasonal and trend components of the D-Linear model
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) 
    
    def forward(self, x):
        # x: [B, T, F] 
        front = x[:, 0:1, :].repeat(1, (self.kernel_size-1)//2, 1) # left padding
        end = x[:, -1:, :].repeat(1, (self.kernel_size-1)//2, 1) # right padding
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0,2,1))
        x = x.permute(0,2,1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=5):
        """Computes the seasonal and trend components.

        Args:
            kernel_size (int, optional): number of consecutive time steps (quarters) 
            used to compute the moving average. Defaults to 5.
        """
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)
    
    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# --- Encoders ---
class DLinearEncoder(nn.Module):
    def __init__(self, T, F):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=5)
        self.trend_proj = nn.Linear(T, 1)
        self.seas_proj = nn.Linear(T, 1)
    
    def forward(self, x):
        seas, trend = self.decomp(x)
        # project each feature over time to a single value
        seas = self.seas_proj(seas.permute(0,2,1)).squeeze(-1)
        trend = self.trend_proj(trend.permute(0,2,1)).squeeze(-1)
        return torch.cat([seas, trend], dim=-1)  # [B, 2F]

class MacroEncoder(nn.Module):
    def __init__(self, T, M):
        super().__init__()
        self.proj = nn.Linear(T, 1)
    
    def forward(self, x):
        return self.proj(x.permute(0,2,1)).squeeze(-1)  # [B, M]


# --- Decoder: reconstruct full time series ---
class FinancialDecoder(nn.Module):
    def __init__(self, latent_dim, T, F):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, T*F)
        )
        self.T = T
        self.F = F
    
    def forward(self, z):
        out = self.fc(z)
        return out.view(-1, self.T, self.F) 


# --- Full Autoencoder ---
class CompanyEmbeddingAE(nn.Module):
    def __init__(self, T, F, M, latent_dim):
        super().__init__()
        self.fin_encoder = DLinearEncoder(T, F)
        self.macro_encoder = MacroEncoder(T, M)
        self.to_latent = nn.Linear(2*F + M, latent_dim)
        self.decoder = FinancialDecoder(latent_dim, T, F)
    
    def forward(self, x_fin, x_macro):
        h_fin = self.fin_encoder(x_fin)      # [B, 2F] -> learned seasonal + trend summaries per financial feature (2F bcs it has trend and seasonal component)
        h_macro = self.macro_encoder(x_macro)  # [B, M] -> tensor representing the macro features
        z = self.to_latent(torch.cat([h_fin, h_macro], dim=-1))  # [B, latent_dim] -> embedding representing the financial and macro
        x_fin_hat = self.decoder(z)  # [B, T, F] # initial financial input retrieved
        return z, x_fin_hat
