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
        # Collapse time dimension: [B, F, T] -> [B, F, 1] -> [B, F]
        seas = self.seas_proj(seas.permute(0,2,1)).squeeze(-1)
        trend = self.trend_proj(trend.permute(0,2,1)).squeeze(-1)
        return torch.cat([seas, trend], dim=-1)  # [B, 2F]

class MacroEncoder(nn.Module):
    def __init__(self, T, M):
        super().__init__()
        self.proj = nn.Linear(T, 1)
    
    def forward(self, x):
        # Collapse time dimension: [B, M, T] -> [B, M]
        return self.proj(x.permute(0,2,1)).squeeze(-1)  # [B, M]


# --- Decoder: reconstruct full time series ---
class FinancialDecoder(nn.Module):
    def __init__(self, latent_dim, T, F):
        super().__init__()
        self.T = T
        self.F = F
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, T * F)
        )
    
    def forward(self, z):
        out = self.fc(z)
        return out.view(-1, self.T, self.F)
    

class MacroConditioner(nn.Module):
    def __init__(self, M, fin_dim):
        super().__init__()
        self.net = nn.Linear(M, 2 * fin_dim)

    def forward(self, h_macro):
        gamma, beta = self.net(h_macro).chunk(2, dim=-1)
        return gamma, beta


# --- Full Autoencoder ---
class CompanyEmbeddingAE(nn.Module):
    def __init__(self, T, F, M, latent_dim):
        super().__init__()
        self.fin_encoder = DLinearEncoder(T, F)
        self.macro_encoder = DLinearEncoder(T, M) 
        self.conditioner = MacroConditioner(2 * M, 2 * F)
        
        self.to_latent = nn.Linear(2 * F, latent_dim)
        self.decoder = FinancialDecoder(latent_dim, T, F)

    def forward(self, x_fin, x_macro):
        # 1. Encode both to fixed-size vectors (summaries)
        h_fin = self.fin_encoder(x_fin)      # [B, 2F]
        h_macro = self.macro_encoder(x_macro) # [B, M]

        # 2. Condition financials based on macro state
        gamma, beta = self.conditioner(h_macro)
        h_fin_cond = gamma * h_fin + beta 

        # 3. Project to DNA Embedding (Z)
        z = self.to_latent(h_fin_cond)
        
        # 4. Reconstruct original input length T
        x_hat = self.decoder(z)
        return z, x_hat
