
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt import PoincareBall
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt



class FlattenedEuclideanFAE(nn.Module):
    """Euclidean (Baseline) FAE - Flattened Input/Output."""
    def __init__(self, seq_len, fin_dim, macro_dim, latent_dim, forecast_len):
        super().__init__()
        latent_fin_dim = latent_macro_dim = latent_dim // 2
        total_latent_dim = latent_fin_dim + latent_macro_dim
        self.flat_fin_dim_in = seq_len * fin_dim
        self.flat_fin_dim_out = forecast_len * fin_dim
        self.flat_macro_dim = seq_len * macro_dim
        self.fin_encoder = nn.Sequential(nn.Linear(self.flat_fin_dim_in, latent_fin_dim * 2), nn.ReLU(), nn.Linear(latent_fin_dim * 2, latent_fin_dim))
        self.macro_encoder = nn.Sequential(nn.Linear(self.flat_macro_dim, latent_macro_dim * 2), nn.ReLU(), nn.Linear(latent_macro_dim * 2, latent_macro_dim))
        self.decoder = nn.Sequential(nn.Linear(total_latent_dim, total_latent_dim * 2), nn.ReLU(), nn.Linear(total_latent_dim * 2, self.flat_fin_dim_out))

    def forward(self, x_fin: torch.Tensor, x_macro: torch.Tensor):
        B = x_fin.size(0)
        x_fin_flat = x_fin.view(B, -1); x_macro_flat = x_macro.view(B, -1)
        z_fin_e = self.fin_encoder(x_fin_flat)
        z_macro_e = self.macro_encoder(x_macro_flat)
        z = torch.cat([z_fin_e, z_macro_e], dim=-1)
        forecast_flat = self.decoder(z)
        return forecast_flat, z_fin_e, z_macro_e

class SpecializedHyperbolicFAE(nn.Module):
    """Hyperbolic FAE - Specialized Gating and Poincare Ball projection."""
    def __init__(self, seq_len, fin_dim, macro_dim, latent_fin_dim, latent_macro_dim, forecast_len, c=1.0):
        super().__init__()
        self.c = c
        self.latent_fin_dim = latent_fin_dim
        self.flat_fin_dim_in = seq_len * fin_dim
        self.flat_fin_dim_out = forecast_len * fin_dim
        self.flat_macro_dim = seq_len * macro_dim
        self.manifold = PoincareBall(c=self.c)
        self.fin_encoder = nn.Sequential(nn.Linear(self.flat_fin_dim_in, latent_fin_dim * 2), nn.ReLU(), nn.Linear(latent_fin_dim * 2, latent_fin_dim))
        self.macro_encoder = nn.Sequential(nn.Linear(self.flat_macro_dim, latent_macro_dim * 2), nn.ReLU(), nn.Linear(latent_macro_dim * 2, latent_macro_dim))
        self.gating = nn.Sequential(nn.Linear(latent_macro_dim, latent_fin_dim), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(latent_fin_dim, latent_fin_dim * 2), nn.ReLU(), nn.Linear(latent_fin_dim * 2, self.flat_fin_dim_out))
    
    def forward(self, x_fin: torch.Tensor, x_macro: torch.Tensor):
        B = x_fin.size(0)
        x_fin_flat = x_fin.view(B, -1); x_macro_flat = x_macro.view(B, -1)
        z_fin_pre = self.fin_encoder(x_fin_flat)
        origin = z_fin_pre.new_zeros(z_fin_pre.shape)
        z_fin_h = self.manifold.retr(origin, z_fin_pre)
        z_fin_e = self.manifold.logmap(z_fin_h, origin) # Euc tangent vector for fusion/loss
        z_macro_e = self.macro_encoder(x_macro_flat)
        gate = self.gating(z_macro_e)
        z_combined = gate * z_fin_e
        forecast_flat = self.decoder(z_combined)
        return forecast_flat, z_fin_e, z_macro_e


class RecurrentGRUFAE(nn.Module):
    """GRU-based FAE - Sequence-to-Vector-to-Sequence."""
    def __init__(self, seq_len, fin_dim, macro_dim, latent_dim, forecast_len, rnn_type='GRU'):
        super().__init__()
        self.seq_len = seq_len; self.fin_dim = fin_dim; self.macro_dim = macro_dim; self.forecast_len = forecast_len
        latent_fin_dim = latent_macro_dim = latent_dim // 2
        rnn_class = {'GRU': nn.GRU, 'LSTM': nn.LSTM, 'RNN': nn.RNN}[rnn_type]
        self.fin_encoder = rnn_class(input_size=fin_dim, hidden_size=latent_fin_dim, batch_first=True)
        self.macro_encoder = rnn_class(input_size=macro_dim, hidden_size=latent_macro_dim, batch_first=True)
        self.decoder_input_dim = fin_dim
        self.decoder_hidden_dim = latent_fin_dim + latent_macro_dim
        self.decoder_rnn = rnn_class(input_size=self.decoder_input_dim, hidden_size=self.decoder_hidden_dim, batch_first=True)
        self.output_linear = nn.Linear(self.decoder_hidden_dim, fin_dim)

    def forward(self, x_fin: torch.Tensor, x_macro: torch.Tensor):
        B = x_fin.size(0)
        x_fin_seq = x_fin.view(B, self.seq_len, self.fin_dim)
        x_macro_seq = x_macro.view(B, self.seq_len, self.macro_dim)
        
        # 1. Encoding
        _, h_fin = self.fin_encoder(x_fin_seq); z_fin_e = (h_fin[0] if isinstance(self.fin_encoder, nn.LSTM) else h_fin).squeeze(0)
        _, h_macro = self.macro_encoder(x_macro_seq); z_macro_e = (h_macro[0] if isinstance(self.macro_encoder, nn.LSTM) else h_macro).squeeze(0)
        z = torch.cat([z_fin_e, z_macro_e], dim=-1)
        
        # 2. Decoding (Autoregressive Generation)
        decoder_h = z.unsqueeze(0).contiguous() # (1, B, H_dec)

        # --- LSTM/GRU/RNN Hidden State Initialization FIX ---
        if isinstance(self.decoder_rnn, nn.LSTM):
             # LSTM requires (h, c) tuple. Initialize cell state (c) as zeros.
            decoder_hidden_state = (decoder_h, torch.zeros_like(decoder_h))
        else:
            # GRU/RNN only requires h tensor.
            decoder_hidden_state = decoder_h
        
        decoder_input = torch.zeros(B, 1, self.fin_dim, device=x_fin.device)
        forecast_steps = []; current_hidden = decoder_hidden_state
        
        for t in range(self.forecast_len):
            decoder_output, current_hidden = self.decoder_rnn(decoder_input, current_hidden)
            predicted_step = self.output_linear(decoder_output)
            forecast_steps.append(predicted_step)
            decoder_input = predicted_step
        
        forecast_seq = torch.cat(forecast_steps, dim=1)
        forecast_flat = forecast_seq.view(B, -1)
        return forecast_flat, z_fin_e, z_macro_e

class LSTMFAE(RecurrentGRUFAE):
    def __init__(self, seq_len, fin_dim, macro_dim, latent_dim, forecast_len):
        super().__init__(seq_len, fin_dim, macro_dim, latent_dim, forecast_len, rnn_type='LSTM')

class RNNFAE(RecurrentGRUFAE):
    def __init__(self, seq_len, fin_dim, macro_dim, latent_dim, forecast_len):
        super().__init__(seq_len, fin_dim, macro_dim, latent_dim, forecast_len, rnn_type='RNN')

class TransformerFAE(nn.Module):
    """Transformer FAE - Encoder compression + Linear projection."""
    def __init__(self, seq_len, fin_dim, macro_dim, latent_dim, forecast_len, n_head=6, n_layers=2):
        super().__init__()
        self.seq_len = seq_len; self.fin_dim = fin_dim
        self.flat_fin_dim_out = forecast_len * fin_dim
        self.fin_dim = fin_dim      # <-- MISSING LINE 1: ADD THIS
        self.macro_dim = macro_dim  # <-- MISSING LINE 2: ADD THIS
        D_MODEL = latent_dim # Unified embedding dimension
        self.fin_projection = nn.Linear(fin_dim, D_MODEL)
        self.macro_projection = nn.Linear(macro_dim, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=n_head, dim_feedforward=4*D_MODEL, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Sequential(nn.Linear(D_MODEL, D_MODEL * 2), nn.ReLU(), nn.Linear(D_MODEL * 2, self.flat_fin_dim_out))
        # Dummy latent variables for compatibility
        self.dummy_fin_e = nn.Linear(D_MODEL, latent_dim // 2)
        self.dummy_macro_e = nn.Linear(D_MODEL, latent_dim // 2)

    def forward(self, x_fin: torch.Tensor, x_macro: torch.Tensor):
        B = x_fin.size(0)
        x_fin_seq = x_fin.view(B, self.seq_len, self.fin_dim)
        x_macro_seq = x_macro.view(B, self.seq_len, self.macro_dim)
        x_fin_proj = self.fin_projection(x_fin_seq)
        x_macro_proj = self.macro_projection(x_macro_seq)
        x_combined_proj = x_fin_proj + x_macro_proj
        h_seq = self.transformer_encoder(x_combined_proj)
        z = h_seq.mean(dim=1) # Mean pooling latent vector
        forecast_flat = self.decoder(z)
        z_fin_e = self.dummy_fin_e(z)
        z_macro_e = self.dummy_macro_e(z)
        return forecast_flat, z_fin_e, z_macro_e

class DLinearFAE(nn.Module):
    """D-Linear FAE - Decomposition and Linear fusion."""
    def __init__(self, seq_len, fin_dim, macro_dim, latent_dim, forecast_len, moving_average_window=5):
        super().__init__()
        self.seq_len = seq_len; self.fin_dim = fin_dim; self.forecast_len = forecast_len
        self.flat_fin_dim_out = forecast_len * fin_dim
        self.moving_average = nn.AvgPool1d(kernel_size=moving_average_window, stride=1, padding=(moving_average_window - 1) // 2)
        self.linear_trend = nn.Linear(seq_len, forecast_len)
        self.linear_seasonal = nn.Linear(seq_len, forecast_len)
        self.flat_macro_dim = seq_len * macro_dim
        self.macro_encoder = nn.Linear(self.flat_macro_dim, latent_dim) 
        self.fusion_dim = forecast_len * fin_dim + latent_dim
        self.fusion_decoder = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2), nn.ReLU(), nn.Linear(self.fusion_dim // 2, self.flat_fin_dim_out)
        )
        self.dummy_fin_e = nn.Linear(latent_dim, latent_dim // 2)
        self.dummy_macro_e = nn.Linear(latent_dim, latent_dim // 2)

    def forward(self, x_fin: torch.Tensor, x_macro: torch.Tensor):
        B = x_fin.size(0)
        # D-Linear Decomposition
        x_fin_seq = x_fin.view(B, self.seq_len, self.fin_dim).permute(0, 2, 1)
        trend_init = self.moving_average(x_fin_seq) 
        seasonal_init = x_fin_seq - trend_init
        trend_output = self.linear_trend(trend_init)
        seasonal_output = self.linear_seasonal(seasonal_init)
        dlinear_forecast_seq = trend_output + seasonal_output
        dlinear_forecast_flat = dlinear_forecast_seq.permute(0, 2, 1).contiguous().view(B, -1)
        
        # Macro Encoding and Fusion
        x_macro_flat = x_macro.view(B, -1)
        z_macro = self.macro_encoder(x_macro_flat)
        z_fusion_input = torch.cat([dlinear_forecast_flat, z_macro], dim=-1)
        forecast_flat = self.fusion_decoder(z_fusion_input)
        
        z_fin_e = self.dummy_fin_e(z_macro)
        z_macro_e = self.dummy_macro_e(z_macro)
        return forecast_flat, z_fin_e, z_macro_e

