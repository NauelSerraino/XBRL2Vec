"""
Model evaluation: OOS metrics, permutation importance, macro exposure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import mlflow

from services.config import DEVICE
from services.data import DistanceMetric


# ---------------------------------------------------------------------------
# OOS evaluation
# ---------------------------------------------------------------------------

@dataclass
class OOSResult:
    label: str
    mse: float
    mae: float
    smooth: float

    @property
    def macro_gain_vs(self) -> float:
        """Placeholder — computed externally between two OOSResult instances."""
        raise NotImplementedError


def evaluate_oos(
    model: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    label: str,
    Y_fin: torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> OOSResult:
    print(f"[INFO] OOS evaluation ({label})")
    y_target = (Y_fin if Y_fin is not None else X_fin).to(device)
    model.eval().to(device)

    with torch.no_grad():
        _, x_hat = model(X_fin.to(device), X_macro.to(device))
        mse    = F.mse_loss(x_hat, y_target).item()
        mae    = F.l1_loss(x_hat, y_target).item()
        smooth = F.smooth_l1_loss(x_hat, y_target).item()

    print(f"  MSE={mse:.6f}  MAE={mae:.6f}")
    return OOSResult(label=label, mse=mse, mae=mae, smooth=smooth)


# ---------------------------------------------------------------------------
# Permutation importance matrix
# ---------------------------------------------------------------------------

def compute_importance_matrix(
    model: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    fin_cols: list[str],
    macro_cols: list[str],
    label: str,
    Y_fin: torch.Tensor | None = None,
    seed: int = 42,
    device: torch.device = DEVICE,
) -> pd.DataFrame:
    """
    Permutation-based feature importance.
    Returns a DataFrame [out_features × all_inputs] of Δ MSE per permuted column.
    """
    print(f"[INFO] Importance matrix ({label})")
    model.eval().to(device)
    x_f = X_fin.to(device)
    x_m = X_macro.to(device)
    y_t = (Y_fin if Y_fin is not None else X_fin).to(device)
    N = x_f.shape[0]
    out_cols = list(y_t.shape[-1:])  # F (output features)
    all_inputs = fin_cols + macro_cols

    with torch.no_grad():
        _, x_hat = model(x_f, x_m)
        base_loss = ((x_hat - y_t) ** 2).mean(dim=1)

    imp = np.zeros((y_t.shape[-1], len(all_inputs)))

    for j, feat in enumerate(all_inputs):
        gen = torch.Generator().manual_seed(seed + j)
        perm = torch.randperm(N, generator=gen)

        xf_p = x_f.clone()
        xm_p = x_m.clone()

        if feat in fin_cols:
            idx = fin_cols.index(feat)
            xf_p[:, :, idx] = xf_p[perm, :, idx]
        else:
            idx = macro_cols.index(feat)
            xm_p[:, :, idx] = xm_p[perm, :, idx]

        with torch.no_grad():
            _, x_hat_p = model(xf_p, xm_p)
            perm_loss = ((x_hat_p - y_t) ** 2).mean(dim=1)

        imp[:, j] = (perm_loss - base_loss).mean(dim=0).cpu().numpy()

    return pd.DataFrame(imp, index=fin_cols, columns=all_inputs)


# ---------------------------------------------------------------------------
# Macro exposure (counterfactual latent shift)
# ---------------------------------------------------------------------------

@dataclass
class MacroExposureResult:
    contextual_l2: np.ndarray
    contextual_cosine: np.ndarray
    blind_l2: np.ndarray
    blind_cosine: np.ndarray


def compute_macro_exposure(
    model_ctx: torch.nn.Module,
    model_blind: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    device: torch.device = DEVICE,
) -> MacroExposureResult:
    """
    Counterfactual macro exposure: compare embeddings with real vs permuted macro.
    """
    model_ctx.eval().to(device)
    model_blind.eval().to(device)

    xf = X_fin.to(device)
    xm = X_macro.to(device)

    with torch.no_grad():
        perm_idx = torch.randperm(xm.size(0), device=device)
        xm_perm = xm[perm_idx]

        z_ctx_real, _  = model_ctx(xf, xm)
        z_ctx_cf,   _  = model_ctx(xf, xm_perm)
        z_blind_real, _ = model_blind(xf, xm)
        z_blind_cf,   _ = model_blind(xf, xm_perm)

    def _l2(a, b):
        return torch.norm(a - b, dim=1).cpu().numpy()

    def _cos(a, b):
        return (1 - F.cosine_similarity(a, b, dim=1)).cpu().numpy()

    return MacroExposureResult(
        contextual_l2     = _l2(z_ctx_real, z_ctx_cf),
        contextual_cosine = _cos(z_ctx_real, z_ctx_cf),
        blind_l2          = _l2(z_blind_real, z_blind_cf),
        blind_cosine      = _cos(z_blind_real, z_blind_cf),
    )


# ---------------------------------------------------------------------------
# Variance / linear probe analysis
# ---------------------------------------------------------------------------

def compute_variance_analysis(
    model: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    device: torch.device = DEVICE,
) -> pd.DataFrame:
    """
    Fit linear probes from latent z → macro / fin.
    Returns a [2 × 2] R² DataFrame (latent | recon) × (macro | financial).
    """
    model.eval().to(device)

    with torch.no_grad():
        z, x_hat = model(X_fin.to(device), X_macro.to(device))

    z_np    = z.cpu().numpy()
    xhat_np = x_hat.cpu().numpy().reshape(x_hat.shape[0], -1)
    fin_np  = X_fin.cpu().numpy().reshape(X_fin.shape[0], -1)
    mac_np  = X_macro.cpu().numpy().reshape(X_macro.shape[0], -1)

    def _r2(X, Y):
        pred = LinearRegression().fit(X, Y).predict(X)
        return r2_score(Y, pred)

    r2 = {
        "latent":         {"macro_r2": _r2(z_np, mac_np), "financial_r2": _r2(z_np, fin_np)},
        "reconstruction": {"macro_r2": _r2(xhat_np, mac_np), "financial_r2": _r2(xhat_np, fin_np)},
    }

    df = pd.DataFrame(r2).round(3)

    mlflow.log_metric("macro_linear_probe_r2_latent",     r2["latent"]["macro_r2"])
    mlflow.log_metric("financial_linear_probe_r2_latent", r2["latent"]["financial_r2"])
    mlflow.log_metric("macro_linear_probe_r2_recon",      r2["reconstruction"]["macro_r2"])
    mlflow.log_metric("financial_linear_probe_r2_recon",  r2["reconstruction"]["financial_r2"])

    print(f"  Macro R²   – Latent: {r2['latent']['macro_r2']:.4f}  |  Recon: {r2['reconstruction']['macro_r2']:.4f}")
    print(f"  Financial R² – Latent: {r2['latent']['financial_r2']:.4f}  |  Recon: {r2['reconstruction']['financial_r2']:.4f}")

    return df