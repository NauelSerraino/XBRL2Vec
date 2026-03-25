"""
Input transformations applied to raw tensors before training.
"""
from __future__ import annotations

import numpy as np
import torch
from services.data import AlignedDataset


def symmetric_log(tensor: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|) — preserves sign, compresses outliers."""
    return torch.sign(tensor) * torch.log1p(torch.abs(tensor))


# ---------------------------------------------------------------------------
# Macro weighting strategies
# ---------------------------------------------------------------------------

def _fin_signal(X_fin: torch.Tensor) -> np.ndarray:
    """Aggregate financial tensor to a 1-D signal per company: [N, T]."""
    return X_fin.mean(dim=2).numpy()


def _macro_np(X_macro: torch.Tensor, i: int) -> np.ndarray:
    """Return company i's macro matrix as numpy [T, M]."""
    return (X_macro[i] if X_macro.ndim == 3 else X_macro).numpy()


def _weights_corr(X_fin: torch.Tensor, X_macro: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation between aggregate financial signal and each macro var.
    Returns [N, M]. Range: [-1, 1]. Negative weights flip the macro series.
    """
    N, T, F = X_fin.shape
    M = X_macro.shape[-1]
    fin = _fin_signal(X_fin)  # [N, T]
    weights = []

    for i in range(N):
        f = fin[i]
        m = _macro_np(X_macro, i)  # [T, M]
        f_c = f - f.mean()
        f_std = f_c.std() + 1e-8
        corrs = []
        for j in range(M):
            m_j = m[:, j]
            m_c = m_j - m_j.mean()
            corrs.append((f_c * m_c).mean() / (f_std * (m_c.std() + 1e-8)))
        weights.append(torch.tensor(corrs, dtype=torch.float32))

    return torch.stack(weights)  # [N, M]


def _weights_ridge(X_fin: torch.Tensor, X_macro: torch.Tensor) -> torch.Tensor:
    """
    Ridge regression betas: macro → aggregate financial signal.
    Returns [N, M]. Range: unbounded. Accounts for multicollinearity.
    """
    from sklearn.linear_model import Ridge

    fin = _fin_signal(X_fin)  # [N, T]
    weights = []

    for i in range(X_fin.shape[0]):
        m = _macro_np(X_macro, i)  # [T, M]
        coef = Ridge(alpha=1.0).fit(m, fin[i]).coef_
        weights.append(torch.tensor(coef, dtype=torch.float32))

    return torch.stack(weights)  # [N, M]


def _weights_xgboost(X_fin: torch.Tensor, X_macro: torch.Tensor) -> torch.Tensor:
    """
    XGBoost feature importances: macro → aggregate financial signal.
    Returns [N, M]. Range: [0, 1] (importances). Non-negative, so macro direction
    is not encoded — importances are sign-corrected by the correlation sign.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError(
            "xgboost is required for macro_weight_mode='xgboost': pip install xgboost"
        )

    fin = _fin_signal(X_fin)  # [N, T]
    weights = []

    for i in range(X_fin.shape[0]):
        m = _macro_np(X_macro, i)  # [T, M]
        f = fin[i]
        imp = XGBRegressor(
            n_estimators=50, max_depth=3, verbosity=0, random_state=42
        ).fit(m, f).feature_importances_

        # Sign-correct: multiply importance by sign of Pearson correlation
        f_c = f - f.mean()
        signs = np.sign(
            np.array([(f_c * (m[:, j] - m[:, j].mean())).mean() for j in range(m.shape[1])])
        )
        signs[signs == 0] = 1
        weights.append(torch.tensor(imp * signs, dtype=torch.float32))

    return torch.stack(weights)  # [N, M]


_WEIGHT_FNS = {
    "corr":    _weights_corr,
    "ridge":   _weights_ridge,
    "xgboost": _weights_xgboost,
}


def transform_dataset(dataset: AlignedDataset, macro_weight_mode: str = "corr") -> AlignedDataset:
    """
    Apply symmetric log + macro weighting.
    Returns a new AlignedDataset with transformed tensors.

    macro_weight_mode: "corr" | "ridge" | "xgboost"
    """
    if macro_weight_mode not in _WEIGHT_FNS:
        raise ValueError(f"Unknown macro_weight_mode '{macro_weight_mode}'. Choose from {list(_WEIGHT_FNS)}")

    print(f"[INFO] Macro weighting: {macro_weight_mode}")
    X_fin   = symmetric_log(dataset.X_fin)
    X_macro = symmetric_log(dataset.X_macro)

    weights = _WEIGHT_FNS[macro_weight_mode](X_fin, X_macro)
    X_macro = X_macro * weights.unsqueeze(1)  # [N, T, M]

    return AlignedDataset(
        X_fin      = X_fin,
        X_macro    = X_macro,
        Y_fin      = dataset.Y_fin,
        meta_df    = dataset.meta_df,
        fin_cols   = dataset.fin_cols,
        macro_cols = dataset.macro_cols,
    )
