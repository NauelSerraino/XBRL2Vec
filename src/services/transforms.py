"""
Input transformations applied to raw tensors before training.
"""
from __future__ import annotations

import torch
from services.data import AlignedDataset


def symmetric_log(tensor: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|) — preserves sign, compresses outliers."""
    return torch.sign(tensor) * torch.log1p(torch.abs(tensor))


def compute_company_macro_weights(X_fin: torch.Tensor, X_macro: torch.Tensor) -> torch.Tensor:
    """
    Computes per-company correlation between aggregate financial signal
    and each macro variable.

    Args:
        X_fin:   [N, T, F]
        X_macro: [N, T, M] or [T, M]

    Returns:
        weights: [N, M]
    """
    N, T, F = X_fin.shape
    M = X_macro.shape[-1]

    fin_signal = X_fin.mean(dim=2)  # [N, T]
    weights = []

    for i in range(N):
        f = fin_signal[i]
        m = X_macro[i] if X_macro.ndim == 3 else X_macro

        f_center = f - f.mean()
        f_std = f_center.std() + 1e-8

        w_i = []
        for j in range(M):
            m_j = m[:, j]
            m_center = m_j - m_j.mean()
            m_std = m_center.std() + 1e-8
            corr = (f_center * m_center).mean() / (f_std * m_std)
            w_i.append(corr)

        weights.append(torch.tensor(w_i))

    return torch.stack(weights)  # [N, M]


def apply_macro_weights(X_macro: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Scale each macro variable per company by its correlation weight.

    Args:
        X_macro: [N, T, M]
        weights: [N, M]

    Returns:
        X_macro_weighted: [N, T, M]
    """
    return X_macro * weights.unsqueeze(1)


def transform_dataset(dataset: AlignedDataset) -> AlignedDataset:
    """
    Apply symmetric log + macro weighting in place.
    Returns a new AlignedDataset with transformed tensors.
    """
    X_fin   = symmetric_log(dataset.X_fin)
    X_macro = symmetric_log(dataset.X_macro)

    weights = compute_company_macro_weights(X_fin, X_macro)
    X_macro = apply_macro_weights(X_macro, weights)

    return AlignedDataset(
        X_fin      = X_fin,
        X_macro    = X_macro,
        Y_fin      = dataset.Y_fin,
        meta_df    = dataset.meta_df,
        fin_cols   = dataset.fin_cols,
        macro_cols = dataset.macro_cols,
    )