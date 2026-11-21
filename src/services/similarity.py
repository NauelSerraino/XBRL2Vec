"""
Training data, targets and loss of the encoder.

The encoder turns a firm's window (12 quarters x 32 ratios) into an embedding z. It is trained so that, among the firms
of the same quarter, the cosine similarity of two embeddings reproduces the cosine similarity of two targets:

  Pillar 1 (structure)  the fingerprint f of the window's last quarter: the 32 ratios (attach_fingerprints)
  Pillar 2 (dynamics)   the window deltas d: the year-on-year moves of the fingerprint inside the window, the recent
                        ones weighted more (attach_window_deltas)

  loss = (1 - λ) · mean over pairs i<j of (cos(z_i, z_j) - cos(f_i, f_j))²
       +      λ  · mean over pairs i<j of (cos(z_i, z_j) - cos(d_i, d_j))²            (two_target_loss)

A batch holds firms of one quarter only (quarter_batches), so a firm is always compared with its contemporaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass
class SimilarityData:
    """N windows, one per firm and quarter."""
    X: torch.Tensor                    # [N, T, F] the input windows (the fingerprint of each of the T quarters)
    target: np.ndarray                 # [N, K] Pillar 1 target: the fingerprint of the window's last quarter (K = F = 32)
    meta: pd.DataFrame                 # [N] ticker, quarter (the window's last quarter)
    cols: list | None = None           # the F column names of X
    target2: np.ndarray | None = None  # [N, 8 K] Pillar 2 target: the window deltas (attach_window_deltas)

    def subset(self, mask: np.ndarray) -> "SimilarityData":
        """The windows where the boolean mask [N] is True, every field cut the same way."""
        idx = np.flatnonzero(mask)
        return SimilarityData(self.X[idx], self.target[idx], self.meta.iloc[idx].reset_index(drop=True), self.cols,
                              None if self.target2 is None else self.target2[idx])


def quarter_shift(q: str, h: int) -> str:
    """The quarter h quarters after q ("2024Q4", 1 -> "2025Q1"; h may be negative)."""
    n = int(q[:4]) * 4 + int(q[-1]) - 1 + h
    return f"{n // 4}Q{n % 4 + 1}"


def attach_fingerprints(X: torch.Tensor, meta: pd.DataFrame, cols: list, fp: pd.DataFrame) -> SimilarityData:
    """The windows whose last quarter has a fingerprint, with that fingerprint as their Pillar 1 target.
    fp: ticker, quarter and one column per ratio."""
    fp = fp.set_index(["ticker", "quarter"])
    keys = pd.MultiIndex.from_frame(meta[["ticker", "quarter"]])
    has = keys.isin(fp.index)
    idx = np.flatnonzero(has)
    return SimilarityData(X[idx], fp.loc[keys[has]].to_numpy(np.float32), meta.iloc[idx].reset_index(drop=True), list(cols))


def attach_window_deltas(data: SimilarityData, fp: pd.DataFrame, t_in: int, decay: float = 1.0) -> SimilarityData:
    """
    The Pillar 2 target of every window: the year-on-year moves of the fingerprint inside it.

    A window ends at quarter t and covers t - (t_in - 1) .. t. For lag = t_in - 5, ..., 1, 0 (oldest first) the move
    f(t - lag) - f(t - lag - 4) is a block of K numbers, scaled by decay ** lag; the t_in - 4 blocks are put side by
    side: target2 [N, (t_in - 4) K], with t_in 12 and K 32: 8 x 32 = 256 numbers. decay < 1 makes the recent moves count
    more in the cosine (0.8: the oldest of the 8 moves weighs 0.8^7 = 0.21); decay 1: all moves weigh the same. A row is
    NaN where a quarter of the window has no fingerprint.
    """
    if t_in < 5:
        raise ValueError("window deltas need t_in >= 5")
    fp = fp.set_index(["ticker", "quarter"])
    uq = data.meta.quarter.unique()

    def at(lag: int) -> np.ndarray:                                    # the fingerprint `lag` quarters before the last
        q = data.meta.quarter.map({u: quarter_shift(u, -lag) for u in uq})
        return fp.reindex(pd.MultiIndex.from_arrays([data.meta.ticker, q])).to_numpy(np.float32)

    f = {lag: at(lag) for lag in range(t_in)}
    data.target2 = np.concatenate([(f[lag] - f[lag + 4]) * np.float32(decay ** lag) for lag in range(t_in - 5, -1, -1)],
                                  axis=1)
    return data


def _unit(a: torch.Tensor) -> torch.Tensor:
    """Each row divided by its length: dot products of the result are cosines."""
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def pair_loss(z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Mean squared gap between cos(z_i, z_j) and cos(f_i, f_j) over the pairs i < j of the batch."""
    cz, cf = _unit(z) @ _unit(z).T, _unit(f) @ _unit(f).T
    iu = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    return F.mse_loss(cz[iu[0], iu[1]], cf[iu[0], iu[1]])


def two_target_loss(z: torch.Tensor, f: torch.Tensor, d: torch.Tensor,
                    lambda2: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (1 - lambda2) · L1 + lambda2 · L2: L1 = pair_loss against the fingerprints f (Pillar 1), L2 = the same against the
    window deltas d (Pillar 2), over the firms whose deltas are known (rows of d without NaN). Returns (total, L1, L2).
    """
    l1 = pair_loss(z, f)
    known = torch.isfinite(d).all(dim=1)
    if known.sum() < 2:
        return (1.0 - lambda2) * l1, l1, torch.zeros((), device=z.device)
    zk, dk = z[known], d[known]
    cz, cd = _unit(zk) @ _unit(zk).T, _unit(dk) @ _unit(dk).T
    iu = torch.triu_indices(len(zk), len(zk), offset=1, device=z.device)
    l2 = ((cz[iu[0], iu[1]] - cd[iu[0], iu[1]]) ** 2).mean()
    return (1.0 - lambda2) * l1 + lambda2 * l2, l1, l2


def quarter_batches(meta: pd.DataFrame, batch_size: int, rng: np.random.Generator, min_size: int = 16) -> list[np.ndarray]:
    """Row-index batches of at most batch_size windows, each from one quarter, in random order (a quarter with more
    windows gives several batches; a leftover of fewer than min_size windows is dropped for that epoch)."""
    batches = []
    for _, idx in meta.groupby("quarter").indices.items():
        idx = rng.permutation(idx)
        batches += [idx[i:i + batch_size] for i in range(0, len(idx), batch_size) if len(idx[i:i + batch_size]) >= min_size]
    return [batches[i] for i in rng.permutation(len(batches))]


def train(model, data: SimilarityData, epochs: int, lr: float, batch_size: int, device, seed: int = 0,
          lambda2: float = 0.5, input_mask: float = 0.0, log=print) -> list[float]:
    """
    Train the encoder with Adam on two_target_loss. Returns the mean loss of each epoch.

    input_mask: in every batch, this share of the input cells (ratio x quarter) is set to 0 at random, the targets
    unchanged. 0 is the value of a ratio that is not reported (the mean of the quarter after the z-score), so the encoder
    learns to place firms with missing data.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    target, target2 = torch.tensor(data.target), torch.tensor(data.target2)
    history = []
    for epoch in range(epochs):
        losses, parts = [], []
        for b in quarter_batches(data.meta, batch_size, rng):
            x = data.X[b].to(device)
            if input_mask > 0:
                x = x.masked_fill(torch.rand(x.shape, device=device) < input_mask, 0.0)
            total, l1, l2 = two_target_loss(model(x), target[b].to(device), target2[b].to(device), lambda2)
            opt.zero_grad()
            total.backward()
            opt.step()
            losses.append(total.item())
            parts.append((l1.item(), l2.item()))
        history.append(float(np.mean(losses)))
        log(f"  epoch {epoch + 1}/{epochs}: loss {history[-1]:.4f} (Pillar 1 {np.mean([p[0] for p in parts]):.4f}, "
            f"Pillar 2 {np.mean([p[1] for p in parts]):.4f})")
    return history
