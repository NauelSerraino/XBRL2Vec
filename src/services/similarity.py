"""
Pillar 1 similarity training and evaluation: the embedding of a firm-quarter should reproduce, as cosine similarity
between firms of the same quarter, the cosine similarity of their financial fingerprints.

Training: batches of firms from one quarter; loss = mean squared error between cos(z_i, z_j) and cos(f_i, f_j) over
the pairs of the batch (`pair_loss`, loss "pillar1"). Alternative loss "pillar1+2" (`two_target_loss`):
(1 - lambda2) x that error + lambda2 x the same error against cos(Δf_i, Δf_j), Δf = the change of the fingerprint over
the next h quarters (the Pillar 2 target: firms that react alike), over the pairs where both changes are known;
optionally weighted by the size of the two changes. Alternative Pillar 2 target (`attach_window_deltas`, 1 Nov 2026):
the year-on-year changes of the fingerprint inside the input window, stacked (firms that moved together; nothing after
t). Runs before 1 Nov 2026 used the additive form L1 + lambda2 x L2
(form "additive"), which with Adam trains like the convex form with weight lambda2 / (1 + lambda2).
Evaluation (`evaluate`), for any set of vectors (embeddings or baselines):
  fit           Spearman correlation between the vectors' cosines and the fingerprints' cosines, per quarter
  knn10         share of each firm's 10 nearest neighbours (by the vectors) that are among its 10 nearest by fingerprint
  reid4/reid20  share of firms whose nearest vector 4 / 20 quarters later is their own
  stability     mean cosine of a firm's vector in consecutive quarters
  peers_r2      cross-sectional R² of revenue growth on the mean revenue growth of the 10 nearest firms (Bhojraj, Lee &
                Oler 2003), mean over quarters
  p2 / p3       Spearman correlation between the vectors' cosines and the Pillar 2 / Pillar 3 target cosines
  fit2          as fit, against the Pillar 2 target similarities cos(Δf_i, Δf_j) (where known): the next-h-quarter
                changes, or the window deltas
  knn10_2       as knn10, against the 10 nearest firms by the Pillar 2 target
  fit2_future, knn10_2_future   with the window deltas as Pillar 2 target: fit2 and knn10_2 against the next-h-quarter
                changes (the fixed instrument across Pillar 2 targets)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from services.data import AlignedDataset


@dataclass
class SimilarityData:
    X_fin: torch.Tensor          # [N, T, F]
    X_macro: torch.Tensor        # [N, T, M]
    target: np.ndarray           # [N, K] fingerprint of the window's last quarter
    meta: pd.DataFrame           # [N] ticker, quarter
    fin_cols: list | None = None
    macro_cols: list | None = None
    target2: np.ndarray | None = None        # [N, K] fingerprint change over the next h quarters, NaN rows if unknown
                                             # (or the window deltas, attach_window_deltas)
    past_change: np.ndarray | None = None    # [N, K] change over the previous h quarters (a baseline), NaN if unknown
    scenario: np.ndarray | None = None       # [N, S] realised macro change over the next h quarters (attach_scenario)
    future_change: np.ndarray | None = None  # [N, K] the next-h-quarter change when target2 is another target (fit2_future)

    def subset(self, mask: np.ndarray) -> "SimilarityData":
        idx = np.flatnonzero(mask)
        pick = lambda a: None if a is None else a[idx]
        return SimilarityData(self.X_fin[idx], self.X_macro[idx], self.target[idx], self.meta.iloc[idx].reset_index(drop=True),
                              self.fin_cols, self.macro_cols, pick(self.target2), pick(self.past_change),
                              pick(self.scenario), pick(self.future_change))


def attach_fingerprints(ds: AlignedDataset, fp: pd.DataFrame) -> SimilarityData:
    """Keep the windows whose last quarter has a fingerprint; the fingerprint is their target."""
    meta = ds.meta_df.rename(columns={"end_quarter": "quarter"})
    fp = fp.set_index(["ticker", "quarter"])
    keys = pd.MultiIndex.from_frame(meta[["ticker", "quarter"]])
    has = keys.isin(fp.index)
    idx = np.flatnonzero(has)
    return SimilarityData(ds.X_fin[idx], ds.X_macro[idx], fp.loc[keys[has]].to_numpy(np.float32),
                          meta.iloc[idx].reset_index(drop=True), list(ds.fin_cols), list(ds.macro_cols))


def attach_changes(data: SimilarityData, ch: pd.DataFrame, h: int) -> SimilarityData:
    """Add the Pillar 2 target (change over the next h quarters, from fingerprint_change_h<h>.parquet) and the past
    change (over the previous h quarters, the same file shifted) to every window; NaN where unknown."""
    keys = pd.MultiIndex.from_frame(data.meta[["ticker", "quarter"]])
    nxt = ch.set_index(["ticker", "quarter"])
    shift = lambda q: (lambda n: f"{n // 4}Q{n % 4 + 1}")(int(q[:4]) * 4 + int(q[-1]) - 1 + h)
    past = ch.assign(quarter=ch.quarter.map(shift)).set_index(["ticker", "quarter"])
    get = lambda M: M.reindex(keys).to_numpy(np.float32)
    data.target2, data.past_change = get(nxt), get(past)
    return data


def attach_window_deltas(data: SimilarityData, fp: pd.DataFrame, t_in: int, decay: float = 1.0) -> SimilarityData:
    """Pillar 2 target "window_deltas": the year-on-year changes of the fingerprint inside each window, f(s) - f(s - 4)
    for its last t_in - 4 quarters s, oldest first, stacked into one vector [N, (t_in - 4) K] (firms that moved in the
    same direction at the same time); NaN where a quarter of the window has no fingerprint. The next-h-quarter change
    attached before (attach_changes) is kept as data.future_change. decay < 1: the block of the move ending k quarters
    before the window's last is scaled by decay ** k, so recent moves count more in the cosine (decay 1: all equal)."""
    if t_in < 5:
        raise ValueError("window_deltas needs t_in >= 5")
    F = fp.set_index(["ticker", "quarter"])
    uq = data.meta.quarter.unique()

    def at(lag: int) -> np.ndarray:                                    # fingerprint `lag` quarters before the last one
        q = data.meta.quarter.map({u: _quarter_shift(u, -lag) for u in uq})
        return F.reindex(pd.MultiIndex.from_arrays([data.meta.ticker, q])).to_numpy(np.float32)

    f = {lag: at(lag) for lag in range(t_in)}
    data.future_change = data.target2
    data.target2 = np.concatenate([(f[lag] - f[lag + 4]) * np.float32(decay ** lag) for lag in range(t_in - 5, -1, -1)],
                                  axis=1)
    return data


def attach_window_mean(data: SimilarityData, fp: pd.DataFrame, t_in: int) -> SimilarityData:
    """Pillar 1 target "window_mean": the fingerprint averaged over the window's quarters (the firm's structure over
    the window, less noisy than one quarter's), instead of the last quarter's; quarters without one are skipped."""
    F = fp.set_index(["ticker", "quarter"])
    uq = data.meta.quarter.unique()
    stack = []
    for lag in range(t_in):
        q = data.meta.quarter.map({u: _quarter_shift(u, -lag) for u in uq})
        stack.append(F.reindex(pd.MultiIndex.from_arrays([data.meta.ticker, q])).to_numpy(np.float32))
    with np.errstate(all="ignore"):
        data.target = np.nanmean(np.stack(stack), axis=0).astype(np.float32)
    return data


def attach_scenario(data: SimilarityData, scen: pd.DataFrame) -> SimilarityData:
    """Add the scenario of every window (quarter x series, e.g. the realised macro change over the next h quarters); NaN
    where unknown."""
    data.scenario = scen.reindex(data.meta.quarter).to_numpy(np.float32)
    return data


def _unit(a: torch.Tensor) -> torch.Tensor:
    return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def pair_loss(z: torch.Tensor, f: torch.Tensor, form: str = "mse", tau: float = 0.1) -> torch.Tensor:
    """MSE between the cosine matrices of embeddings and fingerprints, over the pairs i < j. form "neighbourhood":
    instead, each firm's distribution over the others of the batch, softmax(cos / tau), should match the target's
    (mean KL; stochastic neighbour embedding with cosines): the nearest firms count most."""
    cz, cf = _unit(z) @ _unit(z).T, _unit(f) @ _unit(f).T
    if form == "neighbourhood":
        off = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        logp = torch.log_softmax(cf.masked_fill(~off, -1e9) / tau, dim=1)
        logq = torch.log_softmax(cz.masked_fill(~off, -1e9) / tau, dim=1)
        return (logp.exp() * (logp - logq)).masked_fill(~off, 0.0).sum(1).mean()
    iu = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    return F.mse_loss(cz[iu[0], iu[1]], cf[iu[0], iu[1]])


def two_target_loss(z: torch.Tensor, f: torch.Tensor, d: torch.Tensor, lambda2: float = 0.5,
                    weight_moves: bool = False, form: str = "convex",
                    z2: torch.Tensor | None = None, pair_form: str = "mse",
                    tau: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (1 - lambda2) x pair_loss against the fingerprints f (Pillar 1) + lambda2 x the same against the changes d
    (Pillar 2), the latter over the pairs where both changes are known (rows of d without NaN); form "additive": L1 +
    lambda2 x L2 (runs before 1 Nov 2026). weight_moves: weight each pair by the product of the two changes' sizes
    (normalised to mean 1), so pairs of firms that barely move count less. z2: the embeddings of the Pillar 2 term
    (scenario-reweighted, ScenarioHead), default z. pair_form "neighbourhood": both terms are pair_loss's
    neighbourhood form (temperature tau). Returns (total, L1, L2).
    """
    w1 = 1.0 - lambda2 if form == "convex" else 1.0
    l1 = pair_loss(z, f, pair_form, tau)
    known = torch.isfinite(d).all(dim=1)
    if known.sum() < 2:
        return w1 * l1, l1, torch.zeros((), device=z.device)
    zk, dk = (z if z2 is None else z2)[known], d[known]
    if pair_form == "neighbourhood":
        l2 = pair_loss(zk, dk, pair_form, tau)
        return w1 * l1 + lambda2 * l2, l1, l2
    cz, cd = _unit(zk) @ _unit(zk).T, _unit(dk) @ _unit(dk).T
    iu = torch.triu_indices(len(zk), len(zk), offset=1, device=z.device)
    err = (cz[iu[0], iu[1]] - cd[iu[0], iu[1]]) ** 2
    if weight_moves:
        size = dk.norm(dim=1)
        w = size[iu[0]] * size[iu[1]]
        err = err * w / w.mean().clamp_min(1e-12)
    l2 = err.mean()
    return w1 * l1 + lambda2 * l2, l1, l2


def quarter_batches(meta: pd.DataFrame, batch_size: int, rng: np.random.Generator, min_size: int = 16):
    """Index batches, each from one quarter, in random order."""
    batches = []
    for _, idx in meta.groupby("quarter").indices.items():
        idx = rng.permutation(idx)
        batches += [idx[i:i + batch_size] for i in range(0, len(idx), batch_size) if len(idx[i:i + batch_size]) >= min_size]
    return [batches[i] for i in rng.permutation(len(batches))]


def train(model, data: SimilarityData, epochs: int, lr: float, batch_size: int, device, use_macro: bool,
          seed: int = 0, log=print, loss: str = "pillar1", lambda2: float = 0.5, weight_moves: bool = False,
          lambda_form: str = "convex", head=None, resp=None, resp_weight: float = 0.0, resp_cols: int = 0,
          pair_form: str = "mse", tau: float = 0.1, input_mask: float = 0.0) -> list[float]:
    """Train on pairs of firms within each quarter. loss: "pillar1" (pair_loss) or "pillar1+2" (two_target_loss,
    needs data.target2; lambda_form: its "convex" or "additive" form). head: a ScenarioHead trained with the model,
    whose reweighting by data.scenario enters the Pillar 2 term only. resp: a ResponseHead whose prediction of the
    change from (z, the first resp_cols scenario columns) adds resp_weight x its mean squared error to the loss.
    pair_form / tau: pair_loss's form. input_mask: share of the input cells (ratio x quarter) set missing (0, as in
    the data) at random in every batch, the targets unchanged (the encoder learns to place firms with holes).
    Returns the mean (total) loss per epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    params = list(model.parameters())
    if head is not None:
        head.to(device).train()
        params += list(head.parameters())
        scen = torch.tensor(np.nan_to_num(data.scenario))
    if resp is not None:
        resp.to(device).train()
        params += list(resp.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    target = torch.tensor(data.target)
    target2 = torch.tensor(data.target2) if loss == "pillar1+2" else None
    history = []
    for epoch in range(epochs):
        losses, parts = [], []
        for b in quarter_batches(data.meta, batch_size, rng):
            x = data.X_fin[b].to(device)
            if input_mask > 0:
                x = x.masked_fill(torch.rand(x.shape, device=device) < input_mask, 0.0)
            m = data.X_macro[b].to(device) if use_macro else None
            z, zm = model(x, m)                                       # zm: macro-conditioned z (--macro_target pillar2)
            zp2 = z if zm is None else zm
            if loss == "pillar1":
                total = pair_loss(z, target[b].to(device), pair_form, tau)
            else:
                z2 = (None if zm is None else zm) if head is None else head(zp2, scen[b].to(device))
                total, l1, l2 = two_target_loss(z, target[b].to(device), target2[b].to(device), lambda2, weight_moves,
                                                lambda_form, z2, pair_form, tau)
                if resp is not None:
                    d = target2[b].to(device)
                    ok = torch.isfinite(d).all(dim=1)
                    if ok.sum() > 1:
                        total = total + resp_weight * F.mse_loss(resp(z[ok], scen[b].to(device)[ok][:, :resp_cols]), d[ok])
                parts.append((l1.item(), l2.item()))
            opt.zero_grad()
            total.backward()
            opt.step()
            losses.append(total.item())
        history.append(float(np.mean(losses)))
        extra = f" (L1 {np.mean([p[0] for p in parts]):.4f}, L2 {np.mean([p[1] for p in parts]):.4f})" if parts else ""
        log(f"  epoch {epoch + 1}/{epochs}: pair loss {history[-1]:.4f}{extra}")
    return history


@torch.no_grad()
def embed(model, data: SimilarityData, device, macro: str = "real", batch_size: int = 4096, seed: int = 0,
          head=None, scenario: str = "real") -> np.ndarray:
    """Embeddings of all windows; macro = 'real', 'zero' (macro input set to 0), 'shuffled' (macro windows permuted
    across firm-quarters) or 'none' (model without a macro branch). With a ScenarioHead: the reweighted embeddings,
    scenario = 'real', 'zero' (s = 0: the head's bias only) or 'shuffled' (each quarter gets another quarter's
    scenario, the quarters of `data` permuted without fixed points)."""
    model.to(device).eval()
    X_macro = data.X_macro
    if macro == "zero":
        X_macro = torch.zeros_like(X_macro)
    elif macro == "shuffled":
        X_macro = X_macro[torch.randperm(len(X_macro), generator=torch.Generator().manual_seed(seed))]
    out = []
    for i in range(0, len(data.X_fin), batch_size):
        m = None if macro == "none" else X_macro[i:i + batch_size].to(device)
        z, zm = model(data.X_fin[i:i + batch_size].to(device), m)
        out.append((z if zm is None or head is None else zm).cpu().numpy())   # the head works on the Pillar 2 z
    Z = np.concatenate(out)
    if head is None:
        return Z
    S = np.nan_to_num(data.scenario)
    if scenario == "zero":
        S = np.zeros_like(S)
    elif scenario == "shuffled":
        q = data.meta.quarter.to_numpy()
        uq = np.unique(q)
        perm = np.random.default_rng(seed).permutation(uq)
        other = dict(zip(perm, np.roll(perm, 1)))                      # quarter -> another quarter
        first = {k: i for i, k in reversed(list(enumerate(q)))}        # a row of each quarter
        S = S[[first[other[k]] for k in q]]
    head.to(device).eval()
    return head(torch.tensor(Z, device=device), torch.tensor(S, device=device)).cpu().numpy()


# --- evaluation --------------------------------------------------------------------------------------------------

def _unit_np(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _knn(S: np.ndarray, k: int) -> np.ndarray:
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def _quarter_shift(q: str, h: int) -> str:
    n = int(q[:4]) * 4 + int(q[-1]) - 1 + h
    return f"{n // 4}Q{n % 4 + 1}"


def evaluate(V: np.ndarray, data: SimilarityData, growth: pd.Series | None = None, others: dict | None = None,
             max_firms: int = 1500, k: int = 10, seed: int = 0) -> dict:
    """The metrics of the module docstring for vectors V (one row per window of `data`); rows of V with missing values
    (e.g. the past-change baseline without history) are left out."""
    rng = np.random.default_rng(seed)
    valid = np.isfinite(V).all(axis=1)
    meta = data.meta.assign(row=np.arange(len(data.meta)))[valid]
    U, Uf = _unit_np(np.nan_to_num(V)), _unit_np(data.target)
    second = {s: T for s, T in (("", data.target2), ("_future", data.future_change)) if T is not None}
    has2 = {s: np.isfinite(T).all(axis=1) for s, T in second.items()}
    Ud = {s: _unit_np(np.nan_to_num(T)) for s, T in second.items()}
    fit, knn, peers, probes = [], [], [], {name: [] for name in (others or {})}
    fit2, knn2 = {s: [] for s in second}, {s: [] for s in second}
    for q, g in meta.groupby("quarter"):
        rows = g.row.to_numpy()
        if len(rows) < 50:
            continue
        if len(rows) > max_firms:
            rows = rng.choice(rows, max_firms, replace=False)
        S, Sf = U[rows] @ U[rows].T, Uf[rows] @ Uf[rows].T
        iu = np.triu_indices(len(rows), 1)
        fit.append(spearmanr(S[iu], Sf[iu])[0])
        nz, nf = _knn(S, k), _knn(Sf, k)
        knn.append(np.mean([len(set(a) & set(b)) / k for a, b in zip(nz, nf)]))
        for s in second:
            r2 = rows[has2[s][rows]]
            if len(r2) >= 50:
                S2, Sd = U[r2] @ U[r2].T, Ud[s][r2] @ Ud[s][r2].T
                i2 = np.triu_indices(len(r2), 1)
                fit2[s].append(spearmanr(S2[i2], Sd[i2])[0])
                knn2[s].append(np.mean([len(set(a) & set(b)) / k for a, b in zip(_knn(S2, k), _knn(Sd, k))]))
        if growth is not None:
            y = growth.reindex(pd.MultiIndex.from_frame(meta.loc[rows, ["ticker", "quarter"]])).to_numpy(float)
            pred = np.array([np.nanmean(y[n]) if np.isfinite(y[n]).any() else np.nan for n in nz])
            ok = np.isfinite(y) & np.isfinite(pred)
            if ok.sum() > 50:
                peers.append(np.corrcoef(y[ok], pred[ok])[0, 1] ** 2)
        for name, O in (others or {}).items():
            keys = pd.MultiIndex.from_frame(meta.loc[rows, ["ticker", "quarter"]])
            has = keys.isin(O.index)
            if has.sum() > 50:
                Uo = _unit_np(O.loc[keys[has]].to_numpy(float))
                jo = np.triu_indices(int(has.sum()), 1)
                probes[name].append(spearmanr((U[rows[has]] @ U[rows[has]].T)[jo], (Uo @ Uo.T)[jo])[0])
    pos = {key: r for r, key in zip(meta.row, zip(meta.ticker, meta.quarter))}      # key -> row of V
    reid = {}
    for h in (4, 20):
        hits = []
        for q, g in meta.groupby("quarter"):
            q2 = _quarter_shift(q, h)
            pairs = [(pos[(t, q)], pos[(t, q2)]) for t in g.ticker if (t, q2) in pos]
            if len(pairs) < 50:
                continue
            if len(pairs) > max_firms:
                pairs = [pairs[i] for i in rng.choice(len(pairs), max_firms, replace=False)]
            a, b = np.array(pairs).T
            hits.append(np.mean((U[a] @ U[b].T).argmax(1) == np.arange(len(a))))
        reid[f"reid{h}"] = float(np.mean(hits)) if hits else np.nan
    nxt = [(pos[(t, q)], pos[(t, _quarter_shift(q, 1))]) for t, q in zip(meta.ticker, meta.quarter) if (t, _quarter_shift(q, 1)) in pos]
    a, b = np.array(nxt).T if nxt else (np.array([], int), np.array([], int))
    mean = lambda v: float(np.mean(v)) if v else np.nan
    out = {"fit": float(np.mean(fit)), "knn10": float(np.mean(knn)), **reid,
           "fit2": mean(fit2.get("")), "knn10_2": mean(knn2.get("")),
           **{f"{m}{s}": mean(v[s]) for s in second if s for m, v in (("fit2", fit2), ("knn10_2", knn2))},
           "stability": float(np.mean(np.sum(U[a] * U[b], axis=1))) if len(a) else np.nan,
           "peers_r2": float(np.mean(peers)) if peers else np.nan}
    out.update({name: float(np.mean(v)) if v else np.nan for name, v in probes.items()})
    return out


def sic2_peers_r2(data: SimilarityData, growth: pd.Series, sector: pd.Series) -> float:
    """Benchmark for peers_r2: the mean revenue growth of the other firms of the same sector."""
    m = data.meta.assign(g=growth.reindex(pd.MultiIndex.from_frame(data.meta[["ticker", "quarter"]])).to_numpy(float),
                         s=data.meta.ticker.map(sector))
    r2 = []
    for _, d in m.dropna(subset=["g", "s"]).groupby("quarter"):
        tot, cnt = d.groupby("s").g.transform("sum"), d.groupby("s").g.transform("count")
        pred = (tot - d.g) / (cnt - 1).where(lambda c: c > 0)
        ok = pred.notna()
        if ok.sum() > 50:
            r2.append(np.corrcoef(d.g[ok], pred[ok])[0, 1] ** 2)
    return float(np.mean(r2)) if r2 else np.nan
