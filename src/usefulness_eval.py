"""
Step 3 of 3 (src/main.py): score embeddings on tasks they were never trained on (docs/usefulness_protocol.md), then
the evaluation battery. A "vector" is anything that gives one row of numbers per window: a run's embeddings, or a
baseline (the raw fingerprint, a PCA).

    xbrl2vec/bin/python src/usefulness_eval.py                 # every run of main_similarity.VARIANTS, then the battery
    xbrl2vec/bin/python src/usefulness_eval.py <run> [--force]  # only these runs, no battery

The battery (BATTERY, model-evaluation/): the interpretability figures, ablation.xlsx, comparison.xlsx and
robustness.xlsx, each in its folder's out/.

For every quarter, the pool is all firms of that quarter (2,115) and the focal firms are those of the evaluation set:
  unseen firms                the 317 unseen firms, quarters up to the cutoff (2015Q4-2022Q4)
  unseen firms, out of time   the same firms, after the cutoff (2023Q1-2025Q4)
  out of time                 the training firms after the cutoff

Tasks (each score is the mean over the set's quarters; "the 10 nearest" = highest cosine, never the firm itself):
  T1 peers      R² (squared correlation) across the focal firms between a firm's quantity and the mean of that
                quantity over its 10 nearest firms (Bhojraj, Lee & Oler 2003)
  T2 industry   share of the 10 nearest firms in the firm's SIC2; accuracy of their majority SIC2
  T3 groups     k-means of the pool into K groups, K = the number of SIC2 codes in the quarter; R² as in T1 between a
                firm's quantity and the mean over the other members of its group (Hoberg & Phillips 2016). The
                evaluation battery (model-evaluation/) calls this task "T2 groups".
  T4 probe      a ridge regression from the vector to a quantity (minus the quarter's mean), fitted on the training
                firms up to the cutoff; R² on the set, all its quarters pooled
  T5 holes      6 of the 32 ratios of the window's last quarter (20 %) set to 0, as if not reported: share of the 10
                nearest firms that stay the same

Quantities (never an input: the model sees ratios only): size = log total assets; revenue growth, asset growth and
share issuance = log(x_t / x_t-4); and, marked "next year", revenue and asset growth from t to t + 4. Only cells that
were observed or exactly derived (GOOD_CELLS of the panel's mask.parquet), despiked, clipped at the 1st and 99th
percentiles of each quarter.

Output: <run>/usefulness.csv (one row per set, task, quantity and quarter; q "all" is the mean over the quarters) and
data/out/usefulness/baselines_<key>.csv for the baselines and the SIC2 benchmark, computed once per data key.
The second half of the module (runs_of, cells_of, the neighbour helpers) serves the battery's scripts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

import main_similarity as ms
from models.xbrl2vec import XBRL2VecEncoder, embed
from services.data import LAST_QUARTER
from services.similarity import attach_window_deltas

RUNS = ms.RUNS
OUT = ms.DATA / "out" / "usefulness"
BATTERY = [ms.DATA.parent / "model-evaluation" / s for s in ("interpretabilty/interpretability_figures.py",
           "downstream-tasks/ablation.py", "downstream-tasks/comparison.py", "robustness/robustness.py")]
GOOD_CELLS = [1, 2, 4, 6, 7]          # observed, identity, fiscal-year-end redistribution, recomputed, confirmed zero
K_NN, MIN_PEERS, MASK_SHARE = 10, 3, 0.2
QUANTITIES = ["size", "revenue growth", "asset growth", "share issuance"]
FORWARD = ["revenue growth, next year", "asset growth, next year"]
SETS = ["out of time", "unseen firms, out of time", "unseen firms"]
CELL = ["set", "task", "quantity"]
RAW_FP, PCA32 = "raw fingerprint (32)", "PCA-32 of fingerprint + window deltas"


# --- data -----------------------------------------------------------------------------------------------------------

def load_lines(panel: Path) -> dict[str, pd.DataFrame]:
    """{line: ticker x quarter, in dollars} from the {bs,ins,cf}-raw.parquet of a panel."""
    wide = pd.concat([pd.read_parquet(panel / f"{s}-raw.parquet") for s in ("bs", "ins", "cf")], ignore_index=True)
    quarters = sorted(c for c in wide.columns if len(str(c)) == 6 and str(c)[4] == "Q")
    lines = {k: g.set_index("ticker")[quarters] for k, g in wide.groupby("features")}
    tickers = lines["Total Assets"].index
    return {k: v.reindex(tickers).astype(float) for k, v in lines.items()}


def despike(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """
    Values more than `factor` times away from the median of the firm's own five quarters around them (t-2..t+2) set
    missing (and zeros and negatives): total assets and revenue are smooth, and early filings often alternate between
    the true value and a wrong tag's (e.g. 14,068 / 1,145 / 14,027).
    """
    x = df.where(df > 0)
    med = x.T.rolling(5, center=True, min_periods=3).median().T
    return x.where(~(np.abs(np.log(x / med)) > np.log(factor)))


def winsorize(df: pd.DataFrame) -> pd.DataFrame:
    """Clip each quarter's cross-section (a column) at its 1st and 99th percentiles."""
    return df.clip(df.quantile(0.01), df.quantile(0.99), axis=1)


def quantities(panel: Path) -> pd.DataFrame:
    """(ticker, quarter) x the quantities of the module docstring."""
    L = {k: v.loc[:, v.columns <= LAST_QUARTER] for k, v in load_lines(panel).items()}   # "next year" too
    quarters = list(L["Total Assets"].columns)
    mask = pd.read_parquet(panel / "mask.parquet")                     # how each cell of the panel was obtained

    def good(line: str) -> pd.DataFrame:
        m = mask[mask.features == line].drop_duplicates("ticker").set_index("ticker")[quarters]
        return m.reindex(L[line].index).isin(GOOD_CELLS)

    ta = despike(L["Total Assets"]).where(good("Total Assets"))
    rev = despike(L["Total Revenue"]).where(good("Total Revenue"))
    sh = L["Shares Outstanding"].where(L["Shares Outstanding"] > 0).where(good("Shares Outstanding"))
    lag = lambda df: df.shift(4, axis=1)
    with np.errstate(all="ignore"):
        Q = {"size": np.log(ta), "revenue growth": np.log(rev / lag(rev)), "asset growth": np.log(ta / lag(ta)),
             "share issuance": np.log(sh / lag(sh)).where(lambda x: x.abs() <= np.log(1.5))}   # beyond ±50 %: splits
    Q = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in Q.items()}
    Q["revenue growth, next year"] = Q["revenue growth"].shift(-4, axis=1)
    Q["asset growth, next year"] = Q["asset growth"].shift(-4, axis=1)
    Q = {k: winsorize(v) for k, v in Q.items()}
    return pd.DataFrame({k: v.stack() for k, v in Q.items()}).rename_axis(["ticker", "quarter"])


def sic_codes(panels_dir: Path) -> pd.Series:
    """ticker -> 4-digit SIC code (SEC metadata of the download the panels were built from)."""
    meta = ms.DATA / "in" / panels_dir.name.replace("panels_", "download_") / "metadata.parquet"
    m = pd.read_parquet(meta).drop_duplicates("ticker").set_index("ticker")
    return pd.to_numeric(m.sic, errors="coerce").dropna().astype(int)


class Group:
    """
    Everything needed to score the runs that share their data (panels, t_in, cutoff). Every array has one row per
    window (N = 86,715: 2,115 firms x 41 quarters), the training firms first:
      meta        ticker, quarter, firms ("train" / "unseen")
      X           [N, 12, 32] the input windows                    FP  [N, 32] the fingerprint of the last quarter
      D           [N, 256] the window deltas (decay 1)             Q   the quantities, one column each
      sic         [N] SIC2 of the firm (-1: none)                  sic4  [N] its 4-digit SIC
      focal[set]  [N] bool, the rows of each evaluation set        train_rows  the rows of the training firms <= cutoff
      holes       [N, 6] the ratios hidden in T5 (fixed, seed 0)   ratios  the 32 ratio names (the columns of FP)
      xcol        [32] the column of X that holds each ratio       panel  the panel folder
    """

    def __init__(self, run: str):
        cfg = json.loads((RUNS / run / "config.json").read_text())
        self.panels_dir, self.t_in, self.cutoff = Path(cfg["panels_dir"]), int(cfg["t_in"]), cfg["cutoff"]
        self.panel = self.panels_dir / ms.PANEL
        sets, fp = ms.sample_sets(self.panel, self.t_in)
        sets = {k: attach_window_deltas(v, fp, self.t_in) for k, v in sets.items()}
        self.meta = pd.concat([sets["train firms"].meta.assign(firms="train"), sets["unseen firms"].meta.assign(firms="unseen")],
                              ignore_index=True)
        self.FP = np.concatenate([s.target for s in sets.values()])
        self.D = np.concatenate([s.target2 for s in sets.values()])
        self.X = torch.cat([s.X for s in sets.values()])
        self.ratios = [c for c in fp.columns if c not in ("ticker", "quarter")]
        self.xcol = np.array([list(sets["train firms"].cols).index(c) for c in self.ratios])
        keys = pd.MultiIndex.from_frame(self.meta[["ticker", "quarter"]])
        self.Q = quantities(self.panel).reindex(keys).reset_index(drop=True)
        self.sic4 = self.meta.ticker.map(sic_codes(self.panels_dir)).fillna(-1).astype(int).to_numpy()
        self.sic = np.where(self.sic4 >= 0, self.sic4 // 100, -1)
        q = self.meta.quarter
        self.train_rows = np.flatnonzero((self.meta.firms == "train") & q.le(self.cutoff))
        self.focal = {"out of time": ((self.meta.firms == "train") & q.gt(self.cutoff)).to_numpy(),
                      "unseen firms, out of time": ((self.meta.firms == "unseen") & q.gt(self.cutoff)).to_numpy(),
                      "unseen firms": ((self.meta.firms == "unseen") & q.le(self.cutoff)).to_numpy()}
        self.quarters = {s: sorted(self.meta.quarter[f].unique()) for s, f in self.focal.items()}
        self.holes = np.random.default_rng(0).random(self.FP.shape).argsort(axis=1)[:, :round(MASK_SHARE * self.FP.shape[1])]
        # PCA-32 baseline: PCA of [fingerprint, window deltas], each block scaled to the same total variance, fitted
        # on the training rows whose deltas are known
        tr = self.train_rows[np.isfinite(self.D[self.train_rows]).all(axis=1)]
        self.scale = (np.sqrt(self.FP[tr].var(0).sum()), np.sqrt(self.D[tr].var(0).sum()))
        self.pca32 = PCA(32, random_state=0).fit(self.both(self.FP[tr], self.D[tr]))
        print(f"[INFO] {self.panel} t_in {self.t_in} cutoff {self.cutoff}: {len(self.meta):,} windows, "
              f"focal " + ", ".join(f"{s} {f.sum():,}" for s, f in self.focal.items()), flush=True)

    def both(self, FP: np.ndarray, D: np.ndarray) -> np.ndarray:
        """[fingerprint, window deltas] side by side, each block divided by its total standard deviation (PCA-32's input)."""
        return np.concatenate([FP / self.scale[0], D / self.scale[1]], axis=1)

    def holed_X(self, rows: np.ndarray, holes: np.ndarray) -> torch.Tensor:
        """The input windows of `rows` with the ratios `holes` [len(rows), k] of the last quarter set to 0."""
        X = self.X[rows].clone()
        last = X[:, -1, :]
        last.scatter_(1, torch.as_tensor(self.xcol[holes]), 0.0)
        X[:, -1, :] = last
        return X

    def holed_raw(self, rows: np.ndarray, holes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fingerprint and window deltas of `rows` with the ratios `holes` of the last quarter set to 0 (the last move,
        f(t) - f(t - 4), changes with the fingerprint)."""
        FP, D = self.FP[rows].copy(), self.D[rows].copy()
        K = FP.shape[1]
        before = FP - D[:, -K:]                                        # f(t - 4)
        np.put_along_axis(FP, holes, 0.0, axis=1)
        D[:, -K:] = FP - before
        return FP, D

    def pca32_of(self, FP: np.ndarray, D: np.ndarray) -> np.ndarray:
        """The PCA-32 vectors; NaN rows where the deltas are unknown (those rows are then left out of the scores)."""
        A = self.both(FP, D)
        V = np.full((len(A), 32), np.nan, np.float32)
        ok = np.isfinite(A).all(axis=1)
        V[ok] = self.pca32.transform(A[ok])
        return V

    def baselines(self) -> dict:
        """name -> (the vectors of every row, a function giving the vectors of `rows` with their T5 holes)."""
        return {RAW_FP: (lambda: self.FP, lambda r: self.holed_raw(r, self.holes[r])[0]),
                PCA32: (lambda: self.pca32_of(self.FP, self.D), lambda r: self.pca32_of(*self.holed_raw(r, self.holes[r])))}

    def encoder(self, run: str):
        """A function X [n, 12, 32] -> embeddings [n, latent_dim] with the trained encoder of `run`."""
        cfg = json.loads((RUNS / run / "config.json").read_text())
        m = XBRL2VecEncoder(self.t_in, self.X.shape[-1], int(cfg["latent_dim"]), int(cfg["hidden"]))
        m.load_state_dict(torch.load(RUNS / run / "encoder.pth", map_location="cpu"))
        return lambda X: embed(m, X)


# --- tasks ----------------------------------------------------------------------------------------------------------

def _unit(a: np.ndarray) -> np.ndarray:
    """Each row divided by its length: dot products of the result are cosines."""
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _r2(y: np.ndarray, p: np.ndarray, min_n: int = 50) -> float:
    """Squared correlation between y and p over the rows where both are known (NaN with fewer than min_n rows)."""
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < min_n or np.std(p[ok]) == 0:
        return np.nan
    return float(np.corrcoef(y[ok], p[ok])[0, 1] ** 2)


def _peer_mean(y: np.ndarray, nn: np.ndarray) -> np.ndarray:
    """Mean of y over each row's neighbours nn [n, k] (NaN with fewer than MIN_PEERS known values)."""
    v = y[nn]
    n = np.isfinite(v).sum(1)
    with np.errstate(all="ignore"):
        return np.where(n >= MIN_PEERS, np.nansum(v, axis=1) / np.maximum(n, 1), np.nan)


def _group_mean(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean of y over the other firms of the same group (>= MIN_PEERS of them), NaN without a group (label < 0)."""
    df = pd.DataFrame({"y": y, "s": labels})
    ok = df.y.notna() & (df.s >= 0)
    tot = df.y.where(ok, 0).groupby(df.s).transform("sum")
    cnt = ok.astype(int).groupby(df.s).transform("sum")
    own = ok.astype(int)
    with np.errstate(all="ignore"):
        m = (tot - df.y.where(ok, 0)) / (cnt - own)
    return np.where((df.s >= 0) & (cnt - own >= MIN_PEERS), m, np.nan)


def _mean_rows(acc: dict, name: str) -> list[dict]:
    """acc: (set, task, quantity) -> [(quarter, value)]. One row per quarter and one, q "all", with their mean."""
    rows = []
    for (s, t, k), qv in acc.items():
        v = np.array([x for _, x in qv], float)
        rows.append({"vector": name, "set": s, "task": t, "quantity": k, "q": "all",
                     "value": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan, "quarters": int(np.isfinite(v).sum())})
        rows += [{"vector": name, "set": s, "task": t, "quantity": k, "q": q, "value": x, "quarters": 1} for q, x in qv]
    return rows


def score(g: Group, V: np.ndarray, Vholes, name: str, seed: int = 0) -> list[dict]:
    """
    The rows (vector, set, task, quantity, q, value) of the vectors V [N, d], one row of V per row of g.
    Vholes: a function rows -> the vectors of those rows with their T5 holes (None: no T5). seed: of T3's k-means.
    """
    valid = np.isfinite(V).all(axis=1)
    U = _unit(np.nan_to_num(V))
    quants = QUANTITIES + FORWARD
    acc = {}                                                     # (set, task, quantity) -> [(quarter, value)]
    add = lambda s, t, k, v: acc.setdefault((s, t, k), []).append((q, v))
    hole_rows = np.flatnonzero(np.any([g.focal[s] for s in SETS], axis=0) & valid)
    Uh = np.full((len(V), U.shape[1]), np.nan, np.float32)
    if Vholes is not None and len(hole_rows):
        Uh[hole_rows] = _unit(np.nan_to_num(Vholes(hole_rows)))
    for q, pool_df in g.meta.groupby("quarter"):
        pool = pool_df.index.to_numpy()
        pool = pool[valid[pool]]
        sets_here = [s for s in SETS if g.focal[s][pool].sum() >= 50]
        if not sets_here:
            continue
        S = U[pool] @ U[pool].T                                  # cosine of every pair of firms of the quarter
        np.fill_diagonal(S, -np.inf)                             # a firm is never its own neighbour
        nn = np.argpartition(-S, K_NN, axis=1)[:, :K_NN]         # positions in the pool of the 10 nearest
        sic = g.sic[pool]
        Y = g.Q.iloc[pool]
        peer = {k: _peer_mean(Y[k].to_numpy(float), nn) for k in quants}
        K = len(np.unique(sic[sic >= 0]))
        labels = KMeans(K, n_init=2, max_iter=100, random_state=seed).fit_predict(U[pool])
        labels[sic < 0] = -1                                     # the firms SIC2 can score, as its benchmark
        for s in sets_here:
            f = g.focal[s][pool]
            for k in quants:
                add(s, "T1 peers", k, _r2(Y[k].to_numpy(float)[f], peer[k][f]))
            fs = f & (sic >= 0)
            same = (sic[nn] == sic[:, None]) & (sic[nn] >= 0)
            known = (sic[nn] >= 0).sum(1)
            add(s, "T2 industry", "same SIC2 among 10 nearest", float(np.mean(same[fs].sum(1) / np.maximum(known[fs], 1))))
            maj = np.array([np.bincount(r[r >= 0]).argmax() if (r >= 0).any() else -1 for r in sic[nn[fs]]])
            add(s, "T2 industry", "majority SIC2 correct", float(np.mean(maj == sic[fs])))
            for k in quants:
                y = Y[k].to_numpy(float)
                add(s, "T3 groups", k, _r2(y[f], _group_mean(y, labels)[f]))
            fh = np.flatnonzero(f)
            fh = fh[np.isfinite(Uh[pool[fh]]).all(axis=1)]
            if len(fh) >= 50:
                Sh = Uh[pool[fh]] @ U[pool].T                    # the firm with holes against every clean firm
                Sh[np.arange(len(fh)), fh] = -np.inf             # not its own clean self
                nnh = np.argpartition(-Sh, K_NN, axis=1)[:, :K_NN]
                add(s, "T5 holes", "10-NN overlap, 20 % of last quarter missing",
                    float(np.mean([len(set(a) & set(b)) / K_NN for a, b in zip(nnh, nn[fh])])))
    return _mean_rows(acc, name) + probe(g, V, name)


def probe(g: Group, V: np.ndarray, name: str) -> list[dict]:
    """T4: ridge from V to each quantity minus its quarter's mean, fitted on the training rows; R² on each set."""
    valid = np.isfinite(V).all(axis=1)
    rows = []
    Qd = g.Q.groupby(g.meta.quarter).transform(lambda c: c - c.mean())
    tr = g.train_rows[valid[g.train_rows]]
    rng = np.random.default_rng(0)
    for k in QUANTITIES + FORWARD:
        y = Qd[k].to_numpy(float)
        t = tr[np.isfinite(y[tr])]
        if len(t) > 60000:
            t = rng.choice(t, 60000, replace=False)
        mu, sd = V[t].mean(0), V[t].std(0) + 1e-8
        m = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit((V[t] - mu) / sd, y[t])
        for s in SETS:
            e = np.flatnonzero(g.focal[s] & valid & np.isfinite(y))
            if len(e) < 100:
                continue
            p = pd.Series(m.predict((V[e] - mu) / sd)).groupby(g.meta.quarter.iloc[e].to_numpy()).transform(lambda c: c - c.mean()).to_numpy()
            rows.append({"vector": name, "set": s, "task": "T4 probe", "quantity": k,
                         "q": "all", "value": float(1 - ((y[e] - p) ** 2).sum() / ((y[e] - y[e].mean()) ** 2).sum()),
                         "quarters": np.nan})
    return rows


def sic_rows(g: Group, labels: np.ndarray | None = None, name: str = "SIC2") -> list[dict]:
    """Industry codes as the vector (SIC2, or `labels` per row, < 0 for none): the other firms of the same industry are
    both a firm's T1 peers and its T3 group."""
    acc = {}
    for q, pool_df in g.meta.groupby("quarter"):
        pool = pool_df.index.to_numpy()
        sic = (g.sic if labels is None else labels)[pool]
        Y = g.Q.iloc[pool]
        for s in SETS:
            f = g.focal[s][pool]
            if f.sum() < 50:
                continue
            for k in QUANTITIES + FORWARD:
                y = Y[k].to_numpy(float)
                r2 = _r2(y[f], _group_mean(y, sic)[f])
                for t in ("T1 peers", "T3 groups"):
                    acc.setdefault((s, t, k), []).append((q, r2))
    return _mean_rows(acc, name)


def baselines_file(run: str) -> Path:
    """The scores of the baselines and SIC2 on `run`'s data: one file per panels folder, t_in and cutoff."""
    c = json.loads((RUNS / run / "config.json").read_text())
    return OUT / f"baselines_{Path(c['panels_dir']).name}_tin{c['t_in']}_{c['cutoff']}.csv"


def run_all(runs: list[str], force: bool = False) -> None:
    """Score the runs that have no usefulness.csv (all of them with force), and the baselines of their data if missing."""
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {}                                                  # runs that share their data share a Group
    for r in runs:
        groups.setdefault(baselines_file(r), []).append(r)
    for bfile, rs in groups.items():
        todo = [r for r in rs if force or not (RUNS / r / "usefulness.csv").exists()]
        if not todo and bfile.exists():
            continue
        t0 = time.time()
        g = Group(rs[0])
        if force or not bfile.exists():
            rows = sic_rows(g)
            for name, (full, holes) in g.baselines().items():
                rows += score(g, full(), holes, name)
                print(f"[INFO]   baseline {name} ({time.time() - t0:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(bfile, index=False)
        for r in todo:
            enc = g.encoder(r)
            res = score(g, enc(g.X), lambda rows: enc(g.holed_X(rows, g.holes[rows])), "model")
            pd.DataFrame(res).to_csv(RUNS / r / "usefulness.csv", index=False)
            print(f"[INFO]   {r} scored ({time.time() - t0:.0f} s)", flush=True)


# --- runs and neighbours: helpers of the evaluation battery (model-evaluation/) -------------------------------------

seed_of = ms.seed_of


def runs_of(variant: str, scored: bool = False) -> list[str]:
    """The finished runs of a variant of main_similarity.VARIANTS on the latest panels (with usefulness.csv if
    scored), seed 42 first, then by seed."""
    return [r for r in ms.runs(variant) if not scored or (RUNS / r / "usefulness.csv").exists()]


def cells_of(runs: list[str]) -> pd.DataFrame:
    """run x (set, task, quantity): the scores of the runs' usefulness.csv (means over the quarters)."""
    out = {}
    for r in runs:
        u = pd.read_csv(RUNS / r / "usefulness.csv")
        out[r] = u[u.q.eq("all")].set_index(CELL).value
    return pd.DataFrame(out).T


def baselines_of(run: str) -> pd.DataFrame:
    """(set, task, quantity) x vector: the scores of the baselines scored with `run`'s data."""
    b = pd.read_csv(baselines_file(run))
    return b[b.q.eq("all")].pivot_table(index=CELL, columns="vector", values="value")


def pool_of(g: Group, q: str) -> np.ndarray:
    """The rows of g in quarter q (all firms of the quarter)."""
    return np.flatnonzero(g.meta.quarter.eq(q).to_numpy())


def knn(U: np.ndarray, k: int = K_NN) -> np.ndarray:
    """Positions of the k nearest (cosine) of every row of the unit vectors U among the others, nearest first."""
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, nn, axis=1), axis=1)
    return np.take_along_axis(nn, order, axis=1)


def knn_holed(Uh: np.ndarray, U: np.ndarray, own: np.ndarray, k: int = K_NN) -> np.ndarray:
    """The k nearest in the clean pool U of the holed rows Uh, whose own clean positions in U are `own`."""
    S = Uh @ U.T
    S[np.arange(len(Uh)), own] = -np.inf
    return np.argpartition(-S, k, axis=1)[:, :k]


def random_holes(n: int, K: int, k: int, seed: int = 0) -> np.ndarray:
    """For each of n rows, k distinct column indices out of K, at random."""
    return np.random.default_rng(seed).random((n, K)).argsort(axis=1)[:, :k]


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row, the share of a's neighbours that are also b's."""
    return (a[:, :, None] == b[:, None, :]).any(axis=2).mean(axis=1)


def main() -> None:
    """Score the runs named on the command line; with none, every run of every variant, then the battery."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="Run folders in data/out/similarity (default: every run of every variant).")
    p.add_argument("--force", action="store_true", help="Score again runs and baselines already scored.")
    args = p.parse_args()
    run_all(args.runs or [r for v in ms.VARIANTS for r in runs_of(v)], args.force)
    if not args.runs:
        for script in BATTERY:
            subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
