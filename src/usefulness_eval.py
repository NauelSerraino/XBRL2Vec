"""
Are the embeddings useful? Tasks they were never trained on (docs/usefulness_protocol.md), for each run's encoder
(main_similarity.py, --skip_macro_model runs) and the baselines a user could take instead:

  T1 peer explanation     R² of a firm's quantity on the mean over its 10 nearest firms (Bhojraj, Lee & Oler 2003);
                          benchmark SIC2 peers
  T2 industry recovery    share of the 10 nearest firms in the firm's SIC2; accuracy of their majority SIC2
  T3 data-driven groups   k-means into K groups (K = SIC2 codes in the quarter), R² of a quantity on the mean over the
                          other members of its group, as T1 (Hoberg & Phillips 2016); benchmark SIC2 groups (= SIC2 T1)
  T4 linear probe         ridge from the vector to a quantity (demeaned per quarter), trained on the training firms up
                          to the cutoff, R² on the evaluation firms
  T5 holes                20 % of the ratios of the window's last quarter missing (0, as in the data): overlap of the
                          10 nearest firms with the clean ones

Quantities, none of them an input (common-sizing removes scale and level growth): log total assets, year-on-year log
growth of quarterly revenue, of total assets and of the share count (beyond ±50 %: splits, left out), and, secondary
and marked "next year", revenue and asset growth from t to t + 4. Only observed or exactly derived cells (mask codes
1, 2, 4, 6, 7); despiked; winsorised 1 / 99 % per quarter. Peer pool: all firms of the quarter; focal firms: the
evaluation set (out of time; unseen firms out of time; unseen firms up to the cutoff).

    cd src && ../xbrl2vec/bin/python usefulness_eval.py <run> [<run> ...]      # score runs (and their baselines)
    cd src && ../xbrl2vec/bin/python usefulness_eval.py --summary              # tables and claims of the protocol

Runs sharing their data (panel, t_in, cutoff) are scored together, the data built once. Output:
data/out/similarity/<run>/usefulness.csv, data/out/usefulness/baselines_<panel>_tin<t>_<cutoff>.csv.
"""
from __future__ import annotations

import argparse
import json
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
from models.similarity_encoder import SimilarityEncoder
from preprocess.pillar_targets import _winsorize, despike, load_lines
from services.config import DEVICE
from services.similarity import attach_window_deltas

RUNS = ms.DATA / "out" / "similarity"
OUT = ms.DATA / "out" / "usefulness"
GOOD_CELLS = [1, 2, 4, 6, 7]          # observed, identity, fiscal-year-end redistribution, recomputed, confirmed zero
K_NN, MIN_PEERS, MASK_SHARE = 10, 3, 0.2
QUANTITIES = ["size", "revenue growth", "asset growth", "share issuance"]
FORWARD = ["revenue growth, next year", "asset growth, next year"]
SETS = ["out of time", "unseen firms, out of time", "unseen firms"]


# --- data -----------------------------------------------------------------------------------------------------------

def quantities(panel: Path) -> pd.DataFrame:
    """(ticker, quarter) x the held-out quantities."""
    L = {k: v.loc[:, v.columns <= ms.LAST_QUARTER] for k, v in load_lines(panel).items()}   # "next year" too
    quarters = list(L["Total Assets"].columns)
    mask_file = panel / "mask.parquet"
    mask = pd.read_parquet(mask_file) if mask_file.exists() else None

    def good(line: str) -> pd.DataFrame:
        if mask is None:                                               # complete panel: every cell observed
            return pd.DataFrame(True, index=L[line].index, columns=quarters)
        m = mask[mask.features == line].drop_duplicates("ticker").set_index("ticker")[quarters]
        return m.reindex(L[line].index).isin(GOOD_CELLS)

    ta = despike(L["Total Assets"]).where(good("Total Assets"))
    rev = despike(L["Total Revenue"]).where(good("Total Revenue"))
    sh = L["Shares Outstanding"].where(L["Shares Outstanding"] > 0).where(good("Shares Outstanding"))
    lag = lambda df: df.shift(4, axis=1)
    with np.errstate(all="ignore"):
        Q = {"size": np.log(ta), "revenue growth": np.log(rev / lag(rev)), "asset growth": np.log(ta / lag(ta)),
             "share issuance": np.log(sh / lag(sh)).where(lambda x: x.abs() <= np.log(1.5))}
    Q = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in Q.items()}
    Q["revenue growth, next year"] = Q["revenue growth"].shift(-4, axis=1)
    Q["asset growth, next year"] = Q["asset growth"].shift(-4, axis=1)
    Q = {k: _winsorize(v) for k, v in Q.items()}
    return pd.DataFrame({k: v.stack() for k, v in Q.items()}).rename_axis(["ticker", "quarter"])


def sic2(panels_dir: Path) -> pd.Series:
    """ticker -> SIC 2-digit code (SEC metadata of the download the panels were built from)."""
    meta = ms.DATA / "in" / panels_dir.name.replace("panels_", "download_") / "metadata.parquet"
    m = pd.read_parquet(meta).drop_duplicates("ticker").set_index("ticker")
    return (pd.to_numeric(m.sic, errors="coerce") // 100).dropna().astype(int)


class Group:
    """The data of the runs sharing panel, t_in and cutoff: every window of the training and unseen firms (rows), the
    raw vectors, quantities and SIC2 per row, the evaluation sets."""

    def __init__(self, run: str):
        cfg = json.loads((RUNS / run / "config.json").read_text())
        argv, sys.argv = sys.argv, [sys.argv[0]]
        try:
            args = ms.parse_args()
        finally:
            sys.argv = argv
        args.reuse = run
        ms.apply_reuse(args)
        if args.panels_dir is None:
            args.panels_dir = sorted(p for p in (ms.DATA / "in").glob("panels_*") if (p / "exog.parquet").exists())[-1]
        self.args, self.cutoff = args, cfg["cutoff"]
        self.last_train = cfg.get("last_train_quarter", self.cutoff)
        panel = args.panels_dir / args.panel
        self.fingerprint = cfg.get("fingerprint", "zscore")
        self.input_slog = args.input_slog                              # False since 8 Nov: the input is the fingerprint
        sets, fp = ms.sample_sets(args, panel)
        sets = {k: attach_window_deltas(v, fp, args.t_in) for k, v in sets.items()}
        self.sets = sets
        self.meta = pd.concat([sets["train firms"].meta.assign(firms="train"), sets["unseen firms"].meta.assign(firms="unseen")],
                              ignore_index=True)
        self.FP = np.concatenate([s.target for s in sets.values()])
        self.D = np.concatenate([s.target2 for s in sets.values()])
        self.X = torch.cat([s.X_fin for s in sets.values()])
        ratio_cols = [c for c in fp.columns if c not in ("ticker", "quarter")]
        self.xcol = np.array([list(sets["train firms"].fin_cols).index(c) for c in ratio_cols])   # fp column -> X column
        Q = quantities(panel)
        keys = pd.MultiIndex.from_frame(self.meta[["ticker", "quarter"]])
        self.Q = Q.reindex(keys).reset_index(drop=True)
        self.sic = self.meta.ticker.map(sic2(args.panels_dir)).fillna(-1).astype(int).to_numpy()
        q = self.meta.quarter
        self.train_rows = np.flatnonzero((self.meta.firms == "train") & q.le(self.last_train))
        self.focal = {"out of time": ((self.meta.firms == "train") & q.gt(self.cutoff)).to_numpy(),
                      "unseen firms, out of time": ((self.meta.firms == "unseen") & q.gt(self.cutoff)).to_numpy(),
                      "unseen firms": ((self.meta.firms == "unseen") & q.le(self.cutoff)).to_numpy()}
        self.quarters = {s: sorted(self.meta.quarter[f].unique()) for s, f in self.focal.items()}
        self.rng_mask = np.random.default_rng(0)
        self.holes = self.rng_mask.random(self.FP.shape).argsort(axis=1)[:, :round(MASK_SHARE * self.FP.shape[1])]
        ok = np.isfinite(self.D).all(axis=1)
        tr = self.train_rows[ok[self.train_rows]]
        self.scale = (np.sqrt(self.FP[tr].var(0).sum()), np.sqrt(self.D[tr].var(0).sum()))
        self.pca_fp = PCA(16, random_state=0).fit(self.FP[tr])
        self.pca_both = PCA(16, random_state=0).fit(self.both(self.FP[tr], self.D[tr]))
        self.pca_both_k = {k: PCA(k, random_state=0).fit(self.both(self.FP[tr], self.D[tr])) for k in (32, 64)}
        flat = self.X[tr].reshape(len(tr), -1).numpy()
        self.pca_in = PCA(16, random_state=0).fit(flat)
        self.scale_mean = np.sqrt(self.window_mean(self.X[tr]).var(0).sum())
        print(f"[INFO] {panel.name} t_in {args.t_in} cutoff {self.cutoff}: {len(self.meta):,} windows, "
              f"focal " + ", ".join(f"{s} {f.sum():,}" for s, f in self.focal.items()), flush=True)

    def key(self) -> str:
        # shortcut: no data end in the key; delete a cutoff's cached baselines before re-scoring a run from before 14 Nov
        return f"{self.args.panels_dir.name}_{self.args.panel}_tin{self.args.t_in}_{self.cutoff}" + \
            ("" if self.fingerprint == "zscore" else f"_{self.fingerprint}") + ("" if self.input_slog else "_inputF") + \
            ("" if self.args.sample == "all" else f"_{self.args.sample}")

    def window_mean(self, X: torch.Tensor) -> np.ndarray:
        """The fingerprint averaged over the window's quarters (X is the fingerprint, or its symmetric log in the runs
        before 8 Nov, inverted here)."""
        F = torch.sign(X) * torch.expm1(torch.abs(X)) if self.input_slog else X
        return F.mean(1)[:, self.xcol].numpy()

    def both(self, FP: np.ndarray, D: np.ndarray) -> np.ndarray:
        return np.concatenate([FP / self.scale[0], D / self.scale[1]], axis=1)

    def masked_raw(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fingerprint and window deltas of `rows` with their holes: the masked ratios of the last quarter are 0."""
        FP, D = self.FP[rows].copy(), self.D[rows].copy()
        K = FP.shape[1]
        before = FP - D[:, -K:]                                        # f(t - 4)
        np.put_along_axis(FP, self.holes[rows], 0.0, axis=1)
        D[:, -K:] = FP - before
        return FP, D

    def baselines(self) -> dict:
        """name -> (vectors of every row, vectors of `rows` with holes)."""
        def nan_rows(f, A, k=16):                                      # unknown deltas: the row is left out
            V = np.full((len(A), k), np.nan, np.float32)
            ok = np.isfinite(A).all(axis=1)
            V[ok] = f(A[ok])
            return V
        return {
            "raw fingerprint (32)": (lambda: self.FP, lambda r: self.masked_raw(r)[0]),
            "window deltas": (lambda: self.D, lambda r: self.masked_raw(r)[1]),
            "fingerprint + window deltas": (lambda: self.both(self.FP, self.D), lambda r: self.both(*self.masked_raw(r))),
            "PCA-16 of fingerprint": (lambda: self.pca_fp.transform(self.FP), lambda r: self.pca_fp.transform(self.masked_raw(r)[0])),
            "PCA-16 of fingerprint + window deltas": (lambda: nan_rows(self.pca_both.transform, self.both(self.FP, self.D)),
                                                      lambda r: nan_rows(self.pca_both.transform, self.both(*self.masked_raw(r)))),
            "PCA-16 of input window": (lambda: self.pca_in.transform(self.X.reshape(len(self.X), -1).numpy()),
                                       lambda r: self.pca_in.transform(self.masked_X(r).reshape(len(r), -1).numpy())),
            # added after the protocol: the same-size baseline for models with z 32 / 64
            **{f"PCA-{k} of fingerprint + window deltas": (
                (lambda p: (lambda: nan_rows(p.transform, self.both(self.FP, self.D), p.n_components_),
                            lambda r: nan_rows(p.transform, self.both(*self.masked_raw(r)), p.n_components_)))(p))
               for k, p in self.pca_both_k.items()},
            # exploratory, added after the protocol: is the window's average structure a cleaner fingerprint?
            "window-mean fingerprint (32)": (lambda: self.window_mean(self.X), lambda r: self.window_mean(self.masked_X(r))),
            "window-mean fingerprint + window deltas": (
                lambda: np.concatenate([self.window_mean(self.X) / self.scale_mean, self.D / self.scale[1]], axis=1),
                lambda r: np.concatenate([self.window_mean(self.masked_X(r)) / self.scale_mean,
                                          self.masked_raw(r)[1] / self.scale[1]], axis=1)),
        }

    def masked_X(self, rows: np.ndarray) -> torch.Tensor:
        X = self.X[rows].clone()
        cols = torch.tensor(self.xcol[self.holes[rows]])
        last = X[:, -1, :]
        last.scatter_(1, cols, 0.0)
        X[:, -1, :] = last
        return X

    def model(self, run: str):
        cfg = json.loads((RUNS / run / "config.json").read_text())
        m = SimilarityEncoder(self.args.t_in, self.X.shape[-1], self.sets["train firms"].X_macro.shape[-1],
                              int(cfg["latent_dim"]), use_macro=False, hidden=int(cfg["hidden"]))
        m.load_state_dict(torch.load(RUNS / run / "encoder_no_macro.pth", map_location="cpu"))
        m.to(DEVICE).eval()

        @torch.no_grad()
        def emb(X: torch.Tensor) -> np.ndarray:
            return np.concatenate([m(X[i:i + 4096].to(DEVICE), None)[0].cpu().numpy() for i in range(0, len(X), 4096)])
        return (lambda: emb(self.X), lambda r: emb(self.masked_X(r)))


# --- tasks ----------------------------------------------------------------------------------------------------------

def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def _r2(y: np.ndarray, p: np.ndarray, min_n: int = 50) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < min_n or np.std(p[ok]) == 0:
        return np.nan
    return float(np.corrcoef(y[ok], p[ok])[0, 1] ** 2)


def _peer_mean(y: np.ndarray, nn: np.ndarray) -> np.ndarray:
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


def score(g: Group, V: np.ndarray, Vholes, name: str, seed: int = 0) -> list[dict]:
    """Rows (vector, set, task, quantity, value) for vectors V of every row of g."""
    rows = []
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
        S = U[pool] @ U[pool].T
        np.fill_diagonal(S, -np.inf)
        nn = np.argpartition(-S, K_NN, axis=1)[:, :K_NN]
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
                Sh = Uh[pool[fh]] @ U[pool].T
                Sh[np.arange(len(fh)), fh] = -np.inf                     # not its own clean self
                nnh = np.argpartition(-Sh, K_NN, axis=1)[:, :K_NN]
                add(s, "T5 holes", "10-NN overlap, 20 % of last quarter missing",
                    float(np.mean([len(set(a) & set(b)) / K_NN for a, b in zip(nnh, nn[fh])])))
    rows = _mean_rows(acc, name)
    rows += probe(g, V, name)
    return rows


def _mean_rows(acc: dict, name: str) -> list[dict]:
    """The mean over quarters of each cell (q "all") and each quarter's value (for paired tests across quarters)."""
    rows = []
    for (s, t, k), qv in acc.items():
        v = np.array([x for _, x in qv], float)
        rows.append({"vector": name, "set": s, "task": t, "quantity": k, "q": "all",
                     "value": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan, "quarters": int(np.isfinite(v).sum())})
        rows += [{"vector": name, "set": s, "task": t, "quantity": k, "q": q, "value": x, "quarters": 1} for q, x in qv]
    return rows


def probe(g: Group, V: np.ndarray, name: str) -> list[dict]:
    """T4: ridge from V to each quantity, demeaned per quarter; trained on the training rows, R² per evaluation set."""
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


def sic_rows(g: Group) -> list[dict]:
    """SIC2 as the vector: the other firms of the same SIC2 are both its T1 peers and its T3 group."""
    acc = {}
    for q, pool_df in g.meta.groupby("quarter"):
        pool = pool_df.index.to_numpy()
        sic = g.sic[pool]
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
    return _mean_rows(acc, "SIC2")


# --- driver ---------------------------------------------------------------------------------------------------------

def group_key(run: str) -> tuple:
    c = json.loads((RUNS / run / "config.json").read_text())
    return (c.get("panels_dir"), c.get("panel"), c.get("t_in"), c.get("cutoff"), c.get("fingerprint", "zscore"),
            c.get("input_slog", "True"), c.get("sample", "all"))       # runs before 8 Nov: a second slog on the inputs


def run_all(runs: list[str], force: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [r for r in runs if force or not (RUNS / r / "usefulness.csv").exists()]
    groups = {}
    for r in todo:
        groups.setdefault(group_key(r), []).append(r)
    for key, rs in groups.items():
        t0 = time.time()
        g = Group(rs[0])
        bfile = OUT / f"baselines_{g.key()}.csv"
        old = pd.read_csv(bfile) if bfile.exists() and not force else None
        done = set() if old is None else set(old.vector)
        rows = sic_rows(g) if old is None else []
        for name, (full, holes) in g.baselines().items():
            if name not in done:                                       # baselines added later are scored once
                rows += score(g, full(), holes, name)
                print(f"[INFO]   baseline {name} ({time.time() - t0:.0f} s)", flush=True)
        if rows:
            pd.concat([old, pd.DataFrame(rows)] if old is not None else [pd.DataFrame(rows)]).to_csv(bfile, index=False)
        for r in rs:
            full, holes = g.model(r)
            res = pd.DataFrame(score(g, full(), holes, "model"))
            res.to_csv(RUNS / r / "usefulness.csv", index=False)
            print(f"[INFO]   {r} scored ({time.time() - t0:.0f} s)", flush=True)


# --- summary: the claims of the protocol ----------------------------------------------------------------------------

PRIMARY = ["out of time", "unseen firms, out of time"]
CONTEMP = ["T1 peers", "T3 groups", "T4 probe"]
GROWTH = ["revenue growth", "asset growth", "share issuance"]
CONFIG = ["lambda", "z", "t_in", "hidden", "cutoff", "panel", "pair_loss", "input_mask", "pillar1", "epochs", "train_frac"]
CELL = ["set", "task", "quantity"]
MAIN = {"lambda": 0.5, "z": 16, "t_in": 12, "hidden": 128, "cutoff": "2023Q3", "panel": "filled", "pair_loss": "mse",
        "input_mask": 0.0, "pillar1": "last", "epochs": 20, "train_frac": 1.0}
PCA_BOTH, RAW_FP, BOTH = "PCA-16 of fingerprint + window deltas", "raw fingerprint (32)", "fingerprint + window deltas"


def config_of(cfg: dict) -> dict:
    return {"lambda": 0.0 if cfg.get("loss") == "pillar1" else float(cfg["lambda2"]), "z": int(cfg["latent_dim"]),
            "t_in": int(cfg["t_in"]), "hidden": int(cfg["hidden"]), "cutoff": cfg["cutoff"], "panel": cfg["panel"],
            "seed": int(cfg["seed"]), "pair_loss": cfg.get("pair_loss", "mse"),
            "input_mask": float(cfg.get("input_mask", 0) or 0), "pillar1": cfg.get("pillar1", "last"),
            "epochs": int(cfg.get("epochs", 20)), "train_frac": float(cfg.get("train_frac", 1.0))}


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model rows of every scored run with its configuration and data group; baseline rows with their group."""
    rows = []
    for d in sorted(p for p in RUNS.iterdir() if (p / "usefulness.csv").exists()):
        cfg = json.loads((d / "config.json").read_text())
        group = f"{Path(cfg['panels_dir']).name}_{cfg['panel']}_tin{cfg['t_in']}_{cfg['cutoff']}" + \
            ("" if cfg.get("fingerprint", "zscore") == "zscore" else f"_{cfg['fingerprint']}")
        res = pd.read_csv(d / "results.csv").set_index(["set", "variant"])
        p1 = res.loc[("unseen firms, out of time", "model trained without macro")]
        u = pd.read_csv(d / "usefulness.csv")
        u = u[u.q.eq("all")] if "q" in u else u                        # per-quarter rows: paired tests only
        rows.append(u.assign(run=d.name, group=group, fit=p1.fit, fit2=p1.fit2, **config_of(cfg)))
    M = pd.concat(rows, ignore_index=True)
    B = pd.concat([pd.read_csv(f).assign(group=f.stem.replace("baselines_", "")) for f in OUT.glob("baselines_*.csv")],
                  ignore_index=True)
    B = B[B.q.fillna("all").eq("all")] if "q" in B else B
    return M, B


def verdict(diff: float, sd: float) -> str:
    bar = max(2 * (0.0 if pd.isna(sd) else sd), 0.002)
    return "win" if diff > bar else "loss" if diff < -bar else "tie"


def summary() -> None:
    M, B = collect()
    agg = M.groupby(CONFIG + ["group"] + CELL).value.agg(["mean", "std", "count"]).reset_index()
    pick = lambda **kw: agg[np.logical_and.reduce([agg[k] == v for k, v in {**MAIN, **kw}.items()])].set_index(CELL)
    main, struct = pick(), pick(**{"lambda": 0.0})
    Bg = B[B.group == main.group.iloc[0]].pivot_table(index=CELL, columns="vector", values="value")
    T = pd.DataFrame({"model λ 0.5": main["mean"], "sd": main["std"], "λ 0": struct["mean"], "sd λ 0": struct["std"]}).join(Bg)
    cols = ["model λ 0.5", "sd", "λ 0", "SIC2", RAW_FP, "window deltas", BOTH, "PCA-16 of fingerprint", PCA_BOTH,
            "PCA-16 of input window", "window-mean fingerprint (32)", "window-mean fingerprint + window deltas"]
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(f"Main configuration {MAIN}, seeds {int(main['count'].min())}\n")
    T = T.reindex(columns=cols)
    for s in SETS:
        print(f"== {s}")
        print(T.xs(s, level="set").round(3).to_string(na_rep="-"), "\n")
    T.to_csv(OUT / "summary_main.csv")

    def tally(a, b, sd, cells):
        v = [verdict(a[c] - b[c], sd[c]) for c in cells if c in a.index and c in b.index and pd.notna(a[c]) and pd.notna(b[c])]
        return {k: v.count(k) for k in ("win", "tie", "loss")}

    claims = []
    cells = lambda s, tasks, qs: [(s, t, q) for t in tasks for q in qs]
    m, sd = main["mean"], main["std"]
    for s in PRIMARY:
        c12 = cells(s, CONTEMP, QUANTITIES)
        for ref in (PCA_BOTH, RAW_FP, BOTH):
            claims.append({"claim": "C1" if ref != BOTH else "C5", "set": s, "against": ref,
                           **tally(m, Bg[ref], sd, c12)})
        sd2 = np.maximum(sd.fillna(0), struct["std"].reindex(sd.index).fillna(0))
        claims.append({"claim": "C2", "set": s, "against": "λ 0 (structure only)",
                       **tally(m, struct["mean"], sd2, cells(s, ["T1 peers", "T3 groups"], GROWTH))})
        claims.append({"claim": "C2 cost", "set": s, "against": "λ 0 on T2 industry",
                       **tally(m, struct["mean"], sd2, cells(s, ["T2 industry"], ["same SIC2 among 10 nearest", "majority SIC2 correct"]))})
        for t in ("T1 peers", "T3 groups"):
            claims.append({"claim": "C3", "set": s, "against": f"SIC2 ({t})", **tally(m, Bg["SIC2"], sd, cells(s, [t], QUANTITIES))})
        claims.append({"claim": "C4", "set": s, "against": RAW_FP + " (T5)",
                       **tally(m, Bg[RAW_FP], sd, cells(s, ["T5 holes"], ["10-NN overlap, 20 % of last quarter missing"]))})
    C = pd.DataFrame(claims)
    print("Claims (counts of cells: win / tie / loss of the model):")
    print(C.to_string(index=False), "\n")
    C.to_csv(OUT / "summary_claims.csv", index=False)

    # C2b: best λ per growth cell
    lam = agg[np.logical_and.reduce([agg[k] == v for k, v in MAIN.items() if k != "lambda"])]
    gcells = [c for s in PRIMARY for c in cells(s, ["T1 peers", "T3 groups"], GROWTH)]
    L = lam.set_index(CELL + ["lambda"])["mean"].unstack("lambda").reindex(gcells)
    L["best λ"] = L.idxmax(axis=1)
    print(f"C2b, growth cells by λ (lambdas present: {sorted(lam['lambda'].unique())}); interior best: "
          f"{int(L['best λ'].between(0.01, 0.99).sum())} of {len(L)}")
    print(L.round(4).to_string(), "\n")
    L.to_csv(OUT / "summary_lambda.csv")

    # the battery: every configuration against PCA-16 of fingerprint + deltas (its group) and its own λ 0
    rows = []
    for key, a in agg.groupby(CONFIG + ["group"]):
        a = a.set_index(CELL)
        b = B[B.group == key[-1]].pivot_table(index=CELL, columns="vector", values="value")
        cfg = dict(zip(CONFIG, key[:-1]))
        r = {**cfg, "seeds": int(a["count"].min())}
        ref = {32: "PCA-32 of fingerprint + window deltas", 64: "PCA-64 of fingerprint + window deltas"}.get(cfg["z"], PCA_BOTH)
        ref = ref if ref in b.columns else PCA_BOTH
        r["reference"] = ref.split(" of")[0]
        runs = M[np.logical_and.reduce([M[k] == v for k, v in cfg.items()])]
        r["fit"], r["fit2"] = runs.groupby("run").fit.first().mean(), runs.groupby("run").fit2.first().mean()
        for s in PRIMARY:
            c12 = [c for c in cells(s, CONTEMP, QUANTITIES) if c in a.index]
            r[f"mean gain vs PCA both, {s}"] = float(np.mean([a.loc[c, "mean"] - b.loc[c, ref] for c in c12]))
            t = tally(a["mean"], b[ref], a["std"], c12)
            r[f"W/T/L vs PCA both, {s}"] = f"{t['win']}/{t['tie']}/{t['loss']}"
            t = tally(a["mean"], b[RAW_FP], a["std"], c12)
            r[f"W/T/L vs raw fp, {s}"] = f"{t['win']}/{t['tie']}/{t['loss']}"
        if cfg["lambda"] != 0:
            z = agg[np.logical_and.reduce([agg[k] == v for k, v in {**cfg, "lambda": 0.0}.items()])].set_index(CELL)
            if len(z):
                gc = [c for s in PRIMARY for c in cells(s, ["T1 peers", "T3 groups"], GROWTH)]
                t = tally(a["mean"], z["mean"], np.maximum(a["std"].fillna(0), z["std"].reindex(a.index).fillna(0)), gc)
                r["growth W/T/L vs own λ 0 (12 cells)"] = f"{t['win']}/{t['tie']}/{t['loss']}"
        rows.append(r)
    R = pd.DataFrame(rows)
    print("Battery (fit, fit2: unseen firms out of time, from results.csv):")
    print(R.round(4).to_string(index=False))
    R.to_csv(OUT / "summary_battery.csv", index=False)

    # exploratory: one vector for every task? mean rank over the 15 contemporaneous cells of the main data group
    grp = main.group.iloc[0]
    a = agg[agg.group == grp]
    label = lambda r: (f"model λ {r['lambda']:g}, z {r.z}, h {r.hidden}" + (f", {r.pair_loss}" if r.pair_loss != "mse" else "")
                       + (f", mask {r.input_mask:g}" if r.input_mask else "")
                       + (", P1 window mean" if r.pillar1 != "last" else ""))
    V = pd.concat([a.assign(vector=a.apply(label, axis=1)).set_index(CELL + ["vector"])["mean"],
                   B[B.group == grp].set_index(CELL + ["vector"]).value]).unstack("vector")
    keep = [c for c in V.index if c[1] in CONTEMP + ["T2 industry", "T5 holes"] and c[2] not in FORWARD and c[0] in PRIMARY]
    ranks = V.loc[keep].rank(axis=1, ascending=False)
    MR = pd.DataFrame({s: ranks.xs(s, level="set").mean() for s in PRIMARY}).assign(cells=len(keep) // 2)
    MR["mean"] = MR[PRIMARY].mean(axis=1)
    print("\nExploratory: mean rank over the 15 contemporaneous cells (1 = best), main data group:")
    print(MR.sort_values("mean").round(2).to_string())
    MR.to_csv(OUT / "summary_rank.csv")


def confirm(cutoff: str = "2021Q3", panel: str = "filled") -> pd.DataFrame:
    """The confirmation claims of the protocol's addenda (C1′-C4‴), in the other test period (or on the complete
    panel: cutoff 2023Q3, panel "complete")."""
    M, B = collect()
    agg = M.groupby(CONFIG + ["group"] + CELL).value.agg(["mean", "std", "count"]).reset_index()

    def pick(**kw):
        a = agg[np.logical_and.reduce([agg[k] == v for k, v in {**MAIN, "cutoff": cutoff, "panel": panel, **kw}.items()])]
        return a.set_index(CELL), (a.group.iloc[0] if len(a) else None), (int(a["count"].min()) if len(a) else 0)

    def tally(a, b, sd, cells):
        v = [verdict(a[c] - b[c], sd[c]) for c in cells if c in a.index and c in b.index and pd.notna(a[c]) and pd.notna(b[c])]
        return v.count("win"), v.count("tie"), v.count("loss")

    rows = []
    for name, z, mask in [("C1′/C2′", 32, 0.0), ("C1″/C2″", 64, 0.0), ("C1‴/C2‴", 32, 0.2)]:
        (m, grp, n), (m0, _, n0) = pick(z=z, input_mask=mask, **{"lambda": 0.5}), pick(z=z, input_mask=mask, **{"lambda": 0.0})
        if grp is None:
            continue
        b = B[B.group == grp].pivot_table(index=CELL, columns="vector", values="value")
        ref = f"PCA-{z} of fingerprint + window deltas"
        for s in PRIMARY:
            c12 = [(s, t, q) for t in CONTEMP for q in QUANTITIES]
            g6 = [(s, t, q) for t in ("T1 peers", "T3 groups") for q in GROWTH]
            w1, w2 = tally(m["mean"], b[ref], m["std"], c12), tally(m["mean"], b[RAW_FP], m["std"], c12)
            r = {"claims": name, "z": z, "mask": mask, "set": s, "seeds": n,
                 f"vs PCA-{z} W/T/L": "/".join(map(str, w1)), "vs raw fp W/T/L": "/".join(map(str, w2)),
                 "C1 pass": w1[0] >= 8 and w2[0] >= 8}
            if n0:
                sd2 = np.maximum(m["std"].fillna(0), m0["std"].reindex(m.index).fillna(0))
                w3 = tally(m["mean"], m0["mean"], sd2, g6)
                r.update({"vs λ 0 growth W/T/L": "/".join(map(str, w3)), "C2 pass": w3[0] >= 4 and w3[2] == 0})
            if mask:
                (mu, _, nu) = pick(z=z, input_mask=0.0, **{"lambda": 0.5})
                if nu:
                    c5 = (s, "T5 holes", "10-NN overlap, 20 % of last quarter missing")
                    w4 = tally(m["mean"], mu["mean"], np.maximum(m["std"].fillna(0), mu["std"].reindex(m.index).fillna(0)), [c5])
                    r["C4 pass (T5 above unmasked)"] = w4[0] == 1
            rows.append(r)
    # C4″: masked z 64 in the main period
    (mm, grp, n), (mu, _, nu) = pick(z=64, input_mask=0.2, cutoff="2023Q3", **{"lambda": 0.5}), pick(z=64, cutoff="2023Q3", **{"lambda": 0.5})
    if n and nu and panel == "filled":
        for s in PRIMARY:
            c5 = (s, "T5 holes", "10-NN overlap, 20 % of last quarter missing")
            sd2 = np.maximum(mm["std"].fillna(0), mu["std"].reindex(mm.index).fillna(0))
            w = tally(mm["mean"], mu["mean"], sd2, [c5] + [(s, t, q) for t in CONTEMP for q in QUANTITIES])
            t5 = tally(mm["mean"], mu["mean"], sd2, [c5])
            rows.append({"claims": "C4″ (2023Q3)", "z": 64, "mask": 0.2, "set": s, "seeds": n,
                         "masked vs unmasked W/T/L (13 cells)": "/".join(map(str, w)),
                         "C4 pass (T5 up, ≤ 2 losses)": t5[0] == 1 and w[2] <= 2})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print(out.to_string(index=False))
    out.to_csv(OUT / f"summary_confirmation_{cutoff}_{panel}.csv", index=False)
    return out


def paired(runs: list[str], ref: str | list[str]) -> pd.DataFrame:
    """Per cell, the model (mean over the runs, e.g. its seeds) minus a reference, quarter by quarter: mean difference,
    t statistic across quarters, and the share of quarters where the model is ahead. ref: a baseline name (from the
    runs' group file) or the runs of another configuration. Needs the per-quarter rows (runs scored since 1 Nov
    23:00; baselines rescored with --force)."""
    def per_q(rs):
        f = lambda r: RUNS / r / ("usefulness_byq.csv" if (RUNS / r / "usefulness_byq.csv").exists() else "usefulness.csv")
        u = pd.concat([pd.read_csv(f(r)) for r in rs])
        u = u[u.q.ne("all")]
        return u.groupby(CELL + ["q"]).value.mean()
    a = per_q(runs)
    if isinstance(ref, str):
        g = group_key(runs[0])
        key = f"{Path(g[0]).name}_{g[1]}_tin{g[2]}_{g[3]}" + ("" if g[4] == "zscore" else f"_{g[4]}")
        bq = OUT / f"baselines_byq_{key}.csv"
        b = pd.read_csv(bq if bq.exists() else OUT / f"baselines_{key}.csv")
        b = b[(b.vector == ref) & b.q.fillna("all").ne("all")].set_index(CELL + ["q"]).value
    else:
        b = per_q(ref)
    d = (a - b).dropna().groupby(CELL)
    out = pd.DataFrame({"diff": d.mean(), "t": d.mean() / (d.std() / np.sqrt(d.count())), "quarters": d.count(),
                        "share ahead": d.apply(lambda x: (x > 0).mean())})
    return out


def score_by_quarter(runs: list[str], baselines: list[str]) -> None:
    """Per-quarter scores of runs (usefulness_byq.csv) and baselines (baselines_byq_<group>.csv) of one data group,
    for paired(); separate files, so the main ones are not rewritten."""
    g = Group(runs[0])
    rows = []
    for name, (full, holes) in g.baselines().items():
        if name in baselines:
            rows += score(g, full(), holes, name)
    pd.DataFrame(rows).to_csv(OUT / f"baselines_byq_{g.key()}.csv", index=False)
    for r in runs:
        full, holes = g.model(r)
        pd.DataFrame(score(g, full(), holes, "model")).to_csv(RUNS / r / "usefulness_byq.csv", index=False)


def neighbours(runs: dict[str, str], tickers: list[str], quarter: str, k: int = 5) -> pd.DataFrame:
    """Exploratory: the k nearest firms of each ticker in one quarter, by the raw fingerprint, the window deltas and
    each run's embedding ({label: run}, runs of the same data group); peer pool = all firms of the quarter."""
    g = Group(next(iter(runs.values())))
    sector = pd.read_parquet(g.args.panels_dir / "metadata.parquet").drop_duplicates("ticker").set_index("ticker").sector
    pool = np.flatnonzero(g.meta.quarter.eq(quarter).to_numpy())
    vecs = {"fingerprint": g.FP, "window deltas": g.D, **{lab: g.model(r)[0]() for lab, r in runs.items()}}
    rows = []
    for t in tickers:
        hit = np.flatnonzero(g.meta.ticker.iloc[pool].eq(t).to_numpy())
        if not len(hit):
            continue
        for lab, V in vecs.items():
            U = _unit(np.nan_to_num(V[pool]))
            s = U @ U[hit[0]]
            s[hit[0]] = -np.inf
            top = np.argsort(-s)[:k]
            rows.append({"ticker": t, "sector": sector.get(t, ""), "by": lab,
                         "nearest": "; ".join(f"{g.meta.ticker.iloc[pool[j]]} ({sector.get(g.meta.ticker.iloc[pool[j]], '?')[:28]})" for j in top)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*")
    p.add_argument("--force", action="store_true", help="Score again runs and baselines already scored.")
    p.add_argument("--summary", action="store_true", help="Print the tables and claims of the protocol.")
    p.add_argument("--confirm", action="store_true", help="Print the confirmation claims of the addenda (2021Q3).")
    args = p.parse_args()
    if args.confirm:
        confirm()
        print("\nComplete panel (cutoff 2023Q3):")
        confirm("2023Q3", "complete")
    elif args.summary:
        summary()
    else:
        run_all(args.runs, args.force)


if __name__ == "__main__":
    main()
