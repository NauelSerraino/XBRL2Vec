"""
Seed and data stability of the standard model, and the analyses behind the report of 4 Nov 2026
(docs/report_2026-11-04_stability.md; protocol: docs/usefulness_protocol.md, addendum 4). Standard model: z 32,
--input_mask 0.2, λ 0.5, window deltas, no macro (main_similarity.py); runs are found by their folder tags.

    cd src && ../xbrl2vec/bin/python stability_eval.py --part seeds     # agreement of the 10 nearest across seeds / samples
    ... --part claims    # S1-S5 from the runs' usefulness.csv, and the ensemble of 5 seeds (S3)
    ... --part t6        # T6: nowcasting a hidden item from the peers' moves (S4)
    ... --part curves    # peer R² vs k, group R² vs K (SIC 1-4 digits), peers kept vs share missing, persistence
    ... --part map       # t-SNE coordinates of one quarter (model and raw fingerprint)
    ... --part paper     # the ESWA paper's analyses for this model: geometry, input importance per pillar
    ... --part compare --families A B   # family A against family B on every held-out cell (per-seed and mean win)
    ... --part figures   # the figures, from the CSVs of the other parts (stability_figures.py)

Every part reads the same evaluation data as usefulness_eval.py (its Group: universe, focal sets, quantities, SIC).
Output: data/out/stability/*.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

import usefulness_eval as ue
from models.similarity_encoder import SimilarityEncoder

RUNS, OUT = ue.RUNS, ue.ms.DATA / "out" / "stability"
K_NN = ue.K_NN
PRIMARY = ue.PRIMARY
#: run families, by folder tag (the seed or draw is the last part of the tag)
FAMILIES = {
    "standard": r"_(X|S)_z32_mask02_lam05_s\d+$",
    "λ 0": r"_(X|S)_z32_mask02_lam0_s\d+$",
    "data 80 %": r"_D_z32_mask02_lam05_frac08_d\d+$",
    "unmasked": r"_U_z32_lam05_s\d+$",
    "mask 0.1": r"_R_z32_mask01_lam05_s\d+$",
    "mask 0.3": r"_(R|M)_z32_mask03_lam05_s\d+$",
    "mask 0.3, 2021Q3": r"_Q_z32_cut21_mask03_lam05_s\d+$",
    "decay 0.8": r"_E_z32_mask02_lam05_dec08_s\d+$",
    "decay 0.6": r"_E_z32_mask02_lam05_dec06_s\d+$",
    "decay 0.8, 2021Q3": r"_G_z32_cut21_mask02_lam05_dec08_s\d+$",
    "rank fingerprint": r"_F_z32_mask02_lam05_rank_s\d+$",
    "rank fingerprint, λ 0": r"_F_z32_mask02_lam0_rank_s\d+$",
    "rank fingerprint, decay 0.8": r"_H_z32_mask02_lam05_rank_dec08_s\d+$",
    "rank fingerprint, 2021Q3": r"_K_z32_cut21_mask02_lam05_rank_s\d+$",
    "hybrid": r"_W_z32_mask02_lam05_rank_p2z_dec08_s\d+$",
    "hybrid, 2021Q3": r"_V_z32_cut21_mask02_lam05_rank_p2z_dec08_s\d+$",
    "slog z-score": r"_Z_z32_mask02_lam05_slogz_dec08_s\d+$",
    "slog z-score, 2021Q3": r"_Z_z32_cut21_mask02_lam05_slogz_dec08_s\d+$",
    "slog z-score, λ 0": r"_Z_z32_mask02_lam0_slogz_s\d+$",
    "slog z-score, λ 0, 2021Q3": r"_Z_z32_cut21_mask02_lam0_slogz_s\d+$",
    "slog z-score, unmasked": r"_Z_z32_lam05_slogz_dec08_s\d+$",
    "slog z-score, unmasked, 2021Q3": r"_Z_z32_cut21_lam05_slogz_dec08_s\d+$",
    "slog z-score, decay 1": r"_Z_z32_mask02_lam05_slogz_s\d+$",
    "slog z-score, decay 1, 2021Q3": r"_Z_z32_cut21_mask02_lam05_slogz_s\d+$",
    "one slog": r"_N_z32_mask02_lam05_slogz_dec08_s\d+$",
    "one slog, 2021Q3": r"_N_z32_cut21_mask02_lam05_slogz_dec08_s\d+$",
    "one slog, λ 0": r"_N_z32_mask02_lam0_slogz_s\d+$",
    "one slog, λ 0, 2021Q3": r"_N_z32_cut21_mask02_lam0_slogz_s\d+$",
    "one slog, unmasked": r"_N_z32_lam05_slogz_dec08_s\d+$",
    "one slog, unmasked, 2021Q3": r"_N_z32_cut21_lam05_slogz_dec08_s\d+$",
    "one slog, decay 1": r"_N_z32_mask02_lam05_slogz_s\d+$",
    "one slog, decay 1, 2021Q3": r"_N_z32_cut21_mask02_lam05_slogz_s\d+$",
    "balanced": r"_B_z32_mask02_lam05_slogz_dec08_s\d+$",            # --sample balanced, cutoff 2022Q4 (14 Nov)
    "balanced, unmasked": r"_B_z32_lam05_slogz_dec08_s\d+$",
    "balanced, decay 1": r"_B_z32_mask02_lam05_slogz_s\d+$",
    "rank fingerprint, λ 0, 2021Q3": r"_K_z32_cut21_mask02_lam0_rank_s\d+$",
    "mask 0.4": r"_R_z32_mask04_lam05_s\d+$",
    "λ 0.25": r"_R_z32_mask02_lam025_s\d+$",
    "λ 0.75": r"_R_z32_mask02_lam075_s\d+$",
    "40 epochs": r"_R_z32_mask02_lam05_ep40_s\d+$",
    "z 64": r"_X_z64_mask02_lam05_s\d+$",
    "z 64, λ 0": r"_X_z64_mask02_lam0_s\d+$",
    "standard, 2021Q3": r"_(C|P)_z32_cut21_mask02_lam05_s\d+$",
    "λ 0, 2021Q3": r"_(C|P)_z32_cut21_mask02_lam0_s\d+$",
}
ENSEMBLE_SEEDS = [42, 1, 2, 3, 4]
RAW_FP, BOTH, WMEAN, PCA32 = ue.RAW_FP, ue.BOTH, "window-mean fingerprint (32)", "PCA-32 of fingerprint + window deltas"


def seed_of(run: str) -> int:
    return int(re.search(r"_[sd](\d+)$", run).group(1))


def runs_of(family: str, scored: bool = False) -> list[str]:
    """The finished runs of a family, seed 42 first, then by seed."""
    rx = re.compile(FAMILIES[family])
    rs = [d.name for d in RUNS.iterdir() if rx.search(d.name) and (d / "results.csv").exists()
          and (d / "encoder_no_macro.pth").exists() and (not scored or (d / "usefulness.csv").exists())]
    return sorted(rs, key=lambda r: (seed_of(r) != 42, seed_of(r)))


_groups: dict = {}


def group(run: str) -> ue.Group:
    key = ue.group_key(run)
    if key not in _groups:
        _groups[key] = ue.Group(run)
    return _groups[key]


# --- vectors: every row of the group, and focal rows with given holes -------------------------------------------------

def holed_X(g: ue.Group, rows: np.ndarray, holes: np.ndarray) -> torch.Tensor:
    """Input windows of `rows` with the fingerprint columns `holes` [len(rows), k] of the last quarter set to 0."""
    X = g.X[rows].clone()
    last = X[:, -1, :]
    last.scatter_(1, torch.as_tensor(g.xcol[holes]), 0.0)
    X[:, -1, :] = last
    return X


def holed_raw(g: ue.Group, rows: np.ndarray, holes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fingerprint and window deltas of `rows` with the ratios `holes` of the last quarter set to 0."""
    FP, D = g.FP[rows].copy(), g.D[rows].copy()
    K = FP.shape[1]
    before = FP - D[:, -K:]
    np.put_along_axis(FP, holes, 0.0, axis=1)
    D[:, -K:] = FP - before
    return FP, D


def encoder(g: ue.Group, run: str):
    cfg = json.loads((RUNS / run / "config.json").read_text())
    m = SimilarityEncoder(g.args.t_in, g.X.shape[-1], g.sets["train firms"].X_macro.shape[-1], int(cfg["latent_dim"]),
                          use_macro=False, hidden=int(cfg["hidden"]))
    m.load_state_dict(torch.load(RUNS / run / "encoder_no_macro.pth", map_location="cpu"))
    m.eval()

    @torch.no_grad()
    def emb(X: torch.Tensor) -> np.ndarray:
        return np.concatenate([m(X[i:i + 8192], None)[0].numpy() for i in range(0, len(X), 8192)])
    return emb


class Vec:
    """A vector of every row (full) and of focal rows with holes (holed(rows, holes))."""

    def __init__(self, name, full, holed):
        self.name, self._full, self.holed, self._cache = name, full, holed, None

    def full(self) -> np.ndarray:
        if self._cache is None:
            self._cache = self._full()
        return self._cache


def model_vec(g: ue.Group, run: str, name: str | None = None) -> Vec:
    emb = encoder(g, run)
    return Vec(name or run, lambda: emb(g.X), lambda r, h: emb(holed_X(g, r, h)))


def ensemble_vec(g: ue.Group, runs: list[str], name: str) -> Vec:
    """Concatenated unit embeddings of several runs: its cosine is the mean of the runs' cosines."""
    embs = [encoder(g, r) for r in runs]
    cat = lambda Zs: np.concatenate([ue._unit(Z) for Z in Zs], axis=1) / np.sqrt(len(Zs))
    return Vec(name, lambda: cat([e(g.X) for e in embs]), lambda r, h: cat([e(holed_X(g, r, h)) for e in embs]))


def baseline_vecs(g: ue.Group) -> dict[str, Vec]:
    p32 = g.pca_both_k[32]
    return {RAW_FP: Vec(RAW_FP, lambda: g.FP, lambda r, h: holed_raw(g, r, h)[0]),
            BOTH: Vec(BOTH, lambda: g.both(g.FP, g.D), lambda r, h: g.both(*holed_raw(g, r, h))),
            WMEAN: Vec(WMEAN, lambda: g.window_mean(g.X), lambda r, h: g.window_mean(holed_X(g, r, h))),
            PCA32: Vec(PCA32, lambda: p32.transform(g.both(g.FP, g.D)), lambda r, h: p32.transform(g.both(*holed_raw(g, r, h))))}


def random_holes(n: int, K: int, k: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, K)).argsort(axis=1)[:, :k]


# --- neighbours ---------------------------------------------------------------------------------------------------

def oot_quarters(g: ue.Group) -> list[str]:
    return sorted(q for q in g.meta.quarter.unique() if q > g.cutoff)


def pool_of(g: ue.Group, q: str) -> np.ndarray:
    return np.flatnonzero(g.meta.quarter.eq(q).to_numpy())


def knn(U: np.ndarray, k: int = K_NN) -> np.ndarray:
    """Positions of the k nearest (cosine) of every row of U among the others, nearest first."""
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S, nn, axis=1), axis=1)
    return np.take_along_axis(nn, order, axis=1)


def knn_holed(Uh: np.ndarray, U: np.ndarray, own: np.ndarray, k: int = K_NN) -> np.ndarray:
    """k nearest in the clean pool U of the holed rows Uh, whose own clean positions are `own`."""
    S = Uh @ U.T
    S[np.arange(len(Uh)), own] = -np.inf
    return np.argpartition(-S, k, axis=1)[:, :k]


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row, the share of a's neighbours that are also b's."""
    return (a[:, :, None] == b[:, None, :]).any(axis=2).mean(axis=1)


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    A, B = A - A.mean(0), B - B.mean(0)
    return float(np.linalg.norm(B.T @ A) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


# --- part: seeds --------------------------------------------------------------------------------------------------

def part_seeds() -> None:
    """Agreement of the 10 nearest firms (out-of-time quarters, the two primary sets) between: seeds of the standard
    model (same data, different initialisation), runs on different 80 % samples of the training firms (same
    initialisation, seed 42), the λ 0 seeds; each standard seed and the baselines; and, as a yardstick, the raw
    fingerprint with 20 % of the last quarter missing (T5). Also linear CKA and the Spearman of pair cosines."""
    std, lam0, data, z64 = runs_of("standard"), runs_of("λ 0"), runs_of("data 80 %"), runs_of("z 64")
    g = group(std[0])
    V = {r: ue._unit(model_vec(g, r).full()) for r in std + lam0 + data + z64}
    B = {n: ue._unit(np.nan_to_num(v.full())) for n, v in baseline_vecs(g).items()}
    pairs = ([("init seeds", a, b) for i, a in enumerate(std) for b in std[i + 1:]]
             + [("80 % samples", a, b) for i, a in enumerate(data) for b in data[i + 1:]]
             + [("80 % sample vs all firms", d, std[0]) for d in data]
             + [("λ 0 seeds", a, b) for i, a in enumerate(lam0) for b in lam0[i + 1:]]
             + [("z 64 seeds", a, b) for i, a in enumerate(z64) for b in z64[i + 1:]]
             + [("λ 0.5 vs λ 0, same seed", a, b) for a in std for b in lam0 if seed_of(a) == seed_of(b)]
             + [(f"model vs {n}", a, n) for a in std for n in B])
    # ensembles of m seeds (cosines averaged), disjoint seeds: agreement against ensemble size
    for m in (2, 3, 5):
        groups = [std[i * m:(i + 1) * m] for i in range(len(std) // m)]
        for i in range(0, len(groups) - 1, 2):
            a, b = f"ensemble {m}: " + "+".join(map(str, map(seed_of, groups[i]))), \
                f"ensemble {m}: " + "+".join(map(str, map(seed_of, groups[i + 1])))
            for name, rs in ((a, groups[i]), (b, groups[i + 1])):
                V[name] = ue._unit(np.concatenate([V[r] for r in rs], axis=1))
            pairs.append((f"ensemble of {m} seeds vs another", a, b))
    vec = lambda name: V[name] if name in V else B[name]
    rng = np.random.default_rng(0)
    rows, firm_rows = [], []
    for q in oot_quarters(g):
        pool = pool_of(g, q)
        nn20 = {n: knn(vec(n)[pool], 2 * K_NN) for n in {x for _, a, b in pairs for x in (a, b)}}
        nn = {n: v[:, :K_NN] for n, v in nn20.items()}
        # yardstick: the raw fingerprint with the T5 holes
        holes = g.holes[pool]
        Uh = ue._unit(holed_raw(g, pool, holes)[0])
        nn_holes = knn_holed(Uh, B[RAW_FP][pool], np.arange(len(pool)))
        sample = rng.choice(len(pool), min(400, len(pool)), replace=False)
        iu = np.triu_indices(len(sample), 1)
        cos = {n: (vec(n)[pool][sample] @ vec(n)[pool][sample].T)[iu] for n in nn}
        for s in PRIMARY:
            f = g.focal[s][pool]
            if f.sum() < 50:
                continue
            for kind, a, b in pairs:
                rows.append({"quarter": q, "set": s, "kind": kind, "a": a, "b": b,
                             "overlap": float(overlap(nn[a][f], nn[b][f]).mean()),
                             "within 20": float(overlap(nn[a][f], nn20[b][f]).mean()),   # a's 10 among b's 20 nearest
                             "rsa": float(spearmanr(cos[a], cos[b])[0])})
            rows.append({"quarter": q, "set": s, "kind": "raw fingerprint vs itself with 20 % holes", "a": RAW_FP,
                         "b": RAW_FP + " (holes)", "overlap": float(overlap(nn_holes[f], nn[RAW_FP][f]).mean()),
                         "rsa": np.nan})
            # per firm: mean agreement over the pairs of standard seeds
            per = np.mean([overlap(nn[a][f], nn[b][f]) for kind, a, b in pairs if kind == "init seeds"], axis=0)
            firm_rows.append(pd.DataFrame({"quarter": q, "set": s, "row": pool[f], "seed agreement": per}))
        print(f"[INFO] seeds {q} done", flush=True)
    P = pd.DataFrame(rows)
    cka = []
    oot = np.flatnonzero(g.meta.quarter.gt(g.cutoff).to_numpy())
    smp = rng.choice(oot, min(8000, len(oot)), replace=False)
    for kind, a, b in pairs:
        cka.append({"kind": kind, "a": a, "b": b, "cka": linear_cka(vec(a)[smp], vec(b)[smp])})
    F = pd.concat(firm_rows, ignore_index=True)
    F = F.assign(ticker=g.meta.ticker.to_numpy()[F.row], sic=g.sic[F.row],
                 size=g.Q["size"].to_numpy()[F.row]).drop(columns="row")
    OUT.mkdir(parents=True, exist_ok=True)
    P.to_csv(OUT / "seed_pairs.csv", index=False)
    pd.DataFrame(cka).to_csv(OUT / "seed_cka.csv", index=False)
    F.to_csv(OUT / "seed_agreement_by_firm.csv", index=False)
    print(P.groupby(["set", "kind"])[["overlap", "within 20", "rsa"]].mean().round(3).to_string())
    print(pd.DataFrame(cka).groupby("kind").cka.mean().round(3).to_string())


# --- part: claims -------------------------------------------------------------------------------------------------

def cells_of(runs: list[str]) -> pd.DataFrame:
    """run x cell (set, task, quantity) of the runs' usefulness.csv (means over the quarters)."""
    out = {}
    for r in runs:
        u = pd.read_csv(RUNS / r / "usefulness.csv")
        u = u[u.q.eq("all")] if "q" in u else u
        out[r] = u[u.vector.eq("model")].set_index(ue.CELL).value
    return pd.DataFrame(out).T


def baselines_of(run: str) -> pd.DataFrame:
    g = ue.group_key(run)
    key = f"{Path(g[0]).name}_{g[1]}_tin{g[2]}_{g[3]}" + ("" if g[4] == "zscore" else f"_{g[4]}") + \
        ("" if g[5] == "True" else "_inputF") + ("" if g[6] == "all" else f"_{g[6]}")   # as usefulness_eval's Group.key
    b = pd.read_csv(ue.OUT / f"baselines_{key}.csv")
    b = b[b.q.fillna("all").eq("all")] if "q" in b else b
    return b.pivot_table(index=ue.CELL, columns="vector", values="value")


def tally(diff: pd.Series, bar: pd.Series, cells: list) -> dict:
    v = ["win" if diff[c] > bar[c] else "loss" if diff[c] < -bar[c] else "tie"
         for c in cells if c in diff.index and pd.notna(diff[c])]
    return {k: v.count(k) for k in ("win", "tie", "loss")}


def part_claims(period: str = "2023Q3") -> None:
    """S1-S3 and S5 (addendum 4) for the period's standard model and its λ 0 counterpart; 2021Q3: descriptive."""
    sfx = "" if period == "2023Q3" else f", {period}"
    std, lam0 = runs_of("standard" + sfx, scored=True), runs_of("λ 0" + sfx, scored=True)
    tag = "" if period == "2023Q3" else f"_{period}"
    A, L, Bs = cells_of(std), cells_of(lam0), baselines_of(std[0])
    ma, sa, na = A.mean(), A.std(), len(A)
    ml, sl, nl = L.mean(), L.std(), len(L)
    c12 = lambda s: [(s, t, q) for t in ue.CONTEMP for q in ue.QUANTITIES]
    c6 = lambda s: [(s, t, q) for t in ("T1 peers", "T3 groups") for q in ue.GROWTH]
    rows = []
    for s in PRIMARY:
        for ref in (RAW_FP, PCA32):
            d = ma - Bs[ref]
            rows.append({"claim": "S1a", "set": s, "against": ref, "seeds": na,
                         **tally(d, np.maximum(2 * sa, 0.002), c12(s))})
            rows.append({"claim": "S1b", "set": s, "against": ref, "seeds": na, **tally(d, 2 * sa / np.sqrt(na), c12(s))})
        d = ma - ml
        rows.append({"claim": "S2a", "set": s, "against": "λ 0", "seeds": min(na, nl),
                     **tally(d, np.maximum(2 * np.maximum(sa, sl), 0.002), c6(s))})
        rows.append({"claim": "S2b", "set": s, "against": "λ 0", "seeds": min(na, nl),
                     **tally(d, 2 * np.sqrt(sa ** 2 / na + sl ** 2 / nl), c6(s))})
    # S3: the ensemble of 5 seeds against the best of those seeds
    ens_runs = [r for r in std if seed_of(r) in ENSEMBLE_SEEDS]
    if len(ens_runs) == len(ENSEMBLE_SEEDS):
        g = group(ens_runs[0])
        res = {}
        for name, rs in [("ensemble of 5 seeds", ens_runs), ("ensemble of 10 seeds", std[:10])]:
            v = ensemble_vec(g, rs, name)
            sc = pd.DataFrame(ue.score(g, v.full(), lambda r: v.holed(r, g.holes[r]), name))
            res[name] = sc[sc.q.eq("all")].set_index(ue.CELL).value
        E = pd.DataFrame(res)
        E.to_csv(OUT / f"ensemble_cells{tag}.csv")
        best = A.loc[ens_runs].max()
        for s in PRIMARY:
            cs = [c for c in c12(s) if c in E.index]
            rows.append({"claim": "S3", "set": s, "against": "best of the 5 seeds", "seeds": 5,
                         "win": int(sum(E.loc[c, "ensemble of 5 seeds"] >= best[c] for c in cs)), "tie": 0,
                         "loss": int(sum(E.loc[c, "ensemble of 5 seeds"] < best[c] for c in cs))})
    # S5: no baseline above the model on both T5 and T1 asset growth
    t5, ag = "T5 holes", ("T1 peers", "asset growth")
    for s in PRIMARY:
        h = (s, t5, "10-NN overlap, 20 % of last quarter missing")
        a = (s, *ag)
        above = [b for b in Bs.columns if pd.notna(Bs[b].get(h)) and Bs[b][h] > ma[h] and pd.notna(Bs[b].get(a)) and Bs[b][a] > ma[a]]
        rows.append({"claim": "S5", "set": s, "against": "every baseline", "seeds": na,
                     "win": int(not above), "tie": 0, "loss": len(above), "note": ", ".join(above)})
    C = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUT / f"claims{tag}.csv", index=False)
    T = pd.DataFrame({"standard": ma, "sd": sa, "λ 0": ml, "sd λ 0": sl}).join(Bs[[RAW_FP, PCA32, WMEAN, "SIC2"]])
    T.to_csv(OUT / f"cells_standard{tag}.csv")
    A.to_csv(OUT / f"cells_by_seed{tag}.csv")
    L.to_csv(OUT / f"cells_by_seed_lambda0{tag}.csv")
    print(C.to_string(index=False))
    # paired over quarters, every seed averaged
    P = []
    for ref, name in [(RAW_FP, RAW_FP), (PCA32, PCA32), (lam0, "λ 0")]:
        p = ue.paired(std, ref).reset_index().assign(against=name)
        P.append(p)
    P = pd.concat(P, ignore_index=True)
    P.to_csv(OUT / f"paired{tag}.csv", index=False)
    P = P[P.set.isin(PRIMARY) & ~P.quantity.str.contains("next year")]
    print(P.assign(ahead=P.t > 2, behind=P.t < -2).groupby(["against", "set"])[["ahead", "behind"]].sum().to_string())


# --- part: T6 -----------------------------------------------------------------------------------------------------

def observed(g: ue.Group, lines: list[str], lag: int = 1) -> dict[str, np.ndarray]:
    """line -> [rows, 2] observed at the window's last quarter t and at t - lag (mask codes of usefulness_eval)."""
    panel = g.args.panels_dir / g.args.panel
    mask = pd.read_parquet(panel / "mask.parquet")
    quarters = [c for c in mask.columns if len(str(c)) == 6 and str(c)[4] == "Q"]
    prev = {q: quarters[i - lag] for i, q in enumerate(quarters) if i >= lag}
    out = {}
    for line in lines:
        m = mask[mask.features == line].drop_duplicates("ticker").set_index("ticker")[quarters].isin(ue.GOOD_CELLS)
        st = m.stack()
        key_t = pd.MultiIndex.from_arrays([g.meta.ticker, g.meta.quarter])
        key_p = pd.MultiIndex.from_arrays([g.meta.ticker, g.meta.quarter.map(prev)])
        out[line] = np.stack([st.reindex(key_t).fillna(False).to_numpy(bool),
                              st.reindex(key_p).fillna(False).to_numpy(bool)], axis=1)
    return out


def part_t6(max_seeds: int = 10, lag: int = 1, families: tuple = ("standard", "λ 0")) -> None:
    """For every focal firm-quarter of the primary sets and every ratio j: j of quarter t hidden (0), the 10 nearest
    firms found with the hole, the firm's change of j from t - 1 to t predicted by its peers' mean change (peers with j
    observed at both quarters, >= 3 of them). R² per quarter and ratio, averaged over the quarters and over the ratios
    of each statement. Baselines: the raw vectors with the same hole, SIC2 peers, random peers; and the no-change
    forecast (skill = 1 - MSE / MSE of no change). lag 4 (exploratory, not pre-registered): the year-on-year change,
    the horizon of the Pillar 2 target."""
    runs = {f: runs_of(f)[:max_seeds] for f in families}
    std, lam0 = runs.get("standard", []), runs.get("λ 0", [])
    g = group(next(iter(runs.values()))[0])
    ratio_cols = [c for c in pd.read_parquet(g.args.panels_dir / g.args.panel / "targets" / "fingerprint.parquet",
                                             columns=None).columns if c not in ("ticker", "quarter")]
    mask = pd.read_parquet(g.args.panels_dir / g.args.panel / "mask.parquet", columns=["stmt", "features"])
    stmt = mask.drop_duplicates("features").set_index("features").stmt.to_dict()
    obs = observed(g, ratio_cols, lag)
    inv = lambda X: (torch.sign(X) * torch.expm1(torch.abs(X)) if g.input_slog else X).numpy()   # X = F since 8 Nov
    v_t = inv(g.X[:, -1, :])[:, g.xcol]
    v_p = inv(g.X[:, -1 - lag, :])[:, g.xcol]
    dv = v_t - v_p                                                     # change of every ratio from t - lag to t
    vecs = {**{r: model_vec(g, r) for rs in runs.values() for r in rs}, **baseline_vecs(g)}
    fam = {**{r: f for f, rs in runs.items() for r in rs}, **{n: n for n in baseline_vecs(g)}}
    U = {n: ue._unit(np.nan_to_num(v.full())) for n, v in vecs.items()}
    quarters = oot_quarters(g)
    rng = np.random.default_rng(0)
    rec = []
    t0 = time.time()
    for j, col in enumerate(ratio_cols):
        ok_j = obs[col].all(axis=1)
        focal_all = np.flatnonzero((g.focal[PRIMARY[0]] | g.focal[PRIMARY[1]]) & g.meta.quarter.gt(g.cutoff).to_numpy() & ok_j)
        hol = {n: ue._unit(np.nan_to_num(v.holed(focal_all, np.full((len(focal_all), 1), j)))) for n, v in vecs.items()}
        pos_in_focal = {r: i for i, r in enumerate(focal_all)}
        for q in quarters:
            pool = pool_of(g, q)
            at = {r: i for i, r in enumerate(pool)}
            fr = focal_all[g.meta.quarter.to_numpy()[focal_all] == q]
            if len(fr) < 50:
                continue
            own = np.array([at[r] for r in fr])
            y = dv[fr, j]
            peer_ok = ok_j[pool]
            peer_dv = np.where(peer_ok, dv[pool, j], np.nan)
            preds = {}
            for n in vecs:
                nn = knn_holed(hol[n][[pos_in_focal[r] for r in fr]], U[n][pool], own)
                preds[n] = ue._peer_mean(peer_dv, nn)
            sic = g.sic[pool]
            preds["SIC2"] = ue._group_mean(peer_dv, sic)[own]
            rnd = np.array([rng.choice(np.delete(np.arange(len(pool)), o), K_NN, replace=False) for o in own])
            preds["random peers"] = ue._peer_mean(peer_dv, rnd)
            for s in PRIMARY:
                f = g.focal[s][fr]
                if f.sum() < 50:
                    continue
                for n, p in preds.items():
                    ok = np.isfinite(p[f]) & np.isfinite(y[f])
                    mse0 = np.mean(y[f][ok] ** 2)
                    rec.append({"ratio": col, "statement": stmt.get(col, "?"), "quarter": q, "set": s, "vector": n,
                                "family": fam.get(n, n), "r2": ue._r2(y[f], p[f]),
                                "skill": 1 - np.mean((y[f][ok] - p[f][ok]) ** 2) / mse0 if ok.sum() and mse0 > 0 else np.nan,
                                "n": int(ok.sum())})
        print(f"[INFO] T6 {col} done ({time.time() - t0:.0f} s)", flush=True)
    R = pd.DataFrame(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = ("" if lag == 1 else f"_lag{lag}") + ("" if tuple(families) == ("standard", "λ 0") else
                                               "_" + "_".join(f.replace(" ", "").replace(",", "") for f in families))
    R.to_csv(OUT / f"t6_rows{tag}.csv", index=False)
    per_vec = R.groupby(["set", "statement", "vector", "family"])[["r2", "skill"]].mean().reset_index()
    per_fam = per_vec.groupby(["set", "statement", "family"]).agg(r2=("r2", "mean"), sd=("r2", "std"), runs=("r2", "size"),
                                                                 skill=("skill", "mean")).reset_index()
    per_fam.to_csv(OUT / f"t6_summary{tag}.csv", index=False)
    print(per_fam.round(4).to_string(index=False))
    if not (std and lam0):
        return
    # S4: the standard model against the raw fingerprint and SIC2 (S4a), and against λ 0 (S4b)
    piv = per_fam.set_index(["set", "statement", "family"])[["r2", "sd"]].unstack("family")
    rows = []
    for ref in (RAW_FP, "SIC2", "λ 0"):
        d = piv["r2"]["standard"] - piv["r2"][ref]
        sd = piv["sd"]["standard"].fillna(0)
        if ref == "λ 0":
            sd = np.maximum(sd, piv["sd"]["λ 0"].fillna(0))
        bar = np.maximum(2 * sd, 0.002)
        v = np.where(d > bar, "win", np.where(d < -bar, "loss", "tie"))
        rows.append({"claim": "S4b" if ref == "λ 0" else "S4a", "against": ref,
                     **{k: int((v == k).sum()) for k in ("win", "tie", "loss")}})
    pd.DataFrame(rows).to_csv(OUT / f"t6_claims{tag}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


# --- part: curves -------------------------------------------------------------------------------------------------

SIC_DIGITS = {"SIC 1-digit": 1000, "SIC2": 100, "SIC3": 10, "SIC4": 1}


def sic_full(g: ue.Group) -> np.ndarray:
    meta = ue.ms.DATA / "in" / g.args.panels_dir.name.replace("panels_", "download_") / "metadata.parquet"
    m = pd.read_parquet(meta).drop_duplicates("ticker").set_index("ticker")
    return g.meta.ticker.map(pd.to_numeric(m.sic, errors="coerce")).fillna(-1).astype(int).to_numpy()


def part_curves(n_seeds: int = 3) -> None:
    std, lam0, unm = runs_of("standard")[:n_seeds], runs_of("λ 0")[:n_seeds], runs_of("unmasked")[:n_seeds]
    masks = [r for f in ("mask 0.1", "mask 0.3", "mask 0.4") for r in runs_of(f)[:n_seeds]]
    g = group(std[0])
    fam = {**{r: "standard" for r in std}, **{r: "λ 0" for r in lam0}, **{r: "unmasked" for r in unm},
           **{r: f for f in ("mask 0.1", "mask 0.3", "mask 0.4") for r in runs_of(f)[:n_seeds]}}
    vecs = {**{r: model_vec(g, r) for r in std + lam0 + unm + masks}, **baseline_vecs(g)}
    U = {n: ue._unit(np.nan_to_num(v.full())) for n, v in vecs.items()}
    quarters = oot_quarters(g)
    sicf = sic_full(g)
    ks = [1, 2, 3, 5, 10, 20, 50, 100]
    Ks = [10, 25, 50, 100, 200, 400]
    shares = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    rk, rK, rh = [], [], []
    for q in quarters:
        pool = pool_of(g, q)
        Y = g.Q.iloc[pool]
        sets = {s: g.focal[s][pool] for s in PRIMARY if g.focal[s][pool].sum() >= 50}
        for n in vecs:
            S = U[n][pool] @ U[n][pool].T
            np.fill_diagonal(S, -np.inf)
            top = np.argpartition(-S, max(ks), axis=1)[:, :max(ks)]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, axis=1), axis=1), axis=1)
            for k in ks:
                for qty in ue.QUANTITIES:
                    p = ue._peer_mean(Y[qty].to_numpy(float), top[:, :k]) if k >= ue.MIN_PEERS else \
                        np.nanmean(Y[qty].to_numpy(float)[top[:, :k]], axis=1)
                    for s, f in sets.items():
                        rk.append({"quarter": q, "set": s, "vector": n, "family": fam.get(n, n), "k": k,
                                   "quantity": qty, "r2": ue._r2(Y[qty].to_numpy(float)[f], p[f])})
            if n in std[:1] + lam0[:1] + list(baseline_vecs(g)):           # group curves: one seed per model
                for K in Ks:
                    lab = KMeans(K, n_init=1, max_iter=100, random_state=0).fit_predict(U[n][pool])
                    for qty in ue.QUANTITIES:
                        for s, f in sets.items():
                            rK.append({"quarter": q, "set": s, "vector": n, "family": fam.get(n, n), "K": K,
                                       "quantity": qty, "r2": ue._r2(Y[qty].to_numpy(float)[f], ue._group_mean(Y[qty].to_numpy(float), lab)[f])})
            # holes curve
            for sh in shares:
                k = max(1, round(sh * g.FP.shape[1]))
                holes = random_holes(len(pool), g.FP.shape[1], k, seed=1)
                Uh = ue._unit(np.nan_to_num(vecs[n].holed(pool, holes)))
                nnh = knn_holed(Uh, U[n][pool], np.arange(len(pool)))
                for s, f in sets.items():
                    rh.append({"quarter": q, "set": s, "vector": n, "family": fam.get(n, n), "share": sh,
                               "overlap": float(overlap(nnh[f], top[f, :K_NN]).mean())})
        for name, div in SIC_DIGITS.items():                           # industry groups at 1-4 digits
            code = np.where(sicf[pool] >= 0, sicf[pool] // div, -1)
            for qty in ue.QUANTITIES:
                y = Y[qty].to_numpy(float)
                for s, f in sets.items():
                    rK.append({"quarter": q, "set": s, "vector": name, "family": name, "K": len(np.unique(code[code >= 0])),
                               "quantity": qty, "r2": ue._r2(y[f], ue._group_mean(y, code)[f])})
        for qty in ue.QUANTITIES:                                      # SIC2 peers: a flat line on the k curve
            p = ue._group_mean(Y[qty].to_numpy(float), g.sic[pool])
            for s, f in sets.items():
                rk.append({"quarter": q, "set": s, "vector": "SIC2", "family": "SIC2", "k": np.nan, "quantity": qty,
                           "r2": ue._r2(Y[qty].to_numpy(float)[f], p[f])})
        print(f"[INFO] curves {q} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rk).to_csv(OUT / "curve_k.csv", index=False)
    pd.DataFrame(rK).to_csv(OUT / "curve_groups.csv", index=False)
    pd.DataFrame(rh).to_csv(OUT / "curve_holes.csv", index=False)
    persistence(g, {n: U[n] for n in std[:1] + lam0[:1] + list(baseline_vecs(g))}, fam)


def persistence(g: ue.Group, U: dict, fam: dict, hs=(1, 2, 4, 8)) -> None:
    """Share of a firm's 10 nearest at t (tickers) still among its 10 nearest at t + h; focal: the unseen firms, every
    quarter (no model was trained on them)."""
    quarters = sorted(g.meta.quarter.unique())
    nn_t = {}
    for q in quarters:
        pool = pool_of(g, q)
        tick = g.meta.ticker.to_numpy()[pool]
        unseen = g.meta.firms.to_numpy()[pool] == "unseen"
        for n, V in U.items():
            nn = knn(V[pool])
            nn_t[(n, q)] = {t: set(tick[nn[i]]) for i, t in enumerate(tick) if unseen[i]}
    rows = []
    for h in hs:
        for i, q in enumerate(quarters[:-h]):
            q2 = quarters[i + h]
            for n in U:
                a, b = nn_t[(n, q)], nn_t[(n, q2)]
                common = [t for t in a if t in b]
                if len(common) >= 50:
                    rows.append({"quarter": q, "h": h, "vector": n, "family": fam.get(n, n),
                                 "persistence": float(np.mean([len(a[t] & b[t]) / K_NN for t in common]))})
    pd.DataFrame(rows).to_csv(OUT / "persistence.csv", index=False)
    print(pd.DataFrame(rows).groupby(["family", "h"]).persistence.mean().unstack().round(3).to_string())


def part_map(quarter: str = "2024Q4") -> None:
    from sklearn.manifold import TSNE
    std = runs_of("standard")
    g = group(std[0])
    pool = pool_of(g, quarter)
    sicf = sic_full(g)
    rows = []
    for name, V in (("standard", model_vec(g, std[0]).full()), (RAW_FP, g.FP)):
        xy = TSNE(2, perplexity=30, init="pca", metric="cosine", random_state=0).fit_transform(ue._unit(V[pool]))
        rows.append(pd.DataFrame({"vector": name, "quarter": quarter, "ticker": g.meta.ticker.to_numpy()[pool],
                                  "sic": sicf[pool], "x": xy[:, 0], "y": xy[:, 1]}))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(OUT / "map_coords.csv", index=False)
    case_paths(g, std[0])


#: case studies: (anchor, quarter, ratio shown); a window that includes the 2020 shock
CASES = [("DAL", "2021Q4", "Operating Income"), ("MAR", "2021Q4", "Operating Income"), ("JPM", "2023Q2", "Interest Expense")]


def case_paths(g: ue.Group, run: str, tag: str = "", k: int = 5) -> None:
    """For each case: the anchor's and its k nearest firms' path of one ratio over the 12-quarter window, peers by the
    model and by the raw fingerprint (the window's last quarter)."""
    Z = ue._unit(model_vec(g, run).full())
    F = ue._unit(g.FP)
    inv = lambda X: (torch.sign(X) * torch.expm1(torch.abs(X)) if g.input_slog else X).numpy()   # X = F since 8 Nov
    cols = [c for c in pd.read_parquet(g.args.panels_dir / g.args.panel / "targets" / "fingerprint.parquet").columns
            if c not in ("ticker", "quarter")]
    CASES_ = CASES + [("KO", "2024Q4", "Operating Income"), ("NVDA", "2024Q4", "Operating Income")]
    rows = []
    for anchor, q, ratio in CASES_:
        pool = pool_of(g, q)
        tick = g.meta.ticker.to_numpy()[pool]
        if anchor not in set(tick):
            continue
        i = int(np.flatnonzero(tick == anchor)[0])
        j = g.xcol[cols.index(ratio)]
        for name, V in (("standard", Z), (RAW_FP, F)):
            sim = V[pool] @ V[pool][i]
            sim[i] = -np.inf
            peers = np.argsort(-sim)[:k]
            for role, idx in [("anchor", [i])] + [("peer", list(peers))]:
                for r in idx:
                    path = inv(g.X[pool[r], :, j])
                    rows += [{"case": f"{anchor} {q}", "ratio": ratio, "vector": name, "role": role, "ticker": tick[r],
                              "lag": lag, "value": float(v)} for lag, v in zip(range(len(path) - 1, -1, -1), path)]
    pd.DataFrame(rows).to_csv(OUT / f"case_paths{tag}.csv", index=False)


def part_compare(families: tuple, drop_issuance: bool = False) -> None:
    """Family A against family B (e.g. a refinement against the standard) on the 12 cells of T1, T3, T4 and on T2,
    T5: per-seed win (max(2 sd), 0.002) and mean win (2 se of the difference), both primary sets. drop_issuance: the
    9 cells without share issuance (a cover-page share count, not an accounting item; addendum 14)."""
    fa, fb = families
    A, B = cells_of(runs_of(fa, scored=True)), cells_of(runs_of(fb, scored=True))
    ma, mb, sa, sb, na, nb = A.mean(), B.mean(), A.std(), B.std(), len(A), len(B)
    d = ma - mb
    quants = [q for q in ue.QUANTITIES if not (drop_issuance and q == "share issuance")]
    groups = {"T1, T3, T4": lambda s: [(s, t, q) for t in ue.CONTEMP for q in quants],
              "T2 industry": lambda s: [(s, "T2 industry", q) for q in ("same SIC2 among 10 nearest", "majority SIC2 correct")],
              "T5 holes": lambda s: [(s, "T5 holes", "10-NN overlap, 20 % of last quarter missing")]}
    rows = []
    for s in PRIMARY:
        for name, cells in groups.items():
            rows.append({"A": fa, "B": fb, "set": s, "cells": name, "seeds": f"{na} vs {nb}", "rule": "per seed",
                         **tally(d, np.maximum(2 * np.maximum(sa, sb), 0.002), cells(s))})
            rows.append({"A": fa, "B": fb, "set": s, "cells": name, "seeds": f"{na} vs {nb}", "rule": "mean",
                         **tally(d, 2 * np.sqrt(sa ** 2 / na + sb ** 2 / nb), cells(s))})
    C = pd.DataFrame(rows)
    D = pd.DataFrame({fa: ma, f"sd {fa}": sa, fb: mb, f"sd {fb}": sb, "difference": d})
    tag = "_".join(f.replace(" ", "").replace(",", "") for f in families) + ("_9cells" if drop_issuance else "")
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUT / f"compare_{tag}.csv", index=False)
    D.to_csv(OUT / f"compare_{tag}_cells.csv")
    print(C.to_string(index=False))
    D = D[~D.index.get_level_values(2).str.contains("next year")]
    print(D.round(3).to_string())


def part_paper(n_seeds: int = 3, quarter: str = "2024Q4", family: str = "standard") -> None:
    """Geometry (cosines of z against the fingerprint's and the deltas', one quarter) and input importance per pillar:
    each input line set to 0 over the whole window (as if never reported) or in its last quarter only; the increase
    of the Pillar 1 and the Pillar 2 pair loss, quarter batches of the out-of-time firms, mean over the seeds."""
    from services.similarity import pair_loss
    std = runs_of(family)[:n_seeds]
    g = group(std[0])
    tag = "" if family == "standard" else "_" + family.replace(" ", "").replace(",", "")
    rng = np.random.default_rng(0)
    pool = pool_of(g, quarter)
    smp = np.sort(rng.choice(pool, min(1500, len(pool)), replace=False))
    iu = np.triu_indices(len(smp), 1)
    cos = lambda V: (ue._unit(V[smp]) @ ue._unit(V[smp]).T)[iu]
    Z = model_vec(g, std[0]).full()
    pick = rng.choice(len(iu[0]), min(200_000, len(iu[0])), replace=False)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"latent": cos(Z)[pick], "fingerprint": cos(g.FP)[pick], "deltas": cos(g.D)[pick],
                  "quarter": quarter}).to_csv(OUT / f"geometry{tag}.csv", index=False)
    lines = list(g.sets["train firms"].fin_cols)
    batches = []
    for q in oot_quarters(g):
        rows = pool_of(g, q)
        batches.append(np.sort(rng.choice(rows, min(256, len(rows)), replace=False)))
    rec = []
    for r in std:
        cfg = json.loads((RUNS / r / "config.json").read_text())
        m = SimilarityEncoder(g.args.t_in, g.X.shape[-1], g.sets["train firms"].X_macro.shape[-1], int(cfg["latent_dim"]),
                              use_macro=False, hidden=int(cfg["hidden"]))
        m.load_state_dict(torch.load(RUNS / r / "encoder_no_macro.pth", map_location="cpu"))
        m.eval()

        @torch.no_grad()
        def losses(X, b):
            z = m(X, None)[0]
            return (pair_loss(z, torch.as_tensor(g.FP[b])).item(), pair_loss(z, torch.as_tensor(g.D[b])).item())
        base = np.array([losses(g.X[b], b) for b in batches]).mean(0)
        for j, line in enumerate(lines):
            for mode in ("whole window", "last quarter"):
                out = []
                for b in batches:
                    X = g.X[b].clone()
                    if mode == "whole window":
                        X[:, :, j] = 0.0
                    else:
                        X[:, -1, j] = 0.0
                    out.append(losses(X, b))
                d = np.array(out).mean(0) - base
                rec += [{"run": r, "line": line, "mode": mode, "pillar": p_, "increase": float(v), "base": float(b_)}
                        for p_, v, b_ in zip(("Pillar 1", "Pillar 2"), d, base)]
        print(f"[INFO] importance {r} done", flush=True)
    pd.DataFrame(rec).to_csv(OUT / f"importance{tag}.csv", index=False)
    case_paths(g, std[0], tag)
    I = pd.DataFrame(rec).groupby(["mode", "pillar", "line"]).increase.mean().unstack(["mode", "pillar"])
    print(I.sort_values(("whole window", "Pillar 1"), ascending=False).round(5).head(12).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["seeds", "claims", "t6", "curves", "map", "paper", "compare", "figures"], required=True)
    p.add_argument("--seeds", type=int, default=None, help="t6 / curves: number of seeds per model family")
    p.add_argument("--tag", default="", help="subfolder of the output files (tests)")
    p.add_argument("--families", nargs="+", default=["standard", "λ 0"], help="t6: the model families")
    p.add_argument("--lag", type=int, default=1, help="t6: quarters of the change (1 pre-registered, 4 exploratory)")
    p.add_argument("--period", default="2023Q3", choices=["2023Q3", "2021Q3"], help="claims: the test period")
    p.add_argument("--drop_issuance", action="store_true",
                   help="compare: count the 9 cells of T1, T3, T4 without share issuance (addendum 14)")
    a = p.parse_args()
    global OUT
    if a.tag:
        OUT = OUT / a.tag
    t0 = time.time()
    if a.part == "figures":
        import stability_figures
        stability_figures.main()
    else:
        kw = {"t6": {"max_seeds": a.seeds or 10, "lag": a.lag, "families": tuple(a.families)}, "curves": {"n_seeds": a.seeds or 3},
              "claims": {"period": a.period}, "compare": {"families": tuple(a.families), "drop_issuance": a.drop_issuance},
              "paper": {"family": a.families[0] if a.families != ["standard", "λ 0"] else "standard"}}.get(a.part, {})
        {"seeds": part_seeds, "claims": part_claims, "t6": part_t6, "curves": part_curves, "map": part_map,
         "paper": part_paper, "compare": part_compare}[a.part](**kw)
    print(f"[INFO] {a.part} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
