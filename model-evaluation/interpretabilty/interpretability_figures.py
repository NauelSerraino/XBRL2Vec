"""
Interpretability figures of the paper model: the "balanced" family (protocol addendum 16: the 2,115 firms with full
data 2013Q1-2025Q4, 317 unseen, cutoff 2022Q4; z 32, mask 0.2, λ 0.5, signed-log fingerprint, decay 0.8). Maps, pairs,
components and paths use seed 42; importance and concepts the 10 seeds (bar: mean, line: range), on the unseen firms
after the cutoff (2023Q1-2025Q4). Data and helpers are the scoring's own (src/usefulness_eval.py, src/stability_eval.py).

    xbrl2vec/bin/python model-evaluation/interpretabilty/interpretability_figures.py

Output next to this script, NN_<name>.png and NN_<name>.csv (the numbers behind the figure):
  01 map_sectors          t-SNE of the 2,115 firms in 2025Q4, one panel per large sector
  02 map_size_growth      the same map coloured by size and by asset growth
  03 importance_items     which ratios move the embedding (Gaussian noise on one ratio over the whole window)
  04 importance_sectors   the same per sector, against all firms: which ratios matter more or less in each sector
  05 pair_reasons         why two firms are peers: the drop of their cosine when one ratio is missing in both windows
  06 components           what the main directions mean: the top 8 principal components against ratios and quantities
  07 concepts             which concepts a straight line in the embedding detects (logistic probes, AUC on unseen firms)
  08 firm_changes         four firms with a known change: similarity of each firm's embedding with itself a year
                          earlier, 2016Q4-2025Q4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import stability_eval as se            # noqa: E402
import stability_figures as sf         # noqa: E402  (Agg backend, style, sector groups)
import usefulness_eval as ue           # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.manifold import TSNE      # noqa: E402

SET, QUARTER, BLUE = "unseen firms, out of time", "2025Q4", "#2a78d6"
QUANTS = ["size", "asset growth", "revenue growth"]
STMT = {"bs": ("balance sheet", "#2a78d6"), "ins": ("income statement", "#eb6834"), "cf": ("cash flow", "#1baf7a")}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
PAIRS = ["META", "WFC", "JNJ", "TSLA"]                      # unseen firms, each with its nearest firm (seed 42, 2025Q4)
EVENTS = {"CVS": ("2018Q4", "buys Aetna"), "ABBV": ("2020Q2", "buys Allergan"),
          "GE": ("2023Q1", "spins off GE HealthCare"), "NVDA": ("2023Q2", "AI boom")}
LINES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def save(fig, name: str, data: pd.DataFrame) -> None:
    fig.savefig(HERE / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(HERE / f"{name}.csv", index=False)
    print(f"[INFO] {name}", flush=True)


def bare(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def cosr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine."""
    return (ue._unit(A) * ue._unit(B)).sum(1)


def heatmap(fig, ax, D: pd.DataFrame, cmap, vmin, vmax, label: str) -> None:
    im = ax.imshow(D.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(D.shape[1]), D.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(D.shape[0]), D.index, fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label(label, fontsize=8); cb.outline.set_visible(False)


# --- 1-2. the map -----------------------------------------------------------------------------------------------------

def maps(g, U) -> None:
    pool = se.pool_of(g, QUARTER)
    xy = TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(U[pool])
    M = pd.DataFrame({"ticker": g.meta.ticker.iloc[pool].to_numpy(), "firms": g.meta.firms.iloc[pool].to_numpy(),
                      "x": xy[:, 0], "y": xy[:, 1], "sic2": g.sic[pool], "sector": [sf.sector_group(s) for s in g.sic[pool]],
                      **{k: g.Q[k].iloc[pool].to_numpy(float) for k in ["size", "asset growth"]}})
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, sec in zip(axes.flat, M.sector.value_counts().index[:6]):
        h = M.sector.eq(sec)
        ax.scatter(M.x[~h], M.y[~h], s=3, color=sf.GRID, lw=0, rasterized=True)
        ax.scatter(M.x[h], M.y[h], s=5, color=BLUE, lw=0, rasterized=True)
        ax.set_title(f"{sec} ({h.sum()} firms)", fontsize=9)
        bare(ax)
    fig.suptitle(f"The embedding space in {QUARTER} (t-SNE of the 2,115 firms, seed 42): each panel highlights one sector",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=sf.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "01_map_sectors", M)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, k, cmap, title in [(axes[0], "size", SEQ, "size (log total assets)"),
                               (axes[1], "asset growth", DIV, "asset growth (log, year on year)")]:
        v = M[k].to_numpy()
        ok = np.isfinite(v)
        p = np.nanpercentile(abs(v), 95)
        lim = (np.nanpercentile(v, 2), np.nanpercentile(v, 98)) if k == "size" else (-p, p)     # growth: centred on 0
        ax.scatter(M.x[~ok], M.y[~ok], s=3, color=sf.GRID, lw=0, rasterized=True)
        sc = ax.scatter(M.x[ok], M.y[ok], c=np.clip(v[ok], *lim), cmap=cmap, vmin=lim[0], vmax=lim[1], s=5, lw=0,
                        rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01).outline.set_visible(False)
        ax.set_title(title, fontsize=9)
        bare(ax)
    fig.suptitle(f"The same map coloured by two quantities the model never sees ({QUARTER}; grey: not observed)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=sf.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "02_map_size_growth", M)


# --- 3-4. importance, overall and by sector -----------------------------------------------------------------------------

def importance(g, enc: list, Z: list, ratios: list[str], stmt: dict) -> None:
    rows = np.flatnonzero(g.focal[SET])
    X0 = g.X[rows]
    assert np.allclose(cosr(Z[0][rows], enc[0](X0)), 1, atol=1e-5), "the embeddings are reproduced without noise"
    noise = torch.as_tensor(np.random.default_rng(0).standard_normal(X0.shape[:2]), dtype=X0.dtype)
    per_seed, per_firm = {}, {}
    for j, c in enumerate(ratios):
        X = X0.clone()
        X[:, :, int(g.xcol[j])] += noise
        D = np.stack([1 - cosr(Z[s][rows], e(X)) for s, e in enumerate(enc)])          # seeds x firm-quarters
        per_seed[c], per_firm[c] = D.mean(1), D.mean(0)
    I = pd.DataFrame(per_seed).agg(["mean", "min", "max"]).T.sort_values("mean")
    I["statement"] = [STMT[stmt[c]][0] for c in I.index]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    y = np.arange(len(I))
    ax.barh(y, I["mean"], color=[STMT[stmt[c]][1] for c in I.index], height=0.7)
    ax.hlines(y, I["min"], I["max"], color=sf.INK, lw=1)
    ax.set_yticks(y, I.index, fontsize=7.5)
    ax.set_xlabel("change of the embedding, 1 − cos(z, z with noise)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=col, label=name) for name, col in STMT.values()], loc="lower right")
    ax.set_title("Which ratios move the embedding? Gaussian noise (sd 1) on one ratio over the whole window\n"
                 "unseen firms 2023Q1–2025Q4; bar: mean over 10 seeds, line: range", fontsize=9.5)
    save(fig, "03_importance_items", I.reset_index())

    sector = pd.Series([sf.sector_group(s) for s in g.sic[rows]])
    n = sector.value_counts()
    S = pd.DataFrame(per_firm).groupby(sector).mean()[I.index[::-1]].loc[n.index]
    S = np.log2(S.div(S.sum(1), axis=0) / (I["mean"] / I["mean"].sum())[S.columns])     # sector share / overall share
    S.index = [f"{s} ({n[s]:,})" for s in S.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    heatmap(fig, ax, S, DIV, -1, 1, "log2(share in the sector / share over all firms)")
    ax.set_title("Which ratios matter more (red) or less (blue) in each sector than over all firms? +1: twice the "
                 "share, −1: half\nunseen firm-quarters 2023Q1–2025Q4, 10 seeds; ratios ordered as in figure 03",
                 fontsize=9.5)
    save(fig, "04_importance_sectors", S.reset_index(names="sector"))


# --- 5. why two firms are peers -----------------------------------------------------------------------------------------

def pair_reasons(g, enc: list, U: np.ndarray, ratios: list[str], desc: pd.Series) -> None:
    pool = se.pool_of(g, QUARTER)
    tick = g.meta.ticker.iloc[pool].to_numpy()
    ind = lambda t: str(desc.get(t, "?")).title()[:32].rstrip(" ,;")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    out = []
    for ax, t in zip(axes.flat, PAIRS):
        a = int(np.flatnonzero(tick == t)[0])
        S = U[pool] @ U[pool[a]]
        S[a] = -np.inf
        ab = pool[[a, int(np.argmax(S))]]
        X = g.X[ab].repeat(len(ratios), 1, 1)                  # rows 2j, 2j + 1: the pair with ratio j missing in both
        for j in range(len(ratios)):
            X[2 * j:2 * j + 2, :, int(g.xcol[j])] = 0
        drops = lambda e: cosr(*e(g.X[ab])[:, None])[0] - cosr(*e(X).reshape(len(ratios), 2, -1).transpose(1, 0, 2))
        D = pd.DataFrame([drops(e) for e in enc], columns=ratios)
        top = D.mean().nlargest(8)[::-1]
        fa, fb = (pd.Series(g.FP[r], index=ratios) for r in ab)
        y = np.arange(len(top))
        ax.barh(y, top, color=BLUE, height=0.7)
        ax.hlines(y, D[top.index].min(), D[top.index].max(), color=sf.INK, lw=1)
        ax.set_yticks(y, [f"{c}  ({fa[c]:+.1f} / {fb[c]:+.1f})" for c in top.index], fontsize=7.5)
        ax.grid(axis="y", visible=False)
        p = g.meta.ticker.iloc[ab[1]]
        ax.set_title(f"{t} ({ind(t)}) ↔ {p} ({ind(p)}), cosine {U[ab[0]] @ U[ab[1]]:.2f}", fontsize=8.5)
        out += [{"firm": t, "peer": p, "ratio": c, "drop": D[c].mean(), "min": D[c].min(), "max": D[c].max(),
                 "firm z": fa[c], "peer z": fb[c]} for c in ratios]
    for ax in axes[1]:
        ax.set_xlabel("drop of the pair's cosine when the ratio is missing in both windows")
    fig.suptitle(f"Why are they peers? Four unseen firms and their nearest firm in {QUARTER} (seed 42): the 8 ratios that "
                 "hold each pair together\nbar: mean over 10 seeds, line: range; in brackets the two firms' ratio "
                 "(z-score in the quarter)", x=0.01, ha="left", fontsize=10, fontweight="semibold", color=sf.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "05_pair_reasons", pd.DataFrame(out))


# --- 6. what the main directions mean -----------------------------------------------------------------------------------

def components(g, U: np.ndarray, ratios: list[str], stmt: dict) -> None:
    from sklearn.decomposition import PCA
    rows = np.flatnonzero(g.focal[SET])
    pca = PCA(8, random_state=0).fit(U[rows])
    pcs = [f"PC{i + 1} ({v:.0%})" for i, v in enumerate(pca.explained_variance_ratio_)]
    cols = sorted(ratios, key=lambda c: (list(STMT).index(stmt[c]), c))
    D = pd.concat([pd.DataFrame(pca.transform(U[rows]), columns=pcs), pd.DataFrame(g.FP[rows], columns=ratios),
                   g.Q.iloc[rows][QUANTS].reset_index(drop=True)], axis=1).corr().loc[pcs, cols + QUANTS]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    heatmap(fig, ax, D, DIV, -1, 1, "correlation")
    edges = np.cumsum([sum(stmt[c] == s for c in cols) for s in STMT] + [len(QUANTS)])
    for e, name, w in zip(edges, [v[0] for v in STMT.values()] + ["quantities"], np.diff(np.r_[0, edges])):
        if e < D.shape[1]:
            ax.axvline(e - 0.5, color=sf.INK, lw=1.2)
        ax.text(e - 0.5 - w / 2, -0.8, name, ha="center", va="bottom", fontsize=8, color=sf.INK2)
    ax.set_title("What the main directions of the embedding mean: correlation of its top 8 principal components (share "
                 "of variance) with the last quarter's ratios and the quantities\nunseen firm-quarters 2023Q1–2025Q4, "
                 "seed 42", fontsize=9.5, pad=28)
    save(fig, "06_components", D.reset_index(names="component"))


# --- 7. concepts ------------------------------------------------------------------------------------------------------

def concepts(g, Z: list, panel: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from preprocess.fingerprint import load_raw
    L, _ = load_raw(panel)
    ttm = lambda v: v.T.rolling(4, min_periods=4).sum().T
    ta = L["Total Assets"].where(L["Total Assets"] > 0)
    flag = lambda v: (v > 0).astype(float).where(v.notna())                             # 1 / 0, NaN where unknown
    top = lambda v: flag(v.rank(pct=True) - 0.75)                                        # top quarter of all firms
    C = {"loss-making (net income < 0)": flag(-ttm(L["Net Income"])),
         "R&D over 10 % of revenue": flag(ttm(L["Research And Development"].fillna(0))
                                          / ttm(L["Total Revenue"]).where(lambda v: v > 0) - 0.1),
         "top quarter of debt / assets": top((L["Current Debt"].fillna(0) + L["Long Term Debt"].fillna(0)) / ta),
         "top quarter of cash / assets": top(L["End Cash Position"] / ta),
         "asset-light (net PPE < 5 % of assets)": flag(0.05 - L["Net PPE"].fillna(0) / ta)}
    keys = pd.MultiIndex.from_frame(g.meta[["ticker", "quarter"]])
    Y = {k: v.stack(future_stack=True).reindex(keys).to_numpy(float) for k, v in C.items()}
    Y["bank (SIC 60–61)"] = np.isin(g.sic, [60, 61]).astype(float)
    tr, te = g.train_rows, np.flatnonzero(g.focal[SET])
    out = []
    for k, y in Y.items():
        a, b = tr[np.isfinite(y[tr])], te[np.isfinite(y[te])]
        assert 0.01 < y[b].mean() < 0.99, (k, y[b].mean())
        for s, z in enumerate(Z):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(z[a], y[a])
            out.append({"concept": k, "seed": s, "share": y[b].mean(), "AUC": roc_auc_score(y[b], m.decision_function(z[b]))})
    A = pd.DataFrame(out).groupby("concept").agg(share=("share", "first"), mean=("AUC", "mean"), min=("AUC", "min"),
                                                 max=("AUC", "max")).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(A))
    ax.barh(y, A["mean"] - 0.5, left=0.5, color=BLUE, height=0.65)
    ax.hlines(y, A["min"], A["max"], color=sf.INK, lw=1)
    ax.set_yticks(y, [f"{k} ({s:.0%})" for k, s in zip(A.index, A.share)], fontsize=8)
    ax.set_xlim(0.5, 1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC on the unseen firms 2023Q1–2025Q4 (0.5: chance, 1: perfect)")
    ax.set_title("Which concepts can a straight line in the embedding detect? Logistic probes trained on the training "
                 "firms up to 2022Q4\nbar: mean over 10 seeds, line: range; in brackets the share of unseen "
                 "firm-quarters with the concept", fontsize=9.5)
    save(fig, "07_concepts", A.reset_index())


# --- 8. firm changes ----------------------------------------------------------------------------------------------------

def changes(g, Z: list) -> None:
    q = g.meta.quarter
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(14, 3.6), sharey=True)
    out = []
    for ax, (t, (when, what)), col in zip(axes, EVENTS.items(), LINES):
        r = np.flatnonzero(g.meta.ticker.eq(t).to_numpy())
        r = r[np.argsort(q.iloc[r].to_numpy())]
        assert len(r) == q.nunique(), t                                   # every quarter, so r[4:] is a year after r[:-4]
        C = np.stack([cosr(z[r[4:]], z[r[:-4]]) for z in Z])                # seeds x quarters
        x = q.iloc[r[4:]].to_numpy()
        ax.fill_between(range(len(x)), C.min(0), C.max(0), color=col, alpha=0.2, lw=0)
        ax.plot(range(len(x)), C.mean(0), color=col)
        ax.axvline(list(x).index(when), color=sf.INK, ls=":", lw=1)
        ax.set_xticks([i for i, v in enumerate(x) if v.endswith("Q1")][::2], [v[:4] for v in x if v.endswith("Q1")][::2])
        ax.set_title(f"{t}: {what} ({when})", fontsize=9)
        out += [{"firm": t, "quarter": v, "mean": m, "min": lo, "max": hi}
                for v, m, lo, hi in zip(x, C.mean(0), C.min(0), C.max(0))]
    axes[0].set_ylabel("cos(z_t, z_t−4)")
    fig.suptitle("Does a firm's embedding move when the firm changes? Similarity of each firm's embedding with itself a "
                 "year earlier\nline: mean over 10 seeds, band: range; dotted: the event", x=0.01, ha="left",
                 fontsize=10, fontweight="semibold", color=sf.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "08_firm_changes", pd.DataFrame(out))


def main() -> None:
    sf.style()
    runs = se.runs_of("balanced")
    assert len(runs) == 10, len(runs)
    g = se.group(runs[0])
    enc = [se.encoder(g, r) for r in runs]
    Z = [e(g.X) for e in enc]
    U = ue._unit(Z[0])
    panel = g.args.panels_dir / g.args.panel
    fp_file = panel / "targets" / ue.ms.FINGERPRINT[g.args.fingerprint].replace(".parquet", "_balanced.parquet")
    ratios = [c for c in pd.read_parquet(fp_file).columns if c not in ("ticker", "quarter")]
    assert len(ratios) == g.FP.shape[1] == len(g.xcol)
    stmt = {line: s for s in STMT for line in pd.read_parquet(panel / f"{s}-raw.parquet", columns=["features"]).features}
    meta = ue.ms.DATA / "in" / g.args.panels_dir.name.replace("panels_", "download_") / "metadata.parquet"
    desc = pd.read_parquet(meta).drop_duplicates("ticker").set_index("ticker").sic_description
    maps(g, U)
    importance(g, enc, Z, ratios, stmt)
    pair_reasons(g, enc, U, ratios, desc)
    components(g, U, ratios, stmt)
    concepts(g, Z, panel)
    changes(g, Z)


if __name__ == "__main__":
    main()
