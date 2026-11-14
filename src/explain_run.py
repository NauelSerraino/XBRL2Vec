"""
The explainability analyses of notebooks/05_model_explainability.ipynb (the submitted paper's analyses, adapted to the
similarity setting) for one or more runs of main_similarity.py, saving every chart and table so that runs can be
compared afterwards.

    cd src && ../xbrl2vec/bin/python explain_run.py 20261031-130250 20261031-160215
    cd src && ../xbrl2vec/bin/python explain_run.py --compare          # the summaries of all explained runs

Per run, in data/out/similarity/<run>/explain/:
  01_importance_<target>.png/.csv   perturbation importance (Gaussian noise x_j + ε·σ_j, 3 draws) of every input for
                                    the pair loss against each target the run has (pillar1: fingerprint; pillar2:
                                    next-h-quarter change)
  02_geometry.png                   pairwise cosine / Euclidean, latent vs financial input, fingerprint, change
  03_ig_top20.png                   Integrated Gradients (zero baseline, 50 steps): latent mode (‖z‖) and similarity
                                    mode for each target; top-20 features
  04_ig_sectors.png/.csv            macro exposure ρ_c = Σ|IG_macro| / Σ|IG| by sector
  05_probing.csv                    linear probing R² from z (macro model and no-macro model) to the financial window,
                                    the macro window, the fingerprint and the change
  06_aapl_peers.png/.csv            the three firms whose embedding similarity to AAPL departs most from their
                                    fingerprint similarity, over time
  summary.json                      the headline numbers (also collected in data/out/similarity/explain_summary.csv)
Analyses on the unseen firms (companies held out of training), all quarters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import main_similarity as ms
from models.similarity_encoder import SimilarityEncoder
from services.similarity import attach_changes, attach_window_deltas, embed, pair_loss

RUNS = ms.DATA / "out" / "similarity"
MACRO_COLOR, FIN_COLOR = "#c44e52", "#4c72b0"


def load(run: str):
    """The run's configuration, its data rebuilt exactly as main_similarity.py built it, and its two encoders."""
    cfg = json.loads((RUNS / run / "config.json").read_text())
    args = _defaults()
    args.reuse = run
    ms.apply_reuse(args)
    if args.panels_dir is None:
        args.panels_dir = sorted(p for p in (ms.DATA / "in").glob("panels_*") if (p / "exog.parquet").exists())[-1]
    panel = args.panels_dir / args.panel
    fp = pd.read_parquet(panel / "targets" / "fingerprint.parquet")
    sets = ms.windows(args, panel, args.inputs, fp)
    keep = ms.universe(args, panel, fp)
    sets = {k: v.subset(pd.MultiIndex.from_frame(v.meta[["ticker", "quarter"]]).isin(keep)) for k, v in sets.items()}
    change_file = panel / "targets" / f"fingerprint_change_h{args.horizon}.parquet"
    if change_file.exists():
        ch = pd.read_parquet(change_file)
        sets = {k: attach_changes(v, ch, args.horizon) for k, v in sets.items()}
    if args.pillar2 == "window_deltas":
        sets = {k: attach_window_deltas(v, fp, args.t_in) for k, v in sets.items()}
    if args.macro_scale == "train":                                    # as in main_similarity: training windows' scale
        tr = sets["train firms"]
        tr = tr.subset(tr.meta.quarter.le(cfg.get("last_train_quarter", cfg["cutoff"])).to_numpy())
        mu, sd = tr.X_macro.mean(dim=(0, 1)), tr.X_macro.std(dim=(0, 1)).clamp_min(1e-8)
        for d in sets.values():
            d.X_macro = (d.X_macro - mu) / sd
    D = sets["unseen firms"]
    F_dim, M_dim = D.X_fin.shape[-1], D.X_macro.shape[-1]
    models = {}
    for name, use_macro in [("macro", True), ("no macro", False)]:
        m = SimilarityEncoder(args.t_in, F_dim, M_dim, args.latent_dim, use_macro=use_macro, hidden=args.hidden,
                              film_init=args.film_init, conditioning=args.macro_conditioning,
                              macro_target=args.macro_target)
        path = RUNS / run / f"encoder_{name.replace(' ', '_')}.pth"
        if not path.exists():                                          # --skip_macro_model: no macro encoder
            continue
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models[name] = m.eval()
    sector = pd.read_parquet(args.panels_dir / "metadata.parquet").drop_duplicates("ticker").set_index("ticker").sector
    return cfg, args, sets, D, models, sector


def _defaults():
    import sys
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        return ms.parse_args()
    finally:
        sys.argv = argv


def quarter_batches(D, rng, min_size=64, size=256):
    out = []
    for q, g in D.meta.groupby("quarter"):
        if len(g) >= min_size:
            out.append(rng.choice(g.index.to_numpy(), min(size, len(g)), replace=False))
    return out


def targets_of(D):
    """{target name: tensor [N, K]} for the targets this data has (rows of the change target may be NaN)."""
    t = {"pillar1": torch.tensor(D.target)}
    if D.target2 is not None:
        t["pillar2"] = torch.tensor(D.target2)
    return t


def batch_loss(model, X, M, Y):
    """Pair loss of the embeddings against target Y over the rows of Y without NaN."""
    ok = torch.isfinite(Y).all(dim=1)
    if ok.sum() < 16:
        return np.nan
    with torch.no_grad():
        z, _ = model(X[ok], M[ok])
        return pair_loss(z, Y[ok]).item()


def perturbation_importance(model, D, batches, Y, names, F_dim, draws=3):
    sd_fin, sd_mac = D.X_fin.std(dim=(0, 1)), D.X_macro.std(dim=(0, 1))
    gen = torch.Generator().manual_seed(0)
    base = np.nanmean([batch_loss(model, D.X_fin[b], D.X_macro[b], Y[b]) for b in batches])
    imp = {}
    for j, name in enumerate(names):
        deltas = []
        for b in batches:
            for _ in range(draws):
                X, M = D.X_fin[b].clone(), D.X_macro[b].clone()
                if j < F_dim:
                    X[:, :, j] += sd_fin[j] * torch.randn(X.shape[:2], generator=gen)
                else:
                    M[:, :, j - F_dim] += sd_mac[j - F_dim] * torch.randn(M.shape[:2], generator=gen)
                deltas.append(batch_loss(model, X, M, Y[b]))
        imp[name] = np.nanmean(deltas) - base
    return pd.Series(imp).sort_values(ascending=False), base


def integrated_gradients(target_fn, X, M, steps=50):
    gx, gm = torch.zeros_like(X), torch.zeros_like(M)
    for a in torch.linspace(1 / steps, 1, steps):
        Xa, Ma = (a * X).requires_grad_(True), (a * M).requires_grad_(True)
        target_fn(Xa, Ma).sum().backward()
        gx += Xa.grad
        gm += Ma.grad
    return (X * gx / steps).detach(), (M * gm / steps).detach()


def own_pair_loss(model, Y):
    """Each firm's own pair loss within its batch (similarity mode); rows of Y with NaN contribute 0."""
    ok = torch.isfinite(Y).all(dim=1)
    Yf = torch.nan_to_num(Y)

    def f(X, M):
        z, _ = model(X, M)
        u = z / z.norm(dim=1, keepdim=True).clamp_min(1e-8)
        uf = Yf / Yf.norm(dim=1, keepdim=True).clamp_min(1e-8)
        err = (u @ u.T - uf @ uf.T) ** 2 * (ok[:, None] & ok[None, :])
        return err.sum(dim=1) / ok.sum().clamp_min(1)
    return f


def barh(ax, s, names_macro, title, xlabel):
    s = s[::-1]
    ax.barh(s.index, s.values, color=[MACRO_COLOR if n in names_macro else FIN_COLOR for n in s.index])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.tick_params(labelsize=7)


def explain(run: str) -> dict:
    cfg, args, sets, D, models, sector = load(run)
    out = RUNS / run / "explain"
    out.mkdir(exist_ok=True)
    model, blind = models["macro"], models["no macro"]
    FIN, MAC = D.fin_cols, D.macro_cols
    names, F_dim = FIN + MAC, len(D.fin_cols)
    rng = np.random.default_rng(0)
    batches = quarter_batches(D, rng)
    T = targets_of(D)
    summary = {"run": run, "loss": cfg.get("loss", "pillar1"), "inputs": cfg.get("inputs"), "hidden": cfg.get("hidden"),
               "film_init": cfg.get("film_init", "default"), "macro_weight_mode": cfg.get("macro_weight_mode", "corr")}
    print(f"[{run}] {len(D.meta):,} windows of unseen firms; targets {list(T)}")

    # 1. perturbation importance, per target
    fig, axes = plt.subplots(1, len(T), figsize=(8 * len(T), 11), squeeze=False)
    for ax, (tname, Y) in zip(axes[0], T.items()):
        imp, base = perturbation_importance(model, D, batches, Y, names, F_dim)
        imp.to_csv(out / f"01_importance_{tname}.csv", header=["importance"])
        share = imp[MAC].clip(lower=0).sum() / max(imp.clip(lower=0).sum(), 1e-12)
        summary[f"macro_share_importance_{tname}"] = float(share)
        summary[f"top_features_{tname}"] = list(imp.index[:5])
        barh(ax, imp, MAC, f"{tname}: perturbation importance (base loss {base:.4f}; macro {share:.1%})",
             "increase of the pair loss under 1σ noise")
    plt.tight_layout(); fig.savefig(out / "01_importance.png", dpi=110); plt.close(fig)

    # 2. geometry: latent vs financial input, fingerprint, change (macro input zeroed for the latent, as in the paper)
    q = D.meta.quarter.value_counts().idxmax()
    rows = np.sort(rng.choice(D.meta.index[D.meta.quarter == q].to_numpy(), min(1500, int((D.meta.quarter == q).sum())), replace=False))
    z0 = embed(model, D.subset(np.isin(np.arange(len(D.meta)), rows)), "cpu", "zero")
    spaces = {"financial input": D.X_fin[rows].reshape(len(rows), -1).numpy(), "fingerprint": D.target[rows]}
    if D.target2 is not None:
        ok2 = np.isfinite(D.target2[rows]).all(axis=1)
        spaces["change"] = (D.target2[rows][ok2], ok2)
    up = lambda M: M[np.triu_indices_from(M, k=1)]
    fig, axes = plt.subplots(len(spaces), 2, figsize=(14, 6 * len(spaces)), squeeze=False)
    for r, (name, X) in enumerate(spaces.items()):
        Z = z0 if not isinstance(X, tuple) else z0[X[1]]
        X = X if not isinstance(X, tuple) else X[0]
        for c, (metric, f) in enumerate([("cosine", cosine_similarity), ("Euclidean", euclidean_distances)]):
            a, b = up(f(X)), up(f(Z))
            rho = float(spearmanr(a, b)[0])
            summary[f"geometry_{name}_{metric}"] = rho
            axes[r, c].hexbin(a, b, gridsize=80, cmap="Blues" if c == 0 else "Oranges", mincnt=1, bins="log")
            axes[r, c].set_xlabel(f"{metric}, {name}"); axes[r, c].set_ylabel(f"{metric}, latent")
            axes[r, c].set_title(f"{name} vs latent ({metric}): Spearman {rho:.3f}", fontsize=10)
    plt.tight_layout(); fig.savefig(out / "02_geometry.png", dpi=110); plt.close(fig)

    # 3-4. Integrated Gradients: latent mode and similarity mode per target
    modes = {"latent": lambda Y: (lambda X, M: model(X, M)[0].norm(dim=1))}
    modes.update({f"similarity {t}": (lambda t: lambda Y: own_pair_loss(model, Y))(t) for t in T})
    IG = {m: [] for m in modes}
    idx = []
    for b in batches:
        X, M = D.X_fin[b], D.X_macro[b]
        for mode, make in modes.items():
            Y = T["pillar2" if mode.endswith("pillar2") else "pillar1"][b]
            ax_, am_ = integrated_gradients(make(Y), X, M)
            IG[mode].append(np.concatenate([ax_.abs().sum(1).numpy(), am_.abs().sum(1).numpy()], axis=1))
        idx.append(b)
    idx = np.concatenate(idx)
    IG = {m: pd.DataFrame(np.concatenate(v), columns=names, index=idx) for m, v in IG.items()}
    fig, axes = plt.subplots(1, len(IG), figsize=(8 * len(IG), 7), squeeze=False)
    for ax, (mode, A) in zip(axes[0], IG.items()):
        share = float(A[MAC].sum().sum() / A.sum().sum())
        summary[f"macro_share_ig_{mode.replace(' ', '_')}"] = share
        barh(ax, A.mean().sort_values(ascending=False).head(20), MAC, f"IG, {mode} mode (macro {share:.1%})", "mean |attribution|")
    plt.tight_layout(); fig.savefig(out / "03_ig_top20.png", dpi=110); plt.close(fig)
    rho = {m: A[MAC].sum(1) / A.sum(1) for m, A in IG.items()}
    per_firm = pd.DataFrame({m: r.groupby(D.meta.loc[r.index, "ticker"].to_numpy()).mean() for m, r in rho.items()})
    per_firm["sector"] = per_firm.index.map(sector)
    sec = per_firm.groupby("sector").agg(**{m: (m, "mean") for m in rho}, firms=("latent", "size"))
    sec = sec[sec.firms >= 15].sort_values(list(rho)[-1], ascending=False)
    sec.to_csv(out / "04_ig_sectors.csv")
    summary["macro_exposure_sector_spread"] = float(sec[list(rho)[-1]].max() - sec[list(rho)[-1]].min()) if len(sec) else np.nan
    summary["macro_exposure_top_sectors"] = list(sec.index[:3])
    fig, ax = plt.subplots(figsize=(13, 8))
    xm, ym = "latent", list(rho)[-1]
    ax.scatter(sec[xm], sec[ym], s=sec.firms * 8, alpha=0.5, color=FIN_COLOR, edgecolor="black")
    for name, r in sec.head(8).iterrows():
        ax.annotate(str(name)[:35], (r[xm], r[ym]), fontsize=7)
    ax.set_xlabel(f"macro exposure ρ, {xm} mode"); ax.set_ylabel(f"macro exposure ρ, {ym} mode")
    ax.set_title("sector macro exposure (bubble size: number of firms)")
    plt.tight_layout(); fig.savefig(out / "04_ig_sectors.png", dpi=110); plt.close(fig)

    # 5. linear probing
    Z, Zb = embed(model, D, "cpu", "real"), embed(blind, D, "cpu", "none")
    sample = rng.choice(len(D.meta), min(20000, len(D.meta)), replace=False)
    def r2(Zm, Y):
        ok = np.isfinite(Y).all(axis=1)
        return float(r2_score(Y[ok], LinearRegression().fit(Zm[ok], Y[ok]).predict(Zm[ok])))
    goals = {"financial window": D.X_fin[sample].reshape(len(sample), -1).numpy(),
             "macro window": D.X_macro[sample].reshape(len(sample), -1).numpy(), "fingerprint": D.target[sample]}
    if D.target2 is not None:
        goals["change"] = D.target2[sample]
    probe = pd.DataFrame({name: {g: r2(Zm[sample], Y) for g, Y in goals.items()}
                          for name, Zm in [("z (with macro)", Z), ("z (trained without macro)", Zb)]})
    probe.to_csv(out / "05_probing.csv")
    for g in probe.index:
        summary[f"probe_{g.replace(' ', '_')}"] = float(probe.loc[g, "z (with macro)"])
        summary[f"probe_{g.replace(' ', '_')}_no_macro"] = float(probe.loc[g, "z (trained without macro)"])

    # 6. divergent peers of AAPL
    S = D if "AAPL" in set(D.meta.ticker) else sets["train firms"]
    ZS = Z if S is D else embed(model, S, "cpu", "real")
    series = {}
    for q, g in S.meta.groupby("quarter"):
        if "AAPL" not in set(g.ticker):
            continue
        i = g.index[g.ticker == "AAPL"][0]
        others = g.index[g.ticker != "AAPL"]
        ce, cf = cosine_similarity(ZS[[i]], ZS[others])[0], cosine_similarity(S.target[[i]], S.target[others])[0]
        for t, e, f in zip(g.loc[others, "ticker"], ce, cf):
            series.setdefault(t, []).append((q, e, f))
    gap = pd.Series({t: np.mean([e - f for _, e, f in v]) for t, v in series.items() if len(v) >= 8}).sort_values(ascending=False)
    gap.head(20).to_csv(out / "06_aapl_peers.csv", header=["mean gap, embedding - fingerprint"])
    summary["aapl_top_gaps"] = {t: round(float(v), 3) for t, v in gap.head(3).items()}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    for ax, t in zip(axes, gap.index[:3]):
        v = pd.DataFrame(series[t], columns=["quarter", "embedding", "fingerprint"]).set_index("quarter")
        v.plot(ax=ax, marker="o", ms=3)
        ax.set_title(f"AAPL vs {t} ({str(sector.get(t) or '')[:30]}), gap {gap[t]:.2f}", fontsize=9); ax.tick_params(labelsize=7)
    plt.tight_layout(); fig.savefig(out / "06_aapl_peers.png", dpi=110); plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[{run}] written {out}")
    return summary


def compare() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*/explain/summary.json"))]
    df = pd.DataFrame(rows).set_index("run")
    df.to_csv(RUNS / "explain_summary.csv")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="Run folders (timestamps) under data/out/similarity.")
    p.add_argument("--compare", action="store_true", help="Only collect the summaries of the runs explained so far.")
    args = p.parse_args()
    for run in args.runs:
        explain(run)
    df = compare()
    cols = [c for c in df.columns if c.startswith(("macro_share", "geometry", "probe", "macro_exposure_sector"))]
    pd.set_option("display.width", 250)
    print(df[["loss", "inputs", "hidden", "film_init", "macro_weight_mode"] + cols].T.to_string())
    print(f"\nAll explained runs: {RUNS / 'explain_summary.csv'}")


if __name__ == "__main__":
    main()
