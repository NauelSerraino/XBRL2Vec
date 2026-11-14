"""
Figures of the report of 4 Nov 2026 (docs/report_2026-11-04_stability.md), from the CSVs of stability_eval.py and
usefulness_eval.py. One colour per vector in every figure (the reference palette of the dataviz guidance):

    standard model (z 32, mask 0.2, λ 0.5)   blue      raw fingerprint      orange
    window-mean fingerprint                  aqua      PCA-32 of fp + Δ     yellow
    λ 0 (no moves)                           magenta   SIC                  green
    fingerprint + window deltas              violet    unmasked model       red

    cd src && ../xbrl2vec/bin/python stability_eval.py --part figures
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import usefulness_eval as ue

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("STABILITY_DATA", ROOT / "data" / "out" / "stability"))
FIG = Path(os.environ.get("STABILITY_FIG", ROOT / "docs" / "figures" / "2026-11-04"))
SET = "unseen firms, out of time"

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GRAY = "#b5b3ac"
COLOR = {"standard": "#2a78d6", ue.RAW_FP: "#eb6834", "window-mean fingerprint (32)": "#1baf7a",
         "PCA-32 of fingerprint + window deltas": "#eda100", "λ 0": "#e87ba4", "SIC2": "#008300",
         ue.BOTH: "#6250d6", "unmasked": "#e34948"}
LABEL = {"standard": "model (z 32, mask 0.2, λ 0.5)", ue.RAW_FP: "raw fingerprint",
         "window-mean fingerprint (32)": "12-quarter mean fingerprint", "PCA-32 of fingerprint + window deltas":
         "PCA-32 of fingerprint + deltas", "λ 0": "model, λ 0 (no moves)", "SIC2": "SIC2",
         ue.BOTH: "fingerprint + deltas (raw)", "unmasked": "model, no input mask"}
BLUES = {0.0: "#86b6ef", 0.1: "#5598e7", 0.2: "#2a78d6", 0.3: "#1c5cab", 0.4: "#104281"}   # mask share, ordinal


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "lines.markersize": 5,
    })


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] figure {name}.png")


def family_of(vector: str, family: str) -> str:
    return family if family in COLOR else vector


# --- 1. seed and data stability ------------------------------------------------------------------------------------

def fig_seed_agreement() -> None:
    P = pd.read_csv(DATA / "seed_pairs.csv")
    P = P[P.set.eq(SET)]
    per_pair = P.groupby(["kind", "a", "b"]).overlap.mean().reset_index()
    order = ["init seeds", "ensemble of 2 seeds vs another", "ensemble of 3 seeds vs another",
             "ensemble of 5 seeds vs another", "80 % samples", "80 % sample vs all firms", "λ 0 seeds", "z 64 seeds",
             "λ 0.5 vs λ 0, same seed",
             f"model vs {ue.RAW_FP}", "model vs PCA-32 of fingerprint + window deltas", f"model vs {ue.BOTH}",
             "model vs window-mean fingerprint (32)", "raw fingerprint vs itself with 20 % holes"]
    names = {"init seeds": "model vs model, another seed", "80 % samples": "model vs model, another 80 % of firms",
             "ensemble of 2 seeds vs another": "ensemble of 2 seeds vs 2 others",
             "ensemble of 3 seeds vs another": "ensemble of 3 seeds vs 3 others",
             "ensemble of 5 seeds vs another": "ensemble of 5 seeds vs 5 others", "z 64 seeds": "z 64 model, another seed",
             "80 % sample vs all firms": "model on 80 % of firms vs on all",
             "λ 0 seeds": "λ 0 model vs λ 0 model, another seed", "λ 0.5 vs λ 0, same seed": "model vs λ 0 model, same seed",
             f"model vs {ue.RAW_FP}": "model vs raw fingerprint", "model vs PCA-32 of fingerprint + window deltas":
             "model vs PCA-32 of fingerprint + deltas", f"model vs {ue.BOTH}": "model vs fingerprint + deltas (raw)",
             "model vs window-mean fingerprint (32)": "model vs 12-quarter mean",
             "raw fingerprint vs itself with 20 % holes": "raw fingerprint vs itself, 20 % missing"}
    order = [k for k in order if k in set(per_pair.kind)]
    g = per_pair.groupby("kind").overlap
    m, lo, hi, n = g.mean().reindex(order), g.min().reindex(order), g.max().reindex(order), g.size().reindex(order)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 0.8))
    y = np.arange(len(order))[::-1]
    stab = [k in ("init seeds", "80 % samples", "80 % sample vs all firms", "λ 0 seeds", "z 64 seeds")
            or k.startswith("ensemble") for k in order]
    ax.barh(y, m, height=0.55, color=[COLOR["standard"] if s else GRAY for s in stab], zorder=2)
    ax.hlines(y, lo, hi, color=INK2, linewidth=1, zorder=3)
    for yi, k in zip(y, order):
        ax.text(max(m[k], hi[k]) + 0.012, yi, f"{m[k]:.2f}" + (f"  ({n[k]} pairs)" if n[k] > 1 else ""), va="center", color=INK2, fontsize=8)
    ax.set_yticks(y, [names[k] for k in order])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("share of the 10 nearest firms in common (unseen firms, out of time)")
    ax.grid(axis="y", visible=False)
    ax.set_title("How much do a firm's 10 nearest firms change?")
    save(fig, "f1_seed_agreement")


def fig_cells_by_seed() -> None:
    A = pd.read_csv(DATA / "cells_by_seed.csv", index_col=0, header=[0, 1, 2])
    L = pd.read_csv(DATA / "cells_by_seed_lambda0.csv", index_col=0, header=[0, 1, 2])
    T = pd.read_csv(DATA / "cells_standard.csv", index_col=[0, 1, 2])
    tasks = ["T1 peers", "T3 groups", "T4 probe"]
    fig, axes = plt.subplots(3, 4, figsize=(10, 6.6))
    rng = np.random.default_rng(0)
    for i, t in enumerate(tasks):
        for j, qn in enumerate(ue.QUANTITIES):
            ax = axes[i, j]
            c = (SET, t, qn)
            a, l = A[c].dropna(), L[c].dropna()
            ax.scatter(0 + rng.uniform(-0.12, 0.12, len(a)), a, s=22, color=COLOR["standard"], edgecolor=SURFACE,
                       linewidth=1, zorder=3)
            ax.scatter(1 + rng.uniform(-0.12, 0.12, len(l)), l, s=22, color=COLOR["λ 0"], edgecolor=SURFACE,
                       linewidth=1, zorder=3)
            for x, ref in ((2, ue.RAW_FP), (3, "PCA-32 of fingerprint + window deltas")):
                ax.hlines(T.loc[c, ref], x - 0.25, x + 0.25, color=COLOR[ref], linewidth=2.5, zorder=3)
            ax.set_xticks(range(4), ["model", "λ 0", "raw fp", "PCA-32"] if i == 2 else [""] * 4, fontsize=7)
            ax.set_xlim(-0.5, 3.5)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{t[3:]}: {qn}", fontsize=8.5)
            ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"Held-out scores of every seed ({len(A)} seeds of the model, {len(L)} of λ 0) and the baselines, "
                 "unseen firms out of time", x=0.01, ha="left", fontsize=10, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f2_cells_by_seed")


def fig_agreement_by_sector() -> None:
    F = pd.read_csv(DATA / "seed_agreement_by_firm.csv")
    F = F[F.set.eq(SET)]
    F["sector"] = F.sic.map(sector_group)
    s = F.groupby("sector")["seed agreement"].agg(["mean", "size"]).query("size >= 200").sort_values("mean")
    F["size decile"] = pd.qcut(F["size"], 5, labels=["smallest", "2", "3", "4", "largest"])
    z = F.groupby("size decile", observed=True)["seed agreement"].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    a1.barh(range(len(s)), s["mean"], height=0.55, color=COLOR["standard"], zorder=2)
    a1.set_yticks(range(len(s)), s.index)
    for i, v in enumerate(s["mean"]):
        a1.text(v + 0.01, i, f"{v:.2f}", va="center", color=INK2, fontsize=8)
    a1.set_xlim(0, 1)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("share of the 10 nearest firms shared by two seeds")
    a1.set_title("Seed agreement by sector")
    a2.bar(range(len(z)), z.values, width=0.55, color=COLOR["standard"], zorder=2)
    a2.set_xticks(range(len(z)), z.index)
    for i, v in enumerate(z.values):
        a2.text(i, v + 0.01, f"{v:.2f}", ha="center", color=INK2, fontsize=8)
    a2.set_ylim(0, 1)
    a2.grid(axis="x", visible=False)
    a2.set_xlabel("size quintile (log total assets)")
    a2.set_title("…and by size")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f3_agreement_by_sector")


# --- 2. the trade-off ---------------------------------------------------------------------------------------------

def fig_frontier() -> None:
    M, B = ue.collect()
    grp = "panels_2026-10-25_filled_tin12_2023Q3"
    M, B = M[M.group.eq(grp) & M.set.eq(SET)], B[B.group.eq(grp) & B.set.eq(SET)]
    x_cell, y_cell = ("T5 holes", "10-NN overlap, 20 % of last quarter missing"), ("T1 peers", "asset growth")
    pick = lambda D, c: D[D.task.eq(c[0]) & D.quantity.eq(c[1])]
    import stability_eval as se
    fams = {"mask 0": "unmasked", "mask 0.1": "mask 0.1", "mask 0.2": "standard", "mask 0.3": "mask 0.3",
            "mask 0.4": "mask 0.4", "λ 0": "λ 0", "λ 0.25": "λ 0.25", "λ 0.75": "λ 0.75"}
    pts = {}
    for label, f in fams.items():
        rs = set(se.runs_of(f, scored=True))
        sub = M[M.run.isin(rs)]
        if sub.empty:
            continue
        xv, yv = pick(sub, x_cell).groupby("run").value.mean(), pick(sub, y_cell).groupby("run").value.mean()
        pts[label] = (xv.mean(), yv.mean(), xv.std(), yv.std(), len(xv))
    bx, by = pick(B, x_cell).set_index("vector").value, pick(B, y_cell).set_index("vector").value
    shown = {ue.RAW_FP: "raw fingerprint", "window-mean fingerprint (32)": "12-quarter mean", ue.BOTH: "fingerprint + deltas",
             "PCA-32 of fingerprint + window deltas": "PCA-32 of fp + deltas", "PCA-16 of input window": "PCA of input window",
             "window deltas": "deltas only", "PCA-16 of fingerprint": "PCA-16 of fingerprint"}
    ens = DATA / "ensemble_cells.csv"
    E = pd.read_csv(ens, index_col=[0, 1, 2]) if ens.exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    paths = (("input mask share (λ 0.5)", ["mask 0", "mask 0.1", "mask 0.2", "mask 0.3", "mask 0.4"], COLOR["standard"]),
             ("λ (mask 0.2); squares: other refinements", ["λ 0", "λ 0.25", "mask 0.2", "λ 0.75"], COLOR[ue.RAW_FP]))
    for ax, (title, keys, color) in zip(axes, paths):
        for v, lab in shown.items():
            if v in bx and v in by:
                ax.scatter(bx[v], by[v], s=36, color=GRAY, edgecolor=SURFACE, linewidth=2, zorder=3)
                off = {"PCA-16 of input window": (-6, 7), "window-mean fingerprint (32)": (-6, -12)}.get(v, (6, -3))
                ax.annotate(lab, (bx[v], by[v]), xytext=off, textcoords="offset points", color=INK2, fontsize=8,
                            ha="right" if off[0] < 0 else "left")
        ks = [k for k in keys if k in pts]
        ax.plot([pts[k][0] for k in ks], [pts[k][1] for k in ks], color=color, linewidth=2, zorder=2)
        for k in ks:
            x, y, sx, sy, n = pts[k]
            std = k == "mask 0.2"
            ax.errorbar(x, y, xerr=sx, yerr=sy, fmt="o", color=COLOR["standard"] if std else color,
                        markersize=9 if std else 6, markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1,
                        capsize=0, zorder=5 if std else 4)
            lab = ("standard: " + k.replace("mask 0.2", "mask 0.2, λ 0.5")) if std else k
            off = (-9, 9) if std else {"mask 0.1": (7, -13), "λ 0": (7, -10)}.get(k, (7, 6))
            ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", color=INK,
                        fontsize=8, ha="right" if std else "left")
        if ax is axes[1]:                                             # other refinements of the standard
            for k in ("decay 0.8", "z 64"):
                rs = set(se.runs_of(k, scored=True))
                sub = M[M.run.isin(rs)]
                if sub.empty:
                    continue
                xv, yv = pick(sub, x_cell).groupby("run").value.mean(), pick(sub, y_cell).groupby("run").value.mean()
                ax.errorbar(xv.mean(), yv.mean(), xerr=xv.std(), yerr=yv.std(), fmt="s", color=COLOR[ue.BOTH],
                            markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1, capsize=0, zorder=4)
                off = (8, 2) if k == "decay 0.8" else (-10, -16)
                ax.annotate(f"{k} ({len(xv)} seeds)", (xv.mean(), yv.mean()), xytext=off, textcoords="offset points",
                            color=INK, fontsize=8, ha="left" if off[0] > 0 else "right")
        if E is not None:
            ex, ey = E.loc[(SET, *x_cell), "ensemble of 5 seeds"], E.loc[(SET, *y_cell), "ensemble of 5 seeds"]
            ax.scatter(ex, ey, s=80, marker="D", color=COLOR["standard"], edgecolor=SURFACE, linewidth=2, zorder=6)
            ax.annotate("standard, ensemble of 5 seeds", (ex, ey), xytext=(-9, 9), textcoords="offset points",
                        color=INK, fontsize=8, ha="right")
        ax.set_title(f"Varying the {title}")
        ax.set_xlabel("peers kept when 20 % of the latest items are missing (T5)")
    axes[0].set_ylabel("asset-growth peers R² (T1)")
    fig.suptitle("Current state vs robustness, unseen firms out of time (bars: ±1 sd over seeds)", x=0.01, ha="left",
                 fontsize=10, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f4_frontier")


def fig_lambda() -> None:
    M, _ = ue.collect()
    grp = "panels_2026-10-25_filled_tin12_2023Q3"
    M = M[M.group.eq(grp) & M.set.eq(SET) & M.task.eq("T1 peers") & M.quantity.eq("size")]
    M = M[M.pair_loss.eq("mse") & M.pillar1.eq("last") & M.t_in.eq(12) & M.hidden.eq(128)]
    import stability_eval as se
    excl = {r for f in ("data 80 %", "40 epochs") for r in se.runs_of(f)}
    M = M[~M.run.isin(excl)]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for (z, mask), color, lab in (((16, 0.0), COLOR["unmasked"], "z 16, no mask"), ((32, 0.2), COLOR["standard"], "z 32, mask 0.2")):
        sub = M[M.z.eq(z) & M.input_mask.eq(mask)].groupby("lambda")[["fit", "fit2"]].agg(["mean", "std"])
        if sub.empty:
            continue
        ax.plot(sub["fit"]["mean"], sub["fit2"]["mean"], "-o", color=color, label=lab, markeredgecolor=SURFACE)
        left = z == 32                                                # the two paths overlap: labels on opposite sides
        for lam, r in sub.iterrows():
            ax.annotate(f"λ {lam:g}", (r["fit"]["mean"], r["fit2"]["mean"]), xytext=(-7 if left else 6, -3 if left else 4),
                        textcoords="offset points", fontsize=8, color=INK2, ha="right" if left else "left")
    ax.set_xlabel("fit to the structure target (Spearman, Pillar 1)")
    ax.set_ylabel("fit to the moves target (Spearman, Pillar 2)")
    ax.set_title("The λ trade-off between the two targets (unseen firms, out of time)")
    ax.legend(loc="lower left")
    save(fig, "f5_lambda")


# --- 3. literature-style curves -----------------------------------------------------------------------------------

LINES = ["standard", "λ 0", ue.RAW_FP, "window-mean fingerprint (32)", "PCA-32 of fingerprint + window deltas"]


def fig_k_curve() -> None:
    C = pd.read_csv(DATA / "curve_k.csv")
    C = C[C.set.eq(SET)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True)
    for ax, qn in zip(axes, ue.QUANTITIES):
        sub = C[C.quantity.eq(qn)]
        for fam in LINES:
            s = sub[sub.family.eq(fam)].groupby(["vector", "k"]).r2.mean().groupby("k").mean()
            if len(s):
                ax.plot(s.index, s.values, "-o", color=COLOR[fam], label=LABEL[fam], markersize=3.5)
        sic = sub[sub.family.eq("SIC2")].r2.mean()
        ax.axhline(sic, color=COLOR["SIC2"], linewidth=1.5)
        ax.annotate("SIC2 peers", (100, sic), xytext=(-2, 3), textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
        ax.set_title(qn)
        ax.set_xlabel("number of peers k")
    axes[0].set_ylabel("R² of the firm on its peers' mean")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Peer R² against the number of peers (as in Lee, Ma & Wang 2015), unseen firms out of time",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    save(fig, "f6_peers_vs_k")


def fig_group_curve() -> None:
    C = pd.read_csv(DATA / "curve_groups.csv")
    C = C[C.set.eq(SET)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True)
    for ax, qn in zip(axes, ue.QUANTITIES):
        sub = C[C.quantity.eq(qn)]
        for fam in LINES:
            s = sub[sub.family.eq(fam)].groupby("K").r2.mean()
            if len(s):
                ax.plot(s.index, s.values, "-o", color=COLOR[fam], label=LABEL[fam], markersize=3.5)
        sic = sub[sub.family.str.startswith("SIC")].groupby("family")[["K", "r2"]].mean()
        ax.scatter(sic.K, sic.r2, s=40, color=COLOR["SIC2"], edgecolor=SURFACE, linewidth=2, zorder=4, label="SIC 1–4 digits")
        for name, r in sic.iterrows():
            ax.annotate(name.replace("SIC 1-digit", "SIC1"), (r.K, r.r2), xytext=(4, -9), textcoords="offset points",
                        fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_title(qn)
        ax.set_xlabel("number of groups K")
    axes[0].set_ylabel("cross-fitted R² on the group mean")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Data-driven groups (k-means) against SIC at 1–4 digits (as in Hoberg & Phillips 2016)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    save(fig, "f7_groups_vs_K")


def fig_holes_curve() -> None:
    C = pd.read_csv(DATA / "curve_holes.csv")
    C = C[C.set.eq(SET)]
    per = C.groupby(["family", "vector", "share"]).overlap.mean().groupby(["family", "share"]).mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for fam, mask in (("unmasked", 0.0), ("mask 0.1", 0.1), ("standard", 0.2), ("mask 0.3", 0.3), ("mask 0.4", 0.4)):
        if fam in per.index.get_level_values(0):
            s = per[fam]
            a1.plot(s.index, s.values, "-o", color=BLUES[mask], label=f"mask {mask:g}" + (" (standard)" if mask == 0.2 else ""),
                    markersize=3.5)
    a1.set_title("Model, by training mask share (λ 0.5)")
    for fam in ["standard", ue.RAW_FP, "window-mean fingerprint (32)", "PCA-32 of fingerprint + window deltas", ue.BOTH]:
        if fam in per.index.get_level_values(0):
            s = per[fam]
            a2.plot(s.index, s.values, "-o", color=COLOR[fam], label=LABEL[fam], markersize=3.5)
    a2.set_title("Model and baselines")
    for ax in (a1, a2):
        ax.set_xlabel("share of the latest quarter's items missing")
        ax.legend(loc="lower left")
        ax.set_ylim(0, 1)
    a1.set_ylabel("share of the 10 nearest firms kept")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f8_holes_curve")


def fig_persistence() -> None:
    P = pd.read_csv(DATA / "persistence.csv")
    per = P.groupby(["family", "h"]).persistence.mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for fam in [f for f in LINES + [ue.BOTH] if f in per.index.get_level_values(0)]:
        s = per[fam]
        ax.plot(s.index, s.values, "-o", color=COLOR[fam], label=LABEL[fam], markersize=4)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlabel("quarters later (h)")
    ax.set_ylabel("share of the 10 nearest firms still nearest")
    ax.set_ylim(0, 1)
    ax.set_title("Peer persistence over time (unseen firms, all quarters)")
    ax.legend(loc="upper right")
    save(fig, "f9_persistence")


def fig_t6(lag: int = 1) -> None:
    S = pd.read_csv(DATA / ("t6_summary.csv" if lag == 1 else f"t6_summary_lag{lag}.csv"))
    dec = DATA / (f"t6_summary_decay0.8_decay0.6.csv" if lag == 1 else f"t6_summary_lag{lag}_decay0.8_standard.csv")
    if dec.exists():                                                  # the decay 0.8 refinement, same task
        Dd = pd.read_csv(dec)
        S = pd.concat([S, Dd[Dd.family.eq("decay 0.8")]])
    S = S[S.set.eq(SET)]
    stmts = {"bs": "balance sheet", "ins": "income statement", "cf": "cash flow"}
    fams = ["standard", "decay 0.8", "λ 0", ue.RAW_FP, ue.BOTH, "PCA-32 of fingerprint + window deltas",
            "window-mean fingerprint (32)", "SIC2", "random peers"]
    fams = [f for f in fams if f in set(S.family)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(fams)
    for i, fam in enumerate(fams):
        sub = S[S.family.eq(fam)].set_index("statement").reindex(list(stmts))
        x = np.arange(len(stmts)) + (i - (len(fams) - 1) / 2) * w
        color = {"decay 0.8": COLOR["unmasked"]}.get(fam, COLOR.get(fam, GRAY))
        label = {"decay 0.8": "model, recent moves weighted (decay 0.8)"}.get(fam, LABEL.get(fam, fam))
        ax.bar(x, sub.r2, width=w * 0.85, color=color, label=label, zorder=2,
               yerr=sub["sd"].fillna(0) if fam in ("standard", "λ 0", "decay 0.8") else None,
               error_kw={"elinewidth": 1, "ecolor": INK2})
    ax.set_xticks(range(len(stmts)), list(stmts.values()))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("R² of the true change on the peers' mean change")
    what = "quarter-on-quarter" if lag == 1 else "year-on-year (exploratory)"
    ax.set_title(f"T6: nowcasting a missing item's {what} change from the 10 nearest firms' changes\n"
                 "(unseen firms, out of time; bars: ±1 sd over seeds)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "f10_t6_nowcast" if lag == 1 else f"f10b_t6_nowcast_lag{lag}")


# --- 4. maps ------------------------------------------------------------------------------------------------------

def sector_group(sic) -> str:
    if pd.isna(sic) or sic < 0:
        return "unknown"
    sic = int(sic)
    s2 = sic if sic < 100 else sic // 100                             # 2-digit codes or full 4-digit SIC
    if s2 in (60, 61):
        return "Banks & credit"
    if 62 <= s2 <= 67:
        return "Insurance, REITs, other finance"
    if s2 == 28:
        return "Pharma & chemicals"
    if s2 in (35, 36, 38):
        return "Tech hardware & instruments"
    if s2 == 73:
        return "Software & business services"
    if s2 in (10, 12, 13, 14, 29):
        return "Energy & mining"
    if 40 <= s2 <= 47:
        return "Transport"
    if s2 in (48, 49):
        return "Telecom & utilities"
    if 50 <= s2 <= 59:
        return "Wholesale & retail"
    if 20 <= s2 <= 39:
        return "Other manufacturing"
    return "Other services & other"


def fig_maps() -> None:
    M = pd.read_csv(DATA / "map_coords.csv")
    M["sector"] = M.sic.map(sector_group)
    show = ["Banks & credit", "Insurance, REITs, other finance", "Pharma & chemicals", "Tech hardware & instruments",
            "Software & business services", "Energy & mining"]
    vecs = [v for v in ["standard", ue.RAW_FP] if v in set(M.vector)]
    fig, axes = plt.subplots(len(vecs), len(show), figsize=(2.1 * len(show), 2.3 * len(vecs)))
    axes = np.atleast_2d(axes)
    for i, v in enumerate(vecs):
        sub = M[M.vector.eq(v)]
        for j, sec in enumerate(show):
            ax = axes[i, j]
            ax.scatter(sub.x, sub.y, s=2, color=GRID, linewidth=0, rasterized=True)
            h = sub[sub.sector.eq(sec)]
            ax.scatter(h.x, h.y, s=4, color=COLOR["standard"] if i == 0 else COLOR[ue.RAW_FP], linewidth=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(sec, fontsize=8)
            if j == 0:
                ax.set_ylabel(LABEL[v], fontsize=8, color=INK2)
    fig.suptitle(f"t-SNE of all firms, {M.quarter.iloc[0]}: each panel highlights one sector", x=0.01, ha="left",
                 fontsize=10, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f11_maps")


def _quarter_shift(q: str, h: int) -> str:
    n = int(q[:4]) * 4 + int(q[-1]) - 1 + h
    return f"{n // 4}Q{n % 4 + 1}"


def fig_cases(tag: str = "") -> None:
    C = pd.read_csv(DATA / f"case_paths{tag}.csv")
    cases = list(dict.fromkeys(C.case))
    vecs = [v for v in ["standard", ue.RAW_FP] if v in set(C.vector)]
    fig, axes = plt.subplots(len(cases), len(vecs), figsize=(5.2 * len(vecs), 2.6 * len(cases)), sharey="row", squeeze=False)
    for i, case in enumerate(cases):
        anchor, q = case.split()
        for j, v in enumerate(vecs):
            ax = axes[i, j]
            sub = C[C.case.eq(case) & C.vector.eq(v)]
            for (role, t), g in sub.groupby(["role", "ticker"], sort=False):
                g = g.sort_values("lag", ascending=False)
                x = [_quarter_shift(q, -l) for l in g.lag]
                if role == "anchor":
                    ax.plot(range(len(x)), g.value, color=INK, linewidth=2.5, zorder=4)
                else:
                    ax.plot(range(len(x)), g.value, color=COLOR[v], linewidth=1.2, alpha=0.85, zorder=3)
                ax.annotate(t, (len(x) - 1, g.value.iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                            color=INK if role == "anchor" else INK2, va="center")
            ax.set_xticks(range(0, len(x), 2), x[::2], fontsize=7)
            ax.set_title(f"{anchor}, {q}: 5 nearest by the {'model' if v == 'standard' else 'raw fingerprint'}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{sub.ratio.iloc[0]} (standardised)", fontsize=8)
    fig.suptitle("A firm (black) and its 5 nearest firms: the path of one ratio over the 12-quarter window"
                 + (" (rank-scaled fingerprint)" if tag else ""),
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f12_cases" + ("b_rank" if tag else ""))


def fig_geometry() -> None:
    G = pd.read_csv(DATA / "geometry.csv")
    from scipy.stats import spearmanr
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, lab in ((axes[0], "fingerprint", "fingerprint (Pillar 1 target)"),
                         (axes[1], "deltas", "window deltas (Pillar 2 target)")):
        rho = spearmanr(G[col], G.latent)[0]
        hb = ax.hexbin(G[col], G.latent, gridsize=70, cmap="Blues", mincnt=1, bins="log", linewidths=0)
        ax.set_xlabel(f"cosine, {lab}")
        ax.set_ylabel("cosine, embedding")
        ax.set_title(f"Spearman {rho:.2f}")
        ax.grid(False)
    fig.colorbar(hb, ax=axes, label="pairs (log)", shrink=0.8)
    fig.suptitle(f"Embedding geometry against the two targets: pairs of firms, {G.quarter.iloc[0]}",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=INK)
    save(fig, "f13_geometry")


def fig_importance(tag: str = "") -> None:
    I = pd.read_csv(DATA / f"importance{tag}.csv")
    I = I[I["mode"].eq("whole window")]
    m = I.groupby(["pillar", "line"]).increase.mean().unstack("pillar")
    sd = I.groupby(["pillar", "line"]).increase.std().unstack("pillar")
    rel = m / m.clip(lower=0).sum()                                    # share of each pillar's total increase
    rel_sd = sd / m.clip(lower=0).sum()
    order = rel.sum(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(8, 8.5))
    y = np.arange(len(order))
    for k, (pil, color) in enumerate((("Pillar 1", COLOR["standard"]), ("Pillar 2", COLOR[ue.RAW_FP]))):
        ax.barh(y + (k - 0.5) * 0.38, rel.loc[order, pil], height=0.36, color=color,
                xerr=rel_sd.loc[order, pil], error_kw={"elinewidth": 0.8, "ecolor": INK2},
                label={"Pillar 1": "structure (Pillar 1)", "Pillar 2": "moves (Pillar 2)"}[pil], zorder=2)
    ax.set_yticks(y, order, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of the pillar's total loss increase when the line is blanked over the window")
    what = " (rank-scaled fingerprint)" if tag else ""
    ax.set_title(f"Which statement lines the embedding relies on, per pillar{what} (3 seeds, ±1 sd)")
    ax.legend(loc="lower right")
    save(fig, "f14_importance" + ("b_rank" if tag else ""))


def main() -> None:
    style()
    for f in (fig_seed_agreement, fig_cells_by_seed, fig_agreement_by_sector, fig_frontier, fig_lambda, fig_k_curve,
              fig_group_curve, fig_holes_curve, fig_persistence, fig_t6, lambda: fig_t6(4), fig_maps, fig_cases,
              fig_geometry, fig_importance, lambda: fig_importance("_rankfingerprint"),
              lambda: fig_cases("_rankfingerprint")):
        try:
            f()
        except FileNotFoundError as e:
            print(f"[WARN] {getattr(f, '__name__', 'figure')}: {e.filename} missing")
