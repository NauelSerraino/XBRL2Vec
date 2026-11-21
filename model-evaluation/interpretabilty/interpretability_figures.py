"""
Interpretability figures of the paper model, the "paper" variant of main_similarity.VARIANTS (protocol addendum 16:
the 2,115 firms with full data 2013Q1-2025Q4, 317 unseen, cutoff 2022Q4; z 32, mask 0.2, λ 0.5, signed-log
fingerprint, decay 0.8). The maps use seed 42; the importance the 10 seeds (dot: mean, line: range), on the unseen
firms after the cutoff (2023Q1-2025Q4).
Data and helpers are the scoring's own (src/usefulness_eval.py).

    xbrl2vec/bin/python model-evaluation/interpretabilty/interpretability_figures.py

Output in out/ next to this script, NN_<name>.png and NN_<name>.csv (the numbers behind the figure):
  01 map_sectors          t-SNE of the 2,115 firms in 2025Q4, one panel per large sector
  02a map_colour, 02b map_dots   the same map with each firm's size, asset growth and revenue growth averaged over its
                          12-quarter window, as colour (a) and as dot area (b), from the 2nd to the 98th percentile
  03 importance_items     which ratios move the embedding: Gaussian noise on one ratio, the same in all 12 quarters
                          (left), and each statement's importance per ratio (right)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parents[1] / "src"))
import usefulness_eval as ue           # noqa: E402
from services.similarity import quarter_shift  # noqa: E402
import matplotlib                      # noqa: E402
matplotlib.use("Agg")                  # files only, no window
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.manifold import TSNE      # noqa: E402

SET, QUARTER, BLUE = "unseen firms, out of time", "2025Q4", "#2a78d6"
WINDOW = f"{quarter_shift(QUARTER, -11)}–{QUARTER}"                 # the 12 quarters behind each 2025Q4 embedding
PERIOD = "2023Q1–2025Q4"                                                 # the unseen firms after the cutoff (SET)
QUANTS = ["size", "asset growth", "revenue growth"]
STMT = {"bs": ("balance sheet", "#2a78d6"), "ins": ("income statement", "#eb6834"), "cf": ("cash flow", "#1baf7a")}
MARK = {"balance sheet": "o", "income statement": "s", "cash flow": "^"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
REDS = LinearSegmentedColormap.from_list("reds", ["#fbe2e1", "#f3a9a8", "#ea7270", "#e34948", "#b8302f", "#7f1d1c"])
GREENS = LinearSegmentedColormap.from_list("greens", ["#ddf1dd", "#a6d9a6", "#62bb62", "#2a9a2a", "#008300", "#005200"])


def style() -> None:
    """The house style of the figures: light surface, no top and right spines, a light grid, sans-serif 9 pt."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "normal", "axes.titlelocation": "center",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "lines.markersize": 5,
    })


def sector_group(sic) -> str:
    """A coarse sector (11 groups) from a SIC code, 2- or 4-digit; "unknown" without one."""
    if pd.isna(sic) or sic < 0:
        return "unknown"
    s2 = int(sic) if sic < 100 else int(sic) // 100
    groups = [((60, 61), "Banks & credit"), (range(62, 68), "Insurance, REITs, other finance"), ((28,), "Pharma & chemicals"),
              ((35, 36, 38), "Tech hardware & instruments"), ((73,), "Software & business services"),
              ((10, 12, 13, 14, 29), "Energy & mining"), (range(40, 48), "Transport"), ((48, 49), "Telecom & utilities"),
              (range(50, 60), "Wholesale & retail"), (range(20, 40), "Other manufacturing")]
    return next((name for codes, name in groups if s2 in codes), "Other services & other")


def save(fig, name: str, data: pd.DataFrame | None) -> None:
    """out/<name>.png and, when given, the numbers behind it in out/<name>.csv."""
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    if data is not None:
        data.to_csv(OUT / f"{name}.csv", index=False)
    print(f"[INFO] {name}", flush=True)


def title(fig, text: str, note: str = "") -> float:
    """A short centred title and, under it, an optional reading note in small grey type, at a fixed height in points
    whatever the figure's size; returns the top of the room left for the panels (figure fraction)."""
    pt = 72 * fig.get_figheight()
    fig.suptitle(text, fontsize=11, color=INK, y=1 - 4 / pt, va="top")
    if note:                                                      # a figure text: the panels' layout ignores it
        fig.text(0.5, 1 - 21 / pt, note, ha="center", va="top", fontsize=8.5, color=INK2)
    return 1 - (38 if note else 22) / pt


def bare(ax) -> None:
    """No ticks, grid or spines: for maps, whose axes have no units."""
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def cosr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine."""
    return (ue._unit(A) * ue._unit(B)).sum(1)


def dotplot(ax, D: pd.DataFrame, labels: list, xlim: tuple) -> None:
    """One row per entry of D (mean, min, max, statement): a dot for the mean, a faint line for the range."""
    col, y = dict(STMT.values()), np.arange(len(D))
    ax.hlines(y, *xlim, color="#ececec", lw=0.6, zorder=0)
    ax.hlines(y, D["min"], D["max"], color=[col[s] for s in D.statement], lw=1.6, alpha=0.45)
    for st, m in MARK.items():
        k = (D.statement == st).to_numpy()
        ax.scatter(D["mean"][k], y[k], marker=m, s=28, color=col[st], edgecolor="white", lw=0.6, zorder=3,
                   label=st.capitalize())
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_ylim(-0.8, len(D) - 0.2)
    ax.set_xlim(*xlim)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)


# --- 1-2. the map -----------------------------------------------------------------------------------------------------

def maps(g, U) -> None:
    """Figures 01, 02a, 02b: a t-SNE of the unit embeddings U of every firm in QUARTER (seed 42)."""
    pool = ue.pool_of(g, QUARTER)
    xy = TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=0).fit_transform(U[pool])
    tick = g.meta.ticker.iloc[pool].to_numpy()
    win = g.meta.quarter.between(quarter_shift(QUARTER, -11), QUARTER).to_numpy()   # the 12 quarters of the window
    avg = g.Q[QUANTS][win].groupby(g.meta.ticker[win].to_numpy()).mean().reindex(tick)  # each firm's mean over them
    M = pd.DataFrame({"ticker": tick, "firms": g.meta.firms.iloc[pool].to_numpy(),
                      "x": xy[:, 0], "y": xy[:, 1], "sic2": g.sic[pool], "sector": [sector_group(s) for s in g.sic[pool]],
                      **{k: avg[k].to_numpy(float) for k in QUANTS}})
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, sec in zip(axes.flat, M.sector.value_counts().index[:6]):
        h = M.sector.eq(sec)
        ax.scatter(M.x[~h], M.y[~h], s=3, color=GRID, lw=0, rasterized=True)
        ax.scatter(M.x[h], M.y[h], s=5, color=BLUE, lw=0, rasterized=True)
        ax.set_title(f"{sec} (n = {h.sum()})", fontsize=9)
        bare(ax)
    room = title(fig, f"Firm embeddings by sector (t-SNE), {WINDOW}",
                 f"One point per firm: its embedding from the 12-quarter window {WINDOW}; seed 42")
    fig.tight_layout(rect=(0, 0, 1, room))
    save(fig, "01_map_sectors", M)

    panels = [("size", BLUE, SEQ, "Size (log total assets)"),
              ("asset growth", "#e34948", REDS, "Asset growth (log, year on year)"),
              ("revenue growth", "#008300", GREENS, "Revenue growth (log, year on year)")]
    for name, style in [("02a_map_colour", "colour"), ("02b_map_dots", "dot area")]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (k, color, cmap, label) in zip(axes, panels):
            v = M[k].to_numpy()
            ok = np.isfinite(v)
            lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)                    # beyond: drawn at the ends
            c = np.clip(v, lo, hi)
            ax.scatter(M.x[~ok], M.y[~ok], s=3, color=GRID, lw=0, rasterized=True)
            if style == "colour":
                sc = ax.scatter(M.x[ok], M.y[ok], c=c[ok], cmap=cmap, vmin=lo, vmax=hi, s=5, lw=0, rasterized=True)
                fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01).outline.set_visible(False)
            else:
                area = lambda x: 1 + 45 * (x - lo) / (hi - lo)
                o = np.flatnonzero(ok)[np.argsort(v[ok])]                                 # the biggest dots on top
                ax.scatter(M.x.iloc[o], M.y.iloc[o], s=area(c[o]), color=color, alpha=0.45, lw=0, rasterized=True)
                ax.legend(handles=[Line2D([], [], marker="o", ls="", markersize=np.sqrt(area(x)), color=color, alpha=0.45,
                                          markeredgewidth=0, label=f"{x:.2f}") for x in (lo, np.nanmedian(v), hi)],
                          title="2nd percentile / median / 98th", loc="lower left", fontsize=7.5, title_fontsize=7.5,
                          labelspacing=1.2)
            ax.set_title(label, fontsize=9)
            bare(ax)
        room = title(fig, f"Size and growth on the embedding map, {WINDOW}",
                     f"Embeddings and each firm's mean of the quantity over the same 12 quarters, {WINDOW}; {style} from "
                     f"the 2nd to the 98th percentile; grey: not observed")
        fig.tight_layout(rect=(0, 0, 1, room))
        save(fig, name, M if style == "colour" else None)


# --- 3. importance ----------------------------------------------------------------------------------------------------

def importance(g, enc: list, Z: list, ratios: list[str], stmt: dict) -> None:
    """Figure 03: how far each ratio's noise moves the embeddings (1 - cosine), per ratio and per statement."""
    rows = np.flatnonzero(g.focal[SET])
    X0 = g.X[rows]
    assert np.allclose(cosr(Z[0][rows], enc[0](X0)), 1, atol=1e-5), "the embeddings are reproduced without noise"
    # the same draw in all 12 quarters: the ratio's level shifts, no fake quarter-to-quarter jumps
    noise = torch.as_tensor(np.random.default_rng(0).standard_normal((len(rows), 1)), dtype=X0.dtype)
    per_seed = {}
    for j, c in enumerate(ratios):
        X = X0.clone()
        X[:, :, int(g.xcol[j])] += noise
        per_seed[c] = np.array([(1 - cosr(Z[s][rows], e(X))).mean() for s, e in enumerate(enc)])
    I = pd.DataFrame(per_seed).agg(["mean", "min", "max"]).T.sort_values("mean")
    I["statement"] = [STMT[stmt[c]][0] for c in I.index]
    by = pd.DataFrame(per_seed).T.groupby(lambda c: STMT[stmt[c]][0])                  # statement x seeds
    A = by.mean().agg(["mean", "min", "max"], axis=1).assign(ratios=by.size()).sort_values("mean", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.2))
    fig.subplots_adjust(left=0.33, right=0.98, bottom=0.11, top=0.93, wspace=0.62)
    xmax = 1.1 * max(I["max"].max(), A["max"].max())
    B = A.iloc[::-1].assign(statement=A.index[::-1])                                     # the most important on top
    for ax, D, labels, name in [(axes[0], I, list(I.index), "Ratios"),
                                 (axes[1], B, [f"{k.capitalize()} ({n})" for k, n in B.ratios.items()],
                                  "Statements, mean per ratio")]:
        dotplot(ax, D, labels, (0, xmax))
        ax.set_xlabel(r"$1-\cos(z,\tilde{z})$", fontsize=9)
        ax.set_title(rf"$\bf{{{'ab'[D is B]}}}$   {name}", fontsize=9.5)
    axes[1].legend(loc="lower right", fontsize=7.5, handletextpad=0.3)
    room = title(fig, f"Input importance, {PERIOD}", "Gaussian noise (sd 1) on one ratio, the same in all 12 quarters; "
                 "unseen firms; mean over 10 seeds (line: range)")
    fig.subplots_adjust(top=room - 22 / (72 * fig.get_figheight()))
    save(fig, "03_importance_items", pd.concat([I.reset_index(names="ratio"),
                                                A.reset_index(names="ratio").assign(statement="(per ratio, whole statement)")]))


def main() -> None:
    """The 10 seeds' embeddings of every window, then the figures."""
    style()
    runs = ue.runs_of("paper")
    assert len(runs) == 10, len(runs)
    g = ue.Group(runs[0])
    enc = [g.encoder(r) for r in runs]
    Z = [e(g.X) for e in enc]
    stmt = {line: s for s in STMT for line in pd.read_parquet(g.panel / f"{s}-raw.parquet", columns=["features"]).features}
    maps(g, ue._unit(Z[0]))
    importance(g, enc, Z, g.ratios, stmt)


if __name__ == "__main__":
    main()
