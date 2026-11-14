"""
Targets of Pillars 2 and 3: one vector per firm-quarter describing how firms like it fared in each macro episode
(Pillar 2) and how they move with the market and their sector (Pillar 3). They are Y for the network, like the
fingerprint similarity of Pillar 1: firms with similar vectors should end up close in the embedding.

    python src/preprocess/pillar_targets.py --panel data/in/panels_2026-10-25/filled \
        --metadata data/in/download_2026-10-25/metadata.parquet

Kept stupid simple:
  groupings   sector (SIC 2-digit, or the SIC division for sectors with fewer than 20 firms), and deciles of size
              (log assets), net leverage ((debt - cash) / assets), liquidity (cash / assets).
  Pillar 2    the change of 7 levels (log revenue, log assets, ROA, operating cash flow / assets, interest expense /
              assets, debt / assets, cash / assets; flows over the trailing 12 months) over each macro episode:
              the two rate-hike cycles, the pandemic shock, the reopening, the rate cuts (EPISODES, dated on the Fed
              funds rate and GDP).
  Pillar 3    the correlation of each firm's year-on-year changes of the same 7 lines with two benchmarks: the
              market (median over all firms) and its sector (median over the firms of its sector).
  test        one firm's value is noise; a group's is not. A component (level x episode x grouping, or outcome x
              benchmark x grouping) is kept when the mean value of a group, estimated on a random half of the firms,
              predicts the values of the other half's firms (out-of-sample R², 10 splits) better than in every
              placebo (at least MIN_PLACEBOS of them, so p < 0.05), and R² >= MIN_R2. Placebos: Pillar 2, the same change over every window of the same length
              that shares no quarter with the episode or with the pandemic (does the episode stand out from
              ordinary times?); Pillar 3, the benchmark shifted in time.
  vectors     a firm's value for a kept component is its group's mean (shrunk towards the overall mean), group taken
              from the firm's values four quarters earlier (sector fixed); columns standardised.

Pillar 3 is estimated without the pandemic quarters (COVID): one shock of that size dominates every correlation.
Its stability is reported (components.csv, not used to select): the group means estimated before and after the
pandemic separately, correlated across groups, against the same with the benchmarks shifted.

Data quality: total assets and revenue go through despike() (early filings alternate between the true value and a
wrong tag's); negative revenue is dropped; levels are winsorised per quarter.

Output in --out (default <panel>/targets): pillar2_episodes.parquet, pillar3_comovement.parquet (index ticker,
quarter), components.csv (every component, its R², placebo and, for Pillar 3, stability), exposures.parquet (every
group's mean for every component).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPINGS = ["sector", "size", "net leverage", "liquidity"]
BENCHMARKS = ["market", "sector"]
#: (start, end) quarters: the level at the end minus the level at the start
EPISODES = {
    "rate hikes 2017-18": ("2016Q4", "2018Q4"),     # Fed funds 0.4 -> 2.2 % (7 of the 9 hikes of 2015-18)
    "pandemic 2020": ("2019Q4", "2020Q4"),          # GDP -9 % (annualised) in 2020Q2; Fed funds 1.6 -> 0.1 %
    "reopening 2021": ("2020Q4", "2021Q4"),
    "rate hikes 2022-23": ("2021Q4", "2023Q4"),     # Fed funds 0.1 -> 5.3 %
    "rate cuts 2024-25": ("2024Q2", "2025Q4"),      # Fed funds 5.3 -> 3.9 %
}
COVID = ("2020Q1", "2021Q4")
MIN_SECTOR_FIRMS = 20
MIN_QUARTERS = 24            # quarters needed for a firm's co-movement
MIN_QUARTERS_PERIOD = 12     # the same, in each half of the stability check
MIN_R2 = 0.02
MIN_PLACEBOS = 19            # beating every one of at least 19 placebos: p < 0.05
SHRINK = 10.0                # group mean shrunk towards the overall mean with weight n / (n + SHRINK)
N_SPLITS = 10
DIVISIONS = [(1, 9, "Agriculture"), (10, 14, "Mining"), (15, 17, "Construction"), (20, 39, "Manufacturing"),
             (40, 49, "Transport & utilities"), (50, 51, "Wholesale"), (52, 59, "Retail"), (60, 67, "Finance"),
             (70, 89, "Services"), (91, 99, "Public administration")]


def load_lines(panel: Path) -> dict[str, pd.DataFrame]:
    """{line: ticker x quarter} from the {bs,ins,cf}-raw.parquet of a panel."""
    wide = pd.concat([pd.read_parquet(panel / f"{s}-raw.parquet") for s in ("bs", "ins", "cf")], ignore_index=True)
    quarters = sorted(c for c in wide.columns if len(str(c)) == 6 and str(c)[4] == "Q")
    lines = {k: g.set_index("ticker")[quarters] for k, g in wide.groupby("features")}
    tickers = lines["Total Assets"].index
    return {k: v.reindex(tickers).astype(float) for k, v in lines.items()}


def despike(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """
    Values more than `factor` times away from the median of the firm's own five quarters around them (t-2..t+2) set
    missing (and zeros and negatives): total assets and revenue are smooth, and early filings (2008-2012) often
    alternate between the true value and a wrong tag's (e.g. 14,068 / 1,145 / 14,027).
    """
    x = df.where(df > 0)
    med = x.T.rolling(5, center=True, min_periods=3).median().T
    return x.where(~(np.abs(np.log(x / med)) > np.log(factor)))


def _winsorize(df: pd.DataFrame) -> pd.DataFrame:
    """Clip each quarter's cross-section at its 1st and 99th percentiles."""
    return df.clip(df.quantile(0.01), df.quantile(0.99), axis=1)


def levels(L: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The 7 levels whose changes describe a firm: log revenue and log assets, and five ratios to assets."""
    ta, rev = despike(L["Total Assets"]), despike(L["Total Revenue"])
    ttm = lambda df: df.T.rolling(4, min_periods=4).sum().T
    debt = L["Long Term Debt"] + L["Current Debt"]
    out = {"log revenue": np.log(ttm(rev)),
           "log assets": np.log(ta),
           "ROA": ttm(L["Net Income"]) / ta,
           "operating cash flow / assets": ttm(L["Operating Cash Flow"]) / ta,
           "interest expense / assets": ttm(L["Interest Expense"]) / ta,
           "debt / assets": debt / ta,
           "cash / assets": L["End Cash Position"] / ta}
    return {k: _winsorize(v.replace([np.inf, -np.inf], np.nan)) for k, v in out.items()}


def changes(V: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """firms: level at quarter b minus level at quarter a, winsorised."""
    return _winsorize((V[b] - V[a]).to_frame()).iloc[:, 0].to_numpy(float)


def yoy(Lv: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Year-on-year changes of the levels (Pillar 3)."""
    return {k: _winsorize(v - v.shift(4, axis=1)) for k, v in Lv.items()}


def sector_labels(sic: pd.Series) -> pd.Series:
    """SIC 2-digit code, or the SIC division for 2-digit sectors with fewer than MIN_SECTOR_FIRMS firms."""
    s2 = pd.to_numeric(sic, errors="coerce") // 100
    div = s2.map(lambda s: None if pd.isna(s) else next((n for a, b, n in DIVISIONS if a <= s <= b), None))
    big = s2.map(s2.value_counts()) >= MIN_SECTOR_FIRMS
    return pd.Series(np.where(big, s2.astype("Int64").astype(str), "div " + div.astype(str)), index=sic.index).where(div.notna())


def memberships(L: dict[str, pd.DataFrame], sic: pd.Series) -> dict[str, pd.DataFrame]:
    """{grouping: ticker x quarter labels}; deciles of each quarter's cross-section four quarters earlier."""
    ta = despike(L["Total Assets"])
    debt, cash = L["Long Term Debt"] + L["Current Debt"], L["End Cash Position"]
    chars = {"size": np.log(ta), "net leverage": (debt - cash) / ta, "liquidity": cash / ta}
    sector = sector_labels(sic.reindex(ta.index))
    mem = {"sector": pd.DataFrame({q: sector for q in ta.columns})}
    for name, c in chars.items():
        c = c.shift(4, axis=1)
        mem[name] = c.apply(lambda col: pd.qcut(col, 10, labels=False, duplicates="drop").map(lambda v: f"d{int(v) + 1}", na_action="ignore")
                            if col.notna().sum() >= 100 else pd.Series(np.nan, index=col.index, dtype=object))
    return mem


def modal_group(mem: pd.DataFrame) -> pd.Series:
    """The group each firm belongs to most often (NaN if never assigned)."""
    s = mem.stack().rename("g").reset_index()
    return s.groupby(s.columns[0]).g.agg(lambda g: g.value_counts().index[0]).reindex(mem.index)


def firm_comovements(Y: np.ndarray, m: np.ndarray, min_obs: int = MIN_QUARTERS) -> np.ndarray:
    """firms: correlation of each firm's series with a benchmark: one series m for all, or one row per firm."""
    M = np.broadcast_to(m, Y.shape)
    ok = np.isfinite(Y) & np.isfinite(M)
    n = ok.sum(axis=1)
    mean = lambda V: np.where(ok, V, 0.0).sum(axis=1) / np.maximum(n, 1)
    yc, mc = np.where(ok, Y - mean(Y)[:, None], 0.0), np.where(ok, M - mean(M)[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (yc * mc).sum(axis=1) / np.sqrt((yc ** 2).sum(axis=1) * (mc ** 2).sum(axis=1))
    return np.where(n >= min_obs, r, np.nan)


def benchmark_comovements(Ya: np.ndarray, sec_codes: np.ndarray, shift: int = 0, min_obs: int = MIN_QUARTERS) -> np.ndarray:
    """firms x 2: correlation with the market and with the firm's sector, benchmarks shifted by `shift` quarters."""
    with np.errstate(all="ignore"):
        market = np.nanmedian(Ya, axis=0)
        sec_median = np.array([np.nanmedian(Ya[sec_codes == c], axis=0) for c in range(sec_codes.max() + 1)])
    sector = np.where((sec_codes >= 0)[:, None], sec_median[np.maximum(sec_codes, 0)], np.nan)
    return np.stack([firm_comovements(Ya, np.roll(market, shift), min_obs),
                     firm_comovements(Ya, np.roll(sector, shift, axis=1), min_obs)], axis=1)


def group_means(b: np.ndarray, codes: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Mean value of each group (codes 0..G-1, -1 = none), shrunk towards the overall mean; firms clipped at 1/99 %."""
    ok = np.isfinite(b) & (codes >= 0)
    if rows is not None:
        ok &= rows
    y = np.clip(b, *np.nanquantile(b[np.isfinite(b)], [0.01, 0.99]))
    G = codes.max() + 1
    s, n = np.bincount(codes[ok], y[ok], G), np.bincount(codes[ok], minlength=G)
    grand = y[ok].mean()
    return (s + SHRINK * grand) / (n + SHRINK)


def crossfit(b: np.ndarray, codes: np.ndarray, splits: list[np.ndarray]) -> float:
    """Out-of-sample R² of firms' values from their group's mean estimated on the other half of the firms."""
    ok = np.isfinite(b) & (codes >= 0)
    if ok.sum() < 100:
        return np.nan
    y = np.clip(b, *np.nanquantile(b[ok], [0.01, 0.99]))
    r = []
    for h in splits:
        te = ok & ~h
        pred = group_means(b, codes, ok & h)[codes[te]]
        r.append(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return float(np.mean(r))


def _expo_rows(pillar, outcome, grouping, factor, values, codes, labels) -> pd.DataFrame:
    return pd.DataFrame({"pillar": pillar, "outcome": outcome, "grouping": grouping, "factor": factor, "group": labels,
                         "exposure": group_means(values, codes), "firms": np.bincount(codes[codes >= 0], minlength=len(labels))})


def score_episodes(Lv: dict, groups: dict, quarters: list[str], splits) -> tuple[list, list]:
    """Pillar 2: every level x episode x grouping, against the same change over every other window of that length."""
    comp, expo = [], []
    for name, V in Lv.items():
        for ep, (a, b) in EPISODES.items():
            if a not in quarters or b not in quarters:
                continue
            ia, ib = quarters.index(a), quarters.index(b)
            k, real = ib - ia, changes(V, a, b)
            ordinary = [i for i in range(len(quarters) - k) if (i + k < ia or i > ib)
                        and not (quarters[i] <= COVID[1] and quarters[i + k] >= COVID[0])]
            others = [changes(V, quarters[i], quarters[i + k]) for i in ordinary]
            for g in GROUPINGS:
                codes, labels = groups[g]
                r2, nl = crossfit(real, codes, splits), [crossfit(o, codes, splits) for o in others]
                comp.append({"pillar": 2, "outcome": name, "grouping": g, "factor": ep, "R2": r2,
                             "placebo mean": np.nanmean(nl), "placebo max": np.nanmax(nl), "placebo windows": len(nl),
                             "firms": int((np.isfinite(real) & (codes >= 0)).sum())})
                expo.append(_expo_rows(2, name, g, ep, real, codes, labels))
        print(f"  Pillar 2: {name}", flush=True)
    return comp, expo


def score_comovement(Lv: dict, groups: dict, quarters: list[str], splits) -> tuple[list, list]:
    """Pillar 3: every outcome x benchmark x grouping, against the benchmark shifted in time; plus stability."""
    est = [q for q in quarters if not COVID[0] <= q <= COVID[1]]
    before, after = [q for q in est if q < COVID[0]], [q for q in est if q > COVID[1]]
    shifts = list(range(4, len(est) - 3, 2))
    period_shifts = [(k, 4 + k % max(len(after) - 7, 1)) for k in range(4, len(before) - 3, 2)]
    has_periods = min(len(before), len(after)) >= MIN_QUARTERS_PERIOD
    sec, mp = groups["sector"][0], MIN_QUARTERS_PERIOD
    comp, expo = [], []
    for name, Y in yoy(Lv).items():
        Ya = Y[est].to_numpy(float)
        real, null = benchmark_comovements(Ya, sec), [benchmark_comovements(Ya, sec, k) for k in shifts]
        if has_periods:
            Yb, Yc = Y[before].to_numpy(float), Y[after].to_numpy(float)
            stab = (benchmark_comovements(Yb, sec, 0, mp), benchmark_comovements(Yc, sec, 0, mp))
            stab_null = [(benchmark_comovements(Yb, sec, kb, mp), benchmark_comovements(Yc, sec, ka, mp)) for kb, ka in period_shifts]
        for g in GROUPINGS:
            codes, labels = groups[g]
            corr = lambda a, b, j: np.corrcoef(group_means(a[:, j], codes), group_means(b[:, j], codes))[0, 1]
            for j, f in enumerate(BENCHMARKS):
                r2, nl = crossfit(real[:, j], codes, splits), [crossfit(b[:, j], codes, splits) for b in null]
                st, st_null = (corr(*stab, j), [corr(a, b, j) for a, b in stab_null]) if has_periods else (np.nan, [np.nan])
                comp.append({"pillar": 3, "outcome": name, "grouping": g, "factor": f, "R2": r2,
                             "placebo mean": np.nanmean(nl), "placebo max": np.nanmax(nl), "placebo windows": len(nl),
                             "stability": st, "stability placebo max": np.nanmax(st_null),
                             "firms": int((np.isfinite(real[:, j]) & (codes >= 0)).sum())})
                expo.append(_expo_rows(3, name, g, f, real[:, j], codes, labels))
        print(f"  Pillar 3: {name}", flush=True)
    return comp, expo


def score(Lv: dict, mem: dict, quarters: list[str], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both pillars. Returns (components, exposures)."""
    n = len(next(iter(Lv.values())))
    splits = [np.random.default_rng(seed + r).random(n) < 0.5 for r in range(N_SPLITS)]
    groups = {g: pd.factorize(modal_group(mem[g][quarters])) for g in GROUPINGS}
    c2, e2 = score_episodes(Lv, groups, quarters, splits)
    c3, e3 = score_comovement(Lv, groups, quarters, splits)
    comp = pd.DataFrame(c2 + c3)
    comp["kept"] = (comp.R2 > comp["placebo max"]) & (comp.R2 >= MIN_R2) & (comp["placebo windows"] >= MIN_PLACEBOS)
    comp["stable"] = comp["stability"] > comp["stability placebo max"]
    return comp, pd.concat(e2 + e3, ignore_index=True)


def column_name(r) -> str:
    return f"{r.outcome} | {r.factor} | {r.grouping}"


def vectors(comp: pd.DataFrame, expo: pd.DataFrame, mem: dict, pillar: int, quarters: list[str]) -> pd.DataFrame:
    """firm-quarter x kept components: the mean value of the firm's group in that quarter, standardised."""
    cols = {}
    for _, r in comp[(comp.pillar == pillar) & comp.kept].iterrows():
        e = expo[(expo.pillar == pillar) & (expo.outcome == r.outcome) & (expo.grouping == r.grouping) & (expo.factor == r.factor)]
        lookup = dict(zip(e.group, e.exposure))
        m = mem[r.grouping][quarters]
        v = m.apply(lambda col: col.map(lookup)).astype(float)
        cols[column_name(r)] = v.fillna(e.exposure.median()).where(m.notna()).stack(future_stack=True)
    V = pd.DataFrame(cols)
    if V.empty:
        return V
    V.index.names = ["ticker", "quarter"]
    V = V.dropna(how="all")
    return ((V - V.mean()) / V.std()).fillna(0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", type=Path, required=True, help="Folder with {bs,ins,cf}-raw.parquet.")
    p.add_argument("--metadata", type=Path, required=True, help="metadata.parquet of the download (SIC codes).")
    p.add_argument("--start", default="2012Q1", help="First quarter used (default 2012Q1: earlier balance sheets are unreliable).")
    p.add_argument("--end", default=None, help="Last quarter used (default: all).")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    L = load_lines(args.panel)
    meta = pd.read_parquet(args.metadata).drop_duplicates("ticker").set_index("ticker")
    Lv = levels(L)
    mem = memberships(L, meta.sic)
    allq = list(L["Total Assets"].columns)
    quarters = [q for q in allq if q >= args.start and (args.end is None or q <= args.end)]
    print(f"{len(L['Total Assets']):,} firms; {quarters[0]}–{quarters[-1]} ({len(quarters)} quarters)", flush=True)
    comp, expo = score(Lv, mem, quarters)
    out = args.out or args.panel / "targets"
    out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "components.csv", index=False)
    expo.to_parquet(out / "exposures.parquet")
    for pillar, name in [(2, "pillar2_episodes"), (3, "pillar3_comovement")]:
        V = vectors(comp, expo, mem, pillar, allq[4:])
        V.to_parquet(out / f"{name}.parquet")
        c = comp[comp.pillar == pillar]
        extra = f" ({int((c.kept & c.stable).sum())} of them also stable)" if pillar == 3 else ""
        print(f"Pillar {pillar}: {V.shape[1]} of {len(c)} components kept{extra}; {len(V):,} firm-quarters -> {out / name}.parquet")


if __name__ == "__main__":
    main()
