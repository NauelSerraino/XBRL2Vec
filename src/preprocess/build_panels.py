"""
Build the two input panels from a finqual download (merged_{balance_sheet,income_stmt,cash_flow}.parquet, plus the
optional metadata.parquet of src/download/sec_metadata.py):

    python src/preprocess/build_panels.py --download_dir data/in/download_2026-10-25 --out_dir data/in/panels_2026-10-25

Both panels: one ticker per issuer (duplicates removed). Filled: quarters --start..--end (start auto: the first
quarter with 20 % of firms reporting); complete: --complete_start..--end.

complete/   firms reported in every quarter of the window in all three statements, with no missing value in any line
            (zeros allowed): the rule behind the Dec-2025 panel (data/in/{bs,ins,cf}-raw.parquet). Nothing is filled;
            smaller and more reliable.

filled/     the gap-filling pipeline agreed on 2026-10-25 (evidence: notebooks/00_data_holes_and_filling.ipynb). A 0 or a NaN
            counts as missing. In this order:
            1. alive >= 4 years: total assets reported over a span of at least 16 quarters, else the firm is dropped;
            2. fiscal-year-end quarter of flows (income statement, cash flow except End Cash Position). finqual
               derives it as annual report minus the three quarterly reports of the fiscal year, so:
               - if 1-3 of those quarters are missing, the year-end value holds them too (case A: all three, the value
                 is the annual total; case C: one or two): it is set aside as a known total of the missing quarters
                 plus the year-end quarter, and redistributed after step 4 (interpolated shape, additive shift so the
                 quarters add up to that total; an even split where they cannot be interpolated);
               - if the four quarters sum to exactly 0, the annual report was missing and the year-end value is
                 0 - (the other three) (case B): it is set missing;
               the fiscal-year-end quarter comes from SEC (metadata.parquet, fye_quarter); December (Q4) when unknown;
            3. accounting identities (finqual enforces them exactly, so one term is derived from the others):
               Total Assets = Total Liabilities + Stockholders Equity = Current Assets + Total Non Current Assets,
               Total Revenue = Gross Profit + Cost Of Revenue, Pretax Income = Net Income + Tax Provision.
               When exactly one term is missing: if the others imply a non-zero value, it is filled from the identity;
               if they imply 0, the other part was derived from the zero (e.g. equity missing -> liabilities = assets):
               that part is set missing too, the missing term is interpolated in step 4 and the part recomputed from
               the identity after it (a zero tax provision confirmed by the identity is kept as a real zero);
            4. linear interpolation of holes of up to 4 quarters inside each item's reporting period (for a
               one-quarter hole: the mean of t-1 and t+1); longer holes: in a core item (Total Assets, Total Revenue,
               Net Income, Operating Cash Flow) the firm is dropped; in other items a run of zeros is kept as real
               zeros (e.g. no short-term debt) and NaN stays missing;
            5. redistribution of step 2, recomputation of the derived identity parts of step 3.
            Missing quarters before an item's first or after its last value (firm not reporting yet / any more) stay
            NaN. mask.parquet codes every cell (see CODES).

Output per panel: {bs,ins,cf}-raw.parquet (ticker, features, one column per quarter: the format read by
src/preprocess/preprocess.py); filled/ also mask.parquet, firms.parquet; out_dir/duplicates.parquet,
out_dir/report.json, out_dir/README.md.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

STATEMENTS = {"balance_sheet": "bs", "income_stmt": "ins", "cash_flow": "cf"}
#: lines of the cash-flow statement that are stocks, not flows
STOCK_LINES_IN_FLOW_STATEMENTS = {"End Cash Position"}
CORE = ["Total Assets", "Total Revenue", "Net Income", "Operating Cash Flow"]
#: total = part 1 + part 2 (all hold exactly in finqual's output)
IDENTITIES = [
    ("Total Assets", ("Total Liabilities Net Minority Interest", "Stockholders Equity")),
    ("Total Assets", ("Current Assets", "Total Non Current Assets")),
    ("Total Revenue", ("Gross Profit", "Cost Of Revenue")),
    ("Pretax Income", ("Net Income", "Tax Provision")),
]
#: a missing term that its identity puts at exactly 0 is a real zero for these lines
REAL_ZERO_OK = {"Tax Provision"}
MIN_SPAN = 16          # quarters: alive at least 4 years
MAX_GAP = 4            # longest hole that is interpolated
CODES = {0: "missing", 1: "observed", 2: "filled from identity", 3: "filled by interpolation",
         4: "filled: fiscal-year-end redistribution", 5: "zero kept (long run of zeros, non-core item)",
         6: "recomputed from identity (derived part)", 7: "zero confirmed by identity"}
QUARTER = re.compile(r"^\d{4}Q[1-4]$")


# --------------------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------------------

def quarter_range(start: str, end: str) -> list[str]:
    (y0, q0), (y1, q1) = (int(start[:4]), int(start[-1])), (int(end[:4]), int(end[-1]))
    out, y, q = [], y0, q0
    while (y, q) <= (y1, q1):
        out.append(f"{y}Q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def load_download(download_dir: Path) -> pd.DataFrame:
    """All statements stacked: ticker, stmt (bs / ins / cf), features, one column per quarter (all quarters)."""
    frames = []
    for name, stmt in STATEMENTS.items():
        d = pd.read_parquet(download_dir / f"merged_{name}.parquet")
        qs = sorted(c for c in d.columns if QUARTER.match(str(c)))
        d = d[["ticker", "features"] + qs].copy()
        d.insert(1, "stmt", stmt)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    qs = sorted(c for c in raw.columns if QUARTER.match(str(c)))
    raw = raw[["ticker", "stmt", "features"] + qs]
    return raw.drop_duplicates(["ticker", "stmt", "features"]).reset_index(drop=True)


def infer_end(raw: pd.DataFrame) -> str:
    """Last quarter in which total assets is reported by at least 80 % of the firms reporting it the quarter before."""
    ta = raw[raw.features == "Total Assets"]
    qs = [c for c in raw.columns if QUARTER.match(str(c))]
    n = (ta[qs].notna() & (ta[qs] != 0)).sum()
    for i in range(len(qs) - 1, 0, -1):
        if n.iloc[i] >= 0.8 * n.iloc[i - 1] and n.iloc[i] > 0:
            return qs[i]
    return qs[-1]


def infer_start(raw: pd.DataFrame, share: float = 0.2) -> str:
    """First quarter in which at least `share` of the firms in the download report total assets."""
    ta = raw[raw.features == "Total Assets"]
    qs = [c for c in raw.columns if QUARTER.match(str(c))]
    n = (ta[qs].notna() & (ta[qs] != 0)).sum()
    ok = n[n >= share * ta.ticker.nunique()]
    return ok.index[0] if len(ok) else qs[0]


def load_metadata(download_dir: Path) -> pd.DataFrame:
    """ticker, cik, fye_quarter (from metadata.parquet, else from the ticker list); empty frame if neither exists."""
    meta_path, tickers_path = download_dir / "metadata.parquet", download_dir / "company_tickers_exchange.json"
    if meta_path.exists():
        m = pd.read_parquet(meta_path)
    elif tickers_path.exists():
        j = json.loads(tickers_path.read_text())
        m = pd.DataFrame(j["data"], columns=j["fields"])
    else:
        return pd.DataFrame(columns=["ticker", "cik", "fye_quarter"])
    if "fye_quarter" not in m:
        m["fye_quarter"] = pd.NA
    return m[["ticker", "cik", "fye_quarter"]].drop_duplicates("ticker")


# --------------------------------------------------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------------------------------------------------

def deduplicate(raw: pd.DataFrame, meta: pd.DataFrame, quarters: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One ticker per issuer. Issuers = tickers sharing a CIK, or with identical total assets and net income in every
    quarter of the window (at least 8 reported quarters; catches tickers without a known CIK). Kept: the ticker with
    the most reported cells, then without '-' or '.', then the shortest, then alphabetical.
    """
    tickers = pd.Index(raw.ticker.unique())
    parent = {t: t for t in tickers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(group):
        group = [t for t in group if t in parent]
        for t in group[1:]:
            parent[find(t)] = find(group[0])

    cik = meta.set_index("ticker").cik.reindex(tickers)
    for _, g in cik.dropna().groupby(cik.dropna()):
        union(list(g.index))
    key_lines = raw[raw.features.isin(["Total Assets", "Net Income"])]
    vals = key_lines.set_index(["ticker", "features"])[quarters].round(0)
    reported = (vals.notna() & (vals != 0)).sum(1).groupby(level=0)
    ok = (reported.min() >= 8) & (reported.size() == 2)
    fp = vals.groupby(level=0).apply(lambda g: hash(g.sort_index().to_numpy().tobytes()))
    fp = fp[ok.reindex(fp.index, fill_value=False).astype(bool)]
    for _, g in fp.groupby(fp):
        union(list(g.index))

    cells = raw.set_index("ticker")[quarters].notna().sum(1).groupby(level=0).sum()
    table = pd.DataFrame({"ticker": tickers, "group": [find(t) for t in tickers], "cik": cik.values,
                          "cells": cells.reindex(tickers).values})
    table["plain"] = ~table.ticker.str.contains(r"[-.]", regex=True)
    table["length"] = table.ticker.str.len()
    order = table.sort_values(["group", "cells", "plain", "length", "ticker"], ascending=[True, False, False, True, True])
    keep = order.drop_duplicates("group").set_index("group").ticker
    table["kept_ticker"] = table.group.map(keep)
    table["group_size"] = table.group.map(table.group.value_counts())
    table = table.drop(columns=["plain", "length", "group"])
    return raw[raw.ticker.isin(set(keep))].reset_index(drop=True), table


# --------------------------------------------------------------------------------------------------------------------
# Panel 1: complete
# --------------------------------------------------------------------------------------------------------------------

def complete_panel(raw: pd.DataFrame, quarters: list[str]) -> pd.DataFrame:
    """Firms present in all three statements with no NaN in any line and quarter of the window (zeros allowed)."""
    has_nan = raw[quarters].isna().any(axis=1).groupby(raw.ticker).any()
    n_stmt = raw.groupby("ticker").stmt.nunique()
    ok = n_stmt[(n_stmt == 3)].index.difference(has_nan[has_nan].index)
    return raw[raw.ticker.isin(ok)].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------
# Panel 2: filled
# --------------------------------------------------------------------------------------------------------------------

def _holes(o: np.ndarray):
    """(start, end) of runs of False strictly between the first and the last True of o (end exclusive)."""
    idx = np.flatnonzero(o)
    if len(idx) < 2:
        return
    t, last = idx[0], idx[-1]
    while t < last:
        if not o[t]:
            s = t
            while not o[t]:
                t += 1
            yield s, t
        else:
            t += 1


def _apply_identities(tick, line, obs, out, code, row_of, T, recompute, zeros_only):
    """
    Step 3 of the filled pipeline, in place. For every identity total = part 1 + part 2 and quarter with exactly one
    term missing: fill it from the identity when the others imply a non-zero value; when they imply 0, keep a real
    zero (REAL_ZERO_OK lines) or, otherwise, reset the other part (derived from the zero) and queue it for
    recomputation. zeros_only: only the real-zero confirmations. Returns (filled, parts reset, zeros confirmed).
    """
    n_fill = n_contam = n_zero = 0
    for total, (p1, p2) in IDENTITIES:
        for firm in np.unique(tick):
            rows = [row_of.get((firm, l)) for l in (total, p1, p2)]
            if None in rows:
                continue
            rt, r1, r2 = rows
            for t in range(T):
                o = obs[rows, t]
                if o.sum() != 2:
                    continue
                if not o[0]:
                    target, other, implied = rt, None, out[r1, t] + out[r2, t]
                elif not o[1]:
                    target, other, implied = r1, r2, out[rt, t] - out[r2, t]
                else:
                    target, other, implied = r2, r1, out[rt, t] - out[r1, t]
                tol = max(1.0, 1e-9 * abs(out[rt, t]) if np.isfinite(out[rt, t]) else 1.0)
                if abs(implied) <= tol and line[target] in REAL_ZERO_OK:
                    out[target, t], obs[target, t], code[target, t] = 0.0, True, 7
                    n_zero += 1
                elif zeros_only:
                    continue
                elif abs(implied) > tol:
                    out[target, t], obs[target, t], code[target, t] = implied, True, 2
                    n_fill += 1
                elif other is not None:                                        # derived from the zero
                    obs[other, t], out[other, t], code[other, t] = False, np.nan, 0
                    recompute.append((other, t, rt, target))
                    n_contam += 1
    return n_fill, n_contam, n_zero


def filled_panel(raw: pd.DataFrame, quarters: list[str], fye: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """The filling pipeline (module docstring). Returns values, mask (same layout), firm table, step counts."""
    rep = {}
    raw = raw.reset_index(drop=True)
    V = raw[quarters].to_numpy(float)
    T = len(quarters)
    obs = np.isfinite(V) & (V != 0)
    zero0 = V == 0
    tick, line, stmt = raw.ticker.to_numpy(), raw.features.to_numpy(), raw.stmt.to_numpy()

    # lines nobody reports carry no information
    reported_lines = pd.Series(obs.any(1)).groupby(line).any()
    keep_rows = np.isin(line, reported_lines[reported_lines].index)

    # 1. alive >= 4 years
    ta = line == "Total Assets"
    span = np.where(obs[ta].any(1), T - obs[ta][:, ::-1].argmax(1) - obs[ta].argmax(1), 0)
    alive = set(tick[ta][span >= MIN_SPAN])
    rep["firms_in_download_window"] = int(len(set(tick)))
    rep["firms_alive_4y"] = len(alive)
    keep_rows &= np.isin(tick, list(alive))
    raw, V, obs, zero0 = raw[keep_rows].reset_index(drop=True), V[keep_rows], obs[keep_rows], zero0[keep_rows]
    tick, line, stmt = raw.ticker.to_numpy(), raw.features.to_numpy(), raw.stmt.to_numpy()
    R = len(raw)
    out = np.where(obs, V, np.nan)
    code = np.where(obs, 1, 0).astype(np.int8)
    row_of = {(t, l): i for i, (t, l) in enumerate(zip(tick, line))}

    # identity-confirmed zeros first (e.g. no tax in a loss year): they are data, not missing quarters
    _, _, n_zero_ok = _apply_identities(tick, line, obs, out, code, row_of, T, [], zeros_only=True)
    rep["identity_real_zeros"] = n_zero_ok

    # 2. fiscal-year-end quarter of flows
    is_flow = np.isin(stmt, ["ins", "cf"]) & ~np.isin(line, list(STOCK_LINES_IN_FLOW_STATEMENTS))
    label = np.array([int(q[-1]) for q in quarters])
    fye_q = fye.reindex(pd.Index(tick)).fillna(4).astype(int).to_numpy()
    constraints, n_a, n_b, n_c = [], 0, 0, 0
    for r in np.flatnonzero(is_flow):
        for t in range(3, T):
            if label[t] != fye_q[r] or not obs[r, t]:
                continue
            blk = [t - 3, t - 2, t - 1]
            if obs[r, blk].all():
                if abs(V[r, blk].sum() + V[r, t]) <= 1.0:                      # case B
                    obs[r, t], out[r, t], code[r, t] = False, np.nan, 0
                    n_b += 1
                continue
            miss = [i for i in blk if not obs[r, i]]                           # case A (3 missing) or C (1-2)
            constraints.append((r, miss + [t], V[r, t]))
            obs[r, t], out[r, t], code[r, t] = False, np.nan, 0
            n_a += len(miss) == 3
            n_c += len(miss) < 3
    rep["fye_case_A_annual_total"], rep["fye_case_B_negative"], rep["fye_case_C_partial"] = n_a, n_b, n_c

    # 3. identities (zeros they confirm were settled before step 2, so a real zero is not taken for a missing quarter)
    recompute = []
    n_id_fill, n_contam, _ = _apply_identities(tick, line, obs, out, code, row_of, T, recompute, zeros_only=False)
    rep["identity_filled"], rep["identity_derived_parts_reset"] = n_id_fill, n_contam

    # 4. interpolation of holes <= MAX_GAP; long holes
    core_long, n_interp, n_zero_kept = set(), 0, 0
    for r in range(R):
        for s, e in _holes(obs[r]):
            if e - s <= MAX_GAP:
                a, b = out[r, s - 1], out[r, e]
                out[r, s:e] = a + (b - a) * np.arange(1, e - s + 1) / (e - s + 1)
                code[r, s:e] = 3
                n_interp += e - s
            elif line[r] in CORE:
                core_long.add(tick[r])
            else:
                z = zero0[r, s:e] & (code[r, s:e] == 0)
                out[r, s:e][z] = 0.0
                code[r, s:e][z] = 5
                n_zero_kept += int(z.sum())
    rep["cells_interpolated"], rep["cells_zero_kept_long_runs"] = n_interp, n_zero_kept

    # 5. redistribution of the fiscal-year-end totals; recomputation of derived identity parts
    n_redist = 0
    for r, idx, total_value in constraints:
        vals = out[r, idx]
        if np.isfinite(vals).all():
            out[r, idx] = vals + (total_value - vals.sum()) / len(idx)
        else:
            out[r, idx] = total_value / len(idx)
        code[r, idx] = 4
        n_redist += len(idx)
    n_recomp = 0
    for other, t, rt, target in recompute:
        if np.isfinite(out[rt, t]) and np.isfinite(out[target, t]):
            out[other, t], code[other, t] = out[rt, t] - out[target, t], 6
            n_recomp += 1
    rep["cells_fye_redistributed"], rep["identity_parts_recomputed"] = n_redist, n_recomp

    rep["firms_dropped_core_long_hole"] = len(core_long)
    keep = ~np.isin(tick, list(core_long))
    rep["firms_filled_panel"] = int(len(set(tick[keep])))
    values = pd.concat([raw.loc[keep, ["ticker", "stmt", "features"]].reset_index(drop=True),
                        pd.DataFrame(out[keep], columns=quarters)], axis=1)
    mask = pd.concat([raw.loc[keep, ["ticker", "stmt", "features"]].reset_index(drop=True),
                      pd.DataFrame(code[keep], columns=quarters)], axis=1)
    ta_rows = values[values.features == "Total Assets"].set_index("ticker")[quarters]
    firms = pd.DataFrame({"first_quarter": ta_rows.apply(lambda s: s.first_valid_index(), axis=1),
                          "last_quarter": ta_rows.apply(lambda s: s.last_valid_index(), axis=1)})
    kept_codes = code[keep]
    counts = pd.DataFrame({f"cells_code_{c}": (kept_codes == c).sum(1) for c in CODES}).groupby(tick[keep]).sum()
    firms = firms.join(counts).join(fye.rename("fye_quarter"), how="left").reset_index().rename(columns={"index": "ticker"})
    rep["cells_by_code"] = {CODES[k]: int(v) for k, v in pd.Series(code[keep].ravel()).value_counts().sort_index().items()}
    return values, mask, firms, rep


# --------------------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------------------

def write_panel(frame: pd.DataFrame, quarters: list[str], out: Path) -> None:
    """{bs,ins,cf}-raw.parquet in the format of src/preprocess/preprocess.py (ticker, features, quarters)."""
    out.mkdir(parents=True, exist_ok=True)
    for stmt in STATEMENTS.values():
        frame[frame.stmt == stmt][["ticker", "features"] + quarters].reset_index(drop=True).to_parquet(out / f"{stmt}-raw.parquet")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=None, help="Default: data/in/panels_<download date>.")
    p.add_argument("--start", default="auto",
                   help="First quarter of the filled panel; auto: the first quarter in which 20 %% of firms report total assets.")
    p.add_argument("--complete_start", default="2011Q1",
                   help="First quarter of the complete panel (every firm must report in every quarter from here: an earlier "
                        "start keeps fewer firms; on the 2025-11-10 download 2010Q1 / 2011Q1 / 2012Q1 give 826 / 1,989 / 2,230).")
    p.add_argument("--end", default=None, help="Default: the last quarter most firms have reported (see infer_end).")
    args = p.parse_args()
    name = args.download_dir.name
    out = args.out_dir or args.download_dir.parent / (name.replace("download_", "panels_", 1) if name.startswith("download_") else f"panels_{name}")

    raw_all = load_download(args.download_dir)
    end = args.end or infer_end(raw_all)
    start = infer_start(raw_all) if args.start == "auto" else args.start
    quarters = quarter_range(start, end)
    for q in quarters:
        if q not in raw_all:
            raw_all[q] = np.nan
    raw = raw_all[["ticker", "stmt", "features"] + quarters]
    meta = load_metadata(args.download_dir)
    report = {"download_dir": str(args.download_dir), "built": date.today().isoformat(), "window": [quarters[0], quarters[-1]],
              "tickers_in_download": int(raw.ticker.nunique())}

    raw, dups = deduplicate(raw, meta, quarters)
    out.mkdir(parents=True, exist_ok=True)
    dups.to_parquet(out / "duplicates.parquet")
    report["issuers_after_dedup"] = int(raw.ticker.nunique())
    report["tickers_removed_as_duplicates"] = int(report["tickers_in_download"] - report["issuers_after_dedup"])

    comp_quarters = [q for q in quarters if q >= args.complete_start] or quarters
    comp = complete_panel(raw[["ticker", "stmt", "features"] + comp_quarters], comp_quarters)
    write_panel(comp, comp_quarters, out / "complete")
    report["complete_panel_firms"] = int(comp.ticker.nunique())
    report["complete_window"] = [comp_quarters[0], comp_quarters[-1]]

    fye = meta.set_index("ticker").fye_quarter.dropna().astype(int)
    report["fye_known_for_firms"] = int(fye.reindex(raw.ticker.unique()).notna().sum())
    values, mask, firms, rep = filled_panel(raw, quarters, fye)
    write_panel(values, quarters, out / "filled")
    mask.to_parquet(out / "filled" / "mask.parquet")
    firms.to_parquet(out / "filled" / "firms.parquet")
    report["filled"] = rep
    (out / "report.json").write_text(json.dumps(report, indent=2, default=_jsonable))
    (out / "README.md").write_text(readme(report))
    print(json.dumps(report, indent=2, default=_jsonable))


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def readme(rep: dict) -> str:
    f = rep["filled"]
    codes = "\n".join(f"| {k} | {v} |" for k, v in CODES.items())
    return f"""# Input panels built from `{rep['download_dir']}`

Built {rep['built']} by `src/preprocess/build_panels.py` (rules in its docstring), quarters {rep['window'][0]}–{rep['window'][1]}.

| step | count |
|---|---|
| tickers in the download | {rep['tickers_in_download']:,} |
| issuers after removing duplicates (`duplicates.parquet`) | {rep['issuers_after_dedup']:,} |
| **complete/**: firms with every line in every quarter ({rep['complete_window'][0]}–{rep['complete_window'][1]}) | **{rep['complete_panel_firms']:,}** |
| filled: firms alive >= 4 years | {f['firms_alive_4y']:,} |
| filled: dropped for a hole > 4 quarters in a core item | {f['firms_dropped_core_long_hole']:,} |
| **filled/**: firms | **{f['firms_filled_panel']:,}** |

Filled panel, cells changed: fiscal-year-end cases A / B / C {f['fye_case_A_annual_total']:,} / {f['fye_case_B_negative']:,} /
{f['fye_case_C_partial']:,} (redistributed cells {f['cells_fye_redistributed']:,}); identity fills {f['identity_filled']:,};
derived parts reset {f['identity_derived_parts_reset']:,} (recomputed {f['identity_parts_recomputed']:,}); zeros confirmed by an
identity {f['identity_real_zeros']:,}; interpolated {f['cells_interpolated']:,}; zeros kept in long runs {f['cells_zero_kept_long_runs']:,}.
Fiscal-year end known (SEC) for {rep['fye_known_for_firms']:,} firms; the others are assumed to end in December.

`filled/mask.parquet` (same layout as the values) codes every cell:

| code | meaning |
|---|---|
{codes}

Both folders can be passed to `src/preprocess/preprocess.py --out_dir <folder>` (filled/ with `--allow_missing`).
"""


if __name__ == "__main__":
    main()
