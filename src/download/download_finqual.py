"""
Download the quarterly financial statements (balance sheet, income statement, cash flow) of every company in SEC's
current ticker list with finqual.

    ~/vscode/finqual/finqual-venv/bin/python src/download/download_finqual.py --out data/in/download_2026-10-25
    ... --limit 5                 # smoke test on the first 5 companies
    ... --merge_only              # only rebuild the merged files from what is already downloaded
    ... --shard 0 --n_shards 3 --no_merge     # one of 3 parallel workers (src/main_download.py --workers 3)

Run with a Python that has finqual installed (this project's venv does not; src/main_download.py uses the venv of
the finqual fork in ~/vscode/finqual by default).

One request per company: tickers are grouped by CIK and one ticker per CIK is downloaded (share classes, preferreds
and notes of the same issuer have the same statements); --all_tickers downloads every ticker instead.

Resumable: each company is saved on its own (<out>/tickers/<TICKER>.parquet) and logged in <out>/progress*.csv
(one file per worker).
Re-running skips companies already done (and, unless --retry_failed, those that failed), so an interrupted night
loses at most the company in progress. At the end the per-company files are merged into
<out>/merged_{balance_sheet,income_stmt,cash_flow}.parquet: one row per (ticker, line), one column per quarter, the
format of data/in/download_2025-11-10.

The ticker list is SEC's company_tickers_exchange.json as of the first run (saved in <out>): companies listed today.
Companies delisted before today are not covered (survivorship; see data/in/download_2025-11-10/README.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

TICKER_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
#: finqual method per statement, in the order of the merged files
STATEMENTS = {"income_stmt": "income_stmt_period", "balance_sheet": "balance_sheet_period", "cash_flow": "cash_flow_period"}
CIK_OVERRIDES = Path(__file__).with_name("cik_overrides.csv")
PROGRESS_FIELDS = ["ticker", "cik", "status", "sector", "seconds", "error", "time"]
DEFAULT_USER_AGENT = "XBRL2Vec research your@email.com"   # placeholder of the form SEC accepts (as finqual's own); pass a real contact

log = logging.getLogger("download_finqual")


class Timeout(Exception):
    pass


REQUESTS_MADE = 0


def install_rate_limit(max_per_second: float, backoff: float = 60.0, retries: int = 3) -> None:
    """
    Cap this process's HTTP requests (ours and finqual's: both go through requests.Session.request) at
    max_per_second, and back off on SEC's throttling answers (403 / 429): wait `backoff` x attempt seconds and retry.
    SEC's fair-access limit is 10 requests per second per client; src/main_download.py gives each of N workers
    1/N of a budget below that, so the total stays under the limit whatever finqual does.
    """
    import requests.sessions
    original = requests.sessions.Session.request
    interval, last = 1.0 / max_per_second, [0.0]

    def throttled(self, method, url, *a, **kw):
        global REQUESTS_MADE
        for attempt in range(retries + 1):
            wait = last[0] + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
            REQUESTS_MADE += 1
            response = original(self, method, url, *a, **kw)
            if response.status_code not in (403, 429) or attempt == retries:
                return response
            log.warning("SEC answered %s for %s; waiting %.0f s (attempt %d)", response.status_code, url, backoff * (attempt + 1), attempt + 1)
            response.close()
            time.sleep(backoff * (attempt + 1))
        return response

    requests.sessions.Session.request = throttled


def _on_alarm(signum, frame):
    raise Timeout()


def sec_headers(user_agent: str) -> dict:
    return {"User-Agent": user_agent, "Accept": "application/json, text/plain, */*", "Accept-Encoding": "gzip, deflate"}


def ticker_list(out: Path, user_agent: str) -> pd.DataFrame:
    """SEC's current ticker list (cik, name, ticker, exchange), fetched once and kept in <out> for resumes."""
    path = out / "company_tickers_exchange.json"
    if not path.exists():
        r = requests.get(TICKER_URL, headers=sec_headers(user_agent), timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
    data = json.loads(path.read_text())
    return pd.DataFrame(data["data"], columns=data["fields"])


def load_cik_overrides(path: Path = CIK_OVERRIDES) -> dict[str, dict]:
    """{ticker: {cik, name, reason}}: tickers to download under another CIK than SEC's current list gives."""
    if not path.exists():
        return {}
    d = pd.read_csv(path, dtype=str, comment="#")
    return {r.ticker.upper(): {"cik": r.cik.zfill(10), "name": r["name"], "reason": r.reason} for _, r in d.iterrows()}


def install_cik_overrides(overrides: dict[str, dict]) -> None:
    """
    Make finqual resolve these tickers to the given CIK. finqual only accepts CIKs found in SEC's current ticker list,
    and after a reorganisation the list points the ticker at the new parent, whose filings start at the
    reorganisation (XOM -> ExxonMobil Holdings Corp, 2026), while the history stays under the old CIK.
    """
    if not overrides:
        return
    from finqual.sec_edgar import sec_api
    from finqual.sec_edgar.entities.models import CompanyIdCode
    original = sec_api.SecApi.get_id_code

    def get_id_code(self, ticker_or_cik):
        o = overrides.get(str(ticker_or_cik).upper())
        if o is None:
            return original(self, ticker_or_cik)
        return CompanyIdCode(cik=o["cik"], name=o["name"], ticker=str(ticker_or_cik).upper(), exchange="")

    sec_api.SecApi.get_id_code = get_id_code


def one_ticker_per_cik(tickers: pd.DataFrame) -> pd.DataFrame:
    """The ticker of each CIK most likely to be the common stock: no '-' or '.', then shortest, then SEC's order."""
    t = tickers.reset_index(drop=True).assign(order=lambda d: d.index)
    t["plain"] = ~t.ticker.str.contains(r"[-.]", regex=True)
    t["length"] = t.ticker.str.len()
    t = t.sort_values(["cik", "plain", "length", "order"], ascending=[True, False, True, True])
    return t.drop_duplicates("cik").sort_values("order").drop(columns=["order", "plain", "length"])


def read_progress(out: Path) -> dict[str, str]:
    """Latest status of every ticker over all progress files (progress.csv, progress_shard<i>.csv), by time."""
    rows = []
    for path in out.glob("progress*.csv"):
        with open(path, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return {row["ticker"]: row["status"] for row in sorted(rows, key=lambda r: r["time"])}


def append_progress(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PROGRESS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def statement_period(company, method: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, list[str]]:
    """
    One statement over start_year..end_year, quarter by quarter (finqual's single-quarter method, e.g. income_stmt),
    as finqual's <method>_period(quarter=True) builds it
    (line column named after the ticker, one column per quarter, newest first, all-zero quarters dropped), except
    that a quarter finqual fails on is skipped instead of failing the whole company (e.g. CAT: only 2007Q1 fails).
    Returns the frame and the skipped quarters.
    """
    func, results, skipped = getattr(company, method), {}, []
    labels = [(y, q) for y in range(end_year, start_year - 1, -1) for q in (4, 3, 2, 1)]
    with ThreadPoolExecutor(max_workers=2) as executor:                  # as finqual does
        futures = {executor.submit(func, y, q): f"{y}Q{q}" for y, q in labels}
        for future in as_completed(futures):
            try:
                results[futures[future]] = future.result()
            except Timeout:
                raise
            except Exception:
                skipped.append(futures[future])
    frames = []
    for label in [f"{y}Q{q}" for y, q in labels]:
        if label in results:
            d = results[label].to_pandas()
            frames.append(d.set_index(d.columns[0])[d.columns[1]].rename(label))
    if not frames:
        return pd.DataFrame(), sorted(skipped)
    df = pd.concat(frames, axis=1, join="inner")
    df = df.loc[:, ~(df == 0).all()]
    if method == "balance_sheet":                                        # as balance_sheet_period does
        df = df.loc[:, ~(df.drop(index="Shares Outstanding", errors="ignore") == 0).all()]
    df.index.name = company.ticker
    return df.reset_index(), sorted(skipped)


def download_one(fq, ticker: str, start_year: int, end_year: int) -> tuple[pd.DataFrame | None, str | None, list[str]]:
    """All three statements of one company, stacked (ticker, statement, features, <quarters>), its sector, and the
    quarters finqual failed on (skipped)."""
    company = fq.Finqual(ticker)
    parts, skipped = [], []
    for name in STATEMENTS:
        df, sk = statement_period(company, name, start_year, end_year)
        skipped += [f"{name} {q}" for q in sk]
        if df.empty:
            continue
        label_col = ticker if ticker in df.columns else df.columns[0]   # finqual names the line column after the ticker
        df = df.rename(columns={label_col: "features"})
        df.insert(0, "statement", name)
        df.insert(0, "ticker", ticker)
        parts.append(df)
    sector = getattr(company, "sector", None)
    return (pd.concat(parts, ignore_index=True) if parts else None), sector, skipped


def safe_name(ticker: str) -> str:
    return ticker.replace("/", "_")


def merge(out: Path) -> None:
    """Per-company files -> merged_{statement}.parquet (+ finqual_sector.parquet) in the 2025-11-10 format."""
    files = sorted((out / "tickers").glob("*.parquet"))
    if not files:
        log.warning("nothing to merge in %s", out / "tickers")
        return
    data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    quarters = sorted(c for c in data.columns if len(str(c)) == 6 and str(c)[4] == "Q")
    for name in STATEMENTS:
        part = data[data.statement == name][["ticker", "features"] + quarters].reset_index(drop=True)
        part[quarters] = part[quarters].apply(pd.to_numeric, errors="coerce")
        part.to_parquet(out / f"merged_{name}.parquet")
        log.info("merged %s: %d rows, %d tickers, %s..%s", name, len(part), part.ticker.nunique(), quarters[0], quarters[-1])
    prog = pd.concat([pd.read_csv(f) for f in out.glob("progress*.csv")]).sort_values("time").drop_duplicates("ticker", keep="last")
    prog[prog.status == "ok"][["ticker", "cik", "sector"]].to_parquet(out / "finqual_sector.parquet")


def finqual_info(fq) -> dict:
    path = Path(fq.__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    try:
        from importlib.metadata import version
        ver = version("finqual")
    except Exception:
        ver = None
    return {"finqual_path": str(path), "finqual_git_commit": commit or None, "finqual_version_metadata": ver}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/in") / f"download_{date.today().isoformat()}")
    p.add_argument("--start_year", type=int, default=2000,
                   help="First year requested; finqual returns what SEC XBRL data exist (in practice from 2006–2008 on).")
    p.add_argument("--end_year", type=int, default=date.today().year)
    p.add_argument("--user_agent", default=DEFAULT_USER_AGENT,
                   help="SEC asks for a contact in the User-Agent, e.g. 'Name surname name@example.org'.")
    p.add_argument("--all_tickers", action="store_true", help="Download every ticker, not one per CIK.")
    p.add_argument("--retry_failed", action="store_true", help="Retry companies that failed in an earlier run.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N companies (smoke test).")
    p.add_argument("--tickers", nargs="+", default=None, help="Only these tickers (with --retry_failed to redo failed ones).")
    p.add_argument("--sleep", type=float, default=0.2, help="Pause between companies, seconds (SEC fair-access limit).")
    p.add_argument("--timeout", type=int, default=180, help="Give up on a company after this many seconds.")
    p.add_argument("--merge_only", action="store_true")
    p.add_argument("--list_only", action="store_true", help="Only fetch the SEC ticker list and record the run (before workers).")
    p.add_argument("--shard", type=int, default=0, help="This worker's share: companies shard, shard + n_shards, ...")
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--max_requests_per_second", type=float, default=8.0,
                   help="Cap for THIS process (SEC allows 10 per second per client; give each of N workers 8 / N).")
    p.add_argument("--no_merge", action="store_true", help="Do not merge at the end (parallel workers; merge once after).")
    args = p.parse_args()

    out = args.out
    (out / "tickers").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / ("download.log" if args.n_shards == 1 else f"download_shard{args.shard}.log")),
                                  logging.StreamHandler()])
    if args.merge_only:
        merge(out)
        return

    install_rate_limit(args.max_requests_per_second)
    import finqual as fq                                      # imported here so --merge_only works without finqual
    overrides = load_cik_overrides()
    install_cik_overrides(overrides)

    if args.shard == 0:                                       # one writer for the run record
        info_path = out / "download_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {"runs": []}
        info.update(finqual_info(fq))
        info["runs"].append({"started": datetime.now().isoformat(timespec="seconds"),
                             "args": {k: str(v) for k, v in vars(args).items()}, "python": platform.python_version()})
        info_path.write_text(json.dumps(info, indent=2))

    tickers = ticker_list(out, args.user_agent)
    if args.list_only:
        log.info("ticker list: %d tickers, %d CIKs", len(tickers), tickers.cik.nunique())
        return
    todo = tickers if args.all_tickers else one_ticker_per_cik(tickers)
    todo = todo.assign(cik=[int(overrides[t]["cik"]) if t in overrides else c for t, c in zip(todo.ticker, todo.cik)])
    if args.tickers:
        todo = todo[todo.ticker.isin([t.upper() for t in args.tickers])]
    if args.limit:
        todo = todo.head(args.limit)
    todo = todo.iloc[args.shard::args.n_shards]
    progress_path = out / ("progress.csv" if args.n_shards == 1 else f"progress_shard{args.shard}.csv")
    done = read_progress(out)
    skip = set() if args.tickers else {"ok", "empty"} | (set() if args.retry_failed else {"failed"})   # named tickers: always redo
    todo = todo[~todo.ticker.map(done).isin(skip)]
    log.info("SEC list: %d tickers, %d CIKs; worker %d of %d, to download now: %d (done overall: %d)", len(tickers),
             tickers.cik.nunique(), args.shard, args.n_shards, len(todo), sum(s in skip for s in done.values()))

    signal.signal(signal.SIGALRM, _on_alarm)
    t0, n_ok = time.time(), 0
    for i, (ticker, cik) in enumerate(zip(todo.ticker, todo.cik), start=1):
        start, status, sector, error = time.time(), "failed", None, ""
        signal.alarm(args.timeout)
        try:
            df, sector, skipped = download_one(fq, ticker, args.start_year, args.end_year)
            if df is None:
                status = "empty"
            else:
                df.to_parquet(out / "tickers" / f"{safe_name(ticker)}.parquet")
                status, n_ok = "ok", n_ok + 1
            if skipped:
                error = f"skipped {len(skipped)} quarter-statements finqual failed on: " + ", ".join(skipped)[:250]
        except Timeout:
            error = f"timeout after {args.timeout} s"
        except Exception as e:                                # finqual raises many kinds; log and move on
            error = f"{type(e).__name__}: {e}"[:300]
        finally:
            signal.alarm(0)
        append_progress(progress_path, {"ticker": ticker, "cik": cik, "status": status, "sector": sector,
                                        "seconds": round(time.time() - start, 1), "error": error,
                                        "time": datetime.now().isoformat(timespec="seconds")})
        if status == "failed":
            log.warning("%s failed: %s", ticker, error)
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            log.info("%d / %d done (%d ok), %.1f s per company, ETA %.1f h; %d requests (%.2f per second)", i, len(todo),
                     n_ok, rate, rate * (len(todo) - i) / 3600, REQUESTS_MADE, REQUESTS_MADE / (time.time() - t0))
        time.sleep(args.sleep)

    if not args.no_merge:
        merge(out)
    log.info("download complete (worker %d of %d)", args.shard, args.n_shards)


if __name__ == "__main__":
    main()
