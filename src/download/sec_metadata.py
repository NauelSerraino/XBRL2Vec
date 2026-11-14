"""
Company metadata from SEC's submissions API, for every company of a download: fiscal-year end, SIC code, name,
entity type. Needed because finqual derives the quarter that ends the fiscal year (annual report minus the three
quarterly reports), and that quarter is Q4 only for December year-ends (Apple's is labelled Q3).

    python src/download/sec_metadata.py --download_dir data/in/download_2026-10-25

Reads the ticker list saved by download_finqual.py (<download_dir>/company_tickers_exchange.json); one request per
CIK (https://data.sec.gov/submissions/CIK##########.json), resumable through <download_dir>/sec_submissions.csv.
Writes <download_dir>/metadata.parquet: ticker, cik, name, exchange, fiscal_year_end (MMDD), fye_quarter (1-4, the
calendar-quarter label of the fiscal-year-end quarter in finqual's data), sic, sic_description, entity_type and,
when the download has it, finqual's sector.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FIELDS = ["cik", "fiscal_year_end", "sic", "sic_description", "entity_type", "sec_name", "status", "error"]
DEFAULT_USER_AGENT = "XBRL2Vec research your@email.com"   # placeholder of the form SEC accepts (as finqual's own); pass a real contact


def fye_quarter(mmdd: str | float | None) -> int | None:
    """
    Calendar-quarter label (1-4) of the quarter that ends the fiscal year. A 52/53-week year ending in the first
    week of a quarter's first month (e.g. 0103, 0402) belongs to the quarter before.
    """
    if not isinstance(mmdd, str) or len(mmdd) != 4 or not mmdd.isdigit():
        return None
    month, day = int(mmdd[:2]), int(mmdd[2:])
    if month in (1, 4, 7, 10) and day <= 7:
        month = 12 if month == 1 else month - 1
    return (month - 1) // 3 + 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download_dir", type=Path, required=True)
    p.add_argument("--user_agent", default=DEFAULT_USER_AGENT)
    p.add_argument("--sleep", type=float, default=0.125, help="Pause between requests: at most 8 per second (SEC's limit is 10).")
    p.add_argument("--only_downloaded", action="store_true",
                   help="Only the CIKs of companies in the download (progress.csv or merged files), not the whole SEC list.")
    args = p.parse_args()

    d = args.download_dir
    headers = {"User-Agent": args.user_agent, "Accept-Encoding": "gzip, deflate"}
    tickers_path = d / "company_tickers_exchange.json"
    if not tickers_path.exists():                       # downloads made before this script (e.g. 2025-11-10)
        r = requests.get("https://www.sec.gov/files/company_tickers_exchange.json", headers=headers, timeout=60)
        r.raise_for_status()
        tickers_path.write_text(r.text)
        print(f"fetched today's SEC ticker list into {tickers_path}")
    raw = json.loads(tickers_path.read_text())
    tickers = pd.DataFrame(raw["data"], columns=raw["fields"])
    if args.only_downloaded:
        merged = d / "merged_balance_sheet.parquet"
        downloaded = set(pd.read_parquet(merged, columns=["ticker"]).ticker) if merged.exists() else set()
        ciks = set(tickers[tickers.ticker.isin(downloaded)].cik)
        tickers = tickers[tickers.cik.isin(ciks)]
    cache = d / "sec_submissions.csv"
    done = set(pd.read_csv(cache).cik.astype(int)) if cache.exists() else set()
    todo = [c for c in tickers.cik.drop_duplicates().astype(int) if c not in done]
    print(f"{tickers.cik.nunique()} CIKs, {len(todo)} to fetch")

    new = not cache.exists()
    with open(cache, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, cik in enumerate(todo, start=1):
            row = {"cik": cik, "status": "ok", "error": ""}
            try:
                r = requests.get(URL.format(cik=cik), headers=headers, timeout=30)
                r.raise_for_status()
                s = r.json()
                row.update(fiscal_year_end=s.get("fiscalYearEnd"), sic=s.get("sic"), sic_description=s.get("sicDescription"),
                           entity_type=s.get("entityType"), sec_name=s.get("name"))
            except Exception as e:
                row.update(status="failed", error=f"{type(e).__name__}: {e}"[:200])
            w.writerow(row)
            if i % 200 == 0:
                fh.flush()
                print(f"{i} / {len(todo)}", flush=True)
            time.sleep(args.sleep)

    sub = pd.read_csv(cache, dtype={"fiscal_year_end": str}).drop_duplicates("cik", keep="last")
    meta = tickers.merge(sub[sub.status == "ok"].drop(columns=["status", "error"]), on="cik", how="left")
    meta["fye_quarter"] = meta.fiscal_year_end.map(fye_quarter).astype("Int64")
    sector = d / "finqual_sector.parquet"
    if sector.exists():
        meta = meta.merge(pd.read_parquet(sector)[["ticker", "sector"]], on="ticker", how="left")
    meta.to_parquet(d / "metadata.parquet")
    print(f"metadata.parquet: {len(meta)} tickers, fiscal-year end known for {meta.fye_quarter.notna().sum()}; "
          f"fye quarter counts: {meta.fye_quarter.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
