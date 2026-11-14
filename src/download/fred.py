"""
Macro series from FRED (public CSV endpoint, no API key), as far back as requested.

    python src/download/fred.py --out data/in/download_2026-10-25/fred

One CSV per series, <out>/<ID>.csv with columns observation_date, <ID>: the format of the files in
data/in/exogenous/ (src/preprocess/preprocess_exog.py and the notebooks read it). The default series are the 7 used
so far: GDP (nominal), industrial production, personal saving rate, CPI, Fed funds rate, retail sales, unemployment.
"""
from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"
SERIES = ["GDP", "INDPRO", "PSAVERT", "CPIAUCSL", "DFF", "MRTSSM44000USS", "UNEMPLOY"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default="1990-01-01", help="First observation date requested.")
    p.add_argument("--series", nargs="+", default=SERIES)
    p.add_argument("--user_agent", default="XBRL2Vec research your@email.com")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for s in args.series:
        r = requests.get(URL.format(series=s, start=args.start), headers={"User-Agent": args.user_agent}, timeout=60)
        r.raise_for_status()
        d = pd.read_csv(StringIO(r.text))
        if list(d.columns) != ["observation_date", s]:
            raise SystemExit(f"{s}: unexpected columns {list(d.columns)}")
        d.to_csv(args.out / f"{s}.csv", index=False)
        print(f"{s}: {len(d)} observations, {d.observation_date.iloc[0]} .. {d.observation_date.iloc[-1]}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
