"""
Data pipeline before modelling: download, metadata, panels, preprocessing.

  1. finqual download of every company in SEC's current ticker list   src/download/download_finqual.py
  2. SEC metadata (fiscal-year end, SIC) and FRED macro series         src/download/sec_metadata.py, fred.py
  3. the two panels, complete/ and filled/                             src/preprocess/build_panels.py
  4. differencing, scaling and company train/test split of each panel  src/preprocess/preprocess.py
     (+ exog.parquet and metadata.parquet next to the panels, read by src/main.py)
  5. Pillar 2-3 target vectors from the filled panel                   src/preprocess/pillar_targets.py

    # tonight: everything, into data/in/download_<today>/ and data/in/panels_<today>/ (resumable: just re-run)
    xbrl2vec/bin/python src/main_download.py --user_agent "Name Surname name@example.org"

    # smoke test on 5 companies, into a scratch folder
    xbrl2vec/bin/python src/main_download.py --limit 5 --download_dir /tmp/dl_test

    # steps 2-5 on an existing download
    xbrl2vec/bin/python src/main_download.py --download_dir data/in/download_2025-11-10 --skip_download

Steps 1-2 run with a Python that has finqual (default: the venv of the finqual fork, override with --finqual_python
or FINQUAL_PYTHON); steps 3-5 with the Python running this script. Each step's output is written to its folder;
logs go to <download_dir>/pipeline.log. Stop with Ctrl-C at any time: re-running resumes the download where it
stopped.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINQUAL_PYTHON = Path.home() / "vscode" / "finqual" / "finqual-venv" / "bin" / "python"


def run(cmd: list[str], log: Path) -> None:
    line = " ".join(str(c) for c in cmd)
    print(f"\n$ {line}", flush=True)
    with open(log, "a") as fh:
        fh.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} $ {line}\n")
    rc = subprocess.call([str(c) for c in cmd], cwd=ROOT)
    if rc != 0:
        raise SystemExit(f"step failed (exit {rc}): {line}")


def run_parallel(cmds: list[list], d: Path, log: Path) -> None:
    """Run the download workers side by side (output: <d>/download_shard<i>.log); Ctrl-C stops them all."""
    # one thread per worker: finqual's numeric libraries otherwise start a thread per core in every worker and the
    # workers fight over the CPU (about 20 s a company instead of 12)
    env = {**os.environ, **{v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                              "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS")}}
    procs = []
    for i, cmd in enumerate(cmds):
        line = " ".join(str(c) for c in cmd)
        with open(log, "a") as fh:
            fh.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} $ {line}\n")
        out = open(d / f"worker{i}.out", "a")
        procs.append(subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT))
    print(f"\n{len(procs)} download workers started; follow with: tail -f {d}/download_shard*.log", flush=True)
    try:
        codes = [pr.wait() for pr in procs]
    except KeyboardInterrupt:
        for pr in procs:
            pr.terminate()
        raise SystemExit("stopped; re-run the same command to resume")
    if any(codes):
        raise SystemExit(f"download workers failed (exit codes {codes}); see {d}/worker*.out")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download_dir", type=Path, default=ROOT / "data" / "in" / f"download_{date.today().isoformat()}")
    p.add_argument("--panels_dir", type=Path, default=None, help="Default: <download_dir with download_ -> panels_>.")
    p.add_argument("--finqual_python", type=Path, default=Path(os.environ.get("FINQUAL_PYTHON", DEFAULT_FINQUAL_PYTHON)))
    p.add_argument("--user_agent", default=None, help="Contact for SEC's User-Agent header, e.g. 'Name name@example.org'.")
    p.add_argument("--start_year", type=int, default=2000, help="First year requested (finqual returns what exists, ~2006–2008 on).")
    p.add_argument("--start", default="auto", help="First quarter of the filled panel (auto: first quarter with 20 %% of firms).")
    p.add_argument("--complete_start", default="2011Q1", help="First quarter of the complete panel.")
    p.add_argument("--fred_start", default="1990-01-01", help="First date of the FRED macro series.")
    p.add_argument("--end", default=None, help="Last quarter of the panels (default: inferred from the data).")
    p.add_argument("--limit", type=int, default=None, help="Download only the first N companies (smoke test).")
    p.add_argument("--max_requests_per_second", type=float, default=8.0,
                   help="Total cap over all workers (SEC's limit is 10 per second per client).")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download workers (each ~12 s a company and one core; with 4, roughly 6 h for ~7,000 companies).")
    p.add_argument("--norm_mode", default="global", choices=["global", "per_ticker"])
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--skip_metadata", action="store_true")
    p.add_argument("--skip_preprocess", action="store_true")
    p.add_argument("--skip_targets", action="store_true")
    args = p.parse_args()

    d = args.download_dir.resolve()
    d.mkdir(parents=True, exist_ok=True)
    panels = (args.panels_dir or d.parent / (d.name.replace("download_", "panels_", 1) if d.name.startswith("download_") else f"panels_{d.name}")).resolve()
    log = d / "pipeline.log"
    ua = ["--user_agent", args.user_agent] if args.user_agent else []
    if not args.user_agent and not args.skip_download:
        print("NOTE: no --user_agent given; SEC asks for a contact in the User-Agent and may throttle anonymous agents.")

    if not args.skip_download:
        if not args.finqual_python.exists():
            raise SystemExit(f"finqual Python not found: {args.finqual_python} (set --finqual_python or FINQUAL_PYTHON)")
        base = [args.finqual_python, ROOT / "src" / "download" / "download_finqual.py", "--out", d,
                "--start_year", args.start_year, *ua] + (["--limit", args.limit] if args.limit else [])
        run(base + ["--list_only"], log)
        per_worker = args.max_requests_per_second / args.workers
        run_parallel([base + ["--shard", i, "--n_shards", args.workers, "--no_merge", "--max_requests_per_second", per_worker]
                      for i in range(args.workers)], d, log)
        run(base + ["--merge_only"], log)
    if not args.skip_metadata:
        cmd = [sys.executable, ROOT / "src" / "download" / "sec_metadata.py", "--download_dir", d, *ua]
        if args.limit:
            cmd += ["--only_downloaded"]
        run(cmd, log)
        run([sys.executable, ROOT / "src" / "download" / "fred.py", "--out", d / "fred", "--start", args.fred_start], log)
    cmd = [sys.executable, ROOT / "src" / "preprocess" / "build_panels.py", "--download_dir", d, "--out_dir", panels,
           "--start", args.start, "--complete_start", args.complete_start] + (["--end", args.end] if args.end else [])
    run(cmd, log)
    import pandas as pd
    # what src/main.py reads next to the panels: the macro series and the firms' sectors
    run([sys.executable, ROOT / "src" / "preprocess" / "preprocess_exog.py", "--in_dir", d / "fred", "--out_dir", panels], log)
    meta = pd.read_parquet(d / "metadata.parquet")
    meta.assign(sector=meta.sic_description)[["ticker", "cik", "sector"]].to_parquet(panels / "metadata.parquet")
    if not args.skip_preprocess:
        for name, extra in [("complete", []), ("filled", ["--allow_missing"])]:
            n_firms = pd.read_parquet(panels / name / "bs-raw.parquet", columns=["ticker"]).ticker.nunique()
            if n_firms < 2:
                print(f"\nskipping preprocess.py for {name}/: {n_firms} firm(s), too few for a train/test split")
                continue
            run([sys.executable, ROOT / "src" / "preprocess" / "preprocess.py", "--out_dir", panels / name,
                 "--norm_mode", args.norm_mode, *extra], log)
    if not args.skip_targets:
        run([sys.executable, ROOT / "src" / "preprocess" / "pillar_targets.py", "--panel", panels / "filled",
             "--metadata", d / "metadata.parquet"], log)
    print(f"\nDone. Download: {d}\nPanels: {panels} (README.md, report.json; Pillar 2-3 targets in filled/targets/)")


if __name__ == "__main__":
    main()
