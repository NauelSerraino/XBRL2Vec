"""
XBRL2Vec end to end, in three steps; each one also runs alone:

  1. data      src/main_download.py     download, SEC metadata, filled panel, amounts, fingerprint, balanced sample
  2. train     src/main_similarity.py   the paper model and its two ablations, 10 seeds each (skips runs already done)
  3. evaluate  src/usefulness_eval.py   the scores of every run and the baselines, then the battery of model-evaluation/

    xbrl2vec/bin/python src/main.py --user_agent "<name> <email>"

Step 1 downloads into data/in/download_<today>/ (a night; resumable). On a laptop, keep it awake for the whole run:
    systemd-inhibit --what=handle-lid-switch:sleep:idle xbrl2vec/bin/python src/main.py --user_agent "<name> <email>"
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def main() -> None:
    """Run the three steps in order; stop at the first that fails."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user_agent", default=None, help="Contact for SEC's User-Agent header (step 1).")
    args = p.parse_args()
    ua = ["--user_agent", args.user_agent] if args.user_agent else []
    for step in (["main_download.py", *ua], ["main_similarity.py"], ["usefulness_eval.py"]):
        subprocess.run([sys.executable, str(SRC / step[0]), *step[1:]], check=True)


if __name__ == "__main__":
    main()
