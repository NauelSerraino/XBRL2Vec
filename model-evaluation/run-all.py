"""
The paper model's standard evaluation battery; each script writes its outputs into its own folder.

    xbrl2vec/bin/python model-evaluation/run-all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ["interpretabilty/interpretability_figures.py", "downstream-tasks/ablation.py",
               "downstream-tasks/comparison.py", "robustness/robustness.py"]:
    subprocess.run([sys.executable, str(HERE / script)], check=True)
