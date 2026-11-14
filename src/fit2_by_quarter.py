"""
fit2 per out-of-time quarter for a run: macro model, no-macro model, macro shuffled across quarters. Does the macro
help in particular quarters (regimes)?

    cd src && ../xbrl2vec/bin/python fit2_by_quarter.py 20261031-160106
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import explain_run as er
from services.similarity import embed
run = sys.argv[1]
cfg, args, sets, D, models, sector = er.load(run)
tr = sets["train firms"]
cut = cfg["cutoff"]
for name, d in [("training firms", tr), ("unseen firms", D)]:
    d = d.subset(d.meta.quarter.gt(cut).to_numpy() & np.isfinite(d.target2).all(1))
    Z = {"macro": embed(models["macro"], d, "cpu", "real"), "no macro": embed(models["no macro"], d, "cpu", "none"),
         "macro shuffled": embed(models["macro"], d, "cpu", "shuffled")}
    U = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    rows = {}
    for q, g in d.meta.groupby("quarter"):
        r = g.index.to_numpy()
        iu = np.triu_indices(len(r), 1); Sd = (U(d.target2[r]) @ U(d.target2[r]).T)[iu]
        rows[q] = {k: spearmanr((U(v[r]) @ U(v[r]).T)[iu], Sd)[0] for k, v in Z.items()} | {"firms": len(r)}
    df = pd.DataFrame(rows).T
    df["macro - no macro"] = df["macro"] - df["no macro"]
    print(f"\n{run} {name}, out of time, fit2 per quarter:\n", df.round(3).to_string())
