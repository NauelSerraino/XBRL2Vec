"""
Comparison on the downstream tasks T1 and T2: the paper model against five alternatives, on the unseen firms before the
cutoff and after it -> out/comparison.xlsx (sheet "T1 and T2": the means, the best of each row in bold; "per seed": every
value; "legend": the terms and their papers).
  SIC2              industry codes: the peers and the box are the other firms of the same SIC2 (the literature's benchmark)
  Fama-French 48    the same with the 48 industries of Fama & French (1997), from the 4-digit SIC (Siccodes48.txt)
  raw fingerprint   the 32 ratios of the last quarter, no model
  PCA-32            PCA of the fingerprint and window deltas: the model's linear twin (same two targets, same width 32)
  autoencoder       the model's encoder (same architecture, input, masking, training firms and settings) trained to
                    reconstruct its input window instead of the two pillars: does the objective matter?
Ten values per method: the training seeds of the model and of the autoencoder (42, 1-9; the autoencoders are cached in
out/autoencoder); for the raw fingerprint and PCA-32, which are not trained, 10 seeds of T2's k-means (their T1 has no
randomness); the industry codes have no randomness at all.

    xbrl2vec/bin/python model-evaluation/downstream-tasks/comparison.py
"""
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ablation import BLO, HERE, HP, MODEL, ORDER, OUT, cells, ue, write
from models.xbrl2vec import XBRL2VecEncoder, embed

ALTERNATIVES = {"raw fingerprint": ue.RAW_FP, "PCA-32": ue.PCA32}
VINCENT = ("Vincent, Larochelle, Bengio & Manzagol (2008), \"Extracting and Composing Robust Features with Denoising "
           "Autoencoders\", ICML")


def ff48(sic: np.ndarray) -> np.ndarray:
    """Fama-French 48 industry of each 4-digit SIC (Ken French's Siccodes48.txt), -1 outside every range."""
    out, industry = np.full(len(sic), -1), 0
    for line in (HERE / "Siccodes48.txt").read_text().splitlines():
        if r := re.search(r"(\d{4})-(\d{4})", line):
            out[(sic >= int(r[1])) & (sic <= int(r[2]))] = industry
        elif h := re.match(r"\s*(\d+)\s+\S", line):
            industry = int(h[1])
    return out


def autoencoder(g, seed: int, epochs: int = 20, mask: float = 0.2) -> np.ndarray:
    """The embedding of every window by the model's encoder trained to reconstruct its (masked) input window."""
    T, F = g.X.shape[1:]
    torch.manual_seed(seed)
    enc = XBRL2VecEncoder(T, F, latent_dim=32, hidden=128)
    dec = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, T * F))
    path = OUT / "autoencoder" / f"s{seed}.pth"
    if path.exists():
        enc.load_state_dict(torch.load(path))
    else:
        X, opt, losses = g.X[g.train_rows], torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-3), []
        for _ in range(epochs):
            tot = 0.0
            for idx in torch.randperm(len(X)).split(256):
                x = X[idx]
                loss = nn.functional.mse_loss(dec(enc(x.masked_fill(torch.rand(x.shape) < mask, 0.0))), x.flatten(1))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            losses.append(tot / len(X))
        assert losses[-1] < losses[0], losses                                       # self-check: it learned
        print(f"[INFO] autoencoder seed {seed}: loss {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(enc.state_dict(), path)
    return embed(enc, g.X)


def main() -> None:
    """The model against the five alternatives, into out/comparison.xlsx."""
    runs = ue.runs_of(MODEL, scored=True)
    g, cached = ue.Group(runs[0]), ue.baselines_of(runs[0]).reindex(ORDER)
    pick = lambda rows: pd.DataFrame(rows).query("q == 'all'").set_index(ue.CELL).value.reindex(ORDER)
    C = {"model": cells(MODEL)}
    for name, vec in ALTERNATIVES.items():
        V = g.baselines()[vec][0]()
        C[name] = pd.DataFrame([pick(ue.score(g, V, None, vec, seed=s)) for s in range(10)])
        # k-means seed 0 = the scored baselines, up to float noise (PCA varies ~1e-5 with the BLAS threads)
        assert np.allclose(C[name].iloc[0], cached[vec], atol=1e-3), name
    C["autoencoder"] = pd.DataFrame([pick(ue.score(g, autoencoder(g, ue.seed_of(r)), None, "autoencoder")) for r in runs])
    ff = ff48(g.sic4)
    assert list(ff48(np.array([6021, 2834, 3711]))) == [44, 13, 23], "Banks, Drugs, Autos"
    print(f"[INFO] Fama-French 48: {np.mean(ff < 0):.1%} of the rows outside every industry", flush=True)
    write(C, pd.DataFrame({"model": C["model"].mean(), "SIC2": pick(ue.sic_rows(g)),
                           "Fama-French 48": pick(ue.sic_rows(g, ff, "FF48")),
                           **{name: C[name].mean() for name in [*ALTERNATIVES, "autoencoder"]}}), "comparison.xlsx", [
        ("model", "as in ablation.xlsx: 10 training seeds", ""),
        ("SIC2", "2-digit SIC industry codes: the peers and the box are the other firms of the same SIC2; no randomness",
         f"the benchmark of {BLO} and {HP}"),
        ("Fama-French 48", "the 48 industries of Fama & French, from the 4-digit SIC: the peers and the box are the other "
         "firms of the same industry; no randomness",
         f"Fama & French (1997), \"Industry costs of equity\", Journal of Financial Economics 43(2); a benchmark in {BLO}"),
        ("raw fingerprint", "the 32 ratios of the last quarter, no model; 10 k-means seeds for T2 (T1 has no randomness)", ""),
        ("PCA-32", "PCA (32 components) of the fingerprint and the window deltas, the model's two training targets; no "
         "training; 10 k-means seeds for T2", ""),
        ("autoencoder", "the model's encoder (same architecture, input, 20 % input masking, training firms and settings) "
         "trained to reconstruct its 12-quarter input window instead of the two pillars; 10 training seeds (42, 1–9)",
         f"denoising autoencoder: {VINCENT}")])


if __name__ == "__main__":
    main()
