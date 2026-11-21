# XBRL2Vec: firm embeddings from financial statements

XBRL2Vec turns 12 quarters of a company's financial statements (SEC XBRL filings) into a 32-number embedding. Two
firms are similar when their embeddings have a high cosine. The encoder is trained on two targets at once: the
firm's current financial structure (Pillar 1: 32 ratios of its last quarter) and its recent dynamics (Pillar 2: the
year-on-year moves of those ratios inside the window). The embeddings are then scored on tasks they never saw:
peer firms, industry, groups, size and growth.

Accounting data only: no prices, no text, no industry codes as inputs.

## Pipeline

| step | script | output |
| --- | --- | --- |
| 1–4 | [src/main_download.py](src/main_download.py): download (finqual), SEC metadata, gap-filled panel, preprocessing | `data/in/download_<date>/`, `data/in/panels_<date>/filled/` |
| 5 | [src/preprocess/fingerprint.py](src/preprocess/fingerprint.py): 32 size-free ratios per firm-quarter | `filled/targets/fingerprint_slogz*.parquet` |
| 6 | [src/main_similarity.py](src/main_similarity.py): train one encoder (one seed) | `data/out/similarity/<run>/encoder.pth` |
| 7 | [src/usefulness_eval.py](src/usefulness_eval.py): score runs and baselines | `<run>/usefulness.csv` |
| 8 | [model-evaluation/run-all.py](model-evaluation/run-all.py): figures, ablation, comparison, robustness | `model-evaluation/*/out/` |

The model is a DLinear encoder (trend and remainder of each ratio over 12 quarters) followed by an MLP, about 29,000
parameters. It is trained with a pairwise cosine-matching loss on batches of firms from the same quarter, with 20 % of
the input cells masked.

## Layout

```text
src/
  download/          finqual download, SEC metadata
  preprocess/        gap-filled panel, scaled amounts, fingerprint
  services/          data windows, targets, loss and training loop, constants
  models/            the encoder
  main_download.py   steps 1-4
  main_similarity.py training
  usefulness_eval.py scoring
model-evaluation/    the standard evaluation battery (each folder writes to its out/)
notebooks/           data holes and filling (00), the fingerprint (01), the ablation (07)
tests/               pytest
slides/              presentation
docs/                private submodule: protocol, reports, the code walkthrough
```

## Setup

```bash
python3.11 -m venv xbrl2vec && xbrl2vec/bin/pip install -r requirements.txt
xbrl2vec/bin/python -m pytest tests
```

The download (step 1) runs with a Python that has a fork of [finqual](https://github.com/harryy-he/finqual)
installed, by default `~/vscode/finqual/finqual-venv` (override with `--finqual_python` or `FINQUAL_PYTHON`). SEC asks for a contact
in the User-Agent: pass `--user_agent "<name> <email>"`.

## Running

```bash
xbrl2vec/bin/python src/main_download.py --user_agent "<name> <email>"           # 1-4, resumable
xbrl2vec/bin/python src/preprocess/fingerprint.py --panel data/in/panels_<date>/filled
cd src && ../xbrl2vec/bin/python main_similarity.py --seed 42 --out ../data/out/similarity/<name>
../xbrl2vec/bin/python usefulness_eval.py <name>
cd .. && xbrl2vec/bin/python model-evaluation/run-all.py
```

The first training run writes the balanced firm list and asks for a second fingerprint pass on those firms
(`fingerprint.py --firms`). `docs/code_walkthrough.md` follows every step in detail: files, functions, shapes,
formulas, and how the 40 runs of the paper (4 variants × 10 seeds) are named and trained.
