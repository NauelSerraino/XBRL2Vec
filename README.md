# XBRL2Vec: firm embeddings from financial statements

XBRL2Vec turns 12 quarters of a company's financial statements (SEC XBRL filings) into a 32-number embedding. Two
firms are similar when their embeddings have a high cosine. The encoder is trained on two targets at once: the
firm's current financial structure (Pillar 1: 32 ratios of its last quarter) and its recent dynamics (Pillar 2: the
year-on-year moves of those ratios inside the window). The embeddings are then scored on tasks they never saw:
peer firms, industry, groups, size and growth.

Accounting data only: no prices, no text, no industry codes as inputs.

## Pipeline

Three steps; [src/main.py](src/main.py) runs them in order, and each also runs alone:

| step | script | output |
| --- | --- | --- |
| 1. data | [src/main_download.py](src/main_download.py): download (finqual), SEC metadata, gap-filled panel, amounts, the 32-ratio fingerprint, the balanced sample | `data/in/download_<date>/`, `data/in/panels_<date>/filled/` |
| 2. train | [src/main_similarity.py](src/main_similarity.py): the paper model and its two ablations, 10 seeds each (skips runs already trained) | `data/out/similarity/<run>/encoder.pth` |
| 3. evaluate | [src/usefulness_eval.py](src/usefulness_eval.py): scores of every run and the baselines, then the battery (figures, ablation, comparison, robustness) | `<run>/usefulness.csv`, `model-evaluation/*/out/` |

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
  main.py            the three steps in order
  main_download.py   1. data
  main_similarity.py 2. train
  usefulness_eval.py 3. evaluate
model-evaluation/    the evaluation battery, run by step 3 (each folder writes to its out/)
tests/               pytest
slides/              presentation
docs/                private submodule: the code walkthrough, the model in formulas, the protocol
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
xbrl2vec/bin/python src/main.py --user_agent "<name> <email>"           # everything; the download takes a night

xbrl2vec/bin/python src/main_download.py --user_agent "<name> <email>"  # 1. data (resumable)
xbrl2vec/bin/python src/main_similarity.py                              # 2. train
xbrl2vec/bin/python src/usefulness_eval.py                              # 3. evaluate
```

`docs/code_walkthrough.md` follows every step in detail: files, functions, shapes, formulas, and how the runs are
named and trained.
