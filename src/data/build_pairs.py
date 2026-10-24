"""Build (incorrect, corrected) sentence pairs from the Lang-8 English corpus v1.0.

Usage, from the repo root:
    python -m src.data.build_pairs --src path/to/lang-8-en-1.0 --out data

Input : entries.train and entries.test (tab-separated: number of corrections, serial number, URL,
        sentence number (0 = journal title), learner sentence, correction(s))
Output: train.jsonl, val.jsonl, test.jsonl in --out, one {"source", "target"} object per line.
The corpus is research-only (no redistribution): keep the output directory out of git.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sacremoses import MosesDetokenizer

SEED = 42
_MD = MosesDetokenizer(lang="en")
_STANFORD = [("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("``", '"'), ("''", '"')]


def parse_entries(path: Path) -> pd.DataFrame:
    """One row per corrected sentence (first correction only); journal titles are skipped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6 or cols[3] == "0":      # blank separator / no correction / title
                continue
            src, tgt = cols[4].strip(), cols[5].strip()
            if src and tgt:
                rows.append((src, tgt))
    return pd.DataFrame(rows, columns=["source", "target"])


def detok(s: str) -> str:
    """The corpus is Stanford-tokenized ("did n't ... !"); the app receives natural text."""
    for a, b in _STANFORD:
        s = s.replace(a, b)
    return re.sub(r" n't\b", "n't", _MD.detokenize(s.split()))


def clean(d: pd.DataFrame) -> pd.DataFrame:
    probe = d.source.sample(min(len(d), 2000), random_state=0)
    if probe.str.contains(r" n't| [.,!?]$").mean() > 0.3:
        d = d.assign(source=d.source.map(detok), target=d.target.map(detok))
    d = d[d.source.map(str.isascii) & d.target.map(str.isascii)]
    wc = lambda s: s.str.split().str.len()
    d = d[wc(d.source).between(3, 50) & wc(d.target).between(3, 50)]
    d = d[(wc(d.target) / wc(d.source)).between(0.5, 2.0)]            # drop misaligned pairs
    return d.drop_duplicates("source").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="folder containing entries.train and entries.test")
    ap.add_argument("--out", default=Path("data"), type=Path)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=3_000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tr = clean(parse_entries(args.src / "entries.train")).sample(frac=1, random_state=SEED).reset_index(drop=True)
    te = clean(parse_entries(args.src / "entries.test"))
    tr = tr[~tr.source.isin(set(te.source))].reset_index(drop=True)   # no train/test overlap
    n_val = min(args.n_val, len(tr) // 10)
    va, tr = tr[:n_val], tr[n_val:n_val + args.n_train]
    for name, part in {"train": tr, "val": va, "test": te}.items():
        part.to_json(args.out / f"{name}.jsonl", orient="records", lines=True)
        print(f"{name}: {len(part)} pairs -> {args.out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()