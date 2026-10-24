"""Error analysis of the model on a sample of the test set. From the repo root:

    python -m src.evaluation.error_analysis --model models/t5-base --n 200 --identity 100

Needs data/test.jsonl (python -m src.data.build_pairs ...). Outputs, in --out (default reports/):
    error_summary.csv    counts per error type, no corpus text: safe to commit
    error_over_correction.csv   how often the model changes sentences that were already correct
    error_analysis.xlsx  sentence-level table WITH corpus text: local only, gitignored
Edit types are assigned by simple rules (heuristic): read them as approximate.
"""
import argparse
import difflib
import re
import time
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import evaluate

SEED = 42
TOKEN = re.compile(r"\w+(?:'\w+)*|[^\w\s]")
ARTICLES = {"a", "an", "the"}
PREPS = {"in", "on", "at", "of", "for", "to", "with", "by", "from", "about", "into", "over", "under", "after", "before"}
_IRREGULAR = ["go went gone", "buy bought", "have has had", "do does did done", "see saw seen", "take took taken",
              "make made", "get got gotten", "come came", "write wrote written", "eat ate eaten", "say said",
              "think thought", "know knew known", "give gave given", "find found", "tell told", "run ran",
              "leave left", "bring brought", "begin began begun", "speak spoke spoken", "be am is are was were been",
              "teach taught", "catch caught", "sleep slept", "feel felt", "meet met", "lose lost", "read read"]
LEMMA = {w: g.split()[0] for g in _IRREGULAR for w in g.split()}
AGREEMENT = {"is", "are", "was", "were", "am", "be", "been", "has", "have", "had", "does", "do", "did"}


def edits(src_tokens, other_tokens):
    """Word-level edits turning src into other: (start, end, replacement tokens)."""
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src_tokens, other_tokens, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:          # same-length replacement: one edit per word
            out += [(i1 + k, i1 + k + 1, (other_tokens[j1 + k],)) for k in range(i2 - i1)]
        else:
            out.append((i1, i2, tuple(other_tokens[j1:j2])))
    return out


def classify(a, b, known):
    """Edit type from the source tokens `a` and the replacement tokens `b`. `known` = dictionary words."""
    la, lb = [t.lower() for t in a], [t.lower() for t in b]
    punct = lambda ts: bool(ts) and all(not t[0].isalnum() for t in ts)
    if not a:
        if punct(b): return "punctuation"
        if len(b) == 1 and lb[0] in ARTICLES: return "article"
        if len(b) == 1 and lb[0] in PREPS: return "preposition"
        return "missing word"
    if not b:
        if punct(a): return "punctuation"
        if len(a) == 1 and la[0] in ARTICLES: return "article"
        if len(a) == 1 and la[0] in PREPS: return "preposition"
        return "extra word"
    if la == lb: return "casing"
    if punct(a) and punct(b): return "punctuation"
    if len(a) == 1 and len(b) == 1:
        x, y = la[0], lb[0]
        if x in ARTICLES and y in ARTICLES: return "article"
        if x in PREPS and y in PREPS: return "preposition"
        if x in AGREEMENT and y in AGREEMENT: return "agreement"
        if x in LEMMA and LEMMA.get(x) == LEMMA.get(y): return "word form"     # irregular verb forms
        ratio = difflib.SequenceMatcher(None, x, y).ratio()
        if x not in known and ratio >= 0.7: return "spelling"
        if ratio >= 0.6 or x[:3] == y[:3]: return "word form"      # plural, tense, derivation
        return "word choice"
    return "phrase rewrite"


def analyse(df, known):
    """df columns: source, target, prediction, kind ('changed' | 'identity'). Returns (sentences, summary, over_correction)."""
    norm = lambda s: " ".join(s.split())
    rows, gold_n, rec_n, mod_n, mat_n = [], {}, {}, {}, {}
    for r in df.itertuples():
        st = TOKEN.findall(r.source)
        gold, model = edits(st, TOKEN.findall(r.target)), edits(st, TOKEN.findall(r.prediction))
        G, M = set(gold), set(model)
        show = lambda e: f"{' '.join(st[e[0]:e[1]]) or '∅'} -> {' '.join(e[2]) or '∅'}"
        for e in gold:
            c = classify(st[e[0]:e[1]], list(e[2]), known)
            gold_n[c] = gold_n.get(c, 0) + 1
            rec_n[c] = rec_n.get(c, 0) + (e in M)
        for e in model:
            c = classify(st[e[0]:e[1]], list(e[2]), known)
            mod_n[c] = mod_n.get(c, 0) + 1
            mat_n[c] = mat_n.get(c, 0) + (e in G)
        rows.append({"kind": r.kind, "source": r.source, "reference": r.target, "prediction": r.prediction,
                     "exact": norm(r.prediction) == norm(r.target),
                     "missed (gold edits not made)": "; ".join(show(e) for e in gold if e not in M),
                     "unmatched (model edits not in gold)": "; ".join(show(e) for e in model if e not in G)})
    cats = sorted(set(gold_n) | set(mod_n), key=lambda c: -gold_n.get(c, 0))
    summary = pd.DataFrame([{
        "edit_type": c, "gold_edits": gold_n.get(c, 0), "recovered": rec_n.get(c, 0),
        "recall_%": round(100 * rec_n.get(c, 0) / gold_n[c], 1) if gold_n.get(c) else None,
        "model_edits": mod_n.get(c, 0), "matched_gold": mat_n.get(c, 0),
        "matched_%": round(100 * mat_n.get(c, 0) / mod_n[c], 1) if mod_n.get(c) else None} for c in cats])
    ident = df[df.kind == "identity"]
    over = pd.DataFrame([{"already_correct_sentences": len(ident),
                          "changed_by_model": int((ident.prediction.map(norm) != ident.source.map(norm)).sum()),
                          "changed_%": round(100 * (ident.prediction.map(norm) != ident.source.map(norm)).mean(), 1) if len(ident) else None}])
    return pd.DataFrame(rows), summary, over


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/t5-base", type=Path)
    ap.add_argument("--data", default="data/test.jsonl", type=Path)
    ap.add_argument("--n", type=int, default=200, help="sentences whose reference differs from the source")
    ap.add_argument("--identity", type=int, default=100, help="sentences that were already correct")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--out", default=Path("reports"), type=Path)
    args = ap.parse_args()

    from src.model.infer import Corrector, _get_sym
    test = pd.read_json(args.data, lines=True, dtype=False)
    changed = test[test.source != test.target].sample(min(args.n, len(test)), random_state=SEED).assign(kind="changed")
    same = test[test.source == test.target]
    same = same.sample(min(args.identity, len(same)), random_state=SEED).assign(kind="identity")
    df = pd.concat([changed, same]).reset_index(drop=True)
    print(f"{len(changed)} changed + {len(same)} already-correct sentences; model {args.model}, beams {args.beams}")

    corr = Corrector(args.model)
    t0 = time.time()
    df["prediction"] = corr.correct_sentences(list(df.source), num_beams=args.beams)
    print(f"correction took {time.time() - t0:.0f} s ({(time.time() - t0) / len(df):.2f} s/sentence)")

    sents, summary, over = analyse(df, _get_sym().words)
    ch = df[df.kind == "changed"]
    print("GLEU on the changed sample:", evaluate(ch.source, ch.prediction, ch.target),
          "| copy baseline:", evaluate(ch.source, ch.source, ch.target))
    print(summary.to_string(index=False)); print(over.to_string(index=False))

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "error_summary.csv", index=False)
    over.to_csv(args.out / "error_over_correction.csv", index=False)
    try:
        with pd.ExcelWriter(args.out / "error_analysis.xlsx") as w:
            sents.to_excel(w, sheet_name="sentences", index=False)
            summary.to_excel(w, sheet_name="summary", index=False)
            over.to_excel(w, sheet_name="over_correction", index=False)
        print("wrote", args.out / "error_analysis.xlsx", "(contains corpus text: do not commit)")
    except ImportError:
        sents.to_csv(args.out / "error_analysis_sentences.csv", index=False)
        print("openpyxl missing (pip install openpyxl): wrote a CSV instead")


if __name__ == "__main__":
    main()