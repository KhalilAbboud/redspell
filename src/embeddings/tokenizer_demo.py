"""Representation step: subword tokenization and embeddings.

    python -m src.embeddings.tokenizer_demo [--model models/t5-base] [--data data/train.jsonl]

T5 splits words into SentencePiece subwords, so a misspelling is a sequence of known pieces, not <unk>.
Each subword id is mapped to a learned vector (the input embedding); a word vector is the mean of its
subword vectors, a sentence vector is the mean of the encoder's output.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def word_vector(word, tok, emb):
    """Mean of the input embeddings of the word's subwords."""
    ids = tok(word, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        return emb(torch.tensor(ids)).mean(0)


@torch.no_grad()
def sentence_vectors(sentences, tok, model):
    """Mean-pooled encoder output, one vector per sentence."""
    enc = tok(list(sentences), return_tensors="pt", padding=True, truncation=True, max_length=128)
    h = model.get_encoder()(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/t5-base")
    ap.add_argument("--data", default="data/train.jsonl")
    args = ap.parse_args()
    name = args.model if Path(args.model).exists() else "t5-base"      # falls back to the pretrained model (needs internet)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    emb = model.get_input_embeddings()

    print("== Subword split (typos are known pieces, not <unk>) ==")
    for w in ["permission", "premision", "received", "recieved", "the", "teh", "unbelievable", "unbelivable"]:
        print(f"{w:14s} {tok.tokenize(w)}")

    print("\n== Cosine similarity of word vectors (mean of subword embeddings) ==")
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    for a, b in [("premision", "permission"), ("premision", "precision"), ("premision", "banana"),
                 ("recieved", "received"), ("recieved", "banana"), ("teh", "the")]:
        print(f"{a:10s} vs {b:10s} {cos(word_vector(a, tok, emb), word_vector(b, tok, emb)):.3f}")

    print("\n== Sentence vectors (encoder output) ==")
    v = sentence_vectors(["She go to school.", "She goes to school."], tok, model)
    print("shape:", tuple(v.shape), "| cosine:", round(torch.nn.functional.cosine_similarity(v[0], v[1], dim=0).item(), 3))

    if Path(args.data).exists():
        import pandas as pd
        df = pd.read_json(args.data, lines=True, dtype=False).sample(5000, random_state=42)
        ids = tok(list(df.source), add_special_tokens=True)["input_ids"]
        lens = np.array([len(x) for x in ids])
        unk = np.mean([tok.unk_token_id in x for x in ids])
        print(f"\n== Training sample (5000 sentences) ==\ntokens per sentence: mean {lens.mean():.1f}, "
              f"p99.5 {int(np.percentile(lens, 99.5))}, max {lens.max()} | sentences with <unk>: {100 * unk:.2f}%")


if __name__ == "__main__":
    main()