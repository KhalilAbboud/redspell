"""GLEU for grammatical error correction (Napoles et al., 2016) and simple sentence-level rates."""
import math
import re
from collections import Counter

import numpy as np

_tokens = lambda s: re.findall(r"\w+|[^\w\s]", s)
_ngrams = lambda t, n: Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def gleu_corpus(sources, hyps, refs, max_n=4):
    """Corpus-level GLEU, one reference per sentence.
    p_n = (N(H,R) - [N(H,S) - N(H,S,R)]) / N(H), with N(A,B) = sum over n-grams g of min(count_A(g), count_B(g)):
    n-grams copied from the source but absent from the reference are penalized."""
    num, den, hl, rl = [0.0] * max_n, [0] * max_n, 0, 0
    for s, h, r in zip(sources, hyps, refs):
        s, h, r = _tokens(s), _tokens(h), _tokens(r)
        hl += len(h); rl += len(r)
        for n in range(1, max_n + 1):
            H, S, R = _ngrams(h, n), _ngrams(s, n), _ngrams(r, n)
            num[n - 1] += max(sum((H & R).values()) - (sum((H & S).values()) - sum((H & S & R).values())), 0)
            den[n - 1] += sum(H.values())
    if hl == 0 or any(d == 0 for d in den):
        return 0.0
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    return bp * math.exp(sum(math.log(max(a, 1e-9) / d) for a, d in zip(num, den)) / max_n)


def evaluate(sources, hyps, refs):
    sources, hyps, refs = list(sources), list(hyps), list(refs)
    return {"GLEU": round(100 * gleu_corpus(sources, hyps, refs), 2),
            "exact_match_%": round(100 * np.mean([h == r for h, r in zip(hyps, refs)]), 2),
            "changed_%": round(100 * np.mean([h != s for h, s in zip(hyps, sources)]), 2)}