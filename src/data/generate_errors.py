"""Synthetic error injection for English grammar/spelling correction (RedSpell).

Takes clean sentences and returns (incorrect, correct) pairs.
Usage:
    from src.data.generate_errors import make_pair, generate_pairs
"""
import random
import re
from typing import Iterable, Iterator, Optional

# --- error resources ---------------------------------------------------------
KEYBOARD = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz",
    "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}

_CONFUSION_PAIRS = [
    ("their", "there"), ("your", "you're"), ("its", "it's"), ("then", "than"),
    ("to", "too"), ("affect", "effect"), ("lose", "loose"), ("were", "where"),
    ("whether", "weather"), ("accept", "except"), ("principal", "principle"),
]
CONFUSIONS = {}
for a, b in _CONFUSION_PAIRS:
    CONFUSIONS.setdefault(a, []).append(b)
    CONFUSIONS.setdefault(b, []).append(a)

AGREEMENT = {"is": "are", "are": "is", "was": "were", "were": "was",
             "has": "have", "have": "has", "does": "do", "do": "does"}
ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {"in": ["on", "at"], "on": ["in", "at"], "at": ["in", "on"],
                "of": ["for", "from"], "for": ["of", "to"], "with": ["by", "of"]}

_TOKEN_RE = re.compile(r"\s+|[A-Za-z']+|[^A-Za-z'\s]+")


def _is_word(tok: str) -> bool:
    return bool(tok) and (tok[0].isalpha() or tok[0] == "'") and not tok.isspace()


def _match_case(src: str, new: str) -> str:
    if src.isupper() and len(src) > 1:
        return new.upper()
    if src[0].isupper():
        return new[0].upper() + new[1:]
    return new


def _word_idx(tokens: list) -> list:
    return [i for i, t in enumerate(tokens) if _is_word(t)]


# --- individual error operations (mutate `tokens`, return True if applied) ----
def op_typo(tokens, rng):
    cands = [i for i in _word_idx(tokens) if len(tokens[i]) >= 4 and tokens[i].isalpha()]
    if not cands:
        return False
    i = rng.choice(cands)
    w = tokens[i]
    p = rng.randrange(1, len(w) - 1)  # never touch first/last char: keeps typos realistic
    kind = rng.choice(["swap", "delete", "duplicate", "keyboard"])
    if kind == "swap":
        new = w[:p] + w[p + 1] + w[p] + w[p + 2:]
    elif kind == "delete":
        new = w[:p] + w[p + 1:]
    elif kind == "duplicate":
        new = w[:p] + w[p] + w[p:]
    else:
        nb = KEYBOARD.get(w[p].lower())
        if not nb:
            return False
        r = rng.choice(nb)
        new = w[:p] + (r.upper() if w[p].isupper() else r) + w[p + 1:]
    if new == w:
        return False
    tokens[i] = new
    return True


def op_confusion(tokens, rng):
    cands = [i for i in _word_idx(tokens) if tokens[i].lower() in CONFUSIONS]
    if not cands:
        return False
    i = rng.choice(cands)
    tokens[i] = _match_case(tokens[i], rng.choice(CONFUSIONS[tokens[i].lower()]))
    return True


def op_agreement(tokens, rng):
    cands = [i for i in _word_idx(tokens) if tokens[i].lower() in AGREEMENT]
    if not cands:
        return False
    i = rng.choice(cands)
    tokens[i] = _match_case(tokens[i], AGREEMENT[tokens[i].lower()])
    return True


def op_article(tokens, rng):
    cands = [i for i in _word_idx(tokens) if tokens[i].lower() in ARTICLES]
    if not cands:
        return False
    i = rng.choice(cands)
    w = tokens[i].lower()
    if w in {"a", "an"} and rng.random() < 0.4:
        tokens[i] = _match_case(tokens[i], "an" if w == "a" else "a")
        return True
    # drop the article (and the space after it)
    if i + 1 < len(tokens) and tokens[i + 1].isspace():
        del tokens[i:i + 2]
        if i == 0 and tokens and tokens[0][0].isalpha():  # keep sentence capitalisation
            tokens[0] = tokens[0][0].upper() + tokens[0][1:]
        return True
    return False


def op_preposition(tokens, rng):
    cands = [i for i in _word_idx(tokens) if tokens[i].lower() in PREPOSITIONS]
    if not cands:
        return False
    i = rng.choice(cands)
    tokens[i] = _match_case(tokens[i], rng.choice(PREPOSITIONS[tokens[i].lower()]))
    return True


def op_repeat(tokens, rng):
    cands = [i for i in _word_idx(tokens) if tokens[i].isalpha() and len(tokens[i]) <= 6]
    if not cands:
        return False
    i = rng.choice(cands)
    tokens[i:i + 1] = [tokens[i], " ", tokens[i].lower()]
    return True


# (name, function, sampling weight)
OPS = [
    ("typo", op_typo, 0.35),
    ("confusion", op_confusion, 0.15),
    ("agreement", op_agreement, 0.15),
    ("article", op_article, 0.15),
    ("preposition", op_preposition, 0.12),
    ("repeat", op_repeat, 0.08),
]


def make_pair(sentence: str, rng: random.Random,
              n_errors: Optional[int] = None) -> Optional[dict]:
    """Return {'source': incorrect, 'target': sentence, 'errors': [...]} or None."""
    tokens = _TOKEN_RE.findall(sentence)
    if "".join(tokens) != sentence:  # tokenizer round-trip must be lossless
        return None
    if n_errors is None:
        n_errors = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    names = [o[0] for o in OPS]
    funcs = {o[0]: o[1] for o in OPS}
    weights = [o[2] for o in OPS]
    applied = []
    for _ in range(n_errors * 3):  # a few retries: some ops don't apply to a sentence
        if len(applied) >= n_errors:
            break
        name = rng.choices(names, weights=weights)[0]
        if funcs[name](tokens, rng):
            applied.append(name)
    source = "".join(tokens)
    if not applied or source == sentence:
        return None
    return {"source": source, "target": sentence, "errors": applied}


def generate_pairs(sentences: Iterable[str], rng: random.Random,
                   identity_ratio: float = 0.05) -> Iterator[dict]:
    """Yield pairs. `identity_ratio` of them are unchanged (source == target) so the
    model learns to leave correct text alone."""
    for s in sentences:
        if rng.random() < identity_ratio:
            yield {"source": s, "target": s, "errors": []}
            continue
        pair = make_pair(s, rng)
        if pair:
            yield pair


if __name__ == "__main__":
    demo = [
        "The students are going to their classroom after the lecture.",
        "She has finished the report and sent it to the manager.",
        "We need to decide whether the weather is good enough for a trip.",
    ]
    r = random.Random(0)
    for s in demo:
        print(make_pair(s, r))
