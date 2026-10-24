"""Token-level diff between the original text and the corrected text."""
import difflib
import html
import re

_TOKEN = re.compile(r"\s+|\w+(?:'\w+)*|[^\w\s]")      # whitespace, words, punctuation: all kept, so ''.join(tokens) == text


def diff_ops(original: str, corrected: str) -> list:
    """Return a list of {'a': original piece, 'b': corrected piece, 'changed': bool}.
    Concatenating all 'a' gives `original`; all 'b' gives `corrected`."""
    a, b = _TOKEN.findall(original), _TOKEN.findall(corrected)
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        sa, sb = "".join(a[i1:i2]), "".join(b[j1:j2])
        changed = tag != "equal" and not (not sa.strip() and not sb.strip())   # whitespace-only edits are not "errors"
        ops.append({"a": sa, "b": sb, "changed": changed})
    return ops


def apply_choices(ops: list, accepted: dict) -> str:
    """Rebuild the text: for each changed op, take the correction if accepted.get(index, True), else keep the original."""
    return "".join(o["b"] if (not o["changed"] or accepted.get(i, True)) else o["a"] for i, o in enumerate(ops))


def render_html(ops: list) -> tuple:
    """(original_html, corrected_html): errors underlined on the left, fixes highlighted on the right."""
    esc = lambda s: html.escape(s).replace("\n", "<br>")
    left, right = [], []
    for o in ops:
        if not o["changed"]:
            left.append(esc(o["a"])); right.append(esc(o["b"]))
            continue
        left.append(f'<span class="rs-err">{esc(o["a"])}</span>' if o["a"].strip() else '<span class="rs-gap"></span>')
        if o["b"].strip():
            right.append(f'<span class="rs-fix">{esc(o["b"])}</span>')
    return "".join(left), "".join(right)
