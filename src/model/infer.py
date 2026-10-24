"""Load the fine-tuned T5 and correct text (runs on CPU)."""
import re
from pathlib import Path

PREFIX = "grammar: "          # must match the prefix used in training
MAX_SRC, MAX_NEW = 128, 128   # tokens
_SPLIT = re.compile(r"((?<=[.!?])\s+|\n+)")   # sentence/paragraph boundaries, kept as separators
_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def normalize(text: str) -> str:
    """The model was trained on ASCII text: map curly quotes and dashes to plain ones."""
    return text.translate(_QUOTES)


def split_text(text: str) -> list:
    """[sentence, separator, sentence, ...]; ''.join(result) == text."""
    return _SPLIT.split(text)


class Corrector:
    def __init__(self, model_dir, quantize: bool = False):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(Path(model_dir)))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(Path(model_dir))).eval()
        if quantize:                                   # int8 linear layers: faster on CPU, may cost some accuracy
            try:
                self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            except Exception as e:                     # keep working in fp32 if this torch build cannot quantize
                print("quantization failed, using fp32:", e)

    def correct_sentences(self, sentences, num_beams: int = 4, batch_size: int = 8) -> list:
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))   # similar lengths -> less padding
        out = [None] * len(sentences)
        with self.torch.inference_mode():
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                enc = self.tok([PREFIX + sentences[j] for j in idx], return_tensors="pt",
                               padding=True, truncation=True, max_length=MAX_SRC)
                gen = self.model.generate(**enc, num_beams=num_beams, max_new_tokens=MAX_NEW)
                for j, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True)):
                    out[j] = t
        return out

    def correct_text(self, text: str, num_beams: int = 4) -> str:
        """Correct a paragraph sentence by sentence, keeping line breaks and spacing."""
        pieces = split_text(text)
        todo = [i for i in range(0, len(pieces), 2) if len(pieces[i].strip()) > 1]
        fixed = self.correct_sentences([pieces[i].strip() for i in todo], num_beams)
        for i, f in zip(todo, fixed):
            s = pieces[i]
            lead, trail = len(s) - len(s.lstrip()), len(s) - len(s.rstrip())
            pieces[i] = s[:lead] + f + (s[len(s) - trail:] if trail else "")
        return "".join(pieces)
