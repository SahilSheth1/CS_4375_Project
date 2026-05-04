from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

UNK = "<UNK>"
PAD = "<PAD>"
UNK_IDX = 0
PAD_IDX = 1

FIELDS = ["vendor", "date", "total", "address"]
_DF_COL = {"vendor": "company", "date": "date", "total": "total", "address": "address"}

# Maximum output length per field (in characters) - future reference, maybe string if not compatable wit the other methdods
MAX_LEN = {"vendor": 48, "date": 10, "total": 8, "address": 64}
MAX_LEN_INT = max(MAX_LEN.values())  # value is 64 as of Apr 28

CHAR_VOCAB = (
    [UNK, PAD]
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")  # lowercase list 
    + list("0123456789")
    + list(" .,/-:()&'@#%+*=_\"!?;")  # punctuation list - HERE IS WHERE WE ADD MORE IN THE FUTURE
)
CHAR2IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) 


def normalize_label(field: str, value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if field == "date":
        for fmt in (
            "%d/%m/%Y",
            "%d/%m/%y",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d-%m-%y",
            "%d %b %Y",
            "%d %B %Y",
            "%d.%m.%Y",
            "%d.%m.%y",
            "%m/%d/%Y",
            "%m/%d/%y",
        ):
            try:
                return datetime.strptime(value, fmt).strftime("%d/%m/%Y")
            except ValueError:
                pass
        return value.upper()
    if field == "total":
        num = re.sub(r"[^\d.]", "", value)
        try:
            return f"{float(num):.2f}"
        except ValueError:
            return value.upper()
    return re.sub(r"\s+", " ", value.upper())


def encode_chars(field: str, text: str) -> list[int]:
    maxlen = MAX_LEN[field]
    text = normalize_label(field, text)[:maxlen]
    idxs = [CHAR2IDX.get(c, UNK_IDX) for c in text]
    # Pad to maxlen
    idxs += [PAD_IDX] * (maxlen - len(idxs))
    return idxs


def decode_chars(field: str, indices: list[int]) -> str:
    chars = [IDX2CHAR.get(i, "") for i in indices if i not in (PAD_IDX, UNK_IDX)]
    return "".join(chars).strip()


class FieldVocab:
    def __init__(self, vocabs: dict = None):
        # vocabs param kept for load() compatibility but not used
        pass

    @classmethod
    def build(cls, df_train: pd.DataFrame) -> "FieldVocab":
        return cls()

    def encode(self, field: str, label: str) -> list[int]:
        return encode_chars(field, label)

    def decode(self, field: str, indices: list[int] | int) -> str:
        if isinstance(indices, int):
            return IDX2CHAR.get(indices, "")
        return decode_chars(field, indices)

    def size(self, field: str) -> int:
        return CHAR_VOCAB_SIZE

    def vocab_sizes(self) -> dict[str, int]:
        return {f: CHAR_VOCAB_SIZE for f in FIELDS}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"type": "char_level", "vocab": CHAR_VOCAB, "max_len": MAX_LEN},
                f,
                indent=2,
            )
        print(f" Char vocab saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FieldVocab":
        return cls()
