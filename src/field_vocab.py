from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

UNK = "<UNK>"
UNK_IDX = 0

FIELDS = ["vendor", "date", "total", "address"]

_DF_COL = {"vendor": "company", "date": "date", "total": "total", "address": "address"}


class FieldVocab:
    def __init__(self, vocabs: dict[str, dict[str, int]]):
        self.vocabs   = vocabs
        self.inv      = {f: {v: k for k, v in vocabs[f].items()} for f in FIELDS}

    @classmethod
    def build(cls, df_train: pd.DataFrame) -> "FieldVocab":
        vocabs = {}
        for field in FIELDS:
            col = _DF_COL[field]
            labels = df_train[col].dropna().astype(str).str.strip().unique().tolist()
            labels = sorted(set(labels))          # deterministic ordering
            mapping = {UNK: UNK_IDX}
            for i, lbl in enumerate(labels, start=1):
                mapping[lbl] = i
            vocabs[field] = mapping
        return cls(vocabs)

    def encode(self, field: str, label: str) -> int:
        return self.vocabs[field].get(label.strip(), UNK_IDX)

    def decode(self, field: str, idx: int) -> str:
        return self.inv[field].get(idx, UNK)

    def size(self, field: str) -> int:
        return len(self.vocabs[field])

    def vocab_sizes(self) -> dict[str, int]:
        return {f: self.size(f) for f in FIELDS}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.vocabs, fh, indent=2)
        print(f"✓ Vocab saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FieldVocab":
        with open(path) as fh:
            vocabs = json.load(fh)
        return cls(vocabs)