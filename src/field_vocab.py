from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

UNK = "<UNK>"
UNK_IDX = 0

FIELDS = ["vendor", "date", "total", "address"]

_DF_COL = {"vendor": "company", "date": "date", "total": "total", "address": "address"}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_label(field: str, value: str) -> str:
    """Normalize a raw label string before vocab build and at inference time."""
    if not value:
        return ""
    value = value.strip()

    if field == "date":
        # Try common date formats → canonical DD/MM/YYYY
        for fmt in (
            "%d/%m/%Y", "%d/%m/%y",
            "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y",
            "%d %b %Y", "%d %B %Y",
            "%d.%m.%Y", "%d.%m.%y",
            "%m/%d/%Y", "%m/%d/%y",
        ):
            try:
                return datetime.strptime(value, fmt).strftime("%d/%m/%Y")
            except ValueError:
                pass
        return value.upper()

    if field == "total":
        # Strip currency symbols and thousands separators, keep decimal
        num = re.sub(r"[^\d.]", "", value)
        try:
            return f"{float(num):.2f}"
        except ValueError:
            return value.upper()

    # vendor and address — uppercase + collapse whitespace
    return re.sub(r"\s+", " ", value.upper())


# ---------------------------------------------------------------------------
# FieldVocab
# ---------------------------------------------------------------------------

class FieldVocab:
    def __init__(self, vocabs: dict[str, dict[str, int]]):
        self.vocabs = vocabs
        self.inv    = {f: {v: k for k, v in vocabs[f].items()} for f in FIELDS}

    @classmethod
    def build(cls, df_train: pd.DataFrame) -> "FieldVocab":
        vocabs = {}
        for field in FIELDS:
            col = _DF_COL[field]
            raw_labels = df_train[col].dropna().astype(str).tolist()
            # Normalize before building vocab so unseen formats still map correctly
            labels = sorted(set(normalize_label(field, l) for l in raw_labels))
            labels = [l for l in labels if l]   # drop empty strings
            mapping = {UNK: UNK_IDX}
            for i, lbl in enumerate(labels, start=1):
                mapping[lbl] = i
            vocabs[field] = mapping
        return cls(vocabs)

    def encode(self, field: str, label: str) -> int:
        """Normalize then encode; unknown values map to UNK_IDX."""
        normalized = normalize_label(field, label)
        return self.vocabs[field].get(normalized, UNK_IDX)

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