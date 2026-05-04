from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from field_vocab import FIELDS, _DF_COL


# Data classes
@dataclass
class FieldResult:
    field_name: str
    predicted_text: str
    # calibrated probability in [0, 1]
    confidence: float
    needs_review: bool = False
    # filled in after human review
    corrected_text: Optional[str] = None


@dataclass
class ReceiptResult:
    # filename or unique ID
    receipt_id: str
    fields: Dict[str, FieldResult] = field(default_factory=dict)
    # True iff ALL fields above threshold
    auto_accepted: bool = False
    # True after a human has confirmed it
    reviewed: bool = False

    def flag_fields(self, thresholds: Dict[str, float]) -> None:
        any_flagged = False
        for fname, result in self.fields.items():
            # fallback to 0.80 if missing
            t = thresholds.get(fname, 0.80)
            result.needs_review = result.confidence < t
            if result.needs_review:
                any_flagged = True
        self.auto_accepted = not any_flagged

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ReviewQueue


class ReviewQueue:
    def __init__(self, thresholds: Dict[str, float]):

        self.thresholds: Dict[str, float] = thresholds
        self.auto_accepted: List[ReceiptResult] = []
        self.pending_review: List[ReceiptResult] = []

    # Core ingestion

    def ingest(
        self,
        receipt_id: str,
        field_predictions: Dict[str, str],
        field_confidences: Dict[str, float],
    ) -> ReceiptResult:
        result = ReceiptResult(receipt_id=receipt_id)

        for fname in FIELDS:
            pred = field_predictions.get(fname, "")
            conf = field_confidences.get(fname, 0.0)
            result.fields[fname] = FieldResult(
                field_name=fname,
                predicted_text=pred,
                confidence=conf,
            )

        result.flag_fields(self.thresholds)

        if result.auto_accepted:
            self.auto_accepted.append(result)
        else:
            self.pending_review.append(result)

        return result

    # Human correction

    def apply_correction(
        self,
        receipt_id: str,
        corrections: Dict[str, str],
    ) -> Optional[ReceiptResult]:
        for result in self.pending_review:
            if result.receipt_id == receipt_id:
                for fname, text in corrections.items():
                    if fname in result.fields:
                        result.fields[fname].corrected_text = text
                result.reviewed = True
                return result
        return None

    # Statistics

    def review_rate(self) -> float:
        total = len(self.auto_accepted) + len(self.pending_review)
        if total == 0:
            return 0.0
        return len(self.pending_review) / total

    def summary(self) -> Dict:
        total = len(self.auto_accepted) + len(self.pending_review)
        reviewed = sum(1 for r in self.pending_review if r.reviewed)
        return {
            "total_processed": total,
            "auto_accepted": len(self.auto_accepted),
            "pending_review": len(self.pending_review),
            "human_reviewed": reviewed,
            "review_rate": round(self.review_rate(), 4),
            "thresholds_used": self.thresholds,
        }

    # Persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "summary": self.summary(),
            "auto_accepted": [r.to_dict() for r in self.auto_accepted],
            "pending_review": [r.to_dict() for r in self.pending_review],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"ReviewQueue saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ReviewQueue":
        path = Path(path)
        with open(path) as f:
            state = json.load(f)

        thresholds = state["summary"]["thresholds_used"]
        queue = cls(thresholds)

        for d in state["auto_accepted"]:
            r = ReceiptResult(
                receipt_id=d["receipt_id"], auto_accepted=True, reviewed=d["reviewed"]
            )
            for fname, fd in d["fields"].items():
                r.fields[fname] = FieldResult(**fd)
            queue.auto_accepted.append(r)

        for d in state["pending_review"]:
            r = ReceiptResult(
                receipt_id=d["receipt_id"], auto_accepted=False, reviewed=d["reviewed"]
            )
            for fname, fd in d["fields"].items():
                r.fields[fname] = FieldResult(**fd)
            queue.pending_review.append(r)

        return queue


# Convenience: run the full test set through the queue


def build_review_queue_from_loader(
    model,
    loader,
    vocab,
    scorer,
    thresholds: Dict[str, float],
    device,
    temperatures: Optional[Dict[str, float]] = None,
) -> Tuple[ReviewQueue, List[dict]]:
    from field_vocab import FIELDS, _DF_COL

    queue = ReviewQueue(thresholds)
    row_records = []

    for images, annotations in loader:
        # predict_with_confidence handles .to(device) and no_grad internally
        results_batch = scorer.predict_with_confidence(
            images, temperatures=temperatures
        )
        # results_batch keys may be FIELDS values ('vendor') OR _DF_COL values ('company')
        # detect once per batch and map accordingly
        sample_keys = set(results_batch.keys())
        use_col_key = sample_keys == set(_DF_COL.values())

        batch_size = images.size(0)
        for i in range(batch_size):
            ann = annotations[i]
            receipt_id = ann.get("file", f"receipt_{i}")

            field_preds = {}
            field_confs = {}
            for fname in FIELDS:
                rb_key = _DF_COL[fname] if use_col_key else fname
                field_preds[fname] = results_batch[rb_key]["label"][i]
                field_confs[fname] = results_batch[rb_key]["confidence"][i]

            result = queue.ingest(receipt_id, field_preds, field_confs)

            row_records.append(
                {
                    "receipt_id": receipt_id,
                    "auto_accepted": result.auto_accepted,
                    **{f"conf_{f}": result.fields[f].confidence for f in FIELDS},
                    **{f"pred_{f}": result.fields[f].predicted_text for f in FIELDS},
                    **{f"gt_{f}": ann.get(_DF_COL[f], "") for f in FIELDS},
                }
            )

    return queue, row_records
