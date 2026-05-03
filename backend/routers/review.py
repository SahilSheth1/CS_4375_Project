"""
review.py — Step 5.3
GET  /review              →  list all receipts pending human review
POST /review/{receipt_id} →  submit corrections for a receipt
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

import model_loader
from schemas import (
    CorrectionRequest,
    CorrectionResponse,
    FieldResult,
    ReviewItem,
    ReviewListResponse,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from review_queue import ReviewQueue

router = APIRouter(prefix="/review", tags=["review"])


def _load_queue() -> ReviewQueue:
    """Load the current review queue from disk."""
    if not model_loader.QUEUE_PATH.exists():
        return ReviewQueue(model_loader.thresholds)
    return ReviewQueue.load(model_loader.QUEUE_PATH)


def _queue_field_to_schema(fr) -> FieldResult:
    """Convert a review_queue.FieldResult to a schemas.FieldResult."""
    return FieldResult(
        value=fr.corrected_text if fr.corrected_text else fr.predicted_text,
        confidence=fr.confidence,
        needs_review=fr.needs_review,
    )


@router.get("/", response_model=ReviewListResponse)
def get_review_queue():
    """
    Return all receipts currently pending human review.
    Only includes receipts where at least one field is below the confidence threshold.
    """
    queue = _load_queue()

    items = [
        ReviewItem(
            receipt_id=r.receipt_id,
            reviewed=r.reviewed,
            fields={
                fname: _queue_field_to_schema(fr)
                for fname, fr in r.fields.items()
            },
        )
        for r in queue.pending_review
    ]

    return ReviewListResponse(total=len(items), items=items)


@router.get("/{receipt_id}", response_model=ReviewItem)
def get_receipt(receipt_id: str):
    """
    Return a single receipt from the review queue by ID.
    """
    queue = _load_queue()

    for r in queue.pending_review:
        if r.receipt_id == receipt_id:
            return ReviewItem(
                receipt_id=r.receipt_id,
                reviewed=r.reviewed,
                fields={
                    fname: _queue_field_to_schema(fr)
                    for fname, fr in r.fields.items()
                },
            )

    raise HTTPException(
        status_code=404,
        detail=f"Receipt '{receipt_id}' not found in review queue."
    )


@router.post("/{receipt_id}", response_model=CorrectionResponse)
def submit_correction(receipt_id: str, body: CorrectionRequest):
    """
    Submit human corrections for a flagged receipt.

    Only send corrections for the fields that need fixing —
    unflagged fields don't need to be included.

    Example body:
```json
    {
      "corrections": {
        "vendor": "GARDENIA BAKERIES (KL) SDN BHD",
        "total":  "12.50"
      }
    }
```
    """
    queue = _load_queue()

    result = queue.apply_correction(receipt_id, body.corrections)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Receipt '{receipt_id}' not found in review queue."
        )

    queue.save(model_loader.QUEUE_PATH)

    return CorrectionResponse(
        receipt_id=receipt_id,
        status="corrected",
        corrections_applied=body.corrections,
    )


@router.get("/stats/summary")
def get_review_stats():
    """
    Return review queue statistics — useful for the report and the dashboard.
    """
    queue = _load_queue()
    summary = queue.summary()
    return {
        **summary,
        "unreviewed": sum(1 for r in queue.pending_review if not r.reviewed),
    }