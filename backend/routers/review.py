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

# path configuration
# makes sure backend can access logic in src directory
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from review_queue import ReviewQueue

# initialize router
router = APIRouter(prefix="/review", tags=["review"])


def _load_queue() -> ReviewQueue:
    if not model_loader.QUEUE_PATH.exists():
        return ReviewQueue(model_loader.thresholds)
    return ReviewQueue.load(model_loader.QUEUE_PATH)


def _queue_field_to_schema(fr) -> FieldResult:
    return FieldResult(
        value=fr.corrected_text if fr.corrected_text else fr.predicted_text,
        confidence=fr.confidence,
        needs_review=fr.needs_review,
    )


@router.get("/", response_model=ReviewListResponse)
def get_review_queue():
    queue = _load_queue()

    items = [
        ReviewItem(
            receipt_id=r.receipt_id,
            reviewed=r.reviewed,
            fields={
                fname: _queue_field_to_schema(fr) for fname, fr in r.fields.items()
            },
        )
        for r in queue.pending_review
    ]

    return ReviewListResponse(total=len(items), items=items)


@router.get("/{receipt_id}", response_model=ReviewItem)
def get_receipt(receipt_id: str):
    queue = _load_queue()

    for r in queue.pending_review:
        if r.receipt_id == receipt_id:
            return ReviewItem(
                receipt_id=r.receipt_id,
                reviewed=r.reviewed,
                fields={
                    fname: _queue_field_to_schema(fr) for fname, fr in r.fields.items()
                },
            )

    raise HTTPException(
        status_code=404, detail=f"Receipt '{receipt_id}' not found in review queue."
    )


@router.post("/{receipt_id}", response_model=CorrectionResponse)
def submit_correction(receipt_id: str, body: CorrectionRequest):
    queue = _load_queue()
    
    result = queue.apply_correction(receipt_id, body.corrections)

    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Receipt '{receipt_id}' not found in review queue."
        )

    # persist changes to disk immediately
    queue.save(model_loader.QUEUE_PATH)

    return CorrectionResponse(
        receipt_id=receipt_id,
        status="corrected",
        corrections_applied=body.corrections,
    )


@router.get("/stats/summary")
def get_review_stats():
    queue = _load_queue()
    summary = queue.summary()
    return {
        **summary,
        "unreviewed": sum(1 for r in queue.pending_review if not r.reviewed),
    }
