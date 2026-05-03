"""
review.py — Placeholder for Step 5.3
GET  /review          →  list receipts pending human review
POST /review/{id}     →  submit corrections for a receipt
"""

from fastapi import APIRouter

router = APIRouter(prefix="/review", tags=["review"])


@router.get("/")
def review_placeholder():
    return {"message": "Step 5.3 — GET /review endpoint coming soon"}


@router.post("/{receipt_id}")
def correction_placeholder(receipt_id: str):
    return {"message": f"Step 5.3 — POST /review/{receipt_id} endpoint coming soon"}