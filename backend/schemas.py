"""
schemas.py — Pydantic models for all request and response bodies
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class FieldResult(BaseModel):
    value: str                  # predicted text
    confidence: float           # calibrated probability in [0, 1]
    needs_review: bool          # True if confidence < threshold


class ReceiptResponse(BaseModel):
    receipt_id: str
    fields: dict[str, FieldResult]   # keys: vendor, date, total, address
    auto_accepted: bool              # True if ALL fields above threshold
    message: str


class CorrectionRequest(BaseModel):
    corrections: dict[str, str]      # {field_name: corrected_text}


class CorrectionResponse(BaseModel):
    receipt_id: str
    status: str
    corrections_applied: dict[str, str]


class ReviewItem(BaseModel):
    receipt_id: str
    fields: dict[str, FieldResult]
    reviewed: bool


class ReviewListResponse(BaseModel):
    total: int
    items: list[ReviewItem]