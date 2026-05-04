from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class FieldResult(BaseModel):
    value: str
    confidence: float
    needs_review: bool


class ReceiptResponse(BaseModel):
    receipt_id: str
    fields: dict[str, FieldResult]
    auto_accepted: bool
    message: str


class CorrectionRequest(BaseModel):
    corrections: dict[str, str]


class CorrectionResponse(BaseModel):
    receipt_id: str
    status: str
    corrections_applied: dict[str, str]


class ReviewItem(BaseModel):
    receipt_id: str
    reviewed: bool
    fields: dict[str, FieldResult]


class ReviewListResponse(BaseModel):
    total: int
    items: list[ReviewItem]
