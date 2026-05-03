"""
upload.py — Step 5.2
POST /upload  →  accept receipt image, run model, return fields + confidence scores
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

import model_loader
from schemas import FieldResult, ReceiptResponse

router = APIRouter(prefix="/upload", tags=["upload"])

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    """Load image bytes → normalised (1, 3, 384, 384) tensor."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return TRANSFORM(img).unsqueeze(0)          # (1, 3, 384, 384)


def _run_inference(tensor: torch.Tensor) -> dict[str, FieldResult]:
    """
    Run the ViT model + temperature scaling on a single-image batch.
    Returns a dict of FieldResult objects keyed by field name.
    """
    model        = model_loader.model
    vocab        = model_loader.vocab
    temperatures = model_loader.temperatures
    thresholds   = model_loader.thresholds

    tensor = tensor.to(model_loader.device)

    # predict_with_confidence returns {field: {"label": [...], "confidence": [...]}}
    with torch.no_grad():
        results = model_loader.scorer.predict_with_confidence(tensor, temperatures)

    field_results: dict[str, FieldResult] = {}
    for fname, data in results.items():
        value      = data["label"][0]
        confidence = data["confidence"][0]
        threshold  = thresholds.get(fname, 0.80)
        field_results[fname] = FieldResult(
            value=value,
            confidence=round(confidence, 4),
            needs_review=confidence < threshold,
        )
    return field_results


@router.post("/", response_model=ReceiptResponse)
async def upload_receipt(file: UploadFile = File(...)):
    """
    Upload a receipt image and get back extracted fields with confidence scores.

    - **file**: JPEG or PNG receipt image
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Send JPEG or PNG."
        )

    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image_bytes  = await file.read()
    tensor       = _preprocess(image_bytes)
    field_results = _run_inference(tensor)

    auto_accepted = not any(f.needs_review for f in field_results.values())
    receipt_id    = f"{Path(file.filename).stem}_{uuid.uuid4().hex[:8]}"

    # Persist to review queue on disk so /review endpoint can serve it
    _save_to_queue(receipt_id, field_results, auto_accepted)

    return ReceiptResponse(
        receipt_id=receipt_id,
        fields=field_results,
        auto_accepted=auto_accepted,
        message="Auto-accepted" if auto_accepted else "Flagged for manual review",
    )


def _save_to_queue(
    receipt_id: str,
    field_results: dict[str, "FieldResult"],
    auto_accepted: bool,
) -> None:
    """Append this receipt to the on-disk review queue JSON."""
    import json
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

    from review_queue import ReviewQueue
    from review_queue import FieldResult as QueueField
    from review_queue import ReceiptResult

    queue_path = model_loader.QUEUE_PATH

    if queue_path.exists():
        queue = ReviewQueue.load(queue_path)
    else:
        queue = ReviewQueue(model_loader.thresholds)

    result = ReceiptResult(receipt_id=receipt_id, auto_accepted=auto_accepted)

    from field_vocab import FIELDS
    for fname in FIELDS:
        fr = field_results.get(fname)
        if fr:
            result.fields[fname] = QueueField(
                field_name=fname,
                predicted_text=fr.value,
                confidence=fr.confidence,
                needs_review=fr.needs_review,
            )

    if auto_accepted:
        queue.auto_accepted.append(result)
    else:
        queue.pending_review.append(result)

    queue.save(queue_path)