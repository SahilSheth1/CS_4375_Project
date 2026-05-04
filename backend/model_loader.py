from __future__ import annotations
from src.data_setup import setup_all

import json
import sys
import torch
from pathlib import Path

# Make src/ importable from the backend folder
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from vit_model import ReceiptViT
from field_vocab import FieldVocab

# Paths
CKPT_PATH = REPO_ROOT / "Experiments" / "checkpoints" / "exp3" / "best_model.pt"
VOCAB_PATH = REPO_ROOT / "Experiments" / "vocab.json"
TEMPS_PATH = REPO_ROOT / "Experiments" / "temperatures.json"
THRESH_PATH = REPO_ROOT / "Experiments" / "thresholds.json"
QUEUE_PATH = REPO_ROOT / "Experiments" / "review_queue.json"

IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 256

# Device
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model() -> tuple[ReceiptViT, FieldVocab, dict, dict]:
    setup_all()
    vocab = FieldVocab.load(VOCAB_PATH)

    model = ReceiptViT(
        vocab_sizes=vocab.vocab_sizes(),
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        embed_dim=EMBED_DIM,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
    )
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with open(TEMPS_PATH) as f:
        temperatures = json.load(f)
    with open(THRESH_PATH) as f:
        raw_thresh = json.load(f)

    thresholds = raw_thresh

    print(f"[model_loader] Model loaded on {device}")
    print(f"[model_loader] Thresholds: {thresholds}")
    return model, vocab, temperatures, thresholds


# Shared state (populated at startup)
model: ReceiptViT | None = None
vocab: FieldVocab | None = None
temperatures: dict | None = None
thresholds: dict | None = None
# ConfidenceScorer, typed as object to avoid circular import
scorer: object | None = None
