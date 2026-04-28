"""
Per-field confidence thresholds for selective prediction on ReceiptViT.

Each field head emits a calibrated softmax confidence (see conf_scoring.py).
This module decides, per field, the cutoff below which the model abstains
and the receipt is flagged for human review. The chosen thresholds drive the
`review_rate` column in experiment_log.csv.

Three threshold-selection modes:
  - "target_precision":  smallest t such that precision on retained samples >= P.
                         Best when wrong predictions are costly. Less stable on a
                         small val set because the tail-precision estimate is noisy.
  - "target_coverage":   largest t such that fraction retained >= C.
                         Best when the review budget is fixed. Most stable on small
                         val sets because coverage estimates are well-behaved.
  - "max_f1":            sweep t, treat below-threshold as abstain, pick t maximising
                         selective F1 = 2PR/(P+R) where R = correct_retained / total.
                         Balanced default when no operating constraint is fixed.

Stability note: the SROIE val split is ~63 receipts (90/10 of the train split).
Thresholds picked there carry meaningful sampling noise — prefer "target_coverage"
or "max_f1" if your val set is this small.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from vit_model         import ReceiptViT
from field_vocab       import FieldVocab, FIELDS
from conf_scoring      import (
    predict_with_confidence,
    _collect_logits_labels,
    _conf_acc_from_logits,
    _build_val_loader,
    DATASET_BASE,
    DEFAULT_CKPT,
    DEFAULT_VOCAB,
    DEFAULT_TEMPS,
)
from data_loader       import SROIEDataset, load_sroie_split, collate_fn
from torchvision       import transforms
from experiment_logger import log_experiment


REPO_ROOT      = Path(__file__).resolve().parent.parent
DEFAULT_THRESH = REPO_ROOT / "Experiments" / "thresholds.json"
DEFAULT_LOG    = REPO_ROOT / "Experiments" / "experiment_log.csv"


# ---------------------------------------------------------------------------
# 1. Per-field threshold selection
# ---------------------------------------------------------------------------

def select_threshold(
    confidences: np.ndarray,
    correct:     np.ndarray,
    mode:        str   = "target_precision",
    target:      float = 0.95,
) -> float:
    """
    Pick a single confidence threshold for one field head.

    confidences: 1D float array in [0, 1] (calibrated softmax max-probabilities).
    correct:     1D bool/0-1 array — True if the top-1 prediction matched the label.
    mode:        "target_precision" | "target_coverage" | "max_f1".
    target:      target precision (mode=target_precision) or target coverage
                 (mode=target_coverage). Ignored when mode="max_f1".

    Stability: the SROIE val split is ~63 receipts. Tail-precision estimates
    needed by "target_precision" are noisy at that size — "target_coverage" and
    "max_f1" are more stable.
    """
    conf    = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct,     dtype=np.float64).reshape(-1)
    n = conf.shape[0]
    if n == 0:
        return 0.0

    if mode == "target_precision":
        # Smallest t such that precision on retained samples >= target.
        # Sweep over unique confidence values (and 0.0 for the keep-all option).
        candidates = np.unique(np.concatenate([[0.0], conf]))
        candidates.sort()
        for t in candidates:
            mask = conf >= t
            kept = int(mask.sum())
            if kept == 0:
                continue
            prec = correct[mask].mean()
            if prec >= target:
                return float(t)
        # No threshold met the target — abstain on everything except the very top.
        return float(conf.max())

    if mode == "target_coverage":
        # Largest t such that fraction retained >= target.
        candidates = np.unique(np.concatenate([[0.0], conf]))
        candidates.sort()
        best_t = 0.0
        for t in candidates:
            cov = float((conf >= t).mean())
            if cov >= target:
                best_t = float(t)
            else:
                break
        return best_t

    if mode == "max_f1":
        # Selective F1 over a dense sweep:
        #   precision = correct_retained / retained
        #   recall    = correct_retained / total
        sweep = np.linspace(0.0, 1.0, 200)
        best_f1 = -1.0
        best_t  = 0.0
        for t in sweep:
            mask = conf >= t
            kept = int(mask.sum())
            if kept == 0:
                f1 = 0.0
            else:
                tp = float(correct[mask].sum())
                p  = tp / kept
                r  = tp / n
                f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = float(t)
        return best_t

    raise ValueError(f"Unknown mode: {mode!r}. Use 'target_precision', 'target_coverage', or 'max_f1'.")


# ---------------------------------------------------------------------------
# 2. Per-field thresholds over a val loader, persisted
# ---------------------------------------------------------------------------

def select_all_thresholds(
    model:        ReceiptViT,
    val_loader:   DataLoader,
    device:       torch.device,
    vocab:        FieldVocab,
    temperatures: dict[str, float],
    mode:         str   = "target_precision",
    target:       float = 0.95,
    save_path:    str | Path | None = DEFAULT_THRESH,
) -> dict[str, float]:
    """
    Fit one threshold per field over val_loader, optionally persist to JSON.

    Mirrors the {field: float} schema of Experiments/temperatures.json.
    Pass save_path=None to skip writing (useful when comparing modes).
    """
    field_logits, field_labels = _collect_logits_labels(model, val_loader, device, vocab)

    thresholds: dict[str, float] = {}
    for field in FIELDS:
        T = float(temperatures.get(field, 1.0))
        conf, correct = _conf_acc_from_logits(field_logits[field], field_labels[field], T=T)
        t = select_threshold(conf, correct, mode=mode, target=target)
        thresholds[field] = t
        print(f"  {field:8s}  t = {t:.4f}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as fh:
            json.dump(thresholds, fh, indent=2)
        print(f"✓ Thresholds saved to {save_path}")

    return thresholds


# ---------------------------------------------------------------------------
# 3. Apply thresholds at inference time
# ---------------------------------------------------------------------------

def apply_thresholds(
    model:         ReceiptViT,
    images:        torch.Tensor,
    vocab:         FieldVocab,
    temperatures:  dict[str, float],
    thresholds:    dict[str, float],
    abstain_token: str = "<REVIEW>",
) -> dict[str, dict[str, list]]:
    """
    Forward pass that returns calibrated label + confidence per field, with the
    label replaced by `abstain_token` whenever confidence < threshold[field].

    Returns: {field: {"label": [...], "confidence": [...], "abstained": [bool, ...]}}
    """
    raw = predict_with_confidence(model, images, vocab, temperatures=temperatures)

    out: dict[str, dict[str, list]] = {}
    for field, payload in raw.items():
        t       = float(thresholds.get(field, 0.0))
        labels  = list(payload["label"])
        confs   = [float(c) for c in payload["confidence"]]
        abstain = [c < t for c in confs]
        labels  = [abstain_token if a else l for l, a in zip(labels, abstain)]
        out[field] = {"label": labels, "confidence": confs, "abstained": abstain}
    return out


# ---------------------------------------------------------------------------
# 4. Operating point on a held-out loader
# ---------------------------------------------------------------------------

def compute_operating_point(
    model:        ReceiptViT,
    loader:       DataLoader,
    device:       torch.device,
    vocab:        FieldVocab,
    temperatures: dict[str, float],
    thresholds:   dict[str, float],
) -> dict[str, Any]:
    """
    Per-field precision/coverage/review_rate plus aggregate review-rate scalars.

    Per field:
      precision_retained = accuracy on samples whose calibrated confidence >= t
      coverage           = fraction of samples retained
      review_rate        = 1 - coverage

    Aggregates:
      mean_review_rate   = mean of per-field review rates
      any_abstained_rate = fraction of receipts where >= 1 field abstained
    """
    field_logits, field_labels = _collect_logits_labels(model, loader, device, vocab)

    n_total: int | None = None
    per_field: dict[str, dict[str, float]] = {}
    abstained_matrix: list[np.ndarray] = []   # one bool array per field, length N

    for field in FIELDS:
        T = float(temperatures.get(field, 1.0))
        conf, correct = _conf_acc_from_logits(field_logits[field], field_labels[field], T=T)
        N = conf.shape[0]
        if n_total is None:
            n_total = N

        t        = float(thresholds.get(field, 0.0))
        retained = conf >= t
        kept     = int(retained.sum())

        coverage    = kept / N if N > 0 else 0.0
        review_rate = 1.0 - coverage
        precision   = float(correct[retained].mean()) if kept > 0 else float("nan")

        per_field[field] = {
            "precision_retained": precision,
            "coverage":           float(coverage),
            "review_rate":        float(review_rate),
        }
        abstained_matrix.append(~retained)

    mean_review_rate = float(np.mean([per_field[f]["review_rate"] for f in FIELDS]))
    any_abstained    = np.any(np.stack(abstained_matrix, axis=0), axis=0) if abstained_matrix else np.array([])
    any_rate         = float(any_abstained.mean()) if any_abstained.size > 0 else 0.0

    return {
        "per_field":          per_field,
        "mean_review_rate":   mean_review_rate,
        "any_abstained_rate": any_rate,
    }


# ---------------------------------------------------------------------------
# 5. Log the operating point's review_rate into experiment_log.csv
# ---------------------------------------------------------------------------

def log_operating_point(
    operating_point_dict: dict[str, Any],
    experiment_name:      str,
    extra_metrics:        dict[str, Any] | None = None,
    log_path:             str | Path = DEFAULT_LOG,
) -> None:
    """
    Write `mean_review_rate` (and any extras whose keys match existing CSV
    columns) into the experiment_log.csv row for `experiment_name`. Columns
    not in the existing header are silently dropped by experiment_logger —
    that's the contract; we don't mutate the schema here.
    """
    results: dict[str, Any] = {
        "review_rate": operating_point_dict.get("mean_review_rate"),
    }
    if extra_metrics:
        results.update(extra_metrics)

    log_experiment(experiment_name, results, path=log_path)


# ---------------------------------------------------------------------------
# Helpers for __main__
# ---------------------------------------------------------------------------

def _build_test_loader(image_size: int = 384, batch_size: int = 16) -> DataLoader:
    """Mirror conf_scoring._build_val_loader for the test split."""
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    _, _, df_test = load_sroie_split(str(DATASET_BASE))
    test_dataset = SROIEDataset(df_test, base_path=str(DATASET_BASE), transform=test_transform)
    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )


def _print_operating_point(op: dict[str, Any]) -> None:
    print(f"\n{'Field':10s}  {'precision':>10s}  {'coverage':>10s}  {'review':>10s}")
    print("-" * 46)
    for field, m in op["per_field"].items():
        prec = m["precision_retained"]
        prec_str = f"{prec:10.4f}" if not np.isnan(prec) else f"{'n/a':>10s}"
        print(f"{field:10s}  {prec_str}  {m['coverage']:10.4f}  {m['review_rate']:10.4f}")
    print(f"\n  mean_review_rate   = {op['mean_review_rate']:.4f}")
    print(f"  any_abstained_rate = {op['any_abstained_rate']:.4f}")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

def main() -> None:
    # Match the Exp2 checkpoint architecture (384x384 / patch=16 / 4L / 4H).
    IMAGE_SIZE = 384
    PATCH_SIZE = 16
    EMBED_DIM  = 256
    NUM_LAYERS = 4
    NUM_HEADS  = 4

    device_str = (
        "mps"  if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    print("\n── Loading vocab ──")
    vocab = FieldVocab.load(DEFAULT_VOCAB)

    print("\n── Loading temperatures ──")
    with open(DEFAULT_TEMPS) as fh:
        temperatures: dict[str, float] = json.load(fh)
    for f, T in temperatures.items():
        print(f"  {f:8s}  T = {T:.4f}")

    print("\n── Building model and loading checkpoint ──")
    model = ReceiptViT(
        vocab_sizes = vocab.vocab_sizes(),
        image_size  = IMAGE_SIZE,
        patch_size  = PATCH_SIZE,
        in_channels = 3,
        embed_dim   = EMBED_DIM,
        num_layers  = NUM_LAYERS,
        num_heads   = NUM_HEADS,
        dropout     = 0.1,
    )
    model.load_state_dict(torch.load(DEFAULT_CKPT, map_location=device))
    model = model.to(device)
    print(f"  loaded {DEFAULT_CKPT}")

    print("\n── Building val + test loaders ──")
    val_loader  = _build_val_loader(image_size=IMAGE_SIZE, batch_size=16)
    test_loader = _build_test_loader(image_size=IMAGE_SIZE, batch_size=16)
    print(f"  val:  {len(val_loader.dataset)} samples")
    print(f"  test: {len(test_loader.dataset)} samples")

    # --- Compare all three modes side-by-side on val ---
    modes = [
        ("target_precision", 0.95),
        ("target_coverage",  0.80),
        ("max_f1",           None),
    ]

    print("\n── Fitting thresholds in all three modes (no save) ──")
    by_mode: dict[str, dict[str, float]] = {}
    for mode, target in modes:
        label = f"{mode}" + (f" (target={target})" if target is not None else "")
        print(f"\n[{label}]")
        thr = select_all_thresholds(
            model, val_loader, device, vocab, temperatures,
            mode=mode, target=target if target is not None else 0.0,
            save_path=None,
        )
        by_mode[mode] = thr

    print("\n── Side-by-side thresholds ──")
    print(f"{'Field':10s}  {'tgt_prec=0.95':>14s}  {'tgt_cov=0.80':>14s}  {'max_f1':>10s}")
    print("-" * 56)
    for field in FIELDS:
        print(
            f"{field:10s}  "
            f"{by_mode['target_precision'][field]:14.4f}  "
            f"{by_mode['target_coverage'][field]:14.4f}  "
            f"{by_mode['max_f1'][field]:10.4f}"
        )

    # --- Persist max_f1 as the default operating point ---
    print("\n── Saving max_f1 thresholds as default ──")
    chosen = by_mode["max_f1"]
    DEFAULT_THRESH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_THRESH, "w") as fh:
        json.dump(chosen, fh, indent=2)
    print(f"✓ Thresholds saved to {DEFAULT_THRESH}")

    # --- Evaluate operating point on test ---
    print("\n── Operating point on test ──")
    op = compute_operating_point(model, test_loader, device, vocab, temperatures, chosen)
    _print_operating_point(op)

    # --- Log review_rate back into experiment_log.csv ---
    print("\n── Logging review_rate to experiment_log.csv ──")
    log_operating_point(op, experiment_name="Exp2", log_path=DEFAULT_LOG)


if __name__ == "__main__":
    main()
