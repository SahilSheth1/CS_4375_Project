"""
Per-field confidence scoring and calibration for ReceiptViT.

Pipeline:
  1. predict_with_confidence       — softmax-max confidence per field
  2. fit_temperature               — single-scalar T fit by LBFGS on NLL
  3. fit_all_temperatures          — per-field T over a val loader, saved to JSON
  4. compute_ece                   — Expected Calibration Error
  5. evaluate_calibration          — per-field ECE before/after temperature scaling
  6. plot_reliability_diagram      — confidence-vs-accuracy bar chart
  7. precision_coverage_curve      — selective-prediction curves per field
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

# Make sibling modules importable when this file is run directly.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from vit_model   import ReceiptViT
from field_vocab import FieldVocab, FIELDS, _DF_COL
from data_loader import SROIEDataset, load_sroie_split, collate_fn


REPO_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_CKPT  = REPO_ROOT / "Experiments" / "checkpoints" / "exp2" / "best_model.pt"
DEFAULT_VOCAB = REPO_ROOT / "Experiments" / "vocab.json"
DEFAULT_TEMPS = REPO_ROOT / "Experiments" / "temperatures.json"
DEFAULT_PLOTS = REPO_ROOT / "Experiments" / "calibration_plots"
DATASET_BASE  = REPO_ROOT / "sroie-receipt-dataset" / "SROIE2019"


# ---------------------------------------------------------------------------
# 1. Inference with confidence
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_confidence(
    model:        ReceiptViT,
    images:       torch.Tensor,
    vocab:        FieldVocab,
    temperatures: dict[str, float] | None = None,
) -> dict[str, dict[str, list]]:
    """
    Forward pass returning the top label and its softmax probability per field.

    Returns: {field: {"label": [str, ...], "confidence": [float, ...]}}
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)

    logits_dict = model(images)
    out: dict[str, dict[str, list]] = {}
    for field, logits in logits_dict.items():
        T = float((temperatures or {}).get(field, 1.0))
        T = max(T, 1e-3)
        probs     = torch.softmax(logits / T, dim=-1)
        conf, idx = probs.max(dim=-1)
        labels    = [vocab.decode(field, int(i)) for i in idx.cpu().tolist()]
        out[field] = {
            "label":      labels,
            "confidence": [float(c) for c in conf.cpu().tolist()],
        }
    return out


# ---------------------------------------------------------------------------
# 2. Temperature scaling — single field
# ---------------------------------------------------------------------------

def fit_temperature(
    logits_val: torch.Tensor,
    labels_val: torch.Tensor,
    max_iter:   int = 100,
) -> float:
    """
    Learn a single positive scalar T that minimises NLL of (logits / T) on
    the supplied validation logits/labels (both frozen). Standard post-hoc
    calibration from Guo et al., 2017.
    """
    logits_val = logits_val.detach().cpu()
    labels_val = labels_val.detach().cpu().long()

    T   = nn.Parameter(torch.ones(1))
    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = nll(logits_val / T.clamp(min=1e-3), labels_val)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


# ---------------------------------------------------------------------------
# Helper — collect logits and ground-truth indices per field over a loader
# ---------------------------------------------------------------------------

def _collect_logits_labels(
    model:  ReceiptViT,
    loader: DataLoader,
    device: torch.device,
    vocab:  FieldVocab,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.eval()
    field_logits: dict[str, list[torch.Tensor]] = {f: [] for f in FIELDS}
    field_labels: dict[str, list[torch.Tensor]] = {f: [] for f in FIELDS}

    with torch.no_grad():
        for images, annotations in loader:
            images = images.to(device)
            logits = model(images)
            for field in FIELDS:
                col     = _DF_COL[field]
                indices = [vocab.encode(field, ann.get(col, "")) for ann in annotations]
                field_logits[field].append(logits[field].detach().cpu())
                field_labels[field].append(torch.tensor(indices, dtype=torch.long))

    return (
        {f: torch.cat(field_logits[f], dim=0) for f in FIELDS},
        {f: torch.cat(field_labels[f], dim=0) for f in FIELDS},
    )


# ---------------------------------------------------------------------------
# 3. Per-field temperature fit + persistence
# ---------------------------------------------------------------------------

def fit_all_temperatures(
    model:      ReceiptViT,
    val_loader: DataLoader,
    device:     torch.device,
    vocab:      FieldVocab,
    save_path:  str | Path = DEFAULT_TEMPS,
) -> dict[str, float]:
    """
    Run model over val_loader, fit one temperature per field head, save to JSON.

    `vocab` is required to encode the string annotations the loader yields into
    integer class indices, matching trainer._encode_batch.
    """
    field_logits, field_labels = _collect_logits_labels(model, val_loader, device, vocab)

    temperatures: dict[str, float] = {}
    for field in FIELDS:
        T = fit_temperature(field_logits[field], field_labels[field])
        temperatures[field] = T
        print(f"  {field:8s}  T = {T:.4f}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fh:
        json.dump(temperatures, fh, indent=2)
    print(f"✓ Temperatures saved to {save_path}")
    return temperatures


# ---------------------------------------------------------------------------
# 4. Expected Calibration Error
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    accuracies:  np.ndarray,
    n_bins:      int = 10,
) -> float:
    """
    Expected Calibration Error.

    confidences: 1D array of top-1 softmax probabilities, in [0, 1].
    accuracies:  1D array of 0/1 indicating whether the top-1 prediction was right.
    """
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    accuracies  = np.asarray(accuracies,  dtype=np.float64).reshape(-1)
    n = confidences.shape[0]
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences <  hi)
        m = int(mask.sum())
        if m == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = accuracies[mask].mean()
        ece += (m / n) * abs(avg_conf - avg_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# 5. Calibration evaluation (raw vs. temperature-scaled)
# ---------------------------------------------------------------------------

def _conf_acc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    T:      float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    T = max(float(T), 1e-3)
    probs       = torch.softmax(logits / T, dim=-1)
    conf, pred  = probs.max(dim=-1)
    correct     = (pred == labels).float()
    return conf.numpy(), correct.numpy()


def evaluate_calibration(
    model:        ReceiptViT,
    val_loader:   DataLoader,
    device:       torch.device,
    vocab:        FieldVocab,
    temperatures: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Per-field ECE before and after temperature scaling.

    Returns: {field: {"ece_raw": float,
                      "ece_calibrated": float | None,
                      "mean_confidence": float}}
    """
    field_logits, field_labels = _collect_logits_labels(model, val_loader, device, vocab)

    results: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        conf_raw, acc_raw = _conf_acc_from_logits(field_logits[field], field_labels[field], T=1.0)
        ece_raw = compute_ece(conf_raw, acc_raw)

        if temperatures and field in temperatures:
            T = temperatures[field]
            conf_cal, acc_cal = _conf_acc_from_logits(field_logits[field], field_labels[field], T=T)
            ece_cal   = compute_ece(conf_cal, acc_cal)
            mean_conf = float(conf_cal.mean())
        else:
            ece_cal   = None
            mean_conf = float(conf_raw.mean())

        results[field] = {
            "ece_raw":         ece_raw,
            "ece_calibrated":  ece_cal,
            "mean_confidence": mean_conf,
        }
    return results


# ---------------------------------------------------------------------------
# 6. Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies:  np.ndarray,
    field_name:  str,
    n_bins:      int = 10,
    save_path:   str | Path | None = None,
) -> None:
    """
    Bar chart of binned accuracy vs. binned confidence, with the y=x ideal line.
    """
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    accuracies  = np.asarray(accuracies,  dtype=np.float64).reshape(-1)

    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    bin_acc = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences <  hi)
        if int(mask.sum()) > 0:
            bin_acc[i] = accuracies[mask].mean()

    width = 1.0 / n_bins
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(centers, bin_acc, width=width * 0.95,
           edgecolor="black", color="steelblue", alpha=0.85, label="Accuracy")
    ax.plot([0, 1], [0, 1], "--", color="red", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability — {field_name}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Precision / coverage curve
# ---------------------------------------------------------------------------

def precision_coverage_curve(
    model:        ReceiptViT,
    val_loader:   DataLoader,
    device:       torch.device,
    vocab:        FieldVocab,
    temperatures: dict[str, float] | None = None,
    thresholds:   np.ndarray | None = None,
) -> dict[str, dict[str, list]]:
    """
    Selective-prediction curve per field. For each confidence threshold t:
      coverage  = fraction of samples with confidence >= t
      precision = accuracy on the retained subset
    """
    field_logits, field_labels = _collect_logits_labels(model, val_loader, device, vocab)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    results: dict[str, dict[str, list]] = {}
    for field in FIELDS:
        T = float((temperatures or {}).get(field, 1.0))
        conf, correct = _conf_acc_from_logits(field_logits[field], field_labels[field], T=T)
        n = len(conf)

        precs: list[float] = []
        covs:  list[float] = []
        for t in thresholds:
            mask = conf >= t
            kept = int(mask.sum())
            if kept == 0:
                precs.append(float("nan"))
                covs.append(0.0)
            else:
                precs.append(float(correct[mask].mean()))
                covs.append(float(kept / n))

        results[field] = {
            "thresholds": [float(t) for t in thresholds],
            "precision":  precs,
            "coverage":   covs,
        }
    return results


# ---------------------------------------------------------------------------
# __main__: end-to-end calibration on the saved Exp2 checkpoint
# ---------------------------------------------------------------------------

def _build_val_loader(image_size: int = 384, batch_size: int = 16) -> DataLoader:
    """
    Build the val DataLoader with a transform sized to match the trained model.
    Supplies an explicit transform (rather than calling get_dataloaders) so the
    image size is decoupled from data_loader.RESOLUTION and pinned to whatever
    the checkpoint expects.
    """
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    _, df_val, _ = load_sroie_split(str(DATASET_BASE))
    val_dataset = SROIEDataset(df_val, base_path=str(DATASET_BASE), transform=val_transform)
    return DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )


def main() -> None:
    # Architecture must match the checkpoint at Experiments/checkpoints/exp2/best_model.pt.
    # The Exp2 checkpoint was trained at 384x384 (24x24=576 positional embeddings),
    # which matches data_loader.RESOLUTION = 384.
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
    state = torch.load(DEFAULT_CKPT, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"  loaded {DEFAULT_CKPT}")

    print("\n── Building val loader ──")
    val_loader = _build_val_loader(image_size=IMAGE_SIZE, batch_size=16)
    print(f"  {len(val_loader.dataset)} val samples")

    print("\n── Fitting per-field temperatures ──")
    temperatures = fit_all_temperatures(
        model, val_loader, device, vocab, save_path=DEFAULT_TEMPS,
    )

    print("\n── Calibration metrics (val) ──")
    cal = evaluate_calibration(model, val_loader, device, vocab, temperatures=temperatures)
    print(f"\n{'Field':10s}  {'ECE raw':>10s}  {'ECE calib':>10s}  {'mean conf':>10s}")
    print("-" * 46)
    for field, m in cal.items():
        ece_cal = m["ece_calibrated"] if m["ece_calibrated"] is not None else float("nan")
        print(f"{field:10s}  {m['ece_raw']:10.4f}  {ece_cal:10.4f}  {m['mean_confidence']:10.4f}")

    print("\n── Saving reliability diagrams ──")
    DEFAULT_PLOTS.mkdir(parents=True, exist_ok=True)
    field_logits, field_labels = _collect_logits_labels(model, val_loader, device, vocab)
    for field in FIELDS:
        T = temperatures.get(field, 1.0)
        conf, correct = _conf_acc_from_logits(field_logits[field], field_labels[field], T=T)
        out_path = DEFAULT_PLOTS / f"reliability_{field}.png"
        plot_reliability_diagram(conf, correct, field_name=field, save_path=out_path)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
