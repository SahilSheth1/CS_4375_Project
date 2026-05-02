"""
Class-based confidence scoring and calibration for ReceiptViT.
Import this in notebooks instead of running conf_scoring.py as a script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from vit_model import ReceiptViT
from field_vocab import FieldVocab, FIELDS, _DF_COL
from data_loader import SROIEDataset, load_sroie_split, collate_fn


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "Experiments" / "checkpoints" / "exp2" / "best_model.pt"
DEFAULT_VOCAB = REPO_ROOT / "Experiments" / "vocab.json"
DEFAULT_TEMPS = REPO_ROOT / "Experiments" / "temperatures.json"
DEFAULT_PLOTS = REPO_ROOT / "Experiments" / "calibration_plots"
DATASET_BASE = REPO_ROOT / "sroie-receipt-dataset" / "SROIE2019"


class ConfidenceScorer:
    def __init__(self, model: ReceiptViT, vocab: FieldVocab, device: torch.device | str):
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_with_confidence(
        self,
        images: torch.Tensor,
        temperatures: dict[str, float] | None = None,
    ) -> dict[str, dict[str, list]]:
        images = images.to(self.device)
        logits_dict = self.model(images)

        out: dict[str, dict[str, list]] = {}
        for field, logits in logits_dict.items():
            T = float((temperatures or {}).get(field, 1.0))
            T = max(T, 1e-3)
            probs = torch.softmax(logits / T, dim=-1)
            conf, idx = probs.max(dim=-1)
            labels = [self.vocab.decode(field, int(i)) for i in idx.cpu().tolist()]
            out[field] = {
                "label": labels,
                "confidence": [float(c) for c in conf.cpu().tolist()],
            }
        return out

    def collect_logits_labels(
        self,
        loader: DataLoader,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.model.eval()
        field_logits: dict[str, list[torch.Tensor]] = {f: [] for f in FIELDS}
        field_labels: dict[str, list[torch.Tensor]] = {f: [] for f in FIELDS}

        with torch.no_grad():
            for images, annotations in loader:
                images = images.to(self.device)
                logits = self.model(images)
                for field in FIELDS:
                    col = _DF_COL[field]
                    indices = [self.vocab.encode(field, ann.get(col, "")) for ann in annotations]
                    field_logits[field].append(logits[field].detach().cpu())
                    field_labels[field].append(torch.tensor(indices, dtype=torch.long))

        return (
            {f: torch.cat(field_logits[f], dim=0) for f in FIELDS},
            {f: torch.cat(field_labels[f], dim=0) for f in FIELDS},
        )

    @staticmethod
    def fit_temperature(logits_val: torch.Tensor, labels_val: torch.Tensor, max_iter: int = 100) -> float:
        logits_val = logits_val.detach().cpu()
        labels_val = labels_val.detach().cpu().long()

        T = nn.Parameter(torch.ones(1))
        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(logits_val / T.clamp(min=1e-3), labels_val)
            loss.backward()
            return loss

        opt.step(closure)
        return float(T.detach().clamp(min=1e-3).item())

    def fit_all_temperatures(
        self,
        val_loader: DataLoader,
        save_path: str | Path | None = DEFAULT_TEMPS,
    ) -> dict[str, float]:
        field_logits, field_labels = self.collect_logits_labels(val_loader)

        temperatures: dict[str, float] = {}
        for field in FIELDS:
            T = self.fit_temperature(field_logits[field], field_labels[field])
            temperatures[field] = T
            print(f"  {field:8s}  T = {T:.4f}")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fh:
                json.dump(temperatures, fh, indent=2)
            print(f"✓ Temperatures saved to {save_path}")

        return temperatures

    @staticmethod
    def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
        accuracies = np.asarray(accuracies, dtype=np.float64).reshape(-1)
        n = confidences.shape[0]
        if n == 0:
            return 0.0

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            m = int(mask.sum())
            if m == 0:
                continue
            ece += (m / n) * abs(confidences[mask].mean() - accuracies[mask].mean())
        return float(ece)

    @staticmethod
    def conf_acc_from_logits(
        logits: torch.Tensor,
        labels: torch.Tensor,
        T: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        T = max(float(T), 1e-3)
        probs = torch.softmax(logits / T, dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = (pred == labels).float()
        return conf.numpy(), correct.numpy()

    def evaluate_calibration(
        self,
        val_loader: DataLoader,
        temperatures: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        field_logits, field_labels = self.collect_logits_labels(val_loader)

        results: dict[str, dict[str, float]] = {}
        for field in FIELDS:
            conf_raw, acc_raw = self.conf_acc_from_logits(field_logits[field], field_labels[field], T=1.0)
            ece_raw = self.compute_ece(conf_raw, acc_raw)

            if temperatures and field in temperatures:
                conf_cal, acc_cal = self.conf_acc_from_logits(
                    field_logits[field], field_labels[field], T=temperatures[field]
                )
                ece_cal = self.compute_ece(conf_cal, acc_cal)
                mean_conf = float(conf_cal.mean())
            else:
                ece_cal = None
                mean_conf = float(conf_raw.mean())

            results[field] = {
                "ece_raw": ece_raw,
                "ece_calibrated": ece_cal,
                "mean_confidence": mean_conf,
            }
        return results

    @staticmethod
    def plot_reliability_diagram(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        field_name: str,
        n_bins: int = 10,
        save_path: str | Path | None = None,
    ) -> None:
        confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
        accuracies = np.asarray(accuracies, dtype=np.float64).reshape(-1)

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        bin_acc = np.zeros(n_bins)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            if int(mask.sum()) > 0:
                bin_acc[i] = accuracies[mask].mean()

        width = 1.0 / n_bins
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(centers, bin_acc, width=width * 0.95, edgecolor="black", alpha=0.85, label="Accuracy")
        ax.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability — {field_name}")
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=120)
        plt.close(fig)

    def save_reliability_diagrams(
        self,
        val_loader: DataLoader,
        temperatures: dict[str, float],
        save_dir: str | Path = DEFAULT_PLOTS,
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        field_logits, field_labels = self.collect_logits_labels(val_loader)

        for field in FIELDS:
            T = temperatures.get(field, 1.0)
            conf, correct = self.conf_acc_from_logits(field_logits[field], field_labels[field], T=T)
            out_path = save_dir / f"reliability_{field}.png"
            self.plot_reliability_diagram(conf, correct, field_name=field, save_path=out_path)
            print(f"  saved {out_path}")

    def precision_coverage_curve(
        self,
        val_loader: DataLoader,
        temperatures: dict[str, float] | None = None,
        thresholds: np.ndarray | None = None,
    ) -> dict[str, dict[str, list]]:
        field_logits, field_labels = self.collect_logits_labels(val_loader)
        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 21)

        results: dict[str, dict[str, list]] = {}
        for field in FIELDS:
            T = float((temperatures or {}).get(field, 1.0))
            conf, correct = self.conf_acc_from_logits(field_logits[field], field_labels[field], T=T)
            n = len(conf)

            precs: list[float] = []
            covs: list[float] = []
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
                "precision": precs,
                "coverage": covs,
            }
        return results


class ConfidenceDataModule:
    @staticmethod
    def build_val_loader(
        dataset_base: str | Path = DATASET_BASE,
        image_size: int = 384,
        batch_size: int = 16,
        num_workers: int = 0,
    ) -> DataLoader:
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _, df_val, _ = load_sroie_split(str(dataset_base))
        val_dataset = SROIEDataset(df_val, base_path=str(dataset_base), transform=val_transform)
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
