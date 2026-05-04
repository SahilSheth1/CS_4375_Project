from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from field_vocab import FieldVocab, FIELDS
from vit_model import ReceiptViT
from data_loader import SROIEDataset, load_sroie_split, collate_fn
from experiment_logger import log_experiment
from conf_scoring_class import (
    ConfidenceScorer,
    DATASET_BASE,
    DEFAULT_CKPT,
    DEFAULT_VOCAB,
    DEFAULT_TEMPS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESH = REPO_ROOT / "Experiments" / "thresholds.json"
DEFAULT_LOG = REPO_ROOT / "Experiments" / "experiment_log.csv"

# Applying to prediction of the model based on conf socres and thresholds - used at the model.ipynb end 
class ConfidenceThresholdManager:
    def __init__(self, scorer: ConfidenceScorer):
        self.scorer = scorer

    @staticmethod
    def select_threshold(
        confidences: np.ndarray,
        correct: np.ndarray,
        mode: str = "target_precision",
        target: float = 0.95,
    ) -> float:
        conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
        correct = np.asarray(correct, dtype=np.float64).reshape(-1)
        n = conf.shape[0]
        if n == 0:
            return 0.0
        # will based on the specific mode ie precision, coverage or maximun f1 score - within the list 
        if mode == "target_precision":
            candidates = np.unique(np.concatenate([[0.0], conf]))
            candidates.sort()
            for t in candidates:
                mask = conf >= t
                kept = int(mask.sum())
                if kept == 0:
                    continue
                precision = correct[mask].mean()
                if precision >= target:
                    return float(t)
            return float(conf.max())
        # at least target percentage of data is held 
        if mode == "target_coverage":
            candidates = np.unique(np.concatenate([[0.0], conf]))
            candidates.sort()
            best_t = 0.0
            for t in candidates:
                coverage = float((conf >= t).mean())
                if coverage >= target:
                    best_t = float(t)
                else:
                    break
            return best_t
        # Best f1 score in the list of them 
        if mode == "max_f1":
            sweep = np.linspace(0.0, 1.0, 200)
            best_f1 = -1.0
            best_t = 0.0
            for t in sweep:
                mask = conf >= t
                kept = int(mask.sum())
                if kept == 0:
                    f1 = 0.0
                else:
                    tp = float(correct[mask].sum())
                    p = tp / kept
                    r = tp / n
                    f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            return best_t

        raise ValueError(
            "mode must be 'target_precision', 'target_coverage', or 'max_f1'"
        )
    # Renovated to work on all thresholds at once instead of one by one in each field - should cover the rest of the operations
    def select_all_thresholds(
        self,
        val_loader: DataLoader,
        temperatures: dict[str, float],
        mode: str = "target_precision",
        target: float = 0.95,
        save_path: str | Path | None = DEFAULT_THRESH,
    ) -> dict[str, float]:
        field_logits, field_labels = self.scorer.collect_logits_labels(val_loader)

        thresholds: dict[str, float] = {}
        for field in FIELDS:
            T = float(temperatures.get(field, 1.0))
            conf, correct = self.scorer.conf_acc_from_logits(
                field_logits[field], field_labels[field], T=T
            )
            t = self.select_threshold(conf, correct, mode=mode, target=target)
            thresholds[field] = t
            print(f"  {field:8s}  t = {t:.4f}")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fh:
                json.dump(thresholds, fh, indent=2)
            print(f" Thresholds saved to {save_path}")

        return thresholds
    # Goes to all batches of images at once 
    def apply_thresholds(
        self,
        images: torch.Tensor,
        temperatures: dict[str, float],
        thresholds: dict[str, float],
        abstain_token: str = "<REVIEW>",
    ) -> dict[str, dict[str, list]]:
        raw = self.scorer.predict_with_confidence(images, temperatures=temperatures)

        out: dict[str, dict[str, list]] = {} # label is string, conf is float, abstained is bool 
        for field, payload in raw.items():
            t = float(thresholds.get(field, 0.0))
            labels = list(payload["label"])
            confs = [float(c) for c in payload["confidence"]]
            abstain = [c < t for c in confs]
            labels = [
                abstain_token if a else label for label, a in zip(labels, abstain)
            ]
            out[field] = {"label": labels, "confidence": confs, "abstained": abstain}
        return out
    # Important for evaluating on valid. set when selecting thresholds in the end.
    def compute_operating_point(
        self,
        loader: DataLoader,
        temperatures: dict[str, float],
        thresholds: dict[str, float],
    ) -> dict[str, Any]:
        field_logits, field_labels = self.scorer.collect_logits_labels(loader)

        per_field: dict[str, dict[str, float]] = {}
        abstained_matrix: list[np.ndarray] = []

        for field in FIELDS:
            T = float(temperatures.get(field, 1.0))
            conf, correct = self.scorer.conf_acc_from_logits(
                field_logits[field], field_labels[field], T=T
            )
            N = conf.shape[0]
            t = float(thresholds.get(field, 0.0))
            retained = conf >= t
            kept = int(retained.sum())

            coverage = kept / N if N > 0 else 0.0
            review_rate = 1.0 - coverage
            precision = float(correct[retained].mean()) if kept > 0 else float("nan")

            per_field[field] = {
                "precision_retained": precision,
                "coverage": float(coverage),
                "review_rate": float(review_rate),
            }
            abstained_matrix.append(~retained)

        mean_review_rate = float(np.mean([per_field[f]["review_rate"] for f in FIELDS]))
        any_abstained = (
            np.any(np.stack(abstained_matrix, axis=0), axis=0)
            if abstained_matrix
            else np.array([])
        )
        any_rate = float(any_abstained.mean()) if any_abstained.size > 0 else 0.0

        return {
            "per_field": per_field,
            "mean_review_rate": mean_review_rate,
            "any_abstained_rate": any_rate,
        }

    # TODO: implement extra metrics such as F1/ preciision and recall 
    @staticmethod
    def log_operating_point(
        operating_point_dict: dict[str, Any],
        experiment_name: str,
        extra_metrics: dict[str, Any] | None = None,
        log_path: str | Path = DEFAULT_LOG,
    ) -> None:
        results: dict[str, Any] = {
            "review_rate": operating_point_dict.get("mean_review_rate")
        }
        if extra_metrics:
            results.update(extra_metrics)
        log_experiment(experiment_name, results, path=log_path)

    @staticmethod
    def print_operating_point(op: dict[str, Any]) -> None:
        print(
            f"\n{'Field':10s}  {'precision':>10s}  {'coverage':>10s}  {'review':>10s}"
        )
        print("-" * 46)
        for field, metrics in op["per_field"].items():
            prec = metrics["precision_retained"]
            prec_str = f"{prec:10.4f}" if not np.isnan(prec) else f"{'n/a':>10s}"
            print(
                f"{field:10s}  {prec_str}  {metrics['coverage']:10.4f}  {metrics['review_rate']:10.4f}"
            )
        print(f"\n  mean_review_rate   = {op['mean_review_rate']:.4f}")
        print(f"  any_abstained_rate = {op['any_abstained_rate']:.4f}")


class ThresholdDataModule:
    @staticmethod
    def build_test_loader(
        dataset_base: str | Path = DATASET_BASE,
        image_size: int = 384,
        batch_size: int = 16,
        num_workers: int = 0,
    ) -> DataLoader:
        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        _, _, df_test = load_sroie_split(str(dataset_base))
        test_dataset = SROIEDataset(
            df_test, base_path=str(dataset_base), transform=test_transform
        )
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
