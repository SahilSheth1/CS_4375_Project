from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit_model  import ReceiptViT
from field_vocab import FieldVocab, FIELDS, _DF_COL

def _encode_batch(annotations: list[dict], vocab: FieldVocab) -> dict[str, torch.Tensor]:
    encoded = {}
    for field in FIELDS:
        col = _DF_COL[field]
        indices = [vocab.encode(field, ann.get(col, "")) for ann in annotations]
        encoded[field] = torch.tensor(indices, dtype=torch.long)
    return encoded


def _run_epoch(
    model:      ReceiptViT,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer | None,
    criterion:  nn.CrossEntropyLoss,
    vocab:      FieldVocab,
    device:     torch.device,
    train:      bool,
) -> dict[str, float]:
    model.train(train)
    total_loss   = 0.0
    field_correct = {f: 0 for f in FIELDS}
    n_samples    = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, annotations in loader:
            images  = images.to(device)
            labels  = _encode_batch(annotations, vocab)
            labels  = {f: t.to(device) for f, t in labels.items()}

            logits  = model(images)

            loss = sum(criterion(logits[f], labels[f]) for f in FIELDS)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            B = images.size(0)
            total_loss += loss.item() * B
            n_samples  += B

            for f in FIELDS:
                preds = logits[f].argmax(dim=-1)
                field_correct[f] += (preds == labels[f]).sum().item()

    avg_loss = total_loss / n_samples
    accs     = {f: field_correct[f] / n_samples for f in FIELDS}
    overall  = sum(accs.values()) / len(FIELDS)
    return {"loss": avg_loss, "accs": accs, "overall_acc": overall}

def train_model(
    model:         ReceiptViT,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    vocab:         FieldVocab,
    num_epochs:    int   = 20,
    lr:            float = 1e-4,
    device_str:    str   = "cpu",
    checkpoint_dir: str  = "../Experiments/checkpoints",
) -> dict[str, Any]:
    device = torch.device(device_str)
    model  = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    ckpt_dir  = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc  = 0.0
    best_ckpt     = ckpt_dir / "best_model.pt"
    history       = []

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("─" * 52)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_stats = _run_epoch(model, train_loader, optimizer, criterion, vocab, device, train=True)
        val_stats   = _run_epoch(model, val_loader,   None,      criterion, vocab, device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"{epoch:5d}  {train_stats['loss']:10.4f}  "
            f"{train_stats['overall_acc']:9.4f}  "
            f"{val_stats['loss']:8.4f}  "
            f"{val_stats['overall_acc']:7.4f}  "
            f"({elapsed:.0f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc":  train_stats["overall_acc"],
            "val_loss":   val_stats["loss"],
            "val_acc":    val_stats["overall_acc"],
        })

        if val_stats["overall_acc"] > best_val_acc:
            best_val_acc = val_stats["overall_acc"]
            torch.save(model.state_dict(), best_ckpt)

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    final = _run_epoch(model, val_loader, None, criterion, vocab, device, train=False)

    results = {
        "train_test_split": "80/10/10",
        "overall_f1":       round(final["overall_acc"], 4),
    }
    for f in FIELDS:
        results[f"exact_match_{f}"] = round(final["accs"][f], 4)

    print(f"\n✓ Best val overall acc: {best_val_acc:.4f}  |  checkpoint: {best_ckpt}")
    return results, history