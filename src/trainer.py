from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit_model   import ReceiptViT
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
    total_loss    = 0.0
    field_correct = {f: 0 for f in FIELDS}
    n_samples     = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, annotations in loader:
            images = images.to(device, non_blocking=True)
            labels = _encode_batch(annotations, vocab)
            labels = {f: t.to(device, non_blocking=True) for f, t in labels.items()}

            logits = model(images)

            # FIX 1: average loss across fields instead of summing
            # Summing made the loss 4x too large, destabilising the optimizer
            loss = sum(criterion(logits[f], labels[f]) for f in FIELDS) / len(FIELDS)

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
    model:          ReceiptViT,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    vocab:          FieldVocab,
    num_epochs:     int   = 20,
    lr:             float = 1e-4,
    device_str:     str   = "cpu",
    checkpoint_dir: str   = "../Experiments/checkpoints",
    warmup_epochs:  int   = 5,
) -> dict[str, Any]:
    device = torch.device(device_str)
    model  = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # FIX 2: linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs      # ramp from lr/warmup_epochs → lr
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pt"

    best_val_acc = 0.0
    history      = []

    # FIX 3: print per-field accuracy so you can see what's actually learning
    header = (f"{'Ep':>3}  {'TrLoss':>7}  "
              + "  ".join(f"{'tr_'+f[:3]:>6}" for f in FIELDS)
              + f"  {'VaLoss':>7}  "
              + "  ".join(f"{'va_'+f[:3]:>6}" for f in FIELDS)
              + "  {'s':>4}")
    print(header)
    print("─" * len(header))

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_stats = _run_epoch(model, train_loader, optimizer, criterion, vocab, device, train=True)
        val_stats   = _run_epoch(model, val_loader,   None,      criterion, vocab, device, train=False)
        scheduler.step()

        elapsed = time.time() - t0

        tr_accs = "  ".join(f"{train_stats['accs'][f]:6.3f}" for f in FIELDS)
        va_accs = "  ".join(f"{val_stats['accs'][f]:6.3f}"   for f in FIELDS)
        print(f"{epoch:3d}  {train_stats['loss']:7.4f}  {tr_accs}  "
              f"{val_stats['loss']:7.4f}  {va_accs}  {elapsed:4.0f}s")

        history.append({
            "epoch":      epoch,
            "train_loss": train_stats["loss"],
            "train_acc":  train_stats["overall_acc"],
            "val_loss":   val_stats["loss"],
            "val_acc":    val_stats["overall_acc"],
            **{f"train_acc_{f}": train_stats["accs"][f] for f in FIELDS},
            **{f"val_acc_{f}":   val_stats["accs"][f]   for f in FIELDS},
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