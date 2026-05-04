from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit_model import ReceiptViT
from field_vocab import FieldVocab, FIELDS, _DF_COL, normalize_label


def _encode_batch(annotations, vocab):
    encoded = {}

    for field in FIELDS:
        col = _DF_COL[field]
        batch_idxs = [
            torch.tensor(vocab.encode(field, ann.get(col, "") or ""), dtype=torch.long)
            for ann in annotations
        ]
        encoded[field] = torch.stack(batch_idxs)

    return encoded


def _decoded_exact_match(logits, labels, annotations, vocab):
    correct = {f: 0 for f in FIELDS}
    n = len(annotations)

    for field in FIELDS:
        col = _DF_COL[field]
        preds = logits[field].argmax(dim=-1).detach().cpu().tolist()

        for pred_idx, ann in zip(preds, annotations):
            pred_text = vocab.decode(field, pred_idx)
            true_text = normalize_label(field, ann.get(col, "") or "")

            if pred_text == true_text:
                correct[field] += 1

    return {f: correct[f] / n for f in FIELDS}


def _run_epoch(
    model, loader, optimizer, criterion, vocab, device, train, scheduler=None
):
    model.train(train)

    total_loss = 0.0
    field_correct = {f: 0.0 for f in FIELDS}
    n_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for images, annotations in loader:
            images = images.to(device, non_blocking=True)

            labels = _encode_batch(annotations, vocab)
            labels = {f: t.to(device, non_blocking=True) for f, t in labels.items()}

            logits = model(images)

            loss = sum(
                criterion(logits[f].permute(0, 2, 1), labels[f]) for f in FIELDS
            ) / len(FIELDS)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            B = images.size(0)
            total_loss += loss.item() * B
            n_samples += B

            batch_accs = _decoded_exact_match(logits, labels, annotations, vocab)
            for f in FIELDS:
                field_correct[f] += batch_accs[f] * B

    avg_loss = total_loss / n_samples
    accs = {f: field_correct[f] / n_samples for f in FIELDS}
    overall = sum(accs.values()) / len(FIELDS)

    return {"loss": avg_loss, "accs": accs, "overall_acc": overall}


def train_model(
    model: ReceiptViT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab: FieldVocab,
    num_epochs: int = 20,
    lr: float = 3e-5,
    device_str: str = "cpu",
    checkpoint_dir: str = "../Experiments/checkpoints",
    warmup_epochs: int = 5,
) -> dict[str, Any]:

    device = torch.device(device_str)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Do NOT ignore PAD.
    # The model must learn when the field ends.
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / "best_model.pt"
    hist_path = ckpt_dir / "history.json"
    resl_path = ckpt_dir / "results.json"

    best_val_acc = -1.0
    history = []

    header = (
        f"{'Ep':>3}  {'TrLoss':>7}  "
        + "  ".join(f"{'tr_' + f[:3]:>6}" for f in FIELDS)
        + f"  {'VaLoss':>7}  "
        + "  ".join(f"{'va_' + f[:3]:>6}" for f in FIELDS)
        + f"  {'s':>4}"
    )

    print(header)
    print("─" * len(header))

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_stats = _run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            vocab,
            device,
            train=True,
            scheduler=scheduler,
        )

        val_stats = _run_epoch(
            model, val_loader, None, criterion, vocab, device, train=False
        )

        elapsed = time.time() - t0

        tr_accs = "  ".join(f"{train_stats['accs'][f]:6.3f}" for f in FIELDS)
        va_accs = "  ".join(f"{val_stats['accs'][f]:6.3f}" for f in FIELDS)

        print(
            f"{epoch:3d}  {train_stats['loss']:7.4f}  {tr_accs}  "
            f"{val_stats['loss']:7.4f}  {va_accs}  {elapsed:4.0f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["overall_acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["overall_acc"],
                **{f"train_acc_{f}": train_stats["accs"][f] for f in FIELDS},
                **{f"val_acc_{f}": val_stats["accs"][f] for f in FIELDS},
            }
        )

        if val_stats["overall_acc"] > best_val_acc:
            best_val_acc = val_stats["overall_acc"]
            torch.save(model.state_dict(), best_ckpt)

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    final = _run_epoch(model, val_loader, None, criterion, vocab, device, train=False)

    results = {
        "train_test_split": "80/10/10",
        "overall_f1": round(final["overall_acc"], 4),
    }

    for f in FIELDS:
        results[f"exact_match_{f}"] = round(final["accs"][f], 4)

    with open(resl_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Best val overall acc: {best_val_acc:.4f}  |  checkpoint: {best_ckpt}")

    return results, history
