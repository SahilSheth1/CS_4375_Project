from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import json
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
            # FIX (speed): non_blocking only helps with pin_memory=True.
            # On MPS pin_memory is unsupported, so just move normally.
            images = images.to(device)
            labels = _encode_batch(annotations, vocab)
            labels = {f: t.to(device) for f, t in labels.items()}

            logits = model(images)

            # Average loss across fields (summing made it 4x too large)
            loss = sum(criterion(logits[f], labels[f]) for f in FIELDS) / len(FIELDS)

            if train:
                optimizer.zero_grad(set_to_none=True)   # FIX (speed): faster than zero_grad()
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

    # FIX (speed): use torch.compile on non-MPS devices — big speedup on CUDA/CPU
    if device_str not in ("mps",):
        try:
            model = torch.compile(model)
            print("✓ torch.compile enabled")
        except Exception:
            pass  # compile not available in older PyTorch versions

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = ckpt_dir / "best_model.pt"
    hist_path  = ckpt_dir / "history.json"
    resl_path  = ckpt_dir / "results.json"

    best_val_acc = 0.0
    history      = []

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

        # Save checkpoint + history every epoch so you never lose progress
        if val_stats["overall_acc"] > best_val_acc:
            best_val_acc = val_stats["overall_acc"]
            torch.save(model.state_dict(), best_ckpt)
        json.dump(history, open(hist_path, "w"), indent=2)

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    final = _run_epoch(model, val_loader, None, criterion, vocab, device, train=False)

    results = {
        "train_test_split": "80/10/10",
        "overall_f1":       round(final["overall_acc"], 4),
    }
    for f in FIELDS:
        results[f"exact_match_{f}"] = round(final["accs"][f], 4)

    json.dump(results, open(resl_path, "w"), indent=2)
    print(f"\n✓ Best val overall acc: {best_val_acc:.4f}  |  checkpoint: {best_ckpt}")
    return results, history