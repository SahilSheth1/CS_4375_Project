import os
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE = "../sroie-receipt-dataset/SROIE2019"
CACHE_DIR = Path("../Experiments/img_cache")
SPLITS = ["train", "test"]
SEED = 42
RESOLUTION = 224  # FIX (speed): reduced from 384 → 224 (9x fewer attention ops)


def build_dataframe(base_path=BASE):
    records = []
    for split in SPLITS:
        img_dir = os.path.join(base_path, split, "img")
        ent_dir = os.path.join(base_path, split, "entities")

        for img_file in sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg")):
            stem     = os.path.splitext(img_file)[0]
            ent_path = os.path.join(ent_dir, stem + ".txt")
            img      = Image.open(os.path.join(img_dir, img_file))
            w, h     = img.size

            fields = {"company": None, "date": None, "address": None, "total": None}
            if os.path.exists(ent_path):
                with open(ent_path, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        data = json.load(f)
                        for key in fields:
                            fields[key] = data.get(key) or None
                    except json.JSONDecodeError:
                        pass

            records.append({"split": split, "file": img_file,
                             "width": w, "height": h, **fields})

    return pd.DataFrame(records)


def load_sroie_split(base_path=BASE):
    df = build_dataframe(base_path)
    df_train_full = df[df["split"] == "train"].reset_index(drop=True)
    df_test       = df[df["split"] == "test"].reset_index(drop=True)
    df_train, df_val = train_test_split(df_train_full, test_size=0.1, random_state=SEED)
    return df_train, df_val, df_test


class SROIEDataset(Dataset):
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    VAL_TRANSFORM = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, dataframe, base_path=BASE, transform=None, is_train=False,
                 cache_dir=CACHE_DIR):
        self.df        = dataframe.reset_index(drop=True)
        self.base_path = base_path
        self.is_train  = is_train
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = self.TRAIN_TRANSFORM
        else:
            self.transform = self.VAL_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cache_path = None
        if self.cache_dir and not self.is_train:
            cache_path = self.cache_dir / row["file"].replace(".jpg", ".pt")

        if cache_path and cache_path.exists():
            image = torch.load(cache_path, weights_only=True)
        else:
            img_path = os.path.join(self.base_path, row["split"], "img", row["file"])
            image = self.transform(Image.open(img_path).convert("RGB"))
            if cache_path and self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(image, cache_path)

        annotation = {
            "company": row["company"] or "",
            "date":    row["date"]    or "",
            "address": row["address"] or "",
            "total":   row["total"]   or "",
        }
        return image, annotation


def collate_fn(batch):
    images      = torch.stack([item[0] for item in batch])
    annotations = [item[1] for item in batch]
    return images, annotations


def get_dataloaders(base_path=BASE, batch_size=32, num_workers=None, cache_dir=CACHE_DIR):
    # FIX (speed): auto-detect MPS and set safe defaults
    is_mps = torch.backends.mps.is_available()
    if num_workers is None:
        num_workers = 0 if is_mps else 4   # MPS + multiprocessing can deadlock
    pin_memory = not is_mps                # MPS does not support pin_memory

    df_train, df_val, df_test = load_sroie_split(base_path)

    train_loader = DataLoader(
        SROIEDataset(df_train, base_path, is_train=True,  cache_dir=cache_dir),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),  # FIX (speed): keeps workers alive between epochs
    )
    val_loader = DataLoader(
        SROIEDataset(df_val, base_path, is_train=False, cache_dir=cache_dir),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        SROIEDataset(df_test, base_path, is_train=False, cache_dir=cache_dir),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader