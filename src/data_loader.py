import os
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

BASE = "../sroie-receipt-dataset/SROIE2019"
CACHE_DIR = Path("../Experiments/img_cache")
SPLITS = ["train", "test"]
SEED = 42
RESOLUTION = 224


# SROIE


def build_dataframe(base_path=BASE):
    records = []
    for split in SPLITS:
        img_dir = os.path.join(base_path, split, "img")
        ent_dir = os.path.join(base_path, split, "entities")

        for img_file in sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg")):
            stem = os.path.splitext(img_file)[0]
            ent_path = os.path.join(ent_dir, stem + ".txt")
            img = Image.open(os.path.join(img_dir, img_file))
            w, h = img.size

            fields = {"company": None, "date": None, "address": None, "total": None}
            if os.path.exists(ent_path):
                with open(ent_path, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        data = json.load(f)
                        for key in fields:
                            fields[key] = data.get(key) or None
                    except json.JSONDecodeError:
                        pass

            records.append(
                {"split": split, "file": img_file, "width": w, "height": h, **fields}
            )

    return pd.DataFrame(records)


def load_sroie_split(base_path=BASE):
    df = build_dataframe(base_path)
    df_train_full = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    df_train, df_val = train_test_split(df_train_full, test_size=0.1, random_state=SEED)
    return df_train, df_val, df_test


# CORD

CORD_CACHE_DIR = Path("../Experiments/cord_img_cache")


def load_cord_split(cord_cache_dir=CORD_CACHE_DIR):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    cord_cache_dir = Path(cord_cache_dir)
    index_path = cord_cache_dir / "index.json"

    # If already cached to disk, just load the index
    if index_path.exists():
        print("  CORD: loading from disk cache")
        with open(index_path) as f:
            index = json.load(f)
        return (
            pd.DataFrame(index["train"]),
            pd.DataFrame(index["val"]),
            pd.DataFrame(index["test"]),
        )

    print("  CORD: first run — saving images to disk (runs once)...")
    cord_cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("naver-clova-ix/cord-v2")

    def parse_and_save(examples, split_name):
        records = []
        for i, example in enumerate(examples):
            gt = json.loads(example["ground_truth"])
            gt_parse = gt.get("gt_parse", {})

            company = gt_parse.get("store_name", "") or ""
            address = gt_parse.get("store_addr", "") or ""
            total_block = gt_parse.get("total", {})
            total = ""
            if isinstance(total_block, dict):
                total = str(total_block.get("total_price", "") or "")
            elif isinstance(total_block, str):
                total = total_block

            # Save image to disk
            img_path = cord_cache_dir / f"{split_name}_{i:04d}.jpg"
            if not img_path.exists():
                example["image"].convert("RGB").save(img_path, "JPEG")

            records.append(
                {
                    "img_path": str(img_path),
                    "company": company.strip(),
                    "date": "",
                    "address": address.strip(),
                    "total": total.strip(),
                }
            )
        return records

    train_records = parse_and_save(ds["train"], "train")
    val_records = parse_and_save(ds["validation"], "val")
    test_records = parse_and_save(ds["test"], "test")

    # Save index so we never need to reload from HuggingFace again
    with open(index_path, "w") as f:
        json.dump({"train": train_records, "val": val_records, "test": test_records}, f)
    print(
        f"  CORD: saved {len(train_records)} train / {len(val_records)} val / {len(test_records)} test images"
    )

    # Free HuggingFace dataset from memory immediately
    del ds

    return (
        pd.DataFrame(train_records),
        pd.DataFrame(val_records),
        pd.DataFrame(test_records),
    )


class CORDDataset(Dataset):
    def __init__(self, dataframe, is_train=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = TRAIN_TRANSFORM if is_train else VAL_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = self.transform(Image.open(row["img_path"]).convert("RGB"))

        annotation = {
            "company": row["company"] or "",
            "date": row["date"] or "",
            "address": row["address"] or "",
            "total": row["total"] or "",
        }

        return image, annotation


# WildReceipt

_WILDRECEIPT_FIELD_IDS = {
    "company": {1},
    "date": {3},
    "total": {25},
    "address": {5},
}


def load_wildreceipt_split(base_path="../wildreceipt"):
    base_path = Path(base_path)
    train_txt = base_path / "train.txt"
    test_txt = base_path / "test.txt"

    if not train_txt.exists():
        raise FileNotFoundError(
            f"WildReceipt not found at {base_path}.\n"
            "Download: https://download.openmmlab.com/mmocr/data/wildreceipt.tar\n"
            "Extract to ../wildreceipt/"
        )

    def parse_file(txt_path):
        records = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                img_path = base_path / data["file_name"]
                fields = {"company": [], "date": [], "total": [], "address": []}

                for ann in data.get("annotations", []):
                    cat_id = ann.get("category_id", -1)
                    text = ann.get("text", "").strip()
                    for field, ids in _WILDRECEIPT_FIELD_IDS.items():
                        if cat_id in ids and text:
                            fields[field].append(text)

                records.append(
                    {
                        "img_path": str(img_path),
                        "company": " ".join(fields["company"]),
                        "date": " ".join(fields["date"]),
                        "address": " ".join(fields["address"]),
                        "total": " ".join(fields["total"]),
                    }
                )
        return pd.DataFrame(records)

    df_all = parse_file(train_txt)
    df_test = parse_file(test_txt)
    df_train, df_val = train_test_split(df_all, test_size=0.1, random_state=SEED)
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# Shared transforms

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Datasets
class SROIEDataset(Dataset):
    def __init__(
        self,
        dataframe,
        base_path=BASE,
        transform=None,
        is_train=False,
        cache_dir=CACHE_DIR,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.base_path = base_path
        self.is_train = is_train
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform or (TRAIN_TRANSFORM if is_train else VAL_TRANSFORM)

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
            "date": row["date"] or "",
            "address": row["address"] or "",
            "total": row["total"] or "",
        }
        return image, annotation


class WildReceiptDataset(Dataset):
    def __init__(self, dataframe, is_train=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = TRAIN_TRANSFORM if is_train else VAL_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(Image.open(row["img_path"]).convert("RGB"))
        annotation = {
            "company": row["company"] or "",
            "date": row["date"] or "",
            "address": row["address"] or "",
            "total": row["total"] or "",
        }
        return image, annotation


# Collate


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    annotations = [item[1] for item in batch]
    return images, annotations


# Main entry point


def get_dataloaders(
    base_path=BASE,
    batch_size=32,
    num_workers=None,
    cache_dir=CACHE_DIR,
    use_cord=False,
    use_wildreceipt=False,
    wildreceipt_path="../wildreceipt",
):
    is_mps = torch.backends.mps.is_available()
    if num_workers is None:
        num_workers = 0 if is_mps else 4
    pin_memory = not is_mps

    # Always load SROIE
    df_train, df_val, df_test = load_sroie_split(base_path)
    train_datasets = [
        SROIEDataset(df_train, base_path, is_train=True, cache_dir=cache_dir)
    ]
    val_datasets = [
        SROIEDataset(df_val, base_path, is_train=False, cache_dir=cache_dir)
    ]
    test_datasets = [
        SROIEDataset(df_test, base_path, is_train=False, cache_dir=cache_dir)
    ]

    # Optionally add CORD
    if use_cord:
        print("Loading CORD dataset...")
        cord_train, cord_val, cord_test = load_cord_split()
        train_datasets.append(CORDDataset(cord_train, is_train=True))
        val_datasets.append(CORDDataset(cord_val, is_train=False))
        test_datasets.append(CORDDataset(cord_test, is_train=False))
        print(
            f"  CORD: {len(cord_train)} train / {len(cord_val)} val / {len(cord_test)} test"
        )

    # Optionally add WildReceipt
    if use_wildreceipt:
        print("Loading WildReceipt dataset...")
        wr_train, wr_val, wr_test = load_wildreceipt_split(wildreceipt_path)
        train_datasets.append(WildReceiptDataset(wr_train, is_train=True))
        val_datasets.append(WildReceiptDataset(wr_val, is_train=False))
        test_datasets.append(WildReceiptDataset(wr_test, is_train=False))
        print(
            f"  WildReceipt: {len(wr_train)} train / {len(wr_val)} val / {len(wr_test)} test"
        )

    # Combine
    train_set = (
        ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    )
    val_set = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    test_set = (
        ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
    )

    print(f"Total: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader
