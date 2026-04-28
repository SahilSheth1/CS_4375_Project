import os
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE = "../sroie-receipt-dataset/SROIE2019"
SPLITS = ["train", "test"]
SEED = 42
RESOLUTION = 384


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
    VAL_TRANSFORM = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, dataframe, base_path=BASE, transform=None, is_train=False):
        self.df        = dataframe.reset_index(drop=True)
        self.base_path = base_path
        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = self.TRAIN_TRANSFORM
        else:
            self.transform = self.VAL_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.base_path, row["split"], "img", row["file"])
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)

        annotation = {
            "company" : row["company"] or "",
            "date"    : row["date"]    or "",
            "address" : row["address"] or "",
            "total"   : row["total"]   or "",
        } # - changed this from Nan Floats to strings since model runs on strings 
    #     annotation = {                                                                                              
    #   "company" : "" if pd.isna(row["company"]) else row["company"],
    #   "date"    : "" if pd.isna(row["date"])    else row["date"],
    #   "address" : "" if pd.isna(row["address"]) else row["address"],                                          
    #   "total"   : "" if pd.isna(row["total"])   else row["total"],
    # }
        
        return image, annotation


def collate_fn(batch):
    images      = torch.stack([item[0] for item in batch])
    annotations = [item[1] for item in batch]
    return images, annotations


def get_dataloaders(base_path=BASE, batch_size=16):
    df_train, df_val, df_test = load_sroie_split(base_path)

    train_loader = DataLoader(
        SROIEDataset(df_train, base_path, is_train=True),
        batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        SROIEDataset(df_val, base_path, is_train=False),
        batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SROIEDataset(df_test, base_path, is_train=False),
        batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader