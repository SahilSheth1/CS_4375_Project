
from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]

# Best option for TA reproducibility:
# Upload one public zip to Google Drive, GitHub Release, or AWS S3.
# The zip should extract directly into the project root and contain:
#   sroie-receipt-dataset/
#   CORD/
#   wildreceipt/

# Manraj recommend - zip up all datasets so we can easily add more in the future. 
PUBLIC_DATA_BUNDLE_URL = "PASTE_YOUR_PUBLIC_DATA_BUNDLE_ZIP_URL_HERE"

WILDRECEIPT_URL = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already downloaded: {dest}")
        return

    print(f"Downloading:\n  {url}\n→ {dest}")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

def _extract_tar(tar_path: Path, extract_to: Path) -> None:
    print(f"Extracting tar: {tar_path}")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(extract_to)


def _exists_sroie() -> bool:
    return (REPO_ROOT / "sroie-receipt-dataset" / "SROIE2019").exists()


def _exists_wildreceipt() -> bool:
    root = REPO_ROOT / "wildreceipt"
    return (root / "train.txt").exists() and (root / "test.txt").exists()

# Helpful later on in FasstAPI setup ****** 
def setup_public_bundle() -> bool:

    url = PUBLIC_DATA_BUNDLE_URL.strip()

    if not url or "PASTE_YOUR_PUBLIC_DATA_BUNDLE_ZIP_URL_HERE" in url:
        return False

    if _exists_sroie() and _exists_wildreceipt():
        print("Dataset folders already exist. Skipping public bundle download.")
        return True

    bundle_path = REPO_ROOT / "project_data_bundle.zip"
    _download(url, bundle_path)
    _extract_zip(bundle_path, REPO_ROOT)

    print("Public data bundle extracted.")
    return True


def setup_wildreceipt() -> None:
    if _exists_wildreceipt():
        print("WildReceipt already exists.")
        return

    tar_path = REPO_ROOT / "wildreceipt.tar"
    _download(WILDRECEIPT_URL, tar_path)
    _extract_tar(tar_path, REPO_ROOT)

    if not _exists_wildreceipt():
        print(
            "Error: WildReceipt extracted, but the expected train.txt/test.txt files "
            "were not found directly under CS_4375_Project/wildreceipt/. "
            "Check the extracted folder and rename/move it to wildreceipt/ if needed."
        )
    else:
        print("WildReceipt ready.")

# Resort to this if HuggingFace is not working 
def check_sroie() -> None:
    if _exists_sroie():
        print("SROIE found.")
        return

    print(
        "\nSROIE was not found at:\n"
        f"  {REPO_ROOT / 'sroie-receipt-dataset' / 'SROIE2019'}\n\n"
        "To make this fully automatic for the TA, upload a public zip bundle "
        "containing sroie-receipt-dataset/, CORD/, and wildreceipt/, then paste "
        "that URL into PUBLIC_DATA_BUNDLE_URL in src/data_setup.py.\n"
    )


def setup_all() -> None:
    used_bundle = setup_public_bundle()

    if not used_bundle:
        setup_wildreceipt()
        check_sroie()

    print("\nDataset setup check complete.")
    print("Expected project-root folders:")
    print("  sroie-receipt-dataset/SROIE2019/")
    print("  CORD/ or Hugging Face CORD cache")
    print("  wildreceipt/")


if __name__ == "__main__":
    setup_all()
