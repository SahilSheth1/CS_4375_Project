# Transformer-based Vision Encoder for Automated Receipt Digitization

**CS 4375 ML Project**

A receipt digitization system that reads receipt images and extracts structured fields using a custom-implemented Vision Transformer encoder. Low-confidence predictions are flagged for manual human review.

## Dataset

The project uses the **ICDAR 2019 SROIE** dataset (987 annotated English receipt images).

Dataset hosted on Hugging Face (no download required):
**https://huggingface.co/datasets/SahilSheth1/sroie-receipt-dataset**

> The dataset is **not** included in this repository

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

**Hugging Face (recommended, no account needed):**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SahilSheth1/sroie-receipt-dataset",
    repo_type="dataset",
    local_dir="./sroie-receipt-dataset"
)
```

## Running the Notebook

Once the dataset is in place, open and run `model.ipynb`: