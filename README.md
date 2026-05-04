# Receipt Understanding with Vision Transformers
Sahil Sheth - SRS210011
Manraj Singh - MXS220007
Aditya Desai - ASD210004

## Overview
This project implements a Vision Transformer-based system for extracting structured data from receipts.

Fields extracted:
- Vendor
- Date
- Total
- Address

Includes:
- Confidence calibration (temperature scaling)
- Human-in-the-loop review queue
- FastAPI backend

## Dataset (Public)
SROIE: https://rrc.cvc.uab.es/?ch=13  
CORD: https://github.com/clovaai/cord  
WildReceipt (optional): https://github.com/AlibabaResearch/WildReceipt

Place datasets in project root:
CORD/
sroie-receipt-dataset/
wildreceipt/

## Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Train
Open Notebooks/model.ipynb and run all cells.

Model saves to:
Experiments/checkpoints/exp3/best_model.pt

## Run Backend
cd backend
```uvicorn main:app --reload --port 8000```

## Run Frontend
cd fronend
```npm run dev```

## Results
Vendor ~78%
Date ~78%
Total ~44%
Address ~78%

## Notes
Dataset is not included per project requirements.
