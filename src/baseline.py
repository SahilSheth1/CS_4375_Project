import os
import re
import json
import pytesseract
from PIL import Image


def run_ocr(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    custom_config = r"--oem 3 --psm 6"
    raw_text = pytesseract.image_to_string(image, config=custom_config)
    return raw_text



def extract_fields(raw_text: str) -> dict:
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]

    extracted = {"company": "", "date": "", "address": "", "total": ""}

   #Invutively company name will be the first line on the receipt 
    if lines:
        extracted["company"] = lines[0]

    #   Setting all dates to DD/MM/YYYY format for the vocab 
    date_pattern = re.compile(
        r"""
        \b(
            \d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}
            |
            \d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}
            |
            \d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|
                         JUL|AUG|SEP|OCT|NOV|DEC)
                         \s+\d{2,4}
        )\b
        """,
        re.VERBOSE | re.IGNORECASE,
    )
    date_match = date_pattern.search(raw_text)
    if date_match:
        extracted["date"] = date_match.group(0).strip()


    # SInce our dataset is mostly english - just look for common total price keywords
    total_pattern = re.compile(
        r"""
        (?:TOTAL|GRAND\s*TOTAL|AMOUNT\s*DUE|AMOUNT|CASH|BALANCE)
        [^\d]{0,20}
        (RM\s?)?
        (\d{1,6}[.,]\d{2})
        """,
        re.VERBOSE | re.IGNORECASE,
    )
    total_match = total_pattern.search(raw_text)
    if total_match:
        currency = total_match.group(1) or ""
        amount = total_match.group(2)
        extracted["total"] = (currency + amount).strip()

    # In laymens term: looking for lines that contain common address strings 
    address_keywords = re.compile(
        r"\b(JALAN|JLN|LORONG|LRG|STREET|ST|AVENUE|AVE|ROAD|RD|"
        r"TAMAN|BANDAR|NO\.|BLOCK|BLK|FLOOR|LEVEL|MALL|PLAZA|"
        r"\d{5})\b",
        re.IGNORECASE,
    )
    address_lines = []
    for line in lines[1:]:  # skip company (line 0)
        if address_keywords.search(line):
            address_lines.append(line)
        if date_pattern.search(line) or total_pattern.search(line):
            break
    extracted["address"] = ", ".join(address_lines)

    return extracted


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s\.\,\/\-]", "", value)
    return value

def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gt_counts = {}
    for t in gt_tokens:
        gt_counts[t] = gt_counts.get(t, 0) + 1

    overlap = sum(min(gt_counts[t], pred_counts.get(t, 0)) for t in gt_counts)

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Submitting to all the fields for F1 scores with the average in all 4 fields.
def evaluate(predictions: list, ground_truths: list) -> dict:
    fields = ["company", "date", "address", "total"]
    results = {}

    for field in fields:
        exact_matches = []
        f1_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_val = normalize(pred.get(field, ""))
            gt_val = normalize(gt.get(field, ""))

            exact_matches.append(1 if pred_val == gt_val else 0)
            f1_scores.append(token_f1(pred_val, gt_val))

        results[field] = {
            "exact_match": sum(exact_matches) / len(exact_matches),
            "f1": sum(f1_scores) / len(f1_scores),
        }

    # Overall averages across all 4 fields
    results["overall"] = {
        "exact_match": sum(results[f]["exact_match"] for f in fields) / len(fields),
        "f1": sum(results[f]["f1"] for f in fields) / len(fields),
    }

    return results


# MAIN PIPELINE


def run_baseline(data_split: list) -> dict:
    predictions = []
    ground_truths = []

    for i, sample in enumerate(data_split):
        print(
            f"Processing {i+1}/{len(data_split)}: {os.path.basename(sample['image_path'])}"
        )

        raw_text = run_ocr(sample["image_path"])
        pred = extract_fields(raw_text)

        predictions.append(pred)
        ground_truths.append(sample["annotation"])

    return evaluate(predictions, ground_truths)
