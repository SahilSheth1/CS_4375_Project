from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Schema

COLUMNS: list[str] = [
    "experiment_id",
    "description",
    "model_type",
    "patch_size",
    "num_layers",
    "num_heads",
    "learning_rate",
    "train_test_split",
    "exact_match_vendor",
    "exact_match_date",
    "exact_match_address",
    "exact_match_total",
    "f1_vendor",
    "f1_date",
    "f1_address",
    "f1_total",
    "overall_f1",
    "review_rate",
    "notes",
]

# Fields produced by baseline.py that need renaming before writing to the CSV.
# key = name in results dict / JSON, value = CSV column name.
_FIELD_ALIASES: dict[str, str] = {
    "company": "vendor",
}

# Metric columns  values here should be rounded floats or blank.
_METRIC_COLS: set[str] = {
    "exact_match_vendor",
    "exact_match_date",
    "exact_match_address",
    "exact_match_total",
    "f1_vendor",
    "f1_date",
    "f1_address",
    "f1_total",
    "overall_f1",
    "review_rate",
}

# Initialize_log


def initialize_log(path: str | Path = "experiment_log.csv") -> None:
    csv_path = Path(path)
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUMNS).to_csv(csv_path, index=False)
    print(f"Initialized {csv_path}")


# log_experiment


def log_experiment(
    experiment_id: str | int,
    results_dict: dict[str, Any],
    path: str | Path = "experiment_log.csv",
) -> None:
    csv_path = Path(path)
    initialize_log(csv_path)

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise RuntimeError(f"Could not read {csv_path}: {exc}") from exc

    # Ensure all expected columns exist, preserve any extras already in the file.
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    row: dict[str, Any] = {"experiment_id": str(experiment_id)}
    for col in df.columns:
        if col == "experiment_id":
            continue
        val = results_dict.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            row[col] = ""
        elif col in _METRIC_COLS and val != "":
            try:
                row[col] = str(round(float(val), 4))
            except (ValueError, TypeError):
                row[col] = str(val)
        else:
            row[col] = "" if val == "" else str(val)

    new_row_df = pd.DataFrame([row], columns=df.columns)

    mask = df["experiment_id"].astype(str) == str(experiment_id)
    try:
        if mask.any():
            df.loc[mask, :] = new_row_df.values
        else:
            df = pd.concat([df, new_row_df], ignore_index=True)

        df.to_csv(csv_path, index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to {csv_path}. "
            "Close experiment_log.csv in Excel/Sheets and retry."
        )

    print(f" {experiment_id} logged to {csv_path}")


# log_from_json


def log_from_json(
    experiment_id: str | int,
    json_path: str | Path,
    description: str,
    model_type: str,
    extra_params: dict[str, Any] | None = None,
    path: str | Path = "experiment_log.csv",
) -> None:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")

    try:
        with json_path.open() as fh:
            raw: dict = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {json_path}: {exc}") from exc

    results: dict[str, Any] = {
        "description": description,
        "model_type": model_type,
    }

    # Detect and flatten nested format produced by evaluate()
    # Keys are field names; values are dicts with "exact_match" and "f1".
    field_names = {"company", "vendor", "date", "address", "total"}
    is_nested = any(isinstance(raw.get(k), dict) for k in field_names | {"overall"})

    if is_nested:
        for src_field, csv_suffix in [
            ("company", "vendor"),
            ("vendor", "vendor"),
            ("date", "date"),
            ("address", "address"),
            ("total", "total"),
        ]:
            sub = raw.get(src_field)
            if isinstance(sub, dict):
                if "exact_match" in sub:
                    results[f"exact_match_{csv_suffix}"] = sub["exact_match"]
                if "f1" in sub:
                    results[f"f1_{csv_suffix}"] = sub["f1"]

        overall = raw.get("overall", {})
        if isinstance(overall, dict) and "f1" in overall:
            results["overall_f1"] = overall["f1"]
    else:
        # Flat format, copy recognised columns directly, apply alias mapping.
        for key, val in raw.items():
            # Apply alias: "company" → "vendor"
            for src, dst in _FIELD_ALIASES.items():
                key = key.replace(src, dst)
            results[key] = val

    if extra_params:
        results.update(extra_params)

    log_experiment(experiment_id, results, path=path)


# prepare_vit_rows


def prepare_vit_rows(path: str | Path = "experiment_log.csv") -> None:
    vit_configs = [
        {
            "experiment_id": "Exp2",
            "description": "ViT  patch=16, layers=4, heads=4",
            "model_type": "ViT",
            "patch_size": 16,
            "num_layers": 4,
            "num_heads": 4,
            "learning_rate": 1e-4,
            "train_test_split": "80/10/10",
        },
        {
            "experiment_id": "Exp3",
            "description": "ViT  patch=16, layers=6, heads=8",
            "model_type": "ViT",
            "patch_size": 16,
            "num_layers": 6,
            "num_heads": 8,
            "learning_rate": 1e-4,
            "train_test_split": "80/10/10",
        },
    ]

    csv_path = Path(path)
    initialize_log(csv_path)

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise RuntimeError(f"Could not read {csv_path}: {exc}") from exc

    existing_ids = (
        set(df["experiment_id"].astype(str).tolist())
        if "experiment_id" in df.columns
        else set()
    )

    for cfg in vit_configs:
        exp_id = cfg["experiment_id"]
        if exp_id in existing_ids:
            print(f"  {exp_id} already exists  skipping")
            continue
        params = {k: v for k, v in cfg.items() if k != "experiment_id"}
        log_experiment(exp_id, params, path=csv_path)


# CLI / demo

if __name__ == "__main__":
    LOG = Path(__file__).parent.parent / "experiment_log.csv"

    print("── Initializing log ──")
    initialize_log(LOG)

    print("\n── Pre-populating ViT rows ──")
    prepare_vit_rows(LOG)

    print("\n── Current log ──")
    df = pd.read_csv(LOG)
    print(df.to_string(index=False))
