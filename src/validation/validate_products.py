#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["id", "title", "price", "category"]


@dataclass
class ValidationMetrics:
    dataset_name: str
    file_path: str
    total_rows: int
    total_columns: int
    required_columns: list[str]
    missing_columns: list[str]
    extra_columns: list[str]
    missing_values_total: int
    missing_values_by_column: dict[str, int]
    duplicate_rows: int
    rule_violations: dict[str, int]
    valid_rows_estimate: int
    quality_score: float
    validation_passed: bool
    issues: list[str]
    checked_at_utc: str


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("validate_products")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def normalize_product_record(record: dict[str, Any]) -> dict[str, Any]:
    rec = dict(record)

    # Flatten nested rating object if present
    rating = rec.get("rating")
    if isinstance(rating, dict):
        rec["rating_rate"] = rating.get("rate")
        rec["rating_count"] = rating.get("count")
        rec.pop("rating", None)

    # Flatten nested category object if present
    category = rec.get("category")
    if isinstance(category, dict):
        rec["category_id"] = category.get("id")
        rec["category"] = category.get("name") or category.get("title")

    # Flatten nested image/list fields into strings if needed
    images = rec.get("images")
    if isinstance(images, list):
        rec["images"] = ", ".join(str(x) for x in images)

    return rec

def load_products(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if "products" in payload and isinstance(payload["products"], list):
            records = payload["products"]
        elif "data" in payload and isinstance(payload["data"], list):
            records = payload["data"]
        else:
            raise ValueError("Unsupported JSON format. Expected a list or an object with a 'products' list.")
    else:
        raise ValueError("Unsupported JSON format.")

    return [normalize_product_record(r) for r in records]


def _blank_mask(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"", "nan", "none", "nat"})


def _extract_rating(row: pd.Series) -> Any:
    value = row.get("rating", None)
    if isinstance(value, dict):
        return value.get("rate", None)
    return value


def compute_validation_metrics(records: list[dict[str, Any]], json_path: Path) -> ValidationMetrics:
    df = pd.DataFrame(records)

    present_columns = list(df.columns)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in present_columns]
    extra_columns = [c for c in present_columns if c not in REQUIRED_COLUMNS]

    missing_values_by_column: dict[str, int] = {}
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            missing_values_by_column[col] = int(df[col].isna().sum() + _blank_mask(df[col]).sum() - df[col].isna().sum())
        else:
            missing_values_by_column[col] = int(len(df))

    missing_values_total = int(sum(missing_values_by_column.values()))
    duplicate_subset = [c for c in ["id", "title", "price", "category"] if c in df.columns]
    duplicate_rows = int(df.duplicated(subset=duplicate_subset).sum()) if duplicate_subset else 0

    if "id" in df.columns:
        numeric_id = pd.to_numeric(df["id"], errors="coerce")
        invalid_id_rows = int((numeric_id.isna() | (numeric_id <= 0)).sum())
    else:
        invalid_id_rows = int(len(df))

    if "title" in df.columns:
        invalid_title_rows = int(_blank_mask(df["title"]).sum())
    else:
        invalid_title_rows = int(len(df))

    if "price" in df.columns:
        numeric_price = pd.to_numeric(df["price"], errors="coerce")
        invalid_price_rows = int((numeric_price.isna() | (numeric_price <= 0)).sum())
    else:
        invalid_price_rows = int(len(df))

    if "category" in df.columns:
        invalid_category_rows = int(_blank_mask(df["category"]).sum())
    else:
        invalid_category_rows = int(len(df))

    invalid_rating_rows = 0
    if "rating" in df.columns:
        for _, row in df.iterrows():
            rating = _extract_rating(row)
            if rating is None or (isinstance(rating, float) and pd.isna(rating)):
                continue
            try:
                rating_value = float(rating)
                if rating_value < 1 or rating_value > 5:
                    invalid_rating_rows += 1
            except Exception:
                invalid_rating_rows += 1

    invalid_image_rows = 0
    if "image" in df.columns:
        invalid_image_rows = int(_blank_mask(df["image"]).sum())

    rule_violations = {
        "missing_values_total": missing_values_total,
        "duplicate_rows": duplicate_rows,
        "invalid_id_rows": invalid_id_rows,
        "invalid_title_rows": invalid_title_rows,
        "invalid_price_rows": invalid_price_rows,
        "invalid_category_rows": invalid_category_rows,
        "invalid_rating_rows": invalid_rating_rows,
        "invalid_image_rows": invalid_image_rows,
    }

    bad_rows = (
        missing_values_total
        + duplicate_rows
        + invalid_id_rows
        + invalid_title_rows
        + invalid_price_rows
        + invalid_category_rows
        + invalid_rating_rows
        + invalid_image_rows
    )
    valid_rows_estimate = max(0, int(len(df)) - min(int(len(df)), bad_rows))
    quality_score = 100.0 if len(df) == 0 else round((valid_rows_estimate / len(df)) * 100, 2)

    issues: list[str] = []
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    if extra_columns:
        issues.append(f"Extra columns present: {extra_columns}")
    if missing_values_total:
        issues.append(f"Missing values found: {missing_values_total}")
    if duplicate_rows:
        issues.append(f"Duplicate rows found: {duplicate_rows}")
    if invalid_id_rows:
        issues.append(f"Invalid id values: {invalid_id_rows}")
    if invalid_title_rows:
        issues.append(f"Invalid title values: {invalid_title_rows}")
    if invalid_price_rows:
        issues.append(f"Invalid price values: {invalid_price_rows}")
    if invalid_category_rows:
        issues.append(f"Invalid category values: {invalid_category_rows}")
    if invalid_rating_rows:
        issues.append(f"Invalid rating values: {invalid_rating_rows}")
    if invalid_image_rows:
        issues.append(f"Missing/blank image values: {invalid_image_rows}")

    validation_passed = len(issues) == 0

    return ValidationMetrics(
        dataset_name="products",
        file_path=str(json_path),
        total_rows=int(len(df)),
        total_columns=int(len(present_columns)),
        required_columns=REQUIRED_COLUMNS,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        missing_values_total=missing_values_total,
        missing_values_by_column=missing_values_by_column,
        duplicate_rows=duplicate_rows,
        rule_violations=rule_violations,
        valid_rows_estimate=valid_rows_estimate,
        quality_score=quality_score,
        validation_passed=validation_passed,
        issues=issues,
        checked_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def write_json_report(metrics: ValidationMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate product catalog JSON data quality.")
    parser.add_argument("--input-json", required=True, help="Path to the product JSON file")
    parser.add_argument(
        "--output-json",
        default="validation/reports/products_validation.json",
        help="Path to write the validation summary JSON",
    )
    parser.add_argument(
        "--log-file",
        default="validation/logs/validate_products.log",
        help="Path to the validation log file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    log_file = Path(args.log_file)

    logger = setup_logger(log_file)
    logger.info("Starting product validation | input=%s", input_json)

    try:
        records = load_products(input_json)
        metrics = compute_validation_metrics(records, input_json)
        write_json_report(metrics, output_json)

        logger.info("Validation report written to %s", output_json)
        logger.info("Rows=%d | Columns=%d | Passed=%s", metrics.total_rows, metrics.total_columns, metrics.validation_passed)
        return 0 if metrics.validation_passed else 2

    except Exception as exc:
        logger.exception("Validation failed | error=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())