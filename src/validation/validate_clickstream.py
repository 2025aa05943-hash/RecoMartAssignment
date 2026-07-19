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

REQUIRED_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
VALID_EVENTS = {"view", "addtocart", "transaction"}


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

    logger = logging.getLogger("validate_clickstream")
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


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    return pd.read_csv(csv_path)


def _blank_mask(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"", "nan", "none", "nat"})


def compute_validation_metrics(df: pd.DataFrame, csv_path: Path) -> ValidationMetrics:
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
    duplicate_rows = int(df.duplicated().sum())

    if "timestamp" in df.columns:
        parsed_timestamp = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        invalid_timestamp_rows = int(parsed_timestamp.isna().sum())
    else:
        invalid_timestamp_rows = int(len(df))

    if "event" in df.columns:
        normalized_event = df["event"].astype(str).str.strip().str.lower()
        invalid_event_rows = int((~normalized_event.isin(VALID_EVENTS)).sum())
    else:
        invalid_event_rows = int(len(df))

    if "visitorid" in df.columns:
        visitorid_blank = _blank_mask(df["visitorid"])
        invalid_visitorid_rows = int(visitorid_blank.sum())
    else:
        invalid_visitorid_rows = int(len(df))

    if "itemid" in df.columns:
        item_numeric = pd.to_numeric(df["itemid"], errors="coerce")
        invalid_itemid_rows = int((item_numeric.isna() | (item_numeric <= 0)).sum())
    else:
        invalid_itemid_rows = int(len(df))

    if "event" in df.columns and "transactionid" in df.columns:
        event_normalized = df["event"].astype(str).str.strip().str.lower()
        transaction_mask = event_normalized.eq("transaction")
        tx_blank = _blank_mask(df["transactionid"])
        missing_transactionid_on_transaction_rows = int((transaction_mask & tx_blank).sum())
    else:
        missing_transactionid_on_transaction_rows = int(len(df))

    rule_violations = {
        "missing_values_total": missing_values_total,
        "duplicate_rows": duplicate_rows,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "invalid_event_rows": invalid_event_rows,
        "invalid_visitorid_rows": invalid_visitorid_rows,
        "invalid_itemid_rows": invalid_itemid_rows,
        "missing_transactionid_on_transaction_rows": missing_transactionid_on_transaction_rows,
    }

    bad_rows = (
        missing_values_total
        + duplicate_rows
        + invalid_timestamp_rows
        + invalid_event_rows
        + invalid_visitorid_rows
        + invalid_itemid_rows
        + missing_transactionid_on_transaction_rows
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
    if invalid_timestamp_rows:
        issues.append(f"Invalid timestamps: {invalid_timestamp_rows}")
    if invalid_event_rows:
        issues.append(f"Invalid event values: {invalid_event_rows}")
    if invalid_visitorid_rows:
        issues.append(f"Invalid visitorid values: {invalid_visitorid_rows}")
    if invalid_itemid_rows:
        issues.append(f"Invalid itemid values: {invalid_itemid_rows}")
    if missing_transactionid_on_transaction_rows:
        issues.append(
            f"Missing transactionid where event=transaction: {missing_transactionid_on_transaction_rows}"
        )

    validation_passed = len(issues) == 0

    return ValidationMetrics(
        dataset_name="clickstream",
        file_path=str(csv_path),
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
    parser = argparse.ArgumentParser(description="Validate clickstream CSV data quality.")
    parser.add_argument("--input-csv", required=True, help="Path to the clickstream CSV file")
    parser.add_argument(
        "--output-json",
        default="validation/reports/clickstream_validation.json",
        help="Path to write the validation summary JSON",
    )
    parser.add_argument(
        "--log-file",
        default="validation/logs/validate_clickstream.log",
        help="Path to the validation log file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_json = Path(args.output_json)
    log_file = Path(args.log_file)

    logger = setup_logger(log_file)
    logger.info("Starting clickstream validation | input=%s", input_csv)

    try:
        df = load_csv(input_csv)
        metrics = compute_validation_metrics(df, input_csv)
        write_json_report(metrics, output_json)

        logger.info("Validation report written to %s", output_json)
        logger.info("Rows=%d | Columns=%d | Passed=%s", metrics.total_rows, metrics.total_columns, metrics.validation_passed)
        return 0 if metrics.validation_passed else 2

    except Exception as exc:
        logger.exception("Validation failed | error=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())