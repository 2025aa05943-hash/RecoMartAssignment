#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_CLICKSTREAM_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
VALID_EVENTS = {"view", "addtocart", "transaction"}

REQUIRED_PRODUCT_COLUMNS = ["id", "title", "price", "category"]


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("prepare_data")
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_clickstream(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Clickstream file not found: {csv_path}")
    return pd.read_csv(csv_path)


def normalize_blank_to_na(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA, "nat": pd.NA, "NaT": pd.NA})


def clean_clickstream(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CLICKSTREAM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Clickstream missing required columns: {missing}")

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["visitorid"] = normalize_blank_to_na(df["visitorid"])
    df["event"] = df["event"].astype(str).str.strip().str.lower()
    df["itemid"] = normalize_blank_to_na(df["itemid"])
    df["transactionid"] = normalize_blank_to_na(df["transactionid"])

    before = len(df)
    df = df.dropna(subset=["timestamp", "visitorid", "event", "itemid"])
    logger.info("Clickstream rows dropped for missing required values: %d", before - len(df))

    before = len(df)
    df = df[df["event"].isin(VALID_EVENTS)]
    logger.info("Clickstream rows dropped for invalid events: %d", before - len(df))

    before = len(df)
    df = df.drop_duplicates()
    logger.info("Clickstream duplicate rows dropped: %d", before - len(df))

    event_weight = {"view": 1, "addtocart": 2, "transaction": 3}
    df["interaction_weight"] = df["event"].map(event_weight).astype("int64")

    df["timestamp_epoch"] = (
    df["timestamp"].apply(lambda x: int(x.timestamp()))
)
    tmin = df["timestamp_epoch"].min()
    tmax = df["timestamp_epoch"].max()
    if tmax != tmin:
        df["timestamp_norm"] = (df["timestamp_epoch"] - tmin) / (tmax - tmin)
    else:
        df["timestamp_norm"] = 0.0

    df["year"] = df["timestamp"].dt.year.astype("int64")
    df["month"] = df["timestamp"].dt.month.astype("int64")
    df["day"] = df["timestamp"].dt.day.astype("int64")
    df["hour"] = df["timestamp"].dt.hour.astype("int64")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int64")

    # Stable integer encodings for modeling
    df["visitorid_encoded"], visitor_uniques = pd.factorize(df["visitorid"], sort=True)
    df["itemid_encoded"], item_uniques = pd.factorize(df["itemid"], sort=True)

    logger.info("Unique visitors: %d", len(visitor_uniques))
    logger.info("Unique items in clickstream: %d", len(item_uniques))

    return df.reset_index(drop=True)


def load_products(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Products file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("products"), list):
            records = payload["products"]
        elif isinstance(payload.get("data"), list):
            records = payload["data"]
        else:
            raise ValueError("Unsupported products JSON format. Expected a list or an object with a 'products' list.")
    else:
        raise ValueError("Unsupported products JSON format.")

    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        rec = dict(record)

        rating = rec.get("rating")
        if isinstance(rating, dict):
            rec["rating_rate"] = rating.get("rate")
            rec["rating_count"] = rating.get("count")
            rec.pop("rating", None)

        category = rec.get("category")
        if isinstance(category, dict):
            rec["category_id"] = category.get("id")
            rec["category"] = category.get("name") or category.get("title")

        images = rec.get("images")
        if isinstance(images, list):
            rec["images"] = ", ".join(str(x) for x in images)

        normalized.append(rec)

    return normalized


def clean_products(records: list[dict[str, Any]], logger: logging.Logger) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Products dataset is empty.")

    if "id" not in df.columns:
        raise ValueError("Products missing required column: id")

    df["id"] = normalize_blank_to_na(df["id"])
    df["title"] = normalize_blank_to_na(df["title"]) if "title" in df.columns else pd.NA
    df["category"] = normalize_blank_to_na(df["category"]) if "category" in df.columns else pd.NA

    df["price"] = pd.to_numeric(df["price"], errors="coerce") if "price" in df.columns else np.nan
    if "rating_rate" in df.columns:
        df["rating_rate"] = pd.to_numeric(df["rating_rate"], errors="coerce")
    else:
        df["rating_rate"] = np.nan

    if "rating_count" in df.columns:
        df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
    else:
        df["rating_count"] = np.nan

    # Keep only rows with a valid item id
    before = len(df)
    df = df.dropna(subset=["id"])
    logger.info("Product rows dropped for missing id: %d", before - len(df))

    df["id"] = df["id"].astype(str).str.strip()

    # Fill missing descriptive fields
    if "title" not in df.columns:
        df["title"] = "unknown"
    df["title"] = df["title"].fillna("unknown").astype(str).str.strip()
    df.loc[df["title"].eq(""), "title"] = "unknown"

    if "category" not in df.columns:
        df["category"] = "unknown"
    df["category"] = df["category"].fillna("unknown").astype(str).str.strip()
    df.loc[df["category"].eq(""), "category"] = "unknown"

    # Price handling
    price_median = float(df["price"].median()) if df["price"].notna().any() else 0.0
    df["price"] = df["price"].fillna(price_median)
    df = df[df["price"] > 0]
    df = df.drop_duplicates(subset=["id"], keep="first")

    # Normalize numeric fields
    price_min = df["price"].min()
    price_max = df["price"].max()
    if price_max != price_min:
        df["price_scaled"] = (df["price"] - price_min) / (price_max - price_min)
    else:
        df["price_scaled"] = 0.0

    if df["rating_rate"].notna().any():
        rating_min = df["rating_rate"].min()
        rating_max = df["rating_rate"].max()
        if rating_max != rating_min:
            df["rating_rate_scaled"] = (df["rating_rate"] - rating_min) / (rating_max - rating_min)
        else:
            df["rating_rate_scaled"] = 0.0
    else:
        df["rating_rate_scaled"] = np.nan

    # Category encoding
    df["category_encoded"], category_uniques = pd.factorize(df["category"], sort=True)

    logger.info("Unique product categories: %d", len(category_uniques))
    logger.info("Unique products: %d", len(df))

    return df.reset_index(drop=True)


def save_mapping_json(output_path: Path, visitor_df: pd.DataFrame, product_df: pd.DataFrame) -> None:
    visitor_map = (
        visitor_df[["visitorid", "visitorid_encoded"]]
        .drop_duplicates()
        .sort_values("visitorid_encoded")
        .to_dict(orient="records")
    )
    item_map = (
        visitor_df[["itemid", "itemid_encoded"]]
        .drop_duplicates()
        .sort_values("itemid_encoded")
        .to_dict(orient="records")
    )
    category_map = (
        product_df[["category", "category_encoded"]]
        .drop_duplicates()
        .sort_values("category_encoded")
        .to_dict(orient="records")
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "visitor_map": visitor_map,
        "item_map": item_map,
        "category_map": category_map,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def enrich_interactions(clickstream_df: pd.DataFrame, products_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    product_join = products_df.copy()
    product_join["id"] = product_join["id"].astype(str)

    enriched = clickstream_df.merge(
        product_join,
        left_on="itemid",
        right_on="id",
        how="left",
        suffixes=("", "_product"),
    )

    enriched["product_match"] = np.where(enriched["id"].notna(), 1, 0)

    # Fill missing metadata after join
    for col in ["title", "category"]:
        if col in enriched.columns:
            enriched[col] = enriched[col].fillna("unknown").astype(str)

    for col in ["price", "price_scaled", "rating_rate", "rating_rate_scaled", "rating_count"]:
        if col in enriched.columns:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce")

    # Product category encoding for unmatched rows
    if "category_encoded" in enriched.columns:
        enriched["category_encoded"] = enriched["category_encoded"].fillna(-1).astype(int)

    logger.info("Enriched rows: %d", len(enriched))
    logger.info("Matched rows after join: %d", int(enriched["product_match"].sum()))

    return enriched.reset_index(drop=True)


def save_outputs(
    clickstream_cleaned: pd.DataFrame,
    products_cleaned: pd.DataFrame,
    enriched: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    ensure_dir(output_dir)

    clickstream_path = output_dir / "clickstream_cleaned.csv"
    products_path = output_dir / "products_cleaned.csv"
    enriched_path = output_dir / "interactions_enriched.csv"
    mapping_path = output_dir / "encodings.json"

    clickstream_cleaned.to_csv(clickstream_path, index=False)
    products_cleaned.to_csv(products_path, index=False)
    enriched.to_csv(enriched_path, index=False)
    save_mapping_json(mapping_path, clickstream_cleaned, products_cleaned)

    logger.info("Saved clickstream cleaned dataset to %s", clickstream_path)
    logger.info("Saved products cleaned dataset to %s", products_path)
    logger.info("Saved enriched dataset to %s", enriched_path)
    logger.info("Saved encoding maps to %s", mapping_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare clickstream and product data for recommendation modeling.")
    parser.add_argument("--clickstream-csv", required=True, help="Path to raw clickstream CSV")
    parser.add_argument("--products-json", required=True, help="Path to raw products JSON")
    parser.add_argument("--output-dir", default="data_lake/processed", help="Output directory for prepared data")
    parser.add_argument("--log-file", default="data_lake/logs/prepare_data.log", help="Log file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_file = Path(args.log_file)
    output_dir = Path(args.output_dir)

    logger = setup_logger(log_file)
    logger.info("Starting data preparation")

    try:
        clickstream_raw = load_clickstream(Path(args.clickstream_csv))
        products_raw = load_products(Path(args.products_json))

        clickstream_cleaned = clean_clickstream(clickstream_raw, logger)
        products_cleaned = clean_products(products_raw, logger)
        enriched = enrich_interactions(clickstream_cleaned, products_cleaned, logger)

        save_outputs(clickstream_cleaned, products_cleaned, enriched, output_dir, logger)

        logger.info("Data preparation completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Data preparation failed | error=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())