#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_WEIGHTS = {"view": 1, "addtocart": 2, "transaction": 3}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("feature_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_blank_to_na(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA, "nat": pd.NA, "NaT": pd.NA})


def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required = {"timestamp", "visitorid", "event", "itemid"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prepared data: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "visitorid", "event", "itemid"])

    df["visitorid"] = df["visitorid"].astype("category")
    df["itemid"] = df["itemid"].astype("category")
    df["event"] = df["event"].astype("category")
    df = df[df["event"].isin(EVENT_WEIGHTS)].copy()

    df["interaction_weight"] = df["event"].map(EVENT_WEIGHTS).astype("int64")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "rating_rate" in df.columns:
        df["rating_rate"] = pd.to_numeric(df["rating_rate"], errors="coerce")
    if "rating_count" in df.columns:
        df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

    if "transactionid" in df.columns:
        df["transactionid"] = normalize_blank_to_na(df["transactionid"])

    return df.reset_index(drop=True)


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    global_max_ts = df["timestamp"].max()

    work = df.copy()
    work["is_view"] = (work["event"] == "view").astype("int64")
    work["is_addtocart"] = (work["event"] == "addtocart").astype("int64")
    work["is_transaction"] = (work["event"] == "transaction").astype("int64")

    agg = work.groupby("visitorid", sort=False).agg(
        total_interactions=("itemid", "size"),
        unique_items=("itemid", "nunique"),
        views=("is_view", "sum"),
        addtocarts=("is_addtocart", "sum"),
        transactions=("is_transaction", "sum"),
        avg_interaction_weight=("interaction_weight", "mean"),
        first_interaction_ts=("timestamp", "min"),
        last_interaction_ts=("timestamp", "max"),
    ).reset_index()

    agg["active_days"] = (
        (agg["last_interaction_ts"] - agg["first_interaction_ts"])
        .dt.total_seconds()
        .fillna(0) / 86400.0
    ).clip(lower=0) + 1.0

    agg["activity_frequency_per_day"] = agg["total_interactions"] / agg["active_days"]
    agg["transaction_rate"] = np.where(
        agg["total_interactions"] > 0,
        agg["transactions"] / agg["total_interactions"],
        0.0,
    )
    agg["recency_days"] = (
        (global_max_ts - agg["last_interaction_ts"]).dt.total_seconds().fillna(0) / 86400.0
    )

    if "rating_rate" in work.columns:
        agg = agg.merge(
            work.groupby("visitorid", sort=False)["rating_rate"].mean().reset_index(name="avg_rating_per_user"),
            on="visitorid",
            how="left",
        )
    else:
        agg["avg_rating_per_user"] = np.nan

    if "price" in work.columns:
        agg = agg.merge(
            work.groupby("visitorid", sort=False)["price"].mean().reset_index(name="avg_price_seen"),
            on="visitorid",
            how="left",
        )
    else:
        agg["avg_price_seen"] = np.nan

    agg["first_interaction_ts"] = agg["first_interaction_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    agg["last_interaction_ts"] = agg["last_interaction_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return agg


def build_item_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("itemid").agg(
        total_interactions=("visitorid", "size"),
        unique_users=("visitorid", "nunique"),
        views=("event", lambda s: int((s == "view").sum())),
        addtocarts=("event", lambda s: int((s == "addtocart").sum())),
        transactions=("event", lambda s: int((s == "transaction").sum())),
        avg_interaction_weight=("interaction_weight", "mean"),
        first_seen_ts=("timestamp", "min"),
        last_seen_ts=("timestamp", "max"),
    ).reset_index()

    agg["popularity_score"] = (
        agg["views"] * 1.0
        + agg["addtocarts"] * 2.0
        + agg["transactions"] * 3.0
    )

    if "price" in df.columns:
        item_price = df.groupby("itemid")["price"].mean().reset_index(name="avg_price_per_item")
        agg = agg.merge(item_price, on="itemid", how="left")
    else:
        agg["avg_price_per_item"] = np.nan

    if "rating_rate" in df.columns:
        item_rating = df.groupby("itemid")["rating_rate"].mean().reset_index(name="avg_rating_per_item")
        agg = agg.merge(item_rating, on="itemid", how="left")
    else:
        agg["avg_rating_per_item"] = np.nan

    if "rating_count" in df.columns:
        item_rating_count = df.groupby("itemid")["rating_count"].mean().reset_index(name="avg_rating_count")
        agg = agg.merge(item_rating_count, on="itemid", how="left")
    else:
        agg["avg_rating_count"] = np.nan

    agg["first_seen_ts"] = agg["first_seen_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    agg["last_seen_ts"] = agg["last_seen_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return agg


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp",
        "visitorid",
        "itemid",
        "event",
        "interaction_weight",
    ]

    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()

    if "price" in df.columns:
        out["price"] = df["price"]
    if "rating_rate" in df.columns:
        out["rating_rate"] = df["rating_rate"]
    if "category" in df.columns:
        out["category"] = df["category"].astype(str)
    if "title" in df.columns:
        out["title"] = df["title"].astype(str)

    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Encoding for modeling
    out["visitorid_encoded"], _ = pd.factorize(out["visitorid"], sort=True)
    out["itemid_encoded"], _ = pd.factorize(out["itemid"], sort=True)

    return out


def build_cooccurrence_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Basket definition:
    1) Use transactionid when available for transaction rows
    2) Fallback to visitorid + date for view/addtocart rows
    """
    work = df.copy()
    work["date"] = work["timestamp"].dt.date.astype(str)

    if "transactionid" in work.columns:
        tx_mask = work["event"].eq("transaction") & work["transactionid"].notna()
    else:
        tx_mask = pd.Series(False, index=work.index)

    purchase_baskets = work.loc[tx_mask, ["transactionid", "itemid"]].copy()
    purchase_baskets = purchase_baskets.rename(columns={"transactionid": "basket_id"})
    purchase_baskets["basket_id"] = purchase_baskets["basket_id"].astype(str)

    fallback = work.loc[~tx_mask, ["visitorid", "date", "itemid"]].copy()
    fallback["basket_id"] = fallback["visitorid"].astype(str) + "_" + fallback["date"].astype(str)
    fallback = fallback[["basket_id", "itemid"]]

    baskets = pd.concat([purchase_baskets[["basket_id", "itemid"]], fallback], ignore_index=True)
    baskets = baskets.dropna(subset=["basket_id", "itemid"]).drop_duplicates()

    pair_counts: dict[tuple[str, str], int] = {}
    item_support: dict[str, int] = {}

    grouped = baskets.groupby("basket_id")["itemid"].apply(lambda s: sorted(set(map(str, s))))

    for items in grouped:
        for item in items:
            item_support[item] = item_support.get(item, 0) + 1
        if len(items) < 2:
            continue
        for a, b in combinations(items, 2):
            key = tuple(sorted((a, b)))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    rows = []
    for (a, b), co_count in pair_counts.items():
        sup_a = item_support.get(a, 1)
        sup_b = item_support.get(b, 1)
        cosine_like = co_count / np.sqrt(sup_a * sup_b) if sup_a > 0 and sup_b > 0 else 0.0

        rows.append(
            {
                "itemid_a": a,
                "itemid_b": b,
                "co_occurrence_count": co_count,
                "support_a": sup_a,
                "support_b": sup_b,
                "co_occurrence_score": round(float(cosine_like), 6),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        logger.info("No co-occurrence pairs found.")
    else:
        result = result.sort_values(["co_occurrence_count", "co_occurrence_score"], ascending=False).reset_index(drop=True)

    return result


def save_to_sqlite(
    db_path: Path,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    cooccurrence_features: pd.DataFrame,
) -> None:
    ensure_dir(db_path.parent)
    with sqlite3.connect(db_path) as conn:
        user_features.to_sql("user_features", conn, if_exists="replace", index=False)
        item_features.to_sql("item_features", conn, if_exists="replace", index=False)
        interaction_features.to_sql("interaction_features", conn, if_exists="replace", index=False)
        cooccurrence_features.to_sql("item_cooccurrence_features", conn, if_exists="replace", index=False)


def save_flat_files(output_dir: Path, user_features: pd.DataFrame, item_features: pd.DataFrame, interaction_features: pd.DataFrame, cooccurrence_features: pd.DataFrame) -> None:
    ensure_dir(output_dir)
    user_features.to_csv(output_dir / "user_features.csv", index=False)
    item_features.to_csv(output_dir / "item_features.csv", index=False)
    interaction_features.to_csv(output_dir / "interaction_features.csv", index=False)
    cooccurrence_features.to_csv(output_dir / "item_cooccurrence_features.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recommendation features and store them in SQLite.")
    parser.add_argument(
        "--input-csv",
        default="data_lake/processed/interactions_enriched.csv",
        help="Prepared interactions CSV",
    )
    parser.add_argument(
        "--db-path",
        default="warehouse/recommendation_features.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--output-dir",
        default="data_lake/processed/features",
        help="Optional flat-file output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()

    input_csv = Path(args.input_csv)
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    logger.info("Loading prepared data from %s", input_csv)
    df = load_data(input_csv)

    logger.info("Building user features")
    user_features = build_user_features(df)

    logger.info("Building item features")
    item_features = build_item_features(df)

    logger.info("Building interaction features")
    interaction_features = build_interaction_features(df)

    logger.info("Building co-occurrence features")
    cooccurrence_features = build_cooccurrence_features(df, logger)

    logger.info("Saving outputs to SQLite database: %s", db_path)
    save_to_sqlite(db_path, user_features, item_features, interaction_features, cooccurrence_features)

    logger.info("Saving flat files to %s", output_dir)
    save_flat_files(output_dir, user_features, item_features, interaction_features, cooccurrence_features)

    logger.info("Feature generation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())