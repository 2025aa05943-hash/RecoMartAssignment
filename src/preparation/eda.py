#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def plot_event_distribution(df: pd.DataFrame, outdir: Path) -> None:
    counts = df["event"].value_counts().reindex(["view", "addtocart", "transaction"]).dropna()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Event Distribution")
    plt.xlabel("Event")
    plt.ylabel("Count")
    savefig(outdir / "event_distribution.png")


def plot_item_popularity(df: pd.DataFrame, outdir: Path) -> None:
    top_items = df["itemid"].value_counts().head(20)
    plt.figure(figsize=(10, 7))
    plt.barh(top_items.index[::-1], top_items.values[::-1])
    plt.title("Top 20 Item Popularity")
    plt.xlabel("Interactions")
    plt.ylabel("Item ID")
    savefig(outdir / "item_popularity.png")


def plot_category_distribution(df: pd.DataFrame, outdir: Path) -> None:
    if "category" not in df.columns:
        return
    cat_counts = df["category"].fillna("unknown").astype(str).value_counts().head(20)

    plt.figure(figsize=(10, 6))
    plt.bar(cat_counts.index, cat_counts.values)
    plt.title("Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    savefig(outdir / "category_distribution.png")


def plot_price_distribution(df: pd.DataFrame, outdir: Path) -> None:
    if "price" not in df.columns:
        return
    prices = pd.to_numeric(df["price"], errors="coerce").dropna()
    if prices.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(prices, bins=30)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    savefig(outdir / "price_distribution.png")


def plot_user_activity(df: pd.DataFrame, outdir: Path) -> None:
    user_counts = df["visitorid"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.hist(user_counts.values, bins=30)
    plt.title("User Activity Distribution")
    plt.xlabel("Interactions per User")
    plt.ylabel("Number of Users")
    savefig(outdir / "user_activity.png")


def build_user_item_matrix(df: pd.DataFrame, max_users: int = 100, max_items: int = 100) -> pd.DataFrame:
    top_users = df["visitorid"].value_counts().head(max_users).index
    top_items = df["itemid"].value_counts().head(max_items).index

    subset = df[df["visitorid"].isin(top_users) & df["itemid"].isin(top_items)].copy()

    if "interaction_weight" not in subset.columns:
        subset["interaction_weight"] = 1

    matrix = subset.pivot_table(
        index="visitorid",
        columns="itemid",
        values="interaction_weight",
        aggfunc="max",
        fill_value=0,
    )

    matrix = matrix.reindex(index=top_users, columns=top_items, fill_value=0)
    return matrix


def plot_interaction_heatmap(matrix: pd.DataFrame, outdir: Path) -> None:
    if matrix.empty:
        return

    plt.figure(figsize=(12, 8))
    plt.imshow(matrix.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Interaction Weight")
    plt.title("User-Item Interaction Heatmap (Top 100 x Top 100)")
    plt.xlabel("Items")
    plt.ylabel("Users")
    savefig(outdir / "interaction_matrix_heatmap.png")


def plot_sparsity_heatmap(matrix: pd.DataFrame, outdir: Path) -> None:
    if matrix.empty:
        return

    sparsity = (matrix.values == 0).astype(int)

    plt.figure(figsize=(12, 8))
    plt.imshow(sparsity, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Sparse Cell (1 = No Interaction)")
    plt.title("Sparsity Pattern Heatmap (Top 100 x Top 100)")
    plt.xlabel("Items")
    plt.ylabel("Users")
    savefig(outdir / "sparsity_heatmap.png")


def print_summary(df: pd.DataFrame, matrix: pd.DataFrame) -> None:
    total_cells = 0
    sparsity_pct = 0.0
    if not matrix.empty:
        total_cells = matrix.shape[0] * matrix.shape[1]
        nonzero = int((matrix.values > 0).sum())
        sparsity_pct = round((1 - (nonzero / total_cells)) * 100, 2) if total_cells else 0.0

    print("=== EDA Summary ===")
    print(f"Rows: {len(df):,}")
    print(f"Users: {df['visitorid'].nunique():,}")
    print(f"Items: {df['itemid'].nunique():,}")
    print(f"Events: {df['event'].value_counts().to_dict()}")
    if "category" in df.columns:
        print(f"Categories: {df['category'].nunique(dropna=True):,}")
    if total_cells:
        print(f"Sparsity on sampled matrix: {sparsity_pct}%")
    print("===================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and generate plots for prepared recommendation data.")
    parser.add_argument(
        "--input-csv",
        default="data_lake/processed/interactions_enriched.csv",
        help="Prepared interactions CSV",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory to store output plots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    plots_dir = Path(args.plots_dir)
    ensure_dir(plots_dir)

    df = load_data(input_csv)

    # Basic cleanup for plotting
    df = df.dropna(subset=["visitorid", "itemid", "event"]).copy()
    df["visitorid"] = df["visitorid"].astype(str)
    df["itemid"] = df["itemid"].astype(str)
    df["event"] = df["event"].astype(str).str.lower().str.strip()
    if "category" in df.columns:
        df["category"] = df["category"].fillna("unknown").astype(str)

    plot_event_distribution(df, plots_dir)
    plot_item_popularity(df, plots_dir)
    plot_category_distribution(df, plots_dir)
    plot_price_distribution(df, plots_dir)
    plot_user_activity(df, plots_dir)

    matrix = build_user_item_matrix(df, max_users=100, max_items=100)
    plot_interaction_heatmap(matrix, plots_dir)
    plot_sparsity_heatmap(matrix, plots_dir)

    print_summary(df, matrix)
    print(f"Plots saved to: {plots_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())