#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml


@dataclass
class FeatureStoreConfig:
    warehouse_db: str
    registry_file: str
    feature_sets: dict[str, Any]


class FeatureStore:
    def __init__(self, config_path: str | Path):
        script_dir = Path(__file__).parent 
        json_file_path = script_dir / "feature_registry.json"

        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.registry = self._load_registry(Path(json_file_path))

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_registry(self, registry_path: Path) -> dict[str, Any]:
        if not registry_path.exists():
            raise FileNotFoundError(f"Registry file not found: {registry_path}")
        with registry_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def db_path(self) -> Path:
        return Path(self.config["paths"]["warehouse_db"])

    def list_feature_views(self) -> list[str]:
        return list(self.config["feature_sets"].keys())

    def describe_feature_view(self, feature_view: str) -> pd.DataFrame:
        items = self.registry.get(feature_view, [])
        if not items:
            raise KeyError(f"Feature view not found in registry: {feature_view}")
        return pd.DataFrame(items)

    def _query_table(self, table: str, columns: list[str] | None = None, where: str | None = None) -> pd.DataFrame:
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite warehouse DB not found: {self.db_path}")

        cols = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols} FROM {table}"
        if where:
            sql += f" WHERE {where}"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(sql, conn)

    def get_training_features(
        self,
        feature_view: str,
        keys: list[str] | None = None,
        as_of: str | None = None,
    ) -> pd.DataFrame:
        """
        Versioned retrieval for training:
        - keys: optional list of primary keys to filter
        - as_of: optional timestamp filter for temporal consistency
        """
        if feature_view not in self.config["feature_sets"]:
            raise KeyError(f"Unknown feature view: {feature_view}")

        table = self.config["feature_sets"][feature_view]["table"]
        primary_key = self.config["feature_sets"][feature_view]["primary_key"]

        where_clauses = []
        if keys:
            if isinstance(primary_key, list):
                raise ValueError("Composite primary keys need a custom key filter.")
            key_list = ",".join([f"'{k}'" for k in keys])
            where_clauses.append(f"{primary_key} IN ({key_list})")

        if as_of:
            if feature_view in {"user_features", "item_features"}:
                ts_col = "last_interaction_ts" if feature_view == "user_features" else "last_seen_ts"
                where_clauses.append(f"{ts_col} <= '{as_of}'")

        where = " AND ".join(where_clauses) if where_clauses else None
        return self._query_table(table=table, where=where)

    def get_inference_features(
        self,
        feature_view: str,
        keys: list[str],
    ) -> pd.DataFrame:
        """
        Versioned retrieval for inference:
        fetch the latest available feature rows for a set of keys.
        """
        if feature_view not in self.config["feature_sets"]:
            raise KeyError(f"Unknown feature view: {feature_view}")

        table = self.config["feature_sets"][feature_view]["table"]
        primary_key = self.config["feature_sets"][feature_view]["primary_key"]

        if isinstance(primary_key, list):
            raise ValueError("This demo supports single-key inference retrieval only.")

        key_list = ",".join([f"'{k}'" for k in keys])
        where = f"{primary_key} IN ({key_list})"
        return self._query_table(table=table, where=where)

    def feature_metadata(self, feature_view: str) -> pd.DataFrame:
        return self.describe_feature_view(feature_view)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple feature store demo.")
    parser.add_argument("--config", default="feature_store/feature_store_config.yaml")
    parser.add_argument("--view", default="user_features")
    parser.add_argument("--keys", nargs="*", default=[])
    parser.add_argument("--mode", choices=["metadata", "train", "infer"], default="metadata")
    parser.add_argument("--as-of", default=None, help="Training-time timestamp filter, e.g. 2026-07-18T00:00:00Z")
    args = parser.parse_args()

    store = FeatureStore(args.config)

    if args.mode == "metadata":
        print(store.feature_metadata(args.view).to_string(index=False))
        return 0

    if args.mode == "train":
        df = store.get_training_features(args.view, keys=args.keys or None, as_of=args.as_of)
        print(df.head().to_string(index=False))
        print(f"\nRows: {len(df)}")
        return 0

    if args.mode == "infer":
        if not args.keys:
            raise ValueError("Provide one or more keys for inference retrieval.")
        df = store.get_inference_features(args.view, keys=args.keys)
        print(df.head().to_string(index=False))
        print(f"\nRows: {len(df)}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())