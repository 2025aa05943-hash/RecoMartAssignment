#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


def build_target_path(base_dir: Path, source: str, file_type: str, filename: str) -> Path:
    now = datetime.now(timezone.utc)
    dt = now.strftime("%Y-%m-%d")
    run_id = now.strftime("%Y%m%dT%H%M%SZ")

    return (
        base_dir
        / "raw"
        / f"source={source}"
        / f"type={file_type}"
        / f"dt={dt}"
        / f"run_id={run_id}"
        / filename
    )


def store_file(input_path: Path, base_dir: Path, source: str, file_type: str) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    target_path = build_target_path(base_dir, source, file_type, input_path.name)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, target_path)
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Store raw ingested data in partitioned folders.")
    parser.add_argument("--input", required=True, help="Path to the source file")
    parser.add_argument("--base-dir", default="data_lake", help="Base data lake directory")
    parser.add_argument("--source", required=True, help="Source name, e.g. clickstream or products_api")
    parser.add_argument("--type", required=True, help="File type, e.g. csv or json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stored_path = store_file(
        input_path=Path(args.input),
        base_dir=Path(args.base_dir),
        source=args.source,
        file_type=args.type,
    )
    print(f"Stored at: {stored_path}")


if __name__ == "__main__":
    main()