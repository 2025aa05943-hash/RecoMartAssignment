#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("csv_raw_ingestion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ingest_raw_csv(input_csv: Path, raw_dir: Path, log_file: Path) -> int:
    logger = setup_logger(log_file)
    start = time.time()

    logger.info("Starting raw CSV ingestion | input=%s", input_csv)

    if not input_csv.exists():
        logger.error("Source file not found: %s", input_csv)
        return 1

    try:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = raw_dir / "interactions_csv" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / input_csv.name
        shutil.copy2(input_csv, output_file)

        elapsed = time.time() - start
        logger.info("Raw ingestion successful | output=%s | elapsed_seconds=%.2f", output_file, elapsed)
        return 0

    except Exception as exc:
        elapsed = time.time() - start
        logger.exception("Raw ingestion failed | elapsed_seconds=%.2f | error=%s", elapsed, exc)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw CSV ingestion without cleaning.")
    parser.add_argument("--input-csv", required=True, help="Path to the source CSV file")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data output directory")
    parser.add_argument("--log-file", default="data/logs/csv_ingestion.log", help="Log file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = ingest_raw_csv(Path(args.input_csv), Path(args.raw_dir), Path(args.log_file))
    raise SystemExit(code)


if __name__ == "__main__":
    main()