#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi
import requests

API_URL = "https://fakestoreapi.com/products"


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("api_raw_ingestion")
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


def save_raw_snapshot(data: list[dict[str, Any]], raw_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = raw_dir / "products_api" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "products_raw.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_file


def get_data_request(url: str, logger: logging.Logger, max_retries: int = 3) -> list[dict[str, Any]]:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching API data | attempt=%d | url=%s", attempt, url)

            response = requests.get(
                url,
                timeout=30,
                verify=False,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                raise ValueError("API response is not a list")

            return data

        except Exception as exc:
            last_error = exc
            logger.exception("API request attempt %d failed", attempt)

            if attempt < max_retries:
                sleep_for = 2 * attempt
                logger.info("Retrying in %d seconds...", sleep_for)
                time.sleep(sleep_for)

    raise RuntimeError(f"Failed to fetch API data after {max_retries} attempts") from last_error


def ingest_api(raw_dir: Path, log_file: Path, api_url: str) -> int:
    logger = setup_logger(log_file)
    start_time = time.time()

    logger.info("Starting API ingestion | url=%s", api_url)

    try:
        data = get_data_request(api_url, logger)
        output_file = save_raw_snapshot(data, raw_dir)

        elapsed = time.time() - start_time
        logger.info(
            "API ingestion successful | output=%s | records_written=%d | elapsed_seconds=%.2f",
            output_file,
            len(data),
            elapsed,
        )
        return 0

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.exception("API ingestion failed | elapsed_seconds=%.2f | error=%s", elapsed, exc)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw API ingestion for FakeStore products.")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data output directory")
    parser.add_argument("--log-file", default="data/logs/api_ingestion.log", help="Log file path")
    parser.add_argument("--api-url", default=API_URL, help="API endpoint URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = ingest_api(Path(args.raw_dir), Path(args.log_file), args.api_url)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()