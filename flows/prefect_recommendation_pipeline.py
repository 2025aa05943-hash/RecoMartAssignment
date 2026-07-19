#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from prefect import flow, get_run_logger, task
from prefect.tasks import exponential_backoff


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def latest_file(search_root: Path, patterns: Sequence[str]) -> Path:
    if not search_root.exists():
        raise FileNotFoundError(f"Search root does not exist: {search_root}")

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(search_root.rglob(pattern))

    files = [p for p in candidates if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found under {search_root} matching {patterns}")

    return max(files, key=lambda p: p.stat().st_mtime)


@task(retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=5), retry_jitter_factor=0.2)
def run_command(step_name: str, command: Sequence[str], cwd: str | None = None) -> None:
    logger = get_run_logger()
    logger.info("Starting step: %s", step_name)
    logger.info("Command: %s", " ".join(command))

    started = time.time()
    proc = subprocess.run(list(command), cwd=cwd, text=True, capture_output=True)

    if proc.stdout:
        logger.info("[%s stdout]\n%s", step_name, proc.stdout.strip())
    if proc.stderr:
        logger.warning("[%s stderr]\n%s", step_name, proc.stderr.strip())

    if proc.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {proc.returncode}")

    logger.info("Completed step: %s in %.2f sec", step_name, time.time() - started)


@task(retries=0)
def run_validation(step_name: str, command: Sequence[str], cwd: str | None = None) -> None:
    logger = get_run_logger()
    logger.info("Starting validation: %s", step_name)
    logger.info("Command: %s", " ".join(command))

    started = time.time()
    proc = subprocess.run(list(command), cwd=cwd, text=True, capture_output=True)

    if proc.stdout:
        logger.info("[%s stdout]\n%s", step_name, proc.stdout.strip())
    if proc.stderr:
        logger.warning("[%s stderr]\n%s", step_name, proc.stderr.strip())

    logger.info("Completed validation: %s in %.2f sec", step_name, time.time() - started)


@task
def resolve_latest_inputs(raw_root: str) -> dict[str, str]:
    raw_root_path = Path(raw_root)

    events_path = latest_file(raw_root_path / "interactions_csv", ("events.csv", "*.csv"))
    products_path = latest_file(raw_root_path / "products_api", ("products_raw.json", "*.json"))

    return {
        "events_csv": str(events_path),
        "products_json": str(products_path),
    }


@task
def resolve_latest_stored_inputs(lake_root: str) -> dict[str, str]:
    lake_root_path = Path(lake_root)

    clickstream_path = latest_file(
        lake_root_path / "raw" / "source=clickstream" / "type=csv",
        ("events.csv", "*.csv"),
    )
    products_path = latest_file(
        lake_root_path / "raw" / "source=products_api" / "type=json",
        ("products_raw.json", "*.json"),
    )

    return {
        "clickstream_csv": str(clickstream_path),
        "products_json": str(products_path),
    }


@task
def write_manifest(manifest: dict, output_path: str) -> None:
    output = Path(output_path)
    ensure_parent(output)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


@flow(name="recommendation-end-to-end-pipeline")
def recommendation_pipeline(
    raw_input_root: str = "data/raw",
    data_lake_root: str = "data_lake",
    runs_root: str = "pipeline_runs",
) -> dict[str, str]:
    logger = get_run_logger()
    run_id = utc_run_id()
    run_root = Path(runs_root) / run_id

    logs_dir = run_root / "logs"
    validation_dir = run_root / "validation"
    ensure_parent(logs_dir / "placeholder.txt")
    ensure_parent(validation_dir / "placeholder.txt")

    logger.info("Pipeline run_id=%s", run_id)
    logger.info("raw_input_root=%s", raw_input_root)
    logger.info("data_lake_root=%s", data_lake_root)

    # 1) Ingestion
    run_command(
        "ingest_events",
        [
            sys.executable,
            "src/ingestion/ingest_events.py",
            "--input-csv",
            str(Path(raw_input_root) / "events.csv"),
        ],
    )

    run_command(
        "ingest_api",
        [
            sys.executable,
            "src/ingestion/ingest_api.py",
        ],
    )

    latest_raw = resolve_latest_inputs(raw_input_root)
    events_csv = latest_raw["events_csv"]
    products_json = latest_raw["products_json"]

    # 2) Data Storage
    run_command(
        "store_clickstream_raw",
        [
            sys.executable,
            "src/store_data/store_raw_data.py",
            "--input",
            events_csv,
            "--source",
            "clickstream",
            "--type",
            "csv",
        ],
    )

    run_command(
        "store_products_raw",
        [
            sys.executable,
            "src/store_data/store_raw_data.py",
            "--input",
            products_json,
            "--source",
            "products_api",
            "--type",
            "json",
        ],
    )

    latest_stored = resolve_latest_stored_inputs(data_lake_root)
    clickstream_stored = latest_stored["clickstream_csv"]
    products_stored = latest_stored["products_json"]

    # 3) Validation
    clickstream_validation_json = str(validation_dir / "reports/clickstream_validation.json")
    products_validation_json = str(validation_dir / "reports/products_validation.json")

    run_validation(
        "validate_clickstream",
        [
            sys.executable,
            "src/validation/validate_clickstream.py",
            "--input-csv",
            clickstream_stored,
            "--output-json",
            clickstream_validation_json,
            "--log-file",
            str(validation_dir / "validate_clickstream.log"),
        ],
    )

    run_validation(
        "validate_products",
        [
            sys.executable,
            "src/validation/validate_products.py",
            "--input-json",
            products_stored,
            "--output-json",
            products_validation_json,
            "--log-file",
            str(validation_dir / "validate_products.log"),
        ],
    )

    # 4) Preparation
    prepared_dir = Path(data_lake_root) / "processed"
    run_command(
        "prepare_data",
        [
            sys.executable,
            "src/preparation/prepare_data.py",
            "--clickstream-csv",
            clickstream_stored,
            "--products-json",
            products_stored,
            "--output-dir",
            str(prepared_dir),
            "--log-file",
            str(run_root / "logs" / "prepare_data.log"),
        ],
    )

    prepared_csv = str(prepared_dir / "interactions_enriched.csv")

    # 5) Feature Engineering
    run_command(
        "build_features",
        [
            sys.executable,
            "src/feature_engineering/build_features.py",
            "--input-csv",
            prepared_csv,
        ],
    )

    # 6) Feature Store demo
    run_command(
        "sample_feature_retrieval",
        [
            sys.executable,
            "src/feature_store/sample_feature_retrieval.py",
        ],
    )

    # 7) Model Training
    run_command(
        "train_recommender",
        [
            sys.executable,
            "train_recommender_scaled.py",
            "--input-csv",
            prepared_csv,
            "--artifacts-dir",
            str(run_root / "models"),
            "--report-path",
            str(run_root / "reports" / "model_performance_report.json"),
            "--tracking-db",
            str(run_root / "tracking" / "model_runs.db"),
        ],
    )

    manifest = {
        "run_id": run_id,
        "raw_input_root": raw_input_root,
        "data_lake_root": data_lake_root,
        "events_csv": events_csv,
        "products_json": products_json,
        "clickstream_stored": clickstream_stored,
        "products_stored": products_stored,
        "prepared_csv": prepared_csv,
        "clickstream_validation_json": clickstream_validation_json,
        "products_validation_json": products_validation_json,
    }

    manifest_path = run_root / "manifest.json"
    write_manifest(manifest, str(manifest_path))

    logger.info("Pipeline completed successfully")
    logger.info("Manifest written to %s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end recommendation pipeline in Prefect.")
    parser.add_argument("--raw-input-root", default="data/raw", help="Root folder containing ingested raw files")
    parser.add_argument("--data-lake-root", default="data_lake", help="Root folder for the data lake")
    parser.add_argument("--runs-root", default="pipeline_runs", help="Folder for run outputs and manifest")
    args = parser.parse_args()

    recommendation_pipeline(
        raw_input_root=args.raw_input_root,
        data_lake_root=args.data_lake_root,
        runs_root=args.runs_root,
    )


if __name__ == "__main__":
    main()