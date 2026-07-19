# Raw Data Storage Structure

## Overview

The raw data storage layer serves as the landing zone (Bronze layer) of the data pipeline. It stores data exactly as received from the source systems without any transformations or cleansing. This approach ensures data lineage, reproducibility, and auditability.

The project stores raw data in a local filesystem organized using a partitioned folder structure based on:

- Data Source
- File Type
- Ingestion Date
- Ingestion Run ID

---

## Folder Structure

```text
data_lake/
│
├── raw/
│   ├── source=clickstream/
│   │   └── type=csv/
│   │       └── dt=2026-07-18/
│   │           └── run_id=20260718T091500Z/
│   │               └── interactions.csv
│   │
│   └── source=products_api/
│       └── type=json/
│           └── dt=2026-07-18/
│               └── run_id=20260718T091500Z/
│                   └── products.json
│
├── logs/
│   ├── csv_ingestion.log
│   └── api_ingestion.log
│
├── processed/
│
├── configs/
│
└── archive/
```

---

## Directory Description

| Directory | Description |
|------------|-------------|
| raw/ | Stores raw ingested data exactly as received from source systems. |
| logs/ | Stores ingestion logs for monitoring and auditing. |
| processed/ | Stores cleaned and transformed datasets generated in later stages. |
| configs/ | Stores pipeline configuration files. |
| archive/ | Stores historical datasets if data retention policies are implemented. |

---

## Partitioning Strategy

The raw data is partitioned using the following hierarchy.

| Level | Description | Example |
|---------|-------------|----------|
| source | Data source name | clickstream |
| type | File format | csv |
| dt | Ingestion date | 2026-07-18 |
| run_id | Unique ingestion timestamp | 20260718T091500Z |

Example:

```text
raw/
    source=clickstream/
        type=csv/
            dt=2026-07-18/
                run_id=20260718T091500Z/
                    interactions.csv
```

---

## Data Sources

### Source 1 – Clickstream CSV

Description:
User interaction events collected from website clickstream logs.

Format:
CSV

Attributes:

- timestamp
- visitorid
- event
- itemid
- transactionid

---

### Source 2 – Product REST API

Description:
Product catalog metadata collected from a REST API.

Format:
JSON

Typical Attributes:

- id
- title
- description
- category
- price
- image
- rating

---

## Naming Convention

Files retain their original names.

Examples

```
interactions.csv
products.json
```

The directory structure itself uniquely identifies the ingestion batch.

---

## Benefits of the Design

- Supports historical data snapshots.
- Enables easy auditing of ingestion runs.
- Simplifies debugging and replay of data.
- Separates raw data from transformed data.
- Compatible with data lake architectures such as AWS S3, Azure Data Lake, and Hadoop.

---

## Future Extension

The same folder structure can be migrated directly to cloud object storage.

Example:

```
s3://recommendation-data-lake/raw/source=clickstream/type=csv/dt=2026-07-18/run_id=20260718T091500Z/interactions.csv
```

No changes to the ingestion logic would be required except replacing the local filesystem with cloud storage APIs.