# Feature Store Metadata

## Purpose

The feature store provides a simple, versioned registry of recommendation features derived from clickstream and product catalog data. It supports both training-time retrieval and inference-time lookup.

## Storage

- **Warehouse database**: `warehouse/recommendation_features.db`
- **Feature registry**: `feature_store/feature_registry.json`
- **Configuration**: `feature_store/feature_store_config.yaml`

## Feature Views

### 1. user_features
User-level aggregated features derived from clickstream behavior.

Primary key: `visitorid`

Examples:
- total_interactions
- unique_items
- activity_frequency_per_day
- transaction_rate
- recency_days
- avg_rating_per_user

### 2. item_features
Item-level aggregated features derived from clickstream and product metadata.

Primary key: `itemid`

Examples:
- total_interactions
- unique_users
- popularity_score
- avg_price_per_item
- avg_rating_per_item

### 3. interaction_features
Row-level enriched interaction data for model training.

Primary key: `interaction_id`

Examples:
- interaction_weight
- visitorid_encoded
- itemid_encoded
- category_encoded

### 4. item_cooccurrence_features
Item-to-item similarity features computed from purchase baskets.

Composite primary key:
- `itemid_a`
- `itemid_b`

Examples:
- co_occurrence_count
- co_occurrence_score

## Versioning Strategy

Each feature view and feature definition is stored with a version number.

- Version `1` contains the first approved implementation.
- Future changes can be introduced as version `2`, `3`, and so on.
- Training jobs may request features as of a specific timestamp.
- Inference jobs retrieve the latest available values for the requested keys.

## Source-to-Feature Mapping

| Feature | Source | Transformation |
|---------|--------|----------------|
| total_interactions | clickstream | count rows per user/item |
| unique_items | clickstream | count distinct itemid per user |
| activity_frequency_per_day | clickstream | total_interactions / active_days |
| popularity_score | clickstream | weighted sum of event counts |
| avg_price_per_item | products | mean price by item |
| avg_rating_per_item | products | mean rating by item |
| co_occurrence_score | transaction baskets | normalized item-pair frequency |

## Inference Usage

For inference, the feature store returns the latest available features for the requested user or item keys. These features can be joined to a ranking model input pipeline or used for candidate scoring.

## Training Usage

For training, the feature store can return data filtered by key list and optional `as_of` timestamp to support point-in-time correctness.