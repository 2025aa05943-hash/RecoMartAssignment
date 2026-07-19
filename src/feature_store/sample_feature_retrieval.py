#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from feature_store import FeatureStore


def main() -> None:
    script_dir = Path(__file__).parent 
    file_path = script_dir / "feature_store_config.yaml"

    store = FeatureStore(file_path)

    print("Available feature views:")
    print(store.list_feature_views())

    print("\nUser feature metadata:")
    print(store.feature_metadata("user_features").to_string(index=False))

    print("\nTraining retrieval example:")
    train_df = store.get_training_features(
        "user_features",
        keys=["user_1", "user_2"],
        as_of="2026-07-18T00:00:00Z",
    )
    print(train_df.head().to_string(index=False))

    print("\nInference retrieval example:")
    infer_df = store.get_inference_features(
        "item_features",
        keys=["1001", "1002"],
    )
    print(infer_df.head().to_string(index=False))


if __name__ == "__main__":
    main()