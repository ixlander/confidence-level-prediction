from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from app.ml.pipeline import FEATURE_SPEC, build_model_pipeline


def train_and_save_model(data_path: str, artifact_path: str) -> str:
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")

    df = pd.read_csv(data_file)
    X = df[FEATURE_SPEC.numeric_features + FEATURE_SPEC.categorical_features]
    y = df[FEATURE_SPEC.target_column]

    pipeline = build_model_pipeline()
    pipeline.fit(X, y)

    artifact_file = Path(artifact_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "classes": sorted(df[FEATURE_SPEC.target_column].unique().tolist()),
        "feature_spec": FEATURE_SPEC,
    }
    joblib.dump(bundle, artifact_file)
    return str(artifact_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train confidence-level model artifact")
    parser.add_argument("--data-path", default="confidence_features.csv")
    parser.add_argument("--artifact-path", default="app/artifacts/model.joblib")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = train_and_save_model(data_path=args.data_path, artifact_path=args.artifact_path)
    print(f"Model artifact created at: {artifact}")


if __name__ == "__main__":
    main()
