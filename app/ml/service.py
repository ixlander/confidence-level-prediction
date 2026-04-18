from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd

from app.core.config import settings
from app.ml.pipeline import FEATURE_SPEC
from app.ml.training import train_and_save_model


@dataclass
class PredictionResult:
    label: str
    probabilities: dict[str, float]


class ModelService:
    def __init__(self) -> None:
        self._bundle = None
        self._lock = Lock()
        self._feature_columns = FEATURE_SPEC.numeric_features + FEATURE_SPEC.categorical_features

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    def load(self) -> None:
        with self._lock:
            artifact_path = Path(settings.model_artifact_path)
            if not artifact_path.exists():
                if not settings.train_if_missing:
                    raise FileNotFoundError(f"Model artifact not found at {artifact_path}")
                train_and_save_model(settings.data_path, settings.model_artifact_path)

            self._bundle = joblib.load(artifact_path)

    def predict_one(self, payload: dict) -> PredictionResult:
        if self._bundle is None:
            raise RuntimeError("Model not loaded")

        frame = pd.DataFrame([payload], columns=self._feature_columns)
        pipeline = self._bundle["pipeline"]
        pred_label = str(pipeline.predict(frame)[0])

        proba = pipeline.predict_proba(frame)[0]
        classes = [str(c) for c in pipeline.classes_]
        probabilities = {label: float(prob) for label, prob in zip(classes, proba)}

        return PredictionResult(label=pred_label, probabilities=probabilities)

    def predict_batch(self, payloads: list[dict]) -> list[PredictionResult]:
        if self._bundle is None:
            raise RuntimeError("Model not loaded")

        frame = pd.DataFrame(payloads, columns=self._feature_columns)
        pipeline = self._bundle["pipeline"]

        labels = [str(label) for label in pipeline.predict(frame)]
        all_probabilities = pipeline.predict_proba(frame)
        classes = [str(c) for c in pipeline.classes_]

        return [
            PredictionResult(
                label=label,
                probabilities={class_label: float(prob) for class_label, prob in zip(classes, probabilities)},
            )
            for label, probabilities in zip(labels, all_probabilities)
        ]


model_service = ModelService()
