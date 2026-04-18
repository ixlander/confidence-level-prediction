from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "eye_shoulder_y_ratio",
    "shoulder_y_diff",
    "wrist_distance_x",
    "wrist_shoulder_ratio",
    "nose_eye_center_offset_x",
    "shoulder_span",
    "hip_shoulder_y_diff",
    "body_lean_x",
    "shoulder_center_x",
    "hip_center_x",
    "spine_angle",
    "eye_distance",
    "head_tilt_angle",
    "eye_distance_ratio",
    "shoulder_slope",
]

CATEGORICAL_FEATURES = ["head_direction", "arm_position", "posture"]
TARGET_COLUMN = "confidence_label"

ENGINEERED_FEATURES = [
    "posture_stability",
    "lean_ratio",
    "eye_ratio_adjusted",
    "body_tension",
]
EPSILON = 1e-6


@dataclass(frozen=True)
class FeatureSpec:
    numeric_features: list[str]
    categorical_features: list[str]
    engineered_features: list[str]
    target_column: str


FEATURE_SPEC = FeatureSpec(
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    engineered_features=ENGINEERED_FEATURES,
    target_column=TARGET_COLUMN,
)


class AddFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X["posture_stability"] = X["spine_angle"] + X["head_tilt_angle"]
        X["lean_ratio"] = X["body_lean_x"] / (X["shoulder_span"] + EPSILON)
        X["eye_ratio_adjusted"] = X["eye_distance"] / (X["eye_shoulder_y_ratio"] + EPSILON)
        X["body_tension"] = X["spine_angle"].abs() + X["shoulder_slope"].abs()
        return X


def build_preprocessor() -> Pipeline:
    numeric_features_with_engineering = FEATURE_SPEC.numeric_features + FEATURE_SPEC.engineered_features

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_SPEC.categorical_features),
            ("num", "passthrough", numeric_features_with_engineering),
        ]
    )

    return Pipeline(
        steps=[
            ("add_features", AddFeatures()),
            ("columns", column_transformer),
        ]
    )


def build_model_pipeline() -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])
