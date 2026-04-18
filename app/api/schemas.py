from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    eye_shoulder_y_ratio: float
    shoulder_y_diff: float
    wrist_distance_x: float
    wrist_shoulder_ratio: float
    nose_eye_center_offset_x: float
    shoulder_span: float
    hip_shoulder_y_diff: float
    body_lean_x: float
    shoulder_center_x: float
    hip_center_x: float
    spine_angle: float
    eye_distance: float
    head_tilt_angle: float
    eye_distance_ratio: float
    shoulder_slope: float
    head_direction: str = Field(min_length=1)
    arm_position: str = Field(min_length=1)
    posture: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class BatchPredictionRequest(BaseModel):
    items: list[PredictionInput]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
