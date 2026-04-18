from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionInput,
    PredictionResponse,
)
from app.core.config import settings
from app.ml.service import model_service

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "model_loaded": model_service.is_loaded,
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionInput) -> PredictionResponse:
    try:
        result = model_service.predict_one(request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictionResponse(prediction=result.label, probabilities=result.probabilities)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        results = model_service.predict_batch([item.model_dump() for item in request.items])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(prediction=result.label, probabilities=result.probabilities)
            for result in results
        ]
    )
