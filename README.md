# Confidence Level Detection from Posture Analysis

Production-ready FastAPI service for predicting confidence levels (`Confident`, `Neutral`, `Low`) from posture and skeletal keypoint features.

The API implementation is based on the notebook workflow in `confidence-level-prediction.ipynb`, using:
- same engineered features
- same categorical preprocessing
- same final model choice (Random Forest)

## Project structure

```text
.
├── app/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   │   └── config.py
│   ├── ml/
│   │   ├── pipeline.py
│   │   ├── service.py
│   │   └── training.py
│   ├── artifacts/
│   └── main.py
├── confidence-level-prediction.ipynb
├── confidence_features.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── utils.py
```

## API endpoints

- `GET /health` → service status and model load state
- `POST /predict` → single-item prediction
- `POST /predict/batch` → batch predictions

Interactive docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Local run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model artifact (optional; API can auto-train if missing):

```bash
python -m app.ml.training --data-path confidence_features.csv --artifact-path app/artifacts/model.joblib
```

4. Start API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

## Docker run

### Build and run with Docker

```bash
docker build -t confidence-level-prediction-api .
docker run --rm -p 8000:8000 confidence-level-prediction-api
```

### Run with Docker Compose

```bash
docker compose up --build
```

## Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "eye_shoulder_y_ratio": -0.5035,
    "shoulder_y_diff": 0.0074,
    "wrist_distance_x": 0.5790,
    "wrist_shoulder_ratio": 1.2652,
    "nose_eye_center_offset_x": 0.0051,
    "shoulder_span": 0.4576,
    "hip_shoulder_y_diff": 0.9403,
    "body_lean_x": -0.0131,
    "shoulder_center_x": 0.5287,
    "hip_center_x": 0.5418,
    "spine_angle": 89.2006,
    "eye_distance": 0.1459,
    "head_tilt_angle": -9.4289,
    "eye_distance_ratio": 0.3188,
    "shoulder_slope": 0.0074,
    "head_direction": "Looking Straight",
    "arm_position": "Partially Open",
    "posture": "Upright"
  }'
```

## Configuration (environment variables)

- `APP_NAME` (default: `confidence-level-prediction-api`)
- `APP_ENV` (default: `production`)
- `APP_HOST` (default: `0.0.0.0`)
- `APP_PORT` (default: `8000`)
- `MODEL_ARTIFACT_PATH` (default: `app/artifacts/model.joblib`)
- `DATA_PATH` (default: `confidence_features.csv`)
- `TRAIN_IF_MISSING` (default: `true`)
