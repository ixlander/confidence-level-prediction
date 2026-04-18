from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "confidence-level-prediction-api"
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_artifact_path: str = "app/artifacts/model.joblib"
    data_path: str = "confidence_features.csv"
    train_if_missing: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
