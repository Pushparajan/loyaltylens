"""Centralised environment-variable configuration for all modules."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # External APIs
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    sagemaker_endpoint: str = Field(default="", alias="SAGEMAKER_ENDPOINT")

    # Data stores
    postgres_url: str = Field(
        default="postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens",
        alias="POSTGRES_URL",
    )
    weaviate_url: str = Field(default="http://localhost:8080", alias="WEAVIATE_URL")
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="loyaltylens", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # Local data paths
    raw_events_path: str = Field(default="data/raw/events.parquet", alias="RAW_EVENTS_PATH")
    processed_features_path: str = Field(
        default="data/processed/features.parquet", alias="PROCESSED_FEATURES_PATH"
    )
    duckdb_path: str = Field(default="data/feature_store.duckdb", alias="DUCKDB_PATH")

    # Pipeline tunables
    batch_size: int = Field(default=512, alias="BATCH_SIZE")
    eval_pass_threshold: float = Field(default=0.75, alias="EVAL_PASS_THRESHOLD")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
