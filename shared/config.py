"""Centralised environment-variable configuration for all modules."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Search for .env in the working directory first, then walk up to the repo root.
    # All modules are run from the repo root so ".env" always resolves correctly.
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── External APIs ────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    sagemaker_endpoint: str = Field(default="", alias="SAGEMAKER_ENDPOINT")

    # ── Infrastructure ports (override via env var to avoid touching URLs) ──
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    weaviate_http_port: int = Field(default=8080, alias="WEAVIATE_HTTP_PORT")
    weaviate_grpc_port: int = Field(default=50051, alias="WEAVIATE_GRPC_PORT")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    mlflow_port: int = Field(default=5000, alias="MLFLOW_PORT")

    # ── Service API ports ────────────────────────────────────────────────────
    port_feature_store: int = Field(default=8001, alias="PORT_FEATURE_STORE")
    port_propensity: int = Field(default=8002, alias="PORT_PROPENSITY")
    port_rag_retrieval: int = Field(default=8003, alias="PORT_RAG_RETRIEVAL")
    port_llm_generator: int = Field(default=8004, alias="PORT_LLM_GENERATOR")
    port_feedback_loop: int = Field(default=8005, alias="PORT_FEEDBACK_LOOP")
    port_metrics: int = Field(default=8006, alias="PORT_METRICS")

    # ── Data store URLs (include port; kept as full URLs for driver compat) ─
    postgres_url: str = Field(
        default="postgresql://loyaltylens:loyaltylens@localhost:5432/loyaltylens",
        alias="POSTGRES_URL",
    )
    weaviate_url: str = Field(default="http://localhost:8080", alias="WEAVIATE_URL")
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="loyaltylens", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # ── Local data paths ─────────────────────────────────────────────────────
    raw_events_path: str = Field(default="data/raw/events.parquet", alias="RAW_EVENTS_PATH")
    processed_features_path: str = Field(
        default="data/processed/features.parquet", alias="PROCESSED_FEATURES_PATH"
    )
    duckdb_path: str = Field(default="data/feature_store.duckdb", alias="DUCKDB_PATH")

    # ── Propensity model serving ─────────────────────────────────────────────
    propensity_model_version: str = Field(default="1", alias="PROPENSITY_MODEL_VERSION")
    propensity_models_dir: str = Field(default="models", alias="PROPENSITY_MODELS_DIR")

    # ── Pipeline tunables ────────────────────────────────────────────────────
    batch_size: int = Field(default=512, alias="BATCH_SIZE")
    eval_pass_threshold: float = Field(default=0.75, alias="EVAL_PASS_THRESHOLD")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
