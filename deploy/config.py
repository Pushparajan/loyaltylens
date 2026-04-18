"""Cloud deployment configuration — isolated from shared/config.py."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeploySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("deploy/.env.sagemaker", "deploy/.env.vertex", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deploy_target: str = Field(default="local", alias="DEPLOY_TARGET")

    # ── AWS / SageMaker ──────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    sagemaker_role_arn: str = Field(default="", alias="SAGEMAKER_ROLE_ARN")
    sagemaker_instance_type: str = Field(default="ml.t2.medium", alias="SAGEMAKER_INSTANCE_TYPE")
    sagemaker_endpoint_name: str = Field(
        default="loyaltylens-propensity", alias="SAGEMAKER_ENDPOINT_NAME"
    )

    # ── GCP / Vertex AI ──────────────────────────────────────────────────────
    gcp_project: str = Field(default="", alias="GCP_PROJECT")
    gcp_region: str = Field(default="us-central1", alias="GCP_REGION")
    vertex_endpoint_id: str = Field(default="", alias="VERTEX_ENDPOINT_ID")
    vertex_index_id: str = Field(default="", alias="VERTEX_INDEX_ID")

    # ── Storage ──────────────────────────────────────────────────────────────
    s3_bucket: str = Field(default="", alias="S3_BUCKET")
    gcs_bucket: str = Field(default="", alias="GCS_BUCKET")
    save_outputs_to_cloud: bool = Field(default=False, alias="SAVE_OUTPUTS_TO_CLOUD")

    @field_validator("deploy_target")
    @classmethod
    def _valid_target(cls, v: str) -> str:
        allowed = {"local", "sagemaker", "vertex"}
        if v not in allowed:
            raise ValueError(f"DEPLOY_TARGET must be one of {allowed}, got {v!r}")
        return v


def get_deploy_settings() -> DeploySettings:
    return DeploySettings()
