"""Unit tests for deploy/ — all cloud calls are mocked; no live AWS/GCP needed."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


def _stub_google() -> MagicMock:
    """Inject google.cloud.aiplatform into sys.modules so patches resolve."""
    google = sys.modules.setdefault("google", MagicMock())
    cloud = sys.modules.setdefault("google.cloud", MagicMock())
    aip = sys.modules.setdefault("google.cloud.aiplatform", MagicMock())
    google.cloud = cloud
    cloud.aiplatform = aip
    return aip


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_deploy_settings(**kwargs):
    from deploy.config import DeploySettings
    # _env_file=() prevents reading any .env file, giving us clean defaults
    return DeploySettings(_env_file=(), **kwargs)


# ── config tests ──────────────────────────────────────────────────────────────

class TestDeployConfig:
    def test_default_target_is_local(self) -> None:
        s = _make_deploy_settings()
        assert s.deploy_target == "local"

    def test_valid_targets_accepted(self) -> None:
        for target in ("local", "sagemaker", "vertex"):
            s = _make_deploy_settings(DEPLOY_TARGET=target)
            assert s.deploy_target == target

    def test_invalid_target_raises(self) -> None:
        from deploy.config import DeploySettings
        with pytest.raises(ValidationError):
            DeploySettings(_env_file=(), DEPLOY_TARGET="azure")

    def test_sagemaker_defaults(self) -> None:
        s = _make_deploy_settings()
        assert s.sagemaker_instance_type == "ml.t2.medium"
        assert s.sagemaker_endpoint_name == "loyaltylens-propensity"
        assert s.aws_region == "us-east-1"

    def test_vertex_defaults(self) -> None:
        s = _make_deploy_settings()
        assert s.gcp_region == "us-central1"
        assert s.save_outputs_to_cloud is False


# ── cloud_storage tests ───────────────────────────────────────────────────────

class TestCloudStorage:
    def test_upload_artifact_routes_to_s3(self) -> None:
        from deploy import cloud_storage

        mock_settings = MagicMock(deploy_target="sagemaker", s3_bucket="my-bucket")

        with (
            patch("deploy.config.get_deploy_settings", return_value=mock_settings),
            patch.object(cloud_storage, "upload_to_s3", return_value="s3://b/k") as mock_s3,
        ):
            result = cloud_storage.upload_artifact("local/path.joblib", "models/model.joblib")

        mock_s3.assert_called_once_with("local/path.joblib", "my-bucket", "models/model.joblib")
        assert result == "s3://b/k"

    def test_upload_artifact_routes_to_gcs(self) -> None:
        from deploy import cloud_storage

        mock_settings = MagicMock(deploy_target="vertex", gcs_bucket="my-bucket")

        with (
            patch("deploy.config.get_deploy_settings", return_value=mock_settings),
            patch.object(cloud_storage, "upload_to_gcs", return_value="gs://b/k") as mock_gcs,
        ):
            result = cloud_storage.upload_artifact("local/path.joblib", "models/model.joblib")

        mock_gcs.assert_called_once_with("local/path.joblib", "my-bucket", "models/model.joblib")
        assert result == "gs://b/k"

    def test_upload_artifact_raises_for_local(self) -> None:
        from deploy import cloud_storage

        with patch("deploy.config.get_deploy_settings", return_value=MagicMock(deploy_target="local")):
            with pytest.raises(ValueError, match="unsupported DEPLOY_TARGET"):
                cloud_storage.upload_artifact("path.joblib", "key")

    def test_upload_artifact_raises_missing_s3_bucket(self) -> None:
        from deploy import cloud_storage

        with patch(
            "deploy.config.get_deploy_settings",
            return_value=MagicMock(deploy_target="sagemaker", s3_bucket=""),
        ):
            with pytest.raises(ValueError, match="S3_BUCKET"):
                cloud_storage.upload_artifact("path.joblib", "key")

    def test_upload_artifact_raises_missing_gcs_bucket(self) -> None:
        from deploy import cloud_storage

        with patch(
            "deploy.config.get_deploy_settings",
            return_value=MagicMock(deploy_target="vertex", gcs_bucket=""),
        ):
            with pytest.raises(ValueError, match="GCS_BUCKET"):
                cloud_storage.upload_artifact("path.joblib", "key")


# ── SageMaker tests ───────────────────────────────────────────────────────────

class TestSageMakerDeployer:
    def test_invoke_endpoint_parses_response(self) -> None:
        from deploy.sagemaker_deploy import SageMakerDeployer

        fake_body = MagicMock()
        fake_body.read.return_value = json.dumps(
            {"propensity_score": 0.87, "model_version": "xgb-v1"}
        ).encode()

        mock_client = MagicMock()
        mock_client.invoke_endpoint.return_value = {"Body": fake_body}

        with patch("deploy.sagemaker_deploy.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            deployer = SageMakerDeployer(region="us-east-1")
            result = deployer.invoke_endpoint(
                "loyaltylens-propensity",
                {"recency_days": 3, "frequency_30d": 5},
            )

        assert result["propensity_score"] == pytest.approx(0.87)
        assert result["model_version"] == "xgb-v1"

    def test_teardown_calls_three_delete_apis(self) -> None:
        from deploy.sagemaker_deploy import SageMakerDeployer

        mock_client = MagicMock()

        with patch("deploy.sagemaker_deploy.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            deployer = SageMakerDeployer()
            deployer.teardown("loyaltylens-propensity")

        mock_client.delete_endpoint.assert_called_once()
        mock_client.delete_endpoint_config.assert_called_once()
        mock_client.delete_model.assert_called_once()


# ── Vertex AI tests ───────────────────────────────────────────────────────────

class TestVertexModelDeployer:
    def test_predict_parses_vertex_response(self) -> None:
        aip = _stub_google()
        aip.init = MagicMock()

        mock_endpoint_instance = MagicMock()
        mock_endpoint_instance.predict.return_value = MagicMock(predictions=[0.73])
        aip.Endpoint.return_value = mock_endpoint_instance

        from deploy.vertex_deploy import VertexModelDeployer

        deployer = VertexModelDeployer.__new__(VertexModelDeployer)
        deployer._project = "test-project"
        deployer._region = "us-central1"

        result = deployer.predict("projects/123/endpoints/456", {"recency_days": 2})

        assert result["propensity_score"] == pytest.approx(0.73)

    def test_predict_handles_empty_predictions(self) -> None:
        aip = _stub_google()
        aip.init = MagicMock()

        mock_endpoint_instance = MagicMock()
        mock_endpoint_instance.predict.return_value = MagicMock(predictions=[])
        aip.Endpoint.return_value = mock_endpoint_instance

        from deploy.vertex_deploy import VertexModelDeployer

        deployer = VertexModelDeployer.__new__(VertexModelDeployer)
        deployer._project = "test-project"
        deployer._region = "us-central1"

        result = deployer.predict("projects/123/endpoints/456", {})

        assert result["propensity_score"] == pytest.approx(0.0)
