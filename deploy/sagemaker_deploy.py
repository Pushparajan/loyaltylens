"""Deploy and invoke the LoyaltyLens propensity model on AWS SageMaker."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

try:
    import boto3  # type: ignore[import-untyped]
except ImportError:
    boto3 = MagicMock()  # type: ignore[assignment]

from shared.logger import get_logger

logger = get_logger(__name__)

# SageMaker PyTorch serving entry point — written to propensity_model/inference.py
# SageMaker PyTorch container loads model.pt via torch.jit.load (TorchScript).
# input_fn handles channel_preference string → float encoding.
_INFERENCE_TEMPLATE = '''\
"""SageMaker entry point — TorchScript model, custom input/output handlers."""
import json
import os
from pathlib import Path
import torch

_CHANNEL_MAP = {"in-store": 0.0, "mobile": 1.0, "web": 2.0}


def model_fn(model_dir):
    return torch.jit.load(str(Path(model_dir) / "model.pt"), map_location="cpu")


def input_fn(request_body, content_type="application/json"):
    data = json.loads(request_body)
    channel_val = _CHANNEL_MAP.get(str(data.get("channel_preference", "web")).lower(), 2.0)
    return torch.tensor([[
        float(data.get("recency_days", 0)),
        float(data.get("frequency_30d", 0)),
        float(data.get("monetary_90d", 0)),
        float(data.get("offer_redemption_rate", 0)),
        channel_val,
        float(data.get("engagement_score", 0)),
    ]], dtype=torch.float32)


def predict_fn(input_data, model):
    with torch.no_grad():
        score = float(model(input_data).squeeze().item())
    return {"propensity_score": score, "model_version": os.environ.get("MODEL_VERSION", "1")}


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), accept
'''


class SageMakerDeployer:
    def __init__(self, region: str = "us-east-1") -> None:
        self._region = region

    def build_serving_container(self, output_dir: str = "propensity_model") -> str:
        """Write inference.py into *output_dir*. Returns its path."""
        path = Path(output_dir) / "inference.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_INFERENCE_TEMPLATE, encoding="utf-8")
        logger.info("inference_script_written", path=str(path))
        return str(path)

    def export_to_torchscript(self, checkpoint_path: str, output_path: str = "model.pt") -> str:
        """Convert a propensity_v1.pt checkpoint (state_dict + config) to TorchScript.

        SageMaker PyTorch container requires torch.jit-compatible format.
        Uses torch.jit.trace with a 1×6 dummy input since the architecture is fixed.
        """
        import torch  # type: ignore[import-untyped]
        from propensity_model.model import TabTransformerNet

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        net = TabTransformerNet(cfg)
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()

        dummy = torch.zeros(1, cfg.n_features)
        # check_trace=False: TransformerEncoderLayer takes a different fused-op path
        # across invocations (non-deterministic graph, deterministic values).
        scripted = torch.jit.trace(net, dummy, check_trace=False)
        torch.jit.save(scripted, output_path)
        logger.info("torchscript_exported", source=checkpoint_path, dest=output_path)
        return output_path

    def package_model(self, model_path: str, output_dir: str = ".") -> str:
        """Package model.pt + code/inference.py into model.tar.gz for SageMaker.

        SageMaker PyTorch serving container expects:
          model.tar.gz/
            model.pt              ← TorchScript weights loaded by model_fn
            code/inference.py     ← entry point (SAGEMAKER_PROGRAM)
        """
        inference_src = Path("propensity_model") / "inference.py"
        if not inference_src.exists():
            self.build_serving_container()

        archive = Path(output_dir) / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(model_path, arcname="model.pt")
            tar.add(str(inference_src), arcname="code/inference.py")
        logger.info("model_packaged", archive=str(archive), source=model_path)
        return str(archive)

    def ensure_bucket(self, bucket: str) -> None:
        """Create the S3 bucket if it does not already exist."""
        s3 = boto3.client("s3", region_name=self._region)
        try:
            s3.head_bucket(Bucket=bucket)
            logger.info("s3_bucket_exists", bucket=bucket)
        except boto3.exceptions.ClientError if hasattr(boto3, "exceptions") else Exception:
            try:
                if self._region == "us-east-1":
                    s3.create_bucket(Bucket=bucket)
                else:
                    s3.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={"LocationConstraint": self._region},
                    )
                logger.info("s3_bucket_created", bucket=bucket)
            except Exception as exc:
                logger.warning("s3_bucket_create_failed", bucket=bucket, error=str(exc))

    def deploy(
        self,
        model_artifact_s3_uri: str,
        role_arn: str,
        instance_type: str = "ml.t2.medium",
        endpoint_name: str = "loyaltylens-propensity",
    ) -> str:
        """Create a SageMaker PyTorch model and deploy to a real-time endpoint via boto3."""
        sm = boto3.client("sagemaker", region_name=self._region)

        # AWS Deep Learning Container for PyTorch 2.1 CPU inference
        image_uri = (
            f"763104351884.dkr.ecr.{self._region}.amazonaws.com"
            "/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
        )
        model_name = endpoint_name

        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": model_artifact_s3_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "MODEL_VERSION": "1",
                },
            },
            ExecutionRoleArn=role_arn,
        )
        logger.info("sagemaker_model_created", model_name=model_name)

        config_name = f"{endpoint_name}-config"
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName": "primary",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
            }],
        )
        logger.info("sagemaker_endpoint_config_created", config_name=config_name)

        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        logger.info("sagemaker_endpoint_creating", endpoint_name=endpoint_name)

        print("Waiting for endpoint to be InService (~10 min)…")
        waiter = sm.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 30, "MaxAttempts": 40})
        logger.info("sagemaker_endpoint_deployed", endpoint_name=endpoint_name)
        return endpoint_name

    def invoke_endpoint(self, endpoint_name: str, feature_dict: dict[str, Any]) -> dict[str, Any]:
        """Call a deployed SageMaker endpoint and return the parsed prediction."""
        from botocore.config import Config  # type: ignore[import-untyped]
        client = boto3.client(
            "sagemaker-runtime",
            region_name=self._region,
            config=Config(read_timeout=120, connect_timeout=10),
        )
        payload = json.dumps(feature_dict).encode()
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        body = response["Body"].read().decode()
        result: dict[str, Any] = json.loads(body)
        logger.info("sagemaker_invoke_complete", endpoint_name=endpoint_name, score=result.get("propensity_score"))
        return result

    def teardown(self, endpoint_name: str) -> None:
        """Delete the endpoint, its config, and the underlying model."""
        sm = boto3.client("sagemaker", region_name=self._region)
        sm.delete_endpoint(EndpointName=endpoint_name)
        sm.delete_endpoint_config(EndpointConfigName=f"{endpoint_name}-config")
        sm.delete_model(ModelName=endpoint_name)
        logger.info("sagemaker_teardown_complete", endpoint_name=endpoint_name)


def _cli() -> None:
    from deploy.config import get_deploy_settings

    parser = argparse.ArgumentParser(description="SageMaker deploy/invoke/teardown for LoyaltyLens propensity model.")
    parser.add_argument("--action", choices=["deploy", "invoke", "teardown"], required=True)
    parser.add_argument("--config", default="deploy/.env.sagemaker", help="Path to env file (unused — reads from env)")
    parser.add_argument("--model-path", default="models/propensity_v1.pt", help="Local .pt model path (deploy)")
    parser.add_argument("--s3-uri", default="", help="Pre-existing S3 model.tar.gz URI — skips upload (deploy)")
    parser.add_argument("--payload", default="{}", help="JSON feature dict (invoke)")
    args = parser.parse_args()

    settings = get_deploy_settings()
    deployer = SageMakerDeployer(region=settings.aws_region)

    if args.action == "deploy":
        s3_uri = args.s3_uri
        if not s3_uri:
            deployer.ensure_bucket(settings.s3_bucket)
            deployer.build_serving_container()
            with tempfile.TemporaryDirectory() as tmpdir:
                ts_path = deployer.export_to_torchscript(args.model_path, str(Path(tmpdir) / "model.pt"))
                archive = deployer.package_model(ts_path, tmpdir)
                from deploy.cloud_storage import upload_to_s3
                s3_uri = upload_to_s3(archive, settings.s3_bucket, "models/model.tar.gz")
        arn = deployer.deploy(
            s3_uri,
            settings.sagemaker_role_arn,
            settings.sagemaker_instance_type,
            settings.sagemaker_endpoint_name,
        )
        print(f"Deployed: {arn}")

    elif args.action == "invoke":
        feature_dict = json.loads(args.payload)
        result = deployer.invoke_endpoint(settings.sagemaker_endpoint_name, feature_dict)
        print(json.dumps(result, indent=2))

    elif args.action == "teardown":
        deployer.teardown(settings.sagemaker_endpoint_name)
        print("Teardown complete.")


if __name__ == "__main__":
    _cli()
