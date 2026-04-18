"""Thin routing layer over S3 and GCS — callers never need to know which cloud."""

from __future__ import annotations

from shared.logger import get_logger
from deploy.config import get_deploy_settings

logger = get_logger(__name__)


def upload_to_s3(local_path: str, bucket: str, key: str) -> str:
    """Upload *local_path* to s3://*bucket*/*key*. Returns the S3 URI."""
    import boto3  # type: ignore[import-untyped]

    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    uri = f"s3://{bucket}/{key}"
    logger.info("s3_upload_complete", uri=uri)
    return uri


def upload_to_gcs(local_path: str, bucket: str, blob_name: str) -> str:
    """Upload *local_path* to gs://*bucket*/*blob_name*. Returns the GCS URI."""
    from google.cloud import storage  # type: ignore[import-untyped]

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(blob_name)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket}/{blob_name}"
    logger.info("gcs_upload_complete", uri=uri)
    return uri


def upload_artifact(local_path: str, key: str) -> str:
    """Route to S3 or GCS based on DEPLOY_TARGET; raise if target is 'local'."""
    from deploy.config import get_deploy_settings

    settings = get_deploy_settings()
    target = settings.deploy_target

    if target == "sagemaker":
        if not settings.s3_bucket:
            raise ValueError("S3_BUCKET must be set when DEPLOY_TARGET=sagemaker")
        return upload_to_s3(local_path, settings.s3_bucket, key)

    if target == "vertex":
        if not settings.gcs_bucket:
            raise ValueError("GCS_BUCKET must be set when DEPLOY_TARGET=vertex")
        return upload_to_gcs(local_path, settings.gcs_bucket, key)

    raise ValueError(f"upload_artifact: unsupported DEPLOY_TARGET={target!r} (use sagemaker or vertex)")
