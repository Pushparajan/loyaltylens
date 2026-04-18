"""Deploy the propensity model to Vertex AI Prediction and offers to Vertex Vector Search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shared.logger import get_logger

logger = get_logger(__name__)


class VertexModelDeployer:
    """Upload and serve Module 2 (propensity) on Vertex AI Prediction."""

    def __init__(self, project: str, region: str = "us-central1") -> None:
        self._project = project
        self._region = region
        self._init_sdk()

    def _init_sdk(self) -> None:
        from google.cloud import aiplatform  # type: ignore[import-untyped]
        aiplatform.init(project=self._project, location=self._region)

    def upload_model(
        self,
        model_artifact_gcs_uri: str,
        display_name: str = "loyaltylens-propensity",
    ) -> str:
        """Upload a model artifact from GCS. Returns the model resource name."""
        from google.cloud import aiplatform  # type: ignore[import-untyped]

        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=model_artifact_gcs_uri,
            serving_container_image_uri=(
                "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
            ),
        )
        logger.info("vertex_model_uploaded", resource_name=model.resource_name)
        return model.resource_name

    def deploy_endpoint(
        self,
        model_resource_name: str,
        endpoint_display_name: str = "loyaltylens-propensity-endpoint",
    ) -> str:
        """Create an endpoint and deploy the model. Returns the endpoint resource name."""
        from google.cloud import aiplatform  # type: ignore[import-untyped]

        model = aiplatform.Model(model_name=model_resource_name)
        endpoint = model.deploy(
            endpoint=aiplatform.Endpoint.create(display_name=endpoint_display_name),
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=2,
        )
        logger.info("vertex_endpoint_deployed", resource_name=endpoint.resource_name)
        return endpoint.resource_name

    def predict(self, endpoint_resource_name: str, feature_dict: dict[str, Any]) -> dict[str, Any]:
        """Call a Vertex AI Prediction endpoint. Returns {'propensity_score': float}."""
        from google.cloud import aiplatform  # type: ignore[import-untyped]

        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_resource_name)
        response = endpoint.predict(instances=[feature_dict])
        predictions = response.predictions
        score = float(predictions[0]) if predictions else 0.0
        result = {"propensity_score": score}
        logger.info("vertex_predict_complete", score=score)
        return result


class VertexVectorSearchDeployer:
    """Index Module 3 offer embeddings in Vertex AI Vector Search (Matching Engine)."""

    def __init__(self, project: str, region: str = "us-central1") -> None:
        self._project = project
        self._region = region
        self._init_sdk()

    def _init_sdk(self) -> None:
        from google.cloud import aiplatform  # type: ignore[import-untyped]
        aiplatform.init(project=self._project, location=self._region)

    def create_index(
        self,
        display_name: str,
        gcs_embeddings_uri: str,
        dimensions: int = 384,
    ) -> str:
        """Create a Tree-AH ANN index from a GCS JSONL embeddings file.

        Each line in the JSONL must be: {"id": "...", "embedding": [...]}
        Returns the index resource name.
        """
        from google.cloud import aiplatform  # type: ignore[import-untyped]

        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=display_name,
            contents_delta_uri=gcs_embeddings_uri,
            dimensions=dimensions,
            approximate_neighbors_count=10,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
        )
        logger.info("vertex_index_created", resource_name=index.resource_name)
        return index.resource_name

    def deploy_index(
        self,
        index_resource_name: str,
        endpoint_display_name: str = "loyaltylens-offer-search",
    ) -> str:
        """Create an index endpoint and deploy the index. Returns the endpoint resource name."""
        from google.cloud import aiplatform  # type: ignore[import-untyped]

        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=endpoint_display_name,
            public_endpoint_enabled=True,
        )
        endpoint.deploy_index(
            index=aiplatform.MatchingEngineIndex(index_name=index_resource_name),
            deployed_index_id="loyaltylens_offers",
            display_name=endpoint_display_name,
        )
        logger.info("vertex_index_deployed", resource_name=endpoint.resource_name)
        return endpoint.resource_name

    def query(
        self,
        endpoint_resource_name: str,
        query_embedding: list[float],
        num_neighbors: int = 5,
    ) -> list[dict[str, Any]]:
        """Find nearest neighbors. Returns list of {'id': str, 'distance': float}."""
        from google.cloud import aiplatform  # type: ignore[import-untyped]

        endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_resource_name
        )
        response = endpoint.find_neighbors(
            deployed_index_id="loyaltylens_offers",
            queries=[query_embedding],
            num_neighbors=num_neighbors,
        )
        neighbors = response[0] if response else []
        results = [{"id": n.id, "distance": n.distance} for n in neighbors]
        logger.info("vertex_query_complete", num_results=len(results))
        return results


def _cli() -> None:
    from deploy.config import get_deploy_settings

    parser = argparse.ArgumentParser(
        description="Vertex AI deploy/predict/query for LoyaltyLens."
    )
    parser.add_argument(
        "--action",
        choices=["deploy-model", "deploy-index", "predict", "query"],
        required=True,
    )
    parser.add_argument("--config", default="deploy/.env.vertex", help="Path to env file")
    parser.add_argument("--gcs-uri", default="", help="GCS artifact URI")
    parser.add_argument("--payload", default="{}", help="JSON feature dict (predict)")
    parser.add_argument("--embedding", default="[]", help="JSON float list (query)")
    args = parser.parse_args()

    settings = get_deploy_settings()
    project = settings.gcp_project
    region = settings.gcp_region

    if args.action == "deploy-model":
        gcs_uri = args.gcs_uri
        if not gcs_uri:
            from deploy.cloud_storage import upload_to_gcs
            gcs_uri = upload_to_gcs("models/model.joblib", settings.gcs_bucket, "models/model.joblib")
        deployer = VertexModelDeployer(project=project, region=region)
        resource = deployer.upload_model(gcs_uri)
        endpoint = deployer.deploy_endpoint(resource)
        print(f"Endpoint: {endpoint}")

    elif args.action == "deploy-index":
        deployer = VertexVectorSearchDeployer(project=project, region=region)
        index = deployer.create_index("loyaltylens-offers", args.gcs_uri)
        endpoint = deployer.deploy_index(index)
        print(f"Index endpoint: {endpoint}")

    elif args.action == "predict":
        feature_dict = json.loads(args.payload)
        deployer = VertexModelDeployer(project=project, region=region)
        result = deployer.predict(settings.vertex_endpoint_id, feature_dict)
        print(json.dumps(result, indent=2))

    elif args.action == "query":
        embedding = json.loads(args.embedding)
        deployer = VertexVectorSearchDeployer(project=project, region=region)
        results = deployer.query(settings.vertex_index_id, embedding)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _cli()
