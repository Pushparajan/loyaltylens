"""Deployment stub — logs the deployment action and records it to disk.

In a real production setup this would call SageMaker / Vertex / k8s.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from shared.logger import get_logger

logger = get_logger(__name__)

_DEPLOY_LOG = Path("data/deploy_log.jsonl")


def deploy(model_version: str, env: str = "staging") -> None:
    """Log a deployment event."""
    record = {
        "model_version": model_version,
        "env": env,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    _DEPLOY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _DEPLOY_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info("deploy", **record)
    print(f"Deploying model version {model_version} to {env}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LoyaltyLens deploy stub")
    parser.add_argument("--version", default="latest")
    parser.add_argument("--env", default="staging")
    args = parser.parse_args(argv)
    deploy(model_version=args.version, env=args.env)


if __name__ == "__main__":
    main()
