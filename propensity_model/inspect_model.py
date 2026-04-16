"""Inspect a saved propensity model checkpoint and its metadata.

Usage:
    python -m propensity_model.inspect_model
    python -m propensity_model.inspect_model --version 2
    python -m propensity_model.inspect_model --version 1 --models-dir models
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

import torch

from propensity_model.model import TabTransformerConfig, TabTransformerNet
from shared.logger import get_logger

logger = get_logger(__name__)


def inspect(version: str = "1", models_dir: str = "models") -> None:
    base = pathlib.Path(models_dir)
    model_path = base / f"propensity_v{version}.pt"
    meta_path = base / f"propensity_v{version}_meta.json"

    # ── Checkpoint ────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return

    ckpt: dict = torch.load(str(model_path), map_location="cpu", weights_only=False)
    cfg: TabTransformerConfig = ckpt["config"]
    state: dict = ckpt["state_dict"]

    print("\n" + "=" * 60)
    print(f"  Checkpoint: {model_path}")
    print("=" * 60)
    print("\n[Config]")
    for k, v in asdict(cfg).items():
        print(f"  {k:<28} {v}")

    print("\n[Architecture]")
    net = TabTransformerNet(cfg)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Total parameters     {total_params:,}")
    print(f"  Trainable parameters {trainable_params:,}")

    print("\n[State dict layers]")
    for name, tensor in state.items():
        print(f"  {name:<50} {list(tensor.shape)}")

    # ── Metadata ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Metadata: {meta_path}")
    print("=" * 60)
    if meta_path.exists():
        meta: dict = json.loads(meta_path.read_text())
        print("\n[Meta]")
        for k, v in meta.items():
            print(f"  {k:<28} {v}")
    else:
        print(f"  [WARNING] Metadata file not found: {meta_path}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a propensity model checkpoint.")
    parser.add_argument("--version", default="1", help="Model version (default: 1)")
    parser.add_argument("--models-dir", default="models", help="Directory containing model files")
    args = parser.parse_args()
    inspect(version=args.version, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
