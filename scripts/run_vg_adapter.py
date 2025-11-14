"""
CLI wrapper to run the Visual Genome adapter experiment.

Example:
  python scripts/run_vg_adapter.py --env local --output-name vg_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime import DEFAULT_ENV, resolve_paths
from src.training.vg_adapter import AdapterExperimentConfig, run_visual_genome_adapter

# ---------------------------------------------------------------------------
# Default overrides (edit here when changing the standard experiment)
# ---------------------------------------------------------------------------

BASE_CONFIG = AdapterExperimentConfig()
DEFAULT_DISTILL_WEIGHTS_STR = ",".join(f"{w:.2f}" for w in BASE_CONFIG.distill_weights)


def _parse_weights(value: str) -> Tuple[float, ...]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return BASE_CONFIG.distill_weights
    return tuple(float(token) for token in tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Visual Genome adapter pipeline.")
    parser.add_argument("--env", default=DEFAULT_ENV, choices=("local", "colab"), help="Environment preset.")
    parser.add_argument("--output-name", default=BASE_CONFIG.output_name, help="Subdirectory under output/.")
    parser.add_argument("--model", default=BASE_CONFIG.model_name, help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default=BASE_CONFIG.pretrained, help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", default=BASE_CONFIG.device_preference, help="Device preference (cuda/mps/cpu).")
    parser.add_argument("--seed", type=int, default=BASE_CONFIG.seed, help="Random seed.")
    default_limit = BASE_CONFIG.limit_images if BASE_CONFIG.limit_images is not None else -1
    parser.add_argument("--limit-images", type=int, default=default_limit, help="Max images per split (-1 for all).")

    parser.add_argument("--chunk-words", type=int, default=BASE_CONFIG.chunk_words, help="Chunk size; set 0 to disable.")
    parser.add_argument("--chunk-stride", type=int, default=BASE_CONFIG.chunk_stride, help="Chunk stride.")
    parser.add_argument("--chunk-threshold", type=int, default=BASE_CONFIG.chunk_threshold, help="Minimum words before chunking.")
    parser.add_argument("--text-pooling", default=BASE_CONFIG.text_pooling, help="Chunk pooling mode (mean/max/attn).")

    parser.add_argument("--adapter-steps", type=int, default=BASE_CONFIG.adapter_steps, help="Adapter training steps.")
    parser.add_argument("--adapter-batch", type=int, default=BASE_CONFIG.adapter_batch, help="Adapter batch size.")
    parser.add_argument("--adapter-hidden", type=int, default=BASE_CONFIG.adapter_hidden, help="Adapter bottleneck hidden size.")
    parser.add_argument("--adapter-lr", type=float, default=BASE_CONFIG.adapter_lr, help="Adapter learning rate.")
    parser.add_argument("--logit-lr", type=float, default=BASE_CONFIG.adapter_logit_lr, help="Logit scale learning rate.")
    parser.add_argument("--fine-weight", type=float, default=BASE_CONFIG.fine_grained_weight, help="Fine-grained loss weight.")
    parser.add_argument("--distill", default=DEFAULT_DISTILL_WEIGHTS_STR, help="Comma-separated distillation weights.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.env)

    distill_weights = _parse_weights(args.distill)
    limit_images = None if args.limit_images < 0 else args.limit_images

    config = AdapterExperimentConfig(
        model_name=args.model,
        pretrained=args.pretrained,
        device_preference=args.device,
        seed=args.seed,
        limit_images=limit_images if limit_images is not None else BASE_CONFIG.limit_images,
        chunk_words=args.chunk_words,
        chunk_stride=args.chunk_stride,
        chunk_threshold=args.chunk_threshold,
        text_pooling=args.text_pooling,
        adapter_steps=args.adapter_steps,
        adapter_batch=args.adapter_batch,
        adapter_hidden=args.adapter_hidden,
        adapter_lr=args.adapter_lr,
        adapter_logit_lr=args.logit_lr,
        fine_grained_weight=args.fine_weight,
        distill_weights=distill_weights,
        output_name=args.output_name,
    )

    results = run_visual_genome_adapter(paths, config)
    metrics_path = results.get("metrics_path")
    if metrics_path:
        print(f"Metrics saved to {metrics_path}")

    test_split = results["adapter_metrics"]
    if isinstance(test_split, dict) and distill_weights:
        weight = distill_weights[-1]
        adapter_metrics = test_split.get(weight, {})
        test_metrics = adapter_metrics.get("test")
        if test_metrics:
            print("Test split metrics:")
            for direction, values in test_metrics.items():
                formatted = ", ".join(f"{k}@{v:.2%}" for k, v in values.items())
                print(f"  {direction}: {formatted}")


if __name__ == "__main__":
    main()


