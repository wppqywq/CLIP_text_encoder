"""
CLI helper to download and process the Visual Genome dataset.

Usage examples:
  python scripts/prepare_visual_genome.py --env local
  python scripts/prepare_visual_genome.py --env colab --no-images --skip-verify
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime import DEFAULT_ENV, resolve_paths
from src.data.visual_genome import (
    VisualGenomeProcessConfig,
    download_visual_genome,
    process_visual_genome,
    verify_visual_genome,
)

# ---------------------------------------------------------------------------
# Default parameters (tweak here instead of hard-coding later)
# ---------------------------------------------------------------------------

DEFAULT_INCLUDE_IMAGES = True
DEFAULT_FORCE_DOWNLOAD = False
DEFAULT_MAX_IMAGES = 5000
DEFAULT_MAX_REGIONS_PER_IMAGE = 6
DEFAULT_MIN_REGION_WORDS = 3
DEFAULT_VALID_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_PROCESS_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Visual Genome data.")
    parser.add_argument("--env", default=DEFAULT_ENV, choices=("local", "colab"), help="Environment preset.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download stage.")
    parser.add_argument("--skip-process", action="store_true", help="Skip dataset processing.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification after processing.")

    parser.add_argument("--include-images", dest="include_images", action="store_true", help="Download image archives.")
    parser.add_argument("--no-images", dest="include_images", action="store_false", help="Skip image archives.")
    parser.set_defaults(include_images=DEFAULT_INCLUDE_IMAGES)

    parser.add_argument("--force-download", action="store_true", help="Redownload assets even if they exist.")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES, help="Maximum images to keep (None for all).")
    parser.add_argument("--max-regions", type=int, default=DEFAULT_MAX_REGIONS_PER_IMAGE, help="Max region phrases per image.")
    parser.add_argument("--min-region-words", type=int, default=DEFAULT_MIN_REGION_WORDS, help="Minimum words per region phrase.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VALID_RATIO, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=DEFAULT_PROCESS_SEED, help="Random seed for shuffling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.env)

    if not args.skip_download:
        download_visual_genome(
            paths,
            include_images=args.include_images,
            force=args.force_download or DEFAULT_FORCE_DOWNLOAD,
        )
    else:
        print("Skipping download stage.")

    if not args.skip_process:
        cfg = VisualGenomeProcessConfig(
            max_images=args.max_images if args.max_images >= 0 else None,
            max_regions_per_image=args.max_regions,
            min_region_words=args.min_region_words,
            validation_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        process_visual_genome(paths, cfg)
    else:
        print("Skipping process stage.")

    if not args.skip_verify:
        ok = verify_visual_genome(paths)
        if not ok:
            raise SystemExit("Verification failed.")
    else:
        print("Skipping verification stage.")


if __name__ == "__main__":
    main()


