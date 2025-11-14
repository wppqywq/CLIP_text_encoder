"""
Run-time path resolution utilities for local and Colab executions.

This module centralises path handling so the rest of the codebase can
remain agnostic to where datasets and outputs live. All configuration
constants are declared at the top of the file to avoid hard-coded values
inside functions, following the project style guidelines.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path
from typing import Callable, Dict

# ---------------------------------------------------------------------------
# Configuration constants (edit here if directory conventions change)
# ---------------------------------------------------------------------------

DEFAULT_ENV = "local"

DATA_DIRNAME = "data"
RAW_SUBDIR = "visual_genome_raw"
PROCESSED_SUBDIR = "visual_genome"
IMAGES_SUBDIR = "images"
CACHE_SUBDIR = "cache"
OUTPUT_DIRNAME = "output"

COLAB_REPO_ENV = "VG_COLAB_REPO_ROOT"
COLAB_DATA_ENV = "VG_COLAB_DATA_ROOT"
COLAB_OUTPUT_ENV = "VG_COLAB_OUTPUT_ROOT"

DEFAULT_COLAB_REPO = "/content/comp545_final"
DEFAULT_COLAB_DATA = "/content/drive/MyDrive/comp545_final/data"
DEFAULT_COLAB_OUTPUT = "/content/drive/MyDrive/comp545_final/output"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimePaths:
    """Container for canonical project directories."""

    repo_root: Path
    data_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    images_dir: Path
    output_dir: Path
    cache_dir: Path

    def ensure(self) -> "RuntimePaths":
        """Create all directories if they are missing."""
        for path in (
            self.data_root,
            self.raw_data_dir,
            self.processed_data_dir,
            self.images_dir,
            self.output_dir,
            self.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-serialisable view of the paths."""
        return {key: str(value) for key, value in asdict(self).items()}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_local_paths() -> RuntimePaths:
    repo_root = PROJECT_ROOT
    data_root = repo_root / DATA_DIRNAME
    raw_dir = data_root / RAW_SUBDIR
    processed_dir = data_root / PROCESSED_SUBDIR
    images_dir = processed_dir / IMAGES_SUBDIR
    output_dir = repo_root / OUTPUT_DIRNAME
    cache_dir = output_dir / CACHE_SUBDIR
    return RuntimePaths(
        repo_root=repo_root,
        data_root=data_root,
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        images_dir=images_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )


def _build_colab_paths() -> RuntimePaths:
    repo_root = Path(os.environ.get(COLAB_REPO_ENV, DEFAULT_COLAB_REPO)).expanduser().resolve()
    data_root = Path(os.environ.get(COLAB_DATA_ENV, DEFAULT_COLAB_DATA)).expanduser().resolve()
    raw_dir = data_root / RAW_SUBDIR
    processed_dir = data_root / PROCESSED_SUBDIR
    images_dir = processed_dir / IMAGES_SUBDIR
    output_dir = Path(os.environ.get(COLAB_OUTPUT_ENV, DEFAULT_COLAB_OUTPUT)).expanduser().resolve()
    cache_dir = output_dir / CACHE_SUBDIR
    return RuntimePaths(
        repo_root=repo_root,
        data_root=data_root,
        raw_data_dir=raw_dir,
        processed_data_dir=processed_dir,
        images_dir=images_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )


ENV_BUILDERS: Dict[str, Callable[[], RuntimePaths]] = {
    "local": _build_local_paths,
    "colab": _build_colab_paths,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_paths(env: str = DEFAULT_ENV) -> RuntimePaths:
    """Resolve standard project paths for the requested environment."""
    key = env.strip().lower()
    if key not in ENV_BUILDERS:
        raise ValueError(f"Unsupported environment '{env}'. Expected one of {list(ENV_BUILDERS)}.")
    paths = ENV_BUILDERS[key]()
    return paths.ensure()


__all__ = [
    "DEFAULT_ENV",
    "RuntimePaths",
    "resolve_paths",
]


