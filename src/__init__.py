"""
Project package marker for COMP 545 final project code.

This top-level package exposes common configuration helpers so import
paths stay short inside notebooks or scripts.
"""

from .config.runtime import DEFAULT_ENV, RuntimePaths, resolve_paths

__all__ = [
    "DEFAULT_ENV",
    "RuntimePaths",
    "resolve_paths",
]
