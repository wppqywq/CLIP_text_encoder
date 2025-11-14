"""
Visual Genome data management utilities.

Functions in this module take care of downloading, unpacking, processing,
and verifying the Visual Genome dataset so the training code can operate
on a clean JSON split description. All tunable parameters are declared at
the top of the file to avoid magic numbers inside the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import random
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.config.runtime import RuntimePaths

# ---------------------------------------------------------------------------
# Download configuration
# ---------------------------------------------------------------------------

REGION_DESCRIPTIONS_URL = "https://visualgenome.org/static/data/dataset/region_descriptions.json.zip"
IMAGE_DATA_URL = "https://visualgenome.org/static/data/dataset/image_data.json.zip"
IMAGE_ARCHIVE_URLS: Tuple[Tuple[str, str], ...] = (
    ("VG_100K.zip", "https://visualgenome.org/static/data/dataset/VG_100K.zip"),
    ("VG_100K_2.zip", "https://visualgenome.org/static/data/dataset/VG_100K_2.zip"),
)

PROCESSED_JSON_NAME = "visual_genome_splits.json"
REGION_JSON_NAME = "region_descriptions.json"
IMAGE_METADATA_JSON_NAME = "image_data.json"

HTTP_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_RETRIES = 3
DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
RETRY_BACKOFF_SECONDS = 5

# ---------------------------------------------------------------------------
# Processing configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_IMAGES = 5000
DEFAULT_MAX_REGIONS_PER_IMAGE = 6
DEFAULT_MIN_REGION_WORDS = 3
DEFAULT_VALID_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42
DEFAULT_LOWERCASE = False


# ---------------------------------------------------------------------------
# Helper data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualGenomeProcessConfig:
    max_images: Optional[int] = DEFAULT_MAX_IMAGES
    max_regions_per_image: int = DEFAULT_MAX_REGIONS_PER_IMAGE
    min_region_words: int = DEFAULT_MIN_REGION_WORDS
    validation_ratio: float = DEFAULT_VALID_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    seed: int = DEFAULT_RANDOM_SEED
    lowercase: bool = DEFAULT_LOWERCASE

    def to_dict(self) -> Dict[str, object]:
        return dict(asdict(self))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, destination: Path, *, timeout: int, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while attempt <= retries:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, destination.open("wb") as handle:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)


def _extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(destination)


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalise_phrase(phrase: str, lowercase: bool) -> str:
    text = str(phrase).strip()
    if lowercase:
        text = text.lower()
    return text


def _deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        key = item.strip()
        if not key:
            continue
        canonical = key.lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        ordered.append(key)
    return ordered


def _candidate_image_paths(image_id: int, images_root: Path) -> List[Path]:
    filename = f"{image_id}.jpg"
    candidates = [
        images_root / filename,
        images_root / "VG_100K" / filename,
        images_root / "VG_100K_2" / filename,
    ]
    # Provide zero-padded fallback (some archives use 6-digit names)
    padded = f"{image_id:06d}.jpg"
    candidates.extend(
        [
            images_root / padded,
            images_root / "VG_100K" / padded,
            images_root / "VG_100K_2" / padded,
        ]
    )
    return candidates


def _resolve_image_path(image_id: int, images_root: Path) -> Optional[Path]:
    for candidate in _candidate_image_paths(image_id, images_root):
        if candidate.is_file():
            return candidate
    return None


def _save_json(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_visual_genome(
    paths: RuntimePaths,
    *,
    include_images: bool = True,
    force: bool = False,
    timeout: int = HTTP_TIMEOUT_SECONDS,
    retries: int = MAX_DOWNLOAD_RETRIES,
) -> None:
    """Download Visual Genome archives into the raw data directory."""
    raw_dir = paths.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    json_archives = [
        (REGION_DESCRIPTIONS_URL, raw_dir / Path(REGION_DESCRIPTIONS_URL).name),
        (IMAGE_DATA_URL, raw_dir / Path(IMAGE_DATA_URL).name),
    ]

    for url, target in json_archives:
        if force or not target.is_file():
            print(f"Downloading {target.name}...")
            _download_file(url, target, timeout=timeout, retries=retries)
        else:
            print(f"Skipping download (exists): {target.name}")
        extracted_target = raw_dir / Path(target.stem)
        if force or not extracted_target.exists():
            print(f"Extracting {target.name}...")
            _extract_zip(target, raw_dir)

    if include_images:
        for archive_name, url in IMAGE_ARCHIVE_URLS:
            target = raw_dir / archive_name
            if force or not target.is_file():
                print(f"Downloading {archive_name} (large file)...")
                _download_file(url, target, timeout=timeout, retries=retries)
            else:
                print(f"Skipping download (exists): {archive_name}")
            extract_dest = paths.images_dir
            marker = extract_dest / archive_name.replace(".zip", "")
            if force or (not marker.exists()):
                print(f"Extracting {archive_name}...")
                _extract_zip(target, extract_dest)
            else:
                print(f"Skipping extract (exists): {archive_name}")


def process_visual_genome(
    paths: RuntimePaths,
    config: Optional[VisualGenomeProcessConfig] = None,
) -> Path:
    """Process Visual Genome region descriptions into retrieval splits."""
    cfg = config or VisualGenomeProcessConfig()
    config_dict = cfg.to_dict()

    region_json_path = paths.raw_data_dir / REGION_JSON_NAME
    if not region_json_path.is_file():
        zipped = paths.raw_data_dir / f"{REGION_JSON_NAME}.zip"
        if zipped.is_file():
            print("Extracting region descriptions...")
            _extract_zip(zipped, paths.raw_data_dir)
        else:
            raise FileNotFoundError(f"Missing {REGION_JSON_NAME} in {paths.raw_data_dir}")

    image_metadata_path = paths.raw_data_dir / IMAGE_METADATA_JSON_NAME
    if not image_metadata_path.is_file():
        zipped = paths.raw_data_dir / f"{IMAGE_METADATA_JSON_NAME}.zip"
        if zipped.is_file():
            print("Extracting image metadata...")
            _extract_zip(zipped, paths.raw_data_dir)
        else:
            print("Warning: image metadata not found; continuing without it.")
            image_metadata_path = None  # type: ignore[assignment]

    region_payload = _load_json(region_json_path)
    if not isinstance(region_payload, list):
        raise ValueError("Unexpected format for region descriptions JSON.")

    metadata_map: Dict[int, Dict[str, object]] = {}
    if image_metadata_path:
        meta_payload = _load_json(image_metadata_path)
        if isinstance(meta_payload, list):
            for entry in meta_payload:
                if isinstance(entry, dict) and "image_id" in entry:
                    metadata_map[int(entry["image_id"])] = entry

    rng = random.Random(cfg.seed)
    samples: List[Dict[str, object]] = []

    for entry in region_payload:
        if not isinstance(entry, dict):
            continue
        image_id_raw = entry.get("image_id", entry.get("id"))
        if image_id_raw is None:
            continue
        try:
            image_id = int(image_id_raw)
        except (TypeError, ValueError):
            continue
        if image_id < 0:
            continue
        regions = entry.get("regions", [])
        if not isinstance(regions, list):
            continue
        phrases = []
        for region in regions:
            if not isinstance(region, dict):
                continue
            phrase = _normalise_phrase(region.get("phrase", ""), cfg.lowercase)
            if len(phrase.split()) < cfg.min_region_words:
                continue
            phrases.append(phrase)
        deduped = _deduplicate_preserve_order(phrases)
        if not deduped:
            continue
        if cfg.max_regions_per_image > 0:
            deduped = deduped[: cfg.max_regions_per_image]

        image_path = _resolve_image_path(image_id, paths.images_dir)
        if image_path is None:
            continue

        sample = {
            "image_id": image_id,
            "image_path": str(image_path),
            "captions": deduped,
        }
        if image_id in metadata_map:
            sample["metadata"] = metadata_map[image_id]
        samples.append(sample)

        if cfg.max_images is not None and len(samples) >= cfg.max_images:
            break

    if not samples:
        raise RuntimeError("No samples produced during processing; check dataset availability.")

    rng.shuffle(samples)

    val_count = int(len(samples) * cfg.validation_ratio)
    test_count = int(len(samples) * cfg.test_ratio)
    train_count = max(len(samples) - val_count - test_count, 0)

    train_split = samples[:train_count]
    val_split = samples[train_count:train_count + val_count]
    test_split = samples[train_count + val_count:train_count + val_count + test_count]

    if not train_split:
        raise RuntimeError("Training split is empty; adjust process configuration.")

    payload = {
        "config": config_dict,
        "counts": {
            "total": len(samples),
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split),
        },
        "splits": {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        },
    }

    output_path = paths.processed_data_dir / PROCESSED_JSON_NAME
    print(f"Saving processed dataset to {output_path}...")
    _save_json(payload, output_path)
    return output_path


def verify_visual_genome(paths: RuntimePaths) -> bool:
    """Return True if processed splits and corresponding images are present."""
    processed_path = paths.processed_data_dir / PROCESSED_JSON_NAME
    if not processed_path.is_file():
        print(f"Missing processed dataset: {processed_path}")
        return False

    payload = _load_json(processed_path)
    if not isinstance(payload, dict):
        print("Processed dataset has unexpected structure.")
        return False
    splits = payload.get("splits", {})
    if not isinstance(splits, dict):
        print("Processed dataset missing 'splits' section.")
        return False

    def _check_split(entries: Iterable[object]) -> Optional[str]:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            image_path = Path(entry.get("image_path", ""))
            if image_path.is_file():
                return None
        return "No valid image paths found."

    for name in ("train", "val", "test"):
        entries = splits.get(name)
        if not isinstance(entries, list):
            print(f"Split '{name}' missing or invalid.")
            return False
        issue = _check_split(entries)
        if issue:
            print(f"Split '{name}' failed verification: {issue}")
            return False
    print("Visual Genome verification succeeded.")
    return True


__all__ = [
    "VisualGenomeProcessConfig",
    "download_visual_genome",
    "process_visual_genome",
    "verify_visual_genome",
    "PROCESSED_JSON_NAME",
]


