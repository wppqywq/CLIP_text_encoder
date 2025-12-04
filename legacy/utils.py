from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Union
from typing import List, Dict, Optional, Tuple, cast

import json

import numpy as np
import torch
from PIL import Image

__all__ = [
    "ensure_dir",
    "select_torch_device",
    "set_all_seeds",
    "load_flickr30k_karpathy_json",
    "chunk_caption_words",
    "pool_chunk_embeddings",
    "encode_openclip_embeddings",
    "recall_at_k_text_to_image",
    "recall_at_k_image_to_text",
]

PathLike = Union[str, os.PathLike]


def ensure_dir(path: PathLike) -> str:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def select_torch_device(device_preference: str = "cuda") -> torch.device:
    pref = (device_preference or "").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)



def load_flickr30k_karpathy_json(
    annotations_path: PathLike,
    images_root: PathLike,
    split: str = "test",
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    annotations_file = Path(annotations_path).expanduser().resolve()
    images_dir = Path(images_root).expanduser().resolve()
    if not annotations_file.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images root not found: {images_dir}")

    with annotations_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset: List[Dict[str, object]] = []

    # Karpathy format: {"images": [{"split": str, "filename": str, "sentences": [{"raw": str}, ...]}, ...]}
    if isinstance(data, dict) and "images" in data:
        for item in data["images"]:
            if str(item.get("split", "")).lower() != split.lower():
                continue
            filename = item.get("filename")
            sentences = item.get("sentences", [])
            captions: List[str] = [str(s.get("raw", "")).strip() for s in sentences]
            if not filename or not captions:
                continue
            dataset.append({
                "image_path": str(images_dir / filename),
                "captions": captions,
            })
    # Alternative simple format: {"annotations": [{"image": str, "captions": [str, ...]}]}
    elif isinstance(data, dict) and "annotations" in data:
        for item in data["annotations"]:
            filename = item.get("image")
            captions = item.get("captions", [])
            if not filename or not captions:
                continue
            dataset.append({
                "image_path": str(images_dir / filename),
                "captions": [str(c).strip() for c in captions],
            })
    # Fallback: list of {"image": str, "captions": [...]}
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            filename = item.get("image")
            captions = item.get("captions", [])
            if not filename or not captions:
                continue
            dataset.append({
                "image_path": str(images_dir / filename),
                "captions": [str(c).strip() for c in captions],
            })
    else:
        raise ValueError("Unsupported annotations format")

    if limit is not None:
        dataset = dataset[: int(limit)]

    # Validate image existence lazily (do not filter aggressively to keep behavior simple)
    return dataset


def _normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def chunk_caption_words(caption: str, chunk_words: int, stride_words: Optional[int] = None) -> List[str]:
    """
    Split a caption into word chunks of size chunk_words. Non-overlapping by default.
    """
    if chunk_words is None or chunk_words <= 0:
        return [caption]
    words = str(caption).split()
    if len(words) == 0:
        return []
    stride = chunk_words if not stride_words or stride_words <= 0 else int(stride_words)
    chunks: List[str] = []
    for start in range(0, len(words), stride):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def pool_chunk_embeddings(
    chunk_embeds: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Aggregate chunk embeddings into a single embedding.
    mode in {"mean", "max", "attn"}. For attn, use softmax over similarity to the mean.
    """
    if chunk_embeds.ndim == 1:
        chunk_embeds = chunk_embeds.unsqueeze(0)
    if mode == "max":
        pooled, _ = torch.max(chunk_embeds, dim=0)
    elif mode == "attn":
        with torch.no_grad():
            q = chunk_embeds.mean(dim=0, keepdim=True)  # (1, d)
            logits = (chunk_embeds @ _normalize_features(q).T).squeeze(-1)  # (n)
            weights = torch.softmax(logits, dim=0)  # (n)
        pooled = (weights[:, None] * chunk_embeds).sum(dim=0)
    else:  # mean
        pooled = chunk_embeds.mean(dim=0)
    return _normalize_features(pooled)


def encode_openclip_embeddings(
    dataset: List[Dict[str, object]],
    preprocess,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
    text_chunk_words: Optional[int] = None,
    text_chunk_stride: Optional[int] = None,
    text_pooling: str = "mean",
    progress: bool = False,
    progress_desc: str = "",
    chunk_threshold_tokens: Optional[int] = None,
) -> Dict[str, object]:
    # Prepare image paths and captions
    image_paths: List[str] = [str(item["image_path"]) for item in dataset]
    captions_nested: List[List[str]] = [list(cast(List[str], item["captions"])) for item in dataset]

    # Flatten captions and build mappings
    captions_flat: List[str] = []
    caption_to_image_index: List[int] = []
    image_to_caption_indices: List[List[int]] = []
    running_idx = 0
    for image_idx, caps in enumerate(captions_nested):
        idxs: List[int] = []
        for c in caps:
            captions_flat.append(str(c))
            caption_to_image_index.append(image_idx)
            idxs.append(running_idx)
            running_idx += 1
        image_to_caption_indices.append(idxs)

    # Encode images
    image_features: List[torch.Tensor] = []
    model_device = device
    # optional progress bar
    _range: object = range  # type: ignore[assignment]
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore
            _range = lambda *args, **kwargs: tqdm(range(*args, **kwargs), desc=(progress_desc or "images"))  # type: ignore[assignment]
        except Exception:
            _range = range

    for start in _range(0, len(image_paths), batch_size):  # type: ignore[misc]
        batch_paths = image_paths[start:start + batch_size]
        images: List[torch.Tensor] = []
        for p in batch_paths:
            with Image.open(p) as img:
                images.append(preprocess(img.convert("RGB")))
        image_tensor = torch.stack(images, dim=0).to(model_device)
        with torch.no_grad():
            feats = model.encode_image(image_tensor)
            feats = feats.float()
            feats = _normalize_features(feats)
        image_features.append(feats.cpu())
    image_embeds = torch.cat(image_features, dim=0).numpy()

    # Encode captions (optionally chunked and pooled)
    if text_chunk_words and text_chunk_words > 0:
        # Build chunk lists per caption
        all_chunks: List[str] = []
        caption_chunk_slices: List[Tuple[int, int]] = []
        for cap in captions_flat:
            if chunk_threshold_tokens is not None and len(cap.split()) < chunk_threshold_tokens:
                chunks = [cap]
            else:
                chunks = chunk_caption_words(cap, text_chunk_words, text_chunk_stride)
            if len(chunks) == 0:
                chunks = [cap]
            start = len(all_chunks)
            all_chunks.extend(chunks)
            end = len(all_chunks)
            caption_chunk_slices.append((start, end))

        # Encode all chunks
        chunk_features: List[torch.Tensor] = []
        text_batch = max(batch_size, 64)
        _range_t: object = range  # type: ignore[assignment]
        if progress:
            try:
                from tqdm.auto import tqdm  # type: ignore
                _range_t = lambda *args, **kwargs: tqdm(range(*args, **kwargs), desc=(progress_desc or "text-chunks"))  # type: ignore[assignment]
            except Exception:
                _range_t = range
        for start in _range_t(0, len(all_chunks), text_batch):  # type: ignore[misc]
            batch_caps = all_chunks[start:start + text_batch]
            tokens = tokenizer(batch_caps)
            tokens = tokens.to(model_device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats.float()
                feats = _normalize_features(feats)
            chunk_features.append(feats.cpu())
        chunk_matrix = torch.cat(chunk_features, dim=0)  # (total_chunks, d)

        # Pool back to one embedding per caption
        pooled_caption_embeds: List[torch.Tensor] = []
        for s, e in caption_chunk_slices:
            pooled = pool_chunk_embeddings(chunk_matrix[s:e, :], mode=text_pooling)
            pooled_caption_embeds.append(pooled.unsqueeze(0))
        caption_embeds = torch.cat(pooled_caption_embeds, dim=0).numpy()
    else:
        text_features: List[torch.Tensor] = []
        text_batch = max(batch_size, 64)
        _range_t: object = range  # type: ignore[assignment]
        if progress:
            try:
                from tqdm.auto import tqdm  # type: ignore
                _range_t = lambda *args, **kwargs: tqdm(range(*args, **kwargs), desc=(progress_desc or "text"))  # type: ignore[assignment]
            except Exception:
                _range_t = range
        for start in _range_t(0, len(captions_flat), text_batch):  # type: ignore[misc]
            batch_caps = captions_flat[start:start + text_batch]
            tokens = tokenizer(batch_caps)
            tokens = tokens.to(model_device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats.float()
                feats = _normalize_features(feats)
            text_features.append(feats.cpu())
        caption_embeds = torch.cat(text_features, dim=0).numpy()

    return {
        "image_embeddings": image_embeds,
        "caption_embeddings": caption_embeds,
        "caption_to_image_index": np.asarray(caption_to_image_index, dtype=np.int64),
        "image_to_caption_indices": image_to_caption_indices,
    }


def _recall_from_hits(hits: np.ndarray, ks: Tuple[int, ...]) -> Dict[int, float]:
    n = float(hits.shape[0])
    recalls: Dict[int, float] = {}
    for k in ks:
        topk = hits[:, :k].any(axis=1).sum()
        recalls[k] = float(topk) / n
    return recalls


def recall_at_k_text_to_image(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    caption_to_image_index: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[int, float]:
    # Similarity: captions x images
    sim = (caption_embeddings @ image_embeddings.T).astype(np.float32)  # (M, N)
    M, N = sim.shape
    # Argpartition for top-k indices per caption
    max_k = max(ks)
    topk_idx = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    # For correctness check, we need sorted top-k by score for consistency
    row_indices = np.arange(M)[:, None]
    topk_scores = sim[row_indices, topk_idx]
    order = np.argsort(-topk_scores, axis=1)
    topk_sorted = topk_idx[row_indices, order]
    correct_img = caption_to_image_index
    hits = (topk_sorted == correct_img[:, None])
    return _recall_from_hits(hits, ks)


def recall_at_k_image_to_text(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    image_to_caption_indices: List[List[int]],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[int, float]:
    # Similarity: images x captions
    sim = (image_embeddings @ caption_embeddings.T).astype(np.float32)  # (N, M)
    N, M = sim.shape
    max_k = max(ks)
    topk_idx = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    row_indices = np.arange(N)[:, None]
    topk_scores = sim[row_indices, topk_idx]
    order = np.argsort(-topk_scores, axis=1)
    topk_sorted = topk_idx[row_indices, order]

    # Build a boolean hits matrix: whether any correct caption is in top-k
    hits = np.zeros_like(topk_sorted, dtype=bool)
    for i in range(N):
        correct_caps = set(image_to_caption_indices[i])
        hits[i, :] = np.isin(topk_sorted[i, :], list(correct_caps))
    return _recall_from_hits(hits, ks)

