"""
Adapter training pipeline for the Visual Genome dataset.

This module mirrors the functionality of the original notebook-style
script while providing a cleaner callable API that works in both local
and Colab environments. Configuration defaults live at the top of the
file and can be overridden via the `AdapterExperimentConfig` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import open_clip  # type: ignore

from adapter import attach_text_adapters, detach_text_adapters, normalize_features
from src.config.runtime import RuntimePaths
from src.data.visual_genome import PROCESSED_JSON_NAME, VisualGenomeProcessConfig
from utils import (
    chunk_caption_words,
    encode_openclip_embeddings,
    pool_chunk_embeddings,
    recall_at_k_image_to_text,
    recall_at_k_text_to_image,
    select_torch_device,
    set_all_seeds,
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42

DEFAULT_RUN_SPLITS: Tuple[str, ...] = ("train", "val", "test")
DEFAULT_LIMIT_IMAGES: Optional[int] = 8000
DEFAULT_BATCH_SIZE = 32
DEFAULT_K_VALUES: Tuple[int, ...] = (1, 5, 10)

DEFAULT_CHUNK_WORDS = 8
DEFAULT_CHUNK_STRIDE = 4
DEFAULT_CHUNK_THRESHOLD = 12
DEFAULT_TEXT_POOLING = "attn"

DEFAULT_ADAPTER_STEPS = 300
DEFAULT_ADAPTER_LR = 1e-5
DEFAULT_ADAPTER_LOGIT_LR = 5e-7
DEFAULT_ADAPTER_BATCH = 32
DEFAULT_ADAPTER_HIDDEN = 64
DEFAULT_DISTILL_WEIGHTS: Tuple[float, ...] = (0.0, 0.25)
DEFAULT_FINE_GRAINED_WEIGHT = 0.5

DEFAULT_CACHE_CHUNK_IMAGE_EMBEDS = True
DEFAULT_TEACHER_CACHE_ON_CPU = True
DEFAULT_OUTPUT_NAME = "visual_genome_adapter"
DEFAULT_SHOW_PROGRESS = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdapterExperimentConfig:
    model_name: str = DEFAULT_MODEL_NAME
    pretrained: str = DEFAULT_PRETRAINED
    device_preference: str = DEFAULT_DEVICE
    seed: int = DEFAULT_SEED
    run_splits: Tuple[str, ...] = DEFAULT_RUN_SPLITS
    limit_images: Optional[int] = DEFAULT_LIMIT_IMAGES
    batch_size: int = DEFAULT_BATCH_SIZE
    k_values: Tuple[int, ...] = DEFAULT_K_VALUES
    chunk_words: int = DEFAULT_CHUNK_WORDS
    chunk_stride: int = DEFAULT_CHUNK_STRIDE
    chunk_threshold: int = DEFAULT_CHUNK_THRESHOLD
    text_pooling: str = DEFAULT_TEXT_POOLING
    adapter_steps: int = DEFAULT_ADAPTER_STEPS
    adapter_lr: float = DEFAULT_ADAPTER_LR
    adapter_logit_lr: float = DEFAULT_ADAPTER_LOGIT_LR
    adapter_batch: int = DEFAULT_ADAPTER_BATCH
    adapter_hidden: int = DEFAULT_ADAPTER_HIDDEN
    distill_weights: Tuple[float, ...] = DEFAULT_DISTILL_WEIGHTS
    fine_grained_weight: float = DEFAULT_FINE_GRAINED_WEIGHT
    cache_chunk_image_embeds: bool = DEFAULT_CACHE_CHUNK_IMAGE_EMBEDS
    teacher_cache_on_cpu: bool = DEFAULT_TEACHER_CACHE_ON_CPU
    output_name: str = DEFAULT_OUTPUT_NAME
    process_config: VisualGenomeProcessConfig = field(default_factory=VisualGenomeProcessConfig)
    show_progress: bool = DEFAULT_SHOW_PROGRESS

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["process_config"] = self.process_config.to_dict()
        return payload


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_visual_genome_dataset(
    paths: RuntimePaths,
    run_splits: Sequence[str],
    limit_images: Optional[int],
) -> Dict[str, List[Dict[str, object]]]:
    processed_path = paths.processed_data_dir / PROCESSED_JSON_NAME
    if not processed_path.is_file():
        raise FileNotFoundError(
            f"Processed Visual Genome dataset not found at {processed_path}. "
            "Run process_visual_genome() first."
        )
    with processed_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "splits" not in payload:
        raise ValueError("Processed dataset has an unexpected structure.")
    splits = payload["splits"]
    datasets: Dict[str, List[Dict[str, object]]] = {}
    for split in run_splits:
        entries = splits.get(split)
        if not isinstance(entries, list):
            raise ValueError(f"Split '{split}' not found in processed dataset.")
        result: List[Dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            image_path = entry.get("image_path")
            captions = entry.get("captions")
            if not image_path or not captions:
                continue
            sample = {
                "image_path": image_path,
                "captions": list(captions),  # type: ignore[arg-type]
            }
            result.append(sample)
            if limit_images is not None and len(result) >= limit_images:
                break
        datasets[split] = result
    return datasets


def _chunk_segments(
    caption: str,
    chunk_words: int,
    chunk_stride: int,
    chunk_threshold: int,
) -> List[str]:
    words = caption.split()
    if chunk_words <= 0 or len(words) < chunk_threshold:
        return [caption]
    segments = chunk_caption_words(caption, chunk_words, chunk_stride)
    return segments or [caption]


def _encode_chunked_embeddings(
    dataset: List[Dict[str, object]],
    *,
    preprocess,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    chunk_words: int,
    chunk_stride: int,
    chunk_threshold: int,
    text_pooling: str,
    image_embed_cache: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    if image_embed_cache is not None:
        image_embeds = np.asarray(image_embed_cache, dtype=np.float32)
    else:
        image_paths = [str(item["image_path"]) for item in dataset]
        image_embeddings: List[torch.Tensor] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images: List[torch.Tensor] = []
            for path in batch_paths:
                with Image.open(path).convert("RGB") as img:
                    images.append(preprocess(img))
            image_tensor = torch.stack(images, dim=0).to(device)
            with torch.no_grad():
                feats = model.encode_image(image_tensor).float()
            image_embeddings.append(normalize_features(feats).cpu())
        image_embeds = torch.cat(image_embeddings, dim=0).cpu().numpy()

    caption_info: List[Tuple[int, int, int]] = []
    segments: List[str] = []
    for image_idx, item in enumerate(dataset):
        captions = cast(List[str], item["captions"])
        for caption in captions:
            chunked = _chunk_segments(
                caption,
                chunk_words=chunk_words,
                chunk_stride=chunk_stride,
                chunk_threshold=chunk_threshold,
            )
            start = len(segments)
            segments.extend(chunked)
            end = len(segments)
            caption_info.append((image_idx, start, end))

    segment_embeds: List[torch.Tensor] = []
    for start in range(0, len(segments), batch_size):
        batch_segments = segments[start:start + batch_size]
        tokens = tokenizer(batch_segments).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens).float()
        segment_embeds.append(normalize_features(feats).cpu())
    segment_matrix = torch.cat(segment_embeds, dim=0)

    caption_embeddings: List[np.ndarray] = []
    caption_to_image_index: List[int] = []
    image_to_caption_indices: List[List[int]] = []
    caption_ptr = 0
    for image_idx, item in enumerate(dataset):
        captions = cast(List[str], item["captions"])
        indices: List[int] = []
        for _caption in captions:
            img_idx, start, end = caption_info[caption_ptr]
            chunk_tensor = segment_matrix[start:end]
            pooled = pool_chunk_embeddings(chunk_tensor, mode=text_pooling)
            caption_embeddings.append(pooled.cpu().numpy())
            caption_to_image_index.append(img_idx)
            indices.append(len(caption_embeddings) - 1)
            caption_ptr += 1
        image_to_caption_indices.append(indices)

    return {
        "image_embeddings": image_embeds,
        "caption_embeddings": np.asarray(caption_embeddings, dtype=np.float32),
        "caption_to_image_index": np.asarray(caption_to_image_index, dtype=np.int64),
        "image_to_caption_indices": image_to_caption_indices,
    }


def _compute_recalls(embeddings: Dict[str, object], ks: Sequence[int]) -> Dict[str, Dict[int, float]]:
    img = np.asarray(embeddings["image_embeddings"], dtype=np.float32)
    cap = np.asarray(embeddings["caption_embeddings"], dtype=np.float32)
    cap2img = np.asarray(embeddings["caption_to_image_index"], dtype=np.int64)
    img2cap = cast(List[List[int]], embeddings["image_to_caption_indices"])
    return {
        "t2i": recall_at_k_text_to_image(img, cap, cap2img, ks=tuple(ks)),
        "i2t": recall_at_k_image_to_text(img, cap, img2cap, ks=tuple(ks)),
    }


def _prepare_teacher_cache(
    embeddings: Dict[str, Dict[str, object]],
    split: str,
    cache_on_cpu: bool,
) -> Tuple[Optional[torch.Tensor], Optional[List[List[int]]]]:
    teacher_source = embeddings.get(split)
    if not teacher_source:
        return None, None
    caps_tensor = torch.tensor(
        teacher_source["caption_embeddings"],
        dtype=torch.float32,
    )
    caps_tensor = normalize_features(caps_tensor)
    teacher_caps = caps_tensor.cpu() if cache_on_cpu else caps_tensor
    teacher_img2cap = cast(List[List[int]], teacher_source["image_to_caption_indices"])
    return teacher_caps, teacher_img2cap


def _train_adapter(
    model,
    tokenizer,
    preprocess,
    dataset: List[Dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
    steps: int,
    chunk_words: int,
    chunk_stride: int,
    chunk_threshold: int,
    text_pooling: str,
    fine_grained_weight: float,
    distill_weight: float,
    teacher_caps: Optional[torch.Tensor],
    teacher_img2cap: Optional[List[List[int]]],
    adapter_hidden: int,
    adapter_lr: float,
    adapter_logit_lr: float,
    progress: bool = False,
    progress_desc: str = "",
) -> List[Tuple[int, float, float]]:
    adapter = attach_text_adapters(model, hidden_dim=adapter_hidden)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(True)

    logit_scale = cast(torch.nn.Parameter, model.logit_scale)
    logit_scale.requires_grad_(True)

    adapter_params = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": adapter_lr, "weight_decay": 1e-4},
            {"params": [logit_scale], "lr": adapter_logit_lr, "weight_decay": 0.0},
        ]
    )

    indices = list(range(len(dataset)))
    loss_log: List[Tuple[int, float, float]] = []

    iterator = range(steps)
    progress_bar = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(range(steps), desc=progress_desc or "adapter-train")  # type: ignore[assignment]
            progress_bar = iterator  # type: ignore[assignment]
        except Exception:
            iterator = range(steps)
            progress_bar = None

    for step in iterator:
        batch_idx = random.sample(indices, k=min(batch_size, len(indices)))
        images: List[torch.Tensor] = []
        captions: List[str] = []
        caption_segments: List[str] = []
        caption_spans: List[Tuple[int, int]] = []
        segment_image_indices: List[int] = []
        teacher_pick: List[int] = []

        for idx in batch_idx:
            entry = dataset[idx]
            caption_list = cast(List[str], entry["captions"])
            cap_pos = random.randrange(len(caption_list))
            selected_caption = caption_list[cap_pos]
            captions.append(selected_caption)
            with Image.open(entry["image_path"]).convert("RGB") as img:
                images.append(preprocess(img))

            chunked = _chunk_segments(
                selected_caption,
                chunk_words=chunk_words,
                chunk_stride=chunk_stride,
                chunk_threshold=chunk_threshold,
            )
            start = len(caption_segments)
            caption_segments.extend(chunked)
            caption_spans.append((start, len(caption_segments)))
            segment_image_indices.extend([len(images) - 1] * len(chunked))

            if teacher_img2cap is not None:
                teacher_pick.append(teacher_img2cap[idx][cap_pos])

        image_tensor = torch.stack(images, dim=0).to(device)
        with torch.no_grad():
            encoded_images = model.encode_image(image_tensor).float()
        image_features = normalize_features(encoded_images)

        if not caption_segments:
            raise RuntimeError("No caption segments generated in adapter batch.")

        segment_tokens = tokenizer(caption_segments).to(device)
        segment_embeds = model.encode_text(segment_tokens).float()
        segment_embeds = normalize_features(segment_embeds)

        pooled_text: List[torch.Tensor] = []
        for start, end in caption_spans:
            pooled_text.append(pool_chunk_embeddings(segment_embeds[start:end], mode=text_pooling))
        text_features = torch.stack(pooled_text, dim=0)

        logits = (text_features @ image_features.t()) * logit_scale.exp()
        targets = torch.arange(logits.size(0), device=device)
        loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) * 0.5

        if fine_grained_weight > 0 and segment_image_indices:
            seg_targets = torch.tensor(segment_image_indices, dtype=torch.long, device=device)
            fine_logits = (segment_embeds @ image_features.t()) * logit_scale.exp()
            fine_loss = F.cross_entropy(fine_logits, seg_targets)
            loss = loss + fine_grained_weight * fine_loss

        if distill_weight > 0 and teacher_caps is not None and teacher_pick:
            teacher_device = teacher_caps.device
            idx_tensor = torch.tensor(teacher_pick, dtype=torch.long, device=teacher_device)
            teacher_batch = teacher_caps[idx_tensor].to(device)
            loss = loss + distill_weight * F.mse_loss(text_features, teacher_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params + [logit_scale], max_norm=1.0)
        optimizer.step()
        logit_scale.data.clamp_(min=math.log(1 / 20), max=math.log(50))

        if step % 10 == 0 or step == steps - 1:
            loss_log.append((step, float(loss.item()), float(logit_scale.exp().item())))
        if progress_bar is not None:
            try:
                progress_bar.set_postfix(
                    loss=float(loss.item()),
                    scale=float(logit_scale.exp().item()),
                )
            except Exception:
                pass

    if progress_bar is not None:
        try:
            progress_bar.close()
        except Exception:
            pass

    detach_text_adapters(model)
    logit_scale.requires_grad_(False)
    return loss_log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_visual_genome_adapter(paths: RuntimePaths, config: AdapterExperimentConfig) -> Dict[str, object]:
    """Execute the Visual Genome adapter experiment and return metrics."""
    set_all_seeds(config.seed)
    device = select_torch_device(config.device_preference)

    datasets = _load_visual_genome_dataset(
        paths,
        run_splits=config.run_splits,
        limit_images=config.limit_images,
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model_name,
        pretrained=config.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(config.model_name)
    model.eval()

    baseline_embeddings: Dict[str, Dict[str, object]] = {}
    baseline_metrics: Dict[str, Dict[str, Dict[int, float]]] = {}
    chunk_embeddings: Dict[str, Dict[str, object]] = {}
    chunk_metrics: Dict[str, Dict[str, Dict[int, float]]] = {}
    chunk_image_cache: Dict[str, np.ndarray] = {}

    for split in config.run_splits:
        dataset = datasets[split]
        emb = encode_openclip_embeddings(
            dataset=dataset,
            preprocess=preprocess,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=config.batch_size,
            text_chunk_words=0,
            text_chunk_stride=0,
            text_pooling="mean",
            progress=False,
            progress_desc=f"{split}|baseline",
            chunk_threshold_tokens=10_000,
        )
        metrics = _compute_recalls(emb, config.k_values)
        baseline_embeddings[split] = emb
        baseline_metrics[split] = metrics

    if config.chunk_words > 0:
        for split in config.run_splits:
            dataset = datasets[split]
            cache = chunk_image_cache.get(split) if config.cache_chunk_image_embeds else None
            emb = _encode_chunked_embeddings(
                dataset=dataset,
                preprocess=preprocess,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=config.batch_size,
                chunk_words=config.chunk_words,
                chunk_stride=config.chunk_stride,
                chunk_threshold=config.chunk_threshold,
                text_pooling=config.text_pooling,
                image_embed_cache=cache,
            )
            metrics = _compute_recalls(emb, config.k_values)
            chunk_embeddings[split] = emb
            chunk_metrics[split] = metrics
            if config.cache_chunk_image_embeds and cache is None:
                chunk_image_cache[split] = np.asarray(emb["image_embeddings"], dtype=np.float32)
    else:
        chunk_embeddings = baseline_embeddings
        chunk_metrics = baseline_metrics

    teacher_caps, teacher_img2cap = (None, None)
    if config.distill_weights and any(weight > 0 for weight in config.distill_weights):
        teacher_caps, teacher_img2cap = _prepare_teacher_cache(
            chunk_embeddings if config.chunk_words > 0 else baseline_embeddings,
            split="train",
            cache_on_cpu=config.teacher_cache_on_cpu,
        )

    adapter_results: Dict[float, Dict[str, Dict[str, Dict[int, float]]]] = {}
    loss_logs: Dict[float, List[Tuple[int, float, float]]] = {}

    for weight in config.distill_weights:
        if weight < 0:
            raise ValueError("Distillation weight must be non-negative.")
        loss_log = _train_adapter(
            model,
            tokenizer,
            preprocess,
            datasets["train"],
            device=device,
            batch_size=config.adapter_batch,
            steps=config.adapter_steps,
            chunk_words=config.chunk_words,
            chunk_stride=config.chunk_stride,
            chunk_threshold=config.chunk_threshold,
            text_pooling=config.text_pooling,
            fine_grained_weight=config.fine_grained_weight,
            distill_weight=weight,
            teacher_caps=teacher_caps,
            teacher_img2cap=teacher_img2cap,
            adapter_hidden=config.adapter_hidden,
            adapter_lr=config.adapter_lr,
            adapter_logit_lr=config.adapter_logit_lr,
            progress=config.show_progress,
            progress_desc=f"distill={weight:.2f}",
        )
        loss_logs[weight] = loss_log

        current_metrics: Dict[str, Dict[str, Dict[int, float]]] = {}
        for split in config.run_splits:
            cache = chunk_image_cache.get(split) if config.cache_chunk_image_embeds else None
            if config.chunk_words > 0:
                emb = _encode_chunked_embeddings(
                    dataset=datasets[split],
                    preprocess=preprocess,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=config.batch_size,
                    chunk_words=config.chunk_words,
                    chunk_stride=config.chunk_stride,
                    chunk_threshold=config.chunk_threshold,
                    text_pooling=config.text_pooling,
                    image_embed_cache=cache,
                )
                if config.cache_chunk_image_embeds and cache is None:
                    chunk_image_cache[split] = np.asarray(emb["image_embeddings"], dtype=np.float32)
            else:
                emb = encode_openclip_embeddings(
                    dataset=datasets[split],
                    preprocess=preprocess,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=config.batch_size,
                    text_chunk_words=0,
                    text_chunk_stride=0,
                    text_pooling="mean",
                    progress=False,
                    progress_desc=f"{split}|adapter",
                    chunk_threshold_tokens=10_000,
                )
            metrics = _compute_recalls(emb, config.k_values)
            current_metrics[split] = metrics
        adapter_results[weight] = current_metrics

    summary_rows: List[Dict[str, object]] = []
    for split in config.run_splits:
        base_metrics = baseline_metrics[split]
        chunk_base_metrics = chunk_metrics[split]
        for direction in ("t2i", "i2t"):
            for k in config.k_values:
                row = {
                    "split": split,
                    "metric": f"{direction}@{k}",
                    "baseline": base_metrics[direction][k],
                    "chunk_baseline": chunk_base_metrics[direction][k],
                }
                for weight, metrics in adapter_results.items():
                    row[f"adapter_{weight:.2f}"] = metrics[split][direction][k]
                summary_rows.append(row)

    results: Dict[str, object] = {
        "config": config.to_dict(),
        "paths": paths.to_dict(),
        "baseline_metrics": baseline_metrics,
        "chunk_metrics": chunk_metrics,
        "adapter_metrics": adapter_results,
        "loss_logs": {f"{w:.2f}": log for w, log in loss_logs.items()},
        "summary": summary_rows,
    }

    output_dir = Path(paths.output_dir) / config.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    results["metrics_path"] = str(metrics_path)
    return results


__all__ = [
    "AdapterExperimentConfig",
    "run_visual_genome_adapter",
]


