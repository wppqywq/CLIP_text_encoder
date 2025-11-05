from __future__ import annotations

import math
import random
from typing import Dict, Optional, List, Tuple, cast

import numpy as np
import torch
import functools


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


class TextAdapter(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, hidden_dim, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, embed_dim, bias=False),
            )
            for m in self.net:
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
        else:
            lin = torch.nn.Linear(embed_dim, embed_dim, bias=False)
            with torch.no_grad():
                lin.weight.copy_(torch.eye(embed_dim))
            self.net = lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def infonce_symm_loss(img: torch.Tensor, txt: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    img = normalize_features(img)
    txt = normalize_features(txt)
    logits = torch.matmul(txt, img.t()) * logit_scale.exp()
    target = torch.arange(img.size(0), device=img.device)
    loss_i = torch.nn.functional.cross_entropy(logits.t(), target)
    loss_t = torch.nn.functional.cross_entropy(logits, target)
    return (loss_i + loss_t) * 0.5


def train_text_adapter_infonce(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    caption_to_image_index: np.ndarray,
    steps: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.07,
    hidden_dim: int = 0,
    device: Optional[torch.device] = None,
    progress: bool = True,
    teacher_embeddings: Optional[np.ndarray] = None,
    distill_weight: float = 0.0,
) -> Dict[str, object]:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.tensor(image_embeddings, dtype=torch.float32, device=dev)
    txt0 = torch.tensor(caption_embeddings, dtype=torch.float32, device=dev)
    cap2img = torch.tensor(caption_to_image_index, dtype=torch.long, device=dev)
    if teacher_embeddings is not None:
        teacher = torch.tensor(teacher_embeddings, dtype=torch.float32, device=dev)
    else:
        teacher = txt0.detach().clone()

    adapter = TextAdapter(txt0.shape[1], hidden_dim=hidden_dim).to(dev)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / max(temperature, 1e-6)), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(list(adapter.parameters()) + [logit_scale], lr=lr)

    num_caps = txt0.shape[0]
    rng = range
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore
            rng = lambda *a, **k: tqdm(range(*a, **k), desc="train")  # type: ignore[assignment]
        except Exception:
            rng = range

    losses = []
    adapter.train()
    for step in rng(steps):
        idx = torch.tensor([random.randrange(num_caps) for __ in range(batch_size)], device=dev)
        img_idx = cap2img[idx]
        img_b = img[img_idx]
        txt_b = adapter(txt0[idx])
        loss = infonce_symm_loss(img_b, txt_b, logit_scale)
        if distill_weight > 0.0:
            teacher_b = teacher[idx]
            distill = torch.nn.functional.mse_loss(
                normalize_features(txt_b),
                normalize_features(teacher_b),
            )
            loss = loss + float(distill_weight) * distill
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip logit_scale gradient to prevent explosion
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + [logit_scale], max_norm=1.0)
        opt.step()
        # clip logit_scale value
        with torch.no_grad():
            logit_scale.data.clamp_(min=math.log(1.0 / 100), max=math.log(100.0))
        
        if step % max(50, steps // 10) == 0 or step == steps - 1:
            losses.append(float(loss.item()))

    adapter.eval()
    return {
        "adapter_state": adapter.state_dict(),
        "adapter_hidden": int(hidden_dim),
        "logit_scale": float(logit_scale.exp().item()),
        "losses": losses,
        "distill_weight": float(distill_weight),
    }


def train_logit_scale_only(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    caption_to_image_index: np.ndarray,
    steps: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.07,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> Dict[str, object]:
    """
    Train only logit_scale (no adapter). Safer approach that doesn't modify embeddings.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.tensor(image_embeddings, dtype=torch.float32, device=dev)
    txt = torch.tensor(caption_embeddings, dtype=torch.float32, device=dev)
    cap2img = torch.tensor(caption_to_image_index, dtype=torch.long, device=dev)

    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / max(temperature, 1e-6)), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW([logit_scale], lr=lr)

    num_caps = txt.shape[0]
    rng = range
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore
            rng = lambda *a, **k: tqdm(range(*a, **k), desc="train_scale")  # type: ignore[assignment]
        except Exception:
            rng = range

    losses = []
    for step in rng(steps):
        idx = torch.tensor([random.randrange(num_caps) for __ in range(batch_size)], device=dev)
        img_idx = cap2img[idx]
        img_b = img[img_idx]
        txt_b = txt[idx]
        loss = infonce_symm_loss(img_b, txt_b, logit_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([logit_scale], max_norm=1.0)
        opt.step()
        with torch.no_grad():
            logit_scale.data.clamp_(min=math.log(1.0 / 100), max=math.log(100.0))
        
        if step % max(50, steps // 10) == 0 or step == steps - 1:
            losses.append(float(loss.item()))

    return {
        "logit_scale": float(logit_scale.exp().item()),
        "losses": losses,
    }


class ResidualMLP(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embed_dim, bias=False),
        )
        last_layer = cast(torch.nn.Linear, self.mlp[-1])
        torch.nn.init.zeros_(last_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class TransformerAdapter(torch.nn.Module):
    def __init__(
        self,
        transformer: torch.nn.Module,
        hidden_dim: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive for TransformerAdapter")
        if not hasattr(transformer, "resblocks"):
            raise AttributeError("Transformer module must have resblocks attribute")
        transformer = cast(torch.nn.Module, transformer)
        blocks: List[torch.nn.Module] = list(getattr(transformer, "resblocks"))  # type: ignore[arg-type]
        if not blocks:
            raise ValueError("Transformer must contain at least one block")
        first_block = cast(torch.nn.Module, blocks[0])
        ln1 = cast(torch.nn.LayerNorm, getattr(first_block, "ln_1"))
        normalized_shape = cast(Tuple[int, ...], ln1.normalized_shape)
        embed_dim = int(normalized_shape[0])
        self.bottlenecks = torch.nn.ModuleList(
            [ResidualMLP(embed_dim, hidden_dim) for _ in blocks]
        )
        if device is not None:
            self.to(device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("TransformerAdapter should be injected, not called directly")


def attach_text_adapters(model: torch.nn.Module, hidden_dim: int) -> TransformerAdapter:
    transformer = cast(torch.nn.Module, model.transformer)
    try:
        device = next(transformer.parameters()).device
    except StopIteration:
        device = next(model.parameters()).device
    adapter = TransformerAdapter(transformer, hidden_dim, device=device)

    blocks: List[torch.nn.Module] = list(getattr(transformer, "resblocks"))  # type: ignore[arg-type]

    def make_patched(orig_forward, bottleneck_module):
        @functools.wraps(orig_forward)
        def patched_forward(x, attn_mask=None):
            out = orig_forward(x, attn_mask=attn_mask)
            return bottleneck_module(out)

        return patched_forward

    for block, bottleneck in zip(blocks, adapter.bottlenecks):
        original_forward = block.forward
        patched = make_patched(original_forward, bottleneck)
        block.forward = patched  # type: ignore
    return adapter


def detach_text_adapters(model: torch.nn.Module) -> None:
    transformer = cast(torch.nn.Module, model.transformer)
    blocks: List[torch.nn.Module] = list(getattr(transformer, "resblocks"))  # type: ignore[arg-type]
    for block in blocks:
        forward_fn = getattr(block, "forward", None)
        orig = getattr(forward_fn, "__wrapped__", None)
        if orig is not None:
            block.forward = orig  # type: ignore

