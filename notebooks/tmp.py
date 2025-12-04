# %% [markdown]
# 
# 

# %%
# Parameter block (edit these before running anything else)
CONFIG = {
    "repo_url": "https://github.com/wppqywq/CLIP_text_encoder.git",  # GitHub repo to clone
    "repo_branch": "main",
    "repo_dir": "/content/comp545_final_github",
    "drive_mount": "/content/drive",
    "data_root": "/content/drive/MyDrive/comp545_data",
    "output_root": "/content/drive/MyDrive/comp545_outputs",
    "env": "colab",
    "packages": [
        "open-clip-torch",
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "matplotlib",
        "pandas",
        "tqdm",
    ],
    "smoke_limit": 1000,  # number of images for the quick smoke test
    "smoke_distill": (0.2, 0.8),
    "smoke_output": "vg_smoke",
    "full_limit": None,  # set to None for the full dataset
    "full_distill": (0.0, 0.2, 0.8),
    "full_output": "vg_full",
    "adapter_steps": 300,
    "adapter_batch": 64,
    "chunk_words": 8,
    "chunk_stride": 4,
    "chunk_threshold": 15,
    "text_pooling": "attn",
    "show_progress": True,
}



# %%
# Install dependencies (no-op if already satisfied)
if CONFIG["packages"]:
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *CONFIG["packages"]]
    subprocess.run(cmd, check=True)
else:
    print("No extra packages listed.")


# %%
# Mount Google Drive and clone the repository if needed
import subprocess, shutil
from pathlib import Path

from google.colab import drive

MOUNT_POINT = Path(CONFIG["drive_mount"])
if not MOUNT_POINT.is_dir():
    drive.mount(str(MOUNT_POINT))

repo_dir = Path(CONFIG["repo_dir"]).resolve()
shutil.rmtree(repo_dir, ignore_errors=True)
# if not repo_dir.exists():
if 1:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = [
        "git",
        "clone",
        CONFIG["repo_url"],
        str(repo_dir),
        "--branch",
        CONFIG["repo_branch"],
        "--single-branch",
    ]
    subprocess.run(clone_cmd, check=True)
else:
    print(f"Repo already present at {repo_dir}")


# %%
# Configure project paths and ensure modules import correctly
import os
import sys
import time

from pathlib import Path

repo_root = Path(CONFIG["repo_dir"]).resolve()
data_root = Path(CONFIG["data_root"]).resolve()
output_root = Path(CONFIG["output_root"]).resolve()

os.environ["VG_COLAB_REPO_ROOT"] = str(repo_root)
os.environ["VG_COLAB_DATA_ROOT"] = str(data_root)
os.environ["VG_COLAB_OUTPUT_ROOT"] = str(output_root)

repo_root.mkdir(parents=True, exist_ok=True)
data_root.mkdir(parents=True, exist_ok=True)
output_root.mkdir(parents=True, exist_ok=True)

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print("repo_root:", repo_root)
print("data_root:", data_root)
print("output_root:", output_root)


# %%
import zipfile
from pathlib import Path

if 0:
  # Define paths based on user query
  zip_path = Path("/content/drive/MyDrive/comp545_data/visual_genome_raw.zip")
  target_dir = Path("/content/drive/MyDrive/comp545_data/")

  # Ensure the target directory exists
  target_dir.mkdir(parents=True, exist_ok=True)

  if zip_path.is_file():
      print(f"Extracting {zip_path} to {target_dir}")
      with zipfile.ZipFile(zip_path, 'r') as zf:
          zf.extractall(target_dir)
      print("Extraction complete.")
  else:
      print(f"Error: Zip file not found at {zip_path}")

# %%
# Optional helper: unzip image archives from Drive into the expected directory

if 0:
  zip_map = {
      "images.zip": data_root / "visual_genome" / "images",
      "images2.zip": data_root / "visual_genome" / "images",
  }

  for zip_name, target_dir in zip_map.items():
      archive_path = data_root / "visual_genome_raw" / zip_name
      if not archive_path.is_file():
          print(f"Archive not found: {archive_path}")
          continue
      target_dir.mkdir(parents=True, exist_ok=True)
      with zipfile.ZipFile(archive_path, "r") as zf:
          print(f"Extracting {zip_name} -> {target_dir}")
          zf.extractall(target_dir)
  print("Done extracting image archives.")


# %% [markdown]
# ## Dataset Checklist
# Ensure Drive contains the Visual Genome archives under `CONFIG['data_root']`:
# ```
# visual_genome_raw/
#   region_descriptions.json (or .zip)
#   image_data.json (or .zip)
#   VG_100K.zip
#   VG_100K_2.zip
# visual_genome/
#   images/VG_100K/
#   images/VG_100K_2/
# ```
# If you already processed splits elsewhere, copy `visual_genome/visual_genome_splits.json` here. Otherwise, run the next cell to prepare everything from scratch.
# 

# %%
import os
test_path = os.path.join(CONFIG["data_root"], "visual_genome/images/VG_100K/2.jpg")
!ls -lh "$test_path"

# %%
# Optional: download/process/verify Visual Genome (set RUN_PROCESS=True when needed)
RUN_PROCESS = 1
PROCESS_CONFIG = {
    "max_images": 5000,
    "max_regions_per_image": 6,
    "min_region_words": 3,
    "validation_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
}

if RUN_PROCESS:
    from src.config.runtime import resolve_paths
    from src.data.visual_genome import (
        VisualGenomeProcessConfig,
        download_visual_genome,
        process_visual_genome,
        verify_visual_genome,
    )

    # paths = resolve_paths("colab")
    paths = resolve_paths(CONFIG.get("env", "colab"))
    process_cfg = VisualGenomeProcessConfig(**PROCESS_CONFIG)
    # download_visual_genome(paths, include_images=True, force=False)
    processed_path = process_visual_genome(paths, process_cfg)
    print("Processed splits saved to:", processed_path)
    verify_visual_genome(paths)
else:
    print("Skipping data preparation. Toggle RUN_PROCESS=True if required.")


# %%
# Inspect processed dataset statistics
import json

splits_path = data_root / "visual_genome" / "visual_genome_splits.json"
if splits_path.is_file():
    with splits_path.open("r", encoding="utf-8") as f:
        vg_payload = json.load(f)
    counts = vg_payload.get("counts", {})
    splits = vg_payload.get("splits", {})
    print("Processed dataset counts:", counts)
    for name, entries in splits.items():
        size = len(entries) if isinstance(entries, list) else 0
        print(f"  {name}: {size} images")
else:
    print(f"Processed splits not found at {splits_path}; run the preparation cell if needed.")

# %%
# Helper utilities for running experiments and plotting metrics
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from PIL import Image
import open_clip  # type: ignore

from src.config.runtime import resolve_paths
from src.training import vg_adapter as vg_module
from src.training.vg_adapter import AdapterExperimentConfig, run_visual_genome_adapter


def run_experiment(limit, distill_weights: Iterable[float], output_name: str):
    config = AdapterExperimentConfig(
        output_name=output_name,
        limit_images=limit,
        distill_weights=tuple(distill_weights),
        adapter_steps=CONFIG["adapter_steps"],
        adapter_batch=CONFIG["adapter_batch"],
        chunk_words=CONFIG["chunk_words"],
        chunk_stride=CONFIG["chunk_stride"],
        chunk_threshold=CONFIG["chunk_threshold"],
        text_pooling=CONFIG["text_pooling"],
        device_preference=CONFIG.get("device", "cuda"),
        show_progress=CONFIG.get("show_progress", True),
    )
    paths = resolve_paths(CONFIG.get("env", "colab"))
    print("Using device:", config.device_preference)
    results = run_visual_genome_adapter(paths, config)
    metrics_path = results.get("metrics_path")
    print("Metrics stored at:", metrics_path)
    summary_df = pd.DataFrame(results["summary"])
    return results, summary_df


def _collect_value_columns(summary_df: pd.DataFrame):
    base_cols = ["baseline", "chunk_baseline"]
    adapter_cols = sorted(col for col in summary_df.columns if col.startswith("adapter_"))
    return [col for col in base_cols + adapter_cols if col in summary_df]


def plot_summary(summary_df: pd.DataFrame, title: str):
    value_cols = _collect_value_columns(summary_df)
    melted = summary_df.melt(
        id_vars=["metric", "split"],
        value_vars=value_cols,
        var_name="variant",
        value_name="recall",
    )
    splits = sorted(melted["split"].unique())
    fig, axes = plt.subplots(len(splits), 1, figsize=(8, 4 * len(splits)), sharex=True)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        sub = melted[melted["split"] == split]
        pivot = sub.pivot(index="metric", columns="variant", values="recall")
        for variant in pivot.columns:
            ax.plot(pivot.index, pivot[variant], marker="o", label=variant)
        ax.set_title(f"{split} split")
        ax.set_ylabel("Recall (%)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Metric")
    axes[0].legend()
    fig.suptitle(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_distribution(summary_df: pd.DataFrame, title: str):
    value_cols = _collect_value_columns(summary_df)
    plt.figure(figsize=(8, 4))
    for col in value_cols:
        plt.hist(summary_df[col], bins=10, alpha=0.4, label=col)
    plt.title(title)
    plt.xlabel("Recall (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_summary_and_distribution(summary_df: pd.DataFrame, title: str):
    """
    Combined figure: left = metric curves, right = histogram over all recalls.
    """
    value_cols = _collect_value_columns(summary_df)
    melted = summary_df.melt(
        id_vars=["metric", "split"],
        value_vars=value_cols,
        var_name="variant",
        value_name="recall",
    )
    splits = sorted(melted["split"].unique())

    fig, (ax_curves, ax_hist) = plt.subplots(1, 2, figsize=(12, 4), sharex=False)

    # Prefer test split curves if available; otherwise use all splits.
    if "test" in splits:
        curve_splits = ["test"]
    else:
        curve_splits = splits
    for split in curve_splits:
        sub = melted[melted["split"] == split]
        if sub.empty:
            continue
        pivot = sub.pivot(index="metric", columns="variant", values="recall")
        for variant in pivot.columns:
            ax_curves.plot(pivot.index, pivot[variant], marker="o", label=f"{split}:{variant}")
        ax_curves.set_xticks(range(len(pivot.index)))
        ax_curves.set_xticklabels(pivot.index, rotation=45)
    ax_curves.set_ylabel("Recall (%)")
    ax_curves.grid(alpha=0.3)
    ax_curves.legend(fontsize=8)

    for col in value_cols:
        ax_hist.hist(summary_df[col], bins=10, alpha=0.4, label=col)
    ax_hist.set_xlabel("Recall (%)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(alpha=0.3)
    ax_hist.legend(fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# %%

def plot_loss_logs(results: dict, title: str):
    loss_logs = results.get("loss_logs")
    if not loss_logs:
        print("No loss logs recorded.")
        return
    fig, (ax_loss, ax_scale) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for weight_key, entries in loss_logs.items():
        if not entries:
            continue
        df = pd.DataFrame.from_records(entries, columns=["step", "loss", "scale"])
        if df.empty:
            continue
        label = f"distill={weight_key}"
        ax_loss.plot(df["step"], df["loss"], marker="o", label=label)
        ax_scale.plot(df["step"], df["scale"], marker="o", label=label)
    ax_loss.set_title("Adapter loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3)
    ax_scale.set_title("Logit scale")
    ax_scale.set_xlabel("Step")
    ax_scale.set_ylabel("Scale")
    ax_scale.grid(alpha=0.3)
    ax_loss.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def summarize_best_metrics(summary_df: pd.DataFrame):
    if summary_df.empty:
        print("Summary is empty.")
        return
    adapter_cols = [col for col in summary_df.columns if col.startswith("adapter_")]
    if not adapter_cols:
        print("No adapter columns found; ensure the training cell ran successfully.")
        display(summary_df.head())
        return
    melted = summary_df.melt(
        id_vars=["split", "metric", "baseline", "chunk_baseline"],
        value_vars=adapter_cols,
        var_name="variant",
        value_name="adapter_score",
    )
    best_rows = melted.loc[melted.groupby(["split", "metric"])["adapter_score"].idxmax()].sort_values(["split", "metric"])
    display(best_rows[["split", "metric", "variant", "adapter_score", "baseline", "chunk_baseline"]])

def show_test_metrics(results: dict, weight: float):
    adapter_metrics = results.get("adapter_metrics", {})
    block = adapter_metrics.get(weight)
    if not block:
        print(f"No metrics found for distill={weight}.")
        return
    print(f"Test metrics for distill={weight}:")
    test_df = pd.DataFrame.from_dict(block.get("test", {}), orient="index")
    display(test_df)



# %% [markdown]
# ## 3. Smoke Test
# Run a smaller experiment to confirm everything is wired correctly. This keeps runtime manageable before committing to the full dataset.
# 

# %%
smoke_start = time.perf_counter()
smoke_results, smoke_summary = run_experiment(
    limit=CONFIG["smoke_limit"],
    distill_weights=CONFIG["smoke_distill"],
    output_name=CONFIG["smoke_output"],
)
smoke_time = time.perf_counter() - smoke_start
print("Smoke test wall time (attention):", f"{smoke_time:.1f}s")
print("Smoke test metrics head:")
print(smoke_summary.head())


# %%
summarize_best_metrics(smoke_summary.loc[smoke_summary["split"] == "test"])
plot_loss_logs(smoke_results, title="Smoke Test Adapter Dynamics")
if CONFIG["smoke_distill"]:
    show_test_metrics(smoke_results, CONFIG["smoke_distill"][-1])

# %%
plot_summary_and_distribution(smoke_summary, title="Smoke Test Recall + Distribution (attention)")


# %%
# Visualize a qualitative Visual Genome retrieval example (smoke, attention)
try:
    paths_vg = resolve_paths(CONFIG.get("env", "colab"))
    vg_data = vg_module._load_visual_genome_dataset(
        paths_vg,
        run_splits=("train",),
        limit_images=CONFIG["smoke_limit"],
    )
    vg_train = vg_data["train"]

    device_vg = torch.device(CONFIG.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model_vg, _, preprocess_vg = open_clip.create_model_and_transforms(
        vg_module.DEFAULT_MODEL_NAME,
        pretrained=vg_module.DEFAULT_PRETRAINED,
        device=device_vg,
    )
    tokenizer_vg = open_clip.get_tokenizer(vg_module.DEFAULT_MODEL_NAME)
    model_vg.eval()

    vg_embeds = vg_module._encode_chunked_embeddings(
        dataset=vg_train,
        preprocess=preprocess_vg,
        tokenizer=tokenizer_vg,
        model=model_vg,
        device=device_vg,
        batch_size=CONFIG["adapter_batch"],
        chunk_words=CONFIG["chunk_words"],
        chunk_stride=CONFIG["chunk_stride"],
        chunk_threshold=CONFIG["chunk_threshold"],
        text_pooling=CONFIG["text_pooling"],
        image_embed_cache=None,
    )

    vg_img_emb = np.asarray(vg_embeds["image_embeddings"], dtype=np.float32)
    vg_cap_emb = np.asarray(vg_embeds["caption_embeddings"], dtype=np.float32)
    vg_cap2img = np.asarray(vg_embeds["caption_to_image_index"], dtype=np.int64)

    vg_image_paths = [str(item["image_path"]) for item in vg_train]
    vg_all_captions = []
    for item in vg_train:
        vg_all_captions.extend(item["captions"])

    # Choose a query image and one of its captions
    vg_img_idx = 0
    same_img_caps = np.where(vg_cap2img == vg_img_idx)[0]
    vg_cap_idx = int(same_img_caps[0]) if same_img_caps.size > 0 else 0

    # T2I: caption -> images
    q_cap = vg_cap_emb[vg_cap_idx : vg_cap_idx + 1]
    sims_t2i = (q_cap @ vg_img_emb.T).flatten()
    top_img_idx = sims_t2i.argsort()[::-1][:5]
    top_img_scores = sims_t2i[top_img_idx]

    # I2T: image -> captions
    q_img = vg_img_emb[vg_img_idx : vg_img_idx + 1]
    sims_i2t = (q_img @ vg_cap_emb.T).flatten()
    top_cap_idx = sims_i2t.argsort()[::-1][:5]
    top_cap_scores = sims_i2t[top_cap_idx]

    fig = plt.figure(figsize=(12, 7))

    # Top row: T2I images
    for rank, (idx, score) in enumerate(zip(top_img_idx, top_img_scores), start=1):
        ax = fig.add_subplot(2, 5, rank)
        with Image.open(vg_image_paths[int(idx)]).convert("RGB") as im:
            ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"#{rank} {score:.2f}", fontsize=8)

    # Bottom-left: query image
    ax_img = fig.add_subplot(2, 5, 6)
    with Image.open(vg_image_paths[vg_img_idx]).convert("RGB") as im:
        ax_img.imshow(im)
    ax_img.axis("off")
    ax_img.set_title("query image", fontsize=8)

    # Bottom-right: I2T captions
    ax_txt = fig.add_subplot(2, 5, 7)
    ax_txt.axis("off")
    lines = [
        f"#{i + 1} ({float(top_cap_scores[i]):.2f}) {vg_all_captions[int(top_cap_idx[i])]}"
        for i in range(len(top_cap_idx))
    ]
    ax_txt.text(
        0.0,
        1.0,
        "I2T | top captions\n\n" + "\n".join(lines),
        fontsize=8,
        va="top",
        wrap=True,
    )

    caption_text = vg_all_captions[int(vg_cap_idx)]
    fig.suptitle(f"T2I | caption: {caption_text}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
except Exception as e:
    print("Skipping Visual Genome qualitative figure due to error:", repr(e))

# %%
# close chunking
CONFIG.update({
    "chunk_words": 0,
    "chunk_threshold": 0,
    "smoke_distill": (0.0,),
})

# %%
smoke_no_chunk_start = time.perf_counter()
smoke_nc_results, smoke_nc_summary = run_experiment(
    limit=CONFIG["smoke_limit"],
    distill_weights=CONFIG["smoke_distill"],
    output_name=CONFIG["smoke_output"] + "_nochunk",
)
smoke_no_chunk_time = time.perf_counter() - smoke_no_chunk_start
print("Smoke test wall time (no-chunk mean):", f"{smoke_no_chunk_time:.1f}s")
print("Smoke no-chunk metrics head:")
print(smoke_nc_summary.head())

# %%
summarize_best_metrics(smoke_nc_summary.loc[smoke_nc_summary["split"] == "test"])
plot_loss_logs(smoke_nc_results, title="Smoke Test Adapter Dynamics (no-chunk)")
if CONFIG["smoke_distill"]:
    show_test_metrics(smoke_nc_results, CONFIG["smoke_distill"][-1])

# %% [markdown]
# ## 4. Full Experiment
# Once the smoke test looks good, launch the full dataset run below. This may take significantly longer.
# 

# %%
# Full experiment 1: attention pooling (chunked)
CONFIG["chunk_words"] = 8
CONFIG["chunk_threshold"] = 15
CONFIG["text_pooling"] = "attn"

full_attn_start = time.perf_counter()
full_attn_results, full_attn_summary = run_experiment(
    limit=CONFIG["full_limit"],
    distill_weights=CONFIG["full_distill"],
    output_name=CONFIG["full_output"] + "_attn",
)
full_attn_time = time.perf_counter() - full_attn_start
print("Full run wall time (attention):", f"{full_attn_time:.1f}s")
print("Full run (attention) metrics head:")
print(full_attn_summary.head())


# %%
summarize_best_metrics(full_attn_summary.loc[full_attn_summary["split"] == "test"])
plot_loss_logs(full_attn_results, title="Full Experiment Adapter Dynamics (attention)")
if CONFIG["full_distill"]:
    show_test_metrics(full_attn_results, CONFIG["full_distill"][-1])

# %%
plot_summary_and_distribution(full_attn_summary, title="Full Experiment Recall + Distribution (attention)")


# %%
# Full experiment 2: mean pooling (chunked)
CONFIG["text_pooling"] = "mean"

full_mean_start = time.perf_counter()
full_mean_results, full_mean_summary = run_experiment(
    limit=CONFIG["full_limit"],
    distill_weights=CONFIG["full_distill"],
    output_name=CONFIG["full_output"] + "_mean",
)
full_mean_time = time.perf_counter() - full_mean_start
print("Full run wall time (mean):", f"{full_mean_time:.1f}s")
print("Full run (mean) metrics head:")
print(full_mean_summary.head())


# %%
summarize_best_metrics(full_mean_summary.loc[full_mean_summary["split"] == "test"])
plot_loss_logs(full_mean_results, title="Full Experiment Adapter Dynamics (mean)")
if CONFIG["full_distill"]:
    show_test_metrics(full_mean_results, CONFIG["full_distill"][-1])

# %%
plot_summary_and_distribution(full_mean_summary, title="Full Experiment Recall + Distribution (mean)")


# %% [markdown]
# ## 5. Compare Smoke vs Full
# The cell below merges both runs (if available) to compare adapter performance.
# 

# %%
def merge_runs(smoke_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    smoke_df = smoke_df.copy()
    smoke_df["run"] = "smoke"
    full_df = full_df.copy()
    full_df["run"] = "full"
    return pd.concat([smoke_df, full_df], ignore_index=True)

if "smoke_summary" in globals() and "full_attn_summary" in globals():
    combined = merge_runs(smoke_summary, full_attn_summary)
    display(combined.head())
    value_cols = _collect_value_columns(smoke_summary)
    for split in sorted(combined["split"].unique()):
        subset = combined[combined["split"] == split]
        fig, ax = plt.subplots(figsize=(10, 4))
        positions = list(range(len(value_cols)))
        width = 0.35
        smoke_vals = [float(subset[subset["run"] == "smoke"][col].mean()) for col in value_cols]
        full_vals = [float(subset[subset["run"] == "full"][col].mean()) for col in value_cols]
        ax.bar([p - width / 2 for p in positions], smoke_vals, width=width, label="smoke")
        ax.bar([p + width / 2 for p in positions], full_vals, width=width, label="full_attn")
        ax.set_xticks(positions)
        ax.set_xticklabels(value_cols, rotation=45)
        ax.set_ylabel("Mean Recall (%)")
        ax.set_title(f"Smoke vs Full (attention) comparison ({split} split)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        plt.tight_layout()
        plt.show()
else:
    print("Smoke and full attention summaries must be available to compare.")


# %% [markdown]
# ## 6. Final metrics and timing summary
# This cell prints a compact view of key test metrics and total runtimes.


# %%
KEY_METRICS = ["t2i@1", "t2i@5", "t2i@10", "i2t@1", "i2t@5", "i2t@10"]


def extract_test_rows(summary_df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = summary_df.copy()
    df = df[df["split"] == "test"]
    df = df[df["metric"].isin(KEY_METRICS)]
    df.insert(0, "run", label)
    return df


final_tables: list[pd.DataFrame] = []
if "full_attn_summary" in globals():
    final_tables.append(extract_test_rows(full_attn_summary, "full_attn"))
if "full_mean_summary" in globals():
    final_tables.append(extract_test_rows(full_mean_summary, "full_mean"))

if final_tables:
    final_df = pd.concat(final_tables, ignore_index=True)
    print("Key test metrics (baseline, chunk_baseline, adapters):")
    display(final_df.sort_values(["run", "metric"]))
else:
    print("No full summaries available for final metrics table.")


print("Timing summary (seconds):")
timing_rows = []
if "smoke_time" in globals():
    timing_rows.append(("smoke_attn", smoke_time))
if "smoke_no_chunk_time" in globals():
    timing_rows.append(("smoke_no_chunk_mean", smoke_no_chunk_time))
if "full_attn_time" in globals():
    timing_rows.append(("full_attn", full_attn_time))
if "full_mean_time" in globals():
    timing_rows.append(("full_mean", full_mean_time))

if timing_rows:
    for name, value in timing_rows:
        print(f"  {name}: {value:.1f}s")
else:
    print("  (no timings recorded)")


# %% [markdown]
# ## 6. Wrap Up
# - Push code changes back to GitHub once satisfied.
# - Large data stays on Drive; processed results land under `CONFIG['output_root']`.
# - Adjust the configuration cell at the top to try different adapters, chunking strategies, or distillation weights.
# 

# %%



