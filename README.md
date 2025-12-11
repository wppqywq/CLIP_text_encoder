# CLIP encoder NLU (COMP 545 final)

This repo contains code for our COMP 545 final project on fine-grained retrieval with CLIP-style encoders. Some of experiments were run in Google Colab, and the corresponding results and hyperparameters are also stored in this repository.

- Dataset: `Flickr30k Entities`: phrase-to-box annotations on Flickr30k, used for initial experiments and qualitative examples. (legacy)
- Dataset: `Visual Genome`: region-level descriptions paired with images; we process a 5k-image subset and use it as the main benchmark for adapter experiments.

- Model: `ViT-B-32` from OpenCLIP.
- Weights: `laion2b_s34b_b79k`.

---
### Install deps

Python 3.10+ with:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu/mps per env
pip install open_clip_torch pillow numpy matplotlib pandas
```

---
### File tree
```
final/
  ├─ src/
  │   ├─ config/runtime.py          # shared path presets (local vs Colab)
  │   ├─ data/visual_genome.py      # download/process helpers
  │   └─ training/vg_adapter.py     # Visual Genome adapter pipeline
  ├─ scripts/
  │   ├─ prepare_visual_genome.py   # CLI wrapper for downloading + processing
  │   └─ run_vg_adapter.py          # CLI wrapper for training + evaluation
  ├─ notebooks/
  │   └─ vg_adapter_colab.ipynb     # Colab-friendly VG adapter notebook
  ├─ output/                        # generated metrics, plots, etc. (git-ignored)
  └─ data/                          # datasets live here (git-ignored)
      ├─ flickr30k/
      └─ visual_genome/
```

---
### Runtime paths

- `src/config/runtime.py` defines two presets: `local` (default) and `colab`.
- For Colab runs, mount Drive and optionally override the defaults by exporting:
  - `VG_COLAB_REPO_ROOT` – directory containing this repo copy
  - `VG_COLAB_DATA_ROOT` – location for datasets (defaults to `Drive/.../data`)
  - `VG_COLAB_OUTPUT_ROOT` – where metrics and logs are written (defaults to `Drive/.../output`)

---
**Main pipeline**: use `notebooks/vg_adapter_colab.ipynb` or `notebooks/tmp.py` on Colab for a guided run (all parameters reside in a single configuration block).

---
### Chunking modes

`scripts/run_vg_adapter.py` exposes `--chunk-mode` so experiments can switch between the legacy fixed-size sliding window (`fixed`) and entity-driven segmentation (`entity`). The latter consumes optional `caption_entities` metadata (e.g., Flickr30k Entities phrase lists) and falls back to the fixed strategy whenever entity spans are unavailable.

---
### Data sources
- Flickr30k images: apply for access at `https://shannon.cs.illinois.edu/DenotationGraph/`.
- Flickr30k Entities annotations and splits: `https://github.com/BryanPlummer/flickr30k_entities`.
- Visual Genome: `https://homes.cs.washington.edu/~ranjay/visualgenome/api.html`





