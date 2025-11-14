# CLIP encoder NLU (COMP 545 final)

This repo runs open-CLIP retrieval baselines.

- Dataset: `Flickr30k Entities`: each noun phrase (e.g., "the red car", "boy on the left") is linked to entity boxes. We can probe whether models distinguish phrases like "red car" vs "blue car" without re-authoring captions.
- Dataset: `Visual Genome`: region-level descriptions paired with the original Flickr images, used for the adapter experiments.

    (COCO/LAION captions: sentence-level only; phrase grounding requires extra work.)
    
    Update: the captions in this dataset are too small might result in chunch-adapter behavior poorly.

- model: `ViT-B-32` (small, fast). If resources allow, try `ViT-B-16` later.
- weights: `laion2b_s34b_b79k` (OpenCLIP SOTA, great B-32 checkpoint).

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
  ├─ openclip_{0/1}.ipynb
  ├─ utils.py
  ├─ adapter.py
  ├─ tmp.py                         # Flickr30k baseline notebook-style script
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
- The thin wrappers `tmp_new.py` and `download_newdata.py` expose an `ENV` toggle near the top; flip between `"local"` and `"colab"` as needed.

---
Visual Genome workflow
1. **Configure paths**: leave `DEFAULT_ENV = "local"` for GitHub runs. When using Colab, comment that line and uncomment the Colab variant in the runner scripts or wrappers.
2. **Prepare data** (local example):
   ```
   python scripts/prepare_visual_genome.py --env local
   ```
   On Colab, mount Drive, copy the repo, set `--env colab`, and re-run. Archives download into `data/visual_genome_raw`, processed JSON lives in `data/visual_genome`. Use `--no-images` if the image archives already exist.
3. **Train the adapter**:
   ```
   python scripts/run_vg_adapter.py --env local --output-name vg_baseline
   ```
   Results land under `output/vg_baseline/metrics.json`. All plots or logs should be written inside a subfolder of `output/`.
4. **Notebook parity**: use `notebooks/vg_adapter_colab.ipynb` on Colab for a guided run (all parameters reside at the top of each code cell).

---
Data sources
- Flickr30k images: apply for access at `https://shannon.cs.illinois.edu/DenotationGraph/`.
- Flickr30k Entities annotations and splits: GitHub `https://github.com/BryanPlummer/flickr30k_entities`.

Download and put them as the file structure tree before runing the notebook, and Unzip the `flickr30k_entities-master/annotations`. 

---
Install deps

I use Python 3.10+ with:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu/mps per env
pip install open_clip_torch pillow numpy matplotlib pandas
```




