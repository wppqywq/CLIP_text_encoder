# CLIP encoder NLU (COMP 545 final)

This repo runs open-CLIP retrieval baselines.

- Dataset: `Flickr30k Entities`: each noun phrase (e.g., "the red car", "boy on the left") is linked to entity boxes. We can probe whether models distinguish phrases like "red car" vs "blue car" without re-authoring captions. 

    (COCO/LAION captions: sentence-level only; phrase grounding requires extra work.)
    
    Update: the captions in this dataset are too small might result in chunch-adapter behavior poorly.

- model: `ViT-B-32` (small, fast). If resources allow, try `ViT-B-16` later.
- weights: `laion2b_s34b_b79k` (OpenCLIP SOTA, great B-32 checkpoint).

---
### File tree
```
final/
  ├─ openclip_{0/1}.ipynb
  ├─ utils.py
  ├─ adapter.py  
  ├─ output/
  └─ data/            #  (git-ignored)
      ├─ flickr30k/flickr30k-images/          # putFlickr30k images here, expect 31,783 jpg
      └─ flickr30k/flickr30k_entities-master/ # put the github repo here, and unzip annotations.zip
            ├─ annotations/Annotations/*.xml
            ├─ annotations/Sentences/*.txt
            ├─ train.txt  val.txt  test.txt
            ...
```

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




