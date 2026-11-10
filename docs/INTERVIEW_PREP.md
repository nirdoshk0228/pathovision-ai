# Interview prep: PathoVision AI

Answers are written so you can say them out loud. Each one points to the file where the idea lives, so you can open the code if they ask "show me".

---

## 1. The 60-second pitch

> "Pathology slides are scanned at 40x into images of around 100,000 by 200,000 pixels, which is 60 GB of raw RGB, so you can't feed them to a CNN. I built a pipeline that works on the slide's image pyramid. It detects tissue on a 2k thumbnail, tiles only the tissue into model-sized patches at a fixed physical resolution, classifies each patch with a ResNet trained on PatchCamelyon, then runs a U-Net only around suspicious regions and blends the overlapping predictions into one probability map. From that map it measures each lesion's area and largest diameter and assigns the clinical size category: ITC, micrometastasis or macrometastasis. It runs in a few hundred MB of RAM regardless of slide size, exports GeoJSON for QuPath, and has a FastAPI service plus a Streamlit viewer. On the PCam test set the classifier gets AUC ___, and on held-out CAMELYON16 slides the segmentation Dice is ___."

Fill the blanks with **your own** Kaggle numbers.

---

## 2. Gigapixel handling (the JD's "50K x 50K")

**Why not just resize the slide?**
Shrinking 100k px to 1k px discards a factor of 100 in each dimension. Tumour cells are about 10 µm, i.e. ~40 px at 40x, and they vanish. Diagnosis needs cell-level detail *and* the whole slide, so we tile.

**What is the image pyramid?** (`wsi/reader.py`)
WSI files store the same image at several resolutions (levels at downsample 1, 2, 4, 8...), each cut into small JPEG tiles. `read_region` decodes only the tiles a request overlaps. We always read from the level just finer than needed (`best_level_for_downsample`) and shrink. Reading a coarser level and upsampling would invent detail that isn't there.

**Why is resolution expressed in µm/px rather than "level 2"?**
Scanners differ: 40x on one scanner is 0.25 µm/px, on another 0.23. A model trained at 10x (≈1 µm/px) must see 10x patches at inference, or nuclei appear the wrong size. Each checkpoint stores its `mpp` and `patch_size`, and inference reads from the stored value (`save_checkpoint` in `utils/common.py`). That mismatch is one of the most common silent bugs in pathology ML.

**Memory math** (`inference/stitching.py`)
- Level-0 RGB: 100k × 200k × 3 B = 60 GB. Never materialised.
- The probability canvas is kept at 1/32 resolution: 3,125 × 6,250 float32 × 2 buffers ≈ 156 MB.
- Patches are streamed in batches through a DataLoader, so peak memory is set by batch size, not slide size.

**Tissue detection** (`wsi/tissue.py`)
Glass is white or grey and H&E tissue is coloured, so threshold HSV *saturation*. Otsu picks the threshold per slide (robust to stain and scanner), then morphological open and close plus small-object removal. 70-90% of a slide is background, so this is the biggest single speed-up. The dashboard shows the exact % skipped.

**Integral image** (`wsi/tiler.py`, DSA question bait)
To keep only patches with ≥25% tissue, I build a summed-area table of the tissue mask once (O(N)). After that, the tissue fraction of any rectangle is 4 lookups: `S[y1,x1] − S[y0,x1] − S[y1,x0] + S[y0,x0]`. Filtering 100k candidate patches is O(1) each and fully vectorised. The same trick gates the segmenter.

**Stitching / overlap blending**
CNN outputs are worst at patch borders (no context, zero padding). Patches overlap by 25%, and each prediction is weighted by a Hann window (1 at the centre, small at the edges); the final map is Σw·p / Σw. That removes seams and stops lesions from being cut at tile edges. A real bug I fixed: rounding each tile's *start* and adding a rounded *size* left 1-pixel gaps between neighbours on the low-res canvas, which split lesions in two. The fix rounds both edges from level-0 coordinates. There is a unit test for it (`test_stitch_no_gaps_with_awkward_footprint`).

---

## 3. Data and labels

**PCam**: 327,680 patches, 96×96 px at 10x, derived from CAMELYON16. Label = 1 if the **centre 32×32 px** contains tumour (the border is context). Official splits are slide-disjoint.

**CAMELYON16**: 400 WSIs of sentinel lymph nodes from breast cancer patients (2 hospitals), with polygon annotations of metastases in ASAP XML. Group `_2` polygons are *exclusions*: normal tissue inside a tumour outline. They must be subtracted, otherwise those holes count as tumour (`wsi/annotations.py`, tested).

**Data leakage: split by slide, never by patch.** Patches from the same slide share staining, scanner and patient, so a random patch split inflates metrics. `extract_patches.py` assigns whole slides to train/val/test. (The Kaggle HCD version has no slide IDs, which is why PCam's official split is preferred.)

**Class imbalance**: tumour is a small fraction of tissue. I sample positives and negatives in balance for the classifier, and use BCE + Dice loss for segmentation.

---

## 4. Models

**Why ResNet-18 instead of something bigger?**
96 px inputs and a binary task, so ResNet-18 already reaches high AUC on PCam and is ~4× faster than ResNet-50. Throughput matters when one slide has 30-100k patches. EfficientNet-B0 and DenseNet-121 are one flag away (`--arch`).

**Why does ImageNet pretraining help on histology?**
Early filters (edges, blobs, colour contrasts, textures) are generic. Pretraining gives faster convergence and better results with less data. Pathology-specific foundation models (e.g. UNI, CONCH, Virchow) are the next step up.

**U-Net in one breath** (`models/unet.py`)
An encoder downsamples to get context ("is this region tumour?"); a decoder upsamples back to full resolution; skip connections concatenate encoder features into the decoder so fine boundary detail lost in pooling comes back. I use a ResNet-34 encoder pretrained on ImageNet for the real model.

**Losses** (`training/losses.py`)
- BCE alone can be "good" by predicting all background when tumour pixels are rare.
- Soft Dice = 1 − 2|P∩G| / (|P|+|G|) optimises overlap directly and ignores the background count.
- I use 0.5·BCE + 0.5·Dice: BCE gives stable early gradients, Dice handles imbalance.
- Focal loss (available for the classifier) down-weights easy negatives.

**Augmentation** (`data/augment.py`)
- D4 dihedral group (4 rotations × flip = 8 views): tissue has no "up", so all are valid. The same 8 views are used for test-time augmentation.
- HED stain jitter: perturbs haematoxylin and eosin concentrations to simulate lab variation.
- A bug worth mentioning: forked DataLoader workers inherit the same RNG state, so every worker applies the *same* random augmentation. `worker_init_fn` reseeds each worker.

**Stain normalisation (Macenko)** (`preprocessing/stain.py`)
Convert RGB to optical density (Beer-Lambert: OD = −log(I/I₀)). Take the plane of the top-2 eigenvectors of tissue OD, find the extreme angles (the H and E stain vectors), solve for per-pixel stain concentrations, then rebuild with a reference stain matrix. Trade-off: it costs CPU per patch and can fail on patches with little tissue (handled: those pass through unchanged). Augmentation is often enough, which is why it's an optional toggle.

---

## 5. Evaluation

| Level | Metric | Why |
|---|---|---|
| Patch | ROC-AUC | threshold-independent ranking quality |
| Patch | precision / recall / F1 / specificity | at a threshold **chosen on validation**, reported on test |
| Pixel | Dice, IoU | overlap with pathologist polygons |
| Slide | AUC of max-patch score | CAMELYON16's slide classification task |
| System | patches/s, total s, peak RAM/GPU | "real-time, efficient" in the JD |

**Why not accuracy?** With imbalance, accuracy hides missed tumours. In screening, **recall** (missed metastasis = false negative) is the costly error, so I'd tune the threshold for high sensitivity and let the pathologist review the heatmap.

**CAMELYON16's lesion metric** is FROC: sensitivity at 0.25-8 false positives per slide. I didn't implement it; it's on the roadmap. Say so honestly if asked.

**Clinical quantification** (`inference/quantify.py`)
Largest lesion diameter → category: ITC ≤ 0.2 mm, micrometastasis 0.2-2 mm, macrometastasis > 2 mm (the convention used in CAMELYON17 / TNM staging). Diameter = maximum Feret diameter. The farthest pair of points lies on the convex hull, so compute the hull in O(n log n), then compare hull vertices only. Rotating calipers would make that last step O(h).

---

## 6. Speed: "how would you make it real-time?"

Already done:
1. Tissue masking skips 70-90% of patches.
2. Segmentation is *gated*: the U-Net runs only near patches the classifier flagged (cascade: cheap model everywhere, expensive model where needed).
3. Batched GPU inference with AMP (fp16) and `channels_last`.
4. Parallel tile decoding in DataLoader workers (each worker opens its own OpenSlide handle; they can't be shared across processes).

Next steps I would take:
- Export to ONNX / TensorRT, INT8 quantisation for the classifier.
- Run the classifier at lower magnification (5x) first and zoom in only where needed, as a pathologist does.
- Fully convolutional inference on large regions instead of many small patches (shared computation in overlaps).
- Move tile decode + resize to C++ (libvips/OpenSlide C API, pybind11) if decoding becomes the bottleneck: `scripts/benchmark.py` shows whether you're I/O- or GPU-bound.

---

## 7. Likely questions and short answers

- **What happens with pen marks, folds, blur?** Dark pixels are excluded in tissue detection; folds and blur remain failure modes. A blur detector (variance of Laplacian per patch) is cheap to add.
- **How would you get a slide-level label without pixel annotations?** Multiple-instance learning: a slide is a bag of patches labelled only at bag level; attention-based MIL (ABMIL / CLAM) learns which patches matter. This is the next feature I'd add.
- **Why max-probability as the slide score?** A single metastasis makes a node positive, so max is the right inductive bias. It's noisy, so top-k mean is reported too.
- **How do you choose patch size?** Field of view vs. detail: 96 px at 10x ≈ 93 µm sees ~10-15 cells plus context. Larger fields help segmentation boundaries (256 px).
- **Overfitting signs and fixes?** Train loss falling while val AUC flattens. Fixes: early stopping on val AUC (implemented), stronger stain augmentation, weight decay, more slides (diversity matters more than patch count).
- **Calibration?** Max-prob scores are often overconfident; temperature scaling on validation fixes that without changing AUC.
- **Deploying in a lab?** Async jobs (`POST /analyze` → job id → poll), because a slide takes seconds to minutes. Slides stay on shared storage and are referenced by name, never uploaded over HTTP. Results go out as GeoJSON for QuPath review. An audit trail via the model metadata in every report.

---

## 8. DSA crossover (they listed "strong DSA")

- Summed-area table / 2-D prefix sums (tile filtering): LeetCode 304 "Range Sum Query 2D".
- Connected components (lesions): BFS/DFS or union-find. Merging lesions across tile boundaries is a union-find problem.
- Convex hull (Graham scan / Andrew's monotone chain) + rotating calipers (Feret diameter).
- Flood fill (hole filling in the tissue mask, `fill_holes`).
- Sliding-window / overlap-add (stitching).

Practise those five; they map one-to-one onto code you wrote.

---

## 9. Honest limitations (say these before they find them)

- Trained on 2 hospitals' data (CAMELYON16); real deployment needs multi-centre validation.
- Segmentation is trained on a handful of slides on a free GPU, so Dice is lower than state of the art.
- No FROC / lesion-level metric yet; no MIL slide classifier yet.
- Research prototype, not a medical device.
