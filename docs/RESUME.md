# Resume entry

Replace every `[ ]` with a number from **your own** Kaggle run (`runs/*/metrics.json`, `outputs/holdout/*/report.json`, `scripts/benchmark.py`). Never put a number on the resume that you didn't measure.

---

**PathoVision AI: Gigapixel Whole-Slide Image Analysis for Cancer Metastasis Detection**
*Python, PyTorch, OpenSlide, OpenCV, FastAPI, Streamlit* | [github.com/nirdoshk0228/pathovision-ai](https://github.com/nirdoshk0228/pathovision-ai)

- Built an end-to-end pipeline for **gigapixel (~100K × 200K px) H&E slides** using pyramid-level, resolution-aware (µm/px) patch streaming; analyses a full CAMELYON16 slide in **[ ] s** with **[ ] MB** peak RAM vs. [ ] GB for the raw image.
- Trained a **ResNet-18 patch classifier** on PatchCamelyon (262K patches) with HED stain + D4 augmentation and 8-view TTA: **test ROC-AUC [ ]**, recall [ ], F1 [ ] (threshold selected on validation).
- Developed a **ResNet34-U-Net** for pixel-level tumour segmentation (BCE + Dice loss), with Hann-weighted **overlap stitching** for seamless slide-level masks; **Dice [ ]** on held-out slides.
- Cut inference work by **[ ]%** with Otsu tissue detection, O(1) summed-area-table tile filtering, and classifier-gated segmentation; **[ ] patches/s** on a T4 GPU.
- Automated **lesion quantification** (area, max Feret diameter, ITC / micro- / macrometastasis category) and exported predictions as **GeoJSON for QuPath**; served via async **FastAPI** jobs and an interactive **Streamlit** viewer.

---

### Shorter 3-bullet version (if space is tight)

- Built a gigapixel whole-slide pathology pipeline (OpenSlide pyramid streaming, tissue detection, overlap-blended stitching) that analyses a 100K × 200K px slide in [ ] s using [ ] MB RAM.
- ResNet-18 metastasis classifier (PCam test AUC [ ]) + ResNet34-U-Net segmenter (held-out Dice [ ]); stain augmentation, TTA, slide-disjoint splits.
- Lesion quantification (area, Feret diameter, clinical size category), QuPath GeoJSON export, FastAPI job service and Streamlit viewer.

### Skills line additions
PyTorch · CNNs (ResNet, EfficientNet) · U-Net · OpenCV · OpenSlide · Digital pathology / WSI · Image segmentation · FastAPI · Streamlit
