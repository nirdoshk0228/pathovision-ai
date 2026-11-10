"""One-command CPU demo of the full pipeline on synthetic slides (no downloads, no GPU).

1. Generate synthetic pyramidal slides with tumour annotations.
2. Extract classification (PCam-format) and segmentation patches, split by slide.
3. Train a small classifier and U-Net from scratch (a few minutes on CPU).
4. Analyse a held-out ~1.2-gigapixel slide and write all reports.

    python scripts/demo_synthetic.py            # full demo
    python scripts/demo_synthetic.py --quick    # smaller slides, fewer epochs

The resulting numbers only prove that the pipeline works end to end. Report
results from PCam/CAMELYON16 training (see README), never from this demo.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from pathovision.utils.synthetic import make_synthetic_slide  # noqa: E402


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data", default=os.path.join(ROOT, "data", "synthetic"))
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    t0 = time.time()
    img_dir, xml_dir = os.path.join(a.data, "images"), os.path.join(a.data, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)

    train_wh = (12000, 9000) if a.quick else (20000, 15000)
    test_wh = (16000, 12000) if a.quick else (40000, 30000)
    step("1. synthetic slides")
    specs = [(f"synth_{i:03d}", train_wh, i) for i in range(1, 5)] + [("synth_test", test_wh, 99)]
    for sid, (w, h), seed in specs:
        path = os.path.join(img_dir, sid + ".tif")
        if not os.path.exists(path):
            make_synthetic_slide(path, os.path.join(xml_dir, sid + ".xml"), w, h, n_tumors=5, seed=seed, verbose=False)
        print(f"  {sid}: {w} x {h} = {w * h / 1e9:.2f} gigapixel", flush=True)

    step("2. patch extraction (split by slide)")
    import extract_patches

    patches = os.path.join(a.data, "patches")
    if os.path.exists(patches):
        shutil.rmtree(patches)
    # 'synth_test' is held out completely; synth_004 is the validation slide
    os.rename(os.path.join(img_dir, "synth_test.tif"), os.path.join(a.data, "synth_test.tif"))
    try:
        extract_patches.main(["--slides", os.path.join(img_dir, "synth_*.tif"), "--xml-dir", xml_dir,
                              "--out", patches, "--val-slides", "synth_004", "--test-slides", "synth_003",
                              "--max-per-slide", "800" if a.quick else "1600",
                              "--seg-per-slide", "120" if a.quick else "250",
                              "--seg-size", "128", "--seg-mpp", "1.944"])
    finally:
        os.rename(os.path.join(a.data, "synth_test.tif"), os.path.join(img_dir, "synth_test.tif"))

    step("3a. train classifier (ResNet-18, from scratch, CPU)")
    from pathovision.training import train_classifier, train_segmentation

    train_classifier.main(["--data-dir", patches, "--arch", "resnet18", "--no-pretrained",
                           "--epochs", "2" if a.quick else "4", "--batch-size", "64", "--lr", "2e-3",
                           "--workers", str(a.workers), "--tta", "1", "--out-dir", os.path.join(ROOT, "runs", "demo_cls")])
    step("3b. train segmenter (U-Net, from scratch, CPU)")
    train_segmentation.main(["--data-dir", os.path.join(patches, "seg"), "--arch", "unet", "--no-pretrained",
                             "--epochs", "3" if a.quick else "8", "--batch-size", "16", "--lr", "2e-3",
                             "--workers", str(a.workers), "--out-dir", os.path.join(ROOT, "runs", "demo_seg")])
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "runs", "demo_cls", "best.pt"), os.path.join(ROOT, "models", "classifier.pt"))
    shutil.copy(os.path.join(ROOT, "runs", "demo_seg", "best.pt"), os.path.join(ROOT, "models", "segmenter.pt"))
    slides_dir = os.path.join(ROOT, "data", "slides")
    os.makedirs(slides_dir, exist_ok=True)
    for ext in (".tif", ".xml"):
        src = os.path.join(img_dir if ext == ".tif" else xml_dir, "synth_test" + ext)
        dst = os.path.join(slides_dir, "synth_test" + ext)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    step("4. whole-slide analysis of the held-out slide")
    from pathovision.inference import slide_inference

    slide_inference.main(["--slide", os.path.join(slides_dir, "synth_test.tif"),
                          "--cls-ckpt", os.path.join(ROOT, "models", "classifier.pt"),
                          "--seg-ckpt", os.path.join(ROOT, "models", "segmenter.pt"),
                          "--gt-xml", os.path.join(slides_dir, "synth_test.xml"),
                          "--out", os.path.join(ROOT, "outputs", "synth_test"), "--workers", str(a.workers)])
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in outputs/synth_test; "
          f"launch the dashboard with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
