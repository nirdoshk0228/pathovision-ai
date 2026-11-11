"""Evaluate whole-slide predictions on a set of annotated slides.

Reports per-slide Dice/IoU against the ASAP annotation (at the analysis map
resolution) and slide-level ROC-AUC of the max-patch-probability score
(tumour slide vs normal slide), the CAMELYON16 slide-classification task.

    python scripts/evaluate_slides.py --slides "data/camelyon16/test/*.tif" \
        --xml-dir data/camelyon16/annotations --cls-ckpt models/classifier.pt \
        --seg-ckpt models/segmenter.pt --out outputs/eval
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathovision.evaluation.metrics import classification_metrics  # noqa: E402
from pathovision.inference.slide_inference import AnalysisConfig, SlideAnalyzer  # noqa: E402
from pathovision.utils.common import write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slides", required=True)
    p.add_argument("--xml-dir", required=True)
    p.add_argument("--cls-ckpt", required=True)
    p.add_argument("--seg-ckpt", default=None)
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--seg-gate", type=float, default=0.5)
    p.add_argument("--no-agreement", action="store_true")
    a = p.parse_args()
    an = SlideAnalyzer(AnalysisConfig(classifier_ckpt=a.cls_ckpt, segmenter_ckpt=a.seg_ckpt, num_workers=a.workers,
                                      tta=a.tta, seg_gate=a.seg_gate, require_agreement=not a.no_agreement))
    rows = []
    for sp in sorted(glob.glob(a.slides)):
        sid = os.path.splitext(os.path.basename(sp))[0]
        xml = os.path.join(a.xml_dir, sid + ".xml")
        rep = an.analyze(sp, os.path.join(a.out, sid), xml if os.path.exists(xml) else None)
        rep.pop("_artefacts")
        gt = rep.get("ground_truth", {})
        rows.append({"slide": sid, "label": int(os.path.exists(xml)), "score": rep["slide_score"]["max_tile_prob"],
                     "dice": gt.get("dice", np.nan), "iou": gt.get("iou", np.nan),
                     "pred_area_mm2": rep["quantification"]["tumor_area_mm2"],
                     "gt_lesions": gt.get("gt_lesions", 0), "detected_lesions": gt.get("detected_lesions", 0),
                     "fp_lesions": gt.get("fp_lesions", rep["quantification"]["n_lesions"]),
                     "gt_area_mm2": gt.get("gt_tumor_area_mm2", 0.0),
                     "seconds": rep["benchmark"]["total_seconds"],
                     "tiles_per_sec": rep["benchmark"]["classification_tiles_per_sec"]})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_slide.csv"), index=False)
    summary = {"n_slides": len(df), "mean_dice_tumor_slides": float(df.loc[df.label == 1, "dice"].mean()),
               "mean_iou_tumor_slides": float(df.loc[df.label == 1, "iou"].mean()),
               "mean_seconds_per_slide": float(df.seconds.mean()),
               "lesion_sensitivity": float(df.detected_lesions.sum() / max(df.gt_lesions.sum(), 1)),
               "detected_lesions": int(df.detected_lesions.sum()), "gt_lesions": int(df.gt_lesions.sum()),
               "fp_lesions_per_slide": float(df.fp_lesions.mean()),
               "fp_lesions_per_normal_slide": float(df.loc[df.label == 0, "fp_lesions"].mean()) if (df.label == 0).any() else None}
    if df.label.nunique() == 2:
        summary["slide_level"] = classification_metrics(df.label.values, df.score.values)
    write_json(summary, os.path.join(a.out, "summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
