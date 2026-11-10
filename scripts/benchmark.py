"""Throughput benchmark: how batch size and DataLoader workers affect whole-slide inference speed.

    python scripts/benchmark.py --slide data/slides/tumor_001.tif --cls-ckpt models/classifier.pt \
        --batch-sizes 32 128 256 --workers 0 2 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathovision.inference.slide_inference import AnalysisConfig, SlideAnalyzer, load_classifier  # noqa: E402
from pathovision.utils.common import get_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slide", required=True)
    p.add_argument("--cls-ckpt", required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 256])
    p.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device = get_device(a.device)
    cls = load_classifier(a.cls_ckpt, device)
    print(f"device: {device}")
    print(f"{'batch':>6} {'workers':>8} {'tiles':>8} {'cls_sec':>8} {'tiles/s':>9} {'total_s':>8} {'peak_RSS_MB':>12}")
    for w in a.workers:
        for b in a.batch_sizes:
            cfg = AnalysisConfig(batch_size=b, num_workers=w, device=str(device))
            rep = SlideAnalyzer(cfg, classifier=cls).analyze(a.slide)
            bm = rep["benchmark"]
            print(f"{b:>6} {w:>8} {bm['tiles_classified']:>8} {bm['stage_seconds']['classification']:>8.2f} "
                  f"{bm['classification_tiles_per_sec']:>9.1f} {bm['total_seconds']:>8.2f} {bm['peak_rss_mb']:>12.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
