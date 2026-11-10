"""Generate synthetic pyramidal H&E-like slides + ASAP XML annotations for demos and tests.

    python scripts/make_synthetic_slide.py --out data/synthetic --n 4 --width 40000 --height 30000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathovision.utils.synthetic import make_synthetic_slide  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--width", type=int, default=40000)
    p.add_argument("--height", type=int, default=30000)
    p.add_argument("--mpp", type=float, default=0.243)
    p.add_argument("--tumors", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="synth")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    for i in range(a.n):
        sid = f"{a.prefix}_{i + 1:03d}"
        t0 = time.time()
        info = make_synthetic_slide(os.path.join(a.out, "images", sid + ".tif"),
                                    os.path.join(a.out, "annotations", sid + ".xml"),
                                    a.width, a.height, a.mpp, a.tumors, seed=a.seed + i)
        mb = os.path.getsize(info["path"]) / 1e6
        print(f"{sid}: {a.width}x{a.height} ({a.width * a.height / 1e9:.2f} gigapixel), "
              f"{info['n_tumor_polygons']} tumour polygons, {mb:.0f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
