"""Extract training patches from annotated whole-slide images.

Produces, from the same slides:

* classification patches in **PCam HDF5 format** (``train_x.h5`` / ``train_y.h5`` ...),
  labelled with the PCam rule: positive if the centre 32x32 region contains tumour;
* segmentation patch/mask PNG pairs (``seg/train/images``, ``seg/train/masks`` ...).

Slides are split into train/valid/test **by slide**, never by patch.

    python scripts/extract_patches.py --slides "data/camelyon16/images/tumor_*.tif" \
        --xml-dir data/camelyon16/annotations --out data/patches \
        --val-slides tumor_009 --test-slides tumor_010 --max-per-slide 3000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathovision.wsi.annotations import load_asap_xml, rasterize  # noqa: E402
from pathovision.wsi.reader import WSIReader  # noqa: E402
from pathovision.wsi.tiler import IntegralImage, TileGrid  # noqa: E402
from pathovision.wsi.tissue import tissue_mask  # noqa: E402


def slide_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def assign_splits(ids, val, test, val_frac, test_frac, seed):
    ids = sorted(ids)
    split = {}
    if val or test:
        for i in ids:
            split[i] = "valid" if i in val else "test" if i in test else "train"
        return split
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(ids))
    n_te = max(1, int(round(test_frac * len(ids)))) if len(ids) >= 3 else 0
    n_va = max(1, int(round(val_frac * len(ids)))) if len(ids) >= 2 else 0
    for k, i in enumerate(order):
        split[i] = "test" if k < n_te else "valid" if k < n_te + n_va else "train"
    return split


def candidates(reader, polys, size, mpp, thumb, thumb_ds, tissue, min_tissue, overlap=0.0):
    """Grid tiles over tissue + their coarse tumour fraction (from a thumbnail-resolution tumour mask)."""
    W, H = reader.dimensions
    fp = reader.footprint(size, mpp)
    grid = TileGrid((W, H), fp, overlap, tissue, thumb_ds, min_tissue)
    c = grid.coords()
    tum_lr = rasterize(polys, tissue.shape, thumb_ds) if polys else np.zeros(tissue.shape, bool)
    ii = IntegralImage(tum_lr)
    x0 = np.floor(c[:, 0] / thumb_ds).astype(np.int64)
    y0 = np.floor(c[:, 1] / thumb_ds).astype(np.int64)
    x1 = np.ceil((c[:, 0] + fp) / thumb_ds).astype(np.int64)
    y1 = np.ceil((c[:, 1] + fp) / thumb_ds).astype(np.int64)
    return c, ii.rect_mean(x0, y0, x1, y1), fp


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slides", required=True, help="glob of WSI files")
    p.add_argument("--xml-dir", default=None, help="folder of ASAP XMLs named <slide_id>.xml (missing = normal slide)")
    p.add_argument("--out", required=True)
    p.add_argument("--cls-size", type=int, default=96)
    p.add_argument("--cls-mpp", type=float, default=0.972)
    p.add_argument("--seg-size", type=int, default=256)
    p.add_argument("--seg-mpp", type=float, default=0.972)
    p.add_argument("--max-per-slide", type=int, default=2000, help="classification patches per slide")
    p.add_argument("--seg-per-slide", type=int, default=400)
    p.add_argument("--seg-per-normal-slide", type=int, default=None,
                   help="segmentation patches from slides without annotation (default: same as --seg-per-slide)")
    p.add_argument("--seg-neg-frac", type=float, default=0.3, help="share of tumour-free segmentation patches")
    p.add_argument("--no-cls", action="store_true")
    p.add_argument("--no-seg", action="store_true")
    p.add_argument("--val-slides", nargs="*", default=[])
    p.add_argument("--test-slides", nargs="*", default=[])
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--mpp", type=float, default=None, help="override level-0 mpp")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    import h5py

    paths = sorted(glob.glob(a.slides))
    if not paths:
        sys.exit(f"no slides match {a.slides}")
    rng = np.random.default_rng(a.seed)
    splits = assign_splits([slide_id(s) for s in paths], set(a.val_slides), set(a.test_slides),
                           a.val_frac, a.test_frac, a.seed)
    os.makedirs(a.out, exist_ok=True)
    cls_buf = {k: ([], []) for k in ("train", "valid", "test")}
    stats = {}

    for sp in paths:
        sid = slide_id(sp)
        split = splits[sid]
        xml = os.path.join(a.xml_dir, sid + ".xml") if a.xml_dir else None
        polys = load_asap_xml(xml) if xml and os.path.exists(xml) else []
        reader = WSIReader(sp, mpp_override=a.mpp)
        thumb, tds = reader.thumbnail(2048)
        tissue = tissue_mask(thumb)
        st = {"split": split, "has_annotation": bool(polys)}

        # ---- classification patches (PCam rule)
        if not a.no_cls:
            c, frac, fp = candidates(reader, polys, a.cls_size, a.cls_mpp, thumb, tds, tissue, 0.25)
            pos_idx, neg_idx = np.nonzero(frac > 0)[0], np.nonzero(frac == 0)[0]
            n_half = a.max_per_slide // 2
            # coarse positives (tumour anywhere in the tile); the exact centre-region label is checked below.
            # Labelling is a cheap polygon rasterisation, so every candidate can be tried.
            pos_pick = rng.permutation(pos_idx)
            neg_pick = rng.permutation(neg_idx)[: max(n_half, a.max_per_slide - min(len(pos_pick), n_half))]
            xs, ys = cls_buf[split]
            n_pos = n_neg = 0
            ds = a.cls_mpp / reader.mpp
            for j in np.concatenate([pos_pick, neg_pick]):
                x, y = int(c[j, 0]), int(c[j, 1])
                m = rasterize(polys, (a.cls_size, a.cls_size), ds, (x, y)) if polys else None
                lo, hi = a.cls_size // 2 - 16, a.cls_size // 2 + 16
                label = int(m is not None and m[lo:hi, lo:hi].any())
                if label and n_pos >= n_half:
                    continue
                if not label and n_neg >= a.max_per_slide - n_half:
                    continue
                xs.append(reader.read_at_mpp(x, y, a.cls_size, a.cls_mpp))
                ys.append(label)
                n_pos += label
                n_neg += 1 - label
            st.update({"cls_pos": n_pos, "cls_neg": n_neg})

        # ---- segmentation patches
        if not a.no_seg:
            c, frac, fp = candidates(reader, polys, a.seg_size, a.seg_mpp, thumb, tds, tissue, 0.1, overlap=0.5)
            pos_idx, neg_idx = np.nonzero(frac > 0)[0], np.nonzero(frac == 0)[0]
            per = a.seg_per_slide if polys or a.seg_per_normal_slide is None else a.seg_per_normal_slide
            n_neg = int(round(per * (a.seg_neg_frac if len(pos_idx) else 1.0)))
            n_pos = per - n_neg
            pick = np.concatenate([rng.permutation(pos_idx)[:n_pos], rng.permutation(neg_idx)[:n_neg]])
            d_img = os.path.join(a.out, "seg", split, "images")
            d_msk = os.path.join(a.out, "seg", split, "masks")
            os.makedirs(d_img, exist_ok=True)
            os.makedirs(d_msk, exist_ok=True)
            ds = a.seg_mpp / reader.mpp
            tum_px = 0
            for j in pick:
                x, y = int(c[j, 0]), int(c[j, 1])
                img = reader.read_at_mpp(x, y, a.seg_size, a.seg_mpp)
                m = rasterize(polys, (a.seg_size, a.seg_size), ds, (x, y)) if polys else np.zeros((a.seg_size,) * 2, bool)
                name = f"{sid}_{x}_{y}.png"
                cv2.imwrite(os.path.join(d_img, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(d_msk, name), (m * 255).astype(np.uint8))
                tum_px += int(m.sum())
            st.update({"seg_patches": int(len(pick)), "seg_tumor_pixel_frac": tum_px / max(1, len(pick) * a.seg_size ** 2)})
        reader.close()
        stats[sid] = st
        print(sid, st, flush=True)

    if not a.no_cls:
        for split, (xs, ys) in cls_buf.items():
            if not xs:
                continue
            order = rng.permutation(len(xs))
            X = np.stack(xs)[order]
            Y = np.array(ys, np.uint8)[order].reshape(-1, 1, 1, 1)
            with h5py.File(os.path.join(a.out, f"{split}_x.h5"), "w") as f:
                f.create_dataset("x", data=X, chunks=(min(64, len(X)), a.cls_size, a.cls_size, 3), compression="lzf")
            with h5py.File(os.path.join(a.out, f"{split}_y.h5"), "w") as f:
                f.create_dataset("y", data=Y)
            print(f"{split}: {len(X)} classification patches, {Y.mean() * 100:.1f}% positive")
    with open(os.path.join(a.out, "extraction.json"), "w") as f:
        json.dump({"args": vars(a), "cls_patch_size": a.cls_size, "cls_mpp": a.cls_mpp,
                   "seg_patch_size": a.seg_size, "seg_mpp": a.seg_mpp, "slides": stats}, f, indent=2)
    # the segmentation trainer looks for extraction.json next to train/ and val/
    seg_root = os.path.join(a.out, "seg")
    if os.path.isdir(seg_root):
        with open(os.path.join(seg_root, "extraction.json"), "w") as f:
            json.dump({"seg_patch_size": a.seg_size, "seg_mpp": a.seg_mpp}, f)


if __name__ == "__main__":
    main()
