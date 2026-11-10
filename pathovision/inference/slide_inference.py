"""End-to-end whole-slide analysis.

Pipeline (each stage only touches what the previous one selected):

1. Thumbnail (~2k px) from the lowest pyramid level -> tissue mask.
2. Classifier grid: tissue patches at the classifier's resolution (default 96 px
   @ 0.972 um/px = PCam 10x) -> tumour probability per patch -> heatmap.
3. Segmenter (optional): overlapping patches only around suspicious regions
   (classifier prob >= gate) -> pixel probabilities blended into one map.
4. Quantification: lesion areas, diameters, tumour burden, size category.
5. Exports: overlays, masks, CSV/JSON reports, GeoJSON (QuPath) and ASAP XML.

Run from the command line::

    python -m pathovision.inference.slide_inference --slide data/slides/tumor_001.tif \
        --cls-ckpt runs/cls/best.pt --seg-ckpt runs/seg/best.pt \
        --gt-xml data/annotations/tumor_001.xml --out outputs/tumor_001
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from pathovision.data.datasets import WSITileDataset
from pathovision.evaluation.metrics import mask_dice_iou
from pathovision.inference.quantify import quantify
from pathovision.inference.stitching import StitchAccumulator
from pathovision.inference.viz import add_colorbar, draw_mask_contours, overlay_heatmap, overlay_mask, resize_to
from pathovision.models.classifier import PatchClassifier
from pathovision.models.unet import build_segmenter
from pathovision.preprocessing.stain import MacenkoNormalizer
from pathovision.training.engine import tta_logits, tta_seg_logits
from pathovision.utils.common import Timer, get_device, load_checkpoint, peak_rss_mb, write_json
from pathovision.wsi.annotations import load_asap_xml, mask_to_polygons, rasterize, save_asap_xml, save_geojson
from pathovision.wsi.reader import WSIReader
from pathovision.wsi.tiler import IntegralImage, TileGrid
from pathovision.wsi.tissue import tissue_mask

ProgressFn = Callable[[str, float], None]


@dataclass
class AnalysisConfig:
    classifier_ckpt: Optional[str] = None
    segmenter_ckpt: Optional[str] = None
    thumb_max_side: int = 2048
    map_max_side: int = 4096
    min_tissue: float = 0.25
    cls_overlap: float = 0.0
    batch_size: int = 128           # classifier patches per batch
    seg_batch_size: int = 16        # U-Net activations are ~50x larger per patch than the classifier's
    num_workers: int = 2
    stain_norm: bool = False
    tta: int = 1
    cls_threshold: Optional[float] = None  # None -> value stored in checkpoint (else 0.5)
    seg_threshold: float = 0.5
    seg_overlap: float = 0.25
    seg_gate: float = 0.3
    min_lesion_um2: float = 200.0   # smaller connected components are treated as noise
    device: str = "auto"
    mpp_override: Optional[float] = None


@dataclass
class LoadedModel:
    model: torch.nn.Module
    meta: Dict = field(default_factory=dict)

    @property
    def patch_size(self) -> int:
        return int(self.meta.get("patch_size", 96))

    @property
    def mpp(self) -> float:
        return float(self.meta.get("mpp", 0.972))


def load_classifier(path: str, device) -> LoadedModel:
    sd, meta = load_checkpoint(path)
    m = PatchClassifier(meta.get("arch", "resnet18"), pretrained=False)
    m.load_state_dict(sd)
    return LoadedModel(m.to(device).eval(), meta)


def load_segmenter(path: str, device) -> LoadedModel:
    sd, meta = load_checkpoint(path)
    m = build_segmenter(meta.get("arch", "resnet34_unet"), pretrained=False)
    m.load_state_dict(sd)
    return LoadedModel(m.to(device).eval(), meta)


class SlideAnalyzer:
    def __init__(self, cfg: AnalysisConfig, classifier: Optional[LoadedModel] = None,
                 segmenter: Optional[LoadedModel] = None):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.classifier = classifier or (load_classifier(cfg.classifier_ckpt, self.device) if cfg.classifier_ckpt else None)
        self.segmenter = segmenter or (load_segmenter(cfg.segmenter_ckpt, self.device) if cfg.segmenter_ckpt else None)
        if self.classifier is None:
            raise ValueError("a classifier checkpoint is required")
        self.normalizer = MacenkoNormalizer() if cfg.stain_norm else None

    # ----------------------------------------------------------------- helpers
    def _loader(self, slide_path: str, coords: np.ndarray, size: int, mpp: float, batch_size: int) -> DataLoader:
        ds = WSITileDataset(slide_path, coords, size, mpp, self.normalizer, self.cfg.mpp_override)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=self.cfg.num_workers,
                          pin_memory=self.device.type == "cuda")

    @torch.no_grad()
    def _run(self, model, loader, seg: bool, progress: Optional[ProgressFn], stage: str):
        use_amp = self.device.type == "cuda"
        n_batches = max(1, len(loader))
        for b, (x, idx) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                if seg:
                    logits = tta_seg_logits(model, x, self.cfg.tta) if self.cfg.tta > 1 else model(x)
                else:
                    logits = tta_logits(model, x, self.cfg.tta) if self.cfg.tta > 1 else model(x)
            yield idx.numpy(), torch.sigmoid(logits.float()).cpu().numpy()
            if progress:
                progress(stage, (b + 1) / n_batches)

    # -------------------------------------------------------------------- main
    def analyze(self, slide_path: str, out_dir: Optional[str] = None, gt_xml: Optional[str] = None,
                progress: Optional[ProgressFn] = None) -> Dict:
        cfg, timer = self.cfg, Timer()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        reader = WSIReader(slide_path, mpp_override=cfg.mpp_override)
        W, H = reader.dimensions

        # 1. thumbnail + tissue ------------------------------------------------
        with timer("tissue_detection"):
            thumb, thumb_ds = reader.thumbnail(cfg.thumb_max_side)
            tissue = tissue_mask(thumb)
        if progress:
            progress("tissue", 1.0)

        map_ds = max(1.0, max(W, H) / float(cfg.map_max_side))
        map_shape = (int(np.ceil(H / map_ds)), int(np.ceil(W / map_ds)))
        tissue_map = resize_to(tissue.astype(np.uint8), map_shape, nearest=True) > 0

        # 2. classifier --------------------------------------------------------
        cls = self.classifier
        cls_fp = reader.footprint(cls.patch_size, cls.mpp)
        grid = TileGrid((W, H), cls_fp, cfg.cls_overlap, tissue, thumb_ds, cfg.min_tissue)
        coords = grid.coords()
        tile_probs = np.zeros(len(coords), np.float32)
        cls_acc = StitchAccumulator((W, H), map_ds, window="flat")
        with timer("classification"):
            if len(coords):
                for idx, p in self._run(cls.model, self._loader(slide_path, coords, cls.patch_size, cls.mpp, cfg.batch_size),
                                        False, progress, "classification"):
                    tile_probs[idx] = p
                for (x, y, _, _), p in zip(coords, tile_probs):
                    cls_acc.add(np.full((2, 2), p, np.float32), x, y, cls_fp)
        cls_map = cls_acc.result() * tissue_map
        cls_thr = cfg.cls_threshold if cfg.cls_threshold is not None else float(cls.meta.get("threshold", 0.5))

        # 3. segmentation (gated) -----------------------------------------------
        seg_tiles, seg_bytes = 0, 0
        if self.segmenter is not None:
            seg = self.segmenter
            seg_fp = reader.footprint(seg.patch_size, seg.mpp)
            seg_grid = TileGrid((W, H), seg_fp, cfg.seg_overlap, tissue, thumb_ds, min_tissue=0.05)
            sc = seg_grid.coords()
            if len(sc):
                ii = IntegralImage(cls_map >= cfg.seg_gate)
                x0 = np.floor(sc[:, 0] / map_ds).astype(np.int64)
                y0 = np.floor(sc[:, 1] / map_ds).astype(np.int64)
                x1 = np.ceil((sc[:, 0] + seg_fp) / map_ds).astype(np.int64)
                y1 = np.ceil((sc[:, 1] + seg_fp) / map_ds).astype(np.int64)
                sc = sc[ii.rect_mean(x0, y0, x1, y1) > 0]
            seg_acc = StitchAccumulator((W, H), map_ds, window="hann")
            seg_bytes = seg_acc.nbytes
            with timer("segmentation"):
                if len(sc):
                    for idx, p in self._run(seg.model, self._loader(slide_path, sc, seg.patch_size, seg.mpp, cfg.seg_batch_size),
                                            True, progress, "segmentation"):
                        for j, pj in zip(idx, p):
                            seg_acc.add(pj[0], sc[j, 0], sc[j, 1], seg_fp)
            seg_tiles = len(sc)
            final_map = seg_acc.result() * tissue_map
            final_thr = cfg.seg_threshold
            source = "segmentation"
        else:
            final_map, final_thr, source = cls_map, cls_thr, "classification"

        # 4. quantification ------------------------------------------------------
        with timer("quantification"):
            q = quantify(final_map, tissue_map, map_ds, reader.mpp, final_thr, cfg.min_lesion_um2)
        top = np.sort(tile_probs)[::-1]
        slide_score = {
            "max_tile_prob": float(top[0]) if len(top) else 0.0,
            "top10_mean_prob": float(top[:10].mean()) if len(top) else 0.0,
            "n_tiles_above_threshold": int((tile_probs >= cls_thr).sum()),
            "predicted_label": "tumor" if q["n_lesions"] > 0 else "normal",
        }

        total = time.perf_counter() - t_start
        n_cls = int(len(coords))
        bench = {
            "stage_seconds": timer.as_dict(),
            "total_seconds": round(total, 3),
            "grid_cells_total": grid.total_cells,
            "tiles_classified": n_cls,
            "tiles_skipped_background_pct": round(100 * (1 - n_cls / max(grid.total_cells, 1)), 1),
            "tiles_segmented": seg_tiles,
            "classification_tiles_per_sec": round(n_cls / max(timer.stages.get("classification", 1e-9), 1e-9), 1),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if self.device.type == "cuda" else None,
            "canvas_mb": round((cls_acc.nbytes + seg_bytes) / 2 ** 20, 1),
            "full_res_rgb_would_need_gb": round(W * H * 3 / 2 ** 30, 2),
            "device": str(self.device),
        }

        report = {
            "slide": reader.info.to_dict(),
            "config": asdict(cfg),
            "models": {"classifier": cls.meta, "segmenter": self.segmenter.meta if self.segmenter else None},
            "map_source": source,
            "classifier_threshold": cls_thr,
            "slide_score": slide_score,
            "quantification": q,
            "benchmark": bench,
        }

        # 5. ground-truth comparison --------------------------------------------
        gt_map = None
        if gt_xml and os.path.exists(gt_xml):
            gt_map = rasterize(load_asap_xml(gt_xml), map_shape, map_ds)
            pred_mask = (final_map >= final_thr) & tissue_map
            report["ground_truth"] = {"xml": os.path.basename(gt_xml), **mask_dice_iou(pred_mask, gt_map),
                                      "gt_tumor_area_mm2": float(gt_map.sum()) * (map_ds * reader.mpp / 1000) ** 2}

        artefacts = {"thumb": thumb, "tissue": tissue, "map": final_map, "cls_map": cls_map, "tissue_map": tissue_map,
                     "gt_map": gt_map, "coords": coords, "tile_probs": tile_probs, "threshold": final_thr,
                     "map_ds": map_ds, "cls_footprint": cls_fp}
        if out_dir:
            self.save(report, artefacts, out_dir)
        reader.close()
        report["_artefacts"] = artefacts
        return report

    # -------------------------------------------------------------------- export
    @staticmethod
    def save(report: Dict, a: Dict, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        thumb = a["thumb"]
        th, tw = thumb.shape[:2]
        wr = lambda name, img: cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        wr("thumbnail.png", thumb)
        wr("tissue_mask.png", overlay_mask(thumb, a["tissue"], (0, 160, 255), 0.4))
        tissue_t = resize_to(a["tissue_map"].astype(np.uint8), (th, tw), nearest=True) > 0
        heat = overlay_heatmap(thumb, a["cls_map"], tissue_t, alpha=0.6, min_prob=0.1)
        wr("heatmap_overlay.png", add_colorbar(heat))
        pred = (a["map"] >= a["threshold"]) & a["tissue_map"]
        seg_img = overlay_mask(thumb, pred, (230, 30, 30), 0.45)
        seg_img = draw_mask_contours(seg_img, pred, (200, 0, 0), 1)
        if a.get("gt_map") is not None:
            seg_img = draw_mask_contours(seg_img, a["gt_map"], (0, 200, 0), 2)
        wr("prediction_overlay.png", seg_img)
        cv2.imwrite(os.path.join(out_dir, "tumor_mask.png"), (pred * 255).astype(np.uint8))
        np.save(os.path.join(out_dir, "prob_map.npy"), a["map"].astype(np.float16))

        with open(os.path.join(out_dir, "tiles.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "row", "col", "footprint", "prob"])
            for (x, y, r, c), p in zip(a["coords"], a["tile_probs"]):
                w.writerow([int(x), int(y), int(r), int(c), round(a["cls_footprint"], 2), round(float(p), 5)])
        lesions = report["quantification"]["lesions"]
        with open(os.path.join(out_dir, "lesions.csv"), "w", newline="") as f:
            if lesions:
                w = csv.DictWriter(f, fieldnames=list(lesions[0].keys()))
                w.writeheader()
                w.writerows(lesions)
        polys = mask_to_polygons(pred, a["map_ds"])
        save_geojson(polys, os.path.join(out_dir, "predictions.geojson"))
        save_asap_xml(polys, os.path.join(out_dir, "predictions.xml"))
        write_json(report, os.path.join(out_dir, "report.json"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide", required=True)
    p.add_argument("--cls-ckpt", required=True)
    p.add_argument("--seg-ckpt", default=None)
    p.add_argument("--gt-xml", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seg-batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--stain-norm", action="store_true")
    p.add_argument("--map-max-side", type=int, default=4096)
    p.add_argument("--seg-gate", type=float, default=0.3)
    p.add_argument("--device", default="auto")
    p.add_argument("--mpp", type=float, default=None, help="override level-0 microns/px")
    a = p.parse_args(argv)
    cfg = AnalysisConfig(classifier_ckpt=a.cls_ckpt, segmenter_ckpt=a.seg_ckpt, batch_size=a.batch_size,
                         seg_batch_size=a.seg_batch_size,
                         num_workers=a.workers, tta=a.tta, stain_norm=a.stain_norm, map_max_side=a.map_max_side,
                         seg_gate=a.seg_gate, device=a.device, mpp_override=a.mpp)
    last = {"stage": None}

    def prog(stage, frac):
        if stage != last["stage"] or frac >= 1.0:
            print(f"  [{stage}] {frac * 100:5.1f}%", flush=True)
            last["stage"] = stage

    rep = SlideAnalyzer(cfg).analyze(a.slide, a.out, a.gt_xml, prog)
    rep.pop("_artefacts")
    q, b = rep["quantification"], rep["benchmark"]
    print(json.dumps({"slide_score": rep["slide_score"],
                      "tumor_area_mm2": round(q["tumor_area_mm2"], 3), "tumor_burden_pct": round(q["tumor_burden_pct"], 2),
                      "n_lesions": q["n_lesions"], "largest_lesion_mm": round(q["largest_lesion_diameter_mm"], 3),
                      "category": q["slide_category"], "ground_truth": rep.get("ground_truth"),
                      "benchmark": b}, indent=2))


if __name__ == "__main__":
    main()
