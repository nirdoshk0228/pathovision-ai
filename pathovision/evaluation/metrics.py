"""Evaluation metrics for patch classification, segmentation and slide-level prediction."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "specificity": float(tn / max(tn + fp, 1))})
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    return out


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising F1 on a validation set (pick on val, report on test)."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def roc_points(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return fpr, tpr, thr


@torch.no_grad()
def dice_iou_batch(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """Sums for micro-averaged Dice/IoU over a batch: returns (intersection, pred_sum, target_sum)."""
    pred = (torch.sigmoid(logits) >= threshold).float()
    inter = (pred * target).sum().item()
    return inter, pred.sum().item(), target.sum().item()


def dice_iou_from_sums(inter: float, psum: float, tsum: float, eps: float = 1e-6) -> Dict[str, float]:
    dice = (2 * inter + eps) / (psum + tsum + eps)
    iou = (inter + eps) / (psum + tsum - inter + eps)
    return {"dice": float(dice), "iou": float(iou)}


def mask_dice_iou(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred, target = pred.astype(bool), target.astype(bool)
    inter = float((pred & target).sum())
    return dice_iou_from_sums(inter, float(pred.sum()), float(target.sum()))


def lesion_detection(pred: np.ndarray, gt: np.ndarray, um_per_px: float, min_lesion_um2: float = 200.0) -> Dict[str, float]:
    """Lesion-level detection, in the spirit of the CAMELYON16 FROC evaluation.

    A ground-truth lesion is *detected* if any predicted pixel overlaps it; a
    predicted lesion is a *false positive* if it overlaps no ground-truth lesion.
    Components smaller than ``min_lesion_um2`` are ignored on both sides.
    """
    import cv2

    min_px = max(1, int(np.ceil(min_lesion_um2 / um_per_px ** 2)))

    def comps(m):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
        return lab, keep

    gl, gk = comps(gt)
    pl, pk = comps(pred)
    pred_any = np.isin(pl, pk)
    gt_any = np.isin(gl, gk)
    # overlap counts per label in one pass each (no per-lesion full-image scans)
    gt_hits = np.bincount(gl[pred_any], minlength=gl.max() + 1)
    pred_hits = np.bincount(pl[gt_any], minlength=pl.max() + 1)
    detected = int(sum(gt_hits[i] > 0 for i in gk))
    fp = int(sum(pred_hits[i] == 0 for i in pk))
    return {"gt_lesions": len(gk), "detected_lesions": detected,
            "lesion_sensitivity": float(detected / len(gk)) if gk else float("nan"),
            "pred_lesions": len(pk), "fp_lesions": fp}
