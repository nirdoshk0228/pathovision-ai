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
