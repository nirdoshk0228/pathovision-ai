"""Train the patch-level tumour classifier.

Examples
--------
PCam (recommended; official train/valid/test splits are slide-disjoint):

    python -m pathovision.training.train_classifier \
        --data-format pcam_h5 --data-dir data/pcam --arch resnet18 --epochs 8 \
        --batch-size 256 --max-train 100000 --out-dir runs/cls_resnet18

Kaggle "Histopathologic Cancer Detection" folder:

    python -m pathovision.training.train_classifier \
        --data-format kaggle_hcd --hcd-dir /kaggle/input/histopathologic-cancer-detection ...
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pathovision.data.augment import Augmenter
from pathovision.data.datasets import KaggleHCDDataset, PCamH5Dataset
from pathovision.evaluation.metrics import best_f1_threshold, classification_metrics, roc_points
from pathovision.models.classifier import PatchClassifier
from pathovision.training.engine import EarlyStopping, predict_probs, worker_init_fn
from pathovision.training.losses import FocalLoss
from pathovision.utils.common import get_device, save_checkpoint, seed_everything, write_json

PCAM_FILES = {
    "train": ("camelyonpatch_level_2_split_train_x.h5", "camelyonpatch_level_2_split_train_y.h5"),
    "valid": ("camelyonpatch_level_2_split_valid_x.h5", "camelyonpatch_level_2_split_valid_y.h5"),
    "test": ("camelyonpatch_level_2_split_test_x.h5", "camelyonpatch_level_2_split_test_y.h5"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-format", choices=["pcam_h5", "kaggle_hcd"], default="pcam_h5")
    p.add_argument("--data-dir", default="data/pcam", help="folder with PCam-format *_x.h5/*_y.h5 files")
    p.add_argument("--hcd-dir", default=None, help="Kaggle HCD root containing train/ and train_labels.csv")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=["bce", "focal"], default="bce")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-train", type=int, default=0, help="random subset of training patches (0 = all)")
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--tta", type=int, default=8, help="dihedral TTA views at test time (1 = off)")
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--mpp", type=float, default=0.972, help="microns/px of the training patches (PCam = 10x)")
    p.add_argument("--out-dir", default="runs/classifier")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def _subset(n_total: int, n_max: int, rng: np.random.Generator):
    if n_max and n_max < n_total:
        return np.sort(rng.choice(n_total, n_max, replace=False))
    return None


def build_datasets(args, rng):
    aug = Augmenter(stain=True, color=True)
    if args.data_format == "pcam_h5":
        paths = {k: (os.path.join(args.data_dir, x), os.path.join(args.data_dir, y)) for k, (x, y) in PCAM_FILES.items()}
        # also accept our own extracted patches named train_x.h5 / train_y.h5 etc.
        for k in paths:
            if not os.path.exists(paths[k][0]):
                alt = (os.path.join(args.data_dir, f"{k}_x.h5"), os.path.join(args.data_dir, f"{k}_y.h5"))
                if os.path.exists(alt[0]):
                    paths[k] = alt
        n = {k: len(PCamH5Dataset(*paths[k])) for k in paths}
        tr = PCamH5Dataset(*paths["train"], augment=aug, indices=_subset(n["train"], args.max_train, rng))
        va = PCamH5Dataset(*paths["valid"], indices=_subset(n["valid"], args.max_val, rng))
        te = PCamH5Dataset(*paths["test"], indices=_subset(n["test"], args.max_test, rng))
        return tr, va, te
    # Kaggle HCD: no slide IDs are published, so the split is random (a known leakage risk; PCam is preferred).
    import pandas as pd

    df = pd.read_csv(os.path.join(args.hcd_dir, "train_labels.csv")).sample(frac=1.0, random_state=args.seed)
    n = len(df)
    tr_df, va_df, te_df = df.iloc[: int(0.8 * n)], df.iloc[int(0.8 * n): int(0.9 * n)], df.iloc[int(0.9 * n):]
    if args.max_train:
        tr_df = tr_df.iloc[: args.max_train]
    if args.max_val:
        va_df = va_df.iloc[: args.max_val]
    if args.max_test:
        te_df = te_df.iloc[: args.max_test]
    img_dir = os.path.join(args.hcd_dir, "train")
    mk = lambda d, a=None: KaggleHCDDataset(img_dir, d["id"].tolist(), d["label"].tolist(), a)
    return mk(tr_df, aug), mk(va_df), mk(te_df)


def save_roc_plot(y, p, path, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fpr, tpr, _ = roc_points(y, p)
    m = classification_metrics(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device}")

    tr, va, te = build_datasets(args, rng)
    print(f"patches: train={len(tr)} val={len(va)} test={len(te)}")
    pin = device.type == "cuda"
    dl_kw = dict(num_workers=args.workers, pin_memory=pin, worker_init_fn=worker_init_fn,
                 persistent_workers=args.workers > 0)
    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(va, batch_size=args.batch_size * 2, shuffle=False, **dl_kw)
    te_dl = DataLoader(te, batch_size=args.batch_size * 2, shuffle=False, **dl_kw)

    model = PatchClassifier(args.arch, pretrained=not args.no_pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr_dl),
                                                pct_start=0.15)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = FocalLoss() if args.loss == "focal" else None
    stopper = EarlyStopping(args.patience, "max")
    ckpt_path = os.path.join(args.out_dir, "best.pt")
    meta = {"task": "classification", "arch": args.arch, "patch_size": args.patch_size, "mpp": args.mpp,
            "label_rule": "positive if centre 32x32 region contains tumour (PCam convention)",
            "input_norm": "imagenet", "data_format": args.data_format}

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(tr_dl):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = crit(logits.float(), y) if crit else F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(tr_dl)} loss {run_loss / max(seen, 1):.4f}", flush=True)
        p_val, y_val = predict_probs(model, va_dl, device)
        m = classification_metrics(y_val, p_val)
        row = {"epoch": epoch, "train_loss": run_loss / max(seen, 1), "val_auc": m["roc_auc"],
               "val_acc": m["accuracy"], "val_f1": m["f1"], "lr": sched.get_last_lr()[0],
               "epoch_sec": time.time() - t0, "train_patches_per_sec": seen / (time.time() - t0)}
        history.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} val_auc {m['roc_auc']:.4f} "
              f"val_acc {m['accuracy']:.4f} ({row['epoch_sec']:.0f}s, {row['train_patches_per_sec']:.0f} patch/s)")
        if stopper.step(m["roc_auc"]):
            save_checkpoint(ckpt_path, model, {**meta, "epoch": epoch, "val_auc": m["roc_auc"]})
        if stopper.should_stop:
            print("early stopping")
            break

    with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    # ---- final evaluation with the best checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    p_val, y_val = predict_probs(model, va_dl, device)
    thr = best_f1_threshold(y_val, p_val)  # chosen on validation only
    results = {"val": classification_metrics(y_val, p_val, thr)}
    p_te, y_te = predict_probs(model, te_dl, device)
    results["test"] = classification_metrics(y_te, p_te, thr)
    if args.tta > 1:
        p_tta, _ = predict_probs(model, te_dl, device, tta=args.tta)
        results["test_tta"] = classification_metrics(y_te, p_tta, thr)
        save_roc_plot(y_te, p_tta, os.path.join(args.out_dir, "roc_test.png"), f"{args.arch} test ROC (TTA x{args.tta})")
    else:
        save_roc_plot(y_te, p_te, os.path.join(args.out_dir, "roc_test.png"), f"{args.arch} test ROC")
    save_checkpoint(ckpt_path, model, {**state["meta"], "threshold": thr,
                                       "test_auc": results.get("test_tta", results["test"])["roc_auc"]})
    write_json({"args": vars(args), "results": results, "history": history},
               os.path.join(args.out_dir, "metrics.json"))
    for k, v in results.items():
        print(f"{k:9s} AUC {v['roc_auc']:.4f}  acc {v['accuracy']:.4f}  P {v['precision']:.4f}  "
              f"R {v['recall']:.4f}  F1 {v['f1']:.4f}")
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
