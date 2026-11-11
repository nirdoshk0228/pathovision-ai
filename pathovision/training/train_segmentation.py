"""Train the U-Net tumour segmenter on patches extracted from annotated WSIs.

Expects the layout written by ``scripts/extract_patches.py --seg-out``::

    <data-dir>/train/images/*.png  <data-dir>/train/masks/*.png
    <data-dir>/valid/images/*.png  <data-dir>/valid/masks/*.png   (or val/)

Train and val are split **by slide** so no tissue from a validation slide is
ever seen in training (patches from the same slide are highly correlated).

    python -m pathovision.training.train_segmentation --data-dir data/seg_patches \
        --arch resnet34_unet --epochs 25 --out-dir runs/seg_resnet34
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pathovision.data.augment import Augmenter
from pathovision.data.datasets import SegPatchDataset
from pathovision.evaluation.metrics import dice_iou_batch, dice_iou_from_sums
from pathovision.models.unet import build_segmenter
from pathovision.training.engine import EarlyStopping, tta_seg_logits, worker_init_fn
from pathovision.training.losses import BCEDiceLoss
from pathovision.utils.common import get_device, save_checkpoint, seed_everything, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/seg_patches")
    p.add_argument("--arch", default="resnet34_unet", choices=["unet", "resnet18_unet", "resnet34_unet", "resnet50_unet"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--stain-sigma", type=float, default=0.08,
                   help="HED stain jitter strength (slides from different labs/batches differ in colour)")
    p.add_argument("--out-dir", default="runs/segmentation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, device, tta=1):
    model.eval()
    inter = psum = tsum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = tta_seg_logits(model, x, tta) if tta > 1 else model(x)
        i, p, t = dice_iou_batch(logits.float(), y)
        inter, psum, tsum = inter + i, psum + p, tsum + t
    return dice_iou_from_sums(inter, psum, tsum)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tr = SegPatchDataset(os.path.join(args.data_dir, "train"), augment=Augmenter(stain_sigma=args.stain_sigma))
    val_dir = next((os.path.join(args.data_dir, d) for d in ("val", "valid")
                    if os.path.isdir(os.path.join(args.data_dir, d))), os.path.join(args.data_dir, "val"))
    va = SegPatchDataset(val_dir)
    info_path = os.path.join(args.data_dir, "extraction.json")
    extraction = json.load(open(info_path)) if os.path.exists(info_path) else {}
    print(f"device={device} train={len(tr)} val={len(va)}")

    dl_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn,
                 persistent_workers=args.workers > 0)
    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=len(tr) > args.batch_size, **dl_kw)
    va_dl = DataLoader(va, batch_size=args.batch_size, shuffle=False, **dl_kw)

    model = build_segmenter(args.arch, pretrained=not args.no_pretrained).to(device)
    crit = BCEDiceLoss(args.bce_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopping(args.patience, "max")
    ckpt_path = os.path.join(args.out_dir, "best.pt")
    meta = {"task": "segmentation", "arch": args.arch,
            "patch_size": extraction.get("seg_patch_size", 256), "mpp": extraction.get("seg_mpp", 0.972),
            "threshold": 0.5, "input_norm": "imagenet"}

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()
        m = evaluate(model, va_dl, device)
        row = {"epoch": epoch, "train_loss": run / max(seen, 1), "val_dice": m["dice"], "val_iou": m["iou"],
               "epoch_sec": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} val_dice {m['dice']:.4f} val_iou {m['iou']:.4f} "
              f"({row['epoch_sec']:.0f}s)", flush=True)
        if stopper.step(m["dice"]):
            save_checkpoint(ckpt_path, model, {**meta, "epoch": epoch, "val_dice": m["dice"], "val_iou": m["iou"]})
        if stopper.should_stop:
            print("early stopping")
            break

    with open(os.path.join(args.out_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    final = {"val": evaluate(model, va_dl, device)}
    if args.tta > 1:
        final["val_tta"] = evaluate(model, va_dl, device, args.tta)
    write_json({"args": vars(args), "results": final, "history": history}, os.path.join(args.out_dir, "metrics.json"))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
