#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASSES = ["background", "live", "death", "bubble"]
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def read_ids(path):
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def find_image(img_dir, img_id):
    for suffix in [".jpg", ".png", ".tif", ".tiff", ".jpeg"]:
        path = img_dir / f"{img_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for {img_id}")


class VocSegDataset(Dataset):
    def __init__(self, voc_root, split_file, crop_size=512, train=True):
        self.voc_root = Path(voc_root)
        self.img_dir = self.voc_root / "JPEGImages"
        self.ann_dir = self.voc_root / "SegmentationClass"
        self.ids = read_ids(split_file)
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.ids)

    def _load(self, img_id):
        img = cv2.imread(str(find_image(self.img_dir, img_id)), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image {img_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.ann_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {img_id}")
        return img, mask

    def __getitem__(self, index):
        img_id = self.ids[index]
        img, mask = self._load(img_id)
        if self.train:
            img, mask = augment(img, mask, self.crop_size)
        img = normalize_to_tensor(img)
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask, img_id


def normalize_to_tensor(img):
    img = img.astype(np.float32)
    img = (img - MEAN) / STD
    img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
    return img


def augment(img, mask, crop_size):
    scale = random.uniform(0.5, 2.0)
    h, w = mask.shape
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    if random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    h, w = mask.shape
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        h, w = mask.shape

    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)
    img = img[y:y + crop_size, x:x + crop_size]
    mask = mask[y:y + crop_size, x:x + crop_size]
    return img, mask


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes=4, base=64):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.center = DoubleConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(F.max_pool2d(d1, 2))
        d3 = self.d3(F.max_pool2d(d2, 2))
        d4 = self.d4(F.max_pool2d(d3, 2))
        c = self.center(F.max_pool2d(d4, 2))
        x = self.c4(torch.cat([self.u4(c), d4], dim=1))
        x = self.c3(torch.cat([self.u3(x), d3], dim=1))
        x = self.c2(torch.cat([self.u2(x), d2], dim=1))
        x = self.c1(torch.cat([self.u1(x), d1], dim=1))
        return self.out(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=4, base=64):
        super().__init__()
        nb = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = DoubleConv(3, nb[0])
        self.conv1_0 = DoubleConv(nb[0], nb[1])
        self.conv2_0 = DoubleConv(nb[1], nb[2])
        self.conv3_0 = DoubleConv(nb[2], nb[3])
        self.conv4_0 = DoubleConv(nb[3], nb[4])

        self.conv0_1 = DoubleConv(nb[0] + nb[1], nb[0])
        self.conv1_1 = DoubleConv(nb[1] + nb[2], nb[1])
        self.conv2_1 = DoubleConv(nb[2] + nb[3], nb[2])
        self.conv3_1 = DoubleConv(nb[3] + nb[4], nb[3])

        self.conv0_2 = DoubleConv(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = DoubleConv(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = DoubleConv(nb[2] * 2 + nb[3], nb[2])

        self.conv0_3 = DoubleConv(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = DoubleConv(nb[1] * 3 + nb[2], nb[1])

        self.conv0_4 = DoubleConv(nb[0] * 4 + nb[1], nb[0])
        self.out = nn.Conv2d(nb[0], num_classes, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))
        return self.out(x0_4)


def poly_lr(base_lr, cur_iter, max_iters, power=0.9):
    return base_lr * (1 - cur_iter / max_iters) ** power


def compute_metrics(preds, gts, num_classes=4):
    inter = np.zeros(num_classes, dtype=np.int64)
    pred_area = np.zeros(num_classes, dtype=np.int64)
    gt_area = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    for pred, gt in zip(preds, gts):
        for c in range(num_classes):
            p = pred == c
            g = gt == c
            i = int(np.logical_and(p, g).sum())
            pa = int(p.sum())
            ga = int(g.sum())
            inter[c] += i
            pred_area[c] += pa
            gt_area[c] += ga
            union[c] += pa + ga - i
    iou = inter / np.maximum(union, 1)
    dice = 2 * inter / np.maximum(pred_area + gt_area, 1)
    acc = inter / np.maximum(gt_area, 1)
    return {
        "per_class": {
            name: {
                "intersection": int(inter[i]),
                "pred_area": int(pred_area[i]),
                "gt_area": int(gt_area[i]),
                "union": int(union[i]),
                "iou": float(iou[i]),
                "dice": float(dice[i]),
                "acc": float(acc[i]),
            }
            for i, name in enumerate(CLASSES)
        },
        "mIoU": float(np.nanmean(iou)),
        "mDice": float(np.nanmean(dice)),
        "mAcc": float(np.nanmean(acc)),
        "aAcc": float(inter.sum() / max(gt_area.sum(), 1)),
    }


def pad_tensor(x, div=16):
    _, _, h, w = x.shape
    ph = (div - h % div) % div
    pw = (div - w % div) % div
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0)
    return x, h, w


@torch.no_grad()
def predict_dataset(model, voc_root, split_file, device, amp=True):
    model.eval()
    voc_root = Path(voc_root)
    img_dir = voc_root / "JPEGImages"
    ann_dir = voc_root / "SegmentationClass"
    ids = read_ids(split_file)
    preds, gts = [], []
    for img_id in tqdm(ids, desc="infer", leave=False):
        img = cv2.imread(str(find_image(img_dir, img_id)), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(ann_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        x = normalize_to_tensor(img).unsqueeze(0).to(device)
        x, h, w = pad_tensor(x)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(x)
        pred = logits[:, :, :h, :w].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        preds.append(pred)
        gts.append(gt.astype(np.uint8))
    return ids, preds, gts


def relative_area_errors(ids, preds, gts, out_csv):
    rows = []
    rel_by_class = {1: [], 2: [], 3: []}
    rel_by_day = {}
    pred_area = np.zeros(4, dtype=np.int64)
    gt_area = np.zeros(4, dtype=np.int64)
    for img_id, pred, gt in zip(ids, preds, gts):
        day_match = re.match(r"D(\d+)_", img_id)
        day = "D" + day_match.group(1) if day_match else "unknown"
        rel_by_day.setdefault(day, {1: [], 2: [], 3: []})
        row = {"image_id": img_id, "day": day}
        for c, name in enumerate(CLASSES):
            cur_pred = int((pred == c).sum())
            cur_gt = int((gt == c).sum())
            pred_area[c] += cur_pred
            gt_area[c] += cur_gt
            row[f"pred_{name}_area"] = cur_pred
            row[f"gt_{name}_area"] = cur_gt
            if c in [1, 2, 3]:
                if cur_gt > 0:
                    rel = abs(cur_pred - cur_gt) / cur_gt
                    row[f"{name}_relative_error"] = rel
                    rel_by_class[c].append(rel)
                    rel_by_day[day][c].append(rel)
                else:
                    row[f"{name}_relative_error"] = "NA"
        rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    overall = {}
    for c, name in [(1, "live"), (2, "death"), (3, "bubble")]:
        vals = np.array(rel_by_class[c], dtype=float)
        total_rel = abs(int(pred_area[c]) - int(gt_area[c])) / int(gt_area[c]) if gt_area[c] else None
        overall[name] = {
            "n_images_with_gt": int(len(vals)),
            "mean_relative_error": float(vals.mean()) if len(vals) else None,
            "median_relative_error": float(np.median(vals)) if len(vals) else None,
            "std_relative_error": float(vals.std()) if len(vals) else None,
            "total_pred_area": int(pred_area[c]),
            "total_gt_area": int(gt_area[c]),
            "total_relative_error": float(total_rel) if total_rel is not None else None,
        }
    by_day = {}
    for day, d in sorted(rel_by_day.items(), key=lambda kv: int(kv[0][1:]) if kv[0].startswith("D") and kv[0][1:].isdigit() else 999):
        by_day[day] = {}
        for c, name in [(1, "live"), (2, "death"), (3, "bubble")]:
            vals = np.array(d[c], dtype=float)
            by_day[day][name] = {
                "n_images_with_gt": int(len(vals)),
                "mean_relative_error": float(vals.mean()) if len(vals) else None,
                "median_relative_error": float(np.median(vals)) if len(vals) else None,
            }
    return {"overall_by_class": overall, "by_day": by_day}


def connected_components(mask, min_area):
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep.append((idx, area))
    return labels, keep


def match_counts(gt_mask, pred_mask, min_area, iou_thr):
    gt_labels, gt_keep = connected_components(gt_mask, min_area)
    pred_labels, pred_keep = connected_components(pred_mask, min_area)
    gt_ids = [idx for idx, _ in gt_keep]
    pred_ids = [idx for idx, _ in pred_keep]
    gt_areas = dict(gt_keep)
    pred_areas = dict(pred_keep)
    if not gt_ids or not pred_ids:
        return 0, len(pred_ids), len(gt_ids)
    pairs = []
    both = (gt_labels > 0) & (pred_labels > 0)
    if both.any():
        mult = int(pred_labels.max()) + 1
        codes = gt_labels[both].astype(np.int64) * mult + pred_labels[both].astype(np.int64)
        uniq, counts = np.unique(codes, return_counts=True)
        valid_gt = set(gt_ids)
        valid_pred = set(pred_ids)
        for code, inter in zip(uniq, counts):
            gt_idx = int(code // mult)
            pred_idx = int(code % mult)
            if gt_idx not in valid_gt or pred_idx not in valid_pred:
                continue
            union = gt_areas[gt_idx] + pred_areas[pred_idx] - int(inter)
            iou = float(inter) / union if union else 0.0
            ok = iou > 0.0 if iou_thr == 0.0 else iou >= iou_thr
            if ok:
                pairs.append((iou, gt_idx, pred_idx))
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    tp = 0
    for _, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        tp += 1
    return tp, len(pred_ids) - tp, len(gt_ids) - tp


def safe_div(num, den):
    return num / den if den else 0.0


def object_metrics(preds, gts, min_areas=(0, 100, 500), iou_thrs=(0.0,)):
    rows = []
    for min_area in min_areas:
        for iou_thr in iou_thrs:
            tp = fp = fn = 0
            for pred, gt in zip(preds, gts):
                gt_org = (gt == 1) | (gt == 2)
                pred_org = (pred == 1) | (pred == 2)
                cur_tp, cur_fp, cur_fn = match_counts(gt_org, pred_org, min_area, iou_thr)
                tp += cur_tp
                fp += cur_fp
                fn += cur_fn
            recall = safe_div(tp, tp + fn)
            precision = safe_div(tp, tp + fp)
            f1 = safe_div(2 * precision * recall, precision + recall)
            rows.append({"min_area": min_area, "iou_thr": iou_thr, "tp": tp, "fp": fp, "fn": fn,
                         "recall": recall, "precision": precision, "f1": f1})

    pix_tp = pix_fp = pix_fn = pix_tn = 0
    for pred, gt in zip(preds, gts):
        gt_org = (gt == 1) | (gt == 2)
        pred_org = (pred == 1) | (pred == 2)
        pix_tp += int(np.logical_and(gt_org, pred_org).sum())
        pix_fp += int(np.logical_and(~gt_org, pred_org).sum())
        pix_fn += int(np.logical_and(gt_org, ~pred_org).sum())
        pix_tn += int(np.logical_and(~gt_org, ~pred_org).sum())
    pix_precision = safe_div(pix_tp, pix_tp + pix_fp)
    pix_recall = safe_div(pix_tp, pix_tp + pix_fn)
    pix_f1 = safe_div(2 * pix_precision * pix_recall, pix_precision + pix_recall)
    pix_iou = safe_div(pix_tp, pix_tp + pix_fp + pix_fn)
    pix_dice = safe_div(2 * pix_tp, 2 * pix_tp + pix_fp + pix_fn)
    return {
        "definition": "organoid = live(1) + death(2); non-organoid = background(0) + bubble(3)",
        "object_metrics": rows,
        "pixel_organoid": {"tp": pix_tp, "fp": pix_fp, "fn": pix_fn, "tn": pix_tn,
                           "iou": pix_iou, "dice": pix_dice, "recall": pix_recall,
                           "precision": pix_precision, "f1": pix_f1},
    }


def class_weights(voc_root, split_file):
    ann_dir = Path(voc_root) / "SegmentationClass"
    counts = np.zeros(4, dtype=np.int64)
    for img_id in read_ids(split_file):
        mask = cv2.imread(str(ann_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        binc = np.bincount(mask.reshape(-1), minlength=4)[:4]
        counts += binc
    freq = counts / max(counts.sum(), 1)
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.2, 10.0)
    return counts, weights.astype(np.float32)


def save_checkpoint(path, model, optimizer, scaler, iteration, best_miou, args):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "iter": iteration,
        "best_miou": best_miou,
        "args": vars(args),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("iter", 0)), float(state.get("best_miou", -1.0))


def append_log(path, row):
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", default="OrganoidSegBaseline/VOC2012")
    parser.add_argument("--train-split", default="OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/train_cond.txt")
    parser.add_argument("--val-split", default="OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/val_cond.txt")
    parser.add_argument("--test-split", default="OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/test_cond.txt")
    parser.add_argument("--work-dir", default="OrganoidSegBaseline/runs/unetpp_cond_40k_torch")
    parser.add_argument("--max-iters", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    model = UNetPlusPlus(num_classes=4, base=args.base_channels).to(device)
    counts, weights = class_weights(args.voc_root, args.train_split)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_iter = 0
    best_miou = -1.0
    latest_path = Path(args.work_dir) / "latest.pth"
    best_path = Path(args.work_dir) / "best_miou.pth"
    if args.checkpoint:
        start_iter, best_miou = load_checkpoint(args.checkpoint, model, optimizer if not args.test_only else None, scaler, device)
    elif args.resume and latest_path.exists():
        start_iter, best_miou = load_checkpoint(latest_path, model, optimizer, scaler, device)

    meta = {
        "model": "UNet++",
        "classes": CLASSES,
        "class_pixel_counts_train": counts.tolist(),
        "class_weights": weights.tolist(),
        "splits": {
            "train": args.train_split,
            "val": args.val_split,
            "test": args.test_split,
        },
        "args": vars(args),
    }
    Path(args.work_dir, "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not args.test_only:
        train_ds = VocSegDataset(args.voc_root, args.train_split, crop_size=args.crop_size, train=True)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
        data_iter = iter(loader)
        print(f"device={device} amp={amp} start_iter={start_iter} max_iters={args.max_iters}", flush=True)
        print(f"class_counts={counts.tolist()} class_weights={weights.tolist()}", flush=True)
        last_time = time.time()
        for it in range(start_iter + 1, args.max_iters + 1):
            try:
                imgs, masks, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                imgs, masks, _ = next(data_iter)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            lr = poly_lr(args.lr, it, args.max_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if it == 1 or it % 50 == 0:
                now = time.time()
                ips = args.batch_size * 50 / max(now - last_time, 1e-6)
                last_time = now
                print(f"iter={it:06d}/{args.max_iters} lr={lr:.6g} loss={loss.item():.5f} imgs_per_s={ips:.2f}", flush=True)

            if it % args.save_interval == 0:
                save_checkpoint(latest_path, model, optimizer, scaler, it, best_miou, args)
                save_checkpoint(Path(args.work_dir) / f"iter_{it}.pth", model, optimizer, scaler, it, best_miou, args)

            if it % args.val_interval == 0 or it == args.max_iters:
                ids, preds, gts = predict_dataset(model, args.voc_root, args.val_split, device, amp=amp)
                metrics = compute_metrics(preds, gts)
                row = {"iter": it, "loss": float(loss.item()), "lr": lr, **{k: metrics[k] for k in ["mIoU", "mDice", "mAcc", "aAcc"]}}
                for name in CLASSES:
                    row[f"{name}_iou"] = metrics["per_class"][name]["iou"]
                    row[f"{name}_dice"] = metrics["per_class"][name]["dice"]
                append_log(Path(args.work_dir) / "val_metrics.csv", row)
                Path(args.work_dir, f"val_iter_{it}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                print(f"VAL iter={it:06d} mIoU={metrics['mIoU']:.4f} mDice={metrics['mDice']:.4f} "
                      f"liveIoU={metrics['per_class']['live']['iou']:.4f} "
                      f"deathIoU={metrics['per_class']['death']['iou']:.4f} "
                      f"bubbleIoU={metrics['per_class']['bubble']['iou']:.4f}", flush=True)
                if metrics["mIoU"] > best_miou:
                    best_miou = metrics["mIoU"]
                    save_checkpoint(best_path, model, optimizer, scaler, it, best_miou, args)
                    print(f"BEST iter={it:06d} mIoU={best_miou:.4f}", flush=True)

        save_checkpoint(latest_path, model, optimizer, scaler, args.max_iters, best_miou, args)

    if args.skip_final_test and not args.test_only:
        return

    if args.test_only:
        ckpt = args.checkpoint or str(best_path if best_path.exists() else latest_path)
        load_checkpoint(ckpt, model, device=device)
    else:
        ckpt = str(best_path if best_path.exists() else latest_path)
        load_checkpoint(ckpt, model, device=device)

    ids, preds, gts = predict_dataset(model, args.voc_root, args.test_split, device, amp=amp)
    pred_pkl = Path(args.work_dir) / "test_best_predictions.pkl"
    with open(pred_pkl, "wb") as f:
        pickle.dump(preds, f)
    sem = compute_metrics(preds, gts)
    rel = relative_area_errors(ids, preds, gts, Path(args.work_dir) / "test_relative_area_errors.csv")
    obj = object_metrics(preds, gts)
    result = {
        "checkpoint": ckpt,
        "prediction": str(pred_pkl),
        "num_images": len(ids),
        "semantic_segmentation": sem,
        "relative_area_error": rel,
        "object_level": obj,
    }
    Path(args.work_dir, "test_full_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.work_dir, "test_object_metrics.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("TEST semantic", json.dumps(sem, indent=2), flush=True)
    print("TEST relative_area_error", json.dumps(rel["overall_by_class"], indent=2), flush=True)
    print("TEST object_level", json.dumps(obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
