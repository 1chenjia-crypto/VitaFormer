#!/usr/bin/env python3
"""Evaluate the frozen five-fold nnU-Net predictions on official test members."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


BASE = Path("3D_OrganoidSeg/baselines_fair_20261030/nnunet")
PRED_BASE = BASE / "brain20261106_official_test_predictions"
GT_DIR = BASE / "nnUNet_raw/Dataset802_BrainOrganoid2024/labelsTr"
MEMBERS = BASE / "brain20261104_test_members.json"
OUT_DIR = BASE / "brain20261106_official_test_metrics"


def metrics(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    pred = prediction > 0
    ref = reference > 0
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    dice_denom = 2 * tp + fp + fn
    iou_denom = tp + fp + fn
    dice = 1.0 if dice_denom == 0 else 2.0 * tp / dice_denom
    iou = 1.0 if iou_denom == 0 else tp / iou_denom
    return dice, iou


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing metrics: {OUT_DIR}")
    test_members = json.loads(MEMBERS.read_text())
    if len(test_members) != 5:
        raise RuntimeError("expected five official test folds")
    flattened = [case for fold in test_members for case in fold]
    if len(flattened) != 1407 or len(set(flattened)) != 1407:
        raise RuntimeError("official test folds must uniquely cover 1,407 images")

    rows: list[dict[str, object]] = []
    for fold, members in enumerate(test_members):
        pred_dir = PRED_BASE / f"fold{fold}"
        predicted = {path.stem for path in pred_dir.glob("*.png")}
        if predicted != set(members):
            missing = sorted(set(members) - predicted)
            extra = sorted(predicted - set(members))
            raise RuntimeError(
                f"fold {fold} prediction membership mismatch: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        for case in members:
            pred = np.asarray(Image.open(pred_dir / f"{case}.png"))
            ref = np.asarray(Image.open(GT_DIR / f"{case}.png"))
            if pred.shape != ref.shape:
                raise RuntimeError(
                    f"shape mismatch for {case}: {pred.shape} vs {ref.shape}"
                )
            dice, iou = metrics(pred, ref)
            rows.append({"fold": fold, "case": case, "dice": dice, "iou": iou})

    OUT_DIR.mkdir(parents=False, exist_ok=False)
    with (OUT_DIR / "per_image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "case", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)

    fold_results = []
    for fold in range(5):
        fold_rows = [row for row in rows if row["fold"] == fold]
        dices = [float(row["dice"]) for row in fold_rows]
        ious = [float(row["iou"]) for row in fold_rows]
        fold_results.append(
            {
                "fold": fold,
                "n": len(fold_rows),
                "mean_dice": statistics.mean(dices),
                "sd_dice": statistics.stdev(dices),
                "mean_iou": statistics.mean(ious),
                "sd_iou": statistics.stdev(ious),
            }
        )
    all_dice = [float(row["dice"]) for row in rows]
    all_iou = [float(row["iou"]) for row in rows]
    report = {
        "protocol": {
            "membership": str(MEMBERS),
            "checkpoint": "checkpoint_best.pth for each fold",
            "tta": False,
            "postprocessing": "none",
            "primary_metric": "mean of per-image binary foreground Dice over 1,407 OOF test images",
        },
        "folds": fold_results,
        "overall": {
            "n": len(rows),
            "mean_dice": statistics.mean(all_dice),
            "sd_dice": statistics.stdev(all_dice),
            "mean_iou": statistics.mean(all_iou),
            "sd_iou": statistics.stdev(all_iou),
            "fold_mean_dice": statistics.mean(row["mean_dice"] for row in fold_results),
            "fold_sd_dice": statistics.stdev(row["mean_dice"] for row in fold_results),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
