#!/usr/bin/env python3
"""A0 step 3: whole-image VALIDATION reference values for the Stage B decision rule (M0, 3 seeds) and, for context,
nnU-Net (3 seeds, its own fold_0/validation masks).

- mIoU / per-class IoU / Dead recall: pooled confusion counts over the 164 images (shared-evaluator convention,
  valid GT 0..3, mIoU over 4 classes).
- Boundary F1: Stage-30 definitions copied verbatim (boundary = mask & ~erode3x3, border_value=0; BF1 at tolerance
  t from distance transforms; both-empty = 1, one-empty = 0; GT values >3 treated as ignore = not foreground).
  Foreground (alive U dead) BF1 at 2 px averaged over all 164 images = decision endpoint;
  Dead BF1 at 2 px averaged over the validation images whose GT contains Dead = descriptive.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path('3D_OrganoidSeg')
VOC = Path('OrganoidSegBaseline/VOC2012')
OUT = ROOT / 'vco/runs/arch_exploration_20261104/A0_gap_decomposition'
NN = ROOT / 'baselines_fair_20261030/nnunet'
SEEDS = [42, 3407, 2026]
NC = 4
SRC = {'M0': lambda s: OUT / f'm0_val_masks/seed{s}',
       'nnUNet': lambda s: NN / f'nnUNet_results_run2_seed{s}/Dataset801_CRCOrganoid/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation'}


def boundary(mask):
    mask = np.asarray(mask, bool)
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def bf1(gt_mask, pred_mask, tol=2):
    gt_b, pred_b = boundary(gt_mask), boundary(pred_mask)
    n_gt, n_pred = int(gt_b.sum()), int(pred_b.sum())
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0
    d_gt = ndi.distance_transform_edt(~gt_b)
    d_pr = ndi.distance_transform_edt(~pred_b)
    p = float((d_gt[pred_b] <= tol).mean())
    r = float((d_pr[gt_b] <= tol).mean())
    return 2 * p * r / (p + r) if p + r else 0.0


def one(args):
    model, seed, stem = args
    gt = np.asarray(Image.open(VOC / 'SegmentationClass' / f'{stem}.png'), dtype=np.int16)
    gt[(gt >= 250) | (gt > 3)] = -1
    pred = np.asarray(Image.open(SRC[model](seed) / f'{stem}.png'), dtype=np.int16)
    valid = gt >= 0
    cm = np.bincount(gt[valid] * NC + pred[valid], minlength=NC * NC).reshape(NC, NC)
    has_dead = bool((gt == 2).any())
    return {'model': model, 'seed': seed, 'stem': stem, 'cm': cm,
            'fg_bf1_2px': bf1(np.isin(gt, (1, 2)), np.isin(pred, (1, 2))),
            'dead_bf1_2px': bf1(gt == 2, pred == 2) if has_dead else np.nan, 'has_dead': has_dead}


def main():
    stems = [s.strip() for s in (VOC / 'ImageSets/Segmentation/val_cond.txt').read_text().splitlines() if s.strip()]
    jobs = [(m, s, st) for m in SRC for s in SEEDS for st in stems]
    with Pool(32) as pool:
        recs = pool.map(one, jobs, chunksize=4)
    rows = []
    for m in SRC:
        for s in SEEDS:
            rr = [r for r in recs if r['model'] == m and r['seed'] == s]
            cm = np.sum([r['cm'] for r in rr], axis=0).astype(float)
            tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
            iou = np.divide(tp, tp + fp + fn, out=np.full(NC, np.nan), where=(tp + fp + fn) > 0) * 100
            rec = np.divide(tp, tp + fn, out=np.full(NC, np.nan), where=(tp + fn) > 0) * 100
            prec = np.divide(tp, tp + fp, out=np.full(NC, np.nan), where=(tp + fp) > 0) * 100
            rows.append({'model': m, 'seed': s, 'n_images': len(rr), 'mIoU': np.nanmean(iou),
                         'iou_background': iou[0], 'iou_alive': iou[1], 'iou_dead': iou[2], 'iou_bubble': iou[3],
                         'dead_recall': rec[2], 'dead_precision': prec[2],
                         'fg_bf1_2px_mean': float(np.mean([r['fg_bf1_2px'] for r in rr])),
                         'dead_bf1_2px_mean_deadimgs': float(np.nanmean([r['dead_bf1_2px'] for r in rr])),
                         'n_dead_images': int(sum(r['has_dead'] for r in rr))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'a0_val_reference_per_seed.csv', index=False)
    summ = df.groupby('model').agg(['mean', 'min', 'max', 'std']).drop(columns=['seed', 'n_images', 'n_dead_images'])
    summ.to_csv(OUT / 'a0_val_reference_summary.csv')
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
