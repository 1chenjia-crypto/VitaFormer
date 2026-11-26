#!/usr/bin/env python3
"""Shared mask-level evaluator for all methods on the CRC test set (P0-B protocol §1).

Per image and class: tp/fp/fn over valid GT pixels (0 <= gt < 4), exactly as vco/organoid_seg/evaluate_test.py
(`Metrics.update`). Output JSON mirrors test_best_miou_uniform.json (`per_image[*].confusion_counts`) so the
existing 3-seed paired hierarchical bootstrap can consume it. Aggregation = pooled pixels over images,
mIoU over 4 classes incl. background (= analyze_multiseed_ablation.metric_values).

Usage: eval_masks_shared.py --masks DIR --out JSON [--check-json EXISTING_TEST_JSON]
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image

VOC = Path('OrganoidSegBaseline/VOC2012')
SPLIT = VOC / 'ImageSets/Segmentation/test_cond.txt'
NC = 4


def counts(pred, gt):
    valid = (gt >= 0) & (gt < NC)
    p, t = pred[valid].astype(np.int64), gt[valid].astype(np.int64)
    tp, fp, fn = [], [], []
    for c in range(NC):
        pc, tc = p == c, t == c
        tp.append(int((pc & tc).sum())); fp.append(int((pc & ~tc).sum())); fn.append(int((~pc & tc).sum()))
    return {'tp': tp, 'fp': fp, 'fn': fn}


def summarize(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    iou = np.divide(tp, tp + fp + fn, out=np.full(NC, np.nan), where=(tp + fp + fn) > 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(NC, np.nan), where=(2 * tp + fp + fn) > 0)
    rec = np.divide(tp, tp + fn, out=np.full(NC, np.nan), where=(tp + fn) > 0)
    prec = np.divide(tp, tp + fp, out=np.full(NC, np.nan), where=(tp + fp) > 0)
    names = ['background', 'alive', 'dead', 'bubble']
    out = {'mIoU': float(np.nanmean(iou) * 100), 'mDice': float(np.nanmean(dice) * 100)}
    for i, n in enumerate(names):
        out |= {f'iou_{n}': float(iou[i] * 100), f'dice_{n}': float(dice[i] * 100),
                f'recall_{n}': float(rec[i] * 100), f'precision_{n}': float(prec[i] * 100)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check-json', default=None)
    args = ap.parse_args()
    stems = [s.strip() for s in SPLIT.read_text().splitlines() if s.strip()]
    per = []
    for s in stems:
        pred = np.asarray(Image.open(Path(args.masks) / f'{s}.png'))
        gt = np.asarray(Image.open(VOC / 'SegmentationClass' / f'{s}.png'))
        assert pred.shape == gt.shape, s
        per.append({'stem': s, 'confusion_counts': counts(pred, gt)})
    tot = {k: np.sum([p['confusion_counts'][k] for p in per], axis=0) for k in ('tp', 'fp', 'fn')}
    res = {**summarize(tot['tp'], tot['fp'], tot['fn']), 'n_test_images': len(stems), 'masks': args.masks,
           'split': str(SPLIT), 'split_sha256': hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
           'evaluator': __file__, 'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'confusion_counts': {k: v.tolist() for k, v in tot.items()}, 'per_image': per}
    if args.check_json:
        ref = json.load(open(args.check_json))
        diffs = {k: abs(res[k] - ref[k]) for k in ('mIoU', 'iou_alive', 'iou_dead', 'iou_bubble')}
        same_counts = all(a['confusion_counts'] == b['confusion_counts'] for a, b in zip(per, ref['per_image']))
        res['consistency_check'] = {'reference': args.check_json, 'abs_diff_pp': diffs,
                                    'per_image_counts_identical': same_counts,
                                    'passes_0.01pp': all(v < 0.01 for v in diffs.values())}
        print(json.dumps(res['consistency_check'], indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print({k: round(res[k], 3) for k in ('mIoU', 'iou_alive', 'iou_dead', 'recall_dead', 'precision_dead')})


if __name__ == '__main__':
    main()
