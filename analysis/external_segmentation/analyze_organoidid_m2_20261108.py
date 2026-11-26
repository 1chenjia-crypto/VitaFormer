#!/usr/bin/env python3
"""OrganoID human/mouse protocol-B analysis (organoidid_*/ANALYSIS_PLAN.md). Author metric: per-image binary IoU in 512x512 space,
GT = resize(512).convert("1"); SD reported both as numpy population SD (author) and sample SD. New directory only.
Usage: python analyze_organoidid_m2_20261108.py organoidid_human|organoidid_mouse"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

E = Path(__file__).resolve().parent; ds = sys.argv[1]; D = E / ds; OUT = D / 'analysis_B'
base = Path('3D_OrganoidSeg/external_datasets/OrganoID_Data') / {'organoidid_human': 'OriginalData', 'organoidid_mouse': 'MouseOrganoids'}[ds] / 'testing'
B, RNG = 10000, 20261108

def grp(s):
    if ds == 'organoidid_mouse': return 'mouse'
    for k in ('PDAC', 'ACC', 'Lung'):
        if s.startswith(k): return k
    return 'Colon' if s.startswith('C') else '?'

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    stems = sorted(p.stem for p in (base / 'segmentations').glob('*.png'))
    res = {r['image']: r for r in json.load(open(D / 'm2_B/result.json'))['per_image']}
    rows = []; bad = 0
    for s in stems:
        gt = np.asarray(Image.open(base / 'segmentations' / f'{s}.png').resize((512, 512)).convert('1'))
        assert (gt & gt).sum() / max((gt | gt).sum(), 1) == 1 or gt.sum() == 0
        p = np.asarray(Image.open(D / f'm2_B/test_masks/{s}.png')) > 0
        if p.shape != gt.shape: raise RuntimeError(f'{s} {p.shape} vs {gt.shape}')
        i = (p & gt).sum(); u = (p | gt).sum(); tp, fp, fn = i, (p & ~gt).sum(), (~p & gt).sum()
        iou = i / u
        if abs(iou - res[s]['iou']) > 1e-6: bad += 1
        rows.append(dict(image=s, group=grp(s), iou=iou, dice=2 * i / (p.sum() + gt.sum()), precision=tp / max(tp + fp, 1),
                         recall=tp / max(tp + fn, 1), gt_fg=gt.mean()))
    print('evaluator vs result.json mismatches:', bad)
    if bad: sys.exit('stop: evaluator mismatch')
    df = pd.DataFrame(rows); OUT.mkdir(); df.to_csv(OUT / 'per_image_metrics.csv', index=False)
    df['pdac'] = np.where(df.group == 'PDAC', 'PDAC', 'non-PDAC') if ds == 'organoidid_human' else 'mouse'
    out = []
    for key in (['group', 'pdac'] if ds == 'organoidid_human' else ['group']):
        for g, x in df.groupby(key):
            out.append(dict(stratum=f'{key}={g}', n=len(x), iou_mean=x.iou.mean(), iou_sd_pop=x.iou.std(ddof=0), iou_sd=x.iou.std(ddof=1) if len(x) > 1 else np.nan,
                            dice_mean=x.dice.mean(), precision=x.precision.mean(), recall=x.recall.mean()))
    out.append(dict(stratum='all', n=len(df), iou_mean=df.iou.mean(), iou_sd_pop=df.iou.std(ddof=0), iou_sd=df.iou.std(ddof=1),
                    dice_mean=df.dice.mean(), precision=df.precision.mean(), recall=df.recall.mean()))
    s = pd.DataFrame(out); s.to_csv(OUT / 'summary.csv', index=False)
    pd.set_option('display.width', 200); print(s.round(3).to_string()); print(df.round(3).to_string())

if __name__ == '__main__':
    main()
