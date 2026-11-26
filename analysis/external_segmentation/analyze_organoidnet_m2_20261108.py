#!/usr/bin/env python3
"""OrganoIDNetData protocol-B analysis (ANALYSIS_PLAN.md, approved 2026-11-08). Writes a new directory only.
Step 1 sanity: re-implemented author metrics on author StarDist/Cellpose masks must reproduce image_metrics.csv (3 dp).
Step 2: M2 (and nnU-Net if present) scored with the same evaluator; paired species-stratified bootstrap (10,000, RNG 20261108)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from skimage.measure import label

D = Path(__file__).resolve().parent; OUT = D / 'analysis_B'
R = Path('3D_OrganoidSeg/external_datasets/OrganoIDNetData/OrganoIDNetData')
GTD = R / 'Dataset/Test/Masks'; SC = R / 'Source_Codes'
B, RNG = 10000, 20261108
ARMS = {'StarDist (author)': lambda s: SC / f'StarDist/AlgorithmPredictions/{s}_StarDistMask.tif',
        'Cellpose (author)': lambda s: SC / f'Cellpose/AlgorithmPredictions/{s}_CellposeMask.tif',
        'M2': lambda s: D / f'm2_B/test_masks/{s}.png'}
if (D / 'nnunet_B/test_predictions').exists():
    ARMS['nnU-Net'] = lambda s: D / f'nnunet_B/test_predictions/{s}.png'

def m(gt, p):
    inter = np.logical_and(gt, p).sum(); uni = np.logical_or(gt, p).sum()
    tp, fp, fn = inter, (p & ~gt).sum(), (~p & gt).sum()
    return dict(iou=inter / uni, dice=2. * inter / (gt.sum() + p.sum()), pix_acc=(gt == p).mean(),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), tp=int(tp), fp=int(fp), fn=int(fn))

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    stems = sorted(p.name[:-len('_mask.tif')] for p in GTD.glob('*_mask.tif')); assert len(stems) == 10, stems
    ref = pd.read_csv(D / 'author_reference/image_metrics.csv'); ref['stem'] = ref.Image.str.replace('.tif', '', regex=False)
    rows = []
    for s in stems:
        gti = np.array(Image.open(GTD / f'{s}_mask.tif')); gt = gti > 0
        n_gt = len(np.unique(gti[gti > 0]))
        for arm, f in ARMS.items():
            p = np.array(Image.open(f(s))) > 0
            if p.ndim == 3: p = p.any(-1)
            if p.shape != gt.shape: raise RuntimeError(f'{arm} {s} shape {p.shape} vs {gt.shape}')
            n_pred = int(label(p, connectivity=2).max())
            rows.append(dict(arm=arm, image=s, species='human' if 'Human' in s else 'mouse', n_gt=n_gt, n_pred_cc=n_pred,
                             count_rel_err=(n_pred - n_gt) / n_gt, **m(gt, p)))
    df = pd.DataFrame(rows)
    # sanity check
    bad = []
    for arm, pre in [('StarDist (author)', 'StarDist'), ('Cellpose (author)', 'Cellpose')]:
        for _, r in ref.iterrows():
            o = df[(df.arm == arm) & (df.image == r.stem)].iloc[0]
            for k, col in [('iou', 'IoU'), ('dice', 'Dice Coefficient'), ('pix_acc', 'Pixel Accuracy')]:
                if round(o[k], 3) != round(r[f'{pre} {col}'], 3): bad.append((arm, r.stem, k, o[k], r[f'{pre} {col}']))
    print('SANITY mismatches:', len(bad)); [print(b) for b in bad]
    if bad: sys.exit('sanity check failed: stop, fix evaluator before scoring models')
    OUT.mkdir(); df.to_csv(OUT / 'per_image_metrics.csv', index=False)
    summ = []
    for arm, g in df.groupby('arm', sort=False):
        r = dict(arm=arm, n=len(g))
        for k in ('iou', 'dice', 'pix_acc', 'precision', 'recall', 'count_rel_err'):
            r[f'{k}_mean'] = g[k].mean(); r[f'{k}_sd'] = g[k].std(ddof=1)
        r['abs_count_rel_err_mean'] = g.count_rel_err.abs().mean()
        r['micro_iou'] = g.tp.sum() / (g.tp.sum() + g.fp.sum() + g.fn.sum())
        for sp in ('human', 'mouse'): r[f'iou_{sp}'] = g.loc[g.species == sp, 'iou'].mean(); r[f'dice_{sp}'] = g.loc[g.species == sp, 'dice'].mean()
        summ.append(r)
    summ = pd.DataFrame(summ); summ.to_csv(OUT / 'summary.csv', index=False)
    con = []
    others = [('nnU-Net', 'primary')] if 'nnU-Net' in ARMS else []
    others += [('Cellpose (author)', 'secondary'), ('StarDist (author)', 'secondary')]
    for other, role in others:
        for k in ('iou', 'dice'):
            piv = df.pivot(index='image', columns='arm', values=k); dif = piv['M2'] - piv[other]
            sp = np.array(['human' if 'Human' in i else 'mouse' for i in dif.index]); dv = dif.to_numpy()
            groups = [dv[sp == 'human'], dv[sp == 'mouse']]
            rng = np.random.default_rng(RNG); est = np.empty(B)
            for b in range(B):
                est[b] = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]).mean()
            p = min(1.0, 2 * min((est <= 0).mean(), (est >= 0).mean()))
            con.append(dict(contrast=f'M2 - {other}', metric=k, role=role, n=len(dv), delta=dv.mean(),
                            ci_lo=np.percentile(est, 2.5), ci_hi=np.percentile(est, 97.5), p_boot=p,
                            wins_M2=int((dv > 0).sum()), delta_human=groups[0].mean(), delta_mouse=groups[1].mean()))
    con = pd.DataFrame(con); con.to_csv(OUT / 'contrasts.csv', index=False)
    pd.set_option('display.width', 250)
    print(summ.round(4).T.to_string()); print(con.round(4).to_string())
    print(df[['arm', 'image', 'iou', 'dice', 'precision', 'recall', 'n_gt', 'n_pred_cc']].round(3).to_string())

if __name__ == '__main__':
    main()
