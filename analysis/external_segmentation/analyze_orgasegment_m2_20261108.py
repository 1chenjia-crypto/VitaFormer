#!/usr/bin/env python3
"""OrgaSegment protocol-B analysis (ANALYSIS_PLAN.md, approved 2026-11-08). New directory only.
Semantic -> instances: skimage.measure.label(connectivity=2), no min area. AP = archived, unmodified author lib/metrics.py.
Sanity: author average_precision(GT, GT) == 1 for all thresholds and all 12 images, else stop."""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from skimage.measure import label
from skimage.segmentation import relabel_sequential

D = Path(__file__).resolve().parent; OUT = D / 'analysis_B'
EV = Path('3D_OrganoidSeg/external_datasets/OrgaSegment/OrganoidBasic_v20211206/eval')
spec = importlib.util.spec_from_file_location('author_metrics', D / 'author_code/metrics.py')
am = importlib.util.module_from_spec(spec); spec.loader.exec_module(am)
TH = np.round(np.arange(0.5, 1.0001, 0.05), 2)  # author EvalConfig 0.50:0.05:1.00
B, RNG = 10000, 20261108
ARMS = {'M2': D / 'm2_B/test_masks'}
if (D / 'nnunet_B/test_predictions').exists(): ARMS['nnU-Net'] = D / 'nnunet_B/test_predictions'

def merged_rate(gt, pred_inst):
    """GT instances with > 50% of their area inside one predicted component that also holds > 50% of >= 1 other GT instance."""
    ids = np.unique(gt[gt > 0]); owner = {}
    for g in ids:
        pv = pred_inst[gt == g]; pv = pv[pv > 0]
        if pv.size == 0: continue
        c, n = np.unique(pv, return_counts=True); k = n.argmax()
        if n[k] > 0.5 * (gt == g).sum(): owner[g] = c[k]
    from collections import Counter
    cnt = Counter(owner.values())
    return sum(1 for g in owner if cnt[owner[g]] >= 2) / len(ids)

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    stems = sorted(p.name[:-len('_masks_organoid.png')] for p in EV.glob('*_masks_organoid.png')); assert len(stems) == 12
    # GT relabelled 1..N exactly as the author pipeline does (io.load_mask: one layer per present id -> utils.mask_projection count+1);
    # needed because one eval PNG has a gap in its ids (86 instances, max id 87). Evaluator-only fix, before any model was scored.
    gts = {s: relabel_sequential(np.array(Image.open(EV / f'{s}_masks_organoid.png')).astype(np.int32))[0] for s in stems}
    for s, g in gts.items(): assert g.ndim == 2, (s, g.shape)
    ap_gt = am.average_precision([gts[s] for s in stems], [gts[s] for s in stems], threshold=list(TH))[0]
    print('SANITY GT-vs-GT AP min =', ap_gt.min())
    if not np.allclose(ap_gt, 1.0): sys.exit('sanity failed')
    rows = []
    for arm, md in ARMS.items():
        have = {p.stem for p in md.glob('*.png')}
        if have != set(stems): raise RuntimeError(f'{arm}: membership mismatch {len(have)}')
        for s in stems:
            p = np.array(Image.open(md / f'{s}.png')) > 0; g = gts[s]
            if p.shape != g.shape: raise RuntimeError(f'{arm} {s} {p.shape} vs {g.shape}')
            inst = label(p, connectivity=2).astype(np.int32)
            ap, tp, fp, fn = am.average_precision(g, inst, threshold=list(TH))
            gb = g > 0; i = (p & gb).sum()
            r = dict(arm=arm, image=s, size=g.shape[1], n_gt=len(np.unique(g[g > 0])), n_pred=int(inst.max()),
                     ap50=ap[0], ap75=ap[list(TH).index(0.75)], map50_95=ap[:10].mean(),
                     tp50=int(tp[0]), fp50=int(fp[0]), fn50=int(fn[0]),
                     dice=2 * i / (p.sum() + gb.sum()), iou=i / (p | gb).sum(), merged_rate=merged_rate(g, inst))
            r['count_abs_rel_err'] = abs(r['n_pred'] - r['n_gt']) / r['n_gt']
            for t, a in zip(TH, ap): r[f'ap_{t:.2f}'] = a
            rows.append(r)
    df = pd.DataFrame(rows); OUT.mkdir(); df.to_csv(OUT / 'per_image_metrics.csv', index=False)
    keys = ['ap50', 'ap75', 'map50_95', 'dice', 'iou', 'count_abs_rel_err', 'merged_rate']
    summ = df.groupby('arm', sort=False)[keys].agg(['mean', 'std']); summ.to_csv(OUT / 'summary.csv')
    df.groupby(['arm', 'size'])[['ap50', 'dice']].agg(['count', 'mean']).to_csv(OUT / 'strata_size.csv')
    con = []
    if 'nnU-Net' in ARMS:
        for k in ('ap50', 'map50_95', 'dice'):
            piv = df.pivot(index='image', columns='arm', values=k); dv = (piv['M2'] - piv['nnU-Net']).to_numpy()
            rng = np.random.default_rng(RNG); est = np.array([dv[rng.integers(0, 12, 12)].mean() for _ in range(B)])
            con.append(dict(contrast='M2 - nnU-Net', metric=k, delta=dv.mean(), ci_lo=np.percentile(est, 2.5),
                            ci_hi=np.percentile(est, 97.5), p_boot=min(1, 2 * min((est <= 0).mean(), (est >= 0).mean()))))
        pd.DataFrame(con).to_csv(OUT / 'contrasts.csv', index=False)
    pd.set_option('display.width', 250)
    print(summ.round(4).to_string()); print(df.groupby(['arm', 'size'])[['ap50', 'dice']].agg(['count', 'mean']).round(3))
    print(df[['arm', 'image', 'size', 'n_gt', 'n_pred', 'ap50', 'ap75', 'dice', 'merged_rate']].round(3).to_string())
    if con: print(pd.DataFrame(con).round(4))

if __name__ == '__main__':
    main()
