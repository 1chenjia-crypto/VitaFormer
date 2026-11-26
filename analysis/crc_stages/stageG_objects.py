#!/usr/bin/env python3
"""Stage G object-level Dead evaluation (AMENDMENT_1 §3). Usage: stageG_objects.py OUTCSV ARM:DELTA [ARM:DELTA ...]
8-connected components of GT Dead and predicted Dead; one-to-one Hungarian matching maximising IoU; a pair counts iff
IoU >= tau. tau in {0.5 (primary), 0.3, 0.1}; minimum component area {0 (primary), 20} px applied to GT and prediction.
Per image, per seed: n_gt, n_pred, n_matched. Prediction at delta from the saved level map (Dead iff level > j).
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104')
sys.path.insert(0, str(EXP / 'scripts'))
import stageG_common as C  # noqa
EXPORT = EXP / 'stageG_dead_pr_val/export'
EIGHT = np.ones((3, 3), bool); TAUS = (0.5, 0.3, 0.1); MINA = (0, 20); SEEDS = (42, 3407, 2026)


def comps(mask, mina):
    lab, n = ndi.label(mask, structure=EIGHT)
    if n and mina:
        area = np.bincount(lab.ravel(), minlength=n + 1); keep = area >= mina; keep[0] = False
        remap = np.zeros(n + 1, np.int64); remap[keep] = np.arange(1, keep.sum() + 1); lab = remap[lab]; n = int(keep.sum())
    return lab, n


def match(gl, ng, pl, npd):
    if ng == 0 or npd == 0: return {t: 0 for t in TAUS}
    ga = np.bincount(gl.ravel(), minlength=ng + 1); pa = np.bincount(pl.ravel(), minlength=npd + 1)
    both = (gl > 0) & (pl > 0)
    code = gl[both].astype(np.int64) * (npd + 1) + pl[both]
    u, inter = np.unique(code, return_counts=True)
    gi, pi = u // (npd + 1), u % (npd + 1)
    iou = inter / (ga[gi] + pa[pi] - inter)
    # restrict Hungarian to components with any overlap (others can never match)
    G, P = np.unique(gi), np.unique(pi)
    M = np.zeros((len(G), len(P))); M[np.searchsorted(G, gi), np.searchsorted(P, pi)] = iou
    r, c = linear_sum_assignment(-M); v = M[r, c]
    return {t: int((v >= t).sum()) for t in TAUS}


def job(a):
    arm, seed, delta, stem = a
    j = int(np.where(np.isclose(C.MAIN, delta))[0][0])
    d = EXPORT / f'{arm}_seed{seed}'
    level = np.asarray(Image.open(d / 'level' / f'{stem}.png')); gt = C.load_gt(stem)
    pred = level > j
    valid = gt < C.NC
    rows = []
    for mina in MINA:
        gl, ng = comps((gt == 2) & valid, mina); pl, npd = comps(pred & valid, mina)
        m = match(gl, ng, pl, npd)
        for t in TAUS:
            rows.append({'arm': arm, 'seed': seed, 'delta': delta, 'stem': stem, 'min_area': mina, 'tau': t,
                         'n_gt': ng, 'n_pred': npd, 'n_matched': m[t]})
    return rows


def main():
    out = Path(sys.argv[1]); assert not out.exists(), out
    specs = [(x.split(':')[0], float(x.split(':')[1])) for x in sys.argv[2:]]
    stems = list(np.load(EXPORT / f'{specs[0][0]}_seed42/stats.npz')['stems'])
    jobs = [(a, s, d, st) for a, d in specs for s in SEEDS for st in stems]
    with Pool(24) as p:
        rows = [r for rr in p.imap_unordered(job, jobs, chunksize=4) for r in rr]
    pd.DataFrame(rows).to_csv(out, index=False); print('done', len(rows))


if __name__ == '__main__':
    main()
