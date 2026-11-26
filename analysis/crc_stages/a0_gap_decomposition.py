#!/usr/bin/env python3
"""A0 step 2: decompose the M0 (Simplified VitaFormer) vs nnU-Net v2 gap by error location.

Primary (decision-relevant) split: CRC validation (164 images).
  M0: whole-image masks from a0_m0_val_inference.py (best_miou.pth, 3 seeds).
  nnU-Net: fold_0/validation masks written by nnU-Net itself (checkpoint_final, run2 seeds).
  Caveat: M0's checkpoint was chosen on (random-crop) validation; nnU-Net's final checkpoint was not -> the
  validation gap is, if anything, biased in M0's favour.
Descriptive only: CRC locked test (167 images), existing masks (P0-M simplified_seed*, pred_run2_seed*_noTTA).
  Test results are NOT used for any Go/No-Go decision.

Metric convention = shared evaluator: pooled confusion counts over images, per-class IoU, mIoU over 4 classes
(background, alive, dead, bubble), valid GT pixels 0..3. Per seed, then mean over 3 seeds. M0 seed s is paired
with nnU-Net seed s (42/3407/2026); pairing is arbitrary, so all 9 pairs are also summarised.

Region-swap decomposition: hybrid = M0 prediction, with nnU-Net's prediction inside region R.
  share(R) = (mIoU(hybrid_R) - mIoU(M0)) / (mIoU(nnU-Net) - mIoU(M0)).
Regions (defined from GT only):
  B3 / B8 : pixels within 3 / 8 px of a GT label transition (any class change, 8-neighbourhood)
  INT8    : complement of B8 (interiors and far background)
  SMALL / MID / LARGE : GT organoid components (alive U dead, 8-connected, >=20 px) split into area tertiles
             (tertile edges from the validation GT components), each component dilated by 8 px
  FARBG   : pixels > 8 px away from any GT non-background pixel (false-positive blobs in background)
Object level: per GT alive / dead component (>=20 px), "missed" if <10 % of its pixels get the correct class,
"detected" if >=50 %.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path('3D_OrganoidSeg')
VOC = Path('OrganoidSegBaseline/VOC2012')
GT_DIR = VOC / 'SegmentationClass'
OUT = ROOT / 'vco/runs/arch_exploration_20261104/A0_gap_decomposition'
SEEDS = [42, 3407, 2026]
NC = 4
CLASSES = ['background', 'alive', 'dead', 'bubble']
NN = ROOT / 'baselines_fair_20261030/nnunet'
SRC = {
    'val': {'stems': VOC / 'ImageSets/Segmentation/val_cond.txt',
            'm0': lambda s: OUT / f'm0_val_masks/seed{s}',
            'nn': lambda s: NN / f'nnUNet_results_run2_seed{s}/Dataset801_CRCOrganoid/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation'},
    'test': {'stems': VOC / 'ImageSets/Segmentation/test_cond.txt',
             'm0': lambda s: ROOT / f'vco/runs/module_endpoints_multiseed_20261030/simplified_seed{s}/masks',
             'nn': lambda s: NN / f'pred_run2_seed{s}_noTTA'},
}
REGIONS = ['ALL', 'B3', 'B8', 'INT8', 'SMALL', 'MID', 'LARGE', 'FARBG']
EIGHT = np.ones((3, 3), bool)


def load(p):
    return np.asarray(Image.open(p))


def gt_regions(gt, edges):
    valid = gt < NC
    g = np.where(valid, gt, 0)
    # label transitions in an 8-neighbourhood
    trans = ndi.maximum_filter(g, size=3) != ndi.minimum_filter(g, size=3)
    dist = ndi.distance_transform_edt(~trans) if trans.any() else np.full(g.shape, np.inf)
    reg = {'ALL': valid, 'B3': valid & (dist <= 3), 'B8': valid & (dist <= 8)}
    reg['INT8'] = valid & ~reg['B8']
    nonbg = g > 0
    dist_fg = ndi.distance_transform_edt(~nonbg) if nonbg.any() else np.full(g.shape, np.inf)
    reg['FARBG'] = valid & (dist_fg > 8)
    org = (g == 1) | (g == 2)
    lab, n = ndi.label(org, structure=EIGHT)
    areas = ndi.sum(org, lab, index=np.arange(1, n + 1)) if n else np.array([])
    comp_areas = []
    for name in ['SMALL', 'MID', 'LARGE']:
        reg[name] = np.zeros(g.shape, bool)
    for k, a in enumerate(areas, 1):
        if a < 20:
            continue
        comp_areas.append(float(a))
        t = 'SMALL' if a < edges[0] else ('MID' if a < edges[1] else 'LARGE')
        reg[t] |= lab == k
    for name in ['SMALL', 'MID', 'LARGE']:
        if reg[name].any():
            reg[name] = valid & ndi.binary_dilation(reg[name], structure=EIGHT, iterations=8)
    return reg, comp_areas


def conf(pred, gt, mask):
    p = pred[mask].astype(np.int64)
    t = gt[mask].astype(np.int64)
    cm = np.bincount(t * NC + p, minlength=NC * NC).reshape(NC, NC)  # rows=gt, cols=pred
    return cm


def objects(pred, gt, cls):
    lab, n = ndi.label(gt == cls, structure=EIGHT)
    rows = []
    if not n:
        return rows
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(lab), lab, idx)
    corr = ndi.sum(pred == cls, lab, idx)
    for a, c in zip(area, corr):
        if a >= 20:
            rows.append((float(a), float(c / a)))
    return rows


def per_image(args):
    split, stem, edges = args
    gt = load(GT_DIR / f'{stem}.png')
    reg, comp_areas = gt_regions(gt, edges)
    out = {'stem': stem, 'comp_areas': comp_areas, 'region_px': {r: int(reg[r].sum()) for r in REGIONS}, 'seeds': {}}
    for s in SEEDS:
        m0 = load(SRC[split]['m0'](s) / f'{stem}.png')
        nn = load(SRC[split]['nn'](s) / f'{stem}.png')
        assert m0.shape == gt.shape == nn.shape, (stem, m0.shape, nn.shape, gt.shape)
        r = {'m0': {k: conf(m0, gt, reg[k]).tolist() for k in REGIONS},
             'nn': {k: conf(nn, gt, reg[k]).tolist() for k in REGIONS},
             'err_px': {mod: {k: int(((p != gt) & reg[k]).sum()) for k in REGIONS} for mod, p in (('m0', m0), ('nn', nn))},
             'obj': {mod: {c: objects(p, gt, ci) for ci, c in ((1, 'alive'), (2, 'dead'))} for mod, p in (('m0', m0), ('nn', nn))}}
        out['seeds'][s] = r
    return out


def ious(cm):
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = tp + fp + fn
    iou = np.divide(tp, den, out=np.full(NC, np.nan), where=den > 0) * 100
    return iou


def summarise(split, recs):
    rows, shares, errloc, objrows = [], [], [], []
    pairs = [(s, s) for s in SEEDS]
    allpairs = [(a, b) for a in SEEDS for b in SEEDS]
    def tot(model, seed, region):
        return np.sum([np.asarray(r['seeds'][seed][model][region]) for r in recs], axis=0)
    for s in SEEDS:
        for model in ('m0', 'nn'):
            iou = ious(tot(model, s, 'ALL'))
            rows.append({'split': split, 'model': model, 'seed': s, 'mIoU': np.nanmean(iou), **{f'iou_{c}': iou[i] for i, c in enumerate(CLASSES)}})
    for kind, plist in (('matched', pairs), ('all9', allpairs)):
        for (sm, sn) in plist:
            m0_all = tot('m0', sm, 'ALL')
            nn_all = tot('nn', sn, 'ALL')
            base = ious(m0_all)
            full = ious(nn_all)
            for R in REGIONS[1:]:
                hyb = m0_all - tot('m0', sm, R) + tot('nn', sn, R)
                h = ious(hyb)
                rec = {'split': split, 'pairing': kind, 'm0_seed': sm, 'nn_seed': sn, 'region': R}
                for i, c in enumerate(['mIoU'] + CLASSES):
                    if c == 'mIoU':
                        b, f, x = np.nanmean(base), np.nanmean(full), np.nanmean(h)
                    else:
                        b, f, x = base[i - 1], full[i - 1], h[i - 1]
                    rec[f'gap_{c}'] = f - b
                    rec[f'gain_{c}'] = x - b
                    rec[f'share_{c}'] = (x - b) / (f - b) if abs(f - b) > 1e-9 else np.nan
                shares.append(rec)
    for s in SEEDS:
        for model in ('m0', 'nn'):
            e = {k: sum(r['seeds'][s]['err_px'][model][k] for r in recs) for k in REGIONS}
            errloc.append({'split': split, 'model': model, 'seed': s, 'err_total': e['ALL'],
                           **{f'frac_err_in_{k}': e[k] / e['ALL'] for k in REGIONS[1:]}})
    edges = None
    for s in SEEDS:
        for model in ('m0', 'nn'):
            for c in ('alive', 'dead'):
                objs = [o for r in recs for o in r['seeds'][s]['obj'][model][c]]
                if not objs:
                    continue
                a = np.array(objs)
                q = np.quantile(a[:, 0], [1 / 3, 2 / 3])
                for name, m in (('all', np.ones(len(a), bool)), ('small', a[:, 0] < q[0]),
                                ('mid', (a[:, 0] >= q[0]) & (a[:, 0] < q[1])), ('large', a[:, 0] >= q[1])):
                    sub = a[m]
                    objrows.append({'split': split, 'model': model, 'seed': s, 'class': c, 'size': name,
                                    'n_obj': int(m.sum()), 'missed_frac(<10%)': float((sub[:, 1] < 0.1).mean()) if len(sub) else np.nan,
                                    'detected_frac(>=50%)': float((sub[:, 1] >= 0.5).mean()) if len(sub) else np.nan,
                                    'mean_coverage': float(sub[:, 1].mean()) if len(sub) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(shares), pd.DataFrame(errloc), pd.DataFrame(objrows)


def val_edges():
    stems = [s.strip() for s in SRC['val']['stems'].read_text().splitlines() if s.strip()]
    areas = []
    for st in stems:
        g = load(GT_DIR / f'{st}.png')
        org = (g == 1) | (g == 2)
        lab, n = ndi.label(org, structure=EIGHT)
        if n:
            a = ndi.sum(org, lab, np.arange(1, n + 1))
            areas += [float(x) for x in a if x >= 20]
    return list(np.quantile(areas, [1 / 3, 2 / 3])), len(areas)


def main():
    edges, ncomp = val_edges()
    print('val GT organoid component area tertile edges (px):', edges, 'n components', ncomp, flush=True)
    meta = {'size_tertile_edges_px_from_val_gt': edges, 'n_val_components': ncomp}
    tabs = {k: [] for k in ('iou', 'share', 'errloc', 'obj')}
    for split in ('val', 'test'):
        stems = [s.strip() for s in SRC[split]['stems'].read_text().splitlines() if s.strip()]
        with Pool(24) as pool:
            recs = pool.map(per_image, [(split, st, edges) for st in stems], chunksize=2)
        meta[f'n_{split}'] = len(recs)
        meta[f'region_px_{split}'] = {k: int(sum(r['region_px'][k] for r in recs)) for k in REGIONS}
        for k, df in zip(tabs, summarise(split, recs)):
            tabs[k].append(df)
        print(split, 'done', flush=True)
    for k in tabs:
        pd.concat(tabs[k]).to_csv(OUT / f'a0_{k}.csv', index=False)
    (OUT / 'a0_meta.json').write_text(json.dumps(meta, indent=2))
    print('written', flush=True)


if __name__ == '__main__':
    main()
