#!/usr/bin/env python3
"""Layer 1 of ovit_recipe/ANALYSIS_PLAN.md: OrganoidViT-style mIoU(bg,fg) on existing protocol-B M2 predictions (no training).
Writes ovit_recipe/layer1/ only."""
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
E = Path(__file__).resolve().parents[1]; OUT = E / 'ovit_recipe/layer1'
X = Path('3D_OrganoidSeg/external_datasets')
def gt_orgaseg(s): return np.asarray(Image.open(X / f'OrgaSegment/OrganoidBasic_v20211206/eval/{s}_masks_organoid.png')) > 0
def gt_oid(sub): return lambda s: np.asarray(Image.open(X / f'OrganoID_Data/{sub}/testing/segmentations/{s}.png').resize((512, 512)).convert('1'))
SETS = [('OrgaSegment (human intestinal)', E / 'orgasegment/m2_B/test_masks', gt_orgaseg, None, 91.82, 0.76),
        ('OrganoID human PDAC', E / 'organoidid_human/m2_B/test_masks', gt_oid('OriginalData'), 'PDAC', 93.20, 0.74),
        ('OrganoID human all 28 (PDAC+ACC+Colon+Lung)', E / 'organoidid_human/m2_B/test_masks', gt_oid('OriginalData'), None, None, None),
        ('OrganoID mouse small intestinal', E / 'organoidid_mouse/m2_B/test_masks', gt_oid('MouseOrganoids'), None, 96.18, 0.70)]
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    rows, per = [], []
    for name, md, gtf, pref, ovit, src in SETS:
        cm = np.zeros((2, 2), np.int64); pi = []
        for f in sorted(md.glob('*.png')):
            if pref and not f.stem.startswith(pref): continue
            p = np.asarray(Image.open(f)) > 0; g = gtf(f.stem); assert p.shape == g.shape
            tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum(); tn = (~p & ~g).sum()
            cm += [[tn, fp], [fn, tp]]
            fg = tp / (tp + fp + fn); bg = tn / (tn + fp + fn); pi.append((fg, bg)); per.append(dict(set=name, image=f.stem, fg_iou=fg, bg_iou=bg))
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        fgp, bgp = tp / (tp + fp + fn), tn / (tn + fp + fn); pi = np.array(pi)
        rows.append(dict(set=name, n=len(pi), fg_iou_per_image=pi[:, 0].mean(), bg_iou_per_image=pi[:, 1].mean(),
                         miou_bgfg_per_image=pi.mean(1).mean(), fg_iou_pooled=fgp, bg_iou_pooled=bgp, miou_bgfg_pooled=(fgp + bgp) / 2,
                         organoidvit_table1_miou=ovit, source_paper_value_in_table1=src))
    OUT.mkdir(parents=True); pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False); pd.DataFrame(per).to_csv(OUT / 'per_image.csv', index=False)
    pd.set_option('display.width', 250); print(pd.DataFrame(rows).round(4).to_string())
if __name__ == '__main__': main()
