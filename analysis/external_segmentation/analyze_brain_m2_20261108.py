#!/usr/bin/env python3
"""BrainOrganoid2024 M2 five-fold analysis (ANALYSIS_PLAN_BASE_20261103 + AMENDMENT_M2 + AMENDMENT_2_NO_M0).

One evaluator for every arm (binary foreground, native resolution, GT = author .npy labels; empty-vs-empty = 1).
Primary: M2 - nnU-Net, image-level Dice over the 1,407 OOF images, paired organoid-ID cluster bootstrap (10,000, RNG 20261108).
Secondary/descriptive: author SegFormer-B0 OOF + classical tools (author df_pred.csv dice columns), M0/SegFormer-B2 fold0 only.
Writes into a new directory only.
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path('3D_OrganoidSeg'); BF = ROOT / 'baselines_fair_20261030'
D = BF / 'brain_organoid2024_m2_20261108'; OUT = D / 'analysis'
GT = ROOT / 'external_datasets/BrainOrganoid2024/data/labels'
SPLITS = D / 'official_splits.csv'
DF_PRED = Path('OrganoidSegBaseline/results/brain_organoid2024_metric_alignment_audit/source_cache/'
               'robust_monitoring_organoid_growth-main/df_pred.csv')
M2_DIRS = {f: d / 'test_masks' for f, d in {0: D / 'm2_rerun/fold0', 1: D / 'm2_rerun/fold1', 2: D / 'm2/fold2', 3: D / 'm2/fold3', 4: D / 'm2/fold4'}.items()}
NN = BF / 'nnunet/brain20261106_official_test_predictions'
CODEX = BF / 'brain_organoid2024_strong_baselines_20261104_gpu01_authorized'
B, RNG = 10000, 20261108


def metr(pred, ref):
    p, r = pred > 0, ref > 0
    tp = int((p & r).sum()); fp = int((p & ~r).sum()); fn = int((~p & r).sum())
    d = 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    i = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return dict(dice=d, iou=i, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), tp=tp, fp=fp, fn=fn)


def score(arm, mask_dir_by_fold, members):
    rows = []
    for f, cases in members.items():
        md = mask_dir_by_fold.get(f)
        if md is None: continue
        have = {p.stem for p in Path(md).glob('*.png')}
        if have != set(cases):
            raise RuntimeError(f'{arm} fold{f}: membership mismatch ({len(have)} vs {len(cases)})')
        for c in cases:
            pred = np.asarray(Image.open(Path(md) / f'{c}.png')); ref = np.load(GT / f'{c}.npy')
            if pred.shape != ref.shape: raise RuntimeError(f'{arm} {c} shape {pred.shape} vs {ref.shape}')
            rows.append(dict(arm=arm, fold=f, case=c, **metr(pred, ref)))
    return rows


def cluster_boot(diff_by_img, org_of_img, seed=RNG):
    orgs = sorted(set(org_of_img.values())); idx = {o: [k for k, v in org_of_img.items() if v == o] for o in orgs}
    sums = np.array([sum(diff_by_img[k] for k in idx[o]) for o in orgs]); cnts = np.array([len(idx[o]) for o in orgs])
    rng = np.random.default_rng(seed); est = []
    for _ in range(B):
        s = rng.integers(0, len(orgs), len(orgs)); est.append(sums[s].sum() / cnts[s].sum())
    est = np.array(est); obs = sums.sum() / cnts.sum()
    p = min(1.0, 2 * min((est <= 0).mean(), (est >= 0).mean()))
    return obs, float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), p, len(orgs)


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    sp = pd.read_csv(SPLITS); assert len(sp) == 1407
    members = {f: sp.loc[sp[f'split{f}'] == 'test', 'img_id'].tolist() for f in range(5)}
    meta = pd.read_csv(DF_PRED).set_index('img_id')
    rows = score('M2', M2_DIRS, members) + score('nnU-Net', {f: NN / f'fold{f}' for f in range(5)}, members)
    rows += score('M0 (fold0 only)', {0: CODEX / 'vitaformer/fold0/test_masks'}, members)
    sfb2 = CODEX / 'segformer_b2/fold0/test_masks'
    if sfb2.exists(): rows += score('SegFormer-B2 (fold0 only)', {0: sfb2}, members)
    df = pd.DataFrame(rows)
    for k in ('org_id', 'Imaging', 'Clone', 'Day'): df[k] = df['case'].map(meta[k])
    author = {'Author SegFormer-B0 OOF': 'pred_SegFormer_dice', 'Author OrganoSeg': 'pred_OrganoSeg_dice',
              'Author CellProfiler': 'pred_CellProfiler_dice', 'Author MOrgAna MLP_W': 'pred_MOrgAna_MLP_W_dice'}
    for arm, col in author.items():
        a = meta[[col, 'org_id', 'Imaging', 'Clone', 'Day']].reset_index().rename(columns={col: 'dice', 'img_id': 'case'})
        a['arm'] = arm; df = pd.concat([df, a], ignore_index=True)
    OUT.mkdir()
    df.to_csv(OUT / 'per_image_metrics.csv', index=False)
    summ = []
    for arm, g in df.groupby('arm', sort=False):
        d = g['dice'].to_numpy()
        r = dict(arm=arm, n=len(g), dice_mean=d.mean(), dice_sd=d.std(ddof=1), dice_median=np.median(d),
                 dice_q1=np.percentile(d, 25), dice_q3=np.percentile(d, 75), fail_lt_0_5=(d < 0.5).mean(), fail_lt_0_9=(d < 0.9).mean())
        if 'iou' in g and g['iou'].notna().all():
            r.update(iou_mean=g['iou'].mean(), precision_mean=g['precision'].mean(), recall_mean=g['recall'].mean(),
                     micro_dice=2 * g.tp.sum() / (2 * g.tp.sum() + g.fp.sum() + g.fn.sum()))
        for lab in ('LabA', 'LabB'): r[f'dice_{lab}'] = g.loc[g.Imaging == lab, 'dice'].mean()
        summ.append(r)
    pd.DataFrame(summ).to_csv(OUT / 'summary.csv', index=False)
    fold = df[df.arm.isin(['M2', 'nnU-Net'])].groupby(['arm', 'fold'])['dice'].agg(['count', 'mean', 'std']).reset_index()
    fold.to_csv(OUT / 'fold_summary.csv', index=False)
    piv = df.pivot_table(index='case', columns='arm', values='dice'); org = meta['org_id'].astype(str).to_dict()
    con = []
    for other, role in [('nnU-Net', 'primary'), ('Author SegFormer-B0 OOF', 'secondary'), ('Author OrganoSeg', 'descriptive'),
                        ('M0 (fold0 only)', 'descriptive fold0'), ('SegFormer-B2 (fold0 only)', 'descriptive fold0')]:
        if other not in piv: continue
        sub = piv[['M2', other]].dropna(); dif = (sub['M2'] - sub[other]).to_dict()
        obs, lo, hi, p, norg = cluster_boot(dif, {k: org[k] for k in dif})
        con.append(dict(contrast=f'M2 - {other}', role=role, n_images=len(sub), n_organoids=norg, delta=obs, ci_lo=lo, ci_hi=hi, p_boot=p))
        for lab in ('LabA', 'LabB'):
            ks = [k for k in dif if meta.loc[k, 'Imaging'] == lab]
            o2, l2, h2, p2, n2 = cluster_boot({k: dif[k] for k in ks}, {k: org[k] for k in ks})
            con.append(dict(contrast=f'M2 - {other} [{lab}]', role='secondary strata', n_images=len(ks), n_organoids=n2,
                            delta=o2, ci_lo=l2, ci_hi=h2, p_boot=p2))
    pd.DataFrame(con).to_csv(OUT / 'contrasts.csv', index=False)
    strata = df[df.arm.isin(['M2', 'nnU-Net', 'Author SegFormer-B0 OOF'])].groupby(['arm', 'Clone'])['dice'].mean().unstack(0)
    strata.to_csv(OUT / 'strata_clone.csv')
    df[df.arm.isin(['M2', 'nnU-Net', 'Author SegFormer-B0 OOF'])].groupby(['arm', 'Day'])['dice'].mean().unstack(0).to_csv(OUT / 'strata_day.csv')
    print(pd.DataFrame(summ).round(4).to_string()); print(pd.DataFrame(con).round(4).to_string()); print(fold.round(4).to_string())


if __name__ == '__main__':
    main()
