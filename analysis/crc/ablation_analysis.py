#!/usr/bin/env python3
"""M2 ablations 2026-11-13 (PLAN.md in m2_ablation_no_rare_dead_20261112 / m2_ablation_imagenet_20261112): copy of
stageD_analysis.py with ONLY mask sources and comparison pairs changed (M2-noRD, M2-ImageNet; final.pth). Original header:
Stage D (Amendment 4, POST HOC) = copy of stageC_analysis.py with --ckpt best|final mask source.
Stage C analysis (Amendment 3, written 2026-11-05 before any Stage C result). VALIDATION only (164 images).

Models: M0 (A0 masks), M1/M2 (Stage B/C eval masks), nnU-Net (own fold_0/validation; descriptive only).
Per model x seed: pooled IoU / precision / recall, fg + alive BF1 (1/2/4/8 px), far-background Dead FP pixels,
object detection / miss rates by size (A0 edges), region errors.
Pairs (seed-matched): M2-M1, M1-M0, M2-M0 (and M2-nnU-Net, descriptive).
  * per seed: 154-well cluster paired bootstrap (10,000) -- images in a well are not independent;
  * pooled over seeds: hierarchical bootstrap = resample wells, average the 3 seed-matched deltas (seed set fixed,
    so CIs do not include seed-to-seed training variance beyond the 3 seeds observed).
Missing models are skipped (pipeline can run before Stage C finishes). Outputs -> --out.
"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104')
sys.path.insert(0, str(EXP / 'scripts'))
import stageB_evaluate as B  # noqa: E402  (metric functions reused unchanged)
from a0_gap_decomposition import gt_regions, conf, REGIONS  # noqa: E402

SEEDS = [42, 3407, 2026]
NN = Path('3D_OrganoidSeg/baselines_fair_20261030/nnunet')


CKPT = "best"


ABL = Path(__file__).resolve().parent / "final_masks"


def mask_dir(model, s):
    if model in ("noRD", "ImageNet"):
        return ABL / f"{model}_seed{s}"
    if CKPT == "final" and model != "nnUNet":
        return EXP / f"stageD_selection_free/final_masks/{model}_seed{s}"
    if model == 'M0':
        return EXP / f'A0_gap_decomposition/m0_val_masks/seed{s}'
    if model == 'nnUNet':
        return NN / f'nnUNet_results_run2_seed{s}/Dataset801_CRCOrganoid/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation'
    if model == 'M2' and s == 42:
        return EXP / 'stageB_m2_seed42_eval/masks'
    return EXP / f'stageC_{model.lower()}_seed{s}_eval/masks'


def one(args):
    stem, srcs = args
    raw = np.asarray(Image.open(B.GT_DIR / f'{stem}.png'))
    gtv = raw.astype(np.int16); gtv[(gtv >= 250) | (gtv > 3)] = -1
    reg, _ = gt_regions(raw, B.EDGES)
    out = {'stem': stem, 'well': B.parse_filename(stem)['well_id']}
    valid = gtv >= 0
    for tag, d in srcs:
        pred = np.asarray(Image.open(Path(d) / f'{stem}.png'), dtype=np.int16)
        out[f'{tag}|cm'] = np.bincount(gtv[valid] * B.NC + pred[valid], minlength=B.NC ** 2).reshape(B.NC, B.NC)
        fg = B.bmetrics(np.isin(gtv, (1, 2)), np.isin(pred, (1, 2)))
        al = B.bmetrics(gtv == 1, pred == 1)
        out |= {f'{tag}|fg_{k}': v for k, v in fg.items()} | {f'{tag}|alive_{k}': v for k, v in al.items()}
        out[f'{tag}|obj'] = {c: B.objects(pred, gtv, ci) for ci, c in ((1, 'alive'), (2, 'dead'))}
        out[f'{tag}|far_dead_fp'] = int(((pred == 2) & (gtv != 2) & reg['FARBG']).sum())
        out[f'{tag}|regerr'] = {k: int((lambda c: c.sum() - np.trace(c))(conf(pred.astype(np.uint8), raw, reg[k]))) for k in REGIONS}
    return out


IMG_KEYS = ['fg_bf1_2', 'alive_bf1_2']
POOL_KEYS = ['mIoU', 'iou_alive', 'iou_dead', 'iou_bubble', 'dead_precision', 'dead_recall']


def well_arrays(recs, tag, wi, nw):
    cm = np.zeros((nw, B.NC, B.NC)); np.add.at(cm, wi, np.stack([r[f'{tag}|cm'] for r in recs]))
    iv = {k: np.bincount(wi, np.array([r[f'{tag}|{k}'] for r in recs]), nw) for k in IMG_KEYS}
    return cm, iv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--ckpt', choices=['best', 'final'], required=True); ap.add_argument('--n_boot', type=int, default=10000)
    a = ap.parse_args(); global CKPT; CKPT = a.ckpt; out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    stems = [s.strip() for s in B.VAL.read_text().splitlines() if s.strip()]
    srcs = [(f'{m}_s{s}', mask_dir(m, s)) for m in ('M2', 'noRD', 'ImageNet') for s in SEEDS
            if mask_dir(m, s).is_dir() and len(list(mask_dir(m, s).glob('*.png'))) >= len(stems)]
    print('available:', [t for t, _ in srcs], flush=True)
    with Pool(24) as pool:
        recs = pool.map(one, [(s, srcs) for s in stems], chunksize=2)
    tags = [t for t, _ in srcs]
    # per model x seed metrics
    rows, objrows, regrows = [], [], []
    for t in tags:
        m, s = t.split('_s')
        row = {'model': m, 'seed': int(s)} | B.summ(np.sum([r[f'{t}|cm'] for r in recs], axis=0))
        for k in [f'fg_bf1_{x}' for x in B.TOLS] + [f'alive_bf1_{x}' for x in B.TOLS]:
            row[k] = float(np.mean([r[f'{t}|{k}'] for r in recs]))
        for k in ('fg_hd95', 'fg_assd', 'alive_hd95', 'alive_assd'):
            row[k] = float(np.nanmean([r[f'{t}|{k}'] for r in recs]))
        row['far_bg_dead_fp_px'] = int(sum(r[f'{t}|far_dead_fp'] for r in recs))
        rows.append(row)
        for o in B.obj_table([{f'x_obj': r[f'{t}|obj']} for r in recs], 'x'):
            objrows.append({'model': m, 'seed': int(s)} | {k: v for k, v in o.items() if k != 'model'})
        for k in REGIONS:
            regrows.append({'model': m, 'seed': int(s), 'region': k, 'error_px': sum(r[f'{t}|regerr'][k] for r in recs)})
    met = pd.DataFrame(rows); met.to_csv(out / 'per_model_seed_metrics.csv', index=False)
    num = met.drop(columns='seed').groupby('model')
    pd.concat({'mean': num.mean(), 'min': num.min(), 'max': num.max(), 'sd': num.std()}, axis=1).to_csv(out / 'per_model_summary.csv')
    obj = pd.DataFrame(objrows); obj.to_csv(out / 'objects_per_model_seed.csv', index=False)
    pd.DataFrame(regrows).to_csv(out / 'region_errors_per_model_seed.csv', index=False)
    # paired comparisons
    wells = sorted({r['well'] for r in recs}); wi = np.array([wells.index(r['well']) for r in recs]); nw = len(wells)
    wcnt = np.bincount(wi, minlength=nw).astype(float)
    W = {t: well_arrays(recs, t, wi, nw) for t in tags}
    rng = np.random.default_rng(20261105)
    IDX = rng.integers(0, nw, (a.n_boot, nw))

    def delta(t1, t0, idx):
        c1, i1 = W[t1]; c0, i0 = W[t0]
        s1, s0 = B.summ(c1[idx].sum(0)), B.summ(c0[idx].sum(0))
        d = {k: s1[k] - s0[k] for k in POOL_KEYS}
        d |= {k: (i1[k][idx].sum() - i0[k][idx].sum()) / wcnt[idx].sum() for k in IMG_KEYS}
        return d
    prow = []
    for A_, B_ in (('M2', 'noRD'), ('M2', 'ImageNet')):
        seeds = [s for s in SEEDS if f'{A_}_s{s}' in W and f'{B_}_s{s}' in W]
        if not seeds:
            continue
        allidx = np.arange(nw)
        per = {s: [delta(f'{A_}_s{s}', f'{B_}_s{s}', IDX[b]) for b in range(a.n_boot)] for s in seeds}
        for s in seeds:
            pt = delta(f'{A_}_s{s}', f'{B_}_s{s}', allidx)
            for k in pt:
                dr = np.array([d[k] for d in per[s]])
                prow.append({'comparison': f'{A_}-{B_}', 'seed': s, 'endpoint': k, 'delta': pt[k],
                             'ci_lo': np.nanpercentile(dr, 2.5), 'ci_hi': np.nanpercentile(dr, 97.5)})
        for k in POOL_KEYS + IMG_KEYS:
            pt = np.mean([delta(f'{A_}_s{s}', f'{B_}_s{s}', allidx)[k] for s in seeds])
            dr = np.mean([[d[k] for d in per[s]] for s in seeds], axis=0)
            prow.append({'comparison': f'{A_}-{B_}', 'seed': f'mean_of_{len(seeds)}', 'endpoint': k, 'delta': pt,
                         'ci_lo': np.nanpercentile(dr, 2.5), 'ci_hi': np.nanpercentile(dr, 97.5),
                         'n_seeds_positive': int(sum(delta(f'{A_}_s{s}', f'{B_}_s{s}', allidx)[k] > 0 for s in seeds))})
    pd.DataFrame(prow).to_csv(out / 'paired_comparisons.csv', index=False)
    # small-object miss rates, compact
    sm = obj[obj['size'] == 'small'][['model', 'seed', 'class', 'n', 'missed(<10%)', 'detected(>=50%)']]
    sm.to_csv(out / 'small_object_rates.csv', index=False)
    pd.set_option('display.width', 250)
    print(met.round(3).to_string(index=False)); print(sm.round(3).to_string(index=False))
    pc = pd.DataFrame(prow); print(pc[pc.seed.astype(str).str.startswith('mean')].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
