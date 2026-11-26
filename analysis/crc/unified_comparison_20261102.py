#!/usr/bin/env python3
"""Unified 3-seed comparison of VitaFormer variants, SegFormer-B2, P1-S and nnU-Net on the 167 CRC test images
(baselines_fair_20261030/unified_comparison_20261102/ANALYSIS_PLAN.md). Per-image tp/fp/fn from saved JSONs only;
no inference. Hierarchical paired bootstrap as in analyze_multiseed_ablation.py (3 seed positions with replacement,
physical wells with replacement per seed position, pixels pooled), 10,000 replicates, RNG seed 20261102; all groups
share the same draws (needed for the interaction contrast).
"""

from pathlib import Path
import hashlib
import io
import contextlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path('3D_OrganoidSeg')
sys.path.insert(0, str(ROOT / 'vco/organoid_seg'))
import analyze_multiseed_ablation as M  # noqa: E402
from dataset import build_dataset_index, stratified_train_val_test_split  # noqa: E402

B = ROOT / 'baselines_fair_20261030'
SE = B / 'shared_eval'
OUT = B / 'unified_comparison_20261102'
N_BOOT, RNG_SEED, SEEDS = 10000, 20261102, [42, 3407, 2026]
ENDPOINTS = ['mIoU', 'mDice', 'Alive IoU', 'Dead IoU', 'Bubble IoU', 'Dead recall', 'Dead precision']

GROUPS = {
    'VF-Simplified': {s: M.RUNS / d / 'test_best_miou_uniform.json' for s, d in M.GROUPS['No BAM + No MPM'].items()},
    'VF-Full': {s: M.RUNS / d / 'test_best_miou_uniform.json' for s, d in M.GROUPS['Full Model'].items()},
    'VF-Full-noRD': {s: M.RUNS / d / 'test_best_miou_uniform.json' for s, d in M.GROUPS['No Rare-Dead'].items()},
    'SegFormer-B2': {s: SE / f'segformer_b2_seed{s}.json' for s in SEEDS},
    'P1-S': {s: SE / f'segformer_b2_raredead_seed{s}.json' for s in SEEDS},
    'nnU-Net': {s: SE / f'nnunet_run2_seed{s}_noTTA.json' for s in SEEDS},
    'nnU-Net TTA': {s: SE / f'nnunet_run2_seed{s}_TTA.json' for s in SEEDS},
}
CONTRASTS = [  # (name, [(coef, group), ...])
    ('VF-Simplified - SegFormer-B2', [(1, 'VF-Simplified'), (-1, 'SegFormer-B2')]),
    ('VF-Simplified - nnU-Net', [(1, 'VF-Simplified'), (-1, 'nnU-Net')]),
    ('VF-Full - SegFormer-B2', [(1, 'VF-Full'), (-1, 'SegFormer-B2')]),
    ('VF-Full - nnU-Net', [(1, 'VF-Full'), (-1, 'nnU-Net')]),
    ('SegFormer-B2 - nnU-Net', [(1, 'SegFormer-B2'), (-1, 'nnU-Net')]),
    ('Rare-Dead in SegFormer: P1-S - SegFormer-B2', [(1, 'P1-S'), (-1, 'SegFormer-B2')]),
    ('Rare-Dead in VitaFormer: VF-Full - VF-Full-noRD', [(1, 'VF-Full'), (-1, 'VF-Full-noRD')]),
    ('Interaction: (P1-S - SegFormer-B2) - (VF-Full - VF-Full-noRD)',
     [(1, 'P1-S'), (-1, 'SegFormer-B2'), (-1, 'VF-Full'), (1, 'VF-Full-noRD')]),
    ('VF-Simplified - P1-S', [(1, 'VF-Simplified'), (-1, 'P1-S')]),
    ('Secondary: nnU-Net TTA - nnU-Net', [(1, 'nnU-Net TTA'), (-1, 'nnU-Net')]),
]


def load(p):
    r = json.loads(Path(p).read_text())
    per = r['per_image']
    return ([x['stem'] for x in per],
            np.stack([np.asarray([x['confusion_counts'][k] for x in per], np.int64) for k in ('tp', 'fp', 'fn')]))


def metrics(c):  # c: (..., 3, 4) pooled tp/fp/fn
    tp, fp, fn = (c[..., i, :].astype(np.float64) for i in range(3))
    with np.errstate(invalid='ignore', divide='ignore'):
        iou = tp / (tp + fp + fn); dice = 2 * tp / (2 * tp + fp + fn)
        rec = tp[..., 2] / (tp[..., 2] + fn[..., 2]); prec = tp[..., 2] / (tp[..., 2] + fp[..., 2])
    return {'mIoU': 100 * np.nanmean(iou, -1), 'mDice': 100 * np.nanmean(dice, -1), 'Alive IoU': 100 * iou[..., 1],
            'Dead IoU': 100 * iou[..., 2], 'Bubble IoU': 100 * iou[..., 3], 'Dead recall': 100 * rec,
            'Dead precision': 100 * prec}


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'contrasts.csv').exists():
        raise SystemExit('outputs exist; refusing to overwrite')
    data, ref, hashes = {}, None, {}
    for g, m in GROUPS.items():
        data[g] = {}
        for s, p in m.items():
            stems, c = load(p)
            ref = ref or stems
            if sorted(stems) != sorted(ref) or len(set(stems)) != 167:
                raise RuntimeError(f'test image set mismatch: {p}')
            order = {st: i for i, st in enumerate(stems)}
            data[g][s] = c[:, [order[st] for st in ref]]  # (3, 167, 4), aligned to the reference stem order
            hashes[str(p)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    _, wg = build_dataset_index('OrganoidSegBaseline/VOC2012')
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test = stratified_train_val_test_split(wg, 0.6, 0.2, seed=42, use_condition=True)
    s2w = {x['stem']: x['well_id'] for x in test}
    wells = sorted({s2w[s] for s in ref}); wid = np.array([wells.index(s2w[s]) for s in ref])
    nw = len(wells)
    # per-well sums: (group, seed) -> (nw, 3, 4)
    wsum = {g: {s: np.stack([np.add.reduceat(c[k][np.argsort(wid, kind='stable')], np.r_[0, np.cumsum(np.bincount(wid))[:-1]])
                             for k in range(3)], 1) for s, c in d.items()} for g, d in data.items()}
    for g in GROUPS:
        for s in SEEDS:
            assert (wsum[g][s].sum(0) == data[g][s].sum(1)).all()
    obs = {g: {s: metrics(c.sum(1)) for s, c in d.items()} for g, d in data.items()}

    rows = []
    for g in GROUPS:
        for e in ENDPOINTS:
            v = np.array([obs[g][s][e] for s in SEEDS])
            rows.append({'group': g, 'endpoint': e, **{f'seed{s}': float(x) for s, x in zip(SEEDS, v)},
                         'mean': v.mean(), 'sd': v.std(ddof=1), 'range': v.max() - v.min()})
    seed_df = pd.DataFrame(rows); seed_df.to_csv(OUT / 'seed_level.csv', index=False)

    rng = np.random.default_rng(RNG_SEED)
    boot = {g: {e: np.empty((N_BOOT, 3)) for e in ENDPOINTS} for g in GROUPS}
    for b in range(N_BOOT):
        sp = rng.integers(0, 3, 3)
        for j, pos in enumerate(sp):
            cnt = np.bincount(rng.integers(0, nw, nw), minlength=nw).astype(np.float64)
            s = SEEDS[pos]
            for g in GROUPS:
                m = metrics(np.tensordot(cnt, wsum[g][s], axes=(0, 0)))
                for e in ENDPOINTS:
                    boot[g][e][b, j] = m[e]
    crow = []
    for name, terms in CONTRASTS:
        for e in ENDPOINTS:
            per_seed = np.array([sum(c * obs[g][s][e] for c, g in terms) for s in SEEDS])
            bd = sum(c * boot[g][e].mean(1) for c, g in terms)
            lo, hi = np.nanpercentile(bd, [2.5, 97.5])
            crow.append({'contrast': name, 'endpoint': e, 'delta_mean': per_seed.mean(), 'ci95_low': lo, 'ci95_high': hi,
                         'ci_excludes_0': bool(lo > 0 or hi < 0), 'seed_deltas': ';'.join(f'{x:+.2f}' for x in per_seed),
                         'seeds_positive': int((per_seed > 0).sum()), 'boot_frac_positive': float((bd > 0).mean())})
    con = pd.DataFrame(crow); con.to_csv(OUT / 'contrasts.csv', index=False)
    (OUT / 'manifest.json').write_text(json.dumps({
        'created': '2026-11-02', 'script': __file__, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'bootstrap': f'hierarchical paired (seed positions + {nw} physical wells), {N_BOOT}, RNG {RNG_SEED}; shared draws',
        'n_images': len(ref), 'n_wells': nw, 'inputs_sha256': hashes, 'inference_rerun': False}, indent=2))
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    print(seed_df.round(2).to_string(index=False))
    print(con.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
