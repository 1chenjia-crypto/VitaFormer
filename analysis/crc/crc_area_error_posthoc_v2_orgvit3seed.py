#!/usr/bin/env python3
"""CRC test set: per-image area / viability-ratio error for VitaFormer (M2), nnU-Net, SegFormer-B2 (3 seeds each)
and the OrganoidViT reproduction (1 seed, descriptive). Secondary, post hoc.
Plan: crc_area_error_posthoc_20261111/ANALYSIS_PLAN.md (written before this script was run).
No training, no inference: saved per-image confusion counts only.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path('3D_OrganoidSeg')
OUT = ROOT / 'crc_area_error_posthoc_v2_orgvit3seed'
PLAN = ROOT / 'crc_area_error_posthoc_20261111/ANALYSIS_PLAN.md'
sys.path.insert(0, str(ROOT / 'vco/organoid_seg'))
from dataset import build_dataset_index, stratified_train_val_test_split  # noqa: E402

SEEDS = [42, 3407, 2026]
N_BOOT, RNG_SEED = 10000, 20261111
SE = ROOT / 'baselines_fair_20261030/shared_eval'
MODELS = {
    'VitaFormer': {s: ROOT / f'vco/runs/arch_exploration_20261104/stageF_m2_test/shared_eval/M2_final_seed{s}.json' for s in SEEDS},
    'nnU-Net': {s: SE / f'nnunet_run2_seed{s}_noTTA.json' for s in SEEDS},
    'SegFormer-B2': {s: SE / f'segformer_b2_seed{s}.json' for s in SEEDS},
}
OVIT_CSV = Path('OrganoidSegBaseline/runs/orgvit_b0_1446_40k_cond/'
                'test_best_iter_32000_relative_area_errors.csv')
REF = {'OrganoidViT': {s: Path(f'OrganoidSegBaseline/orgvit_3seed_20261111/shared_eval/organoidvit_seed{s}.json') for s in SEEDS}}
import os  # test hooks (dry run only); unset in the real run
if os.environ.get('AE_TEST_JSON'):
    REF = {'OrganoidViT': {s: Path(os.environ['AE_TEST_JSON']) for s in SEEDS}}
    OUT = Path(os.environ['AE_TEST_OUT']); N_BOOT = 200
CLS = {'Alive': 1, 'Dead': 2, 'Bubble': 3}
CONTRASTS = [('VitaFormer', 'nnU-Net'), ('VitaFormer', 'SegFormer-B2'), ('SegFormer-B2', 'nnU-Net')]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def miou(tp, fp, fn):
    return float(np.mean(tp / (tp + fp + fn)) * 100)


def load(path):
    d = json.load(open(path))
    stems = [x['stem'] for x in d['per_image']]
    c = {k: np.array([x['confusion_counts'][k] for x in d['per_image']], dtype=np.int64) for k in ('tp', 'fp', 'fn')}
    tot = {k: v.sum(0).astype(float) for k, v in c.items()}
    return {'stems': stems, 'pred': c['tp'] + c['fp'], 'gt': c['tp'] + c['fn'],
            'miou_json': d['mIoU'], 'miou_recomputed': miou(tot['tp'], tot['fp'], tot['fn'])}


def viability(a, d):
    s = a + d
    return np.divide(a, s, out=np.full(a.shape, np.nan, dtype=float), where=s > 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('plan sha256', sha256(PLAN))
    runs = {m: {s: load(p) for s, p in sp.items()} for m, sp in {**MODELS, **REF}.items()}
    stems = runs['VitaFormer'][42]['stems']
    gt = runs['VitaFormer'][42]['gt']
    checks = {'stem_order_identical': True, 'gt_identical': True, 'miou': {}}
    for m, sp in runs.items():
        for s, r in sp.items():
            checks['stem_order_identical'] &= r['stems'] == stems
            checks['gt_identical'] &= bool((r['gt'] == gt).all())
            checks['miou'][f'{m} s{s}'] = {'json': r['miou_json'], 'recomputed': r['miou_recomputed'],
                                           'abs_diff_pp': abs(r['miou_json'] - r['miou_recomputed'])}
    assert checks['stem_order_identical'], 'stem order differs -> stop'
    assert checks['gt_identical'], 'GT areas differ -> stop'
    assert all(v['abs_diff_pp'] < 0.01 for v in checks['miou'].values()), 'mIoU mismatch -> stop'

    # OrganoidViT
    ov_ok = False  # v2: OrganoidViT comes from the 3 shared-evaluator JSONs (REF), not the seed-0 CSV
    ovp = ovg = None
    if ov_ok:
        ov = ov.loc[stems]
        ovg = np.stack([np.zeros(len(stems)), ov.gt_live_area, ov.gt_death_area, ov.gt_bubble_area], 1).astype(np.int64)
        ovp = np.stack([np.zeros(len(stems)), ov.pred_live_area, ov.pred_death_area, ov.pred_bubble_area], 1).astype(np.int64)
        diff = ovg[:, 1:] - gt[:, 1:]
        checks['organoidvit'] = {'same_image_set': True, 'gt_identical_all': bool((diff == 0).all()),
                                 'n_cells_differing': int((diff != 0).sum()),
                                 'max_abs_gt_diff_px': int(np.abs(diff).max()),
                                 'total_gt_by_class_ovit': ovg[:, 1:].sum(0).tolist(),
                                 'total_gt_by_class_shared': gt[:, 1:].sum(0).tolist()}
    else:
        checks['organoidvit'] = {'source': 'shared evaluator JSONs, seeds 42/3407/2026 (A800)'}
    ov_comparable = True
    print(json.dumps(checks, indent=1))

    # wells
    _, wg = build_dataset_index('OrganoidSegBaseline/VOC2012')
    _, _, test = stratified_train_val_test_split(wg, 0.6, 0.2, seed=42, use_condition=True)
    s2w = {x['stem']: x['well_id'] for x in test}
    assert set(s2w) == set(stems)
    well = np.array([s2w[s] for s in stems])

    # per-image table
    preds = {(m, s): r['pred'] for m, sp in runs.items() for s, r in sp.items()}
    if ov_ok:
        preds[('OrganoidViT (repro)', 42)] = ovp
    rows, vrows = [], []
    for (m, s), p in preds.items():
        g = ovg if m == 'OrganoidViT (repro)' else gt
        for cname, c in CLS.items():
            for i, st in enumerate(stems):
                if g[i, c] > 0:
                    rows.append({'model': m, 'seed': s, 'class': cname, 'stem': st, 'well_id': well[i],
                                 'gt_px': int(g[i, c]), 'pred_px': int(p[i, c]),
                                 'signed_px': int(p[i, c] - g[i, c]), 'abs_px': int(abs(p[i, c] - g[i, c])),
                                 'signed_rel': (p[i, c] - g[i, c]) / g[i, c],
                                 'abs_rel': abs(p[i, c] - g[i, c]) / g[i, c]})
        vg, vp = viability(g[:, 1], g[:, 2]), viability(p[:, 1], p[:, 2])
        for i, st in enumerate(stems):
            if g[i, 1] + g[i, 2] > 0:
                vrows.append({'model': m, 'seed': s, 'stem': st, 'well_id': well[i], 'gt_dead_present': bool(g[i, 2] > 0),
                              'V_gt': vg[i], 'V_pred': vp[i], 'pred_undefined': bool(np.isnan(vp[i])),
                              'dV_pp': (vp[i] - vg[i]) * 100, 'abs_dV_pp': abs(vp[i] - vg[i]) * 100})
    d, v = pd.DataFrame(rows), pd.DataFrame(vrows)
    d.to_csv(OUT / 'per_image_area_errors.csv', index=False)
    v.to_csv(OUT / 'viability_per_image.csv', index=False)

    # seed summaries
    q = lambda x, k: float(np.nanpercentile(x, k))  # noqa: E731
    srows = []
    for (m, s), dd in d.groupby(['model', 'seed']):
        g = ovg if m == 'OrganoidViT (repro)' else gt
        p = preds[(m, s)]
        for cname, c in CLS.items():
            x = dd[dd['class'] == cname]
            el = g[:, c] > 0
            for ep in ('abs_rel', 'signed_rel', 'abs_px', 'signed_px'):
                srows.append({'model': m, 'seed': s, 'class': cname, 'endpoint': ep, 'n': len(x),
                              'median': q(x[ep], 50), 'q1': q(x[ep], 25), 'q3': q(x[ep], 75), 'mean': float(x[ep].mean())})
            srows.append({'model': m, 'seed': s, 'class': cname, 'endpoint': 'pooled_signed_rel', 'n': int(el.sum()),
                          'median': float((p[el, c].sum() - g[el, c].sum()) / g[el, c].sum())})
        vv = v[(v.model == m) & (v.seed == s)]
        for setname, sub in (('primary', vv), ('dead_present', vv[vv.gt_dead_present])):
            ok = sub[~sub.pred_undefined]
            for ep in ('dV_pp', 'abs_dV_pp'):
                srows.append({'model': m, 'seed': s, 'class': f'Viability[{setname}]', 'endpoint': ep, 'n': len(ok),
                              'n_pred_undefined': int(sub.pred_undefined.sum()), 'median': q(ok[ep], 50),
                              'q1': q(ok[ep], 25), 'q3': q(ok[ep], 75), 'mean': float(ok[ep].mean())})
            el = np.isin(stems, sub.stem)
            pv = p[el, 1].sum() / (p[el, 1].sum() + p[el, 2].sum())
            gv = g[el, 1].sum() / (g[el, 1].sum() + g[el, 2].sum())
            srows.append({'model': m, 'seed': s, 'class': f'Viability[{setname}]', 'endpoint': 'pooled_dV_pp',
                          'n': int(el.sum()), 'median': float((pv - gv) * 100)})
    sdf = pd.DataFrame(srows)
    sdf.to_csv(OUT / 'seed_summary.csv', index=False)
    across = sdf.groupby(['class', 'endpoint', 'model']).agg(
        n=('n', 'first'), mean_of_seed_medians=('median', 'mean'), min_seed_median=('median', 'min'),
        max_seed_median=('median', 'max'), mean_of_seed_means=('mean', 'mean')).reset_index()
    across.to_csv(OUT / 'across_seed_summary.csv', index=False)

    # bootstrap contrasts
    arr = {}  # (model, seed, key) -> per-image array over 167 stems (NaN if ineligible)
    for (m, s), p in preds.items():
        if m.startswith('OrganoidViT'):
            continue
        for cname, c in CLS.items():
            el = gt[:, c] > 0
            arr[(m, s, cname, 'P')] = np.where(el, p[:, c], np.nan).astype(float)
            arr[(m, s, cname, 'G')] = np.where(el, gt[:, c], np.nan).astype(float)
        vg, vp = viability(gt[:, 1], gt[:, 2]), viability(p[:, 1], p[:, 2])
        prim = (gt[:, 1] + gt[:, 2]) > 0
        arr[(m, s, 'dV_primary')] = np.where(prim, (vp - vg) * 100, np.nan)
        arr[(m, s, 'dV_dead')] = np.where(gt[:, 2] > 0, (vp - vg) * 100, np.nan)

    def stats(m, seed_list, idx_list):
        out = {}
        for cname in CLS:
            P = np.concatenate([arr[(m, s, cname, 'P')][ix] for s, ix in zip(seed_list, idx_list)])
            G = np.concatenate([arr[(m, s, cname, 'G')][ix] for s, ix in zip(seed_list, idx_list)])
            ok = ~np.isnan(G)
            P, G = P[ok], G[ok]
            out[f'{cname} median abs rel err'] = np.median(np.abs(P - G) / G)
            out[f'{cname} median signed rel err'] = np.median((P - G) / G)
            out[f'{cname} pooled signed rel err'] = (P.sum() - G.sum()) / G.sum()
            out[f'{cname} median abs px err'] = np.median(np.abs(P - G))
        for key, lab in (('dV_primary', 'primary'), ('dV_dead', 'Dead-present')):
            x = np.concatenate([arr[(m, s, key)][ix] for s, ix in zip(seed_list, idx_list)])
            x = x[~np.isnan(x)]
            out[f'Viability[{lab}] median |dV| pp'] = np.median(np.abs(x))
            out[f'Viability[{lab}] mean signed dV pp'] = x.mean()
        return out

    allidx = np.arange(len(stems))
    obs = {m: stats(m, SEEDS, [allidx] * 3) for m in MODELS}
    per_seed = {(m, s): stats(m, [s], [allidx]) for m in MODELS for s in SEEDS}
    wells = np.unique(well)
    by_well = {w: np.flatnonzero(well == w) for w in wells}
    rng = np.random.default_rng(RNG_SEED)
    keys = list(obs['VitaFormer'])
    boot = {c: {k: np.empty(N_BOOT) for k in keys} for c in CONTRASTS}
    mboot = {m: {k: np.empty(N_BOOT) for k in keys} for m in MODELS}  # amendment 1: per-model CIs (plan bias rule)
    for b in range(N_BOOT):
        sp = rng.integers(0, 3, 3)
        seed_list = [SEEDS[i] for i in sp]
        idx_list = [np.concatenate([by_well[w] for w in rng.choice(wells, len(wells))]) for _ in range(3)]
        st = {m: stats(m, seed_list, idx_list) for m in MODELS}
        for a, bb in CONTRASTS:
            for k in keys:
                boot[(a, bb)][k][b] = st[a][k] - st[bb][k]
        for m in MODELS:
            for k in keys:
                mboot[m][k][b] = st[m][k]
        if b % 2000 == 0:
            print('bootstrap', b, flush=True)
    crows = []
    for a, bb in CONTRASTS:
        for k in keys:
            sd = [per_seed[(a, s)][k] - per_seed[(bb, s)][k] for s in SEEDS]
            bt = boot[(a, bb)][k]
            crows.append({'contrast': f'{a} - {bb}', 'endpoint': k, 'A_obs': obs[a][k], 'B_obs': obs[bb][k],
                          'delta_obs': obs[a][k] - obs[bb][k], 'ci_lo': np.percentile(bt, 2.5),
                          'ci_hi': np.percentile(bt, 97.5), 'seed_deltas': ';'.join(f'{x:+.4g}' for x in sd),
                          'n_seeds_A_lower': int(sum(x < 0 for x in sd)), 'n_seeds_A_higher': int(sum(x > 0 for x in sd))})
    cdf = pd.DataFrame(crows)
    mrows = []
    for m in MODELS:
        for k in keys:
            sv = [per_seed[(m, s)][k] for s in SEEDS]
            mrows.append({'model': m, 'endpoint': k, 'obs': obs[m][k], 'ci_lo': np.percentile(mboot[m][k], 2.5),
                          'ci_hi': np.percentile(mboot[m][k], 97.5), 'seed_values': ';'.join(f'{x:+.4g}' for x in sv),
                          'all_seeds_same_sign': bool(all(x > 0 for x in sv) or all(x < 0 for x in sv))})
    pd.DataFrame(mrows).to_csv(OUT / 'per_model_ci.csv', index=False)
    # amendment 1 (descriptive, added after first run): false Dead in images without GT Dead
    frows = []
    for (m, s), p in preds.items():
        g = ovg if m == 'OrganoidViT (repro)' else gt
        nd = (g[:, 2] == 0) & ((g[:, 1] + g[:, 2]) > 0)
        fd = p[nd, 2]
        frac = fd / np.maximum(p[nd, 1] + p[nd, 2], 1)
        frows.append({'model': m, 'seed': s, 'n_images_no_gt_dead': int(nd.sum()), 'n_with_pred_dead': int((fd > 0).sum()),
                      'total_false_dead_px': int(fd.sum()), 'median_false_dead_px_if_any': float(np.median(fd[fd > 0])) if (fd > 0).any() else 0.0,
                      'mean_false_dead_fraction_of_organoid_pct': float(frac.mean() * 100),
                      'n_false_dead_fraction_gt_10pct': int((frac > 0.10).sum())})
    pd.DataFrame(frows).to_csv(OUT / 'false_dead_no_gt_dead.csv', index=False)
    print(pd.DataFrame(mrows).round(4).to_string(index=False))
    print(pd.DataFrame(frows).round(3).to_string(index=False))
    cdf.to_csv(OUT / 'contrasts.csv', index=False)
    pd.DataFrame([{'model': m, **obs[m]} for m in MODELS]).to_csv(OUT / 'observed_pooled_3seed.csv', index=False)
    if ov_ok:
        pd.DataFrame([{'model': 'OrganoidViT (repro)', **{k: vv for k, vv in stats_ovit(ovp, ovg).items()}}]).to_csv(
            OUT / 'observed_organoidvit.csv', index=False)

    checks['organoidvit_comparable'] = bool(ov_comparable)
    (OUT / 'consistency_checks.json').write_text(json.dumps(checks, indent=1, default=float))
    (OUT / 'run_manifest.json').write_text(json.dumps({
        'created': '2026-11-11', 'script': __file__, 'script_sha256': sha256(__file__),
        'plan_sha256': sha256(PLAN),
        'inputs': {str(p): sha256(p) for sp in {**MODELS, **REF}.values() for p in sp.values()},
        'n_images': len(stems), 'n_wells': int(len(wells)),
        'bootstrap': f'hierarchical paired: 3 seed positions + physical wells per position, {N_BOOT}, seed {RNG_SEED}',
        'training_rerun': False, 'inference_rerun': False, 'status': 'secondary, post hoc'}, indent=2))
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    print(across.round(4).to_string(index=False))
    print(cdf.round(4).to_string(index=False))


def stats_ovit(p, g):
    out = {}
    for cname, c in CLS.items():
        el = g[:, c] > 0
        P, G = p[el, c].astype(float), g[el, c].astype(float)
        out[f'{cname} median abs rel err'] = np.median(np.abs(P - G) / G)
        out[f'{cname} median signed rel err'] = np.median((P - G) / G)
        out[f'{cname} pooled signed rel err'] = (P.sum() - G.sum()) / G.sum()
        out[f'{cname} median abs px err'] = np.median(np.abs(P - G))
    vg, vp = viability(g[:, 1], g[:, 2]), viability(p[:, 1], p[:, 2])
    for el, lab in (((g[:, 1] + g[:, 2]) > 0, 'primary'), (g[:, 2] > 0, 'Dead-present')):
        x = ((vp - vg) * 100)[el]
        x = x[~np.isnan(x)]
        out[f'Viability[{lab}] median |dV| pp'] = np.median(np.abs(x))
        out[f'Viability[{lab}] mean signed dV pp'] = x.mean()
    return out


if __name__ == '__main__':
    main()
