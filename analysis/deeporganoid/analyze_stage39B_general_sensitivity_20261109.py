#!/usr/bin/env python3
"""Stage 39 copy of Stage 38B (pre-specified in 39_segformer_b2_deeporganoid_20261109/ANALYSIS_PLAN.md §4): SegFormer-B2
seeds added; only SRC and OUT changed. Original docstring: Stage 38B (POST HOC, 2026-11-09): 'general line sensitivity' baseline for the within-drug line ranking.
Plan: 38_posthoc_robustness_20261109/PLAN_B_general_sensitivity_baseline.md. Existing features only (no inference).

AUC tables exactly as Stage 34/37 (drug wells SING/DRC1, area relative to the line's vehicle median, per-dose median, mean over
doses). G(line, d) = mean CTG AUC of that line over all OTHER drugs (d excluded), computed after cohort filtering.
1 baseline-only within-drug rho; 2 image-only rho (check); 3 partial rho (rank-residualised on G); 4 leave-one-line-out
held-out prediction CTG AUC ~ G vs ~ G + image AUC (fit on training lines, all drugs pooled), within-drug rho of predictions.
Drug bootstrap 2,000 (RNG 20261030, Stage 34), paired deltas on shared draws. Drugs with >= 5 lines.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, str(Path(__file__).parent))
import stage34_core_claims_robustness_20261030 as s34  # noqa: E402

BASE = Path('3D_OrganoidSeg/external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment')
OUT = BASE / '39_segformer_b2_deeporganoid_20261109/B_general_sensitivity'
LABELS = BASE / '19_label_cohort_audit_20261027/row_label_audit.csv'
SRC = {'vf_m2_s42': [BASE / '36_m2_deeporganoid_20261109/m2_seed42/features_shard0of1.csv'],
       'vf_simplified_M0': sorted((BASE / '31_final_model_p1p99_full_20261029/simplified').glob('features_shard*of3.csv')),
       'nnunet_s42_V0': [BASE / '35_nnunet_external_comparison_20261102/nnunet_seed42/features.csv'],
       **{f'segformer_s{k}': [BASE / f'39_segformer_b2_deeporganoid_20261109/segformer_seed{k}/features_shard0of1.csv'] for k in (42, 3407, 2026)}}
PRIMARY = 'vf_m2_s42'; NBOOT, SEED, MIN_LINES = 2000, 20261030, 5


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def auc_tables():
    lab = pd.read_csv(LABELS)[['sample_id', 'relative_ctg_activity', 'label_role', 'cohort_drug_only']]
    out = {}
    for m, files in SRC.items():
        f = pd.concat([pd.read_csv(p) for p in files], ignore_index=True).merge(lab, on='sample_id', how='left', validate='one_to_one')
        if len(f) != 3456: raise RuntimeError(m)
        neg = f[f.label_role.eq('vehicle_viable_reference')].groupby('CCL').foreground_area_ratio.median()
        d = f[f.cohort_drug_only.astype(bool) & f.Func.isin(['SING', 'DRC1'])].copy()
        d['area_rel_neg'] = d.foreground_area_ratio / d.CCL.map(neg)
        out[m] = s34.auc_table(d, 'area_rel_neg')
    return out


def add_G(a):
    a = a.copy(); tot = a.groupby('CCL').auc_ctg.transform('sum'); n = a.groupby('CCL').auc_ctg.transform('count')
    a['G'] = (tot - a.auc_ctg) / (n - 1)
    return a


def resid(v, z):
    rv, rz = rankdata(v), rankdata(z); Z = np.c_[np.ones_like(rz), rz]
    return rv - Z @ np.linalg.lstsq(Z, rv, rcond=None)[0]


def per_drug(a, kind):
    res = {}
    for drug, h in a.groupby('Cmpd1'):
        if h.CCL.nunique() < MIN_LINES: continue
        if kind == 'baseline': res[drug] = spearmanr(h.G, h.auc_ctg).statistic
        elif kind == 'image': res[drug] = spearmanr(h.auc_x, h.auc_ctg).statistic
        elif kind == 'partial': res[drug] = float(np.corrcoef(resid(h.auc_x, h.G), resid(h.auc_ctg, h.G))[0, 1])
        elif kind in ('pred_G', 'pred_GX'): res[drug] = spearmanr(h[kind], h.auc_ctg).statistic
    return pd.Series(res)


def loo_predict(a):
    a = a.copy(); a['pred_G'] = np.nan; a['pred_GX'] = np.nan
    for line in a.CCL.unique():
        tr, te = a.CCL != line, a.CCL == line
        for col, X in (('pred_G', ['G']), ('pred_GX', ['G', 'auc_x'])):
            A = np.c_[np.ones(tr.sum()), a.loc[tr, X].to_numpy(float)]
            beta = np.linalg.lstsq(A, a.loc[tr, 'auc_ctg'].to_numpy(float), rcond=None)[0]
            a.loc[te, col] = np.c_[np.ones(te.sum()), a.loc[te, X].to_numpy(float)] @ beta
    return a


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    tabs = auc_tables(); OUT.mkdir(parents=True)
    variants = {'all9_all_drugs': lambda a: a, 'excl_Organoid52': lambda a: a[a.CCL != 'Organoid52'],
                'excl_LPA_posthoc': lambda a: a[a.Cmpd1 != 'LPA']}
    rows, perdrug = [], []
    for cohort, f in variants.items():
        rng = np.random.default_rng(SEED)
        series = {}
        for m, t in tabs.items():
            a = loo_predict(add_G(f(t)))
            a.assign(model=m, cohort=cohort).to_csv(OUT / f'auc_G_pred_{m}_{cohort}.csv', index=False)
            for kind in ('baseline', 'image', 'partial', 'pred_G', 'pred_GX'):
                series[(m, kind)] = per_drug(a, kind)
        drugs = np.array(sorted(set.intersection(*[set(s.index) for s in series.values()])))
        draws = [rng.choice(drugs, len(drugs)) for _ in range(NBOOT)]
        for (m, kind), s in series.items():
            s = s.loc[drugs]; b = np.array([s.loc[dr].mean() for dr in draws])
            rows.append(dict(cohort=cohort, model=m, metric=kind, n_drugs=len(drugs), mean_rho=s.mean(),
                             ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5)))
            for d, v in s.items(): perdrug.append(dict(cohort=cohort, model=m, metric=kind, drug=d, rho=v))
        for m in tabs:
            gx, g = series[(m, 'pred_GX')].loc[drugs], series[(m, 'pred_G')].loc[drugs]
            b = np.array([gx.loc[dr].mean() - g.loc[dr].mean() for dr in draws])
            rows.append(dict(cohort=cohort, model=m, metric='delta_pred_GX_minus_pred_G', n_drugs=len(drugs),
                             mean_rho=gx.mean() - g.mean(), ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5)))
    S = pd.DataFrame(rows); S.to_csv(OUT / 'summary.csv', index=False); pd.DataFrame(perdrug).to_csv(OUT / 'per_drug.csv', index=False)
    p = S[(S.cohort == 'all9_all_drugs') & (S.model == PRIMARY)].set_index('metric')
    reading = ('image adds drug-specific information beyond general line sensitivity'
               if p.loc['partial', 'ci_lo'] > 0 and p.loc['delta_pred_GX_minus_pred_G', 'ci_lo'] > 0
               else 'image ranking mainly reflects overall line sensitivity (claim to be narrowed)')
    (OUT / 'run_manifest.json').write_text(json.dumps(dict(created='2026-11-09', posthoc=True, script=__file__, script_sha256=sha(__file__),
        inputs_sha256={str(p_): sha(p_) for fs in SRC.values() for p_ in fs} | {str(LABELS): sha(LABELS)},
        primary=PRIMARY, nboot=NBOOT, seed=SEED, min_lines=MIN_LINES, reading_primary_all9=reading,
        caveat='8 of 10 drugs are HSP90 inhibitors: G is strong by construction; not a cross-mechanism test'), indent=2))
    pd.set_option('display.width', 200); print(S.round(3).to_string(index=False)); print('READING:', reading)


if __name__ == '__main__':
    main()
