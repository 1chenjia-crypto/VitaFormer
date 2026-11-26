#!/usr/bin/env python3
"""Stage 36 copy of the Stage 31 post-hoc potency script (now PRE-SPECIFIED for M2 in 36_m2_deeporganoid_20261109/ANALYSIS_PLAN.md §5);
only paths/model names changed. Original docstring:
Stage 31 POST-HOC (not in ANALYSIS_PLAN.md; added after the pre-specified
compound-fingerprint test was null and the panel was found to be 8/10 HSP90
inhibitors). Exploratory only.

Q1  Drug x cell-line potency: does image-derived morphology reproduce the CTG
    dose-response area-under-curve (AUC) of each drug in each patient line?
    Two morphology readouts: (a) model-free organoid area relative to the line's
    NEG median; (b) LOCO morphology-predicted activity (core_neg_delta OOF).
Q2  Patient heterogeneity: within each drug, do cell-line sensitivity ranks agree?
Q3  Drug ranking: within each cell line, do drug potency ranks agree?
Q4  Mechanism contrast: Anisomycin (protein-synthesis inhibitor) vs pooled HSP90
    inhibitors after adjusting for concentration and CTG (LOCO logistic).
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

A = Path('3D_OrganoidSeg/external_datasets/deepOrganoid/TrainingData/'
         'analysis_ctg_alignment/36_m2_deeporganoid_20261109/analysis')
OUT = A / 'potency'
OUT.mkdir(exist_ok=False)
SEED = 20261030
HSP90 = ['17-AAG', 'Alvespimycin (17-DMAG) HCl', 'BIIB021', 'NMS-E973', 'Onalespib', 'SNX-2112', 'VER-49009', 'XL888']
CORE = ['foreground_area_ratio', 'fg_component_count', 'fg_area_mean', 'fg_area_max', 'fg_circularity_mean',
        'fg_mean_intensity', 'fg_pellucidity', 'fg_entropy', 'fg_gradient']


def ccl_boot(df, fn, n=2000):
    rng = np.random.default_rng(SEED)
    ccls = np.array(sorted(df.CCL.unique()))
    vals = [fn(pd.concat([df[df.CCL == c] for c in rng.choice(ccls, len(ccls))])) for _ in range(n)]
    return np.nanpercentile(vals, [2.5, 97.5])


def potency(feat, oof):
    d = feat[feat.cohort_drug_only.astype(bool) & feat.Func.isin(['SING', 'DRC1'])].copy()
    neg = feat[feat.label_role.eq('vehicle_viable_reference')].groupby(['model', 'CCL']).foreground_area_ratio.median()
    d['area_rel_neg'] = d.foreground_area_ratio / d.set_index(['model', 'CCL']).index.map(neg).values
    o = oof[(oof.cohort == 'main_all9') & (oof.spec == 'core_neg_delta')][['model', 'sample_id', 'prediction']]
    d = d.merge(o, on=['model', 'sample_id'], how='left', validate='one_to_one')
    z = d.groupby(['model', 'CCL', 'Cmpd1', 'Conc1'], as_index=False)[
        ['relative_ctg_activity', 'area_rel_neg', 'prediction']].median()
    # AUC over ranked concentrations = mean of the 8 per-concentration medians (lower = more potent)
    return z.groupby(['model', 'CCL', 'Cmpd1'], as_index=False).agg(
        n_conc=('Conc1', 'nunique'), auc_ctg=('relative_ctg_activity', 'mean'),
        auc_area=('area_rel_neg', 'mean'), auc_morph_pred=('prediction', 'mean'))


def q1_q3(p):
    rows, within_drug, within_line = [], [], []
    for model, g in p.groupby('model'):
        for excl, cohort in ((False, 'main_all9'), (True, 'excl_Organoid52')):
            h = g[g.CCL != 'Organoid52'] if excl else g
            for readout in ('auc_area', 'auc_morph_pred'):
                f = lambda x: spearmanr(x.auc_ctg, x[readout]).statistic
                lo, hi = ccl_boot(h, f)
                rows.append({'model': model, 'cohort': cohort, 'readout': readout, 'n_pairs': len(h),
                             'spearman_auc_ctg_vs_readout': f(h), 'ci_lo': lo, 'ci_hi': hi})
        for readout in ('auc_area', 'auc_morph_pred'):
            for cmpd, h in g.groupby('Cmpd1'):
                if h.CCL.nunique() >= 5:
                    within_drug.append({'model': model, 'readout': readout, 'Cmpd1': cmpd, 'n_ccl': h.CCL.nunique(),
                                        'spearman_ccl_rank': spearmanr(h.auc_ctg, h[readout]).statistic})
            for ccl, h in g.groupby('CCL'):
                within_line.append({'model': model, 'readout': readout, 'CCL': ccl, 'n_drugs': h.Cmpd1.nunique(),
                                    'spearman_drug_rank': spearmanr(h.auc_ctg, h[readout]).statistic})
    return pd.DataFrame(rows), pd.DataFrame(within_drug), pd.DataFrame(within_line)


def q4(feat):
    rows = []
    for model in ('m2_seed42', 'm2_seed3407', 'm2_seed2026', 'm0_simplified'):
        d = feat[(feat.model == model) & feat.Cmpd1.isin(HSP90 + ['Anisomycin'])]
        cols = [c + '__neg_delta' for c in CORE]
        z = d.groupby(['CCL', 'Cmpd1', 'Conc1'], as_index=False)[cols + ['relative_ctg_activity']].median()
        z['y'] = (z.Cmpd1 == 'Anisomycin').astype(int)
        z['logc'] = np.log10(z.Conc1)
        for excl, cohort in ((False, 'main_all9'), (True, 'excl_Organoid52')):
            w = z[z.CCL != 'Organoid52'] if excl else z
            B, M, y, groups = (w[['logc', 'relative_ctg_activity']].to_numpy(), w[cols].to_numpy(),
                               w.y.to_numpy(), w.CCL.to_numpy())
            pred = {k: np.zeros(len(w), int) for k in ('baseline', 'baseline_plus_residual')}
            for held in np.unique(groups):
                tr, te = groups != held, groups == held
                imp = SimpleImputer(strategy='median').fit(M[tr])
                mtr, mte = imp.transform(M[tr]), imp.transform(M[te])
                reg = LinearRegression().fit(B[tr], mtr)
                rtr, rte = mtr - reg.predict(B[tr]), mte - reg.predict(B[te])
                for k, (xtr, xte) in {'baseline': (B[tr], B[te]),
                                      'baseline_plus_residual': (np.c_[B[tr], rtr], np.c_[B[te], rte])}.items():
                    sc = StandardScaler().fit(xtr)
                    clf = LogisticRegression(class_weight='balanced', max_iter=3000).fit(sc.transform(xtr), y[tr])
                    pred[k][te] = clf.predict(sc.transform(xte))
            inc = np.array([balanced_accuracy_score(y[groups == c], pred['baseline_plus_residual'][groups == c])
                            - balanced_accuracy_score(y[groups == c], pred['baseline'][groups == c])
                            for c in np.unique(groups)])
            rng = np.random.default_rng(SEED)
            boot = [inc[rng.integers(0, len(inc), len(inc))].mean() for _ in range(2000)]
            rows.append({'model': model, 'cohort': cohort, 'n_conditions': len(w),
                         'ba_baseline': balanced_accuracy_score(y, pred['baseline']),
                         'ba_baseline_plus_residual': balanced_accuracy_score(y, pred['baseline_plus_residual']),
                         'mean_ccl_increment': inc.mean(), 'ccl_positive': int((inc > 0).sum()), 'n_ccl': len(inc),
                         'ci_lo': np.percentile(boot, 2.5), 'ci_hi': np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


def main():
    feat = pd.read_csv(A / 'features_raw_and_neg_delta.csv')
    oof = pd.read_csv(A / 'loco_oof_predictions.csv')
    p = potency(feat, oof)
    p.to_csv(OUT / 'drug_cell_line_potency_auc.csv', index=False)
    s, wd, wl = q1_q3(p)
    s.to_csv(OUT / 'potency_agreement_summary.csv', index=False)
    wd.to_csv(OUT / 'within_drug_cell_line_rank_agreement.csv', index=False)
    wl.to_csv(OUT / 'within_cell_line_drug_rank_agreement.csv', index=False)
    m = q4(feat)
    m.to_csv(OUT / 'mechanism_anisomycin_vs_hsp90.csv', index=False)
    (OUT / 'README.md').write_text('# Stage 36 potency analysis (pre-specified for M2; mechanism contrast Q4 remains exploratory)\n\n' + __doc__)
    (OUT / 'run_manifest.json').write_text(json.dumps({'created': '2026-11-09', 'posthoc_for_stage31_prespecified_for_stage36': True,
                                                       'inference_rerun': False, 'script': __file__}, indent=2))
    pd.set_option('display.width', 220)
    print(s.to_string(index=False))
    print(wd.groupby(['model', 'readout']).spearman_ccl_rank.describe().to_string())
    print(wd[(wd.model == 'm2_seed42') & (wd.readout == 'auc_area')].to_string(index=False))
    print(wl.groupby(['model', 'readout']).spearman_drug_rank.describe().to_string())
    print(m.to_string(index=False))


if __name__ == '__main__':
    main()
