#!/usr/bin/env python3
"""Stage 36 analysis: frozen M2 (3 seeds) vs M0 (Stage 31 'simplified', read-only) on DeepOrganoid (p1-p99).
Copy of analyze_stage31_deeporganoid_two_models_20261029.py; changes: model list/paths, reproducibility block replaced by
M2-vs-M0 mask agreement, permutations only for the primary model (m2_seed42). Plan: 36_m2_deeporganoid_20261109/ANALYSIS_PLAN.md.
Original docstring:
Stage 31 analysis: simplified vs Full VitaFormer on DeepOrganoid (p1-p99).

Methods copied unchanged from Stages 21 (NEG normalization), 22 (nested LOCO +
hierarchical bootstrap), 23 (within-cell-line label permutation), 24 (dose
trajectories) and 25 (compound fingerprint + paired cell-line bootstrap).
Pre-specified in 31_final_model_p1p99_full_20261029/ANALYSIS_PLAN.md.
No segmentation inference; reads the Stage 31 feature shards only.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, top_k_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path('3D_OrganoidSeg')
BASE = ROOT / 'external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment'
STAGE = BASE / '36_m2_deeporganoid_20261109'
S31 = BASE / '31_final_model_p1p99_full_20261029'
OUT = STAGE / 'analysis'
OUT.mkdir(exist_ok=False)
LABELS = BASE / '19_label_cohort_audit_20261027/row_label_audit.csv'
LEGACY_FULL = ROOT / 'deepOrganoid_v1/features/full_features.csv'
MODELS = ('m2_seed42', 'm2_seed3407', 'm2_seed2026', 'm0_simplified')
PRIMARY_MODEL = 'm2_seed42'
SHARDS = {'m2_seed42': (STAGE / 'm2_seed42', 'features_shard*of1.csv', 1),
          'm2_seed3407': (STAGE / 'm2_seed3407', 'features_shard*of1.csv', 1),
          'm2_seed2026': (STAGE / 'm2_seed2026', 'features_shard*of1.csv', 1),
          'm0_simplified': (S31 / 'simplified', 'features_shard*of3.csv', 3)}
SEED = 20261029
ALPHAS = [0.1, 1.0, 10.0, 100.0]
N_PERM = 200
CORE = ['foreground_area_ratio', 'fg_component_count', 'fg_area_mean', 'fg_area_max', 'fg_circularity_mean',
        'fg_mean_intensity', 'fg_pellucidity', 'fg_entropy', 'fg_gradient']
RICH = ['foreground_area_ratio', 'foreground_prob_sum_ratio', 'alive_prob_sum_ratio', 'fg_component_count',
        'fg_area_mean', 'fg_area_max', 'fg_equiv_diam_median', 'fg_large_frac', 'fg_small_frac',
        'fg_solidity_mean', 'fg_circularity_mean', 'fg_boundary_irreg_mean', 'fg_hole_frac_mean',
        'fg_fragmentation', 'fg_eccentricity_mean', 'fg_mean_intensity', 'fg_intensity_std', 'fg_pellucidity',
        'fg_entropy', 'fg_gradient', 'fg_edge_density']
LOCO_SPECS = {'area_raw': ['foreground_area_ratio'], 'core_raw': CORE,
              'core_neg_delta': [f + '__neg_delta' for f in CORE],
              'rich_neg_delta': [f + '__neg_delta' for f in RICH]}
PRIMARY_SPEC = 'core_neg_delta'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 1. load + reproducibility
def load():
    labels = pd.read_csv(LABELS)
    meta = labels[['sample_id', 'relative_ctg_activity', 'label_role', 'cohort_assay_wide', 'cohort_drug_only']]
    frames, inputs = [], {}
    for model in MODELS:
        folder, pat, nexp = SHARDS[model]
        shards = sorted(folder.glob(pat))
        if len(shards) != nexp:
            raise RuntimeError(f'{model}: expected 3 shards, found {len(shards)}')
        d = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
        if len(d) != 3456 or d.sample_id.duplicated().any():
            raise RuntimeError(f'{model}: not 3456 unique rows')
        if d.mask_reused.any():
            raise RuntimeError(f'{model}: soft features unavailable for reused masks')
        inputs.update({str(p): sha256(p) for p in shards})
        frames.append(d.assign(model=model).merge(meta, on='sample_id', how='left', validate='one_to_one'))
    return pd.concat(frames, ignore_index=True), inputs


def reproducibility(data):
    """M2 (each seed) vs M0 hard masks: pixel class agreement and foreground-pixel Dice per image (descriptive)."""
    rows = []
    m0 = S31 / 'simplified' / 'hard_class_masks'
    for model in MODELS[:3]:
        agree, dice = [], []
        for p in sorted((STAGE / model / 'hard_class_masks').glob('*_classes.png')):
            a = np.asarray(Image.open(p)); b = np.asarray(Image.open(m0 / p.name))
            agree.append(float((a == b).mean()))
            fa, fb = np.isin(a, (1, 2)), np.isin(b, (1, 2)); den = fa.sum() + fb.sum()
            dice.append(1.0 if den == 0 else 2 * float((fa & fb).sum()) / den)
        rows.append({'check': f'{model}_vs_m0_masks', 'n': len(agree), 'pixel_agreement_mean': float(np.mean(agree)),
                     'pixel_agreement_min': float(np.min(agree)), 'fg_dice_median': float(np.median(dice)),
                     'fg_dice_p05': float(np.percentile(dice, 5))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. QC
def qc(data):
    d = data[data.cohort_assay_wide.astype(bool)]
    return d.groupby(['model', 'CCL']).agg(
        n=('sample_id', 'size'),
        fg_median=('foreground_area_ratio', 'median'),
        fg_p05=('foreground_area_ratio', lambda x: x.quantile(.05)),
        fg_p95=('foreground_area_ratio', lambda x: x.quantile(.95)),
        all_background_n=('foreground_area_ratio', lambda x: int((x == 0).sum())),
        near_empty_lt_0p001_n=('foreground_area_ratio', lambda x: int((x < 1e-3).sum())),
        dead_positive_n=('dead_hard_area_ratio', lambda x: int((x > 0).sum())),
        border_touch_pixel_fraction_median=('border_touch_pixel_fraction', 'median'),
        any_border_touch_n=('border_touch_pixel_fraction', lambda x: int((x > 0).sum())),
        component_count_median=('fg_component_count', 'median'),
    ).reset_index()


# ---------------------------------------------------------------- 3. NEG normalization (Stage 21)
def neg_normalize(data):
    out = []
    for (model, ccl), g in data.groupby(['model', 'CCL'], sort=True):
        ref = g[g.label_role.eq('vehicle_viable_reference')]
        if ref.empty:
            raise RuntimeError(f'missing NEG reference {model}/{ccl}')
        x = g.copy()
        for f in RICH:
            x[f + '__neg_delta'] = x[f] - float(ref[f].median())
        out.append(x)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 4. nested LOCO (Stage 22)
def prep_fit(xtr, xte):
    imp = SimpleImputer(strategy='median').fit(xtr)
    a, b = imp.transform(xtr), imp.transform(xte)
    sc = StandardScaler().fit(a)
    return sc.transform(a), sc.transform(b)


def pick_alpha(X, y, groups):
    best_score, best_alpha = -np.inf, 10.0
    for alpha in ALPHAS:
        pred = np.full(len(y), np.nan)
        for tr, va in GroupKFold(min(5, len(np.unique(groups)))).split(X, y, groups):
            a, b = prep_fit(X[tr], X[va])
            pred[va] = Ridge(alpha=alpha).fit(a, y[tr]).predict(b)
        score = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


def loco(X, y, groups):
    pred = np.full(len(y), np.nan)
    for held in sorted(np.unique(groups)):
        tr, te = groups != held, groups == held
        alpha = pick_alpha(X[tr], y[tr], groups[tr])
        a, b = prep_fit(X[tr], X[te])
        pred[te] = Ridge(alpha=alpha).fit(a, y[tr]).predict(b)
    return pred


def metrics(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return {'n': int(len(y)), 'pearson': float(np.corrcoef(y, p)[0, 1]),
            'spearman': float(spearmanr(y, p).statistic),
            'r2': float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            'mae': float(np.mean(np.abs(y - p)))}


def bootstrap_draws(df, nboot=1000):
    rng = np.random.default_rng(SEED)
    ccl = df.CCL.to_numpy()
    cond = (df.CCL.astype(str) + '|' + df.Func.astype(str) + '|' + df.Cmpd1.astype(str) + '|'
            + df.Conc1.astype(str)).to_numpy()
    ccls = np.unique(ccl)
    by_ccl = {c: [np.where(cond == k)[0] for k in np.unique(cond[ccl == c])] for c in ccls}
    draws = []
    for _ in range(nboot):
        pieces = []
        for c in rng.choice(ccls, len(ccls), replace=True):
            groups = by_ccl[c]
            pieces.extend(groups[j] for j in rng.integers(0, len(groups), size=len(groups)))
        draws.append(np.concatenate(pieces))
    return draws


def run_loco(norm):
    results, preds, per_ccl, perms = [], [], [], []
    base = norm[norm.cohort_assay_wide.astype(bool)]
    for cohort, exclude52 in (('main_all9', False), ('sensitivity_excl_Organoid52', True)):
        draws = None
        for model in MODELS:
            d = base[base.model == model]
            if exclude52:
                d = d[d.CCL != 'Organoid52']
            d = d.sort_values('sample_id').reset_index(drop=True)
            if draws is None:
                draws = bootstrap_draws(d)
            y, groups = d.relative_ctg_activity.to_numpy(float), d.CCL.to_numpy()
            for spec, cols in LOCO_SPECS.items():
                X = d[cols].to_numpy(float)
                p = loco(X, y, groups)
                m = metrics(y, p)
                boot = np.array([[metrics(y[i], p[i])[k] for k in ('pearson', 'spearman', 'r2')] for i in draws])
                ci = {f'{k}_ci_lo': float(np.percentile(boot[:, j], 2.5)) for j, k in enumerate(('pearson', 'spearman', 'r2'))}
                ci |= {f'{k}_ci_hi': float(np.percentile(boot[:, j], 97.5)) for j, k in enumerate(('pearson', 'spearman', 'r2'))}
                results.append({'cohort': cohort, 'model': model, 'spec': spec, **m, **ci})
                preds.append(pd.DataFrame({'cohort': cohort, 'model': model, 'spec': spec, 'sample_id': d.sample_id,
                                           'CCL': d.CCL, 'Func': d.Func, 'Cmpd1': d.Cmpd1, 'Conc1': d.Conc1,
                                           'relative_ctg_activity': y, 'prediction': p}))
                for ccl, g in pd.DataFrame({'CCL': groups, 'y': y, 'p': p}).groupby('CCL'):
                    per_ccl.append({'cohort': cohort, 'model': model, 'spec': spec, 'CCL': ccl,
                                    **metrics(g.y.to_numpy(), g.p.to_numpy())})
                print(cohort, model, spec, {k: round(v, 3) for k, v in m.items()}, flush=True)
                if cohort == 'main_all9' and spec == PRIMARY_SPEC and model == PRIMARY_MODEL:
                    rng = np.random.default_rng(SEED + 1)
                    for r in range(N_PERM):
                        yp = y.copy()
                        for c in np.unique(groups):  # permute within training cell lines (Stage 23)
                            idx = np.where(groups == c)[0]
                            yp[idx] = yp[rng.permutation(idx)]
                        pp = loco(X, yp, groups)
                        perms.append({'model': model, 'spec': spec, 'permutation': r, **metrics(y, pp),
                                      'observed_pearson': m['pearson']})
    return pd.DataFrame(results), pd.concat(preds, ignore_index=True), pd.DataFrame(per_ccl), pd.DataFrame(perms)


# ---------------------------------------------------------------- 5. dose trajectories (Stage 24)
def dose_trajectories(oof):
    d = oof[(oof.cohort == 'main_all9') & oof.Func.isin(['SING', 'DRC1'])
            & oof.spec.isin([PRIMARY_SPEC, 'rich_neg_delta'])]
    rows = []
    for (model, spec, ccl, cmpd), g in d.groupby(['model', 'spec', 'CCL', 'Cmpd1']):
        z = g.groupby('Conc1', as_index=False).agg(n=('sample_id', 'size'), ctg=('relative_ctg_activity', 'median'),
                                                   morph=('prediction', 'median')).sort_values('Conc1')
        if z.Conc1.nunique() < 3:
            continue
        r_ctg = spearmanr(z.Conc1, z.ctg).statistic
        r_morph = spearmanr(z.Conc1, z.morph).statistic
        r_align = spearmanr(z.ctg, z.morph).statistic
        if abs(r_ctg) >= .5 and abs(r_morph) >= .5:
            cls = 'concordant' if np.sign(r_ctg) == np.sign(r_morph) else 'discordant'
        else:
            cls = 'weak_or_nonmonotonic'
        rows.append({'model': model, 'spec': spec, 'CCL': ccl, 'Cmpd1': cmpd, 'n_concentrations': len(z),
                     'rho_dose_ctg': r_ctg, 'rho_dose_morphology': r_morph, 'rho_ctg_vs_morphology': r_align,
                     'trajectory_class': cls})
    traj = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED + 2)
    summary = []
    for (model, spec), g in traj.groupby(['model', 'spec']):
        ccls = np.array(sorted(g.CCL.unique()))
        boot = []
        for _ in range(1000):
            z = pd.concat([g[g.CCL == c] for c in rng.choice(ccls, len(ccls), replace=True)])
            e = z[z.rho_dose_ctg.abs() >= .5]
            boot.append([z.rho_ctg_vs_morphology.median(), (e.trajectory_class == 'concordant').mean()])
        boot = np.asarray(boot)
        e = g[g.rho_dose_ctg.abs() >= .5]
        summary.append({'model': model, 'spec': spec, 'n_trajectories': len(g),
                        'median_rho_ctg_vs_morphology': g.rho_ctg_vs_morphology.median(),
                        'ci_lo': np.nanpercentile(boot[:, 0], 2.5), 'ci_hi': np.nanpercentile(boot[:, 0], 97.5),
                        'strong_ctg_n': len(e), 'concordant_fraction': (e.trajectory_class == 'concordant').mean(),
                        'concordant_ci_lo': np.nanpercentile(boot[:, 1], 2.5),
                        'concordant_ci_hi': np.nanpercentile(boot[:, 1], 97.5)})
    per_drug = traj.groupby(['model', 'spec', 'Cmpd1']).agg(
        n_cell_lines=('CCL', 'nunique'), median_rho_dose_ctg=('rho_dose_ctg', 'median'),
        median_rho_dose_morphology=('rho_dose_morphology', 'median'),
        median_rho_ctg_vs_morphology=('rho_ctg_vs_morphology', 'median'),
        concordant_n=('trajectory_class', lambda x: (x == 'concordant').sum()),
        discordant_n=('trajectory_class', lambda x: (x == 'discordant').sum())).reset_index()
    return traj, pd.DataFrame(summary), per_drug


# ---------------------------------------------------------------- 6. compound fingerprint (Stage 25)
def condition_table(norm, model, cols, exclude52):
    d = norm[(norm.model == model) & norm.cohort_drug_only.astype(bool)]
    presence = d.groupby('Cmpd1').CCL.nunique()
    common = sorted(presence[presence.eq(9)].index)
    d = d[d.Cmpd1.isin(common)]
    if exclude52:
        d = d[d.CCL != 'Organoid52']
    agg = {c: 'median' for c in cols} | {'relative_ctg_activity': 'median', 'sample_id': 'size'}
    z = d.groupby(['CCL', 'Cmpd1', 'Conc1'], as_index=False).agg(agg).rename(columns={'sample_id': 'n_wells'})
    z['log10_concentration'] = np.log10(z.Conc1.astype(float))
    return z, common


def fingerprint(norm):
    rows, per, preds = [], [], []
    for cohort, exclude52 in (('main_all9', False), ('sensitivity_excl_Organoid52', True)):
        for model in MODELS:
            for fset, base_cols in (('core', CORE), ('rich', RICH)):
                cols = [c + '__neg_delta' for c in base_cols]
                z, common = condition_table(norm, model, cols, exclude52)
                y = LabelEncoder().fit(common).transform(z.Cmpd1)
                groups = z.CCL.to_numpy()
                B = z[['log10_concentration', 'relative_ctg_activity']].to_numpy(float)
                M = z[cols].to_numpy(float)
                out = {k: np.zeros(len(z), int) for k in ('baseline', 'residual', 'baseline_plus_residual')}
                prob = {k: np.zeros((len(z), len(common))) for k in out}
                for held in sorted(np.unique(groups)):
                    tr, te = groups != held, groups == held
                    imp = SimpleImputer(strategy='median').fit(M[tr])
                    mtr, mte = imp.transform(M[tr]), imp.transform(M[te])
                    reg = LinearRegression().fit(B[tr], mtr)
                    rtr, rte = mtr - reg.predict(B[tr]), mte - reg.predict(B[te])
                    mats = {'baseline': (B[tr], B[te]), 'residual': (rtr, rte),
                            'baseline_plus_residual': (np.column_stack([B[tr], rtr]), np.column_stack([B[te], rte]))}
                    for k, (xtr, xte) in mats.items():
                        a, b = prep_fit(xtr, xte)
                        clf = LogisticRegression(C=1.0, max_iter=3000).fit(a, y[tr])
                        out[k][te] = clf.predict(b)
                        prob[k][te] = clf.predict_proba(b)
                for k in out:
                    preds.append(pd.DataFrame({'cohort': cohort, 'model': model, 'feature_set': fset, 'classifier': k,
                                               'CCL': z.CCL, 'Cmpd1': z.Cmpd1, 'Conc1': z.Conc1,
                                               'true': y, 'pred': out[k]}))
                ccl_ba = {k: np.array([balanced_accuracy_score(y[groups == c], out[k][groups == c])
                                       for c in sorted(np.unique(groups))]) for k in out}
                inc = ccl_ba['baseline_plus_residual'] - ccl_ba['baseline']
                rng = np.random.default_rng(SEED + 3)
                boot = np.array([inc[rng.integers(0, len(inc), len(inc))].mean() for _ in range(2000)])
                for c, ib, ip in zip(sorted(np.unique(groups)), ccl_ba['baseline'], ccl_ba['baseline_plus_residual']):
                    per.append({'cohort': cohort, 'model': model, 'feature_set': fset, 'CCL': c,
                                'baseline_ba': ib, 'baseline_plus_residual_ba': ip, 'increment': ip - ib})
                rows.append({'cohort': cohort, 'model': model, 'feature_set': fset,
                             'n_conditions': len(z), 'n_compounds': len(common), 'chance': 1 / len(common),
                             **{f'ba_{k}': balanced_accuracy_score(y, out[k]) for k in out},
                             'top3_baseline_plus_residual': top_k_accuracy_score(
                                 y, prob['baseline_plus_residual'], k=3, labels=np.arange(len(common))),
                             'mean_ccl_increment': inc.mean(), 'ccl_positive': int((inc > 0).sum()),
                             'n_ccl': len(inc), 'increment_ci_lo': np.percentile(boot, 2.5),
                             'increment_ci_hi': np.percentile(boot, 97.5),
                             'bootstrap_fraction_le_zero': float((boot <= 0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(per), pd.concat(preds, ignore_index=True)


# ---------------------------------------------------------------- 7. descriptive drug phenotypes
def drug_phenotypes(norm):
    """Residual RICH morphology per compound after concentration + CTG adjustment (pooled fit),
    standardized per feature; with the number of cell lines sharing the sign of the pooled mean."""
    rows = []
    for model in MODELS:
        cols = [c + '__neg_delta' for c in RICH]
        z, common = condition_table(norm, model, cols, exclude52=False)
        B = z[['log10_concentration', 'relative_ctg_activity']].to_numpy(float)
        M = SimpleImputer(strategy='median').fit_transform(z[cols].to_numpy(float))
        R = M - LinearRegression().fit(B, M).predict(B)
        R = (R - R.mean(0)) / np.where(R.std(0) > 0, R.std(0), 1)
        r = pd.DataFrame(R, columns=RICH).assign(CCL=z.CCL.values, Cmpd1=z.Cmpd1.values)
        for cmpd, g in r.groupby('Cmpd1'):
            by_ccl = g.groupby('CCL')[RICH].mean()
            for f in RICH:
                mean = float(g[f].mean())
                rows.append({'model': model, 'Cmpd1': cmpd, 'feature': f, 'residual_z_mean': mean,
                             'n_ccl': len(by_ccl), 'ccl_same_sign': int((np.sign(by_ccl[f]) == np.sign(mean)).sum())})
    return pd.DataFrame(rows)


def main():
    data, inputs = load()
    repro = reproducibility(data)
    repro.to_csv(OUT / 'reproducibility_checks.csv', index=False)
    print(repro.to_string(index=False), flush=True)
    q = qc(data)
    q.to_csv(OUT / 'qc_per_cell_line.csv', index=False)
    print(q.to_string(index=False), flush=True)
    norm = neg_normalize(data)
    norm.to_csv(OUT / 'features_raw_and_neg_delta.csv', index=False)

    res, oof, per_ccl, perms = run_loco(norm)
    res.to_csv(OUT / 'loco_model_comparison.csv', index=False)
    oof.to_csv(OUT / 'loco_oof_predictions.csv', index=False)
    per_ccl.to_csv(OUT / 'loco_per_cell_line.csv', index=False)
    perms.to_csv(OUT / 'loco_permutation_null.csv', index=False)
    perm_summary = perms.groupby('model').apply(lambda g: pd.Series({
        'observed_pearson': g.observed_pearson.iloc[0], 'null_max_pearson': g.pearson.max(),
        'empirical_p': (1 + (g.pearson >= g.observed_pearson.iloc[0]).sum()) / (1 + len(g))})).reset_index()
    perm_summary.to_csv(OUT / 'loco_permutation_summary.csv', index=False)

    traj, traj_summary, per_drug = dose_trajectories(oof)
    traj.to_csv(OUT / 'dose_trajectories.csv', index=False)
    traj_summary.to_csv(OUT / 'dose_summary.csv', index=False)
    per_drug.to_csv(OUT / 'dose_per_compound.csv', index=False)

    fp, fp_per, fp_pred = fingerprint(norm)
    fp.to_csv(OUT / 'fingerprint_summary.csv', index=False)
    fp_per.to_csv(OUT / 'fingerprint_per_cell_line.csv', index=False)
    fp_pred.to_csv(OUT / 'fingerprint_oof_predictions.csv', index=False)

    pheno = drug_phenotypes(norm)
    pheno.to_csv(OUT / 'drug_residual_phenotypes.csv', index=False)

    (OUT / 'run_manifest.json').write_text(json.dumps({
        'created': '2026-11-09', 'script': __file__, 'script_sha256': sha256(__file__),
        'inputs_sha256': inputs | {str(LABELS): sha256(LABELS)}, 'plan': str(STAGE / 'ANALYSIS_PLAN.md'),
        'primary_model': PRIMARY_MODEL, 'primary_spec': PRIMARY_SPEC, 'permutations': N_PERM,
        'loco_bootstrap': '1000 hierarchical (cell line -> condition)', 'fingerprint_bootstrap': '2000 cell-line',
        'segmentation_inference_rerun': False, 'training_rerun': False,
        'architecture_selection_by_ctg': False}, indent=2))
    pd.set_option('display.width', 250)
    print(res.to_string(index=False))
    print(perm_summary.to_string(index=False))
    print(traj_summary.to_string(index=False))
    print(fp.to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
