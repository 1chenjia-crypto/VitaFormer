#!/usr/bin/env python3
"""Stage 38A analysis (POST HOC, 2026-11-09): copy of the Stage 37 script; models = M2 s42 (REF) + nnU-Net s42 V0 (Stage 35)
+ V1/V2/V3 (Stage 38A) [+ seeds 3407/2026 of a variant if the confirmation rule fires, via --confirm]. Output 38/A_analysis.
Original: Stage 37 (2026-11-09): copy of the Stage 35 analysis with REF = frozen M2 seed42 (Stage 36 features); only model map, paths,
reference and output dir changed. Plan: 37_m2_vs_nnunet_20261109/ANALYSIS_PLAN.md.
Original docstring: Stage 35 analysis: frozen nnU-Net v2 (3 seeds) vs frozen VitaFormer (Stage 31) on DeepOrganoid.
Plan: 35_nnunet_external_comparison_20261102/ANALYSIS_PLAN.md (written before inference). Methods are the Stage 31/32/34
functions, unchanged; added here only: nnU-Net as a representation and paired differences with shared draws.
E1 LOCO CTG association (+ permutations for nnU-Net s42), E2 dose-trajectory consistency, E3 patient-line ranking and
pooled potency; cohorts all nine / excluding Organoid52 (/ excluding LPA for E3). QC tables + contact sheets.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import analyze_stage31_deeporganoid_two_models_20261029 as s31  # noqa: E402
import stage34_core_claims_robustness_20261030 as s34  # noqa: E402

BASE = s31.BASE
STAGE = BASE / '35_nnunet_external_comparison_20261102'   # nnU-Net inputs (read-only)
OUTSTAGE = BASE / '38_posthoc_robustness_20261109'
RES = OUTSTAGE / 'A_analysis'
S36 = BASE / '36_m2_deeporganoid_20261109'
S31 = BASE / '31_final_model_p1p99_full_20261029'
S32N = BASE / '32_segmentation_value_baselines_20261029/analysis/all_representations_raw_and_neg_delta.csv'
REF = 'vf_m2_s42'
MODELS = {'vf_m2_s42': ('s36', 'm2_seed42'), 'nnunet_s42_V0': ('s35', STAGE / 'nnunet_seed42'),
          'nnunet_s42_V1': ('s38', OUTSTAGE / 'A_nnunet_V1/nnunet_seed42'), 'nnunet_s42_V2': ('s38', OUTSTAGE / 'A_nnunet_V2/nnunet_seed42'),
          'nnunet_s42_V3': ('s38', OUTSTAGE / 'A_nnunet_V3/nnunet_seed42')}
if '--confirm' in sys.argv:  # confirmation rule fired for variant sys.argv[idx+1]: add its seeds 3407/2026 (+ V0 seeds)
    _v = sys.argv[sys.argv.index('--confirm') + 1]
    RES = OUTSTAGE / f'A_analysis_confirm_{_v}'
    for _s in (3407, 2026):
        MODELS[f'nnunet_s{_s}_V0'] = ('s35', STAGE / f'nnunet_seed{_s}')
        MODELS[f'nnunet_s{_s}_{_v}'] = ('s38', OUTSTAGE / f'A_nnunet_{_v}/nnunet_seed{_s}')
PRIMARY_NN = 'nnunet_s42_V0'
SEED_LOCO, SEED_E2, SEED_E3 = s31.SEED, 20261102, s34.SEED
NBOOT_E2, NBOOT_E3 = 1000, s34.NBOOT


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    labels = pd.read_csv(s31.LABELS)
    meta = labels[['sample_id', 'relative_ctg_activity', 'label_role', 'cohort_assay_wide', 'cohort_drug_only']]
    frames, inputs = [], {}
    for name, (src, key) in MODELS.items():
        files = (sorted((S31 / key).glob('features_shard*of3.csv')) if src == 's31' else
                 [S36 / key / 'features_shard0of1.csv'] if src == 's36' else [Path(key) / 'features.csv'])
        d = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        if len(d) != 3456 or d.sample_id.duplicated().any() or d.mask_reused.any():
            raise RuntimeError(f'{name}: not 3456 unique freshly inferred rows')
        inputs.update({str(p): sha256(p) for p in files})
        frames.append(d.assign(model=name).merge(meta, on='sample_id', how='left', validate='one_to_one'))
    return pd.concat(frames, ignore_index=True), inputs


# ------------------------------------------------------------------ QC + domain differences
def mask_agreement(data):
    """Descriptive: per-well foreground Dice between VF-simplified and each nnU-Net seed (assay-wide wells)."""
    ids = data[(data.model == REF) & data.cohort_assay_wide.astype(bool)].sample_id.tolist()
    rows = []
    for name, (src, key) in MODELS.items():
        if src not in ('s35', 's38'):
            continue
        for sid in ids:
            a = np.isin(np.asarray(Image.open(S36 / 'm2_seed42/hard_class_masks' / f'{sid}_classes.png')), (1, 2))
            b = np.isin(np.asarray(Image.open(Path(key) / f'hard_class_masks/{sid}_classes.png')), (1, 2))
            s = a.sum() + b.sum()
            rows.append({'model': name, 'sample_id': sid, 'fg_dice_vs_vf_m2_s42': 2 * (a & b).sum() / s if s else np.nan,
                         'vf_fg': a.mean(), 'nn_fg': b.mean()})
    return pd.DataFrame(rows)


def contact_sheets(data):
    """Per line: 6 wells at fixed CTG quantiles (chosen by CTG only); overlays M2 s42 | nnU-Net s42 V0 | V1 | V2 | V3."""
    out = RES / 'review_panels'; out.mkdir(exist_ok=False)
    cols = [S36 / 'm2_seed42/overlays_jpg'] + [Path(MODELS[k][1]) / 'overlays_jpg' for k in
                                               ('nnunet_s42_V0', 'nnunet_s42_V1', 'nnunet_s42_V2', 'nnunet_s42_V3')]
    d = data[(data.model == REF) & data.cohort_assay_wide.astype(bool)]
    for ccl, g in d.groupby('CCL'):
        g = g.sort_values('relative_ctg_activity').reset_index(drop=True)
        pick = [g.iloc[int(round(q * (len(g) - 1)))] for q in (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)]
        tile, rows = 240, []
        for r in pick:
            row = Image.new('RGB', (len(cols) * tile, tile + 20), 'white')
            for j, c in enumerate(cols):
                row.paste(Image.open(c / f'{r.sample_id}_overlay.jpg').resize((tile, tile)), (j * tile, 20))
            ImageDraw.Draw(row).text((4, 4), f'{r.sample_id} CTG {r.relative_ctg_activity:.2f} | M2 s42 | nnU-Net V0 | V1 | V2 | V3', fill='black')
            rows.append(row)
        sheet = Image.new('RGB', (len(cols) * tile, len(rows) * (tile + 20)), 'white')
        for i, row in enumerate(rows):
            sheet.paste(row, (0, i * (tile + 20)))
        sheet.save(out / f'{ccl}_m2_vs_nnunet_variants.jpg', quality=88)


# ------------------------------------------------------------------ E1 LOCO, paired
def e1(norm):
    base = norm[norm.cohort_assay_wide.astype(bool)]
    res, diffs, oof, perms = [], [], [], []
    for cohort, excl in (('main_all9', False), ('sensitivity_excl_Organoid52', True)):
        ref = base[base.model == REF]
        if excl:
            ref = ref[ref.CCL != 'Organoid52']
        ref = ref.sort_values('sample_id').reset_index(drop=True)
        draws = s31.bootstrap_draws(ref)  # identical to Stage 31/32 (same seed, same ordering)
        y, groups = ref.relative_ctg_activity.to_numpy(float), ref.CCL.to_numpy()
        boot = {}
        for m in MODELS:
            d = base[base.model == m].set_index('sample_id').loc[ref.sample_id].reset_index()
            for spec, cols in s31.LOCO_SPECS.items():
                p = s31.loco(d[cols].to_numpy(float), y, groups)
                met = s31.metrics(y, p)
                b = np.array([[s31.metrics(y[i], p[i])[k] for k in ('pearson', 'spearman', 'r2')] for i in draws])
                boot[(m, spec)] = b
                res.append({'cohort': cohort, 'model': m, 'spec': spec, **met,
                            'pearson_ci_lo': np.percentile(b[:, 0], 2.5), 'pearson_ci_hi': np.percentile(b[:, 0], 97.5),
                            'spearman_ci_lo': np.percentile(b[:, 1], 2.5), 'spearman_ci_hi': np.percentile(b[:, 1], 97.5)})
                oof.append(pd.DataFrame({'cohort': cohort, 'model': m, 'spec': spec, 'sample_id': ref.sample_id, 'CCL': groups,
                                         'Func': ref.Func, 'Cmpd1': ref.Cmpd1, 'Conc1': ref.Conc1,
                                         'relative_ctg_activity': y, 'prediction': p}))
                print(cohort, m, spec, round(met['pearson'], 4), flush=True)
                if cohort == 'main_all9' and spec == s31.PRIMARY_SPEC and m == PRIMARY_NN:
                    rng = np.random.default_rng(SEED_LOCO + 1)  # Stage 31 permutation scheme
                    X = d[cols].to_numpy(float)
                    for r in range(s31.N_PERM):
                        yp = y.copy()
                        for c in np.unique(groups):
                            idx = np.where(groups == c)[0]
                            yp[idx] = yp[rng.permutation(idx)]
                        perms.append({'model': m, 'permutation': r, **s31.metrics(y, s31.loco(X, yp, groups)),
                                      'observed_pearson': met['pearson']})
        for m in MODELS:
            if m == REF:
                continue
            for spec in s31.LOCO_SPECS:
                dl = boot[(REF, spec)] - boot[(m, spec)]
                o = {r['model']: r for r in res if r['cohort'] == cohort and r['spec'] == spec}
                diffs.append({'cohort': cohort, 'spec': spec, 'comparison': f'{REF} - {m}',
                              'delta_pearson': o[REF]['pearson'] - o[m]['pearson'],
                              'ci_lo': np.percentile(dl[:, 0], 2.5), 'ci_hi': np.percentile(dl[:, 0], 97.5),
                              'delta_spearman': o[REF]['spearman'] - o[m]['spearman'],
                              'spearman_ci_lo': np.percentile(dl[:, 1], 2.5), 'spearman_ci_hi': np.percentile(dl[:, 1], 97.5),
                              'delta_r2': o[REF]['r2'] - o[m]['r2'], 'frac_boot_le_0': float((dl[:, 0] <= 0).mean())})
    return pd.DataFrame(res), pd.DataFrame(diffs), pd.concat(oof, ignore_index=True), pd.DataFrame(perms)


# ------------------------------------------------------------------ E2 dose trajectories, paired
def e2(oof):
    traj, summary_unpaired, per_drug = s31.dose_trajectories(oof)
    rows = []
    rng = np.random.default_rng(SEED_E2)
    for spec, g in traj.groupby('spec'):
        ccls = np.array(sorted(g.CCL.unique()))
        draws = [rng.choice(ccls, len(ccls)) for _ in range(NBOOT_E2)]

        def stats(h):
            e = h[h.rho_dose_ctg.abs() >= .5]
            return h.rho_ctg_vs_morphology.median(), (e.trajectory_class == 'concordant').mean()
        by = {m: {c: h[h.CCL == c] for c in ccls} for m, h in g.groupby('model')}  # every model has every line here
        boot = {m: np.array([stats(pd.concat([by[m][c] for c in dr])) for dr in draws]) for m in by}
        obs = {m: stats(h) for m, h in g.groupby('model')}
        for m in by:
            rows.append({'spec': spec, 'model': m, 'median_rho_ctg_vs_morphology': obs[m][0],
                         'rho_ci_lo': np.nanpercentile(boot[m][:, 0], 2.5), 'rho_ci_hi': np.nanpercentile(boot[m][:, 0], 97.5),
                         'concordant_fraction': obs[m][1], 'conc_ci_lo': np.nanpercentile(boot[m][:, 1], 2.5),
                         'conc_ci_hi': np.nanpercentile(boot[m][:, 1], 97.5)})
            if m != REF:
                d = boot[REF] - boot[m]
                rows.append({'spec': spec, 'model': f'delta {REF} - {m}', 'median_rho_ctg_vs_morphology': obs[REF][0] - obs[m][0],
                             'rho_ci_lo': np.nanpercentile(d[:, 0], 2.5), 'rho_ci_hi': np.nanpercentile(d[:, 0], 97.5),
                             'concordant_fraction': obs[REF][1] - obs[m][1], 'conc_ci_lo': np.nanpercentile(d[:, 1], 2.5),
                             'conc_ci_hi': np.nanpercentile(d[:, 1], 97.5)})
    return traj, summary_unpaired, per_drug, pd.DataFrame(rows)


# ------------------------------------------------------------------ E3 line ranking + pooled potency, paired
def model_free_auc(norm):
    d = norm[norm.cohort_drug_only.astype(bool) & norm.Func.isin(['SING', 'DRC1'])].copy()
    neg = norm[norm.label_role.eq('vehicle_viable_reference')].groupby(['model', 'CCL']).foreground_area_ratio.median()
    d['area_rel_neg'] = d.foreground_area_ratio / d.set_index(['model', 'CCL']).index.map(neg).values
    return {m: s34.auc_table(g, 'area_rel_neg') for m, g in d.groupby('model')}


def e3(aucs):
    s34.PRIMARY = REF
    rng = np.random.default_rng(SEED_E3)
    rows, pot = [], []
    variants = {'all9_all_drugs': lambda a: a, 'excl_Organoid52': lambda a: a[a.CCL != 'Organoid52'],
                'excl_LPA_posthoc': lambda a: a[a.Cmpd1 != 'LPA']}
    for label, f in variants.items():
        sub = {k: f(v) for k, v in aucs.items()}
        rows += s34.ranking_block(sub, label, rng)
        ccls = np.array(sorted(sub[REF].CCL.unique()))
        draws = [rng.choice(ccls, len(ccls)) for _ in range(NBOOT_E3)]
        by = {k: {c: v[v.CCL == c] for c in ccls} for k, v in sub.items()}
        b = {k: np.array([spearmanr(*pd.concat([by[k][c] for c in dr])[['auc_ctg', 'auc_x']].T.to_numpy()).statistic
                          for dr in draws]) for k in sub}
        for k, v in sub.items():
            est = spearmanr(v.auc_ctg, v.auc_x).statistic
            pot.append({'cohort': label, 'representation': k, 'n_pairs': len(v), 'spearman_auc': est,
                        'ci_lo': np.nanpercentile(b[k], 2.5), 'ci_hi': np.nanpercentile(b[k], 97.5)})
            if k != REF:
                d = b[REF] - b[k]
                pot.append({'cohort': label, 'representation': f'delta {REF} - {k}', 'n_pairs': len(v),
                            'spearman_auc': spearmanr(sub[REF].auc_ctg, sub[REF].auc_x).statistic - est,
                            'ci_lo': np.nanpercentile(d, 2.5), 'ci_hi': np.nanpercentile(d, 97.5)})
    return pd.DataFrame(rows), pd.DataFrame(pot)


def main():
    RES.mkdir(parents=True, exist_ok=False)
    data, inputs = load()
    q = s31.qc(data); q.to_csv(RES / 'qc_per_cell_line.csv', index=False)
    print(q.to_string(index=False), flush=True)
    agree = mask_agreement(data); agree.to_csv(RES / 'mask_agreement_vs_vf_m2_s42.csv', index=False)
    print(agree.groupby('model')[['fg_dice_vs_vf_m2_s42', 'vf_fg', 'nn_fg']].median(), flush=True)
    contact_sheets(data)
    norm = s31.neg_normalize(data); norm.to_csv(RES / 'features_raw_and_neg_delta.csv', index=False)
    res, diffs, oof, perms = e1(norm)
    res.to_csv(RES / 'E1_loco.csv', index=False); diffs.to_csv(RES / 'E1_paired_differences.csv', index=False)
    oof.to_csv(RES / 'E1_loco_oof_predictions.csv', index=False); perms.to_csv(RES / 'E1_permutation_null_nnunet_s42_V0.csv', index=False)
    perm_p = (1 + (perms.pearson >= perms.observed_pearson.iloc[0]).sum()) / (1 + len(perms))
    traj, traj_unpaired, per_drug, e2p = e2(oof[oof.cohort == 'main_all9'])
    traj.to_csv(RES / 'E2_dose_trajectories.csv', index=False); traj_unpaired.to_csv(RES / 'E2_dose_summary_stage31_method.csv', index=False)
    per_drug.to_csv(RES / 'E2_dose_per_compound.csv', index=False); e2p.to_csv(RES / 'E2_paired.csv', index=False)
    aucs = model_free_auc(norm)
    thr = pd.read_csv(S32N)
    thr = thr[thr.representation.isin(['threshold_local_contrast', 'threshold_global_otsu'])].copy()
    thr['model'] = thr.representation
    aucs |= model_free_auc(thr)
    pd.concat([v.assign(representation=k) for k, v in aucs.items()]).to_csv(RES / 'E3_auc_table.csv', index=False)
    rank, pot = e3(aucs)
    rank.to_csv(RES / 'E3_line_ranking.csv', index=False); pot.to_csv(RES / 'E3_pooled_potency.csv', index=False)
    (RES / 'run_manifest.json').write_text(json.dumps({
        'created': '2026-11-09', 'script': __file__, 'script_sha256': sha256(__file__),
        'plan': str(OUTSTAGE / 'PLAN_A_nnunet_config_sensitivity.md'), 'posthoc': True, 'inputs_sha256': inputs | {str(s31.LABELS): sha256(s31.LABELS), str(S32N): sha256(S32N)},
        'reference': REF, 'primary_nnunet': PRIMARY_NN, 'nnunet_s42_core_permutation_p': perm_p,
        'nnunet_s42_core_permutation_null_max': float(perms.pearson.max()),
        'seeds': {'loco_draws': SEED_LOCO, 'e2': SEED_E2, 'e3': SEED_E3}, 'training_rerun': False,
        'checkpoint_selection_by_deeporganoid': False}, indent=2))
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 50)
    print(res[res.spec.isin(['core_neg_delta', 'area_raw', 'rich_neg_delta'])].round(3).to_string(index=False))
    print(diffs.round(3).to_string(index=False))
    print('nnU-Net s42 core permutation p', perm_p, 'null max', perms.pearson.max())
    print(e2p.round(3).to_string(index=False))
    print(rank.drop(columns=['per_drug'], errors='ignore').round(3).to_string(index=False))
    print(pot.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
