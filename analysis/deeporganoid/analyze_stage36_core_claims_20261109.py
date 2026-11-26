#!/usr/bin/env python3
"""Stage 36 core claims (Stage 34 part A + noise ceiling, methods unchanged) with M2 as the primary representation.
Representations: Stage 32 table (Otsu, local contrast, VitaFormer simplified = M0, Full; read-only) + M2 seeds 42/3407/2026
(Stage 36 features). Functions are imported from stage34_core_claims_robustness_20261030.py unchanged; only PRIMARY/REPS
and the noise-ceiling feature source are set here. Plan: 36_m2_deeporganoid_20261109/ANALYSIS_PLAN.md §5. New directory only."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import stage34_core_claims_robustness_20261030 as s34  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

BASE = s34.BASE; STAGE = BASE / '36_m2_deeporganoid_20261109'; RES = STAGE / 'analysis/core_claims'
S36F = STAGE / 'analysis/features_raw_and_neg_delta.csv'
M2 = {'m2_seed42': 'vitaformer_m2', 'm2_seed3407': 'vitaformer_m2_seed3407', 'm2_seed2026': 'vitaformer_m2_seed2026'}
s34.PRIMARY = 'vitaformer_m2'
s34.REPS = ['threshold_global_otsu', 'threshold_local_contrast', 'vitaformer_simplified', 'vitaformer_full'] + list(M2.values())


def noise_ceiling(f_all, model, rng):
    """Stage 34 noise_ceiling, unchanged except the feature source/model."""
    f = f_all[(f_all.model == model) & f_all.cohort_drug_only.astype(bool) & f_all.Func.isin(['SING', 'DRC1'])].copy()
    neg = f_all[(f_all.model == model) & f_all.label_role.eq('vehicle_viable_reference')].groupby('CCL').foreground_area_ratio.median()
    f['area_rel_neg'] = f.foreground_area_ratio / f.CCL.map(neg)
    f = f.sort_values(['CCL', 'Cmpd1', 'Conc1', 'Row', 'Col']); f['rep'] = f.groupby(['CCL', 'Cmpd1', 'Conc1']).cumcount()
    assert f.groupby(['CCL', 'Cmpd1', 'Conc1']).size().eq(4).all()
    rows = []
    for a_idx in ((0, 1), (0, 2), (0, 3)):
        A, B = f[f.rep.isin(a_idx)], f[~f.rep.isin(a_idx)]
        aa = A.groupby(['CCL', 'Cmpd1', 'Conc1'])[['relative_ctg_activity', 'area_rel_neg']].mean().groupby(['CCL', 'Cmpd1']).mean()
        bb = B.groupby(['CCL', 'Cmpd1', 'Conc1'])[['relative_ctg_activity', 'area_rel_neg']].mean().groupby(['CCL', 'Cmpd1']).mean()
        pairs = {'ctg_A_vs_ctg_B': (aa.relative_ctg_activity, bb.relative_ctg_activity),
                 'area_A_vs_ctg_B': (aa.area_rel_neg, bb.relative_ctg_activity),
                 'area_B_vs_ctg_A': (bb.area_rel_neg, aa.relative_ctg_activity),
                 'area_A_vs_area_B': (aa.area_rel_neg, bb.area_rel_neg)}
        for name, (x, y) in pairs.items():
            t = pd.DataFrame({'auc_x': x, 'auc_ctg': y}).reset_index()
            for label, sub in (('all9', t), ('excl_Organoid52', t[t.CCL != 'Organoid52'])):
                r = s34.per_drug_rho(sub)
                rows.append({'pairing': str(a_idx), 'comparison': name, 'cohort': label, 'within_drug_mean_rho': r.mean(),
                             'pooled_potency_spearman': spearmanr(sub.auc_x, sub.auc_ctg).statistic})
    d = pd.DataFrame(rows)
    return d, d.groupby(['cohort', 'comparison'])[['within_drug_mean_rho', 'pooled_potency_spearman']].mean().reset_index()


def main():
    RES.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(s34.SEED)
    s32 = pd.read_csv(s34.S32 / 'all_representations_raw_and_neg_delta.csv', low_memory=False)
    f36 = pd.read_csv(S36F, low_memory=False)
    add = f36[f36.model.isin(list(M2))].copy(); add['representation'] = add.model.map(M2)
    cols = ['representation', 'sample_id', 'CCL', 'Func', 'Cmpd1', 'Conc1', 'cohort_drug_only', 'cohort_assay_wide',
            'label_role', 'foreground_area_ratio', 'relative_ctg_activity']
    norm = pd.concat([s32[cols], add[cols]], ignore_index=True)
    a_rank, a_pot, aucs = s34.part_a(norm, rng)
    a_rank.to_csv(RES / 'A_line_ranking_ci_perm_paired.csv', index=False)
    a_pot.to_csv(RES / 'A_pooled_potency_by_cohort.csv', index=False)
    pd.concat([v.assign(representation=k) for k, v in aucs.items()]).to_csv(RES / 'A_auc_table.csv', index=False)
    nc_all, nc = noise_ceiling(f36, 'm2_seed42', rng)
    nc_all.to_csv(RES / 'A_noise_ceiling_per_pairing.csv', index=False); nc.to_csv(RES / 'A_noise_ceiling.csv', index=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (RES / 'run_manifest.json').write_text(json.dumps({'created': '2026-11-09', 'script': __file__, 'script_sha256': sha(__file__),
        'inputs': {str(p): sha(p) for p in (s34.S32 / 'all_representations_raw_and_neg_delta.csv', S36F)},
        'primary': s34.PRIMARY, 'seed': s34.SEED, 'n_boot': s34.NBOOT, 'n_perm': s34.NPERM}, indent=2))
    pd.set_option('display.width', 250)
    print(a_rank.drop(columns=['per_drug']).round(3).to_string(index=False)); print(a_pot.round(3).to_string(index=False))
    print(nc.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
