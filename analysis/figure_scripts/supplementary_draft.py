#!/usr/bin/env python3
"""Supplementary drafts.
Table S1: per-seed values for all three segmenters (CRC, DeepOrganoid E1/E2/E3/E3b, NViR N1/N1b/Dice) + seed ranges.
Table S2: NViR seed-42 results at every candidate scale (pre-specified supplementary table; ATP-optimal scale = post hoc reading).
Figure S3: DeepOrganoid mask inspection (9 lines x 3 CTG quantiles x 9 model-seed columns; wells chosen by CTG before viewing).
Figure S4: NViR stratified inspection (4 domains x 3 ATP tertiles x 2 wells; VitaFormer/SegFormer/nnU-Net s42 vs reference)."""
import pandas as pd
from PIL import Image
from style import R, ANA, NV, OUT

S39 = ANA / '39_segformer_b2_deeporganoid_20261109/analysis'
crc = pd.read_csv(R / 'vco/runs/arch_exploration_20261104/stageF_m2_test/comparison/seed_level.csv')
crc = crc[crc.endpoint == 'mIoU'].set_index('group')
e1 = pd.read_csv(S39 / 'E1_loco.csv'); e1 = e1[(e1.cohort == 'main_all9') & (e1.spec == 'core_neg_delta')].set_index('model')
e3 = pd.read_csv(S39 / 'E3_line_ranking.csv'); e3 = e3[e3.cohort == 'all9_all_drugs'].set_index('representation')
e3b = pd.read_csv(S39 / 'E3_pooled_potency.csv'); e3b = e3b[e3b.cohort == 'all9_all_drugs'].set_index('representation')
e2 = pd.read_csv(S39 / 'E2_dose_summary_stage31_method.csv'); e2 = e2[e2.spec == 'core_neg_delta'].set_index('model')
n1 = pd.read_csv(NV / 'analysis_n1_cluster_fix_20261110/N1_pooled_corrected.csv'); n1 = n1[n1.analysis == 'pooled_all_images_physical_clusters'].set_index('rep')
n1b = pd.read_csv(NV / 'analysis/N1b_dose_concordance.csv').set_index('rep')
n3 = pd.read_csv(NV / 'analysis/N3_dice_vs_reference_test.csv')
FAM = {'VitaFormer': ('M2-final', 'vf_m2_s{}', 'm2_s{}'), 'SegFormer-B2': ('SegFormer-B2', 'segformer_s{}', 'segformer_s{}'),
       'nnU-Net (V0)': ('nnU-Net', 'nnunet_s{}', 'nnunet_s{}')}
rows = []
for fam, (cg, dk, nk) in FAM.items():
    for s in ['42', '3407', '2026']:
        d, n = dk.format(s), nk.format(s)
        dice = n3[n3.rep == n].set_index('domain').dice_vs_ref
        rows.append({'model': fam, 'seed': s, 'CRC_mIoU': crc.loc[cg, f'seed{s}'],
                     'DO_E1_pearson': e1.loc[d, 'pearson'], 'DO_E1_ci': f"[{e1.loc[d, 'pearson_ci_lo']:.3f}, {e1.loc[d, 'pearson_ci_hi']:.3f}]",
                     'DO_E2_concordant': e2.loc[d, 'concordant_fraction'] if d in e2.index else float('nan'),
                     'DO_E3_mean_rho': e3.loc[d, 'mean_rho'], 'DO_E3_ci': f"[{e3.loc[d, 'ci_lo']:.3f}, {e3.loc[d, 'ci_hi']:.3f}]",
                     'DO_E3b_rho': e3b.loc[d, 'spearman_auc'],
                     'NViR_N1_rho': n1.loc[n, 'rho'], 'NViR_N1_ci': f"[{n1.loc[n, 'ci_lo']:.3f}, {n1.loc[n, 'ci_hi']:.3f}]",
                     'NViR_N1b': f"{int(n1b.loc[n, 'concordant'])}/{int(n1b.loc[n, 'n_series'])}",
                     'NViR_test_Dice_A/B/C/D': '/'.join(f'{dice[k]:.2f}' for k in ['A_echo_rebel', 'B_echo_revolve_v0', 'A_cytation_v0', 'A_thorlabs'])})
t1 = pd.DataFrame(rows)
rng = t1.groupby('model')[['CRC_mIoU', 'DO_E1_pearson', 'DO_E3_mean_rho', 'DO_E3b_rho', 'NViR_N1_rho']].agg(lambda x: x.max() - x.min())
t1.to_csv(OUT / 'TableS1_per_seed.csv', index=False); rng.to_csv(OUT / 'TableS1_observed_seed_ranges.csv')
# S2
a = pd.read_csv(NV / 'analysis/N1_area_vs_atp.csv'); a = a[a.domain == 'pooled']
ss = pd.read_csv(NV / 'scale_selection_summary.csv'); n1b_all = pd.read_csv(NV / 'analysis/N1b_dose_concordance.csv').set_index('rep')
choice = {'m2': '256', 'segformer': '256', 'nnunet': '512'}
import sys
sys.path.insert(0, str(R / 'organoid_readout_eval'))
from oreval import core  # same cluster bootstrap as the corrected primary analysis (seed 20261109, 2,000 draws)
_f = pd.read_csv(NV / 'analysis/features_normalised.csv', usecols=['rep', 'image', 'fg_area_n', 'atp_n', 'split', 'location', 'study', 'plate', 'day'], low_memory=False)
_f = _f[_f.split == 'test']


def _assoc(rep):
    g = _f[_f.rep == rep].sort_values('image').reset_index(drop=True)
    return core.association(g, 'fg_area_n', 'atp_n', ['location', 'study', 'plate', 'day']).iloc[0]
r2 = []
for fam in ['m2', 'segformer', 'nnunet']:
    for sc in ['256', '384', '512', '768', 'native']:
        rep = f'{fam}_s42_S{sc}'; x = _assoc(rep)
        r2.append({'family': fam, 'scale': sc, 'pre_specified_choice': sc == choice[fam],
                   'train_median_Dice': ss[(ss.family == fam) & (ss.scale == sc)]['median'].iloc[0],
                   'test_N1_rho': x.rho, 'ci_lo (physical-plate clusters)': x.ci_lo, 'ci_hi': x.ci_hi,
                   'N1b': f"{int(n1b_all.loc[rep, 'concordant'])}/{int(n1b_all.loc[rep, 'n_series'])}"})
t2 = pd.DataFrame(r2); t2.to_csv(OUT / 'TableS2_nvir_all_scales.csv', index=False)
# S3 / S4 multipage PDFs from existing inspection sheets (unchanged images)
s3 = sorted((ANA / '39_segformer_b2_deeporganoid_20261109/review_panels_all_models_20261110').glob('*_9models.jpg'))
s4 = sorted((NV / 'review_stratified_20261110').glob('*_stratified.jpg'))
for name, files in [('FigS3_deeporganoid_mask_inspection.pdf', s3), ('FigS4_nvir_stratified_inspection.pdf', s4)]:
    ims = [Image.open(f).convert('RGB') for f in files]
    ims[0].save(OUT / name, save_all=True, append_images=ims[1:], resolution=200)
print(t1.round(3).to_string(index=False)); print(rng.round(3)); print(t2.round(3).to_string(index=False))
print(len(s3), 'S3 pages;', len(s4), 'S4 pages')
