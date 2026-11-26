#!/usr/bin/env python3
"""Copy (never modify) the result files behind each main-figure panel into paper_materials_20261110/source_data/,
with a SHA256 manifest of the originals. One derived table is written: Fig6_per_seed_summary.csv, assembled from E1/E3 CSVs.
"""
from pathlib import Path
import hashlib
import shutil
import pandas as pd

R = Path('3D_OrganoidSeg')
A = R / 'external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment'
B = R / 'baselines_fair_20261030'
N = R / 'functional_validation_nvir_20261109'
OUT = R / 'paper_materials_20261110/source_data'

FILES = {
    'Fig2_CRC': [R / 'crc_all_baselines_4class_20261109/all_models_4class_test167.csv',
                 R / 'vco/runs/arch_exploration_20261104/stageF_m2_test/RESULTS.md',
                 B / 'unified_comparison_20261102/contrasts.csv', B / 'unified_comparison_stable_20261102/contrasts.csv'],
    'Fig3_external_seg': [B / 'external_m2_20261108/organoidnet/analysis_B/per_image_metrics.csv',
                          B / 'external_m2_20261108/organoidnet/analysis_B/contrasts.csv',
                          B / 'brain_organoid2024_m2_20261108/analysis/per_image_metrics.csv',
                          B / 'brain_organoid2024_m2_20261108/analysis/contrasts.csv',
                          B / 'brain_organoid2024_m2_20261108/analysis/fold_summary.csv',
                          B / 'external_m2_20261108/organoidid_human/analysis_B/per_image_metrics.csv',
                          B / 'external_m2_20261108/organoidid_mouse/analysis_B/per_image_metrics.csv',
                          B / 'external_m2_20261108/orgasegment/analysis_B/per_image_metrics.csv',
                          B / 'external_m2_20261108/orgasegment/analysis_B/summary.csv'],
    'Fig4_DeepOrganoid': [A / '39_segformer_b2_deeporganoid_20261109/analysis/E1_loco.csv',
                          A / '39_segformer_b2_deeporganoid_20261109/analysis/E1_paired_differences.csv',
                          A / '39_segformer_b2_deeporganoid_20261109/analysis/E2_dose_trajectories.csv',
                          A / '39_segformer_b2_deeporganoid_20261109/analysis/E3_line_ranking.csv',
                          A / '39_segformer_b2_deeporganoid_20261109/analysis/E3_pooled_potency.csv',
                          A / '39_segformer_b2_deeporganoid_20261109/analysis/E1_permutation_null_segformer_s42.csv',
                          A / '39_segformer_b2_deeporganoid_20261109/B_general_sensitivity/summary.csv',
                          A / '36_m2_deeporganoid_20261109/README.md'],
    'Fig5_NViR_Gemcitabine': [N / 'analysis/N1_area_vs_atp.csv', N / 'analysis_n1_cluster_fix_20261110/N1_pooled_corrected.csv',
                              N / 'analysis_n1_cluster_fix_20261110/N1_pooled_paired_corrected.csv',
                              N / 'analysis_n1_cluster_fix_20261110/sample_counts.csv',
                              N / 'analysis/N1b_dose_concordance.csv', N / 'analysis/N2_supervised.csv',
                              N / 'analysis/N3_dice_vs_reference_test.csv', N / 'scale_selection_summary.csv',
                              N / 'review_stratified_20261110/picked_wells.csv',
                              B / 'external_m2_20261108/gemcitabine_pi/analysis/summary.csv',
                              B / 'external_m2_20261108/gemcitabine_pi/analysis/timecourse_summary.csv'],
    'Fig6_accuracy_vs_reproducibility': [A / '38_posthoc_robustness_20261109/A_RESULTS.md',
                                         A / '39_segformer_b2_deeporganoid_20261109/review_panels_all_models_20261110/picked_wells.csv'],
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    man = []
    for fig, files in FILES.items():
        (OUT / fig).mkdir()
        for p in files:
            if not p.exists():
                man.append({'figure': fig, 'source': str(p), 'copied_as': '', 'sha256': 'MISSING'}); continue
            dst = OUT / fig / ('__'.join(p.relative_to(R).parts[-3:]))
            shutil.copy2(p, dst)
            man.append({'figure': fig, 'source': str(p.relative_to(R)), 'copied_as': str(dst.relative_to(OUT)), 'sha256': sha(p)})
    # derived Fig 6 table (from Stage 39 analysis CSVs, which contain M2, SegFormer, nnU-Net V0 rows)
    e1 = pd.read_csv(A / '39_segformer_b2_deeporganoid_20261109/analysis/E1_loco.csv')
    e1 = e1[(e1.cohort == 'main_all9') & (e1.spec == 'core_neg_delta')][['model', 'pearson', 'pearson_ci_lo', 'pearson_ci_hi']]
    e3 = pd.read_csv(A / '39_segformer_b2_deeporganoid_20261109/analysis/E3_line_ranking.csv')
    e3 = e3[(e3.cohort == 'all9_all_drugs') & ~e3.representation.str.startswith('delta')][['representation', 'mean_rho', 'ci_lo', 'ci_hi']]
    crc = {'vf_m2': 83.34, 'segformer': 82.92, 'nnunet': 85.05, 'vf_simplified': 82.78}
    t = e1.merge(e3, left_on='model', right_on='representation', how='outer')
    t['family'] = t.model.fillna(t.representation).str.replace(r'_s\d+$', '', regex=True)
    t['crc_mIoU_3seed_mean'] = t.family.map(crc)
    t['note'] = 'E1 = LOCO Ridge (CTG-fitted); E3 = within-drug line ranking (no CTG fitting); CI E1 over 9 lines, E3 over 10 drugs; nnU-Net = V0 default'
    t.to_csv(OUT / 'Fig6_accuracy_vs_reproducibility/Fig6_per_seed_summary.csv', index=False)
    pd.DataFrame(man).to_csv(OUT / 'MANIFEST.csv', index=False)
    print(pd.DataFrame(man)[['figure', 'sha256']].assign(ok=lambda d: d.sha256 != 'MISSING').groupby('figure').ok.agg(['sum', 'count']))


if __name__ == '__main__':
    main()
