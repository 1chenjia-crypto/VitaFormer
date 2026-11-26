#!/usr/bin/env python3
"""Build oreval inputs from the paper's existing per-image readouts (local only; not for redistribution of third-party data).
Outputs go to validation/work/ (readouts per run + metadata) for NViR and DeepOrganoid."""
from pathlib import Path
import pandas as pd

R = Path('3D_OrganoidSeg')
W = Path(__file__).parent / 'work'; W.mkdir(exist_ok=False)
# NViR: readouts from the analysis feature table (fg_area of each run), metadata from the authors' CSV
f = pd.read_csv(R / 'functional_validation_nvir_20261109/analysis/features_normalised.csv', low_memory=False)
runs = ['m2_s42', 'm2_s3407', 'm2_s2026', 'segformer_s42', 'segformer_s3407', 'segformer_s2026', 'nnunet_s42', 'nnunet_s3407', 'nnunet_s2026']
(W / 'nvir').mkdir()
for r in runs:
    f[f.rep == r][['image', 'fg_area']].rename(columns={'fg_area': 'foreground_area_px'}).to_csv(W / f'nvir/{r}.csv', index=False)
m = pd.read_csv(R / 'external_datasets/NViR_zenodo15374443/raw/data_labeled.csv').rename(columns={'location_in_bucket': 'image'})
m[['image', 'location', 'microscope', 'study', 'plate', 'day', 'well', 'split', 'domain', 'compound', 'treatment', 'concentration_uM',
   'value']].rename(columns={'value': 'ATP'}).to_csv(W / 'nvir/metadata.csv', index=False)
# DeepOrganoid: foreground_area_ratio per sample from the Stage 39 feature table (contains M2, SegFormer and nnU-Net V0 rows)
d = pd.read_csv(R / 'external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment/39_segformer_b2_deeporganoid_20261109/analysis/features_raw_and_neg_delta.csv',
                usecols=['sample_id', 'model', 'CCL', 'Func', 'Cmpd1', 'Conc1', 'foreground_area_ratio', 'relative_ctg_activity',
                         'label_role', 'cohort_drug_only'], low_memory=False)
(W / 'deeporganoid').mkdir()
for mdl in ['vf_m2_s42', 'vf_m2_s3407', 'vf_m2_s2026', 'segformer_s42', 'segformer_s3407', 'segformer_s2026', 'nnunet_s42', 'nnunet_s3407', 'nnunet_s2026']:
    d[d.model == mdl][['sample_id', 'foreground_area_ratio']].rename(columns={'sample_id': 'image'}).to_csv(W / f'deeporganoid/{mdl}.csv', index=False)
meta = d[d.model == 'vf_m2_s42'].drop(columns=['model', 'foreground_area_ratio']).rename(columns={'sample_id': 'image'})
meta['ranking_row'] = meta.cohort_drug_only.astype(bool) & meta.Func.isin(['SING', 'DRC1'])
meta.to_csv(W / 'deeporganoid/metadata.csv', index=False)
print('ok', len(m), len(meta))
