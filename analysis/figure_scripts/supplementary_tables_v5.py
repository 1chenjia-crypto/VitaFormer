#!/usr/bin/env python3
"""Supplementary Tables S1-S5 for manuscript v5 (2026-11-16). Reads existing result files only; writes CSVs and one Markdown file
(figures_draft/SUPPLEMENTARY_TABLES_v5.md). No new analysis.
S1 per-seed functional values (existing CSV) | S2 NViR all scales (existing CSV) | S3 CRC all seven models (FigS2 per-seed CSV)
S4 VitaFormer ablations (Stage D + m2_ablations_eval) | S5 nnU-Net ablations (nnunet_ablations_eval)."""
import json
import pandas as pd
from style import R, OUT

D = R / 'vco/runs/arch_exploration_20261104/stageD_selection_free/analysis_final'
A = R / 'vco/runs/m2_ablations_eval_20261113'
N = R / 'baselines_fair_20261030/nnunet_ablations_eval_20261113'
F = R / 'vco/runs/arch_exploration_20261104/stageF_m2_test/comparison/seed_level.csv'
EPS = [('mIoU', 'mIoU'), ('iou_alive', 'Alive IoU'), ('iou_dead', 'Dead IoU'), ('dead_recall', 'Dead recall'),
       ('dead_precision', 'Dead precision'), ('fg_bf1_2', 'fg BF1 (2 px)')]


def fmt_ci(r, nd=2):
    return f'{r.delta:+.{nd}f} [{r.ci_lo:+.{nd}f}, {r.ci_hi:+.{nd}f}]'


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    out += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(out)


# ---- S4 VitaFormer ablations
pcD = pd.read_csv(D / 'paired_comparisons.csv'); pcA = pd.read_csv(A / 'analysis_final/paired_comparisons.csv')
sl = pd.read_csv(F).set_index(['group', 'endpoint'])
test = {'w/o CNN detail branch': sl.loc[('M0-final', 'mIoU'), ['seed42', 'seed3407', 'seed2026']].astype(float)}
for k, nm in [('noRD', 'w/o rare-class recipe'), ('ImageNet', 'ImageNet initialisation')]:
    test[nm] = pd.Series([json.load(open(A / f'shared_eval_test/{k}_final_seed{s}.json'))['mIoU'] for s in ['42', '3407', '2026']])
ABL = [('w/o CNN detail branch', pcD, 'M2-M0'), ('CNN branch, zero input', pcD, 'M2-M1'),
       ('w/o rare-class recipe', pcA, 'M2-noRD'), ('ImageNet initialisation', pcA, 'M2-ImageNet')]
rows = []
for nm, pc, comp in ABL:
    row = {'Variant (full VitaFormer − variant)': nm}
    for ep, lab in EPS:
        r = pc[(pc.comparison == comp) & (pc.seed == 'mean_of_3') & (pc.endpoint == ep)]
        if r.empty:
            row[lab] = 'n/a'; continue
        r = r.iloc[0]; nd = 3 if ep.startswith('fg_bf1') else 2
        row[lab] = f'{fmt_ci(r, nd)}; {int(r.n_seeds_positive)}/3'
    t = test.get(nm)
    row['Test mIoU of variant (descriptive)'] = 'not evaluated' if t is None else f'{t.mean():.2f} ± {t.std(ddof=1):.2f}'
    rows.append(row)
s4 = pd.DataFrame(rows)
fp = pd.read_csv(A / 'analysis_final/per_model_summary.csv', header=[0, 1], index_col=0)
fpm = fp[('mean', 'far_bg_dead_fp_px')]
s4_note = (f'Validation set (164 images), final checkpoint (step 40,000), seed-matched mean of 3 seeds, hierarchical bootstrap over 154 wells '
           f'(10,000 replicates); "k/3" = seeds with Δ > 0. Full VitaFormer test mIoU 83.34 ± 0.25. Far-background false Dead pixels '
           f'(validation, per seed, mean of 3 seeds): full model {fpm["M2"]:,.0f}; w/o rare-class recipe {fpm["noRD"]:,.0f}; ImageNet '
           f'{fpm["ImageNet"]:,.0f}. The rare-class flag switches off focal loss, class weights, Dead-centred cropping, sparse-Dead zoom, '
           f'5-FU and general oversampling, Dead-pixel dropout and Dead copy-paste.')
s4.to_csv(OUT / 'TableS4_vitaformer_ablations.csv', index=False)

# ---- S5 nnU-Net ablations
pcN = pd.read_csv(N / 'analysis_final/paired_comparisons.csv')
rows = []
for nm, comp, seed, tst in [('512 × 512 patches (3 seeds)', 'nnUNet-p512', 'mean_of_3', '84.77 ± 0.46 (default 85.05 ± 0.71)'),
                            ('cross-entropy only (seed 42)', 'nnUNet-CE', '42', '83.66 (default s42 84.28); Dead recall 64.95 vs 73.76')]:
    row = {'Variant (default nnU-Net − variant)': nm}
    for ep, lab in EPS:
        r = pcN[(pcN.comparison == comp) & (pcN.seed == seed) & (pcN.endpoint == ep)].iloc[0]
        nd = 3 if ep.startswith('fg_bf1') else 2
        row[lab] = fmt_ci(r, nd) + (f'; {int(r.n_seeds_positive)}/3' if seed == 'mean_of_3' else '')
    row['Test (descriptive)'] = tst
    rows.append(row)
s5 = pd.DataFrame(rows)
s5_note = ('Exploratory. Validation set, nnU-Net\'s own fold-0 validation predictions from checkpoint_final; same bootstrap as Table S4. '
           'Pre-specified rules: patch 512 |Δ mIoU| < 1.0 → advantage not explained by field of view (met); CE-only Dead-recall drop '
           '5–10 points → inconclusive (one seed). Runs were resumed after file-system outages (recorded in the deposited plans).')
s5.to_csv(OUT / 'TableS5_nnunet_ablations.csv', index=False)

# ---- S3 all CRC models
p = pd.read_csv(OUT / 'FigS2_source_per_seed.csv')
cols = [c for c in p.columns if c not in ('model', 'seed')]
order = ['VitaFormer', 'nnU-Net', 'SegFormer-B2', 'OrganoidViT', 'DeepLabv3', 'U-Net++', 'U-Net']
g = p.groupby('model')
s3 = pd.DataFrame({c: (g[c].mean().map('{:.2f}'.format) + ' ± ' + g[c].std(ddof=1).map('{:.2f}'.format)) for c in cols}).loc[order]
s3.insert(0, 'Model', s3.index); s3.to_csv(OUT / 'TableS3_crc_all_models.csv', index=False)
s3_note = ('CRC test set (167 images), shared evaluator, mean ± SD over seeds 42/3407/2026. Area error = median per-image |pred − GT|/GT '
           'over images whose annotation contains the class (Alive 167, Dead 24, Bubble 22). OrganoidViT, DeepLabv3, U-Net++ and U-Net: '
           'models of the source publication, best-validation checkpoints; several runs diverged (Figure S1). Descriptive; no contrasts.')

# ---- S1, S2 existing
s1 = pd.read_csv(OUT / 'TableS1_per_seed.csv'); s1r = pd.read_csv(OUT / 'TableS1_observed_seed_ranges.csv')
s2 = pd.read_csv(OUT / 'TableS2_nvir_all_scales.csv')
for t in (s1, s1r, s2):
    for c in t.columns:
        if t[c].dtype.kind == 'f':
            t[c] = t[c].map(lambda x: f'{x:.3f}' if pd.notna(x) else '')

doc = ['# Supplementary tables (draft v5, 2026-11-16)', '',
       '*Generated by `figure_scripts/supplementary_tables_v5.py` from existing result files; no new analysis.*', '',
       '## Table S1. Per-seed values of all endpoints', '', md(s1), '',
       'Observed seed range (maximum − minimum over three seeds):', '', md(s1r), '',
       'DO = DeepOrganoid; E1 Pearson r (calibrated, CI over lines); E2 dose-direction concordance; E3 mean within-drug ρ (CI over drugs); '
       'E3b pooled ρ; NViR N1 pooled area–ATP ρ (CI over physical plate-day clusters); N1b dose-direction series; test Dice per domain.', '',
       '## Table S2. NViR results at every candidate input scale (seed 42)', '', md(s2), '',
       'Pre-specified supplementary table. Scale chosen by training-split Dice against the reference masks (ATP not read); identifying '
       'an ATP-optimal scale from this table is post hoc.', '',
       '## Table S3. CRC test results for all seven models', '', md(s3), '', s3_note, '',
       '## Table S4. VitaFormer ablations', '', md(s4), '', s4_note, '',
       '## Table S5. Exploratory nnU-Net ablations', '', md(s5), '', s5_note, '']
(OUT / 'SUPPLEMENTARY_TABLES_v5.md').write_text('\n'.join(doc))
print(s4.to_string()); print(s5.to_string())
