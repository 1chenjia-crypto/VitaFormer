#!/usr/bin/env python3
"""Visual-review sheets for Stage 39 (2026-11-10): all 3 models x 3 seeds, every cell line.
Wells chosen by CTG only (fixed quantiles 0.05 / 0.50 / 0.95 within each line, assay-wide cohort) before viewing any image.
Columns: M2 s42/s3407/s2026 | SegFormer s42/s3407/s2026 | nnU-Net V0 s42/s3407/s2026 (existing overlay JPGs, unchanged).
Output: 39_.../review_panels_all_models_20261110/ (new dir). Descriptive only."""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

A = Path('3D_OrganoidSeg/external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment')
S39 = A / '39_segformer_b2_deeporganoid_20261109'
OUT = S39 / 'review_panels_all_models_20261110'
COLS = [(f'M2 s{s}', A / f'36_m2_deeporganoid_20261109/m2_seed{s}/overlays_jpg') for s in (42, 3407, 2026)] + \
       [(f'SegF s{s}', S39 / f'segformer_seed{s}/overlays_jpg') for s in (42, 3407, 2026)] + \
       [(f'nnU s{s}', A / f'35_nnunet_external_comparison_20261102/nnunet_seed{s}/overlays_jpg') for s in (42, 3407, 2026)]
Q = (0.05, 0.50, 0.95)
T = 220


def main():
    OUT.mkdir(exist_ok=False)
    d = pd.read_csv(S39 / 'analysis/features_raw_and_neg_delta.csv', usecols=['sample_id', 'CCL', 'Func', 'Cmpd1', 'Conc1', 'model',
                                                                           'relative_ctg_activity', 'cohort_assay_wide'], low_memory=False)
    d = d[(d.model == 'vf_m2_s42') & d.cohort_assay_wide.astype(bool)]
    picks = []
    for ccl, g in d.groupby('CCL'):
        g = g.sort_values(['relative_ctg_activity', 'sample_id']).reset_index(drop=True)
        rows = [g.iloc[int(round(q * (len(g) - 1)))] for q in Q]
        sheet = Image.new('RGB', (len(COLS) * T, len(rows) * (T + 34) + 20), 'white')
        dr = ImageDraw.Draw(sheet)
        for j, (name, _) in enumerate(COLS):
            dr.text((j * T + 4, 4), name, fill='black')
        for i, r in enumerate(rows):
            y = 20 + i * (T + 34)
            dr.text((4, y + 2), f'{r.sample_id} | {r.Func} {r.Cmpd1} {r.Conc1} | CTG {r.relative_ctg_activity:.2f} (q{Q[i]:.2f})', fill='black')
            for j, (_, src) in enumerate(COLS):
                sheet.paste(Image.open(src / f'{r.sample_id}_overlay.jpg').convert('RGB').resize((T, T)), (j * T, y + 32))
            picks.append({'CCL': ccl, 'quantile': Q[i], 'sample_id': r.sample_id, 'Func': r.Func, 'Cmpd1': r.Cmpd1,
                          'Conc1': r.Conc1, 'relative_ctg_activity': r.relative_ctg_activity})
        sheet.save(OUT / f'{ccl}_9models.jpg', quality=88)
    pd.DataFrame(picks).to_csv(OUT / 'picked_wells.csv', index=False)
    print(pd.DataFrame(picks).to_string(index=False))


if __name__ == '__main__':
    main()
