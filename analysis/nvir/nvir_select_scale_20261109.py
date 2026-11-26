#!/usr/bin/env python3
"""NViR scale selection (ANALYSIS_PLAN §4 + Amendment 1). Seed 42, train_val rows only, foreground (classes 1|2) Dice vs the
authors' reference masks cropped exactly as the authors' crop_spheroid.py (PIL crop, box rounded by Pillow, zero fill
outside). Chooses, per family, the scale with the highest median Dice; ties -> smaller S. Never reads the ATP column.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path('3D_OrganoidSeg')
DATA = ROOT / 'external_datasets/NViR_zenodo15374443'
OUT = ROOT / 'functional_validation_nvir_20261109'
SCALES = ['256', '384', '512', '768', 'native']
USECOLS = ['location_in_bucket', 'split', 'domain', 'xmin', 'ymin', 'xmax', 'ymax']


def ref_crop(row, size=0.2):
    """Authors' crop_spheroid.py applied to the reference mask (same geometry as the image crop)."""
    m = Image.open(DATA / 'extracted/images_labeled_masks' / row.location_in_bucket).convert('L')
    s = min(m.height, m.width) * size
    xmin, ymin, xmax, ymax = row.xmin * m.width, row.ymin * m.height, row.xmax * m.width, row.ymax * m.height
    cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
    top, left = cy - s, cx - s
    return np.asarray(m.crop((left, top, left + 2 * s, top + 2 * s))) > 0


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def main(families):
    df = pd.read_csv(DATA / 'raw/data_labeled.csv', usecols=USECOLS)
    df = df[df.split == 'train_val']
    rows, choice = [], {}
    for fam in families:
        for S in SCALES:
            d = OUT / 'masks' / f'{fam}_s42_S{S}'
            for r in df.itertuples():
                ref = ref_crop(r)
                pred = np.isin(np.asarray(Image.open(d / r.location_in_bucket)), (1, 2))
                if pred.shape != ref.shape:
                    raise RuntimeError(f'{r.location_in_bucket}: shape {pred.shape} vs ref {ref.shape}')
                rows.append({'family': fam, 'scale': S, 'image': r.location_in_bucket, 'domain': r.domain,
                             'dice': dice(pred, ref), 'pred_fg': pred.mean(), 'ref_fg': ref.mean()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'scale_selection_per_image.csv', index=False)
    summ = t.groupby(['family', 'scale'], sort=False).dice.agg(['median', 'mean', 'count']).reset_index()
    summ.to_csv(OUT / 'scale_selection_summary.csv', index=False)
    for fam, g in summ.groupby('family', sort=False):
        best = g['median'].max()
        cand = [s for s in SCALES if np.isclose(g.set_index('scale').loc[s, 'median'], best)]
        choice[fam] = cand[0]
    (OUT / 'scale_selection_choice.json').write_text(json.dumps(choice, indent=2))
    print(summ.round(4).to_string(index=False)); print('chosen', choice)


if __name__ == '__main__':
    main(sys.argv[1:] or ['m2', 'segformer', 'nnunet'])
