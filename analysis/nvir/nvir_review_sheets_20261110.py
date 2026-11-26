#!/usr/bin/env python3
"""NViR visual review (2026-11-10). Pre-fixed stratified sample, not chosen by looking at images:
per test domain x ATP tertile (low / mid / high of normalised ATP within that domain) -> 2 wells, random_state=20261110.
Each row = one well; columns = M2 s42 | SegFormer s42 | nnU-Net s42 (red contour, at their pre-selected scales) vs reference mask
(green contour). Output: functional_validation_nvir_20261109/review_stratified_20261110/ + picked_wells.csv."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, '3D_OrganoidSeg/scripts')
from nvir_select_scale_20261109 import ref_crop, DATA  # noqa: E402

O = Path('3D_OrganoidSeg/functional_validation_nvir_20261109')
OUT = O / 'review_stratified_20261110'
COLS = [('M2 s42 (S256)', 'm2_s42_S256'), ('SegFormer s42 (S256)', 'segformer_s42_S256'), ('nnU-Net s42 (S512)', 'nnunet_s42_S512')]
T = 300


def edge(b):
    e = np.zeros_like(b)
    e[:-1] |= b[:-1] ^ b[1:]; e[:, :-1] |= b[:, :-1] ^ b[:, 1:]
    return np.asarray(Image.fromarray(e.astype(np.uint8) * 255).resize(b.shape[::-1])) > 0


def main():
    OUT.mkdir(exist_ok=False)
    f = pd.read_csv(O / 'analysis/features_normalised.csv', usecols=['rep', 'image', 'atp_n', 'fg_area_n', 'split', 'domain', 'compound',
                                                                     'concentration_uM', 'day', 'study', 'plate', 'well'], low_memory=False)
    ref = f[(f.rep == 'reference') & (f.split == 'test')].copy()
    area = f[f.split == 'test'].pivot_table(index='image', columns='rep', values='fg_area_n')
    csv = pd.read_csv(DATA / 'raw/data_labeled.csv').set_index('location_in_bucket')
    picks = []
    for dom, g in ref.groupby('domain'):
        g = g.copy(); g['stratum'] = pd.qcut(g.atp_n.rank(method='first'), 3, labels=['low', 'mid', 'high'])
        sel = pd.concat([s.sample(2, random_state=20261110) for _, s in g.groupby('stratum', observed=True)])
        sheet = Image.new('RGB', (len(COLS) * T, len(sel) * (T + 30) + 20), 'white'); dr = ImageDraw.Draw(sheet)
        for j, (name, _) in enumerate(COLS):
            dr.text((j * T + 4, 4), f'{name} red | reference green', fill='black')
        for i, r in enumerate(sel.itertuples()):
            y = 20 + i * (T + 30)
            dr.text((4, y + 2), f'{r.stratum} | {r.study} p{r.plate} {r.well} d{r.day} | {str(r.compound)[:14]} {r.concentration_uM} | ATP {r.atp_n:.2f}', fill='black')
            row = csv.loc[r.image]
            gray = np.asarray(Image.open(DATA / 'extracted/images_labeled_cropped_0.2' / r.image).convert('RGB'))
            rm = ref_crop(row.rename(None).to_frame().T.assign(location_in_bucket=r.image).iloc[0])
            for j, (name, d) in enumerate(COLS):
                pm = np.isin(np.asarray(Image.open(O / 'masks' / d / r.image)), (1, 2))
                a = gray.copy()
                k = max(1, a.shape[0] // 300)
                for m, c in ((rm, (0, 220, 0)), (pm, (230, 0, 0))):
                    e = edge(m)
                    for _ in range(k):  # thicken for visibility
                        e = e | np.roll(e, 1, 0) | np.roll(e, 1, 1)
                    a[e] = c
                t = Image.fromarray(a).resize((T, T)); ImageDraw.Draw(t).text((4, T - 14), f'area_n {area.loc[r.image, d.rsplit("_S", 1)[0]]:.2f} ref {area.loc[r.image, "reference"]:.2f}', fill=(255, 255, 0))
                sheet.paste(t, (j * T, y + 28))
            picks.append({'domain': dom, 'stratum': r.stratum, 'image': r.image, 'atp_n': r.atp_n})
        sheet.save(OUT / f'{dom}_stratified.jpg', quality=88)
    pd.DataFrame(picks).to_csv(OUT / 'picked_wells.csv', index=False)
    print(pd.DataFrame(picks).to_string(index=False))


if __name__ == '__main__':
    main()
