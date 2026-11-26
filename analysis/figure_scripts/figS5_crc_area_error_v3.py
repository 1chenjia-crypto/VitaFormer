#!/usr/bin/env python3
"""Figure S5 (draft v3, 2026-11-16, review): H adds per-seed MEAN (x) next to the median/IQR so the text's mean signed values are visible; caption states statistics and n.
Figure S5 (draft): CRC test-set area and viability-ratio error (secondary, post hoc).
Source: crc_area_error_posthoc_20261111/ (plan written before computation). Per training seed: median (dot) and IQR (bar)
over eligible images. OrganoidViT reproduction = single seed, neutral reference marker."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from style import R, COL, MRK, INK2, MUTED, BASE, SURF, REF, clean, dot, save, OUT

import os
A = Path(os.environ.get('S5_SRC', str(R / 'crc_area_error_posthoc_v2_orgvit3seed')))
ss = pd.read_csv(A / 'seed_summary.csv')
fd = pd.read_csv(A / 'false_dead_no_gt_dead.csv')
ORDER = ['VitaFormer', 'SegFormer-B2', 'nnU-Net', 'OrganoidViT']
XL = ['Vita-\nFormer', 'SegFormer\n-B2', 'nnU-\nNet', 'Organoid-\nViT*']
SEEDS = [42, 3407, 2026]


def panel(ax, cls, ep, scale=1.0, zero=False, show_mean=False):
    for i, m in enumerate(ORDER):
        sub = ss[(ss.model == m) & (ss['class'] == cls) & (ss.endpoint == ep)]
        for j, s in enumerate(SEEDS):
            r = sub[sub.seed == s]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (j - 1) * .17
            if m.startswith('Organoid'):
                dot(ax, x, r['median'] * scale, None, r.q1 * scale, r.q3 * scale, color=REF, marker='D', ms=4.5)
            else:
                dot(ax, x, r['median'] * scale, m, r.q1 * scale, r.q3 * scale)
            if show_mean:
                ax.plot(x + .07, r['mean'] * scale, 'x', color=INK2, ms=4, mew=1.1, zorder=6)
    if zero:
        ax.axhline(0, color=INK2, lw=.8, zorder=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(XL, color=INK2, fontsize=6.5); clean(ax)


fig = plt.figure(figsize=(7.4, 7.6)); gs = fig.add_gridspec(3, 3, hspace=.75, wspace=.42)
n = {c: int(ss[(ss['class'] == c) & (ss.endpoint == 'abs_rel')].n.iloc[0]) for c in ['Alive', 'Dead', 'Bubble']}
L = 'ABC'
for k, c in enumerate(['Alive', 'Dead', 'Bubble']):
    ax = fig.add_subplot(gs[0, k]); panel(ax, c, 'abs_rel', 100); ax.set_ylim(bottom=0)
    ax.set_title(f'{L[k]}  {c} |area error| (n={n[c]})'); ax.set_ylabel('|P − G| / G (%)' if k == 0 else '')
L = 'DEF'
for k, c in enumerate(['Alive', 'Dead', 'Bubble']):
    ax = fig.add_subplot(gs[1, k]); panel(ax, c, 'signed_rel', 100, zero=True)
    ax.set_title(f'{L[k]}  {c} signed error'); ax.set_ylabel('(P − G) / G (%)\n← under · over →' if k == 0 else '')
ax = fig.add_subplot(gs[2, 0]); panel(ax, 'Dead', 'abs_px', 1e-3); ax.set_ylim(bottom=0)
ax.set_title('G  Dead |pixel error| (n=24)'); ax.set_ylabel('|P − G| (×10³ px)')
ax = fig.add_subplot(gs[2, 1]); panel(ax, 'Viability[dead_present]', 'dV_pp', 1, zero=True, show_mean=True)
ax.set_title('H  Alive fraction error\n    Dead present (n=24)'); ax.set_ylabel('pred − GT (pp)')
ax = fig.add_subplot(gs[2, 2])
for i, m in enumerate(ORDER):
    sub = fd[fd.model == m]
    for j, s in enumerate(SEEDS):
        r = sub[sub.seed == s]
        if r.empty:
            continue
        y = r.n_with_pred_dead.iloc[0] / r.n_images_no_gt_dead.iloc[0] * 100
        if m.startswith('Organoid'):
            dot(ax, i + (j - 1) * .17, y, None, color=REF, marker='D', ms=4.5)
        else:
            dot(ax, i + (j - 1) * .17, y, m)
ax.set_xticks(range(4)); ax.set_xticklabels(XL, color=INK2, fontsize=6.5); clean(ax); ax.set_ylim(0, 45)
ax.set_title('I  False Dead\n    no Dead in GT (n=143)'); ax.set_ylabel('Images with any\npredicted Dead (%)')
fig.text(0.01, -0.01, 'Dots: per-seed median over images (seeds 42 / 3407 / 2026, left→right); bars: IQR over images (not CIs); H, ×: per-seed mean. '
         '*OrganoidViT reproduction, 3 seeds (descriptive reference, no contrasts).\nSecondary, post hoc analysis; '
         'CRC test set (167 images, 154 wells), not blind. A–F computed on images whose annotation contains the class (Alive 167, Dead 24, Bubble 22);\nI on the 143 images without annotated Dead. H: means are pulled by a few images with large errors, so mean and median can differ in sign (text reports means).', fontsize=6.2, color=MUTED, va='top')
save(fig, os.environ.get('S5_NAME', 'FigS5_crc_area_error_v3'))
src = ss[ss.endpoint.isin(['abs_rel', 'signed_rel', 'abs_px', 'dV_pp'])]
n = os.environ.get('S5_NAME', 'FigS5_crc_area_error_v3')
src.to_csv(OUT / f'{n}_source_data.csv', index=False)
fd.to_csv(OUT / f'{n}_I_source_false_dead.csv', index=False)
print('saved')
