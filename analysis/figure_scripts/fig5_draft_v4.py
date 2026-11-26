#!/usr/bin/env python3
"""v4 (2026-11-18): E now shows all three families x 3 seeds (Amendment 2, analysis_families/summary.csv); rest unchanged from v3.
v3 (2026-11-16, review): E unit "frames with any predicted Dead pixel", E status = pre-specified comparator (72 h and seeds 3407/2026 descriptive); D pixel scale bar (no physical calibration).
Figure 5 (draft v2, 2026-11-16; v1 = fig5_draft.py): v2 enlarges D (full-width row), moves the contour key under D,
puts C labels above points, adds status badges. Content unchanged.
 input scale, boundary placement and the Dead class.
A NViR scale selection: median foreground Dice vs author reference masks, train_val, seed 42 (pre-specified; ATP not read)
B NViR test: median control-normalised area by ATP quintile (VitaFormer s42, reference masks, Otsu)   [post hoc summary]
C NViR test: VitaFormer s42 Dice vs reference by ATP quintile                                           [post hoc summary]
D Example wells (pre-fixed random draw, review_stratified_20261110): low and high ATP, VitaFormer vs reference contours [post hoc]
E GemcitabineScreen: area fold change vs PI, Spearman per seed at 52/72 h; author OrganoID area for reference (same wells)
F NViR N2 (ATP-fitted, secondary): GBR Set A R2 per domain, 3 VitaFormer seeds vs published CVE / Pathology+CVE / NViR.
NViR images: Zenodo 15374443, MIT licence (attribution in legend)."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from style import R, NV, EXT, COL, INK2, MUTED, BASE, SURF, REF, clean, save, tag, OUT

sys.path.insert(0, str(R / 'scripts'))
from nvir_select_scale_20261109 import ref_crop, DATA  # noqa: E402

ss = pd.read_csv(NV / 'scale_selection_summary.csv')
f = pd.read_csv(NV / 'analysis/features_normalised.csv', usecols=['rep', 'image', 'fg_area_n', 'atp_n', 'dice_vs_ref', 'split', 'domain'], low_memory=False)
t = f[f.split == 'test']
p = t.pivot_table(index='image', columns='rep', values='fg_area_n')
p['atp'] = t[t.rep == 'reference'].set_index('image').atp_n
p['dice_m2'] = t[t.rep == 'm2_s42'].set_index('image').dice_vs_ref
p['q'] = pd.qcut(p.atp.rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
qb = p.groupby('q', observed=True).agg(atp=('atp', 'median'), m2=('m2_s42', 'median'), reference=('reference', 'median'), otsu=('otsu', 'median'),
                                       dice_m2=('dice_m2', 'median'), frac_dice_lt_08=('dice_m2', lambda x: (x < .8).mean()), n=('atp', 'size'))
gem = pd.read_csv(EXT / 'external_m2_20261108/gemcitabine_pi/analysis_families/summary.csv')
n2 = pd.read_csv(NV / 'analysis/N2_supervised.csv')
n2 = n2[(n2.model == 'gbr') & (n2.features == 'A') & n2.rep.isin(['m2_s42', 'm2_s3407', 'm2_s2026'])]
DOM = {'A_echo_rebel': 'A', 'B_echo_revolve_v0': 'B', 'A_cytation_v0': 'C', 'A_thorlabs': 'D'}
PUB = {'CVE': [.16, .56, .32, .68], 'Pathology+CVE': [.56, .70, .50, .62], 'NViR (end-to-end)': [.83, .84, .70, .80]}  # Table 1, PDF-verified
ss.to_csv(OUT / 'Fig5A_source_scale.csv', index=False); qb.to_csv(OUT / 'Fig5BC_source_quintiles.csv')
gem[['model', 'time_h', 'n_wells', 'rho_M2_fg_fc_PI', 'rho_M2_fg_fc_PI_lo', 'rho_M2_fg_fc_PI_hi', 'rho_author_area_fc_PI',
     'rho_author_area_fc_PI_lo', 'rho_author_area_fc_PI_hi']].to_csv(OUT / 'Fig5E_source_gemcitabine.csv', index=False)
n2.to_csv(OUT / 'Fig5F_source_N2.csv', index=False)

fig = plt.figure(figsize=(7.6, 9.0)); G = fig.add_gridspec(3, 3, hspace=.62, wspace=.55, height_ratios=[1, 1.45, 1])
# A
ax = fig.add_subplot(G[0, 0]); scales = ['256', '384', '512', '768', 'native']
for fam, m in [('m2', 'VitaFormer'), ('segformer', 'SegFormer-B2'), ('nnunet', 'nnU-Net')]:
    y = ss[ss.family == fam].set_index('scale').loc[scales, 'median']
    ax.plot(range(5), y, '-', color=COL[m], lw=1.6); ax.plot(range(5), y, 'o' if m == 'VitaFormer' else ('s' if 'Seg' in m else '^'), color=COL[m], ms=4)
    best = int(np.argmax(y.values)); ax.plot(best, y.iloc[best], 'o', ms=9, mfc='none', mec=COL[m], mew=1)
ax.set_xticks(range(5)); ax.set_xticklabels(['256', '384', '512', '768', 'nat.'], color=INK2); ax.set_xlabel('Input side (px)')
ax.set_ylabel('Median Dice vs reference'); ax.set_ylim(0, 1.02); clean(ax); ax.set_title('A  Scale selection (train split)', pad=12); tag(ax, 'pre-specified', y=1.0)
for m, mk in [('VitaFormer', 'o'), ('SegFormer-B2', 's'), ('nnU-Net', '^')]:
    ax.plot([], [], mk + '-', color=COL[m], ms=4, label=m)
ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.32), ncol=3, labelcolor=INK2, handlelength=1.2, columnspacing=.8)
ax.text(.98, .97, 'ring = selected', transform=ax.transAxes, ha='right', va='top', fontsize=6, color=INK2)
# B
ax = fig.add_subplot(G[0, 1]); x = np.arange(1, 6)
ax.plot(x, qb.m2, '-o', color=COL['VitaFormer'], ms=4, lw=1.6, label='VitaFormer s42')
ax.plot(x, qb.reference, '--D', color=REF, ms=3.5, lw=1.2, label='Reference masks')
ax.plot(x, qb.otsu, ':x', color=MUTED, ms=4, lw=1.2, label='Otsu')
ax.set_xticks(x); ax.set_xticklabels([f'Q{i}\n{a:.2f}' for i, a in zip(x, qb.atp)], fontsize=6.5, color=INK2)
ax.set_xlabel('ATP quintile (median ATP)'); ax.set_ylabel('Median area (rel. control)'); clean(ax)
ax.legend(frameon=False, fontsize=6, loc='lower right', labelcolor=INK2); ax.set_title('B  Area by ATP quintile', pad=12); tag(ax, 'post hoc summary', y=1.0)
# C
ax = fig.add_subplot(G[0, 2])
ax.plot(x, qb.dice_m2, '-o', color=COL['VitaFormer'], ms=4, lw=1.6)
for i, (d, fr) in enumerate(zip(qb.dice_m2, qb.frac_dice_lt_08)):
    ax.text(x[i] + (.15 if i == 0 else 0), d + (0 if i == 0 else .015), f'{fr * 100:.0f}%', ha='left' if i == 0 else 'center',
            va='center' if i == 0 else 'bottom', fontsize=6, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([f'Q{i}' for i in x], color=INK2); ax.set_ylim(.6, 1.04)
ax.set_xlabel('ATP quintile'); ax.set_ylabel('Median Dice vs reference'); clean(ax)
ax.set_title('C  Agreement with reference\n    (labels: % wells Dice < 0.8)', fontsize=8, pad=12); tag(ax, 'post hoc summary', y=1.0)
# D examples
csv = pd.read_csv(DATA / 'raw/data_labeled.csv').set_index('location_in_bucket')
ex = [('A_ms40_thorlabs_P0935_D07_2023-07-19_G07.png', 'low ATP (0.01)'), ('A_ms40_thorlabs_P0935_D07_2023-07-19_H12.png', 'high ATP (0.91)')]


def edge(m):
    return m & ~ndi.binary_erosion(m, iterations=9)


GD = G[1, :].subgridspec(1, 2, wspace=.08)
for k, (img, lab) in enumerate(ex):
    ax = fig.add_subplot(GD[0, k])
    g = np.asarray(Image.open(DATA / 'extracted/images_labeled_cropped_0.2' / img).convert('RGB')).copy()
    row = csv.loc[img].copy(); row['location_in_bucket'] = img
    rm = ref_crop(pd.Series(row)); pm = np.isin(np.asarray(Image.open(NV / 'masks/m2_s42_S256' / img)), (1, 2))
    g[edge(rm)] = (237, 161, 0); g[edge(pm)] = (42, 120, 214)
    ax.imshow(g); ax.set_axis_off()
    H, W = g.shape[:2]; ax.plot([W * .05, W * .05 + 200], [H * .94] * 2, color='white', lw=2.2, solid_capstyle='butt')
    ax.text(W * .05 + 100, H * .91, '200 px (crop)', color='white', fontsize=5.8, ha='center', va='bottom')
    a_m2 = p.loc[img, 'm2_s42']; a_ref = p.loc[img, 'reference']
    ax.set_title(f'{"D  " if k == 0 else ""}{lab}: area VitaFormer {a_m2:.2f}, reference {a_ref:.2f}', fontsize=7.5)
    if k == 0:
        ax.text(0, -.03, 'Post hoc inspection. Blue = VitaFormer s42 (256 px); yellow = authors\' reference mask. Wells from a pre-fixed random draw per\n'
                'domain × ATP tertile (Thorlabs shown; all 24 wells in Figure S4). NViR images: Dubinsky et al., Zenodo 15374443, MIT licence.',
                transform=ax.transAxes, va='top', ha='left', fontsize=6, color=INK2)
# E Gemcitabine (v4: three families)
ax = fig.add_subplot(G[2, 0])
FAM = [('VitaFormer', 'm2'), ('SegFormer-B2', 'segformer'), ('nnU-Net', 'nnunet')]
from style import MRK
for k, (lab, key) in enumerate(FAM):
    for j, sd in enumerate(['42', '3407', '2026']):
        for i, th in enumerate([52, 72]):
            r = gem[(gem.model == f'{key}_seed{sd}') & (gem.time_h == th)].iloc[0]; xx = i + (k - 1) * .22 + (j - 1) * .06
            ax.plot([xx, xx], [r.rho_M2_fg_fc_PI_lo, r.rho_M2_fg_fc_PI_hi], color=COL[lab], lw=.8, alpha=.5)
            ax.plot(xx, r.rho_M2_fg_fc_PI, MRK[lab], color=COL[lab], ms=3.8, mec=SURF, label=lab if (i == 0 and j == 0) else None)
for i, th in enumerate([52, 72]):
    r = gem[(gem.model == 'm2_seed42') & (gem.time_h == th)].iloc[0]
    ax.plot([i + .42, i + .42], [r.rho_author_area_fc_PI_lo, r.rho_author_area_fc_PI_hi], color=REF, lw=1, alpha=.55)
    ax.plot(i + .42, r.rho_author_area_fc_PI, 'x', color=REF, ms=5)
ax.set_xticks([0, 1]); ax.set_xticklabels(['52 h', '72 h'], color=INK2); ax.set_xlim(-.5, 1.65); ax.set_ylim(-1.15, -.3)
ax.set_ylabel('ρ (area fold change vs PI)'); clean(ax)
ax.legend(frameon=False, fontsize=5.5, loc='upper left', ncol=1, labelcolor=INK2, handletextpad=.2, borderaxespad=.1)
ax.set_title('E  GemcitabineScreen\n    3 models × 3 seeds; × = OrganoID', fontsize=8, pad=12); tag(ax, 'pre-specified comparator', y=1.0)
ax.text(.02, .03, 'Frames with any predicted Dead pixel (of 798):\nVitaFormer 0 / 27 / 0; SegFormer-B2 and nnU-Net 0', transform=ax.transAxes, fontsize=5.5, color=INK2)
# F N2
ax = fig.add_subplot(G[2, 1:3])
doms = ['A', 'B', 'C', 'D']
for j, (rep, mk) in enumerate([('m2_s42', 'o'), ('m2_s3407', 's'), ('m2_s2026', 'D')]):
    v = n2[n2.rep == rep].assign(d=lambda z: z.domain.map(DOM)).dropna(subset=['d']).set_index('d').loc[doms, 'r2']
    ax.plot(np.arange(4) + (j - 1) * .1, v, mk, color=COL['VitaFormer'], ms=4.5, mec=SURF, label=f'VitaFormer s{rep.split("_s")[1]}')
for (lab, vals), (ls, mk) in zip(PUB.items(), [(':', '_'), ('--', '_'), ('-', '_')]):
    for i, val in enumerate(vals):
        ax.plot([i - .3, i + .3], [val, val], ls, color=REF, lw=1)
    ax.text(3.35, vals[-1], lab, fontsize=6, color=INK2, va='center')
ax.set_xticks(range(4)); ax.set_xticklabels(['A\nEcho Rebel', 'B\nEcho Revolve', 'C\nCytation 5', 'D\nThorlabs'], color=INK2, fontsize=6.8)
ax.set_xlim(-.5, 4.4); ax.set_ylim(0, 1); ax.set_ylabel('Test R²'); clean(ax)
ax.legend(frameon=False, fontsize=6, loc='lower left', ncol=3, labelcolor=INK2)
ax.set_title('F  NViR supervised mapping (ATP-fitted) vs published values (lines)', fontsize=8, pad=12); tag(ax, 'secondary', y=1.0)
save(fig, 'Fig5_draft_v4')
print(qb.round(3)); print(n2.assign(d=n2.domain.map(DOM)).groupby('d').r2.mean().round(3))
