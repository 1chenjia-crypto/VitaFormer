#!/usr/bin/env python3
"""v3 (2026-11-16, review): B states 64 organoids / 1,407 images / organoid = resampling unit; D separates AP (only metric compared with the publication) from Dice.
Figure 6 (draft v2, 2026-11-16; v1 = fig6_draft.py): v2 shortens titles to dataset + metric, labels every reference line
with its value, adds status badges (paired = pre-specified contrast; descriptive = published mean only). Content unchanged.
 external segmentation after supervised fine-tuning (protocol B: VitaFormer seed-42 weights fine-tuned on each
source training split, 2-class head; each dataset vs its own source publication, using that publication's split and metric).
A OrganoIDNet per-image IoU (paired with the authors' Cellpose/StarDist predictions; 10 test images)
B BrainOrganoid2024 per-fold Dice (VitaFormer vs nnU-Net, same protocol); authors' SegFormer-B0 (their OOF predictions) as a line
C OrganoID human (PDAC, non-PDAC) and mouse per-image IoU vs published means (descriptive; [VERIFY] published values)
D OrgaSegment per-image instance AP@0.5 and pixel Dice vs published Mask R-CNN AP (descriptive; [VERIFY])."""
import pandas as pd
import matplotlib.pyplot as plt
from style import EXT, COL, INK2, MUTED, BASE, REF, clean, save, tag, OUT

E = EXT / 'external_m2_20261108'
B = EXT / 'brain_organoid2024_m2_20261108'
on = pd.read_csv(E / 'organoidnet/analysis_B/per_image_metrics.csv')
onc = pd.read_csv(E / 'organoidnet/analysis_B/contrasts.csv'); onc = onc[onc.metric == 'iou']
bf = pd.read_csv(B / 'analysis/fold_summary.csv'); bc = pd.read_csv(B / 'analysis/contrasts.csv')
bs = pd.read_csv(B / 'analysis/summary.csv') if (B / 'analysis/summary.csv').exists() else None
oh = pd.read_csv(E / 'organoidid_human/analysis_B/per_image_metrics.csv'); om = pd.read_csv(E / 'organoidid_mouse/analysis_B/per_image_metrics.csv')
og = pd.read_csv(E / 'orgasegment/analysis_B/per_image_metrics.csv')
PUB = {'PDAC': .74, 'non-PDAC': .79, 'mouse': .82, 'Mask R-CNN AP50': .76, 'SegFormer-B0 (authors)': .960}  # [VERIFY] except B0 (author CSV)
for n, d in [('Fig6A_source_organoidnet.csv', on), ('Fig6B_source_brain_folds.csv', bf), ('Fig6C_source_organoidid_human.csv', oh),
             ('Fig6C_source_organoidid_mouse.csv', om), ('Fig6D_source_orgasegment.csv', og)]:
    d.to_csv(OUT / n, index=False)
pd.concat([onc.assign(dataset='OrganoIDNet'), bc.assign(dataset='BrainOrganoid2024')]).to_csv(OUT / 'Fig6_source_contrasts.csv', index=False)

fig = plt.figure(figsize=(7.4, 5.8)); G = fig.add_gridspec(2, 2, hspace=.62, wspace=.35)
c = COL['VitaFormer']
# A
ax = fig.add_subplot(G[0, 0]); arms = ['StarDist (author)', 'Cellpose (author)', 'M2']
w = on.pivot(index='image', columns='arm', values='iou')[arms]
for _, r in w.iterrows():
    ax.plot(range(3), r.values, '-', color=BASE, lw=.7)
ax.plot([0] * len(w), w[arms[0]], 'x', color=REF, ms=4); ax.plot([1] * len(w), w[arms[1]], 'D', color=REF, ms=3.5)
ax.plot([2] * len(w), w['M2'], 'o', color=c, ms=4.5)
ax.set_xticks(range(3)); ax.set_xticklabels(['StarDist\n(authors)', 'Cellpose\n(authors)', 'VitaFormer'], color=INK2)
r = onc[onc.contrast == 'M2 - Cellpose (author)'].iloc[0]; r2 = onc[onc.contrast == 'M2 - StarDist (author)'].iloc[0]
ax.text(.02, .03, f'Δ vs Cellpose {r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]\nΔ vs StarDist {r2.delta:+.3f} [{r2.ci_lo:+.3f}, {r2.ci_hi:+.3f}]',
        transform=ax.transAxes, fontsize=6.3, color=INK2)
ax.set_ylim(.6, .75); ax.set_ylabel('Per-image IoU'); clean(ax); ax.set_title('A  OrganoIDNet · per-image IoU', pad=12); tag(ax, 'paired, n = 10 images', y=1.0)
# B
ax = fig.add_subplot(G[0, 1]); wf = bf.pivot(index='fold', columns='arm', values='mean')
for f_, r_ in wf.iterrows():
    ax.plot([0, 1], [r_['nnU-Net'], r_['M2']], '-', color=BASE, lw=.8)
ax.plot([0] * 5, wf['nnU-Net'], '^', color=COL['nnU-Net'], ms=5); ax.plot([1] * 5, wf['M2'], 'o', color=c, ms=5)
ax.axhline(PUB['SegFormer-B0 (authors)'], color=REF, ls='--', lw=.9); ax.text(1.88, PUB['SegFormer-B0 (authors)'] + .0008, 'authors\' SegFormer-B0 (0.960)', fontsize=6, color=INK2, va='bottom', ha='right')
b1 = bc[bc.contrast == 'M2 - nnU-Net'].iloc[0]; b2 = bc[bc.contrast == 'M2 - Author SegFormer-B0 OOF'].iloc[0]
ax.text(.02, .03, f'Δ vs nnU-Net {b1.delta:+.4f} [{b1.ci_lo:+.4f}, {b1.ci_hi:+.4f}]\nΔ vs authors\' B0 {b2.delta:+.4f} [{b2.ci_lo:+.4f}, {b2.ci_hi:+.4f}]',
        transform=ax.transAxes, fontsize=6.3, color=INK2)
ax.set_xticks([0, 1]); ax.set_xticklabels(['nnU-Net\n(same protocol)', 'VitaFormer'], color=INK2); ax.set_xlim(-.4, 1.9)
ax.set_ylim(.945, .98); ax.set_ylabel('Mean Dice per fold'); clean(ax); ax.set_title('B  BrainOrganoid2024 · Dice', pad=12); tag(ax, '5 folds · 64 organoids · 1,407 images', y=1.0)
# C
ax = fig.add_subplot(G[1, 0])
groups = [('PDAC', oh[oh.group == 'PDAC'].iou), ('non-PDAC', oh[oh.group != 'PDAC'].iou), ('mouse', om.iou)]
for i, (lab, v) in enumerate(groups):
    ax.plot([i + (k - len(v) / 2) * .03 for k in range(len(v))], v, 'o', color=c, ms=3.5, alpha=.85)
    ax.plot([i - .3, i + .3], [v.mean()] * 2, color=c, lw=1.6)
    ax.plot([i - .3, i + .3], [PUB[lab]] * 2, '--', color=REF, lw=1)
    ax.text(i + .32, PUB[lab], f'{PUB[lab]:.2f}', fontsize=5.6, color=INK2, va='center')
    ax.text(i, 1.0, f'n = {len(v)}', ha='center', fontsize=6, color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels(['Human PDAC', 'Human non-PDAC', 'Mouse'], color=INK2); ax.set_ylim(.4, 1.04)
ax.set_ylabel('Per-image IoU'); clean(ax); ax.set_title('C  OrganoID · IoU', pad=12); tag(ax, 'descriptive; published values [VERIFY]', y=1.0)
ax.plot([], [], '-', color=c, lw=1.6, label='VitaFormer mean'); ax.plot([], [], '--', color=REF, lw=1, label='published mean')
ax.legend(frameon=False, fontsize=5.8, loc='lower right', labelcolor=INK2)
# D
ax = fig.add_subplot(G[1, 1])
ax.plot([0 + (k - 6) * .03 for k in range(len(og))], og.ap50, 'o', color=c, ms=3.5); ax.plot([-.3, .3], [og.ap50.mean()] * 2, color=c, lw=1.6)
ax.plot([-.3, .3], [PUB['Mask R-CNN AP50']] * 2, '--', color=REF, lw=1)
ax.text(.32, PUB['Mask R-CNN AP50'], 'published\nMask R-CNN 0.76', fontsize=5.6, color=INK2, va='center')
ax.plot([1 + (k - 6) * .03 for k in range(len(og))], og.dice, 'o', color=c, ms=3.5); ax.plot([.7, 1.3], [og.dice.mean()] * 2, color=c, lw=1.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Instance AP@0.5\n(source metric)', 'Pixel Dice'], color=INK2); ax.set_ylim(.3, 1.02)
ax.text(.02, .03, f'{og.merged_rate.mean() * 100:.0f}% of annotated organoids merged with another\n(semantic mask + connected components; no instance head)',
        transform=ax.transAxes, fontsize=6.2, color=INK2)
ax.axvline(.5, color=INK2, lw=.8)
ax.text(1.0, .62, 'Dice: no published\ncomparator', fontsize=5.6, color=MUTED, ha='center', va='center')
ax.set_ylabel('AP@0.5 (left) · Dice (right)'); clean(ax); ax.set_title('D  OrgaSegment · AP@0.5 | Dice', pad=12); tag(ax, 'descriptive; n = 12; [VERIFY]', y=1.0)
fig.text(.01, -.02, 'Supervised adaptation (fine-tuned from VitaFormer seed-42 weights on each source training split; nnU-Net in B from its CRC weights), not frozen transfer. '
         'Paired CIs: A image bootstrap (10 images); B cluster bootstrap over 64 organoids (organoid = resampling unit; 1,407 out-of-fold images).', fontsize=6, color=MUTED)
save(fig, 'Fig6_draft_v3')
print(w.mean().round(3).to_dict(), wf.mean().round(4).to_dict(), [round(v.mean(), 3) for _, v in groups], og.ap50.mean().round(3), og.dice.mean().round(3))
