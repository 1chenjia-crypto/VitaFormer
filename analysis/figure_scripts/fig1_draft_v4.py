#!/usr/bin/env python3
# v4 (2026-11-18): GemcitabineScreen column filled for SegFormer-B2 and nnU-Net (Amendment 2); otherwise identical to v3.
"""Figure 1 (draft v3, 2026-11-16, review): C separates image-only readouts from annotation-based validation; nnU-Net Brain = FT; smaller text.
Figure 1 (draft, 2026-11-16): evaluation design.
A models x seeds | B coverage matrix (model x dataset; empty = not tested) | C readout tiers | D user tool screenshot (NViR example, MIT).
Coverage cells are hand-entered from the stage RESULTS files listed in COVER_SRC; check before finalising."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from style import R, COL, MRK, INK, INK2, MUTED, BASE, SURF, REF, GRID, save

SHOT = R / 'vitaformer_tool/docs/screenshots/2_inspect_overlay_en.png'
fig = plt.figure(figsize=(7.6, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1.6], hspace=.42, wspace=.25)


def box(ax, x, y, w, h, text, fc, ec, fs=6.5, color=INK, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.02', fc=fc, ec=ec, lw=.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color, weight=weight)


# A models and seeds
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('A  Segmenters and training seeds')
box(ax, .02, .80, .96, .14, 'CRC organoids: 481 / 164 / 167 images (train/val/test)\nclasses: background, Alive, Dead, Bubble', '#f1f0ea', BASE, fs=5.8)
for i, (m, d) in enumerate([('VitaFormer', 'ViT-B/16 + CNN branch\n+ rare-class recipe'),
                            ('SegFormer-B2', 'hierarchical transformer'),
                            ('nnU-Net', 'self-configuring 2D U-Net')]):
    y = .58 - i * .17
    ax.plot(.07, y + .06, MRK[m], color=COL[m], ms=7)
    ax.text(.13, y + .085, m, fontsize=7, color=INK, weight='bold', va='center')
    ax.text(.13, y + .03, d, fontsize=5.2, color=INK2, va='center')
    for k, s in enumerate(['42', '3407', '2026']):
        box(ax, .66 + k * .112, y + .015, .10, .09, s, SURF, COL[m], fs=5.2)
ax.text(.02, .02, 'Source-publication comparison (CRC only, 3 seeds):\nOrganoidViT, U-Net, U-Net++, DeepLabv3', fontsize=5.8, color=MUTED)
ax.annotate('', xy=(.5, .80), xytext=(.5, .66), arrowprops=dict(arrowstyle='<-', color=BASE, lw=.8))

# B coverage matrix
ax = fig.add_subplot(gs[0, 1]); ax.set_title('B  Coverage (empty = not tested)')
cols = ['CRC\ntest', 'Deep-\nOrganoid\nCTG', 'NViR\nATP', 'Gemcit.\nPI', 'Organo-\nIDNet', 'Brain-\nOrganoid\n2024', 'OrganoID\nhuman/\nmouse', 'Orga-\nSegment']
rows = ['VitaFormer', 'SegFormer\n-B2', 'nnU-Net', 'Source-pub.\nmodels']
cell = {
    (0, 0): '3 seeds', (0, 1): '3 seeds', (0, 2): '3 seeds', (0, 3): '3 seeds', (0, 4): 'FT', (0, 5): 'FT\n5-fold', (0, 6): 'FT', (0, 7): 'FT',
    (1, 0): '3 seeds', (1, 1): '3 seeds', (1, 2): '3 seeds', (1, 3): '3 seeds',
    (2, 0): '3 seeds', (2, 1): '3 seeds', (2, 2): '3 seeds', (2, 3): '3 seeds', (2, 5): 'FT\n5-fold',
    (3, 0): '3 seeds',
}
group_fc = {0: '#eef3fb', 1: '#fbf3ee', 2: '#fbf3ee', 3: '#fbf3ee', 4: '#eef7f2', 5: '#eef7f2', 6: '#eef7f2', 7: '#eef7f2'}
for c in range(len(cols)):
    for r in range(len(rows)):
        t = cell.get((r, c))
        ax.add_patch(plt.Rectangle((c, r), 1, 1, fc=group_fc[c] if t else SURF, ec=GRID, lw=.8))
        if t:
            ax.text(c + .5, r + .5, t, ha='center', va='center', fontsize=5.6, color=INK)
ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), -1.05)
ax.set_xticks([c + .5 for c in range(len(cols))]); ax.set_xticklabels(cols, fontsize=5.4, color=INK2)
ax.xaxis.tick_bottom()
ax.set_yticks([r + .5 for r in range(len(rows))]); ax.set_yticklabels(rows, fontsize=6.3, color=INK2)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
for x0, x1, lab in [(0, 1, 'in-domain'), (1, 4, 'functional, frozen weights'), (4, 8, 'supervised fine-tuning')]:
    ax.plot([x0 + .08, x1 - .08], [-.25, -.25], color=INK2, lw=.8)
    ax.text((x0 + x1) / 2, -.45, lab, ha='center', va='bottom', fontsize=6, color=INK2)
ax.text(0, len(rows) + 1.35, 'FT = fine-tuned (VitaFormer from seed-42 weights; nnU-Net from its CRC weights) with the source split and metric. '
        'NViR/Gemcitabine: input scale chosen on annotations, not assay.', fontsize=5.2, color=MUTED, transform=ax.transData)

# C readout tiers
ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('C  Readout tiers')
tiers = [('Image-only', 'masks, areas, region counts, shape;\nno assay or control wells needed', 'raw area, region count, morphology'),
         ('Control-relative', 'ratio/difference to median of\nmetadata-defined control wells', 'DeepOrganoid E3 ranking;\nNViR N1 area–ATP; Gemcitabine PI'),
         ('Calibrated', 'features mapped to assay value\nusing assay labels for fitting', 'DeepOrganoid E1 (LOCO Ridge);\nNViR N2 (gradient boosting)')]
shade = ['#f6f6f2', '#ecebe4', '#dedcd2']
for i, (t, d, e) in enumerate(tiers):
    y = .70 - i * .32
    box(ax, .02, y, .96, .27, '', shade[i], BASE)
    ax.text(.05, y + .215, t, fontsize=7, weight='bold', color=INK, va='center')
    ax.text(.05, y + .14, d, fontsize=5.0, color=INK2, va='center', linespacing=1.05)
    ax.text(.05, y + .06, 'Here: ' + e, fontsize=4.9, color=INK, va='center', style='italic', linespacing=1.1)
    if i < 2:
        ax.annotate('', xy=(.5, y - .045), xytext=(.5, y + .005), arrowprops=dict(arrowstyle='->', color=MUTED, lw=.8))
ax.text(.02, -.04, 'Down the tiers: more information used, more assumptions to validate.', fontsize=5.4, color=MUTED)
ax.text(.02, -.13, 'Separate from the tiers: annotation-based validation (needs reference masks):\nIoU, Dice, area error vs annotation (CRC test; fine-tuning datasets).', fontsize=5.2, color=INK2, style='italic')

# D tool screenshot
ax = fig.add_subplot(gs[1, 1]); ax.axis('off'); ax.set_title('D  User tool (local browser interface)')
try:
    ax.imshow(plt.imread(SHOT))
except Exception as e:
    ax.text(.5, .5, f'screenshot not found:\n{SHOT}', ha='center', va='center', fontsize=6, color=MUTED, transform=ax.transAxes)
ax.text(0, -.04, 'Example image: NViR (Dubinsky et al. 2026), MIT licence.', fontsize=5.6, color=MUTED, transform=ax.transAxes)
save(fig, 'Fig1_draft_v4')
