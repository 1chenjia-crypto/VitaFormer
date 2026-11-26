#!/usr/bin/env python3
"""Figure 2 (draft v4, 2026-11-16, review): B numbers in a separate right-hand column; D badge "test set, descriptive"; A labels less crowded.
Figure 2 (draft v3, 2026-11-16): in-domain accuracy on the CRC test set (167 images, 154 wells), per training seed.
Changes vs v2: M0 removed from the main figure (paper shows only the frozen VitaFormer; M0 = ablation, Table S4);
source-publication baselines U-Net / U-Net++ / DeepLabv3 (3 seeds) added to A; D adds VitaFormer w/o Rare-Dead (test, descriptive).
A mIoU per seed | B paired differences VitaFormer - X | C per-class IoU | D Dead recall vs precision."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from style import tag, R, EXT, COL, MRK, INK2, MUTED, BASE, SURF, REF, SEEDS, clean, dot, save, OUT

F = R / 'vco/runs/arch_exploration_20261104/stageF_m2_test/comparison'
CB = R.parent / 'OrganoidSegBaseline/classic_baselines_3seed_20261111/all_crc_models_per_seed.csv'
AB = R / 'vco/runs/m2_ablations_eval_20261113/shared_eval_test'
ul = pd.read_csv(EXT / 'unified_comparison_20261102/seed_level.csv').set_index(['group', 'endpoint'])
ct = pd.read_csv(F / 'contrasts.csv')
per = pd.read_csv(CB); per['seed'] = per.seed.astype(str)

EPS = ['mIoU', 'Alive IoU', 'Dead IoU', 'Bubble IoU', 'Dead recall', 'Dead precision']
rows = []
for _, r in per.iterrows():
    for ep in EPS:
        rows.append({'model': r.model, 'endpoint': ep, 'seed': r.seed, 'value': r[ep], 'source': 'all_crc_models_per_seed.csv'})
for ep in ['Dead recall', 'Dead precision', 'mIoU']:
    for s in SEEDS:
        rows.append({'model': 'SegFormer-B2 + Rare-Dead', 'endpoint': ep, 'seed': s, 'value': ul.loc[('P1-S', ep), f'seed{s}'],
                     'source': 'unified_comparison seed_level (P1-S)'})
K = {'mIoU': 'mIoU', 'Dead recall': 'recall_dead', 'Dead precision': 'precision_dead'}
for s in SEEDS:
    j = json.load(open(AB / f'noRD_final_seed{s}.json'))
    for ep, k in K.items():
        rows.append({'model': 'VitaFormer w/o Rare-Dead', 'endpoint': ep, 'seed': s, 'value': j[k], 'source': 'm2_ablations_eval test (descriptive)'})
src = pd.DataFrame(rows)
src.to_csv(OUT / 'Fig2_draft_v4_source_data.csv', index=False)
cmap = {'M2-final - nnU-Net': 'nnU-Net', 'M2-final - SegFormer-B2': 'SegFormer-B2'}
cc = ct[(ct.endpoint == 'mIoU') & ct.contrast.isin(cmap)].copy(); cc['vs'] = cc.contrast.map(cmap)
v = lambda m, ep: src[(src.model == m) & (src.endpoint == ep)].sort_values('seed').value.to_numpy()

MAIN = ['VitaFormer', 'SegFormer-B2', 'nnU-Net']
SRC = {'OrganoidViT': 'v', 'DeepLabv3': 'P', 'U-Net++': 'X', 'U-Net': 'h'}
fig = plt.figure(figsize=(7.6, 5.6)); gs = fig.add_gridspec(2, 2, hspace=.6, wspace=.35)

# A
ax = fig.add_subplot(gs[0, 0]); order = MAIN + list(SRC)
for i, m in enumerate(order):
    for j, y in enumerate(v(m, 'mIoU')):
        if m in SRC:
            dot(ax, i + (j - 1) * .15, y, None, color=REF, marker=SRC[m], ms=6)
        else:
            dot(ax, i + (j - 1) * .15, y, m)
ax.axvspan(2.5, 6.5, color=BASE, alpha=.18, lw=0)
ax.text(4.5, 88.2, 'source-publication comparison', ha='center', fontsize=6.3, color=INK2)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(['Vita-\nFormer', 'Seg-\nFormer\n-B2', 'nnU-\nNet', 'Organ-\noid-\nViT', 'Deep-\nLab\nv3', 'U-Net\n++', 'U-Net'], color=INK2, fontsize=5.8)
ax.set_ylabel('Test mIoU (%)'); ax.set_ylim(40, 90); ax.set_xlim(-.5, 6.5); clean(ax)
ax.set_title('A  Mean IoU per training seed')
# B
ax = fig.add_subplot(gs[0, 1])
for i, vs in enumerate(['nnU-Net', 'SegFormer-B2']):
    r = cc[cc.vs == vs].iloc[0]
    ax.plot([r.ci95_low, r.ci95_high], [i, i], color=INK2, lw=1.2); ax.plot(r.delta_mean, i, 'o', color=INK2, ms=5)
    ax.text(1.03, i, f'{r.delta_mean:+.2f}\n[{r.ci95_low:+.2f}, {r.ci95_high:+.2f}]', va='center', ha='left', fontsize=6.3, color=INK2,
            transform=ax.get_yaxis_transform(), linespacing=1.1)
ax.axvline(0, color=BASE, lw=.9); ax.set_yticks(range(2)); ax.set_yticklabels(['vs\nnnU-Net', 'vs\nSegFormer-B2'], color=INK2, fontsize=6.8)
ax.set_xlim(-3.3, 1.8); ax.set_ylim(1.6, -.6); ax.set_xlabel('VitaFormer − comparator, mIoU points'); clean(ax, 'x')
ax.set_title('B  Paired differences (95% CI)')
# C
ax = fig.add_subplot(gs[1, 0])
for k, ep in enumerate(['Alive IoU', 'Dead IoU', 'Bubble IoU']):
    for i, m in enumerate(MAIN + ['OrganoidViT']):
        for j, y in enumerate(v(m, ep)):
            x = k + (i - 1.5) * .2 + (j - 1) * .045
            if m == 'OrganoidViT':
                dot(ax, x, y, None, color=REF, marker='v', ms=4)
            else:
                dot(ax, x, y, m, ms=4)
ax.set_xticks(range(3)); ax.set_xticklabels(['Alive', 'Dead', 'Bubble'], color=INK2); ax.set_ylabel('IoU (%)'); ax.set_ylim(30, 101)
clean(ax); ax.set_title('C  Per-class IoU per seed')
# D
ax = fig.add_subplot(gs[1, 1])
for m in MAIN:
    for rc, pr in zip(v(m, 'Dead recall'), v(m, 'Dead precision')):
        dot(ax, rc, pr, m)
for var, base in [('SegFormer-B2 + Rare-Dead', 'SegFormer-B2'), ('VitaFormer w/o Rare-Dead', 'VitaFormer')]:
    for rc, pr in zip(v(var, 'Dead recall'), v(var, 'Dead precision')):
        dot(ax, rc, pr, base, filled=False)
    a0 = (v(base, 'Dead recall').mean(), v(base, 'Dead precision').mean())
    a1 = (v(var, 'Dead recall').mean(), v(var, 'Dead precision').mean())
    # arrow always points from "without recipe" to "with recipe"
    frm, to = (a0, a1) if var.startswith('SegFormer') else (a1, a0)
    ax.annotate('', xy=to, xytext=frm, arrowprops=dict(arrowstyle='->', color=COL[base], lw=1))
ax.legend(handles=[plt.Line2D([], [], ls='', marker='s', mfc=SURF, mec=COL['SegFormer-B2'], mew=1.4, label='SegFormer-B2 + recipe'),
                   plt.Line2D([], [], ls='', marker='o', mfc=SURF, mec=COL['VitaFormer'], mew=1.4, label='VitaFormer w/o recipe (test)')],
          loc='lower left', fontsize=5.8, frameon=False, labelcolor=INK2, handletextpad=.2)
ax.set_xlabel('Dead recall (%)'); ax.set_ylabel('Dead precision (%)'); ax.set_xlim(50, 93); ax.set_ylim(42, 76); clean(ax, 'both')
ax.set_title('D  Dead class: rare-class recipe', pad=12); tag(ax, 'test set, descriptive', y=1.0)

h = [plt.Line2D([], [], ls='', marker=MRK[m], color=COL[m], label=m) for m in MAIN]
h += [plt.Line2D([], [], ls='', marker=mk, color=REF, label=m) for m, mk in SRC.items()]
fig.legend(handles=h, loc='upper center', ncol=7, frameon=False, bbox_to_anchor=(.5, 1.0), labelcolor=INK2, fontsize=6.5)
fig.text(.01, -.01, 'CRC test set: 167 images from 154 wells; not blind (used before freezing). mIoU = mean over 4 classes of pooled-pixel IoU. '
         'B: hierarchical paired bootstrap over seeds x wells, 10,000 replicates.\nSource-publication models: descriptive, no contrasts. '
         'D (test set, descriptive): arrows point from without to with the rare-class recipe; the ablation inference (Dead recall +26.3 points) is on the validation set (Table S4)\nand is a different quantity from the arrow length shown here.', fontsize=5.8, color=MUTED)
save(fig, 'Fig2_draft_v4')
print(src.groupby(['model', 'endpoint']).value.mean().round(2).unstack())
print(cc[['vs', 'delta_mean', 'ci95_low', 'ci95_high']].round(2))
