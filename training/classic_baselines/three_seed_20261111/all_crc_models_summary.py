#!/usr/bin/env python3
"""All CRC models x 3 seeds (shared evaluator): table + bar chart (descriptive, no contrasts).
Models: VitaFormer, nnU-Net, SegFormer-B2, OrganoidViT, DeepLabv3, U-Net++, U-Net.
Endpoints: mIoU, per-class IoU, Dead recall/precision, median per-image |relative area error| (Alive/Dead/Bubble;
images whose GT contains the class; pred = tp+fp, GT = tp+fn) — the OrganoidViT source paper's Fig. 5 endpoint.
Env DRYRUN=1: missing seeds are substituted by an existing seed (layout test only; output names prefixed _dryrun_)."""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '3D_OrganoidSeg/paper_materials_20261110/figure_scripts')
from style import COL, INK, INK2, MUTED, BASE, SURF, clean, save, OUT  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DRY = os.environ.get('DRYRUN') == '1'
SEEDS = [42, 3407, 2026]
R3 = Path('3D_OrganoidSeg')
CB = Path('OrganoidSegBaseline/classic_baselines_3seed_20261111')
OV = Path('OrganoidSegBaseline/orgvit_3seed_20261111')
SRC = {
    'VitaFormer': lambda s: R3 / f'vco/runs/arch_exploration_20261104/stageF_m2_test/shared_eval/M2_final_seed{s}.json',
    'nnU-Net': lambda s: R3 / f'baselines_fair_20261030/shared_eval/nnunet_run2_seed{s}_noTTA.json',
    'SegFormer-B2': lambda s: R3 / f'baselines_fair_20261030/shared_eval/segformer_b2_seed{s}.json',
    'OrganoidViT': lambda s: OV / f'shared_eval/organoidvit_seed{s}.json',
    'DeepLabv3': lambda s: CB / f'shared_eval/deeplabv3_seed{s}.json',
    'U-Net++': lambda s: CB / f'shared_eval/unetpp_seed{s}.json',
    'U-Net': lambda s: CB / f'shared_eval/unet_seed{s}.json',
}
DRY_FALLBACK = {'OrganoidViT': OV / 'env_check_a800/shared_eval_seed0.json'}
CLS = {'Alive': 1, 'Dead': 2, 'Bubble': 3}

rows, used = [], {}
for m, f in SRC.items():
    for s in SEEDS:
        p = f(s)
        if not p.exists():
            if not DRY:
                sys.exit(f'missing {p}')
            p = DRY_FALLBACK.get(m) or next(f(x) for x in SEEDS if f(x).exists())
        used[f'{m} s{s}'] = str(p)
        j = json.load(open(p))
        tp, fp, fn = (np.array([x['confusion_counts'][k] for x in j['per_image']]) for k in ('tp', 'fp', 'fn'))
        pred, gt = tp + fp, tp + fn
        r = {'model': m, 'seed': s, 'mIoU': j['mIoU'], 'Alive IoU': j['iou_alive'], 'Dead IoU': j['iou_dead'],
             'Bubble IoU': j['iou_bubble'], 'Dead recall': j['recall_dead'], 'Dead precision': j['precision_dead']}
        for c, k in CLS.items():
            el = gt[:, k] > 0
            r[f'{c} |area err| median %'] = float(np.median(np.abs(pred[el, k] - gt[el, k]) / gt[el, k]) * 100)
        rows.append(r)
d = pd.DataFrame(rows)
pre = '_dryrun_' if DRY else ''
d.to_csv(CB / f'{pre}all_crc_models_per_seed.csv', index=False)
eps = [c for c in d.columns if c not in ('model', 'seed')]
g = d.groupby('model', sort=False)[eps].agg(['mean', 'std'])
order = list(SRC)
with open(CB / f'{pre}all_crc_models_table.md', 'w') as fh:
    fh.write('CRC test set (167 images), shared evaluator, mean ± SD over seeds 42/3407/2026. Area error = median per-image '
             '|pred − GT| / GT over images containing the class (Alive 167, Dead 24, Bubble 22). Descriptive; no contrasts.\n\n')
    fh.write('| Model | ' + ' | '.join(eps) + ' |\n|' + '---|' * (len(eps) + 1) + '\n')
    for m in order:
        fh.write(f'| {m} | ' + ' | '.join(f'{g.loc[m, (e, "mean")]:.2f} ± {g.loc[m, (e, "std")]:.2f}' for e in eps) + ' |\n')
print(open(CB / f'{pre}all_crc_models_table.md').read())

# bar chart: bars = mean, dots = seeds, VitaFormer highlighted, others neutral; models labelled on x
panels = [('mIoU', 'Test mIoU (%)'), ('Alive IoU', 'IoU (%)'), ('Dead IoU', 'IoU (%)'), ('Bubble IoU', 'IoU (%)'),
          ('Alive |area err| median %', '|area error| (%)'), ('Dead |area err| median %', '|area error| (%)'),
          ('Bubble |area err| median %', '|area error| (%)')]
titles = ['A  mIoU', 'B  Alive IoU', 'C  Dead IoU', 'D  Bubble IoU', 'E  Alive area error', 'F  Dead area error (n=24)',
          'G  Bubble area error (n=22)']
XL = ['Vita-\nFormer', 'nnU-\nNet', 'SegFormer\n-B2', 'Organoid-\nViT', 'Deep-\nLabv3', 'U-Net\n++', 'U-Net']
XS = ['VitaFormer', 'nnU-Net', 'SegFormer-B2', 'OrganoidViT', 'DeepLabv3', 'U-Net++', 'U-Net']
CAP = {'Bubble |area err| median %': 40}
fig = plt.figure(figsize=(7.4, 8.6)); gs = fig.add_gridspec(3, 3, hspace=.95, wspace=.42)
pos = [(0, slice(0, 3))] + [(1, i) for i in range(3)] + [(2, i) for i in range(3)]
for (row, col), (ep, yl), t in zip(pos, panels, titles):
    ax = fig.add_subplot(gs[row, col])
    for i, m in enumerate(order):
        v = d[d.model == m][ep].to_numpy()
        c = COL['VitaFormer'] if m == 'VitaFormer' else BASE
        cap = CAP.get(ep)
        top = min(v.mean(), cap) if cap else v.mean()
        ax.bar(i, top, width=.62, color=c, alpha=.85 if m == 'VitaFormer' else 1, zorder=2)
        vv = np.minimum(v, cap) if cap else v
        ax.scatter(i + np.array([-.15, 0, .15]), vv, s=9, color=INK if m != 'VitaFormer' else '#123d70', zorder=3, linewidths=0)
        if cap and v.mean() > cap:
            ax.text(i, cap * .93, f'{v.mean():.0f}\n(off\nscale)', ha='center', va='top', fontsize=5.5, color=INK)
        if row == 0:
            ax.text(i, max(v) + 2.5, f'{v.mean():.1f}', ha='center', fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(order)))
    if row == 0:
        ax.set_xticklabels(XL, fontsize=6.8, color=INK2)
    else:
        ax.set_xticklabels(XS, fontsize=6, color=INK2, rotation=55, ha='right', rotation_mode='anchor')
    if CAP.get(ep):
        ax.set_ylim(0, CAP[ep])
    ax.set_ylabel(yl); ax.set_title(t); clean(ax)
    if row == 0:
        ax.set_ylim(0, 100)
fig.text(.01, .02, 'CRC test set, 167 images; shared evaluator; bars = mean of 3 training seeds (42/3407/2026), dots = individual seeds. '
         'U-Net, U-Net++, DeepLabv3 = baselines of the OrganoidViT source paper,\ntrained from scratch with the original scripts. '
         'Area error = median per-image |pred − GT|/GT on images containing the class. Descriptive; no statistical contrasts.'
         + ('  [DRY RUN: placeholder seeds]' if DRY else ''), fontsize=5.8, color=MUTED, va='top')
save(fig, f'{pre}FigS_crc_all_models')
(CB / f'{pre}all_crc_models_inputs.json').write_text(json.dumps(used, indent=1))
print('saved')
