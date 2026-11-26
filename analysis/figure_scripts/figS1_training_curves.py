#!/usr/bin/env python3
"""Figure S1 (draft, 2026-11-16): training curves of the source-publication models (OrganoidViT, U-Net, U-Net++, DeepLabv3) and
SegFormer-B2, 3 seeds each, read directly from the training logs (no new runs).
Top row: training loss (log scale, 500-iteration running median; x = first non-finite loss).
Bottom row: validation mIoU from each training script's own evaluator (not the shared evaluator used for test results);
ring = checkpoint used for test (best validation mIoU)."""
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import R, COL, INK2, MUTED, BASE, REF, SEEDS, clean, save, OUT

RUNS = R.parent / 'OrganoidSegBaseline/runs'
SEG = R / 'baselines_fair_20261030/segformer_b2_runs'
SEED_COL = {'42': '#2a78d6', '3407': '#eb6834', '2026': '#1baf7a'}  # seeds coloured; models are panels
TR = re.compile(r'iter[= ]0*(\d+)\S*\s.*?loss[= ](\S+)')


def classic(model, seed):
    d = RUNS / (f'{model}_cond_40k_torch' if seed == '2026' else f'{model}_cond_40k_torch_seed{seed}')
    log = d / ('train.full.log' if seed != '2026' else f'nohup_{model}_cond_40k_torch.log')
    tr = [(int(m.group(1)), float(m.group(2))) for m in map(TR.match, open(log, errors='ignore')) if m]
    v = pd.read_csv(d / 'val_metrics.csv')
    return tr, list(zip(v['iter'], v['mIoU'] * 100))


def orgvit(seed):
    d = RUNS / f'orgvit_b0_1446_40k_cond_seed{seed}_a800'
    tr, va = [], []
    for f in sorted(d.glob('*.log.json')):
        for line in open(f):
            if not line.startswith('{"mode"'):
                continue
            j = json.loads(line)
            if j['mode'] == 'train':
                tr.append((j['iter'], j['loss']))
            elif j['mode'] == 'val':
                va.append((j['iter'], j['mIoU'] * 100))
    return tr, va


def segformer(seed):
    tr, va = [], []
    for line in open(SEG / f'logs/seed{seed}.full.log', errors='ignore'):
        m = re.match(r'iter (\d+) loss (\S+)', line)
        if m:
            tr.append((int(m.group(1)), float(m.group(2))))
        m = re.match(r'VAL iter (\d+) mIoU (\S+)', line)
        if m:
            va.append((int(m.group(1)), float(m.group(2))))
    return tr, va


MODELS = [('SegFormer-B2', segformer), ('OrganoidViT', orgvit), ('DeepLabv3', lambda s: classic('deeplabv3', s)),
          ('U-Net++', lambda s: classic('unetpp', s)), ('U-Net', lambda s: classic('unet', s))]
rows = []
fig, axs = plt.subplots(2, 5, figsize=(9.2, 4.0), sharex=True, gridspec_kw=dict(hspace=.25, wspace=.35))
for k, (name, fn) in enumerate(MODELS):
    for s in SEEDS:
        tr, va = fn(s)
        tr = pd.DataFrame(tr, columns=['iter', 'loss']).drop_duplicates('iter', keep='last').sort_values('iter')
        va = pd.DataFrame(va, columns=['iter', 'miou']).drop_duplicates('iter', keep='last').sort_values('iter')
        fin = tr[np.isfinite(tr.loss)]
        nan0 = tr[~np.isfinite(tr.loss)].iter.min()
        sm = fin.set_index('iter').loss.rolling(10, min_periods=1).median()
        ax = axs[0, k]; ax.plot(sm.index, sm.values, color=SEED_COL[s], lw=.8, alpha=.9)
        if not (isinstance(nan0, float) and math.isnan(nan0)) and pd.notna(nan0):
            yv = sm[sm.index <= nan0].iloc[-1] if (sm.index <= nan0).any() else sm.iloc[0]
            ax.plot(nan0, yv, 'X', color=SEED_COL[s], ms=8, mec='white', mew=.8, zorder=5)
            axs[1, k].axvline(nan0, color=SEED_COL[s], lw=.7, ls=':', zorder=0)
        ax = axs[1, k]; ax.plot(va.iter, va.miou, '-', color=SEED_COL[s], lw=.9)
        b = va.loc[va.miou.idxmax()] if len(va) else None
        if b is not None:
            ax.plot(b.iter, b.miou, 'o', ms=6, mfc='none', mec=SEED_COL[s], mew=1.1)
        rows.append({'model': name, 'seed': s, 'n_train_log_points': len(tr), 'first_nonfinite_loss_iter': nan0,
                     'n_val_points': len(va), 'best_val_iter': None if b is None else int(b.iter),
                     'best_val_mIoU_script_evaluator': None if b is None else round(float(b.miou), 2)})
    axs[0, k].set_title(name); axs[0, k].set_yscale('log'); clean(axs[0, k])
    clean(axs[1, k]); axs[1, k].set_ylim(0, 100)
    axs[1, k].set_xlabel('iteration (k)')
    axs[1, k].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1000:.0f}'))
axs[0, 0].set_ylabel('training loss (log)'); axs[1, 0].set_ylabel('validation mIoU (%)')
h = [plt.Line2D([], [], color=c, lw=1.2, label=f'seed {s}') for s, c in SEED_COL.items()]
h += [plt.Line2D([], [], ls='', marker='X', color=INK2, ms=7, mec='white', label='first non-finite loss (dotted line below)'),
      plt.Line2D([], [], ls='', marker='o', mfc='none', mec=INK2, label='checkpoint used for test')]
fig.legend(handles=h, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(.5, 1.04), fontsize=7, labelcolor=INK2)
fig.text(.01, -.06, 'Read from the training logs; no new runs. Loss: running median over 10 logged points; each script logs its own loss '
         '(not comparable across models). Validation mIoU from each training script\'s own evaluator\n(not the shared evaluator used for test '
         'results in Table S3). Loss is not drawn after it becomes non-finite; validation mIoU then collapses to ~25% (background only). The checkpoint used for test is the best validation mIoU.',
         fontsize=5.8, color=MUTED)
pd.DataFrame(rows).to_csv(OUT / 'FigS1_source_training_summary.csv', index=False)
save(fig, 'FigS1_training_curves')
print(pd.DataFrame(rows).to_string())
