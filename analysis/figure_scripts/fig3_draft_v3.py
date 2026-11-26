#!/usr/bin/env python3
"""v3 (2026-11-16, review): rank labels removed; seed order fixed left->right 42/3407/2026; legend text revised.
Figure 3 (draft): in-domain accuracy vs functional readouts, per training seed, three segmenters.
A CRC mIoU | B DeepOrganoid E1 (calibrated) and E3 (uncalibrated ranking) | C nnU-Net input configurations (V0 primary; V1-V3 post hoc)
| D NViR area-ATP Spearman (pooled, physical-plate clusters). Reads the stage result files directly; writes PNG + PDF + source CSV.
Palette: validated categorical slots 1-3 (dataviz reference); identity also encoded by marker shape and direct labels."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

R = Path('3D_OrganoidSeg')
A = R / 'external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment'
OUT = R / "paper_materials_20261110/figures_draft"
COL = {'VitaFormer': '#2a78d6', 'SegFormer-B2': '#eb6834', 'nnU-Net': '#1baf7a'}
MRK = {'VitaFormer': 'o', 'SegFormer-B2': 's', 'nnU-Net': '^'}
INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SEEDS = ['42', '3407', '2026']
MODELS = list(COL)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURF, 'figure.facecolor': SURF})
rows = []

# A: CRC per seed
crc_a = pd.read_csv(R / 'vco/runs/arch_exploration_20261104/stageF_m2_test/comparison/seed_level.csv')
crc_a = crc_a[crc_a.endpoint == 'mIoU'].set_index('group')
crc = {'VitaFormer': crc_a.loc['M2-final'], 'SegFormer-B2': crc_a.loc['SegFormer-B2'], 'nnU-Net': crc_a.loc['nnU-Net']}
for m in MODELS:
    for s in SEEDS:
        rows.append({'panel': 'A', 'model': m, 'seed': s, 'metric': 'CRC test mIoU', 'value': crc[m][f'seed{s}']})
# B: DeepOrganoid E1 / E3
e1 = pd.read_csv(A / '39_segformer_b2_deeporganoid_20261109/analysis/E1_loco.csv')
e1 = e1[(e1.cohort == 'main_all9') & (e1.spec == 'core_neg_delta')].set_index('model')
e3 = pd.read_csv(A / '39_segformer_b2_deeporganoid_20261109/analysis/E3_line_ranking.csv')
e3 = e3[e3.cohort == 'all9_all_drugs'].set_index('representation')
key = {'VitaFormer': 'vf_m2_s{}', 'SegFormer-B2': 'segformer_s{}', 'nnU-Net': 'nnunet_s{}'}
for m in MODELS:
    for s in SEEDS:
        k = key[m].format(s)
        rows.append({'panel': 'B1', 'model': m, 'seed': s, 'metric': 'E1 Pearson r (calibrated)', 'value': e1.loc[k, 'pearson'],
                     'ci_lo': e1.loc[k, 'pearson_ci_lo'], 'ci_hi': e1.loc[k, 'pearson_ci_hi']})
        rows.append({'panel': 'B2', 'model': m, 'seed': s, 'metric': 'E3 mean rho (uncalibrated)', 'value': e3.loc[k, 'mean_rho'],
                     'ci_lo': e3.loc[k, 'ci_lo'], 'ci_hi': e3.loc[k, 'ci_hi']})
# C: nnU-Net configurations
cfg = []
for d in ['A_analysis', 'A_analysis_confirm_V2', 'A_analysis_confirm_V3']:
    t = pd.read_csv(A / f'38_posthoc_robustness_20261109/{d}/E3_line_ranking.csv')
    cfg.append(t[(t.cohort == 'all9_all_drugs') & t.representation.str.match(r'nnunet_s\d+_V\d$')])
cfg = pd.concat(cfg).drop_duplicates('representation').set_index('representation')
for v in ['V0', 'V1', 'V2', 'V3']:
    for s in SEEDS:
        k = f'nnunet_s{s}_{v}'
        if k in cfg.index:
            rows.append({'panel': 'C', 'model': 'nnU-Net', 'seed': s, 'metric': f'E3 mean rho, {v}' + (' (primary)' if v == 'V0' else ' (post hoc)'),
                         'config': v, 'value': cfg.loc[k, 'mean_rho'], 'ci_lo': cfg.loc[k, 'ci_lo'], 'ci_hi': cfg.loc[k, 'ci_hi']})
# D: NViR pooled area-ATP (physical-plate clusters)
nv = pd.read_csv(R / 'functional_validation_nvir_20261109/analysis_n1_cluster_fix_20261110/N1_pooled_corrected.csv')
nv = nv[nv.analysis == 'pooled_all_images_physical_clusters'].set_index('rep')
nkey = {'VitaFormer': 'm2_s{}', 'SegFormer-B2': 'segformer_s{}', 'nnU-Net': 'nnunet_s{}'}
for m in MODELS:
    for s in SEEDS:
        k = nkey[m].format(s)
        rows.append({'panel': 'D', 'model': m, 'seed': s, 'metric': 'NViR area-ATP Spearman rho (pooled test)',
                     'value': nv.loc[k, 'rho'], 'ci_lo': nv.loc[k, 'ci_lo'], 'ci_hi': nv.loc[k, 'ci_hi']})
src = pd.DataFrame(rows); src.to_csv(OUT / 'Fig3_source_data.csv', index=False)

# ---------------- v2 plotting (2026-11-16) ----------------
# Changes vs v1: shared style; E1 and E3 as separate titled panels (B left/right); per-model mean dash + rank label in every
# strip panel so the rank crossing between in-domain accuracy and functional readouts is visible; status badges; cleaner "not run".
from style import tag, clean, MUTED as _M  # noqa: E402,F401
import style  # noqa: E402,F401  (applies shared rcParams)

LBL = ['Vita-\nFormer', 'SegFormer\n-B2', 'nnU-\nNet']


def strip(ax, panel, ylabel, ylim, ci=True, rank_top=True):
    d = src[src.panel == panel]
    means = {}
    for i, m in enumerate(MODELS):
        g = d[d.model == m].copy(); g['_o'] = g.seed.astype(str).map({'42': 0, '3407': 1, '2026': 2}); g = g.sort_values('_o')
        for j, (_, r) in enumerate(g.iterrows()):
            x = i + (j - 1) * 0.17
            if ci and pd.notna(r.get('ci_lo')):
                ax.plot([x, x], [r.ci_lo, r.ci_hi], color=COL[m], lw=1, alpha=.5, solid_capstyle='round')
            ax.plot(x, r.value, MRK[m], ms=5.5, color=COL[m], mec=SURF, mew=1.0, zorder=3)
        means[m] = g.value.mean()
        ax.plot([i - .32, i + .32], [means[m]] * 2, color=INK2, lw=1.1, zorder=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(LBL, color=INK2, fontsize=6.6)
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.set_xlim(-.55, 2.55); clean(ax)


fig = plt.figure(figsize=(7.6, 6.2))
gs = fig.add_gridspec(2, 3, hspace=.75, wspace=.55)
axA = fig.add_subplot(gs[0, 0]); strip(axA, 'A', 'CRC test mIoU (%)', (81.5, 86.5), ci=False)
axA.set_title('A  CRC in-domain accuracy', pad=14)
axB1 = fig.add_subplot(gs[0, 1]); strip(axB1, 'B1', 'Pearson r', (0, 1))
axB1.set_title('B  DeepOrganoid E1\n    calibrated CTG prediction', pad=14)
axB2 = fig.add_subplot(gs[0, 2]); strip(axB2, 'B2', 'mean Spearman ρ', (-0.65, 1))
axB2.axhline(0, color=BASE, lw=.8)
axB2.set_title('    DeepOrganoid E3\n    uncalibrated ranking', pad=14)
# C
axC = fig.add_subplot(gs[1, 0:2])
d = src[src.panel == 'C']
labels = {'V0': 'V0 default', 'V1': 'V1 reflect pad', 'V2': 'V2 ×1.27 scale', 'V3': 'V3 mirror TTA'}
smk = {'42': 'o', '3407': 's', '2026': 'D'}
vf = src[(src.panel == 'B2') & (src.model == 'VitaFormer')].value
axC.axhspan(vf.min(), vf.max(), color=COL['VitaFormer'], alpha=.13, lw=0)
axC.text(3.42, vf.mean(), 'VitaFormer\nseed range', color=COL['VitaFormer'], fontsize=6.3, va='center')
axC.axvspan(.5, 3.5, color='#fbf1e6', alpha=.6, lw=0, zorder=0)
axC.text(2.0, .97, 'post hoc (open markers)', ha='center', va='top', fontsize=6.3, color='#8a4b14', style='italic')
axC.text(0, .97, 'pre-specified', ha='center', va='top', fontsize=6.3, color=INK2)
for i, v in enumerate(['V0', 'V1', 'V2', 'V3']):
    missing = []
    for j, s in enumerate(SEEDS):
        g = d[(d.config == v) & (d.seed == s)]
        if g.empty:
            missing.append(s); continue
        r = g.iloc[0]; x = i + (j - 1) * 0.18
        axC.plot([x, x], [r.ci_lo, r.ci_hi], color=COL['nnU-Net'], lw=1, alpha=.5)
        axC.plot(x, r.value, smk[s], ms=5.5, color=COL['nnU-Net'] if v == 'V0' else SURF, mec=COL['nnU-Net'], mew=1.3, zorder=3)
    if missing:
        axC.text(i, -.5, 'seeds ' + ', '.join(missing) + '\nnot run', ha='center', va='center', fontsize=5.8, color=MUTED)
axC.axhline(0, color=BASE, lw=.8)
axC.set_xticks(range(4)); axC.set_xticklabels([labels[v] for v in ['V0', 'V1', 'V2', 'V3']], color=INK2, fontsize=6.8)
axC.set_xlim(-.5, 4.0); axC.set_ylim(-.65, 1); axC.set_ylabel('E3 mean Spearman ρ'); clean(axC)
for s, mk in smk.items():
    axC.plot([], [], mk, color=MUTED, ms=4.5, ls='', label=f'seed {s}')
axC.legend(frameon=False, fontsize=6.2, loc='lower right', bbox_to_anchor=(1.0, .02), labelcolor=INK2)
axC.set_title('C  nnU-Net input configuration (DeepOrganoid E3)')
axD = fig.add_subplot(gs[1, 2]); strip(axD, 'D', 'Spearman ρ (area vs ATP)', (0, 1))
axD.set_title('D  NViR area vs ATP\n    (pooled test split)', pad=14)
h = [plt.Line2D([], [], ls='', marker=MRK[m], color=COL[m], label=m) for m in MODELS]
h.append(plt.Line2D([], [], color=INK2, lw=1.1, label='mean of 3 seeds'))
fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.02), labelcolor=INK2, fontsize=7)
fig.text(.01, -.01, 'Points = training seeds (42, 3407, 2026); bars = 95% bootstrap CIs (E1: over lines; E3: over drugs, may be too narrow as 8/10 '
         'are HSP90 inhibitors; D: physical plate-day clusters).\nWithin each model, points left to right = seeds 42, 3407, 2026; in C, marker shape = seed. Horizontal dash = mean of 3 seeds (descriptive). '
         'A, B, D pre-specified. C: only V0 is the primary result;\nthe best post-hoc value is not substituted.', fontsize=5.8, color=MUTED)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Fig3_draft_v3.{ext}', dpi=300, bbox_inches='tight')
src.to_csv(OUT / 'Fig3_draft_v3_source_data.csv', index=False)
print(src.groupby(['panel', 'model']).value.agg(['min', 'mean', 'max']).round(3))
