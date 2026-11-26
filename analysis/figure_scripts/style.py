"""Shared draft-figure style (validated categorical slots 1-3 of the dataviz reference palette; identity also by marker + labels)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('3D_OrganoidSeg')
ANA = R / 'external_datasets/deepOrganoid/TrainingData/analysis_ctg_alignment'
NV = R / 'functional_validation_nvir_20261109'
EXT = R / 'baselines_fair_20261030'
OUT = R / 'paper_materials_20261110/figures_draft'; OUT.mkdir(exist_ok=True)
COL = {'VitaFormer': '#2a78d6', 'SegFormer-B2': '#eb6834', 'nnU-Net': '#1baf7a'}
MRK = {'VitaFormer': 'o', 'SegFormer-B2': 's', 'nnU-Net': '^'}
INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
REF = '#52514e'  # non-model references (M0, thresholds, published values): neutral ink, distinct markers
SEEDS = ['42', '3407', '2026']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                     'axes.titlesize': 8.5, 'axes.titlecolor': INK, 'axes.titlelocation': 'left'})


def clean(ax, grid='y'):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.6); ax.set_axisbelow(True)


def dot(ax, x, y, model, lo=None, hi=None, filled=True, ms=5.5, color=None, marker=None):
    c = color or COL[model]; m = marker or MRK[model]
    if lo is not None:
        ax.plot([x, x], [lo, hi], color=c, lw=1, alpha=.55, solid_capstyle='round')
    ax.plot(x, y, m, ms=ms, color=c if filled else SURF, mec=c if not filled else SURF, mew=1.2 if filled else 1.4, zorder=3)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')


def tag(ax, text, x=1.0, y=1.02):
    """Small status badge (pre-specified / post hoc / secondary) at the top-right of a panel."""
    post = 'post hoc' in text
    ax.text(x, y, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=5.6,
            color='#8a4b14' if post else INK2, style='italic' if post else 'normal',
            bbox=dict(boxstyle='round,pad=0.25', fc='#fbf1e6' if post else '#f1f0ea', ec='none'))
