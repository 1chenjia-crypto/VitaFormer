#!/usr/bin/env python3
"""Figure 4 (draft): what DeepOrganoid readouts reflect, VitaFormer seed 42 as the worked example (other seeds/models in Fig 3 & Table S1).
A concentration-averaged area vs CTG summaries over line-drug pairs (E3b; no CTG fitting; CI over 9 lines)
B within-drug line ranking per drug (E3; no CTG fitting) vs global Otsu and local-contrast thresholding (CI over 10 drugs)
C calibrated: leave-one-line-out Ridge predictions vs CTG (E1; CI over lines; 200-permutation p)
D post hoc general-sensitivity analysis (Stage 39B; drug bootstrap 2,000)."""
import pandas as pd
import matplotlib.pyplot as plt
from style import ANA, COL, INK2, MUTED, BASE, REF, SURF, clean, save, tag, OUT

S39 = ANA / '39_segformer_b2_deeporganoid_20261109'
auc = pd.read_csv(S39 / 'analysis/E3_auc_table.csv')
a = auc[auc.representation == 'vf_m2_s42']
pot = pd.read_csv(S39 / 'analysis/E3_pooled_potency.csv'); pot = pot[(pot.cohort == 'all9_all_drugs') & (pot.representation == 'vf_m2_s42')].iloc[0]
rk = pd.read_csv(S39 / 'analysis/E3_line_ranking.csv'); rk = rk[rk.cohort == 'all9_all_drugs'].set_index('representation')
per = {}
for rep in ['vf_m2_s42', 'threshold_global_otsu', 'threshold_local_contrast']:
    per[rep] = {k: float(v) for k, v in (x.rsplit(':', 1) for x in rk.loc[rep, 'per_drug'].split(';'))}
per = pd.DataFrame(per)
oof = pd.read_csv(S39 / 'analysis/E1_loco_oof_predictions.csv')
o = oof[(oof.cohort == 'main_all9') & (oof.model == 'vf_m2_s42') & (oof.spec == 'core_neg_delta')]
e1 = pd.read_csv(S39 / 'analysis/E1_loco.csv'); e1 = e1[(e1.cohort == 'main_all9') & (e1.spec == 'core_neg_delta') & (e1.model == 'vf_m2_s42')].iloc[0]
e2 = pd.read_csv(S39 / 'analysis/E2_dose_summary_stage31_method.csv')
e2 = e2[(e2.spec == 'core_neg_delta') & e2.model.str.startswith('vf_m2')][['model', 'strong_ctg_n', 'concordant_fraction', 'concordant_ci_lo', 'concordant_ci_hi']]
gs_ = pd.read_csv(S39 / 'B_general_sensitivity/summary.csv'); g = gs_[(gs_.cohort == 'all9_all_drugs') & (gs_.model == 'vf_m2_s42')].set_index('metric')
a.to_csv(OUT / 'Fig4A_source_auc_pairs.csv', index=False); per.to_csv(OUT / 'Fig4B_source_per_drug_rho.csv')
o[['sample_id', 'CCL', 'Cmpd1', 'Conc1', 'relative_ctg_activity', 'prediction']].to_csv(OUT / 'Fig4C_source_oof.csv', index=False)
e2.to_csv(OUT / 'Fig4C_source_E2.csv', index=False); g.to_csv(OUT / 'Fig4D_source_general_sensitivity.csv')

# ---------------- v2 plotting (2026-11-16) ----------------
# Changes vs v1: balanced layout; status badges (A-C pre-specified, D post hoc); A reports all 3 seeds; B adds model means and the
# CTG split-half ceiling (0.776, Stage 34 CRC_RESULTS_FREEZE / EXPERIMENT_CLOSEOUT) as reference lines; C as density (hexbin);
# D grouped into ranking vs held-out prediction.
CEIL = 0.776
pots = pd.read_csv(S39 / 'analysis/E3_pooled_potency.csv'); pots = pots[pots.cohort == 'all9_all_drugs'].set_index('representation')
c = COL['VitaFormer']
fig = plt.figure(figsize=(7.6, 6.0)); G = fig.add_gridspec(2, 2, hspace=.7, wspace=.75, width_ratios=[1, 1.15])
# A
ax = fig.add_subplot(G[0, 0])
ax.plot(a.auc_x, a.auc_ctg, 'o', ms=3.4, color=c, alpha=.75, mec='none')
s3 = ' / '.join(f'{pots.loc[f"vf_m2_s{s}", "spearman_auc"]:.3f}' for s in ['42', '3407', '2026'])
ax.text(.03, .97, f'Spearman ρ, seed 42: {pot.spearman_auc:.3f} [{pot.ci_lo:.3f}, {pot.ci_hi:.3f}]\nseeds 42 / 3407 / 2026: {s3}\n'
        f'n = {int(pot.n_pairs)} line–drug pairs\n(90 = 10 drugs × 9 lines, + 5-FU in 2 lines)', transform=ax.transAxes, ha='left', va='top', fontsize=5.8, color=INK2)
ax.set_ylim(0, 1.38)
ax.set_xlabel('Area response (rel. to vehicle, conc.-averaged)'); ax.set_ylabel('CTG response (conc.-averaged)')
clean(ax, 'both'); ax.set_title('A  Response summaries, no CTG fitting', pad=12); tag(ax, 'pre-specified', y=1.0)
# B
ax = fig.add_subplot(G[0, 1])
order = per.vf_m2_s42.sort_values().index; y = list(range(len(order)))
ax.axvline(0, color=BASE, lw=.8)
ax.axvline(CEIL, color=INK2, lw=.9, ls=':')
ax.text(CEIL + .02, -.7, f'CTG half vs half\n(2+2 replicates)\n{CEIL}', fontsize=5.2, color=INK2, ha='left', va='center')
for rep, mk, col, lab, ms in [('threshold_global_otsu', 'x', REF, 'Global Otsu', 4), ('threshold_local_contrast', '+', MUTED, 'Local contrast', 5),
                              ('vf_m2_s42', 'o', c, 'VitaFormer s42', 4.5)]:
    ax.plot(per.loc[order, rep], y, mk, color=col, ms=ms, label=f'{lab} (mean {rk.loc[rep, "mean_rho"]:.2f})')
    ax.plot([rk.loc[rep, 'mean_rho']] * 2, [-.9, -.5], color=col, lw=2)
ax.set_yticks(y); ax.set_yticklabels([d.replace(' HCl', '').replace('Alvespimycin (17-DMAG)', '17-DMAG') for d in order], fontsize=6.2, color=INK2)
ax.set_ylim(-1, len(order) - .3); ax.set_xlabel('Spearman ρ across 9 lines (area vs CTG)\n10 drugs × 9 lines = 90 pairs'); clean(ax, 'x')
ax.legend(frameon=False, fontsize=5.8, loc='upper center', bbox_to_anchor=(.45, -.3), ncol=2, labelcolor=INK2)
ax.set_title('B  Within-drug ranking of lines', pad=12); tag(ax, 'pre-specified', y=1.0)
# C
ax = fig.add_subplot(G[1, 0])
hb = ax.hexbin(o.prediction, o.relative_ctg_activity, gridsize=35, bins='log', cmap='Blues', mincnt=1, linewidths=0)
lim = [min(o.prediction.min(), o.relative_ctg_activity.min()), max(o.prediction.max(), o.relative_ctg_activity.max())]
ax.plot(lim, lim, color=BASE, lw=.8)
cb = fig.colorbar(hb, ax=ax, fraction=.045, pad=.02); cb.ax.tick_params(labelsize=5.5); cb.set_label('wells (log)', fontsize=5.8, color=INK2)
txt = ' / '.join(f'{r.concordant_fraction * 100:.1f}%' for r in e2.set_index('model').loc[['vf_m2_s42', 'vf_m2_s3407', 'vf_m2_s2026']].itertuples())
ax.text(.03, .97, f'r = {e1.pearson:.3f} [{e1.pearson_ci_lo:.3f}, {e1.pearson_ci_hi:.3f}]; perm. p = 0.005\n'
        f'dose direction: {txt}\n(seeds 42 / 3407 / 2026)', transform=ax.transAxes, va='top', fontsize=5.4, color=INK2)
ax.set_xlabel('Predicted CTG (held-out line)'); ax.set_ylabel('Relative CTG activity'); clean(ax, 'both')
ax.set_title('C  Calibrated prediction (LOCO Ridge)', pad=12); tag(ax, 'pre-specified', y=1.0)
# D
ax = fig.add_subplot(G[1, 1])
items = [('baseline', 'G alone'), ('image', 'Image readout'), ('partial', 'Image | G (partial)'), None,
         ('pred_G', 'G'), ('pred_GX', 'G + image'), ('delta_pred_GX_minus_pred_G', 'Δρ (G + image − G)')]
yy = 0; yt, yl = [], []
for it in items:
    if it is None:
        yy += .6; continue
    k, lab = it; r = g.loc[k]; col = c if k in ('image', 'partial', 'pred_GX') else REF
    ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=1.1, alpha=.7); ax.plot(r.mean_rho, yy, 'o', color=col, ms=4.5)
    ax.text(r.ci_hi + .03, yy, (f'Δρ {r.mean_rho:+.3f}' if k.startswith('delta') else f'ρ {r.mean_rho:.3f}'), va='center', fontsize=6.2, color=INK2)
    yt.append(yy); yl.append(lab); yy += 1
ax.set_yticks(yt); ax.set_yticklabels(yl, color=INK2, fontsize=6.5); ax.set_ylim(yy - .5, -1.2)
ax.text(-.14, -.75, 'Ranking (ρ)', fontsize=6.2, color=INK2, weight='bold', transform=ax.transData)
ax.text(-.14, 3.05, 'Held-out ranking', fontsize=6.2, color=INK2, weight='bold', transform=ax.transData)
ax.axvline(0, color=BASE, lw=.8); ax.set_xlim(-.15, 1.05); ax.set_xlabel('Mean within-drug ρ or Δρ (95% CI)'); clean(ax, 'x')
ax.set_title('D  General line sensitivity G', pad=12); tag(ax, 'post hoc', y=1.0)
fig.text(.01, -.04, 'VitaFormer seed 42 shown unless stated; all seeds and models in Figure 3 and Table S1. G = each line\'s mean CTG response to the other drugs. '
         'B: short bars at the bottom mark each method\'s mean. CIs: A over 9 lines; B over 10 drugs (8 HSP90 inhibitors;\nmay be too narrow); C over 9 lines; D over 10 drugs. A includes 5-FU (tested in 2 lines; 92 pairs); '
         'B and D use the 10 drugs tested in all 9 lines (90 pairs). CTG half vs half: within-drug line ranking from two halves (2 + 2) of the 4 CTG replicate wells,\naveraged over the 3 pairings; a reference for CTG replicate agreement, not an upper bound for image readouts. G uses CTG of the other drugs (assay-informed baseline).', fontsize=5.8, color=MUTED)
save(fig, 'Fig4_draft_v3')
print(pots.loc[['vf_m2_s42', 'vf_m2_s3407', 'vf_m2_s2026'], 'spearman_auc'].round(3).tolist(), rk.loc[['threshold_global_otsu', 'threshold_local_contrast'], 'mean_rho'].round(3).tolist())
