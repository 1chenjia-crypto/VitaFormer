#!/usr/bin/env python3
"""Cross-check: every key number in MANUSCRIPT_RESOURCES_v4.md is recomputed from the figure source tables and must appear in the
text with the stated rounding. Writes figures_draft/TEXT_SOURCE_CHECK.md. A FAIL means the text or the source must be inspected."""
import re
import pandas as pd
from style import R, OUT

T = (R / 'paper_materials_20261110/MANUSCRIPT_RESOURCES_v4.md').read_text()
T = T[:T.index('## RESOURCE AVAILABILITY')]
s1 = pd.read_csv(OUT / 'TableS1_per_seed.csv'); s2 = pd.read_csv(OUT / 'TableS2_nvir_all_scales.csv')
f2 = pd.read_csv(OUT / 'Fig2_source_data.csv'); c2 = pd.read_csv(OUT / 'Fig2B_source_contrasts.csv')
q = pd.read_csv(OUT / 'Fig5BC_source_quintiles.csv'); g4 = pd.read_csv(OUT / 'Fig4D_source_general_sensitivity.csv').set_index('metric')
gem = pd.read_csv(OUT / 'Fig5E_source_gemcitabine.csv'); n2 = pd.read_csv(OUT / 'Fig5F_source_N2.csv')
DOM = {'A_echo_rebel': 'A', 'B_echo_revolve_v0': 'B', 'A_cytation_v0': 'C', 'A_thorlabs': 'D'}
fmt = lambda v, d: f'{v:.{d}f}'.replace('-', '−')
checks = []


def chk(label, value, d=3, also=None):
    s = fmt(value, d); ok = s in T or (s.replace('−', '-') in T) or (also is not None and also in T)
    checks.append((label, s, 'PASS' if ok else 'FAIL'))


def m(model, col):
    return s1[s1.model == model][col]


for mod, lab in [('VitaFormer', 'VitaFormer'), ('SegFormer-B2', 'SegFormer-B2'), ('nnU-Net (V0)', 'nnU-Net')]:
    for col, d in [('DO_E3_mean_rho', 3), ('NViR_N1_rho', 3)]:
        for v in m(mod, col):
            chk(f'{lab} {col} seed value', v, d)
    e1 = m(mod, 'DO_E1_pearson'); chk(f'{lab} E1 min', e1.min()); chk(f'{lab} E1 max', e1.max())
    for col in ['DO_E3_mean_rho', 'NViR_N1_rho']:
        v = m(mod, col); chk(f'{lab} {col} observed range', v.max() - v.min())
chk('VitaFormer CRC mIoU mean', f2[(f2.model == 'VitaFormer') & (f2.endpoint == 'mIoU')].value.mean(), 2)
for mod in ['SegFormer-B2', 'nnU-Net', 'M0']:
    chk(f'{mod} CRC mIoU mean', f2[(f2.model == mod) & (f2.endpoint == 'mIoU')].value.mean(), 2)
for _, r in c2.iterrows():
    chk(f'CRC {r.contrast} delta (abs)', abs(r.delta_mean), 2); chk(f'CRC {r.contrast} ci_hi', r.ci95_high, 2)
for ep, mod, d in [('Dead recall', 'VitaFormer', 1), ('Dead recall', 'nnU-Net', 1), ('Dead recall', 'SegFormer-B2', 1),
                   ('Dead precision', 'VitaFormer', 1), ('Dead precision', 'nnU-Net', 1)]:
    chk(f'{mod} {ep}', f2[(f2.model == mod) & (f2.endpoint == ep)].value.mean(), d)
for v in m('VitaFormer', 'DO_E3b_rho'):
    chk('VitaFormer E3b seed', v)
for v in m('VitaFormer', 'DO_E2_concordant'):
    chk('VitaFormer E2 concordant %', v * 100, 1)
for k, lab in [('baseline', 'G alone'), ('partial', 'partial'), ('delta_pred_GX_minus_pred_G', 'increment')]:
    chk(f'38B {lab}', g4.loc[k, 'mean_rho'])
chk('NViR lowest-quintile Dice', q.iloc[0].dice_m2, 2)
for (dom, d) in DOM.items():
    chk(f'N2 R2 domain {d} (3-seed mean)', n2[n2.domain == dom].r2.mean(), 2)
for mdl in ['m2_seed42', 'm2_seed3407', 'm2_seed2026']:
    chk(f'Gemcitabine {mdl} 52 h rho', gem[(gem.model == mdl) & (gem.time_h == 52)].rho_M2_fg_fc_PI.iloc[0], 2)
x = s2[(s2.family == 'nnunet') & (s2.scale == '768')].iloc[0]; chk('nnU-Net 768 px rho (Table S2)', x.test_N1_rho)
nat = s2[s2.scale == 'native'].train_median_Dice; sel = s2[s2.pre_specified_choice].train_median_Dice
chk('native Dice min', nat.min(), 2); chk('native Dice max', nat.max(), 2); chk('selected Dice min', sel.min(), 2); chk('selected Dice max', sel.max(), 2)
rep = pd.DataFrame(checks, columns=['claim', 'value_from_source', 'found_in_text'])
(OUT / 'TEXT_SOURCE_CHECK.md').write_text('# Text vs source check (MANUSCRIPT_RESOURCES_v4.md, main text)\n\n' + rep.to_markdown(index=False) +
                                          f'\n\nPASS {sum(rep.found_in_text == "PASS")} / {len(rep)}\n')
print(rep[rep.found_in_text == 'FAIL'].to_string(index=False)); print('PASS', sum(rep.found_in_text == 'PASS'), '/', len(rep))
