#!/usr/bin/env python3
"""v7 (2026-11-18): Table S1 gains GemcitabineScreen columns for all 9 runs (Amendment 2). Otherwise identical to v6.
Supplementary Tables S1-S6 for manuscript v6 (2026-11-16, after review). Reads existing result files only; no new analysis.
Changes vs v5: S3 header no longer uses '|' (broke Markdown columns); S4 note lists the rare-class settings actually active in the
M2 runs (launch command + train.log); S5 note no longer states that the patch-512 rule establishes the field-of-view explanation;
new S6 = checkpoint inventory (target/actual steps, non-finite events, selection rule, step used, hardware, full SHA256).
v5 script and outputs are left unchanged; this writes SUPPLEMENTARY_TABLES_v6.md and *_v6.csv."""
import hashlib
import importlib.util
import pandas as pd
from pathlib import Path
from style import R, OUT

spec = importlib.util.spec_from_file_location('v5', Path(__file__).with_name('supplementary_tables_v5.py'))
# v5 builds S1-S5 and writes v5 files; we reuse its objects by executing it, then overwrite nothing of v5 except by re-writing
# identical content (deterministic). To avoid even that, import its pieces manually:
src = Path(__file__).with_name('supplementary_tables_v5.py').read_text()
src = src.split("(OUT / 'SUPPLEMENTARY_TABLES_v5.md').write_text")[0]
src = src.replace("s4.to_csv(OUT / 'TableS4_vitaformer_ablations.csv', index=False)", '')
src = src.replace("s5.to_csv(OUT / 'TableS5_nnunet_ablations.csv', index=False)", '')
src = src.replace("s3.insert(0, 'Model', s3.index); s3.to_csv(OUT / 'TableS3_crc_all_models.csv', index=False)", "s3.insert(0, 'Model', s3.index)")
ns = {'__file__': __file__}
exec(compile(src, 'supplementary_tables_v5_body', 'exec'), ns)
md, s1, s1r, s2, s3, s4, s5 = (ns[k] for k in ['md', 's1', 's1r', 's2', 's3', 's4', 's5'])


# ---- v7: GemcitabineScreen columns (analysis_families/summary.csv)
import pandas as _pd
_g = _pd.read_csv(R / 'baselines_fair_20261030/external_m2_20261108/gemcitabine_pi/analysis_families/summary.csv')
_g = _g[_g.time_h == 52]
_FAM = {'VitaFormer': 'm2', 'SegFormer-B2': 'segformer', 'nnU-Net (V0)': 'nnunet'}
def _gv(m, sd, col):
    r = _g[_g.model == f"{_FAM[m]}_seed{sd}"].iloc[0]; return r[col]
s1['Gem_area_PI_rho_52h'] = [round(_gv(m, sd, 'rho_M2_fg_fc_PI'), 3) for m, sd in zip(s1.model, s1.seed)]
s1['Gem_area_PI_ci'] = [f"[{_gv(m, sd, 'rho_M2_fg_fc_PI_lo'):.3f}, {_gv(m, sd, 'rho_M2_fg_fc_PI_hi'):.3f}]" for m, sd in zip(s1.model, s1.seed)]
s1['Gem_area_EC50_nM'] = [round(float(_gv(m, sd, 'EC50_M2_fg_px')), 1) for m, sd in zip(s1.model, s1.seed)]
s1['Gem_DF_PI_rho_52h'] = ['NA' if _pd.isna(_gv(m, sd, 'rho_DF_PI')) else round(_gv(m, sd, 'rho_DF_PI'), 3) for m, sd in zip(s1.model, s1.seed)]
s1r['Gem_area_PI_rho_52h'] = [round(s1[s1.model == m].Gem_area_PI_rho_52h.max() - s1[s1.model == m].Gem_area_PI_rho_52h.min(), 3) for m in s1r.model]
s1.to_csv(OUT / 'TableS1_per_seed_v7.csv', index=False); s1r.to_csv(OUT / 'TableS1_observed_seed_ranges_v7.csv', index=False)

# ---- S3: rename columns without '|'
s3 = s3.rename(columns=lambda c: c.replace('|area err| median %', 'abs. area error, median %'))
s3.to_csv(OUT / 'TableS3_crc_all_models_v6.csv', index=False)
s3_note = ('CRC test set (167 images, 154 wells), shared evaluator, mean ± SD over seeds 42/3407/2026. Absolute area error = per-image '
           '|P − G|/G; the value per seed is the median over images whose annotation contains the class (Alive 167, Dead 24, Bubble 22), '
           'and the table gives mean ± SD of these per-seed medians. OrganoidViT, DeepLabv3, U-Net++ and U-Net: models of the source '
           'publication, best-validation checkpoints; several runs produced non-finite losses or had not converged (Figure S1, Table S6). '
           'Descriptive; no contrasts.')

# ---- S4: note with the settings actually used
s4.to_csv(OUT / 'TableS4_vitaformer_ablations_v6.csv', index=False)
fp = pd.read_csv(R / 'vco/runs/m2_ablations_eval_20261113/analysis_final/per_model_summary.csv', header=[0, 1], index_col=0)
fpm = fp[('mean', 'far_bg_dead_fp_px')]
s4_note = (
    'Validation set (164 images), final checkpoint (step 40,000), seed-matched mean of 3 seeds; hierarchical bootstrap over 154 wells '
    '(10,000 replicates); "k/3" = seeds with Δ > 0. fg BF1 (2 px) = per-image boundary F1 of the foreground (Alive ∪ Dead) mask with '
    '2-pixel tolerance at native resolution (boundary = mask minus its 3 × 3 erosion; Euclidean distance transform; 1 if both '
    'boundaries are empty, 0 if only one is), averaged over images. Full VitaFormer test mIoU 83.34 ± 0.25. Far-background false Dead '
    'pixels = predicted Dead pixels on valid annotated pixels > 8 px (Euclidean, native resolution) from any annotated Alive, Dead or '
    'Bubble pixel, summed over the 164 validation images per seed, mean of 3 seeds: full model '
    f'{fpm["M2"]:,.0f}; w/o rare-class recipe {fpm["noRD"]:,.0f}; ImageNet {fpm["ImageNet"]:,.0f}. '
    'Rare-class recipe as run for the full model (launch command and training log): focal loss (γ = 2) with class weights '
    '(background 0.5, Alive 0.5, Dead 10, Bubble 0.9); Dead-centred cropping (probability 0.6, ≥ 20 Dead pixels in the crop); '
    'sparse-Dead zoom (probability 0.8, images with < 5,000 Dead pixels, factor 2); 3× duplication of 5-fluorouracil images with Dead; '
    'minority-biased cropping (fraction 0.4). The "w/o" flag switches all of these off (plain cross-entropy, no class weights, '
    'random crops); it also zeroes Dead-pixel dropout and Dead copy-paste, which were already 0 in the full model. The '
    'condition-stratified data split is not part of the recipe and is unchanged. The effect is that of the recipe as a whole; '
    'components were not ablated individually.')

# ---- S5: rewritten note
s5 = s5.rename(columns={'Variant (default nnU-Net − variant)': 'Variant (default nnU-Net − variant)'})
s5['Variant (default nnU-Net − variant)'] = s5['Variant (default nnU-Net − variant)'].replace(
    {'512 × 512 patches (3 seeds)': '512 × 512 patches, batch 10 (3 seeds)'})
s5.to_csv(OUT / 'TableS5_nnunet_ablations_v6.csv', index=False)
s5_note = ('Exploratory. Validation set, nnU-Net\'s own fold-0 validation predictions from checkpoint_final; same bootstrap as Table S4. '
           'The 512 × 512 configuration changes patch size and batch size together (batch 2 → 10, equal pixels per iteration), so it '
           'does not isolate field of view. Its validation interval (−1.48 to +2.75) is wide and is not evidence of equivalence; the '
           'pre-specified threshold |Δ| < 1.0 refers to the point estimate only. The cross-entropy-only run has one seed and its Dead-'
           'recall interval includes 0 (pre-specified reading: inconclusive). Runs were resumed after file-system outages (recorded in '
           'the deposited plans).')

# ---- S6: checkpoint inventory
def sha(p):
    p = Path(p)
    if not p.exists():
        return 'file not found'
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


BR = R.parent / 'OrganoidSegBaseline/runs'
tr = pd.read_csv(OUT / 'FigS1_source_training_summary.csv'); tr['seed'] = tr.seed.astype(str)
tr = tr.set_index(['model', 'seed'])
rows = []
VF = {'42': 'stageB_m2_seed42', '3407': 'stageC_m2_seed3407', '2026': 'stageC_m2_seed2026'}
for s, d in VF.items():
    rows.append(dict(model='VitaFormer', seed=s, target_steps='40,000 steps', non_finite='none (0 non-finite values in weights)',
                     rule='fixed final step (adopted after an intermediate comparison; applied to all variants)', used='40,000',
                     hardware='A800, TF32' if s != '2026' else 'V100S, fp32',
                     sha256=sha(R / f'vco/runs/arch_exploration_20261104/{d}/final.pth')))
for s in ['42', '3407', '2026']:
    rows.append(dict(model='nnU-Net v2', seed=s, target_steps='1,000 epochs × 250 iterations', non_finite='none logged',
                     rule='checkpoint_final (nnU-Net default)', used='final epoch', hardware='[VERIFY]',
                     sha256=sha(R / f'baselines_fair_20261030/nnunet/nnUNet_results_run2_seed{s}/Dataset801_CRCOrganoid/'
                                f'nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth')))
SEGSHA = pd.read_csv(R / 'baselines_fair_20261030/stability_review_20261102/checkpoint_audit.csv')
SEGSHA = SEGSHA[(SEGSHA.arm == 'baseline') & (SEGSHA.file == 'best_val_miou.pth')]
SEGSHA = SEGSHA.set_index(SEGSHA.seed.astype(str))
for s in ['42', '3407', '2026']:
    t = tr.loc[('SegFormer-B2', s)]
    nf = 'none' if pd.isna(t.first_nonfinite_loss_iter) else f'loss non-finite from {int(t.first_nonfinite_loss_iter):,}'
    rows.append(dict(model='SegFormer-B2', seed=s, target_steps='40,000 iterations', non_finite=nf,
                     rule='best validation mIoU (evaluated every 1,000)', used=f'{int(t.best_val_iter):,}', hardware='[VERIFY]',
                     sha256=SEGSHA.loc[s, 'sha256']))
for name, d in [('OrganoidViT', 'orgvit'), ('DeepLabv3', 'deeplabv3'), ('U-Net++', 'unetpp'), ('U-Net', 'unet')]:
    for s in ['42', '3407', '2026']:
        t = tr.loc[(name, s)]
        nf = 'none' if pd.isna(t.first_nonfinite_loss_iter) else f'loss non-finite from {int(t.first_nonfinite_loss_iter):,}'
        if d == 'orgvit':
            p = BR / f'orgvit_b0_1446_40k_cond_seed{s}_a800/iter_{int(t.best_val_iter)}.pth'; hw = 'A800 (run name)'; ev = 2000
        else:
            p = BR / (f'{d}_cond_40k_torch' if s == '2026' else f'{d}_cond_40k_torch_seed{s}') / 'best_miou.pth'; hw = '[VERIFY]'; ev = 1000
        rows.append(dict(model=name, seed=s, target_steps='40,000 iterations', non_finite=nf,
                         rule=f'best validation mIoU (evaluated every {ev:,})', used=f'{int(t.best_val_iter):,}', hardware=hw,
                         sha256=sha(p)))
s6 = pd.DataFrame(rows).rename(columns={'model': 'Model', 'seed': 'Seed', 'target_steps': 'Planned training', 'non_finite': 'Non-finite events',
                                        'rule': 'Checkpoint rule', 'used': 'Step/iteration used', 'hardware': 'Hardware',
                                        'sha256': 'SHA256 of the weights used'})
s6.to_csv(OUT / 'TableS6_checkpoint_inventory.csv', index=False)
s6_note = ('All checkpoints used for test or external analyses were taken before any non-finite loss. For SegFormer-B2 and the '
           'source-publication models, a non-finite loss ended useful training (validation mIoU then fell to about 25%, background only), so '
           'the best-validation rule selected an earlier checkpoint; these runs therefore reflect a shorter effective training. A '
           'pre-specified SegFormer-B2 re-run with bf16 autocast and gradient clipping (3 seeds, no non-finite loss) gave test mIoU '
           '81.81 ± 1.41 vs 82.92 ± 0.64 for the original runs (Δ −1.11 [−3.11, +0.17]); the original runs remain the primary record. '
           'Differences between runs combine seed, checkpoint rule, training numerics and, for VitaFormer seed 2026, hardware; they are '
           'not pure seed effects. [VERIFY] = hardware not recorded in the run directory.')

doc = ['# Supplementary tables (draft v7, 2026-11-18)', '',
       '*Generated by `figure_scripts/supplementary_tables_v7.py` from existing result files; no new analysis.*', '',
       '## Table S1. Per-seed values of all endpoints', '', md(s1), '',
       'Observed seed range (maximum − minimum over three seeds):', '', md(s1r), '',
       'DO = DeepOrganoid; E1 Pearson r (calibrated, CI over lines); E2 dose-direction concordance; E3 mean within-drug ρ over the 10 drugs '
       'tested in all 9 lines (90 pairs; CI over drugs); E3b pooled ρ over all 92 line–drug pairs (including 5-FU, tested in 2 lines); '
       'NViR N1 pooled area–ATP ρ (CI over physical plate-day clusters); N1b dose-direction series; test Dice per domain. '
       'Gem = GemcitabineScreen at 52 h: area fold change vs PI ρ (CI: 10,000 well bootstraps stratified by dose), area EC50, and the '
       'pre-specified Dead-fraction vs PI ρ (NA = no Dead pixel predicted; SegFormer-B2 and nnU-Net added by amendment).', '',
       '## Table S2. NViR results at every candidate input scale (seed 42)', '', md(s2), '',
       'Pre-specified supplementary table. Scale chosen by training-split Dice against the reference masks (ATP not read); identifying '
       'an ATP-optimal scale from this table is post hoc.', '',
       '## Table S3. CRC test results for all seven models', '', md(s3), '', s3_note, '',
       '## Table S4. VitaFormer ablations', '', md(s4), '', s4_note, '',
       '## Table S5. Exploratory nnU-Net ablations', '', md(s5), '', s5_note, '',
       '## Table S6. Checkpoint inventory', '', md(s6), '', s6_note, '']
(OUT / 'SUPPLEMENTARY_TABLES_v7.md').write_text('\n'.join(doc))
print(s6.drop(columns='SHA256 of the weights used').to_string()); print(s6['SHA256 of the weights used'].str[:12].tolist())
