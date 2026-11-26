# Amendment 5 — Full VitaFormer fixed-40k check (2026-11-06, POST HOC, author approved)
Motivation: Stage D showed random-crop best_miou selection is noise; Full seed 2026's selected checkpoint was step 10k,
so the Full vs Simplified (MPM/BAM removal) comparison may be affected. Inference only; no training; nothing overwritten.
Checkpoints (final.pth, verified step 40000, MPM 4 / BAM 8 tensors present):
- Full seed 42 = expFinal_fromscratch (same Full used in P0-M as full_seed42; May-2026 version of train_v2_stratified.py,
  same hyperparameters as M0 launch; its log has no explicit seed line -> script default; caveat recorded).
- Full seed 3407 / 2026 = full_model_seed3407 / full_model_seed2026 (same config as M0 except use_mpm/use_bam True; 60.63 M vs 60.62 M trainable).
Inference: frozen infer_hard (224/112), 164 CRC val images; model OrganoidSegV2(use_mpm=True, use_bam=True), output tuple -> seg.
Pipeline check: 2 TEST images re-predicted with best_miou.pth must equal existing P0-M full_seed*/masks (test masks; no metrics read).
Analysis: scripts/stageE_analysis.py = stageD_analysis.py with model set {M0, Full, M2, nnUNet} at final (40k),
pairs Full−M0, M2−Full, M2−M0, *−nnUNet; same metrics and well-cluster bootstrap. Validation only; test/external not used.
Owner of stageE_full_40k/: Claude Code session (this one). No other agent writes there.
