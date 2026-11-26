# Amendment 2 (2026-11-18, author request "这个需要再用另外两个模型"): SegFormer-B2 and nnU-Net on GemcitabineScreen
Written after the M2 results (RESULTS.md) were seen and before any SegFormer-B2 / nnU-Net output on this dataset.

## What is added
- Families: SegFormer-B2 (best_val_miou.pth, seeds 42/3407/2026) and nnU-Net (run-2 checkpoint_final, seeds 42/3407/2026);
  checkpoints, hashes and inference functions exactly as in NViR (scripts/nvir_infer_20261109.py build()); nnU-Net without
  mirroring. P(Dead) for the pixel analysis = the model's own softmax/probability output for class 2 (class order
  background/alive/dead/bubble, identical in all three families).
- Input: identical to M2 (per-frame 1st–99th percentile stretch of the uint16 BF frame to uint8, replicated to RGB).
- Scale: per family, seed 42 only, {1.0, 0.75, 0.5, 0.25}, mean foreground Dice vs the authors' OrganoID labels on the t = 0
  frames, largest factor within 0.005 of the best; PI never read at this step (same rule as the M2 plan). If the chosen
  factor's mean Dice < 0.5, the family is "segmentation failed" and its PI analyses are reported as not interpretable.
- Endpoints, statistics, bootstrap (10,000 well draws stratified by dose, seed 20261109) and outputs: the unchanged M2 analysis
  (scripts/gem_pi_analyze_20261109.py), run on all nine model–seed runs into analysis_families/ (copy with only the model list
  and output directory changed). The pre-specified primary for each family is the same as for M2: ρ(Dead fraction, PI) at 52 h,
  seed 42; the area comparator (foreground fold change vs PI at 52 h) is the main cross-family quantity. The M2 permutation
  test is not repeated for the other families.
- Model comparison: descriptive (point estimates and CIs per seed); no new hypothesis test.
- M2 outputs are reused; M2 rows in analysis_families/ must reproduce analysis/summary.csv (check).
