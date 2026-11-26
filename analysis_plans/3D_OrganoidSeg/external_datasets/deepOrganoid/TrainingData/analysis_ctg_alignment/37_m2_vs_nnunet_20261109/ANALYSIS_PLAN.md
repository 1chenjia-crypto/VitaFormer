# Stage 37: frozen M2 vs frozen nnU-Net on DeepOrganoid (plan, 2026-11-09; approved by the author the same day before running)

Purpose: Stage 35 compared nnU-Net with M0 (VF-simplified). After the M2 freeze, the paper's VitaFormer is M2. This stage reruns
the **Stage 35 analysis unchanged** (`35_.../ANALYSIS_PLAN.md`: endpoints E1/E1b/E1c/E2/E3/E3b, cohorts, bootstrap seeds,
decision rules §4) with the reference representation **M2 seed42** (Stage 36 features) in place of VF-simplified.
No new inference and no training.

Inputs, all existing and read-only:
- Stage 36 `m2_seed{42,3407,2026}/features_shard0of1.csv` + `hard_class_masks/`, `overlays_jpg/`;
- Stage 35 `nnunet_seed{42,3407,2026}/features.csv` + masks / overlays;
- Stage 31 `simplified` (M0; kept as a reproduction check: its E1 core must equal 0.779 and its E3 must equal 0.753);
- Stage 32 thresholds.

Representations: vf_m2_s42 (REF, primary), vf_m2_s3407, vf_m2_s2026 (descriptive), vf_simplified (M0, check), nnunet_s42 / s3407 / s2026.
Paired Δ = vf_m2_s42 − X with the same draws as Stage 35.

Decision rules: identical to Stage 35 §4, with "VitaFormer" = M2 seed42:
- "M2 has an independent downstream advantage over nnU-Net" requires:
  - the CI of Δ vs nnU-Net s42 excludes 0 on E1 (primary) or E3;
  - the same direction excluding Organoid52;
  - the same direction vs ≥ 2 of the 3 nnU-Net seeds.
- The symmetric rule applies for "nnU-Net better". Otherwise the result is "equivalent".
- The Stage 35 mandatory qualification (nnU-Net QC failure, absolute vs relative readouts) applies unchanged.

Disclosure: the marginal values of both models are already known from Stages 35 and 36 (E1 core M2 0.769 vs nnU-Net s42 0.777;
E3 M2 0.717 vs nnU-Net s42 0.068). Only the paired differences and CIs are new. The outcome on E3 is therefore largely
predictable; this stage formalises it for M2 and does not provide independent new evidence.

Script: `scripts/analyze_stage37_m2_vs_nnunet_20261109.py`. It is a copy of the Stage 35 script with only the model map,
input paths, reference and output directory changed. Output goes to `37_m2_vs_nnunet_20261109/{analysis,review_panels}` (new).
