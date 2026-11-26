# Stage 31 analysis plan (written 2026-10-29, before any Stage 31 result was inspected)

Models: simplified (No MPM/No BAM, primary) and full (MPM+BAM, comparator), seed 42, p1–p99 stretch, window 224/stride 112.
Cohorts (Stage 19): assay-wide = Func != MEDIA (3,312); drug-only (2,944). CTG = relative_ctg_activity = 1 − Lum_FA.

1. QC gate (before any CTG analysis): per cell line, foreground fraction distribution, all-background/near-empty counts, border-touch fraction; agreement with the Stage 28/29 fixed-90 masks (same model + same preprocessing should reproduce them); contact sheets per cell line for visual review. Organoid52 reported with and without.
2. Functional alignment (primary, unchanged from Stage 22): nested leave-one-cell-line-out ridge, NEG-delta core morphology → relative CTG; area-only comparator; 1,000 hierarchical bootstrap; 200 label permutations.
3. Dose trajectories (Stage 24 method): per cell line × compound Spearman of dose vs CTG and dose vs morphology; concordance.
4. Drug-specific morphology (the paper's question, Stage 25 method): 10 compounds shared by all 9 lines; condition-level medians; LOCO classification; baseline = concentration + CTG; increment from residual morphology; paired cell-line bootstrap. Claim "different drugs leave different morphological death phenotypes" only if the increment CI excludes 0 in all-nine AND excluding-Organoid52.
5. Descriptive per-drug phenotype profiles at matched CTG inhibition (e.g. which features shift: area loss vs fragmentation vs texture/darkening), with example images.
6. Simplified vs Full: same tables side by side; descriptive only, not used for selection.

## Pre-result specifications (added 2026-10-29, still before inspecting any Stage 31 output)

- Prior knowledge declared: Stage 25 "p1_p99_stretch_scale1_legacy" used the same Full checkpoint + same p1–p99 preprocessing + same `feats()`; its RICH NEG-delta compound increment was −0.003 [−0.022, 0.017]. Stage 31 `full` is therefore expected to largely reproduce that table (reproducibility check: per-feature agreement with `deepOrganoid_v1/features/full_features.csv`). The simplified model is the genuinely new result.
- Primary feature set for analyses 2 and 4: CORE 9 features, NEG-delta (same as Stage 22/25 primary). Secondary: RICH 21 features, NEG-delta. Both reported for both models; no switching after seeing results.
- Drug-difference claim threshold as stated in item 4, using the CORE primary set of the simplified model. If it fails, the paper reports drug-level dose–morphology trajectories and descriptive phenotypes only, without a "drug-specific death phenotype" claim.
