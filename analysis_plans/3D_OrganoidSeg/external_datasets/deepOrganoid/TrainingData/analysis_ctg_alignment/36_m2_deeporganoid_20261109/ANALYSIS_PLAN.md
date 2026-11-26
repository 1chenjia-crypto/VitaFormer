# Stage 36 — frozen M2 on full DeepOrganoid (written 2026-11-09, before any M2 DeepOrganoid output exists)
Reason: MODEL_FREEZE_M2_20261108.md §7.6/§8.2 — all DeepOrganoid results so far used M0 ("simplified") or Full and must be re-run with
the frozen main model M2. Author instruction 2026-11-09: run DeepOrganoid first. No training; CTG is never used to change M2.

## Inference (identical to Stage 31 except the model)
`scripts/run_m2_deeporganoid_p1p99_20261109.py`: same 3,456 IXM_BestFocus images (`train_resolved.csv`), per-image p1–p99 stretch,
`legacy.infer` (224/112, reflect pad, ImageNet norm, fp16 autocast, softmax average), argmax, `legacy.feats` (same 21 features).
Models: M2 seed 42 (**primary**, as Stage 31 used seed 42), seeds 3407 and 2026 (seed robustness). Run sequentially on V100-2 GPU0
(exclusive-process). M0 = Stage 31 `simplified` outputs, reused read-only as the reference arm.

## Analyses (methods unchanged from Stage 31 + Stage 34; same cohorts, bootstrap/permutation counts and seeds)
1. QC gate first (Stage 31 §1): per-line foreground distribution, all-background wells, border-touch, contact sheets; agreement of
   M2 masks with M0 masks (pixel agreement, foreground-area correlation) — descriptive.
2. Primary: nested LOCO ridge, CORE 9 NEG-delta features → relative CTG (Pearson, 1,000 hierarchical bootstrap, 200 permutations);
   area-only comparator; excluding Organoid52.
3. Dose trajectories (92 line × drug): median CTG–morphology ρ, concordance among strong-CTG trajectories.
4. Compound-fingerprint increment (pre-specified claim rule of Stage 31 §4 unchanged; expected not met, as for M0).
5. Stage 31 post-hoc potency (area AUC vs CTG AUC) and Stage 34 core claims (within-drug patient-line ranking with drug bootstrap,
   permutation, leave-one-drug-out, noise ceiling; pooled potency) — now pre-specified for M2.
6. M2 − M0: same tables side by side, paired differences where Stage 34 had paired designs; **descriptive, never used for selection**.
   Seeds: each analysis for seed 42; primary LOCO Pearson and within-drug ranking also per seed (mean ± SD).
## Allowed wording (unchanged from Stage 31): morphology tracks orthogonal CTG viability and reproduces potency profiles. Not allowed:
external segmentation accuracy, external alive/dead accuracy, drug-specific death phenotypes, mechanism discrimination.
Results reported whatever they show. New directory only.
