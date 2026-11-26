# Stage 34 — closing the gaps in the four core claims (plan, 2026-10-30, written before any Stage 34 result)

Requested by the author: freeze AutoScale v1 (done: `vco/runs/autoscale_frozen_rule_20261030/FROZEN_RESULT.md`), then verify the four core claims before any figure work.
No training, no inference. Inputs are existing Stage 31/32 outputs and the saved CRC per-image confusion counts. Nothing existing is overwritten; all outputs go to new directories.

## Audit of what already exists (2026-10-30)
| Claim | Exists | Gap |
|---|---|---|
| 1. Fair VitaFormer vs classical area | Stage 32: same images and well mask; threshold parameters fixed a priori; LOCO alignment and potency with CIs and paired differences; Organoid52-excluded LOCO | Within-drug patient-line ranking (the key "unique value" claim) has medians only: no CI, no paired test, no null. Potency for the thresholds excluding Organoid52 is not computed. |
| 2. Morphology beyond area | Stage 32: M2 area + size − area +0.10 [0.002, 0.20] at the pooled level; core − area n.s.; drug fingerprint null (Stage 31) | Not tested **beyond conditions + area** (pooled level is dominated by drug + dose). Not tested on the endpoints where images matter (patient-line ranking, potency). |
| 3. DeepOrganoid statistical robustness | Hierarchical bootstrap, within-line permutation (p = 0.005), LOCO, Organoid52 sensitivity, dose concordance | No CI/null for line ranking; no measurement-noise ceiling (CTG replicate reliability); no leave-one-drug-out stability; no explicit primary vs post-hoc ledger. |
| 4. CRC segmentation frozen | 3 seeds × 6 variants, hierarchical (seed + well) bootstrap with Holm; Rare-Dead P/R for all runs; area error seed 42 only | Area error not computed across seeds / ablations; no single freeze document. |

## A. Patient-line ranking (claims 1 and 3)
Unit: cell line × drug dose–response AUC (Stage 32 definition: mean of per-concentration medians; area relative to the line's NEG median; drugs with ≥ 5 lines).
- Statistic: per drug, Spearman ρ between area AUC and CTG AUC across lines; summary = mean ρ over drugs (the median is also reported, as before).
- Representations: VitaFormer simplified (primary), Full, local-contrast threshold, global Otsu.
- CI: bootstrap over drugs (2,000; seed 20261030), with paired differences simplified − each other representation using the same draws.
- Null: within each drug, permute the line labels of the area AUC (10,000), one-sided p for mean ρ, per representation.
- Noise ceiling: split the 4 replicate wells of each condition into 2 + 2. Average over the 3 distinct pairings, with replicates ordered by (Row, Col). Compute the within-drug line-ranking ρ between CTG half A and CTG half B (CTG self-reliability), and between area half A and CTG half B (simplified).
- Sensitivities: excluding Organoid52; excluding LPA (non-cytotoxic; flagged post hoc in Stage 31); leave-one-drug-out range.
- Same for the pooled potency Spearman (92 pairs): threshold representations excluding Organoid52 (missing from Stage 32).

## B. Morphology beyond area (claim 2), VitaFormer simplified, NEG-delta features
B1 pooled LOCO (Stage 32 code: nested ridge, hierarchical bootstrap, shared draws):
- conditions + area (reference);
- conditions + area + size distribution (M2 features);
- conditions + core 9.
Paired Δ Pearson vs the reference; main cohort and excluding Organoid52.

B2 image-relevant endpoints: from LOCO out-of-fold predictions of image-only models M1 area, M2 area + size, and core 9 (no conditions, so that the line signal comes from the image), compute the AUC per line × drug, then:
- potency Spearman vs CTG AUC (cell-line bootstrap);
- within-drug line-ranking mean ρ (drug bootstrap).
Paired Δ vs M1.

Decision rule: "multi-dimensional morphology adds value beyond area" only if a Δ CI excludes 0 in B1 (beyond conditions + area) or in B2, and in the excluding-Organoid52 cohort too. Otherwise the paper says that the gain is limited to a modest pooled improvement from size distribution.

## C. Robustness ledger (claim 3)
- One table listing every DeepOrganoid endpoint, whether it was pre-specified (Stage 31/32 plans) or post hoc, its estimate, CI, null, Organoid52 sensitivity and leave-one-drug-out range.
- Plate-position check (descriptive): LOCO out-of-fold residual (simplified core) of edge wells (rows A/P, columns 1/24) vs interior wells. Mann–Whitney test and median difference.

## D. CRC area error across seeds and ablations (claim 4) — output `vco/runs/statistical_analysis_20261026/area_error_multiseed_20261030/`
- From the saved per-image confusion counts (pred area = tp + fp; GT area = tp + fn) for all 6 groups × 3 seeds (18 runs).
- Classes: alive, dead, bubble, organoid (alive + dead; organoid TP = TP_alive + TP_dead + alive↔dead confusions is not available from per-class counts, so the organoid area is pred_alive + pred_dead vs gt_alive + gt_dead, which needs only areas).
- Per image: relative error |pred − gt| / gt (gt > 0).
- Seed-level mean; comparator − Full with a paired hierarchical bootstrap (seeds + physical wells, 10,000, seed 20261030), the same resampling as `statistical_analysis_20261026`.
- Consistency check: seed-42 Full and simplified areas must equal the Stage 30 cached-mask areas (`area_error_per_image.csv`) exactly.
- Then `CRC_RESULTS_FREEZE_20261030.md` in the project root: the final segmentation table, ablations (3 seeds), Rare-Dead, area error, baselines with fairness caveat, and the files and hashes behind each number.
