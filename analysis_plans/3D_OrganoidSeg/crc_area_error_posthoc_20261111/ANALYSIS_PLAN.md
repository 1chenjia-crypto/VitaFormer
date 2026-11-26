# CRC test set: area and viability-ratio error (secondary, post hoc) — analysis plan

**Written:** 2026-11-11 09:58 CST, before any number in this analysis was computed.
**Status:** secondary, **post hoc**. Added after the functional (DeepOrganoid / NViR / Gemcitabine) results were known. It is
labelled as such in the manuscript, like the other post hoc analyses.
**No training, no inference.** All inputs are saved per-image confusion counts (tp/fp/fn per class) on the 167 CRC test images.

**Already seen before this plan:**
- pooled per-class IoU, recall and precision of every model (`crc_all_baselines_4class_20261109/`);
- the 2026-10-30 area-error analysis of VitaFormer *ablation variants*. That analysis did not include M2, nnU-Net, SegFormer-B2 or OrganoidViT.

Nothing below has been computed for these models.

## Why the draft numbers are not used
The July draft (alive / dead / bubble area error 15.43 / 17.48 / 2.03%) has three problems:
- it used the early VitaFormer (mIoU 82.75, different architecture and checkpoint rule);
- it was a single seed;
- it compared only against U-Net++, DeepLabv3 and OrganoidViT.

## Inputs
| Model | Seeds | Source (per-image `confusion_counts`) |
|---|---|---|
| VitaFormer (M2, frozen, final.pth) | 42, 3407, 2026 | `vco/runs/arch_exploration_20261104/stageF_m2_test/shared_eval/M2_final_seed*.json` |
| nnU-Net v2 (no TTA, primary config) | 42, 3407, 2026 | `baselines_fair_20261030/shared_eval/nnunet_run2_seed*_noTTA.json` |
| SegFormer-B2 (original runs, as in the main text) | 42, 3407, 2026 | `baselines_fair_20261030/shared_eval/segformer_b2_seed*.json` |
| OrganoidViT reproduction (reference) | 42 only | `OrganoidSegBaseline/runs/orgvit_b0_1446_40k_cond/test_best_iter_32000_relative_area_errors.csv` (per-image pred and GT areas) |

- Predicted area of class c = tp_c + fp_c.
- GT area = tp_c + fn_c, computed over valid GT pixels, which is the shared-evaluator convention.
- Stem order must be identical across all JSONs, or the analysis stops.

## Pre-declared consistency checks
1. GT areas must be identical across all 9 JSON runs, or the analysis stops.
2. The pooled mIoU recomputed from the counts must match each JSON's `mIoU` to < 0.01 pp.
3. **OrganoidViT:** its CSV comes from a different evaluator (mmseg).
   - Its GT areas are compared image by image with the shared GT areas.
   - If they are identical for every image and class, OrganoidViT is shown alongside the other models.
   - Otherwise it is reported descriptively, with the size of the discrepancy, as "different evaluator".
   - OrganoidViT is single-seed. It enters **no** bootstrap contrast; it is a descriptive reference only.

## Endpoints (all reported, whatever the result)
Per image i, class c ∈ {Alive, Dead, Bubble}, computed only on images whose GT contains class c (expected n: Alive 167, Dead ≈ 24, Bubble ≈ 22):
- **E1, absolute relative error:** |P − G| / G.
- **E2, signed relative error:** (P − G) / G. Positive = over-estimation.
- **E3, absolute and signed pixel error:** |P − G| and P − G. For Dead this is the primary way to read the result, because relative error is unstable when G is small.
- **E4, pooled relative error per class:** (ΣP − ΣG) / ΣG over the eligible images, signed. This is the aggregate error, robust to small denominators.

**Viability ratio:** V = Alive / (Alive + Dead).
- **E5:** ΔV = V_pred − V_GT, in percentage points, signed and absolute.
- **Primary set:** images with GT Alive + Dead > 0.
- **Secondary set:** images with GT Dead > 0, where V_GT < 1 is informative.
- In images without GT Dead, V_GT = 1, so ΔV ≤ 0 measures false Dead.
- If the prediction has Alive + Dead = 0, V_pred is undefined. Those images are counted and excluded; no imputation.
- **Pooled viability:** ΣA / (ΣA + ΣD), predicted minus GT, on each set.

## Summaries
- **Per model × seed:** n, median, Q1, Q3 for E1, E2, E3 and E5; mean is also reported for E5 signed.
- **Across seeds:** mean of the seed medians and the range of the seed medians. This is descriptive; no variance test is made with 3 seeds.

## Statistics (model contrasts)
- **Contrasts (A − B):** VitaFormer − nnU-Net, VitaFormer − SegFormer-B2, SegFormer-B2 − nnU-Net.
- **Contrast endpoints:**
  - median E1 per class;
  - median E2 per class;
  - E4 per class;
  - median |ΔV| and mean signed ΔV on the primary set;
  - the same viability endpoints on the Dead-present set.
- **Bootstrap:** the hierarchical paired bootstrap used for mIoU (unified comparison, 2026-11-02).
  - 10,000 replicates, RNG seed 20261111.
  - Each replicate resamples 3 seed positions with replacement, shared by both arms.
  - For each seed position it resamples the physical wells with replacement. Wells come from `build_dataset_index`; the same draw is used for both arms.
  - The statistic is computed on the pooled resampled images.
  - Δ = statistic(A) − statistic(B).
- **Reported per contrast:** observed Δ (statistic on all images × 3 seeds), 95% percentile CI, per-seed Δ, number of seeds favouring each side.
- **Multiplicity:** no correction. Secondary and post hoc; intervals are descriptive.
- **Pairing caveat:** "seed 42" of two models is not a shared random state. Seed pairing is the established convention; image and well pairing is real.

## Interpretation rules (fixed now)
- **Dead:**
  - conclusions about Dead are based on E3 (pixels) and E4 (pooled), not on E1 alone;
  - n ≈ 24 images, so every Dead statement is marked low-precision;
  - Dead area is CRC in-domain only. It says nothing about external validation of the Dead class, which remains unvalidated.
- **Bias direction:** a direction is reported as systematic only if all 3 seeds share its sign and the bootstrap CI of the median E2 (or mean ΔV) excludes 0.
- **No claim of equivalence:** no equivalence margin was pre-specified, so overlapping CIs are written as "no clear difference".
- **No selection:** this analysis may not be used to choose a model, threshold or post-processing step.

## Outputs (new directory, nothing overwritten)
`3D_OrganoidSeg/crc_area_error_posthoc_20261111/`:
- `per_image_area_errors.csv`, `seed_summary.csv`, `contrasts.csv`, `viability_per_image.csv`;
- `consistency_checks.json`, `run_manifest.json` (input SHA256, script SHA256);
- `run.full.log` (unfiltered);
- `RESULTS.md`;
- figure `FigS_crc_area_error.{pdf,png}` and its source CSV.

Planned figure panels:
- (A) E1 per class, per model; seeds shown separately, with median and IQR;
- (B) E2 signed relative error;
- (C) Dead absolute pixel error;
- (D) ΔV on the primary and Dead-present sets.
