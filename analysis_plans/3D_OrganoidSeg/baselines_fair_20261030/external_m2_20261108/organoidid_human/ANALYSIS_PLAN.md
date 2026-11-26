# OrganoID (human OriginalData + MouseOrganoids) external evaluation — ANALYSIS_PLAN (2026-11-08, before any M2 result)

Author instruction 2026-11-08: "查一下历史曾经用哪些数据集做了外部泛化，先把M2在这些上面的都跑一遍" → the historical external sets were
Brain, OrgaSegment, OrganoIDNet, OrganoID-human (`expFinal_organoid_id_pdac*`) and OrganoID-mouse (`expFinal_organoid_id_mouse*`).
This plan covers the two OrganoID sets (one file, copied into both directories). Frozen model: M2 (MODEL_FREEZE_M2_20261108.md).
Rules as for the other external sets: source split + source metric, results reported whatever they show, nothing changes M2.

## 1. Source and what was verified (author repo `external_datasets/OrganoID`, Publication scripts)
- Matthews et al., PLoS Comput Biol 2022, doi 10.1371/journal.pcbi.1010584; data osf.io/xmes4 (local `external_datasets/OrganoID_Data`).
- Split = released folders, produced by the author scripts (`Publication/Training/Split.bat`, `Augment.bat`;
  `ReviewerComments/MouseOrganoids/{SplitRetrain,AugmentRetrain,Retrain}.bat`):
  human: train 2,000 (augmented from 52 PDAC originals) / val 14 / test 28 (PDAC, ACC, Colon "C*", Lung);
  mouse: train 500 (augmented from 11) / val 4 / test 4.
- Author metric (`Core/Model.py ComputeIOUs`, used by `Figure1/1c/ComputeTestingIOUS.py` and `ReviewerComments/MouseOrganoids/ComputeIOU.py`):
  image → (16-bit 'I' frames ×1/255) → 'L' → resize to 512×512 → per-image min-max; GT → resize 512×512 → PIL `convert("1")`;
  prediction ≥ 0.5; **per-image binary IoU in the 512×512 space**, reported as mean (SD, numpy population SD) per tissue group.
- Literature values (from the 11-06 audit; ⚠️ to be confirmed by the author in the paper before RESULTS): human test PDAC 0.74 ± 0.08,
  non-PDAC 0.79 ± 0.10 (original OrganoID model); mouse test (the same 4 images) retrained model 0.82 ± 0.09.
  Correction to the 11-06 audit: the "19 mouse images" are test + pre_augmented + val (ComputeTestingIOUS.py) and were used only for the
  original (non-retrained) model; the retrained number is on the local 4 test images → the mouse split IS comparable.

## 2. Why the historical result (IoU 0.13) is not used
Verified 2026-11-08 (no model run): 16 of the 28 human test images (and part of training) are 16-bit PNG (`I;16`); the old loader
(`finetune_organoid_id*.py`, `convert('RGB')`) clips them to a constant image (SD 0.0) → per-image IoU 0.01–0.03 on exactly those
images. The old runs (Full model) are kept unchanged and are superseded for the paper.

## 3. Protocol B (primary)
Script `scripts/train_organoidid_m2_20261108.py` = `train_external_m2_20261108.py` with only the data layer changed:
author decoding (above) for every image, replicated to 3 channels; train masks = PIL `convert("1")`; val/test images and masks resized
to 512×512 exactly as the author evaluation, M2 predicts in that space (sliding window 224/112, threshold 0.5, no TTA).
Everything else = Brain/OrganoIDNet M2 arm: M2 seed-42 `final.pth` init, 2-class head, flips + 224 crops, class-weighted CE, AdamW
(2e-6 / 2e-5), cosine, 40k steps, batch 8, val every 4k, checkpoint = best val image Dice, OMP/MKL = 4, one run (seed 42).
Both datasets run concurrently on V100-2 GPU0 (GPU1 occupied by another user).
nnU-Net (same split, 512 space) and protocol A zero-shot: later, same as the other sets.

## 4. Endpoints and statistics
- Primary: per-image binary IoU (512 space). Human: mean ± SD per group (PDAC n≈10; ACC, Colon, Lung) and PDAC vs non-PDAC;
  mouse: mean ± SD over 4 images. Secondary: Dice, precision, recall.
- Evaluator sanity check before scoring: author-style GT IoU(GT, GT) = 1 on all test images; our evaluator recomputes M2's own
  `result.json` per-image IoU.
- Comparisons: literature values descriptive only (no author per-image predictions on disk; the author TFLite model is in the repo
  — running it would give a paired comparison and needs TensorFlow; optional, to be proposed separately). Primary paired contrast
  later = M2 − nnU-Net (human: paired bootstrap over 28 images stratified by group; mouse n = 4 → descriptive only).

## 5. Outputs
New directories only: `external_m2_20261108/organoidid_{human,mouse}/{m2_B,...}`, logs in `external_m2_20261108/logs/`.
No change of decoding, resize, threshold or checkpoint rule after any test result.
