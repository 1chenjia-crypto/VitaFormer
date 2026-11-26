# OrganoIDNetData external evaluation — ANALYSIS_PLAN (draft 2026-11-08, awaiting author approval; written before any M2 result)

Frozen main model: VitaFormer M2 (`3D_OrganoidSeg/model_freeze_m2_20261108/MODEL_FREEZE_M2_20261108.md`).
Author rules: source-paper split and metric; results reported whatever they show; nothing here changes M2.
Comparators: nnU-Net v2 2D (CRC-pretrained) + the source paper's own StarDist and Cellpose predictions (on disk).

## 1. Source and what was verified (from the authors' released data + code, 2026-11-08)
- bioRxiv 10.1101/2024.02.12.580032 / Cell Oncol 2024 (doi 10.1007/s13402-024-00958-2). Local: `external_datasets/OrganoIDNetData/`.
  Incucyte phase contrast, 4×; human 4 cultures + mouse 2 cultures.
- Split on disk (= the authors' `Prepare_Dataset.ipynb` output): **Training 713 patches (800², 8-bit gray, 16-bit instance masks)
  + Augmented 1,426 (2 per training patch, used for training by both source models: `include_augmented = "yes"`) / Validation 179
  patches / Test 10 full FOVs (1040×1408; 5 Human_1 at 4–20 h, 5 Mouse_2 at 0–8 h)**.
  Train/val = random patch-level `train_test_split(test_size=0.2)` (overlapping patches of the same FOVs can fall in both → val is
  optimistic; used only for checkpoint selection). Test FOVs are a separate folder; checked 11-08: the test FOV `Test_Human_1_004h` is
  a different image from the annotated training FOV `Human_1_004h` (mean |Δ| 40.9 grey levels, mask IoU 0.005; same for Mouse_2_000h),
  i.e. same culture and time point, different field. Effective independent units in test = 2 cultures.
- Source metric (author `Compare_StarDist_CellPose.ipynb`, archived): **binary foreground** (`mask > 0`) per test image:
  IoU = |P∩G| / |P∪G|, Dice = 2|P∩G| / (|P|+|G|), pixel accuracy; reported per image (`image_metrics.csv`, archived).
  Means from that CSV: **StarDist IoU 0.670, Dice 0.802; Cellpose IoU 0.701, Dice 0.824** (verified against the paper text examples:
  Human_1_004h 0.736, Human_1_020h 0.727, Mouse_2_008h 0.674). Source models: StarDist / Cellpose, 1000 epochs.
- Author prediction masks for all 10 test FOVs are on disk (`Source_Codes/{StarDist,Cellpose}/AlgorithmPredictions/`) → **paired**
  per-image comparison with the source models is possible.

## 2. Protocols
**B (primary) — train on the source split.** Initialise from M2 seed42 `final.pth` (SHA `0f12be44…`), non-head tensors transferred,
`seg_head` reinitialised to 2 classes (foreground = instance label > 0). Training set = Training + Augmented (as the source);
checkpoint selection on all 179 Validation patches. Recipe identical to the Brain M2 arm: 224 crops, random scale 0.5–2.0 + flips,
class-weighted CE, AdamW (encoder 2e-6 / rest 2e-5), cosine, 40k steps, batch 8, validation every 4k, checkpoint = best val image-level
mean Dice, threshold 0.5, no TTA, native-resolution sliding window 224/112. Input: every image converted to 8-bit grayscale (`L`;
test FOVs are stored RGB) and replicated to 3 channels. OMP/MKL threads = 4. One run (source seed 42).
nnU-Net: native `nnUNetTrainer` (1000 epochs), CRC seed-42 weights except `seg_layers`, `splits_final.json` train = Training+Augmented,
val = Validation; predict the 10 test FOVs, no TTA.

**A (secondary) — zero-shot.** M2 seeds 42/3407/2026, native resolution, foreground = Alive ∪ Dead. Robustness only.

## 3. Endpoints (10 test FOVs)
- **Primary:** image-level binary IoU (mean over the 10 FOVs), computed by a re-implementation of the author functions.
  Sanity check first: our evaluator on the author StarDist/Cellpose masks must reproduce `image_metrics.csv` to 3 decimals for all
  10 × 2 rows; otherwise stop and fix the evaluator before any model is scored.
- Secondary: Dice, pixel accuracy; human vs mouse strata; organoid count error (connected components, 8-connectivity, of the binary
  prediction vs number of GT instances — the paper's count analysis); pooled-pixel IoU labelled as micro.

## 4. Comparisons and statistics
- Primary contrast: **M2 − nnU-Net** (per-image IoU). Secondary: **M2 − Cellpose**, **M2 − StarDist** (author predictions, paired).
- Paired bootstrap stratified by species (5 + 5 FOVs, 10,000, RNG 20261108), 95% CI. Because the 10 FOVs come from 2 cultures,
  CIs describe image-level variation within these two cultures only; no generalisation claim beyond them. CI including 0 =
  no demonstrated difference.

## 5. Outputs and rules
New directory only: `baselines_fair_20261030/external_m2_20261108/organoidnet/{m2_B, nnunet_B, m2_A_seed*}`, per-image CSV,
RESULTS.md with every run, crash and deviation. No change of input conversion, schedule, threshold or checkpoint rule after any test
result.

## Approval and erratum (2026-11-08 ~18:20, before any run on this dataset)
- Author approved this plan 2026-11-08; protocol B M2 run approved to start on free GPUs.
- Erratum to §2 (wording only, found while implementing): the Brain M2 arm (whose recipe this plan adopts verbatim) uses
  **random horizontal + vertical flips and random 224 crops, no random rescaling** for VitaFormer (the 0.5–2.0 rescale in that
  script applies to SegFormer only). The run follows the Brain M2 arm exactly; the "random scale 0.5–2.0" phrase is withdrawn.
- Script: `scripts/train_external_m2_20261108.py` (Brain M2 script with only the data layer changed; dry-run 11-08 passed:
  split counts match the source, 254 tensors transferred, seg_head reinitialised).
