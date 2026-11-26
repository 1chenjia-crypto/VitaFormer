# OrgaSegment external evaluation — ANALYSIS_PLAN (draft 2026-11-08, awaiting author approval; written before any M2 result)

Frozen main model: VitaFormer M2 (`3D_OrganoidSeg/model_freeze_m2_20261108/MODEL_FREEZE_M2_20261108.md`).
Author rules: source-paper split and metric; results reported whatever they show; nothing here changes M2.
Comparators: nnU-Net v2 2D (CRC-pretrained, same transfer rule as Brain). SegFormer-B2 and M0 are not run.

## 1. Source and what was verified
- Lefferts et al., Commun Biol 2024 (doi 10.1038/s42003-024-05966-4); data zenodo 10.5281/zenodo.10278229 (local
  `external_datasets/OrgaSegment/OrganoidBasic_v20211206/`); code github.com/Living-Technologies/OrgaSegment commit `aeaec17`
  (cloned 2026-11-08; config, eval script and `lib/metrics.py` archived in `author_code/`, SHA256 in `frozen_hashes.txt`).
- Split on disk (= source folders `data/20211206/{train,val,eval}` in the author config): **train 184 / val 35 / eval 12** image–mask
  pairs; 1 class (organoid), instance-label PNGs (median 60–72 instances/image, max 172).
  Image sizes: 512² / 1024² / 2048², mostly grayscale, 30 of 231 stored as RGB (two microscopes).
- Source model: Matterport Mask R-CNN, ResNet-101, COCO init, grayscale, 512 crops for training, **native size at evaluation**
  (`IMAGE_RESIZE_MODE='none'`), confidence threshold 0.0.
- Source metric (author `lib/metrics.py`, copied from Cellpose): per image, one-to-one Hungarian matching of instances,
  match if IoU ≥ t; **AP_t = TP / (TP + FP + FN)**; thresholds 0.50:0.05:1.00 (author EvalConfig). No minimum object size.
- Source number used as literature reference: **AP@0.5 = 0.76 ± 0.12** (mean ± SD over the 12 eval images), from the 11-06 audit
  (read from PMC10937908). ⚠️ The paper page could not be re-read on 11-08 (PMC socket error, nature.com login redirect): the author
  should confirm 0.76 ± 0.12 and whether the SD is across images, in a browser, before the RESULTS are written. No result depends on it.

## 2. Protocols
**B (primary) — train on the source split.** Initialise from M2 seed42 `final.pth` (SHA `0f12be44…`), all non-head tensors transferred,
`seg_head` reinitialised to 2 classes. Recipe identical to the Brain M2 arm (`scripts/train_brain_m2_20261108.py`): 224 crops,
random scale 0.5–2.0 + flips, class-weighted CE, AdamW (encoder 2e-6 / rest 2e-5), cosine, 40k steps, batch 8, validation every 4k on
all 35 official val images, checkpoint = best val image-level mean Dice (binary), threshold 0.5, no TTA, native-resolution sliding window
224/112. Input: every image converted to 8-bit grayscale (`L`) and replicated to 3 channels (rule fixed now; source used grayscale).
OMP/MKL threads = 4. One run (source seed 42), as for Brain.
nnU-Net: `nnUNetTrainer` native schedule (1000 epochs), CRC seed-42 weights transferred except `seg_layers`, custom
`splits_final.json` with train = official train, val = official val; prediction on the 12 eval images, no TTA.

**A (secondary) — zero-shot.** M2 seeds 42/3407/2026 CRC models, native resolution only, foreground = Alive ∪ Dead (Bubble = background).
No scale rule (AutoScale v1 is frozen NO-GO). Reported as cross-domain robustness, never compared with the source number.

## 3. From semantic masks to instances (fixed now, same for every model)
Binary prediction → `skimage.measure.label(connectivity=2)` (8-connectivity); no minimum area, no hole filling, no watershed.
Ground truth = the author instance PNG as is. Known structural disadvantage: touching organoids merge into one component
(Mask R-CNN separates them). Reported, not corrected.

## 4. Endpoints (12 eval images; per-image values, then mean ± SD)
- **Primary:** AP@0.5 computed with the archived author `average_precision` (unmodified copy).
- Secondary: mean AP over 0.50:0.05:0.95; AP@0.75; image-level binary Dice and IoU; instance-count error
  (|pred − GT| / GT); merged-object rate (GT instances whose matched component covers ≥ 2 GT instances at IoU-overlap > 0.5).
- Stratified (descriptive): by image size (512 / 1024 / 2048).

## 5. Comparisons and statistics
- Primary contrast: **M2 − nnU-Net**, per-image AP@0.5, paired bootstrap over the 12 eval images (10,000, RNG 20261108), 95% CI.
  n = 12 → low power; "CI includes 0" is read as "no demonstrated difference", never equivalence.
- Literature row: source Mask R-CNN 0.76 ± 0.12 (descriptive; no paired test, no author predictions available).
- Sanity check before reporting: the author `average_precision` run on GT vs GT must return 1.0 for all thresholds on all 12 images.

## 6. Outputs and rules
New directory only: `baselines_fair_20261030/external_m2_20261108/orgasegment/{m2_B, nnunet_B, m2_A_seed*}`, per-image CSV, RESULTS.md
with every run, crash and deviation. No change of crop, schedule, threshold, instance rule or checkpoint rule after any eval result.

## Approval and erratum (2026-11-08 ~18:20, before any run on this dataset)
- Author approved this plan 2026-11-08; protocol B M2 run approved to start on free GPUs.
- Erratum to §2 (wording only, found while implementing): the Brain M2 arm (whose recipe this plan adopts verbatim) uses
  **random horizontal + vertical flips and random 224 crops, no random rescaling** for VitaFormer (the 0.5–2.0 rescale in that
  script applies to SegFormer only). The run follows the Brain M2 arm exactly; the "random scale 0.5–2.0" phrase is withdrawn.
- Script: `scripts/train_external_m2_20261108.py` (Brain M2 script with only the data layer changed; dry-run 11-08 passed:
  split counts match the source, 254 tensors transferred, seg_head reinitialised).
