# External generalisation aligned with the OrganoidViT paper (source of the CRC main dataset) — ANALYSIS_PLAN (2026-11-09)
Written before any run of this arm. Frozen model M2 (MODEL_FREEZE_M2_20261108.md); nothing here changes M2.
Author instruction 2026-11-09: "我的主数据集来自这个文献中，参考这个文献中的方法，把这几个外部数据集泛化的方法学东西一下，晚上跑一下"
(reference: `vco/2038903037740847104-mark.pdf`).

## 1. What the OrganoidViT paper did (Liu et al., Bio-Des Manuf 2026, doi 10.1631/bdm.2500438)
- Results text + Fig. 6 + Table 1: OrganoidViT was **trained on** three public datasets, viable organoids only (binary):
  OrgaSegment human intestinal, OrganoID human PDAC, OrganoID mouse small intestinal; "mean IoU" 91.82 / 93.20 / 96.18.
  Table 1 compares these with the source papers' numbers (OrgaSegment 0.76, OrganoID 0.74, 0.70).
- Methods (training): ImageNet-pretrained ViT, input 512×512, batch 4, Adam, initial lr 5e-6, 40,000 steps; augmentation = random
  resizing within a range of scales, horizontal flip, random rotation. CRC split 60/20/20. Public-set split: not stated.
- Not stated: public-set split, scale range, rotation range, lr schedule, init for the public sets, exact "mean IoU" definition.
- **Metric note (verified from the paper's own numbers):** its CRC "mean IoU 0.745" is a class-averaged IoU; the public-set values
  (0.92–0.96) are only reachable as the mean over **background + organoid** classes. The source-paper values in its Table 1 are
  different metrics (OrgaSegment 0.76 = instance AP@0.5, OrganoID 0.74 / 0.70 = foreground IoU, 0.70 = original non-retrained
  OrganoID model on 19 mouse images). Table 1 is therefore not like-for-like; we report both definitions and never mix them.

## 2. Two evaluation layers
**Layer 1 — OrganoidViT metric on existing protocol-B predictions (no training).** For OrgaSegment, OrganoID-PDAC (the 10 PDAC test
images; all 28 also shown) and OrganoID-mouse, compute: (a) "mIoU(bg,fg)" = mean of background IoU and organoid IoU from the
dataset-pooled confusion matrix (mmseg convention, OrganoidViT codebase family); (b) the same per image, averaged; (c) foreground IoU
(source-paper metric, already reported). Evaluation space = that of the source paper (OrgaSegment native; OrganoID 512×512).
Compared descriptively with OrganoidViT Table 1 (91.82 / 93.20 / 96.18) — no predictions of OrganoidViT exist, no test.

**Layer 2 — "protocol B-OViT": M2 fine-tuned with the OrganoidViT recipe (tonight).** Script `scripts/train_ovitrecipe_m2_20261109.py`
(= protocol-B script, only the recipe changed). Same source splits, decoding, evaluation space, loss (class-weighted CE), 40k steps,
val every 4k, checkpoint = best val image Dice, threshold 0.5, no TTA, init = M2 seed42 `final.pth` (as protocol B; the paper's
public-set init is unstated — CRC init keeps this an evaluation of the frozen M2).
Recipe from the paper: batch 4; Adam (no weight decay), lr 5e-6 for all parameters; random resize + horizontal flip + random rotation
(no vertical flip). Assumptions for unstated details (fixed now): scale U(0.5, 2.0); rotation p = 0.5, U(−20°, +20°), padded corners
ignored in the loss; poly decay power 0.9 to 0. Kept from M2 (architecture constraint, not a recipe choice): 224 crops and 224/112
sliding window, because the BiomedCLIP ViT-B/16 position embeddings are fixed at 224 — the paper's 512 input cannot be used without
changing the model.
Datasets: organoidid_mouse, organoidid_human, orgasegment, sequentially on V100-2 GPU0 (exclusive-process; GPU1 used by another user).
OMP/MKL = 4. One run each (seed 42).

## 3. Endpoints and reporting
- Per dataset: foreground IoU (source metric; primary for comparison with the source paper), mIoU(bg,fg) (for comparison with
  OrganoidViT Table 1), Dice; OrgaSegment additionally AP@0.5 with the author code + CC rule of its plan.
- B-OViT vs protocol B on the same test images: paired per-image difference, bootstrap (10,000, RNG 20261108); reported as a
  recipe-sensitivity analysis. **Protocol B remains the primary result**; we do not pick the better arm per dataset.
- Results reported whatever they show; RESULTS in `ovit_recipe/`.
